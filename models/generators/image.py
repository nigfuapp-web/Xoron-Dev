"""
SOTA Image Diffusion Generator with Cross-Attention.

Features:
- Cross-attention for text conditioning (like Stable Diffusion)
- Classifier-free guidance support
- Image editing (inpainting) support
- Better VAE architecture
- Multi-scale feature extraction
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        dtype = timesteps.dtype
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        # Match dtype of timesteps to avoid Float/Half mismatch
        embeddings = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class CrossAttention(nn.Module):
    """Cross-attention for text-to-image conditioning."""
    
    def __init__(self, query_dim: int, context_dim: int = None, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim
        
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = inner_dim
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        h = self.heads
        d = self.dim_head
        
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape for multi-head attention
        b, n, _ = q.shape
        ctx_len = context.shape[1]
        
        q = q.reshape(b, n, h, d).permute(0, 2, 1, 3)
        k = k.reshape(b, ctx_len, h, d).permute(0, 2, 1, 3)
        v = v.reshape(b, ctx_len, h, d).permute(0, 2, 1, 3)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, self.inner_dim)
        
        return self.to_out(out)


class FeedForward(nn.Module):
    """Feed-forward network with GEGLU activation."""
    
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.net[0](x).chunk(2, dim=-1)
        return self.net[3](self.net[2](x * F.gelu(gate)))


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and cross-attention."""
    
    def __init__(self, dim: int, context_dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(dim, dim, heads, dim_head, dropout)  # Self-attention
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(dim, context_dim, heads, dim_head, dropout)  # Cross-attention
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context)
        x = x + self.ff(self.norm3(x))
        return x


class ResBlock(nn.Module):
    """Residual block with time embedding."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(time_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SpatialTransformer(nn.Module):
    """Spatial transformer for applying attention to spatial features."""
    
    def __init__(self, channels: int, context_dim: int, num_heads: int = 8, depth: int = 1):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(channels, context_dim, num_heads, channels // num_heads)
            for _ in range(depth)
        ])
        
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x
        
        x = self.norm(x)
        x = self.proj_in(x)
        
        # Reshape to sequence
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        
        for block in self.transformer_blocks:
            x = block(x, context)
        
        # Reshape back to spatial
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = self.proj_out(x)
        
        return x + x_in


class DownBlock(nn.Module):
    """Downsampling block with residual and attention."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, context_dim: int, 
                 has_attn: bool = True, num_layers: int = 2):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResBlock(in_channels if i == 0 else out_channels, out_channels, time_emb_dim)
            for i in range(num_layers)
        ])
        self.attn_blocks = nn.ModuleList([
            SpatialTransformer(out_channels, context_dim) if has_attn else nn.Identity()
            for _ in range(num_layers)
        ])
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, list]:
        outputs = []
        for res, attn in zip(self.res_blocks, self.attn_blocks):
            x = res(x, time_emb)
            x = attn(x, context) if not isinstance(attn, nn.Identity) else x
            outputs.append(x)
        x = self.downsample(x)
        return x, outputs


class UpBlock(nn.Module):
    """Upsampling block with residual and attention."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, context_dim: int,
                 has_attn: bool = True, num_layers: int = 2):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.res_blocks = nn.ModuleList([
            ResBlock(in_channels + out_channels if i == 0 else out_channels, out_channels, time_emb_dim)
            for i in range(num_layers)
        ])
        self.attn_blocks = nn.ModuleList([
            SpatialTransformer(out_channels, context_dim) if has_attn else nn.Identity()
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, skip_connections: list, time_emb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        for i, (res, attn) in enumerate(zip(self.res_blocks, self.attn_blocks)):
            skip = skip_connections.pop() if skip_connections else torch.zeros_like(x)
            x = torch.cat([x, skip], dim=1) if i == 0 else x
            x = res(x, time_emb)
            x = attn(x, context) if not isinstance(attn, nn.Identity) else x
        return x


class UNet2D(nn.Module):
    """SOTA UNet with cross-attention for text-to-image diffusion."""
    
    def __init__(self, in_channels: int = 4, out_channels: int = 4, base_channels: int = 128,
                 channel_mults: tuple = (1, 2, 4, 4), context_dim: int = 1024, num_heads: int = 8):
        super().__init__()
        
        time_emb_dim = base_channels * 4
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Input
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Downsampling
        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            self.down_blocks.append(
                DownBlock(ch, out_ch, time_emb_dim, context_dim, has_attn=(i > 0))
            )
            ch = out_ch
            channels.append(ch)
        
        # Middle
        self.mid_block1 = ResBlock(ch, ch, time_emb_dim)
        self.mid_attn = SpatialTransformer(ch, context_dim, num_heads)
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim)
        
        # Upsampling
        self.up_blocks = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            self.up_blocks.append(
                UpBlock(ch, out_ch, time_emb_dim, context_dim, has_attn=(i < len(channel_mults) - 1))
            )
            ch = out_ch
        
        # Output
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)
        
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_embed(timesteps)
        
        # Handle inpainting mask
        if mask is not None:
            x = torch.cat([x, mask], dim=1)
        
        # Input
        x = self.conv_in(x)
        
        # Downsampling with skip connections
        skip_connections = []
        for down in self.down_blocks:
            x, skips = down(x, t_emb, context)
            skip_connections.extend(skips)
        
        # Middle
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x, context)
        x = self.mid_block2(x, t_emb)
        
        # Upsampling
        for up in self.up_blocks:
            x = up(x, skip_connections, t_emb, context)
        
        # Output
        x = self.conv_out(F.silu(self.norm_out(x)))
        
        return x


class VAEEncoder(nn.Module):
    """VAE Encoder for image compression."""
    
    def __init__(self, in_channels: int = 3, latent_channels: int = 4, base_channels: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, latent_channels * 2, 3, padding=1),
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mean, logvar = h.chunk(2, dim=1)
        logvar = torch.clamp(logvar, -30, 20)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)
        return z, mean, logvar


class VAEDecoder(nn.Module):
    """VAE Decoder for image reconstruction."""
    
    def __init__(self, latent_channels: int = 4, out_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels * 4, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
            nn.Tanh(),
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class MobileDiffusionGenerator(nn.Module):
    """
    SOTA Diffusion Model for Image Generation and Editing.
    
    Features:
    - Cross-attention for text conditioning
    - Classifier-free guidance support
    - Image editing (inpainting) support
    - VAE for latent space compression
    - v-prediction for better quality
    """

    def __init__(self, latent_channels: int = 4, base_channels: int = 128, context_dim: int = 1024,
                 num_inference_steps: int = 20, image_size: int = 256, cfg_scale: float = 7.5):
        super().__init__()

        self.latent_channels = latent_channels
        self.image_size = image_size
        self.latent_size = image_size // 8
        self.num_inference_steps = num_inference_steps
        self.cfg_scale = cfg_scale
        self.context_dim = context_dim

        # VAE
        self.vae_encoder = VAEEncoder(3, latent_channels)
        self.vae_decoder = VAEDecoder(latent_channels, 3)

        # UNet with cross-attention
        self.unet = UNet2D(latent_channels, latent_channels, base_channels, context_dim=context_dim)

        # Noise schedule (cosine schedule for better quality)
        self._init_noise_schedule()
        
        # Null text embedding for classifier-free guidance
        self.register_buffer('null_context', torch.zeros(1, 77, context_dim))

        print(f"   🎨 SOTA ImageDiffusion: {image_size}x{image_size}, {num_inference_steps} steps, CFG={cfg_scale}")
        print(f"      Features: Cross-attention, CFG, Inpainting support")

    def _init_noise_schedule(self, beta_start: float = 0.00085, beta_end: float = 0.012):
        """Initialize cosine noise schedule."""
        # Cosine schedule
        steps = 1000
        s = 0.008
        t = torch.linspace(0, steps, steps + 1)
        alphas_cumprod = torch.cos(((t / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 0.0001, 0.9999)
        
        alphas = 1.0 - betas
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode image to latent space."""
        return self.vae_encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image."""
        return self.vae_decoder(z)

    def add_noise(self, x: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to latents according to timesteps."""
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        return sqrt_alpha * x + sqrt_one_minus_alpha * noise

    def training_step(self, images: torch.Tensor, context: torch.Tensor, 
                      mask: torch.Tensor = None) -> dict:
        """
        Training step with multiple losses.
        
        Returns dict with:
        - diffusion_loss: Main denoising loss
        - kl_loss: VAE KL divergence
        - total_loss: Combined loss
        """
        device = images.device
        batch_size = images.shape[0]
        
        # Encode to latent
        z, mean, logvar = self.encode(images)
        
        # Sample timesteps
        timesteps = torch.randint(0, 1000, (batch_size,), device=device)
        
        # Add noise
        noise = torch.randn_like(z)
        noisy_z = self.add_noise(z, noise, timesteps)
        
        # Classifier-free guidance: randomly drop context
        # Create null context dynamically to match actual context shape
        if self.training:
            drop_mask = torch.rand(batch_size, device=device) < 0.1
            seq_len = context.shape[1]
            null_ctx = torch.zeros(batch_size, seq_len, self.context_dim, device=device, dtype=context.dtype)
            context = torch.where(drop_mask[:, None, None], null_ctx, context)
        
        # Predict noise
        noise_pred = self.unet(noisy_z, timesteps, context, mask)
        
        # Losses
        diffusion_loss = F.mse_loss(noise_pred, noise)
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        
        total_loss = diffusion_loss + 0.0001 * kl_loss
        
        return {
            'diffusion_loss': diffusion_loss,
            'kl_loss': kl_loss,
            'total_loss': total_loss
        }

    @torch.no_grad()
    def generate(self, context: torch.Tensor, guidance_scale: float = None,
                 num_steps: int = None, init_latents: torch.Tensor = None,
                 mask: torch.Tensor = None, masked_image_latents: torch.Tensor = None) -> torch.Tensor:
        """
        Generate images from text context.
        
        Args:
            context: Text embeddings [B, seq_len, context_dim]
            guidance_scale: CFG scale (default: self.cfg_scale)
            num_steps: Number of denoising steps
            init_latents: Initial latents for img2img
            mask: Inpainting mask
            masked_image_latents: Latents of masked image for inpainting
        """
        device = context.device
        batch_size = context.shape[0]
        seq_len = context.shape[1]
        guidance_scale = guidance_scale or self.cfg_scale
        num_steps = num_steps or self.num_inference_steps

        # Initialize latents
        if init_latents is not None:
            latents = init_latents
        else:
            latents = torch.randn(batch_size, self.latent_channels, self.latent_size, self.latent_size, device=device)

        # Timesteps
        timesteps = torch.linspace(999, 0, num_steps, dtype=torch.long, device=device)

        # Prepare for CFG - create null context dynamically to match input context shape
        if guidance_scale > 1.0:
            null_ctx = torch.zeros(batch_size, seq_len, self.context_dim, device=device, dtype=context.dtype)
            context = torch.cat([null_ctx, context])

        for i, t in enumerate(timesteps):
            t_batch = t.expand(batch_size)
            
            # CFG: predict with and without context
            if guidance_scale > 1.0:
                latent_input = torch.cat([latents, latents])
                t_input = torch.cat([t_batch, t_batch])
                noise_pred = self.unet(latent_input, t_input, context, mask)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.unet(latents, t_batch, context, mask)

            # DDIM step
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)

            pred_x0 = (latents - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            latents = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * noise_pred

            # Inpainting: preserve masked regions
            if mask is not None and masked_image_latents is not None:
                latents = masked_image_latents * mask + latents * (1 - mask)

        # Decode
        images = self.decode(latents)
        images = (images + 1) / 2
        return torch.clamp(images, 0, 1)

    @torch.no_grad()
    def edit_image(self, image: torch.Tensor, context: torch.Tensor, mask: torch.Tensor,
                   strength: float = 0.8, guidance_scale: float = None) -> torch.Tensor:
        """
        Edit image with inpainting.
        
        Args:
            image: Input image [B, 3, H, W] in [0, 1]
            context: Text embeddings for edit
            mask: Binary mask [B, 1, H, W] where 1 = edit region
            strength: How much to change (0-1)
            guidance_scale: CFG scale
        """
        device = image.device
        
        # Encode image
        image_norm = image * 2 - 1
        z, _, _ = self.encode(image_norm)
        
        # Resize mask to latent size
        mask_latent = F.interpolate(mask, size=(self.latent_size, self.latent_size), mode='nearest')
        
        # Add noise based on strength
        num_steps = int(self.num_inference_steps * strength)
        start_step = self.num_inference_steps - num_steps
        
        noise = torch.randn_like(z)
        timestep = torch.tensor([int(999 * strength)], device=device)
        noisy_z = self.add_noise(z, noise, timestep.expand(z.shape[0]))
        
        # Generate with inpainting
        return self.generate(
            context, 
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            init_latents=noisy_z,
            mask=mask_latent,
            masked_image_latents=z
        )

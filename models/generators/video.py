"""
SOTA Video Diffusion Generator with Temporal Attention.

Features:
- AnimateDiff-style motion modules
- Cross-attention for text conditioning
- Image-to-video (I2V) support
- Text-to-video (T2V) support
- Temporal consistency via motion modules
- Classifier-free guidance
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


class TemporalAttention(nn.Module):
    """
    Temporal attention module (AnimateDiff-style).
    Applies attention across the time dimension for temporal consistency.
    """
    def __init__(self, channels: int, num_heads: int = 8, num_frames: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm = nn.GroupNorm(32, channels)
        self.to_qkv = nn.Linear(channels, channels * 3)
        self.to_out = nn.Linear(channels, channels)
        
        # Learnable temporal position embeddings
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, num_frames, channels) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T, H, W]"""
        b, c, t, h, w = x.shape
        
        # Reshape: [B, C, T, H, W] -> [B*H*W, T, C]
        x_reshaped = x.permute(0, 3, 4, 2, 1).reshape(b * h * w, t, c)
        
        # Add temporal position embeddings
        x_reshaped = x_reshaped + self.temporal_pos_embed[:, :t, :]
        
        # Normalize
        x_norm = self.norm(x_reshaped.transpose(1, 2)).transpose(1, 2)
        
        # QKV projection
        qkv = self.to_qkv(x_norm).reshape(b * h * w, t, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        # Attention
        q = q.transpose(1, 2)  # [B*H*W, heads, T, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b * h * w, t, c)
        out = self.to_out(out)
        
        # Reshape back: [B*H*W, T, C] -> [B, C, T, H, W]
        out = out.reshape(b, h, w, t, c).permute(0, 4, 3, 1, 2)
        
        return x + out


class MotionModule(nn.Module):
    """
    Motion module for temporal modeling (AnimateDiff-style).
    Combines temporal attention with temporal convolutions.
    """
    def __init__(self, channels: int, num_heads: int = 8, num_frames: int = 16):
        super().__init__()
        
        # Temporal attention
        self.temporal_attn = TemporalAttention(channels, num_heads, num_frames)
        
        # Temporal convolution for local motion
        self.temporal_conv = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
        )
        
        # Zero-initialized output projection for stable training
        self.out_proj = nn.Conv3d(channels, channels, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T, H, W]"""
        # Temporal attention
        h = self.temporal_attn(x)
        
        # Temporal convolution
        h = h + self.temporal_conv(h)
        
        # Zero-initialized output
        return x + self.out_proj(h)


class CrossAttention3D(nn.Module):
    """Cross-attention for text-to-video conditioning."""
    
    def __init__(self, query_dim: int, context_dim: int = None, heads: int = 8, dim_head: int = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim
        
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = inner_dim
        self.scale = dim_head ** -0.5
        
        self.norm = nn.GroupNorm(32, query_dim)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T, H, W]
        context: [B, seq_len, context_dim]
        """
        b, c, t, h, w = x.shape
        seq_len = t * h * w
        ctx_len = context.shape[1]
        
        # Reshape to sequence: [B, C, T, H, W] -> [B, T*H*W, C]
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(b, seq_len, c)
        
        # Normalize
        x_norm = self.norm(x_flat.transpose(1, 2)).transpose(1, 2)
        
        # QKV
        q = self.to_q(x_norm)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape for multi-head attention
        q = q.reshape(b, seq_len, self.heads, self.dim_head).permute(0, 2, 1, 3)
        k = k.reshape(b, ctx_len, self.heads, self.dim_head).permute(0, 2, 1, 3)
        v = v.reshape(b, ctx_len, self.heads, self.dim_head).permute(0, 2, 1, 3)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, self.inner_dim)
        out = self.to_out(out)
        
        # Reshape back: [B, T*H*W, C] -> [B, C, T, H, W]
        out = out.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        
        return x + out


class ResBlock3D(nn.Module):
    """3D Residual block with time embedding."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(time_emb)[:, :, None, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Conv3DBlock(nn.Module):
    """
    SOTA 3D convolution block with motion module and cross-attention.
    """
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, context_dim: int,
                 use_motion: bool = True, use_cross_attn: bool = True, num_frames: int = 16):
        super().__init__()
        
        # Spatial processing
        self.res_block = ResBlock3D(in_ch, out_ch, time_emb_dim)
        
        # Motion module for temporal consistency
        self.motion_module = MotionModule(out_ch, num_frames=num_frames) if use_motion else nn.Identity()
        
        # Cross-attention for text conditioning
        self.cross_attn = CrossAttention3D(out_ch, context_dim) if use_cross_attn else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.res_block(x, time_emb)
        x = self.motion_module(x) if not isinstance(self.motion_module, nn.Identity) else x
        x = self.cross_attn(x, context) if not isinstance(self.cross_attn, nn.Identity) else x
        return x


class UNet3D(nn.Module):
    """
    SOTA 3D UNet for video diffusion with motion modules and cross-attention.
    """
    def __init__(self, in_channels: int = 4, base_channels: int = 128, context_dim: int = 1024,
                 num_frames: int = 16, channel_mults: tuple = (1, 2, 4)):
        super().__init__()
        self.num_frames = num_frames
        time_emb_dim = base_channels * 4

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Image conditioning projection (for I2V)
        self.image_cond_proj = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )

        # Input conv (with image conditioning channel)
        self.conv_in = nn.Conv3d(in_channels * 2, base_channels, kernel_size=3, padding=1)
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            self.down_blocks.append(nn.ModuleList([
                Conv3DBlock(ch, out_ch, time_emb_dim, context_dim, use_motion=True, use_cross_attn=(i > 0), num_frames=num_frames),
                Conv3DBlock(out_ch, out_ch, time_emb_dim, context_dim, use_motion=True, use_cross_attn=(i > 0), num_frames=num_frames),
                nn.Conv3d(out_ch, out_ch, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            ]))
            ch = out_ch

        # Middle
        self.mid_block1 = Conv3DBlock(ch, ch, time_emb_dim, context_dim, use_motion=True, use_cross_attn=True, num_frames=num_frames)
        self.mid_block2 = Conv3DBlock(ch, ch, time_emb_dim, context_dim, use_motion=True, use_cross_attn=True, num_frames=num_frames)

        # Decoder
        self.up_blocks = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            self.up_blocks.append(nn.ModuleList([
                nn.ConvTranspose3d(ch, ch, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                Conv3DBlock(ch + out_ch, out_ch, time_emb_dim, context_dim, use_motion=True, use_cross_attn=(i < len(channel_mults) - 1), num_frames=num_frames),
                Conv3DBlock(out_ch, out_ch, time_emb_dim, context_dim, use_motion=True, use_cross_attn=(i < len(channel_mults) - 1), num_frames=num_frames),
            ]))
            ch = out_ch

        # Output
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv3d(ch, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor,
                first_frame_latent: torch.Tensor = None) -> torch.Tensor:
        """
        x: [B, C, T, H, W] noisy latents
        timesteps: [B] diffusion timesteps
        context: [B, seq_len, context_dim] text embeddings
        first_frame_latent: [B, C, H, W] first frame latent for I2V
        """
        b, c, t, h, w = x.shape
        
        # Time embedding
        t_emb = self.time_embed(timesteps)
        
        # Image conditioning
        if first_frame_latent is not None:
            img_cond = self.image_cond_proj(first_frame_latent)
        else:
            img_cond = torch.zeros(b, c, h, w, device=x.device)
        
        # Expand image conditioning to all frames
        img_cond_expanded = img_cond.unsqueeze(2).expand(-1, -1, t, -1, -1)
        x = torch.cat([x, img_cond_expanded], dim=1)
        
        # Input
        x = self.conv_in(x)
        
        # Encoder with skip connections
        skips = []
        for block1, block2, downsample in self.down_blocks:
            x = block1(x, t_emb, context)
            x = block2(x, t_emb, context)
            skips.append(x)
            x = downsample(x)
        
        # Middle
        x = self.mid_block1(x, t_emb, context)
        x = self.mid_block2(x, t_emb, context)
        
        # Decoder
        for upsample, block1, block2 in self.up_blocks:
            x = upsample(x)
            skip = skips.pop()
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block1(x, t_emb, context)
            x = block2(x, t_emb, context)
        
        # Output
        x = self.conv_out(F.silu(self.norm_out(x)))
        
        return x


class VideoVAEEncoder(nn.Module):
    """3D VAE Encoder for video compression."""
    
    def __init__(self, in_channels: int = 3, latent_channels: int = 4, base_channels: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(base_channels, base_channels, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(base_channels * 4, latent_channels * 2, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mean, logvar = h.chunk(2, dim=1)
        logvar = torch.clamp(logvar, -30, 20)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)
        return z, mean, logvar


class VideoVAEDecoder(nn.Module):
    """3D VAE Decoder for video reconstruction."""
    
    def __init__(self, latent_channels: int = 4, out_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv3d(latent_channels, base_channels * 4, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.ConvTranspose3d(base_channels * 4, base_channels * 4, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(base_channels * 4, base_channels * 2, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.ConvTranspose3d(base_channels * 2, base_channels * 2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(base_channels * 2, base_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.ConvTranspose3d(base_channels, base_channels, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(base_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.Tanh(),
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class ImageEncoder2D(nn.Module):
    """2D Image encoder for I2V first frame."""
    
    def __init__(self, in_channels: int = 3, latent_channels: int = 4, base_channels: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, latent_channels, 3, padding=1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MobileVideoDiffusion(nn.Module):
    """
    SOTA Video Diffusion Model for T2V and I2V.
    
    Features:
    - AnimateDiff-style motion modules for temporal consistency
    - Cross-attention for text conditioning
    - Classifier-free guidance support
    - Image-to-video (I2V) with first frame conditioning
    - Text-to-video (T2V) generation
    - Temporal consistency loss support
    """

    def __init__(self, latent_channels: int = 4, base_channels: int = 128, context_dim: int = 1024,
                 num_frames: int = 16, image_size: int = 256, num_inference_steps: int = 20,
                 cfg_scale: float = 7.5):
        super().__init__()

        self.latent_channels = latent_channels
        self.num_frames = num_frames
        self.image_size = image_size
        self.latent_size = image_size // 8
        self.num_inference_steps = num_inference_steps
        self.cfg_scale = cfg_scale
        self.context_dim = context_dim

        # Video VAE
        self.vae_encoder = VideoVAEEncoder(3, latent_channels)
        self.vae_decoder = VideoVAEDecoder(latent_channels, 3)
        
        # Image encoder for I2V
        self.image_encoder = ImageEncoder2D(3, latent_channels)

        # 3D UNet with motion modules
        self.unet = UNet3D(latent_channels, base_channels, context_dim, num_frames)

        # Noise schedule
        self._init_noise_schedule()
        
        # Null context for CFG
        self.register_buffer('null_context', torch.zeros(1, 77, context_dim))

        print(f"   🎬 SOTA VideoDiffusion: {num_frames} frames @ {image_size}x{image_size}")
        print(f"      Features: Motion modules, Cross-attention, CFG={cfg_scale}")
        print(f"      Supports: Text-to-Video (T2V) + Image-to-Video (I2V)")

    def _init_noise_schedule(self):
        """Initialize cosine noise schedule."""
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

    def encode_video(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode video to latent space. x: [B, C, T, H, W]"""
        if x.dim() == 5 and x.shape[2] == 3:
            x = x.permute(0, 2, 1, 3, 4)
        return self.vae_encoder(x)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Encode single image to latent. x: [B, C, H, W]"""
        return self.image_encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents to video. z: [B, C, T, H, W]"""
        return self.vae_decoder(z)

    def add_noise(self, x: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to latents."""
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
        return sqrt_alpha * x + sqrt_one_minus_alpha * noise

    def training_step(self, video: torch.Tensor, context: torch.Tensor,
                      first_frame: torch.Tensor = None) -> dict:
        """
        Training step with multiple losses.
        
        Args:
            video: [B, C, T, H, W] video tensor
            context: [B, seq_len, context_dim] text embeddings
            first_frame: [B, C, H, W] first frame for I2V training
        """
        device = video.device
        batch_size = video.shape[0]
        
        # Encode video
        z, mean, logvar = self.encode_video(video)
        
        # Encode first frame if provided (I2V mode)
        first_frame_latent = None
        if first_frame is not None:
            first_frame_latent = self.encode_image(first_frame * 2 - 1)
        
        # Sample timesteps
        timesteps = torch.randint(0, 1000, (batch_size,), device=device)
        
        # Add noise
        noise = torch.randn_like(z)
        noisy_z = self.add_noise(z, noise, timesteps)
        
        # CFG: randomly drop context
        # Create null context dynamically to match actual context shape
        if self.training:
            drop_mask = torch.rand(batch_size, device=device) < 0.1
            seq_len = context.shape[1]
            null_ctx = torch.zeros(batch_size, seq_len, self.context_dim, device=device, dtype=context.dtype)
            context = torch.where(drop_mask[:, None, None], null_ctx, context)
        
        # Predict noise
        noise_pred = self.unet(noisy_z, timesteps, context, first_frame_latent)
        
        # Losses
        diffusion_loss = F.mse_loss(noise_pred, noise)
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        
        # Temporal consistency loss (encourage smooth motion)
        temporal_loss = torch.tensor(0.0, device=device)
        if z.shape[2] > 1:
            z_diff = z[:, :, 1:] - z[:, :, :-1]
            temporal_loss = torch.mean(z_diff ** 2)
        
        total_loss = diffusion_loss + 0.0001 * kl_loss + 0.01 * temporal_loss
        
        return {
            'diffusion_loss': diffusion_loss,
            'kl_loss': kl_loss,
            'temporal_loss': temporal_loss,
            'total_loss': total_loss
        }

    @torch.no_grad()
    def generate_t2v(self, context: torch.Tensor, num_frames: int = None,
                     guidance_scale: float = None, num_steps: int = None) -> torch.Tensor:
        """Text-to-Video generation."""
        device = context.device
        batch_size = context.shape[0]
        seq_len = context.shape[1]
        num_frames = num_frames or self.num_frames
        guidance_scale = guidance_scale or self.cfg_scale
        num_steps = num_steps or self.num_inference_steps

        # Initialize latents
        latents = torch.randn(
            batch_size, self.latent_channels, num_frames,
            self.latent_size, self.latent_size, device=device
        )

        timesteps = torch.linspace(999, 0, num_steps, dtype=torch.long, device=device)

        # Prepare for CFG - create null context dynamically to match input context shape
        if guidance_scale > 1.0:
            null_ctx = torch.zeros(batch_size, seq_len, self.context_dim, device=device, dtype=context.dtype)
            context = torch.cat([null_ctx, context])

        for i, t in enumerate(timesteps):
            t_batch = t.expand(batch_size)
            
            # CFG
            if guidance_scale > 1.0:
                latent_input = torch.cat([latents, latents])
                t_input = torch.cat([t_batch, t_batch])
                noise_pred = self.unet(latent_input, t_input, context, None)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.unet(latents, t_batch, context, None)

            # DDIM step
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)

            pred_x0 = (latents - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            latents = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * noise_pred

        video = self.decode(latents)
        return torch.clamp((video + 1) / 2, 0, 1)

    @torch.no_grad()
    def generate_i2v(self, first_frame: torch.Tensor, context: torch.Tensor = None,
                     num_frames: int = None, guidance_scale: float = None,
                     num_steps: int = None) -> torch.Tensor:
        """Image-to-Video generation."""
        device = first_frame.device
        batch_size = first_frame.shape[0]
        num_frames = num_frames or self.num_frames
        guidance_scale = guidance_scale or self.cfg_scale
        num_steps = num_steps or self.num_inference_steps

        # Encode first frame
        first_frame_norm = first_frame * 2 - 1
        first_frame_latent = self.encode_image(first_frame_norm)

        # Initialize latents
        latents = torch.randn(
            batch_size, self.latent_channels, num_frames,
            self.latent_size, self.latent_size, device=device
        )
        
        # Set first frame
        latents[:, :, 0] = first_frame_latent

        timesteps = torch.linspace(999, 0, num_steps, dtype=torch.long, device=device)

        # Use null context if not provided - default to 77 tokens like CLIP
        if context is None:
            context = torch.zeros(batch_size, 77, self.context_dim, device=device)
        
        seq_len = context.shape[1]

        # Prepare for CFG - create null context dynamically to match input context shape
        if guidance_scale > 1.0:
            null_ctx = torch.zeros(batch_size, seq_len, self.context_dim, device=device, dtype=context.dtype)
            context = torch.cat([null_ctx, context])
            first_frame_latent_cfg = torch.cat([first_frame_latent, first_frame_latent])

        for i, t in enumerate(timesteps):
            t_batch = t.expand(batch_size)
            
            # CFG
            if guidance_scale > 1.0:
                latent_input = torch.cat([latents, latents])
                t_input = torch.cat([t_batch, t_batch])
                noise_pred = self.unet(latent_input, t_input, context, first_frame_latent_cfg)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.unet(latents, t_batch, context, first_frame_latent)

            # DDIM step
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)

            pred_x0 = (latents - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            latents = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * noise_pred

            # Preserve first frame
            latents[:, :, 0] = first_frame_latent

        video = self.decode(latents)
        return torch.clamp((video + 1) / 2, 0, 1)

"""
SOTA Video Generator with Flow Matching, 3D-RoPE, Temporal Expert Routing, 3D Causal Transformers.

Features:
- Flow Matching with CFG for superior generation quality
- 3D-RoPE for flexible (x, y, t) positional encodings
- Temporal-Aware Expert Routing in transformer blocks
- 3D Causal Transformers for autoregressive video generation
- FP16-native numerical stability
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

EPS = 1e-5


class RoPE3D(nn.Module):
    """
    3D Rotary Position Embedding for (x, y, t) dimensions.
    Supports flexible aspect ratios and temporal lengths.
    """

    def __init__(self, dim: int, max_height: int = 64, max_width: int = 64, max_frames: int = 32, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        self.max_frames = max_frames
        self.base = base
        
        dim_per_axis = dim // 3
        self.dim_x = dim_per_axis
        self.dim_y = dim_per_axis
        self.dim_t = dim - 2 * dim_per_axis
        
        inv_freq_x = 1.0 / (base ** (torch.arange(0, self.dim_x, 2, dtype=torch.float32) / self.dim_x))
        inv_freq_y = 1.0 / (base ** (torch.arange(0, self.dim_y, 2, dtype=torch.float32) / self.dim_y))
        inv_freq_t = 1.0 / (base ** (torch.arange(0, self.dim_t, 2, dtype=torch.float32) / self.dim_t))
        
        self.register_buffer('inv_freq_x', inv_freq_x, persistent=False)
        self.register_buffer('inv_freq_y', inv_freq_y, persistent=False)
        self.register_buffer('inv_freq_t', inv_freq_t, persistent=False)

    def forward(self, x: torch.Tensor, height: int, width: int, frames: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        dtype = x.dtype
        
        pos_x = torch.arange(width, device=device, dtype=torch.float32)
        pos_y = torch.arange(height, device=device, dtype=torch.float32)
        pos_t = torch.arange(frames, device=device, dtype=torch.float32)
        
        freqs_x = torch.outer(pos_x, self.inv_freq_x.to(device))
        freqs_y = torch.outer(pos_y, self.inv_freq_y.to(device))
        freqs_t = torch.outer(pos_t, self.inv_freq_t.to(device))
        
        freqs_x = torch.cat([freqs_x, freqs_x], dim=-1)
        freqs_y = torch.cat([freqs_y, freqs_y], dim=-1)
        freqs_t = torch.cat([freqs_t, freqs_t], dim=-1)
        
        cos_x = freqs_x.cos().to(dtype)
        sin_x = freqs_x.sin().to(dtype)
        cos_y = freqs_y.cos().to(dtype)
        sin_y = freqs_y.sin().to(dtype)
        cos_t = freqs_t.cos().to(dtype)
        sin_t = freqs_t.sin().to(dtype)
        
        cos_3d = torch.zeros(frames, height, width, self.dim, device=device, dtype=dtype)
        sin_3d = torch.zeros(frames, height, width, self.dim, device=device, dtype=dtype)
        
        for t in range(frames):
            for y in range(height):
                for w in range(width):
                    cos_3d[t, y, w, :self.dim_x] = cos_x[w]
                    sin_3d[t, y, w, :self.dim_x] = sin_x[w]
                    cos_3d[t, y, w, self.dim_x:self.dim_x+self.dim_y] = cos_y[y]
                    sin_3d[t, y, w, self.dim_x:self.dim_x+self.dim_y] = sin_y[y]
                    cos_3d[t, y, w, self.dim_x+self.dim_y:] = cos_t[t]
                    sin_3d[t, y, w, self.dim_x+self.dim_y:] = sin_t[t]
        
        cos_3d = cos_3d.view(frames * height * width, self.dim)
        sin_3d = sin_3d.view(frames * height * width, self.dim)
        
        return cos_3d, sin_3d


def apply_rope_3d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class TemporalExpertRouter(nn.Module):
    """
    Temporal-Aware Expert Router for video generation.
    Routes tokens based on temporal context and motion patterns.
    """

    def __init__(self, hidden_size: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.temporal_proj = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor, temporal_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if temporal_context is not None:
            x = x + self.temporal_proj(temporal_context)
        
        router_logits = self.gate(x)
        router_probs = F.softmax(router_logits, dim=-1, dtype=x.dtype)
        
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + EPS)
        
        return top_k_probs, top_k_indices


class VideoExpert(nn.Module):
    """Single expert for video processing with SwiGLU."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TemporalMoELayer(nn.Module):
    """
    Temporal-Aware MoE Layer for video generation.
    Uses motion-aware routing for expert selection.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.router = TemporalExpertRouter(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList([
            VideoExpert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
        self.shared_expert = VideoExpert(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, temporal_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        
        top_k_probs, top_k_indices = self.router(x_flat, temporal_context.view(-1, hidden_size) if temporal_context is not None else None)
        
        output = torch.zeros_like(x_flat)
        
        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]
            for k in range(self.top_k):
                mask = (top_k_indices[:, k] == expert_idx)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = expert(expert_input)
                    weight = top_k_probs[mask, k:k+1]
                    output[mask] = output[mask] + weight * expert_output
        
        shared_output = self.shared_expert(x_flat)
        output = output + shared_output
        
        return output.view(batch_size, seq_len, hidden_size)


class Causal3DAttention(nn.Module):
    """
    3D Causal Self-Attention with 3D-RoPE.
    Attends only to past frames and positions for autoregressive generation.
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, max_frames: int = 32, max_height: int = 64, max_width: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.rope_3d = RoPE3D(self.head_dim, max_height, max_width, max_frames)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, height: int, width: int, frames: int, causal: bool = True) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        x = self.norm(x)
        
        qkv = self.to_qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        cos, sin = self.rope_3d(x, height, width, frames)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        q = apply_rope_3d(q, cos, sin)
        k = apply_rope_3d(k, cos, sin)
        
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        if causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(causal_mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(x.dtype)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        out = self.to_out(out)
        
        return out


class CrossAttention3D(nn.Module):
    """Cross-attention for text-to-video conditioning."""

    def __init__(self, query_dim: int, context_dim: int = None, heads: int = 8):
        super().__init__()
        self.heads = heads
        context_dim = context_dim or query_dim
        self.head_dim = query_dim // heads
        self.scale = self.head_dim ** -0.5
        
        self.norm = nn.LayerNorm(query_dim)
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim, bias=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        ctx_len = context.shape[1]
        
        x = self.norm(x)
        
        q = self.to_q(x).reshape(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).reshape(batch_size, ctx_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).reshape(batch_size, ctx_len, self.heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(x.dtype)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.to_out(out)
        
        return out


class Causal3DTransformerBlock(nn.Module):
    """
    3D Causal Transformer Block with Temporal MoE.
    """

    def __init__(self, hidden_size: int, context_dim: int, num_heads: int = 8, num_experts: int = 4, max_frames: int = 32):
        super().__init__()
        
        self.self_attn = Causal3DAttention(hidden_size, num_heads, max_frames)
        self.cross_attn = CrossAttention3D(hidden_size, context_dim, num_heads)
        self.moe = TemporalMoELayer(hidden_size, hidden_size * 4, num_experts)
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, context: torch.Tensor, height: int, width: int, frames: int, temporal_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), height, width, frames, causal=True)
        x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.moe(self.norm3(x), temporal_context)
        
        return x


class FlowMatchingScheduler:
    """
    Flow Matching scheduler for video generation.
    Uses optimal transport paths for superior generation quality.
    """

    def __init__(self, num_steps: int = 50, sigma_min: float = 0.002):
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        
        self.timesteps = torch.linspace(1, 0, num_steps + 1)

    def get_velocity(self, x_t: torch.Tensor, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute target velocity for flow matching."""
        return x_0 - x_t

    def step(self, model_output: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """Single step of flow matching ODE."""
        dt = t - t_prev
        x_prev = x_t + model_output * dt.view(-1, 1, 1, 1, 1)
        return x_prev

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Add noise for training (linear interpolation)."""
        noise = torch.randn_like(x_0)
        # Ensure t has same dtype as x_0 for consistent math
        t = t.to(x_0.dtype).view(-1, 1, 1, 1, 1)
        x_t = t * noise + (1 - t) * x_0
        return x_t


class VideoUNet3D(nn.Module):
    """
    3D U-Net for video generation with Causal Transformers and Temporal MoE.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        hidden_size: int = 512,
        context_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
        num_experts: int = 4,
        num_frames: int = 16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_frames = num_frames
        
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        
        self.input_proj = nn.Conv3d(in_channels, hidden_size, kernel_size=3, padding=1)
        
        self.transformer_blocks = nn.ModuleList([
            Causal3DTransformerBlock(hidden_size, context_dim, num_heads, num_experts, num_frames)
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Sequential(
            nn.GroupNorm(32, hidden_size),
            nn.SiLU(),
            nn.Conv3d(hidden_size, out_channels, kernel_size=3, padding=1),
        )
        
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, first_frame_latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, channels, frames, height, width = x.shape
        
        half_dim = self.hidden_size // 2
        t_emb = math.log(10000) / (half_dim - 1)
        t_emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=x.dtype) * -t_emb)
        t_emb = timesteps[:, None].to(x.dtype) * t_emb[None, :]
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
        t_emb = self.time_embed(t_emb)
        
        h = self.input_proj(x)
        
        h = h.permute(0, 2, 3, 4, 1).reshape(batch_size, frames * height * width, self.hidden_size)
        
        temporal_context = t_emb.unsqueeze(1).expand(-1, frames * height * width, -1)
        
        for block in self.transformer_blocks:
            h = block(h, context, height, width, frames, temporal_context)
        
        h = h.reshape(batch_size, frames, height, width, self.hidden_size).permute(0, 4, 1, 2, 3)
        
        velocity = self.output_proj(h)
        
        return velocity


class VideoVAE3D(nn.Module):
    """3D VAE for video encoding/decoding."""

    def __init__(self, in_channels: int = 3, latent_channels: int = 4, base_channels: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(base_channels, base_channels * 2, 3, stride=(1, 2, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(base_channels * 2, base_channels * 4, 3, stride=(1, 2, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(base_channels * 4, latent_channels * 2, 3, padding=1),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv3d(latent_channels, base_channels * 4, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
            nn.Conv3d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
            nn.Conv3d(base_channels * 2, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(base_channels, in_channels, 3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mean, logvar = h.chunk(2, dim=1)
        logvar = torch.clamp(logvar, -30, 20)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)
        return z, mean, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class MobileVideoDiffusion(nn.Module):
    """
    SOTA Video Diffusion with Flow Matching, 3D-RoPE, Temporal MoE.
    Optimized for 2x T4 GPUs with FP16.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        base_channels: int = 64,
        context_dim: int = 1024,
        num_frames: int = 16,
        image_size: int = 384,  # Match SigLIP 384x384
        num_inference_steps: int = 50,
        cfg_scale: float = 7.5,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.context_dim = context_dim
        self.num_frames = num_frames
        self.image_size = image_size
        self.latent_size = image_size // 4
        self.num_inference_steps = num_inference_steps
        self.cfg_scale = cfg_scale
        
        self.vae = VideoVAE3D(3, latent_channels, base_channels)
        
        self.unet = VideoUNet3D(
            in_channels=latent_channels,
            out_channels=latent_channels,
            hidden_size=base_channels * 4,
            context_dim=context_dim,
            num_layers=4,
            num_heads=8,
            num_experts=4,
            num_frames=num_frames,
        )
        
        self.scheduler = FlowMatchingScheduler(num_inference_steps)

    def encode_video(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.vae.encode(video * 2 - 1)

    def decode_video(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image_expanded = image.unsqueeze(2)
        z, _, _ = self.vae.encode(image_expanded)
        return z.squeeze(2)

    def training_step(self, video: torch.Tensor, context: torch.Tensor, first_frame: Optional[torch.Tensor] = None) -> dict:
        device = video.device
        dtype = video.dtype  # Match dtype to input (FP16/BF16/FP32)
        batch_size = video.shape[0]
        
        # Encode video to latent space (reduces memory: 384x384 -> 96x96)
        z, mean, logvar = self.encode_video(video)
        del video  # Free video memory immediately after encoding
        
        # CRITICAL: Use same dtype as input to avoid "mat1 and mat2 must have the same dtype" errors
        t = torch.rand(batch_size, device=device, dtype=dtype)
        
        x_t = self.scheduler.add_noise(z, t)
        
        target_velocity = self.scheduler.get_velocity(x_t, z, t)
        
        # Classifier-free guidance dropout (10% of the time, drop context)
        if self.training:
            drop_mask = torch.rand(batch_size, device=device) < 0.1
            # Expand drop_mask to match context shape [B, seq_len, context_dim]
            # drop_mask: [B] -> [B, 1, 1] for proper broadcasting
            drop_mask_expanded = drop_mask.view(batch_size, 1, 1).expand_as(context)
            null_ctx = torch.zeros_like(context)
            context = torch.where(drop_mask_expanded, null_ctx, context)
            del drop_mask, drop_mask_expanded, null_ctx
        
        pred_velocity = self.unet(x_t, (t * 1000).to(dtype), context, None)
        del x_t, context  # Free memory
        
        flow_loss = F.mse_loss(pred_velocity, target_velocity)
        del pred_velocity, target_velocity  # Free memory
        
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        
        temporal_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if z.shape[2] > 1:
            z_diff = z[:, :, 1:] - z[:, :, :-1]
            temporal_loss = torch.mean(z_diff ** 2)
            del z_diff
        del z, mean, logvar  # Free memory
        
        total_loss = flow_loss + 0.0001 * kl_loss + 0.01 * temporal_loss
        
        return {
            'flow_loss': flow_loss,
            'kl_loss': kl_loss,
            'temporal_loss': temporal_loss,
            'total_loss': total_loss,
        }

    @torch.no_grad()
    def generate_t2v(self, context: torch.Tensor, num_frames: int = None, guidance_scale: float = None, num_steps: int = None) -> torch.Tensor:
        device = context.device
        batch_size = context.shape[0]
        seq_len = context.shape[1]
        num_frames = num_frames or self.num_frames
        guidance_scale = guidance_scale or self.cfg_scale
        num_steps = num_steps or self.num_inference_steps
        
        latents = torch.randn(
            batch_size, self.latent_channels, num_frames,
            self.latent_size, self.latent_size, device=device
        )
        
        timesteps = torch.linspace(1, 0, num_steps + 1, device=device)
        
        if guidance_scale > 1.0:
            null_ctx = torch.zeros(batch_size, seq_len, self.context_dim, device=device, dtype=context.dtype)
            context = torch.cat([null_ctx, context])
        
        for i in range(num_steps):
            t = timesteps[i]
            t_prev = timesteps[i + 1]
            t_batch = t.expand(batch_size) * 1000
            
            if guidance_scale > 1.0:
                latent_input = torch.cat([latents, latents])
                t_input = torch.cat([t_batch, t_batch])
                velocity_pred = self.unet(latent_input, t_input, context, None)
                velocity_uncond, velocity_cond = velocity_pred.chunk(2)
                velocity_pred = velocity_uncond + guidance_scale * (velocity_cond - velocity_uncond)
            else:
                velocity_pred = self.unet(latents, t_batch, context, None)
            
            latents = self.scheduler.step(velocity_pred, t, t_prev, latents)
        
        video = self.decode_video(latents)
        return torch.clamp((video + 1) / 2, 0, 1)

    @torch.no_grad()
    def generate_i2v(self, first_frame: torch.Tensor, context: Optional[torch.Tensor] = None, num_frames: int = None, guidance_scale: float = None, num_steps: int = None) -> torch.Tensor:
        device = first_frame.device
        batch_size = first_frame.shape[0]
        num_frames = num_frames or self.num_frames
        guidance_scale = guidance_scale or self.cfg_scale
        num_steps = num_steps or self.num_inference_steps
        
        first_frame_latent = self.encode_image(first_frame * 2 - 1)
        
        latents = torch.randn(
            batch_size, self.latent_channels, num_frames,
            self.latent_size, self.latent_size, device=device
        )
        latents[:, :, 0] = first_frame_latent
        
        if context is None:
            context = torch.zeros(batch_size, 77, self.context_dim, device=device)
        
        seq_len = context.shape[1]
        timesteps = torch.linspace(1, 0, num_steps + 1, device=device)
        
        if guidance_scale > 1.0:
            null_ctx = torch.zeros(batch_size, seq_len, self.context_dim, device=device, dtype=context.dtype)
            context = torch.cat([null_ctx, context])
        
        for i in range(num_steps):
            t = timesteps[i]
            t_prev = timesteps[i + 1]
            t_batch = t.expand(batch_size) * 1000
            
            if guidance_scale > 1.0:
                latent_input = torch.cat([latents, latents])
                t_input = torch.cat([t_batch, t_batch])
                velocity_pred = self.unet(latent_input, t_input, context, None)
                velocity_uncond, velocity_cond = velocity_pred.chunk(2)
                velocity_pred = velocity_uncond + guidance_scale * (velocity_cond - velocity_uncond)
            else:
                velocity_pred = self.unet(latents, t_batch, context, None)
            
            latents = self.scheduler.step(velocity_pred, t, t_prev, latents)
            latents[:, :, 0] = first_frame_latent
        
        video = self.decode_video(latents)
        return torch.clamp((video + 1) / 2, 0, 1)

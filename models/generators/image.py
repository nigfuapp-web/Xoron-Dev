"""
SOTA Image Generator with MoE-DiT, Flow Matching, 2D-RoPE, Dual-Stream Attention.

Features:
- MoE-DiT (Diffusion Transformer with Mixture of Experts)
- Flow Matching for superior generation quality
- 2D-RoPE for flexible aspect ratios
- Symmetric Dual-Stream Attention (SD3/Flux-style)
- FP16-native numerical stability
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

EPS = 1e-5


class RoPE2D(nn.Module):
    """
    2D Rotary Position Embedding for flexible aspect ratios.
    Encodes (x, y) spatial positions for patch-based DiT.
    """

    def __init__(self, dim: int, max_height: int = 128, max_width: int = 128, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        self.base = base
        
        self.dim_x = dim // 2
        self.dim_y = dim - self.dim_x
        
        inv_freq_x = 1.0 / (base ** (torch.arange(0, self.dim_x, 2, dtype=torch.float32) / self.dim_x))
        inv_freq_y = 1.0 / (base ** (torch.arange(0, self.dim_y, 2, dtype=torch.float32) / self.dim_y))
        
        self.register_buffer('inv_freq_x', inv_freq_x, persistent=False)
        self.register_buffer('inv_freq_y', inv_freq_y, persistent=False)

    def forward(self, x: torch.Tensor, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        dtype = x.dtype
        
        pos_x = torch.arange(width, device=device, dtype=torch.float32)
        pos_y = torch.arange(height, device=device, dtype=torch.float32)
        
        freqs_x = torch.outer(pos_x, self.inv_freq_x.to(device))
        freqs_y = torch.outer(pos_y, self.inv_freq_y.to(device))
        
        freqs_x = torch.cat([freqs_x, freqs_x], dim=-1)
        freqs_y = torch.cat([freqs_y, freqs_y], dim=-1)
        
        cos_2d = torch.zeros(height, width, self.dim, device=device, dtype=dtype)
        sin_2d = torch.zeros(height, width, self.dim, device=device, dtype=dtype)
        
        for y in range(height):
            for w in range(width):
                cos_2d[y, w, :self.dim_x] = freqs_x[w].cos().to(dtype)
                sin_2d[y, w, :self.dim_x] = freqs_x[w].sin().to(dtype)
                cos_2d[y, w, self.dim_x:] = freqs_y[y].cos().to(dtype)
                sin_2d[y, w, self.dim_x:] = freqs_y[y].sin().to(dtype)
        
        cos_2d = cos_2d.view(height * width, self.dim)
        sin_2d = sin_2d.view(height * width, self.dim)
        
        return cos_2d, sin_2d


def apply_rope_2d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class ImageExpert(nn.Module):
    """Single expert for DiT with SwiGLU activation."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class ImageMoERouter(nn.Module):
    """Router for Image MoE with spatial awareness."""

    def __init__(self, hidden_size: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.norm = nn.LayerNorm(hidden_size)
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.norm(x)
        router_logits = self.gate(x_norm)
        router_probs = F.softmax(router_logits, dim=-1, dtype=x.dtype)
        
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + EPS)
        
        return top_k_probs, top_k_indices


class ImageMoELayer(nn.Module):
    """MoE Layer for DiT with shared expert."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.router = ImageMoERouter(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList([
            ImageExpert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
        self.shared_expert = ImageExpert(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        
        top_k_probs, top_k_indices = self.router(x_flat)
        
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


class DualStreamSelfAttention(nn.Module):
    """
    Symmetric Dual-Stream Self-Attention (SD3/Flux-style).
    Two parallel streams with cross-stream information exchange.
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, max_height: int = 64, max_width: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_qkv_a = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.to_qkv_b = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        
        self.to_out_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_out_b = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.norm_a = nn.LayerNorm(hidden_size)
        self.norm_b = nn.LayerNorm(hidden_size)
        
        self.rope_2d = RoPE2D(self.head_dim, max_height, max_width)

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x_a.shape
        
        x_a = self.norm_a(x_a)
        x_b = self.norm_b(x_b)
        
        qkv_a = self.to_qkv_a(x_a).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv_b = self.to_qkv_b(x_b).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        
        q_a, k_a, v_a = qkv_a.unbind(dim=2)
        q_b, k_b, v_b = qkv_b.unbind(dim=2)
        
        cos, sin = self.rope_2d(x_a, height, width)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        
        q_a = q_a.transpose(1, 2)
        k_a = k_a.transpose(1, 2)
        v_a = v_a.transpose(1, 2)
        q_b = q_b.transpose(1, 2)
        k_b = k_b.transpose(1, 2)
        v_b = v_b.transpose(1, 2)
        
        q_a = apply_rope_2d(q_a, cos, sin)
        k_a = apply_rope_2d(k_a, cos, sin)
        q_b = apply_rope_2d(q_b, cos, sin)
        k_b = apply_rope_2d(k_b, cos, sin)
        
        k_combined = torch.cat([k_a, k_b], dim=2)
        v_combined = torch.cat([v_a, v_b], dim=2)
        
        attn_a = torch.matmul(q_a, k_combined.transpose(-1, -2)) * self.scale
        attn_a = F.softmax(attn_a, dim=-1, dtype=torch.float32).to(x_a.dtype)
        out_a = torch.matmul(attn_a, v_combined)
        
        attn_b = torch.matmul(q_b, k_combined.transpose(-1, -2)) * self.scale
        attn_b = F.softmax(attn_b, dim=-1, dtype=torch.float32).to(x_b.dtype)
        out_b = torch.matmul(attn_b, v_combined)
        
        out_a = out_a.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        out_b = out_b.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        
        out_a = self.to_out_a(out_a)
        out_b = self.to_out_b(out_b)
        
        return out_a, out_b


class CrossAttention(nn.Module):
    """Cross-attention for text conditioning."""

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


class DiTBlock(nn.Module):
    """
    DiT Block with Dual-Stream Attention and MoE FFN.
    """

    def __init__(self, hidden_size: int, context_dim: int, num_heads: int = 8, num_experts: int = 4, max_height: int = 64, max_width: int = 64):
        super().__init__()
        
        self.dual_attn = DualStreamSelfAttention(hidden_size, num_heads, max_height, max_width)
        self.cross_attn_a = CrossAttention(hidden_size, context_dim, num_heads)
        self.cross_attn_b = CrossAttention(hidden_size, context_dim, num_heads)
        self.moe_a = ImageMoELayer(hidden_size, hidden_size * 4, num_experts)
        self.moe_b = ImageMoELayer(hidden_size, hidden_size * 4, num_experts)
        
        self.adaLN_a = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6),
        )
        self.adaLN_b = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6),
        )
        
        self.norm1_a = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm1_b = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2_a = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2_b = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor, context: torch.Tensor, t_emb: torch.Tensor, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        shift_a, scale_a, gate_a, shift2_a, scale2_a, gate2_a = self.adaLN_a(t_emb).chunk(6, dim=-1)
        shift_b, scale_b, gate_b, shift2_b, scale2_b, gate2_b = self.adaLN_b(t_emb).chunk(6, dim=-1)
        
        shift_a = shift_a.unsqueeze(1)
        scale_a = scale_a.unsqueeze(1)
        gate_a = gate_a.unsqueeze(1)
        shift2_a = shift2_a.unsqueeze(1)
        scale2_a = scale2_a.unsqueeze(1)
        gate2_a = gate2_a.unsqueeze(1)
        
        shift_b = shift_b.unsqueeze(1)
        scale_b = scale_b.unsqueeze(1)
        gate_b = gate_b.unsqueeze(1)
        shift2_b = shift2_b.unsqueeze(1)
        scale2_b = scale2_b.unsqueeze(1)
        gate2_b = gate2_b.unsqueeze(1)
        
        x_a_norm = self.norm1_a(x_a) * (1 + scale_a) + shift_a
        x_b_norm = self.norm1_b(x_b) * (1 + scale_b) + shift_b
        
        attn_out_a, attn_out_b = self.dual_attn(x_a_norm, x_b_norm, height, width)
        x_a = x_a + gate_a * attn_out_a
        x_b = x_b + gate_b * attn_out_b
        
        x_a = x_a + self.cross_attn_a(x_a, context)
        x_b = x_b + self.cross_attn_b(x_b, context)
        
        x_a_norm = self.norm2_a(x_a) * (1 + scale2_a) + shift2_a
        x_b_norm = self.norm2_b(x_b) * (1 + scale2_b) + shift2_b
        
        x_a = x_a + gate2_a * self.moe_a(x_a_norm)
        x_b = x_b + gate2_b * self.moe_b(x_b_norm)
        
        return x_a, x_b


class FlowMatchingScheduler:
    """Flow Matching scheduler for image generation."""

    def __init__(self, num_steps: int = 50, sigma_min: float = 0.002):
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.timesteps = torch.linspace(1, 0, num_steps + 1)

    def get_velocity(self, x_t: torch.Tensor, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x_0 - x_t

    def step(self, model_output: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        dt = t - t_prev
        x_prev = x_t + model_output * dt.view(-1, 1, 1, 1)
        return x_prev

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x_0)
        # Ensure t has same dtype as x_0 for consistent math
        t = t.to(x_0.dtype).view(-1, 1, 1, 1)
        x_t = t * noise + (1 - t) * x_0
        return x_t


class PatchEmbed(nn.Module):
    """Patch embedding for DiT."""

    def __init__(self, patch_size: int = 2, in_channels: int = 4, hidden_size: int = 512):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class UnpatchEmbed(nn.Module):
    """Unpatch embedding to reconstruct image from patches."""

    def __init__(self, patch_size: int = 2, out_channels: int = 4, hidden_size: int = 512):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.proj = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = self.proj(x)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, height, width, self.patch_size, self.patch_size, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(batch_size, self.out_channels, height * self.patch_size, width * self.patch_size)
        return x


class MoEDiT(nn.Module):
    """
    MoE Diffusion Transformer with Dual-Stream Attention.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        hidden_size: int = 512,
        context_dim: int = 1024,
        num_layers: int = 8,
        num_heads: int = 8,
        num_experts: int = 4,
        patch_size: int = 2,
        max_image_size: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        max_patches = max_image_size // patch_size
        
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        
        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_size)
        self.context_proj = nn.Linear(context_dim, hidden_size)
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, hidden_size, num_heads, num_experts, max_patches, max_patches)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_size)
        self.unpatch_embed = UnpatchEmbed(patch_size, out_channels, hidden_size)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.unpatch_embed.proj.weight)
        nn.init.zeros_(self.unpatch_embed.proj.bias)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        
        half_dim = self.hidden_size // 2
        t_emb = math.log(10000) / (half_dim - 1)
        t_emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=x.dtype) * -t_emb)
        t_emb = timesteps[:, None].to(x.dtype) * t_emb[None, :]
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
        t_emb = self.time_embed(t_emb)
        
        x_patches = self.patch_embed(x)
        
        context_proj = self.context_proj(context)
        
        x_a = x_patches
        x_b = x_patches.clone()
        
        for block in self.blocks:
            x_a, x_b = block(x_a, x_b, context_proj, t_emb, patch_height, patch_width)
        
        x_combined = (x_a + x_b) / 2
        x_combined = self.final_norm(x_combined)
        
        velocity = self.unpatch_embed(x_combined, patch_height, patch_width)
        
        return velocity


class ImageVAE(nn.Module):
    """Lightweight VAE for image encoding/decoding."""

    def __init__(self, in_channels: int = 3, latent_channels: int = 4, base_channels: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, latent_channels * 2, 3, padding=1),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels * 4, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
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


class MobileDiffusionGenerator(nn.Module):
    """
    SOTA Image Diffusion with MoE-DiT, Flow Matching, 2D-RoPE, Dual-Stream.
    Optimized for 2x T4 GPUs with FP16.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        base_channels: int = 128,
        context_dim: int = 1024,
        num_inference_steps: int = 50,
        image_size: int = 384,  # Match SigLIP 384x384
        cfg_scale: float = 7.5,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.context_dim = context_dim
        self.image_size = image_size
        self.latent_size = image_size // 4
        self.num_inference_steps = num_inference_steps
        self.cfg_scale = cfg_scale
        
        self.vae_encoder = ImageVAE(3, latent_channels, base_channels // 2)
        self.vae_decoder = self.vae_encoder
        
        self.unet = MoEDiT(
            in_channels=latent_channels,
            out_channels=latent_channels,
            hidden_size=base_channels * 4,
            context_dim=context_dim,
            num_layers=8,
            num_heads=8,
            num_experts=4,
            patch_size=2,
            max_image_size=self.latent_size,
        )
        
        self.scheduler = FlowMatchingScheduler(num_inference_steps)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.vae_encoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae_decoder.decode(z)

    def training_step(self, images: torch.Tensor, context: torch.Tensor, mask: Optional[torch.Tensor] = None) -> dict:
        device = images.device
        dtype = images.dtype  # Match dtype to input (FP16/BF16/FP32)
        batch_size = images.shape[0]
        
        z, mean, logvar = self.encode(images * 2 - 1)
        del images  # Free memory immediately after encoding
        
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
        
        pred_velocity = self.unet(x_t, (t * 1000).to(dtype), context, mask)
        del x_t, context  # Free memory
        
        flow_loss = F.mse_loss(pred_velocity, target_velocity)
        del pred_velocity, target_velocity  # Free memory
        
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        del z, mean, logvar  # Free memory
        
        total_loss = flow_loss + 0.0001 * kl_loss
        
        return {
            'flow_loss': flow_loss,
            'kl_loss': kl_loss,
            'total_loss': total_loss,
        }

    @torch.no_grad()
    def generate(self, context: torch.Tensor, guidance_scale: float = None, num_steps: int = None, init_latents: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None, masked_image_latents: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = context.device
        batch_size = context.shape[0]
        seq_len = context.shape[1]
        guidance_scale = guidance_scale or self.cfg_scale
        num_steps = num_steps or self.num_inference_steps
        
        if init_latents is not None:
            latents = init_latents
        else:
            latents = torch.randn(batch_size, self.latent_channels, self.latent_size, self.latent_size, device=device)
        
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
                velocity_pred = self.unet(latent_input, t_input, context, mask)
                velocity_uncond, velocity_cond = velocity_pred.chunk(2)
                velocity_pred = velocity_uncond + guidance_scale * (velocity_cond - velocity_uncond)
            else:
                velocity_pred = self.unet(latents, t_batch, context, mask)
            
            latents = self.scheduler.step(velocity_pred, t, t_prev, latents)
            
            if mask is not None and masked_image_latents is not None:
                latents = masked_image_latents * mask + latents * (1 - mask)
        
        images = self.decode(latents)
        images = (images + 1) / 2
        return torch.clamp(images, 0, 1)

    @torch.no_grad()
    def edit_image(self, image: torch.Tensor, context: torch.Tensor, mask: torch.Tensor, strength: float = 0.8, guidance_scale: float = None) -> torch.Tensor:
        device = image.device
        
        image_norm = image * 2 - 1
        z, _, _ = self.encode(image_norm)
        
        mask_latent = F.interpolate(mask, size=(self.latent_size, self.latent_size), mode='nearest')
        
        num_steps = int(self.num_inference_steps * strength)
        
        t = torch.tensor([strength], device=device)
        noisy_z = self.scheduler.add_noise(z, t.expand(z.shape[0]))
        
        return self.generate(
            context,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            init_latents=noisy_z,
            mask=mask_latent,
            masked_image_latents=z,
        )

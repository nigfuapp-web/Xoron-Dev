"""
SOTA Video Encoder with 3D-RoPE, 3D Causal Attention, and Temporal Expert Routing.

Features:
- 3D-RoPE for flexible (x, y, t) positional encodings (matches video generator)
- 3D Causal Attention for temporal understanding
- Temporal-Aware Expert Routing for motion patterns
- Integrated with vision encoder backbone
- FP16-native numerical stability
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from models.encoders.vision import VisionEncoder

EPS = 1e-5


class RoPE3DEncoder(nn.Module):
    """
    3D Rotary Position Embedding for (x, y, t) dimensions.
    Matches the 3D-RoPE in video generator for seamless integration.
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


def apply_rope_3d_encoder(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply 3D rotary position embedding to tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class TemporalExpertRouterEncoder(nn.Module):
    """
    Temporal-Aware Expert Router for video encoding.
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


class VideoExpertEncoder(nn.Module):
    """Single expert for video encoding with SwiGLU."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TemporalMoELayerEncoder(nn.Module):
    """
    Temporal-Aware MoE Layer for video encoding.
    Uses motion-aware routing for expert selection.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.router = TemporalExpertRouterEncoder(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList([
            VideoExpertEncoder(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
        self.shared_expert = VideoExpertEncoder(hidden_size, intermediate_size)

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


class Causal3DAttentionEncoder(nn.Module):
    """
    3D Causal Self-Attention with 3D-RoPE for video encoding.
    Attends to all positions for encoding (non-causal during encoding).
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, max_frames: int = 32, max_height: int = 64, max_width: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.norm = nn.LayerNorm(hidden_size)
        
        self.rope_3d = RoPE3DEncoder(self.head_dim, max_height, max_width, max_frames)

    def forward(self, x: torch.Tensor, height: int, width: int, frames: int, causal: bool = False) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        x_norm = self.norm(x)
        qkv = self.to_qkv(x_norm).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        cos, sin = self.rope_3d(x, height, width, frames)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        q = apply_rope_3d_encoder(q, cos, sin)
        k = apply_rope_3d_encoder(k, cos, sin)
        
        # Use causal mask for autoregressive temporal modeling
        if causal:
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            attn_output = F.scaled_dot_product_attention(q, k, v)
        
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        return self.to_out(attn_output)


class VideoEncoderBlock(nn.Module):
    """Single block with 3D causal attention and temporal MoE FFN."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        num_experts: int = 4,
        max_frames: int = 32,
        max_height: int = 64,
        max_width: int = 64,
    ):
        super().__init__()
        self.attn = Causal3DAttentionEncoder(hidden_size, num_heads, max_frames, max_height, max_width)
        self.moe = TemporalMoELayerEncoder(hidden_size, hidden_size * 4, num_experts)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, height: int, width: int, frames: int, causal: bool = False) -> torch.Tensor:
        x = x + self.attn(x, height, width, frames, causal)
        x = self.norm(x + self.moe(x))
        return x


class VideoEncoder(nn.Module):
    """
    SOTA Video Encoder with 3D-RoPE, 3D Causal Attention, and Temporal Expert Routing.
    
    Features:
    - 3D-RoPE for flexible (x, y, t) positional encodings
    - 3D Causal Attention for temporal understanding
    - Temporal-Aware Expert Routing for motion patterns
    - Integrated with vision encoder backbone
    - FP16-native numerical stability
    """

    def __init__(
        self,
        vision_encoder: VisionEncoder,
        max_frames: int = 32,
        num_encoder_layers: int = 4,
        num_experts: int = 4,
        use_3d_rope: bool = True,
        use_temporal_moe: bool = True,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.max_frames = max_frames
        self.hidden_size = vision_encoder.hidden_size
        self.use_3d_rope = use_3d_rope
        self.use_temporal_moe = use_temporal_moe
        
        # Get expected image size from vision encoder
        self.image_size = getattr(vision_encoder, 'image_size', 384)
        self.patch_size = getattr(vision_encoder.vision_model.config, 'patch_size', 14)
        self.patches_per_side = self.image_size // self.patch_size

        # 3D-RoPE for spatio-temporal position encoding
        if use_3d_rope:
            self.rope_3d = RoPE3DEncoder(
                dim=self.hidden_size,
                max_height=self.patches_per_side,
                max_width=self.patches_per_side,
                max_frames=max_frames,
            )
            print(f"   📐 3D-RoPE: (x,y,t) position encoding")
        else:
            self.rope_3d = None
        
        # 3D Causal Transformer blocks with temporal MoE
        self.encoder_blocks = nn.ModuleList([
            VideoEncoderBlock(
                hidden_size=self.hidden_size,
                num_heads=8,
                num_experts=num_experts if use_temporal_moe else 1,
                max_frames=max_frames,
                max_height=self.patches_per_side,
                max_width=self.patches_per_side,
            )
            for _ in range(num_encoder_layers)
        ])
        print(f"   🎬 3D Causal Transformer: {num_encoder_layers} layers")
        
        if use_temporal_moe:
            print(f"   🎯 Temporal MoE: {num_experts} experts per layer")

        # Temporal pooling attention for video-level representation
        self.temporal_pool_query = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)
        self.temporal_pool_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            batch_first=True,
            dropout=0.1,
        )
        self.temporal_pool_norm = nn.LayerNorm(self.hidden_size)
        
        # Learnable frame position embeddings
        self.frame_pos_embed = nn.Parameter(torch.randn(1, max_frames, self.hidden_size) * 0.02)

        print(f"   🎬 Video encoder: max {max_frames} frames @ {self.image_size}x{self.image_size}")

    def _extract_frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract per-frame features using vision encoder."""
        batch_size, num_frames = frames.shape[:2]
        
        # Flatten frames for batch processing
        frames_flat = frames.view(-1, *frames.shape[2:])
        
        # Resize to expected size for vision encoder
        if frames_flat.shape[-1] != self.image_size or frames_flat.shape[-2] != self.image_size:
            frames_flat = F.interpolate(
                frames_flat, 
                size=(self.image_size, self.image_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Extract frame features using vision encoder (without TiTok for video)
        if not any(p.requires_grad for p in self.vision_encoder.parameters()):
            with torch.no_grad():
                frame_features = self.vision_encoder(frames_flat, return_titok=False)
        else:
            frame_features = self.vision_encoder(frames_flat, return_titok=False)
        
        return frame_features, batch_size, num_frames

    def forward(self, frames: torch.Tensor, return_all_frames: bool = False, causal: bool = False) -> torch.Tensor:
        """
        Process video frames with 3D-RoPE and Causal Attention.
        
        Args:
            frames: [B, T, C, H, W] tensor of video frames
            return_all_frames: If True, return all frame features; else return pooled
            causal: If True, use causal attention (for autoregressive)
            
        Returns:
            [B, T, hidden_size] if return_all_frames else [B, hidden_size]
        """
        frame_features, batch_size, num_frames = self._extract_frame_features(frames)
        
        # Get spatial dimensions (patches per frame)
        _, num_patches, hidden_size = frame_features.shape
        height = width = int(math.sqrt(num_patches))
        
        # Reshape to [B, T, H*W, D] then [B, T*H*W, D] for 3D processing
        frame_features = frame_features.view(batch_size, num_frames, num_patches, hidden_size)
        
        # Add frame position embeddings
        frame_features = frame_features + self.frame_pos_embed[:, :num_frames].unsqueeze(2)
        
        # Flatten spatio-temporal for 3D attention
        x = frame_features.view(batch_size, num_frames * num_patches, hidden_size)
        
        # Apply 3D transformer blocks
        for block in self.encoder_blocks:
            x = block(x, height, width, num_frames, causal=causal)
        
        if return_all_frames:
            # Return per-frame features (mean over patches per frame)
            x = x.view(batch_size, num_frames, num_patches, hidden_size)
            return x.mean(dim=2)  # [B, T, hidden_size]
        else:
            # Temporal pooling for video-level representation
            query = self.temporal_pool_query.expand(batch_size, -1, -1)
            pooled, _ = self.temporal_pool_attn(query, x, x)
            pooled = self.temporal_pool_norm(query + pooled)
            return pooled.squeeze(1)  # [B, hidden_size]
    
    def encode_frames_separately(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode frames without temporal attention (for generation conditioning).
        
        Args:
            frames: [B, T, C, H, W] tensor of video frames
            
        Returns:
            [B, T, hidden_size] tensor of frame features
        """
        frame_features, batch_size, num_frames = self._extract_frame_features(frames)
        
        # Pool spatial features (mean over patches)
        frame_features = frame_features.mean(dim=1)  # [B*T, hidden_size]
        return frame_features.view(batch_size, num_frames, -1)  # [B, T, hidden_size]
    
    def encode_with_spatial(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode frames preserving spatial structure (for video generation).
        
        Args:
            frames: [B, T, C, H, W] tensor of video frames
            
        Returns:
            [B, T, H, W, hidden_size] tensor of spatio-temporal features
        """
        frame_features, batch_size, num_frames = self._extract_frame_features(frames)
        
        _, num_patches, hidden_size = frame_features.shape
        height = width = int(math.sqrt(num_patches))
        
        # Reshape to spatio-temporal grid
        frame_features = frame_features.view(batch_size, num_frames, height, width, hidden_size)
        
        return frame_features

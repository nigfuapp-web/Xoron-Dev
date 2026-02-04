"""
SOTA Multimodal projectors for mapping between modality spaces.

Features:
- Perceiver Resampler for efficient feature compression
- Spatial-aware projection preserving 2D structure
- C-Abstractor for compressed abstraction
- Multi-scale feature fusion
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PerceiverAttention(nn.Module):
    """
    Perceiver-style cross-attention for resampling.
    Learnable queries attend to input features.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.inner_dim = inner_dim
        self.scale = dim_head ** -0.5
        
        self.norm_latents = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, latents: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        latents: [B, num_latents, dim] - learnable queries
        context: [B, seq_len, dim] - input features to attend to
        """
        latents = self.norm_latents(latents)
        context = self.norm_context(context)
        
        b, n, _ = latents.shape
        ctx_len = context.shape[1]
        h = self.num_heads
        d = self.dim_head
        
        q = self.to_q(latents)
        kv = self.to_kv(context).chunk(2, dim=-1)
        k, v = kv
        
        # Reshape for multi-head attention [B, num_heads, seq_len, dim_head]
        q = q.reshape(b, n, h, d).transpose(1, 2)
        k = k.reshape(b, ctx_len, h, d).transpose(1, 2)
        v = v.reshape(b, ctx_len, h, d).transpose(1, 2)
        
        # Attention (FP16-safe: clamp before softmax to prevent exp() overflow)
        # ln(65504) ≈ 11.09, so exp(11) is the max safe value for FP16 softmax
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = torch.clamp(attn, min=-11.0, max=11.0)
        attn = attn.softmax(dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, self.inner_dim)
        
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler for efficient feature compression.
    
    Uses learnable latent queries to compress variable-length
    input features into fixed-length output.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_latents: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_latents = num_latents
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
        # Learnable latent queries
        self.latents = nn.Parameter(torch.randn(1, num_latents, output_dim) * 0.02)
        
        # Perceiver layers
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(output_dim, num_heads, output_dim // num_heads, dropout),
                nn.Sequential(
                    nn.LayerNorm(output_dim),
                    nn.Linear(output_dim, output_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(output_dim * 4, output_dim),
                    nn.Dropout(dropout),
                )
            ])
            for _ in range(num_layers)
        ])
        
        self.norm_out = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, seq_len, input_dim] - input features
        returns: [B, num_latents, output_dim] - compressed features
        """
        batch_size = x.shape[0]
        
        # Project input
        x = self.input_proj(x)
        
        # Expand latents for batch
        latents = self.latents.expand(batch_size, -1, -1)
        
        # Apply perceiver layers
        for attn, ff in self.layers:
            latents = latents + attn(latents, x)
            latents = latents + ff(latents)
        
        return self.norm_out(latents)


class SpatialAwareProjector(nn.Module):
    """
    Spatial-aware projector that preserves 2D structure.
    
    Uses 2D convolutions to maintain spatial relationships
    before projecting to LLM space.
    """
    
    def __init__(
        self,
        vision_hidden_size: int,
        llm_hidden_size: int,
        num_tokens: int = 64,
        spatial_pool_size: int = 8,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.spatial_pool_size = spatial_pool_size
        
        # Spatial processing with 2D convolutions
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(vision_hidden_size, llm_hidden_size, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(llm_hidden_size, llm_hidden_size, 3, padding=1),
            nn.GELU(),
        )
        
        # Adaptive pooling to fixed spatial size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((spatial_pool_size, spatial_pool_size))
        
        # Final projection
        self.proj = nn.Sequential(
            nn.Linear(llm_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )
        
        self.norm = nn.LayerNorm(llm_hidden_size)
        
        # Learnable position embeddings for spatial tokens
        self.pos_embed = nn.Parameter(
            torch.randn(1, spatial_pool_size * spatial_pool_size, llm_hidden_size) * 0.02
        )
        
    def forward(self, vision_features: torch.Tensor, spatial_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        vision_features: [B, seq_len, vision_hidden_size] or [B, H, W, vision_hidden_size]
        returns: [B, num_tokens, llm_hidden_size]
        """
        batch_size = vision_features.shape[0]
        
        # Reshape to spatial if needed
        if vision_features.dim() == 3:
            seq_len = vision_features.shape[1]
            if spatial_size is None:
                h = w = int(math.sqrt(seq_len))
            else:
                h, w = spatial_size
            vision_features = vision_features.view(batch_size, h, w, -1)
        
        # [B, H, W, C] -> [B, C, H, W]
        x = vision_features.permute(0, 3, 1, 2)
        
        # Spatial convolutions
        x = self.spatial_conv(x)
        
        # Pool to fixed size
        x = self.adaptive_pool(x)
        
        # [B, C, H, W] -> [B, H*W, C]
        x = x.flatten(2).transpose(1, 2)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Final projection
        x = self.proj(x)
        x = self.norm(x)
        
        return x


class CAbstractor(nn.Module):
    """
    C-Abstractor: Compressed Abstraction for efficient multimodal fusion.
    
    Combines spatial pooling with cross-attention for
    efficient feature compression while preserving important information.
    """
    
    def __init__(
        self,
        vision_hidden_size: int,
        llm_hidden_size: int,
        num_tokens: int = 64,
        num_heads: int = 8,
        compression_ratio: int = 4,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        
        # Initial projection
        self.input_proj = nn.Linear(vision_hidden_size, llm_hidden_size)
        
        # Compression via strided convolution
        self.compress = nn.Sequential(
            nn.Conv1d(llm_hidden_size, llm_hidden_size, kernel_size=compression_ratio, stride=compression_ratio),
            nn.GELU(),
        )
        
        # Learnable abstraction queries
        self.queries = nn.Parameter(torch.randn(1, num_tokens, llm_hidden_size) * 0.02)
        
        # Cross-attention for abstraction
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=llm_hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.LayerNorm(llm_hidden_size),
            nn.Linear(llm_hidden_size, llm_hidden_size * 4),
            nn.GELU(),
            nn.Linear(llm_hidden_size * 4, llm_hidden_size),
        )
        
        self.norm = nn.LayerNorm(llm_hidden_size)
        
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        vision_features: [B, seq_len, vision_hidden_size]
        returns: [B, num_tokens, llm_hidden_size]
        """
        batch_size = vision_features.shape[0]
        
        # Project to LLM dimension
        x = self.input_proj(vision_features)
        
        # Compress via strided convolution
        x = x.transpose(1, 2)  # [B, C, seq_len]
        x = self.compress(x)
        x = x.transpose(1, 2)  # [B, compressed_len, C]
        
        # Cross-attention with learnable queries
        queries = self.queries.expand(batch_size, -1, -1)
        abstracted, _ = self.cross_attn(queries, x, x)
        
        # Feed-forward
        abstracted = abstracted + self.ff(abstracted)
        
        return self.norm(abstracted)


class MultimodalProjector(nn.Module):
    """
    SOTA Multimodal Projector with multiple projection strategies.
    
    Combines:
    - Perceiver Resampler for efficient compression
    - Spatial-aware processing
    - Multi-scale feature fusion
    """

    def __init__(
        self,
        vision_hidden_size: int,
        llm_hidden_size: int,
        num_tokens: int = 64,
        projector_type: str = "perceiver",  # "perceiver", "spatial", "c_abstractor", "mlp"
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.projector_type = projector_type

        if projector_type == "perceiver":
            self.projector = PerceiverResampler(
                input_dim=vision_hidden_size,
                output_dim=llm_hidden_size,
                num_latents=num_tokens,
                num_heads=num_heads,
                num_layers=num_layers,
            )
        elif projector_type == "spatial":
            self.projector = SpatialAwareProjector(
                vision_hidden_size=vision_hidden_size,
                llm_hidden_size=llm_hidden_size,
                num_tokens=num_tokens,
            )
        elif projector_type == "c_abstractor":
            self.projector = CAbstractor(
                vision_hidden_size=vision_hidden_size,
                llm_hidden_size=llm_hidden_size,
                num_tokens=num_tokens,
                num_heads=num_heads,
            )
        else:  # "mlp" - simple but effective
            self.projector = nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size),
            )
            # For MLP, we need query tokens and cross-attention
            self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, llm_hidden_size) * 0.02)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=llm_hidden_size,
                num_heads=num_heads,
                batch_first=True
            )
            self.norm = nn.LayerNorm(llm_hidden_size)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Project and resample vision features."""
        if self.projector_type in ["perceiver", "spatial", "c_abstractor"]:
            return self.projector(vision_features)
        else:
            # MLP with cross-attention resampling
            batch_size = vision_features.shape[0]
            projected = self.projector(vision_features)
            queries = self.query_tokens.expand(batch_size, -1, -1)
            resampled, _ = self.cross_attn(queries, projected, projected)
            return self.norm(resampled)

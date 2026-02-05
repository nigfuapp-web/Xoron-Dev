"""
SOTA Multimodal projectors for mapping between modality spaces.

Features:
- Locality-Enhanced ResNet Abstractor: Residual bottleneck blocks for fine-grained features
- Multi-Scale Feature Fusion (MSFF): Features from multiple encoder depths
- Multi-Scale Deformable Attention: Non-uniform region attention
- Dynamic Token Router: Sparse gating for KV-cache efficiency
- 2D/3D Rotary Positional Embeddings (RoPE): Spatial/temporal awareness
- Perceiver Resampler for efficient feature compression
- Spatial-aware projection preserving 2D structure
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


def compute_2d_rope(height: int, width: int, dim: int, device: torch.device, dtype: torch.dtype, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute 2D Rotary Position Embeddings for spatial awareness.
    
    Args:
        height: Image height in patches
        width: Image width in patches
        dim: Embedding dimension (must be divisible by 4)
        device: Target device
        dtype: Target dtype
        base: RoPE base frequency
        
    Returns:
        cos, sin: [height*width, dim] position embeddings
    """
    assert dim % 4 == 0, "dim must be divisible by 4 for 2D RoPE"
    
    half_dim = dim // 2
    quarter_dim = dim // 4
    
    # Compute frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, quarter_dim, device=device, dtype=torch.float32) / quarter_dim))
    
    # Create position grids
    y_pos = torch.arange(height, device=device, dtype=torch.float32)
    x_pos = torch.arange(width, device=device, dtype=torch.float32)
    
    # Compute embeddings for each dimension
    y_emb = torch.outer(y_pos, inv_freq)  # [H, quarter_dim]
    x_emb = torch.outer(x_pos, inv_freq)  # [W, quarter_dim]
    
    # Expand to full grid
    y_emb = y_emb.unsqueeze(1).expand(-1, width, -1)  # [H, W, quarter_dim]
    x_emb = x_emb.unsqueeze(0).expand(height, -1, -1)  # [H, W, quarter_dim]
    
    # Concatenate and flatten
    emb = torch.cat([y_emb, y_emb, x_emb, x_emb], dim=-1)  # [H, W, dim]
    emb = emb.reshape(height * width, dim)
    
    return emb.cos().to(dtype), emb.sin().to(dtype)


def compute_3d_rope(
    depth: int, height: int, width: int, dim: int, 
    device: torch.device, dtype: torch.dtype, base: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute 3D Rotary Position Embeddings for video/temporal awareness.
    
    Args:
        depth: Temporal depth (number of frames)
        height: Image height in patches
        width: Image width in patches
        dim: Embedding dimension (must be divisible by 6)
        device: Target device
        dtype: Target dtype
        base: RoPE base frequency
        
    Returns:
        cos, sin: [depth*height*width, dim] position embeddings
    """
    assert dim % 6 == 0, "dim must be divisible by 6 for 3D RoPE"
    
    sixth_dim = dim // 6
    
    # Compute frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, sixth_dim, device=device, dtype=torch.float32) / sixth_dim))
    
    # Create position grids
    t_pos = torch.arange(depth, device=device, dtype=torch.float32)
    y_pos = torch.arange(height, device=device, dtype=torch.float32)
    x_pos = torch.arange(width, device=device, dtype=torch.float32)
    
    # Compute embeddings for each dimension
    t_emb = torch.outer(t_pos, inv_freq)  # [D, sixth_dim]
    y_emb = torch.outer(y_pos, inv_freq)  # [H, sixth_dim]
    x_emb = torch.outer(x_pos, inv_freq)  # [W, sixth_dim]
    
    # Expand to full grid
    t_emb = t_emb.unsqueeze(1).unsqueeze(2).expand(-1, height, width, -1)  # [D, H, W, sixth_dim]
    y_emb = y_emb.unsqueeze(0).unsqueeze(2).expand(depth, -1, width, -1)  # [D, H, W, sixth_dim]
    x_emb = x_emb.unsqueeze(0).unsqueeze(1).expand(depth, height, -1, -1)  # [D, H, W, sixth_dim]
    
    # Concatenate and flatten
    emb = torch.cat([t_emb, t_emb, y_emb, y_emb, x_emb, x_emb], dim=-1)  # [D, H, W, dim]
    emb = emb.reshape(depth * height * width, dim)
    
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings."""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class ResidualBottleneckBlock(nn.Module):
    """
    Residual Bottleneck Block for locality-enhanced feature extraction.
    
    Preserves small-scale features (OCR, fine audio events) during compression.
    """
    
    def __init__(self, in_channels: int, out_channels: int, bottleneck_ratio: float = 0.25):
        super().__init__()
        bottleneck_channels = int(out_channels * bottleneck_ratio)
        
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out = out + identity
        out = self.relu(out)
        
        return out


class LocalityEnhancedResNetAbstractor(nn.Module):
    """
    Locality-Enhanced ResNet Abstractor.
    
    Upgrades the C-Abstractor with residual bottleneck blocks to preserve
    small-scale features (OCR/fine audio events) during compression.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_tokens: int = 64,
        num_blocks: int = 3,
        use_2d_rope: bool = True,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.use_2d_rope = use_2d_rope
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, output_dim)
        
        # Residual bottleneck blocks
        self.blocks = nn.ModuleList([
            ResidualBottleneckBlock(output_dim, output_dim)
            for _ in range(num_blocks)
        ])
        
        # Learnable abstraction queries
        self.queries = nn.Parameter(torch.randn(1, num_tokens, output_dim) * 0.02)
        
        # Cross-attention with 2D RoPE
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1,
        )
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Linear(output_dim * 4, output_dim),
        )
        
        self.norm = nn.LayerNorm(output_dim)
        
        print(f"   🏗️ LocalityEnhancedResNetAbstractor: {input_dim} -> {output_dim}, {num_tokens} tokens")
    
    def forward(self, features: torch.Tensor, spatial_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Args:
            features: [B, seq_len, input_dim] or [B, H, W, input_dim]
            spatial_size: (H, W) if features are flattened
            
        Returns:
            abstracted: [B, num_tokens, output_dim]
        """
        batch_size = features.shape[0]
        
        # Project to output dimension
        x = self.input_proj(features)
        
        # Reshape to spatial if needed
        if features.dim() == 3:
            seq_len = features.shape[1]
            if spatial_size is None:
                h = w = int(math.sqrt(seq_len))
            else:
                h, w = spatial_size
            x = x.view(batch_size, h, w, -1)
        else:
            h, w = features.shape[1], features.shape[2]
        
        # Apply residual bottleneck blocks
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        # Flatten back
        x = x.reshape(batch_size, h * w, -1)
        
        # Apply 2D RoPE if enabled
        if self.use_2d_rope:
            cos, sin = compute_2d_rope(h, w, x.shape[-1], x.device, x.dtype)
            x = apply_rope(x, cos.unsqueeze(0), sin.unsqueeze(0))
        
        # Cross-attention with learnable queries
        queries = self.queries.expand(batch_size, -1, -1)
        abstracted, _ = self.cross_attn(queries, x, x)
        
        # Feed-forward
        abstracted = abstracted + self.ff(abstracted)
        
        return self.norm(abstracted)


class MultiScaleFeatureFusion(nn.Module):
    """
    Multi-Scale Feature Fusion (MSFF).
    
    Extracts and weights features from multiple encoder depths (early, mid, late)
    to capture both low-level textures and high-level semantics.
    """
    
    def __init__(
        self,
        feature_dims: List[int],
        output_dim: int,
        num_scales: int = 3,
    ):
        super().__init__()
        self.num_scales = num_scales
        
        # Projection for each scale
        self.scale_projs = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in feature_dims
        ])
        
        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
        )
        
        self.norm = nn.LayerNorm(output_dim)
        
        print(f"   🔀 MultiScaleFeatureFusion: {feature_dims} -> {output_dim}")
    
    def forward(self, multi_scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            multi_scale_features: List of [B, seq_len, dim] features from different depths
            
        Returns:
            fused: [B, seq_len, output_dim]
        """
        assert len(multi_scale_features) == self.num_scales
        
        # Project each scale
        projected = []
        for i, (features, proj) in enumerate(zip(multi_scale_features, self.scale_projs)):
            projected.append(proj(features))
        
        # Weighted fusion
        weights = F.softmax(self.scale_weights, dim=0)
        fused = sum(w * p for w, p in zip(weights, projected))
        
        # Additional fusion
        fused = fused + self.fusion(fused)
        
        return self.norm(fused)


class MultiScaleDeformableAttention(nn.Module):
    """
    Multi-Scale Deformable Attention.
    
    Replaces fixed-grid cross-attention in Perceiver Resamplers,
    allowing the projector to "look" at non-uniform regions of interest.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = dim // num_heads
        
        # Sampling offsets predictor
        self.sampling_offsets = nn.Linear(dim, num_heads * num_levels * num_points * 2)
        
        # Attention weights predictor
        self.attention_weights = nn.Linear(dim, num_heads * num_levels * num_points)
        
        # Value projection
        self.value_proj = nn.Linear(dim, dim)
        
        # Output projection
        self.output_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
        
        print(f"   🎯 MultiScaleDeformableAttention: {dim}d, {num_heads}H, {num_levels}L, {num_points}P")
    
    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        nn.init.xavier_uniform_(self.attention_weights.weight)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
    
    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
        input_spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: [B, num_queries, dim]
            reference_points: [B, num_queries, num_levels, 2] normalized reference points
            input_flatten: [B, sum(H*W), dim] flattened multi-scale features
            input_spatial_shapes: [num_levels, 2] spatial shapes of each level
            
        Returns:
            output: [B, num_queries, dim]
        """
        batch_size, num_queries, _ = query.shape
        
        # Predict sampling offsets
        offsets = self.sampling_offsets(query)
        offsets = offsets.view(batch_size, num_queries, self.num_heads, self.num_levels, self.num_points, 2)
        
        # Predict attention weights
        attn_weights = self.attention_weights(query)
        attn_weights = attn_weights.view(batch_size, num_queries, self.num_heads, self.num_levels * self.num_points)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.view(batch_size, num_queries, self.num_heads, self.num_levels, self.num_points)
        
        # Compute sampling locations
        sampling_locations = reference_points.unsqueeze(2).unsqueeze(4) + offsets * 0.1
        sampling_locations = sampling_locations.clamp(0, 1)
        
        # Project values
        value = self.value_proj(input_flatten)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Sample features (simplified bilinear sampling)
        # In practice, this would use grid_sample for proper deformable attention
        output = torch.zeros(batch_size, num_queries, self.num_heads, self.head_dim, device=query.device, dtype=query.dtype)
        
        start_idx = 0
        for level_idx in range(self.num_levels):
            h, w = input_spatial_shapes[level_idx]
            end_idx = start_idx + h * w
            
            level_value = value[:, start_idx:end_idx]  # [B, H*W, num_heads, head_dim]
            level_value = level_value.view(batch_size, h, w, self.num_heads, self.head_dim)
            
            # Get sampling locations for this level
            level_locs = sampling_locations[:, :, :, level_idx]  # [B, Q, H, P, 2]
            level_weights = attn_weights[:, :, :, level_idx]  # [B, Q, H, P]
            
            # Simplified sampling (average over points)
            for point_idx in range(self.num_points):
                loc = level_locs[:, :, :, point_idx]  # [B, Q, H, 2]
                weight = level_weights[:, :, :, point_idx:point_idx+1]  # [B, Q, H, 1]
                
                # Convert normalized coords to indices
                y_idx = (loc[..., 0] * (h - 1)).long().clamp(0, h - 1)
                x_idx = (loc[..., 1] * (w - 1)).long().clamp(0, w - 1)
                
                # Gather values
                for b in range(batch_size):
                    for q in range(num_queries):
                        for head in range(self.num_heads):
                            y, x = y_idx[b, q, head].item(), x_idx[b, q, head].item()
                            output[b, q, head] += weight[b, q, head] * level_value[b, y, x, head]
            
            start_idx = end_idx
        
        # Reshape and project output
        output = output.view(batch_size, num_queries, self.dim)
        output = self.output_proj(output)
        output = self.dropout(output)
        
        return output


class DynamicTokenRouter(nn.Module):
    """
    Dynamic Token Router.
    
    Implements a sparse gating mechanism to drop redundant "background" tokens,
    drastically reducing KV-cache pressure for Ring Attention.
    """
    
    def __init__(
        self,
        dim: int,
        num_tokens: int,
        keep_ratio: float = 0.5,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.keep_ratio = keep_ratio
        self.temperature = temperature
        
        # Importance scorer
        self.scorer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        
        # Learnable threshold
        self.threshold = nn.Parameter(torch.tensor(0.0))
        
        print(f"   🚦 DynamicTokenRouter: keep_ratio={keep_ratio}")
    
    def forward(self, tokens: torch.Tensor, return_mask: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            tokens: [B, num_tokens, dim]
            return_mask: Whether to return the selection mask
            
        Returns:
            selected_tokens: [B, num_kept, dim]
            mask: [B, num_tokens] selection mask (if return_mask=True)
        """
        batch_size, num_tokens, _ = tokens.shape
        num_keep = max(1, int(num_tokens * self.keep_ratio))
        
        # Compute importance scores
        scores = self.scorer(tokens).squeeze(-1)  # [B, num_tokens]
        
        # Apply temperature and threshold
        scores = scores / self.temperature
        
        # Select top-k tokens
        _, indices = torch.topk(scores, num_keep, dim=-1)
        indices = indices.sort(dim=-1).values  # Keep original order
        
        # Gather selected tokens
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, self.dim)
        selected_tokens = torch.gather(tokens, 1, indices_expanded)
        
        if return_mask:
            mask = torch.zeros(batch_size, num_tokens, device=tokens.device, dtype=torch.bool)
            mask.scatter_(1, indices, True)
            return selected_tokens, mask
        
        return selected_tokens, None


class PerceiverAttention(nn.Module):
    """
    Perceiver-style cross-attention for resampling with 2D/3D RoPE support.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        use_rope: bool = True,
    ):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.inner_dim = inner_dim
        self.scale = dim_head ** -0.5
        self.use_rope = use_rope

        self.norm_latents = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        context_rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        latents: [B, num_latents, dim] - learnable queries
        context: [B, seq_len, dim] - input features to attend to
        context_rope: Optional (cos, sin) for context positions
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

        # Reshape for multi-head attention
        q = q.reshape(b, n, h, d).transpose(1, 2)
        k = k.reshape(b, ctx_len, h, d).transpose(1, 2)
        v = v.reshape(b, ctx_len, h, d).transpose(1, 2)

        # Apply RoPE to keys if provided
        if self.use_rope and context_rope is not None:
            cos, sin = context_rope
            cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
            sin = sin.unsqueeze(0).unsqueeze(0)
            k = apply_rope(k, cos, sin)

        # Attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = torch.clamp(attn, min=-11.0, max=11.0)
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, self.inner_dim)

        return self.to_out(out)


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler with 2D/3D RoPE and Dynamic Token Routing.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_latents: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_rope: bool = True,
        use_dynamic_routing: bool = False,
        routing_keep_ratio: float = 0.5,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.use_rope = use_rope
        self.use_dynamic_routing = use_dynamic_routing

        # Input projection
        self.input_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

        # Learnable latent queries
        self.latents = nn.Parameter(torch.randn(1, num_latents, output_dim) * 0.02)

        # Perceiver layers
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(output_dim, num_heads, output_dim // num_heads, dropout, use_rope),
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

        # Dynamic token router
        if use_dynamic_routing:
            self.token_router = DynamicTokenRouter(output_dim, num_latents, routing_keep_ratio)
        else:
            self.token_router = None

        self.norm_out = nn.LayerNorm(output_dim)

    def forward(
        self,
        x: torch.Tensor,
        spatial_size: Optional[Tuple[int, int]] = None,
        temporal_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        x: [B, seq_len, input_dim] - input features
        spatial_size: (H, W) for 2D RoPE
        temporal_size: T for 3D RoPE (video)
        returns: [B, num_latents, output_dim] - compressed features
        """
        batch_size = x.shape[0]

        # Project input
        x = self.input_proj(x)

        # Compute RoPE if spatial info provided
        context_rope = None
        if self.use_rope and spatial_size is not None:
            h, w = spatial_size
            if temporal_size is not None:
                # 3D RoPE for video
                cos, sin = compute_3d_rope(temporal_size, h, w, x.shape[-1], x.device, x.dtype)
            else:
                # 2D RoPE for images
                cos, sin = compute_2d_rope(h, w, x.shape[-1], x.device, x.dtype)
            context_rope = (cos, sin)

        # Expand latents for batch
        latents = self.latents.expand(batch_size, -1, -1)

        # Apply perceiver layers
        for attn, ff in self.layers:
            latents = latents + attn(latents, x, context_rope)
            latents = latents + ff(latents)

        latents = self.norm_out(latents)

        # Apply dynamic token routing if enabled
        if self.token_router is not None:
            latents, _ = self.token_router(latents)

        return latents


class SpatialAwareProjector(nn.Module):
    """
    Spatial-aware projector with 2D RoPE.
    """

    def __init__(
        self,
        vision_hidden_size: int,
        llm_hidden_size: int,
        num_tokens: int = 64,
        spatial_pool_size: int = 8,
        use_rope: bool = True,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.spatial_pool_size = spatial_pool_size
        self.use_rope = use_rope

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

    def forward(self, vision_features: torch.Tensor, spatial_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
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

        # Apply 2D RoPE
        if self.use_rope:
            cos, sin = compute_2d_rope(self.spatial_pool_size, self.spatial_pool_size, x.shape[-1], x.device, x.dtype)
            x = apply_rope(x, cos.unsqueeze(0), sin.unsqueeze(0))

        # Final projection
        x = self.proj(x)
        x = self.norm(x)

        return x


class CAbstractor(nn.Module):
    """
    C-Abstractor: Compressed Abstraction for efficient multimodal fusion.
    Now with 2D RoPE support.
    """

    def __init__(
        self,
        vision_hidden_size: int,
        llm_hidden_size: int,
        num_tokens: int = 64,
        num_heads: int = 8,
        compression_ratio: int = 4,
        use_rope: bool = True,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.use_rope = use_rope

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

    def forward(self, vision_features: torch.Tensor, spatial_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        batch_size = vision_features.shape[0]

        # Project to LLM dimension
        x = self.input_proj(vision_features)

        # Apply 2D RoPE before compression
        if self.use_rope and spatial_size is not None:
            h, w = spatial_size
            cos, sin = compute_2d_rope(h, w, x.shape[-1], x.device, x.dtype)
            x = apply_rope(x, cos.unsqueeze(0), sin.unsqueeze(0))

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
    SOTA Multimodal Projector with all advanced features.

    Combines:
    - Locality-Enhanced ResNet Abstractor
    - Multi-Scale Feature Fusion
    - Multi-Scale Deformable Attention
    - Dynamic Token Router
    - 2D/3D RoPE
    - Perceiver Resampler
    """

    def __init__(
        self,
        vision_hidden_size: int,
        llm_hidden_size: int,
        num_tokens: int = 64,
        projector_type: str = "perceiver",
        num_heads: int = 8,
        num_layers: int = 2,
        use_rope: bool = True,
        use_dynamic_routing: bool = False,
        use_locality_enhanced: bool = False,
        use_msff: bool = False,
        use_deformable_attn: bool = False,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.projector_type = projector_type
        self.use_rope = use_rope

        if projector_type == "perceiver":
            self.projector = PerceiverResampler(
                input_dim=vision_hidden_size,
                output_dim=llm_hidden_size,
                num_latents=num_tokens,
                num_heads=num_heads,
                num_layers=num_layers,
                use_rope=use_rope,
                use_dynamic_routing=use_dynamic_routing,
            )
        elif projector_type == "spatial":
            self.projector = SpatialAwareProjector(
                vision_hidden_size=vision_hidden_size,
                llm_hidden_size=llm_hidden_size,
                num_tokens=num_tokens,
                use_rope=use_rope,
            )
        elif projector_type == "c_abstractor":
            self.projector = CAbstractor(
                vision_hidden_size=vision_hidden_size,
                llm_hidden_size=llm_hidden_size,
                num_tokens=num_tokens,
                num_heads=num_heads,
                use_rope=use_rope,
            )
        elif projector_type == "locality_enhanced":
            self.projector = LocalityEnhancedResNetAbstractor(
                input_dim=vision_hidden_size,
                output_dim=llm_hidden_size,
                num_tokens=num_tokens,
                use_2d_rope=use_rope,
            )
        else:  # "mlp"
            self.projector = nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size),
            )
            self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, llm_hidden_size) * 0.02)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=llm_hidden_size,
                num_heads=num_heads,
                batch_first=True
            )
            self.norm = nn.LayerNorm(llm_hidden_size)

        # Optional MSFF
        if use_msff:
            self.msff = MultiScaleFeatureFusion(
                feature_dims=[vision_hidden_size] * 3,
                output_dim=vision_hidden_size,
            )
        else:
            self.msff = None

        # Optional deformable attention
        if use_deformable_attn:
            self.deformable_attn = MultiScaleDeformableAttention(
                dim=llm_hidden_size,
                num_heads=num_heads,
            )
        else:
            self.deformable_attn = None

        # Optional dynamic token router (post-projection)
        if use_dynamic_routing and projector_type != "perceiver":
            self.token_router = DynamicTokenRouter(llm_hidden_size, num_tokens)
        else:
            self.token_router = None

    def forward(
        self,
        vision_features: torch.Tensor,
        multi_scale_features: Optional[List[torch.Tensor]] = None,
        spatial_size: Optional[Tuple[int, int]] = None,
        temporal_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Project and resample vision features."""
        
        # Apply MSFF if available and multi-scale features provided
        if self.msff is not None and multi_scale_features is not None:
            vision_features = self.msff(multi_scale_features)

        if self.projector_type in ["perceiver"]:
            output = self.projector(vision_features, spatial_size, temporal_size)
        elif self.projector_type in ["spatial", "c_abstractor", "locality_enhanced"]:
            output = self.projector(vision_features, spatial_size)
        else:
            # MLP with cross-attention resampling
            batch_size = vision_features.shape[0]
            projected = self.projector(vision_features)
            queries = self.query_tokens.expand(batch_size, -1, -1)
            resampled, _ = self.cross_attn(queries, projected, projected)
            output = self.norm(resampled)

        # Apply dynamic token routing if available
        if self.token_router is not None:
            output, _ = self.token_router(output)

        return output

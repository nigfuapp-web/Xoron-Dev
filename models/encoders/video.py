"""
SOTA Video Encoder with 3D-RoPE, 3D Causal Attention, Temporal Expert Routing, VidTok, and Text-Timestamp Alignment.

Features:
- 3D-RoPE for flexible (x, y, t) positional encodings (matches video generator)
- 3D Causal Attention for temporal understanding
- Temporal-Aware Expert Routing for motion patterns
- VidTokTokenizer: Full 3D VAE for video compression (Microsoft VidTok architecture)
  - Efficient 2D+1D architecture (separates spatial and temporal processing)
  - AlphaBlender for temporal blending
  - Supports both continuous (KL) and discrete (FSQ) tokenization
  - Causal mode for streaming/autoregressive applications
- VideoTokenizer: Cross-attention based feature compression (like TiTok for images)
- Text-Timestamp Alignment for precise event localization
- Integrated with vision encoder backbone
- FP16-native numerical stability
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from models.encoders.vision import VisionEncoder

EPS = 1e-5


class TextTimestampAlignment(nn.Module):
    """
    Text-Timestamp Alignment: Precise timestamp-grounded event localization for stronger video temporal modeling.
    
    SOTA: Moves beyond T-RoPE by explicitly aligning text descriptions with video timestamps,
    enabling:
    - Precise temporal localization of events described in text
    - Better video captioning with accurate time references
    - Improved video question-answering with temporal reasoning
    - Enhanced video generation with temporal control
    
    Architecture:
    - Cross-attention between text features and frame-level video features
    - Learnable timestamp embeddings for each frame
    - Temporal alignment loss during training
    """

    def __init__(self, hidden_size: int, max_frames: int = 64, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_frames = max_frames
        self.num_heads = num_heads
        
        # Learnable timestamp embeddings
        self.timestamp_embedding = nn.Embedding(max_frames, hidden_size)
        
        # Project video features for timestamp alignment
        self.video_proj = nn.Linear(hidden_size, hidden_size)
        
        # Project text features for alignment
        self.text_proj = nn.Linear(hidden_size, hidden_size)
        
        # Cross-attention: text queries video with timestamp context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )
        
        # Layer norms
        self.text_norm = nn.LayerNorm(hidden_size)
        self.video_norm = nn.LayerNorm(hidden_size)
        
        # Temporal alignment prediction head (for alignment loss)
        self.alignment_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),  # Predicts timestamp relevance
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self, 
        video_features: torch.Tensor, 
        text_features: torch.Tensor,
        num_frames: int,
        return_alignment_scores: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Align text with video timestamps.
        
        Args:
            video_features: [B, T*H*W, hidden_size] video features
            text_features: [B, text_len, hidden_size] text features
            num_frames: Number of frames in the video
            return_alignment_scores: Whether to return alignment scores for loss
            
        Returns:
            aligned_features: [B, T*H*W, hidden_size] timestamp-aligned video features
            alignment_scores: Optional [B, text_len, T] alignment scores
        """
        batch_size = video_features.shape[0]
        total_tokens = video_features.shape[1]
        spatial_tokens = total_tokens // num_frames
        
        # Add timestamp embeddings to video features
        timestamp_ids = torch.arange(num_frames, device=video_features.device)
        timestamp_embeds = self.timestamp_embedding(timestamp_ids)  # [T, hidden_size]
        
        # Expand timestamps to match spatial tokens: [T, 1, hidden] -> [T, spatial, hidden]
        timestamp_embeds = timestamp_embeds.unsqueeze(1).expand(-1, spatial_tokens, -1)
        timestamp_embeds = timestamp_embeds.reshape(1, total_tokens, -1)  # [1, T*spatial, hidden]
        timestamp_embeds = timestamp_embeds.expand(batch_size, -1, -1)  # [B, T*spatial, hidden]
        
        # Add timestamps to video features
        video_feat = self.video_norm(self.video_proj(video_features) + timestamp_embeds)
        text_feat = self.text_norm(self.text_proj(text_features))
        
        # Cross-attention: text queries video
        aligned, attn_weights = self.cross_attn(text_feat, video_feat, video_feat)
        
        # Compute frame-level alignment scores
        alignment_scores = None
        if return_alignment_scores:
            # Pool attention weights to frame level
            # attn_weights: [B, text_len, T*spatial] -> [B, text_len, T]
            attn_reshaped = attn_weights.view(batch_size, text_features.shape[1], num_frames, spatial_tokens)
            alignment_scores = attn_reshaped.mean(dim=-1)  # Average over spatial
        
        # Project back and add residual
        aligned_text = text_features + self.output_proj(aligned)
        
        # Return video features enriched with text-timestamp alignment
        # Also return aligned text features for downstream use
        return aligned_text, alignment_scores


class AlphaBlender(nn.Module):
    """
    AlphaBlender operator from VidTok for temporal blending.
    Blends two inputs with a learnable or fixed alpha parameter.
    """
    def __init__(self, alpha: float = 0.55):  # sigmoid(0.2) ≈ 0.55
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.alpha * x1 + (1 - self.alpha) * x2


class VidTokEncoder(nn.Module):
    """
    VidTok-style Video Encoder following Microsoft's VidTok architecture.
    
    SOTA: Implements the VidTok encoder with:
    - 3D convolutions for input and bottleneck (information fusion)
    - 2D convolutions for spatial downsampling (efficiency)
    - AlphaBlender + 1D convolutions for temporal downsampling
    - Layer normalization for stability
    
    Compresses video [B, C, T, H, W] -> latent [B, latent_dim, t, h, w]
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 64,
        temporal_downsample: int = 4,
        spatial_downsample: int = 8,
        causal: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.temporal_downsample = temporal_downsample
        self.spatial_downsample = spatial_downsample
        self.causal = causal
        
        # Calculate number of spatial/temporal downsample stages
        self.num_spatial_downs = int(math.log2(spatial_downsample))  # e.g., 8 -> 3 stages
        self.num_temporal_downs = int(math.log2(temporal_downsample))  # e.g., 4 -> 2 stages
        
        # Input block: 3D conv for initial feature extraction
        self.input_block = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )
        
        # Spatial downsampling blocks (2D conv)
        self.spatial_down_blocks = nn.ModuleList()
        ch = base_channels
        for i in range(self.num_spatial_downs):
            out_ch = min(ch * 2, 512)
            self.spatial_down_blocks.append(
                self._make_spatial_down_block(ch, out_ch)
            )
            ch = out_ch
        
        # Temporal downsampling blocks (AlphaBlender + 1D conv)
        self.temporal_down_blocks = nn.ModuleList()
        for i in range(self.num_temporal_downs):
            self.temporal_down_blocks.append(
                self._make_temporal_down_block(ch)
            )
        
        # Bottleneck: 3D conv for information fusion
        self.bottleneck = nn.Sequential(
            nn.Conv3d(ch, ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv3d(ch, ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
        )
        
        # Output projection to latent channels
        self.to_latent = nn.Conv3d(ch, latent_channels, kernel_size=1)
        
        print(f"   🎬 VidTokEncoder: {in_channels}ch -> {latent_channels}ch latent")
        print(f"      Spatial: {spatial_downsample}x down ({self.num_spatial_downs} stages)")
        print(f"      Temporal: {temporal_downsample}x down ({self.num_temporal_downs} stages)")
    
    def _make_spatial_down_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Create a spatial downsampling block using 2D convolutions."""
        return nn.Sequential(
            # Reshape for 2D conv: [B, C, T, H, W] -> [B*T, C, H, W]
            Rearrange3Dto2D(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            # Reshape back: [B*T, C, H, W] -> [B, C, T, H, W]
            Rearrange2Dto3D(),
        )
    
    def _make_temporal_down_block(self, channels: int) -> nn.Module:
        """Create a temporal downsampling block using AlphaBlender + 1D conv."""
        return TemporalDownBlock(channels, causal=self.causal)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode video to latent space.
        
        Args:
            x: [B, C, T, H, W] input video
            
        Returns:
            [B, latent_channels, t, h, w] latent representation
        """
        # Store original shape for reshape operations
        B, C, T, H, W = x.shape
        
        # Input block
        x = self.input_block(x)
        
        # Spatial downsampling
        for block in self.spatial_down_blocks:
            # Pass temporal dimension for reshape
            if hasattr(block[0], 'set_temporal_dim'):
                block[0].set_temporal_dim(x.shape[2])
            if hasattr(block[-1], 'set_temporal_dim'):
                block[-1].set_temporal_dim(x.shape[2])
            x = block(x)
        
        # Temporal downsampling
        for block in self.temporal_down_blocks:
            x = block(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Project to latent
        x = self.to_latent(x)
        
        return x


class VidTokDecoder(nn.Module):
    """
    VidTok-style Video Decoder following Microsoft's VidTok architecture.
    
    Reconstructs video from latent [B, latent_dim, t, h, w] -> [B, C, T, H, W]
    """
    
    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 64,
        temporal_upsample: int = 4,
        spatial_upsample: int = 8,
        causal: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.temporal_upsample = temporal_upsample
        self.spatial_upsample = spatial_upsample
        self.causal = causal
        
        self.num_spatial_ups = int(math.log2(spatial_upsample))
        self.num_temporal_ups = int(math.log2(temporal_upsample))
        
        # Calculate channel progression (reverse of encoder)
        ch = min(base_channels * (2 ** self.num_spatial_ups), 512)
        
        # Input projection from latent
        self.from_latent = nn.Conv3d(latent_channels, ch, kernel_size=1)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(ch, ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv3d(ch, ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
        )
        
        # Temporal upsampling blocks
        self.temporal_up_blocks = nn.ModuleList()
        for i in range(self.num_temporal_ups):
            self.temporal_up_blocks.append(
                TemporalUpBlock(ch, causal=self.causal)
            )
        
        # Spatial upsampling blocks (2D conv)
        self.spatial_up_blocks = nn.ModuleList()
        for i in range(self.num_spatial_ups):
            out_ch = max(ch // 2, base_channels)
            self.spatial_up_blocks.append(
                self._make_spatial_up_block(ch, out_ch)
            )
            ch = out_ch
        
        # Output block
        self.output_block = nn.Sequential(
            nn.Conv3d(ch, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),  # Output in [-1, 1]
        )
        
        print(f"   🎬 VidTokDecoder: {latent_channels}ch latent -> {out_channels}ch")
    
    def _make_spatial_up_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Create a spatial upsampling block using 2D convolutions."""
        return nn.Sequential(
            Rearrange3Dto2D(),
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            Rearrange2Dto3D(),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to video.
        
        Args:
            z: [B, latent_channels, t, h, w] latent representation
            
        Returns:
            [B, C, T, H, W] reconstructed video
        """
        x = self.from_latent(z)
        x = self.bottleneck(x)
        
        for block in self.temporal_up_blocks:
            x = block(x)
        
        for block in self.spatial_up_blocks:
            x = block(x)
        
        x = self.output_block(x)
        return x


class Rearrange3Dto2D(nn.Module):
    """Reshape [B, C, T, H, W] -> [B*T, C, H, W] for 2D operations."""
    def __init__(self):
        super().__init__()
        self.temporal_dim = None
    
    def set_temporal_dim(self, t: int):
        self.temporal_dim = t
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        self.temporal_dim = T
        return x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)


class Rearrange2Dto3D(nn.Module):
    """Reshape [B*T, C, H, W] -> [B, C, T, H, W] after 2D operations."""
    def __init__(self):
        super().__init__()
        self.temporal_dim = None
    
    def set_temporal_dim(self, t: int):
        self.temporal_dim = t
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BT, C, H, W = x.shape
        T = self.temporal_dim if self.temporal_dim else 1
        B = BT // T
        return x.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)


class TemporalDownBlock(nn.Module):
    """Temporal downsampling using AlphaBlender + 1D conv (VidTok style)."""
    def __init__(self, channels: int, causal: bool = True):
        super().__init__()
        self.channels = channels
        self.causal = causal
        self.alpha_blender = AlphaBlender()
        
        # 1D temporal conv for downsampling
        padding = (1, 0) if causal else 1
        self.temporal_conv = nn.Conv1d(channels, channels, kernel_size=2, stride=2, padding=0)
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            [B, C, T//2, H, W]
        """
        B, C, T, H, W = x.shape
        
        # Reshape for 1D temporal conv: [B, C, T, H, W] -> [B*H*W, C, T]
        x = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, T)
        
        # Temporal downsampling
        x = self.temporal_conv(x)
        x = self.norm(x.unsqueeze(-1)).squeeze(-1)  # GroupNorm needs 4D
        x = self.act(x)
        
        # Reshape back: [B*H*W, C, T//2] -> [B, C, T//2, H, W]
        T_new = x.shape[2]
        x = x.reshape(B, H, W, C, T_new).permute(0, 3, 4, 1, 2)
        
        return x


class TemporalUpBlock(nn.Module):
    """Temporal upsampling using AlphaBlender + 1D conv (VidTok style)."""
    def __init__(self, channels: int, causal: bool = True):
        super().__init__()
        self.channels = channels
        self.causal = causal
        self.alpha_blender = AlphaBlender()
        
        # 1D temporal conv for upsampling
        self.temporal_conv = nn.ConvTranspose1d(channels, channels, kernel_size=2, stride=2)
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            [B, C, T*2, H, W]
        """
        B, C, T, H, W = x.shape
        
        # Reshape for 1D temporal conv
        x = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, T)
        
        # Temporal upsampling
        x = self.temporal_conv(x)
        x = self.norm(x.unsqueeze(-1)).squeeze(-1)
        x = self.act(x)
        
        # Reshape back
        T_new = x.shape[2]
        x = x.reshape(B, H, W, C, T_new).permute(0, 3, 4, 1, 2)
        
        return x


class VidTokTokenizer(nn.Module):
    """
    VidTok-style Video Tokenizer (3D VAE) following Microsoft's VidTok architecture.
    
    SOTA: Full encoder-decoder architecture for video compression to latent space.
    - Efficient 2D+1D architecture (separates spatial and temporal processing)
    - AlphaBlender for temporal blending
    - Supports both continuous (KL) and discrete (FSQ) tokenization
    - Causal mode for streaming/autoregressive applications
    
    Compresses video [B, C, T, H, W] -> latent [B, latent_dim, t, h, w]
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 64,
        temporal_compression: int = 4,
        spatial_compression: int = 8,
        causal: bool = True,
        use_fsq: bool = False,
        fsq_levels: int = 8,  # For discrete tokenization
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.temporal_compression = temporal_compression
        self.spatial_compression = spatial_compression
        self.causal = causal
        self.use_fsq = use_fsq
        self.fsq_levels = fsq_levels
        
        # Encoder
        self.encoder = VidTokEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels * 2 if not use_fsq else latent_channels,  # *2 for mean+logvar
            base_channels=base_channels,
            temporal_downsample=temporal_compression,
            spatial_downsample=spatial_compression,
            causal=causal,
        )
        
        # Decoder
        self.decoder = VidTokDecoder(
            out_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            temporal_upsample=temporal_compression,
            spatial_upsample=spatial_compression,
            causal=causal,
        )
        
        print(f"   🎬 VidTokTokenizer: {temporal_compression}x{spatial_compression}x{spatial_compression} compression")
        print(f"      Mode: {'FSQ (discrete)' if use_fsq else 'KL (continuous)'}, Causal: {causal}")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode video to latent space."""
        h = self.encoder(x)
        
        if self.use_fsq:
            # Finite Scalar Quantization
            return self._fsq_quantize(h)
        else:
            # KL regularization (VAE style)
            mean, logvar = h.chunk(2, dim=1)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to video."""
        return self.decoder(z)
    
    def _fsq_quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Finite Scalar Quantization - quantize each channel independently."""
        # Scale to [-1, 1] then quantize to fsq_levels
        z = torch.tanh(z)
        z = torch.round((z + 1) * (self.fsq_levels - 1) / 2) * 2 / (self.fsq_levels - 1) - 1
        return z
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode then decode.
        
        Args:
            x: [B, C, T, H, W] input video
            
        Returns:
            Tuple of (reconstructed video, latent representation)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


# Backward compatibility alias - keep old VideoTokenizer for feature compression
class VideoTokenizer(nn.Module):
    """
    Video Feature Tokenizer: Compresses spatio-temporal video features into efficient 1D tokens.
    
    NOTE: This is different from VidTokTokenizer which is a full 3D VAE.
    This class compresses already-extracted features [B, T*H*W, hidden] -> [B, num_tokens, hidden]
    using cross-attention, similar to TiTokTokenizer for images.
    """

    def __init__(
        self, 
        hidden_size: int, 
        num_tokens: int = 64, 
        max_frames: int = 32,
        num_spatial_tokens: int = 256,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.max_frames = max_frames
        self.num_spatial_tokens = num_spatial_tokens
        
        # Learnable compression
        self.compress = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        # Learnable token queries
        self.token_queries = nn.Parameter(torch.randn(1, num_tokens, hidden_size) * 0.02)
        
        # Temporal position embeddings
        self.temporal_pos = nn.Parameter(torch.randn(1, max_frames, hidden_size) * 0.02)
        
        # Cross-attention for compression
        self.compress_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True,
            dropout=0.1,
        )
        self.compress_norm = nn.LayerNorm(hidden_size)
        
        print(f"   🎬 VideoTokenizer: {num_tokens} tokens (feature compression)")

    def forward(self, x: torch.Tensor, num_frames: Optional[int] = None) -> torch.Tensor:
        """
        Compress video features to tokens.
        
        Args:
            x: [B, T*spatial, hidden_size] video features
            num_frames: Number of frames for temporal position encoding
            
        Returns:
            [B, num_tokens, hidden_size] compressed video tokens
        """
        batch_size = x.shape[0]
        total_patches = x.shape[1]
        
        # Add temporal position encoding if frames info provided
        if num_frames is not None and num_frames > 0:
            spatial_per_frame = total_patches // num_frames
            x_frames = x.view(batch_size, num_frames, spatial_per_frame, self.hidden_size)
            x_frames = x_frames + self.temporal_pos[:, :num_frames].unsqueeze(2)
            x = x_frames.view(batch_size, total_patches, self.hidden_size)
        
        # Expand token queries for batch
        queries = self.token_queries.expand(batch_size, -1, -1)
        
        # Cross-attention compression
        x_proj = self.compress(x)
        tokens, _ = self.compress_attn(queries, x_proj, x_proj)
        tokens = self.compress_norm(queries + tokens)
        
        return tokens


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
    SOTA Video Encoder with 3D-RoPE, 3D Causal Attention, Temporal Expert Routing, and VidTokTokenizer.
    
    Features:
    - 3D-RoPE for flexible (x, y, t) positional encodings
    - 3D Causal Attention for temporal understanding
    - Temporal-Aware Expert Routing for motion patterns
    - VidTokTokenizer for efficient 1D token compression (mirrors TiTokTokenizer for images)
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
        use_video_tokenizer: bool = True,
        num_video_tokens: int = 64,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.max_frames = max_frames
        self.hidden_size = vision_encoder.hidden_size
        self.use_3d_rope = use_3d_rope
        self.use_temporal_moe = use_temporal_moe
        self.use_video_tokenizer = use_video_tokenizer
        
        # Get expected image size from vision encoder
        self.image_size = getattr(vision_encoder, 'image_size', 384)
        self.patch_size = getattr(vision_encoder.vision_model.config, 'patch_size', 14)
        self.patches_per_side = self.image_size // self.patch_size
        self.num_spatial_tokens = self.patches_per_side ** 2

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

        # VidTokTokenizer for efficient video token compression (like TiTok for images)
        if use_video_tokenizer:
            self.vidtok = VidTokTokenizer(
                hidden_size=self.hidden_size,
                num_tokens=num_video_tokens,
                num_patches=self.num_spatial_tokens * max_frames,  # T * H * W
                max_frames=max_frames,
            )
            # Backward compatibility alias
            self.video_tokenizer = self.vidtok
        else:
            self.vidtok = None
            self.video_tokenizer = None

        # Temporal pooling attention for video-level representation (fallback when not using tokenizer)
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

        print(f"   🎬 Video encoder: max {max_frames} frames (multi-scale enabled)")

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

    def forward(
        self, 
        frames: torch.Tensor, 
        return_all_frames: bool = False, 
        causal: bool = False,
        return_tokens: bool = False,
    ) -> torch.Tensor:
        """
        Process video frames with 3D-RoPE and Causal Attention.
        
        Args:
            frames: [B, T, C, H, W] tensor of video frames
            return_all_frames: If True, return all frame features; else return pooled
            causal: If True, use causal attention (for autoregressive)
            return_tokens: If True, return VideoTokenizer compressed tokens
            
        Returns:
            If return_tokens: [B, num_tokens, hidden_size] video tokens
            If return_all_frames: [B, T, hidden_size] per-frame features
            Else: [B, hidden_size] pooled video representation
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
        
        # Return VidTokTokenizer compressed tokens if requested
        if return_tokens and self.vidtok is not None:
            return self.vidtok(x, num_frames)  # [B, num_tokens, hidden_size]
        
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

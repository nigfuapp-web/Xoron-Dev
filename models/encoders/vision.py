"""
SOTA Vision Encoder with 2D-RoPE, TiTok-style 1D Tokenization, and Dual-Stream Attention.

Features:
- SigLIP 2 / CLIP backbone for robust visual features
- 2D-RoPE for flexible aspect ratios (matches MoE-DiT generator)
- TiTok-style 1D tokenization for efficient representation
- Dual-stream attention integration for symmetric processing
- FP16-native numerical stability
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

EPS = 1e-5


class RoPE2DEncoder(nn.Module):
    """
    2D Rotary Position Embedding for vision encoder patches.
    Matches the 2D-RoPE in image generator for seamless integration.
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


def apply_rope_2d_encoder(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply 2D rotary position embedding to tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class TiTokTokenizer(nn.Module):
    """
    TiTok-style 1D Tokenizer for efficient visual representation.
    Converts 2D patch grid to 1D token sequence with learnable compression.
    """

    def __init__(self, hidden_size: int, num_tokens: int = 256, num_patches: int = 576):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.num_patches = num_patches
        
        # Learnable compression to 1D tokens
        self.compress = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        # Learnable token queries for compression
        self.token_queries = nn.Parameter(torch.randn(1, num_tokens, hidden_size) * 0.02)
        
        # Cross-attention for compression
        self.compress_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True,
            dropout=0.1,
        )
        self.compress_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress patch features to TiTok-style 1D tokens.
        
        Args:
            x: [B, num_patches, hidden_size] patch features
            
        Returns:
            [B, num_tokens, hidden_size] compressed token features
        """
        batch_size = x.shape[0]
        
        # Expand token queries for batch
        queries = self.token_queries.expand(batch_size, -1, -1)
        
        # Cross-attention compression
        x_proj = self.compress(x)
        tokens, _ = self.compress_attn(queries, x_proj, x_proj)
        tokens = self.compress_norm(queries + tokens)
        
        return tokens


class DeepStack(nn.Module):
    """
    DeepStack: Fuses multi-level ViT features to capture fine-grained details and sharpen image-text alignment.
    
    SOTA: Instead of using only the final layer features, DeepStack combines features from
    multiple intermediate layers of the vision encoder, enabling:
    - Better fine-grained detail capture (early layers have high-resolution features)
    - Stronger image-text alignment (different layers capture different semantic levels)
    - Improved generation quality for both understanding and generation tasks
    
    Architecture:
    - Collects features from selected layers (typically: early, middle, late)
    - Projects each level to a common dimension
    - Combines via learned weighted sum or attention
    """

    def __init__(self, hidden_size: int, num_layers: int = 3, use_attention: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Projection layers for each level
        self.level_projs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        
        # Layer normalization for each level
        self.level_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_layers)
        ])
        
        if use_attention:
            # Learnable queries for cross-attention fusion
            self.fusion_query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
            self.fusion_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                batch_first=True,
                dropout=0.1,
            )
            self.fusion_norm = nn.LayerNorm(hidden_size)
        else:
            # Learnable weights for weighted sum fusion
            self.level_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, multi_level_features: list) -> torch.Tensor:
        """
        Fuse multi-level features.
        
        Args:
            multi_level_features: List of [B, seq_len, hidden_size] features from different layers
            
        Returns:
            [B, seq_len, hidden_size] fused features
        """
        if len(multi_level_features) != self.num_layers:
            # Fallback: use only available layers
            multi_level_features = multi_level_features[-self.num_layers:] if len(multi_level_features) > self.num_layers else multi_level_features
        
        batch_size, seq_len, _ = multi_level_features[0].shape
        
        # Project and normalize each level
        projected = []
        for i, (feat, proj, norm) in enumerate(zip(multi_level_features, self.level_projs, self.level_norms)):
            projected.append(norm(proj(feat)))
        
        if self.use_attention:
            # Stack features for cross-attention: [B, num_layers * seq_len, hidden_size]
            stacked = torch.cat(projected, dim=1)
            
            # Expand fusion query
            query = self.fusion_query.expand(batch_size, seq_len, -1)
            
            # Cross-attention fusion
            fused, _ = self.fusion_attn(query, stacked, stacked)
            fused = self.fusion_norm(query + fused)
        else:
            # Weighted sum fusion
            weights = F.softmax(self.level_weights, dim=0)
            fused = sum(w * feat for w, feat in zip(weights, projected))
        
        # Final projection
        return self.output_proj(fused)


class DualStreamEncoderAttention(nn.Module):
    """
    Symmetric Dual-Stream Self-Attention for vision encoding.
    Matches the dual-stream architecture in image generator.
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
        
        self.rope_2d = RoPE2DEncoder(self.head_dim, max_height, max_width)

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x_a.shape
        
        x_a = self.norm_a(x_a)
        x_b = self.norm_b(x_b)
        
        qkv_a = self.to_qkv_a(x_a).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv_b = self.to_qkv_b(x_b).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        
        q_a, k_a, v_a = qkv_a.unbind(dim=2)
        q_b, k_b, v_b = qkv_b.unbind(dim=2)
        
        cos, sin = self.rope_2d(x_a, height, width)
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        
        q_a = q_a.transpose(1, 2)
        k_a = k_a.transpose(1, 2)
        v_a = v_a.transpose(1, 2)
        q_b = q_b.transpose(1, 2)
        k_b = k_b.transpose(1, 2)
        v_b = v_b.transpose(1, 2)
        
        q_a = apply_rope_2d_encoder(q_a, cos, sin)
        k_a = apply_rope_2d_encoder(k_a, cos, sin)
        q_b = apply_rope_2d_encoder(q_b, cos, sin)
        k_b = apply_rope_2d_encoder(k_b, cos, sin)
        
        # Dual-stream cross-attention
        k_combined = torch.cat([k_a, k_b], dim=2)
        v_combined = torch.cat([v_a, v_b], dim=2)
        
        attn_a = F.scaled_dot_product_attention(q_a, k_combined, v_combined)
        attn_b = F.scaled_dot_product_attention(q_b, k_combined, v_combined)
        
        attn_a = attn_a.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        attn_b = attn_b.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        
        out_a = self.to_out_a(attn_a)
        out_b = self.to_out_b(attn_b)
        
        return out_a, out_b


class VisionEncoderBlock(nn.Module):
    """Single block with dual-stream attention and FFN."""

    def __init__(self, hidden_size: int, num_heads: int = 8, ff_mult: int = 4, max_height: int = 64, max_width: int = 64):
        super().__init__()
        self.dual_attn = DualStreamEncoderAttention(hidden_size, num_heads, max_height, max_width)
        
        self.ffn_a = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * ff_mult),
            nn.GELU(),
            nn.Linear(hidden_size * ff_mult, hidden_size),
        )
        self.ffn_b = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * ff_mult),
            nn.GELU(),
            nn.Linear(hidden_size * ff_mult, hidden_size),
        )

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_a, attn_b = self.dual_attn(x_a, x_b, height, width)
        x_a = x_a + attn_a
        x_b = x_b + attn_b
        x_a = x_a + self.ffn_a(x_a)
        x_b = x_b + self.ffn_b(x_b)
        return x_a, x_b


class VisionEncoder(nn.Module):
    """
    SOTA Vision Encoder with 2D-RoPE, TiTok tokenization, and Dual-Stream Attention.
    
    Features:
    - SigLIP 2 / CLIP backbone for robust visual features
    - 2D-RoPE for flexible aspect ratios
    - TiTok-style 1D tokenization for efficient representation
    - Dual-stream attention for symmetric processing
    - FP16-native numerical stability
    """

    def __init__(
        self, 
        model_name: str = "google/siglip-so400m-patch14-384",
        freeze: bool = False,
        use_pooled_output: bool = False,
        use_dual_stream: bool = True,
        use_titok: bool = True,
        num_titok_tokens: int = 256,
        num_dual_stream_layers: int = 2,
    ):
        super().__init__()
        self.model_name = model_name
        self.use_pooled_output = use_pooled_output
        self.use_dual_stream = use_dual_stream
        self.use_titok = use_titok
        self._is_siglip = "siglip" in model_name.lower()

        print(f"\n👁️ Loading Vision Encoder: {model_name}")
        
        if self._is_siglip:
            self._init_siglip(model_name, freeze)
        else:
            self._init_clip(model_name, freeze)
        
        # 2D-RoPE for spatial position encoding (matches generator)
        self.rope_2d = RoPE2DEncoder(
            dim=self.hidden_size,
            max_height=64,
            max_width=64,
        )
        print(f"   📐 2D-RoPE: Flexible aspect ratio support")
        
        # Dual-stream attention layers for symmetric processing
        if use_dual_stream:
            patch_size = getattr(self.vision_model.config, 'patch_size', 14)
            image_size = getattr(self.vision_model.config, 'image_size', 384)
            max_patches = (image_size // patch_size)
            
            self.dual_stream_layers = nn.ModuleList([
                VisionEncoderBlock(
                    hidden_size=self.hidden_size,
                    num_heads=8,
                    ff_mult=4,
                    max_height=max_patches,
                    max_width=max_patches,
                )
                for _ in range(num_dual_stream_layers)
            ])
            print(f"   🔄 Dual-Stream: {num_dual_stream_layers} layers")
        else:
            self.dual_stream_layers = None
        
        # TiTok-style 1D tokenization
        if use_titok:
            self.titok = TiTokTokenizer(
                hidden_size=self.hidden_size,
                num_tokens=num_titok_tokens,
                num_patches=self.num_patches,
            )
            print(f"   🎫 TiTok: {self.num_patches} patches -> {num_titok_tokens} tokens")
        else:
            self.titok = None

    def _init_siglip(self, model_name: str, freeze: bool):
        """Initialize SigLIP 2 vision encoder."""
        try:
            from transformers import SiglipVisionModel, SiglipImageProcessor
            
            self.vision_model = SiglipVisionModel.from_pretrained(model_name)
            self.image_processor = SiglipImageProcessor.from_pretrained(model_name)
            self.hidden_size = self.vision_model.config.hidden_size
            
            print(f"   🎯 Using SigLIP 2 (recommended for MoE)")
            print(f"   ✅ Hidden size: {self.hidden_size}")
            print(f"   📐 Native size: {self.vision_model.config.image_size} (multi-scale: 256-512px)")
            print(f"   🔲 Patch size: {self.vision_model.config.patch_size}")
            
        except ImportError:
            print("   ⚠️ SigLIP not available, falling back to CLIP")
            self._is_siglip = False
            self._init_clip("openai/clip-vit-large-patch14", freeze)
            return
        
        if freeze:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            print(f"   ❄️ Vision encoder backbone frozen")
        else:
            print(f"   🔥 Vision encoder backbone trainable")

    def _init_clip(self, model_name: str, freeze: bool):
        """Initialize CLIP vision encoder (legacy support)."""
        from transformers import CLIPVisionModel, CLIPImageProcessor
        
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        self.hidden_size = self.vision_model.config.hidden_size
        
        print(f"   📎 Using CLIP")
        print(f"   ✅ Hidden size: {self.hidden_size}")

        if freeze:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            print(f"   ❄️ Vision encoder backbone frozen")
        else:
            print(f"   🔥 Vision encoder backbone trainable")

    def forward(self, pixel_values: torch.Tensor, return_titok: bool = None) -> torch.Tensor:
        """
        Extract vision features from images with SOTA enhancements.
        
        Args:
            pixel_values: [B, C, H, W] tensor of images
            return_titok: Override for TiTok output (None uses self.use_titok)
            
        Returns:
            [B, num_tokens, hidden_size] tensor (TiTok) or
            [B, num_patches, hidden_size] tensor (standard) or
            [B, hidden_size] if use_pooled_output=True
        """
        outputs = self.vision_model(pixel_values=pixel_values)
        features = outputs.last_hidden_state
        
        if self.use_pooled_output:
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                return outputs.pooler_output
            else:
                return features.mean(dim=1)
        
        # Get spatial dimensions
        batch_size, num_patches, hidden_size = features.shape
        patch_size = getattr(self.vision_model.config, 'patch_size', 14)
        image_size = getattr(self.vision_model.config, 'image_size', 384)
        
        # Account for CLS token if present
        if num_patches == (image_size // patch_size) ** 2 + 1:
            # Has CLS token, remove it for spatial processing
            cls_token = features[:, :1]
            features = features[:, 1:]
            num_patches = num_patches - 1
            has_cls = True
        else:
            cls_token = None
            has_cls = False
        
        height = width = int(math.sqrt(num_patches))
        
        # Apply dual-stream attention with 2D-RoPE
        if self.dual_stream_layers is not None:
            x_a = features
            x_b = features.clone()
            
            for layer in self.dual_stream_layers:
                x_a, x_b = layer(x_a, x_b, height, width)
            
            # Merge dual streams
            features = (x_a + x_b) / 2
        
        # Apply TiTok compression if enabled
        use_titok_now = return_titok if return_titok is not None else self.use_titok
        if use_titok_now and self.titok is not None:
            features = self.titok(features)
        
        return features

    def get_image_processor(self):
        """Return the image processor for preprocessing."""
        return self.image_processor
    
    @property
    def num_patches(self) -> int:
        """Get number of patches for the vision model."""
        config = self.vision_model.config
        image_size = config.image_size
        patch_size = config.patch_size
        return (image_size // patch_size) ** 2
    
    @property 
    def image_size(self) -> int:
        """Get expected image size."""
        return self.vision_model.config.image_size
    
    @property
    def output_tokens(self) -> int:
        """Get number of output tokens (considering TiTok compression)."""
        if self.use_titok and self.titok is not None:
            return self.titok.num_tokens
        return self.num_patches


# Convenience aliases for different SigLIP 2 variants
SIGLIP_MODELS = {
    # Base models
    "siglip-base": "google/siglip-base-patch16-224",
    "siglip-base-384": "google/siglip-base-patch16-384",
    
    # Large models (recommended)
    "siglip-large": "google/siglip-large-patch16-256",
    "siglip-large-384": "google/siglip-large-patch16-384",
    
    # SO400M models (best quality)
    "siglip-so400m": "google/siglip-so400m-patch14-384",
    "siglip-so400m-224": "google/siglip-so400m-patch14-224",
    
    # Legacy CLIP
    "clip-base": "openai/clip-vit-base-patch16",
    "clip-large": "openai/clip-vit-large-patch14",
}


def get_vision_encoder(
    model_key: str = "siglip-so400m",
    freeze: bool = False,
    use_dual_stream: bool = True,
    use_titok: bool = True,
    **kwargs
) -> VisionEncoder:
    """
    Get a vision encoder by key name with SOTA enhancements.
    
    Args:
        model_key: Key from SIGLIP_MODELS or full model name
        freeze: Whether to freeze encoder backbone weights
        use_dual_stream: Enable dual-stream attention
        use_titok: Enable TiTok 1D tokenization
        **kwargs: Additional arguments for VisionEncoder
        
    Returns:
        VisionEncoder instance
    """
    model_name = SIGLIP_MODELS.get(model_key, model_key)
    return VisionEncoder(
        model_name=model_name,
        freeze=freeze,
        use_dual_stream=use_dual_stream,
        use_titok=use_titok,
        **kwargs
    )

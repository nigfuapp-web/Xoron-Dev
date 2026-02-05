"""
SOTA Encoder modules for Xoron-Dev.

Vision Encoder:
- SigLIP 2 / CLIP backbone
- 2D-RoPE for flexible aspect ratios
- TiTok-style 1D tokenization
- Dual-stream attention

Video Encoder:
- 3D-RoPE for (x, y, t) positions
- 3D Causal Attention
- Temporal-Aware Expert Routing
- Integrates with video generator

Audio Encoder:
- Conformer-based architecture
- FastSpeech2/VITS-style decoder
- Multi-speaker and emotion support
"""

from models.encoders.vision import (
    VisionEncoder,
    RoPE2DEncoder,
    TiTokTokenizer,
    DualStreamEncoderAttention,
    get_vision_encoder,
    SIGLIP_MODELS,
)
from models.encoders.video import (
    VideoEncoder,
    RoPE3DEncoder,
    Causal3DAttentionEncoder,
    TemporalMoELayerEncoder,
    VideoEncoderBlock,
)
from models.encoders.audio import (
    AudioEncoder,
    AudioDecoder,
    MelSpectrogramExtractor,
    ConformerBlock,
)

__all__ = [
    # Vision
    'VisionEncoder',
    'RoPE2DEncoder',
    'TiTokTokenizer',
    'DualStreamEncoderAttention',
    'get_vision_encoder',
    'SIGLIP_MODELS',
    # Video
    'VideoEncoder',
    'RoPE3DEncoder',
    'Causal3DAttentionEncoder',
    'TemporalMoELayerEncoder',
    'VideoEncoderBlock',
    # Audio
    'AudioEncoder',
    'AudioDecoder',
    'MelSpectrogramExtractor',
    'ConformerBlock',
]

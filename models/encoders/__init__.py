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
- Raw Waveform Tokenizer (replaces mel spectrogram)
- Zero-Shot Speaker Cloning with speaker embedding extraction
- Monotonic Alignment Search (MAS) for fluid text-to-audio alignment
- Rotary Multi-Head Latent Attention (RMLA)
- In-Context Audio Prompting
- Conformer-based encoder
- FastSpeech2/VITS-style decoder with variance adaptor
- Multi-speaker and emotion support
"""

from models.encoders.vision import (
    VisionEncoder,
    RoPE2DEncoder,
    TiTokTokenizer,
    DualStreamEncoderAttention,
    VisionEncoderBlock,
    get_vision_encoder,
    SIGLIP_MODELS,
)
from models.encoders.video import (
    VideoEncoder,
    RoPE3DEncoder,
    Causal3DAttentionEncoder,
    TemporalMoELayerEncoder,
    TemporalExpertRouterEncoder,
    VideoExpertEncoder,
    VideoEncoderBlock,
)
from models.encoders.audio import (
    AudioEncoder,
    AudioDecoder,
    RawWaveformTokenizer,
    SpeakerEncoder,
    MonotonicAlignmentSearch,
    RotaryMultiHeadLatentAttention,
    InContextAudioPrompting,
    ConvolutionModule,
    ConformerBlock,
    VariancePredictor,
    FFTBlock,
)

__all__ = [
    # Vision
    'VisionEncoder',
    'RoPE2DEncoder',
    'TiTokTokenizer',
    'DualStreamEncoderAttention',
    'VisionEncoderBlock',
    'get_vision_encoder',
    'SIGLIP_MODELS',
    # Video
    'VideoEncoder',
    'RoPE3DEncoder',
    'Causal3DAttentionEncoder',
    'TemporalMoELayerEncoder',
    'TemporalExpertRouterEncoder',
    'VideoExpertEncoder',
    'VideoEncoderBlock',
    # Audio
    'AudioEncoder',
    'AudioDecoder',
    'RawWaveformTokenizer',
    'SpeakerEncoder',
    'MonotonicAlignmentSearch',
    'RotaryMultiHeadLatentAttention',
    'InContextAudioPrompting',
    'ConvolutionModule',
    'ConformerBlock',
    'VariancePredictor',
    'FFTBlock',
]

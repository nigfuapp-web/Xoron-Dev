"""
SOTA Encoder modules for Xoron-Dev.

Vision Encoder:
- SigLIP 2 / CLIP backbone
- 2D-RoPE for flexible aspect ratios
- TiTok-style 1D tokenization
- Dual-stream attention
- DeepStack for multi-level ViT feature fusion

Video Encoder:
- 3D-RoPE for (x, y, t) positions
- 3D Causal Attention
- Temporal-Aware Expert Routing
- Text-Timestamp Alignment for precise event localization
- Integrates with video generator

Audio Encoder/Decoder:
- Raw Waveform Tokenizer (replaces mel spectrogram for input)
- Raw Waveform Decoder (direct audio output, no vocoder needed)
- Speech-to-Speech capability (listen and talk back)
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
    DeepStack,
    DualStreamEncoderAttention,
    VisionEncoderBlock,
    get_vision_encoder,
    SIGLIP_MODELS,
)
from models.encoders.video import (
    VideoEncoder,
    RoPE3DEncoder,
    TextTimestampAlignment,
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
    RawWaveformDecoder,
    SnakeActivation,
    ResidualBlock1D,
    MultiReceptiveFieldFusion,
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
    'DeepStack',
    'DualStreamEncoderAttention',
    'VisionEncoderBlock',
    'get_vision_encoder',
    'SIGLIP_MODELS',
    # Video
    'VideoEncoder',
    'RoPE3DEncoder',
    'TextTimestampAlignment',
    'Causal3DAttentionEncoder',
    'TemporalMoELayerEncoder',
    'TemporalExpertRouterEncoder',
    'VideoExpertEncoder',
    'VideoEncoderBlock',
    # Audio
    'AudioEncoder',
    'AudioDecoder',
    'RawWaveformTokenizer',
    'RawWaveformDecoder',
    'SnakeActivation',
    'ResidualBlock1D',
    'MultiReceptiveFieldFusion',
    'SpeakerEncoder',
    'MonotonicAlignmentSearch',
    'RotaryMultiHeadLatentAttention',
    'InContextAudioPrompting',
    'ConvolutionModule',
    'ConformerBlock',
    'VariancePredictor',
    'FFTBlock',
]

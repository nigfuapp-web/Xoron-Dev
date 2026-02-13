"""Xoron model configuration with SOTA features."""

from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class XoronConfig:
    """
    Configuration for Xoron-Dev multimodal model.
    
    SOTA Features:
    - MLA (Multi-Head Latent Attention) for compressed KV cache
    - MoE with shared expert isolation (DeepSeek-style)
    - Ring Attention for distributed 128K+ context
    - YaRN/LongRoPE for superior long-context extrapolation
    - LoRA variants (rsLoRA, DoRA, LoRA+)
    - Perceiver Resampler for vision projection
    - Cross-attention for multimodal fusion
    - MoE-DiT with Flow Matching for image generation
    - 3D-RoPE + 3D Causal Transformers for video generation
    - TiTok-style 1D tokenization for vision encoding
    - VidTok-style 1D tokenization for video encoding (mirrors TiTok)
    - Dual-stream attention for symmetric processing
    - Conformer audio encoder/decoder
    - FP16-native numerical stability
    - Multi-scale training for variable resolution handling
    
    HuggingFace Integration:
    - model_type = "xoron" for AutoConfig/AutoModel support
    - Compatible with trust_remote_code=True loading
    """

    # HuggingFace model type (required for AutoClass integration)
    model_type: str = "xoron"
    
    # Model name
    model_name: str = "Xoron-Dev-MultiMoE"

    # LLM Architecture
    hidden_size: int = 1024
    num_layers: int = 12
    num_heads: int = 16
    intermediate_size: int = 2048
    vocab_size: int = 151643  # Qwen2.5 vocab size
    max_position_embeddings: int = 131072  # 128K context length
    rms_norm_eps: float = 1e-6
    
    # Ring Attention (efficient 128K+ context with FP16)
    use_ring_attention: bool = True
    ring_attention_chunk_size: int = 4096  # Ring attention chunk size
    
    # Tie word embeddings (parameter efficiency)
    tie_word_embeddings: bool = True

    # MoE Configuration (SOTA: Aux-Lossless with shared expert isolation)
    use_moe: bool = True
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_layer_freq: int = 2  # MoE every 2 layers
    use_shared_expert: bool = True  # DeepSeek-style shared expert isolation
    moe_capacity_factor: float = 1.25  # Expert capacity factor
    use_aux_lossless: bool = True  # Aux-lossless MoE routing (no aux loss needed)

    # Vision Configuration (SOTA: SigLIP 2 + TiTok + Dual-Stream)
    vision_model_name: str = "google/siglip-so400m-patch14-384"  # SigLIP 2 - best for MoE
    freeze_vision: bool = False
    num_vision_tokens: int = 64
    projector_type: str = "perceiver"  # "perceiver", "spatial", "c_abstractor", "mlp"
    
    # Vision Encoder SOTA Features
    use_vision_dual_stream: bool = True  # Symmetric dual-stream attention
    use_vision_titok: bool = True  # TiTok-style 1D tokenization
    num_vision_titok_tokens: int = 256  # TiTok compressed token count
    num_vision_dual_stream_layers: int = 2  # Dual-stream encoder layers
    
    # Video Encoder SOTA Features
    use_video_3d_rope: bool = True  # 3D-RoPE for (x,y,t) positions
    use_video_temporal_moe: bool = True  # Temporal-aware expert routing
    num_video_encoder_layers: int = 4  # 3D causal transformer layers
    num_video_experts: int = 4  # Temporal MoE experts
    use_video_vidtok: bool = True  # VidTok 3D VAE (Microsoft VidTok architecture)
    vidtok_latent_channels: int = 4  # VidTok latent channels
    vidtok_temporal_compression: int = 4  # VidTok temporal compression ratio
    vidtok_spatial_compression: int = 8  # VidTok spatial compression ratio
    vidtok_causal: bool = True  # VidTok causal mode for streaming
    vidtok_use_fsq: bool = False  # Use FSQ (discrete) vs KL (continuous)

    # ========== CONTINUOUS-SCALE TRAINING CONFIGURATION (SOTA) ==========
    # SOTA: Continuous-scale training replaces discrete scale lists with continuous sampling
    # This enables the model to train at ANY resolution within min/max bounds
    # Better generalization, more memory efficient, and handles OOM gracefully
    
    # Enable continuous-scale training (SOTA: samples ANY scale in range)
    use_multi_scale: bool = True
    use_continuous_scale: bool = True  # NEW: Enable continuous sampling (vs discrete)
    
    # Image continuous-scale settings
    image_min_size: int = 128   # Minimum image size
    image_max_size: int = 384   # Maximum image size (reduced for memory)
    image_base_size: int = 256  # Base/default image size
    image_size_step: int = 32   # Quantize to multiples of this (for VAE compatibility)
    
    # Video continuous-scale settings  
    video_min_size: int = 128   # Minimum video spatial size
    video_max_size: int = 320   # Maximum video spatial size (reduced for memory)
    video_base_size: int = 192  # Base/default video size (reduced for memory)
    video_size_step: int = 32   # Quantize to multiples of this
    
    # Video temporal continuous-scale settings
    video_min_frames: int = 8    # Minimum frame count
    video_max_frames: int = 24   # Maximum frame count (reduced for memory)
    video_base_frames: int = 16  # Base/default frame count
    video_frame_step: int = 4    # Quantize to multiples of this
    
    # Continuous-scale sampling strategy
    # "uniform" - uniform distribution across range
    # "gaussian" - Gaussian centered on base size (better quality)
    # "adaptive" - adapts based on OOM history (best for limited VRAM)
    multi_scale_strategy: str = "adaptive"
    multi_scale_warmup_epochs: int = 3  # Epochs before reaching full scale range
    
    # Adaptive scaling settings (for strategy="adaptive")
    adaptive_scale_oom_penalty: float = 0.5  # Reduce max scale by this factor on OOM
    adaptive_scale_success_boost: float = 0.1  # Increase max scale on success
    
    # Supported sizes for generation inference
    generation_supported_sizes: Tuple[int, ...] = (192, 256, 320, 384)
    generation_supported_frames: Tuple[int, ...] = (8, 12, 16, 20, 24)
    # ================================================================

    # Image Generation Configuration (SOTA: MoE-DiT with Flow Matching + 2D-RoPE)
    enable_generation: bool = True
    generation_latent_channels: int = 4
    generation_base_channels: int = 128
    generation_inference_steps: int = 50  # Flow Matching needs more steps
    generation_cfg_scale: float = 7.5  # Classifier-free guidance scale
    generation_use_flow_matching: bool = True  # Use Flow Matching instead of DDPM
    generation_num_experts: int = 4  # MoE experts in DiT
    generation_use_dual_stream: bool = True  # Symmetric dual-stream attention
    
    # Video Generation Configuration (SOTA: 3D Causal Transformers + Flow Matching + 3D-RoPE)
    generation_video_cfg_scale: float = 7.5
    generation_video_use_flow_matching: bool = True
    generation_video_num_experts: int = 4
    generation_video_use_3d_rope: bool = True  # 3D-RoPE for (x,y,t)
    generation_video_use_temporal_moe: bool = True  # Temporal-aware expert routing

    # Audio Configuration (SOTA: Raw Waveform Tokenizer, MAS, RMLA, Zero-Shot Cloning)
    audio_sample_rate: int = 16000
    audio_n_mels: int = 80
    audio_max_length: int = 1000  # Max audio sequence length (frames)
    audio_num_speakers: int = 256  # Speaker embedding size
    use_raw_waveform: bool = True  # Use raw waveform tokenizer instead of mel spectrogram
    audio_kv_lora_rank: int = 256  # KV compression rank for RMLA
    audio_speaker_embed_dim: int = 256  # Speaker embedding dimension for zero-shot cloning
    use_mas: bool = True  # Use Monotonic Alignment Search for text-to-audio alignment
    use_in_context_audio_prompting: bool = True  # Enable in-context audio prompting for voice cloning

    # Tokenizer Configuration
    tokenizer_name: str = "Qwen/Qwen2.5-1.5B"

    # LoRA Configuration (SOTA: rsLoRA, DoRA, LoRA+)
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
    )
    train_lora_only: bool = False
    use_rslora: bool = True  # Rank-stabilized LoRA scaling
    use_dora: bool = False  # Weight-Decomposed LoRA (optional)
    lora_plus_lr_ratio: float = 4.0  # LoRA+ B matrix learns faster (use training_config value)

    # Cross-Attention Configuration
    use_cross_attention: bool = True
    cross_attention_layers: int = 4
    cross_attention_heads: int = 8
    cross_attention_dropout: float = 0.1

    # Flash Attention Configuration
    use_flash_attention: bool = True

    # Output path
    output_dir: str = "./xoron-model"

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
        assert self.num_experts_per_tok <= self.num_experts, "num_experts_per_tok must be <= num_experts"

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'intermediate_size': self.intermediate_size,
            'vocab_size': self.vocab_size,
            'max_position_embeddings': self.max_position_embeddings,
            'rms_norm_eps': self.rms_norm_eps,
            'use_ring_attention': self.use_ring_attention,
            'ring_attention_chunk_size': self.ring_attention_chunk_size,
            'tie_word_embeddings': self.tie_word_embeddings,
            'use_moe': self.use_moe,
            'num_experts': self.num_experts,
            'num_experts_per_tok': self.num_experts_per_tok,
            'moe_layer_freq': self.moe_layer_freq,
            'use_shared_expert': self.use_shared_expert,
            'moe_capacity_factor': self.moe_capacity_factor,
            'use_aux_lossless': self.use_aux_lossless,
            'vision_model_name': self.vision_model_name,
            'freeze_vision': self.freeze_vision,
            'num_vision_tokens': self.num_vision_tokens,
            'projector_type': self.projector_type,
            'use_vision_dual_stream': self.use_vision_dual_stream,
            'use_vision_titok': self.use_vision_titok,
            'num_vision_titok_tokens': self.num_vision_titok_tokens,
            'num_vision_dual_stream_layers': self.num_vision_dual_stream_layers,
            'use_video_3d_rope': self.use_video_3d_rope,
            'use_video_temporal_moe': self.use_video_temporal_moe,
            'num_video_encoder_layers': self.num_video_encoder_layers,
            'num_video_experts': self.num_video_experts,
            'use_video_vidtok': self.use_video_vidtok,
            'vidtok_latent_channels': self.vidtok_latent_channels,
            'vidtok_temporal_compression': self.vidtok_temporal_compression,
            'vidtok_spatial_compression': self.vidtok_spatial_compression,
            'vidtok_causal': self.vidtok_causal,
            'vidtok_use_fsq': self.vidtok_use_fsq,
            # Multi-scale configuration (source of truth for all sizes/frames)
            'use_multi_scale': self.use_multi_scale,
            'use_continuous_scale': self.use_continuous_scale,
            'image_min_size': self.image_min_size,
            'image_max_size': self.image_max_size,
            'image_base_size': self.image_base_size,
            'image_size_step': self.image_size_step,
            'video_min_size': self.video_min_size,
            'video_max_size': self.video_max_size,
            'video_base_size': self.video_base_size,
            'video_size_step': self.video_size_step,
            'video_min_frames': self.video_min_frames,
            'video_max_frames': self.video_max_frames,
            'video_base_frames': self.video_base_frames,
            'video_frame_step': self.video_frame_step,
            'multi_scale_strategy': self.multi_scale_strategy,
            'multi_scale_warmup_epochs': self.multi_scale_warmup_epochs,
            'adaptive_scale_oom_penalty': self.adaptive_scale_oom_penalty,
            'adaptive_scale_success_boost': self.adaptive_scale_success_boost,
            'generation_supported_sizes': list(self.generation_supported_sizes),
            'generation_supported_frames': list(self.generation_supported_frames),
            # Generation configs
            'enable_generation': self.enable_generation,
            'generation_latent_channels': self.generation_latent_channels,
            'generation_base_channels': self.generation_base_channels,
            'generation_inference_steps': self.generation_inference_steps,
            'generation_cfg_scale': self.generation_cfg_scale,
            'generation_use_flow_matching': self.generation_use_flow_matching,
            'generation_num_experts': self.generation_num_experts,
            'generation_use_dual_stream': self.generation_use_dual_stream,
            'generation_video_cfg_scale': self.generation_video_cfg_scale,
            'generation_video_use_flow_matching': self.generation_video_use_flow_matching,
            'generation_video_num_experts': self.generation_video_num_experts,
            'generation_video_use_3d_rope': self.generation_video_use_3d_rope,
            'generation_video_use_temporal_moe': self.generation_video_use_temporal_moe,
            # Audio configs
            'audio_sample_rate': self.audio_sample_rate,
            'audio_n_mels': self.audio_n_mels,
            'audio_max_length': self.audio_max_length,
            'audio_num_speakers': self.audio_num_speakers,
            'use_raw_waveform': self.use_raw_waveform,
            'audio_kv_lora_rank': self.audio_kv_lora_rank,
            'audio_speaker_embed_dim': self.audio_speaker_embed_dim,
            'use_mas': self.use_mas,
            'use_in_context_audio_prompting': self.use_in_context_audio_prompting,
            'tokenizer_name': self.tokenizer_name,
            'use_lora': self.use_lora,
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'lora_target_modules': list(self.lora_target_modules),
            'train_lora_only': self.train_lora_only,
            'use_rslora': self.use_rslora,
            'use_dora': self.use_dora,
            'lora_plus_lr_ratio': self.lora_plus_lr_ratio,
            'use_cross_attention': self.use_cross_attention,
            'cross_attention_layers': self.cross_attention_layers,
            'cross_attention_heads': self.cross_attention_heads,
            'cross_attention_dropout': self.cross_attention_dropout,
            'use_flash_attention': self.use_flash_attention,
            'output_dir': self.output_dir,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'XoronConfig':
        """Create config from dictionary."""
        # Convert lists back to tuples for tuple fields
        if 'lora_target_modules' in config_dict and isinstance(config_dict['lora_target_modules'], list):
            config_dict['lora_target_modules'] = tuple(config_dict['lora_target_modules'])
        if 'generation_supported_sizes' in config_dict and isinstance(config_dict['generation_supported_sizes'], list):
            config_dict['generation_supported_sizes'] = tuple(config_dict['generation_supported_sizes'])
        if 'generation_supported_frames' in config_dict and isinstance(config_dict['generation_supported_frames'], list):
            config_dict['generation_supported_frames'] = tuple(config_dict['generation_supported_frames'])
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k) or k in cls.__dataclass_fields__})

    def print_config(self):
        """Print configuration summary."""
        print("=" * 60)
        print("XORON-DEV MODEL CONFIGURATION (SOTA)")
        print("=" * 60)
        print(f"\n🧠 LLM: {self.hidden_size}d, {self.num_layers}L, {self.num_heads}H")
        attn_type = "Ring Attention" if self.use_ring_attention else "Standard"
        print(f"📏 Context: {self.max_position_embeddings//1024}K positions, {attn_type} (chunk={self.ring_attention_chunk_size})")
        print(f"🎯 MoE: {self.num_experts} experts, top-{self.num_experts_per_tok}, shared_expert={self.use_shared_expert}, aux_lossless={self.use_aux_lossless}")
        print(f"👁️ Vision: {self.vision_model_name}")
        print(f"   - TiTok: {self.use_vision_titok} ({self.num_vision_titok_tokens} tokens)")
        print(f"   - Dual-Stream: {self.use_vision_dual_stream} ({self.num_vision_dual_stream_layers} layers)")
        print(f"🎬 Video Encoder: 3D-RoPE={self.use_video_3d_rope}, Temporal MoE={self.use_video_temporal_moe}")
        print(f"   - VidTok: {self.use_video_vidtok} ({self.vidtok_temporal_compression}x{self.vidtok_spatial_compression}x{self.vidtok_spatial_compression} compression)")
        print(f"   - VidTok Mode: {'FSQ (discrete)' if self.vidtok_use_fsq else 'KL (continuous)'}, Causal: {self.vidtok_causal}")
        # Multi-scale info
        if self.use_multi_scale and self.use_continuous_scale:
            print(f"📐 Continuous-Scale Training: ENABLED (strategy={self.multi_scale_strategy})")
            print(f"   - Image: {self.image_min_size}-{self.image_max_size}px (step={self.image_size_step}), base={self.image_base_size}")
            print(f"   - Video: {self.video_min_size}-{self.video_max_size}px (step={self.video_size_step}), base={self.video_base_size}")
            print(f"   - Frames: {self.video_min_frames}-{self.video_max_frames} (step={self.video_frame_step}), base={self.video_base_frames}")
            print(f"🎨 Image Gen: {self.image_min_size}-{self.image_max_size}px, Flow={self.generation_use_flow_matching}, Dual-Stream={self.generation_use_dual_stream}")
            print(f"🎬 Video Gen: {self.video_min_frames}-{self.video_max_frames} frames @ {self.video_min_size}-{self.video_max_size}px, 3D-RoPE={self.generation_video_use_3d_rope}")
        else:
            print(f"📐 Multi-Scale: DISABLED (fixed {self.image_base_size}x{self.image_base_size})")
            print(f"🎨 Image Gen: {self.image_base_size}x{self.image_base_size}, Flow={self.generation_use_flow_matching}, Dual-Stream={self.generation_use_dual_stream}")
            print(f"🎬 Video Gen: {self.video_base_frames} frames @ {self.video_base_size}, 3D-RoPE={self.generation_video_use_3d_rope}")
        print(f"🎤 Audio: {self.audio_sample_rate}Hz, RawWaveform={self.use_raw_waveform}, MAS={self.use_mas}")
        print(f"   - Zero-Shot Cloning: speaker_dim={self.audio_speaker_embed_dim}, In-Context Prompting={self.use_in_context_audio_prompting}")
        print(f"📝 Tokenizer: {self.tokenizer_name} (vocab: {self.vocab_size:,})")
        lora_type = "DoRA" if self.use_dora else ("rsLoRA" if self.use_rslora else "LoRA")
        print(f"🔧 {lora_type}: r={self.lora_r}, alpha={self.lora_alpha}, LoRA+ ratio={self.lora_plus_lr_ratio}")
        print(f"🔀 Cross-Attention: {self.cross_attention_layers} layers")
        print(f"⚡ Flash Attention: {self.use_flash_attention}")
        print(f"📁 Output: {self.output_dir}")
        print("=" * 60)

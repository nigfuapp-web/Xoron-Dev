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
    - Dual-stream attention for symmetric processing
    - Conformer audio encoder/decoder
    - FP16-native numerical stability
    - Multi-scale training for variable resolution handling
    """

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

    # ========== MULTI-SCALE TRAINING CONFIGURATION (SOTA) ==========
    # Multi-scale enables training and inference at multiple resolutions
    # This allows the model to handle various input/output sizes dynamically
    # ALL image/video size and frame settings are consolidated here
    
    # Enable multi-scale training (dynamic resolution selection during training)
    use_multi_scale: bool = True
    
    # Image multi-scale settings
    # Available image resolutions for training (height, width)
    image_scales: Tuple[Tuple[int, int], ...] = (
        (128, 128),   # Low res - fast training, good for low-memory
        (192, 192),   # Medium-low res
        (256, 256),   # Base resolution (default)
        (320, 320),   # Medium-high res
        (384, 384),   # High res - matches SigLIP native resolution
        (448, 448),   # Very high res
        (512, 512),   # Max resolution for high quality
    )
    # Probability distribution for scale sampling (sum should be ~1.0)
    # Higher probability for middle scales, lower for extremes
    image_scale_probs: Tuple[float, ...] = (0.05, 0.10, 0.30, 0.25, 0.15, 0.10, 0.05)
    image_min_size: int = 128   # Minimum image size
    image_max_size: int = 512   # Maximum image size
    image_base_size: int = 256  # Base/default image size (used when multi-scale disabled)
    
    # Video spatial multi-scale settings
    # Available video resolutions (height, width)
    video_scales: Tuple[Tuple[int, int], ...] = (
        (128, 128),   # Low res - fast
        (192, 192),   # Medium-low
        (256, 256),   # Base resolution
        (320, 320),   # Medium-high
        (384, 384),   # High res
    )
    video_scale_probs: Tuple[float, ...] = (0.10, 0.20, 0.35, 0.25, 0.10)
    video_min_size: int = 128   # Minimum video spatial size
    video_max_size: int = 384   # Maximum video spatial size  
    video_base_size: int = 256  # Base/default video size (used when multi-scale disabled)
    
    # Video temporal multi-scale settings (frame counts)
    # Available frame counts for training - supports 8 to 32 frames
    video_frame_scales: Tuple[int, ...] = (8, 12, 16, 20, 24, 32)
    video_frame_scale_probs: Tuple[float, ...] = (0.10, 0.15, 0.30, 0.20, 0.15, 0.10)
    video_min_frames: int = 8    # Minimum frame count
    video_max_frames: int = 32   # Maximum frame count (supports 16+ frames)
    video_base_frames: int = 16  # Base/default frame count
    
    # Multi-scale training strategy
    # "random" - randomly sample scale each batch (best for variety - each sample gets different size)
    # "progressive" - start small, gradually increase scale during training (epoch-based)
    # "curriculum" - alternate between scales in a curriculum
    multi_scale_strategy: str = "random"
    multi_scale_warmup_epochs: int = 5  # For progressive strategy: epochs to reach max scale
    
    # Supported sizes for generation inference
    generation_supported_sizes: Tuple[int, ...] = (256, 320, 384, 448, 512)
    generation_supported_frames: Tuple[int, ...] = (8, 12, 16, 20, 24, 32)
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
            # Multi-scale configuration (source of truth for all sizes/frames)
            'use_multi_scale': self.use_multi_scale,
            'image_scales': list(self.image_scales),
            'image_scale_probs': list(self.image_scale_probs),
            'image_min_size': self.image_min_size,
            'image_max_size': self.image_max_size,
            'image_base_size': self.image_base_size,
            'video_scales': list(self.video_scales),
            'video_scale_probs': list(self.video_scale_probs),
            'video_min_size': self.video_min_size,
            'video_max_size': self.video_max_size,
            'video_base_size': self.video_base_size,
            'video_frame_scales': list(self.video_frame_scales),
            'video_frame_scale_probs': list(self.video_frame_scale_probs),
            'video_min_frames': self.video_min_frames,
            'video_max_frames': self.video_max_frames,
            'video_base_frames': self.video_base_frames,
            'multi_scale_strategy': self.multi_scale_strategy,
            'multi_scale_warmup_epochs': self.multi_scale_warmup_epochs,
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
        if 'image_scales' in config_dict and isinstance(config_dict['image_scales'], list):
            config_dict['image_scales'] = tuple(tuple(s) if isinstance(s, list) else s for s in config_dict['image_scales'])
        if 'image_scale_probs' in config_dict and isinstance(config_dict['image_scale_probs'], list):
            config_dict['image_scale_probs'] = tuple(config_dict['image_scale_probs'])
        if 'video_scales' in config_dict and isinstance(config_dict['video_scales'], list):
            config_dict['video_scales'] = tuple(tuple(s) if isinstance(s, list) else s for s in config_dict['video_scales'])
        if 'video_scale_probs' in config_dict and isinstance(config_dict['video_scale_probs'], list):
            config_dict['video_scale_probs'] = tuple(config_dict['video_scale_probs'])
        if 'video_frame_scales' in config_dict and isinstance(config_dict['video_frame_scales'], list):
            config_dict['video_frame_scales'] = tuple(config_dict['video_frame_scales'])
        if 'video_frame_scale_probs' in config_dict and isinstance(config_dict['video_frame_scale_probs'], list):
            config_dict['video_frame_scale_probs'] = tuple(config_dict['video_frame_scale_probs'])
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
        # Multi-scale info
        if self.use_multi_scale:
            img_sizes = [f"{s[0]}x{s[1]}" for s in self.image_scales]
            vid_sizes = [f"{s[0]}x{s[1]}" for s in self.video_scales]
            print(f"📐 Multi-Scale Training: ENABLED (strategy={self.multi_scale_strategy})")
            print(f"   - Image: {self.image_min_size}-{self.image_max_size}px, base={self.image_base_size}")
            print(f"   - Video: {self.video_min_size}-{self.video_max_size}px, base={self.video_base_size}")
            print(f"   - Frames: {self.video_min_frames}-{self.video_max_frames}, base={self.video_base_frames}")
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

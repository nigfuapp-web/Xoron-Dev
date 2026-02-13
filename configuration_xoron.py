"""
Xoron Model Configuration for HuggingFace Transformers.

This module provides a HuggingFace-compatible configuration class for the Xoron
multimodal model. It inherits from PreTrainedConfig to enable:
- Loading via AutoConfig
- Saving/loading with save_pretrained/from_pretrained
- Hub integration with push_to_hub

Usage:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("your-repo/xoron-model", trust_remote_code=True)
"""

from transformers import PreTrainedConfig
from typing import List, Tuple, Union


class XoronConfig(PreTrainedConfig):
    """
    Configuration class for Xoron-Dev multimodal model.
    
    This is a HuggingFace-compatible configuration that stores all the parameters
    needed to instantiate a XoronMultimodalModel.
    
    Args:
        model_name (`str`, *optional*, defaults to `"Xoron-Dev-MultiMoE"`):
            Name of the model.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the hidden representations.
        num_layers (`int`, *optional*, defaults to 12):
            Number of transformer layers.
        num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimension of the MLP intermediate layer.
        vocab_size (`int`, *optional*, defaults to 151643):
            Vocabulary size (Qwen2.5 tokenizer).
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            Maximum sequence length (128K context).
        
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
        - VidTok-style 1D tokenization for video encoding
        - Dual-stream attention for symmetric processing
        - Conformer audio encoder/decoder
        - FP16-native numerical stability
        - Multi-scale training for variable resolution handling
    """
    
    model_type = "xoron"
    
    def __init__(
        self,
        # Model identification
        model_name: str = "Xoron-Dev-MultiMoE",
        
        # LLM Architecture
        hidden_size: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
        intermediate_size: int = 2048,
        vocab_size: int = 151643,
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-6,
        
        # Ring Attention
        use_ring_attention: bool = True,
        ring_attention_chunk_size: int = 4096,
        
        # Tie word embeddings
        tie_word_embeddings: bool = True,
        
        # MoE Configuration
        use_moe: bool = True,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        moe_layer_freq: int = 2,
        use_shared_expert: bool = True,
        moe_capacity_factor: float = 1.25,
        use_aux_lossless: bool = True,
        
        # Vision Configuration
        vision_model_name: str = "google/siglip-so400m-patch14-384",
        freeze_vision: bool = False,
        num_vision_tokens: int = 64,
        projector_type: str = "perceiver",
        
        # Vision Encoder SOTA Features
        use_vision_dual_stream: bool = True,
        use_vision_titok: bool = True,
        num_vision_titok_tokens: int = 256,
        num_vision_dual_stream_layers: int = 2,
        
        # Video Encoder SOTA Features
        use_video_3d_rope: bool = True,
        use_video_temporal_moe: bool = True,
        num_video_encoder_layers: int = 4,
        num_video_experts: int = 4,
        use_video_vidtok: bool = True,
        vidtok_latent_channels: int = 4,
        vidtok_temporal_compression: int = 4,
        vidtok_spatial_compression: int = 8,
        vidtok_causal: bool = True,
        vidtok_use_fsq: bool = False,
        
        # Continuous-Scale Training Configuration
        use_multi_scale: bool = True,
        use_continuous_scale: bool = True,
        image_min_size: int = 128,
        image_max_size: int = 384,
        image_base_size: int = 256,
        image_size_step: int = 32,
        video_min_size: int = 128,
        video_max_size: int = 320,
        video_base_size: int = 192,
        video_size_step: int = 32,
        video_min_frames: int = 8,
        video_max_frames: int = 24,
        video_base_frames: int = 16,
        video_frame_step: int = 4,
        multi_scale_strategy: str = "adaptive",
        multi_scale_warmup_epochs: int = 3,
        adaptive_scale_oom_penalty: float = 0.5,
        adaptive_scale_success_boost: float = 0.1,
        generation_supported_sizes: Union[List[int], Tuple[int, ...]] = (192, 256, 320, 384),
        generation_supported_frames: Union[List[int], Tuple[int, ...]] = (8, 12, 16, 20, 24),
        
        # Image Generation Configuration
        enable_generation: bool = True,
        generation_latent_channels: int = 4,
        generation_base_channels: int = 128,
        generation_inference_steps: int = 50,
        generation_cfg_scale: float = 7.5,
        generation_use_flow_matching: bool = True,
        generation_num_experts: int = 4,
        generation_use_dual_stream: bool = True,
        
        # Video Generation Configuration
        generation_video_cfg_scale: float = 7.5,
        generation_video_use_flow_matching: bool = True,
        generation_video_num_experts: int = 4,
        generation_video_use_3d_rope: bool = True,
        generation_video_use_temporal_moe: bool = True,
        
        # Audio Configuration
        audio_sample_rate: int = 16000,
        audio_n_mels: int = 80,
        audio_max_length: int = 625,  # Max mel frames (10 seconds at 16kHz with hop=256)
        audio_max_waveform_samples: int = 160000,  # Max raw waveform (10 seconds at 16kHz)
        audio_num_speakers: int = 256,
        use_raw_waveform: bool = True,
        audio_kv_lora_rank: int = 256,
        audio_speaker_embed_dim: int = 256,
        use_mas: bool = True,
        use_in_context_audio_prompting: bool = True,
        
        # Tokenizer Configuration
        tokenizer_name: str = "Qwen/Qwen2.5-1.5B",
        
        # LoRA Configuration
        use_lora: bool = True,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        lora_target_modules: Union[List[str], Tuple[str, ...]] = (
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
        ),
        train_lora_only: bool = False,
        use_rslora: bool = True,
        use_dora: bool = False,
        lora_plus_lr_ratio: float = 4.0,
        
        # Cross-Attention Configuration
        use_cross_attention: bool = True,
        cross_attention_layers: int = 4,
        cross_attention_heads: int = 8,
        cross_attention_dropout: float = 0.1,
        
        # Flash Attention Configuration
        use_flash_attention: bool = True,
        
        # Architecture flags (set during save to track what components exist)
        has_audio_encoder: bool = True,
        has_audio_decoder: bool = True,
        has_waveform_decoder: bool = True,
        has_vision_encoder: bool = True,
        has_video_encoder: bool = True,
        has_generator: bool = True,
        has_video_generator: bool = True,
        has_cross_attention: bool = True,
        lora_applied: bool = False,
        architecture_version: int = 2,
        
        # Output path (used during training)
        output_dir: str = "./xoron-model",
        
        **kwargs,
    ):
        # Call parent init
        super().__init__(**kwargs)
        
        # Model identification
        self.model_name = model_name
        
        # LLM Architecture
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        
        # Ring Attention
        self.use_ring_attention = use_ring_attention
        self.ring_attention_chunk_size = ring_attention_chunk_size
        
        # Tie word embeddings
        self.tie_word_embeddings = tie_word_embeddings
        
        # MoE Configuration
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.use_shared_expert = use_shared_expert
        self.moe_capacity_factor = moe_capacity_factor
        self.use_aux_lossless = use_aux_lossless
        
        # Vision Configuration
        self.vision_model_name = vision_model_name
        self.freeze_vision = freeze_vision
        self.num_vision_tokens = num_vision_tokens
        self.projector_type = projector_type
        
        # Vision Encoder SOTA Features
        self.use_vision_dual_stream = use_vision_dual_stream
        self.use_vision_titok = use_vision_titok
        self.num_vision_titok_tokens = num_vision_titok_tokens
        self.num_vision_dual_stream_layers = num_vision_dual_stream_layers
        
        # Video Encoder SOTA Features
        self.use_video_3d_rope = use_video_3d_rope
        self.use_video_temporal_moe = use_video_temporal_moe
        self.num_video_encoder_layers = num_video_encoder_layers
        self.num_video_experts = num_video_experts
        self.use_video_vidtok = use_video_vidtok
        self.vidtok_latent_channels = vidtok_latent_channels
        self.vidtok_temporal_compression = vidtok_temporal_compression
        self.vidtok_spatial_compression = vidtok_spatial_compression
        self.vidtok_causal = vidtok_causal
        self.vidtok_use_fsq = vidtok_use_fsq
        
        # Continuous-Scale Training Configuration
        self.use_multi_scale = use_multi_scale
        self.use_continuous_scale = use_continuous_scale
        self.image_min_size = image_min_size
        self.image_max_size = image_max_size
        self.image_base_size = image_base_size
        self.image_size_step = image_size_step
        self.video_min_size = video_min_size
        self.video_max_size = video_max_size
        self.video_base_size = video_base_size
        self.video_size_step = video_size_step
        self.video_min_frames = video_min_frames
        self.video_max_frames = video_max_frames
        self.video_base_frames = video_base_frames
        self.video_frame_step = video_frame_step
        self.multi_scale_strategy = multi_scale_strategy
        self.multi_scale_warmup_epochs = multi_scale_warmup_epochs
        self.adaptive_scale_oom_penalty = adaptive_scale_oom_penalty
        self.adaptive_scale_success_boost = adaptive_scale_success_boost
        self.generation_supported_sizes = list(generation_supported_sizes) if not isinstance(generation_supported_sizes, list) else generation_supported_sizes
        self.generation_supported_frames = list(generation_supported_frames) if not isinstance(generation_supported_frames, list) else generation_supported_frames
        
        # Image Generation Configuration
        self.enable_generation = enable_generation
        self.generation_latent_channels = generation_latent_channels
        self.generation_base_channels = generation_base_channels
        self.generation_inference_steps = generation_inference_steps
        self.generation_cfg_scale = generation_cfg_scale
        self.generation_use_flow_matching = generation_use_flow_matching
        self.generation_num_experts = generation_num_experts
        self.generation_use_dual_stream = generation_use_dual_stream
        
        # Video Generation Configuration
        self.generation_video_cfg_scale = generation_video_cfg_scale
        self.generation_video_use_flow_matching = generation_video_use_flow_matching
        self.generation_video_num_experts = generation_video_num_experts
        self.generation_video_use_3d_rope = generation_video_use_3d_rope
        self.generation_video_use_temporal_moe = generation_video_use_temporal_moe
        
        # Audio Configuration
        self.audio_sample_rate = audio_sample_rate
        self.audio_n_mels = audio_n_mels
        self.audio_max_length = audio_max_length
        self.audio_max_waveform_samples = audio_max_waveform_samples
        self.audio_num_speakers = audio_num_speakers
        self.use_raw_waveform = use_raw_waveform
        self.audio_kv_lora_rank = audio_kv_lora_rank
        self.audio_speaker_embed_dim = audio_speaker_embed_dim
        self.use_mas = use_mas
        self.use_in_context_audio_prompting = use_in_context_audio_prompting
        
        # Tokenizer Configuration
        self.tokenizer_name = tokenizer_name
        
        # LoRA Configuration
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = list(lora_target_modules) if not isinstance(lora_target_modules, list) else lora_target_modules
        self.train_lora_only = train_lora_only
        self.use_rslora = use_rslora
        self.use_dora = use_dora
        self.lora_plus_lr_ratio = lora_plus_lr_ratio
        
        # Cross-Attention Configuration
        self.use_cross_attention = use_cross_attention
        self.cross_attention_layers = cross_attention_layers
        self.cross_attention_heads = cross_attention_heads
        self.cross_attention_dropout = cross_attention_dropout
        
        # Flash Attention Configuration
        self.use_flash_attention = use_flash_attention
        
        # Architecture flags
        self.has_audio_encoder = has_audio_encoder
        self.has_audio_decoder = has_audio_decoder
        self.has_waveform_decoder = has_waveform_decoder
        self.has_vision_encoder = has_vision_encoder
        self.has_video_encoder = has_video_encoder
        self.has_generator = has_generator
        self.has_video_generator = has_video_generator
        self.has_cross_attention = has_cross_attention
        self.lora_applied = lora_applied
        self.architecture_version = architecture_version
        
        # Output path
        self.output_dir = output_dir

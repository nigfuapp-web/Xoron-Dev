"""
Xoron-Dev Model Configuration for HuggingFace Transformers.

This configuration class defines the architecture parameters for the Xoron-Dev
multimodal MoE model, compatible with HuggingFace's AutoConfig.

Example:
    >>> from transformers import AutoConfig
    >>> config = AutoConfig.from_pretrained("your-username/Xoron-Dev", trust_remote_code=True)
"""

from transformers import PretrainedConfig
from typing import Optional, List, Tuple, Union


class XoronConfig(PretrainedConfig):
    """
    Configuration class for Xoron-Dev Multimodal MoE Language Model.
    
    This is a HuggingFace-compatible configuration that can be used with AutoConfig.
    
    Architecture:
        - LLM: 12 layers, 1024 hidden, 16 heads, 2048 intermediate
        - MoE: 8 experts, top-2 routing, aux-lossless, shared expert
        - Context: 128K with Ring Attention (4096 chunk)
        - Vocab: 151,643 tokens (Qwen2.5 tokenizer)
        - RoPE: YaRN with LongRoPE improvements
        - Vision: SigLIP SO400M + TiTok + Dual-Stream
        - Video: 3D-RoPE + VidTok + Temporal MoE
        - Audio: Raw Waveform + Conformer + RMLA
        - Generation: MoE-DiT + Flow Matching
    
    Args:
        vocab_size (`int`, *optional*, defaults to 151643):
            Vocabulary size of the Xoron model (Qwen2.5 tokenizer).
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 16):
            Number of key-value heads for GQA.
        hidden_act (`str`, *optional*, defaults to "silu"):
            The non-linear activation function.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            Maximum sequence length (128K context).
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon for RMSNorm layers.
        use_cache (`bool`, *optional*, defaults to True):
            Whether to use KV cache.
        tie_word_embeddings (`bool`, *optional*, defaults to True):
            Whether to tie input/output embeddings.
        rope_theta (`float`, *optional*, defaults to 500000.0):
            Base frequency for RoPE.
        rope_scaling (`dict`, *optional*):
            Configuration for YaRN RoPE scaling.
        use_moe (`bool`, *optional*, defaults to True):
            Whether to use MoE layers.
        num_experts (`int`, *optional*, defaults to 8):
            Number of MoE experts.
        num_experts_per_tok (`int`, *optional*, defaults to 2):
            Number of experts activated per token.
        moe_layer_freq (`int`, *optional*, defaults to 2):
            Frequency of MoE layers (every N layers).
        use_shared_expert (`bool`, *optional*, defaults to True):
            Whether to use a shared expert.
        use_aux_lossless (`bool`, *optional*, defaults to True):
            Whether to use aux-lossless MoE routing.
        use_ring_attention (`bool`, *optional*, defaults to True):
            Whether to use Ring Attention for long context.
        ring_attention_chunk_size (`int`, *optional*, defaults to 4096):
            Chunk size for Ring Attention.
        use_mla (`bool`, *optional*, defaults to True):
            Whether to use Multi-Head Latent Attention.
    """
    
    model_type = "xoron"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # Model name
        model_name: str = "Xoron-Dev-MultiMoE",
        
        # LLM Architecture
        vocab_size: int = 151643,
        hidden_size: int = 1024,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 16,
        num_key_value_heads: Optional[int] = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 151643,
        eos_token_id: int = 151645,
        tie_word_embeddings: bool = True,
        
        # RoPE Configuration
        rope_theta: float = 500000.0,
        rope_scaling: Optional[dict] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        
        # Ring Attention
        use_ring_attention: bool = True,
        ring_attention_chunk_size: int = 4096,
        
        # MoE Configuration
        use_moe: bool = True,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        moe_layer_freq: int = 2,
        use_shared_expert: bool = True,
        use_aux_lossless: bool = True,
        moe_intermediate_size: Optional[int] = None,
        shared_expert_intermediate_size: Optional[int] = None,
        norm_topk_prob: bool = True,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.001,
        moe_capacity_factor: float = 1.25,
        first_k_dense_replace: int = 1,
        
        # MLA (Multi-Head Latent Attention)
        use_mla: bool = True,
        kv_lora_rank: int = 512,
        q_lora_rank: int = 1536,
        qk_rope_head_dim: int = 64,
        qk_nope_head_dim: int = 128,
        v_head_dim: int = 128,
        
        # Vision Configuration
        vision_model_name: str = "google/siglip-so400m-patch14-384",
        freeze_vision: bool = False,
        num_vision_tokens: int = 64,
        projector_type: str = "perceiver",
        use_vision_dual_stream: bool = True,
        use_vision_titok: bool = True,
        num_vision_titok_tokens: int = 256,
        num_vision_dual_stream_layers: int = 2,
        
        # Video Configuration
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
        
        # Multi-scale Configuration
        use_multi_scale: bool = True,
        use_continuous_scale: bool = True,
        multi_scale_strategy: str = "adaptive",
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
        
        # Generation Configuration
        enable_generation: bool = True,
        generation_latent_channels: int = 4,
        generation_base_channels: int = 128,
        generation_inference_steps: int = 50,
        generation_cfg_scale: float = 7.5,
        generation_use_flow_matching: bool = True,
        generation_num_experts: int = 4,
        generation_use_dual_stream: bool = True,
        generation_video_cfg_scale: float = 7.5,
        generation_video_use_flow_matching: bool = True,
        generation_video_num_experts: int = 4,
        generation_video_use_3d_rope: bool = True,
        generation_video_use_temporal_moe: bool = True,
        
        # Audio Configuration
        audio_sample_rate: int = 16000,
        audio_n_mels: int = 80,
        audio_max_length: int = 1000,
        audio_num_speakers: int = 256,
        use_raw_waveform: bool = True,
        audio_kv_lora_rank: int = 256,
        audio_speaker_embed_dim: int = 256,
        use_mas: bool = True,
        use_in_context_audio_prompting: bool = True,
        
        # Tokenizer
        tokenizer_name: str = "Qwen/Qwen2.5-1.5B",
        
        # LoRA Configuration
        use_lora: bool = True,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        train_lora_only: bool = False,
        use_rslora: bool = True,
        use_dora: bool = False,
        lora_plus_lr_ratio: float = 16.0,
        
        # Cross-Attention
        use_cross_attention: bool = True,
        cross_attention_layers: int = 4,
        cross_attention_heads: int = 8,
        cross_attention_dropout: float = 0.1,
        
        # Flash Attention
        use_flash_attention: bool = True,
        
        # Output
        output_dir: str = "./xoron_output",
        
        **kwargs,
    ):
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        
        # Ring Attention
        self.use_ring_attention = use_ring_attention
        self.ring_attention_chunk_size = ring_attention_chunk_size
        
        # MoE Configuration
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.use_shared_expert = use_shared_expert
        self.use_aux_lossless = use_aux_lossless
        self.moe_intermediate_size = moe_intermediate_size or intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size or intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.moe_capacity_factor = moe_capacity_factor
        self.first_k_dense_replace = first_k_dense_replace
        
        # MLA
        self.use_mla = use_mla
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        
        # Vision
        self.vision_model_name = vision_model_name
        self.freeze_vision = freeze_vision
        self.num_vision_tokens = num_vision_tokens
        self.projector_type = projector_type
        self.use_vision_dual_stream = use_vision_dual_stream
        self.use_vision_titok = use_vision_titok
        self.num_vision_titok_tokens = num_vision_titok_tokens
        self.num_vision_dual_stream_layers = num_vision_dual_stream_layers
        
        # Video
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
        
        # Multi-scale
        self.use_multi_scale = use_multi_scale
        self.use_continuous_scale = use_continuous_scale
        self.multi_scale_strategy = multi_scale_strategy
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
        
        # Generation
        self.enable_generation = enable_generation
        self.generation_latent_channels = generation_latent_channels
        self.generation_base_channels = generation_base_channels
        self.generation_inference_steps = generation_inference_steps
        self.generation_cfg_scale = generation_cfg_scale
        self.generation_use_flow_matching = generation_use_flow_matching
        self.generation_num_experts = generation_num_experts
        self.generation_use_dual_stream = generation_use_dual_stream
        self.generation_video_cfg_scale = generation_video_cfg_scale
        self.generation_video_use_flow_matching = generation_video_use_flow_matching
        self.generation_video_num_experts = generation_video_num_experts
        self.generation_video_use_3d_rope = generation_video_use_3d_rope
        self.generation_video_use_temporal_moe = generation_video_use_temporal_moe
        
        # Audio
        self.audio_sample_rate = audio_sample_rate
        self.audio_n_mels = audio_n_mels
        self.audio_max_length = audio_max_length
        self.audio_num_speakers = audio_num_speakers
        self.use_raw_waveform = use_raw_waveform
        self.audio_kv_lora_rank = audio_kv_lora_rank
        self.audio_speaker_embed_dim = audio_speaker_embed_dim
        self.use_mas = use_mas
        self.use_in_context_audio_prompting = use_in_context_audio_prompting
        
        # Tokenizer
        self.tokenizer_name = tokenizer_name
        
        # LoRA
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
        ]
        self.train_lora_only = train_lora_only
        self.use_rslora = use_rslora
        self.use_dora = use_dora
        self.lora_plus_lr_ratio = lora_plus_lr_ratio
        
        # Cross-Attention
        self.use_cross_attention = use_cross_attention
        self.cross_attention_layers = cross_attention_layers
        self.cross_attention_heads = cross_attention_heads
        self.cross_attention_dropout = cross_attention_dropout
        
        # Flash Attention
        self.use_flash_attention = use_flash_attention
        
        # Output
        self.output_dir = output_dir
        
        # Computed properties
        self.head_dim = hidden_size // num_attention_heads
        
        # YaRN RoPE scaling defaults
        if rope_scaling is None:
            self.rope_scaling = {
                "type": "yarn",
                "factor": max_position_embeddings / 8192,
                "original_max_position_embeddings": 8192,
                "beta_fast": 32,
                "beta_slow": 1,
                "mscale": 1.0,
                "mscale_all_dim": 0.0,
            }
        else:
            self.rope_scaling = rope_scaling
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Check if a layer should use MoE based on moe_layer_freq."""
        if not self.use_moe:
            return False
        if layer_idx < self.first_k_dense_replace:
            return False
        return (layer_idx - self.first_k_dense_replace) % self.moe_layer_freq == 0

    # Aliases for compatibility with existing Xoron code
    @property
    def num_layers(self) -> int:
        return self.num_hidden_layers
    
    @property
    def num_heads(self) -> int:
        return self.num_attention_heads

"""Xoron model configuration with SOTA features."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class XoronConfig:
    """
    Configuration for Xoron-Dev multimodal model.
    
    SOTA Features:
    - MoE with shared expert (DeepSeek-style)
    - LoRA variants (rsLoRA, DoRA, LoRA+)
    - Perceiver Resampler for vision projection
    - Cross-attention for multimodal fusion
    - Sliding window attention for 128K context
    - SOTA image/video diffusion with CFG
    - Conformer audio encoder/decoder
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
    
    # Sliding Window Attention (enables efficient 128K context)
    use_sliding_window: bool = True
    sliding_window: int = 4096  # Local attention window size

    # MoE Configuration (SOTA: with shared expert)
    use_moe: bool = True
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_layer_freq: int = 2  # MoE every 2 layers
    router_aux_loss_coef: float = 0.1
    use_shared_expert: bool = True  # DeepSeek-style shared expert
    moe_capacity_factor: float = 1.25  # Expert capacity factor

    # Vision Configuration (SOTA: SigLIP 2)
    vision_model_name: str = "google/siglip-so400m-patch14-384"  # SigLIP 2 - best for MoE
    freeze_vision: bool = False
    num_vision_tokens: int = 64
    max_video_frames: int = 32
    projector_type: str = "perceiver"  # "perceiver", "spatial", "c_abstractor", "mlp"
    vision_image_size: int = 384  # SigLIP SO400M uses 384x384

    # Image Generation Configuration (SOTA: with CFG)
    enable_generation: bool = True
    generation_image_size: int = 256
    generation_latent_channels: int = 4
    generation_base_channels: int = 128
    generation_inference_steps: int = 20  # More steps for quality
    generation_cfg_scale: float = 7.5  # Classifier-free guidance scale
    
    # Video Generation Configuration (SOTA: with motion modules)
    generation_video_size: int = 256
    generation_num_frames: int = 16
    generation_video_cfg_scale: float = 7.5

    # Audio Configuration (SOTA: Conformer)
    audio_sample_rate: int = 16000
    audio_n_mels: int = 80
    audio_num_emotions: int = 13  # neutral, happy, sad, angry, fearful, disgusted, surprised, etc.
    audio_num_speakers: int = 256  # Speaker embedding size

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
    lora_plus_lr_ratio: float = 16.0  # LoRA+ B matrix learns faster

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
            'use_sliding_window': self.use_sliding_window,
            'sliding_window': self.sliding_window,
            'use_moe': self.use_moe,
            'num_experts': self.num_experts,
            'num_experts_per_tok': self.num_experts_per_tok,
            'moe_layer_freq': self.moe_layer_freq,
            'router_aux_loss_coef': self.router_aux_loss_coef,
            'use_shared_expert': self.use_shared_expert,
            'moe_capacity_factor': self.moe_capacity_factor,
            'vision_model_name': self.vision_model_name,
            'freeze_vision': self.freeze_vision,
            'num_vision_tokens': self.num_vision_tokens,
            'max_video_frames': self.max_video_frames,
            'projector_type': self.projector_type,
            'vision_image_size': self.vision_image_size,
            'enable_generation': self.enable_generation,
            'generation_image_size': self.generation_image_size,
            'generation_latent_channels': self.generation_latent_channels,
            'generation_base_channels': self.generation_base_channels,
            'generation_inference_steps': self.generation_inference_steps,
            'generation_cfg_scale': self.generation_cfg_scale,
            'generation_video_size': self.generation_video_size,
            'generation_num_frames': self.generation_num_frames,
            'generation_video_cfg_scale': self.generation_video_cfg_scale,
            'audio_sample_rate': self.audio_sample_rate,
            'audio_n_mels': self.audio_n_mels,
            'audio_num_emotions': self.audio_num_emotions,
            'audio_num_speakers': self.audio_num_speakers,
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
        if 'lora_target_modules' in config_dict and isinstance(config_dict['lora_target_modules'], list):
            config_dict['lora_target_modules'] = tuple(config_dict['lora_target_modules'])
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k) or k in cls.__dataclass_fields__})

    def print_config(self):
        """Print configuration summary."""
        print("=" * 60)
        print("XORON-DEV MODEL CONFIGURATION (SOTA)")
        print("=" * 60)
        print(f"\n🧠 LLM: {self.hidden_size}d, {self.num_layers}L, {self.num_heads}H")
        print(f"📏 Context: {self.max_position_embeddings//1024}K positions, sliding window={self.sliding_window if self.use_sliding_window else 'disabled'}")
        print(f"🎯 MoE: {self.num_experts} experts, top-{self.num_experts_per_tok} routing, shared_expert={self.use_shared_expert}")
        print(f"👁️ Vision: {self.vision_model_name}, projector={self.projector_type}")
        print(f"🎨 Image Generation: {self.generation_image_size}x{self.generation_image_size}, CFG={self.generation_cfg_scale}")
        print(f"🎬 Video Generation: {self.generation_num_frames} frames @ {self.generation_video_size}x{self.generation_video_size}")
        print(f"🎤 Audio: {self.audio_sample_rate}Hz, {self.audio_n_mels} mels, {self.audio_num_emotions} emotions")
        print(f"📝 Tokenizer: {self.tokenizer_name} (vocab: {self.vocab_size:,})")
        lora_type = "DoRA" if self.use_dora else ("rsLoRA" if self.use_rslora else "LoRA")
        print(f"🔧 {lora_type}: r={self.lora_r}, alpha={self.lora_alpha}, LoRA+ ratio={self.lora_plus_lr_ratio}")
        print(f"🔀 Cross-Attention: {self.cross_attention_layers} layers")
        print(f"⚡ Flash Attention: {self.use_flash_attention}")
        print(f"📁 Output: {self.output_dir}")
        print("=" * 60)

"""Training configuration for Xoron-Dev with SOTA features."""

import os
from dataclasses import dataclass, field
from typing import Dict

# Handle torch import gracefully
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Import environment detection
from utils.device import detect_environment, get_environment_paths, print_environment_info


def _get_default_paths():
    """Get default paths based on detected environment."""
    env_info = get_environment_paths()
    return env_info


@dataclass
class TrainingConfig:
    """
    Configuration for training Xoron-Dev model.
    
    SOTA Training Features:
    - LoRA+ with different learning rates for A and B matrices
    - Separate loss weights for different modalities
    - Chain-of-thought weighted loss for reasoning
    - Temporal consistency loss for video
    - Classifier-free guidance training
    """

    # Environment (auto-detected if not specified)
    environment: str = field(default_factory=detect_environment)

    # Model path (where built model will be saved and loaded from)
    model_path: str = field(default_factory=lambda: _get_default_paths().model_dir)

    # Storage paths
    temp_dir: str = field(default_factory=lambda: _get_default_paths().temp_dir)
    datasets_dir: str = field(default_factory=lambda: _get_default_paths().datasets_dir)
    output_dir: str = field(default_factory=lambda: _get_default_paths().output_dir)
    final_model_dir: str = field(default_factory=lambda: _get_default_paths().final_model_dir)

    # Dataset settings
    max_per_epoch: int = 6600
    max_per_dataset: int = 100  # Prevent any single dataset from dominating the epoch
    sample_repeat: int = 4  # Each sample shown N times within gradient accumulation window

    # Training settings - optimized for ~31GB VRAM
    batch_size: int = 1
    gradient_accumulation_steps: int = 128
    learning_rate: float = 5e-5  # REDUCED: was 2e-4, too high for MoE + LoRA stability
    weight_decay: float = 0.01
    num_epochs: int = 2
    warmup_ratio: float = 0.05  # INCREASED: longer warmup for stability
    max_seq_length: int = 1024
    max_grad_norm: float = 0.5  # REDUCED: tighter clipping for MoE stability (was 1.0)
    
    # LoRA+ settings (different LR for A and B matrices)
    use_lora_plus: bool = True
    lora_plus_lr_ratio: float = 4.0  # REDUCED: was 16.0, too aggressive for stability
    
    # Chain-of-thought training settings
    cot_loss_weight: float = 1.5  # Higher weight for reasoning tokens
    
    # Modality loss weights (SOTA: balanced multi-task learning)
    llm_loss_weight: float = 1.0
    image_diffusion_loss_weight: float = 0.1
    video_diffusion_loss_weight: float = 0.1
    asr_loss_weight: float = 0.1
    tts_loss_weight: float = 0.1
    moe_aux_loss_weight: float = 0.02
    
    # Video training settings
    temporal_consistency_weight: float = 0.01  # Encourage smooth motion
    
    # Classifier-free guidance training
    cfg_dropout_rate: float = 0.1  # Probability of dropping context during training

    # Checkpointing
    save_steps: int = 500
    logging_steps: int = 50
    
    # Evaluation/validation settings
    # Eval runs at END of each epoch (not at step intervals)
    # Eval pulls samples SEPARATELY from each dataset for proper validation
    # Total eval = num_active_datasets * max_per_dataset_eval (no cap)
    max_per_dataset_eval: int = 10  # Samples per dataset for eval (e.g., 10 from each dataset)

    # Device settings
    device: str = field(default_factory=lambda: "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
    fp16: bool = True
    bf16: bool = False  # Use bfloat16 if available (better for training)
    use_model_parallel: bool = field(default_factory=lambda: TORCH_AVAILABLE and torch.cuda.is_available() and torch.cuda.device_count() > 1)

    # Memory optimization
    empty_cache_freq: int = 5  # More frequent cache clearing (was 10)
    gradient_checkpointing: bool = True  # Trade compute for memory
    use_8bit_optimizer: bool = True  # Use 8-bit Adam (saves ~75% optimizer memory)
    set_to_none: bool = True  # Use set_to_none in zero_grad (saves memory)

    def __post_init__(self):
        """Create directories and set environment variables."""
        for d in [self.temp_dir, self.datasets_dir, self.output_dir, self.final_model_dir, self.model_path]:
            os.makedirs(d, exist_ok=True)

        os.environ['HF_DATASETS_CACHE'] = self.datasets_dir
        os.environ['HF_HOME'] = self.temp_dir

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size."""
        return self.batch_size * self.gradient_accumulation_steps

    def print_config(self):
        """Print training configuration."""
        env_icons = {'kaggle': '🏆', 'colab': '🔬', 'lightning': '⚡️', 'local': '💻'}
        env_icon = env_icons.get(self.environment, '🖥️')
        
        print("=" * 60)
        print("XORON-DEV TRAINING CONFIGURATION (SOTA)")
        print("=" * 60)
        print(f"{env_icon} Environment: {self.environment.upper()}")
        print(f"\n📁 Paths:")
        print(f"   Model path: {self.model_path}")
        print(f"   Temp dir: {self.temp_dir}")
        print(f"   Datasets dir: {self.datasets_dir}")
        print(f"   Output dir: {self.output_dir}")
        print(f"   Final model dir: {self.final_model_dir}")
        print(f"\n📊 Dataset Settings:")
        print(f"   Max per epoch: {self.max_per_epoch:,}")
        print(f"   Max per dataset: {self.max_per_dataset:,}")
        print(f"   Sample repeat: {self.sample_repeat}x (each sample shown {self.sample_repeat} times)")
        print(f"   Streaming: True (memory efficient)")
        print(f"\n⚙️ Training Settings:")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"   Effective batch size: {self.effective_batch_size}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Epochs: {self.num_epochs}")
        print(f"   Max sequence length: {self.max_seq_length}")
        print(f"   FP16: {self.fp16}, BF16: {self.bf16}")
        print(f"   Model Parallelism: {self.use_model_parallel}")
        print(f"   Gradient Checkpointing: {self.gradient_checkpointing}")
        print(f"   8-bit Optimizer: {self.use_8bit_optimizer}")
        print(f"   Zero Grad set_to_none: {self.set_to_none}")
        print(f"\n🎯 Loss Weights:")
        print(f"   LLM: {self.llm_loss_weight}")
        print(f"   CoT: {self.cot_loss_weight}x")
        print(f"   Image Diffusion: {self.image_diffusion_loss_weight}")
        print(f"   Video Diffusion: {self.video_diffusion_loss_weight}")
        print(f"   ASR: {self.asr_loss_weight}")
        print(f"   TTS: {self.tts_loss_weight}")
        print(f"   MoE Aux: {self.moe_aux_loss_weight}")
        print(f"\n🔧 LoRA+ Settings:")
        print(f"   Enabled: {self.use_lora_plus}")
        print(f"   LR Ratio (B/A): {self.lora_plus_lr_ratio}x")
        print(f"\n📊 Evaluation Settings:")
        print(f"   Samples per dataset (eval): {self.max_per_dataset_eval}")

        if TORCH_AVAILABLE and torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"\n💻 GPU Configuration:")
            for i in range(num_gpus):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
            print(f"   Total GPU Memory: {sum(torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)) / 1e9:.1f} GB")
        print("=" * 60)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'environment': self.environment,
            'model_path': self.model_path,
            'temp_dir': self.temp_dir,
            'datasets_dir': self.datasets_dir,
            'output_dir': self.output_dir,
            'final_model_dir': self.final_model_dir,
            'max_per_epoch': self.max_per_epoch,
            'max_per_dataset': self.max_per_dataset,
            'sample_repeat': self.sample_repeat,
            'batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_epochs': self.num_epochs,
            'warmup_ratio': self.warmup_ratio,
            'max_seq_length': self.max_seq_length,
            'max_grad_norm': self.max_grad_norm,
            'use_lora_plus': self.use_lora_plus,
            'lora_plus_lr_ratio': self.lora_plus_lr_ratio,
            'cot_loss_weight': self.cot_loss_weight,
            'llm_loss_weight': self.llm_loss_weight,
            'image_diffusion_loss_weight': self.image_diffusion_loss_weight,
            'video_diffusion_loss_weight': self.video_diffusion_loss_weight,
            'asr_loss_weight': self.asr_loss_weight,
            'tts_loss_weight': self.tts_loss_weight,
            'moe_aux_loss_weight': self.moe_aux_loss_weight,
            'temporal_consistency_weight': self.temporal_consistency_weight,
            'cfg_dropout_rate': self.cfg_dropout_rate,
            'save_steps': self.save_steps,
            'logging_steps': self.logging_steps,
            'max_per_dataset_eval': self.max_per_dataset_eval,
            'device': self.device,
            'fp16': self.fp16,
            'bf16': self.bf16,
            'use_model_parallel': self.use_model_parallel,
            'empty_cache_freq': self.empty_cache_freq,
            'gradient_checkpointing': self.gradient_checkpointing,
            'use_8bit_optimizer': self.use_8bit_optimizer,
            'set_to_none': self.set_to_none,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


def get_device_map(num_gpus: int) -> Dict[str, str]:
    """
    Create a device map for Model Parallelism.
    Distributes model components across available GPUs.

    Strategy:
    - GPU 0: Encoders (vision, video, audio) + Projectors
    - GPU 1+: LLM backbone + Cross-attention + Generators
    """
    if num_gpus <= 1:
        return {
            'vision_encoder': 'cuda:0',
            'video_encoder': 'cuda:0',
            'audio_encoder': 'cuda:0',
            'audio_decoder': 'cuda:0',
            'projector': 'cuda:0',
            'audio_projector': 'cuda:0',
            'llm': 'cuda:0',
            'cross_attention': 'cuda:0',
            'generator': 'cuda:0',
            'video_generator': 'cuda:0',
            'modality_markers': 'cuda:0',
            'primary': 'cuda:0',
        }
    elif num_gpus == 2:
        return {
            'vision_encoder': 'cuda:0',
            'video_encoder': 'cuda:0',
            'audio_encoder': 'cuda:0',
            'audio_decoder': 'cuda:1',
            'projector': 'cuda:0',
            'audio_projector': 'cuda:0',
            'llm': 'cuda:1',
            'cross_attention': 'cuda:1',
            'generator': 'cuda:1',
            'video_generator': 'cuda:1',
            'modality_markers': 'cuda:1',
            'primary': 'cuda:0',
        }
    else:
        return {
            'vision_encoder': 'cuda:0',
            'video_encoder': 'cuda:0',
            'audio_encoder': 'cuda:0',
            'audio_decoder': 'cuda:2' if num_gpus > 2 else 'cuda:1',
            'projector': 'cuda:0',
            'audio_projector': 'cuda:0',
            'llm': 'cuda:1',
            'cross_attention': 'cuda:1',
            'generator': 'cuda:2' if num_gpus > 2 else 'cuda:1',
            'video_generator': 'cuda:2' if num_gpus > 2 else 'cuda:1',
            'modality_markers': 'cuda:1',
            'primary': 'cuda:0',
        }

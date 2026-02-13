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
    
    SOTA Architecture (v2.0):
    - LLM: MLA (Multi-Head Latent Attention), YaRN/LongRoPE, Ring Attention
    - MoE: Aux-Lossless with Shared Expert Isolation
    - Image: MoE-DiT, Flow Matching, 2D-RoPE, Dual-Stream Attention
    - Video: Flow Matching, 3D-RoPE, Temporal Expert Routing, 3D Causal Transformers
    
    Optimized for Kaggle 2x T4 GPUs (32GB total VRAM)
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
    max_per_epoch: int = 1
    max_per_dataset: int = 50  # Prevent any single dataset from dominating the epoch
    sample_repeat: int = 2  # Each sample shown N times within gradient accumulation window

    # Training settings - optimized for ~31GB VRAM
    batch_size: int = 1
    gradient_accumulation_steps: int = 16  # REDUCED: 128 too high for FP16 stability, causes gradient explosion
    learning_rate: float = 1e-4  # INCREASED: FP32 optimizer states prevent overflow, can use higher LR
    weight_decay: float = 0.01
    num_epochs: int = 1
    warmup_ratio: float = 0.05  # INCREASED: longer warmup for stability
    max_seq_length: int = 1024
    max_grad_norm: float = 1.0  # INCREASED: FP32 optimizer states are stable, can use standard clipping
    
    # LoRA+ settings (different LR for A and B matrices)
    use_lora_plus: bool = True
    lora_plus_lr_ratio: float = 4.0  # REDUCED: was 16.0, too aggressive for stability
    
    # Chain-of-thought training settings
    cot_loss_weight: float = 1.5  # Higher weight for reasoning tokens
    
    # Special token loss weights (SOTA: weighted loss for important token groups)
    tool_loss_weight: float = 1.3  # Tool calling tokens (critical for agentic behavior)
    anti_hallucination_loss_weight: float = 1.2  # Uncertainty, citations tokens
    code_exec_loss_weight: float = 1.2  # Code execution tokens
    
    # Modality loss weights (SOTA: balanced multi-task learning)
    llm_loss_weight: float = 1.0
    image_diffusion_loss_weight: float = 0.1
    video_diffusion_loss_weight: float = 0.1
    asr_loss_weight: float = 0.1
    tts_loss_weight: float = 0.1
    # Note: No MoE aux loss - we use Aux-Lossless MoE routing
       
    # Debug settings
    debug_nan_checks: bool = False  # Enable expensive NaN/Inf checks (causes CPU-GPU sync)
    
    logging_steps: int = 50  # Used in trainer_state.json for HuggingFace compatibility
    
    max_per_dataset_eval: int = 2  # Samples per dataset for eval (e.g., 2 from each dataset)

    # Device settings
    device: str = field(default_factory=lambda: "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
    fp16: bool = True
    bf16: bool = False  # Use bfloat16 if available (better for training)
    use_model_parallel: bool = field(default_factory=lambda: TORCH_AVAILABLE and torch.cuda.is_available() and torch.cuda.device_count() > 1)

    # Memory optimization
    empty_cache_freq: int = 100  # Clear cache less frequently (was 5 - too slow!)
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
        print(f"   CoT: {self.cot_loss_weight}x | Tool: {self.tool_loss_weight}x | Anti-Hallucination: {self.anti_hallucination_loss_weight}x | Code: {self.code_exec_loss_weight}x")
        print(f"   Image Diffusion: {self.image_diffusion_loss_weight}")
        print(f"   Video Diffusion: {self.video_diffusion_loss_weight}")
        print(f"   ASR: {self.asr_loss_weight} | TTS: {self.tts_loss_weight}")
        print(f"   MoE: Aux-Lossless (no aux loss needed)")
        print(f"\n🔧 Debug Settings:")
        print(f"   NaN checks: {self.debug_nan_checks}")
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
            'tool_loss_weight': self.tool_loss_weight,
            'anti_hallucination_loss_weight': self.anti_hallucination_loss_weight,
            'code_exec_loss_weight': self.code_exec_loss_weight,
            'llm_loss_weight': self.llm_loss_weight,
            'image_diffusion_loss_weight': self.image_diffusion_loss_weight,
            'video_diffusion_loss_weight': self.video_diffusion_loss_weight,
            'asr_loss_weight': self.asr_loss_weight,
            'tts_loss_weight': self.tts_loss_weight,
            'debug_nan_checks': self.debug_nan_checks,
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


from typing import Dict, List, Optional

# Component groups mapped to their individual model attributes
COMPONENT_TO_ATTRS = {
    'vision': ['vision_encoder', 'projector'],
    'video': ['video_encoder', 'video_generator'],
    'audio': ['audio_encoder', 'audio_decoder', 'audio_projector', 'waveform_decoder'],
    'image_generation': ['generator'],
    'video_generation': ['video_generator'],
    'llm': ['llm'],
    'cross_attention': ['cross_attention'],
    'modality_markers': ['modality_markers'],
}

# All model component attributes
ALL_COMPONENT_ATTRS = [
    'vision_encoder', 'video_encoder', 'audio_encoder', 'audio_decoder',
    'waveform_decoder', 'projector', 'audio_projector', 'llm',
    'cross_attention', 'generator', 'video_generator', 'modality_markers'
]


def get_device_map(num_gpus: int, trainable_groups: Optional[List[str]] = None, frozen_groups: Optional[List[str]] = None, active_modalities: str = 'all') -> Dict[str, str]:
    """
    Create a dynamic device map for Model Parallelism.
    
    SMART PLACEMENT STRATEGY:
    - cuda:0 (primary): Modality-specific components being trained
    - cuda:1 (secondary): LLM + frozen components
    
    EXCEPTION: Text-only mode puts LLM on cuda:0 (since that's the main thing)
    
    This maximizes VRAM on the training GPU for video/image generation.
    
    Args:
        num_gpus: Number of available GPUs
        trainable_groups: List of component group names that will be trained
                         e.g., ['vision', 'video', 'video_generation', 'llm']
        frozen_groups: List of component group names that are frozen
                      e.g., ['audio', 'image_generation']
        active_modalities: Which modalities are active ('all', 'text', 'image', 'video', 'audio')
    
    Returns:
        Device map dict mapping component names to device strings
    """
    # Single GPU - everything on cuda:0
    if num_gpus <= 1:
        device_map = {attr: 'cuda:0' for attr in ALL_COMPONENT_ATTRS}
        device_map['primary'] = 'cuda:0'
        return device_map
    
    # Multi-GPU with dynamic placement
    if num_gpus >= 2:
        # Default: use static layout if no trainable/frozen info provided
        if trainable_groups is None and frozen_groups is None:
            return _get_static_device_map(num_gpus)
        
        # Dynamic placement based on what's being trained
        text_only = is_text_only_mode(active_modalities)
        return _get_dynamic_device_map(num_gpus, trainable_groups or [], frozen_groups or [], is_text_only=text_only)
    
    return _get_static_device_map(num_gpus)


def _get_static_device_map(num_gpus: int) -> Dict[str, str]:
    """Static device map for default placement when no modality info provided."""
    if num_gpus == 2:
        return {
            'vision_encoder': 'cuda:0',
            'video_encoder': 'cuda:0',
            'audio_encoder': 'cuda:0',
            'audio_decoder': 'cuda:1',
            'waveform_decoder': 'cuda:1',
            'audio_projector': 'cuda:0',
            'projector': 'cuda:0',
            'video_generator': 'cuda:0',
            'generator': 'cuda:0',
            'llm': 'cuda:1',
            'modality_markers': 'cuda:1',
            'cross_attention': 'cuda:1',
            'primary': 'cuda:0',
        }
    else:
        return {
            'vision_encoder': 'cuda:0',
            'video_encoder': 'cuda:0',
            'audio_encoder': 'cuda:0',
            'audio_projector': 'cuda:0',
            'projector': 'cuda:0',
            'llm': 'cuda:1',
            'modality_markers': 'cuda:1',
            'cross_attention': 'cuda:1',
            'audio_decoder': 'cuda:2',
            'waveform_decoder': 'cuda:2',
            'generator': 'cuda:2',
            'video_generator': 'cuda:2',
            'primary': 'cuda:0',
        }


def _get_dynamic_device_map(num_gpus: int, trainable_groups: List[str], frozen_groups: List[str], is_text_only: bool = False) -> Dict[str, str]:
    """
    Dynamic device placement to maximize VRAM for modality-specific training.
    
    Strategy:
    - cuda:0 (primary): Modality-specific components being trained (NOT LLM)
    - cuda:1 (secondary): LLM + frozen components
    
    EXCEPTION: Text-only mode puts LLM on cuda:0 since that's the only thing training
    
    This maximizes VRAM on the training GPU by keeping LLM (which is large)
    on the secondary GPU, leaving room for video/image generation at larger scales.
    """
    device_map = {}
    
    # Determine which component attributes go on primary GPU (cuda:0)
    primary_attrs = set()
    
    if is_text_only:
        # Text-only mode: LLM is the main thing, put it on primary
        primary_attrs.update(['llm', 'modality_markers', 'cross_attention'])
    else:
        # Non-text mode: Put modality-specific components on primary, LLM on secondary
        for group in trainable_groups:
            if group in COMPONENT_TO_ATTRS:
                # Skip LLM-related groups - they go on secondary
                if group not in ['llm', 'cross_attention', 'modality_markers']:
                    primary_attrs.update(COMPONENT_TO_ATTRS[group])
    
    # Assign devices
    for attr in ALL_COMPONENT_ATTRS:
        if attr in primary_attrs:
            device_map[attr] = 'cuda:0'  # Primary GPU for modality training
        else:
            device_map[attr] = 'cuda:1'  # Secondary GPU for LLM + frozen
    
    device_map['primary'] = 'cuda:0'
    
    # Print placement summary
    on_gpu0 = [a for a in ALL_COMPONENT_ATTRS if device_map[a] == 'cuda:0']
    on_gpu1 = [a for a in ALL_COMPONENT_ATTRS if device_map[a] == 'cuda:1']
    
    print(f"\n🔀 Dynamic Device Placement:")
    print(f"   cuda:0 (primary): {', '.join(on_gpu0) if on_gpu0 else 'none'}")
    print(f"   cuda:1 (secondary): {', '.join(on_gpu1) if on_gpu1 else 'none'}")
    
    return device_map


def get_trainable_groups_from_modality(active_modalities) -> List[str]:
    """
    Convert active modality string or list to list of component groups to train.
    
    NOTE: LLM is always trained but placed on secondary GPU (except text-only mode)
    to maximize VRAM for modality-specific components on primary GPU.
    
    Args:
        active_modalities: One of 'all', 'text', 'image', 'video', 'audio', 'multi'
                          OR a list of modalities like ['text', 'image']
    
    Returns:
        List of component group names that should be trainable
    """
    # Handle list of modalities (for combined modes like --text --image)
    if isinstance(active_modalities, (list, tuple)):
        trainable = set(['llm', 'cross_attention', 'modality_markers'])  # Always trained
        for mode in active_modalities:
            if mode == 'text':
                pass  # LLM already added
            elif mode == 'image':
                trainable.update(['vision', 'image_generation'])
            elif mode == 'video':
                trainable.update(['vision', 'video', 'video_generation'])
            elif mode == 'audio' or mode == 'voice':
                trainable.add('audio')
        return list(trainable)
    
    # Handle string mode
    trainable = []
    
    if active_modalities == 'all':
        # All modalities - train everything
        trainable = ['vision', 'video', 'audio', 'image_generation', 'video_generation', 'llm', 'cross_attention', 'modality_markers']
    elif active_modalities == 'multi':
        # Multi mode without specific list defaults to all
        trainable = ['vision', 'video', 'audio', 'image_generation', 'video_generation', 'llm', 'cross_attention', 'modality_markers']
    elif active_modalities == 'text':
        # Text only - just LLM
        trainable = ['llm', 'cross_attention', 'modality_markers']
    elif active_modalities == 'image':
        # Image mode - vision encoder + image generation + LLM
        trainable = ['vision', 'image_generation', 'llm', 'cross_attention', 'modality_markers']
    elif active_modalities == 'video':
        # Video mode - vision (for frames) + video encoder + video generation + LLM
        trainable = ['vision', 'video', 'video_generation', 'llm', 'cross_attention', 'modality_markers']
    elif active_modalities == 'audio' or active_modalities == 'voice':
        # Audio/voice mode - audio components + LLM
        trainable = ['audio', 'llm', 'cross_attention', 'modality_markers']
    
    return trainable


def is_text_only_mode(active_modalities) -> bool:
    """Check if we're in text-only training mode."""
    if isinstance(active_modalities, (list, tuple)):
        return len(active_modalities) == 1 and active_modalities[0] == 'text'
    return active_modalities == 'text'


def get_frozen_groups_from_trainable(trainable_groups: List[str]) -> List[str]:
    """Get list of frozen component groups given trainable groups."""
    all_groups = ['vision', 'video', 'audio', 'image_generation', 'video_generation']
    # LLM, cross_attention, modality_markers are never frozen
    return [g for g in all_groups if g not in trainable_groups]

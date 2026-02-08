# ⚙️ Configuration Module Documentation

The Configuration module defines all settings for the Xoron-Dev model, training, datasets, special tokens, and chat templates.

## 📁 File Structure

```
config/
├── __init__.py           # Module exports
├── model_config.py       # XoronConfig - model architecture settings
├── training_config.py    # TrainingConfig - training hyperparameters
├── dataset_config.py     # Dataset definitions and filtering
├── special_tokens.py     # Special tokens for all modalities
└── chat_template.py      # Chat template for conversations
```

---

## 🏗️ XoronConfig (Model Configuration)

### Overview

`XoronConfig` defines all architectural parameters for the Xoron-Dev model.

### Key Parameters

#### LLM Architecture
```python
@dataclass
class XoronConfig:
    # Model identity
    model_name: str = "Xoron-Dev-MultiMoE"
    
    # Core dimensions
    hidden_size: int = 1024        # Model dimension
    num_layers: int = 12           # Transformer layers
    num_heads: int = 16            # Attention heads
    intermediate_size: int = 2048  # FFN intermediate size
    vocab_size: int = 151643       # Qwen2.5 vocabulary
    
    # Context length
    max_position_embeddings: int = 131072  # 128K context
    
    # Ring Attention
    use_ring_attention: bool = True
    ring_attention_chunk_size: int = 4096
```

#### MoE Configuration
```python
    # MoE settings
    use_moe: bool = True
    num_experts: int = 8           # Routed experts
    num_experts_per_tok: int = 2   # Top-k routing
    moe_layer_freq: int = 2        # MoE every 2nd layer
    use_shared_expert: bool = True # DeepSeek-style
    use_aux_lossless: bool = True  # No auxiliary loss
```

#### Vision Configuration
```python
    # Vision encoder
    vision_model_name: str = "google/siglip-so400m-patch14-384"
    freeze_vision: bool = False
    num_vision_tokens: int = 64
    projector_type: str = "perceiver"
    
    # SOTA features
    use_vision_dual_stream: bool = True
    use_vision_titok: bool = True
    num_vision_titok_tokens: int = 256
    num_vision_dual_stream_layers: int = 2
```

#### Video Configuration
```python
    # Video encoder
    use_video_3d_rope: bool = True
    use_video_temporal_moe: bool = True
    num_video_encoder_layers: int = 4
    num_video_experts: int = 4
```

#### Multi-Scale Training
```python
    # Enable multi-scale
    use_multi_scale: bool = True
    
    # Image scales
    image_scales: Tuple = ((128,128), (192,192), (256,256), (320,320), (384,384), (448,448), (512,512))
    image_scale_probs: Tuple = (0.05, 0.10, 0.30, 0.25, 0.15, 0.10, 0.05)
    image_min_size: int = 128
    image_max_size: int = 512
    image_base_size: int = 256
    
    # Video scales
    video_scales: Tuple = ((128,128), (192,192), (256,256), (320,320), (384,384))
    video_scale_probs: Tuple = (0.10, 0.20, 0.35, 0.25, 0.10)
    video_min_size: int = 128
    video_max_size: int = 384
    video_base_size: int = 256
    
    # Frame scales
    video_frame_scales: Tuple = (8, 12, 16, 20, 24, 32)
    video_frame_scale_probs: Tuple = (0.10, 0.15, 0.30, 0.20, 0.15, 0.10)
    video_min_frames: int = 8
    video_max_frames: int = 32
    video_base_frames: int = 16
    
    # Strategy
    multi_scale_strategy: str = "random"  # "random", "progressive", "curriculum"
```

#### Generation Configuration
```python
    # Image generation
    enable_generation: bool = True
    generation_latent_channels: int = 4
    generation_base_channels: int = 128
    generation_inference_steps: int = 50
    generation_cfg_scale: float = 7.5
    generation_use_flow_matching: bool = True
    generation_num_experts: int = 4
    generation_use_dual_stream: bool = True
    
    # Video generation
    generation_video_cfg_scale: float = 7.5
    generation_video_use_flow_matching: bool = True
    generation_video_num_experts: int = 4
    generation_video_use_3d_rope: bool = True
    generation_video_use_temporal_moe: bool = True
```

#### Audio Configuration
```python
    # Audio settings
    audio_sample_rate: int = 16000
    audio_n_mels: int = 80
    audio_max_length: int = 1000
    audio_num_speakers: int = 256
    use_raw_waveform: bool = True
    audio_kv_lora_rank: int = 256
    audio_speaker_embed_dim: int = 256
    use_mas: bool = True
    use_in_context_audio_prompting: bool = True
```

#### LoRA Configuration
```python
    # LoRA settings
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: Tuple = (
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
    )
    train_lora_only: bool = False
    use_rslora: bool = True
    use_dora: bool = False
    lora_plus_lr_ratio: float = 4.0
```

### Usage

```python
from config import XoronConfig

# Default configuration
config = XoronConfig()

# Custom configuration
config = XoronConfig(
    hidden_size=2048,
    num_layers=24,
    num_experts=16,
)

# Print summary
config.print_config()

# Convert to dict
config_dict = config.to_dict()

# Load from dict
config = XoronConfig.from_dict(config_dict)
```

---

## 🏋️ TrainingConfig (Training Configuration)

### Overview

`TrainingConfig` defines all training hyperparameters and paths.

### Key Parameters

#### Paths (Auto-detected by Environment)
```python
@dataclass
class TrainingConfig:
    # Environment detection
    environment: str = field(default_factory=detect_environment)
    
    # Paths (auto-configured based on environment)
    model_path: str = ...      # Built model location
    temp_dir: str = ...        # Temporary files
    datasets_dir: str = ...    # Dataset cache
    output_dir: str = ...      # Checkpoints
    final_model_dir: str = ... # Final model
```

#### Dataset Settings
```python
    # Dataset limits
    max_per_epoch: int = 1000      # Max samples per epoch
    max_per_dataset: int = 50      # Max per dataset (prevents dominance)
    sample_repeat: int = 2         # Each sample shown N times
```

#### Training Hyperparameters
```python
    # Core settings
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 2
    warmup_ratio: float = 0.05
    max_seq_length: int = 1024
    max_grad_norm: float = 1.0
```

#### LoRA+ Settings
```python
    # LoRA+ (different LR for A and B matrices)
    use_lora_plus: bool = True
    lora_plus_lr_ratio: float = 4.0  # B learns 4x faster
```

#### Loss Weights
```python
    # Chain-of-thought
    cot_loss_weight: float = 1.5
    
    # Special token groups
    tool_loss_weight: float = 1.3
    anti_hallucination_loss_weight: float = 1.2
    code_exec_loss_weight: float = 1.2
    
    # Modality weights
    llm_loss_weight: float = 1.0
    image_diffusion_loss_weight: float = 0.1
    video_diffusion_loss_weight: float = 0.1
    asr_loss_weight: float = 0.1
    tts_loss_weight: float = 0.1
```

#### Device Settings
```python
    # Hardware
    device: str = "cuda"
    fp16: bool = True
    bf16: bool = False
    use_model_parallel: bool = True  # Multi-GPU
    
    # Memory optimization
    gradient_checkpointing: bool = True
    use_8bit_optimizer: bool = True
    empty_cache_freq: int = 100
```

### Environment Detection

The config automatically detects the runtime environment:

```python
def detect_environment() -> str:
    """
    Returns:
        'kaggle' - Running on Kaggle
        'colab' - Running on Google Colab
        'lightning' - Running on Lightning AI
        'local' - Running locally
    """
```

Each environment gets appropriate paths:

| Environment | Temp Dir | Output Dir | Final Model |
|-------------|----------|------------|-------------|
| Kaggle | /kaggle/tmp/xoron | /kaggle/tmp/xoron/checkpoints | /kaggle/working/xoron-final |
| Colab | /content/tmp | /content/xoron-checkpoints | /content/xoron-final |
| Lightning | /tmp/xoron | /teamspace/.../checkpoints | /teamspace/.../xoron-final |
| Local | ./tmp | ./xoron-checkpoints | ./xoron-final |

### Device Map for Multi-GPU

```python
def get_device_map(num_gpus: int) -> Dict[str, str]:
    """
    Create device map for Model Parallelism.
    
    2 GPU Layout:
    GPU 0: vision_encoder, video_encoder, audio_encoder, audio_decoder,
           waveform_decoder, projector, generator, video_generator
    GPU 1: llm, cross_attention, modality_markers
    """
```

---

## 📊 Dataset Configuration

### Overview

`dataset_config.py` defines all training datasets organized by category.

### Dataset Categories

```python
DATASET_CONFIGS = {
    # Text capabilities
    "code": [...],           # Code-Feedback, HumanEval, etc.
    "conversation": [...],   # Dolly, OpenAssistant, etc.
    "tool_use": [...],       # Function calling datasets
    "agentic": [...],        # AgentInstruct, WildChat, etc.
    
    # Reasoning
    "chain_of_thought": [...],  # Synthetic CoT
    "anti_hallucination": [...], # IDK, FactCheck, etc.
    
    # Agentic coding
    "document": [...],       # Document handling
    "fim": [...],            # Fill-in-the-middle
    "git_operations": [...], # Commits, diffs, issues
    "code_execution": [...], # Jupyter, shell
    
    # Multimodal
    "vision": [...],         # Image understanding
    "video": [...],          # Video understanding
    "generation": [...],     # Image/video generation prompts
    "voice_asr": [...],      # Speech-to-text
    "voice_tts": [...],      # Text-to-speech
}
```

### Dataset Entry Format

```python
{
    "name": "Code-Feedback",
    "path": "m-a-p/Code-Feedback",  # HuggingFace path
    "split": "train",
    "streaming": True,
    "config": "python",  # Optional config name
}

# Local datasets
{
    "name": "Synth-CoT",
    "path": "synth/data/cot_dataset.jsonl",
    "split": "train",
    "streaming": False,
    "local": True,
    "format": "jsonl",
}
```

### Modality Groups

```python
MODALITY_GROUPS = {
    'text': ['code', 'conversation', 'tool_use', 'agentic', ...],
    'image': ['vision', 'generation'],
    'video': ['video', 'video_generation'],
    'audio': ['voice_asr', 'voice_tts'],
}

CATEGORY_TO_MODALITY = {
    'code': 'text',
    'vision': 'image',
    'video': 'video',
    'voice_asr': 'audio',
    ...
}
```

### Filtering Functions

```python
def filter_datasets_by_modalities(configs, modalities):
    """Filter datasets to only include specified modalities."""
    
def filter_datasets_by_categories(configs, categories):
    """Filter datasets to only include specified categories."""
    
def get_finetune_datasets(configs, mode):
    """Get datasets for fine-tuning mode (text, image, video, audio, all)."""
```

---

## 🏷️ Special Tokens

### Overview

`special_tokens.py` defines all special tokens for structured generation.

### Token Categories

#### Sequence Control
```python
SPECIAL_TOKENS = {
    "bos": "<|bos|>",
    "eos": "<|eos|>",
    "pad": "<|pad|>",
    "prompt_start": "<|prompt|>",
    "response_start": "<|response|>",
    ...
}
```

#### Conversation
```python
    "system_start": "<|system|>",
    "user_start": "<|user|>",
    "assistant_start": "<|assistant|>",
    ...
```

#### Memory & Context
```python
    "memory_start": "<|memory|>",
    "working_memory_start": "<|working_memory|>",
    "summary_start": "<|summary|>",
    ...
```

#### Fill-in-the-Middle (FIM)
```python
    "fim_prefix": "<|fim_prefix|>",
    "fim_middle": "<|fim_middle|>",
    "fim_suffix": "<|fim_suffix|>",
```

#### Git Operations
```python
    "commit_before": "<|commit_before|>",
    "commit_after": "<|commit_after|>",
    "diff_start": "<|diff|>",
    "diff_add": "<|diff_add|>",
    "diff_del": "<|diff_del|>",
    ...
```

#### Code Execution
```python
    "jupyter_code": "<|jupyter_code|>",
    "jupyter_output": "<|jupyter_output|>",
    "exec_start": "<|exec|>",
    "exec_result": "<|exec_result|>",
    "exec_error": "<|exec_error|>",
    ...
```

#### File Operations
```python
    "add_file": "<|add_file|>",
    "delete_file": "<|delete_file|>",
    "edit_file": "<|edit_file|>",
    "read_file": "<|read_file|>",
    ...
```

#### Reasoning Tokens
```python
REASONING_TOKENS = {
    "think_start": "<|think|>",
    "think_end": "<|/think|>",
    "plan_start": "<|plan|>",
    "step": "<|step|>",
    "verify": "<|verify|>",
    ...
}
```

#### Uncertainty Tokens
```python
UNCERTAINTY_TOKENS = {
    "uncertain": "<|uncertain|>",
    "confident": "<|confident|>",
    "needs_verification": "<|needs_verify|>",
    "idk": "<|idk|>",
    ...
}
```

#### Multimodal Tokens
```python
    "image_start": "<|image|>",
    "video_start": "<|video|>",
    "audio_start": "<|audio|>",
    "gen_image": "<|gen_image|>",
    "gen_video": "<|gen_video|>",
    "gen_audio": "<|gen_audio|>",
    ...
```

### Helper Functions

```python
def get_special_tokens_list() -> List[str]:
    """Get all special tokens as a list."""

def get_reasoning_tokens() -> Dict[str, str]:
    """Get reasoning-related tokens."""

def strip_hidden_tokens(text: str) -> str:
    """Remove hidden/internal tokens from output."""
```

---

## 💬 Chat Template

### Overview

`chat_template.py` defines the Jinja2 template for formatting conversations.

### Template Structure

```jinja2
{% for message in messages %}
{% if message['role'] == 'system' %}
<|system|>
{{ message['content'] }}
<|/system|>
{% elif message['role'] == 'user' %}
<|user|>
{{ message['content'] }}
<|/user|>
{% elif message['role'] == 'assistant' %}
<|assistant|>
{{ message['content'] }}
<|/assistant|>
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
<|assistant|>
{% endif %}
```

### Usage

```python
from config import apply_chat_template_to_tokenizer

# Apply to tokenizer
tokenizer = apply_chat_template_to_tokenizer(tokenizer)

# Format messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
```

---

## 🔗 Related Documentation

- [Training Documentation](../training/README.md) - How configs are used in training
- [Data Documentation](../data/README.md) - Dataset loading details
- [Model Documentation](../models/llm.md) - Model architecture details

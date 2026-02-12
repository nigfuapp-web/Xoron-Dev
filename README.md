# 🚀 Xoron-Dev: Unified Multimodal AI Model

<div align="center">

![Xoron-Dev Logo](https://img.shields.io/badge/Xoron--Dev-MultiMoE-blue?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch)
![Version](https://img.shields.io/badge/Version-2.2-purple?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-482%20Passing-brightgreen?style=for-the-badge)

**A unified multimodal AI model that understands and generates text, images, video, and audio with multi-scale training support.**

[Features](#-features) | [Architecture](#%EF%B8%8F-architecture) | [Installation](#-installation) | [Usage](#-usage) | [Training](#%EF%B8%8F-training) | [Export](#-export)

</div>

---

## 🏗️ Architecture Overview

<div align="center">
<img src="./assets/xoron_architecture.svg" alt="Xoron-Dev Architecture" width="100%">
</div>

### LLM Backbone (12 Layers, 1024d, 16 Heads)
- **Ring Attention** - 128K context with 4096 chunk size
- **Aux-Lossless MoE** - 8 experts, top-2 routing, no auxiliary loss
- **Configurable Shared Expert** - Optional always-active expert for common knowledge
- **Qwen2.5 Tokenizer** - 151K vocab size

### Vision Encoder
- **SigLIP SO400M** - 384×384 native, multi-scale 128-512px
- **TiTok 1D Tokenization** - 576 patches → 256 tokens with cross-attention
- **Dual-Stream Attention** - SD3/Flux-style symmetric processing with 2D-RoPE
- **Multiple Projector Types** - Perceiver Resampler, Spatial-Aware, C-Abstractor

### Video Encoder
- **3D-RoPE** - Spatiotemporal (x, y, t) positional encodings
- **VidTokTokenizer** - Full 3D VAE (Microsoft VidTok architecture)
  - Efficient 2D+1D architecture (separates spatial and temporal processing)
  - AlphaBlender for temporal blending
  - Supports continuous (KL) and discrete (FSQ) tokenization
  - 4x temporal, 8x8 spatial compression (4x8x8 total)
- **Temporal MoE** - 4 experts for motion patterns with expert-choice routing
- **3D Causal Transformer** - Factorized spatio-temporal attention
- **Continuous-scale: 8-24 frames** at 128-320px resolution

### Audio System
- **Raw Waveform Tokenizer** - Direct audio processing at 16kHz with RVQ (4 codebooks)
- **Conformer Encoder** - With Rotary Multi-Head Latent Attention (RMLA)
- **BigVGAN-style Waveform Decoder** - Snake activation + Multi-Receptive Field Fusion
- **Monotonic Alignment Search (MAS)** - Soft/Hard alignment for text-audio
- **Zero-Shot Voice Cloning** - In-context audio prompting with speaker embeddings

### Image Generation
- **MoE-DiT** - Diffusion Transformer with 4 experts and SwiGLU activation
- **Flow Matching** - Replaces DDPM for faster convergence
- **Dual-Stream Self-Attention** - 2D-RoPE for spatial awareness
- **ImageVAE** - For latent space encoding/decoding
- **Continuous-scale: 192-384px** output resolution, CFG scale 7.5

### Video Generation
- **3D Causal Transformers** - Factorized spatial + temporal attention
- **Flow Matching Scheduler** - Smooth frame transitions with log-SNR weighting
- **Temporal MoE** - 4 experts with load balancing loss
- **VideoVAE3D** - 3D VAE for video latent space
- **Continuous-scale: 8-24 frames @ 128-320px**

---

## 🌟 Features

### 🧠 **Multimodal Understanding**
| Modality | Encoder | Input Size (Continuous-Scale) | Output |
|----------|---------|-------------------------------|--------|
| Vision | SigLIP SO400M + TiTok (256 tokens) + Dual-Stream RoPE2D | 128-384px | 64 tokens |
| Video | 3D-RoPE + VidTok (3D VAE) + Temporal MoE | 8-24 frames @ 128-320px | Latent space |
| Audio | Raw Waveform Tokenizer + Conformer + RMLA (MLA-style KV compression) | 16kHz, up to 10s | Variable |
| Text | Qwen2.5 Tokenizer (151K vocab) | 128K context | - |

### 🎨 **Multimodal Generation**
| Output | Architecture | Resolution (Continuous-Scale) |
|--------|--------------|-------------------------------|
| Text | MoE LLM (8 experts, configurable shared) + Chain-of-Thought | 128K tokens |
| Image | MoE-DiT (4 experts, SwiGLU) + Flow Matching + 2D-RoPE | 192-384px (50 steps) |
| Video | VideoUNet3D + Flow Matching + Temporal MoE | 8-24 frames @ 128-320px |
| Audio | BigVGAN-style Decoder + MAS (soft/hard) + Speaker Encoder | 16kHz |

### ⚡ **Training Features**
- **Continuous-Scale Training**: Adaptive strategy samples ANY scale in range for optimal memory usage
- **Mixture of Experts**: 8 experts + configurable shared, top-2 routing, aux-lossless routing
- **Expert Choice MoE**: Alternative routing where experts select tokens
- **LoRA+/rsLoRA/DoRA**: r=32, α=64, B matrix learns 4× faster (LoRA+ ratio)
- **Ring Attention**: Memory-efficient 128K context (4096 chunk size)
- **Flow Matching**: Log-SNR weighting for superior generation quality
- **Gradient Checkpointing**: Memory optimization for encoders/decoders
- **Multi-GPU**: Model parallelism for 2× T4 GPUs (Kaggle)

### 🛠️ **Agentic Capabilities**
- **250+ Special Tokens** for structured outputs
- **Tool Calling**: Function invocation with `<|tool_call|>`, `<|tool_result|>`
- **Code Execution**: Shell, Python, Jupyter with `<|exec|>`, `<|jupyter|>`
- **File Operations**: Create, edit, delete with `<|file_create|>`, `<|file_edit|>`
- **Anti-Hallucination**: `<|uncertain|>`, `<|cite|>`, `<|confidence_*|>`
- **Chain-of-Thought**: `<|think|>`, `<|plan|>`, `<|critique|>`

---

## 📂 Project Structure

```
Xoron-Dev/
├── 📁 models/               # Core model implementations
│   ├── xoron.py             # Main XoronMultimodalModel class
│   ├── 📁 llm/              # MoE-LLM backbone
│   │   └── moe_llama.py     # MoE LLaMA with Ring Attention
│   ├── 📁 encoders/         # Input encoders
│   │   ├── vision.py        # SigLIP + TiTok + Dual-Stream + RoPE2DEncoder
│   │   ├── video.py         # 3D-RoPE + VidTokTokenizer (3D VAE) + Temporal MoE + Causal3DTransformer
│   │   └── audio.py         # RawWaveformTokenizer + Conformer + RMLA + MAS
│   ├── 📁 generators/       # Output generators
│   │   ├── image.py         # MoE-DiT + Flow Matching + ImageVAE
│   │   └── video.py         # VideoUNet3D + Temporal MoE + VideoVAE3D
│   └── 📁 components/       # Shared components
│       ├── moe.py           # MoE (standard + expert-choice routing)
│       ├── attention.py     # Ring, Flash, Cross attention
│       ├── projectors.py    # Perceiver, Spatial-Aware, C-Abstractor
│       └── lora.py          # LoRA/rsLoRA/DoRA/LoRA+
│
├── 📁 config/               # Configuration
│   ├── model_config.py      # XoronConfig dataclass
│   ├── training_config.py   # TrainingConfig
│   ├── dataset_config.py    # 66+ dataset definitions
│   ├── special_tokens.py    # 250+ special tokens
│   └── chat_template.py     # Chat formatting templates
│
├── 📁 training/             # Training utilities
│   ├── trainer.py           # XoronTrainer with weighted loss
│   └── utils.py             # Per-modality training steps
│
├── 📁 data/                 # Data processing
│   ├── dataset.py           # TrueStreamingDataset
│   ├── formatters.py        # 20+ format functions
│   └── processors.py        # Image/Video/Audio processing
│
├── 📁 synth/                # Synthetic data generation
│   ├── generator.py         # Main generator module
│   ├── templates.py         # Data generation templates
│   └── quality_utils.py     # Quality validation utilities
│
├── 📁 export/               # Model export
│   ├── onnx_export.py       # ONNX with quantization
│   └── gguf_export.py       # GGUF for llama.cpp
│
├── 📁 tests/                # Comprehensive test suite (482 tests)
│   ├── 📁 config/           # Configuration tests
│   ├── 📁 models/           # Model component tests
│   ├── 📁 data/             # Data processing tests
│   ├── 📁 training/         # Training utility tests
│   └── 📁 utils/            # Utility function tests
│
├── build.py                 # Main CLI for build/train/export
├── load.py                  # Model loading utilities
└── setup.py                 # Interactive configuration
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 16GB+ VRAM recommended

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/nigfuapp-web/Xoron-Dev.git
cd Xoron-Dev

# Install requirements
pip install -r requirements.txt
```

### Requirements

```
torch
torchvision
torchaudio
transformers
datasets
accelerate
sentencepiece
Pillow
timm
opencv-python-headless
soundfile
librosa
safetensors
onnx
onnxruntime
bitsandbytes
optimum
tqdm
numpy
requests
huggingface_hub
jinja2
```

---

## 💻 Usage

### Quick Start

```python
from models.xoron import XoronMultimodalModel
from config import XoronConfig

# Create model
config = XoronConfig()
model = XoronMultimodalModel(config)

# Load from HuggingFace
model = XoronMultimodalModel.from_pretrained("Backup-bdg/Xoron-Dev-MultiMoe")
```

### Interactive Setup

```bash
# Run interactive configuration tool
python setup.py
```

### CLI Training

```bash
# Build and train new model
python build.py --build

# Train on specific modality
python build.py --build --text
python build.py --build --image
python build.py --build --video
python build.py --build --voice

# Load from HuggingFace and train
python build.py --hf --text

# Resume training
python build.py --resume ./checkpoints/latest

# Export to ONNX/GGUF
python build.py --build --onnx --gguf
```

---

## 🏋️ Training

### Training Configuration

```python
from config import TrainingConfig

config = TrainingConfig(
    batch_size=1,
    gradient_accumulation_steps=64,
    learning_rate=2e-4,
    num_epochs=2,
    max_seq_length=1024,
    
    # Loss weights
    llm_loss_weight=1.0,
    cot_loss_weight=1.5,  # Higher for reasoning
    image_diffusion_loss_weight=0.1,
    video_diffusion_loss_weight=0.1,
    
    # LoRA+ settings
    use_lora_plus=True,
    lora_plus_lr_ratio=16.0,
)
```

### Weighted Loss for Token Groups

The trainer applies higher loss weights to important token groups:

| Token Group | Weight | Purpose |
|-------------|--------|---------|
| Chain-of-Thought | 1.5x | Reasoning tokens (`<\|think\|>`, `<\|plan\|>`) |
| Tool Calling | 1.3x | Function calls (`<\|tool_call\|>`, `<\|function_name\|>`) |
| Anti-Hallucination | 1.2x | Uncertainty (`<\|uncertain\|>`, `<\|cite\|>`) |
| Code Execution | 1.2x | Execution (`<\|exec\|>`, `<\|jupyter\|>`) |

### Synthetic Datasets

<div align="center">
<img src="./assets/datasets_overview.svg" alt="Xoron-Dev Datasets Overview" width="100%">
</div>

Generate training data with:

```python
from synth import generate_all_datasets

generate_all_datasets('./synth/data', samples_per_type=2000)
```

**35 Dataset Types across 6 Categories:**
- **Shell & Execution**: Shell commands, errors, timeouts, multi-step workflows
- **Code & Programming**: Python scripts, Jupyter notebooks, debugging, file operations
- **Git & Version Control**: Commits, diffs, issues, repository context
- **System Administration**: Docker, databases, web servers, SSH, monitoring
- **Anti-Hallucination**: Uncertainty, fact-checking, citations, self-correction
- **Documents & Reasoning**: Document processing, chain-of-thought

---

## 📦 Export

### ONNX Export

```bash
# Export with 4-bit quantization
python build.py --build --onnx --quant-bits 4
```

```python
from export import export_to_onnx

export_to_onnx(
    model, config, output_dir,
    quantize=True,
    quantize_bits=4
)
```

### GGUF Export

```bash
# Export for llama.cpp
python build.py --build --gguf --gguf-quant q4_k_m
```

```python
from export import export_to_gguf

export_to_gguf(
    model, config, output_dir,
    quant_type='q4_k_m'  # q4_0, q4_k_m, q5_k_m, q8_0, f16
)
```

---

## 🔧 Configuration Reference

### XoronConfig (v2.2)

```python
@dataclass
class XoronConfig:
    # LLM Architecture
    hidden_size: int = 1024
    num_layers: int = 12
    num_heads: int = 16
    intermediate_size: int = 2048
    vocab_size: int = 151643
    max_position_embeddings: int = 131072  # 128K
    tie_word_embeddings: bool = True
    rms_norm_eps: float = 1e-6
    
    # Ring Attention for 128K+ context
    use_ring_attention: bool = True
    ring_attention_chunk_size: int = 4096
    
    # MoE (Aux-Lossless with configurable shared expert)
    use_moe: bool = True
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_layer_freq: int = 2
    use_shared_expert: bool = True
    use_aux_lossless: bool = True
    
    # Vision Encoder (SOTA)
    vision_model_name: str = "google/siglip-so400m-patch14-384"
    num_vision_tokens: int = 64
    use_vision_dual_stream: bool = True
    use_vision_titok: bool = True
    num_vision_titok_tokens: int = 256
    
    # Video Encoder (SOTA) - VidTok 3D VAE
    use_video_3d_rope: bool = True
    use_video_temporal_moe: bool = True
    use_video_vidtok: bool = True
    vidtok_latent_channels: int = 4
    vidtok_temporal_compression: int = 4
    vidtok_spatial_compression: int = 8
    vidtok_causal: bool = True
    vidtok_use_fsq: bool = False  # KL (continuous) vs FSQ (discrete)
    num_video_encoder_layers: int = 4
    
    # Continuous-Scale Training (SOTA)
    use_multi_scale: bool = True
    use_continuous_scale: bool = True  # Samples ANY scale in range
    multi_scale_strategy: str = "adaptive"  # "uniform", "gaussian", "adaptive"
    
    # Image continuous-scale settings
    image_min_size: int = 128
    image_max_size: int = 384
    image_base_size: int = 256
    image_size_step: int = 32
    
    # Video continuous-scale settings
    video_min_size: int = 128
    video_max_size: int = 320
    video_base_size: int = 192
    video_size_step: int = 32
    
    # Video temporal continuous-scale settings
    video_min_frames: int = 8
    video_max_frames: int = 24
    video_base_frames: int = 16
    video_frame_step: int = 4
    
    # Image Generation (MoE-DiT + Flow Matching)
    enable_generation: bool = True
    generation_cfg_scale: float = 7.5
    generation_use_flow_matching: bool = True
    generation_use_dual_stream: bool = True
    generation_num_experts: int = 4
    
    # Video Generation (3D Causal + Temporal MoE)
    generation_video_use_flow_matching: bool = True
    generation_video_use_3d_rope: bool = True
    generation_video_num_experts: int = 4
    generation_video_use_temporal_moe: bool = True
    
    # LoRA variants
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    use_rslora: bool = True
    
    # Attention
    use_cross_attention: bool = True
    cross_attention_layers: int = 4
    use_flash_attention: bool = True
```

### Continuous-Scale Training

Continuous-scale training samples ANY resolution within min/max bounds (not discrete scales):

| Type | Range | Base | Step |
|------|-------|------|------|
| **Image** | 128-384px | 256px | 32px |
| **Video** | 128-320px | 192px | 32px |
| **Frames** | 8-24 | 16 | 4 |

The **adaptive** strategy adjusts scale ranges based on OOM history for optimal memory usage.

---

## 📈 Performance

| Configuration | VRAM Required |
|--------------|---------------|
| Training (batch=1, grad_accum=64) | ~24GB |
| Inference (FP16) | ~8GB |
| Inference (INT4) | ~4GB |

### Supported Environments

- ✅ Local GPU (NVIDIA)
- ✅ Google Colab
- ✅ Kaggle Notebooks
- ✅ Lightning.ai
- ✅ Multi-GPU (Model Parallelism)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [SigLIP](https://github.com/google-research/big_vision)
- [DeepSeek MoE](https://github.com/deepseek-ai/DeepSeek-MoE)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)

---

<div align="center">

**Built with ❤️ for the AI community**

</div>

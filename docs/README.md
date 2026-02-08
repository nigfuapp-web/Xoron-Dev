# 📚 Xoron-Dev Documentation

Welcome to the comprehensive documentation for **Xoron-Dev**, a state-of-the-art multimodal AI model that unifies text, image, video, and audio understanding and generation within a single architecture.

## 🗂️ Documentation Structure

This documentation is organized into modular sections, each covering a specific aspect of the Xoron-Dev architecture:

### Core Architecture

| Module | Description | Link |
|--------|-------------|------|
| **LLM** | Mixture of Experts Language Model with MLA, Ring Attention, YaRN | [models/llm.md](models/llm.md) |
| **Encoders** | Vision, Video, and Audio encoders with SOTA features | [models/encoders.md](models/encoders.md) |
| **Generators** | Image and Video generation with Flow Matching | [models/generators.md](models/generators.md) |
| **Components** | Attention, LoRA, MoE, and Projector modules | [models/components.md](models/components.md) |

### Configuration & Training

| Module | Description | Link |
|--------|-------------|------|
| **Config** | Model, training, and dataset configurations | [config/README.md](config/README.md) |
| **Training** | Trainer class and training utilities | [training/README.md](training/README.md) |
| **Data** | Dataset loading and processing | [data/README.md](data/README.md) |

### Utilities & Export

| Module | Description | Link |
|--------|-------------|------|
| **Utils** | Device detection, logging, and helpers | [utils/README.md](utils/README.md) |
| **Export** | GGUF and ONNX export functionality | [export/README.md](export/README.md) |
| **Synth** | Synthetic data generation pipeline | [synth/README.md](synth/README.md) |

### Deployment

| Resource | Description | Link |
|----------|-------------|------|
| **HuggingFace** | Model card and deployment info | [huggingface/hf.md](huggingface/hf.md) |

---

## 🏗️ Architecture Overview

Xoron-Dev is built on a unified multimodal architecture that enables:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           XORON-DEV ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Vision    │  │    Video    │  │    Audio    │  │    Text     │        │
│  │   Encoder   │  │   Encoder   │  │   Encoder   │  │  Tokenizer  │        │
│  │  (SigLIP-2) │  │  (3D-RoPE)  │  │ (Conformer) │  │  (Qwen2.5)  │        │
│  │  + TiTok    │  │  + Temporal │  │  + RMLA     │  │             │        │
│  │  + 2D-RoPE  │  │    MoE      │  │  + MAS      │  │             │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │               │
│         └────────────────┴────────────────┴────────────────┘               │
│                                   │                                        │
│                          ┌────────▼────────┐                               │
│                          │   Multimodal    │                               │
│                          │   Projector     │                               │
│                          │  (Perceiver)    │                               │
│                          └────────┬────────┘                               │
│                                   │                                        │
│                          ┌────────▼────────┐                               │
│                          │  Cross-Attention │                              │
│                          │   Fusion Layers  │                              │
│                          │   (4 layers)     │                              │
│                          └────────┬────────┘                               │
│                                   │                                        │
│                    ┌──────────────▼──────────────┐                         │
│                    │      MoE LLM Backbone       │                         │
│                    │  ┌─────────────────────┐    │                         │
│                    │  │ 12 Layers, 1024 dim │    │                         │
│                    │  │ 8 Experts + Shared  │    │                         │
│                    │  │ Ring Attention 128K │    │                         │
│                    │  │ MLA + YaRN/LongRoPE │    │                         │
│                    │  │ Aux-Lossless MoE    │    │                         │
│                    │  └─────────────────────┘    │                         │
│                    └──────────────┬──────────────┘                         │
│                                   │                                        │
│         ┌─────────────────────────┼─────────────────────────┐              │
│         │                         │                         │              │
│  ┌──────▼──────┐          ┌───────▼───────┐         ┌───────▼───────┐      │
│  │    Image    │          │     Video     │         │     Audio     │      │
│  │  Generator  │          │   Generator   │         │    Decoder    │      │
│  │  (MoE-DiT)  │          │  (3D Causal)  │         │  (BigVGAN)    │      │
│  │  + Flow     │          │  + Flow       │         │  + MAS        │      │
│  │  Matching   │          │  Matching     │         │  + Zero-Shot  │      │
│  └─────────────┘          └───────────────┘         └───────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🔑 Key Features

### Language Model (LLM)
- **MLA (Multi-Head Latent Attention)**: Compressed KV cache for memory efficiency
- **Ring Attention**: Distributed 128K+ context processing
- **YaRN/LongRoPE**: Superior long-context extrapolation
- **Aux-Lossless MoE**: 8 experts + 1 shared, no auxiliary loss needed

### Vision Understanding
- **SigLIP-2 Backbone**: State-of-the-art visual features
- **TiTok Tokenization**: 1D tokenization with 256 compressed tokens
- **2D-RoPE**: Flexible aspect ratio handling
- **Dual-Stream Attention**: Symmetric processing

### Video Understanding
- **3D-RoPE**: Position encoding for (x, y, t) coordinates
- **Temporal MoE**: Motion-aware expert routing
- **3D Causal Attention**: Temporal understanding
- **Multi-scale**: 8-32 frames at 128-384px

### Audio Processing
- **Raw Waveform Tokenizer**: Direct audio processing (no mel spectrograms)
- **Conformer Encoder**: With RMLA for ASR
- **BigVGAN Decoder**: Direct waveform output (no vocoder needed)
- **Zero-Shot Voice Cloning**: Speaker embedding extraction
- **MAS (Monotonic Alignment Search)**: Fluid text-to-audio alignment

### Image Generation
- **MoE-DiT**: Diffusion Transformer with 4 MoE experts
- **Flow Matching**: Superior to DDPM
- **Dual-Stream Attention**: SD3/Flux-style symmetric processing
- **Multi-scale**: 256-512px output

### Video Generation
- **3D Causal Transformers**: Autoregressive video generation
- **Flow Matching**: High-quality temporal coherence
- **Temporal Expert Routing**: Motion-aware processing
- **Multi-scale**: 8-32 frames at 128-384px

## 📊 Model Specifications

| Component | Specification |
|-----------|--------------|
| **Hidden Size** | 1024 |
| **Layers** | 12 |
| **Attention Heads** | 16 |
| **MoE Experts** | 8 + 1 Shared |
| **Experts per Token** | 2 (top-2 routing) |
| **Context Length** | 128K positions |
| **Vocabulary** | 151,643 tokens (Qwen2.5) |
| **Vision Encoder** | SigLIP-so400m-patch14-384 |
| **Audio Sample Rate** | 16kHz |

## 🚀 Quick Start

```python
from models.xoron import XoronMultimodalModel
from config import XoronConfig

# Initialize model
config = XoronConfig()
model = XoronMultimodalModel(config)

# Text generation
output = model.generate_text(
    input_ids=tokenized_input,
    max_new_tokens=100
)

# Image understanding
output = model.forward(
    input_ids=text_tokens,
    images=image_tensor
)

# Image generation
image = model.generate_image(
    prompt_embeds=text_embeddings,
    height=512,
    width=512
)
```

## 📖 Reading Order

For newcomers, we recommend reading the documentation in this order:

1. **[Config Documentation](config/README.md)** - Understand the configuration system
2. **[LLM Documentation](models/llm.md)** - Core language model architecture
3. **[Components Documentation](models/components.md)** - Building blocks (attention, LoRA, MoE)
4. **[Encoders Documentation](models/encoders.md)** - Input processing
5. **[Generators Documentation](models/generators.md)** - Output generation
6. **[Training Documentation](training/README.md)** - How to train the model
7. **[Data Documentation](data/README.md)** - Dataset handling

## 🔧 Development

### Running Tests
```bash
python -m pytest tests/ -v
```

### Building the Model
```bash
python build.py
```

### Training
```bash
python auto.py
```

### Interactive Setup
```bash
python setup.py
```

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please read the documentation thoroughly before submitting PRs.

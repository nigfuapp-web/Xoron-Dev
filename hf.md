---
language:
- en
license: mit
library_name: transformers
tags:
- multimodal
- moe
- text-to-image
- image editing
- image to video
- text-to-video
- video editing
- text-to-speech
- speech-to-text
- image-to-text
- video-to-text
- agentic
- tool-use
- flow-matching
- 3d-rope
- titok
- dual-stream-attention
- zero-shot-voice-cloning
pipeline_tag: any-to-any
inference: false
datasets:
# === Code & Programming ===
- m-a-p/Code-Feedback
- iamtarun/python_code_instructions_18k_alpaca
- codeparrot/codeparrot-clean
- bigcode/humanevalpack
- loubnabnl/github-jupyter-code-to-text
- saurabh5/rlvr-code-data-Swift
- finbarr/rlvr-code-data-swift-code-edit
- ExAi/Code-Golang-QA-2k
- smcleod/golang-coder
# === Conversation & Agentic ===
- databricks/databricks-dolly-15k
- OpenAssistant/oasst1
- HuggingFaceH4/no_robots
- Open-Orca/OpenOrca
- abhi227070/conversation-to-summarization-dataset
- allenai/WildChat-1M
- THUDM/AgentInstruct
- glaiveai/glaive-code-assistant-v2
- stingning/ultrachat
- RyokoAI/ShareGPT52K
- AlicanKiraz0/Agentic-Chain-of-Thought-Coding-SFT-Dataset
# === Tool Use ===
- Locutusque/function-calling-chatml
- driaforall/pythonic-function-calling
- argilla/Synth-APIGen-v0.1
- interstellarninja/tool-calls-singleturn
- interstellarninja/tool-calls-multiturn
# === Vision (Image & Video) ===
- Naveengo/flickr8k
- ybelkada/football-dataset
- jmhessel/newyorker_caption_contest
- derek-thomas/ScienceQA
- HuggingFaceM4/WebSight
- lmms-lab/Video-MME
- MBZUAI/VideoInstruct-100K
# === Generation (Prompts & Media) ===
- Gustavosta/Stable-Diffusion-Prompts
- FredZhang7/stable-diffusion-prompts-2.47M
- succinctly/midjourney-prompts
- osunlp/MagicBrush
- timbrooks/instructpix2pix-clip-filtered
- Rapidata/sora-video-generation-physics-likert-scoring
- Rapidata/sora-video-generation-style-likert-scoring
- Rapidata/sora-video-generation-alignment-likert-scoring
- Rapidata/text-2-video-human-preferences
- Rapidata/text-2-video-human-preferences-sora-2
- TempoFunk/webvid-10M
- multimodalart/panda-70m
- nkp37/OpenVid-1M
- WenhaoWang/VidProM
- WenhaoWang/TIP-I2V
- jovianzm/img2vid-pexels-350k
- TencentARC/MiraData
- APRIL-AIGC/UltraVideo
- Mutonix/Vript
- Rapidata/image-to-video-human-preference-seedance-1-pro
# === Audio ===
- openslr/librispeech_asr
- blabble-io/libritts_r
- parler-tts/mls_eng_10k
- MikhailT/hifi-tts
# === File Ops ===
- renjiepi/medium_20000-file_operations_n100k1
---

# 🚀 Xoron-Dev: State-of-the-Art Multimodal MoE

<div align="center">

![Xoron-Dev Logo](https://img.shields.io/badge/Xoron--Dev-MultiMoE-blue?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Params](https://img.shields.io/badge/Parameters-1.5B_MoE-yellow?style=for-the-badge)
![Context](https://img.shields.io/badge/Context-128K-red?style=for-the-badge)

</div>

**Xoron-Dev** is a unified, multimodal AI model designed to understand and generate text, images, video, and audio within a single architecture. It leverages a **Mixture of Experts (MoE)** backbone with DeepSeek-style shared expert isolation and integrates SOTA encoders (SigLIP-2 with TiTok + Dual-Stream Attention) and generators (MoE-DiT with Flow Matching) for comprehensive any-to-any capabilities.

## 🌟 Model Highlights

* **Architecture:** Mixture of Experts (8 Experts + 1 Shared, top-2 routing) with Ring Attention and Aux-Lossless routing.
* **Vision Encoder:** SigLIP-2 (384px) with **TiTok-style 1D tokenization**, **Dual-Stream Attention**, and **2D-RoPE** for images; **3D-RoPE** + **Temporal MoE** for video (up to 16 frames).
* **Image Generation:** **MoE-DiT** (Diffusion Transformer with MoE) using **Flow Matching**, **2D-RoPE**, and **Symmetric Dual-Stream Attention** (SD3/Flux-style).
* **Video Generation:** **3D Causal Transformers** with **Flow Matching**, **3D-RoPE** for (x,y,t) positions, and **Temporal Expert Routing**.
* **Audio (Speech-to-Speech):** **Conformer encoder with RMLA** and **Raw Waveform Tokenizer** for ASR; **Direct waveform decoder** (no vocoder needed!) with **MAS** for TTS; **Zero-Shot Speaker Cloning** with In-Context Audio Prompting. Talk to it, and it talks back!
* **Agentic:** Trained for tool calling, file operations, and code execution with uncertainty estimation.
* **Context:** Efficient 128K context using Ring Attention (4096 chunk size).
* **Fine-tuning:** LoRA variants including **rsLoRA**, **DoRA**, and **LoRA+** with configurable learning rate ratio.
* **Multimodal Fusion:** Cross-Attention layers (4 layers, 8 heads) for deep multimodal integration.
* **Performance:** Flash Attention support with FP16-native numerical stability.

---

## 🔬 Architecture Deep Dive

### 🧠 LLM Backbone (MoE)
| Component | Specification |
|-----------|--------------|
| Hidden Size | 1024 |
| Layers | 12 |
| Attention Heads | 16 |
| MoE Experts | 8 + 1 Shared (DeepSeek-style isolation) |
| Experts per Token | 2 (top-2 routing) |
| MoE Layer Frequency | Every 2 layers |
| Routing | Aux-Lossless MoE routing |
| Context Length | 128K positions |
| Attention | Ring Attention (4096 chunk) + Flash Attention |
| Tokenizer | Qwen2.5 (151,643 vocab) |

### 👁️ Vision Encoder (SigLIP-2 + SOTA Extensions)
| Feature | Description |
|---------|-------------|
| Base Model | `google/siglip-so400m-patch14-384` |
| Input Resolution | 384×384 |
| TiTok Tokenization | 1D tokenization with 256 compressed tokens |
| Dual-Stream Attention | 2 symmetric dual-stream layers |
| Position Encoding | 2D-RoPE |
| Output Tokens | 64 tokens per image |

### 🎬 Video Encoder (3D Causal Transformers)
| Feature | Description |
|---------|-------------|
| Max Frames | 16 frames |
| Position Encoding | **3D-RoPE** for (x, y, t) coordinates |
| Attention | 3D Causal Self-Attention |
| Expert Routing | **Temporal MoE** (4 experts, temporally-aware) |
| Encoder Layers | 4 layers |

### 🎨 Image Generation (MoE-DiT + Flow Matching)
| Feature | Description |
|---------|-------------|
| Architecture | **MoE-DiT** (Diffusion Transformer with MoE) |
| Scheduler | **Flow Matching** (not DDPM) |
| Output Resolution | 384×384 |
| Position Encoding | 2D-RoPE |
| Attention | **Symmetric Dual-Stream Attention** (SD3/Flux-style) |
| MoE Experts | 4 experts in DiT blocks |
| Inference Steps | 50 steps |
| Guidance Scale | 7.5 (CFG) |

### 📹 Video Generation (3D Causal + Flow Matching)
| Feature | Description |
|---------|-------------|
| Output Resolution | 256×256 |
| Output Frames | 16 frames (default), up to 32 frames (max capacity) |
| Scheduler | **Flow Matching** |
| Position Encoding | **3D-RoPE** for (x, y, t) |
| Attention | Factorized Spatial-Temporal (3D Causal) |
| Expert Routing | **Temporal MoE** (4 experts) |
| Guidance Scale | 7.5 (CFG) |

### 🎤 Audio (Speech-to-Speech with RMLA + MAS + Zero-Shot Cloning)
| Feature | Description |
|---------|-------------|
| Sample Rate | 16kHz |
| **Encoder (ASR)** | **Raw Waveform Tokenizer** → Conformer blocks with **RMLA** |
| **Waveform Decoder** | **Direct audio output** - no external vocoder needed! |
| KV Compression | LoRA-style KV compression (rank 256) |
| Decoder Alignment | **MAS** (Monotonic Alignment Search) for text-to-audio alignment |
| Voice Cloning | **Zero-Shot Speaker Cloning** with speaker embedding (256-dim) |
| In-Context Prompting | Enabled for voice cloning from reference audio |

### 🗣️ Speech-to-Speech API
The model provides three main methods for voice interaction:

| Method | Description |
|--------|-------------|
| `model.listen(audio)` | Encode speech to embeddings (ASR) |
| `model.speak(text)` | Generate playable audio from text (TTS) |
| `model.listen_and_respond(audio)` | Full conversation: listen → think → speak back |

```python
# Example: Talk to the model and it talks back
response_audio = model.listen_and_respond(your_audio)  # Returns playable waveform

# Example: Make the model say something
audio = model.speak(tokenizer.encode("Hello, how can I help you?"))

# Save as WAV file
import soundfile as sf
sf.write("response.wav", audio.cpu().numpy(), 16000)
```

---

## 📚 Training Data

Xoron-Dev is trained on a massive, curated mix of open-source Hugging Face datasets and specialized synthetic data generated to enhance agentic capabilities and reduce hallucinations.

### 🌐 Open Source Datasets
We utilize over 50 high-quality datasets from Hugging Face, categorized by modality:

* **Text & Code:** Includes `Code-Feedback`, `HumanEvalPack`, `OpenOrca`, and `AgentInstruct` for robust coding and reasoning capabilities.
* **Tool Use:** Datasets like `Function-Calling-ChatML`, `Synth-APIGen`, and `Tool-Calls-MultiTurn` enable precise tool invocation across single and multi-turn interactions.
* **Vision (Image/Video):** Visual understanding is grounded in `ScienceQA`, `Video-MME`, and `VideoInstruct-100K`.
* **Generation:** Text-to-Image/Video capabilities are fine-tuned on `Stable-Diffusion-Prompts`, `Sora-Likert-Scoring` datasets by Rapidata, and `WebVid-10M`.
* **Audio:** Speech tasks are powered by `LibriSpeech`, `LibriTTS-R`, and `HiFi-TTS`.

### 🧪 Synthetic Data Pipeline
To bridge the gap between general knowledge and actionable agentic behavior, we generate extensive synthetic datasets locally using our custom `synth` engine. These datasets focus on complex behaviors often missing from public corpuses:

| Category | Description |
|----------|-------------|
| **Anti-Hallucination** | Training the model to say "I don't know" (`Synth-IDK`), verify facts (`Synth-FactCheck`), provide citations (`Synth-Citation`), express uncertainty (`Synth-Uncertainty`), and ground responses (`Synth-GroundedResponse`). |
| **System Administration** | Simulated environments for `Docker` setup, `SSH` configuration, database management, and package installation (`Synth-AptInstall`). |
| **Code Execution** | Traces of code execution including `Shell` errors, timeouts, and multi-step debugging workflows to teach the model how to recover from errors. |
| **Git Operations** | Simulated version control tasks including committing, handling diffs, resolving merge conflicts, and repository context understanding. |
| **Chain-of-Thought** | Explicit `Synth-CoT` data to encourage internal reasoning before generating final answers. |
| **File Operations** | Document handling, FIM (Fill-in-Middle), and edit operations for precise file manipulation. |

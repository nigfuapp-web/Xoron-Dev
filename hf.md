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
- abhi227070/converstion-to-summarization-dataset
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

**Xoron-Dev** is a unified, multimodal AI model designed to understand and generate text, images, video, and audio within a single architecture. It leverages a **Mixture of Experts (MoE)** backbone with DeepSeek-style shared experts and integrates SOTA encoders (SigLIP-2) and diffusers (MobileDiffusion) for comprehensive any-to-any capabilities.

## 🌟 Model Highlights

* **Architecture:** Mixture of Experts (8 Experts + 1 Shared) with Ring Attention.
* **Vision:** Native understanding of images (384px) and video (up to 32 frames) via SigLIP-2.
* **Generation:** Integrated MobileDiffusion for fast on-device Image & Video generation.
* **Audio:** Full duplex capabilities with Conformer-based ASR (Speech-to-Text) and Neural TTS.
* **Agentic:** Trained for tool calling, file operations, and code execution with uncertainty estimation.
* **Context:** Efficient 128K context using Ring Attention (4096 chunk size).

---

## 📚 Training Data

Xoron-Dev is trained on a massive, curated mix of open-source Hugging Face datasets and specialized synthetic data generated to enhance agentic capabilities and reduce hallucinations.

### 🌐 Open Source Datasets
We utilize over 50 high-quality datasets from Hugging Face, categorized by modality:

* **Text & Code:** Includes `Code-Feedback`, `HumanEvalPack`, `OpenOrca`, and `AgentInstruct` for robust coding and reasoning capabilities.
* **Tool Use:** Datasets like `Function-Calling-ChatML` and `Synth-APIGen` enable precise tool invocation.
* **Vision (Image/Video):** Visual understanding is grounded in `ScienceQA`, `Video-MME`, and `VideoInstruct-100K`.
* **Generation:** Text-to-Image/Video capabilities are fine-tuned on `Stable-Diffusion-Prompts`, `Sora-Likert-Scoring` datasets by Rapidata, and `WebVid-10M`.
* **Audio:** Speech tasks are powered by `LibriSpeech`, `LibriTTS-R`, and `HiFi-TTS`.

### 🧪 Synthetic Data Pipeline
To bridge the gap between general knowledge and actionable agentic behavior, we generate extensive synthetic datasets locally using our custom `synth` engine. These datasets focus on complex behaviors often missing from public corpuses:

| Category | Description |
|----------|-------------|
| **Anti-Hallucination** | Training the model to say "I don't know" (`Synth-IDK`), verify facts (`Synth-FactCheck`), and provide citations (`Synth-Citation`) rather than fabricating information. |
| **System Administration** | Simulated environments for `Docker` setup, `SSH` configuration, database management, and package installation (`Synth-AptInstall`). |
| **Code Execution** | Traces of code execution including `Shell` errors, timeouts, and multi-step debugging workflows to teach the model how to recover from errors. |
| **Git Operations** | Simulated version control tasks including committing, handling diffs, and resolving merge conflicts. |
| **Chain-of-Thought** | explicit `Synth-CoT` data to encourage internal reasoning before generating final answers. |

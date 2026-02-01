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
- lmms-lab/Video-MME
- derek-thomas/ScienceQA
- Rapidata/sora-video-generation-physics-likert-scoring
- google/siglip-so400m-patch14-384
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

* **Architecture:** Mixture of Experts (8 Experts + 1 Shared) with Sliding Window Attention.
* **Vision:** Native understanding of images (384px) and video (up to 32 frames) via SigLIP-2.
* **Generation:** Integrated MobileDiffusion for fast on-device Image & Video generation.
* **Audio:** Full duplex capabilities with Conformer-based ASR (Speech-to-Text) and Neural TTS.
* **Agentic:** Trained for tool calling, file operations, and code execution with uncertainty estimation.
* **Context:** Efficient 128K context window using sliding window attention (4096 local window).

---

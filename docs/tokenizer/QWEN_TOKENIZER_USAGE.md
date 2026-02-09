# How Xoron-Dev Uses the Qwen Tokenizer

## Overview

Xoron-Dev is a **ground-up multimodal AI model** where nearly everything is built from scratch. The only external component we use as-is is **SigLIP2** for the vision encoder backbone. Even then, we wrap SigLIP2 with our own custom layers (TiTok, Dual-Stream Attention, 2D-RoPE) to better integrate it with our architecture.

The **Qwen2.5 tokenizer** is used purely as a **vocabulary foundation** - we don't use Qwen's model weights, architecture, or pre-trained capabilities. We leverage only the tokenizer's vocabulary and then heavily customize it for our multimodal needs.

---

## Why We Use Qwen2.5 Tokenizer as a Base

### 1. **Starting Point - Not Dependency**

Think of the Qwen tokenizer like buying a dictionary when writing your own novel. You need words to work with, but the story is entirely yours.

```python
# In config/model_config.py
tokenizer_name: str = "Qwen/Qwen2.5-1.5B"  # Source of base vocabulary
vocab_size: int = 151643                     # Qwen's vocab size (we expand this)
```

### 2. **What We Take from Qwen**

- **Base vocabulary**: ~151,643 tokens covering multilingual text
- **Tokenization algorithm**: BPE (Byte-Pair Encoding) for efficient text encoding
- **Nothing else**: No model weights, no architecture, no pre-trained knowledge

### 3. **What We Build Ourselves**

Literally everything else:
- LLM backbone (MoE with Ring Attention)
- Vision encoder (with custom TiTok, Dual-Stream, 2D-RoPE layers around SigLIP2)
- Video encoder (3D-RoPE, Temporal MoE, Causal 3D Transformers)
- Audio system (Raw Waveform Tokenizer, Conformer, RMLA, BigVGAN decoder)
- Image generator (MoE-DiT with Flow Matching)
- Video generator (3D VAE with Temporal MoE)
- 250+ custom special tokens
- Custom chat templates

---

## The Tokenizer Setup Process

Here's exactly how we use and customize the Qwen tokenizer in `build.py`:

### Step 1: Load the Base Tokenizer

```python
from transformers import AutoTokenizer

def setup_tokenizer(model, xoron_config):
    # Load Qwen tokenizer - just getting the vocabulary
    tokenizer = AutoTokenizer.from_pretrained(xoron_config.tokenizer_name)
    # At this point: vocab_size = 151,643 (Qwen's base)
```

### Step 2: Add Our 250+ Custom Special Tokens

```python
from config.special_tokens import SPECIAL_TOKENS

# Add ALL our custom tokens
special_tokens_list = list(SPECIAL_TOKENS.values())
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_list})
# Now vocab_size = 151,643 + ~250 = ~151,893
```

Our special tokens include:
- **Conversation**: `<|system|>`, `<|user|>`, `<|assistant|>`
- **Multimodal**: `<|image|>`, `<|video|>`, `<|audio|>`
- **Reasoning**: `<|think|>`, `<|plan|>`, `<|critique|>`, `<|reflection|>`
- **Tool Calling**: `<|tool_call|>`, `<|function_name|>`, `<|function_args|>`
- **Code Execution**: `<|exec|>`, `<|jupyter_code|>`, `<|jupyter_output|>`
- **Generation**: `<|gen_image|>`, `<|gen_video|>`, `<|speak|>`
- **Anti-Hallucination**: `<|uncertain|>`, `<|cite|>`, `<|confidence_high|>`
- **File/Git Operations**: `<|add_file|>`, `<|diff|>`, `<|commit_msg|>`
- **FIM (Fill-in-Middle)**: `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`
- And 200+ more...

### Step 3: Replace the Chat Template

```python
from config.chat_template import apply_chat_template_to_tokenizer

# Replace Qwen's chat template with our custom Jinja2 template
tokenizer = apply_chat_template_to_tokenizer(tokenizer, multimodal=True)
```

This completely replaces any Qwen formatting with our own:
- Supports multimodal inputs (images, videos, audio in messages)
- Supports reasoning blocks (think, plan, critique)
- Supports tool calling with structured arguments
- Supports code execution with results
- Supports generation triggers

### Step 4: Set Our Control Tokens

```python
# Override BOS/EOS/PAD with OUR tokens (not Qwen's)
tokenizer.bos_token = SPECIAL_TOKENS['bos']  # <|bos|>
tokenizer.eos_token = SPECIAL_TOKENS['eos']  # <|eos|>
tokenizer.pad_token = SPECIAL_TOKENS['pad']  # <|pad|>
```

### Step 5: Resize Model Embeddings

Since we added tokens, we need to expand the embedding layer:

```python
# Resize embed_tokens to fit new vocabulary
new_vocab_size = len(tokenizer)  # ~151,893
old_embed = model.llm.model.embed_tokens

new_embed = nn.Embedding(new_vocab_size, xoron_config.hidden_size)
with torch.no_grad():
    nn.init.normal_(new_embed.weight, mean=0.0, std=0.02)  # Random init for new tokens
    min_vocab = min(old_embed.weight.shape[0], new_vocab_size)
    new_embed.weight[:min_vocab] = old_embed.weight[:min_vocab]  # Keep base vocab embeddings

model.llm.model.embed_tokens = new_embed

# Same for output lm_head
new_lm_head = nn.Linear(xoron_config.hidden_size, new_vocab_size, bias=False)
# ... similar initialization
model.llm.lm_head = new_lm_head
```

---

## Architecture Diagram: Tokenizer's Role

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT                                     │
│  "What's in this image? <|image|>..." + pixel_values            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  QWEN TOKENIZER (Customized)                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Base Vocab: 151,643 tokens (from Qwen2.5)               │   │
│  │ + 250 Custom Tokens (Xoron special tokens)              │   │
│  │ = ~151,893 total vocabulary                             │   │
│  │                                                          │   │
│  │ Custom Chat Template → Proper formatting                 │   │
│  │ Custom BOS/EOS/PAD → Our control tokens                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                       │
│                          ▼                                       │
│               token_ids: [234, 567, <|image|>, 891, ...]        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│               XORON LLM (100% Custom Built)                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ embed_tokens: Embedding(151,893, 1024)                   │   │
│  │ - First 151,643 dims: initialized from scratch           │   │
│  │ - Last ~250 dims: our special tokens (random init)       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 12 Transformer Layers (Custom MoE + Ring Attention)      │   │
│  │ - MoE: 8 experts, top-2 routing, aux-lossless            │   │
│  │ - Ring Attention: 128K context, 4096 chunk size          │   │
│  │ - Cross-Attention: 4 layers for multimodal fusion        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ lm_head: Linear(1024, 151,893)                           │   │
│  │ → Output logits over our expanded vocabulary             │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why Not Build Our Own Tokenizer from Scratch?

Training a tokenizer from scratch requires:
1. **Massive corpus**: Terabytes of diverse multilingual text
2. **Compute time**: Days to train BPE on such data
3. **Quality validation**: Ensuring good coverage across domains

Using Qwen's tokenizer as a base gives us:
- **Proven multilingual coverage** (Chinese, English, code, etc.)
- **Efficient BPE encoding** already tuned for LLM use
- **Time savings** to focus on our novel architecture

But we still **make it our own** by:
- Adding 250+ domain-specific special tokens
- Replacing the chat template entirely
- Training all embeddings from scratch (not using Qwen's weights)
- Building our own model architecture around it

---

## SigLIP2: The Only External Model Component

While we use Qwen tokenizer for vocabulary, the only actual **pre-trained model** we use externally is **SigLIP2** for the vision encoder backbone.

```python
# In models/encoders/vision.py
class VisionEncoder(nn.Module):
    def _init_siglip(self, model_name: str, freeze: bool):
        from transformers import SiglipVisionModel, SiglipImageProcessor
        
        # Load SigLIP2 - this is the ONLY external model we use
        self.vision_model = SiglipVisionModel.from_pretrained(model_name)
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name)
```

But even SigLIP2 is heavily customized:
- **TiTok Tokenizer**: Compresses 576 patches → 256 tokens
- **Dual-Stream Attention**: SD3/Flux-style symmetric processing
- **2D-RoPE**: Flexible aspect ratio support
- **Perceiver Resampler**: Projects to 64 LLM tokens

So SigLIP2 is more like a "feature extraction engine" that we wrap with our own architecture.

---

## Summary: What's Ours vs. External

| Component | Source | What We Use |
|-----------|--------|-------------|
| **Tokenizer Vocabulary** | Qwen2.5 | Base ~151K tokens |
| **Tokenizer Algorithm** | Qwen2.5 | BPE encoding |
| **Special Tokens** | **Xoron (Custom)** | 250+ tokens |
| **Chat Template** | **Xoron (Custom)** | Full Jinja2 template |
| **LLM Architecture** | **Xoron (Custom)** | MoE + Ring Attention |
| **LLM Weights** | **Xoron (Custom)** | Trained from scratch |
| **Vision Backbone** | SigLIP2 | Feature extraction |
| **Vision Extensions** | **Xoron (Custom)** | TiTok, Dual-Stream, 2D-RoPE |
| **Video Encoder** | **Xoron (Custom)** | 3D-RoPE, Temporal MoE |
| **Audio System** | **Xoron (Custom)** | Raw Waveform, Conformer, BigVGAN |
| **Image Generator** | **Xoron (Custom)** | MoE-DiT + Flow Matching |
| **Video Generator** | **Xoron (Custom)** | 3D VAE + Temporal MoE |

**Bottom line**: The Qwen tokenizer gives us a solid vocabulary foundation. Everything else - the architecture, the training, the special capabilities - is built from the ground up by Xoron-Dev.

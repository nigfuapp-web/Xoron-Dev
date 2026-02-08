# 👁️ Encoders Module Documentation

The Encoders module contains specialized neural networks for processing different input modalities: vision (images), video, and audio. Each encoder transforms raw input into a unified representation that can be processed by the LLM backbone.

## 📁 File Structure

```
models/encoders/
├── __init__.py
├── vision.py    # Image encoder (SigLIP-2 + TiTok + 2D-RoPE)
├── video.py     # Video encoder (3D-RoPE + Temporal MoE)
└── audio.py     # Audio encoder/decoder (Conformer + RMLA + MAS)
```

---

## 👁️ Vision Encoder

### Overview

The Vision Encoder processes static images using a SigLIP-2 backbone enhanced with state-of-the-art features for efficient visual representation.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Vision Encoder                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Image (384×384)                                         │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  SigLIP-2 Backbone (google/siglip-so400m-patch14-384)   │   │
│  │  - Patch size: 14×14                                     │   │
│  │  - Output: 576 patches (24×24 grid)                      │   │
│  │  - Hidden size: 1152                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  2D-RoPE Position Encoding                               │   │
│  │  - Encodes (x, y) spatial positions                      │   │
│  │  - Flexible aspect ratio support                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  TiTok Tokenizer                                         │   │
│  │  - Compresses 576 patches → 256 tokens                   │   │
│  │  - Cross-attention based compression                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Dual-Stream Attention (2 layers)                        │   │
│  │  - Symmetric processing                                  │   │
│  │  - SD3/Flux-style architecture                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  Output: [B, 64, hidden_size] visual tokens                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. 2D Rotary Position Embedding (2D-RoPE)

**Purpose**: Encode spatial positions for image patches, enabling flexible aspect ratio handling.

```python
class RoPE2DEncoder(nn.Module):
    """
    2D Rotary Position Embedding for vision encoder patches.
    Matches the 2D-RoPE in image generator for seamless integration.
    """
    def __init__(self, dim: int, max_height: int = 128, max_width: int = 128, base: float = 10000.0):
        # Split dimension between x and y
        self.dim_x = dim // 2
        self.dim_y = dim - self.dim_x
        
        # Compute inverse frequencies for each axis
        inv_freq_x = 1.0 / (base ** (torch.arange(0, self.dim_x, 2) / self.dim_x))
        inv_freq_y = 1.0 / (base ** (torch.arange(0, self.dim_y, 2) / self.dim_y))
    
    def forward(self, x, height, width):
        # Create 2D position grid
        # cos_2d[y, w] encodes position (y, w) in the image
        for y in range(height):
            for w in range(width):
                cos_2d[y, w, :self.dim_x] = freqs_x[w].cos()
                cos_2d[y, w, self.dim_x:] = freqs_y[y].cos()
        return cos_2d, sin_2d
```

**Why 2D-RoPE?**
- Preserves spatial relationships between patches
- Enables variable resolution processing
- Matches the generator's position encoding for consistency

#### 2. TiTok Tokenizer

**Purpose**: Compress patch features into a smaller set of tokens for efficient processing.

```python
class TiTokTokenizer(nn.Module):
    """
    TiTok-style 1D Tokenizer for efficient visual representation.
    Converts 2D patch grid to 1D token sequence with learnable compression.
    """
    def __init__(self, hidden_size: int, num_tokens: int = 256, num_patches: int = 576):
        # Learnable token queries for compression
        self.token_queries = nn.Parameter(torch.randn(1, num_tokens, hidden_size) * 0.02)
        
        # Cross-attention for compression
        self.compress_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True,
        )
    
    def forward(self, x):
        # x: [B, num_patches, hidden_size] (576 patches)
        # Expand queries for batch
        queries = self.token_queries.expand(batch_size, -1, -1)
        
        # Cross-attention: queries attend to patches
        tokens, _ = self.compress_attn(queries, x, x)
        # tokens: [B, num_tokens, hidden_size] (256 tokens)
        return tokens
```

**Compression Ratio**: 576 patches → 256 tokens (2.25x compression)

**Why TiTok?**
- Reduces sequence length for faster LLM processing
- Learnable compression preserves important information
- Matches state-of-the-art image tokenization approaches

#### 3. Dual-Stream Attention

**Purpose**: Symmetric processing of visual features for better representation.

```python
class DualStreamEncoderAttention(nn.Module):
    """
    Symmetric Dual-Stream Self-Attention for vision encoding.
    Matches the dual-stream architecture in image generator.
    """
    def __init__(self, hidden_size: int, num_heads: int = 8):
        # Two parallel streams
        self.to_qkv_a = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.to_qkv_b = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        
        self.to_out_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_out_b = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.rope_2d = RoPE2D(head_dim, max_height, max_width)
    
    def forward(self, x_a, x_b, height, width):
        # Stream A attends to Stream B and vice versa
        # This creates symmetric information flow
        ...
```

**Why Dual-Stream?**
- Inspired by SD3 and Flux architectures
- Better feature mixing than single-stream attention
- Symmetric processing improves representation quality

---

## 🎬 Video Encoder

### Overview

The Video Encoder extends the vision encoder to handle temporal sequences, using 3D position encoding and temporal-aware expert routing.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Video Encoder                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Video: [B, T, C, H, W] (T frames)                       │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Per-Frame Vision Encoding (SigLIP-2)                    │   │
│  │  - Each frame processed independently                    │   │
│  │  - Output: [B, T, num_patches, hidden_size]              │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  3D-RoPE Position Encoding                               │   │
│  │  - Encodes (x, y, t) positions                           │   │
│  │  - Temporal dimension for frame ordering                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  3D Causal Transformer Layers (4 layers)                 │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │  3D Causal Self-Attention                        │    │   │
│  │  │  - Spatial attention within frames               │    │   │
│  │  │  - Temporal attention across frames              │    │   │
│  │  │  - Causal masking for temporal order             │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │  Temporal MoE Layer                              │    │   │
│  │  │  - 4 experts with temporal-aware routing         │    │   │
│  │  │  - Motion pattern specialization                 │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  Output: [B, T×num_tokens, hidden_size] video tokens           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. 3D Rotary Position Embedding (3D-RoPE)

**Purpose**: Encode spatial and temporal positions for video patches.

```python
class RoPE3DEncoder(nn.Module):
    """
    3D Rotary Position Embedding for (x, y, t) dimensions.
    Matches the 3D-RoPE in video generator.
    """
    def __init__(self, dim: int, max_height: int = 64, max_width: int = 64, max_frames: int = 32):
        # Split dimension among three axes
        dim_per_axis = dim // 3
        self.dim_x = dim_per_axis
        self.dim_y = dim_per_axis
        self.dim_t = dim - 2 * dim_per_axis
        
        # Inverse frequencies for each axis
        inv_freq_x = 1.0 / (base ** (torch.arange(0, self.dim_x, 2) / self.dim_x))
        inv_freq_y = 1.0 / (base ** (torch.arange(0, self.dim_y, 2) / self.dim_y))
        inv_freq_t = 1.0 / (base ** (torch.arange(0, self.dim_t, 2) / self.dim_t))
    
    def forward(self, x, height, width, frames):
        # Create 3D position encoding
        for t in range(frames):
            for y in range(height):
                for w in range(width):
                    cos_3d[t, y, w, :self.dim_x] = cos_x[w]
                    cos_3d[t, y, w, self.dim_x:self.dim_x+self.dim_y] = cos_y[y]
                    cos_3d[t, y, w, self.dim_x+self.dim_y:] = cos_t[t]
        return cos_3d, sin_3d
```

**Why 3D-RoPE?**
- Captures spatial relationships within frames
- Encodes temporal ordering across frames
- Enables variable frame count processing

#### 2. Temporal Expert Router

**Purpose**: Route tokens to experts based on temporal context and motion patterns.

```python
class TemporalExpertRouterEncoder(nn.Module):
    """
    Temporal-Aware Expert Router for video encoding.
    Routes tokens based on temporal context and motion patterns.
    """
    def __init__(self, hidden_size: int, num_experts: int = 4, top_k: int = 2):
        self.temporal_proj = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
    
    def forward(self, x, temporal_context=None):
        # Incorporate temporal context into routing decision
        if temporal_context is not None:
            x = x + self.temporal_proj(temporal_context)
        
        router_logits = self.gate(x)
        router_probs = F.softmax(router_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        return top_k_probs, top_k_indices
```

**Why Temporal MoE?**
- Different experts specialize in different motion patterns
- Static scenes vs. fast motion vs. camera movement
- Better video understanding through specialization

#### 3. 3D Causal Attention

**Purpose**: Process video with proper temporal causality.

```python
class Causal3DAttention(nn.Module):
    """
    3D Causal Self-Attention for video encoding.
    Each position can only attend to previous frames and current frame positions.
    """
    def forward(self, x, height, width, frames):
        # Factorized attention for efficiency:
        # 1. Spatial attention within each frame
        # 2. Temporal attention across frames (causal)
        
        # Spatial attention (no causal mask)
        for t in range(frames):
            frame_tokens = x[:, t*hw:(t+1)*hw, :]
            spatial_out = self.spatial_attn(frame_tokens)
        
        # Temporal attention (causal mask)
        for pos in range(hw):
            temporal_tokens = x[:, pos::hw, :]  # Same position across frames
            temporal_out = self.temporal_attn(temporal_tokens, causal=True)
```

**Why Factorized Attention?**
- Full 3D attention is O((T×H×W)²) - too expensive
- Factorized is O(T×(H×W)² + H×W×T²) - much more efficient
- Maintains quality while reducing computation

---

## 🎤 Audio Encoder

### Overview

The Audio Encoder processes speech and audio using a Conformer architecture with state-of-the-art features for both ASR (speech-to-text) and speaker understanding.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Audio Encoder                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: Raw Waveform [B, T] at 16kHz                           │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Raw Waveform Tokenizer                                  │   │
│  │  - Multi-scale 1D convolutions                           │   │
│  │  - 64x downsampling                                      │   │
│  │  - Optional RVQ (Residual Vector Quantization)           │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Conformer Blocks (6 layers)                             │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │  Feed-Forward Module (1/2)                       │    │   │
│  │  │  RMLA (Rotary Multi-Head Latent Attention)       │    │   │
│  │  │  Convolution Module                              │    │   │
│  │  │  Feed-Forward Module (1/2)                       │    │   │
│  │  │  LayerNorm                                       │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ├──────────────────────────────────────────────────┐   │
│         │                                                  │   │
│         ▼                                                  ▼   │
│  Audio Features                              Speaker Encoder   │
│  [B, T', hidden_size]                        [B, speaker_dim]  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Raw Waveform Tokenizer

**Purpose**: Convert raw audio waveform directly to features without mel spectrograms.

```python
class RawWaveformTokenizer(nn.Module):
    """
    Raw Waveform Tokenizer - directly tokenizes audio waveforms.
    Uses multi-scale 1D convolutions for feature extraction.
    """
    def __init__(
        self,
        hidden_size: int = 1024,
        num_codebooks: int = 8,
        codebook_size: int = 1024,
        sample_rate: int = 16000,
    ):
        # Multi-scale convolutional encoder
        # Progressively increases channels while downsampling
        channels = [32, 64, 128, 256, 512, hidden_size]
        strides = [2, 2, 2, 2, 2, 2]  # Total: 64x downsampling
        
        for i in range(num_conv_layers):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
            ))
        
        # Optional: Residual Vector Quantization codebooks
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, hidden_size)
            for _ in range(num_codebooks)
        ])
    
    def encode(self, waveform):
        # waveform: [B, T] or [B, 1, T]
        x = waveform.unsqueeze(1) if waveform.dim() == 2 else waveform
        
        for conv in self.conv_layers:
            x = conv(x)
        
        # [B, C, T'] -> [B, T', C]
        return x.transpose(1, 2)
```

**Why Raw Waveform?**
- No information loss from mel spectrogram conversion
- End-to-end learnable feature extraction
- Better for zero-shot voice cloning

#### 2. Rotary Multi-Head Latent Attention (RMLA)

**Purpose**: Efficient attention for audio sequences with compressed KV cache.

```python
class RotaryMultiHeadLatentAttention(nn.Module):
    """
    RMLA: Combines Rotary Position Embedding with Multi-Head Latent Attention.
    Efficient for long audio sequences.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        kv_lora_rank: int = 256,  # KV compression rank
    ):
        # KV compression (like MLA in LLM)
        self.kv_a_proj = nn.Linear(hidden_size, kv_lora_rank, bias=False)
        self.kv_b_proj = nn.Linear(kv_lora_rank, num_kv_heads * head_dim * 2, bias=False)
        
        # Rotary embeddings for position encoding
        self.rotary_emb = RotaryEmbedding(head_dim)
    
    def forward(self, x, mask=None):
        # Compress KV
        kv_compressed = self.kv_a_proj(x)
        kv_compressed = self.kv_a_layernorm(kv_compressed)
        kv = self.kv_b_proj(kv_compressed)
        
        # Apply rotary embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Standard attention
        attn_output = scaled_dot_product_attention(q, k, v, mask)
        return attn_output
```

**Why RMLA?**
- Combines benefits of RoPE (position encoding) and MLA (memory efficiency)
- Essential for processing long audio sequences
- Matches the LLM's attention mechanism

#### 3. Conformer Block

**Purpose**: Hybrid convolution-attention architecture optimized for audio.

```python
class ConformerBlock(nn.Module):
    """
    Conformer block: FFN → Attention → Conv → FFN
    Optimized for speech/audio processing.
    """
    def __init__(self, hidden_size: int, num_heads: int, conv_kernel_size: int = 31):
        # Half feed-forward at start
        self.ff1 = FeedForward(hidden_size, expansion=4)
        
        # RMLA attention
        self.attn = RotaryMultiHeadLatentAttention(hidden_size, num_heads, ...)
        
        # Convolution module
        self.conv = ConvolutionModule(hidden_size, conv_kernel_size)
        
        # Half feed-forward at end
        self.ff2 = FeedForward(hidden_size, expansion=4)
    
    def forward(self, x, mask=None):
        # Macaron-style: FFN sandwich
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x, mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)
```

**Why Conformer?**
- Convolutions capture local patterns (phonemes, syllables)
- Attention captures global context (sentence structure)
- State-of-the-art for ASR tasks

#### 4. Speaker Encoder (Zero-Shot Cloning)

**Purpose**: Extract speaker embeddings for voice cloning.

```python
class SpeakerEncoder(nn.Module):
    """
    Zero-Shot Speaker Encoder for speaker cloning.
    Extracts speaker embeddings from reference audio.
    """
    def __init__(self, hidden_size: int = 256, output_size: int = 256):
        # Frame-level encoder
        self.frame_encoder = nn.Sequential(
            nn.Conv1d(80, hidden_size, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            ...
        )
        
        # Utterance-level aggregation (GRU)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        
        # Final projection
        self.output_proj = nn.Linear(hidden_size, output_size)
    
    def forward(self, mel_spectrogram):
        # Process frames
        frame_features = self.frame_encoder(mel_spectrogram)
        
        # Aggregate across time
        _, hidden = self.gru(frame_features.transpose(1, 2))
        
        # Project to speaker embedding
        speaker_embedding = self.output_proj(hidden.squeeze(0))
        return F.normalize(speaker_embedding, dim=-1)
```

**Why Speaker Encoder?**
- Enables zero-shot voice cloning from short reference
- Speaker embedding conditions the decoder
- Preserves voice characteristics in generated speech

---

## 🔊 Audio Decoder

### Overview

The Audio Decoder generates speech from text embeddings, with support for zero-shot voice cloning and direct waveform output.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Audio Decoder                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Text Embeddings [B, T, hidden_size]                           │
│  + Speaker Embedding [B, speaker_dim]                          │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Encoder FFT Blocks (4 layers)                           │   │
│  │  - Self-attention with RMLA                              │   │
│  │  - Conv feed-forward                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Variance Adaptor                                        │   │
│  │  - Duration predictor                                    │   │
│  │  - Pitch predictor                                       │   │
│  │  - Energy predictor                                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  MAS (Monotonic Alignment Search)                        │   │
│  │  - Aligns text to audio frames                           │   │
│  │  - Learns duration without external aligner              │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Decoder FFT Blocks (4 layers)                           │   │
│  │  - Upsampled to mel length                               │   │
│  │  - Pitch/energy embeddings added                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  Mel Spectrogram [B, n_mels, T']                               │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Raw Waveform Decoder (BigVGAN-style)                    │   │
│  │  - Snake activation                                      │   │
│  │  - Multi-Receptive Field Fusion                          │   │
│  │  - 256x upsampling                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  Raw Waveform [B, T_audio] at 16kHz                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Monotonic Alignment Search (MAS)

**Purpose**: Learn text-to-audio alignment without external forced alignment.

```python
class MonotonicAlignmentSearch(nn.Module):
    """
    MAS for learning text-to-audio alignment.
    Finds the most probable monotonic alignment path.
    """
    def __init__(self, hidden_size: int):
        self.text_proj = nn.Linear(hidden_size, hidden_size)
        self.audio_proj = nn.Linear(hidden_size, hidden_size)
        self.duration_proj = nn.Linear(hidden_size, 1)
    
    def forward(self, text_features, audio_features=None, use_hard=False):
        # Compute alignment scores
        text_proj = self.text_proj(text_features)
        
        if audio_features is not None:
            audio_proj = self.audio_proj(audio_features)
            # Compute soft alignment matrix
            alignment = torch.bmm(text_proj, audio_proj.transpose(1, 2))
            alignment = F.softmax(alignment, dim=-1)
            
            if use_hard:
                # Convert to hard alignment using dynamic programming
                alignment = self._mas_dp(alignment)
            
            # Derive durations from alignment
            durations = alignment.sum(dim=-1)
        else:
            # Predict durations directly
            durations = F.softplus(self.duration_proj(text_features).squeeze(-1))
        
        return alignment, durations
```

**Why MAS?**
- No need for external forced alignment tools
- End-to-end trainable
- Better generalization to unseen speakers

#### 2. Raw Waveform Decoder (BigVGAN-style)

**Purpose**: Convert mel spectrograms directly to audio waveform without external vocoder.

```python
class RawWaveformDecoder(nn.Module):
    """
    SOTA Raw Waveform Decoder - BigVGAN/HiFi-GAN style.
    Direct audio output without external vocoder.
    """
    def __init__(
        self,
        hidden_size: int = 1024,
        upsample_rates: List[int] = [8, 8, 2, 2],  # 256x total
        resblock_kernel_sizes: List[int] = [3, 7, 11],
    ):
        # Input projection
        self.input_proj = nn.Conv1d(hidden_size, initial_channels, 7, padding=3)
        
        # Upsampling layers with MRF blocks
        for rate, kernel in zip(upsample_rates, upsample_kernel_sizes):
            self.upsamplers.append(
                nn.ConvTranspose1d(channels, channels // 2, kernel, stride=rate, ...)
            )
            self.mrf_blocks.append(
                MultiReceptiveFieldFusion(channels // 2, resblock_kernel_sizes)
            )
        
        # Final output
        self.final_activation = SnakeActivation(channels)
        self.output_conv = nn.Conv1d(channels, 1, 7, padding=3)
    
    def forward(self, features, target_length=None):
        x = self.input_proj(features.transpose(1, 2))
        
        for upsample, mrf in zip(self.upsamplers, self.mrf_blocks):
            x = upsample(x)
            x = mrf(x)
        
        x = self.final_activation(x)
        waveform = torch.tanh(self.output_conv(x))
        return waveform.squeeze(1)
```

#### 3. Snake Activation

**Purpose**: Preserve audio periodicity better than ReLU/SiLU.

```python
class SnakeActivation(nn.Module):
    """
    Snake activation from BigVGAN.
    x + (1/a) * sin²(a * x)
    Better than ReLU/SiLU for audio - preserves periodicity.
    """
    def __init__(self, channels: int, alpha: float = 1.0):
        self.alpha = nn.Parameter(torch.ones(1, channels, 1) * alpha)
    
    def forward(self, x):
        return x + (1.0 / (self.alpha + 1e-6)) * torch.sin(self.alpha * x) ** 2
```

**Why Snake?**
- Audio is inherently periodic (waveforms)
- Snake preserves periodic structure
- Learnable frequency parameter adapts to content

#### 4. Multi-Receptive Field Fusion (MRF)

**Purpose**: Capture patterns at multiple temporal scales.

```python
class MultiReceptiveFieldFusion(nn.Module):
    """
    MRF from HiFi-GAN.
    Parallel residual stacks with different kernel sizes.
    """
    def __init__(self, channels: int, kernel_sizes: List[int] = [3, 7, 11]):
        self.resblocks = nn.ModuleList()
        for k in kernel_sizes:
            blocks = nn.ModuleList([
                ResidualBlock1D(channels, k, dilation=1),
                ResidualBlock1D(channels, k, dilation=3),
                ResidualBlock1D(channels, k, dilation=5),
            ])
            self.resblocks.append(blocks)
    
    def forward(self, x):
        out = None
        for blocks in self.resblocks:
            h = x
            for block in blocks:
                h = block(h)
            out = h if out is None else out + h
        return out / len(self.resblocks)
```

**Why MRF?**
- Small kernels capture fine details (consonants)
- Large kernels capture broader patterns (vowels, prosody)
- Fusion combines all scales

---

## 📊 Encoder Specifications

| Encoder | Input | Output | Key Features |
|---------|-------|--------|--------------|
| **Vision** | 384×384 image | 64 tokens | SigLIP-2, TiTok, 2D-RoPE |
| **Video** | T×H×W frames | T×64 tokens | 3D-RoPE, Temporal MoE |
| **Audio** | 16kHz waveform | T' tokens | Conformer, RMLA, MAS |

---

## 🔗 Related Documentation

- [LLM Documentation](llm.md) - How encoders connect to the LLM
- [Generators Documentation](generators.md) - Corresponding output generators
- [Components Documentation](components.md) - Shared components (attention, MoE)

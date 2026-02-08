# 🎨 Generators Module Documentation

The Generators module contains neural networks for generating images and videos from text or multimodal embeddings. Both generators use state-of-the-art diffusion techniques with Flow Matching.

## 📁 File Structure

```
models/generators/
├── __init__.py
├── image.py    # Image generator (MoE-DiT + Flow Matching)
└── video.py    # Video generator (3D Causal + Flow Matching)
```

---

## 🖼️ Image Generator (MoE-DiT)

### Overview

The Image Generator uses a Mixture of Experts Diffusion Transformer (MoE-DiT) with Flow Matching for high-quality image synthesis.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MoE-DiT Image Generator                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Text Embeddings [B, T, hidden_size]                           │
│  + Noise [B, C, H, W] (latent space)                           │
│  + Timestep t                                                   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Patch Embedding                                         │   │
│  │  - Conv2d: latent_channels → hidden_size                 │   │
│  │  - Patch size: 2×2                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Timestep Embedding                                      │   │
│  │  - Sinusoidal → MLP → hidden_size                        │   │
│  │  - AdaLN modulation parameters                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  DiT Blocks × 8                                          │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │  AdaLN → Dual-Stream Self-Attention             │    │   │
│  │  │  (with 2D-RoPE position encoding)               │    │   │
│  │  │           │                                      │    │   │
│  │  │           ▼                                      │    │   │
│  │  │  AdaLN → Cross-Attention (to text)              │    │   │
│  │  │           │                                      │    │   │
│  │  │           ▼                                      │    │   │
│  │  │  AdaLN → MoE FFN (4 experts)                    │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Final Layer                                             │   │
│  │  - AdaLN → Linear → Unpatchify                           │   │
│  │  - Output: velocity prediction v(x, t)                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  Predicted Velocity [B, C, H, W]                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Flow Matching

**Purpose**: Superior alternative to DDPM for diffusion-based generation.

**How it works**: Instead of predicting noise (DDPM), Flow Matching predicts the velocity field that transforms noise to data:

```python
class FlowMatchingScheduler:
    """
    Flow Matching scheduler for image generation.
    Predicts velocity v(x, t) instead of noise ε.
    """
    def __init__(self, num_inference_steps: int = 50, sigma_min: float = 0.001):
        self.num_inference_steps = num_inference_steps
        self.sigma_min = sigma_min
    
    def get_velocity_target(self, x_0, x_1, t):
        """
        Compute velocity target for training.
        v = x_1 - x_0 (direction from noise to data)
        """
        return x_1 - x_0
    
    def step(self, velocity, x_t, t, dt):
        """
        Euler step for sampling.
        x_{t+dt} = x_t + v(x_t, t) * dt
        """
        return x_t + velocity * dt
    
    def add_noise(self, x_0, noise, t):
        """
        Interpolate between data and noise.
        x_t = (1 - t) * x_0 + t * noise
        """
        t = t.view(-1, 1, 1, 1)
        return (1 - t) * x_0 + t * noise
```

**Training**:
```python
# Sample timestep
t = torch.rand(batch_size)

# Add noise to create x_t
x_t = scheduler.add_noise(x_0, noise, t)

# Predict velocity
v_pred = model(x_t, t, text_embeddings)

# Loss: MSE between predicted and target velocity
v_target = scheduler.get_velocity_target(x_0, noise, t)
loss = F.mse_loss(v_pred, v_target)
```

**Inference**:
```python
# Start from pure noise
x = torch.randn(batch_size, channels, height, width)

# Euler integration from t=1 to t=0
for t in reversed(timesteps):
    v = model(x, t, text_embeddings)
    x = x - v * dt  # Move towards data
```

**Why Flow Matching?**
- Simpler training objective (velocity vs. noise)
- Better sample quality with fewer steps
- More stable training dynamics

#### 2. 2D Rotary Position Embedding (2D-RoPE)

**Purpose**: Encode spatial positions for flexible aspect ratios.

```python
class RoPE2D(nn.Module):
    """
    2D Rotary Position Embedding for flexible aspect ratios.
    Encodes (x, y) spatial positions for patch-based DiT.
    """
    def __init__(self, dim: int, max_height: int = 128, max_width: int = 128):
        # Split dimension between x and y axes
        self.dim_x = dim // 2
        self.dim_y = dim - self.dim_x
        
        # Inverse frequencies for each axis
        inv_freq_x = 1.0 / (base ** (torch.arange(0, self.dim_x, 2) / self.dim_x))
        inv_freq_y = 1.0 / (base ** (torch.arange(0, self.dim_y, 2) / self.dim_y))
    
    def forward(self, x, height, width):
        # Create 2D position grid
        cos_2d = torch.zeros(height, width, self.dim)
        sin_2d = torch.zeros(height, width, self.dim)
        
        for y in range(height):
            for w in range(width):
                cos_2d[y, w, :self.dim_x] = freqs_x[w].cos()
                cos_2d[y, w, self.dim_x:] = freqs_y[y].cos()
                # Similar for sin_2d
        
        return cos_2d.view(height * width, self.dim), sin_2d.view(...)
```

**Why 2D-RoPE?**
- Handles variable resolutions (256-512px)
- Preserves spatial relationships
- Matches encoder's position encoding

#### 3. Dual-Stream Self-Attention

**Purpose**: Symmetric processing inspired by SD3 and Flux architectures.

```python
class DualStreamSelfAttention(nn.Module):
    """
    Symmetric Dual-Stream Self-Attention (SD3/Flux-style).
    Two parallel streams with cross-stream information exchange.
    """
    def __init__(self, hidden_size: int, num_heads: int = 8):
        # Stream A projections
        self.to_qkv_a = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.to_out_a = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Stream B projections
        self.to_qkv_b = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.to_out_b = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.rope_2d = RoPE2D(head_dim, max_height, max_width)
    
    def forward(self, x_a, x_b, height, width):
        # Get Q, K, V for both streams
        q_a, k_a, v_a = self.to_qkv_a(x_a).chunk(3, dim=-1)
        q_b, k_b, v_b = self.to_qkv_b(x_b).chunk(3, dim=-1)
        
        # Apply 2D-RoPE
        cos, sin = self.rope_2d(x_a, height, width)
        q_a, k_a = apply_rope_2d(q_a, k_a, cos, sin)
        q_b, k_b = apply_rope_2d(q_b, k_b, cos, sin)
        
        # Cross-stream attention
        # Stream A attends to Stream B's keys/values
        out_a = attention(q_a, k_b, v_b)
        # Stream B attends to Stream A's keys/values
        out_b = attention(q_b, k_a, v_a)
        
        return self.to_out_a(out_a), self.to_out_b(out_b)
```

**Why Dual-Stream?**
- Better feature mixing than single-stream
- Symmetric information flow
- State-of-the-art in image generation

#### 4. Image MoE Layer

**Purpose**: Specialized experts for different image regions/patterns.

```python
class ImageMoELayer(nn.Module):
    """MoE Layer for DiT with shared expert."""
    
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int = 4):
        self.router = ImageMoERouter(hidden_size, num_experts, top_k=2)
        self.experts = nn.ModuleList([
            ImageExpert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
        self.shared_expert = ImageExpert(hidden_size, intermediate_size)
    
    def forward(self, x):
        # Route tokens to experts
        top_k_probs, top_k_indices = self.router(x)
        
        # Compute weighted expert outputs
        output = torch.zeros_like(x)
        for expert_idx in range(self.num_experts):
            for k in range(self.top_k):
                mask = (top_k_indices[:, k] == expert_idx)
                if mask.any():
                    expert_output = self.experts[expert_idx](x[mask])
                    output[mask] += top_k_probs[mask, k:k+1] * expert_output
        
        # Always add shared expert
        output = output + self.shared_expert(x)
        return output
```

**Why MoE in DiT?**
- Different experts specialize in different visual patterns
- Edges, textures, colors, objects
- Better capacity without proportional compute increase

#### 5. Adaptive Layer Normalization (AdaLN)

**Purpose**: Condition the model on timestep and text.

```python
class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization.
    Modulates features based on timestep embedding.
    """
    def __init__(self, hidden_size: int):
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),  # scale, shift for 3 sublayers
        )
    
    def forward(self, x, timestep_emb):
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(timestep_emb).chunk(6, dim=-1)
        
        # Apply to normalized input
        x_norm = self.norm(x)
        x_modulated = x_norm * (1 + scale_msa) + shift_msa
        
        return x_modulated, gate_msa
```

**Why AdaLN?**
- Timestep-dependent processing
- Better than simple concatenation
- Standard in modern DiT architectures

---

## 🎬 Video Generator (3D Causal)

### Overview

The Video Generator extends image generation to temporal sequences using 3D Causal Transformers with Flow Matching.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   3D Causal Video Generator                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Text Embeddings [B, T_text, hidden_size]                      │
│  + Noise [B, T_frames, C, H, W]                                │
│  + Timestep t                                                   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  3D Patch Embedding                                      │   │
│  │  - Conv3d: latent_channels → hidden_size                 │   │
│  │  - Patch size: 1×2×2 (temporal×spatial)                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Video DiT Blocks × 8                                    │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │  Spatial Attention (within each frame)           │    │   │
│  │  │  - 2D-RoPE for (x, y) positions                  │    │   │
│  │  │           │                                      │    │   │
│  │  │           ▼                                      │    │   │
│  │  │  Temporal Attention (across frames, causal)      │    │   │
│  │  │  - 1D-RoPE for t positions                       │    │   │
│  │  │  - Causal mask: frame t can only see frames ≤t   │    │   │
│  │  │           │                                      │    │   │
│  │  │           ▼                                      │    │   │
│  │  │  Cross-Attention (to text)                       │    │   │
│  │  │           │                                      │    │   │
│  │  │           ▼                                      │    │   │
│  │  │  Temporal MoE FFN (4 experts)                    │    │   │
│  │  │  - Motion-aware expert routing                   │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  Predicted Velocity [B, T_frames, C, H, W]                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Factorized Spatial-Temporal Attention

**Purpose**: Efficient attention for video by separating spatial and temporal dimensions.

```python
class SpatialAttention(nn.Module):
    """
    Spatial self-attention: each frame attends only within itself.
    Memory: O(T * (H*W)²) instead of O((T*H*W)²)
    """
    def __init__(self, hidden_size: int, num_heads: int = 8):
        self.to_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rope_2d = RoPE2D(head_dim, max_height, max_width)
    
    def forward(self, x, height, width, frames):
        batch_size = x.shape[0]
        hw = height * width
        
        # Reshape to process each frame independently
        # [B, T*H*W, C] -> [B*T, H*W, C]
        x = x.view(batch_size * frames, hw, -1)
        
        # Standard attention within each frame
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        # Apply 2D-RoPE
        cos, sin = self.rope_2d(x, height, width)
        q, k = apply_rope(q, k, cos, sin)
        
        # Attention
        attn_output = scaled_dot_product_attention(q, k, v)
        
        # Reshape back
        return attn_output.view(batch_size, frames * hw, -1)


class TemporalAttention(nn.Module):
    """
    Temporal self-attention: same spatial position attends across frames.
    Causal masking ensures frame t only sees frames ≤ t.
    """
    def __init__(self, hidden_size: int, num_heads: int = 8):
        self.to_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rope_1d = RoPE1D(head_dim, max_len=64)
    
    def forward(self, x, height, width, frames, causal=True):
        batch_size = x.shape[0]
        hw = height * width
        
        # Reshape to process each spatial position across time
        # [B, T*H*W, C] -> [B*H*W, T, C]
        x = x.view(batch_size, frames, hw, -1)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * hw, frames, -1)
        
        # Attention across time
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        # Apply 1D-RoPE for temporal positions
        cos, sin = self.rope_1d(x, frames)
        q, k = apply_rope(q, k, cos, sin)
        
        # Causal attention
        if causal:
            causal_mask = torch.triu(torch.ones(frames, frames), diagonal=1).bool()
            attn_output = scaled_dot_product_attention(q, k, v, attn_mask=~causal_mask)
        else:
            attn_output = scaled_dot_product_attention(q, k, v)
        
        # Reshape back
        attn_output = attn_output.view(batch_size, hw, frames, -1)
        return attn_output.permute(0, 2, 1, 3).reshape(batch_size, frames * hw, -1)
```

**Why Factorized?**
- Full 3D attention: O((T×H×W)²) = O(T²×H²×W²) - prohibitive
- Factorized: O(T×(H×W)² + H×W×T²) - manageable
- Quality comparable to full attention

#### 2. Temporal Expert Router

**Purpose**: Route tokens based on motion patterns and temporal context.

```python
class TemporalExpertRouter(nn.Module):
    """
    Temporal-Aware Expert Router for video generation.
    Routes tokens based on temporal context and motion patterns.
    """
    def __init__(self, hidden_size: int, num_experts: int = 4, top_k: int = 2):
        self.temporal_proj = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
    
    def forward(self, x, temporal_context=None):
        # Incorporate temporal context into routing
        if temporal_context is not None:
            x = x + self.temporal_proj(temporal_context)
        
        router_logits = self.gate(x)
        router_probs = F.softmax(router_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + EPS)
        
        return top_k_probs, top_k_indices
```

**Expert Specialization**:
- Expert 1: Static scenes, backgrounds
- Expert 2: Slow motion, subtle changes
- Expert 3: Fast motion, action scenes
- Expert 4: Camera movement, transitions

#### 3. Temporal MoE Layer

**Purpose**: Process video features with motion-aware expert selection.

```python
class TemporalMoELayer(nn.Module):
    """
    Temporal-Aware MoE Layer for video generation.
    Uses motion-aware routing for expert selection.
    """
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int = 4):
        self.router = TemporalExpertRouter(hidden_size, num_experts, top_k=2)
        self.experts = nn.ModuleList([
            VideoExpert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
        self.shared_expert = VideoExpert(hidden_size, intermediate_size)
    
    def forward(self, x, temporal_context=None):
        # Get routing weights with temporal awareness
        top_k_probs, top_k_indices = self.router(x, temporal_context)
        
        # Compute expert outputs
        output = torch.zeros_like(x)
        for expert_idx in range(self.num_experts):
            for k in range(self.top_k):
                mask = (top_k_indices[:, k] == expert_idx)
                if mask.any():
                    expert_output = self.experts[expert_idx](x[mask])
                    output[mask] += top_k_probs[mask, k:k+1] * expert_output
        
        # Shared expert for common patterns
        output = output + self.shared_expert(x)
        return output
```

#### 4. 3D-RoPE (Combined Spatial-Temporal)

**Purpose**: Full 3D position encoding when needed.

```python
class RoPE3D(nn.Module):
    """
    3D Rotary Position Embedding for (x, y, t) dimensions.
    Used when full 3D attention is needed.
    """
    def __init__(self, dim: int, max_height: int = 64, max_width: int = 64, max_frames: int = 32):
        # Split dimension among three axes
        dim_per_axis = dim // 3
        self.dim_x = dim_per_axis
        self.dim_y = dim_per_axis
        self.dim_t = dim - 2 * dim_per_axis
        
        # Inverse frequencies
        inv_freq_x = 1.0 / (base ** (torch.arange(0, self.dim_x, 2) / self.dim_x))
        inv_freq_y = 1.0 / (base ** (torch.arange(0, self.dim_y, 2) / self.dim_y))
        inv_freq_t = 1.0 / (base ** (torch.arange(0, self.dim_t, 2) / self.dim_t))
    
    def forward(self, x, height, width, frames):
        # Create 3D position encoding
        cos_3d = torch.zeros(frames, height, width, self.dim)
        sin_3d = torch.zeros(frames, height, width, self.dim)
        
        for t in range(frames):
            for y in range(height):
                for w in range(width):
                    cos_3d[t, y, w, :self.dim_x] = cos_x[w]
                    cos_3d[t, y, w, self.dim_x:self.dim_x+self.dim_y] = cos_y[y]
                    cos_3d[t, y, w, self.dim_x+self.dim_y:] = cos_t[t]
        
        return cos_3d.view(-1, self.dim), sin_3d.view(-1, self.dim)
```

---

## 🔄 Generation Process

### Image Generation

```python
@torch.no_grad()
def generate_image(
    self,
    prompt_embeds: torch.Tensor,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
):
    # Initialize from noise
    latents = torch.randn(batch_size, latent_channels, height // 8, width // 8)
    
    # Setup timesteps (1.0 → 0.0)
    timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1)
    
    for i, t in enumerate(timesteps[:-1]):
        t_next = timesteps[i + 1]
        dt = t_next - t
        
        # Classifier-free guidance
        latent_input = torch.cat([latents, latents])
        t_input = torch.cat([t.expand(batch_size), t.expand(batch_size)])
        cond_input = torch.cat([prompt_embeds, null_embeds])
        
        # Predict velocity
        velocity = self.model(latent_input, t_input, cond_input)
        v_cond, v_uncond = velocity.chunk(2)
        velocity = v_uncond + guidance_scale * (v_cond - v_uncond)
        
        # Euler step
        latents = latents + velocity * dt
    
    # Decode latents to image
    image = self.vae.decode(latents)
    return image
```

### Video Generation

```python
@torch.no_grad()
def generate_video(
    self,
    prompt_embeds: torch.Tensor,
    num_frames: int = 16,
    height: int = 256,
    width: int = 256,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
):
    # Initialize from noise
    latents = torch.randn(batch_size, num_frames, latent_channels, height // 8, width // 8)
    
    # Setup timesteps
    timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1)
    
    for i, t in enumerate(timesteps[:-1]):
        t_next = timesteps[i + 1]
        dt = t_next - t
        
        # CFG
        latent_input = torch.cat([latents, latents])
        velocity = self.model(latent_input, t, prompt_embeds)
        v_cond, v_uncond = velocity.chunk(2)
        velocity = v_uncond + guidance_scale * (v_cond - v_uncond)
        
        # Euler step
        latents = latents + velocity * dt
    
    # Decode to video frames
    video = self.vae.decode(latents)
    return video
```

---

## 📊 Generator Specifications

| Generator | Output | Resolution | Features |
|-----------|--------|------------|----------|
| **Image** | Single image | 256-512px | MoE-DiT, Flow Matching, 2D-RoPE |
| **Video** | T frames | 128-384px, 8-32 frames | 3D Causal, Temporal MoE, Flow Matching |

### Multi-Scale Support

Both generators support multi-scale generation:

**Image Scales**: 256, 320, 384, 448, 512px
**Video Spatial Scales**: 128, 192, 256, 320, 384px
**Video Temporal Scales**: 8, 12, 16, 20, 24, 32 frames

---

## 💡 Design Decisions

### Why Flow Matching over DDPM?
- **Simpler objective**: Predict velocity instead of noise
- **Fewer steps**: 50 steps vs. 1000 for DDPM
- **Better quality**: Straighter sampling paths

### Why MoE in Generators?
- **Specialization**: Different experts for different visual patterns
- **Efficiency**: More capacity without proportional compute
- **Quality**: Better handling of diverse content

### Why Factorized Attention for Video?
- **Memory**: Full 3D attention is O((T×H×W)²) - prohibitive
- **Speed**: Factorized is much faster
- **Quality**: Comparable to full attention

### Why Causal Temporal Attention?
- **Consistency**: Each frame builds on previous frames
- **Autoregressive**: Natural for video generation
- **Efficiency**: Can generate frame-by-frame if needed

---

## 🔗 Related Documentation

- [Encoders Documentation](encoders.md) - Corresponding input encoders
- [Components Documentation](components.md) - Shared components
- [Training Documentation](../training/README.md) - How to train generators

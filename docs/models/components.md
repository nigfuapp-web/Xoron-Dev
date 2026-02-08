# 🔧 Components Module Documentation

The Components module contains reusable building blocks used across the Xoron-Dev architecture: attention mechanisms, LoRA adapters, MoE layers, and multimodal projectors.

## 📁 File Structure

```
models/components/
├── __init__.py
├── attention.py    # Flash Attention, Cross-Attention, KV Cache
├── lora.py         # LoRA, DoRA, rsLoRA, LoRA+
├── moe.py          # MoE Router, Experts, Shared Expert
└── projectors.py   # Multimodal Projector, Perceiver Resampler
```

---

## ⚡ Attention Module

### Overview

The attention module provides efficient attention implementations with KV caching for autoregressive generation.

### Flash Attention

**Purpose**: Memory-efficient attention using PyTorch's scaled_dot_product_attention.

```python
class FlashAttention(nn.Module):
    """
    SOTA Flash Attention with KV cache support and FP16-safe Q/K pre-scaling.
    
    Features:
    - KV caching for efficient generation
    - Causal masking
    - Pre-scaled Q/K for FP16 stability
    """
    def __init__(self, dropout: float = 0.0, causal: bool = False, head_dim: int = None):
        self.dropout = dropout
        self.causal = causal
        self._flash_available = flash_attention_available()
        
        # Pre-scaling for FP16 stability
        # By scaling Q and K by head_dim^-0.25 each,
        # Q@K^T is scaled by head_dim^-0.5 (standard scaling)
        self._qk_scale = head_dim ** -0.25 if head_dim else None
    
    def forward(
        self,
        query: torch.Tensor,      # [B, num_heads, seq_len, head_dim]
        key: torch.Tensor,        # [B, num_heads, kv_len, head_dim]
        value: torch.Tensor,      # [B, num_heads, kv_len, head_dim]
        attn_mask: torch.Tensor = None,
        past_key_value: Tuple = None,
        use_cache: bool = False,
    ):
        # Update KV cache if provided
        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=2)
            value = torch.cat([past_key_value[1], value], dim=2)
        
        present_key_value = (key, value) if use_cache else None
        
        # Pre-scale Q and K for FP16 stability
        if self._qk_scale is not None:
            query = query * self._qk_scale
            key = key * self._qk_scale
        
        # Use Flash Attention if available
        if self._flash_available:
            attn_output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal and attn_mask is None,
            )
        else:
            # Fallback to standard attention
            attn_weights = torch.matmul(query, key.transpose(-1, -2))
            if self.causal:
                causal_mask = torch.triu(torch.ones(...), diagonal=1).bool()
                attn_weights.masked_fill_(causal_mask, float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, value)
        
        return attn_output, present_key_value, None
```

**Why Pre-scaling?**
- In FP16, large Q@K^T values can overflow
- Pre-scaling by head_dim^-0.25 each prevents this
- Mathematically equivalent to standard scaling

### KV Cache

**Purpose**: Store key-value pairs for efficient autoregressive generation.

```python
@dataclass
class AttentionKVCache:
    """
    KV Cache for efficient autoregressive attention.
    """
    key_cache: torch.Tensor = None
    value_cache: torch.Tensor = None
    seen_tokens: int = 0
    
    def update(self, key_states, value_states):
        """Append new KV states to cache."""
        if self.key_cache is None:
            self.key_cache = key_states
            self.value_cache = value_states
        else:
            self.key_cache = torch.cat([self.key_cache, key_states], dim=2)
            self.value_cache = torch.cat([self.value_cache, value_states], dim=2)
        
        self.seen_tokens += key_states.shape[2]
        return self.key_cache, self.value_cache
    
    def get_seq_length(self):
        return self.key_cache.shape[2] if self.key_cache is not None else 0
    
    def reset(self):
        self.key_cache = None
        self.value_cache = None
        self.seen_tokens = 0
```

### Multimodal Fusion Layer

**Purpose**: Cross-attention between text and multimodal features.

```python
class MultimodalFusionLayer(nn.Module):
    """
    Cross-attention layer for fusing multimodal features with text.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_flash_attention: bool = True,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Query from text, Key/Value from multimodal
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.flash_attn = FlashAttention(dropout=dropout, causal=False)
    
    def forward(self, text_hidden, multimodal_hidden, attention_mask=None):
        """
        Args:
            text_hidden: [B, T_text, hidden_size] - text features (query)
            multimodal_hidden: [B, T_mm, hidden_size] - multimodal features (key/value)
        """
        residual = text_hidden
        text_hidden = self.norm(text_hidden)
        
        # Project
        query = self.q_proj(text_hidden)
        key = self.k_proj(multimodal_hidden)
        value = self.v_proj(multimodal_hidden)
        
        # Reshape for multi-head attention
        query = query.view(B, T_text, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(B, T_mm, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(B, T_mm, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Cross-attention
        attn_output, _, _ = self.flash_attn(query, key, value, attn_mask=attention_mask)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(B, T_text, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return residual + self.dropout(attn_output)
```

---

## 🔄 LoRA Module

### Overview

The LoRA module implements efficient fine-tuning through low-rank adaptation with several variants.

### LoRA Linear Layer

**Purpose**: Add trainable low-rank matrices to frozen base weights.

```python
class LoRALinear(nn.Module):
    """
    SOTA LoRA layer with multiple variants.
    
    Supports:
    - Standard LoRA
    - DoRA (Weight-Decomposed LoRA)
    - rsLoRA (rank-stabilized scaling)
    
    MEMORY OPTIMIZATION:
    - Shares base weights with original module (no cloning!)
    - Only LoRA params (A, B, magnitude) consume additional memory
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,              # Rank
        lora_alpha: int = 16,    # Scaling factor
        lora_dropout: float = 0.05,
        use_dora: bool = False,  # Weight-Decomposed LoRA
        use_rslora: bool = True, # Rank-stabilized scaling
        base_layer: nn.Linear = None,  # Share weights!
    ):
        # CRITICAL: Share base layer, don't clone!
        if base_layer is not None:
            self.linear = base_layer  # No memory duplication
        else:
            self.linear = nn.Linear(in_features, out_features, bias=False)
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Scaling
        if use_rslora:
            self.scaling = lora_alpha / math.sqrt(r)  # rsLoRA
        else:
            self.scaling = lora_alpha / r  # Standard
        
        self.lora_dropout = nn.Dropout(lora_dropout)
        
        # Initialize: A with Kaiming, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # DoRA: learnable magnitude
        if use_dora:
            self.magnitude = nn.Parameter(torch.ones(out_features))
        
        # Freeze base weights
        self.linear.weight.requires_grad = False
    
    def forward(self, x):
        if self.r > 0 and not self.merged:
            # LoRA update: x @ A^T @ B^T * scaling
            lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
            
            if self.use_dora:
                # DoRA: W' = m * (W + BA) / ||W + BA||
                weight = self.linear.weight + (self.lora_B @ self.lora_A) * self.scaling
                weight_norm = weight.norm(dim=1, keepdim=True)
                weight_normalized = weight / (weight_norm + 1e-6)
                return F.linear(x, weight_normalized * self.magnitude.unsqueeze(1))
            else:
                return self.linear(x) + lora_out
        else:
            return self.linear(x)
    
    def merge_lora_weights(self):
        """Merge LoRA into base weights for inference."""
        if self.r > 0 and not self.merged:
            delta = (self.lora_B @ self.lora_A) * self.scaling
            self.linear.weight.data += delta
            self.merged = True
    
    def unmerge_lora_weights(self):
        """Unmerge for continued training."""
        if self.r > 0 and self.merged:
            self.linear.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
```

### LoRA Variants Explained

#### Standard LoRA
```
Output = W @ x + (B @ A) @ x * (alpha / r)
```
- Simple and effective
- Scaling: `alpha / r`

#### rsLoRA (Rank-Stabilized)
```
Output = W @ x + (B @ A) @ x * (alpha / sqrt(r))
```
- Better for higher ranks
- Scaling: `alpha / sqrt(r)`
- Prevents gradient explosion with large r

#### DoRA (Weight-Decomposed)
```
W' = magnitude * normalize(W + B @ A * scaling)
Output = W' @ x
```
- Decomposes weight into magnitude and direction
- Learns magnitude separately
- Better for some tasks

#### LoRA+ (Different Learning Rates)
```
lr_A = base_lr
lr_B = base_lr * ratio  # e.g., 4x faster
```
- B matrix learns faster than A
- Implemented in optimizer, not layer

### Applying LoRA to Model

```python
def apply_lora_to_model(model, config: LoRAConfig):
    """
    Apply LoRA to specified modules in the model.
    """
    target_modules = config.target_modules or [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
    ]
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Create LoRA layer sharing base weights
                lora_layer = LoRALinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    r=config.r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    use_dora=config.use_dora,
                    use_rslora=config.use_rslora,
                    base_layer=module,  # Share weights!
                )
                # Replace module
                parent = get_parent_module(model, name)
                setattr(parent, name.split('.')[-1], lora_layer)
    
    return model


def get_lora_parameters(model):
    """Get only LoRA parameters for optimizer."""
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name or 'magnitude' in name:
            lora_params.append(param)
    return lora_params


def freeze_non_lora_params(model):
    """Freeze all non-LoRA parameters."""
    for name, param in model.named_parameters():
        if 'lora_' not in name and 'magnitude' not in name:
            param.requires_grad = False
```

---

## 🎯 MoE Module

### Overview

The MoE module implements Mixture of Experts with Aux-Lossless routing and shared expert isolation.

### MoE Router

**Purpose**: Route tokens to appropriate experts without auxiliary loss.

```python
class MoERouter(nn.Module):
    """
    SOTA Router for Mixture of Experts - FP16 native.
    Supports aux-lossless routing.
    """
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.01,
        capacity_factor: float = 1.25,
        aux_lossless: bool = True,
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_lossless = aux_lossless
        
        # Input normalization for stability
        self.input_norm = nn.LayerNorm(hidden_size)
        
        # Routing gate
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)  # Small init
        
        # Aux-lossless: learnable expert bias
        if aux_lossless:
            self.expert_bias = nn.Parameter(torch.zeros(num_experts))
    
    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)
        
        # Normalize for stability
        hidden_norm = self.input_norm(hidden_flat)
        
        # Compute routing logits
        router_logits = self.gate(hidden_norm)
        
        # Add expert bias for load balancing
        if self.aux_lossless:
            router_logits = router_logits + self.expert_bias
        
        # Add noise during training
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise
        
        # Softmax routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize selected probabilities
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + EPS)
        
        return top_k_probs, top_k_indices, router_logits
```

### MoE Expert

**Purpose**: Single expert FFN with SwiGLU activation.

```python
class MoEExpert(nn.Module):
    """
    Single expert FFN with SwiGLU activation - FP16 native.
    """
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        # Small init for FP16 stability
        std = 0.02
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=std)
    
    def forward(self, x):
        # SwiGLU: SiLU(gate(x)) * up(x), then down
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))
```

### MoE Layer with Shared Expert

**Purpose**: Combine routed experts with always-active shared expert.

```python
class MoELayer(nn.Module):
    """
    MoE Layer with Shared Expert Isolation (DeepSeek-style).
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        aux_lossless: bool = True,
    ):
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        
        # Router
        self.router = MoERouter(
            hidden_size, num_experts, top_k=num_experts_per_tok,
            aux_lossless=aux_lossless
        )
        
        # Routed experts
        self.experts = nn.ModuleList([
            MoEExpert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
        
        # Shared expert (always active)
        self.shared_expert = MoEExpert(hidden_size, intermediate_size)
    
    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        x_flat = hidden_states.view(-1, hidden_size)
        
        # Get routing weights
        top_k_probs, top_k_indices, router_logits = self.router(hidden_states)
        
        # Compute weighted expert outputs
        output = torch.zeros_like(x_flat)
        
        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]
            for k in range(self.top_k):
                # Find tokens routed to this expert at position k
                mask = (top_k_indices[:, k] == expert_idx)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = expert(expert_input)
                    weight = top_k_probs[mask, k:k+1]
                    output[mask] = output[mask] + weight * expert_output
        
        # Always add shared expert output
        shared_output = self.shared_expert(x_flat)
        output = output + shared_output
        
        output = output.view(batch_size, seq_len, hidden_size)
        
        # Return output and aux loss (0 for aux-lossless)
        aux_loss = torch.tensor(0.0, device=output.device)
        return output, aux_loss
```

### Why Aux-Lossless?

Traditional MoE requires auxiliary loss to prevent expert collapse:
```python
# Traditional: aux_loss = load_balance_loss + router_z_loss
loss = main_loss + aux_loss_weight * aux_loss
```

Problems:
- Requires tuning aux_loss_weight
- Can conflict with main objective
- Unstable training

Aux-Lossless achieves balance through:
1. **Input normalization**: Stable routing decisions
2. **Expert bias**: Learnable load balancing
3. **Small gate initialization**: Prevents early expert dominance
4. **Shared expert**: Captures common patterns

---

## 🔗 Projectors Module

### Overview

The Projectors module transforms encoder outputs to match the LLM's hidden dimension.

### Multimodal Projector

**Purpose**: Project vision/audio features to LLM dimension.

```python
class MultimodalProjector(nn.Module):
    """
    Projects multimodal features to LLM hidden dimension.
    Supports multiple projection types.
    """
    def __init__(
        self,
        input_dim: int,      # Encoder output dimension
        output_dim: int,     # LLM hidden dimension
        num_tokens: int = 64,  # Number of output tokens
        projector_type: str = "perceiver",  # "perceiver", "mlp", "spatial"
    ):
        self.projector_type = projector_type
        
        if projector_type == "perceiver":
            self.projector = PerceiverResampler(
                input_dim=input_dim,
                output_dim=output_dim,
                num_latents=num_tokens,
            )
        elif projector_type == "mlp":
            self.projector = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
            )
        elif projector_type == "spatial":
            self.projector = SpatialProjector(input_dim, output_dim, num_tokens)
    
    def forward(self, x):
        return self.projector(x)
```

### Perceiver Resampler

**Purpose**: Compress variable-length encoder output to fixed number of tokens.

```python
class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler for compressing encoder outputs.
    Uses cross-attention to resample to fixed number of latents.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_latents: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        self.num_latents = num_latents
        
        # Learnable latent queries
        self.latents = nn.Parameter(torch.randn(1, num_latents, output_dim) * 0.02)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, output_dim)
        
        # Cross-attention layers
        self.layers = nn.ModuleList([
            PerceiverLayer(output_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, input_dim] encoder output (variable length)
        Returns:
            [B, num_latents, output_dim] resampled output (fixed length)
        """
        batch_size = x.shape[0]
        
        # Project input
        x = self.input_proj(x)
        
        # Expand latents for batch
        latents = self.latents.expand(batch_size, -1, -1)
        
        # Cross-attention: latents attend to encoder output
        for layer in self.layers:
            latents = layer(latents, x)
        
        return self.norm(latents)


class PerceiverLayer(nn.Module):
    """Single Perceiver layer with cross-attention and FFN."""
    
    def __init__(self, hidden_size: int, num_heads: int):
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_size)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, latents, encoder_output):
        # Cross-attention
        residual = latents
        latents = self.cross_attn_norm(latents)
        latents, _ = self.cross_attn(latents, encoder_output, encoder_output)
        latents = residual + latents
        
        # FFN
        residual = latents
        latents = self.ffn_norm(latents)
        latents = self.ffn(latents)
        latents = residual + latents
        
        return latents
```

**Why Perceiver Resampler?**
- Handles variable-length encoder outputs
- Compresses to fixed number of tokens
- Learnable compression preserves important information
- Efficient for LLM processing

---

## 📊 Component Specifications

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **Flash Attention** | Efficient attention | KV cache, FP16-safe scaling |
| **LoRA** | Efficient fine-tuning | rsLoRA, DoRA, LoRA+ |
| **MoE** | Sparse computation | Aux-lossless, shared expert |
| **Projector** | Dimension matching | Perceiver resampler |

---

## 🔗 Related Documentation

- [LLM Documentation](llm.md) - How components are used in LLM
- [Encoders Documentation](encoders.md) - Encoder-specific components
- [Generators Documentation](generators.md) - Generator-specific components

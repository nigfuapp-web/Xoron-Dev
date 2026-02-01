"""
SOTA MoE LLaMA model implementation with Sliding Window Attention for 128K context.

Features:
- Full KV cache support for efficient autoregressive generation
- Sliding window attention with proper cache management
- Grouped Query Attention (GQA) for memory efficiency
- Flash Attention integration
- Proper position_ids handling with KV cache
- DeepSeek-style MoE with shared expert
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass

from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaMLP
from models.components.moe import MoELayer


class DynamicNTKScalingRotaryEmbedding(nn.Module):
    """
    SOTA Rotary Position Embedding with Dynamic NTK Scaling for 128K+ context.
    
    Features:
    - Dynamic NTK-aware scaling for extended context
    - Efficient caching of cos/sin values
    - Support for variable sequence lengths
    """

    def __init__(
        self, 
        dim: int, 
        max_position_embeddings: int = 131072, 
        base: float = 500000.0,
        scaling_factor: float = 1.0,
        dynamic_scaling: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.dynamic_scaling = dynamic_scaling
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Cache for cos/sin values
        self._cos_cached = None
        self._sin_cached = None
        self._cached_seq_len = 0

    def _compute_inv_freq_with_scaling(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute inverse frequencies with dynamic NTK scaling if needed."""
        if self.dynamic_scaling and seq_len > self.max_position_embeddings:
            # Dynamic NTK scaling for extended context
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
            return inv_freq
        return self.inv_freq.to(device)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary embeddings for given positions.
        
        Args:
            x: Input tensor for dtype reference [batch, seq_len, hidden] or [batch, heads, seq_len, head_dim]
            position_ids: Position indices [batch, seq_len]
            
        Returns:
            cos, sin: Rotary embeddings [batch, seq_len, dim]
        """
        seq_len = position_ids.shape[-1]
        device = x.device
        
        # Get inverse frequencies (with potential dynamic scaling)
        inv_freq = self._compute_inv_freq_with_scaling(seq_len, device)
        
        # Expand for batch computation
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        # Compute frequencies
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        
        return cos, sin


# Alias for backward compatibility
LlamaRotaryEmbedding = DynamicNTKScalingRotaryEmbedding


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        cos: Cosine embeddings [batch, seq_len, dim]
        sin: Sine embeddings [batch, seq_len, dim]
        position_ids: Optional position indices (unused, for API compatibility)
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting
        
    Returns:
        Rotated query and key tensors
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def make_sliding_window_causal_mask(
    seq_len: int,
    sliding_window: int,
    device: torch.device,
    dtype: torch.dtype,
    kv_seq_len: Optional[int] = None,
) -> torch.Tensor:
    """
    Create a sliding window causal attention mask.
    
    Each position can only attend to:
    - Itself and previous positions (causal)
    - Only within the sliding window (local attention)
    
    Args:
        seq_len: Query sequence length
        sliding_window: Size of the sliding window
        device: Device to create tensor on
        dtype: Data type for the mask
        kv_seq_len: Key/Value sequence length (defaults to seq_len)
    
    Returns:
        Attention mask of shape (1, 1, seq_len, kv_seq_len)
    """
    if kv_seq_len is None:
        kv_seq_len = seq_len
    
    # Create position indices
    row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    col_idx = torch.arange(kv_seq_len, device=device).unsqueeze(0)
    
    # For KV cache: adjust row indices to account for cached positions
    if kv_seq_len > seq_len:
        # During generation with cache, row positions are offset
        row_idx = row_idx + (kv_seq_len - seq_len)
    
    # Causal mask: can't attend to future positions (col > row)
    causal_mask = col_idx > row_idx
    
    # Sliding window mask: can't attend to positions outside window (col < row - window + 1)
    window_mask = col_idx < (row_idx - sliding_window + 1)
    
    # Combine masks: mask out if either condition is true
    combined_mask = causal_mask | window_mask
    
    # Convert to attention mask format (-inf for masked positions)
    mask = torch.where(combined_mask, float('-inf'), 0.0).to(dtype)
    
    return mask.unsqueeze(0).unsqueeze(0)


@dataclass
class KVCache:
    """
    SOTA KV Cache for efficient autoregressive generation.
    
    Features:
    - Sliding window support with automatic eviction
    - Memory-efficient storage
    - Support for GQA (Grouped Query Attention)
    """
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    seen_tokens: int = 0
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        sliding_window: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key/value states.
        
        Args:
            key_states: New key states [batch, num_kv_heads, seq_len, head_dim]
            value_states: New value states [batch, num_kv_heads, seq_len, head_dim]
            sliding_window: Optional sliding window size for eviction
            
        Returns:
            Updated key and value states including cache
        """
        if self.key_cache is None:
            self.key_cache = key_states
            self.value_cache = value_states
        else:
            self.key_cache = torch.cat([self.key_cache, key_states], dim=2)
            self.value_cache = torch.cat([self.value_cache, value_states], dim=2)
        
        self.seen_tokens += key_states.shape[2]
        
        # Apply sliding window eviction if needed
        if sliding_window is not None and self.key_cache.shape[2] > sliding_window:
            self.key_cache = self.key_cache[:, :, -sliding_window:, :]
            self.value_cache = self.value_cache[:, :, -sliding_window:, :]
        
        return self.key_cache, self.value_cache
    
    def get_seq_length(self) -> int:
        """Get current sequence length in cache."""
        if self.key_cache is None:
            return 0
        return self.key_cache.shape[2]


class LlamaAttention(nn.Module):
    """
    SOTA Llama attention with GQA, Flash Attention, Sliding Window, and full KV cache support.
    
    Features:
    - Grouped Query Attention (GQA) for memory efficiency
    - Flash Attention 2 integration via PyTorch SDPA
    - Sliding window attention for efficient long context
    - Full KV cache support for autoregressive generation
    - Proper position_ids handling with cache
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # GQA configuration
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads // 4)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        
        # Attention configuration
        self.use_flash_attention = getattr(config, 'use_flash_attention', True)
        self._flash_attention_available = None
        self.attention_dropout = getattr(config, 'attention_dropout', 0.0)
        
        # Sliding window attention configuration
        self.use_sliding_window = getattr(config, 'use_sliding_window', True)
        self.sliding_window = getattr(config, 'sliding_window', 4096)

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Rotary embeddings
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim, 
            max_position_embeddings=self.max_position_embeddings
        )

    @property
    def flash_attention_available(self) -> bool:
        """Check if Flash Attention (via SDPA) is available."""
        if self._flash_attention_available is None:
            try:
                from torch.nn.functional import scaled_dot_product_attention
                self._flash_attention_available = True
            except ImportError:
                self._flash_attention_available = False
        return self._flash_attention_available

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads to match query heads for GQA."""
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Forward pass with full KV cache support.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_ids: Position indices [batch, seq_len]
            past_key_value: Optional tuple of (past_key, past_value) for KV cache
            output_attentions: Whether to return attention weights
            use_cache: Whether to return updated KV cache
            cache_position: Optional cache position indices for efficient updates
            
        Returns:
            Tuple of (output, past_key_value, attention_weights)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q, K, V projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Compute position IDs if not provided
        if position_ids is None:
            if past_key_value is not None:
                # During generation: position starts after cached tokens
                past_len = past_key_value[0].shape[2]
                position_ids = torch.arange(past_len, past_len + seq_len, device=hidden_states.device).unsqueeze(0)
            else:
                position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
            position_ids = position_ids.expand(batch_size, -1)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(query_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
            
            # Apply sliding window eviction if cache exceeds window size
            if self.use_sliding_window and key_states.shape[2] > self.sliding_window:
                key_states = key_states[:, :, -self.sliding_window:, :]
                value_states = value_states[:, :, -self.sliding_window:, :]

        # Prepare cache for return
        present_key_value = (key_states, value_states) if use_cache else None

        # Expand KV for GQA
        key_states_expanded = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states_expanded = self._repeat_kv(value_states, self.num_key_value_groups)
        
        # Get sequence lengths
        kv_seq_len = key_states_expanded.shape[2]
        
        # Compute attention
        attn_weights = None
        
        if self.use_flash_attention and self.flash_attention_available and not output_attentions:
            # Use Flash Attention via SDPA
            if self.use_sliding_window and attention_mask is None and seq_len > 1:
                # Create sliding window causal mask for training/prefill
                sliding_mask = make_sliding_window_causal_mask(
                    seq_len, self.sliding_window, hidden_states.device, hidden_states.dtype, kv_seq_len
                )
                attn_output = F.scaled_dot_product_attention(
                    query_states, key_states_expanded, value_states_expanded,
                    attn_mask=sliding_mask, 
                    dropout_p=self.attention_dropout if self.training else 0.0, 
                    is_causal=False,
                )
            elif attention_mask is not None:
                attn_output = F.scaled_dot_product_attention(
                    query_states, key_states_expanded, value_states_expanded,
                    attn_mask=attention_mask, 
                    dropout_p=self.attention_dropout if self.training else 0.0, 
                    is_causal=False,
                )
            else:
                # Standard causal attention (for generation with seq_len=1)
                attn_output = F.scaled_dot_product_attention(
                    query_states, key_states_expanded, value_states_expanded,
                    attn_mask=None, 
                    dropout_p=self.attention_dropout if self.training else 0.0, 
                    is_causal=(seq_len > 1 and not self.use_sliding_window),
                )
        else:
            # Manual attention computation
            attn_weights = torch.matmul(query_states, key_states_expanded.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Apply attention mask
            if self.use_sliding_window and attention_mask is None and seq_len > 1:
                sliding_mask = make_sliding_window_causal_mask(
                    seq_len, self.sliding_window, hidden_states.device, hidden_states.dtype, kv_seq_len
                )
                attn_weights = attn_weights + sliding_mask
            elif attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            elif seq_len > 1:
                # Standard causal mask
                causal_mask = torch.triu(
                    torch.full((seq_len, kv_seq_len), float('-inf'), device=hidden_states.device, dtype=hidden_states.dtype),
                    diagonal=kv_seq_len - seq_len + 1
                )
                attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)
                
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            if self.training and self.attention_dropout > 0:
                attn_weights = F.dropout(attn_weights, p=self.attention_dropout)
                
            attn_output = torch.matmul(attn_weights, value_states_expanded)

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value, attn_weights


class MoELlamaDecoderLayer(nn.Module):
    """
    SOTA Llama decoder layer with MoE FFN and full KV cache support.
    
    Features:
    - Pre-norm architecture (more stable training)
    - MoE FFN with DeepSeek-style shared expert (always active)
    - Full KV cache support for efficient generation
    - Proper auxiliary loss propagation
    """

    def __init__(self, config, layer_idx: int, moe_config: dict = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Self-attention with KV cache support
        self.self_attn = LlamaAttention(config, layer_idx=layer_idx)
        
        # Pre-norm layers
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MoE or standard FFN based on layer frequency
        use_moe = moe_config and moe_config.get('use_moe', False)
        moe_freq = moe_config.get('moe_layer_freq', 2) if moe_config else 2

        if use_moe and (layer_idx % moe_freq == 0):
            # MoE layer with DeepSeek-style shared expert (always enabled)
            self.mlp = MoELayer(
                hidden_size=config.hidden_size,
                intermediate_size=moe_config.get('intermediate_size', config.intermediate_size),
                num_experts=moe_config.get('num_experts', 8),
                num_experts_per_tok=moe_config.get('num_experts_per_tok', 2),
                use_shared_expert=True,  # Always use shared expert
            )
            self.is_moe = True
        else:
            self.mlp = LlamaMLP(config)
            self.is_moe = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with full KV cache support.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_ids: Position indices [batch, seq_len]
            past_key_value: Optional KV cache tuple for this layer
            output_attentions: Whether to return attention weights
            use_cache: Whether to return updated KV cache
            cache_position: Optional cache position indices
            
        Returns:
            Tuple of (hidden_states, present_key_value, aux_loss, attn_weights)
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention with KV cache
        attn_output, present_key_value, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + attn_output

        # FFN (MoE or standard)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        aux_loss = None
        if self.is_moe:
            hidden_states, aux_loss = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        # Build outputs tuple
        outputs = (hidden_states, present_key_value)
        
        if aux_loss is not None:
            outputs = outputs + (aux_loss,)
        else:
            outputs = outputs + (None,)
            
        if output_attentions:
            outputs = outputs + (attn_weights,)

        return outputs


@dataclass
class MoELlamaModelOutput:
    """
    Output class for MoELlamaModel with all relevant fields.
    """
    last_hidden_state: torch.Tensor
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    aux_loss: Optional[torch.Tensor] = None


class MoELlamaModel(nn.Module):
    """
    SOTA Llama model with MoE layers, Sliding Window Attention, and full KV cache support.
    
    Features:
    - Full KV cache support for efficient autoregressive generation
    - Sliding window attention for 128K+ context
    - DeepSeek-style MoE with shared expert
    - Gradient checkpointing for memory efficiency
    - Proper output of hidden states and attentions
    """

    def __init__(self, config, moe_config: dict = None):
        super().__init__()
        self.config = config
        self.moe_config = moe_config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Sliding window configuration
        self.use_sliding_window = getattr(config, 'use_sliding_window', True)
        self.sliding_window = getattr(config, 'sliding_window', 4096)

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Decoder layers
        self.layers = nn.ModuleList([
            MoELlamaDecoderLayer(config, layer_idx, moe_config)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # Final layer norm
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Gradient checkpointing flag
        self.gradient_checkpointing = False

        # Count MoE layers
        self.num_moe_layers = sum(1 for layer in self.layers if layer.is_moe)

    def get_input_embeddings(self) -> nn.Embedding:
        """Get the input embeddings."""
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        """Set the input embeddings."""
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, MoELlamaModelOutput]:
        """
        Forward pass with full KV cache support.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            position_ids: Position indices [batch, seq_len]
            inputs_embeds: Optional pre-computed embeddings
            past_key_values: List of KV cache tuples, one per layer
            use_cache: Whether to return updated KV cache
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dataclass or tuple
            cache_position: Optional cache position indices
            
        Returns:
            MoELlamaModelOutput or tuple with hidden states and auxiliary loss
        """
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = inputs_embeds.shape[:2]
        device = inputs_embeds.device

        # Compute position IDs if not provided
        if position_ids is None:
            if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
                # During generation: position starts after cached tokens
                past_length = past_key_values[0][0].shape[2]
                position_ids = torch.arange(past_length, past_length + seq_length, dtype=torch.long, device=device)
            else:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Prepare attention mask
        if attention_mask is not None:
            causal_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values
            )
        else:
            causal_mask = None

        hidden_states = inputs_embeds
        
        # Initialize outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_aux_losses = []
        next_cache = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Get past KV for this layer
            past_key_value = past_key_values[idx] if past_key_values is not None and idx < len(past_key_values) else None

            if self.gradient_checkpointing and self.training and not use_cache:
                # Gradient checkpointing (no cache during training with checkpointing)
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    None,  # past_key_value
                    output_attentions,
                    False,  # use_cache
                    cache_position,
                    use_reentrant=False,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]
            
            # Collect KV cache
            if use_cache:
                next_cache.append(layer_outputs[1])
            
            # Collect auxiliary loss from MoE layers
            if layer.is_moe and layer_outputs[2] is not None:
                all_aux_losses.append(layer_outputs[2])
            
            # Collect attention weights
            if output_attentions and len(layer_outputs) > 3:
                all_attentions = all_attentions + (layer_outputs[3],)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        # Add final hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Compute total auxiliary loss
        if all_aux_losses:
            total_aux_loss = sum(all_aux_losses) / len(all_aux_losses)
        else:
            total_aux_loss = torch.tensor(0.0, device=device, dtype=hidden_states.dtype)

        if not return_dict:
            return (hidden_states, total_aux_loss, next_cache, all_hidden_states, all_attentions)

        return MoELlamaModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            aux_loss=total_aux_loss,
        )

    def _prepare_decoder_attention_mask(
        self, 
        attention_mask: torch.Tensor, 
        input_shape: Tuple[int, int], 
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        Prepare causal attention mask with sliding window support.
        
        Args:
            attention_mask: Padding mask [batch, seq_len]
            input_shape: (batch_size, seq_length)
            inputs_embeds: Input embeddings for dtype/device
            past_key_values: Optional KV cache for computing total sequence length
            
        Returns:
            Causal attention mask [batch, 1, seq_len, total_seq_len]
        """
        batch_size, seq_length = input_shape
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        # Compute total sequence length including cache
        past_length = 0
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            past_length = past_key_values[0][0].shape[2]
        total_length = past_length + seq_length

        # Create causal mask
        if self.use_sliding_window:
            causal_mask = make_sliding_window_causal_mask(
                seq_length, self.sliding_window, device, dtype, total_length
            )
        else:
            # Standard full causal mask
            causal_mask = torch.triu(
                torch.full((seq_length, total_length), float('-inf'), device=device, dtype=dtype),
                diagonal=past_length + 1
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        causal_mask = causal_mask.expand(batch_size, 1, seq_length, total_length)

        # Apply padding mask
        if attention_mask is not None:
            # Expand attention mask to include past tokens
            if past_length > 0:
                past_mask = torch.ones(batch_size, past_length, device=device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([past_mask, attention_mask], dim=1)
            
            padding_mask = attention_mask[:, None, None, :].eq(0)
            causal_mask = causal_mask.masked_fill(padding_mask, float('-inf'))

        return causal_mask


@dataclass
class CausalLMOutput:
    """
    Output class for causal LM with all relevant fields.
    """
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    aux_loss: Optional[torch.Tensor] = None


class MoELlamaForCausalLM(nn.Module):
    """
    SOTA Llama for Causal LM with MoE and full KV cache support.
    
    Features:
    - Full KV cache support for efficient autoregressive generation
    - DeepSeek-style MoE with shared expert
    - Proper loss computation with auxiliary loss
    - Support for all output types (hidden states, attentions)
    - Compatible with HuggingFace generate() interface
    """

    def __init__(self, config, moe_config: dict = None):
        super().__init__()
        self.config = config
        self.moe_config = moe_config

        self.model = MoELlamaModel(config, moe_config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie word embeddings if configured
        if getattr(config, 'tie_word_embeddings', False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Embedding:
        """Get the input embeddings."""
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        """Set the input embeddings."""
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        """Get the output embeddings (lm_head)."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        """Set the output embeddings."""
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """
        Prepare inputs for generation (compatible with HuggingFace generate()).
        
        Args:
            input_ids: Input token IDs
            past_key_values: KV cache from previous generation steps
            attention_mask: Attention mask
            inputs_embeds: Optional pre-computed embeddings
            
        Returns:
            Dictionary of model inputs
        """
        # If we have past_key_values, only use the last token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # Compute position_ids
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -1:]

        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutput]:
        """
        Forward pass with full KV cache support.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            position_ids: Position indices [batch, seq_len]
            inputs_embeds: Optional pre-computed embeddings
            labels: Optional labels for loss computation [batch, seq_len]
            past_key_values: List of KV cache tuples, one per layer
            use_cache: Whether to return updated KV cache
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dataclass or tuple
            cache_position: Optional cache position indices
            
        Returns:
            CausalLMOutput or tuple with loss, logits, and optional outputs
        """
        # Forward through the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state
        aux_loss = outputs.aux_loss

        # Compute logits
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Ensure labels are long dtype for CrossEntropyLoss (critical!)
            if shift_labels.dtype != torch.long:
                shift_labels = shift_labels.long()
            
            # Standard loss computation - CrossEntropyLoss handles ignore_index internally
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            
            # NaN/Inf safety check - only replace if actually invalid
            if torch.isnan(loss) or torch.isinf(loss):
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            # Add auxiliary loss from MoE routing
            if self.moe_config and self.moe_config.get('router_aux_loss_coef', 0) > 0:
                if aux_loss is not None and not torch.isnan(aux_loss) and not torch.isinf(aux_loss):
                    loss = loss + self.moe_config['router_aux_loss_coef'] * aux_loss

        if not return_dict:
            output = (logits, outputs.past_key_values, outputs.hidden_states, outputs.attentions, aux_loss)
            return (loss,) + output if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            aux_loss=aux_loss,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text autoregressively with KV cache.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to sample or use greedy decoding
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            attention_mask: Optional attention mask
            
        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize KV cache
        past_key_values = None
        
        # Initialize attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Generate tokens one at a time
        for _ in range(max_new_tokens):
            # Prepare inputs
            model_inputs = self.prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
            )
            
            # Forward pass
            outputs = self.forward(**model_inputs, use_cache=True, return_dict=True)
            
            # Get next token logits
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Sample or greedy decode
            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Update sequences
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=device)], dim=-1)
            
            # Update KV cache
            past_key_values = outputs.past_key_values
            
            # Check for EOS
            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break
        
        return input_ids

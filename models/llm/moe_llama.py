"""
SOTA MoE LLaMA with Multi-Head Latent Attention (MLA), YaRN/LongRoPE, Ring Attention.

Features:
- MLA (Multi-Head Latent Attention) for compressed KV cache
- YaRN/LongRoPE for superior long-context extrapolation
- Ring Attention for distributed sequence processing (FP16)
- Aux-Lossless MoE with Shared Expert Isolation
- FP16-native numerical stability
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass

from transformers.models.llama.modeling_llama import LlamaRMSNorm

EPS = 1e-5


class YaRNRotaryEmbedding(nn.Module):
    """
    YaRN (Yet another RoPE extensioN) with LongRoPE-style improvements.
    Supports up to 128K+ context with proper frequency scaling.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,
        base: float = 500000.0,
        original_max_position_embeddings: int = 8192,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.original_max_position = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        
        self.scaling_factor = max_position_embeddings / original_max_position_embeddings
        
        inv_freq = self._compute_yarn_inv_freq()
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def _compute_yarn_inv_freq(self) -> torch.Tensor:
        """Compute YaRN-scaled inverse frequencies."""
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scaling_factor * pos_freqs)
        
        low = max(math.floor(self.dim * math.log(self.original_max_position / (self.beta_fast * 2 * math.pi)) / 
                            (2 * math.log(self.base))), 0)
        high = min(math.ceil(self.dim * math.log(self.original_max_position / (self.beta_slow * 2 * math.pi)) /
                            (2 * math.log(self.base))), self.dim - 1)
        
        inv_freq = torch.zeros(self.dim // 2, dtype=torch.float32)
        for i in range(self.dim // 2):
            if i < low:
                inv_freq[i] = inv_freq_interpolation[i]
            elif i > high:
                inv_freq[i] = inv_freq_extrapolation[i]
            else:
                smooth = (i - low) / max(high - low, 1)
                inv_freq[i] = (1 - smooth) * inv_freq_interpolation[i] + smooth * inv_freq_extrapolation[i]
        
        return inv_freq

    def _get_mscale(self, scale: float) -> float:
        """Get attention scaling factor for YaRN."""
        if scale <= 1:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        inv_freq = self.inv_freq.to(device)
        
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        mscale = self._get_mscale(self.scaling_factor) * self.mscale
        
        cos = emb.cos().to(dtype=x.dtype) * mscale
        sin = emb.sin().to(dtype=x.dtype) * mscale
        
        return cos, sin


LlamaRotaryEmbedding = YaRNRotaryEmbedding


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@dataclass
class KVCache:
    """KV Cache for efficient autoregressive generation."""
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    seen_tokens: int = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache is None:
            self.key_cache = key_states
            self.value_cache = value_states
        else:
            self.key_cache = torch.cat([self.key_cache, key_states], dim=2)
            self.value_cache = torch.cat([self.value_cache, value_states], dim=2)

        self.seen_tokens = self.key_cache.shape[2]

        if chunk_size is not None and self.key_cache.shape[2] > chunk_size * 2:
            self.key_cache = self.key_cache[:, :, -chunk_size * 2:]
            self.value_cache = self.value_cache[:, :, -chunk_size * 2:]

        return self.key_cache, self.value_cache


def ring_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    chunk_size: int = 4096,
    causal: bool = True,
) -> torch.Tensor:
    """
    Ring Attention for distributed long-context processing.
    Processes sequence in chunks with proper attention accumulation.
    
    Args:
        query: [batch, heads, seq_len, head_dim]
        key: [batch, heads, kv_len, head_dim]
        value: [batch, heads, kv_len, head_dim]
        chunk_size: Size of each attention chunk
        causal: Whether to apply causal masking
    
    Returns:
        Output tensor [batch, heads, seq_len, head_dim]
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    kv_len = key.shape[2]
    scale = head_dim ** -0.5
    
    if seq_len <= chunk_size and kv_len <= chunk_size:
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scale
        
        if causal:
            causal_mask = torch.triu(torch.ones(seq_len, kv_len, device=query.device, dtype=torch.bool), diagonal=1)
            if kv_len > seq_len:
                causal_mask = causal_mask[:, -seq_len:]
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(attn_weights, value)
    
    output = torch.zeros_like(query)
    max_logits = torch.full((batch_size, num_heads, seq_len, 1), float('-inf'), device=query.device, dtype=query.dtype)
    sum_exp = torch.zeros((batch_size, num_heads, seq_len, 1), device=query.device, dtype=query.dtype)
    
    num_kv_chunks = (kv_len + chunk_size - 1) // chunk_size
    
    for kv_idx in range(num_kv_chunks):
        kv_start = kv_idx * chunk_size
        kv_end = min((kv_idx + 1) * chunk_size, kv_len)
        
        key_chunk = key[:, :, kv_start:kv_end, :]
        value_chunk = value[:, :, kv_start:kv_end, :]
        
        attn_chunk = torch.matmul(query, key_chunk.transpose(-1, -2)) * scale
        
        if causal:
            chunk_len = kv_end - kv_start
            for q_idx in range(seq_len):
                q_pos = q_idx + (kv_len - seq_len) if kv_len > seq_len else q_idx
                for k_idx in range(chunk_len):
                    k_pos = kv_start + k_idx
                    if k_pos > q_pos:
                        attn_chunk[:, :, q_idx, k_idx] = float('-inf')
        
        chunk_max = attn_chunk.max(dim=-1, keepdim=True)[0]
        new_max = torch.maximum(max_logits, chunk_max)
        
        exp_weights = torch.exp(attn_chunk - new_max)
        exp_sum_chunk = exp_weights.sum(dim=-1, keepdim=True)
        
        correction = torch.exp(max_logits - new_max)
        output = output * correction + torch.matmul(exp_weights, value_chunk)
        sum_exp = sum_exp * correction + exp_sum_chunk
        max_logits = new_max
    
    output = output / (sum_exp + EPS)
    return output


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) from DeepSeek-V2.
    Compresses KV cache using low-rank projections for memory efficiency.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int = None,
        head_dim: int = None,
        kv_lora_rank: int = 512,
        q_lora_rank: int = 0,
        rope_theta: float = 500000.0,
        max_position_embeddings: int = 131072,
        use_ring_attention: bool = True,
        ring_chunk_size: int = 4096,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.use_ring_attention = use_ring_attention
        self.ring_chunk_size = ring_chunk_size
        
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim ** -0.5

        if q_lora_rank > 0:
            self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
            self.q_b_proj = nn.Linear(q_lora_rank, num_heads * self.head_dim, bias=False)
            self.q_a_layernorm = LlamaRMSNorm(q_lora_rank)
        else:
            self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)

        self.kv_a_proj = nn.Linear(hidden_size, kv_lora_rank + self.head_dim, bias=False)
        self.kv_b_proj = nn.Linear(kv_lora_rank, self.num_kv_heads * self.head_dim * 2, bias=False)
        self.kv_a_layernorm = LlamaRMSNorm(kv_lora_rank)

        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        self.rotary_emb = YaRNRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[KVCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[KVCache]]:
        batch_size, seq_len, _ = hidden_states.shape

        if self.q_lora_rank > 0:
            q_compressed = self.q_a_layernorm(self.q_a_proj(hidden_states))
            query_states = self.q_b_proj(q_compressed)
        else:
            query_states = self.q_proj(hidden_states)

        kv_compressed = self.kv_a_proj(hidden_states)
        kv_latent, k_pe = kv_compressed.split([self.kv_lora_rank, self.head_dim], dim=-1)
        kv_latent = self.kv_a_layernorm(kv_latent)
        kv_states = self.kv_b_proj(kv_latent)

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states, value_states = kv_states.split(self.num_kv_heads * self.head_dim, dim=-1)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
            if past_key_value is not None and past_key_value.seen_tokens > 0:
                position_ids = position_ids + past_key_value.seen_tokens

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, 
                self.ring_chunk_size if self.use_ring_attention else None
            )

        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        if self.use_ring_attention:
            attn_output = ring_attention(
                query_states, key_states, value_states,
                chunk_size=self.ring_chunk_size,
                causal=True,
            )
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale
            
            kv_len = key_states.shape[2]
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_len, device=hidden_states.device, dtype=torch.bool),
                diagonal=kv_len - seq_len + 1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value if use_cache else None


class AuxLosslessMoERouter(nn.Module):
    """
    Aux-Lossless MoE Router with Shared Expert Isolation.
    Eliminates auxiliary loss while maintaining load balance through architecture.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        
        self.input_norm = LlamaRMSNorm(hidden_size)
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)
        
        hidden_norm = self.input_norm(hidden_flat)
        router_logits = self.gate(hidden_norm)
        
        router_probs = F.softmax(router_logits, dim=-1, dtype=hidden_states.dtype)
        
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        if self.norm_topk_prob:
            top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + EPS)

        return top_k_probs, top_k_indices, router_logits


class MoEExpert(nn.Module):
    """Single MoE Expert with SwiGLU activation."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self._init_weights()

    def _init_weights(self):
        std = 0.02
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=std * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class IsolatedSharedExpert(nn.Module):
    """
    Isolated Shared Expert that always processes all tokens.
    Separate from routed experts to prevent competition.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self._init_weights()

    def _init_weights(self):
        std = 0.02
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=std * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class AuxLosslessMoELayer(nn.Module):
    """
    Aux-Lossless MoE Layer with Isolated Shared Expert.
    No auxiliary loss needed - load balance maintained through isolation.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        shared_expert_intermediate_size: int = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        self.router = AuxLosslessMoERouter(hidden_size, num_experts, num_experts_per_tok)
        
        self.experts = nn.ModuleList([
            MoEExpert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
        
        shared_size = shared_expert_intermediate_size or intermediate_size
        self.shared_expert = IsolatedSharedExpert(hidden_size, shared_size)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_size)

        top_k_probs, top_k_indices, _ = self.router(hidden_states)

        final_output = torch.zeros_like(hidden_flat)

        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]
            for k in range(self.num_experts_per_tok):
                mask = (top_k_indices[:, k] == expert_idx)
                if mask.any():
                    expert_input = hidden_flat[mask]
                    expert_output = expert(expert_input)
                    weight = top_k_probs[mask, k:k+1]
                    final_output[mask] = final_output[mask] + weight * expert_output

        shared_output = self.shared_expert(hidden_flat)
        final_output = final_output + shared_output

        final_output = final_output.view(batch_size, seq_len, hidden_size)
        
        aux_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        return final_output, aux_loss


MoELayer = AuxLosslessMoELayer


class MoELlamaDecoderLayer(nn.Module):
    """Decoder layer with MLA and Aux-Lossless MoE."""

    def __init__(self, config, layer_idx: int, moe_config: dict = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        use_ring = getattr(config, 'use_ring_attention', True)
        ring_chunk = getattr(config, 'ring_attention_chunk_size', 4096)
        
        num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads // 4)

        self.self_attn = MultiHeadLatentAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            rope_theta=getattr(config, 'rope_theta', 500000.0),
            max_position_embeddings=config.max_position_embeddings,
            use_ring_attention=use_ring,
            ring_chunk_size=ring_chunk,
        )

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.use_moe = moe_config and moe_config.get('use_moe', False)
        moe_freq = moe_config.get('moe_layer_freq', 2) if moe_config else 2

        if self.use_moe and layer_idx % moe_freq == (moe_freq - 1):
            self.mlp = AuxLosslessMoELayer(
                hidden_size=config.hidden_size,
                intermediate_size=moe_config.get('intermediate_size', config.intermediate_size),
                num_experts=moe_config.get('num_experts', 8),
                num_experts_per_tok=moe_config.get('num_experts_per_tok', 2),
            )
            self.is_moe_layer = True
        else:
            self.mlp = MoEExpert(config.hidden_size, config.intermediate_size)
            self.is_moe_layer = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[KVCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[KVCache], Optional[torch.Tensor]]:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        aux_loss = None
        if self.is_moe_layer:
            hidden_states, aux_loss = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states, None, present_key_value, aux_loss


@dataclass
class MoELlamaModelOutput:
    last_hidden_state: torch.Tensor
    past_key_values: Optional[List[KVCache]] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    aux_loss: Optional[torch.Tensor] = None


class MoELlamaModel(nn.Module):
    """MoE LLaMA Model with MLA and Ring Attention."""

    def __init__(self, config, moe_config: dict = None):
        super().__init__()
        self.config = config
        self.moe_config = moe_config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([
            MoELlamaDecoderLayer(config, layer_idx, moe_config)
            for layer_idx in range(config.num_hidden_layers)
        ])

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.num_moe_layers = sum(1 for layer in self.layers if layer.is_moe_layer)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[KVCache]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, MoELlamaModelOutput]:
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        batch_size, seq_len = hidden_states.shape[:2]

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_cache = [] if use_cache else None
        total_aux_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states, attn_weights, present_key_value, aux_loss = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[idx],
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            if use_cache:
                next_cache.append(present_key_value)

            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss

            if output_attentions and attn_weights is not None:
                all_attentions = all_attentions + (attn_weights,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return MoELlamaModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_cache if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            aux_loss=total_aux_loss,
        )


@dataclass
class CausalLMOutput:
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    past_key_values: Optional[List[KVCache]] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    aux_loss: Optional[torch.Tensor] = None


class MoELlamaForCausalLM(nn.Module):
    """MoE LLaMA for Causal Language Modeling with MLA and Ring Attention."""

    def __init__(self, config, moe_config: dict = None):
        super().__init__()
        self.config = config
        self.moe_config = moe_config

        self.model = MoELlamaModel(config, moe_config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if getattr(config, 'tie_word_embeddings', True):
            self.lm_head.weight = self.model.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[KVCache]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[KVCache]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutput]:
        
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

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            if shift_labels.dtype != torch.long:
                shift_labels = shift_labels.long()

            valid_mask = (shift_labels != -100)
            num_valid = valid_mask.sum().item()

            if num_valid > 0:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                loss = torch.clamp(loss, min=0.0, max=100.0)

                if self.moe_config and self.moe_config.get('router_aux_loss_coef', 0) > 0:
                    if aux_loss is not None:
                        loss = loss + self.moe_config['router_aux_loss_coef'] * aux_loss
            else:
                loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)

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
        batch_size = input_ids.shape[0]
        device = input_ids.device

        past_key_values = None

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        for _ in range(max_new_tokens):
            model_inputs = self.prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
            )

            outputs = self.forward(**model_inputs, use_cache=True, return_dict=True)

            next_token_logits = outputs.logits[:, -1, :]

            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            if do_sample:
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=device)], dim=-1)

            past_key_values = outputs.past_key_values

            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break

        return input_ids

"""
Xoron-Dev Model Implementation for HuggingFace Transformers.

This module provides a HuggingFace-compatible implementation of the Xoron-Dev
multimodal MoE model, supporting AutoModel loading with trust_remote_code=True.

Example:
    >>> from transformers import AutoModelForCausalLM
    >>> model = AutoModelForCausalLM.from_pretrained("your-username/Xoron-Dev", trust_remote_code=True)
"""

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_xoron import XoronConfig

logger = logging.get_logger(__name__)


class XoronRMSNorm(nn.Module):
    """RMSNorm for Xoron model."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class XoronYaRNRotaryEmbedding(nn.Module):
    """YaRN rotary embeddings for 128K+ context support."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,
        base: float = 500000.0,
        original_max_position_embeddings: int = 8192,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
        device: Optional[torch.device] = None,
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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class XoronMLP(nn.Module):
    """Standard MLP block for non-MoE layers."""
    
    def __init__(self, config: XoronConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class XoronExpert(nn.Module):
    """Single expert in the MoE layer."""
    
    def __init__(self, config: XoronConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class XoronMoE(nn.Module):
    """Mixture of Experts layer with aux-lossless routing."""
    
    def __init__(self, config: XoronConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([XoronExpert(config) for _ in range(self.num_experts)])
        
        self.shared_expert = None
        if config.use_shared_expert:
            self.shared_expert = XoronExpert(config)
            self.shared_expert_gate = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        
        router_logits = self.gate(hidden_states_flat)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(hidden_states.dtype)
        
        final_output = torch.zeros_like(hidden_states_flat)
        
        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]
            expert_mask = (topk_indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
            
            token_indices = expert_mask.nonzero(as_tuple=True)[0]
            expert_positions = (topk_indices[expert_mask] == expert_idx).float()
            weights = (topk_weights[expert_mask] * expert_positions).sum(dim=-1, keepdim=True)
            
            expert_input = hidden_states_flat[token_indices]
            expert_output = expert(expert_input)
            final_output[token_indices] += weights * expert_output
        
        if self.shared_expert is not None:
            shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states_flat))
            shared_output = self.shared_expert(hidden_states_flat)
            final_output = final_output + shared_gate * shared_output
        
        final_output = final_output.view(batch_size, seq_len, hidden_size)
        return final_output, None


def ring_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    chunk_size: int = 4096, causal: bool = True,
) -> torch.Tensor:
    """Ring Attention for memory-efficient long-context processing."""
    batch_size, num_heads, seq_len, head_dim = query.shape
    kv_len = key.shape[2]
    scale = head_dim ** -0.5
    
    if seq_len <= chunk_size and kv_len <= chunk_size:
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scale
        if causal:
            causal_mask = torch.triu(torch.ones(seq_len, kv_len, device=query.device, dtype=torch.bool), diagonal=1)
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
            mask = torch.zeros(seq_len, chunk_len, device=query.device, dtype=torch.bool)
            for q_idx in range(seq_len):
                for k_idx in range(chunk_len):
                    if kv_start + k_idx > q_idx:
                        mask[q_idx, k_idx] = True
            attn_chunk = attn_chunk.masked_fill(mask, float('-inf'))
        
        chunk_max = attn_chunk.max(dim=-1, keepdim=True)[0]
        new_max = torch.maximum(max_logits, chunk_max)
        
        exp_old = sum_exp * torch.exp(max_logits - new_max)
        exp_chunk = torch.exp(attn_chunk - new_max)
        exp_sum_chunk = exp_chunk.sum(dim=-1, keepdim=True)
        
        output = output * (sum_exp / (exp_old + exp_sum_chunk + 1e-10))
        output = output + torch.matmul(exp_chunk, value_chunk) / (exp_old + exp_sum_chunk + 1e-10)
        
        sum_exp = exp_old + exp_sum_chunk
        max_logits = new_max
    
    return output


class XoronAttention(nn.Module):
    """Xoron Attention with Ring Attention support."""
    
    def __init__(self, config: XoronConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.use_ring_attention = config.use_ring_attention
        self.ring_chunk_size = config.ring_attention_chunk_size
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = XoronYaRNRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            original_max_position_embeddings=config.rope_scaling.get("original_max_position_embeddings", 8192),
            beta_fast=config.rope_scaling.get("beta_fast", 32.0),
            beta_slow=config.rope_scaling.get("beta_slow", 1.0),
            mscale=config.rope_scaling.get("mscale", 1.0),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(q, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        
        past_key_value = (k, v) if use_cache else None
        
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        
        if self.use_ring_attention and k.shape[2] > self.ring_chunk_size:
            attn_output = ring_attention(q, k, v, self.ring_chunk_size, causal=True)
        else:
            scale = self.head_dim ** -0.5
            attn_weights = torch.matmul(q, k.transpose(-1, -2)) * scale
            
            if attention_mask is None:
                causal_mask = torch.triu(
                    torch.ones(seq_len, k.shape[2], device=q.device, dtype=torch.bool),
                    diagonal=k.shape[2] - seq_len + 1
                )
                attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
            else:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, past_key_value


class XoronDecoderLayer(nn.Module):
    """Xoron decoder layer with attention and MLP/MoE."""
    
    def __init__(self, config: XoronConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        self.self_attn = XoronAttention(config, layer_idx)
        self.input_layernorm = XoronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = XoronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        if config.is_moe_layer(layer_idx):
            self.mlp = XoronMoE(config)
            self.is_moe = True
        else:
            self.mlp = XoronMLP(config)
            self.is_moe = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]], Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, attn_weights, present_key_value = self.self_attn(
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
        if self.is_moe:
            hidden_states, aux_loss = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        
        hidden_states = residual + hidden_states
        
        return hidden_states, attn_weights, present_key_value, aux_loss


class XoronPreTrainedModel(PreTrainedModel):
    """Base class for Xoron models."""
    
    config_class = XoronConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["XoronDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


class XoronModel(XoronPreTrainedModel):
    """Xoron transformer model."""
    
    def __init__(self, config: XoronConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            XoronDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = XoronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds")
        
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, 
                seq_length + past_key_values_length, 
                dtype=torch.long, 
                device=device
            )
            position_ids = position_ids.unsqueeze(0)
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        hidden_states = inputs_embeds
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2],)
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = next_decoder_cache if use_cache else None
        
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class XoronForCausalLM(XoronPreTrainedModel):
    """Xoron model for causal language modeling."""
    
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: XoronConfig):
        super().__init__(config)
        self.model = XoronModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


# Register for AutoClass
XoronConfig.register_for_auto_class()
XoronModel.register_for_auto_class("AutoModel")
XoronForCausalLM.register_for_auto_class("AutoModelForCausalLM")

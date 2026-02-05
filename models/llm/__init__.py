"""LLM module for Xoron-Dev.

SOTA Features:
- YaRN Rotary Embeddings for extended context
- Multi-Head Latent Attention (MLA) for compressed KV cache
- Ring Attention for distributed long-context processing
- Aux-Lossless MoE Router with load balancing
- Isolated Shared Expert for stable training
"""

from models.llm.moe_llama import (
    MoELlamaModel,
    MoELlamaForCausalLM,
    MoELlamaDecoderLayer,
    MoELlamaModelOutput,
    CausalLMOutput,
    YaRNRotaryEmbedding,
    MultiHeadLatentAttention,
    AuxLosslessMoERouter,
    AuxLosslessMoELayer,
    MoEExpert,
    IsolatedSharedExpert,
    KVCache,
    ring_attention,
    rotate_half,
    apply_rotary_pos_emb,
)

__all__ = [
    'MoELlamaModel',
    'MoELlamaForCausalLM',
    'MoELlamaDecoderLayer',
    'MoELlamaModelOutput',
    'CausalLMOutput',
    'YaRNRotaryEmbedding',
    'MultiHeadLatentAttention',
    'AuxLosslessMoERouter',
    'AuxLosslessMoELayer',
    'MoEExpert',
    'IsolatedSharedExpert',
    'KVCache',
    'ring_attention',
    'rotate_half',
    'apply_rotary_pos_emb',
]

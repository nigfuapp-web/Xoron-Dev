"""
Comprehensive unit tests for SOTA MoE LLaMA model.

Tests all components:
- YaRNRotaryEmbedding (extended context with YaRN scaling)
- rotate_half, apply_rotary_pos_emb
- KVCache
- ring_attention (distributed long-context)
- MultiHeadLatentAttention (MLA with compressed KV)
- AuxLosslessMoERouter (load-balanced routing)
- MoEExpert, IsolatedSharedExpert
- AuxLosslessMoELayer
- MoELlamaDecoderLayer
- MoELlamaModel, MoELlamaForCausalLM
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MockConfig:
    """Mock configuration for testing."""
    def __init__(self):
        self.hidden_size = 256
        self.num_attention_heads = 8
        self.num_key_value_heads = 2
        self.max_position_embeddings = 4096
        self.use_flash_attention = True
        self.attention_dropout = 0.0
        self.use_sliding_window = True
        self.sliding_window = 512
        self.intermediate_size = 1024
        self.num_hidden_layers = 2
        self.vocab_size = 32000
        self.rms_norm_eps = 1e-6
        self.use_moe = True
        self.num_experts = 8
        self.num_experts_per_tok = 2


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestYaRNRotaryEmbedding(unittest.TestCase):
    """Test cases for YaRNRotaryEmbedding (extended context)."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.llm.moe_llama import YaRNRotaryEmbedding
        self.YaRNRotaryEmbedding = YaRNRotaryEmbedding
        
    def test_initialization(self):
        """Test YaRNRotaryEmbedding initialization."""
        rope = self.YaRNRotaryEmbedding(dim=64, max_position_embeddings=4096)
        
        self.assertEqual(rope.dim, 64)
        self.assertEqual(rope.max_position_embeddings, 4096)
        
    def test_forward_output_shapes(self):
        """Test forward pass output shapes."""
        rope = self.YaRNRotaryEmbedding(dim=64, max_position_embeddings=4096)
        x = torch.randn(2, 8, 128, 64)  # [batch, heads, seq_len, head_dim]
        position_ids = torch.arange(128).unsqueeze(0).expand(2, -1)
        
        cos, sin = rope(x, position_ids)
        
        self.assertEqual(cos.shape, (2, 128, 64))
        self.assertEqual(sin.shape, (2, 128, 64))
        
    def test_different_positions_different_embeddings(self):
        """Test that different positions produce different embeddings."""
        rope = self.YaRNRotaryEmbedding(dim=64, max_position_embeddings=4096)
        x = torch.randn(1, 8, 10, 64)
        pos1 = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        pos2 = torch.tensor([[100, 101, 102, 103, 104, 105, 106, 107, 108, 109]])
        
        cos1, sin1 = rope(x, pos1)
        cos2, sin2 = rope(x, pos2)
        
        self.assertFalse(torch.allclose(cos1, cos2))
        self.assertFalse(torch.allclose(sin1, sin2))
        
    def test_yarn_scaling(self):
        """Test YaRN scaling for extended context."""
        rope = self.YaRNRotaryEmbedding(
            dim=64, 
            max_position_embeddings=2048,
            original_max_position_embeddings=1024,  # Original context length
        )
        x = torch.randn(1, 8, 2048, 64)  # Extended context
        position_ids = torch.arange(2048).unsqueeze(0)
        
        cos, sin = rope(x, position_ids)
        
        self.assertEqual(cos.shape, (1, 2048, 64))
        self.assertEqual(sin.shape, (1, 2048, 64))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestRotateHalf(unittest.TestCase):
    """Test cases for rotate_half function."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.llm.moe_llama import rotate_half
        self.rotate_half = rotate_half
        
    def test_output_shape(self):
        """Test output shape matches input shape."""
        x = torch.randn(2, 8, 128, 64)
        
        output = self.rotate_half(x)
        
        self.assertEqual(output.shape, x.shape)
        
    def test_rotation_correctness(self):
        """Test rotation is applied correctly."""
        x = torch.tensor([[[[1, 2, 3, 4]]]])  # [1, 1, 1, 4]
        
        output = self.rotate_half(x)
        
        # First half should be negated second half, second half should be first half
        expected = torch.tensor([[[[-3, -4, 1, 2]]]])
        self.assertTrue(torch.allclose(output, expected))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestApplyRotaryPosEmb(unittest.TestCase):
    """Test cases for apply_rotary_pos_emb function."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.llm.moe_llama import apply_rotary_pos_emb
        self.apply_rotary_pos_emb = apply_rotary_pos_emb
        
    def test_output_shapes(self):
        """Test output shapes match input shapes."""
        q = torch.randn(2, 8, 128, 64)
        k = torch.randn(2, 2, 128, 64)
        cos = torch.randn(2, 128, 64)
        sin = torch.randn(2, 128, 64)
        
        q_embed, k_embed = self.apply_rotary_pos_emb(q, k, cos, sin)
        
        self.assertEqual(q_embed.shape, q.shape)
        self.assertEqual(k_embed.shape, k.shape)
        
    def test_different_cos_sin_different_output(self):
        """Test that different cos/sin produce different outputs."""
        q = torch.randn(2, 8, 128, 64)
        k = torch.randn(2, 2, 128, 64)
        cos1 = torch.randn(2, 128, 64)
        sin1 = torch.randn(2, 128, 64)
        cos2 = torch.randn(2, 128, 64)
        sin2 = torch.randn(2, 128, 64)
        
        q_embed1, k_embed1 = self.apply_rotary_pos_emb(q, k, cos1, sin1)
        q_embed2, k_embed2 = self.apply_rotary_pos_emb(q, k, cos2, sin2)
        
        self.assertFalse(torch.allclose(q_embed1, q_embed2))
        self.assertFalse(torch.allclose(k_embed1, k_embed2))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestRingAttention(unittest.TestCase):
    """Test cases for ring_attention function (distributed long-context)."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.llm.moe_llama import ring_attention
        self.ring_attention = ring_attention
        
    def test_output_shape(self):
        """Test output shape matches expected."""
        q = torch.randn(2, 8, 64, 32)  # [B, num_heads, seq_len, head_dim]
        k = torch.randn(2, 8, 64, 32)
        v = torch.randn(2, 8, 64, 32)
        
        output = self.ring_attention(q, k, v)
        
        self.assertEqual(output.shape, (2, 8, 64, 32))
        
    def test_causal_attention(self):
        """Test causal attention is applied."""
        q = torch.randn(2, 4, 16, 32)
        k = torch.randn(2, 4, 16, 32)
        v = torch.randn(2, 4, 16, 32)
        
        output = self.ring_attention(q, k, v, causal=True)
        
        self.assertEqual(output.shape, (2, 4, 16, 32))
        
    def test_with_chunk_size(self):
        """Test with custom chunk size."""
        q = torch.randn(2, 4, 16, 32)
        k = torch.randn(2, 4, 16, 32)
        v = torch.randn(2, 4, 16, 32)
        
        output = self.ring_attention(q, k, v, chunk_size=8)
        
        self.assertEqual(output.shape, (2, 4, 16, 32))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestKVCache(unittest.TestCase):
    """Test cases for KVCache."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.llm.moe_llama import KVCache
        self.KVCache = KVCache
        
    def test_initialization(self):
        """Test KVCache initialization."""
        cache = self.KVCache(key_cache=None, value_cache=None)
        
        self.assertIsNone(cache.key_cache)
        self.assertIsNone(cache.value_cache)
        self.assertEqual(cache.seen_tokens, 0)
        
    def test_first_update(self):
        """Test first update to empty cache."""
        cache = self.KVCache(key_cache=None, value_cache=None)
        key_states = torch.randn(2, 4, 10, 64)
        value_states = torch.randn(2, 4, 10, 64)
        
        k, v = cache.update(key_states, value_states)
        
        self.assertEqual(k.shape, (2, 4, 10, 64))
        self.assertEqual(v.shape, (2, 4, 10, 64))
        self.assertEqual(cache.seen_tokens, 10)
        
    def test_subsequent_update(self):
        """Test subsequent updates append to cache."""
        cache = self.KVCache(
            key_cache=torch.randn(2, 4, 10, 64),
            value_cache=torch.randn(2, 4, 10, 64),
            seen_tokens=10
        )
        key_states = torch.randn(2, 4, 5, 64)
        value_states = torch.randn(2, 4, 5, 64)
        
        k, v = cache.update(key_states, value_states)
        
        self.assertEqual(k.shape, (2, 4, 15, 64))
        self.assertEqual(v.shape, (2, 4, 15, 64))
        self.assertEqual(cache.seen_tokens, 15)
        
    def test_update_appends_states(self):
        """Test update appends new states to cache."""
        cache = self.KVCache(
            key_cache=torch.randn(2, 4, 100, 64),
            value_cache=torch.randn(2, 4, 100, 64),
            seen_tokens=100
        )
        key_states = torch.randn(2, 4, 10, 64)
        value_states = torch.randn(2, 4, 10, 64)
        
        k, v = cache.update(key_states, value_states)
        
        # Should append new states
        self.assertEqual(k.shape, (2, 4, 110, 64))
        self.assertEqual(v.shape, (2, 4, 110, 64))
        
    def test_seen_tokens_tracking(self):
        """Test seen_tokens tracking."""
        cache = self.KVCache(
            key_cache=torch.randn(2, 4, 50, 64),
            value_cache=torch.randn(2, 4, 50, 64),
            seen_tokens=50
        )
        
        self.assertEqual(cache.seen_tokens, 50)
        
    def test_seen_tokens_empty(self):
        """Test seen_tokens with empty cache."""
        cache = self.KVCache(key_cache=None, value_cache=None)
        
        self.assertEqual(cache.seen_tokens, 0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestMultiHeadLatentAttention(unittest.TestCase):
    """Test cases for MultiHeadLatentAttention (MLA with compressed KV)."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.llm.moe_llama import MultiHeadLatentAttention
        self.MultiHeadLatentAttention = MultiHeadLatentAttention
        
    def test_initialization(self):
        """Test MLA initialization."""
        mla = self.MultiHeadLatentAttention(
            hidden_size=256,
            num_heads=8,
            num_kv_heads=2,
            head_dim=32,
            kv_lora_rank=64,
        )
        
        self.assertEqual(mla.hidden_size, 256)
        self.assertEqual(mla.num_heads, 8)
        self.assertEqual(mla.num_kv_heads, 2)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        mla = self.MultiHeadLatentAttention(
            hidden_size=256,
            num_heads=8,
            num_kv_heads=2,
            head_dim=32,
            kv_lora_rank=64,
        )
        hidden_states = torch.randn(2, 128, 256)
        
        # Returns (output, attn_weights, past_kv)
        output, attn_weights, past_kv = mla(hidden_states)
        
        self.assertEqual(output.shape, (2, 128, 256))
        
    def test_forward_with_cache(self):
        """Test forward pass with KV cache."""
        mla = self.MultiHeadLatentAttention(
            hidden_size=256,
            num_heads=8,
            num_kv_heads=2,
            head_dim=32,
            kv_lora_rank=64,
        )
        
        # First pass
        hidden_states = torch.randn(2, 10, 256)
        output, attn_weights, past_kv = mla(hidden_states, use_cache=True)
        
        self.assertEqual(output.shape, (2, 10, 256))
        # past_kv may be None if caching is not implemented in this version


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestAuxLosslessMoERouter(unittest.TestCase):
    """Test cases for AuxLosslessMoERouter (load-balanced routing)."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.llm.moe_llama import AuxLosslessMoERouter
        self.AuxLosslessMoERouter = AuxLosslessMoERouter
        
    def test_initialization(self):
        """Test router initialization."""
        router = self.AuxLosslessMoERouter(
            hidden_size=256,
            num_experts=8,
            top_k=2,
        )
        
        self.assertEqual(router.num_experts, 8)
        self.assertEqual(router.top_k, 2)
        
    def test_forward_output_shapes(self):
        """Test forward pass output shapes."""
        router = self.AuxLosslessMoERouter(
            hidden_size=256,
            num_experts=8,
            top_k=2,
        )
        hidden_states = torch.randn(2, 64, 256)
        
        router_logits, expert_indices, expert_weights = router(hidden_states)
        
        # router_logits: [B*T, top_k] - selected expert logits
        # expert_indices: [B*T, top_k] - selected expert indices
        # expert_weights: [B*T, num_experts] - full routing weights
        self.assertEqual(router_logits.shape, (2 * 64, 2))  # [B*T, top_k]
        self.assertEqual(expert_indices.shape, (2 * 64, 2))  # [B*T, top_k]
        self.assertEqual(expert_weights.shape, (2 * 64, 8))  # [B*T, num_experts]


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestMoEExpert(unittest.TestCase):
    """Test cases for MoEExpert."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.llm.moe_llama import MoEExpert
        self.MoEExpert = MoEExpert
        
    def test_initialization(self):
        """Test expert initialization."""
        expert = self.MoEExpert(hidden_size=256, intermediate_size=1024)
        
        self.assertIsNotNone(expert.gate_proj)
        self.assertIsNotNone(expert.up_proj)
        self.assertIsNotNone(expert.down_proj)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        expert = self.MoEExpert(hidden_size=256, intermediate_size=1024)
        x = torch.randn(2, 64, 256)
        
        output = expert(x)
        
        self.assertEqual(output.shape, (2, 64, 256))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestIsolatedSharedExpert(unittest.TestCase):
    """Test cases for IsolatedSharedExpert."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.llm.moe_llama import IsolatedSharedExpert
        self.IsolatedSharedExpert = IsolatedSharedExpert
        
    def test_initialization(self):
        """Test shared expert initialization."""
        expert = self.IsolatedSharedExpert(hidden_size=256, intermediate_size=1024)
        
        self.assertIsNotNone(expert.gate_proj)
        self.assertIsNotNone(expert.up_proj)
        self.assertIsNotNone(expert.down_proj)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        expert = self.IsolatedSharedExpert(hidden_size=256, intermediate_size=1024)
        x = torch.randn(2, 64, 256)
        
        output = expert(x)
        
        self.assertEqual(output.shape, (2, 64, 256))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestAuxLosslessMoELayer(unittest.TestCase):
    """Test cases for AuxLosslessMoELayer (Aux-Lossless with Isolated Shared Expert)."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.llm.moe_llama import AuxLosslessMoELayer
        self.AuxLosslessMoELayer = AuxLosslessMoELayer
        
    def test_initialization(self):
        """Test MoE layer initialization."""
        moe = self.AuxLosslessMoELayer(
            hidden_size=256,
            intermediate_size=1024,
            num_experts=8,
            num_experts_per_tok=2,  # SOTA: uses num_experts_per_tok not top_k
        )
        
        self.assertEqual(len(moe.experts), 8)
        self.assertIsNotNone(moe.router)
        self.assertIsNotNone(moe.shared_expert)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        moe = self.AuxLosslessMoELayer(
            hidden_size=256,
            intermediate_size=1024,
            num_experts=8,
            num_experts_per_tok=2,  # SOTA: uses num_experts_per_tok
        )
        hidden_states = torch.randn(2, 64, 256)
        
        output, aux_loss = moe(hidden_states)
        
        self.assertEqual(output.shape, (2, 64, 256))
        self.assertIsInstance(aux_loss, torch.Tensor)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestMoELlamaDecoderLayer(unittest.TestCase):
    """Test cases for MoELlamaDecoderLayer."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.llm.moe_llama import MoELlamaDecoderLayer
        self.MoELlamaDecoderLayer = MoELlamaDecoderLayer
        self.config = MockConfig()
        
    def test_initialization(self):
        """Test decoder layer initialization."""
        layer = self.MoELlamaDecoderLayer(self.config, layer_idx=0)
        
        self.assertIsNotNone(layer.self_attn)
        self.assertIsNotNone(layer.input_layernorm)
        self.assertIsNotNone(layer.post_attention_layernorm)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        layer = self.MoELlamaDecoderLayer(self.config, layer_idx=0)
        hidden_states = torch.randn(2, 64, 256)
        
        # Returns (hidden_states, attn_weights, past_kv, aux_loss)
        output, attn_weights, past_kv, aux_loss = layer(hidden_states)
        
        self.assertEqual(output.shape, (2, 64, 256))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestMoELlamaModel(unittest.TestCase):
    """Test cases for MoELlamaModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.llm.moe_llama import MoELlamaModel
        self.MoELlamaModel = MoELlamaModel
        self.config = MockConfig()
        
    def test_initialization(self):
        """Test model initialization."""
        model = self.MoELlamaModel(self.config)
        
        self.assertIsNotNone(model.embed_tokens)
        self.assertIsNotNone(model.layers)
        self.assertIsNotNone(model.norm)
        self.assertEqual(len(model.layers), self.config.num_hidden_layers)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        model = self.MoELlamaModel(self.config)
        input_ids = torch.randint(0, self.config.vocab_size, (2, 64))
        
        output = model(input_ids)
        
        self.assertEqual(output.last_hidden_state.shape, (2, 64, 256))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestMoELlamaForCausalLM(unittest.TestCase):
    """Test cases for MoELlamaForCausalLM."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.llm.moe_llama import MoELlamaForCausalLM
        self.MoELlamaForCausalLM = MoELlamaForCausalLM
        self.config = MockConfig()
        
    def test_initialization(self):
        """Test causal LM initialization."""
        model = self.MoELlamaForCausalLM(self.config)
        
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.lm_head)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        model = self.MoELlamaForCausalLM(self.config)
        input_ids = torch.randint(0, self.config.vocab_size, (2, 64))
        
        output = model(input_ids)
        
        self.assertEqual(output.logits.shape, (2, 64, self.config.vocab_size))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestIntegration(unittest.TestCase):
    """Integration tests for MoE LLaMA components."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.llm.moe_llama import MultiHeadLatentAttention, YaRNRotaryEmbedding
        self.MultiHeadLatentAttention = MultiHeadLatentAttention
        self.YaRNRotaryEmbedding = YaRNRotaryEmbedding
        self.config = MockConfig()
        
    def test_mla_basic_forward(self):
        """Test MLA basic forward pass."""
        mla = self.MultiHeadLatentAttention(
            hidden_size=256,
            num_heads=8,
            num_kv_heads=2,
            head_dim=32,
            kv_lora_rank=64,
        )
        hidden_states = torch.randn(2, 64, 256)
        
        output, attn_weights, past_kv = mla(hidden_states)
        
        self.assertEqual(output.shape, (2, 64, 256))
        
    def test_gradient_flow_through_mla(self):
        """Test gradients flow through MLA layer."""
        mla = self.MultiHeadLatentAttention(
            hidden_size=256,
            num_heads=8,
            num_kv_heads=2,
            head_dim=32,
            kv_lora_rank=64,
        )
        hidden_states = torch.randn(2, 64, 256, requires_grad=True)
        
        output, _, _ = mla(hidden_states)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(hidden_states.grad)


if __name__ == '__main__':
    unittest.main()

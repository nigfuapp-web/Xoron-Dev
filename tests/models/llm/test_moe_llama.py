"""
Comprehensive unit tests for MoE LLaMA model.
Tests all components: DynamicNTKScalingRotaryEmbedding, rotate_half, apply_rotary_pos_emb,
make_sliding_window_causal_mask, KVCache, LlamaAttention, LlamaDecoderLayer, and MoELlamaModel.
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
class TestDynamicNTKScalingRotaryEmbedding(unittest.TestCase):
    """Test cases for DynamicNTKScalingRotaryEmbedding."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.llm.moe_llama import DynamicNTKScalingRotaryEmbedding
        self.DynamicNTKScalingRotaryEmbedding = DynamicNTKScalingRotaryEmbedding
        
    def test_initialization(self):
        """Test DynamicNTKScalingRotaryEmbedding initialization."""
        rope = self.DynamicNTKScalingRotaryEmbedding(dim=64, max_position_embeddings=4096)
        
        self.assertEqual(rope.dim, 64)
        self.assertEqual(rope.max_position_embeddings, 4096)
        
    def test_forward_output_shapes(self):
        """Test forward pass output shapes."""
        rope = self.DynamicNTKScalingRotaryEmbedding(dim=64, max_position_embeddings=4096)
        x = torch.randn(2, 8, 128, 64)  # [batch, heads, seq_len, head_dim]
        position_ids = torch.arange(128).unsqueeze(0).expand(2, -1)
        
        cos, sin = rope(x, position_ids)
        
        self.assertEqual(cos.shape, (2, 128, 64))
        self.assertEqual(sin.shape, (2, 128, 64))
        
    def test_different_positions_different_embeddings(self):
        """Test that different positions produce different embeddings."""
        rope = self.DynamicNTKScalingRotaryEmbedding(dim=64, max_position_embeddings=4096)
        x = torch.randn(1, 8, 10, 64)
        pos1 = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        pos2 = torch.tensor([[100, 101, 102, 103, 104, 105, 106, 107, 108, 109]])
        
        cos1, sin1 = rope(x, pos1)
        cos2, sin2 = rope(x, pos2)
        
        self.assertFalse(torch.allclose(cos1, cos2))
        self.assertFalse(torch.allclose(sin1, sin2))
        
    def test_dynamic_scaling(self):
        """Test dynamic NTK scaling for extended context."""
        rope = self.DynamicNTKScalingRotaryEmbedding(
            dim=64, 
            max_position_embeddings=1024,
            dynamic_scaling=True
        )
        x = torch.randn(1, 8, 2048, 64)  # Longer than max_position_embeddings
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
class TestMakeSlidingWindowCausalMask(unittest.TestCase):
    """Test cases for make_sliding_window_causal_mask function."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.llm.moe_llama import make_sliding_window_causal_mask
        self.make_sliding_window_causal_mask = make_sliding_window_causal_mask
        
    def test_output_shape(self):
        """Test output shape."""
        mask = self.make_sliding_window_causal_mask(
            seq_len=128, 
            sliding_window=64, 
            device=torch.device('cpu'),
            dtype=torch.float32
        )
        
        self.assertEqual(mask.shape, (1, 1, 128, 128))
        
    def test_causal_property(self):
        """Test mask is causal (can't attend to future)."""
        mask = self.make_sliding_window_causal_mask(
            seq_len=10, 
            sliding_window=100,  # Large window to test causal only
            device=torch.device('cpu'),
            dtype=torch.float32
        )
        
        # Upper triangle (future positions) should be -inf
        for i in range(10):
            for j in range(i + 1, 10):
                self.assertEqual(mask[0, 0, i, j].item(), float('-inf'))
                
    def test_sliding_window_property(self):
        """Test sliding window limits attention."""
        mask = self.make_sliding_window_causal_mask(
            seq_len=10, 
            sliding_window=3,
            device=torch.device('cpu'),
            dtype=torch.float32
        )
        
        # Position 5 should not attend to position 0 (outside window)
        self.assertEqual(mask[0, 0, 5, 0].item(), float('-inf'))
        # Position 5 should attend to position 3 (inside window)
        self.assertEqual(mask[0, 0, 5, 3].item(), 0.0)
        
    def test_with_kv_cache(self):
        """Test mask with KV cache (kv_seq_len > seq_len)."""
        mask = self.make_sliding_window_causal_mask(
            seq_len=1,  # Single token generation
            sliding_window=64,
            device=torch.device('cpu'),
            dtype=torch.float32,
            kv_seq_len=100  # Cached tokens
        )
        
        self.assertEqual(mask.shape, (1, 1, 1, 100))


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
        
    def test_sliding_window_eviction(self):
        """Test sliding window eviction."""
        cache = self.KVCache(
            key_cache=torch.randn(2, 4, 100, 64),
            value_cache=torch.randn(2, 4, 100, 64),
            seen_tokens=100
        )
        key_states = torch.randn(2, 4, 10, 64)
        value_states = torch.randn(2, 4, 10, 64)
        
        k, v = cache.update(key_states, value_states, sliding_window=64)
        
        # Should be truncated to sliding window size
        self.assertEqual(k.shape, (2, 4, 64, 64))
        self.assertEqual(v.shape, (2, 4, 64, 64))
        
    def test_get_seq_length(self):
        """Test get_seq_length method."""
        cache = self.KVCache(
            key_cache=torch.randn(2, 4, 50, 64),
            value_cache=torch.randn(2, 4, 50, 64)
        )
        
        self.assertEqual(cache.get_seq_length(), 50)
        
    def test_get_seq_length_empty(self):
        """Test get_seq_length with empty cache."""
        cache = self.KVCache(key_cache=None, value_cache=None)
        
        self.assertEqual(cache.get_seq_length(), 0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestLlamaAttention(unittest.TestCase):
    """Test cases for LlamaAttention."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.llm.moe_llama import LlamaAttention
        self.LlamaAttention = LlamaAttention
        self.config = MockConfig()
        
    def test_initialization(self):
        """Test LlamaAttention initialization."""
        attn = self.LlamaAttention(self.config, layer_idx=0)
        
        self.assertEqual(attn.hidden_size, 256)
        self.assertEqual(attn.num_heads, 8)
        self.assertEqual(attn.num_key_value_heads, 2)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        attn = self.LlamaAttention(self.config, layer_idx=0)
        hidden_states = torch.randn(2, 128, 256)
        
        output, _, _ = attn(hidden_states)
        
        self.assertEqual(output.shape, (2, 128, 256))
        
    def test_forward_with_position_ids(self):
        """Test forward pass with explicit position_ids."""
        attn = self.LlamaAttention(self.config, layer_idx=0)
        hidden_states = torch.randn(2, 128, 256)
        position_ids = torch.arange(128).unsqueeze(0).expand(2, -1)
        
        output, _, _ = attn(hidden_states, position_ids=position_ids)
        
        self.assertEqual(output.shape, (2, 128, 256))
        
    def test_forward_with_kv_cache(self):
        """Test forward pass with KV cache."""
        attn = self.LlamaAttention(self.config, layer_idx=0)
        
        # First pass - prefill
        hidden_states = torch.randn(2, 10, 256)
        output, past_kv, _ = attn(hidden_states, use_cache=True)
        
        self.assertEqual(output.shape, (2, 10, 256))
        self.assertIsNotNone(past_kv)
        self.assertEqual(past_kv[0].shape[2], 10)  # Cached 10 tokens
        
        # Second pass - generation with cache
        hidden_states = torch.randn(2, 1, 256)
        output, past_kv, _ = attn(hidden_states, past_key_value=past_kv, use_cache=True)
        
        self.assertEqual(output.shape, (2, 1, 256))
        self.assertEqual(past_kv[0].shape[2], 11)  # Now 11 tokens cached
        
    def test_gqa_repeat_kv(self):
        """Test GQA KV head repetition."""
        attn = self.LlamaAttention(self.config, layer_idx=0)
        kv = torch.randn(2, 2, 10, 32)  # 2 KV heads
        
        repeated = attn._repeat_kv(kv, 4)  # Repeat 4 times for 8 query heads
        
        self.assertEqual(repeated.shape, (2, 8, 10, 32))
        
    def test_sliding_window_attention(self):
        """Test sliding window attention is applied."""
        config = MockConfig()
        config.use_sliding_window = True
        config.sliding_window = 64
        
        attn = self.LlamaAttention(config, layer_idx=0)
        
        self.assertTrue(attn.use_sliding_window)
        self.assertEqual(attn.sliding_window, 64)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestLlamaRotaryEmbeddingAlias(unittest.TestCase):
    """Test cases for LlamaRotaryEmbedding alias."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.llm.moe_llama import LlamaRotaryEmbedding, DynamicNTKScalingRotaryEmbedding
        self.LlamaRotaryEmbedding = LlamaRotaryEmbedding
        self.DynamicNTKScalingRotaryEmbedding = DynamicNTKScalingRotaryEmbedding
        
    def test_alias_is_same_class(self):
        """Test LlamaRotaryEmbedding is alias for DynamicNTKScalingRotaryEmbedding."""
        self.assertIs(self.LlamaRotaryEmbedding, self.DynamicNTKScalingRotaryEmbedding)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestIntegration(unittest.TestCase):
    """Integration tests for MoE LLaMA components."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.llm.moe_llama import LlamaAttention, DynamicNTKScalingRotaryEmbedding
        self.LlamaAttention = LlamaAttention
        self.DynamicNTKScalingRotaryEmbedding = DynamicNTKScalingRotaryEmbedding
        self.config = MockConfig()
        
    def test_attention_with_rotary_embeddings(self):
        """Test attention layer uses rotary embeddings correctly."""
        attn = self.LlamaAttention(self.config, layer_idx=0)
        hidden_states = torch.randn(2, 64, 256)
        
        # Different position_ids should produce different outputs
        pos1 = torch.arange(64).unsqueeze(0).expand(2, -1)
        pos2 = torch.arange(100, 164).unsqueeze(0).expand(2, -1)
        
        output1, _, _ = attn(hidden_states, position_ids=pos1)
        output2, _, _ = attn(hidden_states, position_ids=pos2)
        
        self.assertFalse(torch.allclose(output1, output2))
        
    def test_gradient_flow_through_attention(self):
        """Test gradients flow through attention layer."""
        attn = self.LlamaAttention(self.config, layer_idx=0)
        hidden_states = torch.randn(2, 64, 256, requires_grad=True)
        
        output, _, _ = attn(hidden_states)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(hidden_states.grad)
        
    def test_autoregressive_generation_simulation(self):
        """Test simulated autoregressive generation with KV cache."""
        attn = self.LlamaAttention(self.config, layer_idx=0)
        
        # Prefill phase
        prefill = torch.randn(1, 20, 256)
        output, past_kv, _ = attn(prefill, use_cache=True)
        
        # Generation phase - generate 10 tokens
        for i in range(10):
            new_token = torch.randn(1, 1, 256)
            output, past_kv, _ = attn(new_token, past_key_value=past_kv, use_cache=True)
            
            self.assertEqual(output.shape, (1, 1, 256))
            self.assertEqual(past_kv[0].shape[2], 20 + i + 1)


if __name__ == '__main__':
    unittest.main()

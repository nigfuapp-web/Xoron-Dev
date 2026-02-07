"""Unit tests for models/components/attention.py - Attention mechanisms."""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestFlashAttentionAvailable(unittest.TestCase):
    """Test cases for flash_attention_available function."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.components.attention import flash_attention_available
        self.flash_attention_available = flash_attention_available
        
    def test_returns_bool(self):
        """Test function returns boolean."""
        result = self.flash_attention_available()
        self.assertIsInstance(result, bool)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestAttentionKVCache(unittest.TestCase):
    """Test cases for AttentionKVCache."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.components.attention import AttentionKVCache
        self.AttentionKVCache = AttentionKVCache
        
    def test_initialization(self):
        """Test AttentionKVCache initialization."""
        cache = self.AttentionKVCache()
        
        self.assertIsNone(cache.key_cache)
        self.assertIsNone(cache.value_cache)
        self.assertEqual(cache.seen_tokens, 0)
        
    def test_update_first_call(self):
        """Test update on first call."""
        cache = self.AttentionKVCache()
        key = torch.randn(2, 4, 10, 64)
        value = torch.randn(2, 4, 10, 64)
        
        k, v = cache.update(key, value)
        
        self.assertEqual(k.shape, (2, 4, 10, 64))
        self.assertEqual(v.shape, (2, 4, 10, 64))
        self.assertEqual(cache.seen_tokens, 10)
        
    def test_update_concatenates(self):
        """Test update concatenates with existing cache."""
        cache = self.AttentionKVCache()
        key1 = torch.randn(2, 4, 10, 64)
        value1 = torch.randn(2, 4, 10, 64)
        key2 = torch.randn(2, 4, 5, 64)
        value2 = torch.randn(2, 4, 5, 64)
        
        cache.update(key1, value1)
        k, v = cache.update(key2, value2)
        
        self.assertEqual(k.shape, (2, 4, 15, 64))
        self.assertEqual(v.shape, (2, 4, 15, 64))
        self.assertEqual(cache.seen_tokens, 15)
        
    def test_get_seq_length(self):
        """Test get_seq_length method."""
        cache = self.AttentionKVCache()
        
        self.assertEqual(cache.get_seq_length(), 0)
        
        key = torch.randn(2, 4, 10, 64)
        value = torch.randn(2, 4, 10, 64)
        cache.update(key, value)
        
        self.assertEqual(cache.get_seq_length(), 10)
        
    def test_reset(self):
        """Test reset method."""
        cache = self.AttentionKVCache()
        key = torch.randn(2, 4, 10, 64)
        value = torch.randn(2, 4, 10, 64)
        cache.update(key, value)
        
        cache.reset()
        
        self.assertIsNone(cache.key_cache)
        self.assertIsNone(cache.value_cache)
        self.assertEqual(cache.seen_tokens, 0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestFlashAttention(unittest.TestCase):
    """Test cases for FlashAttention module."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.components.attention import FlashAttention
        self.FlashAttention = FlashAttention
        
    def test_initialization(self):
        """Test FlashAttention initialization."""
        attn = self.FlashAttention(dropout=0.1, causal=True)
        
        self.assertEqual(attn.dropout, 0.1)
        self.assertTrue(attn.causal)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        attn = self.FlashAttention()
        query = torch.randn(2, 4, 10, 64)
        key = torch.randn(2, 4, 10, 64)
        value = torch.randn(2, 4, 10, 64)
        
        output, _, _ = attn(query, key, value)
        
        self.assertEqual(output.shape, (2, 4, 10, 64))
        
    def test_causal_attention(self):
        """Test causal attention masking."""
        attn = self.FlashAttention(causal=True)
        query = torch.randn(2, 4, 10, 64)
        key = torch.randn(2, 4, 10, 64)
        value = torch.randn(2, 4, 10, 64)
        
        output, _, _ = attn(query, key, value)
        
        self.assertEqual(output.shape, (2, 4, 10, 64))
        
    def test_with_kv_cache(self):
        """Test attention with KV cache."""
        attn = self.FlashAttention()
        query = torch.randn(2, 4, 1, 64)
        key = torch.randn(2, 4, 1, 64)
        value = torch.randn(2, 4, 1, 64)
        past_key = torch.randn(2, 4, 10, 64)
        past_value = torch.randn(2, 4, 10, 64)
        
        output, present_kv, _ = attn(
            query, key, value,
            past_key_value=(past_key, past_value),
            use_cache=True
        )
        
        self.assertEqual(output.shape, (2, 4, 1, 64))
        self.assertIsNotNone(present_kv)
        self.assertEqual(present_kv[0].shape, (2, 4, 11, 64))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestMultimodalCrossAttention(unittest.TestCase):
    """Test cases for MultimodalCrossAttention."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.components.attention import MultimodalCrossAttention
        self.MultimodalCrossAttention = MultimodalCrossAttention
        
    def test_initialization(self):
        """Test MultimodalCrossAttention initialization."""
        attn = self.MultimodalCrossAttention(
            hidden_size=256,
            num_heads=8,
            dropout=0.1,
        )
        
        self.assertEqual(attn.hidden_size, 256)
        self.assertEqual(attn.num_heads, 8)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        attn = self.MultimodalCrossAttention(256, num_heads=8)
        text_hidden = torch.randn(2, 10, 256)
        modality_hidden = torch.randn(2, 20, 256)
        
        output, _, _ = attn(text_hidden, modality_hidden)
        
        self.assertEqual(output.shape, (2, 10, 256))
        
    def test_gated_residual(self):
        """Test gated residual connection."""
        attn = self.MultimodalCrossAttention(256, num_heads=8, gate_init=0.0)
        
        # Gate should be learnable
        self.assertTrue(attn.gate.requires_grad)
        
    def test_with_kv_cache(self):
        """Test cross-attention with KV cache."""
        attn = self.MultimodalCrossAttention(256, num_heads=8)
        text_hidden = torch.randn(2, 1, 256)
        modality_hidden = torch.randn(2, 20, 256)
        
        # First pass - compute KV
        output1, kv_cache, _ = attn(text_hidden, modality_hidden, use_cache=True)
        
        # Second pass - use cached KV
        output2, _, _ = attn(text_hidden, modality_hidden, past_key_value=kv_cache)
        
        self.assertEqual(output1.shape, output2.shape)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestMultimodalFusionLayer(unittest.TestCase):
    """Test cases for MultimodalFusionLayer."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.components.attention import MultimodalFusionLayer
        self.MultimodalFusionLayer = MultimodalFusionLayer
        
    def test_initialization(self):
        """Test MultimodalFusionLayer initialization."""
        layer = self.MultimodalFusionLayer(
            hidden_size=256,
            num_heads=8,
            dropout=0.1,
        )
        
        self.assertEqual(layer.hidden_size, 256)
        self.assertIsNotNone(layer.image_cross_attn)
        self.assertIsNotNone(layer.video_cross_attn)
        self.assertIsNotNone(layer.audio_cross_attn)
        
    def test_forward_text_only(self):
        """Test forward with text only."""
        layer = self.MultimodalFusionLayer(256, num_heads=8)
        text_hidden = torch.randn(2, 10, 256)
        
        output, cache = layer(text_hidden)
        
        self.assertEqual(output.shape, (2, 10, 256))
        
    def test_forward_with_image(self):
        """Test forward with image features."""
        layer = self.MultimodalFusionLayer(256, num_heads=8)
        text_hidden = torch.randn(2, 10, 256)
        image_hidden = torch.randn(2, 64, 256)
        
        output, cache = layer(text_hidden, image_hidden=image_hidden)
        
        self.assertEqual(output.shape, (2, 10, 256))
        
    def test_forward_with_video(self):
        """Test forward with video features."""
        layer = self.MultimodalFusionLayer(256, num_heads=8)
        text_hidden = torch.randn(2, 10, 256)
        video_hidden = torch.randn(2, 32, 256)
        
        output, cache = layer(text_hidden, video_hidden=video_hidden)
        
        self.assertEqual(output.shape, (2, 10, 256))
        
    def test_forward_with_audio(self):
        """Test forward with audio features."""
        layer = self.MultimodalFusionLayer(256, num_heads=8)
        text_hidden = torch.randn(2, 10, 256)
        audio_hidden = torch.randn(2, 100, 256)
        
        output, cache = layer(text_hidden, audio_hidden=audio_hidden)
        
        self.assertEqual(output.shape, (2, 10, 256))
        
    def test_forward_all_modalities(self):
        """Test forward with all modalities."""
        layer = self.MultimodalFusionLayer(256, num_heads=8)
        text_hidden = torch.randn(2, 10, 256)
        image_hidden = torch.randn(2, 64, 256)
        video_hidden = torch.randn(2, 32, 256)
        audio_hidden = torch.randn(2, 100, 256)
        
        output, cache = layer(
            text_hidden,
            image_hidden=image_hidden,
            video_hidden=video_hidden,
            audio_hidden=audio_hidden,
        )
        
        self.assertEqual(output.shape, (2, 10, 256))
        
    def test_with_cache(self):
        """Test forward with KV cache."""
        layer = self.MultimodalFusionLayer(256, num_heads=8)
        text_hidden = torch.randn(2, 10, 256)
        image_hidden = torch.randn(2, 64, 256)
        
        output, cache = layer(text_hidden, image_hidden=image_hidden, use_cache=True)
        
        self.assertIsNotNone(cache)


if __name__ == '__main__':
    unittest.main()

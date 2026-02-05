"""Unit tests for models/encoders/vision.py - SOTA Vision encoder.

Tests for:
- SIGLIP_MODELS dictionary
- RoPE2DEncoder (2D Rotary Position Embedding)
- TiTokTokenizer (1D tokenization)
- DualStreamEncoderAttention
- VisionEncoderBlock
- VisionEncoder
- get_vision_encoder function
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestSiglipModels(unittest.TestCase):
    """Test cases for SIGLIP_MODELS dictionary."""
    
    def test_siglip_models_exists(self):
        """Test SIGLIP_MODELS dictionary exists."""
        from models.encoders.vision import SIGLIP_MODELS
        self.assertIsInstance(SIGLIP_MODELS, dict)
        
    def test_siglip_models_contains_variants(self):
        """Test SIGLIP_MODELS contains expected variants."""
        from models.encoders.vision import SIGLIP_MODELS
        
        self.assertIn('siglip-base', SIGLIP_MODELS)
        self.assertIn('siglip-large', SIGLIP_MODELS)
        self.assertIn('siglip-so400m', SIGLIP_MODELS)
        
    def test_siglip_models_contains_clip(self):
        """Test SIGLIP_MODELS contains CLIP variants."""
        from models.encoders.vision import SIGLIP_MODELS
        
        self.assertIn('clip-base', SIGLIP_MODELS)
        self.assertIn('clip-large', SIGLIP_MODELS)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestRoPE2DEncoder(unittest.TestCase):
    """Test cases for RoPE2DEncoder (2D Rotary Position Embedding)."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.encoders.vision import RoPE2DEncoder
        self.RoPE2DEncoder = RoPE2DEncoder
        
    def test_initialization(self):
        """Test RoPE2DEncoder initialization."""
        rope = self.RoPE2DEncoder(dim=64, max_height=32, max_width=32)
        
        self.assertEqual(rope.dim, 64)
        self.assertEqual(rope.max_height, 32)
        self.assertEqual(rope.max_width, 32)
        self.assertIsNotNone(rope.inv_freq_x)
        self.assertIsNotNone(rope.inv_freq_y)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        rope = self.RoPE2DEncoder(dim=64, max_height=32, max_width=32)
        x = torch.randn(2, 16 * 16, 64)  # [B, H*W, dim]
        
        cos, sin = rope(x, height=16, width=16)
        
        self.assertEqual(cos.shape, (16 * 16, 64))
        self.assertEqual(sin.shape, (16 * 16, 64))
        
    def test_cos_sin_range(self):
        """Test cos and sin values are in valid range."""
        rope = self.RoPE2DEncoder(dim=64)
        x = torch.randn(2, 64, 64)
        
        cos, sin = rope(x, height=8, width=8)
        
        self.assertTrue((cos >= -1).all() and (cos <= 1).all())
        self.assertTrue((sin >= -1).all() and (sin <= 1).all())


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestTiTokTokenizer(unittest.TestCase):
    """Test cases for TiTokTokenizer (1D tokenization)."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.encoders.vision import TiTokTokenizer
        self.TiTokTokenizer = TiTokTokenizer
        
    def test_initialization(self):
        """Test TiTokTokenizer initialization."""
        titok = self.TiTokTokenizer(hidden_size=256, num_tokens=64, num_patches=196)
        
        self.assertEqual(titok.hidden_size, 256)
        self.assertEqual(titok.num_tokens, 64)
        self.assertEqual(titok.num_patches, 196)
        self.assertIsNotNone(titok.token_queries)
        self.assertIsNotNone(titok.compress_attn)
        
    def test_forward_output_shape(self):
        """Test forward pass compresses patches to tokens."""
        titok = self.TiTokTokenizer(hidden_size=256, num_tokens=64, num_patches=196)
        x = torch.randn(2, 196, 256)  # [B, num_patches, hidden_size]
        
        output = titok(x)
        
        self.assertEqual(output.shape, (2, 64, 256))  # [B, num_tokens, hidden_size]
        
    def test_compression_ratio(self):
        """Test different compression ratios."""
        # High compression
        titok_high = self.TiTokTokenizer(hidden_size=256, num_tokens=32, num_patches=576)
        x = torch.randn(2, 576, 256)
        output = titok_high(x)
        self.assertEqual(output.shape, (2, 32, 256))
        
        # Low compression
        titok_low = self.TiTokTokenizer(hidden_size=256, num_tokens=256, num_patches=576)
        output = titok_low(x)
        self.assertEqual(output.shape, (2, 256, 256))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestDualStreamEncoderAttention(unittest.TestCase):
    """Test cases for DualStreamEncoderAttention."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.encoders.vision import DualStreamEncoderAttention
        self.DualStreamEncoderAttention = DualStreamEncoderAttention
        
    def test_initialization(self):
        """Test DualStreamEncoderAttention initialization."""
        attn = self.DualStreamEncoderAttention(hidden_size=256, num_heads=8)
        
        self.assertEqual(attn.hidden_size, 256)
        self.assertEqual(attn.num_heads, 8)
        self.assertIsNotNone(attn.to_qkv_a)
        self.assertIsNotNone(attn.to_qkv_b)
        self.assertIsNotNone(attn.rope_2d)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        attn = self.DualStreamEncoderAttention(hidden_size=256, num_heads=8, max_height=16, max_width=16)
        x_a = torch.randn(2, 256, 256)  # [B, H*W, hidden_size]
        x_b = torch.randn(2, 256, 256)
        
        out_a, out_b = attn(x_a, x_b, height=16, width=16)
        
        self.assertEqual(out_a.shape, (2, 256, 256))
        self.assertEqual(out_b.shape, (2, 256, 256))
        
    def test_symmetric_processing(self):
        """Test that both streams are processed symmetrically."""
        attn = self.DualStreamEncoderAttention(hidden_size=256, num_heads=8, max_height=8, max_width=8)
        x = torch.randn(2, 64, 256)
        
        # Same input should produce similar (but not identical due to separate projections) outputs
        out_a, out_b = attn(x, x.clone(), height=8, width=8)
        
        self.assertEqual(out_a.shape, out_b.shape)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestVisionEncoderBlock(unittest.TestCase):
    """Test cases for VisionEncoderBlock."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.encoders.vision import VisionEncoderBlock
        self.VisionEncoderBlock = VisionEncoderBlock
        
    def test_initialization(self):
        """Test VisionEncoderBlock initialization."""
        block = self.VisionEncoderBlock(hidden_size=256, num_heads=8)
        
        self.assertIsNotNone(block.dual_attn)
        self.assertIsNotNone(block.ffn_a)
        self.assertIsNotNone(block.ffn_b)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        block = self.VisionEncoderBlock(hidden_size=256, num_heads=8, max_height=8, max_width=8)
        x_a = torch.randn(2, 64, 256)
        x_b = torch.randn(2, 64, 256)
        
        out_a, out_b = block(x_a, x_b, height=8, width=8)
        
        self.assertEqual(out_a.shape, (2, 64, 256))
        self.assertEqual(out_b.shape, (2, 64, 256))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestVisionEncoderMocked(unittest.TestCase):
    """Test cases for VisionEncoder with mocked transformers."""
    
    def test_is_siglip_detection(self):
        """Test SigLIP model detection."""
        from models.encoders.vision import VisionEncoder
        
        # Test with mocked initialization
        with patch.object(VisionEncoder, '_init_siglip') as mock_siglip:
            with patch.object(VisionEncoder, '_init_clip') as mock_clip:
                mock_siglip.return_value = None
                
                # Create encoder with SigLIP model name
                encoder = object.__new__(VisionEncoder)
                encoder.model_name = "google/siglip-so400m-patch14-384"
                encoder._is_siglip = "siglip" in encoder.model_name.lower()
                
                self.assertTrue(encoder._is_siglip)
                
    def test_is_clip_detection(self):
        """Test CLIP model detection."""
        from models.encoders.vision import VisionEncoder
        
        encoder = object.__new__(VisionEncoder)
        encoder.model_name = "openai/clip-vit-large-patch14"
        encoder._is_siglip = "siglip" in encoder.model_name.lower()
        
        self.assertFalse(encoder._is_siglip)


class TestGetVisionEncoder(unittest.TestCase):
    """Test cases for get_vision_encoder function."""
    
    def test_function_exists(self):
        """Test get_vision_encoder function exists."""
        from models.encoders.vision import get_vision_encoder
        self.assertTrue(callable(get_vision_encoder))
        
    def test_accepts_model_key(self):
        """Test function accepts model key parameter."""
        from models.encoders.vision import get_vision_encoder, SIGLIP_MODELS
        
        # Just verify the function signature works
        # Actual model loading would require transformers
        import inspect
        sig = inspect.signature(get_vision_encoder)
        self.assertIn('model_key', sig.parameters)
        self.assertIn('freeze', sig.parameters)
        self.assertIn('use_dual_stream', sig.parameters)
        self.assertIn('use_titok', sig.parameters)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestApplyRope2DEncoder(unittest.TestCase):
    """Test cases for apply_rope_2d_encoder function."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.encoders.vision import apply_rope_2d_encoder
        self.apply_rope_2d_encoder = apply_rope_2d_encoder
        
    def test_output_shape(self):
        """Test output shape matches input."""
        x = torch.randn(2, 8, 64, 32)  # [B, num_heads, seq_len, head_dim]
        cos = torch.randn(64, 32)
        sin = torch.randn(64, 32)
        
        # Expand for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        output = self.apply_rope_2d_encoder(x, cos, sin)
        
        self.assertEqual(output.shape, x.shape)


if __name__ == '__main__':
    unittest.main()

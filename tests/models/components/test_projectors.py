"""Unit tests for models/components/projectors.py - Multimodal projectors."""

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
class TestPerceiverAttention(unittest.TestCase):
    """Test cases for PerceiverAttention."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.components.projectors import PerceiverAttention
        self.PerceiverAttention = PerceiverAttention
        
    def test_initialization(self):
        """Test PerceiverAttention initialization."""
        attn = self.PerceiverAttention(dim=256, num_heads=8)
        
        self.assertEqual(attn.num_heads, 8)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        attn = self.PerceiverAttention(dim=256, num_heads=8, dim_head=32)
        latents = torch.randn(2, 64, 256)
        context = torch.randn(2, 196, 256)
        
        output = attn(latents, context)
        
        self.assertEqual(output.shape, (2, 64, 256))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestPerceiverResampler(unittest.TestCase):
    """Test cases for PerceiverResampler."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.components.projectors import PerceiverResampler
        self.PerceiverResampler = PerceiverResampler
        
    def test_initialization(self):
        """Test PerceiverResampler initialization."""
        resampler = self.PerceiverResampler(
            input_dim=768,
            output_dim=256,
            num_latents=64,
        )
        
        self.assertEqual(resampler.num_latents, 64)
        self.assertEqual(resampler.latents.shape, (1, 64, 256))
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        resampler = self.PerceiverResampler(768, 256, num_latents=64)
        x = torch.randn(2, 196, 768)
        
        output = resampler(x)
        
        self.assertEqual(output.shape, (2, 64, 256))
        
    def test_variable_input_length(self):
        """Test with variable input sequence length."""
        resampler = self.PerceiverResampler(768, 256, num_latents=64)
        
        x1 = torch.randn(2, 100, 768)
        x2 = torch.randn(2, 500, 768)
        
        output1 = resampler(x1)
        output2 = resampler(x2)
        
        # Output should always be num_latents
        self.assertEqual(output1.shape, (2, 64, 256))
        self.assertEqual(output2.shape, (2, 64, 256))
        
    def test_same_input_output_dim(self):
        """Test when input and output dimensions are the same."""
        resampler = self.PerceiverResampler(256, 256, num_latents=32)
        x = torch.randn(2, 100, 256)
        
        output = resampler(x)
        
        self.assertEqual(output.shape, (2, 32, 256))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestSpatialAwareProjector(unittest.TestCase):
    """Test cases for SpatialAwareProjector."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.components.projectors import SpatialAwareProjector
        self.SpatialAwareProjector = SpatialAwareProjector
        
    def test_initialization(self):
        """Test SpatialAwareProjector initialization."""
        projector = self.SpatialAwareProjector(
            vision_hidden_size=768,
            llm_hidden_size=256,
            num_tokens=64,
        )
        
        self.assertEqual(projector.num_tokens, 64)
        
    def test_forward_from_sequence(self):
        """Test forward pass from sequence input."""
        projector = self.SpatialAwareProjector(768, 256, num_tokens=64, spatial_pool_size=8)
        x = torch.randn(2, 196, 768)  # 14x14 patches
        
        output = projector(x)
        
        self.assertEqual(output.shape, (2, 64, 256))
        
    def test_forward_from_spatial(self):
        """Test forward pass from spatial input."""
        projector = self.SpatialAwareProjector(768, 256, num_tokens=64, spatial_pool_size=8)
        x = torch.randn(2, 14, 14, 768)
        
        output = projector(x)
        
        self.assertEqual(output.shape, (2, 64, 256))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestCAbstractor(unittest.TestCase):
    """Test cases for CAbstractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.components.projectors import CAbstractor
        self.CAbstractor = CAbstractor
        
    def test_initialization(self):
        """Test CAbstractor initialization."""
        abstractor = self.CAbstractor(
            vision_hidden_size=768,
            llm_hidden_size=256,
            num_tokens=64,
        )
        
        self.assertEqual(abstractor.num_tokens, 64)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        abstractor = self.CAbstractor(768, 256, num_tokens=64)
        x = torch.randn(2, 196, 768)
        
        output = abstractor(x)
        
        self.assertEqual(output.shape, (2, 64, 256))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestMultimodalProjector(unittest.TestCase):
    """Test cases for MultimodalProjector."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.components.projectors import MultimodalProjector
        self.MultimodalProjector = MultimodalProjector
        
    def test_perceiver_type(self):
        """Test perceiver projector type."""
        projector = self.MultimodalProjector(
            vision_hidden_size=768,
            llm_hidden_size=256,
            num_tokens=64,
            projector_type="perceiver",
        )
        
        self.assertEqual(projector.projector_type, "perceiver")
        
        x = torch.randn(2, 196, 768)
        output = projector(x)
        
        self.assertEqual(output.shape, (2, 64, 256))
        
    def test_spatial_type(self):
        """Test spatial projector type."""
        projector = self.MultimodalProjector(
            vision_hidden_size=768,
            llm_hidden_size=256,
            num_tokens=64,
            projector_type="spatial",
        )
        
        self.assertEqual(projector.projector_type, "spatial")
        
        x = torch.randn(2, 196, 768)
        output = projector(x)
        
        self.assertEqual(output.shape, (2, 64, 256))
        
    def test_c_abstractor_type(self):
        """Test c_abstractor projector type."""
        projector = self.MultimodalProjector(
            vision_hidden_size=768,
            llm_hidden_size=256,
            num_tokens=64,
            projector_type="c_abstractor",
        )
        
        self.assertEqual(projector.projector_type, "c_abstractor")
        
        x = torch.randn(2, 196, 768)
        output = projector(x)
        
        self.assertEqual(output.shape, (2, 64, 256))
        
    def test_mlp_type(self):
        """Test MLP projector type."""
        projector = self.MultimodalProjector(
            vision_hidden_size=768,
            llm_hidden_size=256,
            num_tokens=64,
            projector_type="mlp",
        )
        
        self.assertEqual(projector.projector_type, "mlp")
        
        x = torch.randn(2, 196, 768)
        output = projector(x)
        
        self.assertEqual(output.shape, (2, 64, 256))
        
    def test_gradient_flow(self):
        """Test gradients flow through projector."""
        projector = self.MultimodalProjector(768, 256, num_tokens=64, projector_type="perceiver")
        x = torch.randn(2, 196, 768, requires_grad=True)
        
        output = projector(x)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)


if __name__ == '__main__':
    unittest.main()

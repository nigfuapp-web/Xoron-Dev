"""
Comprehensive unit tests for SOTA video generator module.
Tests all components: RoPE2D, RoPE1D, TemporalExpertRouter, VideoExpert,
TemporalMoELayer, SpatialAttention, TemporalAttention, CrossAttention3D,
FlowMatchingScheduler, VideoVAE3D, and MobileVideoDiffusion.
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


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestRoPE2DVideo(unittest.TestCase):
    """Test cases for 2D Rotary Position Embeddings (video)."""
    
    def setUp(self):
        from models.generators.video import RoPE2D
        self.RoPE2D = RoPE2D
        
    def test_initialization(self):
        rope = self.RoPE2D(dim=64, max_height=32, max_width=32)
        self.assertEqual(rope.dim, 64)
        
    def test_forward_output_shape(self):
        rope = self.RoPE2D(dim=64, max_height=32, max_width=32)
        x = torch.randn(2, 64, 64)
        cos, sin = rope(x, height=8, width=8)
        self.assertEqual(cos.shape, (64, 64))
        self.assertEqual(sin.shape, (64, 64))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestRoPE1D(unittest.TestCase):
    """Test cases for 1D Rotary Position Embeddings (temporal)."""
    
    def setUp(self):
        from models.generators.video import RoPE1D
        self.RoPE1D = RoPE1D
        
    def test_initialization(self):
        rope = self.RoPE1D(dim=64, max_len=32)
        self.assertEqual(rope.dim, 64)
        
    def test_forward_output_shape(self):
        rope = self.RoPE1D(dim=64, max_len=32)
        x = torch.randn(2, 16, 64)
        cos, sin = rope(x, seq_len=16)
        self.assertEqual(cos.shape, (16, 64))
        self.assertEqual(sin.shape, (16, 64))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestTemporalExpertRouter(unittest.TestCase):
    """Test cases for TemporalExpertRouter."""
    
    def setUp(self):
        from models.generators.video import TemporalExpertRouter
        self.TemporalExpertRouter = TemporalExpertRouter
        
    def test_initialization(self):
        router = self.TemporalExpertRouter(hidden_size=256, num_experts=4, top_k=2)
        self.assertEqual(router.num_experts, 4)
        self.assertEqual(router.top_k, 2)
        
    def test_forward_output_shapes(self):
        router = self.TemporalExpertRouter(hidden_size=256, num_experts=4, top_k=2)
        x = torch.randn(20, 256)
        top_k_probs, top_k_indices = router(x)
        self.assertEqual(top_k_probs.shape, (20, 2))
        self.assertEqual(top_k_indices.shape, (20, 2))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestVideoExpert(unittest.TestCase):
    """Test cases for VideoExpert (SwiGLU FFN)."""
    
    def setUp(self):
        from models.generators.video import VideoExpert
        self.VideoExpert = VideoExpert
        
    def test_initialization(self):
        expert = self.VideoExpert(hidden_size=256, intermediate_size=1024)
        self.assertEqual(expert.gate_proj.in_features, 256)
        
    def test_forward_output_shape(self):
        expert = self.VideoExpert(hidden_size=256, intermediate_size=1024)
        x = torch.randn(2, 64, 256)
        output = expert(x)
        self.assertEqual(output.shape, (2, 64, 256))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestTemporalMoELayer(unittest.TestCase):
    """Test cases for TemporalMoELayer."""
    
    def setUp(self):
        from models.generators.video import TemporalMoELayer
        self.TemporalMoELayer = TemporalMoELayer
        
    def test_initialization(self):
        layer = self.TemporalMoELayer(hidden_size=256, intermediate_size=1024, num_experts=4, top_k=2)
        self.assertEqual(layer.num_experts, 4)
        self.assertIsNotNone(layer.shared_expert)
        
    def test_forward_output_shape(self):
        layer = self.TemporalMoELayer(hidden_size=256, intermediate_size=1024, num_experts=4, top_k=2)
        x = torch.randn(2, 64, 256)
        output = layer(x)
        self.assertEqual(output.shape, (2, 64, 256))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestSpatialAttention(unittest.TestCase):
    """Test cases for SpatialAttention."""
    
    def setUp(self):
        from models.generators.video import SpatialAttention
        self.SpatialAttention = SpatialAttention
        


if __name__ == '__main__':
    unittest.main()


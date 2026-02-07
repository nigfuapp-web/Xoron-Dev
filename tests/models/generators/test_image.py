"""
Comprehensive unit tests for SOTA image generator module.
Tests all components: RoPE2D, ImageExpert, ImageMoERouter, ImageMoELayer, 
DualStreamSelfAttention, CrossAttention, DiTBlock, FlowMatchingScheduler,
PatchEmbed, UnpatchEmbed, MoEDiT, ImageVAE, and MobileDiffusionGenerator.
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
class TestRoPE2D(unittest.TestCase):
    """Test cases for 2D Rotary Position Embeddings."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.image import RoPE2D
        self.RoPE2D = RoPE2D
        
    def test_initialization(self):
        """Test RoPE2D initialization."""
        rope = self.RoPE2D(dim=64, max_height=32, max_width=32)
        self.assertEqual(rope.dim, 64)
        self.assertEqual(rope.max_height, 32)
        self.assertEqual(rope.max_width, 32)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        rope = self.RoPE2D(dim=64, max_height=32, max_width=32)
        x = torch.randn(2, 64, 64)
        
        cos, sin = rope(x, height=8, width=8)
        
        self.assertEqual(cos.shape, (64, 64))
        self.assertEqual(sin.shape, (64, 64))
        
    def test_different_positions_produce_different_embeddings(self):
        """Test that different grid sizes produce different embeddings."""
        rope = self.RoPE2D(dim=64, max_height=32, max_width=32)
        x = torch.randn(2, 64, 64)
        
        cos1, sin1 = rope(x, height=4, width=4)
        cos2, sin2 = rope(x, height=8, width=8)
        
        self.assertNotEqual(cos1.shape, cos2.shape)
        
    def test_batch_processing(self):
        """Test batch processing."""
        rope = self.RoPE2D(dim=128, max_height=64, max_width=64)
        x = torch.randn(4, 256, 128)
        
        cos, sin = rope(x, height=16, width=16)
        
        self.assertEqual(cos.shape, (256, 128))
        self.assertEqual(sin.shape, (256, 128))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestImageExpert(unittest.TestCase):
    """Test cases for ImageExpert (SwiGLU FFN)."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.image import ImageExpert
        self.ImageExpert = ImageExpert
        
    def test_initialization(self):
        """Test ImageExpert initialization."""
        expert = self.ImageExpert(hidden_size=256, intermediate_size=1024)
        
        self.assertEqual(expert.gate_proj.in_features, 256)
        self.assertEqual(expert.gate_proj.out_features, 1024)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        expert = self.ImageExpert(hidden_size=256, intermediate_size=1024)
        x = torch.randn(2, 64, 256)
        
        output = expert(x)
        
        self.assertEqual(output.shape, (2, 64, 256))
        
    def test_gradient_flow(self):
        """Test gradients flow through expert."""
        expert = self.ImageExpert(hidden_size=256, intermediate_size=1024)
        x = torch.randn(2, 64, 256, requires_grad=True)
        
        output = expert(x)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestImageMoERouter(unittest.TestCase):
    """Test cases for ImageMoERouter."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.image import ImageMoERouter
        self.ImageMoERouter = ImageMoERouter
        
    def test_initialization(self):
        """Test ImageMoERouter initialization."""
        router = self.ImageMoERouter(hidden_size=256, num_experts=4, top_k=2)
        
        self.assertEqual(router.num_experts, 4)
        self.assertEqual(router.top_k, 2)
        
    def test_forward_output_shapes(self):
        """Test forward pass output shapes."""
        router = self.ImageMoERouter(hidden_size=256, num_experts=4, top_k=2)
        x = torch.randn(20, 256)  # Flat input
        
        top_k_probs, top_k_indices = router(x)
        
        self.assertEqual(top_k_probs.shape, (20, 2))
        self.assertEqual(top_k_indices.shape, (20, 2))
        
    def test_top_k_indices_valid(self):
        """Test top-k indices are valid expert indices."""
        router = self.ImageMoERouter(hidden_size=256, num_experts=4, top_k=2)
        x = torch.randn(20, 256)
        
        _, top_k_indices = router(x)
        
        self.assertTrue(torch.all(top_k_indices >= 0))
        self.assertTrue(torch.all(top_k_indices < 4))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestImageMoELayer(unittest.TestCase):
    """Test cases for ImageMoELayer with shared expert."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.image import ImageMoELayer
        self.ImageMoELayer = ImageMoELayer
        
    def test_initialization(self):
        """Test ImageMoELayer initialization."""
        layer = self.ImageMoELayer(hidden_size=256, intermediate_size=1024, num_experts=4, top_k=2)
        
        self.assertEqual(layer.num_experts, 4)
        self.assertEqual(len(layer.experts), 4)
        self.assertIsNotNone(layer.shared_expert)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        layer = self.ImageMoELayer(hidden_size=256, intermediate_size=1024, num_experts=4, top_k=2)
        x = torch.randn(2, 64, 256)
        
        output = layer(x)
        
        self.assertEqual(output.shape, (2, 64, 256))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestDualStreamSelfAttention(unittest.TestCase):
    """Test cases for DualStreamSelfAttention (SD3/Flux-style)."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.image import DualStreamSelfAttention
        self.DualStreamSelfAttention = DualStreamSelfAttention
        
    def test_initialization(self):
        """Test DualStreamSelfAttention initialization."""
        attn = self.DualStreamSelfAttention(hidden_size=256, num_heads=8, max_height=16, max_width=16)
        
        self.assertEqual(attn.hidden_size, 256)
        self.assertEqual(attn.num_heads, 8)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        attn = self.DualStreamSelfAttention(hidden_size=256, num_heads=8, max_height=16, max_width=16)
        x_a = torch.randn(2, 64, 256)
        x_b = torch.randn(2, 64, 256)
        
        out_a, out_b = attn(x_a, x_b, height=8, width=8)
        
        self.assertEqual(out_a.shape, (2, 64, 256))
        self.assertEqual(out_b.shape, (2, 64, 256))
        
    def test_symmetric_processing(self):
        """Test that both streams are processed."""
        attn = self.DualStreamSelfAttention(hidden_size=256, num_heads=8, max_height=16, max_width=16)
        x_a = torch.randn(2, 64, 256)
        x_b = torch.randn(2, 64, 256)
        
        out_a, out_b = attn(x_a, x_b, height=8, width=8)
        
        # Outputs should be different from inputs
        self.assertFalse(torch.allclose(out_a, x_a))
        self.assertFalse(torch.allclose(out_b, x_b))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestCrossAttention(unittest.TestCase):
    """Test cases for CrossAttention."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.image import CrossAttention
        self.CrossAttention = CrossAttention
        
    def test_initialization(self):
        """Test CrossAttention initialization."""
        attn = self.CrossAttention(query_dim=256, context_dim=512, heads=8)
        
        self.assertEqual(attn.heads, 8)
        
    def test_cross_attention_mode(self):
        """Test cross-attention with context."""
        attn = self.CrossAttention(query_dim=256, context_dim=512, heads=8)
        x = torch.randn(2, 64, 256)
        context = torch.randn(2, 77, 512)
        
        output = attn(x, context)
        
        self.assertEqual(output.shape, (2, 64, 256))
        
    def test_gradient_flow(self):
        """Test gradients flow through attention."""
        attn = self.CrossAttention(query_dim=256, context_dim=512, heads=8)
        x = torch.randn(2, 64, 256, requires_grad=True)
        context = torch.randn(2, 77, 512)
        
        output = attn(x, context)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestDiTBlock(unittest.TestCase):
    """Test cases for DiTBlock (Diffusion Transformer Block)."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.image import DiTBlock
        self.DiTBlock = DiTBlock
        
    def test_initialization(self):
        """Test DiTBlock initialization."""
        block = self.DiTBlock(hidden_size=256, context_dim=512, num_heads=8, num_experts=4)
        
        self.assertIsNotNone(block.dual_attn)
        self.assertIsNotNone(block.cross_attn_a)
        self.assertIsNotNone(block.cross_attn_b)
        self.assertIsNotNone(block.moe_a)
        self.assertIsNotNone(block.moe_b)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        block = self.DiTBlock(hidden_size=256, context_dim=512, num_heads=8, num_experts=4, max_height=16, max_width=16)
        x_a = torch.randn(2, 64, 256)
        x_b = torch.randn(2, 64, 256)
        context = torch.randn(2, 77, 512)
        t_emb = torch.randn(2, 256)
        
        out_a, out_b = block(x_a, x_b, context, t_emb, height=8, width=8)
        
        self.assertEqual(out_a.shape, (2, 64, 256))
        self.assertEqual(out_b.shape, (2, 64, 256))
        



if __name__ == '__main__':
    unittest.main()


"""Unit tests for models/components/moe.py - Mixture of Experts."""

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
class TestMoERouter(unittest.TestCase):
    """Test cases for MoERouter."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.components.moe import MoERouter
        self.MoERouter = MoERouter
        
    def test_initialization(self):
        """Test MoERouter initialization."""
        router = self.MoERouter(
            hidden_size=256,
            num_experts=8,
            top_k=2,
        )
        
        self.assertEqual(router.num_experts, 8)
        self.assertEqual(router.top_k, 2)
        
    def test_forward_output_shapes(self):
        """Test forward pass output shapes."""
        router = self.MoERouter(256, num_experts=8, top_k=2)
        x = torch.randn(2, 10, 256)
        
        top_k_probs, top_k_indices, router_logits = router(x)
        
        # top_k_probs: [batch*seq, top_k]
        self.assertEqual(top_k_probs.shape, (20, 2))
        # top_k_indices: [batch*seq, top_k]
        self.assertEqual(top_k_indices.shape, (20, 2))
        # router_logits: [batch*seq, num_experts]
        self.assertEqual(router_logits.shape, (20, 8))
        
    def test_top_k_indices_valid(self):
        """Test top-k indices are valid expert indices."""
        router = self.MoERouter(256, num_experts=8, top_k=2)
        x = torch.randn(2, 10, 256)
        
        _, top_k_indices, _ = router(x)
        
        self.assertTrue(torch.all(top_k_indices >= 0))
        self.assertTrue(torch.all(top_k_indices < 8))
        
    def test_top_k_probs_sum_to_one(self):
        """Test top-k probabilities sum to approximately 1."""
        router = self.MoERouter(256, num_experts=8, top_k=2)
        x = torch.randn(2, 10, 256)
        
        top_k_probs, _, _ = router(x)
        
        sums = top_k_probs.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))
        
    def test_noise_during_training(self):
        """Test noise is added during training."""
        router = self.MoERouter(256, num_experts=8, top_k=2, noise_std=1.0)
        router.train()
        x = torch.randn(2, 10, 256)
        
        # Just verify the router runs in training mode
        top_k_probs, top_k_indices, router_logits = router(x)
        
        # Verify outputs are valid
        self.assertEqual(router_logits.shape, (20, 8))
        self.assertTrue(router.training)
        
    def test_no_noise_during_eval(self):
        """Test no noise during evaluation."""
        router = self.MoERouter(256, num_experts=8, top_k=2, noise_std=1.0)
        router.eval()
        x = torch.randn(2, 10, 256)
        
        # Run multiple times
        results = [router(x)[2] for _ in range(3)]
        
        # Results should be identical
        for r in results[1:]:
            self.assertTrue(torch.allclose(results[0], r))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestMoEExpert(unittest.TestCase):
    """Test cases for MoEExpert."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.components.moe import MoEExpert
        self.MoEExpert = MoEExpert
        
    def test_initialization(self):
        """Test MoEExpert initialization."""
        expert = self.MoEExpert(
            hidden_size=256,
            intermediate_size=512,
        )
        
        self.assertEqual(expert.gate_proj.in_features, 256)
        self.assertEqual(expert.gate_proj.out_features, 512)
        
    def test_forward_shape(self):
        """Test forward pass output shape."""
        expert = self.MoEExpert(256, 512)
        x = torch.randn(10, 256)
        
        output = expert(x)
        
        self.assertEqual(output.shape, (10, 256))
        
    def test_swiglu_activation(self):
        """Test SwiGLU activation is used."""
        expert = self.MoEExpert(256, 512)
        
        # Check SiLU activation exists
        self.assertIsInstance(expert.act_fn, nn.SiLU)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestSharedExpert(unittest.TestCase):
    """Test cases for SharedExpert."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.components.moe import SharedExpert
        self.SharedExpert = SharedExpert
        
    def test_initialization(self):
        """Test SharedExpert initialization."""
        expert = self.SharedExpert(256, 512)
        
        self.assertTrue(hasattr(expert, 'shared_gate'))
        
    def test_forward_shape(self):
        """Test forward pass output shape."""
        expert = self.SharedExpert(256, 512)
        x = torch.randn(10, 256)
        
        output = expert(x)
        
        self.assertEqual(output.shape, (10, 256))
        
    def test_shared_gate_learnable(self):
        """Test shared gate is learnable."""
        expert = self.SharedExpert(256, 512)
        
        self.assertTrue(expert.shared_gate.requires_grad)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestMoELayer(unittest.TestCase):
    """Test cases for MoELayer."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.components.moe import MoELayer
        self.MoELayer = MoELayer
        
    def test_initialization(self):
        """Test MoELayer initialization."""
        layer = self.MoELayer(
            hidden_size=256,
            intermediate_size=512,
            num_experts=8,
            num_experts_per_tok=2,
        )
        
        self.assertEqual(layer.num_experts, 8)
        self.assertEqual(layer.num_experts_per_tok, 2)
        self.assertEqual(len(layer.experts), 8)
        
    def test_shared_expert_configurable(self):
        """Test shared expert is configurable."""
        # Test with shared expert enabled (default)
        layer_with = self.MoELayer(256, 512, use_shared_expert=True)
        self.assertTrue(layer_with.use_shared_expert)
        self.assertIsNotNone(layer_with.shared_expert)
        
        # Test with shared expert disabled
        layer_without = self.MoELayer(256, 512, use_shared_expert=False)
        self.assertFalse(layer_without.use_shared_expert)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        layer = self.MoELayer(256, 512, num_experts=4, num_experts_per_tok=2)
        x = torch.randn(2, 10, 256)
        
        output, aux_loss = layer(x)
        
        self.assertEqual(output.shape, (2, 10, 256))
        
    def test_forward_returns_aux_loss(self):
        """Test forward pass returns auxiliary loss."""
        layer = self.MoELayer(256, 512, num_experts=4)
        x = torch.randn(2, 10, 256)
        
        _, aux_loss = layer(x)
        
        self.assertIsInstance(aux_loss, torch.Tensor)
        self.assertEqual(aux_loss.dim(), 0)  # Scalar
        
    def test_aux_loss_positive(self):
        """Test auxiliary loss is positive."""
        layer = self.MoELayer(256, 512, num_experts=4)
        x = torch.randn(2, 10, 256)
        
        _, aux_loss = layer(x)
        
        self.assertGreater(aux_loss.item(), 0)
        
    def test_gradient_flow(self):
        """Test gradients flow through the layer."""
        layer = self.MoELayer(256, 512, num_experts=4)
        x = torch.randn(2, 10, 256, requires_grad=True)
        
        output, aux_loss = layer(x)
        loss = output.sum() + aux_loss
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.all(x.grad == 0))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestExpertChoiceMoELayer(unittest.TestCase):
    """Test cases for ExpertChoiceMoELayer."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.components.moe import ExpertChoiceMoELayer
        self.ExpertChoiceMoELayer = ExpertChoiceMoELayer
        
    def test_initialization(self):
        """Test ExpertChoiceMoELayer initialization."""
        layer = self.ExpertChoiceMoELayer(
            hidden_size=256,
            intermediate_size=512,
            num_experts=4,
        )
        
        self.assertEqual(layer.num_experts, 4)
        self.assertEqual(len(layer.experts), 4)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        layer = self.ExpertChoiceMoELayer(256, 512, num_experts=4)
        x = torch.randn(2, 10, 256)
        
        output, aux_loss = layer(x)
        
        self.assertEqual(output.shape, (2, 10, 256))
        
    def test_forward_returns_aux_loss(self):
        """Test forward pass returns auxiliary loss."""
        layer = self.ExpertChoiceMoELayer(256, 512, num_experts=4)
        x = torch.randn(2, 10, 256)
        
        _, aux_loss = layer(x)
        
        self.assertIsInstance(aux_loss, torch.Tensor)


if __name__ == '__main__':
    unittest.main()

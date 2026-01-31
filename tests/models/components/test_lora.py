"""Unit tests for models/components/lora.py - LoRA implementation."""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import torch if available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestLoRALinear(unittest.TestCase):
    """Test cases for LoRALinear layer."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.components.lora import LoRALinear
        self.LoRALinear = LoRALinear
        
    def test_initialization(self):
        """Test LoRALinear initialization."""
        layer = self.LoRALinear(
            in_features=64,
            out_features=128,
            r=8,
            lora_alpha=16,
        )
        
        self.assertEqual(layer.r, 8)
        self.assertEqual(layer.lora_alpha, 16)
        self.assertEqual(layer.lora_A.shape, (8, 64))
        self.assertEqual(layer.lora_B.shape, (128, 8))
        
    def test_lora_a_initialization(self):
        """Test LoRA A matrix is initialized with Kaiming."""
        layer = self.LoRALinear(64, 128, r=8)
        
        # A should not be all zeros (Kaiming init)
        self.assertFalse(torch.all(layer.lora_A == 0))
        
    def test_lora_b_initialization(self):
        """Test LoRA B matrix is initialized with zeros."""
        layer = self.LoRALinear(64, 128, r=8)
        
        # B should be all zeros
        self.assertTrue(torch.all(layer.lora_B == 0))
        
    def test_forward_shape(self):
        """Test forward pass output shape."""
        layer = self.LoRALinear(64, 128, r=8)
        x = torch.randn(2, 10, 64)
        
        output = layer(x)
        
        self.assertEqual(output.shape, (2, 10, 128))
        
    def test_original_weights_frozen(self):
        """Test original linear weights are frozen."""
        layer = self.LoRALinear(64, 128, r=8)
        
        self.assertFalse(layer.linear.weight.requires_grad)
        
    def test_lora_weights_trainable(self):
        """Test LoRA weights are trainable."""
        layer = self.LoRALinear(64, 128, r=8)
        
        self.assertTrue(layer.lora_A.requires_grad)
        self.assertTrue(layer.lora_B.requires_grad)
        
    def test_rslora_scaling(self):
        """Test rsLoRA scaling factor."""
        layer = self.LoRALinear(64, 128, r=16, lora_alpha=32, use_rslora=True)
        
        expected_scaling = 32 / math.sqrt(16)
        self.assertAlmostEqual(layer.scaling, expected_scaling, places=5)
        
    def test_standard_lora_scaling(self):
        """Test standard LoRA scaling factor."""
        layer = self.LoRALinear(64, 128, r=16, lora_alpha=32, use_rslora=False)
        
        expected_scaling = 32 / 16
        self.assertEqual(layer.scaling, expected_scaling)
        
    def test_dora_magnitude_parameter(self):
        """Test DoRA magnitude parameter exists when enabled."""
        layer = self.LoRALinear(64, 128, r=8, use_dora=True)
        
        self.assertTrue(hasattr(layer, 'magnitude'))
        self.assertEqual(layer.magnitude.shape, (128,))
        
    def test_merge_lora_weights(self):
        """Test merging LoRA weights."""
        layer = self.LoRALinear(64, 128, r=8)
        original_weight = layer.linear.weight.data.clone()
        
        # Set non-zero LoRA weights
        layer.lora_A.data = torch.randn_like(layer.lora_A)
        layer.lora_B.data = torch.randn_like(layer.lora_B)
        
        layer.merge_lora_weights()
        
        self.assertTrue(layer.merged)
        # Weight should have changed
        self.assertFalse(torch.allclose(layer.linear.weight.data, original_weight))
        
    def test_unmerge_lora_weights(self):
        """Test unmerging LoRA weights."""
        layer = self.LoRALinear(64, 128, r=8)
        
        layer.merge_lora_weights()
        layer.unmerge_lora_weights()
        
        self.assertFalse(layer.merged)
        
    def test_dropout_applied(self):
        """Test dropout is applied during training."""
        layer = self.LoRALinear(64, 128, r=8, lora_dropout=0.5)
        layer.train()
        
        self.assertIsInstance(layer.lora_dropout, nn.Dropout)
        
    def test_no_dropout_when_zero(self):
        """Test no dropout when dropout rate is zero."""
        layer = self.LoRALinear(64, 128, r=8, lora_dropout=0.0)
        
        self.assertIsInstance(layer.lora_dropout, nn.Identity)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestLoRAConfig(unittest.TestCase):
    """Test cases for LoRAConfig."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.components.lora import LoRAConfig
        self.LoRAConfig = LoRAConfig
        
    def test_default_initialization(self):
        """Test LoRAConfig default initialization."""
        config = self.LoRAConfig()
        
        self.assertEqual(config.r, 8)
        self.assertEqual(config.lora_alpha, 16)
        self.assertEqual(config.lora_dropout, 0.05)
        self.assertTrue(config.enable_lora)
        
    def test_custom_initialization(self):
        """Test LoRAConfig custom initialization."""
        config = self.LoRAConfig(
            r=32,
            lora_alpha=64,
            use_dora=True,
        )
        
        self.assertEqual(config.r, 32)
        self.assertEqual(config.lora_alpha, 64)
        self.assertTrue(config.use_dora)
        
    def test_default_target_modules(self):
        """Test default target modules."""
        config = self.LoRAConfig()
        
        self.assertIn('q_proj', config.target_modules)
        self.assertIn('v_proj', config.target_modules)
        self.assertIn('gate_proj', config.target_modules)
        
    def test_custom_target_modules(self):
        """Test custom target modules."""
        config = self.LoRAConfig(target_modules=['q_proj', 'k_proj'])
        
        self.assertEqual(config.target_modules, ['q_proj', 'k_proj'])
        
    def test_lora_plus_lr_ratio(self):
        """Test LoRA+ learning rate ratio."""
        config = self.LoRAConfig()
        
        self.assertEqual(config.lora_plus_lr_ratio, 16.0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestApplyLoraToModel(unittest.TestCase):
    """Test cases for apply_lora_to_model function."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.components.lora import apply_lora_to_model, LoRAConfig, LoRALinear
        self.apply_lora_to_model = apply_lora_to_model
        self.LoRAConfig = LoRAConfig
        self.LoRALinear = LoRALinear
        
    def test_apply_lora_to_simple_model(self):
        """Test applying LoRA to a simple model."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)
                self.other = nn.Linear(64, 64)
                
        model = SimpleModel()
        config = self.LoRAConfig(target_modules=['q_proj', 'v_proj'])
        
        model = self.apply_lora_to_model(model, config)
        
        self.assertIsInstance(model.q_proj, self.LoRALinear)
        self.assertIsInstance(model.v_proj, self.LoRALinear)
        self.assertIsInstance(model.other, nn.Linear)  # Not converted
        
    def test_disabled_lora_returns_unchanged(self):
        """Test disabled LoRA returns model unchanged."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                
        model = SimpleModel()
        config = self.LoRAConfig(enable_lora=False)
        
        result = self.apply_lora_to_model(model, config)
        
        self.assertIsInstance(result.q_proj, nn.Linear)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestGetLoraParameters(unittest.TestCase):
    """Test cases for get_lora_parameters function."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.components.lora import get_lora_parameters, LoRALinear
        self.get_lora_parameters = get_lora_parameters
        self.LoRALinear = LoRALinear
        
    def test_returns_only_lora_params(self):
        """Test function returns only LoRA parameters."""
        model = nn.Module()
        model.lora_layer = self.LoRALinear(64, 64, r=8)
        model.regular_layer = nn.Linear(64, 64)
        
        lora_params = self.get_lora_parameters(model)
        
        # Should only contain lora_A and lora_B parameters
        param_names = []
        for name, param in model.named_parameters():
            # Check by id instead of value comparison
            if any(param is p for p in lora_params):
                param_names.append(name)
                
        self.assertTrue(all('lora_' in name for name in param_names))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestGetLoraPlusParamGroups(unittest.TestCase):
    """Test cases for get_lora_plus_param_groups function."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.components.lora import get_lora_plus_param_groups, LoRALinear
        self.get_lora_plus_param_groups = get_lora_plus_param_groups
        self.LoRALinear = LoRALinear
        
    def test_returns_param_groups(self):
        """Test function returns parameter groups."""
        model = nn.Module()
        model.lora_layer = self.LoRALinear(64, 64, r=8)
        
        param_groups = self.get_lora_plus_param_groups(model, base_lr=1e-4)
        
        self.assertIsInstance(param_groups, list)
        
    def test_different_lr_for_a_and_b(self):
        """Test different learning rates for A and B matrices."""
        model = nn.Module()
        model.lora_layer = self.LoRALinear(64, 64, r=8)
        
        param_groups = self.get_lora_plus_param_groups(model, base_lr=1e-4, lr_ratio=16.0)
        
        # Find A and B groups
        a_group = next((g for g in param_groups if g.get('name') == 'lora_A'), None)
        b_group = next((g for g in param_groups if g.get('name') == 'lora_B'), None)
        
        if a_group and b_group:
            self.assertEqual(b_group['lr'], a_group['lr'] * 16.0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestCountLoraParameters(unittest.TestCase):
    """Test cases for count_lora_parameters function."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.components.lora import count_lora_parameters, LoRALinear
        self.count_lora_parameters = count_lora_parameters
        self.LoRALinear = LoRALinear
        
    def test_counts_lora_params(self):
        """Test counting LoRA parameters."""
        model = nn.Module()
        model.lora_layer = self.LoRALinear(64, 128, r=8)
        model.regular_layer = nn.Linear(64, 64)
        
        lora_params, total_params, percentage = self.count_lora_parameters(model)
        
        # LoRA params: A (8*64) + B (128*8) = 512 + 1024 = 1536
        expected_lora = 8 * 64 + 128 * 8
        self.assertEqual(lora_params, expected_lora)
        self.assertGreater(total_params, lora_params)
        self.assertGreater(percentage, 0)
        self.assertLess(percentage, 100)


if __name__ == '__main__':
    unittest.main()

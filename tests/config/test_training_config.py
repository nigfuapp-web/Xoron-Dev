"""Unit tests for config/training_config.py - TrainingConfig."""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestTrainingConfig(unittest.TestCase):
    """Test cases for TrainingConfig dataclass."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Import here to allow mocking
        from config.training_config import TrainingConfig
        self.TrainingConfig = TrainingConfig
        
    def test_default_initialization(self):
        """Test TrainingConfig initializes with SOTA default values."""
        config = self.TrainingConfig()
        
        self.assertEqual(config.batch_size, 1)
        self.assertEqual(config.gradient_accumulation_steps, 16)  # Reduced for FP16 stability
        self.assertEqual(config.learning_rate, 1e-4)  # Increased with FP32 optimizer states
        self.assertEqual(config.num_epochs, 1)
        
    def test_custom_initialization(self):
        """Test TrainingConfig with custom values."""
        config = self.TrainingConfig(
            batch_size=4,
            learning_rate=1e-4,
            num_epochs=5,
        )
        
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.num_epochs, 5)
        
    def test_effective_batch_size(self):
        """Test effective batch size calculation."""
        config = self.TrainingConfig(
            batch_size=2,
            gradient_accumulation_steps=8,
        )
        
        self.assertEqual(config.effective_batch_size, 16)
        
    def test_loss_weights(self):
        """Test loss weight configuration."""
        config = self.TrainingConfig()
        
        self.assertEqual(config.llm_loss_weight, 1.0)
        self.assertEqual(config.image_diffusion_loss_weight, 0.1)
        self.assertEqual(config.video_diffusion_loss_weight, 0.1)
        self.assertEqual(config.asr_loss_weight, 0.1)
        self.assertEqual(config.tts_loss_weight, 0.1)
        self.assertEqual(config.moe_aux_loss_weight, 0.02)
        
    def test_cot_loss_weight(self):
        """Test chain-of-thought loss weight."""
        config = self.TrainingConfig()
        
        self.assertEqual(config.cot_loss_weight, 1.5)
        
    def test_lora_plus_settings(self):
        """Test LoRA+ configuration (reduced ratio for FP16 stability)."""
        config = self.TrainingConfig()
        
        self.assertTrue(config.use_lora_plus)
        self.assertEqual(config.lora_plus_lr_ratio, 4.0)  # Reduced from 16.0 for stability
        
    def test_checkpointing_settings(self):
        """Test checkpointing configuration (eval at end of epoch, not step-based)."""
        config = self.TrainingConfig()
        
        self.assertEqual(config.save_steps, 500)
        self.assertEqual(config.logging_steps, 50)
        self.assertEqual(config.max_per_dataset_eval, 10)  # Eval samples per dataset
        
    def test_memory_optimization_settings(self):
        """Test memory optimization configuration (less frequent cache clearing)."""
        config = self.TrainingConfig()
        
        self.assertEqual(config.empty_cache_freq, 100)  # Increased from 5 for performance
        self.assertTrue(config.gradient_checkpointing)
        self.assertTrue(config.use_8bit_optimizer)
        self.assertTrue(config.set_to_none)
        
    def test_precision_settings(self):
        """Test precision configuration."""
        config = self.TrainingConfig()
        
        self.assertTrue(config.fp16)
        self.assertFalse(config.bf16)
        
    def test_to_dict(self):
        """Test config serialization to dictionary."""
        config = self.TrainingConfig(batch_size=4, num_epochs=3)
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['batch_size'], 4)
        self.assertEqual(config_dict['num_epochs'], 3)
        
    def test_from_dict(self):
        """Test config creation from dictionary."""
        config_dict = {
            'batch_size': 8,
            'learning_rate': 5e-5,
            'num_epochs': 10,
        }
        
        config = self.TrainingConfig.from_dict(config_dict)
        
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.learning_rate, 5e-5)
        self.assertEqual(config.num_epochs, 10)
        
    def test_cfg_dropout_rate(self):
        """Test classifier-free guidance dropout rate."""
        config = self.TrainingConfig()
        
        self.assertEqual(config.cfg_dropout_rate, 0.1)
        
    def test_temporal_consistency_weight(self):
        """Test temporal consistency weight for video."""
        config = self.TrainingConfig()
        
        self.assertEqual(config.temporal_consistency_weight, 0.01)
        
    def test_max_grad_norm(self):
        """Test gradient clipping configuration."""
        config = self.TrainingConfig()
        
        self.assertEqual(config.max_grad_norm, 1.0)
        
    def test_warmup_ratio(self):
        """Test warmup ratio configuration (increased for FP16 stability)."""
        config = self.TrainingConfig()
        
        self.assertEqual(config.warmup_ratio, 0.05)  # Increased from 0.03 for stability


class TestGetDeviceMap(unittest.TestCase):
    """Test cases for get_device_map function."""
    
    def setUp(self):
        """Set up test fixtures."""
        from config.training_config import get_device_map
        self.get_device_map = get_device_map
        
    def test_single_gpu_device_map(self):
        """Test device map for single GPU."""
        device_map = self.get_device_map(1)
        
        self.assertEqual(device_map['vision_encoder'], 'cuda:0')
        self.assertEqual(device_map['llm'], 'cuda:0')
        self.assertEqual(device_map['generator'], 'cuda:0')
        self.assertEqual(device_map['primary'], 'cuda:0')
        
    def test_dual_gpu_device_map(self):
        """Test device map for dual GPU setup."""
        device_map = self.get_device_map(2)
        
        self.assertEqual(device_map['vision_encoder'], 'cuda:0')
        self.assertEqual(device_map['llm'], 'cuda:1')
        self.assertEqual(device_map['generator'], 'cuda:1')
        
    def test_multi_gpu_device_map(self):
        """Test device map for 3+ GPU setup."""
        device_map = self.get_device_map(3)
        
        self.assertEqual(device_map['vision_encoder'], 'cuda:0')
        self.assertEqual(device_map['llm'], 'cuda:1')
        self.assertEqual(device_map['generator'], 'cuda:2')
        
    def test_device_map_contains_all_components(self):
        """Test that device map contains all required components."""
        device_map = self.get_device_map(1)
        
        required_components = [
            'vision_encoder', 'video_encoder', 'audio_encoder', 'audio_decoder',
            'projector', 'audio_projector', 'llm', 'cross_attention',
            'generator', 'video_generator', 'modality_markers', 'primary'
        ]
        
        for component in required_components:
            self.assertIn(component, device_map)


if __name__ == '__main__':
    unittest.main()

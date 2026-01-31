"""Unit tests for config/model_config.py - XoronConfig."""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.model_config import XoronConfig


class TestXoronConfig(unittest.TestCase):
    """Test cases for XoronConfig dataclass."""
    
    def test_default_initialization(self):
        """Test XoronConfig initializes with default values."""
        config = XoronConfig()
        
        self.assertEqual(config.model_name, "Xoron-Dev-MultiMoE")
        self.assertEqual(config.hidden_size, 1024)
        self.assertEqual(config.num_layers, 12)
        self.assertEqual(config.num_heads, 16)
        self.assertEqual(config.vocab_size, 151643)
        self.assertEqual(config.max_position_embeddings, 131072)
        
    def test_custom_initialization(self):
        """Test XoronConfig with custom values."""
        config = XoronConfig(
            hidden_size=512,
            num_layers=6,
            num_heads=8,
            num_experts=4,
        )
        
        self.assertEqual(config.hidden_size, 512)
        self.assertEqual(config.num_layers, 6)
        self.assertEqual(config.num_heads, 8)
        self.assertEqual(config.num_experts, 4)
        
    def test_moe_configuration(self):
        """Test MoE-related configuration."""
        config = XoronConfig()
        
        self.assertTrue(config.use_moe)
        self.assertEqual(config.num_experts, 8)
        self.assertEqual(config.num_experts_per_tok, 2)
        self.assertEqual(config.moe_layer_freq, 2)
        self.assertTrue(config.use_shared_expert)
        
    def test_lora_configuration(self):
        """Test LoRA-related configuration."""
        config = XoronConfig()
        
        self.assertTrue(config.use_lora)
        self.assertEqual(config.lora_r, 32)
        self.assertEqual(config.lora_alpha, 64)
        self.assertEqual(config.lora_dropout, 0.05)
        self.assertTrue(config.use_rslora)
        self.assertFalse(config.use_dora)
        
    def test_vision_configuration(self):
        """Test vision-related configuration."""
        config = XoronConfig()
        
        self.assertEqual(config.vision_model_name, "google/siglip-so400m-patch14-384")
        self.assertFalse(config.freeze_vision)
        self.assertEqual(config.num_vision_tokens, 64)
        self.assertEqual(config.vision_image_size, 384)
        
    def test_generation_configuration(self):
        """Test generation-related configuration."""
        config = XoronConfig()
        
        self.assertTrue(config.enable_generation)
        self.assertEqual(config.generation_image_size, 256)
        self.assertEqual(config.generation_cfg_scale, 7.5)
        self.assertEqual(config.generation_num_frames, 16)
        
    def test_audio_configuration(self):
        """Test audio-related configuration."""
        config = XoronConfig()
        
        self.assertEqual(config.audio_sample_rate, 16000)
        self.assertEqual(config.audio_n_mels, 80)
        self.assertEqual(config.audio_num_emotions, 13)
        
    def test_post_init_validation_hidden_size(self):
        """Test __post_init__ validates hidden_size divisibility."""
        with self.assertRaises(AssertionError):
            XoronConfig(hidden_size=100, num_heads=16)
            
    def test_post_init_validation_experts(self):
        """Test __post_init__ validates expert configuration."""
        with self.assertRaises(AssertionError):
            XoronConfig(num_experts=4, num_experts_per_tok=8)
            
    def test_to_dict(self):
        """Test config serialization to dictionary."""
        config = XoronConfig(hidden_size=512, num_layers=6)
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['hidden_size'], 512)
        self.assertEqual(config_dict['num_layers'], 6)
        self.assertIn('model_name', config_dict)
        self.assertIn('use_moe', config_dict)
        
    def test_from_dict(self):
        """Test config creation from dictionary."""
        config_dict = {
            'hidden_size': 768,
            'num_layers': 8,
            'num_heads': 12,
            'lora_target_modules': ['q_proj', 'v_proj'],
        }
        
        config = XoronConfig.from_dict(config_dict)
        
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.num_layers, 8)
        self.assertEqual(config.num_heads, 12)
        self.assertEqual(config.lora_target_modules, ('q_proj', 'v_proj'))
        
    def test_to_dict_from_dict_roundtrip(self):
        """Test roundtrip serialization."""
        original = XoronConfig(
            hidden_size=512,
            num_layers=6,
            num_heads=8,
            use_moe=False,
        )
        
        config_dict = original.to_dict()
        restored = XoronConfig.from_dict(config_dict)
        
        self.assertEqual(original.hidden_size, restored.hidden_size)
        self.assertEqual(original.num_layers, restored.num_layers)
        self.assertEqual(original.use_moe, restored.use_moe)
        
    def test_sliding_window_configuration(self):
        """Test sliding window attention configuration."""
        config = XoronConfig()
        
        self.assertTrue(config.use_sliding_window)
        self.assertEqual(config.sliding_window, 4096)
        
    def test_cross_attention_configuration(self):
        """Test cross-attention configuration."""
        config = XoronConfig()
        
        self.assertTrue(config.use_cross_attention)
        self.assertEqual(config.cross_attention_layers, 4)
        self.assertEqual(config.cross_attention_heads, 8)
        
    def test_flash_attention_configuration(self):
        """Test flash attention configuration."""
        config = XoronConfig()
        
        self.assertTrue(config.use_flash_attention)


class TestXoronConfigEdgeCases(unittest.TestCase):
    """Edge case tests for XoronConfig."""
    
    def test_minimum_valid_config(self):
        """Test minimum valid configuration."""
        config = XoronConfig(
            hidden_size=64,
            num_layers=1,
            num_heads=1,
            num_experts=1,
            num_experts_per_tok=1,
        )
        self.assertEqual(config.hidden_size, 64)
        
    def test_lora_target_modules_tuple_conversion(self):
        """Test that lora_target_modules is properly converted to tuple."""
        config_dict = {
            'lora_target_modules': ['q_proj', 'k_proj', 'v_proj'],
        }
        config = XoronConfig.from_dict(config_dict)
        self.assertIsInstance(config.lora_target_modules, tuple)
        
    def test_from_dict_ignores_unknown_keys(self):
        """Test that from_dict ignores unknown keys."""
        config_dict = {
            'hidden_size': 512,
            'unknown_key': 'should_be_ignored',
            'another_unknown': 123,
        }
        config = XoronConfig.from_dict(config_dict)
        self.assertEqual(config.hidden_size, 512)
        self.assertFalse(hasattr(config, 'unknown_key'))


if __name__ == '__main__':
    unittest.main()

"""Base test utilities and fixtures for Xoron-Dev test suite."""

import sys
import os
import unittest
from typing import Dict, Any, Optional
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BaseTestCase(unittest.TestCase):
    """Base test case with common utilities."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures."""
        cls.mock_torch_available = True
        
    def setUp(self):
        """Set up test fixtures."""
        pass
    
    def tearDown(self):
        """Clean up after tests."""
        pass


class MockTensor:
    """Mock tensor for testing without torch dependency."""
    
    def __init__(self, shape, dtype=None, device='cpu'):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self._data = [[0] * shape[-1] for _ in range(shape[0] if len(shape) > 1 else 1)]
    
    def size(self, dim=None):
        if dim is not None:
            return self.shape[dim]
        return self.shape
    
    def dim(self):
        return len(self.shape)
    
    def to(self, device):
        self.device = device
        return self
    
    def view(self, *args):
        return MockTensor(args)
    
    def unsqueeze(self, dim):
        new_shape = list(self.shape)
        new_shape.insert(dim, 1)
        return MockTensor(tuple(new_shape))
    
    def squeeze(self, dim=None):
        new_shape = [s for s in self.shape if s != 1]
        return MockTensor(tuple(new_shape) if new_shape else (1,))
    
    def mean(self, dim=None, keepdim=False):
        if dim is None:
            return MockTensor((1,))
        new_shape = list(self.shape)
        if keepdim:
            new_shape[dim] = 1
        else:
            new_shape.pop(dim)
        return MockTensor(tuple(new_shape) if new_shape else (1,))
    
    def sum(self, dim=None, keepdim=False):
        return self.mean(dim, keepdim)
    
    def numel(self):
        result = 1
        for s in self.shape:
            result *= s
        return result
    
    def __repr__(self):
        return f"MockTensor(shape={self.shape}, device={self.device})"


def create_mock_config(**overrides):
    """Create a mock XoronConfig for testing."""
    defaults = {
        'model_name': 'Xoron-Dev-Test',
        'hidden_size': 256,
        'num_layers': 2,
        'num_heads': 4,
        'intermediate_size': 512,
        'vocab_size': 1000,
        'max_position_embeddings': 512,
        'rms_norm_eps': 1e-6,
        'use_sliding_window': False,
        'sliding_window': 256,
        'use_moe': True,
        'num_experts': 4,
        'num_experts_per_tok': 2,
        'moe_layer_freq': 2,
        'router_aux_loss_coef': 0.1,
        'use_shared_expert': True,
        'moe_capacity_factor': 1.25,
        'vision_model_name': 'test-vision',
        'freeze_vision': True,
        'num_vision_tokens': 16,
        'max_video_frames': 8,
        'projector_type': 'mlp',
        'vision_image_size': 224,
        'enable_generation': False,
        'generation_image_size': 64,
        'generation_latent_channels': 4,
        'generation_base_channels': 32,
        'generation_inference_steps': 5,
        'generation_cfg_scale': 7.5,
        'generation_video_size': 64,
        'generation_num_frames': 4,
        'generation_video_cfg_scale': 7.5,
        'audio_sample_rate': 16000,
        'audio_n_mels': 80,
        'audio_num_emotions': 13,
        'audio_num_speakers': 256,
        'tokenizer_name': 'test-tokenizer',
        'use_lora': False,
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.05,
        'lora_target_modules': ('q_proj', 'v_proj'),
        'train_lora_only': False,
        'use_rslora': True,
        'use_dora': False,
        'lora_plus_lr_ratio': 16.0,
        'use_cross_attention': False,
        'cross_attention_layers': 2,
        'cross_attention_heads': 4,
        'cross_attention_dropout': 0.1,
        'use_flash_attention': False,
        'output_dir': './test-output',
    }
    defaults.update(overrides)
    
    config = MagicMock()
    for key, value in defaults.items():
        setattr(config, key, value)
    
    return config


def create_mock_training_config(**overrides):
    """Create a mock TrainingConfig for testing."""
    defaults = {
        'environment': 'local',
        'model_path': './test-model',
        'temp_dir': './test-tmp',
        'datasets_dir': './test-datasets',
        'output_dir': './test-output',
        'final_model_dir': './test-final',
        'max_per_epoch': 1000,
        'max_per_dataset': 100,
        'sample_repeat': 2,
        'batch_size': 2,
        'gradient_accumulation_steps': 4,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'num_epochs': 1,
        'warmup_ratio': 0.03,
        'max_seq_length': 256,
        'max_grad_norm': 1.0,
        'use_lora_plus': False,
        'lora_plus_lr_ratio': 16.0,
        'cot_loss_weight': 1.5,
        'llm_loss_weight': 1.0,
        'image_diffusion_loss_weight': 0.1,
        'video_diffusion_loss_weight': 0.1,
        'asr_loss_weight': 0.1,
        'tts_loss_weight': 0.1,
        'moe_aux_loss_weight': 0.01,
        'temporal_consistency_weight': 0.01,
        'cfg_dropout_rate': 0.1,
        'save_steps': 100,
        'logging_steps': 10,
        'eval_steps': 100,
        'device': 'cpu',
        'fp16': False,
        'bf16': False,
        'use_model_parallel': False,
        'empty_cache_freq': 10,
        'gradient_checkpointing': False,
    }
    defaults.update(overrides)
    
    config = MagicMock()
    for key, value in defaults.items():
        setattr(config, key, value)
    
    config.effective_batch_size = defaults['batch_size'] * defaults['gradient_accumulation_steps']
    
    return config


class TestResult:
    """Container for test results."""
    
    def __init__(self, name: str, passed: bool, message: str = "", duration: float = 0.0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration = duration
    
    def __repr__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} {self.name}: {self.message} ({self.duration:.3f}s)"


def run_test_module(module_name: str, verbosity: int = 2) -> Dict[str, Any]:
    """Run tests from a specific module and return results."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(module_name)
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return {
        'module': module_name,
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': len(result.skipped),
        'success': result.wasSuccessful(),
    }

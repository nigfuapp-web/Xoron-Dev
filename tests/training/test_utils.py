"""Unit tests for training/utils.py - Training utilities."""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Check torch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestTrainingUtilsModule(unittest.TestCase):
    """Test cases for training utils module."""
    
    def test_module_imports(self):
        """Test training utils module can be imported."""
        from training import utils
        self.assertTrue(hasattr(utils, 'create_collate_fn'))
        self.assertTrue(hasattr(utils, 'create_optimizer_and_scheduler'))
        self.assertTrue(hasattr(utils, 'train_image_diffusion_step'))
        self.assertTrue(hasattr(utils, 'train_video_diffusion_step'))
        self.assertTrue(hasattr(utils, 'train_voice_asr_step'))
        self.assertTrue(hasattr(utils, 'train_voice_tts_step'))
    
    def test_create_collate_fn(self):
        """Test create_collate_fn function exists and is callable."""
        from training.utils import create_collate_fn
        self.assertTrue(callable(create_collate_fn))
        
        # Test it creates a collate function
        collate_fn = create_collate_fn(video_frames=16, video_size=256)
        self.assertTrue(callable(collate_fn))
    
    def test_create_optimizer_and_scheduler(self):
        """Test create_optimizer_and_scheduler function exists and is callable."""
        from training.utils import create_optimizer_and_scheduler
        self.assertTrue(callable(create_optimizer_and_scheduler))
    
    def test_training_step_functions_exist(self):
        """Test all training step functions exist."""
        from training.utils import (
            train_image_diffusion_step,
            train_video_diffusion_step,
            train_voice_asr_step,
            train_voice_tts_step,
        )
        self.assertTrue(callable(train_image_diffusion_step))
        self.assertTrue(callable(train_video_diffusion_step))
        self.assertTrue(callable(train_voice_asr_step))
        self.assertTrue(callable(train_voice_tts_step))
    
    def test_training_step_handles_none_inputs(self):
        """Test training steps handle None inputs gracefully."""
        from training.utils import (
            train_image_diffusion_step,
            train_video_diffusion_step,
            train_voice_asr_step,
            train_voice_tts_step,
        )
        
        # All functions should return None when given None inputs
        self.assertIsNone(train_image_diffusion_step(None, None, None))
        self.assertIsNone(train_video_diffusion_step(None, None, None))
        self.assertIsNone(train_voice_asr_step(None, None, None))
        self.assertIsNone(train_voice_tts_step(None, None, None))


if __name__ == '__main__':
    unittest.main()

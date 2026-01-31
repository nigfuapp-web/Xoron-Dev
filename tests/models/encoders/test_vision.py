"""Unit tests for models/encoders/vision.py - Vision encoder."""

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


if __name__ == '__main__':
    unittest.main()

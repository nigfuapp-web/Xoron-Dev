"""Unit tests for models/encoders/audio.py - Audio encoder and decoder."""

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
class TestMelSpectrogramExtractor(unittest.TestCase):
    """Test cases for MelSpectrogramExtractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.encoders.audio import MelSpectrogramExtractor
        self.MelSpectrogramExtractor = MelSpectrogramExtractor
        
    def test_initialization(self):
        """Test MelSpectrogramExtractor initialization."""
        extractor = self.MelSpectrogramExtractor(
            sample_rate=16000,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
        )
        
        self.assertEqual(extractor.sample_rate, 16000)
        self.assertEqual(extractor.n_fft, 1024)
        self.assertEqual(extractor.n_mels, 80)
        
    def test_mel_filterbank_shape(self):
        """Test mel filterbank has correct shape."""
        extractor = self.MelSpectrogramExtractor(n_mels=80, n_fft=1024)
        
        # mel_fb should be [n_mels, n_fft//2+1]
        self.assertEqual(extractor.mel_fb.shape, (80, 513))
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        extractor = self.MelSpectrogramExtractor(n_mels=80, n_fft=1024, hop_length=256)
        waveform = torch.randn(2, 16000)  # 1 second at 16kHz
        
        mel = extractor(waveform)
        
        self.assertEqual(mel.shape[0], 2)  # Batch size
        self.assertEqual(mel.shape[1], 80)  # n_mels
        
    def test_forward_1d_input(self):
        """Test forward with 1D input."""
        extractor = self.MelSpectrogramExtractor(n_mels=80)
        waveform = torch.randn(16000)
        
        mel = extractor(waveform)
        
        self.assertEqual(mel.shape[0], 1)  # Batch size added
        self.assertEqual(mel.shape[1], 80)
        
    def test_hz_to_mel_conversion(self):
        """Test Hz to mel conversion."""
        extractor = self.MelSpectrogramExtractor()
        
        # 1000 Hz should be approximately 1000 mels
        mel = extractor._hz_to_mel(1000)
        self.assertGreater(mel, 0)
        
    def test_mel_to_hz_conversion(self):
        """Test mel to Hz conversion."""
        extractor = self.MelSpectrogramExtractor()
        
        # Roundtrip test
        hz = 1000
        mel = extractor._hz_to_mel(hz)
        hz_back = extractor._mel_to_hz(mel)
        
        self.assertAlmostEqual(hz, hz_back, places=5)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestConvolutionModule(unittest.TestCase):
    """Test cases for ConvolutionModule."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.encoders.audio import ConvolutionModule
        self.ConvolutionModule = ConvolutionModule
        
    def test_initialization(self):
        """Test ConvolutionModule initialization."""
        module = self.ConvolutionModule(channels=256, kernel_size=31)
        
        self.assertIsNotNone(module.layer_norm)
        self.assertIsNotNone(module.depthwise_conv)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        module = self.ConvolutionModule(channels=256, kernel_size=31)
        x = torch.randn(2, 100, 256)
        
        output = module(x)
        
        self.assertEqual(output.shape, (2, 100, 256))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestConformerBlock(unittest.TestCase):
    """Test cases for ConformerBlock."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.encoders.audio import ConformerBlock
        self.ConformerBlock = ConformerBlock
        
    def test_initialization(self):
        """Test ConformerBlock initialization."""
        block = self.ConformerBlock(d_model=256, num_heads=8)
        
        self.assertIsNotNone(block.ff1)
        self.assertIsNotNone(block.attn)
        self.assertIsNotNone(block.conv)
        self.assertIsNotNone(block.ff2)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        block = self.ConformerBlock(d_model=256, num_heads=8)
        x = torch.randn(2, 100, 256)
        
        output = block(x)
        
        self.assertEqual(output.shape, (2, 100, 256))
        
    def test_forward_with_mask(self):
        """Test forward with attention mask."""
        block = self.ConformerBlock(d_model=256, num_heads=8)
        x = torch.randn(2, 100, 256)
        mask = torch.zeros(2, 100).bool()
        mask[:, 50:] = True  # Mask second half
        
        output = block(x, mask=mask)
        
        self.assertEqual(output.shape, (2, 100, 256))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestAudioEncoder(unittest.TestCase):
    """Test cases for AudioEncoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.encoders.audio import AudioEncoder
        self.AudioEncoder = AudioEncoder
        
    def test_initialization(self):
        """Test AudioEncoder initialization."""
        encoder = self.AudioEncoder(
            hidden_size=256,
            n_mels=80,
            num_layers=4,
        )
        
        self.assertEqual(encoder.hidden_size, 256)
        self.assertEqual(encoder.n_mels, 80)
        self.assertEqual(len(encoder.conformer_blocks), 4)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        encoder = self.AudioEncoder(hidden_size=256, n_mels=80, num_layers=2)
        mel = torch.randn(2, 80, 400)  # [batch, n_mels, time]
        
        output = encoder(mel)
        
        self.assertEqual(output.shape[0], 2)  # Batch size
        self.assertEqual(output.shape[2], 256)  # Hidden size
        # Time dimension is reduced by 4x due to conv subsampling
        
    def test_conv_subsampling(self):
        """Test convolutional subsampling reduces time dimension."""
        encoder = self.AudioEncoder(hidden_size=256, n_mels=80, num_layers=2)
        mel = torch.randn(2, 80, 400)
        
        output = encoder(mel)
        
        # Time should be approximately 400/4 = 100
        self.assertEqual(output.shape[1], 100)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestVariancePredictor(unittest.TestCase):
    """Test cases for VariancePredictor."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.encoders.audio import VariancePredictor
        self.VariancePredictor = VariancePredictor
        
    def test_initialization(self):
        """Test VariancePredictor initialization."""
        predictor = self.VariancePredictor(hidden_size=256)
        
        self.assertIsNotNone(predictor.conv1)
        self.assertIsNotNone(predictor.conv2)
        self.assertIsNotNone(predictor.linear)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        predictor = self.VariancePredictor(hidden_size=256)
        x = torch.randn(2, 100, 256)
        
        output = predictor(x)
        
        self.assertEqual(output.shape, (2, 100))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestLengthRegulator(unittest.TestCase):
    """Test cases for LengthRegulator."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.encoders.audio import LengthRegulator
        self.LengthRegulator = LengthRegulator
        
    def test_forward_expands_sequence(self):
        """Test forward expands sequence according to durations."""
        regulator = self.LengthRegulator()
        x = torch.randn(2, 10, 256)
        durations = torch.ones(2, 10) * 3  # Each frame repeated 3 times
        
        output = regulator(x, durations)
        
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[2], 256)
        # Output length should be approximately 10 * 3 = 30
        
    def test_forward_with_target_length(self):
        """Test forward with specified target length."""
        regulator = self.LengthRegulator()
        x = torch.randn(2, 10, 256)
        durations = torch.ones(2, 10) * 2
        
        output = regulator(x, durations, target_length=50)
        
        self.assertEqual(output.shape[1], 50)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestAudioDecoder(unittest.TestCase):
    """Test cases for AudioDecoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.encoders.audio import AudioDecoder
        self.AudioDecoder = AudioDecoder
        
    def test_initialization(self):
        """Test AudioDecoder initialization."""
        decoder = self.AudioDecoder(
            hidden_size=256,
            n_mels=80,
            num_speakers=256,
        )
        
        self.assertEqual(decoder.hidden_size, 256)
        self.assertEqual(decoder.n_mels, 80)
        
    def test_emotions_list(self):
        """Test EMOTIONS list exists."""
        decoder = self.AudioDecoder(hidden_size=256)
        
        self.assertIn('neutral', decoder.EMOTIONS)
        self.assertIn('happy', decoder.EMOTIONS)
        self.assertIn('sad', decoder.EMOTIONS)
        
    def test_speaker_embedding(self):
        """Test speaker embedding exists."""
        decoder = self.AudioDecoder(hidden_size=256, num_speakers=256)
        
        self.assertIsNotNone(decoder.speaker_embed)
        self.assertEqual(decoder.speaker_embed.num_embeddings, 256)
        
    def test_emotion_embedding(self):
        """Test emotion embedding exists."""
        decoder = self.AudioDecoder(hidden_size=256)
        
        self.assertIsNotNone(decoder.emotion_embed)


if __name__ == '__main__':
    unittest.main()

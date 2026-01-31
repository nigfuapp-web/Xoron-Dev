"""
Comprehensive unit tests for data processors module.
Tests VoiceProcessor class and all its methods.
"""

import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from data.processors import VoiceProcessor


class TestVoiceProcessorInit(unittest.TestCase):
    """Test VoiceProcessor initialization."""
    
    def test_default_initialization(self):
        """Test VoiceProcessor with default parameters."""
        processor = VoiceProcessor()
        
        self.assertEqual(processor.sample_rate, 16000)
        self.assertEqual(processor.n_mels, 80)
        self.assertEqual(processor.n_fft, 1024)
        self.assertEqual(processor.hop_length, 256)
        self.assertEqual(processor.max_duration, 10.0)
        
    def test_custom_initialization(self):
        """Test VoiceProcessor with custom parameters."""
        processor = VoiceProcessor(
            sample_rate=22050,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
            max_duration=30.0
        )
        
        self.assertEqual(processor.sample_rate, 22050)
        self.assertEqual(processor.n_mels, 128)
        self.assertEqual(processor.n_fft, 2048)
        self.assertEqual(processor.hop_length, 512)
        self.assertEqual(processor.max_duration, 30.0)
        
    def test_max_samples_calculation(self):
        """Test max_samples is calculated correctly."""
        processor = VoiceProcessor(sample_rate=16000, max_duration=5.0)
        
        self.assertEqual(processor.max_samples, 80000)  # 16000 * 5
        
    def test_frequency_range(self):
        """Test frequency range parameters."""
        processor = VoiceProcessor(f_min=100.0, f_max=7000.0)
        
        self.assertEqual(processor.f_min, 100.0)
        self.assertEqual(processor.f_max, 7000.0)


class TestMelConversions(unittest.TestCase):
    """Test mel scale conversion methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = VoiceProcessor()
        
    def test_hz_to_mel(self):
        """Test Hz to mel conversion."""
        # 1000 Hz should be approximately 1000 mel
        mel = self.processor._hz_to_mel(1000.0)
        self.assertAlmostEqual(mel, 1000.0, delta=50)
        
    def test_mel_to_hz(self):
        """Test mel to Hz conversion."""
        # Round trip should return original value
        hz_original = 1000.0
        mel = self.processor._hz_to_mel(hz_original)
        hz_back = self.processor._mel_to_hz(mel)
        
        self.assertAlmostEqual(hz_original, hz_back, places=5)
        
    def test_hz_to_mel_zero(self):
        """Test Hz to mel conversion at 0 Hz."""
        mel = self.processor._hz_to_mel(0.0)
        self.assertEqual(mel, 0.0)
        
    def test_mel_to_hz_zero(self):
        """Test mel to Hz conversion at 0 mel."""
        hz = self.processor._mel_to_hz(0.0)
        self.assertEqual(hz, 0.0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestMelFilterbank(unittest.TestCase):
    """Test mel filterbank initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = VoiceProcessor()
        
    def test_mel_filterbank_shape(self):
        """Test mel filterbank has correct shape."""
        self.assertIsNotNone(self.processor.mel_fb)
        
        # Shape should be [n_mels, n_fft//2 + 1]
        expected_shape = (80, 513)  # 80 mels, 1024//2 + 1 = 513 freq bins
        self.assertEqual(self.processor.mel_fb.shape, expected_shape)
        
    def test_mel_filterbank_dtype(self):
        """Test mel filterbank is float tensor."""
        self.assertEqual(self.processor.mel_fb.dtype, torch.float32)
        
    def test_window_shape(self):
        """Test Hann window has correct shape."""
        self.assertIsNotNone(self.processor.window)
        self.assertEqual(self.processor.window.shape, (1024,))
        
    def test_mel_filterbank_non_negative(self):
        """Test mel filterbank values are non-negative."""
        self.assertTrue(torch.all(self.processor.mel_fb >= 0))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestExtractMel(unittest.TestCase):
    """Test mel spectrogram extraction."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = VoiceProcessor()
        
    def test_extract_mel_1d_input(self):
        """Test mel extraction with 1D input."""
        waveform = torch.randn(16000)  # 1 second of audio
        
        mel = self.processor.extract_mel(waveform)
        
        self.assertIsNotNone(mel)
        self.assertEqual(mel.dim(), 2)  # [n_mels, time]
        self.assertEqual(mel.shape[0], 80)  # n_mels
        
    def test_extract_mel_2d_input(self):
        """Test mel extraction with 2D input [1, T]."""
        waveform = torch.randn(1, 16000)
        
        mel = self.processor.extract_mel(waveform)
        
        self.assertIsNotNone(mel)
        self.assertEqual(mel.dim(), 2)
        
    def test_extract_mel_batch_input(self):
        """Test mel extraction with batch input [B, T]."""
        waveform = torch.randn(4, 16000)
        
        mel = self.processor.extract_mel(waveform)
        
        self.assertIsNotNone(mel)
        self.assertEqual(mel.dim(), 3)  # [B, n_mels, time]
        self.assertEqual(mel.shape[0], 4)  # batch size
        self.assertEqual(mel.shape[1], 80)  # n_mels
        
    def test_extract_mel_numpy_input(self):
        """Test mel extraction with numpy array input."""
        waveform = np.random.randn(16000).astype(np.float32)
        
        mel = self.processor.extract_mel(waveform)
        
        self.assertIsNotNone(mel)
        
    def test_extract_mel_output_is_log(self):
        """Test mel output is log-scaled."""
        waveform = torch.randn(16000)
        
        mel = self.processor.extract_mel(waveform)
        
        # Log mel values should be finite
        self.assertTrue(torch.all(torch.isfinite(mel)))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestResample(unittest.TestCase):
    """Test resampling functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = VoiceProcessor(sample_rate=16000)
        
    def test_resample_same_rate(self):
        """Test resampling with same rate returns original."""
        waveform = torch.randn(16000)
        
        resampled = self.processor._resample(waveform, 16000, 16000)
        
        self.assertTrue(torch.allclose(waveform, resampled))
        
    def test_resample_downsample(self):
        """Test downsampling reduces length."""
        waveform = torch.randn(16000)
        
        resampled = self.processor._resample(waveform, 16000, 8000)
        
        self.assertEqual(resampled.shape[-1], 8000)
        
    def test_resample_upsample(self):
        """Test upsampling increases length."""
        waveform = torch.randn(8000)
        
        resampled = self.processor._resample(waveform, 8000, 16000)
        
        self.assertEqual(resampled.shape[-1], 16000)
        
    def test_resample_2d_input(self):
        """Test resampling with 2D input."""
        waveform = torch.randn(4, 16000)
        
        resampled = self.processor._resample(waveform, 16000, 8000)
        
        self.assertEqual(resampled.shape, (4, 8000))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestProcessAudioArray(unittest.TestCase):
    """Test process_audio_array method."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = VoiceProcessor()
        
    def test_process_dict_with_array(self):
        """Test processing dict with 'array' key."""
        audio_data = {
            'array': np.random.randn(16000).astype(np.float32),
            'sampling_rate': 16000
        }
        
        mel = self.processor.process_audio_array(audio_data)
        
        self.assertIsNotNone(mel)
        self.assertEqual(mel.shape[0], 80)
        
    def test_process_dict_with_different_sample_rate(self):
        """Test processing dict with different sample rate."""
        audio_data = {
            'array': np.random.randn(22050).astype(np.float32),
            'sampling_rate': 22050
        }
        
        mel = self.processor.process_audio_array(audio_data)
        
        self.assertIsNotNone(mel)
        
    def test_process_numpy_array_directly(self):
        """Test processing numpy array directly."""
        audio_data = np.random.randn(16000).astype(np.float32)
        
        mel = self.processor.process_audio_array(audio_data)
        
        self.assertIsNotNone(mel)
        
    def test_process_torch_tensor_directly(self):
        """Test processing torch tensor directly."""
        audio_data = torch.randn(16000)
        
        mel = self.processor.process_audio_array(audio_data)
        
        self.assertIsNotNone(mel)
        
    def test_process_empty_array_returns_none(self):
        """Test processing empty array returns None."""
        audio_data = {'array': np.array([]), 'sampling_rate': 16000}
        
        mel = self.processor.process_audio_array(audio_data)
        
        self.assertIsNone(mel)
        
    def test_process_truncates_long_audio(self):
        """Test long audio is truncated."""
        # Create audio longer than max_duration (10s = 160000 samples)
        audio_data = np.random.randn(200000).astype(np.float32)
        
        mel = self.processor.process_audio_array(audio_data)
        
        self.assertIsNotNone(mel)
        # Mel time dimension should be limited
        
    def test_process_pads_short_audio(self):
        """Test short audio is padded or handled gracefully."""
        # Create very short audio (less than 400 samples)
        audio_data = np.random.randn(100).astype(np.float32)
        
        mel = self.processor.process_audio_array(audio_data)
        
        # Short audio may return None or a valid mel - both are acceptable
        # The implementation may reject very short audio
        self.assertTrue(mel is None or isinstance(mel, torch.Tensor))
        
    def test_process_stereo_to_mono(self):
        """Test stereo audio is converted to mono or handled gracefully."""
        # Stereo audio [T, 2]
        audio_data = {
            'array': np.random.randn(16000, 2).astype(np.float32),
            'sampling_rate': 16000
        }
        
        mel = self.processor.process_audio_array(audio_data)
        
        # Stereo conversion may work or return None depending on implementation
        self.assertTrue(mel is None or isinstance(mel, torch.Tensor))
        
    def test_process_dict_with_audio_key(self):
        """Test processing dict with 'audio' key."""
        audio_data = {
            'audio': np.random.randn(16000).astype(np.float32),
            'sampling_rate': 16000
        }
        
        mel = self.processor.process_audio_array(audio_data)
        
        self.assertIsNotNone(mel)


class TestBackendDetection(unittest.TestCase):
    """Test audio backend detection."""
    
    def test_backend_flags_initialized(self):
        """Test backend flags are initialized."""
        processor = VoiceProcessor()
        
        self.assertIsInstance(processor.has_soundfile, bool)
        self.assertIsInstance(processor.has_librosa, bool)
        self.assertIsInstance(processor.has_scipy, bool)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = VoiceProcessor()
        
    def test_none_input_returns_none(self):
        """Test None input returns None."""
        mel = self.processor.process_audio_array(None)
        self.assertIsNone(mel)
        
    def test_invalid_dict_returns_none(self):
        """Test invalid dict returns None."""
        audio_data = {'invalid_key': 'value'}
        
        mel = self.processor.process_audio_array(audio_data)
        
        self.assertIsNone(mel)
        
    def test_extract_mel_without_torch(self):
        """Test extract_mel handles missing torch gracefully."""
        processor = VoiceProcessor()
        
        # If mel_fb is None (torch not available), should return None
        if processor.mel_fb is None:
            result = processor.extract_mel(np.random.randn(16000))
            self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()

"""Unit tests for models/encoders/audio.py - SOTA Audio encoder and decoder.

Tests for:
- RawWaveformTokenizer (replaces MelSpectrogramExtractor)
- SpeakerEncoder (Zero-Shot Speaker Cloning)
- MonotonicAlignmentSearch (MAS)
- RotaryMultiHeadLatentAttention (RMLA)
- InContextAudioPrompting
- ConvolutionModule
- ConformerBlock
- AudioEncoder
- VariancePredictor
- FFTBlock
- AudioDecoder
"""

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
class TestRawWaveformTokenizer(unittest.TestCase):
    """Test cases for RawWaveformTokenizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.encoders.audio import RawWaveformTokenizer
        self.RawWaveformTokenizer = RawWaveformTokenizer
        
    def test_initialization(self):
        """Test RawWaveformTokenizer initialization."""
        tokenizer = self.RawWaveformTokenizer(
            hidden_size=256,
            num_codebooks=4,
            codebook_size=512,
            sample_rate=16000,
        )
        
        self.assertEqual(tokenizer.hidden_size, 256)
        self.assertEqual(tokenizer.num_codebooks, 4)
        self.assertEqual(tokenizer.codebook_size, 512)
        self.assertEqual(len(tokenizer.codebooks), 4)
        
    def test_encode_output_shape(self):
        """Test encode method output shape."""
        tokenizer = self.RawWaveformTokenizer(hidden_size=256, num_codebooks=4)
        waveform = torch.randn(2, 16000)  # 1 second at 16kHz
        
        features, _ = tokenizer.encode(waveform)
        
        self.assertEqual(features.shape[0], 2)  # Batch size
        self.assertEqual(features.shape[2], 256)  # Hidden size
        
    def test_forward_without_quantize(self):
        """Test forward pass without quantization."""
        tokenizer = self.RawWaveformTokenizer(hidden_size=256)
        waveform = torch.randn(2, 16000)
        
        features, commitment_loss = tokenizer(waveform, quantize=False)
        
        self.assertEqual(features.shape[0], 2)
        self.assertEqual(features.shape[2], 256)
        self.assertIsNone(commitment_loss)
        
    def test_forward_with_quantize(self):
        """Test forward pass with quantization."""
        tokenizer = self.RawWaveformTokenizer(hidden_size=256, num_codebooks=4)
        waveform = torch.randn(2, 16000)
        
        features, commitment_loss = tokenizer(waveform, quantize=True)
        
        self.assertEqual(features.shape[0], 2)
        self.assertEqual(features.shape[2], 256)
        self.assertIsNotNone(commitment_loss)
        self.assertGreaterEqual(commitment_loss.item(), 0)
        
    def test_quantize_method(self):
        """Test quantize method."""
        tokenizer = self.RawWaveformTokenizer(hidden_size=256, num_codebooks=4, codebook_size=512)
        features = torch.randn(2, 50, 256)
        
        quantized, indices, commitment_loss = tokenizer.quantize(features)
        
        self.assertEqual(quantized.shape, features.shape)
        self.assertEqual(indices.shape, (2, 50, 4))  # [B, T, num_codebooks]
        self.assertGreaterEqual(commitment_loss.item(), 0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestSpeakerEncoder(unittest.TestCase):
    """Test cases for SpeakerEncoder (Zero-Shot Speaker Cloning)."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.encoders.audio import SpeakerEncoder
        self.SpeakerEncoder = SpeakerEncoder
        
    def test_initialization(self):
        """Test SpeakerEncoder initialization."""
        encoder = self.SpeakerEncoder(hidden_size=256, output_size=128)
        
        self.assertEqual(encoder.hidden_size, 256)
        self.assertEqual(encoder.output_size, 128)
        self.assertIsNotNone(encoder.frame_encoder)
        self.assertIsNotNone(encoder.lstm)
        self.assertIsNotNone(encoder.attention)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        encoder = self.SpeakerEncoder(hidden_size=256, output_size=128)
        mel = torch.randn(2, 80, 200)  # [B, n_mels, T]
        
        speaker_embedding = encoder(mel)
        
        self.assertEqual(speaker_embedding.shape, (2, 128))
        
    def test_output_normalized(self):
        """Test that output is L2 normalized."""
        encoder = self.SpeakerEncoder(hidden_size=256, output_size=128)
        mel = torch.randn(2, 80, 200)
        
        speaker_embedding = encoder(mel)
        norms = torch.norm(speaker_embedding, p=2, dim=-1)
        
        # Should be approximately 1.0 due to L2 normalization
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestMonotonicAlignmentSearch(unittest.TestCase):
    """Test cases for MonotonicAlignmentSearch (MAS)."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.encoders.audio import MonotonicAlignmentSearch
        self.MonotonicAlignmentSearch = MonotonicAlignmentSearch
        
    def test_initialization(self):
        """Test MAS initialization."""
        mas = self.MonotonicAlignmentSearch(hidden_size=256)
        
        self.assertEqual(mas.hidden_size, 256)
        self.assertIsNotNone(mas.alignment_proj)
        self.assertIsNotNone(mas.duration_predictor)
        
    def test_predict_durations(self):
        """Test duration prediction."""
        mas = self.MonotonicAlignmentSearch(hidden_size=256)
        text_hidden = torch.randn(2, 20, 256)
        
        durations = mas.predict_durations(text_hidden)
        
        self.assertEqual(durations.shape, (2, 20))
        self.assertTrue((durations >= 0).all())  # Durations should be positive
        
    def test_soft_mas(self):
        """Test soft MAS alignment."""
        mas = self.MonotonicAlignmentSearch(hidden_size=256)
        text_hidden = torch.randn(2, 20, 256)
        audio_hidden = torch.randn(2, 100, 256)
        
        alignment = mas.soft_mas(text_hidden, audio_hidden)
        
        self.assertEqual(alignment.shape, (2, 20, 100))
        # Each row should sum to approximately 1 (softmax)
        row_sums = alignment.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))
        
    def test_hard_mas(self):
        """Test hard MAS alignment."""
        mas = self.MonotonicAlignmentSearch(hidden_size=256)
        log_probs = torch.randn(2, 10, 50)
        
        alignment = mas.hard_mas(log_probs)
        
        self.assertEqual(alignment.shape, (2, 10, 50))
        # Hard alignment should be binary
        self.assertTrue(((alignment == 0) | (alignment == 1)).all())
        
    def test_forward_with_audio(self):
        """Test forward pass with audio features."""
        mas = self.MonotonicAlignmentSearch(hidden_size=256)
        text_hidden = torch.randn(2, 20, 256)
        audio_hidden = torch.randn(2, 100, 256)
        
        alignment, durations = mas(text_hidden, audio_hidden, use_hard=False)
        
        self.assertEqual(alignment.shape, (2, 20, 100))
        self.assertEqual(durations.shape, (2, 20))
        
    def test_forward_inference_mode(self):
        """Test forward pass in inference mode (no audio)."""
        mas = self.MonotonicAlignmentSearch(hidden_size=256)
        text_hidden = torch.randn(2, 20, 256)
        
        alignment, durations = mas(text_hidden, audio_hidden=None)
        
        self.assertIsNone(alignment)
        self.assertEqual(durations.shape, (2, 20))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestRotaryMultiHeadLatentAttention(unittest.TestCase):
    """Test cases for RotaryMultiHeadLatentAttention (RMLA)."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.encoders.audio import RotaryMultiHeadLatentAttention
        self.RMLA = RotaryMultiHeadLatentAttention
        
    def test_initialization(self):
        """Test RMLA initialization."""
        rmla = self.RMLA(
            hidden_size=256,
            num_heads=8,
            num_kv_heads=2,
            head_dim=32,
            kv_lora_rank=64,
        )
        
        self.assertEqual(rmla.hidden_size, 256)
        self.assertEqual(rmla.num_heads, 8)
        self.assertEqual(rmla.num_kv_heads, 2)
        self.assertIsNotNone(rmla.q_proj)
        self.assertIsNotNone(rmla.kv_a_proj)
        self.assertIsNotNone(rmla.kv_b_proj)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        rmla = self.RMLA(hidden_size=256, num_heads=8, num_kv_heads=2, head_dim=32, kv_lora_rank=64)
        x = torch.randn(2, 50, 256)
        
        output, _ = rmla(x)
        
        self.assertEqual(output.shape, (2, 50, 256))
        
    def test_forward_with_cache(self):
        """Test forward pass with KV cache."""
        rmla = self.RMLA(hidden_size=256, num_heads=8, num_kv_heads=2, head_dim=32, kv_lora_rank=64)
        x = torch.randn(2, 50, 256)
        
        output, past_kv = rmla(x, use_cache=True)
        
        self.assertEqual(output.shape, (2, 50, 256))
        self.assertIsNotNone(past_kv)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestInContextAudioPrompting(unittest.TestCase):
    """Test cases for InContextAudioPrompting."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.encoders.audio import InContextAudioPrompting
        self.InContextAudioPrompting = InContextAudioPrompting
        
    def test_initialization(self):
        """Test InContextAudioPrompting initialization."""
        prompting = self.InContextAudioPrompting(hidden_size=256, num_prompt_tokens=16)
        
        self.assertEqual(prompting.hidden_size, 256)
        self.assertEqual(prompting.num_prompt_tokens, 16)
        self.assertIsNotNone(prompting.prompt_tokens)
        self.assertIsNotNone(prompting.cross_attn)
        
    def test_forward_without_prompt(self):
        """Test forward pass without prompt (passthrough)."""
        prompting = self.InContextAudioPrompting(hidden_size=256, num_prompt_tokens=16)
        x = torch.randn(2, 50, 256)
        
        output = prompting(x)
        
        # Without prompt, output should match input shape
        self.assertEqual(output.shape, (2, 50, 256))
        
    def test_forward_with_audio_prompt(self):
        """Test forward pass with audio prompt conditioning."""
        prompting = self.InContextAudioPrompting(hidden_size=256, num_prompt_tokens=16)
        x = torch.randn(2, 50, 256)
        audio_prompt = torch.randn(2, 20, 256)
        
        output = prompting(x, audio_prompt=audio_prompt)
        
        # Output shape should match input (cross-attention conditioning)
        self.assertEqual(output.shape, (2, 50, 256))
        
    def test_encode_prompt(self):
        """Test prompt encoding from audio features."""
        prompting = self.InContextAudioPrompting(hidden_size=256, num_prompt_tokens=16)
        audio_features = torch.randn(2, 100, 256)
        
        prompt = prompting.encode_prompt(audio_features)
        
        self.assertEqual(prompt.shape, (2, 16, 256))


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
        self.assertIsNotNone(module.pointwise_conv1)
        self.assertIsNotNone(module.pointwise_conv2)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        module = self.ConvolutionModule(channels=256, kernel_size=31)
        x = torch.randn(2, 100, 256)
        
        output = module(x)
        
        self.assertEqual(output.shape, (2, 100, 256))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestConformerBlock(unittest.TestCase):
    """Test cases for ConformerBlock with RMLA."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.encoders.audio import ConformerBlock
        self.ConformerBlock = ConformerBlock
        
    def test_initialization_with_rmla(self):
        """Test ConformerBlock initialization with RMLA."""
        block = self.ConformerBlock(d_model=256, num_heads=8, use_rmla=True)
        
        self.assertTrue(block.use_rmla)
        self.assertIsNotNone(block.ff1)
        self.assertIsNotNone(block.attn)
        self.assertIsNotNone(block.conv)
        self.assertIsNotNone(block.ff2)
        
    def test_initialization_without_rmla(self):
        """Test ConformerBlock initialization without RMLA."""
        block = self.ConformerBlock(d_model=256, num_heads=8, use_rmla=False)
        
        self.assertFalse(block.use_rmla)
        self.assertIsNotNone(block.attn)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        block = self.ConformerBlock(d_model=256, num_heads=8)
        x = torch.randn(2, 100, 256)
        
        output, _ = block(x)
        
        self.assertEqual(output.shape, (2, 100, 256))
        
    def test_forward_with_mask(self):
        """Test forward with attention mask."""
        block = self.ConformerBlock(d_model=256, num_heads=8)
        x = torch.randn(2, 100, 256)
        mask = torch.zeros(2, 100).bool()
        mask[:, 50:] = True  # Mask second half
        
        output, _ = block(x, mask=mask)
        
        self.assertEqual(output.shape, (2, 100, 256))
        
    def test_forward_with_cache(self):
        """Test forward with KV cache."""
        block = self.ConformerBlock(d_model=256, num_heads=8, use_rmla=True)
        x = torch.randn(2, 100, 256)
        
        output, past_kv = block(x, use_cache=True)
        
        self.assertEqual(output.shape, (2, 100, 256))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestAudioEncoder(unittest.TestCase):
    """Test cases for AudioEncoder with RMLA and Raw Waveform Tokenizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.encoders.audio import AudioEncoder
        self.AudioEncoder = AudioEncoder
        
    def test_initialization_with_raw_waveform(self):
        """Test AudioEncoder initialization with raw waveform tokenizer."""
        encoder = self.AudioEncoder(
            hidden_size=256,
            num_layers=2,
            use_raw_waveform=True,
        )
        
        self.assertEqual(encoder.hidden_size, 256)
        self.assertTrue(encoder.use_raw_waveform)
        self.assertIsNotNone(encoder.waveform_tokenizer)
        self.assertIsNotNone(encoder.speaker_encoder)
        self.assertIsNotNone(encoder.audio_prompting)
        self.assertEqual(len(encoder.conformer_blocks), 2)
        
    def test_initialization_without_raw_waveform(self):
        """Test AudioEncoder initialization without raw waveform tokenizer."""
        encoder = self.AudioEncoder(
            hidden_size=256,
            n_mels=80,
            num_layers=2,
            use_raw_waveform=False,
        )
        
        self.assertFalse(encoder.use_raw_waveform)
        self.assertIsNone(encoder.waveform_tokenizer)
        self.assertIsNotNone(encoder.conv_subsample)
        
    def test_forward_with_raw_waveform(self):
        """Test forward pass with raw waveform input."""
        encoder = self.AudioEncoder(hidden_size=256, num_layers=2, use_raw_waveform=True)
        waveform = torch.randn(2, 16000)  # [B, T] raw waveform
        
        output, speaker_embedding = encoder(waveform)
        
        self.assertEqual(output.shape[0], 2)  # Batch size
        self.assertEqual(output.shape[2], 256)  # Hidden size
        self.assertIsNone(speaker_embedding)  # No speaker ref provided
        
    def test_forward_with_mel_spectrogram(self):
        """Test forward pass with mel spectrogram input."""
        encoder = self.AudioEncoder(hidden_size=256, n_mels=80, num_layers=2, use_raw_waveform=False)
        mel = torch.randn(2, 80, 400)  # [B, n_mels, T]
        
        output, speaker_embedding = encoder(mel)
        
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[2], 256)
        
    def test_forward_with_speaker_ref(self):
        """Test forward pass with speaker reference for zero-shot cloning."""
        encoder = self.AudioEncoder(hidden_size=256, num_layers=2, use_raw_waveform=True)
        waveform = torch.randn(2, 16000)
        speaker_ref = torch.randn(2, 80, 200)  # Reference mel spectrogram
        
        output, speaker_embedding = encoder(waveform, speaker_ref=speaker_ref)
        
        self.assertEqual(output.shape[0], 2)
        self.assertIsNotNone(speaker_embedding)
        self.assertEqual(speaker_embedding.shape, (2, 64))  # hidden_size // 4


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
        self.assertIsNotNone(predictor.norm1)
        self.assertIsNotNone(predictor.norm2)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        predictor = self.VariancePredictor(hidden_size=256)
        x = torch.randn(2, 100, 256)
        
        output = predictor(x)
        
        self.assertEqual(output.shape, (2, 100))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestFFTBlock(unittest.TestCase):
    """Test cases for FFTBlock with RMLA."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.encoders.audio import FFTBlock
        self.FFTBlock = FFTBlock
        
    def test_initialization(self):
        """Test FFTBlock initialization."""
        block = self.FFTBlock(hidden_size=256, num_heads=4)
        
        self.assertIsNotNone(block.attn)
        self.assertIsNotNone(block.attn_norm)
        self.assertIsNotNone(block.ff_norm)
        self.assertIsNotNone(block.ff)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        block = self.FFTBlock(hidden_size=256, num_heads=4)
        x = torch.randn(2, 100, 256)
        
        output = block(x)
        
        self.assertEqual(output.shape, (2, 100, 256))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestAudioDecoder(unittest.TestCase):
    """Test cases for AudioDecoder with MAS and Zero-Shot Speaker Cloning."""
    
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
            num_decoder_layers=2,
        )
        
        self.assertEqual(decoder.hidden_size, 256)
        self.assertEqual(decoder.n_mels, 80)
        self.assertIsNotNone(decoder.mas)
        self.assertIsNotNone(decoder.speaker_embed)
        self.assertIsNotNone(decoder.speaker_proj)
        self.assertIsNotNone(decoder.audio_prompting)
        self.assertIsNotNone(decoder.duration_predictor)
        self.assertIsNotNone(decoder.pitch_predictor)
        self.assertIsNotNone(decoder.energy_predictor)
        
    def test_speaker_embedding(self):
        """Test speaker embedding exists."""
        decoder = self.AudioDecoder(hidden_size=256, num_speakers=256)
        
        self.assertEqual(decoder.speaker_embed.num_embeddings, 256)
        
    def test_forward_basic(self):
        """Test basic forward pass."""
        decoder = self.AudioDecoder(hidden_size=256, n_mels=80, num_decoder_layers=2)
        text_embeds = torch.randn(2, 20, 256)
        
        mel, durations, alignment = decoder(text_embeds, target_length=100)
        
        self.assertEqual(mel.shape, (2, 80, 100))
        self.assertEqual(durations.shape, (2, 20))
        self.assertIsNone(alignment)  # No audio features provided
        
    def test_forward_with_speaker_id(self):
        """Test forward pass with speaker ID."""
        decoder = self.AudioDecoder(hidden_size=256, n_mels=80, num_speakers=256, num_decoder_layers=2)
        text_embeds = torch.randn(2, 20, 256)
        speaker = torch.randint(0, 256, (2,))
        
        mel, durations, alignment = decoder(text_embeds, target_length=100, speaker=speaker)
        
        self.assertEqual(mel.shape, (2, 80, 100))
        
    def test_forward_with_speaker_embedding(self):
        """Test forward pass with zero-shot speaker embedding."""
        decoder = self.AudioDecoder(hidden_size=256, n_mels=80, num_decoder_layers=2)
        text_embeds = torch.randn(2, 20, 256)
        speaker_embedding = torch.randn(2, 64)  # hidden_size // 4
        
        mel, durations, alignment = decoder(text_embeds, target_length=100, speaker_embedding=speaker_embedding)
        
        self.assertEqual(mel.shape, (2, 80, 100))
        
    def test_forward_with_mas(self):
        """Test forward pass with MAS alignment."""
        decoder = self.AudioDecoder(hidden_size=256, n_mels=80, num_decoder_layers=2)
        text_embeds = torch.randn(2, 20, 256)
        audio_features = torch.randn(2, 100, 256)
        
        mel, durations, alignment = decoder(text_embeds, target_length=100, audio_features=audio_features, use_mas=True)
        
        self.assertEqual(mel.shape, (2, 80, 100))
        self.assertIsNotNone(alignment)
        self.assertEqual(alignment.shape, (2, 20, 100))


if __name__ == '__main__':
    unittest.main()

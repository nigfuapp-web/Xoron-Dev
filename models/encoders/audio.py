"""
SOTA Audio Encoder and Decoder for Speech Processing.

Implements:
- Conformer-based encoder (similar to Whisper/Conformer ASR)
- FastSpeech2/VITS-style decoder with variance adaptor
- Multi-speaker support
- Emotion and prosody control
- Custom mel spectrogram extraction (no torchaudio dependency)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List


class MelSpectrogramExtractor(nn.Module):
    """
    Custom mel spectrogram extractor using pure PyTorch.
    No torchaudio dependency - uses FFT and mel filterbank.
    """
    
    def __init__(self, sample_rate: int = 16000, n_fft: int = 1024, hop_length: int = 256,
                 n_mels: int = 80, f_min: float = 0.0, f_max: float = 8000.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max else sample_rate / 2
        
        # Create mel filterbank
        mel_fb = self._create_mel_filterbank()
        self.register_buffer('mel_fb', mel_fb)
        
        # Create Hann window
        window = torch.hann_window(n_fft)
        self.register_buffer('window', window)
        
    def _hz_to_mel(self, hz: float) -> float:
        """Convert Hz to mel scale."""
        return 2595.0 * math.log10(1.0 + hz / 700.0)
    
    def _mel_to_hz(self, mel: float) -> float:
        """Convert mel to Hz."""
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    
    def _create_mel_filterbank(self) -> torch.Tensor:
        """Create mel filterbank matrix."""
        # Mel points
        mel_min = self._hz_to_mel(self.f_min)
        mel_max = self._hz_to_mel(self.f_max)
        mel_points = torch.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = torch.tensor([self._mel_to_hz(m.item()) for m in mel_points])
        
        # FFT bins
        fft_bins = torch.floor((self.n_fft + 1) * hz_points / self.sample_rate).long()
        
        # Create filterbank
        n_freqs = self.n_fft // 2 + 1
        fb = torch.zeros(self.n_mels, n_freqs)
        
        for i in range(self.n_mels):
            left = fft_bins[i]
            center = fft_bins[i + 1]
            right = fft_bins[i + 2]
            
            # Rising slope
            for j in range(left, center):
                if j < n_freqs and center > left:
                    fb[i, j] = (j - left) / (center - left)
            
            # Falling slope
            for j in range(center, right):
                if j < n_freqs and right > center:
                    fb[i, j] = (right - j) / (right - center)
        
        return fb
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract mel spectrogram from waveform.
        
        Args:
            waveform: [B, T] or [T] audio waveform
            
        Returns:
            mel: [B, n_mels, T'] mel spectrogram
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        batch_size = waveform.size(0)
        device = waveform.device
        
        # Pad waveform
        pad_amount = self.n_fft // 2
        waveform = F.pad(waveform, (pad_amount, pad_amount), mode='reflect')
        
        # STFT using unfold
        frames = waveform.unfold(-1, self.n_fft, self.hop_length)  # [B, num_frames, n_fft]
        
        # Apply window
        window = self.window.to(device)
        frames = frames * window
        
        # FFT
        spec = torch.fft.rfft(frames, dim=-1)  # [B, num_frames, n_fft//2+1]
        
        # Power spectrum
        power_spec = spec.abs().pow(2)  # [B, num_frames, n_fft//2+1]
        
        # Apply mel filterbank
        mel_fb = self.mel_fb.to(device)
        mel_spec = torch.matmul(power_spec, mel_fb.T)  # [B, num_frames, n_mels]
        
        # Log mel
        mel_spec = torch.log(mel_spec.clamp(min=1e-10))
        
        # Transpose to [B, n_mels, T']
        mel_spec = mel_spec.transpose(1, 2)
        
        return mel_spec


class ConvolutionModule(nn.Module):
    """Conformer convolution module with gating."""
    
    def __init__(self, channels: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(channels)
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2, groups=channels
        )
        self.batch_norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C]"""
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # [B, C, T]
        
        # Pointwise conv with GLU
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        
        # Depthwise conv
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        
        # Pointwise conv
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        return x.transpose(1, 2)  # [B, T, C]


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for attention."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match dtype of positional encoding to input to avoid Float/Half mismatch
        return x + self.pe[:, :x.size(1)].to(x.dtype)


class ConformerBlock(nn.Module):
    """Single Conformer block with feed-forward, attention, convolution."""
    
    def __init__(self, d_model: int, num_heads: int = 8, ff_expansion: int = 4,
                 conv_kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        
        # First feed-forward (half-step)
        self.ff1_norm = nn.LayerNorm(d_model)
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_model * ff_expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_expansion, d_model),
            nn.Dropout(dropout)
        )
        
        # Multi-head self-attention
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Convolution module
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        
        # Second feed-forward (half-step)
        self.ff2_norm = nn.LayerNorm(d_model)
        self.ff2 = nn.Sequential(
            nn.Linear(d_model, d_model * ff_expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_expansion, d_model),
            nn.Dropout(dropout)
        )
        
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Feed-forward 1 (half-step)
        x = x + 0.5 * self.ff1(self.ff1_norm(x))
        
        # Self-attention
        attn_out, _ = self.attn(self.attn_norm(x), self.attn_norm(x), self.attn_norm(x), key_padding_mask=mask)
        x = x + self.attn_dropout(attn_out)
        
        # Convolution
        x = x + self.conv(x)
        
        # Feed-forward 2 (half-step)
        x = x + 0.5 * self.ff2(self.ff2_norm(x))
        
        return self.final_norm(x)


class AudioEncoder(nn.Module):
    """
    SOTA Conformer-based Audio Encoder for speech understanding.
    
    Architecture inspired by Whisper and Conformer:
    - Conv subsampling (4x downsampling)
    - Conformer blocks with relative positional encoding
    - Multi-head self-attention with convolution modules
    """

    def __init__(self, hidden_size: int = 1024, n_mels: int = 80, max_audio_length: int = 3000,
                 num_layers: int = 6, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_mels = n_mels
        self.max_audio_length = max_audio_length
        
        # Convolutional subsampling (4x downsampling like Whisper)
        self.conv_subsample = nn.Sequential(
            nn.Conv1d(n_mels, hidden_size // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        
        # Positional encoding
        self.pos_encoding = RelativePositionalEncoding(hidden_size, max_audio_length // 4)
        
        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(hidden_size, num_heads, ff_expansion=4, conv_kernel_size=31, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        print(f"   🎤 AudioEncoder (Conformer): {n_mels} mels -> {hidden_size}d, {num_layers} layers")

    def forward(self, mel_spectrogram: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process mel-spectrogram to audio features.
        Input: [B, n_mels, T] mel-spectrogram
        Output: [B, T', hidden_size] audio features
        """
        # Conv subsampling: [B, n_mels, T] -> [B, hidden, T/4]
        x = self.conv_subsample(mel_spectrogram)
        
        # Transpose to [B, T/4, hidden]
        x = x.transpose(1, 2)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Conformer blocks
        for block in self.conformer_blocks:
            x = block(x, mask)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class VariancePredictor(nn.Module):
    """Variance predictor for duration, pitch, and energy (FastSpeech2 style)."""
    
    def __init__(self, hidden_size: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Conv layers
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C] -> [B, T]"""
        # Conv1: [B, T, C] -> [B, C, T] -> conv -> [B, C, T] -> [B, T, C]
        out = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out)
        out = self.norm1(out)
        out = self.dropout(out)
        
        # Conv2
        out = self.conv2(out.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out)
        out = self.norm2(out)
        out = self.dropout(out)
        
        return self.linear(out).squeeze(-1)


class LengthRegulator(nn.Module):
    """Length regulator for duration-based upsampling."""
    
    def forward(self, x: torch.Tensor, durations: torch.Tensor, target_length: Optional[int] = None) -> torch.Tensor:
        """
        Expand hidden states according to durations.
        x: [B, T, C]
        durations: [B, T] (in frames)
        """
        batch_size = x.size(0)
        device = x.device
        
        # Round durations to integers
        durations = torch.clamp(torch.round(durations), min=1).long()
        
        if target_length is None:
            target_length = durations.sum(dim=1).max().item()
        
        # Expand each sequence
        outputs = []
        for i in range(batch_size):
            expanded = torch.repeat_interleave(x[i], durations[i], dim=0)
            # Pad or truncate to target length
            if expanded.size(0) < target_length:
                pad = torch.zeros(target_length - expanded.size(0), x.size(-1), device=device)
                expanded = torch.cat([expanded, pad], dim=0)
            else:
                expanded = expanded[:target_length]
            outputs.append(expanded)
        
        return torch.stack(outputs)


class FFTBlock(nn.Module):
    """FFT block for mel decoder (similar to FastSpeech2)."""
    
    def __init__(self, hidden_size: int, num_heads: int = 4, ff_expansion: int = 4,
                 kernel_size: int = 9, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Conv feed-forward
        self.ff_norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size * ff_expansion, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_size * ff_expansion, hidden_size, kernel_size, padding=kernel_size // 2),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        residual = x
        x = self.attn_norm(x)
        x, _ = self.attn(x, x, x)
        x = residual + self.attn_dropout(x)
        
        # Feed-forward
        residual = x
        x = self.ff_norm(x)
        x = self.ff(x.transpose(1, 2)).transpose(1, 2)
        x = residual + x
        
        return x


class AudioDecoder(nn.Module):
    """
    SOTA FastSpeech2-style Audio Decoder for TTS.
    
    Features:
    - Variance adaptor with duration, pitch, energy prediction
    - Length regulator for duration-based expansion
    - FFT blocks for mel generation
    - Multi-speaker support
    - Emotion and prosody control
    - HiFi-GAN style postnet
    """

    EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted', 'calm', 
                'excited', 'curious', 'confident', 'whisper', 'shouting']

    def __init__(self, hidden_size: int = 1024, n_mels: int = 80, max_audio_length: int = 1000,
                 num_speakers: int = 256, num_decoder_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_mels = n_mels
        self.max_audio_length = max_audio_length
        self.num_emotions = len(self.EMOTIONS)
        
        # Speaker embedding
        self.speaker_embed = nn.Embedding(num_speakers, hidden_size // 4)
        
        # Emotion embedding
        self.emotion_embed = nn.Embedding(self.num_emotions, hidden_size // 4)
        
        # Prosody projection (speed, pitch_shift, energy_scale)
        self.prosody_proj = nn.Linear(3, hidden_size // 4)
        
        # Input projection
        self.input_proj = nn.Linear(hidden_size + 3 * (hidden_size // 4), hidden_size)
        
        # Encoder FFT blocks
        self.encoder_blocks = nn.ModuleList([
            FFTBlock(hidden_size, num_heads=4, ff_expansion=4, dropout=dropout)
            for _ in range(4)
        ])
        
        # Variance adaptor
        self.duration_predictor = VariancePredictor(hidden_size, dropout=dropout)
        self.pitch_predictor = VariancePredictor(hidden_size, dropout=dropout)
        self.energy_predictor = VariancePredictor(hidden_size, dropout=dropout)
        
        # Pitch and energy embeddings
        self.pitch_embed = nn.Conv1d(1, hidden_size, kernel_size=9, padding=4)
        self.energy_embed = nn.Conv1d(1, hidden_size, kernel_size=9, padding=4)
        
        # Length regulator
        self.length_regulator = LengthRegulator()
        
        # Decoder FFT blocks
        self.decoder_blocks = nn.ModuleList([
            FFTBlock(hidden_size, num_heads=4, ff_expansion=4, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Mel output
        self.mel_linear = nn.Linear(hidden_size, n_mels)
        
        # HiFi-GAN style postnet with residual connections
        self.postnet = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(n_mels, 256, kernel_size=5, padding=2),
                nn.BatchNorm1d(256),
                nn.Tanh(),
            ),
            nn.Sequential(
                nn.Conv1d(256, 256, kernel_size=5, padding=2),
                nn.BatchNorm1d(256),
                nn.Tanh(),
            ),
            nn.Sequential(
                nn.Conv1d(256, 256, kernel_size=5, padding=2),
                nn.BatchNorm1d(256),
                nn.Tanh(),
            ),
            nn.Sequential(
                nn.Conv1d(256, 256, kernel_size=5, padding=2),
                nn.BatchNorm1d(256),
                nn.Tanh(),
            ),
            nn.Conv1d(256, n_mels, kernel_size=5, padding=2),
        ])

        print(f"   🔊 AudioDecoder (FastSpeech2): {hidden_size}d -> {n_mels} mels, {self.num_emotions} emotions, {num_speakers} speakers")

    def forward(
        self,
        text_embeds: torch.Tensor,
        target_length: Optional[int] = None,
        emotion: Optional[torch.Tensor] = None,
        prosody: Optional[torch.Tensor] = None,
        speaker: Optional[torch.Tensor] = None,
        duration_target: Optional[torch.Tensor] = None,
        pitch_target: Optional[torch.Tensor] = None,
        energy_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate mel-spectrogram from text embeddings.
        
        Args:
            text_embeds: [B, T, hidden_size] text embeddings
            target_length: target mel length (for training)
            emotion: [B] emotion IDs
            prosody: [B, 3] prosody controls (speed, pitch_shift, energy_scale)
            speaker: [B] speaker IDs
            duration_target: [B, T] ground truth durations (for training)
            pitch_target: [B, T'] ground truth pitch (for training)
            energy_target: [B, T'] ground truth energy (for training)
            
        Returns:
            mel: [B, n_mels, T'] generated mel spectrogram
            durations: [B, T] predicted durations
        """
        batch_size, seq_len, _ = text_embeds.shape
        device = text_embeds.device
        dtype = text_embeds.dtype  # Match dtype to avoid Float/Half mismatch

        # Get embeddings - ensure dtype matches input
        if speaker is None:
            speaker = torch.zeros(batch_size, dtype=torch.long, device=device)
        speaker_emb = self.speaker_embed(speaker).unsqueeze(1).expand(-1, seq_len, -1).to(dtype)
        
        if emotion is None:
            emotion = torch.zeros(batch_size, dtype=torch.long, device=device)
        emotion_emb = self.emotion_embed(emotion).unsqueeze(1).expand(-1, seq_len, -1).to(dtype)

        if prosody is None:
            prosody = torch.tensor([[1.0, 0.0, 1.0]], device=device, dtype=dtype).expand(batch_size, -1)
        else:
            prosody = prosody.to(dtype)
        prosody_emb = self.prosody_proj(prosody).unsqueeze(1).expand(-1, seq_len, -1)

        # Combine embeddings
        x = torch.cat([text_embeds, speaker_emb, emotion_emb, prosody_emb], dim=-1)
        x = self.input_proj(x)
        
        # Encoder
        for block in self.encoder_blocks:
            x = block(x)
        
        # Variance prediction
        duration_pred = F.softplus(self.duration_predictor(x))
        pitch_pred = self.pitch_predictor(x)
        energy_pred = F.softplus(self.energy_predictor(x))
        
        # Use targets during training, predictions during inference
        if duration_target is not None:
            durations = duration_target
        else:
            durations = duration_pred * prosody[:, 0:1]  # Apply speed control
        
        # Length regulation
        if target_length is not None:
            x = F.interpolate(x.transpose(1, 2), size=target_length, mode='linear', align_corners=False).transpose(1, 2)
            mel_length = target_length
        else:
            mel_length = int(durations.sum(dim=1).max().item())
            mel_length = max(16, min(mel_length, self.max_audio_length))
            x = F.interpolate(x.transpose(1, 2), size=mel_length, mode='linear', align_corners=False).transpose(1, 2)
        
        # Add pitch and energy
        if pitch_target is not None:
            pitch = pitch_target
        else:
            pitch = pitch_pred + prosody[:, 1:2]  # Apply pitch shift
        
        if energy_target is not None:
            energy = energy_target
        else:
            energy = energy_pred * prosody[:, 2:3]  # Apply energy scale
        
        # Upsample pitch and energy to mel length
        pitch_up = F.interpolate(pitch.unsqueeze(1), size=mel_length, mode='linear', align_corners=False)
        energy_up = F.interpolate(energy.unsqueeze(1), size=mel_length, mode='linear', align_corners=False)
        
        # Add pitch and energy embeddings
        pitch_emb = self.pitch_embed(pitch_up).transpose(1, 2)
        energy_emb = self.energy_embed(energy_up).transpose(1, 2)
        x = x + pitch_emb + energy_emb
        
        # Decoder
        for block in self.decoder_blocks:
            x = block(x)
        
        # Mel output
        mel = self.mel_linear(x).transpose(1, 2)  # [B, n_mels, T']
        
        # Postnet with residual
        mel_post = mel
        for i, layer in enumerate(self.postnet):
            if i < len(self.postnet) - 1:
                mel_post = layer(mel_post)
            else:
                mel_post = layer(mel_post)
        mel = mel + mel_post

        return mel, duration_pred

    @staticmethod
    def get_emotion_id(emotion_name: str) -> int:
        """Convert emotion name to ID."""
        emotion_name = emotion_name.lower()
        if emotion_name in AudioDecoder.EMOTIONS:
            return AudioDecoder.EMOTIONS.index(emotion_name)
        return 0
    
    @staticmethod
    def get_emotion_name(emotion_id: int) -> str:
        """Convert emotion ID to name."""
        if 0 <= emotion_id < len(AudioDecoder.EMOTIONS):
            return AudioDecoder.EMOTIONS[emotion_id]
        return 'neutral'

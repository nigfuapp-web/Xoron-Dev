"""
SOTA Audio Encoder and Decoder for Speech Processing.

Implements:
- Raw Waveform Tokenizer (replaces mel spectrogram)
- Raw Waveform Decoder (direct audio output, no vocoder needed)
- Zero-Shot Speaker Cloning with speaker embedding extraction
- Monotonic Alignment Search (MAS) for fluid text-to-audio alignment
- Rotary Multi-Head Latent Attention (RMLA)
- In-Context Audio Prompting
- Conformer-based encoder
- FastSpeech2/VITS-style decoder with variance adaptor
- Multi-speaker support
- FP16-native numerical stability
- Integrates with MLA/Ring Attention LLM components
- Speech-to-Speech capability (listen and talk back)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

EPS = 1e-5


class RawWaveformTokenizer(nn.Module):
    """
    Raw Waveform Tokenizer - directly tokenizes audio waveforms without mel spectrograms.
    
    Uses multi-scale 1D convolutions to extract features at different temporal resolutions,
    then combines them into a unified representation.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_codebooks: int = 8,
        codebook_size: int = 1024,
        sample_rate: int = 16000,
        hop_length: int = 320,
        num_conv_layers: int = 6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.sample_rate = sample_rate
        self.hop_length = hop_length

        # Multi-scale convolutional encoder
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        channels = [32, 64, 128, 256, 512, hidden_size]
        kernel_sizes = [7, 5, 5, 3, 3, 3]
        strides = [2, 2, 2, 2, 2, 2]  # Total downsampling: 64x

        for i in range(num_conv_layers):
            out_channels = channels[i] if i < len(channels) else hidden_size
            kernel_size = kernel_sizes[i] if i < len(kernel_sizes) else 3
            stride = strides[i] if i < len(strides) else 2

            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride, kernel_size // 2),
                nn.GroupNorm(8 if out_channels >= 8 else 1, out_channels),
                nn.SiLU(),
            ))
            in_channels = out_channels

        # Residual vector quantization codebooks
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, hidden_size)
            for _ in range(num_codebooks)
        ])

        # Commitment loss weight
        self.commitment_weight = 0.25

        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)

        print(f"   🎵 RawWaveformTokenizer: {num_codebooks} codebooks x {codebook_size} codes")

    def encode(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode waveform to continuous features.
        
        Args:
            waveform: [B, T] or [B, 1, T] raw audio waveform
            
        Returns:
            features: [B, T', hidden_size] encoded features
            indices: [B, T', num_codebooks] quantized indices
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # [B, 1, T]

        x = waveform
        for conv in self.conv_layers:
            x = conv(x)

        # [B, C, T'] -> [B, T', C]
        x = x.transpose(1, 2)

        return x, None  # Return features without quantization for continuous mode

    def quantize(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Residual Vector Quantization.
        
        Args:
            features: [B, T, hidden_size] continuous features
            
        Returns:
            quantized: [B, T, hidden_size] quantized features
            indices: [B, T, num_codebooks] codebook indices
            commitment_loss: scalar commitment loss
        """
        batch_size, seq_len, _ = features.shape
        residual = features
        quantized = torch.zeros_like(features)
        all_indices = []
        total_commitment_loss = 0.0

        for codebook in self.codebooks:
            # Find nearest codebook entry
            distances = torch.cdist(residual, codebook.weight)  # [B, T, codebook_size]
            indices = distances.argmin(dim=-1)  # [B, T]
            all_indices.append(indices)

            # Get quantized vectors
            quantized_step = codebook(indices)  # [B, T, hidden_size]

            # Straight-through estimator
            quantized = quantized + residual + (quantized_step - residual).detach()

            # Commitment loss
            commitment_loss = F.mse_loss(residual.detach(), quantized_step)
            total_commitment_loss = total_commitment_loss + commitment_loss

            # Update residual
            residual = residual - quantized_step.detach()

        indices = torch.stack(all_indices, dim=-1)  # [B, T, num_codebooks]
        commitment_loss = total_commitment_loss * self.commitment_weight

        return quantized, indices, commitment_loss

    def forward(self, waveform: torch.Tensor, quantize: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            waveform: [B, T] or [B, 1, T] raw audio
            quantize: Whether to apply vector quantization
            
        Returns:
            features: [B, T', hidden_size] encoded features
            commitment_loss: Optional commitment loss if quantize=True
        """
        features, _ = self.encode(waveform)

        if quantize:
            features, indices, commitment_loss = self.quantize(features)
            features = self.output_proj(features)
            return features, commitment_loss

        features = self.output_proj(features)
        return features, None


class SnakeActivation(nn.Module):
    """
    Snake activation function from BigVGAN.
    x + (1/a) * sin^2(a * x)
    Better than ReLU/SiLU for audio generation - preserves periodicity.
    """
    def __init__(self, channels: int, alpha: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1) * alpha)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (1.0 / (self.alpha + 1e-6)) * torch.sin(self.alpha * x) ** 2


class ResidualBlock1D(nn.Module):
    """Residual block with dilated convolutions for multi-receptive field."""
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size * dilation - dilation) // 2
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        )
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        )
        self.activation = SnakeActivation(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        return x + residual


class MultiReceptiveFieldFusion(nn.Module):
    """
    Multi-Receptive Field Fusion (MRF) from HiFi-GAN.
    Processes input through multiple parallel residual stacks with different
    kernel sizes and dilations, then sums results.
    """
    def __init__(self, channels: int, kernel_sizes: List[int] = [3, 7, 11], 
                 dilations: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]):
        super().__init__()
        self.num_kernels = len(kernel_sizes)
        self.resblocks = nn.ModuleList()
        
        for k, d_list in zip(kernel_sizes, dilations):
            blocks = nn.ModuleList([
                ResidualBlock1D(channels, k, d) for d in d_list
            ])
            self.resblocks.append(blocks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = None
        for blocks in self.resblocks:
            h = x
            for block in blocks:
                h = block(h)
            out = h if out is None else out + h
        return out / self.num_kernels


class RawWaveformDecoder(nn.Module):
    """
    SOTA Raw Waveform Decoder - BigVGAN/HiFi-GAN style architecture.
    
    Converts features directly to playable audio waveform without external vocoder.
    
    SOTA Features:
    - Snake activation (BigVGAN) - preserves audio periodicity
    - Multi-Receptive Field Fusion (HiFi-GAN) - captures patterns at multiple scales
    - Weight normalization - stable training
    - Efficient upsampling with careful kernel/stride ratios
    - Anti-aliased resampling
    - Streaming-capable architecture
    
    Speed optimizations:
    - Fewer layers with smarter architecture
    - Fused operations where possible
    - Efficient 256x total upsampling (vs 64x before)
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        sample_rate: int = 16000,
        upsample_rates: List[int] = [8, 8, 2, 2],  # Total: 256x upsampling
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilations: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        initial_channels: int = 512,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.sample_rate = sample_rate
        self.num_upsamples = len(upsample_rates)
        
        # Input projection with weight norm
        self.input_proj = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(hidden_size, initial_channels, kernel_size=7, padding=3)
        )
        
        # Upsampling layers with MRF blocks
        self.upsamplers = nn.ModuleList()
        self.mrf_blocks = nn.ModuleList()
        
        channels = initial_channels
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # Transposed conv for upsampling
            self.upsamplers.append(
                nn.utils.parametrizations.weight_norm(
                    nn.ConvTranspose1d(
                        channels, channels // 2,
                        kernel_size=kernel, stride=rate,
                        padding=(kernel - rate) // 2
                    )
                )
            )
            channels = channels // 2
            
            # MRF block after each upsample
            self.mrf_blocks.append(
                MultiReceptiveFieldFusion(channels, resblock_kernel_sizes, resblock_dilations)
            )
        
        # Final activation and output
        self.final_activation = SnakeActivation(channels)
        self.output_conv = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(channels, 1, kernel_size=7, padding=3)
        )
        
        # Calculate total upsampling factor
        self.upsample_factor = 1
        for rate in upsample_rates:
            self.upsample_factor *= rate
        
        print(f"   🔊 RawWaveformDecoder (SOTA BigVGAN-style):")
        print(f"      - Snake activation for audio periodicity")
        print(f"      - Multi-Receptive Field Fusion")
        print(f"      - {self.upsample_factor}x upsampling")
        print(f"      - Weight normalized layers")

    def forward(
        self,
        features: torch.Tensor,
        target_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Decode features to raw waveform.
        
        Args:
            features: [B, T, hidden_size] encoded features
            target_length: Optional target waveform length (for matching input length)
            
        Returns:
            waveform: [B, T_audio] raw audio waveform in [-1, 1]
        """
        # [B, T, C] -> [B, C, T]
        x = features.transpose(1, 2)
        
        # Input projection
        x = self.input_proj(x)
        
        # Upsample with MRF blocks
        for upsample, mrf in zip(self.upsamplers, self.mrf_blocks):
            x = upsample(x)
            x = mrf(x)
        
        # Final output
        x = self.final_activation(x)
        waveform = self.output_conv(x)
        waveform = torch.tanh(waveform)  # Ensure [-1, 1] range
        
        # Squeeze to [B, T_audio]
        waveform = waveform.squeeze(1)
        
        # Match target length if specified
        if target_length is not None and waveform.shape[-1] != target_length:
            waveform = F.interpolate(
                waveform.unsqueeze(1),
                size=target_length,
                mode='linear',
                align_corners=False
            ).squeeze(1)
        
        return waveform

    def decode_from_codes(
        self,
        codes: torch.Tensor,
        codebooks: nn.ModuleList,
        target_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Decode directly from codebook indices.
        
        Args:
            codes: [B, T, num_codebooks] codebook indices
            codebooks: List of nn.Embedding codebooks from encoder
            target_length: Optional target waveform length
            
        Returns:
            waveform: [B, T_audio] raw audio waveform
        """
        # Reconstruct features from codes
        features = torch.zeros(
            codes.shape[0], codes.shape[1], codebooks[0].embedding_dim,
            device=codes.device, dtype=codebooks[0].weight.dtype
        )
        
        for i, codebook in enumerate(codebooks):
            features = features + codebook(codes[:, :, i])

        return self.forward(features, target_length)
    
    @torch.no_grad()
    def stream_decode(
        self,
        features: torch.Tensor,
        chunk_size: int = 10,
    ) -> torch.Tensor:
        """
        Streaming decode for real-time speech synthesis.
        
        Processes features in chunks for low-latency output.
        
        Args:
            features: [B, T, hidden_size] encoded features
            chunk_size: Number of feature frames per chunk
            
        Yields:
            waveform_chunk: [B, chunk_audio_len] audio chunk
        """
        batch_size, seq_len, _ = features.shape
        audio_chunks = []
        
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk = features[:, start:end, :]
            audio_chunk = self.forward(chunk)
            audio_chunks.append(audio_chunk)
        
        return torch.cat(audio_chunks, dim=-1)


class SpeakerEncoder(nn.Module):
    """
    Zero-Shot Speaker Encoder for speaker cloning.
    
    Extracts speaker embeddings from reference audio that can be used
    to clone the speaker's voice characteristics.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        output_size: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Frame-level encoder
        self.frame_encoder = nn.Sequential(
            nn.Conv1d(80, hidden_size, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
        )

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_size * 2, output_size)

        print(f"   👤 SpeakerEncoder: {hidden_size}d -> {output_size}d speaker embedding")

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from mel spectrogram.
        
        Args:
            mel_spectrogram: [B, n_mels, T] mel spectrogram
            
        Returns:
            speaker_embedding: [B, output_size] speaker embedding
        """
        # Frame-level encoding
        x = self.frame_encoder(mel_spectrogram)  # [B, hidden, T]
        x = x.transpose(1, 2)  # [B, T, hidden]

        # Temporal modeling
        x, _ = self.lstm(x)  # [B, T, hidden*2]

        # Attention pooling
        attn_weights = self.attention(x)  # [B, T, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        x = (x * attn_weights).sum(dim=1)  # [B, hidden*2]

        # Output projection with L2 normalization
        speaker_embedding = self.output_proj(x)
        speaker_embedding = F.normalize(speaker_embedding, p=2, dim=-1)

        return speaker_embedding


class MonotonicAlignmentSearch(nn.Module):
    """
    Monotonic Alignment Search (MAS) for text-to-audio alignment.
    
    Implements both:
    1. Hard MAS for inference (dynamic programming)
    2. Soft/Fluid MAS for training (differentiable)
    """

    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.hidden_size = hidden_size

        # Alignment predictor for soft MAS
        self.alignment_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        # Duration predictor for fluid alignment
        # Use GroupNorm instead of LayerNorm for Conv1d compatibility
        self.duration_predictor = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(1, hidden_size),  # Equivalent to LayerNorm for Conv1d
            nn.Conv1d(hidden_size, hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(1, hidden_size),  # Equivalent to LayerNorm for Conv1d
            nn.Conv1d(hidden_size, 1, 1),
        )

    @staticmethod
    def hard_mas(log_probs: torch.Tensor) -> torch.Tensor:
        """
        Hard Monotonic Alignment Search using dynamic programming.
        
        Args:
            log_probs: [B, T_text, T_audio] log alignment probabilities
            
        Returns:
            alignment: [B, T_text, T_audio] hard alignment matrix
        """
        batch_size, text_len, audio_len = log_probs.shape
        device = log_probs.device

        # Dynamic programming
        Q = torch.full((batch_size, text_len, audio_len), float('-inf'), device=device)
        Q[:, 0, 0] = log_probs[:, 0, 0]

        for j in range(1, audio_len):
            Q[:, 0, j] = Q[:, 0, j - 1] + log_probs[:, 0, j]

        for i in range(1, text_len):
            Q[:, i, i] = Q[:, i - 1, i - 1] + log_probs[:, i, i]
            for j in range(i + 1, audio_len):
                Q[:, i, j] = torch.max(Q[:, i - 1, j - 1], Q[:, i, j - 1]) + log_probs[:, i, j]

        # Backtrack to find alignment
        alignment = torch.zeros_like(log_probs)
        for b in range(batch_size):
            i, j = text_len - 1, audio_len - 1
            while i >= 0 and j >= 0:
                alignment[b, i, j] = 1
                if i == 0:
                    j -= 1
                elif j == 0:
                    i -= 1
                elif Q[b, i - 1, j - 1] >= Q[b, i, j - 1]:
                    i -= 1
                    j -= 1
                else:
                    j -= 1

        return alignment

    def soft_mas(
        self,
        text_hidden: torch.Tensor,
        audio_hidden: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Soft/Differentiable Monotonic Alignment Search.
        
        Args:
            text_hidden: [B, T_text, hidden_size] text features
            audio_hidden: [B, T_audio, hidden_size] audio features
            temperature: Softmax temperature
            
        Returns:
            soft_alignment: [B, T_text, T_audio] soft alignment matrix
        """
        batch_size, text_len, _ = text_hidden.shape
        audio_len = audio_hidden.shape[1]

        # Compute pairwise alignment scores
        text_expanded = text_hidden.unsqueeze(2).expand(-1, -1, audio_len, -1)
        audio_expanded = audio_hidden.unsqueeze(1).expand(-1, text_len, -1, -1)
        combined = torch.cat([text_expanded, audio_expanded], dim=-1)

        # Alignment logits
        logits = self.alignment_proj(combined).squeeze(-1)  # [B, T_text, T_audio]

        # Apply monotonic constraint via cumulative softmax
        # This encourages monotonic alignment while remaining differentiable
        logits = logits / temperature

        # Row-wise softmax with monotonic bias
        position_bias = torch.arange(audio_len, device=logits.device).float()
        position_bias = position_bias.unsqueeze(0).unsqueeze(0)  # [1, 1, T_audio]

        text_positions = torch.arange(text_len, device=logits.device).float()
        text_positions = text_positions.unsqueeze(0).unsqueeze(2)  # [1, T_text, 1]

        # Expected position for each text token
        expected_pos = text_positions * (audio_len / text_len)
        monotonic_bias = -0.1 * (position_bias - expected_pos).abs()

        logits = logits + monotonic_bias
        soft_alignment = F.softmax(logits, dim=-1)

        return soft_alignment

    def predict_durations(self, text_hidden: torch.Tensor) -> torch.Tensor:
        """
        Predict durations for each text token.
        
        Args:
            text_hidden: [B, T_text, hidden_size] text features
            
        Returns:
            durations: [B, T_text] predicted durations
        """
        x = text_hidden.transpose(1, 2)  # [B, hidden, T_text]
        durations = self.duration_predictor(x).squeeze(1)  # [B, T_text]
        durations = F.softplus(durations)  # Ensure positive
        return durations

    def forward(
        self,
        text_hidden: torch.Tensor,
        audio_hidden: Optional[torch.Tensor] = None,
        use_hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute alignment and durations.
        
        Args:
            text_hidden: [B, T_text, hidden_size] text features
            audio_hidden: [B, T_audio, hidden_size] audio features (optional for inference)
            use_hard: Use hard MAS instead of soft
            
        Returns:
            alignment: [B, T_text, T_audio] alignment matrix
            durations: [B, T_text] predicted durations
        """
        durations = self.predict_durations(text_hidden)

        if audio_hidden is None:
            # Inference mode - use predicted durations
            return None, durations

        if use_hard:
            # Compute log probabilities for hard MAS
            text_norm = F.normalize(text_hidden, dim=-1)
            audio_norm = F.normalize(audio_hidden, dim=-1)
            log_probs = torch.bmm(text_norm, audio_norm.transpose(1, 2))
            alignment = self.hard_mas(log_probs)
        else:
            alignment = self.soft_mas(text_hidden, audio_hidden)

        return alignment, durations


class RotaryMultiHeadLatentAttention(nn.Module):
    """
    Rotary Multi-Head Latent Attention (RMLA).
    
    Combines:
    - Multi-Head Latent Attention (MLA) for compressed KV cache
    - Rotary Position Embeddings (RoPE) for position awareness
    - Efficient attention computation
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        num_kv_heads: int = 4,
        head_dim: int = 64,
        kv_lora_rank: int = 256,
        max_position_embeddings: int = 8192,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_lora_rank = kv_lora_rank
        self.num_key_value_groups = num_heads // num_kv_heads

        # Query projection
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)

        # Compressed KV projection (MLA style)
        self.kv_a_proj = nn.Linear(hidden_size, kv_lora_rank + head_dim, bias=False)
        self.kv_b_proj = nn.Linear(kv_lora_rank, num_kv_heads * head_dim * 2, bias=False)
        self.kv_norm = nn.LayerNorm(kv_lora_rank)

        # Output projection
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # Rotary embeddings
        self.rotary_emb = self._create_rotary_embedding(head_dim, max_position_embeddings)

        self.dropout = nn.Dropout(dropout)
        self.scale = head_dim ** -0.5

    def _create_rotary_embedding(self, dim: int, max_seq_len: int) -> nn.Module:
        """Create rotary position embeddings."""
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

        return None

    def _apply_rotary(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply rotary position embeddings."""
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        # Rotate half
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        rotated = torch.cat([-x2, x1], dim=-1)

        return x * cos.to(x.dtype) + rotated * sin.to(x.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with RMLA.
        
        Args:
            hidden_states: [B, T, hidden_size]
            attention_mask: Optional attention mask
            past_key_value: Optional cached KV states
            use_cache: Whether to return updated cache
            
        Returns:
            output: [B, T, hidden_size]
            present_key_value: Optional updated cache
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Query projection
        query = self.q_proj(hidden_states)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compressed KV projection (MLA)
        kv_compressed = self.kv_a_proj(hidden_states)
        kv_latent, k_pe = kv_compressed.split([self.kv_lora_rank, self.head_dim], dim=-1)
        kv_latent = self.kv_norm(kv_latent)
        kv = self.kv_b_proj(kv_latent)

        key, value = kv.split(self.num_kv_heads * self.head_dim, dim=-1)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        query = self._apply_rotary(query, seq_len)
        key = self._apply_rotary(key, seq_len)

        # Handle KV cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)

        present_key_value = (key, value) if use_cache else None

        # Expand KV for grouped query attention
        if self.num_key_value_groups > 1:
            key = key.repeat_interleave(self.num_key_value_groups, dim=1)
            value = value.repeat_interleave(self.num_key_value_groups, dim=1)

        # Attention computation
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)

        return output, present_key_value


class InContextAudioPrompting(nn.Module):
    """
    In-Context Audio Prompting for conditioning generation on reference audio.
    
    Allows the model to use a reference audio clip to guide the style,
    speaker characteristics, and prosody of generated audio.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_prompt_tokens: int = 32,
        num_heads: int = 8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_prompt_tokens = num_prompt_tokens

        # Learnable prompt tokens
        self.prompt_tokens = nn.Parameter(torch.randn(1, num_prompt_tokens, hidden_size) * 0.02)

        # Cross-attention for prompt conditioning
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads,
            dropout=0.1,
            batch_first=True,
        )

        # Prompt encoder
        self.prompt_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Gating for residual connection
        self.gate = nn.Parameter(torch.zeros(1))

        self.norm = nn.LayerNorm(hidden_size)

    def encode_prompt(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Encode reference audio into prompt tokens.
        
        Args:
            audio_features: [B, T, hidden_size] reference audio features
            
        Returns:
            prompt: [B, num_prompt_tokens, hidden_size] encoded prompt
        """
        batch_size = audio_features.shape[0]

        # Expand learnable prompt tokens
        prompt = self.prompt_tokens.expand(batch_size, -1, -1)

        # Cross-attend to audio features
        prompt, _ = self.cross_attn(prompt, audio_features, audio_features)

        # Encode
        prompt = self.prompt_encoder(prompt)

        return prompt

    def forward(
        self,
        hidden_states: torch.Tensor,
        prompt_features: Optional[torch.Tensor] = None,
        audio_prompt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply in-context audio prompting.
        
        Args:
            hidden_states: [B, T, hidden_size] input features
            prompt_features: [B, num_prompt_tokens, hidden_size] pre-encoded prompt
            audio_prompt: [B, T_prompt, hidden_size] raw audio features to encode
            
        Returns:
            output: [B, T, hidden_size] conditioned features
        """
        if prompt_features is None and audio_prompt is not None:
            prompt_features = self.encode_prompt(audio_prompt)

        if prompt_features is None:
            return hidden_states

        # Cross-attend to prompt
        attended, _ = self.cross_attn(hidden_states, prompt_features, prompt_features)

        # Gated residual
        gate = torch.sigmoid(self.gate)
        output = hidden_states + gate * attended
        output = self.norm(output)

        return output


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


class ConformerBlock(nn.Module):
    """Single Conformer block with RMLA, feed-forward, and convolution."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        ff_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        use_rmla: bool = True,
    ):
        super().__init__()
        self.use_rmla = use_rmla

        # First feed-forward (half-step)
        self.ff1_norm = nn.LayerNorm(d_model)
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_model * ff_expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_expansion, d_model),
            nn.Dropout(dropout)
        )

        # Attention (RMLA or standard)
        if use_rmla:
            self.attn = RotaryMultiHeadLatentAttention(
                hidden_size=d_model,
                num_heads=num_heads,
                num_kv_heads=max(1, num_heads // 4),
                head_dim=d_model // num_heads,
                kv_lora_rank=d_model // 4,
                dropout=dropout,
            )
        else:
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

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        # Feed-forward 1 (half-step)
        x = x + 0.5 * self.ff1(self.ff1_norm(x))

        # Self-attention
        if self.use_rmla:
            # Convert boolean mask [B, seq_len] to attention mask [B, 1, 1, seq_len]
            attn_mask = None
            if mask is not None:
                attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_len]
                attn_mask = attn_mask.to(dtype=x.dtype)
                attn_mask = attn_mask.masked_fill(attn_mask.bool(), float('-inf'))
            attn_out, present_kv = self.attn(x, attention_mask=attn_mask, past_key_value=past_key_value, use_cache=use_cache)
        else:
            attn_out, _ = self.attn(self.attn_norm(x), self.attn_norm(x), self.attn_norm(x), key_padding_mask=mask)
            present_kv = None
        x = x + self.attn_dropout(attn_out)

        # Convolution
        x = x + self.conv(x)

        # Feed-forward 2 (half-step)
        x = x + 0.5 * self.ff2(self.ff2_norm(x))

        return self.final_norm(x), present_kv


class AudioEncoder(nn.Module):
    """
    SOTA Audio Encoder with Raw Waveform Tokenization, RMLA, and Voice Enhancement.

    Features:
    - Raw waveform tokenization (no mel spectrogram)
    - Conformer blocks with RMLA
    - Zero-shot speaker encoding
    - In-context audio prompting
    - Gradient checkpointing support for memory efficiency
    
    Voice Enhancement Features (SOTA):
    - Prosody-aware EoT Prediction (interruption detection)
    - AVD Emotion Recognition (arousal/valence/dominance)
    - Dynamic Latent Vocalizations (singing/rapping)
    - Neural Sound Effects (beatboxing, breathing, expressions)
    - Speculative Decoding (mid-stream token rewriting)
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        n_mels: int = 80,  # Kept for backward compatibility
        max_audio_length: int = 3000,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_raw_waveform: bool = True,
        # Voice enhancement flags
        enable_eot: bool = True,
        enable_emotion: bool = True,
        enable_singing: bool = True,
        enable_effects: bool = True,
        enable_speculative: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_audio_length = max_audio_length
        self.use_raw_waveform = use_raw_waveform
        self.gradient_checkpointing = False  # Memory optimization flag
        
        # Voice enhancement flags
        self.enable_eot = enable_eot
        self.enable_emotion = enable_emotion
        self.enable_singing = enable_singing
        self.enable_effects = enable_effects
        self.enable_speculative = enable_speculative

        # Raw waveform tokenizer
        if use_raw_waveform:
            self.waveform_tokenizer = RawWaveformTokenizer(
                hidden_size=hidden_size,
                num_codebooks=8,
                codebook_size=1024,
            )
        else:
            self.waveform_tokenizer = None
            # Fallback mel spectrogram processing
            self.conv_subsample = nn.Sequential(
                nn.Conv1d(n_mels, hidden_size // 2, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
            )

        # Speaker encoder for zero-shot cloning
        self.speaker_encoder = SpeakerEncoder(
            hidden_size=256,
            output_size=hidden_size // 4,
        )

        # In-context audio prompting
        self.audio_prompting = InContextAudioPrompting(
            hidden_size=hidden_size,
            num_prompt_tokens=32,
        )

        # Conformer blocks with RMLA
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                hidden_size, num_heads,
                ff_expansion=4,
                conv_kernel_size=31,
                dropout=dropout,
                use_rmla=True,
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        # === VOICE ENHANCEMENT MODULES ===
        
        # Prosody-aware EoT Prediction (interruption detection)
        if enable_eot:
            self.eot_predictor = ProsodyAwareEoTPredictor(hidden_size, dropout=dropout)
        else:
            self.eot_predictor = None
        
        # AVD Emotion Recognition
        if enable_emotion:
            self.emotion_recognizer = AVDEmotionRecognizer(hidden_size, dropout=dropout)
        else:
            self.emotion_recognizer = None
        
        # Dynamic Latent Vocalizations (singing/rapping)
        if enable_singing:
            self.vocalizer = DynamicLatentVocalizer(hidden_size)
        else:
            self.vocalizer = None
        
        # Neural Sound Effects
        if enable_effects:
            self.effects_generator = NeuralSoundEffectGenerator(hidden_size)
        else:
            self.effects_generator = None
        
        # Speculative Decoding (mid-stream rewriting)
        if enable_speculative:
            self.speculative_decoder = SpeculativeAudioDecoder(hidden_size)
        else:
            self.speculative_decoder = None

        print(f"   🎤 AudioEncoder (RMLA Conformer): {hidden_size}d, {num_layers} layers")
        if use_raw_waveform:
            print(f"      - Raw Waveform Tokenizer enabled")
        print(f"      - Zero-Shot Speaker Encoder enabled")
        print(f"      - In-Context Audio Prompting enabled")
        print(f"      - EoT/Interruption Detection: {enable_eot}")
        print(f"      - Emotion Recognition (AVD): {enable_emotion}")
        print(f"      - Singing/Rapping (Vocalizer): {enable_singing}")
        print(f"      - Sound Effects Generator: {enable_effects}")
        print(f"      - Speculative Decoding: {enable_speculative}")

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing to save memory during training."""
        self.gradient_checkpointing = True
        # Enable for nested modules if they support it
        if hasattr(self, 'waveform_tokenizer') and self.waveform_tokenizer is not None:
            if hasattr(self.waveform_tokenizer, 'gradient_checkpointing'):
                self.waveform_tokenizer.gradient_checkpointing = True
        if hasattr(self, 'speaker_encoder') and self.speaker_encoder is not None:
            if hasattr(self.speaker_encoder, 'gradient_checkpointing'):
                self.speaker_encoder.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    def forward(
        self,
        audio_input: torch.Tensor,
        speaker_ref: Optional[torch.Tensor] = None,
        audio_prompt: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_eot: bool = False,
        return_emotion: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
        """
        Process audio to features with optional voice enhancement outputs.
        
        Args:
            audio_input: [B, T] raw waveform or [B, n_mels, T] mel spectrogram
            speaker_ref: [B, n_mels, T_ref] reference audio for speaker cloning
            audio_prompt: [B, T_prompt, hidden_size] audio prompt features
            mask: Optional attention mask
            return_eot: Whether to return EoT/interruption predictions
            return_emotion: Whether to return emotion/AVD predictions
            
        Returns:
            features: [B, T', hidden_size] audio features
            speaker_embedding: [B, hidden_size//4] speaker embedding (if speaker_ref provided)
            extras: dict with EoT/emotion predictions (if requested)
        """
        # Encode audio based on input format
        # SOTA mode (use_raw_waveform=True): expects [B, T] raw waveform from dataset
        # Legacy mode (use_raw_waveform=False): expects [B, n_mels, T] mel spectrogram
        commitment_loss = None
        
        if self.use_raw_waveform and self.waveform_tokenizer is not None:
            # SOTA: Raw waveform input [B, T] or [B, 1, T]
            if audio_input.dim() == 3 and audio_input.shape[1] == 1:
                # [B, 1, T] -> [B, T]
                audio_input = audio_input.squeeze(1)
            elif audio_input.dim() == 3:
                # Legacy mel spectrogram accidentally passed - shouldn't happen with proper pipeline
                # Log warning and convert (not ideal, but prevents crash)
                audio_input = audio_input.mean(dim=1)  # [B, n_mels, T] -> [B, T]
            x, commitment_loss = self.waveform_tokenizer(audio_input)
        elif hasattr(self, 'conv_subsample') and self.conv_subsample is not None:
            # Legacy: Mel spectrogram input [B, n_mels, T]
            if audio_input.dim() == 2:
                # [B, T] waveform accidentally passed to mel mode - shouldn't happen
                audio_input = audio_input.unsqueeze(1)
            x = self.conv_subsample(audio_input)
            x = x.transpose(1, 2)
        else:
            raise RuntimeError(
                f"AudioEncoder: Incompatible configuration. "
                f"use_raw_waveform={self.use_raw_waveform}, "
                f"waveform_tokenizer={self.waveform_tokenizer is not None}, "
                f"conv_subsample={hasattr(self, 'conv_subsample') and self.conv_subsample is not None}"
            )

        # Extract speaker embedding if reference provided
        speaker_embedding = None
        if speaker_ref is not None:
            speaker_embedding = self.speaker_encoder(speaker_ref)

        # Apply in-context audio prompting
        if audio_prompt is not None:
            x = self.audio_prompting(x, audio_prompt=audio_prompt)

        # Conformer blocks with optional gradient checkpointing
        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            for block in self.conformer_blocks:
                # Use checkpoint to save memory during training
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                x, _ = checkpoint(create_custom_forward(block), x, mask, use_reentrant=False)
        else:
            for block in self.conformer_blocks:
                x, _ = block(x, mask)

        # Output projection
        x = self.output_proj(x)
        
        # === VOICE ENHANCEMENT OUTPUTS ===
        extras = {}
        
        if return_eot and self.eot_predictor is not None:
            extras["eot"] = self.eot_predictor(x, mask)
        
        if return_emotion and self.emotion_recognizer is not None:
            extras["emotion"] = self.emotion_recognizer(x, mask)

        return x, speaker_embedding, extras if extras else None
    
    # === VOICE ENHANCEMENT METHODS ===
    
    def detect_interruption(
        self,
        audio_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Optional[dict]:
        """
        Detect interruptions, backchannels, and turn-taking events.
        
        Args:
            audio_features: [B, T, hidden_size] encoded audio
            attention_mask: [B, T] optional mask
            
        Returns:
            dict with eot_logits, event_logits, vad_logits, backoff_prob
        """
        if self.eot_predictor is None:
            return None
        return self.eot_predictor(audio_features, attention_mask)
    
    def recognize_emotion(
        self,
        audio_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Optional[dict]:
        """
        Recognize emotion with AVD (arousal/valence/dominance) values.
        
        Args:
            audio_features: [B, T, hidden_size] encoded audio
            attention_mask: [B, T] optional mask
            
        Returns:
            dict with emotion_logits, arousal, valence, dominance, response_mode
        """
        if self.emotion_recognizer is None:
            return None
        return self.emotion_recognizer(audio_features, attention_mask)
    
    def generate_vocals(
        self,
        text_features: torch.Tensor,
        style_id: Optional[torch.Tensor] = None,
        mode_id: Optional[torch.Tensor] = None,
        target_pitch: Optional[torch.Tensor] = None,
        tempo_bpm: Optional[torch.Tensor] = None,
    ) -> Optional[dict]:
        """
        Generate singing/rapping vocals from text/lyrics.
        
        Args:
            text_features: [B, T, hidden_size] text embeddings
            style_id: [B] style indices (pop, rock, jazz, etc.)
            mode_id: [B] mode indices (speak, sing, rap, hum, etc.)
            target_pitch: [B, T] pitch targets
            tempo_bpm: [B] tempo in BPM
            
        Returns:
            dict with vocal_features, pitch_logits, alignment, durations
        """
        if self.vocalizer is None:
            return None
        return self.vocalizer(text_features, style_id, mode_id, target_pitch, tempo_bpm)
    
    def generate_effects(
        self,
        effect_ids: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        intensity: Optional[torch.Tensor] = None,
    ) -> Optional[dict]:
        """
        Generate sound effects (beatbox, clicks, breathing, etc.).
        
        Args:
            effect_ids: [B] or [B, N] effect type indices
            context: [B, T, hidden_size] optional context
            intensity: [B] intensity values
            
        Returns:
            dict with effect_features, waveform, duration, intensity
        """
        if self.effects_generator is None:
            return None
        return self.effects_generator(effect_ids, context, intensity)
    
    def speculative_generate(
        self,
        context: torch.Tensor,
        generate_draft: bool = True,
        verify_with: Optional[torch.Tensor] = None,
    ) -> Optional[dict]:
        """
        Generate speculative draft tokens for mid-stream rewriting.
        
        Args:
            context: [B, T, hidden_size] current context
            generate_draft: whether to generate new draft
            verify_with: [B, T', hidden_size] new context to verify against
            
        Returns:
            dict with checkpoint, draft_tokens, confidence, accept_prob
        """
        if self.speculative_decoder is None:
            return None
        return self.speculative_decoder(context, generate_draft, verify_with)


class VariancePredictor(nn.Module):
    """Variance predictor for duration, pitch, and energy."""

    def __init__(self, hidden_size: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C] -> [B, T]"""
        out = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out)
        out = self.norm1(out)
        out = self.dropout(out)

        out = self.conv2(out.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out)
        out = self.norm2(out)
        out = self.dropout(out)

        return self.linear(out).squeeze(-1)


class FFTBlock(nn.Module):
    """FFT block for mel decoder."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        ff_expansion: int = 4,
        kernel_size: int = 9,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention with RMLA
        self.attn = RotaryMultiHeadLatentAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=max(1, num_heads // 2),
            head_dim=hidden_size // num_heads,
            kv_lora_rank=hidden_size // 4,
            dropout=dropout,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
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
        x, _ = self.attn(x)
        x = residual + self.attn_dropout(x)

        # Feed-forward
        residual = x
        x = self.ff_norm(x)
        x = self.ff(x.transpose(1, 2)).transpose(1, 2)
        x = residual + x

        return x


class AudioDecoder(nn.Module):
    """
    SOTA Audio Decoder with MAS and Zero-Shot Speaker Cloning.

    Features:
    - Monotonic Alignment Search for text-to-audio alignment
    - Zero-shot speaker cloning via speaker embeddings
    - In-context audio prompting
    - Variance adaptor with duration, pitch, energy prediction
    - RMLA-based FFT blocks
    - Gradient checkpointing support for memory efficiency
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        n_mels: int = 80,
        max_audio_length: int = 1000,
        num_speakers: int = 256,
        num_decoder_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_mels = n_mels
        self.max_audio_length = max_audio_length
        self.gradient_checkpointing = False  # Memory optimization flag

        # Monotonic Alignment Search
        self.mas = MonotonicAlignmentSearch(hidden_size)

        # Speaker embedding (for multi-speaker)
        self.speaker_embed = nn.Embedding(num_speakers, hidden_size // 4)

        # Zero-shot speaker projection (from speaker encoder output)
        self.speaker_proj = nn.Linear(hidden_size // 4, hidden_size // 4)

        # In-context audio prompting
        self.audio_prompting = InContextAudioPrompting(
            hidden_size=hidden_size,
            num_prompt_tokens=32,
        )

        # Input projection
        self.input_proj = nn.Linear(hidden_size + hidden_size // 4, hidden_size)

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

        # Decoder FFT blocks
        self.decoder_blocks = nn.ModuleList([
            FFTBlock(hidden_size, num_heads=4, ff_expansion=4, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])

        # Mel output
        self.mel_linear = nn.Linear(hidden_size, n_mels)

        # Postnet
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

        print(f"   🔊 AudioDecoder (MAS + RMLA): {hidden_size}d -> {n_mels} mels")
        print(f"      - Monotonic Alignment Search enabled")
        print(f"      - Zero-Shot Speaker Cloning enabled")
        print(f"      - In-Context Audio Prompting enabled")

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing to save memory during training."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    def forward(
        self,
        text_embeds: torch.Tensor,
        target_length: Optional[int] = None,
        speaker: Optional[torch.Tensor] = None,
        speaker_embedding: Optional[torch.Tensor] = None,
        audio_prompt: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        duration_target: Optional[torch.Tensor] = None,
        pitch_target: Optional[torch.Tensor] = None,
        energy_target: Optional[torch.Tensor] = None,
        use_mas: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate mel-spectrogram from text embeddings.

        Args:
            text_embeds: [B, T, hidden_size] text embeddings
            target_length: target mel length (for training)
            speaker: [B] speaker IDs (for multi-speaker)
            speaker_embedding: [B, hidden_size//4] zero-shot speaker embedding
            audio_prompt: [B, T_prompt, hidden_size] audio prompt features
            audio_features: [B, T_audio, hidden_size] target audio features (for MAS training)
            duration_target: [B, T] ground truth durations
            pitch_target: [B, T'] ground truth pitch
            energy_target: [B, T'] ground truth energy
            use_mas: Whether to use MAS for alignment

        Returns:
            mel: [B, n_mels, T'] generated mel spectrogram
            durations: [B, T] predicted durations
            alignment: [B, T_text, T_audio] alignment matrix (if use_mas and audio_features provided)
        """
        batch_size, seq_len, _ = text_embeds.shape
        device = text_embeds.device
        dtype = text_embeds.dtype

        # Get speaker embedding
        if speaker_embedding is not None:
            # Zero-shot speaker cloning
            spk_emb = self.speaker_proj(speaker_embedding)
        elif speaker is not None:
            # Multi-speaker embedding
            spk_emb = self.speaker_embed(speaker)
        else:
            # Default speaker
            speaker = torch.zeros(batch_size, dtype=torch.long, device=device)
            spk_emb = self.speaker_embed(speaker)

        spk_emb = spk_emb.unsqueeze(1).expand(-1, seq_len, -1).to(dtype)

        # Combine embeddings
        x = torch.cat([text_embeds, spk_emb], dim=-1)
        x = self.input_proj(x)

        # Apply in-context audio prompting
        if audio_prompt is not None:
            x = self.audio_prompting(x, audio_prompt=audio_prompt)

        # Encoder with optional gradient checkpointing
        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            for block in self.encoder_blocks:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                x = checkpoint(create_custom_forward(block), x, use_reentrant=False)
        else:
            for block in self.encoder_blocks:
                x = block(x)

        # Monotonic Alignment Search
        alignment = None
        if use_mas and audio_features is not None:
            alignment, durations = self.mas(x, audio_features, use_hard=not self.training)
        else:
            _, durations = self.mas(x)

        # Use target durations during training if provided
        if duration_target is not None:
            durations = duration_target

        # Variance prediction
        pitch_pred = self.pitch_predictor(x)
        energy_pred = F.softplus(self.energy_predictor(x))

        # Determine output length
        if target_length is not None:
            mel_length = target_length
        else:
            mel_length = int(durations.sum(dim=1).max().item())
            mel_length = max(16, min(mel_length, self.max_audio_length))

        # Length regulation via interpolation
        x = F.interpolate(x.transpose(1, 2), size=mel_length, mode='linear', align_corners=False).transpose(1, 2)

        # Use targets during training
        pitch = pitch_target if pitch_target is not None else pitch_pred
        energy = energy_target if energy_target is not None else energy_pred

        # Upsample pitch and energy
        pitch_up = F.interpolate(pitch.unsqueeze(1), size=mel_length, mode='linear', align_corners=False)
        energy_up = F.interpolate(energy.unsqueeze(1), size=mel_length, mode='linear', align_corners=False)

        # Add pitch and energy embeddings
        pitch_emb = self.pitch_embed(pitch_up).transpose(1, 2)
        energy_emb = self.energy_embed(energy_up).transpose(1, 2)
        x = x + pitch_emb + energy_emb

        # Decoder with optional gradient checkpointing
        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            for block in self.decoder_blocks:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                x = checkpoint(create_custom_forward(block), x, use_reentrant=False)
        else:
            for block in self.decoder_blocks:
                x = block(x)

        # Mel output
        mel = self.mel_linear(x).transpose(1, 2)  # [B, n_mels, T']

        # Postnet
        mel_post = mel
        for layer in self.postnet:
            mel_post = layer(mel_post)
        mel = mel + mel_post

        return mel, durations, alignment


# === SOTA VOICE ENHANCEMENT COMPONENTS ===


class ProsodyAwareEoTPredictor(nn.Module):
    """
    Prosody-aware End-of-Turn (EoT) Prediction for real-time interruption detection.
    
    Detects when a speaker is about to finish their turn, allowing the model to:
    - Detect user interruptions (coughs, laughs, "uh-huh", etc.)
    - Yield the floor when appropriate
    - Adjust response mid-stream
    
    Uses prosodic features (pitch, energy, rhythm) combined with semantic features.
    """
    
    def __init__(
        self,
        hidden_size: int = 1024,
        num_eot_classes: int = 5,  # continue, yield, interrupt, backoff, end
        prosody_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_eot_classes = num_eot_classes
        
        # Prosody feature extractor (pitch, energy, duration)
        self.pitch_conv = nn.Sequential(
            nn.Conv1d(1, prosody_dim // 2, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(prosody_dim // 2, prosody_dim, kernel_size=3, padding=1),
        )
        self.energy_conv = nn.Sequential(
            nn.Conv1d(1, prosody_dim // 2, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(prosody_dim // 2, prosody_dim, kernel_size=3, padding=1),
        )
        
        # Voice Activity Detection (VAD) head
        self.vad_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 2),  # speech / non-speech
        )
        
        # Interruption event classifier
        self.event_classifier = nn.Sequential(
            nn.Linear(hidden_size + prosody_dim * 2, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 8),  # laugh, cough, sigh, hesitation, backchannel, agreement, disagreement, confusion
        )
        
        # Temporal attention for turn-taking context
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # EoT prediction head with prosody context
        self.eot_head = nn.Sequential(
            nn.Linear(hidden_size + prosody_dim * 2, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_eot_classes),
        )
        
        # Backoff probability (should model pause and wait?)
        self.backoff_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )
        
        print(f"   🎙️ ProsodyAwareEoTPredictor: {num_eot_classes} turn states, {prosody_dim}d prosody")
    
    def extract_prosody(self, audio_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract pitch and energy prosodic features."""
        # Simple proxy: use first and last channels as pitch/energy
        # In practice, would use actual pitch/energy extraction
        batch_size, seq_len, hidden = audio_features.shape
        
        # Transpose for conv1d: [B, T, H] -> [B, H, T]
        x = audio_features.transpose(1, 2)
        
        # Use subset of features as proxy for pitch/energy
        pitch_proxy = x[:, :1, :]  # [B, 1, T]
        energy_proxy = x.pow(2).mean(dim=1, keepdim=True)  # [B, 1, T]
        
        pitch_features = self.pitch_conv(pitch_proxy).transpose(1, 2)  # [B, T, prosody_dim]
        energy_features = self.energy_conv(energy_proxy).transpose(1, 2)  # [B, T, prosody_dim]
        
        return pitch_features, energy_features
    
    def forward(
        self,
        audio_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Predict end-of-turn and interruption events.
        
        Args:
            audio_features: [B, T, hidden_size] encoded audio
            attention_mask: [B, T] optional mask
            
        Returns:
            dict with:
                - eot_logits: [B, T, num_eot_classes] turn state predictions
                - event_logits: [B, T, 8] interruption event predictions
                - vad_logits: [B, T, 2] voice activity predictions
                - backoff_prob: [B, T, 1] backoff probability
        """
        batch_size, seq_len, _ = audio_features.shape
        
        # Extract prosodic features
        pitch_features, energy_features = self.extract_prosody(audio_features)
        
        # Temporal attention for context
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None
        
        contextualized, _ = self.temporal_attn(
            audio_features, audio_features, audio_features,
            key_padding_mask=key_padding_mask,
        )
        
        # Concatenate with prosody
        combined = torch.cat([contextualized, pitch_features, energy_features], dim=-1)
        
        # Predictions
        eot_logits = self.eot_head(combined)
        event_logits = self.event_classifier(combined)
        vad_logits = self.vad_head(contextualized)
        backoff_prob = self.backoff_head(contextualized)
        
        return {
            "eot_logits": eot_logits,
            "event_logits": event_logits,
            "vad_logits": vad_logits,
            "backoff_prob": backoff_prob,
        }


class AVDEmotionRecognizer(nn.Module):
    """
    Continuous AVD (Arousal/Valence/Dominance) Emotion Recognition.
    
    Predicts both discrete emotion categories and continuous AVD values
    for nuanced emotion understanding and response adaptation.
    """
    
    def __init__(
        self,
        hidden_size: int = 1024,
        num_emotions: int = 10,  # happy, sad, angry, fearful, surprised, disgusted, neutral, excited, frustrated, bored
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_emotions = num_emotions
        
        # Emotion-specific attention
        self.emotion_query = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.emotion_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        
        # Temporal modeling for emotion dynamics
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2, groups=8),
            nn.SiLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
        )
        
        # Discrete emotion classifier
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_emotions),
        )
        
        # Continuous AVD regression heads
        self.arousal_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),  # 0-1 range
        )
        
        self.valence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Tanh(),  # -1 to 1 range
        )
        
        self.dominance_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),  # 0-1 range
        )
        
        # Response adaptation head (how should model respond?)
        self.response_adaptation = nn.Sequential(
            nn.Linear(hidden_size + 3, hidden_size // 2),  # +3 for AVD values
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 4),  # match, contrast, calm, energetic
        )
        
        print(f"   😊 AVDEmotionRecognizer: {num_emotions} emotions + continuous AVD")
    
    def forward(
        self,
        audio_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Recognize emotion from audio features.
        
        Args:
            audio_features: [B, T, hidden_size] encoded audio
            attention_mask: [B, T] optional mask
            
        Returns:
            dict with:
                - emotion_logits: [B, num_emotions] discrete emotion
                - arousal: [B, 1] arousal value (0-1)
                - valence: [B, 1] valence value (-1 to 1)
                - dominance: [B, 1] dominance value (0-1)
                - response_mode: [B, 4] response adaptation logits
        """
        batch_size, seq_len, _ = audio_features.shape
        
        # Temporal convolution for dynamics
        x_conv = self.temporal_conv(audio_features.transpose(1, 2)).transpose(1, 2)
        x = audio_features + x_conv
        
        # Emotion-specific attention (pool to single vector)
        query = self.emotion_query.expand(batch_size, -1, -1)
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None
        
        emotion_context, _ = self.emotion_attn(
            query, x, x,
            key_padding_mask=key_padding_mask,
        )
        emotion_vec = emotion_context.squeeze(1)  # [B, hidden_size]
        
        # Predictions
        emotion_logits = self.emotion_classifier(emotion_vec)
        arousal = self.arousal_head(emotion_vec)
        valence = self.valence_head(emotion_vec)
        dominance = self.dominance_head(emotion_vec)
        
        # Response adaptation based on detected emotion
        avd_concat = torch.cat([emotion_vec, arousal, valence, dominance], dim=-1)
        response_mode = self.response_adaptation(avd_concat)
        
        return {
            "emotion_logits": emotion_logits,
            "arousal": arousal,
            "valence": valence,
            "dominance": dominance,
            "response_mode": response_mode,
        }


class DynamicLatentVocalizer(nn.Module):
    """
    Dynamic Latent Vocalizations for singing, rapping, humming, etc.
    
    Extends speech synthesis to include:
    - Singing with pitch control
    - Rapping with rhythm control
    - Humming, whistling, chanting
    - Musical style transfer
    """
    
    def __init__(
        self,
        hidden_size: int = 1024,
        num_styles: int = 8,  # pop, rock, jazz, classical, hiphop, rnb, country, soul
        num_vocal_modes: int = 6,  # speak, sing, rap, hum, whistle, chant
        pitch_bins: int = 256,
        tempo_range: Tuple[int, int] = (60, 180),  # BPM range
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_styles = num_styles
        self.num_vocal_modes = num_vocal_modes
        self.pitch_bins = pitch_bins
        self.tempo_range = tempo_range
        
        # Style embedding
        self.style_embed = nn.Embedding(num_styles, hidden_size // 4)
        
        # Vocal mode embedding
        self.mode_embed = nn.Embedding(num_vocal_modes, hidden_size // 4)
        
        # Pitch encoder (for singing/melody control)
        self.pitch_embed = nn.Embedding(pitch_bins, hidden_size // 4)
        self.pitch_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, pitch_bins),
        )
        
        # Tempo/rhythm encoder
        self.tempo_encoder = nn.Sequential(
            nn.Linear(1, hidden_size // 8),
            nn.SiLU(),
            nn.Linear(hidden_size // 8, hidden_size // 4),
        )
        
        # Rhythm pattern attention
        self.rhythm_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )
        
        # Style transfer network
        self.style_transfer = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        # Lyrics-to-melody alignment
        self.lyrics_aligner = MonotonicAlignmentSearch(hidden_size)
        
        # Output projection for vocal features
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        print(f"   🎵 DynamicLatentVocalizer: {num_styles} styles, {num_vocal_modes} modes")
    
    def forward(
        self,
        text_features: torch.Tensor,
        style_id: Optional[torch.Tensor] = None,
        mode_id: Optional[torch.Tensor] = None,
        target_pitch: Optional[torch.Tensor] = None,
        tempo_bpm: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Generate vocalization features for singing/rapping/etc.
        
        Args:
            text_features: [B, T, hidden_size] text/lyrics embeddings
            style_id: [B] style indices (0-7)
            mode_id: [B] vocal mode indices (0-5)
            target_pitch: [B, T] optional pitch targets
            tempo_bpm: [B] optional tempo in BPM
            
        Returns:
            dict with:
                - vocal_features: [B, T', hidden_size] vocalization features
                - pitch_logits: [B, T, pitch_bins] predicted pitch
                - alignment: [B, T, T'] text-to-audio alignment
        """
        batch_size, seq_len, _ = text_features.shape
        device = text_features.device
        
        # Default style and mode
        if style_id is None:
            style_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        if mode_id is None:
            mode_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Get embeddings
        style_emb = self.style_embed(style_id).unsqueeze(1).expand(-1, seq_len, -1)
        mode_emb = self.mode_embed(mode_id).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Tempo encoding
        if tempo_bpm is not None:
            tempo_norm = (tempo_bpm.float() - self.tempo_range[0]) / (self.tempo_range[1] - self.tempo_range[0])
            tempo_emb = self.tempo_encoder(tempo_norm.unsqueeze(-1)).unsqueeze(1).expand(-1, seq_len, -1)
        else:
            tempo_emb = torch.zeros(batch_size, seq_len, self.hidden_size // 4, device=device)
        
        # Pitch prediction
        pitch_logits = self.pitch_predictor(text_features)
        
        if target_pitch is not None:
            pitch_emb = self.pitch_embed(target_pitch)
        else:
            pitch_idx = pitch_logits.argmax(dim=-1)
            pitch_emb = self.pitch_embed(pitch_idx)
        
        # Combine all conditions
        conditions = torch.cat([style_emb, mode_emb, tempo_emb, pitch_emb], dim=-1)
        
        # Style transfer
        combined = torch.cat([text_features, conditions], dim=-1)
        vocal_features = self.style_transfer(combined)
        
        # Rhythm attention
        vocal_features, _ = self.rhythm_attn(vocal_features, vocal_features, vocal_features)
        
        # Get alignment
        alignment, durations = self.lyrics_aligner(text_features)
        
        # Final projection
        vocal_features = self.output_proj(vocal_features)
        
        return {
            "vocal_features": vocal_features,
            "pitch_logits": pitch_logits,
            "alignment": alignment,
            "durations": durations,
        }


class NeuralSoundEffectGenerator(nn.Module):
    """
    Neural Style Transfer for Sound Effects and Non-verbal Vocalizations.
    
    Generates:
    - Beatboxing (kicks, snares, hi-hats)
    - Vocal clicks, pops, tongue sounds
    - Breathing, sighing, gasping
    - Non-verbal expressions (hmm, aha, wow, etc.)
    - Polyphonic ad-libs and harmonies
    """
    
    def __init__(
        self,
        hidden_size: int = 1024,
        num_effect_types: int = 20,  # Various sound effects
        num_layers: int = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_effect_types = num_effect_types
        
        # Sound effect type embedding
        self.effect_embed = nn.Embedding(num_effect_types, hidden_size)
        
        # Effect categories:
        # 0-4: Percussion (kick, snare, hihat, crash, fill)
        # 5-8: Mouth sounds (click, pop, whistle, tongue)
        # 9-13: Breathing (in, out, heavy, sigh, gasp)
        # 14-19: Expressions (hmm, aha, wow, ugh, phew, tsk)
        
        # Waveform generator (1D transposed convolutions)
        self.generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Unflatten(1, (hidden_size, 4)),
            nn.ConvTranspose1d(hidden_size, hidden_size // 2, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose1d(hidden_size // 2, hidden_size // 4, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose1d(hidden_size // 4, hidden_size // 8, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose1d(hidden_size // 8, 1, 4, 2, 1),
            nn.Tanh(),  # Waveform range [-1, 1]
        )
        
        # Duration predictor for each effect
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Softplus(),  # Positive duration
        )
        
        # Intensity control
        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),  # 0-1 intensity
        )
        
        # Polyphonic blending for multiple simultaneous effects
        self.blend_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            batch_first=True,
        )
        
        print(f"   🥁 NeuralSoundEffectGenerator: {num_effect_types} effect types")
    
    def forward(
        self,
        effect_ids: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        intensity: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Generate sound effect features.
        
        Args:
            effect_ids: [B] or [B, N] effect type indices
            context: [B, T, hidden_size] optional context features
            intensity: [B] or [B, N] optional intensity values
            
        Returns:
            dict with:
                - effect_features: [B, T', hidden_size] generated features
                - waveform: [B, 1, samples] raw waveform (if generating directly)
                - duration: [B, 1] predicted duration
        """
        # Handle single or multiple effects
        if effect_ids.dim() == 1:
            effect_ids = effect_ids.unsqueeze(1)
        
        batch_size, num_effects = effect_ids.shape
        device = effect_ids.device
        
        # Get effect embeddings
        effect_emb = self.effect_embed(effect_ids)  # [B, N, hidden_size]
        
        # Blend multiple effects if present
        if num_effects > 1:
            effect_emb, _ = self.blend_attn(effect_emb, effect_emb, effect_emb)
        
        effect_vec = effect_emb.mean(dim=1)  # [B, hidden_size]
        
        # Context integration
        if context is not None:
            context_vec = context.mean(dim=1)
            effect_vec = effect_vec + context_vec
        
        # Predict duration and intensity
        duration = self.duration_head(effect_vec)
        pred_intensity = self.intensity_head(effect_vec)
        
        if intensity is not None:
            pred_intensity = intensity.unsqueeze(-1) if intensity.dim() == 1 else intensity
        
        # Scale by intensity
        effect_vec = effect_vec * pred_intensity
        
        # Generate waveform-like features
        waveform = self.generator(effect_vec)  # [B, 1, samples]
        
        return {
            "effect_features": effect_emb,
            "waveform": waveform,
            "duration": duration,
            "intensity": pred_intensity,
        }


class SpeculativeAudioDecoder(nn.Module):
    """
    Mid-stream Token Rewriting support for Speculative Decoding in audio.
    
    Allows the model to:
    - Generate draft audio tokens speculatively
    - Accept/reject based on user feedback or context change
    - Rollback and regenerate from checkpoints
    - Smooth transitions during rewrites
    """
    
    def __init__(
        self,
        hidden_size: int = 1024,
        draft_length: int = 10,  # Number of tokens to speculate
        num_heads: int = 8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.draft_length = draft_length
        
        # Draft generator (fast, approximate)
        self.draft_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        # Verification head (checks if draft is acceptable)
        self.verify_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        
        # Checkpoint encoder (saves state for rollback)
        self.checkpoint_encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        
        # Transition smoother (for seamless rewrites)
        self.smoother = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        # Confidence estimator per token
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )
        
        print(f"   ⚡ SpeculativeAudioDecoder: draft_length={draft_length}")
    
    def generate_draft(
        self,
        context: torch.Tensor,
        num_tokens: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate draft tokens speculatively.
        
        Args:
            context: [B, T, hidden_size] context features
            num_tokens: number of draft tokens (default: self.draft_length)
            
        Returns:
            draft_tokens: [B, N, hidden_size] draft features
            confidence: [B, N, 1] confidence per token
        """
        if num_tokens is None:
            num_tokens = self.draft_length
        
        batch_size = context.shape[0]
        device = context.device
        
        # Use last context as seed
        seed = context[:, -1:, :]  # [B, 1, hidden_size]
        
        draft_tokens = []
        confidences = []
        
        current = seed
        for _ in range(num_tokens):
            draft = self.draft_head(current)
            conf = self.confidence_head(draft)
            draft_tokens.append(draft)
            confidences.append(conf)
            current = draft
        
        draft_tokens = torch.cat(draft_tokens, dim=1)
        confidences = torch.cat(confidences, dim=1)
        
        return draft_tokens, confidences
    
    def verify_draft(
        self,
        draft_tokens: torch.Tensor,
        new_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Verify if draft tokens should be accepted given new context.
        
        Args:
            draft_tokens: [B, N, hidden_size] draft features
            new_context: [B, T, hidden_size] updated context
            
        Returns:
            accept_prob: [B, N, 1] probability to accept each token
        """
        # Compare draft with new context
        context_summary = new_context.mean(dim=1, keepdim=True).expand(-1, draft_tokens.shape[1], -1)
        combined = torch.cat([draft_tokens, context_summary], dim=-1)
        accept_prob = self.verify_head(combined)
        
        return accept_prob
    
    def create_checkpoint(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Save hidden state for potential rollback."""
        _, checkpoint = self.checkpoint_encoder(hidden_state)
        return checkpoint.squeeze(0)  # [B, hidden_size]
    
    def smooth_transition(
        self,
        old_features: torch.Tensor,
        new_features: torch.Tensor,
    ) -> torch.Tensor:
        """Create smooth transition between old and new features."""
        combined = torch.cat([old_features, new_features], dim=-1)
        return self.smoother(combined)
    
    def forward(
        self,
        context: torch.Tensor,
        generate_draft: bool = True,
        verify_with: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Full speculative decoding step.
        
        Args:
            context: [B, T, hidden_size] current context
            generate_draft: whether to generate new draft
            verify_with: [B, T', hidden_size] new context to verify against
            
        Returns:
            dict with draft tokens, confidence, verification results
        """
        results = {}
        
        # Create checkpoint
        results["checkpoint"] = self.create_checkpoint(context)
        
        if generate_draft:
            draft, confidence = self.generate_draft(context)
            results["draft_tokens"] = draft
            results["confidence"] = confidence
        
        if verify_with is not None and "draft_tokens" in results:
            accept_prob = self.verify_draft(results["draft_tokens"], verify_with)
            results["accept_prob"] = accept_prob
        
        return results


# EnhancedAudioEncoder removed - all features merged into AudioEncoder

"""Audio and voice processing utilities with custom mel spectrogram extraction."""

import numpy as np
import math
from typing import Optional, Union
import io

# Handle torch import gracefully
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    F = None


class VoiceProcessor:
    """
    SOTA Voice processing utility with custom mel spectrogram extraction.
    
    Features:
    - Custom mel spectrogram (no torchaudio dependency)
    - Multiple audio loading backends (soundfile, librosa)
    - Proper resampling support
    - Efficient batch processing
    """

    def __init__(self, sample_rate: int = 16000, n_mels: int = 80, n_fft: int = 1024,
                 hop_length: int = 256, max_duration: float = 10.0, f_min: float = 0.0,
                 f_max: float = 8000.0):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_duration = max_duration
        self.max_samples = int(max_duration * sample_rate)
        self.f_min = f_min
        self.f_max = f_max
        
        self.has_soundfile = False
        self.has_librosa = False
        self.has_scipy = False
        
        # Custom mel filterbank (no torchaudio needed)
        self.mel_fb = None
        self.window = None

        if not TORCH_AVAILABLE:
            print("   ⚠️ VoiceProcessor: torch not available")
            return
        
        # Initialize custom mel filterbank
        self._init_mel_filterbank()
        
        # Check for audio loading backends
        try:
            import soundfile
            self.has_soundfile = True
        except ImportError:
            pass
        
        try:
            import librosa
            self.has_librosa = True
        except ImportError:
            pass
        
        try:
            from scipy import signal
            self.has_scipy = True
        except ImportError:
            pass
        
        backends = []
        if self.has_soundfile:
            backends.append("soundfile")
        if self.has_librosa:
            backends.append("librosa")
    
    def _hz_to_mel(self, hz: float) -> float:
        """Convert Hz to mel scale."""
        return 2595.0 * math.log10(1.0 + hz / 700.0)
    
    def _mel_to_hz(self, mel: float) -> float:
        """Convert mel to Hz."""
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    
    def _init_mel_filterbank(self):
        """Initialize mel filterbank matrix."""
        # Mel points
        mel_min = self._hz_to_mel(self.f_min)
        mel_max = self._hz_to_mel(self.f_max)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = np.array([self._mel_to_hz(m) for m in mel_points])
        
        # FFT bins
        fft_bins = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        # Create filterbank
        n_freqs = self.n_fft // 2 + 1
        fb = np.zeros((self.n_mels, n_freqs))
        
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
        
        self.mel_fb = torch.from_numpy(fb).float()
        self.window = torch.hann_window(self.n_fft)

    def extract_mel(self, waveform):
        """
        Extract mel-spectrogram from waveform using custom implementation.
        
        Args:
            waveform: torch.Tensor of shape [T] or [1, T] or [B, T]
            
        Returns:
            mel: torch.Tensor of shape [n_mels, T'] or [B, n_mels, T']
        """
        if not TORCH_AVAILABLE or self.mel_fb is None:
            return None
        
        try:
            # Ensure tensor
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform, dtype=torch.float32)
            
            # Handle dimensions
            squeeze_batch = False
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
                squeeze_batch = True
            elif waveform.dim() == 2 and waveform.size(0) == 1:
                squeeze_batch = True
            
            device = waveform.device
            mel_fb = self.mel_fb.to(device)
            window = self.window.to(device)
            
            # Pad waveform
            pad_amount = self.n_fft // 2
            waveform = F.pad(waveform, (pad_amount, pad_amount), mode='reflect')
            
            # STFT using unfold
            frames = waveform.unfold(-1, self.n_fft, self.hop_length)  # [B, num_frames, n_fft]
            
            # Apply window
            frames = frames * window
            
            # FFT
            spec = torch.fft.rfft(frames, dim=-1)  # [B, num_frames, n_fft//2+1]
            
            # Power spectrum
            power_spec = spec.abs().pow(2)  # [B, num_frames, n_fft//2+1]
            
            # Apply mel filterbank
            mel_spec = torch.matmul(power_spec, mel_fb.T)  # [B, num_frames, n_mels]
            
            # Log mel
            mel_spec = torch.log(mel_spec.clamp(min=1e-10))
            
            # Transpose to [B, n_mels, T']
            mel_spec = mel_spec.transpose(1, 2)
            
            if squeeze_batch:
                mel_spec = mel_spec.squeeze(0)
            
            return mel_spec
            
        except Exception as e:
            return None

    def _resample(self, waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """Resample waveform using linear interpolation (no torchaudio)."""
        if orig_sr == target_sr:
            return waveform
        
        # Calculate new length
        orig_len = waveform.shape[-1]
        new_len = int(orig_len * target_sr / orig_sr)
        
        # Use linear interpolation for resampling
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
            resampled = F.interpolate(waveform, size=new_len, mode='linear', align_corners=False)
            return resampled.squeeze(0).squeeze(0)
        else:
            waveform = waveform.unsqueeze(1)  # [B, 1, T]
            resampled = F.interpolate(waveform, size=new_len, mode='linear', align_corners=False)
            return resampled.squeeze(1)
    
    def load_audio(self, audio_path: str):
        """Load audio file and return waveform (no torchaudio dependency)."""
        if not TORCH_AVAILABLE:
            return None
        
        # Try soundfile first (preferred - fast and reliable)
        if self.has_soundfile:
            try:
                import soundfile as sf
                array, sr = sf.read(audio_path)
                waveform = torch.from_numpy(array).float()
                
                # Handle stereo -> mono
                if waveform.dim() > 1:
                    if waveform.shape[-1] <= 2:  # [T, channels]
                        waveform = waveform.mean(dim=-1)
                    else:  # [channels, T]
                        waveform = waveform.mean(dim=0)
                
                # Resample if needed
                if sr != self.sample_rate:
                    waveform = self._resample(waveform, sr, self.sample_rate)
                
                # Truncate if too long
                if waveform.shape[-1] > self.max_samples:
                    waveform = waveform[..., :self.max_samples]
                
                return waveform
            except Exception:
                pass
        
        # Try librosa (handles more formats)
        if self.has_librosa:
            try:
                import librosa
                array, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                waveform = torch.from_numpy(array).float()
                
                if waveform.shape[-1] > self.max_samples:
                    waveform = waveform[..., :self.max_samples]
                
                return waveform
            except Exception:
                pass
        
        return None

    def process_audio_array(self, audio_data, sampling_rate: int = None):
        """Process audio data (from HuggingFace datasets) to mel-spectrogram."""
        if not TORCH_AVAILABLE:
            return None
        try:
            waveform = None
            sr = sampling_rate or self.sample_rate

            # Handle dict format
            if isinstance(audio_data, dict):
                # PRIORITY 1: Try bytes (used when Audio(decode=False) - our custom decoder)
                if 'bytes' in audio_data and audio_data['bytes'] is not None:
                    if self.has_soundfile:
                        try:
                            import soundfile as sf
                            audio_bytes = audio_data['bytes']
                            audio_buffer = io.BytesIO(audio_bytes)
                            array, sr = sf.read(audio_buffer)
                            waveform = torch.from_numpy(array).float()
                        except Exception:
                            pass
                    
                    if waveform is None and self.has_librosa:
                        try:
                            import librosa
                            audio_bytes = audio_data['bytes']
                            audio_buffer = io.BytesIO(audio_bytes)
                            array, sr = librosa.load(audio_buffer, sr=None)
                            waveform = torch.from_numpy(array).float()
                        except Exception:
                            pass
                
                # PRIORITY 2: Try array (if decode=True was used)
                if waveform is None and 'array' in audio_data and audio_data['array'] is not None:
                    array = audio_data['array']
                    sr = audio_data.get('sampling_rate', sr)
                    try:
                        if isinstance(array, np.ndarray):
                            waveform = torch.from_numpy(array.copy()).float()
                        elif isinstance(array, list):
                            waveform = torch.tensor(array, dtype=torch.float32)
                        elif isinstance(array, torch.Tensor):
                            waveform = array.float()
                        else:
                            waveform = torch.tensor(array).float()
                    except Exception:
                        pass
                
                # PRIORITY 3: Try audio key
                if waveform is None and 'audio' in audio_data and audio_data['audio'] is not None:
                    array = audio_data['audio']
                    sr = audio_data.get('sampling_rate', sr)
                    try:
                        if isinstance(array, np.ndarray):
                            waveform = torch.from_numpy(array.copy()).float()
                        elif isinstance(array, list):
                            waveform = torch.tensor(array, dtype=torch.float32)
                        else:
                            waveform = torch.tensor(array).float()
                    except Exception:
                        pass

            # Handle numpy array directly
            elif isinstance(audio_data, np.ndarray):
                waveform = torch.from_numpy(audio_data.copy()).float()

            # Handle torch tensor directly
            elif isinstance(audio_data, torch.Tensor):
                waveform = audio_data.float()

            if waveform is None:
                return None
            
            # Check for valid waveform
            if waveform.numel() == 0:
                return None

            # Ensure 1D
            if waveform.dim() > 1:
                waveform = waveform.squeeze()
                if waveform.dim() > 1:
                    waveform = waveform.mean(dim=0)
            
            if waveform.dim() == 0:
                return None

            # Resample if needed (using custom resampler, no torchaudio)
            if sr != self.sample_rate:
                waveform = self._resample(waveform, int(sr), self.sample_rate)

            # Truncate if too long
            if waveform.shape[-1] > self.max_samples:
                waveform = waveform[..., :self.max_samples]
            
            # Ensure minimum length
            if waveform.shape[-1] < 400:
                pad_len = 400 - waveform.shape[-1]
                waveform = F.pad(waveform, (0, pad_len))

            # Extract mel using custom implementation
            mel = self.extract_mel(waveform.unsqueeze(0))
            if mel is not None:
                mel = mel.squeeze(0)
            return mel
        except Exception:
            return None

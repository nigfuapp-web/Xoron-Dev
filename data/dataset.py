"""Streaming dataset for multimodal training."""

import gc
import json
import logging
import os
import random
import time
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from typing import Dict, List, Any, Optional, Callable, Iterator
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class TrueStreamingDataset(IterableDataset):
    """
    True streaming dataset with proper iterator-based data loading.
    
    Key features for memory efficiency:
    - Uses IterableDataset for true streaming (no data stored in memory)
    - Round-robin iteration across all dataset sources
    - On-demand processing: Media (images/video/audio) is processed only when accessed
    - Streams until max_per_epoch samples are yielded per epoch
    - Works with both HuggingFace streaming datasets and local JSONL files
    - Samples are formatted using format_functions before being returned
    - Supports resuming from saved streaming state (skip already-seen samples)
    """

    def __init__(
        self,
        dataset_configs: Dict[str, List[Dict]],
        format_functions: Dict[str, Callable],
        tokenizer,
        tokens: Dict[str, str],
        image_processor,
        max_length: int = 1024,  # From TrainingConfig.max_seq_length
        max_per_epoch: int = 2000,  # From TrainingConfig.max_per_epoch
        max_per_dataset: int = 100,  # From TrainingConfig.max_per_dataset
        sample_repeat: int = 2,  # From TrainingConfig.sample_repeat
        voice_processor=None,
        max_video_frames: int = 32,  # From XoronConfig.video_max_frames (multi-scale)
        video_size: int = 256,  # From XoronConfig.video_base_size (multi-scale)
        image_size: int = 256,  # From XoronConfig.image_base_size (multi-scale)
        audio_n_mels: int = 80,  # From XoronConfig.audio_n_mels
        audio_max_length: int = 1000,  # From XoronConfig.audio_max_length
        audio_sample_rate: int = 16000,  # From XoronConfig.audio_sample_rate
        resume_state_path: str = None,
        use_raw_waveform: bool = True,  # From XoronConfig.use_raw_waveform
        modality_max_values: Dict[str, int] = None,  # Per-modality max samples for dual training
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_per_epoch = max_per_epoch
        self.max_per_dataset = max_per_dataset
        self.sample_repeat = max(1, sample_repeat)  # At least 1
        self.tokens = tokens
        self.format_functions = format_functions
        self.dataset_configs = dataset_configs
        self.image_processor = image_processor
        self.voice_processor = voice_processor
        self.max_video_frames = max_video_frames
        self.video_size = video_size
        self.image_size = image_size
        self.audio_n_mels = audio_n_mels
        self.audio_max_length = audio_max_length
        self.audio_sample_rate = audio_sample_rate
        self.use_raw_waveform = use_raw_waveform
        
        # Per-modality max samples for dual/multi training mode
        # e.g., {'text': 3300, 'video': 1000} means text stops at 3300, video at 1000
        self.modality_max_values = modality_max_values or {}
        self.is_dual_training = bool(modality_max_values)

        self.total_datasets = sum(len(configs) for configs in dataset_configs.values() if configs)
        
        # Cache for Vript video metadata (video_id -> URL mapping)
        self._vript_meta_cache = None

        # Dataset sources list - will be populated in _init_iterators
        self._dataset_sources = []
        
        # Streaming state for resume support - organized by modality for proper tracking
        self._streaming_state = {
            "epoch": 0,
            "unique_samples": 0,
            "total_yields": 0,
            "dataset_positions": {},  # dataset_name -> samples_consumed
            "modality_positions": {   # Per-modality tracking for --text, --image, --video, --voice flags
                "text": {},           # text dataset positions
                "image": {},          # image dataset positions
                "video": {},          # video dataset positions
                "voice": {},          # voice/audio dataset positions
            },
            "modality_counts": {      # Per-modality sample counts for dual training
                "text": 0,
                "image": 0,
                "video": 0,
                "audio": 0,
            },
            "last_modality": None,    # Which modality was last trained
        }
        self._state_save_path = None
        
        # Load resume state if provided
        if resume_state_path and os.path.exists(resume_state_path):
            self._load_streaming_state(resume_state_path)
        
        # Initialize dataset sources
        self._init_iterators()

    def _init_iterators(self):
        """Initialize dataset sources for streaming."""
        from datasets import load_dataset
        
        # Import Audio for casting audio columns with decode=False
        # This disables HF's default decoder so we use our custom soundfile/librosa decoder
        try:
            from datasets import Audio
            has_audio_feature = True
        except ImportError:
            has_audio_feature = False

        # Audio dataset types that need decode=False
        audio_dtypes = {'voice_asr', 'voice_tts'}

        self._dataset_sources = []
        failed_datasets = []

        for dtype, configs in self.dataset_configs.items():
            if not configs:
                continue
            for cfg in configs:
                max_retries = 3
                retry_delay = 2
                ds = None
                
                # Handle local JSONL files - create streaming generator
                if cfg.get("local", False):
                    local_path = self._get_local_path(cfg)
                    if local_path:
                        self._dataset_sources.append({
                            "dtype": dtype,
                            "name": cfg["name"],
                            "config": cfg,
                            "is_local": True,
                            "local_path": local_path,
                            "hf_dataset": None,
                        })
                    else:
                        failed_datasets.append(cfg['name'])
                    continue
                
                # Handle remote HuggingFace datasets
                for attempt in range(max_retries):
                    try:
                        load_kwargs = {
                            "path": cfg["path"],
                            "split": cfg["split"],
                            "streaming": True,  # Always use streaming for HF datasets
                        }
                        if "config" in cfg:
                            load_kwargs["name"] = cfg["config"]
                        ds = load_dataset(**load_kwargs)
                        
                        # For audio datasets, disable HF's automatic decoding
                        # We use our custom decoder (soundfile/librosa) instead
                        if dtype in audio_dtypes and has_audio_feature:
                            try:
                                ds = ds.cast_column('audio', Audio(decode=False))
                            except Exception:
                                pass  # Column might not exist
                        
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            retry_delay *= 2
                        else:
                            failed_datasets.append(cfg['name'])
                            ds = None

                if ds is not None:
                    self._dataset_sources.append({
                        "dtype": dtype,
                        "name": cfg["name"],
                        "config": cfg,
                        "is_local": False,
                        "local_path": None,
                        "hf_dataset": ds,
                    })
        
        # Print concise summary
        print(f"   ✅ {len(self._dataset_sources)} datasets initialized", flush=True)
        if failed_datasets:
            print(f"   ⚠️ {len(failed_datasets)} failed: {', '.join(failed_datasets[:3])}{'...' if len(failed_datasets) > 3 else ''}")
    
    def _load_streaming_state(self, path: str):
        """Load streaming state from JSON file for resuming."""
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            
            # Merge loaded state with default structure (handles old format gracefully)
            self._streaming_state["epoch"] = state.get("epoch", 0)
            self._streaming_state["unique_samples"] = state.get("unique_samples", 0)
            self._streaming_state["total_yields"] = state.get("total_yields", 0)
            self._streaming_state["dataset_positions"] = state.get("dataset_positions", {})
            self._streaming_state["last_modality"] = state.get("last_modality", None)
            
            # Load per-modality positions if available
            if "modality_positions" in state:
                for modality in ["text", "image", "video", "voice"]:
                    if modality in state["modality_positions"]:
                        self._streaming_state["modality_positions"][modality] = state["modality_positions"][modality]
            
            print(f"   📂 Resumed from epoch {state.get('epoch', 0)}, {state.get('unique_samples', 0)} samples seen")
        except Exception as e:
            print(f"   ⚠️ Could not load streaming state: {e}")
    
    def save_streaming_state(self, path: str):
        """Save current streaming state to JSON file for resuming later."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump(self._streaming_state, f, indent=2)
            return True
        except Exception as e:
            print(f"   ⚠️ Could not save streaming state: {e}")
            return False
    
    def get_streaming_state(self) -> dict:
        """Get current streaming state for external saving."""
        return self._streaming_state.copy()
    
    def set_state_save_path(self, path: str):
        """Set path for auto-saving streaming state during iteration."""
        self._state_save_path = path
    
    def _skip_to_position(self, iterator, dataset_name: str, is_local: bool, local_path: str = None):
        """Skip iterator to saved position for resuming."""
        skip_count = self._streaming_state.get("dataset_positions", {}).get(dataset_name, 0)
        
        if skip_count == 0:
            return iterator, 0
        
        print(f"   ⏩ Skipping {skip_count} samples in {dataset_name}...", end="", flush=True)
        skipped = 0
        
        if is_local and local_path:
            # For local JSONL, create new iterator starting from line number
            return self._create_local_iterator_from_line(local_path, skip_count), skip_count
        else:
            # For HF streaming, skip samples (this iterates through them)
            try:
                for _ in range(skip_count):
                    next(iterator)
                    skipped += 1
            except StopIteration:
                pass
        
        print(f" done ({skipped} skipped)", flush=True)
        return iterator, skipped
    
    def _create_local_iterator_from_line(self, path: str, start_line: int):
        """Create a streaming iterator for a local JSONL file starting from a specific line."""
        with open(path, 'r', encoding='utf-8') as f:
            # Skip to start line
            for _ in range(start_line):
                f.readline()
            # Yield remaining lines
            for line in f:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    
    def _get_local_path(self, cfg):
        """Get the full path for a local dataset."""
        path = cfg.get("path", "")
        
        if os.path.isabs(path):
            return path if os.path.exists(path) else None
        
        if os.path.exists(path):
            return path
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(project_root, path)
        return full_path if os.path.exists(full_path) else None
    
    def _create_local_iterator(self, path):
        """Create a streaming iterator for a local JSONL file."""
        import json
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    def _process_image(self, image_data) -> Optional[torch.Tensor]:
        """Process image data to tensor."""
        try:
            if image_data is None:
                return None

            if isinstance(image_data, torch.Tensor):
                if image_data.dim() == 3 and image_data.shape[0] == 3:
                    return image_data
                return None

            if isinstance(image_data, Image.Image):
                img = image_data
            elif isinstance(image_data, dict):
                if 'bytes' in image_data and image_data['bytes']:
                    from io import BytesIO
                    img = Image.open(BytesIO(image_data['bytes']))
                elif 'path' in image_data and image_data['path']:
                    img = Image.open(image_data['path'])
                else:
                    return None
            elif isinstance(image_data, str):
                if image_data.startswith('http'):
                    import requests
                    from io import BytesIO
                    response = requests.get(image_data, timeout=5)
                    img = Image.open(BytesIO(response.content))
                else:
                    img = Image.open(image_data)
            elif isinstance(image_data, np.ndarray):
                img = Image.fromarray(image_data)
            else:
                return None

            img = img.convert('RGB')

            if self.image_processor:
                processed = self.image_processor(img, return_tensors="pt")
                tensor = processed['pixel_values'].squeeze(0)
            else:
                # Fallback: use image_size from config for SigLIP compatibility with proper normalization
                # SigLIP uses ImageNet mean/std normalization
                img = img.resize((self.image_size, self.image_size))
                tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                # Apply ImageNet normalization (used by SigLIP/CLIP)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                tensor = (tensor - mean) / std
            
            # Ensure no inf/nan values that cause training instability
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=10.0, neginf=-10.0)
            tensor = torch.clamp(tensor, min=-10.0, max=10.0)
            return tensor

        except Exception:
            return None

    def _download_video_from_url(self, url: str, timeout: int = 30) -> Optional[str]:
        """Download video from URL to a temporary file."""
        temp_path = None
        try:
            import tempfile
            
            # Check if it's a YouTube or TikTok URL - use yt-dlp
            if 'youtube.com' in url or 'youtu.be' in url or 'tiktok.com' in url:
                return self._download_youtube_video(url)
            
            import requests
            response = requests.get(url, timeout=timeout, stream=True)
            if response.status_code != 200:
                return None
            
            # Determine file extension from URL or content-type
            content_type = response.headers.get('content-type', '')
            if 'gif' in url.lower() or 'gif' in content_type:
                ext = '.gif'
            elif 'mp4' in url.lower() or 'mp4' in content_type:
                ext = '.mp4'
            elif 'webm' in url.lower() or 'webm' in content_type:
                ext = '.webm'
            elif 'avi' in url.lower() or 'avi' in content_type:
                ext = '.avi'
            else:
                ext = '.mp4'
            
            # Create temp file
            fd, temp_path = tempfile.mkstemp(suffix=ext)
            with os.fdopen(fd, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return temp_path
        except Exception:
            # Clean up temp file on failure
            if temp_path:
                try:
                    os.remove(temp_path)
                except:
                    pass
            return None

    def _download_youtube_video(self, url: str) -> Optional[str]:
        """Download YouTube video using yt-dlp."""
        try:
            import tempfile
            import subprocess
            import shutil
            
            if not shutil.which('yt-dlp'):
                return None
            
            fd, temp_path = tempfile.mkstemp(suffix='.mp4')
            os.close(fd)
            
            cmd = [
                'yt-dlp',
                '-f', 'best[height<=480][ext=mp4]/best[height<=480]/bestvideo[height<=480]+bestaudio/best',
                '-o', temp_path,
                '--no-playlist',
                '--quiet',
                '--no-warnings',
                '--socket-timeout', '30',
                '--retries', '3',
                '--fragment-retries', '3',
                '--extractor-retries', '3',
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=180)
            
            if result.returncode == 0 and os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                return temp_path
            else:
                try:
                    os.remove(temp_path)
                except:
                    pass
                return None
        except:
            return None

    def _extract_frames_from_video(self, video_path: str) -> List[torch.Tensor]:
        """Extract frames from a video file (mp4, webm, avi) or animated GIF."""
        frames = []
        try:
            # Handle GIF files with PIL (cv2 doesn't handle animated GIFs well)
            if video_path.lower().endswith('.gif'):
                gif = Image.open(video_path)
                gif_frames = []
                try:
                    while True:
                        gif_frames.append(gif.copy().convert('RGB'))
                        gif.seek(gif.tell() + 1)
                except EOFError:
                    pass
                
                if len(gif_frames) > 0:
                    # Sample frames evenly
                    indices = np.linspace(0, len(gif_frames) - 1, min(self.max_video_frames, len(gif_frames)), dtype=int)
                    for idx in indices:
                        processed = self._process_image(gif_frames[idx])
                        if processed is not None:
                            frames.append(processed)
                return frames
            
            # Handle video files with cv2
            import cv2
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames > 0:
                indices = np.linspace(0, total_frames - 1, self.max_video_frames, dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_pil = Image.fromarray(frame_rgb)
                        processed = self._process_image(frame_pil)
                        if processed is not None:
                            frames.append(processed)
            cap.release()
        except Exception:
            pass
        return frames

    def _process_video_frames(self, video_data, sample: dict) -> Optional[torch.Tensor]:
        """Process video data to frame tensors."""
        import os as _os
        try:
            frames = []
            temp_video_path = None

            # Handle URL - download video first
            if isinstance(video_data, str) and video_data.startswith('http'):
                temp_video_path = self._download_video_from_url(video_data)
                if temp_video_path:
                    frames = self._extract_frames_from_video(temp_video_path)
                    # Clean up temp file
                    try:
                        _os.remove(temp_video_path)
                    except:
                        pass

            # Handle direct frames (list of images)
            elif isinstance(video_data, (list, tuple)):
                for frame in video_data[:self.max_video_frames]:
                    processed = self._process_image(frame)
                    if processed is not None:
                        frames.append(processed)

            # Handle video file path
            elif isinstance(video_data, str):
                frames = self._extract_frames_from_video(video_data)

            # Handle dict with bytes
            elif isinstance(video_data, dict):
                if 'bytes' in video_data and video_data['bytes']:
                    import tempfile
                    fd, temp_path = tempfile.mkstemp(suffix='.mp4')
                    try:
                        with _os.fdopen(fd, 'wb') as f:
                            f.write(video_data['bytes'])
                        frames = self._extract_frames_from_video(temp_path)
                    finally:
                        try:
                            _os.remove(temp_path)
                        except:
                            pass
                elif 'path' in video_data and video_data['path']:
                    frames = self._extract_frames_from_video(video_data['path'])

            if len(frames) == 0:
                return None

            # Pad to max_video_frames
            while len(frames) < self.max_video_frames:
                frames.append(frames[-1].clone())

            frames = frames[:self.max_video_frames]

            # Resize frames and ensure valid values (no inf/nan)
            frame_tensors = []
            for f in frames:
                if f.shape[1] != self.video_size or f.shape[2] != self.video_size:
                    f = F.interpolate(f.unsqueeze(0), size=(self.video_size, self.video_size), mode='bilinear', align_corners=False).squeeze(0)
                # Clamp to prevent inf/nan values that cause training instability
                f = torch.clamp(f, min=-10.0, max=10.0)
                # Replace any nan with 0
                f = torch.nan_to_num(f, nan=0.0, posinf=10.0, neginf=-10.0)
                frame_tensors.append(f)

            return torch.stack(frame_tensors)

        except Exception:
            return None

    def _download_audio_from_url(self, url: str, timeout: int = 30) -> Optional[str]:
        """Download audio from URL to a temporary file."""
        temp_path = None
        try:
            import tempfile
            import requests
            
            response = requests.get(url, timeout=timeout, stream=True)
            if response.status_code != 200:
                return None
            
            # Determine file extension from URL or content-type
            content_type = response.headers.get('content-type', '')
            if 'wav' in url.lower() or 'wav' in content_type:
                ext = '.wav'
            elif 'mp3' in url.lower() or 'mp3' in content_type or 'mpeg' in content_type:
                ext = '.mp3'
            elif 'flac' in url.lower() or 'flac' in content_type:
                ext = '.flac'
            elif 'ogg' in url.lower() or 'ogg' in content_type:
                ext = '.ogg'
            else:
                ext = '.wav'
            
            fd, temp_path = tempfile.mkstemp(suffix=ext)
            with os.fdopen(fd, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return temp_path
        except Exception:
            # Clean up temp file on failure
            if temp_path:
                try:
                    os.remove(temp_path)
                except:
                    pass
            return None

    def _process_audio(self, audio_data, sample: dict) -> Optional[torch.Tensor]:
        """Process audio data to raw waveform (SOTA) or mel-spectrogram (legacy).
        
        When use_raw_waveform=True (default, SOTA):
            Returns: [T] raw waveform tensor for RawWaveformTokenizer
        When use_raw_waveform=False (legacy):
            Returns: [n_mels, T] mel spectrogram for conv_subsample
        """
        if self.voice_processor is None:
            return None

        import os as _os
        
        # If audio_data is None, try to get it from sample directly
        if audio_data is None:
            return None
        
        # Get sampling rate from various possible fields (default from config)
        sampling_rate = sample.get('sampling_rate', sample.get('sample_rate', self.audio_sample_rate))
        
        def _preprocess_waveform(waveform, sr):
            """Preprocess waveform: resample, truncate, normalize."""
            try:
                # Ensure 1D waveform (mono)
                if waveform.dim() == 2:
                    waveform = waveform.mean(dim=0)
                elif waveform.dim() > 2:
                    waveform = waveform.squeeze()
                    if waveform.dim() > 1:
                        waveform = waveform.mean(dim=0)
                elif waveform.dim() == 0:
                    return None
                
                # Check for valid waveform
                if waveform.numel() == 0 or waveform.shape[-1] < 100:
                    return None
                
                # Resample if needed
                if sr != self.voice_processor.sample_rate:
                    try:
                        import torchaudio
                        resampler = torchaudio.transforms.Resample(int(sr), self.voice_processor.sample_rate)
                        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
                    except ImportError:
                        pass  # Keep original sample rate if torchaudio not available
                
                # Truncate if too long
                if waveform.shape[-1] > self.voice_processor.max_samples:
                    waveform = waveform[..., :self.voice_processor.max_samples]
                
                # Normalize waveform to [-1, 1] range (important for raw waveform tokenizer)
                max_val = waveform.abs().max()
                if max_val > 0:
                    waveform = waveform / max_val
                
                return waveform
            except Exception:
                return None
        
        def _waveform_to_output(waveform, sr):
            """Convert waveform to appropriate output format based on use_raw_waveform setting."""
            waveform = _preprocess_waveform(waveform, sr)
            if waveform is None:
                return None
            
            if self.use_raw_waveform:
                # SOTA: Return raw waveform [T] for RawWaveformTokenizer
                return waveform
            else:
                # Legacy: Convert to mel spectrogram [n_mels, T]
                # Ensure minimum length for mel extraction
                if waveform.shape[-1] < 400:  # Minimum for FFT
                    pad_len = 400 - waveform.shape[-1]
                    waveform = torch.nn.functional.pad(waveform, (0, pad_len))
                
                mel = self.voice_processor.extract_mel(waveform.unsqueeze(0)).squeeze(0)
                return mel
        
        try:
            # Handle URL - download audio first
            if isinstance(audio_data, str) and audio_data.startswith('http'):
                temp_audio_path = self._download_audio_from_url(audio_data)
                if temp_audio_path:
                    waveform = self.voice_processor.load_audio(temp_audio_path)
                    try:
                        _os.remove(temp_audio_path)
                    except:
                        pass
                    if waveform is not None:
                        return _waveform_to_output(waveform, self.voice_processor.sample_rate)
                return None
            
            # Handle file path
            elif isinstance(audio_data, str):
                waveform = self.voice_processor.load_audio(audio_data)
                if waveform is not None:
                    return _waveform_to_output(waveform, self.voice_processor.sample_rate)
                return None
            
            # Handle dict format (common in HuggingFace datasets)
            elif isinstance(audio_data, dict):
                # PRIORITY 1: Try bytes field (used when Audio(decode=False) - our custom decoder)
                if 'bytes' in audio_data and audio_data['bytes']:
                    try:
                        import io
                        import soundfile as sf
                        audio_buffer = io.BytesIO(audio_data['bytes'])
                        array, sr = sf.read(audio_buffer)
                        waveform = torch.from_numpy(array).float()
                        return _waveform_to_output(waveform, sr)
                    except Exception:
                        # Fallback to librosa
                        try:
                            import io
                            import librosa
                            audio_buffer = io.BytesIO(audio_data['bytes'])
                            array, sr = librosa.load(audio_buffer, sr=None)
                            waveform = torch.from_numpy(array).float()
                            return _waveform_to_output(waveform, sr)
                        except Exception:
                            pass
                
                # PRIORITY 2: Try array (if decode=True was used somehow)
                if 'array' in audio_data and audio_data['array'] is not None:
                    array = audio_data['array']
                    sr = audio_data.get('sampling_rate', sampling_rate)
                    
                    try:
                        if isinstance(array, np.ndarray):
                            waveform = torch.from_numpy(array.copy()).float()
                        elif isinstance(array, list):
                            waveform = torch.tensor(array, dtype=torch.float32)
                        elif isinstance(array, torch.Tensor):
                            waveform = array.float()
                        else:
                            waveform = torch.tensor(array).float()
                        
                        return _waveform_to_output(waveform, sr)
                    except Exception:
                        pass
                
                # PRIORITY 3: Try path field
                if 'path' in audio_data and audio_data['path']:
                    audio_path = audio_data['path']
                    # Check if it's a URL
                    if isinstance(audio_path, str) and audio_path.startswith('http'):
                        temp_audio_path = self._download_audio_from_url(audio_path)
                        if temp_audio_path:
                            waveform = self.voice_processor.load_audio(temp_audio_path)
                            try:
                                _os.remove(temp_audio_path)
                            except:
                                pass
                            if waveform is not None:
                                return _waveform_to_output(waveform, self.voice_processor.sample_rate)
                    else:
                        waveform = self.voice_processor.load_audio(audio_path)
                        if waveform is not None:
                            return _waveform_to_output(waveform, self.voice_processor.sample_rate)
                
                return None
            
            # Handle numpy array directly
            elif isinstance(audio_data, np.ndarray):
                waveform = torch.from_numpy(audio_data.copy()).float()
                return _waveform_to_output(waveform, sampling_rate)
            
            # Handle torch tensor directly
            elif isinstance(audio_data, torch.Tensor):
                waveform = audio_data.float()
                return _waveform_to_output(waveform, sampling_rate)
            
            # Fallback - if we got here and use_raw_waveform is False, try legacy mel processing
            if not self.use_raw_waveform:
                mel = self.voice_processor.process_audio_array(audio_data, sampling_rate)
                return mel
            return None
        except Exception:
            return None

    def _extract_image_data(self, sample: Dict, dtype: str) -> Any:
        """Extract raw image data from sample."""
        # Common image fields across different datasets
        # Image_Prompt: TIP-I2V dataset
        # source_img: image editing datasets
        # prompt_asset: UI/design datasets
        # column0: Pexels dataset (thumbnail URL)
        image_fields = ["image", "Image_Prompt", "jpg", "source_img", "original_image", "input_image", "prompt_asset", "column0"]
        for field in image_fields:
            if field in sample and sample[field] is not None:
                val = sample[field]
                # Skip header row values (Pexels has 'thumbnail_loc' as header)
                if isinstance(val, str) and val in ['thumbnail_loc', 'content_loc', 'image', 'url']:
                    continue
                return val
        return None

    def _get_vript_video_url(self, video_id: str) -> Optional[str]:
        """Look up video URL from Vript metadata cache.
        
        Vript stores video metadata in JSON files on HuggingFace:
        - vript_meta/vript_short_videos_meta.json
        - vript_meta/vript_long_videos_meta.json
        
        Each entry contains 'webpage_url' or 'original_url' with the actual video URL.
        """
        # Lazy load the metadata cache
        if self._vript_meta_cache is None:
            self._vript_meta_cache = {}
            try:
                import requests
                # Load short videos meta
                short_url = "https://huggingface.co/datasets/Mutonix/Vript/resolve/main/vript_meta/vript_short_videos_meta.json"
                resp = requests.get(short_url, timeout=60)
                if resp.status_code == 200:
                    self._vript_meta_cache.update(resp.json())
                    print(f"   📥 Loaded Vript short videos metadata: {len(self._vript_meta_cache)} entries")
            except Exception as e:
                print(f"   ⚠️ Failed to load Vript metadata: {e}")
        
        # Look up the video ID
        if video_id in self._vript_meta_cache:
            meta = self._vript_meta_cache[video_id]
            # Prefer original_url (YouTube Shorts), then webpage_url
            url = meta.get('original_url') or meta.get('webpage_url')
            return url
        
        return None

    def _extract_video_data(self, sample: Dict, dtype: str) -> Any:
        """Extract raw video data from sample."""
        if dtype not in ['video_caption', 'video_qa', 'video_generation', 'image_to_video', 'video_preference', 'video_likert']:
            return None
        
        # Direct video data fields
        video_fields = ["video", "video_path", "video_bytes", "frames", "video_data"]
        for field in video_fields:
            if field in sample and sample[field] is not None:
                val = sample[field]
                if isinstance(val, str):
                    if val.startswith('http') or val.endswith(('.mp4', '.webm', '.avi', '.gif', '.mov', '.mkv')):
                        return val
                    elif val.startswith('s3://') or val.startswith('gs://') or '/videos/' in val:
                        return val
                elif isinstance(val, (bytes, list, dict)):
                    return val
        
        # URL fields
        url_fields = ["Video", "video_url", "video1", "column1", "contentUrl", "url", "video_link", "mp4_url", "media_url"]
        for field in url_fields:
            if field in sample and sample[field]:
                url = sample[field]
                if isinstance(url, str) and url.startswith('http'):
                    if url not in ['content_loc', 'thumbnail_loc', 'url', 'video_url']:
                        return url
        
        # clip_id -> YouTube URL
        if "clip_id" in sample:
            clip_id = sample["clip_id"]
            if clip_id and isinstance(clip_id, str) and len(clip_id) == 11:
                if clip_id.replace('-', '').replace('_', '').isalnum():
                    return f"https://www.youtube.com/watch?v={clip_id}"
        
        # video_id -> YouTube URL
        if "video_id" in sample:
            vid_id = sample["video_id"]
            if vid_id and isinstance(vid_id, str):
                if vid_id.startswith("v_"):
                    return f"https://www.youtube.com/watch?v={vid_id[2:]}"
                elif len(vid_id) == 11:
                    return f"https://www.youtube.com/watch?v={vid_id}"
        
        # videoID -> YouTube URL
        if "videoID" in sample:
            vid_id = sample["videoID"]
            if vid_id and isinstance(vid_id, str) and len(vid_id) == 11:
                return f"https://www.youtube.com/watch?v={vid_id}"
        
        # Vript meta.video_id
        if "meta" in sample and isinstance(sample["meta"], dict):
            meta = sample["meta"]
            if "video_id" in meta:
                vid_id = str(meta["video_id"])
                url = self._get_vript_video_url(vid_id)
                if url:
                    return url
                if len(vid_id) == 11:
                    return f"https://www.youtube.com/watch?v={vid_id}"
                elif vid_id.isdigit() and len(vid_id) > 15:
                    return f"https://www.tiktok.com/@user/video/{vid_id}"
        
        return None

    def _extract_audio_data(self, sample: Dict, dtype: str) -> Any:
        """Extract raw audio data from sample."""
        if dtype not in ['voice_asr', 'voice_tts']:
            return None
        
        audio_fields = ["audio", "speech", "waveform", "audio_path", "file"]
        for field in audio_fields:
            if field in sample and sample[field] is not None:
                return sample[field]
        
        # URL fields
        url_fields = ["audio_url", "url", "file_url"]
        for field in url_fields:
            if field in sample and sample[field]:
                url = sample[field]
                if isinstance(url, str) and url.startswith('http'):
                    return url
        
        return None

    def _process_raw_sample(self, raw_sample: Dict, dtype: str, cfg: Dict) -> Optional[Dict[str, torch.Tensor]]:
        """
        Process a raw sample into model-ready tensors.
        
        This includes:
        1. Filtering (if needed)
        2. Extracting media data
        3. Formatting text using format_functions
        4. Tokenizing
        5. Processing media on-demand
        """
        try:
            # Filter samples if needed
            if cfg.get("filter_images") and raw_sample.get("image") is None:
                return None
            
            # Extract raw media data
            raw_image_data = self._extract_image_data(raw_sample, dtype)
            raw_video_data = self._extract_video_data(raw_sample, dtype)
            raw_audio_data = self._extract_audio_data(raw_sample, dtype)
            
            # Create sample metadata (small, serializable data only for formatting)
            # Note: Conversation fields can be very large (100k+ chars) but are essential for training
            sample_metadata = {}
            # Fields that can have large content but are essential for training
            essential_text_fields = {
                "conversations", "conversation", "messages", "dialog", "dialogue", 
                "turns", "data", "chat", "translated_problem", "translated_solution"
            }
            for k, v in raw_sample.items():
                if k in ["image", "video", "frames", "jpg", "jpeg", "png",
                        "source_img", "target_img", "audio", "speech", "waveform"]:
                    continue
                if isinstance(v, (str, int, float, bool, type(None))):
                    sample_metadata[k] = v
                elif isinstance(v, (list, dict)):
                    # Allow larger content for essential text fields (up to 500KB)
                    # Other fields still have 10KB limit to avoid memory issues
                    max_size = 500000 if k in essential_text_fields else 10000
                    if len(str(v)) < max_size:
                        sample_metadata[k] = v
            
            # Format the sample text using the appropriate format function
            format_fn = self.format_functions.get(dtype)
            if not format_fn:
                return None
            
            formatted = format_fn(sample_metadata)
            if not formatted or not formatted.get("text"):
                return None
            
            text = formatted["text"]
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            
            # Create labels - only compute loss on ASSISTANT responses
            # This is the gold standard for instruction tuning
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # Mask padding
            
            # Mask everything EXCEPT assistant responses
            # Use get_vocab() for reliable token ID lookup (encode() can return multiple tokens)
            assistant_start_token = self.tokens.get('assistant_start', '<|assistant|>')
            assistant_end_token = self.tokens.get('assistant_end', '<|/assistant|>')
            
            vocab = self.tokenizer.get_vocab()
            assistant_start_id = vocab.get(assistant_start_token)
            assistant_end_id = vocab.get(assistant_end_token)
            
            if assistant_start_id is not None and assistant_end_id is not None:
                in_assistant = False
                for i in range(len(input_ids)):
                    token_id = input_ids[i].item()
                    
                    if token_id == assistant_start_id:
                        in_assistant = True
                        labels[i] = -100  # Don't predict the start token itself
                    elif token_id == assistant_end_id:
                        in_assistant = False
                        # Keep the end token in labels (model should learn to end)
                    elif not in_assistant:
                        labels[i] = -100  # Mask non-assistant tokens
            
            # Process media on-demand
            pixel_values = self._process_image(raw_image_data) if raw_image_data else None
            if pixel_values is None:
                pixel_values = torch.zeros(3, self.image_size, self.image_size)
            
            video_frames = None
            is_video_dtype = dtype in ['video_caption', 'video_qa', 'video_generation', 'image_to_video', 'video_preference', 'video_likert']
            
            if raw_video_data and is_video_dtype:
                video_frames = self._process_video_frames(raw_video_data, sample_metadata)
                if dtype == 'image_to_video' and video_frames is not None and raw_image_data is None:
                    pixel_values = video_frames[0]
            
            # For image_to_video: if no video data but we have an image, create video frames from image
            if video_frames is None and dtype == 'image_to_video' and raw_image_data is not None:
                img_tensor = self._process_image(raw_image_data)
                if img_tensor is not None:
                    if img_tensor.shape[1] != self.video_size or img_tensor.shape[2] != self.video_size:
                        img_tensor = F.interpolate(
                            img_tensor.unsqueeze(0), 
                            size=(self.video_size, self.video_size), 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze(0)
                    video_frames = img_tensor.unsqueeze(0).expand(self.max_video_frames, -1, -1, -1).clone()
            
            # Skip video samples with no valid frames - get next sample instead
            if is_video_dtype:
                if video_frames is None or video_frames.abs().mean() < 1e-6:
                    return None  # Skip this sample
            
            if video_frames is None:
                video_frames = torch.zeros(self.max_video_frames, 3, self.video_size, self.video_size)
            
            # Only process audio for audio samples - use minimal placeholder for others
            is_audio_sample = dtype in ['voice_asr', 'voice_tts']
            audio_features = None
            speaker_ref_audio = None  # For voice cloning
            
            if is_audio_sample and raw_audio_data:
                audio_features = self._process_audio(raw_audio_data, sample_metadata)
                
                # For TTS with voice cloning: use the same audio as speaker reference
                # This teaches the model to capture speaker characteristics
                if dtype == 'voice_tts' and audio_features is not None:
                    use_self_as_ref = sample_metadata.get('use_self_as_reference', True)
                    if use_self_as_ref:
                        # Clone the audio for speaker reference (will be processed separately)
                        speaker_ref_audio = audio_features.clone()
            
            if audio_features is None:
                # Use minimal tensor for non-audio samples to save memory
                # For audio samples with failed processing, use proper size
                if is_audio_sample:
                    if self.use_raw_waveform:
                        max_waveform_samples = self.audio_sample_rate * 10  # 10 seconds max
                        audio_features = torch.zeros(max_waveform_samples)
                    else:
                        audio_features = torch.zeros(self.audio_n_mels, self.audio_max_length)
                else:
                    # Minimal placeholder for non-audio samples (image, video, text)
                    # Collate function will handle this appropriately
                    audio_features = torch.zeros(1)
            else:
                # Pad/truncate actual audio data
                if self.use_raw_waveform:
                    max_waveform_samples = self.audio_sample_rate * 10  # 10 seconds max
                    if audio_features.dim() == 1:
                        if audio_features.shape[0] > max_waveform_samples:
                            audio_features = audio_features[:max_waveform_samples]
                        elif audio_features.shape[0] < max_waveform_samples:
                            pad = torch.zeros(max_waveform_samples - audio_features.shape[0])
                            audio_features = torch.cat([audio_features, pad], dim=0)
                else:
                    if audio_features.dim() == 2:
                        if audio_features.shape[1] > self.audio_max_length:
                            audio_features = audio_features[:, :self.audio_max_length]
                        elif audio_features.shape[1] < self.audio_max_length:
                            pad = torch.zeros(audio_features.shape[0], self.audio_max_length - audio_features.shape[1])
                            audio_features = torch.cat([audio_features, pad], dim=1)
            
            # Process speaker reference audio for voice cloning
            # Use a shorter segment (3 seconds) as reference to learn speaker characteristics
            if speaker_ref_audio is not None:
                if self.use_raw_waveform:
                    # Use first 3 seconds as speaker reference (enough to capture voice characteristics)
                    ref_samples = self.audio_sample_rate * 3  # 3 seconds
                    if speaker_ref_audio.dim() == 1:
                        if speaker_ref_audio.shape[0] > ref_samples:
                            speaker_ref_audio = speaker_ref_audio[:ref_samples]
                        elif speaker_ref_audio.shape[0] < ref_samples:
                            pad = torch.zeros(ref_samples - speaker_ref_audio.shape[0])
                            speaker_ref_audio = torch.cat([speaker_ref_audio, pad], dim=0)
                else:
                    # For mel spectrogram, use first ~3 seconds worth of frames
                    ref_frames = int(3 * self.audio_sample_rate / 256)  # hop_length=256
                    if speaker_ref_audio.dim() == 2:
                        if speaker_ref_audio.shape[1] > ref_frames:
                            speaker_ref_audio = speaker_ref_audio[:, :ref_frames]
                        elif speaker_ref_audio.shape[1] < ref_frames:
                            pad = torch.zeros(speaker_ref_audio.shape[0], ref_frames - speaker_ref_audio.shape[1])
                            speaker_ref_audio = torch.cat([speaker_ref_audio, pad], dim=1)
            else:
                # No speaker reference - use zeros (model will learn generic voice)
                if is_audio_sample and dtype == 'voice_tts':
                    if self.use_raw_waveform:
                        speaker_ref_audio = torch.zeros(self.audio_sample_rate * 3)
                    else:
                        ref_frames = int(3 * self.audio_sample_rate / 256)
                        speaker_ref_audio = torch.zeros(self.audio_n_mels, ref_frames)
                else:
                    # Minimal placeholder for non-TTS samples
                    speaker_ref_audio = torch.zeros(1)
            
            # Validate: ensure we have at least some valid labels to train on
            # Samples with ALL -100 labels cause NaN loss and waste compute
            num_valid_labels = (labels != -100).sum().item()
            if num_valid_labels == 0:
                # No valid labels - skip this sample entirely
                return None
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "pixel_values": pixel_values,
                "video_frames": video_frames,
                "audio_features": audio_features,
                "speaker_ref_audio": speaker_ref_audio,  # For voice cloning
                "sample_type": dtype,
            }
        
        except Exception:
            return None

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate through all datasets in round-robin fashion until max_per_epoch UNIQUE samples.
        
        This is memory-efficient because:
        - Only one sample is processed at a time
        - No samples are stored in memory
        - Media is processed on-demand and immediately returned
        - Samples are formatted using format_functions before yielding
        - Each dataset is capped at max_per_dataset UNIQUE samples to prevent domination
        - Each unique sample is yielded sample_repeat times for stronger learning signal
        
        In DUAL TRAINING mode (modality_max_values set):
        - Each modality respects its own max (e.g., text: 3300, video: 1000)
        - Epoch ends when ALL active modalities reach their limits
        - Modalities wait for each other before moving to next epoch
        """
        # Check if resuming from saved state
        resume_unique = self._streaming_state.get("unique_samples", 0)
        resume_yields = self._streaming_state.get("total_yields", 0)
        has_resume_state = resume_unique > 0
        
        unique_samples = resume_unique
        total_yields = resume_yields
        
        # Per-modality counters for dual training
        modality_counts = self._streaming_state.get("modality_counts", {}).copy()
        if not modality_counts:
            modality_counts = {"text": 0, "image": 0, "video": 0, "audio": 0}
        
        # Map dataset types to modalities
        modality_map = {
            "text": ["text", "code", "conversation", "tool_use", "agentic"],
            "image": ["image_caption", "image_vqa", "image_generation", "image_editing", "ui_to_code"],
            "video": ["video_caption", "video_qa", "video_generation", "image_to_video", "video_preference", "video_likert"],
            "audio": ["voice_asr", "voice_tts"],
        }
        
        # Reverse map: dtype -> modality
        dtype_to_modality = {}
        for modality, dtypes in modality_map.items():
            for dtype in dtypes:
                dtype_to_modality[dtype] = modality
        
        # Create iterators for all sources with per-dataset counters
        # IMPORTANT: We track TWO separate things:
        # 1. stream_position: cumulative position in data stream (for resuming across epochs)
        # 2. epoch_count: samples used THIS epoch (resets each epoch, capped at max_per_dataset)
        active_sources = []
        for source in self._dataset_sources:
            dataset_name = source["name"]
            stream_position = self._streaming_state.get("dataset_positions", {}).get(dataset_name, 0)
            
            if source["is_local"]:
                # For local files, skip directly to line number (efficient)
                if stream_position > 0:
                    iterator = self._create_local_iterator_from_line(source["local_path"], stream_position)
                    print(f"   ⏩ {dataset_name}: resuming from line {stream_position}")
                else:
                    iterator = self._create_local_iterator(source["local_path"])
            else:
                # For HF streaming, need to skip through samples
                iterator = iter(source["hf_dataset"])
                if stream_position > 0:
                    print(f"   ⏩ {dataset_name}: skipping {stream_position} samples...", end="", flush=True)
                    skipped = 0
                    try:
                        for _ in range(stream_position):
                            next(iterator)
                            skipped += 1
                    except StopIteration:
                        pass
                    print(f" done", flush=True)
            
            # Get modality for this source
            source_modality = dtype_to_modality.get(source["dtype"], "text")
            
            active_sources.append({
                **source,
                "iterator": iterator,
                "exhausted": False,
                "epoch_count": 0,  # Per-epoch count (resets each epoch)
                "stream_position": stream_position,  # Cumulative position in data stream
                "modality": source_modality,  # Which modality this source belongs to
            })
        
        # Track stats by category
        type_counts = {}
        dataset_counts = {}
        
        total_expected = self.max_per_epoch * self.sample_repeat
        repeat_info = f" × {self.sample_repeat} repeat = {total_expected:,} total" if self.sample_repeat > 1 else ""
        
        if has_resume_state:
            print(f"\n🔄 Resuming streaming iteration from {unique_samples:,} unique samples...", flush=True)
        else:
            if self.is_dual_training:
                print(f"\n🚀 Starting DUAL TRAINING iteration...", flush=True)
                for mod, max_val in self.modality_max_values.items():
                    print(f"   {mod}: 0/{max_val} samples", flush=True)
            else:
                print(f"\n🚀 Starting streaming iteration ({self.max_per_epoch:,} unique samples{repeat_info})...", flush=True)
        print(f"   📊 Max {self.max_per_dataset} unique per dataset", flush=True)
        
        def check_modality_limit(modality: str) -> bool:
            """Check if a modality has reached its limit in dual training mode."""
            if not self.is_dual_training:
                return False
            max_for_modality = self.modality_max_values.get(modality, float('inf'))
            return modality_counts.get(modality, 0) >= max_for_modality
        
        def all_modalities_done() -> bool:
            """Check if all active modalities have reached their limits."""
            if not self.is_dual_training:
                return unique_samples >= self.max_per_epoch
            # Check if ALL modalities in modality_max_values have reached their limits
            for modality, max_val in self.modality_max_values.items():
                if modality_counts.get(modality, 0) < max_val:
                    return False
            return True
        
        # Round-robin through sources - count UNIQUE samples toward max_per_epoch
        while not all_modalities_done() and active_sources:
            # Shuffle sources each round for variety
            random.shuffle(active_sources)
            
            sources_to_remove = []
            
            for source_idx, source in enumerate(active_sources):
                # Check stopping condition
                if all_modalities_done():
                    break
                
                if source["exhausted"]:
                    sources_to_remove.append(source_idx)
                    continue
                
                # Check if this dataset has hit its PER-EPOCH limit
                if source["epoch_count"] >= self.max_per_dataset:
                    source["exhausted"] = True
                    sources_to_remove.append(source_idx)
                    continue
                
                # In dual training, check if this source's modality has hit its limit
                source_modality = source.get("modality", "text")
                if check_modality_limit(source_modality):
                    source["exhausted"] = True
                    sources_to_remove.append(source_idx)
                    continue
                
                try:
                    raw_sample = next(source["iterator"])
                    
                    # Process and format the sample ONCE from streaming
                    processed = self._process_raw_sample(raw_sample, source["dtype"], source["config"])
                    
                    if processed is not None:
                        # Yield the SAME processed sample multiple times
                        # Sample is loaded from stream once, processed once, yielded sample_repeat times
                        # This is memory efficient - same tensors are reused
                        for repeat_idx in range(self.sample_repeat):
                            yield processed
                            total_yields += 1
                        
                        # Count as one unique sample
                        unique_samples += 1
                        source["epoch_count"] += 1  # Per-epoch count (for limit check)
                        source["stream_position"] += 1  # Cumulative position (for resume)
                        
                        # Update per-modality count for dual training
                        modality_counts[source_modality] = modality_counts.get(source_modality, 0) + 1
                        
                        dtype = source["dtype"]
                        type_counts[dtype] = type_counts.get(dtype, 0) + 1
                        dataset_counts[source["name"]] = source["epoch_count"]
                        
                        # Update streaming state for resume support
                        # Save the STREAM POSITION (cumulative) so next epoch continues from here
                        self._streaming_state["unique_samples"] = unique_samples
                        self._streaming_state["total_yields"] = total_yields
                        self._streaming_state["dataset_positions"][source["name"]] = source["stream_position"]
                        self._streaming_state["modality_counts"] = modality_counts.copy()
                        
                        # Also update per-modality positions for --text, --image, --video, --voice resume
                        for modality, dtypes in modality_map.items():
                            if dtype in dtypes:
                                self._streaming_state["modality_positions"][modality][source["name"]] = source["stream_position"]
                                break
                        
                        # Log progress and save state every 500 unique samples
                        if unique_samples % 500 == 0:
                            if self.is_dual_training:
                                progress_parts = []
                                for mod, max_val in self.modality_max_values.items():
                                    current = modality_counts.get(mod, 0)
                                    progress_parts.append(f"{mod}: {current}/{max_val}")
                                print(f"   📈 {' | '.join(progress_parts)}", flush=True)
                            else:
                                print(f"   📈 {unique_samples:,}/{self.max_per_epoch:,} samples", flush=True)
                            # Auto-save state if path is set
                            if self._state_save_path:
                                self.save_streaming_state(self._state_save_path)
                
                except StopIteration:
                    source["exhausted"] = True
                    sources_to_remove.append(source_idx)
                except Exception as e:
                    # Skip problematic samples but log for debugging
                    logger.debug(f"Sample processing skipped in {source.get('name', 'unknown')}: {e}")
            
            # Remove exhausted sources
            for idx in sorted(sources_to_remove, reverse=True):
                if idx < len(active_sources):
                    del active_sources[idx]
        
        # Print final stats
        print(f"\n✅ Epoch complete!", flush=True)
        if self.is_dual_training:
            print(f"   🔀 Dual training results:", flush=True)
            for mod, max_val in self.modality_max_values.items():
                current = modality_counts.get(mod, 0)
                print(f"      {mod}: {current}/{max_val} samples", flush=True)
        print(f"   📊 {unique_samples:,} unique samples × {self.sample_repeat} = {total_yields:,} total yields", flush=True)
        print(f"   📂 By category:", flush=True)
        for dtype, count in sorted(type_counts.items()):
            print(f"      {dtype}: {count} unique samples", flush=True)
        
        # Save final state
        if self._state_save_path:
            self._streaming_state["epoch"] = self._streaming_state.get("epoch", 0) + 1
            self.save_streaming_state(self._state_save_path)
            print(f"   💾 Streaming state saved to {self._state_save_path}")
        
        gc.collect()

    def __len__(self):
        """Return total yields (unique samples × repeat) as the effective length."""
        return self.max_per_epoch * self.sample_repeat

    def reset(self, clear_state: bool = False):
        """Reset dataset for new epoch - reinitialize all sources.
        
        Args:
            clear_state: If True, completely clear streaming state (restart from beginning of all datasets)
                        If False (default), keep dataset_positions to continue from where we left off
                        
        Both training and eval datasets advance each epoch to get NEW samples.
        They have separate position tracking so they don't interfere with each other.
        """
        if clear_state:
            # Full reset - start from beginning of all datasets
            self._streaming_state = {
                "epoch": self._streaming_state.get("epoch", 0) + 1,
                "unique_samples": 0,
                "total_yields": 0,
                "dataset_positions": {},  # Clear positions to restart from beginning
                "modality_positions": {"text": {}, "image": {}, "video": {}, "voice": {}},
                "last_modality": None,
            }
        else:
            # Epoch reset - keep stream positions but reset epoch counters
            # This allows each epoch to get NEW data (continuing from where last epoch left off)
            self._streaming_state = {
                "epoch": self._streaming_state.get("epoch", 0) + 1,
                "unique_samples": 0,  # Reset for this epoch
                "total_yields": 0,    # Reset for this epoch
                "dataset_positions": self._streaming_state.get("dataset_positions", {}),  # KEEP positions!
                "modality_positions": self._streaming_state.get("modality_positions", {"text": {}, "image": {}, "video": {}, "voice": {}}),
                "last_modality": self._streaming_state.get("last_modality", None),
            }
        self._init_iterators()
        gc.collect()


def create_train_eval_datasets(
    dataset_configs: Dict[str, List[Dict]],
    format_functions: Dict[str, Callable],
    tokenizer,
    tokens: Dict[str, str],
    image_processor,
    max_length: int = 1024,  # From TrainingConfig.max_seq_length
    max_per_epoch_train: int = 2000,  # From TrainingConfig.max_per_epoch
    max_per_dataset_train: int = 100,  # From TrainingConfig.max_per_dataset
    max_per_dataset_eval: int = 10,  # From TrainingConfig.max_per_dataset_eval
    sample_repeat: int = 2,  # From TrainingConfig.sample_repeat
    voice_processor=None,
    max_video_frames: int = 32,  # From XoronConfig.video_max_frames (multi-scale)
    video_size: int = 256,  # From XoronConfig.video_base_size (multi-scale)
    image_size: int = 256,  # From XoronConfig.image_base_size (multi-scale)
    audio_n_mels: int = 80,  # From XoronConfig.audio_n_mels
    audio_max_length: int = 1000,  # From XoronConfig.audio_max_length
    audio_sample_rate: int = 16000,  # From XoronConfig.audio_sample_rate
    use_raw_waveform: bool = True,  # From XoronConfig.use_raw_waveform
):
    """
    Create separate train and eval datasets that sample INDEPENDENTLY from each dataset.
    
    This ensures proper validation by:
    1. Train dataset: pulls max_per_dataset_train samples from each dataset
    2. Eval dataset: pulls max_per_dataset_eval samples from each dataset (held-out data)
    
    Both datasets stream from the same sources but:
    - Eval uses skip_initial to start AFTER where train ends
    - Each dataset contributes equal samples to eval (fair evaluation across all modalities)
    - Total eval samples = num_datasets * max_per_dataset_eval (no artificial cap)
    
    Args:
        dataset_configs: Dataset configurations by type
        format_functions: Formatter functions by dataset type
        tokenizer: Tokenizer instance
        tokens: Special tokens dict
        image_processor: Image processor instance
        max_length: Max sequence length
        max_per_epoch_train: Total train samples per epoch
        max_per_dataset_train: Samples per dataset for training (e.g., 100)
        max_per_dataset_eval: Samples per dataset for eval (e.g., 10)
        sample_repeat: How many times to repeat each sample in training
        voice_processor: Voice processor instance
        max_video_frames: Max video frames
        video_size: Video frame size
        
    Returns:
        tuple: (train_dataset, eval_dataset)
        
    Example:
        # Training: 100 samples from each dataset
        # Eval: 10 samples from each dataset (held out, not seen during training)
        # If you have 66 datasets: eval = 66 * 10 = 660 samples
        train_dataset, eval_dataset = create_train_eval_datasets(
            dataset_configs=configs,
            format_functions=formatters,
            tokenizer=tokenizer,
            tokens=tokens,
            image_processor=image_processor,
            max_per_epoch_train=6600,
            max_per_dataset_train=100,  # 100 per dataset for train
            max_per_dataset_eval=10,    # 10 per dataset for eval (separate samples)
        )
    """
    # Count total datasets to calculate dynamic eval size
    total_datasets = sum(len(configs) for configs in dataset_configs.values() if configs)
    max_per_epoch_eval = total_datasets * max_per_dataset_eval
    
    print("\n📊 Creating train and eval datasets with per-dataset sampling...")
    print(f"   Train: {max_per_dataset_train} samples/dataset, {max_per_epoch_train} max total/epoch")
    print(f"   Eval:  {max_per_dataset_eval} samples/dataset × {total_datasets} datasets = {max_per_epoch_eval} total")
    
    # Create train dataset - starts from beginning of each dataset stream
    train_dataset = TrueStreamingDataset(
        dataset_configs=dataset_configs,
        format_functions=format_functions,
        tokenizer=tokenizer,
        tokens=tokens,
        image_processor=image_processor,
        max_length=max_length,
        max_per_epoch=max_per_epoch_train,
        max_per_dataset=max_per_dataset_train,
        sample_repeat=sample_repeat,
        voice_processor=voice_processor,
        max_video_frames=max_video_frames,
        video_size=video_size,
        image_size=image_size,
        audio_n_mels=audio_n_mels,
        audio_max_length=audio_max_length,
        audio_sample_rate=audio_sample_rate,
        use_raw_waveform=use_raw_waveform,
    )
    
    # Create eval dataset - starts AFTER training samples in the stream
    # Both train and eval advance each epoch to get NEW samples
    # max_per_epoch for eval = total_datasets * max_per_dataset_eval (no cap)
    eval_dataset = TrueStreamingDataset(
        dataset_configs=dataset_configs,
        format_functions=format_functions,
        tokenizer=tokenizer,
        tokens=tokens,
        image_processor=image_processor,
        max_length=max_length,
        max_per_epoch=max_per_epoch_eval,  # Dynamic: num_datasets * samples_per_dataset
        max_per_dataset=max_per_dataset_eval,
        sample_repeat=1,  # No repeat for eval - just evaluate once per sample
        voice_processor=voice_processor,
        max_video_frames=max_video_frames,
        video_size=video_size,
        image_size=image_size,
        audio_n_mels=audio_n_mels,
        audio_max_length=audio_max_length,
        audio_sample_rate=audio_sample_rate,
        use_raw_waveform=use_raw_waveform,
    )
    
    # Set initial skip positions for eval dataset
    # Eval starts AFTER training's initial samples, then BOTH advance each epoch
    # Epoch 1: Train 0-99, Eval 100-109
    # Epoch 2: Train 100-199, Eval 110-119 (both advanced by their respective amounts)
    eval_skip_positions = {}
    for dtype, configs in dataset_configs.items():
        if configs:
            for cfg in configs:
                # Eval starts after max_per_dataset_train samples
                eval_skip_positions[cfg["name"]] = max_per_dataset_train
    
    eval_dataset._streaming_state["dataset_positions"] = eval_skip_positions.copy()
    
    print(f"   ✅ Train dataset: {train_dataset.total_datasets} dataset sources")
    print(f"   ✅ Eval dataset: {eval_dataset.total_datasets} dataset sources")
    print(f"      → Epoch 1: Train samples 0-{max_per_dataset_train-1}, Eval samples {max_per_dataset_train}-{max_per_dataset_train + max_per_dataset_eval - 1}")
    print(f"      → Both advance each epoch - NEW samples every epoch!")
    
    return train_dataset, eval_dataset

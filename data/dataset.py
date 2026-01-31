"""Streaming dataset for multimodal training."""

import gc
import os
import random
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset
from typing import Dict, List, Any, Optional, Callable
from PIL import Image
import numpy as np


class TrueStreamingDataset(Dataset):
    """
    True streaming dataset with lazy loading and on-demand media processing.
    
    Key improvements for memory efficiency:
    - Lazy initialization: First chunk is NOT loaded until first __getitem__ call
    - On-demand processing: Media (images/video/audio) is processed only when accessed
    - Stores only raw sample metadata in chunks, not processed tensors
    - Proper chunk rotation during iteration
    - Memory-efficient: Only current batch's media is in memory at any time
    """

    def __init__(
        self,
        dataset_configs: Dict[str, List[Dict]],
        format_functions: Dict[str, Callable],
        tokenizer,
        tokens: Dict[str, str],
        image_processor,
        max_length: int = 1024,
        max_per_epoch: int = 12000,
        voice_processor=None,
        max_video_frames: int = 32,
        video_size: int = 256,
        samples_per_category: Optional[Dict[str, int]] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples_per_category = samples_per_category or {}
        self.max_per_epoch = max_per_epoch
        self.tokens = tokens
        self.format_functions = format_functions
        self.dataset_configs = dataset_configs
        self.image_processor = image_processor
        self.voice_processor = voice_processor
        self.max_video_frames = max_video_frames
        self.video_size = video_size

        self.total_datasets = sum(len(configs) for configs in dataset_configs.values() if configs)

        # Calculate actual chunk size
        actual_chunk_size = 0
        for cat, configs in dataset_configs.items():
            if configs:
                cat_limit = self.samples_per_category.get(cat, 1)
                actual_chunk_size += len(configs) * cat_limit
        self.chunk_size = actual_chunk_size

        self.dataset_iterators = {}
        # Store raw sample data (metadata only, no processed tensors)
        self.current_samples = []
        self.samples_loaded = 0
        self.total_chunks_loaded = 0
        self._initialized = False
        self._current_chunk_idx = 0

        # Lazy initialization - only set up iterators, don't load data yet
        self._init_iterators()

    def _init_iterators(self):
        """Initialize dataset iterators."""
        from datasets import load_dataset
        
        # Import Audio for casting audio columns
        try:
            from datasets import Audio
            has_audio_feature = True
        except ImportError:
            has_audio_feature = False

        for dtype, configs in self.dataset_configs.items():
            if not configs:
                continue
            self.dataset_iterators[dtype] = []
            for cfg in configs:
                max_retries = 3
                retry_delay = 2
                ds = None
                
                # Handle local JSONL files
                if cfg.get("local", False):
                    ds = self._load_local_dataset(cfg, dtype)
                    if ds is not None:
                        sample_limit = self.samples_per_category.get(dtype, 25)
                        self.dataset_iterators[dtype].append({
                            "name": cfg["name"],
                            "iterator": iter(ds),
                            "config": cfg,
                            "exhausted": False,
                        })
                        print(f"   ✅ {cfg['name']} ({len(ds)}/{sample_limit} samples)")
                    continue
                
                # Handle remote HuggingFace datasets
                for attempt in range(max_retries):
                    try:
                        load_kwargs = {
                            "path": cfg["path"],
                            "split": cfg["split"],
                            "streaming": cfg.get("streaming", True),
                        }
                        if "config" in cfg:
                            load_kwargs["name"] = cfg["config"]
                        ds = load_dataset(**load_kwargs)
                        
                        # For voice datasets, disable automatic audio decoding to avoid torchcodec issues
                        # We'll decode manually using soundfile/librosa in _process_audio
                        if dtype in ['voice_asr', 'voice_tts'] and has_audio_feature:
                            try:
                                ds = ds.cast_column('audio', Audio(decode=False))
                            except Exception:
                                pass  # Some datasets may not have audio column
                        
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            retry_delay *= 2
                        else:
                            print(f"   ⚠️ Failed to load {cfg['name']}: {e}")
                            ds = None

                if ds is not None:
                    self.dataset_iterators[dtype].append({
                        "name": cfg["name"],
                        "iterator": iter(ds),
                        "config": cfg,
                        "exhausted": False,
                    })
                    print(f"   ✅ {cfg['name']}")

    def _load_local_dataset(self, cfg, dtype):
        """Load a local JSONL dataset file with sample limit."""
        import json
        
        path = cfg.get("path", "")
        
        # Handle relative paths
        if not os.path.isabs(path):
            # Try relative to current directory first
            if not os.path.exists(path):
                # Try relative to project root
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                path = os.path.join(project_root, path)
        
        if not os.path.exists(path):
            print(f"   ⚠️ Local dataset not found: {path}")
            print(f"      Run: python -m synth.generate_dataset to generate it")
            return None
        
        try:
            # Get sample limit for this category
            sample_limit = self.samples_per_category.get(dtype, 25)
            
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
                        # Stop loading once we hit the limit
                        if len(data) >= sample_limit:
                            break
            
            # Don't print here - will be printed with ✅ in _init_iterators
            return data
        except Exception as e:
            print(f"   ⚠️ Failed to load local dataset {path}: {e}")
            return None

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
                return processed['pixel_values'].squeeze(0)
            else:
                # Fallback: use 384x384 for SigLIP compatibility
                img = img.resize((384, 384))
                tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                return tensor

        except Exception:
            return None

    def _download_video_from_url(self, url: str, timeout: int = 30) -> Optional[str]:
        """Download video from URL to a temporary file."""
        temp_path = None
        try:
            import tempfile
            
            # Check if it's a YouTube URL
            if 'youtube.com' in url or 'youtu.be' in url:
                return self._download_youtube_video(url)
            
            import requests
            response = requests.get(url, timeout=timeout, stream=True)
            if response.status_code != 200:
                return None
            
            # Determine file extension from URL or content-type
            content_type = response.headers.get('content-type', '')
            if 'mp4' in url.lower() or 'mp4' in content_type:
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
            
            # Create temp file for output
            fd, temp_path = tempfile.mkstemp(suffix='.mp4')
            os.close(fd)
            
            # Use yt-dlp to download
            cmd = [
                'yt-dlp',
                '-f', 'best[height<=480][ext=mp4]/best[height<=480]/best',  # Limit quality for speed
                '-o', temp_path,
                '--no-playlist',
                '--quiet',
                '--no-warnings',
                '--socket-timeout', '30',
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            
            if result.returncode == 0 and os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                return temp_path
            else:
                # Clean up failed download
                try:
                    os.remove(temp_path)
                except:
                    pass
                return None
        except Exception:
            return None

    def _extract_frames_from_video(self, video_path: str) -> List[torch.Tensor]:
        """Extract frames from a video file."""
        frames = []
        try:
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

            # Resize frames
            frame_tensors = []
            for f in frames:
                if f.shape[1] != self.video_size or f.shape[2] != self.video_size:
                    f = F.interpolate(f.unsqueeze(0), size=(self.video_size, self.video_size), mode='bilinear', align_corners=False).squeeze(0)
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
        """Process audio data to mel-spectrogram."""
        if self.voice_processor is None:
            return None

        import os as _os
        
        # If audio_data is None, try to get it from sample directly
        if audio_data is None:
            return None
        
        # Get sampling rate from various possible fields
        sampling_rate = sample.get('sampling_rate', sample.get('sample_rate', 16000))
        
        def _waveform_to_mel(waveform, sr):
            """Helper to convert waveform to mel spectrogram."""
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
                if sr != self.voice_processor.sample_rate and self.voice_processor.has_torchaudio:
                    import torchaudio
                    resampler = torchaudio.transforms.Resample(int(sr), self.voice_processor.sample_rate)
                    waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
                
                # Truncate if too long
                if waveform.shape[-1] > self.voice_processor.max_samples:
                    waveform = waveform[..., :self.voice_processor.max_samples]
                
                # Ensure minimum length for mel extraction
                if waveform.shape[-1] < 400:  # Minimum for FFT
                    pad_len = 400 - waveform.shape[-1]
                    waveform = torch.nn.functional.pad(waveform, (0, pad_len))
                
                mel = self.voice_processor.extract_mel(waveform.unsqueeze(0)).squeeze(0)
                return mel
            except Exception:
                return None
        
        try:
            # Handle torchcodec AudioDecoder (new HuggingFace datasets format)
            if hasattr(audio_data, 'get_all_samples'):
                try:
                    samples = audio_data.get_all_samples()
                    waveform = samples.data.float()
                    sr = samples.sample_rate
                    return _waveform_to_mel(waveform, sr)
                except Exception:
                    pass
            
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
                        return _waveform_to_mel(waveform, self.voice_processor.sample_rate)
                return None
            
            # Handle file path
            elif isinstance(audio_data, str):
                waveform = self.voice_processor.load_audio(audio_data)
                if waveform is not None:
                    return _waveform_to_mel(waveform, self.voice_processor.sample_rate)
                return None
            
            # Handle dict format (common in HuggingFace datasets)
            elif isinstance(audio_data, dict):
                # Try to get array directly (most common format for HuggingFace audio datasets)
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
                        
                        return _waveform_to_mel(waveform, sr)
                    except Exception:
                        pass
                
                # Try path field (HuggingFace datasets often have this)
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
                                return _waveform_to_mel(waveform, self.voice_processor.sample_rate)
                    else:
                        waveform = self.voice_processor.load_audio(audio_path)
                        if waveform is not None:
                            return _waveform_to_mel(waveform, self.voice_processor.sample_rate)
                
                # Try bytes field
                if 'bytes' in audio_data and audio_data['bytes']:
                    try:
                        import io
                        import soundfile as sf
                        audio_buffer = io.BytesIO(audio_data['bytes'])
                        array, sr = sf.read(audio_buffer)
                        waveform = torch.from_numpy(array).float()
                        return _waveform_to_mel(waveform, sr)
                    except Exception:
                        pass
                
                return None
            
            # Handle numpy array directly
            elif isinstance(audio_data, np.ndarray):
                waveform = torch.from_numpy(audio_data.copy()).float()
                return _waveform_to_mel(waveform, sampling_rate)
            
            # Handle torch tensor directly
            elif isinstance(audio_data, torch.Tensor):
                waveform = audio_data.float()
                return _waveform_to_mel(waveform, sampling_rate)
            
            # Fallback to voice processor's array processing
            mel = self.voice_processor.process_audio_array(audio_data, sampling_rate)
            return mel
        except Exception:
            return None

    def _load_next_chunk(self):
        """
        Load next chunk of samples - stores only RAW DATA, not processed tensors.
        
        This is memory-efficient because:
        - Only text and metadata are stored in memory
        - Raw media references (paths, URLs, bytes) are stored, not processed tensors
        - Actual media processing happens on-demand in __getitem__
        """
        self.current_samples = []
        self._current_chunk_idx = 0

        print(f"\n📦 Loading chunk {self.total_chunks_loaded + 1} (metadata only)...", flush=True)
        
        # Track statistics
        samples_with_image_data = 0
        samples_with_video_data = 0
        samples_with_audio_data = 0

        for dtype, iterators in self.dataset_iterators.items():
            if not iterators:
                continue

            format_fn = self.format_functions.get(dtype)
            if not format_fn:
                continue

            type_total = 0

            for ds_info in iterators:
                if ds_info["exhausted"]:
                    continue

                ds_count = 0
                attempts = 0
                category_limit = self.samples_per_category.get(dtype, 1)
                max_attempts = category_limit * 10

                try:
                    while ds_count < category_limit and attempts < max_attempts:
                        if self.samples_loaded >= self.max_per_epoch:
                            break

                        attempts += 1
                        sample = next(ds_info["iterator"])

                        cfg = ds_info["config"]
                        if cfg.get("filter_images") and sample.get("image") is None:
                            continue

                        # Store RAW image data reference (not processed tensor)
                        raw_image_data = None
                        image_fields = ["image", "jpg", "source_img", "original_image", "input_image", "prompt_asset"]
                        for img_field in image_fields:
                            if img_field in sample and sample[img_field] is not None:
                                raw_image_data = sample[img_field]
                                samples_with_image_data += 1
                                break

                        # Store RAW video data reference (not processed frames)
                        raw_video_data = None
                        if dtype in ['video_caption', 'video_qa', 'video_generation', 'image_to_video', 'video_preference', 'video_likert']:
                            # Direct video data fields
                            video_fields = ["video", "video_path", "video_bytes", "frames", "video_data"]
                            for vid_field in video_fields:
                                if vid_field in sample and sample[vid_field] is not None:
                                    raw_video_data = sample[vid_field]
                                    samples_with_video_data += 1
                                    break

                            # URL fields for video
                            if raw_video_data is None:
                                url_fields = ["contentUrl", "video_url", "videoUrl", "url", "Video", "video1", "video2"]
                                for url_field in url_fields:
                                    if url_field in sample and sample[url_field]:
                                        url = sample[url_field]
                                        if isinstance(url, str) and url.startswith('http'):
                                            raw_video_data = url
                                            samples_with_video_data += 1
                                            break

                        # Store RAW audio data reference (not processed mel)
                        raw_audio_data = None
                        if dtype in ['voice_asr', 'voice_tts']:
                            audio_fields = ["audio", "speech", "waveform", "audio_path", "file"]
                            for aud_field in audio_fields:
                                if aud_field in sample and sample[aud_field] is not None:
                                    raw_audio_data = sample[aud_field]
                                    samples_with_audio_data += 1
                                    break
                            
                            # Try URL fields if no audio found
                            if raw_audio_data is None:
                                url_fields = ["audio_url", "url", "file_url"]
                                for url_field in url_fields:
                                    if url_field in sample and sample[url_field]:
                                        url = sample[url_field]
                                        if isinstance(url, str) and url.startswith('http'):
                                            raw_audio_data = url
                                            samples_with_audio_data += 1
                                            break

                        # Store only serializable metadata (no tensors, no large objects)
                        sample_metadata = {}
                        for k, v in sample.items():
                            # Skip large binary/tensor data - we already captured what we need
                            if k in ["image", "video", "frames", "jpg", "jpeg", "png",
                                    "source_img", "target_img", "audio", "speech", "waveform"]:
                                continue
                            # Only store simple types that are memory-efficient
                            if isinstance(v, (str, int, float, bool, type(None))):
                                sample_metadata[k] = v
                            elif isinstance(v, (list, dict)) and len(str(v)) < 10000:
                                sample_metadata[k] = v

                        formatted = format_fn(sample_metadata)
                        if formatted and formatted.get("text"):
                            # Store raw data references, NOT processed tensors
                            self.current_samples.append({
                                "text": formatted["text"],
                                "raw_image_data": raw_image_data,
                                "raw_video_data": raw_video_data,
                                "raw_audio_data": raw_audio_data,
                                "sample_metadata": sample_metadata,
                                "type": dtype
                            })
                            ds_count += 1
                            type_total += 1
                            self.samples_loaded += 1

                except StopIteration:
                    ds_info["exhausted"] = True
                except Exception as e:
                    if dtype in ['voice_asr', 'voice_tts']:
                        print(f"      ⚠️ Error in {dtype} ({ds_info['name']}): {type(e).__name__}: {str(e)[:100]}", flush=True)

            if type_total > 0:
                print(f"  📂 {dtype}: {type_total} samples", flush=True)
            elif dtype in ['voice_asr', 'voice_tts']:
                print(f"  📂 {dtype}: 0 samples (check dataset format)", flush=True)

            if self.samples_loaded >= self.max_per_epoch:
                break

        self.total_chunks_loaded += 1
        self._initialized = True
        random.shuffle(self.current_samples)

        print(f"  📦 Chunk {self.total_chunks_loaded}: {len(self.current_samples)} samples (metadata only)", flush=True)
        print(f"      With image refs: {samples_with_image_data}, video refs: {samples_with_video_data}, audio refs: {samples_with_audio_data}", flush=True)
        print(f"      💾 Memory efficient: Media will be processed on-demand", flush=True)

        gc.collect()

    def __len__(self):
        return self.max_per_epoch

    def _ensure_initialized(self):
        """Ensure first chunk is loaded (lazy initialization)."""
        if not self._initialized:
            print("\n📥 Loading first chunk on first access...")
            self._load_next_chunk()

    def __getitem__(self, idx):
        """
        Get item with ON-DEMAND media processing.
        
        This is memory-efficient because:
        - Media is processed only when this specific sample is accessed
        - Processed tensors are returned but not stored in the dataset
        - Each batch only has its own media in memory
        """
        # Lazy initialization - load first chunk on first access
        self._ensure_initialized()
        
        # Calculate local index within current chunk
        local_idx = idx % len(self.current_samples) if self.current_samples else 0
        
        # Check if we need to load a new chunk
        # Load new chunk when we've cycled through current chunk AND have more data to load
        if local_idx == 0 and idx > 0 and self.samples_loaded < self.max_per_epoch:
            self._load_next_chunk()

        # Return empty sample if no data available
        if not self.current_samples:
            return {
                "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
                "labels": torch.full((self.max_length,), -100, dtype=torch.long),
                "pixel_values": torch.zeros(3, 224, 224),
                "video_frames": torch.zeros(self.max_video_frames, 3, self.video_size, self.video_size),
                "audio_features": torch.zeros(80, 1000),
            }

        sample_idx = local_idx % len(self.current_samples)
        sample = self.current_samples[sample_idx]
        text = sample["text"]
        dtype = sample.get("type", "text")

        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # ON-DEMAND image processing - process raw data NOW, not during chunk loading
        pixel_values = None
        raw_image_data = sample.get("raw_image_data")
        if raw_image_data is not None:
            pixel_values = self._process_image(raw_image_data)
        
        if pixel_values is None:
            pixel_values = torch.zeros(3, 224, 224)

        # ON-DEMAND video processing - process raw data NOW
        video_frames = None
        raw_video_data = sample.get("raw_video_data")
        if raw_video_data is not None and dtype in ['video_caption', 'video_qa', 'video_generation', 'image_to_video', 'video_preference', 'video_likert']:
            video_frames = self._process_video_frames(raw_video_data, sample.get("sample_metadata", {}))
            
            # For image_to_video, use first frame as image if no image data
            if dtype == 'image_to_video' and video_frames is not None and raw_image_data is None:
                pixel_values = video_frames[0]
        
        if video_frames is None:
            video_frames = torch.zeros(self.max_video_frames, 3, self.video_size, self.video_size)

        # ON-DEMAND audio processing - process raw data NOW
        audio_features = None
        raw_audio_data = sample.get("raw_audio_data")
        if raw_audio_data is not None and dtype in ['voice_asr', 'voice_tts']:
            audio_features = self._process_audio(raw_audio_data, sample.get("sample_metadata", {}))
        
        max_audio_len = 1000
        if audio_features is None:
            audio_features = torch.zeros(80, max_audio_len)
        else:
            if audio_features.shape[1] > max_audio_len:
                audio_features = audio_features[:, :max_audio_len]
            elif audio_features.shape[1] < max_audio_len:
                pad = torch.zeros(audio_features.shape[0], max_audio_len - audio_features.shape[1])
                audio_features = torch.cat([audio_features, pad], dim=1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "video_frames": video_frames,
            "audio_features": audio_features,
            "sample_type": dtype,
        }

    def reset(self):
        """Reset dataset for new epoch."""
        print(f"\n🔄 Resetting dataset for new epoch...")
        self.samples_loaded = 0
        self.total_chunks_loaded = 0
        self.current_samples = []
        self._initialized = False
        self._current_chunk_idx = 0
        self._init_iterators()
        # Don't load chunk here - lazy loading will handle it on first access

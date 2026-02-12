#!/usr/bin/env python3
"""
Xoron Unified Multimodal Dataset Builder

This script builds a unified HuggingFace dataset from all configured dataset sources.
It downloads actual media (audio, video, images) - not just URLs - and creates
proper splits for each modality.

Designed to run on Kaggle with /tmp storage (150GB available).

Usage (on Kaggle):
    python build_unified_dataset.py

The script will:
1. Download samples from all HuggingFace datasets in the config
2. Download actual video/audio files using yt-dlp where needed
3. Create splits: text, audio, image, video (each has train split)
4. Upload to HuggingFace Hub

Author: Xoron Team
"""

import os
import sys

# Force CPU only - must be set before importing any ML libraries
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["USE_CUDA"] = "0"
os.environ["NVIDIA_VISIBLE_DEVICES"] = ""
os.environ["NO_CUDA"] = "1"
os.environ["FORCE_CPU"] = "1"

# Suppress NVML warnings
import warnings
warnings.filterwarnings("ignore", message=".*NVML.*")
warnings.filterwarnings("ignore", message=".*CUDA.*")
warnings.filterwarnings("ignore", category=UserWarning)

import json
import shutil
import tempfile
import subprocess
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# HuggingFace token - get from Kaggle secrets
HF_TOKEN = ""
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
    print("✅ Got HF_TOKEN from Kaggle secrets")
except Exception as e:
    # Fallback to environment variable if not on Kaggle
    HF_TOKEN = os.environ.get("hf_token", "") or os.environ.get("HF_TOKEN", "")
    if HF_TOKEN:
        print("✅ Got HF_TOKEN from environment variable")
    else:
        print(f"⚠️ Could not get HF_TOKEN from Kaggle secrets: {e}")

# Dataset name on HuggingFace
HF_DATASET_NAME = "nigfuapp-web/moe-data"

# Base directory for temporary storage (Kaggle uses /tmp with 150GB)
BASE_DIR = "/tmp/xoron_dataset_build"
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

# Limits per dataset (adjust based on available storage/time)
MAX_SAMPLES_PER_DATASET = 50000  # How many samples to take from each dataset
MAX_VIDEO_SIZE_MB = 50  # Skip videos larger than this
MAX_AUDIO_SIZE_MB = 20  # Skip audio files larger than this
MAX_IMAGE_SIZE_MB = 10  # Skip images larger than this

# Concurrent downloads
MAX_WORKERS = 4

# ============================================================================
# DATASET CONFIGURATIONS (from config/dataset_config.py)
# ============================================================================

DATASET_CONFIGS = {
    # === TEXT DATASETS ===
    "code": [
        {"name": "Code-Feedback", "path": "m-a-p/Code-Feedback", "split": "train"},
        {"name": "Python-Code-18k", "path": "iamtarun/python_code_instructions_18k_alpaca", "split": "train"},
        {"name": "HumanEval-Python", "path": "bigcode/humanevalpack", "config": "python", "split": "test"},
        {"name": "HumanEval-JavaScript", "path": "bigcode/humanevalpack", "config": "js", "split": "test"},
        {"name": "Jupyter-Code", "path": "loubnabnl/github-jupyter-code-to-text", "split": "train"},
    ],
    "conversation": [
        {"name": "Dolly-15k", "path": "databricks/databricks-dolly-15k", "split": "train"},
        {"name": "OpenAssistant", "path": "OpenAssistant/oasst1", "split": "train"},
        {"name": "NoRobots", "path": "HuggingFaceH4/no_robots", "split": "train"},
    ],
    "tool_use": [
        {"name": "Function-Calling-ChatML", "path": "Locutusque/function-calling-chatml", "split": "train"},
        {"name": "Tool-Calls-SingleTurn", "path": "interstellarninja/tool-calls-singleturn", "split": "train"},
        {"name": "Tool-Calls-Multiturn", "path": "interstellarninja/tool-calls-multiturn", "split": "train"},
    ],
    "agentic": [
        {"name": "AgentInstruct", "path": "THUDM/AgentInstruct", "split": "os"},
        {"name": "Glaive-Code-Assistant", "path": "glaiveai/glaive-code-assistant-v2", "split": "train"},
        {"name": "Agentic-CoT-Coding", "path": "AlicanKiraz0/Agentic-Chain-of-Thought-Coding-SFT-Dataset", "split": "train"},
    ],
    
    # === IMAGE DATASETS ===
    "image_caption": [
        {"name": "Flickr8k", "path": "Naveengo/flickr8k", "split": "train"},
        {"name": "Football", "path": "ybelkada/football-dataset", "split": "train"},
        {"name": "NewYorker", "path": "jmhessel/newyorker_caption_contest", "config": "explanation", "split": "train"},
    ],
    "image_vqa": [
        {"name": "ScienceQA", "path": "derek-thomas/ScienceQA", "split": "train"},
    ],
    "image_editing": [
        {"name": "MagicBrush", "path": "osunlp/MagicBrush", "split": "train"},
        {"name": "InstructPix2Pix", "path": "timbrooks/instructpix2pix-clip-filtered", "split": "train"},
    ],
    "ui_to_code": [
        {"name": "WebSight", "path": "HuggingFaceM4/WebSight", "split": "train"},
    ],
    "image_prompts": [
        {"name": "SD-Prompts", "path": "Gustavosta/Stable-Diffusion-Prompts", "split": "train"},
        {"name": "Midjourney-Prompts", "path": "succinctly/midjourney-prompts", "split": "train"},
    ],
    
    # === VIDEO DATASETS ===
    "video_caption": [
        {"name": "Video-MME", "path": "lmms-lab/Video-MME", "split": "test"},
    ],
    "video_qa": [
        {"name": "VideoInstruct-100K", "path": "MBZUAI/VideoInstruct-100K", "split": "train"},
    ],
    "video_generation": [
        {"name": "Sora-Physics-Likert", "path": "Rapidata/sora-video-generation-physics-likert-scoring", "split": "train"},
        {"name": "Sora-Style-Likert", "path": "Rapidata/sora-video-generation-style-likert-scoring", "split": "train"},
        {"name": "Sora-Alignment-Likert", "path": "Rapidata/sora-video-generation-alignment-likert-scoring", "split": "train"},
        {"name": "WebVid-10M", "path": "TempoFunk/webvid-10M", "split": "train"},
        {"name": "Panda-70M", "path": "multimodalart/panda-70m", "split": "train"},
    ],
    "video_preference": [
        {"name": "T2V-Human-Preferences", "path": "Rapidata/text-2-video-human-preferences", "split": "train"},
    ],
    "image_to_video": [
        {"name": "TIP-I2V", "path": "WenhaoWang/TIP-I2V", "split": "Full"},
        {"name": "Pexels-I2V-350k", "path": "jovianzm/img2vid-pexels-350k", "split": "train"},
    ],
    
    # === AUDIO/VOICE DATASETS ===
    "voice_asr": [
        {"name": "LibriSpeech-Clean", "path": "openslr/librispeech_asr", "config": "clean", "split": "train.100"},
    ],
    "voice_tts": [
        {"name": "LibriTTS-R-Clean", "path": "blabble-io/libritts_r", "config": "clean", "split": "train.clean.100"},
        {"name": "MLS-Eng-10k", "path": "parler-tts/mls_eng_10k", "split": "train"},
        {"name": "HiFi-TTS-Clean", "path": "MikhailT/hifi-tts", "config": "clean", "split": "train"},
    ],
    "voice_emotion": [
        {"name": "VoxPopuli-EN", "path": "facebook/voxpopuli", "config": "en", "split": "train"},
        {"name": "VCTK-MultiSpeaker", "path": "sanchit-gandhi/vctk", "split": "train"},
    ],
    "voice_singing": [
        {"name": "MusicCaps", "path": "google/MusicCaps", "split": "train"},
    ],
    "voice_expressive": [
        {"name": "JennyTTS", "path": "reach-vb/jenny_tts_dataset", "split": "train"},
        {"name": "MLS-SpeakerDescriptions", "path": "parler-tts/mls-eng-speaker-descriptions", "split": "train"},
    ],
}

# Map categories to modalities
MODALITY_MAP = {
    # Text
    "code": "text",
    "conversation": "text",
    "tool_use": "text",
    "agentic": "text",
    "image_prompts": "text",
    # Image
    "image_caption": "image",
    "image_vqa": "image",
    "image_editing": "image",
    "ui_to_code": "image",
    # Video
    "video_caption": "video",
    "video_qa": "video",
    "video_generation": "video",
    "video_preference": "video",
    "image_to_video": "video",
    # Audio
    "voice_asr": "audio",
    "voice_tts": "audio",
    "voice_emotion": "audio",
    "voice_singing": "audio",
    "voice_expressive": "audio",
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_dirs():
    """Create necessary directories."""
    for d in [BASE_DIR, DOWNLOAD_DIR, OUTPUT_DIR, CACHE_DIR]:
        os.makedirs(d, exist_ok=True)
    logger.info(f"Directories created under {BASE_DIR}")


def get_file_hash(filepath: str) -> str:
    """Get MD5 hash of a file for deduplication."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_file_size_mb(filepath: str) -> float:
    """Get file size in MB."""
    return os.path.getsize(filepath) / (1024 * 1024)


def install_dependencies():
    """Install required packages."""
    packages = [
        "datasets",
        "huggingface_hub",
        "Pillow",
        "soundfile",
        "librosa",
        "opencv-python-headless",
        "requests",
        "tqdm",
        "pyarrow",
    ]
    logger.info("Installing Python dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + packages, check=True)
    
    # Install yt-dlp
    logger.info("Installing yt-dlp...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yt-dlp"], check=True)
    
    # Install ffmpeg if not available
    if not shutil.which("ffmpeg"):
        logger.info("Installing ffmpeg...")
        subprocess.run(["apt-get", "update", "-qq"], check=False)
        subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=False)


# ============================================================================
# MEDIA DOWNLOAD FUNCTIONS
# ============================================================================

def download_video_ytdlp(url: str, output_path: str, timeout: int = 180) -> Optional[str]:
    """Download video using yt-dlp."""
    try:
        if not shutil.which('yt-dlp'):
            logger.warning("yt-dlp not found")
            return None
        
        cmd = [
            'yt-dlp',
            '-f', 'best[height<=480][ext=mp4]/best[height<=480]/bestvideo[height<=480]+bestaudio/best',
            '-o', output_path,
            '--no-playlist',
            '--quiet',
            '--no-warnings',
            '--socket-timeout', '30',
            '--retries', '3',
            '--max-filesize', f'{MAX_VIDEO_SIZE_MB}M',
            url
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            if os.path.exists(output_path):
                os.remove(output_path)
            return None
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout downloading video: {url}")
        return None
    except Exception as e:
        logger.warning(f"Error downloading video {url}: {e}")
        return None


def download_video_direct(url: str, output_path: str, timeout: int = 60) -> Optional[str]:
    """Download video directly from URL."""
    try:
        import requests
        response = requests.get(url, timeout=timeout, stream=True)
        if response.status_code != 200:
            return None
        
        # Check content length
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > MAX_VIDEO_SIZE_MB * 1024 * 1024:
            logger.warning(f"Video too large: {url}")
            return None
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        return None
    except Exception as e:
        logger.warning(f"Error downloading video {url}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return None


def download_video(url: str, output_dir: str, video_id: str) -> Optional[str]:
    """Download video from URL (YouTube or direct)."""
    output_path = os.path.join(output_dir, f"{video_id}.mp4")
    
    if os.path.exists(output_path):
        return output_path
    
    # Check if it's YouTube/TikTok
    if 'youtube.com' in url or 'youtu.be' in url or 'tiktok.com' in url:
        return download_video_ytdlp(url, output_path)
    else:
        return download_video_direct(url, output_path)


def download_audio_direct(url: str, output_path: str, timeout: int = 60) -> Optional[str]:
    """Download audio directly from URL."""
    try:
        import requests
        response = requests.get(url, timeout=timeout, stream=True)
        if response.status_code != 200:
            return None
        
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > MAX_AUDIO_SIZE_MB * 1024 * 1024:
            logger.warning(f"Audio too large: {url}")
            return None
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        return None
    except Exception as e:
        logger.warning(f"Error downloading audio {url}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return None


def download_image(url: str, output_path: str, timeout: int = 30) -> Optional[str]:
    """Download image from URL."""
    try:
        import requests
        response = requests.get(url, timeout=timeout, stream=True)
        if response.status_code != 200:
            return None
        
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
            return None
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        return None
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        return None


def save_audio_bytes(audio_data: bytes, output_path: str) -> Optional[str]:
    """Save audio bytes to file."""
    try:
        with open(output_path, 'wb') as f:
            f.write(audio_data)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        return None
    except Exception:
        return None


def save_image_pil(image, output_path: str) -> Optional[str]:
    """Save PIL image to file."""
    try:
        from PIL import Image
        if isinstance(image, Image.Image):
            image.save(output_path)
            return output_path
        return None
    except Exception:
        return None


# ============================================================================
# SAMPLE EXTRACTION FUNCTIONS
# ============================================================================

def serialize_value(val: Any) -> Any:
    """Serialize a value to be JSON-compatible for the dataset."""
    if val is None:
        return None
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, (list, tuple)):
        return [serialize_value(v) for v in val]
    if isinstance(val, dict):
        return {k: serialize_value(v) for k, v in val.items()}
    # For PIL images, bytes, etc - skip (handled separately)
    return str(val) if val else None


def extract_all_metadata(sample: Dict, skip_fields: List[str] = None) -> Dict:
    """Extract ALL metadata fields from a sample, preserving original columns."""
    skip_fields = skip_fields or []
    # Fields that contain binary/media data - handle separately
    media_fields = ['image', 'img', 'audio', 'video', 'speech', 'waveform', 'photo', 'picture']
    
    metadata = {}
    for key, val in sample.items():
        if key in skip_fields:
            continue
        if key.lower() in media_fields:
            continue
        
        # Serialize the value
        try:
            serialized = serialize_value(val)
            if serialized is not None:
                metadata[key] = serialized
        except Exception:
            # If we can't serialize, convert to string
            try:
                metadata[key] = str(val)
            except:
                pass
    
    return metadata


def extract_video_url(sample: Dict) -> Optional[str]:
    """Extract video URL from sample."""
    url_fields = [
        'Video', 'video_url', 'video', 'url', 'video_link', 'mp4_url',
        'media_url', 'contentUrl', 'video1', 'column1'
    ]
    
    for field in url_fields:
        if field in sample:
            url = sample[field]
            if isinstance(url, str) and (url.startswith('http') or url.startswith('//')):
                if url.startswith('//'):
                    url = 'https:' + url
                return url
    
    # Check for YouTube video ID
    for field in ['video_id', 'youtube_id', 'clip_id', 'ytid']:
        if field in sample:
            vid_id = str(sample[field])
            if vid_id and len(vid_id) == 11:
                return f"https://www.youtube.com/watch?v={vid_id}"
    
    return None


def extract_image_data(sample: Dict) -> Tuple[Optional[Any], Optional[str]]:
    """Extract image data or URL from sample. Returns (image_data, url)."""
    image_fields = ['image', 'img', 'picture', 'photo', 'source_image', 'input_image']
    url_fields = ['image_url', 'url', 'img_url', 'picture_url']
    
    # Check for PIL image or bytes
    for field in image_fields:
        if field in sample:
            img = sample[field]
            if img is not None:
                return (img, None)
    
    # Check for URL
    for field in url_fields:
        if field in sample:
            url = sample[field]
            if isinstance(url, str) and url.startswith('http'):
                return (None, url)
    
    return (None, None)


def extract_audio_data(sample: Dict) -> Tuple[Optional[Any], Optional[str]]:
    """Extract audio data or URL from sample. Returns (audio_data, url)."""
    audio_fields = ['audio', 'speech', 'sound', 'waveform']
    
    for field in audio_fields:
        if field in sample:
            audio = sample[field]
            if audio is not None:
                # HuggingFace audio format
                if isinstance(audio, dict):
                    if 'bytes' in audio:
                        return (audio['bytes'], None)
                    if 'path' in audio and audio['path']:
                        return (None, audio['path'])
                    if 'array' in audio:
                        return (audio, None)
                return (audio, None)
    
    return (None, None)


# ============================================================================
# DATASET PROCESSING
# ============================================================================

def process_text_dataset(config: Dict, category: str, max_samples: int) -> List[Dict]:
    """Process a text dataset and extract ALL columns/metadata."""
    from datasets import load_dataset
    
    samples = []
    name = config['name']
    logger.info(f"Processing text dataset: {name}")
    
    try:
        load_kwargs = {"path": config["path"], "split": config["split"], "streaming": True}
        if "config" in config:
            load_kwargs["name"] = config["config"]
        
        ds = load_dataset(**load_kwargs)
        
        count = 0
        for sample in ds:
            if count >= max_samples:
                break
            
            # Extract ALL metadata from the sample
            sample_data = extract_all_metadata(sample)
            
            # Skip empty samples
            if not sample_data:
                continue
            
            # Add our tracking fields
            sample_data['_category'] = category
            sample_data['_source_dataset'] = name
            sample_data['_modality'] = 'text'
            
            samples.append(sample_data)
            count += 1
        
        logger.info(f"  Extracted {len(samples)} text samples from {name}")
    except Exception as e:
        logger.error(f"  Error processing {name}: {e}")
        traceback.print_exc()
    
    return samples


def process_image_dataset(config: Dict, category: str, max_samples: int, output_dir: str) -> List[Dict]:
    """Process an image dataset, download images, and extract ALL metadata."""
    from datasets import load_dataset
    from PIL import Image
    
    samples = []
    name = config['name']
    img_dir = os.path.join(output_dir, "images", name.replace(" ", "_").replace("/", "_"))
    os.makedirs(img_dir, exist_ok=True)
    
    logger.info(f"Processing image dataset: {name}")
    
    try:
        load_kwargs = {"path": config["path"], "split": config["split"], "streaming": True}
        if "config" in config:
            load_kwargs["name"] = config["config"]
        
        ds = load_dataset(**load_kwargs)
        
        count = 0
        for idx, sample in enumerate(ds):
            if count >= max_samples:
                break
            
            img_data, img_url = extract_image_data(sample)
            
            img_path = None
            if img_data is not None:
                img_filename = f"{idx:06d}.jpg"
                img_path = os.path.join(img_dir, img_filename)
                
                if isinstance(img_data, Image.Image):
                    try:
                        img_data.convert('RGB').save(img_path, 'JPEG')
                    except:
                        img_path = None
                elif isinstance(img_data, bytes):
                    try:
                        with open(img_path, 'wb') as f:
                            f.write(img_data)
                    except:
                        img_path = None
            elif img_url:
                img_filename = f"{idx:06d}.jpg"
                img_path = os.path.join(img_dir, img_filename)
                img_path = download_image(img_url, img_path)
            
            if img_path and os.path.exists(img_path):
                # Extract ALL metadata from the sample
                sample_data = extract_all_metadata(sample)
                
                # Add image path and tracking fields
                sample_data['_image_path'] = img_path
                sample_data['_category'] = category
                sample_data['_source_dataset'] = name
                sample_data['_modality'] = 'image'
                
                samples.append(sample_data)
                count += 1
        
        logger.info(f"  Extracted {len(samples)} image samples from {name}")
    except Exception as e:
        logger.error(f"  Error processing {name}: {e}")
        traceback.print_exc()
    
    return samples


def process_video_dataset(config: Dict, category: str, max_samples: int, output_dir: str) -> List[Dict]:
    """Process a video dataset, download videos, and extract ALL metadata."""
    from datasets import load_dataset
    
    samples = []
    name = config['name']
    vid_dir = os.path.join(output_dir, "videos", name.replace(" ", "_").replace("/", "_"))
    os.makedirs(vid_dir, exist_ok=True)
    
    logger.info(f"Processing video dataset: {name}")
    
    try:
        load_kwargs = {"path": config["path"], "split": config["split"], "streaming": True}
        if "config" in config:
            load_kwargs["name"] = config["config"]
        
        ds = load_dataset(**load_kwargs)
        
        count = 0
        downloaded = 0
        failed = 0
        
        for idx, sample in enumerate(ds):
            if count >= max_samples:
                break
            
            if failed > max_samples * 2:
                logger.warning(f"  Too many download failures for {name}, stopping")
                break
            
            video_url = extract_video_url(sample)
            
            if not video_url:
                continue
            
            video_id = f"{idx:06d}"
            video_path = download_video(video_url, vid_dir, video_id)
            
            if video_path and os.path.exists(video_path):
                size_mb = get_file_size_mb(video_path)
                if size_mb > MAX_VIDEO_SIZE_MB:
                    os.remove(video_path)
                    failed += 1
                    continue
                
                # Extract ALL metadata from the sample
                sample_data = extract_all_metadata(sample)
                
                # Add video path and tracking fields
                sample_data['_video_path'] = video_path
                sample_data['_category'] = category
                sample_data['_source_dataset'] = name
                sample_data['_modality'] = 'video'
                
                samples.append(sample_data)
                count += 1
                downloaded += 1
            else:
                failed += 1
        
        logger.info(f"  Extracted {len(samples)} video samples from {name} (downloaded: {downloaded}, failed: {failed})")
    except Exception as e:
        logger.error(f"  Error processing {name}: {e}")
        traceback.print_exc()
    
    return samples


def process_audio_dataset(config: Dict, category: str, max_samples: int, output_dir: str) -> List[Dict]:
    """Process an audio dataset, save audio files, and extract ALL metadata."""
    from datasets import load_dataset
    import soundfile as sf
    import numpy as np
    
    samples = []
    name = config['name']
    audio_dir = os.path.join(output_dir, "audio", name.replace(" ", "_").replace("/", "_"))
    os.makedirs(audio_dir, exist_ok=True)
    
    logger.info(f"Processing audio dataset: {name}")
    
    try:
        load_kwargs = {"path": config["path"], "split": config["split"], "streaming": True}
        if "config" in config:
            load_kwargs["name"] = config["config"]
        
        ds = load_dataset(**load_kwargs)
        
        count = 0
        for idx, sample in enumerate(ds):
            if count >= max_samples:
                break
            
            audio_data, audio_url = extract_audio_data(sample)
            
            audio_path = None
            audio_filename = f"{idx:06d}.wav"
            target_path = os.path.join(audio_dir, audio_filename)
            
            if audio_data is not None:
                try:
                    if isinstance(audio_data, bytes):
                        with open(target_path, 'wb') as f:
                            f.write(audio_data)
                        audio_path = target_path
                    elif isinstance(audio_data, dict):
                        if 'bytes' in audio_data and audio_data['bytes']:
                            with open(target_path, 'wb') as f:
                                f.write(audio_data['bytes'])
                            audio_path = target_path
                        elif 'array' in audio_data:
                            arr = np.array(audio_data['array'])
                            sr = audio_data.get('sampling_rate', 16000)
                            sf.write(target_path, arr, sr)
                            audio_path = target_path
                        elif 'path' in audio_data and audio_data['path']:
                            src = audio_data['path']
                            if os.path.exists(src):
                                shutil.copy(src, target_path)
                                audio_path = target_path
                except Exception as e:
                    logger.warning(f"  Error saving audio {idx}: {e}")
                    audio_path = None
            elif audio_url:
                audio_path = download_audio_direct(audio_url, target_path)
            
            if audio_path and os.path.exists(audio_path):
                size_mb = get_file_size_mb(audio_path)
                if size_mb > MAX_AUDIO_SIZE_MB:
                    os.remove(audio_path)
                    continue
                
                # Extract ALL metadata from the sample
                sample_data = extract_all_metadata(sample)
                
                # Add audio path and tracking fields
                sample_data['_audio_path'] = audio_path
                sample_data['_category'] = category
                sample_data['_source_dataset'] = name
                sample_data['_modality'] = 'audio'
                
                samples.append(sample_data)
                count += 1
        
        logger.info(f"  Extracted {len(samples)} audio samples from {name}")
    except Exception as e:
        logger.error(f"  Error processing {name}: {e}")
        traceback.print_exc()
    
    return samples


# ============================================================================
# MAIN DATASET BUILDER
# ============================================================================

def build_unified_dataset():
    """Build the unified multimodal dataset."""
    from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage, Audio as HFAudio
    from huggingface_hub import HfApi, login
    
    logger.info("=" * 60)
    logger.info("XORON UNIFIED MULTIMODAL DATASET BUILDER")
    logger.info("=" * 60)
    
    # Validate HF token
    if not HF_TOKEN:
        logger.error("HF_TOKEN not set!")
        logger.error("Set it via environment variable: export HF_TOKEN='your_token'")
        logger.error("Or on Kaggle: Add-ons -> Secrets -> Add HF_TOKEN")
        sys.exit(1)
    
    # Setup
    ensure_dirs()
    install_dependencies()
    
    # Login to HuggingFace
    logger.info("Logging into HuggingFace...")
    login(token=HF_TOKEN)
    
    # Collect samples by modality
    all_samples = {
        "text": [],
        "image": [],
        "video": [],
        "audio": [],
    }
    
    # Process each category
    for category, configs in DATASET_CONFIGS.items():
        modality = MODALITY_MAP.get(category, "text")
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing category: {category} (modality: {modality})")
        logger.info(f"{'='*40}")
        
        for config in configs:
            try:
                if modality == "text":
                    samples = process_text_dataset(config, category, MAX_SAMPLES_PER_DATASET)
                elif modality == "image":
                    samples = process_image_dataset(config, category, MAX_SAMPLES_PER_DATASET, OUTPUT_DIR)
                elif modality == "video":
                    samples = process_video_dataset(config, category, MAX_SAMPLES_PER_DATASET, OUTPUT_DIR)
                elif modality == "audio":
                    samples = process_audio_dataset(config, category, MAX_SAMPLES_PER_DATASET, OUTPUT_DIR)
                else:
                    samples = []
                
                all_samples[modality].extend(samples)
                
            except Exception as e:
                logger.error(f"Failed to process {config['name']}: {e}")
                traceback.print_exc()
    
    # Print statistics
    logger.info("\n" + "=" * 60)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 60)
    for modality, samples in all_samples.items():
        logger.info(f"  {modality}: {len(samples)} samples")
    
    # Create HuggingFace datasets
    logger.info("\nCreating HuggingFace datasets...")
    
    datasets_dict = {}
    
    def samples_to_dataset(samples: List[Dict], modality: str) -> Optional[Dataset]:
        """Convert samples with varying columns to a Dataset, preserving ALL columns."""
        if not samples:
            return None
        
        # Collect ALL unique keys across all samples
        all_keys = set()
        for s in samples:
            all_keys.update(s.keys())
        
        # Build column data - each column is a list
        column_data = {key: [] for key in all_keys}
        
        for s in samples:
            for key in all_keys:
                # Get value or None if missing
                val = s.get(key, None)
                # Convert complex types to JSON strings for storage
                if isinstance(val, (list, dict)):
                    try:
                        val = json.dumps(val)
                    except:
                        val = str(val)
                column_data[key].append(val)
        
        # Create dataset
        try:
            ds = Dataset.from_dict(column_data)
            return ds
        except Exception as e:
            logger.error(f"Error creating {modality} dataset: {e}")
            return None
    
    # Text split - keep ALL original columns
    if all_samples["text"]:
        ds = samples_to_dataset(all_samples["text"], "text")
        if ds:
            datasets_dict["text"] = ds
            logger.info(f"  Text dataset: {len(ds)} samples, {len(ds.column_names)} columns")
            logger.info(f"    Columns: {ds.column_names[:10]}{'...' if len(ds.column_names) > 10 else ''}")
    
    # Image split - keep ALL original columns + image
    if all_samples["image"]:
        # Filter to only samples with valid image paths
        valid_samples = [s for s in all_samples["image"] 
                        if s.get("_image_path") and os.path.exists(s.get("_image_path", ""))]
        
        if valid_samples:
            ds = samples_to_dataset(valid_samples, "image")
            if ds and "_image_path" in ds.column_names:
                # Rename _image_path to image and cast to Image type
                ds = ds.rename_column("_image_path", "image")
                ds = ds.cast_column("image", HFImage())
                datasets_dict["image"] = ds
                logger.info(f"  Image dataset: {len(ds)} samples, {len(ds.column_names)} columns")
                logger.info(f"    Columns: {ds.column_names[:10]}{'...' if len(ds.column_names) > 10 else ''}")
    
    # Audio split - keep ALL original columns + audio
    if all_samples["audio"]:
        # Filter to only samples with valid audio paths
        valid_samples = [s for s in all_samples["audio"]
                        if s.get("_audio_path") and os.path.exists(s.get("_audio_path", ""))]
        
        if valid_samples:
            ds = samples_to_dataset(valid_samples, "audio")
            if ds and "_audio_path" in ds.column_names:
                # Rename _audio_path to audio and cast to Audio type
                ds = ds.rename_column("_audio_path", "audio")
                ds = ds.cast_column("audio", HFAudio())
                datasets_dict["audio"] = ds
                logger.info(f"  Audio dataset: {len(ds)} samples, {len(ds.column_names)} columns")
                logger.info(f"    Columns: {ds.column_names[:10]}{'...' if len(ds.column_names) > 10 else ''}")
    
    # Video split - keep ALL original columns + video path
    if all_samples["video"]:
        # Filter to only samples with valid video paths
        valid_samples = [s for s in all_samples["video"]
                        if s.get("_video_path") and os.path.exists(s.get("_video_path", ""))]
        
        if valid_samples:
            ds = samples_to_dataset(valid_samples, "video")
            if ds and "_video_path" in ds.column_names:
                # Rename _video_path to video_path for clarity
                ds = ds.rename_column("_video_path", "video")
                datasets_dict["video"] = ds
                logger.info(f"  Video dataset: {len(ds)} samples, {len(ds.column_names)} columns")
                logger.info(f"    Columns: {ds.column_names[:10]}{'...' if len(ds.column_names) > 10 else ''}")
    
    if not datasets_dict:
        logger.error("No datasets created! Check for errors above.")
        return
    
    # Create DatasetDict
    final_dataset = DatasetDict(datasets_dict)
    
    # Save locally first
    local_save_path = os.path.join(OUTPUT_DIR, "xoron_unified_dataset")
    logger.info(f"\nSaving dataset locally to {local_save_path}...")
    final_dataset.save_to_disk(local_save_path)
    
    # Upload to HuggingFace
    logger.info(f"\nUploading to HuggingFace Hub: {HF_DATASET_NAME}...")
    try:
        final_dataset.push_to_hub(
            HF_DATASET_NAME,
            token=HF_TOKEN,
            private=False,
        )
        logger.info("✅ Dataset uploaded successfully!")
        logger.info(f"   View at: https://huggingface.co/datasets/{HF_DATASET_NAME}")
    except Exception as e:
        logger.error(f"Failed to upload to HuggingFace: {e}")
        traceback.print_exc()
    
    # Cleanup (optional - comment out to keep local files)
    # logger.info("Cleaning up temporary files...")
    # shutil.rmtree(BASE_DIR)
    
    logger.info("\n" + "=" * 60)
    logger.info("DATASET BUILD COMPLETE")
    logger.info("=" * 60)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    build_unified_dataset()

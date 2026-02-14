#!/usr/bin/env python3
"""
Xoron Unified Multimodal Dataset Builder

This script builds a unified HuggingFace dataset from all configured dataset sources.
It downloads actual media (audio, video, images) - not just URLs - and creates
proper splits for each modality.

Designed to run on Kaggle with /tmp storage (150GB available).

Usage (on Kaggle):
    # Build everything at once
    python build_unified_dataset.py
    
    # Build only audio first
    python build_unified_dataset.py --audio
    
    # Then add text (loads existing dataset from HF first)
    python build_unified_dataset.py --text --hf
    
    # Then add images
    python build_unified_dataset.py --image --hf
    
    # Then add videos
    python build_unified_dataset.py --video --hf
    
    # Or combine: build text and audio together
    python build_unified_dataset.py --text --audio

Flags:
    --text   : Process text datasets only
    --image  : Process image datasets only
    --video  : Process video datasets only
    --audio  : Process audio datasets only
    --hf     : Load existing dataset from HuggingFace first, then merge new data
    --all    : Process all modalities (default if no flags)

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
import argparse
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
HF_DATASET_NAME = "Backup-bdg/moe-training"

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

# Concurrent downloads (can exceed CPU count since this is I/O bound)
MAX_WORKERS = 8

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
#        {"name": "Video-MME", "path": "lmms-lab/Video-MME", "split": "test"},
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
        # LibriSpeech-Clean already uploaded
    ],
    "voice_tts": [
        # LibriTTS-R-Clean already uploaded
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


def get_video_extension(url: str) -> str:
    """Get video extension from URL, defaulting to .mp4"""
    # Supported video formats by HuggingFace
    SUPPORTED_VIDEO_EXTS = {'.mp4', '.gif', '.webm', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.ogv'}
    
    # Extract extension from URL (before query params)
    url_path = url.split('?')[0].lower()
    for ext in SUPPORTED_VIDEO_EXTS:
        if url_path.endswith(ext):
            return ext
    return '.mp4'  # Default


def download_video(url: str, output_dir: str, video_id: str) -> Optional[str]:
    """Download video from URL (YouTube, TikTok, or direct). Supports multiple formats including .gif"""
    # Get appropriate extension from URL
    ext = get_video_extension(url)
    output_path = os.path.join(output_dir, f"{video_id}{ext}")
    
    # Check if file already exists with any supported extension
    SUPPORTED_VIDEO_EXTS = ['.mp4', '.gif', '.webm', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.ogv']
    for check_ext in SUPPORTED_VIDEO_EXTS:
        check_path = os.path.join(output_dir, f"{video_id}{check_ext}")
        if os.path.exists(check_path):
            return check_path
    
    # Check if it's YouTube/TikTok - use yt-dlp (always outputs mp4)
    if 'youtube.com' in url or 'youtu.be' in url or 'tiktok.com' in url:
        mp4_path = os.path.join(output_dir, f"{video_id}.mp4")
        return download_video_ytdlp(url, mp4_path)
    else:
        # Try direct download first (preserves original format)
        result = download_video_direct(url, output_path)
        if result:
            return result
        # Fallback to yt-dlp for other sites it supports
        mp4_path = os.path.join(output_dir, f"{video_id}.mp4")
        return download_video_ytdlp(url, mp4_path)


# Vript metadata cache for resolving video IDs to URLs
_vript_meta_cache = None

def get_vript_video_url(video_id: str) -> Optional[str]:
    """Look up video URL from Vript metadata (for Vript dataset)."""
    global _vript_meta_cache
    
    if _vript_meta_cache is None:
        _vript_meta_cache = {}
        try:
            import requests
            # Load Vript short videos metadata
            short_url = "https://huggingface.co/datasets/Mutonix/Vript/resolve/main/vript_meta/vript_short_videos_meta.json"
            resp = requests.get(short_url, timeout=60)
            if resp.status_code == 200:
                _vript_meta_cache.update(resp.json())
                logger.info(f"  Loaded Vript metadata: {len(_vript_meta_cache)} entries")
        except Exception as e:
            logger.warning(f"  Could not load Vript metadata: {e}")
    
    if video_id in _vript_meta_cache:
        meta = _vript_meta_cache[video_id]
        return meta.get('original_url') or meta.get('webpage_url')
    
    return None


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
# SAMPLE EXTRACTION - MATCHING YOUR EXISTING CODE'S EXPECTED COLUMNS
# ============================================================================

# Standardized column names that match your formatters.py and dataset.py:
#
# TEXT: instruction, response, system, conversations, context, category, source
#
# IMAGE: image, caption, question, answer, choices, category, source
#   - caption: from caption, text, caption_1, image_description, image_uncanny_description
#   - question: from question
#   - answer: from answer (can be int index into choices)
#   - choices: from choices (list)
#
# AUDIO: audio, text, speaker_id, sampling_rate, gender, age, language, emotion, category, source
#   - text: from text, transcript, transcription, sentence, normalized_text, text_normalized, raw_text
#   - speaker_id: from speaker_id, speaker, spk_id, speaker_name (also extract from path)
#   - sampling_rate: from sampling_rate, sample_rate
#   - gender: from gender, sex
#   - age: from age
#   - language: from language, lang, locale
#   - emotion: from emotion, label, emotion_label
#
# VIDEO: video, caption, prompt, question, answer, options, duration, category, source
#   - caption: from caption, text, description, name, page_dir
#   - prompt: from Prompt, prompt, caption, text, description
#   - question: from question
#   - answer: from answer, a
#   - options: from options (list like ["A. text", "B. text"])
#   - duration: from duration

def get_field(sample: Dict, field_names: List[str], default=None):
    """Get first matching field from sample."""
    for name in field_names:
        if name in sample and sample[name] is not None:
            val = sample[name]
            if isinstance(val, str) and val.strip():
                return val.strip()
            elif not isinstance(val, str) and val:
                return val
    return default


def extract_text_sample(sample: Dict, category: str, source: str) -> Optional[Dict]:
    """Extract text sample matching your code's expected format."""
    result = {
        "instruction": None,
        "response": None,
        "system": None,
        "conversations": None,
        "context": None,
        "category": category,
        "source": source,
    }
    
    # Instruction/prompt fields
    result["instruction"] = get_field(sample, [
        'instruction', 'prompt', 'query', 'question', 'input', 'user', 'human', 'problem', 'task'
    ])
    
    # Response/answer fields  
    result["response"] = get_field(sample, [
        'response', 'output', 'answer', 'completion', 'assistant', 'gpt', 'solution', 'code', 'canonical_solution'
    ])
    
    # System prompt
    result["system"] = get_field(sample, ['system', 'system_prompt', 'system_message'])
    
    # Context/document
    result["context"] = get_field(sample, ['context', 'document', 'passage', 'text', 'description'])
    
    # Conversations (multi-turn)
    convs = get_field(sample, ['conversations', 'messages', 'dialogue', 'chat'])
    if convs and isinstance(convs, list):
        try:
            result["conversations"] = json.dumps(convs)
        except:
            result["conversations"] = str(convs)
        
        # Extract first turn as instruction/response if not already set
        if not result["instruction"] and len(convs) >= 1:
            first = convs[0]
            if isinstance(first, dict):
                result["instruction"] = first.get('content', first.get('value', first.get('text', '')))
        if not result["response"] and len(convs) >= 2:
            second = convs[1]
            if isinstance(second, dict):
                result["response"] = second.get('content', second.get('value', second.get('text', '')))
    
    # Skip empty
    if not result["instruction"] and not result["response"] and not result["conversations"] and not result["context"]:
        return None
    
    return result


def extract_image_sample(sample: Dict, category: str, source: str) -> Tuple[Optional[Dict], Optional[Any], Optional[str]]:
    """Extract image sample matching your formatters (format_image_caption_sample, format_image_vqa_sample)."""
    result = {
        "caption": None,
        "question": None, 
        "answer": None,
        "choices": None,
        "category": category,
        "source": source,
    }
    
    # Caption fields (matching format_image_caption_sample)
    result["caption"] = get_field(sample, [
        'caption', 'text', 'caption_1', 'image_description', 'image_uncanny_description',
        'description', 'label', 'title', 'sentence'
    ])
    
    # VQA fields (matching format_image_vqa_sample)
    result["question"] = get_field(sample, ['question', 'query', 'prompt'])
    result["answer"] = get_field(sample, ['answer', 'response', 'label', 'output'])
    
    # Choices for multiple choice VQA
    choices = get_field(sample, ['choices', 'options', 'candidates'])
    if choices:
        try:
            result["choices"] = json.dumps(choices) if isinstance(choices, list) else str(choices)
        except:
            result["choices"] = str(choices)
    
    # Extract image data
    image_data = None
    image_url = None
    
    for field in ['image', 'img', 'picture', 'photo', 'source_image', 'input_image']:
        if field in sample and sample[field] is not None:
            val = sample[field]
            # Skip integers, floats, and plain strings (these are IDs or paths, not image data)
            # Valid image data types: PIL Image, bytes, dict with 'bytes'/'path' key, or objects with 'save' method
            if isinstance(val, (int, float)):
                continue
            if isinstance(val, str) and not val.startswith('http'):
                # It's a local path, not actual image data - skip
                continue
            if isinstance(val, dict):
                # Check if it's a valid HuggingFace image dict
                if 'bytes' in val or 'path' in val:
                    image_data = val
                    break
                # Skip other dict types
                continue
            # PIL Image, bytes, or object with save method
            image_data = val
            break
    
    # If no valid image data, try to get URL
    if image_data is None:
        for field in ['image_url', 'url', 'img_url', 'picture_url']:
            if field in sample:
                url = sample[field]
                if isinstance(url, str) and url.startswith('http'):
                    image_url = url
                    break
    
    return result, image_data, image_url


def extract_audio_sample(sample: Dict, category: str, source: str) -> Tuple[Optional[Dict], Optional[Any], Optional[str]]:
    """
    Extract audio sample matching your formatters:
    - format_voice_asr_sample: uses text, transcript, transcription, sentence
    - format_voice_tts_sample: uses text_normalized, text, transcript, normalized_text, sentence
                               speaker_id, speaker, spk_id, speaker_name (+ path extraction)
    - format_voice_emotion_sample: uses text, raw_text, normalized_text, transcript
                                   speaker_id, audio_id, gender, age, language
                                   emotion, label, emotion_label, arousal, valence, dominance
    """
    result = {
        # Transcription - ALL the field names your code checks
        "text": None,  # Primary field your code uses
        
        # Speaker info - ALL fields your code checks  
        "speaker_id": None,  # Unified: speaker_id, speaker, spk_id, speaker_name all -> speaker_id
        
        # Audio metadata
        "sampling_rate": None,
        "gender": None,
        "age": None,
        "language": None,
        
        # Emotion fields (for voice_emotion)
        "emotion": None,
        "arousal": None,
        "valence": None,
        "dominance": None,
        
        # Tracking
        "category": category,
        "source": source,
    }
    
    # Text/transcription - try all fields your code uses
    result["text"] = get_field(sample, [
        'text', 'text_normalized', 'normalized_text', 'transcript', 'transcription', 
        'sentence', 'raw_text', 'caption', 'description'
    ])
    
    # Speaker ID - unify all variations to speaker_id
    speaker = get_field(sample, ['speaker_id', 'speaker', 'spk_id', 'speaker_name', 'audio_id', 'client_id'])
    
    # Also try to extract from audio path (LibriTTS format: speaker/chapter/utterance)
    if speaker is None:
        audio_info = sample.get('audio', {})
        if isinstance(audio_info, dict):
            audio_path = audio_info.get('path', '')
            if isinstance(audio_path, str) and '/' in audio_path:
                parts = audio_path.split('/')
                if len(parts) >= 3:
                    speaker = parts[-3]
                elif len(parts) >= 2:
                    speaker = parts[-2]
    
    if speaker is not None:
        result["speaker_id"] = str(speaker)
    
    # Sampling rate
    sr = get_field(sample, ['sampling_rate', 'sample_rate'])
    if sr is not None:
        try:
            result["sampling_rate"] = int(sr)
        except:
            pass
    
    # Gender, age, language
    result["gender"] = get_field(sample, ['gender', 'sex'])
    result["age"] = get_field(sample, ['age'])
    result["language"] = get_field(sample, ['language', 'lang', 'locale'])
    
    # Emotion fields
    result["emotion"] = get_field(sample, ['emotion', 'label', 'emotion_label'])
    
    arousal = get_field(sample, ['arousal', 'activation'])
    if arousal is not None:
        try:
            result["arousal"] = float(arousal)
        except:
            pass
    
    valence = get_field(sample, ['valence', 'pleasure'])
    if valence is not None:
        try:
            result["valence"] = float(valence)
        except:
            pass
            
    dominance = get_field(sample, ['dominance', 'power'])
    if dominance is not None:
        try:
            result["dominance"] = float(dominance)
        except:
            pass
    
    # Extract audio data
    audio_data = None
    audio_path = None
    
    for field in ['audio', 'speech', 'sound', 'waveform']:
        if field in sample and sample[field] is not None:
            audio = sample[field]
            if isinstance(audio, dict):
                # Get sampling_rate from audio dict
                if result["sampling_rate"] is None and 'sampling_rate' in audio:
                    result["sampling_rate"] = audio['sampling_rate']
                
                if 'bytes' in audio and audio['bytes']:
                    audio_data = audio['bytes']
                    break
                elif 'array' in audio and audio['array'] is not None:
                    audio_data = audio
                    break
                elif 'path' in audio and audio['path']:
                    # Return the full dict so we can handle path (URL or local)
                    audio_data = audio
                    break
            else:
                audio_data = audio
                break
    
    return result, audio_data, audio_path


def extract_video_sample(sample: Dict, category: str, source: str) -> Tuple[Optional[Dict], Optional[Any], Optional[str]]:
    """
    Extract video sample matching your formatters:
    - format_video_caption_sample: question, options, answer/a, domain, sub_category, duration
    - format_video_generation_sample: caption, text, description, Prompt, prompt, name, page_dir
    - format_image_to_video_sample: Text_Prompt, prompt, short_caption, dense_caption, 
                                     Brief Description, Detailed Description, etc.
    
    Returns: (metadata_dict, video_data, video_url)
    - video_data: bytes, dict with 'bytes'/'path', or None
    - video_url: URL string or None
    """
    result = {
        # Caption/prompt fields
        "caption": None,
        "prompt": None,
        
        # QA fields (Video-MME style)
        "question": None,
        "answer": None,
        "options": None,  # List like ["A. text", "B. text"]
        
        # Metadata
        "duration": None,
        "domain": None,
        "sub_category": None,
        
        # Tracking
        "category": category,
        "source": source,
    }
    
    # Caption - for video description
    result["caption"] = get_field(sample, [
        'caption', 'text', 'description', 'title', 'name', 'short_caption', 'dense_caption',
        'Brief Description', 'Detailed Description', 'Summarized Description'
    ])
    
    # Handle caption dict (Vript format)
    if result["caption"] is None:
        caption_obj = sample.get('caption')
        if isinstance(caption_obj, dict):
            result["caption"] = caption_obj.get('content', caption_obj.get('shot_type', ''))
    
    # Pexels format: extract caption from image filename (column0)
    if result["caption"] is None and sample.get("column0") and sample.get("column1"):
        col0 = str(sample.get("column0", ""))
        if col0 != "thumbnail_loc" and ('pexels.com' in col0 or 'images.pexels' in col0):
            filename = col0.split('/')[-1].replace('.jpeg', '').replace('.jpg', '').replace('.png', '')
            parts = filename.split('-')[:-1]  # Remove ID at end
            if parts:
                result["caption"] = ' '.join(parts)
    
    # Prompt - for generation
    result["prompt"] = get_field(sample, [
        'Prompt', 'prompt', 'Text_Prompt', 'instruction', 'query'
    ])
    
    # QA fields (Video-MME format, VideoInstruct-100K uses 'q'/'a')
    result["question"] = get_field(sample, ['question', 'q', 'query'])
    result["answer"] = get_field(sample, ['answer', 'a'])
    
    # Options list
    options = get_field(sample, ['options', 'choices'])
    if options:
        try:
            result["options"] = json.dumps(options) if isinstance(options, list) else str(options)
        except:
            result["options"] = str(options)
    
    # Metadata
    result["domain"] = get_field(sample, ['domain', 'category'])
    result["sub_category"] = get_field(sample, ['sub_category', 'subcategory'])
    
    duration = get_field(sample, ['duration', 'length', 'video_length'])
    if duration is not None:
        try:
            result["duration"] = str(duration)
        except:
            pass
    
    # Extract video data (bytes/dict) and URL
    video_data = None
    video_url = None
    
    # First try to get actual video data (bytes or dict with bytes/path)
    for field in ['video', 'Video', 'clip', 'media']:
        if field in sample and sample[field] is not None:
            val = sample[field]
            # Skip integers, floats (these are IDs, not video data)
            if isinstance(val, (int, float)):
                continue
            # Check for dict with bytes or path (HuggingFace Video format)
            if isinstance(val, dict):
                if 'bytes' in val or 'path' in val:
                    video_data = val
                    break
            # Raw bytes
            elif isinstance(val, bytes):
                video_data = val
                break
            # String could be URL or path
            elif isinstance(val, str):
                if val.startswith('http') or val.startswith('//'):
                    video_url = 'https:' + val if val.startswith('//') else val
                # Skip local paths for now, will be handled below
                continue
    
    # If no video data, try to get URL
    if video_data is None and video_url is None:
        # Direct URL fields (matches dataset.py)
        for field in ['Video', 'video_url', 'video1', 'column1', 'contentUrl', 'url', 'video_link', 'mp4_url', 'gif_url', 'media_url']:
            if field in sample:
                url = sample[field]
                if isinstance(url, str) and (url.startswith('http') or url.startswith('//')):
                    # Skip placeholder values
                    if url not in ['content_loc', 'thumbnail_loc', 'url', 'video_url']:
                        video_url = 'https:' + url if url.startswith('//') else url
                        break
    
    # clip_id -> YouTube URL (with alphanumeric validation like dataset.py)
    if video_data is None and video_url is None:
        if "clip_id" in sample:
            clip_id = sample["clip_id"]
            if clip_id and isinstance(clip_id, str) and len(clip_id) == 11:
                if clip_id.replace('-', '').replace('_', '').isalnum():
                    video_url = f"https://www.youtube.com/watch?v={clip_id}"
    
    # video_id -> YouTube/TikTok URL
    if video_data is None and video_url is None:
        if "video_id" in sample:
            vid_id = sample["video_id"]
            if vid_id and isinstance(vid_id, str):
                # Strip "v_" prefix (ActivityNet/VideoInstruct-100K format)
                if vid_id.startswith("v_"):
                    video_url = f"https://www.youtube.com/watch?v={vid_id[2:]}"
                elif len(vid_id) == 11:
                    video_url = f"https://www.youtube.com/watch?v={vid_id}"
                elif vid_id.isdigit() and len(vid_id) > 15:
                    # TikTok ID (long numeric)
                    video_url = f"https://www.tiktok.com/@user/video/{vid_id}"
    
    # videoID -> YouTube URL
    if video_data is None and video_url is None:
        if "videoID" in sample:
            vid_id = sample["videoID"]
            if vid_id and isinstance(vid_id, str) and len(vid_id) == 11:
                video_url = f"https://www.youtube.com/watch?v={vid_id}"
    
    # Vript meta.video_id (nested field)
    if video_data is None and video_url is None:
        if "meta" in sample and isinstance(sample.get("meta"), dict):
            meta = sample["meta"]
            if "video_id" in meta:
                vid_id = str(meta["video_id"])
                # Try Vript metadata first
                vript_url = get_vript_video_url(vid_id)
                if vript_url:
                    video_url = vript_url
                elif len(vid_id) == 11:
                    video_url = f"https://www.youtube.com/watch?v={vid_id}"
                elif vid_id.isdigit() and len(vid_id) > 15:
                    # TikTok ID (long numeric)
                    video_url = f"https://www.tiktok.com/@user/video/{vid_id}"
    
    # Fallback: youtube_id, ytid fields
    if video_data is None and video_url is None:
        for field in ['youtube_id', 'ytid']:
            if field in sample:
                vid_id = str(sample[field])
                if vid_id and len(vid_id) == 11:
                    video_url = f"https://www.youtube.com/watch?v={vid_id}"
                    break
    
    return result, video_data, video_url


# ============================================================================
# DATASET PROCESSING - CLEAN STANDARDIZED OUTPUT
# ============================================================================

def process_text_dataset(config: Dict, category: str, max_samples: int) -> List[Dict]:
    """Process a text dataset with standardized schema."""
    from datasets import load_dataset
    
    samples = []
    name = config['name']
    logger.info(f"Processing text dataset: {name}")
    
    try:
        # Pull exactly max_samples in one batch (no per-sample HTTP requests)
        split_str = f"{config['split']}[:{max_samples}]"
        load_kwargs = {"path": config["path"], "split": split_str, "streaming": False}
        if "config" in config:
            load_kwargs["name"] = config["config"]
        
        ds = load_dataset(**load_kwargs)
        
        count = 0
        for sample in ds:
            if count >= max_samples:
                break
            
            # Extract with standardized schema
            extracted = extract_text_sample(sample, category, name)
            if extracted:
                samples.append(extracted)
                count += 1
        
        logger.info(f"  ✓ {len(samples)} samples from {name}")
    except Exception as e:
        logger.error(f"  ✗ Error processing {name}: {e}")
        traceback.print_exc()
    
    return samples


def process_image_dataset(config: Dict, category: str, max_samples: int, output_dir: str) -> List[Dict]:
    """Process an image dataset with standardized schema."""
    from datasets import load_dataset
    from PIL import Image
    
    samples = []
    name = config['name']
    img_dir = os.path.join(output_dir, "images", name.replace(" ", "_").replace("/", "_"))
    os.makedirs(img_dir, exist_ok=True)
    
    logger.info(f"Processing image dataset: {name}")
    
    try:
        # Use STREAMING to avoid downloading entire dataset
        load_kwargs = {"path": config["path"], "split": config['split'], "streaming": True}
        if "config" in config:
            load_kwargs["name"] = config["config"]
        
        ds = load_dataset(**load_kwargs)
        logger.info(f"  📥 Streaming mode: will fetch only {max_samples} samples")
        
        count = 0
        for idx, sample in enumerate(ds):
            if count >= max_samples:
                break
            
            # Extract with standardized schema
            metadata, img_data, img_url = extract_image_sample(sample, category, name)
            
            # Save image
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
                # Handle dict with 'bytes' key (HuggingFace Image format)
                elif isinstance(img_data, dict) and 'bytes' in img_data and img_data['bytes'] is not None:
                    try:
                        with open(img_path, 'wb') as f:
                            f.write(img_data['bytes'])
                    except:
                        img_path = None
                # Handle dict with 'path' key
                elif isinstance(img_data, dict) and 'path' in img_data and img_data['path']:
                    try:
                        src_path = img_data['path']
                        if os.path.exists(src_path):
                            shutil.copy(src_path, img_path)
                        else:
                            img_path = None
                    except:
                        img_path = None
                elif hasattr(img_data, 'save'):
                    try:
                        img_data.save(img_path)
                    except:
                        img_path = None
                # Skip integers and other non-image types
                elif isinstance(img_data, (int, float, str)):
                    img_path = None
            if img_path is None and img_url:
                img_filename = f"{idx:06d}.jpg"
                img_path = os.path.join(img_dir, img_filename)
                img_path = download_image(img_url, img_path)
            
            if img_path and os.path.exists(img_path):
                metadata['image_path'] = img_path
                samples.append(metadata)
                count += 1
        
        logger.info(f"  ✓ {len(samples)} samples from {name}")
    except Exception as e:
        logger.error(f"  ✗ Error processing {name}: {e}")
        traceback.print_exc()
    
    return samples


def get_video_ext_from_path(path: str) -> str:
    """Get video extension from path, defaulting to .mp4"""
    SUPPORTED_VIDEO_EXTS = ['.mp4', '.gif', '.webm', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.ogv']
    path_lower = path.lower()
    for ext in SUPPORTED_VIDEO_EXTS:
        if path_lower.endswith(ext):
            return ext
    return '.mp4'


def detect_video_format_from_bytes(video_bytes: bytes) -> str:
    """
    Detect video format from magic bytes.
    Custom video format detection - avoids torchcodec dependency.
    
    Supports: MP4, GIF, WebM, AVI, MOV, MKV, FLV, WMV
    
    Args:
        video_bytes: Raw video bytes
        
    Returns:
        File extension string (e.g., '.mp4', '.gif')
    """
    if len(video_bytes) < 12:
        return '.mp4'  # Default for too-short data
    
    # GIF: starts with GIF87a or GIF89a
    if video_bytes[:6] in (b'GIF87a', b'GIF89a') or video_bytes[:4] == b'GIF8':
        return '.gif'
    
    # WebM/MKV: EBML header (0x1A 0x45 0xDF 0xA3)
    if video_bytes[:4] == b'\x1a\x45\xdf\xa3':
        # Check for WebM doctype vs MKV
        if b'webm' in video_bytes[:64].lower():
            return '.webm'
        return '.mkv'
    
    # MP4/MOV: ftyp box (typically at byte 4)
    if video_bytes[4:8] == b'ftyp':
        ftyp_brand = video_bytes[8:12]
        # QuickTime MOV
        if ftyp_brand in (b'qt  ', b'MSNV'):
            return '.mov'
        # MP4 variants
        return '.mp4'
    
    # AVI: RIFF....AVI
    if video_bytes[:4] == b'RIFF' and video_bytes[8:12] == b'AVI ':
        return '.avi'
    
    # FLV: FLV header
    if video_bytes[:3] == b'FLV':
        return '.flv'
    
    # WMV/ASF: ASF header GUID
    if video_bytes[:16] == b'\x30\x26\xb2\x75\x8e\x66\xcf\x11\xa6\xd9\x00\xaa\x00\x62\xce\x6c':
        return '.wmv'
    
    # OGV/OGG: OggS
    if video_bytes[:4] == b'OggS':
        return '.ogv'
    
    # Default to MP4 for unknown formats
    return '.mp4'


def save_video_bytes(video_bytes: bytes, target_path: str) -> Optional[str]:
    """
    Save raw video bytes to file with proper extension detection.
    Custom video handler - avoids torchcodec dependency entirely.
    
    Supports all major video formats: MP4, GIF, WebM, AVI, MOV, MKV, FLV, WMV, OGV
    
    Args:
        video_bytes: Raw video bytes
        target_path: Base path to save (extension will be adjusted based on format)
        
    Returns:
        Path to saved video file or None if failed
    """
    if not video_bytes or len(video_bytes) == 0:
        return None
    
    try:
        # Detect format and get proper extension
        detected_ext = detect_video_format_from_bytes(video_bytes)
        
        # Adjust target path with correct extension
        base_path = os.path.splitext(target_path)[0]
        final_path = base_path + detected_ext
        
        # Write bytes directly - no transcoding needed
        with open(final_path, 'wb') as f:
            f.write(video_bytes)
        
        # Verify file was written
        if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
            return final_path
        
        return None
    except Exception:
        return None


def _download_video_task(task: Dict) -> Dict:
    """
    Worker function for parallel video downloading.
    Downloads a single video and returns result dict.
    """
    video_url = task['video_url']
    vid_dir = task['vid_dir']
    video_id = task['video_id']
    metadata = task['metadata']
    
    try:
        video_path = download_video(video_url, vid_dir, video_id)
        if video_path and os.path.exists(video_path):
            size_mb = get_file_size_mb(video_path)
            if size_mb > MAX_VIDEO_SIZE_MB:
                os.remove(video_path)
                return {'success': False, 'metadata': None, 'video_id': video_id}
            metadata['video_path'] = video_path
            return {'success': True, 'metadata': metadata, 'video_id': video_id}
    except Exception:
        pass
    return {'success': False, 'metadata': None, 'video_id': video_id}


def process_video_dataset(config: Dict, category: str, max_samples: int, output_dir: str) -> List[Dict]:
    """
    Process a video dataset with standardized schema.
    
    Uses custom video handling (NOT torchcodec) to avoid dependency issues.
    Supports all major formats: MP4, GIF, WebM, AVI, MOV, MKV, FLV, WMV, OGV.
    
    Uses parallel downloading for URL-based videos to maximize throughput.
    Disables HuggingFace's automatic video decoding to prevent torchcodec usage.
    """
    from datasets import load_dataset, Video
    
    samples = []
    name = config['name']
    vid_dir = os.path.join(output_dir, "videos")
    os.makedirs(vid_dir, exist_ok=True)
    
    logger.info(f"Processing video dataset: {name}")
    
    try:
        # Use streaming to avoid downloading entire dataset
        load_kwargs = {"path": config["path"], "split": config['split'], "streaming": True}
        if "config" in config:
            load_kwargs["name"] = config["config"]
        
        ds = load_dataset(**load_kwargs)
        
        # Phase 1: Collect samples - process embedded data immediately, queue URL downloads
        embedded_samples = []  # Samples with embedded video data (processed immediately)
        url_download_tasks = []  # Tasks for parallel URL downloading
        
        collected = 0
        embedded_count = 0
        failed_embedded = 0
        idx = 0
        
        # Collect up to max_samples * 1.5 candidates (some URL downloads will fail)
        max_candidates = int(max_samples * 1.5) + 100
        
        logger.info(f"  Phase 1: Collecting video samples (target: {max_samples})...")
        
        for sample in ds:
            if collected >= max_candidates:
                break
            
            # Extract with standardized schema
            metadata, video_data, video_url = extract_video_sample(sample, category, name)
            
            video_id = f"{name.replace(' ', '_')}_{idx:06d}"
            base_target = os.path.join(vid_dir, video_id)
            video_path = None
            
            # Try to save embedded video data directly
            if video_data is not None:
                try:
                    if isinstance(video_data, dict) and 'bytes' in video_data and video_data['bytes'] is not None:
                        video_path = save_video_bytes(video_data['bytes'], base_target + '.mp4')
                    elif isinstance(video_data, dict) and 'path' in video_data and video_data['path']:
                        src_path = video_data['path']
                        if os.path.exists(src_path):
                            ext = get_video_ext_from_path(src_path)
                            target_path = base_target + ext
                            shutil.copy(src_path, target_path)
                            video_path = target_path
                    elif isinstance(video_data, bytes):
                        video_path = save_video_bytes(video_data, base_target + '.mp4')
                except Exception:
                    video_path = None
            
            if video_path and os.path.exists(video_path):
                size_mb = get_file_size_mb(video_path)
                if size_mb <= MAX_VIDEO_SIZE_MB:
                    metadata['video_path'] = video_path
                    embedded_samples.append(metadata)
                    embedded_count += 1
                    collected += 1
                else:
                    os.remove(video_path)
                    failed_embedded += 1
            elif video_url:
                # Queue for parallel download
                url_download_tasks.append({
                    'video_url': video_url,
                    'vid_dir': vid_dir,
                    'video_id': video_id,
                    'metadata': metadata
                })
                collected += 1
            else:
                failed_embedded += 1
            
            idx += 1
            
            # Progress logging every 1000 samples
            if idx % 1000 == 0:
                logger.info(f"    Scanned {idx} samples, collected {len(embedded_samples)} embedded + {len(url_download_tasks)} URLs queued")
        
        logger.info(f"  Phase 1 complete: {len(embedded_samples)} embedded, {len(url_download_tasks)} URLs to download")
        
        # Add embedded samples first
        samples.extend(embedded_samples)
        
        # Phase 2: Parallel URL downloads
        if url_download_tasks and len(samples) < max_samples:
            remaining_needed = max_samples - len(samples)
            tasks_to_process = url_download_tasks[:int(remaining_needed * 1.3) + 50]  # Extra for failures
            
            logger.info(f"  Phase 2: Downloading {len(tasks_to_process)} videos in parallel (workers: {MAX_WORKERS})...")
            
            downloaded = 0
            failed_downloads = 0
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all tasks
                future_to_task = {executor.submit(_download_video_task, task): task for task in tasks_to_process}
                
                # Process completed downloads
                for future in as_completed(future_to_task):
                    if len(samples) >= max_samples:
                        # Cancel remaining futures
                        for f in future_to_task:
                            f.cancel()
                        break
                    
                    try:
                        result = future.result(timeout=300)  # 5 min timeout per download
                        if result['success'] and result['metadata']:
                            samples.append(result['metadata'])
                            downloaded += 1
                        else:
                            failed_downloads += 1
                    except Exception:
                        failed_downloads += 1
                    
                    # Progress logging
                    total_processed = downloaded + failed_downloads
                    if total_processed % 50 == 0:
                        logger.info(f"    Progress: {downloaded} downloaded, {failed_downloads} failed, {len(samples)}/{max_samples} total")
            
            logger.info(f"  Phase 2 complete: {downloaded} downloaded, {failed_downloads} failed")
        
        # Trim to max_samples if we got more
        if len(samples) > max_samples:
            samples = samples[:max_samples]
        
        logger.info(f"  ✓ {len(samples)} samples from {name}")
        
    except Exception as e:
        logger.error(f"  ✗ Error processing {name}: {e}")
        traceback.print_exc()
    
    return samples


def decode_audio_bytes(audio_bytes: bytes, target_path: str) -> Optional[str]:
    """
    Decode audio bytes using soundfile/librosa (NOT torchcodec).
    This is our custom audio decoder that avoids torchcodec dependency.
    
    Args:
        audio_bytes: Raw audio bytes
        target_path: Path to save the decoded audio
        
    Returns:
        Path to saved audio file or None if failed
    """
    import io
    import numpy as np
    
    # Try soundfile first (preferred - fast and reliable)
    try:
        import soundfile as sf
        audio_buffer = io.BytesIO(audio_bytes)
        array, sr = sf.read(audio_buffer)
        
        # Ensure float32 and handle stereo -> mono
        array = np.array(array, dtype=np.float32)
        if array.ndim > 1:
            array = array.mean(axis=1)  # Convert stereo to mono
        
        sf.write(target_path, array, sr)
        return target_path
    except Exception:
        pass
    
    # Try librosa as fallback (handles more formats)
    try:
        import librosa
        audio_buffer = io.BytesIO(audio_bytes)
        array, sr = librosa.load(audio_buffer, sr=None, mono=True)
        
        import soundfile as sf
        sf.write(target_path, array, sr)
        return target_path
    except Exception:
        pass
    
    return None


def process_audio_dataset(config: Dict, category: str, max_samples: int, output_dir: str) -> List[Dict]:
    """
    Process an audio dataset with standardized schema.
    
    Uses STREAMING to avoid downloading entire dataset - only fetches max_samples.
    Uses custom audio decoding (soundfile/librosa) instead of torchcodec.
    """
    from datasets import load_dataset, Audio
    import soundfile as sf
    import numpy as np
    
    samples = []
    name = config['name']
    audio_dir = os.path.join(output_dir, "audio", name.replace(" ", "_").replace("/", "_"))
    os.makedirs(audio_dir, exist_ok=True)
    
    logger.info(f"Processing audio dataset: {name}")
    
    try:
        # Use STREAMING to avoid downloading entire dataset
        load_kwargs = {"path": config["path"], "split": config['split'], "streaming": True}
        if "config" in config:
            load_kwargs["name"] = config["config"]
        
        ds = load_dataset(**load_kwargs)
        logger.info(f"  📥 Streaming mode: will fetch only {max_samples} samples")
        
        count = 0
        for idx, sample in enumerate(ds):
            if count >= max_samples:
                break
            
            # Extract with standardized schema
            metadata, audio_data, audio_src_path = extract_audio_sample(sample, category, name)
            
            audio_path = None
            audio_filename = f"{idx:06d}.wav"
            target_path = os.path.join(audio_dir, audio_filename)
            
            if audio_data is not None:
                try:
                    # Handle bytes (from decode=False or raw bytes)
                    if isinstance(audio_data, bytes):
                        audio_path = decode_audio_bytes(audio_data, target_path)
                    
                    # Handle dict with 'bytes' key (from Audio(decode=False))
                    elif isinstance(audio_data, dict) and 'bytes' in audio_data and audio_data['bytes'] is not None:
                        audio_path = decode_audio_bytes(audio_data['bytes'], target_path)
                    
                    # Handle dict with 'array' key (if decode=True was used somehow)
                    elif isinstance(audio_data, dict) and 'array' in audio_data:
                        arr = np.array(audio_data['array'], dtype=np.float32)
                        sr = audio_data.get('sampling_rate', 16000)
                        # Handle stereo -> mono
                        if arr.ndim > 1:
                            arr = arr.mean(axis=1)
                        sf.write(target_path, arr, sr)
                        audio_path = target_path
                    
                    # Handle dict with 'path' key (local or URL)
                    elif isinstance(audio_data, dict) and 'path' in audio_data and audio_data['path']:
                        src_path = audio_data['path']
                        if isinstance(src_path, str) and src_path.startswith('http'):
                            # Download from URL
                            audio_path = download_audio_direct(src_path, target_path)
                        elif os.path.exists(src_path):
                            shutil.copy(src_path, target_path)
                            audio_path = target_path
                
                except Exception as e:
                    logger.warning(f"  Error saving audio {idx}: {e}")
                    audio_path = None
            
            elif audio_src_path:
                try:
                    if isinstance(audio_src_path, str) and audio_src_path.startswith('http'):
                        # Download from URL
                        audio_path = download_audio_direct(audio_src_path, target_path)
                    elif os.path.exists(audio_src_path):
                        shutil.copy(audio_src_path, target_path)
                        audio_path = target_path
                except:
                    audio_path = None
            
            if audio_path and os.path.exists(audio_path):
                size_mb = get_file_size_mb(audio_path)
                if size_mb > MAX_AUDIO_SIZE_MB:
                    os.remove(audio_path)
                    continue
                
                metadata['audio_path'] = audio_path
                samples.append(metadata)
                count += 1
        
        logger.info(f"  ✓ {len(samples)} samples from {name}")
    except Exception as e:
        logger.error(f"  ✗ Error processing {name}: {e}")
        traceback.print_exc()
    
    return samples


# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build unified multimodal dataset for Xoron",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build everything at once
    python build_unified_dataset.py
    
    # Build only audio first
    python build_unified_dataset.py --audio
    
    # Then add text (loads existing dataset from HF first)
    python build_unified_dataset.py --text --hf
    
    # Build multiple modalities
    python build_unified_dataset.py --text --audio
        """
    )
    
    parser.add_argument("--text", action="store_true", help="Process text datasets")
    parser.add_argument("--image", action="store_true", help="Process image datasets")
    parser.add_argument("--video", action="store_true", help="Process video datasets")
    parser.add_argument("--audio", action="store_true", help="Process audio datasets")
    parser.add_argument("--hf", action="store_true", help="Load existing dataset from HuggingFace first, then merge")
    parser.add_argument("--all", action="store_true", help="Process all modalities (default)")
    
    args = parser.parse_args()
    
    # If no modality flags specified, default to all
    if not any([args.text, args.image, args.video, args.audio, args.all]):
        args.all = True
    
    return args


# ============================================================================
# MAIN DATASET BUILDER
# ============================================================================

def _cleanup_dataset_files(dataset_name: str, modality: str, output_dir: str):
    """Clean up downloaded files for a specific dataset to free disk space."""
    safe_name = dataset_name.replace(" ", "_").replace("/", "_")
    
    if modality == "image":
        cleanup_path = os.path.join(output_dir, "images", safe_name)
    elif modality == "video":
        cleanup_path = os.path.join(output_dir, "videos")  # Videos use flat structure
    elif modality == "audio":
        cleanup_path = os.path.join(output_dir, "audio", safe_name)
    else:
        return
    
    if os.path.exists(cleanup_path):
        try:
            shutil.rmtree(cleanup_path)
            logger.info(f"    🗑️ Cleaned up: {cleanup_path}")
        except Exception as e:
            logger.warning(f"    Could not cleanup {cleanup_path}: {e}")


def _upload_dataset_to_hub(modality: str, dataset, hf_dataset_name: str, hf_token: str):
    """Upload dataset samples to HuggingFace, appending to existing split."""
    from datasets import DatasetDict, load_dataset, concatenate_datasets
    from huggingface_hub import create_repo
    
    try:
        # Create repo if needed
        try:
            create_repo(hf_dataset_name, repo_type="dataset", token=hf_token, exist_ok=True)
        except Exception:
            pass
        
        # Try to load existing data for this modality and merge
        try:
            existing = load_dataset(hf_dataset_name, split=modality, token=hf_token)
            dataset = concatenate_datasets([existing, dataset])
            logger.info(f"    Merged with existing {modality}: now {len(dataset)} total samples")
        except Exception:
            # No existing data for this modality, that's fine
            pass
        
        # Create DatasetDict and push
        ds_dict = DatasetDict({modality: dataset})
        ds_dict.push_to_hub(
            hf_dataset_name,
            token=hf_token,
            private=False,
        )
        return True
    except Exception as e:
        logger.error(f"    ✗ Upload failed: {e}")
        traceback.print_exc()
        return False


def build_unified_dataset(args):
    """
    Build the unified multimodal dataset.
    
    DISK-EFFICIENT: Processes each individual dataset, uploads to HuggingFace
    immediately, then cleans up files before moving to the next dataset.
    """
    from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage, Audio as HFAudio, load_dataset, concatenate_datasets
    from huggingface_hub import HfApi, login
    import gc
    
    logger.info("=" * 60)
    logger.info("XORON UNIFIED MULTIMODAL DATASET BUILDER")
    logger.info("=" * 60)
    logger.info("⚡ DISK-EFFICIENT: Upload & cleanup after EACH dataset")
    
    # Show what we're building
    modalities_to_build = []
    if args.all:
        modalities_to_build = ["text", "image", "video", "audio"]
    else:
        if args.text:
            modalities_to_build.append("text")
        if args.image:
            modalities_to_build.append("image")
        if args.video:
            modalities_to_build.append("video")
        if args.audio:
            modalities_to_build.append("audio")
    
    logger.info(f"Building modalities: {modalities_to_build}")
    
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
    
    # Helper function to ensure consistent string types for PyArrow
    def to_string_or_none(val):
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return str(val)
        if isinstance(val, bytes):
            try:
                return val.decode('utf-8')
            except:
                return str(val)
        return str(val) if not isinstance(val, str) else val
    
    def create_and_upload_audio_dataset(samples, dataset_name):
        """Create HF audio dataset from samples and upload immediately."""
        valid = [s for s in samples if s.get("audio_path") and isinstance(s.get("audio_path"), str) and os.path.exists(s.get("audio_path", ""))]
        if not valid:
            return False
        audio_data = {"audio": [], "text": [], "speaker_id": [], "sampling_rate": [], "gender": [], "age": [], "language": [], "emotion": [], "arousal": [], "valence": [], "dominance": [], "category": [], "source": []}
        for s in valid:
            audio_path = s.get("audio_path")
            if isinstance(audio_path, str) and os.path.exists(audio_path):
                audio_data["audio"].append(audio_path)
                audio_data["text"].append(to_string_or_none(s.get("text")))
                audio_data["speaker_id"].append(to_string_or_none(s.get("speaker_id")))
                sr = s.get("sampling_rate")
                audio_data["sampling_rate"].append(int(sr) if sr is not None else None)
                audio_data["gender"].append(to_string_or_none(s.get("gender")))
                audio_data["age"].append(to_string_or_none(s.get("age")))
                audio_data["language"].append(to_string_or_none(s.get("language")))
                audio_data["emotion"].append(to_string_or_none(s.get("emotion")))
                audio_data["arousal"].append(to_string_or_none(s.get("arousal")))
                audio_data["valence"].append(to_string_or_none(s.get("valence")))
                audio_data["dominance"].append(to_string_or_none(s.get("dominance")))
                audio_data["category"].append(to_string_or_none(s.get("category")))
                audio_data["source"].append(to_string_or_none(s.get("source")))
        if not audio_data["audio"]:
            return False
        try:
            ds = Dataset.from_dict(audio_data)
            ds = ds.cast_column("audio", HFAudio())
            logger.info(f"    ✓ Created: {len(ds)} samples")
            logger.info(f"    📤 Uploading to HuggingFace...")
            return _upload_dataset_to_hub("audio", ds, HF_DATASET_NAME, HF_TOKEN)
        except Exception as e:
            logger.error(f"    ✗ Failed: {e}")
            return False
    
    def create_and_upload_video_dataset(samples, dataset_name):
        """Create HF video dataset from samples and upload immediately."""
        valid = [s for s in samples if s.get("video_path") and isinstance(s.get("video_path"), str) and os.path.exists(s.get("video_path", ""))]
        if not valid:
            return False
        video_data = {"video": [], "caption": [], "question": [], "answer": [], "prompt": [], "options": [], "duration": [], "domain": [], "sub_category": [], "category": [], "source": []}
        for s in valid:
            video_path = s.get("video_path")
            if isinstance(video_path, str) and os.path.exists(video_path):
                try:
                    with open(video_path, 'rb') as f:
                        video_bytes = f.read()
                    video_data["video"].append({"bytes": video_bytes, "path": None})
                    video_data["caption"].append(to_string_or_none(s.get("caption")))
                    video_data["question"].append(to_string_or_none(s.get("question")))
                    video_data["answer"].append(to_string_or_none(s.get("answer")))
                    video_data["prompt"].append(to_string_or_none(s.get("prompt")))
                    video_data["options"].append(to_string_or_none(s.get("options")))
                    video_data["duration"].append(to_string_or_none(s.get("duration")))
                    video_data["domain"].append(to_string_or_none(s.get("domain")))
                    video_data["sub_category"].append(to_string_or_none(s.get("sub_category")))
                    video_data["category"].append(to_string_or_none(s.get("category")))
                    video_data["source"].append(to_string_or_none(s.get("source")))
                except Exception as e:
                    logger.warning(f"    Could not read {video_path}: {e}")
        if not video_data["video"]:
            return False
        try:
            from datasets import Video as HFVideo
            ds = Dataset.from_dict(video_data)
            ds = ds.cast_column("video", HFVideo(decode=False))
            logger.info(f"    ✓ Created: {len(ds)} samples")
            logger.info(f"    📤 Uploading to HuggingFace...")
            return _upload_dataset_to_hub("video", ds, HF_DATASET_NAME, HF_TOKEN)
        except Exception as e:
            logger.error(f"    ✗ Failed: {e}")
            return False
    
    def create_and_upload_image_dataset(samples, dataset_name):
        """Create HF image dataset from samples and upload immediately."""
        valid = [s for s in samples if s.get("image_path") and isinstance(s.get("image_path"), str) and os.path.exists(s.get("image_path", ""))]
        if not valid:
            return False
        image_data = {"image": [], "caption": [], "question": [], "answer": [], "choices": [], "category": [], "source": []}
        for s in valid:
            img_path = s.get("image_path")
            if isinstance(img_path, str) and os.path.exists(img_path):
                image_data["image"].append(img_path)
                image_data["caption"].append(to_string_or_none(s.get("caption")))
                image_data["question"].append(to_string_or_none(s.get("question")))
                image_data["answer"].append(to_string_or_none(s.get("answer")))
                image_data["choices"].append(to_string_or_none(s.get("choices")))
                image_data["category"].append(to_string_or_none(s.get("category")))
                image_data["source"].append(to_string_or_none(s.get("source")))
        if not image_data["image"]:
            return False
        try:
            ds = Dataset.from_dict(image_data)
            ds = ds.cast_column("image", HFImage())
            logger.info(f"    ✓ Created: {len(ds)} samples")
            logger.info(f"    📤 Uploading to HuggingFace...")
            return _upload_dataset_to_hub("image", ds, HF_DATASET_NAME, HF_TOKEN)
        except Exception as e:
            logger.error(f"    ✗ Failed: {e}")
            return False
    
    def create_and_upload_text_dataset(samples, dataset_name):
        """Create HF text dataset from samples and upload immediately."""
        if not samples:
            return False
        text_data = {"instruction": [], "response": [], "system": [], "conversations": [], "context": [], "category": [], "source": []}
        for s in samples:
            text_data["instruction"].append(to_string_or_none(s.get("instruction")))
            text_data["response"].append(to_string_or_none(s.get("response")))
            text_data["system"].append(to_string_or_none(s.get("system")))
            text_data["conversations"].append(to_string_or_none(s.get("conversations")))
            text_data["context"].append(to_string_or_none(s.get("context")))
            text_data["category"].append(to_string_or_none(s.get("category")))
            text_data["source"].append(to_string_or_none(s.get("source")))
        try:
            ds = Dataset.from_dict(text_data)
            logger.info(f"    ✓ Created: {len(ds)} samples")
            logger.info(f"    📤 Uploading to HuggingFace...")
            return _upload_dataset_to_hub("text", ds, HF_DATASET_NAME, HF_TOKEN)
        except Exception as e:
            logger.error(f"    ✗ Failed: {e}")
            return False
    
    uploaded_datasets = []
    
    # =========================================================================
    # PROCESS EACH DATASET INDIVIDUALLY - Upload and cleanup after EACH one
    # =========================================================================
    
    for target_modality in modalities_to_build:
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING MODALITY: {target_modality.upper()}")
        logger.info(f"{'='*60}")
        
        for category, configs in DATASET_CONFIGS.items():
            modality = MODALITY_MAP.get(category, "text")
            
            if modality != target_modality:
                continue
            
            logger.info(f"\n  Category: {category}")
            
            for config in configs:
                dataset_name = config['name']
                logger.info(f"\n  {'─'*50}")
                logger.info(f"  📦 Dataset: {dataset_name}")
                logger.info(f"  {'─'*50}")
                
                try:
                    # Process this single dataset
                    if modality == "text":
                        samples = process_text_dataset(config, category, MAX_SAMPLES_PER_DATASET)
                        if samples:
                            success = create_and_upload_text_dataset(samples, dataset_name)
                    elif modality == "image":
                        samples = process_image_dataset(config, category, MAX_SAMPLES_PER_DATASET, OUTPUT_DIR)
                        if samples:
                            success = create_and_upload_image_dataset(samples, dataset_name)
                            # Cleanup image files for this dataset
                            _cleanup_dataset_files(dataset_name, "image", OUTPUT_DIR)
                    elif modality == "video":
                        samples = process_video_dataset(config, category, MAX_SAMPLES_PER_DATASET, OUTPUT_DIR)
                        if samples:
                            success = create_and_upload_video_dataset(samples, dataset_name)
                            # Cleanup video files
                            _cleanup_dataset_files(dataset_name, "video", OUTPUT_DIR)
                    elif modality == "audio":
                        samples = process_audio_dataset(config, category, MAX_SAMPLES_PER_DATASET, OUTPUT_DIR)
                        if samples:
                            success = create_and_upload_audio_dataset(samples, dataset_name)
                            # Cleanup audio files for this dataset
                            _cleanup_dataset_files(dataset_name, "audio", OUTPUT_DIR)
                    else:
                        samples = []
                        success = False
                    
                    if samples and success:
                        uploaded_datasets.append(dataset_name)
                        logger.info(f"    ✅ {dataset_name} complete!")
                    elif not samples:
                        logger.info(f"    ⚠️ No samples from {dataset_name}")
                    
                    # Clear memory after each dataset
                    del samples
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"  ✗ Failed to process {dataset_name}: {e}")
                    traceback.print_exc()
    
    # Final cleanup
    logger.info("\n" + "=" * 60)
    logger.info("FINAL CLEANUP")
    logger.info("=" * 60)
    try:
        if os.path.exists(DOWNLOAD_DIR):
            shutil.rmtree(DOWNLOAD_DIR)
            logger.info(f"  Cleaned: {DOWNLOAD_DIR}")
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
            logger.info(f"  Cleaned: {OUTPUT_DIR}")
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            logger.info(f"  Cleaned: {CACHE_DIR}")
    except Exception as e:
        logger.warning(f"  Cleanup warning: {e}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DATASET BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Uploaded {len(uploaded_datasets)} datasets: {uploaded_datasets}")
    logger.info(f"  View at: https://huggingface.co/datasets/{HF_DATASET_NAME}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    build_unified_dataset(args)

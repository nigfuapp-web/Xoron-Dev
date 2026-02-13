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
                if 'bytes' in audio and audio['bytes']:
                    audio_data = audio['bytes']
                    # Also get sampling_rate from audio dict
                    if result["sampling_rate"] is None and 'sampling_rate' in audio:
                        result["sampling_rate"] = audio['sampling_rate']
                    break
                elif 'array' in audio:
                    audio_data = audio
                    if result["sampling_rate"] is None and 'sampling_rate' in audio:
                        result["sampling_rate"] = audio['sampling_rate']
                    break
                elif 'path' in audio and audio['path']:
                    audio_path = audio['path']
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
    
    # Prompt - for generation
    result["prompt"] = get_field(sample, [
        'Prompt', 'prompt', 'Text_Prompt', 'instruction', 'query'
    ])
    
    # QA fields (Video-MME format)
    result["question"] = get_field(sample, ['question', 'query'])
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
        # Direct URL fields
        for field in ['Video', 'video_url', 'video', 'url', 'video_link', 'mp4_url', 'gif_url', 'media_url', 'contentUrl', 'column1']:
            if field in sample:
                url = sample[field]
                if isinstance(url, str) and (url.startswith('http') or url.startswith('//')):
                    video_url = 'https:' + url if url.startswith('//') else url
                    break
    
    # YouTube/TikTok video ID
    if video_data is None and video_url is None:
        for field in ['videoID', 'video_id', 'youtube_id', 'clip_id', 'ytid']:
            if field in sample:
                vid_id = str(sample[field])
                if vid_id:
                    # Try Vript metadata first
                    vript_url = get_vript_video_url(vid_id)
                    if vript_url:
                        video_url = vript_url
                        break
                    # YouTube ID (11 chars)
                    if len(vid_id) == 11:
                        video_url = f"https://www.youtube.com/watch?v={vid_id}"
                        break
                    # TikTok ID (long numeric)
                    if vid_id.isdigit() and len(vid_id) > 15:
                        video_url = f"https://www.tiktok.com/@user/video/{vid_id}"
                        break
                    # Default to YouTube
                    if len(vid_id) >= 11:
                        video_url = f"https://www.youtube.com/watch?v={vid_id[:11]}"
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
        # Pull exactly max_samples in one batch (no per-sample HTTP requests)
        split_str = f"{config['split']}[:{max_samples}]"
        load_kwargs = {"path": config["path"], "split": split_str, "streaming": False}
        if "config" in config:
            load_kwargs["name"] = config["config"]
        
        ds = load_dataset(**load_kwargs)
        
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


def process_video_dataset(config: Dict, category: str, max_samples: int, output_dir: str) -> List[Dict]:
    """Process a video dataset with standardized schema. Supports .mp4, .gif, .webm and other formats."""
    from datasets import load_dataset
    
    samples = []
    name = config['name']
    vid_dir = os.path.join(output_dir, "videos", name.replace(" ", "_").replace("/", "_"))
    os.makedirs(vid_dir, exist_ok=True)
    
    logger.info(f"Processing video dataset: {name}")
    
    try:
        # Pull exactly max_samples in one batch (no per-sample HTTP requests)
        split_str = f"{config['split']}[:{max_samples}]"
        load_kwargs = {"path": config["path"], "split": split_str, "streaming": False}
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
                logger.warning(f"  Too many failures, stopping {name}")
                break
            
            # Extract with standardized schema (now returns video_data too)
            metadata, video_data, video_url = extract_video_sample(sample, category, name)
            
            video_path = None
            video_id = f"{idx:06d}"
            
            # First try to save video data directly (bytes or dict)
            if video_data is not None:
                try:
                    # Handle dict with 'bytes' key (HuggingFace Video format)
                    if isinstance(video_data, dict) and 'bytes' in video_data and video_data['bytes'] is not None:
                        # Try to detect format from bytes magic numbers
                        video_bytes = video_data['bytes']
                        ext = '.mp4'  # default
                        if video_bytes[:4] == b'GIF8' or video_bytes[:6] == b'GIF89a' or video_bytes[:6] == b'GIF87a':
                            ext = '.gif'
                        elif video_bytes[:4] == b'\x1a\x45\xdf\xa3':  # webm/mkv
                            ext = '.webm'
                        target_path = os.path.join(vid_dir, f"{video_id}{ext}")
                        with open(target_path, 'wb') as f:
                            f.write(video_bytes)
                        video_path = target_path
                    
                    # Handle dict with 'path' key
                    elif isinstance(video_data, dict) and 'path' in video_data and video_data['path']:
                        src_path = video_data['path']
                        if os.path.exists(src_path):
                            ext = get_video_ext_from_path(src_path)
                            target_path = os.path.join(vid_dir, f"{video_id}{ext}")
                            shutil.copy(src_path, target_path)
                            video_path = target_path
                    
                    # Handle raw bytes
                    elif isinstance(video_data, bytes):
                        ext = '.mp4'  # default
                        if video_data[:4] == b'GIF8' or video_data[:6] == b'GIF89a' or video_data[:6] == b'GIF87a':
                            ext = '.gif'
                        elif video_data[:4] == b'\x1a\x45\xdf\xa3':  # webm/mkv
                            ext = '.webm'
                        target_path = os.path.join(vid_dir, f"{video_id}{ext}")
                        with open(target_path, 'wb') as f:
                            f.write(video_data)
                        video_path = target_path
                
                except Exception as e:
                    logger.warning(f"  Error saving video {idx}: {e}")
                    video_path = None
            
            # If no video data saved, try downloading from URL
            if video_path is None and video_url:
                video_path = download_video(video_url, vid_dir, video_id)
            
            if video_path and os.path.exists(video_path):
                size_mb = get_file_size_mb(video_path)
                if size_mb > MAX_VIDEO_SIZE_MB:
                    os.remove(video_path)
                    failed += 1
                    continue
                
                metadata['video_path'] = video_path
                samples.append(metadata)
                count += 1
                downloaded += 1
            else:
                failed += 1
        
        logger.info(f"  ✓ {len(samples)} samples from {name} (downloaded: {downloaded}, failed: {failed})")
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
    
    Uses custom audio decoding (soundfile/librosa) instead of torchcodec.
    Disables HuggingFace's automatic audio decoding to avoid torchcodec dependency.
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
        # Pull exactly max_samples in one batch (no per-sample HTTP requests)
        split_str = f"{config['split']}[:{max_samples}]"
        load_kwargs = {"path": config["path"], "split": split_str, "streaming": False}
        if "config" in config:
            load_kwargs["name"] = config["config"]
        
        ds = load_dataset(**load_kwargs)
        
        # Disable automatic audio decoding - use our custom soundfile/librosa decoder
        # This avoids torchcodec dependency
        try:
            ds = ds.cast_column('audio', Audio(decode=False))
            logger.info(f"  📝 Using custom audio decoder (soundfile/librosa), NOT torchcodec")
        except Exception:
            # Column might not exist or already in different format
            pass
        
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
                    
                    # Handle dict with 'path' key
                    elif isinstance(audio_data, dict) and 'path' in audio_data and audio_data['path']:
                        src_path = audio_data['path']
                        if os.path.exists(src_path):
                            shutil.copy(src_path, target_path)
                            audio_path = target_path
                
                except Exception as e:
                    logger.warning(f"  Error saving audio {idx}: {e}")
                    audio_path = None
            
            elif audio_src_path and os.path.exists(audio_src_path):
                try:
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

def build_unified_dataset(args):
    """Build the unified multimodal dataset."""
    from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage, Audio as HFAudio, load_dataset, concatenate_datasets
    from huggingface_hub import HfApi, login
    
    logger.info("=" * 60)
    logger.info("XORON UNIFIED MULTIMODAL DATASET BUILDER")
    logger.info("=" * 60)
    
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
    if args.hf:
        logger.info(f"Will load existing dataset from HF: {HF_DATASET_NAME}")
    
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
    
    # Load existing dataset from HF if --hf flag is set
    existing_datasets = {}
    if args.hf:
        logger.info(f"\nLoading existing dataset from HuggingFace: {HF_DATASET_NAME}")
        try:
            existing_ds = load_dataset(HF_DATASET_NAME, token=HF_TOKEN)
            if isinstance(existing_ds, DatasetDict):
                for split_name in existing_ds:
                    existing_datasets[split_name] = existing_ds[split_name]
                    logger.info(f"  Loaded existing '{split_name}' split: {len(existing_ds[split_name])} samples")
            else:
                existing_datasets["train"] = existing_ds
                logger.info(f"  Loaded existing dataset: {len(existing_ds)} samples")
        except Exception as e:
            logger.warning(f"Could not load existing dataset (may not exist yet): {e}")
            logger.info("Will create new dataset from scratch")
    
    # Collect samples by modality
    all_samples = {
        "text": [],
        "image": [],
        "video": [],
        "audio": [],
    }
    
    # Process each category (only for selected modalities)
    for category, configs in DATASET_CONFIGS.items():
        modality = MODALITY_MAP.get(category, "text")
        
        # Skip if this modality is not selected
        if modality not in modalities_to_build:
            continue
        
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
    
    # Create HuggingFace datasets with CLEAN standardized schemas
    logger.info("\nCreating HuggingFace datasets with standardized schemas...")
    
    # Helper function to ensure consistent string types for PyArrow
    def to_string_or_none(val):
        """Convert value to string or None to avoid PyArrow type errors."""
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
    
    datasets_dict = {}
    
    # TEXT DATASET
    # Schema: instruction, response, system, conversations, context, category, source
    if all_samples["text"]:
        logger.info("\n📝 Creating TEXT dataset...")
        text_data = {
            "instruction": [],
            "response": [],
            "system": [],
            "conversations": [],
            "context": [],
            "category": [],
            "source": [],
        }
        
        for s in all_samples["text"]:
            text_data["instruction"].append(to_string_or_none(s.get("instruction")))
            text_data["response"].append(to_string_or_none(s.get("response")))
            text_data["system"].append(to_string_or_none(s.get("system")))
            text_data["conversations"].append(to_string_or_none(s.get("conversations")))
            text_data["context"].append(to_string_or_none(s.get("context")))
            text_data["category"].append(to_string_or_none(s.get("category")))
            text_data["source"].append(to_string_or_none(s.get("source")))
        
        try:
            datasets_dict["text"] = Dataset.from_dict(text_data)
            logger.info(f"  ✓ Text: {len(datasets_dict['text'])} samples")
            logger.info(f"    Columns: {datasets_dict['text'].column_names}")
        except Exception as e:
            logger.error(f"  ✗ Failed to create text dataset: {e}")
    
    # IMAGE DATASET
    # Schema: image, caption, question, answer, choices, category, source
    if all_samples["image"]:
        logger.info("\n🖼️ Creating IMAGE dataset...")
        # Validate: image_path must be a string and file must exist
        valid_samples = [s for s in all_samples["image"] 
                        if s.get("image_path") 
                        and isinstance(s.get("image_path"), str)
                        and os.path.exists(s.get("image_path", ""))]
        
        if valid_samples:
            image_data = {
                "image": [],
                "caption": [],
                "question": [],
                "answer": [],
                "choices": [],
                "category": [],
                "source": [],
            }
            
            for s in valid_samples:
                img_path = s.get("image_path")
                # Double-check path is valid string before adding
                if isinstance(img_path, str) and os.path.exists(img_path):
                    image_data["image"].append(img_path)
                    # Convert all metadata to string or None to avoid PyArrow type errors
                    image_data["caption"].append(to_string_or_none(s.get("caption")))
                    image_data["question"].append(to_string_or_none(s.get("question")))
                    image_data["answer"].append(to_string_or_none(s.get("answer")))
                    image_data["choices"].append(to_string_or_none(s.get("choices")))
                    image_data["category"].append(to_string_or_none(s.get("category")))
                    image_data["source"].append(to_string_or_none(s.get("source")))
            
            if image_data["image"]:
                try:
                    ds = Dataset.from_dict(image_data)
                    ds = ds.cast_column("image", HFImage())
                    datasets_dict["image"] = ds
                    logger.info(f"  ✓ Image: {len(ds)} samples")
                    logger.info(f"    Columns: {ds.column_names}")
                except Exception as e:
                    logger.error(f"  ✗ Failed to create image dataset: {e}")
                    traceback.print_exc()
    
    # AUDIO DATASET
    # Schema: audio, text, speaker_id, sampling_rate, gender, age, language, emotion, arousal, valence, dominance, category, source
    if all_samples["audio"]:
        logger.info("\n🔊 Creating AUDIO dataset...")
        # Validate: audio_path must be a string and file must exist
        valid_samples = [s for s in all_samples["audio"]
                        if s.get("audio_path") 
                        and isinstance(s.get("audio_path"), str)
                        and os.path.exists(s.get("audio_path", ""))]
        
        if valid_samples:
            audio_data = {
                "audio": [],
                "text": [],  # Unified transcription field
                "speaker_id": [],
                "sampling_rate": [],
                "gender": [],
                "age": [],
                "language": [],
                "emotion": [],
                "arousal": [],
                "valence": [],
                "dominance": [],
                "category": [],
                "source": [],
            }
            
            for s in valid_samples:
                audio_path = s.get("audio_path")
                # Double-check path is valid string before adding
                if isinstance(audio_path, str) and os.path.exists(audio_path):
                    audio_data["audio"].append(audio_path)
                    audio_data["text"].append(to_string_or_none(s.get("text")))
                    audio_data["speaker_id"].append(to_string_or_none(s.get("speaker_id")))
                    # sampling_rate should stay as int or None
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
            
            if audio_data["audio"]:
                try:
                    ds = Dataset.from_dict(audio_data)
                    ds = ds.cast_column("audio", HFAudio())
                    datasets_dict["audio"] = ds
                    logger.info(f"  ✓ Audio: {len(ds)} samples (with embedded audio files)")
                    logger.info(f"    Columns: {ds.column_names}")
                except Exception as e:
                    logger.error(f"  ✗ Failed to create audio dataset: {e}")
                    traceback.print_exc()
    
    # VIDEO DATASET
    # Schema: video, caption, question, answer, prompt, options, duration, domain, sub_category, category, source
    if all_samples["video"]:
        logger.info("\n🎬 Creating VIDEO dataset...")
        # Validate: video_path must be a string and file must exist
        valid_samples = [s for s in all_samples["video"]
                        if s.get("video_path") 
                        and isinstance(s.get("video_path"), str)
                        and os.path.exists(s.get("video_path", ""))]
        
        if valid_samples:
            video_data = {
                "video": [],
                "caption": [],
                "question": [],
                "answer": [],
                "prompt": [],
                "options": [],
                "duration": [],
                "domain": [],
                "sub_category": [],
                "category": [],
                "source": [],
            }
            
            for s in valid_samples:
                video_path = s.get("video_path")
                # Double-check path is valid string before adding
                if isinstance(video_path, str) and os.path.exists(video_path):
                    # Read actual video bytes so they get embedded in the dataset
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
            
            if video_data["video"]:
                try:
                    # Use HF Video type to store actual video files
                    from datasets import Video as HFVideo
                    ds = Dataset.from_dict(video_data)
                    ds = ds.cast_column("video", HFVideo())
                    datasets_dict["video"] = ds
                    logger.info(f"  ✓ Video: {len(ds)} samples (with embedded .mp4 files)")
                    logger.info(f"    Columns: {ds.column_names}")
                except ImportError:
                    # Fallback if Video type not available (older datasets version)
                    logger.warning("  ⚠ HF Video type not available, storing as paths")
                    datasets_dict["video"] = Dataset.from_dict(video_data)
                    logger.info(f"  ✓ Video: {len(datasets_dict['video'])} samples (as paths)")
                except Exception as e:
                    logger.error(f"  ✗ Failed to create video dataset: {e}")
                    traceback.print_exc()
    
    # Merge with existing datasets if --hf flag was used
    if existing_datasets:
        logger.info("\nMerging with existing datasets from HuggingFace...")
        for split_name, existing_ds in existing_datasets.items():
            if split_name in datasets_dict:
                # We have new data for this split - need to merge
                # Note: Can only merge if columns are compatible
                logger.info(f"  Split '{split_name}' exists in both - merging...")
                try:
                    new_ds = datasets_dict[split_name]
                    # Try to concatenate
                    merged = concatenate_datasets([existing_ds, new_ds])
                    datasets_dict[split_name] = merged
                    logger.info(f"    Merged: {len(existing_ds)} + {len(new_ds)} = {len(merged)} samples")
                except Exception as e:
                    logger.warning(f"    Could not merge '{split_name}': {e}")
                    logger.info(f"    Keeping new data only ({len(new_ds)} samples)")
            else:
                # No new data for this split - keep existing
                datasets_dict[split_name] = existing_ds
                logger.info(f"  Keeping existing '{split_name}' split: {len(existing_ds)} samples")
    
    if not datasets_dict:
        logger.error("No datasets created! Check for errors above.")
        return
    
    # Create DatasetDict
    final_dataset = DatasetDict(datasets_dict)
    
    # Print final summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL DATASET SUMMARY")
    logger.info("=" * 60)
    for split_name, ds in final_dataset.items():
        logger.info(f"  {split_name}: {len(ds)} samples, {len(ds.column_names)} columns")
    
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
    args = parse_args()
    build_unified_dataset(args)

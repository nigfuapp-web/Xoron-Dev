"""Logging utilities for Xoron-Dev."""

import os
import warnings
import logging


def suppress_warnings():
    """Suppress noisy warnings from various libraries."""
    warnings.filterwarnings('ignore')

    # Environment variables to suppress various library outputs
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['DATASETS_VERBOSITY'] = 'error'
    os.environ['HF_DATASETS_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['FFMPEG_LOG_LEVEL'] = 'quiet'
    os.environ['AV_LOG_LEVEL'] = 'quiet'
    os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
    os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
    os.environ['FFREPORT'] = ''

    # CUDA memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # CUDA debugging for accurate stack traces
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # Suppress specific loggers
    for logger_name in [
        'huggingface_hub', 'huggingface_hub.utils._http', 'transformers',
        'datasets', 'urllib3', 'filelock', 'PIL', 'av', 'ffmpeg',
        'torchvision', 'torchaudio', 'httpx', 'httpcore', 'requests'
    ]:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)


def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create logger for Xoron
    logger = logging.getLogger('xoron')
    logger.setLevel(level)

    return logger


def print_banner():
    """Print Xoron-Dev banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██╗  ██╗ ██████╗ ██████╗  ██████╗ ███╗   ██╗              ║
║   ╚██╗██╔╝██╔═══██╗██╔══██╗██╔═══██╗████╗  ██║              ║
║    ╚███╔╝ ██║   ██║██████╔╝██║   ██║██╔██╗ ██║              ║
║    ██╔██╗ ██║   ██║██╔══██╗██║   ██║██║╚██╗██║              ║
║   ██╔╝ ██╗╚██████╔╝██║  ██║╚██████╔╝██║ ╚████║              ║
║   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝              ║
║                                                              ║
║              Multimodal AI Training Framework                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

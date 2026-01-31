"""Device and environment utilities for Xoron-Dev."""

import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# Handle torch import gracefully
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class EnvironmentInfo:
    """Information about the runtime environment."""
    name: str  # 'kaggle', 'colab', 'local'
    temp_dir: str
    datasets_dir: str
    output_dir: str
    model_dir: str
    final_model_dir: str


def detect_environment() -> str:
    """
    Detect the runtime environment.
    
    Returns:
        'kaggle' - Running on Kaggle
        'colab' - Running on Google Colab
        'lightning' - Running on Lightning AI
        'local' - Running locally or other environment
    """
    # Check for Kaggle
    if os.path.exists('/kaggle') or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        return 'kaggle'
    
    # Check for Google Colab
    try:
        import google.colab
        return 'colab'
    except ImportError:
        pass
    
    # Check for Colab environment variable
    if 'COLAB_GPU' in os.environ or 'COLAB_RELEASE_TAG' in os.environ:
        return 'colab'
    
    # Check for Lightning AI
    # Lightning AI uses /teamspace/studios for studio environments
    # and sets LIGHTNING_* environment variables
    if (os.path.exists('/teamspace') or 
        'LIGHTNING_CLOUDSPACE_HOST' in os.environ or
        'LIGHTNING_CLOUD_PROJECT_ID' in os.environ or
        'LIGHTNING_STUDIO_ID' in os.environ or
        os.environ.get('LIGHTNING_CLOUD_URL')):
        return 'lightning'
    
    return 'local'


def get_environment_paths(env: Optional[str] = None) -> EnvironmentInfo:
    """
    Get appropriate paths based on the detected environment.
    
    Args:
        env: Override environment detection ('kaggle', 'colab', 'lightning', 'local')
        
    Returns:
        EnvironmentInfo with appropriate paths for the environment
    """
    if env is None:
        env = detect_environment()
    
    if env == 'kaggle':
        # Kaggle: /kaggle/tmp for checkpoints (to save disk space in /working)
        # Only final model goes to /kaggle/working for persistence
        return EnvironmentInfo(
            name='kaggle',
            temp_dir='/kaggle/tmp/xoron',
            datasets_dir='/kaggle/tmp/xoron/datasets',
            output_dir='/kaggle/tmp/xoron/checkpoints',  # Checkpoints in /tmp to save disk space
            model_dir='/kaggle/tmp/xoron/model',  # Built model in /tmp
            final_model_dir='/kaggle/working/xoron-final',  # Only final model in /working
        )
    
    elif env == 'colab':
        # Colab: /content for everything, can mount Google Drive
        return EnvironmentInfo(
            name='colab',
            temp_dir='/content/tmp',
            datasets_dir='/content/tmp/datasets',
            output_dir='/content/xoron-checkpoints',
            model_dir='/content/xoron-dev-model',
            final_model_dir='/content/xoron-final',
        )
    
    elif env == 'lightning':
        # Lightning AI: /teamspace/studios/this_studio for persistent storage
        # Use /tmp for temporary files, /teamspace for persistent outputs
        studio_path = '/teamspace/studios/this_studio'
        if not os.path.exists(studio_path):
            studio_path = '/teamspace'
        return EnvironmentInfo(
            name='lightning',
            temp_dir='/tmp/xoron',
            datasets_dir='/tmp/xoron/datasets',
            output_dir=f'{studio_path}/xoron-checkpoints',
            model_dir=f'{studio_path}/xoron-dev-model',
            final_model_dir=f'{studio_path}/xoron-final',
        )
    
    else:
        # Local: use current directory
        return EnvironmentInfo(
            name='local',
            temp_dir='./tmp',
            datasets_dir='./tmp/datasets',
            output_dir='./xoron-checkpoints',
            model_dir='./xoron-dev-model',
            final_model_dir='./xoron-final',
        )


def print_environment_info():
    """Print detected environment information."""
    env = detect_environment()
    paths = get_environment_paths(env)
    
    env_icons = {
        'kaggle': '🏆',
        'colab': '🔬',
        'lightning': '⚡️',
        'local': '💻',
    }
    
    print(f"\n{env_icons.get(env, '🖥️')} Environment: {env.upper()}")
    print(f"   Temp directory: {paths.temp_dir}")
    print(f"   Datasets directory: {paths.datasets_dir}")
    print(f"   Output directory: {paths.output_dir}")
    print(f"   Model directory: {paths.model_dir}")
    print(f"   Final model directory: {paths.final_model_dir}")


def get_device_info() -> Dict[str, any]:
    """Get information about available devices."""
    if not TORCH_AVAILABLE:
        return {
            'cuda_available': False,
            'device': 'cpu',
            'num_gpus': 0,
            'gpus': [],
            'total_memory_gb': 0,
        }

    info = {
        'cuda_available': torch.cuda.is_available(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_gpus': 0,
        'gpus': [],
        'total_memory_gb': 0,
    }

    if torch.cuda.is_available():
        info['num_gpus'] = torch.cuda.device_count()
        total_memory = 0
        for i in range(info['num_gpus']):
            props = torch.cuda.get_device_properties(i)
            gpu_info = {
                'index': i,
                'name': torch.cuda.get_device_name(i),
                'memory_gb': props.total_memory / 1e9,
                'compute_capability': f"{props.major}.{props.minor}",
            }
            info['gpus'].append(gpu_info)
            total_memory += props.total_memory
        info['total_memory_gb'] = total_memory / 1e9

    return info


def print_device_info():
    """Print device information."""
    info = get_device_info()

    print("\n💻 Device Configuration:")
    print(f"   CUDA Available: {info['cuda_available']}")
    print(f"   Device: {info['device']}")

    if info['cuda_available']:
        print(f"   Number of GPUs: {info['num_gpus']}")
        for gpu in info['gpus']:
            print(f"   GPU {gpu['index']}: {gpu['name']} - {gpu['memory_gb']:.1f} GB")
        print(f"   Total GPU Memory: {info['total_memory_gb']:.1f} GB")


def clear_cuda_cache():
    """Clear CUDA cache to free memory."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def get_optimal_device() -> str:
    """Get the optimal device for training."""
    if not TORCH_AVAILABLE:
        return 'cpu'
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def move_to_device(tensor_or_model, device: str):
    """Move tensor or model to specified device."""
    if hasattr(tensor_or_model, 'to'):
        return tensor_or_model.to(device)
    return tensor_or_model

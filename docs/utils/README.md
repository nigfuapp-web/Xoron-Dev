# 🔧 Utils Module Documentation

The Utils module provides utility functions for device detection, logging, and environment management.

## 📁 File Structure

```
utils/
├── __init__.py
├── device.py    # Device and environment utilities
└── logging.py   # Logging configuration
```

---

## 🖥️ Device Utilities

### Environment Detection

```python
def detect_environment() -> str:
    """
    Detect the runtime environment.
    
    Returns:
        'kaggle' - Running on Kaggle
        'colab' - Running on Google Colab
        'lightning' - Running on Lightning AI
        'local' - Running locally
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
    
    # Check for Lightning AI
    if (os.path.exists('/teamspace') or
        'LIGHTNING_CLOUDSPACE_HOST' in os.environ):
        return 'lightning'
    
    return 'local'
```

### Environment Paths

```python
@dataclass
class EnvironmentInfo:
    """Information about the runtime environment."""
    name: str           # 'kaggle', 'colab', 'lightning', 'local'
    temp_dir: str       # Temporary files
    datasets_dir: str   # Dataset cache
    output_dir: str     # Checkpoints
    model_dir: str      # Built model
    final_model_dir: str  # Final model


def get_environment_paths(env: str = None) -> EnvironmentInfo:
    """Get appropriate paths based on environment."""
    if env is None:
        env = detect_environment()
    
    if env == 'kaggle':
        return EnvironmentInfo(
            name='kaggle',
            temp_dir='/kaggle/tmp/xoron',
            datasets_dir='/kaggle/tmp/xoron/datasets',
            output_dir='/kaggle/tmp/xoron/checkpoints',
            model_dir='/kaggle/tmp/xoron/model',
            final_model_dir='/kaggle/working/xoron-final',
        )
    elif env == 'colab':
        return EnvironmentInfo(
            name='colab',
            temp_dir='/content/tmp',
            datasets_dir='/content/tmp/datasets',
            output_dir='/content/xoron-checkpoints',
            model_dir='/content/xoron-dev-model',
            final_model_dir='/content/xoron-final',
        )
    elif env == 'lightning':
        studio_path = '/teamspace/studios/this_studio'
        return EnvironmentInfo(
            name='lightning',
            temp_dir='/tmp/xoron',
            datasets_dir='/tmp/xoron/datasets',
            output_dir=f'{studio_path}/xoron-checkpoints',
            model_dir=f'{studio_path}/xoron-dev-model',
            final_model_dir=f'{studio_path}/xoron-final',
        )
    else:
        return EnvironmentInfo(
            name='local',
            temp_dir='./tmp',
            datasets_dir='./tmp/datasets',
            output_dir='./xoron-checkpoints',
            model_dir='./xoron-dev-model',
            final_model_dir='./xoron-final',
        )
```

### Device Information

```python
def get_device_info() -> Dict[str, any]:
    """Get information about available devices."""
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
```

### Memory Management

```python
def clear_cuda_cache():
    """Clear CUDA cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def get_optimal_device() -> str:
    """Get the optimal device for training."""
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
```

---

## 📝 Logging Utilities

### Logger Setup

```python
import logging

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup a logger with standard formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger."""
    return logging.getLogger(name)
```

### Training Logger

```python
class TrainingLogger:
    """Logger for training metrics."""
    
    def __init__(self, log_dir: str, tensorboard: bool = True):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics = []
        self.step = 0
        
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir)
            except ImportError:
                self.writer = None
        else:
            self.writer = None
    
    def log(self, metrics: Dict[str, float], step: int = None):
        """Log metrics."""
        if step is None:
            step = self.step
            self.step += 1
        
        # Add timestamp
        metrics['step'] = step
        metrics['timestamp'] = time.time()
        self.metrics.append(metrics)
        
        # TensorBoard
        if self.writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step)
    
    def save(self, path: str = None):
        """Save metrics to JSON."""
        if path is None:
            path = os.path.join(self.log_dir, 'metrics.json')
        
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def close(self):
        """Close logger."""
        if self.writer is not None:
            self.writer.close()
```

---

## 🔧 Usage Examples

### Environment Setup

```python
from utils.device import (
    detect_environment,
    get_environment_paths,
    print_environment_info,
    print_device_info,
)

# Detect and print environment
env = detect_environment()
print(f"Running on: {env}")

# Get paths
paths = get_environment_paths()
print(f"Model dir: {paths.model_dir}")
print(f"Output dir: {paths.output_dir}")

# Print full info
print_environment_info()
print_device_info()
```

### Memory Management

```python
from utils.device import clear_cuda_cache, get_optimal_device

# Get best device
device = get_optimal_device()
model = model.to(device)

# Clear cache periodically
for step, batch in enumerate(dataloader):
    # ... training step ...
    
    if step % 100 == 0:
        clear_cuda_cache()
```

### Logging

```python
from utils.logging import setup_logger, TrainingLogger

# Setup module logger
logger = setup_logger('training')
logger.info("Starting training...")

# Training metrics logger
metrics_logger = TrainingLogger('./logs', tensorboard=True)

for step, batch in enumerate(dataloader):
    loss = train_step(batch)
    
    metrics_logger.log({
        'loss': loss.item(),
        'lr': scheduler.get_last_lr()[0],
    })

metrics_logger.save()
metrics_logger.close()
```

---

## 🔗 Related Documentation

- [Config Documentation](../config/README.md) - Environment-aware configuration
- [Training Documentation](../training/README.md) - Training utilities

#!/usr/bin/env python3
"""
Xoron-Dev Interactive Setup Script

This script provides an interactive interface to configure:
- Dataset selection (add/remove datasets)
- Training parameters (epochs, batch size, learning rate, etc.)
- Model configuration (hidden size, layers, experts, etc.)
- Video/audio settings (frames, resolution, etc.)
- Fine-tuning options (freeze components, select modalities)
- Resume training from checkpoints

Usage:
    python setup.py              # Interactive mode
"""

import os
import sys
import json
import copy
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import XoronConfig, TrainingConfig, DATASET_CONFIGS, MODALITY_GROUPS
from config.dataset_config import get_total_datasets

# Import COMPONENT_GROUPS safely
try:
    from models.xoron import COMPONENT_GROUPS
except ImportError:
    COMPONENT_GROUPS = {
        'vision': ['vision_encoder', 'projector'],
        'video': ['video_encoder'],
        'audio': ['audio_encoder', 'audio_decoder', 'audio_projector'],
        'llm': ['llm'],
        'cross_attention': ['cross_attention_layers'],
        'image_generation': ['generator'],
        'video_generation': ['video_generator'],
        'modality_markers': ['image_start', 'image_end', 'video_start', 'video_end', 'audio_start', 'audio_end'],
    }

CONFIG_FILE = "xoron_config.json"


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_menu(options: List[str], title: str = "Options"):
    """Print a numbered menu."""
    print(f"\n{title}:")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    print(f"  [0] Back/Exit")


def get_input(prompt: str, default: Any = None, input_type: type = str) -> Any:
    """Get user input with default value."""
    if default is not None:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "

    value = input(prompt).strip()

    if not value and default is not None:
        return default

    try:
        if input_type == bool:
            return value.lower() in ('y', 'yes', 'true', '1')
        return input_type(value)
    except ValueError:
        print(f"Invalid input. Using default: {default}")
        return default


def load_config() -> Dict[str, Any]:
    """Load configuration from file or create defaults."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            # Ensure finetune section exists
            if 'finetune' not in config:
                config['finetune'] = get_default_finetune_config()
            return config

    # Create default config
    xoron = XoronConfig()
    training = TrainingConfig()

    return {
        'model': xoron.to_dict(),
        'training': training.to_dict(),
        'datasets': copy.deepcopy(DATASET_CONFIGS),
        'finetune': get_default_finetune_config(),
    }


def get_default_finetune_config() -> Dict[str, Any]:
    """Get default fine-tuning configuration."""
    return {
        'enabled': False,
        'mode': 'all',  # all, text, image, video, audio, vision, generation
        'freeze_components': [],
        'train_only_components': [],
        'load_model': '',
        'resume_from': '',
        'lora_only': False,
    }


def save_config(config: Dict[str, Any]):
    """Save configuration to file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n✅ Configuration saved to {CONFIG_FILE}")


def show_current_config(config: Dict[str, Any]):
    """Display current configuration."""
    print_header("CURRENT CONFIGURATION")

    print("\n📊 Model Configuration:")
    model = config['model']
    print(f"   Model Name: {model.get('model_name', 'Xoron-Dev')}")
    print(f"   Hidden Size: {model.get('hidden_size', 1024)}")
    print(f"   Layers: {model.get('num_layers', 12)}")
    print(f"   Heads: {model.get('num_heads', 16)}")
    print(f"   MoE Experts: {model.get('num_experts', 8)}")
    print(f"   LoRA Rank: {model.get('lora_r', 32)}")
    print(f"   Multi-Scale: {model.get('use_multi_scale', True)}")
    print(f"   Video Max Frames: {model.get('video_max_frames', 32)}")
    print(f"   Image Base Size: {model.get('image_base_size', 256)}")
    print(f"   Video Base Size: {model.get('video_base_size', 256)}")

    print("\n⚙️ Training Configuration:")
    training = config['training']
    print(f"   Epochs: {training.get('num_epochs', 12)}")
    print(f"   Batch Size: {training.get('batch_size', 5)}")
    print(f"   Gradient Accumulation: {training.get('gradient_accumulation_steps', 32)}")
    print(f"   Learning Rate: {training.get('learning_rate', 2e-4)}")
    print(f"   Max Seq Length: {training.get('max_seq_length', 1024)}")
    print(f"   Max Per Epoch: {training.get('max_per_epoch', 2000)}")
    print(f"   Max Per Dataset: {training.get('max_per_dataset', 100)}")
    print(f"   Sample Repeat: {training.get('sample_repeat', 2)}x")
    print(f"   Streaming: True (memory efficient)")
    print(f"   FP16: {training.get('fp16', True)}")

    print("\n🔧 Fine-tuning Configuration:")
    ft = config.get('finetune', get_default_finetune_config())
    print(f"   Enabled: {ft.get('enabled', False)}")
    if ft.get('enabled'):
        print(f"   Mode: {ft.get('mode', 'all')}")
        if ft.get('load_model'):
            print(f"   Load Model: {ft.get('load_model')}")
        if ft.get('resume_from'):
            print(f"   Resume From: {ft.get('resume_from')}")
        if ft.get('freeze_components'):
            print(f"   Frozen: {', '.join(ft.get('freeze_components', []))}")
        if ft.get('train_only_components'):
            print(f"   Train Only: {', '.join(ft.get('train_only_components', []))}")
        print(f"   LoRA Only: {ft.get('lora_only', False)}")

    print("\n📁 Dataset Configuration:")
    datasets = config.get('datasets', DATASET_CONFIGS)
    total = sum(len(v) for v in datasets.values())
    print(f"   Total Datasets: {total}")
    for category, ds_list in datasets.items():
        if ds_list:
            print(f"   {category}: {len(ds_list)} datasets")


def manage_datasets(config: Dict[str, Any]):
    """Manage dataset configuration."""
    while True:
        print_header("DATASET MANAGEMENT")

        datasets = config.get('datasets', copy.deepcopy(DATASET_CONFIGS))
        config['datasets'] = datasets

        # Show categories
        categories = list(datasets.keys())
        print("\nDataset Categories:")
        for i, cat in enumerate(categories, 1):
            count = len(datasets.get(cat, []))
            print(f"  [{i}] {cat}: {count} datasets")

        print(f"\n  [A] Add new dataset")
        print(f"  [R] Reset to defaults")
        print(f"  [0] Back")

        choice = input("\nSelect category or action: ").strip().upper()

        if choice == '0':
            break
        elif choice == 'A':
            add_dataset(config)
        elif choice == 'R':
            config['datasets'] = copy.deepcopy(DATASET_CONFIGS)
            print("✅ Datasets reset to defaults")
        elif choice.isdigit() and 1 <= int(choice) <= len(categories):
            category = categories[int(choice) - 1]
            manage_category_datasets(config, category)


def manage_category_datasets(config: Dict[str, Any], category: str):
    """Manage datasets in a specific category."""
    while True:
        print_header(f"DATASETS: {category.upper()}")

        datasets = config['datasets'].get(category, [])

        if not datasets:
            print("\n  No datasets in this category.")
        else:
            for i, ds in enumerate(datasets, 1):
                print(f"  [{i}] {ds.get('name', 'Unknown')} - {ds.get('path', 'N/A')}")

        print(f"\n  [A] Add dataset")
        print(f"  [D] Delete dataset")
        print(f"  [0] Back")

        choice = input("\nSelect action: ").strip().upper()

        if choice == '0':
            break
        elif choice == 'A':
            add_dataset_to_category(config, category)
        elif choice == 'D':
            if datasets:
                idx = get_input("Enter dataset number to delete", input_type=int)
                if 1 <= idx <= len(datasets):
                    removed = datasets.pop(idx - 1)
                    print(f"✅ Removed: {removed.get('name', 'Unknown')}")
                else:
                    print("Invalid selection")


def add_dataset_to_category(config: Dict[str, Any], category: str):
    """Add a new dataset to a category."""
    print(f"\n📥 Adding dataset to {category}")

    name = get_input("Dataset name")
    path = get_input("HuggingFace path (e.g., 'username/dataset')")
    split = get_input("Split", default="train")
    ds_config = get_input("Config name (optional, press Enter to skip)", default="")

    new_ds = {
        "name": name,
        "path": path,
        "split": split,
        "streaming": True,
    }

    if ds_config:
        new_ds["config"] = ds_config

    if category not in config['datasets']:
        config['datasets'][category] = []

    config['datasets'][category].append(new_ds)
    print(f"✅ Added: {name}")


def add_dataset(config: Dict[str, Any]):
    """Add a new dataset with category selection."""
    print_header("ADD NEW DATASET")

    categories = list(config['datasets'].keys())
    print("\nSelect category:")
    for i, cat in enumerate(categories, 1):
        print(f"  [{i}] {cat}")

    choice = get_input("Category number", input_type=int)
    if 1 <= choice <= len(categories):
        add_dataset_to_category(config, categories[choice - 1])


def configure_model(config: Dict[str, Any]):
    """Configure model parameters."""
    print_header("MODEL CONFIGURATION")

    model = config['model']

    print("\nCurrent values shown in brackets. Press Enter to keep current value.\n")

    model['hidden_size'] = get_input("Hidden size", model.get('hidden_size', 1024), int)
    model['num_layers'] = get_input("Number of layers", model.get('num_layers', 12), int)
    model['num_heads'] = get_input("Number of attention heads", model.get('num_heads', 16), int)
    model['num_experts'] = get_input("Number of MoE experts", model.get('num_experts', 8), int)
    model['num_experts_per_tok'] = get_input("Experts per token", model.get('num_experts_per_tok', 2), int)
    model['lora_r'] = get_input("LoRA rank", model.get('lora_r', 32), int)
    model['lora_alpha'] = get_input("LoRA alpha", model.get('lora_alpha', 64), int)
    
    # Multi-scale configuration (consolidated size/frame settings)
    model['video_max_frames'] = get_input("Video max frames", model.get('video_max_frames', 32), int)
    model['video_base_frames'] = get_input("Video base frames", model.get('video_base_frames', 16), int)
    model['image_base_size'] = get_input("Image base size", model.get('image_base_size', 256), int)
    model['video_base_size'] = get_input("Video base size", model.get('video_base_size', 256), int)

    use_moe = get_input("Enable MoE? (y/n)", 'y' if model.get('use_moe', True) else 'n')
    model['use_moe'] = use_moe.lower() in ('y', 'yes', 'true', '1')

    use_lora = get_input("Enable LoRA? (y/n)", 'y' if model.get('use_lora', True) else 'n')
    model['use_lora'] = use_lora.lower() in ('y', 'yes', 'true', '1')

    use_cross = get_input("Enable Cross-Attention? (y/n)", 'y' if model.get('use_cross_attention', True) else 'n')
    model['use_cross_attention'] = use_cross.lower() in ('y', 'yes', 'true', '1')
    
    use_multi_scale = get_input("Enable Multi-Scale Training? (y/n)", 'y' if model.get('use_multi_scale', True) else 'n')
    model['use_multi_scale'] = use_multi_scale.lower() in ('y', 'yes', 'true', '1')

    print("\n✅ Model configuration updated")


def configure_training(config: Dict[str, Any]):
    """Configure training parameters."""
    print_header("TRAINING CONFIGURATION")

    training = config['training']

    print("\nCurrent values shown in brackets. Press Enter to keep current value.\n")

    training['num_epochs'] = get_input("Number of epochs", training.get('num_epochs', 12), int)
    training['batch_size'] = get_input("Batch size", training.get('batch_size', 5), int)
    training['gradient_accumulation_steps'] = get_input("Gradient accumulation steps", training.get('gradient_accumulation_steps', 32), int)
    training['learning_rate'] = get_input("Learning rate", training.get('learning_rate', 2e-4), float)
    training['weight_decay'] = get_input("Weight decay", training.get('weight_decay', 0.01), float)
    training['warmup_ratio'] = get_input("Warmup ratio", training.get('warmup_ratio', 0.03), float)
    training['max_seq_length'] = get_input("Max sequence length", training.get('max_seq_length', 1024), int)
    training['max_per_epoch'] = get_input("Max samples per epoch", training.get('max_per_epoch', 2000), int)
    training['max_per_dataset'] = get_input("Max samples per dataset", training.get('max_per_dataset', 100), int)
    training['sample_repeat'] = get_input("Sample repeat (each sample shown N times)", training.get('sample_repeat', 2), int)
    training['save_steps'] = get_input("Save checkpoint every N steps", training.get('save_steps', 500), int)
    training['logging_steps'] = get_input("Log every N steps", training.get('logging_steps', 50), int)

    fp16 = get_input("Enable FP16? (y/n)", 'y' if training.get('fp16', True) else 'n')
    training['fp16'] = fp16.lower() in ('y', 'yes', 'true', '1')

    print("\n✅ Training configuration updated")


def configure_paths(config: Dict[str, Any]):
    """Configure output paths."""
    print_header("PATH CONFIGURATION")

    training = config['training']

    print("\nCurrent values shown in brackets. Press Enter to keep current value.\n")

    training['model_path'] = get_input("Model save path", training.get('model_path', './xoron-dev-model'))
    training['output_dir'] = get_input("Checkpoint output directory", training.get('output_dir', './xoron-checkpoints'))
    training['final_model_dir'] = get_input("Final model directory", training.get('final_model_dir', './xoron-final'))
    training['temp_dir'] = get_input("Temporary directory", training.get('temp_dir', './tmp'))
    training['datasets_dir'] = get_input("Datasets cache directory", training.get('datasets_dir', './tmp/datasets'))

    print("\n✅ Path configuration updated")


def configure_finetune(config: Dict[str, Any]):
    """Configure fine-tuning options."""
    print_header("FINE-TUNING CONFIGURATION")
    
    # Initialize finetune config if not present
    if 'finetune' not in config:
        config['finetune'] = {
            'enabled': False,
            'mode': 'all',
            'freeze_components': [],
            'train_only_components': [],
            'resume_from': None,
            'load_model': None,
        }
    
    ft = config['finetune']
    
    print("\nFine-tuning allows you to continue training from a checkpoint")
    print("or train only specific parts of the model.\n")
    
    # Enable fine-tuning
    enabled = get_input("Enable fine-tuning mode? (y/n)", 'y' if ft.get('enabled', False) else 'n')
    ft['enabled'] = enabled.lower() in ('y', 'yes', 'true', '1')
    
    if not ft['enabled']:
        print("\n✅ Fine-tuning disabled")
        return
    
    # Fine-tune mode
    print("\nFine-tune modes:")
    print("  all       - Train on all datasets")
    print("  text      - Train on text datasets only (code, conversation, etc.)")
    print("  image     - Train on image datasets only")
    print("  video     - Train on video datasets only")
    print("  audio     - Train on audio datasets only (ASR, TTS)")
    print("  vision    - Train on image + video datasets")
    print("  generation - Train on generation datasets only")
    
    ft['mode'] = get_input("Fine-tune mode", ft.get('mode', 'all'))
    
    # Resume from checkpoint
    print("\nResume training from a checkpoint (loads model + training state):")
    ft['resume_from'] = get_input("Checkpoint path (or empty to skip)", ft.get('resume_from', '') or '')
    if not ft['resume_from']:
        ft['resume_from'] = None
    
    # Load pretrained model
    print("\nLoad pretrained model (model only, fresh training state):")
    ft['load_model'] = get_input("Model path (or empty to skip)", ft.get('load_model', '') or '')
    if not ft['load_model']:
        ft['load_model'] = None
    
    # Component freezing
    print("\nAvailable component groups:")
    for i, group in enumerate(COMPONENT_GROUPS.keys(), 1):
        print(f"  [{i}] {group}")
    
    print("\nFreeze specific components (comma-separated numbers or names):")
    freeze_input = get_input("Components to freeze (or empty)", ','.join(ft.get('freeze_components', [])))
    if freeze_input:
        # Parse input - could be numbers or names
        freeze_list = []
        groups = list(COMPONENT_GROUPS.keys())
        for item in freeze_input.split(','):
            item = item.strip()
            if item.isdigit():
                idx = int(item) - 1
                if 0 <= idx < len(groups):
                    freeze_list.append(groups[idx])
            elif item in COMPONENT_GROUPS:
                freeze_list.append(item)
        ft['freeze_components'] = freeze_list
    else:
        ft['freeze_components'] = []
    
    print("\nTrain only specific components (freezes all others):")
    train_only_input = get_input("Components to train (or empty for all)", ','.join(ft.get('train_only_components', [])))
    if train_only_input:
        train_list = []
        groups = list(COMPONENT_GROUPS.keys())
        for item in train_only_input.split(','):
            item = item.strip()
            if item.isdigit():
                idx = int(item) - 1
                if 0 <= idx < len(groups):
                    train_list.append(groups[idx])
            elif item in COMPONENT_GROUPS:
                train_list.append(item)
        ft['train_only_components'] = train_list
    else:
        ft['train_only_components'] = []
    
    print("\n✅ Fine-tuning configuration updated")
    print(f"   Mode: {ft['mode']}")
    if ft['resume_from']:
        print(f"   Resume from: {ft['resume_from']}")
    if ft['load_model']:
        print(f"   Load model: {ft['load_model']}")
    if ft['freeze_components']:
        print(f"   Freeze: {ft['freeze_components']}")
    if ft['train_only_components']:
        print(f"   Train only: {ft['train_only_components']}")


def list_checkpoints(config: Dict[str, Any]):
    """List available checkpoints."""
    print_header("AVAILABLE CHECKPOINTS")
    
    training = config.get('training', {})
    output_dir = training.get('output_dir', './xoron-checkpoints')
    final_dir = training.get('final_model_dir', './xoron-final')
    model_path = training.get('model_path', './xoron-dev-model')
    
    checkpoints = []
    
    # Check output directory for checkpoints
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                config_file = os.path.join(item_path, 'config.json')
                model_file = os.path.join(item_path, 'model.safetensors')
                if os.path.exists(config_file) and os.path.exists(model_file):
                    # Check for training state
                    has_state = os.path.exists(os.path.join(item_path, 'training_state.pt'))
                    checkpoints.append((item_path, item, has_state))
    
    # Check final model directory
    if os.path.exists(final_dir):
        config_file = os.path.join(final_dir, 'config.json')
        model_file = os.path.join(final_dir, 'model.safetensors')
        if os.path.exists(config_file) and os.path.exists(model_file):
            has_state = os.path.exists(os.path.join(final_dir, 'training_state.pt'))
            checkpoints.append((final_dir, 'final-model', has_state))
    
    # Check model path
    if os.path.exists(model_path):
        config_file = os.path.join(model_path, 'config.json')
        model_file = os.path.join(model_path, 'model.safetensors')
        if os.path.exists(config_file) and os.path.exists(model_file):
            has_state = os.path.exists(os.path.join(model_path, 'training_state.pt'))
            checkpoints.append((model_path, 'built-model', has_state))
    
    if not checkpoints:
        print("\n  No checkpoints found.")
        print(f"\n  Searched in:")
        print(f"    - {output_dir}")
        print(f"    - {final_dir}")
        print(f"    - {model_path}")
    else:
        print("\n  Available checkpoints:\n")
        for path, name, has_state in checkpoints:
            state_str = "✅ has training state" if has_state else "❌ no training state"
            print(f"  📁 {name}")
            print(f"     Path: {path}")
            print(f"     {state_str}")
            print()


def generate_build_command(config: Dict[str, Any]):
    """Generate the build.py command based on current configuration."""
    print_header("GENERATE BUILD COMMAND")
    
    cmd_parts = ["python build.py"]
    
    ft = config.get('finetune', {})
    training = config.get('training', {})
    export = config.get('export', {})
    
    # Fine-tuning options
    if ft.get('enabled'):
        if ft.get('resume_from'):
            cmd_parts.append(f"--resume {ft['resume_from']}")
        elif ft.get('load_model'):
            cmd_parts.append(f"--load-model {ft['load_model']}")
        
        if ft.get('mode') and ft['mode'] != 'all':
            cmd_parts.append(f"--finetune {ft['mode']}")
        
        if ft.get('freeze_components'):
            cmd_parts.append(f"--freeze {','.join(ft['freeze_components'])}")
        
        if ft.get('train_only_components'):
            cmd_parts.append(f"--train-only {','.join(ft['train_only_components'])}")
    else:
        cmd_parts.append("--build")
    
    # Training overrides
    if training.get('num_epochs'):
        cmd_parts.append(f"--epochs {training['num_epochs']}")
    
    if training.get('batch_size'):
        cmd_parts.append(f"--batch-size {training['batch_size']}")
    
    if training.get('learning_rate'):
        cmd_parts.append(f"--lr {training['learning_rate']}")
    
    # Export options
    if export.get('onnx'):
        cmd_parts.append("--onnx")
        if export.get('quant_bits', 4) != 4:
            cmd_parts.append(f"--quant-bits {export['quant_bits']}")
    
    if export.get('gguf'):
        cmd_parts.append("--gguf")
        if export.get('gguf_quant', 'q4_k_m') != 'q4_k_m':
            cmd_parts.append(f"--gguf-quant {export['gguf_quant']}")
    
    command = " \\\n    ".join(cmd_parts)
    
    print("\n📋 Generated command:\n")
    print(f"  {command}")
    print("\n")
    
    print("📝 Additional export options:")
    print("   --onnx           Export to ONNX format with 4-bit quantization")
    print("   --gguf           Export to GGUF format for llama.cpp")
    print("   --quant-bits N   ONNX quantization bits (4 or 8)")
    print("   --gguf-quant T   GGUF quantization type (q4_0, q4_k_m, q5_k_m, q8_0, f16)")
    print("\n")


def main_menu():
    """Main interactive menu."""
    config = load_config()

    while True:
        clear_screen()
        print_header("XORON-DEV SETUP")

        print("""
Welcome to the Xoron-Dev configuration tool!

This tool allows you to customize:
  - Dataset selection and configuration
  - Model architecture parameters
  - Training hyperparameters
  - Output paths and directories
  - Fine-tuning and checkpoint options
""")

        options = [
            "Show Current Configuration",
            "Manage Datasets",
            "Configure Model",
            "Configure Training",
            "Configure Paths",
            "Configure Fine-tuning",
            "List Available Checkpoints",
            "Generate Build Command",
            "Save Configuration",
            "Reset to Defaults",
        ]

        print_menu(options, "Main Menu")

        choice = input("\nSelect option: ").strip()

        if choice == '0':
            save_prompt = input("\nSave configuration before exit? (y/n): ").strip().lower()
            if save_prompt in ('y', 'yes'):
                save_config(config)
            print("\nGoodbye! 👋")
            break
        elif choice == '1':
            show_current_config(config)
            input("\nPress Enter to continue...")
        elif choice == '2':
            manage_datasets(config)
        elif choice == '3':
            configure_model(config)
            input("\nPress Enter to continue...")
        elif choice == '4':
            configure_training(config)
            input("\nPress Enter to continue...")
        elif choice == '5':
            configure_paths(config)
            input("\nPress Enter to continue...")
        elif choice == '6':
            configure_finetune(config)
            input("\nPress Enter to continue...")
        elif choice == '7':
            list_checkpoints(config)
            input("\nPress Enter to continue...")
        elif choice == '8':
            generate_build_command(config)
            input("\nPress Enter to continue...")
        elif choice == '9':
            save_config(config)
            input("\nPress Enter to continue...")
        elif choice == '10':
            confirm = input("\nReset all settings to defaults? (y/n): ").strip().lower()
            if confirm in ('y', 'yes'):
                config = load_config.__wrapped__() if hasattr(load_config, '__wrapped__') else {
                    'model': XoronConfig().to_dict(),
                    'training': TrainingConfig().to_dict(),
                    'datasets': copy.deepcopy(DATASET_CONFIGS),
                    'finetune': {
                        'enabled': False,
                        'mode': 'all',
                        'freeze_components': [],
                        'train_only_components': [],
                        'resume_from': None,
                        'load_model': None,
                    },
                }
                print("✅ Configuration reset to defaults")
            input("\nPress Enter to continue...")


def main():
    """Main entry point - runs interactive menu."""
    main_menu()


if __name__ == "__main__":
    main()

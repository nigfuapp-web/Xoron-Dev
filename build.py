#!/usr/bin/env python3
"""
Xoron-Dev Build Script

This script runs the entire workflow of building and training the Xoron multimodal model.
It provides an interactive menu for:
1. Building a new model from scratch
2. Loading and continuing training from checkpoints
3. Fine-tuning with frozen components
4. Selecting specific datasets for training
5. Model saving and optional ONNX export

Usage:
    python build.py              # Interactive mode
"""

import os
import sys
import json
import copy
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import utilities first to setup logging
from utils.logging import suppress_warnings, print_banner
from utils.device import print_device_info, clear_cuda_cache

# Suppress warnings before other imports
suppress_warnings()

# Now import the rest
from config import (
    XoronConfig, TrainingConfig, SPECIAL_TOKENS, DATASET_CONFIGS, 
    get_format_functions, get_finetune_datasets, MODALITY_GROUPS,
    apply_chat_template_to_tokenizer, XORON_CHAT_TEMPLATE
)
from config.dataset_config import print_dataset_config, filter_datasets_by_modalities
from models import XoronMultimodalModel, COMPONENT_GROUPS
from data import MultimodalFormatter, TrueStreamingDataset, VoiceProcessor
from training import XoronTrainer, create_collate_fn, create_optimizer_and_scheduler
from export import export_to_onnx

from transformers import AutoTokenizer, CLIPImageProcessor

CONFIG_FILE = "xoron_config.json"


def safe_convert_to_fp16(model):
    """
    Safely convert model to FP16, handling potential overflow issues.
    
    FP16 has a max value of ~65504. Any value larger becomes inf,
    and inf * 0 = nan, which corrupts the entire model.
    
    This function:
    1. Checks all params for values > FP16 max
    2. Clamps them to safe range
    3. Converts to FP16
    4. Verifies no NaN/Inf was created
    """
    FP16_MAX = 65504.0
    
    # Step 1: Check and clamp unsafe values BEFORE conversion
    print("   🔍 Checking for FP16-unsafe values...")
    unsafe_params = []
    
    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            abs_max = param.abs().max().item()
            if abs_max > FP16_MAX:
                unsafe_params.append((name, abs_max))
                # Clamp to FP16-safe range
                with torch.no_grad():
                    param.clamp_(-FP16_MAX, FP16_MAX)
    
    if unsafe_params:
        print(f"   ⚠️ Found {len(unsafe_params)} params with values > FP16 max ({FP16_MAX}):")
        for name, val in unsafe_params[:10]:
            print(f"      - {name}: max_abs={val:.2f} (clamped)")
        if len(unsafe_params) > 10:
            print(f"      ... and {len(unsafe_params) - 10} more")
        print("   ✅ All values clamped to FP16-safe range")
    else:
        print("   ✅ All values are FP16-safe")
    
    # Step 2: Convert to FP16
    model = model.half()
    
    # Step 3: Verify no NaN/Inf was created
    print("   🔍 Verifying FP16 conversion...")
    nan_params = []
    
    for name, param in model.named_parameters():
        has_nan = torch.isnan(param).any().item()
        has_inf = torch.isinf(param).any().item()
        if has_nan or has_inf:
            nan_count = torch.isnan(param).sum().item()
            inf_count = torch.isinf(param).sum().item()
            nan_params.append((name, nan_count, inf_count, param.shape))
    
    if nan_params:
        print(f"\n   ❌ CRITICAL: FP16 conversion created {len(nan_params)} NaN/Inf params!")
        for name, nan_count, inf_count, shape in nan_params[:20]:
            total = 1
            for s in shape:
                total *= s
            pct = 100 * (nan_count + inf_count) / total
            print(f"      - {name}: nan={nan_count}, inf={inf_count}, shape={list(shape)} ({pct:.1f}%)")
        if len(nan_params) > 20:
            print(f"      ... and {len(nan_params) - 20} more")
        
        raise RuntimeError(
            f"FP16 conversion failed - {len(nan_params)} parameters have NaN/Inf. "
            "This usually means the model initialization created extreme values. "
            "Try using bf16 instead (set bf16=True in training config) or check model initialization."
        )
    
    print("   ✅ FP16 conversion verified - no NaN/Inf")
    return model


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
    print(f"  [0] Back/Cancel")


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


def load_saved_config() -> Dict[str, Any]:
    """Load configuration from file or create defaults."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return None


def load_configs_from_file() -> tuple:
    """Load configuration from saved file or create defaults."""
    saved = load_saved_config()
    if saved:
        xoron_config = XoronConfig.from_dict(saved.get('model', {}))
        training_config = TrainingConfig.from_dict(saved.get('training', {}))
        finetune_config = saved.get('finetune', {})
        dataset_configs = saved.get('datasets', copy.deepcopy(DATASET_CONFIGS))
    else:
        xoron_config = XoronConfig()
        training_config = TrainingConfig()
        finetune_config = {}
        dataset_configs = copy.deepcopy(DATASET_CONFIGS)
    
    return xoron_config, training_config, finetune_config, dataset_configs


def list_available_checkpoints(training_config) -> List[tuple]:
    """List available checkpoints."""
    checkpoints = []
    
    output_dir = training_config.output_dir
    final_dir = training_config.final_model_dir
    model_path = training_config.model_path
    
    # Check output directory for checkpoints
    if os.path.exists(output_dir):
        for item in sorted(os.listdir(output_dir)):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                config_file = os.path.join(item_path, 'config.json')
                model_file = os.path.join(item_path, 'model.safetensors')
                if os.path.exists(config_file) and os.path.exists(model_file):
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
    
    return checkpoints


def select_checkpoint_menu(training_config) -> Optional[str]:
    """Interactive menu to select a checkpoint."""
    checkpoints = list_available_checkpoints(training_config)
    
    if not checkpoints:
        print("\n  ❌ No checkpoints found.")
        print(f"\n  Searched in:")
        print(f"    - {training_config.output_dir}")
        print(f"    - {training_config.final_model_dir}")
        print(f"    - {training_config.model_path}")
        return None
    
    print("\n  Available checkpoints:\n")
    for i, (path, name, has_state) in enumerate(checkpoints, 1):
        state_str = "✅ has training state" if has_state else "❌ no training state"
        print(f"  [{i}] {name}")
        print(f"      Path: {path}")
        print(f"      {state_str}")
    print(f"  [0] Cancel")
    
    choice = get_input("\nSelect checkpoint", "0")
    if choice == '0' or not choice.isdigit():
        return None
    
    idx = int(choice) - 1
    if 0 <= idx < len(checkpoints):
        return checkpoints[idx][0]
    return None


def select_finetune_mode_menu() -> str:
    """Interactive menu to select fine-tune mode."""
    print_header("SELECT FINE-TUNE MODE")
    
    modes = [
        ("all", "Train on all datasets"),
        ("text", "Train on text datasets only (code, conversation, tool_use, agentic)"),
        ("image", "Train on image datasets only (caption, VQA, generation, editing)"),
        ("video", "Train on video datasets only (caption, QA, generation, image-to-video)"),
        ("audio", "Train on audio datasets only (ASR, TTS)"),
        ("vision", "Train on image + video datasets"),
        ("generation", "Train on generation datasets only (image_gen, video_gen, i2v)"),
    ]
    
    print("\nFine-tune modes determine which datasets are used:\n")
    for i, (mode, desc) in enumerate(modes, 1):
        print(f"  [{i}] {mode}: {desc}")
    print(f"  [0] Cancel")
    
    choice = get_input("\nSelect mode", "1")
    if choice == '0' or not choice.isdigit():
        return 'all'
    
    idx = int(choice) - 1
    if 0 <= idx < len(modes):
        return modes[idx][0]
    return 'all'


def select_components_to_freeze_menu() -> List[str]:
    """Interactive menu to select components to freeze."""
    print_header("SELECT COMPONENTS TO FREEZE")
    
    groups = list(COMPONENT_GROUPS.keys())
    
    print("\nAvailable component groups:\n")
    for i, group in enumerate(groups, 1):
        components = COMPONENT_GROUPS[group]
        print(f"  [{i}] {group}: {', '.join(components)}")
    print(f"\n  [A] Select all")
    print(f"  [0] None (train all)")
    
    choice = get_input("\nEnter numbers separated by commas (e.g., 1,2,3)", "0")
    
    if choice == '0':
        return []
    if choice.upper() == 'A':
        return groups.copy()
    
    selected = []
    for item in choice.split(','):
        item = item.strip()
        if item.isdigit():
            idx = int(item) - 1
            if 0 <= idx < len(groups):
                selected.append(groups[idx])
        elif item in groups:
            selected.append(item)
    
    return selected


def select_components_to_train_menu() -> List[str]:
    """Interactive menu to select components to train (freezes all others)."""
    print_header("SELECT COMPONENTS TO TRAIN")
    
    groups = list(COMPONENT_GROUPS.keys())
    
    print("\nSelect which components to train (all others will be frozen):\n")
    for i, group in enumerate(groups, 1):
        components = COMPONENT_GROUPS[group]
        print(f"  [{i}] {group}: {', '.join(components)}")
    print(f"\n  [A] Train all (no freezing)")
    print(f"  [0] Cancel")
    
    choice = get_input("\nEnter numbers separated by commas (e.g., 1,2,3)", "A")
    
    if choice == '0':
        return []
    if choice.upper() == 'A':
        return []  # Empty means train all
    
    selected = []
    for item in choice.split(','):
        item = item.strip()
        if item.isdigit():
            idx = int(item) - 1
            if 0 <= idx < len(groups):
                selected.append(groups[idx])
        elif item in groups:
            selected.append(item)
    
    return selected


def build_new_model(xoron_config, training_config):
    """Build a new model from scratch."""
    print("\n" + "=" * 60)
    print("🔨 BUILDING NEW MODEL")
    print("=" * 60)
    
    # Get device map for model parallelism
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_map = None

    if training_config.use_model_parallel and num_gpus > 1:
        from config.training_config import get_device_map
        device_map = get_device_map(num_gpus)
        print(f"⚡ Model Parallelism enabled across {num_gpus} GPUs")
    
    # Determine device
    if training_config.use_model_parallel and num_gpus > 1:
        device = device_map['primary']
    else:
        device = training_config.device
    
    # Build model
    model = XoronMultimodalModel(xoron_config, device_map)

    # Apply LoRA
    if xoron_config.use_lora:
        model.apply_lora()

    # Apply model parallelism or move to device
    if training_config.use_model_parallel and num_gpus > 1:
        model.apply_model_parallel(device_map)
    else:
        model = model.to(device)

    clear_cuda_cache()
    
    return model, device


def load_model_from_checkpoint(checkpoint_path, xoron_config, training_config):
    """Load model from a checkpoint."""
    print("\n" + "=" * 60)
    print(f"📂 LOADING MODEL FROM: {checkpoint_path}")
    print("=" * 60)
    
    # Get device map for model parallelism
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_map = None

    if training_config.use_model_parallel and num_gpus > 1:
        from config.training_config import get_device_map
        device_map = get_device_map(num_gpus)
        print(f"⚡ Model Parallelism enabled across {num_gpus} GPUs")
    
    # Determine device
    if training_config.use_model_parallel and num_gpus > 1:
        device = device_map['primary']
    else:
        device = training_config.device
    
    model = XoronMultimodalModel.from_pretrained(
        checkpoint_path,
        device=device if device_map is None else None,
        device_map=device_map,
        apply_lora=xoron_config.use_lora,
    )
    
    clear_cuda_cache()
    
    return model, device


def load_model_from_huggingface(hf_model_id, training_config):
    """
    Load model from HuggingFace Hub.
    
    This downloads the model from HuggingFace and loads it with proper
    vocab size detection and LoRA structure handling.
    
    Supports both:
    - Single file format: model.safetensors or pytorch_model.bin
    - Component format: separate .safetensors files per component (llm.safetensors, vision_encoder.safetensors, etc.)
    
    Args:
        hf_model_id: HuggingFace model identifier (e.g., 'Backup-bdg/Xoron-Dev-MultiMoe')
        training_config: Training configuration
        
    Returns:
        Tuple of (model, device, xoron_config)
    """
    print("\n" + "=" * 60)
    print(f"📥 LOADING MODEL FROM HUGGINGFACE: {hf_model_id}")
    print("=" * 60)
    
    try:
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        
        # Download model from HuggingFace
        print(f"\n🔄 Downloading model from HuggingFace...")
        cache_dir = snapshot_download(
            repo_id=hf_model_id,
            repo_type="model",
            local_files_only=False,
        )
        print(f"✅ Model downloaded to: {cache_dir}")
        
        # Load config from downloaded model
        config_path = os.path.join(cache_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Check if LoRA was applied when saving
        lora_was_applied = config_dict.pop('lora_applied', False)
        
        # Extract architecture markers from config (if saved with new format)
        config_arch_version = config_dict.pop('architecture_version', 1)
        config_has_waveform_decoder = config_dict.pop('has_waveform_decoder', False)
        config_has_vision_encoder = config_dict.pop('has_vision_encoder', None)  # None = unknown
        config_has_video_encoder = config_dict.pop('has_video_encoder', None)
        config_has_generator = config_dict.pop('has_generator', None)
        config_has_video_generator = config_dict.pop('has_video_generator', None)
        config_has_cross_attention = config_dict.pop('has_cross_attention', None)
        config_dict.pop('has_audio_encoder', None)
        config_dict.pop('has_audio_decoder', None)
        
        if config_arch_version >= 2:
            print(f"\n   📋 Config reports architecture version {config_arch_version}")
            print(f"      (Config has component markers - will verify against weights)")
        
        # Check for component-based save format (separate safetensors files per component)
        components_json_path = os.path.join(cache_dir, "components.json")
        is_component_format = os.path.exists(components_json_path)
        component_files = {}
        available_components = []
        
        if is_component_format:
            with open(components_json_path, 'r') as f:
                components_info = json.load(f)
            available_components = components_info.get('components', [])
            print(f"\n   📦 Detected COMPONENT-BASED save format")
            print(f"   📋 Available components: {', '.join(available_components)}")
            
            # Map component names to their safetensors files
            for comp in available_components:
                comp_file = os.path.join(cache_dir, f"{comp}.safetensors")
                if os.path.exists(comp_file):
                    component_files[comp] = comp_file
        
        # Detect architecture features from checkpoint weights
        model_path = os.path.join(cache_dir, "model.safetensors")
        pytorch_path = os.path.join(cache_dir, "pytorch_model.bin")
        checkpoint_vocab_size = None
        checkpoint_has_lora_structure = False
        
        # Architecture detection flags
        checkpoint_has_waveform_decoder = False
        checkpoint_has_video_generator = False
        checkpoint_has_generator = False
        checkpoint_has_cross_attention = False
        checkpoint_has_audio_encoder = False
        checkpoint_has_audio_decoder = False
        checkpoint_has_vision_encoder = False
        checkpoint_has_video_encoder = False
        checkpoint_has_projector = False
        checkpoint_has_audio_projector = False
        checkpoint_has_modality_markers = False
        
        def detect_architecture_from_keys(keys):
            """Detect which components exist in checkpoint from parameter keys."""
            nonlocal checkpoint_vocab_size, checkpoint_has_lora_structure
            nonlocal checkpoint_has_waveform_decoder, checkpoint_has_video_generator
            nonlocal checkpoint_has_generator, checkpoint_has_cross_attention
            nonlocal checkpoint_has_audio_encoder, checkpoint_has_audio_decoder
            nonlocal checkpoint_has_vision_encoder, checkpoint_has_video_encoder
            nonlocal checkpoint_has_projector, checkpoint_has_audio_projector
            nonlocal checkpoint_has_modality_markers
            
            # Check for LoRA structure in keys
            checkpoint_has_lora_structure = any('.lora_A' in k or '.lora_B' in k or '.linear.weight' in k for k in keys)
            
            # Detect components from key prefixes
            for key in keys:
                if key.startswith('waveform_decoder.'):
                    checkpoint_has_waveform_decoder = True
                elif key.startswith('video_generator.'):
                    checkpoint_has_video_generator = True
                elif key.startswith('generator.'):
                    checkpoint_has_generator = True
                elif key.startswith('cross_attention_layers.') or key.startswith('cross_attention.'):
                    checkpoint_has_cross_attention = True
                elif key.startswith('audio_encoder.'):
                    checkpoint_has_audio_encoder = True
                elif key.startswith('audio_decoder.'):
                    checkpoint_has_audio_decoder = True
                elif key.startswith('vision_encoder.'):
                    checkpoint_has_vision_encoder = True
                elif key.startswith('video_encoder.'):
                    checkpoint_has_video_encoder = True
                elif key.startswith('projector.'):
                    checkpoint_has_projector = True
                elif key.startswith('audio_projector.'):
                    checkpoint_has_audio_projector = True
                elif key.startswith('modality_markers.'):
                    checkpoint_has_modality_markers = True
        
        # For component format, detect architecture from available component files
        if is_component_format and component_files:
            print(f"\n   🔍 Detecting architecture from component files...")
            
            # Set flags based on which component files exist
            checkpoint_has_vision_encoder = 'vision_encoder' in component_files
            checkpoint_has_video_encoder = 'video_encoder' in component_files
            checkpoint_has_audio_encoder = 'audio_encoder' in component_files
            checkpoint_has_audio_decoder = 'audio_decoder' in component_files
            checkpoint_has_waveform_decoder = 'waveform_decoder' in component_files
            checkpoint_has_generator = 'generator' in component_files
            checkpoint_has_video_generator = 'video_generator' in component_files
            checkpoint_has_cross_attention = 'cross_attention' in component_files
            checkpoint_has_projector = 'projector' in component_files
            checkpoint_has_audio_projector = 'audio_projector' in component_files
            checkpoint_has_modality_markers = 'modality_markers' in component_files
            
            # Check LLM for vocab size and LoRA structure
            if 'llm' in component_files:
                try:
                    with safe_open(component_files['llm'], framework="pt") as f:
                        keys = list(f.keys())
                        checkpoint_has_lora_structure = any('.lora_A' in k or '.lora_B' in k or '.linear.weight' in k for k in keys)
                        
                        if checkpoint_has_lora_structure:
                            print(f"   🔧 Detected LoRA structure in LLM checkpoint")
                        
                        # Detect vocab size from embed_tokens or lm_head
                        for key in keys:
                            if 'embed_tokens.weight' in key or 'lm_head.weight' in key:
                                shape = f.get_tensor(key).shape
                                checkpoint_vocab_size = shape[0]
                                print(f"   📊 Detected checkpoint vocab size: {checkpoint_vocab_size}")
                                break
                except Exception as e:
                    print(f"   ⚠️ Could not inspect llm.safetensors: {e}")
        
        # For single file format, detect from model.safetensors or pytorch_model.bin
        elif os.path.exists(model_path):
            try:
                with safe_open(model_path, framework="pt") as f:
                    keys = list(f.keys())
                    detect_architecture_from_keys(keys)
                    
                    if checkpoint_has_lora_structure:
                        print(f"   🔧 Detected LoRA structure in checkpoint")
                    
                    # Detect vocab size
                    for key in keys:
                        if 'embed_tokens.weight' in key or 'lm_head.weight' in key:
                            shape = f.get_tensor(key).shape
                            checkpoint_vocab_size = shape[0]
                            print(f"   📊 Detected checkpoint vocab size: {checkpoint_vocab_size}")
                            break
            except Exception as e:
                print(f"   ⚠️ Could not inspect safetensors: {e}")
        elif os.path.exists(pytorch_path):
            try:
                state_dict = torch.load(pytorch_path, map_location='cpu')
                keys = list(state_dict.keys())
                detect_architecture_from_keys(keys)
                
                if checkpoint_has_lora_structure:
                    print(f"   🔧 Detected LoRA structure in checkpoint")
                
                for key in keys:
                    if 'embed_tokens.weight' in key or 'lm_head.weight' in key:
                        checkpoint_vocab_size = state_dict[key].shape[0]
                        print(f"   📊 Detected checkpoint vocab size: {checkpoint_vocab_size}")
                        break
            except Exception as e:
                print(f"   ⚠️ Could not inspect pytorch_model.bin: {e}")
        
        # Print detected architecture
        print(f"\n   🔍 Detected checkpoint architecture:")
        print(f"      - LLM: {'✅' if 'llm' in component_files or os.path.exists(model_path) or os.path.exists(pytorch_path) else '❌'} (vocab_size={checkpoint_vocab_size or 'unknown'})")
        print(f"      - Vision Encoder: {'✅' if checkpoint_has_vision_encoder else '❌ (will init randomly)'}")
        print(f"      - Video Encoder: {'✅' if checkpoint_has_video_encoder else '❌ (will init randomly)'}")
        print(f"      - Audio Encoder: {'✅' if checkpoint_has_audio_encoder else '❌ (will init randomly)'}")
        print(f"      - Audio Decoder: {'✅' if checkpoint_has_audio_decoder else '❌ (will init randomly)'}")
        print(f"      - Waveform Decoder: {'✅' if checkpoint_has_waveform_decoder else '❌ (will init randomly)'}")
        print(f"      - Image Generator: {'✅' if checkpoint_has_generator else '❌ (will init randomly)'}")
        print(f"      - Video Generator: {'✅' if checkpoint_has_video_generator else '❌ (will init randomly)'}")
        print(f"      - Cross Attention: {'✅' if checkpoint_has_cross_attention else '❌ (will init randomly)'}")
        print(f"      - Projector: {'✅' if checkpoint_has_projector else '❌ (will init randomly)'}")
        print(f"      - Audio Projector: {'✅' if checkpoint_has_audio_projector else '❌ (will init randomly)'}")
        print(f"      - Modality Markers: {'✅' if checkpoint_has_modality_markers else '❌ (will init randomly)'}")
        print(f"      - LoRA: {'✅' if checkpoint_has_lora_structure else '❌'}")
        
        # Update config vocab_size if checkpoint has different size
        config_vocab_size = config_dict.get('vocab_size', 151643)
        if checkpoint_vocab_size is not None and checkpoint_vocab_size != config_vocab_size:
            print(f"   🔄 Updating vocab_size from {config_vocab_size} to {checkpoint_vocab_size}")
            config_dict['vocab_size'] = checkpoint_vocab_size
        
        # Create XoronConfig from loaded config
        xoron_config = XoronConfig.from_dict(config_dict)
        
        # Get device map for model parallelism
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        device_map = None

        if training_config.use_model_parallel and num_gpus > 1:
            from config.training_config import get_device_map
            device_map = get_device_map(num_gpus)
            print(f"⚡ Model Parallelism enabled across {num_gpus} GPUs")
        
        # Determine device
        if training_config.use_model_parallel and num_gpus > 1:
            device = device_map['primary']
        else:
            device = training_config.device
        
        # Create model
        print(f"\n🔨 Building model architecture...")
        model = XoronMultimodalModel(xoron_config, device_map=device_map)
        
        # If checkpoint has LoRA structure, apply LoRA BEFORE loading weights
        if checkpoint_has_lora_structure and xoron_config.use_lora:
            print(f"   🔧 Applying LoRA before loading weights...")
            model.apply_lora()
        
        # Load weights based on format
        if is_component_format and component_files:
            # COMPONENT FORMAT: Load each component's weights separately
            # Build a combined state dict from all component files
            print(f"\n   📦 Loading weights from component files...")
            
            # Component name mapping (file name -> model prefix to ADD if key doesn't have one)
            # Some components save keys WITH prefix (projector -> projector.layers.0...)
            # Some save keys WITHOUT prefix (video_encoder -> encoder_blocks.0...)
            # Some save keys as bare names (modality_markers -> audio_start)
            component_prefix_map = {
                'llm': 'llm.',
                'vision_encoder': 'vision_encoder.',
                'video_encoder': 'video_encoder.',
                'audio_encoder': 'audio_encoder.',
                'audio_decoder': 'audio_decoder.',
                'projector': '',  # projector.safetensors already has 'projector.' prefix in keys
                'audio_projector': '',  # likely already has prefix
                'cross_attention': 'cross_attention_layers.',
                'generator': 'generator.',
                'video_generator': 'video_generator.',
                'waveform_decoder': 'waveform_decoder.',
                'modality_markers': '',  # keys are bare: audio_start, video_end, etc.
            }
            
            # Build combined checkpoint state dict from all component files
            checkpoint_state_dict = {}
            model_state_dict = model.state_dict()
            model_keys_set = set(model_state_dict.keys())
            
            for comp_name, comp_file in component_files.items():
                prefix = component_prefix_map.get(comp_name, f'{comp_name}.')
                print(f"      Loading {comp_name}...")
                
                try:
                    with safe_open(comp_file, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            tensor = f.get_tensor(key)
                            
                            # Try multiple key formats to find a match in model
                            possible_keys = [
                                key,  # As-is (for keys that already have correct prefix)
                                f"{prefix}{key}" if prefix else key,  # With component prefix
                                f"{comp_name}.{key}",  # With component name as prefix
                            ]
                            
                            # Find the first key that exists in the model
                            matched_key = None
                            for pk in possible_keys:
                                if pk in model_keys_set:
                                    matched_key = pk
                                    break
                            
                            if matched_key:
                                checkpoint_state_dict[matched_key] = tensor
                            else:
                                # If no match found, use the key with prefix (will be reported as skipped)
                                checkpoint_state_dict[f"{prefix}{key}" if prefix else key] = tensor
                                
                except Exception as e:
                    print(f"      ⚠️ Error loading {comp_name}: {e}")
            
            # Filter out tensors with shape mismatches
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            skipped_keys = []
            size_mismatch_keys = []
            
            for key, checkpoint_tensor in checkpoint_state_dict.items():
                if key in model_state_dict:
                    model_tensor = model_state_dict[key]
                    if checkpoint_tensor.shape == model_tensor.shape:
                        filtered_state_dict[key] = checkpoint_tensor
                    else:
                        size_mismatch_keys.append((key, checkpoint_tensor.shape, model_tensor.shape))
                else:
                    skipped_keys.append(key)
            
            # Load filtered state dict using load_state_dict for reliability
            missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
            
            # Report results
            loaded_count = len(filtered_state_dict)
            total_model_params = len(model_state_dict)
            
            print(f"\n   ✅ Loaded {loaded_count}/{total_model_params} parameters from checkpoint")
            
            if size_mismatch_keys:
                print(f"   ⚠️ Size mismatches: {len(size_mismatch_keys)} keys")
                components = {}
                for key, ckpt_shape, model_shape in size_mismatch_keys:
                    comp = key.split('.')[0]
                    components[comp] = components.get(comp, 0) + 1
                for comp, count in sorted(components.items()):
                    print(f"      - {comp}: {count} parameters (will be randomly initialized)")
            
            if missing:
                print(f"   ⚠️ Missing keys (not in checkpoint): {len(missing)} keys")
                components = {}
                for key in missing:
                    comp = key.split('.')[0]
                    components[comp] = components.get(comp, 0) + 1
                for comp, count in sorted(components.items()):
                    print(f"      - {comp}: {count} parameters (will be randomly initialized)")
            
            if skipped_keys:
                print(f"   ⚠️ Skipped keys (not in model): {len(skipped_keys)} keys")
            
            model.lora_applied = lora_was_applied or checkpoint_has_lora_structure
            
        elif os.path.exists(model_path):
            # SINGLE FILE FORMAT: model.safetensors
            print(f"   📦 Loading weights from safetensors (non-strict mode)...")
            checkpoint_state_dict = {}
            with safe_open(model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    checkpoint_state_dict[key] = f.get_tensor(key)
            
            # Filter out tensors with shape mismatches
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            skipped_keys = []
            size_mismatch_keys = []
            
            for key, checkpoint_tensor in checkpoint_state_dict.items():
                if key in model_state_dict:
                    model_tensor = model_state_dict[key]
                    if checkpoint_tensor.shape == model_tensor.shape:
                        filtered_state_dict[key] = checkpoint_tensor
                    else:
                        size_mismatch_keys.append((key, checkpoint_tensor.shape, model_tensor.shape))
                else:
                    skipped_keys.append(key)
            
            # Load filtered state dict
            missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
            
            # Report what happened
            loaded_count = len(filtered_state_dict)
            total_model_params = len(model_state_dict)
            print(f"   ✅ Loaded {loaded_count}/{total_model_params} parameters from checkpoint")
            
            if size_mismatch_keys:
                print(f"   ⚠️ Size mismatches (architecture changed): {len(size_mismatch_keys)} keys")
                components = {}
                for key, ckpt_shape, model_shape in size_mismatch_keys:
                    comp = key.split('.')[0]
                    components[comp] = components.get(comp, 0) + 1
                for comp, count in sorted(components.items()):
                    print(f"      - {comp}: {count} parameters (will be randomly initialized)")
            
            if missing:
                print(f"   ⚠️ Missing keys (new architecture): {len(missing)} keys")
                components = {}
                for key in missing:
                    comp = key.split('.')[0]
                    components[comp] = components.get(comp, 0) + 1
                for comp, count in sorted(components.items()):
                    print(f"      - {comp}: {count} parameters (will be randomly initialized)")
            
            if skipped_keys:
                print(f"   ⚠️ Skipped keys (old architecture): {len(skipped_keys)} keys")
            
            model.lora_applied = lora_was_applied or checkpoint_has_lora_structure
            
        elif os.path.exists(pytorch_path):
            # SINGLE FILE FORMAT: pytorch_model.bin
            print(f"   📦 Loading weights from pytorch_model.bin...")
            if 'state_dict' not in dir():
                checkpoint_state_dict = torch.load(pytorch_path, map_location='cpu')
            else:
                checkpoint_state_dict = state_dict
            
            # Filter out tensors with shape mismatches
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            size_mismatch_keys = []
            
            for key, checkpoint_tensor in checkpoint_state_dict.items():
                if key in model_state_dict:
                    model_tensor = model_state_dict[key]
                    if checkpoint_tensor.shape == model_tensor.shape:
                        filtered_state_dict[key] = checkpoint_tensor
                    else:
                        size_mismatch_keys.append(key)
            
            missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
            
            print(f"   ✅ Loaded {len(filtered_state_dict)}/{len(model_state_dict)} parameters")
            if size_mismatch_keys:
                print(f"   ⚠️ Size mismatches: {len(size_mismatch_keys)} (will be randomly initialized)")
            if missing:
                print(f"   ⚠️ Missing keys: {len(missing)} (new architecture)")
                
            model.lora_applied = lora_was_applied or checkpoint_has_lora_structure
        else:
            raise FileNotFoundError(f"No model weights found at {cache_dir}")
        
        # Apply LoRA if not already applied
        if xoron_config.use_lora and not model.lora_applied:
            model.apply_lora()
        
        # Move to device
        if device_map is not None:
            model.apply_model_parallel(device_map)
        else:
            model = model.to(device)
        
        clear_cuda_cache()
        
        print(f"✅ Model loaded successfully from HuggingFace!")
        model._print_stats()
        
        return model, device, xoron_config
        
    except Exception as e:
        print(f"❌ Error loading from HuggingFace: {e}")
        raise


def apply_freezing(model, freeze_components: List[str], train_only_components: List[str]):
    """Apply component freezing to the model."""
    if train_only_components:
        model.freeze_all_except(train_only_components)
    elif freeze_components:
        model.freeze_components(freeze_components)


def print_component_status(model):
    """Print component training status with 🔥 for trainable and ❄️ for frozen."""
    trainable, frozen = model.get_component_status()
    
    trainable_str = ', '.join(trainable) if trainable else 'none'
    frozen_str = ', '.join(frozen) if frozen else 'none'
    
    print(f"\n🔥 Trainable: {trainable_str}")
    if frozen:
        print(f"❄️ Frozen: {frozen_str}")


def setup_tokenizer(model, xoron_config):
    """Setup tokenizer with special tokens and custom chat template."""
    print("\n📝 Setting up tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(xoron_config.tokenizer_name)
    print(f"   ✅ Loaded base tokenizer: {xoron_config.tokenizer_name}")

    # Add special tokens
    special_tokens_list = list(SPECIAL_TOKENS.values())
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_list})

    # Apply our custom chat template (replaces pretrained tokenizer's template)
    tokenizer = apply_chat_template_to_tokenizer(tokenizer, multimodal=True)
    print(f"   ✅ Applied Xoron custom chat template")

    # Ensure pad token is set
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = SPECIAL_TOKENS['pad']
    
    # Set BOS and EOS to our tokens
    tokenizer.bos_token = SPECIAL_TOKENS['bos']
    tokenizer.eos_token = SPECIAL_TOKENS['eos']

    # Resize embeddings
    new_vocab_size = len(tokenizer)
    llm_device = model.get_llm_device() if hasattr(model, 'get_llm_device') else next(model.parameters()).device

    # Resize embed_tokens
    old_embed = model.llm.model.embed_tokens
    new_embed = nn.Embedding(new_vocab_size, xoron_config.hidden_size)

    with torch.no_grad():
        nn.init.normal_(new_embed.weight, mean=0.0, std=0.02)
        min_vocab = min(old_embed.weight.shape[0], new_vocab_size)
        new_embed.weight[:min_vocab] = old_embed.weight[:min_vocab].cpu()

    model.llm.model.embed_tokens = new_embed.to(llm_device)

    # Resize lm_head
    old_lm_head = model.llm.lm_head
    new_lm_head = nn.Linear(xoron_config.hidden_size, new_vocab_size, bias=False)

    with torch.no_grad():
        nn.init.normal_(new_lm_head.weight, mean=0.0, std=0.02)
        min_vocab = min(old_lm_head.weight.shape[0], new_vocab_size)
        new_lm_head.weight[:min_vocab] = old_lm_head.weight[:min_vocab].cpu()

    model.llm.lm_head = new_lm_head.to(llm_device)

    print(f"   📁 Vocab: {new_vocab_size:,} tokens")
    print(f"   📁 Special tokens: {len(special_tokens_list)} added")
    print(f"   ✅ Embeddings resized")
    print(f"   ✅ BOS: {tokenizer.bos_token}, EOS: {tokenizer.eos_token}, PAD: {tokenizer.pad_token}")

    return tokenizer


def save_model(model, tokenizer, training_config):
    """Save the built model."""
    print("\n" + "=" * 60)
    print("💾 SAVING BUILT MODEL")
    print("=" * 60)

    output_path = training_config.model_path
    os.makedirs(output_path, exist_ok=True)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    with open(os.path.join(output_path, "special_tokens.json"), "w") as f:
        json.dump(SPECIAL_TOKENS, f, indent=2)

    print("\nFiles saved:")
    for f in os.listdir(output_path):
        size = os.path.getsize(os.path.join(output_path, f)) / 1e6
        print(f"   - {f} ({size:.1f} MB)")


def setup_training(model, tokenizer, xoron_config, training_config, dataset_configs, active_modalities: str = 'all', resume_streaming_state: str = None):
    """Setup training components.
    
    Args:
        active_modalities: Which modalities are active ('all', 'text', 'image', 'video', 'audio')
            Inactive modalities use minimal tensors to save RAM (~27MB per batch for text-only)
        resume_streaming_state: Path to streaming_state.json to resume from (optional)
    
    Returns:
        train_dataset, eval_dataset, optimizer, scheduler, collate_fn
    """
    print("\n⚙️ Setting up training...")
    
    # Load image processor
    try:
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    except:
        image_processor = None

    # Initialize voice processor
    try:
        voice_proc = VoiceProcessor()
    except:
        voice_proc = None

    # Setup formatter
    formatter = MultimodalFormatter(SPECIAL_TOKENS, image_processor)
    format_functions = get_format_functions(formatter)

    # Get audio mode from config (SOTA: raw waveform vs legacy mel spectrogram)
    use_raw_waveform = getattr(xoron_config, 'use_raw_waveform', True)
    
    # Check if dual/multi training mode is active
    is_dual_training = getattr(training_config, 'is_dual_training', False)
    modality_max_values = getattr(training_config, 'modality_max_values', {})
    
    if is_dual_training:
        mode_names = list(modality_max_values.keys())
        print(f"\n🔀 Using DUAL TRAINING mode: {' + '.join(mode_names)}")
        for mod, max_val in modality_max_values.items():
            print(f"   {mod}: {max_val} samples/epoch")
    
    # Create streaming dataset with per-dataset limits and sample repetition
    # Pass modality_max_values for per-modality limits in dual training mode
    print(f"\n📁 Loading train datasets...")
    train_dataset = TrueStreamingDataset(
        dataset_configs=dataset_configs,
        format_functions=format_functions,
        tokenizer=tokenizer,
        tokens=SPECIAL_TOKENS,
        image_processor=image_processor,
        max_length=training_config.max_seq_length,
        max_per_epoch=training_config.max_per_epoch,
        max_per_dataset=training_config.max_per_dataset,
        sample_repeat=training_config.sample_repeat,
        voice_processor=voice_proc,
        max_video_frames=xoron_config.video_max_frames,  # Multi-scale config
        video_size=xoron_config.video_base_size,         # Multi-scale config
        resume_state_path=resume_streaming_state,
        use_raw_waveform=use_raw_waveform,
        modality_max_values=modality_max_values if is_dual_training else None,  # Per-modality limits
    )
    
    # Count train datasets
    train_dataset_count = sum(len(configs) for configs in dataset_configs.values() if configs)
    print(f"✅ {train_dataset_count} datasets initialized")
    
    # Set auto-save path for streaming state (saved alongside checkpoints)
    streaming_state_path = os.path.join(training_config.output_dir, "streaming_state.json")
    train_dataset.set_state_save_path(streaming_state_path)
    
    # Create eval dataset - pulls max_per_dataset_eval samples from each dataset for validation
    eval_samples_per_dataset = training_config.max_per_dataset_eval
    num_datasets = sum(len(configs) for configs in dataset_configs.values() if configs)
    eval_total = eval_samples_per_dataset * num_datasets
    
    print(f"\n📁 Loading eval datasets...")
    eval_dataset = TrueStreamingDataset(
        dataset_configs=dataset_configs,
        format_functions=format_functions,
        tokenizer=tokenizer,
        tokens=SPECIAL_TOKENS,
        image_processor=image_processor,
        max_length=training_config.max_seq_length,
        max_per_epoch=eval_total,
        max_per_dataset=eval_samples_per_dataset,
        sample_repeat=1,  # No repetition for eval
        voice_processor=voice_proc,
        max_video_frames=xoron_config.video_max_frames,  # Multi-scale config
        video_size=xoron_config.video_base_size,         # Multi-scale config
        resume_state_path=None,  # Don't resume eval dataset
        use_raw_waveform=use_raw_waveform,
    )
    print(f"✅ {num_datasets} datasets initialized")
    
    # Set initial skip positions for eval dataset (held-out data)
    train_samples_per_dataset = training_config.max_per_dataset
    eval_skip_positions = {}
    for dtype, configs in dataset_configs.items():
        if configs:
            for cfg in configs:
                eval_skip_positions[cfg["name"]] = train_samples_per_dataset
    eval_dataset._streaming_state["dataset_positions"] = eval_skip_positions.copy()

    # Create collate function (modality-specific modes use minimal tensors for inactive modalities to save RAM)
    collate_fn = create_collate_fn(xoron_config.video_max_frames, xoron_config.video_base_size, active_modalities=active_modalities)

    # Calculate training steps
    estimated_samples = min(training_config.max_per_epoch, len(train_dataset))
    steps_per_epoch = estimated_samples // (training_config.batch_size * training_config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * training_config.num_epochs

    # Create optimizer and scheduler
    # Use 8-bit optimizer to prevent OOM on multi-GPU setups
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        training_config.learning_rate,
        training_config.weight_decay,
        training_config.warmup_ratio,
        total_steps,
        use_8bit_optimizer=training_config.use_8bit_optimizer,
    )

    precision = 'BF16' if training_config.bf16 else ('FP16' if training_config.fp16 else 'FP32')
    print(f"   📊 {total_steps} steps | {precision} | lr={training_config.learning_rate}")

    return train_dataset, eval_dataset, optimizer, scheduler, collate_fn


def run_test(model, tokenizer, training_config):
    """Run a quick test of the model."""
    print("\n🧪 TESTING MODEL")

    model.eval()

    test_prompts = [
        f"{SPECIAL_TOKENS['user_start']}\nWrite a Python function to reverse a string.\n{SPECIAL_TOKENS['user_end']}\n{SPECIAL_TOKENS['assistant_start']}\n",
        f"{SPECIAL_TOKENS['user_start']}\n{SPECIAL_TOKENS['image_start']}[IMAGE]{SPECIAL_TOKENS['image_end']}\nWhat is in this image?\n{SPECIAL_TOKENS['user_end']}\n{SPECIAL_TOKENS['assistant_start']}\n",
    ]

    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: {prompt[:60]}...")

        inputs = tokenizer(prompt, return_tensors="pt").to(training_config.device)

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        print(f"Output logits shape: {outputs.logits.shape}")

    print("\n✅ Testing complete!")


def run_build_and_train(
    xoron_config,
    training_config,
    dataset_configs,
    checkpoint_path: Optional[str] = None,
    resume_training: bool = False,
    freeze_components: List[str] = None,
    train_only_components: List[str] = None,
    build_only: bool = False,
    export_onnx: bool = False,
    export_gguf: bool = False,
    onnx_quant_bits: int = 4,
    gguf_quant_type: str = 'q4_k_m',
    run_test_after: bool = False,
    active_modalities: str = 'all',
):
    """
    Run the build and training process.
    
    Args:
        xoron_config: Model configuration
        training_config: Training configuration
        dataset_configs: Dataset configurations
        checkpoint_path: Path to checkpoint to load from
        resume_training: Whether to resume training state
        freeze_components: Components to freeze during training
        train_only_components: Components to train (freezes others)
        build_only: If True, only build model without training
        export_onnx: Export to ONNX format after training
        export_gguf: Export to GGUF format after training
        onnx_quant_bits: Quantization bits for ONNX (4 or 8)
        gguf_quant_type: GGUF quantization type
        run_test_after: Run tests after training
    """
    
    # Print configurations
    xoron_config.print_config()
    training_config.print_config()
    
    # Print dataset config summary
    is_dual_training = getattr(training_config, 'is_dual_training', False)
    modality_max_values = getattr(training_config, 'modality_max_values', {})
    
    print("\n📊 Dataset Configuration:")
    total = 0
    for cat, datasets in dataset_configs.items():
        if datasets:
            # Show per-modality max if in dual training mode
            max_info = ""
            if is_dual_training and cat in modality_max_values:
                max_info = f" (max: {modality_max_values[cat]})"
            print(f"   {cat}: {len(datasets)} datasets{max_info}")
            total += len(datasets)
    print(f"   Total: {total} datasets")
    
    if is_dual_training:
        print(f"   Mode: Dual Training ({' + '.join(modality_max_values.keys())})")
        print(f"   Combined max_per_epoch: {training_config.max_per_epoch}")

    # Build or load model
    if checkpoint_path:
        model, device = load_model_from_checkpoint(checkpoint_path, xoron_config, training_config)
        xoron_config = model.config
    else:
        model, device = build_new_model(xoron_config, training_config)
    
    # Apply freezing
    if freeze_components or train_only_components:
        apply_freezing(model, freeze_components or [], train_only_components or [])
    
    print(f"\n✅ Model ready")
    print_component_status(model)

    # Setup tokenizer
    tokenizer = setup_tokenizer(model, xoron_config)

    # Save built model (only if not resuming)
    if not resume_training:
        save_model(model, tokenizer, training_config)

    if build_only:
        print("\n✅ Build complete! (skipping training)")
        return

    # Convert model to half precision for memory efficiency
    # This cuts model memory usage in half (fp32 -> fp16/bf16)
    if training_config.device == "cuda" and torch.cuda.is_available():
        if training_config.bf16:
            print("\n🔧 Converting model to bfloat16 for memory efficiency...")
            model = model.to(torch.bfloat16)
            print("   ✅ Model converted to bfloat16 (2x memory savings)")
        elif training_config.fp16:
            print("\n🔧 Converting model to float16 for memory efficiency...")
            model = safe_convert_to_fp16(model)
            print("   ✅ Model converted to float16 (2x memory savings)")
        
        # Clear cache after conversion
        clear_cuda_cache()

    # Check for streaming state to resume from
    resume_streaming_state = None
    if resume_training:
        streaming_state_path = os.path.join(training_config.output_dir, "streaming_state.json")
        if os.path.exists(streaming_state_path):
            resume_streaming_state = streaming_state_path
            print(f"\n📂 Found streaming state to resume from: {streaming_state_path}")

    # Setup training with filtered dataset configs
    train_dataset, eval_dataset, optimizer, scheduler, collate_fn = setup_training(
        model, tokenizer, xoron_config, training_config, dataset_configs, 
        active_modalities=active_modalities,
        resume_streaming_state=resume_streaming_state
    )

    # Create trainer with resume support
    trainer = XoronTrainer(
        model=model,
        train_dataset=train_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        config=training_config,
        xoron_config=xoron_config,
        collate_fn=collate_fn,
        resume_from=checkpoint_path if resume_training else None,
        tokenizer=tokenizer,  # Pass tokenizer for saving with checkpoints
        eval_dataset=eval_dataset,  # Pass eval dataset for validation after each epoch
    )

    trainer.train()

    # Export to ONNX if requested
    if export_onnx:
        from export import export_to_onnx as onnx_export
        onnx_export(
            model, xoron_config, training_config.final_model_dir, 
            device=device, quantize=True, quantize_bits=onnx_quant_bits
        )
    
    # Export to GGUF if requested
    if export_gguf:
        from export import export_to_gguf as gguf_export
        gguf_export(
            model, xoron_config, training_config.final_model_dir,
            quant_type=gguf_quant_type
        )

    # Run test if requested
    if run_test_after:
        run_test(model, tokenizer, training_config)

    print("\n🎉 BUILD AND TRAINING COMPLETE!")


def run_hf_training(
    hf_model_id: str,
    training_config,
    dataset_configs,
    freeze_components: List[str] = None,
    train_only_components: List[str] = None,
    resume_checkpoint: str = None,
    export_onnx: bool = False,
    export_gguf: bool = False,
    onnx_quant_bits: int = 4,
    gguf_quant_type: str = 'q4_k_m',
    run_test_after: bool = False,
    active_modalities: str = 'all',
):
    """
    Load model from HuggingFace and run training.
    
    If a checkpoint exists from previous training, it will resume from there.
    Otherwise, it loads fresh from HuggingFace.
    
    Args:
        hf_model_id: HuggingFace model identifier
        training_config: Training configuration
        dataset_configs: Dataset configurations
        freeze_components: Components to freeze during training
        train_only_components: Components to train (freezes others)
        resume_checkpoint: Path to checkpoint to resume from (optional)
        export_onnx: Whether to export to ONNX after training
        export_gguf: Whether to export to GGUF after training
        onnx_quant_bits: Quantization bits for ONNX (4 or 8)
        gguf_quant_type: GGUF quantization type
        run_test_after: Whether to run tests after training
    """
    print("\n" + "=" * 60)
    print("🚀 HUGGINGFACE MODEL TRAINING")
    print("=" * 60)
    
    # Check if we should resume from an existing checkpoint
    resume_from = None
    if resume_checkpoint:
        resume_from = resume_checkpoint
        print(f"\n📂 Will resume training from: {resume_checkpoint}")
    else:
        # Check for existing checkpoints to auto-resume
        checkpoints = list_available_checkpoints(training_config)
        if checkpoints:
            # Find the latest checkpoint with training state
            for path, name, has_state in reversed(checkpoints):
                if has_state:
                    print(f"\n📂 Found existing checkpoint with training state: {name}")
                    user_input = input("   Resume from this checkpoint? (y/n) [y]: ").strip().lower()
                    if user_input != 'n':
                        resume_from = path
                        print(f"   ✅ Will resume training from: {path}")
                    break
    
    # Load model - either from checkpoint or HuggingFace
    if resume_from:
        print(f"\n📂 Loading model from checkpoint: {resume_from}")
        model, device, xoron_config = load_model_from_checkpoint_with_config(resume_from, training_config)
    else:
        # Load model from HuggingFace
        model, device, xoron_config = load_model_from_huggingface(hf_model_id, training_config)
    
    # Print configurations
    xoron_config.print_config()
    training_config.print_config()
    
    # Print dataset config summary
    is_dual_training = getattr(training_config, 'is_dual_training', False)
    modality_max_values = getattr(training_config, 'modality_max_values', {})
    
    print("\n📊 Dataset Configuration:")
    total = 0
    for cat, datasets in dataset_configs.items():
        if datasets:
            # Show per-modality max if in dual training mode
            max_info = ""
            if is_dual_training and cat in modality_max_values:
                max_info = f" (max: {modality_max_values[cat]})"
            print(f"   {cat}: {len(datasets)} datasets{max_info}")
            total += len(datasets)
    print(f"   Total: {total} datasets")
    
    if is_dual_training:
        print(f"   Mode: Dual Training ({' + '.join(modality_max_values.keys())})")
        print(f"   Combined max_per_epoch: {training_config.max_per_epoch}")
    
    # Apply freezing
    if freeze_components or train_only_components:
        apply_freezing(model, freeze_components or [], train_only_components or [])
    
    print(f"\n✅ Model ready")
    print_component_status(model)

    # Setup tokenizer
    tokenizer = setup_tokenizer(model, xoron_config)

    # Save model before training (only if not resuming)
    if not resume_from:
        save_model(model, tokenizer, training_config)

    # Convert model to half precision for memory efficiency
    # This cuts model memory usage in half (fp32 -> fp16/bf16)
    if training_config.device == "cuda" and torch.cuda.is_available():
        if training_config.bf16:
            print("\n🔧 Converting model to bfloat16 for memory efficiency...")
            model = model.to(torch.bfloat16)
            print("   ✅ Model converted to bfloat16 (2x memory savings)")
        elif training_config.fp16:
            print("\n🔧 Converting model to float16 for memory efficiency...")
            model = safe_convert_to_fp16(model)
            print("   ✅ Model converted to float16 (2x memory savings)")
        
        # Clear cache after conversion
        clear_cuda_cache()

    # Check for streaming state to resume from
    resume_streaming_state = None
    if resume_from:
        streaming_state_path = os.path.join(training_config.output_dir, "streaming_state.json")
        if os.path.exists(streaming_state_path):
            resume_streaming_state = streaming_state_path
            print(f"\n📂 Found streaming state to resume from: {streaming_state_path}")

    # Setup training with filtered dataset configs
    train_dataset, eval_dataset, optimizer, scheduler, collate_fn = setup_training(
        model, tokenizer, xoron_config, training_config, dataset_configs, 
        active_modalities=active_modalities,
        resume_streaming_state=resume_streaming_state
    )

    # Create trainer with resume support
    trainer = XoronTrainer(
        model=model,
        train_dataset=train_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        config=training_config,
        xoron_config=xoron_config,
        collate_fn=collate_fn,
        resume_from=resume_from,  # Pass checkpoint path for resuming
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,  # Pass eval dataset for validation after each epoch
    )

    trainer.train()

    # Export to ONNX if requested
    if export_onnx:
        from export import export_to_onnx as onnx_export
        onnx_export(
            model, xoron_config, training_config.final_model_dir, 
            device=device, quantize=True, quantize_bits=onnx_quant_bits
        )
    
    # Export to GGUF if requested
    if export_gguf:
        from export import export_to_gguf as gguf_export
        gguf_export(
            model, xoron_config, training_config.final_model_dir,
            quant_type=gguf_quant_type
        )

    # Run test if requested
    if run_test_after:
        run_test(model, tokenizer, training_config)

    print("\n🎉 HUGGINGFACE MODEL TRAINING COMPLETE!")


def load_model_from_checkpoint_with_config(checkpoint_path, training_config):
    """
    Load model from a checkpoint and return the config as well.
    
    Args:
        checkpoint_path: Path to checkpoint
        training_config: Training configuration
        
    Returns:
        Tuple of (model, device, xoron_config)
    """
    print("\n" + "=" * 60)
    print(f"📂 LOADING MODEL FROM CHECKPOINT: {checkpoint_path}")
    print("=" * 60)
    
    # Load config from checkpoint
    config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Clean up config dict
        config_dict.pop('lora_applied', None)
        config_dict.pop('has_audio_encoder', None)
        config_dict.pop('has_audio_decoder', None)
        
        xoron_config = XoronConfig.from_dict(config_dict)
    else:
        # Fall back to default config
        xoron_config = XoronConfig()
    
    # Get device map for model parallelism
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_map = None

    if training_config.use_model_parallel and num_gpus > 1:
        from config.training_config import get_device_map
        device_map = get_device_map(num_gpus)
        print(f"⚡ Model Parallelism enabled across {num_gpus} GPUs")
    
    # Determine device
    if training_config.use_model_parallel and num_gpus > 1:
        device = device_map['primary']
    else:
        device = training_config.device
    
    model = XoronMultimodalModel.from_pretrained(
        checkpoint_path,
        device=device if device_map is None else None,
        device_map=device_map,
        apply_lora=xoron_config.use_lora,
    )
    
    clear_cuda_cache()
    
    return model, device, xoron_config


def main_menu():
    """Main interactive menu."""
    while True:
        clear_screen()
        print_banner()
        print_device_info()
        
        print_header("XORON-DEV BUILD & TRAIN")
        
        print("""
Welcome to the Xoron-Dev build and training tool!

This tool allows you to:
  - Build a new model from scratch
  - Load and continue training from checkpoints
  - Fine-tune with frozen components
  - Select specific datasets for training
""")
        
        options = [
            "🚀 Build New Model & Train",
            "📂 Load Checkpoint & Continue Training",
            "🎯 Fine-tune Model (Load + Freeze Components)",
            "🔨 Build Model Only (No Training)",
            "📋 List Available Checkpoints",
            "⚙️ Run Setup (Configure Settings)",
        ]
        
        print_menu(options, "Main Menu")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == '0':
            print("\nGoodbye! 👋")
            break
        elif choice == '1':
            # Build new model and train
            run_new_build_workflow()
        elif choice == '2':
            # Load checkpoint and continue
            run_continue_training_workflow()
        elif choice == '3':
            # Fine-tune workflow
            run_finetune_workflow()
        elif choice == '4':
            # Build only
            run_build_only_workflow()
        elif choice == '5':
            # List checkpoints
            run_list_checkpoints()
        elif choice == '6':
            # Run setup
            run_setup()
        
        input("\nPress Enter to continue...")


def run_new_build_workflow():
    """Workflow for building a new model and training."""
    print_header("BUILD NEW MODEL & TRAIN")
    
    # Load configs
    xoron_config, training_config, finetune_config, dataset_configs = load_configs_from_file()
    
    # Ask for fine-tune mode
    print("\nWould you like to filter datasets by modality?")
    use_filter = get_input("Filter datasets? (y/n)", "n")
    
    mode = 'all'
    if use_filter.lower() in ('y', 'yes'):
        mode = select_finetune_mode_menu()
        if mode != 'all':
            dataset_configs = get_finetune_datasets(mode)
            print(f"\n✅ Using {mode} datasets")
    
    # Confirm and run
    print("\n" + "=" * 60)
    print("Ready to build and train with current configuration.")
    confirm = get_input("Proceed? (y/n)", "y")
    
    if confirm.lower() in ('y', 'yes'):
        run_build_and_train(
            xoron_config,
            training_config,
            dataset_configs,
            active_modalities=mode,
        )


def run_continue_training_workflow():
    """Workflow for continuing training from a checkpoint."""
    print_header("CONTINUE TRAINING FROM CHECKPOINT")
    
    # Load configs
    xoron_config, training_config, finetune_config, dataset_configs = load_configs_from_file()
    
    # Select checkpoint
    checkpoint_path = select_checkpoint_menu(training_config)
    
    if not checkpoint_path:
        print("\n❌ No checkpoint selected.")
        return
    
    print(f"\n✅ Selected: {checkpoint_path}")
    
    # Ask about dataset filtering
    print("\nWould you like to filter datasets by modality?")
    use_filter = get_input("Filter datasets? (y/n)", "n")
    
    mode = 'all'
    if use_filter.lower() in ('y', 'yes'):
        mode = select_finetune_mode_menu()
        if mode != 'all':
            dataset_configs = get_finetune_datasets(mode)
    
    # Confirm and run
    print("\n" + "=" * 60)
    print(f"Ready to continue training from: {checkpoint_path}")
    confirm = get_input("Proceed? (y/n)", "y")
    
    if confirm.lower() in ('y', 'yes'):
        run_build_and_train(
            xoron_config,
            training_config,
            dataset_configs,
            checkpoint_path=checkpoint_path,
            resume_training=True,
            active_modalities=mode,
        )


def run_finetune_workflow():
    """Workflow for fine-tuning with frozen components."""
    print_header("FINE-TUNE MODEL")
    
    # Load configs
    xoron_config, training_config, finetune_config, dataset_configs = load_configs_from_file()
    
    # Select checkpoint to load
    print("\nFirst, select a model to fine-tune:")
    checkpoint_path = select_checkpoint_menu(training_config)
    
    if not checkpoint_path:
        print("\n❌ No checkpoint selected.")
        return
    
    print(f"\n✅ Selected: {checkpoint_path}")
    
    # Select fine-tune mode (datasets)
    mode = select_finetune_mode_menu()
    if mode != 'all':
        dataset_configs = get_finetune_datasets(mode)
    
    # Select freezing strategy
    print_header("FREEZING STRATEGY")
    print("\nChoose how to freeze components:\n")
    print("  [1] Select components to FREEZE (train everything else)")
    print("  [2] Select components to TRAIN (freeze everything else)")
    print("  [3] No freezing (train all components)")
    print("  [0] Cancel")
    
    freeze_choice = get_input("\nSelect strategy", "3")
    
    freeze_components = []
    train_only_components = []
    
    if freeze_choice == '1':
        freeze_components = select_components_to_freeze_menu()
    elif freeze_choice == '2':
        train_only_components = select_components_to_train_menu()
    elif freeze_choice == '0':
        return
    
    # Summary
    print("\n" + "=" * 60)
    print("FINE-TUNE CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"  Model: {checkpoint_path}")
    print(f"  Dataset mode: {mode}")
    if freeze_components:
        print(f"  Frozen components: {freeze_components}")
    if train_only_components:
        print(f"  Training only: {train_only_components}")
    if not freeze_components and not train_only_components:
        print(f"  Training all components")
    
    confirm = get_input("\nProceed with fine-tuning? (y/n)", "y")
    
    if confirm.lower() in ('y', 'yes'):
        run_build_and_train(
            xoron_config,
            training_config,
            dataset_configs,
            checkpoint_path=checkpoint_path,
            resume_training=False,  # Fresh training state for fine-tuning
            freeze_components=freeze_components,
            train_only_components=train_only_components,
            active_modalities=mode,
        )


def run_build_only_workflow():
    """Workflow for building model without training."""
    print_header("BUILD MODEL ONLY")
    
    # Load configs
    xoron_config, training_config, finetune_config, dataset_configs = load_configs_from_file()
    
    print("\nThis will build the model and save it without training.")
    confirm = get_input("Proceed? (y/n)", "y")
    
    if confirm.lower() in ('y', 'yes'):
        run_build_and_train(
            xoron_config,
            training_config,
            dataset_configs,
            build_only=True,
        )


def run_list_checkpoints():
    """List available checkpoints."""
    print_header("AVAILABLE CHECKPOINTS")
    
    # Load configs
    _, training_config, _, _ = load_configs_from_file()
    
    checkpoints = list_available_checkpoints(training_config)
    
    if not checkpoints:
        print("\n  ❌ No checkpoints found.")
        print(f"\n  Searched in:")
        print(f"    - {training_config.output_dir}")
        print(f"    - {training_config.final_model_dir}")
        print(f"    - {training_config.model_path}")
    else:
        print("\n  Found checkpoints:\n")
        for path, name, has_state in checkpoints:
            state_str = "✅ has training state" if has_state else "❌ no training state"
            print(f"  📁 {name}")
            print(f"     Path: {path}")
            print(f"     {state_str}")
            print()


def run_setup():
    """Run the setup script."""
    print("\nLaunching setup script...")
    os.system(f"{sys.executable} {os.path.join(os.path.dirname(__file__), 'setup.py')}")


def parse_args():
    """Parse command line arguments for non-interactive mode."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Xoron-Dev Build and Training Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build.py                    # Interactive mode (default)
  python build.py --build            # Build new model and train
  python build.py --build-only       # Build model without training
  python build.py --resume <path>    # Resume training from checkpoint
  python build.py --finetune <path>  # Fine-tune from checkpoint
  python build.py --list             # List available checkpoints
  
Modality-specific training (can combine multiple flags):
  python build.py --video            # Train on video datasets only
  python build.py --image            # Train on image datasets only
  python build.py --text             # Train on text datasets only
  python build.py --voice            # Train on voice/audio datasets only
  
  # Combined modality training:
  python build.py --text --image     # Train on text + image datasets
  python build.py --text --video     # Train on text + video datasets
  python build.py --image --video    # Train on image + video datasets
  python build.py --text --image --video  # Train on text + image + video
  python build.py --text --image --video --voice  # Same as no flags (all)
  
  # With max samples per epoch override (global):
  python build.py --image --max 500  # Train image with 500 samples/epoch
  python build.py --text --max 3300  # Train text with 3300 samples/epoch
  
  # Per-modality max samples (for combined training):
  python build.py --text --max-text 3300 --image --max-image 500
  python build.py --text --max-text 3300 --video --max-video 1000
  python build.py --hf --text --max-text 3300 --image --max-image 500 --video --max-video 1000

HuggingFace model training:
  python build.py --hf --text        # Load from HF and train on text
  python build.py --hf --image       # Load from HF and train on images
  python build.py --hf --video       # Load from HF and train on video
  python build.py --hf --voice       # Load from HF and train on voice/audio
  python build.py --hf --text --image  # Load from HF and train on text + image
  python build.py --hf               # Load from HF and train on all modalities

Export options:
  python build.py --build --onnx     # Build, train, and export to ONNX
  python build.py --hf --onnx        # Load from HF, train, and export to ONNX
  python build.py --build --gguf     # Build, train, and export to GGUF
  python build.py --build --onnx --gguf  # Export to both formats
        """
    )
    
    # Mode selection
    parser.add_argument('--build', action='store_true', 
                       help='Build new model and start training (non-interactive)')
    parser.add_argument('--build-only', action='store_true',
                       help='Build model without training')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint path')
    parser.add_argument('--finetune', type=str, default=None,
                       help='Fine-tune from checkpoint path')
    parser.add_argument('--list', action='store_true',
                       help='List available checkpoints')
    
    # HuggingFace model loading
    parser.add_argument('--hf', action='store_true',
                       help='Load pretrained model from HuggingFace (Backup-bdg/Xoron-Dev-MultiMoe)')
    parser.add_argument('--hf-model', type=str, default='Backup-bdg/Xoron-Dev-MultiMoe',
                       help='HuggingFace model ID (default: Backup-bdg/Xoron-Dev-MultiMoe)')
    
    # Modality-specific shorthand flags
    parser.add_argument('--video', action='store_true',
                       help='Train on video datasets only (shorthand for --mode video)')
    parser.add_argument('--image', action='store_true',
                       help='Train on image datasets only (shorthand for --mode image)')
    parser.add_argument('--text', action='store_true',
                       help='Train on text datasets only (shorthand for --mode text)')
    parser.add_argument('--voice', action='store_true',
                       help='Train on voice/audio datasets only (shorthand for --mode audio)')
    
    # Fine-tuning options
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'text', 'image', 'video', 'audio', 'vision', 'generation'],
                       help='Dataset mode for training')
    parser.add_argument('--freeze', type=str, default=None,
                       help='Comma-separated list of components to freeze')
    parser.add_argument('--train-only', type=str, default=None,
                       help='Comma-separated list of components to train (freezes others)')
    
    # Training overrides
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--max', type=int, default=None,
                       help='Override max samples per epoch (global, used if per-modality not set)')
    
    # Per-modality max samples (for combined training with different limits)
    parser.add_argument('--max-text', type=int, default=None,
                       help='Max samples per epoch for text modality')
    parser.add_argument('--max-image', type=int, default=None,
                       help='Max samples per epoch for image modality')
    parser.add_argument('--max-video', type=int, default=None,
                       help='Max samples per epoch for video modality')
    parser.add_argument('--max-voice', type=int, default=None,
                       help='Max samples per epoch for voice/audio modality')
    
    # Export options
    parser.add_argument('--onnx', action='store_true',
                       help='Export model to ONNX format after training (with 4-bit quantization)')
    parser.add_argument('--gguf', action='store_true',
                       help='Export model to GGUF format after training (for llama.cpp)')
    parser.add_argument('--quant-bits', type=int, default=4, choices=[4, 8],
                       help='Quantization bits for ONNX export (default: 4)')
    parser.add_argument('--gguf-quant', type=str, default='q4_k_m',
                       choices=['q4_0', 'q4_k_m', 'q5_k_m', 'q8_0', 'f16'],
                       help='GGUF quantization type (default: q4_k_m)')
    
    return parser.parse_args()


def run_cli_mode(args):
    """Run in CLI (non-interactive) mode based on arguments."""
    
    # Load configs
    xoron_config, training_config, finetune_config, dataset_configs = load_configs_from_file()
    
    # Apply training overrides
    if args.epochs:
        training_config.num_epochs = args.epochs
    if args.batch_size:
        training_config.batch_size = args.batch_size
    if args.lr:
        training_config.learning_rate = args.lr
    
    # Determine the effective mode(s) from shorthand flags or --mode argument
    # Supports combining multiple flags: --text --image --video
    active_modes = []
    modality_max_values = {}  # Store per-modality max values
    
    # Collect all active mode flags and their max values
    if args.text:
        active_modes.append('text')
        if args.max_text:
            modality_max_values['text'] = args.max_text
    if args.image:
        active_modes.append('image')
        if args.max_image:
            modality_max_values['image'] = args.max_image
    if args.video:
        active_modes.append('video')
        if args.max_video:
            modality_max_values['video'] = args.max_video
    if args.voice:
        active_modes.append('audio')  # --voice maps to 'audio' mode
        if args.max_voice:
            modality_max_values['audio'] = args.max_voice
    
    # Calculate total max_per_epoch from per-modality values or use global --max
    if modality_max_values:
        # Sum up all per-modality max values for total max_per_epoch
        total_max = sum(modality_max_values.values())
        training_config.max_per_epoch = total_max
        # Store per-modality limits for dataset filtering
        training_config.modality_max_values = modality_max_values
        # Mark as dual/multi training mode
        training_config.is_dual_training = len(modality_max_values) >= 2
    elif args.max:
        # Use global --max if no per-modality values
        training_config.max_per_epoch = args.max
    
    # If no flags specified, use --mode argument or default to 'all'
    if not active_modes:
        if args.mode and args.mode != 'all':
            active_modes = [args.mode]
        else:
            active_modes = []  # Empty means 'all'
    
    # For backwards compatibility, set effective_mode for single mode or 'multi' for multiple
    if len(active_modes) == 0:
        effective_mode = 'all'
    elif len(active_modes) == 1:
        effective_mode = active_modes[0]
    else:
        effective_mode = 'multi'  # Multiple modes combined
    
    # Apply dataset mode - use filter_datasets_by_modalities for multi-mode support
    if active_modes:
        from config.dataset_config import filter_datasets_by_modalities
        dataset_configs = filter_datasets_by_modalities(modalities=active_modes)
        mode_str = ' + '.join(active_modes)
        print(f"\n📊 Using {mode_str} datasets")
    else:
        print(f"\n📊 Using all datasets")
    
    # Parse freeze/train-only components
    freeze_components = []
    train_only_components = []
    
    if args.freeze:
        freeze_components = [c.strip() for c in args.freeze.split(',')]
    if args.train_only:
        train_only_components = [c.strip() for c in args.train_only.split(',')]
    
    # Apply auto-freeze based on modality flags (applies to ALL modes: --build, --hf, etc.)
    # All component groups: vision, video, audio, llm, cross_attention, image_generation, video_generation, modality_markers
    # 
    # IMPORTANT: LLM is NEVER frozen - it's trained from scratch and always needs full weight training + LoRA
    # Only non-LLM components are selectively frozen based on training mode
    # 
    # Supports combining flags: --text --image will train text + image components
    auto_freeze = []
    
    # All possible components that can be frozen (excluding LLM which is never frozen)
    all_freezable = {'vision', 'video', 'audio', 'image_generation', 'video_generation'}
    
    # Determine which components to TRAIN based on active modes
    components_to_train = set()
    
    if 'text' in active_modes:
        # Text mode: train LLM only (no additional components)
        pass  # LLM is always trained
    if 'image' in active_modes:
        # Image mode: train vision + image_generation
        components_to_train.update(['vision', 'image_generation'])
    if 'video' in active_modes:
        # Video mode: train video + vision (for frame encoding) + video_generation
        components_to_train.update(['vision', 'video', 'video_generation'])
    if 'audio' in active_modes:
        # Audio/voice mode: train audio
        components_to_train.add('audio')
    
    # If specific modes are active, freeze everything NOT being trained
    if active_modes:
        auto_freeze = list(all_freezable - components_to_train)
    # If no modes specified (all), don't auto-freeze anything
    
    # NEVER freeze LLM - remove it if somehow added
    if 'llm' in auto_freeze:
        auto_freeze.remove('llm')
    
    # Combine auto-freeze with any user-specified freeze components
    if auto_freeze:
        freeze_components = list(set(freeze_components + auto_freeze))
    
    # Safety: ensure LLM is never in freeze list
    if 'llm' in freeze_components:
        freeze_components.remove('llm')
        print("⚠️ Removed 'llm' from freeze list - LLM must always train (from scratch)")
    
    # List checkpoints
    if args.list:
        print_header("AVAILABLE CHECKPOINTS")
        checkpoints = list_available_checkpoints(training_config)
        if not checkpoints:
            print("\n  ❌ No checkpoints found.")
        else:
            for path, name, has_state in checkpoints:
                state_str = "✅ has training state" if has_state else "❌ no training state"
                print(f"  📁 {name}: {path} ({state_str})")
        return
    
    # Build new model
    if args.build or args.build_only:
        print_banner()
        print_device_info()
        run_build_and_train(
            xoron_config,
            training_config,
            dataset_configs,
            build_only=args.build_only,
            freeze_components=freeze_components,
            train_only_components=train_only_components,
            export_onnx=args.onnx,
            export_gguf=args.gguf,
            onnx_quant_bits=args.quant_bits,
            gguf_quant_type=args.gguf_quant,
            active_modalities=effective_mode or 'all',
        )
        return
    
    # Resume training
    if args.resume:
        print_banner()
        print_device_info()
        run_build_and_train(
            xoron_config,
            training_config,
            dataset_configs,
            checkpoint_path=args.resume,
            resume_training=True,
            freeze_components=freeze_components,
            train_only_components=train_only_components,
            export_onnx=args.onnx,
            export_gguf=args.gguf,
            onnx_quant_bits=args.quant_bits,
            gguf_quant_type=args.gguf_quant,
            active_modalities=effective_mode or 'all',
        )
        return
    
    # Fine-tune
    if args.finetune:
        print_banner()
        print_device_info()
        run_build_and_train(
            xoron_config,
            training_config,
            dataset_configs,
            checkpoint_path=args.finetune,
            resume_training=False,
            freeze_components=freeze_components,
            train_only_components=train_only_components,
            export_onnx=args.onnx,
            export_gguf=args.gguf,
            onnx_quant_bits=args.quant_bits,
            gguf_quant_type=args.gguf_quant,
            active_modalities=effective_mode or 'all',
        )
        return
    
    # HuggingFace model training
    if args.hf:
        modality_name = 'video' if args.video else ('image' if args.image else ('text' if args.text else ('voice/audio' if args.voice else 'all modalities')))
        
        print_banner()
        print_device_info()
        print(f"\n🎯 Loading model from HuggingFace and training on {modality_name}...")
        run_hf_training(
            hf_model_id=args.hf_model,
            training_config=training_config,
            dataset_configs=dataset_configs,
            freeze_components=freeze_components,  # Already has auto-freeze applied
            train_only_components=train_only_components,
            export_onnx=args.onnx,
            export_gguf=args.gguf,
            onnx_quant_bits=args.quant_bits,
            gguf_quant_type=args.gguf_quant,
            active_modalities=effective_mode or 'all',
        )
        return
    
    # Modality-specific training (--video, --image, --text, --voice) without --build or --hf
    # This is for when ONLY a modality flag is passed without --build
    if args.video or args.image or args.text or args.voice:
        modality_name = 'video' if args.video else ('image' if args.image else ('text' if args.text else 'voice/audio'))
        
        print_banner()
        print_device_info()
        print(f"\n🎯 Starting {modality_name} training mode...")
        run_build_and_train(
            xoron_config,
            training_config,
            dataset_configs,
            build_only=False,
            freeze_components=freeze_components,  # Already has auto-freeze applied
            train_only_components=train_only_components,
            export_onnx=args.onnx,
            export_gguf=args.gguf,
            onnx_quant_bits=args.quant_bits,
            gguf_quant_type=args.gguf_quant,
            active_modalities=effective_mode or 'all',
        )
        return
    
    # No CLI args - run interactive menu
    return False


def main():
    """Main entry point - supports both CLI and interactive modes."""
    args = parse_args()
    
    # Check if any CLI mode flags are set
    # Include modality shorthand flags (--video, --image, --text, --voice) and --hf
    cli_mode = (args.build or args.build_only or args.resume or args.finetune or args.list or
                args.video or args.image or args.text or args.voice or args.hf)
    
    if cli_mode:
        run_cli_mode(args)
    else:
        # Run interactive menu
        main_menu()


if __name__ == "__main__":
    main()

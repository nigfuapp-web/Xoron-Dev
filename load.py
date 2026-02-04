#!/usr/bin/env python3
"""
Xoron-Dev Model Loader
======================

This script loads a trained Xoron-Dev multimodal model from HuggingFace or local storage.
It properly sets up all multimodal components (text, image, video, audio) for inference.

Usage:
    python load.py                          # Interactive mode
    python load.py --from-hf                # Load from HuggingFace
    python load.py --from-local ./path      # Load from local directory
    python load.py --test                   # Run tests after loading

Features:
    - Loads model from HuggingFace: Backup-bdg/Xoron-Dev-MultiMoe
    - Sets up all multimodal encoders (vision, video, audio)
    - Configures generation modules (image diffusion, video diffusion)
    - Supports model parallelism across multiple GPUs
    - Provides inference examples for all modalities
"""

import os
import sys
import argparse
import torch
from typing import Optional, Dict

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import XoronConfig
from models.xoron import XoronMultimodalModel


def _load_model_with_vocab_fix(
    path: str,
    device: str = None,
    device_map: Optional[Dict[str, str]] = None,
    apply_lora: bool = True,
) -> XoronMultimodalModel:
    """
    Load model with automatic vocab size detection and fix.
    
    This handles the case where the checkpoint has a different vocab size
    than the default config (e.g., checkpoint has 151930 but config has 151643).
    
    It also handles checkpoints saved with LoRA already applied by detecting
    LoRA keys in the state dict and applying LoRA before loading weights.
    
    Args:
        path: Path to the model directory
        device: Device to load model to
        device_map: Device map for model parallelism
        apply_lora: Whether to apply LoRA after loading
        
    Returns:
        Loaded XoronMultimodalModel instance
    """
    import json
    import os
    
    print(f"\n📂 Loading model from {path}...")
    
    # Load config
    config_path = os.path.join(path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Check if LoRA was applied when saving
    lora_was_applied = config_dict.pop('lora_applied', False)
    config_dict.pop('has_audio_encoder', None)
    config_dict.pop('has_audio_decoder', None)
    
    # Detect vocab size and LoRA structure from checkpoint weights
    model_path = os.path.join(path, "model.safetensors")
    pytorch_path = os.path.join(path, "pytorch_model.bin")
    checkpoint_vocab_size = None
    checkpoint_has_lora_structure = False
    
    if os.path.exists(model_path):
        try:
            from safetensors import safe_open
            with safe_open(model_path, framework="pt") as f:
                keys = list(f.keys())
                # Check for LoRA structure in keys
                checkpoint_has_lora_structure = any('.lora_A' in k or '.lora_B' in k or '.linear.weight' in k for k in keys)
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
            # Check for LoRA structure
            checkpoint_has_lora_structure = any('.lora_A' in k or '.lora_B' in k or '.linear.weight' in k for k in keys)
            if checkpoint_has_lora_structure:
                print(f"   🔧 Detected LoRA structure in checkpoint")
            
            # Detect vocab size
            for key in keys:
                if 'embed_tokens.weight' in key or 'lm_head.weight' in key:
                    checkpoint_vocab_size = state_dict[key].shape[0]
                    print(f"   📊 Detected checkpoint vocab size: {checkpoint_vocab_size}")
                    break
        except Exception as e:
            print(f"   ⚠️ Could not inspect pytorch_model.bin: {e}")
    
    # Update config vocab_size if checkpoint has different size
    config_vocab_size = config_dict.get('vocab_size', 151643)
    if checkpoint_vocab_size is not None and checkpoint_vocab_size != config_vocab_size:
        print(f"   🔄 Updating vocab_size from {config_vocab_size} to {checkpoint_vocab_size}")
        config_dict['vocab_size'] = checkpoint_vocab_size
    
    # Create config and model
    config = XoronConfig.from_dict(config_dict)
    model = XoronMultimodalModel(config, device_map=device_map)
    
    # If checkpoint has LoRA structure, apply LoRA BEFORE loading weights
    if checkpoint_has_lora_structure and config.use_lora:
        print(f"   🔧 Applying LoRA before loading weights...")
        model.apply_lora()
    
    # Load weights
    if os.path.exists(model_path):
        print(f"   📦 Loading weights from safetensors...")
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
        # Use strict=False to handle missing/extra keys (e.g., _dtype_tracker, audio_decoder)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            # Filter out known non-critical missing keys
            critical_missing = [k for k in missing_keys if '_dtype_tracker' not in k]
            if critical_missing:
                print(f"   ⚠️ Missing keys: {len(critical_missing)} (showing first 5)")
                for k in critical_missing[:5]:
                    print(f"      - {k}")
        if unexpected_keys:
            print(f"   ⚠️ Unexpected keys (ignored): {len(unexpected_keys)}")
        model.lora_applied = lora_was_applied or checkpoint_has_lora_structure
    elif os.path.exists(pytorch_path):
        print(f"   📦 Loading weights from pytorch_model.bin...")
        if 'state_dict' not in dir():
            state_dict = torch.load(pytorch_path, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            critical_missing = [k for k in missing_keys if '_dtype_tracker' not in k]
            if critical_missing:
                print(f"   ⚠️ Missing keys: {len(critical_missing)} (showing first 5)")
                for k in critical_missing[:5]:
                    print(f"      - {k}")
        if unexpected_keys:
            print(f"   ⚠️ Unexpected keys (ignored): {len(unexpected_keys)}")
        model.lora_applied = lora_was_applied or checkpoint_has_lora_structure
    else:
        raise FileNotFoundError(f"No model weights found at {path}")
    
    # Apply LoRA if requested and not already applied (for checkpoints without LoRA)
    if apply_lora and config.use_lora and not model.lora_applied:
        model.apply_lora()
    
    # Move to device
    if device_map is not None:
        model.apply_model_parallel(device_map)
    elif device is not None:
        model = model.to(device)
    
    print(f"✅ Model loaded successfully!")
    model._print_stats()
    
    return model


def load_from_huggingface(
    model_name: str = "Backup-bdg/Xoron-Dev-MultiMoe",
    device: str = None,
    device_map: Optional[Dict[str, str]] = None,
    use_fp16: bool = True,
) -> XoronMultimodalModel:
    """
    Load Xoron-Dev model from HuggingFace.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model to (e.g., 'cuda:0', 'cpu')
        device_map: Device map for model parallelism
        use_fp16: Whether to use half precision
        
    Returns:
        Loaded XoronMultimodalModel instance
    """
    print("\n" + "=" * 70)
    print("📥 LOADING XORON-DEV MODEL FROM HUGGINGFACE")
    print("=" * 70)
    print(f"Model: {model_name}")
    
    try:
        from huggingface_hub import snapshot_download
        import json
        
        # Download model from HuggingFace
        print(f"\n🔄 Downloading model from HuggingFace...")
        cache_dir = snapshot_download(
            repo_id=model_name,
            repo_type="model",
            local_files_only=False,
        )
        print(f"✅ Model downloaded to: {cache_dir}")
        
        # Load the model using custom loading to handle vocab size mismatch
        model = _load_model_with_vocab_fix(
            path=cache_dir,
            device=device,
            device_map=device_map,
            apply_lora=True,
        )
        
        # Apply FP16 if requested
        if use_fp16 and device and 'cuda' in device:
            model = model.half()
            print("✅ Applied FP16 (half precision)")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading from HuggingFace: {e}")
        print("\n💡 Alternative: Download manually and use --from-local")
        raise


def load_from_local(
    path: str,
    device: str = None,
    device_map: Optional[Dict[str, str]] = None,
    use_fp16: bool = True,
) -> XoronMultimodalModel:
    """
    Load Xoron-Dev model from local directory.
    
    Args:
        path: Path to local model directory
        device: Device to load model to
        device_map: Device map for model parallelism
        use_fp16: Whether to use half precision
        
    Returns:
        Loaded XoronMultimodalModel instance
    """
    print("\n" + "=" * 70)
    print("📂 LOADING XORON-DEV MODEL FROM LOCAL STORAGE")
    print("=" * 70)
    print(f"Path: {path}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model directory not found: {path}")
    
    # Use custom loading to handle vocab size mismatch
    model = _load_model_with_vocab_fix(
        path=path,
        device=device,
        device_map=device_map,
        apply_lora=True,
    )
    
    # Apply FP16 if requested
    if use_fp16 and device and 'cuda' in device:
        model = model.half()
        print("✅ Applied FP16 (half precision)")
    
    return model


def setup_device_map(num_gpus: int) -> Dict[str, str]:
    """
    Setup device map for multi-GPU inference.
    
    Args:
        num_gpus: Number of GPUs available
        
    Returns:
        Device map dictionary
    """
    if num_gpus <= 1:
        return None
    
    print(f"\n⚡ Setting up model parallelism for {num_gpus} GPUs...")
    
    # Use the same device map as in training
    from config.training_config import get_device_map
    device_map = get_device_map(num_gpus)
    
    for component, device in device_map.items():
        print(f"   {component:20s} -> {device}")
    
    return device_map


def run_generation_tests(model: XoronMultimodalModel, device: str, output_dir: str = "./generated_outputs"):
    """
    Run full generation tests on the loaded model and save outputs.
    
    This function actually generates images, videos, and text, saving them to disk.
    
    Args:
        model: Loaded XoronMultimodalModel
        device: Device to run tests on
        output_dir: Directory to save generated outputs
    """
    import os
    from datetime import datetime
    
    print("\n" + "=" * 70)
    print("🚀 RUNNING GENERATION TESTS")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model.eval()
    
    # Detect model dtype for proper input casting
    model_dtype = next(model.parameters()).dtype
    print(f"\n📊 Model dtype: {model_dtype}")
    
    # Load tokenizer once
    from transformers import AutoTokenizer
    tokenizer_name = model.config.tokenizer_name
    print(f"📝 Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    # ========== TEXT GENERATION ==========
    print("\n" + "-" * 50)
    print("📝 TEXT GENERATION TEST")
    print("-" * 50)
    
    text_prompts = [
        "Write a short poem about the ocean:",
        "Explain quantum computing in simple terms:",
        "What is the capital of France?",
    ]
    
    # Generation parameters
    max_new_tokens = 256
    temperature = 0.7
    top_p = 0.9
    top_k = 50
    
    print(f"   Generation config: max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}, top_k={top_k}")
    
    for i, prompt in enumerate(text_prompts):
        print(f"\n   Prompt {i+1}: {prompt}")
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # Proper autoregressive text generation
            with torch.no_grad():
                generated_ids = input_ids.clone()
                
                for _ in range(max_new_tokens):
                    # Forward pass
                    if 'cuda' in device and model_dtype == torch.float16:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            outputs = model(
                                input_ids=generated_ids,
                                attention_mask=torch.ones_like(generated_ids),
                            )
                    else:
                        outputs = model(
                            input_ids=generated_ids,
                            attention_mask=torch.ones_like(generated_ids),
                        )
                    
                    # Get logits for the last token
                    next_token_logits = outputs.logits[:, -1, :].float()
                    
                    # Apply temperature
                    if temperature > 0:
                        next_token_logits = next_token_logits / temperature
                    
                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample from the distribution
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Append to generated sequence
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    
                    # Check for EOS token
                    if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                        break
                
                # Decode the generated text (excluding the prompt)
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                response_text = generated_text[len(prompt):].strip() if generated_text.startswith(prompt) else generated_text
            
            print(f"   Response: {response_text[:300]}...")
            
            # Save to file
            text_file = os.path.join(output_dir, f"text_{timestamp}_{i+1}.txt")
            with open(text_file, 'w') as f:
                f.write(f"Prompt: {prompt}\n\nResponse:\n{response_text}\n\nFull output:\n{generated_text}")
            print(f"   ✅ Saved to: {text_file}")
            
        except Exception as e:
            print(f"   ⚠️ Text generation failed: {e}")
    
    # ========== IMAGE GENERATION ==========
    if model.generator is not None:
        print("\n" + "-" * 50)
        print("🎨 IMAGE GENERATION TEST")
        print("-" * 50)
        
        image_prompts = [
            "A beautiful sunset over mountains with orange and purple sky",
            "A cute cat sitting on a windowsill",
            "A futuristic city with flying cars",
        ]
        
        for i, prompt in enumerate(image_prompts):
            print(f"\n   Prompt {i+1}: {prompt}")
            try:
                inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                
                # Use autocast for mixed precision to handle dtype mismatches
                with torch.no_grad():
                    if 'cuda' in device and model_dtype == torch.float16:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            generated_image = model.generate_image(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                            )
                    else:
                        generated_image = model.generate_image(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                
                print(f"   Generated shape: {generated_image.shape}")
                
                # Save image
                try:
                    from PIL import Image
                    import numpy as np
                    
                    # Convert tensor to image (assuming BCHW format)
                    img_tensor = generated_image[0].cpu().float()
                    if img_tensor.dim() == 3:
                        # Normalize to 0-255
                        img_tensor = img_tensor.clamp(0, 1)
                        img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        img = Image.fromarray(img_array)
                        
                        img_file = os.path.join(output_dir, f"image_{timestamp}_{i+1}.png")
                        img.save(img_file)
                        print(f"   ✅ Saved to: {img_file}")
                except Exception as save_e:
                    print(f"   ⚠️ Could not save image: {save_e}")
                    
            except Exception as e:
                print(f"   ⚠️ Image generation failed: {e}")
    else:
        print("\n   ⚠️ Image generator not available")
    
    # ========== VIDEO GENERATION ==========
    if model.video_generator is not None:
        print("\n" + "-" * 50)
        print("🎬 VIDEO GENERATION TEST")
        print("-" * 50)
        
        video_prompts = [
            "A dog running through a field of flowers",
            "Ocean waves crashing on a beach",
        ]
        
        for i, prompt in enumerate(video_prompts):
            print(f"\n   Prompt {i+1}: {prompt}")
            try:
                inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                
                # Use autocast for mixed precision to handle dtype mismatches
                with torch.no_grad():
                    if 'cuda' in device and model_dtype == torch.float16:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            generated_video = model.generate_video(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                num_frames=16,
                            )
                    else:
                        generated_video = model.generate_video(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            num_frames=16,
                        )
                
                print(f"   Generated shape: {generated_video.shape}")
                
                # Save video frames as GIF
                try:
                    from PIL import Image
                    import numpy as np
                    
                    # Convert tensor to frames (assuming BCTHW or BTCHW format)
                    vid_tensor = generated_video[0].cpu().float()
                    
                    # Handle different tensor formats
                    if vid_tensor.dim() == 4:
                        if vid_tensor.shape[0] == 3:  # CTHW
                            vid_tensor = vid_tensor.permute(1, 0, 2, 3)  # TCHW
                        # Now TCHW
                        frames = []
                        for t in range(vid_tensor.shape[0]):
                            frame = vid_tensor[t].clamp(0, 1)
                            frame_array = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                            frames.append(Image.fromarray(frame_array))
                        
                        if frames:
                            gif_file = os.path.join(output_dir, f"video_{timestamp}_{i+1}.gif")
                            frames[0].save(
                                gif_file,
                                save_all=True,
                                append_images=frames[1:],
                                duration=100,
                                loop=0
                            )
                            print(f"   ✅ Saved to: {gif_file}")
                except Exception as save_e:
                    print(f"   ⚠️ Could not save video: {save_e}")
                    
            except Exception as e:
                print(f"   ⚠️ Video generation failed: {e}")
    else:
        print("\n   ⚠️ Video generator not available")
    
    # ========== TEXT-TO-SPEECH ==========
    print("\n" + "-" * 50)
    print("🔊 TEXT-TO-SPEECH TEST")
    print("-" * 50)
    
    tts_prompts = [
        "Hello, this is a test of the text to speech system.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    
    for i, prompt in enumerate(tts_prompts):
        print(f"\n   Prompt {i+1}: {prompt}")
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # Use autocast for mixed precision to handle dtype mismatches
            with torch.no_grad():
                if 'cuda' in device and model_dtype == torch.float16:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        mel, durations = model.generate_speech(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                else:
                    mel, durations = model.generate_speech(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
            
            print(f"   Generated mel-spectrogram shape: {mel.shape}")
            
            # Save mel spectrogram as numpy file
            try:
                import numpy as np
                mel_file = os.path.join(output_dir, f"speech_{timestamp}_{i+1}_mel.npy")
                np.save(mel_file, mel.cpu().float().numpy())
                print(f"   ✅ Saved mel to: {mel_file}")
            except Exception as save_e:
                print(f"   ⚠️ Could not save mel: {save_e}")
                
        except Exception as e:
            print(f"   ⚠️ Text-to-speech failed: {e}")
    
    print("\n" + "=" * 70)
    print(f"✅ GENERATION TESTS COMPLETE")
    print(f"   Outputs saved to: {output_dir}")
    print("=" * 70)


def interactive_mode():
    """Run interactive mode to load and test the model."""
    print("\n" + "=" * 70)
    print("🤖 XORON-DEV MODEL LOADER - INTERACTIVE MODE")
    print("=" * 70)
    
    # Choose load source
    print("\nSelect model source:")
    print("  1. Load from HuggingFace (Backup-bdg/Xoron-Dev-MultiMoe)")
    print("  2. Load from local directory")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    model = None
    device = None
    device_map = None
    
    if choice == "1":
        # Load from HuggingFace
        model_name = input("Enter HuggingFace model ID [Backup-bdg/Xoron-Dev-MultiMoe]: ").strip()
        if not model_name:
            model_name = "Backup-bdg/Xoron-Dev-MultiMoe"
        
        # Device selection
        print("\nSelect device:")
        print("  1. CUDA (GPU)")
        print("  2. CPU")
        device_choice = input("Enter choice (1 or 2): ").strip()
        
        if device_choice == "1":
            if torch.cuda.is_available():
                device = "cuda:0"
                num_gpus = torch.cuda.device_count()
                
                if num_gpus > 1:
                    print(f"\nDetected {num_gpus} GPUs")
                    use_parallel = input("Use model parallelism? (y/n): ").strip().lower()
                    if use_parallel == "y":
                        device_map = setup_device_map(num_gpus)
            else:
                print("⚠️  CUDA not available, falling back to CPU")
                device = "cpu"
        else:
            device = "cpu"
        
        try:
            model = load_from_huggingface(
                model_name=model_name,
                device=device,
                device_map=device_map,
                use_fp16=True,
            )
        except Exception as e:
            print(f"\n❌ Failed to load model: {e}")
            return
    
    elif choice == "2":
        # Load from local directory
        path = input("Enter local model path: ").strip()
        
        # Device selection
        print("\nSelect device:")
        print("  1. CUDA (GPU)")
        print("  2. CPU")
        device_choice = input("Enter choice (1 or 2): ").strip()
        
        if device_choice == "1":
            if torch.cuda.is_available():
                device = "cuda:0"
                num_gpus = torch.cuda.device_count()
                
                if num_gpus > 1:
                    print(f"\nDetected {num_gpus} GPUs")
                    use_parallel = input("Use model parallelism? (y/n): ").strip().lower()
                    if use_parallel == "y":
                        device_map = setup_device_map(num_gpus)
            else:
                print("⚠️  CUDA not available, falling back to CPU")
                device = "cpu"
        else:
            device = "cpu"
        
        try:
            model = load_from_local(
                path=path,
                device=device,
                device_map=device_map,
                use_fp16=True,
            )
        except Exception as e:
            print(f"\n❌ Failed to load model: {e}")
            return
    
    else:
        print("❌ Invalid choice")
        return
    
    # Automatically run generation tests after loading
    if model is not None:
        run_generation_tests(model, device or "cpu")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Load Xoron-Dev multimodal model from HuggingFace or local storage and run generation tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python load.py
  
  # Load from HuggingFace and run generation tests
  python load.py --from-hf
  
  # Load from local directory and run generation tests
  python load.py --from-local ./xoron-model
  
  # Load on specific GPU
  python load.py --from-hf --device cuda:0
  
  # Load with model parallelism (multi-GPU)
  python load.py --from-hf --device cuda:0 --num-gpus 2
  
  # Specify output directory for generated files
  python load.py --from-hf --output-dir ./my_outputs
        """
    )
    
    parser.add_argument(
        "--from-hf",
        action="store_true",
        help="Load model from HuggingFace (Backup-bdg/Xoron-Dev-MultiMoe)"
    )
    
    parser.add_argument(
        "--from-local",
        type=str,
        help="Load model from local directory"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="Backup-bdg/Xoron-Dev-MultiMoe",
        help="HuggingFace model identifier (default: Backup-bdg/Xoron-Dev-MultiMoe)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        help="Device to load model to (e.g., cuda:0, cpu)"
    )
    
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for model parallelism (default: 1)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./generated_outputs",
        help="Directory to save generated outputs (default: ./generated_outputs)"
    )
    
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 (use full precision)"
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, run interactive mode
    if not (args.from_hf or args.from_local):
        interactive_mode()
        return
    
    # Determine device
    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    
    # Setup device map for multi-GPU
    device_map = None
    if args.num_gpus > 1 and 'cuda' in device:
        device_map = setup_device_map(args.num_gpus)
    
    # Load model
    model = None
    if args.from_hf:
        model = load_from_huggingface(
            model_name=args.model_name,
            device=device,
            device_map=device_map,
            use_fp16=not args.no_fp16,
        )
    elif args.from_local:
        model = load_from_local(
            path=args.from_local,
            device=device,
            device_map=device_map,
            use_fp16=not args.no_fp16,
        )
    
    # Automatically run generation tests after loading
    if model is not None:
        run_generation_tests(model, device, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

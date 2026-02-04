import os
import json
import torch
from typing import Optional, Dict
from transformers import AutoTokenizer
from config import XoronConfig
from models.xoron import XoronMultimodalModel # Keep the class, we just use text parts

def load_llm_from_hf(
    model_name: str = "Backup-bdg/Xoron-Dev-MultiMoe",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_fp16: bool = True,
) -> XoronMultimodalModel:
    """Loads only the LLM portion of Xoron-Dev from HuggingFace."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_model
    
    print(f"📥 Downloading LLM weights: {model_name}")
    path = snapshot_download(repo_id=model_name)
    
    # 1. Load and Fix Config
    with open(os.path.join(path, "config.json"), 'r') as f:
        config_dict = json.load(f)
    
    # Force disable multimodal generators for a pure LLM load
    config_dict['has_audio_encoder'] = False
    config_dict['has_video_generator'] = False
    config_dict['has_image_generator'] = False
    
    # 2. Detect Vocab Size from Safetensors
    model_ts_path = os.path.join(path, "model.safetensors")
    from safetensors import safe_open
    with safe_open(model_ts_path, framework="pt") as f:
        for key in f.keys():
            if 'embed_tokens.weight' in key:
                checkpoint_vocab_size = f.get_tensor(key).shape[0]
                if checkpoint_vocab_size != config_dict.get('vocab_size'):
                    print(f"🔄 Patching vocab: {checkpoint_vocab_size}")
                    config_dict['vocab_size'] = checkpoint_vocab_size
                break

    # 3. Initialize Model & Apply LoRA if checkpoint requires it
    config = XoronConfig.from_dict(config_dict)
    model = XoronMultimodalModel(config)
    
    # Detect LoRA structure in keys to apply before loading weights
    with safe_open(model_ts_path, framework="pt") as f:
        if any('.lora_A' in k for k in f.keys()):
            print("🔧 Applying LoRA layers for compatibility...")
            model.apply_lora()

    # 4. Load Weights
    print("📦 Loading weights into LLM...")
    load_model(model, model_ts_path, strict=False)
    
    model = model.to(device)
    if use_fp16 and "cuda" in device:
        model = model.half()
    
    model.eval()
    return model, path

def generate_text(model, tokenizer, prompt, device, max_tokens=128):
    """Simple greedy generation for the LLM."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Using standard HF-style generation if Xoron supports it, 
        # otherwise use the model's forward pass
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7
        )
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Model
    llm, model_path = load_llm_from_hf()
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm.config.tokenizer_name, trust_remote_code=True)
    
    # Test Inference
    prompt = "Explain why Mixture of Experts is efficient:"
    print(f"\nPrompt: {prompt}")
    response = generate_text(llm, tokenizer, prompt, device)
    print(f"\nResponse: {response}")

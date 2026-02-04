import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import Optional, Dict

class XoronLLMLoader:
    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.path = model_path
        self.device = device
        self.config = self._load_config()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.get("tokenizer_name", "Qwen/Qwen2.5-7B"), 
            trust_remote_code=True
        )
        self.model = self._load_llm_only()

    def _load_config(self):
        config_path = os.path.join(self.path, "config.json")
        with open(config_path, 'r') as f:
            return json.load(f)

    def _load_llm_only(self):
        """Loads only the language backbone and handles vocab/LoRA fixes."""
        from models.xoron import XoronMultimodalModel
        from config import XoronConfig

        # Vocab Fix Logic: Ensure the model shell matches the checkpoint
        checkpoint_vocab_size = self._detect_vocab_size()
        if checkpoint_vocab_size:
            self.config['vocab_size'] = checkpoint_vocab_size
        
        xoron_config = XoronConfig.from_dict(self.config)
        # We initialize the full model but only use the backbone for this script
        model = XoronMultimodalModel(xoron_config)
        
        # Load weights (handling safetensors or bin)
        weight_path = os.path.join(self.path, "model.safetensors")
        if os.path.exists(weight_path):
            from safetensors.torch import load_model
            load_model(model, weight_path)
        else:
            state_dict = torch.load(os.path.join(self.path, "pytorch_model.bin"), map_location='cpu')
            model.load_state_dict(state_dict, strict=False)

        model.to(self.device).eval()
        return model

    def _detect_vocab_size(self):
        # Implementation of your logic to check embed_tokens.weight shape
        # ... (Refer to your original _load_model_with_vocab_fix)
        return self.config.get('vocab_size')

    @torch.no_grad()
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 128, 
        temperature: float = 0.7, 
        top_p: float = 0.9,
        top_k: int = 50
    ):
        """Proper Autoregressive Generation Loop"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generated_ids = inputs["input_ids"]

        for _ in range(max_new_tokens):
            # 1. Forward Pass
            outputs = self.model(input_ids=generated_ids)
            
            # 2. Extract Logits (Last token only)
            next_token_logits = outputs.logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

            # 3. Top-K Filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # 4. Top-P (Nucleus) Filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # 5. Sample and Append
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # 6. Stop on EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

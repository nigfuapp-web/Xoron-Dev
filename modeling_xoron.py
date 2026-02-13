"""
Xoron Model for HuggingFace Transformers.

This module provides a HuggingFace-compatible model class for the Xoron
multimodal model. It inherits from PreTrainedModel to enable:
- Loading via AutoModel
- Saving/loading with save_pretrained/from_pretrained  
- Hub integration with push_to_hub
- trust_remote_code support

Usage:
    from transformers import AutoModel
    model = AutoModel.from_pretrained("your-repo/xoron-model", trust_remote_code=True)
"""

import os
import sys
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union, Tuple
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

# Import configuration - handle both package and standalone imports
try:
    from .configuration_xoron import XoronConfig
except ImportError:
    from configuration_xoron import XoronConfig

logger = logging.getLogger(__name__)

# FP16 safe max value
MAX_HIDDEN = 10000.0


def safe_clamp_tensor(x: torch.Tensor, max_val: float = MAX_HIDDEN) -> torch.Tensor:
    """Clamp tensor values for FP16 safety."""
    if x is None or x.numel() == 0:
        return x
    x = torch.nan_to_num(x, nan=0.0, posinf=max_val, neginf=-max_val)
    return x.clamp(-max_val, max_val)


class XoronPreTrainedModel(PreTrainedModel):
    """
    Base class for Xoron models providing HuggingFace integration.
    
    This is the base class that provides weight initialization and
    a simple interface for loading pretrained models.
    """
    
    config_class = XoronConfig
    base_model_prefix = "xoron"
    supports_gradient_checkpointing = True
    _no_split_modules = ["XoronMultimodalModel"]
    _skip_keys_device_placement = "past_key_values"
    
    def _init_weights(self, module):
        """Initialize the weights."""
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class XoronModel(XoronPreTrainedModel):
    """
    Xoron Multimodal Model for HuggingFace.
    
    This is a wrapper around the internal XoronMultimodalModel that provides
    HuggingFace compatibility for loading via AutoModel with trust_remote_code=True.
    
    The model supports:
    - Image/video understanding (SigLIP encoder)
    - Text generation (MoE LLM)
    - Image/video generation (MobileDiffusion)
    - Voice understanding and generation (ASR/TTS)
    - Cross-attention for multimodal fusion
    - LoRA support for efficient fine-tuning
    
    Example:
        >>> from transformers import AutoModel, AutoConfig
        >>> config = AutoConfig.from_pretrained("your-repo/xoron", trust_remote_code=True)
        >>> model = AutoModel.from_pretrained("your-repo/xoron", trust_remote_code=True)
        >>> # Forward pass
        >>> outputs = model(input_ids=input_ids, pixel_values=images)
    """
    
    def __init__(self, config: XoronConfig):
        super().__init__(config)
        self.config = config
        
        # Import the internal model - this handles all the actual implementation
        # We use lazy import to avoid circular dependencies
        self._internal_model = None
        self._internal_config = None
        
    def _ensure_internal_model(self):
        """Lazily initialize the internal model."""
        if self._internal_model is None:
            # Convert HF config to internal config
            # Try importing from the Xoron-Dev package (if installed)
            # or from the local directory structure
            try:
                from config.model_config import XoronConfig as InternalConfig
            except ImportError:
                try:
                    # Try alternative import path for when running from HuggingFace Hub
                    import importlib.util
                    import sys
                    
                    # Get the directory containing this file
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    
                    # Add to path if not already there
                    if current_dir not in sys.path:
                        sys.path.insert(0, current_dir)
                    
                    # Try importing again
                    from config.model_config import XoronConfig as InternalConfig
                except ImportError:
                    raise ImportError(
                        "Could not import XoronConfig from config.model_config. "
                        "Please install the Xoron-Dev package first:\n"
                        "  pip install git+https://github.com/nigfuapp-web/Xoron-Dev.git@beta\n"
                        "Or clone the repository and install locally:\n"
                        "  git clone -b beta https://github.com/nigfuapp-web/Xoron-Dev.git\n"
                        "  cd Xoron-Dev && pip install -e ."
                    )
            
            # Create internal config from HF config
            config_dict = {k: v for k, v in self.config.to_dict().items() 
                         if not k.startswith('_') and k not in ['transformers_version', 'model_type', 'torch_dtype', 'auto_map']}
            
            # Handle tuple conversions
            if 'lora_target_modules' in config_dict and isinstance(config_dict['lora_target_modules'], list):
                config_dict['lora_target_modules'] = tuple(config_dict['lora_target_modules'])
            if 'generation_supported_sizes' in config_dict and isinstance(config_dict['generation_supported_sizes'], list):
                config_dict['generation_supported_sizes'] = tuple(config_dict['generation_supported_sizes'])
            if 'generation_supported_frames' in config_dict and isinstance(config_dict['generation_supported_frames'], list):
                config_dict['generation_supported_frames'] = tuple(config_dict['generation_supported_frames'])
            
            self._internal_config = InternalConfig.from_dict(config_dict)
            
            # Import and create internal model
            try:
                from models.xoron import XoronMultimodalModel
            except ImportError:
                raise ImportError(
                    "Could not import XoronMultimodalModel from models.xoron. "
                    "Please install the Xoron-Dev package first:\n"
                    "  pip install git+https://github.com/nigfuapp-web/Xoron-Dev.git@beta\n"
                    "Or clone the repository and install locally:\n"
                    "  git clone -b beta https://github.com/nigfuapp-web/Xoron-Dev.git\n"
                    "  cd Xoron-Dev && pip install -e ."
                )
            
            self._internal_model = XoronMultimodalModel(self._internal_config)
            
    @property
    def internal_model(self):
        """Get the internal XoronMultimodalModel."""
        self._ensure_internal_model()
        return self._internal_model
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        video_frames: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass for the Xoron multimodal model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            pixel_values: Image inputs of shape (batch_size, channels, height, width)
            video_frames: Video inputs of shape (batch_size, num_frames, channels, height, width)
            audio_features: Audio inputs (mel spectrogram or raw waveform)
            labels: Labels for language modeling loss
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a ModelOutput object
            
        Returns:
            CausalLMOutputWithPast containing loss, logits, and optionally hidden states
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Ensure internal model is initialized
        self._ensure_internal_model()
        
        # Call internal model forward
        outputs = self._internal_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            video_frames=video_frames,
            audio_features=audio_features,
            labels=labels,
        )
        
        if not return_dict:
            return (outputs.get('loss'), outputs.get('logits'), outputs.get('hidden_states'))
        
        return CausalLMOutputWithPast(
            loss=outputs.get('loss'),
            logits=outputs.get('logits'),
            past_key_values=None,
            hidden_states=outputs.get('hidden_states') if output_hidden_states else None,
            attentions=None,
        )
    
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        video_frames: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text given inputs.
        
        Args:
            input_ids: Input token IDs
            pixel_values: Image inputs
            video_frames: Video inputs
            audio_features: Audio inputs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Generated token IDs
        """
        self._ensure_internal_model()
        
        # Use internal model's generate method if available
        if hasattr(self._internal_model, 'generate'):
            return self._internal_model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                video_frames=video_frames,
                audio_features=audio_features,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                **kwargs,
            )
        
        # Fallback to basic autoregressive generation
        return self._basic_generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            video_frames=video_frames,
            audio_features=audio_features,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
        )
    
    def _basic_generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        video_frames: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Basic autoregressive generation."""
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            outputs = self.forward(
                input_ids=generated,
                pixel_values=pixel_values if generated.shape[1] == input_ids.shape[1] else None,
                video_frames=video_frames if generated.shape[1] == input_ids.shape[1] else None,
                audio_features=audio_features if generated.shape[1] == input_ids.shape[1] else None,
            )
            
            logits = outputs.logits[:, -1, :]
            
            if do_sample:
                # Apply temperature
                logits = logits / temperature
                
                # Apply top-k
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus sampling)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS token
            if hasattr(self.config, 'eos_token_id') and self.config.eos_token_id is not None:
                if (next_token == self.config.eos_token_id).all():
                    break
        
        return generated
    
    def generate_image(
        self,
        prompt_embeds: torch.Tensor,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        **kwargs,
    ) -> torch.Tensor:
        """Generate image from text embeddings."""
        self._ensure_internal_model()
        if hasattr(self._internal_model, 'generate_image'):
            return self._internal_model.generate_image(
                prompt_embeds=prompt_embeds,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                **kwargs,
            )
        raise NotImplementedError("Image generation not available")
    
    def generate_video(
        self,
        prompt_embeds: torch.Tensor,
        num_frames: int = 16,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        **kwargs,
    ) -> torch.Tensor:
        """Generate video from text embeddings."""
        self._ensure_internal_model()
        if hasattr(self._internal_model, 'generate_video'):
            return self._internal_model.generate_video(
                prompt_embeds=prompt_embeds,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                **kwargs,
            )
        raise NotImplementedError("Video generation not available")
    
    def generate_audio(
        self,
        text_embeds: torch.Tensor,
        speaker_embedding: Optional[torch.Tensor] = None,
        max_length: int = 1000,
        **kwargs,
    ) -> torch.Tensor:
        """Generate audio from text embeddings (TTS)."""
        self._ensure_internal_model()
        if hasattr(self._internal_model, 'generate_audio'):
            return self._internal_model.generate_audio(
                text_embeds=text_embeds,
                speaker_embedding=speaker_embedding,
                max_length=max_length,
                **kwargs,
            )
        raise NotImplementedError("Audio generation not available")
    
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode image to embeddings."""
        self._ensure_internal_model()
        return self._internal_model.encode_image(pixel_values)
    
    def encode_video(self, video_frames: torch.Tensor) -> torch.Tensor:
        """Encode video to embeddings."""
        self._ensure_internal_model()
        return self._internal_model.encode_video(video_frames)
    
    def encode_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Encode audio to embeddings."""
        self._ensure_internal_model()
        return self._internal_model.encode_audio(audio_features)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load pretrained model from HuggingFace Hub or local path.
        
        This method handles loading the model weights from component files
        created by the save_pretrained method.
        """
        # First load config and create model shell
        config = kwargs.pop('config', None)
        if config is None:
            config = XoronConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        model = cls(config)
        
        # Now load the actual weights
        model._ensure_internal_model()
        
        # Check if this is a local path or HF hub
        if os.path.isdir(pretrained_model_name_or_path):
            model_path = pretrained_model_name_or_path
        else:
            # Download from HuggingFace Hub
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(
                repo_id=pretrained_model_name_or_path,
                allow_patterns=["*.safetensors", "*.json", "*.py"],
            )
        
        # Load weights into internal model
        model._internal_model.load_pretrained(model_path)
        
        return model
    
    def save_pretrained(
        self,
        save_directory: str,
        is_main_process: bool = True,
        state_dict: Optional[Dict] = None,
        save_function = None,
        push_to_hub: bool = False,
        max_shard_size: str = "2GB",
        safe_serialization: bool = True,
        **kwargs,
    ):
        """
        Save model to directory in HuggingFace format.
        
        This saves both the model weights and dynamically builds a self-contained
        modeling_xoron.py that can be loaded with trust_remote_code=True without
        requiring the full Xoron-Dev package.
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        self.config.save_pretrained(save_directory)
        
        # Save internal model weights
        if self._internal_model is not None:
            self._internal_model.save_pretrained(save_directory)
        
        # Copy configuration file
        import shutil
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        config_src = os.path.join(current_dir, 'configuration_xoron.py')
        config_dst = os.path.join(save_directory, 'configuration_xoron.py')
        if os.path.exists(config_src):
            shutil.copy2(config_src, config_dst)
        
        # Build self-contained modeling_xoron.py dynamically
        modeling_dst = os.path.join(save_directory, 'modeling_xoron.py')
        self._build_self_contained_modeling_file(current_dir, modeling_dst)
        
        # Update config.json with auto_map for trust_remote_code
        config_path = os.path.join(save_directory, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            config_dict['auto_map'] = {
                'AutoConfig': 'configuration_xoron.XoronConfig',
                'AutoModel': 'modeling_xoron.XoronModel',
            }
            config_dict['model_type'] = 'xoron'
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        
        if push_to_hub:
            self.push_to_hub(save_directory, **kwargs)
    
    def _build_self_contained_modeling_file(self, repo_root: str, output_path: str):
        """
        Build a self-contained modeling_xoron.py by combining all model components.
        
        This allows HuggingFace users to load the model with trust_remote_code=True
        without needing to install the full Xoron-Dev package.
        """
        import re
        
        # Component files to combine (in dependency order)
        component_files = [
            "models/components/lora.py",
            "models/components/attention.py",
            "models/components/projectors.py",
            "models/components/moe.py",
            "models/encoders/vision.py",
            "models/encoders/video.py",
            "models/encoders/audio.py",
            "models/generators/image.py",
            "models/generators/video.py",
            "models/llm/moe_llama.py",
            "models/xoron.py",
        ]
        
        # Internal imports to remove
        internal_import_patterns = [
            r"^from config import.*$",
            r"^from config\..*import.*$",
            r"^from models\..*import.*$",
            r"^from models import.*$",
        ]
        
        def is_internal_import(line):
            line = line.strip()
            for pattern in internal_import_patterns:
                if re.match(pattern, line):
                    return True
            return False
        
        def is_external_import(line):
            line = line.strip()
            return (line.startswith("import ") or 
                    (line.startswith("from ") and not is_internal_import(line)))
        
        def extract_code_body(content):
            """Extract code body, removing module docstring and imports."""
            lines = content.split('\n')
            code_lines = []
            i = 0
            
            # Skip leading whitespace
            while i < len(lines) and not lines[i].strip():
                i += 1
            
            # Skip module docstring if present
            if i < len(lines):
                stripped = lines[i].strip()
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    docstring_char = stripped[:3]
                    if stripped.count(docstring_char) >= 2:
                        i += 1
                    else:
                        i += 1
                        while i < len(lines):
                            if docstring_char in lines[i]:
                                i += 1
                                break
                            i += 1
            
            # Process remaining lines
            for line in lines[i:]:
                stripped = line.strip()
                
                # Skip empty lines at start
                if not code_lines and not stripped:
                    continue
                
                # Skip internal imports
                if is_internal_import(line):
                    continue
                
                # Skip external imports (we'll add our own)
                if is_external_import(line):
                    continue
                
                # Skip logger setup lines
                if stripped.startswith("logger = logging.getLogger"):
                    continue
                
                code_lines.append(line)
            
            # Remove trailing empty lines
            while code_lines and not code_lines[-1].strip():
                code_lines.pop()
            
            return '\n'.join(code_lines)
        
        # Build the file
        header = '''"""
Xoron Model for HuggingFace Transformers - Self-Contained Implementation.

AUTO-GENERATED FILE - Do not edit directly!

This module provides a complete, self-contained HuggingFace-compatible model class
for the Xoron multimodal model. All components are embedded directly in this file
to enable loading via AutoModel with trust_remote_code=True WITHOUT requiring
the full Xoron-Dev package to be installed.

Usage:
    from transformers import AutoModel, AutoConfig
    config = AutoConfig.from_pretrained("your-repo/xoron-model", trust_remote_code=True)
    model = AutoModel.from_pretrained("your-repo/xoron-model", trust_remote_code=True)
"""

import os
import math
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from safetensors.torch import save_model, load_model
except ImportError:
    save_model, load_model = None, None

from transformers import PreTrainedModel, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

try:
    from transformers.models.llama.modeling_llama import (
        LlamaAttention, LlamaDecoderLayer, LlamaRMSNorm, LlamaMLP,
        LlamaRotaryEmbedding, apply_rotary_pos_emb, repeat_kv
    )
except ImportError:
    LlamaAttention = LlamaDecoderLayer = LlamaRMSNorm = LlamaMLP = None
    LlamaRotaryEmbedding = apply_rotary_pos_emb = repeat_kv = None

# Import configuration
try:
    from .configuration_xoron import XoronConfig
except ImportError:
    from configuration_xoron import XoronConfig

logger = logging.getLogger(__name__)

'''
        
        all_code = [header]
        
        # Process each component file
        for filepath in component_files:
            full_path = os.path.join(repo_root, filepath)
            if not os.path.exists(full_path):
                continue
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            code = extract_code_body(content)
            if code.strip():
                section_name = filepath.replace('/', '.').replace('.py', '').upper()
                section_header = f"\n\n# {'='*78}\n# {section_name}\n# {'='*78}\n\n"
                all_code.append(section_header + code)
        
        # Add HuggingFace wrapper classes
        hf_wrapper = '''

# ==============================================================================
# HUGGINGFACE WRAPPER CLASSES
# ==============================================================================

class XoronPreTrainedModel(PreTrainedModel):
    """Base class for Xoron models providing HuggingFace integration."""
    
    config_class = XoronConfig
    base_model_prefix = "xoron"
    supports_gradient_checkpointing = True
    _no_split_modules = ["XoronMultimodalModel"]
    _skip_keys_device_placement = "past_key_values"
    
    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class XoronModel(XoronPreTrainedModel):
    """Xoron Multimodal Model for HuggingFace."""
    
    def __init__(self, config: XoronConfig):
        super().__init__(config)
        self.config = config
        self._internal_model = XoronMultimodalModel(config)
        self.post_init()
    
    @property
    def internal_model(self):
        return self._internal_model
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        video_frames: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self._internal_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=pixel_values,
            video=video_frames,
            audio=audio_features,
            labels=labels,
        )
        
        if return_dict:
            return CausalLMOutputWithPast(
                loss=outputs.get('loss'),
                logits=outputs.get('logits'),
                past_key_values=outputs.get('past_key_values'),
                hidden_states=outputs.get('hidden_states'),
                attentions=outputs.get('attentions'),
            )
        
        return (outputs.get('loss'), outputs.get('logits'))
    
    def generate_image(self, prompt_embeds: torch.Tensor, **kwargs):
        return self._internal_model.generate_image(prompt_embeds, **kwargs)
    
    def generate_video(self, prompt_embeds: torch.Tensor, **kwargs):
        return self._internal_model.generate_video(prompt_embeds, **kwargs)
    
    def generate_speech(self, text_embeds: torch.Tensor, **kwargs):
        return self._internal_model.generate_speech(text_embeds, **kwargs)


class XoronForCausalLM(XoronModel):
    """Alias for XoronModel for compatibility."""
    pass


# Register for AutoClass
XoronConfig.register_for_auto_class()
XoronModel.register_for_auto_class("AutoModel")
XoronForCausalLM.register_for_auto_class("AutoModelForCausalLM")
'''
        all_code.append(hf_wrapper)
        
        # Write output file
        final_content = '\n'.join(all_code)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        line_count = final_content.count('\n')
        print(f"   📦 Built self-contained modeling_xoron.py ({line_count:,} lines)")


class XoronForCausalLM(XoronModel):
    """
    Xoron model with a causal language modeling head.
    
    This is an alias for XoronModel that provides compatibility
    with AutoModelForCausalLM.
    """
    pass


# Register for AutoClass - these will be called when the model is loaded
# with trust_remote_code=True
XoronConfig.register_for_auto_class()
XoronModel.register_for_auto_class("AutoModel")
XoronForCausalLM.register_for_auto_class("AutoModelForCausalLM")

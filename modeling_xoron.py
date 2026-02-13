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
        
        This saves both the model weights and the custom code files
        needed for trust_remote_code loading.
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        self.config.save_pretrained(save_directory)
        
        # Save internal model weights
        if self._internal_model is not None:
            self._internal_model.save_pretrained(save_directory)
        
        # Copy custom code files for trust_remote_code
        import shutil
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Files to copy
        files_to_copy = [
            'configuration_xoron.py',
            'modeling_xoron.py',
        ]
        
        for filename in files_to_copy:
            src = os.path.join(current_dir, filename)
            dst = os.path.join(save_directory, filename)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
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

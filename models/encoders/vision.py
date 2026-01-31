"""
Vision encoder supporting multiple backends.

Supports:
- SigLIP 2 (default, recommended for MoE architectures)
- CLIP (legacy support)

SigLIP 2 advantages:
- Better zero-shot performance
- Sigmoid loss instead of softmax (better for multi-label)
- More efficient training
- Better suited for MoE architectures
"""

import torch
import torch.nn as nn
from typing import Optional


class VisionEncoder(nn.Module):
    """
    Vision encoder supporting SigLIP 2 and CLIP backends.
    
    SigLIP 2 is recommended for MoE architectures due to:
    - Better feature quality for downstream tasks
    - More efficient sigmoid-based contrastive learning
    - Improved zero-shot and few-shot performance
    """

    def __init__(
        self, 
        model_name: str = "google/siglip-so400m-patch14-384",
        freeze: bool = False,
        use_pooled_output: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.use_pooled_output = use_pooled_output
        self._is_siglip = "siglip" in model_name.lower()

        print(f"\n👁️ Loading Vision Encoder: {model_name}")
        
        if self._is_siglip:
            self._init_siglip(model_name, freeze)
        else:
            self._init_clip(model_name, freeze)

    def _init_siglip(self, model_name: str, freeze: bool):
        """Initialize SigLIP 2 vision encoder."""
        try:
            from transformers import SiglipVisionModel, SiglipImageProcessor
            
            self.vision_model = SiglipVisionModel.from_pretrained(model_name)
            self.image_processor = SiglipImageProcessor.from_pretrained(model_name)
            self.hidden_size = self.vision_model.config.hidden_size
            
            print(f"   🎯 Using SigLIP 2 (recommended for MoE)")
            print(f"   ✅ Hidden size: {self.hidden_size}")
            print(f"   📐 Image size: {self.vision_model.config.image_size}")
            print(f"   🔲 Patch size: {self.vision_model.config.patch_size}")
            
        except ImportError:
            print("   ⚠️ SigLIP not available, falling back to CLIP")
            self._is_siglip = False
            self._init_clip("openai/clip-vit-large-patch14", freeze)
            return
        
        if freeze:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            print(f"   ❄️ Vision encoder frozen")
        else:
            print(f"   🔥 Vision encoder trainable")

    def _init_clip(self, model_name: str, freeze: bool):
        """Initialize CLIP vision encoder (legacy support)."""
        from transformers import CLIPVisionModel, CLIPImageProcessor
        
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        self.hidden_size = self.vision_model.config.hidden_size
        
        print(f"   📎 Using CLIP")
        print(f"   ✅ Hidden size: {self.hidden_size}")

        if freeze:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            print(f"   ❄️ Vision encoder frozen")
        else:
            print(f"   🔥 Vision encoder trainable")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract vision features from images.
        
        Args:
            pixel_values: [B, C, H, W] tensor of images
            
        Returns:
            [B, num_patches, hidden_size] tensor of patch features
            or [B, hidden_size] if use_pooled_output=True
        """
        outputs = self.vision_model(pixel_values=pixel_values)
        
        if self.use_pooled_output:
            # Return pooled output (CLS token or mean pooling)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                return outputs.pooler_output
            else:
                # Mean pooling over patches
                return outputs.last_hidden_state.mean(dim=1)
        
        return outputs.last_hidden_state

    def get_image_processor(self):
        """Return the image processor for preprocessing."""
        return self.image_processor
    
    @property
    def num_patches(self) -> int:
        """Get number of patches for the vision model."""
        config = self.vision_model.config
        image_size = config.image_size
        patch_size = config.patch_size
        return (image_size // patch_size) ** 2
    
    @property 
    def image_size(self) -> int:
        """Get expected image size."""
        return self.vision_model.config.image_size


# Convenience aliases for different SigLIP 2 variants
SIGLIP_MODELS = {
    # Base models
    "siglip-base": "google/siglip-base-patch16-224",
    "siglip-base-384": "google/siglip-base-patch16-384",
    
    # Large models (recommended)
    "siglip-large": "google/siglip-large-patch16-256",
    "siglip-large-384": "google/siglip-large-patch16-384",
    
    # SO400M models (best quality)
    "siglip-so400m": "google/siglip-so400m-patch14-384",
    "siglip-so400m-224": "google/siglip-so400m-patch14-224",
    
    # Legacy CLIP
    "clip-base": "openai/clip-vit-base-patch16",
    "clip-large": "openai/clip-vit-large-patch14",
}


def get_vision_encoder(
    model_key: str = "siglip-so400m",
    freeze: bool = False,
    **kwargs
) -> VisionEncoder:
    """
    Get a vision encoder by key name.
    
    Args:
        model_key: Key from SIGLIP_MODELS or full model name
        freeze: Whether to freeze encoder weights
        **kwargs: Additional arguments for VisionEncoder
        
    Returns:
        VisionEncoder instance
    """
    model_name = SIGLIP_MODELS.get(model_key, model_key)
    return VisionEncoder(model_name=model_name, freeze=freeze, **kwargs)

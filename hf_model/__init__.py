"""
Xoron-Dev HuggingFace Model Package.

This package provides HuggingFace-compatible model and configuration classes
for the Xoron-Dev multimodal MoE model.

Usage:
    from transformers import AutoConfig, AutoModelForCausalLM
    
    # Load configuration
    config = AutoConfig.from_pretrained("your-username/Xoron-Dev", trust_remote_code=True)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained("your-username/Xoron-Dev", trust_remote_code=True)
"""

from .configuration_xoron import XoronConfig
from .modeling_xoron import (
    XoronPreTrainedModel,
    XoronModel,
    XoronForCausalLM,
    XoronRMSNorm,
    XoronMoE,
    XoronAttention,
    XoronDecoderLayer,
)

__all__ = [
    "XoronConfig",
    "XoronPreTrainedModel",
    "XoronModel",
    "XoronForCausalLM",
    "XoronRMSNorm",
    "XoronMoE",
    "XoronAttention",
    "XoronDecoderLayer",
]

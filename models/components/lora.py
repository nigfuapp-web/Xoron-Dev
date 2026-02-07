"""
SOTA LoRA (Low-Rank Adaptation) implementation for efficient fine-tuning.

Features:
- Standard LoRA with improved initialization
- DoRA (Weight-Decomposed LoRA) for better performance
- LoRA+ with different learning rates for A and B matrices
- Rank-stabilized LoRA (rsLoRA) scaling
- Memory-efficient: shares base weights instead of cloning (CRITICAL for multi-GPU training)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple


class LoRALinear(nn.Module):
    """
    SOTA LoRA layer with multiple variants.
    
    Supports:
    - Standard LoRA
    - DoRA (Weight-Decomposed LoRA)
    - rsLoRA (rank-stabilized scaling)
    
    MEMORY OPTIMIZATION:
    - Does NOT clone base weights - shares them with original module
    - Only LoRA params (A, B, magnitude) consume additional memory
    - Base weights are frozen and can be kept in lower precision
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        merge_weights: bool = False,
        use_dora: bool = False,
        use_rslora: bool = True,
        base_layer: nn.Linear = None,  # Pass existing layer to share weights
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.merge_weights = merge_weights
        self.merged = False
        self.use_dora = use_dora
        self.use_rslora = use_rslora
        self.in_features = in_features
        self.out_features = out_features

        # MEMORY OPTIMIZATION: Share base layer instead of creating new one
        # This is CRITICAL - cloning weights doubles memory usage!
        if base_layer is not None:
            # Use the existing layer directly (no memory duplication!)
            self.linear = base_layer
        else:
            # Only create new layer if no base provided (for standalone use)
            self.linear = nn.Linear(in_features, out_features, bias=False)

        # LoRA low-rank matrices - these are the ONLY new parameters
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            
            # Scaling factor
            if use_rslora:
                # rsLoRA: scale by sqrt(r) for rank-stabilized training
                self.scaling = lora_alpha / math.sqrt(r)
            else:
                self.scaling = lora_alpha / r
                
            self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
            
            # Initialize A with Kaiming, B with zeros (standard)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            
            # DoRA: learnable magnitude vector
            if use_dora:
                self.magnitude = nn.Parameter(torch.ones(out_features))

        # Freeze original weights by default - CRITICAL for memory savings
        # When requires_grad=False, PyTorch doesn't allocate gradient buffers
        self.linear.weight.requires_grad = False
        if hasattr(self.linear, 'bias') and self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r > 0 and not self.merged:
            # Compute LoRA update
            lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
            
            if self.use_dora:
                # DoRA: decompose into magnitude and direction
                # W' = m * (W + BA) / ||W + BA||
                weight = self.linear.weight + (self.lora_B @ self.lora_A) * self.scaling
                weight_norm = weight.norm(dim=1, keepdim=True)
                # FP16-safe: use 1e-6 instead of 1e-8 (1e-8 underflows in fp16)
                weight_normalized = weight / (weight_norm + 1e-6)
                result = F.linear(x, weight_normalized * self.magnitude.unsqueeze(1))
            else:
                result = self.linear(x) + lora_out
        else:
            result = self.linear(x)

        return result

    def merge_lora_weights(self):
        """Merge LoRA weights into the main weights for inference."""
        if self.r > 0 and not self.merged:
            delta = (self.lora_B @ self.lora_A) * self.scaling
            if self.use_dora:
                # For DoRA, we need to handle magnitude
                # FP16-safe: use 1e-6 instead of 1e-8 (1e-8 underflows in FP16)
                weight = self.linear.weight + delta
                weight_norm = weight.norm(dim=1, keepdim=True)
                self.linear.weight.data = (weight / (weight_norm + 1e-6)) * self.magnitude.unsqueeze(1)
            else:
                self.linear.weight.data += delta
            self.merged = True

    def unmerge_lora_weights(self):
        """Unmerge LoRA weights for continued training."""
        if self.r > 0 and self.merged:
            # Note: DoRA unmerge is approximate
            self.linear.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False


class LoRAConfig:
    """
    Configuration for SOTA LoRA adaptation.
    
    Supports multiple LoRA variants and configurations.
    """
    def __init__(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        enable_lora: bool = True,
        use_dora: bool = False,
        use_rslora: bool = True,
        lora_plus_lr_ratio: float = 16.0,  # LoRA+: B learns faster than A
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',  # Attention
            'gate_proj', 'up_proj', 'down_proj',  # MLP
        ]
        self.enable_lora = enable_lora
        self.use_dora = use_dora
        self.use_rslora = use_rslora
        self.lora_plus_lr_ratio = lora_plus_lr_ratio


def apply_lora_to_model(model: nn.Module, lora_config: LoRAConfig) -> nn.Module:
    """
    Apply LoRA to specified modules in a model.
    Returns the model with LoRA layers applied.
    
    MEMORY OPTIMIZATION:
    - Passes the original nn.Linear layer directly to LoRALinear
    - This SHARES weights instead of cloning them (saves ~50% memory for target modules)
    - Only LoRA parameters (A, B, magnitude) are newly allocated
    
    For a 16GB model with 30% of weights in target modules:
    - Old behavior: Clone ~5GB = 21GB total
    - New behavior: Share weights = 16GB + ~50MB LoRA params
    """
    if not lora_config.enable_lora:
        return model

    lora_layers_added = 0
    modules_to_replace = []
    total_base_params = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        module_name = name.split('.')[-1]
        if module_name in lora_config.target_modules:
            modules_to_replace.append((name, module))
            total_base_params += module.weight.numel()

    for name, module in modules_to_replace:
        parts = name.split('.')
        attr_name = parts[-1]
        parent_name = '.'.join(parts[:-1])

        if parent_name:
            parent = model.get_submodule(parent_name)
        else:
            parent = model

        # MEMORY OPTIMIZATION: Pass existing layer to share weights!
        # This is the KEY change - we don't clone weights anymore
        lora_layer = LoRALinear(
            in_features=module.in_features,
            out_features=module.out_features,
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            use_dora=lora_config.use_dora,
            use_rslora=lora_config.use_rslora,
            base_layer=module,  # PASS EXISTING LAYER - NO CLONING!
        )

        # NOTE: We no longer clone weights here!
        # The LoRALinear now uses the original module directly
        # This saves memory equal to the size of all target module weights

        setattr(parent, attr_name, lora_layer)
        lora_layers_added += 1

    # Calculate memory savings
    lora_params = lora_layers_added * (lora_config.r * (modules_to_replace[0][1].in_features + modules_to_replace[0][1].out_features)) if modules_to_replace else 0
    base_mem_saved_mb = (total_base_params * 2) / (1024 * 1024)  # FP16 = 2 bytes
    lora_mem_added_mb = (lora_params * 4) / (1024 * 1024)  # FP32 LoRA params = 4 bytes
    
    variant = "DoRA" if lora_config.use_dora else ("rsLoRA" if lora_config.use_rslora else "LoRA")
    print(f"✅ {variant} applied to {lora_layers_added} layers (r={lora_config.r}, alpha={lora_config.lora_alpha})")
    print(f"   💾 Memory optimization: {base_mem_saved_mb:.1f}MB base weights SHARED (not cloned)")
    print(f"   📊 New LoRA params: ~{lora_mem_added_mb:.1f}MB (trainable)")
    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get only the LoRA parameters from a model.
    
    NOTE: This does NOT change requires_grad on any parameters!
    It simply returns the LoRA params (lora_A, lora_B, magnitude).
    
    Use this when you want to get LoRA params for separate optimizer groups
    or for LoRA-only training mode.
    """
    lora_params = []
    
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name or 'magnitude' in name:
            lora_params.append(param)
    
    return lora_params


def enable_lora_training(model: nn.Module) -> List[nn.Parameter]:
    """
    Enable training for LoRA parameters (ensure requires_grad=True).
    
    Returns list of LoRA parameters.
    """
    lora_params = []
    
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name or 'magnitude' in name:
            param.requires_grad = True
            lora_params.append(param)
    
    return lora_params


def freeze_non_lora_params(model: nn.Module) -> int:
    """
    Freeze all non-LoRA parameters and clear their gradients.
    
    USE THIS ONLY FOR LORA-ONLY TRAINING MODE (train_lora_only=True).
    
    For normal training with parallel fine-tuning (LoRA + full weights on
    active components), use the model's freeze_components() method instead,
    which respects the training mode flags (--text, --video, --image, --voice).
    
    Returns:
        Number of frozen parameters
    """
    frozen_params = 0
    freed_memory = 0
    
    for name, param in model.named_parameters():
        is_lora = 'lora_A' in name or 'lora_B' in name or 'magnitude' in name
        if not is_lora:
            param.requires_grad = False
            frozen_params += param.numel()
            # Clear any accumulated gradients
            if param.grad is not None:
                freed_memory += param.grad.numel() * param.grad.element_size()
                param.grad = None
    
    print(f"   ❄️ Frozen {frozen_params:,} non-LoRA parameters")
    if freed_memory > 0:
        print(f"   🧹 Freed {freed_memory / (1024**2):.1f}MB of gradient memory")
    
    return frozen_params


def get_lora_plus_param_groups(
    model: nn.Module, 
    base_lr: float, 
    lr_ratio: float = 16.0
) -> List[Dict]:
    """
    Get parameter groups for LoRA+ training.
    
    LoRA+ uses different learning rates for A and B matrices:
    - B matrix: base_lr * lr_ratio (learns faster)
    - A matrix: base_lr
    
    This improves convergence and final performance.
    """
    lora_a_params = []
    lora_b_params = []
    magnitude_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora_A' in name:
            lora_a_params.append(param)
        elif 'lora_B' in name:
            lora_b_params.append(param)
        elif 'magnitude' in name:
            magnitude_params.append(param)
        else:
            other_params.append(param)
    
    param_groups = []
    
    if lora_a_params:
        param_groups.append({'params': lora_a_params, 'lr': base_lr, 'name': 'lora_A'})
    if lora_b_params:
        param_groups.append({'params': lora_b_params, 'lr': base_lr * lr_ratio, 'name': 'lora_B'})
    if magnitude_params:
        param_groups.append({'params': magnitude_params, 'lr': base_lr, 'name': 'magnitude'})
    if other_params:
        param_groups.append({'params': other_params, 'lr': base_lr, 'name': 'other'})
    
    return param_groups


def get_trainable_parameters(model: nn.Module, train_lora_only: bool = False) -> List[nn.Parameter]:
    """Get trainable parameters, optionally only LoRA params."""
    if train_lora_only:
        return get_lora_parameters(model)
    else:
        return [p for p in model.parameters() if p.requires_grad]


def count_lora_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """
    Count LoRA parameters vs total parameters.
    
    Returns:
        (lora_params, total_params, percentage)
    """
    lora_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if 'lora_A' in name or 'lora_B' in name or 'magnitude' in name:
            lora_params += param.numel()
    
    percentage = 100.0 * lora_params / total_params if total_params > 0 else 0.0
    return lora_params, total_params, percentage

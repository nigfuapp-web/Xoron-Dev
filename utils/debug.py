"""
Deep debugging utilities for FP16 NaN/Inf issues.

This module provides comprehensive debugging tools to trace exactly where
NaN/Inf values originate in the forward pass.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict


class NaNDebugger:
    """
    Comprehensive NaN/Inf debugger that hooks into model layers
    to trace exactly where numerical instability occurs.
    """
    
    def __init__(self, enabled: bool = True, verbose: bool = True):
        self.enabled = enabled
        self.verbose = verbose
        self.hooks = []
        self.nan_locations = []
        self.tensor_stats = {}
        self.layer_outputs = {}
        
    def reset(self):
        """Reset tracking state for new forward pass."""
        self.nan_locations = []
        self.tensor_stats = {}
        self.layer_outputs = {}
        
    def check_tensor(self, name: str, tensor: torch.Tensor, 
                      record_stats: bool = True) -> Tuple[bool, bool, Dict]:
        """
        Check a tensor for NaN/Inf and record statistics.
        
        Returns:
            (has_nan, has_inf, stats_dict)
        """
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return False, False, {}
            
        if tensor.numel() == 0:
            return False, False, {}
        
        # Detach and move to CPU for analysis if needed
        t = tensor.detach()
        
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        
        stats = {}
        if record_stats:
            try:
                flat = t.float().flatten()
                finite_mask = torch.isfinite(flat)
                if finite_mask.any():
                    finite_vals = flat[finite_mask]
                    stats = {
                        'shape': list(tensor.shape),
                        'dtype': str(tensor.dtype),
                        'min': finite_vals.min().item(),
                        'max': finite_vals.max().item(),
                        'mean': finite_vals.mean().item(),
                        'std': finite_vals.std().item() if finite_vals.numel() > 1 else 0.0,
                        'abs_max': finite_vals.abs().max().item(),
                        'nan_count': torch.isnan(flat).sum().item(),
                        'inf_count': torch.isinf(flat).sum().item(),
                        'nan_pct': 100 * torch.isnan(flat).sum().item() / flat.numel(),
                        'inf_pct': 100 * torch.isinf(flat).sum().item() / flat.numel(),
                    }
                else:
                    stats = {
                        'shape': list(tensor.shape),
                        'dtype': str(tensor.dtype),
                        'all_nan_or_inf': True,
                        'nan_count': torch.isnan(flat).sum().item(),
                        'inf_count': torch.isinf(flat).sum().item(),
                    }
            except Exception as e:
                stats = {'error': str(e)}
        
        if has_nan or has_inf:
            self.nan_locations.append({
                'name': name,
                'has_nan': has_nan,
                'has_inf': has_inf,
                'stats': stats
            })
            
        self.tensor_stats[name] = stats
        return has_nan, has_inf, stats
        
    def print_tensor_info(self, name: str, tensor: torch.Tensor, 
                          always_print: bool = False):
        """Print detailed tensor information."""
        has_nan, has_inf, stats = self.check_tensor(name, tensor)
        
        if not self.verbose and not has_nan and not has_inf and not always_print:
            return
            
        status = "✅" if not (has_nan or has_inf) else "❌ NaN/Inf"
        
        if stats:
            print(f"  [{status}] {name}:")
            print(f"      shape={stats.get('shape')}, dtype={stats.get('dtype')}")
            if 'all_nan_or_inf' in stats:
                print(f"      ⚠️ ALL VALUES ARE NaN/Inf!")
                print(f"      nan_count={stats.get('nan_count')}, inf_count={stats.get('inf_count')}")
            else:
                print(f"      min={stats.get('min', 'N/A'):.6f}, max={stats.get('max', 'N/A'):.6f}")
                print(f"      mean={stats.get('mean', 'N/A'):.6f}, std={stats.get('std', 'N/A'):.6f}")
                print(f"      abs_max={stats.get('abs_max', 'N/A'):.6f}")
                if stats.get('nan_count', 0) > 0 or stats.get('inf_count', 0) > 0:
                    print(f"      ⚠️ nan_count={stats.get('nan_count')}, inf_count={stats.get('inf_count')}")
                    print(f"      ⚠️ nan_pct={stats.get('nan_pct', 0):.2f}%, inf_pct={stats.get('inf_pct', 0):.2f}%")
    
    def get_summary(self) -> str:
        """Get a summary of all NaN/Inf locations found."""
        if not self.nan_locations:
            return "✅ No NaN/Inf detected in any tracked tensors."
        
        lines = [f"❌ NaN/Inf detected in {len(self.nan_locations)} locations:"]
        for loc in self.nan_locations:
            lines.append(f"  - {loc['name']}: nan={loc['has_nan']}, inf={loc['has_inf']}")
            if loc['stats']:
                lines.append(f"    stats: {loc['stats']}")
        return "\n".join(lines)


def debug_attention_forward(attn_module, hidden_states, attention_mask=None, 
                            position_ids=None, past_key_value=None,
                            debugger: Optional[NaNDebugger] = None,
                            layer_name: str = "attention"):
    """
    Debug wrapper for attention forward pass.
    Tracks tensor values at each step to identify where NaN/Inf originates.
    """
    if debugger is None:
        debugger = NaNDebugger(verbose=True)
    
    print(f"\n{'='*60}")
    print(f"🔍 DEBUGGING {layer_name}")
    print(f"{'='*60}")
    
    # Check input
    debugger.print_tensor_info(f"{layer_name}/input_hidden", hidden_states, always_print=True)
    
    # Get internal attributes
    head_dim = getattr(attn_module, 'head_dim', None)
    num_heads = getattr(attn_module, 'num_heads', None)
    qk_scale = getattr(attn_module, 'qk_scale', None)
    
    print(f"  Config: head_dim={head_dim}, num_heads={num_heads}, qk_scale={qk_scale}")
    
    # Track Q, K, V projections
    if hasattr(attn_module, 'q_proj'):
        q = attn_module.q_proj(hidden_states)
        debugger.print_tensor_info(f"{layer_name}/Q_proj_output", q, always_print=True)
        
    if hasattr(attn_module, 'k_proj'):
        k = attn_module.k_proj(hidden_states)
        debugger.print_tensor_info(f"{layer_name}/K_proj_output", k, always_print=True)
        
    if hasattr(attn_module, 'v_proj'):
        v = attn_module.v_proj(hidden_states)
        debugger.print_tensor_info(f"{layer_name}/V_proj_output", v, always_print=True)
    
    # Check projection weights
    if hasattr(attn_module, 'q_proj'):
        debugger.print_tensor_info(f"{layer_name}/Q_proj_weight", attn_module.q_proj.weight)
    if hasattr(attn_module, 'k_proj'):
        debugger.print_tensor_info(f"{layer_name}/K_proj_weight", attn_module.k_proj.weight)
    if hasattr(attn_module, 'v_proj'):
        debugger.print_tensor_info(f"{layer_name}/V_proj_weight", attn_module.v_proj.weight)
    
    print(f"\n{debugger.get_summary()}")
    return debugger


def debug_full_forward(model, input_ids, attention_mask=None, labels=None,
                       max_layers_to_check: int = 3):
    """
    Run a full debugging forward pass through the model.
    Checks tensors at every critical point.
    """
    debugger = NaNDebugger(verbose=True)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    print(f"\n{'#'*70}")
    print(f"# DEEP DEBUG FORWARD PASS")
    print(f"# Device: {device}, Model dtype: {dtype}")
    print(f"{'#'*70}\n")
    
    # Move inputs to device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    if labels is not None:
        labels = labels.to(device)
    
    # Debug input
    print("="*60)
    print("STEP 1: INPUT")
    print("="*60)
    debugger.print_tensor_info("input_ids", input_ids.float(), always_print=True)
    print(f"  input_ids range: [{input_ids.min().item()}, {input_ids.max().item()}]")
    
    # Get LLM from model
    llm = None
    if hasattr(model, 'llm'):
        llm = model.llm
    elif hasattr(model, 'model'):
        llm = model.model
    
    if llm is None:
        print("❌ Could not find LLM in model!")
        return debugger
        
    # Check embedding
    print("\n" + "="*60)
    print("STEP 2: EMBEDDINGS")
    print("="*60)
    
    if hasattr(llm, 'model') and hasattr(llm.model, 'embed_tokens'):
        embed = llm.model.embed_tokens
    elif hasattr(llm, 'embed_tokens'):
        embed = llm.embed_tokens
    else:
        print("❌ Could not find embedding layer!")
        return debugger
    
    debugger.print_tensor_info("embed_weight", embed.weight, always_print=True)
    
    # Get embeddings
    with torch.no_grad():
        embeddings = embed(input_ids)
    debugger.print_tensor_info("embeddings_output", embeddings, always_print=True)
    
    # Check first few transformer layers
    print("\n" + "="*60)
    print("STEP 3: TRANSFORMER LAYERS")
    print("="*60)
    
    layers = None
    if hasattr(llm, 'model') and hasattr(llm.model, 'layers'):
        layers = llm.model.layers
    elif hasattr(llm, 'layers'):
        layers = llm.layers
    
    if layers is None:
        print("❌ Could not find transformer layers!")
        return debugger
        
    print(f"  Found {len(layers)} layers, checking first {max_layers_to_check}")
    
    hidden = embeddings
    for i, layer in enumerate(layers[:max_layers_to_check]):
        print(f"\n  --- Layer {i} ---")
        
        # Input to layer
        debugger.print_tensor_info(f"layer_{i}/input", hidden, always_print=True)
        
        # Check layer norm weights
        if hasattr(layer, 'input_layernorm'):
            debugger.print_tensor_info(f"layer_{i}/input_ln_weight", layer.input_layernorm.weight)
        
        # Check attention
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            debugger.print_tensor_info(f"layer_{i}/attn_q_weight", attn.q_proj.weight)
            debugger.print_tensor_info(f"layer_{i}/attn_k_weight", attn.k_proj.weight)
            debugger.print_tensor_info(f"layer_{i}/attn_v_weight", attn.v_proj.weight)
            debugger.print_tensor_info(f"layer_{i}/attn_o_weight", attn.o_proj.weight)
            
            # Check qk_scale
            if hasattr(attn, 'qk_scale'):
                print(f"    qk_scale = {attn.qk_scale}")
        
        # Check MLP/MoE
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            if hasattr(mlp, 'router'):  # MoE layer
                print(f"    MoE layer detected")
                debugger.print_tensor_info(f"layer_{i}/moe_router_weight", mlp.router.gate.weight)
            elif hasattr(mlp, 'gate_proj'):
                debugger.print_tensor_info(f"layer_{i}/mlp_gate_weight", mlp.gate_proj.weight)
        
        # Run layer forward
        with torch.no_grad():
            try:
                # Create position_ids
                seq_len = hidden.shape[1]
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                
                layer_out = layer(hidden, position_ids=position_ids)
                if isinstance(layer_out, tuple):
                    hidden = layer_out[0]
                else:
                    hidden = layer_out
                    
                debugger.print_tensor_info(f"layer_{i}/output", hidden, always_print=True)
            except Exception as e:
                print(f"    ❌ Layer forward failed: {e}")
                break
    
    # Check final norm and lm_head
    print("\n" + "="*60)
    print("STEP 4: OUTPUT PROJECTION")
    print("="*60)
    
    if hasattr(llm, 'model') and hasattr(llm.model, 'norm'):
        debugger.print_tensor_info("final_norm_weight", llm.model.norm.weight, always_print=True)
        with torch.no_grad():
            hidden = llm.model.norm(hidden)
        debugger.print_tensor_info("after_final_norm", hidden, always_print=True)
        
    if hasattr(llm, 'lm_head'):
        debugger.print_tensor_info("lm_head_weight", llm.lm_head.weight, always_print=True)
        with torch.no_grad():
            logits = llm.lm_head(hidden)
        debugger.print_tensor_info("logits", logits, always_print=True)
    
    # Final summary
    print("\n" + "#"*70)
    print("# SUMMARY")
    print("#"*70)
    print(debugger.get_summary())
    
    return debugger


def add_nan_hooks(model, print_every: int = 1):
    """
    Add forward hooks to all layers to detect NaN/Inf.
    
    Returns:
        list of hooks (call hook.remove() to remove them)
    """
    hooks = []
    nan_detected = {'count': 0, 'locations': []}
    
    def make_hook(name):
        call_count = [0]
        def hook(module, input, output):
            call_count[0] += 1
            
            # Check input
            if isinstance(input, tuple):
                for i, inp in enumerate(input):
                    if isinstance(inp, torch.Tensor):
                        if torch.isnan(inp).any() or torch.isinf(inp).any():
                            nan_detected['count'] += 1
                            nan_detected['locations'].append(f"{name}/input[{i}]")
                            if call_count[0] % print_every == 0:
                                print(f"❌ NaN/Inf in {name}/input[{i}]")
            
            # Check output
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any() or torch.isinf(output).any():
                    nan_detected['count'] += 1
                    nan_detected['locations'].append(f"{name}/output")
                    if call_count[0] % print_every == 0:
                        has_nan = torch.isnan(output).any().item()
                        has_inf = torch.isinf(output).any().item()
                        nan_pct = 100 * torch.isnan(output).sum().item() / output.numel()
                        inf_pct = 100 * torch.isinf(output).sum().item() / output.numel()
                        print(f"❌ NaN/Inf in {name}/output: nan={has_nan}({nan_pct:.2f}%), inf={has_inf}({inf_pct:.2f}%)")
                        
                        # Print more details
                        finite = output[torch.isfinite(output)]
                        if finite.numel() > 0:
                            print(f"   Finite values: min={finite.min():.4f}, max={finite.max():.4f}, mean={finite.mean():.4f}")
            elif isinstance(output, tuple):
                for i, out in enumerate(output):
                    if isinstance(out, torch.Tensor):
                        if torch.isnan(out).any() or torch.isinf(out).any():
                            nan_detected['count'] += 1
                            nan_detected['locations'].append(f"{name}/output[{i}]")
                            if call_count[0] % print_every == 0:
                                print(f"❌ NaN/Inf in {name}/output[{i}]")
                                
        return hook
    
    for name, module in model.named_modules():
        if name:  # Skip root module
            hook = module.register_forward_hook(make_hook(name))
            hooks.append(hook)
    
    print(f"📌 Added {len(hooks)} NaN detection hooks")
    return hooks, nan_detected


def remove_hooks(hooks):
    """Remove all hooks."""
    for hook in hooks:
        hook.remove()
    print(f"🗑️ Removed {len(hooks)} hooks")

"""
Deep debugging utilities for FP16 NaN/Inf issues.

This module provides comprehensive debugging tools to trace exactly where
NaN/Inf values originate in the forward pass and during optimizer steps.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict


class WeightCorruptionTracker:
    """
    Aggressive weight corruption tracker that checks weights AFTER every optimizer step
    to pinpoint exactly when and where corruption occurs.
    
    Usage:
        tracker = WeightCorruptionTracker(model)
        
        # In training loop, AFTER optimizer.step():
        tracker.check_and_report(step=global_step, batch_idx=batch_idx)
    """
    
    def __init__(self, model, verbose: bool = True, check_freq: int = 1):
        """
        Args:
            model: The model to track
            verbose: Whether to print detailed info
            check_freq: How often to check (1 = every step, 10 = every 10 steps)
        """
        self.model = model
        self.verbose = verbose
        self.check_freq = check_freq
        self.last_healthy_step = -1
        self.corruption_detected_at = None
        self.corruption_details = {}
        self.weight_history = {}  # Store snapshots of key weights
        self._snapshot_key_weights()
        
    def _snapshot_key_weights(self):
        """Take a snapshot of key weight statistics for comparison."""
        self.weight_history['initial'] = {}
        for name, param in self.model.named_parameters():
            if param.numel() > 0:
                with torch.no_grad():
                    self.weight_history['initial'][name] = {
                        'mean': param.float().mean().item(),
                        'std': param.float().std().item() if param.numel() > 1 else 0.0,
                        'abs_max': param.float().abs().max().item(),
                        'nan_count': torch.isnan(param).sum().item(),
                        'inf_count': torch.isinf(param).sum().item(),
                    }
    
    def check_weights(self, step: int = -1, batch_idx: int = -1) -> Tuple[bool, Dict]:
        """
        Check all model weights for NaN/Inf.
        
        Returns:
            (is_healthy, details_dict)
        """
        corrupted_params = []
        healthy_count = 0
        total_params = 0
        
        for name, param in self.model.named_parameters():
            if param.numel() == 0:
                continue
                
            total_params += 1
            has_nan = torch.isnan(param).any().item()
            has_inf = torch.isinf(param).any().item()
            
            if has_nan or has_inf:
                nan_count = torch.isnan(param).sum().item()
                inf_count = torch.isinf(param).sum().item()
                nan_pct = 100 * nan_count / param.numel()
                inf_pct = 100 * inf_count / param.numel()
                
                corrupted_params.append({
                    'name': name,
                    'shape': list(param.shape),
                    'dtype': str(param.dtype),
                    'nan_count': nan_count,
                    'inf_count': inf_count,
                    'nan_pct': nan_pct,
                    'inf_pct': inf_pct,
                    'total_elements': param.numel(),
                })
            else:
                healthy_count += 1
        
        is_healthy = len(corrupted_params) == 0
        
        details = {
            'step': step,
            'batch_idx': batch_idx,
            'total_params': total_params,
            'healthy_count': healthy_count,
            'corrupted_count': len(corrupted_params),
            'corrupted_params': corrupted_params,
        }
        
        return is_healthy, details
    
    def check_gradients(self, step: int = -1, batch_idx: int = -1) -> Tuple[bool, Dict]:
        """
        Check all gradients for NaN/Inf BEFORE optimizer step.
        
        Returns:
            (is_healthy, details_dict)
        """
        corrupted_grads = []
        healthy_count = 0
        total_grads = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if param.grad.numel() == 0:
                continue
                
            total_grads += 1
            has_nan = torch.isnan(param.grad).any().item()
            has_inf = torch.isinf(param.grad).any().item()
            
            if has_nan or has_inf:
                nan_count = torch.isnan(param.grad).sum().item()
                inf_count = torch.isinf(param.grad).sum().item()
                nan_pct = 100 * nan_count / param.grad.numel()
                inf_pct = 100 * inf_count / param.grad.numel()
                
                # Get gradient stats
                finite_mask = torch.isfinite(param.grad)
                if finite_mask.any():
                    finite_grads = param.grad[finite_mask].float()
                    grad_mean = finite_grads.mean().item()
                    grad_abs_max = finite_grads.abs().max().item()
                else:
                    grad_mean = float('nan')
                    grad_abs_max = float('nan')
                
                corrupted_grads.append({
                    'name': name,
                    'shape': list(param.grad.shape),
                    'dtype': str(param.grad.dtype),
                    'nan_count': nan_count,
                    'inf_count': inf_count,
                    'nan_pct': nan_pct,
                    'inf_pct': inf_pct,
                    'grad_mean': grad_mean,
                    'grad_abs_max': grad_abs_max,
                })
            else:
                healthy_count += 1
        
        is_healthy = len(corrupted_grads) == 0
        
        details = {
            'step': step,
            'batch_idx': batch_idx,
            'total_grads': total_grads,
            'healthy_count': healthy_count,
            'corrupted_count': len(corrupted_grads),
            'corrupted_grads': corrupted_grads,
        }
        
        return is_healthy, details
    
    def check_and_report(self, step: int = -1, batch_idx: int = -1, 
                         phase: str = "post_optimizer") -> bool:
        """
        Check weights and report if corruption detected.
        
        Args:
            step: Global training step
            batch_idx: Current batch index
            phase: When this check is being done (pre_backward, post_backward, 
                   pre_optimizer, post_optimizer)
        
        Returns:
            True if healthy, False if corrupted
        """
        if step % self.check_freq != 0 and self.corruption_detected_at is None:
            return True
            
        is_healthy, details = self.check_weights(step, batch_idx)
        
        if is_healthy:
            self.last_healthy_step = step
            if self.verbose and step % 100 == 0:
                print(f"   ✅ [{phase}] Step {step}, batch {batch_idx}: All {details['total_params']} params healthy")
            return True
        
        # CORRUPTION DETECTED!
        if self.corruption_detected_at is None:
            self.corruption_detected_at = step
            self.corruption_details = details
            
            print(f"\n{'!'*70}")
            print(f"! 🚨 WEIGHT CORRUPTION DETECTED!")
            print(f"! Phase: {phase}")
            print(f"! Step: {step}, Batch: {batch_idx}")
            print(f"! Last healthy step: {self.last_healthy_step}")
            print(f"! Corrupted params: {details['corrupted_count']}/{details['total_params']}")
            print(f"{'!'*70}")
            
            # Print details of first 10 corrupted params
            print(f"\n📋 CORRUPTED PARAMETERS (first 10):")
            for i, cp in enumerate(details['corrupted_params'][:10]):
                print(f"   {i+1}. {cp['name']}")
                print(f"      shape={cp['shape']}, dtype={cp['dtype']}")
                print(f"      NaN: {cp['nan_count']}/{cp['total_elements']} ({cp['nan_pct']:.2f}%)")
                print(f"      Inf: {cp['inf_count']}/{cp['total_elements']} ({cp['inf_pct']:.2f}%)")
            
            if len(details['corrupted_params']) > 10:
                print(f"   ... and {len(details['corrupted_params']) - 10} more corrupted params")
            
            # Check if it's the embedding layer (common culprit)
            embed_corrupted = [p for p in details['corrupted_params'] if 'embed' in p['name'].lower()]
            if embed_corrupted:
                print(f"\n⚠️ EMBEDDING LAYER IS CORRUPTED!")
                print(f"   This is often caused by:")
                print(f"   1. Large gradients on rare tokens causing fp16 overflow")
                print(f"   2. Learning rate too high for embedding layer")
                print(f"   3. Bad token IDs in input (out of vocab range)")
            
            # Compare with initial snapshot
            print(f"\n📊 COMPARISON WITH INITIAL WEIGHTS:")
            for cp in details['corrupted_params'][:5]:
                name = cp['name']
                if name in self.weight_history.get('initial', {}):
                    init = self.weight_history['initial'][name]
                    print(f"   {name}:")
                    print(f"      Initial: mean={init['mean']:.6f}, abs_max={init['abs_max']:.6f}")
                    print(f"      Now: CORRUPTED ({cp['nan_pct']:.1f}% NaN, {cp['inf_pct']:.1f}% Inf)")
            
            print(f"\n{'!'*70}\n")
        
        return False
    
    def get_corruption_report(self) -> Optional[Dict]:
        """Get the full corruption report if corruption was detected."""
        if self.corruption_detected_at is None:
            return None
        return {
            'detected_at_step': self.corruption_detected_at,
            'last_healthy_step': self.last_healthy_step,
            'details': self.corruption_details,
        }


def check_weights_detailed(model, step: int = -1, prefix: str = "") -> Tuple[bool, List[Dict]]:
    """
    Detailed weight check - returns list of all parameter stats.
    Useful for logging to file.
    """
    results = []
    all_healthy = True
    
    for name, param in model.named_parameters():
        if param.numel() == 0:
            continue
            
        has_nan = torch.isnan(param).any().item()
        has_inf = torch.isinf(param).any().item()
        
        with torch.no_grad():
            p = param.float()
            finite_mask = torch.isfinite(p)
            
            if finite_mask.any():
                finite_vals = p[finite_mask]
                stats = {
                    'step': step,
                    'name': f"{prefix}{name}",
                    'shape': list(param.shape),
                    'dtype': str(param.dtype),
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                    'nan_count': torch.isnan(param).sum().item(),
                    'inf_count': torch.isinf(param).sum().item(),
                    'mean': finite_vals.mean().item(),
                    'std': finite_vals.std().item() if finite_vals.numel() > 1 else 0.0,
                    'min': finite_vals.min().item(),
                    'max': finite_vals.max().item(),
                    'abs_max': finite_vals.abs().max().item(),
                }
            else:
                stats = {
                    'step': step,
                    'name': f"{prefix}{name}",
                    'shape': list(param.shape),
                    'dtype': str(param.dtype),
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                    'nan_count': torch.isnan(param).sum().item(),
                    'inf_count': torch.isinf(param).sum().item(),
                    'mean': float('nan'),
                    'std': float('nan'),
                    'min': float('nan'),
                    'max': float('nan'),
                    'abs_max': float('nan'),
                    'ALL_CORRUPTED': True,
                }
        
        results.append(stats)
        
        if has_nan or has_inf:
            all_healthy = False
    
    return all_healthy, results


def check_gradients_detailed(model, step: int = -1, prefix: str = "") -> Tuple[bool, List[Dict]]:
    """
    Detailed gradient check - returns list of all gradient stats.
    """
    results = []
    all_healthy = True
    
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if param.grad.numel() == 0:
            continue
            
        has_nan = torch.isnan(param.grad).any().item()
        has_inf = torch.isinf(param.grad).any().item()
        
        with torch.no_grad():
            g = param.grad.float()
            finite_mask = torch.isfinite(g)
            
            if finite_mask.any():
                finite_grads = g[finite_mask]
                stats = {
                    'step': step,
                    'name': f"{prefix}{name}",
                    'shape': list(param.grad.shape),
                    'dtype': str(param.grad.dtype),
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                    'nan_count': torch.isnan(param.grad).sum().item(),
                    'inf_count': torch.isinf(param.grad).sum().item(),
                    'mean': finite_grads.mean().item(),
                    'std': finite_grads.std().item() if finite_grads.numel() > 1 else 0.0,
                    'min': finite_grads.min().item(),
                    'max': finite_grads.max().item(),
                    'abs_max': finite_grads.abs().max().item(),
                }
            else:
                stats = {
                    'step': step,
                    'name': f"{prefix}{name}",
                    'shape': list(param.grad.shape),
                    'dtype': str(param.grad.dtype),
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                    'nan_count': torch.isnan(param.grad).sum().item(),
                    'inf_count': torch.isinf(param.grad).sum().item(),
                    'mean': float('nan'),
                    'std': float('nan'),
                    'min': float('nan'),
                    'max': float('nan'),
                    'abs_max': float('nan'),
                    'ALL_CORRUPTED': True,
                }
        
        results.append(stats)
        
        if has_nan or has_inf:
            all_healthy = False
    
    return all_healthy, results


def diagnose_corruption_source(model, input_ids, labels=None) -> Dict:
    """
    Run a diagnostic to try to identify the source of weight corruption.
    Call this AFTER corruption is detected.
    
    Returns a diagnosis dict with suspected causes.
    """
    diagnosis = {
        'suspected_causes': [],
        'embedding_issues': [],
        'layer_issues': [],
        'recommendations': [],
    }
    
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    # Check 1: Input token IDs
    # NOTE: Use ACTUAL embedding size, not config.vocab_size (may be stale after resize)
    vocab_size = None
    if hasattr(model, 'llm'):
        if hasattr(model.llm, 'model') and hasattr(model.llm.model, 'embed_tokens'):
            vocab_size = model.llm.model.embed_tokens.weight.shape[0]
        elif hasattr(model.llm, 'embed_tokens'):
            vocab_size = model.llm.embed_tokens.weight.shape[0]
        # Fallback to config
        if vocab_size is None and hasattr(model.llm, 'config'):
            vocab_size = getattr(model.llm.config, 'vocab_size', None)
    
    max_token = input_ids.max().item()
    min_token = input_ids.min().item()
    
    if vocab_size is not None and max_token >= vocab_size:
        diagnosis['suspected_causes'].append(
            f"TOKEN ID OUT OF RANGE: max_token={max_token} >= vocab_size={vocab_size}"
        )
        diagnosis['recommendations'].append(
            "Check tokenizer and data pipeline - token IDs exceed vocabulary size"
        )
    
    if min_token < 0:
        diagnosis['suspected_causes'].append(
            f"NEGATIVE TOKEN ID: min_token={min_token}"
        )
    
    # Check 2: Embedding layer
    embed_layer = None
    if hasattr(model, 'llm'):
        llm = model.llm
        if hasattr(llm, 'model') and hasattr(llm.model, 'embed_tokens'):
            embed_layer = llm.model.embed_tokens
        elif hasattr(llm, 'embed_tokens'):
            embed_layer = llm.embed_tokens
    
    if embed_layer is not None:
        weight = embed_layer.weight
        nan_rows = torch.isnan(weight).any(dim=1).sum().item()
        inf_rows = torch.isinf(weight).any(dim=1).sum().item()
        total_rows = weight.shape[0]
        
        diagnosis['embedding_issues'].append({
            'total_vocab': total_rows,
            'nan_rows': nan_rows,
            'inf_rows': inf_rows,
            'nan_pct': 100 * nan_rows / total_rows,
        })
        
        if nan_rows > 0 or inf_rows > 0:
            # Which specific rows are corrupted?
            nan_row_indices = torch.where(torch.isnan(weight).any(dim=1))[0].tolist()[:20]
            inf_row_indices = torch.where(torch.isinf(weight).any(dim=1))[0].tolist()[:20]
            
            diagnosis['embedding_issues'].append({
                'first_nan_rows': nan_row_indices,
                'first_inf_rows': inf_row_indices,
            })
            
            # Check if input contains these corrupted tokens
            tokens_used = input_ids.unique().tolist()
            corrupted_tokens_in_input = [t for t in tokens_used if t in nan_row_indices or t in inf_row_indices]
            
            if corrupted_tokens_in_input:
                diagnosis['suspected_causes'].append(
                    f"INPUT USES CORRUPTED EMBEDDING ROWS: tokens {corrupted_tokens_in_input[:10]}"
                )
    
    # Check 3: LayerNorm scales
    for name, param in model.named_parameters():
        if 'norm' in name.lower() and 'weight' in name.lower():
            if torch.isnan(param).any() or torch.isinf(param).any():
                diagnosis['layer_issues'].append(
                    f"LayerNorm corrupted: {name}"
                )
    
    # Generate recommendations
    if len(diagnosis['suspected_causes']) == 0 and len(diagnosis['embedding_issues']) > 0:
        emb = diagnosis['embedding_issues'][0]
        if emb['nan_pct'] > 50:
            diagnosis['recommendations'].append(
                "More than 50% of embeddings corrupted - likely gradient explosion. "
                "Try: 1) Lower learning rate 10x, 2) Enable gradient clipping, "
                "3) Use loss scaling, 4) Check for rare tokens with high gradients"
            )
        else:
            diagnosis['recommendations'].append(
                f"Partial embedding corruption ({emb['nan_pct']:.1f}%). "
                "Likely caused by high gradients on specific tokens. "
                "Try: 1) Add embedding gradient clipping, 2) Warm up embedding LR"
            )
    
    return diagnosis


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


class GradientNormTracker:
    """
    Track gradient norms over time to identify when they start spiking
    before corruption occurs.
    
    Usage:
        tracker = GradientNormTracker(model)
        
        # After backward, before optimizer step:
        tracker.record(step=global_step)
        
        # If corruption occurs, print history:
        tracker.print_history(last_n=20)
    """
    
    def __init__(self, model, track_embeddings: bool = True, track_layers: bool = True):
        self.model = model
        self.track_embeddings = track_embeddings
        self.track_layers = track_layers
        self.history = []
        self.max_history = 1000
        
    def compute_grad_norms(self) -> Dict[str, float]:
        """Compute gradient norms for key parameter groups."""
        norms = {}
        
        total_norm = 0.0
        embed_norm = 0.0
        layer_norms = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            
            grad_norm = param.grad.data.float().norm(2).item()
            total_norm += grad_norm ** 2
            
            # Track embedding gradients specifically
            if 'embed' in name.lower():
                embed_norm += grad_norm ** 2
            
            # Track per-layer gradients
            if self.track_layers:
                for layer_num in range(100):  # Reasonable max
                    if f'.{layer_num}.' in name or f'layer_{layer_num}' in name or f'layers.{layer_num}' in name:
                        if layer_num not in layer_norms:
                            layer_norms[layer_num] = 0.0
                        layer_norms[layer_num] += grad_norm ** 2
                        break
        
        norms['total'] = total_norm ** 0.5
        norms['embed'] = embed_norm ** 0.5
        
        for layer_num, norm_sq in layer_norms.items():
            norms[f'layer_{layer_num}'] = norm_sq ** 0.5
        
        return norms
    
    def record(self, step: int, batch_idx: int = -1, loss: float = None) -> Dict[str, float]:
        """Record gradient norms at this step."""
        norms = self.compute_grad_norms()
        
        record = {
            'step': step,
            'batch_idx': batch_idx,
            'loss': loss,
            **norms
        }
        
        self.history.append(record)
        
        # Keep history bounded
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        return norms
    
    def check_spike(self, threshold_multiplier: float = 10.0) -> Tuple[bool, Optional[Dict]]:
        """
        Check if current gradient norm is a spike compared to recent history.
        
        Returns:
            (is_spike, details)
        """
        if len(self.history) < 5:
            return False, None
        
        current = self.history[-1]
        recent = self.history[-10:-1]  # Last 9 before current
        
        if len(recent) == 0:
            return False, None
        
        avg_total = sum(r['total'] for r in recent) / len(recent)
        
        if current['total'] > avg_total * threshold_multiplier:
            return True, {
                'current_norm': current['total'],
                'avg_recent': avg_total,
                'multiplier': current['total'] / avg_total if avg_total > 0 else float('inf'),
                'step': current['step'],
                'batch_idx': current['batch_idx'],
            }
        
        return False, None
    
    def print_history(self, last_n: int = 20):
        """Print the last N gradient norm records."""
        records = self.history[-last_n:]
        
        print(f"\n{'='*70}")
        print(f"GRADIENT NORM HISTORY (last {len(records)} steps)")
        print(f"{'='*70}")
        
        print(f"{'Step':>6} {'Batch':>6} {'Total Norm':>12} {'Embed Norm':>12} {'Loss':>10}")
        print(f"{'-'*6} {'-'*6} {'-'*12} {'-'*12} {'-'*10}")
        
        for r in records:
            loss_str = f"{r['loss']:.4f}" if r['loss'] is not None else "N/A"
            print(f"{r['step']:>6} {r['batch_idx']:>6} {r['total']:>12.4f} {r['embed']:>12.4f} {loss_str:>10}")
        
        # Print analysis
        if len(records) >= 2:
            total_norms = [r['total'] for r in records]
            embed_norms = [r['embed'] for r in records]
            
            print(f"\n📊 ANALYSIS:")
            print(f"   Total norm: min={min(total_norms):.4f}, max={max(total_norms):.4f}, "
                  f"ratio={max(total_norms)/max(min(total_norms), 1e-8):.2f}x")
            print(f"   Embed norm: min={min(embed_norms):.4f}, max={max(embed_norms):.4f}, "
                  f"ratio={max(embed_norms)/max(min(embed_norms), 1e-8):.2f}x")
            
            if max(total_norms) / max(min(total_norms), 1e-8) > 100:
                print(f"   ⚠️ HIGH VARIANCE in gradient norms detected!")
                print(f"   This often precedes NaN/Inf corruption.")
        
        print(f"{'='*70}\n")
    
    def get_corruption_context(self, before_n: int = 10) -> List[Dict]:
        """Get the gradient history leading up to potential corruption."""
        return self.history[-before_n:]

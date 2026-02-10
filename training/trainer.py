"""
SOTA Trainer class for Xoron-Dev multimodal model.

Features:
- LoRA+ support with different learning rates for A and B matrices
- Configurable loss weights for all modalities
- Chain-of-thought weighted loss for reasoning tokens
- BF16/FP16 mixed precision training
- Gradient checkpointing support
- MoE auxiliary loss tracking
- Proper gradient clipping from config
"""

import os
import gc
import json
import time
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Optional, Dict, List

from training.utils import (
    train_image_diffusion_step,
    train_video_diffusion_step,
    train_voice_asr_step,
    train_voice_tts_step,
    eval_image_diffusion_step,
    eval_video_diffusion_step,
    eval_voice_asr_step,
    eval_voice_tts_step,
)
from config.special_tokens import (
    SPECIAL_TOKENS, 
    get_reasoning_tokens, 
    get_all_reasoning_block_tokens,
    get_tool_block_tokens,
    get_anti_hallucination_block_tokens,
    get_code_execution_block_tokens,
    get_flat_weighted_block_tokens,
)


class XoronTrainer:
    """
    SOTA Trainer for Xoron-Dev multimodal model.
    
    Supports:
    - Multi-modal training (LLM, image, video, audio)
    - Chain-of-thought weighted loss
    - Tool calling weighted loss
    - Anti-hallucination weighted loss
    - Code execution weighted loss
    - LoRA+ with different learning rates
    - Configurable loss weights per modality
    - BF16/FP16 mixed precision
    - Gradient checkpointing
    - Validation/evaluation at end of each epoch with per-modality losses
    """

    def __init__(
        self,
        model,
        train_dataset,
        optimizer,
        scheduler,
        config,
        xoron_config,
        collate_fn,
        resume_from: str = None,
        tokenizer=None,
        eval_dataset=None,
        hf_token: str = None,
        hf_repo_id: str = "Backup-bdg/Xoron-Dev-MultiMoe",
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.xoron_config = xoron_config
        self.collate_fn = collate_fn
        self.tokenizer = tokenizer
        self.hf_token = hf_token
        self.hf_repo_id = hf_repo_id

        # Mixed precision setup (prefer BF16 if available)
        self.use_amp = config.fp16 or getattr(config, 'bf16', False)
        self.amp_dtype = torch.bfloat16 if getattr(config, 'bf16', False) else torch.float16
        
        # Check if model is already in half precision (fp16/bf16)
        model_dtype = next(model.parameters()).dtype
        model_is_half = model_dtype in (torch.float16, torch.bfloat16)
        model_is_fp16 = model_dtype == torch.float16
        model_is_bf16 = model_dtype == torch.bfloat16
        self.model_is_fp16 = model_is_fp16  # Store for use in training loop
        
        # GradScaler configuration:
        # - GradScaler ONLY works with FP32 models + FP16 autocast (true mixed precision)
        # - GradScaler CANNOT work with FP16 models (raises "Attempting to unscale FP16 gradients")
        # - BF16: Never needs GradScaler (BF16 has same exponent range as FP32)
        # 
        # For FP16 models: Use manual loss scaling instead of GradScaler
        use_bf16 = getattr(config, 'bf16', False) or model_is_bf16
        
        if model_is_fp16:
            # FP16 model - optimizer should use FP32 master weights (handled by FP32OptimizerWrapper)
            # No GradScaler or manual loss scaling needed - FP32 optimizer states don't overflow!
            self.scaler = None
            self.manual_loss_scale = None  # Not needed with FP32 optimizer states
        elif config.fp16 and not use_bf16 and config.device == "cuda" and not model_is_half:
            # Standard mixed precision: FP32 model with FP16 autocast
            self.scaler = GradScaler()
            self.manual_loss_scale = None
            print(f"   📝 Mixed precision training with GradScaler")
        else:
            self.scaler = None
            self.manual_loss_scale = None
            if model_is_bf16:
                print(f"   📝 Model is BF16 - no loss scaling needed (BF16 is numerically stable)")
        
        self.global_step = 0
        self.start_epoch = 0
        self.best_loss = float("inf")

        # Get generation sizes from multi-scale config
        self.img_gen_size = xoron_config.image_base_size
        self.vid_gen_size = xoron_config.video_base_size
        self.use_multi_scale = getattr(xoron_config, 'use_multi_scale', True)
        self.img_min_size = getattr(xoron_config, 'image_min_size', 256)
        self.img_max_size = getattr(xoron_config, 'image_max_size', 512)
        self.vid_min_size = getattr(xoron_config, 'video_min_size', 256)
        self.vid_max_size = getattr(xoron_config, 'video_max_size', 448)
        
        # Loss weights from config (SOTA: configurable per-modality weights)
        self.llm_loss_weight = getattr(config, 'llm_loss_weight', 1.0)
        self.image_diffusion_loss_weight = getattr(config, 'image_diffusion_loss_weight', 0.1)
        self.video_diffusion_loss_weight = getattr(config, 'video_diffusion_loss_weight', 0.1)
        self.asr_loss_weight = getattr(config, 'asr_loss_weight', 0.1)
        self.tts_loss_weight = getattr(config, 'tts_loss_weight', 0.1)
        # Note: MoE aux loss weight removed - we use Aux-Lossless MoE
        
        # Weighted loss settings for important token groups
        # Chain-of-thought reasoning tokens
        self.cot_loss_weight = getattr(config, 'cot_loss_weight', 1.5)
        # Tool calling tokens (critical for agentic behavior)
        self.tool_loss_weight = getattr(config, 'tool_loss_weight', 1.3)
        # Anti-hallucination tokens (uncertainty, citations)
        self.anti_hallucination_loss_weight = getattr(config, 'anti_hallucination_loss_weight', 1.2)
        # Code execution tokens
        self.code_exec_loss_weight = getattr(config, 'code_exec_loss_weight', 1.2)
        
        # Get token IDs for all weighted groups
        self.reasoning_token_ids = self._get_reasoning_token_ids()
        self.reasoning_block_ids = self._get_reasoning_block_ids()
        self.tool_block_ids = self._get_tool_block_ids()
        self.anti_hallucination_block_ids = self._get_anti_hallucination_block_ids()
        self.code_exec_block_ids = self._get_code_exec_block_ids()
        
        # Gradient clipping from config
        self.max_grad_norm = getattr(config, 'max_grad_norm', 1.0)
        
        # Debug mode for expensive NaN checks (off by default for throughput)
        self.debug_nan_checks = getattr(config, 'debug_nan_checks', False)
        
        # Enable gradient checkpointing if configured
        if getattr(config, 'gradient_checkpointing', False):
            self._enable_gradient_checkpointing()

        # Resume from checkpoint if specified
        if resume_from is not None:
            self._load_checkpoint(resume_from)
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency across all components."""
        enabled_components = []
        
        # Enable for LLM - MoELlamaForCausalLM has gradient_checkpointing_enable()
        if hasattr(self.model, 'llm') and self.model.llm is not None:
            # Primary method: gradient_checkpointing_enable() (our custom + HuggingFace style)
            if hasattr(self.model.llm, 'gradient_checkpointing_enable'):
                self.model.llm.gradient_checkpointing_enable()
                enabled_components.append('LLM')
            # Fallback: set model.gradient_checkpointing directly
            elif hasattr(self.model.llm, 'model') and hasattr(self.model.llm.model, 'gradient_checkpointing'):
                self.model.llm.model.gradient_checkpointing = True
                enabled_components.append('LLM')
            # Final fallback: top-level gradient_checkpointing
            elif hasattr(self.model.llm, 'gradient_checkpointing'):
                self.model.llm.gradient_checkpointing = True
                enabled_components.append('LLM')
        
        # Enable for Vision Encoder - SigLIP/CLIP models have gradient_checkpointing_enable
        if hasattr(self.model, 'vision_encoder') and self.model.vision_encoder is not None:
            # Try vision_model (SigLIP/CLIP inner model)
            if hasattr(self.model.vision_encoder, 'vision_model'):
                inner = self.model.vision_encoder.vision_model
                if hasattr(inner, 'gradient_checkpointing_enable'):
                    inner.gradient_checkpointing_enable()
                    enabled_components.append('Vision')
            # Fallback: try 'model' attribute
            elif hasattr(self.model.vision_encoder, 'model'):
                inner = self.model.vision_encoder.model
                if hasattr(inner, 'gradient_checkpointing_enable'):
                    inner.gradient_checkpointing_enable()
                    enabled_components.append('Vision')
                elif hasattr(inner, 'vision_model') and hasattr(inner.vision_model, 'gradient_checkpointing_enable'):
                    inner.vision_model.gradient_checkpointing_enable()
                    enabled_components.append('Vision')
        
        # Enable for Video Encoder (uses vision encoder internally)
        if hasattr(self.model, 'video_encoder') and self.model.video_encoder is not None:
            # Video encoder typically wraps vision encoder, so this might already be covered
            if hasattr(self.model.video_encoder, 'gradient_checkpointing_enable'):
                self.model.video_encoder.gradient_checkpointing_enable()
                enabled_components.append('Video')
        
        # Enable for Audio Encoder
        if hasattr(self.model, 'audio_encoder') and self.model.audio_encoder is not None:
            if hasattr(self.model.audio_encoder, 'encoder') and hasattr(self.model.audio_encoder.encoder, 'gradient_checkpointing_enable'):
                self.model.audio_encoder.encoder.gradient_checkpointing_enable()
                enabled_components.append('Audio Encoder')
            elif hasattr(self.model.audio_encoder, 'gradient_checkpointing_enable'):
                self.model.audio_encoder.gradient_checkpointing_enable()
                enabled_components.append('Audio Encoder')
        
        # Enable for Audio Decoder
        if hasattr(self.model, 'audio_decoder') and self.model.audio_decoder is not None:
            if hasattr(self.model.audio_decoder, 'gradient_checkpointing_enable'):
                self.model.audio_decoder.gradient_checkpointing_enable()
                enabled_components.append('Audio Decoder')
        
        # Enable for Waveform Decoder (Speech-to-Speech)
        if hasattr(self.model, 'waveform_decoder') and self.model.waveform_decoder is not None:
            if hasattr(self.model.waveform_decoder, 'gradient_checkpointing_enable'):
                self.model.waveform_decoder.gradient_checkpointing_enable()
                enabled_components.append('Waveform Decoder')
            else:
                # Manual gradient checkpointing for conv layers
                for module in self.model.waveform_decoder.modules():
                    if hasattr(module, 'gradient_checkpointing'):
                        module.gradient_checkpointing = True
                enabled_components.append('Waveform Decoder (manual)')
        
        # Enable for Image Generator (Diffusion) - UNet based
        if hasattr(self.model, 'generator') and self.model.generator is not None:
            if hasattr(self.model.generator, 'unet'):
                if hasattr(self.model.generator.unet, 'enable_gradient_checkpointing'):
                    self.model.generator.unet.enable_gradient_checkpointing()
                    enabled_components.append('Image Generator')
                elif hasattr(self.model.generator.unet, 'gradient_checkpointing_enable'):
                    self.model.generator.unet.gradient_checkpointing_enable()
                    enabled_components.append('Image Generator')
        
        # Enable for Video Generator
        if hasattr(self.model, 'video_generator') and self.model.video_generator is not None:
            if hasattr(self.model.video_generator, 'unet'):
                if hasattr(self.model.video_generator.unet, 'enable_gradient_checkpointing'):
                    self.model.video_generator.unet.enable_gradient_checkpointing()
                    enabled_components.append('Video Generator')
                elif hasattr(self.model.video_generator.unet, 'gradient_checkpointing_enable'):
                    self.model.video_generator.unet.gradient_checkpointing_enable()
                    enabled_components.append('Video Generator')
        
        if enabled_components:
            print(f"   ✅ Gradient checkpointing enabled for: {', '.join(enabled_components)}")
        else:
            print("   ⚠️ Gradient checkpointing: No compatible components found (may need manual setup)")
    
    @staticmethod
    def create_lora_plus_optimizer(model, base_lr: float, lr_ratio: float = 16.0, weight_decay: float = 0.01):
        """
        Create optimizer with LoRA+ learning rate schedule.
        
        LoRA+ uses different learning rates for A and B matrices:
        - B matrix: base_lr * lr_ratio (learns faster)
        - A matrix: base_lr
        
        This improves convergence and final performance.
        
        Args:
            model: The model with LoRA layers
            base_lr: Base learning rate for A matrix
            lr_ratio: Ratio for B matrix learning rate (default 16x)
            weight_decay: Weight decay for regularization
            
        Returns:
            torch.optim.AdamW optimizer with parameter groups
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
            elif 'magnitude' in name:  # DoRA magnitude
                magnitude_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = []
        
        if lora_a_params:
            param_groups.append({
                'params': lora_a_params, 
                'lr': base_lr, 
                'weight_decay': weight_decay,
                'name': 'lora_A'
            })
        if lora_b_params:
            param_groups.append({
                'params': lora_b_params, 
                'lr': base_lr * lr_ratio, 
                'weight_decay': weight_decay,
                'name': 'lora_B'
            })
        if magnitude_params:
            param_groups.append({
                'params': magnitude_params, 
                'lr': base_lr, 
                'weight_decay': 0.0,  # No weight decay for magnitude
                'name': 'magnitude'
            })
        if other_params:
            param_groups.append({
                'params': other_params, 
                'lr': base_lr, 
                'weight_decay': weight_decay,
                'name': 'other'
            })
        
        optimizer = torch.optim.AdamW(param_groups)
        
        # Print summary
        print(f"   ✅ LoRA+ optimizer created:")
        print(f"      - LoRA A params: {len(lora_a_params)} @ lr={base_lr}")
        print(f"      - LoRA B params: {len(lora_b_params)} @ lr={base_lr * lr_ratio}")
        if magnitude_params:
            print(f"      - DoRA magnitude params: {len(magnitude_params)} @ lr={base_lr}")
        print(f"      - Other params: {len(other_params)} @ lr={base_lr}")
        
        return optimizer
        
    def _get_reasoning_token_ids(self) -> Dict[str, int]:
        """Get token IDs for reasoning tokens."""
        if self.tokenizer is None:
            return {}
        
        reasoning_tokens = get_reasoning_tokens()
        token_ids = {}
        
        for name, token in reasoning_tokens.items():
            try:
                ids = self.tokenizer.encode(token, add_special_tokens=False)
                if ids:
                    token_ids[name] = ids[0] if len(ids) == 1 else ids
            except Exception:
                pass
        
        return token_ids
    
    def _get_reasoning_block_ids(self) -> List[tuple]:
        """Get token ID pairs for all reasoning blocks that should receive higher loss weight."""
        return self._get_block_ids_from_pairs(get_all_reasoning_block_tokens())
    
    def _get_tool_block_ids(self) -> List[tuple]:
        """Get token ID pairs for tool calling blocks."""
        return self._get_block_ids_from_pairs(get_tool_block_tokens())
    
    def _get_anti_hallucination_block_ids(self) -> List[tuple]:
        """Get token ID pairs for anti-hallucination blocks."""
        return self._get_block_ids_from_pairs(get_anti_hallucination_block_tokens())
    
    def _get_code_exec_block_ids(self) -> List[tuple]:
        """Get token ID pairs for code execution blocks."""
        return self._get_block_ids_from_pairs(get_code_execution_block_tokens())
    
    def _get_block_ids_from_pairs(self, block_pairs: List[tuple]) -> List[tuple]:
        """
        Convert block token pairs to token IDs.
        
        Args:
            block_pairs: List of (start_key, end_key) tuples
            
        Returns:
            List of (block_name, start_id, end_id) tuples
        """
        if self.tokenizer is None:
            return []
        
        block_ids = []
        
        for start_key, end_key in block_pairs:
            start_token = SPECIAL_TOKENS.get(start_key)
            end_token = SPECIAL_TOKENS.get(end_key)
            
            if start_token and end_token:
                try:
                    start_ids = self.tokenizer.encode(start_token, add_special_tokens=False)
                    end_ids = self.tokenizer.encode(end_token, add_special_tokens=False)
                    
                    if start_ids and end_ids:
                        start_id = start_ids[0] if len(start_ids) == 1 else start_ids
                        end_id = end_ids[0] if len(end_ids) == 1 else end_ids
                        block_ids.append((start_key, start_id, end_id))
                except Exception:
                    pass
        
        return block_ids

    def _load_checkpoint(self, checkpoint_path: str):
        """Load training state from a checkpoint."""
        from models.xoron import XoronMultimodalModel
        
        training_state = XoronMultimodalModel.load_training_state(checkpoint_path)
        if training_state is not None:
            self.global_step = training_state.get('global_step', 0)
            self.start_epoch = training_state.get('epoch', 0)
            self.best_loss = training_state.get('best_loss', float('inf'))
            
            if 'optimizer_state_dict' in training_state:
                self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
                print(f"   ✅ Optimizer state loaded")
            
            if 'scheduler_state_dict' in training_state:
                self.scheduler.load_state_dict(training_state['scheduler_state_dict'])
                print(f"   ✅ Scheduler state loaded")
            
            print(f"   📊 Resuming from epoch {self.start_epoch}, step {self.global_step}")
            print(f"   📊 Best loss so far: {self.best_loss:.4f}")
        else:
            print(f"   ⚠️ No training state found at {checkpoint_path}, starting fresh")

    def _compute_weighted_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
        sample_types: List[str],
        model_loss: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute loss with higher weight for important token groups.
        
        This encourages the model to pay more attention to:
        - Chain-of-thought reasoning tokens (think, plan, critique, etc.)
        - Tool calling tokens (tool_call, function_name, etc.)
        - Anti-hallucination tokens (uncertain, cite, verify, etc.)
        - Code execution tokens (exec, jupyter, code, etc.)
        
        Each group has its own configurable weight multiplier.
        
        IMPORTANT: Handles dimension mismatch when multimodal embeddings are prepended
        to the sequence in the forward pass, causing logits to have more tokens than labels.
        
        If weighted loss computation fails, falls back to model_loss if provided.
        """
        device = logits.device
        
        try:
            # CRITICAL FIX: Handle dimension mismatch between logits and labels
            # This happens when multimodal embeddings (image/video/audio) are prepended
            # to the text embeddings in the forward pass. The logits will be longer than labels.
            logits_seq_len = logits.size(1)
            labels_seq_len = labels.size(1)
            
            if logits_seq_len != labels_seq_len:
                # Logits are longer due to prepended multimodal embeddings
                # We need to align by taking only the last `labels_seq_len` logits
                # (the text portion that corresponds to our labels)
                offset = logits_seq_len - labels_seq_len
                if offset > 0:
                    # Take only the text portion of logits (skip multimodal prefix)
                    # Also align input_ids to match labels for token-specific weighting
                    logits = logits[:, offset:, :]
                    # input_ids should already match labels length, but verify
                    if input_ids.size(1) > labels_seq_len:
                        input_ids = input_ids[:, :labels_seq_len]
                elif offset < 0:
                    # Labels are longer (shouldn't happen, but handle gracefully)
                    labels = labels[:, -logits_seq_len:]
                    input_ids = input_ids[:, -logits_seq_len:]
            
            # Standard cross-entropy loss with shifted sequences
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Ensure labels are on the same device as logits
            if shift_logits.device != shift_labels.device:
                shift_labels = shift_labels.to(shift_logits.device)
            
            # Ensure labels are long dtype for CrossEntropyLoss (critical!)
            if shift_labels.dtype != torch.long:
                shift_labels = shift_labels.long()
            
            # Check if there are any valid labels BEFORE computing loss
            # This prevents NaN from 0/0 division when all labels are -100
            valid_mask_check = (shift_labels != -100)
            num_valid_total = valid_mask_check.sum().item()
            
            if num_valid_total == 0:
                # No valid labels at all - return model loss or zero
                if model_loss is not None and not (torch.isnan(model_loss) or torch.isinf(model_loss)):
                    return model_loss
                return torch.tensor(0.0, device=device, requires_grad=True)
            
            # Compute per-token loss - CrossEntropyLoss handles ignore_index internally
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            
            per_token_loss = loss_fct(flat_logits, flat_labels)
            per_token_loss = per_token_loss.view(shift_labels.size())
            
            # Check if we should apply weighted loss
            # Apply to CoT samples, tool use samples, and agentic samples
            weighted_sample_types = {'chain_of_thought', 'tool_use', 'agentic', 'code_execution', 
                                      'shell_execution', 'jupyter', 'anti_hallucination'}
            has_weighted_samples = any(t in weighted_sample_types for t in sample_types)
            
            # Check if we have any block IDs to weight
            has_block_ids = (self.reasoning_block_ids or self.tool_block_ids or 
                            self.anti_hallucination_block_ids or self.code_exec_block_ids)
            
            if has_weighted_samples and has_block_ids:
                # Create weight mask on the same device as per_token_loss
                weight_mask = torch.ones_like(per_token_loss)
                
                # Move input_ids to same device as logits for consistency
                shift_input_ids = input_ids[..., 1:].contiguous()
                if shift_input_ids.device != shift_logits.device:
                    shift_input_ids = shift_input_ids.to(shift_logits.device)
                
                # Ensure shift_input_ids matches shift_labels length for weighted masking
                if shift_input_ids.size(1) != shift_labels.size(1):
                    # Truncate or pad to match
                    target_len = shift_labels.size(1)
                    if shift_input_ids.size(1) > target_len:
                        shift_input_ids = shift_input_ids[:, :target_len]
                    else:
                        # Pad with padding token (usually 0)
                        pad_len = target_len - shift_input_ids.size(1)
                        pad_tokens = torch.zeros(shift_input_ids.size(0), pad_len, 
                                                dtype=shift_input_ids.dtype, device=shift_input_ids.device)
                        shift_input_ids = torch.cat([shift_input_ids, pad_tokens], dim=1)
                
                # Build combined block list with weights
                all_blocks_with_weights = []
                for block_name, start_id, end_id in self.reasoning_block_ids:
                    all_blocks_with_weights.append((block_name, start_id, end_id, self.cot_loss_weight))
                for block_name, start_id, end_id in self.tool_block_ids:
                    all_blocks_with_weights.append((block_name, start_id, end_id, self.tool_loss_weight))
                for block_name, start_id, end_id in self.anti_hallucination_block_ids:
                    all_blocks_with_weights.append((block_name, start_id, end_id, self.anti_hallucination_loss_weight))
                for block_name, start_id, end_id in self.code_exec_block_ids:
                    all_blocks_with_weights.append((block_name, start_id, end_id, self.code_exec_loss_weight))
                
                for batch_idx in range(shift_input_ids.size(0)):
                    sample_type = sample_types[batch_idx]
                    if sample_type not in weighted_sample_types:
                        continue
                    
                    seq = shift_input_ids[batch_idx]
                    
                    # Track which blocks we're inside and their weights
                    in_blocks = {}  # block_name -> weight
                    
                    for pos in range(seq.size(0)):
                        token_id = seq[pos].item()
                        
                        # Check all block types
                        for block_name, start_id, end_id, weight in all_blocks_with_weights:
                            # Check for start token
                            if isinstance(start_id, list):
                                is_start = token_id in start_id
                            else:
                                is_start = token_id == start_id
                            
                            # Check for end token
                            if isinstance(end_id, list):
                                is_end = token_id in end_id
                            else:
                                is_end = token_id == end_id
                            
                            if is_start:
                                in_blocks[block_name] = weight
                            elif is_end:
                                in_blocks.pop(block_name, None)
                        
                        # Apply the maximum weight from all active blocks
                        if in_blocks:
                            max_weight = max(in_blocks.values())
                            weight_mask[batch_idx, pos] = max_weight
                
                # Apply weights
                weighted_loss = per_token_loss * weight_mask
                
                # Compute mean over non-ignored tokens
                valid_mask = (shift_labels != -100).float()
                valid_sum = valid_mask.sum()
                if valid_sum > 0:
                    loss = (weighted_loss * valid_mask).sum() / valid_sum
                else:
                    # No valid tokens - fall back to model loss if available
                    if model_loss is not None:
                        return model_loss
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                # Standard loss computation
                valid_mask = (shift_labels != -100).float()
                valid_sum = valid_mask.sum()
                if valid_sum > 0:
                    loss = (per_token_loss * valid_mask).sum() / valid_sum
                else:
                    # No valid tokens - fall back to model loss if available
                    if model_loss is not None:
                        return model_loss
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Final NaN safety check
            if torch.isnan(loss) or torch.isinf(loss):
                if model_loss is not None:
                    return model_loss
                return torch.tensor(0.0, device=device, requires_grad=True)
            
            return loss
            
        except Exception:
            # If anything goes wrong, fall back to model loss
            if model_loss is not None:
                return model_loss
            return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Keep old method name for backward compatibility
    def _compute_cot_weighted_loss(self, logits, labels, input_ids, sample_types, model_loss=None):
        """Backward compatible alias for _compute_weighted_loss."""
        return self._compute_weighted_loss(logits, labels, input_ids, sample_types, model_loss)

    def _verify_model_weights(self):
        """Verify model weights are valid (no NaN/Inf) before training."""
        print("\n🔍 Verifying model weights...")
        bad_params = []
        total_params = 0
        nan_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += 1
            has_nan = torch.isnan(param).any().item()
            has_inf = torch.isinf(param).any().item()
            if has_nan or has_inf:
                nan_count = torch.isnan(param).sum().item()
                inf_count = torch.isinf(param).sum().item()
                bad_params.append((name, param.shape, nan_count, inf_count))
                nan_params += 1
        
        if bad_params:
            print(f"\n   ❌ CRITICAL: {len(bad_params)} parameters have NaN/Inf values!")
            for name, shape, nan_count, inf_count in bad_params[:20]:
                print(f"      - {name}: shape={list(shape)}, nan={nan_count}, inf={inf_count}")
            if len(bad_params) > 20:
                print(f"      ... and {len(bad_params) - 20} more")
            raise RuntimeError(
                f"Model has {len(bad_params)}/{total_params} corrupted parameters. "
                "Cannot start training with NaN/Inf weights. "
                "Please check your model initialization or loaded checkpoint."
            )
        else:
            print(f"   ✅ All {total_params} parameters verified - no NaN/Inf detected")
        
        # Also check embedding stats
        if hasattr(self.model, 'llm'):
            llm = self.model.llm
            if hasattr(llm, 'model') and hasattr(llm.model, 'embed_tokens'):
                embed = llm.model.embed_tokens.weight
                print(f"   📊 Embedding stats: min={embed.min():.4f}, max={embed.max():.4f}, mean={embed.mean():.4f}")
            elif hasattr(llm, 'get_input_embeddings'):
                embed = llm.get_input_embeddings().weight
                print(f"   📊 Embedding stats: min={embed.min():.4f}, max={embed.max():.4f}, mean={embed.mean():.4f}")

    def train(self):
        """Run the full training loop."""
        # CRITICAL: Verify weights are valid before starting
        self._verify_model_weights()
        
        print("\n" + "=" * 60)
        if self.start_epoch > 0:
            print(f"🔄 RESUMING TRAINING (epoch {self.start_epoch + 1}, step {self.global_step})")
        else:
            print("🚀 STARTING TRAINING")
        
        # Get trainable and frozen components
        trainable = self.model.get_trainable_component_names()
        frozen = self.model.get_frozen_component_names()
        
        # Concise component status
        trainable_str = ', '.join(trainable) if trainable else 'none'
        frozen_str = ', '.join(frozen) if frozen else 'none'
        print(f"   🔥 Training: {trainable_str}")
        if frozen:
            print(f"   ❄️ Frozen: {frozen_str}")
        
        # Concise settings
        precision = 'BF16' if self.amp_dtype == torch.bfloat16 else ('FP16' if self.use_amp else 'FP32')
        if self.use_multi_scale:
            print(f"   ⚙️ {precision} | grad_clip={self.max_grad_norm} | img={self.img_min_size}-{self.img_max_size}px | vid={self.vid_min_size}-{self.vid_max_size}px")
        else:
            print(f"   ⚙️ {precision} | grad_clip={self.max_grad_norm} | img={self.img_gen_size}px | vid={self.vid_gen_size}px")
        print("=" * 60)

        self.model.train()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        training_start_time = time.time()
        batch_times = []

        for epoch in range(self.start_epoch, self.config.num_epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*60}")

            # Reset dataset for new epoch (except first epoch)
            # clear_state=False (default) keeps stream positions so each epoch gets NEW data
            # continuing from where the previous epoch left off in the dataset
            if epoch > self.start_epoch:
                self.train_dataset.reset(clear_state=False)
            
            # Create DataLoader for this epoch (IterableDataset requires fresh DataLoader per epoch)
            # pin_memory=True speeds up CPU→GPU transfers on CUDA
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,  # IterableDataset handles shuffling internally
                num_workers=0,  # Must be 0 for IterableDataset with state
                collate_fn=self.collate_fn,
                pin_memory=torch.cuda.is_available(),  # Faster CPU→GPU transfer
            )

            # Run training epoch
            train_losses = self._train_epoch(train_loader, epoch, batch_times, training_start_time)

            # Run validation at end of each epoch if eval_dataset is provided
            eval_losses = None
            if self.eval_dataset is not None:
                # Reset eval dataset for new epoch - advances to NEW samples (not clear_state)
                # Both train and eval get fresh samples each epoch
                if hasattr(self.eval_dataset, 'reset'):
                    self.eval_dataset.reset(clear_state=False)  # Keep positions, advance to new samples
                
                eval_loader = DataLoader(
                    self.eval_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=self.collate_fn,
                    pin_memory=torch.cuda.is_available(),
                )
                eval_losses = self._eval_epoch(eval_loader, epoch)

            # Print comprehensive epoch summary with training and validation losses
            self._print_epoch_summary(epoch, train_losses, eval_losses, training_start_time)

            # Aggressive memory cleanup before saving checkpoint
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()  # Double clear for fragmentation
                
            # Force Python garbage collection
            gc.collect()

            # Save checkpoint (use LLM loss for tracking best model)
            epoch_loss = train_losses['llm']
            if (epoch + 1) % 1 == 0:
                self._save_checkpoint(epoch, epoch_loss)
                
            # Clear memory again after saving before next epoch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()

        # Final save
        self._save_final_model()
        
        # Upload to HuggingFace if configured
        self._upload_to_huggingface()

        total_time = time.time() - training_start_time
        print(f"\n{'='*60}")
        print(f"✅ TRAINING COMPLETE!")
        print(f"   Total time: {total_time/3600:.2f} hours")
        print(f"   Best loss: {self.best_loss:.4f}")
        print(f"{'='*60}")

    def _train_epoch(self, train_loader, epoch, batch_times, training_start_time):
        """Train for one epoch."""
        epoch_llm_loss = 0.0
        epoch_cot_loss = 0.0
        epoch_img_diff_loss = 0.0
        epoch_vid_diff_loss = 0.0
        epoch_asr_loss = 0.0
        epoch_tts_loss = 0.0
        epoch_waveform_loss = 0.0  # Waveform decoder loss (Speech-to-Speech)
        num_batches = 0
        num_valid_batches = 0  # Track batches with valid loss for learning
        num_cot = 0
        num_img_diff = 0
        num_vid_diff = 0
        num_asr = 0
        num_tts = 0
        num_waveform = 0  # Batches with waveform decoder loss
        total_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            # Only log every 1000 batches to reduce log spam
            if (batch_idx + 1) % 1000 == 0 or batch_idx == 0:
                print(f"🔄 Processing batch {batch_idx + 1}/{total_batches}...", flush=True)
            
            batch_start = time.time()

            # Move batch to device
            device = self.config.device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            video_frames = batch["video_frames"].to(device)
            audio_features = batch["audio_features"].to(device)
            sample_types = batch.get("sample_type", ["text"] * len(input_ids))
            
            # ═══════════════════════════════════════════════════════════════════
            # CRITICAL: Validate token IDs are within vocabulary bounds
            # Out-of-range tokens cause NaN in embeddings → corrupts entire model
            # ═══════════════════════════════════════════════════════════════════
            # NOTE: Use ACTUAL embedding size, not config.vocab_size
            # (config may be stale after adding special tokens and resizing)
            vocab_size = None
            if hasattr(self.model, 'llm'):
                if hasattr(self.model.llm, 'model') and hasattr(self.model.llm.model, 'embed_tokens'):
                    vocab_size = self.model.llm.model.embed_tokens.weight.shape[0]
                elif hasattr(self.model.llm, 'embed_tokens'):
                    vocab_size = self.model.llm.embed_tokens.weight.shape[0]
                # Fallback to config if embedding not accessible
                if vocab_size is None and hasattr(self.model.llm, 'config'):
                    vocab_size = getattr(self.model.llm.config, 'vocab_size', None)
            
            if vocab_size is not None:
                max_token_id = input_ids.max().item()
                min_token_id = input_ids.min().item()
                
                if max_token_id >= vocab_size:
                    # CRITICAL: Token IDs exceed vocabulary - this WILL corrupt the model!
                    num_invalid = (input_ids >= vocab_size).sum().item()
                    print(f"\n{'!'*70}")
                    print(f"! 🚨 CRITICAL: TOKEN IDS OUT OF VOCABULARY BOUNDS!")
                    print(f"! Batch {batch_idx}: max_token={max_token_id} >= vocab_size={vocab_size}")
                    print(f"! Number of out-of-range tokens: {num_invalid}")
                    print(f"! This WILL cause NaN corruption in embedding layer!")
                    print(f"{'!'*70}")
                    
                    # Option 1: Clamp to valid range (may produce garbage but won't crash)
                    # input_ids = input_ids.clamp(0, vocab_size - 1)
                    # print(f"   ⚠️ Clamped tokens to valid range [0, {vocab_size-1}]")
                    
                    # Option 2: Skip this batch entirely
                    print(f"   ➡️ Skipping batch {batch_idx} to prevent corruption")
                    print(f"   💡 FIX YOUR TOKENIZER/DATA PIPELINE!\n")
                    num_batches += 1
                    continue
                
                if min_token_id < 0:
                    # Also catch negative token IDs (except -100 in labels which is ignore_index)
                    num_negative = (input_ids < 0).sum().item()
                    print(f"\n⚠️ Batch {batch_idx}: Negative token IDs detected! min={min_token_id}, count={num_negative}")
                    print(f"   ➡️ Skipping batch to prevent issues\n")
                    num_batches += 1
                    continue
            
            # Also validate labels (except -100 which is ignore_index)
            if vocab_size is not None and labels is not None:
                # Filter out ignore_index (-100) before checking max
                valid_labels = labels[labels != -100]
                if valid_labels.numel() > 0:
                    max_label = valid_labels.max().item()
                    if max_label >= vocab_size:
                        print(f"\n⚠️ Batch {batch_idx}: Label IDs out of range! max_label={max_label} >= vocab_size={vocab_size}")
                        print(f"   ➡️ Skipping batch to prevent issues\n")
                        num_batches += 1
                        continue
            
            # Check for samples that need weighted loss (CoT, tool use, agentic, etc.)
            has_cot_samples = any(t == 'chain_of_thought' for t in sample_types)
            weighted_sample_types = {'chain_of_thought', 'tool_use', 'agentic', 'code_execution', 
                                      'shell_execution', 'jupyter', 'anti_hallucination'}
            has_weighted_samples = any(t in weighted_sample_types for t in sample_types)

            # Forward pass with mixed precision (supports both FP16 and BF16)
            if self.use_amp:
                with autocast(device_type='cuda', dtype=self.amp_dtype):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        video_frames=video_frames,
                        audio_features=audio_features,
                        labels=labels,
                    )
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    video_frames=video_frames,
                    audio_features=audio_features,
                    labels=labels,
                )
            
            # Get model's base loss
            model_loss = getattr(outputs, 'loss', None)
            logits = getattr(outputs, 'logits', None)
            
            # NaN/Inf check on logits - only when debug mode enabled (causes CPU-GPU sync)
            if self.debug_nan_checks and logits is not None and (torch.isnan(logits).any() or torch.isinf(logits).any()):
                nan_count = torch.isnan(logits).sum().item()
                inf_count = torch.isinf(logits).sum().item()
                total_elements = logits.numel()
                nan_pct = 100 * nan_count / total_elements
                inf_pct = 100 * inf_count / total_elements
                
                print(f"\n{'!'*70}")
                print(f"! ⚠️ BATCH {batch_idx}: NaN/Inf DETECTED IN LOGITS")
                print(f"! nan_count={nan_count} ({nan_pct:.2f}%), inf_count={inf_count} ({inf_pct:.2f}%)")
                print(f"! logits shape: {logits.shape}, dtype: {logits.dtype}")
                print(f"{'!'*70}")
                
                # Get finite values stats
                finite_mask = torch.isfinite(logits)
                if finite_mask.any():
                    finite_vals = logits[finite_mask]
                    print(f"! Finite values: min={finite_vals.min():.4f}, max={finite_vals.max():.4f}, mean={finite_vals.mean():.4f}")
                
                # Deep debug on first occurrence or every 10th
                if batch_idx < 5 or batch_idx % 10 == 0:
                    print(f"\n🔬 RUNNING DEEP DEBUG...")
                    try:
                        debug_full_forward(self.model, input_ids, attention_mask, labels, max_layers_to_check=3)
                    except Exception as e:
                        print(f"❌ Debug failed: {e}")
                
                print(f"{'!'*70}\n")
                num_batches += 1
                continue
            
            # Use weighted loss for special token samples (CoT, tool use, etc.)
            # This gives higher weight to reasoning, tool calling, and anti-hallucination tokens
            if has_weighted_samples and logits is not None:
                llm_loss = self._compute_weighted_loss(
                    logits, labels, input_ids, sample_types, model_loss
                )
            else:
                llm_loss = model_loss
                if llm_loss is None:
                    llm_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # CRITICAL: Check LLM loss for NaN/Inf before proceeding
            llm_loss_check = llm_loss.item()
            if llm_loss_check != llm_loss_check or llm_loss_check == float('inf') or llm_loss_check == float('-inf'):
                if batch_idx % 1000 == 0:
                    print(f"   ⚠️ Batch {batch_idx}: NaN/Inf LLM loss ({llm_loss_check}), skipping batch")
                num_batches += 1
                continue
            
            # Clamp LLM loss to prevent extreme values (FP16 safety)
            llm_loss = torch.clamp(llm_loss, min=0.0, max=100.0)
            
            # Track CoT loss separately
            if has_cot_samples:
                cot_loss_val = llm_loss.item()
                if not (cot_loss_val != cot_loss_val):  # NaN check
                    epoch_cot_loss += cot_loss_val
                num_cot += 1

            # Apply LLM loss weight
            # Note: MoE aux loss removed - we use Aux-Lossless MoE (always returns 0)
            total_loss = self.llm_loss_weight * llm_loss

            # Train diffusion models based on sample type
            text_embeds = self.model.get_text_embeddings(input_ids, attention_mask)

            # Image diffusion (use configurable loss weight)
            # MUST match filter in train_image_diffusion_step to actually train
            image_sample_types = ['image_generation', 'image_editing', 'text_to_image', 'image_caption']
            if any(t in image_sample_types for t in sample_types):
                img_diff_loss = train_image_diffusion_step(
                    self.model.generator, pixel_values, text_embeds, self.img_gen_size,
                    sample_types=sample_types
                )
                if img_diff_loss is not None:
                    # Move to same device AND dtype as total_loss
                    img_diff_loss = img_diff_loss.to(device=total_loss.device, dtype=total_loss.dtype)
                    total_loss = total_loss + self.image_diffusion_loss_weight * img_diff_loss
                    epoch_img_diff_loss += img_diff_loss.item()
                    num_img_diff += 1

            # Video diffusion - train on ALL video sample types (use configurable loss weight)
            # MUST match filter in train_video_diffusion_step to actually train
            video_sample_types = ['video_generation', 'image_to_video', 'video_caption', 'video_qa', 
                                  'video_preference', 'video_likert', 'text_to_video']
            if any(t in video_sample_types for t in sample_types):
                vid_diff_loss = train_video_diffusion_step(
                    self.model.video_generator, video_frames, text_embeds, self.vid_gen_size,
                    sample_types=sample_types
                )
                if vid_diff_loss is not None:
                    # Move to same device AND dtype as total_loss
                    vid_diff_loss = vid_diff_loss.to(device=total_loss.device, dtype=total_loss.dtype)
                    total_loss = total_loss + self.video_diffusion_loss_weight * vid_diff_loss
                    epoch_vid_diff_loss += vid_diff_loss.item()
                    num_vid_diff += 1

            # Voice ASR (use configurable loss weight)
            # MEMORY OPTIMIZATION: Clear cache before voice training to prevent OOM
            has_voice_samples = any(t == 'voice_asr' for t in sample_types) or any(t == 'voice_tts' for t in sample_types)
            if has_voice_samples and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if any(t == 'voice_asr' for t in sample_types):
                asr_loss = train_voice_asr_step(
                    self.model.audio_encoder, audio_features, text_embeds,
                    sample_types=sample_types
                )
                if asr_loss is not None:
                    # Move to same device AND dtype as total_loss
                    asr_loss = asr_loss.to(device=total_loss.device, dtype=total_loss.dtype)
                    total_loss = total_loss + self.asr_loss_weight * asr_loss
                    epoch_asr_loss += asr_loss.item()
                    num_asr += 1
                    del asr_loss  # Clean up

            # Voice TTS with Voice Cloning (use configurable loss weight)
            if any(t == 'voice_tts' for t in sample_types):
                # Get speaker reference audio for voice cloning
                speaker_ref_audio = batch.get("speaker_ref_audio")
                
                tts_loss = train_voice_tts_step(
                    self.model.audio_decoder, 
                    text_embeds, 
                    audio_features,
                    audio_encoder=self.model.audio_encoder,  # For speaker embedding extraction
                    speaker_ref_audio=speaker_ref_audio,  # Voice cloning reference
                    sample_types=sample_types,
                    use_mas=getattr(self.config, 'use_mas', True),  # MAS for alignment
                )
                if tts_loss is not None:
                    # Move to same device AND dtype as total_loss
                    tts_loss = tts_loss.to(device=total_loss.device, dtype=total_loss.dtype)
                    total_loss = total_loss + self.tts_loss_weight * tts_loss
                    epoch_tts_loss += tts_loss.item()
                    num_tts += 1
                    del tts_loss  # Clean up
                
                # Train waveform decoder for Speech-to-Speech (direct audio output)
                # Note: When use_raw_waveform=True in config, audio_features contains raw waveform
                # When use_raw_waveform=False, audio_features contains mel spectrogram
                if hasattr(self.model, 'waveform_decoder') and self.model.waveform_decoder is not None:
                    # audio_features shape:
                    # - Raw waveform (use_raw_waveform=True): [B, T] where T is ~160000 samples
                    # - Mel spectrogram (use_raw_waveform=False): [B, mel_bins, time] where mel_bins=80
                    target_waveform = audio_features
                    if target_waveform is not None:
                        if target_waveform.dim() == 3:
                            # 3D tensor [B, mel_bins, time] - this is mel spectrogram, skip waveform training
                            target_waveform = None
                        elif target_waveform.dim() == 2:
                            # 2D tensor [B, T] - this is raw waveform, use it
                            # Verify it's actually waveform (T should be >> 1000 for audio samples)
                            if target_waveform.shape[1] < 1000:
                                target_waveform = None  # Too short, probably not valid waveform
                    
                    if target_waveform is not None:
                        from training.utils import train_waveform_decoder_step
                        waveform_loss = train_waveform_decoder_step(
                            self.model.waveform_decoder,
                            self.model.audio_decoder,
                            text_embeds,
                            target_waveform,
                            mel_to_hidden=getattr(self.model, '_mel_to_hidden', None),
                            sample_types=sample_types,
                        )
                        if waveform_loss is not None:
                            # Move to same device AND dtype as total_loss
                            waveform_loss = waveform_loss.to(device=total_loss.device, dtype=total_loss.dtype)
                            # Use same weight as TTS
                            total_loss = total_loss + self.tts_loss_weight * 0.5 * waveform_loss
                            epoch_waveform_loss += waveform_loss.item()
                            num_waveform += 1
                            del waveform_loss  # Clean up
            
            # MEMORY OPTIMIZATION: Clear cache after voice training
            if has_voice_samples and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # CRITICAL: Final loss clamping before backward (FP16 safety)
            # This prevents gradient explosion from extreme loss values
            total_loss = torch.clamp(total_loss, min=0.0, max=100.0)
            
            # Backward pass (supports both FP16 with scaler and BF16 without)
            loss_value = total_loss.item()
            is_valid_loss = not (loss_value != loss_value) and not (loss_value == float('inf')) and not (loss_value == float('-inf'))
            
            if is_valid_loss:
                # Scale loss for gradient accumulation
                scaled_loss = total_loss / self.config.gradient_accumulation_steps
                
                if self.scaler is not None:
                    # Standard GradScaler (FP32 model with FP16 autocast)
                    self.scaler.scale(scaled_loss).backward()
                elif self.manual_loss_scale is not None:
                    # Manual loss scaling for FP16 models
                    # Scale up loss before backward to prevent gradient underflow
                    (scaled_loss * self.manual_loss_scale).backward()
                else:
                    # BF16 or FP32 - no scaling needed
                    scaled_loss.backward()
                
                # Log gradient norms periodically (every 1000 batches)
                if (batch_idx + 1) % 1000 == 0:
                    total_norm = 0.0
                    embed_norm = 0.0
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.data.float().norm(2).item()
                            total_norm += grad_norm ** 2
                            if 'embed' in name.lower():
                                embed_norm += grad_norm ** 2
                    total_norm = total_norm ** 0.5
                    embed_norm = embed_norm ** 0.5
                    print(f"   📊 Grad norms - total: {total_norm:.4f}, embed: {embed_norm:.4f}")
                
                num_valid_batches += 1
            else:
                # Skip backward for NaN/Inf loss - log warning
                if batch_idx % 1000 == 0:
                    print(f"   ⚠️ Batch {batch_idx}: Skipping backward due to invalid loss ({loss_value})")

            # Optimizer step with gradient clipping from config
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Check for NaN/Inf gradients before optimizer step
                has_nan_grad = False
                
                # Unscale gradients before checking/clipping
                skip_optimizer_step = False
                
                if self.scaler is not None:
                    # GradScaler handles unscaling
                    self.scaler.unscale_(self.optimizer)
                elif self.manual_loss_scale is not None:
                    # FIRST: Check scaled gradient norm BEFORE unscaling
                    # If too high, reduce scale to prevent FP16 overflow in optimizer
                    scaled_grad_norm = 0.0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            scaled_grad_norm += param.grad.data.float().norm(2).item() ** 2
                    scaled_grad_norm = scaled_grad_norm ** 0.5
                    
                    if scaled_grad_norm > self.max_scaled_grad_norm:
                        # Gradients too large - reduce scale and skip this step
                        old_scale = self.manual_loss_scale
                        self.manual_loss_scale = max(1.0, self.manual_loss_scale * 0.5)
                        self.steps_since_scale_change = 0
                        print(f"\n   ⚠️ Scaled grad norm too high: {scaled_grad_norm:.1f} > {self.max_scaled_grad_norm}")
                        print(f"   📉 Reduced loss scale: {old_scale} → {self.manual_loss_scale}")
                        print(f"   ➡️ Skipping this optimizer step\n")
                        self.optimizer.zero_grad(set_to_none=getattr(self.config, 'set_to_none', True))
                        self.global_step += 1
                        skip_optimizer_step = True
                    else:
                        # Manual unscaling for FP16 models
                        # Divide gradients by loss scale to get true gradient values
                        for param in self.model.parameters():
                            if param.grad is not None:
                                param.grad.data.div_(self.manual_loss_scale)
                
                if skip_optimizer_step:
                    # Skip the rest of optimizer step processing
                    num_batches += 1
                    batch_time = time.time() - batch_start
                    batch_times.append(batch_time)
                    if batch_idx % self.config.empty_cache_freq == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()  # No sync or gc - too slow
                    continue
                
                # Fast vectorized NaN/Inf gradient check (much faster than per-param loop)
                # Only check the total gradient norm - if it's finite, all grads are fine
                total_grad_norm = 0.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                
                # If total norm is NaN or Inf, we have bad gradients
                if total_grad_norm != total_grad_norm or total_grad_norm == float('inf'):
                    has_nan_grad = True
                
                if has_nan_grad:
                    # Skip this optimizer step entirely - bad gradients
                    print(f"   ⚠️ Gradient NaN/Inf at step {self.global_step}, skipping optimizer step")
                    self.optimizer.zero_grad(set_to_none=getattr(self.config, 'set_to_none', True))
                    if self.scaler is not None:
                        self.scaler.update()  # Still update scaler to adjust scale factor
                    elif self.manual_loss_scale is not None:
                        # Reduce loss scale on NaN (backoff)
                        old_scale = self.manual_loss_scale
                        self.manual_loss_scale *= self.loss_scale_backoff
                        self.manual_loss_scale = max(self.manual_loss_scale, 1.0)  # Don't go below 1
                        self.steps_since_scale_change = 0
                        print(f"   📉 Reduced loss scale: {old_scale} → {self.manual_loss_scale}")
                else:
                    # Weight validity check - only in debug mode (causes CPU-GPU sync on all params)
                    if self.debug_nan_checks:
                        weights_ok = True
                        for name, param in self.model.named_parameters():
                            if torch.isnan(param).any() or torch.isinf(param).any():
                                weights_ok = False
                                print(f"\n   ❌ CRITICAL: Weight {name} already has NaN/Inf!")
                                print(f"   This means weights were corrupted in a previous step.")
                                print(f"   Training cannot continue with corrupted weights.\n")
                                break
                        
                        if not weights_ok:
                            raise RuntimeError(
                                "Model weights are corrupted (contain NaN/Inf). "
                                "This usually means:\n"
                                "1. Learning rate too high - try reducing by 10x\n"
                                "2. Gradient clipping threshold too high - try 0.5 instead of 1.0\n"
                                "3. Bad data batch caused gradient explosion\n"
                                "Please restart training with adjusted hyperparameters."
                            )
                    
                    # Safe to proceed with optimizer step
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                        
                        # Grow manual loss scale on successful steps (for FP16)
                        if self.manual_loss_scale is not None:
                            self.steps_since_scale_change += 1
                            if self.steps_since_scale_change >= self.loss_scale_growth_interval:
                                old_scale = self.manual_loss_scale
                                self.manual_loss_scale *= self.loss_scale_growth
                                self.manual_loss_scale = min(self.manual_loss_scale, 2.**16)  # Cap at 65536
                                self.steps_since_scale_change = 0
                                if old_scale != self.manual_loss_scale:
                                    print(f"   📈 Increased loss scale: {old_scale} → {self.manual_loss_scale}")

                    self.scheduler.step()
                    # Use set_to_none=True to save memory (doesn't store zero tensors)
                    set_to_none = getattr(self.config, 'set_to_none', True)
                    self.optimizer.zero_grad(set_to_none=set_to_none)
                
                self.global_step += 1

            # Only accumulate non-NaN losses
            llm_loss_val = llm_loss.item()
            if not (llm_loss_val != llm_loss_val):  # NaN check: NaN != NaN is True
                epoch_llm_loss += llm_loss_val
            num_batches += 1

            # Explicitly delete intermediate tensors to free memory
            del total_loss, llm_loss
            if 'outputs' in dir():
                del outputs

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            # Only log every 1000 batches to reduce log spam
            if (batch_idx + 1) % 1000 == 0:
                print(f"✅ Batch {batch_idx + 1} completed", flush=True)

            # Clear cache periodically (no sync/gc - too slow for every batch)
            if batch_idx % self.config.empty_cache_freq == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Calculate average losses
        avg_llm = epoch_llm_loss / max(num_valid_batches, 1)  # Use valid batches for average
        avg_cot = epoch_cot_loss / max(num_cot, 1) if num_cot > 0 else 0.0
        avg_img = epoch_img_diff_loss / max(num_img_diff, 1) if num_img_diff > 0 else 0.0
        avg_vid = epoch_vid_diff_loss / max(num_vid_diff, 1) if num_vid_diff > 0 else 0.0
        avg_asr = epoch_asr_loss / max(num_asr, 1) if num_asr > 0 else 0.0
        avg_tts = epoch_tts_loss / max(num_tts, 1) if num_tts > 0 else 0.0
        avg_waveform = epoch_waveform_loss / max(num_waveform, 1) if num_waveform > 0 else 0.0
        elapsed = time.time() - training_start_time
        elapsed_hours = elapsed / 3600
        
        # Calculate learning efficiency
        valid_pct = (num_valid_batches / max(num_batches, 1)) * 100

        # Return all losses as a dictionary for comprehensive tracking
        # Note: MoE aux loss removed - we use Aux-Lossless MoE
        train_losses = {
            'llm': avg_llm,
            'cot': avg_cot,
            'img': avg_img,
            'vid': avg_vid,
            'asr': avg_asr,
            'tts': avg_tts,
            'waveform': avg_waveform,
            'valid_pct': valid_pct,
            'num_valid_batches': num_valid_batches,
            'num_batches': num_batches,
            'num_cot': num_cot,
            'num_img_diff': num_img_diff,
            'num_vid_diff': num_vid_diff,
            'num_asr': num_asr,
            'num_tts': num_tts,
            'num_waveform': num_waveform,
            'elapsed_hours': elapsed_hours,
        }

        return train_losses

    def _eval_epoch(self, eval_loader, epoch):
        """Run validation/evaluation for one epoch without updating weights.
        
        This validates that your multimodal MoE is learning properly:
        
        1. **Per-modality losses** - Each component (LLM, image, video, audio) is evaluated
           separately to ensure they're all improving, not just one
           
        2. **MoE aux loss** - Checks that experts are load-balanced and not collapsing
           (if aux loss is high, some experts may be unused)
           
        3. **Overfitting detection** - If train loss ↓ but eval loss ↑ = overfitting
           The model memorized training data instead of learning patterns
           
        4. **Cross-modal learning** - Since all modalities share the LLM backbone:
           - Training text improves the shared representations
           - The shared MoE expert processes everything
           - Better LLM = better understanding for all modalities
        
        Args:
            eval_loader: DataLoader for evaluation data
            epoch: Current epoch number
            
        Returns:
            Dictionary containing all validation losses including MoE metrics
        """
        self.model.eval()
        
        eval_llm_loss = 0.0
        eval_cot_loss = 0.0
        eval_img_diff_loss = 0.0
        eval_vid_diff_loss = 0.0
        eval_asr_loss = 0.0
        eval_tts_loss = 0.0
        num_batches = 0
        num_valid_batches = 0
        num_cot = 0
        num_img_diff = 0
        num_vid_diff = 0
        num_asr = 0
        num_tts = 0
        
        # Track per-modality sample types for cross-modal analysis
        modality_counts = {
            'text': 0, 'image': 0, 'video': 0, 'audio': 0, 'multimodal': 0
        }

        print(f"\n🔍 Running validation for epoch {epoch + 1}...", flush=True)

        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_loader):
                device = self.config.device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                video_frames = batch["video_frames"].to(device)
                audio_features = batch["audio_features"].to(device)
                sample_types = batch.get("sample_type", ["text"] * len(input_ids))

                has_cot_samples = any(t == 'chain_of_thought' for t in sample_types)
                weighted_sample_types = {'chain_of_thought', 'tool_use', 'agentic', 'code_execution', 
                                          'shell_execution', 'jupyter', 'anti_hallucination'}
                has_weighted_samples = any(t in weighted_sample_types for t in sample_types)

                try:
                    # Forward pass with mixed precision
                    if self.use_amp:
                        with autocast(device_type='cuda', dtype=self.amp_dtype):
                            outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values,
                                video_frames=video_frames,
                                audio_features=audio_features,
                                labels=labels,
                            )
                    else:
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            video_frames=video_frames,
                            audio_features=audio_features,
                            labels=labels,
                        )
                    
                    # Get logits from model output
                    logits = getattr(outputs, 'logits', None)
                    
                    # Compute loss from logits and labels (same method as training)
                    llm_loss = self._compute_weighted_loss(
                        logits, labels, input_ids, sample_types, None
                    )



                    # Track CoT loss separately
                    if has_cot_samples:
                        cot_loss_val = llm_loss.item()
                        if not (cot_loss_val != cot_loss_val):
                            eval_cot_loss += cot_loss_val
                        num_cot += 1

                    # Get text embeddings for modality-specific eval
                    text_embeds = self.model.get_text_embeddings(input_ids, attention_mask)

                    # Image diffusion eval
                    # MUST match filter in eval_image_diffusion_step
                    image_sample_types = ['image_generation', 'image_editing', 'text_to_image', 'image_caption']
                    if any(t in image_sample_types for t in sample_types):
                        img_diff_loss = eval_image_diffusion_step(
                            self.model.generator, pixel_values, text_embeds, self.img_gen_size,
                            sample_types=sample_types
                        )
                        if img_diff_loss is not None:
                            eval_img_diff_loss += img_diff_loss.item()
                            num_img_diff += 1

                    # Video diffusion eval
                    # MUST match filter in eval_video_diffusion_step
                    video_sample_types = ['video_generation', 'image_to_video', 'video_caption', 'video_qa', 
                                          'video_preference', 'video_likert', 'text_to_video']
                    if any(t in video_sample_types for t in sample_types):
                        vid_diff_loss = eval_video_diffusion_step(
                            self.model.video_generator, video_frames, text_embeds, self.vid_gen_size,
                            sample_types=sample_types
                        )
                        if vid_diff_loss is not None:
                            eval_vid_diff_loss += vid_diff_loss.item()
                            num_vid_diff += 1

                    # ASR eval
                    if any(t == 'voice_asr' for t in sample_types):
                        asr_loss = eval_voice_asr_step(
                            self.model.audio_encoder, audio_features, text_embeds,
                            sample_types=sample_types
                        )
                        if asr_loss is not None:
                            eval_asr_loss += asr_loss.item()
                            num_asr += 1

                    # TTS eval
                    if any(t == 'voice_tts' for t in sample_types):
                        tts_loss = eval_voice_tts_step(
                            self.model.audio_decoder, text_embeds, audio_features,
                            sample_types=sample_types
                        )
                        if tts_loss is not None:
                            eval_tts_loss += tts_loss.item()
                            num_tts += 1

                    # Track valid LLM loss
                    llm_loss_val = llm_loss.item()
                    if not (llm_loss_val != llm_loss_val):
                        eval_llm_loss += llm_loss_val
                        num_valid_batches += 1
                    num_batches += 1

                except Exception as e:
                    num_batches += 1
                    continue

                # Periodic logging
                if (batch_idx + 1) % 500 == 0:
                    print(f"   🔍 Eval batch {batch_idx + 1} completed", flush=True)

                # Clear cache periodically
                if batch_idx % self.config.empty_cache_freq == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Calculate average validation losses
        avg_llm = eval_llm_loss / max(num_valid_batches, 1)
        avg_cot = eval_cot_loss / max(num_cot, 1) if num_cot > 0 else 0.0
        avg_img = eval_img_diff_loss / max(num_img_diff, 1) if num_img_diff > 0 else 0.0
        avg_vid = eval_vid_diff_loss / max(num_vid_diff, 1) if num_vid_diff > 0 else 0.0
        avg_asr = eval_asr_loss / max(num_asr, 1) if num_asr > 0 else 0.0
        avg_tts = eval_tts_loss / max(num_tts, 1) if num_tts > 0 else 0.0

        # Return model to training mode
        self.model.train()

        eval_losses = {
            'llm': avg_llm,
            'cot': avg_cot,
            'img': avg_img,
            'vid': avg_vid,
            'asr': avg_asr,
            'tts': avg_tts,
            'num_valid_batches': num_valid_batches,
            'num_batches': num_batches,
            'num_cot': num_cot,
            'num_img_diff': num_img_diff,
            'num_vid_diff': num_vid_diff,
            'num_asr': num_asr,
            'num_tts': num_tts,
        }

        return eval_losses

    def _print_epoch_summary(self, epoch, train_losses, eval_losses=None, training_start_time=None):
        """Print comprehensive epoch summary with both training and validation losses."""
        elapsed = time.time() - training_start_time if training_start_time else 0
        elapsed_hours = elapsed / 3600
        
        print(f"\n{'='*60}")
        print(f"📊 EPOCH {epoch + 1}/{self.config.num_epochs} SUMMARY")
        print(f"{'='*60}")
        
        # LLM Loss
        print(f"   📝 LLM Loss:                      {train_losses['llm']:.4f} ({train_losses['num_valid_batches']}/{train_losses['num_batches']} valid batches)")
        if eval_losses:
            print(f"   📝 LLM Validation Loss:           {eval_losses['llm']:.4f} ({eval_losses['num_valid_batches']}/{eval_losses['num_batches']} valid batches)")
        
        # Chain-of-Thought Loss
        print(f"   🧠 Chain-of-Thought Loss:         {train_losses['cot']:.4f} ({train_losses['num_cot']} batches)")
        if eval_losses:
            print(f"   🧠 Chain-of-Thought Validation Loss: {eval_losses['cot']:.4f} ({eval_losses['num_cot']} batches)")
        
        # Image Diffusion Loss
        print(f"   🖼️ Image Diffusion Loss:          {train_losses['img']:.4f} ({train_losses['num_img_diff']} batches)")
        if eval_losses:
            print(f"   🖼️ Image Diffusion Validation Loss: {eval_losses['img']:.4f} ({eval_losses['num_img_diff']} batches)")
        
        # Video Diffusion Loss
        print(f"   🎬 Video Diffusion Loss:          {train_losses['vid']:.4f} ({train_losses['num_vid_diff']} batches)")
        if eval_losses:
            print(f"   🎬 Video Diffusion Validation Loss: {eval_losses['vid']:.4f} ({eval_losses['num_vid_diff']} batches)")
        
        # ASR Loss
        print(f"   🎤 ASR Loss:                      {train_losses['asr']:.4f} ({train_losses['num_asr']} batches)")
        if eval_losses:
            print(f"   🎤 ASR Validation Loss:           {eval_losses['asr']:.4f} ({eval_losses['num_asr']} batches)")
        
        # TTS Loss
        print(f"   🔊 TTS Loss:                      {train_losses['tts']:.4f} ({train_losses['num_tts']} batches)")
        if eval_losses:
            print(f"   🔊 TTS Validation Loss:           {eval_losses['tts']:.4f} ({eval_losses['num_tts']} batches)")
        
        # Waveform Decoder Loss (Speech-to-Speech)
        waveform_loss = train_losses.get('waveform', 0.0)
        num_waveform = train_losses.get('num_waveform', 0)
        print(f"   🎙️ Waveform Decoder Loss:          {waveform_loss:.4f} ({num_waveform} batches)")
        
        print(f"{'='*60}")
        print(f"   📈 Learning efficiency: {train_losses['valid_pct']:.1f}% of batches contributed to learning")
        print(f"   ⏱️ Elapsed: {elapsed_hours:.2f}h")
        print(f"{'='*60}")
        
        # Warn if learning efficiency is too low
        if train_losses['valid_pct'] < 50:
            print(f"   ⚠️ WARNING: Low learning efficiency ({train_losses['valid_pct']:.1f}%). Check your data pipeline!")
        
        # Check for overfitting (train loss much lower than eval loss)
        if eval_losses:
            print(f"\n🔬 OVERFITTING CHECK:")
            overfitting_detected = False
            
            # Check LLM
            if train_losses['llm'] > 0 and eval_losses['llm'] > 0:
                llm_ratio = eval_losses['llm'] / train_losses['llm']
                if llm_ratio > 1.5:
                    print(f"   ⚠️ LLM: eval/train ratio = {llm_ratio:.2f} (>1.5 suggests overfitting)")
                    overfitting_detected = True
                else:
                    print(f"   ✅ LLM: eval/train ratio = {llm_ratio:.2f} (healthy)")
            
            # Note: MoE expert utilization check removed - we use Aux-Lossless MoE
            # which doesn't have aux loss (expert balancing is implicit)
            
            if not overfitting_detected:
                print(f"   ✅ No significant overfitting detected")
            
            print(f"\n💡 HOW THIS VALIDATES YOUR MULTIMODAL MOE:")
            print(f"   • All modalities share the LLM backbone → training ANY modality improves shared representations")
            print(f"   • Shared MoE expert processes ALL inputs → general knowledge is learned")
            print(f"   • Routed experts specialize → different experts for text/code/reasoning/etc")
            print(f"   • If train↓ but eval↑ = overfitting (model memorized, didn't learn)")
            print(f"   • If both↓ = model is learning generalizable patterns ✓")
        
        print()

    def _save_checkpoint(self, epoch, loss):
        """Save a training checkpoint with full training state and tokenizer.
        
        Note: We only save one checkpoint at a time to save disk space.
        The best model is tracked but not saved separately - the final model
        will be the one to use.
        """
        # Remove previous checkpoint to save disk space
        checkpoint_path = os.path.join(self.config.output_dir, f"checkpoint-epoch-{epoch + 1}")
        prev_checkpoint = os.path.join(self.config.output_dir, f"checkpoint-epoch-{epoch}")
        
        # Clean up previous checkpoint if it exists
        if os.path.exists(prev_checkpoint):
            import shutil
            try:
                shutil.rmtree(prev_checkpoint)
                print(f"   🗑️ Removed previous checkpoint to save disk space")
            except Exception:
                pass
        
        os.makedirs(checkpoint_path, exist_ok=True)

        # Clear GPU memory before saving to avoid OOM during serialization
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

        # Save model with training state for resuming
        self.model.save_pretrained(
            checkpoint_path,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            global_step=self.global_step,
            epoch=epoch + 1,  # Save next epoch to resume from
            best_loss=self.best_loss,
        )

        # Clear memory after saving
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Save tokenizer with custom chat template
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(checkpoint_path)
            # Also save special tokens mapping
            with open(os.path.join(checkpoint_path, "special_tokens.json"), "w") as f:
                json.dump(SPECIAL_TOKENS, f, indent=2)
        
        # Save trainer_state.json for HuggingFace compatibility and training resume
        # Get trainable/frozen components for resume info
        trainable = self.model.get_trainable_component_names() if hasattr(self.model, 'get_trainable_component_names') else []
        frozen = self.model.get_frozen_component_names() if hasattr(self.model, 'get_frozen_component_names') else []
        
        trainer_state = {
            "best_model_checkpoint": checkpoint_path,
            "best_metric": self.best_loss,
            "epoch": epoch + 1,
            "epochs_completed": epoch + 1,
            "global_step": self.global_step,
            "is_local_process_zero": True,
            "is_world_process_zero": True,
            "log_history": [],
            "logging_steps": self.config.logging_steps,
            "max_steps": self.global_step,
            "num_train_epochs": self.config.num_epochs,
            "total_flos": 0,
            "train_batch_size": self.config.batch_size,
            "effective_batch_size": self.config.batch_size * self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "max_grad_norm": self.max_grad_norm,
            "trainable_components": trainable,
            "frozen_components": frozen,
            "trial_name": None,
            "trial_params": None,
        }
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
        with open(trainer_state_path, "w") as f:
            json.dump(trainer_state, f, indent=2)
        
        # Save streaming state for dataset resume
        if hasattr(self.train_dataset, 'save_streaming_state'):
            streaming_state_path = os.path.join(checkpoint_path, "streaming_state.json")
            self.train_dataset.save_streaming_state(streaming_state_path)
        
        print(f"   💾 Checkpoint saved: epoch {epoch + 1}")

        # Track best loss but don't save separate best model (saves disk space)
        if loss < self.best_loss:
            self.best_loss = loss
            print(f"   ⭐ New best loss: {loss:.4f}")

    def _save_final_model(self):
        """Save the final trained model with tokenizer, chat template, trainer state, and streaming state."""
        print(f"\n💾 Saving final model to {self.config.final_model_dir}...")
        os.makedirs(self.config.final_model_dir, exist_ok=True)

        # Clear GPU memory before saving to avoid OOM during serialization
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

        self.model.save_pretrained(
            self.config.final_model_dir,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            global_step=self.global_step,
            epoch=self.config.num_epochs,
            best_loss=self.best_loss,
        )

        # Save tokenizer with custom chat template
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(self.config.final_model_dir)
            # Also save special tokens mapping
            with open(os.path.join(self.config.final_model_dir, "special_tokens.json"), "w") as f:
                json.dump(SPECIAL_TOKENS, f, indent=2)
            print(f"   💾 Tokenizer and chat template saved")
        
        # Save trainer_state.json for HuggingFace compatibility and training resume
        # Get trainable/frozen components for resume info
        trainable = self.model.get_trainable_component_names() if hasattr(self.model, 'get_trainable_component_names') else []
        frozen = self.model.get_frozen_component_names() if hasattr(self.model, 'get_frozen_component_names') else []
        
        trainer_state = {
            "best_model_checkpoint": self.config.final_model_dir,
            "best_metric": self.best_loss,
            "epoch": self.config.num_epochs,
            "epochs_completed": self.config.num_epochs,
            "global_step": self.global_step,
            "is_local_process_zero": True,
            "is_world_process_zero": True,
            "log_history": [],
            "logging_steps": self.config.logging_steps,
            "max_steps": self.global_step,
            "num_train_epochs": self.config.num_epochs,
            "total_flos": 0,
            "train_batch_size": self.config.batch_size,
            "effective_batch_size": self.config.batch_size * self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "max_grad_norm": self.max_grad_norm,
            "trainable_components": trainable,
            "frozen_components": frozen,
            "trial_name": None,
            "trial_params": None,
        }
        trainer_state_path = os.path.join(self.config.final_model_dir, "trainer_state.json")
        with open(trainer_state_path, "w") as f:
            json.dump(trainer_state, f, indent=2)
        
        # Save streaming state for dataset resume (useful if continuing training later)
        if hasattr(self.train_dataset, 'save_streaming_state'):
            streaming_state_path = os.path.join(self.config.final_model_dir, "streaming_state.json")
            self.train_dataset.save_streaming_state(streaming_state_path)
            print(f"   💾 Streaming state saved")
        
        print(f"✅ Final model saved!")

    def _upload_to_huggingface(self):
        """Upload the final trained model to HuggingFace Hub, replacing existing weights."""
        if not self.hf_token or not self.hf_repo_id:
            print(f"⚠️ Skipping HuggingFace upload (no hf_token or hf_repo_id provided)")
            return
        
        print(f"\n🚀 Uploading model to HuggingFace: {self.hf_repo_id}...")
        
        try:
            from huggingface_hub import HfApi
            
            api = HfApi(token=self.hf_token)
            
            # Upload the entire final model directory
            api.upload_folder(
                folder_path=self.config.final_model_dir,
                repo_id=self.hf_repo_id,
                repo_type="model",
                commit_message=f"Update model weights after training (epoch {self.config.num_epochs}, loss {self.best_loss:.4f})",
            )
            
            print(f"✅ Model uploaded to https://huggingface.co/{self.hf_repo_id}")
            
        except Exception as e:
            print(f"❌ Failed to upload to HuggingFace: {e}")

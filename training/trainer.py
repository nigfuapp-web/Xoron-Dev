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
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.xoron_config = xoron_config
        self.collate_fn = collate_fn
        self.tokenizer = tokenizer

        # Mixed precision setup (prefer BF16 if available)
        self.use_amp = config.fp16 or getattr(config, 'bf16', False)
        self.amp_dtype = torch.bfloat16 if getattr(config, 'bf16', False) else torch.float16
        self.scaler = GradScaler() if config.fp16 and not getattr(config, 'bf16', False) and config.device == "cuda" else None
        
        self.global_step = 0
        self.start_epoch = 0
        self.best_loss = float("inf")

        # Get generation sizes from config
        self.img_gen_size = xoron_config.generation_image_size
        self.vid_gen_size = getattr(xoron_config, 'generation_video_size', xoron_config.generation_image_size)
        
        # Loss weights from config (SOTA: configurable per-modality weights)
        self.llm_loss_weight = getattr(config, 'llm_loss_weight', 1.0)
        self.image_diffusion_loss_weight = getattr(config, 'image_diffusion_loss_weight', 0.1)
        self.video_diffusion_loss_weight = getattr(config, 'video_diffusion_loss_weight', 0.1)
        self.asr_loss_weight = getattr(config, 'asr_loss_weight', 0.1)
        self.tts_loss_weight = getattr(config, 'tts_loss_weight', 0.1)
        self.moe_aux_loss_weight = getattr(config, 'moe_aux_loss_weight', 0.01)
        
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
        
        # Enable gradient checkpointing if configured
        if getattr(config, 'gradient_checkpointing', False):
            self._enable_gradient_checkpointing()

        # Resume from checkpoint if specified
        if resume_from is not None:
            self._load_checkpoint(resume_from)
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.model, 'llm') and hasattr(self.model.llm, 'gradient_checkpointing'):
            self.model.llm.gradient_checkpointing = True
            print("   ✅ Gradient checkpointing enabled for LLM")
    
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
    ) -> torch.Tensor:
        """
        Compute loss with higher weight for important token groups.
        
        This encourages the model to pay more attention to:
        - Chain-of-thought reasoning tokens (think, plan, critique, etc.)
        - Tool calling tokens (tool_call, function_name, etc.)
        - Anti-hallucination tokens (uncertain, cite, verify, etc.)
        - Code execution tokens (exec, jupyter, code, etc.)
        
        Each group has its own configurable weight multiplier.
        """
        # Standard cross-entropy loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # CRITICAL FIX: Ensure labels are on the same device as logits
        if shift_logits.device != shift_labels.device:
            shift_labels = shift_labels.to(shift_logits.device)
        
        # Compute per-token loss
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
            loss = (weighted_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        else:
            # Standard loss computation
            valid_mask = (shift_labels != -100).float()
            loss = (per_token_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        
        return loss
    
    # Keep old method name for backward compatibility
    def _compute_cot_weighted_loss(self, logits, labels, input_ids, sample_types):
        """Backward compatible alias for _compute_weighted_loss."""
        return self._compute_weighted_loss(logits, labels, input_ids, sample_types)

    def train(self):
        """Run the full training loop."""
        print("\n" + "=" * 60)
        if self.start_epoch > 0:
            print("🔄 RESUMING MULTIMODAL TRAINING (SOTA)")
            print(f"   Starting from epoch {self.start_epoch + 1}, step {self.global_step}")
        else:
            print("🚀 STARTING FULL MULTIMODAL TRAINING (SOTA)")
        print("   ✓ LLM: text/conversation/code/tools/agentic")
        print("   ✓ Chain-of-Thought: structured reasoning with special tokens")
        print("   ✓ Vision: image/video understanding")
        print("   ✓ Image Diffusion: text-to-image generation")
        print("   ✓ Video Diffusion: text-to-video + image-to-video (REAL DATA)")
        print("   ✓ Voice: ASR (speech-to-text) + TTS (text-to-speech)")
        print("   ✓ Image Editing: instruction-guided image editing")
        
        # Show trainable components
        trainable = self.model.get_trainable_component_names()
        print(f"   🔥 Trainable components: {', '.join(trainable)}")
        print("=" * 60)

        print(f"\n📐 Generation sizes:")
        print(f"   Image: {self.img_gen_size}x{self.img_gen_size}")
        print(f"   Video: {self.vid_gen_size}x{self.vid_gen_size}")
        
        print(f"\n🎯 Loss weights (SOTA configurable):")
        print(f"   LLM: {self.llm_loss_weight}")
        print(f"   Image Diffusion: {self.image_diffusion_loss_weight}")
        print(f"   Video Diffusion: {self.video_diffusion_loss_weight}")
        print(f"   ASR: {self.asr_loss_weight}")
        print(f"   TTS: {self.tts_loss_weight}")
        print(f"   MoE Aux: {self.moe_aux_loss_weight}")
        
        print(f"\n🧠 Token group weights (for focused learning):")
        print(f"   Chain-of-Thought: {self.cot_loss_weight}x (reasoning tokens)")
        print(f"   Tool Calling: {self.tool_loss_weight}x (function/tool tokens)")
        print(f"   Anti-Hallucination: {self.anti_hallucination_loss_weight}x (uncertainty/citation tokens)")
        print(f"   Code Execution: {self.code_exec_loss_weight}x (exec/jupyter tokens)")
        
        print(f"\n⚙️ Training settings:")
        print(f"   Mixed precision: {'BF16' if self.amp_dtype == torch.bfloat16 else 'FP16' if self.use_amp else 'FP32'}")
        print(f"   Gradient clipping: {self.max_grad_norm}")

        self.model.train()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        training_start_time = time.time()
        batch_times = []

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self.collate_fn
        )

        for epoch in range(self.start_epoch, self.config.num_epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*60}")

            if epoch > self.start_epoch:
                self.train_dataset.reset()

            epoch_loss = self._train_epoch(train_loader, epoch, batch_times, training_start_time)

            # Aggressive memory cleanup before saving checkpoint
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()  # Double clear for fragmentation
                
            # Force Python garbage collection
            gc.collect()

            # Save checkpoint
            if (epoch + 1) % 1 == 0:
                self._save_checkpoint(epoch, epoch_loss)
                
            # Clear memory again after saving before next epoch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()

        # Final save
        self._save_final_model()

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
        num_batches = 0
        num_cot = 0
        num_img_diff = 0
        num_vid_diff = 0
        num_asr = 0
        num_tts = 0
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
            
            # Check for chain-of-thought samples
            has_cot_samples = any(t == 'chain_of_thought' for t in sample_types)

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
                    
                    # Use weighted loss for CoT samples (safely get logits)
                    logits = getattr(outputs, 'logits', None)
                    if has_cot_samples and logits is not None:
                        llm_loss = self._compute_cot_weighted_loss(
                            logits, labels, input_ids, sample_types
                        )
                    else:
                        llm_loss = getattr(outputs, 'loss', None)
                        if llm_loss is None:
                            llm_loss = torch.tensor(0.0, device=device)
                    
                    # Get MoE auxiliary loss if available
                    moe_aux_loss = getattr(outputs, 'aux_loss', None)
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    video_frames=video_frames,
                    audio_features=audio_features,
                    labels=labels,
                )
                
                # Use weighted loss for CoT samples (safely get logits)
                logits = getattr(outputs, 'logits', None)
                if has_cot_samples and logits is not None:
                    llm_loss = self._compute_cot_weighted_loss(
                        logits, labels, input_ids, sample_types
                    )
                else:
                    llm_loss = getattr(outputs, 'loss', None)
                    if llm_loss is None:
                        llm_loss = torch.tensor(0.0, device=device)
                
                # Get MoE auxiliary loss if available
                moe_aux_loss = getattr(outputs, 'aux_loss', None)
            
            # Track CoT loss separately
            if has_cot_samples:
                epoch_cot_loss += llm_loss.item()
                num_cot += 1

            # Apply LLM loss weight
            total_loss = self.llm_loss_weight * llm_loss
            
            # Add MoE auxiliary loss if available (SOTA: proper MoE training)
            if moe_aux_loss is not None and moe_aux_loss.item() > 0:
                total_loss = total_loss + self.moe_aux_loss_weight * moe_aux_loss

            # Train diffusion models based on sample type
            text_embeds = self.model.get_text_embeddings(input_ids, attention_mask)

            # Image diffusion (use configurable loss weight)
            if any(t in ['image_generation', 'image_editing'] for t in sample_types):
                img_diff_loss = train_image_diffusion_step(
                    self.model.generator, pixel_values, text_embeds, self.img_gen_size,
                    sample_types=sample_types
                )
                if img_diff_loss is not None:
                    img_diff_loss = img_diff_loss.to(total_loss.device)
                    total_loss = total_loss + self.image_diffusion_loss_weight * img_diff_loss
                    epoch_img_diff_loss += img_diff_loss.item()
                    num_img_diff += 1

            # Video diffusion - train on ALL video sample types (use configurable loss weight)
            video_sample_types = ['video_generation', 'image_to_video', 'video_caption', 'video_qa', 'video_preference', 'video_likert']
            if any(t in video_sample_types for t in sample_types):
                vid_diff_loss = train_video_diffusion_step(
                    self.model.video_generator, video_frames, text_embeds, self.vid_gen_size,
                    sample_types=sample_types
                )
                if vid_diff_loss is not None:
                    vid_diff_loss = vid_diff_loss.to(total_loss.device)
                    total_loss = total_loss + self.video_diffusion_loss_weight * vid_diff_loss
                    epoch_vid_diff_loss += vid_diff_loss.item()
                    num_vid_diff += 1

            # Voice ASR (use configurable loss weight)
            if any(t == 'voice_asr' for t in sample_types):
                asr_loss = train_voice_asr_step(
                    self.model.audio_encoder, audio_features, text_embeds,
                    sample_types=sample_types
                )
                if asr_loss is not None:
                    asr_loss = asr_loss.to(total_loss.device)
                    total_loss = total_loss + self.asr_loss_weight * asr_loss
                    epoch_asr_loss += asr_loss.item()
                    num_asr += 1

            # Voice TTS (use configurable loss weight)
            if any(t == 'voice_tts' for t in sample_types):
                tts_loss = train_voice_tts_step(
                    self.model.audio_decoder, text_embeds, audio_features,
                    sample_types=sample_types
                )
                if tts_loss is not None:
                    tts_loss = tts_loss.to(total_loss.device)
                    total_loss = total_loss + self.tts_loss_weight * tts_loss
                    epoch_tts_loss += tts_loss.item()
                    num_tts += 1

            # Backward pass (supports both FP16 with scaler and BF16 without)
            if self.scaler is not None:
                self.scaler.scale(total_loss / self.config.gradient_accumulation_steps).backward()
            else:
                (total_loss / self.config.gradient_accumulation_steps).backward()

            # Optimizer step with gradient clipping from config
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                self.scheduler.step()
                # Use set_to_none=True to save memory (doesn't store zero tensors)
                set_to_none = getattr(self.config, 'set_to_none', True)
                self.optimizer.zero_grad(set_to_none=set_to_none)
                self.global_step += 1

            epoch_llm_loss += llm_loss.item()
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

            # Clear cache periodically with proper GPU sync
            if batch_idx % self.config.empty_cache_freq == 0 and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()  # Also run garbage collection

        # Print epoch summary with all component losses
        avg_llm = epoch_llm_loss / max(num_batches, 1)
        avg_cot = epoch_cot_loss / max(num_cot, 1) if num_cot > 0 else 0.0
        avg_img = epoch_img_diff_loss / max(num_img_diff, 1) if num_img_diff > 0 else 0.0
        avg_vid = epoch_vid_diff_loss / max(num_vid_diff, 1) if num_vid_diff > 0 else 0.0
        avg_asr = epoch_asr_loss / max(num_asr, 1) if num_asr > 0 else 0.0
        avg_tts = epoch_tts_loss / max(num_tts, 1) if num_tts > 0 else 0.0
        elapsed = time.time() - training_start_time
        elapsed_hours = elapsed / 3600
        
        print(f"\n{'='*50}")
        print(f"📊 EPOCH {epoch + 1}/{self.config.num_epochs} SUMMARY")
        print(f"{'='*50}")
        print(f"   📝 LLM Loss:              {avg_llm:.4f} ({num_batches} batches)")
        print(f"   🧠 Chain-of-Thought Loss: {avg_cot:.4f} ({num_cot} batches)")
        print(f"   🖼️ Image Diffusion Loss:  {avg_img:.4f} ({num_img_diff} batches)")
        print(f"   🎬 Video Diffusion Loss:  {avg_vid:.4f} ({num_vid_diff} batches)")
        print(f"   🎤 ASR Loss:              {avg_asr:.4f} ({num_asr} batches)")
        print(f"   🔊 TTS Loss:              {avg_tts:.4f} ({num_tts} batches)")
        print(f"{'='*50}")
        print(f"   ⏱️ Elapsed: {elapsed_hours:.2f}h")
        print(f"{'='*50}\n")

        return avg_llm

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
            print(f"   💾 Checkpoint saved: epoch {epoch + 1}")

        # Track best loss but don't save separate best model (saves disk space)
        if loss < self.best_loss:
            self.best_loss = loss
            print(f"   ⭐ New best loss: {loss:.4f}")

    def _save_final_model(self):
        """Save the final trained model with tokenizer and chat template."""
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
        
        print(f"✅ Final model saved!")

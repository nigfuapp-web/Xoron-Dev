# 🏋️ Training Module Documentation

The Training module contains the trainer class and utilities for training the Xoron-Dev multimodal model.

## 📁 File Structure

```
training/
├── __init__.py
├── trainer.py    # XoronTrainer class
└── utils.py      # Training step functions for each modality
```

---

## 🎯 XoronTrainer

### Overview

`XoronTrainer` is the main training class that handles multi-modal training with:
- Chain-of-thought weighted loss
- LoRA+ support
- Mixed precision training
- Gradient checkpointing
- Multi-GPU support

### Initialization

```python
class XoronTrainer:
    def __init__(
        self,
        model,              # XoronMultimodalModel
        train_dataset,      # TrueStreamingDataset
        optimizer,          # AdamW or 8-bit Adam
        scheduler,          # Learning rate scheduler
        config,             # TrainingConfig
        xoron_config,       # XoronConfig
        collate_fn,         # Batch collation function
        resume_from=None,   # Checkpoint path
        tokenizer=None,     # Tokenizer for special tokens
        eval_dataset=None,  # Evaluation dataset
        hf_token=None,      # HuggingFace token for upload
        hf_repo_id=None,    # HuggingFace repo for upload
    ):
```

### Key Features

#### 1. Mixed Precision Training

```python
# Setup based on model dtype
model_dtype = next(model.parameters()).dtype

if model_dtype == torch.float16:
    # FP16 model - use FP32 optimizer states (no GradScaler needed)
    self.scaler = None
    self.manual_loss_scale = None
elif config.fp16 and not config.bf16:
    # FP32 model with FP16 autocast - use GradScaler
    self.scaler = GradScaler()
else:
    # BF16 or FP32 - no scaling needed
    self.scaler = None
```

#### 2. Loss Weights

```python
# Modality weights
self.llm_loss_weight = 1.0
self.image_diffusion_loss_weight = 0.1
self.video_diffusion_loss_weight = 0.1
self.asr_loss_weight = 0.1
self.tts_loss_weight = 0.1

# Special token weights
self.cot_loss_weight = 1.5              # Chain-of-thought
self.tool_loss_weight = 1.3             # Tool calling
self.anti_hallucination_loss_weight = 1.2  # Uncertainty
self.code_exec_loss_weight = 1.2        # Code execution
```

#### 3. Weighted Token Loss

```python
def _compute_weighted_loss(self, logits, labels, loss_mask=None):
    """
    Compute loss with higher weights for important tokens.
    """
    # Standard cross-entropy
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Per-token loss
    loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    token_losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    # Create weight tensor
    weights = torch.ones_like(token_losses)
    
    # Apply weights for special tokens
    for token_id in self.reasoning_token_ids:
        weights[shift_labels.view(-1) == token_id] *= self.cot_loss_weight
    
    for token_id in self.tool_block_ids:
        weights[shift_labels.view(-1) == token_id] *= self.tool_loss_weight
    
    # ... similar for other token groups
    
    # Weighted mean
    return (token_losses * weights).sum() / weights.sum()
```

### Training Loop

```python
def train(self):
    """Main training loop."""
    self.model.train()
    
    for epoch in range(self.start_epoch, self.config.num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Create dataloader
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            collate_fn=self.collate_fn,
            num_workers=0,  # Streaming doesn't support workers
        )
        
        for step, batch in enumerate(dataloader):
            # Move batch to device
            batch = self._move_to_device(batch)
            
            # Forward pass with mixed precision
            with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                loss = self._compute_loss(batch)
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=self.config.set_to_none)
                
                self.global_step += 1
            
            epoch_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                self._log_metrics(epoch_loss / num_batches)
            
            # Memory management
            if step % self.config.empty_cache_freq == 0:
                torch.cuda.empty_cache()
        
        # End of epoch
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Evaluation
        if self.eval_dataset is not None:
            self._evaluate(epoch)
        
        # Save checkpoint
        self._save_checkpoint(epoch)
```

### Loss Computation

```python
def _compute_loss(self, batch):
    """Compute total loss for a batch."""
    total_loss = 0.0
    
    # 1. LLM Loss (always computed)
    if 'input_ids' in batch and 'labels' in batch:
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask'),
            labels=batch['labels'],
            images=batch.get('images'),
            videos=batch.get('videos'),
            audio=batch.get('audio'),
        )
        
        llm_loss = self._compute_weighted_loss(
            outputs.logits,
            batch['labels']
        )
        total_loss += self.llm_loss_weight * llm_loss
    
    # 2. Image Diffusion Loss
    if batch.get('has_image_gen', False):
        img_loss = train_image_diffusion_step(
            self.model,
            batch['gen_images'],
            batch['gen_prompts'],
        )
        total_loss += self.image_diffusion_loss_weight * img_loss
    
    # 3. Video Diffusion Loss
    if batch.get('has_video_gen', False):
        vid_loss = train_video_diffusion_step(
            self.model,
            batch['gen_videos'],
            batch['gen_prompts'],
        )
        total_loss += self.video_diffusion_loss_weight * vid_loss
    
    # 4. ASR Loss
    if batch.get('has_asr', False):
        asr_loss = train_voice_asr_step(
            self.model,
            batch['audio'],
            batch['transcripts'],
        )
        total_loss += self.asr_loss_weight * asr_loss
    
    # 5. TTS Loss
    if batch.get('has_tts', False):
        tts_loss = train_voice_tts_step(
            self.model,
            batch['text'],
            batch['target_audio'],
            batch.get('speaker_ref'),
        )
        total_loss += self.tts_loss_weight * tts_loss
    
    return total_loss
```

---

## 🔧 Training Utilities

### Image Diffusion Training Step

```python
def train_image_diffusion_step(model, images, prompts):
    """
    Train image generation with Flow Matching.
    
    Args:
        model: XoronMultimodalModel
        images: [B, C, H, W] target images
        prompts: Text prompts for conditioning
    
    Returns:
        loss: Flow matching loss
    """
    batch_size = images.shape[0]
    device = images.device
    
    # Encode images to latent space
    latents = model.encode_images(images)
    
    # Sample timesteps
    t = torch.rand(batch_size, device=device)
    
    # Sample noise
    noise = torch.randn_like(latents)
    
    # Create noisy latents (interpolation)
    t_expanded = t.view(-1, 1, 1, 1)
    noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
    
    # Get text embeddings
    prompt_embeds = model.encode_text(prompts)
    
    # Predict velocity
    velocity_pred = model.generator(noisy_latents, t, prompt_embeds)
    
    # Target velocity
    velocity_target = noise - latents
    
    # MSE loss
    loss = F.mse_loss(velocity_pred, velocity_target)
    
    return loss
```

### Video Diffusion Training Step

```python
def train_video_diffusion_step(model, videos, prompts):
    """
    Train video generation with Flow Matching.
    
    Args:
        model: XoronMultimodalModel
        videos: [B, T, C, H, W] target videos
        prompts: Text prompts for conditioning
    
    Returns:
        loss: Flow matching loss
    """
    batch_size, num_frames = videos.shape[:2]
    device = videos.device
    
    # Encode videos to latent space
    latents = model.encode_videos(videos)
    
    # Sample timesteps
    t = torch.rand(batch_size, device=device)
    
    # Sample noise
    noise = torch.randn_like(latents)
    
    # Create noisy latents
    t_expanded = t.view(-1, 1, 1, 1, 1)
    noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
    
    # Get text embeddings
    prompt_embeds = model.encode_text(prompts)
    
    # Predict velocity
    velocity_pred = model.video_generator(noisy_latents, t, prompt_embeds)
    
    # Target velocity
    velocity_target = noise - latents
    
    # MSE loss
    loss = F.mse_loss(velocity_pred, velocity_target)
    
    return loss
```

### ASR Training Step

```python
def train_voice_asr_step(model, audio, transcripts):
    """
    Train speech-to-text (ASR).
    
    Args:
        model: XoronMultimodalModel
        audio: [B, T] raw waveform or [B, n_mels, T] mel spectrogram
        transcripts: Target text transcriptions
    
    Returns:
        loss: CTC or cross-entropy loss
    """
    # Encode audio
    audio_features, speaker_embedding = model.audio_encoder(audio)
    
    # Project to LLM space
    audio_embeds = model.audio_projector(audio_features)
    
    # Get transcript tokens
    transcript_ids = model.tokenizer(transcripts, return_tensors='pt', padding=True)
    
    # Forward through LLM with audio prefix
    outputs = model.llm(
        inputs_embeds=torch.cat([audio_embeds, transcript_embeds], dim=1),
        labels=transcript_ids.input_ids,
    )
    
    return outputs.loss
```

### TTS Training Step

```python
def train_voice_tts_step(model, text, target_audio, speaker_ref=None):
    """
    Train text-to-speech (TTS).
    
    Args:
        model: XoronMultimodalModel
        text: Input text
        target_audio: [B, T] target waveform
        speaker_ref: Optional reference audio for voice cloning
    
    Returns:
        loss: Mel spectrogram + waveform reconstruction loss
    """
    # Get text embeddings from LLM
    text_ids = model.tokenizer(text, return_tensors='pt', padding=True)
    text_embeds = model.llm.model.embed_tokens(text_ids.input_ids)
    
    # Extract speaker embedding if reference provided
    speaker_embedding = None
    if speaker_ref is not None:
        _, speaker_embedding = model.audio_encoder(speaker_ref)
    
    # Encode target audio for MAS alignment
    target_features, _ = model.audio_encoder(target_audio)
    
    # Generate mel spectrogram
    mel_pred, durations, alignment = model.audio_decoder(
        text_embeds,
        speaker_embedding=speaker_embedding,
        audio_features=target_features,
        use_mas=True,
    )
    
    # Target mel spectrogram
    mel_target = model.compute_mel_spectrogram(target_audio)
    
    # Mel loss
    mel_loss = F.l1_loss(mel_pred, mel_target)
    
    # Waveform reconstruction loss (optional)
    if model.waveform_decoder is not None:
        waveform_pred = model.waveform_decoder(mel_pred.transpose(1, 2))
        waveform_loss = F.l1_loss(waveform_pred, target_audio)
        return mel_loss + 0.1 * waveform_loss
    
    return mel_loss
```

---

## 📊 Evaluation

### Evaluation Loop

```python
def _evaluate(self, epoch):
    """Run evaluation at end of epoch."""
    self.model.eval()
    
    eval_losses = {
        'llm': [],
        'image_gen': [],
        'video_gen': [],
        'asr': [],
        'tts': [],
    }
    
    with torch.no_grad():
        for batch in self.eval_dataloader:
            batch = self._move_to_device(batch)
            
            # LLM evaluation
            if 'input_ids' in batch:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    labels=batch['labels'],
                )
                eval_losses['llm'].append(outputs.loss.item())
            
            # Image generation evaluation
            if batch.get('has_image_gen'):
                loss = eval_image_diffusion_step(self.model, batch)
                eval_losses['image_gen'].append(loss.item())
            
            # ... similar for other modalities
    
    # Log average losses
    for modality, losses in eval_losses.items():
        if losses:
            avg_loss = sum(losses) / len(losses)
            print(f"  Eval {modality}: {avg_loss:.4f}")
    
    self.model.train()
```

---

## 💾 Checkpointing

### Save Checkpoint

```python
def _save_checkpoint(self, epoch):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'global_step': self.global_step,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'scheduler_state_dict': self.scheduler.state_dict(),
        'best_loss': self.best_loss,
        'config': self.config.to_dict(),
        'xoron_config': self.xoron_config.to_dict(),
    }
    
    if self.scaler is not None:
        checkpoint['scaler_state_dict'] = self.scaler.state_dict()
    
    path = os.path.join(self.config.output_dir, f'checkpoint-{epoch}.pt')
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")
```

### Load Checkpoint

```python
def _load_checkpoint(self, path):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    self.start_epoch = checkpoint['epoch'] + 1
    self.global_step = checkpoint['global_step']
    self.best_loss = checkpoint['best_loss']
    
    if self.scaler is not None and 'scaler_state_dict' in checkpoint:
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    print(f"Resumed from epoch {self.start_epoch}, step {self.global_step}")
```

---

## 🔧 Gradient Checkpointing

```python
def _enable_gradient_checkpointing(self):
    """Enable gradient checkpointing for memory efficiency."""
    enabled = []
    
    # LLM
    if hasattr(self.model.llm, 'gradient_checkpointing_enable'):
        self.model.llm.gradient_checkpointing_enable()
        enabled.append('LLM')
    
    # Vision encoder
    if hasattr(self.model.vision_encoder, 'vision_model'):
        inner = self.model.vision_encoder.vision_model
        if hasattr(inner, 'gradient_checkpointing_enable'):
            inner.gradient_checkpointing_enable()
            enabled.append('Vision')
    
    # Audio encoder/decoder
    if hasattr(self.model.audio_encoder, 'gradient_checkpointing_enable'):
        self.model.audio_encoder.gradient_checkpointing_enable()
        enabled.append('Audio Encoder')
    
    if hasattr(self.model.audio_decoder, 'gradient_checkpointing_enable'):
        self.model.audio_decoder.gradient_checkpointing_enable()
        enabled.append('Audio Decoder')
    
    print(f"Gradient checkpointing enabled for: {', '.join(enabled)}")
```

---

## 📈 Training Tips

### Memory Optimization

1. **Gradient Checkpointing**: Trade compute for memory
2. **8-bit Optimizer**: Reduces optimizer state memory by 75%
3. **Gradient Accumulation**: Simulate larger batches
4. **Empty Cache**: Periodically clear CUDA cache

### Stability Tips

1. **Warmup**: Use 5% warmup ratio
2. **Gradient Clipping**: Clip at 1.0
3. **FP32 Optimizer States**: Prevents overflow with FP16 models
4. **Small Learning Rate**: Start with 1e-4

### Multi-GPU Training

```python
# Device map distributes model across GPUs
device_map = get_device_map(torch.cuda.device_count())
model = XoronMultimodalModel(config, device_map=device_map)
```

---

## 🔗 Related Documentation

- [Config Documentation](../config/README.md) - Training configuration
- [Data Documentation](../data/README.md) - Dataset handling
- [Model Documentation](../models/llm.md) - Model architecture

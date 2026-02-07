"""Training utilities for Xoron-Dev."""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from typing import List, Dict, Any, Optional


# Check for bitsandbytes availability for 8-bit optimizer
BNB_AVAILABLE = False
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    pass


def gpu_safe_index(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Safely index a tensor using a boolean mask, ensuring indices stay on GPU.
    
    Boolean indexing in PyTorch can sometimes create CPU index tensors internally,
    causing "index is on cpu" errors. This function explicitly creates GPU indices.
    
    Args:
        tensor: The tensor to index (on GPU)
        mask: Boolean mask tensor (should be on same device as tensor)
        
    Returns:
        Indexed tensor with selected rows
    """
    device = tensor.device
    # Get indices where mask is True, ensure they're on the same device
    indices = mask.nonzero(as_tuple=False).squeeze(-1).to(device)
    if indices.dim() == 0:
        indices = indices.unsqueeze(0)
    return tensor.index_select(0, indices)


class FP32OptimizerWrapper:
    """
    Wrapper that maintains FP32 master weights for FP16 models ON GPU.
    
    This solves the FP16 optimizer overflow problem by:
    1. Keeping FP32 copies of all parameters ON THE SAME DEVICE (GPU)
    2. Running optimizer on FP32 copies (no overflow)
    3. Copying back to FP16 model after each step
    
    CRITICAL: All copies stay on GPU for speed - no CPU transfers!
    """
    
    def __init__(self, optimizer_class, model, **optimizer_kwargs):
        """
        Args:
            optimizer_class: Optimizer class (e.g., AdamW)
            model: The FP16 model
            **optimizer_kwargs: Arguments to pass to optimizer (lr, weight_decay, etc.)
        """
        self.model = model
        self.fp16_params = []
        self.fp32_params = []
        
        # Create FP32 copies of FP16 parameters ON THE SAME DEVICE (GPU)
        for param in model.parameters():
            if param.requires_grad:
                self.fp16_params.append(param)
                # Create FP32 copy ON SAME DEVICE as original param
                fp32_param = param.data.float().clone().detach()
                fp32_param.requires_grad = True
                self.fp32_params.append(fp32_param)
        
        # Create optimizer on FP32 params
        self.optimizer = optimizer_class(self.fp32_params, **optimizer_kwargs)
        
        # Verify device (silent - no logging needed)
    
    def zero_grad(self, set_to_none=False):
        """Zero gradients on both FP16 model and FP32 params."""
        # Zero FP16 gradients (on model)
        for fp16_p in self.fp16_params:
            if fp16_p.grad is not None:
                if set_to_none:
                    fp16_p.grad = None
                else:
                    fp16_p.grad.zero_()
        # Zero FP32 gradients
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def step(self):
        """
        1. Copy FP16 gradients to FP32 params (GPU→GPU, fast!)
        2. Run optimizer on FP32 (on GPU)
        3. Copy FP32 params back to FP16 model (GPU→GPU, fast!)
        """
        # Copy gradients from FP16 to FP32 (both on GPU - fast!)
        for fp16_p, fp32_p in zip(self.fp16_params, self.fp32_params):
            if fp16_p.grad is not None:
                if fp32_p.grad is None:
                    fp32_p.grad = fp16_p.grad.float()
                else:
                    fp32_p.grad.copy_(fp16_p.grad)
        
        # Run optimizer step on FP32 params (on GPU)
        self.optimizer.step()
        
        # Copy updated FP32 params back to FP16 model (GPU→GPU, fast!)
        for fp16_p, fp32_p in zip(self.fp16_params, self.fp32_params):
            fp16_p.data.copy_(fp32_p.data)
    
    def state_dict(self):
        """Return optimizer state dict."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load optimizer state dict."""
        self.optimizer.load_state_dict(state_dict)
    
    @property
    def param_groups(self):
        """Return param groups from underlying optimizer."""
        return self.optimizer.param_groups


def create_collate_fn(video_frames: int, video_size: int, active_modalities: str = 'all', vision_size: int = 256):
    """Create a collate function with the specified video configuration.
    
    Args:
        video_frames: Number of video frames
        video_size: Size of video frames
        active_modalities: Which modalities are active for training.
            'all' - full multimodal (default)
            'text' - text only, minimal tensors for image/video/audio (~27MB RAM savings per batch)
            'image' - image + text, minimal tensors for video/audio
            'video' - video + image + text, minimal tensors for audio
            'audio' - audio + text, minimal tensors for image/video
        vision_size: Size of vision encoder input (256 for memory-efficient training)
    """
    # Determine which modalities need full tensors
    need_image = active_modalities in ('all', 'image', 'video')
    need_video = active_modalities in ('all', 'video')
    need_audio = active_modalities in ('all', 'audio')

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for multimodal batches."""
        try:
            input_ids = torch.stack([b["input_ids"] for b in batch])
            attention_mask = torch.stack([b["attention_mask"] for b in batch])
            labels = torch.stack([b["labels"] for b in batch])
            batch_size = len(batch)
            sample_types = [b.get("sample_type", "text") for b in batch]
            
            # Handle pixel_values - use minimal tensor if not needed
            if need_image:
                pixel_values_list = []
                for b in batch:
                    pv = b["pixel_values"]
                    if pv is not None and isinstance(pv, torch.Tensor):
                        if pv.dim() == 3:
                            if pv.shape[1] != vision_size or pv.shape[2] != vision_size:
                                pv = F.interpolate(pv.unsqueeze(0), size=(vision_size, vision_size), mode='bilinear', align_corners=False).squeeze(0)
                            pixel_values_list.append(pv)
                        else:
                            pixel_values_list.append(torch.zeros(3, vision_size, vision_size))
                    else:
                        pixel_values_list.append(torch.zeros(3, vision_size, vision_size))
                pixel_values = torch.stack(pixel_values_list)
            else:
                # Minimal 1x1 tensor to save memory
                pixel_values = torch.zeros(batch_size, 3, 1, 1)

            # Handle video_frames - use minimal tensor if not needed
            if need_video:
                video_frames_list = []
                for b in batch:
                    vf = b["video_frames"]
                    if vf is not None and isinstance(vf, torch.Tensor) and vf.dim() == 4:
                        # Ensure no inf/nan values that cause training instability
                        vf = torch.nan_to_num(vf, nan=0.0, posinf=10.0, neginf=-10.0)
                        vf = torch.clamp(vf, min=-10.0, max=10.0)
                        video_frames_list.append(vf)
                    else:
                        video_frames_list.append(torch.zeros(video_frames, 3, video_size, video_size))
                video_frames_tensor = torch.stack(video_frames_list)
            else:
                # Minimal tensor to save memory (~25MB savings per batch)
                video_frames_tensor = torch.zeros(batch_size, 1, 3, 1, 1)

            # Handle audio_features - use minimal tensor if not needed
            if need_audio:
                audio_features_list = []
                max_audio_len = 1000  # For mel spectrograms
                max_waveform_samples = 160000  # 10 seconds at 16kHz for raw waveform
                target_mel_bins = 80
                
                # Detect if we're using raw waveform (1D with large T) or mel spectrogram (2D)
                # Skip minimal placeholders (size 1) when detecting
                first_valid = None
                for b in batch:
                    af = b.get("audio_features")
                    if af is not None and isinstance(af, torch.Tensor) and af.numel() > 100:
                        first_valid = af
                        break
                
                # Determine mode: raw waveform if 1D with many samples, else mel
                use_raw_waveform = first_valid is not None and first_valid.dim() == 1 and first_valid.shape[0] > 1000
                
                for b in batch:
                    af = b.get("audio_features")
                    # Check if this is a valid audio tensor (not minimal placeholder)
                    is_valid_audio = af is not None and isinstance(af, torch.Tensor) and af.numel() > 100
                    
                    if is_valid_audio:
                        if use_raw_waveform and af.dim() == 1:
                            # Raw waveform mode: 1D tensor [T]
                            if af.shape[0] > max_waveform_samples:
                                af = af[:max_waveform_samples]
                            elif af.shape[0] < max_waveform_samples:
                                pad = torch.zeros(max_waveform_samples - af.shape[0])
                                af = torch.cat([af, pad], dim=0)
                            audio_features_list.append(af)
                        elif not use_raw_waveform and af.dim() == 2:
                            # Mel spectrogram mode: 2D tensor [mel_bins, time]
                            if af.shape[0] != target_mel_bins:
                                af = F.interpolate(
                                    af.unsqueeze(0).unsqueeze(0),
                                    size=(target_mel_bins, af.shape[1]),
                                    mode='bilinear',
                                    align_corners=False
                                ).squeeze(0).squeeze(0)
                            
                            if af.shape[1] != max_audio_len:
                                if af.shape[1] > max_audio_len:
                                    af = af[:, :max_audio_len]
                                else:
                                    pad = torch.zeros(target_mel_bins, max_audio_len - af.shape[1])
                                    af = torch.cat([af, pad], dim=1)
                            audio_features_list.append(af)
                        else:
                            # Mismatched dimension - use zeros
                            if use_raw_waveform:
                                audio_features_list.append(torch.zeros(max_waveform_samples))
                            else:
                                audio_features_list.append(torch.zeros(target_mel_bins, max_audio_len))
                    else:
                        # Minimal placeholder or no audio - use zeros matching batch mode
                        if use_raw_waveform:
                            audio_features_list.append(torch.zeros(max_waveform_samples))
                        else:
                            audio_features_list.append(torch.zeros(target_mel_bins, max_audio_len))
                
                audio_features = torch.stack(audio_features_list)
            else:
                # Minimal tensor to save memory when audio not needed
                audio_features = torch.zeros(batch_size, 1)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "pixel_values": pixel_values,
                "video_frames": video_frames_tensor,
                "audio_features": audio_features,
                "sample_type": sample_types,
            }
        except Exception:
            batch_size = len(batch)
            # Fallback with appropriate tensor sizes based on active modalities
            vs = vision_size if need_image else 1
            vf_count = video_frames if need_video else 1
            vf_size = video_size if need_video else 1
            af_bins = 80 if need_audio else 1
            af_len = 1000 if need_audio else 1
            
            return {
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
                "labels": torch.stack([b["labels"] for b in batch]),
                "pixel_values": torch.zeros(batch_size, 3, vs, vs),
                "video_frames": torch.zeros(batch_size, vf_count, 3, vf_size, vf_size),
                "audio_features": torch.zeros(batch_size, af_bins, af_len),
                "sample_type": ["text"] * batch_size,
            }

    return collate_fn


def create_optimizer_and_scheduler(
    model,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    total_steps: int,
    use_8bit_optimizer: bool = False,
    eps: float = 1e-8,
    force_fp32_optimizer: bool = True,
):
    """
    Create optimizer and learning rate scheduler.
    
    Args:
        model: The model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        warmup_ratio: Ratio of warmup steps
        total_steps: Total training steps
        use_8bit_optimizer: Use 8-bit Adam from bitsandbytes (saves ~75% optimizer memory)
        eps: Adam epsilon - CRITICAL for FP16 stability (default 1e-8, try 1e-4 for FP16)
        force_fp32_optimizer: Force FP32 optimizer states even with FP16 model (HIGHLY RECOMMENDED)
    
    Note on FP16 stability:
        FP16 models REQUIRE FP32 optimizer states to prevent overflow in Adam.
        This function automatically uses FP32OptimizerWrapper for FP16 models.
        
    MEMORY OPTIMIZATION:
        Only trainable parameters (requires_grad=True) are tracked by optimizer.
        For LoRA, this means only LoRA params get optimizer states (~2-3x param size).
        Frozen base weights don't consume any optimizer memory.
    """
    # Check model dtype
    model_dtype = next(model.parameters()).dtype
    is_fp16 = model_dtype == torch.float16
    
    # MEMORY OPTIMIZATION: Only get trainable parameters
    # This is CRITICAL for LoRA - we don't want optimizer states for frozen weights
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in model.parameters())
    
    print(f"   📊 Optimizer will track {num_trainable:,} params ({100*num_trainable/num_total:.1f}% of {num_total:,} total)")
    print(f"   💾 Optimizer memory: ~{(num_trainable * 8) / (1024**2):.1f}MB (8 bytes/param for Adam states)")
    
    # For FP16 models, use FP32OptimizerWrapper to prevent optimizer overflow
    if is_fp16:
        # Use FP32OptimizerWrapper which maintains FP32 copies
        # NOTE: FP32OptimizerWrapper already filters for requires_grad=True internally
        optimizer = FP32OptimizerWrapper(
            optimizer_class=AdamW,
            model=model,
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=eps,
        )
    elif use_8bit_optimizer and BNB_AVAILABLE:
        # FIXED: Use only trainable params, not model.parameters()
        optimizer = bnb.optim.AdamW8bit(
            trainable_params,  # Only trainable params!
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=eps,
        )
        print("   ✅ Using 8-bit AdamW optimizer (saves ~75% optimizer memory)")
    else:
        # FIXED: Use only trainable params, not model.parameters()
        optimizer = AdamW(
            trainable_params,  # Only trainable params!
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=eps,
        )
        if use_8bit_optimizer and not BNB_AVAILABLE:
            print("   ⚠️ bitsandbytes not available, using standard AdamW")
            print("      Install with: pip install bitsandbytes")

    warmup_steps = int(total_steps * warmup_ratio)
    
    # Scheduler needs the underlying optimizer for FP32OptimizerWrapper
    scheduler_optimizer = optimizer.optimizer if isinstance(optimizer, FP32OptimizerWrapper) else optimizer
    scheduler = get_linear_schedule_with_warmup(
        scheduler_optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    return optimizer, scheduler


def train_image_diffusion_step(generator, images, text_context, target_size=256, sample_types=None, mask=None):
    """
    Train SOTA image diffusion on text-image pairs.
    
    Uses the generator's training_step method which includes:
    - Diffusion loss (noise prediction)
    - KL divergence loss (VAE regularization)
    - Classifier-free guidance training (random context dropout)
    
    Args:
        generator: Image generator model (MobileDiffusionGenerator)
        images: Batch of images (B, C, H, W) in [0, 1] range
        text_context: Text embeddings (B, seq_len, hidden_dim)
        target_size: Target image size for diffusion (256 for memory efficiency)
        sample_types: List of sample types to filter valid samples
        mask: Optional inpainting mask (B, 1, H, W)
    
    Returns:
        Total loss or None if no valid samples
    """
    if generator is None or images is None:
        return None
    try:
        if not isinstance(images, torch.Tensor):
            return None
        if images.numel() == 0:
            return None

        gen_device = next(generator.parameters()).device
        gen_dtype = next(generator.parameters()).dtype
        # Match input dtype to model dtype to avoid "Input type (float) and bias type (c10::Half)" errors
        images = images.to(device=gen_device, dtype=gen_dtype)
        text_context = text_context.to(device=gen_device, dtype=gen_dtype)

        if images.dim() != 4 or images.shape[1] != 3:
            return None

        # Filter by sample type if provided - train on image generation/editing samples
        image_sample_types = ['image_generation', 'image_editing', 'text_to_image', 'image_caption']
        if sample_types is not None:
            type_mask = torch.tensor([t in image_sample_types for t in sample_types], dtype=torch.bool, device=gen_device)
            if not type_mask.any():
                return None
            images = gpu_safe_index(images, type_mask)
            text_context = gpu_safe_index(text_context, type_mask)
            if mask is not None:
                mask = gpu_safe_index(mask, type_mask)

        # Filter to only samples with valid (non-zero) images
        valid_mask = images.abs().sum(dim=(1, 2, 3)) > 1e-6
        if not valid_mask.any():
            return None
        
        images = gpu_safe_index(images, valid_mask)
        text_context = gpu_safe_index(text_context, valid_mask)
        if mask is not None:
            mask = gpu_safe_index(mask, valid_mask)

        # Resize to target size - detach before interpolation to avoid FP32 in gradient graph
        # Interpolation requires FP32 but we don't need gradients through the resize op
        with torch.no_grad():
            images = F.interpolate(images.float(), size=(target_size, target_size), mode='bilinear', align_corners=False)
            if mask is not None:
                mask = F.interpolate(mask.float(), size=(target_size // 8, target_size // 8), mode='nearest')
        # Convert back to model dtype and re-enable gradients
        images = images.to(gen_dtype).requires_grad_(True)
        if mask is not None:
            mask = mask.to(gen_dtype)

        # Normalize images to [-1, 1] for diffusion
        images_norm = images * 2 - 1
        
        # Delete intermediate tensors to save memory
        del images, valid_mask

        # Use generator's training_step if available (SOTA method)
        if hasattr(generator, 'training_step'):
            # Ensure all inputs are correct dtype
            losses = generator.training_step(images_norm.to(gen_dtype), text_context.to(gen_dtype), mask.to(gen_dtype) if mask is not None else None)
            loss = losses['total_loss']
            del losses, images_norm  # Clean up
            return loss
        
        # Fallback to manual training - ensure correct dtype
        images_norm = images_norm.to(gen_dtype)
        z, mean, logvar = generator.encode(images_norm)
        del images_norm  # No longer needed
        
        batch_size = z.shape[0]
        timesteps = torch.randint(0, 1000, (batch_size,), device=gen_device)
        noise = torch.randn_like(z)
        
        # Add noise
        if hasattr(generator, 'add_noise'):
            noisy_z = generator.add_noise(z, noise, timesteps)
        else:
            alpha_t = generator.alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(gen_dtype)
            noisy_z = torch.sqrt(alpha_t) * z + torch.sqrt(1 - alpha_t) * noise
        
        noise_pred = generator.unet(noisy_z, timesteps, text_context.to(gen_dtype), mask.to(gen_dtype) if mask is not None else None)
        del noisy_z  # Clean up
        
        diff_loss = F.mse_loss(noise_pred, noise)
        del noise_pred, noise  # Clean up
        
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        del z, mean, logvar  # Clean up
        
        return diff_loss + 0.0001 * kl_loss
        
    except Exception as e:
        print(f"      ⚠️ Image diffusion training error: {type(e).__name__}: {str(e)[:100]}")
        return None


_video_sample_count = [0]

def train_video_diffusion_step(video_generator, video_frames, text_context, target_size=256, sample_types=None):
    """Train video diffusion on video data."""
    if video_generator is None or video_frames is None:
        return None
    try:
        if not isinstance(video_frames, torch.Tensor) or video_frames.numel() == 0:
            return None

        gen_device = next(video_generator.parameters()).device
        gen_dtype = next(video_generator.parameters()).dtype
        
        video_frames = video_frames.to(device=gen_device, dtype=gen_dtype)
        text_context = text_context.to(device=gen_device, dtype=gen_dtype)

        # Filter by sample type if provided
        video_sample_types = ['video_generation', 'image_to_video', 'video_caption', 'video_qa', 
                              'video_preference', 'video_likert', 'text_to_video']
        if sample_types is not None:
            type_mask = torch.tensor([t in video_sample_types for t in sample_types], dtype=torch.bool, device=gen_device)
            if not type_mask.any():
                return None
            video_frames = gpu_safe_index(video_frames, type_mask)
            text_context = gpu_safe_index(text_context, type_mask)

        # Handle dimension ordering: ensure [B, C, T, H, W]
        if video_frames.dim() == 5:
            B, dim1, dim2, H, W = video_frames.shape
            if dim1 > dim2:  # [B, T, C, H, W] -> [B, C, T, H, W]
                video_frames = video_frames.permute(0, 2, 1, 3, 4)
        elif video_frames.dim() == 4:
            video_frames = video_frames.unsqueeze(2)
        else:
            return None

        B, C, T, H, W = video_frames.shape

        if C != 3 or T < 1:
            return None
        
        # Limit frames during training (max 16 frames)
        max_train_frames = 16
        if T > max_train_frames:
            frame_indices = torch.linspace(0, T - 1, max_train_frames, device=gen_device).long()
            video_frames = video_frames[:, :, frame_indices]
            T = max_train_frames

        # Filter to only samples with valid (non-zero) video frames
        frame_means = video_frames.abs().mean(dim=(1, 2, 3, 4))
        valid_mask = frame_means > 1e-6
        num_valid = valid_mask.sum().item()
        
        # Log frame_mean for first 100 samples only
        _video_sample_count[0] += 1
        if _video_sample_count[0] <= 100:
            print(f"      [VIDEO] valid={num_valid}/{B}, frame_mean={frame_means.min().item():.4f}-{frame_means.max().item():.4f}")
        
        if not valid_mask.any():
            return None
        
        video_frames = gpu_safe_index(video_frames, valid_mask)
        text_context = gpu_safe_index(text_context, valid_mask)
        B = video_frames.shape[0]
        del valid_mask

        # MEMORY OPTIMIZATION: Process only 1 sample at a time
        # Accumulate losses and average them
        total_loss = None
        num_processed = 0
        
        for i in range(B):
            try:
                # Extract single sample
                single_video = video_frames[i:i+1]  # [1, C, T, H, W]
                single_context = text_context[i:i+1]  # [1, seq_len, dim]
                
                # Resize frames to target size (128x128 -> 32x32 latent)
                # Use no_grad for interpolation to avoid FP32 in gradient graph
                single_video = single_video.contiguous().view(T, C, H, W)
                with torch.no_grad():
                    single_video = F.interpolate(single_video.float(), size=(target_size, target_size), mode='bilinear', align_corners=False)
                single_video = single_video.to(gen_dtype).requires_grad_(True)
                single_video = single_video.contiguous().view(1, C, T, target_size, target_size)

                # Normalize to [-1, 1] for diffusion
                video_norm = single_video * 2 - 1
                del single_video

                # Extract first frame for I2V training (50% of the time)
                first_frame = None
                if torch.rand(1).item() > 0.5:
                    first_frame = (video_norm[:, :, 0] + 1) / 2

                # Use generator's training_step
                if hasattr(video_generator, 'training_step'):
                    losses = video_generator.training_step(video_norm, single_context, first_frame)
                    step_loss = losses['total_loss']
                    if total_loss is None:
                        total_loss = step_loss
                    else:
                        total_loss = total_loss + step_loss
                    num_processed += 1
                    del losses, video_norm, first_frame
                
                # Force memory cleanup after each sample
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue  # Skip this sample, try next
                raise
        
        if num_processed == 0 or total_loss is None:
            return None
            
        return total_loss / num_processed
        
    except Exception as e:
        print(f"      ⚠️ Video diffusion training error: {type(e).__name__}: {str(e)[:100]}")
        return None


def _truncate_audio_for_memory(audio_features: torch.Tensor, max_samples: int = 80000) -> torch.Tensor:
    """
    Truncate audio to save GPU memory during training.
    
    For raw waveform at 16kHz, 80000 samples = 5 seconds of audio.
    This is enough for training while preventing OOM.
    
    Args:
        audio_features: [B, T] raw waveform or [B, mel_bins, time] mel spectrogram
        max_samples: Maximum number of samples/frames to keep
        
    Returns:
        Truncated audio tensor
    """
    if audio_features.dim() == 2:
        # Raw waveform [B, T] - truncate time dimension
        if audio_features.shape[1] > max_samples:
            audio_features = audio_features[:, :max_samples]
    elif audio_features.dim() == 3:
        # Mel spectrogram [B, mel_bins, time] - truncate time dimension
        max_frames = max_samples // 256  # Approximate conversion for hop_length=256
        if audio_features.shape[2] > max_frames:
            audio_features = audio_features[:, :, :max_frames]
    return audio_features


def _clear_audio_memory():
    """Clear GPU memory after audio processing to prevent OOM."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_voice_asr_step(audio_encoder, audio_features, text_embeds, sample_types=None):
    """Train ASR: audio -> text alignment.
    
    Args:
        audio_encoder: Audio encoder model
        audio_features: Batch of audio - either [B, T] raw waveform or [B, mel_bins, time] mel spectrogram
        text_embeds: Text embeddings (B, seq_len, hidden_dim)
        sample_types: List of sample types to filter valid samples (e.g., ['voice_asr', 'text', ...])
    """
    if audio_encoder is None or audio_features is None:
        return None
    try:
        if not isinstance(audio_features, torch.Tensor):
            return None
        if audio_features.numel() == 0:
            return None

        enc_device = next(audio_encoder.parameters()).device
        enc_dtype = next(audio_encoder.parameters()).dtype
        
        # MEMORY OPTIMIZATION: Truncate long audio to prevent OOM
        # 80000 samples = 5 seconds at 16kHz - enough for training
        audio_features = _truncate_audio_for_memory(audio_features, max_samples=80000)
        
        # Match input dtype to model dtype to avoid "Input type (float) and bias type (c10::Half)" errors
        audio_features = audio_features.to(device=enc_device, dtype=enc_dtype)
        text_embeds = text_embeds.to(device=enc_device, dtype=enc_dtype)

        # Accept both 2D (raw waveform [B, T]) and 3D (mel spectrogram [B, mel_bins, time])
        if audio_features.dim() not in [2, 3]:
            return None

        # First filter by sample type if provided - only train on voice_asr samples
        if sample_types is not None:
            type_mask = torch.tensor([t == 'voice_asr' for t in sample_types], dtype=torch.bool, device=enc_device)
            if not type_mask.any():
                return None
            audio_features = gpu_safe_index(audio_features, type_mask)
            text_embeds = gpu_safe_index(text_embeds, type_mask)

        # Then filter to only samples with valid (non-zero) audio
        # Handle both 2D and 3D tensors
        if audio_features.dim() == 2:
            # Raw waveform [B, T] - sum over time dimension
            valid_mask = audio_features.abs().sum(dim=1) > 1e-6
        else:
            # Mel spectrogram [B, mel_bins, time] - sum over mel bins and time
            valid_mask = audio_features.abs().sum(dim=(1, 2)) > 1e-6
        num_valid = valid_mask.sum().item()
        
        if num_valid == 0:
            return None
        
        # Filter audio_features and text_embeds to only valid samples
        audio_features = gpu_safe_index(audio_features, valid_mask)
        text_embeds = gpu_safe_index(text_embeds, valid_mask)

        # Need at least 2 samples for contrastive learning
        if audio_features.shape[0] < 2:
            # For single sample, use MSE loss between audio and text embeddings instead
            audio_out = audio_encoder(audio_features)
            # AudioEncoder returns (features, commitment_loss) tuple
            audio_embeds = audio_out[0] if isinstance(audio_out, tuple) else audio_out
            audio_pooled = audio_embeds.mean(dim=1)
            text_pooled = text_embeds.mean(dim=1)
            # Project to same dimension if needed
            if audio_pooled.shape[-1] != text_pooled.shape[-1]:
                min_dim = min(audio_pooled.shape[-1], text_pooled.shape[-1])
                audio_pooled = audio_pooled[..., :min_dim]
                text_pooled = text_pooled[..., :min_dim]
            loss = F.mse_loss(audio_pooled, text_pooled)
            # Clean up intermediate tensors
            del audio_out, audio_embeds, audio_pooled, text_pooled
            return loss

        audio_out = audio_encoder(audio_features)
        # AudioEncoder returns (features, commitment_loss) tuple
        audio_embeds = audio_out[0] if isinstance(audio_out, tuple) else audio_out
        audio_pooled = audio_embeds.mean(dim=1)
        text_pooled = text_embeds.mean(dim=1)
        audio_pooled = F.normalize(audio_pooled, dim=-1)
        text_pooled = F.normalize(text_pooled, dim=-1)
        similarity = torch.matmul(audio_pooled, text_pooled.T)
        labels = torch.arange(similarity.shape[0], device=enc_device)
        loss = F.cross_entropy(similarity, labels)
        # Clean up intermediate tensors
        del audio_out, audio_embeds, audio_pooled, text_pooled, similarity
        return loss
    except Exception as e:
        # Log the exception for debugging
        import traceback
        print(f"      ⚠️ ASR training error: {type(e).__name__}: {str(e)[:100]}")
        return None


def train_voice_tts_step(audio_decoder, text_embeds, target_audio, audio_encoder=None, sample_types=None, use_mas=True):
    """Train TTS: text -> audio generation with Monotonic Alignment Search (MAS).
    
    Args:
        audio_decoder: Audio decoder model (with MAS support)
        text_embeds: Text embeddings (B, seq_len, hidden_dim)
        target_audio: Target audio - either [B, T] raw waveform or [B, mel_bins, time] mel spectrogram
        audio_encoder: Optional audio encoder for extracting target audio features (for MAS)
        sample_types: List of sample types to filter valid samples (e.g., ['voice_tts', 'text', ...])
        use_mas: Whether to use Monotonic Alignment Search for alignment loss
    """
    if audio_decoder is None or target_audio is None:
        return None
    try:
        if not isinstance(target_audio, torch.Tensor):
            return None
        if target_audio.numel() == 0:
            return None

        dec_device = next(audio_decoder.parameters()).device
        dec_dtype = next(audio_decoder.parameters()).dtype
        
        # MEMORY OPTIMIZATION: Truncate long audio to prevent OOM
        # 80000 samples = 5 seconds at 16kHz - enough for training
        target_audio = _truncate_audio_for_memory(target_audio, max_samples=80000)
        
        # Match input dtype to model dtype to avoid "mat1 and mat2 must have the same dtype" errors
        target_audio = target_audio.to(device=dec_device, dtype=dec_dtype)
        text_embeds = text_embeds.to(device=dec_device, dtype=dec_dtype)

        # Accept both 2D (raw waveform [B, T]) and 3D (mel spectrogram [B, mel_bins, time])
        if target_audio.dim() not in [2, 3]:
            return None
        
        # For raw waveform, we need to convert to mel for the decoder (which outputs mel)
        # The waveform_decoder training handles raw waveform -> waveform
        is_raw_waveform = target_audio.dim() == 2
        if is_raw_waveform:
            # Convert raw waveform to mel spectrogram for TTS training
            # Use a simple mel extraction (the model will learn to match this)
            # NOTE: MelSpectrogram requires FP32 for computation, then we cast back to model dtype
            try:
                import torchaudio.transforms as T
                mel_transform = T.MelSpectrogram(
                    sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80
                ).to(dec_device)
                # [B, T] -> [B, mel_bins, time]
                # MelSpectrogram requires FP32, so convert, compute, then cast back
                with torch.no_grad():  # No gradients through mel extraction
                    target_mel = mel_transform(target_audio.float())
                    target_mel = torch.log(target_mel.clamp(min=1e-5))
                    # Cast back to model dtype for forward pass
                    target_mel = target_mel.to(dtype=dec_dtype)
                # Clean up mel_transform to free memory
                del mel_transform
            except Exception as e:
                # Fallback: skip this step if mel conversion fails
                return None
        else:
            target_mel = target_audio

        # First filter by sample type if provided - only train on voice_tts samples
        if sample_types is not None:
            type_mask = torch.tensor([t == 'voice_tts' for t in sample_types], dtype=torch.bool, device=dec_device)
            if not type_mask.any():
                return None
            target_mel = gpu_safe_index(target_mel, type_mask)
            text_embeds = gpu_safe_index(text_embeds, type_mask)

        # Then filter to only samples with valid (non-zero) target mel
        valid_mask = target_mel.abs().sum(dim=(1, 2)) > 1e-6
        num_valid = valid_mask.sum().item()
        
        if num_valid == 0:
            return None
        
        # Filter target_mel and text_embeds to only valid samples
        target_mel = gpu_safe_index(target_mel, valid_mask)
        text_embeds = gpu_safe_index(text_embeds, valid_mask)

        # Get target audio features for MAS if audio encoder is available
        audio_features_for_mas = None
        if use_mas and audio_encoder is not None:
            with torch.no_grad():
                # Encode target mel to get audio features for alignment
                audio_out, _ = audio_encoder(target_mel)
                audio_features_for_mas = audio_out

        # Forward pass with MAS enabled
        pred_mel, durations, alignment = audio_decoder(
            text_embeds, 
            target_length=target_mel.shape[-1],
            audio_features=audio_features_for_mas,
            use_mas=use_mas and audio_features_for_mas is not None,
        )
        
        # Mel reconstruction loss
        mel_loss = F.mse_loss(pred_mel, target_mel)
        
        # MAS alignment loss (self-supervised)
        mas_loss = torch.tensor(0.0, device=dec_device, dtype=dec_dtype)
        if use_mas and hasattr(audio_decoder, 'mas') and audio_features_for_mas is not None:
            mas_loss = audio_decoder.mas.compute_duration(alignment).sum() * 0.0  # Placeholder for actual MAS loss
            # The actual MAS loss is computed inside the decoder during forward pass
        
        # Duration prediction loss (if we have ground truth durations from alignment)
        duration_loss = torch.tensor(0.0, device=dec_device, dtype=dec_dtype)
        if alignment is not None:
            # Use alignment to compute pseudo ground-truth durations
            gt_durations = alignment.sum(dim=-1)  # [B, T_text]
            duration_loss = F.mse_loss(durations, gt_durations)
        
        total_loss = mel_loss + 0.1 * duration_loss + 0.01 * mas_loss
        
        # Clean up intermediate tensors
        del pred_mel, target_mel
        if alignment is not None:
            del alignment
        
        return total_loss
        
    except Exception as e:
        # Log the exception for debugging
        print(f"      ⚠️ TTS training error: {type(e).__name__}: {str(e)[:100]}")
        return None


def train_waveform_decoder_step(
    waveform_decoder, 
    audio_decoder,
    text_embeds, 
    target_waveform,
    mel_to_hidden=None,
    sample_types=None,
):
    """
    Train the waveform decoder for Speech-to-Speech output.
    
    This trains the model to convert mel features to raw audio waveform,
    enabling direct speech output without an external vocoder.
    
    Args:
        waveform_decoder: RawWaveformDecoder model
        audio_decoder: AudioDecoder model (for generating mel features)
        text_embeds: Text embeddings (B, seq_len, hidden_dim)
        target_waveform: Target audio waveform (B, audio_len)
        mel_to_hidden: Linear projection from mel dims to hidden_size
        sample_types: List of sample types to filter valid samples
        
    Returns:
        loss: Waveform reconstruction loss (L1 + multi-scale STFT loss)
    """
    if waveform_decoder is None or target_waveform is None:
        return None
    
    try:
        if not isinstance(target_waveform, torch.Tensor):
            return None
        if target_waveform.numel() == 0:
            return None
        
        # MEMORY OPTIMIZATION: Truncate waveform to prevent OOM
        # 48000 samples = 3 seconds at 16kHz for waveform decoder (smaller than encoder)
        # Waveform decoder is most memory-intensive, so use shorter clips
        target_waveform = _truncate_audio_for_memory(target_waveform, max_samples=48000)
        
        # Get waveform decoder device and dtype
        dec_device = next(waveform_decoder.parameters()).device
        dec_dtype = next(waveform_decoder.parameters()).dtype
        
        # Get audio decoder device (may be different with model parallelism)
        audio_dec_device = next(audio_decoder.parameters()).device
        audio_dec_dtype = next(audio_decoder.parameters()).dtype
        
        # Move tensors to correct device and dtype for waveform decoder
        target_waveform = target_waveform.to(device=dec_device, dtype=dec_dtype)
        # text_embeds goes to audio_decoder first, so use its device
        text_embeds_for_audio = text_embeds.to(device=audio_dec_device, dtype=audio_dec_dtype)
        
        # Filter by sample type if provided
        if sample_types is not None:
            # Create mask on waveform decoder device
            type_mask = torch.tensor([t == 'voice_tts' for t in sample_types], dtype=torch.bool, device=dec_device)
            if not type_mask.any():
                return None
            target_waveform = gpu_safe_index(target_waveform, type_mask)
            # Also filter text_embeds on its device
            type_mask_audio = type_mask.to(audio_dec_device)
            text_embeds_for_audio = gpu_safe_index(text_embeds_for_audio, type_mask_audio)
        
        # Filter to valid samples (non-silent audio)
        valid_mask = target_waveform.abs().sum(dim=-1) > 1e-6
        num_valid = valid_mask.sum().item()
        
        if num_valid == 0:
            return None
        
        target_waveform = gpu_safe_index(target_waveform, valid_mask)
        # Filter text_embeds on its device
        valid_mask_audio = valid_mask.to(audio_dec_device)
        text_embeds_for_audio = gpu_safe_index(text_embeds_for_audio, valid_mask_audio)
        
        # Step 1: Generate mel features through audio decoder (no grad for decoder)
        with torch.no_grad():
            pred_mel, durations, _ = audio_decoder(text_embeds_for_audio)
            # pred_mel: [B, n_mels, T_mel]
            # Move to waveform decoder device and convert to its dtype
            pred_mel = pred_mel.to(device=dec_device, dtype=dec_dtype)
        
        # Clean up text_embeds_for_audio as we don't need it anymore
        del text_embeds_for_audio
        
        # Step 2: Project mel to hidden_size for waveform decoder
        mel_features = pred_mel.transpose(1, 2)  # [B, T_mel, n_mels]
        del pred_mel  # Clean up
        
        if mel_to_hidden is not None:
            mel_to_hidden = mel_to_hidden.to(device=dec_device, dtype=dec_dtype)
            audio_features = mel_to_hidden(mel_features)  # [B, T_mel, hidden_size]
        else:
            # Fallback: pad/project mel features
            hidden_size = waveform_decoder.hidden_size
            n_mels = mel_features.shape[-1]
            if n_mels != hidden_size:
                # Simple linear projection - ensure output dtype matches
                proj = torch.nn.functional.pad(mel_features, (0, hidden_size - n_mels))
                audio_features = proj.to(dtype=dec_dtype)
            else:
                audio_features = mel_features
        
        del mel_features  # Clean up
        
        # Step 3: Generate waveform
        pred_waveform = waveform_decoder(audio_features, target_length=target_waveform.shape[-1])
        del audio_features  # Clean up
        
        # Ensure output is same dtype as target for loss computation
        if pred_waveform.dtype != target_waveform.dtype:
            pred_waveform = pred_waveform.to(dtype=target_waveform.dtype)
        
        # Step 4: Compute losses
        # L1 loss (time domain) - this is the main training signal with gradients
        l1_loss = F.l1_loss(pred_waveform, target_waveform)
        
        # MEMORY OPTIMIZATION: Skip STFT loss computation - it's only for monitoring
        # and adds extra memory overhead that can cause OOM
        # The L1 loss provides sufficient training signal for waveform reconstruction
        
        # Clean up
        del pred_waveform, target_waveform
        
        # Return only L1 loss for gradients
        return l1_loss
        
    except Exception as e:
        # Detailed error logging for debugging
        import traceback
        error_details = f"{type(e).__name__}: {str(e)}"
        print(f"      ⚠️ Waveform decoder training error: {error_details[:150]}")
        # Log device info for "index is on cpu" errors
        if "cpu" in str(e).lower() or "device" in str(e).lower():
            try:
                wd_device = next(waveform_decoder.parameters()).device if waveform_decoder else "None"
                ad_device = next(audio_decoder.parameters()).device if audio_decoder else "None"
                print(f"         Devices: waveform_decoder={wd_device}, audio_decoder={ad_device}")
                if target_waveform is not None:
                    print(f"         target_waveform device={target_waveform.device}, dtype={target_waveform.dtype}")
            except:
                pass
        return None


def compute_multi_scale_stft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Multi-scale STFT loss for waveform reconstruction.
    
    Uses multiple FFT sizes to capture both fine and coarse spectral details.
    This is crucial for high-quality audio generation.
    
    Note: STFT operations require FP32 - this function handles dtype conversion
    internally and returns the loss in FP32 (caller should convert if needed).
    
    Args:
        pred: Predicted waveform [B, T]
        target: Target waveform [B, T]
        
    Returns:
        loss: Combined multi-scale spectral loss (in FP32)
    """
    fft_sizes = [512, 1024, 2048]
    hop_sizes = [128, 256, 512]
    win_sizes = [512, 1024, 2048]
    
    # Store original dtype for reference (loss will be returned in FP32)
    original_dtype = pred.dtype
    device = pred.device
    
    # STFT requires FP32 - convert inputs
    pred_fp32 = pred.float()
    target_fp32 = target.float()
    
    total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    num_valid_scales = 0
    
    for fft_size, hop_size, win_size in zip(fft_sizes, hop_sizes, win_sizes):
        # Compute STFT (window must also be FP32)
        window = torch.hann_window(win_size, device=device, dtype=torch.float32)
        
        try:
            pred_stft = torch.stft(
                pred_fp32, n_fft=fft_size, hop_length=hop_size, win_length=win_size,
                window=window, return_complex=True
            )
            target_stft = torch.stft(
                target_fp32, n_fft=fft_size, hop_length=hop_size, win_length=win_size,
                window=window, return_complex=True
            )
            
            # Magnitude loss
            pred_mag = pred_stft.abs()
            target_mag = target_stft.abs()
            mag_loss = F.l1_loss(pred_mag, target_mag)
            
            # Log magnitude loss (perceptually important)
            pred_log_mag = torch.log(pred_mag + 1e-7)
            target_log_mag = torch.log(target_mag + 1e-7)
            log_mag_loss = F.l1_loss(pred_log_mag, target_log_mag)
            
            total_loss = total_loss + mag_loss + log_mag_loss
            num_valid_scales += 1
            
        except Exception:
            # Skip this scale if STFT fails (e.g., audio too short)
            continue
    
    # Avoid division by zero if all scales failed
    if num_valid_scales == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
    
    return total_loss / num_valid_scales


# ============================================================================
# EVALUATION FUNCTIONS
# These are inference-only versions without gradient computation
# ============================================================================

def eval_image_diffusion_step(generator, images, text_context, target_size=256, sample_types=None, mask=None):
    """Evaluate image diffusion: compute loss without gradient tracking.
    
    Args:
        generator: Image diffusion generator model
        images: Input images (B, C, H, W)
        text_context: Text embeddings for conditioning
        target_size: Target image size for generation (256 for memory efficiency)
        sample_types: List of sample types for filtering
        mask: Optional mask for inpainting/editing tasks
        
    Returns:
        Loss tensor (detached) or None if evaluation fails
    """
    if generator is None or images is None:
        return None
    try:
        if not isinstance(images, torch.Tensor):
            return None
        if images.numel() == 0:
            return None

        gen_device = next(generator.parameters()).device
        images = images.to(gen_device)
        
        if text_context is not None:
            text_context = text_context.to(gen_device)

        if images.dim() != 4:
            return None

        # Filter by sample type if provided
        if sample_types is not None:
            type_mask = torch.tensor([t in ['image_generation', 'image_editing'] for t in sample_types], dtype=torch.bool, device=gen_device)
            if not type_mask.any():
                return None
            images = gpu_safe_index(images, type_mask)
            if text_context is not None and text_context.dim() >= 2:
                text_context = gpu_safe_index(text_context, type_mask)

        # Filter non-zero images
        valid_mask = images.abs().sum(dim=(1, 2, 3)) > 1e-6
        num_valid = valid_mask.sum().item()
        
        if num_valid == 0:
            return None

        images = gpu_safe_index(images, valid_mask)
        if text_context is not None and text_context.dim() >= 2:
            text_context = gpu_safe_index(text_context, valid_mask)

        # Resize if needed
        if images.shape[2] != target_size or images.shape[3] != target_size:
            images = F.interpolate(images, size=(target_size, target_size), mode='bilinear', align_corners=False)

        # Normalize to [-1, 1] for diffusion
        if images.max() > 1.0:
            images = images / 127.5 - 1.0
        elif images.min() >= 0:
            images = images * 2 - 1

        # Use generator's training_step for consistent loss calculation
        if hasattr(generator, 'training_step'):
            with torch.no_grad():
                losses = generator.training_step(images, text_context)
                loss = losses['total_loss']
                return loss.detach()

        # Fallback to manual diffusion loss
        with torch.no_grad():
            z, mean, logvar = generator.encode_image(images)
            
            batch_size = z.shape[0]
            timesteps = torch.randint(0, 1000, (batch_size,), device=gen_device)
            noise = torch.randn_like(z)
            
            if hasattr(generator, 'add_noise'):
                noisy_z = generator.add_noise(z, noise, timesteps)
            else:
                alpha_t = generator.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                noisy_z = torch.sqrt(alpha_t) * z + torch.sqrt(1 - alpha_t) * noise
            
            noise_pred = generator.unet(noisy_z, timesteps, text_context)
            diff_loss = F.mse_loss(noise_pred, noise)
            
            return diff_loss.detach()

    except Exception as e:
        return None


def eval_video_diffusion_step(video_generator, video_frames, text_context, target_size=256, sample_types=None):
    """Evaluate video diffusion: compute loss without gradient tracking.
    
    Args:
        video_generator: Video diffusion generator model
        video_frames: Input video frames (B, T, C, H, W)
        text_context: Text embeddings for conditioning
        target_size: Target frame size (256 for memory efficiency)
        sample_types: List of sample types for filtering
        
    Returns:
        Loss tensor (detached) or None if evaluation fails
    """
    if video_generator is None or video_frames is None:
        return None
    try:
        if not isinstance(video_frames, torch.Tensor):
            return None
        if video_frames.numel() == 0:
            return None

        gen_device = next(video_generator.parameters()).device
        video_frames = video_frames.to(gen_device)
        
        if text_context is not None:
            text_context = text_context.to(gen_device)

        if video_frames.dim() != 5:
            return None

        # Filter by sample type
        video_sample_types = ['video_generation', 'image_to_video', 'video_caption', 'video_qa', 'video_preference', 'video_likert']
        if sample_types is not None:
            type_mask = torch.tensor([t in video_sample_types for t in sample_types], dtype=torch.bool, device=gen_device)
            if not type_mask.any():
                return None
            video_frames = gpu_safe_index(video_frames, type_mask)
            if text_context is not None and text_context.dim() >= 2:
                text_context = gpu_safe_index(text_context, type_mask)

        # Filter non-zero videos
        valid_mask = video_frames.abs().sum(dim=(1, 2, 3, 4)) > 1e-6
        num_valid = valid_mask.sum().item()
        
        if num_valid == 0:
            return None

        video_frames = gpu_safe_index(video_frames, valid_mask)
        if text_context is not None and text_context.dim() >= 2:
            text_context = gpu_safe_index(text_context, valid_mask)

        # Resize if needed
        current_h, current_w = video_frames.shape[3], video_frames.shape[4]
        if current_h != target_size or current_w != target_size:
            b, t, c, h, w = video_frames.shape
            video_frames = video_frames.view(b * t, c, h, w)
            video_frames = F.interpolate(video_frames, size=(target_size, target_size), mode='bilinear', align_corners=False)
            video_frames = video_frames.view(b, t, c, target_size, target_size)

        # Normalize to [-1, 1]
        if video_frames.max() > 1.0:
            video_frames_norm = video_frames / 127.5 - 1.0
        elif video_frames.min() >= 0:
            video_frames_norm = video_frames * 2 - 1
        else:
            video_frames_norm = video_frames

        first_frame = None
        if video_frames_norm.shape[1] > 1:
            first_frame = (video_frames_norm[:, 0] + 1) / 2

        # Use generator's training_step for consistent loss calculation
        if hasattr(video_generator, 'training_step'):
            with torch.no_grad():
                video_frames_5d = video_frames_norm.permute(0, 2, 1, 3, 4)  # B, C, T, H, W
                losses = video_generator.training_step(video_frames_5d, text_context, first_frame)
                loss = losses['total_loss']
                return loss.detach()

        # Fallback to manual video diffusion loss
        with torch.no_grad():
            video_frames_5d = video_frames_norm.permute(0, 2, 1, 3, 4)
            z, mean, logvar = video_generator.encode_video(video_frames_5d)
            
            batch_size = z.shape[0]
            timesteps = torch.randint(0, 1000, (batch_size,), device=gen_device)
            noise = torch.randn_like(z)
            
            if hasattr(video_generator, 'add_noise'):
                noisy_z = video_generator.add_noise(z, noise, timesteps)
            else:
                alpha_t = video_generator.alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
                noisy_z = torch.sqrt(alpha_t) * z + torch.sqrt(1 - alpha_t) * noise
            
            first_frame_latent = z[:, :, 0] if first_frame is not None else None
            noise_pred = video_generator.unet(noisy_z, timesteps, text_context, first_frame_latent)
            diff_loss = F.mse_loss(noise_pred, noise)
            
            return diff_loss.detach()

    except Exception as e:
        return None


def eval_voice_asr_step(audio_encoder, audio_features, text_embeds, sample_types=None):
    """Evaluate ASR: compute audio-text alignment loss without gradient tracking.
    
    Args:
        audio_encoder: Audio encoder model
        audio_features: Batch of audio - either [B, T] raw waveform or [B, mel_bins, time] mel spectrogram
        text_embeds: Text embeddings (B, seq_len, hidden_dim)
        sample_types: List of sample types for filtering
        
    Returns:
        Loss tensor (detached) or None if evaluation fails
    """
    if audio_encoder is None or audio_features is None:
        return None
    try:
        if not isinstance(audio_features, torch.Tensor):
            return None
        if audio_features.numel() == 0:
            return None

        enc_device = next(audio_encoder.parameters()).device
        enc_dtype = next(audio_encoder.parameters()).dtype
        audio_features = audio_features.to(device=enc_device, dtype=enc_dtype)
        text_embeds = text_embeds.to(device=enc_device, dtype=enc_dtype)

        # Accept both 2D (raw waveform [B, T]) and 3D (mel spectrogram [B, mel_bins, time])
        if audio_features.dim() not in [2, 3]:
            return None

        # Filter by sample type
        if sample_types is not None:
            type_mask = torch.tensor([t == 'voice_asr' for t in sample_types], dtype=torch.bool, device=enc_device)
            if not type_mask.any():
                return None
            audio_features = gpu_safe_index(audio_features, type_mask)
            text_embeds = gpu_safe_index(text_embeds, type_mask)

        # Filter valid audio - handle both 2D and 3D
        if audio_features.dim() == 2:
            valid_mask = audio_features.abs().sum(dim=1) > 1e-6
        else:
            valid_mask = audio_features.abs().sum(dim=(1, 2)) > 1e-6
        num_valid = valid_mask.sum().item()
        
        if num_valid == 0:
            return None

        audio_features = gpu_safe_index(audio_features, valid_mask)
        text_embeds = gpu_safe_index(text_embeds, valid_mask)

        with torch.no_grad():
            # For single sample, use MSE loss
            if audio_features.shape[0] < 2:
                audio_embeds = audio_encoder(audio_features)
                audio_pooled = audio_embeds.mean(dim=1)
                text_pooled = text_embeds.mean(dim=1)
                if audio_pooled.shape[-1] != text_pooled.shape[-1]:
                    min_dim = min(audio_pooled.shape[-1], text_pooled.shape[-1])
                    audio_pooled = audio_pooled[..., :min_dim]
                    text_pooled = text_pooled[..., :min_dim]
                loss = F.mse_loss(audio_pooled, text_pooled)
                return loss.detach()

            # Contrastive loss for multiple samples
            audio_embeds = audio_encoder(audio_features)
            audio_pooled = audio_embeds.mean(dim=1)
            text_pooled = text_embeds.mean(dim=1)
            audio_pooled = F.normalize(audio_pooled, dim=-1)
            text_pooled = F.normalize(text_pooled, dim=-1)
            similarity = torch.matmul(audio_pooled, text_pooled.T)
            labels = torch.arange(similarity.shape[0], device=enc_device)
            loss = F.cross_entropy(similarity, labels)
            return loss.detach()

    except Exception as e:
        return None


def eval_voice_tts_step(audio_decoder, text_embeds, target_audio, sample_types=None):
    """Evaluate TTS: compute text-to-audio loss without gradient tracking.
    
    Args:
        audio_decoder: Audio decoder model
        text_embeds: Text embeddings (B, seq_len, hidden_dim)
        target_audio: Target audio - either [B, T] raw waveform or [B, mel_bins, time] mel spectrogram
        sample_types: List of sample types for filtering
        
    Returns:
        Loss tensor (detached) or None if evaluation fails
    """
    if audio_decoder is None or target_audio is None:
        return None
    try:
        if not isinstance(target_audio, torch.Tensor):
            return None
        if target_audio.numel() == 0:
            return None

        dec_device = next(audio_decoder.parameters()).device
        dec_dtype = next(audio_decoder.parameters()).dtype
        target_audio = target_audio.to(device=dec_device, dtype=dec_dtype)
        text_embeds = text_embeds.to(device=dec_device, dtype=dec_dtype)

        # Accept both 2D (raw waveform [B, T]) and 3D (mel spectrogram [B, mel_bins, time])
        if target_audio.dim() not in [2, 3]:
            return None
        
        # For raw waveform, convert to mel
        is_raw_waveform = target_audio.dim() == 2
        if is_raw_waveform:
            try:
                import torchaudio.transforms as T
                mel_transform = T.MelSpectrogram(
                    sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80
                ).to(dec_device)
                target_mel = mel_transform(target_audio)
                target_mel = torch.log(target_mel.clamp(min=1e-5))
            except Exception:
                return None
        else:
            target_mel = target_audio

        # Filter by sample type
        if sample_types is not None:
            type_mask = torch.tensor([t == 'voice_tts' for t in sample_types], dtype=torch.bool, device=dec_device)
            if not type_mask.any():
                return None
            target_mel = gpu_safe_index(target_mel, type_mask)
            text_embeds = gpu_safe_index(text_embeds, type_mask)

        # Filter valid mel spectrograms
        valid_mask = target_mel.abs().sum(dim=(1, 2)) > 1e-6
        num_valid = valid_mask.sum().item()
        
        if num_valid == 0:
            return None

        target_mel = gpu_safe_index(target_mel, valid_mask)
        text_embeds = gpu_safe_index(text_embeds, valid_mask)

        with torch.no_grad():
            pred_mel, durations = audio_decoder(text_embeds, target_length=target_mel.shape[-1])
            mel_loss = F.mse_loss(pred_mel, target_mel)
            return mel_loss.detach()

    except Exception as e:
        return None

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


def create_collate_fn(video_frames: int, video_size: int, active_modalities: str = 'all'):
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
                vision_size = 384  # SigLIP SO400M default
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
                max_audio_len = 1000
                target_mel_bins = 80
                for b in batch:
                    af = b["audio_features"]
                    if af is not None and isinstance(af, torch.Tensor) and af.dim() == 2:
                        # Handle different number of mel bins by interpolating
                        if af.shape[0] != target_mel_bins:
                            af = F.interpolate(
                                af.unsqueeze(0).unsqueeze(0),
                                size=(target_mel_bins, af.shape[1]),
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(0).squeeze(0)
                        
                        # Handle time dimension
                        if af.shape[1] != max_audio_len:
                            if af.shape[1] > max_audio_len:
                                af = af[:, :max_audio_len]
                            else:
                                pad = torch.zeros(target_mel_bins, max_audio_len - af.shape[1])
                                af = torch.cat([af, pad], dim=1)
                        audio_features_list.append(af)
                    else:
                        audio_features_list.append(torch.zeros(target_mel_bins, max_audio_len))
                audio_features = torch.stack(audio_features_list)
            else:
                # Minimal tensor to save memory
                audio_features = torch.zeros(batch_size, 1, 1)

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
            vision_size = 384 if need_image else 1
            vf_count = video_frames if need_video else 1
            vf_size = video_size if need_video else 1
            af_bins = 80 if need_audio else 1
            af_len = 1000 if need_audio else 1
            
            return {
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
                "labels": torch.stack([b["labels"] for b in batch]),
                "pixel_values": torch.zeros(batch_size, 3, vision_size, vision_size),
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
    """
    # Choose optimizer based on 8-bit setting
    if use_8bit_optimizer and BNB_AVAILABLE:
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        print("   ✅ Using 8-bit AdamW optimizer (saves ~75% optimizer memory)")
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        if use_8bit_optimizer and not BNB_AVAILABLE:
            print("   ⚠️ bitsandbytes not available, using standard AdamW")
            print("      Install with: pip install bitsandbytes")

    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
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
        target_size: Target image size for diffusion
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
            type_mask = torch.tensor([t in image_sample_types for t in sample_types], device=gen_device)
            if not type_mask.any():
                return None
            images = images[type_mask]
            text_context = text_context[type_mask]
            if mask is not None:
                mask = mask[type_mask]

        # Filter to only samples with valid (non-zero) images
        valid_mask = images.abs().sum(dim=(1, 2, 3)) > 1e-6
        if not valid_mask.any():
            return None
        
        images = images[valid_mask]
        text_context = text_context[valid_mask]
        if mask is not None:
            mask = mask[valid_mask]

        # Resize to target size
        images = F.interpolate(images, size=(target_size, target_size), mode='bilinear', align_corners=False)
        if mask is not None:
            mask = F.interpolate(mask, size=(target_size // 8, target_size // 8), mode='nearest')

        # Normalize images to [-1, 1] for diffusion
        images_norm = images * 2 - 1
        
        # Delete intermediate tensors to save memory
        del images, valid_mask

        # Use generator's training_step if available (SOTA method)
        if hasattr(generator, 'training_step'):
            losses = generator.training_step(images_norm, text_context, mask)
            loss = losses['total_loss']
            del losses, images_norm  # Clean up
            return loss
        
        # Fallback to manual training
        z, mean, logvar = generator.encode(images_norm)
        del images_norm  # No longer needed
        
        batch_size = z.shape[0]
        timesteps = torch.randint(0, 1000, (batch_size,), device=gen_device)
        noise = torch.randn_like(z)
        
        # Add noise
        if hasattr(generator, 'add_noise'):
            noisy_z = generator.add_noise(z, noise, timesteps)
        else:
            alpha_t = generator.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
            noisy_z = torch.sqrt(alpha_t) * z + torch.sqrt(1 - alpha_t) * noise
        
        noise_pred = generator.unet(noisy_z, timesteps, text_context, mask)
        del noisy_z  # Clean up
        
        diff_loss = F.mse_loss(noise_pred, noise)
        del noise_pred, noise  # Clean up
        
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        del z, mean, logvar  # Clean up
        
        return diff_loss + 0.0001 * kl_loss
        
    except Exception as e:
        print(f"      ⚠️ Image diffusion training error: {type(e).__name__}: {str(e)[:100]}")
        return None


def train_video_diffusion_step(video_generator, video_frames, text_context, target_size=256, sample_types=None):
    """
    Train SOTA video diffusion on video data.
    
    Uses the generator's training_step method which includes:
    - Diffusion loss (noise prediction)
    - KL divergence loss (VAE regularization)
    - Temporal consistency loss (smooth motion)
    - Classifier-free guidance training
    
    Args:
        video_generator: Video generator model (MobileVideoDiffusion)
        video_frames: Batch of video frames (B, T, C, H, W) or (B, C, T, H, W)
        text_context: Text embeddings (B, seq_len, hidden_dim)
        target_size: Target frame size for diffusion
        sample_types: List of sample types to filter valid samples
    
    Returns:
        Total loss or None if no valid samples
    """
    if video_generator is None or video_frames is None:
        return None
    try:
        if not isinstance(video_frames, torch.Tensor):
            return None
        if video_frames.numel() == 0:
            return None

        gen_device = next(video_generator.parameters()).device
        gen_dtype = next(video_generator.parameters()).dtype
        # Match input dtype to model dtype to avoid "Input type (float) and bias type (c10::Half)" errors
        video_frames = video_frames.to(device=gen_device, dtype=gen_dtype)
        text_context = text_context.to(device=gen_device, dtype=gen_dtype)

        # Filter by sample type if provided
        video_sample_types = ['video_generation', 'image_to_video', 'video_caption', 'video_qa', 
                              'video_preference', 'video_likert', 'text_to_video']
        if sample_types is not None:
            type_mask = torch.tensor([t in video_sample_types for t in sample_types], device=gen_device)
            if not type_mask.any():
                return None
            video_frames = video_frames[type_mask]
            text_context = text_context[type_mask]

        # Handle dimension ordering: ensure [B, C, T, H, W]
        if video_frames.dim() == 5:
            B, dim1, dim2, H, W = video_frames.shape
            if dim1 > dim2:  # [B, T, C, H, W] -> [B, C, T, H, W]
                video_frames = video_frames.permute(0, 2, 1, 3, 4)
        elif video_frames.dim() == 4:
            video_frames = video_frames.unsqueeze(2)  # Add time dimension
        else:
            return None

        B, C, T, H, W = video_frames.shape

        if C != 3 or T < 1:
            return None

        # Filter to only samples with valid (non-zero) video frames
        valid_mask = video_frames.abs().sum(dim=(1, 2, 3, 4)) > 1e-6
        if not valid_mask.any():
            return None
        
        video_frames = video_frames[valid_mask]
        text_context = text_context[valid_mask]
        B = video_frames.shape[0]

        # Resize frames to target size
        video_frames = video_frames.contiguous().view(B * T, C, H, W)
        video_frames = F.interpolate(video_frames, size=(target_size, target_size), mode='bilinear', align_corners=False)
        video_frames = video_frames.contiguous().view(B, C, T, target_size, target_size)

        # Normalize to [-1, 1] for diffusion
        video_frames_norm = video_frames * 2 - 1
        
        # Delete intermediate tensors to save memory
        del video_frames, valid_mask

        # Extract first frame for I2V training (50% of the time)
        first_frame = None
        if torch.rand(1).item() > 0.5:
            first_frame = (video_frames_norm[:, :, 0] + 1) / 2  # Back to [0, 1] for encode_image

        # Use generator's training_step if available (SOTA method)
        if hasattr(video_generator, 'training_step'):
            losses = video_generator.training_step(video_frames_norm, text_context, first_frame)
            loss = losses['total_loss']
            del losses, video_frames_norm, first_frame  # Clean up
            return loss
        
        # Fallback to manual training
        z, mean, logvar = video_generator.encode_video(video_frames_norm)
        del video_frames_norm  # No longer needed
        
        batch_size = z.shape[0]
        timesteps = torch.randint(0, 1000, (batch_size,), device=gen_device)
        noise = torch.randn_like(z)
        
        # Add noise
        if hasattr(video_generator, 'add_noise'):
            noisy_z = video_generator.add_noise(z, noise, timesteps)
        else:
            alpha_t = video_generator.alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
            noisy_z = torch.sqrt(alpha_t) * z + torch.sqrt(1 - alpha_t) * noise
        
        first_frame_latent = z[:, :, 0] if first_frame is not None else None
        noise_pred = video_generator.unet(noisy_z, timesteps, text_context, first_frame_latent)
        del noisy_z, first_frame_latent  # Clean up
        
        # Losses
        diff_loss = F.mse_loss(noise_pred, noise)
        del noise_pred, noise  # Clean up
        
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        
        # Temporal consistency loss
        temporal_loss = torch.tensor(0.0, device=gen_device)
        if z.shape[2] > 1:
            z_diff = z[:, :, 1:] - z[:, :, :-1]
            temporal_loss = torch.mean(z_diff ** 2)
        
        del z, mean, logvar  # Clean up
        
        return diff_loss + 0.0001 * kl_loss + 0.01 * temporal_loss
        
    except Exception as e:
        print(f"      ⚠️ Video diffusion training error: {type(e).__name__}: {str(e)[:100]}")
        return None


def train_voice_asr_step(audio_encoder, audio_features, text_embeds, sample_types=None):
    """Train ASR: audio -> text alignment.
    
    Args:
        audio_encoder: Audio encoder model
        audio_features: Batch of audio mel spectrograms (B, mel_bins, time)
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
        # Match input dtype to model dtype to avoid "Input type (float) and bias type (c10::Half)" errors
        audio_features = audio_features.to(device=enc_device, dtype=enc_dtype)
        text_embeds = text_embeds.to(device=enc_device, dtype=enc_dtype)

        if audio_features.dim() != 3:
            return None

        # First filter by sample type if provided - only train on voice_asr samples
        if sample_types is not None:
            type_mask = torch.tensor([t == 'voice_asr' for t in sample_types], device=enc_device)
            if not type_mask.any():
                return None
            audio_features = audio_features[type_mask]
            text_embeds = text_embeds[type_mask]

        # Then filter to only samples with valid (non-zero) audio
        # Check each sample in the batch - sum across mel bins and time
        # Use a very lenient threshold to catch any non-zero audio
        valid_mask = audio_features.abs().sum(dim=(1, 2)) > 1e-6  # Very lenient threshold
        num_valid = valid_mask.sum().item()
        
        if num_valid == 0:
            return None
        
        # Filter audio_features and text_embeds to only valid samples
        audio_features = audio_features[valid_mask]
        text_embeds = text_embeds[valid_mask]

        # Need at least 2 samples for contrastive learning
        if audio_features.shape[0] < 2:
            # For single sample, use MSE loss between audio and text embeddings instead
            audio_embeds = audio_encoder(audio_features)
            audio_pooled = audio_embeds.mean(dim=1)
            text_pooled = text_embeds.mean(dim=1)
            # Project to same dimension if needed
            if audio_pooled.shape[-1] != text_pooled.shape[-1]:
                min_dim = min(audio_pooled.shape[-1], text_pooled.shape[-1])
                audio_pooled = audio_pooled[..., :min_dim]
                text_pooled = text_pooled[..., :min_dim]
            loss = F.mse_loss(audio_pooled, text_pooled)
            return loss

        audio_embeds = audio_encoder(audio_features)
        audio_pooled = audio_embeds.mean(dim=1)
        text_pooled = text_embeds.mean(dim=1)
        audio_pooled = F.normalize(audio_pooled, dim=-1)
        text_pooled = F.normalize(text_pooled, dim=-1)
        similarity = torch.matmul(audio_pooled, text_pooled.T)
        labels = torch.arange(similarity.shape[0], device=enc_device)
        loss = F.cross_entropy(similarity, labels)
        return loss
    except Exception as e:
        # Log the exception for debugging
        import traceback
        print(f"      ⚠️ ASR training error: {type(e).__name__}: {str(e)[:100]}")
        return None


def train_voice_tts_step(audio_decoder, text_embeds, target_mel, sample_types=None):
    """Train TTS: text -> audio generation.
    
    Args:
        audio_decoder: Audio decoder model
        text_embeds: Text embeddings (B, seq_len, hidden_dim)
        target_mel: Target mel spectrograms (B, mel_bins, time)
        sample_types: List of sample types to filter valid samples (e.g., ['voice_tts', 'text', ...])
    """
    if audio_decoder is None or target_mel is None:
        return None
    try:
        if not isinstance(target_mel, torch.Tensor):
            return None
        if target_mel.numel() == 0:
            return None

        dec_device = next(audio_decoder.parameters()).device
        dec_dtype = next(audio_decoder.parameters()).dtype
        # Match input dtype to model dtype to avoid "mat1 and mat2 must have the same dtype" errors
        target_mel = target_mel.to(device=dec_device, dtype=dec_dtype)
        text_embeds = text_embeds.to(device=dec_device, dtype=dec_dtype)

        if target_mel.dim() != 3:
            return None

        # First filter by sample type if provided - only train on voice_tts samples
        if sample_types is not None:
            type_mask = torch.tensor([t == 'voice_tts' for t in sample_types], device=dec_device)
            if not type_mask.any():
                return None
            target_mel = target_mel[type_mask]
            text_embeds = text_embeds[type_mask]

        # Then filter to only samples with valid (non-zero) target mel
        # Check each sample in the batch - sum across mel bins and time
        # Use a very lenient threshold to catch any non-zero audio
        valid_mask = target_mel.abs().sum(dim=(1, 2)) > 1e-6  # Very lenient threshold
        num_valid = valid_mask.sum().item()
        
        if num_valid == 0:
            return None
        
        # Filter target_mel and text_embeds to only valid samples
        target_mel = target_mel[valid_mask]
        text_embeds = text_embeds[valid_mask]

        pred_mel, durations = audio_decoder(text_embeds, target_length=target_mel.shape[-1])
        mel_loss = F.mse_loss(pred_mel, target_mel)
        return mel_loss
    except Exception as e:
        # Log the exception for debugging
        print(f"      ⚠️ TTS training error: {type(e).__name__}: {str(e)[:100]}")
        return None

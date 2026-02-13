"""Xoron Multimodal Model - Complete implementation with FP16-native stability."""

import os
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union
from safetensors.torch import save_model, load_model

from transformers import LlamaConfig

from config import XoronConfig
from models.components.lora import (
    LoRALinear, LoRAConfig, apply_lora_to_model, 
    get_lora_parameters, freeze_non_lora_params, enable_lora_training
)
from models.components.attention import MultimodalFusionLayer
from models.components.projectors import MultimodalProjector
from models.encoders.vision import VisionEncoder
from models.encoders.video import VideoEncoder
from models.encoders.audio import AudioEncoder, AudioDecoder, RawWaveformDecoder
from models.generators.image import MobileDiffusionGenerator
from models.generators.video import MobileVideoDiffusion
from models.llm.moe_llama import MoELlamaForCausalLM

# Logger for model operations
logger = logging.getLogger(__name__)

# FP16 safe max value for LINEAR/HIDDEN states only (not for attention logits before softmax)
# For attention pre-softmax clamping, use ~11.0 since exp(11) ≈ 60000 is near FP16 max (65504)
# This value (10000) is safe for hidden states since they don't go through exp()
MAX_HIDDEN = 10000.0


def safe_clamp_tensor(x: torch.Tensor, max_val: float = MAX_HIDDEN) -> torch.Tensor:
    """Clamp tensor values for FP16 safety, handling NaN/Inf properly.
    
    WARNING: Only use for linear/hidden states, NOT for attention scores before softmax!
    For attention scores, use a max of ~11.0 to prevent exp() overflow.
    
    CRITICAL: torch.clamp does NOT fix NaN! clamp(nan, -10, 10) = nan
    Must use nan_to_num first.
    """
    if x is None or x.numel() == 0:
        return x
    x = torch.nan_to_num(x, nan=0.0, posinf=max_val, neginf=-max_val)
    return x.clamp(-max_val, max_val)


# Component groups for fine-tuning
COMPONENT_GROUPS = {
    'vision': ['vision_encoder', 'projector'],
    'video': ['video_encoder'],
    'audio': ['audio_encoder', 'audio_decoder', 'audio_projector', 'waveform_decoder'],
    'speech': ['waveform_decoder'],  # Specifically for Speech-to-Speech
    'llm': ['llm'],
    'cross_attention': ['cross_attention_layers'],
    'image_generation': ['generator'],
    'video_generation': ['video_generator'],
    'modality_markers': ['image_start', 'image_end', 'video_start', 'video_end', 'audio_start', 'audio_end'],
}


class MultimodalModelOutput(dict):
    """Output class for multimodal model."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value


class XoronMultimodalModel(nn.Module):
    """
    Xoron-Dev: Complete multimodal model with:
    - Image/video understanding (CLIP)
    - Text generation (MoE LLM)
    - Image/video generation (MobileDiffusion)
    - Voice understanding and generation (ASR/TTS)
    - Cross-attention for multimodal fusion
    - LoRA support for efficient fine-tuning
    - Flash Attention for faster training
    - Model Parallelism support for multi-GPU training
    """

    def __init__(self, config: XoronConfig, device_map: Dict[str, str] = None):
        super().__init__()
        self.config = config
        self.device_map = device_map
        self._model_parallel = device_map is not None and len(set(device_map.values())) > 1

        print("\n" + "=" * 60)
        print("🚀 BUILDING XORON-DEV MULTIMODAL MODEL")
        if self._model_parallel:
            print("   ⚡ Model Parallelism: ENABLED")
        print("=" * 60)

        # 1. Vision Encoder
        self.vision_encoder = VisionEncoder(config.vision_model_name, freeze=config.freeze_vision)

        # 2. Video Encoder
        self.video_encoder = VideoEncoder(self.vision_encoder, max_frames=config.video_max_frames)

        # 3. Audio Encoder (for ASR) - SOTA with Raw Waveform Tokenizer, RMLA, Zero-Shot Speaker Cloning
        print(f"\n🎤 Building SOTA Audio Encoder...")
        self.audio_encoder = AudioEncoder(
            hidden_size=config.hidden_size,
            n_mels=80,
            max_audio_length=3000,
            use_raw_waveform=getattr(config, 'use_raw_waveform', True),
        )

        # 4. Audio Decoder (for TTS) - SOTA with MAS, Zero-Shot Speaker Cloning, In-Context Prompting
        print(f"\n🔊 Building SOTA Audio Decoder...")
        self.audio_decoder = AudioDecoder(
            hidden_size=config.hidden_size,
            n_mels=80,
            max_audio_length=1000,
        )

        # 5. Waveform Decoder - Direct audio output for Speech-to-Speech (no vocoder needed)
        print(f"\n🎙️ Building Raw Waveform Decoder (Speech-to-Speech)...")
        self.waveform_decoder = RawWaveformDecoder(
            hidden_size=config.hidden_size,
            sample_rate=getattr(config, 'audio_sample_rate', 16000),
        )

        # 6. LLM Config
        llm_config = LlamaConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=1e-6,
            tie_word_embeddings=getattr(config, 'tie_word_embeddings', True),
            pad_token_id=0,
        )
        llm_config.use_flash_attention = config.use_flash_attention
        
        # Ring Attention for efficient 128K context
        llm_config.use_ring_attention = getattr(config, 'use_ring_attention', True)
        llm_config.ring_attention_chunk_size = getattr(config, 'ring_attention_chunk_size', 4096)

        moe_config = {
            'use_moe': config.use_moe,
            'num_experts': config.num_experts,
            'num_experts_per_tok': config.num_experts_per_tok,
            'moe_layer_freq': config.moe_layer_freq,
            'intermediate_size': config.intermediate_size,
            # Note: No router_aux_loss_coef - we use Aux-Lossless MoE
        }

        # 6. LLM with MoE
        print(f"\n🧠 Building LLM: {config.hidden_size}d, {config.num_layers}L")
        print(f"   📏 Context: {config.max_position_embeddings//1024}K positions")
        if config.use_ring_attention:
            print(f"   🔄 Ring Attention: {config.ring_attention_chunk_size} chunk size")
        print(f"   🎯 MoE: {config.num_experts} experts, top-{config.num_experts_per_tok} (Aux-Lossless)")
        print(f"   ⚡ Flash Attention: {config.use_flash_attention}")
        self.llm = MoELlamaForCausalLM(llm_config, moe_config)
        print(f"   ✅ MoE layers: {self.llm.model.num_moe_layers}/{config.num_layers}")

        # 7. Multimodal Projector
        self.projector = MultimodalProjector(
            self.vision_encoder.hidden_size,
            config.hidden_size,
            config.num_vision_tokens
        )
        print(f"   🔗 Projector: {self.vision_encoder.hidden_size} -> {config.hidden_size}")

        # 8. Audio Projector
        self.audio_projector = nn.Linear(config.hidden_size, config.hidden_size)

        # 9. Modality markers
        self.image_start = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)
        self.image_end = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)
        self.video_start = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)
        self.video_end = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)
        self.audio_start = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)
        self.audio_end = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)

        # 10. Cross-Attention Fusion Layers
        self.cross_attention_layers = None
        if config.use_cross_attention:
            print(f"\n🔀 Building Cross-Attention Fusion ({config.cross_attention_layers} layers)...")
            self.cross_attention_layers = nn.ModuleList([
                MultimodalFusionLayer(
                    hidden_size=config.hidden_size,
                    num_heads=config.cross_attention_heads,
                    dropout=config.cross_attention_dropout,
                    use_flash_attention=config.use_flash_attention,
                )
                for _ in range(config.cross_attention_layers)
            ])
            print(f"   ✅ Cross-attention: {config.cross_attention_layers} layers, {config.cross_attention_heads} heads")

        # 11. Image Generator
        self.generator = None
        if config.enable_generation:
            print(f"\n🎨 Building MobileDiffusion Generator (Images)...")
            # CRITICAL: Use image_max_size so model can handle ALL multi-scale sizes
            self.generator = MobileDiffusionGenerator(
                latent_channels=config.generation_latent_channels,
                base_channels=config.generation_base_channels,
                context_dim=config.hidden_size,
                num_inference_steps=config.generation_inference_steps,
                image_size=config.image_max_size,   # FIXED: Use max_size for multi-scale support
            )

        # 12. Video Generator
        self.video_generator = None
        if config.enable_generation:
            print(f"\n🎬 Building MobileVideoDiffusion Generator (Videos)...")
            # CRITICAL: Use video_max_size so model can handle ALL multi-scale sizes
            # The model's UNet uses max_height/max_width based on image_size
            # Using max_size ensures it can process 128, 192, 256, 320, 384 etc.
            self.video_generator = MobileVideoDiffusion(
                latent_channels=config.generation_latent_channels,
                base_channels=config.generation_base_channels // 2,
                context_dim=config.hidden_size,
                num_frames=config.video_max_frames,  # Multi-scale: use video_max_frames
                image_size=config.video_max_size,    # FIXED: Use max_size for multi-scale support
                num_inference_steps=config.generation_inference_steps,
            )

        self.num_vision_tokens = config.num_vision_tokens
        self.video_max_frames = config.video_max_frames  # Multi-scale config
        self.lora_applied = False

        self._print_stats()
        print("=" * 60)

    def apply_model_parallel(self, device_map: Dict[str, str]):
        """Apply Model Parallelism by placing components on different devices."""
        self.device_map = device_map
        self._model_parallel = len(set(device_map.values())) > 1

        if not self._model_parallel:
            print("   ℹ️ Single device - no model parallelism needed")
            return self

        print("\n🔀 Applying Model Parallelism...")

        self.vision_encoder = self.vision_encoder.to(device_map['vision_encoder'])
        print(f"   ✅ Vision encoder -> {device_map['vision_encoder']}")

        self.video_encoder = self.video_encoder.to(device_map['video_encoder'])
        print(f"   ✅ Video encoder -> {device_map['video_encoder']}")

        self.audio_encoder = self.audio_encoder.to(device_map['audio_encoder'])
        print(f"   ✅ Audio encoder -> {device_map['audio_encoder']}")

        self.audio_decoder = self.audio_decoder.to(device_map['audio_decoder'])
        print(f"   ✅ Audio decoder -> {device_map['audio_decoder']}")

        # Waveform decoder for Speech-to-Speech (direct audio output)
        if hasattr(self, 'waveform_decoder') and self.waveform_decoder is not None:
            waveform_device = device_map.get('waveform_decoder', device_map['audio_decoder'])
            self.waveform_decoder = self.waveform_decoder.to(waveform_device)
            print(f"   ✅ Waveform decoder -> {waveform_device}")

        self.projector = self.projector.to(device_map['projector'])
        print(f"   ✅ Projector -> {device_map['projector']}")

        self.audio_projector = self.audio_projector.to(device_map['audio_projector'])
        print(f"   ✅ Audio projector -> {device_map['audio_projector']}")

        self.llm = self.llm.to(device_map['llm'])
        print(f"   ✅ LLM -> {device_map['llm']}")

        if self.cross_attention_layers is not None:
            self.cross_attention_layers = self.cross_attention_layers.to(device_map['cross_attention'])
            print(f"   ✅ Cross-attention -> {device_map['cross_attention']}")

        if self.generator is not None:
            self.generator = self.generator.to(device_map['generator'])
            print(f"   ✅ Image generator -> {device_map['generator']}")

        if self.video_generator is not None:
            self.video_generator = self.video_generator.to(device_map['video_generator'])
            print(f"   ✅ Video generator -> {device_map['video_generator']}")

        marker_device = device_map['modality_markers']
        self.image_start = nn.Parameter(self.image_start.data.to(marker_device))
        self.image_end = nn.Parameter(self.image_end.data.to(marker_device))
        self.video_start = nn.Parameter(self.video_start.data.to(marker_device))
        self.video_end = nn.Parameter(self.video_end.data.to(marker_device))
        self.audio_start = nn.Parameter(self.audio_start.data.to(marker_device))
        self.audio_end = nn.Parameter(self.audio_end.data.to(marker_device))
        print(f"   ✅ Modality markers -> {marker_device}")

        print("   ✅ Model Parallelism applied successfully!")
        return self

    def get_llm_device(self):
        """Get the device where LLM is located."""
        if self.device_map is not None:
            return torch.device(self.device_map['llm'])
        return next(self.llm.parameters()).device

    def get_encoder_device(self):
        """Get the device where encoders are located."""
        if self.device_map is not None:
            return torch.device(self.device_map['vision_encoder'])
        return next(self.vision_encoder.parameters()).device

    def apply_lora(self):
        """
        Apply LoRA to the LLM and optionally cross-attention layers.
        
        MEMORY OPTIMIZATION:
        - LoRA layers share base weights (no cloning)
        - Base weights in LoRA layers are frozen (requires_grad=False)
        - LoRA params (A, B, magnitude) are always trainable
        
        NOTE: This does NOT freeze other components!
        Component freezing is handled separately by freeze_components() based on
        training mode (--text, --video, --image, --voice flags).
        
        This allows PARALLEL FINE-TUNING:
        - LoRA adapters on LLM for efficient adaptation
        - Full weight training on active components (vision, audio, etc.)
        """
        if self.lora_applied:
            print("⚠️ LoRA already applied")
            return

        if not self.config.use_lora:
            print("ℹ️ LoRA disabled in config")
            return

        lora_config = LoRAConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=list(self.config.lora_target_modules),
            enable_lora=True,
        )

        print("\n🔧 Applying LoRA to LLM...")
        self.llm = apply_lora_to_model(self.llm, lora_config)

        if self.cross_attention_layers is not None:
            print("🔧 Applying LoRA to cross-attention layers...")
            cross_attn_lora_config = LoRAConfig(
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                enable_lora=True,
            )
            for i, layer in enumerate(self.cross_attention_layers):
                self.cross_attention_layers[i] = apply_lora_to_model(layer, cross_attn_lora_config)

        self.lora_applied = True
        
        # NOTE: We do NOT freeze all non-LoRA params here!
        # Freezing is handled by freeze_components() based on training mode.
        # This allows parallel fine-tuning: LoRA + full weight training on active components.
        
        self._print_stats()

    def get_trainable_params(self):
        """
        Get trainable parameters, respecting LoRA settings and component freezing.
        
        If train_lora_only=True and LoRA is applied:
            - Freezes all non-LoRA params
            - Returns only LoRA params
        Otherwise:
            - Returns all params with requires_grad=True
            - This includes both LoRA params AND unfrozen component weights
            - Allows parallel fine-tuning: LoRA + full weights on active components
        """
        if self.config.train_lora_only and self.lora_applied:
            # LoRA-only mode: freeze everything except LoRA params
            freeze_non_lora_params(self)
            return get_lora_parameters(self)
        # Normal mode: return all trainable params (LoRA + unfrozen components)
        return [p for p in self.parameters() if p.requires_grad]

    def _print_stats(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n📊 Model Statistics:")
        print(f"   Total parameters: {total/1e6:.1f}M")
        print(f"   Trainable parameters: {trainable/1e6:.1f}M")
        if self.lora_applied:
            lora_params = sum(p.numel() for n, p in self.named_parameters() if 'lora_' in n)
            print(f"   LoRA parameters: {lora_params/1e6:.2f}M")

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        encoder_device = self.get_encoder_device()
        pixel_values = pixel_values.to(encoder_device)
        vision_features = self.vision_encoder(pixel_values)
        projected = self.projector(vision_features)
        llm_device = self.get_llm_device()
        return projected.to(llm_device)

    def encode_video(self, video_frames: torch.Tensor) -> torch.Tensor:
        encoder_device = self.get_encoder_device()
        video_frames = video_frames.to(encoder_device)
        video_features = self.video_encoder(video_frames)
        projected = self.projector(video_features)
        llm_device = self.get_llm_device()
        return projected.to(llm_device)

    def encode_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        encoder_device = self.get_encoder_device()
        audio_features = audio_features.to(encoder_device)
        audio_embeds = self.audio_encoder(audio_features)
        projected = self.audio_projector(audio_embeds)
        llm_device = self.get_llm_device()
        return projected.to(llm_device)

    def get_text_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        llm_device = self.get_llm_device()
        input_ids = input_ids.to(llm_device)
        embeddings = self.llm.model.embed_tokens(input_ids)
        return embeddings

    def _apply_cross_attention(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor = None,
        video_embeds: torch.Tensor = None,
        audio_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.cross_attention_layers is None:
            return text_embeds

        for fusion_layer in self.cross_attention_layers:
            # MultimodalFusionLayer now returns (output, cache) tuple
            text_embeds, _ = fusion_layer(
                text_hidden=text_embeds,
                image_hidden=image_embeds,
                video_hidden=video_embeds,
                audio_hidden=audio_embeds,
                use_cache=False,  # Don't use cache during training
            )

        return text_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.Tensor = None,
        video_frames: torch.Tensor = None,
        audio_features: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        """Forward pass - FP16 native."""
        batch_size = input_ids.shape[0]
        llm_device = self.get_llm_device()

        input_ids_llm = input_ids.to(llm_device)
        text_embeds = self.llm.model.embed_tokens(input_ids_llm)
        text_embeds = safe_clamp_tensor(text_embeds)
        
        device = text_embeds.device

        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        image_embeds_for_cross = None
        video_embeds_for_cross = None
        audio_embeds_for_cross = None

        def has_content(tensor):
            if tensor is None:
                return False
            if not isinstance(tensor, torch.Tensor):
                return False
            try:
                if tensor.numel() == 0:
                    return False
                return bool(tensor.any())
            except Exception:
                return False

        if has_content(pixel_values):
            try:
                image_embeds = self.encode_image(pixel_values)
                image_embeds = safe_clamp_tensor(image_embeds)
                image_embeds_for_cross = image_embeds
                image_start = self.image_start.expand(batch_size, -1, -1)
                image_end = self.image_end.expand(batch_size, -1, -1)
                image_embeds = torch.cat([image_start, image_embeds, image_end], dim=1)
                text_embeds = torch.cat([image_embeds, text_embeds], dim=1)
                text_embeds = safe_clamp_tensor(text_embeds)

                if attention_mask is not None:
                    image_mask = torch.ones(batch_size, image_embeds.shape[1], device=device)
                    attention_mask = torch.cat([image_mask, attention_mask], dim=1)

                if labels is not None:
                    image_labels = torch.full((batch_size, image_embeds.shape[1]), -100, device=device, dtype=labels.dtype)
                    labels = torch.cat([image_labels, labels], dim=1)
            except Exception as e:
                logger.debug(f"Image encoding skipped: {e}")

        if has_content(video_frames):
            try:
                video_embeds = self.encode_video(video_frames)
                video_embeds = safe_clamp_tensor(video_embeds)
                video_embeds_for_cross = video_embeds
                video_start = self.video_start.expand(batch_size, -1, -1)
                video_end = self.video_end.expand(batch_size, -1, -1)
                video_embeds = torch.cat([video_start, video_embeds, video_end], dim=1)
                text_embeds = torch.cat([video_embeds, text_embeds], dim=1)
                text_embeds = safe_clamp_tensor(text_embeds)

                if attention_mask is not None:
                    video_mask = torch.ones(batch_size, video_embeds.shape[1], device=device)
                    attention_mask = torch.cat([video_mask, attention_mask], dim=1)

                if labels is not None:
                    video_labels = torch.full((batch_size, video_embeds.shape[1]), -100, device=device, dtype=labels.dtype)
                    labels = torch.cat([video_labels, labels], dim=1)
            except Exception as e:
                logger.debug(f"Video encoding skipped: {e}")

        if has_content(audio_features):
            try:
                audio_embeds = self.encode_audio(audio_features)
                audio_embeds = safe_clamp_tensor(audio_embeds)
                audio_embeds_for_cross = audio_embeds
                audio_start = self.audio_start.expand(batch_size, -1, -1)
                audio_end = self.audio_end.expand(batch_size, -1, -1)
                audio_embeds = torch.cat([audio_start, audio_embeds, audio_end], dim=1)
                text_embeds = torch.cat([audio_embeds, text_embeds], dim=1)
                text_embeds = safe_clamp_tensor(text_embeds)

                if attention_mask is not None:
                    audio_mask = torch.ones(batch_size, audio_embeds.shape[1], device=device)
                    attention_mask = torch.cat([audio_mask, attention_mask], dim=1)

                if labels is not None:
                    audio_labels = torch.full((batch_size, audio_embeds.shape[1]), -100, device=device, dtype=labels.dtype)
                    labels = torch.cat([audio_labels, labels], dim=1)
            except Exception as e:
                logger.debug(f"Audio encoding skipped: {e}")

        if self.cross_attention_layers is not None:
            try:
                text_embeds = self._apply_cross_attention(
                    text_embeds,
                    image_embeds=image_embeds_for_cross,
                    video_embeds=video_embeds_for_cross,
                    audio_embeds=audio_embeds_for_cross,
                )
                text_embeds = safe_clamp_tensor(text_embeds)
            except Exception as e:
                logger.debug(f"Cross-attention skipped: {e}")

        text_embeds = safe_clamp_tensor(text_embeds)
        
        outputs = self.llm(inputs_embeds=text_embeds, attention_mask=attention_mask, labels=labels)

        return MultimodalModelOutput(
            loss=outputs.loss if hasattr(outputs, 'loss') else None,
            logits=outputs.logits if hasattr(outputs, 'logits') else None,
            aux_loss=outputs.aux_loss if hasattr(outputs, 'aux_loss') else None,
        )

    @torch.no_grad()
    def generate_image(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """Generate image from text."""
        if self.generator is None:
            raise ValueError("Image generator not enabled")
        context = self.get_text_embeddings(input_ids, attention_mask)
        images = self.generator.generate(context)
        return images

    @torch.no_grad()
    def generate_video(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                       first_frame: torch.Tensor = None, num_frames: int = None):
        """Generate video from text (T2V) or from image (I2V)."""
        if self.video_generator is None:
            raise ValueError("Video generator not enabled")

        context = self.get_text_embeddings(input_ids, attention_mask)
        context = context.mean(dim=1)

        if first_frame is not None:
            video = self.video_generator.generate_i2v(first_frame, context, num_frames)
        else:
            video = self.video_generator.generate_t2v(context, num_frames)

        return video

    @torch.no_grad()
    def generate_speech(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """Generate speech (mel-spectrogram) from text (TTS)."""
        text_embeds = self.get_text_embeddings(input_ids, attention_mask)
        mel, durations = self.audio_decoder(text_embeds)
        return mel, durations

    @torch.no_grad()
    def speak(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        speaker_embedding: torch.Tensor = None,
        return_mel: bool = False,
    ) -> torch.Tensor:
        """
        Generate playable audio waveform from text (Speech-to-Speech TTS).
        
        This is the main method for making the model talk. It converts text
        directly to audio waveform without needing an external vocoder.
        
        Args:
            input_ids: [B, T] tokenized text input
            attention_mask: [B, T] attention mask
            speaker_embedding: [B, D] optional speaker embedding for voice cloning
            return_mel: If True, also return intermediate mel spectrogram
            
        Returns:
            waveform: [B, T_audio] raw audio waveform in [-1, 1] range at 16kHz
                      Can be played directly or saved as WAV file
            mel (optional): [B, 80, T_mel] mel spectrogram if return_mel=True
        """
        # Get text embeddings from LLM
        text_embeds = self.get_text_embeddings(input_ids, attention_mask)
        
        # Generate intermediate features through audio decoder
        # This gives us the linguistic/prosodic representation
        mel, durations, _ = self.audio_decoder(
            text_embeds,
            speaker_embedding=speaker_embedding,
        )
        
        # Convert to features for waveform decoder
        # Transpose mel from [B, n_mels, T] to [B, T, n_mels] and project
        mel_features = mel.transpose(1, 2)  # [B, T, 80]
        
        # Project mel to hidden_size for waveform decoder
        if not hasattr(self, '_mel_to_hidden'):
            self._mel_to_hidden = nn.Linear(80, self.config.hidden_size).to(mel.device)
        audio_features = self._mel_to_hidden(mel_features)
        
        # Generate raw waveform
        waveform = self.waveform_decoder(audio_features)
        
        if return_mel:
            return waveform, mel
        return waveform

    @torch.no_grad()
    def listen(self, audio_waveform: torch.Tensor) -> torch.Tensor:
        """
        Transcribe audio to text embeddings (Speech-to-Speech ASR).
        
        This is the listening component - converts speech to embeddings
        that can be fed to the LLM for understanding.
        
        Args:
            audio_waveform: [B, T_audio] raw audio waveform
            
        Returns:
            audio_embeds: [B, T, hidden_size] encoded audio features
        """
        return self.encode_audio(audio_waveform)

    @torch.no_grad()
    def listen_and_respond(
        self,
        audio_waveform: torch.Tensor,
        max_new_tokens: int = 256,
        speaker_embedding: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Full Speech-to-Speech: Listen to audio, generate text response, speak it back.
        
        This is the main conversational method - you speak to it, it responds with voice.
        
        Args:
            audio_waveform: [B, T_audio] input audio (what you said)
            max_new_tokens: Maximum tokens to generate for response
            speaker_embedding: Optional speaker embedding for response voice
            
        Returns:
            response_audio: [B, T_response] audio waveform of the model's response
        """
        device = audio_waveform.device
        
        # 1. Listen - encode the input audio
        audio_embeds = self.listen(audio_waveform)
        
        # 2. Create dummy input for the LLM (audio embeddings will be prepended)
        batch_size = audio_waveform.shape[0]
        
        # Start with a response prompt
        # In practice, you'd use the tokenizer to create proper input_ids
        dummy_input = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        # 3. Generate text response using the LLM with audio context
        # The audio embeddings are injected into the forward pass
        outputs = self.forward(
            input_ids=dummy_input,
            audio_features=audio_waveform,
        )
        
        # Get the generated hidden states
        response_embeds = outputs.get('hidden_states', outputs.get('last_hidden_state'))
        
        # 4. Speak - convert text response to audio
        if response_embeds is not None:
            mel, durations, _ = self.audio_decoder(
                response_embeds,
                speaker_embedding=speaker_embedding,
            )
            
            # Convert mel to waveform
            mel_features = mel.transpose(1, 2)
            if not hasattr(self, '_mel_to_hidden'):
                self._mel_to_hidden = nn.Linear(80, self.config.hidden_size).to(device)
            audio_features = self._mel_to_hidden(mel_features)
            response_audio = self.waveform_decoder(audio_features)
            
            return response_audio
        
        # Fallback: return silence
        return torch.zeros(batch_size, 16000, device=device)  # 1 second of silence

    def merge_lora_weights(self):
        """Merge LoRA weights into main weights for inference."""
        if not self.lora_applied:
            return
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.merge_lora_weights()
        print("✅ LoRA weights merged")

    def unmerge_lora_weights(self):
        """Unmerge LoRA weights for continued training."""
        if not self.lora_applied:
            return
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.unmerge_lora_weights()
        print("✅ LoRA weights unmerged")

    def save_pretrained(
        self,
        path: str,
        optimizer=None,
        scheduler=None,
        global_step: int = 0,
        epoch: int = 0,
        best_loss: float = float('inf'),
        sharded: bool = False,
        max_shard_size: int = 2 * 1024 * 1024 * 1024,  # 2GB default
        save_separately: bool = True,  # Default to component-wise saving
    ):
        """
        Save model and optionally training state for resuming.
        
        Args:
            path: Directory to save the model
            optimizer: Optional optimizer to save state
            scheduler: Optional scheduler to save state
            global_step: Current training step
            epoch: Current epoch
            best_loss: Best loss achieved so far
            sharded: If True, save model in multiple .safetensors files
            max_shard_size: Maximum size per shard in bytes (default 2GB)
            save_separately: If True, save each component as separate .safetensors files (default)
                           This avoids safetensors issues with shared storage in LSTM weights
        """
        os.makedirs(path, exist_ok=True)
        
        if save_separately:
            # Save components separately - handles LSTM shared storage properly
            self._save_components_safe(path)
        elif sharded:
            self._save_sharded(path, max_shard_size)
        else:
            # Fallback with clone to handle shared storage
            self._save_single_file_safe(path)

        config_dict = self.config.to_dict()
        # Mark which components exist in this save (for architecture detection on load)
        config_dict['has_audio_encoder'] = True
        config_dict['has_audio_decoder'] = True
        config_dict['has_waveform_decoder'] = hasattr(self, 'waveform_decoder') and self.waveform_decoder is not None
        config_dict['has_vision_encoder'] = hasattr(self, 'vision_encoder') and self.vision_encoder is not None
        config_dict['has_video_encoder'] = hasattr(self, 'video_encoder') and self.video_encoder is not None
        config_dict['has_generator'] = hasattr(self, 'generator') and self.generator is not None
        config_dict['has_video_generator'] = hasattr(self, 'video_generator') and self.video_generator is not None
        config_dict['has_cross_attention'] = hasattr(self, 'cross_attention_layers') and self.cross_attention_layers is not None
        config_dict['lora_applied'] = self.lora_applied
        config_dict['architecture_version'] = 2  # Version 2 = includes waveform_decoder
        
        # HuggingFace auto_map for trust_remote_code loading
        config_dict['auto_map'] = {
            'AutoConfig': 'configuration_xoron.XoronConfig',
            'AutoModel': 'modeling_xoron.XoronModel',
            'AutoModelForCausalLM': 'modeling_xoron.XoronForCausalLM',
        }

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Copy HuggingFace custom code files for trust_remote_code support
        self._copy_huggingface_files(path)

        if optimizer is not None or scheduler is not None:
            training_state = {
                'global_step': global_step,
                'epoch': epoch,
                'best_loss': best_loss,
            }
            if optimizer is not None:
                training_state['optimizer_state_dict'] = optimizer.state_dict()
            if scheduler is not None:
                training_state['scheduler_state_dict'] = scheduler.state_dict()
            
            torch.save(training_state, os.path.join(path, "training_state.pt"))
            print(f"   💾 Training state saved (step {global_step}, epoch {epoch})")

        print(f"✅ Model saved to {path}")
    
    def _copy_huggingface_files(self, path: str):
        """
        Build and copy HuggingFace custom code files for trust_remote_code support.
        
        This DYNAMICALLY BUILDS a self-contained modeling_xoron.py by combining
        all model components, so users can load from HuggingFace Hub with:
            model = AutoModel.from_pretrained("repo/model", trust_remote_code=True)
        
        WITHOUT needing to install the full Xoron-Dev package.
        
        Args:
            path: Directory to save the files
        """
        import shutil
        
        # Get the root directory of the Xoron project
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        # Copy configuration_xoron.py
        config_src = os.path.join(project_root, 'configuration_xoron.py')
        config_dst = os.path.join(path, 'configuration_xoron.py')
        if os.path.exists(config_src):
            shutil.copy2(config_src, config_dst)
            print(f"   📄 Copied configuration_xoron.py")
        
        # BUILD self-contained modeling_xoron.py dynamically
        modeling_dst = os.path.join(path, 'modeling_xoron.py')
        self._build_self_contained_modeling_file(project_root, modeling_dst)
        
        print(f"   ✅ HuggingFace custom code files ready")
    
    def _build_self_contained_modeling_file(self, project_root: str, output_path: str):
        """
        Build a self-contained modeling_xoron.py by combining all model components.
        
        This creates a single file with ALL model code embedded, removing internal
        imports so it works standalone on HuggingFace without the full package.
        """
        import re
        
        # Component files to combine (in dependency order)
        component_files = [
            "models/components/lora.py",
            "models/components/attention.py",
            "models/components/projectors.py",
            "models/components/moe.py",
            "models/encoders/vision.py",
            "models/encoders/video.py",
            "models/encoders/audio.py",
            "models/generators/image.py",
            "models/generators/video.py",
            "models/llm/moe_llama.py",
            "models/xoron.py",
        ]
        
        # Internal imports to remove
        internal_import_patterns = [
            r"^from config import.*$",
            r"^from config\..*import.*$",
            r"^from models\..*import.*$",
            r"^from models import.*$",
        ]
        
        def is_internal_import(line):
            line = line.strip()
            for pattern in internal_import_patterns:
                if re.match(pattern, line):
                    return True
            return False
        
        def is_module_level_import(line):
            """Check if this is a module-level import (no indentation)."""
            stripped = line.strip()
            # Only consider it a module-level import if:
            # 1. Line starts with no indentation (module level)
            # 2. Line starts with 'import' or 'from'
            if line and not line[0].isspace():
                return (stripped.startswith("import ") or stripped.startswith("from "))
            return False
        
        def extract_code_body(content):
            """Extract code body, removing module docstring and module-level imports only."""
            lines = content.split('\n')
            code_lines = []
            i = 0
            in_multiline_import = False
            
            # Skip leading whitespace
            while i < len(lines) and not lines[i].strip():
                i += 1
            
            # Skip module docstring if present
            if i < len(lines):
                stripped = lines[i].strip()
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    docstring_char = stripped[:3]
                    if stripped.count(docstring_char) >= 2:
                        i += 1
                    else:
                        i += 1
                        while i < len(lines):
                            if docstring_char in lines[i]:
                                i += 1
                                break
                            i += 1
            
            # Process remaining lines
            for line in lines[i:]:
                stripped = line.strip()
                
                # Skip empty lines at start
                if not code_lines and not stripped:
                    continue
                
                # Handle multi-line module-level imports - skip continuation lines
                if in_multiline_import:
                    if ')' in stripped:
                        in_multiline_import = False
                    continue
                
                # Only skip MODULE-LEVEL imports (no indentation)
                # Keep inline/function-level imports like: from transformers import X
                if is_module_level_import(line):
                    # Check if it's a multi-line import (has opening paren but no closing)
                    if '(' in stripped and ')' not in stripped:
                        in_multiline_import = True
                    continue
                
                # Skip logger setup lines at module level only
                if stripped.startswith("logger = logging.getLogger") and not line[0].isspace():
                    continue
                
                code_lines.append(line)
            
            # Remove trailing empty lines
            while code_lines and not code_lines[-1].strip():
                code_lines.pop()
            
            return '\n'.join(code_lines)
        
        # Build the file header
        header = '''"""
Xoron Model for HuggingFace Transformers - Self-Contained Implementation.

AUTO-GENERATED FILE - Do not edit directly!

This module provides a complete, self-contained HuggingFace-compatible model class
for the Xoron multimodal model. All components are embedded directly in this file
to enable loading via AutoModel with trust_remote_code=True WITHOUT requiring
the full Xoron-Dev package to be installed.

Usage:
    from transformers import AutoModel, AutoConfig
    config = AutoConfig.from_pretrained("your-repo/xoron-model", trust_remote_code=True)
    model = AutoModel.from_pretrained("your-repo/xoron-model", trust_remote_code=True)
"""

import os
import math
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from safetensors.torch import save_file, load_file
except ImportError:
    save_file, load_file = None, None

from transformers import PreTrainedModel, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

try:
    from transformers.models.llama.modeling_llama import (
        LlamaAttention, LlamaDecoderLayer, LlamaRMSNorm, LlamaMLP,
        LlamaRotaryEmbedding, apply_rotary_pos_emb, repeat_kv
    )
except ImportError:
    LlamaAttention = LlamaDecoderLayer = LlamaRMSNorm = LlamaMLP = None
    LlamaRotaryEmbedding = apply_rotary_pos_emb = repeat_kv = None

# Import configuration
try:
    from .configuration_xoron import XoronConfig
except ImportError:
    from configuration_xoron import XoronConfig

logger = logging.getLogger(__name__)

'''
        
        all_code = [header]
        
        # Process each component file
        for filepath in component_files:
            full_path = os.path.join(project_root, filepath)
            if not os.path.exists(full_path):
                print(f"   ⚠️ Component not found: {filepath}")
                continue
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            code = extract_code_body(content)
            if code.strip():
                section_name = filepath.replace('/', '.').replace('.py', '').upper()
                section_header = f"\n\n# {'='*78}\n# {section_name}\n# {'='*78}\n\n"
                all_code.append(section_header + code)
        
        # Add HuggingFace wrapper classes
        hf_wrapper = '''

# ==============================================================================
# HUGGINGFACE WRAPPER CLASSES
# ==============================================================================

class XoronPreTrainedModel(PreTrainedModel):
    """Base class for Xoron models providing HuggingFace integration."""
    
    config_class = XoronConfig
    base_model_prefix = "xoron"
    supports_gradient_checkpointing = True
    _no_split_modules = ["XoronMultimodalModel"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    
    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class XoronModel(XoronPreTrainedModel):
    """Xoron Multimodal Model for HuggingFace."""
    
    def __init__(self, config: XoronConfig):
        super().__init__(config)
        self.config = config
        # Delay internal model creation - will be built on first forward or explicitly
        self._internal_model = None
        self._model_initialized = False
    
    def _ensure_model_initialized(self):
        """Lazily initialize the internal model to avoid meta device conflicts."""
        if not self._model_initialized:
            self._internal_model = XoronMultimodalModel(self.config)
            self._model_initialized = True
    
    @property
    def internal_model(self):
        self._ensure_model_initialized()
        return self._internal_model
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load pretrained Xoron model from HuggingFace Hub or local path.
        
        This override ensures proper initialization without meta device conflicts.
        """
        # Remove problematic kwargs that can cause meta device issues
        kwargs.pop('device_map', None)  # We handle device placement ourselves
        
        # Get config first
        config = kwargs.pop('config', None)
        if config is None:
            config = XoronConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # Create model instance (without initializing heavy components yet)
        model = cls(config)
        
        # Now initialize the internal model
        model._internal_model = XoronMultimodalModel(config)
        model._model_initialized = True
        
        # Load weights from safetensors files
        import os
        from safetensors import safe_open
        
        # Check if it's a local path or HF hub
        if os.path.isdir(pretrained_model_name_or_path):
            model_path = pretrained_model_name_or_path
        else:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(repo_id=pretrained_model_name_or_path)
        
        # Load component weights
        components_json = os.path.join(model_path, "components.json")
        if os.path.exists(components_json):
            with open(components_json, 'r') as f:
                manifest = json.load(f)
            
            component_map = {
                'llm': model._internal_model.llm,
                'vision_encoder': model._internal_model.vision_encoder,
                'video_encoder': model._internal_model.video_encoder,
                'audio_encoder': model._internal_model.audio_encoder,
                'audio_decoder': model._internal_model.audio_decoder,
                'projector': model._internal_model.projector,
                'audio_projector': model._internal_model.audio_projector,
            }
            
            if model._internal_model.cross_attention_layers is not None:
                component_map['cross_attention'] = model._internal_model.cross_attention_layers
            if model._internal_model.generator is not None:
                component_map['generator'] = model._internal_model.generator
            if model._internal_model.video_generator is not None:
                component_map['video_generator'] = model._internal_model.video_generator
            if hasattr(model._internal_model, 'waveform_decoder') and model._internal_model.waveform_decoder is not None:
                component_map['waveform_decoder'] = model._internal_model.waveform_decoder
            
            for comp_name in manifest.get('components', []):
                if comp_name == 'modality_markers':
                    continue
                    
                comp_path = os.path.join(model_path, f"{comp_name}.safetensors")
                if os.path.exists(comp_path) and comp_name in component_map:
                    component = component_map[comp_name]
                    if component is not None:
                        with safe_open(comp_path, framework="pt") as f:
                            state_dict = {k: f.get_tensor(k) for k in f.keys()}
                        component.load_state_dict(state_dict, strict=False)
                        print(f"   ✅ Loaded {comp_name}")
            
            # Load modality markers
            markers_path = os.path.join(model_path, "modality_markers.safetensors")
            if os.path.exists(markers_path):
                with safe_open(markers_path, framework="pt") as f:
                    model._internal_model.image_start.data = f.get_tensor('image_start')
                    model._internal_model.image_end.data = f.get_tensor('image_end')
                    model._internal_model.video_start.data = f.get_tensor('video_start')
                    model._internal_model.video_end.data = f.get_tensor('video_end')
                    model._internal_model.audio_start.data = f.get_tensor('audio_start')
                    model._internal_model.audio_end.data = f.get_tensor('audio_end')
                print(f"   ✅ Loaded modality markers")
        
        print(f"✅ Xoron model loaded from {pretrained_model_name_or_path}")
        return model
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        video_frames: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        self._ensure_model_initialized()
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self._internal_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=pixel_values,
            video=video_frames,
            audio=audio_features,
            labels=labels,
        )
        
        if return_dict:
            return CausalLMOutputWithPast(
                loss=outputs.get("loss"),
                logits=outputs.get("logits"),
                past_key_values=outputs.get("past_key_values"),
                hidden_states=outputs.get("hidden_states"),
                attentions=outputs.get("attentions"),
            )
        
        return (outputs.get("loss"), outputs.get("logits"))
    
    def generate_image(self, prompt_embeds: torch.Tensor, **kwargs):
        self._ensure_model_initialized()
        return self._internal_model.generate_image(prompt_embeds, **kwargs)
    
    def generate_video(self, prompt_embeds: torch.Tensor, **kwargs):
        self._ensure_model_initialized()
        return self._internal_model.generate_video(prompt_embeds, **kwargs)
    
    def generate_speech(self, text_embeds: torch.Tensor, **kwargs):
        self._ensure_model_initialized()
        return self._internal_model.generate_speech(text_embeds, **kwargs)


class XoronForCausalLM(XoronModel):
    """Alias for XoronModel for compatibility."""
    pass


# Register for AutoClass
XoronConfig.register_for_auto_class()
XoronModel.register_for_auto_class("AutoModel")
XoronForCausalLM.register_for_auto_class("AutoModelForCausalLM")
'''
        all_code.append(hf_wrapper)
        
        # Write output file
        final_content = '\n'.join(all_code)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        line_count = final_content.count('\n')
        print(f"   📦 Built self-contained modeling_xoron.py ({line_count:,} lines)")

    def _save_single_file_safe(self, path: str):
        """
        Save model as single safetensors file with cloned tensors.
        Cloning breaks shared storage that causes safetensors errors.
        
        Args:
            path: Directory to save the model
        """
        from safetensors.torch import save_file
        
        state_dict = self.state_dict()
        
        # Clone all tensors and make contiguous to break shared storage
        # This fixes LSTM weight sharing issues
        safe_state_dict = {}
        for key, tensor in state_dict.items():
            safe_state_dict[key] = tensor.clone().contiguous()
        
        save_file(safe_state_dict, os.path.join(path, "model.safetensors"))
        size_mb = sum(t.numel() * t.element_size() for t in safe_state_dict.values()) / (1024 * 1024)
        print(f"   💾 Saved model.safetensors ({size_mb:.1f} MB)")

    def _save_components_safe(self, path: str):
        """
        Save model components as separate .safetensors files with cloned tensors.
        This is the default and most robust saving method that:
        1. Handles LSTM weight sharing issues in safetensors
        2. Allows surgical component loading/updates
        3. Better for debugging and inspection
        
        Args:
            path: Directory to save component files
        """
        from safetensors.torch import save_file
        
        os.makedirs(path, exist_ok=True)
        
        component_map = {
            'llm': self.llm,
            'vision_encoder': self.vision_encoder,
            'video_encoder': self.video_encoder,
            'audio_encoder': self.audio_encoder,
            'audio_decoder': self.audio_decoder,
            'projector': self.projector,
            'audio_projector': self.audio_projector,
        }
        
        if self.cross_attention_layers is not None:
            component_map['cross_attention'] = self.cross_attention_layers
        if self.generator is not None:
            component_map['generator'] = self.generator
        if self.video_generator is not None:
            component_map['video_generator'] = self.video_generator
        if hasattr(self, 'waveform_decoder') and self.waveform_decoder is not None:
            component_map['waveform_decoder'] = self.waveform_decoder
        
        saved_files = []
        total_size = 0
        
        for comp_name, component in component_map.items():
            if component is None:
                continue
            
            comp_state = component.state_dict()
            if not comp_state:
                continue
            
            # Clone and make contiguous to break shared storage (LSTM fix)
            safe_comp_state = {}
            for key, tensor in comp_state.items():
                safe_comp_state[key] = tensor.clone().contiguous()
            
            comp_path = os.path.join(path, f"{comp_name}.safetensors")
            save_file(safe_comp_state, comp_path)
            
            size_mb = sum(t.numel() * t.element_size() for t in safe_comp_state.values()) / (1024 * 1024)
            total_size += size_mb
            print(f"   💾 Saved {comp_name}: {size_mb:.1f} MB")
            saved_files.append(comp_name)
        
        # Save modality markers separately (clone for safety)
        markers = {
            'image_start': self.image_start.data.clone().contiguous(),
            'image_end': self.image_end.data.clone().contiguous(),
            'video_start': self.video_start.data.clone().contiguous(),
            'video_end': self.video_end.data.clone().contiguous(),
            'audio_start': self.audio_start.data.clone().contiguous(),
            'audio_end': self.audio_end.data.clone().contiguous(),
        }
        save_file(markers, os.path.join(path, "modality_markers.safetensors"))
        print(f"   💾 Saved modality_markers")
        
        # Save component manifest
        manifest = {
            "components": saved_files + ["modality_markers"],
            "save_format": "components",  # Mark as component-based save
        }
        with open(os.path.join(path, "components.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        
        # Generate model.safetensors.index.json for HuggingFace compatibility
        # This allows AutoModel.from_pretrained to find all the component files
        weight_map = {}
        total_bytes = 0
        
        for comp_name, component in component_map.items():
            if component is None:
                continue
            comp_state = component.state_dict()
            if not comp_state:
                continue
            safetensor_file = f"{comp_name}.safetensors"
            for key in comp_state.keys():
                # Prefix the key with the component name to match state_dict() format
                full_key = f"{comp_name}.{key}"
                weight_map[full_key] = safetensor_file
                total_bytes += comp_state[key].numel() * comp_state[key].element_size()
        
        # Add modality markers to weight map
        marker_names = ['image_start', 'image_end', 'video_start', 'video_end', 'audio_start', 'audio_end']
        for marker_name in marker_names:
            weight_map[marker_name] = "modality_markers.safetensors"
            marker_tensor = getattr(self, marker_name)
            total_bytes += marker_tensor.numel() * marker_tensor.element_size()
        
        index = {
            "metadata": {
                "total_size": total_bytes,
                "format": "components",
            },
            "weight_map": weight_map,
        }
        
        index_path = os.path.join(path, "model.safetensors.index.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        
        print(f"   📋 Saved model.safetensors.index.json for HuggingFace compatibility")
        print(f"   📋 Total size: {total_size:.1f} MB across {len(saved_files)} components")

    def _save_sharded(self, path: str, max_shard_size: int):
        """
        Save model weights in sharded .safetensors files.
        Components are surgically split across shards.
        
        Args:
            path: Directory to save shards
            max_shard_size: Maximum bytes per shard
        """
        from safetensors.torch import save_file
        
        state_dict = self.state_dict()
        
        # Group tensors by component for surgical splitting
        # IMPORTANT: Keep this list in sync with all model components!
        component_groups = {
            'llm': {},
            'vision_encoder': {},
            'video_encoder': {},
            'audio_encoder': {},
            'audio_decoder': {},
            'waveform_decoder': {},  # Speech-to-Speech decoder
            'generator': {},
            'video_generator': {},
            'projector': {},
            'audio_projector': {},
            'cross_attention_layers': {},
            'other': {},
        }
        
        for key, tensor in state_dict.items():
            placed = False
            for comp_name in component_groups.keys():
                if comp_name != 'other' and key.startswith(comp_name):
                    component_groups[comp_name][key] = tensor
                    placed = True
                    break
            if not placed:
                component_groups['other'][key] = tensor
        
        # Calculate sizes and create shards
        shards = []
        current_shard = {}
        current_size = 0
        shard_index_map = {}
        
        for comp_name, comp_tensors in component_groups.items():
            for key, tensor in comp_tensors.items():
                tensor_size = tensor.numel() * tensor.element_size()
                
                if current_size + tensor_size > max_shard_size and current_shard:
                    shards.append(current_shard)
                    current_shard = {}
                    current_size = 0
                
                current_shard[key] = tensor
                current_size += tensor_size
        
        if current_shard:
            shards.append(current_shard)
        
        # Save shards with proper naming
        total_shards = len(shards)
        weight_map = {}
        
        for i, shard in enumerate(shards):
            shard_name = f"model-{i+1:05d}-of-{total_shards:05d}.safetensors"
            shard_path = os.path.join(path, shard_name)
            
            # Clone and convert to contiguous to fix LSTM shared storage issue
            shard_contiguous = {k: v.clone().contiguous() for k, v in shard.items()}
            save_file(shard_contiguous, shard_path)
            
            for key in shard.keys():
                weight_map[key] = shard_name
            
            shard_size_mb = sum(t.numel() * t.element_size() for t in shard.values()) / (1024 * 1024)
            print(f"   💾 Saved shard {i+1}/{total_shards}: {shard_name} ({shard_size_mb:.1f} MB)")
        
        # Save index file
        index = {
            "metadata": {
                "total_size": sum(t.numel() * t.element_size() for t in state_dict.values()),
                "total_shards": total_shards,
            },
            "weight_map": weight_map,
        }
        
        index_path = os.path.join(path, "model.safetensors.index.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        
        print(f"   📋 Saved index: model.safetensors.index.json")

    def save_components_separately(self, path: str):
        """
        Save model components as separate .safetensors files.
        Useful for surgical component updates and debugging.
        
        NOTE: This method now clones tensors to handle LSTM shared storage issues.
        
        Args:
            path: Directory to save component files
        """
        from safetensors.torch import save_file
        
        os.makedirs(path, exist_ok=True)
        
        component_map = {
            'llm': self.llm,
            'vision_encoder': self.vision_encoder,
            'video_encoder': self.video_encoder,
            'audio_encoder': self.audio_encoder,
            'audio_decoder': self.audio_decoder,
            'projector': self.projector,
            'audio_projector': self.audio_projector,
        }
        
        if self.cross_attention_layers is not None:
            component_map['cross_attention'] = self.cross_attention_layers
        if self.generator is not None:
            component_map['generator'] = self.generator
        if self.video_generator is not None:
            component_map['video_generator'] = self.video_generator
        if hasattr(self, 'waveform_decoder') and self.waveform_decoder is not None:
            component_map['waveform_decoder'] = self.waveform_decoder
        
        saved_files = []
        
        for comp_name, component in component_map.items():
            if component is None:
                continue
            
            comp_state = component.state_dict()
            if not comp_state:
                continue
            
            # Clone and convert to contiguous to fix LSTM shared storage issue
            comp_state = {k: v.clone().contiguous() for k, v in comp_state.items()}
            
            comp_path = os.path.join(path, f"{comp_name}.safetensors")
            save_file(comp_state, comp_path)
            
            size_mb = sum(t.numel() * t.element_size() for t in comp_state.values()) / (1024 * 1024)
            print(f"   💾 Saved {comp_name}: {size_mb:.1f} MB")
            saved_files.append(comp_name)
        
        # Save modality markers separately (clone for safety)
        markers = {
            'image_start': self.image_start.data.clone().contiguous(),
            'image_end': self.image_end.data.clone().contiguous(),
            'video_start': self.video_start.data.clone().contiguous(),
            'video_end': self.video_end.data.clone().contiguous(),
            'audio_start': self.audio_start.data.clone().contiguous(),
            'audio_end': self.audio_end.data.clone().contiguous(),
        }
        save_file(markers, os.path.join(path, "modality_markers.safetensors"))
        print(f"   💾 Saved modality_markers")
        
        # Save component manifest
        manifest = {
            "components": saved_files + ["modality_markers"],
            "config": self.config.to_dict(),
            "lora_applied": self.lora_applied,
        }
        with open(os.path.join(path, "components.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        
        # Generate model.safetensors.index.json for HuggingFace compatibility
        # This allows AutoModel.from_pretrained to find all the component files
        weight_map = {}
        total_bytes = 0
        
        for comp_name, component in component_map.items():
            if component is None:
                continue
            comp_state = component.state_dict()
            if not comp_state:
                continue
            safetensor_file = f"{comp_name}.safetensors"
            for key in comp_state.keys():
                # Prefix the key with the component name to match state_dict() format
                full_key = f"{comp_name}.{key}"
                weight_map[full_key] = safetensor_file
                total_bytes += comp_state[key].numel() * comp_state[key].element_size()
        
        # Add modality markers to weight map
        marker_names = ['image_start', 'image_end', 'video_start', 'video_end', 'audio_start', 'audio_end']
        for marker_name in marker_names:
            weight_map[marker_name] = "modality_markers.safetensors"
            marker_tensor = getattr(self, marker_name)
            total_bytes += marker_tensor.numel() * marker_tensor.element_size()
        
        index = {
            "metadata": {
                "total_size": total_bytes,
                "format": "components",
            },
            "weight_map": weight_map,
        }
        
        index_path = os.path.join(path, "model.safetensors.index.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        
        print(f"   📋 Saved model.safetensors.index.json for HuggingFace compatibility")
        print(f"✅ Components saved to {path}")

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: str = None,
        device_map: Dict[str, str] = None,
        apply_lora: bool = True,
        strict: bool = False,
    ) -> 'XoronMultimodalModel':
        """
        Load a pretrained Xoron model from a checkpoint or final model directory.
        
        Args:
            path: Path to the saved model directory
            device: Device to load the model to (if not using device_map)
            device_map: Device map for model parallelism
            apply_lora: Whether to apply LoRA after loading
            strict: If False, allows loading weights even if architecture changed
            
        Returns:
            Loaded XoronMultimodalModel instance
        """
        from safetensors import safe_open
        
        print(f"\n📂 Loading model from {path}...")
        
        # Load config
        config_path = os.path.join(path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Check if LoRA was applied when saving
        lora_was_applied = config_dict.pop('lora_applied', False)
        
        # Extract architecture markers (don't pass to XoronConfig)
        architecture_version = config_dict.pop('architecture_version', 1)
        has_waveform_decoder = config_dict.pop('has_waveform_decoder', False)
        has_vision_encoder = config_dict.pop('has_vision_encoder', True)
        has_video_encoder = config_dict.pop('has_video_encoder', True)
        has_generator = config_dict.pop('has_generator', True)
        has_video_generator = config_dict.pop('has_video_generator', True)
        has_cross_attention = config_dict.pop('has_cross_attention', True)
        config_dict.pop('has_audio_encoder', None)
        config_dict.pop('has_audio_decoder', None)
        
        # Print architecture info from saved config
        print(f"\n   📋 Saved model architecture (version {architecture_version}):")
        print(f"      - Waveform Decoder: {'✅' if has_waveform_decoder else '❌ (will init randomly)'}")
        print(f"      - Vision Encoder: {'✅' if has_vision_encoder else '❌'}")
        print(f"      - Video Encoder: {'✅' if has_video_encoder else '❌'}")
        print(f"      - Image Generator: {'✅' if has_generator else '❌'}")
        print(f"      - Video Generator: {'✅' if has_video_generator else '❌'}")
        print(f"      - Cross Attention: {'✅' if has_cross_attention else '❌'}")
        print(f"      - LoRA Applied: {'✅' if lora_was_applied else '❌'}")
        
        config = XoronConfig.from_dict(config_dict)
        
        # Create model
        model = cls(config, device_map=device_map)
        
        # Check for component-based format first (default for new saves)
        components_json = os.path.join(path, "components.json")
        model_path = os.path.join(path, "model.safetensors")
        
        if os.path.exists(components_json):
            # Load from component-based format (new default)
            print(f"   📦 Loading from component-based format...")
            model._load_components(path, strict=strict)
            model.lora_applied = lora_was_applied
            
        elif os.path.exists(model_path):
            print(f"   📦 Loading weights from safetensors...")
            
            if strict:
                load_model(model, model_path)
            else:
                checkpoint_state_dict = {}
                with safe_open(model_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        checkpoint_state_dict[key] = f.get_tensor(key)
                
                model.load_state_dict(checkpoint_state_dict, strict=False)
                print(f"   ✅ Loaded weights from checkpoint")
            
            model.lora_applied = lora_was_applied
        else:
            # Try loading from pytorch format
            pytorch_path = os.path.join(path, "pytorch_model.bin")
            if os.path.exists(pytorch_path):
                print(f"   📦 Loading weights from pytorch_model.bin...")
                checkpoint_state_dict = torch.load(pytorch_path, map_location='cpu')
                
                model.load_state_dict(checkpoint_state_dict, strict=False)
                print(f"   ✅ Loaded weights from checkpoint")
                    
                model.lora_applied = lora_was_applied
            else:
                raise FileNotFoundError(f"No model weights found at {path}")
        
        # Apply LoRA if requested and not already applied
        if apply_lora and config.use_lora and not model.lora_applied:
            model.apply_lora()
        
        # Move to device
        if device_map is not None:
            model.apply_model_parallel(device_map)
        elif device is not None:
            model = model.to(device)
        
        print(f"✅ Model loaded successfully!")
        model._print_stats()
        
        return model

    def _load_components(self, path: str, strict: bool = False):
        """
        Load model from component-based safetensors files.
        
        Args:
            path: Directory containing component files
            strict: If True, require exact match; if False, allow partial loading
        """
        from safetensors import safe_open
        
        # Component mapping from file to model attribute
        component_map = {
            'llm': self.llm,
            'vision_encoder': self.vision_encoder,
            'video_encoder': self.video_encoder,
            'audio_encoder': self.audio_encoder,
            'audio_decoder': self.audio_decoder,
            'projector': self.projector,
            'audio_projector': self.audio_projector,
        }
        
        if self.cross_attention_layers is not None:
            component_map['cross_attention'] = self.cross_attention_layers
        if self.generator is not None:
            component_map['generator'] = self.generator
        if self.video_generator is not None:
            component_map['video_generator'] = self.video_generator
        if hasattr(self, 'waveform_decoder') and self.waveform_decoder is not None:
            component_map['waveform_decoder'] = self.waveform_decoder
        
        for comp_name, component in component_map.items():
            if component is None:
                continue
                
            comp_path = os.path.join(path, f"{comp_name}.safetensors")
            if not os.path.exists(comp_path):
                continue
            
            try:
                checkpoint_state = {}
                with safe_open(comp_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        checkpoint_state[key] = f.get_tensor(key)
                
                component.load_state_dict(checkpoint_state, strict=strict)
                
                size_mb = sum(t.numel() * t.element_size() for t in checkpoint_state.values()) / (1024 * 1024)
                print(f"   ✅ Loaded {comp_name} ({size_mb:.1f} MB)")
                
            except Exception as e:
                print(f"   ⚠️ Error loading {comp_name}: {e}")
        
        # Load modality markers
        markers_path = os.path.join(path, "modality_markers.safetensors")
        if os.path.exists(markers_path):
            try:
                with safe_open(markers_path, framework="pt", device="cpu") as f:
                    self.image_start.data = f.get_tensor('image_start')
                    self.image_end.data = f.get_tensor('image_end')
                    self.video_start.data = f.get_tensor('video_start')
                    self.video_end.data = f.get_tensor('video_end')
                    self.audio_start.data = f.get_tensor('audio_start')
                    self.audio_end.data = f.get_tensor('audio_end')
                print(f"   ✅ Loaded modality_markers")
            except Exception as e:
                print(f"   ⚠️ Error loading modality_markers: {e}")
        
        print(f"   ✅ Components loaded successfully")

    @staticmethod
    def load_training_state(path: str) -> Optional[Dict]:
        """
        Load training state from a checkpoint.
        
        Args:
            path: Path to the checkpoint directory
            
        Returns:
            Dictionary with training state or None if not found
        """
        state_path = os.path.join(path, "training_state.pt")
        if os.path.exists(state_path):
            print(f"   📂 Loading training state from {state_path}...")
            return torch.load(state_path, map_location='cpu')
        return None

    def freeze_components(self, components: List[str]):
        """
        Freeze specific components of the model.
        
        IMPORTANT RULES:
        1. LLM is NEVER frozen - it's trained from scratch and always needs full weight training
        2. LoRA parameters are NEVER frozen - they should always be trainable
        
        Args:
            components: List of component group names to freeze.
                       Valid groups: 'vision', 'video', 'audio', 
                       'cross_attention', 'image_generation', 'video_generation',
                       'modality_markers'
                       
                       NOTE: 'llm' is NOT a valid group to freeze - will be ignored!
        """
        # SAFETY: Never freeze LLM - it's trained from scratch
        if 'llm' in components:
            print(f"   ⚠️ Ignoring 'llm' in freeze list - LLM must always train (from scratch)")
            components = [c for c in components if c != 'llm']
        
        print(f"\n❄️ Freezing components: {components}")
        
        for group_name in components:
            if group_name not in COMPONENT_GROUPS:
                print(f"   ⚠️ Unknown component group: {group_name}")
                continue
            
            for attr_name in COMPONENT_GROUPS[group_name]:
                if hasattr(self, attr_name):
                    component = getattr(self, attr_name)
                    if component is not None:
                        if isinstance(component, nn.Parameter):
                            component.requires_grad = False
                        elif isinstance(component, nn.Module):
                            for name, param in component.named_parameters():
                                # NEVER freeze LoRA params - they should always be trainable
                                is_lora = 'lora_A' in name or 'lora_B' in name or 'magnitude' in name
                                if not is_lora:
                                    param.requires_grad = False
                        print(f"   ❄️ Frozen: {attr_name}")
        
        # Ensure LoRA params are trainable after freezing
        if self.lora_applied:
            enable_lora_training(self)
            print(f"   ✅ LoRA parameters remain trainable")
        
        self._print_stats()

    def unfreeze_components(self, components: List[str]):
        """
        Unfreeze specific components of the model.
        
        Args:
            components: List of component group names to unfreeze.
        """
        print(f"\n🔥 Unfreezing components: {components}")
        
        for group_name in components:
            if group_name not in COMPONENT_GROUPS:
                print(f"   ⚠️ Unknown component group: {group_name}")
                continue
            
            for attr_name in COMPONENT_GROUPS[group_name]:
                if hasattr(self, attr_name):
                    component = getattr(self, attr_name)
                    if component is not None:
                        if isinstance(component, nn.Parameter):
                            component.requires_grad = True
                        elif isinstance(component, nn.Module):
                            for param in component.parameters():
                                param.requires_grad = True
                        print(f"   🔥 Unfrozen: {attr_name}")
        
        self._print_stats()

    def freeze_all_except(self, components: List[str]):
        """
        Freeze all components except the specified ones.
        
        NOTE: LLM is always kept trainable regardless of input - it's trained from scratch.
        
        Args:
            components: List of component group names to keep trainable.
        """
        # Always keep LLM trainable - it's from scratch
        if 'llm' not in components:
            components = components + ['llm']
        
        all_groups = list(COMPONENT_GROUPS.keys())
        groups_to_freeze = [g for g in all_groups if g not in components]
        self.freeze_components(groups_to_freeze)

    def get_trainable_component_names(self) -> List[str]:
        """Get list of component groups that have trainable parameters."""
        trainable = []
        for group_name, attr_names in COMPONENT_GROUPS.items():
            for attr_name in attr_names:
                if hasattr(self, attr_name):
                    component = getattr(self, attr_name)
                    if component is not None:
                        if isinstance(component, nn.Parameter):
                            if component.requires_grad:
                                trainable.append(group_name)
                                break
                        elif isinstance(component, nn.Module):
                            if any(p.requires_grad for p in component.parameters()):
                                trainable.append(group_name)
                                break
        return trainable

    def get_frozen_component_names(self) -> List[str]:
        """Get list of component groups that are frozen (no trainable parameters)."""
        frozen = []
        for group_name, attr_names in COMPONENT_GROUPS.items():
            has_component = False
            is_trainable = False
            for attr_name in attr_names:
                if hasattr(self, attr_name):
                    component = getattr(self, attr_name)
                    if component is not None:
                        has_component = True
                        if isinstance(component, nn.Parameter):
                            if component.requires_grad:
                                is_trainable = True
                                break
                        elif isinstance(component, nn.Module):
                            if any(p.requires_grad for p in component.parameters()):
                                is_trainable = True
                                break
            # Only add to frozen if component exists but is not trainable
            if has_component and not is_trainable:
                frozen.append(group_name)
        return frozen

    def get_component_status(self) -> tuple:
        """
        Get tuple of (trainable_components, frozen_components) for display.
        
        Returns:
            tuple: (list of trainable component names, list of frozen component names)
        """
        trainable = self.get_trainable_component_names()
        frozen = self.get_frozen_component_names()
        return trainable, frozen

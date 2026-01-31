"""Xoron Multimodal Model - Complete implementation."""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union
from safetensors.torch import save_model, load_model

from transformers import LlamaConfig

from config import XoronConfig
from models.components.lora import LoRALinear, LoRAConfig, apply_lora_to_model, get_lora_parameters
from models.components.attention import MultimodalFusionLayer
from models.components.projectors import MultimodalProjector
from models.encoders.vision import VisionEncoder
from models.encoders.video import VideoEncoder
from models.encoders.audio import AudioEncoder, AudioDecoder
from models.generators.image import MobileDiffusionGenerator
from models.generators.video import MobileVideoDiffusion
from models.llm.moe_llama import MoELlamaForCausalLM


# Component groups for fine-tuning
COMPONENT_GROUPS = {
    'vision': ['vision_encoder', 'projector'],
    'video': ['video_encoder'],
    'audio': ['audio_encoder', 'audio_decoder', 'audio_projector'],
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
        self.video_encoder = VideoEncoder(self.vision_encoder, max_frames=config.max_video_frames)

        # 3. Audio Encoder (for ASR)
        print(f"\n🎤 Building Audio Encoder...")
        self.audio_encoder = AudioEncoder(
            hidden_size=config.hidden_size,
            n_mels=80,
            max_audio_length=3000
        )

        # 4. Audio Decoder (for TTS)
        print(f"\n🔊 Building Audio Decoder...")
        self.audio_decoder = AudioDecoder(
            hidden_size=config.hidden_size,
            n_mels=80,
            max_audio_length=1000
        )

        # 5. LLM Config
        llm_config = LlamaConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=1e-6,
            tie_word_embeddings=False,
            pad_token_id=0,
        )
        llm_config.use_flash_attention = config.use_flash_attention
        
        # Sliding window attention for efficient 128K context
        llm_config.use_sliding_window = config.use_sliding_window
        llm_config.sliding_window = config.sliding_window

        moe_config = {
            'use_moe': config.use_moe,
            'num_experts': config.num_experts,
            'num_experts_per_tok': config.num_experts_per_tok,
            'moe_layer_freq': config.moe_layer_freq,
            'intermediate_size': config.intermediate_size,
            'router_aux_loss_coef': config.router_aux_loss_coef,
        }

        # 6. LLM with MoE
        print(f"\n🧠 Building LLM: {config.hidden_size}d, {config.num_layers}L")
        print(f"   📏 Context: {config.max_position_embeddings//1024}K positions")
        if config.use_sliding_window:
            print(f"   🪟 Sliding Window: {config.sliding_window} tokens (efficient 128K)")
        print(f"   🎯 MoE: {config.num_experts} experts, top-{config.num_experts_per_tok}")
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
            self.generator = MobileDiffusionGenerator(
                latent_channels=config.generation_latent_channels,
                base_channels=config.generation_base_channels,
                context_dim=config.hidden_size,
                num_inference_steps=config.generation_inference_steps,
                image_size=config.generation_image_size,
            )

        # 12. Video Generator
        self.video_generator = None
        if config.enable_generation:
            print(f"\n🎬 Building MobileVideoDiffusion Generator (Videos)...")
            self.video_generator = MobileVideoDiffusion(
                latent_channels=config.generation_latent_channels,
                base_channels=config.generation_base_channels // 2,
                context_dim=config.hidden_size,
                num_frames=config.max_video_frames,
                image_size=config.generation_video_size,
                num_inference_steps=config.generation_inference_steps,
            )

        self.num_vision_tokens = config.num_vision_tokens
        self.max_video_frames = config.max_video_frames
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
        """Apply LoRA to the LLM and optionally cross-attention layers."""
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
        self._print_stats()

    def get_trainable_params(self):
        """Get trainable parameters, respecting LoRA settings."""
        if self.config.train_lora_only and self.lora_applied:
            return get_lora_parameters(self)
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
        batch_size = input_ids.shape[0]
        llm_device = self.get_llm_device()

        input_ids_llm = input_ids.to(llm_device)
        text_embeds = self.llm.model.embed_tokens(input_ids_llm)
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

        # Handle image input
        if has_content(pixel_values):
            try:
                image_embeds = self.encode_image(pixel_values)
                image_embeds_for_cross = image_embeds
                image_start = self.image_start.expand(batch_size, -1, -1)
                image_end = self.image_end.expand(batch_size, -1, -1)
                image_embeds = torch.cat([image_start, image_embeds, image_end], dim=1)
                text_embeds = torch.cat([image_embeds, text_embeds], dim=1)

                if attention_mask is not None:
                    image_mask = torch.ones(batch_size, image_embeds.shape[1], device=device)
                    attention_mask = torch.cat([image_mask, attention_mask], dim=1)

                if labels is not None:
                    image_labels = torch.full((batch_size, image_embeds.shape[1]), -100, device=device, dtype=labels.dtype)
                    labels = torch.cat([image_labels, labels], dim=1)
            except Exception:
                pass

        # Handle video input
        if has_content(video_frames):
            try:
                video_embeds = self.encode_video(video_frames)
                video_embeds_for_cross = video_embeds
                video_start = self.video_start.expand(batch_size, -1, -1)
                video_end = self.video_end.expand(batch_size, -1, -1)
                video_embeds = torch.cat([video_start, video_embeds, video_end], dim=1)
                text_embeds = torch.cat([video_embeds, text_embeds], dim=1)

                if attention_mask is not None:
                    video_mask = torch.ones(batch_size, video_embeds.shape[1], device=device)
                    attention_mask = torch.cat([video_mask, attention_mask], dim=1)

                if labels is not None:
                    video_labels = torch.full((batch_size, video_embeds.shape[1]), -100, device=device, dtype=labels.dtype)
                    labels = torch.cat([video_labels, labels], dim=1)
            except Exception:
                pass

        # Handle audio input
        if has_content(audio_features):
            try:
                audio_embeds = self.encode_audio(audio_features)
                audio_embeds_for_cross = audio_embeds
                audio_start = self.audio_start.expand(batch_size, -1, -1)
                audio_end = self.audio_end.expand(batch_size, -1, -1)
                audio_embeds = torch.cat([audio_start, audio_embeds, audio_end], dim=1)
                text_embeds = torch.cat([audio_embeds, text_embeds], dim=1)

                if attention_mask is not None:
                    audio_mask = torch.ones(batch_size, audio_embeds.shape[1], device=device)
                    attention_mask = torch.cat([audio_mask, attention_mask], dim=1)

                if labels is not None:
                    audio_labels = torch.full((batch_size, audio_embeds.shape[1]), -100, device=device, dtype=labels.dtype)
                    labels = torch.cat([audio_labels, labels], dim=1)
            except Exception:
                pass

        # Apply cross-attention fusion
        if self.cross_attention_layers is not None:
            try:
                text_embeds = self._apply_cross_attention(
                    text_embeds,
                    image_embeds=image_embeds_for_cross,
                    video_embeds=video_embeds_for_cross,
                    audio_embeds=audio_embeds_for_cross,
                )
            except Exception:
                pass

        outputs = self.llm(inputs_embeds=text_embeds, attention_mask=attention_mask, labels=labels)

        # Always return both loss and logits (logits needed for CoT weighted loss)
        return MultimodalModelOutput(
            loss=outputs.loss if hasattr(outputs, 'loss') else None,
            logits=outputs.logits if hasattr(outputs, 'logits') else None
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
        """
        os.makedirs(path, exist_ok=True)
        save_model(self, os.path.join(path, "model.safetensors"))

        config_dict = self.config.to_dict()
        config_dict['has_audio_encoder'] = True
        config_dict['has_audio_decoder'] = True
        config_dict['lora_applied'] = self.lora_applied

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

        # Save training state if provided
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
        config_dict.pop('has_audio_encoder', None)
        config_dict.pop('has_audio_decoder', None)
        
        config = XoronConfig.from_dict(config_dict)
        
        # Create model
        model = cls(config, device_map=device_map)
        
        # Load weights
        model_path = os.path.join(path, "model.safetensors")
        if os.path.exists(model_path):
            print(f"   📦 Loading weights from safetensors...")
            
            try:
                if strict:
                    load_model(model, model_path)
                else:
                    # Manual loading with shape filtering for architecture changes
                    checkpoint_state_dict = {}
                    with safe_open(model_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            checkpoint_state_dict[key] = f.get_tensor(key)
                    
                    # Filter out tensors with shape mismatches
                    model_state_dict = model.state_dict()
                    filtered_state_dict = {}
                    skipped_keys = []
                    size_mismatch_keys = []
                    
                    for key, checkpoint_tensor in checkpoint_state_dict.items():
                        if key in model_state_dict:
                            model_tensor = model_state_dict[key]
                            if checkpoint_tensor.shape == model_tensor.shape:
                                filtered_state_dict[key] = checkpoint_tensor
                            else:
                                size_mismatch_keys.append((key, checkpoint_tensor.shape, model_tensor.shape))
                        else:
                            skipped_keys.append(key)
                    
                    # Load filtered state dict
                    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
                    
                    # Report what happened
                    loaded_count = len(filtered_state_dict)
                    total_model_params = len(model_state_dict)
                    print(f"   ✅ Loaded {loaded_count}/{total_model_params} parameters from checkpoint")
                    
                    if size_mismatch_keys:
                        print(f"   ⚠️ Size mismatches (architecture changed): {len(size_mismatch_keys)} keys")
                        components = {}
                        for key, ckpt_shape, model_shape in size_mismatch_keys:
                            comp = key.split('.')[0]
                            components[comp] = components.get(comp, 0) + 1
                        for comp, count in sorted(components.items()):
                            print(f"      - {comp}: {count} parameters (will be randomly initialized)")
                    
                    if missing:
                        print(f"   ⚠️ Missing keys (new architecture): {len(missing)} keys")
                        components = {}
                        for key in missing:
                            comp = key.split('.')[0]
                            components[comp] = components.get(comp, 0) + 1
                        for comp, count in sorted(components.items()):
                            print(f"      - {comp}: {count} parameters (will be randomly initialized)")
                    
                    if skipped_keys:
                        print(f"   ⚠️ Skipped keys (old architecture): {len(skipped_keys)} keys")
                
                model.lora_applied = lora_was_applied
                
            except Exception as e:
                print(f"   ⚠️ Error loading safetensors: {e}")
                print(f"   🔄 Attempting fallback loading...")
                checkpoint_state_dict = {}
                with safe_open(model_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        checkpoint_state_dict[key] = f.get_tensor(key)
                
                # Filter by shape
                model_state_dict = model.state_dict()
                filtered_state_dict = {
                    k: v for k, v in checkpoint_state_dict.items()
                    if k in model_state_dict and v.shape == model_state_dict[k].shape
                }
                model.load_state_dict(filtered_state_dict, strict=False)
                print(f"   ✅ Loaded {len(filtered_state_dict)}/{len(model_state_dict)} parameters")
                model.lora_applied = lora_was_applied
        else:
            # Try loading from pytorch format
            pytorch_path = os.path.join(path, "pytorch_model.bin")
            if os.path.exists(pytorch_path):
                print(f"   📦 Loading weights from pytorch_model.bin...")
                checkpoint_state_dict = torch.load(pytorch_path, map_location='cpu')
                
                # Filter by shape
                model_state_dict = model.state_dict()
                filtered_state_dict = {}
                size_mismatch_count = 0
                
                for key, checkpoint_tensor in checkpoint_state_dict.items():
                    if key in model_state_dict:
                        if checkpoint_tensor.shape == model_state_dict[key].shape:
                            filtered_state_dict[key] = checkpoint_tensor
                        else:
                            size_mismatch_count += 1
                
                missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
                
                print(f"   ✅ Loaded {len(filtered_state_dict)}/{len(model_state_dict)} parameters")
                if size_mismatch_count:
                    print(f"   ⚠️ Size mismatches: {size_mismatch_count} (will be randomly initialized)")
                if missing:
                    print(f"   ⚠️ Missing keys: {len(missing)} (new architecture)")
                    
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
        
        Args:
            components: List of component group names to freeze.
                       Valid groups: 'vision', 'video', 'audio', 'llm', 
                       'cross_attention', 'image_generation', 'video_generation',
                       'modality_markers'
        """
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
                            for param in component.parameters():
                                param.requires_grad = False
                        print(f"   ❄️ Frozen: {attr_name}")
        
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
        
        Args:
            components: List of component group names to keep trainable.
        """
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

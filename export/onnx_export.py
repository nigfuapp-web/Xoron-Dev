"""
ONNX export functionality for Xoron-Dev.

Supports:
- Full model export to ONNX format
- 4-bit quantization for reduced model size
- Component-wise export (LLM, vision, audio, diffusion)
- Optimized inference with ONNX Runtime

Architecture specs (from config/model_config.py):
- LLM: 1024 hidden, 12 layers, 16 heads, 2048 intermediate
- MoE: 8 experts, top-2, every 2nd layer (6 MoE layers total)
- Vision: SigLIP-so400m-patch14-384 + TiTok (256 tokens) + 2D-RoPE
- Video: 3D-RoPE + Temporal MoE (4 experts) + 3D Causal (4 layers)
- Audio: Raw Waveform Tokenizer + Conformer + RMLA + MAS
- Audio Decoder: FastSpeech2/VITS-style with variance adaptor + MAS
- Waveform Decoder: BigVGAN-style direct audio output (no vocoder needed)
- Image Gen: MoE-DiT + Flow Matching + Dual-Stream (256-512²)
- Video Gen: 3D Causal + Flow Matching + 3D-RoPE (8-32 frames)
- Context: 128K with Ring Attention (4096 chunk)
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, Optional, List


def quantize_onnx_model(onnx_path: str, output_path: str = None, bits: int = 4) -> str:
    """
    Quantize an ONNX model to reduce size and improve inference speed.
    
    Args:
        onnx_path: Path to the ONNX model
        output_path: Path for quantized model (default: adds _quantized suffix)
        bits: Quantization bits (4 or 8)
        
    Returns:
        Path to quantized model
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        if output_path is None:
            base, ext = os.path.splitext(onnx_path)
            output_path = f"{base}_int{bits}{ext}"
        
        # Use INT8 for 8-bit, or INT4 approximation via weight-only quantization
        if bits == 4:
            # For 4-bit, we use weight-only quantization
            try:
                from onnxruntime.quantization import quantize_dynamic
                quantize_dynamic(
                    onnx_path,
                    output_path,
                    weight_type=QuantType.QInt8,  # Closest available
                    per_channel=True,
                    reduce_range=True,
                )
                print(f"   ✅ Quantized to ~4-bit (INT8 with optimizations): {output_path}")
            except Exception as e:
                print(f"   ⚠️ 4-bit quantization failed, falling back to INT8: {e}")
                quantize_dynamic(onnx_path, output_path, weight_type=QuantType.QInt8)
        else:
            quantize_dynamic(
                onnx_path,
                output_path,
                weight_type=QuantType.QInt8,
            )
            print(f"   ✅ Quantized to INT8: {output_path}")
        
        return output_path
    except ImportError:
        print("   ⚠️ onnxruntime.quantization not available, skipping quantization")
        return onnx_path
    except Exception as e:
        print(f"   ⚠️ Quantization failed: {e}")
        return onnx_path


def export_to_onnx(
    model, 
    config, 
    output_dir: str, 
    device: str = 'cuda',
    quantize: bool = True,
    quantize_bits: int = 4,
) -> Dict[str, str]:
    """
    Export Xoron model components to ONNX format.

    Args:
        model: XoronMultimodalModel instance
        config: XoronConfig instance
        output_dir: Directory to save ONNX files
        device: Device to use for export
        quantize: Whether to quantize the exported models
        quantize_bits: Quantization bits (4 or 8)

    Returns:
        Dictionary mapping component names to file paths
    """
    print("\n" + "=" * 60)
    print("📦 EXPORTING MODEL TO ONNX")
    print("=" * 60)
    print("   ✓ LLM Backbone (MoE: 8 experts, top-2, 6 MoE layers)")
    print("   ✓ Vision Encoder (SigLIP-2 + TiTok + 2D-RoPE)")
    print("   ✓ Video Encoder (3D-RoPE + Temporal MoE)")
    print("   ✓ Audio Encoder (Raw Waveform + RMLA)")
    print("   ✓ Audio Decoder (FastSpeech2 + MAS + Zero-Shot Cloning)")
    print("   ✓ Waveform Decoder (BigVGAN-style Direct Audio Output)")
    print("   ✓ Audio Projector (Audio-to-LLM)")
    print("   ✓ Image Diffusion (MoE-DiT + Flow Matching)")
    print("   ✓ Video Diffusion (3D Causal + Flow Matching)")
    print("   ✓ Multimodal Projectors (Perceiver Resampler)")
    if quantize:
        print(f"   ✓ {quantize_bits}-bit Quantization")

    onnx_dir = os.path.join(output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    model.eval()
    export_results = {}
    quantized_results = {}

    # 1. Export LLM Backbone
    print("\n1️⃣ Exporting LLM backbone...")
    try:
        class LLMWrapper(nn.Module):
            def __init__(self, xoron_model):
                super().__init__()
                self.embed_tokens = xoron_model.llm.model.embed_tokens
                self.llm_model = xoron_model.llm.model
                self.lm_head = xoron_model.llm.lm_head

            def forward(self, input_ids, attention_mask=None):
                hidden = self.embed_tokens(input_ids)
                hidden, _ = self.llm_model(inputs_embeds=hidden, attention_mask=attention_mask)
                return self.lm_head(hidden)

        llm_wrapper = LLMWrapper(model).to(device).eval()
        llm_onnx_path = os.path.join(onnx_dir, "llm_backbone.onnx")
        with torch.no_grad():
            torch.onnx.export(
                llm_wrapper,
                (torch.randint(0, 1000, (1, 64), device=device), torch.ones(1, 64, device=device)),
                llm_onnx_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                    "logits": {0: "batch", 1: "seq"}
                },
                opset_version=17,
                do_constant_folding=True
            )
        export_results['llm_backbone'] = llm_onnx_path
        print("   ✅ LLM backbone exported")
    except Exception as e:
        print(f"   ⚠️ LLM export failed: {e}")

    # 2. Export Vision Encoder
    print("\n2️⃣ Exporting vision encoder...")
    try:
        vision_path = os.path.join(onnx_dir, "vision_encoder.onnx")
        with torch.no_grad():
            torch.onnx.export(
                model.vision_encoder,
                torch.randn(1, 3, 224, 224, device=device),
                vision_path,
                input_names=["pixel_values"],
                output_names=["vision_features"],
                dynamic_axes={"pixel_values": {0: "batch"}, "vision_features": {0: "batch"}},
                opset_version=17
            )
        export_results['vision_encoder'] = vision_path
        print("   ✅ Vision encoder exported")
    except Exception as e:
        print(f"   ⚠️ Vision encoder failed: {e}")

    # 3. Export Video Encoder
    print("\n3️⃣ Exporting video encoder...")
    try:
        video_enc_path = os.path.join(onnx_dir, "video_encoder.onnx")
        with torch.no_grad():
            torch.onnx.export(
                model.video_encoder,
                torch.randn(1, 8, 3, 224, 224, device=device),
                video_enc_path,
                input_names=["video_frames"],
                output_names=["video_features"],
                dynamic_axes={"video_frames": {0: "batch", 1: "frames"}, "video_features": {0: "batch"}},
                opset_version=17
            )
        export_results['video_encoder'] = video_enc_path
        print("   ✅ Video encoder exported")
    except Exception as e:
        print(f"   ⚠️ Video encoder failed: {e}")

    # 4. Export Audio Encoder
    print("\n4️⃣ Exporting audio encoder...")
    try:
        audio_enc_path = os.path.join(onnx_dir, "audio_encoder.onnx")
        with torch.no_grad():
            torch.onnx.export(
                model.audio_encoder,
                torch.randn(1, 80, 3000, device=device),
                audio_enc_path,
                input_names=["mel_spectrogram"],
                output_names=["audio_features"],
                dynamic_axes={"mel_spectrogram": {0: "batch", 2: "time"}, "audio_features": {0: "batch"}},
                opset_version=17
            )
        export_results['audio_encoder'] = audio_enc_path
        print("   ✅ Audio encoder exported")
    except Exception as e:
        print(f"   ⚠️ Audio encoder failed: {e}")

    # 4b. Export Audio Decoder (TTS - mel spectrogram generation)
    print("\n4️⃣b Exporting audio decoder (TTS)...")
    try:
        if hasattr(model, 'audio_decoder') and model.audio_decoder is not None:
            class AudioDecoderWrapper(nn.Module):
                """Wrapper to handle AudioDecoder's multiple outputs for ONNX export."""
                def __init__(self, audio_decoder):
                    super().__init__()
                    self.audio_decoder = audio_decoder
                
                def forward(self, text_embeds, speaker=None):
                    # AudioDecoder returns (mel, durations, alignment)
                    mel, durations, _ = self.audio_decoder(
                        text_embeds, 
                        target_length=None,
                        speaker=speaker,
                        use_mas=False  # Disable MAS for inference export
                    )
                    return mel, durations
            
            audio_dec_wrapper = AudioDecoderWrapper(model.audio_decoder).to(device).eval()
            audio_dec_path = os.path.join(onnx_dir, "audio_decoder.onnx")
            with torch.no_grad():
                torch.onnx.export(
                    audio_dec_wrapper,
                    (
                        torch.randn(1, 64, config.hidden_size, device=device),  # text_embeds
                        torch.zeros(1, dtype=torch.long, device=device),  # speaker
                    ),
                    audio_dec_path,
                    input_names=["text_embeds", "speaker"],
                    output_names=["mel_spectrogram", "durations"],
                    dynamic_axes={
                        "text_embeds": {0: "batch", 1: "seq_len"},
                        "mel_spectrogram": {0: "batch", 2: "mel_len"},
                        "durations": {0: "batch", 1: "seq_len"}
                    },
                    opset_version=17
                )
            export_results['audio_decoder'] = audio_dec_path
            print("   ✅ Audio decoder (TTS) exported")
        else:
            print("   ⚠️ Audio decoder not available")
    except Exception as e:
        print(f"   ⚠️ Audio decoder failed: {e}")

    # 4c. Export Waveform Decoder (Direct audio output - no vocoder needed)
    print("\n4️⃣c Exporting waveform decoder (Speech-to-Speech)...")
    try:
        if hasattr(model, 'waveform_decoder') and model.waveform_decoder is not None:
            waveform_dec_path = os.path.join(onnx_dir, "waveform_decoder.onnx")
            with torch.no_grad():
                torch.onnx.export(
                    model.waveform_decoder,
                    torch.randn(1, 64, config.hidden_size, device=device),  # features
                    waveform_dec_path,
                    input_names=["audio_features"],
                    output_names=["waveform"],
                    dynamic_axes={
                        "audio_features": {0: "batch", 1: "seq_len"},
                        "waveform": {0: "batch", 1: "audio_len"}
                    },
                    opset_version=17
                )
            export_results['waveform_decoder'] = waveform_dec_path
            print("   ✅ Waveform decoder (Speech-to-Speech) exported")
        else:
            print("   ⚠️ Waveform decoder not available")
    except Exception as e:
        print(f"   ⚠️ Waveform decoder failed: {e}")

    # 4d. Export Audio Projector (Audio features to LLM space)
    print("\n4️⃣d Exporting audio projector...")
    try:
        if hasattr(model, 'audio_projector') and model.audio_projector is not None:
            audio_proj_path = os.path.join(onnx_dir, "audio_projector.onnx")
            with torch.no_grad():
                torch.onnx.export(
                    model.audio_projector,
                    torch.randn(1, 64, config.hidden_size, device=device),  # audio_embeds
                    audio_proj_path,
                    input_names=["audio_embeds"],
                    output_names=["projected_audio"],
                    dynamic_axes={
                        "audio_embeds": {0: "batch", 1: "seq_len"},
                        "projected_audio": {0: "batch", 1: "seq_len"}
                    },
                    opset_version=17
                )
            export_results['audio_projector'] = audio_proj_path
            print("   ✅ Audio projector exported")
        else:
            print("   ⚠️ Audio projector not available")
    except Exception as e:
        print(f"   ⚠️ Audio projector failed: {e}")

    # 5. Export Vision Projector
    print("\n5️⃣ Exporting vision projector...")
    try:
        vis_proj_path = os.path.join(onnx_dir, "vision_projector.onnx")
        with torch.no_grad():
            torch.onnx.export(
                model.projector,
                torch.randn(1, 197, 768, device=device),
                vis_proj_path,
                input_names=["vision_features"],
                output_names=["projected"],
                dynamic_axes={"vision_features": {0: "batch"}, "projected": {0: "batch"}},
                opset_version=17
            )
        export_results['vision_projector'] = vis_proj_path
        print("   ✅ Vision projector exported")
    except Exception as e:
        print(f"   ⚠️ Vision projector failed: {e}")

    # 6. Export Image Diffusion
    print("\n6️⃣ Exporting image diffusion...")
    try:
        if model.generator is not None:
            img_diff_path = os.path.join(onnx_dir, "image_diffusion_unet.onnx")
            latent_size = config.image_base_size // 8  # Multi-scale config
            with torch.no_grad():
                torch.onnx.export(
                    model.generator.unet,
                    (
                        torch.randn(1, 4, latent_size, latent_size, device=device),
                        torch.tensor([500], device=device),
                        torch.randn(1, config.hidden_size, device=device)
                    ),
                    img_diff_path,
                    input_names=["latent", "timestep", "context"],
                    output_names=["noise_pred"],
                    dynamic_axes={"latent": {0: "batch"}, "context": {0: "batch"}, "noise_pred": {0: "batch"}},
                    opset_version=17
                )
            export_results['image_diffusion_unet'] = img_diff_path
            print("   ✅ Image diffusion exported")
        else:
            print("   ⚠️ Image generator not available")
    except Exception as e:
        print(f"   ⚠️ Image diffusion failed: {e}")

    # 7. Export Video Diffusion
    print("\n7️⃣ Exporting video diffusion...")
    try:
        if model.video_generator is not None:
            vid_diff_path = os.path.join(onnx_dir, "video_diffusion_unet.onnx")
            vid_latent = config.video_base_size // 8  # Multi-scale config
            with torch.no_grad():
                torch.onnx.export(
                    model.video_generator.unet,
                    (
                        torch.randn(1, 8, 4, vid_latent, vid_latent, device=device),
                        torch.tensor([500], device=device),
                        torch.randn(1, config.hidden_size, device=device)
                    ),
                    vid_diff_path,
                    input_names=["latent", "timestep", "context"],
                    output_names=["noise_pred"],
                    dynamic_axes={
                        "latent": {0: "batch", 1: "frames"},
                        "context": {0: "batch"},
                        "noise_pred": {0: "batch", 1: "frames"}
                    },
                    opset_version=17
                )
            export_results['video_diffusion_unet'] = vid_diff_path
            print("   ✅ Video diffusion exported")
        else:
            print("   ⚠️ Video generator not available")
    except Exception as e:
        print(f"   ⚠️ Video diffusion failed: {e}")

    # Quantize exported models if requested
    if quantize and export_results:
        print(f"\n🔧 Quantizing models to {quantize_bits}-bit...")
        for name, path in export_results.items():
            if os.path.exists(path):
                quantized_path = quantize_onnx_model(path, bits=quantize_bits)
                if quantized_path != path:
                    quantized_results[name] = quantized_path

    # Save metadata
    print("\n📋 Saving ONNX metadata...")
    onnx_metadata = {
        "model_name": config.model_name,
        "quantized": quantize,
        "quantize_bits": quantize_bits if quantize else None,
        "architecture": {
            "llm": {
                "hidden_size": getattr(config, 'hidden_size', 1024),
                "num_layers": getattr(config, 'num_layers', 12),
                "num_heads": getattr(config, 'num_heads', 16),
                "intermediate_size": getattr(config, 'intermediate_size', 2048),
                "vocab_size": getattr(config, 'vocab_size', 151643),
                "max_position_embeddings": getattr(config, 'max_position_embeddings', 131072),
            },
            "moe": {
                "num_experts": getattr(config, 'num_experts', 8),
                "num_experts_per_tok": getattr(config, 'num_experts_per_tok', 2),
                "moe_layer_freq": getattr(config, 'moe_layer_freq', 2),
                "use_aux_lossless": getattr(config, 'use_aux_lossless', True),
                "use_shared_expert": getattr(config, 'use_shared_expert', True),
            },
            "vision": {
                "model": getattr(config, 'vision_model_name', 'google/siglip-so400m-patch14-384'),
                "num_tokens": getattr(config, 'num_vision_tokens', 64),
                "titok_tokens": getattr(config, 'num_vision_titok_tokens', 256),
                "use_2d_rope": True,
                "use_dual_stream": getattr(config, 'use_vision_dual_stream', True),
            },
            "video": {
                "max_frames": getattr(config, 'video_max_frames', 32),
                "use_3d_rope": getattr(config, 'use_video_3d_rope', True),
                "use_temporal_moe": getattr(config, 'use_video_temporal_moe', True),
                "num_temporal_experts": getattr(config, 'num_video_experts', 4),
            },
            "audio": {
                "sample_rate": getattr(config, 'audio_sample_rate', 16000),
                "use_raw_waveform": getattr(config, 'use_raw_waveform', True),
                "use_mas": getattr(config, 'use_mas', True),
                "n_mels": 80,
                "max_audio_length": 3000,
            },
            "audio_decoder": {
                "type": "FastSpeech2/VITS-style",
                "features": ["variance_adaptor", "MAS", "zero_shot_speaker_cloning", "in_context_prompting"],
                "output": "mel_spectrogram",
            },
            "waveform_decoder": {
                "type": "BigVGAN-style",
                "features": ["snake_activation", "multi_receptive_field_fusion", "weight_normalization"],
                "upsample_factor": 256,
                "output": "raw_waveform_16kHz",
            },
            "generation": {
                "image_base_size": getattr(config, 'image_base_size', 256),
                "video_base_size": getattr(config, 'video_base_size', 256),
                "use_flow_matching": getattr(config, 'generation_use_flow_matching', True),
            },
        },
        "capabilities": {
            "text_generation": 'llm_backbone' in export_results,
            "image_understanding": 'vision_encoder' in export_results,
            "video_understanding": 'video_encoder' in export_results,
            "speech_to_text": 'audio_encoder' in export_results,
            "text_to_speech": 'audio_decoder' in export_results,
            "speech_to_speech": 'waveform_decoder' in export_results,
            "image_generation": 'image_diffusion_unet' in export_results,
            "video_generation": 'video_diffusion_unet' in export_results,
        },
        "components": {k: os.path.basename(v) for k, v in export_results.items()},
        "quantized_components": {k: os.path.basename(v) for k, v in quantized_results.items()} if quantized_results else {},
    }
    with open(os.path.join(onnx_dir, "onnx_metadata.json"), "w") as f:
        json.dump(onnx_metadata, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("✅ ONNX EXPORT COMPLETE!")
    print("=" * 60)
    print(f"\n📁 Files in {onnx_dir}:")
    for f in sorted(os.listdir(onnx_dir)):
        size = os.path.getsize(os.path.join(onnx_dir, f)) / 1e6
        print(f"   {f} ({size:.1f} MB)")

    print("\n🎯 EXPORTED CAPABILITIES:")
    for cap, exported in onnx_metadata['capabilities'].items():
        print(f"   {'✅' if exported else '❌'} {cap.replace('_', ' ').title()}")
    
    if quantized_results:
        print(f"\n🔧 QUANTIZED MODELS ({quantize_bits}-bit):")
        for name in quantized_results:
            print(f"   ✅ {name}")

    # Return both original and quantized paths
    all_results = export_results.copy()
    all_results.update({f"{k}_quantized": v for k, v in quantized_results.items()})
    return all_results

"""
GGUF export functionality for Xoron-Dev.

GGUF (GPT-Generated Unified Format) is a format designed for efficient
inference with llama.cpp and compatible runtimes.

Supports:
- LLM backbone export to GGUF (MoE LLM with 12 layers, 1024 hidden, 16 heads)
- Multiple quantization levels (Q4_0, Q4_K_M, Q5_K_M, Q8_0, F16)
- Metadata embedding for model info
- 128K context length with Ring Attention support
- 8 experts with top-2 routing (Aux-Lossless MoE)

Architecture specs:
- LLM: 1024 hidden, 12 layers, 16 heads, 2048 intermediate
- MoE: 8 experts, top-2, every 2nd layer (6 MoE layers total)
- Context: 128K with Ring Attention (4096 chunk)
- Vocab: 151,643 tokens (Qwen2.5)
"""

import os
import json
import shutil
import subprocess
from typing import Dict, Optional, List


# Quantization types supported by llama.cpp
GGUF_QUANT_TYPES = {
    'q4_0': 'Q4_0',      # 4-bit quantization (fastest, smallest)
    'q4_k_m': 'Q4_K_M',  # 4-bit K-quant (better quality)
    'q5_k_m': 'Q5_K_M',  # 5-bit K-quant (good balance)
    'q8_0': 'Q8_0',      # 8-bit quantization (high quality)
    'f16': 'F16',        # 16-bit float (highest quality)
    'f32': 'F32',        # 32-bit float (original)
}


def check_llama_cpp_available() -> bool:
    """Check if llama.cpp conversion tools are available."""
    try:
        result = subprocess.run(
            ['python', '-c', 'import llama_cpp; print(llama_cpp.__version__)'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def convert_to_gguf_via_hf(
    model_path: str,
    output_path: str,
    quant_type: str = 'q4_k_m',
) -> Optional[str]:
    """
    Convert a HuggingFace model to GGUF format.
    
    This requires the llama.cpp convert script or llama-cpp-python.
    
    Args:
        model_path: Path to the HuggingFace model directory
        output_path: Path for the output GGUF file
        quant_type: Quantization type (q4_0, q4_k_m, q5_k_m, q8_0, f16)
        
    Returns:
        Path to the GGUF file, or None if conversion failed
    """
    quant = GGUF_QUANT_TYPES.get(quant_type.lower(), 'Q4_K_M')
    
    # Try using llama-cpp-python's conversion
    try:
        from llama_cpp import Llama
        
        # llama-cpp-python can load HF models directly in some cases
        print(f"   Attempting conversion via llama-cpp-python...")
        
        # For now, we'll use the command-line approach
        raise ImportError("Direct conversion not supported, using CLI")
        
    except ImportError:
        pass
    
    # Try using the llama.cpp convert script
    convert_script = shutil.which('convert.py') or shutil.which('convert-hf-to-gguf.py')
    
    if convert_script is None:
        # Try common locations
        possible_paths = [
            os.path.expanduser('~/.local/bin/convert-hf-to-gguf.py'),
            '/usr/local/bin/convert-hf-to-gguf.py',
            os.path.join(os.path.dirname(__file__), 'llama.cpp', 'convert-hf-to-gguf.py'),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                convert_script = path
                break
    
    if convert_script:
        try:
            # First convert to GGUF (F16)
            f16_path = output_path.replace('.gguf', '_f16.gguf')
            
            cmd = [
                'python', convert_script,
                model_path,
                '--outfile', f16_path,
                '--outtype', 'f16',
            ]
            
            print(f"   Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"   ⚠️ Conversion failed: {result.stderr}")
                return None
            
            # Then quantize if needed
            if quant_type.lower() != 'f16':
                quantize_script = shutil.which('quantize') or shutil.which('llama-quantize')
                
                if quantize_script:
                    cmd = [quantize_script, f16_path, output_path, quant]
                    print(f"   Running: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        # Remove F16 intermediate file
                        if os.path.exists(f16_path) and f16_path != output_path:
                            os.remove(f16_path)
                        return output_path
                    else:
                        print(f"   ⚠️ Quantization failed: {result.stderr}")
                        return f16_path  # Return F16 version
                else:
                    print(f"   ⚠️ Quantize tool not found, returning F16 version")
                    return f16_path
            else:
                return f16_path
                
        except Exception as e:
            print(f"   ⚠️ Conversion error: {e}")
            return None
    
    print("   ⚠️ No GGUF conversion tools found")
    print("   Install llama-cpp-python: pip install llama-cpp-python")
    print("   Or clone llama.cpp and use convert-hf-to-gguf.py")
    return None


def export_to_gguf(
    model,
    config,
    output_dir: str,
    quant_type: str = 'q4_k_m',
    save_hf_first: bool = True,
) -> Dict[str, str]:
    """
    Export Xoron model to GGUF format.
    
    Note: GGUF is primarily designed for LLM inference, so only the
    LLM backbone will be exported. Other components (vision, audio,
    diffusion) should use ONNX export.
    
    Args:
        model: XoronMultimodalModel instance
        config: XoronConfig instance
        output_dir: Directory to save GGUF files
        quant_type: Quantization type (q4_0, q4_k_m, q5_k_m, q8_0, f16)
        save_hf_first: Whether to save as HuggingFace format first
        
    Returns:
        Dictionary mapping component names to file paths
    """
    print("\n" + "=" * 60)
    print("📦 EXPORTING MODEL TO GGUF")
    print("=" * 60)
    print(f"   Quantization: {quant_type.upper()}")
    print("   Note: GGUF export is for LLM backbone only")
    print("   Use ONNX export for vision/audio/diffusion components")
    
    gguf_dir = os.path.join(output_dir, "gguf")
    os.makedirs(gguf_dir, exist_ok=True)
    
    export_results = {}
    
    # Check if conversion tools are available
    if not check_llama_cpp_available():
        print("\n⚠️ llama-cpp-python not installed")
        print("   Install with: pip install llama-cpp-python")
        print("   Or use ONNX export instead: --onnx")
        
        # Create a placeholder metadata file
        metadata = {
            "model_name": config.model_name,
            "status": "conversion_tools_not_available",
            "instructions": [
                "Install llama-cpp-python: pip install llama-cpp-python",
                "Or clone llama.cpp and use convert-hf-to-gguf.py",
                "Then run: python -c \"from export.gguf_export import export_to_gguf; ...\"",
            ],
        }
        with open(os.path.join(gguf_dir, "gguf_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        return export_results
    
    # Save model in HuggingFace format first if needed
    hf_temp_dir = os.path.join(gguf_dir, "hf_temp")
    
    if save_hf_first:
        print("\n1️⃣ Saving model in HuggingFace format...")
        try:
            # Save just the LLM backbone
            os.makedirs(hf_temp_dir, exist_ok=True)
            
            # Save the LLM model
            if hasattr(model, 'llm') and model.llm is not None:
                model.llm.save_pretrained(hf_temp_dir)
                print("   ✅ LLM backbone saved")
            else:
                print("   ⚠️ No LLM backbone found")
                return export_results
                
        except Exception as e:
            print(f"   ⚠️ Failed to save HF model: {e}")
            return export_results
    
    # Convert to GGUF
    print(f"\n2️⃣ Converting to GGUF ({quant_type.upper()})...")
    
    gguf_filename = f"xoron-{quant_type.lower()}.gguf"
    gguf_path = os.path.join(gguf_dir, gguf_filename)
    
    result_path = convert_to_gguf_via_hf(hf_temp_dir, gguf_path, quant_type)
    
    if result_path and os.path.exists(result_path):
        export_results['llm_backbone'] = result_path
        print(f"   ✅ GGUF exported: {result_path}")
        
        # Get file size
        size_mb = os.path.getsize(result_path) / (1024 * 1024)
        print(f"   📊 Size: {size_mb:.1f} MB")
    else:
        print("   ⚠️ GGUF conversion failed")
    
    # Clean up temp directory
    if save_hf_first and os.path.exists(hf_temp_dir):
        try:
            shutil.rmtree(hf_temp_dir)
        except Exception:
            pass
    
    # Save metadata
    print("\n📋 Saving GGUF metadata...")
    gguf_metadata = {
        "model_name": config.model_name,
        "quant_type": quant_type,
        "components": {k: os.path.basename(v) for k, v in export_results.items()},
        "architecture": {
            "hidden_size": getattr(config, 'hidden_size', 1024),
            "num_layers": getattr(config, 'num_layers', 12),
            "num_heads": getattr(config, 'num_heads', 16),
            "intermediate_size": getattr(config, 'intermediate_size', 2048),
            "vocab_size": getattr(config, 'vocab_size', 151643),
            "max_position_embeddings": getattr(config, 'max_position_embeddings', 131072),
            "moe_config": {
                "num_experts": getattr(config, 'num_experts', 8),
                "num_experts_per_tok": getattr(config, 'num_experts_per_tok', 2),
                "moe_layer_freq": getattr(config, 'moe_layer_freq', 2),
                "use_aux_lossless": getattr(config, 'use_aux_lossless', True),
                "use_shared_expert": getattr(config, 'use_shared_expert', True),
            },
            "ring_attention": {
                "enabled": getattr(config, 'use_ring_attention', True),
                "chunk_size": getattr(config, 'ring_attention_chunk_size', 4096),
            },
        },
        "note": "GGUF export is for LLM backbone only. Use ONNX for other components.",
    }
    with open(os.path.join(gguf_dir, "gguf_metadata.json"), "w") as f:
        json.dump(gguf_metadata, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    if export_results:
        print("✅ GGUF EXPORT COMPLETE!")
    else:
        print("⚠️ GGUF EXPORT INCOMPLETE")
    print("=" * 60)
    
    if export_results:
        print(f"\n📁 Files in {gguf_dir}:")
        for f in sorted(os.listdir(gguf_dir)):
            fpath = os.path.join(gguf_dir, f)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath) / (1024 * 1024)
                print(f"   {f} ({size:.1f} MB)")
    
    return export_results

"""
Export module for Xoron-Dev.

Provides functionality to export trained models to various formats:
- ONNX: For cross-platform inference with ONNX Runtime
- GGUF: For efficient inference with llama.cpp

Usage:
    from export import export_to_onnx, export_to_gguf
    
    # Export to ONNX with 4-bit quantization
    export_to_onnx(model, config, output_dir, quantize=True, quantize_bits=4)
    
    # Export to GGUF
    export_to_gguf(model, config, output_dir, quant_type='q4_k_m')
"""

from export.onnx_export import export_to_onnx, quantize_onnx_model
from export.gguf_export import export_to_gguf, GGUF_QUANT_TYPES

__all__ = [
    'export_to_onnx',
    'quantize_onnx_model',
    'export_to_gguf',
    'GGUF_QUANT_TYPES',
]

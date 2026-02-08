# 📦 Export Module Documentation

The Export module provides functionality to export Xoron-Dev models to various formats for deployment.

## 📁 File Structure

```
export/
├── __init__.py
├── gguf_export.py   # GGUF export for llama.cpp
└── onnx_export.py   # ONNX export for inference
```

---

## 🦙 GGUF Export

### Overview

GGUF (GPT-Generated Unified Format) is designed for efficient inference with llama.cpp and compatible runtimes.

### Supported Quantization Types

| Type | Description | Size | Quality |
|------|-------------|------|---------|
| `Q4_0` | 4-bit quantization | Smallest | Good |
| `Q4_K_M` | 4-bit K-quant | Small | Better |
| `Q5_K_M` | 5-bit K-quant | Medium | Good balance |
| `Q8_0` | 8-bit quantization | Large | High |
| `F16` | 16-bit float | Largest | Highest |

### Usage

```python
from export.gguf_export import convert_to_gguf_via_hf

# Convert model to GGUF
gguf_path = convert_to_gguf_via_hf(
    model_path='./xoron-final',
    output_path='./xoron.gguf',
    quant_type='q4_k_m',
)
```

### Architecture Specs

The GGUF export includes:
- LLM backbone: 1024 hidden, 12 layers, 16 heads
- MoE: 8 experts, top-2, every 2nd layer
- Context: 128K with Ring Attention
- Vocab: 151,643 tokens (Qwen2.5)

### Requirements

- llama.cpp or llama-cpp-python installed
- Sufficient disk space for conversion

---

## 🔷 ONNX Export

### Overview

ONNX (Open Neural Network Exchange) enables deployment across various inference runtimes.

### Supported Components

| Component | Export Support | Notes |
|-----------|---------------|-------|
| LLM | ✅ | Full model |
| Vision Encoder | ✅ | SigLIP backbone |
| Audio Encoder | ✅ | Conformer |
| Image Generator | ⚠️ | Partial (DiT only) |
| Video Generator | ⚠️ | Partial |

### Usage

```python
from export.onnx_export import export_to_onnx, quantize_onnx_model

# Export model components
onnx_paths = export_to_onnx(
    model=model,
    config=config,
    output_dir='./onnx_export',
    device='cuda',
    quantize=True,
    quantize_bits=4,
)

# Quantize existing ONNX model
quantized_path = quantize_onnx_model(
    onnx_path='./model.onnx',
    output_path='./model_int4.onnx',
    bits=4,
)
```

### Quantization

```python
def quantize_onnx_model(onnx_path: str, output_path: str = None, bits: int = 4) -> str:
    """
    Quantize an ONNX model.
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Output path (default: adds _quantized suffix)
        bits: Quantization bits (4 or 8)
    
    Returns:
        Path to quantized model
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    if bits == 4:
        # Weight-only quantization for 4-bit
        quantize_dynamic(
            onnx_path,
            output_path,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=True,
        )
    else:
        # Standard INT8 quantization
        quantize_dynamic(
            onnx_path,
            output_path,
            weight_type=QuantType.QInt8,
        )
    
    return output_path
```

### Export Function

```python
def export_to_onnx(
    model,
    config,
    output_dir: str,
    device: str = 'cuda',
    quantize: bool = True,
    quantize_bits: int = 4,
) -> Dict[str, str]:
    """
    Export Xoron model components to ONNX.
    
    Args:
        model: XoronMultimodalModel instance
        config: XoronConfig instance
        output_dir: Directory for ONNX files
        device: Device for export
        quantize: Whether to quantize
        quantize_bits: Quantization bits
    
    Returns:
        Dict mapping component names to ONNX paths
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    
    # Export LLM
    llm_path = os.path.join(output_dir, 'llm.onnx')
    export_llm_to_onnx(model.llm, llm_path, config)
    paths['llm'] = llm_path
    
    # Export Vision Encoder
    vision_path = os.path.join(output_dir, 'vision_encoder.onnx')
    export_vision_to_onnx(model.vision_encoder, vision_path, config)
    paths['vision_encoder'] = vision_path
    
    # Export Audio Encoder
    audio_path = os.path.join(output_dir, 'audio_encoder.onnx')
    export_audio_to_onnx(model.audio_encoder, audio_path, config)
    paths['audio_encoder'] = audio_path
    
    # Quantize if requested
    if quantize:
        for name, path in paths.items():
            quantized = quantize_onnx_model(path, bits=quantize_bits)
            paths[name] = quantized
    
    return paths
```

---

## 📊 Export Specifications

### Model Sizes (Approximate)

| Format | Quantization | Size |
|--------|--------------|------|
| PyTorch | FP16 | ~5.6 GB |
| GGUF | Q4_K_M | ~1.5 GB |
| GGUF | Q8_0 | ~2.8 GB |
| ONNX | INT8 | ~2.8 GB |
| ONNX | INT4 | ~1.5 GB |

### Inference Speed (Approximate)

| Runtime | Device | Tokens/sec |
|---------|--------|------------|
| PyTorch | A100 | ~50 |
| llama.cpp | CPU | ~10-20 |
| llama.cpp | GPU | ~40-60 |
| ONNX Runtime | CPU | ~15-25 |
| ONNX Runtime | GPU | ~45-55 |

---

## ⚠️ Limitations

### GGUF Export
- Only LLM backbone is exported
- Vision/audio encoders not included
- MoE may have compatibility issues

### ONNX Export
- Dynamic shapes may cause issues
- Some operations not supported
- Generator models partially supported

---

## 🔗 Related Documentation

- [Model Documentation](../models/llm.md) - Model architecture
- [Config Documentation](../config/README.md) - Model configuration

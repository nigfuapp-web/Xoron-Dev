"""
Synthetic dataset generation for Xoron-Dev multimodal model training.

This module provides generators for:
- Chain-of-thought reasoning datasets
- Agentic coding datasets (FIM, Git, Jupyter, shell execution)
- Anti-hallucination datasets (uncertainty, citations, fact-checking)
- Document processing datasets
- System administration datasets

Usage:
    from synth import generate_all_datasets
    generate_all_datasets('./synth/data', samples_per_type=2000)
    
    # Or use individual generators
    from synth.generator import ChainOfThoughtGenerator
    gen = ChainOfThoughtGenerator()
    example = gen.generate_arithmetic('medium')
"""

__all__ = [
    'ChainOfThoughtGenerator',
    'generate_all_datasets',
    'generate_cot_dataset',
    'generate_agentic_datasets',
    'generate_anti_hallucination_datasets',
    'generate_document_datasets',
    'generate_system_admin_datasets',
]


def get_generator():
    """Lazy import to avoid circular imports."""
    from synth.generator import ChainOfThoughtGenerator
    return ChainOfThoughtGenerator


def generate_all_datasets(output_dir: str, samples_per_type: int = 2000):
    """
    Generate all synthetic datasets.
    
    This is the main entry point for dataset generation. It uses the
    unique_generator module which provides parameterized generation
    to ensure no duplicate samples.
    
    Args:
        output_dir: Directory to save generated datasets
        samples_per_type: Number of samples per dataset type
    """
    from synth.unique_generator import generate_all_datasets as _generate
    return _generate(output_dir, samples_per_type)


def generate_cot_dataset(output_path: str, count: int = 300000, **kwargs):
    """Generate chain-of-thought reasoning dataset."""
    from synth.generator import ChainOfThoughtGenerator
    gen = ChainOfThoughtGenerator()
    return gen.generate_dataset(count, output_path, **kwargs)


def generate_agentic_datasets(output_dir: str, samples_per_type: int = 2000):
    """Generate agentic coding datasets (FIM, Git, Jupyter, etc.)."""
    from synth.agentic_dataset_generator import generate_dataset
    return generate_dataset(output_dir, samples_per_type)


def generate_anti_hallucination_datasets(output_dir: str, samples_per_type: int = 2000):
    """Generate anti-hallucination datasets (uncertainty, citations, etc.)."""
    from synth.anti_hallucination_generator import generate_dataset
    return generate_dataset(output_dir, samples_per_type)


def generate_document_datasets(output_dir: str, samples_per_type: int = 2000):
    """Generate document processing datasets."""
    from synth.document_generator import generate_dataset
    return generate_dataset(output_dir, samples_per_type)


def generate_system_admin_datasets(output_dir: str, samples_per_type: int = 2000):
    """Generate system administration datasets."""
    from synth.system_admin_generator import generate_dataset
    return generate_dataset(output_dir, samples_per_type)

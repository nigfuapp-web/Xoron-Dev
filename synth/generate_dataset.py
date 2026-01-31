#!/usr/bin/env python3
"""
Script to generate synthetic chain-of-thought reasoning dataset.

Usage:
    python -m synth.generate_dataset --output synth/data/cot_dataset.jsonl --count 300000
    
    # Generate smaller test dataset
    python -m synth.generate_dataset --output synth/data/cot_test.jsonl --count 1000
    
    # Generate with specific categories
    python -m synth.generate_dataset --output synth/data/math_only.jsonl --count 50000 --categories math_arithmetic math_word_problem math_algebra
"""

import argparse
import os
import sys
import time
import json
from typing import List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synth.generator import ChainOfThoughtGenerator


# Category descriptions for help text
CATEGORY_DESCRIPTIONS = {
    "math_arithmetic": "Basic arithmetic operations (+, -, ×, ÷)",
    "math_word_problem": "Word problems involving real-world scenarios",
    "math_algebra": "Algebraic equations and expressions",
    "logic_deduction": "Logical deduction and syllogisms",
    "logic_ordering": "Ordering and ranking problems",
    "code_debug": "Code debugging and bug fixing",
    "reasoning_cause_effect": "Cause and effect analysis",
    "reasoning_comparison": "Comparison and contrast analysis",
    "reasoning_planning": "Multi-step planning problems",
    "science": "Science problems (physics, chemistry, biology)",
}

ALL_CATEGORIES = list(CATEGORY_DESCRIPTIONS.keys())
ALL_DIFFICULTIES = ["easy", "medium", "hard"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic chain-of-thought reasoning dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Categories available:
""" + "\n".join(f"  {k}: {v}" for k, v in CATEGORY_DESCRIPTIONS.items())
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="synth/data/cot_dataset.jsonl",
        help="Output file path (default: synth/data/cot_dataset.jsonl)"
    )
    
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=300000,
        help="Number of examples to generate (default: 300000)"
    )
    
    parser.add_argument(
        "--categories", "-c",
        nargs="+",
        choices=ALL_CATEGORIES,
        default=None,
        help="Categories to include (default: all)"
    )
    
    parser.add_argument(
        "--difficulties", "-d",
        nargs="+",
        choices=ALL_DIFFICULTIES,
        default=None,
        help="Difficulty levels to include (default: all)"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=10000,
        help="Batch size for progress reporting (default: 10000)"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics after generation"
    )
    
    return parser.parse_args()


def show_statistics(output_path: str):
    """Show statistics about the generated dataset."""
    print("\n📊 Dataset Statistics:")
    print("=" * 50)
    
    category_counts = {}
    difficulty_counts = {}
    total = 0
    
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            category = data.get("category", "unknown")
            difficulty = data.get("difficulty", "unknown")
            
            category_counts[category] = category_counts.get(category, 0) + 1
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
            total += 1
    
    print(f"\nTotal examples: {total:,}")
    
    print("\nBy Category:")
    for cat, count in sorted(category_counts.items()):
        pct = 100 * count / total
        print(f"  {cat}: {count:,} ({pct:.1f}%)")
    
    print("\nBy Difficulty:")
    for diff, count in sorted(difficulty_counts.items()):
        pct = 100 * count / total
        print(f"  {diff}: {count:,} ({pct:.1f}%)")
    
    # File size
    file_size = os.path.getsize(output_path)
    if file_size > 1024 * 1024 * 1024:
        size_str = f"{file_size / (1024**3):.2f} GB"
    elif file_size > 1024 * 1024:
        size_str = f"{file_size / (1024**2):.2f} MB"
    else:
        size_str = f"{file_size / 1024:.2f} KB"
    
    print(f"\nFile size: {size_str}")
    print("=" * 50)


def main():
    args = parse_args()
    
    print("=" * 60)
    print("🧠 Chain-of-Thought Dataset Generator")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Output: {args.output}")
    print(f"  Count: {args.count:,}")
    print(f"  Categories: {args.categories or 'all'}")
    print(f"  Difficulties: {args.difficulties or 'all'}")
    print(f"  Seed: {args.seed}")
    print(f"  Batch size: {args.batch_size:,}")
    
    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize generator
    print("\n🔧 Initializing generator...")
    generator = ChainOfThoughtGenerator(seed=args.seed)
    
    # Generate dataset
    print("\n🚀 Starting generation...")
    start_time = time.time()
    
    output_path = generator.generate_dataset(
        total_examples=args.count,
        output_path=args.output,
        batch_size=args.batch_size,
        categories=args.categories,
        difficulties=args.difficulties,
        show_progress=True
    )
    
    elapsed = time.time() - start_time
    rate = args.count / elapsed
    
    print(f"\n⏱️  Generation completed in {elapsed:.1f} seconds")
    print(f"   Rate: {rate:.0f} examples/second")
    
    if args.stats:
        show_statistics(output_path)
    
    print("\n✅ Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

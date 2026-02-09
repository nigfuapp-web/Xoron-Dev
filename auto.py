import subprocess
import random
import argparse
import sys


def run_training(use_hf: bool = True):
    """
    Run training with randomly selected dual modalities.
    
    Args:
        use_hf: If True, load from HuggingFace (--hf). If False, build from scratch (--build).
    """
    # Define the flags and their associated max values
    # Maps modality flag to (max_flag_name, max_value)
    flag_configs = {
        "--text": ("--max-text", 3300),
        "--image": ("--max-image", 500),
        "--video": ("--max-video", 1000),
        "--voice": ("--max-voice", 200)
    }

    # Pick 2 unique flags randomly
    selected_flags = random.sample(list(flag_configs.keys()), 2)
    
    # Construct the command
    # Base command: python build.py --hf or python build.py --build
    if use_hf:
        cmd = ["python", "build.py", "--hf"]
        mode_str = "--hf (HuggingFace pretrained)"
    else:
        cmd = ["python", "build.py", "--build"]
        mode_str = "--build (from scratch)"
    
    # Add each modality flag with its per-modality max
    for flag in selected_flags:
        max_flag, max_val = flag_configs[flag]
        cmd.append(flag)
        cmd.extend([max_flag, str(max_val)])

    print(f"🚀 Launching training with {mode_str}")
    print(f"   Command: {' '.join(cmd)}")
    print(f"   Modalities: {', '.join(selected_flags)}")
    
    # Run the script
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Error: build.py not found in the current directory.")
        sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Auto-train Xoron-Dev with randomly selected dual modalities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python auto.py              # Default: Load from HuggingFace and train with 2 random modalities
  python auto.py --no-hf      # Build from scratch and train with 2 random modalities
  
The script randomly picks 2 modalities from: text, image, video, voice
Each modality gets its own max samples per epoch:
  - text: 3300 samples
  - image: 500 samples  
  - video: 1000 samples
  - voice: 200 samples
        """
    )
    
    parser.add_argument('--no-hf', action='store_true',
                       help='Build model from scratch instead of loading from HuggingFace')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # --no-hf means use --build, otherwise use --hf
    use_hf = not args.no_hf
    
    run_training(use_hf=use_hf)

import subprocess
import random

def run_training():
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
    # Base command: python build.py --hf
    cmd = ["python", "build.py", "--hf"]
    
    # Add each modality flag with its per-modality max
    for flag in selected_flags:
        max_flag, max_val = flag_configs[flag]
        cmd.append(flag)
        cmd.extend([max_flag, str(max_val)])

    print(f"🚀 Launching training with: {' '.join(cmd)}")
    
    # Run the script
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
    except FileNotFoundError:
        print("❌ Error: build.py not found in the current directory.")

if __name__ == "__main__":
    run_training()

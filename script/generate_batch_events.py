import subprocess
import argparse
import re
from pathlib import Path

# --- Dynamic Path Setup ---
# Absolute path of the script file itself
SCRIPT_FILE = Path(__file__).resolve()
# Directory where the script is located (e.g., /.../ESOT500syn/script)
SCRIPT_DIR = SCRIPT_FILE.parent
# Root directory of the project (e.g., /.../ESOT500syn)
PROJECT_ROOT = SCRIPT_DIR.parent
# Directory of the V2CE-Toolbox toolbox
V2CE_TOOLBOX_DIR = PROJECT_ROOT / "libs" / "V2CE-Toolbox"
# Full path to the v2ce.py script
V2CE_SCRIPT_PATH = V2CE_TOOLBOX_DIR / "v2ce.py"


def process_all_sequences(
    root_data_dir: Path, 
    fps: int, 
    batch_size: int, 
    stage2_batch_size: int
):
    """
    Traverse all 'seq_xxxx' folders under the specified root directory, 
    find 'rgb' folders, and use v2ce.py to generate event streams (.npz files).
    Video visualization is explicitly disabled.
    """
    if not V2CE_SCRIPT_PATH.exists():
        print(f"Error: V2CE-Toolbox script not found. Expected path: {V2CE_SCRIPT_PATH}")
        return

    seq_pattern = re.compile(r"^seq_\d{4}$")
    subfolders = [d for d in root_data_dir.iterdir() if d.is_dir() and seq_pattern.match(d.name)]
    
    if not subfolders:
        print(f"No 'seq_xxxx' format folders found in directory '{root_data_dir}'.")
        return

    print(f"Found a total of {len(subfolders)} sequence folders to process.")

    for seq_folder in sorted(subfolders):
        print("-" * 50)
        print(f"Starting processing: {seq_folder.name}")

        rgb_folders_found = list(seq_folder.rglob('rgb'))
        
        if not rgb_folders_found:
            print(f"Warning: No 'rgb' subfolder found in '{seq_folder}', skipping this sequence.")
            continue
        
        rgb_folder = rgb_folders_found[0]
        print(f"Found RGB image path: {rgb_folder}")

        # Dynamically get the number of RGB images as max_frame_num
        image_paths = list(rgb_folder.glob('*.png'))
        max_frame_num = len(image_paths)
        
        if max_frame_num == 0:
            print(f"Warning: No .png images found in '{rgb_folder}', skipping this sequence.")
            continue
        
        print(f"Detected {max_frame_num} frames of RGB images.")

        # Build the command, focusing only on event stream generation.
        # Video writing is hardcoded to False.
        command = [
            "python", str(V2CE_SCRIPT_PATH),
            "--image_folder", str(rgb_folder.resolve()),
            "--out_folder", str(seq_folder.resolve()),
            "--max_frame_num", str(max_frame_num),
            "--fps", str(fps),
            "--batch_size", str(batch_size),
            "--stage2_batch_size", str(stage2_batch_size),
            "--write_event_frame_video", "False",  # <-- Hardcoded to disable video generation
            "--infer_type", "pano",
            "--log_level", "info"
        ]
        
        print(f"Executing command: {' '.join(command)}")
        print(f"Working directory will be set to: {V2CE_TOOLBOX_DIR}")
        
        try:
            subprocess.run(command, check=True, cwd=V2CE_TOOLBOX_DIR)
            print(f"Successfully processed: {seq_folder.name}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while processing '{seq_folder.name}': {e}")
        except FileNotFoundError:
            print("Error: 'python' command not found. Please ensure Python is installed and configured in the system PATH.")
            break

    print("-" * 50)
    print("All sequences processed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automate processing of all seq_xxxx folders' RGB images into event streams (.npz files). This script does NOT generate video files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Core parameters for data generation
    parser.add_argument(
        "--root_data_dir", 
        type=str,
        default="/home/chujie/Data/ESOT500syn/test/output/ESOT500syn_dataset",
        help="Path to the root data directory containing all seq_xxxx folders."
    )
    parser.add_argument("--fps", type=int, default=30, help="CRITICAL: Frame rate of the input image sequence. Must match the physical capture rate.")
    parser.add_argument("--batch_size", "-b", type=int, default=8, help="Batch size for first stage inference (video->voxel).")
    parser.add_argument("--stage2_batch_size", type=int, default=24, help="Batch size for second stage inference (voxel->event).")
    
    args = parser.parse_args()
    
    root_path = Path(args.root_data_dir).resolve()
    
    if not root_path.is_dir():
        print(f"Error: The specified path '{root_path}' is not a valid directory.")
    else:
        process_all_sequences(
            root_data_dir=root_path,
            fps=args.fps,
            batch_size=args.batch_size,
            stage2_batch_size=args.stage2_batch_size
        )
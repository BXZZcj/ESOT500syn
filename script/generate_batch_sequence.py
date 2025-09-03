import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="ESOT500syn: Batch Runner")
    parser.add_argument("--configs_dir", type=str, default="/DATA/jiechu/datasets/ESOT500syn_dataset", help="Root directory containing the generated config folders.")
    parser.add_argument("--runner_script", type=str, default="/home/chujie/Data/ESOT500syn/script/generate_single_sequence.py", help="Path to the single sequence runner script.")
    args = parser.parse_args()

    configs_root = Path(args.configs_dir)
    if not configs_root.exists():
        print(f"Error: Configurations directory not found at '{configs_root.resolve()}'")
        return

    # find all config.yaml files
    config_files = sorted(list(configs_root.rglob("config.yaml")))
    
    if not config_files:
        print(f"No 'config.yaml' files found in '{configs_root.resolve()}'")
        return

    print(f"Found {len(config_files)} sequences to run.")

    # run each sequence
    for config_path in tqdm(config_files, desc="Processing sequences"):
        print(f"\n--- Running sequence with config: {config_path} ---")
        
        # use subprocess to call the existing script, and pass the config path
        # this ensures that each run is a clean, independent process
        command = [
            "python",
            args.runner_script,
            "--config",
            str(config_path)
        ]
        
        # run the command, and wait for it to complete
        # check=True will throw an exception if the script fails to run
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"!!!!!! ERROR running sequence for config: {config_path} !!!!!!")
            print(f"Return code: {e.returncode}")
        except FileNotFoundError:
            print(f"Error: Could not find runner script at '{args.runner_script}'.")
            break

    print("\n--- Batch processing complete! ---")

if __name__ == "__main__":
    main()
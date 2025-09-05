import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import os


def run_single_sequence(config_path: Path, runner_script: str) -> tuple:
    print(f"--- Starting sequence: {config_path.parent.name} ---")
    
    command = [
        "python",
        runner_script,
        "--config",
        str(config_path)
    ]
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return (True, config_path, f"Successfully processed {config_path.parent.name}")
    except subprocess.CalledProcessError as e:
        error_message = (
            f"!!!!!! ERROR running sequence for config: {config_path} !!!!!!\n"
            f"Return code: {e.returncode}\n"
            f"--- STDOUT ---\n{e.stdout}\n"
            f"--- STDERR ---\n{e.stderr}\n"
        )
        return (False, config_path, error_message)
    except FileNotFoundError:
        error_message = f"Error: Could not find runner script at '{runner_script}'."
        return (False, config_path, error_message)
    except Exception as e:
        return (False, config_path, f"An unexpected error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(description="ESOT500syn: Batch Runner")
    parser.add_argument("--configs_dir", type=str, default="/mnt/f1590153-780c-408d-b394-7b3b56082548/ESOT500/ESOT500syn_dataset", help="Root directory containing the generated config folders.")
    parser.add_argument("--runner_script", type=str, default="/mnt/f1590153-780c-408d-b394-7b3b56082548/ESOT500/ESOT500syn/script/generate_single_sequence.py", help="Path to the single sequence runner script.")
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=0,
        # default=os.cpu_count(), 
        help="Number of parallel processes to use. Defaults to the number of CPU cores."
    )
    args = parser.parse_args()

    configs_root = Path(args.configs_dir)
    if not configs_root.exists():
        print(f"Error: Configurations directory not found at '{configs_root.resolve()}'")
        return

    config_files = sorted(list(configs_root.rglob("config.yaml")))
    
    if not config_files:
        print(f"No 'config.yaml' files found in '{configs_root.resolve()}'")
        return

    print(f"Found {len(config_files)} sequences to run using up to {args.num_workers} parallel processes.")
    
    tasks = [(path, args.runner_script) for path in config_files]

    if args.num_workers > 0:
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            results = list(tqdm(pool.starmap(run_single_sequence, tasks), total=len(tasks), desc="Processing sequences"))
    else:
        print("Running in sequential mode.")
        results = []
        for task in tqdm(tasks, desc="Processing sequences"):
            results.append(run_single_sequence(*task))
            
    print("\n--- Batch processing complete! ---")
    success_count = 0
    failures = []
    for success, config_path, message in results:
        if success:
            success_count += 1
        else:
            failures.append((config_path, message))
    
    print(f"\nSummary: {success_count}/{len(config_files)} sequences processed successfully.")
    
    if failures:
        print("\n--- Failures ---")
        for config_path, error_message in failures:
            print(error_message)


if __name__ == "__main__":
    main()
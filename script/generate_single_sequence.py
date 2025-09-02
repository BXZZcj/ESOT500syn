import argparse
import yaml
import sys
from pathlib import Path

# Add src to Python path to allow for clean imports from our library
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Import the main runner function from our library
from esot500syn.runner import run

def main():
    parser = argparse.ArgumentParser(description="ESOT500syn: Data Generation Pipeline")
    parser.add_argument("--config", type=str, default="/home/chujie/Data/ESOT500syn/configs/debug_single_sequence_configs.yaml", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading or parsing config file '{args.config}': {e}")
        sys.exit(1)

    # Call the main logic from the library, passing the loaded config
    run(config)

if __name__ == "__main__":
    main()
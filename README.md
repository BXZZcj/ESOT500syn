# ESOT500syn

A tool to record event-based vision-oriented tracking (VOT) datasets using the [ManiSkill](https://maniskill.readthedocs.io/en/latest/index.html) simulation environment.

## Installation

1.  Clone the repository:

    ```bash
    git clone <your-repo-url>
    cd ESOT500syn
    ```

2.  Install the package in editable mode. This allows you to modify the source code and use the changes immediately without reinstalling.
    ```bash
    pip install -e .
    ```
    This command reads the `pyproject.toml` file, installs all required dependencies, and creates the `esot-recorder` command-line tool.

## Usage

The recording process is controlled by YAML configuration files located in the `configs/` directory.

1.  **Customize a Configuration:**
    Open `configs/box_on_table.yaml` and modify parameters like the environment (`env_id`), output directory, or motion parameters.

2.  **Run the Recorder:**
    Execute the recording tool by pointing it to your desired configuration file.

    ```bash
    esot-recorder --config-path configs/box_on_table.yaml
    ```

3.  **Check the Output:**
    The generated video (`.mp4`) will be saved in the `output_dir` specified in your configuration file (e.g., `data/push_cube_circular_motion/`).

## How it Works

- `main.py`: The main entry point, parses the config file path.
- `recorder.py`: Orchestrates the simulation, connecting the environment with the motion logic.
- `environment.py`: A wrapper around ManiSkill to simplify environment setup.
- `motion.py`: Defines how objects or agents should move during the simulation. You can add new motion patterns here.
- `configs/`: Contains all declarative configurations for different recording scenarios.

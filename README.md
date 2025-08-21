# ESOT500syn: Synthetic Multi-Modality Perception Dataset Generation Pipeline

**ESOT500syn** is a easy-to-use and highly extensible synthetic event stream dataset generation pipeline. It currently can generate dataset for the following tasks, and these tasks are also what this repo is initially developed for:
- Object Bounding Box Tracking (modal/amodal)
- Object Mask Tracking (modal/amodal)
- 6D Pose Estimation and Tracking
However, it can further generate dataset for more tasks (such as Depth Estimation, Visual-based Odemetry), depending on the user custom configuration. And the custom configuration process is simple. 
ESOT500 use [ManiSkill3](https://maniskill.readthedocs.io/en/latest/) framework to generate RGB dataset, and use [V2CE-Toolbox](https://github.com/ucsd-hdsi-dvs/V2CE-Toolbox) to convert the RGB dataset into event streams.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Key Features

-   **Fully Configuration-Driven**: Define entire datasets—from scenes to assets, lighting, object motion, camera motion and so on—in human-readable YAML files. No code changes needed for generating new scenarios.
-   **Procedural Generation Pipeline**: A two-stage workflow allows you to first generate thousands of deterministic sequence configurations in YAML format, then load the YAML configurations and run the simulation to generate RGB dataset, then convert the RGB dataset into event stream. The above is a separate, robust batch process.
-   **Extensive Motion Library**: Features a rich, expandable library of deterministic (the motion pattern is hard-coded) and stochastic (the motion pattern code is with `random`) motion patterns for both objects and the camera.
-   **Composable Motion System**: Combine simple motion patterns (e.g., "oscillate" + "spin") to create infinitely complex and challenging object motion trajectories.
-   **Rich Domain Randomization**: Automatically randomize scenes, target objects, distractor objects, lighting conditions, and camera perspectives to create highly diverse datasets.
-   **Intelligent Asset Spawning**: Ensures that all procedurally generated objects are spawned within the camera's field of view using a visibility-checking algorithm.
-   **Automatic Annotation Generation**: Automatically saves RGB frames, modal/amodal masks, 2D modal/amodal bounding boxes, and 6D object poses in camera coordinates for each sequence.
-   **Integrated Event Stream Conversion**: Seamlessly converts the generated RGB sequences into event streams using the integrated `V2CE-Toolbox` submodule.

## Project Structure

The project is organized into a modular structure to ensure clarity and extensibility:

```
ESOT500syn/
├── configs/                  # All user-facing configuration files
│   ├── meta_base_config.yaml        # Shared settings for all sequences
│   ├── meta_generation_config.yaml  # Rules for randomizing the dataset
│   └── single_sequence_configs.yaml # Just a demo configuration
│
├── libs/                     # Third-party code managed as git submodules
│   └── V2CE-Toolbox/         # Convert RGB into event streams
│
├── scripts/                  # User-facing executable scripts
│   ├── generate_batch_configs.py     # Stage 1: Generate all sequence configs
│   ├── generate_batch_sequence.py    # Stage 2: Run simulation for all generated configs
│   ├── generate_batch_events.py      # Stage 3: Convert all RGB sequences to events
│   ├── generate_single_sequence.py   # runs a single config file
│   ├── imgs_to_video.py              # Convert RGB sequence into video
│   └── vis_events.py                 # Visualize event stream
│
└── src/
    └── esot500syn/           # The core Python library for the project
        ├── runner.py           # Core simulation loop and setup logic
        ├── motion/             # Object and camera motion pattern definitions
        ├── processing/         # Annotation logic
        ├── simulation/         # ManiSkill environment wrappers and mixins
        └── demo/               # Some demo scripts
```

## Setup and Installation

**1. Prerequisites:**
-   A Conda or venv environment (Python 3.10+ is recommended).

**2. Clone the Repository:**
Clone this repository along with its git submodules (for V2CE-Toolbox):
```bash
git clone --recurse-submodules https://github.com/your-username/ESOT500syn.git
cd ESOT500syn
```

**3. Install Dependencies:**
This project uses `pyproject.toml` for dependency management. Install the project in editable mode:
```bash
pip install -e .
```

**4. Download Required Assets:**
-   **ManiSkill Assets:** Ensure you have downloaded the required scene datasets (e.g., `ReplicaCAD`, `ArchitecTHOR`) and the YCB object models as per the ManiSkill3 documentation.
    ```bash
    python -m mani_skill.utils.download_asset ReplicaCAD
    python -m mani_skill.utils.download_asset RoboCasa
    python -m mani_skill.utils.download_asset AI2THOR
    python -m mani_skill.utils.download_asset ycb
    ```
-   **V2CE Model Weights:** Download the pre-trained model from the [V2CE Google Drive link](https://drive.google.com/file/d/1-aC6CTGZgAZk3snANZ46FAGNkPzu_Scw/view?usp=sharing) and place it in the following directory:
    ```
    libs/V2CE-Toolbox/weights/v2ce_3d.pt
    ```

**5. Note on Headless Rendering:**
Although ESOT500syn leverage ManiSkill3 simulation framework, it is developed on a headless device, so all the codes are serving for headless runs. Moreover, to as much as possible reduce the visual sim2real gap, ESOT500 by default open the ray-tracing technique (from ManiSkill3) to produce RGB datasets, but the ray-tracing technique seems to be not supported by GPU with ManiSkill3 (at least produces bugs when I'm developing ESOT500syn), to the default device to produce RGB is CPU. Users can also change the device and rendering technique in `meta_base_configs.yaml`.

## Usage: The 3-Stage Generation Workflow

This project uses a three-stage process to generate the final event stream dataset.

### Stage 1: Generate Sequence Configurations

This stage reads your randomization rules (defined by users in `meta_gen_configs.yaml` & `meta_base_configs.yaml`) and generates a unique, deterministic configuration file for every sequence in your dataset.

**➡️ Action:** Run the `generate_configs.py` script.
```bash
python path/to/scripts/generate_batch_configs.py \
    --base_config path/to/meta_base_configs.yaml \
    --gen_config path/to/meta_gen_configs.yaml \
    --output_dir path/to/output_dir
```
**⬅️ Result:** The `path/to/output_dir/` directory will be populated with `seq_0000/`, `seq_0001/`, etc., each containing a `config.yaml` file.

### Stage 2: Run Batch Simulation

This stage iterates through the previously generated configurations and runs the ManiSkill simulation for each one, saving the RGB frames and annotations.

**➡️ Action:** Run the `generate_batch_sequence.py` script, pointing it to the directory from Stage 1.
```bash
python path/to/scripts/generate_batch_sequence.py \
    --configs_dir path/to/output_dir \
    --runner_script path/to/generate_single_sequence.py
```
**⬅️ Result:** The output directory specified in your `meta_base_config.yaml` (e.g., `path/to/output_dir/seq_{index}/{scene}/{target_object}`) will be populated with the full data for each sequence (`rgb/`, `modal_mask/`, `amodal_mask/`, `annotations.json`).

### Stage 3: Convert RGB to Events

This final stage converts all the generated RGB sequences into event streams.

**➡️ Action:** Run the `generate_batch_events.py` script, pointing it to the dataset directory from Stage 2.
```bash
# There are many other parameters optional, please refer to the source code of scripts/convert_to_events.py, or directly chech the V2CE-Toolbox.
python scripts/convert_to_events.py \
    --dataset_dir path/to/output_dir \
    --fps 30 # The fps of your input RGB sequence
```
**⬅️ Result:** Each `seq_{index}/` directory will now also contain an `.npz` file with the event stream data.

## Customizing Your Dataset

The entire generation process is controlled by `configs/meta_base_config.yaml` and `configs/meta_gen_configs.yaml`.

-   **`meta_base_config.yaml`**: Defines static parameters shared across all generated sequences, such as image resolution, simulation quality, and batch settings (`seed`, `num_sequences`). It also provides fallback values, like a default lighting setup if randomization is disabled.

-   **`meta_gen_configs.yaml`**: This is the creative heart of your dataset. It defines the **rules and sampling space** for randomization. To change the characteristics of your dataset, simply edit this file:
    -   **Scenes**: Adjust `scene_id_ranges` to control which environments are used.
    -   **Assets & Motion**: Modify the `motion_pool` lists to change the dynamics. For example, to make objects move slower, reduce the `speed` ranges. To have more static distractors, duplicate the `{ type: "static" }` entry in the `distractor_motion_pool`.
    -   **Camera**: Add specific camera poses for certain scenes in `poses_by_scene` for artistic control, or adjust the `motion_pool` to favor more dynamic or static camera work.
    -   **Lighting**: Uncomment the `lighting` block in `continuous_sampling` to enable fully randomized lighting for every sequence.

## Extending the Motion Library

Adding new custom motions is designed to be simple and requires no changes to the core runner logic.

1.  **Open the relevant file**: `src/esot500syn/motion/object.py` or `src/esot500syn/motion/camera.py`.
2.  **Create a new Python function** that accepts `(mixin, start_pose, config)` or `(step, env, sensor, initial_pose, cfg)`, and returns a `sapien.Pose`.
3.  **Decorate it** with `@register_motion_pattern("your_new_motion_name")`.

**Example:**
```python
@register_motion_pattern("new_spiral_motion")
def motion_spiral(mixin, start_pose, config):
    # ... your logic to calculate new_p and new_q ...
    return sapien.Pose(p=new_p, q=new_q)
```
4.  You can now immediately use `your_new_motion_name` in the `motion_pool` of your `meta_gen_configs.yaml`!

## Debugging a Single Sequence

When a batch generation fails or produces unexpected results, you can easily debug the problematic sequence in isolation.

Every generated sequence folder (e.g., `path/to/output/seq_0123/`) contains a `config.yaml`. This file is a complete, deterministic snapshot of the run.

Use the `generate_single_sequence.py` script to run it:
```bash
python scripts/generate_single_sequence.py --config path/to/output/seq_0123/config.yaml
```

## License

This project is licensed under the MIT License following the V2CE. See the `LICENSE` file for details.


## Acknowledgments
- Thanks for the great works including [ManiSkill3](https://maniskill.readthedocs.io/en/latest/) and [V2CE-Toolbox](https://github.com/ucsd-hdsi-dvs/V2CE-Toolbox).
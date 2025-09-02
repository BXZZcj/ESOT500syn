import argparse
import yaml
import sys
import random
import copy
import os
from pathlib import Path
import tqdm
import numpy as np
import sapien
import transforms3d.quaternions as t3d_quat

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from mani_skill.utils import sapien_utils
from esot500syn.processing.annotations import (
    load_mesh_vertices_faces,
    rasterize_amodal_mask,
    bbox_from_mask
)

def discover_ycb_assets(discovery_paths):
    asset_pool = []
    for path_str in discovery_paths:
        base_path = Path(os.path.expanduser(path_str))
        if not base_path.exists():
            continue
        for asset_dir in base_path.iterdir():
            if (
                asset_dir.is_dir() and
                (asset_dir / "textured.obj").exists() and
                (asset_dir / "collision.ply").exists()
            ):
                asset_pool.append({
                    "name": asset_dir.name,
                    "filepaths": {
                        "visual": str(asset_dir / "textured.obj"),
                        "collision": str(asset_dir / "collision.ply")
                    }
                })
    print(f"Discovered {len(asset_pool)} valid YCB assets.")
    return asset_pool

def sample_from_range(value_range):
    if isinstance(value_range[0], list):
        return [
            random.uniform(min_v, max_v)
            for min_v, max_v in zip(*value_range)
        ]
    else:
        return random.uniform(value_range[0], value_range[1])

def sample_from_box(box):
    return [
        random.uniform(box['min'][i], box['max'][i])
        for i in range(len(box['min']))
    ]

def randomize_motion_params(params_config):
    if isinstance(params_config, list):
        return [randomize_motion_params(p) for p in params_config]

    if not isinstance(params_config, dict):
        return params_config

    randomized_params = {}
    for key, value in params_config.items():
        if (
            isinstance(value, list) and
            len(value) == 2 and
            isinstance(value[0], (int, float, list))
        ):
            randomized_params[key] = sample_from_range(value)
        elif isinstance(value, dict):
            randomized_params[key] = randomize_motion_params(value)
        else:
            randomized_params[key] = value
    return randomized_params

def sample_point_in_camera_view(
    camera_world_pose: sapien.Pose,
    local_spawn_rules: dict
):
    x, y, z = (
        random.uniform(*local_spawn_rules['x_forward']),
        random.uniform(*local_spawn_rules['y_left']),
        random.uniform(*local_spawn_rules['z_up'])
    )
    point_in_camera_frame = sapien.Pose(p=[x, y, z])
    world_pose = camera_world_pose * point_in_camera_frame
    return world_pose.p.cpu().numpy().flatten().tolist()

def sample_visible_asset_pose(
    camera_world_pose: sapien.Pose,
    cam_intrinsics: dict,
    asset_mesh: tuple,
    local_spawn_rules: dict,
    max_retries=20
):
    """
    Repeatedly sample until finding a pose that ensures the asset is visible in the camera view.
    """
    vertices, faces = asset_mesh
    if vertices is None:  # if the mesh loading fails
        p = sample_point_in_camera_view(camera_world_pose, local_spawn_rules)
        q = t3d_quat.normalize([random.uniform(-1, 1) for _ in range(4)])
        return p, q.tolist()

    for _ in range(max_retries):
        # step A: sample a candidate pose
        candidate_p = sample_point_in_camera_view(camera_world_pose, local_spawn_rules)
        axis = np.random.randn(3)
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        angle = random.uniform(0, 2 * np.pi)
        candidate_q = t3d_quat.axangle2quat(axis, angle)

        candidate_world_pose = sapien.Pose(p=candidate_p, q=candidate_q)

        # step B: calculate its pose relative to the camera
        pose_in_cam = camera_world_pose.inv() * candidate_world_pose
        t_cam = pose_in_cam.p
        q_cam_wxyz = pose_in_cam.q

        # step C: render the virtual mask
        mask = rasterize_amodal_mask(
            vertices, faces,
            t_cam.cpu().numpy()[0], q_cam_wxyz.cpu().numpy()[0],  # This is single environment.
            cam_intrinsics['fx'], cam_intrinsics['fy'],
            cam_intrinsics['cx'], cam_intrinsics['cy'],
            cam_intrinsics['width'], cam_intrinsics['height']
        )

        # step D: check visibility
        if np.any(mask):  # as long as there is a non-zero pixel in the mask, it is considered visible
            # print("INFO: Found visible pose.")
            return candidate_p, candidate_q.tolist()

    print(f"WARNING: Could not find a visible pose after {max_retries} retries. Using last attempt.")
    return candidate_p, candidate_q.tolist()

# --- core: random configuration generator (refactored version) ---
def generate_randomized_config(
    base_config,
    gen_config,
    asset_pool,
    camera_intrinsics
):
    new_config = copy.deepcopy(base_config)
    if 'env' not in new_config:
        new_config['env'] = {}

    # 1. scene
    scene_rules = gen_config['scene_sampling']['scene_id_ranges']
    scene_type = random.choice(list(scene_rules.keys()))
    new_config['env']['env_type'] = scene_type
    scene_id, layout_id, style_id = None, None, None
    if scene_type == "RoboCasa":
        layout_id, style_id = (
            random.randint(*scene_rules[scene_type]['layout_ids']),
            random.randint(*scene_rules[scene_type]['style_ids'])
        )
        new_config['scene'] = {
            "robocasa_params": {
                'layout_ids': layout_id,
                'style_ids': style_id
            }
        }
    else:
        scene_id = random.randint(*scene_rules[scene_type])
        new_config['scene'] = {
            f"{scene_type.lower()}_params": {
                'build_config_idx': scene_id
            }
        }

    # 2. camera
    cam_setup = gen_config['camera_setup']
    specific_poses = cam_setup.get('poses_by_scene', {})
    pose_cfg, specific_pose_found = None, False
    if scene_type in specific_poses:
        scene_pose_configs = specific_poses[scene_type]
        if scene_type == "RoboCasa":
            scene_key = f"layout{layout_id}_style{style_id}"
            if scene_key in scene_pose_configs:
                pose_cfg, specific_pose_found = scene_pose_configs[scene_key], True
        else:
            if scene_id in scene_pose_configs:
                pose_cfg, specific_pose_found = scene_pose_configs[scene_id], True
    if specific_pose_found:
        print(f"INFO: Using specific camera pose for {scene_type} scene key: '{scene_key if scene_type == 'RoboCasa' else scene_id}'.")
        cam_pose_p, cam_pose_g, cam_pose_q = (
            pose_cfg.get('pose_p'),
            pose_cfg.get('pose_g'),
            pose_cfg.get('initial_pose_q')
        )
    else:
        cam_pose_p, cam_pose_g, cam_pose_q = (
            sample_from_box(cam_setup['random_sampling']['position_box']),
            sample_from_box(cam_setup['random_sampling']['target_box']),
            None
        )
    new_config['camera']['pose_p'], new_config['camera']['pose_g'] = cam_pose_p, cam_pose_g
    new_config['camera']['motion'] = randomize_motion_params(random.choice(cam_setup['motion_pool']))
    if cam_pose_q:
        cam_world_pose = sapien.Pose(p=cam_pose_p, q=cam_pose_q)
    else:
        cam_world_pose = sapien_utils.look_at(cam_pose_p, cam_pose_g)

    # 3. asset
    asset_rules = gen_config['asset_sampling']
    target_asset_template = random.choice(asset_pool)
    chosen_target_motion = random.choice(asset_rules['target_motion_pool'])
    target_motion_type, target_motion_params = (
        chosen_target_motion['type'],
        randomize_motion_params(chosen_target_motion.get('params', {}))
    )

    p, q = sample_visible_asset_pose(
        cam_world_pose,
        camera_intrinsics,
        target_asset_template['mesh'],
        asset_rules['local_spawn_box']['target']
    )
    new_config['custom_asset'] = {
        'enable': True,
        'name': target_asset_template['name'],
        'visual_filepath': target_asset_template['filepaths']['visual'],
        'collision_filepath': target_asset_template['filepaths']['collision'],
        'initial_pose_p': p,
        'initial_pose_q': q,
        'motion_mode': target_motion_type,
        'motion_params': target_motion_params
    }

    # distractor
    num_distractors = random.randint(*asset_rules['num_distractors_range'])
    available_distractors = [
        a for a in asset_pool if a['name'] != target_asset_template['name']
    ]
    distractor_templates = random.sample(
        available_distractors,
        min(num_distractors, len(available_distractors))
    )

    new_config['distractor_assets'] = []
    for i, distractor in enumerate(distractor_templates):
        chosen_distractor_motion = random.choice(asset_rules['distractor_motion_pool'])
        distractor_motion_type, distractor_motion_params = (
            chosen_distractor_motion['type'],
            randomize_motion_params(chosen_distractor_motion.get('params', {}))
        )

        p_d, q_d = sample_visible_asset_pose(
            cam_world_pose,
            camera_intrinsics,
            distractor['mesh'],
            asset_rules['local_spawn_box']['distractor']
        )
        new_config['distractor_assets'].append({
            'enable': True,
            'name': f"{distractor['name']}_{i}",
            'visual_filepath': distractor['filepaths']['visual'],
            'collision_filepath': distractor['filepaths']['collision'],
            'initial_pose_p': p_d,
            'initial_pose_q': q_d,
            'motion_mode': distractor_motion_type,
            'motion_params': distractor_motion_params
        })

    # 4. lighting
    if gen_config.get('continuous_sampling') and 'lighting' in gen_config['continuous_sampling']:
        light_rules = gen_config['continuous_sampling']['lighting']
        lighting = {
            'ambient': sample_from_range(light_rules['ambient_range'])
        }
        num_point = random.randint(*light_rules['point_lights']['num_lights_range'])
        lighting['point_lights'] = [
            {
                'position': sample_from_box(light_rules['point_lights']['position_box']),
                'color': sample_from_box(light_rules['point_lights']['color_range'])
            }
            for _ in range(num_point)
        ]
        lighting['directional_light'] = {
            'direction': sample_from_box(light_rules['directional_light']['direction_box']),
            'color': sample_from_box(light_rules['directional_light']['color_range']),
            'shadow': random.random() < light_rules['directional_light']['shadow_probability']
        }
        new_config['lighting'] = lighting
    else:
        new_config['lighting'] = base_config.get('lighting', {})

    
    # 5. Add an extra point light for ArchitecTHOR scenes at the camera's position
    if scene_type == "ArchitecTHOR":
        # Create the new light. The position is the camera position determined earlier.
        new_light = {
            'position': cam_pose_p,
            'color': [1, 1, 1]
        }
        
        # Safely add the new light to the point_lights list, ensuring it exists first.
        if 'point_lights' not in new_config['lighting'] or new_config['lighting']['point_lights'] is None:
            new_config['lighting']['point_lights'] = []
        
        new_config['lighting']['point_lights'].append(new_light)
        print(f"INFO: Added a point light at camera position for ArchitecTHOR scene {scene_id}.")

    return new_config

def main():
    parser = argparse.ArgumentParser(description="ESOT500syn: Batch Config Generation")
    parser.add_argument(
        "--base_config",
        type=str,
        default="/home/chujie/Data/ESOT500syn/configs/meta_base_configs.yaml",
        help="Path to the base YAML config."
    )
    parser.add_argument(
        "--gen_config",
        type=str,
        default="/home/chujie/Data/ESOT500syn/configs/meta_gen_configs.yaml",
        help="Path to the generation space YAML config."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/DATA/jiechu/datasets/ESOT500syn",
        help="Root directory to save the generated config folders."
    )
    args = parser.parse_args()

    with open(args.base_config, 'r') as f:
        base_config = yaml.safe_load(f)
    with open(args.gen_config, 'r') as f:
        gen_config = yaml.safe_load(f)

    batch_cfg = base_config.get('batch_settings', {})
    num_sequences, start_index, master_seed = (
        batch_cfg.get('num_sequences', 1),
        batch_cfg.get('start_index', 0),
        batch_cfg.get('master_seed', 42)
    )

    # pre-calculate camera intrinsics
    # we assume that all generated sequences use the same camera intrinsics, which is reasonable for the dataset
    width, height = base_config['camera']['width'], base_config['camera']['height']
    fov = np.pi / 2  # Hardcoded from environment.py, can be moved to config if needed
    fy = height / 2 / np.tan(fov / 2)
    fx = fy
    cx, cy = width / 2, height / 2
    camera_intrinsics = {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'width': width,
        'height': height
    }

    # pre-load all asset meshes for efficiency
    asset_pool = discover_ycb_assets(gen_config['asset_sampling']['discovery_paths'])
    if not asset_pool:
        return
    print("Pre-loading asset meshes...")
    for asset in tqdm.tqdm(asset_pool, desc="Loading Meshes"):
        vertices, faces = load_mesh_vertices_faces(asset['filepaths']['visual'])
        asset['mesh'] = (vertices, faces)

    root_output_dir = Path(args.output_dir)
    root_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating {num_sequences} configurations in '{root_output_dir.resolve()}'...")

    for i in range(start_index, start_index + num_sequences):
        sequence_seed = master_seed + i
        random.seed(sequence_seed)
        np.random.seed(sequence_seed)

        # pass the camera intrinsics and asset pool to the generator
        sequence_config = generate_randomized_config(
            base_config,
            gen_config,
            asset_pool,
            camera_intrinsics
        )

        sequence_config['simulation']['seed'] = sequence_seed
        seq_output_dir = root_output_dir / f"seq_{i:04d}"
        seq_output_dir.mkdir(exist_ok=True)
        sequence_config['output']['output_dir'] = str(seq_output_dir)
        sequence_config.pop('batch_settings', None)
        config_path = seq_output_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(sequence_config, f, sort_keys=False, default_flow_style=False)

    print(f"\nSuccessfully generated {num_sequences} configuration files.")

if __name__ == "__main__":
    main()
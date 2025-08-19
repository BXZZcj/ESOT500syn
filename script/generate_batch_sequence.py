# scripts/generate_data.py (最终版)

import argparse, yaml, sys, random, copy, os
from pathlib import Path
import numpy as np
import sapien
import transforms3d.quaternions as t3d_quat

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from esot500syn.runner import run
from mani_skill.utils import sapien_utils

# --- (所有辅助函数保持不变) ---
def discover_ycb_assets(discovery_paths):
    asset_pool = []
    for path_str in discovery_paths:
        base_path = Path(os.path.expanduser(path_str))
        if not base_path.exists(): continue
        for asset_dir in base_path.iterdir():
            if asset_dir.is_dir() and (asset_dir / "textured.obj").exists() and (asset_dir / "collision.ply").exists():
                asset_pool.append({"name": asset_dir.name, "filepaths": {"visual": str(asset_dir / "textured.obj"), "collision": str(asset_dir / "collision.ply")}})
    print(f"Discovered {len(asset_pool)} valid YCB assets.")
    return asset_pool
def sample_from_range(value_range):
    if isinstance(value_range[0], list): return [random.uniform(min_v, max_v) for min_v, max_v in zip(*value_range)]
    else: return random.uniform(value_range[0], value_range[1])
def sample_from_box(box):
    return [random.uniform(box['min'][i], box['max'][i]) for i in range(len(box['min']))]
def sample_point_in_camera_view(camera_world_pose: sapien.Pose, local_spawn_box: dict):
    local_p = sample_from_box(local_spawn_box)
    point_in_camera_frame = sapien.Pose(p=local_p)
    world_pose = camera_world_pose * point_in_camera_frame
    return world_pose.p
def randomize_motion_params(params_config):
    if not isinstance(params_config, dict): return params_config
    randomized_params = {}
    for key, value in params_config.items():
        if isinstance(value, list) and len(value) == 2 and isinstance(value[0], (int, float, list)):
            randomized_params[key] = sample_from_range(value)
        elif isinstance(value, dict):
            randomized_params[key] = randomize_motion_params(value)
        else:
            randomized_params[key] = value
    return randomized_params

# --- 核心：随机配置生成器 (重构版) ---
def generate_randomized_config(base_config, gen_config, asset_pool):
    new_config = copy.deepcopy(base_config)
    
    # 1. 场景
    scene_rules = gen_config['scene_sampling']['scene_id_ranges']
    scene_type = random.choice(list(scene_rules.keys()))
    new_config['env']['env_type'] = scene_type
    
    # 初始化场景标识符
    scene_id, layout_id, style_id = None, None, None
    
    if scene_type == "RoboCasa":
        layout_id = random.randint(*scene_rules[scene_type]['layout_ids'])
        style_id = random.randint(*scene_rules[scene_type]['style_ids'])
        new_config['scene'] = {"robocasa_params": {'layout_ids': layout_id, 'style_ids': style_id}}
    else:
        scene_id = random.randint(*scene_rules[scene_type])
        new_config['scene'] = {f"{scene_type.lower()}_params": {'build_config_idx': scene_id}}

    # 2. 相机
    cam_setup = gen_config['camera_setup']
    specific_poses = cam_setup.get('poses_by_scene', {})
    
    # <<<--- 修正：添加处理 RoboCasa 的特殊逻辑 ---
    pose_cfg, specific_pose_found = None, False
    if scene_type in specific_poses:
        scene_pose_configs = specific_poses[scene_type]
        if scene_type == "RoboCasa":
            # 构建RoboCasa的特殊键
            scene_key = f"layout{layout_id}_style{style_id}"
            if scene_key in scene_pose_configs:
                pose_cfg = scene_pose_configs[scene_key]
                specific_pose_found = True
        else: # ArchitecTHOR or ReplicaCAD
            if scene_id in scene_pose_configs:
                pose_cfg = scene_pose_configs[scene_id]
                specific_pose_found = True

    if specific_pose_found:
        print(f"INFO: Using specific camera pose for {scene_type} scene key: '{scene_key if scene_type == 'RoboCasa' else scene_id}'.")
        cam_pose_p = pose_cfg.get('pose_p')
        cam_pose_g = pose_cfg.get('pose_g')
        cam_pose_q = pose_cfg.get('initial_pose_q')
    else: # 回退到通用随机采样
        cam_pose_p = sample_from_box(cam_setup['random_sampling']['position_box'])
        cam_pose_g = sample_from_box(cam_setup['random_sampling']['target_box'])
        cam_pose_q = None

    new_config['camera']['pose_p'], new_config['camera']['pose_g'] = cam_pose_p, cam_pose_g
    new_config['camera']['motion'] = randomize_motion_params(random.choice(cam_setup['motion_pool']))

    # 3. 计算相机世界位姿
    if cam_pose_q: cam_world_pose = sapien.Pose(p=cam_pose_p, q=cam_pose_q)
    else: cam_world_pose = sapien_utils.look_at(cam_pose_p, cam_pose_g)

    # ... (资产和光照部分的代码完全不变)
    asset_rules = gen_config['asset_sampling']
    target_asset_template = random.choice(asset_pool)
    chosen_target_motion = random.choice(asset_rules['target_motion_pool'])
    target_motion_type, target_motion_params = chosen_target_motion['type'], randomize_motion_params(chosen_target_motion.get('params', {}))
    axis = np.random.randn(3); axis = (axis / (np.linalg.norm(axis) + 1e-9))
    angle = random.uniform(0, 2*np.pi)
    init_q_wxyz = t3d_quat.axangle2quat(axis, angle)
    new_config['custom_asset'] = {'enable': True, 'name': target_asset_template['name'], 'visual_filepath': target_asset_template['filepaths']['visual'], 'collision_filepath': target_asset_template['filepaths']['collision'], 'initial_pose_p': sample_point_in_camera_view(cam_world_pose, asset_rules['local_spawn_box']['target']), 'initial_pose_q': init_q_wxyz.tolist(), 'motion_mode': target_motion_type, 'motion_params': target_motion_params}
    num_distractors = random.randint(*asset_rules['num_distractors_range'])
    available_distractors = [a for a in asset_pool if a['name'] != target_asset_template['name']]
    distractor_templates = random.sample(available_distractors, min(num_distractors, len(available_distractors)))
    new_config['distractor_assets'] = []
    for i, distractor in enumerate(distractor_templates):
        chosen_distractor_motion = random.choice(asset_rules['distractor_motion_pool'])
        distractor_motion_type, distractor_motion_params = chosen_distractor_motion['type'], randomize_motion_params(chosen_distractor_motion.get('params', {}))
        new_config['distractor_assets'].append({'enable': True, 'name': f"{distractor['name']}_{i}", 'visual_filepath': distractor['filepaths']['visual'], 'collision_filepath': distractor['filepaths']['collision'], 'initial_pose_p': sample_point_in_camera_view(cam_world_pose, asset_rules['local_spawn_box']['distractor']), 'motion_mode': distractor_motion_type, 'motion_params': distractor_motion_params})
    light_rules = gen_config['continuous_sampling']['lighting']
    lighting = {'ambient': sample_from_range(light_rules['ambient_range'])}
    num_point = random.randint(*light_rules['point_lights']['num_lights_range'])
    lighting['point_lights'] = [{'position': sample_from_box(light_rules['point_lights']['position_box']), 'color': sample_from_box(light_rules['point_lights']['color_range'])} for _ in range(num_point)]
    lighting['directional_light'] = {'direction': sample_from_box(light_rules['directional_light']['direction_box']),'color': sample_from_box(light_rules['directional_light']['color_range']),'shadow': random.random() < light_rules['directional_light']['shadow_probability']}
    new_config['lighting'] = lighting

    return new_config


# --- 主函数 main() (最终版) ---
def main():
    parser = argparse.ArgumentParser(description="ESOT500syn: Batch Data Generation Pipeline")
    parser.add_argument("--base_config", type=str, default="/home/chujie/Data/ESOT500syn/configs/batch_sequence_base_configs.yaml", help="Path to the base YAML config.")
    parser.add_argument("--gen_config", type=str, default="/home/chujie/Data/ESOT500syn/configs/batch_sequence_gen_configs.yaml", help="Path to the generation space YAML config.")
    args = parser.parse_args()

    with open(args.base_config, 'r') as f: base_config = yaml.safe_load(f)
    with open(args.gen_config, 'r') as f: gen_config = yaml.safe_load(f)

    # <<<--- 修正 #2：实现“配置优先，命令行覆盖”的逻辑 ---
    batch_cfg = base_config.get('batch_settings', {})
    
    num_sequences = batch_cfg.get('num_sequences')
    start_index = batch_cfg.get('start_index', 0)
    master_seed = batch_cfg.get('master_seed')

    if num_sequences is None or master_seed is None:
        raise ValueError("'num_sequences' and 'master_seed' must be defined in the base_config.yaml under 'batch_settings' or provided via command line.")
    
    # <<<--- (其余部分逻辑与之前版本完全相同) ---
    asset_pool = discover_ycb_assets(gen_config['asset_sampling']['discovery_paths'])
    if not asset_pool: return

    for i in range(start_index, start_index + num_sequences):
        print(f"\n{'='*80}\nGenerating Sequence: {i+1} / ({num_sequences} total, Master Seed: {master_seed})\n{'='*80}")
        sequence_seed = master_seed + i
        random.seed(sequence_seed)
        np.random.seed(sequence_seed)
        
        sequence_config = generate_randomized_config(base_config, gen_config, asset_pool)
        sequence_config['simulation']['seed'] = sequence_seed
        
        seq_output_dir = Path(base_config['output']['output_dir']) / f"seq_{i:04d}"
        sequence_config['output']['output_dir'] = str(seq_output_dir)
        
        seq_output_dir.mkdir(parents=True, exist_ok=True)
        with open(seq_output_dir / 'generated_config.yaml', 'w') as f:
            yaml.dump(sequence_config, f, sort_keys=False, default_flow_style=False)
        try:
            run(sequence_config)
            print(f"--- Sequence {i} completed successfully! ---")
        except Exception as e:
            print(f"!!!!!! ERROR during generation of sequence {i} !!!!!!")
            print(f"Config saved to: {seq_output_dir / 'generated_config.yaml'}")
            print(f"Error details: {e}")
            import traceback; traceback.print_exc()
            continue

if __name__ == "__main__":
    main()
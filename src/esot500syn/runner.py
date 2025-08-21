import gymnasium as gym
import numpy as np
import json
from pathlib import Path
from PIL import Image

from .simulation.environment import create_and_register_env
from .motion.camera import _apply_final_pose
from .processing.annotations import load_mesh_vertices_faces, rasterize_amodal_mask, bbox_from_mask


def run(config: dict):
    np.set_printoptions(suppress=True, precision=3)
    env_cfg, scene_cfg, sim_cfg, out_cfg, cam_cfg, asset_cfg, light_cfg, distractor_cfg = (
        config['env'], config['scene'], config['simulation'], config['output'],
        config['camera'], config.get('custom_asset', {}), config.get('lighting', {}),
        config.get('distractor_assets', [])
    )

    env_id = create_and_register_env(env_cfg['env_type'], cam_cfg['pose_p'], cam_cfg['pose_g'])
    env_kwargs = {
        "robot_uids": "none",
        "sensor_configs": {
            "shader_pack": cam_cfg.get("shader_pack", "default"),
            "base_camera": {"width": cam_cfg['width'], "height": cam_cfg['height']}
        },
        "custom_asset_config": asset_cfg,
        "distractor_assets_config": distractor_cfg,
        "lighting_config": light_cfg,
        "sim_config": {
            "control_freq": sim_cfg.get('control_freq', 30),
            "sim_freq": sim_cfg.get('sim_freq', 300)
        }
    }
    if env_cfg['env_type'] == "RoboCasa":
        env_kwargs["robocasa_params"] = scene_cfg.get('robocasa_params', {})

    output_dir = Path(out_cfg['output_dir']) / f"{env_cfg['env_type']}" / f"{asset_cfg.get('name', 'no_asset')}"
    rgb_dir = output_dir / "rgb"
    modal_mask_dir = output_dir / "modal_mask"
    amodal_mask_dir = output_dir / "amodal_mask"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    if out_cfg.get("save_annotations"):
        modal_mask_dir.mkdir(parents=True, exist_ok=True)
        amodal_mask_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        env_id,
        render_mode=out_cfg['render_mode'],
        obs_mode=env_cfg['obs_mode'],
        sim_backend=sim_cfg['sim_backend'],
        **env_kwargs
    )

    print("-" * 50 + f"\nStarting Runner for: {env_cfg['env_type']}")
    scene_idx = 0
    if env_cfg['env_type'] in ["ArchitecTHOR", "ReplicaCAD"]:
        num_scenes = len(env.unwrapped.scene_builder.build_configs)
        scene_idx = scene_cfg[f"{env_cfg['env_type'].lower()}_params"]['build_config_idx']
        print(f"Available scenes: {num_scenes}. Loading index: {scene_idx}")
        if not (0 <= scene_idx < num_scenes):
            scene_idx = 0
            print(f"Warning: Scene index out of bounds. Defaulting to {scene_idx}.")
    print(f"Saving outputs to: {output_dir.resolve()}\n" + "-" * 50)

    obs, _ = env.reset(seed=sim_cfg['seed'], options=dict(reconfigure=True, build_config_idxs=[scene_idx]))

    cam_sensor = env.unwrapped._sensors['base_camera']
    
    # use .sp to get the native sapien.Pose object
    initial_cam_pose = cam_sensor.camera.get_local_pose().sp
    
    cam_motion_cfg = cam_cfg.get("motion", {"type": "static"})

    asset_actor = env.unwrapped.custom_actor if asset_cfg.get("enable") and hasattr(env.unwrapped, "custom_actor") else None
    asset_id = asset_actor.per_scene_id.item() if asset_actor else None

    all_frames = []
    intrinsic_matrix = cam_sensor.camera.get_intrinsic_matrix().cpu().numpy()[0]
    fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    amodal_verts, amodal_faces = (
        load_mesh_vertices_faces(asset_cfg["visual_filepath"])
        if out_cfg.get("save_annotations") and asset_cfg.get("enable") else (None, None)
    )

    try:
        for step in range(sim_cfg['num_steps']):
            _apply_final_pose(step, env, cam_sensor, initial_cam_pose, cam_motion_cfg)

            action = env.action_space.sample() if env.action_space else None
            obs, *_ = env.step(action)

            sensor_data = obs.get("sensor_data", {}).get("base_camera", {})
            if "Color" not in sensor_data:
                continue

            rgb_tensor = sensor_data["Color"].squeeze(0).cpu().numpy()
            img_to_save = (rgb_tensor[..., :3] * 255).astype(np.uint8) if rgb_tensor.dtype == np.float32 else rgb_tensor
            Image.fromarray(img_to_save).save(rgb_dir / f"rgb_{step:04d}.png")

            if out_cfg.get("save_annotations") and asset_id is not None and "Segmentation" in sensor_data:
                modal_mask = (sensor_data["Segmentation"][..., 1] == asset_id).squeeze().cpu().numpy()
                if bbox_from_mask(modal_mask):
                    Image.fromarray((modal_mask * 255).astype(np.uint8)).save(modal_mask_dir / f"modal_mask_{step:04d}.png")
                    pose_in_cam = (cam_sensor.camera.get_global_pose().inv() * asset_actor.pose)
                    t_cam = pose_in_cam.p.cpu().numpy().flatten()
                    q_cam_wxyz = pose_in_cam.q.cpu().numpy().flatten()
                    if amodal_verts is not None:
                        amodal_mask = rasterize_amodal_mask(
                            amodal_verts, amodal_faces, t_cam, q_cam_wxyz,
                            fx, fy, cx, cy, cam_cfg['width'], cam_cfg['height']
                        )
                        if bbox_from_mask(amodal_mask):
                            Image.fromarray(amodal_mask).save(amodal_mask_dir / f"amodal_mask_{step:04d}.png")
                            all_frames.append({
                                "rgb_path": f"rgb/rgb_{step:04d}.png",
                                "modal_mask_path": f"modal_mask/modal_mask_{step:04d}.png",
                                "amodal_mask_path": f"amodal_mask/amodal_mask_{step:04d}.png",
                                "bbox_modal_xywh": bbox_from_mask(modal_mask),
                                "bbox_amodal_xywh": bbox_from_mask(amodal_mask),
                                "class": asset_cfg.get("name"),
                                "pose_in_camera": {
                                    "position": t_cam.tolist(),
                                    "orientation_wxyz": q_cam_wxyz.tolist()
                                }
                            })

            if (step + 1) % 50 == 0 or step == sim_cfg['num_steps'] - 1:
                print(f"  ... simulating and saving frame {step + 1}/{sim_cfg['num_steps']}")
    finally:
        if out_cfg.get("save_annotations") and all_frames:
            with open(output_dir / "annotations.json", 'w') as f:
                json.dump({
                    "camera_intrinsics": intrinsic_matrix.tolist(),
                    "frames": all_frames
                }, f, indent=4)
            print(f"Annotations saved to '{output_dir / 'annotations.json'}'.")
        env.close()

    print(f"\nSimulation finished. Outputs saved in '{output_dir}'.")
import gymnasium as gym
import numpy as np
import sapien
import torch
import os
import json
from pathlib import Path
from PIL import Image
import argparse
import yaml
from typing import Dict, Callable

import transforms3d.euler as t3d_euler
import transforms3d.quaternions as t3d_quat

import trimesh
import cv2

from mani_skill.envs.scenes.base_env import SceneManipulationEnv
from mani_skill.envs.tasks.mobile_manipulation.robocasa.kitchen import RoboCasaKitchenEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.scene_builder.ai2thor import ArchitecTHORSceneBuilder
from mani_skill.utils.scene_builder.replicacad import ReplicaCADSceneBuilder
from mani_skill.utils.scene_builder.registration import register_scene_builder
from mani_skill.utils.structs import Actor, Articulation
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder

MOTION_PATTERNS: Dict[str, Callable] = {}

def register_motion_pattern(name: str):
    def decorator(func: Callable):
        MOTION_PATTERNS[name] = func
        return func
    return decorator

@register_motion_pattern("static")
def motion_static(mixin, start_pose, config):
    pass

@register_motion_pattern("oscillate_z")
def motion_oscillate_z(mixin, start_pose, config):
    start_p = start_pose.p
    z_center = start_p[2]
    new_z = z_center + 0.5 * np.sin(mixin._elapsed_steps[0].item() * 0.1)
    return sapien.Pose(p=[start_p[0], start_p[1], new_z])

@register_motion_pattern("circular_xy")
def motion_circular_xy(mixin, start_pose, config):
    params = config.get("motion_params", {})
    radius, speed, start_p = params.get("radius", 0.5), params.get("speed", 0.1), start_pose.p
    angle = mixin._elapsed_steps[0].item() * speed
    new_x, new_y = start_p[0] + radius * np.cos(angle), start_p[1] + radius * np.sin(angle)
    return sapien.Pose(p=[new_x, new_y, start_p[2]])

CAMERA_MOTION_PATTERNS: Dict[str, Callable] = {}

def register_camera_motion(name: str):
    def decorator(func: Callable):
        CAMERA_MOTION_PATTERNS[name] = func
        return func
    return decorator

@register_camera_motion("static")
def camera_motion_static(step, sensor, initial_pose, cfg):
    pass

@register_camera_motion("shake")
def camera_motion_shake(step, sensor, initial_pose, cfg):
    params = cfg.get("params", {})
    pos_amp, rot_amp = params.get("pos_amp", 0.05), params.get("rot_amp", 0.05)
    time = float(step)
    pos_offset = np.array([pos_amp * np.sin(time * 0.2), pos_amp * np.cos(time * 0.35), 0])
    rot_offset_q = t3d_euler.euler2quat(0, rot_amp * np.cos(time * 0.15), rot_amp * np.sin(time * 0.4), 'sxyz')
    initial_q_flat = np.array(initial_pose.q).flatten()
    new_q = t3d_quat.qmult(initial_q_flat, rot_offset_q)
    new_p_tensor = initial_pose.p + pos_offset
    final_p = new_p_tensor.cpu().numpy().flatten().astype(np.float32)
    final_q = np.array(new_q).flatten().astype(np.float32)
    sensor.camera.set_local_pose(sapien.Pose(p=final_p, q=final_q))

class ConfigurableLightingSceneBuilder:
    def __init__(self, *args, **kwargs):
        self.lighting_config = None
        super().__init__(*args, **kwargs)

    def build_lighting(self):
        if not self.lighting_config:
            return
        scene = self.env.scene
        if 'ambient' in self.lighting_config:
            scene.set_ambient_light(self.lighting_config['ambient'])
        for light_info in self.lighting_config.get('point_lights', []):
            scene.add_point_light(light_info['position'], color=light_info['color'])
        if 'directional_light' in self.lighting_config:
            dl_cfg = self.lighting_config['directional_light']
            scene.add_directional_light(dl_cfg['direction'], color=dl_cfg['color'], shadow=dl_cfg.get('shadow', False))

@register_scene_builder("ESOT500syn_ArchitecTHOR_NoRobot")
class ESOT500synArchitecTHORSceneBuilder(ConfigurableLightingSceneBuilder, ArchitecTHORSceneBuilder):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        self.build_lighting()

    def initialize(self, env_idx):
        for obj, pose in self._default_object_poses:
            obj.set_pose(pose)
            if isinstance(obj, Articulation):
                obj.set_qpos(obj.qpos[0] * 0)

@register_scene_builder("ESOT500syn_ReplicaCAD_NoRobot")
class ESOT500synReplicaCADSceneBuilder(ConfigurableLightingSceneBuilder, ReplicaCADSceneBuilder):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        self.build_lighting()

    def initialize(self, env_idx: torch.Tensor):
        for obj, pose in self._default_object_poses:
            obj.set_pose(pose)
            if isinstance(obj, Articulation):
                obj.set_qpos(obj.qpos[0] * 0)
                obj.set_qvel(obj.qvel[0] * 0)

class CustomAssetMixin:
    def __init__(self, *args, **kwargs):
        self.custom_asset_config = kwargs.pop("custom_asset_config", {})
        self.distractor_assets_config = kwargs.pop("distractor_assets_config", [])
        self.custom_actor = None
        self.distractors = []
        super().__init__(*args, **kwargs)

    def _build_actor_from_config(self, config: Dict) -> Actor | None:
        if not config.get("enable"):
            return None
        visual_path = Path(os.path.expanduser(config["visual_filepath"]))
        collision_path = Path(os.path.expanduser(config["collision_filepath"]))
        if not (visual_path.exists() and collision_path.exists()):
            print(f"Warning: Asset files not found for '{config.get('name')}'. Skipping.")
            return None
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose(p=config.get("initial_pose_p", [0, 0, 1]))
        builder.add_convex_collision_from_file(filename=str(collision_path))
        builder.add_visual_from_file(filename=str(visual_path))
        return builder.build_kinematic(name=config.get("name", visual_path.stem))

    def add_assets_to_scene(self):
        self.custom_actor = self._build_actor_from_config(self.custom_asset_config)
        for d_config in self.distractor_assets_config:
            actor = self._build_actor_from_config(d_config)
            if actor:
                motion_func = MOTION_PATTERNS.get(d_config.get("motion_mode", "static"), MOTION_PATTERNS["static"])
                self.distractors.append({"actor": actor, "config": d_config, "motion_func": motion_func, "start_pose": None})

    def initialize_asset_poses(self):
        if self.custom_actor:
            pos = self.custom_asset_config.get("initial_pose_p", [0, 0, 1])
            self.custom_asset_start_pose = sapien.Pose(p=pos)
            self.custom_actor.set_pose(self.custom_asset_start_pose)
        for d in self.distractors:
            pos = d["config"].get("initial_pose_p", [0, 0, 1])
            d["start_pose"] = sapien.Pose(p=pos)
            d["actor"].set_pose(d["start_pose"])

    def update_asset_poses(self):
        if self.custom_actor:
            motion_func = MOTION_PATTERNS.get(self.custom_asset_config.get("motion_mode", "static"))
            if motion_func:
                new_pose = motion_func(self, self.custom_asset_start_pose, self.custom_asset_config)
                if new_pose:
                    self.custom_actor.set_pose(new_pose)
        for d in self.distractors:
            new_pose = d["motion_func"](self, d["start_pose"], d["config"])
            if new_pose:
                d["actor"].set_pose(new_pose)

def _to_build_config_idxs_from_pairs(pairs, num_envs):
    idxs = [int(l)*12 + int(s) for (l, s) in pairs]
    if not idxs:
        return None
    if len(idxs) < num_envs:
        idxs = [idxs[i % len(idxs)] for i in range(num_envs)]
    return idxs

def create_and_register_env(env_type: str, camera_pos: list, camera_target: list):
    class CustomEnvBase(CustomAssetMixin):
        def __init__(self, *args, **kwargs):
            self.lighting_config = kwargs.pop("lighting_config", {})
            super().__init__(*args, **kwargs)
            if hasattr(self, "scene_builder") and isinstance(self.scene_builder, ConfigurableLightingSceneBuilder):
                self.scene_builder.lighting_config = self.lighting_config

        def _get_obs_agent(self):
            return {}

        @property
        def _default_sensor_configs(self):
            return [CameraConfig("base_camera", sapien_utils.look_at(camera_pos, camera_target), 512, 512, np.pi/2, 0.01, 100)]

        def _load_scene(self, options: dict):
            super()._load_scene(options)
            self.add_assets_to_scene()

        def _initialize_episode(self, env_idx, options):
            super()._initialize_episode(env_idx, options)
            self.initialize_asset_poses()

        def _step_action(self, action):
            return_val = super()._step_action(action)
            self.update_asset_poses()
            return return_val

    env_id = "ESOT500syn-v1"

    if env_type == "RoboCasa":
        @register_env(env_id, max_episode_steps=50, override=True)
        class ESOT_RoboCasaEnv(CustomEnvBase, RoboCasaKitchenEnv):
            def _load_scene(self, options: dict):
                # use self.layout_and_style_ids to force construct build_config_idxs
                self.scene_builder = RoboCasaSceneBuilder(self)
                pairs = getattr(self, "layout_and_style_ids", None)
                if pairs:
                    build_idxs = _to_build_config_idxs_from_pairs(pairs, self.num_envs)
                    self.scene_builder.build(build_config_idxs=build_idxs)
                else:
                    self.scene_builder.build()
                # the rest of the logic is the same as the official _load_scene (slightly simplified, only keeping the critical pipeline)
                self.fixture_refs, self.objects, self.object_cfgs, self.object_actors = [], [], [], []
                for _ in range(self.num_envs):
                    self.fixture_refs.append({}); self.objects.append({}); self.object_cfgs.append({}); self.object_actors.append({})
                if not self.fixtures_only and hasattr(self, "_get_obj_cfgs"):
                    for scene_idx in range(self.num_envs):
                        self._scene_idx_to_be_loaded = scene_idx
                        self._setup_kitchen_references()
                        def _create_obj(cfg):
                            from mani_skill.utils.scene_builder.robocasa.objects.objects import MJCFObject
                            obj_groups = cfg.get("obj_groups", "all")
                            exclude_obj_groups = cfg.get("exclude_obj_groups", None)
                            object_kwargs, object_info = self.sample_object(
                                obj_groups, exclude_groups=exclude_obj_groups,
                                graspable=cfg.get("graspable", None),
                                washable=cfg.get("washable", None),
                                microwavable=cfg.get("microwavable", None),
                                cookable=cfg.get("cookable", None),
                                freezable=cfg.get("freezable", None),
                                max_size=cfg.get("max_size", (None, None, None)),
                                object_scale=cfg.get("object_scale", None),
                                rng=self._batched_episode_rng[scene_idx],
                            )
                            if "name" not in cfg:
                                cfg["name"] = "obj_{}".format(obj_num + 1)
                            info = object_info
                            object = MJCFObject(self.scene, name=cfg["name"], **object_kwargs)
                            return object, info
                        for _ in range(10):
                            objects = {}
                            if "object_cfgs" in self._ep_meta:
                                object_cfgs = self._ep_meta["object_cfgs"]
                                for obj_num, cfg in enumerate(object_cfgs):
                                    model, info = _create_obj(cfg); cfg["info"]=info; objects[model.name]=model
                            else:
                                object_cfgs = self._get_obj_cfgs()
                                addl_obj_cfgs = []
                                for obj_num, cfg in enumerate(object_cfgs):
                                    cfg["type"] = "object"
                                    model, info = _create_obj(cfg); cfg["info"]=info; objects[model.name]=model
                                object_cfgs = addl_obj_cfgs + object_cfgs
                            self.object_cfgs[scene_idx] = object_cfgs; self.objects[scene_idx]=objects
                            placement_initializer = self.scene_builder._get_placement_initializer(
                                self.scene_builder.scene_data[self._scene_idx_to_be_loaded]["fixtures"],
                                objects, object_cfgs, rng=self._batched_episode_rng[scene_idx],
                            )
                            object_placements = None
                            for i in range(10):
                                try:
                                    object_placements = placement_initializer.sample(
                                        placed_objects=self.scene_builder.scene_data[self._scene_idx_to_be_loaded]["fxtr_placements"]
                                    )
                                except Exception:
                                    continue
                                break
                            if object_placements is None:
                                print("Could not place objects. Trying again with new objects")
                                continue
                            for obj_pos, obj_quat, obj in object_placements.values():
                                obj.pos = obj_pos; obj.quat = obj_quat
                                actor = obj.build(scene_idxs=[scene_idx]).actor
                                self.object_actors[scene_idx][obj.name] = {"actor": actor, "pose": sapien.Pose(obj_pos, obj_quat)}
                            break

            # the rest of the logic is the same as the official _load_scene (slightly simplified, only keeping the critical pipeline)

    else:
        @register_env(env_id, max_episode_steps=50, override=True)
        class ESOT500synEnv(CustomEnvBase, SceneManipulationEnv):
            def __init__(self, *args, **kwargs):
                if env_type in ["ArchitecTHOR", "ReplicaCAD"]:
                    kwargs['scene_builder_cls'] = f"ESOT500syn_{env_type}_NoRobot"
                super().__init__(*args, **kwargs)
    return env_id

def load_mesh_vertices_faces(mesh_path: str):
    mesh_path = os.path.expanduser(mesh_path)
    try:
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32)
    except Exception as e:
        print(f"Error loading mesh {mesh_path}: {e}")
        return None, None

def rasterize_amodal_mask(vertices_obj, faces, t_cam_robot, q_wxyz_robot, fx, fy, cx, cy, width, height):
    R_robot_to_cv = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    R_robot = t3d_quat.quat2mat(q_wxyz_robot.astype(np.float64))
    verts_robot_cam = (R_robot @ vertices_obj.T).T + t_cam_robot.reshape(3)
    verts_cv_cam = (R_robot_to_cv @ verts_robot_cam.T).T
    mask = np.zeros((height, width), dtype=np.uint8)
    for tri in faces:
        pts = verts_cv_cam[tri]
        Z = pts[:, 2]
        if np.any(Z <= 1e-6):
            continue
        u, v = fx * (pts[:, 0] / Z) + cx, fy * (pts[:, 1] / Z) + cy
        poly = np.stack([u, v], axis=1)
        if np.any(~np.isfinite(poly)):
            continue
        cv2.fillConvexPoly(mask, np.round(poly).astype(np.int32), 255)
    return mask

def bbox_from_mask(mask: np.ndarray):
    rows, cols = np.any(mask > 0, axis=1), np.any(mask > 0, axis=0)
    if not (np.any(rows) and np.any(cols)):
        return None
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]

def main(config: dict):
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
        # the original logic is preserved: pass layout_ids/style_ids to __init__ to generate layout_and_style_ids
        env_kwargs.update(scene_cfg['robocasa_params'])

    output_dir = Path(out_cfg['output_dir']) / f"{env_cfg['env_type']}" / f"{asset_cfg.get('name', 'no_asset')}"
    rgb_dir, modal_mask_dir, amodal_mask_dir = output_dir / "rgb", output_dir / "modal_mask", output_dir / "amodal_mask"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    if out_cfg.get("save_annotations"):
        modal_mask_dir.mkdir(parents=True, exist_ok=True)
        amodal_mask_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(env_id, render_mode=out_cfg['render_mode'], obs_mode=env_cfg['obs_mode'], sim_backend=sim_cfg['sim_backend'], **env_kwargs)

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
    initial_cam_pose = cam_sensor.camera.get_local_pose()
    cam_motion_cfg = cam_cfg.get("motion", {"type": "static"})
    cam_motion_func = CAMERA_MOTION_PATTERNS.get(cam_motion_cfg.get("type", "static"), CAMERA_MOTION_PATTERNS["static"])

    asset_actor = env.unwrapped.custom_actor if asset_cfg.get("enable") and hasattr(env.unwrapped, "custom_actor") else None
    asset_id = asset_actor.per_scene_id.item() if asset_actor else None

    all_frames, intrinsic_matrix = [], cam_sensor.camera.get_intrinsic_matrix().cpu().numpy()[0]
    fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    amodal_verts, amodal_faces = None, None
    if out_cfg.get("save_annotations") and asset_cfg.get("enable"):
        amodal_verts, amodal_faces = load_mesh_vertices_faces(asset_cfg["visual_filepath"])

    try:
        for step in range(sim_cfg['num_steps']):
            cam_motion_func(step, cam_sensor, initial_cam_pose, cam_motion_cfg)
            action = env.action_space.sample() if env.action_space else None
            obs, *_ = env.step(action)

            sensor_data = obs.get("sensor_data", {}).get("base_camera", {})
            if "Color" not in sensor_data:
                continue

            rgb_tensor = sensor_data["Color"].squeeze(0).cpu().numpy()
            img_to_save = (rgb_tensor[...,:3] * 255).astype(np.uint8) if rgb_tensor.dtype == np.float32 else rgb_tensor
            Image.fromarray(img_to_save).save(rgb_dir / f"rgb_{step:04d}.png")

            if out_cfg.get("save_annotations") and asset_id is not None and "Segmentation" in sensor_data:
                modal_mask = (sensor_data["Segmentation"][..., 1] == asset_id).squeeze().cpu().numpy()
                modal_bbox = bbox_from_mask(modal_mask)
                if modal_bbox:
                    Image.fromarray((modal_mask * 255).astype(np.uint8)).save(modal_mask_dir / f"modal_mask_{step:04d}.png")
                    pose_in_cam = (cam_sensor.camera.get_global_pose().inv() * asset_actor.pose)
                    t_cam = pose_in_cam.p.cpu().numpy().flatten()
                    q_cam_wxyz = pose_in_cam.q.cpu().numpy().flatten()
                    amodal_mask, amodal_bbox = None, None
                    if amodal_verts is not None:
                        amodal_mask = rasterize_amodal_mask(amodal_verts, amodal_faces, t_cam, q_cam_wxyz, fx, fy, cx, cy, cam_cfg['width'], cam_cfg['height'])
                        amodal_bbox = bbox_from_mask(amodal_mask)
                    if amodal_bbox:
                        Image.fromarray(amodal_mask).save(amodal_mask_dir / f"amodal_mask_{step:04d}.png")
                        all_frames.append({
                            "rgb_path": f"rgb/rgb_{step:04d}.png",
                            "modal_mask_path": f"modal_mask/modal_mask_{step:04d}.png",
                            "amodal_mask_path": f"amodal_mask/amodal_mask_{step:04d}.png",
                            "bbox_modal_xywh": modal_bbox,
                            "bbox_amodal_xywh": amodal_bbox,
                            "class": asset_cfg.get("name"),
                            "pose_in_camera": {"position": t_cam.tolist(), "orientation_wxyz": q_cam_wxyz.tolist()}
                        })
            if (step + 1) % 50 == 0 or step == sim_cfg['num_steps'] - 1:
                print(f"  ... simulating and saving frame {step+1}/{sim_cfg['num_steps']}")
    finally:
        if out_cfg.get("save_annotations") and all_frames:
            with open(output_dir / "annotations.json", 'w') as f:
                json.dump({"camera_intrinsics": intrinsic_matrix.tolist(), "frames": all_frames}, f, indent=4)
            print(f"Annotations saved to '{output_dir / 'annotations.json'}'.")
        env.close()

    print(f"\nSimulation finished. Outputs saved in '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESOT500syn: A Universal ManiSkill Scene Runner")
    parser.add_argument("--config", type=str, default="/home/chujie/Data/ESOT500syn/test/environment/config.yaml", help="Path to the YAML configuration file.")
    args = parser.parse_args()
    try:
        with open(args.config, 'r') as f:
            config_from_yaml = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at '{args.config}'"); exit(1)
    except Exception as e:
        print(f"Error parsing YAML file: {e}"); exit(1)
    main(config_from_yaml)
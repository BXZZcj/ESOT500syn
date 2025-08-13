import gymnasium as gym
import numpy as np
import sapien
import torch
import os
from pathlib import Path
from PIL import Image
import argparse
import yaml
from typing import List, Optional

# ===============================================================================================
# ... 导入部分和SceneBuilder补丁部分 (1-2) 完全不变 ...
# ===============================================================================================
from mani_skill.envs.scenes.base_env import SceneManipulationEnv
from mani_skill.envs.tasks.mobile_manipulation.robocasa.kitchen import RoboCasaKitchenEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.scene_builder.ai2thor import ArchitecTHORSceneBuilder
from mani_skill.utils.scene_builder.replicacad import ReplicaCADSceneBuilder
from mani_skill.utils.scene_builder.registration import register_scene_builder
from mani_skill.utils.structs import Actor, Articulation

@register_scene_builder("MyArchitecTHOR_NoRobot")
class MyArchitecTHORSceneBuilder_NoRobot(ArchitecTHORSceneBuilder):
    def initialize(self, env_idx):
        for obj, pose in self._default_object_poses:
            obj.set_pose(pose)
            if isinstance(obj, Articulation):
                obj.set_qpos(obj.qpos[0] * 0)

@register_scene_builder("MyReplicaCAD_NoRobot")
class MyReplicaCADSceneBuilder_NoRobot(ReplicaCADSceneBuilder):
    def initialize(self, env_idx: torch.Tensor):
        for obj, pose in self._default_object_poses:
            obj.set_pose(pose)
            if isinstance(obj, Articulation):
                obj.set_qpos(obj.qpos[0] * 0)
                obj.set_qvel(obj.qvel[0] * 0)
        if self.scene.gpu_sim_enabled and len(env_idx) == self.env.num_envs:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene.px.step()
            self.scene._gpu_fetch_all()


# ===============================================================================================
# 3. 修正 BananaAddonMixin
# ===============================================================================================
class BananaAddonMixin:
    # +++ 核心修正 1：在__init__中添加一个实例变量来存储起始姿态 +++
    def __init__(self, *args, **kwargs):
        self.banana_start_pose = None # 用于存储每个episode的起始姿态
        super().__init__(*args, **kwargs)

    def add_banana_to_scene(self):
        banana_assets_dir = os.path.expanduser("~/.maniskill/data/assets/mani_skill2_ycb/models/011_banana")
        visual_filepath = os.path.join(banana_assets_dir, "textured.obj")
        collision_filepath = os.path.join(banana_assets_dir, "collision.ply")
        builder = self.scene.create_actor_builder()
        # 设置一个默认的build-time初始姿态，以避免ManiSkill警告
        builder.initial_pose = sapien.Pose() 
        builder.add_convex_collision_from_file(filename=collision_filepath)
        builder.add_visual_from_file(filename=visual_filepath)
        self.banana_actor = builder.build_kinematic(name="banana")

    def initialize_banana_pose(self, pose: sapien.Pose):
        # +++ 核心修正 2：在初始化时，不仅设置当前姿态，还记录下这个起始姿态 +++
        self.banana_actor.set_pose(pose)
        self.banana_start_pose = pose # 记录本episode的起始点

    def update_banana_pose(self):
        if self.banana_actor and self.banana_start_pose is not None:
            # +++ 核心修正 3：使用记录的起始姿态，而不是错误的initial_pose +++
            start_p = self.banana_start_pose.p # 这是一个 (3,) 的numpy数组
            base_z = start_p[2] # 正确地获取z坐标
            
            new_z = base_z + 0.5 * np.sin(self._elapsed_steps[0].item() * 0.1)
            # 围绕正确的x, y坐标进行振荡
            new_pose = sapien.Pose(p=[start_p[0], start_p[1], new_z])
            self.banana_actor.set_pose(new_pose)

# ===============================================================================================
# 4. 工厂函数现在可以正确工作了，因为它依赖的Mixin被修正了 (无变动)
# ===============================================================================================
def create_and_register_env(env_type: str, camera_pose_p: list, camera_pose_g: list):
    # ... (这部分代码无需改动) ...
    class CustomEnvBase(BananaAddonMixin):
        def _get_obs_agent(self):
            return {} if self.agent is None else super()._get_obs_agent()
        @property
        def _default_sensor_configs(self):
            camera_pose = sapien_utils.look_at(camera_pose_p, camera_pose_g)
            return [CameraConfig("base_camera", camera_pose, 512, 512, np.pi / 2, 0.01, 100)]
        def _load_scene(self, options: dict):
            super()._load_scene(options)
            self.add_banana_to_scene()
        def _step_action(self, action):
            ret = super()._step_action(action)
            self.update_banana_pose()
            return ret

    if env_type == "RoboCasa":
        @register_env("UniversalManiSkillEnv-v1", max_episode_steps=50, override=True)
        class UniversalRoboCasaEnv(CustomEnvBase, RoboCasaKitchenEnv):
            def _initialize_episode(self, env_idx: int, options: dict):
                super()._initialize_episode(env_idx, options)
                self.initialize_banana_pose(sapien.Pose(p=[3.0, -2.0, 1.0]))
    
    elif env_type in ["ArchitecTHOR", "ReplicaCAD"]:
        @register_env("UniversalManiSkillEnv-v1", max_episode_steps=50, override=True)
        class UniversalSceneManipulationEnv(CustomEnvBase, SceneManipulationEnv):
            def __init__(self, *args, **kwargs):
                self.banana_actor = None
                kwargs_copy = kwargs.copy()
                if env_type == "ArchitecTHOR":
                    kwargs_copy['scene_builder_cls'] = "MyArchitecTHOR_NoRobot"
                else:
                    kwargs_copy['scene_builder_cls'] = "MyReplicaCAD_NoRobot"
                super().__init__(*args, **kwargs_copy)
            
            def _initialize_episode(self, env_idx: int, options: dict):
                super()._initialize_episode(env_idx, options)
                self.initialize_banana_pose(sapien.Pose(p=[-1.0, 0.0, 1.5]))
    else:
        raise ValueError(f"Unknown env_type: {env_type}")

# ===============================================================================================
# ... main函数和启动器 (6-7) 完全不变 ...
# ===============================================================================================
def main(cfg: dict):
    # ... (这部分代码无需改动) ...
    np.set_printoptions(suppress=True, precision=3)
    
    env_cfg = cfg['env']
    scene_cfg = cfg['scene']
    sim_cfg = cfg['simulation']
    output_cfg = cfg['output']
    camera_cfg = cfg['camera']

    create_and_register_env(env_cfg['env_type'], camera_cfg['pose_p'], camera_cfg['pose_g'])
    env_id = "UniversalManiSkillEnv-v1"

    env_kwargs = {
        "robot_uids": env_cfg['robot_uids'],
        "sensor_configs": {
            "base_camera": {"width": camera_cfg['width'], "height": camera_cfg['height']}
        }
    }
    
    if env_cfg['env_type'] == "RoboCasa":
        env_kwargs.update(scene_cfg['robocasa_params'])
    # ArchitecTHOR 和 ReplicaCAD 的 build_config_idxs 将在 reset 时传入

    image_dir = Path(output_cfg['image_dir']) / env_cfg['env_type']
    image_dir.mkdir(parents=True, exist_ok=True)

    # 步骤 A: 通用初始化
    env = gym.make(
        env_id,
        render_mode=output_cfg['render_mode'],
        obs_mode=env_cfg['obs_mode'],
        sim_backend=sim_cfg['sim_backend'],
        **env_kwargs
    )
    
    print("-" * 50)
    print(f"Running simulation for: {env_cfg['env_type']}")
    
    scene_to_load = None
    if env_cfg['env_type'] in ["ArchitecTHOR", "ReplicaCAD"]:
        scene_builder = env.unwrapped.scene_builder
        num_available_scenes = len(scene_builder.build_configs)
        
        if env_cfg['env_type'] == "ArchitecTHOR":
            scene_to_load = scene_cfg['archithor_params']['build_config_idx']
        else: # ReplicaCAD
            scene_to_load = scene_cfg['replicacad_params']['build_config_idx']

        print(f"Total available scenes for {env_cfg['env_type']}: {num_available_scenes}")
        
        if not (0 <= scene_to_load < num_available_scenes):
            print(f"Warning: Configured scene index {scene_to_load} is out of bounds (0-{num_available_scenes-1}).")
            scene_to_load = 0
            print(f"Defaulting to scene index {scene_to_load}.")
    
    print(f"Saving images to: {image_dir.resolve()}")
    print("-" * 50)

    reset_options = dict(reconfigure=True)
    if scene_to_load is not None:
        reset_options['build_config_idxs'] = [scene_to_load]

    obs, _ = env.reset(seed=sim_cfg['seed'], options=reset_options)
    
    try:
        for step in range(sim_cfg['num_steps']):
            action = env.action_space.sample() if env.action_space is not None else None
            obs, reward, terminated, truncated, info = env.step(action)
            if "sensor_data" in obs and "base_camera" in obs["sensor_data"] and "Color" in obs["sensor_data"]["base_camera"]:
                rgb_tensor = obs["sensor_data"]["base_camera"]["Color"]
                image_data = rgb_tensor.squeeze(0).cpu().numpy()
                Image.fromarray(image_data).save(image_dir / f"step_{step:04d}.png")
            if (step + 1) % 50 == 0 or step == sim_cfg['num_steps'] - 1:
                print(f"  ... simulating and saving frame {step+1}/{sim_cfg['num_steps']}")
    finally:
        env.close()

    print("\nSimulation finished.")
    print(f"Images saved successfully in '{image_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/chujie/Data/ESOT500syn/test/environment/config.yaml", help="Path to the YAML configuration file.")
    args = parser.parse_args()
    try:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config}'")
        exit(1)
    except Exception as e:
        print(f"Error parsing YAML file: {e}")
        exit(1)
    main(config_dict)
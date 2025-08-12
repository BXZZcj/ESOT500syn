import os
import gymnasium as gym
import numpy as np
import torch
import sapien
import importlib
import traceback
from pathlib import Path

# ManiSkill核心组件
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.sensors.camera import CameraConfig

# ------------------- 已修复的辅助函数 -------------------
def _get_env_class(env_id: str):
    """
    动态获取环境ID对应的类定义。
    使用标准的gymnasium.spec()方法，这是正确的、公开的API。
    """
    # 1. 使用gym.spec()获取环境规约
    env_spec = gym.spec(env_id)
    if env_spec is None:
        raise gym.error.Error(f"Environment with ID '{env_id}' not found.")

    # 2. 从规约中获取入口点字符串
    # e.g., "mani_skill.envs.tasks.tabletop.push_cube:PushCubeEnv"
    entry_point_str = env_spec.entry_point
    
    # 3. 动态导入模块和类
    module_path, class_name = entry_point_str.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
# ---------------------------------------------------------

class ESOT500SynEnvFactory:
    """
    一个工厂类，用于创建、配置和运行一个用于数据采集的自定义ManiSkill环境。
    """
    def __init__(self, env_config: dict, asset_config: dict, recorder_config: dict):
        """
        Args:
            env_config (dict): 环境相关配置。
            asset_config (dict): 要加载的资产的配置。
            recorder_config (dict): 录制器相关配置。
        """
        self.env_config = env_config
        self.asset_config = asset_config
        self.recorder_config = recorder_config
        
        # 1. 动态创建一个继承自基础环境的自定义环境类
        base_env_cls = _get_env_class(self.env_config["base_env_id"])
        
        _asset_cfg = self.asset_config
        _camera_cfg = self.recorder_config["camera"]
        
        @register_env("ESOT500Syn-Task-v0", max_episode_steps=self.env_config["num_steps"])
        class CustomTaskEnv(base_env_cls):
            def __init__(self, *args, **kwargs):
                self.custom_asset = None
                super().__init__(*args, **kwargs)

            def _load_scene(self, options: dict):
                super()._load_scene(options)
                builder = self.scene.create_actor_builder()
                builder.add_convex_collision_from_file(filename=_asset_cfg["collision_path"])
                builder.add_visual_from_file(filename=_asset_cfg["visual_path"])
                self.custom_asset = builder.build_kinematic(name=_asset_cfg["name"])

            def _register_render_cameras(self):
                super()._register_render_cameras()
                pose = sapien.Pose(p=_camera_cfg["p"], q=_camera_cfg["q"])
                self.render_camera_cfgs["esot_recorder_camera"] = CameraConfig(
                    pose=pose,
                    width=_camera_cfg["resolution"][0],
                    height=_camera_cfg["resolution"][1],
                    fov=_camera_cfg["fov"],
                )

        # 2. 使用我们刚刚动态创建并注册的环境
        print("--- Initializing Custom Environment ---")
        self.env = gym.make(
            "ESOT500Syn-Task-v0",
            render_mode="rgb_array",
            sim_backend="gpu" if torch.cuda.is_available() else "cpu",
        )
        
        # 3. 使用RecordEpisode包装器，并明确指定录制机位
        output_dir = self.recorder_config["output_dir"]
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.env = RecordEpisode(
            self.env, 
            output_dir,
            render_camera_names=["esot_recorder_camera"],
            info_on_video=False, 
            save_trajectory=False,
        )
        
        print("--- Environment Initialized Successfully ---")

    def run_episode(self, motion_planner, seed=None):
        """
        运行一个完整的、带有确定性运动的episode。
        """
        num_steps = self.env_config["num_steps"]
        print(f"\n--- Starting simulation for {num_steps} steps ---")
        
        obs, _ = self.env.reset(seed=seed)
        
        custom_asset_actor = self.env.unwrapped.custom_asset
        
        for i in range(num_steps):
            new_pose = motion_planner.get_pose(i)
            custom_asset_actor.set_pose(new_pose)

            action = np.zeros(self.env.action_space.shape)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.env.render()

            if (i + 1) % 50 == 0:
                print(f"  ... simulating step {i+1}/{num_steps}")
        
        print("--- Simulation Finished ---")

    def close(self):
        """关闭环境并保存视频。"""
        self.env.close()
        print(f"Environment closed. Video saved in '{self.recorder_config['output_dir']}'")

# --- 以下是当该文件被直接执行时的用法示例 ---
if __name__ == '__main__':
    class CircularMotion:
        def get_pose(self, step):
            angle = 0.05 * step
            x = 0.15 * np.cos(angle)
            y = 0.15 * np.sin(angle)
            return sapien.Pose(p=[x, y, 0.5], q=[1, 0, 0, 0])
    
    # 1. 定义所有配置
    env_config = {
        "base_env_id": "ReplicaCAD_SceneManipulation-v1",
        "num_steps": 300,
    }

    # 重要：请确保此路径在您的系统上是正确的
    # 例如: "/home/YOUR_USER/.maniskill/data/..."
    ycb_banana_dir = "/home/chujie/.maniskill/data/assets/mani_skill2_ycb/models/011_banana"
    asset_config = {
        "name": "target_object",
        "visual_path": os.path.join(ycb_banana_dir, "textured.obj"),
        "collision_path": os.path.join(ycb_banana_dir, "collision.ply"),
    }

    recorder_config = {
        "output_dir": "videos/replicaCAD_with_banana",
        "camera": {
            "p": [0.5, 0, 0.8],
            "q": [0.707, 0, -0.707, 0],
            "fov": 1.57,
            "resolution": [512, 512]
        }
    }

    # 2. 实例化并运行
    env_factory = None
    try:
        env_factory = ESOT500SynEnvFactory(
            env_config=env_config,
            asset_config=asset_config,
            recorder_config=recorder_config
        )
        motion_planner = CircularMotion()
        env_factory.run_episode(motion_planner, seed=42)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        if env_factory:
            env_factory.close()
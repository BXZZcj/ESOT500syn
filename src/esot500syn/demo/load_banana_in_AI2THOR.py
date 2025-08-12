import gymnasium as gym
import numpy as np
import sapien
import torch
import os
from pathlib import Path
from PIL import Image


# ... All your imports and custom class definitions remain unchanged ...
# (MyArchitecTHORSceneBuilder_NoRobot, CustomAblationEnv, etc.)
from mani_skill.envs.scenes.base_env import SceneManipulationEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.scene_builder.ai2thor import ArchitecTHORSceneBuilder
from mani_skill.utils.scene_builder.registration import register_scene_builder


@register_scene_builder("MyArchitecTHOR_NoRobot")
class MyArchitecTHORSceneBuilder_NoRobot(ArchitecTHORSceneBuilder):
    def initialize(self, env_idx):
        pass


@register_env("AblationArchitecTHOR-v1", max_episode_steps=50)
class CustomAblationEnv(SceneManipulationEnv):
    def __init__(self, *args, **kwargs):
        self.banana_actor = None
        kwargs_copy = kwargs.copy()
        kwargs_copy['scene_builder_cls'] = "MyArchitecTHOR_NoRobot"
        super().__init__(*args, **kwargs_copy)


    def _get_obs_agent(self):
        if self.agent is None:
            return {}
        return super()._get_obs_agent()


    @property
    def _default_sensor_configs(self):
        camera_pose = sapien_utils.look_at([-1, 1.5, 1.5], [-1, 3.0, 1.0])
        return [
            CameraConfig("base_camera", camera_pose, 512, 512, np.pi / 2, 0.01, 100)
        ]


    def _load_scene(self, options: dict):
        super()._load_scene(options)
        banana_assets_dir = "/home/chujie/.maniskill/data/assets/mani_skill2_ycb/models/011_banana"
        visual_filepath = os.path.join(banana_assets_dir, "textured.obj")
        collision_filepath = os.path.join(banana_assets_dir, "collision.ply")
        builder = self.scene.create_actor_builder()
        builder.add_convex_collision_from_file(filename=collision_filepath)
        builder.add_visual_from_file(filename=visual_filepath)
        self.banana_actor = builder.build_kinematic(name="banana")
    
    def _initialize_episode(self, env_idx: int, options: dict):
        super()._initialize_episode(env_idx, options)
        initial_pose = sapien.Pose(p=[-1, 3.0, 1.0])
        self.banana_actor.set_pose(initial_pose)


    def _step_action(self, action):
        ret = super()._step_action(action) 
        if self.banana_actor:
            new_z = 1.0 + 0.5 * np.sin(self._elapsed_steps[0].item() * 0.1)
            new_pose = sapien.Pose(p=[-1, 3.0, new_z])
            self.banana_actor.set_pose(new_pose)
        return ret



def main():
    np.set_printoptions(suppress=True, precision=3)
    env_id = "AblationArchitecTHOR-v1" 
    # ... Other settings remain unchanged ...
    render_mode = "rgb_array" 
    obs_mode = "sensor_data"
    image_dir = "images/archithor_ablation_test_no_robot"
    Path(image_dir).mkdir(parents=True, exist_ok=True)
    sim_backend = "gpu" if torch.cuda.is_available() else "cpu"
    
    # --- We can choose a scene to load, or temporarily not specify one ---
    scene_to_load = 0


    env_kwargs = dict(
        # Pass the selected scene index here
        build_config_idxs=[scene_to_load], 
        robot_uids="none",
        sensor_configs={
            "base_camera": {"width": 512, "height": 512}
        }
    )


    env = gym.make(
        env_id,
        render_mode=render_mode,
        obs_mode=obs_mode,
        sim_backend=sim_backend,
        **env_kwargs
    )
    
    # +++ Core change: Here check and print available scene information +++
    # Use env.unwrapped to access the most original environment object, bypassing possible gym wrappers
    scene_builder = env.unwrapped.scene_builder
    num_scenes = len(scene_builder.build_configs)
    print("-" * 50)
    print(f"Successfully loaded SceneBuilder for '{scene_builder.scene_dataset}' dataset.")
    print(f"Total number of available scenes: {num_scenes}")
    if num_scenes > 0:
        print(f"You can use any index from 0 to {num_scenes - 1} for 'build_config_idxs'.")
    print(f"Currently configured to load scene index: {scene_to_load}")
    print("-" * 50)


    # ... Subsequent main loop and saving logic remain exactly the same ...
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    
    # Note: Since we already passed build_config_idxs in gym.make,
    # the first reset will automatically load it. So strictly speaking reconfigure is not necessary,
    # but keeping it makes it easy to switch scenes later by calling reset again.
    obs, _ = env.reset(seed=42, options=dict(reconfigure=True)) 
    
    print("Scene setup complete. Starting simulation...")
    # ... Loop ...
    for i in range(50):
        action = env.action_space.sample() if env.action_space is not None else None
        obs, reward, terminated, truncated, info = env.step(action)
        if "sensor_data" in obs and "base_camera" in obs["sensor_data"] and "Color" in obs["sensor_data"]["base_camera"]:
            rgb_tensor = obs["sensor_data"]["base_camera"]["Color"]
            image_data = rgb_tensor.squeeze(0).cpu().numpy()
            Image.fromarray(image_data).save(os.path.join(image_dir, f"step_{i:04d}.png"))
        if (i + 1) % 50 == 0:
            print(f"  ... simulating and saving frame {i+1}/50")
            
    env.close()
    print("\nEnvironment closed.")
    print(f"Images saved successfully in '{image_dir}'.")


if __name__ == "__main__":
    main()

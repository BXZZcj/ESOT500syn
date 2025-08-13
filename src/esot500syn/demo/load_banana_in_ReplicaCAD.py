import gymnasium as gym
import numpy as np
import sapien
import torch
import os
from pathlib import Path
from PIL import Image


# 1. Import all necessary modules
from mani_skill.envs.scenes.base_env import SceneManipulationEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig


# Import the original builder to inherit and register our own builder
from mani_skill.utils.scene_builder.replicacad import ReplicaCADSceneBuilder
from mani_skill.utils.scene_builder.registration import register_scene_builder
from mani_skill.utils.structs import Actor, Articulation


# Your custom "patch" builder, this part is perfect, keep it unchanged
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


# Your custom environment class, also keep unchanged
@register_env("AblationReplicaCADNoRobot-v1", max_episode_steps=50)
class CustomAblationEnv(SceneManipulationEnv):
    def __init__(self, *args, **kwargs):
        self.banana_actor = None
        kwargs_copy = kwargs.copy()
        kwargs_copy['scene_builder_cls'] = "MyReplicaCAD_NoRobot"
        super().__init__(*args, **kwargs_copy)


    def _get_obs_agent(self):
        if self.agent is None:
            return {}
        return super()._get_obs_agent()


    @property
    def _default_sensor_configs(self):
        camera_pose = sapien_utils.look_at([0, -3, 2], [0, 0, 1])
        return [
            CameraConfig("base_camera", camera_pose, 512, 512, np.pi / 2, 0.01, 100)
        ]


    def _load_scene(self, options: dict):
        super()._load_scene(options)
        
        banana_visual_path = "/home/chujie/.maniskill/data/assets/mani_skill2_ycb/models/011_banana/textured.obj"
        banana_collision_path = "/home/chujie/.maniskill/data/assets/mani_skill2_ycb/models/011_banana/collision.ply"
        
        builder = self.scene.create_actor_builder()
        builder.add_convex_collision_from_file(filename=banana_collision_path)
        builder.add_visual_from_file(filename=banana_visual_path)
        self.banana_actor = builder.build_kinematic(name="banana")
    
    def _initialize_episode(self, env_idx: int, options: dict):
        super()._initialize_episode(env_idx, options)
        initial_pose = sapien.Pose(p=[-1.0, 0.0, 1.0])
        self.banana_actor.set_pose(initial_pose)


    def _step_action(self, action):
        ret = super()._step_action(action)
        if self.banana_actor:
            new_z = 1.0 + 0.5 * np.sin(self._elapsed_steps[0].item() * 0.1)
            new_pose = sapien.Pose(p=[-1.0, 0.0, new_z])
            self.banana_actor.set_pose(new_pose)
        return ret


# main function with modifications inspired by ArchitecTHOR example
def main():
    np.set_printoptions(suppress=True, precision=3)


    env_id = "AblationReplicaCADNoRobot-v1"
    
    render_mode = None 
    obs_mode = "sensor_data"
    image_dir = "images/replicacad_single_scene_demo" # changed to a more precise directory name
    Path(image_dir).mkdir(parents=True, exist_ok=True)


    sim_backend = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Using sim backend: {sim_backend}")
    print(f"Running in headless mode, saving images to '{image_dir}'")
    
    # +++ Core Change 1: Explicitly specify the scene index to load here +++
    scene_to_load = 0 # You can change this to any integer from 0 to (total scenes - 1)


    env_kwargs = dict(
        build_config_idxs=[scene_to_load], 
        robot_uids="none",
        sensor_configs=dict(
            base_camera=dict(
                width=512, 
                height=512
            )
        )
        # Note: We do not specify build_config_idxs during creation, but pass it during reset for flexibility
    )


    env = gym.make(
        env_id,
        render_mode=render_mode,
        obs_mode=obs_mode,
        sim_backend=sim_backend,
        **env_kwargs
    )
    
    # Print available scene information
    num_available_scenes = len(env.unwrapped.scene_builder.build_configs)
    print("-" * 50)
    print(f"Successfully loaded SceneBuilder for ReplicaCAD dataset.")
    print(f"Total number of available scenes: {num_available_scenes}")
    if num_available_scenes > 0:
        print(f"You can use any index from 0 to {num_available_scenes - 1} for 'scene_to_load'.")
    # Check if the scene ID you chose is valid
    if scene_to_load >= num_available_scenes:
        print(f"Warning: Selected scene index {scene_to_load} is out of bounds. Defaulting to 0.")
        scene_to_load = 0
    print(f"Attempting to load scene index: {scene_to_load}")
    print("-" * 50)
    
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)


    # +++ Core Change 2: Pass the chosen scene index during reset +++
    obs, _ = env.reset(seed=42, options=dict(reconfigure=True))
    
    print("Scene setup complete. Starting simulation...")
    
    # +++ Core Change 3: Remove outer loop, simulate only one scene +++
    try:
        for step in range(50):
            action = None
            obs, reward, terminated, truncated, info = env.step(action)
            
            if "sensor_data" in obs and "base_camera" in obs["sensor_data"] and "Color" in obs["sensor_data"]["base_camera"]:
                rgb_tensor = obs["sensor_data"]["base_camera"]["Color"]
                image_data = rgb_tensor.squeeze(0).cpu().numpy()
                # Save directly to main image directory
                filepath = Path(image_dir) / f"step_{step:04d}.png"
                img = Image.fromarray(image_data)
                img.save(filepath)


            if (step + 1) % 50 == 0:
                print(f"  ... simulating and saving frame {step+1}/50")
    finally:
        env.close()


    print("\nSimulation finished.")
    print(f"Images for scene {scene_to_load} saved successfully in '{image_dir}'.")


if __name__ == "__main__":
    main()

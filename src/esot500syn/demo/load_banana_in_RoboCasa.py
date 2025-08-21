import gymnasium as gym
import numpy as np
import sapien
import torch
import os
from pathlib import Path
from PIL import Image


from mani_skill.envs.tasks.mobile_manipulation.robocasa.kitchen import RoboCasaKitchenEnv
from mani_skill.utils.registration import register_env


@register_env("AblationRoboCasaNoRobot-v1", max_episode_steps=50)
class CustomAblationEnv(RoboCasaKitchenEnv):
    def __init__(self, *args, **kwargs):
        self.banana_actor = None
        super().__init__(*args, **kwargs)


    def _get_obs_agent(self):
        if self.agent is None:
            return {}
        return super()._get_obs_agent()


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
        initial_pose = sapien.Pose(p=[3.0, -2.0, 1.0])
        self.banana_actor.set_pose(initial_pose)


    def _step_action(self, action):
        super()._step_action(action)
        if self.banana_actor:
            new_z = 1.0 + 0.5 * np.sin(self._elapsed_steps[0].item() * 0.1)
            new_pose = sapien.Pose(p=[3.0, -2.0, new_z])
            self.banana_actor.set_pose(new_pose)


def main():
    np.set_printoptions(suppress=True, precision=3)


    env_id = "AblationRoboCasaNoRobot-v1"
    render_mode = None 
    obs_mode = "sensor_data"
    image_dir = "images/final_random_scenes_demo"  # use new output directory
    Path(image_dir).mkdir(parents=True, exist_ok=True)


    sim_backend = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"using simulation backend: {sim_backend}")
    print(f"running in headless mode, images will be saved to '{image_dir}'")
    
    # <--- 1. enable full scene randomization ---
    env_kwargs = dict(
        layout_ids=-1,  # randomly select from all layouts
        style_ids=-1,   # randomly select from all styles
        robot_uids="none",
        sensor_configs=dict(
            base_camera=dict(
                width=512, 
                height=512
            )
        )
    )


    env = gym.make(
        env_id,
        render_mode=render_mode,
        obs_mode=obs_mode,
        sim_backend=sim_backend,
        **env_kwargs
    )
    
    print("observation space", env.observation_space)
    print("action space", env.action_space)


    print("\nscene setup completed. starting to simulate 5 different scenes...")
    
    # <--- 2. adjust the loop structure to generate 5 different scenes ---
    # outer loop controls episodes (generating 5 different scenes)
    try:
        for i in range(5):
            print(f"\n--- 开始第 {i+1}/5 个 Episode ---")
            
            # forcefully rebuild a new random scene every time reset is called
            obs, _ = env.reset(seed=i, options=dict(reconfigure=True))
            
            # create a subfolder for each episode's images to prevent overlap
            episode_dir = Path(image_dir) / f"episode_{i}"
            episode_dir.mkdir(exist_ok=True)


            # inner loop controls steps within a single episode
            for step in range(50):
                action = None
                obs, reward, terminated, truncated, info = env.step(action)
                
                # <--- 3. update the file saving path ---
                if "sensor_data" in obs:
                    if "base_camera" in obs["sensor_data"]:
                        if "Color" in obs["sensor_data"]["base_camera"]:
                            rgb_tensor = obs["sensor_data"]["base_camera"]["Color"]
                            
                            if isinstance(rgb_tensor, torch.Tensor):
                                image_data = rgb_tensor.squeeze(0).cpu().numpy()
                            else:
                                image_data = rgb_tensor.squeeze(0)


                            img = Image.fromarray(image_data)
                            # save the image to the corresponding episode subfolder
                            filepath = episode_dir / f"step_{step:04d}.png"
                            img.save(filepath)


                if (step + 1) % 50 == 0:
                    print(f"  ... simulating and saving frame {step+1}/50")
            
            print(f"Episode {i+1} completed. images saved to {episode_dir}")


    finally:
        env.close()


    print("\nsimulation completed.")
    print(f"5 different scenes' images successfully saved to '{image_dir}'.")


if __name__ == "__main__":
    main()

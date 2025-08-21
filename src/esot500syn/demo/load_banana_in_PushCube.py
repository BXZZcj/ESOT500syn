import gymnasium as gym
import numpy as np
import sapien
import torch
import os
from pathlib import Path
import traceback
from PIL import Image  # Import Pillow library for image saving


# Import core components from ManiSkill and Gymnasium
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils.registration import register_env


# ------------------- 1. Define the custom environment (same as before) -------------------
@register_env("CustomPushCube-v1", max_episode_steps=200)
class CustomPushCubeEnv(PushCubeEnv):
    def __init__(self, *args, **kwargs):
        self.banana_actor = None
        super().__init__(*args, **kwargs)


    def _load_scene(self, options: dict):
        super()._load_scene(options)
        
        # Note: Please ensure this path is correct on your system
        banana_assets_dir = "/home/chujie/.maniskill/data/assets/mani_skill2_ycb/models/011_banana"
        visual_filepath = os.path.join(banana_assets_dir, "textured.obj")
        collision_filepath = os.path.join(banana_assets_dir, "collision.ply")
        
        builder = self.scene.create_actor_builder()
        builder.add_convex_collision_from_file(filename=collision_filepath)
        builder.add_visual_from_file(filename=visual_filepath)
        self.banana_actor = builder.build_kinematic(name="banana")
    
    def _initialize_episode(self, env_idx: int, options: dict):
        super()._initialize_episode(env_idx, options)
        initial_pose = sapien.Pose(p=[0, 0.2, 0.1])
        self.banana_actor.set_pose(initial_pose)


def main():
    np.set_printoptions(suppress=True, precision=3)


    env_id = "CustomPushCube-v1"
    render_mode = "rgb_array" 
    
    image_dir = "images/custom_env_demo"
    Path(image_dir).mkdir(parents=True, exist_ok=True)


    sim_backend = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Using sim backend: {sim_backend}")
    print(f"Running in headless '{render_mode}' mode. Images will be saved to '{image_dir}'")
    
    # --- Environment initialization ---
    env = gym.make(
        env_id,
        render_mode=render_mode,
        sim_backend=sim_backend,
    )
    
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)


    # --- Environment reset ---
    obs, _ = env.reset(seed=42)


    # --- Rendering and main loop ---
    print("Scene setup complete. Starting simulation...")
    
    step_count = 0
    for i in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        banana_actor = env.unwrapped.banana_actor
        if banana_actor:
            new_x = 0.2 * np.sin(step_count * 0.1)
            new_pose = sapien.Pose(p=[new_x, 0.2, 0.1])
            banana_actor.set_pose(new_pose)
        
        # Manually render and save each frame as an image
        frame = env.render()
        if frame is not None:
            # --- Simplified type and shape handling logic ---
            # 1. If it is a Tensor (GPU mode), convert it to NumPy array
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()


            # 2. Extract single image (H, W, 3) from batch shape (1, H, W, 3)
            #     This is the key fix for KeyError
            image_data = frame[0]
            
            # 3. Convert the processed NumPy array to a PIL image
            img = Image.fromarray(image_data)
            
            # 4. Save the image
            filepath = os.path.join(image_dir, f"step_{i:04d}.png")
            img.save(filepath)


        step_count += 1
        
        if (i + 1) % 50 == 0:
            print(f"  ... simulating and saving frame {i+1}/200")


    env.close()
    print("\nEnvironment closed.")
    print(f"Images saved successfully in '{image_dir}'.")


if __name__ == "__main__":
    main()

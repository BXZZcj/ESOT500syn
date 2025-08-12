import gymnasium as gym
import numpy as np
import sapien
import torch
import os
from pathlib import Path
from PIL import Image


# 导入基类
from mani_skill.envs.tasks.mobile_manipulation.robocasa.kitchen import RoboCasaKitchenEnv
from mani_skill.utils.registration import register_env


# 您的自定义环境代码完全保持不变
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
    image_dir = "images/final_random_scenes_demo"  # 使用新的输出目录
    Path(image_dir).mkdir(parents=True, exist_ok=True)


    sim_backend = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"使用的模拟后端: {sim_backend}")
    print(f"以无头模式运行，图片将保存到 '{image_dir}'")
    
    # <--- 修改点 1：启用完整的场景随机化 ---
    env_kwargs = dict(
        layout_ids=-1,  # 从所有布局中随机选择
        style_ids=-1,   # 从所有风格中随机选择
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
    
    print("观测空间", env.observation_space)
    print("动作空间", env.action_space)


    print("\n场景设置完成。开始对5个不同的场景进行模拟...")
    
    # <--- 修改点 2：调整循环结构以生成5个不同的场景 ---
    # 外层循环控制 episodes（生成5个不同的场景）
    try:
        for i in range(5):
            print(f"\n--- 开始第 {i+1}/5 个 Episode ---")
            
            # 每次重置时都强制重新构建一个随机的新场景
            obs, _ = env.reset(seed=i, options=dict(reconfigure=True))
            
            # 为每个 episode 的图片创建一个子文件夹，防止互相覆盖
            episode_dir = Path(image_dir) / f"episode_{i}"
            episode_dir.mkdir(exist_ok=True)


            # 内层循环控制单个 episode 内的步骤
            for step in range(50):
                action = None
                obs, reward, terminated, truncated, info = env.step(action)
                
                # <--- 修改点 3：更新文件保存路径 ---
                if "sensor_data" in obs:
                    if "base_camera" in obs["sensor_data"]:
                        if "Color" in obs["sensor_data"]["base_camera"]:
                            rgb_tensor = obs["sensor_data"]["base_camera"]["Color"]
                            
                            if isinstance(rgb_tensor, torch.Tensor):
                                image_data = rgb_tensor.squeeze(0).cpu().numpy()
                            else:
                                image_data = rgb_tensor.squeeze(0)


                            img = Image.fromarray(image_data)
                            # 将图片保存到对应的 episode 子文件夹中
                            filepath = episode_dir / f"step_{step:04d}.png"
                            img.save(filepath)


                if (step + 1) % 50 == 0:
                    print(f"  ... 模拟并保存帧 {step+1}/50")
            
            print(f"Episode {i+1} 完成。图片已保存到 {episode_dir}")


    finally:
        env.close()


    print("\n模拟结束。")
    print(f"5个不同场景的图片已成功保存至 '{image_dir}'。")


if __name__ == "__main__":
    main()

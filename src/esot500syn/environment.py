import gymnasium as gym
import numpy as np
import torch

# ManiSkill相关的核心引用
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import RecordEpisode

# 这是一个推荐的实践：将核心环境设置封装在一个类中
# 未来您的 main.py 可以直接实例化这个类，而不是重复设置代码
class ESOT500SynEnv:
    """
    封装与ManiSkill/Sapien交互的环境类。
    负责加载环境、配置相机和渲染模式。
    """
    def __init__(self, env_id, render_mode, record_dir=None, **kwargs):
        """
        初始化 ManiSkill 环境。

        Args:
            env_id (str): 要加载的环境ID。
            render_mode (str): 渲染模式 (e.g., "rgb_array", "human")。
            record_dir (str, optional): 视频录制保存的目录。
            **kwargs: 其他传递给 gym.make 的参数。
        """
        print("Initializing environment...")
        
        # 构建传递给 gym.make 的参数字典
        env_kwargs = dict(
            obs_mode=kwargs.get("obs_mode", "none"),
            reward_mode=kwargs.get("reward_mode", "dense"),
            control_mode=kwargs.get("control_mode", "pd_joint_delta_pos"),
            render_mode=render_mode,
            sim_backend="gpu" if torch.cuda.is_available() else "cpu",
        )
        # 合并任何额外的用户定义参数
        env_kwargs.update(kwargs)

        # 创建 ManiSkill 环境
        self.env: BaseEnv = gym.make(env_id, **env_kwargs)

        # 如果指定了录制目录，则使用 RecordEpisode 包装器
        if record_dir:
            print(f"Recording is enabled. Videos will be saved to: {record_dir}")
            self.env = RecordEpisode(
                self.env, 
                record_dir, 
                info_on_video=False, 
                save_trajectory=False, 
                max_steps_per_video=gym_utils.find_max_episode_steps_value(self.env)
            )
        
        print("Environment initialized successfully.")
        print(f" - Observation space: {self.env.observation_space}")
        print(f" - Action space: {self.env.action_space}")
        if self.env.unwrapped.agent is not None:
            print(f" - Control mode: {self.env.unwrapped.control_mode}")

    def run_random_actions(self, seed=None):
        """
        运行一个完整的 episode，其中 agent 执行随机动作。
        """
        print("\nStarting simulation with random actions...")
        
        obs, _ = self.env.reset(seed=seed)
        
        if seed is not None and self.env.action_space is not None:
            self.env.action_space.seed(seed)

        while True:
            # 核心动作逻辑：目前是随机采样
            # TODO: 未来这里将替换为 motion.py 中定义的确定性或随机性运动
            action = self.env.action_space.sample() if self.env.action_space is not None else None
            
            # 环境步进
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # 在非 human 模式下，这一行对于录制是必须的，它会触发图像的渲染和捕获
            if self.env.render_mode != "human":
                self.env.render()
            
            # 检查 episode 是否结束
            if (terminated | truncated).any():
                print("Episode finished.")
                break
        
        print("Simulation finished.")

    def close(self):
        """
        关闭环境并释放资源。
        """
        self.env.close()
        print("Environment closed.")


# --- 以下是当该文件被直接执行时的入口 ---
# python src/esot500syn/environment.py
if __name__ == '__main__':
    # 1. 定义与您命令行完全一致的参数
    # python -m mani_skill.examples.demo_random_action \
    #   -e "ReplicaCAD_SceneManipulation-v1" \
    #   --render-mode="rgb_array" --record-dir="videos"
    
    ENV_ID = "ReplicaCAD_SceneManipulation-v1"
    RENDER_MODE = "rgb_array"
    RECORD_DIR = "videos"
    
    print("--- Running ESOT500Syn Environment Standalone Test ---")
    print(f"Environment ID: {ENV_ID}")
    print(f"Render Mode: {RENDER_MODE}")
    print(f"Record Directory: {RECORD_DIR}")
    print("----------------------------------------------------")

    # 2. 实例化并运行环境
    # 我们将 obs_mode 设置为 'none' 以匹配 demo_random_action 的默认行为
    # 这可以减少不必要的计算开销
    esot_env = ESOT500SynEnv(
        env_id=ENV_ID,
        render_mode=RENDER_MODE,
        record_dir=RECORD_DIR,
        obs_mode="none" 
    )

    try:
        # 运行一个随机动作的 episode
        esot_env.run_random_actions(seed=42) # 使用固定的种子以保证可复现性
    finally:
        # 确保环境被正确关闭
        esot_env.close()
        
    if RECORD_DIR:
        print(f"\nCheck for the recorded video in the '{RECORD_DIR}/' directory.")
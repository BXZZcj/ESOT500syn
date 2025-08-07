# src/esot500syn/motion.py
from abc import ABC, abstractmethod
import numpy as np

class BaseMotion(ABC):
    """运动模式的抽象基类"""
    def __init__(self, env, **kwargs):
        self.env = env
        self.max_steps = self.env.unwrapped.max_episode_steps

    @abstractmethod
    def reset(self):
        """在每个episode开始时重置运动状态"""
        pass

    @abstractmethod
    def get_action(self, obs, info):
        """
        根据当前状态和时间步计算下一个动作。
        
        Args:
            obs: 环境观察值
            info: 环境信息字典

        Returns:
            一个与环境动作空间匹配的动作
        """
        pass

class CircularPushMotion(BaseMotion):
    """
    控制机械臂末端执行一个圆周运动来推动物体。
    这是一个开环（Open-loop）控制器，不依赖物体位置反馈。
    """
    def __init__(self, env, center, radius, speed, height, **kwargs):
        super().__init__(env, **kwargs)
        self.center = np.array(center)
        self.radius = radius
        self.speed = speed
        self.height = height
        
        # 确认动作空间是期望的类型
        # PushCube-v1 的 action 是 [dx, dy, dz, gripper_action]
        assert len(self.env.action_space.low) == 4, "This motion is designed for a 4-dim action space."
        
        self.reset()

    def reset(self):
        """重置时间和初始位置"""
        self.time_step = 0
        # 获取机械臂的初始位置作为参考
        # 注意：这里我们假设机械臂在重置后位置相对固定
        # 更好的做法是在 get_action 中使用 obs 获取实时位置
        self.initial_tcp_pose = self.env.unwrapped.agent.tcp.pose
        self.initial_pos = self.initial_tcp_pose.p

    def get_action(self, obs, info):
        """计算下一个动作"""
        if self.time_step >= self.max_steps:
            return np.zeros_like(self.env.action_space.sample())

        # 计算当前角度
        angle = self.speed * self.time_step * self.env.unwrapped.sim_timestep

        # 计算世界坐标系下的目标位置
        target_x = self.center[0] + self.radius * np.cos(angle)
        target_y = self.center[1] + self.radius * np.sin(angle)
        target_z = self.height
        target_pos = np.array([target_x, target_y, target_z])
        
        # 获取当前的TCP位置
        # 'state' observation 包含了 agent 的所有状态信息
        current_tcp_pos = obs['agent']['tcp_pose'][:3]

        # 计算差分作为动作
        # 我们希望机械臂移动到目标位置，所以动作为 (目标 - 当前)
        # 通过乘以一个增益来控制移动速度
        delta_pos = (target_pos - current_tcp_pos) * 10.0
        
        # 保持gripper闭合以推动物体
        gripper_action = -1.0 # 负值表示闭合

        action = np.hstack([delta_pos, gripper_action])
        
        self.time_step += 1
        return action

# 运动模式的工厂函数，方便recorder调用
MOTION_REGISTRY = {
    "circular_push": CircularPushMotion,
}

def get_motion_planner(name, env, params):
    if name not in MOTION_REGISTRY:
        raise ValueError(f"Motion '{name}' not found in registry. Available motions: {list(MOTION_REGISTRY.keys())}")
    return MOTION_REGISTRY[name](env=env, **params)
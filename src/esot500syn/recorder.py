# src/esot500syn/recorder.py
import yaml
import torch # 引入torch以进行cuda检查

from .environment import ESOT500SynEnv
from .motion import get_motion_planner

class Recorder:
    """
    负责录制循环的核心逻辑。
    """
    def __init__(self, config_path):
        """
        Args:
            config_path (str): .yaml配置文件的路径。
        """
        print(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.env_config = self.config['environment']
        self.recorder_config = self.config['recorder']
        self.motion_config = self.config['motion']

    def run(self):
        """
        启动录制流程。
        """
        # 1. 初始化环境
        esot_env_wrapper = ESOT500SynEnv(
            env_id=self.env_config['env_id'],
            render_mode=self.env_config['render_mode'],
            record_dir=self.recorder_config['output_dir'],
            obs_mode=self.env_config['obs_mode']
        )
        env = esot_env_wrapper.env

        # 2. 初始化运动规划器
        motion_planner = get_motion_planner(
            name=self.motion_config['name'],
            env=env,
            params=self.motion_config['params']
        )

        print("\n--- Starting Recording Session ---")
        try:
            # 3. 运行录制主循环
            seed = self.recorder_config.get('seed')
            obs, info = env.reset(seed=seed)
            motion_planner.reset()
            
            while True:
                # 从运动规划器获取确定性动作
                action = motion_planner.get_action(obs, info)
                
                # 环境步进
                obs, reward, terminated, truncated, info = env.step(action)
                
                # 渲染（对于rgb_array模式，这是捕获帧所必需的）
                env.render()
                
                if (terminated | truncated).any():
                    print("Episode finished. Saving recording...")
                    break
        finally:
            # 4. 确保环境被正确关闭
            esot_env_wrapper.close()
            print(f"--- Recording Session Finished ---")
            print(f"Video saved in: {self.recorder_config['output_dir']}")
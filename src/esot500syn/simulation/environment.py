# src/esot500syn/simulation/environment.py
import numpy as np
from mani_skill.envs.scenes.base_env import SceneManipulationEnv
from mani_skill.envs.tasks.mobile_manipulation.robocasa.kitchen import RoboCasaKitchenEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig
from .mixins import CustomAssetMixin
from .scene_builders import ConfigurableLightingSceneBuilder # Note: Ensure builders are imported

def create_and_register_env(env_type: str, camera_pos: list, camera_target: list):
    class CustomEnvBase(CustomAssetMixin):
        def __init__(self, *args, **kwargs):
            self.lighting_config = kwargs.pop("lighting_config", {})
            super().__init__(*args, **kwargs)
            if hasattr(self, "scene_builder") and isinstance(self.scene_builder, ConfigurableLightingSceneBuilder):
                self.scene_builder.lighting_config = self.lighting_config
        def _get_obs_agent(self): return {}
        @property
        def _default_sensor_configs(self):
            return [CameraConfig("base_camera", sapien_utils.look_at(camera_pos, camera_target), 512, 512, np.pi/2, 0.01, 100)]
        def _load_scene(self, options: dict):
            super()._load_scene(options)
            self.add_assets_to_scene()
        def _initialize_episode(self, env_idx, options):
            super()._initialize_episode(env_idx, options)
            self.initialize_asset_poses()
        def _step_action(self, action):
            return_val = super()._step_action(action)
            self.update_asset_poses()
            return return_val

    env_id = "ESOT500syn-v1"
    base_classes = (CustomEnvBase, SceneManipulationEnv) if env_type != "RoboCasa" else (CustomEnvBase, RoboCasaKitchenEnv)
    
    @register_env(env_id, max_episode_steps=50, override=True)
    class ESOT500synEnv(*base_classes):
        def __init__(self, *args, **kwargs):
            if env_type in ["ArchitecTHOR", "ReplicaCAD"]:
                kwargs['scene_builder_cls'] = f"ESOT500syn_{env_type}_NoRobot"
            super().__init__(*args, **kwargs)
    return env_id
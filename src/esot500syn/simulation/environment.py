import numpy as np
from mani_skill.envs.scenes.base_env import SceneManipulationEnv
from mani_skill.envs.tasks.mobile_manipulation.robocasa.kitchen import RoboCasaKitchenEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
import sapien

from .mixins import CustomAssetMixin
from .scene_builders import ConfigurableLightingSceneBuilder

def _rc_build_config_idxs_from_params(rc_params, num_envs: int, rng: np.random.RandomState):
    """
    Calculate the build_config_idxs required by RoboCasaSceneBuilder based on the given layout_ids and style_ids.
    -1 is used as a special value to indicate random sampling from all available options.
    """
    layout_ids = rc_params.get("layout_ids", -1)
    style_ids = rc_params.get("style_ids", -1)
    
    # step 1 & 2: smart handling of layout_ids
    temp_ls = layout_ids if isinstance(layout_ids, (list, tuple)) else [layout_ids]
    if -1 in temp_ls:
        # RoboCasa has 10 layouts (0-9)
        Ls = list(range(10)) 
        print("INFO: Randomly sampling from all available RoboCasa layouts.")
    else:
        Ls = temp_ls

    # step 1 & 3: smart handling of style_ids
    temp_ss = style_ids if isinstance(style_ids, (list, tuple)) else [style_ids]
    if -1 in temp_ss:
        # RoboCasa has 12 styles (0-11)
        Ss = list(range(12))
        print("INFO: Randomly sampling from all available RoboCasa styles.")
    else:
        Ss = temp_ss

    build_idxs = []
    # step 4: randomly sample combinations for each environment instance
    for _ in range(num_envs):
        l = rng.choice(Ls)
        s = rng.choice(Ss)
        # step 5: calculate the final index
        build_idxs.append(int(l) * 12 + int(s))
        
    return build_idxs


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
    
    if env_type == "RoboCasa":
        @register_env(env_id, max_episode_steps=50, override=True)
        class ESOT_RoboCasaEnv(CustomEnvBase, RoboCasaKitchenEnv):
            def __init__(self, *args, **kwargs):
                self._rc_params_from_config = kwargs.pop("robocasa_params", {})
                super().__init__(*args, **kwargs)

            def _load_scene(self, options: dict):
                self.scene_builder = RoboCasaSceneBuilder(self)
                
                build_idxs = _rc_build_config_idxs_from_params(self._rc_params_from_config, self.num_envs, self.unwrapped._episode_rng)
                
                if build_idxs and len(build_idxs) == self.num_envs:
                    self.scene_builder.build(build_config_idxs=build_idxs)
                else:
                    # if no valid configuration or calculation error, fall back to the official random logic
                    print("WARNING: Could not determine specific RoboCasa indices, falling back to random.")
                    self.scene_builder.build()
                
                self.fixture_refs, self.objects, self.object_cfgs, self.object_actors = [], [], [], []
                for _ in range(self.num_envs):
                    self.fixture_refs.append({}); self.objects.append({}); self.object_cfgs.append({}); self.object_actors.append({})
                if not self.fixtures_only and hasattr(self, "_get_obj_cfgs"):
                    pass
                self.add_assets_to_scene()
                
    else: # ArchitecTHOR or ReplicaCAD
        @register_env(env_id, max_episode_steps=50, override=True)
        class ESOT500synEnv(CustomEnvBase, SceneManipulationEnv):
            def __init__(self, *args, **kwargs):
                if env_type in ["ArchitecTHOR", "ReplicaCAD"]:
                    kwargs['scene_builder_cls'] = f"ESOT500syn_{env_type}_NoRobot"
                super().__init__(*args, **kwargs)
                
    return env_id
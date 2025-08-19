import torch
from mani_skill.utils.scene_builder.ai2thor import ArchitecTHORSceneBuilder
from mani_skill.utils.scene_builder.replicacad import ReplicaCADSceneBuilder
from mani_skill.utils.scene_builder.registration import register_scene_builder
from mani_skill.utils.structs import Articulation

class ConfigurableLightingSceneBuilder:
    def __init__(self, *args, **kwargs):
        self.lighting_config = None
        super().__init__(*args, **kwargs)

    def build_lighting(self):
        if not self.lighting_config: return
        scene = self.env.scene
        if 'ambient' in self.lighting_config: scene.set_ambient_light(self.lighting_config['ambient'])
        for light_info in self.lighting_config.get('point_lights', []): scene.add_point_light(light_info['position'], color=light_info['color'])
        if 'directional_light' in self.lighting_config:
            dl_cfg = self.lighting_config['directional_light']
            scene.add_directional_light(dl_cfg['direction'], color=dl_cfg['color'], shadow=dl_cfg.get('shadow', False))

@register_scene_builder("ESOT500syn_ArchitecTHOR_NoRobot")
class ESOT500synArchitecTHORSceneBuilder(ConfigurableLightingSceneBuilder, ArchitecTHORSceneBuilder):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        self.build_lighting()
    def initialize(self, env_idx):
        for obj, pose in self._default_object_poses:
            obj.set_pose(pose)
            if isinstance(obj, Articulation): obj.set_qpos(obj.qpos[0] * 0)

@register_scene_builder("ESOT500syn_ReplicaCAD_NoRobot")
class ESOT500synReplicaCADSceneBuilder(ConfigurableLightingSceneBuilder, ReplicaCADSceneBuilder):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        self.build_lighting()
    def initialize(self, env_idx: torch.Tensor):
        for obj, pose in self._default_object_poses:
            obj.set_pose(pose)
            if isinstance(obj, Articulation):
                obj.set_qpos(obj.qpos[0] * 0)
                obj.set_qvel(obj.qvel[0] * 0)
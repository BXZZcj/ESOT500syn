# src/esot500syn/simulation/mixins.py
import os
import sapien
from pathlib import Path
from typing import Dict

from mani_skill.utils.structs import Actor
from ..motion.object import MOTION_PATTERNS # Relative import

class CustomAssetMixin:
    def __init__(self, *args, **kwargs):
        self.custom_asset_config = kwargs.pop("custom_asset_config", {})
        self.distractor_assets_config = kwargs.pop("distractor_assets_config", [])
        self.custom_actor = None
        self.distractors = []
        super().__init__(*args, **kwargs)

    def _build_actor_from_config(self, config: Dict) -> Actor | None:
        if not config.get("enable"): return None
        visual_path = Path(os.path.expanduser(config["visual_filepath"]))
        collision_path = Path(os.path.expanduser(config["collision_filepath"]))
        if not (visual_path.exists() and collision_path.exists()):
            print(f"Warning: Asset files not found for '{config.get('name')}'. Skipping.")
            return None
        builder = self.scene.create_actor_builder()
        initial_pos = config.get("initial_pose_p", [0, 0, 1])
        builder.initial_pose = sapien.Pose(p=initial_pos)
        builder.add_convex_collision_from_file(filename=str(collision_path))
        builder.add_visual_from_file(filename=str(visual_path))
        return builder.build_kinematic(name=config.get("name", visual_path.stem))

    def add_assets_to_scene(self):
        self.custom_actor = self._build_actor_from_config(self.custom_asset_config)
        for d_config in self.distractor_assets_config:
            actor = self._build_actor_from_config(d_config)
            if actor:
                motion_func = MOTION_PATTERNS.get(d_config.get("motion_mode", "static"), MOTION_PATTERNS["static"])
                self.distractors.append({"actor": actor, "config": d_config, "motion_func": motion_func, "start_pose": None})

    def initialize_asset_poses(self):
        if self.custom_actor:
            pos = self.custom_asset_config.get("initial_pose_p", [0, 0, 1])
            self.custom_asset_start_pose = sapien.Pose(p=pos)
            self.custom_actor.set_pose(self.custom_asset_start_pose)
        for d in self.distractors:
            pos = d["config"].get("initial_pose_p", [0, 0, 1])
            d["start_pose"] = sapien.Pose(p=pos)
            d["actor"].set_pose(d["start_pose"])

    def update_asset_poses(self):
        if self.custom_actor:
            motion_func = MOTION_PATTERNS.get(self.custom_asset_config.get("motion_mode", "static"))
            if motion_func:
                new_pose = motion_func(self, self.custom_asset_start_pose, self.custom_asset_config)
                if new_pose: self.custom_actor.set_pose(new_pose)
        for d in self.distractors:
            new_pose = d["motion_func"](self, d["start_pose"], d["config"])
            if new_pose: d["actor"].set_pose(new_pose)
# src/esot500syn/motion/object.py
import numpy as np
import sapien
from typing import Dict, Callable

MOTION_PATTERNS: Dict[str, Callable] = {}

def register_motion_pattern(name: str):
    def decorator(func: Callable):
        MOTION_PATTERNS[name] = func
        return func
    return decorator

@register_motion_pattern("static")
def motion_static(mixin, start_pose, config):
    pass

@register_motion_pattern("oscillate_z")
def motion_oscillate_z(mixin, start_pose, config):
    start_p = start_pose.p
    z_center = start_p[2]
    new_z = z_center + 0.5 * np.sin(mixin._elapsed_steps[0].item() * 0.1)
    return sapien.Pose(p=[start_p[0], start_p[1], new_z])

@register_motion_pattern("circular_xy")
def motion_circular_xy(mixin, start_pose, config):
    params = config.get("motion_params", {})
    radius, speed, start_p = params.get("radius", 0.5), params.get("speed", 0.1), start_pose.p
    angle = mixin._elapsed_steps[0].item() * speed
    new_x, new_y = start_p[0] + radius * np.cos(angle), start_p[1] + radius * np.sin(angle)
    return sapien.Pose(p=[new_x, new_y, start_p[2]])
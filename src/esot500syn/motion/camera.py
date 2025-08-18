# src/esot500syn/motion/camera.py
import numpy as np
import sapien
import transforms3d.euler as t3d_euler
import transforms3d.quaternions as t3d_quat
from typing import Dict, Callable

CAMERA_MOTION_PATTERNS: Dict[str, Callable] = {}

def register_camera_motion(name: str):
    def decorator(func: Callable):
        CAMERA_MOTION_PATTERNS[name] = func
        return func
    return decorator

@register_camera_motion("static")
def camera_motion_static(step, sensor, initial_pose, cfg):
    pass

@register_camera_motion("shake")
def camera_motion_shake(step, sensor, initial_pose, cfg):
    params = cfg.get("params", {})
    pos_amp, rot_amp = params.get("pos_amp", 0.05), params.get("rot_amp", 0.05)
    time = float(step)
    pos_offset = np.array([pos_amp * np.sin(time * 0.2), pos_amp * np.cos(time * 0.35), 0])
    rot_offset_q = t3d_euler.euler2quat(0, rot_amp * np.cos(time * 0.15), rot_amp * np.sin(time * 0.4), 'sxyz')
    initial_q_flat = np.array(initial_pose.q).flatten()
    new_q = t3d_quat.qmult(initial_q_flat, rot_offset_q)
    new_p_tensor = initial_pose.p + pos_offset
    final_p = new_p_tensor.cpu().numpy().flatten().astype(np.float32)
    final_q = np.array(new_q).flatten().astype(np.float32)
    sensor.camera.set_local_pose(sapien.Pose(p=final_p, q=final_q))
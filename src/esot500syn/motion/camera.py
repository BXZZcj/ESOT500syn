import numpy as np
import sapien
import transforms3d.euler as t3d_euler
import transforms3d.quaternions as t3d_quat
import transforms3d.axangles as t3d_axangles
from typing import Dict, Callable

CAMERA_MOTION_PATTERNS: Dict[str, Callable] = {}

def register_camera_motion(name: str):
    def decorator(func: Callable):
        CAMERA_MOTION_PATTERNS[name] = func
        return func
    return decorator

@register_camera_motion("static")
def camera_motion_static(step, env, sensor, initial_pose, cfg):
    return initial_pose

@register_camera_motion("pan")
def camera_motion_pan(step, env, sensor, initial_pose, cfg):
    params = cfg.get("params", {})
    amplitude = params.get("amplitude", np.pi / 12)
    speed = params.get("speed", 0.02)
    up_axis = initial_pose.to_transformation_matrix()[:3, 2]
    angle = amplitude * np.sin(step * speed)
    
    delta_q = t3d_quat.axangle2quat(up_axis, angle, is_normalized=True)
    new_q = t3d_quat.qmult(initial_pose.q, delta_q)
    return sapien.Pose(p=initial_pose.p, q=new_q)

@register_camera_motion("tilt")
def camera_motion_tilt(step, env, sensor, initial_pose, cfg):
    params = cfg.get("params", {})
    amplitude = params.get("amplitude", np.pi / 18)
    speed = params.get("speed", 0.03)
    right_axis = initial_pose.to_transformation_matrix()[:3, 1]
    angle = amplitude * np.sin(step * speed)

    delta_q = t3d_quat.axangle2quat(right_axis, angle, is_normalized=True)
    new_q = t3d_quat.qmult(initial_pose.q, delta_q)
    return sapien.Pose(p=initial_pose.p, q=new_q)

@register_camera_motion("dolly")
def camera_motion_dolly(step, env, sensor, initial_pose, cfg):
    params = cfg.get("params", {})
    amplitude = params.get("amplitude", 0.2)
    speed = params.get("speed", 0.02)
    forward_axis = initial_pose.to_transformation_matrix()[:3, 0]
    offset = forward_axis * amplitude * np.sin(step * speed)
    new_p = np.array(initial_pose.p) + offset
    return sapien.Pose(p=new_p, q=initial_pose.q)

@register_camera_motion("breathing")
def camera_motion_breathing(step, env, sensor, initial_pose, cfg):
    params = cfg.get("params", {})
    pos_amp, rot_amp, speed = params.get("pos_amp", 0.005), params.get("rot_amp", 0.002), params.get("speed", 0.04)
    pos_offset = np.array([0, 0, pos_amp * np.sin(step * speed)])
    rot_offset_q = t3d_euler.euler2quat(rot_amp * np.sin(step * speed * 0.7), 0, 0, 'sxyz')
    new_p = np.array(initial_pose.p) + pos_offset
    new_q = t3d_quat.qmult(initial_pose.q, rot_offset_q)
    return sapien.Pose(p=new_p, q=new_q)

@register_camera_motion("gentle_drift")
def camera_motion_gentle_drift(step, env, sensor, initial_pose, cfg):
    if not hasattr(sensor, '_gentle_drift_state'):
        rng = env.unwrapped._episode_rng
        sensor._gentle_drift_state = {
            "pos_velocity": rng.randn(3), "rot_velocity": rng.randn(3),
            "current_pos_offset": np.zeros(3), "current_rot_offset_euler": np.zeros(3)
        }
    
    state = sensor._gentle_drift_state
    params = cfg.get("params", {})
    max_p, max_r, turn_rate = params.get("max_pos_offset", 0.03), params.get("max_rot_offset", 0.01), params.get("turn_rate", 0.01)
    rng = env.unwrapped._episode_rng

    state["pos_velocity"] = (state["pos_velocity"] + rng.randn(3) * turn_rate)
    state["pos_velocity"] /= np.linalg.norm(state["pos_velocity"])
    state["rot_velocity"] = (state["rot_velocity"] + rng.randn(3) * turn_rate)
    state["rot_velocity"] /= np.linalg.norm(state["rot_velocity"])
    
    state["current_pos_offset"] += state["pos_velocity"] * 0.001
    state["current_rot_offset_euler"] += state["rot_velocity"] * 0.0005
    pos_offset = np.tanh(state["current_pos_offset"]) * max_p
    rot_offset_euler = np.tanh(state["current_rot_offset_euler"]) * max_r
    
    new_p = np.array(initial_pose.p) + pos_offset
    delta_q = t3d_euler.euler2quat(*rot_offset_euler, 'sxyz')
    new_q = t3d_quat.qmult(initial_pose.q, delta_q)
    return sapien.Pose(p=new_p, q=new_q)

def _apply_final_pose(step, env, sensor, initial_pose, cfg):
    motion_type = cfg.get("type", "static")
    motion_func = CAMERA_MOTION_PATTERNS.get(motion_type, camera_motion_static)
    final_pose = motion_func(step, env, sensor, initial_pose, cfg)
    if final_pose:
        sensor.camera.set_local_pose(final_pose)

@register_camera_motion("composite")
def camera_motion_composite(step, env, sensor, initial_pose, cfg):
    sub_patterns = cfg.get("params", [])
    if not sub_patterns:
        return initial_pose
    start_p_np = np.array(initial_pose.p)
    accumulated_p_offset = np.zeros(3)
    
    # Accumulate total rotation increment starting from the identity quaternion
    accumulated_q_delta = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
    for sub_cfg in sub_patterns:
        sub_motion_type = sub_cfg.get("type")
        if not sub_motion_type:
            continue
            
        sub_motion_func = CAMERA_MOTION_PATTERNS.get(sub_motion_type)
        if not sub_motion_func:
            print(f"Warning: Composite camera motion could not find sub-pattern '{sub_motion_type}'.")
            continue
            
        # Camera sub-patterns can directly use sub_cfg, as naming conventions are consistent
        sub_pose = sub_motion_func(step, env, sensor, initial_pose, sub_cfg)
        
        if sub_pose is None:
            continue
        # Calculate translation offset and accumulate
        p_offset = np.array(sub_pose.p) - start_p_np
        accumulated_p_offset += p_offset
        # Calculate rotation increment and accumulate
        q_delta = t3d_quat.qmult(np.array(sub_pose.q), t3d_quat.qinverse(initial_pose.q))
        accumulated_q_delta = t3d_quat.qmult(q_delta, accumulated_q_delta)
    # Apply accumulated translation offset to the start position
    final_p = start_p_np + accumulated_p_offset
    
    # Apply accumulated rotation increment to the start rotation
    final_q = t3d_quat.qmult(accumulated_q_delta, initial_pose.q)
    
    return sapien.Pose(p=final_p.tolist(), q=final_q.tolist())

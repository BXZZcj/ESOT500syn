import numpy as np
import sapien
from typing import Dict, Callable
import transforms3d.euler as t3d_euler
import transforms3d.quaternions as t3d_quat
import transforms3d.axangles as t3d_axangles

MOTION_PATTERNS: Dict[str, Callable] = {}

def register_motion_pattern(name: str):
    def decorator(func: Callable):
        MOTION_PATTERNS[name] = func
        return func
    return decorator

# ===================================================================
# I. Translational Patterns
# ===================================================================

@register_motion_pattern("static")
def motion_static(mixin, start_pose, config):
    """The object remains stationary."""
    pass

@register_motion_pattern("oscillate_along_axis")
def motion_oscillate_along_axis(mixin, start_pose, config):
    """
    Oscillate along an arbitrary axis.
    - axis: The direction vector of the oscillation axis [x, y, z].
    - amplitude: The amplitude of the oscillation (half-distance).
    - speed: The speed of the oscillation.
    """
    params = config.get("motion_params", {})
    axis = np.array(params.get("axis", [1, 0, 0]))
    amplitude = params.get("amplitude", 0.5)
    speed = params.get("speed", 0.1)
    start_p = np.array(start_pose.p)
    
    normalized_axis = axis / np.linalg.norm(axis)
    offset = normalized_axis * amplitude * np.sin(mixin._elapsed_steps[0].item() * speed)
    new_p = start_p + offset
    return sapien.Pose(p=new_p.tolist())

@register_motion_pattern("linear_patrol_with_stops")
def motion_linear_patrol_with_stops(mixin, start_pose, config):
    """
    Patrol between two points, with a noticeable pause at each endpoint.
    - target_p: The target point coordinates [x, y, z] for patrol.
    - move_duration: The number of time steps required to complete a one-way move.
    - pause_duration: The number of time steps for a pause at each endpoint.
    """
    params = config.get("motion_params", {})
    target_p = np.array(params.get("target_p", [1, 0, 1]))
    move_duration = params.get("move_duration", 100)
    pause_duration = params.get("pause_duration", 30)
    start_p = np.array(start_pose.p)

    cycle_duration = (move_duration + pause_duration) * 2
    time_in_cycle = mixin._elapsed_steps[0].item() % cycle_duration

    if 0 <= time_in_cycle < move_duration:
        progress = time_in_cycle / move_duration
    elif move_duration <= time_in_cycle < move_duration + pause_duration:
        progress = 1.0
    elif move_duration + pause_duration <= time_in_cycle < 2 * move_duration + pause_duration:
        progress = 1.0 - ((time_in_cycle - (move_duration + pause_duration)) / move_duration)
    else:
        progress = 0.0
    
    smooth_progress = progress * progress * (3 - 2 * progress)
    new_p = start_p + (target_p - start_p) * smooth_progress
    return sapien.Pose(p=new_p.tolist())

@register_motion_pattern("bouncing_z")
def motion_bouncing_z(mixin, start_pose, config):
    """
    Simulate bouncing motion along the Z-axis, bouncing off the virtual ground.
    - height: The maximum height of the bounce.
    - duration: The number of time steps required for a complete bounce (fall + bounce).
    """
    params = config.get("motion_params", {})
    height = params.get("height", 0.5)
    duration = params.get("duration", 60)
    start_p = start_pose.p
    time = mixin._elapsed_steps[0].item() * (np.pi / duration)
    new_z = start_p[2] + height * abs(np.sin(time))
    return sapien.Pose(p=[start_p[0], start_p[1], new_z])

@register_motion_pattern("lissajous_xy")
def motion_lissajous_xy(mixin, start_pose, config):
    """
    Perform Lissajous curve motion (generalization of figure-8) on the XY plane.
    - x_radius, y_radius: The amplitudes of the X and Y axes.
    - x_freq, y_freq: The frequencies of the X and Y axes.
    - speed: The overall motion speed.
    - phase_shift: The phase difference between the X and Y axes (in radians).
    """
    params = config.get("motion_params", {})
    x_radius, y_radius = params.get("x_radius", 0.5), params.get("y_radius", 0.5)
    x_freq, y_freq = params.get("x_freq", 1), params.get("y_freq", 2) # 默认8字形
    speed, phase_shift = params.get("speed", 0.1), params.get("phase_shift", np.pi / 2)
    start_p = start_pose.p

    time = mixin._elapsed_steps[0].item() * speed
    new_x = start_p[0] + x_radius * np.sin(x_freq * time + phase_shift)
    new_y = start_p[1] + y_radius * np.sin(y_freq * time)
    return sapien.Pose(p=[new_x, new_y, start_p[2]])

# ===================================================================
# II. Rotational Patterns
# ===================================================================

@register_motion_pattern("spin_in_place")
def motion_spin_in_place(mixin, start_pose, config):
    params = config.get("motion_params", {})
    axis = np.array(params.get("axis", [0, 0, 1]))
    speed = params.get("speed", 0.05)
    
    angle = mixin._elapsed_steps[0].item() * speed
    delta_q_wxyz = t3d_quat.axangle2quat(axis, angle, is_normalized=True)
    new_q = t3d_quat.qmult(start_pose.q, delta_q_wxyz)
    return sapien.Pose(p=start_pose.p, q=new_q)

@register_motion_pattern("nod_in_place")
def motion_nod_in_place(mixin, start_pose, config):
    params = config.get("motion_params", {})
    local_axis = np.array(params.get("axis", [1, 0, 0]))
    amplitude = params.get("amplitude", np.pi / 4)
    speed = params.get("speed", 0.05)

    angle = amplitude * np.sin(mixin._elapsed_steps[0].item() * speed)
    delta_q_local_wxyz = t3d_quat.axangle2quat(local_axis, angle, is_normalized=True)
    new_q = t3d_quat.qmult(start_pose.q, delta_q_local_wxyz)
    return sapien.Pose(p=start_pose.p, q=new_q)

@register_motion_pattern("tumble")
def motion_tumble(mixin, start_pose, config):
    params = config.get("motion_params", {})
    spin_speeds = params.get("spin_speeds", [0.05, 0.03, 0.08])
    time = mixin._elapsed_steps[0].item()

    qx = t3d_quat.axangle2quat([1, 0, 0], time * spin_speeds[0])
    qy = t3d_quat.axangle2quat([0, 1, 0], time * spin_speeds[1])
    qz = t3d_quat.axangle2quat([0, 0, 1], time * spin_speeds[2])
    delta_q = t3d_quat.qmult(t3d_quat.qmult(qx, qy), qz)
    new_q = t3d_quat.qmult(start_pose.q, delta_q)
    return sapien.Pose(p=start_pose.p, q=new_q)

# ===================================================================
# III. Compound Patterns - Basic
# ===================================================================

@register_motion_pattern("circular_xy")
def motion_circular_xy(mixin, start_pose, config):
    params, start_p = config.get("motion_params", {}), start_pose.p
    radius, speed = params.get("radius", 0.5), params.get("speed", 0.1)
    angle = mixin._elapsed_steps[0].item() * speed
    new_x, new_y = start_p[0] + radius * np.cos(angle), start_p[1] + radius * np.sin(angle)
    return sapien.Pose(p=[new_x, new_y, start_p[2]], q=start_pose.q)

@register_motion_pattern("swoop_and_dive")
def motion_swoop_and_dive(mixin, start_pose, config):
    params = config.get("motion_params", {})
    h_radius, v_amp, speed, start_p = params.get("horizontal_radius", 0.8), params.get("vertical_amplitude", 0.4), params.get("speed", 0.1), start_pose.p
    angle = mixin._elapsed_steps[0].item() * speed
    new_x, new_y = start_p[0] + h_radius * np.cos(angle), start_p[1] + h_radius * np.sin(angle)
    new_z = start_p[2] - v_amp * np.sin(2 * angle)
    return sapien.Pose(p=[new_x, new_y, new_z], q=start_pose.q)

def _calculate_look_along_path_orientation(p_curr, p_next, up_vector):
    if np.allclose(p_curr, p_next): return None
    forward = p_next - p_curr
    if np.linalg.norm(forward) < 1e-6: return None
    forward /= np.linalg.norm(forward)
    right = np.cross(up_vector, forward)
    if np.linalg.norm(right) < 1e-6:
        temp_up = np.array([1,0,0]) if not np.allclose(up_vector, [1,0,0]) else np.array([0,1,0])
        right = np.cross(temp_up, forward)
    right /= np.linalg.norm(right)
    new_up = np.cross(forward, right)
    rot_mat = np.eye(3)
    rot_mat[:, 0], rot_mat[:, 1], rot_mat[:, 2] = forward, -right, new_up
    return t3d_quat.mat2quat(rot_mat)

@register_motion_pattern("path_following_with_roll")
def motion_path_following_with_roll(mixin, start_pose, config):
    params = config.get("motion_params", {})
    x_radius, y_radius, speed = params.get("x_radius", 0.5), params.get("y_radius", 0.4), params.get("speed", 0.1)
    roll_speed = params.get("roll_speed", 0.0) 
    up_vector = np.array(params.get("up_vector", [0, 0, 1]))
    start_p = start_pose.p
    step = mixin._elapsed_steps[0].item()

    def get_pos_at_step(s):
        angle = s * speed
        x = start_p[0] + x_radius * np.sin(angle)
        y = start_p[1] + y_radius * np.sin(2 * angle)
        z = start_p[2]
        return np.array([x, y, z])

    p_current = get_pos_at_step(step)
    p_next = get_pos_at_step(step + 1) 

    q_look = _calculate_look_along_path_orientation(p_current, p_next, up_vector)
    if q_look is None:
        q_look = start_pose.q

    if abs(roll_speed) > 1e-6:
        roll_angle = step * roll_speed
        q_roll = t3d_quat.axangle2quat([1, 0, 0], roll_angle)
        final_q = t3d_quat.qmult(q_look, q_roll)
    else:
        final_q = q_look
    return sapien.Pose(p=p_current.tolist(), q=final_q)

# ===================================================================
# IV. Stochastic Patterns
# ===================================================================

@register_motion_pattern("randomized_circular_xy")
def motion_randomized_circular_xy(mixin, start_pose, config):
    if not hasattr(mixin, '_randomized_circular_params'):
        rng, params = mixin.unwrapped._episode_rng, config.get("motion_params", {})
        radius_range, speed_range = params.get("radius_range", [0.3, 0.8]), params.get("speed_range", [0.05, 0.15])
        mixin._randomized_circular_params = {"radius": rng.uniform(radius_range[0], radius_range[1]), "speed": rng.uniform(speed_range[0], speed_range[1])}
    rand_params, start_p = mixin._randomized_circular_params, start_pose.p
    radius, speed = rand_params["radius"], rand_params["speed"]
    angle = mixin._elapsed_steps[0].item() * speed
    new_x, new_y = start_p[0] + radius * np.cos(angle), start_p[1] + radius * np.sin(angle)
    return sapien.Pose(p=[new_x, new_y, start_p[2]], q=start_pose.q)

@register_motion_pattern("random_goal_patrol")
def motion_random_goal_patrol(mixin, start_pose, config):
    if not hasattr(mixin, '_random_goal_params'):
        rng, params = mixin.unwrapped._episode_rng, config.get("motion_params", {})
        half_size, duration_range = np.array(params.get("target_box_half_size", [0.5, 0.5, 0.2])), params.get("duration_range", [80, 150])
        random_offset = rng.uniform(-half_size, half_size)
        mixin._random_goal_params = {"target_p": np.array(start_pose.p) + random_offset, "duration": rng.randint(duration_range[0], duration_range[1])}
    rand_params, start_p = mixin._random_goal_params, np.array(start_pose.p)
    target_p, duration = rand_params["target_p"], rand_params["duration"]
    progress = (1 - np.cos(np.pi * mixin._elapsed_steps[0].item() / duration)) / 2
    new_p = start_p + (target_p - start_p) * progress
    return sapien.Pose(p=new_p.tolist())

@register_motion_pattern("random_walk_xy")
def motion_random_walk_xy(mixin, start_pose, config):
    if not hasattr(mixin, '_random_walk_state'):
        rng = mixin.unwrapped._episode_rng
        mixin._random_walk_state = {"current_pos": np.array(start_pose.p), "current_angle": rng.uniform(0, 2 * np.pi)}
    state, params, rng = mixin._random_walk_state, config.get("motion_params", {}), mixin.unwrapped._episode_rng
    speed, turn_rate = params.get("speed", 0.01), params.get("turn_rate", 0.1)
    state["current_angle"] += rng.uniform(-turn_rate, turn_rate)
    dx, dy = speed * np.cos(state["current_angle"]), speed * np.sin(state["current_angle"])
    state["current_pos"] += np.array([dx, dy, 0])
    return sapien.Pose(p=state["current_pos"].tolist())

@register_motion_pattern("drunkards_walk_3d")
def motion_drunkards_walk_3d(mixin, start_pose, config):
    if not hasattr(mixin, '_drunkards_walk_state'):
        rng = mixin.unwrapped._episode_rng
        mixin._drunkards_walk_state = {
            "current_pose": start_pose, "pos_velocity": rng.randn(3), "rot_velocity": rng.randn(3)
        }
    state, params, rng = mixin._drunkards_walk_state, config.get("motion_params", {}), mixin.unwrapped._episode_rng
    p_speed, p_turn, r_speed, r_turn = params.get("pos_speed", 0.01), params.get("pos_turn_rate", 0.1), params.get("rot_speed", 0.05), params.get("rot_turn_rate", 0.2)
    
    state["pos_velocity"] = (state["pos_velocity"] + rng.randn(3) * p_turn)
    state["pos_velocity"] /= np.linalg.norm(state["pos_velocity"])
    new_p = state["current_pose"].p + state["pos_velocity"] * p_speed
    
    state["rot_velocity"] = (state["rot_velocity"] + rng.randn(3) * r_turn)
    state["rot_velocity"] /= np.linalg.norm(state["rot_velocity"])
    angle = rng.uniform(0, r_speed)
    
    delta_q = t3d_quat.axangle2quat(state["rot_velocity"], angle, is_normalized=True)
    new_q = t3d_quat.qmult(state["current_pose"].q, delta_q)
    state["current_pose"] = sapien.Pose(p=new_p, q=new_q)
    return state["current_pose"]

# ===================================================================
# V. Meta Patterns - Used to combine other patterns
# ===================================================================

@register_motion_pattern("composite")
def motion_composite(mixin, start_pose, config):
    sub_patterns = config.get("motion_params", [])
    if not sub_patterns: return None
    start_p_np, accumulated_p_offset, accumulated_q_delta = np.array(start_pose.p), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    start_q_inv = start_pose.inv().q
    for sub_config in sub_patterns:
        sub_motion_mode = sub_config.get("motion_mode")
        if not sub_motion_mode: continue
        sub_motion_func = MOTION_PATTERNS.get(sub_motion_mode)
        if not sub_motion_func:
            print(f"Warning: Composite motion could not find sub-pattern '{sub_motion_mode}'. Skipping.")
            continue
        new_pose = sub_motion_func(mixin, start_pose, sub_config)
        if new_pose is None: continue
        p_offset = np.array(new_pose.p) - start_p_np
        accumulated_p_offset += p_offset
        q_delta = t3d_quat.qmult(np.array(new_pose.q), start_q_inv)
        accumulated_q_delta = t3d_quat.qmult(q_delta, accumulated_q_delta)
    final_p = start_p_np + accumulated_p_offset
    final_q = t3d_quat.qmult(accumulated_q_delta, start_pose.q)
    return sapien.Pose(p=final_p.tolist(), q=final_q.tolist())
# src/esot500syn/processing/annotations.py
import os
import numpy as np
import trimesh
import cv2
import transforms3d.quaternions as t3d_quat

def load_mesh_vertices_faces(mesh_path: str):
    mesh_path = os.path.expanduser(mesh_path)
    try:
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32)
    except Exception as e:
        print(f"Error loading mesh {mesh_path}: {e}")
        return None, None

def rasterize_amodal_mask(vertices_obj, faces, t_cam_robot, q_wxyz_robot, fx, fy, cx, cy, width, height):
    R_robot_to_cv = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    R_robot = t3d_quat.quat2mat(q_wxyz_robot.astype(np.float64))
    verts_robot_cam = (R_robot @ vertices_obj.T).T + t_cam_robot.reshape(3)
    verts_cv_cam = (R_robot_to_cv @ verts_robot_cam.T).T
    mask = np.zeros((height, width), dtype=np.uint8)
    for tri in faces:
        # <<<--- 修正：将错误的一行代码拆分为正确的两行 ---
        pts = verts_cv_cam[tri]
        Z = pts[:, 2]
        # --- 修正结束 ---

        if np.any(Z <= 1e-6):
            continue
        u, v = fx * (pts[:, 0] / Z) + cx, fy * (pts[:, 1] / Z) + cy
        poly = np.stack([u, v], axis=1)
        if np.any(~np.isfinite(poly)):
            continue
        cv2.fillConvexPoly(mask, np.round(poly).astype(np.int32), 255)
    return mask

def bbox_from_mask(mask: np.ndarray):
    rows, cols = np.any(mask > 0, axis=1), np.any(mask > 0, axis=0)
    if not (np.any(rows) and np.any(cols)):
        return None
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]
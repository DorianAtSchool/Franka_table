import numpy as np
import mujoco


def jacobian_ik6_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    robot_idx: int,
    v_des: np.ndarray,
    w_des: np.ndarray,
    dt: float,
    damping: float = 0.1,
) -> np.ndarray:
    """Compute joint velocity step dq for desired 6D twist.

    v_des: linear velocity (3,) in world frame
    w_des: angular velocity (3,) in world frame
    """
    nv = model.nv
    jacp = np.zeros((3, nv), dtype=np.float64)
    jacr = np.zeros((3, nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    qvel_start = 6 + robot_idx * 9
    Jp = jacp[:, qvel_start : qvel_start + 7]
    Jr = jacr[:, qvel_start : qvel_start + 7]
    J = np.vstack([Jp, Jr])  # 6x7
    v6 = np.concatenate([v_des.astype(np.float64), w_des.astype(np.float64)])
    JJt = J @ J.T
    lam2I = (damping**2) * np.eye(6)
    dq = J.T @ np.linalg.solve(JJt + lam2I, v6)
    return dq * dt


def yaw_to_face_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    target_pos: np.ndarray,
) -> float:
    """Compute a small yaw rotation so the site's x-axis faces the target in XY.

    Returns dyaw (radians), wrapped to [-pi, pi]. If the XY distance to target
    is tiny, returns 0.0.
    """
    ee_pos = data.site_xpos[site_id]
    vec = np.asarray(target_pos, dtype=np.float64) - ee_pos
    vec_xy = vec[:2]
    if np.linalg.norm(vec_xy) < 1e-6:
        return 0.0
    desired_yaw = float(np.arctan2(vec_xy[1], vec_xy[0]))

    # Current yaw of site's x-axis projected to XY plane
    mat = data.site_xmat[site_id].reshape(3, 3)
    x_axis = mat[:, 0]
    x_xy = x_axis[:2]
    if np.linalg.norm(x_xy) < 1e-6:
        return 0.0
    current_yaw = float(np.arctan2(x_xy[1], x_xy[0]))

    dyaw = desired_yaw - current_yaw
    # Wrap to [-pi, pi]
    dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
    return float(dyaw)

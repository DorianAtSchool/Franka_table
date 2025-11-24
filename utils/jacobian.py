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


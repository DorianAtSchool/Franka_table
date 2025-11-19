import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pyarrow.parquet as pq

from franka_table.environments.franka_4robots_env import FrankaTable4RobotsEnv
from franka_table.datasets.lerobot_writer import LeRobotDatasetWriter
import mujoco


def jacobian_ik_step(model: mujoco.MjModel, data: mujoco.MjData, site_id: int, robot_idx: int, dpos: np.ndarray, dt: float, damping: float = 0.1) -> np.ndarray:
    nv = model.nv
    jacp = np.zeros((3, nv), dtype=np.float64)
    jacr = np.zeros((3, nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    qvel_start = 6 + robot_idx * 9
    J = jacp[:, qvel_start:qvel_start + 7]  # 3x7
    JJt = J @ J.T
    lam2I = (damping ** 2) * np.eye(3)
    inv = np.linalg.inv(JJt + lam2I)
    dq = J.T @ (inv @ (dpos / max(dt, 1e-6)))
    return dq * dt


def randomize_visuals(env: FrankaTable4RobotsEnv, rng: np.random.RandomState) -> None:
    # Randomize target object color only; avoid global ambient changes (no 'xray' effects)
    geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
    if geom_id >= 0:
        rgba = rng.rand(4)
        rgba[3] = 1.0
        env.model.geom_rgba[geom_id] = rgba
    # Slight camera jitter for side camera to diversify framing
    side_cam = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "side")
    if side_cam >= 0:
        env.model.cam_pos[side_cam] += rng.randn(3) * 0.01


def main():
    parser = argparse.ArgumentParser(description="Replay a recorded episode with domain randomization to generate synthetic variants.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--source_episode", type=int, required=True, help="episode_index of source demo to replay")
    parser.add_argument("--copies", type=int, default=5)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--cameras", type=str, nargs="+", default=["robot1_wrist", "side"])
    args = parser.parse_args()

    pkg_root = Path(__file__).resolve().parents[1]
    root = Path(args.dataset_root)
    if not root.is_absolute():
        root = pkg_root / root
    episodes_file = root / "episodes.jsonl"
    meta_tasks = root / "meta" / "tasks.jsonl"

    # Load episodes
    episodes: List[dict] = []
    with open(episodes_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))

    src = next((ep for ep in episodes if ep.get("episode_index") == args.source_episode), None)
    if src is None:
        raise ValueError(f"Source episode_index {args.source_episode} not found")

    # Load source parquet
    parquet_rel = src["parquet"]
    df = pq.read_table(root / parquet_rel).to_pandas()

    # Resolve task text for the episode (first task_index)
    task_index = src.get("tasks", [0])[0]
    task_text = None
    with open(meta_tasks, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec.get("task_index") == task_index:
                    task_text = rec.get("task")
                    break
    if task_text is None:
        task_text = "replay"

    # Setup env
    env_scene = Path(__file__).resolve().parents[1] / "scenes" / "scene_4robots.xml"
    env = FrankaTable4RobotsEnv(mjcf_path=str(env_scene), render_mode="rgb_array")

    writer = LeRobotDatasetWriter(root=root, fps=args.fps, cameras=args.cameras)

    rng = np.random.RandomState(42)
    for k in range(args.copies):
        obs, info = env.reset()
        randomize_visuals(env, rng)
        episode = writer.start_episode(task_text=task_text)

        site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "robot1_gripper_center")
        action = np.zeros(env.n_actuators, dtype=np.float32)
        action[:7] = env.data.qpos[7:14].copy()
        action[7] = 255.0

        for t in range(len(df)):
            row = df.iloc[t]
            # Source EE delta action [dx, dy, dz, ...]
            a = row["action"] if isinstance(row["action"], list) else list(row["action"])  # type: ignore
            dpos = np.array(a[:3], dtype=np.float64)
            # Add small noise to vary trajectory
            dpos += rng.randn(3) * 0.001
            dq = jacobian_ik_step(env.model, env.data, site_id, robot_idx=0, dpos=dpos, dt=env.control_dt, damping=0.1)
            action[:7] = action[:7] + dq.astype(np.float32)
            # Gripper from sign of 7th element
            grip = 255.0 if float(a[6]) > 0 else 0.0
            action[7] = grip

            obs, reward, terminated, truncated, i2 = env.step(action)

            # Observation state similar to recorder
            qpos = env.data.qpos[7:14].copy()
            qvel = env.data.qvel[6:13].copy()
            grip_l = env.data.qpos[14]
            grip_r = env.data.qpos[15]
            grip_state = np.array([grip_l, grip_r], dtype=np.float32)
            obs_state = np.concatenate([qpos, qvel, grip_state]).astype(np.float32)

            images = env.render_cameras(args.cameras, width=args.width, height=args.height)
            done = bool(terminated or truncated or (t == len(df) - 1))
            episode.add_frame(
                observation_state=obs_state,
                action=np.array(a, dtype=np.float32),
                timestamp=t * env.control_dt,
                done=done,
                images=images,
                frame_index=t,
                frames_dir=writer.frames_dir,
                episode_prefix=f"episode_{episode.episode_index:06d}",
            )
            if done:
                break

        writer.end_episode(episode, length=len(episode.frames), task_text=task_text)

    writer.finalize()
    env.close()
    print(f"Generated {args.copies} randomized replays for episode {args.source_episode}.")


if __name__ == "__main__":
    main()

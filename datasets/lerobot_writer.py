import os
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import imageio.v2 as imageio
import mujoco


@dataclass
class EpisodeBuffer:
    task_index: int
    episode_index: int
    fps: int
    frames: List[Dict[str, Any]] = field(default_factory=list)
    # For future use (e.g., video shards), if needed
    camera_frame_paths: Dict[str, List[str]] = field(default_factory=dict)

    def add_frame(
        self,
        observation_state: np.ndarray,
        action: np.ndarray,
        timestamp: float,
        done: bool,
        images: Dict[str, np.ndarray],
        frame_index: int,
        frames_dir: Path,
        episode_prefix: str,
    ) -> None:
        row: Dict[str, Any] = {
            "observation.state": observation_state.astype(np.float32).tolist(),
            "action": action.astype(np.float32).tolist(),
            "timestamp": float(timestamp),
            "episode_index": int(self.episode_index),
            "frame_index": int(frame_index),
            "index": int(frame_index),  # unique within-episode; global ID can be built downstream
            "next.done": bool(done),
            "task_index": int(self.task_index),
        }

        self.frames.append(row)

    def to_parquet(self, out_parquet: Path) -> None:
        if not self.frames:
            raise ValueError("No frames to write for this episode.")
        df = pd.DataFrame(self.frames)
        table = pa.Table.from_pandas(df, preserve_index=False)
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_parquet)


class LeRobotDatasetWriter:
    """
    Minimal LeRobot-like dataset writer.

    Structure:
    - root/
      - meta/info.json
      - meta/tasks.jsonl
      - episodes.jsonl
      - data/chunk-000/episode_000000.parquet
      - frames/<camera>/<episode_prefix>/frame_XXXXXX.png

    Parquet columns (per frame):
      observation.state, action, timestamp, episode_index, frame_index, index,
      next.done, task_index, observation.images.<camera>
    """

    def __init__(
        self,
        root: Path,
        dataset_name: str = "franka_table_synth",
        robot_type: str = "FrankaPanda",
        fps: int = 25,
        cameras: Optional[List[str]] = None,
    ) -> None:
        self.root = Path(root)
        self.meta_dir = self.root / "meta"
        # Frame-by-frame data (Parquet shards)
        self.data_root = self.root / "data"
        self.data_dir = self.data_root / "chunk-000"
        # Per-episode metadata (Parquet shards)
        self.episodes_root = self.meta_dir / "episodes"
        self.episodes_chunk_dir = self.episodes_root / "chunk-000"
        # Video shards root: videos/{camera}/chunk-xxx/file-yyy.mp4
        self.videos_root = self.root / "videos"
        # Legacy attribute used by callers; kept for compatibility but not used to store PNG frames.
        self.frames_dir = self.root / "frames"
        # Optional JSONL index of episodes (for convenience)
        self.episodes_file = self.root / "episodes.jsonl"
        # Tasks metadata as parquet
        self.tasks_file = self.meta_dir / "tasks.parquet"
        self.info_file = self.meta_dir / "info.json"
        self.dataset_name = dataset_name
        self.robot_type = robot_type
        self.fps = int(fps)
        self.cameras = cameras or []

        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_chunk_dir.mkdir(parents=True, exist_ok=True)
        self.videos_root.mkdir(parents=True, exist_ok=True)

        self._tasks: Dict[str, int] = {}
        self._episodes_meta: List[Dict[str, Any]] = []
        self._episode_count: int = 0
        self._next_task_index: int = 0

        # Load existing tasks and episode index if dataset already exists
        if self.tasks_file.exists():
            try:
                tasks_df = pd.read_parquet(self.tasks_file)
                for _, rec in tasks_df.iterrows():
                    task = rec.get("task")
                    idx = int(rec.get("task_index", 0))
                    if task is not None:
                        self._tasks[task] = idx
                if self._tasks:
                    self._next_task_index = max(self._tasks.values()) + 1
            except Exception:
                pass

        if self.episodes_file.exists():
            try:
                max_idx = -1
                with open(self.episodes_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            rec = json.loads(line)
                            if "episode_index" in rec:
                                max_idx = max(max_idx, int(rec["episode_index"]))
                if max_idx >= 0:
                    self._episode_count = max_idx + 1
            except Exception:
                pass

    def _get_task_index(self, task_text: str) -> int:
        if task_text not in self._tasks:
            self._tasks[task_text] = self._next_task_index
            self._next_task_index += 1
        return self._tasks[task_text]

    def start_episode(self, task_text: str) -> EpisodeBuffer:
        task_idx = self._get_task_index(task_text)
        ep_idx = self._episode_count
        self._episode_count += 1
        return EpisodeBuffer(task_index=task_idx, episode_index=ep_idx, fps=self.fps)

    def end_episode(self, episode: EpisodeBuffer, length: int, task_text: str) -> None:
        ep_idx = episode.episode_index
        ep_prefix = f"episode_{ep_idx:06d}"
        parquet_path = self.data_dir / f"{ep_prefix}.parquet"
        episode.to_parquet(parquet_path)

        # Record episode metadata
        ep_meta = {
            "episode_index": ep_idx,
            "length": length,
            "tasks": [episode.task_index],
            "parquet": os.path.relpath(parquet_path, self.root).replace(os.sep, "/"),
            "cameras": self.cameras,
        }
        self._episodes_meta.append(ep_meta)

    def finalize(self) -> None:
        # Merge tasks (preserve existing indices; assign new indices after max existing)
        existing_tasks: Dict[str, int] = {}
        max_idx = -1
        if self.tasks_file.exists():
            try:
                tasks_df = pd.read_parquet(self.tasks_file)
                for _, rec in tasks_df.iterrows():
                    t = rec.get("task")
                    idx = int(rec.get("task_index", 0))
                    if t is not None:
                        existing_tasks[t] = idx
                        max_idx = max(max_idx, idx)
            except Exception:
                pass
        merged = dict(existing_tasks)
        next_idx = max_idx + 1
        for t, idx in self._tasks.items():
            if t not in merged:
                merged[t] = next_idx
                next_idx += 1
        # Write tasks as parquet sorted by index
        tasks_records = [{"task_index": idx, "task": t} for t, idx in sorted(merged.items(), key=lambda x: x[1])]
        if tasks_records:
            tasks_df_out = pd.DataFrame(tasks_records)
            tasks_df_out.to_parquet(self.tasks_file, index=False)

        # Append episodes.jsonl with new episodes (optional convenience index)
        with open(self.episodes_file, "a", encoding="utf-8") as f:
            for ep in self._episodes_meta:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")

        # Compute global counts and feature shapes
        all_eps: List[dict] = []
        try:
            with open(self.episodes_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_eps.append(json.loads(line))
        except Exception:
            pass

        n_episodes = len(all_eps)
        n_frames = sum(int(ep.get("length", 0)) for ep in all_eps)
        obs_state_dim = None
        action_dim = None
        # Try infer from the most recent parquet written
        last_parquet = self.root / self._episodes_meta[-1]["parquet"] if self._episodes_meta else None
        if last_parquet and last_parquet.exists():
            table = pq.read_table(last_parquet)
            df = table.to_pandas()
            if len(df) > 0:
                if "observation.state" in df.columns:
                    obs_state_dim = len(df.iloc[0]["observation.state"])  # type: ignore
                if "action" in df.columns:
                    action_dim = len(df.iloc[0]["action"])  # type: ignore

        # Write per-episode metadata as Parquet under meta/episodes/chunk-000/
        if all_eps:
            episodes_df = pd.DataFrame(all_eps)
            episodes_table = pa.Table.from_pandas(episodes_df, preserve_index=False)
            episodes_parquet_path = self.episodes_chunk_dir / "file-000.parquet"
            pq.write_table(episodes_table, episodes_parquet_path)

        # Compute simple global stats for state and actions
        stats: Dict[str, Dict[str, Any]] = {}
        all_states: List[np.ndarray] = []
        all_actions: List[np.ndarray] = []
        for ep in all_eps:
            rel_pq = ep.get("parquet")
            if not rel_pq:
                continue
            pq_path = self.root / rel_pq
            if not pq_path.exists():
                continue
            table = pq.read_table(pq_path)
            df = table.to_pandas()
            if "observation.state" in df.columns and len(df) > 0:
                states_ep = np.asarray(df["observation.state"].to_list(), dtype=np.float32)
                all_states.append(states_ep)
            if "action" in df.columns and len(df) > 0:
                actions_ep = np.asarray(df["action"].to_list(), dtype=np.float32)
                all_actions.append(actions_ep)

        if all_states:
            states_arr = np.concatenate(all_states, axis=0)
            stats["state"] = {
                "mean": states_arr.mean(axis=0).tolist(),
                "std": states_arr.std(axis=0).tolist(),
                "min": states_arr.min(axis=0).tolist(),
                "max": states_arr.max(axis=0).tolist(),
            }
        if all_actions:
            actions_arr = np.concatenate(all_actions, axis=0)
            stats["actions"] = {
                "mean": actions_arr.mean(axis=0).tolist(),
                "std": actions_arr.std(axis=0).tolist(),
                "min": actions_arr.min(axis=0).tolist(),
                "max": actions_arr.max(axis=0).tolist(),
            }

        # Write stats.json if we computed anything
        if stats:
            stats_file = self.meta_dir / "stats.json"
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)

        # Assume images are 720x1280x3 as configured in renderers
        image_shape = [720, 1280, 3]

        info = {
            "codebase_version": "v2.0",
            "name": self.dataset_name,
            "robot_type": self.robot_type,
            "fps": self.fps,
            "total_episodes": n_episodes,
            "total_frames": n_frames,
            "total_tasks": len(merged),
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": n_episodes,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/{camera}/chunk-{episode_chunk:03d}/file-{file_index:03d}.mp4",
            "features": {
                "image": {
                    "dtype": "image",
                    "shape": image_shape,
                    "names": ["height", "width", "channel"],
                },
                "second_image": {
                    "dtype": "image",
                    "shape": image_shape,
                    "names": ["height", "width", "channel"],
                },
                "wrist_image": {
                    "dtype": "image",
                    "shape": image_shape,
                    "names": ["height", "width", "channel"],
                },
                "state": {
                    "dtype": "float32",
                    "shape": [obs_state_dim] if obs_state_dim is not None else None,
                    "names": ["state"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": [action_dim] if action_dim is not None else None,
                    "names": ["actions"],
                },
                "timestamp": {
                    "dtype": "float32",
                    "shape": [1],
                    "names": None,
                },
                "frame_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None,
                },
                "episode_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None,
                },
                "index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None,
                },
                "task_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None,
                },
            },
            "cameras": self.cameras,
            "format_version": "v2.1-simplified",
        }
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        with open(self.info_file, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)


def _quat_to_rpy(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float, float]:
    """Convert quaternion (w, x, y, z) to roll, pitch, yaw (XYZ convention)."""
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def build_dataset_from_sweep_npz(
    npz_episodes_dir: str | Path,
    out_root: str | Path,
    mj_model_path: str | Path,
    ee_site: str = "robot1_gripper_center",
    finger_joint1: str = "robot1_finger_joint1",
    finger_joint2: str = "robot1_finger_joint2",
    task_text: str = "Pick up and lift object",
    fps: int = 25,
) -> None:
    """
    Turn NPZ episodes created by the synthetic sweep/variants into a dataset where:

      state  = [x, y, z, qx, qy, qz, qw]  (end-effector pose)
      action = [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, Δgripper]

    All quantities are computed in the MuJoCo world frame using the specified
    end-effector site and gripper joints.
    """
    npz_episodes_dir = Path(npz_episodes_dir)
    out_root = Path(out_root)

    writer = LeRobotDatasetWriter(
        root=out_root,
        dataset_name="franka_table_synth",
        robot_type="FrankaPanda",
        fps=fps,
        cameras=[],  # state-only; no images by default
    )

    # Load MuJoCo model to compute forward kinematics for the end-effector
    model = mujoco.MjModel.from_xml_path(str(mj_model_path))
    data = mujoco.MjData(model)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
    j1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, finger_joint1)
    j2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, finger_joint2)
    j1_qadr = model.jnt_qposadr[j1_id]
    j2_qadr = model.jnt_qposadr[j2_id]

    npz_paths = sorted(npz_episodes_dir.glob("*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No .npz episodes found in {npz_episodes_dir}")

    for npz_path in npz_paths:
        data_npz = np.load(npz_path)

        timestamps = data_npz["timestamp"]
        qpos = data_npz["qpos"]

        T = len(timestamps)
        episode = writer.start_episode(task_text=task_text)
        ep_idx = episode.episode_index
        ep_prefix = f"episode_{ep_idx:06d}"

        prev_state = None
        prev_grip = None

        for t in range(T):
            q = qpos[t]
            data.qpos[:] = q
            mujoco.mj_forward(model, data)

            pos = data.site_xpos[site_id].copy()  # (3,)
            quat_wxyz = data.site_xquat[site_id].copy()  # (4,) w,x,y,z
            qw, qx, qy, qz = quat_wxyz
            # State as [x, y, z, qx, qy, qz, qw]
            state = np.array(
                [pos[0], pos[1], pos[2], qx, qy, qz, qw], dtype=np.float32
            )

            # Gripper opening as average of two finger joints
            grip = float(0.5 * (q[j1_qadr] + q[j2_qadr]))

            if prev_state is None:
                action = np.zeros(7, dtype=np.float32)
            else:
                dx = state[0] - prev_state[0]
                dy = state[1] - prev_state[1]
                dz = state[2] - prev_state[2]

                roll, pitch, yaw = _quat_to_rpy(qw, qx, qy, qz)
                prev_qw = prev_state[6]
                prev_qx = prev_state[3]
                prev_qy = prev_state[4]
                prev_qz = prev_state[5]
                prev_roll, prev_pitch, prev_yaw = _quat_to_rpy(
                    prev_qw, prev_qx, prev_qy, prev_qz
                )

                droll = roll - prev_roll
                dpitch = pitch - prev_pitch
                dyaw = yaw - prev_yaw

                dgrip = grip - prev_grip

                action = np.array(
                    [dx, dy, dz, droll, dpitch, dyaw, dgrip], dtype=np.float32
                )

            prev_state = state
            prev_grip = grip

            done = bool(t == T - 1)
            episode.add_frame(
                observation_state=state,
                action=action,
                timestamp=float(timestamps[t]),
                done=done,
                images={},  # no cameras recorded in NPZ
                frame_index=int(t),
                frames_dir=writer.frames_dir,
                episode_prefix=ep_prefix,
            )

        writer.end_episode(episode, length=T, task_text=task_text)

    writer.finalize()

"""
Robot 1 Pick-up Trajectory Sweep

This script runs many variations of the Robot 1 pick-up demo
by perturbing the target joint configuration and recording:
  - videos for each camera view (side, top, wrist), and
  - a LeRobot-style dataset directly (no npz intermediate).

All three camera views are rendered from a **single** MuJoCo
rollout per trial for efficiency.

LeRobot dataset root (default):
    franka_table/datasets/franka_table_synth

Videos and a CSV summary are saved under:
    franka_table/videos/robot1_pickup_sweeps/run_YYYYMMDD_HHMMSS/
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import mediapy as media
import mujoco
import numpy as np

from demos.robot1_pickup_demo import create_pickup_trajectory
from datasets.lerobot_writer import LeRobotDatasetWriter


# Base joint configuration used in the original demo
BASE_TARGET_JOINTS = np.array(
    [1.2999, 1.5000, 0.0499, -0.5208, 1.6501, 1.8710, -1.2852], dtype=float
)


def find_scene_path() -> str:
    """Locate a 4-robot scene XML, trying several known locations."""
    candidates = [
        os.path.join("scenes", "scene_4robots.xml"),
        os.path.join("scenes", "scene_4robots_real.xml"),
        os.path.join(
            "scenes", "franka_emika_panda", "scene_4pandas_table.xml"
        ),
    ]

    for path in candidates:
        if os.path.exists(path):
            print(f"Using scene file: {path}")
            return path

    raise FileNotFoundError(
        "Could not find a 4-robot scene XML. "
        "Looked for:\n  - " + "\n  - ".join(candidates)
    )


def init_scene_state(model: mujoco.MjModel) -> mujoco.MjData:
    """Initialize object and all four robots to the same state as the demo."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Object at center of table
    data.qpos[0:3] = [0.0, 0.0, 0.535]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    # Initialize all robots to home position (same as robot1_pickup_demo)
    for i in range(4):
        start_idx = 7 + i * 9
        data.qpos[start_idx:start_idx + 7] = [
            0.0,
            0.0,
            0.0,
            -1.57079,
            0.0,
            1.57079,
            -0.7853,
        ]
        data.qpos[start_idx + 7:start_idx + 9] = [0.04, 0.04]

        ctrl_start = i * 8
        data.ctrl[ctrl_start:ctrl_start + 7] = data.qpos[start_idx:start_idx + 7].copy()
        data.ctrl[ctrl_start + 7] = 255  # open gripper

    mujoco.mj_forward(model, data)
    return data


def generate_target_sets(
    num_trials: int, noise_std: float = 0.05, seed: int = 0
) -> List[np.ndarray]:
    """
    Generate a list of target joint configurations.

    Each configuration is BASE_TARGET_JOINTS plus small Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    targets: List[np.ndarray] = []
    for _ in range(num_trials):
        noise = rng.normal(0.0, noise_std, size=BASE_TARGET_JOINTS.shape)
        target = BASE_TARGET_JOINTS + noise
        # Conservative clipping to keep joints well within limits
        target = np.clip(target, -2.5, 2.5)
        targets.append(target.astype(float))
    return targets


def evaluate_success(initial_z: float, final_z: float, dz_threshold: float = 0.03) -> bool:
    """
    Heuristic success metric:
    - success if object COM in z has increased by more than dz_threshold.
    """
    dz = final_z - initial_z
    return dz > dz_threshold


def ensure_output_dirs(base_dir: str, views: Iterable[str]) -> dict:
    """Create output directories for each camera view and return mapping."""
    view_dirs = {}
    for view in views:
        path = os.path.join(base_dir, view)
        os.makedirs(path, exist_ok=True)
        view_dirs[view] = path
    return view_dirs


def quat_to_rpy(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float, float]:
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


def run_sweep(
    num_trials: int = 16,
    views: List[str] = ["side", "top", "wrist"],
    noise_std: float = 0.05,
    seed: int = 0,
) -> None:
    """Run a sweep of pick-up attempts, saving videos + LeRobot dataset + summary CSV."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    # LeRobot dataset root
    dataset_root = Path(root_dir) / "datasets" / "franka_table_synth"
    writer = LeRobotDatasetWriter(
        root=dataset_root,
        dataset_name="franka_table_synth",
        robot_type="FrankaPanda",
        fps=30,  # match video FPS
        cameras=views,
    )

    # Dataset video shards root: videos/{camera}/chunk-000/file-XXX.mp4
    videos_root = dataset_root / "videos"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    view_dirs: Dict[str, str] = {}
    for cam in views:
        cam_chunk_dir = videos_root / cam / "chunk-000"
        cam_chunk_dir.mkdir(parents=True, exist_ok=True)
        view_dirs[cam] = str(cam_chunk_dir)

    # Separate summary directory for human inspection
    summary_root = os.path.join(root_dir, "videos", "robot1_pickup_sweeps")
    os.makedirs(summary_root, exist_ok=True)
    run_dir = os.path.join(summary_root, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    scene_path = find_scene_path()
    model = mujoco.MjModel.from_xml_path(scene_path)

    # Pre-compute IDs for EE site, wrist camera, and finger joints
    ee_site_name = "robot1_gripper_center"
    finger_joint1_name = "robot1_finger_joint1"
    finger_joint2_name = "robot1_finger_joint2"

    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
    wrist_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "robot1_wrist")
    j1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, finger_joint1_name)
    j2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, finger_joint2_name)
    j1_qadr = model.jnt_qposadr[j1_id]
    j2_qadr = model.jnt_qposadr[j2_id]

    # Side camera setup for dataset images
    side_cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(side_cam)
    side_cam.azimuth = 90
    side_cam.elevation = -10
    side_cam.distance = 2.5
    side_cam.lookat[:] = [0.0, 0.0, 0.5]
    side_renderer = mujoco.Renderer(model, height=720, width=1280)

    # Wrist camera setup for dataset images
    wrist_cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(wrist_cam)
    wrist_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    wrist_cam.fixedcamid = wrist_cam_id
    wrist_renderer = mujoco.Renderer(model, height=720, width=1280)

    # Top camera for additional view
    top_cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(top_cam)
    top_cam.azimuth = 90
    top_cam.elevation = -89
    top_cam.distance = 3.5
    top_cam.lookat[:] = [0.0, 0.0, 0.5]
    top_renderer = mujoco.Renderer(model, height=720, width=1280)

    print(f"main view: {views[0]}")
    if views[0] == "side":
        main_cam = side_cam
    elif views[0] == "top":
        main_cam = top_cam
    elif views[0] == "wrist":
        main_cam = wrist_cam

    targets = generate_target_sets(num_trials=num_trials, noise_std=noise_std, seed=seed)

    summary_rows: List[str] = []
    header = "trial,view,video_path,delta_z,success,joint_values\n"
    summary_rows.append(header)

    for trial_idx, target_joints in enumerate(targets):
        print("\n" + "=" * 60)
        print(f"TRIAL {trial_idx + 1}/{num_trials}")
        print("=" * 60)
        print("Target joints (radians):", np.round(target_joints, 4))

        data = init_scene_state(model)
        initial_z = float(data.qpos[2])

        episode = writer.start_episode(task_text="Pick up and lift object")
        ep_idx = episode.episode_index
        ep_prefix = f"episode_{ep_idx:06d}"

        frame_counter = 0
        prev_state = None
        prev_grip = None

        # Buffers for videos from a single rollout
        side_frames: List[np.ndarray] = []
        top_frames: List[np.ndarray] = []
        wrist_frames: List[np.ndarray] = []

        def on_frame_cb(
            model_inner: mujoco.MjModel,
            data_inner: mujoco.MjData,
            side_frame: np.ndarray,
        ) -> None:
            nonlocal frame_counter, prev_state, prev_grip

            # Main camera frame as side/top/wrist depending on views[0]
            if views[0] == "side":
                side_frames.append(side_frame)
            elif views[0] == "top":
                top_frames.append(side_frame)
            elif views[0] == "wrist":
                wrist_frames.append(side_frame)

            # Render remaining cameras from the same state
            for view in views[1:]:
                if view == "side":
                    side_renderer.update_scene(data_inner, side_cam)
                    side_frame = side_renderer.render()
                    side_frames.append(side_frame)
                elif view == "top":
                    top_renderer.update_scene(data_inner, top_cam)
                    top_frame = top_renderer.render()
                    top_frames.append(top_frame)
                elif view == "wrist":
                    wrist_renderer.update_scene(data_inner, wrist_cam)
                    wrist_frame = wrist_renderer.render()
                    wrist_frames.append(wrist_frame)

            # End-effector pose
            pos = data_inner.site_xpos[ee_site_id].copy()
            mat_flat = data_inner.site_xmat[ee_site_id].copy()
            quat = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quat, mat_flat)
            qw, qx, qy, qz = quat
            state = np.array([pos[0], pos[1], pos[2], qx, qy, qz, qw], dtype=np.float32)

            # Gripper scalar
            q = data_inner.qpos
            grip = float(0.5 * (q[j1_qadr] + q[j2_qadr]))

            if prev_state is None:
                action = np.zeros(7, dtype=np.float32)
            else:
                dx = state[0] - prev_state[0]
                dy = state[1] - prev_state[1]
                dz = state[2] - prev_state[2]

                roll, pitch, yaw = quat_to_rpy(qw, qx, qy, qz)
                prev_qw = prev_state[6]
                prev_qx = prev_state[3]
                prev_qy = prev_state[4]
                prev_qz = prev_state[5]
                prev_roll, prev_pitch, prev_yaw = quat_to_rpy(
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

            images = {}
            if "side" in views:
                images["side"] = side_frames[-1]
            if "top" in views:
                images["top"] = top_frames[-1]
            if "wrist" in views:
                images["wrist"] = wrist_frames[-1]

            done = False  # mark final frame as done after rollout
            episode.add_frame(
                observation_state=state,
                action=action,
                timestamp=float(data_inner.time),
                done=done,
                images=images,
                frame_index=frame_counter,
                frames_dir=writer.frames_dir,
                episode_prefix=ep_prefix,
            )
            frame_counter += 1

        # Run trajectory once and capture frames + dataset episode
        print(f"\n  Running trajectory with main view camera {views[0]} (all videos/dataset are from this rollout)")
        _ = create_pickup_trajectory(
            model,
            data,
            camera_view=views[0],
            camera=main_cam,
            target_joints=target_joints,
            on_frame=on_frame_cb,
        )

        final_z = float(data.qpos[2])
        delta_z = final_z - initial_z
        success = evaluate_success(initial_z, final_z)

        # Mark final frame as done
        if frame_counter > 0:
            episode.frames[-1]["next.done"] = True

        writer.end_episode(episode, length=frame_counter, task_text="Pick up and lift object")

        joint_str = " ".join(f"{v:.4f}" for v in target_joints)

        # Save videos for each requested view and collect paths for mapping
        video_paths: Dict[str, str] = {}

        if "side" in views and side_frames:
            filename = (
                f"trial_{trial_idx:03d}_view-side_dz-{delta_z:+.3f}_"
                f"success-{int(success)}.mp4"
            )
            output_path = os.path.join(view_dirs["side"], filename)
            print(f"    Saving SIDE video to: {output_path}")
            media.write_video(output_path, side_frames, fps=30)
            summary_rows.append(
                f"{trial_idx},side,{output_path},{delta_z:.6f},{int(success)},\"{joint_str}\"\n"
            )
            video_paths["side"] = os.path.relpath(output_path, dataset_root)

        if "top" in views and top_frames:
            filename = (
                f"trial_{trial_idx:03d}_view-top_dz-{delta_z:+.3f}_"
                f"success-{int(success)}.mp4"
            )
            output_path = os.path.join(view_dirs["top"], filename)
            print(f"    Saving TOP video to: {output_path}")
            media.write_video(output_path, top_frames, fps=30)
            summary_rows.append(
                f"{trial_idx},top,{output_path},{delta_z:.6f},{int(success)},\"{joint_str}\"\n"
            )
            video_paths["top"] = os.path.relpath(output_path, dataset_root)

        if "wrist" in views and wrist_frames:
            filename = (
                f"trial_{trial_idx:03d}_view-wrist_dz-{delta_z:+.3f}_"
                f"success-{int(success)}.mp4"
            )
            output_path = os.path.join(view_dirs["wrist"], filename)
            print(f"    Saving WRIST video to: {output_path}")
            media.write_video(output_path, wrist_frames, fps=30)
            summary_rows.append(
                f"{trial_idx},wrist,{output_path},{delta_z:.6f},{int(success)},\"{joint_str}\"\n"
            )
            video_paths["wrist"] = os.path.relpath(output_path, dataset_root)

        # Append mapping from trial_index -> episode_index + all relevant files
        trial_map_path = dataset_root / "meta" / "trial_mapping.jsonl"
        trial_record = {
            "run_id": timestamp,
            "trial_index": trial_idx,
            "episode_index": ep_idx,
            "delta_z": float(delta_z),
            "success": int(success),
            "data_parquet": f"data/chunk-000/episode_{ep_idx:06d}.parquet",
            "episodes_parquet": "meta/episodes/chunk-000/file-000.parquet",
            "episodes_jsonl": "episodes.jsonl",
            "target_joints": [float(v) for v in target_joints],
            "videos": video_paths,
        }
        with open(trial_map_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trial_record) + "\n")

    # Finalize LeRobot dataset metadata
    writer.finalize()

    summary_path = os.path.join(run_dir, "summary.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.writelines(summary_rows)

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)
    print(f"Videos and summary written to: {run_dir}")
    print(f"Summary CSV: {summary_path}")
    print(f"LeRobot dataset written to: {dataset_root}")


def main():
    # Feel free to tweak these defaults
    run_sweep(
        num_trials=2,
        views=["side"],
        noise_std=0.05,
        seed=0,
    )


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        if "mediapy" in str(e):
            print("\nError: mediapy is not installed.")
            print("Please install it with: pip install mediapy")
        else:
            raise
    except Exception as e:
        print(f"\nError: {e}")
        raise

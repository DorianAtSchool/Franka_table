"""
Robot 1 Pick-up Trajectory Variants

This script explores different *paths* to the same final
target joint configuration used in the original demo.

Instead of changing the target joints, it changes how the
arm moves to that target (ordering of joints, overshoot-
then-back, mirrored/swinging approaches, etc.).

Outputs are saved under:
    franka_table/videos/robot1_pickup_path_variants/run_YYYYMMDD_HHMMSS/

Within each run directory:
    <variant_name>/<view>/trial_000_...mp4
and a summary CSV describing Δz and success for each rollout.
"""

import os
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import mujoco
import numpy as np
import mediapy as media


# Same base target joints as in the original demo / sweep
BASE_TARGET_JOINTS = np.array(
    [1.2999, 1.5000, 0.0499, -0.5208, 1.6501, 1.8710, -1.2852], dtype=float
)


def find_scene_path() -> str:
    """Locate a 4-robot scene XML, trying several known locations."""
    candidates = [
        os.path.join("franka_table", "scenes", "scene_4robots.xml"),
        os.path.join("franka_table", "scenes", "scene_4robots_real.xml"),
        os.path.join(
            "franka_table", "scenes", "franka_emika_panda", "scene_4pandas_table.xml"
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


def setup_camera(model: mujoco.MjModel, camera_view: str) -> Tuple[mujoco.Renderer, mujoco.MjvCamera]:
    """Create renderer and camera matching the original demo views."""
    renderer = mujoco.Renderer(model, height=720, width=1280)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)

    if camera_view == "top":
        # Top-down view of the table
        camera.azimuth = 90
        camera.elevation = -89  # Almost directly from above
        camera.distance = 3.5
        camera.lookat[:] = [0.0, 0.0, 0.5]
        print("Using TOP-DOWN camera view")
    else:
        # Side view aligned with Y-axis, looking straight at the table from the side
        camera.azimuth = 90
        camera.elevation = -10
        camera.distance = 2.5
        camera.lookat[:] = [0.0, 0.0, 0.5]
        print("Using SIDE camera view")

    return renderer, camera


def evaluate_success(initial_z: float, final_z: float, dz_threshold: float = 0.03) -> bool:
    """
    Heuristic success metric:
    - success if object COM in z has increased by more than dz_threshold.
    """
    dz = final_z - initial_z
    return dz > dz_threshold


def ensure_variant_dirs(base_dir: str, variants: Iterable[str], views: Iterable[str]) -> Dict[Tuple[str, str], str]:
    """Create output directories for each (variant, view) and return mapping."""
    mapping: Dict[Tuple[str, str], str] = {}
    for variant in variants:
        for view in views:
            path = os.path.join(base_dir, variant, view)
            os.makedirs(path, exist_ok=True)
            mapping[(variant, view)] = path
    return mapping


def phase1_baseline(initial_joints: np.ndarray, target: np.ndarray, num_steps: int, i: int) -> np.ndarray:
    """Original straight-line cubic interpolation in joint space."""
    t = i / float(num_steps)
    smooth_t = 3 * t**2 - 2 * t**3
    return (1.0 - smooth_t) * initial_joints + smooth_t * target


def phase1_elbow_first(initial_joints: np.ndarray, target: np.ndarray, num_steps: int, i: int) -> np.ndarray:
    """
    Two-phase approach within Phase 1:
    - first half: mostly move shoulder/elbow joints
    - second half: move all joints to target
    """
    half = num_steps // 2
    # Choose "elbow-related" joints (indices 1, 2, 3 here)
    elbow_indices = [1, 2, 3]

    # Phase 1a: move only elbow-related joints toward target
    mid_joints = initial_joints.copy()
    for idx in elbow_indices:
        mid_joints[idx] = target[idx]

    if i < half:
        t = i / float(half)
        smooth_t = 3 * t**2 - 2 * t**3
        return (1.0 - smooth_t) * initial_joints + smooth_t * mid_joints

    # Phase 1b: move full arm from mid_joints to final target
    t = (i - half) / float(num_steps - half)
    smooth_t = 3 * t**2 - 2 * t**3
    return (1.0 - smooth_t) * mid_joints + smooth_t * target


def phase1_overshoot(initial_joints: np.ndarray, target: np.ndarray, num_steps: int, i: int) -> np.ndarray:
    """
    Overshoot-then-back path:
    - first half: move slightly beyond target in joint space
    - second half: move back from overshoot to target
    """
    half = num_steps // 2

    delta = target - initial_joints
    overshoot_target = initial_joints + 1.15 * delta
    overshoot_target = np.clip(overshoot_target, -2.5, 2.5)

    if i < half:
        t = i / float(half)
        smooth_t = 3 * t**2 - 2 * t**3
        return (1.0 - smooth_t) * initial_joints + smooth_t * overshoot_target

    t = (i - half) / float(num_steps - half)
    smooth_t = 3 * t**2 - 2 * t**3
    return (1.0 - smooth_t) * overshoot_target + smooth_t * target


def phase1_swing(initial_joints: np.ndarray, target: np.ndarray, num_steps: int, i: int) -> np.ndarray:
    """
    Swinging/mirrored approach:
    - follow the baseline path but add a sinusoidal offset on joint 1,
      which swings the approach around the object before settling.
    """
    base = phase1_baseline(initial_joints, target, num_steps, i)

    t = i / float(num_steps)
    # Sinusoidal offset that starts and ends at 0 (so final pose is identical)
    swing_offset = 0.35 * np.sin(np.pi * t)
    base = base.copy()
    base[0] += swing_offset  # modify joint 1 only
    return base


PHASE1_VARIANTS = {
    "baseline": phase1_baseline,
    "elbow_first": phase1_elbow_first,
    "overshoot": phase1_overshoot,
    "swing": phase1_swing,
}


def run_pickup_variant(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_view: str,
    variant_name: str,
    target_joints: np.ndarray,
) -> List[np.ndarray]:
    """
    Run a pick-up trajectory using a specific Phase 1 variant.

    The phases are:
      1) Move to grasp pose (using variant-specific path)
      2) Hold before grasp
      3) Close gripper
      4) Lift object
      5) Hold lifted pose
    """
    frames: List[np.ndarray] = []

    renderer, camera = setup_camera(model, camera_view)

    ctrl_start = 0  # Robot 1

    # Ensure we have a clean copy of the target
    target = np.array(target_joints, dtype=float)

    phase1_fn = PHASE1_VARIANTS[variant_name]

    # Phase 1: Move to grasp position
    print(f"Phase 1 ({variant_name}): Moving to grasp position...")
    initial_joints = data.ctrl[ctrl_start:ctrl_start + 7].copy()

    phase1_steps = 700
    for i in range(phase1_steps):
        joints = phase1_fn(initial_joints, target, phase1_steps, i)
        data.ctrl[ctrl_start:ctrl_start + 7] = joints
        data.ctrl[ctrl_start + 7] = 255  # keep gripper open

        for _ in range(5):
            mujoco.mj_step(model, data)

        if i % 3 == 0:
            renderer.update_scene(data, camera)
            frame = renderer.render()
            frames.append(frame)

    # Phase 2: Hold position briefly before grasping
    print("Phase 2: Holding position before grasp...")
    for i in range(200):
        mujoco.mj_step(model, data)

        if i % 3 == 0:
            renderer.update_scene(data, camera)
            frame = renderer.render()
            frames.append(frame)

    # Phase 3: Close gripper
    print("Phase 3: Closing gripper...")
    for i in range(100):
        t = i / 100.0
        smooth_t = 3 * t**2 - 2 * t**3
        data.ctrl[ctrl_start + 7] = 255 * (1.0 - smooth_t)

        mujoco.mj_step(model, data)

        if i % 3 == 0:
            renderer.update_scene(data, camera)
            frame = renderer.render()
            frames.append(frame)

    # Phase 4: Lift up by moving Joint 2 up (same as original demo)
    print("Phase 4: Lifting object up...")
    target_joints_up = target.copy()
    # From interactive session: joint 2 ~ 1.7 -> 1.1002 rad when lifting
    target_joints_up[1] = 1.1002
    current_joints = data.ctrl[ctrl_start:ctrl_start + 7].copy()

    for i in range(150):
        t = i / 150.0
        smooth_t = 3 * t**2 - 2 * t**3
        data.ctrl[ctrl_start:ctrl_start + 7] = (1.0 - smooth_t) * current_joints + smooth_t * target_joints_up
        data.ctrl[ctrl_start + 7] = 0  # keep gripper closed

        mujoco.mj_step(model, data)

        if i % 3 == 0:
            renderer.update_scene(data, camera)
            frame = renderer.render()
            frames.append(frame)

    # Phase 5: Hold for a moment
    print("Phase 5: Holding lifted position...")
    for i in range(60):
        mujoco.mj_step(model, data)

        if i % 3 == 0:
            renderer.update_scene(data, camera)
            frame = renderer.render()
            frames.append(frame)

    return frames


def run_variants(
    variants: Tuple[str, ...] = ("baseline", "elbow_first", "overshoot", "swing"),
    views: Tuple[str, ...] = ("side", "top"),
) -> None:
    """
    Run each trajectory variant once for each camera view.

    All runs use the same target joints; only the path to
    get there is changed.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    videos_root = os.path.join(
        os.path.dirname(script_dir), "videos", "robot1_pickup_path_variants"
    )
    os.makedirs(videos_root, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(videos_root, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    variant_dirs = ensure_variant_dirs(run_dir, variants, views)

    scene_path = find_scene_path()
    model = mujoco.MjModel.from_xml_path(scene_path)

    summary_rows: List[str] = []
    header = "variant,view,video_path,delta_z,success\n"
    summary_rows.append(header)

    trial_idx = 0
    for variant in variants:
        if variant not in PHASE1_VARIANTS:
            print(f"Skipping unknown variant: {variant}")
            continue

        print("\n" + "=" * 60)
        print(f"VARIANT: {variant}")
        print("=" * 60)

        for view in views:
            print(f"\n  Camera view: {view}")
            data = init_scene_state(model)

            initial_z = float(data.qpos[2])
            frames = run_pickup_variant(
                model=model,
                data=data,
                camera_view=view,
                variant_name=variant,
                target_joints=BASE_TARGET_JOINTS,
            )
            final_z = float(data.qpos[2])
            delta_z = final_z - initial_z
            success = evaluate_success(initial_z, final_z)

            filename = (
                f"trial_{trial_idx:03d}_variant-{variant}_view-{view}_"
                f"dz-{delta_z:+.3f}_success-{int(success)}.mp4"
            )
            output_dir = variant_dirs[(variant, view)]
            output_path = os.path.join(output_dir, filename)

            print(f"    Saving video to: {output_path}")
            media.write_video(output_path, frames, fps=30)
            print(
                f"    Done. Δz={delta_z:+.3f} m, success={success}, "
                f"frames={len(frames)}"
            )

            summary_rows.append(
                f"{variant},{view},{output_path},{delta_z:.6f},{int(success)}\n"
            )
            trial_idx += 1

    summary_path = os.path.join(run_dir, "summary.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.writelines(summary_rows)

    print("\n" + "=" * 60)
    print("PATH VARIANTS COMPLETE")
    print("=" * 60)
    print(f"Videos and summary written to: {run_dir}")
    print(f"Summary CSV: {summary_path}")


def main():
    run_variants()


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


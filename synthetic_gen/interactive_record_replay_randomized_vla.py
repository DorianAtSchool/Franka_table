"""
Record/replay wrapper using a randomized environment and VLA-style GUI controls.

This script is similar to ``interactive_record_replay_randomized.py`` but the
GUI exposes end-effector controls matching VLA-style actions:

  - Position deltas:  dx, dy, dz  (world frame)
  - Rotation deltas:  droll, dpitch, dyaw
  - Gripper:          open / close

The dataset written by ``LeRobotDatasetWriter`` follows the same convention as
``interactive_record_replay.py``:

  state  = [x, y, z, qx, qy, qz, qw]
  action = [dx, dy, dz, droll, dpitch, dyaw, dgrip]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except ImportError:  # pragma: no cover
    tk = None
    ttk = None

# Allow importing sibling/parent modules without altering existing files.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import mujoco
import mujoco.viewer

from environments import RandomizedFrankaTable4RobotsEnv  # type: ignore
from synthetic_gen.interactive_record_replay import RecordReplayWrapper  # type: ignore
from datasets.lerobot_writer import LeRobotDatasetWriter  # type: ignore
from utils.jacobian import jacobian_ik6_step  # type: ignore

# import threading?

class RandomizedVLAController:
    """
    Controller that exposes VLA-style end-effector controls on top of the
    RandomizedFrankaTable4RobotsEnv, while sharing its MuJoCo model/data.

    The interface is intentionally compatible with RecordReplayWrapper.
    """

    def __init__(
        self,
        mjcf_path: str,
        robot_index: int = 0,
        object_xy_range: tuple[float, float, float, float] = (-0.3, 0.3, -0.25, 0.25),
        object_z: float = 0.535,
        randomize_object_orientation: bool = False,
        joint_range_fraction: float = 0.3,
    ) -> None:
        # Underlying randomized Gymnasium environment.
        self.env = RandomizedFrankaTable4RobotsEnv(
            mjcf_path=mjcf_path,
            render_mode="human",
            object_xy_range=object_xy_range,
            object_z=object_z,
            randomize_object_orientation=randomize_object_orientation,
            joint_range_fraction=joint_range_fraction,
            extra_settle_steps=0,
        )
        self.model = self.env.model
        self.data = self.env.data

        self.robot_index = robot_index
        self.robot_prefix = f"robot{robot_index + 1}_"

        # Joint segment indices for this robot within qpos/ctrl.
        self.qpos_start = 7 + robot_index * 9
        self.ctrl_start = robot_index * 8

        # End-effector site for VLA control (center of gripper).
        ee_site_name = f"robot{robot_index + 1}_gripper_center"
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        if self.ee_site_id < 0:
            raise RuntimeError(f"EE site '{ee_site_name}' not found in model.")

        # Viewer / simulation loop state.
        self.viewer: mujoco.viewer.Handle | None = None  # type: ignore[attr-defined]
        self.running: bool = True
        self.simulation_thread: threading.Thread | None = None

        # Control step sizes in task space.
        self.pos_step = 0.02  # meters per GUI click
        self.rot_step = np.deg2rad(5.0)  # radians per GUI click
        self.gripper_step = 20.0  # actuator units per click (0..255)

        # Initialize to a randomized configuration.
        self.reset_robot()

    # ------------------------------------------------------------------
    # Lifecycle / simulation
    # ------------------------------------------------------------------
    def reset_robot(self) -> None:
        """Reset environment to a new randomized configuration."""

        self.env.reset()
        print("Scene reset to randomized initial configuration (VLA controller).")


    def simulation_loop(self) -> None:
        """Run passive MuJoCo viewer in a background thread."""
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.azimuth = 135
        self.viewer.cam.elevation = -20
        self.viewer.cam.distance = 3.0
        self.viewer.cam.lookat[:] = [0, 0, 0.5]

        while self.running and self.viewer.is_running():
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.002)

    def start_simulation(self) -> None:
        import threading

        self.simulation_thread = threading.Thread(target=self.simulation_loop, daemon=True)
        self.simulation_thread.start()

    def stop(self) -> None:
        """Stop background stepping and close viewer."""
        self.running = False
        if self.viewer is not None:
            self.viewer.close()
        if getattr(self, "simulation_thread", None) is not None:
            self.simulation_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # VLA-style control interface
    # ------------------------------------------------------------------
    def apply_ee_delta(
        self,
        dpos: np.ndarray,
        drot_rpy: np.ndarray,
        dgrip: float,
    ) -> None:
        """
        Apply a small end-effector delta using a single Jacobian IK step.

        dpos: desired EE translation [dx, dy, dz] in meters.
        drot_rpy: desired EE rotation [droll, dpitch, dyaw] in radians.
        dgrip: additive delta to gripper actuator (0..255 range).
        """
        # Map discrete delta to a 6D twist over one control_dt.
        dt = float(self.env.control_dt)
        v_des = (dpos / dt).astype(np.float32)
        w_des = (drot_rpy / dt).astype(np.float32)

        dq = jacobian_ik6_step(
            model=self.model,
            data=self.data,
            site_id=self.ee_site_id,
            robot_idx=self.robot_index,
            v_des=v_des,
            w_des=w_des,
            dt=dt,
            damping=0.1,
        ).astype(np.float32)

        # Update joint targets for this robot in ctrl (do not touch qpos directly).
        q_target = self.data.ctrl[self.ctrl_start : self.ctrl_start + 7].copy()
        if not np.any(q_target):
            # If ctrl is zero-initialized, start from current qpos to avoid jumps.
            q_target = self.data.qpos[self.qpos_start : self.qpos_start + 7].copy()
        q_target = q_target + dq
        self.data.ctrl[self.ctrl_start : self.ctrl_start + 7] = q_target

        # Gripper control in [0, 255].
        grip_val = float(self.data.ctrl[self.ctrl_start + 7]) + float(dgrip)
        grip_val = float(np.clip(grip_val, 0.0, 255.0))
        self.data.ctrl[self.ctrl_start + 7] = grip_val

    # ------------------------------------------------------------------
    # Minimal API expected by RecordReplayWrapper
    # ------------------------------------------------------------------
    def get_joint_position(self, joint_index: int) -> float:
        """Compatibility shim; not used by VLA GUI but required by wrapper."""
        pos_rad = self.data.qpos[self.qpos_start + joint_index]
        return float(np.degrees(pos_rad))

    def get_gripper_position(self) -> float:
        return float(self.data.ctrl[self.ctrl_start + 7])


def create_vla_gui(controller: RandomizedVLAController) -> tk.Tk | None:
    """Create a simple Tk GUI exposing VLA-style EE controls."""
    if tk is None or ttk is None:
        print("Error: tkinter is required for VLA GUI mode")
        return None

    root = tk.Tk()
    root.title(f"Robot {controller.robot_index + 1} VLA Control (Randomized)")
    root.geometry("420x420")

    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    title = ttk.Label(
        frame,
        text=f"Robot {controller.robot_index + 1} End-Effector Control",
        font=("Arial", 12, "bold"),
    )
    title.grid(row=0, column=0, columnspan=3, pady=(0, 10))

    info = ttk.Label(
        frame,
        text=f"Pos step: {controller.pos_step:.3f} m | Rot step: {np.degrees(controller.rot_step):.1f} deg",
        font=("Arial", 9, "italic"),
    )
    info.grid(row=1, column=0, columnspan=3, pady=(0, 10))

    def do_move(dpos, drot, dgrip=0.0) -> None:
        controller.apply_ee_delta(np.array(dpos, dtype=np.float32), np.array(drot, dtype=np.float32), dgrip)

    row = 2
    # Position controls
    ttk.Label(frame, text="Position (world frame)", font=("Arial", 10, "bold")).grid(
        row=row, column=0, columnspan=3, pady=(5, 5)
    )
    row += 1

    ttk.Button(frame, text="X-", width=8, command=lambda: do_move((-controller.pos_step, 0, 0), (0, 0, 0))).grid(
        row=row, column=0, padx=2, pady=2
    )
    ttk.Button(frame, text="X+", width=8, command=lambda: do_move((controller.pos_step, 0, 0), (0, 0, 0))).grid(
        row=row, column=2, padx=2, pady=2
    )
    ttk.Label(frame, text="X axis").grid(row=row, column=1)
    row += 1

    ttk.Button(frame, text="Y-", width=8, command=lambda: do_move((0, -controller.pos_step, 0), (0, 0, 0))).grid(
        row=row, column=0, padx=2, pady=2
    )
    ttk.Button(frame, text="Y+", width=8, command=lambda: do_move((0, controller.pos_step, 0), (0, 0, 0))).grid(
        row=row, column=2, padx=2, pady=2
    )
    ttk.Label(frame, text="Y axis").grid(row=row, column=1)
    row += 1

    ttk.Button(frame, text="Z-", width=8, command=lambda: do_move((0, 0, -controller.pos_step), (0, 0, 0))).grid(
        row=row, column=0, padx=2, pady=2
    )
    ttk.Button(frame, text="Z+", width=8, command=lambda: do_move((0, 0, controller.pos_step), (0, 0, 0))).grid(
        row=row, column=2, padx=2, pady=2
    )
    ttk.Label(frame, text="Z axis").grid(row=row, column=1)
    row += 1

    # Rotation controls
    ttk.Label(frame, text="Rotation (roll, pitch, yaw)", font=("Arial", 10, "bold")).grid(
        row=row, column=0, columnspan=3, pady=(10, 5)
    )
    row += 1

    r = controller.rot_step
    ttk.Button(frame, text="Roll-", width=8, command=lambda: do_move((0, 0, 0), (-r, 0, 0))).grid(
        row=row, column=0, padx=2, pady=2
    )
    ttk.Button(frame, text="Roll+", width=8, command=lambda: do_move((0, 0, 0), (r, 0, 0))).grid(
        row=row, column=2, padx=2, pady=2
    )
    ttk.Label(frame, text="Roll").grid(row=row, column=1)
    row += 1

    ttk.Button(frame, text="Pitch-", width=8, command=lambda: do_move((0, 0, 0), (0, -r, 0))).grid(
        row=row, column=0, padx=2, pady=2
    )
    ttk.Button(frame, text="Pitch+", width=8, command=lambda: do_move((0, 0, 0), (0, r, 0))).grid(
        row=row, column=2, padx=2, pady=2
    )
    ttk.Label(frame, text="Pitch").grid(row=row, column=1)
    row += 1

    ttk.Button(frame, text="Yaw-", width=8, command=lambda: do_move((0, 0, 0), (0, 0, -r))).grid(
        row=row, column=0, padx=2, pady=2
    )
    ttk.Button(frame, text="Yaw+", width=8, command=lambda: do_move((0, 0, 0), (0, 0, r))).grid(
        row=row, column=2, padx=2, pady=2
    )
    ttk.Label(frame, text="Yaw").grid(row=row, column=1)
    row += 1

    # Gripper controls
    ttk.Label(frame, text="Gripper", font=("Arial", 10, "bold")).grid(
        row=row, column=0, columnspan=3, pady=(10, 5)
    )
    row += 1

    ttk.Button(
        frame,
        text="Close",
        width=8,
        command=lambda: do_move((0, 0, 0), (0, 0, 0), -controller.gripper_step),
    ).grid(row=row, column=0, padx=2, pady=2)

    grip_label = ttk.Label(frame, text="0", width=10, relief="sunken", anchor="center")
    grip_label.grid(row=row, column=1, padx=2)

    ttk.Button(
        frame,
        text="Open",
        width=8,
        command=lambda: do_move((0, 0, 0), (0, 0, 0), controller.gripper_step),
    ).grid(row=row, column=2, padx=2, pady=2)
    row += 1

    # Reset button
    ttk.Button(frame, text="Reset (randomized)", command=controller.reset_robot).grid(
        row=row, column=0, columnspan=3, pady=(15, 5)
    )
    row += 1

    # Periodically update gripper display
    def update_status() -> None:
        if getattr(controller, "running", False):
            try:
                grip_val = controller.get_gripper_position()
                grip_label.config(text=f"{grip_val:.0f}")
            except Exception:
                pass
        root.after(100, update_status)

    root.after(100, update_status)

    def on_closing() -> None:
        controller.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    return root


def _resolve_scene(scene_arg: str) -> Path:
    scene_path = Path(scene_arg)
    if not scene_path.is_absolute():
        scene_path = (REPO_ROOT / scene_arg).resolve()
    return scene_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive VLA-style control with randomized starts and record/replay wrapper."
    )
    parser.add_argument("--robot", type=int, default=0, choices=[0, 1, 2, 3], help="Robot index to control.")
    parser.add_argument(
        "--scene",
        type=str,
        default="scenes/scene_4robots.xml",
        help="Path to the MuJoCo scene XML (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(REPO_ROOT / "datasets" / "franka_table_manual_randomized_vla"),
        help="Where to store the recorded dataset.",
    )
    parser.add_argument("--task", type=str, default="manual demonstration", help="Task text stored with the episode.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="franka_table_manual_randomized_vla",
        help="Name stored in info.json.",
    )
    parser.add_argument("--fps", type=int, default=25, help="Recording frames per second.")
    parser.add_argument("--camera", type=str, default="side", help="Camera name to render for video.")
    parser.add_argument("--video-width", type=int, default=1280, help="Video width for captured frames.")
    parser.add_argument("--video-height", type=int, default=720, help="Video height for captured frames.")
    # Optional hooks into the randomized env behaviour.
    parser.add_argument(
        "--object-xy-range",
        type=float,
        nargs=4,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        default=(-0.3, 0.3, -0.25, 0.25),
        help="Sampling range for object (x, y) on the table.",
    )
    parser.add_argument(
        "--object-z",
        type=float,
        default=0.535,
        help="Fixed z-height for the object on reset.",
    )
    parser.add_argument(
        "--randomize-object-orientation",
        action="store_true",
        help="If set, sample a random orientation for the object on reset.",
    )
    parser.add_argument(
        "--joint-range-fraction",
        type=float,
        default=0.3,
        help="Fraction of each joint's range to sample from (0-1).",
    )
    args = parser.parse_args()

    if tk is None or ttk is None:
        raise RuntimeError("tkinter is required to run the VLA record/replay GUI.")

    scene_path = _resolve_scene(args.scene)
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    dataset_root = Path(args.dataset_root).resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    controller = RandomizedVLAController(
        mjcf_path=str(scene_path),
        robot_index=args.robot,
        object_xy_range=tuple(args.object_xy_range),
        object_z=args.object_z,
        randomize_object_orientation=args.randomize_object_orientation,
        joint_range_fraction=args.joint_range_fraction,
    )
    controller.start_simulation()

    root = create_vla_gui(controller)
    if root is None:
        controller.stop()
        return

    writer = LeRobotDatasetWriter(
        root=dataset_root,
        dataset_name=args.dataset_name,
        fps=args.fps,
        cameras=[args.camera],
    )

    wrapper = RecordReplayWrapper(
        controller=controller,
        root=root,
        writer=writer,
        task_text=args.task,
        fps=args.fps,
        camera=args.camera,
        video_width=args.video_width,
        video_height=args.video_height,
    )

    try:
        root.mainloop()
    finally:
        controller.stop()
        _ = wrapper


if __name__ == "__main__":
    main()

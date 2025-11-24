"""
Mixed intervention recording with randomized env and VLA-style controls.

This script combines:
- RandomizedFrankaTable4RobotsEnv (shared MuJoCo model/data)
- A VLA-style end-effector controller (RandomizedVLAController)
- An automatic pick-up policy similar in spirit to robot1_pickup_demo
- Manual GUI controls for end-effector deltas + gripper
- RecordReplayWrapper + LeRobotDatasetWriter for dataset logging

You can switch between automatic pick-up and manual intervention at any time.
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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from synthetic_gen.interactive_record_replay_randomized_vla import (  # type: ignore
    RandomizedVLAController,
)
from synthetic_gen.interactive_record_replay import RecordReplayWrapper  # type: ignore
from datasets.lerobot_writer import LeRobotDatasetWriter  # type: ignore


class AutoPickupPolicy:
    """
    Simple phased pick-up policy operating in EE space.

    Phases (robot_index refers to controller.robot_index):
      0: Move EE above object (offset_z_high)
      1: Move EE down to near object (offset_z_low)
      2: Close gripper
      3: Lift object up (offset_z_lift)
      4: Hold (no-op, policy finished)

    At each call to step(), we compute a small EE delta using controller.apply_ee_delta.
    """

    def __init__(
        self,
        controller: RandomizedVLAController,
        offset_z_high: float = 0.20,
        offset_z_low: float = 0.03,
        offset_z_lift: float = 0.30,
    ) -> None:
        self.controller = controller
        self.offset_z_high = float(offset_z_high)
        self.offset_z_low = float(offset_z_low)
        self.offset_z_lift = float(offset_z_lift)
        self.phase: int = 0
        self.active: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """
        (Re)start the phased policy.

        - If a run has finished (phase >= 4), start a fresh sequence from phase 0.
        - If the policy was paused mid-run via manual intervention, resume from
          the current phase and the *current* world state (EE/object positions).
        """
        if self.phase >= 4 or self.phase < 0:
            self.phase = 0
        self.active = True

    def stop(self) -> None:
        self.active = False

    def is_running(self) -> bool:
        return self.active

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    def step(self) -> None:
        """Advance one small step of the automatic policy."""
        if not self.active:
            return

        data = self.controller.data
        # Object position is first body qpos[0:3]
        obj_pos = data.qpos[0:3].copy()
        ee_pos = data.site_xpos[self.controller.ee_site_id].copy()

        dpos = np.zeros(3, dtype=np.float32)
        drot = np.zeros(3, dtype=np.float32)
        dgrip = 0.0

        # Phase 0: move above object
        if self.phase == 0:
            target = obj_pos + np.array([0.0, 0.0, self.offset_z_high], dtype=np.float32)
            dpos = self._clamped_delta(ee_pos, target, 0.5 * self.controller.pos_step)
            if np.linalg.norm(target - ee_pos) < 0.01:
                self.phase = 1

        # Phase 1: move down near object
        elif self.phase == 1:
            target = obj_pos + np.array([0.0, 0.0, self.offset_z_low], dtype=np.float32)
            dpos = self._clamped_delta(ee_pos, target, 0.5 * self.controller.pos_step)
            if np.linalg.norm(target - ee_pos) < 0.005:
                self.phase = 2

        # Phase 2: close gripper
        elif self.phase == 2:
            dgrip = -self.controller.gripper_step
            # Consider gripper "closed enough" when command < threshold
            if self.controller.get_gripper_position() < 80.0:
                self.phase = 3

        # Phase 3: lift object
        elif self.phase == 3:
            target = obj_pos + np.array([0.0, 0.0, self.offset_z_lift], dtype=np.float32)
            dpos = self._clamped_delta(ee_pos, target, 0.5 * self.controller.pos_step)
            if np.linalg.norm(target - ee_pos) < 0.01:
                self.phase = 4

        # Phase 4: hold; policy finished
        elif self.phase == 4:
            self.active = False
            return

        # Apply the EE delta for this step.
        if self.active:
            self.controller.apply_ee_delta(dpos, drot, dgrip)

    @staticmethod
    def _clamped_delta(current: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
        delta = target - current
        dist = float(np.linalg.norm(delta))
        if dist <= max_step or dist == 0.0:
            return delta.astype(np.float32)
        return (delta / dist * max_step).astype(np.float32)


def create_mixed_gui(
    controller: RandomizedVLAController,
    auto_policy: AutoPickupPolicy,
    root: tk.Tk | None = None,
    parent: tk.Widget | None = None,
    column: int = 0,
) -> tk.Tk | None:
    """GUI with both automatic and manual VLA-style controls.

    If ``root`` is None, a new top-level Tk window is created. Otherwise, the
    controls are attached to the provided root at the given ``row_offset`` so
    multiple robots can share a single window.
    """
    if tk is None or ttk is None:
        print("Error: tkinter is required for mixed intervention GUI mode")
        return None

    own_root = False
    if root is None:
        root = tk.Tk()
        root.title(f"Robot {controller.robot_index + 1} Mixed Control (Randomized)")
        root.geometry("460x520")
        own_root = True

    container = parent if parent is not None else root

    frame = ttk.Frame(container, padding="10", borderwidth=2, relief="groove")
    frame.grid(row=0, column=column, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

    title = ttk.Label(
        frame,
        text=f"Robot {controller.robot_index + 1} VLA Mixed Control",
        font=("Arial", 12, "bold"),
    )
    title.grid(row=0, column=0, columnspan=3, pady=(0, 10))

    info = ttk.Label(
        frame,
        text=f"Pos step: {controller.pos_step:.3f} m | Rot step: {np.degrees(controller.rot_step):.1f} deg",
        font=("Arial", 9, "italic"),
    )
    info.grid(row=1, column=0, columnspan=3, pady=(0, 10))

    # Status label for automatic policy
    auto_status = tk.StringVar(value="Auto: idle")
    # Keep references to human-control buttons so we can enable/disable them
    manual_buttons: list[ttk.Button] = []

    def update_auto_status() -> None:
        running = auto_policy.is_running()
        if running:
            auto_status.set(f"Auto: running (phase {auto_policy.phase})")
        else:
            auto_status.set("Auto: idle")

        # Disable manual controls while auto policy is running.
        for btn in manual_buttons:
            try:
                btn.configure(state="disabled" if running else "normal")
            except Exception:
                pass

        root.after(200, update_auto_status)

    status_label = ttk.Label(frame, textvariable=auto_status, font=("Arial", 9, "italic"))
    status_label.grid(row=2, column=0, columnspan=3, pady=(0, 10))

    def do_move(dpos, drot, dgrip=0.0) -> None:
        # Manual VLA-style intervention (guarded when auto is running)
        if auto_policy.is_running():
            return
        controller.apply_ee_delta(np.array(dpos, dtype=np.float32), np.array(drot, dtype=np.float32), dgrip)

    row = 3
    # Automatic control buttons
    ttk.Label(frame, text="Automatic pick-up", font=("Arial", 10, "bold")).grid(
        row=row, column=0, columnspan=3, pady=(5, 5)
    )
    row += 1

    def start_auto() -> None:
        auto_policy.reset()

    def stop_auto() -> None:
        auto_policy.stop()

    ttk.Button(frame, text="Start Auto Pickup", command=start_auto).grid(
        row=row, column=0, columnspan=2, padx=2, pady=2, sticky=(tk.W, tk.E)
    )
    ttk.Button(frame, text="Stop Auto", command=stop_auto).grid(
        row=row, column=2, padx=2, pady=2, sticky=(tk.W, tk.E)
    )
    row += 1

    # Manual position controls
    ttk.Label(frame, text="Manual EE Position", font=("Arial", 10, "bold")).grid(
        row=row, column=0, columnspan=3, pady=(10, 5)
    )
    row += 1

    btn_x_neg = ttk.Button(frame, text="X-", width=8, command=lambda: do_move((-controller.pos_step, 0, 0), (0, 0, 0)))
    btn_x_neg.grid(row=row, column=0, padx=2, pady=2)
    btn_x_pos = ttk.Button(frame, text="X+", width=8, command=lambda: do_move((controller.pos_step, 0, 0), (0, 0, 0)))
    btn_x_pos.grid(row=row, column=2, padx=2, pady=2)
    manual_buttons.extend([btn_x_neg, btn_x_pos])
    ttk.Label(frame, text="X axis").grid(row=row, column=1)
    row += 1

    btn_y_neg = ttk.Button(frame, text="Y-", width=8, command=lambda: do_move((0, -controller.pos_step, 0), (0, 0, 0)))
    btn_y_neg.grid(row=row, column=0, padx=2, pady=2)
    btn_y_pos = ttk.Button(frame, text="Y+", width=8, command=lambda: do_move((0, controller.pos_step, 0), (0, 0, 0)))
    btn_y_pos.grid(row=row, column=2, padx=2, pady=2)
    manual_buttons.extend([btn_y_neg, btn_y_pos])
    ttk.Label(frame, text="Y axis").grid(row=row, column=1)
    row += 1

    btn_z_neg = ttk.Button(frame, text="Z-", width=8, command=lambda: do_move((0, 0, -controller.pos_step), (0, 0, 0)))
    btn_z_neg.grid(row=row, column=0, padx=2, pady=2)
    btn_z_pos = ttk.Button(frame, text="Z+", width=8, command=lambda: do_move((0, 0, controller.pos_step), (0, 0, 0)))
    btn_z_pos.grid(row=row, column=2, padx=2, pady=2)
    manual_buttons.extend([btn_z_neg, btn_z_pos])
    ttk.Label(frame, text="Z axis").grid(row=row, column=1)
    row += 1

    # Manual rotation controls
    ttk.Label(frame, text="Manual EE Rotation", font=("Arial", 10, "bold")).grid(
        row=row, column=0, columnspan=3, pady=(10, 5)
    )
    row += 1

    r = controller.rot_step
    btn_roll_neg = ttk.Button(frame, text="Roll-", width=8, command=lambda: do_move((0, 0, 0), (-r, 0, 0)))
    btn_roll_neg.grid(row=row, column=0, padx=2, pady=2)
    btn_roll_pos = ttk.Button(frame, text="Roll+", width=8, command=lambda: do_move((0, 0, 0), (r, 0, 0)))
    btn_roll_pos.grid(row=row, column=2, padx=2, pady=2)
    manual_buttons.extend([btn_roll_neg, btn_roll_pos])
    ttk.Label(frame, text="Roll").grid(row=row, column=1)
    row += 1

    btn_pitch_neg = ttk.Button(frame, text="Pitch-", width=8, command=lambda: do_move((0, 0, 0), (0, -r, 0)))
    btn_pitch_neg.grid(row=row, column=0, padx=2, pady=2)
    btn_pitch_pos = ttk.Button(frame, text="Pitch+", width=8, command=lambda: do_move((0, 0, 0), (0, r, 0)))
    btn_pitch_pos.grid(row=row, column=2, padx=2, pady=2)
    manual_buttons.extend([btn_pitch_neg, btn_pitch_pos])
    ttk.Label(frame, text="Pitch").grid(row=row, column=1)
    row += 1

    btn_yaw_neg = ttk.Button(frame, text="Yaw-", width=8, command=lambda: do_move((0, 0, 0), (0, 0, -r)))
    btn_yaw_neg.grid(row=row, column=0, padx=2, pady=2)
    btn_yaw_pos = ttk.Button(frame, text="Yaw+", width=8, command=lambda: do_move((0, 0, 0), (0, 0, r)))
    btn_yaw_pos.grid(row=row, column=2, padx=2, pady=2)
    manual_buttons.extend([btn_yaw_neg, btn_yaw_pos])
    ttk.Label(frame, text="Yaw").grid(row=row, column=1)
    row += 1

    # Gripper controls
    ttk.Label(frame, text="Gripper", font=("Arial", 10, "bold")).grid(
        row=row, column=0, columnspan=3, pady=(10, 5)
    )
    row += 1

    def grip_delta(delta: float) -> None:
        do_move((0, 0, 0), (0, 0, 0), delta)

    btn_grip_close = ttk.Button(
        frame,
        text="Close",
        width=8,
        command=lambda: grip_delta(-controller.gripper_step),
    )
    btn_grip_close.grid(row=row, column=0, padx=2, pady=2)

    grip_label = ttk.Label(frame, text="0", width=10, relief="sunken", anchor="center")
    grip_label.grid(row=row, column=1, padx=2)

    btn_grip_open = ttk.Button(
        frame,
        text="Open",
        width=8,
        command=lambda: grip_delta(controller.gripper_step),
    )
    btn_grip_open.grid(row=row, column=2, padx=2, pady=2)
    manual_buttons.extend([btn_grip_close, btn_grip_open])
    row += 1

    # Reset button (randomized reset via env)
    ttk.Button(frame, text="Reset (randomized)", command=controller.reset_robot).grid(
        row=row, column=0, columnspan=3, pady=(15, 5)
    )
    row += 1

    # Periodic updates for auto policy and gripper display
    def tick_auto() -> None:
        if auto_policy.is_running():
            auto_policy.step()
        root.after(40, tick_auto)

    def update_grip_label() -> None:
        if getattr(controller, "running", False):
            try:
                grip_val = controller.get_gripper_position()
                grip_label.config(text=f"{grip_val:.0f}")
            except Exception:
                pass
        root.after(100, update_grip_label)

    root.after(40, tick_auto)
    root.after(100, update_grip_label)
    root.after(200, update_auto_status)

    if own_root:
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
        description="Mixed automatic/manual VLA control with randomized starts and record/replay."
    )
    parser.add_argument("--robot", type=int, default=0, choices=[0, 1, 2, 3], help="Robot index to control.")
    parser.add_argument(
        "--robots",
        type=int,
        nargs="+",
        help="Optional list of robot indices to control (overrides --robot when provided).",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="scenes/scene_4robots.xml",
        help="Path to the MuJoCo scene XML (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(REPO_ROOT / "datasets" / "franka_table_manual_randomized_vla_mixed"),
        help="Where to store the recorded dataset.",
    )
    parser.add_argument("--task", type=str, default="mixed intervention demonstration", help="Task text for the episode.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="franka_table_manual_randomized_vla_mixed",
        help="Name stored in info.json.",
    )
    parser.add_argument("--fps", type=int, default=25, help="Recording frames per second.")
    parser.add_argument("--camera", type=str, default="side", help="Camera name to render for video.")
    parser.add_argument("--video-width", type=int, default=1280, help="Video width for captured frames.")
    parser.add_argument("--video-height", type=int, default=720, help="Video height for captured frames.")
    # Randomized env hooks
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
        raise RuntimeError("tkinter is required to run the mixed intervention GUI.")

    scene_path = _resolve_scene(args.scene)
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    dataset_root = Path(args.dataset_root).resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    # Determine which robots to control.
    robot_indices = args.robots if args.robots is not None else [args.robot]

    controllers: list[RandomizedVLAController] = []
    auto_policies: list[AutoPickupPolicy] = []
    wrappers: list[RecordReplayWrapper] = []

    # Single-robot case: keep existing behavior (one window).
    if len(robot_indices) == 1:
        idx = robot_indices[0]

        writer = LeRobotDatasetWriter(
            root=dataset_root,
            dataset_name=args.dataset_name,
            fps=args.fps,
            cameras=[args.camera],
        )

        controller = RandomizedVLAController(
            mjcf_path=str(scene_path),
            robot_index=idx,
            object_xy_range=tuple(args.object_xy_range),
            object_z=args.object_z,
            randomize_object_orientation=args.randomize_object_orientation,
            joint_range_fraction=args.joint_range_fraction,
        )
        controller.start_simulation()
        controllers.append(controller)

        auto_policy = AutoPickupPolicy(controller)
        auto_policies.append(auto_policy)

        root = create_mixed_gui(controller, auto_policy)
        if root is None:
            controller.stop()
            return

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
        wrappers.append(wrapper)

        try:
            root.mainloop()
        finally:
            for c in controllers:
                c.stop()
    else:
        # Multi-robot: share a single MuJoCo env/model/data and a single viewer,
        # but provide a horizontally stacked, scrollable GUI panel (and recorder)
        # per robot. Each robot writes to its own dataset root, plus a central
        # controller can broadcast commands to all robots.
        root = tk.Tk()
        root.title("Franka Mixed Control (Multiple Robots)")

        # Scrollable canvas for horizontal stacking of controller panels.
        canvas = tk.Canvas(root)
        h_scroll = ttk.Scrollbar(root, orient="horizontal", command=canvas.xview)
        canvas.configure(xscrollcommand=h_scroll.set)
        canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        h_scroll.grid(row=1, column=0, sticky=(tk.W, tk.E))

        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)

        panels_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=panels_frame, anchor="nw")

        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        panels_frame.bind("<Configure>", _on_frame_configure)

        column = 0
        shared_env = None

        for j, idx in enumerate(robot_indices):
            if j == 0:
                # First robot: create shared env and viewer
                controller = RandomizedVLAController(
                    mjcf_path=str(scene_path),
                    robot_index=idx,
                    object_xy_range=tuple(args.object_xy_range),
                    object_z=args.object_z,
                    randomize_object_orientation=args.randomize_object_orientation,
                    joint_range_fraction=args.joint_range_fraction,
                )
                shared_env = controller.env
                controller.start_simulation()  # single viewer for shared env
            else:
                controller = RandomizedVLAController(
                    mjcf_path=str(scene_path),
                    robot_index=idx,
                    object_xy_range=tuple(args.object_xy_range),
                    object_z=args.object_z,
                    randomize_object_orientation=args.randomize_object_orientation,
                    joint_range_fraction=args.joint_range_fraction,
                    env=shared_env,
                )
            controllers.append(controller)

            auto_policy = AutoPickupPolicy(controller)
            auto_policies.append(auto_policy)

            create_mixed_gui(controller, auto_policy, root=root, parent=panels_frame, column=column)
            panel = panels_frame.winfo_children()[-1]
            column += 1

            # Separate dataset per robot under dataset_root / f"robot{idx}"
            robot_root = (dataset_root / f"robot{idx}").resolve()
            robot_root.mkdir(parents=True, exist_ok=True)
            writer = LeRobotDatasetWriter(
                root=robot_root,
                dataset_name=f"{args.dataset_name}_robot{idx}",
                fps=args.fps,
                cameras=[args.camera],
            )

            wrapper = RecordReplayWrapper(
                controller=controller,
                root=root,
                writer=writer,
                task_text=f"{args.task} (robot {idx})",
                fps=args.fps,
                camera=args.camera,
                video_width=args.video_width,
                video_height=args.video_height,
                controls_parent=panel,
            )
            wrappers.append(wrapper)

        # ------------------------------------------------------------------
        # Central controller: broadcast control to all robots at once
        # ------------------------------------------------------------------
        central_frame = ttk.LabelFrame(root, text="All Robots Control", padding=8)
        central_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=10, pady=(5, 10))

        def all_start_auto() -> None:
            for policy in auto_policies:
                policy.reset()

        def all_stop_auto() -> None:
            for policy in auto_policies:
                policy.stop()

        def all_reset() -> None:
            # Single env/shared data: one reset is sufficient; use first controller.
            if controllers:
                controllers[0].reset_robot()

        def all_start_recording() -> None:
            for w in wrappers:
                try:
                    w.start_recording()
                except Exception:
                    pass

        def all_discard() -> None:
            for w in wrappers:
                try:
                    w.restart_recording()
                except Exception:
                    pass

        def all_save() -> None:
            for w in wrappers:
                try:
                    w.save_recording()
                except Exception:
                    pass

        def all_replay() -> None:
            for w in wrappers:
                try:
                    w.start_replay()
                except Exception:
                    pass

        row = 0
        ttk.Button(central_frame, text="Start Auto Pickup (All)", command=all_start_auto).grid(
            row=row, column=0, padx=4, pady=2, sticky=(tk.W, tk.E)
        )
        ttk.Button(central_frame, text="Stop Auto (All)", command=all_stop_auto).grid(
            row=row, column=1, padx=4, pady=2, sticky=(tk.W, tk.E)
        )
        ttk.Button(central_frame, text="Reset (All)", command=all_reset).grid(
            row=row, column=2, padx=4, pady=2, sticky=(tk.W, tk.E)
        )
        row += 1
        ttk.Button(central_frame, text="Start Recording (All)", command=all_start_recording).grid(
            row=row, column=0, padx=4, pady=2, sticky=(tk.W, tk.E)
        )
        ttk.Button(central_frame, text="Discard (All)", command=all_discard).grid(
            row=row, column=1, padx=4, pady=2, sticky=(tk.W, tk.E)
        )
        ttk.Button(central_frame, text="Save (All)", command=all_save).grid(
            row=row, column=2, padx=4, pady=2, sticky=(tk.W, tk.E)
        )
        ttk.Button(central_frame, text="Replay (All)", command=all_replay).grid(
            row=row, column=3, padx=4, pady=2, sticky=(tk.W, tk.E)
        )

        try:
            root.mainloop()
        finally:
            for c in controllers:
                c.stop()


if __name__ == "__main__":
    main()

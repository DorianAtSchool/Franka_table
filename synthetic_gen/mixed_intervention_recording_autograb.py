"""
Mixed intervention recording variant that uses AutoGrabPolicy instead of
AutoPickupPolicy.

AutoGrabPolicy:
- Drives the end-effector to a graspable distance from the object.
- Once close, closes the gripper.
- Then pulls the object upward toward a target global z-height.
"""

from __future__ import annotations

import argparse
import sys
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
from synthetic_gen.intervention_policies import AutoGrabPolicy  # type: ignore
from synthetic_gen.intervention_guis import create_mixed_gui_autograb  # type: ignore
from datasets.lerobot_writer import LeRobotDatasetWriter  # type: ignore


def _resolve_scene(scene_arg: str) -> Path:
    scene_path = Path(scene_arg)
    if not scene_path.is_absolute():
        scene_path = (REPO_ROOT / scene_arg).resolve()
    return scene_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mixed automatic/manual VLA control with AutoGrabPolicy and randomized starts."
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
        default=str(REPO_ROOT / "datasets" / "franka_table_manual_randomized_vla_autograb"),
        help="Where to store the recorded dataset.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mixed intervention demonstration (autograb)",
        help="Task text for the episode.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="franka_table_manual_randomized_vla_autograb",
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
    auto_policies: list[AutoGrabPolicy] = []
    wrappers: list[RecordReplayWrapper] = []

    # Single-robot case: similar to mixed_intervention_recording.py but with AutoGrabPolicy.
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

        auto_policy = AutoGrabPolicy(controller)
        auto_policies.append(auto_policy)

        root = create_mixed_gui_autograb(controller, auto_policy)
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
        # Multi-robot: single env/model/data and viewer, horizontally stacked panels, per-robot datasets.
        root = tk.Tk()
        root.title("Franka Mixed Control (AutoGrab, Multiple Robots)")

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
                controller = RandomizedVLAController(
                    mjcf_path=str(scene_path),
                    robot_index=idx,
                    object_xy_range=tuple(args.object_xy_range),
                    object_z=args.object_z,
                    randomize_object_orientation=args.randomize_object_orientation,
                    joint_range_fraction=args.joint_range_fraction,
                )
                shared_env = controller.env
                controller.start_simulation()
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

            auto_policy = AutoGrabPolicy(controller)
            auto_policies.append(auto_policy)

            create_mixed_gui_autograb(controller, auto_policy, root=root, parent=panels_frame, column=column)
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

        # Central controls for all robots.
        central_frame = ttk.LabelFrame(root, text="All Robots Control (AutoGrab)", padding=8)
        central_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=10, pady=(5, 10))

        def all_start_auto() -> None:
            for policy in auto_policies:
                policy.reset()

        def all_stop_auto() -> None:
            for policy in auto_policies:
                policy.stop()

        def all_reset() -> None:
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
        ttk.Button(central_frame, text="Start Auto Grab (All)", command=all_start_auto).grid(
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

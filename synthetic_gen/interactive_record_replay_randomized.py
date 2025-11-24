"""
Record/replay wrapper using the randomized interactive GUI.

This script is analogous to ``interactive_record_replay.py`` but uses
``RandomizedInteractiveFrankaGUI``, which shares its MuJoCo model/data with
``RandomizedFrankaTable4RobotsEnv`` for randomized initial states.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import tkinter as tk  # type: ignore
except ImportError:  # pragma: no cover
    tk = None

# Allow importing sibling/parent modules without altering existing files.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from synthetic_gen.interactive_control_gui_randomized import (  # type: ignore
    RandomizedInteractiveFrankaGUI,
    create_gui,
)
from synthetic_gen.interactive_record_replay import (  # type: ignore
    RecordReplayWrapper,
)
from datasets.lerobot_writer import LeRobotDatasetWriter  # type: ignore


def _resolve_scene(scene_arg: str) -> Path:
    scene_path = Path(scene_arg)
    if not scene_path.is_absolute():
        scene_path = (REPO_ROOT / scene_arg).resolve()
    return scene_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive control with randomized starts and record/replay wrapper."
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
        default=str(REPO_ROOT / "datasets" / "franka_table_manual_randomized"),
        help="Where to store the recorded dataset.",
    )
    parser.add_argument("--task", type=str, default="manual demonstration", help="Task text stored with the episode.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="franka_table_manual_randomized",
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

    if tk is None:
        raise RuntimeError("tkinter is required to run the record/replay GUI.")

    scene_path = _resolve_scene(args.scene)
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    dataset_root = Path(args.dataset_root).resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    controller = RandomizedInteractiveFrankaGUI(
        mjcf_path=str(scene_path),
        robot_index=args.robot,
        object_xy_range=tuple(args.object_xy_range),
        object_z=args.object_z,
        randomize_object_orientation=args.randomize_object_orientation,
        joint_range_fraction=args.joint_range_fraction,
    )
    controller.start_simulation()

    root = create_gui(controller)
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


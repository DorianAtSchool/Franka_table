"""
Interactive Franka Robot Control with GUI (Randomized Start)

This module provides a variant of the interactive control GUI that uses the
`RandomizedFrankaTable4RobotsEnv` Gymnasium environment for its MuJoCo model
and data. Compared to `interactive_contol_gui.InteractiveFrankaGUI`, the only
behavioral difference is that calling `reset_robot` randomizes the initial
object pose and robot joint configurations according to the environment
wrapper, instead of resetting to a fixed home pose.

The rest of the GUI (buttons, joint displays, etc.) is identical and
implemented by reusing the original `create_gui` helper.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import mujoco
import numpy as np

try:
    import tkinter as tk  # noqa: F401
    from tkinter import ttk  # noqa: F401
    HAS_TK = True
except ImportError:
    HAS_TK = False
    print("Warning: tkinter not available. Using command-line interface.")

from environments import RandomizedFrankaTable4RobotsEnv
from synthetic_gen.interactive_control_gui import (
    InteractiveFrankaGUI as _BaseInteractiveFrankaGUI,
    create_gui,
)


class RandomizedInteractiveFrankaGUI(_BaseInteractiveFrankaGUI):
    """
    Interactive GUI that uses RandomizedFrankaTable4RobotsEnv for MuJoCo state.

    The interface is the same as InteractiveFrankaGUI, but `reset_robot`
    samples a random initial configuration via the Gymnasium environment.
    """

    def __init__(
        self,
        mjcf_path: str = "scenes/scene_4robots.xml",
        robot_index: int = 0,
        object_xy_range: tuple[float, float, float, float] = (-0.3, 0.3, -0.25, 0.25),
        object_z: float = 0.535,
        randomize_object_orientation: bool = False,
        joint_range_fraction: float = 0.3,
    ) -> None:
        # Instantiate the randomized Gym environment and reuse its model/data.
        self.env = RandomizedFrankaTable4RobotsEnv(
            mjcf_path=mjcf_path,
            render_mode="human",
            object_xy_range=object_xy_range,
            object_z=object_z,
            randomize_object_orientation=randomize_object_orientation,
            joint_range_fraction=joint_range_fraction,
            # Avoid extra mj_step calls inside reset while the simulation
            # thread is also stepping; the base GUI handles stepping.
            extra_settle_steps=0,
        )
        self.model = self.env.model
        self.data = self.env.data

        self.robot_index = robot_index
        self.robot_prefix = f"robot{robot_index + 1}_"

        # Joint information (copied from base class definition).
        self.joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        self.joint_descriptions = [
            "Base Rotation (Yaw)",
            "Shoulder (Pitch)",
            "Elbow Roll",
            "Elbow Pitch",
            "Wrist Roll",
            "Wrist Pitch",
            "Wrist Yaw",
        ]

        # Control parameters
        self.joint_step = 0.05  # Radians per button press (~2.86 degrees)
        self.gripper_step = 20  # Gripper control step

        # Indices for qpos/ctrl segments of the selected robot.
        self.qpos_start = 7 + robot_index * 9
        self.ctrl_start = robot_index * 8

        # Viewer and simulation loop state.
        self.viewer = None
        self.running = True
        self.simulation_thread: threading.Thread | None = None

        # Initialize to a randomized configuration.
        self.reset_robot()

        print(f"Interactive GUI Control (randomized) for Robot {robot_index + 1}")
        print(f"Joint step size: {self.joint_step:.4f} rad ({np.degrees(self.joint_step):.2f} deg)")

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------
    def reset_robot(self) -> None:
        """Reset the scene and robot to a randomized configuration.

        To avoid MuJoCo threading issues (mj_step while the viewer is
        stepping), we temporarily stop the simulation thread and viewer,
        perform the randomized reset on the environment, then restart
        the simulation loop.
        """
        # Check if the simulation thread is currently running.
        was_running = bool(
            getattr(self, "simulation_thread", None)
            and self.simulation_thread.is_alive()
        )

        if was_running:
            # Stop background stepping and close viewer before reset.
            self.stop()

        # Reset MuJoCo state via the randomized environment.
        self.env.reset()
        print("Scene reset to randomized initial configuration.")

        if was_running:
            # Restart simulation loop and viewer.
            self.running = True
            self.start_simulation()


def main() -> None:
    """Entry point for the randomized interactive GUI."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Interactive Franka Robot Control with GUI (Randomized Starts)")
    parser.add_argument(
        "--robot",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Which robot to control (0-3).",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="..\\scenes\\scene_4robots.xml",
        help="Path to the scene XML file (relative to this script).",
    )
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

    # Resolve scene path relative to this script.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scene_path = os.path.join(script_dir, args.scene)

    if not os.path.exists(scene_path):
        print(f"Error: Scene file not found: {scene_path}")
        return

    print(f"Loading scene from: {scene_path}")
    print(f"Controlling Robot {args.robot + 1}")
    print("\nStarting GUI and MuJoCo viewer (randomized starts)...")

    # Create controller using the randomized environment.
    controller = RandomizedInteractiveFrankaGUI(
        mjcf_path=scene_path,
        robot_index=args.robot,
        object_xy_range=tuple(args.object_xy_range),
        object_z=args.object_z,
        randomize_object_orientation=args.randomize_object_orientation,
        joint_range_fraction=args.joint_range_fraction,
    )

    # Start simulation in background thread.
    controller.start_simulation()

    # Wait briefly for viewer to initialize.
    time.sleep(0.5)

    try:
        root = create_gui(controller)
        if root:
            root.mainloop()
        else:
            print("Failed to create GUI window")
            controller.stop()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        controller.stop()
        print("\nCleaning up...")


if __name__ == "__main__":
    main()

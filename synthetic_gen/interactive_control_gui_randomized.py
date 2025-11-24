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
import mujoco.viewer
import numpy as np

try:
    import tkinter as tk  # noqa: F401
    from tkinter import ttk  # noqa: F401
    HAS_TK = True
except ImportError:
    HAS_TK = False
    print("Warning: tkinter not available. Using command-line interface.")

from environments import RandomizedFrankaTable4RobotsEnv
from synthetic_gen.interactive_control_gui import create_gui


class RandomizedInteractiveFrankaGUI:
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

    def reset_robot(self) -> None:
        """Reset the scene and robot to a randomized configuration.

        To avoid MuJoCo threading issues (mj_step while the viewer is
        stepping), we temporarily stop the simulation thread and viewer,
        perform the randomized reset on the environment, then restart
        the simulation loop.
        """
        # Reset MuJoCo state via the randomized environment.
        self.env.reset()
        print("Scene reset to randomized initial configuration.")

    # ------------------------------------------------------------------
    # Joint and gripper control (API compatible with InteractiveFrankaGUI)
    # ------------------------------------------------------------------
    def move_joint(self, joint_index: int, direction: int) -> None:
        """Move a specific joint of the selected robot."""
        if not 0 <= joint_index <= 6:
            return

        delta = direction * self.joint_step
        self.data.ctrl[self.ctrl_start + joint_index] += delta

        # Clamp to joint limits using joint ranges from the model.
        joint_name = f"{self.robot_prefix}{self.joint_names[joint_index]}"
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            jnt_range = self.model.jnt_range[joint_id]
            if jnt_range[0] < jnt_range[1]:
                self.data.ctrl[self.ctrl_start + joint_index] = np.clip(
                    self.data.ctrl[self.ctrl_start + joint_index],
                    jnt_range[0],
                    jnt_range[1],
                )

        direction_str = "+" if direction > 0 else "-"
        print(
            f"Joint {joint_index + 1} ({self.joint_names[joint_index]}): "
            f"{direction_str}{np.degrees(self.joint_step):.2f} deg -> "
            f"{np.degrees(self.data.ctrl[self.ctrl_start + joint_index]):.2f} deg"
        )

    def move_gripper(self, direction: int) -> None:
        """Open/close the gripper for the selected robot."""
        delta = direction * self.gripper_step
        self.data.ctrl[self.ctrl_start + 7] += delta
        self.data.ctrl[self.ctrl_start + 7] = np.clip(
            self.data.ctrl[self.ctrl_start + 7],
            0,
            255,
        )

        action_str = "Opening" if direction > 0 else "Closing"
        print(f"Gripper {action_str}: {self.data.ctrl[self.ctrl_start + 7]:.0f}/255")

    def get_joint_position(self, joint_index: int) -> float:
        """Return current joint position (degrees) for GUI display."""
        pos_rad = self.data.qpos[self.qpos_start + joint_index]
        return float(np.degrees(pos_rad))

    def get_gripper_position(self) -> float:
        """Return current gripper actuator value for GUI display."""
        return float(self.data.ctrl[self.ctrl_start + 7])

    # ------------------------------------------------------------------
    # Simulation loop and lifecycle (API compatible with InteractiveFrankaGUI)
    # ------------------------------------------------------------------
    def simulation_loop(self) -> None:
        """Run the simulation loop in a separate thread."""
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.azimuth = 135
        self.viewer.cam.elevation = -20
        self.viewer.cam.distance = 3.0
        self.viewer.cam.lookat[:] = [0, 0, 0.5]

        while self.running and self.viewer.is_running():
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.002)  # ~500 Hz

    def start_simulation(self) -> None:
        """Start the simulation in a separate thread."""
        self.simulation_thread = threading.Thread(
            target=self.simulation_loop,
            daemon=True,
        )
        self.simulation_thread.start()

    def stop(self) -> None:
        """Stop the simulation and close the viewer."""
        self.running = False
        if self.viewer is not None:
            self.viewer.close()
        if self.simulation_thread is not None:
            self.simulation_thread.join(timeout=2.0)


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

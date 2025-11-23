"""
Wrapper around the interactive control GUI that adds record/replay controls.
Relies on the existing InteractiveFrankaGUI for simulation; captures video
frames directly from its MuJoCo viewer (no separate renderer).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import math
import mujoco
import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk
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

from interactive_contol_gui import InteractiveFrankaGUI, create_gui  # type: ignore
from datasets.lerobot_writer import LeRobotDatasetWriter  # type: ignore


class RecordReplayWrapper:
    """Attach recording and replay controls to an existing Tk root."""

    def __init__(
        self,
        controller: InteractiveFrankaGUI,
        root: tk.Tk,
        writer: LeRobotDatasetWriter,
        task_text: str,
        fps: int = 25,
        camera: str = "side",
        video_width: int = 1280,
        video_height: int = 720,
    ) -> None:
        if tk is None or ttk is None:
            raise RuntimeError("tkinter is required for the recording UI.")

        self.controller = controller
        self.root = root
        self.writer = writer
        self.task_text = task_text
        self.fps = int(fps)
        self.capture_interval_ms = max(1, int(1000 / self.fps))

        self.episode = None
        self.recording = False
        self.replaying = False
        self.recording_start_time: Optional[float] = None
        self.frame_index = 0
        self.replay_frames: List[dict] = []
        self._replay_idx = 0
        self._episode_prefix = ""
        self.video_frames: List[np.ndarray] = []
        self.camera_name = camera
        self.camera_id = None
        self.video_width = int(video_width)
        self.video_height = int(video_height)

        # Trial tracking metadata
        self.trial_counter = 0
        self.initial_object_z: Optional[float] = None

        try:
            self.camera_id = mujoco.mj_name2id(
                controller.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name
            )
        except Exception:
            self.camera_id = None
        # No separate renderer: rely on the GUI viewer for frame capture.

        # UI state
        self.status_var = tk.StringVar(value="Idle (not recording)")
        self.frames_var = tk.StringVar(value="0 frames")

        self._build_controls()
        self.root.after(self.capture_interval_ms, self._capture_loop)

    def _build_controls(self) -> None:
        """Add a compact control panel under the existing GUI."""
        control_frame = ttk.LabelFrame(self.root, text="Record / Replay", padding=8)
        control_frame.grid(row=99, column=0, sticky=(tk.W, tk.E), padx=10, pady=(0, 10))

        btn_start = ttk.Button(control_frame, text="Start Recording", command=self.start_recording)
        btn_start.grid(row=0, column=0, padx=4, pady=4, sticky=(tk.W, tk.E))

        btn_restart = ttk.Button(control_frame, text="Restart (Discard)", command=self.restart_recording)
        btn_restart.grid(row=0, column=1, padx=4, pady=4, sticky=(tk.W, tk.E))

        btn_save = ttk.Button(control_frame, text="Save", command=self.save_recording)
        btn_save.grid(row=0, column=2, padx=4, pady=4, sticky=(tk.W, tk.E))

        btn_replay = ttk.Button(control_frame, text="Replay", command=self.start_replay)
        btn_replay.grid(row=0, column=3, padx=4, pady=4, sticky=(tk.W, tk.E))

        status_label = ttk.Label(control_frame, textvariable=self.status_var, anchor="w")
        status_label.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(6, 2))

        frames_label = ttk.Label(control_frame, textvariable=self.frames_var, anchor="w")
        frames_label.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E))

    def start_recording(self) -> None:
        if self.recording:
            self.status_var.set("Already recording...")
            return

        self.episode = self.writer.start_episode(task_text=self.task_text)
        self._episode_prefix = f"episode_{self.episode.episode_index:06d}"
        self.recording = True
        self.replaying = False
        self.recording_start_time = time.time()
        self.frame_index = 0
        self.video_frames = []
        self.frames_var.set("0 frames")
        self.status_var.set(f"Recording {self._episode_prefix}")

        # Capture initial object z position for success metric
        try:
            self.initial_object_z = float(self.controller.data.qpos[2])
        except Exception:
            self.initial_object_z = None

    def restart_recording(self) -> None:
        self.recording = False
        self.replaying = False
        self.episode = None
        self.replay_frames = []
        self.frame_index = 0
        self.video_frames = []
        # Also reset the robot/environment state.
        try:
            self.controller.reset_robot()
        except Exception:
            pass
        self.frames_var.set("0 frames")
        self.status_var.set("Recording reset; idle")

    def _capture_loop(self) -> None:
        """Sample state/action at the chosen FPS while recording."""
        if self.recording and self.episode is not None and self.recording_start_time is not None:
            timestamp = time.time() - self.recording_start_time
            state = self._get_state_vector()
            action = self._get_action_vector()
            self.episode.add_frame(
                observation_state=state,
                action=action,
                timestamp=timestamp,
                done=False,
                images={},
                frame_index=self.frame_index,
                frames_dir=self.writer.frames_dir,
                episode_prefix=self._episode_prefix,
            )
            self.frame_index += 1
            self.frames_var.set(f"{self.frame_index} frames")
            # Capture camera image for video export using the GUI viewer.
            try:
                viewer = getattr(self.controller, "viewer", None)
                if viewer and hasattr(viewer, "read_pixels"):
                    frame, _ = viewer.read_pixels(width=self.video_width, height=self.video_height, camid=self.camera_id)
                    if frame is not None:
                        if frame.dtype != np.uint8:
                            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
                        self.video_frames.append(frame)
            except Exception:
                pass
        self.root.after(self.capture_interval_ms, self._capture_loop)

    def save_recording(self) -> None:
        if not self.episode or not self.episode.frames:
            self.status_var.set("Nothing to save yet.")
            return

        # Mark the final frame as terminal before writing.
        self.episode.frames[-1]["next.done"] = True
        length = len(self.episode.frames)

        # Compute success metrics if object z position was tracked
        delta_z = None
        success = None
        if self.initial_object_z is not None:
            try:
                final_object_z = float(self.controller.data.qpos[2])
                delta_z = final_object_z - self.initial_object_z
                success = 1 if delta_z > 0.03 else 0
            except Exception:
                pass

        # Current joint positions (achieved target)
        target_joints = None
        try:
            qpos = self.controller.data.qpos
            start = self.controller.qpos_start
            target_joints = qpos[start : start + 7].tolist()
        except Exception:
            pass

        video_relpath = None
        if self.video_frames:
            try:
                video_relpath = self.writer.write_video(
                    camera=self.camera_name,
                    frames=self.video_frames,
                    episode_index=self.episode.episode_index,
                    fps=self.fps,
                    metadata={
                        "trial_index": self.trial_counter,
                        "delta_z": delta_z,
                        "success": success,
                    },
                )
            except Exception as exc:
                self.status_var.set(f"Video save failed: {exc}")

        video_paths = {self.camera_name: video_relpath} if video_relpath else None

        from datetime import datetime

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        trial_info = {
            "run_id": run_id,
            "trial_index": self.trial_counter,
            "delta_z": delta_z,
            "success": success,
            "target_joints": target_joints,
        }

        self.writer.end_episode(
            self.episode,
            length=length,
            task_text=self.task_text,
            video_paths=video_paths,
            trial_info=trial_info,
        )
        self.writer.finalize()
        self.replay_frames = list(self.episode.frames)

        self.recording = False
        status_msg = f"Saved {self._episode_prefix} ({length} frames)"
        if delta_z is not None:
            status_msg += f" | dz={delta_z:+.3f}m | {'SUCCESS' if success else 'FAIL'}"
        self.status_var.set(status_msg)

        self.episode = None
        self.video_frames = []
        self.trial_counter += 1

    def start_replay(self) -> None:
        if not self.replay_frames and self.episode is not None:
            self.replay_frames = list(self.episode.frames)

        if not self.replay_frames:
            self.status_var.set("No recording to replay.")
            return

        self.replaying = True
        self.recording = False
        self._replay_idx = 0
        self.status_var.set("Replaying recorded trajectory...")
        self.root.after(1, self._replay_step)

    def _replay_step(self) -> None:
        if not self.replaying or self._replay_idx >= len(self.replay_frames):
            self.replaying = False
            self.status_var.set("Replay finished.")
            return

        frame = self.replay_frames[self._replay_idx]
        action = np.asarray(frame.get("action", []), dtype=np.float32)
        if action.size > 0:
            act_slice = action[:8]
            ctrl_slice = self.controller.data.ctrl[
                self.controller.ctrl_start : self.controller.ctrl_start + len(act_slice)
            ]
            ctrl_slice[:] = act_slice
        self._replay_idx += 1
        self.root.after(self.capture_interval_ms, self._replay_step)

    def _get_state_vector(self) -> np.ndarray:
        """State = robot joint positions (7) + gripper joints (2)."""
        qpos = self.controller.data.qpos
        start = self.controller.qpos_start
        return np.array(qpos[start : start + 9], dtype=np.float32)

    def _get_action_vector(self) -> np.ndarray:
        """Action = current control targets (7 joints + gripper control)."""
        ctrl = self.controller.data.ctrl
        start = self.controller.ctrl_start
        return np.array(ctrl[start : start + 8], dtype=np.float32)


def _resolve_scene(scene_arg: str) -> Path:
    scene_path = Path(scene_arg)
    if not scene_path.is_absolute():
        scene_path = (SCRIPT_DIR.parent / scene_arg).resolve()
    return scene_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive control with record/replay wrapper.")
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
        default=str(REPO_ROOT / "datasets" / "franka_table_manual"),
        help="Where to store the recorded dataset.",
    )
    parser.add_argument("--task", type=str, default="manual demonstration", help="Task text stored with the episode.")
    parser.add_argument("--dataset-name", type=str, default="franka_table_manual", help="Name stored in info.json.")
    parser.add_argument("--fps", type=int, default=25, help="Recording frames per second.")
    parser.add_argument("--camera", type=str, default="side", help="Camera name to render for video.")
    parser.add_argument("--video-width", type=int, default=1280, help="Video width for captured frames.")
    parser.add_argument("--video-height", type=int, default=720, help="Video height for captured frames.")
    args = parser.parse_args()

    if tk is None:
        raise RuntimeError("tkinter is required to run the record/replay GUI.")

    scene_path = _resolve_scene(args.scene)
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    dataset_root = Path(args.dataset_root).resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    controller = InteractiveFrankaGUI(mjcf_path=str(scene_path), robot_index=args.robot)
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

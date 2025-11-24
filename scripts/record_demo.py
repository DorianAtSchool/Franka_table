import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from pynput import keyboard


from franka_table.environments.franka_4robots_env import FrankaTable4RobotsEnv
from franka_table.datasets.lerobot_writer import LeRobotDatasetWriter
from franka_table.utils.jacobian import jacobian_ik6_step
import mujoco


class KeyTeleop:
    """Simple keyboard teleop for robot1 end-effector 6D twist and gripper.

    Controls:
      - W/S: +X / -X (table frame)
      - A/D: +Y / -Y
      - R/F: +Z / -Z
      - ,/. : roll -/+
      - Arrow keys: pitch (up/down), yaw (left/right)
      - O: open gripper, C: close gripper
      - L: reset episode
      - Q: stop recording (ESC also works)
    """

    def __init__(self, lin_speed: float = 0.10, ang_speed: float = 0.8):
        self.v = np.zeros(3, dtype=np.float32)
        self.grip = 255.0  # 0..255
        self.lin_speed = float(lin_speed)
        self.ang_speed = float(ang_speed)
        self._pressed: Dict[str, bool] = {}
        self._stop = False
        self._reset = False

    def on_press(self, key):
        try:
            k = key.char.lower()
        except Exception:
            k = str(key)
        self._pressed[k] = True
        if k == 'o':
            self.grip = 255.0
        elif k == 'c':
            self.grip = 0.0
        elif k == 'q' or key == keyboard.Key.esc:
            self._stop = True
        elif k == 'l':
            self._reset = True

    def on_release(self, key):
        try:
            k = key.char.lower()
        except Exception:
            k = str(key)
        self._pressed[k] = False
        if k == 'l':
            # Allow new reset on next press
            self._reset = False

    def desired_velocity(self) -> np.ndarray:
        v = np.zeros(3, dtype=np.float32)
        # X axis
        if self._pressed.get('w', False):
            v[0] += self.lin_speed
        if self._pressed.get('s', False):
            v[0] -= self.lin_speed
        # Y axis
        if self._pressed.get('d', False):
            v[1] += self.lin_speed
        if self._pressed.get('a', False):
            v[1] -= self.lin_speed
        # Z axis
        if self._pressed.get('r', False):
            v[2] += self.lin_speed
        if self._pressed.get('f', False):
            v[2] -= self.lin_speed
        return v

    def desired_angular_velocity(self) -> np.ndarray:
        w = np.zeros(3, dtype=np.float32)  # wx, wy, wz
        if self._pressed.get(',', False):
            w[0] -= self.ang_speed  # roll -
        if self._pressed.get('.', False):
            w[0] += self.ang_speed  # roll +
        if self._pressed.get('Key.up', False):
            w[1] += self.ang_speed  # pitch +
        if self._pressed.get('Key.down', False):
            w[1] -= self.ang_speed  # pitch -
        if self._pressed.get('Key.left', False):
            w[2] -= self.ang_speed  # yaw -
        if self._pressed.get('Key.right', False):
            w[2] += self.ang_speed  # yaw +
        return w


def get_robot1_qpos(data: mujoco.MjData) -> np.ndarray:
    return data.qpos[7:14].copy()


def main():
    parser = argparse.ArgumentParser(description="Record a teleoperated demo and export to a LeRobot-like dataset")
    parser.add_argument("--dataset_root", type=str, default="datasets/franka_table_synth")
    parser.add_argument("--task", type=str, help="Natural language instruction for the episode", default="Pick up the object on the table.")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--lin_speed", type=float, default=0.10, help="EE linear speed m/s for teleop")
    parser.add_argument("--cameras", type=str, nargs="+", default=["robot1_wrist", "side"], help="Camera names to record")
    parser.add_argument("--show", action="store_true", help="Show a live preview (offscreen render)", default=True)
    args = parser.parse_args()

    # Locate scene file relative to env module
    pkg_root = Path(__file__).resolve().parents[1]
    env_scene = pkg_root / "scenes" / "scene_4robots.xml"
    # Use offscreen rendering so MuJoCo viewer keyboard shortcuts cannot change lighting/textures.
    render_mode = "human"
    env = FrankaTable4RobotsEnv(mjcf_path=str(env_scene), render_mode=render_mode)
    obs, info = env.reset()

    # Resolve dataset root within franka_table/ if a relative path is given
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = pkg_root / dataset_root
    writer = LeRobotDatasetWriter(root=dataset_root, fps=args.fps, cameras=args.cameras)
    episode = writer.start_episode(task_text=args.task)

    teleop = KeyTeleop(lin_speed=args.lin_speed, ang_speed=0.8)
    listener = keyboard.Listener(on_press=teleop.on_press, on_release=teleop.on_release)
    listener.start()

    # Prepare stepping
    dt = env.control_dt
    sleep_dt = 1.0 / float(args.fps)
    site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "robot1_gripper_center")
    if site_id < 0:
        raise RuntimeError("Missing site 'robot1_gripper_center' in model")

    # Action vector across all robots
    action = np.zeros(env.n_actuators, dtype=np.float32)
    # Initialize arm to current qpos to avoid jumps
    action[:7] = env.data.qpos[7:14].copy()
    action[7] = 255.0  # open gripper initially

    prev_ee_pos = env.data.site_xpos[site_id].copy()

    print("Controls: W/S X, A/D Y, R/F Z | ,/. roll | arrows pitch/yaw | O/C grip | L reset | Q stop")
    print("Recording...")
    t0 = time.time()
    frame_idx = 0
    global_frame = 0
    while not teleop._stop:
        # Handle user-triggered reset: end current episode and start a fresh one
        if teleop._reset:
            if episode.frames:
                episode.frames[-1]["next.done"] = True
                writer.end_episode(episode, length=len(episode.frames), task_text=args.task)
            obs, info = env.reset()
            if args.show:
                env.render()
            episode = writer.start_episode(task_text=args.task)
            action = np.zeros(env.n_actuators, dtype=np.float32)
            action[:7] = env.data.qpos[7:14].copy()
            action[7] = 255.0
            prev_ee_pos = env.data.site_xpos[site_id].copy()
            frame_idx = 0
            teleop._reset = False

        # Desired EE vel in world frame
        v = teleop.desired_velocity()
        w = teleop.desired_angular_velocity()
        dq = jacobian_ik6_step(env.model, env.data, site_id, robot_idx=0, v_des=v, w_des=w, dt=dt, damping=0.1)
        # Update action targets for robot1 arm
        action[:7] = action[:7] + dq.astype(np.float32)
        # Clamp within joint ranges if desired (optional: rely on actuator ctrlrange)
        action[7] = teleop.grip

        obs, reward, terminated, truncated, info = env.step(action)
        
        if args.show:
            env.render()

        # Compute EE delta action for dataset
        ee_pos = env.data.site_xpos[site_id].copy()
        dpos = (ee_pos - prev_ee_pos).astype(np.float32)
        prev_ee_pos = ee_pos
        ee_action = np.zeros(7, dtype=np.float32)
        ee_action[:3] = dpos
        ee_action[3:6] = w.astype(np.float32)
        ee_action[6] = 1.0 if teleop.grip > 128 else -1.0

        # Observation state (robot1 local-ish: 7 qpos, 7 qvel, gripper width approx from finger joints)
        qpos = env.data.qpos[7:14].copy()
        qvel = env.data.qvel[6:13].copy()
        grip_l = env.data.qpos[14]
        grip_r = env.data.qpos[15]
        grip = np.array([grip_l, grip_r], dtype=np.float32)
        obs_state = np.concatenate([qpos, qvel, grip]).astype(np.float32)

        # Render cameras
        images = env.render_cameras(args.cameras, width=args.width, height=args.height)

        # Add frame to episode buffer
        timestamp = frame_idx / float(args.fps)
        done = bool(terminated or truncated)
        episode.add_frame(
            observation_state=obs_state,
            action=ee_action,
            timestamp=timestamp,
            done=done,
            images=images,
            frame_index=frame_idx,
            frames_dir=writer.frames_dir,
            episode_prefix=f"episode_{episode.episode_index:06d}",
        )

        # Real-time pacing for visual fluency
        elapsed = time.time() - t0
        target = (global_frame + 1) * sleep_dt
        if target > elapsed:
            time.sleep(target - elapsed)

        frame_idx += 1
        global_frame += 1

        # If env ended, start a new episode automatically
        if done:
            if episode.frames:
                episode.frames[-1]["next.done"] = True
                writer.end_episode(episode, length=len(episode.frames), task_text=args.task)
            obs, info = env.reset()
            if args.show:
                env.render()
            episode = writer.start_episode(task_text=args.task)
            action = np.zeros(env.n_actuators, dtype=np.float32)
            action[:7] = env.data.qpos[7:14].copy()
            action[7] = 255.0
            prev_ee_pos = env.data.site_xpos[site_id].copy()
            frame_idx = 0

    # Close out final episode if it contains frames
    if episode.frames:
        episode.frames[-1]["next.done"] = True
        writer.end_episode(episode, length=len(episode.frames), task_text=args.task)
    writer.finalize()
    listener.stop()
    env.close()
    print(f"Saved dataset to: {dataset_root.resolve()}")


if __name__ == "__main__":
    main()

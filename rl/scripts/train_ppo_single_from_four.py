#!/usr/bin/env python
"""
Train PPO on the 4-robot Franka scene by controlling a single arm.

Environment: SingleRobotFromFourEnv
 - Wraps FrankaTable4RobotsEnv and exposes a single-agent 8D action space
   (7 joints + 1 gripper) for a selected robot (1..4). Others hold position.

Requirements:
  pip install mujoco gymnasium stable-baselines3[extra] numpy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym

from franka_table.rl.envs.single_from_four import SingleRobotFromFourEnv
from franka_table.config import get_scene_path


def make_env(scene: str, robot_index: int, render_mode: str = "rgb_array") -> gym.Env:
    return SingleRobotFromFourEnv(
        mjcf_path=scene,
        robot_index=robot_index,
        render_mode=render_mode,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO training on 4-robot scene (single arm controller)")
    parser.add_argument("--robot-index", type=int, default=1, choices=[1, 2, 3, 4], help="Which robot to control (1..4)")
    parser.add_argument("--steps", type=int, default=200_000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--logdir", type=Path, default=Path("franka_table/outputs/rl/ppo_single_from_four"))
    parser.add_argument("--save-name", type=str, default="ppo_single_from_four")
    parser.add_argument("--render-mode", type=str, default="rgb_array", choices=["rgb_array", "human"], help="Env render mode")
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO n_steps")
    parser.add_argument("--batch-size", type=int, default=256, help="PPO batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="PPO learning rate")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except Exception as e:
        raise SystemExit(
            "stable-baselines3 is required. Install with: pip install stable-baselines3[extra]\n"
            f"Import error: {e}"
        )

    # Resolve scene path (4-robot scene)
    scene_path = get_scene_path("scene_4robots.xml")

    # Logging dirs
    args.logdir.mkdir(parents=True, exist_ok=True)
    tb_dir = args.logdir / "tb"
    (args.logdir / "ckpts").mkdir(parents=True, exist_ok=True)

    # Build environment
    env = make_env(scene=scene_path, robot_index=args.robot_index, render_mode=args.render_mode)

    # Policy: observation is a flat vector, so use MlpPolicy
    policy = "MlpPolicy"

    model = PPO(
        policy,
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(tb_dir),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        learning_rate=float(args.lr),
    )

    model.learn(total_timesteps=int(args.steps))

    save_path = args.logdir / f"{args.save_name}.zip"
    model.save(str(save_path))
    print(f"Saved policy to {save_path}")


if __name__ == "__main__":
    main()


RL Training (PPO) on Franka Table – 4‑Robot Scene

Overview
- Train a single PPO agent to control one arm in a 4‑robot Franka Panda scene. The other robots hold their positions. The scene and assets are self‑contained under this folder.

Requirements
- Python 3.10+
- Packages: mujoco, gymnasium, stable-baselines3[extra], numpy
- Install: pip install mujoco gymnasium stable-baselines3[extra] numpy

Key Files
- environments/franka_4robots_env.py – Base MuJoCo env (4 robots)
- rl/envs/single_from_four.py – Single‑agent wrapper for one robot
- scenes/scene_4robots.xml – 4‑robot scene (meshes in assets/)
- rl/scripts/train_ppo_single_from_four.py – PPO trainer script

Quick Start
- From the repository root, run:
  python -m franka_table.rl.scripts.train_ppo_single_from_four --robot-index 1 --steps 200000

Defaults
- Scene: scenes/scene_4robots.xml
- Policy: MlpPolicy (vector observation)
- Logging: franka_table/outputs/rl/ppo_single_from_four/

Common Flags
- --robot-index {1,2,3,4}  Select which robot to control (default 1)
- --steps INT               Total training timesteps (default 200000)
- --seed INT                RNG seed (default 7)
- --logdir PATH             Output directory
- --n-steps INT             PPO n_steps (default 2048)
- --batch-size INT          PPO batch size (default 256)
- --lr FLOAT                PPO learning rate (default 3e-4)

What the Wrapper Does
- Exposes a single robot’s 8D action space (7 arm joints + gripper)
- Holds other robots at their current joint positions
- Dense shaping for reach and lift of the object; penalizes robot‑robot contacts

Evaluate a Saved Policy (Example)
- After training, you can run this snippet to roll out a trained model:
  from stable_baselines3 import PPO
  from franka_table.rl.envs.single_from_four import SingleRobotFromFourEnv
  from franka_table.config import get_scene_path

  env = SingleRobotFromFourEnv(mjcf_path=get_scene_path("scene_4robots.xml"), robot_index=1)
  model = PPO.load("franka_table/outputs/rl/ppo_single_from_four/ppo_single_from_four.zip")

  obs, _ = env.reset()
  for _ in range(1000):
      action, _ = model.predict(obs, deterministic=True)
      obs, reward, done, truncated, info = env.step(action)
      if done or truncated:
          obs, _ = env.reset()
  env.close()

Notes
- This folder is trimmed for PPO training on the 4‑robot scene (including single‑arm manipulation).
- Multi‑agent training and interactive control scripts were removed to keep the package focused.

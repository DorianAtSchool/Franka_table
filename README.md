# Franka Table – MuJoCo Franka Panda Tabletop Suite

This package provides MuJoCo scenes and Python helpers for Franka Emika Panda
tabletop manipulation, including:

- A 4‑robot Franka Panda table scene
- Gymnasium environments for single‑arm PPO training
- Scripted pickup/sweep demos for robot 1
- Tools to record and build LeRobot‑style datasets (state only or with images)

Everything under this folder is self‑contained and can be imported as the
`franka_table` Python package from the repository root.

## Requirements

From the repository root (`mujoco_sim/`):

- Python 3.10+
- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

Key libraries used by `franka_table`:

- `mujoco`, `gymnasium`, `stable-baselines3[extra]`, `numpy`
- `imageio`, `mediapy` for rendering/export
- `pynput` for keyboard teleoperation
- `pandas`, `pyarrow`, `datasets`, `torch` (for LeRobot‑style datasets)

## Project Layout

- `config.py` – central path configuration (scenes, outputs, docs)
- `scenes/`
  - `scene_4robots.xml` – four Panda arms around a table
  - `franka_emika_panda/scene_4pandas_table.xml` and related assets
- `environments/`
  - `franka_4robots_env.FrankaTable4RobotsEnv` – base MuJoCo env for the 4‑robot scene
- `rl/`
  - `envs/single_from_four.py` – single‑agent wrapper exposing one robot’s action space
  - `scripts/train_ppo_single_from_four.py` – PPO training script
- `demos/`
  - `robot1_pickup_demo.py` – hand‑tuned pickup + lift demo
  - `robot1_pickup_path_variants.py` – scripted trajectory variants
  - `robot1_pickup_sweep.py` – sweep over random placements to generate many episodes
- `datasets/`
  - `lerobot_writer.py` – utilities to build LeRobot‑style datasets
  - `franka_table_synth/` – example synthetic dataset built from scripted sweeps
- `scripts/`
  - `record_demo.py` – keyboard teleop recorder that writes LeRobot‑style datasets
  - `augment_dataset.py`, `export_video.py`, `replay_randomized.py` – dataset tools
- `outputs/` – default output root (RL logs, figures, etc.)
- `videos/` – pre‑rendered demo videos

`config.py` also exposes helpers such as `get_scene_path("scene_4robots.xml")` and
`get_output_path(...)` so scripts do not need hard‑coded paths.

## Running Scripted Demos

From the repository root (`mujoco_sim/`):

```bash
python franka_table/demos/robot1_pickup_demo.py
```

This loads `scenes/scene_4robots.xml`, controls robot 1 with a scripted policy,
and renders the scene (see the script for camera options and other settings).
The other robots hold their current joint positions.

You can similarly run:

- `python franka_table/demos/robot1_pickup_sweep.py`
- `python franka_table/demos/robot1_pickup_path_variants.py`

These scripts generate `.npz` episodes and demo videos under `datasets/` and
`videos/` for downstream dataset construction.

## PPO Training on the 4‑Robot Scene

`rl/envs/single_from_four.py` wraps the base 4‑robot environment into a
single‑agent Gymnasium env. The agent controls one arm (7 joints + gripper);
the remaining robots hold their positions. Reward shaping encourages reaching,
grasping and lifting while discouraging robot–robot collisions.

Quick start (from `mujoco_sim/`):

```bash
python -m franka_table.rl.scripts.train_ppo_single_from_four \
  --robot-index 1 \
  --steps 200000
```

Defaults:

- Scene: `scenes/scene_4robots.xml`
- Policy: `MlpPolicy` on vector observations
- Logging and checkpoints: `franka_table/outputs/rl/ppo_single_from_four/`

Useful flags:

- `--robot-index {1,2,3,4}` – which robot to control (default 1)
- `--steps INT` – total training timesteps (default `200000`)
- `--seed INT` – RNG seed (default `7`)
- `--logdir PATH` – output directory for logs and models
- `--n-steps INT` – PPO `n_steps` (default `2048`)
- `--batch-size INT` – PPO batch size (default `256`)
- `--lr FLOAT` – PPO learning rate (default `3e-4`)

### Evaluating a Saved PPO Policy

Example snippet for rolling out a trained policy:

```python
from stable_baselines3 import PPO
from franka_table.rl.envs.single_from_four import SingleRobotFromFourEnv
from franka_table.config import get_scene_path

env = SingleRobotFromFourEnv(mjcf_path=get_scene_path("scene_4robots.xml"), robot_index=1)
model = PPO.load("franka_table/outputs/rl/ppo_single_from_four/ppo_single_from_four.zip")

obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
env.close()
```

## Recording Teleoperated Demos (LeRobot‑Style Dataset)

`scripts/record_demo.py` lets you control robot 1 via the keyboard and record
episodes to a LeRobot‑like dataset on disk. Controls (see the docstring for
details):

- `W/S` – +X / −X, `A/D` – −Y / +Y, `R/F` – +Z / −Z
- `,`/`.` – roll −/+; arrow keys – pitch/yaw
- `O` / `C` – open / close gripper
- `L` – reset episode, `Q` or `Esc` – stop recording

Example:

```bash
python franka_table/scripts/record_demo.py \
  --dataset_root datasets/franka_table_synth \
  --task "Pick up the object on the table." \
  --fps 25 --steps 1500
```

This writes:

- Per‑frame data as Parquet under `datasets/franka_table_synth/data/`
- Episode metadata under `datasets/franka_table_synth/meta/`
- Optional videos under `datasets/franka_table_synth/videos/`

The dataset format is compatible with LeRobot‑style training pipelines: you can
point LeRobot configs to `datasets/franka_table_synth` as the dataset root.

## Building Synthetic Datasets from Scripted Sweeps

`datasets/lerobot_writer.py` also provides
`build_dataset_from_sweep_npz(...)`, which turns `.npz` episodes created by
`demos/robot1_pickup_sweep.py` (and related scripts) into a structured
LeRobot‑style dataset (`franka_table_synth/`). At a high level:

1. Run sweep/variant demo scripts to generate `.npz` episodes.
2. Call `build_dataset_from_sweep_npz(...)` from a small Python script,
   pointing it at the `.npz` directory and the MuJoCo XML model.
3. Use the resulting dataset with your imitation / RL training setup.

This makes it easy to go from MuJoCo simulation rollouts to standardized
datasets and RL experiments using the same Franka table scenes.


# Franka Table – MuJoCo Franka Panda Tabletop Suite

This package provides MuJoCo scenes and Python helpers for Franka Emika Panda
tabletop manipulation, including:

- A 4-robot Franka Panda table scene
- Gymnasium environments for single-arm PPO training
- Scripted pickup/sweep demos for robot 1
- Tools to record and build LeRobot-style datasets (state only or with images)

Everything under this folder is self-contained and can be imported as the
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
- `pandas`, `pyarrow`, `datasets`, `torch` (for LeRobot-style datasets)

## Project Layout

- `config.py` – central path configuration (scenes, outputs, docs)
- `scenes/`
  - `scene_4robots.xml` – four Panda arms around a table
  - `franka_emika_panda/scene_4pandas_table.xml` and related assets
- `environments/`
  - `franka_4robots_env.FrankaTable4RobotsEnv` – base MuJoCo env for the 4-robot scene
- `rl/`
  - `envs/single_from_four.py` – single-agent wrapper exposing one robot’s action space
  - `scripts/train_ppo_single_from_four.py` – PPO training script
- `demos/`
  - `robot1_pickup_demo.py` – hand-tuned pickup + lift demo
  - `robot1_pickup_path_variants.py` – scripted trajectory variants
  - `robot1_pickup_sweep.py` – sweep over random placements to generate many episodes
- `datasets/`
  - `lerobot_writer.py` – utilities to build LeRobot-style datasets
  - `franka_table_synth/` – example synthetic dataset built from scripted sweeps
- `scripts/`
  - `record_demo.py` – keyboard teleop recorder that writes LeRobot-style datasets
  - `augment_dataset.py`, `export_video.py`, `replay_randomized.py` – dataset tools
- `outputs/` – default output root (RL logs, figures, etc.)
- `videos/` – pre-rendered demo videos

`config.py` also exposes helpers such as `get_scene_path("scene_4robots.xml")` and
`get_output_path(...)` so scripts do not need hard-coded paths.

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

These scripts generate synthetic episodes and demo videos under `datasets/` and
`videos/` for downstream dataset construction.

## PPO Training on the 4-Robot Scene

`rl/envs/single_from_four.py` wraps the base 4-robot environment into a
single-agent Gymnasium env. The agent controls one arm (7 joints + gripper);
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

## Recording Teleoperated Demos (LeRobot-Style Dataset)

`scripts/record_demo.py` lets you control robot 1 via the keyboard and record
episodes to a LeRobot-like dataset on disk. Controls (see the docstring for
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

- Per-frame data as Parquet under `datasets/franka_table_synth/data/`
- Episode metadata under `datasets/franka_table_synth/meta/`
- Optional videos under `datasets/franka_table_synth/videos/`

The dataset format is compatible with LeRobot-style training pipelines: you can
point LeRobot configs to `datasets/franka_table_synth` as the dataset root.

## Synthetic Dataset Generation via Sweep

The main synthetic dataset is produced directly by
`demos/robot1_pickup_sweep.py`, which drives MuJoCo and writes into
`LeRobotDatasetWriter` on each simulation frame (no `.npz` intermediate).

High-level flow in `run_sweep(...)`:

- Load a 4-robot scene XML (prefers `scenes/scene_4robots.xml` but falls back
  to `scenes/scene_4robots_real.xml` or `scenes/franka_emika_panda/scene_4pandas_table.xml`).
- Initialize the scene (`init_scene_state`) so all four robots and the object
  match the pickup demo initial state.
- Construct a `LeRobotDatasetWriter(root=dataset_root, cameras=views)` with
  `dataset_root` typically `datasets/franka_table_synth`.
- For each trial:
  - Sample a noisy target joint configuration around `BASE_TARGET_JOINTS`.
  - Start a new episode with `writer.start_episode(...)` and keep the returned
    episode buffer and its `episode_index`.
  - Prepare MuJoCo renderers for the requested camera views (side/top/wrist).
  - Call `create_pickup_trajectory(...)` with an `on_frame` callback that:
    - Renders all requested camera views from the *same* MuJoCo state.
    - Computes an end-effector state
      `state = [x, y, z, qx, qy, qz, qw]` from the EE site pose.
    - Computes a 7-D action as finite differences between consecutive states:
      `[dx, dy, dz, droll, dpitch, dyaw, dgrip]` using quaternion→RPY via
      `quat_to_rpy(...)`.
    - Packs the latest RGB frames per camera into an `images` dict
      (`{"side": ..., "top": ..., "wrist": ...}` for the enabled views).
    - Calls `episode.add_frame(...)` with
      `observation_state=state`, `action=action`, `timestamp=data.time`,
      `done=False` (final frame is marked later), and the `images` dict.
  - After the rollout, mark the last frame’s `next.done=True` and close the
    episode with `writer.end_episode(...)`.
  - Save per-view `.mp4` videos with filenames that encode trial index,
    camera view, and lift height `delta_z`, and append a row to a run-level
    `summary.csv` in `videos/robot1_pickup_sweeps/run_*/`.
  - Append a JSON line to `datasets/franka_table_synth/meta/trial_mapping.jsonl`
    linking `trial_index` → `episode_index`, lift metrics, target joints and
    relative video/data paths.
- After all trials, call `writer.finalize()` to flush metadata, including
  `meta/info.json`, `meta/episodes/chunk-000/file-000.parquet` and
  `episodes.jsonl`.

On disk, the resulting dataset under `datasets/franka_table_synth` follows a
LeRobot-style layout:

- Per-frame data: `data/chunk-000/episode_XXXXXX.parquet`
- Episode metadata: `meta/episodes/chunk-000/file-000.parquet`
- Global index: `episodes.jsonl`, `meta/info.json`, `meta/tasks.parquet`
- Optional videos: `videos/<camera>/...` plus `meta/trial_mapping.jsonl`

This design lets downstream LeRobot or imitation-learning code treat
`datasets/franka_table_synth` as a standard dataset, while `robot1_pickup_sweep`
efficiently reuses a single MuJoCo rollout per trial to generate both RGB
videos and structured trajectories.

## Randomized Environments, VLA Control, and Mixed Interventions

In addition to scripted sweeps and keyboard teleop, `franka_table` includes a
set of Gymnasium-backed interactive tools under `synthetic_gen/` that share
MuJoCo state with the training environments.

### Randomized 4-Robot Env

- `environments/franka_4robots_env.py` defines:
  - `FrankaTable4RobotsEnv` �?" base MuJoCo env with fixed start pose.
  - `RandomizedFrankaTable4RobotsEnv` �?" wrapper that randomizes object pose
    (XY region, fixed Z, optional random orientation) and robot joint angles
    within configurable fractions of their limits.

The randomized env uses Gymnasium’s `Env.reset` for seeding and then applies
MuJoCo state changes directly to `MjData`, ensuring compatibility with the
original 4-robot scene while providing diverse starts for data collection.

### VLA-Style Interactive GUIs

`synthetic_gen/interactive_record_replay_randomized_vla.py` introduces a
VLA-style controller and GUI:

- `RandomizedVLAController`:
  - Wraps `RandomizedFrankaTable4RobotsEnv` and exposes VLA-style controls in
    **end-effector space**.
  - Uses `utils/jacobian.jacobian_ik6_step` to convert a desired 6D twist at
    the gripper site into joint deltas for the chosen robot.
  - Shares a single `MjModel`/`MjData` instance between env and GUI to avoid
    drift and duplication.

- The VLA GUI exposes:
  - Position deltas: `dx, dy, dz` in world frame via buttons.
  - Rotation deltas: `droll, dpitch, dyaw` (radians) via buttons.
  - Gripper open/close via a single scalar actuator in `[0, 255]`.

`RecordReplayWrapper` + `LeRobotDatasetWriter` are reused so that each frame
records:

- `observation.state = [x, y, z, qx, qy, qz, qw]` (EE pose).
- `action = [dx, dy, dz, droll, dpitch, dyaw, dgrip]` derived from finite
  differences between consecutive states, regardless of whether the underlying
  motion came from GUI or scripted logic.

### Mixed Automatic / Manual Intervention Recording

For mixed-control data collection, use:

- `synthetic_gen/mixed_intervention_recording.py`

Key components:

- `RandomizedVLAController` �?" as above, shared MuJoCo state with randomized env.
- `AutoPickupPolicy` �?" a phased pick-up routine operating in EE space:
  1. Move above the object by a configurable height.
  2. Move down near the object.
  3. Close the gripper.
  4. Lift the object.
  5. Hold, then stop.
- `create_mixed_gui` �?" Tk GUI that lets you:
  - Start/stop the automatic pick-up policy.
  - Intervene manually with VLA-style position/rotation/gripper buttons
    (disabled while auto is running to avoid conflicts).
  - Randomize/reset the underlying environment state.

The mixed recorder uses the same `RecordReplayWrapper` and `LeRobotDatasetWriter`
as the other interactive tools, writing a LeRobot-style dataset under
`datasets/franka_table_manual_randomized_vla_mixed` by default. Automatic
motions and human interventions are both represented as consistent
VLA-style actions in the recorded trajectories.

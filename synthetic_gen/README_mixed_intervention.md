# Mixed Intervention Recording (Randomized VLA)

This module adds a **mixed automatic / manual intervention** recording setup on
top of the randomized 4-robot Franka table environment.

It combines:

- `RandomizedFrankaTable4RobotsEnv` (Gymnasium env with randomized starts)
- `RandomizedVLAController` (VLA-style end-effector control, shared MuJoCo state)
- `AutoPickupPolicy` (scripted pick-up trajectory in EE space)
- Manual GUI controls for VLA-style deltas (`dx, dy, dz, droll, dpitch, dyaw, dgrip`)
- `RecordReplayWrapper` + `LeRobotDatasetWriter` for LeRobot-style datasets

The core script is:

- `mixed_intervention_recording.py`

---

## Concepts

### Randomized VLA Controller

`RandomizedVLAController` (defined in
`interactive_record_replay_randomized_vla.py`) wraps
`RandomizedFrankaTable4RobotsEnv` and exposes end-effector controls:

- Internally uses `jacobian_ik6_step` from `utils/jacobian.py` to map a desired
  6D twist `(v, w)` at the gripper site into joint deltas for the selected
  robot.
- Shares `model` / `data` with the Gym env, so the GUI and env are always
  consistent.

The interface is compatible with `RecordReplayWrapper`, which expects:

- `controller.data` (MuJoCo data)
- `controller.qpos_start`, `controller.ctrl_start`
- `controller.get_gripper_position()`
- `controller.simulation_loop()`, `start_simulation()`, `stop()`

### AutoPickupPolicy

Defined in `mixed_intervention_recording.py`, `AutoPickupPolicy` implements a
simple phased pick-up routine in EE space:

1. **Phase 0** – Move EE above the object by `offset_z_high` (default 0.20 m).
2. **Phase 1** – Move down near the object by `offset_z_low` (default 0.03 m).
3. **Phase 2** – Close the gripper.
4. **Phase 3** – Lift the object up by `offset_z_lift` (default 0.30 m).
5. **Phase 4** – Hold; policy marks itself idle.

Each `step()`:

- Reads current object position from `data.qpos[0:3]`.
- Reads current EE position from the gripper site in `data.site_xpos`.
- Computes a small clamped delta toward the current phase target, using
  `0.5 * controller.pos_step` as a maximum per-step motion.
- Calls `controller.apply_ee_delta(dpos, drot, dgrip)` so that both automatic
  and manual interventions write compatible VLA-style actions.

The policy is **resume-aware**:

- `reset()` will start a **new** sequence only if the previous run finished
  (`phase >= 4`). Otherwise it resumes from the current phase using whatever
  EE/object state you reached via manual intervention.

### Mixed GUI

`create_mixed_gui(controller, auto_policy)` provides a Tk GUI that exposes:

- Automatic controls:
  - `Start Auto Pickup` – calls `auto_policy.reset()`.
  - `Stop Auto` – calls `auto_policy.stop()`.
  - Status label: `Auto: running (phase X)` / `Auto: idle`.
- Manual VLA-style controls (when auto is **not** running):
  - Position: `X±`, `Y±`, `Z±` (world frame, `controller.pos_step` per click).
  - Rotation: `Roll±`, `Pitch±`, `Yaw±` (`controller.rot_step` per click).
  - Gripper: `Open`, `Close` (`controller.gripper_step` per click).
  - `Reset (randomized)` – calls `controller.reset_robot()` to sample a new
    randomized start from the env.

While the auto policy is running:

- All manual motion / gripper buttons are disabled and guarded in code, so
  only the automatic VLA commands are applied.

The GUI periodically:

- Calls `auto_policy.step()` at ~25 Hz while auto is active.
- Updates gripper readout and auto-status labels.

---

## Recording Dataset

`mixed_intervention_recording.py` uses `RecordReplayWrapper` +
`LeRobotDatasetWriter` exactly like `interactive_record_replay.py`:

- For each frame:
  - `observation.state = [x, y, z, qx, qy, qz, qw]` – EE pose in world frame.
  - `action = [dx, dy, dz, droll, dpitch, dyaw, dgrip]` – finite differences
    between consecutive EE states and gripper, so both automatic and manual
    interventions are recorded uniformly.
  - Timestamps, `next.done`, and optional camera video frames.

Default dataset root and name:

- Root: `datasets/franka_table_manual_randomized_vla_mixed`
- Dataset name in `meta/info.json`:
  - `"dataset_name": "franka_table_manual_randomized_vla_mixed"`

Layout follows the standard LeRobot-style format:

- `data/chunk-000/episode_XXXXXX.parquet`
- `meta/episodes/chunk-000/file-000.parquet`
- `meta/info.json`, `meta/tasks.parquet`, `episodes.jsonl`
- `videos/<camera>/...` and `meta/trial_mapping.jsonl` for videos.

---

## Usage

From the repository root `mujoco_sim/`:

```bash
python -m franka_table.synthetic_gen.mixed_intervention_recording \
  --robot 0 \
  --scene scenes/scene_4robots.xml \
  --fps 25 \
  --camera side
```

Key arguments:

- `--robot {0,1,2,3}` – which robot to control (0-based index).
- `--scene PATH` – MuJoCo scene XML, relative to repo root or absolute.
- `--dataset-root PATH` – output dataset root (default as above).
- `--dataset-name STR` – stored in `meta/info.json`.
- `--fps INT` – recording FPS used for timestamps and video.
- `--camera NAME` – camera name for videos (default `side`).
- `--object-xy-range X_MIN X_MAX Y_MIN Y_MAX` – object sampling region.
- `--object-z FLOAT` – fixed object Z at reset.
- `--randomize-object-orientation` – sample a random object quaternion at reset.
- `--joint-range-fraction FLOAT` – fraction of each joint’s range used when
  randomizing robot joints.

During a run:

- Use **Start Auto Pickup** to let the scripted policy attempt a lift.
- Hit **Stop Auto** at any time to intervene manually with the VLA buttons.
- You can restart auto pickup after intervening; the policy will continue from
  the current state, not the original starting pose.


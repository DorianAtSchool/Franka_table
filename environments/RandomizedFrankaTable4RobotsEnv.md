# RandomizedFrankaTable4RobotsEnv

`RandomizedFrankaTable4RobotsEnv` is a thin wrapper around
`FrankaTable4RobotsEnv` that **only changes the initial configuration** of the
4-robot MuJoCo scene. The action/observation spaces, reward, and `step`
semantics are identical to the base environment.

Use this wrapper when you want episodes to start from **random robot joint
configurations and random object placements** on the table, while keeping
compatibility with any existing code that expects the original env API.

## Import

From the repository root (where `franka_table` is importable as a package):

```python
from franka_table.environments import RandomizedFrankaTable4RobotsEnv
```

or, if you are working from within the `franka_table` package:

```python
from environments import RandomizedFrankaTable4RobotsEnv
```

## Constructor

```python
env = RandomizedFrankaTable4RobotsEnv(
    mjcf_path="scene_4pandas_table.xml",
    render_mode="human",
    control_dt=0.02,
    physics_dt=0.002,
    object_xy_range=(-0.3, 0.3, -0.25, 0.25),
    object_z=0.535,
    randomize_object_orientation=False,
    joint_range_fraction=0.3,
)
```

Arguments (in addition to those from `FrankaTable4RobotsEnv`):

- `object_xy_range`: `(x_min, x_max, y_min, y_max)` range on the table
  surface from which the cube's `(x, y)` is sampled uniformly on each reset.
- `object_z`: Fixed `z` height for the cube (default matches the base env).
- `randomize_object_orientation`: If `True`, sample a random unit quaternion
  `(w, x, y, z)` for the cube orientation on reset. If `False`, keep the
  default upright orientation `[1, 0, 0, 0]`.
- `joint_range_fraction`: Fraction of each joint's physical range to sample
  from, centered around the midpoint of the joint limits. For example, `0.3`
  means "sample within the middle 30% of the allowed range" for each joint.

## Reset Behavior

`RandomizedFrankaTable4RobotsEnv.reset`:

1. Seeds the environment RNG via Gymnasium's `Env.reset` and clears MuJoCo
   state to a valid default configuration.
2. Randomizes the **object pose**:
   - Samples `x, y` uniformly from `object_xy_range`.
   - Sets `z = object_z`.
   - If `randomize_object_orientation=True`, samples a random unit quaternion.
   - Otherwise, sets quaternion to `[1, 0, 0, 0]` (upright).
3. Randomizes the **arm joints** for each of the 4 robots:
   - For each joint `robot{i}_joint{j}` (7 per arm), looks up its joint limits
     in `model.jnt_range`.
   - Samples uniformly from the middle `joint_range_fraction` band around the
     midpoint of those limits.
   - If a joint does not have a valid range, falls back to adding small noise
     around the current joint value.
   - Keeps both gripper finger joints open at `0.04` as in the base env.
4. Synchronizes MuJoCo state (`mj_forward`) and steps the physics for a few
   substeps to let the system settle.
5. Returns `(obs, info)`:
   - `obs` has the same structure and dimensionality as in
     `FrankaTable4RobotsEnv` (80-D vector).
   - `info` includes the base env's info plus two extra keys:
     - `initial_object_position`: copy of `qpos[0:3]` after randomization.
     - `initial_object_orientation`: copy of `qpos[3:7]` after randomization.

## Usage Example

Basic rollout with randomized starts:

```python
from franka_table.environments import RandomizedFrankaTable4RobotsEnv

env = RandomizedFrankaTable4RobotsEnv(
    mjcf_path="scenes/scene_4robots.xml",
    render_mode="human",
    object_xy_range=(-0.25, 0.25, -0.2, 0.2),
    randomize_object_orientation=False,
    joint_range_fraction=0.4,
)

obs, info = env.reset(seed=123)
print("Initial object pos:", info["initial_object_position"])
print("Initial object quat:", info["initial_object_orientation"])

for t in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Compatibility Notes

- `RandomizedFrankaTable4RobotsEnv` **does not modify** the base
  `FrankaTable4RobotsEnv`. Existing scripts that import or construct the base
  env behave exactly as before.
- The wrapper preserves:
  - `action_space`, `observation_space`
  - `step` signature and reward definition
  - `render` and camera utilities
- Any RL code or evaluation script that currently expects
  `FrankaTable4RobotsEnv` can be switched to randomized starts by simply
  constructing `RandomizedFrankaTable4RobotsEnv` instead, without needing to
  change policy architectures or training loops.

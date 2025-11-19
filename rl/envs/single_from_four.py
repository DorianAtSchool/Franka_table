import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from franka_table.environments.franka_4robots_env import FrankaTable4RobotsEnv


class SingleRobotFromFourEnv(gym.Env):
    """
    Single-agent Gymnasium wrapper around the 4-robot environment.

    - Controls only one robot (robot_index in {1,2,3,4}).
    - Action: 8D (7 joints + 1 gripper) for the selected robot; others receive zeros.
    - Observation: 32D = own robot (16) + shared object+goal (16), consistent with PettingZoo wrapper.
    - Reward/termination: forwarded from the underlying cooperative env.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, mjcf_path: str, robot_index: int = 1, render_mode: str = "rgb_array"):
        assert robot_index in (1, 2, 3, 4), "robot_index must be 1..4"
        self._env = FrankaTable4RobotsEnv(mjcf_path=mjcf_path, render_mode=render_mode)
        self.robot_index = robot_index - 1  # 0-based
        self.render_mode = render_mode

        # Actions are normalized; we map to joint position deltas (rad) and gripper command delta (0..255)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32)

        # Control mapping parameters
        self._joint_step = np.array([0.05] * 7, dtype=np.float32)  # rad per step at action=1.0
        self._grip_step = 25.0  # ctrl units per step at action=1.0 (0..255)

        # Joint limits (Franka)
        self._joint_limits = np.array([
            [-2.8973, 2.8973],
            [-1.7628, 1.7628],
            [-2.8973, 2.8973],
            [-3.0718, -0.0698],
            [-2.8973, 2.8973],
            [-0.0175, 3.7525],
            [-2.8973, 2.8973],
        ], dtype=np.float32)

        # Gripper site for reach reward
        site_name = f"robot{self.robot_index+1}_gripper_center"
        self._eef_site_id = mujoco.mj_name2id(self._env.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self._z_init = None

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        # cache initial object height for lift shaping
        self._z_init = float(self._env.data.qpos[2])
        return self._select_obs(obs), info

    def step(self, action):
        # Build full ctrl: hold all other robots at current positions, apply deltas to selected robot
        data = self._env.data
        full = np.zeros((self._env.n_actuators,), dtype=np.float32)

        # Fill with holds: set joint targets to current qpos for each robot, keep gripper ctrl unchanged
        for i in range(4):
            qpos_start = 7 + i * 9
            ctrl_start = i * 8
            cur_q = data.qpos[qpos_start : qpos_start + 7].astype(np.float32)
            full[ctrl_start : ctrl_start + 7] = cur_q
            full[ctrl_start + 7] = float(data.ctrl[ctrl_start + 7])

        # Apply selected robot action mapping: joint deltas + gripper delta
        i = self.robot_index
        qpos_start = 7 + i * 9
        ctrl_start = i * 8
        cur_q = data.qpos[qpos_start : qpos_start + 7].astype(np.float32)
        dq = np.asarray(action[:7], dtype=np.float32) * self._joint_step
        target_q = np.clip(cur_q + dq, self._joint_limits[:, 0], self._joint_limits[:, 1])
        full[ctrl_start : ctrl_start + 7] = target_q

        # Gripper: map [-1,1] to delta in [−_grip_step, +_grip_step], clamp to [0,255]
        cur_g = float(data.ctrl[ctrl_start + 7])
        dg = float(action[7]) * self._grip_step
        full[ctrl_start + 7] = float(np.clip(cur_g + dg, 0.0, 255.0))

        obs, _, terminated, truncated, info = self._env.step(full)

        # Dense reward shaping
        object_pos = self._env.data.qpos[0:3]
        # For pickup we do not use object-to-goal distance as a reward component here
        goal_pos = np.array([0.15, -0.15, 0.48], dtype=np.float32)
        object_to_goal = float(np.linalg.norm(object_pos - goal_pos))

        # End-effector reach distance
        if self._eef_site_id >= 0:
            eef_pos = self._env.data.site_xpos[self._eef_site_id]
            reach_dist = float(np.linalg.norm(eef_pos - object_pos))
        else:
            reach_dist = 0.0

        # Lift progress (over 10cm)
        z_init = self._z_init if self._z_init is not None else float(object_pos[2])
        lift = float(np.clip((object_pos[2] - z_init) / 0.10, 0.0, 1.0))

        # Reward components
        # Stronger reach reward: ~2.0 when within grasp distance, fades out by 10 cm
        grasp_radius = 0.05  # 5 cm: considered graspable proximity
        near_radius = 0.10   # 10 cm: outer reach shaping bound
        if reach_dist <= grasp_radius:
            r_reach = 2.0
        else:
            # Linear fade from 2.0 at 5 cm to 0.0 at 10 cm
            r_reach = 2.0 * np.clip((near_radius - reach_dist) / (near_radius - grasp_radius), 0.0, 1.0)

        r_lift = lift  # [0,1]

        # Base reward without goal distance (pickup-focused)
        reward = 1.0 * r_reach + 0.7 * r_lift - 0.05 * reach_dist

        # Harsh penalty if object is dropped below table height threshold
        if object_pos[2] < 0.30:
            reward -= 8.0

        # Penalize contacts with other robots (harsh)
        contact_penalty = 0.0
        n_rob_contacts = 0
        m = self._env.model
        d = self._env.data
        for k in range(d.ncon):
            con = d.contact[k]
            g1, g2 = int(con.geom1), int(con.geom2)
            b1 = int(m.geom_bodyid[g1])
            b2 = int(m.geom_bodyid[g2])
            name1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1) or ""
            name2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2) or ""

            def which_robot(nm: str):
                if nm.startswith("robot1_"): return 0
                if nm.startswith("robot2_"): return 1
                if nm.startswith("robot3_"): return 2
                if nm.startswith("robot4_"): return 3
                return None

            r1 = which_robot(name1)
            r2 = which_robot(name2)
            if r1 is not None and r2 is not None and r1 != r2:
                # robot-robot contact
                if r1 == self.robot_index or r2 == self.robot_index:
                    n_rob_contacts += 1

        if n_rob_contacts > 0:
            contact_penalty = -5.0 * n_rob_contacts
            reward += contact_penalty

        info = dict(info)
        info.update({
            "reach_dist": reach_dist,
            "lift": lift,
            "object_to_goal_distance": object_to_goal,
            "gripper_ctrl": full[ctrl_start + 7],
            "robot_robot_contacts": n_rob_contacts,
        })

        return self._select_obs(obs), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    # helpers
    def _select_obs(self, flat_obs: np.ndarray) -> np.ndarray:
        # flat_obs: [robots(4*16)=64, shared(16)]
        i = self.robot_index
        own = flat_obs[i * 16 : (i + 1) * 16]
        shared = flat_obs[64:]
        return np.concatenate([own, shared]).astype(np.float32)

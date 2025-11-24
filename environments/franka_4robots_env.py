import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class FrankaTable4RobotsEnv(gym.Env):
    """
    Franka Panda multi-robot environment with 4 robots and a moveable object.
    
    Four Franka Panda robots, one at each corner of the table.
    Each robot has 8 actuators:
    - 7 joint actuators for the arm (joint1-joint7)
    - 1 gripper actuator (controls both fingers via tendon)
    
    Total: 32 actuators (8 per robot × 4 robots)
    
    The environment includes:
    - A table centered at origin (1.2m x 1.0m x 0.48m height)
    - A blue square cube (7cm sides) on the table
    - A transparent green goal marker indicating target position
    - 4 Franka robots, one at each corner of the table
    
    Task: Cooperatively move the blue cube to the green goal marker location.
    
    Reward:
    - Penalty based on distances from each gripper to object (encourages reaching)
    - Large penalty for object-to-goal distance (main objective)
    - +10 bonus when object is within 5cm of goal
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, mjcf_path: str = "scene_4pandas_table.xml", 
                 render_mode="human", control_dt=0.02, physics_dt=0.002):
        # Convert Path to string if needed
        if isinstance(mjcf_path, Path):
            mjcf_path = str(mjcf_path)
        
        # mjcf_path = "franka_table/scenes/franka_emika_panda/scene_4pandas_table.xml"  # Hardcoded path for this example
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.control_dt = control_dt
        self.n_substeps = int(control_dt // physics_dt)
        self.render_mode = render_mode
        self.viewer = None
        self._camera_renderers: Dict[Tuple[str, int, int], mujoco.Renderer] = {}
        
        self.num_robots = 4
        self.actuators_per_robot = 8

        # Action space: 32 actuators (8 per robot × 4 robots)
        self.n_actuators = self.model.nu
        assert self.n_actuators == 32, f"Expected 32 actuators, got {self.n_actuators}"
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_actuators,), dtype=np.float32)
        
        # Observation space: 
        # Per robot (×4):
        #   - Robot joint positions (7)
        #   - Robot joint velocities (7)
        #   - Gripper state (2 finger joints)
        # Shared:
        #   - Object position (3)
        #   - Object orientation (4 quaternion)
        #   - Object linear velocity (3)
        #   - Object angular velocity (3)
        #   - Goal position (3)
        obs_dim = 4 * (7 + 7 + 2) + 3 + 4 + 3 + 3 + 3  # 80 total
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Robot prefixes for accessing their components
        self.robot_prefixes = [f"robot{i+1}_" for i in range(self.num_robots)]
        
        print(f"Franka Table 4-Robots Environment initialized:")
        print(f"  - Number of robots: {self.num_robots}")
        print(f"  - Number of actuators: {self.n_actuators} (8 per robot)")
        print(f"  - Number of bodies: {self.model.nbody}")
        print(f"  - Number of joints (nq): {self.model.nq}")
        print(f"  - Number of velocities (nv): {self.model.nv}")
        print(f"  - Observation dimension: {obs_dim}")
        print(f"  - Action dimension: {self.n_actuators}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial pose
        # Object position (slightly offset from center)
        self.data.qpos[0:3] = [-0.05, -0.05, 0.535]  # x, y, z
        self.data.qpos[3:7] = [1, 0, 0, 0]  # quaternion (w, x, y, z)
        
        # Set robot arm joints to home position for all robots
        for i in range(self.num_robots):
            start_idx = 7 + i * 9  # Object (7 DOF) + robot i joints
            # Home position for arm (7 joints)
            self.data.qpos[start_idx:start_idx+7] = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
            # Gripper fingers (open position, 2 joints)
            self.data.qpos[start_idx+7:start_idx+9] = [0.04, 0.04]
        
        # Set control to match initial joint positions
        for i in range(self.num_robots):
            ctrl_start = i * 8
            qpos_start = 7 + i * 9
            # Arm controls
            self.data.ctrl[ctrl_start:ctrl_start+7] = self.data.qpos[qpos_start:qpos_start+7].copy()
            # Gripper control (open = 255)
            self.data.ctrl[ctrl_start+7] = 255
        
        # Let the simulation settle
        mujoco.mj_forward(self.model, self.data)
        
        # Run for a few steps to let physics stabilize
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Ensure action has correct shape
        if action.shape != (self.n_actuators,):
            raise ValueError(f"Expected action shape {(self.n_actuators,)}, got {action.shape}")
        
        # Apply action to actuators
        self.data.ctrl[:] = action
        
        # Step the simulation for multiple substeps
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        
        # Get observation
        obs = self._get_obs()
        
        # Get positions
        object_pos = self.data.qpos[0:3]
        goal_pos = np.array([0.15, -0.15, 0.48])  # Goal marker position
        
        # Compute reward components
        gripper_distances = []
        for i in range(self.num_robots):
            prefix = self.robot_prefixes[i]
            gripper_site_name = f"{prefix}gripper_site"
            gripper_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, gripper_site_name)
            if gripper_site_id >= 0:
                gripper_pos = self.data.site_xpos[gripper_site_id]
                dist = np.linalg.norm(gripper_pos - object_pos)
                gripper_distances.append(dist)
        
        # Distance from object to goal (main objective)
        object_to_goal = np.linalg.norm(object_pos - goal_pos)
        
        # Combined reward: penalty for all gripper distances, large penalty for object-goal distance
        avg_gripper_distance = np.mean(gripper_distances) if gripper_distances else 0
        reward = -0.05 * avg_gripper_distance - 1.0 * object_to_goal
        
        # Bonus for getting object close to goal
        if object_to_goal < 0.05:
            reward += 10.0  # Success bonus
        
        # Check termination
        terminated = object_pos[2] < 0.3  # Object fell below table
        
        # Success condition
        success = object_to_goal < 0.05
        
        truncated = False
        
        info = {
            "object_position": object_pos.copy(),
            "goal_position": goal_pos.copy(),
            "object_to_goal_distance": object_to_goal,
            "gripper_distances": gripper_distances,
            "success": success,
        }
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """Get the current observation."""
        # Object state (position, orientation, velocities)
        object_pos = self.data.qpos[0:3].copy()
        object_quat = self.data.qpos[3:7].copy()
        object_linvel = self.data.qvel[0:3].copy()
        object_angvel = self.data.qvel[3:6].copy()
        
        # Goal position
        goal_pos = np.array([0.15, -0.15, 0.48])
        
        # Robot states
        robot_obs = []
        for i in range(self.num_robots):
            qpos_start = 7 + i * 9  # Object (7) + previous robots
            qvel_start = 6 + i * 9  # Object freejoint (6) + previous robots
            
            # Robot joint positions and velocities
            robot_qpos = self.data.qpos[qpos_start:qpos_start+7].copy()  # 7 arm joints
            robot_qvel = self.data.qvel[qvel_start:qvel_start+7].copy()  # 7 arm joint velocities
            
            # Gripper state
            gripper_qpos = self.data.qpos[qpos_start+7:qpos_start+9].copy()  # 2 finger joints
            
            robot_obs.extend(robot_qpos)
            robot_obs.extend(robot_qvel)
            robot_obs.extend(gripper_qpos)
        
        obs = np.concatenate([
            robot_obs,
            object_pos,
            object_quat,
            object_linvel,
            object_angvel,
            goal_pos,
        ])
        
        return obs.astype(np.float32)

    def render(self):
        if self.render_mode == "rgb_array":
            if self.viewer is None:
                self.viewer = mujoco.Renderer(self.model, height=480, width=640)
            
            self.viewer.update_scene(self.data)
            return self.viewer.render()
        
        else:
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                self.viewer.sync()

    def render_camera(self, camera_name: str, width: int = 640, height: int = 480) -> np.ndarray:
        """Render a specific MuJoCo camera by name to an RGB array.

        Caches per-(camera,width,height) renderers for performance.
        """
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise ValueError(f"Camera '{camera_name}' not found in model.")
        key = (camera_name, height, width)
        if key not in self._camera_renderers:
            self._camera_renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        renderer = self._camera_renderers[key]
        # Select camera via update_scene; render() has no camera args in some versions
        renderer.update_scene(self.data, camera=camera_name)
        return renderer.render()

    def render_cameras(self, camera_names: List[str], width: int = 640, height: int = 480) -> Dict[str, np.ndarray]:
        """Render multiple cameras and return dict of name->RGB array."""
        return {name: self.render_camera(name, width=width, height=height) for name in camera_names}

    def close(self):
        if self.viewer is not None:
            if self.render_mode == "human":
                self.viewer.close()
            self.viewer = None


def test_env():
    """Test the Franka 4-robots environment."""
    import os
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scene_path = os.path.join(script_dir, "scene_4robots.xml")
    
    print(f"Loading scene from: {scene_path}")
    
    # Create environment
    env = FrankaTable4RobotsEnv(mjcf_path=scene_path, render_mode="human")
    
    # Reset environment
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial observation (first 20 values): {obs[:20]}")
    
    # Run simulation
    print("\nRunning simulation...")
    for i in range(1000):
        # Random action for all robots
        action = env.action_space.sample() * 0.05  # Small random actions
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.render()
        
        if i % 100 == 0:
            print(f"Step {i}: reward={reward:.3f}, object_to_goal={info['object_to_goal_distance']:.3f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {i}")
            obs, info = env.reset()
    
    env.close()
    print("Test complete!")


if __name__ == "__main__":
    test_env()

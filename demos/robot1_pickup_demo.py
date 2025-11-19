"""
Robot 1 Pick-up Demonstration

This script creates a video of Robot 1 (top-right) picking up the box
from the center of the table and lifting it.
"""

import mujoco
import numpy as np
import mediapy as media
import os


def create_pickup_trajectory(
    model,
    data,
    camera_view: str = "side",
    camera=None,
    target_joints=None,
    record_trajectory: bool = False,
    on_frame=None,
    frame_update_interval: int = 3,
):
    """
    Create a trajectory for Robot 1 to pick up the box.
    
    Args:
        camera_view: "side" for side view or "top" for top-down view
        target_joints: Custom joint angles (radians). If None, uses default verified positions.
    
    Returns:
        frames: List of RGB frames for video

    Notes:
        - record_trajectory is kept for backward compatibility but is no longer used.
        - If on_frame is not None, it will be called as on_frame(model, data)
          at every recorded video frame, so you can stream data into a writer.
    """
    frames = []
    
    # Create renderer once
    renderer = mujoco.Renderer(model, height=720, width=1280)
    
    # Set up camera view based on requested view
    if camera is None:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        
        if camera_view == "top":
            # Top-down view of the table
            camera.azimuth = 90
            camera.elevation = -89  # Almost directly from above
            camera.distance = 3.5
            camera.lookat[:] = [0.0, 0.0, 0.5]
            print(f"Using TOP-DOWN camera view as main view")
        elif camera_view == "side":  # side view
            # Side view aligned with Y-axis, looking straight at the table from the side
            # Azimuth 90 = looking along +X axis (from right side of table)
            # This is perpendicular to the table, between Robot 1 and Robot 4
            camera.azimuth = 90
            camera.elevation = -10
            camera.distance = 2.5
            camera.lookat[:] = [0.0, 0.0, 0.5]
            print(f"Using SIDE camera view as main view")
        elif camera_view == "wrist":
            print("using WRIST camera view as main view")
        else:
            print("Unknown camera view")

    # Robot 1 control indices (first robot: indices 0-7)
    ctrl_start = 0
    qpos_start = 7  # After object's 7 DOFs
    
    # Robot 1 is at position (0.6, 0.5, 0.48) facing inward
    # Object is a high-friction rubber cylinder at (0.0, 0.0, 0.53) - exact center of table
    # Cylinder dimensions: radius=0.025m (50mm), height=0.10m (100mm)
    # High friction (2.0) makes it very easy to grip
    # Horizontal distance: ~0.78m, well within robot reach (~0.855m max reach)
    
    # Verified working joint positions from interactive session (converted from degrees to radians):
    # Joint 1: 74.48° = 1.2999 rad
    # Joint 2: 97.40° = 1.7000 rad
    # Joint 3: 2.86° = 0.0499 rad
    # Joint 4: -29.84° = -0.5208 rad
    # Joint 5: 94.54° = 1.6501 rad
    # Joint 6: 107.19° = 1.8710 rad
    # Joint 7: -73.64° = -1.2852 rad
    
    # Use provided target joints or default verified positions
    if target_joints is None:
        target_joints = [1.2999, 1.5000, 0.0499, -0.5208, 1.6501, 1.8710, -1.2852]
    else:
        target_joints = list(target_joints)  # Convert to list if numpy array
    
    # Phase 1: Move to grasp position (700 outer steps, each with 5 mj_step)
    print("Phase 1: Moving to grasp position...")
    initial_joints = data.ctrl[ctrl_start:ctrl_start+7].copy()
    
    for i in range(700):
        t = i / 700.0
        # Smooth interpolation
        smooth_t = 3*t**2 - 2*t**3
        data.ctrl[ctrl_start:ctrl_start+7] = (1-smooth_t) * initial_joints + smooth_t * np.array(target_joints)
        data.ctrl[ctrl_start+7] = 255  # Keep gripper open
        
        for _ in range(5):
            mujoco.mj_step(model, data)

        if i % frame_update_interval == 0:  # Record every frame_update_interval frame for smoother video
            renderer.update_scene(data, camera)
            frame = renderer.render()
            frames.append(frame)
            if on_frame is not None:
                on_frame(model, data, frame)

    # Phase 2: Hold position briefly before grasping (200 steps - slightly longer)
    print("Phase 2: Holding position before grasp...")
    for i in range(200):
        mujoco.mj_step(model, data)

        if i % frame_update_interval == 0:
            renderer.update_scene(data, camera)
            frame = renderer.render()
            frames.append(frame)
            if on_frame is not None:
                on_frame(model, data, frame)
    
    # Phase 3: Close gripper (100 steps - slowed down)
    print("Phase 3: Closing gripper...")
    for i in range(100):
        t = i / 100.0
        smooth_t = 3*t**2 - 2*t**3
        data.ctrl[ctrl_start+7] = 255 * (1 - smooth_t)  # Close gripper
        
        mujoco.mj_step(model, data)

        if i % frame_update_interval == 0:
            renderer.update_scene(data, camera)
            frame = renderer.render()
            frames.append(frame)
            if on_frame is not None:
                on_frame(model, data, frame)
    
    # Phase 4: Lift up by moving Joint 2 up (150 steps - slowed down)
    print("Phase 4: Lifting object up...")
    # Lift by decreasing Joint 2 angle (moving shoulder up)
    # From interactive session: Joint 2 went from 97.40° to 63.03° = 1.7000 to 1.1002 rad
    target_joints_up = target_joints.copy()
    target_joints_up[1] = 1.1002  # Move Joint 2 from 1.7000 to 1.1002 (lift up)
    current_joints = data.ctrl[ctrl_start:ctrl_start+7].copy()
    
    for i in range(150):
        t = i / 150.0
        smooth_t = 3*t**2 - 2*t**3
        data.ctrl[ctrl_start:ctrl_start+7] = (1-smooth_t) * current_joints + smooth_t * np.array(target_joints_up)
        data.ctrl[ctrl_start+7] = 0  # Keep gripper closed
        
        mujoco.mj_step(model, data)
        
        if i % frame_update_interval == 0:
            renderer.update_scene(data, camera)
            frame = renderer.render()
            frames.append(frame)
            if on_frame is not None:
                on_frame(model, data, frame)
    
    # Phase 5: Hold for a moment (60 steps - slightly longer)
    print("Phase 5: Holding lifted position...")
    for i in range(60):
        mujoco.mj_step(model, data)

        if i % frame_update_interval == 0:
            renderer.update_scene(data, camera)
            frame = renderer.render()
            frames.append(frame)
            if on_frame is not None:
                on_frame(model, data, frame)
    
    return frames


def setup_camera(model):
    """Add a fixed camera to the model for better viewing angle."""
    # Note: Camera should ideally be added to XML, but we'll use default camera
    pass


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scene_path = os.path.join("franka_table", "scenes", "scene_4robots.xml")
    
    if not os.path.exists(scene_path):
        print(f"Error: Scene file not found: {scene_path}")
        return
    
    print("Loading scene...")
    model = mujoco.MjModel.from_xml_path(scene_path)
    
    # Generate SIDE VIEW video
    print("\n" + "="*60)
    print("GENERATING SIDE VIEW VIDEO")
    print("="*60)
    
    data = mujoco.MjData(model)
    
    # Initialize scene
    print("Initializing scene...")
    mujoco.mj_resetData(model, data)
    
    # Set object position (center of table, matches env/top-view)
    data.qpos[0:3] = [0.0, 0.0, 0.535]  # x, y, z
    data.qpos[3:7] = [1, 0, 0, 0]  # quaternion (no rotation)
    
    # Initialize all robots to home position
    for i in range(4):
        start_idx = 7 + i * 9
        data.qpos[start_idx:start_idx+7] = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
        data.qpos[start_idx+7:start_idx+9] = [0.04, 0.04]
        
        ctrl_start = i * 8
        data.ctrl[ctrl_start:ctrl_start+7] = data.qpos[start_idx:start_idx+7].copy()
        data.ctrl[ctrl_start+7] = 255  # Open gripper
    
    mujoco.mj_forward(model, data)
    
    print("\nStarting pickup demonstration (side view)...")
    print("This will take a moment to render...")
    
    # Create the trajectory and capture frames - SIDE VIEW
    frames_side = create_pickup_trajectory(model, data, camera_view="side")
    
    # Save side view video
    output_path_side = os.path.join(script_dir, "robot1_pickup_demo_side.mp4")
    print(f"\nSaving side view video to: {output_path_side}")
    media.write_video(output_path_side, frames_side, fps=30)
    
    print(f"✓ Side view video saved successfully!")
    print(f"  - Total frames: {len(frames_side)}")
    print(f"  - Duration: {len(frames_side)/30:.1f} seconds")
    print(f"  - Resolution: 1280x720")
    
    # Generate TOP VIEW video
    print("\n" + "="*60)
    print("GENERATING TOP VIEW VIDEO")
    print("="*60)
    
    # Reset data for second video
    data = mujoco.MjData(model)
    
    print("Initializing scene...")
    mujoco.mj_resetData(model, data)
    
    # Set object position (center of table)
    data.qpos[0:3] = [0.0, 0.0, 0.535]
    data.qpos[3:7] = [1, 0, 0, 0]
    
    # Initialize all robots to home position
    for i in range(4):
        start_idx = 7 + i * 9
        data.qpos[start_idx:start_idx+7] = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
        data.qpos[start_idx+7:start_idx+9] = [0.04, 0.04]
        
        ctrl_start = i * 8
        data.ctrl[ctrl_start:ctrl_start+7] = data.qpos[start_idx:start_idx+7].copy()
        data.ctrl[ctrl_start+7] = 255  # Open gripper
    
    mujoco.mj_forward(model, data)
    
    print("\nStarting pickup demonstration (top view)...")
    print("This will take a moment to render...")
    
    # Create the trajectory and capture frames - TOP VIEW
    frames_top = create_pickup_trajectory(model, data, camera_view="top")
    
    # Save top view video
    output_path_top = os.path.join(script_dir, "robot1_pickup_demo_top.mp4")
    print(f"\nSaving top view video to: {output_path_top}")
    media.write_video(output_path_top, frames_top, fps=30)
    
    print(f"✓ Top view video saved successfully!")
    print(f"  - Total frames: {len(frames_top)}")
    print(f"  - Duration: {len(frames_top)/30:.1f} seconds")
    print(f"  - Resolution: 1280x720")
    
    print("\n" + "="*60)
    print("BOTH VIDEOS COMPLETE!")
    print("="*60)
    print(f"Side view: {output_path_side}")
    print(f"Top view:  {output_path_top}")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        if "mediapy" in str(e):
            print("\nError: mediapy is not installed.")
            print("Please install it with: pip install mediapy")
        else:
            raise
    except Exception as e:
        print(f"\nError: {e}")
        raise

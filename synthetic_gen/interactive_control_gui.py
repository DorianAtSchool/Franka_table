"""
Interactive Franka Robot Control with GUI

This script provides a GUI window with buttons to control each joint of one robot
in the 4-robot environment. You can test positive and negative movements
for each of the 7 arm joints and the gripper.

The GUI shows:
- Buttons for each joint (+/- movement)
- Current joint position display
- Reset button
- Real-time visualization in MuJoCo viewer
"""

import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
try:
    import tkinter as tk
    from tkinter import ttk
    HAS_TK = True
except ImportError:
    HAS_TK = False
    print("Warning: tkinter not available. Using command-line interface.")


class InteractiveFrankaGUI:
    def __init__(self, mjcf_path: str = "scenes/scene_4robots.xml", robot_index: int = 0):
        """
        Initialize the interactive control environment with GUI.
        
        Args:
            mjcf_path: Path to the MuJoCo scene XML file
            robot_index: Which robot to control (0-3)
        """
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.robot_index = robot_index
        self.robot_prefix = f"robot{robot_index + 1}_"
        
        # Joint information
        self.joint_names = [
            "joint1", "joint2", "joint3", "joint4", 
            "joint5", "joint6", "joint7"
        ]
        self.joint_descriptions = [
            "Base Rotation (Yaw)",
            "Shoulder (Pitch)",
            "Elbow Roll",
            "Elbow Pitch",
            "Wrist Roll",
            "Wrist Pitch",
            "Wrist Yaw"
        ]
        
        # Control parameters
        self.joint_step = 0.05  # Radians per button press (~2.86 degrees)
        self.gripper_step = 20   # Gripper control step
        
        # Calculate indices
        self.qpos_start = 7 + robot_index * 9
        self.ctrl_start = robot_index * 8
        
        # Initialize viewer in separate thread
        self.viewer = None
        self.running = True
        self.simulation_thread = None
        
        # Reset to home position
        self.reset_robot()
        
        print(f"Interactive GUI Control for Robot {robot_index + 1}")
        print(f"Joint step size: {self.joint_step:.4f} rad ({np.degrees(self.joint_step):.2f} deg)")
        
    def reset_robot(self):
        """Reset the robot to home position."""
        mujoco.mj_resetData(self.model, self.data)
        
        # Object position
        self.data.qpos[0:3] = [-0.05, -0.05, 0.535]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        
        # Reset all robots to home position
        for i in range(4):
            start_idx = 7 + i * 9
            self.data.qpos[start_idx:start_idx+7] = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
            self.data.qpos[start_idx+7:start_idx+9] = [0.04, 0.04]
            
            ctrl_start = i * 8
            self.data.ctrl[ctrl_start:ctrl_start+7] = self.data.qpos[start_idx:start_idx+7].copy()
            self.data.ctrl[ctrl_start+7] = 255  # Open gripper
        
        mujoco.mj_forward(self.model, self.data)
        print("✓ Robot reset to home position")
        
    def move_joint(self, joint_index: int, direction: int):
        """Move a specific joint."""
        if not 0 <= joint_index <= 6:
            return
        
        # Update control target
        delta = direction * self.joint_step
        self.data.ctrl[self.ctrl_start + joint_index] += delta
        
        # Clamp to joint limits
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, 
            f"{self.robot_prefix}{self.joint_names[joint_index]}"
        )
        if joint_id >= 0:
            jnt_range = self.model.jnt_range[joint_id]
            if jnt_range[0] < jnt_range[1]:
                self.data.ctrl[self.ctrl_start + joint_index] = np.clip(
                    self.data.ctrl[self.ctrl_start + joint_index],
                    jnt_range[0], jnt_range[1]
                )
        
        direction_str = "+" if direction > 0 else "-"
        print(f"Joint {joint_index + 1} ({self.joint_names[joint_index]}): "
              f"{direction_str}{np.degrees(self.joint_step):.2f}° → "
              f"{np.degrees(self.data.ctrl[self.ctrl_start + joint_index]):.2f}°")
        
    def move_gripper(self, direction: int):
        """Move the gripper."""
        delta = direction * self.gripper_step
        self.data.ctrl[self.ctrl_start + 7] += delta
        self.data.ctrl[self.ctrl_start + 7] = np.clip(
            self.data.ctrl[self.ctrl_start + 7], 0, 255
        )
        
        action_str = "Opening" if direction > 0 else "Closing"
        print(f"Gripper {action_str}: {self.data.ctrl[self.ctrl_start + 7]:.0f}/255")
        
    def get_joint_position(self, joint_index: int):
        """Get current position of a joint in degrees."""
        pos_rad = self.data.qpos[self.qpos_start + joint_index]
        return np.degrees(pos_rad)
    
    def get_gripper_position(self):
        """Get current gripper control value."""
        return self.data.ctrl[self.ctrl_start + 7]
        
    def simulation_loop(self):
        """Run the simulation loop in a separate thread."""
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.azimuth = 135
        self.viewer.cam.elevation = -20
        self.viewer.cam.distance = 3.0
        self.viewer.cam.lookat[:] = [0, 0, 0.5]
        
        while self.running and self.viewer.is_running():
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.002)  # ~500 Hz
            
    def start_simulation(self):
        """Start the simulation in a separate thread."""
        self.simulation_thread = threading.Thread(target=self.simulation_loop, daemon=True)
        self.simulation_thread.start()
        
    def stop(self):
        """Stop the simulation."""
        self.running = False
        if self.viewer:
            self.viewer.close()
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2.0)


def create_gui(controller):
    """Create the GUI window."""
    if not HAS_TK:
        print("Error: tkinter is required for GUI mode")
        return None
        
    root = tk.Tk()
    root.title(f"Franka Robot {controller.robot_index + 1} Control")
    root.geometry("600x700")
    
    # Style
    style = ttk.Style()
    style.configure("TButton", padding=5, font=("Arial", 10))
    style.configure("Title.TLabel", font=("Arial", 12, "bold"))
    style.configure("Joint.TLabel", font=("Arial", 10))
    
    # Main container
    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Title
    title = ttk.Label(main_frame, text=f"Robot {controller.robot_index + 1} Joint Control", 
                     style="Title.TLabel")
    title.grid(row=0, column=0, columnspan=4, pady=(0, 10))
    
    # Info label
    info = ttk.Label(main_frame, 
                    text=f"Step size: {np.degrees(controller.joint_step):.2f}° per click",
                    font=("Arial", 9, "italic"))
    info.grid(row=1, column=0, columnspan=4, pady=(0, 15))
    
    # Joint labels for real-time display
    joint_value_labels = []
    
    # Create controls for each joint
    row = 2
    for i, (name, desc) in enumerate(zip(controller.joint_names, controller.joint_descriptions)):
        # Joint label
        joint_label = ttk.Label(main_frame, text=f"Joint {i+1}: {desc}", 
                               style="Joint.TLabel", width=25, anchor="w")
        joint_label.grid(row=row, column=0, sticky=tk.W, pady=5, padx=5)
        
        # Negative button
        btn_neg = ttk.Button(main_frame, text="◄ -", width=8,
                            command=lambda idx=i: controller.move_joint(idx, -1))
        btn_neg.grid(row=row, column=1, padx=2)
        
        # Position display
        value_label = ttk.Label(main_frame, text="0.00°", width=10, 
                               font=("Courier", 10), anchor="center",
                               relief="sunken")
        value_label.grid(row=row, column=2, padx=2)
        joint_value_labels.append((i, value_label))
        
        # Positive button
        btn_pos = ttk.Button(main_frame, text="+ ►", width=8,
                            command=lambda idx=i: controller.move_joint(idx, +1))
        btn_pos.grid(row=row, column=3, padx=2)
        
        row += 1
    
    # Separator
    sep = ttk.Separator(main_frame, orient="horizontal")
    sep.grid(row=row, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=15)
    row += 1
    
    # Gripper control
    gripper_label = ttk.Label(main_frame, text="Gripper Control", 
                             style="Joint.TLabel", width=25, anchor="w")
    gripper_label.grid(row=row, column=0, sticky=tk.W, pady=5, padx=5)
    
    btn_close = ttk.Button(main_frame, text="◄ Close", width=8,
                          command=lambda: controller.move_gripper(-1))
    btn_close.grid(row=row, column=1, padx=2)
    
    gripper_value_label = ttk.Label(main_frame, text="255", width=10,
                                   font=("Courier", 10), anchor="center",
                                   relief="sunken")
    gripper_value_label.grid(row=row, column=2, padx=2)
    
    btn_open = ttk.Button(main_frame, text="Open ►", width=8,
                         command=lambda: controller.move_gripper(+1))
    btn_open.grid(row=row, column=3, padx=2)
    row += 1
    
    # Separator
    sep2 = ttk.Separator(main_frame, orient="horizontal")
    sep2.grid(row=row, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=15)
    row += 1
    
    # Reset button
    btn_reset = ttk.Button(main_frame, text="↺ Reset",
                          command=controller.reset_robot)
    btn_reset.grid(row=row, column=0, columnspan=4, pady=10)
    row += 1
    
    # Instructions
    instructions = tk.Text(main_frame, height=8, width=60, font=("Arial", 9),
                          wrap=tk.WORD, relief="sunken", borderwidth=1)
    instructions.grid(row=row, column=0, columnspan=4, pady=10)
    instructions.insert("1.0", """Instructions:
• Click - or + buttons to move each joint incrementally
• Watch the MuJoCo viewer to see the robot move in real-time
• Position values update to show current joint angles
• Use Close/Open buttons for gripper control
• Click Reset to return robot to home position

Joint Functions:
• Joint 1 (Base): Rotates the entire arm left/right
• Joint 2 (Shoulder): Raises/lowers the arm
• Joint 3 (Elbow Roll): Twists the elbow
• Joint 4 (Elbow Pitch): Bends the elbow up/down
• Joint 5-7 (Wrist): Control end-effector orientation""")
    instructions.config(state="disabled")
    
    # Update loop for joint values
    def update_values():
        if controller.running:
            # Update joint positions
            for joint_idx, label in joint_value_labels:
                pos = controller.get_joint_position(joint_idx)
                label.config(text=f"{pos:6.2f}°")
            
            # Update gripper
            gripper_val = controller.get_gripper_position()
            gripper_value_label.config(text=f"{gripper_val:.0f}")
            
            # Schedule next update
            root.after(100, update_values)  # Update every 100ms
    
    # Start updating values
    root.after(100, update_values)
    
    # Handle window close
    def on_closing():
        controller.stop()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    return root


def main():
    """Main function to run the interactive GUI control."""
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Franka Robot Control with GUI")
    parser.add_argument("--robot", type=int, default=0, choices=[0, 1, 2, 3],
                       help="Which robot to control (0-3)")
    parser.add_argument("--scene", type=str, default="..\scenes\scene_4robots.xml",
                       help="Path to the scene XML file")
    args = parser.parse_args()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scene_path = os.path.join(script_dir, args.scene)
    
    if not os.path.exists(scene_path):
        print(f"Error: Scene file not found: {scene_path}")
        return
    
    print(f"Loading scene from: {scene_path}")
    print(f"Controlling Robot {args.robot + 1}")
    print("\nStarting GUI and MuJoCo viewer...")
    
    # Create controller
    controller = InteractiveFrankaGUI(mjcf_path=scene_path, robot_index=args.robot)
    
    # Start simulation in background thread
    controller.start_simulation()
    
    # Wait a moment for viewer to initialize
    time.sleep(0.5)
    
    try:
        # Create and run GUI
        root = create_gui(controller)
        if root:
            root.mainloop()
        else:
            print("Failed to create GUI window")
            controller.stop()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        controller.stop()
        print("\nCleaning up...")


if __name__ == "__main__":
    main()

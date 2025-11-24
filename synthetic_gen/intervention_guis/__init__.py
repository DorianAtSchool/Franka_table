from __future__ import annotations

import numpy as np

try:
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except ImportError:  # pragma: no cover
    tk = None
    ttk = None

from synthetic_gen.interactive_record_replay_randomized_vla import (  # type: ignore
    RandomizedVLAController,
)
from synthetic_gen.intervention_policies import AutoPickupPolicy, AutoGrabPolicy  # type: ignore


def create_mixed_gui(
    controller: RandomizedVLAController,
    auto_policy: AutoPickupPolicy,
    root: tk.Tk | None = None,
    parent: tk.Widget | None = None,
    column: int = 0,
) -> tk.Tk | None:
    """GUI with both automatic and manual VLA-style controls (AutoPickupPolicy)."""
    if tk is None or ttk is None:
        print("Error: tkinter is required for mixed intervention GUI mode")
        return None

    own_root = False
    if root is None:
        root = tk.Tk()
        root.title(f"Robot {controller.robot_index + 1} Mixed Control (Randomized)")
        root.geometry("460x520")
        own_root = True

    container = parent if parent is not None else root

    frame = ttk.Frame(container, padding="10", borderwidth=2, relief="groove")
    frame.grid(row=0, column=column, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

    title = ttk.Label(
        frame,
        text=f"Robot {controller.robot_index + 1} VLA Mixed Control",
        font=("Arial", 12, "bold"),
    )
    title.grid(row=0, column=0, columnspan=3, pady=(0, 10))

    info = ttk.Label(
        frame,
        text=f"Pos step: {controller.pos_step:.3f} m | Rot step: {np.degrees(controller.rot_step):.1f} deg",
        font=("Arial", 9, "italic"),
    )
    info.grid(row=1, column=0, columnspan=3, pady=(0, 10))

    auto_status = tk.StringVar(value="Auto: idle")
    manual_buttons: list[ttk.Button] = []

    def update_auto_status() -> None:
        running = auto_policy.is_running()
        if running:
            auto_status.set(f"Auto: running (phase {auto_policy.phase})")
        else:
            auto_status.set("Auto: idle")

        for btn in manual_buttons:
            try:
                btn.configure(state="disabled" if running else "normal")
            except Exception:
                pass

        root.after(200, update_auto_status)

    status_label = ttk.Label(frame, textvariable=auto_status, font=("Arial", 9, "italic"))
    status_label.grid(row=2, column=0, columnspan=3, pady=(0, 10))

    def do_move(dpos, drot, dgrip=0.0) -> None:
        if auto_policy.is_running():
            return
        controller.apply_ee_delta(np.array(dpos, dtype=np.float32), np.array(drot, dtype=np.float32), dgrip)

    row = 3
    ttk.Label(frame, text="Automatic pick-up", font=("Arial", 10, "bold")).grid(
        row=row, column=0, columnspan=3, pady=(5, 5)
    )
    row += 1

    def start_auto() -> None:
        auto_policy.reset()

    def stop_auto() -> None:
        auto_policy.stop()

    ttk.Button(frame, text="Start Auto Pickup", command=start_auto).grid(
        row=row, column=0, columnspan=2, padx=2, pady=2, sticky=(tk.W, tk.E)
    )
    ttk.Button(frame, text="Stop Auto", command=stop_auto).grid(
        row=row, column=2, padx=2, pady=2, sticky=(tk.W, tk.E)
    )
    row += 1

    ttk.Label(frame, text="Manual EE Position", font=("Arial", 10, "bold")).grid(
        row=row, column=0, columnspan=3, pady=(10, 5)
    )
    row += 1

    btn_x_neg = ttk.Button(frame, text="X-", width=8, command=lambda: do_move((-controller.pos_step, 0, 0), (0, 0, 0)))
    btn_x_neg.grid(row=row, column=0, padx=2, pady=2)
    btn_x_pos = ttk.Button(frame, text="X+", width=8, command=lambda: do_move((controller.pos_step, 0, 0), (0, 0, 0)))
    btn_x_pos.grid(row=row, column=2, padx=2, pady=2)
    manual_buttons.extend([btn_x_neg, btn_x_pos])
    ttk.Label(frame, text="X axis").grid(row=row, column=1)
    row += 1

    btn_y_neg = ttk.Button(frame, text="Y-", width=8, command=lambda: do_move((0, -controller.pos_step, 0), (0, 0, 0)))
    btn_y_neg.grid(row=row, column=0, padx=2, pady=2)
    btn_y_pos = ttk.Button(frame, text="Y+", width=8, command=lambda: do_move((0, controller.pos_step, 0), (0, 0, 0)))
    btn_y_pos.grid(row=row, column=2, padx=2, pady=2)
    manual_buttons.extend([btn_y_neg, btn_y_pos])
    ttk.Label(frame, text="Y axis").grid(row=row, column=1)
    row += 1

    btn_z_neg = ttk.Button(frame, text="Z-", width=8, command=lambda: do_move((0, 0, -controller.pos_step), (0, 0, 0)))
    btn_z_neg.grid(row=row, column=0, padx=2, pady=2)
    btn_z_pos = ttk.Button(frame, text="Z+", width=8, command=lambda: do_move((0, 0, controller.pos_step), (0, 0, 0)))
    btn_z_pos.grid(row=row, column=2, padx=2, pady=2)
    manual_buttons.extend([btn_z_neg, btn_z_pos])
    ttk.Label(frame, text="Z axis").grid(row=row, column=1)
    row += 1

    ttk.Label(frame, text="Manual EE Rotation", font=("Arial", 10, "bold")).grid(
        row=row, column=0, columnspan=3, pady=(10, 5)
    )
    row += 1

    r = controller.rot_step
    btn_roll_neg = ttk.Button(frame, text="Roll-", width=8, command=lambda: do_move((0, 0, 0), (-r, 0, 0)))
    btn_roll_neg.grid(row=row, column=0, padx=2, pady=2)
    btn_roll_pos = ttk.Button(frame, text="Roll+", width=8, command=lambda: do_move((0, 0, 0), (r, 0, 0)))
    btn_roll_pos.grid(row=row, column=2, padx=2, pady=2)
    manual_buttons.extend([btn_roll_neg, btn_roll_pos])
    ttk.Label(frame, text="Roll").grid(row=row, column=1)
    row += 1

    btn_pitch_neg = ttk.Button(frame, text="Pitch-", width=8, command=lambda: do_move((0, 0, 0), (0, -r, 0)))
    btn_pitch_neg.grid(row=row, column=0, padx=2, pady=2)
    btn_pitch_pos = ttk.Button(frame, text="Pitch+", width=8, command=lambda: do_move((0, 0, 0), (0, r, 0)))
    btn_pitch_pos.grid(row=row, column=2, padx=2, pady=2)
    manual_buttons.extend([btn_pitch_neg, btn_pitch_pos])
    ttk.Label(frame, text="Pitch").grid(row=row, column=1)
    row += 1

    btn_yaw_neg = ttk.Button(frame, text="Yaw-", width=8, command=lambda: do_move((0, 0, 0), (0, 0, -r)))
    btn_yaw_neg.grid(row=row, column=0, padx=2, pady=2)
    btn_yaw_pos = ttk.Button(frame, text="Yaw+", width=8, command=lambda: do_move((0, 0, 0), (0, 0, r)))
    btn_yaw_pos.grid(row=row, column=2, padx=2, pady=2)
    manual_buttons.extend([btn_yaw_neg, btn_yaw_pos])
    ttk.Label(frame, text="Yaw").grid(row=row, column=1)
    row += 1

    ttk.Label(frame, text="Gripper", font=("Arial", 10, "bold")).grid(
        row=row, column=0, columnspan=3, pady=(10, 5)
    )
    row += 1

    def grip_delta(delta: float) -> None:
        do_move((0, 0, 0), (0, 0, 0), delta)

    btn_grip_close = ttk.Button(
        frame,
        text="Close",
        width=8,
        command=lambda: grip_delta(-controller.gripper_step),
    )
    btn_grip_close.grid(row=row, column=0, padx=2, pady=2)

    grip_label = ttk.Label(frame, text="0", width=10, relief="sunken", anchor="center")
    grip_label.grid(row=row, column=1, padx=2)

    btn_grip_open = ttk.Button(
        frame,
        text="Open",
        width=8,
        command=lambda: grip_delta(controller.gripper_step),
    )
    btn_grip_open.grid(row=row, column=2, padx=2, pady=2)
    manual_buttons.extend([btn_grip_close, btn_grip_open])
    row += 1

    ttk.Button(frame, text="Reset (randomized)", command=controller.reset_robot).grid(
        row=row, column=0, columnspan=3, pady=(15, 5)
    )
    row += 1

    def tick_auto() -> None:
        if auto_policy.is_running():
            auto_policy.step()
        root.after(40, tick_auto)

    def update_grip_label() -> None:
        if getattr(controller, "running", False):
            try:
                grip_val = controller.get_gripper_position()
                grip_label.config(text=f"{grip_val:.0f}")
            except Exception:
                pass
        root.after(100, update_grip_label)

    root.after(40, tick_auto)
    root.after(100, update_grip_label)
    root.after(200, update_auto_status)

    if own_root:
        def on_closing() -> None:
            controller.stop()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)

    return root


def create_mixed_gui_autograb(
    controller: RandomizedVLAController,
    auto_policy: AutoGrabPolicy,
    root: tk.Tk | None = None,
    parent: tk.Widget | None = None,
    column: int = 0,
) -> tk.Tk | None:
    """GUI variant wired to AutoGrabPolicy (labels mention 'grab' instead of 'pickup')."""
    if tk is None or ttk is None:
        print("Error: tkinter is required for mixed intervention GUI mode")
        return None

    own_root = False
    if root is None:
        root = tk.Tk()
        root.title(f"Robot {controller.robot_index + 1} Mixed Control (Randomized, AutoGrab)")
        root.geometry("460x520")
        own_root = True

    container = parent if parent is not None else root

    frame = ttk.Frame(container, padding="10", borderwidth=2, relief="groove")
    frame.grid(row=0, column=column, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

    title = ttk.Label(
        frame,
        text=f"Robot {controller.robot_index + 1} VLA Mixed Control (AutoGrab)",
        font=("Arial", 12, "bold"),
    )
    title.grid(row=0, column=0, columnspan=3, pady=(0, 10))

    info = ttk.Label(
        frame,
        text=f"Pos step: {controller.pos_step:.3f} m | Rot step: {np.degrees(controller.rot_step):.1f} deg",
        font=("Arial", 9, "italic"),
    )
    info.grid(row=1, column=0, columnspan=3, pady=(0, 10))

    auto_status = tk.StringVar(value="AutoGrab: idle")
    manual_buttons: list[ttk.Button] = []

    def update_auto_status() -> None:
        running = auto_policy.is_running()
        if running:
            auto_status.set(f"AutoGrab: running (phase {auto_policy.phase})")
        else:
            auto_status.set("AutoGrab: idle")

        for btn in manual_buttons:
            try:
                btn.configure(state="disabled" if running else "normal")
            except Exception:
                pass

        root.after(200, update_auto_status)

    status_label = ttk.Label(frame, textvariable=auto_status, font=("Arial", 9, "italic"))
    status_label.grid(row=2, column=0, columnspan=3, pady=(0, 10))

    def do_move(dpos, drot, dgrip=0.0) -> None:
        if auto_policy.is_running():
            return
        controller.apply_ee_delta(np.array(dpos, dtype=np.float32), np.array(drot, dtype=np.float32), dgrip)

    row = 3
    ttk.Label(frame, text="Automatic grab", font=("Arial", 10, "bold")).grid(
        row=row, column=0, columnspan=3, pady=(5, 5)
    )
    row += 1

    def start_auto() -> None:
        auto_policy.reset()

    def stop_auto() -> None:
        auto_policy.stop()

    ttk.Button(frame, text="Start Auto Grab", command=start_auto).grid(
        row=row, column=0, columnspan=2, padx=2, pady=2, sticky=(tk.W, tk.E)
    )
    ttk.Button(frame, text="Stop Auto", command=stop_auto).grid(
        row=row, column=2, padx=2, pady=2, sticky=(tk.W, tk.E)
    )
    row += 1

    ttk.Label(frame, text="Manual EE Position", font=("Arial", 10, "bold")).grid(
        row=row, column=0, columnspan=3, pady=(10, 5)
    )
    row += 1

    btn_x_neg = ttk.Button(frame, text="X-", width=8, command=lambda: do_move((-controller.pos_step, 0, 0), (0, 0, 0)))
    btn_x_neg.grid(row=row, column=0, padx=2, pady=2)
    btn_x_pos = ttk.Button(frame, text="X+", width=8, command=lambda: do_move((controller.pos_step, 0, 0), (0, 0, 0)))
    btn_x_pos.grid(row=row, column=2, padx=2, pady=2)
    manual_buttons.extend([btn_x_neg, btn_x_pos])
    ttk.Label(frame, text="X axis").grid(row=row, column=1)
    row += 1

    btn_y_neg = ttk.Button(frame, text="Y-", width=8, command=lambda: do_move((0, -controller.pos_step, 0), (0, 0, 0)))
    btn_y_neg.grid(row=row, column=0, padx=2, pady=2)
    btn_y_pos = ttk.Button(frame, text="Y+", width=8, command=lambda: do_move((0, controller.pos_step, 0), (0, 0, 0)))
    btn_y_pos.grid(row=row, column=2, padx=2, pady=2)
    manual_buttons.extend([btn_y_neg, btn_y_pos])
    ttk.Label(frame, text="Y axis").grid(row=row, column=1)
    row += 1

    btn_z_neg = ttk.Button(frame, text="Z-", width=8, command=lambda: do_move((0, 0, -controller.pos_step), (0, 0, 0)))
    btn_z_neg.grid(row=row, column=0, padx=2, pady=2)
    btn_z_pos = ttk.Button(frame, text="Z+", width=8, command=lambda: do_move((0, 0, controller.pos_step), (0, 0, 0)))
    btn_z_pos.grid(row=row, column=2, padx=2, pady=2)
    manual_buttons.extend([btn_z_neg, btn_z_pos])
    ttk.Label(frame, text="Z axis").grid(row=row, column=1)
    row += 1

    ttk.Label(frame, text="Manual EE Rotation", font=("Arial", 10, "bold")).grid(
        row=row, column=0, columnspan=3, pady=(10, 5)
    )
    row += 1

    r = controller.rot_step
    btn_roll_neg = ttk.Button(frame, text="Roll-", width=8, command=lambda: do_move((0, 0, 0), (-r, 0, 0)))
    btn_roll_neg.grid(row=row, column=0, padx=2, pady=2)
    btn_roll_pos = ttk.Button(frame, text="Roll+", width=8, command=lambda: do_move((0, 0, 0), (r, 0, 0)))
    btn_roll_pos.grid(row=row, column=2, padx=2, pady=2)
    manual_buttons.extend([btn_roll_neg, btn_roll_pos])
    ttk.Label(frame, text="Roll").grid(row=row, column=1)
    row += 1

    btn_pitch_neg = ttk.Button(frame, text="Pitch-", width=8, command=lambda: do_move((0, 0, 0), (0, -r, 0)))
    btn_pitch_neg.grid(row=row, column=0, padx=2, pady=2)
    btn_pitch_pos = ttk.Button(frame, text="Pitch+", width=8, command=lambda: do_move((0, 0, 0), (0, r, 0)))
    btn_pitch_pos.grid(row=row, column=2, padx=2, pady=2)
    manual_buttons.extend([btn_pitch_neg, btn_pitch_pos])
    ttk.Label(frame, text="Pitch").grid(row=row, column=1)
    row += 1

    btn_yaw_neg = ttk.Button(frame, text="Yaw-", width=8, command=lambda: do_move((0, 0, 0), (0, 0, -r)))
    btn_yaw_neg.grid(row=row, column=0, padx=2, pady=2)
    btn_yaw_pos = ttk.Button(frame, text="Yaw+", width=8, command=lambda: do_move((0, 0, 0), (0, 0, r)))
    btn_yaw_pos.grid(row=row, column=2, padx=2, pady=2)
    manual_buttons.extend([btn_yaw_neg, btn_yaw_pos])
    ttk.Label(frame, text="Yaw").grid(row=row, column=1)
    row += 1

    ttk.Label(frame, text="Gripper", font=("Arial", 10, "bold")).grid(
        row=row, column=0, columnspan=3, pady=(10, 5)
    )
    row += 1

    def grip_delta(delta: float) -> None:
        do_move((0, 0, 0), (0, 0, 0), delta)

    btn_grip_close = ttk.Button(
        frame,
        text="Close",
        width=8,
        command=lambda: grip_delta(-controller.gripper_step),
    )
    btn_grip_close.grid(row=row, column=0, padx=2, pady=2)

    grip_label = ttk.Label(frame, text="0", width=10, relief="sunken", anchor="center")
    grip_label.grid(row=row, column=1, padx=2)

    btn_grip_open = ttk.Button(
        frame,
        text="Open",
        width=8,
        command=lambda: grip_delta(controller.gripper_step),
    )
    btn_grip_open.grid(row=row, column=2, padx=2, pady=2)
    manual_buttons.extend([btn_grip_close, btn_grip_open])
    row += 1

    ttk.Button(frame, text="Reset (randomized)", command=controller.reset_robot).grid(
        row=row, column=0, columnspan=3, pady=(15, 5)
    )
    row += 1

    def tick_auto() -> None:
        if auto_policy.is_running():
            auto_policy.step()
        root.after(40, tick_auto)

    def update_grip_label() -> None:
        if getattr(controller, "running", False):
            try:
                grip_val = controller.get_gripper_position()
                grip_label.config(text=f"{grip_val:.0f}")
            except Exception:
                pass
        root.after(100, update_grip_label)

    root.after(40, tick_auto)
    root.after(100, update_grip_label)
    root.after(200, update_auto_status)

    if own_root:
        def on_closing() -> None:
            controller.stop()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)

    return root


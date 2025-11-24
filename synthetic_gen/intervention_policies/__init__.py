from __future__ import annotations

from typing import Optional

import numpy as np

from synthetic_gen.interactive_record_replay_randomized_vla import (  # type: ignore
    RandomizedVLAController,
)
from utils.jacobian import yaw_to_face_target  # type: ignore


class AutoPickupPolicy:
    """
    Simple phased pick-up policy operating in EE space.

    Phases (robot_index refers to controller.robot_index):
      0: Move EE above object (offset_z_high)
      1: Move EE down to near object (offset_z_low)
      2: Close gripper
      3: Lift object up (offset_z_lift)
      4: Hold (no-op, policy finished)

    At each call to step(), we compute a small EE delta using controller.apply_ee_delta.
    """

    def __init__(
        self,
        controller: RandomizedVLAController,
        offset_z_high: float = 0.20,
        offset_z_low: float = 0.03,
        offset_z_lift: float = 0.30,
        align_orientation: bool = False,
    ) -> None:
        self.controller = controller
        self.offset_z_high = float(offset_z_high)
        self.offset_z_low = float(offset_z_low)
        self.offset_z_lift = float(offset_z_lift)
        self.align_orientation = bool(align_orientation)
        self.phase: int = 0
        self.active: bool = False

    def reset(self) -> None:
        """
        (Re)start the phased policy.

        - If a run has finished (phase >= 4), start a fresh sequence from phase 0.
        - If the policy was paused mid-run via manual intervention, resume from
          the current phase and the *current* world state (EE/object positions).
        """
        if self.phase >= 4 or self.phase < 0:
            self.phase = 0
        self.active = True

    def stop(self) -> None:
        self.active = False

    def is_running(self) -> bool:
        return self.active

    def step(self) -> None:
        """Advance one small step of the automatic policy."""
        if not self.active:
            return

        data = self.controller.data
        obj_pos = data.qpos[0:3].copy()
        ee_pos = data.site_xpos[self.controller.ee_site_id].copy()

        dpos = np.zeros(3, dtype=np.float32)
        drot = np.zeros(3, dtype=np.float32)
        dgrip = 0.0

        # Phase 0: move above object
        if self.phase == 0:
            target = obj_pos + np.array([0.0, 0.0, self.offset_z_high], dtype=np.float32)
            dpos = self._clamped_delta(ee_pos, target, 0.5 * self.controller.pos_step)
            if np.linalg.norm(target - ee_pos) < 0.01:
                self.phase = 1

        # Phase 1: move down near object
        elif self.phase == 1:
            target = obj_pos + np.array([0.0, 0.0, self.offset_z_low], dtype=np.float32)
            dpos = self._clamped_delta(ee_pos, target, 0.5 * self.controller.pos_step)
            if np.linalg.norm(target - ee_pos) < 0.005:
                self.phase = 2

        # Phase 2: close gripper
        elif self.phase == 2:
            dgrip = -self.controller.gripper_step
            if self.controller.get_gripper_position() < 80.0:
                self.phase = 3

        # Phase 3: lift object
        elif self.phase == 3:
            target = obj_pos + np.array([0.0, 0.0, self.offset_z_lift], dtype=np.float32)
            dpos = self._clamped_delta(ee_pos, target, 0.5 * self.controller.pos_step)
            if np.linalg.norm(target - ee_pos) < 0.01:
                self.phase = 4

        # Phase 4: hold; policy finished
        elif self.phase == 4:
            self.active = False
            return

        if self.align_orientation:
            dyaw = yaw_to_face_target(
                self.controller.model,
                self.controller.data,
                self.controller.ee_site_id,
                obj_pos,
            )
            drot[2] = float(dyaw)

        if self.active:
            self.controller.apply_ee_delta(dpos, drot, dgrip)

    @staticmethod
    def _clamped_delta(current: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
        delta = target - current
        dist = float(np.linalg.norm(delta))
        if dist <= max_step or dist == 0.0:
            return delta.astype(np.float32)
        return (delta / dist * max_step).astype(np.float32)


class AutoGrabPolicy:
    """
    Alternative auto policy that always tries to move the end-effector to a
    graspable distance from the object, then pulls the object toward a global
    target position in (x, y, z).

    Phases:
      0: Approach until EE-object distance < grasp_distance.
      1: Close gripper while maintaining proximity.
      2: Lift/translate object so its z approaches target_object_z and x,y -> (0,0).
      3: Hold; policy finished.
    """

    def __init__(
        self,
        controller: RandomizedVLAController,
        grasp_distance: float = 0.04,
        target_object_z: float = 0.75,
        align_orientation: bool = False,
    ) -> None:
        self.controller = controller
        self.grasp_distance = float(grasp_distance)
        self.target_object_z = float(target_object_z)
        self.align_orientation = bool(align_orientation)
        self.phase: int = 0
        self.active: bool = False

    def reset(self) -> None:
        """(Re)start the policy from the current world state."""
        if self.phase >= 3 or self.phase < 0:
            self.phase = 0
        self.active = True

    def stop(self) -> None:
        self.active = False

    def is_running(self) -> bool:
        return self.active

    def step(self) -> None:
        if not self.active:
            return

        data = self.controller.data
        obj_pos = data.qpos[0:3].copy()
        ee_pos = data.site_xpos[self.controller.ee_site_id].copy()

        dpos = np.zeros(3, dtype=np.float32)
        drot = np.zeros(3, dtype=np.float32)
        dgrip = 0.0

        dist = float(np.linalg.norm(ee_pos - obj_pos))

        # Phase 0: approach object until within grasp_distance
        if self.phase == 0:
            if dist > self.grasp_distance:
                dpos = self._clamped_delta(
                    ee_pos,
                    obj_pos,
                    0.5 * self.controller.pos_step,
                )
            else:
                self.phase = 1

        # Phase 1: close gripper while staying near object
        if self.phase == 1:
            if dist > self.grasp_distance:
                dpos = self._clamped_delta(
                    ee_pos,
                    obj_pos,
                    0.5 * self.controller.pos_step,
                )
            dgrip = -self.controller.gripper_step
            if self.controller.get_gripper_position() < 80.0:
                self.phase = 2

        # Phase 2: pull object toward (0,0,target_object_z)
        if self.phase == 2:
            current_z = float(obj_pos[2])
            target = np.array([0.0, 0.0, self.target_object_z], dtype=np.float32)
            dpos = self._clamped_delta(
                ee_pos,
                target,
                0.5 * self.controller.pos_step,
            )
            if abs(current_z - self.target_object_z) < 0.01:
                self.phase = 3

        # Phase 3: hold / finished
        if self.phase >= 3:
            self.active = False
            return

        if self.align_orientation:
            dyaw = yaw_to_face_target(
                self.controller.model,
                self.controller.data,
                self.controller.ee_site_id,
                obj_pos,
            )
            drot[2] = float(dyaw)

        self.controller.apply_ee_delta(dpos, drot, dgrip)

    @staticmethod
    def _clamped_delta(current: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
        delta = target - current
        dist = float(np.linalg.norm(delta))
        if dist <= max_step or dist == 0.0:
            return delta.astype(np.float32)
        return (delta / dist * max_step).astype(np.float32)

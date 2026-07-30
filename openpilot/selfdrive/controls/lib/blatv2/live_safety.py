"""Invalid-output safety wrapper for the modular controller.

Valid controller output is changed only by the one runtime actuator envelope.
This wrapper has no steering policy. It handles only nonfinite/invalid core
results using the previously field-reviewed lifecycle:

* first invalid active frame: hold the last applied torque;
* persistent invalidity: request zero through the normal down-rate envelope;
* after 250 ms: keep requesting zero through that same envelope and latch the
  existing commIssue valid-bit path;
* re-entry: ten consecutive valid finite frames, then resume through the
  normal envelope from the current applied torque.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

from openpilot.selfdrive.controls.lib.blatv2.actuator import (
  RuntimeTorqueLimits,
  apply_torque_envelope,
)


INVALID_LATCH_SECONDS = 0.250
RECOVERY_OK_FRAMES = 10


class LiveSafetyState(IntEnum):
  INACTIVE = 0
  OK = 1
  HOLDING_FIRST_INVALID = 2
  DECAYING_INVALID = 3
  COMM_ISSUE_LATCHED = 4
  RECOVERING = 5


@dataclass(frozen=True, slots=True)
class SafeCommand:
  torque: float
  controls_valid: bool
  car_control_valid: bool
  state: LiveSafetyState
  invalid_frames: int
  recovery_ok_frames: int
  constrained: bool


class InvalidOutputGuard:
  def __init__(self, dt: float):
    self.dt = float(dt)
    if not math.isfinite(self.dt) or self.dt <= 0.0:
      raise ValueError("guard dt must be finite and positive")
    self.invalid_latch_frames = max(
      int(math.ceil(INVALID_LATCH_SECONDS / self.dt)), 1,
    )
    self.invalid_frames = 0
    self.recovery_ok_frames = 0
    self.comm_issue_latched = False

  def reset(self) -> None:
    self.invalid_frames = 0
    self.recovery_ok_frames = 0
    self.comm_issue_latched = False

  def update(
    self,
    *,
    active: bool,
    core_ok: bool,
    raw_torque: float,
    applied_torque: float,
    driver_torque: float,
    limits: RuntimeTorqueLimits,
  ) -> SafeCommand:
    raw = float(raw_torque)
    applied = float(applied_torque)
    finite = math.isfinite(raw) and math.isfinite(applied)
    frame_ok = bool(core_ok) and finite

    if not active:
      self.reset()
      return SafeCommand(
        torque=0.0,
        controls_valid=True,
        car_control_valid=True,
        state=LiveSafetyState.INACTIVE,
        invalid_frames=0,
        recovery_ok_frames=0,
        constrained=False,
      )

    if not frame_ok:
      self.invalid_frames += 1
      self.recovery_ok_frames = 0
      if self.invalid_frames >= self.invalid_latch_frames:
        self.comm_issue_latched = True
      if self.invalid_frames == 1:
        torque = applied if math.isfinite(applied) else 0.0
        constrained = True
        state = LiveSafetyState.HOLDING_FIRST_INVALID
      else:
        result = apply_torque_envelope(
          limits, 0.0, applied if math.isfinite(applied) else 0.0,
          driver_torque,
        )
        # Latching validity must never bypass the sole actuator envelope.
        # Telemetry, the internal command, and opendbc therefore all observe
        # the same down-rate-limited request even when 250 ms expires before
        # a high starting command has reached zero.
        torque = result.applied_torque
        constrained = True
        state = (
          LiveSafetyState.COMM_ISSUE_LATCHED
          if self.comm_issue_latched
          else LiveSafetyState.DECAYING_INVALID
        )
      return SafeCommand(
        torque=torque,
        controls_valid=not self.comm_issue_latched,
        car_control_valid=not self.comm_issue_latched,
        state=state,
        invalid_frames=self.invalid_frames,
        recovery_ok_frames=0,
        constrained=constrained,
      )

    if self.invalid_frames > 0 or self.comm_issue_latched:
      self.recovery_ok_frames += 1
      if self.recovery_ok_frames < RECOVERY_OK_FRAMES:
        result = apply_torque_envelope(
          limits, 0.0, applied, driver_torque,
        )
        return SafeCommand(
          torque=result.applied_torque,
          controls_valid=not self.comm_issue_latched,
          car_control_valid=not self.comm_issue_latched,
          state=(
            LiveSafetyState.COMM_ISSUE_LATCHED
            if self.comm_issue_latched
            else LiveSafetyState.RECOVERING
          ),
          invalid_frames=self.invalid_frames,
          recovery_ok_frames=self.recovery_ok_frames,
          constrained=True,
        )
      self.invalid_frames = 0
      self.recovery_ok_frames = 0
      self.comm_issue_latched = False

    result = apply_torque_envelope(
      limits, raw, applied, driver_torque,
    )
    return SafeCommand(
      torque=result.applied_torque,
      controls_valid=True,
      car_control_valid=True,
      state=LiveSafetyState.OK,
      invalid_frames=0,
      recovery_ok_frames=0,
      constrained=result.constrained,
    )

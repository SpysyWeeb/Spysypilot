"""Live BLaTv2 LQI adapter and its explicit invalid-output contract.

The numerical controller remains :class:`ShadowCore`: controlsd calls the same
reference, observer, plant, workspace, and LQI implementation imported by the
route-audit harness.  This module owns only the live failure policy around that
artifact.

An active invalid result holds the previous request for one frame, then decays
toward zero through the plant's runtime 409/4/7 limiter.  At 250 ms it requests
zero and marks both controlsState and carControl invalid; selfdrived therefore
uses its existing ``commIssue`` soft-disable/no-entry path.  Recovery requires
ten consecutive finite OK solves.  The LQI integral resets on the recovery
frame and the resumed command remains slew-feasible from measured applied
torque.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.blatv2.controller import (
  CandidateStatus,
  ControllerParams,
)
from openpilot.selfdrive.controls.lib.blatv2.plant import PlantParams
from openpilot.selfdrive.controls.lib.blatv2.shadow import ShadowCore


INVALID_DISENGAGE_FRAMES = int(round(0.250 / DT_CTRL))
RECOVERY_OK_FRAMES = 10

SEED_PATH = Path(__file__).resolve().parent / "plant_seed_params.json"
CONTROLLER_SEED_PATH = (
  Path(__file__).resolve().parent / "controller_seed_params.json"
)


@dataclass(slots=True)
class LiveLQIResult:
  command_torque: float = 0.0
  reference_curvature: float = 0.0
  status: int = int(CandidateStatus.INPUT_INVALID)
  output_valid: bool = True
  invalid_frames: int = 0
  recovery_ok_frames: int = 0
  internal_ok: bool = False


class LiveLQIController:
  """Thin stateful safety adapter around the shared tournament artifact."""

  def __init__(
    self,
    car_params: Any,
    car_controller_params: Any,
    torque_params: Any,
  ) -> None:
    plant_params = PlantParams.from_seed_file(
      SEED_PATH, car_controller_params,
    )
    controller_params = ControllerParams.from_seed_file(CONTROLLER_SEED_PATH)
    self.core = ShadowCore(
      plant_params, torque_params, car_params, controller_params,
    )
    self.result = LiveLQIResult()
    self.command_torque = 0.0
    self.invalid_frames = 0
    self.recovery_ok_frames = 0
    self.recovering = False
    self.comm_issue_latched = False

  def _reset_inactive(self) -> None:
    self.core.fallback.reset()
    self.command_torque = 0.0
    self.invalid_frames = 0
    self.recovery_ok_frames = 0
    self.recovering = False
    self.comm_issue_latched = False

  def _decay(self) -> float:
    self.command_torque = self.core.twin.apply_slew(
      self.command_torque, 0.0,
    )
    return self.command_torque

  def step(
    self,
    model: Any,
    car_state: Any,
    observer_car_control: Any,
    car_output: Any,
    live_parameters: Any,
    live_parameters_valid: bool,
    lateral_delay: float,
    lateral_delay_valid: bool,
    model_valid: bool,
    lateral_active: bool,
  ) -> LiveLQIResult:
    """Compute one frame and apply the pinned live invalid/re-entry policy."""
    common = self.core.result
    prepared = False
    candidate_status = CandidateStatus.INPUT_INVALID
    candidate_command = self.command_torque
    try:
      common = self.core.begin_frame(
        model,
        car_state,
        observer_car_control,
        car_output,
        live_parameters,
        live_parameters_valid,
        lateral_delay,
        lateral_delay_valid,
        model_valid,
      )
      prepared = True
      self.core.compute_fallback()
      candidate_status = CandidateStatus(common.fallback_status)
      candidate_command = float(common.fallback_command_torque)
    except (RuntimeError, ValueError, OverflowError):
      common = self.core.invalid_result()
      prepared = False

    internal_ok = bool(
      common.valid
      and candidate_status == CandidateStatus.OK
      and math.isfinite(candidate_command)
    )

    if not lateral_active and not (
      self.recovering or self.comm_issue_latched
    ):
      self._reset_inactive()
    elif not lateral_active:
      # A latched no-entry must not disappear merely because selfdrived
      # completed the soft disable. Keep exercising the passive numerical path
      # and clear both controller recovery and commIssue after the same ten OK
      # frames. No candidate command is ever applied while lateral is inactive.
      self.command_torque = 0.0
      if internal_ok:
        self.recovery_ok_frames += 1
        self.invalid_frames = 0
        if self.recovery_ok_frames >= RECOVERY_OK_FRAMES:
          self.core.fallback.reset()
          self.recovering = False
          self.comm_issue_latched = False
          self.recovery_ok_frames = RECOVERY_OK_FRAMES
      else:
        self.invalid_frames += 1
        self.recovery_ok_frames = 0
    elif internal_ok:
      if self.recovering:
        self.recovery_ok_frames += 1
        if self.recovery_ok_frames >= RECOVERY_OK_FRAMES:
          # The nine quarantined solves cannot contaminate re-entry.
          self.core.fallback.reset()
          if prepared:
            self.core.compute_fallback()
            candidate_status = CandidateStatus(common.fallback_status)
            candidate_command = float(common.fallback_command_torque)
          internal_ok = bool(
            common.valid
            and candidate_status == CandidateStatus.OK
            and math.isfinite(candidate_command)
          )
          if internal_ok:
            self.command_torque = candidate_command
            self.invalid_frames = 0
            self.recovery_ok_frames = RECOVERY_OK_FRAMES
            self.recovering = False
            self.comm_issue_latched = False
          else:
            self.recovery_ok_frames = 0
            self._decay()
        else:
          self.invalid_frames = 0
          self._decay()
      else:
        self.command_torque = candidate_command
        self.invalid_frames = 0
        self.recovery_ok_frames = RECOVERY_OK_FRAMES
    else:
      self.invalid_frames += 1
      self.recovery_ok_frames = 0
      self.recovering = True
      if self.invalid_frames == 1:
        # Frame one deliberately holds the last known-finite request.
        pass
      elif self.invalid_frames >= INVALID_DISENGAGE_FRAMES:
        self.command_torque = 0.0
        self.comm_issue_latched = True
      else:
        self._decay()

    if prepared:
      self.core.end_frame()

    result = self.result
    result.command_torque = float(self.command_torque)
    result.reference_curvature = float(common.reference_curvature)
    result.status = int(candidate_status)
    result.output_valid = not self.comm_issue_latched
    result.invalid_frames = int(self.invalid_frames)
    result.recovery_ok_frames = int(self.recovery_ok_frames)
    result.internal_ok = bool(internal_ok)
    return result

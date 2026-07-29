"""Shared numerical contracts for the two BLaTv2 shadow candidates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import json
import math
from pathlib import Path

from openpilot.common.realtime import DT_MDL


DECISION_DT = DT_MDL
LIVE_CONTROLLER_VERSION = 205


class CandidateStatus(IntEnum):
  OK = 0
  INPUT_INVALID = 1
  INFEASIBLE = 2
  NON_CONVERGED = 3
  ENUMERATION_EXHAUSTED = 4


class ObserverStatus(IntEnum):
  ACTIVE = 0
  FROZEN_RECORDED_CONSTRAINT = 1
  RESET_LATERAL_INVALID = 2
  RESET_STEERING_PRESSED = 3
  RESET_STANDSTILL = 4
  RESET_MODEL_INVALID = 5
  RESET_ENGAGEMENT = 6

  @property
  def reset(self) -> bool:
    return self in (
      ObserverStatus.RESET_LATERAL_INVALID,
      ObserverStatus.RESET_STEERING_PRESSED,
      ObserverStatus.RESET_STANDSTILL,
      ObserverStatus.RESET_MODEL_INVALID,
      ObserverStatus.RESET_ENGAGEMENT,
    )

  @property
  def frozen(self) -> bool:
    return self == ObserverStatus.FROZEN_RECORDED_CONSTRAINT


@dataclass(frozen=True, slots=True)
class ControllerParams:
  """Controller tolerances plus the physical observer time scale.

  ``sigma_curvature`` and ``kinetic_friction`` are the b7 response-surface
  axes selected for v203. A zero curvature tolerance remains available to
  reproduce the pre-selection controller in replay; the kinetic default
  preserves its full-breakaway feedforward.
  """

  sigma_y: float
  sigma_heading: float
  sigma_torque_rate: float
  tau_disturbance: float
  provisional: bool
  sigma_curvature: float = 0.0
  kinetic_friction: float = 0.09

  @classmethod
  def from_seed_file(cls, path: str | Path) -> ControllerParams:
    with Path(path).open(encoding="utf-8") as stream:
      seed = json.load(stream)
    params = cls(
      sigma_y=float(seed["sigma_y"]["value"]),
      sigma_heading=float(seed["sigma_heading"]["value"]),
      sigma_torque_rate=float(seed["sigma_torque_rate"]["value"]),
      tau_disturbance=float(seed["tau_disturbance"]["value"]),
      provisional=bool(seed["provisional"]),
      sigma_curvature=float(seed.get("sigma_curvature", {}).get("value", 0.0)),
      kinetic_friction=float(seed.get("kinetic_friction", {}).get("value", 0.09)),
    )
    params.validate()
    return params

  def validate(self) -> None:
    values = (self.sigma_y, self.sigma_heading, self.sigma_torque_rate, self.tau_disturbance)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
      raise ValueError("controller scales must be finite and positive")
    if not math.isfinite(self.sigma_curvature) or self.sigma_curvature < 0.0:
      raise ValueError("sigma_curvature must be finite and non-negative")
    if not math.isfinite(self.kinetic_friction) or self.kinetic_friction < 0.0:
      raise ValueError("kinetic_friction must be finite and non-negative")


@dataclass(slots=True)
class CandidateResult:
  command_torque: float = 0.0
  raw_command_torque: float = 0.0
  feedforward_torque: float = 0.0
  feedback_torque: float = 0.0
  desired_angle_deg: float = 0.0
  desired_rate_deg_s: float = 0.0
  desired_acceleration_deg_s2: float = 0.0
  predicted_angle_deg: float = 0.0
  predicted_rate_deg_s: float = 0.0
  required_acceleration_deg_s2: float = 0.0
  action_speed_mps: float = 0.0
  aligning_torque: float = 0.0
  friction_torque: float = 0.0
  dynamic_torque: float = 0.0
  action_time_seconds: float = 0.0
  slew_constrained: bool = False
  status: CandidateStatus = CandidateStatus.INPUT_INVALID
  candidate_count: int = 0
  available_schedule_count: int = 0
  optimality_residual: float = 0.0

  def invalidate(self, applied_torque: float = 0.0, status: CandidateStatus = CandidateStatus.INPUT_INVALID) -> None:
    self.command_torque = float(applied_torque)
    self.raw_command_torque = float(applied_torque)
    self.feedforward_torque = 0.0
    self.feedback_torque = 0.0
    self.desired_angle_deg = 0.0
    self.desired_rate_deg_s = 0.0
    self.desired_acceleration_deg_s2 = 0.0
    self.predicted_angle_deg = 0.0
    self.predicted_rate_deg_s = 0.0
    self.required_acceleration_deg_s2 = 0.0
    self.action_speed_mps = 0.0
    self.aligning_torque = 0.0
    self.friction_torque = 0.0
    self.dynamic_torque = 0.0
    self.action_time_seconds = 0.0
    self.slew_constrained = False
    self.status = status
    self.candidate_count = 0
    self.available_schedule_count = 0
    self.optimality_residual = 0.0

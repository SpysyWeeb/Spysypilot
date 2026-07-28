"""Shared numerical contracts for the two BLaTv2 shadow candidates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import json
import math
from pathlib import Path

from openpilot.common.realtime import DT_MDL


DECISION_DT = DT_MDL
LIVE_CONTROLLER_VERSION = 202


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

  ``sigma_curvature`` and ``kinetic_friction`` are response-surface overrides
  until a measured cell is selected. A zero curvature tolerance disables the
  additive tracking term; the kinetic default preserves the shipped
  full-breakaway feedforward.
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
  status: CandidateStatus = CandidateStatus.INPUT_INVALID
  candidate_count: int = 0
  available_schedule_count: int = 0
  optimality_residual: float = 0.0

  def invalidate(self, applied_torque: float = 0.0, status: CandidateStatus = CandidateStatus.INPUT_INVALID) -> None:
    self.command_torque = float(applied_torque)
    self.status = status
    self.candidate_count = 0
    self.available_schedule_count = 0
    self.optimality_residual = 0.0

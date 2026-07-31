"""Canonical measured-frame construction shared by live and rlog learning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from openpilot.cereal.services import SERVICE_LIST
from openpilot.selfdrive.controls.lib.blatv2.learning_runtime import (
  MeasuredLearningFrame,
)


# Each source may be at most one-and-a-half declared publication periods older
# than a controls witness. This is an alignment bound, not controller tuning.
MAX_SOURCE_AGE_PERIODS = 1.5


def _copy_message(message: Any) -> Any:
  builder = getattr(message, "as_builder", None)
  return builder() if callable(builder) else message


@dataclass(frozen=True, slots=True)
class CanonicalSourceSnapshot:
  message: Any
  mono_ns: int
  valid: bool
  alive: bool


class CanonicalSourceHistory:
  """Retain the two newest snapshots and select one preceding a witness."""

  __slots__ = ("_current", "_previous")

  def __init__(self) -> None:
    self._current: CanonicalSourceSnapshot | None = None
    self._previous: CanonicalSourceSnapshot | None = None

  def update(
    self,
    *,
    message: Any,
    mono_ns: int,
    valid: bool,
    alive: bool,
  ) -> None:
    timestamp = int(mono_ns)
    if timestamp <= 0:
      return
    snapshot = CanonicalSourceSnapshot(
      message=_copy_message(message),
      mono_ns=timestamp,
      valid=bool(valid),
      alive=bool(alive),
    )
    if self._current is None or timestamp > self._current.mono_ns:
      self._previous = self._current
      self._current = snapshot
    elif timestamp == self._current.mono_ns:
      self._current = snapshot
    elif (
      self._previous is None
      or timestamp >= self._previous.mono_ns
    ):
      self._previous = snapshot

  def select(
    self,
    *,
    witness_mono_ns: int,
    maximum_age_ns: int,
  ) -> CanonicalSourceSnapshot | None:
    witness = int(witness_mono_ns)
    maximum_age = int(maximum_age_ns)
    if witness <= 0 or maximum_age < 0:
      return None
    eligible = tuple(
      snapshot
      for snapshot in (self._current, self._previous)
      if (
        snapshot is not None
        and snapshot.mono_ns <= witness
        and witness - snapshot.mono_ns <= maximum_age
      )
    )
    if not eligible:
      return None
    selected = max(eligible, key=lambda snapshot: snapshot.mono_ns)
    if not selected.valid or not selected.alive:
      return None
    return selected


def maximum_source_age_ns(service: str) -> int:
  frequency = float(SERVICE_LIST[service].frequency)
  if not math.isfinite(frequency) or frequency <= 0.0:
    raise ValueError("learning source must have a positive declared frequency")
  return int(round(MAX_SOURCE_AGE_PERIODS * 1e9 / frequency))


def measured_learning_frame(
  *,
  witness_mono_ns: int,
  car_state: Any,
  car_control: Any,
  car_output: Any,
  live_parameters: Any,
) -> MeasuredLearningFrame:
  """Copy only measured response and validity facts into learner input."""
  return MeasuredLearningFrame(
    sample_mono_ns=int(witness_mono_ns),
    speed_mps=float(car_state.vEgo),
    steering_angle_deg=float(car_state.steeringAngleDeg),
    steering_rate_deg_s=float(car_state.steeringRateDeg),
    steering_torque=float(car_state.steeringTorque),
    steering_pressed=bool(car_state.steeringPressed),
    standstill=bool(car_state.standstill),
    steer_fault_temporary=bool(car_state.steerFaultTemporary),
    steer_fault_permanent=bool(car_state.steerFaultPermanent),
    can_valid=bool(car_state.canValid),
    can_timeout=bool(car_state.canTimeout),
    applied_torque=float(car_output.actuatorsOutput.torque),
    # carControl contributes only the proof that lateral output was enabled.
    lateral_active=bool(car_control.latActive),
    live_parameters_valid=bool(live_parameters.valid),
    angle_offset_valid=bool(live_parameters.angleOffsetValid),
    steer_ratio_valid=bool(live_parameters.steerRatioValid),
    stiffness_factor_valid=bool(
      live_parameters.stiffnessFactorValid,
    ),
    angle_offset_deg=float(live_parameters.angleOffsetDeg),
    steer_ratio=float(live_parameters.steerRatio),
    stiffness_factor=float(live_parameters.stiffnessFactor),
    roll_rad=float(live_parameters.roll),
    inputs_valid=True,
  )

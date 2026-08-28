"""Deterministic signed rack-motion reconstruction for learning.

Some vehicles publish a signed steering rate. Hyundai classic and CAN-FD
platforms instead publish an unsigned steering-rate magnitude. The physical
learner needs one signed coordinate for damping, acceleration, friction, and
reversal evidence, so direction is reconstructed from measured steering-angle
motion while the sensor-provided rate magnitude is preserved exactly.

No vehicle fingerprint is consulted. A negative raw rate proves that the
current uninterrupted motion episode already has a signed source. A positive
raw rate remains ambiguous until measured angle motion establishes direction.
Once established, direction may be retained only while the reported magnitude
stays nonzero and the measurement lifecycle remains continuous. A zero rate,
gap, invalid input, driver override, standstill, or disengagement clears that
permission.

The angle-motion threshold is one half of the vehicle-declared rate quantum
integrated over the observation interval. It is a measurement-resolution
test, not a steering feel dial. It rejects sub-quantum angle jitter without
replacing the measured rate magnitude with an angle-derived rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math


class RackMotionSource(StrEnum):
  RESET = "reset"
  ZERO = "zero"
  RAW_SIGNED = "raw_signed"
  ANGLE_DELTA = "angle_delta"
  CONTINUOUS_HOLD = "continuous_hold"
  UNRESOLVED = "unresolved"
  DISAGREEMENT = "disagreement"


@dataclass(frozen=True, slots=True)
class RackMotionObservation:
  dt_s: float
  signed_rate_deg_s: float
  rack_acceleration_deg_s2: float
  sign_valid: bool
  derivative_continuous: bool
  direction_reversal: bool
  source: RackMotionSource


class SignedRackMotionNormalizer:
  """Recover signed rate without filtering the reported magnitude."""

  __slots__ = (
    "_direction",
    "_motion_anchor_angle_deg",
    "_motion_anchor_mono_ns",
    "_previous_angle_deg",
    "_previous_mono_ns",
    "_previous_rate_deg_s",
    "_previous_rate_valid",
    "_raw_signed_episode",
  )

  def __init__(self) -> None:
    self._previous_mono_ns: int | None = None
    self._previous_angle_deg = 0.0
    self._previous_rate_deg_s = 0.0
    self._previous_rate_valid = False
    self._direction = 0
    self._motion_anchor_mono_ns = 0
    self._motion_anchor_angle_deg = 0.0
    self._raw_signed_episode = False

  def reset(self) -> None:
    self._previous_mono_ns = None
    self._previous_angle_deg = 0.0
    self._previous_rate_deg_s = 0.0
    self._previous_rate_valid = False
    self._direction = 0
    self._motion_anchor_mono_ns = 0
    self._motion_anchor_angle_deg = 0.0
    self._raw_signed_episode = False

  @staticmethod
  def _invalid(source: RackMotionSource) -> RackMotionObservation:
    return RackMotionObservation(
      dt_s=0.0,
      signed_rate_deg_s=0.0,
      rack_acceleration_deg_s2=0.0,
      sign_valid=False,
      derivative_continuous=False,
      direction_reversal=False,
      source=source,
    )

  def _seed(
    self,
    *,
    sample_mono_ns: int,
    angle: float,
    raw_rate: float,
  ) -> RackMotionObservation:
    magnitude = abs(raw_rate)
    self._previous_mono_ns = sample_mono_ns
    self._previous_angle_deg = angle
    self._motion_anchor_mono_ns = sample_mono_ns
    self._motion_anchor_angle_deg = angle
    self._previous_rate_valid = raw_rate <= 0.0
    self._raw_signed_episode = raw_rate < 0.0
    self._direction = -1 if raw_rate < 0.0 else 0
    self._previous_rate_deg_s = (
      -magnitude if raw_rate < 0.0 else 0.0
    )
    source = (
      RackMotionSource.RAW_SIGNED
      if raw_rate < 0.0
      else RackMotionSource.ZERO
      if magnitude == 0.0
      else RackMotionSource.UNRESOLVED
    )
    return RackMotionObservation(
      dt_s=0.0,
      signed_rate_deg_s=self._previous_rate_deg_s,
      rack_acceleration_deg_s2=0.0,
      sign_valid=self._previous_rate_valid,
      derivative_continuous=False,
      direction_reversal=False,
      source=source,
    )

  def update(
    self,
    *,
    sample_mono_ns: int,
    steering_angle_deg: float,
    raw_rate_deg_s: float,
    rate_resolution_deg_s: float,
    lifecycle_valid: bool,
    maximum_gap_ns: int,
  ) -> RackMotionObservation:
    try:
      timestamp_ns = int(sample_mono_ns)
      maximum_gap = int(maximum_gap_ns)
      angle = float(steering_angle_deg)
      raw_rate = float(raw_rate_deg_s)
      resolution = float(rate_resolution_deg_s)
    except (TypeError, ValueError, OverflowError):
      self.reset()
      return self._invalid(RackMotionSource.RESET)
    if (
      not lifecycle_valid
      or isinstance(sample_mono_ns, bool)
      or isinstance(maximum_gap_ns, bool)
      or timestamp_ns != sample_mono_ns
      or maximum_gap != maximum_gap_ns
      or timestamp_ns < 0
      or not all(math.isfinite(value) for value in (
        angle,
        raw_rate,
        resolution,
      ))
      or resolution < 0.0
      or maximum_gap <= 0
    ):
      self.reset()
      return self._invalid(RackMotionSource.RESET)

    if self._previous_mono_ns is None:
      return self._seed(
        sample_mono_ns=timestamp_ns,
        angle=angle,
        raw_rate=raw_rate,
      )

    dt_ns = timestamp_ns - self._previous_mono_ns
    if dt_ns == 0:
      # A repeated canonical carState snapshot is not a new physical
      # measurement. Do not advance or destroy the unique-sample history.
      if angle == self._previous_angle_deg:
        return self._invalid(RackMotionSource.UNRESOLVED)
      self.reset()
      return self._invalid(RackMotionSource.DISAGREEMENT)
    if dt_ns < 0 or dt_ns > maximum_gap:
      self.reset()
      return self._seed(
        sample_mono_ns=timestamp_ns,
        angle=angle,
        raw_rate=raw_rate,
      )
    dt_s = dt_ns * 1e-9

    magnitude = abs(raw_rate)
    previous_direction = self._direction
    source = RackMotionSource.UNRESOLVED
    direction = 0
    sign_valid = False

    if magnitude == 0.0:
      direction = 0
      sign_valid = True
      source = RackMotionSource.ZERO
      self._raw_signed_episode = False
      self._motion_anchor_mono_ns = timestamp_ns
      self._motion_anchor_angle_deg = angle
    elif raw_rate < 0.0:
      direction = -1
      sign_valid = True
      source = RackMotionSource.RAW_SIGNED
      self._raw_signed_episode = True
    elif self._raw_signed_episode:
      direction = 1
      sign_valid = True
      source = RackMotionSource.RAW_SIGNED
    elif self._direction != 0:
      # Unsigned rate magnitude can remain nonzero across a physical
      # reversal. A resolved direction is therefore only a bridge across
      # angle quantization plateaus, not permission to ignore later measured
      # motion. Re-evaluate any resolvable per-frame angle delta.
      angle_delta = angle - self._previous_angle_deg
      minimum_angle_motion = 0.5 * resolution * dt_s
      if (
        angle_delta != 0.0
        and abs(angle_delta) + 1e-12 >= minimum_angle_motion
      ):
        direction = 1 if angle_delta > 0.0 else -1
        source = RackMotionSource.ANGLE_DELTA
      else:
        direction = self._direction
        source = RackMotionSource.CONTINUOUS_HOLD
      sign_valid = True
    else:
      elapsed_s = (timestamp_ns - self._motion_anchor_mono_ns) * 1e-9
      angle_delta = angle - self._motion_anchor_angle_deg
      minimum_angle_motion = 0.5 * resolution * elapsed_s
      if (
        angle_delta != 0.0
        and abs(angle_delta) + 1e-12 >= minimum_angle_motion
      ):
        direction = 1 if angle_delta > 0.0 else -1
        sign_valid = True
        source = RackMotionSource.ANGLE_DELTA

    direction_reversal = (
      sign_valid
      and direction != 0
      and previous_direction != 0
      and direction != previous_direction
    )
    signed_rate = direction * magnitude if sign_valid else 0.0
    derivative_continuous = (
      sign_valid
      and self._previous_rate_valid
      and not direction_reversal
    )
    rack_acceleration = (
      (signed_rate - self._previous_rate_deg_s) / dt_s
      if derivative_continuous
      else 0.0
    )

    self._previous_mono_ns = timestamp_ns
    self._previous_angle_deg = angle
    self._previous_rate_deg_s = signed_rate
    self._previous_rate_valid = sign_valid
    self._direction = direction
    if magnitude == 0.0:
      self._direction = 0
    # A resolved direction bridges quantized plateaus, but resolvable angle
    # motion is still consulted on every nonzero frame so a reversal cannot
    # remain latched to the old direction.

    return RackMotionObservation(
      # A reversal retains its real interval for coverage while its
      # acceleration remains invalid for regression.
      dt_s=dt_s if derivative_continuous or direction_reversal else 0.0,
      signed_rate_deg_s=signed_rate,
      rack_acceleration_deg_s2=rack_acceleration,
      sign_valid=sign_valid,
      derivative_continuous=derivative_continuous,
      direction_reversal=direction_reversal,
      source=source,
    )

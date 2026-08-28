"""Measured-response adapter for the modular slow learner.

This module is deliberately blind to model intent and controller output
requests. It converts time-aligned vehicle measurements into
``LearningSample`` values using the same rack mapping and gravity/roll
convention as the plant.

Rack acceleration is a timestamped finite difference of the normalized signed
steering rate. The vehicle-published magnitude is preserved exactly; measured
steering-angle motion supplies direction on platforms whose rate signal is
unsigned. The first sample after startup, disengagement, driver override,
standstill, invalid input, or a dropped-frame gap is marked invalid so a
discontinuity cannot become false plant excitation. A physical direction
reversal is retained as coverage, but its quantized sign-crossing acceleration
is excluded from regression. A valid driver-free actuator-limit boundary
remains authority evidence because the applied torque is measured; its kind
and full-magnitude dwell are retained so the learner can distinguish a
transient from a delay-safe settled input. No filtering or desired-path
signal is introduced.
"""

from __future__ import annotations

import math

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.selfdrive.controls.lib.blatv2.learner import LearningSample
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import (
  RackMappingSnapshot,
  curvature_from_measured_angle,
)
from openpilot.selfdrive.controls.lib.blatv2.rack_motion import (
  SignedRackMotionNormalizer,
)


# A 100 Hz measurement may jitter by half a nominal frame. A larger gap means
# an unknown intermediate torque/rack transition and is not fit evidence.
MAX_CONTINUOUS_MEASUREMENT_GAP_NS = 15_000_000


class LearningMeasurementBuilder:
  """Stateful derivative builder; it never stores or updates a profile."""

  __slots__ = ("_rack_motion",)

  def __init__(self) -> None:
    self._rack_motion = SignedRackMotionNormalizer()

  def reset(self) -> None:
    self._rack_motion.reset()

  def update(
    self,
    *,
    sample_mono_ns: int,
    speed_mps: float,
    measured_rack_angle_deg: float,
    measured_rack_rate_deg_s: float,
    rack_rate_resolution_deg_s: float,
    applied_torque: float,
    lateral_accel_offset: float,
    live_mapping: RackMappingSnapshot | None,
    nominal_mapping: RackMappingSnapshot,
    engaged: bool,
    inputs_valid: bool,
    steering_pressed: bool,
    actuator_constrained: bool,
    standstill: bool,
  ) -> LearningSample:
    """Build one measured-only sample and update derivative history."""
    try:
      timestamp_ns = int(sample_mono_ns)
    except (TypeError, ValueError, OverflowError):
      timestamp_ns = -1
    timestamp_valid = (
      not isinstance(sample_mono_ns, bool)
      and timestamp_ns == sample_mono_ns
      and timestamp_ns >= 0
    )
    speed = float(speed_mps)
    angle = float(measured_rack_angle_deg)
    rate = float(measured_rack_rate_deg_s)
    rate_resolution = float(rack_rate_resolution_deg_s)
    applied = float(applied_torque)
    offset = float(lateral_accel_offset)
    numeric_finite = all(math.isfinite(value) for value in (
      speed,
      angle,
      rate,
      rate_resolution,
      applied,
      offset,
    ))
    mapping_valid = (
      live_mapping is not None
      and live_mapping.valid
      and nominal_mapping.valid
    )
    direction_angle = (
      angle - live_mapping.angle_offset_deg
      if live_mapping is not None
      else angle
    )
    lifecycle_continuous = (
      bool(engaged)
      and timestamp_valid
      and not bool(standstill)
      and bool(inputs_valid)
      and numeric_finite
      and speed >= 0.0
      and mapping_valid
      and not bool(steering_pressed)
      and rate_resolution >= 0.0
    )

    motion = self._rack_motion.update(
      sample_mono_ns=timestamp_ns,
      # Direction is physical rack motion, not a slowly changing learned
      # alignment offset. Curvature reconstruction below still receives the
      # raw steering angle and applies the same offset exactly once.
      steering_angle_deg=direction_angle,
      raw_rate_deg_s=rate,
      rate_resolution_deg_s=rate_resolution,
      lifecycle_valid=lifecycle_continuous,
      maximum_gap_ns=MAX_CONTINUOUS_MEASUREMENT_GAP_NS,
    )
    dt_s = motion.dt_s
    derivative_valid = motion.derivative_continuous
    rack_acceleration = motion.rack_acceleration_deg_s2

    lateral_acceleration = 0.0
    sample_valid = lifecycle_continuous and (
      derivative_valid or motion.direction_reversal
    )
    mapping_evaluation_valid = not lifecycle_continuous
    if lifecycle_continuous:
      try:
        curvature = curvature_from_measured_angle(
          angle,
          speed,
          live_mapping,
          nominal_mapping,
        )
        lateral_acceleration = (
          curvature.curvature * speed * speed
          - live_mapping.roll_rad * ACCELERATION_DUE_TO_GRAVITY
          - offset
        )
        sample_valid = (
          sample_valid
          and curvature.valid
          and math.isfinite(lateral_acceleration)
          and math.isfinite(rack_acceleration)
        )
        mapping_evaluation_valid = (
          curvature.valid
          and math.isfinite(lateral_acceleration)
          and math.isfinite(rack_acceleration)
        )
      except (ValueError, OverflowError):
        sample_valid = False
        mapping_evaluation_valid = False
    if lifecycle_continuous and not mapping_evaluation_valid:
      # A reverse-mapping singularity is an unknown physical frame. Do not
      # bridge its already-observed rack motion into the next derivative.
      self._rack_motion.reset()

    return LearningSample(
      speed_mps=speed if math.isfinite(speed) else 0.0,
      dt_s=dt_s,
      applied_torque=applied if math.isfinite(applied) else 0.0,
      measured_lateral_accel_mps2=(
        lateral_acceleration
        if math.isfinite(lateral_acceleration)
        else 0.0
      ),
      rack_rate_deg_s=(
        motion.signed_rate_deg_s
        if math.isfinite(motion.signed_rate_deg_s)
        else 0.0
      ),
      rack_acceleration_deg_s2=(
        rack_acceleration
        if math.isfinite(rack_acceleration)
        else 0.0
      ),
      engaged=bool(engaged),
      valid=sample_valid,
      steering_pressed=bool(steering_pressed),
      actuator_constrained=bool(actuator_constrained),
      standstill=bool(standstill),
      rack_direction_reversal=motion.direction_reversal,
      measured_rack_angle_deg=(
        angle if math.isfinite(angle) else 0.0
      ),
    )

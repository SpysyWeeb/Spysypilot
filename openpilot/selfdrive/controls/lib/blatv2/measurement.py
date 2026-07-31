"""Measured-response adapter for the modular slow learner.

This module is deliberately blind to model intent and controller output
requests. It converts time-aligned vehicle measurements into
``LearningSample`` values using the same rack mapping and gravity/roll
convention as the plant.

Rack acceleration is a timestamped finite difference of the measured steering
rate. The first sample after startup, disengagement, standstill, invalid input,
or a dropped-frame gap is marked invalid so a discontinuity cannot become
false plant excitation. A valid driver-free actuator-limit boundary remains
authority evidence because the applied torque is measured; its kind and
full-magnitude dwell are retained so the learner can distinguish a transient
from a delay-safe settled input. No filtering or desired-path signal is
introduced.
"""

from __future__ import annotations

import math

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.selfdrive.controls.lib.blatv2.learner import LearningSample
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import (
  RackMappingSnapshot,
  curvature_from_measured_angle,
)


# A 100 Hz measurement may jitter by half a nominal frame. A larger gap means
# an unknown intermediate torque/rack transition and is not fit evidence.
MAX_CONTINUOUS_MEASUREMENT_GAP_S = 0.015


class LearningMeasurementBuilder:
  """Stateful derivative builder; it never stores or updates a profile."""

  __slots__ = ("_previous_time_s", "_previous_rate_deg_s")

  def __init__(self) -> None:
    self._previous_time_s: float | None = None
    self._previous_rate_deg_s = 0.0

  def reset(self) -> None:
    self._previous_time_s = None
    self._previous_rate_deg_s = 0.0

  def update(
    self,
    *,
    sample_time_s: float,
    speed_mps: float,
    measured_rack_angle_deg: float,
    measured_rack_rate_deg_s: float,
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
    timestamp = float(sample_time_s)
    speed = float(speed_mps)
    angle = float(measured_rack_angle_deg)
    rate = float(measured_rack_rate_deg_s)
    applied = float(applied_torque)
    offset = float(lateral_accel_offset)
    numeric_finite = all(math.isfinite(value) for value in (
      timestamp,
      speed,
      angle,
      rate,
      applied,
      offset,
    ))
    mapping_valid = (
      live_mapping is not None
      and live_mapping.valid
      and nominal_mapping.valid
    )
    lifecycle_continuous = (
      bool(engaged)
      and not bool(standstill)
      and bool(inputs_valid)
      and numeric_finite
      and speed >= 0.0
      and mapping_valid
    )

    dt_s = 0.0
    rack_acceleration = 0.0
    derivative_valid = False
    if lifecycle_continuous and self._previous_time_s is not None:
      dt_s = timestamp - self._previous_time_s
      derivative_valid = (
        0.0 < dt_s <= MAX_CONTINUOUS_MEASUREMENT_GAP_S
      )
      if derivative_valid:
        rack_acceleration = (
          rate - self._previous_rate_deg_s
        ) / dt_s

    lateral_acceleration = 0.0
    sample_valid = lifecycle_continuous and derivative_valid
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
      except (ValueError, OverflowError):
        sample_valid = False

    if lifecycle_continuous:
      self._previous_time_s = timestamp
      self._previous_rate_deg_s = rate
    else:
      self.reset()

    return LearningSample(
      speed_mps=speed if math.isfinite(speed) else 0.0,
      dt_s=dt_s if derivative_valid else 0.0,
      applied_torque=applied if math.isfinite(applied) else 0.0,
      measured_lateral_accel_mps2=(
        lateral_acceleration
        if math.isfinite(lateral_acceleration)
        else 0.0
      ),
      rack_rate_deg_s=rate if math.isfinite(rate) else 0.0,
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
    )

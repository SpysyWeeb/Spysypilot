from __future__ import annotations

import math
from unittest.mock import patch

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.selfdrive.controls.lib.blatv2.measurement import (
  MAX_CONTINUOUS_MEASUREMENT_GAP_S,
  LearningMeasurementBuilder,
)
from openpilot.selfdrive.controls.lib.blatv2.learner import (
  ActuatorBoundary,
  _attest_authority_sample,
)
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import (
  RackMappingSnapshot,
  curvature_from_measured_angle,
)


def mapping(*, roll: float = 0.02, offset: float = 1.5, valid: bool = True):
  return RackMappingSnapshot(
    mass_kg=2200.0,
    wheelbase_m=2.9,
    center_to_front_m=1.2,
    center_to_rear_m=1.7,
    tire_stiffness_front=120_000.0,
    tire_stiffness_rear=130_000.0,
    steer_ratio_rear=0.0,
    steer_ratio=14.5,
    roll_rad=roll,
    angle_offset_deg=offset,
    valid=valid,
  )


def update(
  builder: LearningMeasurementBuilder,
  *,
  timestamp: float,
  speed: float = 10.0,
  angle: float = 8.0,
  rate: float = 4.0,
  resolution: float = 4.0,
  applied: float = -0.2,
  offset: float = 0.03,
  live=None,
  engaged: bool = True,
  valid: bool = True,
  steering_pressed: bool = False,
  constrained: bool = False,
  standstill: bool = False,
):
  live_mapping = mapping() if live is None else live
  return builder.update(
    sample_time_s=timestamp,
    speed_mps=speed,
    measured_rack_angle_deg=angle,
    measured_rack_rate_deg_s=rate,
    rack_rate_resolution_deg_s=resolution,
    applied_torque=applied,
    lateral_accel_offset=offset,
    live_mapping=live_mapping,
    nominal_mapping=mapping(roll=0.0, offset=0.0),
    engaged=engaged,
    inputs_valid=valid,
    steering_pressed=steering_pressed,
    actuator_constrained=constrained,
    standstill=standstill,
  )


def test_first_frame_is_not_false_acceleration_evidence():
  sample = update(LearningMeasurementBuilder(), timestamp=1.0)
  assert not sample.valid
  assert not sample.clean
  assert sample.dt_s == 0.0
  assert sample.rack_acceleration_deg_s2 == 0.0


def test_measured_rate_acceleration_and_gravity_convention_are_exact():
  builder = LearningMeasurementBuilder()
  update(builder, timestamp=1.0, angle=8.0, rate=2.0)
  update(builder, timestamp=1.01, angle=8.1, rate=3.0)
  sample = update(builder, timestamp=1.02, angle=8.2, rate=4.0)
  live = mapping()
  expected_curvature = curvature_from_measured_angle(
    8.2, 10.0, live, mapping(roll=0.0, offset=0.0),
  ).curvature
  expected_lateral_accel = (
    expected_curvature * 100.0
    - live.roll_rad * ACCELERATION_DUE_TO_GRAVITY
    - 0.03
  )

  assert sample.valid
  assert sample.clean
  assert math.isclose(sample.dt_s, 0.01)
  assert math.isclose(sample.rack_acceleration_deg_s2, 100.0)
  assert math.isclose(
    sample.measured_lateral_accel_mps2,
    expected_lateral_accel,
  )


def test_direction_uses_offset_corrected_angle_motion():
  builder = LearningMeasurementBuilder()
  update(
    builder,
    timestamp=1.0,
    angle=8.0,
    rate=4.0,
    live=mapping(offset=1.5),
  )
  offset_only = update(
    builder,
    timestamp=1.01,
    angle=8.1,
    rate=4.0,
    live=mapping(offset=1.6),
  )
  direction_seed = update(
    builder,
    timestamp=1.02,
    angle=8.2,
    rate=4.0,
    live=mapping(offset=1.6),
  )
  resolved = update(
    builder,
    timestamp=1.03,
    angle=8.3,
    rate=4.0,
    live=mapping(offset=1.6),
  )

  assert not offset_only.valid
  assert offset_only.rack_rate_deg_s == 0.0
  assert not direction_seed.valid
  assert resolved.valid
  assert resolved.rack_rate_deg_s == 4.0


def test_unsigned_reversal_is_coverage_not_regression_evidence():
  builder = LearningMeasurementBuilder()
  update(builder, timestamp=1.00, angle=8.0, rate=8.0)
  update(builder, timestamp=1.01, angle=8.1, rate=8.0)
  forward = update(
    builder, timestamp=1.02, angle=8.2, rate=8.0,
  )
  reversal = update(
    builder, timestamp=1.03, angle=8.1, rate=8.0,
  )
  reverse = update(
    builder, timestamp=1.04, angle=8.0, rate=8.0,
  )

  assert forward.clean
  assert not forward.rack_direction_reversal
  assert reversal.valid
  assert reversal.rack_direction_reversal
  assert reversal.rack_rate_deg_s == -8.0
  assert math.isclose(reversal.dt_s, 0.01)
  assert reversal.rack_acceleration_deg_s2 == 0.0
  assert not reversal.clean
  assert not reversal.authority_evidence
  assert reverse.clean
  assert reverse.rack_rate_deg_s == -8.0


def test_driver_excludes_but_valid_constraint_is_continuous_evidence():
  builder = LearningMeasurementBuilder()
  update(builder, timestamp=1.0, angle=8.0, rate=0.0)
  pressed = update(
    builder,
    timestamp=1.01,
    angle=8.1,
    rate=1.0,
    steering_pressed=True,
  )
  update(builder, timestamp=1.02, angle=8.2, rate=2.0)
  update(builder, timestamp=1.03, angle=8.3, rate=2.0)
  constrained_raw = update(
    builder,
    timestamp=1.04,
    angle=8.4,
    rate=3.0,
    constrained=True,
  )
  constrained = _attest_authority_sample(
    constrained_raw,
    boundary=ActuatorBoundary.SLEW_BUILD,
    magnitude_boundary_dwell_s=0.0,
  )
  recovered = update(
    builder, timestamp=1.05, angle=8.5, rate=4.0,
  )

  assert not pressed.valid and not pressed.clean
  assert constrained.valid and not constrained.clean
  assert constrained.authority_evidence
  assert constrained.actuator_constrained
  assert recovered.clean
  assert math.isclose(recovered.rack_acceleration_deg_s2, 100.0)


def test_gap_resets_derivative_evidence():
  builder = LearningMeasurementBuilder()
  update(builder, timestamp=1.0, angle=8.0, rate=0.0)
  update(builder, timestamp=1.01, angle=8.1, rate=1.0)
  gap_time = 1.01 + MAX_CONTINUOUS_MEASUREMENT_GAP_S + 1e-6
  gap = update(
    builder,
    timestamp=gap_time,
    angle=9.0,
    rate=100.0,
  )
  next_seed = update(
    builder,
    timestamp=gap_time + 0.01,
    angle=10.0,
    rate=101.0,
  )
  next_sample = update(
    builder,
    timestamp=gap_time + 0.02,
    angle=11.0,
    rate=102.0,
  )

  assert not gap.valid
  assert gap.dt_s == 0.0
  assert not next_seed.valid
  assert next_sample.clean
  assert math.isclose(next_sample.rack_acceleration_deg_s2, 100.0)


def test_mapping_failure_cannot_bridge_derivative_history():
  builder = LearningMeasurementBuilder()
  update(builder, timestamp=1.0, angle=8.0, rate=2.0)
  update(builder, timestamp=1.01, angle=8.1, rate=3.0)
  assert update(
    builder, timestamp=1.02, angle=8.2, rate=4.0,
  ).valid

  with patch(
    "openpilot.selfdrive.controls.lib.blatv2.measurement.curvature_from_measured_angle",
    side_effect=ValueError("injected mapping singularity"),
  ):
    failed = update(
      builder, timestamp=1.03, angle=8.3, rate=5.0,
    )
  first_after = update(
    builder, timestamp=1.04, angle=8.4, rate=6.0,
  )

  assert not failed.valid
  assert not first_after.valid
  assert first_after.dt_s == 0.0


def test_disengagement_standstill_and_invalid_mapping_reset_history():
  for reset_kwargs in (
    {"engaged": False},
    {"standstill": True},
    {"valid": False},
    {"live": mapping(valid=False)},
  ):
    builder = LearningMeasurementBuilder()
    update(builder, timestamp=1.0, angle=8.0, rate=0.0)
    reset_sample = update(
      builder,
      timestamp=1.01,
      angle=8.1,
      rate=1.0,
      **reset_kwargs,
    )
    first_after = update(
      builder, timestamp=1.02, angle=8.2, rate=2.0,
    )

    assert not reset_sample.clean
    assert not first_after.valid
    assert first_after.dt_s == 0.0


def test_nonfinite_inputs_fail_closed_and_do_not_retain_history():
  builder = LearningMeasurementBuilder()
  update(builder, timestamp=1.0, angle=8.0, rate=0.0)
  invalid = update(
    builder, timestamp=math.nan, angle=8.1, rate=1.0,
  )
  recovered_first = update(
    builder, timestamp=1.02, angle=8.2, rate=2.0,
  )

  assert not invalid.valid
  assert not invalid.clean
  assert not recovered_first.valid
  assert all(math.isfinite(value) for value in (
    invalid.speed_mps,
    invalid.dt_s,
    invalid.applied_torque,
    invalid.measured_lateral_accel_mps2,
    invalid.rack_rate_deg_s,
    invalid.rack_acceleration_deg_s2,
  ))


def test_source_contract_has_no_desired_or_candidate_signal():
  fields = set(__import__(
    "openpilot.selfdrive.controls.lib.blatv2.learner",
    fromlist=["LearningSample"],
  ).LearningSample.__dataclass_fields__)
  forbidden = {"desired", "reference", "candidate", "request", "command"}
  assert not any(
    token in name.lower() for name in fields for token in forbidden
  )

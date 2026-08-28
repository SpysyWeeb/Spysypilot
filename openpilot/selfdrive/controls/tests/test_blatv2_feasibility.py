from __future__ import annotations

import inspect
import math
import random
import struct

from openpilot.selfdrive.controls.lib.blatv2 import feasibility
from openpilot.selfdrive.controls.lib.blatv2.actuator import (
  RuntimeTorqueLimits,
  apply_torque_envelope_counts,
)
from openpilot.selfdrive.controls.lib.blatv2.feasibility import (
  ConstraintReason,
  FeasibilityStatus,
  inspect_current_torque_feasibility,
  project_torque_feasibility_into,
)


# Deliberately non-Hyundai values prove the module consumes runtime limits.
LIMITS = RuntimeTorqueLimits(
  steer_max=271,
  delta_up=3,
  delta_down=8,
  steer_step=2,
  driver_allowance=20,
  driver_multiplier=2,
  driver_factor=1,
)


def project(
  limits: RuntimeTorqueLimits,
  initial_applied_counts: int,
  raw_requests: list[float],
  driver_torques: list[float] | None = None,
  capacity: int | None = None,
  sentinel: float = -12345.0,
):
  count = len(raw_requests)
  buffer_capacity = count if capacity is None else capacity
  drivers = [0.0] * count if driver_torques is None else driver_torques
  outputs = (
    [sentinel] * buffer_capacity,
    [sentinel] * buffer_capacity,
    [sentinel] * buffer_capacity,
    [-12345] * buffer_capacity,
    [-12345] * buffer_capacity,
    [True] * buffer_capacity,
    [-12345] * buffer_capacity,
  )
  status = project_torque_feasibility_into(
    limits,
    initial_applied_counts,
    raw_requests,
    drivers,
    count,
    *outputs,
  )
  return status, outputs


def test_projection_matches_repeated_direct_envelope_calls_randomly():
  limits = LIMITS
  random_source = random.Random(0xB1A7)
  raw_requests = [random_source.uniform(-1.5, 1.5) for _ in range(2000)]
  driver_torques = [random_source.uniform(-300.0, 300.0) for _ in raw_requests]
  initial_counts = -37
  status, outputs = project(
    limits,
    initial_counts,
    raw_requests,
    driver_torques,
  )
  assert status == FeasibilityStatus.OK
  (
    raw_output,
    feasible_output,
    residual_output,
    request_counts_output,
    feasible_counts_output,
    constrained_output,
    reason_output,
  ) = outputs

  previous_counts = initial_counts
  for index, (raw_request, driver_torque) in enumerate(
    zip(raw_requests, driver_torques, strict=True),
  ):
    requested_counts = int(round(raw_request * limits.steer_max))
    feasible_counts = apply_torque_envelope_counts(
      limits,
      requested_counts,
      previous_counts,
      driver_torque,
    )
    feasible_torque = feasible_counts / limits.steer_max
    constrained = feasible_counts != requested_counts
    assert raw_output[index] == raw_request
    assert request_counts_output[index] == requested_counts
    assert feasible_counts_output[index] == feasible_counts
    assert feasible_output[index] == feasible_torque
    assert residual_output[index] == raw_request - feasible_torque
    assert constrained_output[index] is constrained
    assert reason_output[index] == int(ConstraintReason.ACTUATOR_ENVELOPE if constrained else ConstraintReason.NONE)
    previous_counts = feasible_counts


def test_current_helper_matches_direct_envelope_exhaustively():
  limits = LIMITS
  for previous_counts in range(
    -limits.steer_max,
    limits.steer_max + 1,
    19,
  ):
    for requested_counts in range(
      -limits.steer_max - 50,
      limits.steer_max + 51,
      13,
    ):
      raw_request = requested_counts / limits.steer_max
      for driver_torque in (-250.0, -19.0, 0.0, 19.0, 250.0):
        expected = apply_torque_envelope_counts(
          limits,
          requested_counts,
          previous_counts,
          driver_torque,
        )
        result = inspect_current_torque_feasibility(
          limits,
          previous_counts,
          raw_request,
          driver_torque,
        )
        assert result.valid
        assert result.requested_counts == requested_counts
        assert result.feasible_counts == expected
        assert result.feasible_applied_torque == expected / limits.steer_max
        assert result.constraint_active is (expected != requested_counts)


def test_build_release_asymmetry_comes_from_runtime_envelope():
  limits = LIMITS
  building = inspect_current_torque_feasibility(
    limits,
    previous_applied_counts=0,
    raw_requested_torque=1.0,
    driver_torque=0.0,
  )
  releasing = inspect_current_torque_feasibility(
    limits,
    previous_applied_counts=100,
    raw_requested_torque=0.0,
    driver_torque=0.0,
  )
  assert building.feasible_counts == limits.delta_up
  assert releasing.feasible_counts == 100 - limits.delta_down
  assert limits.delta_down > limits.delta_up


def test_sign_crossing_preserves_decay_then_build_semantics():
  limits = LIMITS
  for previous_counts, raw_request, expected_sign in (
    (2, -1.0, -1),
    (-2, 1.0, 1),
  ):
    result = inspect_current_torque_feasibility(
      limits,
      previous_counts,
      raw_request,
      0.0,
    )
    direct = apply_torque_envelope_counts(
      limits,
      int(round(raw_request * limits.steer_max)),
      previous_counts,
      0.0,
    )
    assert result.feasible_counts == direct
    assert math.copysign(1, result.feasible_counts) == expected_sign


def test_driver_torque_interaction_is_preserved():
  limits = LIMITS
  unopposed = inspect_current_torque_feasibility(
    limits,
    0,
    1.0,
    0.0,
  )
  opposed = inspect_current_torque_feasibility(
    limits,
    0,
    1.0,
    -200.0,
  )
  assert unopposed.feasible_counts == limits.delta_up
  assert opposed.feasible_counts == 0
  assert opposed.constraint_active
  assert opposed.constraint_reason == ConstraintReason.ACTUATOR_ENVELOPE


def test_magnitude_saturation_and_signed_residual():
  limits = LIMITS
  positive = inspect_current_torque_feasibility(
    limits,
    limits.steer_max,
    2.0,
    0.0,
  )
  negative = inspect_current_torque_feasibility(
    limits,
    -limits.steer_max,
    -2.0,
    0.0,
  )
  assert positive.feasible_counts == limits.steer_max
  assert positive.unmet_torque > 0.0
  assert negative.feasible_counts == -limits.steer_max
  assert negative.unmet_torque < 0.0


def test_reachable_request_is_strictly_transparent_in_count_space():
  limits = LIMITS
  raw_request = 2 / limits.steer_max
  result = inspect_current_torque_feasibility(
    limits,
    0,
    raw_request,
    0.0,
  )
  assert result.count_exactly_reachable
  assert result.requested_counts == 2
  assert result.feasible_counts == 2
  assert result.feasible_applied_torque == raw_request
  assert result.unmet_torque == 0.0
  assert not result.constraint_active
  assert result.constraint_reason == ConstraintReason.NONE


def test_reachable_non_grid_request_reports_only_count_quantization():
  limits = LIMITS
  raw_request = 0.25 / limits.steer_max
  result = inspect_current_torque_feasibility(
    limits,
    0,
    raw_request,
    0.0,
  )
  assert result.count_exactly_reachable
  assert result.requested_counts == 0
  assert result.feasible_counts == 0
  assert result.feasible_applied_torque == 0.0
  assert result.unmet_torque == raw_request
  assert not result.constraint_active


def test_infeasible_request_preserves_raw_and_reports_residual():
  limits = LIMITS
  result = inspect_current_torque_feasibility(
    limits,
    0,
    1.0,
    0.0,
  )
  assert result.raw_requested_torque == 1.0
  assert result.feasible_applied_torque == limits.delta_up / limits.steer_max
  assert result.unmet_torque == 1.0 - result.feasible_applied_torque
  assert result.constraint_active
  assert not result.count_exactly_reachable


def test_projection_is_bit_repeatable():
  limits = LIMITS
  raw_requests = [0.0, 0.1, 0.4, -0.3, -1.2, 0.02]
  driver_torques = [0.0, 10.0, -30.0, 0.0, 200.0, -200.0]

  def packed_projection() -> bytes:
    status, outputs = project(
      limits,
      -11,
      raw_requests,
      driver_torques,
    )
    (
      raw_output,
      feasible_output,
      residual_output,
      request_counts_output,
      feasible_counts_output,
      constrained_output,
      reason_output,
    ) = outputs
    return (
      struct.pack(
        f"<{len(raw_output) * 3}d",
        *raw_output,
        *feasible_output,
        *residual_output,
      )
      + struct.pack(
        f"<{len(request_counts_output) * 3}q",
        *request_counts_output,
        *feasible_counts_output,
        *reason_output,
      )
      + bytes(constrained_output)
      + bytes((int(status),))
    )

  expected = packed_projection()
  for _ in range(20):
    assert packed_projection() == expected


def test_invalid_raw_request_never_produces_finite_projection():
  limits = LIMITS
  for bad in (math.nan, math.inf, -math.inf, 1e308):
    result = inspect_current_torque_feasibility(limits, 0, bad, 0.0)
    assert result.status == FeasibilityStatus.INVALID_REQUEST
    assert not result.valid
    assert not math.isfinite(result.feasible_applied_torque)
    assert not math.isfinite(result.unmet_torque)
    assert result.constraint_reason == ConstraintReason.INVALID_REQUEST


def test_invalid_driver_torque_never_produces_finite_projection():
  limits = LIMITS
  for bad in (math.nan, math.inf, -math.inf):
    result = inspect_current_torque_feasibility(limits, 0, 0.2, bad)
    assert result.status == FeasibilityStatus.INVALID_DRIVER_TORQUE
    assert not result.valid
    assert not math.isfinite(result.feasible_applied_torque)
    assert not math.isfinite(result.unmet_torque)
    assert result.requested_counts is not None
    assert result.feasible_counts is None


def test_invalid_initial_count_returns_status_not_hot_loop_exception():
  limits = LIMITS
  status, outputs = project(
    limits,
    math.nan,  # type: ignore[arg-type]
    [0.1, 0.2],
  )
  assert status == FeasibilityStatus.INVALID_INITIAL_STATE
  assert all(math.isnan(value) for value in outputs[1])
  assert all(math.isnan(value) for value in outputs[2])
  assert outputs[6] == [
    int(ConstraintReason.INVALID_INITIAL_STATE),
    int(ConstraintReason.INVALID_DEPENDENT_STATE),
  ]


def test_invalid_input_overwrites_dependent_suffix_in_reused_buffers():
  limits = LIMITS
  raw_requests = [0.1, 0.2, 0.3, 0.4]
  driver_torques = [0.0] * len(raw_requests)
  outputs = (
    [99.0] * 6,
    [99.0] * 6,
    [99.0] * 6,
    [99] * 6,
    [99] * 6,
    [True] * 6,
    [99] * 6,
  )
  assert (
    project_torque_feasibility_into(
      limits,
      0,
      raw_requests,
      driver_torques,
      len(raw_requests),
      *outputs,
    )
    == FeasibilityStatus.OK
  )

  invalid_requests = [0.1, math.nan, 0.3, 0.4]
  status = project_torque_feasibility_into(
    limits,
    0,
    invalid_requests,
    driver_torques,
    len(invalid_requests),
    *outputs,
  )
  assert status == FeasibilityStatus.INVALID_REQUEST
  assert math.isfinite(outputs[1][0])
  assert all(math.isnan(value) for value in outputs[1][1:4])
  assert all(math.isnan(value) for value in outputs[2][1:4])
  assert outputs[6][1:4] == [
    int(ConstraintReason.INVALID_REQUEST),
    int(ConstraintReason.INVALID_DEPENDENT_STATE),
    int(ConstraintReason.INVALID_DEPENDENT_STATE),
  ]
  # Capacity beyond the caller-declared prefix remains untouched.
  assert all(output[4:] == [99, 99] for output in outputs[:5])
  assert outputs[5][4:] == [True, True]
  assert outputs[6][4:] == [99, 99]


def test_invalid_driver_overwrites_hot_path_projection():
  limits = LIMITS
  status, outputs = project(
    limits,
    0,
    [0.1, 0.2, 0.3],
    [0.0, math.inf, 0.0],
  )
  assert status == FeasibilityStatus.INVALID_DRIVER_TORQUE
  assert math.isfinite(outputs[1][0])
  assert all(math.isnan(value) for value in outputs[1][1:])
  assert all(math.isnan(value) for value in outputs[2][1:])
  assert outputs[3][1] == int(round(0.2 * limits.steer_max))
  assert outputs[6][1:] == [
    int(ConstraintReason.INVALID_DRIVER_TORQUE),
    int(ConstraintReason.INVALID_DEPENDENT_STATE),
  ]


def test_caller_buffer_prefix_does_not_touch_tail():
  limits = LIMITS
  status, outputs = project(
    limits,
    0,
    [0.0, 0.1, 0.2],
    capacity=6,
  )
  assert status == FeasibilityStatus.OK
  assert all(output[3:] == [-12345.0] * 3 for output in outputs[:3])
  assert all(output[3:] == [-12345] * 3 for output in outputs[3:5])
  assert outputs[5][3:] == [True] * 3
  assert outputs[6][3:] == [-12345] * 3


def test_module_has_no_platform_limits_or_duplicate_limiter():
  source = inspect.getsource(feasibility)
  assert "409" not in source
  assert "STEER_DELTA_UP" not in source
  assert "STEER_DELTA_DOWN" not in source
  assert ".delta_up" not in source
  assert ".delta_down" not in source
  assert "apply_driver_steer_torque_limits" not in source
  assert "apply_torque_envelope_counts" in source

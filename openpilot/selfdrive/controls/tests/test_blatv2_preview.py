from __future__ import annotations

import math

from openpilot.selfdrive.controls.lib.blatv2.actuator import (
  RuntimeTorqueLimits,
  apply_torque_envelope_counts,
)
from openpilot.selfdrive.controls.lib.blatv2.preview import (
  REACHABILITY_FIXED_DT_S,
  REACHABILITY_HORIZON_S,
  REACHABILITY_LAST_SAMPLE_INDEX,
  REACHABILITY_SAMPLE_COUNT,
  ReachabilityStatus,
  ReachableCountProjector,
)


DT = REACHABILITY_FIXED_DT_S
CAPACITY = REACHABILITY_SAMPLE_COUNT
STEER_MAX = 409


def limits(
  *,
  steer_step: int = 1,
  production_envelope_verified: bool = True,
) -> RuntimeTorqueLimits:
  return RuntimeTorqueLimits(
    steer_max=STEER_MAX,
    delta_up=4,
    delta_down=7,
    steer_step=steer_step,
    driver_allowance=50,
    driver_multiplier=2,
    driver_factor=1,
    production_envelope_verified=production_envelope_verified,
  )


def projector() -> ReachableCountProjector:
  return ReachableCountProjector(
    fixed_dt_s=DT,
    limits=limits(),
  )


def torque(counts: int) -> float:
  return counts / STEER_MAX


def held_target(initial_counts: int, target_counts: int, target_index: int) -> list[float]:
  values = [torque(initial_counts)] * CAPACITY
  for index in range(target_index, CAPACITY):
    values[index] = torque(target_counts)
  return values


def update(
  p: ReachableCountProjector,
  ideal_torques: object,
  *,
  previous_applied_counts: object = 0,
  driver_torque: object = 0.0,
  steering_pressed: object = False,
):
  return p.update(
    ideal_torques=ideal_torques,  # type: ignore[arg-type]
    previous_applied_counts=previous_applied_counts,  # type: ignore[arg-type]
    driver_torque=driver_torque,  # type: ignore[arg-type]
    steering_pressed=steering_pressed,  # type: ignore[arg-type]
  )


def assert_raises(
  expected: type[BaseException] | tuple[type[BaseException], ...],
  callback,
) -> None:
  try:
    callback()
  except expected:
    return
  raise AssertionError(f"{expected.__name__} was not raised")


def assert_invalid_and_cleared(result, p: ReachableCountProjector) -> None:
  assert not result.valid
  assert result.status == ReachabilityStatus.INVALID_INPUT
  assert not result.authored_sequence_exactly_reachable
  assert not result.witness_sequence_reachable
  assert p.authored_counts == [0] * CAPACITY
  assert p.reactive_counts == [0] * CAPACITY
  assert p.backward_witness_counts == [0] * CAPACITY
  assert p.witness_counts == [0] * CAPACITY


def assert_exact_witness_transitions(
  p: ReachableCountProjector,
  previous_counts: int,
  driver_torque: float,
) -> None:
  assert (
    apply_torque_envelope_counts(
      limits(),
      p.witness_counts[0],
      previous_counts,
      driver_torque,
    )
    == p.witness_counts[0]
  )
  for index in range(1, CAPACITY):
    assert (
      apply_torque_envelope_counts(
        limits(),
        p.witness_counts[index],
        p.witness_counts[index - 1],
        0.0,
      )
      == p.witness_counts[index]
    )


def test_exact_live_grid_constructor_contract() -> None:
  assert REACHABILITY_HORIZON_S == 2.0
  assert REACHABILITY_LAST_SAMPLE_INDEX == 200
  assert CAPACITY == 201
  assert projector().fixed_dt_s == 0.01

  one_ulp = math.nextafter(DT, math.inf)
  ReachableCountProjector(fixed_dt_s=one_ulp, limits=limits())
  two_ulps = math.nextafter(one_ulp, math.inf)
  for invalid_dt in (0.03, two_ulps, 0.0, math.nan, math.inf, None, "0.01", True):
    assert_raises(
      (TypeError, ValueError),
      lambda invalid_dt=invalid_dt: ReachableCountProjector(
        fixed_dt_s=invalid_dt,  # type: ignore[arg-type]
        limits=limits(),
      ),
    )

  assert_raises(
    ValueError,
    lambda: ReachableCountProjector(fixed_dt_s=DT, limits=limits(steer_step=2)),
  )
  assert_raises(
    ValueError,
    lambda: ReachableCountProjector(
      fixed_dt_s=DT,
      limits=limits(production_envelope_verified=False),
    ),
  )


def test_update_requires_exact_201_sample_prefix() -> None:
  p = projector()
  values = [0.0] * CAPACITY
  assert_invalid_and_cleared(update(p, values[:-1]), p)
  assert_invalid_and_cleared(update(p, values + [0.0]), p)

  result = update(p, values)
  assert result.valid
  assert result.status == ReachabilityStatus.OK
  assert p.authored_counts == [0] * CAPACITY


def test_constant_reachable_authored_sequence_is_exact() -> None:
  p = projector()
  result = update(
    p,
    [torque(120)] * CAPACITY,
    previous_applied_counts=120,
  )
  assert result.valid
  assert result.status == ReachabilityStatus.OK
  assert p.authored_counts == [120] * CAPACITY
  assert p.reactive_counts == [120] * CAPACITY
  assert p.backward_witness_counts == [120] * CAPACITY
  assert p.witness_counts == [120] * CAPACITY
  assert not result.preparation_active_now
  assert not result.preparation_scheduled_later
  assert result.authored_sequence_exactly_reachable
  assert result.witness_sequence_reachable
  assert result.first_authored_miss_index == -1
  assert result.first_authored_miss_time_s is None
  assert result.maximum_absolute_authored_residual_counts == 0
  assert_exact_witness_transitions(p, 120, 0.0)


def test_cached_neutral_images_match_every_production_count_pair() -> None:
  p = projector()
  for previous_counts in range(-STEER_MAX, STEER_MAX + 1):
    for requested_counts in range(-STEER_MAX, STEER_MAX + 1):
      production_counts = apply_torque_envelope_counts(
        limits(),
        requested_counts,
        previous_counts,
        0.0,
      )
      assert p.project_neutral_counts(requested_counts, previous_counts) == production_counts
      target_index = requested_counts + STEER_MAX
      predecessor_contains_previous = p._predecessor_lower[target_index] <= previous_counts <= p._predecessor_upper[target_index]
      assert predecessor_contains_previous == (production_counts == requested_counts)


def test_authored_exactness_uses_every_sample_from_actual_anchor() -> None:
  p = projector()
  exact_counts = [min((index + 1) * limits().delta_up, STEER_MAX) for index in range(CAPACITY)]
  result = update(p, [torque(value) for value in exact_counts])
  assert result.status == ReachabilityStatus.OK
  assert result.authored_sequence_exactly_reachable
  assert p.authored_counts == exact_counts
  assert p.reactive_counts == exact_counts
  assert p.witness_counts == exact_counts

  prepared = update(p, held_target(0, STEER_MAX, 102))
  assert prepared.valid
  assert prepared.status == ReachabilityStatus.AUTHORED_SEQUENCE_MISS
  assert not prepared.authored_sequence_exactly_reachable
  assert prepared.witness_sequence_reachable
  assert prepared.first_authored_miss_index == 0
  assert prepared.first_authored_miss_time_s == 0.0
  assert p.authored_counts[0] - p.witness_counts[0] == -1
  assert prepared.maximum_absolute_authored_residual_counts == 405
  assert p.witness_counts[102] == STEER_MAX


def test_exact_transition_boundaries() -> None:
  cases = (
    ("turn-in", 0, STEER_MAX, 103, ((101, 404, 5, 4), (102, 408, 1, 4), (103, 409, 0, 1))),
    ("release", STEER_MAX, 0, 59, ((57, 10, -10, 402), (58, 3, -3, 402), (59, 0, 0, 406))),
    ("reversal", STEER_MAX, -STEER_MAX, 161, ((159, -404, -5, 402), (160, -408, -1, 402), (161, -409, 0, 405))),
  )
  for _, initial, target, exact_transition, expected in cases:
    for transition_count, delivered, signed_miss, requested_first in expected:
      p = projector()
      target_index = transition_count - 1
      result = update(
        p,
        held_target(initial, target, target_index),
        previous_applied_counts=initial,
      )
      assert result.valid
      assert p.witness_counts[0] == requested_first
      assert p.authored_counts[target_index] == target
      assert p.witness_counts[target_index] == delivered
      assert p.authored_counts[target_index] - p.witness_counts[target_index] == signed_miss
      assert (p.witness_counts[target_index] == target) == (transition_count >= exact_transition)
      assert_exact_witness_transitions(p, initial, 0.0)


def test_full_and_partial_targets_at_zone_times() -> None:
  cases = (
    ("full turn-in", 0, 409, (204, 409, 409)),
    ("partial turn-in", 0, 200, (200, 200, 200)),
    ("full release", 409, 0, (52, 0, 0)),
    ("partial release", 200, 0, (0, 0, 0)),
    ("full reversal", 409, -409, (52, -252, -409)),
    ("partial reversal", 200, -200, (-91, -200, -200)),
  )
  deadline_indices = (50, 120, 200)
  for _, initial, target, expected_deliveries in cases:
    for deadline_index, expected_delivery in zip(deadline_indices, expected_deliveries, strict=True):
      p = projector()
      result = update(
        p,
        held_target(initial, target, deadline_index),
        previous_applied_counts=initial,
      )
      assert result.valid
      assert p.authored_counts[deadline_index] == target
      assert p.witness_counts[deadline_index] == expected_delivery
      assert p.authored_counts[deadline_index] - p.witness_counts[deadline_index] == target - expected_delivery


def test_preparation_now_and_later_are_distinct_reachability_facts() -> None:
  p = projector()
  near = update(p, held_target(0, STEER_MAX, 50))
  assert near.preparation_active_now
  assert near.preparation_scheduled_later
  assert p.witness_counts[0] == limits().delta_up
  assert p.reactive_counts[0] == 0

  far = update(p, held_target(0, STEER_MAX, 200))
  assert not far.preparation_active_now
  assert far.preparation_scheduled_later
  assert p.witness_counts[0] == 0
  assert p.reactive_counts[0] == 0
  assert p.witness_counts[97] == 0
  assert p.witness_counts[98] == 1


def test_initial_and_future_transitions_use_the_exact_driver_contract() -> None:
  p = projector()
  result = update(
    p,
    held_target(0, STEER_MAX, 102),
    previous_applied_counts=0,
    driver_torque=50.0,
  )
  assert result.valid
  assert p.witness_counts[0] == 1
  assert (
    apply_torque_envelope_counts(
      limits(),
      p.witness_counts[0],
      0,
      50.0,
    )
    == p.witness_counts[0]
  )
  assert_exact_witness_transitions(p, 0, 50.0)


def test_driver_allowance_boundaries_suppress_unknown_future() -> None:
  for target in (-STEER_MAX, STEER_MAX):
    for driver in (-51.0, -50.0, 0.0, 50.0, 51.0):
      p = projector()
      result = update(
        p,
        [torque(target)] * CAPACITY,
        previous_applied_counts=0,
        driver_torque=driver,
      )
      expected_reactive = apply_torque_envelope_counts(limits(), target, 0, driver)
      assert result.valid
      assert p.reactive_counts[0] == expected_reactive
      if abs(driver) > limits().driver_allowance:
        assert result.status == ReachabilityStatus.DRIVER_SUPPRESSED
        assert p.witness_counts[0] == expected_reactive
        assert p.reactive_counts[1:] == [0] * (CAPACITY - 1)
        assert p.witness_counts[1:] == [0] * (CAPACITY - 1)
        assert not result.preparation_active_now
        assert not result.preparation_scheduled_later
        assert not result.authored_sequence_exactly_reachable
        assert not result.witness_sequence_reachable
      else:
        assert result.status == ReachabilityStatus.AUTHORED_SEQUENCE_MISS
        assert result.witness_sequence_reachable


def test_driver_and_pressed_state_are_replanned_without_stale_future() -> None:
  p = projector()
  ideal = held_target(0, STEER_MAX, 102)
  result_id = id(p.result)

  suppressed = update(p, ideal, driver_torque=-51.0)
  assert id(suppressed) == result_id
  assert suppressed.status == ReachabilityStatus.DRIVER_SUPPRESSED
  assert p.witness_counts[0] == p.reactive_counts[0] == 0

  neutral = update(p, ideal, driver_torque=0.0)
  assert id(neutral) == result_id
  assert neutral.witness_sequence_reachable
  assert p.witness_counts[0] == 1
  assert neutral.preparation_active_now

  pressed = update(p, ideal, driver_torque=0.0, steering_pressed=True)
  assert pressed.status == ReachabilityStatus.DRIVER_SUPPRESSED
  assert p.witness_counts[0] == p.reactive_counts[0] == 0
  assert not pressed.witness_sequence_reachable

  allowance_edge = update(p, ideal, driver_torque=-50.0, steering_pressed=False)
  assert allowance_edge.witness_sequence_reachable
  assert p.witness_counts[0] == 1

  opposite_suppressed = update(p, ideal, driver_torque=51.0)
  assert opposite_suppressed.status == ReachabilityStatus.DRIVER_SUPPRESSED
  assert not opposite_suppressed.witness_sequence_reachable


def test_authored_counts_clamp_and_scaled_overflow_fails_closed() -> None:
  p = projector()
  values = [0.0] * CAPACITY
  values[0] = 2.0
  values[-1] = -1.25
  result = update(p, values)
  assert result.valid
  assert p.authored_counts[0] == STEER_MAX
  assert p.authored_counts[-1] == -STEER_MAX

  extreme = [0.5] * CAPACITY
  extreme[100] = 1e308
  assert_invalid_and_cleared(update(p, extreme), p)


class LengthFailure:
  def __len__(self) -> int:
    raise RuntimeError("bad length")


class IndexFailure:
  def __len__(self) -> int:
    return CAPACITY

  def __getitem__(self, index: int) -> float:
    raise RuntimeError("bad index")


def test_malformed_runtime_inputs_fail_closed_without_raising() -> None:
  p = projector()
  valid_values = [0.0] * CAPACITY
  assert update(p, valid_values).valid
  bad_middle = [0.5] * CAPACITY
  bad_middle[100] = math.nan
  bad_last = [-0.5] * CAPACITY
  bad_last[-1] = None  # type: ignore[assignment]

  invalid_calls = (
    lambda: update(p, None),
    lambda: update(p, "0" * CAPACITY),
    lambda: update(p, LengthFailure()),
    lambda: update(p, IndexFailure()),
    lambda: update(p, valid_values, previous_applied_counts=True),
    lambda: update(p, valid_values, previous_applied_counts=0.0),
    lambda: update(p, valid_values, previous_applied_counts=None),
    lambda: update(p, valid_values, previous_applied_counts="0"),
    lambda: update(p, valid_values, previous_applied_counts=STEER_MAX + 1),
    lambda: update(p, valid_values, driver_torque=None),
    lambda: update(p, valid_values, driver_torque="0"),
    lambda: update(p, valid_values, driver_torque=math.nan),
    lambda: update(p, valid_values, driver_torque=math.inf),
    lambda: update(p, valid_values, driver_torque=1e308),
    lambda: update(p, valid_values, driver_torque=10**10000),
    lambda: update(p, valid_values, steering_pressed=1),
    lambda: update(p, valid_values, steering_pressed=None),
    lambda: update(p, [False] * CAPACITY),
    lambda: update(p, [None] * CAPACITY),
    lambda: update(p, ["0"] * CAPACITY),
    lambda: update(p, [math.nan] * CAPACITY),
    lambda: update(p, [math.inf] * CAPACITY),
    lambda: update(p, bad_middle),
    lambda: update(p, bad_last),
  )
  for invalid_call in invalid_calls:
    assert_invalid_and_cleared(invalid_call(), p)


def test_reused_result_and_arrays_are_deterministic_after_invalid_input() -> None:
  p = projector()
  ideal = [math.sin(index * 0.07) for index in range(CAPACITY)]
  result_id = id(p.result)
  array_ids = tuple(
    id(array)
    for array in (
      p.authored_counts,
      p.reactive_counts,
      p.backward_witness_counts,
      p.witness_counts,
    )
  )

  first = update(
    p,
    ideal,
    previous_applied_counts=-17,
    driver_torque=3.25,
  )
  first_snapshot = (
    tuple(getattr(first, field) for field in first.__slots__),
    tuple(p.authored_counts),
    tuple(p.reactive_counts),
    tuple(p.backward_witness_counts),
    tuple(p.witness_counts),
  )
  assert_invalid_and_cleared(update(p, ideal, driver_torque=None), p)

  second = update(
    p,
    ideal,
    previous_applied_counts=-17,
    driver_torque=3.25,
  )
  second_snapshot = (
    tuple(getattr(second, field) for field in second.__slots__),
    tuple(p.authored_counts),
    tuple(p.reactive_counts),
    tuple(p.backward_witness_counts),
    tuple(p.witness_counts),
  )
  assert id(second) == result_id
  assert (
    tuple(
      id(array)
      for array in (
        p.authored_counts,
        p.reactive_counts,
        p.backward_witness_counts,
        p.witness_counts,
      )
    )
    == array_ids
  )
  assert first_snapshot == second_snapshot

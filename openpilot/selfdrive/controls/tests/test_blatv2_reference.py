from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect
import math
import struct
import unittest

from openpilot.selfdrive.controls.lib.blatv2 import reference as reference_module
from openpilot.selfdrive.controls.lib.blatv2.contracts import (
  FrameTiming,
)
from openpilot.selfdrive.controls.lib.blatv2.reference import (
  compile_reference,
  compile_reference_into,
  sample_reference,
  sample_reference_into,
)


def rates_for_curvatures(
  curvatures: list[float],
  speeds: list[float],
) -> list[float]:
  return [curvature * speed for curvature, speed in zip(curvatures, speeds, strict=True)]


def result_bytes(result) -> bytes:
  values = (
    *result.times_s,
    *result.curvatures,
    *result.curvature_rates,
    *result.curvature_accelerations,
    *result.planned_speeds,
    result.scalar_curvature,
    result.scalar_action_plan_s,
  )
  return struct.pack(f"<{len(values)}d", *values) + bytes((result.valid, result.scalar_only))


def query_result_bytes(result) -> bytes:
  values = (
    *result.times_s,
    *result.curvatures,
    *result.curvature_rates,
    *result.curvature_accelerations,
    *result.planned_speeds,
    *result.planned_speed_rates,
    *result.planned_speed_accelerations,
    result.scalar_curvature,
    result.scalar_action_plan_s,
    result.plan_time_now_s,
    result.measured_v_ego,
  )
  return struct.pack(f"<{len(values)}d", *values) + bytes((result.valid, result.scalar_only, result.degraded))


class TestBLaTv2Reference(unittest.TestCase):
  def setUp(self) -> None:
    self.times = [0.0, 0.1, 0.2, 0.3, 0.4]
    self.speeds = [10.0, 10.5, 11.0, 11.5, 12.0]
    self.curvatures = [0.001, 0.002, 0.004, 0.007, 0.011]
    self.rates = rates_for_curvatures(self.curvatures, self.speeds)

  def compile(
    self,
    *,
    scalar: float = 0.02,
    action: float = 0.15,
    horizon: float = 0.4,
    times: list[float] | None = None,
    rates: list[float] | None = None,
    speeds: list[float] | None = None,
  ):
    return compile_reference(
      self.times if times is None else times,
      self.rates if rates is None else rates,
      self.speeds if speeds is None else speeds,
      scalar,
      action,
      horizon,
    )

  def sample(
    self,
    *,
    scalar: float = 0.02,
    action: float = 0.15,
    plan_now: float = 0.125,
    measured_speed: float = 9.75,
    queries: list[float] | None = None,
    times: list[float] | None = None,
    rates: list[float] | None = None,
    speeds: list[float] | None = None,
  ):
    return sample_reference(
      self.times if times is None else times,
      self.rates if rates is None else rates,
      self.speeds if speeds is None else speeds,
      scalar,
      action,
      plan_now,
      measured_speed,
      [0.1, 0.15, 0.25, 0.35] if queries is None else queries,
    )

  def test_exact_between_node_scalar_anchor(self) -> None:
    result = self.compile(action=0.15)
    action_index = result.times_s.index(0.15)
    self.assertEqual(result.curvatures[action_index], 0.02)
    self.assertTrue(result.valid)
    self.assertFalse(result.scalar_only)

  def test_constant_plan_is_scalar_everywhere(self) -> None:
    speeds = [7.0, 8.0, 9.0, 10.0, 11.0]
    curvature = 0.006
    result = self.compile(
      scalar=-0.013,
      rates=rates_for_curvatures([curvature] * len(speeds), speeds),
      speeds=speeds,
    )
    self.assertTrue(all(value == -0.013 for value in result.curvatures))
    self.assertTrue(
      all(abs(value) <= 1e-15 for value in result.curvature_rates),
    )
    self.assertTrue(
      all(abs(value) <= 1e-14 for value in result.curvature_accelerations),
    )

  def test_invalid_model_data_returns_scalar_only(self) -> None:
    invalid_cases = (
      (self.times, self.rates, [10.0, 10.5, 0.0, 11.5, 12.0]),
      (
        self.times,
        [self.rates[0], self.rates[1], math.nan, *self.rates[3:]],
        self.speeds,
      ),
      (self.times, self.rates[:-1], self.speeds),
      ([0.0, 0.1, 0.1, 0.3, 0.4], self.rates, self.speeds),
    )
    for times, rates, speeds in invalid_cases:
      with self.subTest(times=times, rates=rates, speeds=speeds):
        result = compile_reference(
          times,
          rates,
          speeds,
          0.025,
          0.15,
          0.4,
        )
        self.assertFalse(result.valid)
        self.assertTrue(result.scalar_only)
        self.assertEqual(result.times_s, (0.15,))
        self.assertEqual(result.curvatures, (0.025,))
        self.assertEqual(result.curvature_rates, (0.0,))
        self.assertEqual(result.curvature_accelerations, (0.0,))
        self.assertEqual(result.planned_speeds, (0.0,))

  def test_repeated_runs_are_byte_identical(self) -> None:
    first = result_bytes(self.compile())
    for _ in range(20):
      self.assertEqual(result_bytes(self.compile()), first)

  def test_time_origin_translation_changes_only_times(self) -> None:
    original = self.compile(action=0.15, horizon=0.4)
    shift = 17.0
    shifted = self.compile(
      action=0.15 + shift,
      horizon=0.4 + shift,
      times=[value + shift for value in self.times],
    )
    self.assertEqual(
      shifted.times_s,
      tuple(value + shift for value in original.times_s),
    )
    for shifted_value, original_value in zip(
      shifted.curvatures,
      original.curvatures,
      strict=True,
    ):
      self.assertAlmostEqual(shifted_value, original_value, places=12)
    for shifted_value, original_value in zip(
      shifted.curvature_rates,
      original.curvature_rates,
      strict=True,
    ):
      self.assertAlmostEqual(shifted_value, original_value, places=12)
    for shifted_value, original_value in zip(
      shifted.curvature_accelerations,
      original.curvature_accelerations,
      strict=True,
    ):
      self.assertAlmostEqual(shifted_value, original_value, places=10)
    for shifted_value, original_value in zip(
      shifted.planned_speeds,
      original.planned_speeds,
      strict=True,
    ):
      self.assertAlmostEqual(shifted_value, original_value, places=12)

  def test_scalar_shift_changes_only_position(self) -> None:
    lower = self.compile(scalar=-0.03)
    upper = self.compile(scalar=0.07)
    self.assertEqual(lower.times_s, upper.times_s)
    for lower_value, upper_value in zip(
      lower.curvatures,
      upper.curvatures,
      strict=True,
    ):
      self.assertAlmostEqual(upper_value - lower_value, 0.1, places=15)
    self.assertEqual(lower.curvature_rates, upper.curvature_rates)
    self.assertEqual(
      lower.curvature_accelerations,
      upper.curvature_accelerations,
    )
    self.assertEqual(lower.planned_speeds, upper.planned_speeds)

  def test_monotonic_plan_has_no_interpolation_overshoot(self) -> None:
    result = self.compile(action=0.15)
    self.assertTrue(
      all(
        right >= left
        for left, right in zip(
          result.curvatures,
          result.curvatures[1:],
          strict=False,
        )
      ),
    )
    self.assertGreaterEqual(min(result.curvatures), result.curvatures[0])
    self.assertLessEqual(max(result.curvatures), result.curvatures[-1])

  def test_one_plan_blip_leaves_no_residual(self) -> None:
    clean_before = self.compile()
    blipped_rates = list(self.rates)
    blipped_rates[2] += 0.5
    blipped = self.compile(rates=blipped_rates)
    clean_after = self.compile()
    self.assertNotEqual(result_bytes(blipped), result_bytes(clean_before))
    self.assertEqual(result_bytes(clean_after), result_bytes(clean_before))

  def test_explicit_action_and_physical_effect_queries_are_distinct(self) -> None:
    action_time = 0.15
    physical_effect_time = 0.28
    result = self.sample(
      action=action_time,
      queries=[action_time, physical_effect_time],
    )
    self.assertTrue(result.valid)
    self.assertEqual(result.curvatures[0], 0.02)
    self.assertNotEqual(result.curvatures[1], result.curvatures[0])
    self.assertGreater(result.curvatures[1], result.curvatures[0])

  def test_speed_is_anchored_to_measured_speed_between_nodes(self) -> None:
    result = self.sample(
      plan_now=0.15,
      measured_speed=8.25,
      queries=[0.1, 0.15, 0.2],
    )
    self.assertTrue(result.valid)
    self.assertEqual(result.planned_speeds[1], 8.25)
    self.assertAlmostEqual(result.planned_speeds[0], 8.0, places=14)
    self.assertAlmostEqual(result.planned_speeds[2], 8.5, places=14)

  def test_speed_pchip_derivatives_are_analytic(self) -> None:
    result = self.sample(
      plan_now=0.15,
      measured_speed=8.25,
      queries=[0.05, 0.15, 0.275, 0.4],
    )
    for speed_rate in result.planned_speed_rates:
      self.assertAlmostEqual(speed_rate, 5.0, places=12)
    for speed_acceleration in result.planned_speed_accelerations:
      self.assertAlmostEqual(speed_acceleration, 0.0, places=10)
    # The supported endpoint retains the analytic endpoint derivative. It is
    # not silently changed into a terminal zero-rate clamp.
    self.assertEqual(result.times_s[-1], self.times[-1])
    self.assertAlmostEqual(result.planned_speed_rates[-1], 5.0, places=12)

  def test_out_of_support_query_degrades_whole_request(self) -> None:
    result = self.sample(
      scalar=-0.027,
      plan_now=0.15,
      measured_speed=6.5,
      queries=[0.15, 0.3, 0.41],
    )
    self.assertFalse(result.valid)
    self.assertTrue(result.scalar_only)
    self.assertTrue(result.degraded)
    self.assertEqual(result.times_s, (0.15, 0.3, 0.41))
    self.assertEqual(result.curvatures, (-0.027, -0.027, -0.027))
    self.assertEqual(result.curvature_rates, (0.0, 0.0, 0.0))
    self.assertEqual(result.curvature_accelerations, (0.0, 0.0, 0.0))
    self.assertEqual(result.planned_speeds, (6.5, 6.5, 6.5))
    self.assertEqual(result.planned_speed_rates, (0.0, 0.0, 0.0))
    self.assertEqual(
      result.planned_speed_accelerations,
      (0.0, 0.0, 0.0),
    )

  def test_scalar_and_measured_speed_anchors_are_independent(self) -> None:
    baseline = self.sample()
    scalar_shifted = self.sample(scalar=0.12)
    speed_shifted = self.sample(measured_speed=12.75)

    self.assertEqual(baseline.times_s, scalar_shifted.times_s)
    self.assertEqual(baseline.times_s, speed_shifted.times_s)
    for baseline_value, shifted_value in zip(
      baseline.curvatures,
      scalar_shifted.curvatures,
      strict=True,
    ):
      self.assertAlmostEqual(shifted_value - baseline_value, 0.1, places=15)
    self.assertEqual(
      baseline.curvature_rates,
      scalar_shifted.curvature_rates,
    )
    self.assertEqual(
      baseline.curvature_accelerations,
      scalar_shifted.curvature_accelerations,
    )
    self.assertEqual(baseline.planned_speeds, scalar_shifted.planned_speeds)
    self.assertEqual(
      baseline.planned_speed_rates,
      scalar_shifted.planned_speed_rates,
    )
    self.assertEqual(
      baseline.planned_speed_accelerations,
      scalar_shifted.planned_speed_accelerations,
    )

    self.assertEqual(baseline.curvatures, speed_shifted.curvatures)
    self.assertEqual(
      baseline.curvature_rates,
      speed_shifted.curvature_rates,
    )
    self.assertEqual(
      baseline.curvature_accelerations,
      speed_shifted.curvature_accelerations,
    )
    for baseline_value, shifted_value in zip(
      baseline.planned_speeds,
      speed_shifted.planned_speeds,
      strict=True,
    ):
      self.assertAlmostEqual(shifted_value - baseline_value, 3.0, places=15)
    self.assertEqual(
      baseline.planned_speed_rates,
      speed_shifted.planned_speed_rates,
    )
    self.assertEqual(
      baseline.planned_speed_accelerations,
      speed_shifted.planned_speed_accelerations,
    )

  def test_explicit_query_is_monotonic_without_overshoot(self) -> None:
    times = [index * 0.01 for index in range(41)]
    result = self.sample(queries=times)
    self.assertTrue(
      all(
        right >= left
        for left, right in zip(
          result.curvatures,
          result.curvatures[1:],
          strict=False,
        )
      ),
    )
    self.assertTrue(
      all(
        right >= left
        for left, right in zip(
          result.planned_speeds,
          result.planned_speeds[1:],
          strict=False,
        )
      ),
    )
    self.assertTrue(result.curvatures[0] <= min(result.curvatures) <= max(result.curvatures) <= result.curvatures[-1])
    self.assertTrue(result.planned_speeds[0] <= min(result.planned_speeds) <= max(result.planned_speeds) <= result.planned_speeds[-1])

  def test_explicit_query_is_repeatedly_bit_identical(self) -> None:
    first = query_result_bytes(self.sample())
    for _ in range(20):
      self.assertEqual(query_result_bytes(self.sample()), first)

  def test_explicit_query_is_invariant_to_time_origin_translation(self) -> None:
    original = self.sample()
    shift = 11.0
    shifted = self.sample(
      action=0.15 + shift,
      plan_now=0.125 + shift,
      queries=[value + shift for value in original.times_s],
      times=[value + shift for value in self.times],
    )
    self.assertEqual(
      shifted.times_s,
      tuple(value + shift for value in original.times_s),
    )
    for shifted_field, original_field in (
      (shifted.curvatures, original.curvatures),
      (shifted.curvature_rates, original.curvature_rates),
      (
        shifted.curvature_accelerations,
        original.curvature_accelerations,
      ),
      (shifted.planned_speeds, original.planned_speeds),
      (shifted.planned_speed_rates, original.planned_speed_rates),
      (
        shifted.planned_speed_accelerations,
        original.planned_speed_accelerations,
      ),
    ):
      for shifted_value, original_value in zip(
        shifted_field,
        original_field,
        strict=True,
      ):
        self.assertAlmostEqual(shifted_value, original_value, places=9)

  def test_reference_has_no_implicit_timing_or_smoothing_dependency(self) -> None:
    source = inspect.getsource(reference_module)
    self.assertNotIn("LAT_SMOOTH_SECONDS", source)
    self.assertNotIn("lateralDelay", source)
    self.assertNotIn("DT_MDL", source)

  def test_published_action_time_is_independent_of_plant_delay(self) -> None:
    early_plant = FrameTiming(
      state_sample_mono_ns=100_000_000_000,
      control_witness_mono_ns=100_000_000_000,
      plan_origin_mono_ns=99_900_000_000,
      plan_publication_mono_ns=99_950_000_000,
      scalar_action_plan_s=0.237,
      transport_delay_s=0.08,
    )
    late_plant = FrameTiming(
      state_sample_mono_ns=100_000_000_000,
      control_witness_mono_ns=100_000_000_000,
      plan_origin_mono_ns=99_900_000_000,
      plan_publication_mono_ns=99_950_000_000,
      scalar_action_plan_s=0.237,
      transport_delay_s=0.30,
    )
    early_reference = self.compile(
      action=early_plant.scalar_action_plan_s,
    )
    late_reference = self.compile(
      action=late_plant.scalar_action_plan_s,
    )
    self.assertEqual(
      early_plant.scalar_action_effect_time_s,
      late_plant.scalar_action_effect_time_s,
    )
    self.assertNotEqual(
      early_plant.plant_effect_time_s,
      late_plant.plant_effect_time_s,
    )
    self.assertEqual(
      result_bytes(early_reference),
      result_bytes(late_reference),
    )
    self.assertIn(0.237, early_reference.times_s)

  def test_canonical_timing_is_immutable_and_validated(self) -> None:
    timing = FrameTiming(
      10_000_000_000,
      10_000_000_000,
      9_700_000_000,
      9_800_000_000,
      0.2,
      0.1,
    )
    self.assertAlmostEqual(timing.plan_age_s, 0.2)
    self.assertEqual(timing.state_age_s, 0.0)
    self.assertAlmostEqual(timing.scalar_action_effect_time_s, 9.9)
    self.assertAlmostEqual(timing.plant_effect_time_s, 10.1)
    self.assertAlmostEqual(timing.total_prediction_horizon_s, 0.1)
    with self.assertRaises(FrozenInstanceError):
      timing.scalar_action_plan_s = 0.3
    with self.assertRaises(ValueError):
      FrameTiming(
        10_000_000_000,
        10_000_000_000,
        9_700_000_000,
        10_100_000_000,
        0.2,
        0.1,
      )
    with self.assertRaises(ValueError):
      FrameTiming(
        10_000_000_000,
        10_000_000_000,
        9_700_000_000,
        9_800_000_000,
        math.nan,
        0.1,
      )
    with self.assertRaises(ValueError):
      FrameTiming(
        10_000_000_000,
        10_000_000_000,
        9_900_000_000,
        9_800_000_000,
        0.2,
        0.1,
      )
    with self.assertRaises(ValueError):
      FrameTiming(
        10_000_000_001,
        10_000_000_000,
        9_700_000_000,
        9_800_000_000,
        0.2,
        0.1,
      )

  def test_caller_buffer_api_and_horizon_action_boundary(self) -> None:
    capacity = len(self.times) + 1
    outputs = tuple([0.0] * capacity for _ in range(5))
    scratch_curvatures = [0.0] * len(self.times)
    scratch_tangents = [0.0] * len(self.times)
    status = compile_reference_into(
      self.times,
      self.rates,
      self.speeds,
      0.02,
      0.15,
      0.15,
      *outputs,
      scratch_curvatures,
      scratch_tangents,
    )
    self.assertTrue(status.valid)
    self.assertEqual(status.count, 3)
    self.assertEqual(outputs[0][: status.count], [0.0, 0.1, 0.15])
    self.assertEqual(outputs[1][status.count - 1], 0.02)

  def test_explicit_query_populates_only_caller_owned_buffers(self) -> None:
    queries = [0.1, 0.15, 0.225, 0.35]
    capacity = 8
    sentinel = -12345.0
    outputs = tuple([sentinel] * capacity for _ in range(7))
    scratch = tuple([0.0] * len(self.times) for _ in range(3))
    status = sample_reference_into(
      self.times,
      self.rates,
      self.speeds,
      0.02,
      0.15,
      0.125,
      9.75,
      queries,
      len(queries),
      *outputs,
      *scratch,
    )
    self.assertTrue(status.valid)
    self.assertFalse(status.degraded)
    self.assertEqual(status.count, len(queries))
    self.assertEqual(outputs[0][: status.count], queries)
    self.assertEqual(outputs[1][1], 0.02)
    self.assertEqual(outputs[4][0], 9.625)
    for output in outputs:
      self.assertEqual(output[status.count :], [sentinel] * 4)


if __name__ == "__main__":
  unittest.main()

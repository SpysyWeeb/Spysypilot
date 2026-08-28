from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect
import math
import struct
import unittest

from openpilot.selfdrive.controls.lib.blatv2.intent import (
  INTENT_CAPACITY,
  MAX_MODEL_PUBLICATION_AGE_NS,
  IntentStatusCode,
  adapt_model_intent_into,
)


def rates_for_curvatures(
  curvatures: list[float],
  speeds: list[float],
) -> list[float]:
  return [
    curvature * speed
    for curvature, speed in zip(curvatures, speeds, strict=True)
  ]


def result_bytes(result, outputs: tuple[list[float], ...]) -> bytes:
  if result.frame is None:
    frame_mono_values = (0,) * 4
    frame_values = (math.nan,) * 2
    frame_flags = (False,) * 4
  else:
    timing = result.frame.timing
    frame_mono_values = (
      timing.state_sample_mono_ns,
      timing.control_witness_mono_ns,
      timing.plan_origin_mono_ns,
      timing.plan_publication_mono_ns,
    )
    frame_values = (
      timing.scalar_action_plan_s,
      timing.transport_delay_s,
    )
    validity = result.frame.validity
    frame_flags = (
      validity.model_valid,
      validity.plan_valid,
      validity.vehicle_state_valid,
      validity.calibration_valid,
    )
  status = result.status
  status_values = (
    status.publication_age_s,
    status.state_age_s,
    status.plan_time_now_s,
    status.scalar_deadline_mono_s,
    status.physical_effect_mono_s,
    status.physical_effect_plan_s,
    status.total_prediction_horizon_s,
  )
  status_integers = (
    int(status.code),
    status.count,
    status.model_frame_id,
  )
  status_flags = (
    status.scalar_valid,
    status.future_valid,
    status.scalar_only,
    status.action_query_supported,
    status.current_query_supported,
    status.effect_query_supported,
    status.usable,
    status.stale,
  )
  floats = (*frame_values, *status_values, *(value for output in outputs for value in output))
  return (
    struct.pack("<4Q", *frame_mono_values)
    + struct.pack(f"<{len(floats)}d", *floats)
    + struct.pack("<3q", *status_integers)
    + bytes((*frame_flags, *status_flags))
  )


class TestBLaTv2Intent(unittest.TestCase):
  def setUp(self) -> None:
    self.times = [0.0, 0.1, 0.2, 0.3, 0.4]
    self.speeds = [8.0, 8.5, 9.0, 9.5, 10.0]
    self.curvatures = [0.001, 0.002, 0.004, 0.007, 0.011]
    self.rates = rates_for_curvatures(self.curvatures, self.speeds)
    self.arguments = {
      "state_sample_mono_ns": 10_295_000_000,
      "control_witness_mono_ns": 10_300_000_000,
      "model_publication_mono_ns": 10_250_000_000,
      "plan_origin_mono_ns": 10_100_000_000,
      "model_frame_id": 123,
      "message_valid": True,
      "message_alive": True,
      "scalar_desired_curvature": 0.0125,
      "published_desired_curvature_time_s": 0.275,
      "native_plan_times_s": self.times,
      "native_orientation_rates_z": self.rates,
      "native_velocities_x": self.speeds,
      "current_v_ego_m_s": 9.25,
      "physical_transport_delay_s": 0.12,
    }

  @staticmethod
  def outputs(fill: float = 0.0) -> tuple[list[float], ...]:
    return tuple([fill] * INTENT_CAPACITY for _ in range(4))

  def adapt(self, outputs=None, **overrides):
    caller_outputs = self.outputs() if outputs is None else outputs
    arguments = dict(self.arguments)
    arguments.update(overrides)
    result = adapt_model_intent_into(
      **arguments,
      output_plan_times_s=caller_outputs[0],
      output_orientation_rates_z=caller_outputs[1],
      output_velocities_x=caller_outputs[2],
      output_plan_curvatures=caller_outputs[3],
    )
    return result, caller_outputs

  def test_distinct_clocks_and_raw_plan_are_exact(self) -> None:
    result, outputs = self.adapt()
    self.assertEqual(result.status.code, IntentStatusCode.OK)
    self.assertEqual(result.status.model_frame_id, 123)
    self.assertAlmostEqual(result.status.publication_age_s, 0.05)
    self.assertAlmostEqual(result.status.state_age_s, 0.005)
    self.assertAlmostEqual(result.status.plan_time_now_s, 0.2)
    self.assertAlmostEqual(result.status.scalar_deadline_mono_s, 10.375)
    self.assertAlmostEqual(result.status.physical_effect_mono_s, 10.42)
    self.assertAlmostEqual(result.status.physical_effect_plan_s, 0.32)
    self.assertAlmostEqual(
      result.status.total_prediction_horizon_s,
      0.125,
    )
    self.assertEqual(
      result.frame.timing.state_sample_mono_ns,
      10_295_000_000,
    )
    self.assertEqual(
      result.frame.timing.control_witness_mono_ns,
      10_300_000_000,
    )
    self.assertEqual(
      result.frame.timing.plan_origin_mono_ns,
      10_100_000_000,
    )
    self.assertEqual(
      result.frame.timing.plan_publication_mono_ns,
      10_250_000_000,
    )
    self.assertEqual(result.frame.timing.scalar_action_plan_s, 0.275)
    self.assertEqual(outputs[0][:5], self.times)
    self.assertEqual(outputs[1][:5], self.rates)
    self.assertEqual(outputs[2][:5], self.speeds)
    for actual, expected in zip(outputs[3][:5], self.curvatures, strict=True):
      self.assertAlmostEqual(actual, expected)

  def test_published_action_time_is_only_action_timing_input(self) -> None:
    delay_parameters = [
      name
      for name in inspect.signature(adapt_model_intent_into).parameters
      if "delay" in name
    ]
    self.assertEqual(delay_parameters, ["physical_transport_delay_s"])
    early, _ = self.adapt(
      published_desired_curvature_time_s=0.237,
      physical_transport_delay_s=0.08,
    )
    late, _ = self.adapt(
      published_desired_curvature_time_s=0.237,
      physical_transport_delay_s=0.16,
    )
    self.assertEqual(
      early.frame.timing.scalar_action_plan_s,
      late.frame.timing.scalar_action_plan_s,
    )
    self.assertEqual(
      early.status.scalar_deadline_mono_s,
      late.status.scalar_deadline_mono_s,
    )
    self.assertNotEqual(
      early.status.physical_effect_mono_s,
      late.status.physical_effect_mono_s,
    )

  def test_one_missing_publication_period_is_accepted_then_stale(self) -> None:
    control_ns = self.arguments["control_witness_mono_ns"]
    accepted, _ = self.adapt(
      model_publication_mono_ns=control_ns - MAX_MODEL_PUBLICATION_AGE_NS,
    )
    stale, _ = self.adapt(
      model_publication_mono_ns=control_ns - MAX_MODEL_PUBLICATION_AGE_NS - 1,
    )
    self.assertEqual(accepted.status.code, IntentStatusCode.OK)
    self.assertTrue(accepted.frame.validity.model_valid)
    self.assertEqual(stale.status.code, IntentStatusCode.MESSAGE_STALE)
    self.assertTrue(stale.status.stale)
    self.assertFalse(stale.frame.validity.model_valid)
    self.assertEqual(stale.status.count, 0)

  def test_negative_and_raced_timestamps_are_invalid(self) -> None:
    cases = (
      {"state_sample_mono_ns": True},
      {"state_sample_mono_ns": -1},
      {"state_sample_mono_ns": math.nan},
      {"control_witness_mono_ns": -1},
      {"model_publication_mono_ns": -1},
      {"plan_origin_mono_ns": -1},
      {"state_sample_mono_ns": 10_300_000_001},
      {"model_publication_mono_ns": 10_300_000_001},
      {"plan_origin_mono_ns": 10_250_000_001},
      {"published_desired_curvature_time_s": -0.001},
      {"physical_transport_delay_s": -0.001},
    )
    for overrides in cases:
      with self.subTest(overrides=overrides):
        result, outputs = self.adapt(**overrides)
        self.assertIsNone(result.frame)
        self.assertEqual(result.status.code, IntentStatusCode.INVALID_TIMING)
        self.assertEqual(result.status.count, 0)
        self.assertTrue(all(value == 0.0 for output in outputs for value in output))

  def test_finite_scalar_survives_malformed_future(self) -> None:
    rates = list(self.rates)
    rates[2] = math.nan
    result, outputs = self.adapt(native_orientation_rates_z=rates)
    self.assertEqual(
      result.status.code,
      IntentStatusCode.SCALAR_ONLY_MALFORMED_FUTURE,
    )
    self.assertTrue(result.status.usable)
    self.assertTrue(result.status.scalar_only)
    self.assertFalse(result.status.future_valid)
    self.assertTrue(result.frame.validity.model_valid)
    self.assertFalse(result.frame.validity.plan_valid)
    self.assertEqual(result.status.count, 1)
    self.assertEqual(outputs[0][0], 0.275)
    self.assertEqual(outputs[3][0], 0.0125)
    self.assertTrue(all(value == 0.0 for value in outputs[1]))
    self.assertTrue(all(value == 0.0 for value in outputs[2]))
    self.assertTrue(all(value == 0.0 for value in outputs[0][1:]))
    self.assertTrue(all(value == 0.0 for value in outputs[3][1:]))

  def test_malformed_grids_and_zero_speed_are_scalar_only(self) -> None:
    cases = (
      {"native_plan_times_s": [0.0, 0.1, 0.1, 0.3, 0.4]},
      {"native_plan_times_s": [0.0, -0.1, 0.2, 0.3, 0.4]},
      {"native_plan_times_s": [0.0, 0.1, math.nan, 0.3, 0.4]},
      {"native_orientation_rates_z": self.rates[:-1]},
      {"native_velocities_x": [8.0, 8.5, 0.0, 9.5, 10.0]},
    )
    for overrides in cases:
      with self.subTest(overrides=overrides):
        result, _ = self.adapt(**overrides)
        self.assertEqual(
          result.status.code,
          IntentStatusCode.SCALAR_ONLY_MALFORMED_FUTURE,
        )
        self.assertTrue(result.status.scalar_only)

  def test_development_mode_uses_only_a_query_complete_positive_prefix(
    self,
  ) -> None:
    speeds = list(self.speeds)
    speeds[-1] = -0.01
    strict, _ = self.adapt(
      native_velocities_x=speeds,
      physical_transport_delay_s=0.09,
    )
    self.assertEqual(
      strict.status.code,
      IntentStatusCode.SCALAR_ONLY_MALFORMED_FUTURE,
    )

    truncated, outputs = self.adapt(
      native_velocities_x=speeds,
      physical_transport_delay_s=0.09,
      allow_truncated_future_prefix=True,
    )
    self.assertEqual(
      truncated.status.code,
      IntentStatusCode.MALFORMED_SUFFIX_TRUNCATED,
    )
    self.assertTrue(truncated.status.usable)
    self.assertTrue(truncated.status.future_valid)
    self.assertFalse(truncated.status.scalar_only)
    self.assertEqual(truncated.status.count, 4)
    self.assertTrue(truncated.frame.validity.plan_valid)
    self.assertEqual(outputs[0][:4], self.times[:4])
    self.assertEqual(outputs[1][:4], self.rates[:4])
    self.assertEqual(outputs[2][:4], self.speeds[:4])
    self.assertTrue(
      all(value == 0.0 for output in outputs for value in output[4:])
    )

    speeds[3] = 0.0
    unsupported, _ = self.adapt(
      native_velocities_x=speeds,
      physical_transport_delay_s=0.09,
      allow_truncated_future_prefix=True,
    )
    self.assertEqual(
      unsupported.status.code,
      IntentStatusCode.SCALAR_ONLY_MALFORMED_FUTURE,
    )
    self.assertTrue(unsupported.status.scalar_only)
    self.assertFalse(unsupported.status.action_query_supported)

  def test_action_current_and_effect_query_support_are_explicit(self) -> None:
    unsupported_action, _ = self.adapt(
      published_desired_curvature_time_s=0.45,
    )
    self.assertFalse(unsupported_action.status.action_query_supported)
    self.assertTrue(unsupported_action.status.current_query_supported)
    self.assertTrue(unsupported_action.status.effect_query_supported)

    unsupported_current, _ = self.adapt(
      plan_origin_mono_ns=10_250_000_000,
      native_plan_times_s=[0.1, 0.2, 0.3, 0.4, 0.5],
    )
    self.assertTrue(unsupported_current.status.action_query_supported)
    self.assertFalse(unsupported_current.status.current_query_supported)
    self.assertTrue(unsupported_current.status.effect_query_supported)

    unsupported_effect, _ = self.adapt(
      physical_transport_delay_s=0.25,
    )
    self.assertTrue(unsupported_effect.status.action_query_supported)
    self.assertTrue(unsupported_effect.status.current_query_supported)
    self.assertFalse(unsupported_effect.status.effect_query_supported)

    for result in (
      unsupported_action,
      unsupported_current,
      unsupported_effect,
    ):
      self.assertEqual(
        result.status.code,
        IntentStatusCode.SCALAR_ONLY_UNSUPPORTED_QUERY,
      )
      self.assertTrue(result.status.scalar_only)

  def test_invalid_future_cannot_retain_previous_plan(self) -> None:
    outputs = self.outputs(fill=99.0)
    valid, _ = self.adapt(outputs=outputs)
    self.assertEqual(valid.status.code, IntentStatusCode.OK)
    self.assertNotEqual(outputs[1][1], 0.0)
    malformed, _ = self.adapt(
      outputs=outputs,
      native_velocities_x=[8.0, 8.5, 0.0, 9.5, 10.0],
    )
    self.assertTrue(malformed.status.scalar_only)
    self.assertEqual(outputs[0][0], 0.275)
    self.assertEqual(outputs[3][0], 0.0125)
    self.assertTrue(all(value == 0.0 for output in outputs for value in output[1:]))
    self.assertTrue(all(value == 0.0 for value in outputs[1]))
    self.assertTrue(all(value == 0.0 for value in outputs[2]))

  def test_repeated_runs_are_byte_deterministic(self) -> None:
    first_result, first_outputs = self.adapt()
    expected = result_bytes(first_result, first_outputs)
    for _ in range(20):
      result, outputs = self.adapt()
      self.assertEqual(result_bytes(result, outputs), expected)

  def test_plan_age_uses_publication_while_queries_use_plan_origin(self) -> None:
    baseline, _ = self.adapt()
    older_publication, _ = self.adapt(
      model_publication_mono_ns=10_230_000_000,
    )
    older_origin, _ = self.adapt(
      plan_origin_mono_ns=10_080_000_000,
    )
    self.assertNotEqual(
      older_publication.status.publication_age_s,
      baseline.status.publication_age_s,
    )
    self.assertEqual(
      older_publication.status.plan_time_now_s,
      baseline.status.plan_time_now_s,
    )
    self.assertEqual(
      older_origin.status.publication_age_s,
      baseline.status.publication_age_s,
    )
    self.assertNotEqual(
      older_origin.status.plan_time_now_s,
      baseline.status.plan_time_now_s,
    )

  def test_message_vehicle_and_scalar_validity_are_legible(self) -> None:
    cases = (
      (
        {"message_alive": False},
        IntentStatusCode.MESSAGE_NOT_ALIVE,
      ),
      (
        {"message_valid": False},
        IntentStatusCode.MESSAGE_INVALID,
      ),
      (
        {"scalar_desired_curvature": math.nan},
        IntentStatusCode.INVALID_SCALAR,
      ),
      (
        {"current_v_ego_m_s": math.nan},
        IntentStatusCode.INVALID_VEHICLE_STATE,
      ),
    )
    for overrides, expected in cases:
      with self.subTest(overrides=overrides):
        result, _ = self.adapt(**overrides)
        self.assertEqual(result.status.code, expected)
        self.assertFalse(result.status.usable)
        self.assertEqual(result.status.count, 0)

  def test_canonical_result_is_immutable(self) -> None:
    result, _ = self.adapt()
    with self.assertRaises(FrozenInstanceError):
      result.status.count = 2
    with self.assertRaises(FrozenInstanceError):
      result.frame.timing.scalar_action_plan_s = 0.3

  def test_capacity_errors_are_not_silently_truncated(self) -> None:
    short_outputs = (
      [0.0] * (INTENT_CAPACITY - 1),
      *self.outputs()[1:],
    )
    with self.assertRaisesRegex(ValueError, "exactly 33 samples"):
      self.adapt(outputs=short_outputs)

    long_times = [index * 0.01 for index in range(INTENT_CAPACITY + 1)]
    long_speeds = [10.0] * len(long_times)
    long_rates = [0.01] * len(long_times)
    with self.assertRaisesRegex(ValueError, "exceeds fixed capacity 33"):
      self.adapt(
        native_plan_times_s=long_times,
        native_orientation_rates_z=long_rates,
        native_velocities_x=long_speeds,
      )


if __name__ == "__main__":
  unittest.main()

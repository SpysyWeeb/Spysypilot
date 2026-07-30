import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from openpilot.selfdrive.controls.lib.blatv2.plant import (
  AlignInputs,
  AlignParams,
  AlignRuntimeTerms,
  PlantParams,
  PlantSensitivity,
  PlantState,
  PlantTwin,
  SignedRackRate,
)


SPEED_NODES = (2.5, 5.5, 8.5, 12.0, 16.5, 21.0)
TORQUE_PER_LATACCEL = (0.85, 0.39, 0.38, 0.36, 0.286, 0.288)


def params() -> PlantParams:
  return PlantParams(
    k_t=4000.0,
    b_steer=10.0,
    t_breakaway=0.09,
    actuation_delay=0.12,
    steer_max=409,
    delta_up=4,
    delta_down=7,
    steer_step=1,
    provisional=True,
    torque_per_lataccel_speed_nodes=SPEED_NODES,
    torque_per_lataccel_values=TORQUE_PER_LATACCEL,
  )


def align_params() -> AlignParams:
  return AlignParams(
    mass=2000.0,
    wheelbase=3.0,
    center_to_front=1.2,
    tire_stiffness_front=100000.0,
    tire_stiffness_rear=110000.0,
    nominal_steer_ratio=15.0,
    steer_ratio_rear=0.0,
    lat_accel_offset=0.0,
  )


def assert_state_close(
  case: unittest.TestCase,
  actual: PlantState,
  expected: PlantState,
) -> None:
  case.assertTrue(
    math.isclose(actual.angle_deg, expected.angle_deg, abs_tol=1e-12),
  )
  case.assertTrue(
    math.isclose(actual.rate_deg_s, expected.rate_deg_s, abs_tol=1e-12),
  )
  case.assertEqual(actual.applied_torque, expected.applied_torque)
  case.assertEqual(actual.v_ego, expected.v_ego)


class TestPlantTwin(unittest.TestCase):
  def test_hyundai_unsigned_rack_rate_recovers_angle_direction(self) -> None:
    rate = SignedRackRate()
    self.assertEqual(rate.update(10.0, 4.0), 0.0)
    self.assertEqual(rate.update(10.1, 4.0), 4.0)
    self.assertEqual(rate.update(10.1, 4.0), 4.0)
    self.assertEqual(rate.update(10.1, 0.0), 0.0)
    self.assertEqual(rate.update(10.0, 4.0), -4.0)
    self.assertEqual(rate.update(10.0, 8.0), -8.0)

  def test_already_signed_negative_rack_rate_is_preserved(self) -> None:
    rate = SignedRackRate()
    self.assertEqual(rate.update(0.0, -12.0), -12.0)
    self.assertEqual(rate.update(0.0, 12.0), -12.0)
    rate.reset()
    self.assertEqual(rate.update(0.0, 12.0), 0.0)

  def test_signed_rack_rate_rejects_nonfinite_inputs(self) -> None:
    rate = SignedRackRate()
    for angle, reported in ((math.nan, 0.0), (0.0, math.inf)):
      with self.subTest(angle=angle, reported=reported):
        with self.assertRaises(ValueError):
          rate.update(angle, reported)

  def test_load_seed_uses_runtime_limits(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      seed = Path(directory) / "seed.json"
      seed.write_text("""{
        "k_t": 4000.0, "b_steer": 10.0, "t_breakaway": 0.09,
        "actuation_delay": 0.12, "provisional": true,
        "steady_state_torque_per_lateral_accel": {
          "speed_nodes_mps": [2.5, 5.5, 8.5, 12.0, 16.5, 21.0],
          "torque_per_mps2": [0.85, 0.39, 0.38, 0.36, 0.286, 0.288]
        }
      }""")
      limits = SimpleNamespace(STEER_MAX=123, STEER_DELTA_UP=2, STEER_DELTA_DOWN=5, STEER_STEP=2)
      loaded = PlantParams.from_seed_file(seed, limits)
      self.assertEqual((loaded.steer_max, loaded.delta_up, loaded.delta_down, loaded.steer_step), (123, 2, 5, 2))

  def test_slew_asymmetry(self) -> None:
    cases = (
      (0.0, 1.0, 4 / 409),
      (0.5, 1.0, 0.5 + 4 / 409),
      (0.5, 0.0, 0.5 - 7 / 409),
      (-0.5, -1.0, -0.5 - 4 / 409),
      (-0.5, 0.0, -0.5 + 7 / 409),
      (0.5, 0.501, 0.501),
    )
    twin = PlantTwin(params(), align_params())
    for previous, requested, expected in cases:
      with self.subTest(previous=previous, requested=requested):
        self.assertTrue(math.isclose(twin.apply_slew(previous, requested), expected, abs_tol=1e-15))

  def test_slew_crossing_spends_decay_before_build(self) -> None:
    twin = PlantTwin(params(), align_params())
    previous = 2 / 409
    expected = -(4 / 409) * (1.0 - 2 / 7)
    self.assertTrue(math.isclose(twin.apply_slew(previous, -1.0), expected, abs_tol=1e-15))

  def test_slew_crossing_cannot_reach_zero_in_one_frame(self) -> None:
    twin = PlantTwin(params(), align_params())
    self.assertTrue(math.isclose(twin.apply_slew(8 / 409, -1.0), 1 / 409, abs_tol=1e-15))

  def test_stiction_and_zero_crossing_are_deterministic(self) -> None:
    twin = PlantTwin(params(), align_params())
    stuck = PlantState(10.0, 0.0, 0.0, 0.0)
    angle, rate = twin.predict(stuck, [0.08] * 20, 0.01)
    self.assertEqual(angle, (10.0,) * 20)
    self.assertEqual(rate, (0.0,) * 20)

    moving = PlantState(0.0, 1.0, 0.0, 0.0)
    _, moving_rate = twin.predict(moving, [-1.0] * 20, 0.01)
    self.assertTrue(all(value >= 0.0 for value in moving_rate[:12]))

  def test_calibrated_steady_state_gain_at_every_speed_node(self) -> None:
    twin = PlantTwin(params(), align_params())
    inputs = AlignInputs(
      roll=0.0,
      angle_offset_deg=0.0,
      stiffness_factor=1.0,
      steer_ratio=15.0,
      valid=True,
    )
    lateral_accel = 0.5
    for speed, expected_gain in zip(
      SPEED_NODES, TORQUE_PER_LATACCEL, strict=True
    ):
      with self.subTest(speed=speed):
        curvature = lateral_accel / (speed * speed)
        angle = twin.angle_from_curvature(curvature, speed, inputs)
        state = PlantState(angle, 0.0, 0.0, speed)
        steady_torque = twin.aligning_torque(state, inputs)
        self.assertTrue(
          math.isclose(
            abs(steady_torque) / lateral_accel,
            expected_gain,
            abs_tol=1e-15,
          )
        )
        next_state = twin.advance_applied(
          state, steady_torque, 0.01, inputs
        )
        self.assertEqual(next_state.angle_deg, state.angle_deg)
        self.assertEqual(next_state.rate_deg_s, 0.0)

  def test_calibrated_gain_interpolates_and_extrapolates_flat(self) -> None:
    p = params()
    self.assertEqual(p.torque_per_lateral_accel(0.0), 0.85)
    self.assertEqual(p.torque_per_lateral_accel(30.0), 0.288)
    self.assertTrue(
      math.isclose(
        p.torque_per_lateral_accel(4.0),
        (0.85 + 0.39) / 2.0,
        abs_tol=1e-15,
      )
    )

  def test_one_step_residual_does_not_redelay_applied_torque(self) -> None:
    twin = PlantTwin(params(), align_params())
    state = PlantState(0.0, 0.0, 0.0, 0.0)
    next_state = PlantState(0.004, 0.4, 0.1, 0.0)
    self.assertTrue(
      math.isclose(
        twin.one_step_residual(state, 0.1, next_state),
        0.0,
        abs_tol=1e-12,
      )
    )

  def test_predict_holds_measured_torque_until_zoh_delay_expires(self) -> None:
    delayed = params().with_actuation_delay(0.02)
    twin = PlantTwin(delayed, align_params())
    state = PlantState(0.0, 0.0, 0.0, 0.0)
    _, rates = twin.predict(state, [1.0] * 30, 0.01)
    self.assertEqual(rates[:2], (0.0, 0.0))
    self.assertGreater(rates[15], 0.0)

  def test_held_state_prediction_is_identity_at_zero_delay(self) -> None:
    twin = PlantTwin(params(), align_params())
    inputs = AlignInputs(0.01, 0.2, 0.9, 15.5, True)
    state = PlantState(12.0, -4.0, 0.3, 5.0)
    target = PlantState(0.0, 0.0, 0.0, 0.0)
    twin.predict_held_state_into(
      state, 0.0, inputs, 0.05, target, 0.05,
    )
    self.assertEqual(
      (target.angle_deg, target.rate_deg_s, target.applied_torque, target.v_ego),
      (state.angle_deg, state.rate_deg_s, state.applied_torque, state.v_ego),
    )

  def test_applied_history_prediction_uses_queued_torque_in_order(self) -> None:
    twin = PlantTwin(params(), align_params())
    inputs = AlignInputs(0.0, 0.0, 1.0, 15.0, True)
    state = PlantState(0.0, 0.0, 0.0, 0.0)
    history = [0.1, 0.2, 0.3]
    predicted = PlantState(0.0, 0.0, 0.0, 0.0)
    expected = PlantState(0.0, 0.0, 0.0, 0.0)
    twin.predict_applied_history_into(
      state, 0.03, history, 0, 3, inputs, 0.0, predicted, 0.01,
    )
    for applied in history:
      expected = twin.advance_applied(
        expected, applied, 0.01, inputs,
      )
    assert_state_close(self, predicted, expected)

  def test_applied_history_prediction_holds_current_torque_for_unfilled_prefix(
    self,
  ) -> None:
    twin = PlantTwin(params(), align_params())
    inputs = AlignInputs(0.0, 0.0, 1.0, 15.0, True)
    state = PlantState(0.0, 0.0, 0.2, 0.0)
    history = [0.8, 99.0, 99.0]
    predicted = PlantState(0.0, 0.0, 0.0, 0.0)
    expected = PlantState(0.0, 0.0, 0.2, 0.0)
    twin.predict_applied_history_into(
      state, 0.03, history, 0, 1, inputs, 0.0, predicted, 0.01,
    )
    for applied in (0.2, 0.2, 0.8):
      expected = twin.advance_applied(
        expected, applied, 0.01, inputs,
      )
    assert_state_close(self, predicted, expected)

  def test_applied_history_prediction_reads_circular_buffer_oldest_first(
    self,
  ) -> None:
    twin = PlantTwin(params(), align_params())
    inputs = AlignInputs(0.0, 0.0, 1.0, 15.0, True)
    state = PlantState(0.0, 0.0, 0.0, 0.0)
    history = [0.3, 0.1, 0.2]
    predicted = PlantState(0.0, 0.0, 0.0, 0.0)
    expected = PlantState(0.0, 0.0, 0.0, 0.0)
    twin.predict_applied_history_into(
      state, 0.03, history, 1, 3, inputs, 0.0, predicted, 0.01,
    )
    for applied in (0.1, 0.2, 0.3):
      expected = twin.advance_applied(
        expected, applied, 0.01, inputs,
      )
    assert_state_close(self, predicted, expected)

  def test_prepared_applied_history_matches_public_plant_contract(self) -> None:
    twin = PlantTwin(params(), align_params())
    inputs = AlignInputs(0.015, -0.3, 0.92, 15.7, True)
    state = PlantState(37.0, -8.0, 0.21, 6.5)
    history = [0.18, 0.14, 0.08, -0.02, -0.09]
    expected = PlantState(0.0, 0.0, 0.0, 0.0)
    prepared = PlantState(0.0, 0.0, 0.0, 0.0)
    terms = AlignRuntimeTerms()
    twin.predict_applied_history_into(
      state, 0.05, history, 3, 5, inputs, 0.025, expected, 0.01,
    )
    twin.prepare_align_runtime_terms(state.v_ego, inputs, terms)
    twin.predict_applied_history_prepared_into(
      state,
      0.05,
      history,
      3,
      5,
      0.025,
      terms,
      prepared,
      0.01,
    )
    assert_state_close(self, prepared, expected)

  def test_terminal_sensitivity_matches_neighboring_prepared_rollout(self) -> None:
    twin = PlantTwin(params(), align_params())
    inputs = AlignInputs(0.01, 0.2, 0.95, 15.2, True)
    state = PlantState(14.0, 4.0, 0.12, 5.0)
    terms = AlignRuntimeTerms()
    target = PlantState(0.0, 0.0, 0.0, 0.0)
    sensitivity = PlantSensitivity()
    twin.prepare_align_runtime_terms(state.v_ego, inputs, terms)
    request = 0.31
    epsilon = 1e-7
    twin.predict_constant_request_sensitivity_into(
      state,
      0.27,
      request,
      0.02,
      terms,
      target,
      sensitivity,
      0.01,
    )
    neighbor = PlantState(0.0, 0.0, 0.0, 0.0)
    twin.predict_constant_request_prepared_into(
      state,
      0.27,
      request + epsilon,
      0.02,
      terms,
      neighbor,
      0.01,
    )
    self.assertTrue(math.isclose(
      (neighbor.angle_deg - target.angle_deg) / epsilon,
      sensitivity.angle_per_torque,
      rel_tol=2e-5,
      abs_tol=2e-5,
    ))
    self.assertTrue(math.isclose(
      (neighbor.rate_deg_s - target.rate_deg_s) / epsilon,
      sensitivity.rate_per_torque,
      rel_tol=2e-5,
      abs_tol=2e-5,
    ))

  def test_allocation_free_rollout_accepts_live_delay_without_rebuilding_twin(self) -> None:
    twin = PlantTwin(params(), align_params())
    state = PlantState(0.0, 0.0, 0.0, 0.0)
    inputs = AlignInputs(roll=0.0, angle_offset_deg=0.0, stiffness_factor=1.0, steer_ratio=15.0, valid=True)
    requested = [1.0] * 30
    applied = [0.0] * 30
    angles = [0.0] * 30
    no_delay_rates = [0.0] * 30
    delayed_rates = [0.0] * 30
    twin.predict_into(state, requested, 30, 0.01, inputs, applied, angles, no_delay_rates, actuation_delay=0.0)
    twin.predict_into(state, requested, 30, 0.01, inputs, applied, angles, delayed_rates, actuation_delay=0.1)
    self.assertGreater(no_delay_rates[12], delayed_rates[12])

  def test_aligning_load_uses_offset_corrected_angle_and_live_roll(self) -> None:
    twin = PlantTwin(params(), align_params())
    state = PlantState(-10.0, 0.0, 0.0, 15.0)
    nominal = AlignInputs(roll=0.0, angle_offset_deg=0.0, stiffness_factor=1.0, steer_ratio=15.0, valid=True)
    corrected = AlignInputs(roll=0.02, angle_offset_deg=1.0, stiffness_factor=0.9, steer_ratio=16.0, valid=True)
    self.assertTrue(math.isfinite(twin.aligning_torque(state, nominal)))
    self.assertNotEqual(twin.aligning_torque(state, nominal), twin.aligning_torque(state, corrected))

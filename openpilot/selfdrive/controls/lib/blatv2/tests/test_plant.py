import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from openpilot.selfdrive.controls.lib.blatv2.plant import AlignInputs, AlignParams, PlantParams, PlantState, PlantTwin


def params() -> PlantParams:
  return PlantParams(
    k_t=4000.0,
    b_steer=10.0,
    t_breakaway=0.05,
    actuation_delay=0.12,
    steer_max=409,
    delta_up=4,
    delta_down=7,
    steer_step=1,
    provisional=True,
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
    lat_accel_factor=2.5,
    lat_accel_offset=0.0,
  )


class TestPlantTwin(unittest.TestCase):
  def test_load_seed_uses_runtime_limits(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      seed = Path(directory) / "seed.json"
      seed.write_text("""{
        "k_t": 4000.0, "b_steer": 10.0, "t_breakaway": 0.05,
        "actuation_delay": 0.12, "provisional": true
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
    angle, rate = twin.predict(stuck, [0.04] * 20, 0.01)
    self.assertEqual(angle, (10.0,) * 20)
    self.assertEqual(rate, (0.0,) * 20)

    moving = PlantState(0.0, 1.0, 0.0, 0.0)
    _, moving_rate = twin.predict(moving, [-1.0] * 20, 0.01)
    self.assertTrue(all(value >= 0.0 for value in moving_rate[:12]))

  def test_one_step_residual_does_not_redelay_applied_torque(self) -> None:
    twin = PlantTwin(params(), align_params())
    state = PlantState(0.0, 0.0, 0.0, 0.0)
    next_state = PlantState(0.02, 2.0, 0.1, 0.0)
    self.assertTrue(math.isclose(twin.one_step_residual(state, 0.1, next_state), 0.0))

  def test_predict_holds_measured_torque_until_zoh_delay_expires(self) -> None:
    delayed = params().with_actuation_delay(0.02)
    twin = PlantTwin(delayed, align_params())
    state = PlantState(0.0, 0.0, 0.0, 0.0)
    _, rates = twin.predict(state, [1.0] * 10, 0.01)
    self.assertEqual(rates[:2], (0.0, 0.0))
    self.assertGreater(rates[7], 0.0)

  def test_allocation_free_rollout_accepts_live_delay_without_rebuilding_twin(self) -> None:
    twin = PlantTwin(params(), align_params())
    state = PlantState(0.0, 0.0, 0.0, 0.0)
    inputs = AlignInputs(roll=0.0, angle_offset_deg=0.0, stiffness_factor=1.0, steer_ratio=15.0, valid=True)
    requested = [1.0] * 20
    applied = [0.0] * 20
    angles = [0.0] * 20
    no_delay_rates = [0.0] * 20
    delayed_rates = [0.0] * 20
    twin.predict_into(state, requested, 20, 0.01, inputs, applied, angles, no_delay_rates, actuation_delay=0.0)
    twin.predict_into(state, requested, 20, 0.01, inputs, applied, angles, delayed_rates, actuation_delay=0.1)
    self.assertGreater(no_delay_rates[5], delayed_rates[5])

  def test_aligning_load_uses_offset_corrected_angle_and_live_roll(self) -> None:
    twin = PlantTwin(params(), align_params())
    state = PlantState(-10.0, 0.0, 0.0, 15.0)
    nominal = AlignInputs(roll=0.0, angle_offset_deg=0.0, stiffness_factor=1.0, steer_ratio=15.0, valid=True)
    corrected = AlignInputs(roll=0.02, angle_offset_deg=1.0, stiffness_factor=0.9, steer_ratio=16.0, valid=True)
    self.assertTrue(math.isfinite(twin.aligning_torque(state, nominal)))
    self.assertNotEqual(twin.aligning_torque(state, nominal), twin.aligning_torque(state, corrected))

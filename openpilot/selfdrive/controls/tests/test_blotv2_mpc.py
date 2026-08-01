from types import SimpleNamespace
import unittest

import numpy as np
from opendbc.car.interfaces import ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.selfdrive.controls.lib.blotv2 import BLOTV2_ACCEL_MAX, BLOTV2_ACCEL_REQUEST_MAX

from openpilot.selfdrive.controls.lib import longitudinal_planner
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  LEAD_T_IDXS_MODEL,
  T_IDXS,
  LongitudinalMpc,
)
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  get_cruise_accel,
  get_cruise_comfort_accel,
  get_max_accel,
  ordinary_cruise_comfort_enabled,
)


def radar_lead(**overrides):
  values = {
    "present": True,
    "dRel": 20.0,
    "vLead": 8.0,
    "aLeadK": 0.0,
    "aLeadTau": 1.5,
    "modelProb": 1.0,
  }
  values.update(overrides)
  return SimpleNamespace(**values)


def model_lead(**overrides):
  values = {
    "prob": 1.0,
    "x": 8.0 * LEAD_T_IDXS_MODEL,
    "v": np.full_like(LEAD_T_IDXS_MODEL, 8.0),
  }
  values.update(overrides)
  return SimpleNamespace(**values)


class TestModelLeadTrajectory(unittest.TestCase):
  def setUp(self):
    self.mpc = LongitudinalMpc()
    self.mpc.set_cur_state(10.0, 0.0)

  def test_valid_trajectory_is_anchored_to_radar(self):
    trajectory = self.mpc.process_lead_model(model_lead(), radar_lead())
    self.assertAlmostEqual(trajectory[0, 0], 20.0)
    self.assertAlmostEqual(trajectory[0, 1], 8.0)
    np.testing.assert_allclose(trajectory[:, 0], 20.0 + 8.0 * T_IDXS)

  def test_model_contributes_future_shape_not_absolute_offset(self):
    model = model_lead(
      x=100.0 + 8.0 * LEAD_T_IDXS_MODEL,
      v=30.0 + LEAD_T_IDXS_MODEL,
    )
    trajectory = self.mpc.process_lead_model(model, radar_lead())
    self.assertAlmostEqual(trajectory[0, 0], 20.0)
    self.assertAlmostEqual(trajectory[0, 1], 8.0)
    self.assertGreater(trajectory[-1, 1], trajectory[0, 1])

  def test_invalid_model_shape_uses_exact_stock_radar_fallback(self):
    radar = radar_lead(aLeadK=-1.0)
    expected = self.mpc.process_lead(radar)
    actual = self.mpc.process_lead_model(model_lead(x=[0.0], v=[0.0]), radar)
    np.testing.assert_allclose(actual, expected)

  def test_nonfinite_model_uses_exact_stock_radar_fallback(self):
    radar = radar_lead(aLeadK=-1.0)
    expected = self.mpc.process_lead(radar)
    x = np.array(model_lead().x)
    x[2] = np.nan
    actual = self.mpc.process_lead_model(model_lead(x=x), radar)
    np.testing.assert_allclose(actual, expected)

  def test_missing_radar_cannot_activate_model_trajectory(self):
    radar = radar_lead(present=False)
    expected = self.mpc.process_lead(radar)
    actual = self.mpc.process_lead_model(model_lead(), radar)
    np.testing.assert_allclose(actual, expected)

  def test_runtime_policy_range_keeps_solver_finite(self):
    radar_state = SimpleNamespace(
      leadOne=radar_lead(),
      leadTwo=radar_lead(present=False),
    )
    models = [model_lead(), model_lead(prob=0.0)]
    for jerk_scale, t_follow in ((1.0, 1.0), (0.5, 1.45), (0.3, 1.9)):
      self.mpc.set_weights(jerk_factor_scale=jerk_scale)
      self.mpc.update(
        radar_state,
        t_follow=t_follow,
        model_leads=models,
      )
      self.assertEqual(self.mpc.solution_status, 0)
      self.assertTrue(np.all(np.isfinite(self.mpc.v_solution)))
      self.assertTrue(np.all(np.isfinite(self.mpc.a_solution)))
      np.testing.assert_allclose(self.mpc.params[:, 4], t_follow)


class TestStrongCruiseEnvelope(unittest.TestCase):
  def test_low_speed_requests_blotv1_authority_with_platform_clamp(self):
    self.assertEqual(BLOTV2_ACCEL_REQUEST_MAX, 4.0)
    self.assertEqual(BLOTV2_ACCEL_MAX, min(ACCEL_MAX, BLOTV2_ACCEL_REQUEST_MAX))
    self.assertEqual(get_max_accel(0.0), BLOTV2_ACCEL_MAX)
    speeds = np.linspace(0.0, 50.0, 101)
    self.assertLessEqual(max(get_max_accel(speed) for speed in speeds), BLOTV2_ACCEL_MAX)

  def test_launch_schedule_rejoins_stock_gate_at_10_mps(self):
    np.testing.assert_allclose(
      longitudinal_planner.A_CRUISE_MAX_VALS,
      [4.0, 1.2, 0.8, 0.6],
    )
    np.testing.assert_allclose(
      longitudinal_planner.A_CRUISE_MAX_BP,
      [0.0, 10.0, 25.0, 40.0],
    )

    stock_bp = [0.0, 10.0, 25.0, 40.0]
    stock_vals = [1.6, 1.2, 0.8, 0.6]
    speeds = np.linspace(10.0, 50.0, 81)
    np.testing.assert_allclose(
      [get_max_accel(speed) for speed in speeds],
      np.interp(speeds, stock_bp, stock_vals),
    )

  def test_route_d2_urban_cruise_cap(self):
    # Route 000000d2--a62f0c1831 reached 1.79 m/s² near 40 mph for a
    # 5.4 mph set-speed error. Fade the extra launch authority out by 10 m/s,
    # then use stock's speed gate exactly for this urban correction.
    requested_at_10 = np.interp(
      10.0,
      longitudinal_planner.A_CRUISE_MAX_BP,
      longitudinal_planner.A_CRUISE_MAX_VALS,
    )
    self.assertAlmostEqual(requested_at_10, 1.2)
    stock_at_40_mph = np.interp(40.0 * 0.44704, [0.0, 10.0, 25.0, 40.0], [1.6, 1.2, 0.8, 0.6])
    self.assertAlmostEqual(get_max_accel(40.0 * 0.44704), stock_at_40_mph)

  def test_turn_budget_does_not_clip_straight_launch(self):
    np.testing.assert_allclose(longitudinal_planner._A_TOTAL_MAX_V, [4.0, 4.0])

  def test_existing_blotv2_jerk_tune_is_preserved(self):
    np.testing.assert_allclose(
      longitudinal_planner.J_CRUISE_BP,
      [0.0, 10.0, 25.0, 40.0],
    )
    np.testing.assert_allclose(
      longitudinal_planner.J_CRUISE_VALS,
      [2.0, 1.6, 1.0, 0.6],
    )


class TestOrdinaryCruiseComfort(unittest.TestCase):
  CP = SimpleNamespace(steerRatio=15.0, wheelbase=2.7)
  DT = 0.05

  def settled_target(self, v_cruise, v_ego, accel_coast=-0.3, comfort_enabled=True, e2e=False):
    target = 0.0
    for _ in range(200):
      target = get_cruise_accel(
        e2e,
        v_cruise,
        v_ego,
        target,
        0.0,
        self.CP,
        self.DT,
        accel_coast,
        True,
        comfort_enabled=comfort_enabled,
      )
    return target

  def test_route_d9_five_mph_increase_uses_proportional_target(self):
    v_ego = 74.5 * CV.MPH_TO_MS
    v_cruise = 79.5 * CV.MPH_TO_MS
    expected = longitudinal_planner.CRUISE_COMFORT_ACCEL_KP * (v_cruise - v_ego)

    self.assertAlmostEqual(self.settled_target(v_cruise, v_ego, accel_coast=-0.25), expected)
    self.assertAlmostEqual(expected, 0.402336, places=6)
    self.assertLess(expected, get_max_accel(v_ego))

  def test_route_d9_five_mph_reduction_coasts_instead_of_saturating_brakes(self):
    v_ego = 79.6 * CV.MPH_TO_MS
    v_cruise = 74.6 * CV.MPH_TO_MS
    target = self.settled_target(v_cruise, v_ego, accel_coast=-0.39)

    self.assertAlmostEqual(target, -0.402336, places=6)
    self.assertGreater(target, longitudinal_planner.A_CRUISE_MIN)

  def test_reduction_uses_natural_uphill_coast_and_not_downhill_acceleration(self):
    v_ego = 80.0 * CV.MPH_TO_MS
    v_cruise = 75.0 * CV.MPH_TO_MS

    self.assertAlmostEqual(get_cruise_comfort_accel(v_cruise, v_ego, -0.6), -0.6)
    self.assertAlmostEqual(get_cruise_comfort_accel(v_cruise, v_ego, 0.2), -0.402336, places=6)

  def test_small_corrections_taper_continuously(self):
    v_ego = 75.0 * CV.MPH_TO_MS
    target_2_5_mph = get_cruise_comfort_accel(v_ego + 2.5 * CV.MPH_TO_MS, v_ego, -0.3)
    target_5_mph = get_cruise_comfort_accel(v_ego + 5.0 * CV.MPH_TO_MS, v_ego, -0.3)

    self.assertAlmostEqual(target_2_5_mph, target_5_mph / 2.0)

  def test_large_errors_retain_existing_acceleration_and_braking_limits(self):
    v_ego = 75.0 * CV.MPH_TO_MS

    self.assertAlmostEqual(
      self.settled_target(v_ego + 15.0 * CV.MPH_TO_MS, v_ego),
      get_max_accel(v_ego),
    )
    self.assertAlmostEqual(
      self.settled_target(v_ego - 15.0 * CV.MPH_TO_MS, v_ego),
      longitudinal_planner.A_CRUISE_MIN,
    )

  def test_low_speed_launch_retains_legacy_authority(self):
    v_ego = 5.0
    v_cruise = v_ego + 5.0 * CV.MPH_TO_MS

    comfort = self.settled_target(v_cruise, v_ego, comfort_enabled=True)
    legacy = self.settled_target(v_cruise, v_ego, comfort_enabled=False)
    self.assertAlmostEqual(comfort, legacy)

  def test_existing_jerk_schedule_still_bounds_response(self):
    v_ego = 75.0 * CV.MPH_TO_MS
    jerk = np.interp(v_ego, longitudinal_planner.J_CRUISE_BP, longitudinal_planner.J_CRUISE_VALS)
    first_target = get_cruise_accel(
      False,
      v_ego + 5.0 * CV.MPH_TO_MS,
      v_ego,
      0.0,
      0.0,
      self.CP,
      self.DT,
      -0.3,
      True,
      comfort_enabled=True,
    )

    self.assertAlmostEqual(first_target, jerk * self.DT)

  def test_comfort_eligibility_preserves_other_longitudinal_strategies(self):
    self.assertTrue(ordinary_cruise_comfort_enabled(False, False, True, False))
    self.assertFalse(ordinary_cruise_comfort_enabled(True, False, True, False))
    self.assertFalse(ordinary_cruise_comfort_enabled(False, True, True, False))
    self.assertFalse(ordinary_cruise_comfort_enabled(False, False, False, False))
    self.assertFalse(ordinary_cruise_comfort_enabled(False, False, True, True))
    self.assertFalse(ordinary_cruise_comfort_enabled(False, False, True, False, speed_limiter_active=True))

  def test_disabled_comfort_matches_legacy_targets(self):
    v_ego = 80.0 * CV.MPH_TO_MS

    self.assertAlmostEqual(
      self.settled_target(v_ego + 5.0 * CV.MPH_TO_MS, v_ego, comfort_enabled=False),
      get_max_accel(v_ego),
    )
    self.assertAlmostEqual(
      self.settled_target(v_ego - 5.0 * CV.MPH_TO_MS, v_ego, comfort_enabled=False),
      longitudinal_planner.A_CRUISE_MIN,
    )


if __name__ == "__main__":
  unittest.main()

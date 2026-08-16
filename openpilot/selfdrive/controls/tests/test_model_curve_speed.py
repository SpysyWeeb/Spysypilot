from types import SimpleNamespace
import unittest

import numpy as np

from opendbc.car.hyundai.values import CAR
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX
from openpilot.selfdrive.controls.lib.model_curve_speed import (
  CURVATURE_BP,
  CURVE_TARGET_RELEASE_RATE,
  CURVE_SPEED_V,
  MAX_CURVE_SPEED,
  ModelCurveSpeedLimiter,
  curve_speed_for_curvature,
)
from openpilot.selfdrive.modeld.constants import ModelConstants


def make_model(curvature=None, position_x=None, position_y=None, speed=15.0):
  sample_count = ModelConstants.IDX_N
  curvature = np.zeros(sample_count) if curvature is None else np.asarray(curvature)
  position_x = np.arange(sample_count, dtype=float) if position_x is None else np.asarray(position_x)
  position_y = np.zeros(sample_count) if position_y is None else np.asarray(position_y)
  velocity_x = np.full(sample_count, speed)
  return SimpleNamespace(
    position=SimpleNamespace(x=position_x, y=position_y),
    velocity=SimpleNamespace(x=velocity_x),
    orientationRate=SimpleNamespace(z=curvature * velocity_x),
  )


def make_cp(factor=2.0, offset=0.0, friction=0.1):
  torque = SimpleNamespace(latAccelFactor=factor, latAccelOffset=offset, friction=friction)
  return SimpleNamespace(carFingerprint=CAR.HYUNDAI_PALISADE,
                         lateralTuning=SimpleNamespace(which=lambda: "torque", torque=torque))


class TestModelCurveSpeed(unittest.TestCase):
  def test_field_calibration_points(self):
    for curvature, speed in zip(CURVATURE_BP, CURVE_SPEED_V, strict=True):
      with self.subTest(curvature=curvature):
        self.assertAlmostEqual(curve_speed_for_curvature(curvature), speed)

  def test_curvature_speed_envelope_is_monotonic_and_fail_open_for_straight_path(self):
    curvature = np.array([0.0, CURVATURE_BP[0] / 2.0, *CURVATURE_BP, CURVATURE_BP[-1] * 2.0])
    speed = curve_speed_for_curvature(curvature)

    self.assertEqual(speed[0], MAX_CURVE_SPEED)
    self.assertTrue(np.all(np.diff(speed) <= 0.0))
    self.assertLess(speed[-1], CURVE_SPEED_V[-1])

  def test_single_sample_prediction_spike_is_rejected(self):
    for spike_index in (0, 12, ModelConstants.IDX_N - 1):
      with self.subTest(spike_index=spike_index):
        curvature = np.zeros(ModelConstants.IDX_N)
        curvature[spike_index] = CURVATURE_BP[-1]
        limiter = ModelCurveSpeedLimiter()
        v_cruise = 30.0

        self.assertEqual(limiter.update(make_model(curvature=curvature), v_cruise), v_cruise)
        self.assertFalse(limiter.active)

  def test_default_approach_deceleration_is_half_mps2(self):
    self.assertAlmostEqual(ModelCurveSpeedLimiter().approach_decel, 0.5)

  def test_sustained_curve_caps_cruise_using_approach_distance(self):
    curvature = np.zeros(ModelConstants.IDX_N)
    curvature[12:17] = CURVATURE_BP[1]
    limiter = ModelCurveSpeedLimiter()

    model = make_model(curvature=curvature)
    limiter.update(model, 30.0)
    target = limiter.update(model, 30.0)

    expected = np.sqrt(CURVE_SPEED_V[1] ** 2 + 2.0 * limiter.approach_decel * limiter.distance)
    self.assertTrue(limiter.active)
    self.assertAlmostEqual(limiter.curvature, CURVATURE_BP[1])
    self.assertGreater(limiter.distance, 0.0)
    self.assertAlmostEqual(target, expected)

  def test_curve_target_releases_slowly_until_curve_clears(self):
    limiter = ModelCurveSpeedLimiter()
    tight = make_model(curvature=np.full(ModelConstants.IDX_N, CURVATURE_BP[-1]))
    opening = make_model(curvature=np.full(ModelConstants.IDX_N, CURVATURE_BP[1]))
    straight = make_model()
    v_cruise = 30.0

    limiter.update(tight, v_cruise)
    tight_target = limiter.update(tight, v_cruise)
    limiter.update(opening, v_cruise)
    opening_target = limiter.update(opening, v_cruise)

    self.assertAlmostEqual(opening_target - tight_target, CURVE_TARGET_RELEASE_RATE * DT_MDL)
    limiter.update(straight, v_cruise)
    self.assertEqual(limiter.update(straight, v_cruise), v_cruise)

  def test_single_frame_prediction_spike_is_rejected(self):
    limiter = ModelCurveSpeedLimiter()
    straight_model = make_model()
    spike_model = make_model(curvature=np.full(ModelConstants.IDX_N, CURVATURE_BP[-1]))

    self.assertEqual(limiter.update(straight_model, 30.0), 30.0)
    self.assertEqual(limiter.update(spike_model, 30.0), 30.0)
    self.assertEqual(limiter.update(straight_model, 30.0), 30.0)
    self.assertFalse(limiter.active)

  def test_future_torque_veto_is_debounced_and_fails_safe(self):
    limiter = ModelCurveSpeedLimiter(make_cp())
    curve = make_model(curvature=np.full(ModelConstants.IDX_N, CURVATURE_BP[-1]))

    limiter.update(curve, 30.0, v_ego=10.0, lateral_active=True)
    self.assertFalse(limiter.torque_veto)
    limiter.update(curve, 30.0, v_ego=10.0, lateral_active=True)
    self.assertTrue(limiter.torque_veto)

    curve.orientationRate.z = curve.orientationRate.z[:-1]
    limiter.update(curve, 30.0, v_ego=10.0, lateral_active=True)
    self.assertTrue(limiter.torque_veto)
    limiter.update(curve, 30.0, v_ego=10.0, lateral_active=True)
    self.assertFalse(limiter.torque_veto)

  def test_partial_car_params_keeps_predictor_inactive(self):
    limiter = ModelCurveSpeedLimiter(SimpleNamespace())
    curve = make_model(curvature=np.full(ModelConstants.IDX_N, CURVATURE_BP[-1]))
    limiter.update(curve, 30.0, v_ego=10.0, lateral_active=True)

    self.assertFalse(limiter.torque_veto)

  def test_optional_torque_service_does_not_invalidate_planner_outputs(self):
    from openpilot.selfdrive.controls import plannerd

    sm = plannerd.get_submaster()
    for service in sm.services:
      sm.alive[service] = sm.freq_ok[service] = sm.valid[service] = True
    sm.alive['lateralTorqueParameters'] = False
    sm.freq_ok['lateralTorqueParameters'] = False
    sm.valid['lateralTorqueParameters'] = False

    self.assertTrue(sm.all_checks())

  def test_unhealthy_live_torque_params_fall_back_to_static(self):
    from openpilot.selfdrive.controls.lib import longitudinal_planner

    class FakeSubMaster(dict):
      pass

    live = SimpleNamespace(useParams=True)
    sm = FakeSubMaster(lateralTorqueParameters=live)
    sm.alive = {'lateralTorqueParameters': False}
    sm.freq_ok = {'lateralTorqueParameters': True}
    sm.valid = {'lateralTorqueParameters': True}

    self.assertIsNone(longitudinal_planner.get_live_torque_params(sm))
    sm.alive['lateralTorqueParameters'] = True
    self.assertIs(longitudinal_planner.get_live_torque_params(sm), live)

  def test_live_torque_params_and_positive_accel_clamp(self):
    from openpilot.selfdrive.controls.lib import longitudinal_planner

    live = SimpleNamespace(latAccelFactorFiltered=20.0, latAccelOffsetFiltered=0.0,
                           frictionCoefficientFiltered=0.1)
    limiter = ModelCurveSpeedLimiter(make_cp())
    curve = make_model(curvature=np.full(ModelConstants.IDX_N, CURVATURE_BP[-1]))
    limiter.update(curve, 30.0, v_ego=10.0, lateral_active=True, torque_params=live)
    limiter.update(curve, 30.0, v_ego=10.0, lateral_active=True, torque_params=live)

    self.assertFalse(limiter.torque_veto)
    self.assertEqual(longitudinal_planner.limit_accel_for_torque(0.4, True), 0.0)
    self.assertEqual(longitudinal_planner.limit_accel_for_torque(-0.4, True), -0.4)

  def test_approach_distance_follows_path_arc_not_only_forward_position(self):
    curvature = np.zeros(ModelConstants.IDX_N)
    curvature[20:25] = CURVATURE_BP[-1]
    x = np.minimum(np.arange(ModelConstants.IDX_N, dtype=float), 10.0)
    y = np.maximum(np.arange(ModelConstants.IDX_N, dtype=float) - 10.0, 0.0)
    limiter = ModelCurveSpeedLimiter()

    model = make_model(curvature=curvature, position_x=x, position_y=y)
    limiter.update(model, 30.0)
    limiter.update(model, 30.0)

    self.assertTrue(limiter.active)
    self.assertGreater(limiter.distance, np.ptp(x))

  def test_invalid_model_data_does_not_change_cruise(self):
    for invalid_value in ("short", "nan"):
      with self.subTest(invalid_value=invalid_value):
        model = make_model()
        if invalid_value == "short":
          model.orientationRate.z = model.orientationRate.z[:-1]
        else:
          model.position.x[5] = np.nan
        limiter = ModelCurveSpeedLimiter()

        self.assertEqual(limiter.update(model, 20.0), 20.0)
        self.assertFalse(limiter.active)

  def test_existing_cruise_controller_accelerates_only_below_curve_cap(self):
    from openpilot.selfdrive.controls.lib.longitudinal_planner import get_cruise_accel

    cap = 15.0
    common = {"e2e": False, "a_cruise_prev": 0.0, "dt": 0.05, "accel_coast": 0.0, "allow_throttle": True}

    self.assertGreater(get_cruise_accel(v_cruise=cap, v_ego=cap - 1.0, **common), 0.0)
    self.assertAlmostEqual(get_cruise_accel(v_cruise=cap, v_ego=cap, **common), 0.0)
    self.assertLess(get_cruise_accel(v_cruise=cap, v_ego=cap + 1.0, **common), 0.0)

  def test_curve_speed_units_match_field_mph_values(self):
    np.testing.assert_allclose(CURVE_SPEED_V / CV.MPH_TO_MS, [50.0, 22.0, 13.0])
    self.assertEqual(MAX_CURVE_SPEED, V_CRUISE_MAX * CV.KPH_TO_MS)

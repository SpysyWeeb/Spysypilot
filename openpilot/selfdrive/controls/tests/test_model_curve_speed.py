from types import SimpleNamespace
import math
import unittest

import numpy as np

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.model_curve_speed import (
  A_CURVE_MIN,
  A_LAT_COMFORT,
  AUTHORITY_BOUNDS,
  BRAKE_ENTER_S,
  COAST_ENTER_S,
  COAST_EXIT_S,
  J_DOWN,
  J_UP,
  REGIME_ANTICIPATE,
  REGIME_BRAKE,
  REGIME_COAST,
  REGIME_FREE,
  T_APPROACH,
  T_COAST,
  TORQUE_BUDGET,
  LateralState,
  ModelCurveSpeedLimiter,
  curve_speed_limits,
)
from openpilot.selfdrive.modeld.constants import ModelConstants

N = ModelConstants.IDX_N
T = np.array(ModelConstants.T_IDXS)
FACTOR, FRICTION = 2.7, 0.11
TORQUE = (FACTOR, 0.0, FRICTION)


def make_model(speed=15.0, curvature=None, position_y=None):
  x = speed * T
  curvature = np.zeros(N) if curvature is None else np.asarray(curvature, dtype=float)
  return SimpleNamespace(position=SimpleNamespace(x=x, y=np.zeros(N) if position_y is None else np.asarray(position_y)),
                         velocity=SimpleNamespace(x=np.full(N, speed)),
                         orientationRate=SimpleNamespace(z=curvature * speed))


def curve_ahead(speed, start, length, curvature):
  x = speed * T
  return make_model(speed, np.where((x >= start) & (x <= start + length), curvature, 0.0))


def make_cp(factor=FACTOR, friction=FRICTION, delay=0.5):
  torque = SimpleNamespace(latAccelFactor=factor, latAccelOffset=0.0, friction=friction)
  return SimpleNamespace(longitudinalActuatorDelay=delay, lateralTuning=SimpleNamespace(which=lambda: "torque", torque=torque))


def tracking(torque=0.3, error=0.05, lat=1.0, pinned=False, desired=None):
  return LateralState(True, torque, error, lat, lat + error if desired is None else desired, pinned)


def settle(limiter, model, frames, **kwargs):
  result = None
  for _ in range(frames):
    result = limiter.update(model, **kwargs)
  return result


class TestSpeedLimits(unittest.TestCase):
  def test_authority_is_the_torque_budget_at_that_node_and_comfort_the_owner_number(self):
    kappa = np.array([0.05])
    authority = math.sqrt((TORQUE_BUDGET - FRICTION) * FACTOR / 0.05)
    self.assertAlmostEqual(float(curve_speed_limits(kappa, TORQUE, 0.0)[0]), authority)
    self.assertAlmostEqual(float(curve_speed_limits(kappa, None, 0.0)[0]), math.sqrt(A_LAT_COMFORT / 0.05))
    self.assertLess(authority, math.sqrt(A_LAT_COMFORT / 0.05))            # the Palisade's authority binds before comfort

  def test_banking_helps_the_inside_and_a_straight_has_no_limit(self):
    kappa = np.array([0.05, -0.05, 0.0])
    limits = curve_speed_limits(kappa, TORQUE, roll=0.05)
    self.assertGreater(limits[0], limits[1])                                 # roll bias adds authority in one direction
    self.assertTrue(math.isinf(limits[2]))


class TestAnticipation(unittest.TestCase):
  def test_a_curve_ahead_asks_the_kinematic_deceleration_to_its_limit(self):
    limiter = ModelCurveSpeedLimiter(make_cp())
    model = curve_ahead(15.0, 60.0, 60.0, 0.05)
    result = settle(limiter, model, 60, v_ego=15.0, a_ego=-1.5, lateral_active=True, lateral_state=tracking())
    limit = math.sqrt((TORQUE_BUDGET - FRICTION) * FACTOR / 0.05)
    need = (limit ** 2 - 15.0 ** 2) / (2.0 * (result.distance - 15.0 * limiter.response_time))
    self.assertEqual(result.regime, REGIME_ANTICIPATE)
    self.assertAlmostEqual(result.v_limit, limit)
    self.assertAlmostEqual(result.a_target, need, places=5)

  def test_near_a_limit_the_candidate_is_a_small_proportional_approach_not_a_burst(self):
    limiter = ModelCurveSpeedLimiter(make_cp())
    model = make_model(6.0, np.full(N, 0.05))                                # already in the curve, 0.5 m/s under its limit
    result = settle(limiter, model, 20, v_ego=6.0, a_ego=0.0, lateral_active=True, lateral_state=tracking())
    limit = math.sqrt((TORQUE_BUDGET - FRICTION) * FACTOR / 0.05)
    self.assertAlmostEqual(result.a_target, (limit - 6.0) / T_APPROACH, places=5)
    self.assertLess(result.a_target, 0.5)

  def test_a_straight_path_has_nothing_to_say(self):
    limiter = ModelCurveSpeedLimiter(make_cp())
    result = settle(limiter, make_model(20.0), 5, v_ego=20.0, lateral_active=True, lateral_state=tracking())
    self.assertIsNone(result.a_target)
    self.assertEqual(result.regime, REGIME_FREE)

  def test_one_frame_spikes_are_rejected_and_invalid_models_ignored(self):
    limiter = ModelCurveSpeedLimiter(make_cp())
    straight, spike = make_model(20.0), make_model(20.0, np.full(N, 0.1))
    limiter.update(straight, v_ego=20.0)
    self.assertIsNone(limiter.update(spike, v_ego=20.0).a_target)
    self.assertIsNone(limiter.update(straight, v_ego=20.0).a_target)
    self.assertIsNone(limiter.update(SimpleNamespace(), v_ego=20.0).a_target)

  def test_the_candidate_is_jerk_limited_from_the_car_and_floored(self):
    limiter = ModelCurveSpeedLimiter(make_cp())
    model = make_model(20.0, np.full(N, 0.1))                                 # deep inside a tight curve at speed
    first = settle(limiter, model, 3, v_ego=20.0, a_ego=0.5, lateral_active=True, lateral_state=tracking())
    self.assertGreaterEqual(first.a_target, 0.5 - 2 * J_DOWN * DT_MDL - 1e-9)   # pulls down from the car's acceleration
    last = settle(limiter, model, 60, v_ego=20.0, a_ego=-2.0, lateral_active=True, lateral_state=tracking())
    self.assertAlmostEqual(last.a_target, A_CURVE_MIN)
    released = limiter.update(make_model(20.0), v_ego=20.0, a_ego=-2.0, lateral_active=True, lateral_state=tracking())
    self.assertLessEqual(released.a_target - last.a_target, J_UP * DT_MDL + 1e-9)


class TestReaction(unittest.TestCase):
  def test_heavy_but_tracking_coasts_and_releases_with_hysteresis(self):
    limiter = ModelCurveSpeedLimiter(make_cp())
    straight = make_model(15.0)
    result = settle(limiter, straight, round(COAST_ENTER_S / DT_MDL) - 1, v_ego=15.0, lateral_active=True, accel_coast=-0.3,
                    lateral_state=tracking(torque=T_COAST, error=0.1, lat=2.0))
    self.assertNotEqual(result.regime, REGIME_COAST)                        # heavy torque must dwell before the foot comes off
    result = settle(limiter, straight, 12, v_ego=15.0, lateral_active=True, accel_coast=-0.3, lateral_state=tracking(torque=T_COAST, error=0.1, lat=2.0))
    self.assertEqual(result.regime, REGIME_COAST)
    self.assertLessEqual(result.a_target, -0.3 + 1e-9)
    downhill = settle(limiter, straight, 3, v_ego=15.0, lateral_active=True, accel_coast=0.4, lateral_state=tracking(torque=T_COAST, error=0.1, lat=2.0))
    self.assertLessEqual(downhill.a_target, 0.0)                            # never a net acceleration while heavy, downhill included
    heavy_wide = settle(ModelCurveSpeedLimiter(make_cp()), straight, 12, v_ego=15.0, lateral_active=True, accel_coast=-0.3,
                        lateral_state=tracking(torque=0.94, error=0.33, lat=2.0))
    self.assertEqual(heavy_wide.regime, REGIME_COAST)                       # heavy and wide but not pinned: foot off, no brake
    result = settle(limiter, straight, 3, v_ego=15.0, lateral_active=True, accel_coast=-0.3, lateral_state=tracking(torque=0.8, error=0.1, lat=2.0))
    self.assertEqual(result.regime, REGIME_COAST)                            # still heavy: no release above T_COAST_EXIT
    self.assertAlmostEqual(result.a_target, -0.15)                           # and the lift is proportional to how heavy
    result = settle(limiter, straight, round(COAST_EXIT_S / DT_MDL) + 2, v_ego=15.0, lateral_active=True, lateral_state=tracking(torque=0.5))
    self.assertEqual(result.regime, REGIME_FREE)
    self.assertIsNone(result.a_target)

  def test_pinned_and_losing_brakes_toward_the_speed_that_restores_margin_after_a_debounce(self):
    limiter = ModelCurveSpeedLimiter(make_cp())
    straight = make_model(15.0)
    losing = tracking(torque=1.0, error=0.6, lat=2.3, pinned=True)
    result = settle(limiter, straight, round(BRAKE_ENTER_S / DT_MDL) - 1, v_ego=15.0, lateral_active=True, lateral_state=losing)
    self.assertNotEqual(result.regime, REGIME_BRAKE)                        # not on a single sample
    result = settle(limiter, straight, 40, v_ego=15.0, a_ego=-1.0, lateral_active=True, accel_coast=-0.3, lateral_state=losing)
    self.assertEqual(result.regime, REGIME_BRAKE)
    a_lat_ok = (TORQUE_BUDGET - FRICTION) * result.authority_factor
    v_ok = math.sqrt(a_lat_ok / (2.9 / 15.0 ** 2))                           # the demanded 2.9, not the achieved 2.3
    self.assertAlmostEqual(result.a_target, max((v_ok - 15.0) / 1.0, A_CURVE_MIN), places=5)
    overshoot = tracking(torque=1.0, error=-0.6, lat=2.3, pinned=True)     # actual above desired: an exit, not understeer
    fresh = ModelCurveSpeedLimiter(make_cp())
    self.assertNotEqual(settle(fresh, straight, 40, v_ego=15.0, lateral_active=True, lateral_state=overshoot).regime, REGIME_BRAKE)
    right = LateralState(True, 1.0, -0.6, -2.3, -2.9, True)                # a right-hand curve: understeer reads negative
    right_hand = settle(ModelCurveSpeedLimiter(make_cp()), straight, 40, v_ego=15.0, lateral_active=True, lateral_state=right)
    self.assertEqual(right_hand.regime, REGIME_BRAKE)
    back = settle(limiter, straight, 3, v_ego=15.0, lateral_active=True, lateral_state=tracking(torque=1.0, error=0.1, lat=2.3, pinned=True))
    self.assertEqual(back.regime, REGIME_COAST)                              # tracking again: hand back to coasting

  def test_no_reaction_without_active_lateral_or_with_the_driver_steering_or_at_crawl_speed(self):
    limiter = ModelCurveSpeedLimiter(make_cp())
    straight = make_model(15.0)
    losing = tracking(torque=1.0, error=0.6, lat=2.3, pinned=True)
    self.assertIsNone(settle(limiter, straight, 20, v_ego=15.0, lateral_active=False, lateral_state=losing).a_target)
    self.assertIsNone(settle(limiter, straight, 20, v_ego=15.0, lateral_active=True, steering_pressed=True, lateral_state=losing).a_target)
    slow = settle(limiter, make_model(2.0), 20, v_ego=2.0, lateral_active=True, lateral_state=losing)
    self.assertNotEqual(slow.regime, REGIME_BRAKE)

  def test_lateral_state_reads_both_controllers_and_fails_closed(self):
    torque = SimpleNamespace(active=True, output=-0.9, error=0.2, actualLateralAccel=-2.0, desiredLateralAccel=-2.2, saturated=False)
    cs = SimpleNamespace(lateralControlState=SimpleNamespace(which=lambda: 'torqueState', torqueState=torque))
    state = LateralState.from_controls_state(cs)
    self.assertTrue(state.active and math.isclose(state.torque, 0.9) and not state.pinned)
    self.assertAlmostEqual(state.understeer, -0.2)                            # desired -2.2, actual -2.0: exit overshoot, not understeer
    rack = SimpleNamespace(active=True, output=0.99, error=0.5, actualLateralAccel=2.3, desiredLateralAccel=2.8, saturated=False,
                           torqueLimited=True)
    cs = SimpleNamespace(lateralControlState=SimpleNamespace(which=lambda: 'rackState', rackState=rack))
    self.assertTrue(LateralState.from_controls_state(cs).pinned)
    self.assertFalse(LateralState.from_controls_state(SimpleNamespace(lateralControlState=SimpleNamespace(which=lambda: 'pidState'))).active)
    bad = SimpleNamespace(active=True, output=math.nan, error=0.0, actualLateralAccel=0.0, desiredLateralAccel=0.0, saturated=False)
    cs = SimpleNamespace(lateralControlState=SimpleNamespace(which=lambda: 'torqueState', torqueState=bad))
    self.assertFalse(LateralState.from_controls_state(cs).active)


class TestPlannerIntegration(unittest.TestCase):
  def test_the_candidate_only_ever_lowers_the_plan(self):
    from openpilot.selfdrive.controls.lib import longitudinal_planner
    from unittest.mock import patch
    model = make_model(15.0)
    model.meta = SimpleNamespace(disengagePredictions=SimpleNamespace(gasPressProbs=[1.0, 1.0]))
    model.action = SimpleNamespace(desiredAcceleration=1.0, shouldStop=False)
    absent = SimpleNamespace(present=False)
    messages = {
      'carState': SimpleNamespace(vEgo=15.0, vCruise=25.0 * 3.6, aEgo=0.0, standstill=False, steeringPressed=False),
      'carControl': SimpleNamespace(orientationNED=[], latActive=True),
      'controlsState': SimpleNamespace(forceDecel=False, longControlState=longitudinal_planner.LongCtrlState.pid,
                                       lateralControlState=SimpleNamespace(which=lambda: 'pidState')),
      'selfdriveState': SimpleNamespace(enabled=True, experimentalMode=False, personality=1),
      'vehicleParameters': SimpleNamespace(roll=0.0),
      'modelV2': model,
      'radarState': SimpleNamespace(leadOne=absent, leadTwo=absent),
      'lateralTorqueParameters': SimpleNamespace(useParams=False),
    }

    class FakeSubMaster(dict):
      def __init__(self, values):
        super().__init__(values)
        self.alive = self.freq_ok = self.valid = dict.fromkeys(values, True)

      def all_checks(self, services=None):
        return True

    planner = longitudinal_planner.LongitudinalPlanner(SimpleNamespace(openpilotLongitudinalControl=True, longitudinalActuatorDelay=0.2))
    sm = FakeSubMaster(messages)
    with (patch.object(planner.mpc, 'set_weights'), patch.object(planner.mpc, 'set_cur_state'), patch.object(planner.mpc, 'update'),
          patch.object(longitudinal_planner, 'get_accel_from_plan', return_value=0.5)):
      with patch.object(planner.curve_speed_limiter, 'update', return_value=longitudinal_planner.LateralState.__class__ and
                        __import__('openpilot.selfdrive.controls.lib.model_curve_speed', fromlist=['CurveResult']).CurveResult(2.5, 'anticipate')):
        planner.update(sm)
        self.assertLessEqual(planner.output_a_target, 0.5)                     # a higher candidate never raises the plan
      with patch.object(planner.curve_speed_limiter, 'update', return_value=
                        __import__('openpilot.selfdrive.controls.lib.model_curve_speed', fromlist=['CurveResult']).CurveResult(-1.2, 'coast')):
        planner.update(sm)
        self.assertAlmostEqual(planner.output_a_target, -1.2)                  # a lower one binds


if __name__ == "__main__":
  unittest.main()


class TestAuthorityCalibration(unittest.TestCase):
  def test_the_measured_ratio_moves_the_authority_inside_its_bounds(self):
    limiter = ModelCurveSpeedLimiter(make_cp())
    straight = make_model(20.0)
    self.assertAlmostEqual(limiter.update(straight, v_ego=20.0, lateral_active=True, lateral_state=tracking()).authority_factor, FACTOR)
    strong = tracking(torque=0.9, error=0.1, lat=3.1)                       # route 22: 3.1 m/s^2 at 0.9 torque
    result = settle(limiter, straight, 400, v_ego=20.0, lateral_active=True, lateral_state=strong)
    self.assertAlmostEqual(result.authority_factor, min(3.1 / (0.9 - FRICTION), AUTHORITY_BOUNDS[1] * FACTOR), places=1)
    weak = tracking(torque=0.9, error=0.1, lat=1.0)
    result = settle(limiter, straight, 1200, v_ego=20.0, lateral_active=True, lateral_state=weak)
    self.assertAlmostEqual(result.authority_factor, AUTHORITY_BOUNDS[0] * FACTOR, places=2)
    idle = settle(limiter, straight, 100, v_ego=20.0, lateral_active=True, lateral_state=tracking(torque=0.2, lat=0.1))
    self.assertAlmostEqual(idle.authority_factor, result.authority_factor)    # light steering teaches nothing

  def test_calibrated_authority_raises_the_speed_limit(self):
    limiter = ModelCurveSpeedLimiter(make_cp())
    model = curve_ahead(15.0, 60.0, 60.0, 0.05)
    before = settle(limiter, model, 60, v_ego=15.0, a_ego=-1.5, lateral_active=True, lateral_state=tracking()).v_limit
    settle(limiter, make_model(15.0), 400, v_ego=15.0, lateral_active=True, lateral_state=tracking(torque=0.9, error=0.1, lat=3.1))
    after = settle(limiter, model, 60, v_ego=15.0, a_ego=-1.5, lateral_active=True, lateral_state=tracking()).v_limit
    self.assertGreater(after, before)


class TestLegacySurface(unittest.TestCase):
  def test_the_planner_no_longer_carries_the_veto_or_the_cap(self):
    import inspect
    from openpilot.selfdrive.controls.lib import longitudinal_planner
    source = inspect.getsource(longitudinal_planner)
    for legacy in ('torque_veto', 'limit_accel_for_torque', 'speed_limiter_active', 'predicted_torque'):
      self.assertNotIn(legacy, source)

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
  REGIME_ANTICIPATE,
  REGIME_BRAKE,
  REGIME_COAST,
  REGIME_FREE,
  ROLL_HORIZON_S,
  AUTHORITY_MIN_LATERAL,
  AUTHORITY_MIN_SPEED,
  J_UP,
  A_CURVE_FREE,
  BEND_OPEN_A_LAT,
  J_DOWN,
  BEND_OPEN_S,
  HOLD_MAX_S,
  T_APPROACH,
  T_COAST,
  AUTHORITY_MIN_TORQUE,
  FAR_NODE_DECEL_MAX,
  FAR_LIMIT_UNCERTAINTY,
  V_HOLD_BAND,
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
  # the fields the policy reads, plus what a planner parses from a model message in the integration test
  return SimpleNamespace(position=SimpleNamespace(x=x, y=np.zeros(N) if position_y is None else np.asarray(position_y)),
                         velocity=SimpleNamespace(x=np.full(N, speed)), acceleration=SimpleNamespace(x=np.zeros(N)),
                         orientation=SimpleNamespace(z=np.zeros(N)), orientationRate=SimpleNamespace(z=curvature * speed), leadsV3=[])


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

  def test_at_the_limit_the_candidate_is_a_flat_zero_and_the_band_edges_correct(self):
    # owner ruling 2026-08-31: riding the limit means holding it, not stitching corrections across the zero crossing
    limit = math.sqrt((TORQUE_BUDGET - FRICTION) * FACTOR / 0.05)
    for v_ego, expect_zero in ((limit, True), (limit - V_HOLD_BAND + 0.05, True), (limit + V_HOLD_BAND - 0.05, True),
                               (limit - V_HOLD_BAND - 0.2, False), (limit + V_HOLD_BAND + 0.2, False)):
      limiter = ModelCurveSpeedLimiter(make_cp())
      model = make_model(v_ego, np.full(N, 0.05))
      result = settle(limiter, model, 30, v_ego=v_ego, a_ego=0.0, lateral_active=True, lateral_state=tracking())
      if expect_zero:
        self.assertEqual(result.a_target, 0.0, v_ego)
      else:
        self.assertNotEqual(result.a_target, 0.0, v_ego)
        self.assertAlmostEqual(result.a_target, (limit - v_ego) / T_APPROACH, places=5)

  def test_comfort_sits_above_the_owners_manual_envelope(self):
    # 3.4: deliberately above the manual archive's max lateral (2.86), so the calibrated authority binds and comfort
    # is only the backstop against an implausible learned authority
    self.assertGreater(A_LAT_COMFORT, 3.0)
    kappa = np.full(N, 0.05)
    self.assertLess(float(curve_speed_limits(kappa, TORQUE, 0.0)[0]), math.sqrt(A_LAT_COMFORT / 0.05))

  def test_a_gas_override_earns_a_grace_where_anticipation_never_brakes(self):
    # route 0x2c t=885: released the pedal 1.4 m/s above the in-curve limit and the episode pulled it back to -1.0
    limiter = ModelCurveSpeedLimiter(make_cp())
    model = make_model(9.0, np.full(N, 0.05))                                # ~2.5 m/s over the curve's limit
    limiter.update(model, v_ego=9.0, lateral_active=True, lateral_state=tracking(), gas_pressed=True)
    result = settle(limiter, model, 20, v_ego=9.0, a_ego=0.0, lateral_active=True, lateral_state=tracking())
    self.assertEqual(result.a_target, 0.0)                                   # holds, does not brake
    for _ in range(int(5.0 / DT_MDL)):                                       # the grace expires ...
      result = limiter.update(model, v_ego=9.0, a_ego=0.0, lateral_active=True, lateral_state=tracking())
    self.assertLess(result.a_target, -0.5)                                   # ... and the limit binds again

  def test_the_brake_regime_still_runs_inside_the_grace(self):
    limiter = ModelCurveSpeedLimiter(make_cp())
    model = make_model(9.0, np.full(N, 0.05))
    limiter.update(model, v_ego=9.0, lateral_active=True, lateral_state=tracking(), gas_pressed=True)
    pinned = tracking(torque=0.97, error=0.5, lat=3.0, pinned=True)
    for _ in range(int(BRAKE_ENTER_S / DT_MDL) + 2):
      result = limiter.update(model, v_ego=9.0, a_ego=0.0, lateral_active=True, lateral_state=pinned)
    self.assertEqual(limiter.regime, REGIME_BRAKE)
    self.assertLess(result.a_target, 0.0)

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
    self.assertNotEqual(result.regime, REGIME_COAST)                        # the lift has ended ...
    self.assertTrue(result.holding)                                           # ... into the hold: the road still reads 1.0 m/s^2
    result = settle(limiter, straight, round(BEND_OPEN_S / DT_MDL) + 2, v_ego=15.0, lateral_active=True, lateral_state=tracking(torque=0.5, lat=0.5))
    self.assertEqual(result.regime, REGIME_FREE)                              # the road open for the dwell: released ...
    self.assertGreater(result.a_target, 0.0)                                  # ... into a ramp, not a step
    result = settle(limiter, straight, round(A_CURVE_FREE / J_UP / DT_MDL) + 2, v_ego=15.0, lateral_active=True, lateral_state=tracking(torque=0.5, lat=0.5))
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
    helped = settle(ModelCurveSpeedLimiter(make_cp()), straight, 40, v_ego=15.0, a_ego=-1.0, lateral_active=True, accel_coast=-0.3,
                    roll=0.05, lateral_state=losing)                        # a bank in the turn's favour (the same sign as its lateral) ...
    self.assertGreater(helped.a_target, result.a_target)                     # ... asks for less than the flat ...
    adverse = settle(ModelCurveSpeedLimiter(make_cp()), straight, 40, v_ego=15.0, a_ego=-1.0, lateral_active=True, accel_coast=-0.3,
                     roll=-0.05, lateral_state=losing)                      # ... and a crown against it never for less
    self.assertLessEqual(adverse.a_target, result.a_target + 1e-9)
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
    model.action = SimpleNamespace(desiredAcceleration=1.0, shouldStop=False, desiredCurvature=0.0)
    absent = SimpleNamespace(present=False)
    messages = {
      'carState': SimpleNamespace(vEgo=15.0, vCruise=25.0 * 3.6, aEgo=0.0, standstill=False, steeringPressed=False, leftBlinker=False,
                                  rightBlinker=False, steeringAngleDeg=0.0, gasPressed=False, brakePressed=False),
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
        # the model message here is a stub: planners that classify stops or anchor leads from it must see it invalid
        return services is not None and 'modelV2' not in services

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


class TestRollHorizon(unittest.TestCase):
  def test_a_far_node_gets_no_bank_a_near_one_the_live_roll(self):
    kappa = np.full(N, 0.01)
    x = 20.0 * T                                                              # nodes from 0 to 200 m at 20 m/s
    near = x <= 20.0 * ROLL_HORIZON_S
    adverse = curve_speed_limits(kappa, TORQUE, np.where(near, -0.05, 0.0))   # what _anticipate builds: live roll near, zero far
    flat = curve_speed_limits(kappa, TORQUE, 0.0)
    np.testing.assert_allclose(adverse[~near], flat[~near])                   # beyond the horizon the crown is not charged
    self.assertLess(adverse[near][0], flat[near][0])                          # within it, a crown against the turn still lowers the limit

  def test_the_anticipation_ignores_the_crown_for_a_curve_far_ahead(self):
    model = curve_ahead(20.0, 120.0, 40.0, 0.006)                            # a bend 120 m out at 20 m/s (6 s of travel)
    crowned = ModelCurveSpeedLimiter(make_cp())
    level = ModelCurveSpeedLimiter(make_cp())
    a = settle(crowned, model, 3, v_ego=20.0, lateral_active=True, roll=0.05, lateral_state=tracking())
    b = settle(level, model, 3, v_ego=20.0, lateral_active=True, roll=0.0, lateral_state=tracking())
    self.assertAlmostEqual(a.v_limit, b.v_limit, places=6)

  def test_a_far_bend_a_bank_could_explain_asks_for_the_foot_off_at_most(self):
    # a sweeper 4 s out at 30 m/s whose limit sits 15 % under the car: far, and within the doubt a bank leaves
    v = 30.0
    kappa = (TORQUE_BUDGET - FRICTION) * FACTOR / (v * (1.0 - FAR_LIMIT_UNCERTAINTY + 0.10)) ** 2
    far = curve_ahead(v, 120.0, 60.0, kappa)
    result = settle(ModelCurveSpeedLimiter(make_cp()), far, 3, v_ego=v, lateral_active=True, lateral_state=tracking())
    self.assertLess(result.v_limit, v)
    self.assertGreaterEqual(result.a_target, -FAR_NODE_DECEL_MAX - 1e-9)
    # the same bend 1 s out: the bank is the car's own and the brake is allowed
    near = curve_ahead(v, 30.0, 60.0, kappa)
    result = settle(ModelCurveSpeedLimiter(make_cp()), near, 20, v_ego=v, lateral_active=True, lateral_state=tracking())   # past the jerk ramp
    self.assertLess(result.a_target, -FAR_NODE_DECEL_MAX)
    # a genuinely tight bend far out (its limit well under the doubt) still asks for its full deceleration
    tight = curve_ahead(v, 120.0, 60.0, 0.02)
    result = settle(ModelCurveSpeedLimiter(make_cp()), tight, 20, v_ego=v, lateral_active=True, lateral_state=tracking())
    self.assertLess(result.a_target, -FAR_NODE_DECEL_MAX)

  def test_the_anticipation_charges_the_crown_for_a_bend_within_the_horizon(self):
    model = curve_ahead(20.0, 10.0, 20.0, 0.01)                              # a bend 10 m out: inside 2 s of travel
    crowned = ModelCurveSpeedLimiter(make_cp())
    level = ModelCurveSpeedLimiter(make_cp())
    a = settle(crowned, model, 3, v_ego=20.0, lateral_active=True, roll=-0.05, lateral_state=tracking())
    b = settle(level, model, 3, v_ego=20.0, lateral_active=True, roll=0.0, lateral_state=tracking())
    self.assertLess(a.v_limit, b.v_limit)


class TestAuthorityCalibration(unittest.TestCase):
  def test_town_corners_and_light_corners_teach_nothing(self):
    limiter = ModelCurveSpeedLimiter(make_cp())
    straight = make_model(8.0)
    slow = settle(limiter, straight, 400, v_ego=AUTHORITY_MIN_SPEED - 1.0, lateral_active=True, lateral_state=tracking(torque=0.9, error=0.1, lat=1.5))
    self.assertAlmostEqual(slow.authority_factor, FACTOR)                     # a tight town corner: the assist is not the highway's
    light = settle(limiter, make_model(20.0), 400, v_ego=20.0, lateral_active=True,
                   lateral_state=tracking(torque=0.9, error=0.1, lat=AUTHORITY_MIN_LATERAL - 0.1))
    self.assertAlmostEqual(light.authority_factor, FACTOR)                    # heavy torque for little lateral: friction, not authority
    banked = settle(limiter, make_model(20.0), 400, v_ego=20.0, lateral_active=True, roll=0.07,
                    lateral_state=tracking(torque=0.4, error=0.1, lat=1.0))
    self.assertAlmostEqual(banked.authority_factor, FACTOR)                   # a bank's 0.7 m/s^2 of the 1.0 was not the steering's

  def test_the_bank_is_not_the_steerings_doing(self):
    banked = ModelCurveSpeedLimiter(make_cp())
    flat = ModelCurveSpeedLimiter(make_cp())
    straight = make_model(20.0)
    state = tracking(torque=0.9, error=0.1, lat=-2.9)                        # a left sweeper: 2.9 of ground-plane lateral at 0.9 torque
    b = settle(banked, straight, 400, v_ego=20.0, lateral_active=True, roll=-0.05, lateral_state=state)   # banked 0.05 rad in its favour
    f = settle(flat, straight, 400, v_ego=20.0, lateral_active=True, roll=0.0, lateral_state=state)
    self.assertLess(b.authority_factor, f.authority_factor)                   # the bank's 0.49 m/s^2 is not credited to the steering
    self.assertAlmostEqual(b.authority_factor, (2.9 - 0.05 * 9.81) / (0.9 - FRICTION), places=1)

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
    self.assertAlmostEqual(idle.authority_factor, result.authority_factor)    # light steering teaches nothing ...
    moderate = settle(limiter, straight, 100, v_ego=20.0, lateral_active=True, lateral_state=tracking(torque=AUTHORITY_MIN_TORQUE - 0.1, error=0.1, lat=3.0))
    self.assertAlmostEqual(moderate.authority_factor, result.authority_factor)  # ... and so does moderate torque: the ratio there is not the budget's

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


class TestHold(unittest.TestCase):
  # a curve whose limit sits just above the car: the anticipation has something to say (a small positive candidate) and the
  # steering, fed heavy, says the limit is here
  CURVATURE = 0.0074

  def _lifted(self):
    limiter = ModelCurveSpeedLimiter(make_cp())
    bend = make_model(15.0, np.full(N, self.CURVATURE))
    settle(limiter, bend, round(COAST_ENTER_S / DT_MDL) + 12, v_ego=15.0, lateral_active=True, accel_coast=-0.3,
           lateral_state=tracking(torque=T_COAST, error=0.1, lat=2.0))
    self.assertEqual(limiter.regime, REGIME_COAST)
    result = settle(limiter, bend, round(COAST_EXIT_S / DT_MDL) + 8, v_ego=15.0, lateral_active=True, accel_coast=-0.3,
                    lateral_state=self.held())
    self.assertNotEqual(result.regime, REGIME_COAST)
    return limiter, bend, result

  @staticmethod
  def held():
    # light torque, the bend still on; the tracking error keeps the authority calibration out of these tests
    return tracking(torque=0.5, error=0.35, lat=2.0)

  def test_the_lift_ends_into_a_hold_while_the_bend_is_still_on(self):
    limiter, bend, result = self._lifted()
    self.assertTrue(result.holding)
    self.assertLessEqual(result.a_target, 1e-9)                              # the candidate is zero, not the anticipation's +1.3
    result = settle(limiter, bend, 60, v_ego=15.0, lateral_active=True, accel_coast=-0.3, lateral_state=self.held())
    self.assertTrue(result.holding)
    self.assertLessEqual(result.a_target, 1e-9)                              # and stays zero for as long as the bend reads on

  def test_the_bend_reading_open_for_a_second_releases_the_hold(self):
    limiter, bend, _ = self._lifted()
    result = settle(limiter, bend, round(BEND_OPEN_S / DT_MDL) - 2, v_ego=15.0, lateral_active=True, lateral_state=tracking(torque=0.3, lat=0.5))
    self.assertTrue(result.holding)                                           # not before the dwell
    result = settle(limiter, bend, 4, v_ego=15.0, lateral_active=True, lateral_state=tracking(torque=0.3, lat=0.5))
    self.assertFalse(result.holding)
    self.assertGreater(result.a_target, 0.0)                                  # the anticipation's own candidate is back

  def test_the_hold_times_out_on_a_bend_that_never_reads_open(self):
    limiter, bend, _ = self._lifted()
    result = settle(limiter, bend, round(HOLD_MAX_S / DT_MDL) + 2, v_ego=15.0, lateral_active=True, lateral_state=self.held())
    self.assertFalse(result.holding)
    self.assertGreater(result.a_target, 0.0)

  def test_the_path_seeing_the_exit_does_not_end_the_hold_the_road_opening_does(self):
    limiter, _, _ = self._lifted()
    straight = make_model(15.0)
    result = settle(limiter, straight, 3, v_ego=15.0, lateral_active=True, lateral_state=self.held())
    self.assertTrue(result.holding)                                           # the model sees the exit; the car is still in the bend
    self.assertLessEqual(result.a_target, 1e-9)
    result = settle(limiter, straight, round(BEND_OPEN_S / DT_MDL) + 2, v_ego=15.0, lateral_active=True, lateral_state=tracking(torque=0.3, lat=0.3))
    self.assertFalse(result.holding)                                          # the road reads open: released within the dwell
    self.assertGreater(result.a_target, 0.0)                                  # the candidate ramps out from the hold's zero ...
    result = settle(limiter, straight, round(A_CURVE_FREE / J_UP / DT_MDL) + 2, v_ego=15.0, lateral_active=True, lateral_state=tracking(torque=0.3, lat=0.3))
    self.assertIsNone(result.a_target)                                        # ... and a straight road then has nothing to say

  def test_the_drivers_gas_is_never_held(self):
    limiter, bend, _ = self._lifted()
    result = limiter.update(bend, v_ego=15.0, lateral_active=True, lateral_state=self.held(), gas_pressed=True)
    self.assertGreater(result.a_target, 0.0)

  def test_a_steering_dropout_keeps_the_hold_for_the_resumption(self):
    limiter, bend, _ = self._lifted()
    # the idle controller reports zero lateral acceleration; that must not read as the bend opening
    result = settle(limiter, bend, round(2.0 / DT_MDL), v_ego=15.0, lateral_active=False, lateral_state=LateralState())
    self.assertTrue(result.holding)
    self.assertIsNone(result.a_target)                                        # nothing is held while the driver steers
    result = settle(limiter, bend, 8, v_ego=15.0, lateral_active=True, lateral_state=self.held())
    self.assertTrue(result.holding)
    self.assertLessEqual(result.a_target, 1e-9)                              # held again the moment we steer

  def test_a_lift_that_ends_in_a_steering_dropout_arms_the_hold(self):
    limiter = ModelCurveSpeedLimiter(make_cp())
    bend = make_model(15.0, np.full(N, self.CURVATURE))
    settle(limiter, bend, round(COAST_ENTER_S / DT_MDL) + 12, v_ego=15.0, lateral_active=True, accel_coast=-0.3,
           lateral_state=tracking(torque=T_COAST, error=0.1, lat=2.0))
    result = limiter.update(bend, v_ego=15.0, lateral_active=True, steering_pressed=True, lateral_state=tracking(torque=T_COAST, lat=2.0))
    self.assertNotEqual(result.regime, REGIME_COAST)
    self.assertTrue(result.holding)

  def test_a_disengagement_clears_the_hold(self):
    limiter, bend, _ = self._lifted()
    limiter.reset()
    result = settle(limiter, bend, 3, v_ego=15.0, lateral_active=True, lateral_state=self.held())
    self.assertFalse(result.holding)
    self.assertEqual(result.regime, REGIME_ANTICIPATE)
    self.assertGreater(result.a_target, 0.0)

  def test_a_lift_inside_the_hold_does_not_restart_its_clock(self):
    limiter, bend, _ = self._lifted()
    settle(limiter, bend, round(HOLD_MAX_S / DT_MDL) - 40, v_ego=15.0, lateral_active=True, lateral_state=self.held())
    settle(limiter, bend, round(COAST_ENTER_S / DT_MDL) + 4, v_ego=15.0, lateral_active=True, accel_coast=-0.3,
           lateral_state=tracking(torque=T_COAST, error=0.35, lat=2.0))                 # heavy again: a second lift ...
    result = settle(limiter, bend, round(COAST_EXIT_S / DT_MDL) + 40, v_ego=15.0, lateral_active=True, lateral_state=self.held())
    self.assertFalse(result.holding)                                          # ... and the backstop still ends the hold on time
    self.assertGreater(result.a_target, 0.0)

  def test_the_release_is_a_ramp_at_the_up_jerk(self):
    limiter, bend, _ = self._lifted()
    settle(limiter, bend, round(BEND_OPEN_S / DT_MDL) + 1, v_ego=15.0, lateral_active=True, lateral_state=tracking(torque=0.3, lat=0.5))
    straight = make_model(15.0)
    first = limiter.update(straight, v_ego=15.0, lateral_active=True, lateral_state=tracking(torque=0.3, lat=0.5)).a_target
    second = limiter.update(straight, v_ego=15.0, lateral_active=True, lateral_state=tracking(torque=0.3, lat=0.5)).a_target
    self.assertIsNotNone(first)
    self.assertAlmostEqual(second - first, J_UP * DT_MDL, places=6)

  def test_a_bend_that_pinches_again_during_the_release_pulls_from_the_car(self):
    limiter, bend, _ = self._lifted()
    settle(limiter, bend, round(BEND_OPEN_S / DT_MDL) + 1, v_ego=15.0, lateral_active=True, lateral_state=tracking(torque=0.3, lat=0.5))
    straight = make_model(15.0)
    settle(limiter, straight, 10, v_ego=15.0, a_ego=0.3, lateral_active=True, lateral_state=tracking(torque=0.3, lat=0.5))   # 0.5 s of ramp-out
    tight = make_model(15.0, np.full(N, 0.03))                                # a bend the limit puts well below 15 m/s
    first = settle(limiter, tight, 2, v_ego=15.0, a_ego=0.3, lateral_active=True, lateral_state=tracking(torque=0.3, lat=0.5))   # the 3-sample median needs two
    self.assertLessEqual(first.a_target, 0.3 + J_DOWN * DT_MDL + 1e-9)       # anchored on the car, not on the ramp
    later = settle(limiter, tight, round(0.7 / DT_MDL), v_ego=15.0, a_ego=-0.5, lateral_active=True, lateral_state=tracking(torque=0.3, lat=0.5))
    self.assertLessEqual(later.a_target, -1.0)                                # and braking within the second

  def test_the_drivers_wheel_ends_the_release_ramp_at_once(self):
    limiter, bend, _ = self._lifted()
    settle(limiter, bend, round(BEND_OPEN_S / DT_MDL) + 1, v_ego=15.0, lateral_active=True, lateral_state=tracking(torque=0.3, lat=0.5))
    straight = make_model(15.0)
    self.assertIsNotNone(limiter.update(straight, v_ego=15.0, lateral_active=True, lateral_state=tracking(torque=0.3, lat=0.5)).a_target)
    self.assertIsNone(limiter.update(straight, v_ego=15.0, lateral_active=True, steering_pressed=True, lateral_state=tracking(torque=0.3, lat=0.5)).a_target)

  def test_the_hold_constants_are_ordered(self):
    self.assertGreater(BEND_OPEN_A_LAT, 0.0)
    self.assertLess(0.0, BEND_OPEN_S)
    self.assertLess(BEND_OPEN_S, HOLD_MAX_S)

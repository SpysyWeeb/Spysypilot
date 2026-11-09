from types import SimpleNamespace
import math
import unittest

import numpy as np

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.model_curve_speed import (
  A_CURVE_MIN,
  A_FLOOR_LAT,
  A_LAT_COMFORT,
  AUTHORITY_BOUNDS,
  BANK_CONTINUATION_END,
  BANK_CONTINUATION_MIN_LATERAL,
  BANK_CONTINUATION_TOLERANCE,
  AUTHORITY_HELD_LATERAL,
  AUTHORITY_HELD_SPEED,
  authority_margin,
  held_lateral,
  BRAKE_ENTER_S,
  BRAKE_EXIT_S,
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
  T_RESTORE,
  V_CURVE_FLOOR,
  V_REACT_MIN,
  AUTHORITY_MIN_TORQUE,
  FAR_NODE_DECEL_MAX,
  FAR_LIMIT_UNCERTAINTY,
  V_HOLD_BAND,
  TORQUE_BUDGET,
  LateralState,
  ModelCurveSpeedLimiter,
  bank_continuation,
  curve_floor,
  curve_speed_limits,
)
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.selfdrive.modeld.constants import ModelConstants

N = ModelConstants.IDX_N
T = np.array(ModelConstants.T_IDXS)
FACTOR, FRICTION = 2.7, 0.11
TORQUE = (FACTOR, 0.0, FRICTION)
MODEL_MARGIN = (TORQUE_BUDGET - FRICTION) * FACTOR    # what the torque model prices the budget at, 2.13 m/s^2


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


def make_cp(factor=FACTOR, friction=FRICTION, delay=0.5, fingerprint="HYUNDAI_PALISADE"):
  torque = SimpleNamespace(latAccelFactor=factor, latAccelOffset=0.0, friction=friction)
  return SimpleNamespace(carFingerprint=fingerprint, longitudinalActuatorDelay=delay,
                         lateralTuning=SimpleNamespace(which=lambda: "torque", torque=torque))


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


class TestDemonstratedHold(unittest.TestCase):
  # route 0x5b, the owner's sweeper: the car held 2.28 m/s^2 of lateral at 0.70 rack torque, above the 2.13 the torque model
  # says the FULL 0.90 budget could buy. The model extrapolates a slope the car saturates out of, so what the rack has been
  # measured holding inside the budget is a floor under the price, not a ceiling on it

  def test_the_floor_only_ever_raises_the_torque_model(self):
    model_margin = (TORQUE_BUDGET - FRICTION) * FACTOR
    self.assertAlmostEqual(authority_margin(FACTOR, FRICTION, 0.0), model_margin)
    self.assertAlmostEqual(authority_margin(FACTOR, FRICTION, model_margin - 1.0), model_margin)   # a lower hold never lowers it
    self.assertAlmostEqual(authority_margin(FACTOR, FRICTION, model_margin + 1.0), model_margin + 1.0)

  def test_the_floor_is_a_highway_statement_and_town_keeps_the_torque_model(self):
    # every one of route 0x5b's at-budget frames was under 20 m/s: in town the rack genuinely pins and the term is honest
    self.assertEqual(held_lateral(0.0), 0.0)
    self.assertEqual(held_lateral(AUTHORITY_HELD_SPEED[0]), 0.0)
    self.assertAlmostEqual(held_lateral(AUTHORITY_HELD_SPEED[1]), AUTHORITY_HELD_LATERAL)
    self.assertAlmostEqual(held_lateral(35.0), AUTHORITY_HELD_LATERAL)
    mid = held_lateral(sum(AUTHORITY_HELD_SPEED) / 2.0)                       # blended, so the cap has no step at the edge
    self.assertGreater(mid, 0.0)
    self.assertLess(mid, AUTHORITY_HELD_LATERAL)

  def test_a_highway_bend_is_priced_higher_than_the_same_bend_in_town(self):
    kappa = np.array([0.005])
    town = float(curve_speed_limits(kappa, TORQUE, 0.0, True, held_lateral(10.0))[0])
    highway = float(curve_speed_limits(kappa, TORQUE, 0.0, True, held_lateral(25.0))[0])
    self.assertAlmostEqual(town, math.sqrt((TORQUE_BUDGET - FRICTION) * FACTOR / 0.005))
    self.assertAlmostEqual(highway, math.sqrt(AUTHORITY_HELD_LATERAL / 0.005))
    self.assertGreater(highway, town)
    # the owner's sweeper, priced at his own ask: about 5 mph more through it
    self.assertGreater(highway - town, 1.5)

  def test_comfort_is_still_the_ceiling_over_the_floor(self):
    kappa = np.array([0.05])
    self.assertLess(AUTHORITY_HELD_LATERAL, A_LAT_COMFORT)
    limit = float(curve_speed_limits(kappa, TORQUE, 0.0, True, held_lateral(30.0))[0])
    self.assertAlmostEqual(limit, math.sqrt(AUTHORITY_HELD_LATERAL / 0.05))
    # and a learned authority above comfort still hands the ceiling back to comfort
    strong = (AUTHORITY_BOUNDS[1] * FACTOR, 0.0, FRICTION)
    self.assertAlmostEqual(float(curve_speed_limits(kappa, strong, 0.0, True, held_lateral(30.0))[0]),
                           math.sqrt(A_LAT_COMFORT / 0.05))

  def test_the_floor_belongs_to_the_car_it_was_measured_on(self):
    # the hold is a measurement of the Palisade's rack; another car's saturation is its own, so it keeps the torque model
    bend = make_model(25.0, np.full(N, 0.005))
    palisade = settle(ModelCurveSpeedLimiter(make_cp()), bend, 5, v_ego=25.0, lateral_active=True, lateral_state=tracking())
    other = settle(ModelCurveSpeedLimiter(make_cp(fingerprint="HONDA_CIVIC")), bend, 5, v_ego=25.0, lateral_active=True,
                   lateral_state=tracking())
    self.assertAlmostEqual(palisade.v_limit, math.sqrt(AUTHORITY_HELD_LATERAL / 0.005), places=3)
    self.assertAlmostEqual(other.v_limit, math.sqrt((TORQUE_BUDGET - FRICTION) * FACTOR / 0.005), places=3)

  def test_the_floor_does_not_touch_the_measured_authority_factor(self):
    # the reported factor stays a measurement of the torque ratio; the floor is a statement about the budget, not the ratio
    limiter = ModelCurveSpeedLimiter(make_cp())
    result = settle(limiter, make_model(25.0), 20, v_ego=25.0, lateral_active=True, lateral_state=tracking(torque=0.3, lat=0.5))
    self.assertAlmostEqual(result.authority_factor, FACTOR)


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
    losing = tracking(torque=1.0, error=0.35, lat=2.3, pinned=True)          # a loss mild enough that no request below meets the floor
    result = settle(limiter, straight, round(BRAKE_ENTER_S / DT_MDL) - 1, v_ego=15.0, lateral_active=True, lateral_state=losing)
    self.assertNotEqual(result.regime, REGIME_BRAKE)                        # not on a single sample
    result = settle(limiter, straight, 40, v_ego=15.0, a_ego=-1.0, lateral_active=True, accel_coast=-0.3, lateral_state=losing)
    self.assertEqual(result.regime, REGIME_BRAKE)
    measured = 2.3 / (1.0 - FRICTION)                                        # the authority the pinned steering shows now
    a_lat_ok = (TORQUE_BUDGET - FRICTION) * min(result.authority_factor, measured)
    v_ok = math.sqrt(a_lat_ok / (2.65 / 15.0 ** 2))                          # the demanded 2.65, not the achieved 2.3
    self.assertAlmostEqual(result.a_target, (v_ok - 15.0) / 1.0, places=5)
    self.assertGreater(result.a_target, A_CURVE_MIN)
    helped = settle(ModelCurveSpeedLimiter(make_cp()), straight, 40, v_ego=15.0, a_ego=-1.0, lateral_active=True, accel_coast=-0.3,
                    roll=0.05, lateral_state=losing)                        # a bank in the turn's favor (the same sign as its lateral) ...
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
    recovering = tracking(torque=1.0, error=0.1, lat=2.3, pinned=True)
    back = settle(limiter, straight, round(BRAKE_EXIT_S / DT_MDL) - 1, v_ego=15.0, lateral_active=True, lateral_state=recovering)
    self.assertEqual(back.regime, REGIME_BRAKE)                              # recovery must persist ...
    back = settle(limiter, straight, 2, v_ego=15.0, lateral_active=True, lateral_state=recovering)
    self.assertEqual(back.regime, REGIME_COAST)                              # ... then tracking again hands back to coasting

  def test_pinned_and_losing_brakes_below_a_hold_the_steering_is_not_delivering(self):
    # 25 m/s, pinned at full torque, 2.0 achieved of 2.4 asked: the demonstrated hold alone would call the bend holdable at
    # this speed and leave the brake regime only the coast; the steering's own measurement says it is not
    limiter = ModelCurveSpeedLimiter(make_cp())
    bend = make_model(25.0, np.full(N, 2.4 / 25.0 ** 2))
    losing = tracking(torque=1.0, error=0.4, lat=2.0, pinned=True)
    result = settle(limiter, bend, 60, v_ego=25.0, a_ego=-2.0, lateral_active=True, accel_coast=-0.3, lateral_state=losing)
    self.assertEqual(result.regime, REGIME_BRAKE)
    self.assertGreater(held_lateral(25.0), 2.4)
    v_ok = math.sqrt((TORQUE_BUDGET - FRICTION) * 2.0 / (1.0 - FRICTION) / (2.4 / 25.0 ** 2))
    self.assertLess(v_ok, 25.0)
    self.assertAlmostEqual(result.a_target, max(v_ok - 25.0, A_CURVE_MIN), places=5)

  def test_a_flag_pinned_below_the_torque_limit_prices_the_restore_from_the_budget(self):
    # saturated can mean a rate-limited curvature request: 0.11 torque and 0.02 of 1.5 m/s^2 is not the rack's authority,
    # and must not turn the priced coast into the braking floor
    limiter = ModelCurveSpeedLimiter(make_cp())
    straight = make_model(15.0)
    settle(limiter, straight, 40, v_ego=15.0, a_ego=-2.0, lateral_active=True, lateral_state=tracking(torque=1.0, error=0.6, lat=2.3, pinned=True))
    limited = LateralState(True, 0.11, 1.48, 0.02, 1.5, True)
    result = settle(limiter, straight, 40, v_ego=15.0, a_ego=-0.3, lateral_active=True, accel_coast=-0.3, lateral_state=limited)
    self.assertEqual(result.regime, REGIME_BRAKE)
    v_ok = math.sqrt((TORQUE_BUDGET - FRICTION) * FACTOR / (1.5 / 15.0 ** 2))
    self.assertGreater(v_ok, 15.0)                                           # the budget holds this bend at this speed ...
    self.assertAlmostEqual(result.a_target, -0.3, places=5)                   # ... so the regime asks for the coast, not a brake

  def test_one_recovered_sample_does_not_release_the_brake(self):
    limiter = ModelCurveSpeedLimiter(make_cp())
    straight = make_model(15.0)
    losing = tracking(torque=1.0, error=0.6, lat=2.3, pinned=True)
    braking = settle(limiter, straight, 40, v_ego=15.0, a_ego=-2.0, lateral_active=True, lateral_state=losing)
    self.assertEqual(braking.regime, REGIME_BRAKE)
    blip = limiter.update(straight, v_ego=15.0, a_ego=-2.0, lateral_active=True, lateral_state=tracking(torque=1.0, error=0.19, lat=2.3, pinned=True))
    self.assertEqual(blip.regime, REGIME_BRAKE)
    after = settle(limiter, straight, round(BRAKE_EXIT_S / DT_MDL) // 2, v_ego=15.0, a_ego=-2.0, lateral_active=True, lateral_state=losing)
    self.assertEqual(after.regime, REGIME_BRAKE)
    self.assertAlmostEqual(after.a_target, braking.a_target)

  def test_the_brake_releases_on_sustained_recovery_by_tracking_or_by_coming_off_the_pin(self):
    self.assertGreaterEqual(BRAKE_EXIT_S, BRAKE_ENTER_S)
    straight = make_model(15.0)
    losing = tracking(torque=1.0, error=0.6, lat=2.3, pinned=True)
    for recovered in (tracking(torque=1.0, error=0.1, lat=2.3, pinned=True),     # tracking again at the pin
                      tracking(torque=0.8, error=0.6, lat=2.3, pinned=False)):   # off the pin, still wide
      limiter = ModelCurveSpeedLimiter(make_cp())
      settle(limiter, straight, 40, v_ego=15.0, lateral_active=True, lateral_state=losing)
      held = settle(limiter, straight, round(BRAKE_EXIT_S / DT_MDL) - 1, v_ego=15.0, lateral_active=True, lateral_state=recovered)
      self.assertEqual(held.regime, REGIME_BRAKE)
      self.assertEqual(limiter.update(straight, v_ego=15.0, lateral_active=True, lateral_state=recovered).regime, REGIME_COAST)

  def test_the_drivers_wheel_a_steering_dropout_and_a_disengagement_end_the_brake_at_once(self):
    # the measurement that justified braking is no longer the steering's own: the dwell never delays these exits
    straight = make_model(15.0)
    losing = tracking(torque=1.0, error=0.6, lat=2.3, pinned=True)
    for interrupt in ({'lateral_active': True, 'steering_pressed': True}, {'lateral_active': False}):
      limiter = ModelCurveSpeedLimiter(make_cp())
      self.assertEqual(settle(limiter, straight, 40, v_ego=15.0, lateral_active=True, lateral_state=losing).regime, REGIME_BRAKE)
      self.assertEqual(limiter.update(straight, v_ego=15.0, lateral_state=losing, **interrupt).regime, REGIME_FREE)
    limiter = ModelCurveSpeedLimiter(make_cp())
    settle(limiter, straight, 40, v_ego=15.0, lateral_active=True, lateral_state=losing)
    limiter.reset()
    self.assertEqual(limiter.regime, REGIME_FREE)
    self.assertIsNone(limiter.update(straight, v_ego=15.0, lateral_active=False, lateral_state=losing).a_target)

  def test_no_reaction_without_active_lateral_or_with_the_driver_steering_or_at_crawl_speed(self):
    limiter = ModelCurveSpeedLimiter(make_cp())
    straight = make_model(15.0)
    losing = tracking(torque=1.0, error=0.6, lat=2.3, pinned=True)
    self.assertIsNone(settle(limiter, straight, 20, v_ego=15.0, lateral_active=False, lateral_state=losing).a_target)
    self.assertIsNone(settle(limiter, straight, 20, v_ego=15.0, lateral_active=True, steering_pressed=True, lateral_state=losing).a_target)
    slow = settle(limiter, make_model(2.0), 20, v_ego=2.0, lateral_active=True, lateral_state=losing)
    self.assertNotEqual(slow.regime, REGIME_BRAKE)

  def test_lateral_state_reads_the_torque_controller_and_fails_closed(self):
    torque = SimpleNamespace(active=True, output=-0.9, error=0.2, actualLateralAccel=-2.0, desiredLateralAccel=-2.2, saturated=False)
    cs = SimpleNamespace(lateralControlState=SimpleNamespace(which=lambda: 'torqueState', torqueState=torque))
    state = LateralState.from_controls_state(cs)
    self.assertTrue(state.active and math.isclose(state.torque, 0.9) and not state.pinned)
    self.assertAlmostEqual(state.understeer, -0.2)                            # desired -2.2, actual -2.0: exit overshoot, not understeer
    held = SimpleNamespace(active=True, output=0.6, error=0.5, actualLateralAccel=2.3, desiredLateralAccel=2.8, saturated=True)
    cs = SimpleNamespace(lateralControlState=SimpleNamespace(which=lambda: 'torqueState', torqueState=held))
    self.assertTrue(LateralState.from_controls_state(cs).pinned)                # the controller's own saturation pins below T_PIN
    rack = SimpleNamespace(active=True, output=0.99, error=0.5, actualLateralAccel=2.3, desiredLateralAccel=2.8, saturated=False,
                           torqueLimited=True)
    cs = SimpleNamespace(lateralControlState=SimpleNamespace(which=lambda: 'rackState', rackState=rack))
    self.assertFalse(LateralState.from_controls_state(cs).active)               # the retired rack controller is not read
    self.assertFalse(LateralState.from_controls_state(SimpleNamespace(lateralControlState=SimpleNamespace(which=lambda: 'pidState'))).active)
    bad = SimpleNamespace(active=True, output=math.nan, error=0.0, actualLateralAccel=0.0, desiredLateralAccel=0.0, saturated=False)
    cs = SimpleNamespace(lateralControlState=SimpleNamespace(which=lambda: 'torqueState', torqueState=bad))
    self.assertFalse(LateralState.from_controls_state(cs).active)


class TestPlannerIntegration(unittest.TestCase):
  def test_the_candidate_only_ever_lowers_the_plan(self):
    from openpilot.selfdrive.controls.lib import longitudinal_planner
    from unittest.mock import patch
    from openpilot.selfdrive.controls.lib.model_curve_speed import CurveResult
    from openpilot.selfdrive.controls.tests.test_daily_curve_integration import curve_messages
    from openpilot.selfdrive.controls.tests.test_daily_blotv3_integration import car_params

    # Daily validates model health before admitting a curve; exercise real shaped cereal inputs and its real MPC.
    planner = longitudinal_planner.LongitudinalPlanner(car_params(), init_v=15.0)
    sm = curve_messages(curvature=0.0, v_ego=15.0)
    with patch.object(planner.curve_speed_limiter, 'update', return_value=CurveResult(2.5, 'anticipate')):
      planner.update(sm)
      self.assertLessEqual(planner.output_a_target, 0.5)                     # a higher candidate never raises the plan
    with patch.object(planner.curve_speed_limiter, 'update', return_value=CurveResult(-1.2, 'coast')):
      planner.update(sm)
      self.assertAlmostEqual(planner.output_a_target, -1.2)                  # a lower one binds

  def test_the_policys_decision_is_published(self):
    from openpilot.selfdrive.controls.lib import longitudinal_planner
    from openpilot.selfdrive.controls.lib.model_curve_speed import CurveResult

    class FakeSubMaster(dict):
      logMonoTime = {'modelV2': 0}

      def all_checks(self, services=None):
        return True

    planner = longitudinal_planner.LongitudinalPlanner(SimpleNamespace(openpilotLongitudinalControl=True, longitudinalActuatorDelay=0.2,
                                                                        radarUnavailable=True, flags=0, brand=''))
    sm = FakeSubMaster(radarState=SimpleNamespace(leadOne=SimpleNamespace(present=False)), carState=SimpleNamespace(vEgo=0.0))
    sent = {}
    pm = SimpleNamespace(send=lambda service, msg: sent.__setitem__(service, msg))
    for result in (CurveResult(-1.1, REGIME_BRAKE, 21.5, 34.0, 2.6, True), CurveResult()):
      planner.curve = result
      planner.publish(sm, pm)
      state = sent['curvePolicyState'].curvePolicyState
      self.assertEqual(state.regime, result.regime)
      self.assertEqual(state.active, result.a_target is not None)
      self.assertAlmostEqual(state.aTarget, result.a_target if result.a_target is not None else 0.0, places=5)
      self.assertAlmostEqual(state.vLimit, result.v_limit, places=4)
      self.assertAlmostEqual(state.limitDistance, result.distance, places=4)
      self.assertAlmostEqual(state.authorityFactor, result.authority_factor, places=5)
      self.assertEqual(state.holding, result.holding)


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
    b = settle(banked, straight, 400, v_ego=20.0, lateral_active=True, roll=-0.05, lateral_state=state)   # banked 0.05 rad in its favor
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

  def test_the_hold_releases_where_the_owner_goes_back_to_power(self):
    # his six clean accelerate-out onsets sit at 1.13-2.26 m/s^2 of lateral; the policy used to wait for under 1.0
    limiter, bend, _ = self._lifted()
    opening = tracking(torque=0.3, lat=(BEND_OPEN_A_LAT + 1.0) / 2.0)         # 1.25: he is already on the throttle here
    result = settle(limiter, bend, round(BEND_OPEN_S / DT_MDL) + 4, v_ego=15.0, lateral_active=True, lateral_state=opening)
    self.assertFalse(result.holding)
    self.assertGreater(result.a_target, 0.0)

  def test_a_bend_still_at_the_owners_own_cornering_level_still_holds(self):
    limiter, bend, _ = self._lifted()
    cornering = tracking(torque=0.5, error=0.35, lat=BEND_OPEN_A_LAT + 0.5)
    result = settle(limiter, bend, round(HOLD_MAX_S / DT_MDL) - 20, v_ego=15.0, lateral_active=True, lateral_state=cornering)
    self.assertTrue(result.holding)
    self.assertLessEqual(result.a_target, 1e-9)

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


class TestCurveFloor(unittest.TestCase):
  # the floor under a turn: the owner's own pace through a right-angle turn, less in a hairpin; neither layer slows a turn below it

  @staticmethod
  def turning(v, k, ratio, torque, pinned):
    # the steering in a turn of curvature k at speed v: asked k*v^2, delivering `ratio` of it at `torque`
    desired = k * v * v
    return LateralState(True, torque, desired * (1.0 - ratio), ratio * desired, desired, pinned)

  def test_the_floor_is_the_owners_pace_in_a_right_angle_turn_and_lower_in_a_hairpin(self):
    self.assertAlmostEqual(curve_floor(0.0), V_CURVE_FLOOR)                       # a straight: the speed floor alone
    self.assertAlmostEqual(curve_floor(0.08), V_CURVE_FLOOR)                      # R 12.5 m, a right-angle street corner: 12 mph
    self.assertAlmostEqual(curve_floor(0.10), math.sqrt(A_FLOOR_LAT / 0.10))      # R 10 m: 5.39, where the two terms meet
    self.assertAlmostEqual(curve_floor(0.20), math.sqrt(A_FLOOR_LAT / 0.20))      # a R 5 m hairpin: 3.8, the lateral term
    self.assertAlmostEqual(curve_floor(-0.20), curve_floor(0.20))
    self.assertIsInstance(curve_floor(0.1), float)
    np.testing.assert_allclose(curve_floor(np.array([0.0, 0.1, 0.4])),
                               [V_CURVE_FLOOR, math.sqrt(A_FLOOR_LAT / 0.1), math.sqrt(A_FLOOR_LAT / 0.4)])
    self.assertLess(A_FLOOR_LAT, A_LAT_COMFORT)                                   # comfort stays the ceiling over the floor
    self.assertGreater(V_CURVE_FLOOR, V_REACT_MIN)                                # the curvature read keeps its own floor

  def test_a_walking_pace_pin_on_a_straight_reads_free(self):
    # route 0x14b t=505: 0.67 m/s on a straight, torque 1.0 against a desired of 0.017 m/s^2 (an exit unwind), error 0.016; the
    # coast regime asked -0.23 there. Under the floor the reaction has nothing to say and arms no hold
    limiter = ModelCurveSpeedLimiter(make_cp())
    straight = make_model(0.67)
    unwinding = LateralState(True, 1.0, 0.016, 0.002, 0.017, False)
    for _ in range(round(3.0 / DT_MDL)):
      result = limiter.update(straight, v_ego=0.67, a_ego=-0.6, lateral_active=True, accel_coast=-0.3, lateral_state=unwinding)
      self.assertEqual(result.regime, REGIME_FREE)
      self.assertIsNone(result.a_target)
      self.assertFalse(result.holding)

  def test_the_reaction_never_pulls_a_turn_below_its_floor_and_settles_there(self):
    # a R 10 m turn entered at 6.5 m/s with the steering pinned and losing the line: the brake regime engages above the floor
    # and the car, following the candidate, settles at the floor (5.39 m/s) instead of being pulled toward 2.9. Under the floor
    # the regime is free and the candidate never negative
    limiter = ModelCurveSpeedLimiter(make_cp())
    k, v, a = 0.10, 6.5, 0.0
    floor = curve_floor(k)
    regimes, speeds = [], []
    for _ in range(round(12.0 / DT_MDL)):
      result = limiter.update(make_model(v, np.full(N, k)), v_ego=v, a_ego=a, lateral_active=True, accel_coast=-0.3,
                              lateral_state=self.turning(v, k, 0.8, 1.0, True))
      regimes.append(result.regime)
      a = result.a_target if result.a_target is not None else 0.0
      if v < floor - 0.02:
        self.assertNotIn(result.regime, (REGIME_COAST, REGIME_BRAKE))
        self.assertGreaterEqual(a, 0.0)
      v = max(v + a * DT_MDL, 0.0)
      speeds.append(v)
    self.assertIn(REGIME_BRAKE, regimes)
    self.assertGreaterEqual(min(speeds), floor - 0.05)
    self.assertAlmostEqual(speeds[-1], floor, delta=0.05)

  def _fade(self, regime, ratio, torque, pinned):
    # heavy but tracking (coast) or pinned and losing (brake), swept down through the floor of a R 10 m turn at 0.01 m/s a frame
    # on a straight path (the reaction alone): the request never deepens past the approach to the floor, shrinks to zero at it
    # and is gone under it, where the regime is free and no hold is armed. The hand-back is the module's own ramp, not a step
    k, floor, step = 0.10, curve_floor(0.10), 0.01
    limiter = ModelCurveSpeedLimiter(make_cp())
    straight = make_model(floor + 0.6)

    def frame(v):
      return limiter.update(straight, v_ego=v, a_ego=-0.2, lateral_active=True, accel_coast=-0.3,
                            lateral_state=self.turning(v, k, ratio, torque, pinned))
    result = None
    for _ in range(round(1.0 / DT_MDL)):
      result = frame(floor + 0.6)
    self.assertEqual(result.regime, regime)
    trace = []
    for v in np.arange(floor + 0.6, floor - 0.2, -step):
      result = frame(float(v))
      trace.append((float(v), result.a_target, result.regime, result.holding))
    requests = [a if a is not None else 0.0 for _, a, _, _ in trace]
    self.assertLessEqual(max(abs(x - y) for x, y in zip(requests[1:], requests[:-1], strict=True)), J_UP * DT_MDL + 3 * step)
    # the curvature read is the median of the last three lateral samples over the current speed squared, so on a slowing car
    # the floor reads a few mm/s low: the frames within 2 cm/s of it are the crossing
    for v, a, reg, holding in trace:
      self.assertFalse(holding, v)
      if v >= floor + 2 * step:
        self.assertEqual(reg, regime, v)
        self.assertLessEqual(a, 0.0, v)
        self.assertGreaterEqual(a, (floor - v) / T_RESTORE - 2 * step, v)        # never deeper than the approach to the floor
      elif v <= floor - 2 * step:
        self.assertEqual(reg, REGIME_FREE, v)
        self.assertTrue(a is None or a >= 0.0, v)
    above = [a for v, a, _, _ in trace if v >= floor]
    self.assertLess(abs(above[-1]), 3 * step)                                    # about zero when the regime lets go
    self.assertLessEqual(above[0], -0.3 + 1e-9)                                  # a real request before the fade

  def test_the_lift_fades_to_zero_at_the_floor_without_a_step(self):
    self._fade(REGIME_COAST, ratio=0.97, torque=0.9, pinned=False)

  def test_the_brake_fades_to_zero_at_the_floor_without_a_step(self):
    self._fade(REGIME_BRAKE, ratio=0.8, torque=1.0, pinned=True)

  def test_a_hold_lets_the_car_back_up_to_the_floor_and_resumes_above_it(self):
    # a lift at 8 m/s in a bend ends into a hold with the bend still on. Slowed under the floor by something else, the car is
    # free to come back up to it; above it the hold clamps again until the bend reads open
    limiter = ModelCurveSpeedLimiter(make_cp())
    straight = make_model(8.0)
    settle(limiter, straight, round(COAST_ENTER_S / DT_MDL) + 4, v_ego=8.0, lateral_active=True, accel_coast=-0.3,
           lateral_state=tracking(torque=0.9, error=0.1, lat=2.0))
    self.assertEqual(limiter.regime, REGIME_COAST)
    held = settle(limiter, straight, round(COAST_EXIT_S / DT_MDL) + 4, v_ego=8.0, lateral_active=True, lateral_state=tracking(torque=0.5, lat=2.0))
    self.assertTrue(held.holding)
    self.assertEqual(held.a_target, 0.0)
    under = [limiter.update(straight, v_ego=5.0, lateral_active=True, lateral_state=tracking(torque=0.5, lat=2.0))
             for _ in range(round(A_CURVE_FREE / J_UP / DT_MDL) + 2)]
    self.assertTrue(all(r.holding for r in under))                               # the hold is kept ...
    self.assertTrue(all(r.a_target is None or r.a_target > 0.0 for r in under))  # ... but under the floor its clamp ramps out ...
    self.assertIsNone(under[-1].a_target)                                        # ... and is gone
    back = settle(limiter, straight, 2, v_ego=6.0, lateral_active=True, lateral_state=tracking(torque=0.5, lat=2.0))
    self.assertTrue(back.holding)
    self.assertEqual(back.a_target, 0.0)                                         # above it the clamp is back at once
    opened = settle(limiter, straight, round(BEND_OPEN_S / DT_MDL) + 2, v_ego=6.0, lateral_active=True, lateral_state=tracking(torque=0.5, lat=0.5))
    self.assertFalse(opened.holding)

  def test_a_dwell_does_not_carry_across_the_floor(self):
    # heavy for most of the coast dwell above the floor, one frame under it, heavy again above it: the dwell starts over, so the
    # crossing cannot complete a lift the steering has not asked for continuously. R 17 m, where the floor is the 5.4 m/s term
    # whatever the lateral read's lag says
    limiter = ModelCurveSpeedLimiter(make_cp())
    straight, k = make_model(6.0), 0.06
    frames = round(COAST_ENTER_S / DT_MDL)
    for _ in range(frames - 1):
      result = limiter.update(straight, v_ego=6.0, lateral_active=True, lateral_state=self.turning(6.0, k, 0.97, 0.9, False))
    self.assertNotEqual(result.regime, REGIME_COAST)
    limiter.update(straight, v_ego=5.0, lateral_active=True, lateral_state=self.turning(5.0, k, 0.97, 0.9, False))
    for i in range(frames - 1):
      result = limiter.update(straight, v_ego=6.0, lateral_active=True, lateral_state=self.turning(6.0, k, 0.97, 0.9, False))
      self.assertNotEqual(result.regime, REGIME_COAST, i)
    result = limiter.update(straight, v_ego=6.0, lateral_active=True, lateral_state=self.turning(6.0, k, 0.97, 0.9, False))
    self.assertEqual(result.regime, REGIME_COAST)

  def test_a_hairpin_the_model_sees_is_priced_at_its_own_floor(self):
    # R 5 m: the torque model says 3.27 m/s, comfort 4.12; the floor is the lateral term, 3.81, not the 5.4 of a right-angle turn
    result = settle(ModelCurveSpeedLimiter(make_cp()), make_model(5.0, np.full(N, 0.20)), 5, v_ego=5.0, lateral_active=True, lateral_state=tracking())
    self.assertAlmostEqual(result.v_limit, math.sqrt(A_FLOOR_LAT / 0.20), places=6)
    self.assertLess(result.v_limit, V_CURVE_FLOOR)

  def test_the_anticipation_prices_no_node_below_the_floor_and_leaves_the_road_above_it_alone(self):
    # in town (no demonstrated hold) the torque model priced a R 10 m node at 4.62 m/s; it reads the floor, 5.39. A R 50 m node
    # (10.3) and a highway sweeper are untouched, and so is the per-node pricing itself
    for k, expect in ((0.10, curve_floor(0.10)), (0.02, math.sqrt(MODEL_MARGIN / 0.02))):
      for v in (4.0, 6.0):
        result = settle(ModelCurveSpeedLimiter(make_cp()), make_model(v, np.full(N, k)), 5, v_ego=v, lateral_active=True, lateral_state=tracking())
        self.assertAlmostEqual(result.v_limit, expect, places=6, msg=(k, v))
    highway = settle(ModelCurveSpeedLimiter(make_cp()), make_model(25.0, np.full(N, 0.005)), 5, v_ego=25.0, lateral_active=True, lateral_state=tracking())
    self.assertAlmostEqual(highway.v_limit, math.sqrt(AUTHORITY_HELD_LATERAL / 0.005), places=4)
    self.assertAlmostEqual(float(curve_speed_limits(np.array([0.10]), TORQUE, 0.0)[0]), math.sqrt(MODEL_MARGIN / 0.10))


class TestBankContinuation(unittest.TestCase):
  # beyond the roll horizon, the bend the car is in keeps the bank the car is measuring; anything else there is bank-blind as before
  V = 24.0

  def near(self):
    return self.V * T <= self.V * ROLL_HORIZON_S

  def test_a_far_node_of_the_bend_the_car_is_in_keeps_the_banks_share_in_proportion(self):
    near, v = self.near(), self.V
    same = np.full(N, 0.005)                                                     # one R 200 m bend, continuing
    np.testing.assert_allclose(bank_continuation(same, near, v), 1.0)
    opening = np.where(near, 0.005, 0.0025)                                      # opening to R 400 m: half the bank
    np.testing.assert_allclose(bank_continuation(opening, near, v)[~near], 0.5)
    tightening = np.where(near, 0.005, 0.005 * (1.0 + BANK_CONTINUATION_TOLERANCE / 2))
    np.testing.assert_allclose(bank_continuation(tightening, near, v)[~near], 0.5)   # half way into the tolerance: half
    past = np.where(near, 0.005, 0.005 * (1.0 + BANK_CONTINUATION_TOLERANCE))
    np.testing.assert_allclose(bank_continuation(past, near, v)[~near], 0.0)
    reversing = np.where(near, 0.005, -0.005)                                     # the second half of an S
    np.testing.assert_allclose(bank_continuation(reversing, near, v)[~near], 0.0)
    np.testing.assert_allclose(bank_continuation(reversing, near, v)[near], 1.0)  # the near nodes keep the live roll regardless

  def test_a_straight_or_a_light_bend_lends_its_roll_to_nothing_beyond_the_horizon(self):
    near, v = self.near(), self.V
    light = np.full(N, 0.9 * BANK_CONTINUATION_MIN_LATERAL / v ** 2)             # under 1 m/s^2 near the car: the roll is a crown
    np.testing.assert_allclose(bank_continuation(light, near, v)[~near], 0.0)
    np.testing.assert_allclose(bank_continuation(light, near, v)[near], 1.0)
    approach = np.where(near, 0.0, 0.006)                                        # a bend ahead seen from a straight: as before
    np.testing.assert_allclose(bank_continuation(approach, near, v)[~near], 0.0)
    loaded = np.full(N, 1.01 * BANK_CONTINUATION_MIN_LATERAL / v ** 2)           # just over it: the bend's own far nodes keep it
    np.testing.assert_allclose(bank_continuation(loaded, near, v)[~near], 1.0)

  def test_a_bend_of_its_own_behind_a_straight_keeps_no_bank(self):
    # the bend the car is in ends in a straight 60 m out, and another bend the same way follows: the first bend's bank says
    # nothing about the second, so the second is priced bank-blind and, being the strictest, sets the limit
    v, k, roll = self.V, 0.005, 0.04
    x = v * T
    near = self.near()
    gap = (x >= 60.0) & (x < 100.0)
    path = np.where(gap, BANK_CONTINUATION_END * k / 2, k)                        # the gap curves the same way, under the end mark
    weights = bank_continuation(path, near, v)
    np.testing.assert_allclose(weights[gap | (x >= 100.0)], 0.0)
    np.testing.assert_allclose(weights[near], 1.0)
    result = settle(ModelCurveSpeedLimiter(make_cp()), make_model(v, path), 3, v_ego=v, lateral_active=True, roll=roll, lateral_state=tracking())
    self.assertAlmostEqual(result.v_limit, math.sqrt(AUTHORITY_HELD_LATERAL / k), places=4)
    self.assertGreater(result.distance, 100.0)

  def test_the_reference_is_the_near_paths_mean_not_its_tightest_node(self):
    # the near path tightening from R 500 to R 250 m, the far nodes at R 222 m: against the mean (R 333) that is 50 % tighter and
    # keeps no bank; against the tightest near node it would be 12 % tighter and keep half of it
    v, roll = self.V, 0.04
    x = v * T
    near = self.near()
    graded = np.where(near, 0.002 + 0.002 * x / (v * ROLL_HORIZON_S), 0.0045)
    np.testing.assert_allclose(bank_continuation(graded, near, v)[~near], 0.0)
    result = settle(ModelCurveSpeedLimiter(make_cp()), make_model(v, graded), 3, v_ego=v, lateral_active=True, roll=roll, lateral_state=tracking())
    self.assertAlmostEqual(result.v_limit, math.sqrt(AUTHORITY_HELD_LATERAL / 0.0045), places=4)

  def test_an_adverse_roll_is_charged_to_the_bend_beyond_the_horizon(self):
    # the bend continuing 20 % tighter with the car on a crown against it: the far node keeps a fifth of the adverse roll, so its
    # limit is lower than bank-blind, never higher
    v, roll = self.V, -0.04
    near = self.near()
    model = make_model(v, np.where(near, 0.004, 0.0048))
    charged = settle(ModelCurveSpeedLimiter(make_cp()), model, 3, v_ego=v, lateral_active=True, roll=roll, lateral_state=tracking())
    weight = float(np.interp(1.2, (0.0, 1.0, 1.0 + BANK_CONTINUATION_TOLERANCE), (0.0, 1.0, 0.0)))
    self.assertAlmostEqual(charged.v_limit, math.sqrt((AUTHORITY_HELD_LATERAL + weight * roll * ACCELERATION_DUE_TO_GRAVITY) / 0.0048), places=4)
    self.assertLess(charged.v_limit, math.sqrt(AUTHORITY_HELD_LATERAL / 0.0048))

  def test_a_steady_banked_bend_is_priced_with_its_bank_beyond_the_horizon(self):
    # one R 200 m bend at 24 m/s banked 2.3 deg in its favour (the commute bend): its strictest node sits past the 2 s horizon
    # and used to be priced without the bank; it now reads as the near nodes do. A crown against the turn is charged there too
    v, k, roll = self.V, 0.005, 0.04
    bend = make_model(v, np.full(N, k))
    limits = {r: settle(ModelCurveSpeedLimiter(make_cp()), bend, 3, v_ego=v, lateral_active=True, roll=r, lateral_state=tracking()).v_limit
              for r in (roll, 0.0, -roll)}
    g = ACCELERATION_DUE_TO_GRAVITY
    self.assertAlmostEqual(limits[0.0], math.sqrt(AUTHORITY_HELD_LATERAL / k), places=4)
    self.assertAlmostEqual(limits[roll], math.sqrt((AUTHORITY_HELD_LATERAL + roll * g) / k), places=4)
    self.assertAlmostEqual(limits[-roll], math.sqrt((AUTHORITY_HELD_LATERAL - roll * g) / k), places=4)

  def test_a_tightening_bend_ahead_keeps_no_bank_it_has_not_shown(self):
    # route 0x19e t=1415: a right-hander banked in its favour that the model read at half its curvature until 80 m out. With the
    # near path in the gentle part and the far nodes tightening past the tolerance, those are priced bank-blind, as before
    v, roll = self.V, 0.06
    near = self.near()
    model = make_model(v, np.where(near, 0.004, 0.004 * 1.6))
    banked = settle(ModelCurveSpeedLimiter(make_cp()), model, 3, v_ego=v, lateral_active=True, roll=roll, lateral_state=tracking())
    level = settle(ModelCurveSpeedLimiter(make_cp()), model, 3, v_ego=v, lateral_active=True, roll=0.0, lateral_state=tracking())
    self.assertAlmostEqual(banked.v_limit, level.v_limit, places=6)
    self.assertAlmostEqual(banked.v_limit, math.sqrt(AUTHORITY_HELD_LATERAL / (0.004 * 1.6)), places=4)
    self.assertAlmostEqual(banked.a_target, level.a_target, places=6)

  def test_the_bank_fades_out_as_the_bend_ahead_tightens(self):
    # between the near path's curvature and the tolerance the credit is continuous, so a bend read a little tighter ahead lowers
    # the limit smoothly instead of flipping it between banked and bank-blind
    v, roll = self.V, 0.04
    near = self.near()
    limits = [settle(ModelCurveSpeedLimiter(make_cp()), make_model(v, np.where(near, 0.005, 0.005 * ratio)), 3, v_ego=v, lateral_active=True,
                     roll=roll, lateral_state=tracking()).v_limit for ratio in np.linspace(1.0, 1.0 + BANK_CONTINUATION_TOLERANCE, 6)]
    self.assertAlmostEqual(limits[0], math.sqrt((AUTHORITY_HELD_LATERAL + roll * ACCELERATION_DUE_TO_GRAVITY) / 0.005), places=4)
    self.assertAlmostEqual(limits[-1], math.sqrt(AUTHORITY_HELD_LATERAL / (0.005 * (1.0 + BANK_CONTINUATION_TOLERANCE))), places=4)
    steps = [a - b for a, b in zip(limits[:-1], limits[1:], strict=True)]
    self.assertGreaterEqual(min(steps), -1e-9)                                     # monotone: tighter and less bank both lower it
    self.assertLess(max(steps), 1.2)                                               # 5 % tighter is never a jump

  def test_the_demand_is_never_positive_while_the_steering_is_heavy_dwell_or_not(self):
    # a limit above the car wants a small acceleration; with the torque at the coast threshold for one frame the demand is clamped
    # to zero at once, not after the coast's dwell (the 2026-10-31 package replay saw +0.5 through it on 0x185). Below the
    # threshold the candidate stays positive
    limiter = ModelCurveSpeedLimiter(make_cp())
    bend = make_model(15.0, np.full(N, TestHold.CURVATURE))
    light = settle(limiter, bend, 3, v_ego=15.0, lateral_active=True, lateral_state=tracking(torque=0.5, error=0.1, lat=2.0))
    self.assertGreater(light.a_target, 0.0)
    firm = limiter.update(bend, v_ego=15.0, lateral_active=True, lateral_state=tracking(torque=T_COAST - 0.05, error=0.1, lat=2.0))
    self.assertGreater(firm.a_target, 0.0)
    heavy = limiter.update(bend, v_ego=15.0, lateral_active=True, lateral_state=tracking(torque=T_COAST, error=0.1, lat=2.0))
    self.assertNotEqual(heavy.regime, REGIME_COAST)                              # the dwell has not elapsed ...
    self.assertLessEqual(heavy.a_target, 0.0)                                      # ... and the candidate is already not positive

  def test_a_heavy_frame_on_a_straight_has_nothing_to_clamp(self):
    # with no candidate the guard must not invent one: a torque blip on a straight, or a torque breathing around the threshold
    # so the dwell never completes, publishes nothing
    limiter = ModelCurveSpeedLimiter(make_cp())
    straight = make_model(20.0)
    for torque in (0.9, 0.5, 0.5):
      self.assertIsNone(limiter.update(straight, v_ego=20.0, lateral_active=True, lateral_state=tracking(torque=torque, lat=1.2)).a_target)
    for i in range(60):
      result = limiter.update(straight, v_ego=20.0, lateral_active=True, lateral_state=tracking(torque=(0.80, 0.86)[i % 2], lat=1.2))
      self.assertIsNone(result.a_target, i)
      self.assertEqual(result.regime, REGIME_FREE, i)

  def test_the_owners_foot_keeps_its_grace_from_the_guard(self):
    # the 5 s after a gas override: the anticipation may hold but never brake, and heavy steering does not clamp the owner's speed
    # inside the grace; a fresh limiter in the same frame does
    bend = make_model(15.0, np.full(N, TestHold.CURVATURE))
    graced = ModelCurveSpeedLimiter(make_cp())
    settle(graced, bend, 3, v_ego=15.0, lateral_active=True, gas_pressed=True, lateral_state=tracking(torque=0.5, error=0.1, lat=2.0))
    inside = graced.update(bend, v_ego=15.0, lateral_active=True, lateral_state=tracking(torque=T_COAST, error=0.1, lat=2.0))
    self.assertGreater(inside.a_target, 0.0)
    plain = ModelCurveSpeedLimiter(make_cp())
    settle(plain, bend, 3, v_ego=15.0, lateral_active=True, lateral_state=tracking(torque=0.5, error=0.1, lat=2.0))
    self.assertLessEqual(plain.update(bend, v_ego=15.0, lateral_active=True, lateral_state=tracking(torque=T_COAST, error=0.1, lat=2.0)).a_target, 0.0)


if __name__ == "__main__":
  unittest.main()

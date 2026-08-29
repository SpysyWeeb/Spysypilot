import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MAX
from opendbc.car.structs import car
from openpilot.cereal import log
import openpilot.cereal.messaging as messaging
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_planner import (A_CRUISE_MAX_LAUNCH, A_CRUISE_MAX_HIGH_SPEED, A_CRUISE_MAX_SPEED,
                                                                   A_CRUISE_MIN, CRUISE_COMFORT_KP, J_CRUISE_BP, J_CRUISE_VALS,
                                                                   LongitudinalPlanner, get_cruise_accel, get_cruise_comfort_accel,
                                                                   get_max_accel, get_max_accel_request, ordinary_cruise_comfort_enabled)
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant, _PlantSubMaster

LongitudinalPlanSource = log.LongitudinalPlan.LongitudinalPlanSource

CP = car.CarParams.new_message(steerRatio=17.94, wheelbase=2.9)  # Palisade


def steer_angle_for(a_y, v_ego):
  return a_y * CP.steerRatio * CP.wheelbase / (v_ego ** 2 * CV.DEG_TO_RAD)


def settled_cruise_accel(v_cruise, v_ego, angle_steers=0.0, accel_coast=-0.3, comfort=True, e2e=False):
  target = 0.0
  for _ in range(200):
    target = get_cruise_accel(e2e, v_cruise, v_ego, target, angle_steers, CP, DT_MDL, accel_coast, True, comfort)
  return target


class TestAccelEnvelope:
  def test_curve_endpoints_and_floor(self):
    assert get_max_accel_request(0.0) == pytest.approx(A_CRUISE_MAX_LAUNCH)
    assert get_max_accel_request(A_CRUISE_MAX_SPEED) == pytest.approx(A_CRUISE_MAX_HIGH_SPEED)
    assert get_max_accel_request(A_CRUISE_MAX_SPEED + 15.0) == pytest.approx(A_CRUISE_MAX_HIGH_SPEED)

  def test_curve_is_smooth_monotonic_and_convex(self):
    speeds = np.linspace(0.0, A_CRUISE_MAX_SPEED, 401)
    requests = np.array([get_max_accel_request(v) for v in speeds])
    assert np.all(np.diff(requests) <= 0.0)
    assert np.all(np.diff(requests, n=2) >= -1e-9)
    # the floor is reached with zero slope, not a corner
    assert abs(requests[-1] - requests[-2]) < 1e-4

  def test_deployed_platform_clamps_the_request(self):
    assert ACCEL_MAX <= A_CRUISE_MAX_LAUNCH
    assert get_max_accel(0.0) == pytest.approx(ACCEL_MAX)
    assert get_max_accel(20.0) == pytest.approx(1.025, abs=1e-3)
    assert get_max_accel(30.0) == pytest.approx(0.653, abs=1e-3)

  def test_jerk_schedule_bounds_the_first_step(self):
    v_ego = 75.0 * CV.MPH_TO_MS
    first = get_cruise_accel(False, v_ego + 5.0 * CV.MPH_TO_MS, v_ego, 0.0, 0.0, CP, DT_MDL, -0.3, True)
    assert first == pytest.approx(np.interp(v_ego, J_CRUISE_BP, J_CRUISE_VALS) * DT_MDL)


class TestTurnBudget:
  def test_straight_launch_is_never_clipped(self):
    for v_ego in (0.5, 5.0, 10.0):
      assert settled_cruise_accel(v_ego + 20.0, v_ego, comfort=False) == pytest.approx(get_max_accel(v_ego))

  def test_lateral_acceleration_consumes_the_budget_at_road_speed(self):
    v_ego = 20.0
    a_x_allowed = np.sqrt(1.7 ** 2 - 1.5 ** 2)
    assert a_x_allowed < get_max_accel(v_ego)
    settled = settled_cruise_accel(v_ego + 10.0, v_ego, angle_steers=steer_angle_for(1.5, v_ego), comfort=False)
    assert settled == pytest.approx(a_x_allowed)

  def test_budget_grows_with_the_envelope_at_low_speed(self):
    v_ego = 5.0
    envelope = get_max_accel(v_ego)
    a_x_allowed = np.sqrt(max(envelope, 1.7) ** 2 - 1.5 ** 2)
    settled = settled_cruise_accel(v_ego + 20.0, v_ego, angle_steers=steer_angle_for(1.5, v_ego), comfort=False)
    assert settled == pytest.approx(min(envelope, a_x_allowed))


class TestCruiseComfort:
  def test_five_mph_increase_at_highway_speed_is_proportional(self):
    v_ego = 74.5 * CV.MPH_TO_MS
    v_cruise = 79.5 * CV.MPH_TO_MS
    expected = CRUISE_COMFORT_KP * (v_cruise - v_ego)
    assert expected == pytest.approx(0.402, abs=1e-3)
    assert settled_cruise_accel(v_cruise, v_ego, accel_coast=-0.25) == pytest.approx(expected)

  def test_five_mph_reduction_coasts_instead_of_full_braking(self):
    v_ego = 79.6 * CV.MPH_TO_MS
    v_cruise = 74.6 * CV.MPH_TO_MS
    settled = settled_cruise_accel(v_cruise, v_ego, accel_coast=-0.39)
    assert settled == pytest.approx(-0.402, abs=1e-3)
    assert settled > A_CRUISE_MIN

  def test_reduction_follows_an_uphill_coast_but_not_a_downhill_push(self):
    v_ego = 80.0 * CV.MPH_TO_MS
    v_cruise = 75.0 * CV.MPH_TO_MS
    assert get_cruise_comfort_accel(v_cruise, v_ego, -0.6) == pytest.approx(-0.6)
    assert get_cruise_comfort_accel(v_cruise, v_ego, 0.2) == pytest.approx(-0.402, abs=1e-3)

  def test_small_corrections_taper_continuously(self):
    v_ego = 75.0 * CV.MPH_TO_MS
    half = get_cruise_comfort_accel(v_ego + 2.5 * CV.MPH_TO_MS, v_ego, -0.3)
    full = get_cruise_comfort_accel(v_ego + 5.0 * CV.MPH_TO_MS, v_ego, -0.3)
    assert half == pytest.approx(full / 2.0)

  def test_large_errors_keep_the_envelope_and_braking_limit(self):
    v_ego = 75.0 * CV.MPH_TO_MS
    assert settled_cruise_accel(v_ego + 15.0 * CV.MPH_TO_MS, v_ego) == pytest.approx(get_max_accel(v_ego))
    assert settled_cruise_accel(v_ego - 15.0 * CV.MPH_TO_MS, v_ego) == pytest.approx(A_CRUISE_MIN)

  def test_low_speed_launch_keeps_legacy_authority(self):
    v_ego = 5.0
    v_cruise = v_ego + 5.0 * CV.MPH_TO_MS
    assert settled_cruise_accel(v_cruise, v_ego, comfort=True) == pytest.approx(settled_cruise_accel(v_cruise, v_ego, comfort=False))

  def test_comfort_is_only_for_ordinary_chill_cruise(self):
    assert ordinary_cruise_comfort_enabled(False, False, True)
    assert not ordinary_cruise_comfort_enabled(True, False, True)
    assert not ordinary_cruise_comfort_enabled(False, True, True)
    assert not ordinary_cruise_comfort_enabled(False, False, False)
    assert not ordinary_cruise_comfort_enabled(False, False, True, speed_limiter_active=True)

  def test_coast_limit_still_applies_with_comfort(self):
    v_ego = 4.0
    v_cruise = v_ego + 5.0 * CV.MPH_TO_MS
    for comfort in (False, True):
      target = 0.0
      for _ in range(200):
        target = get_cruise_accel(False, v_cruise, v_ego, target, 0.0, CP, DT_MDL, -0.3, False, comfort)
      assert target == pytest.approx(np.interp(v_ego, [2.5, 5.0], [get_max_accel(v_ego), -0.3]))

  def test_comfort_blends_in_between_8_and_15_mps(self):
    v_ego = 11.5
    v_cruise = v_ego + 5.0 * CV.MPH_TO_MS
    legacy = settled_cruise_accel(v_cruise, v_ego, comfort=False)
    full = np.clip(get_cruise_comfort_accel(v_cruise, v_ego, -0.3), A_CRUISE_MIN, get_max_accel(v_ego))
    assert settled_cruise_accel(v_cruise, v_ego) == pytest.approx((legacy + full) / 2.0)


class TestPlannerCruise:
  def run_plant(self, seconds, **kwargs):
    plant = Plant(**kwargs)
    log = []
    while plant.current_time < seconds:
      plant.step(v_cruise=50.0)
      log.append((plant.current_time, plant.speed, plant.acceleration))
    return plant, log

  def test_standstill_launch_starts_smooth_and_grows_quickly(self):
    # owner acceptance: smooth onset, then at least half the envelope within a second; the cruise candidate owns this launch
    plant, log = self.run_plant(1.2, speed=0.0, distance_lead=200.0, lead_relevancy=False)
    assert plant.planner.mpc.source == LongitudinalPlanSource.cruise
    first = [a for t, _, a in log if t <= 0.1]
    assert max(first) <= np.interp(0.0, J_CRUISE_BP, J_CRUISE_VALS) * 0.1 + 1e-6
    by_one_second = [a for t, v, a in log if 0.9 <= t <= 1.0]
    v_at_one = [v for t, v, _ in log if 0.9 <= t <= 1.0][-1]
    assert max(by_one_second) >= 0.5 * get_max_accel(v_at_one)
    accels = [a for _, _, a in log]
    assert all(b >= a - 1e-6 for a, b in zip(accels, accels[1:], strict=False) if b < get_max_accel(0.0) - 1e-3)

  def test_lead_launch_starts_smooth_because_the_change_cost_stays_on(self):
    # behind a departing lead the MPC owns the launch; keeping the change cost through standstill softens its first step
    def launch(keep_cost):
      plant = Plant(speed=0.0, distance_lead=7.0, lead_relevancy=True)
      set_weights = plant.planner.mpc.set_weights
      plant.planner.mpc.set_weights = lambda prev_accel_constraint, **kwargs: set_weights(prev_accel_constraint and keep_cost, **kwargs)
      accels, sources = [], []
      while plant.current_time < 2.0:
        plant.step(v_lead=np.interp(plant.current_time, [0.5, 2.0], [0.0, 7.0]), v_cruise=20.0)
        accels.append(plant.acceleration)
        sources.append(plant.planner.mpc.source)
      return np.array(accels), sources

    smooth, sources = launch(True)
    stock_standstill, _ = launch(False)
    assert LongitudinalPlanSource.lead0 in sources
    assert np.max(np.diff(smooth)) < np.max(np.diff(stock_standstill))
    assert np.max(smooth) >= 0.8 * np.max(stock_standstill)

  def test_e2e_candidate_needs_a_valid_model(self):
    planner = LongitudinalPlanner(car.CarParams.new_message(openpilotLongitudinalControl=True, longitudinalActuatorDelay=0.5,
                                                            steerRatio=CP.steerRatio, wheelbase=CP.wheelbase))
    car_state = messaging.new_message('carState').carState
    car_state.vEgo = 10.0
    car_state.vCruise = 100.0
    model = messaging.new_message('modelV2').modelV2
    model.action.shouldStop = True
    model.action.desiredAcceleration = -3.0
    controls_state = messaging.new_message('controlsState').controlsState
    controls_state.longControlState = LongCtrlState.pid
    selfdrive_state = messaging.new_message('selfdriveState').selfdriveState
    selfdrive_state.enabled = True
    selfdrive_state.experimentalMode = True
    data = {'carState': car_state, 'modelV2': model, 'controlsState': controls_state, 'selfdriveState': selfdrive_state,
            'radarState': messaging.new_message('radarState').radarState, 'carControl': messaging.new_message('carControl').carControl,
            'vehicleParameters': messaging.new_message('vehicleParameters').vehicleParameters}
    for _ in range(5):
      planner.update(_PlantSubMaster(data, 0))
    assert planner.mpc.source == LongitudinalPlanSource.e2e
    assert planner.output_a_target == pytest.approx(-3.0)
    for _ in range(5):
      planner.update(_PlantSubMaster(data, 0, invalid=('modelV2',)))
    assert planner.mpc.source != LongitudinalPlanSource.e2e
    assert planner.output_a_target > -3.0

  def test_standstill_keeps_the_acceleration_change_cost(self):
    plant = Plant(speed=0.0, distance_lead=200.0, lead_relevancy=False)
    calls = []
    original = plant.planner.mpc.set_weights
    plant.planner.mpc.set_weights = lambda *args, **kwargs: (calls.append((args, kwargs)), original(*args, **kwargs))
    plant.step(v_cruise=50.0)
    assert plant.last_sm['carState'].standstill
    assert calls[-1][0][0] is True

  def test_comfort_only_for_ordinary_chill_cruise_with_a_healthy_radar(self):
    v_ego = 33.0
    v_cruise = v_ego + 5.0 * CV.MPH_TO_MS
    comfort_target = CRUISE_COMFORT_KP * (v_cruise - v_ego)
    for radar_valid, e2e, expected in ((True, False, comfort_target), (False, False, get_max_accel(v_ego)), (True, True, ACCEL_MAX)):
      plant = Plant(speed=v_ego, distance_lead=300.0, lead_relevancy=False, e2e=e2e)
      for _ in range(100):
        plant.step(v_cruise=v_cruise, radar_valid=radar_valid)
        plant.speed = v_ego  # hold speed so the settled cruise target is observable
      assert plant.planner.a_cruise == pytest.approx(expected, abs=0.02), (radar_valid, e2e)

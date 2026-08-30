import math
import numpy as np

from opendbc.car.interfaces import ACCEL_MAX
from opendbc.car.structs import car
from openpilot.cereal import log
import openpilot.cereal.messaging as messaging
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.force_stops import NO_CAP, ForceStopsResult
from openpilot.selfdrive.controls.lib.necessity_supervisor import LongitudinalPolicy
from openpilot.selfdrive.controls.lib.stop_landing import KISS_DECEL, LEAD_FULL_AUTHORITY, landing_bound
from openpilot.selfdrive.controls.lib.longitudinal_planner import (LAUNCH_MAX_ACCEL, LAUNCH_OPEN_LENGTH,
                                                                   A_CRUISE_MAX_LAUNCH, A_CRUISE_MAX_HIGH_SPEED, A_CRUISE_MAX_SPEED,
                                                                   A_CRUISE_MIN, CRUISE_COMFORT_KP, J_CRUISE_BP, J_CRUISE_VALS,
                                                                   LongitudinalPlanner, get_cruise_accel, get_cruise_comfort_accel,
                                                                   get_max_accel, get_max_accel_request, ordinary_cruise_comfort_enabled)
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant, _PlantSubMaster

LongitudinalPlanSource = log.LongitudinalPlan.LongitudinalPlanSource

CP = car.CarParams.new_message(steerRatio=17.94, wheelbase=2.9)  # Palisade


def settled_cruise_accel(v_cruise, v_ego, accel_coast=-0.3, comfort=True, e2e=False):
  target = 0.0
  for _ in range(200):
    target = get_cruise_accel(e2e, v_cruise, v_ego, target, DT_MDL, accel_coast, True, comfort)
  return target


class TestAccelEnvelope:
  def test_curve_endpoints_and_floor(self):
    assert math.isclose(get_max_accel_request(0.0), A_CRUISE_MAX_LAUNCH, rel_tol=1e-6, abs_tol=1e-9)
    assert math.isclose(get_max_accel_request(A_CRUISE_MAX_SPEED), A_CRUISE_MAX_HIGH_SPEED, rel_tol=1e-6, abs_tol=1e-9)
    assert math.isclose(get_max_accel_request(A_CRUISE_MAX_SPEED + 15.0), A_CRUISE_MAX_HIGH_SPEED, rel_tol=1e-6, abs_tol=1e-9)

  def test_curve_is_smooth_monotonic_and_convex(self):
    speeds = np.linspace(0.0, A_CRUISE_MAX_SPEED, 401)
    requests = np.array([get_max_accel_request(v) for v in speeds])
    assert np.all(np.diff(requests) <= 0.0)
    assert np.all(np.diff(requests, n=2) >= -1e-9)
    # the floor is reached with zero slope, not a corner
    assert abs(requests[-1] - requests[-2]) < 1e-4

  def test_deployed_platform_clamps_the_request(self):
    assert ACCEL_MAX <= A_CRUISE_MAX_LAUNCH
    assert math.isclose(get_max_accel(0.0), ACCEL_MAX, rel_tol=1e-6, abs_tol=1e-9)
    assert abs((get_max_accel(20.0)) - (1.025)) <= 1e-3
    assert abs((get_max_accel(30.0)) - (0.653)) <= 1e-3

  def test_jerk_schedule_bounds_the_first_step(self):
    v_ego = 75.0 * CV.MPH_TO_MS
    first = get_cruise_accel(False, v_ego + 5.0 * CV.MPH_TO_MS, v_ego, 0.0, DT_MDL, -0.3, True)
    assert math.isclose(first, np.interp(v_ego, J_CRUISE_BP, J_CRUISE_VALS) * DT_MDL, rel_tol=1e-6, abs_tol=1e-9)


class TestCruiseComfort:
  def test_five_mph_increase_at_highway_speed_is_proportional(self):
    v_ego = 74.5 * CV.MPH_TO_MS
    v_cruise = 79.5 * CV.MPH_TO_MS
    expected = CRUISE_COMFORT_KP * (v_cruise - v_ego)
    assert abs((expected) - (0.402)) <= 1e-3
    assert math.isclose(settled_cruise_accel(v_cruise, v_ego, accel_coast=-0.25), expected, rel_tol=1e-6, abs_tol=1e-9)

  def test_five_mph_reduction_coasts_instead_of_full_braking(self):
    v_ego = 79.6 * CV.MPH_TO_MS
    v_cruise = 74.6 * CV.MPH_TO_MS
    settled = settled_cruise_accel(v_cruise, v_ego, accel_coast=-0.39)
    assert abs((settled) - (-0.402)) <= 1e-3
    assert settled > A_CRUISE_MIN

  def test_reduction_follows_an_uphill_coast_but_not_a_downhill_push(self):
    v_ego = 80.0 * CV.MPH_TO_MS
    v_cruise = 75.0 * CV.MPH_TO_MS
    assert math.isclose(get_cruise_comfort_accel(v_cruise, v_ego, -0.6), -0.6, rel_tol=1e-6, abs_tol=1e-9)
    assert abs((get_cruise_comfort_accel(v_cruise, v_ego, 0.2)) - (-0.402)) <= 1e-3

  def test_small_corrections_taper_continuously(self):
    v_ego = 75.0 * CV.MPH_TO_MS
    half = get_cruise_comfort_accel(v_ego + 2.5 * CV.MPH_TO_MS, v_ego, -0.3)
    full = get_cruise_comfort_accel(v_ego + 5.0 * CV.MPH_TO_MS, v_ego, -0.3)
    assert math.isclose(half, full / 2.0, rel_tol=1e-6, abs_tol=1e-9)

  def test_large_errors_keep_the_envelope_and_braking_limit(self):
    v_ego = 75.0 * CV.MPH_TO_MS
    assert math.isclose(settled_cruise_accel(v_ego + 15.0 * CV.MPH_TO_MS, v_ego), get_max_accel(v_ego), rel_tol=1e-6, abs_tol=1e-9)
    assert math.isclose(settled_cruise_accel(v_ego - 15.0 * CV.MPH_TO_MS, v_ego), A_CRUISE_MIN, rel_tol=1e-6, abs_tol=1e-9)

  def test_low_speed_launch_keeps_legacy_authority(self):
    v_ego = 5.0
    v_cruise = v_ego + 5.0 * CV.MPH_TO_MS
    assert math.isclose(settled_cruise_accel(v_cruise, v_ego, comfort=True), settled_cruise_accel(v_cruise, v_ego, comfort=False), rel_tol=1e-6, abs_tol=1e-9)

  def test_comfort_is_only_for_ordinary_chill_cruise(self):
    assert ordinary_cruise_comfort_enabled(False, False, True)
    assert not ordinary_cruise_comfort_enabled(True, False, True)
    assert not ordinary_cruise_comfort_enabled(False, True, True)
    assert not ordinary_cruise_comfort_enabled(False, False, False)

  def test_coast_limit_still_applies_with_comfort(self):
    v_ego = 4.0
    v_cruise = v_ego + 5.0 * CV.MPH_TO_MS
    for comfort in (False, True):
      target = 0.0
      for _ in range(200):
        target = get_cruise_accel(False, v_cruise, v_ego, target, DT_MDL, -0.3, False, comfort)
      assert math.isclose(target, np.interp(v_ego, [2.5, 5.0], [get_max_accel(v_ego), -0.3]), rel_tol=1e-6, abs_tol=1e-9)

  def test_comfort_blends_in_between_8_and_15_mps(self):
    v_ego = 11.5
    v_cruise = v_ego + 5.0 * CV.MPH_TO_MS
    legacy = settled_cruise_accel(v_cruise, v_ego, comfort=False)
    full = np.clip(get_cruise_comfort_accel(v_cruise, v_ego, -0.3), A_CRUISE_MIN, get_max_accel(v_ego))
    assert math.isclose(settled_cruise_accel(v_cruise, v_ego), (legacy + full) / 2.0, rel_tol=1e-6, abs_tol=1e-9)


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
      plant.planner.mpc.set_weights = lambda prev_accel_constraint, *args: set_weights(prev_accel_constraint and keep_cost, *args)
      accels, sources = [], []
      while plant.current_time < 3.0:
        plant.step(v_lead=np.interp(plant.current_time, [0.5, 2.0], [0.0, 7.0]), v_cruise=20.0)
        accels.append(plant.acceleration)
        sources.append(plant.planner.mpc.source)
      return np.array(accels), sources

    smooth, sources = launch(True)
    stock_standstill, _ = launch(False)
    # smooth start: a gentler first step; quick growth: at least half of the uncosted peak within three seconds
    assert LongitudinalPlanSource.lead0 in sources
    assert np.max(np.diff(smooth)) < 0.5 * np.max(np.diff(stock_standstill))
    assert np.max(smooth) >= 0.5 * np.max(stock_standstill)

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
    assert math.isclose(planner.output_a_target, -3.0, rel_tol=1e-6, abs_tol=1e-9)
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
    e2e_target = min(v_cruise - v_ego, ACCEL_MAX)
    for radar_valid, e2e, expected in ((True, False, comfort_target), (False, False, get_max_accel(v_ego)), (True, True, e2e_target)):
      plant = Plant(speed=v_ego, distance_lead=300.0, lead_relevancy=False, e2e=e2e)
      for _ in range(100):
        plant.step(v_cruise=v_cruise, radar_valid=radar_valid)
        plant.speed = v_ego  # hold speed so the settled cruise target is observable
      assert abs((plant.planner.a_cruise) - (expected)) <= 0.02, (radar_valid, e2e)

  def test_fcw_comes_only_from_the_mpc_crash_counter(self):
    plant = Plant(speed=15.0, distance_lead=40.0, lead_relevancy=True)
    plant.planner.supervisor.update = lambda *args: LongitudinalPolicy(1.0, 0.0, True)
    plant.step(v_lead=15.0)
    assert plant.planner.supervisor.update().stand_down
    assert not plant.planner.fcw

  def test_a_force_stops_hold_forces_should_stop_and_caps_cruise(self):
    plant = Plant(speed=0.0, distance_lead=200.0, lead_relevancy=False, e2e=True)
    plant.planner.force_stops.update = lambda *args: ForceStopsResult(0.0, 0.0, True)
    plant.step(v_cruise=20.0)
    assert plant.planner.output_should_stop
    assert plant.planner.output_a_target <= 0.0

  def test_a_nonfinite_stop_point_never_reaches_the_mpc(self):
    plant = Plant(speed=10.0, distance_lead=200.0, lead_relevancy=False, e2e=True)
    plant.planner.force_stops.update = lambda *args: ForceStopsResult(NO_CAP, float('nan'), False)
    plant.step(v_cruise=20.0)
    assert plant.planner.mpc.source != LongitudinalPlanSource.stop


class TestStopLanding:
  def planner_and_data(self, v_ego, e2e_accel, lead_distance=None):
    planner = LongitudinalPlanner(car.CarParams.new_message(openpilotLongitudinalControl=True, longitudinalActuatorDelay=0.5,
                                                            steerRatio=CP.steerRatio, wheelbase=CP.wheelbase))
    car_state = messaging.new_message('carState').carState
    car_state.vEgo = v_ego
    car_state.vCruise = 100.0
    model = messaging.new_message('modelV2').modelV2
    model.action.shouldStop = True
    model.action.desiredAcceleration = e2e_accel
    controls_state = messaging.new_message('controlsState').controlsState
    controls_state.longControlState = LongCtrlState.pid
    selfdrive_state = messaging.new_message('selfdriveState').selfdriveState
    selfdrive_state.enabled = True
    selfdrive_state.experimentalMode = True
    radar = messaging.new_message('radarState').radarState
    if lead_distance is not None:
      radar.leadOne.present = True
      radar.leadOne.dRel = lead_distance
      radar.leadOne.modelProb = 1.0
    data = {'carState': car_state, 'modelV2': model, 'controlsState': controls_state, 'selfdriveState': selfdrive_state,
            'radarState': radar, 'carControl': messaging.new_message('carControl').carControl,
            'vehicleParameters': messaging.new_message('vehicleParameters').vehicleParameters}
    return planner, data

  def test_the_law_bounds_whichever_candidate_lands_the_stop(self):
    # the model calls a stop and asks for -3.0 at walking pace: e2e wins the arbitration and the landing law bounds it
    planner, data = self.planner_and_data(1.0, -3.0)
    for _ in range(5):
      planner.update(_PlantSubMaster(data, 0))
    assert planner.mpc.source == LongitudinalPlanSource.e2e
    assert math.isclose(planner.output_a_target, -landing_bound(1.0), rel_tol=1e-6, abs_tol=1e-9)
    assert planner.stop_landing.active

  def test_the_law_is_off_above_its_window_and_next_to_a_close_lead(self):
    planner, data = self.planner_and_data(10.0, -3.0)
    for _ in range(5):
      planner.update(_PlantSubMaster(data, 0))
    assert math.isclose(planner.output_a_target, -3.0, rel_tol=1e-6, abs_tol=1e-9)
    planner, data = self.planner_and_data(1.0, -3.0, lead_distance=LEAD_FULL_AUTHORITY - 1.0)
    for _ in range(5):
      planner.update(_PlantSubMaster(data, 0))
    assert math.isclose(planner.output_a_target, -3.0, rel_tol=1e-6, abs_tol=1e-9)
    assert not planner.stop_landing.active

  def test_a_lead_stop_lands_on_the_kiss_holds_its_stop_bit_and_launches_when_the_lead_leaves(self):
    # route 28 (2026-08-30): behind a stopped lead the MPC lets go of the brake by 0.2 m/s and hovers around zero; the
    # plan must not flicker between the floor and a throttle blip, the stop bit must hold, the wheels stop under the kiss
    plant = Plant(speed=6.0, distance_lead=60.0, lead_relevancy=True)
    log = []
    v_lead = 0.0
    while plant.current_time < 20.0:
      if plant.current_time > 14.0:
        v_lead = min(v_lead + 2.0 * DT_MDL, 5.0)         # the lead pulls away at 2 m/s^2 after 14 s
      plant.step(v_lead=v_lead, v_cruise=10.0)
      log.append((plant.current_time, plant.speed, float(plant.planner.output_a_target), bool(plant.planner.output_should_stop)))
    stopped = [i for i, (_, v, _, _) in enumerate(log) if v < 0.05]
    assert stopped, 'did not stop'
    i_stop = stopped[0]
    assert log[i_stop][0] < 14.0, 'stopped only after the lead left'
    last_rolling = [a for _, v, a, _ in log[:i_stop] if v > 0.05][-1]
    assert -0.3 <= last_rolling <= -KISS_DECEL + 1e-6, last_rolling                       # the wheels stop under a whisper
    tail = [a for t, _, a, _ in log[:i_stop] if t >= log[i_stop][0] - 1.0]
    assert max(tail) < 0.0, max(tail)                                                     # no throttle blip in the last second
    assert max(abs(b - a) for a, b in zip(tail, tail[1:], strict=False)) < 0.15             # and no square wave
    held = [(t, s) for t, v, _, s in log if log[i_stop][0] <= t <= 14.0]
    assert all(s for _, s in held), 'stop bit dropped while the lead stood still'
    launched = [t for t, _, a, s in log if t > 14.0 and a > 0.1 and not s]
    assert launched and launched[0] - 14.0 < 1.5, launched[:1]                             # the lead leaving releases the landing


class TestGreenLaunch:
  # route 28 t=2245 (2026-08-30): the model's path opened at the green 1.5-2 s before its shouldStop bit cleared, and its own
  # acceleration request stayed +0.05 m/s^2 the whole time; the e2e candidate won the min() and the car sat there
  def planner_and_data(self, e2e_accel=0.05, lead_distance=None):
    planner = LongitudinalPlanner(car.CarParams.new_message(openpilotLongitudinalControl=True, longitudinalActuatorDelay=0.5,
                                                            steerRatio=CP.steerRatio, wheelbase=CP.wheelbase))
    car_state = messaging.new_message('carState').carState
    car_state.vEgo = 0.0
    car_state.standstill = True
    car_state.vCruise = 100.0
    model = messaging.new_message('modelV2').modelV2
    model.action.shouldStop = True
    model.action.desiredAcceleration = e2e_accel
    controls_state = messaging.new_message('controlsState').controlsState
    controls_state.longControlState = LongCtrlState.stopping
    selfdrive_state = messaging.new_message('selfdriveState').selfdriveState
    selfdrive_state.enabled = True
    selfdrive_state.experimentalMode = True
    radar = messaging.new_message('radarState').radarState
    if lead_distance is not None:
      radar.leadOne.present = True
      radar.leadOne.dRel = lead_distance
      radar.leadOne.modelProb = 1.0
    data = {'carState': car_state, 'modelV2': model, 'controlsState': controls_state, 'selfdriveState': selfdrive_state,
            'radarState': radar, 'carControl': messaging.new_message('carControl').carControl,
            'vehicleParameters': messaging.new_message('vehicleParameters').vehicleParameters}
    return planner, data

  @staticmethod
  def set_path(model, length, terminal_speed):
    # the model's plan: a path of the given length, its speed plan creeping up to terminal_speed (below the assist's 2 m/s commit)
    n = ModelConstants.IDX_N
    t = np.array(ModelConstants.T_IDXS)
    model.position.x = [float(x) for x in np.linspace(0.0, length, n)]
    model.velocity.x = [float(v) for v in np.linspace(0.0, terminal_speed, n)]
    model.acceleration.x = [0.0] * n
    model.position.y = [0.0] * n
    _ = t

  def run(self, planner, data, seconds):
    targets = []
    for _ in range(round(seconds / DT_MDL)):
      planner.update(_PlantSubMaster(data, 0))
      targets.append((float(planner.output_a_target), bool(planner.output_should_stop)))
    return targets

  def test_an_open_path_launches_on_the_cruise_ramp_when_the_models_own_plan_has_not_committed(self):
    planner, data = self.planner_and_data()
    self.set_path(data['modelV2'], 5.0, 0.0)
    held = self.run(planner, data, 1.0)
    assert all(stop for _, stop in held) and max(a for a, _ in held) <= 0.1
    # the green: the path opens, the model's bit and its request lag
    self.set_path(data['modelV2'], LAUNCH_OPEN_LENGTH * 2.0, 1.0)
    launch = self.run(planner, data, 1.5)
    released = [i for i, (_, stop) in enumerate(launch) if not stop]
    assert released and released[0] * DT_MDL < 0.8, released[:1]                       # the stop bit clears on the open path
    peak = max(a for a, _ in launch)
    assert 1.0 <= peak <= LAUNCH_MAX_ACCEL + 1e-6, peak                                  # and the launch ramps under the cap ...
    assert launch[released[0] + round(0.5 / DT_MDL)][0] >= 0.8                            # ... within half a second of the release

  def test_the_boost_never_overrides_the_models_own_braking_or_a_lead(self):
    planner, data = self.planner_and_data(e2e_accel=-0.3)
    self.set_path(data['modelV2'], 5.0, 0.0)
    self.run(planner, data, 1.0)
    self.set_path(data['modelV2'], LAUNCH_OPEN_LENGTH * 2.0, 1.0)
    # the model keeps braking: no launch (the landing corridor pins a standing car at its kiss, not at the model's -0.3)
    assert max(a for a, _ in self.run(planner, data, 1.5)) < 0.0
    planner, data = self.planner_and_data(lead_distance=6.0)
    self.set_path(data['modelV2'], 5.0, 0.0)
    self.run(planner, data, 1.0)
    self.set_path(data['modelV2'], LAUNCH_OPEN_LENGTH * 2.0, 1.0)
    assert max(a for a, _ in self.run(planner, data, 1.5)) <= 0.1

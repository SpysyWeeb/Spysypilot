import math
import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.cereal import log
import openpilot.cereal.messaging as messaging
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib import longitudinal_planner
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc
from openpilot.selfdrive.controls.lib.drive_helpers import should_stop
from openpilot.selfdrive.controls.lib.force_stops import NO_CAP, PROFILE_JERK, RELEASE_OPEN_FRAMES, ForceStopsResult
from openpilot.selfdrive.controls.lib.model_curve_speed import REGIME_BRAKE, REGIME_FREE, CurveResult
from openpilot.selfdrive.controls.lib.necessity_supervisor import LEAD_DEPARTURE_CANCEL, LEAD_DEPARTURE_CONFIRM, LongitudinalPolicy
from openpilot.selfdrive.controls.lib.stop_helpers import PATH_OPEN_LENGTH, STOP_PREDICTION_HORIZON_S
from openpilot.selfdrive.controls.lib.stop_landing import KISS_DECEL, LEAD_MIN_GAP_BUDGET, STANDSTILL_SPEED
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longitudinal_planner import (LAUNCH_MAX_ACCEL, LAUNCH_OPEN_LENGTH,
                                                                   A_CRUISE_MAX_LAUNCH, A_CRUISE_MAX_HIGH_SPEED, A_CRUISE_MAX_SPEED,
                                                                   A_CRUISE_MIN, CRUISE_COMFORT_KP, J_CRUISE_BP, J_CRUISE_VALS,
                                                                   LongitudinalPlanner, get_cruise_accel, get_cruise_comfort_accel,
                                                                   get_max_accel, get_max_accel_request, ordinary_cruise_comfort_enabled)
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import ActuatorLag, Plant, _PlantSubMaster, palisade_car_params
from openpilot.selfdrive.test.longitudinal_maneuvers.test_longitudinal import BRAKE_SHORTFALL, PLAN_JERK_MAX

LongitudinalPlanSource = log.LongitudinalPlan.LongitudinalPlanSource

PALISADE_CP = palisade_car_params()


@pytest.fixture(params=[2.0, 4.0])
def accel_max(request, monkeypatch):
  # stock opendbc's pin and the fork's; the planner and the MPC bind ACCEL_MAX at import
  monkeypatch.setattr(longitudinal_planner, 'ACCEL_MAX', request.param)
  monkeypatch.setattr(long_mpc, 'ACCEL_MAX', request.param)
  return request.param


def frame(v_ego, a_ego=0.0, experimental=False, e2e_accel=0.0, should_stop=False, path=None, lead=None, v_cruise=100.0 * CV.KPH_TO_MS,
          engaged=True, standstill=None, forecast=None, gas_prob=None):
  # one model frame of real messages: path = (end, terminal speed) of the model's plan, lead = (dRel, vLead, aLead) on radar,
  # engaged = openpilot controls the speed (not the driver's foot), standstill = the car's own flag (default: the car is at rest),
  # forecast = the acceleration the model sees the lead hold when it differs from the radar's, gas_prob = the model's
  # gas-press probability
  car_state = messaging.new_message('carState').carState
  car_state.vEgo = v_ego
  car_state.aEgo = a_ego
  car_state.vCruise = v_cruise * CV.MS_TO_KPH
  car_state.standstill = v_ego < 0.01 if standstill is None else standstill
  model = messaging.new_message('modelV2').modelV2
  if path is not None:
    n = ModelConstants.IDX_N
    position = log.XYZTData.new_message()
    position.x = [float(x) for x in np.linspace(0.0, path[0], n)]
    model.position = position
    velocity = log.XYZTData.new_message()
    velocity.x = [float(v) for v in np.linspace(v_ego, path[1], n)]
    model.velocity = velocity
  model.action.shouldStop = should_stop
  model.action.desiredAcceleration = e2e_accel
  if gas_prob is not None:
    model.meta.disengagePredictions.gasPressProbs = [float(gas_prob)] * 6
  radar = messaging.new_message('radarState').radarState
  if lead is not None:
    radar.leadOne.present = True
    radar.leadOne.dRel, radar.leadOne.vLead, radar.leadOne.aLeadK = (float(x) for x in lead)
    radar.leadOne.vLeadK = float(lead[1])
    radar.leadOne.aLeadTau = 1.5
    radar.leadOne.modelProb = 1.0
    radar.leadOne.radar = True
    # the model sees the same car hold its acceleration until it stops
    d, v, a = lead
    a = a if forecast is None else forecast
    t = np.array(ModelConstants.LEAD_T_IDXS)
    t_moving = np.minimum(t, -v / a) if a < 0.0 else t
    model_lead = log.ModelDataV2.LeadDataV3.new_message()
    model_lead.prob = 1.0
    model_lead.t = t.tolist()
    model_lead.x = (d + v * t_moving + 0.5 * a * t_moving ** 2).tolist()
    model_lead.v = np.maximum(v + a * t_moving, 0.0).tolist()
    model_lead.xStd = [1.0] * len(t)
    model_lead.vStd = [0.5] * len(t)
    model_lead.y = [0.0] * len(t)
    model_lead.yStd = [1.0] * len(t)
    model.leadsV3 = [model_lead]
  controls_state = messaging.new_message('controlsState').controlsState
  controls_state.longControlState = LongCtrlState.pid if engaged else LongCtrlState.off
  selfdrive_state = messaging.new_message('selfdriveState').selfdriveState
  selfdrive_state.enabled = True
  selfdrive_state.experimentalMode = experimental
  data = {'carState': car_state, 'modelV2': model, 'controlsState': controls_state, 'selfdriveState': selfdrive_state,
          'radarState': radar, 'carControl': messaging.new_message('carControl').carControl,
          'vehicleParameters': messaging.new_message('vehicleParameters').vehicleParameters,
          'radarTracks': messaging.new_message('radarTracks').radarTracks}
  return _PlantSubMaster(data, 0)


class IdealCar:
  # a car that applies each published target at once
  def __init__(self, v_ego, planner=None):
    self.planner = planner or LongitudinalPlanner(PALISADE_CP, init_v=v_ego)
    self.v, self.a, self.x = v_ego, 0.0, 0.0

  def step(self, **kwargs):
    self.planner.update(frame(self.v, self.a, **kwargs))
    self.a = float(self.planner.output_a_target)
    self.v = max(self.v + self.a * DT_MDL, 0.0)
    self.x += self.v * DT_MDL
    return self.a


def j_cruise_step(v_ego):
  return float(np.interp(v_ego, J_CRUISE_BP, J_CRUISE_VALS)) * DT_MDL


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


class TestResetClearsTheLaunchState:
  # a reset cleared anticipating_prev but not anticipating, so the frame after a re-engage at an open light read as a
  # fresh green-light opening and fired the launch edge that releases the landing law (2026-09-06)
  def _planner(self):
    planner = LongitudinalPlanner(PALISADE_CP, init_v=0.0)
    planner.anticipating = True
    planner.anticipating_prev = True
    planner.launch_armed = True
    planner.launch_open.x = 1.0
    return planner

  def test_a_reset_clears_the_whole_launch_state_not_just_its_edge_memory(self):
    plant = Plant(speed=0.0, distance_lead=200.0, lead_relevancy=False, enabled=False)
    plant.planner = self._planner()
    plant.step(v_cruise=50.0)
    assert not plant.planner.anticipating
    assert not plant.planner.anticipating_prev
    assert plant.planner.launch_open.x == 0.0

  def test_a_reset_while_rolling_below_the_disarm_speed_also_disarms_the_launch(self):
    # launch_armed is sticky between standstill and LAUNCH_DISARM_SPEED, so a reset in that band has to clear it;
    # at a standstill it is re-armed on the same frame by design
    plant = Plant(speed=1.0, distance_lead=200.0, lead_relevancy=False, enabled=False)
    plant.planner = self._planner()
    plant.step(v_cruise=50.0)
    assert not plant.planner.launch_armed

  def test_a_reset_does_not_fire_a_launch_edge(self):
    # the edge is `anticipating and not anticipating_prev`; with both cleared together it cannot fire on the next frame
    plant = Plant(speed=0.0, distance_lead=200.0, lead_relevancy=False, enabled=False)
    plant.planner = self._planner()
    plant.step(v_cruise=50.0)
    assert not (plant.planner.anticipating and not plant.planner.anticipating_prev)


class TestPlannerCruise:
  def run_plant(self, seconds, **kwargs):
    plant = Plant(**kwargs)
    log = []
    while plant.current_time < seconds:
      plant.step(v_cruise=50.0)
      log.append((plant.current_time, plant.speed, float(plant.planner.output_a_target)))
    return plant, log

  def test_standstill_launch_starts_smooth_and_grows_quickly(self):
    # owner acceptance: the plan starts smooth, then asks at least half the envelope within a second; cruise owns this launch
    plant, log = self.run_plant(1.2, speed=0.0, distance_lead=200.0, lead_relevancy=False, CP=PALISADE_CP)
    assert plant.planner.plan_source == LongitudinalPlanSource.cruise
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
        sources.append(plant.planner.plan_source)
      return np.array(accels), sources

    smooth, sources = launch(True)
    stock_standstill, _ = launch(False)
    # smooth start: a gentler first step; quick growth: at least half of the uncosted peak within three seconds
    assert LongitudinalPlanSource.lead0 in sources
    assert np.max(np.diff(smooth)) < 0.5 * np.max(np.diff(stock_standstill))
    assert np.max(smooth) >= 0.5 * np.max(stock_standstill)

  def test_e2e_candidate_needs_a_valid_model(self):
    planner = LongitudinalPlanner(PALISADE_CP)
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
            'vehicleParameters': messaging.new_message('vehicleParameters').vehicleParameters,
            'radarTracks': messaging.new_message('radarTracks').radarTracks}
    for _ in range(5):
      planner.update(_PlantSubMaster(data, 0))
    assert planner.plan_source == LongitudinalPlanSource.e2e
    assert math.isclose(planner.output_a_target, -3.0, rel_tol=1e-6, abs_tol=1e-9)
    for _ in range(5):
      planner.update(_PlantSubMaster(data, 0, invalid=('modelV2',)))
    assert planner.plan_source != LongitudinalPlanSource.e2e
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
    plant.planner.supervisor.update = lambda *args: LongitudinalPolicy(1.0, 0.0)   # a stand-down: stock policy, and nothing else
    plant.step(v_lead=15.0)
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


class TestThrottleGate:
  # Chill's throttle gate caps the cruise candidate at the coast above 5 m/s when the model expects no gas press; it decides on the
  # probability's recent mean, not on each frame's noise
  def test_the_gate_decides_on_the_smoothed_probability(self):
    # a probability whose mean sits clearly above the threshold, jittering by the field's frame-to-frame spread
    rng = np.random.default_rng(0)
    plant = Plant(speed=15.0, distance_lead=300.0, lead_relevancy=False)
    toggles, prev = 0, True
    for _ in range(round(10.0 / DT_MDL)):
      plant.step(v_cruise=15.0, prob_throttle=float(np.clip(0.55 + rng.normal(0.0, 0.078), 0.0, 1.0)))
      toggles += int(plant.planner.allow_throttle != prev)
      prev = plant.planner.allow_throttle
    assert toggles <= 2, toggles

  def test_a_driver_hand_back_restarts_the_gate_from_the_models_reading(self):
    planner = LongitudinalPlanner(PALISADE_CP)
    for _ in range(round(2.0 / DT_MDL)):
      planner.update(frame(20.0, gas_prob=0.9))
    assert planner.allow_throttle
    planner.update(frame(20.0, gas_prob=0.1, engaged=False))
    planner.update(frame(20.0, gas_prob=0.1))
    assert not planner.allow_throttle

  def test_a_nonfinite_probability_does_not_stick_in_the_gate(self):
    planner = LongitudinalPlanner(PALISADE_CP)
    for gas_prob in (0.9, math.nan, 0.9, 0.9):
      planner.update(frame(20.0, gas_prob=gas_prob))
      assert planner.allow_throttle, gas_prob


class TestCommittedProfileArbitration:
  def test_e2e_joins_only_when_clearly_more_urgent_than_the_committed_profile(self):
    # the model's late ramp would overtake the flat profile through min()
    from openpilot.selfdrive.controls.lib.longitudinal_planner import E2E_STOP_MARGIN
    for e2e, expected_winner in ((-1.0 - E2E_STOP_MARGIN + 0.1, 'profile'), (-1.0 - E2E_STOP_MARGIN - 0.3, 'e2e')):
      planner, data = TestStopLanding().planner_and_data(10.0, e2e)
      planner.force_stops.update = lambda *args: ForceStopsResult(NO_CAP, None, False, -1.0)
      for _ in range(5):
        planner.update(_PlantSubMaster(data, 0))
      assert planner.plan_winner == expected_winner, (e2e, planner.plan_winner)

  @pytest.mark.parametrize('e2e_jerk', [3.0, 4.0])
  def test_a_commit_does_not_evict_the_model_request_it_joins(self, e2e_jerk):
    # the model ramps its request deeper from the red's strict onset through the commit. The profile joins from the published
    # target, the request's own value, so it sits inside the margin of it: the request keeps its place while it leads, and the
    # output brakes no faster than the request ramps
    car = IdealCar(13.0)
    ramp, steps = 0.0, []
    while len(steps) < round(3.0 / DT_MDL) and car.x < 110.0:
      d = 110.0 - car.x
      if ramp or d <= car.v * STOP_PREDICTION_HORIZON_S:
        ramp += e2e_jerk * DT_MDL
      told = car.a
      car.step(experimental=True, e2e_accel=max(-ramp, -2.6), path=(d, 0.0))
      if steps or car.planner.force_stops.forcing:
        steps.append(car.a - told)
    assert len(steps) == round(3.0 / DT_MDL) and min(steps) >= -e2e_jerk * DT_MDL - 1e-6, min(steps)


class TestStopLanding:
  def planner_and_data(self, v_ego, e2e_accel, lead_distance=None, a_ego=0.0, stop_ahead=None):
    # stop_ahead: the model's plan is a constant-deceleration stop that far ahead
    planner = LongitudinalPlanner(PALISADE_CP)
    car_state = messaging.new_message('carState').carState
    car_state.vEgo = v_ego
    car_state.aEgo = a_ego
    car_state.vCruise = 100.0
    model = messaging.new_message('modelV2').modelV2
    model.action.shouldStop = True
    model.action.desiredAcceleration = e2e_accel
    if stop_ahead is not None:
      t = np.array(ModelConstants.T_IDXS)
      decel = v_ego ** 2 / (2.0 * stop_ahead)
      t_moving = np.minimum(t, v_ego / decel)
      model.position.x = (v_ego * t_moving - 0.5 * decel * t_moving ** 2).tolist()
      model.velocity.x = (v_ego - decel * t_moving).tolist()
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
            'vehicleParameters': messaging.new_message('vehicleParameters').vehicleParameters,
            'radarTracks': messaging.new_message('radarTracks').radarTracks}
    return planner, data

  def test_the_law_bounds_whichever_candidate_lands_the_stop(self):
    # the model calls a stop 2 m ahead and asks for -3.0 at walking pace: e2e wins the arbitration and the landing law bounds it
    planner, data = self.planner_and_data(1.0, -3.0, a_ego=-1.0, stop_ahead=2.0)
    for _ in range(5):
      planner.update(_PlantSubMaster(data, 0))
    assert planner.plan_source == LongitudinalPlanSource.e2e
    assert math.isclose(planner.output_a_target, -planner.stop_landing.bound(1.0), rel_tol=1e-6, abs_tol=1e-9)
    assert planner.stop_landing.landing

  def test_the_law_is_off_above_its_window_and_lets_a_close_lead_through(self):
    planner, data = self.planner_and_data(10.0, -3.0)
    for _ in range(5):
      planner.update(_PlantSubMaster(data, 0))
    assert math.isclose(planner.output_a_target, -3.0, rel_tol=1e-6, abs_tol=1e-9)
    # a stopped car 4 m ahead at walking pace: the braking that stops the car short of it passes the corridor, and nothing beyond it
    planner, data = self.planner_and_data(1.0, -3.0, lead_distance=4.0, a_ego=-1.0)
    for _ in range(5):
      planner.update(_PlantSubMaster(data, 0))
    decel = -float(planner.output_a_target)
    assert decel > planner.stop_landing.bound(1.0)
    assert 1.0 ** 2 / (2.0 * decel) < 4.0
    assert decel < 3.0

  def test_a_lead_stop_lands_on_the_kiss_holds_its_stop_bit_and_launches_when_the_lead_leaves(self):
    # behind a stopped lead the MPC lets go of the brake by 0.2 m/s and hovers around zero; the plan must not flicker between
    # the floor and a throttle blip, the stop bit must hold, the wheels stop under the kiss
    plant = Plant(speed=6.0, distance_lead=60.0, lead_relevancy=True)
    log = []
    v_lead = 0.0
    while plant.current_time < 24.0:
      if plant.current_time > 16.0:
        v_lead = min(v_lead + 2.0 * DT_MDL, 5.0)         # the lead pulls away at 2 m/s^2 after 14 s
      plant.step(v_lead=v_lead, v_cruise=10.0)
      log.append((plant.current_time, plant.speed, float(plant.planner.output_a_target), bool(plant.planner.output_should_stop)))
    stopped = [i for i, (_, v, _, _) in enumerate(log) if v < 0.05]
    assert stopped, 'did not stop'
    i_stop = stopped[0]
    assert log[i_stop][0] < 16.0, 'stopped only after the lead left'
    last_rolling = [a for _, v, a, _ in log[:i_stop] if v > 0.05][-1]
    assert -0.3 <= last_rolling <= -KISS_DECEL + 1e-6, last_rolling                       # the wheels stop under a whisper
    tail = [a for t, _, a, _ in log[:i_stop] if t >= log[i_stop][0] - 1.0]
    assert max(tail) < 0.0, max(tail)                                                     # no throttle blip in the last second
    assert max(abs(b - a) for a, b in zip(tail, tail[1:], strict=False)) < 0.15             # and no square wave
    held = [(t, s) for t, v, _, s in log if log[i_stop][0] <= t <= 16.0]
    assert all(s for _, s in held), 'stop bit dropped while the lead stood still'
    launched = [t for t, _, a, s in log if t > 16.0 and a > 0.1 and not s]
    assert launched and launched[0] - 16.0 < 1.5, launched[:1]                             # the lead leaving releases the landing


class TestGreenLaunch:
  # route 28 t=2245 (2026-08-30): the model's path opened at the green 1.5-2 s before its shouldStop bit cleared, and its own
  # acceleration request stayed +0.05 m/s^2 the whole time; the e2e candidate won the min() and the car sat there
  def planner_and_data(self, e2e_accel=0.05, lead_distance=None):
    planner = LongitudinalPlanner(PALISADE_CP)
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
            'vehicleParameters': messaging.new_message('vehicleParameters').vehicleParameters,
            'radarTracks': messaging.new_message('radarTracks').radarTracks}
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

  @staticmethod
  def launch_frame(v_ego, a_ego, experimental, commits):
    # the green after a hold: either the model's own plan commits to the launch (its bit clear, 1.2 m/s^2 from rest, above the
    # assist's 2 m/s at 3.5 s) or only its path opens while its bit and request lag (the anticipation's cruise ramp)
    if commits:
      sm = frame(v_ego, a_ego, experimental=experimental, e2e_accel=0.3, path=(40.0, 8.0), v_cruise=13.0)
      t = np.array(ModelConstants.T_IDXS)
      velocity = np.minimum(v_ego + 1.2 * t, 8.0)
      model = sm['modelV2']
      model.velocity.x = velocity.tolist()
      acceleration = log.XYZTData.new_message()
      acceleration.x = np.where(velocity < 8.0, 1.2, 0.0).tolist()
      model.acceleration = acceleration
      return sm
    return frame(v_ego, a_ego, experimental=experimental, e2e_accel=0.05, should_stop=True, path=(40.0, 1.0), v_cruise=13.0)

  def test_a_mode_exit_during_the_launch_hands_over_from_the_output(self):
    # the assist raises the e2e candidate off the line; a mode exit mid-launch takes e2e out of min(), and cruise, which slewed
    # toward ACCEL_MAX while e2e led, carries the car from the published target at the cruise jerk
    for commits in (True, False):
      for exit_speed in (0.3, 1.2, 1.9):
        planner = LongitudinalPlanner(PALISADE_CP)
        v, a_car, out = 0.0, 0.0, 0.0
        for _ in range(round(2.0 / DT_MDL)):
          # held at the line by the model's own stop bit on a 5 m path stub: a shorter one is a stop endpoint that Force Stops
          # holds on, and a green releases that hold through Force Stops, not through the assist
          planner.update(frame(0.0, 0.0, experimental=True, e2e_accel=0.05, should_stop=True, path=(5.0, 0.0), v_cruise=13.0))
        assert planner.output_should_stop and not planner.force_stops.holding
        experimental, previous, exit_frame = True, float(planner.output_a_target), 0
        for _ in range(round(6.0 / DT_MDL)):
          if experimental and v >= exit_speed:
            assert planner.plan_winner == 'e2e' and previous > 0.0, (commits, exit_speed, planner.plan_winner, previous)
            experimental = False
          bound = j_cruise_step(v)
          planner.update(self.launch_frame(v, a_car, experimental, commits))
          out = float(planner.output_a_target)
          if not experimental:
            if exit_frame == 0:
              assert planner.plan_winner == 'cruise', (commits, exit_speed, planner.plan_winner)
            assert out - previous <= bound + 1e-6, (commits, exit_speed, exit_frame, previous, out, planner.plan_winner)
            exit_frame += 1
          previous = out
          # a first order car that stays held while the stop bit is set at rest
          a_car += (out - a_car) * DT_MDL / 0.3
          if v <= 0.0 and (a_car < 0.0 or planner.output_should_stop):
            a_car = 0.0
          v = max(v + a_car * DT_MDL, 0.0)
        assert not experimental and out > 0.0, (commits, exit_speed, v)


class TestStopBit:
  # the wheels stop under the landing's kiss: stock LongControl's stopping state ramps toward stopAccel and ignores the plan, so
  # the stop bit waits for the standstill speed while the landing lands a rolling car
  def test_the_kiss_lands_a_rolling_car_and_the_stop_bit_rises_at_standstill(self):
    planner, long_control = LongitudinalPlanner(PALISADE_CP), LongControl(PALISADE_CP)
    v, a, x = 6.0, 0.0, 0.0
    car_state = messaging.new_message('carState').carState
    rolling, wheel_stop, bit_at = [], None, None
    for _ in range(round(20.0 / DT_MDL)):
      planner.update(frame(v, a, lead=(30.0 - x, 0.0, 0.0), v_cruise=10.0))
      if planner.output_should_stop and bit_at is None:
        bit_at = v
      for _ in range(round(DT_MDL / DT_CTRL)):
        car_state.vEgo, car_state.aEgo = v, a
        a = float(long_control.update(True, car_state, float(planner.output_a_target), planner.output_should_stop, (ACCEL_MIN, ACCEL_MAX)))
        if planner.stop_landing.landing and v > STANDSTILL_SPEED:
          rolling.append((long_control.long_control_state, a, float(planner.output_a_target)))
        if v >= 0.15:
          wheel_stop = a
        v = max(v + a * DT_CTRL, 0.0)
        x += v * DT_CTRL
    assert rolling and all(state == LongCtrlState.pid and out == target for state, out, target in rolling)
    assert bit_at is not None and bit_at <= STANDSTILL_SPEED, bit_at
    assert math.isclose(wheel_stop, -KISS_DECEL, rel_tol=1e-6, abs_tol=1e-9), wheel_stop
    assert v == 0.0 and planner.output_should_stop and long_control.long_control_state == LongCtrlState.stopping

  def test_a_target_nearer_than_the_kiss_can_stop_short_of_takes_the_stop_bit_at_once(self):
    # creeping inside the landing below the stop bit's own speed, a stopped target appears 0.3 m ahead (a cut-in, a pedestrian): that
    # is not the kiss's stop, so the bit rises on that frame and LongControl's stopping ramp brakes, and a frame the radar loses the
    # target does not hand the car back. The car answers through its identified actuator and brakes BRAKE_SHORTFALL less than asked
    planner, long_control, actuator = LongitudinalPlanner(PALISADE_CP), LongControl(PALISADE_CP), ActuatorLag()
    car_state = messaging.new_message('carState').carState
    v, a, x, target, bits, states = 6.0, 0.0, 0.0, None, [], []
    for _ in range(round(20.0 / DT_MDL)):
      if target is None and planner.stop_landing.landing and STANDSTILL_SPEED < v < 0.3:
        target = x + 0.3
      lead = (30.0 - x, 0.0, 0.0) if target is None else None if len(bits) == 2 else (target - x, 0.0, 0.0)
      planner.update(frame(v, a, lead=lead, v_cruise=10.0))
      if target is not None and v > 0.0:
        bits.append(planner.output_should_stop)
      for _ in range(round(DT_MDL / DT_CTRL)):
        car_state.vEgo, car_state.aEgo = v, a
        command = float(long_control.update(True, car_state, float(planner.output_a_target), planner.output_should_stop, (ACCEL_MIN, ACCEL_MAX)))
        if target is not None and v > 0.0:
          states.append(long_control.long_control_state)
        a = actuator.update(command, v) + (BRAKE_SHORTFALL if v > 0.0 else 0.0)
        v = max(v + a * DT_CTRL, 0.0)
        x += v * DT_CTRL
    assert len(bits) > 2 and all(bits), bits
    assert all(state == LongCtrlState.stopping for state in states)
    assert v == 0.0 and target - x > 0.0, target - x

  def test_a_speed_reading_after_the_stop_keeps_the_stop_bit_until_the_car_rolls_again(self):
    # at rest in the landing, one frame of wheel speed above the standstill speed (a body rebound, a wheel-speed tick) is not a
    # car rolling: the bit stays and LongControl keeps ramping. A car that rolls again at walking pace is the kiss's to land again
    planner, long_control = LongitudinalPlanner(PALISADE_CP), LongControl(PALISADE_CP)
    car_state = messaging.new_message('carState').carState
    v, a, x, at_rest = 6.0, 0.0, 0.0, 0
    while at_rest < 10:
      planner.update(frame(v, a, lead=(30.0 - x, 0.0, 0.0), v_cruise=10.0))
      for _ in range(round(DT_MDL / DT_CTRL)):
        car_state.vEgo, car_state.aEgo = v, a
        a = float(long_control.update(True, car_state, float(planner.output_a_target), planner.output_should_stop, (ACCEL_MIN, ACCEL_MAX)))
        v = max(v + a * DT_CTRL, 0.0)
        x += v * DT_CTRL
      at_rest = at_rest + 1 if v == 0.0 else 0
    assert planner.stop_landing.landing and long_control.long_control_state == LongCtrlState.stopping

    def reading(v_ego):
      planner.update(frame(v_ego, 0.0, lead=(30.0 - x, 0.0, 0.0), v_cruise=10.0))
      assert planner.stop_landing.landing, v_ego
      return planner.output_should_stop

    commanded = long_control.last_output_accel
    assert reading(STANDSTILL_SPEED + 0.02)
    car_state.vEgo, car_state.aEgo = STANDSTILL_SPEED + 0.02, 0.0
    command = float(long_control.update(True, car_state, float(planner.output_a_target), planner.output_should_stop, (ACCEL_MIN, ACCEL_MAX)))
    assert long_control.long_control_state == LongCtrlState.stopping and command <= commanded
    assert reading(0.0)
    assert not any(reading(v_ego) for v_ego in (0.35, 0.25, 0.15))
    assert reading(STANDSTILL_SPEED - 0.02)


class TestStandstillRelease:
  # once the MPC holds the car at rest for a lead, the lead-departure pre-release is the one owner of the release: the MPC's plan
  # drifting above the stop bit's threshold does not let the car go, and a release the pre-release takes back holds it again
  def rest_behind(self, v_ego, gap):
    # the car rolls up to a standing lead and comes to rest; returns its planner and the gap
    car = IdealCar(v_ego)
    while car.v >= 0.01:
      car.step(lead=(gap - car.x, 0.0, 0.0), v_cruise=10.0)
    return car.planner, gap - car.x

  def inch(self, planner, d, to):
    # the lead inches forward below any departure evidence while the car stands; returns the new gap and every frame's bit and MPC
    frames = []
    while d < to:
      d += 0.2 * DT_MDL
      planner.update(frame(0.0, lead=(d, 0.2, 0.0), v_cruise=10.0))
      frames.append((planner.output_should_stop, planner.mpc_a_target))
    return d, frames

  def test_the_plans_hover_at_rest_does_not_release_the_stop_bit(self):
    # rolling slowly up to the lead, the MPC's own plan stops the car and no landing starts
    planner, d = self.rest_behind(0.3, 7.5)
    assert not planner.stop_landing.landing and planner.output_should_stop
    d, frames = self.inch(planner, d, 9.0)
    assert not should_stop(0.0, max(mpc for _, mpc in frames)), max(mpc for _, mpc in frames)
    assert all(bit for bit, _ in frames)
    planner.update(frame(0.0, lead=(d, 1.0, 1.0), v_cruise=10.0))
    assert not planner.output_should_stop

  def test_a_lead_the_car_did_not_stop_for_does_not_hold_it(self):
    # engaged at rest with a stationary lead well beyond the stop distance: the MPC does not stop the car for it, and it goes
    planner = LongitudinalPlanner(PALISADE_CP)
    planner.update(frame(0.0, lead=(20.0, 0.0, 0.0), v_cruise=10.0, engaged=False))
    for _ in range(round(1.0 / DT_MDL)):
      planner.update(frame(0.0, lead=(20.0, 0.0, 0.0), v_cruise=10.0))
    assert not planner.output_should_stop and planner.output_a_target > 0.5
    # a Force Stops hold at a red line with a stationary car well beyond it: at the green the car pulls up behind that car
    car = IdealCar(8.0)
    stop = {'x': 25.0}

    def force_stops(*args):
      if stop['x'] is None:
        return ForceStopsResult()
      return ForceStopsResult(NO_CAP, stop['x'] - car.x, car.v < 0.01, None)
    car.planner.force_stops.update = force_stops
    while car.v >= 0.01:
      car.step(lead=(60.0 - car.x, 0.0, 0.0), v_cruise=10.0)
    for _ in range(round(1.0 / DT_MDL)):
      car.planner.update(frame(0.0, lead=(60.0 - car.x, 0.0, 0.0), v_cruise=10.0))
    assert car.planner.output_should_stop
    stop['x'] = None
    released = []
    for _ in range(round(1.0 / DT_MDL)):
      car.planner.update(frame(0.0, lead=(60.0 - car.x, 0.0, 0.0), v_cruise=10.0))
      released.append(not car.planner.output_should_stop)
    assert all(released[1:]), released

  def test_a_measured_departure_counts_only_on_the_track_the_model_confirms(self):
    planner, d = self.rest_behind(1.0, 12.0)

    def step(v_lead, track_id, model_prob=1.0):
      sm = frame(0.0, lead=(d, v_lead, 0.0), v_cruise=10.0)
      sm['radarState'].leadOne.radarTrackId = track_id
      sm['radarState'].leadOne.modelProb = model_prob
      planner.update(sm)
      return planner.output_should_stop

    for _ in range(round(1.0 / DT_MDL)):
      assert step(0.0, 7)
    # radard's low-speed override: a return of the standing car that the model does not confirm reads 0.3 m/s
    assert all(step(0.3, 8, model_prob=0.0) for _ in range(round(1.0 / DT_MDL)))
    # the model-confirmed track reading the same speed, from the frame after it takes the slot back, is the lead leaving
    assert all(step(0.3, 7) for _ in range(round(LEAD_DEPARTURE_CONFIRM / DT_MDL)))
    assert not step(0.3, 7)

  @pytest.mark.parametrize('moved', [0.0, 0.2, 0.5])
  def test_a_release_taken_back_at_rest_holds_the_car_again_only_if_the_lead_has_not_moved(self, moved):
    # the kiss lands the car behind a stopped lead, which then inches forward below any departure evidence, so the MPC comes to
    # plan toward the gap and hover above the stop bit's threshold. The model forecasts the lead leaving and the pre-release
    # lets the car go; the forecast then fades. A lead that reads where it was released (the radar's 0.1 m range steps flicker
    # by up to two on a standing car) is standing, and the car is held on the kiss again; a lead half a metre on is leaving
    planner, d = self.rest_behind(6.0, 30.0)
    assert planner.stop_landing.landing and planner.output_should_stop
    d, frames = self.inch(planner, d, 8.8)
    assert all(bit for bit, _ in frames) and planner.stop_landing.landing
    for _ in range(round(LEAD_DEPARTURE_CONFIRM / DT_MDL) + 2):
      planner.update(frame(0.0, lead=(d, 0.0, 0.0), v_cruise=10.0, forecast=0.5))
    assert not planner.output_should_stop and not planner.stop_landing.landing
    held = []
    for _ in range(round(LEAD_DEPARTURE_CANCEL / DT_MDL) + 10):
      planner.update(frame(0.0, lead=(d + moved, 0.0, 0.0), v_cruise=10.0, forecast=0.12))
      held.append((planner.lead_departure.cancelled, planner.mpc_a_target, planner.output_should_stop, planner.stop_landing.landing,
                   float(planner.output_a_target)))
    cancel = [i for i, h in enumerate(held) if h[0]]
    if moved < 0.3:
      assert len(cancel) == 1, held
      assert not should_stop(0.0, held[cancel[0]][1]), held[cancel[0]]
      assert all(bit and landing and math.isclose(out, -KISS_DECEL, abs_tol=1e-9) for _, _, bit, landing, out in held[cancel[0]:]), held
    else:
      # the launch stays the MPC's plan: its own stop bit, its own target
      assert not cancel and not any(bit or landing for _, _, bit, landing, _ in held), held
      assert math.isclose(held[-1][4], held[-1][1], abs_tol=1e-9) and held[-1][4] > 0.1, held[-1]
    # the lead leaving for real still launches the car at once
    planner.update(frame(0.0, lead=(d + moved, 1.0, 1.0), v_cruise=10.0))
    assert not planner.output_should_stop and not planner.stop_landing.landing

  STOP_PLAN = (1.0, 0.0)
  OPEN_ROAD = (2.0 * PATH_OPEN_LENGTH, 5.0)

  def radar_frame(self, lead, track_id, radar_valid=True, path=None, should_stop=False):
    # a frame at rest with leadOne on a radar track (lead None: no lead), or with radarState failing its checks; path: the model's plan
    sm = frame(0.0, lead=lead, v_cruise=10.0, path=path, should_stop=should_stop)
    sm['radarState'].leadOne.radarTrackId = track_id
    sm.valid['radarState'] = radar_valid
    return sm

  def released_then_taken_back(self):
    # the test above up to the cancel: the car held again on the kiss behind a lead that inched to 8.8 m
    planner, d = self.rest_behind(6.0, 30.0)
    d, _ = self.inch(planner, d, 8.8)
    for _ in range(round(LEAD_DEPARTURE_CONFIRM / DT_MDL) + 2):
      planner.update(frame(0.0, lead=(d, 0.0, 0.0), v_cruise=10.0, forecast=0.5))
    for _ in range(round(LEAD_DEPARTURE_CANCEL / DT_MDL) + 2):
      planner.update(frame(0.0, lead=(d, 0.0, 0.0), v_cruise=10.0, forecast=0.12))
    assert planner.output_should_stop and planner.stop_landing.landing and planner.lead_departure.holding
    return planner, d

  @pytest.mark.parametrize('arrival', [(0.3, 7.5), (6.0, 30.0)], ids=['no_landing', 'landed'])
  def test_a_car_taking_the_slot_after_a_release_does_not_hold_the_car(self, arrival):
    # the lead leaves and, while the car still stands through the brakes' exit, the slot passes to a car standing 20 m ahead:
    # the release is not taken back on that car, and the car goes to it
    planner, d = self.rest_behind(*arrival)
    x, v = d, 0.0
    for _ in range(round(0.6 / DT_MDL)):
      v += 1.5 * DT_MDL
      x += v * DT_MDL
      planner.update(self.radar_frame((x, v, 1.5), 7))
    assert not planner.output_should_stop
    states = []
    for _ in range(round(3.0 / DT_MDL)):
      planner.update(self.radar_frame((20.0, 0.0, 0.0), 9))
      states.append((planner.output_should_stop, planner.stop_landing.landing, planner.lead_departure.cancelled))
    assert not any(any(state) for state in states), states

  @pytest.mark.parametrize('far', [15.0, 25.0])
  @pytest.mark.parametrize('arrival', [(0.3, 7.5), (6.0, 30.0)], ids=['no_landing', 'landed'])
  def test_a_hold_ends_with_the_lead_it_was_for(self, arrival, far):
    # held at rest behind a lead that then leaves the slot sideways: a car standing farther ahead takes it, nothing departed,
    # and the car goes to that car -- the landing included. The MPC's plan takes a few frames to climb to the new car, and
    # its stop bit on those frames is not a hold
    planner, d = self.rest_behind(*arrival)
    for _ in range(round(1.0 / DT_MDL)):
      planner.update(self.radar_frame((d, 0.0, 0.0), 7))
    assert planner.output_should_stop and planner.lead_departure.holding
    states = []
    for _ in range(round(2.0 / DT_MDL)):
      planner.update(self.radar_frame((far, 0.0, 0.0), 9))
      states.append((planner.output_should_stop, planner.stop_landing.landing, float(planner.output_a_target)))
    assert not any(bit or landing for bit, landing, _ in states[round(0.5 / DT_MDL):]), states
    assert not any(landing for _, landing, _ in states) and states[-1][2] > 0.5, states

  @pytest.mark.parametrize('dropout', ['radar_invalid', 'lead_absent'])
  def test_a_lead_missing_at_rest_keeps_the_hold_and_the_landing(self, dropout):
    # landed and held behind a stopped lead; then for a second radarState fails its checks or leadOne is missing, far beyond the
    # MPC's own absence hold. Nothing says the lead has left: the car stays on the kiss, and the lead back where it stood is the
    # same lead, whose departure releases the car
    planner, d = self.rest_behind(6.0, 30.0)
    for _ in range(round(1.0 / DT_MDL)):
      planner.update(self.radar_frame((d, 0.0, 0.0), 7))
    assert planner.output_should_stop and planner.stop_landing.landing and planner.lead_departure.holding
    states = []
    for _ in range(round(1.0 / DT_MDL)):
      planner.update(self.radar_frame((d, 0.0, 0.0), 7, radar_valid=False) if dropout == 'radar_invalid' else self.radar_frame(None, -1))
      states.append((planner.output_should_stop, planner.stop_landing.landing, float(planner.output_a_target)))
    for _ in range(round(1.0 / DT_MDL)):
      planner.update(self.radar_frame((d + 0.1, 0.0, 0.0), 7))
      states.append((planner.output_should_stop, planner.stop_landing.landing, float(planner.output_a_target)))
    assert all(bit and landing and math.isclose(out, -KISS_DECEL, abs_tol=1e-9) for bit, landing, out in states), states
    planner.update(self.radar_frame((d + 0.1, 1.0, 1.0), 7))
    assert not planner.output_should_stop and not planner.stop_landing.landing

  @pytest.mark.parametrize('missing', [0.0, 1.0])
  @pytest.mark.parametrize('arrival', [(0.3, 7.5), (6.0, 30.0)], ids=['no_landing', 'landed'])
  def test_a_car_standing_beyond_the_lead_ends_the_hold_as_a_launch(self, arrival, missing):
    # held at rest behind a lead; at once or after a second with no lead, the slot reads a car standing 5 m beyond where the lead
    # stood: the lead has left, and the car goes to that car, its landing ended
    planner, d = self.rest_behind(*arrival)
    for _ in range(round(1.0 / DT_MDL)):
      planner.update(self.radar_frame((d, 0.0, 0.0), 7))
    for _ in range(round(missing / DT_MDL)):
      planner.update(self.radar_frame(None, -1))
    assert planner.output_should_stop and planner.lead_departure.holding
    states = []
    for _ in range(round(2.0 / DT_MDL)):
      planner.update(self.radar_frame((d + 5.0, 0.0, 0.0), 9))
      states.append((planner.lead_departure.handed_back, planner.output_should_stop, planner.stop_landing.landing, float(planner.output_a_target)))
    assert states[0][0] and not any(handed for handed, _, _, _ in states[1:]), states
    assert not any(bit or landing for _, bit, landing, _ in states[round(0.5 / DT_MDL):]), states
    assert not any(landing for _, _, landing, _ in states) and states[-1][3] > 0.1, states

  @pytest.mark.parametrize('tracks', [(7, 9), (-1, -1)], ids=['radar', 'vision'])
  def test_a_nearer_car_in_the_slot_keeps_the_hold(self, tracks):
    # landed and held behind a lead; a car cuts in between and the slot trades between it and the lead: nothing left
    planner, d = self.rest_behind(6.0, 30.0)
    states = []
    for i in range(round(2.0 / DT_MDL)):
      planner.update(self.radar_frame((3.1, 0.0, 0.0), tracks[1]) if i % 3 else self.radar_frame((d, 0.0, 0.0), tracks[0]))
      states.append((planner.output_should_stop, planner.stop_landing.landing, planner.lead_departure.handed_back))
    assert all(bit and landing and not handed for bit, landing, handed in states), states

  @pytest.mark.parametrize('lead', ['missing', 'standing', 'leaves'])
  def test_the_models_path_opening_ends_a_hold_only_while_its_lead_is_missing(self, lead):
    # stopped at a red line behind a lead the car is held for; at the green Force Stops releases. A lead gone missing before it
    # leaves nothing but the release to go on, and the car goes; with the model planning open road from the release on, a lead
    # still standing there holds it, and a lead in sight on the release frame and missing two frames later has left once the open
    # road outlasts a flash: the car goes then
    car = IdealCar(8.0)
    stop = {'x': 40.0}

    def force_stops(*args):
      if stop['x'] is None:
        return ForceStopsResult()
      return ForceStopsResult(NO_CAP, stop['x'] - car.x, car.v < 0.01, None)
    car.planner.force_stops.update = force_stops
    while car.v >= 0.01:
      car.step(lead=(38.0 - car.x, 0.0, 0.0), v_cruise=10.0)
    planner, d = car.planner, 38.0 - car.x
    for _ in range(round(1.0 / DT_MDL)):
      planner.update(self.radar_frame((d, 0.0, 0.0), 7))
    for _ in range(round(1.0 / DT_MDL)):
      planner.update(self.radar_frame(None if lead == 'missing' else (d, 0.0, 0.0), 7))
    assert planner.output_should_stop and planner.stop_landing.landing and planner.lead_departure.holding
    stop['x'] = None
    gone = {'missing': 0, 'standing': None, 'leaves': 2}[lead]
    handback = {'missing': 0, 'standing': None, 'leaves': 2 + RELEASE_OPEN_FRAMES - 1}[lead]
    states = []
    for i in range(round(2.0 / DT_MDL)):
      planner.update(self.radar_frame(None if gone is not None and i >= gone else (d, 0.0, 0.0), 7,
                                      path=None if lead == 'missing' else self.OPEN_ROAD))
      states.append((planner.lead_departure.handed_back, planner.output_should_stop, planner.stop_landing.landing, float(planner.output_a_target)))
    assert [i for i, (handed, _, _, _) in enumerate(states) if handed] == ([] if handback is None else [handback]), states
    if handback is None:
      assert all(bit for _, bit, _, _ in states), states
    else:
      assert all(bit for _, bit, _, _ in states[:handback]) and not any(bit or landing for _, bit, landing, _ in states[handback:]), states
      assert states[-1][3] > 0.5, states[-1]

  @pytest.mark.parametrize('arrival', [(0.3, 7.5), (6.0, 30.0)], ids=['no_landing', 'landed'])
  def test_a_lead_missing_while_the_model_plans_the_stop_keeps_the_hold(self, arrival):
    # held at rest in a queue, and leadOne goes missing for good while the model, the second sensor, still plans the stop (a truck
    # or a tall trailer the radar loses): nothing says the lead has left, and the car stays held, on the kiss if it landed
    planner, d = self.rest_behind(*arrival)
    for _ in range(round(1.0 / DT_MDL)):
      planner.update(self.radar_frame((d, 0.0, 0.0), 7, path=self.STOP_PLAN))
    landed = planner.stop_landing.landing
    states = []
    for _ in range(round(3.0 / DT_MDL)):
      planner.update(self.radar_frame(None, -1, path=self.STOP_PLAN))
      states.append((planner.output_should_stop, planner.lead_departure.holding, planner.lead_departure.handed_back, planner.stop_landing.landing))
    assert all(bit and held and not handed and landing == landed for bit, held, handed, landing in states), states

  @pytest.mark.parametrize('arrival', [(0.3, 7.5), (6.0, 30.0)], ids=['no_landing', 'landed'])
  def test_a_lead_missing_while_the_model_plans_open_road_ends_the_hold_as_a_launch(self, arrival):
    # held at rest in a queue, no red light anywhere, and the car ahead turns off: leadOne goes missing for good and the model plans
    # open road, which is the evidence that it has left once it outlasts a flash. The car goes then, its landing ended
    planner, d = self.rest_behind(*arrival)
    for _ in range(round(1.0 / DT_MDL)):
      planner.update(self.radar_frame((d, 0.0, 0.0), 7, path=self.STOP_PLAN))
    assert planner.output_should_stop and planner.lead_departure.holding
    landed = planner.stop_landing.landing
    states = []
    for _ in range(round(2.0 / DT_MDL)):
      planner.update(self.radar_frame(None, -1, path=self.OPEN_ROAD))
      states.append((planner.lead_departure.handed_back, planner.output_should_stop, planner.stop_landing.landing, float(planner.output_a_target)))
    handback = RELEASE_OPEN_FRAMES - 1
    assert [i for i, (handed, _, _, _) in enumerate(states) if handed] == [handback], states
    assert all(bit and landing == landed for _, bit, landing, _ in states[:handback]), states
    assert not any(bit or landing for _, bit, landing, _ in states[handback:]), states
    assert states[-1][3] > 0.5, states[-1]

  @pytest.mark.parametrize('reading', ['one_frame', 'two_frames', 'amid_the_stop', 'either_side_of_the_car_in_sight', 'after_open_road_in_sight',
                                       'short_plan_either_side_of_the_car_in_sight', 'still_calling_the_stop'])
  @pytest.mark.parametrize('arrival', [(0.3, 7.5), (6.0, 30.0)], ids=['no_landing', 'landed'])
  def test_open_road_short_of_a_green_while_the_lead_drops_out_keeps_the_hold(self, arrival, reading):
    # held at rest behind a car that stays where it stands; leadOne drops out for a few frames while the model reads open road short
    # of a green: one or two frames of it (alone, amid the stop, or either side of a frame with the car in sight), one frame after a
    # second of open road with the car in sight, a plan reaching 25 m still moving for 0.35 s either side of a frame with the car
    # in sight, or a long plan it still calls the stop on. Nothing says the car has left, and the car stays held
    planner, d = self.rest_behind(*arrival)
    for _ in range(round(1.0 / DT_MDL)):
      planner.update(self.radar_frame((d, 0.0, 0.0), 7, path=self.STOP_PLAN))
    landed = planner.stop_landing.landing
    seen = (d, 0.0, 0.0)
    missing_open, missing_stop, missing_short = (None, self.OPEN_ROAD, False), (None, self.STOP_PLAN, False), (None, (25.0, 4.0), False)
    flash = [missing_open] * (RELEASE_OPEN_FRAMES - 1)
    short = [missing_short] * round(0.35 / DT_MDL)
    frames = {'one_frame': [missing_open], 'two_frames': flash, 'amid_the_stop': [missing_stop, missing_open, missing_stop],
              'either_side_of_the_car_in_sight': flash + [(seen, self.STOP_PLAN, False)] + flash,
              'after_open_road_in_sight': [(seen, self.OPEN_ROAD, False)] * round(1.0 / DT_MDL) + [missing_open],
              'short_plan_either_side_of_the_car_in_sight': short + [(seen, self.STOP_PLAN, False)] + short,
              'still_calling_the_stop': [(None, self.OPEN_ROAD, True)] * round(1.0 / DT_MDL)}[reading]
    frames += [(seen, self.STOP_PLAN, False)] * round(2.0 / DT_MDL)
    states = []
    for lead, path, stopping in frames:
      planner.update(self.radar_frame(lead, 7 if lead is not None else -1, path=path, should_stop=stopping))
      states.append((planner.output_should_stop, planner.lead_departure.holding, planner.lead_departure.handed_back, planner.stop_landing.landing))
    assert all(bit and held and not handed and landing == landed for bit, held, handed, landing in states), states

  @pytest.mark.parametrize('dropout', ['radar_invalid', 'lead_absent'])
  def test_a_dropout_at_rest_does_not_release_the_hold(self, dropout):
    # the hover recipe: the MPC plans above the stop bit's threshold at rest and only the hold keeps the car. A radarState that
    # fails its checks, or a leadOne missing, for a frame or for a second is the same lead
    planner, d = self.rest_behind(0.3, 7.5)
    d, frames = self.inch(planner, d, 9.0)
    assert not should_stop(0.0, frames[-1][1]) and planner.output_should_stop
    for _ in range(round(1.0 / DT_MDL)):
      if dropout == 'radar_invalid':
        planner.update(self.radar_frame((d, 0.0, 0.0), -1, radar_valid=False))
      else:
        planner.update(self.radar_frame(None, -1))
      assert planner.output_should_stop and planner.lead_departure.holding
    for _ in range(round(1.0 / DT_MDL)):
      planner.update(self.radar_frame((d, 0.0, 0.0), -1))
      assert planner.output_should_stop

  def test_a_lead_missing_under_a_hold_leaves_nothing_in_the_launch(self, accel_max):
    # the hover recipe, and leadOne missing for a second under the hold: the MPC plans for no lead meanwhile, but the car stood.
    # Once the lead is back and leaves, the car launches no harder than it would have without the absence
    launches = []
    for missing in (0.0, 1.0):
      planner, d = self.rest_behind(0.3, 7.5)
      d, _ = self.inch(planner, d, 9.0)
      for _ in range(round(missing / DT_MDL)):
        planner.update(self.radar_frame(None, -1))
      for _ in range(2):
        planner.update(self.radar_frame((d, 0.0, 0.0), -1))
      assert planner.output_should_stop
      x, v, launch = d, 0.0, []
      while len(launch) < round(1.0 / DT_MDL):
        v += 1.5 * DT_MDL
        x += v * DT_MDL
        planner.update(self.radar_frame((x, v, 1.5), -1))
        if not planner.output_should_stop:
          launch.append(float(planner.output_a_target))
      launches.append(np.array(launch))
    assert np.all(launches[1] <= launches[0] + 0.01), np.round(launches, 2)


class TestHoldRelease:
  # the real ForceStops drives these: a commit, the hold at standstill, then the two ways a hold can end. Only a release is a
  # launch: a creep or grade resume keeps the commitment (force_stops "the latch survives") and the landing corridor with it
  def step(self, planner, v_ego, path_end, terminal_speed, should_stop, e2e_accel, a_ego=0.0, standstill=None):
    # one model frame: a path ending path_end ahead at terminal_speed, with the model's own stop bit and request
    planner.update(frame(v_ego, a_ego, experimental=True, e2e_accel=e2e_accel, should_stop=should_stop, path=(path_end, terminal_speed),
                         standstill=standstill))

  def land_a_committed_stop(self, e2e_hold=0.0, v_end=0.0):
    # a lead-free strict stop 10 m out at 3 m/s commits and the car lands on it down to v_end. The model asks for 0.7 of the
    # stop's need until the last 3 m and e2e_hold from there on
    car = IdealCar(3.0)
    committed = False
    for _ in range(round(10.0 / DT_MDL)):
      d = 10.0 - car.x
      car.step(experimental=True, e2e_accel=-0.7 * car.v ** 2 / (2.0 * d) if d > 3.0 else e2e_hold, path=(d, 0.0), should_stop=d < 3.0)
      committed = committed or car.planner.force_stops.forcing
      if car.v <= v_end:
        break
    assert committed and car.v <= v_end
    return car.planner

  def hold_at_a_committed_stop(self, seconds=2.0, e2e_hold=0.0):
    # standstill turns the commitment into the hold for as long as a red light lasts
    planner = self.land_a_committed_stop(e2e_hold)
    for _ in range(round(seconds / DT_MDL)):
      self.step(planner, 0.0, 1.0, 0.0, True, e2e_hold)
    assert planner.force_stops.holding
    assert planner.stop_landing.landing
    return planner

  OPEN_ROAD = (2.0 * PATH_OPEN_LENGTH, 5.0, False, 0.5)

  def red_light(self, crossing, reading=lambda i: None, stop_bit=True):
    # first in line at a red: the car lands on the committed stop, the light stays red for 10 s, then turns green and the model
    # plans open road. crossing(i) is the car crossing the intersection that is flagged as the lead on the i-th frame at rest, or
    # None; reading(i) is what the model reads instead of the red's stop on that frame, (path end, terminal speed, shouldStop,
    # request), or None; stop_bit False: the model plans the red's stop without its shouldStop. The car stays at rest while the
    # stop bit is up. Returns each frame's (stop bit, Force Stops hold, pre-release hold, hand-back) and the car's final speed
    planner = self.land_a_committed_stop()
    v, states = 0.0, []
    for i in range(round(12.0 / DT_MDL)):
      green = i >= round(10.0 / DT_MDL)
      end, terminal, stopping, request = reading(i) or (self.OPEN_ROAD if green else (1.0, 0.0, stop_bit, 0.0))
      sm = frame(v, experimental=True, e2e_accel=request, should_stop=stopping, path=(end, terminal), lead=crossing(i))
      sm['radarState'].leadOne.radarTrackId = 7
      planner.update(sm)
      states.append((planner.output_should_stop, planner.force_stops.holding, planner.lead_departure.holding, planner.lead_departure.handed_back))
      if not planner.output_should_stop:
        v = max(v + float(planner.output_a_target) * DT_MDL, 0.0)
    return states, v

  def test_a_car_crossing_at_a_red_light_after_the_car_stopped_is_not_its_lead(self):
    # a car crossing the intersection is flagged as the lead 8 m ahead 2 s into the red and clears a second later: the car did not
    # come to rest behind it, so the pre-release never holds, the stop bit stays up through the red and Force Stops' release at
    # the green launches the car
    states, v = self.red_light(lambda i: (8.0, 0.0, 0.0) if round(2.0 / DT_MDL) <= i < round(3.0 / DT_MDL) else None)
    release = round(10.0 / DT_MDL) + RELEASE_OPEN_FRAMES - 1
    assert v > 0.5
    assert not any(held for _, _, held, _ in states), states
    assert all(bit for bit, _, _, _ in states[:release]) and not any(bit for bit, _, _, _ in states[release:]), states
    assert states[release - 1][1] and not states[release][1]

  def test_a_car_crossing_at_a_red_light_as_the_car_stops_does_not_hold_it_past_the_green(self):
    # a car crossing the intersection is flagged as the lead 8 m ahead from the car's first frames at rest, so the car is held for it
    # as for a car it came to rest behind, and Force Stops' hold ends for it. The light turns green while it is still crossing and
    # it clears two frames later, the model planning open road: the hold ends as a launch once that outlasts a flash
    green = round(10.0 / DT_MDL)
    states, v = self.red_light(lambda i: (8.0, 0.0, 0.0) if 2 <= i < green + 2 else None)
    handback = green + 2 + RELEASE_OPEN_FRAMES - 1
    assert v > 0.5
    assert all(held and not fs_hold for _, fs_hold, held, _ in states[3:green + 2]), states
    assert all(bit for bit, _, _, _ in states[:handback]), states
    assert [i for i, (_, _, _, handed) in enumerate(states) if handed] == [handback], states
    assert not any(bit for bit, _, _, _ in states[handback:]), states

  @pytest.mark.parametrize('stop_bit', [True, False], ids=['should_stop', 'path_only'])
  @pytest.mark.parametrize('flash', [1, RELEASE_OPEN_FRAMES - 1])
  def test_a_car_crossing_as_the_car_stops_that_clears_on_a_flash_of_open_road_keeps_it_held_at_the_red(self, flash, stop_bit):
    # the car crossing from the car's first frames at rest clears 4 s into the red on the very frame the model flashes open road
    # for a frame or two; then the model plans the stop again, with its shouldStop or as a path only. A flash is not a green: the
    # stop bit stays up through the red, and Force Stops' release at the green launches the car
    clear, green = round(4.0 / DT_MDL), round(10.0 / DT_MDL)
    states, v = self.red_light(lambda i: (8.0, 0.0, 0.0) if 2 <= i < clear else None,
                               lambda i: self.OPEN_ROAD if clear <= i < clear + flash else None, stop_bit)
    release = green + RELEASE_OPEN_FRAMES - 1
    assert all(held and not fs_hold for _, fs_hold, held, _ in states[3:clear]), states
    assert all(bit for bit, _, _, _ in states[clear:release]) and not any(bit for bit, _, _, _ in states[release:]), states
    assert not any(handed for _, _, _, handed in states[clear:release]) and v > 0.5, states

  @pytest.mark.parametrize('frames', [1, RELEASE_OPEN_FRAMES - 1, 7, 12, None])
  @pytest.mark.parametrize('reading', [(2.0 * PATH_OPEN_LENGTH, 5.0, False, 0.5), (25.0, 4.0, False, 0.3), (2.0 * PATH_OPEN_LENGTH, 5.0, True, 0.0)],
                           ids=['long', 'short', 'long_still_stopping'])
  def test_the_models_open_road_at_a_red_ends_a_hold_for_a_crossing_car_only_where_it_ends_force_stops_hold(self, reading, frames):
    # 4 s into the red the model reads open road for a few frames or up to the green, as the big model does at a red it expects to
    # turn green: a long plan, one that reaches 25 m still moving, or a long plan it still calls the stop on. With no car crossing,
    # Force Stops' hold judges it. With a car crossing from the car's first frames at rest that clears on the reading's first frame,
    # the pre-release holds the car and Force Stops does not: the stop bit first falls on the same frame either way
    read = round(4.0 / DT_MDL)
    reading_at = lambda i: reading if read <= i < (read + frames if frames is not None else round(10.0 / DT_MDL)) else None  # noqa: E731
    alone, _ = self.red_light(lambda i: None, reading_at)
    crossed, _ = self.red_light(lambda i: (8.0, 0.0, 0.0) if 2 <= i < read else None, reading_at)
    assert all(held for _, _, held, _ in crossed[3:read]) and not any(held for _, _, held, _ in alone), crossed
    assert all(bit for bit, _, _, _ in alone[:read]) and all(bit for bit, _, _, _ in crossed[3:read])
    down = next(i for i in range(read, len(alone)) if not (alone[i][0] and crossed[i][0]))
    assert not alone[down][0] and not crossed[down][0], (down, crossed[read:down + 1], alone[read:down + 1])
    assert [i for i, (_, _, _, handed) in enumerate(crossed[:down + 1]) if handed] == [down], crossed[read:down + 1]

  def test_a_creep_resume_from_a_hold_is_not_a_launch_and_the_landing_survives(self):
    planner = self.hold_at_a_committed_stop()
    self.step(planner, 1.0, 1.0, 0.0, True, 0.0)
    assert planner.force_stops.forcing and not planner.force_stops.holding   # a moving commitment again, not a release
    assert planner.stop_landing.landing
    assert planner.output_a_target <= -planner.stop_landing.floor(1.0)   # the floor holds; a car not slowing is pressed further

  def test_a_hold_keeps_its_stop_bit_while_the_car_rolls(self):
    # the car's standstill flag rises on its wheel speeds while vEgo can still read just above the standstill speed, and a hold rolls
    # on short of the resume speed: while the kiss lands the car, the hold still owns the stop bit
    planner = self.land_a_committed_stop(v_end=2.0 * STANDSTILL_SPEED)
    assert planner.stop_landing.landing and not planner.output_should_stop
    for v_ego, standstill in ((1.05 * STANDSTILL_SPEED, True), (0.5, False)):
      self.step(planner, v_ego, 1.0, 0.0, True, 0.0, standstill=standstill)
      assert planner.force_stops.holding and planner.stop_landing.landing, v_ego
      assert planner.output_should_stop, v_ego

  def test_a_hold_released_by_an_open_road_ends_the_landing_and_launches_from_the_kiss(self):
    # the green: the model drops its stop bit and plans a long moving path, the commitment goes with the hold. Whichever
    # candidate sat under the kiss -- the column early in a hold under a model asking a hair of throttle, cruise once it settles,
    # the model asking a little braking -- the launch starts from the kiss the car was told, at the cruise jerk, and the stop bit
    # drops with the release
    for seconds, e2e_hold, owner in ((0.5, 0.05, 'column'), (2.0, 0.0, 'cruise'), (2.0, -0.1, 'e2e')):
      planner = self.hold_at_a_committed_stop(seconds, e2e_hold)
      assert planner.plan_winner == owner and math.isclose(planner.output_a_target, -KISS_DECEL, rel_tol=1e-6, abs_tol=1e-9)
      targets, stop_bits = [float(planner.output_a_target)], []
      for _ in range(RELEASE_OPEN_FRAMES + 10):
        self.step(planner, 0.0, 2.0 * PATH_OPEN_LENGTH, 5.0, False, 0.5)
        targets.append(float(planner.output_a_target))
        stop_bits.append(bool(planner.output_should_stop))
      assert not planner.force_stops.holding and not planner.force_stops.forcing
      assert not planner.stop_landing.landing
      assert not any(bit for bit, a in zip(stop_bits, targets[1:], strict=True) if a > -KISS_DECEL + 1e-6), (owner, stop_bits)
      steps = np.diff(targets)
      assert np.all(steps >= -1e-9) and np.all(steps <= j_cruise_step(0.0) + 1e-9), (owner, np.round(targets, 3))
      assert targets[-1] > 0.0

  # F035: nothing owns the launch after a green, so a request still braking on the frame after the release re-latches the landing
  @pytest.mark.xfail(strict=True, reason='the landing re-latches at rest after the release, and a car at rest cannot end it')
  def test_a_green_launches_when_the_model_lets_go_a_frame_after_the_path_opens(self):
    planner = self.hold_at_a_committed_stop(2.0, -0.5)
    for i in range(RELEASE_OPEN_FRAMES + 20):
      self.step(planner, 0.0, 2.0 * PATH_OPEN_LENGTH, 5.0, False, -0.5 if i <= RELEASE_OPEN_FRAMES else 0.5)
    assert not planner.force_stops.holding and not planner.stop_landing.landing
    assert planner.output_a_target > 0.0 and not planner.output_should_stop


class TestHandOver:
  # a candidate that leaves min() hands the car to the cruise candidate, which carries it from the published target at the cruise
  # jerk: its own state kept slewing toward the set speed while it lost, up to ACCEL_MAX in Experimental
  def assert_carried(self, car, frames, **kwargs):
    previous = car.a
    for _ in range(frames):
      bound = j_cruise_step(car.v)
      car.step(**kwargs)
      assert car.a - previous <= bound + 1e-6, (previous, car.a, car.planner.plan_winner)
      previous = car.a

  def test_a_mode_exit_hands_over_from_the_output(self):
    car = IdealCar(15.0)
    for _ in range(40):
      car.step(experimental=True, e2e_accel=-1.0)
    assert car.planner.plan_winner == 'e2e' and car.planner.a_cruise > 1.0
    self.assert_carried(car, 20)
    assert car.a > 0.0

  def red_light(self, v_ego, line, release_at):
    # the model calls a line ahead and asks for 0.7 of the stop's need; Force Stops commits on the strict evidence
    car = IdealCar(v_ego)
    while line - car.x > release_at:
      d = line - car.x
      car.step(experimental=True, e2e_accel=-0.7 * car.v ** 2 / (2.0 * d), path=(d, 0.0), should_stop=d < 3.0)
    assert car.planner.force_stops.forcing
    return car

  def test_a_green_that_releases_a_committed_stop_hands_over_from_the_output(self):
    for release_at, winner in ((35.0, 'column'), (12.0, 'profile')):
      car = self.red_light(14.0, 60.0, release_at)
      assert car.planner.plan_winner == winner
      for _ in range(RELEASE_OPEN_FRAMES):
        car.step(experimental=True, e2e_accel=1.0, path=(100.0, car.v))
      assert not car.planner.force_stops.forcing
      self.assert_carried(car, 20, experimental=True, e2e_accel=1.0, path=(100.0, car.v))

  def test_the_profile_ending_while_the_column_stays_hands_over_from_the_output(self):
    car = IdealCar(8.0)
    stop = {'x': 30.0, 'a': -2.0}
    car.planner.force_stops.update = lambda *args: ForceStopsResult(NO_CAP, stop['x'] - car.x, False, stop['a'])
    for _ in range(20):
      car.step()
    assert car.planner.plan_winner == 'profile'
    stop['a'] = None
    self.assert_carried(car, 10)
    assert car.planner.mpc.source == LongitudinalPlanSource.stop

  def test_a_lead_that_leaves_hands_over_from_what_the_car_was_told(self, accel_max):
    # braking behind a lead that brakes hard, then it leaves the radar: when the MPC hands its plan to the free run, the new plan
    # starts from the target the car was given, not from the lead's plan one frame on, which lies well below it
    car = IdealCar(15.0)
    x_lead, v_lead = 25.0, 15.0
    for i in range(round(4.5 / DT_MDL)):
      v_lead = max(v_lead - 3.0 * DT_MDL, 5.0) if i * DT_MDL > 2.0 else v_lead
      x_lead += v_lead * DT_MDL
      car.step(lead=(x_lead - car.x, v_lead, -3.0 if v_lead > 5.0 else 0.0), v_cruise=20.0)
    handed_over = []
    for _ in range(20):
      told, before = car.a, car.planner.mpc.binding_obstacle
      car.step(v_cruise=20.0)
      if before != car.planner.mpc.binding_obstacle:
        handed_over.append((told, float(car.planner.mpc.a_solution[0]), car.a))
    assert len(handed_over) == 1 and handed_over[0][0] > 0.0, handed_over
    told, plan_start, published = handed_over[0]
    assert math.isclose(plan_start, told, abs_tol=1e-6)
    assert published >= told, handed_over

  def test_a_commit_joins_from_the_published_target_not_the_measured_acceleration(self):
    # the car's measured acceleration lags and rings around what it was told; the committed profile starts from the published target
    for bias in (-0.5, 0.3):
      car = IdealCar(14.0)
      step = None
      while step is None and car.x < 40.0:
        d = 60.0 - car.x
        committed, told = car.planner.force_stops.forcing, car.a
        car.planner.update(frame(car.v, car.a + bias, experimental=True, e2e_accel=-0.7 * car.v ** 2 / (2.0 * d), path=(d, 0.0)))
        car.a = float(car.planner.output_a_target)
        car.v += car.a * DT_MDL
        car.x += car.v * DT_MDL
        if car.planner.force_stops.forcing and not committed:
          step = car.a - told
      assert step is not None and abs(step) <= PROFILE_JERK * DT_MDL + 1e-9, (bias, step)

  def test_a_lead_pulling_away_is_released_at_the_mpc_s_own_rate(self):
    # the hand-over is an event: cruise does not rate-limit a candidate that stays, so a lead's release is the MPC's
    car = IdealCar(15.0)
    d_lead, v_lead, rises = 30.0, 15.0, []
    for i in range(round(14.0 / DT_MDL)):
      # the lead brakes to 6 m/s, holds it, then pulls away
      a_lead = 2.5 if i * DT_MDL > 7.0 else -2.0 if i * DT_MDL > 2.0 and v_lead > 6.0 else 0.0
      v_lead = max(v_lead + a_lead * DT_MDL, 0.0)
      d_lead += v_lead * DT_MDL
      previous, bound = car.a, j_cruise_step(car.v)
      car.step(lead=(d_lead - car.x, v_lead, a_lead))
      if car.planner.plan_winner == 'mpc' and car.a > previous:
        rises.append(car.a - previous - bound)
    assert max(rises) > 0.0

  def test_a_lead_leaving_a_standstill_ends_the_kiss_and_the_launch_starts_from_it(self):
    # behind a stopped lead the landing holds the car on the kiss while the MPC wins underneath; the lead's departure is a launch
    # that ends the kiss, and the launch starts from it at the cruise jerk until the MPC's own launch takes over
    car = IdealCar(6.0)
    x_lead, v_lead = 30.0, 0.0
    while car.v > 0.0:
      car.step(lead=(x_lead - car.x, v_lead, 0.0), v_cruise=10.0)
    for _ in range(round(2.0 / DT_MDL)):
      car.step(lead=(x_lead - car.x, v_lead, 0.0), v_cruise=10.0)
    assert car.planner.plan_winner == 'mpc' and math.isclose(car.a, -KISS_DECEL, rel_tol=1e-6, abs_tol=1e-9)
    previous = car.a
    for _ in range(round(2.0 / DT_MDL)):
      v_lead += 1.5 * DT_MDL
      x_lead += v_lead * DT_MDL
      bound = j_cruise_step(car.v)
      car.step(lead=(x_lead - car.x, v_lead, 1.5), v_cruise=10.0)
      assert car.a - previous <= bound + 1e-6, (previous, car.a, car.planner.plan_winner)
      previous = car.a
    assert car.a > 0.5 and not car.planner.output_should_stop


def bend_frame(v_ego, a_ego, x, bend, steering_pressed=False, v_cruise=13.0):
  # combo's curve world on one planner frame (the maneuver plant's, plant.py): a world-fixed bend (start, length, curvature)
  # that the model's path does not show, and a steering that holds it up to the car's torque authority and pins beyond it, so
  # the curve policy reacts to the measured steering alone and the driver's own steering drops it at once
  start, length, curvature = bend
  sm = frame(v_ego, a_ego, v_cruise=v_cruise)
  sm['carState'].steeringPressed = steering_pressed
  torque = PALISADE_CP.lateralTuning.torque
  desired = v_ego ** 2 * (curvature if start <= x <= start + length else 0.0)
  lateral = min(desired, torque.latAccelFactor * (1.0 - torque.friction))
  state = sm['controlsState'].lateralControlState.init('torqueState')
  state.active = True
  state.output = float(min(desired / torque.latAccelFactor + torque.friction, 1.0) if desired > 0.0 else 0.0)
  state.error = float(desired - lateral)
  state.actualLateralAccel = float(lateral)
  state.desiredLateralAccel = float(desired)
  state.saturated = state.output >= 1.0
  sm['carControl'].latActive = True
  return sm


class TestCurveHandOver:
  # combo's curve policy is a candidate like the others: while it sets the output it is the winner, and when it lets go (the
  # driver's own steering drops it at once) cruise carries the car from the published target at the cruise jerk, not from
  # wherever it slewed toward the set speed meanwhile
  @staticmethod
  def scripted(car, results):
    results = iter(results)
    car.planner.curve_speed_limiter.update = lambda *args, **kwargs: next(results)

  def test_the_curve_policy_letting_go_hands_over_from_the_output(self):
    car = IdealCar(15.0)
    self.scripted(car, [CurveResult(-1.0, REGIME_BRAKE)] * 60 + [CurveResult()] * 20)
    for _ in range(60):
      car.step()
      assert car.planner.plan_winner == 'curve' and car.planner.plan_source == LongitudinalPlanSource.curve
    assert math.isclose(car.a, -1.0) and car.planner.a_cruise > 1.0
    previous = car.a
    for i in range(20):
      bound = j_cruise_step(car.v)
      car.step()
      if i == 0:
        assert car.planner.plan_winner == 'cruise' and car.planner.plan_source == LongitudinalPlanSource.cruise
      assert car.a - previous <= bound + 1e-6, (i, previous, car.a, car.planner.plan_winner)
      previous = car.a
    assert car.a > 0.0

  def test_a_curve_policy_easing_off_by_itself_keeps_its_own_rate(self):
    # a candidate that stays is not rate-limited by the hand-over: the policy's own lift out of the bend is the curve's
    car = IdealCar(15.0)
    lift = [-1.0 + 0.1 * k for k in range(1, 11)]
    self.scripted(car, [CurveResult(-1.0, REGIME_BRAKE)] * 40 + [CurveResult(a, REGIME_FREE) for a in lift])
    for _ in range(40):
      car.step()
    for a_target in lift:
      previous = car.a
      car.step()
      assert car.planner.plan_winner == 'curve' and math.isclose(car.a, a_target), (previous, car.a, car.planner.plan_winner)
      assert car.a - previous > j_cruise_step(car.v)

  def test_a_steering_override_that_drops_the_curve_policy_hands_over_from_the_output(self):
    # a bend tighter than the rack can hold at 9 m/s: the pinned steering brakes the car through the curve policy for a second,
    # then the driver takes the wheel and the policy drops out on that frame
    bend = (30.0, 150.0, 0.06)
    planner = LongitudinalPlanner(PALISADE_CP, init_v=9.0)
    v, a, x, braked = 9.0, 0.0, 0.0, 0
    for _ in range(round(8.0 / DT_MDL)):
      planner.update(bend_frame(v, a, x, bend))
      a = float(planner.output_a_target)
      v, x = max(v + a * DT_MDL, 0.0), x + v * DT_MDL
      braked = braked + 1 if planner.plan_winner == 'curve' and a < -0.2 else 0
      if braked == round(1.0 / DT_MDL):
        break
    assert braked == round(1.0 / DT_MDL) and bend[0] <= x <= bend[0] + bend[1], (braked, x)
    for i in range(round(1.0 / DT_MDL)):
      previous, bound = a, j_cruise_step(v)
      planner.update(bend_frame(v, a, x, bend, steering_pressed=True))
      a = float(planner.output_a_target)
      v, x = max(v + a * DT_MDL, 0.0), x + v * DT_MDL
      assert planner.curve.a_target is None
      if i == 0:
        assert planner.plan_winner == 'cruise' and planner.plan_source == LongitudinalPlanSource.cruise
      assert a - previous <= bound + 1e-6, (i, previous, a, planner.plan_winner)


class TestMpcSeed:
  # the published target is the MPC's plan read action_t ahead, not its current acceleration: fed back as the plan's start, every
  # edit of it (the landing law, a hand-over, the clip) comes back through get_accel_from_plan at about -0.5x and rings at 10 Hz
  def stopped_lead(self, edit):
    car = IdealCar(10.0)
    candidates = []
    for i in range(80):
      car.step(lead=(60.0 - car.x, 0.0, 0.0))
      candidates.append(car.planner.mpc_a_target)
      if i == 60:
        assert car.planner.plan_winner == 'mpc'
        car.planner.output_a_target += edit
    return np.array(candidates)

  def test_an_edit_of_the_published_target_does_not_ring_through_the_mpc(self):
    base, edited = self.stopped_lead(0.0), self.stopped_lead(0.3)
    assert np.max(np.abs(edited[61:] - base[61:])) < 0.02, np.round(edited[61:67] - base[61:67], 3)

  def committed_column(self, edit):
    # the committed stop column drives a red light through the real Force Stops, and the published target is edited once
    car = TestHandOver().red_light(14.0, 60.0, 35.0)
    assert car.planner.plan_winner == 'column'
    car.planner.output_a_target += edit
    candidates = []
    for _ in range(10):
      d = 60.0 - car.x
      car.step(experimental=True, e2e_accel=-0.7 * car.v ** 2 / (2.0 * d), path=(d, 0.0), should_stop=d < 3.0)
      candidates.append(car.planner.mpc_a_target)
    return np.array(candidates)

  def test_an_edit_of_the_published_target_does_not_ring_through_the_committed_column(self):
    base, edited = self.committed_column(0.0), self.committed_column(0.3)
    assert np.max(np.abs(edited - base)) < 0.02, np.round(edited - base, 3)

  def test_the_mpc_starts_from_the_published_target_when_another_candidate_drove(self):
    car = IdealCar(15.0)
    for _ in range(20):
      published = float(car.planner.output_a_target)
      car.step()
      assert car.planner.plan_winner == 'cruise'
      assert car.planner.mpc.x0[2] == published

  def test_the_mpc_starts_from_rest_while_the_pre_release_holds_the_car(self):
    # the hover recipe: the pre-release holds the car at rest while the MPC plans above the stop bit's threshold, then leadOne
    # goes missing and comes back. The car stands throughout, and every plan starts from the car at rest
    standstill = TestStandstillRelease()
    planner, d = standstill.rest_behind(0.3, 7.5)
    d, _ = standstill.inch(planner, d, 9.0)
    for lead in [(d, 0.0, 0.0)] * 10 + [None] * 20 + [(d, 0.0, 0.0)] * 10:
      planner.update(standstill.radar_frame(lead, -1))
      assert planner.lead_departure.holding and planner.output_should_stop
      assert planner.mpc.a_solution[0] == 0.0, planner.mpc.a_solution[0]

  def test_the_mpc_plans_from_the_measured_speed(self):
    # nothing below the planner closes a loop on speed, so a car that does not do what it is told -- it answers through the ESP's
    # identified lag, and brakes BRAKE_SHORTFALL less than asked -- is planned from the speed it has, and still rests short of a
    # stopped lead (the maneuver suite's TestACarThatBrakesLessThanAsked runs the same car through the plant)
    for short in (0.0, BRAKE_SHORTFALL):
      planner, long_control, actuator = LongitudinalPlanner(PALISADE_CP, init_v=10.0), LongControl(PALISADE_CP), ActuatorLag()
      car_state = messaging.new_message('carState').carState
      v, a, x = 10.0, 0.0, 0.0
      for _ in range(round(20.0 / DT_MDL)):
        planner.update(frame(v, a, lead=(40.0 - x, 0.0, 0.0), v_cruise=10.0))
        assert math.isclose(planner.mpc.v_solution[0], v, abs_tol=1e-6), (short, planner.mpc.v_solution[0], v)
        for _ in range(round(DT_MDL / DT_CTRL)):
          car_state.vEgo, car_state.aEgo = v, a
          command = float(long_control.update(True, car_state, float(planner.output_a_target), planner.output_should_stop, (ACCEL_MIN, ACCEL_MAX)))
          a = actuator.update(command, v) + (short if v > 0.0 else 0.0)
          v = max(v + a * DT_CTRL, 0.0)
          x += v * DT_CTRL
      assert v == 0.0 and 40.0 - x >= LEAD_MIN_GAP_BUDGET, (short, v, 40.0 - x)

  def test_a_plan_beyond_what_the_car_can_do_is_not_continued(self, accel_max):
    # a stopped car 80 m ahead at 35 m/s: the car brakes at ACCEL_MIN while the MPC's plan asks for more. Continued from its own
    # plan the MPC starts from braking the car is not doing and lets the brake off; it continues from the braking the car can do
    car = IdealCar(35.0)
    for _ in range(60):
      car.step(v_cruise=35.0)
      car.v = 35.0
    car.x = 0.0
    targets, seeds = [], []
    for _ in range(60):
      car.step(lead=(80.0 - car.x, 0.0, 0.0), v_cruise=35.0)
      targets.append(car.a)
      if car.planner.plan_winner in ('mpc', 'column'):
        seeds.append(float(car.planner.mpc.x0[2]))
    assert min(targets) == ACCEL_MIN
    braking = np.array(targets)
    let_off = np.diff(braking)[braking[:-1] < -1.0]
    assert let_off.max() <= PLAN_JERK_MAX * DT_MDL, let_off.max()
    assert min(seeds) >= ACCEL_MIN, min(seeds)

  def test_a_driver_override_restarts_the_mpc_from_the_car(self):
    # braking behind a slower lead, the driver presses the gas for three frames: the MPC restarts from the car's acceleration,
    # not from the braking plan it had before the override
    car = IdealCar(20.0)
    x_lead = 40.0
    for _ in range(60):
      x_lead += 12.0 * DT_MDL
      car.step(lead=(x_lead - car.x, 12.0, 0.0))
    assert car.planner.plan_winner == 'mpc' and car.a < -1.0
    braking = car.a
    car.a = 0.8
    for _ in range(3):
      x_lead += 12.0 * DT_MDL
      car.planner.update(frame(car.v, car.a, lead=(x_lead - car.x, 12.0, 0.0), engaged=False))
      assert math.isclose(car.planner.mpc.x0[2], car.a, rel_tol=1e-6, abs_tol=1e-6)
      car.v += car.a * DT_MDL
      car.x += car.v * DT_MDL
    x_lead += 12.0 * DT_MDL
    first = car.step(lead=(x_lead - car.x, 12.0, 0.0))
    assert abs(first - 0.8) < abs(first - braking), (braking, first)

  def onset_give_back(self, a_brake, gap):
    # a lead gap ahead at 20 m/s brakes at a_brake to a stop. One SQP-RTI iteration over-reads the problem a sudden onset poses
    # and the next frame takes part of it back, the solver's own; with the onset frame solved to convergence (30 iterations,
    # the same as converging every frame) what the target still gives back is the seed's
    car = IdealCar(20.0)
    run = car.planner.mpc.run
    d_lead, v_lead, targets = gap, 20.0, []
    for i in range(round(9.0 / DT_MDL)):
      onset = i * DT_MDL > 4.0
      a_lead = a_brake if onset and v_lead > 0.0 else 0.0
      v_lead = max(v_lead + a_lead * DT_MDL, 0.0)
      d_lead += v_lead * DT_MDL
      car.planner.mpc.run = (lambda iterations=1: run(30)) if onset and not targets else run
      car.step(lead=(d_lead - car.x, v_lead, a_lead), v_cruise=20.0)
      if onset:
        targets.append(car.a)
    onset = targets[:int(np.argmin(targets)) + 1]
    return float(np.max(np.diff(onset)))

  def test_the_seed_gives_nothing_back_at_a_braking_onset(self):
    for a_brake, gap in ((-5.0, 45.0), (-4.0, 35.0), (-3.0, 25.0)):
      assert self.onset_give_back(a_brake, gap) < 0.01, (a_brake, gap)

  def test_at_standstill_the_mpc_does_not_mirror_the_kiss(self):
    # behind a stopped lead the landing publishes the kiss; the MPC's own plan at rest must keep its stop bit, not reflect the kiss
    car = IdealCar(6.0)
    while car.v > 0.0:
      car.step(lead=(30.0 - car.x, 0.0, 0.0), v_cruise=10.0)
    for _ in range(round(3.0 / DT_MDL)):
      car.step(lead=(30.0 - car.x, 0.0, 0.0), v_cruise=10.0)
      assert car.planner.plan_winner == 'mpc' and math.isclose(car.a, -KISS_DECEL, rel_tol=1e-6, abs_tol=1e-9)
      assert should_stop(car.v, car.planner.mpc_a_target), car.planner.mpc_a_target

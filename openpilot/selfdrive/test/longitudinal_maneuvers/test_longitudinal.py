import contextlib
import itertools
import unittest
from unittest import mock

import numpy as np
import opendbc.car.interfaces as car_interfaces
from opendbc.can.dbc import DBC
from opendbc.car import Bus
from opendbc.car.hyundai.radar_interface import RADAR_START_ADDR
from opendbc.car.values import PLATFORMS
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib import longitudinal_planner
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_planner import J_CRUISE_BP, J_CRUISE_VALS
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc
from openpilot.selfdrive.controls.lib.standstill_exit import LAUNCH_ACCEL, StandstillExit
from openpilot.selfdrive.controls.lib.stop_landing import KISS_SPEED, LANDING_SPEED, STALL_MAX, StopLanding, landing_bound
from openpilot.common.test import OpenpilotTestCase
from openpilot.common.parameterized import parameterized_class

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import STOP_DISTANCE
from openpilot.selfdrive.controls.lib.necessity_supervisor import LEAD_DEPARTURE_SPEED
from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import CREEP_HOLD_REQUEST, ActuatorLag, EscStandstill, Plant, palisade_car_params

ACCEL_MAX_PINS = (2.0, 4.0)  # stock opendbc's ACCEL_MAX and the fork's
PALISADE = palisade_car_params()  # BLoTv3's stop maneuvers run on the car it drives
LEAD_REST_GAP = 4.0  # m, the least gap the car may rest at behind a stopped lead
LEAD_CLOSEST_GAP = 2.0  # m, the least gap the car may ever close to behind a lead
PLAN_JERK_MAX = 3.0  # m/s^3, the plan's comfort jerk bound; the car limits a rising request at 3.0 and a braking one at 5.0 (SCC14)
PLAN_REVERSAL = 0.15  # m/s^2, a let-off or a re-brake of this size before the final ease
BRAKE_SHORTFALL = 0.3  # m/s^2, how far short of every request a car falls while it rolls
DRIVELINE_CREEP = 0.5  # m/s^2 at rest, an assumed level: no creep event is in the logs
BRAKING_ONSET = -0.5  # m/s^2, the car's acceleration from which a lead maneuver counts as braking
ANTICIPATION = 0.3  # s, how much earlier the model's forecast of a braking lead brakes the car than the radar alone
ANTICIPATION_GAP = 0.8  # m, how much more room it keeps behind that lead


class PinnedTestCase(OpenpilotTestCase):
  accel_max: float

  def setUp(self):
    super().setUp()
    # the planner and the MPC bind opendbc's ACCEL_MAX at import, LongControl's limits read it from opendbc
    for module in (car_interfaces, longitudinal_planner, long_mpc):
      patcher = mock.patch.object(module, 'ACCEL_MAX', self.accel_max)
      patcher.start()
      self.addCleanup(patcher.stop)


# TODO: make new FCW tests
def create_maneuvers(kwargs):
  maneuvers = [
    Maneuver(
      'approach stopped car at 25m/s, initial distance: 120m',
      duration=20.,
      initial_speed=25.,
      lead_relevancy=True,
      initial_distance_lead=120.,
      speed_lead_values=[30., 0.],
      breakpoints=[0., 1.],
      **kwargs,
    ),
    Maneuver(
      'approach stopped car at 20m/s, initial distance 90m',
      duration=20.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=90.,
      speed_lead_values=[20., 0.],
      breakpoints=[0., 1.],
      **kwargs,
    ),
    Maneuver(
      'steady state following a car at 20m/s, then lead decel to 0mph at 1m/s^2',
      duration=50.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=35.,
      speed_lead_values=[20., 20., 0.],
      breakpoints=[0., 15., 35.0],
      **kwargs,
    ),
    Maneuver(
      'steady state following a car at 20m/s, then lead decel to 0mph at 2m/s^2',
      duration=50.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=35.,
      speed_lead_values=[20., 20., 0.],
      breakpoints=[0., 15., 25.0],
      **kwargs,
    ),
    Maneuver(
      'steady state following a car at 20m/s, then lead decel to 0mph at 3m/s^2',
      duration=50.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=35.,
      speed_lead_values=[20., 20., 0.],
      breakpoints=[0., 15., 21.66],
      **kwargs,
    ),
    Maneuver(
      'steady state following a car at 20m/s, then lead decel to 0mph at 3+m/s^2',
      duration=40.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=35.,
      speed_lead_values=[20., 20., 0.],
      prob_lead_values=[0., 1., 1.],
      cruise_values=[20., 20., 20.],
      breakpoints=[2., 2.01, 8.8],
      **kwargs,
    ),
    Maneuver(
      "approach stopped car at 20m/s, with prob_lead_values",
      duration=30.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=120.,
      speed_lead_values=[0.0, 0., 0.],
      prob_lead_values=[0.0, 0., 1.],
      cruise_values=[20., 20., 20.],
      breakpoints=[0.0, 2., 2.01],
      **kwargs,
    ),
    Maneuver(
      "approach stopped car at 20m/s, with prob_throttle_values and pitch = -0.1",
      duration=30.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=120.,
      speed_lead_values=[0.0, 0., 0.],
      prob_throttle_values=[1., 0., 0.],
      cruise_values=[20., 20., 20.],
      pitch_values=[0., -0.1, -0.1],
      breakpoints=[0.0, 2., 2.01],
      **kwargs,
    ),
    Maneuver(
      "approach stopped car at 20m/s, with prob_throttle_values and pitch = +0.1",
      duration=30.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=120.,
      speed_lead_values=[0.0, 0., 0.],
      prob_throttle_values=[1., 0., 0.],
      cruise_values=[20., 20., 20.],
      pitch_values=[0., 0.1, 0.1],
      breakpoints=[0.0, 2., 2.01],
      **kwargs,
    ),
    Maneuver(
      "approach slower cut-in car at 20m/s",
      duration=20.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=50.,
      speed_lead_values=[15., 15.],
      breakpoints=[1., 11.],
      only_lead2=True,
      **kwargs,
    ),
    Maneuver(
      "stay stopped behind radar override lead",
      duration=20.,
      initial_speed=0.,
      lead_relevancy=True,
      initial_distance_lead=10.,
      speed_lead_values=[0., 0.],
      prob_lead_values=[0., 0.],
      breakpoints=[1., 11.],
      only_radar=True,
      **kwargs,
    ),
    Maneuver(
      "NaN recovery",
      duration=30.,
      initial_speed=15.,
      lead_relevancy=True,
      initial_distance_lead=60.,
      speed_lead_values=[0., 0., 0.0],
      breakpoints=[1., 1.01, 11.],
      cruise_values=[float("nan"), 15., 15.],
      **kwargs,
    ),
    Maneuver(
      'cruising at 25 m/s while disabled',
      duration=20.,
      initial_speed=25.,
      lead_relevancy=False,
      enabled=False,
      **kwargs,
    ),
  ]
  if not kwargs['e2e']:
    maneuvers.append(Maneuver(
      "slow to 5m/s with allow_throttle = False and pitch = +0.1",
      duration=30.,
      initial_speed=20.,
      lead_relevancy=False,
      prob_throttle_values=[1., 0., 0.],
      cruise_values=[20., 20., 20.],
      pitch_values=[0., 0.1, 0.1],
      breakpoints=[0.0, 2., 2.01],
      ensure_slowdown=True,
      **kwargs,
    ))
  if not kwargs['force_decel']:
    # controls relies on planner commanding to move for stock-ACC resume spamming
    maneuvers.append(Maneuver(
      "resume from a stop",
      duration=20.,
      initial_speed=0.,
      lead_relevancy=True,
      initial_distance_lead=STOP_DISTANCE,
      speed_lead_values=[0., 0., 7.],
      breakpoints=[1., 10., 15.],
      ensure_start=True,
      **kwargs,
    ))
  return maneuvers


@parameterized_class(("e2e", "force_decel", "accel_max"), itertools.product([True, False], [True, False], ACCEL_MAX_PINS))
class TestLongitudinalControl(PinnedTestCase):
  e2e: bool
  force_decel: bool

  def test_maneuver(self, subtests):
    for maneuver in create_maneuvers({"e2e": self.e2e, "force_decel": self.force_decel}):
      with subtests.test(title=maneuver.title, e2e=maneuver.e2e, force_decel=maneuver.force_decel):
        print(maneuver.title, f'in {"e2e" if maneuver.e2e else "acc"} mode')
        valid, _ = maneuver.evaluate()
        assert valid


@parameterized_class('accel_max', ACCEL_MAX_PINS)
class TestLeadAnticipation(PinnedTestCase):
  # the model's forecast of the lead, anchored to the radar lead, reaches the MPC: the stock plant's model lead holds the lead's
  # braking to a stop, while the radar's own extrapolation fades it out, so a lead braking hard is braked for earlier and with more room
  def braking(self, maneuver, anchored=True):
    # the time the car first brakes and the least gap, with the forecast, or with the planner reading the radar alone
    if anchored:
      valid, logs = maneuver.evaluate()
    else:
      with mock.patch.object(longitudinal_planner, 'anchor_model_lead', lambda *args: None):
        valid, logs = maneuver.evaluate()
    assert valid, maneuver.title
    t, a, d_rel = logs[:, 0], logs[:, 5], logs[:, 6]
    return float(t[np.flatnonzero(a < BRAKING_ONSET)[0]]), float(d_rel.min())

  def test_a_lead_braking_to_a_stop_is_braked_for_earlier_and_further_back(self, subtests):
    stops = [m for m in create_maneuvers({'e2e': False, 'force_decel': False}) if 'then lead decel to 0mph' in m.title]
    assert len(stops) == 4
    for maneuver in stops:
      with subtests.test(title=maneuver.title):
        (onset, gap), (radar_onset, radar_gap) = self.braking(maneuver), self.braking(maneuver, anchored=False)
        assert radar_onset - onset >= ANTICIPATION, (onset, radar_onset)
        assert gap - radar_gap >= ANTICIPATION_GAP, (gap, radar_gap)

  def test_a_lead_slowing_to_a_crawl_is_braked_for_earlier(self):
    maneuver = Maneuver('following a car at 15 m/s that slows to 5 m/s at 1 m/s^2', duration=30., initial_speed=15., lead_relevancy=True,
                        initial_distance_lead=30., speed_lead_values=[15., 15., 5.], breakpoints=[0., 10., 20.])
    (onset, _), (radar_onset, _) = self.braking(maneuver), self.braking(maneuver, anchored=False)
    assert radar_onset - onset >= ANTICIPATION, (onset, radar_onset)


@parameterized_class('accel_max', ACCEL_MAX_PINS)
class TestEnsureStartLaunchScope(PinnedTestCase):
  # ensure_start is a launch-phase check. Above the 2 m/s launch band a faster lead
  # plus a momentarily flat command is ordinary cruise gap settling, not a stalled launch, and
  # must not fail the maneuver the way an unscoped check would.
  def test_ensure_start_ignores_gap_settling_once_above_the_launch_band(self):
    maneuver = Maneuver(
      'cruising at 5 m/s while a lead pulls away',
      duration=5.0,
      initial_speed=5.0,
      lead_relevancy=True,
      initial_distance_lead=30.0,
      cruise_values=[5.0, 5.0],
      speed_lead_values=[5.0, 8.0],
      breakpoints=[0.0, 1.0],
      ensure_start=True,
    )
    valid, _ = maneuver.evaluate()
    assert valid


class TestManeuverHarnessLiveness(OpenpilotTestCase):
  # the planner never sees sm.all_checks(); this checks the shim itself, since
  # that is the only place a scheduled radar/model validity drop is observable
  def test_scheduled_validity_drop_visible_through_shim(self):
    maneuver = Maneuver(
      'liveness shim schedule',
      duration=0.6,
      radar_valid_breakpoints=[0.0, 0.2, 0.4],
      radar_valid_values=[1.0, 0.0, 1.0],
      model_valid_breakpoints=[0.0, 0.3],
      model_valid_values=[1.0, 0.0],
    )
    plant = Plant()
    seen_radar_drop = False
    seen_model_drop = False
    while plant.current_time < maneuver.duration:
      t = plant.current_time
      radar_valid = bool(np.interp(t, maneuver.radar_valid_breakpoints, maneuver.radar_valid_values) > 0.5)
      model_valid = bool(np.interp(t, maneuver.model_valid_breakpoints, maneuver.model_valid_values) > 0.5)
      plant.step(radar_valid=radar_valid, model_valid=model_valid)

      assert plant.last_sm.all_checks(['radarState']) == radar_valid
      assert plant.last_sm.all_checks(['modelV2']) == model_valid
      assert plant.last_sm.all_checks(['carState']) is True
      assert plant.last_sm.all_checks() == (radar_valid and model_valid)

      seen_radar_drop = seen_radar_drop or not radar_valid
      seen_model_drop = seen_model_drop or not model_valid

    assert seen_radar_drop and seen_model_drop


def hold_plan(plant, a_target, should_stop=False):
  def update(sm):
    plant.planner.output_a_target, plant.planner.output_should_stop = a_target, should_stop
  return mock.patch.object(plant.planner, 'update', update)


class TestPlant(OpenpilotTestCase):
  def test_the_radar_reports_the_lead_in_its_own_distance_steps(self):
    radar = DBC(PLATFORMS[PALISADE.carFingerprint].config.dbc_dict[Bus.radar])
    step = radar.name_to_msg[f'RADAR_TRACK_{RADAR_START_ADDR:x}'].sigs['LONG_DIST'].factor
    plant = Plant(lead_relevancy=True, distance_lead=12.34, CP=PALISADE)
    plant.step()
    d_rel = plant.last_sm['radarState'].leadOne.dRel
    assert abs(d_rel - step * round(d_rel / step)) < 1e-5 and abs(d_rel - 12.34) <= step / 2, d_rel

  def test_held_wheels_do_not_wind_up_the_actuator(self):
    # at rest the car's acceleration is zero whatever the stopping ramp asks, so a launch starts from rest behind the dead time
    rolled = []
    for actuator_lag in (False, True):
      plant = Plant(actuator_lag=actuator_lag, CP=PALISADE)
      with hold_plan(plant, 0.0, should_stop=True):
        for _ in range(40):
          plant.step(v_cruise=0.0)
      with hold_plan(plant, 0.5):
        while plant.step(v_cruise=10.0)['speed'] == 0.0 and plant.current_time < 10.0:
          pass
      rolled.append(plant.current_time)
    assert rolled[1] - rolled[0] <= max(band[0] for band in ActuatorLag.BANDS) + DT_MDL, rolled

  def test_the_planner_sees_the_cars_speed_filter_not_the_applied_acceleration(self):
    plant = Plant(speed=10.0, CP=PALISADE)
    a_ego = []
    with hold_plan(plant, -1.0):
      for _ in range(10):
        plant.step(v_cruise=10.0)
        a_ego.append(plant.last_sm['carState'].aEgo)
    assert plant.acceleration == -1.0
    assert a_ego[0] == 0.0 and -1.0 < a_ego[1] < -0.1, a_ego
    assert abs(a_ego[-1] + 1.0) < 0.05, a_ego

  def test_the_car_can_fall_short_of_its_request_and_creep_until_braked(self):
    plant = Plant(speed=10.0, CP=PALISADE, accel_error=BRAKE_SHORTFALL)
    with hold_plan(plant, -1.0):
      plant.step(v_cruise=10.0)
    assert abs(plant.acceleration - (-1.0 + BRAKE_SHORTFALL)) < 1e-9, plant.acceleration
    for request, creeps in ((CREEP_HOLD_REQUEST + 0.05, True), (CREEP_HOLD_REQUEST - 0.05, False)):
      plant = Plant(CP=PALISADE, creep=DRIVELINE_CREEP)
      with hold_plan(plant, request):
        for _ in range(10):
          plant.step(v_cruise=10.0)
      assert (plant.speed > 0.0) == creeps, (request, plant.speed)

  def test_the_stop_bit_hands_the_car_to_longcontrols_stopping_ramp(self):
    plant = Plant(speed=2.0, CP=PALISADE)
    with hold_plan(plant, -0.15):
      for _ in range(5):
        plant.step(v_cruise=0.0)
    assert plant.acceleration == -0.15
    with hold_plan(plant, -0.15, should_stop=True):
      for _ in range(10):
        plant.step(v_cruise=0.0)
    assert plant.acceleration < -0.5, plant.acceleration


@parameterized_class('accel_max', ACCEL_MAX_PINS)
class TestRedLightStop(PinnedTestCase):
  def test_a_late_red_light_stops_front_loaded_short_of_the_line(self):
    # the model calls a red light ~5 s out; the owner wants the needed deceleration reached within a second, held, and
    # eased off at the end -- never still increasing in the last metres
    maneuver = Maneuver('approach a red light at 14 m/s, seen 5 s out', duration=30.0, initial_speed=14.0,
                        cruise_values=[14.0, 14.0], e2e=True, stop_line=160.0, CP=PALISADE)
    valid, logs = maneuver.evaluate()
    assert valid
    x, v, a = logs[:, 1], logs[:, 3], logs[:, 5]
    stopped = np.flatnonzero(v < 0.05)
    assert len(stopped) > 0, 'did not stop'
    i_stop = int(stopped[0])
    assert maneuver.stop_line - 5.0 <= x[i_stop] <= maneuver.stop_line, x[i_stop]
    onset = int(np.flatnonzero(a < -0.3)[0])
    approach = a[onset:i_stop]
    peak = float(approach.min())
    assert peak >= -2.6, peak
    first_second = a[onset:onset + int(1.0 / DT_MDL)]
    assert first_second.mean() <= 0.6 * peak, (first_second.mean(), peak)        # front-loaded: most of the braking within a second
    last_second = a[max(i_stop - int(1.0 / DT_MDL), onset):i_stop]
    assert last_second.mean() >= 0.5 * peak, (last_second.mean(), peak)         # eased off at the end
    assert approach.max() <= 0.05, approach.max()                                # never lets go during the approach
    assert np.all(v[i_stop:] < 0.3), 'crept away after the stop'


def turning_points(a, threshold):
  # the plan's extrema that a move of at least threshold confirms, in order: ('min', i) is a braking peak, ('max', i) a let-off.
  # The first move is measured from the running high and low, so a gradual one counts like a step
  points, direction, high, low = [], 0, a[0], a[0]
  extreme, i_extreme = a[0], 0
  for i, x in enumerate(a[1:], 1):
    if direction == 0:
      high, low = max(high, x), min(low, x)
      if x - low >= threshold or high - x >= threshold:
        direction, extreme, i_extreme = (1 if x - low >= threshold else -1), x, i
    elif direction * (x - extreme) > 0:
      extreme, i_extreme = x, i
    elif direction * (extreme - x) >= threshold:
      points.append(('max' if direction > 0 else 'min', i_extreme))
      direction, extreme, i_extreme = -direction, x, i
  return points


def reversals_before_the_ease(plan):
  # every turning point but a last braking peak, which the final ease confirms
  points = turning_points(plan, PLAN_REVERSAL)
  return points[:-1] if points and points[-1][0] == 'min' else points


class TestTurningPoints(OpenpilotTestCase):
  def test_a_gradual_let_off_and_re_brake_counts_like_a_step(self):
    brake, hold, ease = np.linspace(-0.5, -1.6, 10), np.full(10, -1.6), np.linspace(-1.6, -0.3, 20)
    let_off = np.r_[np.linspace(-1.6, -1.3, 8), np.linspace(-1.3, -1.6, 8)]  # 0.30 each way, 0.043 per frame
    assert reversals_before_the_ease(np.r_[brake, hold, ease]) == []
    assert [kind for kind, _ in reversals_before_the_ease(np.r_[brake, hold, let_off, hold, ease])] == ['min', 'max']
    assert [kind for kind, _ in reversals_before_the_ease(np.r_[brake, hold, let_off, hold])] == ['min', 'max']


def braking_window(logs, initial_speed):
  # the planner's target from the first frame it brakes while the car is above 1 m/s; each target was planned at the previous
  # row's speed
  v, a_target = logs[:, 3], logs[:, 7]
  v_planned = np.concatenate(([initial_speed], v[:-1]))
  onset = int(np.flatnonzero(a_target < -0.3)[0])
  return onset, onset + int(np.flatnonzero(v_planned[onset:] > 1.0)[-1]) + 1


def assert_the_plan_builds_to_the_stop_without_steps_or_let_offs(plan, onset):
  steps = np.abs(np.diff(plan))
  assert steps.max() <= PLAN_JERK_MAX * DT_MDL, (steps.max(), onset + int(np.argmax(steps)) + 1)
  reversals = reversals_before_the_ease(plan)
  assert not reversals, [(kind, onset + i, round(float(plan[i]), 3)) for kind, i in reversals]


class RedLightPlan(PinnedTestCase):
  # the model calls the line 5 s out after 6.5 s of cruise. The plan judged is the planner's target from that call while the car
  # is above 1 m/s; the call itself is the fake model's instantaneous step, so the window opens after it. Force Stops commits
  # inside the window, so its commit is one of the steps judged
  speed: float
  actuator_lag: bool

  def setUp(self):
    super().setUp()
    maneuver = Maneuver('approach a red light, seen 5 s out', duration=40.0, initial_speed=self.speed, cruise_values=[self.speed, self.speed],
                        e2e=True, stop_line=11.5 * self.speed, actuator_lag=self.actuator_lag, CP=PALISADE)
    valid, logs = maneuver.evaluate()
    assert valid
    x, v, a_target, committed = logs[:, 1], logs[:, 3], logs[:, 7], logs[:, 8] > 0.5
    stopped = np.flatnonzero(v < 0.05)
    assert len(stopped) > 0, 'did not stop'
    assert maneuver.stop_line - 5.0 <= x[int(stopped[0])] <= maneuver.stop_line, x[int(stopped[0])]
    self.onset, end = braking_window(logs, self.speed)
    assert committed.any(), 'no commit'
    self.commit = int(np.flatnonzero(committed)[0])
    assert self.onset < self.commit < end, (self.onset, self.commit, end)
    self.plan = a_target[self.onset:end]


@parameterized_class(('speed', 'accel_max'), itertools.product([14.0, 17.0, 20.0], ACCEL_MAX_PINS))
class TestRedLightPlanContinuity(RedLightPlan):
  actuator_lag = True  # the car's nominal response

  def test_the_plan_builds_to_the_stop_without_steps_or_let_offs(self):
    assert_the_plan_builds_to_the_stop_without_steps_or_let_offs(self.plan, self.onset)


@parameterized_class('accel_max', ACCEL_MAX_PINS)
class TestRedLightPlanContinuityAtTheColumnsNeed(RedLightPlan):
  # through the actuator the car is still at 21.3 m/s when Force Stops commits, and the stop column's first plan there is its
  # converged answer, 0.206 deeper than the model's request (measured at both pins): a real need step, not a solver transient
  speed = 22.0
  actuator_lag = True
  need_step = 0.206

  def test_only_the_commit_steps_beyond_the_bound(self):
    steps = np.diff(self.plan)
    beyond = self.onset + 1 + np.flatnonzero(np.abs(steps) > PLAN_JERK_MAX * DT_MDL)
    assert list(beyond) == [self.commit], (list(beyond), self.commit)
    # it brakes, by no more than the need and the 0.1 a first solve may sit off converged (the MPC's own gate)
    step = steps[self.commit - self.onset - 1]
    assert -(self.need_step + 0.1) <= step < 0.0, step
    assert not reversals_before_the_ease(self.plan)

  # only a stop law that owns the commit, with no stop column while it drives, removes the column's need step
  @unittest.expectedFailure
  def test_the_plan_builds_to_the_stop_without_steps_or_let_offs(self):
    assert_the_plan_builds_to_the_stop_without_steps_or_let_offs(self.plan, self.onset)


@parameterized_class(('speed', 'accel_max'), itertools.product([14.0, 17.0, 20.0, 22.0], ACCEL_MAX_PINS))
class TestRedLightPlanContinuityOnAnIdealActuator(RedLightPlan):
  # an ideal actuator applies the call at once and the car's speed filter rings on that step: a commit entered from the
  # measured acceleration would step with the ring
  actuator_lag = False

  def test_the_plan_builds_to_the_stop_without_steps_or_let_offs(self):
    assert_the_plan_builds_to_the_stop_without_steps_or_let_offs(self.plan, self.onset)


def landing_excess(logs, lead=False, v_min=KISS_SPEED, v_max=LANDING_SPEED):
  # the most the car's braking exceeded the landing corridor through the last metres (KISS_SPEED .. LANDING_SPEED).
  # Below the kiss speed the corridor is the kiss plus the anti-stall integral by design, and the aEgo checks judge that end.
  # Each row's plan was computed from the previous row's state (the plant logs after integrating), so the corridor is judged
  # at that speed and gap. With a lead, the braking that stopping STOP_DISTANCE behind it needs always passes
  v, a, d_rel, v_lead = logs[:, 3], logs[:, 5], logs[:, 6], logs[:, 4]
  excess = 0.0
  for i in range(1, len(v)):
    v_seen, d_seen, v_lead_seen = v[i - 1], d_rel[i - 1], v_lead[i - 1]
    if not (v_min <= v_seen < v_max) or a[i] >= 0.0:
      continue
    allowed = landing_bound(v_seen)
    if lead:
      closing = max(v_seen - v_lead_seen, 0.0)
      allowed = max(allowed, closing ** 2 / (2.0 * max(d_seen - STOP_DISTANCE, 0.5)))
    excess = max(excess, -a[i] - allowed)
  return excess


@parameterized_class('accel_max', ACCEL_MAX_PINS)
class TestStopLanding(PinnedTestCase):
  def test_a_stopped_lead_is_landed_within_the_law_and_the_car_still_stops_behind_it(self):
    maneuver = Maneuver('approach a stopped lead at 10 m/s', duration=25.0, initial_speed=10.0, lead_relevancy=True,
                        initial_distance_lead=90.0, speed_lead_values=[0.0, 0.0], cruise_values=[10.0, 10.0], CP=PALISADE)
    valid, logs = maneuver.evaluate()
    assert valid
    v, d_rel = logs[:, 3], logs[:, 6]
    assert np.any(v < 0.05), 'did not stop'
    assert d_rel[-1] >= LEAD_REST_GAP, d_rel[-1]
    assert landing_excess(logs, lead=True) <= 0.02

  def test_a_red_light_is_landed_within_the_law(self):
    maneuver = Maneuver('approach a red light at 14 m/s, seen 5 s out', duration=30.0, initial_speed=14.0,
                        cruise_values=[14.0, 14.0], e2e=True, stop_line=160.0, CP=PALISADE)
    valid, logs = maneuver.evaluate()
    assert valid
    x, v = logs[:, 1], logs[:, 3]
    stopped = np.flatnonzero(v < 0.05)
    assert len(stopped) > 0, 'did not stop'
    assert maneuver.stop_line - 5.0 <= x[int(stopped[0])] <= maneuver.stop_line
    assert landing_excess(logs) <= 0.02

  def test_a_lead_braking_hard_close_ahead_is_never_softened_into(self):
    # the law only removes surplus braking: with a lead stopping 25 m ahead from 8 m/s the physics floor keeps the
    # deceleration the gap needs, and the car lands behind the lead without a crash
    maneuver = Maneuver('lead 25 m ahead brakes to a stop from 8 m/s', duration=20.0, initial_speed=8.0, lead_relevancy=True,
                        initial_distance_lead=25.0, breakpoints=[0.0, 1.0, 3.5, 20.0], speed_lead_values=[8.0, 8.0, 0.0, 0.0],
                        cruise_values=[8.0, 8.0, 8.0, 8.0], CP=PALISADE)
    valid, logs = maneuver.evaluate()
    assert valid, 'crashed'
    v, d_rel = logs[:, 3], logs[:, 6]
    assert np.any(v < 0.05), 'did not stop'
    assert d_rel.min() >= LEAD_CLOSEST_GAP, d_rel.min()
    assert landing_excess(logs, lead=True) <= 0.02

  def test_the_models_late_ramp_is_bounded_and_the_car_still_stops_before_the_line(self):
    # the model's request ramps hard into the last metres of a stop the car is already landing. The law bounds that landing
    # to its taper; the stop still lands short of the line
    kwargs = {'duration': 30.0, 'initial_speed': 14.0, 'cruise_values': [14.0, 14.0], 'e2e': True, 'stop_line': 160.0, 'e2e_landing_push': 3.5,
              'CP': PALISADE}
    valid, logs = Maneuver('red light with a late model ramp', **kwargs).evaluate()
    assert valid
    x, v = logs[:, 1], logs[:, 3]
    stopped = np.flatnonzero(v < 0.05)
    assert len(stopped) > 0, 'did not stop'
    assert kwargs['stop_line'] - 5.0 <= x[int(stopped[0])] <= kwargs['stop_line'], x[int(stopped[0])]
    assert landing_excess(logs) <= 0.02
    # the same ramp with the law bypassed lands well outside it: the test is about the law, not the plant
    original = StopLanding.update
    StopLanding.update = lambda self, a_target, *args, **kwargs: a_target
    try:
      _, unbounded = Maneuver('red light with a late model ramp, no law', **kwargs).evaluate()
    finally:
      StopLanding.update = original
    assert landing_excess(unbounded) >= 0.5, landing_excess(unbounded)

  def test_a_car_that_lets_go_of_the_brake_slowly_still_lands_close_to_the_law(self):
    # through the Palisade's actuator the car may not brake much harder than the corridor allows in the last metres, and it
    # still stops behind the lead
    kwargs = {'duration': 25.0, 'initial_speed': 10.0, 'lead_relevancy': True, 'initial_distance_lead': 90.0,
              'speed_lead_values': [0.0, 0.0], 'cruise_values': [10.0, 10.0], 'actuator_lag': True, 'CP': PALISADE}
    valid, logs = Maneuver('stopped lead through a slow-release actuator', **kwargs).evaluate()
    assert valid
    v, a, d_rel = logs[:, 3], logs[:, 5], logs[:, 6]
    assert np.any(v < 0.05), 'did not stop'
    assert d_rel[-1] >= LEAD_REST_GAP
    # below ~0.5 m/s the corridor is already at the kiss and the lagged car is by design still catching up, so the
    # corridor-excess check applies above it; the low end is judged by what matters -- the deceleration still on the
    # car as the wheels are about to stop, which rocks the body
    assert landing_excess(logs, lead=True, v_min=0.5) <= 0.15, landing_excess(logs, lead=True, v_min=0.5)
    # below the kiss speed the anti-stall integral may exceed the bound, by at most its cap
    assert landing_excess(logs, lead=True, v_min=0.3, v_max=KISS_SPEED) <= STALL_MAX + 0.02
    last_rolling = int(np.flatnonzero(v >= 0.15)[-1])
    assert a[last_rolling] >= -0.25, a[last_rolling]


@parameterized_class(('speed', 'distance_lead', 'accel_max'), [(v, d, p) for v, d in ((10.0, 90.0), (15.0, 120.0)) for p in ACCEL_MAX_PINS])
class TestACarThatBrakesLessThanAsked(PinnedTestCase):
  # the car rolls with 0.3 m/s^2 more than it is asked from cruise to rest, as brakes that fall short or a downhill the planner does
  # not see would. The MPC must plan from the speed the car has, not the speed it would have had if it had done as told
  speed: float
  distance_lead: float

  def setUp(self):
    super().setUp()
    maneuver = Maneuver('approach a stopped lead braking short of the request', duration=25.0, initial_speed=self.speed, lead_relevancy=True,
                        initial_distance_lead=self.distance_lead, speed_lead_values=[0.0, 0.0], cruise_values=[self.speed, self.speed],
                        actuator_lag=True, CP=PALISADE, accel_error=BRAKE_SHORTFALL)
    valid, self.logs = maneuver.evaluate()
    assert valid, 'crashed'
    assert np.any(self.logs[:, 3] < 0.05), 'did not stop'

  def test_it_stops_clear_of_a_stopped_lead_without_steps_or_let_offs(self):
    assert self.logs[:, 6].min() >= LEAD_CLOSEST_GAP, self.logs[:, 6].min()
    onset, end = braking_window(self.logs, self.speed)
    assert_the_plan_builds_to_the_stop_without_steps_or_let_offs(self.logs[onset:end, 7], onset)

  # the landing's edges assume the car does as asked, and its one feedback, the anti-stall integral, closes on the deceleration (the
  # kiss less its deadband), not on the gap: a car 0.3 short is asked about -0.42, rolls the last metres in at -0.10 and rests ~3 m back
  @unittest.expectedFailure
  def test_it_rests_at_least_the_rest_gap_behind_it(self):
    assert self.logs[-1, 6] >= LEAD_REST_GAP, self.logs[-1, 6]


# (the lead leaves once the landing is latched at or below this speed, 0: after 1 s at rest; creep; the lead's acceleration and speed).
# Without creep the landing ends rolling on the climbing plan, or at rest on the launch; with the assumed creep, at walking pace
LEAD_LEAVES = [(1.0, 0.0, 2.0, 5.0), (0.0, 0.0, 0.5, 1.0), (0.0, 0.0, 2.0, 5.0), (0.6, DRIVELINE_CREEP, 0.5, 1.0), (0.6, DRIVELINE_CREEP, 2.0, 5.0)]


@parameterized_class(('leaves_at', 'creep', 'lead_accel', 'lead_speed', 'accel_max'), [(*case, p) for case in LEAD_LEAVES for p in ACCEL_MAX_PINS])
class TestALeadLeavingALatchedLanding(PinnedTestCase):
  # the lead pulls away or inches forward while the landing is latched, with the car still rolling or at rest. Every frame that
  # starts with the target held below the plan the planner gave the landing rises at most the cruise jerk, however the hold ends,
  # but the frame a launch commits at rest: that one steps to the launch level while the ESC holds the wheels, and the lift goes
  # back at the cruise jerk from then on, braking at once
  leaves_at: float
  creep: float
  lead_accel: float
  lead_speed: float

  def test_the_landing_hands_back_what_it_held_at_the_cruise_jerk(self):
    plant = Plant(lead_relevancy=True, speed=10.0, distance_lead=70.0, actuator_lag=True, CP=PALISADE, creep=self.creep)
    landing, frames_at_rest = plant.planner.stop_landing, 0
    with mock.patch.object(landing, 'update', wraps=landing.update) as law:
      while not (landing.landing and (plant.speed <= self.leaves_at if self.leaves_at > 0.0 else frames_at_rest >= round(1.0 / DT_MDL))):
        frames_at_rest = frames_at_rest + 1 if plant.step(v_cruise=10.0)['speed'] == 0.0 else 0
        assert plant.current_time < 30.0, 'never landed'
      t_departed, v_lead, handed_back, commits = plant.current_time, 0.0, [], 0
      while plant.current_time < t_departed + 8.0:
        v_ego, a_prev, plan_prev, lifted = plant.speed, plant.a_target, law.call_args.args[0], plant.a_target > plant.planner.plan_a_target
        j_cruise = float(np.interp(v_ego, J_CRUISE_BP, J_CRUISE_VALS)) * DT_MDL
        v_lead = min(v_lead + self.lead_accel * DT_MDL, self.lead_speed)
        plant.step(v_lead=v_lead, v_cruise=10.0)
        if plant.planner.standstill_exit.committed:
          commits += 1
          assert plant.a_target == max(plant.planner.plan_a_target, LAUNCH_ACCEL), (plant.a_target, plant.planner.plan_a_target)
        elif a_prev < plan_prev:
          handed_back.append((plant.a_target - a_prev - j_cruise, plant.current_time - t_departed))
        if lifted:
          assert plant.a_target >= a_prev - j_cruise - 1e-9 or plant.a_target == plant.planner.plan_a_target, (a_prev, plant.a_target)
      assert handed_back, 'nothing was held'
      assert commits == 1 if self.leaves_at == 0.0 else commits <= 1, commits
      worst = max(handed_back)
      assert worst[0] <= 1e-9, f'{worst[0]:+.4f} over the cruise jerk {worst[1]:.2f} s after the lead left'
      assert plant.a_target == law.call_args.args[0] and plant.speed > 0.5, (plant.a_target, law.call_args.args[0], plant.speed)


def without_the_exit():
  # the planner with its standstill exit taken out: nothing decided at rest, the stop bit the candidates' own, the plan published
  return mock.patch.multiple(StandstillExit, update=lambda self, *args: None, apply=lambda self, a_target, v_ego: a_target)


@parameterized_class(('esc', 'accel_max'), itertools.product(EscStandstill.FORMS, ACCEL_MAX_PINS))
class TestStandstillLaunch(PinnedTestCase):
  # at rest the stop bit is a hold or a committed launch, through the car's ESC standstill brake under both of its fits: the
  # planner's releases turn the wheels sooner than without the exit, and a release withdrawn before the ESC lets go costs no motion
  esc: str
  REST_GAP = 5.5  # m, behind a standing lead
  LINE = 60.0  # m, a red light's line ahead of a car at 10 m/s

  def drive(self, seconds, script=lambda plant, t: {}, lead=True, track=None, edit=None, **kwargs):
    # the car on its plant; script(plant, t) gives each frame's step arguments, track() the radar track radarState reports the
    # plant's lead on (default: its own), edit(plant, sm) rewrites the frame's messages after that. Returns per frame: time, speed,
    # distance, gap, stop bit, StopReq, launch committed, launch taken back
    plant = Plant(lead_relevancy=lead, distance_lead=self.REST_GAP, actuator_lag=True, CP=PALISADE, esc=self.esc, **kwargs)
    if track is not None or edit is not None:
      update = plant.planner.update

      def tracked(sm, update=update):
        if track is not None:
          for slot in (sm['radarState'].leadOne, sm['radarState'].leadTwo):
            slot.radarTrackId = track()
        if edit is not None:
          edit(plant, sm)
        return update(sm)
      plant.planner.update = tracked
    rows = []
    while plant.current_time < seconds:
      step = {'v_cruise': 10.0, **script(plant, plant.current_time)}
      plant.step(**step)
      standstill_exit = plant.planner.standstill_exit
      rows.append((plant.current_time, plant.speed, plant.distance, plant.distance_lead - plant.distance, plant.should_stop,
                   plant.long_control.long_control_state == LongCtrlState.stopping, standstill_exit.committed, standstill_exit.taken_back))
    return np.array(rows, dtype=float)

  def at(self, rows, mask):
    found = np.flatnonzero(mask)
    return float(rows[found[0], 0]) if len(found) else None

  def test_a_departing_lead_turns_the_wheels_within_1_6_s_of_the_release(self):
    # the lead standing 5.5 m ahead pulls away at a jerk of 0.35 m/s^3 from 0.5 s: the stop bit falls on the pre-release's
    # release and stays down, the wheels turn within 1.6 s of it, and the car reaches 1 m/s sooner than without the exit
    def departing(plant, t):
      return {'v_lead': 0.5 * 0.35 * max(t - 0.5, 0.0) ** 2}
    rows = self.drive(8.0, departing)
    with without_the_exit():
      before = self.drive(8.0, departing)
    release = self.at(rows, rows[:, 4] == 0.0)
    assert release is not None and release == self.at(rows, rows[:, 6] == 1.0)
    wheels = self.at(rows, rows[:, 1] > 0.0)
    assert wheels - release <= 1.6, (release, wheels)
    after = rows[:, 0] >= release
    assert not np.any(rows[after, 5]) and not np.any(rows[after, 4]), 'StopReq after the commit'
    assert self.at(rows, rows[:, 1] >= 1.0) < self.at(before, before[:, 1] >= 1.0)

  def test_a_false_forecast_on_a_standing_lead_does_not_move_the_car(self):
    # the model sees the standing lead leave for 1.0 s: the pre-release lets the car go and takes it back before the ESC lets go
    rows = self.drive(8.0, lambda plant, t: {'forecast': 0.5 if 2.0 <= t < 3.0 else None})
    assert rows[:, 6].sum() == 1 and rows[:, 7].sum() == 1, (rows[:, 6].sum(), rows[:, 7].sum())
    assert np.all(rows[:, 1] == 0.0) and rows[-1, 4]

  def test_a_forecast_on_a_lead_that_never_moves_inches_once(self):
    # the model keeps seeing the standing lead leave: the car inches toward it once, by at most 0.5 m, and stays held after
    rows = self.drive(20.0, lambda plant, t: {'forecast': 0.5 if t >= 2.0 else None})
    moving = rows[:, 1] > 0.0
    assert np.sum(moving[1:] & ~moving[:-1]) <= 1 and rows[-1, 2] <= 0.5, rows[-1, 2]
    assert rows[:, 3].min() >= self.REST_GAP - 0.6, rows[:, 3].min()
    rest = rows[:, 0] >= rows[-1, 0] - 5.0
    assert not np.any(moving[rest]) and np.all(rows[rest, 4])

  def red_light(self, green=lambda t, rest: False, chill=lambda t, rest: False, seconds=6.0):
    # Experimental, the model calls a red light 60 m ahead of the car at 10 m/s and Force Stops holds the car at the line;
    # green(t, rest) says the light reads green, chill(t, rest) that the mode is Chill, with rest the time the car came to rest.
    # Returns the rows from rest on and the rest time
    state = {'rest': None}

    def script(plant, t):
      if state['rest'] is None and plant.speed == 0.0:
        state['rest'] = t
      rest = state['rest']
      plant.e2e = not (rest is not None and chill(t, rest))
      return {'green': rest is not None and green(t, rest)}
    rows = self.drive(30.0, script, lead=False, speed=10.0, e2e=True, stop_line=self.LINE)
    rest = state['rest']
    assert rest is not None and rest + seconds < 30.0
    return rows[rows[:, 0] > rest], rest

  def test_a_green_after_a_force_stops_hold_turns_the_wheels_within_1_5_s(self):
    rows, rest = self.red_light(lambda t, rest: t >= rest + 3.0)
    green = round((rest + 3.0) / DT_MDL) * DT_MDL
    assert np.all(rows[rows[:, 0] <= green, 1] == 0.0) and np.all(rows[rows[:, 0] <= green, 4])
    wheels = self.at(rows, rows[:, 1] > 0.0)
    assert wheels is not None and wheels - green <= 1.5, wheels - green

  def test_a_flash_of_open_road_at_a_red_does_not_move_the_car(self):
    # 0.35 s of open road under the hold: Force Stops lets go and the launch commits, the model's stop comes back and takes it back
    rows, _ = self.red_light(lambda t, rest: rest + 3.0 <= t < rest + 3.35)
    assert rows[:, 6].sum() == 1 and rows[:, 7].sum() == 1
    assert np.all(rows[:, 1] == 0.0) and rows[-1, 4]

  def test_a_cem_re_entry_with_the_models_stop_takes_the_launch_back(self):
    # Chill from a second into the hold; the light turns green and the launch commits, and 0.5 s later CEM re-enters Experimental
    # with the model calling its stop again: the launch is taken back and the car does not move
    rows, rest = self.red_light(lambda t, rest: t >= rest + 3.0, lambda t, rest: t >= rest + 1.0)
    commit = self.at(rows, rows[:, 6] == 1.0)
    assert commit is not None
    rows, _ = self.red_light(lambda t, rest: rest + 3.0 <= t < commit + 0.5, lambda t, rest: rest + 1.0 <= t < commit + 0.5)
    assert self.at(rows, rows[:, 6] == 1.0) == commit and rows[:, 7].sum() == 1
    assert commit + 0.5 < self.at(rows, rows[:, 7] == 1.0) <= commit + 0.5 + 2 * DT_MDL
    assert np.all(rows[:, 1] == 0.0) and rows[-1, 4]

  def car_appearing(self, release, missing=lambda t: False, gap=1.0, after=0.1, **kwargs):
    # release(plant, t) gives the step arguments that release the car at rest; after s after the launch commits, while the ESC
    # still holds the wheels, a car appears standing gap ahead and stays, and the model does not react to it; missing(t) says the
    # radar and the model miss it t s after it appeared. Returns the rows and the times of the commit and of the car's appearance
    state = {'commit': None, 'appear': None}

    def script(plant, t):
      if state['commit'] is None and plant.planner.standstill_exit.committed:
        state['commit'] = t
      if state['commit'] is not None and state['appear'] is None and t >= state['commit'] + after - 1e-6:
        state['appear'] = t
        plant.distance_lead = plant.distance + gap
      if state['appear'] is not None:
        plant.lead_relevancy = not missing(t - state['appear'])
      return release(plant, t)
    rows = self.drive(30.0, script, **kwargs)
    assert state['appear'] is not None and state['appear'] + 4.0 < 30.0
    return rows, state['commit'], state['appear']

  def assert_taken_back_without_motion(self, rows, commit, appear):
    # the launch is taken back on the frame the MPC stops for the car: StopReq within 0.1 s of it, and the wheels never turn
    after = rows[:, 0] > appear
    assert np.all(rows[rows[:, 0] > commit, 1] == 0.0), 'moved'
    assert rows[after, 7].sum() == 1 and self.at(rows[after], rows[after, 5] == 1.0) - appear <= 0.1 + 1e-6
    assert np.all(rows[rows[:, 0] > appear + 0.1 + 1e-6, 4])

  def green(self):
    # Experimental, Force Stops holds the car at a red; the light turns green 3 s into the rest and the launch commits
    state = {'rest': None}

    def green(plant, t):
      if state['rest'] is None and plant.speed == 0.0:
        state['rest'] = t
      return {'green': state['rest'] is not None and t >= state['rest'] + 3.0}
    return green, {'lead': False, 'speed': 10.0, 'e2e': True, 'stop_line': self.LINE}

  @staticmethod
  def gone(plant, t):
    # held at rest behind the lead standing 5.5 m ahead, which leaves the radar 2 s in while the model plans open road: the
    # pre-release hands its hold back and the launch commits
    if 2.0 - 1e-6 <= t < 2.0 + 0.5 * DT_MDL:
      plant.lead_relevancy = False
    return {'green': t >= 2.0 - 1e-6}

  def test_a_car_appearing_after_a_green_takes_the_launch_back_without_motion(self):
    release, kwargs = self.green()
    self.assert_taken_back_without_motion(*self.car_appearing(release, **kwargs))

  def test_a_car_appearing_after_a_hand_back_takes_the_launch_back_without_motion(self):
    self.assert_taken_back_without_motion(*self.car_appearing(self.gone))

  def test_a_car_appearing_at_the_stop_distance_late_in_the_hold_does_not_move_the_car(self):
    # after a green and after a hand-back, a car appears standing 6 m ahead, inside the MPC's stop distance, 0.7 s after the
    # commit, a few frames before the ESC lets go: the launch is taken back on the frame it appears, not once the MPC's plan has
    # fallen to its stop, and the wheels never turn
    green, kwargs = self.green()
    for gap in (6.0, 6.9):
      for release, kw in ((green, kwargs), (self.gone, {})):
        rows, commit, appear = self.car_appearing(release, gap=gap, after=0.7, **kw)
        assert np.all(rows[rows[:, 0] > commit, 1] == 0.0), (gap, 'moved')
        after = rows[:, 0] > appear - 1e-6
        assert rows[after, 7].sum() == 1 and self.at(rows[after], rows[after, 7] == 1.0) - appear <= DT_MDL + 1e-6, gap
        assert np.all(rows[rows[:, 0] > appear + 0.1 + 1e-6, 5]), gap

  def cut_in(self, then):
    # Experimental, Force Stops holds the car at a red and the light turns green; 0.1 s after the launch commits a car appears
    # standing 3 m ahead on radar track 21, and from a second later then(plant, s) gives the step arguments s s on. Returns the rows
    # and the time it appeared
    green, kwargs = self.green()
    state = {'commit': None, 'appear': None, 'track': 21}

    def script(plant, t):
      # the launch's commit, or without the exit the stop bit falling at the green
      step = green(plant, t)
      if state['commit'] is None and step['green'] and (plant.planner.standstill_exit.committed or not plant.planner.output_should_stop):
        state['commit'] = t
      if state['commit'] is not None and state['appear'] is None and t >= state['commit'] + 0.1 - 1e-6:
        state['appear'] = t
        plant.lead_relevancy = True
        plant.distance_lead = plant.distance + 3.0
      if state['appear'] is not None and t >= state['appear'] + 1.0 - 1e-6:
        step.update(then(plant, t - state['appear'] - 1.0, state))
      return step
    rows = self.drive(30.0, script, track=lambda: state['track'], **kwargs)
    assert state['appear'] is not None and state['appear'] + 6.0 < 30.0
    return rows, state['appear']

  def test_a_car_cutting_in_that_drives_off_launches_on_the_pre_releases_release(self):
    # the car that took the green's launch back drives off at 1 m/s: the pre-release releases it on its first frame moving and
    # the launch commits on that release, where without the exit the car stays on the brake under the MPC's hover
    rows, appear = self.cut_in(lambda plant, s, state: {'v_lead': 1.0})
    with without_the_exit():
      exit_out, appear_out = self.cut_in(lambda plant, s, state: {'v_lead': 1.0})
    after = rows[rows[:, 0] > appear + 1e-6]
    assert after[:, 7].sum() == 1 and np.all(after[after[:, 0] < appear + 1.0, 1] == 0.0)
    commits = after[after[:, 6] == 1.0, 0]
    assert len(commits) == 1 and appear + 1.0 - 1e-6 <= commits[0] <= appear + 1.0 + DT_MDL + 1e-6, commits
    wheels = self.at(after, after[:, 1] > 0.0)
    assert wheels is not None and wheels - commits[0] <= 1.6, (wheels, commits[0])
    assert after[:, 3].min() >= 3.0 - 0.05, after[:, 3].min()
    exit_out = exit_out[exit_out[:, 0] > appear_out + 1e-6]
    exit_out_wheels = self.at(exit_out, exit_out[:, 1] > 0.0)
    assert exit_out_wheels is None or exit_out_wheels - appear_out > wheels - appear, (exit_out_wheels, wheels)

  def test_a_car_that_crosses_with_a_car_standing_beyond_does_not_move_the_car(self):
    # the car that took the green's launch back leaves the radar sideways and a car standing 9 m ahead takes its slot on radar
    # track 22: no owner releases the car again, so it stays held, StopReq up, until the driver acts
    def crossing(plant, s, state):
      if s < 0.5 * DT_MDL:
        plant.distance_lead = plant.distance + 9.0
        state['track'] = 22
      return {}
    rows, appear = self.cut_in(crossing)
    assert np.all(rows[rows[:, 0] > appear, 1] == 0.0), 'moved'
    assert np.all(rows[rows[:, 0] > appear + 0.1 + 1e-6, 5]) and rows[rows[:, 0] > appear, 6].sum() == 0

  def test_a_car_taking_the_slot_under_a_standing_release_does_not_move_the_car_while_the_radar_misses_it(self):
    # held at rest behind the lead 5.5 m ahead (radar track 7), which pulls away at 1 m/s^2 from 1 s: the pre-release's release
    # commits. 0.1 s later a car the model does not see (radar track 11) takes leadOne standing 1 or 2 m ahead, the leaving lead in
    # leadTwo; from 0.3 s after it appears the radar misses it on alternate frames for 1 s and then for 1 s in a row, leadOne reading
    # the leaving lead beyond it. The pre-release holds the car behind it through every miss, so the wheels do not turn and the stop
    # bit and StopReq stay up, as without the exit. 3 s after it appeared it drives off at 1 m/s^2: the launch commits on the
    # pre-release's release for it, its first frame above LEAD_DEPARTURE_SPEED, and the wheels turn within 1.6 s of that
    accel = 1.0

    def run(gap):
      state = {'commit': None, 'appear': None, 'x': 0.0, 'v': 0.0}

      def script(plant, t):
        # the leaving lead's launch: the exit's commit, or without the exit the stop bit falling
        if state['commit'] is None and t > 1.0 and (plant.planner.standstill_exit.committed or not plant.planner.output_should_stop):
          state['commit'] = t
        if state['commit'] is not None and state['appear'] is None and t >= state['commit'] + 0.1 - 1e-6:
          state['appear'], state['x'] = t, plant.distance + gap
        if state['appear'] is not None and t >= state['appear'] + 3.0 - 1e-6:
          state['v'] += accel * DT_MDL
          state['x'] += state['v'] * DT_MDL
        return {'v_lead': accel * max(t - 1.0, 0.0)}

      def edit(plant, sm):
        lead_one = sm['radarState'].leadOne
        lead_one.radarTrackId = 7
        sm['radarState'].leadTwo.radarTrackId = 7
        j = None if state['appear'] is None else round((plant.current_time - state['appear'] - 0.3) / DT_MDL)
        if j is not None and not (0 <= j < 20 and j % 2 == 0) and not 20 <= j < 40:
          lead_one.dRel, lead_one.vLead, lead_one.vLeadK, lead_one.aLeadK = state['x'] - plant.distance, state['v'], state['v'], 0.0
          lead_one.modelProb, lead_one.radarTrackId = 0.0, 11
      rows = self.drive(10.0, script, edit=edit)
      return rows, state['appear']

    for gap in (1.0, 2.0):
      rows, appear = run(gap)
      with without_the_exit():
        exit_out, appear_out = run(gap)
      standing = (rows[:, 0] > appear + 1e-6) & (rows[:, 0] < appear + 3.0 - 1e-6)
      assert np.all(rows[standing, 1] == 0.0) and np.all(rows[standing, 4]), (gap, 'moved or the bit fell')
      assert np.all(rows[standing & (rows[:, 0] > appear + 0.1 + 1e-6), 5]), (gap, 'StopReq fell')
      assert np.all(exit_out[(exit_out[:, 0] > appear_out) & (exit_out[:, 0] < appear_out + 3.0 - 1e-6), 1] == 0.0), (gap, 'exit out moved')
      after = rows[rows[:, 0] > appear + 1e-6]
      commits = after[after[:, 6] == 1.0, 0]
      assert len(commits) == 1 and appear + 3.0 < commits[0] <= appear + 3.0 + LEAD_DEPARTURE_SPEED / accel + 2 * DT_MDL, (gap, commits)
      wheels = self.at(after, after[:, 1] > 0.0)
      assert wheels is not None and wheels - commits[0] <= 1.6, (gap, wheels, commits[0])

  def test_a_forecast_led_departure_under_radar_misses_turns_the_wheels_no_later_than_without_the_exit(self):
    # Chill, landed behind a lead 30 m on that inches to 8.8 m; then the model forecasts it leaving while the radar reads it creeping
    # at 0.1 m/s and misses it on every other frame from four frames after the release: the forecast on the frames it is seen keeps
    # the pre-release's release, nothing is taken back, and the wheels turn no later than without the exit
    def run(exit_on=True):
      with contextlib.nullcontext() if exit_on else without_the_exit():
        plant = Plant(lead_relevancy=True, speed=6.0, distance_lead=30.0, actuator_lag=True, CP=PALISADE, esc=self.esc)
        while plant.speed > 0.0 or plant.current_time < 1.0:
          plant.step(v_lead=0.0, v_cruise=10.0)
          assert plant.current_time < 30.0
        while plant.distance_lead - plant.distance < 8.8:
          plant.step(v_lead=0.2, v_cruise=10.0)
          assert plant.current_time < 60.0
        start, rows = plant.current_time, []
        for k in range(round(4.0 / DT_MDL)):
          plant.lead_relevancy = k < 10 or k % 2 == 0
          plant.step(v_lead=0.1, v_cruise=10.0, forecast=0.6)
          rows.append((plant.current_time - start, plant.speed, plant.planner.standstill_exit.taken_back))
      return np.array(rows, dtype=float)
    rows, exit_out = run(), run(exit_on=False)
    wheels, exit_out_wheels = self.at(rows, rows[:, 1] > 0.0), self.at(exit_out, exit_out[:, 1] > 0.0)
    assert rows[:, 2].sum() == 0 and wheels is not None and wheels <= 1.6, wheels
    assert exit_out_wheels is None or wheels <= exit_out_wheels, (wheels, exit_out_wheels)

  def test_a_car_that_appeared_holds_the_car_while_the_radar_misses_it(self):
    # after a green and after a hand-back: the car that took the launch back is missed by the radar and the model for 2 s from
    # 0.3 s after it appeared, while the model plans open road. A car missing has not gone: StopReq stays up and nothing moves
    green, kwargs = self.green()
    for release, kw in ((green, kwargs), (self.gone, {})):
      rows, commit, appear = self.car_appearing(release, missing=lambda t: 0.3 - 1e-6 <= t < 2.3 - 1e-6, **kw)
      self.assert_taken_back_without_motion(rows, commit, appear)
      assert np.all(rows[rows[:, 0] > appear + 0.1 + 1e-6, 5]), 'StopReq fell'

  def test_a_radar_dropout_in_the_pre_releases_cancel_window_does_not_move_the_car(self):
    # landed behind a lead standing 30 m on, which inches to 8.8 m; the model forecasts it leaving for 0.3 s and the pre-release
    # lets the car go, then the forecast fades while the lead stands, and the radar and the model miss it on the fade's second
    # frame, which clears the pre-release's cancel window without a cancel. Chill, and Experimental with the model asking 0.0 at
    # rest: the launch is taken back before the ESC lets go, and the car is held
    for experimental in (False, True):
      plant = Plant(lead_relevancy=True, speed=6.0, distance_lead=30.0, actuator_lag=True, CP=PALISADE, esc=self.esc, e2e=experimental)
      if experimental:
        update = plant.planner.update

        def at_zero(sm, update=update, plant=plant):
          if plant.speed < 0.3:
            sm['modelV2'].action.desiredAcceleration = 0.0
          return update(sm)
        plant.planner.update = at_zero
      while plant.speed > 0.0 or plant.current_time < 1.0:
        plant.step(v_lead=0.0, v_cruise=10.0)
        assert plant.current_time < 30.0
      while plant.distance_lead - plant.distance < 8.8:
        plant.step(v_lead=0.2, v_cruise=10.0)
        assert plant.current_time < 60.0
      for _ in range(6):
        plant.step(v_lead=0.0, v_cruise=10.0, forecast=0.5)
      assert plant.planner.standstill_exit.launching
      start, x0, rows = plant.current_time, plant.distance, []
      for k in range(round(6.0 / DT_MDL)):
        plant.lead_relevancy = k != 1
        plant.step(v_lead=0.0, v_cruise=10.0, forecast=0.12)
        rows.append((plant.current_time - start, plant.distance - x0, plant.should_stop, plant.planner.standstill_exit.taken_back))
      rows = np.array(rows, dtype=float)
      assert np.all(rows[:, 1] == 0.0), (experimental, 'moved', rows[:, 1].max())
      assert rows[:, 3].sum() == 1 and self.at(rows, rows[:, 3] == 1.0) <= 0.35, (experimental, self.at(rows, rows[:, 3] == 1.0))
      assert np.all(rows[rows[:, 0] > self.at(rows, rows[:, 3] == 1.0) - 1e-6, 2]), experimental

  def test_resume_from_a_stop_starts_no_later_than_without_the_exit(self):
    for e2e in (False, True):
      maneuver = next(m for m in create_maneuvers({'e2e': e2e, 'force_decel': False}) if m.title == 'resume from a stop')
      maneuver.esc = self.esc
      valid, logs = maneuver.evaluate()
      with without_the_exit():
        _, before = maneuver.evaluate()
      assert valid
      started, started_before = (float(log[np.flatnonzero((log[:, 0] > 10.0) & (log[:, 3] > 0.0))[0], 0]) for log in (logs, before))
      assert started <= started_before, (e2e, started, started_before)

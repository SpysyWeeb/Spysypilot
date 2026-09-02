import itertools
import numpy as np
from openpilot.selfdrive.controls.lib.model_curve_speed import BEND_OPEN_S
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.stop_landing import (CREEP_PRESS_MAX, KISS_SPEED, LANDING_SPEED, LEAD_LANDING_GAP, LEAD_FULL_AUTHORITY,
                                                            StopLanding, landing_bound)
from openpilot.common.test import OpenpilotTestCase
from openpilot.common.parameterized import parameterized_class

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import STOP_DISTANCE
from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant


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


@parameterized_class(("e2e", "force_decel"), itertools.product([True, False], repeat=2))
class TestLongitudinalControl(OpenpilotTestCase):
  e2e: bool
  force_decel: bool

  def test_maneuver(self, subtests):
    for maneuver in create_maneuvers({"e2e": self.e2e, "force_decel": self.force_decel}):
      with subtests.test(title=maneuver.title, e2e=maneuver.e2e, force_decel=maneuver.force_decel):
        print(maneuver.title, f'in {"e2e" if maneuver.e2e else "acc"} mode')
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


class TestRedLightStop(OpenpilotTestCase):
  def test_a_late_red_light_stops_front_loaded_short_of_the_line(self):
    # route 24, 2026-08-29: the model calls a red light ~5 s out; the owner wants the needed deceleration reached
    # within a second, held, and eased off at the end -- never still increasing in the last metres
    maneuver = Maneuver('approach a red light at 14 m/s, seen 5 s out', duration=30.0, initial_speed=14.0,
                        cruise_values=[14.0, 14.0], e2e=True, stop_line=160.0)
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


class TestCurvePolicy(OpenpilotTestCase):
  def _run(self, **kwargs):
    maneuver = Maneuver(kwargs.pop('title'), duration=kwargs.pop('duration', 30.0), **kwargs)
    valid, logs = maneuver.evaluate()
    assert valid
    return logs[:, 1], logs[:, 3], logs[:, 5]     # distance, speed, acceleration

  def test_a_curve_ahead_is_approached_at_the_needed_deceleration_without_a_burst(self):
    # a 0.05 1/m curve (authority ~6.5 m/s with factor 2.7, friction 0.11) 120 m ahead at 20 m/s
    x, v, a = self._run(title='approach a tight curve', initial_speed=20.0, cruise_values=[20.0, 20.0], curve=(120.0, 80.0, 0.05))
    entry = int(np.flatnonzero(x >= 120.0)[0])
    assert v[entry] <= 7.5, v[entry]                                       # slowed to the curve's limit by its start
    assert a[:entry].min() >= -2.05, a[:entry].min()                       # never harder than the comfort floor
    onset = int(np.flatnonzero(a < -0.3)[0])
    assert a[onset:entry].max() <= 0.3, a[onset:entry].max()                # no burst toward the limit once braking has begun
    exit_ = int(np.flatnonzero(x >= 200.0)[0])
    after = a[exit_:exit_ + int(2.0 / DT_MDL)]
    assert after.max() >= 1.0, after.max()                                  # and it accelerates again within two seconds of the exit

  def test_pinned_but_tracking_coasts_and_pinned_understeering_brakes(self):
    # a curve the steering can only just hold: heavy torque, tracking -> coast; then a tighter one it cannot -> brake
    x, v, a = self._run(title='coast in a heavy curve', initial_speed=8.0, cruise_values=[8.0, 8.0], curve=(30.0, 150.0, 0.038),
                        duration=25.0)
    inside = (x > 40.0) & (x < 150.0)
    assert np.all(a[inside] <= 0.05), a[inside].max()                       # foot off through the curve ...
    assert a[inside].min() >= -1.0, a[inside].min()                         # ... but no real braking while it tracks
    # a curve the model reads at half its true curvature: anticipation lets the car in too fast, the steering pins
    # and understeers, and the reaction layer must brake it down
    x, v, a = self._run(title='brake in a curve the steering cannot hold', initial_speed=9.0, cruise_values=[9.0, 9.0],
                        curve=(30.0, 150.0, 0.06), curve_model_scale=0.5, duration=25.0)
    inside = (x > 35.0) & (x < 150.0)
    assert a[inside].min() <= -0.8, a[inside].min()                         # it brakes ...
    assert a[inside].min() >= -2.05                                         # ... within the floor
    assert v[inside][-1] < v[inside][0]                                     # ... and is slower deep in the curve

  def test_a_lift_ends_in_a_hold_until_the_bend_opens(self):
    # a tight entry the steering can only just hold, then a looser section that is still a bend (1.2 m/s^2 at the settled
    # speed): the coast releases as the torque eases, and the cruise set speed of 12 m/s wants the car back up at once
    x, v, a = self._run(title='hold through the looser half', initial_speed=9.0, cruise_values=[12.0, 12.0],
                        curve=[(30.0, 60.0, 0.038), (90.0, 100.0, 0.025)], duration=40.0)
    enter, exit_ = int(np.flatnonzero(x >= 92.0)[0]), int(np.flatnonzero(x >= 192.0)[0])
    assert a[enter:exit_].max() <= 0.05, a[enter:exit_].max()               # no acceleration for the rest of the bend ...
    assert abs(v[enter:exit_].max() - v[enter]) <= 0.3                      # ... the speed simply holds
    after = a[exit_:exit_ + int((BEND_OPEN_S + 1.5) / DT_MDL)]
    assert after.max() >= 0.3, after.max()                                  # and the bend's end gives the acceleration back
    # the same entry into a section that reads open (0.9 m/s^2 at the settled speed): the hold releases within its dwell
    x, v, a = self._run(title='release when the bend opens', initial_speed=9.0, cruise_values=[12.0, 12.0],
                        curve=[(30.0, 60.0, 0.038), (90.0, 120.0, 0.02)], duration=35.0)
    enter = int(np.flatnonzero(x >= 92.0)[0])
    soon = a[enter:enter + int((BEND_OPEN_S + 1.5) / DT_MDL)]
    assert soon.max() >= 0.3, soon.max()                                    # not held back once the road has opened

def landing_excess(logs, lead=False, v_min=KISS_SPEED, v_max=LANDING_SPEED):
  # the most the commanded braking exceeded the landing law through the last metres (KISS_SPEED .. LANDING_SPEED, plan braking).
  # Below the kiss speed the corridor is the kiss plus the anti-creep press by design, and the aEgo checks judge that end.
  # Each row's plan was computed from the previous row's state (the plant logs after integrating), so the law is judged at
  # that speed and gap; below 0.3 m/s the plant's own stop bit forces -0.5. With a lead, the braking that stopping
  # LEAD_LANDING_GAP behind it needs passes
  v, a, d_rel, v_lead = logs[:, 3], logs[:, 5], logs[:, 6], logs[:, 4]
  excess = 0.0
  for i in range(1, len(v)):
    v_seen, d_seen, v_lead_seen = v[i - 1], d_rel[i - 1], v_lead[i - 1]
    if not (v_min <= v_seen < v_max) or a[i] >= 0.0:
      continue
    allowed = landing_bound(v_seen)
    if lead:
      if d_seen < LEAD_FULL_AUTHORITY:
        continue
      closing = max(v_seen - v_lead_seen, 0.0)
      allowed = max(allowed, closing ** 2 / (2.0 * max(d_seen - LEAD_LANDING_GAP, 0.5)))
    excess = max(excess, -a[i] - allowed)
  return excess


class TestStopLanding(OpenpilotTestCase):
  def test_a_stopped_lead_is_landed_within_the_law_and_the_car_still_stops_behind_it(self):
    maneuver = Maneuver('approach a stopped lead at 10 m/s', duration=25.0, initial_speed=10.0, lead_relevancy=True,
                        initial_distance_lead=90.0, speed_lead_values=[0.0, 0.0], cruise_values=[10.0, 10.0])
    valid, logs = maneuver.evaluate()
    assert valid
    v, d_rel = logs[:, 3], logs[:, 6]
    assert np.any(v < 0.05), 'did not stop'
    assert d_rel[-1] >= LEAD_LANDING_GAP, d_rel[-1]
    assert landing_excess(logs, lead=True) <= 0.02

  def test_a_red_light_is_landed_within_the_law(self):
    maneuver = Maneuver('approach a red light at 14 m/s, seen 5 s out', duration=30.0, initial_speed=14.0,
                        cruise_values=[14.0, 14.0], e2e=True, stop_line=160.0)
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
                        cruise_values=[8.0, 8.0, 8.0, 8.0])
    valid, logs = maneuver.evaluate()
    assert valid, 'crashed'
    v, d_rel = logs[:, 3], logs[:, 6]
    assert np.any(v < 0.05), 'did not stop'
    assert d_rel.min() >= 2.0, d_rel.min()
    assert landing_excess(logs, lead=True) <= 0.02

  def test_the_models_late_ramp_is_bounded_and_the_car_still_stops_before_the_line(self):
    # route 27 t=1052: the model's request ramps hard into the last metres of a stop the car is already landing. The law
    # bounds that landing to its taper; the stop still lands short of the line
    kwargs = {'duration': 30.0, 'initial_speed': 14.0, 'cruise_values': [14.0, 14.0], 'e2e': True, 'stop_line': 160.0, 'e2e_landing_push': 3.5}
    valid, logs = Maneuver('red light with a late model ramp', **kwargs).evaluate()
    assert valid
    x, v = logs[:, 1], logs[:, 3]
    stopped = np.flatnonzero(v < 0.05)
    assert len(stopped) > 0, 'did not stop'
    assert kwargs['stop_line'] - 5.0 <= x[int(stopped[0])] <= kwargs['stop_line'], x[int(stopped[0])]
    assert landing_excess(logs) <= 0.02
    # the same ramp with the law bypassed lands well outside it: the test is about the law, not the plant
    original = StopLanding.update
    StopLanding.update = lambda self, a_target, v_ego, lead, stop_intent, launch=False, a_ego=None: a_target
    try:
      _, unbounded = Maneuver('red light with a late model ramp, no law', **kwargs).evaluate()
    finally:
      StopLanding.update = original
    assert landing_excess(unbounded) >= 0.5, landing_excess(unbounded)

  def test_a_car_that_lets_go_of_the_brake_slowly_still_lands_close_to_the_law(self):
    # the Palisade's ESP: ~0.2 s to take braking up, ~0.7 s to let it off (route 0x2a). Through that actuator the car may
    # not brake much harder than the corridor allows in the last metres, and it still stops behind the lead
    kwargs = {'duration': 25.0, 'initial_speed': 10.0, 'lead_relevancy': True, 'initial_distance_lead': 90.0,
              'speed_lead_values': [0.0, 0.0], 'cruise_values': [10.0, 10.0], 'actuator_lag': (0.2, 0.7)}
    valid, logs = Maneuver('stopped lead through a slow-release actuator', **kwargs).evaluate()
    assert valid
    v, a, d_rel = logs[:, 3], logs[:, 5], logs[:, 6]
    assert np.any(v < 0.05), 'did not stop'
    assert d_rel[-1] >= LEAD_LANDING_GAP
    # below ~0.5 m/s the corridor is already at the kiss and the lagged car is by design still catching up, so the
    # corridor-excess check applies above it; the low end is judged by what matters -- the deceleration still on the
    # car as the wheels are about to stop (route 0x2b: -0.37 measured at 0.15 m/s was the body-rock cause)
    assert landing_excess(logs, lead=True, v_min=0.5) <= 0.15, landing_excess(logs, lead=True, v_min=0.5)
    # below the kiss speed the anti-creep press may exceed the bound, by at most its cap
    assert landing_excess(logs, lead=True, v_min=0.3, v_max=KISS_SPEED) <= CREEP_PRESS_MAX + 0.02
    last_rolling = int(np.flatnonzero(v >= 0.15)[-1])
    assert a[last_rolling] >= -0.25, a[last_rolling]

import itertools
import numpy as np
from openpilot.selfdrive.controls.lib.model_curve_speed import BEND_OPEN_S
from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.common.parameterized import parameterized_class

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import STOP_DISTANCE
from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver


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

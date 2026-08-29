import itertools
import numpy as np
from openpilot.common.realtime import DT_MDL
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
    assert maneuver.stop_line - 9.0 <= x[i_stop] <= maneuver.stop_line, x[i_stop]
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

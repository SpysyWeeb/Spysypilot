import math

import pytest

from opendbc.car.structs import car
from openpilot.selfdrive.car.cruise import CRUISE_LONG_PRESS, IMPERIAL_INCREMENT, VCruiseHelper, V_CRUISE_MAX, V_CRUISE_MIN

ButtonType = car.CarState.ButtonEvent.Type


def cruise(start=40.0, pcm=False):
  helper = VCruiseHelper(car.CarParams(pcmCruise=pcm))
  helper.v_cruise_kph = start
  cs = car.CarState(cruiseState={'available': True})
  return helper, cs


def tick(helper, cs, metric, button=None, pressed=False, enabled=True):
  cs.buttonEvents = [] if button is None else [car.CarState.ButtonEvent(type=button, pressed=pressed)]
  helper.update_v_cruise(cs, enabled, metric)
  return helper.v_cruise_kph


@pytest.mark.parametrize('metric', [False, True])
@pytest.mark.parametrize('button,sign', [(ButtonType.accelCruise, 1), (ButtonType.decelCruise, -1)])
@pytest.mark.parametrize('start', [40.0, 43.0])
def test_tap_uses_five_unit_grid(metric, button, sign, start):
  helper, cs = cruise(start)
  step = 5 * (1.0 if metric else IMPERIAL_INCREMENT)
  round_grid = math.ceil if sign > 0 else math.floor
  expected = round_grid(start / step) * step if start % step else start + sign * step
  assert tick(helper, cs, metric, button, True) == start
  for _ in range(CRUISE_LONG_PRESS - 2):
    assert tick(helper, cs, metric) == start
  assert tick(helper, cs, metric, button, False) == pytest.approx(round(expected, 1))
  assert tick(helper, cs, metric) == pytest.approx(round(expected, 1))


@pytest.mark.parametrize('metric', [False, True])
@pytest.mark.parametrize('button,sign', [(ButtonType.accelCruise, 1), (ButtonType.decelCruise, -1)])
def test_hold_uses_repeated_single_units_without_release_jump(metric, button, sign):
  helper, cs = cruise()
  delta = 1.0 if metric else IMPERIAL_INCREMENT
  assert tick(helper, cs, metric, button, True) == 40.0
  for _ in range(CRUISE_LONG_PRESS - 1):
    assert tick(helper, cs, metric) == 40.0
  assert tick(helper, cs, metric) == pytest.approx(round(40.0 + sign * delta, 1))
  for _ in range(CRUISE_LONG_PRESS):
    tick(helper, cs, metric)
  expected = round(40.0 + 2 * sign * delta, 1)
  assert helper.v_cruise_kph == pytest.approx(expected)
  assert tick(helper, cs, metric, button, False) == pytest.approx(expected)


@pytest.mark.parametrize('start,button,expected', [(V_CRUISE_MAX - 1, ButtonType.accelCruise, V_CRUISE_MAX),
                                                 (V_CRUISE_MIN + 1, ButtonType.decelCruise, V_CRUISE_MIN)])
def test_taps_still_clip_to_speed_limits(start, button, expected):
  helper, cs = cruise(start)
  tick(helper, cs, True, button, True)
  assert tick(helper, cs, True, button, False) == expected


@pytest.mark.parametrize('held', [False, True])
def test_resume_from_standstill_does_not_change_set_speed(held):
  helper, cs = cruise()
  cs.cruiseState.standstill = True
  tick(helper, cs, True, ButtonType.accelCruise, True)
  for _ in range(2 * CRUISE_LONG_PRESS if held else 1):
    assert tick(helper, cs, True) == 40.0
  cs.cruiseState.standstill = False
  assert tick(helper, cs, True, ButtonType.accelCruise, False) == 40.0


def test_pcm_cruise_remains_owned_by_vehicle():
  helper, cs = cruise(pcm=True)
  cs.cruiseState.speed = 20.0
  cs.cruiseState.speedCluster = 20.0
  expected = tick(helper, cs, True)
  for pressed in (True, False):
    assert tick(helper, cs, True, ButtonType.accelCruise, pressed) == expected


@pytest.mark.parametrize('experimental', [False, True])
@pytest.mark.parametrize('speed_kph,expected', [(0., 40), (30., 40), (80., 80), (120., 120), (180., 145)])
def test_initial_set_uses_normal_speed_floor_in_both_modes(experimental, speed_kph, expected):
  from openpilot.common.constants import CV

  helper = VCruiseHelper(car.CarParams(pcmCruise=False))
  cs = car.CarState(vEgo=speed_kph * CV.KPH_TO_MS,
                     buttonEvents=[car.CarState.ButtonEvent(type=ButtonType.decelCruise)])
  helper.initialize_v_cruise(cs, experimental)
  assert helper.v_cruise_kph == expected
  assert helper.v_cruise_cluster_kph == expected
  assert V_CRUISE_MIN == 8, 'the daily adjustable-speed floor is separate from initialization'


@pytest.mark.parametrize('experimental', [False, True])
@pytest.mark.parametrize('button', [ButtonType.resumeCruise, ButtonType.accelCruise])
def test_initial_resume_retains_previous_set_speed(experimental, button):
  helper, cs = cruise()
  helper.v_cruise_kph_last = 73
  cs.buttonEvents = [car.CarState.ButtonEvent(type=button)]
  helper.initialize_v_cruise(cs, experimental)
  assert helper.v_cruise_kph == helper.v_cruise_cluster_kph == 73


@pytest.mark.parametrize('experimental', [False, True])
def test_initial_set_does_not_replace_pcm_speed(experimental):
  helper, cs = cruise(start=73., pcm=True)
  helper.initialize_v_cruise(cs, experimental)
  assert helper.v_cruise_kph == 73

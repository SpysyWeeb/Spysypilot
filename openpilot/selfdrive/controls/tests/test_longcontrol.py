from types import SimpleNamespace

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState, long_control_state_trans
from openpilot.selfdrive.controls.lib.smooth_stops import HOLD_RELEASE_FRAMES


class TestLongControlStateTransition(OpenpilotTestCase):

  def test_stay_stopped(self):
    active = True
    current_state = LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=True, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=True, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=True)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.pid
    active = False
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.off

  def test_engage(self):
    active = True
    current_state = LongCtrlState.off
    next_state = long_control_state_trans(active, current_state,
                             should_stop=True, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=True, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=True)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.pid


def long_control():
  tuning = SimpleNamespace(kpBP=[0.0], kpV=[1.0], kiBP=[0.0], kiV=[0.0])
  return LongControl(SimpleNamespace(longitudinalTuning=tuning, stopAccel=-2.0))


def car_state():
  return SimpleNamespace(
    vEgo=0.0,
    aEgo=0.0,
    brakePressed=False,
    standstill=True,
    cruiseState=SimpleNamespace(standstill=False),
  )


class TestSmoothStopRoute17Continuity(OpenpilotTestCase):
  def test_route17_plan_chatter_does_not_release_hold(self):
    # plan chatter shorter than the release debounce never lets go of the hold, lead or not
    control = long_control()
    control.long_control_state = LongCtrlState.stopping
    control.last_output_accel = -0.75
    control.smooth.arm_hold()

    states = []
    outputs = []
    for should_stop in [True] * 20 + [False] * (HOLD_RELEASE_FRAMES - 1) + [True] * 20:
      outputs.append(control.update(
        True, car_state(), -0.24, should_stop, (-3.5, 2.0),
        lead_distance=4.3, has_lead=True, lead_speed=0.262,
      ))
      states.append(control.long_control_state)

    assert set(states) == {LongCtrlState.stopping}
    assert all(output <= -0.75 for output in outputs)

  def test_stopped_lead_release_is_the_same_debounce(self):
    # a stopped lead in radar view adds nothing to the wait: the car's own standstill exit already costs ~1.3 s
    control = long_control()
    control.long_control_state = LongCtrlState.stopping
    control.smooth.arm_hold()

    control.update(True, car_state(), 0.2, True, (-3.5, 2.0), has_lead=True)
    for _ in range(HOLD_RELEASE_FRAMES - 1):
      control.update(True, car_state(), 0.2, False, (-3.5, 2.0), has_lead=True)
      assert control.long_control_state == LongCtrlState.stopping
    control.update(True, car_state(), 0.2, False, (-3.5, 2.0), has_lead=True)
    assert control.long_control_state == LongCtrlState.pid

  def test_measured_lead_departure_releases_immediately(self):
    control = long_control()
    control.long_control_state = LongCtrlState.stopping
    control.smooth.arm_hold()
    control.update(True, car_state(), 0.2, True, (-3.5, 2.0),
                   lead_distance=4.3, has_lead=True, lead_speed=0.0)

    control.update(True, car_state(), 0.2, False, (-3.5, 2.0),
                   lead_distance=4.3, has_lead=True, lead_speed=0.4)
    assert control.long_control_state == LongCtrlState.pid

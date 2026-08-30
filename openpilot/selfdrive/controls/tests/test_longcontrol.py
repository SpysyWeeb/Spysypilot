from types import SimpleNamespace

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState, long_control_state_trans
from openpilot.selfdrive.controls.lib.smooth_stops import HOLD_RELEASE_FRAMES, STOP_KISS_DECEL


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


class TestSmoothStopHandoff(OpenpilotTestCase):
  def test_the_clamp_waits_for_the_car_to_stop_and_the_landing_owns_the_meantime(self):
    control = long_control()
    control.long_control_state = LongCtrlState.pid
    control.last_output_accel = -0.4
    rolling = car_state()
    rolling.vEgo = 0.25
    rolling.standstill = False
    output = control.update(True, rolling, -0.05, True, (-3.5, 2.0))
    assert control.long_control_state == LongCtrlState.pid               # still rolling: no clamp
    assert output <= -STOP_KISS_DECEL                                     # the landing keeps the kiss on
    stopped = car_state()
    control.update(True, stopped, -0.05, True, (-3.5, 2.0))
    assert control.long_control_state == LongCtrlState.stopping          # stopped: the clamp arms

  def test_plan_chatter_shorter_than_the_debounce_does_not_release_the_hold(self):
    control = long_control()
    control.long_control_state = LongCtrlState.stopping
    control.last_output_accel = -0.75
    control.smooth_stop.arm_hold()
    states = []
    for should_stop in [True] * 20 + [False] * (HOLD_RELEASE_FRAMES - 1) + [True] * 20:
      control.update(True, car_state(), -0.24, should_stop, (-3.5, 2.0))
      states.append(control.long_control_state)
    assert set(states) == {LongCtrlState.stopping}

  def test_the_hold_releases_after_the_debounce(self):
    control = long_control()
    control.long_control_state = LongCtrlState.stopping
    control.smooth_stop.arm_hold()
    control.update(True, car_state(), 0.2, True, (-3.5, 2.0))
    for _ in range(HOLD_RELEASE_FRAMES - 1):
      control.update(True, car_state(), 0.2, False, (-3.5, 2.0))
      assert control.long_control_state == LongCtrlState.stopping
    control.update(True, car_state(), 0.2, False, (-3.5, 2.0))
    assert control.long_control_state == LongCtrlState.pid

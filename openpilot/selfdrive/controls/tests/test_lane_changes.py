from types import SimpleNamespace

import pytest

from opendbc.car.structs import car
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper, LaneChangeDirection, LaneChangeState, LANE_CHANGE_SPEED_MIN, \
                                                         LANE_CHANGE_TIME_MAX


def make_car_state(**kwargs):
  kwargs.setdefault('vEgo', LANE_CHANGE_SPEED_MIN + 1)
  return car.CarState.new_message(**kwargs)


def run(helper, car_state, frames, lateral_active=True, lane_change_prob=0.0):
  # the lane change state after each of frames model frames
  states = []
  for _ in range(frames):
    helper.update(car_state, lateral_active, lane_change_prob)
    states.append(helper.lane_change_state)
  return states


def start_automatically(helper, car_state):
  # a fresh helper sees the held signal come on with lateral active and starts the change on its own
  assert run(helper, car_state, 3, lane_change_prob=1.0) == [LaneChangeState.preLaneChange] * 2 + [LaneChangeState.laneChangeStarting]
  assert not car_state.steeringPressed


def finish(helper, car_state):
  # the model reports the change done; with a signal still held the helper waits in preLaneChange
  for _ in range(20):
    if run(helper, car_state, 1)[-1] != LaneChangeState.laneChangeStarting:
      break
  assert helper.lane_change_state == LaneChangeState.preLaneChange


def nudge_starts(helper, car_state, torque=1.0):
  car_state.steeringPressed = True
  car_state.steeringTorque = torque
  return run(helper, car_state, 1) == [LaneChangeState.laneChangeStarting]


def test_brake_on_blinker_activation_cancels_nudgeless_change():
  car_state = SimpleNamespace(
    vEgo=LANE_CHANGE_SPEED_MIN + 1,
    leftBlinker=False,
    rightBlinker=False,
    brakePressed=False,
    steeringPressed=False,
    steeringTorque=0.0,
    leftBlindspot=False,
    rightBlindspot=False,
  )
  helper = DesireHelper()
  helper.update(car_state, True, 1.0)

  car_state.leftBlinker = True
  car_state.brakePressed = True
  helper.update(car_state, True, 1.0)
  car_state.brakePressed = False
  for _ in range(10):
    helper.update(car_state, True, 1.0)

  assert helper.lane_change_state == LaneChangeState.preLaneChange
  assert helper.brake_cancelled


@pytest.mark.parametrize(('blinker', 'torque'), [('leftBlinker', 1.0), ('rightBlinker', -1.0)])
def test_early_manual_start_does_not_leave_an_automatic_retry(blinker, torque):
  car_state = car.CarState.new_message(vEgo=LANE_CHANGE_SPEED_MIN + 1)
  helper = DesireHelper()
  helper.update(car_state, True, 1.0)

  setattr(car_state, blinker, True)
  car_state.steeringPressed = True
  car_state.steeringTorque = torque
  helper.update(car_state, True, 1.0)
  assert helper.lane_change_state == LaneChangeState.preLaneChange

  # The nudge starts on the first waiting frame, before the auto delay expires.
  helper.update(car_state, True, 1.0)
  assert helper.lane_change_state == LaneChangeState.laneChangeStarting
  car_state.steeringPressed = False
  car_state.steeringTorque = 0.0

  for _ in range(20):
    helper.update(car_state, True, 0.0)
    if helper.lane_change_state == LaneChangeState.preLaneChange:
      break
  assert helper.lane_change_state == LaneChangeState.preLaneChange

  # Keeping the same signal on must not initiate another change without input.
  for _ in range(40):
    helper.update(car_state, True, 0.0)
    assert helper.lane_change_state == LaneChangeState.preLaneChange

  # A later manual nudge is still allowed under the same signal.
  car_state.steeringPressed = True
  car_state.steeringTorque = torque
  helper.update(car_state, True, 0.0)
  assert helper.lane_change_state == LaneChangeState.laneChangeStarting


@pytest.mark.parametrize('blinker', ['leftBlinker', 'rightBlinker'])
def test_automatic_start_requires_signal_cycle_to_rearm(blinker):
  car_state = car.CarState.new_message(vEgo=LANE_CHANGE_SPEED_MIN + 1)
  setattr(car_state, blinker, True)
  helper = DesireHelper()
  helper.update(car_state, True, 1.0)
  helper.update(car_state, True, 1.0)
  assert helper.lane_change_state == LaneChangeState.preLaneChange
  helper.update(car_state, True, 1.0)
  assert helper.lane_change_state == LaneChangeState.laneChangeStarting

  for _ in range(20):
    helper.update(car_state, True, 0.0)
    if helper.lane_change_state == LaneChangeState.preLaneChange:
      break
  for _ in range(40):
    helper.update(car_state, True, 0.0)
    assert helper.lane_change_state == LaneChangeState.preLaneChange

  setattr(car_state, blinker, False)
  helper.update(car_state, True, 0.0)
  assert helper.lane_change_state == LaneChangeState.off
  setattr(car_state, blinker, True)
  for _ in range(3):
    helper.update(car_state, True, 0.0)
  assert helper.lane_change_state == LaneChangeState.laneChangeStarting


def test_brake_cancellation_preserves_manual_start():
  car_state = car.CarState.new_message(vEgo=LANE_CHANGE_SPEED_MIN + 1, leftBlinker=True, brakePressed=True)
  helper = DesireHelper()
  helper.update(car_state, True, 1.0)
  car_state.brakePressed = False
  for _ in range(10):
    helper.update(car_state, True, 1.0)
  assert helper.lane_change_state == LaneChangeState.preLaneChange

  car_state.steeringPressed = True
  car_state.steeringTorque = 1.0
  helper.update(car_state, True, 1.0)
  assert helper.lane_change_state == LaneChangeState.laneChangeStarting


@pytest.mark.parametrize('manual_start', [False, True])
def test_blindspot_blocks_start_without_consuming_allowance(manual_start):
  car_state = car.CarState.new_message(vEgo=LANE_CHANGE_SPEED_MIN + 1, leftBlinker=True, leftBlindspot=True,
                                     steeringPressed=manual_start, steeringTorque=1.0 if manual_start else 0.0)
  helper = DesireHelper()
  for _ in range(10):
    helper.update(car_state, True, 1.0)
  assert helper.lane_change_state == LaneChangeState.preLaneChange
  assert not helper.nudgeless_used

  car_state.leftBlindspot = False
  helper.update(car_state, True, 1.0)
  assert helper.lane_change_state == LaneChangeState.laneChangeStarting


@pytest.mark.parametrize('inactive_frames', [1, 20])
@pytest.mark.parametrize('during_change', [False, True])
def test_lateral_dropout_does_not_restore_a_spent_allowance(during_change, inactive_frames):
  # a model frame or a few without lateral control (a stale permission frame, a steering fault, a disengage and
  # re-engage) while the signal stays on: the automatic start this signal event made is not offered again
  car_state = make_car_state(leftBlinker=True)
  helper = DesireHelper()
  start_automatically(helper, car_state)
  if not during_change:
    finish(helper, car_state)
  assert run(helper, car_state, inactive_frames, lateral_active=False)[-1] == LaneChangeState.off
  assert LaneChangeState.laneChangeStarting not in run(helper, car_state, 40)
  assert helper.lane_change_state == LaneChangeState.preLaneChange
  assert nudge_starts(helper, car_state)


@pytest.mark.parametrize('brake_drops_lateral', [False, True])
def test_brake_cancellation_survives_a_lateral_dropout(brake_drops_lateral):
  car_state = make_car_state()
  helper = DesireHelper()
  run(helper, car_state, 1)
  car_state.leftBlinker = True
  assert run(helper, car_state, 1) == [LaneChangeState.preLaneChange]
  # braking with the signal on, also when the brake press itself takes lateral control away
  car_state.brakePressed = True
  run(helper, car_state, 1, lateral_active=not brake_drops_lateral)
  car_state.brakePressed = False
  run(helper, car_state, 1, lateral_active=False)
  assert LaneChangeState.laneChangeStarting not in run(helper, car_state, 40)
  assert helper.lane_change_state == LaneChangeState.preLaneChange and helper.brake_cancelled
  assert nudge_starts(helper, car_state)


@pytest.mark.parametrize('inactive_frames', [1, 20])
def test_signal_begun_without_lateral_control_waits_for_a_nudge(inactive_frames):
  # a signal already on when lateral control comes back was not given to start a change: it waits for a nudge, as stock
  car_state = make_car_state()
  helper = DesireHelper()
  run(helper, car_state, 1)
  car_state.leftBlinker = True
  run(helper, car_state, inactive_frames, lateral_active=False)
  assert LaneChangeState.laneChangeStarting not in run(helper, car_state, 40)
  assert helper.lane_change_state == LaneChangeState.preLaneChange
  assert nudge_starts(helper, car_state)


@pytest.mark.parametrize('waiting_first', [False, True])
def test_signal_held_through_low_speed_is_not_armed_by_a_lateral_dropout(waiting_first):
  # below the lane-change speed the helper needs a fresh signal to begin again, as stock; a lateral dropout once back
  # above it is no fresh signal
  car_state = make_car_state(leftBlindspot=True)
  if not waiting_first:
    car_state.vEgo = LANE_CHANGE_SPEED_MIN - 1
  helper = DesireHelper()
  run(helper, car_state, 1)
  car_state.leftBlinker = True
  assert run(helper, car_state, 5)[-1] == (LaneChangeState.preLaneChange if waiting_first else LaneChangeState.off)
  car_state.vEgo = LANE_CHANGE_SPEED_MIN - 1
  assert run(helper, car_state, 5)[-1] == LaneChangeState.off
  car_state.vEgo = LANE_CHANGE_SPEED_MIN + 1
  car_state.leftBlindspot = False
  assert run(helper, car_state, 10) == [LaneChangeState.off] * 10
  run(helper, car_state, 1, lateral_active=False)
  assert LaneChangeState.laneChangeStarting not in run(helper, car_state, 40)
  assert helper.lane_change_state == LaneChangeState.preLaneChange


def test_timed_out_change_is_not_restarted_by_a_lateral_dropout():
  car_state = make_car_state(leftBlinker=True)
  helper = DesireHelper()
  start_automatically(helper, car_state)
  assert run(helper, car_state, round(LANE_CHANGE_TIME_MAX / DT_MDL) + 5, lane_change_prob=1.0)[-1] == LaneChangeState.off
  # holding the signal begins nothing, as stock, and a lateral dropout leaves it to a nudge
  assert run(helper, car_state, 40) == [LaneChangeState.off] * 40
  run(helper, car_state, 1, lateral_active=False)
  assert LaneChangeState.laneChangeStarting not in run(helper, car_state, 40)
  assert helper.lane_change_state == LaneChangeState.preLaneChange
  # cycling the signal rearms the automatic start
  car_state.leftBlinker = False
  assert run(helper, car_state, 1) == [LaneChangeState.off]
  car_state.leftBlinker = True
  assert run(helper, car_state, 3)[-1] == LaneChangeState.laneChangeStarting


def test_unused_allowance_survives_a_lateral_dropout():
  # waiting on the blind spot this signal event has not started a change yet, so it still may once lateral control is back
  car_state = make_car_state(leftBlinker=True, leftBlindspot=True)
  helper = DesireHelper()
  assert run(helper, car_state, 10)[-1] == LaneChangeState.preLaneChange
  assert run(helper, car_state, 1, lateral_active=False) == [LaneChangeState.off]
  car_state.leftBlindspot = False
  assert run(helper, car_state, 3)[-1] == LaneChangeState.laneChangeStarting
  assert not car_state.steeringPressed


@pytest.mark.parametrize('signal', [
  [(False, False), (True, False)],          # the signal drops out for a frame
  [(True, True)] * 5 + [(False, True)],     # moved to the other side: a lamp-timed signal reads both on while the first side's hold runs out
  [(False, False)] * 20 + [(True, False)],  # given again, as when a one-touch signal's flashes end mid-change
])
def test_signal_given_again_during_a_change_does_not_rearm(signal):
  car_state = make_car_state(leftBlinker=True)
  helper = DesireHelper()
  start_automatically(helper, car_state)
  for left, right in signal:
    car_state.leftBlinker, car_state.rightBlinker = left, right
    assert run(helper, car_state, 1, lane_change_prob=1.0) == [LaneChangeState.laneChangeStarting]
  finish(helper, car_state)
  assert LaneChangeState.laneChangeStarting not in run(helper, car_state, 40)
  assert helper.lane_change_state == LaneChangeState.preLaneChange


@pytest.mark.parametrize('overlap', [True, False])
def test_switching_signal_sides_after_a_change(overlap):
  car_state = make_car_state(leftBlinker=True)
  helper = DesireHelper()
  start_automatically(helper, car_state)
  finish(helper, car_state)
  if overlap:
    # a lamp-timed signal (Hyundai) reads both sides on while the first side's hold runs out: the signal ends there, and
    # the other side is a new signal event
    car_state.rightBlinker = True
    assert run(helper, car_state, 5)[-1] == LaneChangeState.off
    car_state.leftBlinker = False
    assert run(helper, car_state, 3)[-1] == LaneChangeState.laneChangeStarting
    assert helper.lane_change_direction == LaneChangeDirection.right
  else:
    # a signal that moves straight across with no frame between is the same signal event: only a nudge starts a change
    car_state.leftBlinker, car_state.rightBlinker = False, True
    assert LaneChangeState.laneChangeStarting not in run(helper, car_state, 40)
    assert helper.lane_change_direction == LaneChangeDirection.right
    assert nudge_starts(helper, car_state, torque=-1.0)

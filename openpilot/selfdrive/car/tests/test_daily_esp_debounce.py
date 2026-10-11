import pytest

from opendbc.car.structs import car
from openpilot.cereal import log
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.car.car_events import CarEvents
from openpilot.selfdrive.controls.tests.test_sol_driver import activate, make_driver, step
from openpilot.selfdrive.selfdrived.events import ET
from openpilot.selfdrive.selfdrived.state import SOFT_DISABLE_TIME, State, StateMachine
from openpilot.spysypilot.aol.state import State as SolState

EventName = log.OnroadEvent.EventName
HOLD_FRAMES = int(0.5 / DT_CTRL)


def setup_events():
  driver, cs = make_driver()
  driver.CP.carFingerprint = 'HYUNDAI_PALISADE'
  producer = CarEvents(driver.CP)
  return producer, driver, cs, car.CarControl.new_message()


@pytest.mark.parametrize('overriding', [False, True])
def test_sustained_esp_reaches_both_state_machines(overriding):
  producer, driver, cs, cc = setup_events()
  activate(driver, cs)
  cs.steeringPressed = overriding
  sm = StateMachine()
  sm.state = State.overriding if overriding else State.enabled
  cs.espActive = True
  for frame in range(1, HOLD_FRAMES + 1):
    events = producer.update(cs, cs, cc)
    driver.sd.events = events
    sm.update(events)
    step(driver, cs)
    assert (EventName.espActive in events.names) == (frame == HOLD_FRAMES)
    if frame < HOLD_FRAMES:
      assert sm.state != State.softDisabling
      assert driver.state_machine.state != SolState.softDisabling
  assert sm.state == State.softDisabling
  assert driver.state_machine.state == SolState.softDisabling
  # Preserve the existing downstream soft-disable period, not immediate shutdown.
  for _ in range(int(SOFT_DISABLE_TIME / DT_CTRL) + 1):
    events = producer.update(cs, cs, cc)
    driver.sd.events = events
    sm.update(events)
    step(driver, cs)
  assert sm.state == State.disabled
  assert not driver.active


@pytest.mark.parametrize('flag,event', [('espDisabled', EventName.espDisabled), ('steerFaultPermanent', EventName.steerUnavailable),
                                       ('stockAeb', EventName.stockAeb), ('stockFcw', EventName.stockFcw)])
def test_other_fault_and_collision_events_are_not_debounced(flag, event):
  producer, _, cs, cc = setup_events()
  cs.espActive = True
  setattr(cs, flag, True)
  events = producer.update(cs, cs, cc)
  assert event in events.names
  assert EventName.espActive not in events.names


def test_active_event_keeps_no_entry_and_clears_on_first_inactive_sample():
  producer, _, cs, cc = setup_events()
  cs.espActive = True
  for _ in range(HOLD_FRAMES):
    events = producer.update(cs, cs, cc)
  assert EventName.espActive in events.names
  assert events.contains(ET.NO_ENTRY) and events.contains(ET.SOFT_DISABLE)
  events.add(EventName.buttonEnable)
  sm = StateMachine()
  assert sm.update(events) == (False, False)
  cs.espActive = False
  assert EventName.espActive not in producer.update(cs, cs, cc).names
  cs.espActive = True
  for _ in range(HOLD_FRAMES - 1):
    assert EventName.espActive not in producer.update(cs, cs, cc).names
  assert EventName.espActive in producer.update(cs, cs, cc).names

from types import SimpleNamespace

import pytest

from opendbc.car.structs import car
from openpilot.cereal import log
from openpilot.selfdrive.selfdrived.events import Events
from openpilot.spysypilot.aol.aol import AolDriver
from openpilot.spysypilot.aol.state import State


def make_driver():
  class Sm(dict):
    alive = {'pandaStates': True, 'controlsState': True, 'modelV2': True}
    valid = dict(alive)

  sm = Sm(pandaStates=[log.PandaState.new_message(safetyModel=car.CarParams.SafetyModel.hyundai, lateralAllowed=True)])
  sm.alive = dict(sm.alive)
  sm.valid = dict(sm.valid)
  cp = car.CarParams.new_message(brand='hyundai', minSteerSpeed=0.0, passive=False)
  sd = SimpleNamespace(CP=cp, sm=sm, enabled=False, events=Events())
  driver = AolDriver(sd)
  cs = car.CarState.new_message(vEgo=10.0, canValid=True, gearShifter=car.CarState.GearShifter.drive)
  cs.cruiseState.available = True
  return driver, cs


def step(driver, cs):
  driver.update_events(cs)
  driver.update(cs)


def activate(driver, cs):
  cs.buttonEvents = [{'type': 'mainCruise', 'pressed': True}]
  step(driver, cs)
  cs.buttonEvents = []
  assert driver.enabled and driver.active


@pytest.mark.parametrize('service', ['controlsState', 'pandaStates', 'modelV2'])
@pytest.mark.parametrize('field', ['alive', 'valid'])
def test_invalid_or_stale_control_inputs_disable_sol(service, field):
  driver, cs = make_driver()
  activate(driver, cs)
  getattr(driver.sd.sm, field)[service] = False
  step(driver, cs)
  assert not driver.enabled and not driver.active
  assert driver.create_aol_alerts()
  # A held/repeated enable request cannot bypass the unhealthy input.
  cs.buttonEvents = [{'type': 'mainCruise', 'pressed': True}]
  step(driver, cs)
  assert not driver.enabled


@pytest.mark.parametrize('brand,passive', [('hyundai', True), ('toyota', False)])
def test_sol_available_only_with_matching_active_hyundai_safety(brand, passive):
  driver, cs = make_driver()
  driver.CP.brand = brand
  driver.CP.passive = passive
  assert not driver.available
  cs.buttonEvents = [{'type': 'mainCruise', 'pressed': True}]
  step(driver, cs)
  assert not driver.active


def test_startup_pedals_pause_and_reverse():
  driver, cs = make_driver()
  step(driver, cs)
  assert not driver.enabled  # initial cruise availability is not a button edge
  activate(driver, cs)
  cs.gasPressed = cs.brakePressed = True
  step(driver, cs)
  assert driver.active
  cs.doorOpen = cs.steeringPressed = True
  step(driver, cs)
  assert driver.enabled and not driver.active
  cs.doorOpen = False
  step(driver, cs)
  assert driver.active and driver.state_machine.state == State.overriding
  cs.gearShifter = car.CarState.GearShifter.reverse
  step(driver, cs)
  assert not driver.enabled and not driver.create_aol_alerts()


@pytest.mark.parametrize('event', ['calibrationInvalid', 'modeldLagging', 'controlsMismatch', 'tooDistracted'])
def test_stock_safety_events_cannot_be_bypassed_by_sol(event):
  driver, cs = make_driver()
  activate(driver, cs)
  driver.sd.events.add(getattr(log.OnroadEvent.EventName, event))
  step(driver, cs)
  assert driver.state_machine.state in (State.disabled, State.softDisabling)
  for _ in range(310):
    step(driver, cs)
  assert not driver.active


@pytest.mark.parametrize('event', ['pcmDisable', 'pedalPressed', 'buttonCancel', 'wrongCarMode', 'cruiseDisabled'])
def test_cruise_and_pedal_events_do_not_disable_independent_steering(event):
  driver, cs = make_driver()
  activate(driver, cs)
  driver.sd.events.add(getattr(log.OnroadEvent.EventName, event))
  step(driver, cs)
  assert driver.active

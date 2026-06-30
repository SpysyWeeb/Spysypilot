import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from opendbc.car.structs import car
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.selfdrived.events import ET, Events
from openpilot.spysypilot.aol.state import AolStateMachine, ACTIVE_STATES, ENABLED_STATES, State
from openpilot.spysypilot.aol.helpers import is_hyundai_always_allow

ButtonType = car.CarState.ButtonEvent.Type
EventName = log.OnroadEvent.EventName


class AolDriver:
  """Always-On-Lateral state machine for Spysypilot.

  Runs alongside selfdrived's main state machine. When active,
  controlsd uses aol.active instead of selfdriveState.active
  to determine if lateral steering should be sent.
  """

  def __init__(self, selfdrived):
    self.sd = selfdrived
    self.CP = selfdrived.CP

    self.state_machine = AolStateMachine(DT_CTRL)
    self.enabled = False
    self.active = False

    # Hyundai with LDA button or CANFD can always activate (no cruise required)
    self.allow_always = is_hyundai_always_allow(self.CP)

    self._cruise_available_prev = False

    # Set panda alternativeExperience flags

  @property
  def available(self) -> bool:
    return True  # Always enabled — no toggle in Spysypilot

  def update_events(self, CS) -> None:
    """Process car state and generate AOL events for the state machine."""
    self.state_machine.clear_events()

    cruise_available = CS.cruiseState.available

    # LKAS/LFA button press (rising edge)
    lkas_pressed = any(be.type == ButtonType.lkas and be.pressed for be in CS.buttonEvents)
    if lkas_pressed:
      if self.enabled:
        self.state_machine.add_event('userDisable')
      else:
        if cruise_available or self.allow_always:
          self.state_machine.add_event('lkasEnable')

    # ACC main rising edge → activate AOL
    if cruise_available and not self._cruise_available_prev:
      if not self.enabled:
        self.state_machine.add_event('lkasEnable')

    # ACC main falling edge → deactivate AOL
    if not cruise_available and self._cruise_available_prev:
      if self.enabled:
        self.state_machine.add_event('immediateDisable')

    self._cruise_available_prev = cruise_available

    # Safety: door open or park brake → pause
    if hasattr(CS, 'doorOpen') and CS.doorOpen:
      self.state_machine.add_event('canPause')
    if hasattr(CS, 'parkingBrake') and CS.parkingBrake:
      self.state_machine.add_event('canPause')

    # Steering fault → disable
    if CS.steerFaultPermanent:
      self.state_machine.add_event('immediateDisable')
    elif CS.steerFaultTemporary:
      self.state_machine.add_event('softDisable')

    # REMAIN_ACTIVE: brake/gas press does NOT affect AOL state

  def update(self, CS) -> None:
    self.enabled, self.active = self.state_machine.update()

  def publish(self, pm) -> None:
    msg = messaging.new_message('spysydriveStateSP')
    aol = msg.spysydriveStateSP.aol
    aol.state = self.state_machine.state
    aol.enabled = self.enabled
    aol.active = self.active
    aol.available = self.available
    pm.send('spysydriveStateSP', msg)
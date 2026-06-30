import openpilot.cereal.messaging as messaging
from openpilot.cereal import custom, log
from opendbc.car.structs import car
from opendbc.car.structs import car
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.selfdrived.events import ET, Events
from openpilot.spysypilot.mads.state import MADSStateMachine, ACTIVE_STATES, ENABLED_STATES, State
from openpilot.spysypilot.mads.helpers import set_alternative_experience, is_hyundai_always_allow

ButtonType = car.CarState.ButtonEvent.Type
EventName = log.OnroadEvent.EventName


class MADSDriver:
  """Always-On-Lateral MADS state machine for Spysypilot.

  Runs alongside selfdrived's main state machine. When active,
  controlsd uses mads.active instead of selfdriveState.active
  to determine if lateral steering should be sent.
  """

  def __init__(self, selfdrived):
    self.sd = selfdrived  # reference to SelfdriveD instance
    self.CP = selfdrived.CP

    self.state_machine = MADSStateMachine(DT_CTRL)
    self.enabled = False
    self.active = False

    # Hyundai with LDA button or CANFD can always activate (no cruise required)
    self.allow_always = is_hyundai_always_allow(self.CP)

    # Track previous state for edge detection
    self._cruise_available_prev = False
    self._lkas_button_pressed_prev = False

    # Set panda alternativeExperience flags
    set_alternative_experience(self.CP)

  @property
  def available(self) -> bool:
    return True  # Always enabled -- no toggle in Spysypilot

  def update_events(self, CS) -> None:
    """Process car state and generate MADS events for the state machine."""
    self.state_machine.clear_events()

    cruise_available = CS.cruiseState.available

    # --- Activation triggers ---

    # LKAS/LFA button press (rising edge)
    lkas_pressed = any(be.type == ButtonType.lkas and be.pressed for be in CS.buttonEvents)
    if lkas_pressed:
      if self.enabled:
        # If MADS is on, button toggles it off
        self.state_machine.add_event('userDisable')
      else:
        # If MADS is off, button turns it on
        # For Hyundai allow_always: don't require cruise to be available
        if cruise_available or self.allow_always:
          self.state_machine.add_event('lkasEnable')

    # ACC main rising edge (cruise becoming available)
    if cruise_available and not self._cruise_available_prev:
      if not self.enabled:
        self.state_machine.add_event('lkasEnable')
    self._cruise_available_prev = cruise_available

    # ACC main falling edge (cruise turning off) -> disable MADS
    if not cruise_available and self._cruise_available_prev:
      if self.enabled:
        self.state_machine.add_event('immediateDisable')

    # --- Safety conditions ---

    # Door open -> soft pause (door close will allow re-enable via silentLkasEnable)
    if hasattr(CS, 'doorOpen') and CS.doorOpen:
      self.state_machine.add_event('canPause')

    # Park brake -> pause
    if hasattr(CS, 'parkingBrake') and CS.parkingBrake:
      self.state_machine.add_event('canPause')

    # Steering fault -> soft disable
    if CS.steerFaultPermanent:
      self.state_machine.add_event('immediateDisable')
    elif CS.steerFaultTemporary:
      self.state_machine.add_event('softDisable')

    # NOTE: REMAIN_ACTIVE mode -- brake/gas press does NOT affect MADS state

  def update(self, CS) -> None:
    """Run the state machine and update enabled/active."""
    self.enabled, self.active = self.state_machine.update()

  def publish(self, pm) -> None:
    """Publish MADS state to spysydriveStateSP cereal message."""
    msg = messaging.new_message('spysydriveStateSP')
    mads = msg.spysydriveStateSP.aol
    mads.state = self.state_machine.state
    mads.enabled = self.enabled
    mads.active = self.active
    mads.available = self.available
    pm.send('spysydriveStateSP', msg)
import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from opendbc.car.structs import car
from openpilot.common.realtime import DT_CTRL
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.selfdrived.events import (ET, Events, Alert, Priority, AlertSize,
                                                   AlertStatus, VisualAlert, AudibleAlert)
from openpilot.spysypilot.aol.state import AolStateMachine, ACTIVE_STATES, ENABLED_STATES, State
from openpilot.spysypilot.aol.helpers import is_hyundai_always_allow

ButtonType = car.CarState.ButtonEvent.Type
EventName = log.OnroadEvent.EventName

# Turn-signal / lane-change events surfaced as alerts during AOL-only steering.
# All of these are plain ET.WARNING alerts, which the main state machine only
# permits while fully engaged — so without help they never render under AOL.
LANE_CHANGE_EVENTS = (EventName.preLaneChangeLeft, EventName.preLaneChangeRight,
                      EventName.laneChangeBlocked, EventName.laneChange)

# How long the one-shot "AOL disengaged" warning stays up after AOL turns itself off
DISABLE_ALERT_TIME = 3.0


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

    # None until the first carState frame so the initial value of
    # cruiseState.available can never register as a rising/falling edge
    self._cruise_available_prev: bool | None = None
    self._lateral_mismatch_counter = 0
    self._disable_alert_frames = 0


  @property
  def available(self) -> bool:
    return True  # Always enabled — no toggle in Spysypilot

  def check_panda_mismatch(self) -> None:
    """If AOL is active but panda's lateral_allowed disagrees for ~2s, force AOL off.

    pandaStates goes over a different socket than CAN messages, so allow a couple
    cycles of mismatch before acting (avoids a race where one update arrives before
    the other on a single frame).
    """
    IGNORED_SAFETY_MODES = (car.CarParams.SafetyModel.silent, car.CarParams.SafetyModel.noOutput)
    sm = self.sd.sm
    if not self.active or self.sd.enabled:
      self._lateral_mismatch_counter = 0
      return
    if not sm.alive['pandaStates'] or not sm.valid['pandaStates']:
      return
    mismatched = any(not ps.lateralAllowed for ps in sm['pandaStates'] if ps.safetyModel not in IGNORED_SAFETY_MODES)
    if mismatched:
      self._lateral_mismatch_counter += 1
    else:
      self._lateral_mismatch_counter = 0
    if self._lateral_mismatch_counter >= 200:
      cloudlog.error("AOL: panda lateral_allowed mismatch for 2s, forcing immediateDisable")
      self.state_machine.add_event('immediateDisable')

  def update_events(self, CS) -> None:
    """Process car state and generate AOL events for the state machine."""
    self.state_machine.clear_events()
    self.check_panda_mismatch()

    cruise_available = CS.cruiseState.available

    # Driver is manually overpowering AOL's steering — track as 'overriding' instead of 'enabled'
    if CS.steeringPressed:
      self.state_machine.add_event('overrideLateral')

    # LKAS/LFA button: rising edge toggles AOL on/off
    lkas_pressed = any(be.type == ButtonType.lkas and be.pressed for be in CS.buttonEvents)
    if lkas_pressed:
      if self.enabled:
        cloudlog.warning("AOL: userDisable (lkas button)")
        self.state_machine.add_event('userDisable')
      else:
        if cruise_available or self.allow_always:
          self.state_machine.add_event('lkasEnable')

    # Cruise main button: rising edge toggles AOL on/off.
    # On Hyundai CANFD (e.g. Palisade), cruiseState.available never changes so we
    # must watch the raw button event directly.
    main_pressed = any(be.type == ButtonType.mainCruise and be.pressed for be in CS.buttonEvents)
    if main_pressed:
      if self.enabled:
        cloudlog.warning("AOL: userDisable (mainCruise button)")
        self.state_machine.add_event('userDisable')
      else:
        self.state_machine.add_event('lkasEnable')

    # ACC main rising edge → activate AOL (for cars where cruiseState.available tracks
    # the ACC main switch). Skipped on allow_always cars (Hyundai CANFD, e.g. Palisade):
    # there available means "no TCS fault" and is true from the first frame after
    # ignition, so this edge would phantom-enable AOL at every start while the panda's
    # lateral_allowed latch (armed only by a real button press) stays off — panda then
    # blocks the LKAS frames and the MDPS faults until the user toggles AOL manually.
    if not self.allow_always and self._cruise_available_prev is not None:
      if cruise_available and not self._cruise_available_prev:
        if not self.enabled:
          self.state_machine.add_event('lkasEnable')

    # ACC main falling edge → deactivate AOL (for cars where cruiseState.available changes;
    # on allow_always cars this only fires on a real TCS fault, where disabling is also right)
    if self._cruise_available_prev is not None and not cruise_available and self._cruise_available_prev:
      if self.enabled:
        cloudlog.warning("AOL: immediateDisable (cruiseState.available falling edge)")
        self.state_machine.add_event('immediateDisable')

    self._cruise_available_prev = cruise_available

    # Safety: door open or park brake → pause
    pause_requested = (hasattr(CS, 'doorOpen') and CS.doorOpen) or \
                      (hasattr(CS, 'parkingBrake') and CS.parkingBrake)
    if pause_requested:
      self.state_machine.add_event('canPause')
    elif self.state_machine.state == State.paused:
      # Auto-resume once the pause cause clears — paused otherwise exits only on
      # a button press, so a closed door would leave AOL off for the whole drive
      self.state_machine.add_event('silentLkasEnable')

    # Steering fault → disable
    if CS.steerFaultPermanent:
      if self.enabled:
        cloudlog.error("AOL: immediateDisable (steerFaultPermanent)")
      self.state_machine.add_event('immediateDisable')
    elif CS.steerFaultTemporary:
      # The Palisade MDPS raises a temporary fault (ToiUnavail) about a second
      # after controlsd's standstill gate drops latActive at every stop, and it
      # persists until the car moves off — long enough to run out the 3s soft
      # disable timer at any normal stoplight. Steering is already blocked by
      # controlsd while the fault or standstill persists, so the fault is benign
      # here: only treat it as real while the car is moving. Mirror controlsd's
      # standstill expression so the two gates agree.
      standstill = abs(CS.vEgo) <= max(self.CP.minSteerSpeed, 0.3) or CS.standstill
      if not standstill:
        if self.active:
          cloudlog.warning("AOL: softDisable (steerFaultTemporary)")
        self.state_machine.add_event('softDisable')

    # REMAIN_ACTIVE: brake/gas press does NOT affect AOL state

  def update(self, CS) -> None:
    prev_enabled = self.enabled
    self.enabled, self.active = self.state_machine.update()

    # One-shot warning when AOL turns itself off (steer fault, panda mismatch,
    # cruise loss) — a user button press is intentional and gets no alert
    if prev_enabled and not self.enabled and not self.state_machine.has('userDisable'):
      self._disable_alert_frames = int(DISABLE_ALERT_TIME / DT_CTRL)
    elif self._disable_alert_frames > 0:
      self._disable_alert_frames -= 1

  def create_lane_change_alerts(self, callback_args: list) -> list:
    """Surface turn-signal / lane-change alerts while AOL steers without full engagement.

    selfdrived adds the lane-change events unconditionally, but update_alerts only
    renders alert types permitted by the main state machine — [PERMANENT] when not
    engaged — so the WARNING-type lane-change prompts never show during AOL-only
    driving even though the lane changes themselves happen (modeld's DesireHelper
    keys off carControl.latActive). Return just those alerts for selfdrived to add;
    every other WARNING stays suppressed as stock intends.
    """
    if self.sd.enabled or not self.active:
      return []
    active_lane_events = [e for e in LANE_CHANGE_EVENTS if e in self.sd.events.events]
    if not active_lane_events:
      return []
    lane_change_events = Events()
    for event_name in active_lane_events:
      lane_change_events.add(event_name)
    return lane_change_events.create_alerts([ET.WARNING], callback_args)

  def create_aol_alerts(self) -> list:
    """Alerts for AOL state changes that stock plumbing renders nothing for.

    WARNING alerts only render while fully engaged, so during AOL-only driving a
    steer-fault soft disable or a forced AOL disable is invisible — exactly when
    the driver most needs to know steering stopped.
    """
    if self.sd.enabled:
      return []
    alerts = []
    if self.state_machine.state == State.softDisabling:
      a = Alert("Steering Temporarily Unavailable", "Always-On Lateral disengaging",
                AlertStatus.userPrompt, AlertSize.small, Priority.LOW,
                VisualAlert.steerRequired, AudibleAlert.prompt, .2)
      a.alert_type = "aolSoftDisabling/warning"
      a.event_type = ET.WARNING
      alerts.append(a)
    if self._disable_alert_frames > 0 and not self.enabled:
      a = Alert("Always-On Lateral DISENGAGED", "Steering is off",
                AlertStatus.userPrompt, AlertSize.mid, Priority.MID,
                VisualAlert.steerRequired, AudibleAlert.warningSoft, .2)
      a.alert_type = "aolDisengaged/warning"
      a.event_type = ET.WARNING
      alerts.append(a)
    return alerts

  def publish(self, pm) -> None:
    msg = messaging.new_message('spysydriveStateSP')
    msg.valid = True
    aol = msg.spysydriveStateSP.aol
    aol.state = self.state_machine.state
    aol.enabled = self.enabled
    aol.active = self.active
    aol.available = self.available
    pm.send('spysydriveStateSP', msg)

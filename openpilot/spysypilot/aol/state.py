from openpilot.cereal import custom
from openpilot.selfdrive.selfdrived.events import ET

State = custom.AolState.AolStateEnum

ACTIVE_STATES = (State.enabled, State.softDisabling, State.overriding)
ENABLED_STATES = (State.paused, State.enabled, State.softDisabling, State.overriding)

SOFT_DISABLE_TIME = 3.0  # seconds


class AolStateMachine:
  def __init__(self, DT_CTRL: float):
    self.DT_CTRL = DT_CTRL
    self.state = State.disabled
    self.soft_disable_timer = 0.0
    self._current_events: set = set()

  def add_event(self, event: str) -> None:
    self._current_events.add(event)

  def clear_events(self) -> None:
    self._current_events.clear()

  def has(self, event: str) -> bool:
    return event in self._current_events

  def update(self) -> tuple[bool, bool]:
    """Update state machine. Returns (enabled, active)."""

    if self.has('immediateDisable') or self.has('userDisable'):
      if self.state != State.disabled:
        self.state = State.disabled

    elif self.state == State.disabled:
      if self.has('lkasEnable') or self.has('silentLkasEnable'):
        if self.has('noEntry'):
          if self.has('canPause'):
            self.state = State.paused
        else:
          self.state = State.enabled

    elif self.state == State.paused:
      if self.has('lkasEnable') or self.has('silentLkasEnable'):
        if not self.has('noEntry'):
          if self.has('overrideLateral'):
            self.state = State.overriding
          else:
            self.state = State.enabled

    elif self.state == State.enabled:
      if self.has('softDisable'):
        self.state = State.softDisabling
        self.soft_disable_timer = SOFT_DISABLE_TIME
      elif self.has('overrideLateral'):
        self.state = State.overriding
      elif self.has('canPause'):
        self.state = State.paused

    elif self.state == State.softDisabling:
      if self.has('lkasEnable') or self.has('silentLkasEnable'):
        if not self.has('noEntry'):
          self.state = State.enabled
      elif not self.has('softDisable'):
        self.state = State.enabled
      else:
        self.soft_disable_timer -= self.DT_CTRL
        if self.soft_disable_timer <= 0:
          self.state = State.disabled

    elif self.state == State.overriding:
      if not self.has('overrideLateral'):
        self.state = State.enabled

    enabled = self.state in ENABLED_STATES
    active = self.state in ACTIVE_STATES
    return enabled, active

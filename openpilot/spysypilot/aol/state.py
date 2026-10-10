from openpilot.cereal import custom

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
    """Faults and pauses take priority over driver override and enable requests."""
    enable = self.has('lkasEnable') or self.has('silentLkasEnable')
    active_state = State.overriding if self.has('overrideLateral') else State.enabled

    if self.has('immediateDisable') or self.has('userDisable'):
      self.state = State.disabled
    elif self.state in (State.disabled, State.paused):
      if enable and not self.has('softDisable'):
        self.state = State.paused if self.has('canPause') else active_state
    elif self.has('canPause'):
      self.state = State.paused
    elif self.has('softDisable'):
      if self.state != State.softDisabling:
        self.state = State.softDisabling
        self.soft_disable_timer = SOFT_DISABLE_TIME
      else:
        self.soft_disable_timer = max(0.0, self.soft_disable_timer - self.DT_CTRL)
        if self.soft_disable_timer <= 0:
          self.state = State.disabled
    else:
      self.state = active_state

    if self.state != State.softDisabling:
      self.soft_disable_timer = 0.0
    return self.state in ENABLED_STATES, self.state in ACTIVE_STATES

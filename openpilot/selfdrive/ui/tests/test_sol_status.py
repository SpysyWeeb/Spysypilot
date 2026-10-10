from types import SimpleNamespace
from typing import Any

from openpilot.cereal import custom, log
from openpilot.selfdrive.ui.ui_state import UIState, UIStatus


def test_sol_steering_has_visible_status_and_stale_state_clears_it():
  assert hasattr(UIStatus, 'AOL_ACTIVE')

  class Sm(dict):
    updated = {'selfdriveState': True, 'spysydriveStateSP': True}
    alive = {'spysydriveStateSP': True}
    valid = {'spysydriveStateSP': True}
    frame = 2

  ui: Any = UIState.__new__(UIState)
  ui.started = True
  ui.status = UIStatus.DISENGAGED
  ui._engaged_prev = False
  ui._started_prev = True
  ui._engaged_transition_callbacks = []
  ui._offroad_transition_callbacks = []
  aol = SimpleNamespace(state=custom.AolState.AolStateEnum.enabled, active=True)
  ui.sm = Sm(selfdriveState=SimpleNamespace(state=log.SelfdriveState.OpenpilotState.disabled, enabled=False),
             spysydriveStateSP=SimpleNamespace(aol=aol))
  ui._update_status()
  assert ui.status == UIStatus.AOL_ACTIVE
  aol.state = custom.AolState.AolStateEnum.overriding
  ui._update_status()
  assert ui.status == UIStatus.OVERRIDE
  ui.sm.alive['spysydriveStateSP'] = False
  ui.sm.updated = {'selfdriveState': False, 'spysydriveStateSP': False}
  ui._update_status()
  assert ui.status == UIStatus.DISENGAGED

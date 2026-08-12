from types import SimpleNamespace
from typing import Any, cast

from openpilot.cereal import custom, log
from openpilot.selfdrive.ui.ui_state import UIState, UIStatus


class FakeSubMaster(dict):
  updated = {"selfdriveState": False, "spysydriveStateSP": True}
  frame = 2


def test_sol_driver_override_uses_override_status():
  ui = object.__new__(UIState)
  ui.started = True
  ui.status = UIStatus.AOL_ACTIVE
  ui._engaged_prev = False
  ui._started_prev = True
  ui._engaged_transition_callbacks = []
  ui._offroad_transition_callbacks = []
  ui.sm = cast(Any, FakeSubMaster(
    selfdriveState=SimpleNamespace(state=log.SelfdriveState.OpenpilotState.disabled, enabled=False),
    spysydriveStateSP=SimpleNamespace(aol=SimpleNamespace(
      state=custom.AolState.AolStateEnum.overriding,
      active=True,
    )),
  ))

  ui._update_status()

  assert ui.status == UIStatus.OVERRIDE

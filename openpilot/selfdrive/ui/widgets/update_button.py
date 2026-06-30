import os
import pyray as rl
from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight, FONT_SCALE
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets import Widget

UPDATER_PROC = "openpilot.system.updated.updated"

# Updater internal states -> short label shown while an update is actively in progress
STATE_LABELS = {
  "checking...": "CHECKING...",
  "downloading...": "DOWNLOADING...",
  "finalizing update...": "FINALIZING...",
}


class UpdateButton(Widget):
  def __init__(self):
    super().__init__()
    self._params = Params()
    self._label = tr("CHECK FOR UPDATE")
    self._auto_fetch_pending = False

  def _update_state(self) -> None:
    state = self._params.get("UpdaterState") or "idle"
    fetch_available = self._params.get_bool("UpdaterFetchAvailable")

    if state != "idle":
      # An update is actively in progress; reflect the live status
      self._label = tr(STATE_LABELS.get(state, state.upper()))
    elif (self._params.get("UpdateFailedCount") or 0) > 0:
      self._label = tr("UPDATE FAILED")
    elif self._params.get_bool("UpdateAvailable"):
      self._label = tr("INSTALL UPDATE")
    elif fetch_available:
      if self._auto_fetch_pending:
        self._auto_fetch_pending = False
        os.system(f"pkill -SIGHUP -f {UPDATER_PROC}")
      self._label = tr("DOWNLOAD UPDATE")
    else:
      self._label = tr("CHECK FOR UPDATE")

  def _handle_mouse_release(self, mouse_pos) -> None:
    super()._handle_mouse_release(mouse_pos)

    # Ignore taps while the updater is busy or while onroad (updates only run offroad)
    state = self._params.get("UpdaterState") or "idle"
    if state != "idle" or not ui_state.is_offroad():
      return

    if self._params.get_bool("UpdateAvailable"):
      # Update already downloaded -> reboot to install
      self._params.put_bool("DoReboot", True, block=True)
    elif self._params.get_bool("UpdaterFetchAvailable"):
      # Update available -> start the download
      os.system(f"pkill -SIGHUP -f {UPDATER_PROC}")
    else:
      # Kick off a fresh check; auto-trigger download if one is found
      self._auto_fetch_pending = True
      os.system(f"pkill -SIGUSR1 -f {UPDATER_PROC}")

  def _render(self, rect: rl.Rectangle) -> None:
    alpha = 0xCC if self.is_pressed else 0xFF
    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
    rl.draw_rectangle(int(rect.x), int(rect.y), int(rect.width), int(rect.height), rl.Color(80, 80, 80, alpha))
    rl.draw_rectangle_rounded_lines_ex(self._rect, 0.19, 10, 5, rl.BLACK)
    rl.end_scissor_mode()

    text_y = rect.y + rect.height / 2 - 45 * FONT_SCALE // 2
    rl.draw_text_ex(gui_app.font(FontWeight.NORMAL), self._label, rl.Vector2(int(rect.x + 25), int(text_y)), 45, 0, rl.WHITE)

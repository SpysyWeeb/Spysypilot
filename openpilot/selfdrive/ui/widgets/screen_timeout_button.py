import pyray as rl
from openpilot.common.params import Params
from openpilot.system.ui.lib.application import gui_app, FontWeight, FONT_SCALE
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets import Widget


class ScreenTimeoutButton(Widget):
  def __init__(self):
    super().__init__()
    self._params = Params()
    self._always_on: bool = self._params.get_bool("ScreenAlwaysOn")

  def _update_state(self) -> None:
    self._always_on = self._params.get_bool("ScreenAlwaysOn")

  def _handle_mouse_release(self, _):
    super()._handle_mouse_release(_)
    self._always_on = not self._always_on
    self._params.put_bool("ScreenAlwaysOn", self._always_on)

  def _render(self, rect: rl.Rectangle) -> None:
    alpha = 0xCC if self.is_pressed else 0xFF
    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
    rl.draw_rectangle(int(rect.x), int(rect.y), int(rect.width), int(rect.height), rl.Color(80, 80, 80, alpha))
    rl.draw_rectangle_rounded_lines_ex(self._rect, 0.19, 10, 5, rl.BLACK)
    rl.end_scissor_mode()

    text = tr("SCREEN ALWAYS ON") if self._always_on else tr("AUTO SCREEN TIMEOUT")
    text_y = rect.y + rect.height / 2 - 45 * FONT_SCALE // 2
    rl.draw_text_ex(gui_app.font(FontWeight.NORMAL), text, rl.Vector2(int(rect.x + 25), int(text_y)), 45, 0, rl.WHITE)

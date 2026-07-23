import time

import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.text_measure import measure_text_cached


class LatEventToast:
  DISPLAY_SECONDS = 2.5

  def __init__(self):
    self._font = gui_app.font(FontWeight.SEMI_BOLD)
    self._last_recv_frame = -1
    self._visible_until = 0.0

  def render(self, rect: rl.Rectangle) -> None:
    recv_frame = ui_state.sm.recv_frame["lateralEvent"]
    if recv_frame > self._last_recv_frame:
      self._last_recv_frame = recv_frame
      if ui_state.sm.updated["lateralEvent"] and ui_state.sm.valid["lateralEvent"]:
        self._visible_until = time.monotonic() + self.DISPLAY_SECONDS

    remaining = self._visible_until - time.monotonic()
    if remaining <= 0.0:
      return

    text = "Lat Event Logged"
    scale = max(0.65, min(rect.width / 1920.0, rect.height / 1080.0))
    font_size = round(56 * scale)
    padding_x = round(38 * scale)
    padding_y = round(20 * scale)
    text_size = measure_text_cached(self._font, text, font_size)
    width = text_size.x + 2 * padding_x
    height = text_size.y + 2 * padding_y
    x = rect.x + (rect.width - width) / 2
    y = rect.y + rect.height * 0.76

    fade = min(1.0, remaining / 0.35)
    alpha = round(220 * fade)
    toast_rect = rl.Rectangle(x, y, width, height)
    rl.draw_rectangle_rounded(toast_rect, 0.35, 12, rl.Color(18, 24, 28, alpha))
    rl.draw_rectangle_rounded_lines_ex(toast_rect, 0.35, 12, max(2, round(3 * scale)), rl.Color(51, 205, 122, alpha))
    text_pos = rl.Vector2(x + padding_x, y + padding_y)
    rl.draw_text_ex(self._font, text, text_pos, font_size, 0, rl.Color(255, 255, 255, round(255 * fade)))

import html
import os
import pyray as rl
from openpilot.system.ui.lib.application import gui_app, FontWeight, FONT_SCALE
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.html_render import HtmlModal

ERROR_LOG_PATH = "/data/community/crashes/error.log"


class ErrorLogButton(Widget):
  def _handle_mouse_release(self, mouse_pos) -> None:
    super()._handle_mouse_release(mouse_pos)

    if os.path.exists(ERROR_LOG_PATH):
      with open(ERROR_LOG_PATH) as f:
        content = f.read()
      # Preserve per-line structure (the renderer otherwise collapses whitespace)
      body = html.escape(content).replace("\n", "<br>") if content.strip() else "<p>No errors logged yet.</p>"
    else:
      body = "<p>No errors logged yet.</p>"

    gui_app.push_widget(HtmlModal(text=body))

  def _render(self, rect: rl.Rectangle) -> None:
    alpha = 0xCC if self.is_pressed else 0xFF
    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
    rl.draw_rectangle(int(rect.x), int(rect.y), int(rect.width), int(rect.height), rl.Color(226, 44, 44, alpha))
    rl.draw_rectangle_rounded_lines_ex(self._rect, 0.19, 10, 5, rl.BLACK)
    rl.end_scissor_mode()

    text = tr("ERROR LOG")
    text_y = rect.y + rect.height / 2 - 45 * FONT_SCALE // 2
    rl.draw_text_ex(gui_app.font(FontWeight.NORMAL), text, rl.Vector2(int(rect.x + 25), int(text_y)), 45, 0, rl.WHITE)

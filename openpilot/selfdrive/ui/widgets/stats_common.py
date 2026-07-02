import time
import pyray as rl

from openpilot.common.params import Params
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import Widget

REFRESH_INTERVAL = 2.0

BG      = rl.Color(40, 40, 40, 255)
BLUE    = rl.Color(70, 91, 234, 255)
DIM     = rl.Color(255, 255, 255, 150)
DIVIDER = rl.Color(255, 255, 255, 35)


class StatsPageWidget(Widget):
    """Shared base for the tappable left-column 'window' pages driven by drive_statsd's
    Params output. Keeps the analyzer status line in the same spot on every page, so it
    reads as the same ongoing process no matter which page you're currently looking at."""

    def __init__(self):
        super().__init__()
        self._params = Params()
        self._status: str = ""
        self._last_refresh = 0.0
        self._background_tap_callback = None

    def set_background_tap_callback(self, cb) -> None:
        self._background_tap_callback = cb

    def _handle_mouse_release(self, mouse_pos) -> None:
        if self._background_tap_callback:
            self._background_tap_callback(mouse_pos)

    def _maybe_reload(self):
        now = time.monotonic()
        if now - self._last_refresh >= REFRESH_INTERVAL:
            self._status = self._params.get("SpysyStatsStatus") or ""
            self._reload()
            self._last_refresh = now

    def _reload(self):
        """Override to pull additional Params on refresh."""

    def _draw_frame(self, rect: rl.Rectangle) -> tuple[int, int, int]:
        """Draws the background and (if present) the analyzer status line.
        Returns (x, y, w) for the caller to keep drawing content below it."""
        rl.draw_rectangle_rounded(rect, 0.025, 10, BG)
        pad = 50
        x = int(rect.x + pad)
        w = int(rect.width - pad * 2)
        y = int(rect.y + 45)
        if self._status:
            fn = gui_app.font(FontWeight.NORMAL)
            rl.draw_text_ex(fn, self._status, rl.Vector2(x, y), 32, 0, DIM)
            y += 48
        return x, y, w

    def _draw_section(self, title: str, x: int, y: int, w: int) -> int:
        font = gui_app.font(FontWeight.BOLD)
        rl.draw_text_ex(font, title, rl.Vector2(x, y), 34, 0, BLUE)
        line_y = y + 46
        rl.draw_line_ex(rl.Vector2(x, line_y), rl.Vector2(x + w, line_y), 1, DIVIDER)
        return line_y + 18

    def _draw_stat_triple(self, x: int, y: int, w: int,
                          l1: str, p1: str, s1: str, c1: rl.Color,
                          l2: str, p2: str, s2: str, c2: rl.Color,
                          l3: str, p3: str, s3: str, c3: rl.Color) -> int:
        fn = gui_app.font(FontWeight.NORMAL)
        fb = gui_app.font(FontWeight.BOLD)
        third = w // 3
        for i, (label, primary, secondary, color) in enumerate(
            [(l1, p1, s1, c1), (l2, p2, s2, c2), (l3, p3, s3, c3)]
        ):
            col_x = x + i * third
            rl.draw_text_ex(fn, label,     rl.Vector2(col_x, y),       38, 0, DIM)
            rl.draw_text_ex(fb, primary,   rl.Vector2(col_x, y + 46),  52, 0, color)
            rl.draw_text_ex(fn, secondary, rl.Vector2(col_x, y + 106), 38, 0, color)
        return y + 152

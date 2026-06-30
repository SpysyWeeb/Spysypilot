import json
import time
import pyray as rl

from openpilot.common.params import Params
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import Widget

REFRESH_INTERVAL = 5.0

_BG      = rl.Color(40, 40, 40, 255)
_BLUE    = rl.Color(70, 91, 234, 255)
_GREEN   = rl.Color(70, 200, 100, 255)
_ORANGE  = rl.Color(234, 160, 50, 255)
_DIM     = rl.Color(255, 255, 255, 150)
_DIVIDER = rl.Color(255, 255, 255, 35)


class DriveStatsWidget(Widget):
    def __init__(self):
        super().__init__()
        self._params = Params()
        self._lifetime: dict | None = None
        self._last_drive: dict | None = None
        self._status: str = ""
        self._last_refresh = 0.0

    def _render(self, rect: rl.Rectangle):
        now = time.monotonic()
        if now - self._last_refresh >= REFRESH_INTERVAL:
            self._reload()
            self._last_refresh = now

        rl.draw_rectangle_rounded(rect, 0.025, 10, _BG)

        pad = 50
        x = int(rect.x + pad)
        w = int(rect.width - pad * 2)
        y = int(rect.y + 45)

        if self._status:
            fn = gui_app.font(FontWeight.NORMAL)
            rl.draw_text_ex(fn, self._status, rl.Vector2(x, y), 32, 0, _DIM)
            y += 48

        y = self._draw_section("LIFETIME DRIVING", x, y, w)
        y = self._draw_stat_pair(
            x, y, w,
            "Engaged",    self._fmt_mi('lifetime', 'engaged_mi'),    _GREEN,
            "Disengaged", self._fmt_mi('lifetime', 'disengaged_mi'), _ORANGE,
        )
        y += 38

        y = self._draw_section("LAST DRIVE", x, y, w)
        y = self._draw_stat_pair(
            x, y, w,
            "Engaged",    self._fmt_pct('engaged_pct'),    _GREEN,
            "Disengaged", self._fmt_pct('disengaged_pct'), _ORANGE,
        )
        y += 38

        y = self._draw_section("DISENGAGEMENT REASONS", x, y, w)
        self._draw_reasons(x, y, w)

    def _reload(self):
        self._status = self._params.get("SpysyStatsStatus") or ""
        for attr, key in (('_lifetime', 'SpysyLifetimeStats'), ('_last_drive', 'SpysyLastDriveStats')):
            raw = self._params.get(key)
            try:
                setattr(self, attr, json.loads(raw) if raw else None)
            except Exception:
                setattr(self, attr, None)

    def _fmt_mi(self, source: str, field: str) -> str:
        data = self._lifetime if source == 'lifetime' else self._last_drive
        if data is None:
            return "—"
        return f"{data.get(field, 0.0):,.1f} mi"

    def _fmt_pct(self, field: str) -> str:
        if self._last_drive is None:
            return "—"
        return f"{self._last_drive.get(field, 0.0):.1f}%"

    def _draw_section(self, title: str, x: int, y: int, w: int) -> int:
        font = gui_app.font(FontWeight.BOLD)
        rl.draw_text_ex(font, title, rl.Vector2(x, y), 34, 0, _BLUE)
        line_y = y + 46
        rl.draw_line_ex(rl.Vector2(x, line_y), rl.Vector2(x + w, line_y), 1, _DIVIDER)
        return line_y + 18

    def _draw_stat_pair(self, x: int, y: int, w: int,
                        left_label: str, left_val: str, left_color: rl.Color,
                        right_label: str, right_val: str, right_color: rl.Color) -> int:
        fn = gui_app.font(FontWeight.NORMAL)
        fb = gui_app.font(FontWeight.BOLD)
        half = w // 2
        rl.draw_text_ex(fn, left_label,  rl.Vector2(x,        y),      38, 0, _DIM)
        rl.draw_text_ex(fn, right_label, rl.Vector2(x + half, y),      38, 0, _DIM)
        rl.draw_text_ex(fb, left_val,    rl.Vector2(x,        y + 48), 54, 0, left_color)
        rl.draw_text_ex(fb, right_val,   rl.Vector2(x + half, y + 48), 54, 0, right_color)
        return y + 118

    def _draw_reasons(self, x: int, y: int, w: int):
        fn = gui_app.font(FontWeight.NORMAL)
        fb = gui_app.font(FontWeight.BOLD)

        if self._last_drive and self._last_drive.get('reasons'):
            r = self._last_drive['reasons']
            if sum(r.values()) == 0:
                rl.draw_text_ex(fn, "No interruptions recorded", rl.Vector2(x, y + 12), 40, 0, _DIM)
                return
            items = [
                ("Gas override", f"{r.get('gas', 0.0):.0f}%"),
                ("Steering",     f"{r.get('steer', 0.0):.0f}%"),
                ("Brake",        f"{r.get('brake', 0.0):.0f}%"),
                ("Cancel",       f"{r.get('cancel', 0.0):.0f}%"),
            ]
        else:
            rl.draw_text_ex(fn, "No drive data yet", rl.Vector2(x, y + 12), 40, 0, _DIM)
            return

        col_w = w // 4
        for i, (label, val) in enumerate(items):
            cx = x + i * col_w
            rl.draw_text_ex(fn, label, rl.Vector2(cx, y),      34, 0, _DIM)
            rl.draw_text_ex(fb, val,   rl.Vector2(cx, y + 44), 50, 0, rl.WHITE)

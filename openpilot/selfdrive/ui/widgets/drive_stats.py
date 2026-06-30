import json
import time
import pyray as rl

from openpilot.common.params import Params
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import Widget

REFRESH_INTERVAL = 2.0

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
        self._background_tap_callback = None

    def set_background_tap_callback(self, cb) -> None:
        self._background_tap_callback = cb

    def _handle_mouse_release(self, mouse_pos) -> None:
        if self._background_tap_callback:
            self._background_tap_callback()

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
            "Engaged",
            self._lifetime_pct('engaged'),
            self._fmt_mi('lifetime', 'engaged_mi'),
            _GREEN,
            "Disengaged",
            self._lifetime_pct('disengaged'),
            self._fmt_mi('lifetime', 'disengaged_mi'),
            _ORANGE,
        )
        y += 25

        y = self._draw_section("LAST DRIVE", x, y, w)
        y = self._draw_stat_pair(
            x, y, w,
            "Engaged",
            self._fmt_pct('engaged_pct'),
            self._fmt_mi('last', 'engaged_mi'),
            _GREEN,
            "Disengaged",
            self._fmt_pct('disengaged_pct'),
            self._fmt_mi('last', 'disengaged_mi'),
            _ORANGE,
        )
        y += 25

        y = self._draw_section("OVERRIDES & DISENGAGEMENTS", x, y, w)
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

    def _lifetime_pct(self, side: str) -> str:
        if self._lifetime is None:
            return "—"
        eng = self._lifetime.get('engaged_mi', 0.0)
        dis = self._lifetime.get('disengaged_mi', 0.0)
        total = eng + dis
        if total == 0:
            return "—"
        pct = eng / total * 100 if side == 'engaged' else dis / total * 100
        return f"{pct:.1f}%"

    def _draw_section(self, title: str, x: int, y: int, w: int) -> int:
        font = gui_app.font(FontWeight.BOLD)
        rl.draw_text_ex(font, title, rl.Vector2(x, y), 34, 0, _BLUE)
        line_y = y + 46
        rl.draw_line_ex(rl.Vector2(x, line_y), rl.Vector2(x + w, line_y), 1, _DIVIDER)
        return line_y + 18

    def _draw_stat_pair(self, x: int, y: int, w: int,
                        left_label: str, left_primary: str, left_secondary: str, left_color: rl.Color,
                        right_label: str, right_primary: str, right_secondary: str, right_color: rl.Color) -> int:
        fn = gui_app.font(FontWeight.NORMAL)
        fb = gui_app.font(FontWeight.BOLD)
        half = w // 2

        rl.draw_text_ex(fn, left_label,  rl.Vector2(x,        y), 38, 0, _DIM)
        rl.draw_text_ex(fn, right_label, rl.Vector2(x + half, y), 38, 0, _DIM)

        rl.draw_text_ex(fb, left_primary,  rl.Vector2(x,        y + 46), 52, 0, left_color)
        rl.draw_text_ex(fb, right_primary, rl.Vector2(x + half, y + 46), 52, 0, right_color)

        rl.draw_text_ex(fn, left_secondary,  rl.Vector2(x,        y + 106), 38, 0, left_color)
        rl.draw_text_ex(fn, right_secondary, rl.Vector2(x + half, y + 106), 38, 0, right_color)

        return y + 152

    def _draw_reasons(self, x: int, y: int, w: int):
        fn = gui_app.font(FontWeight.NORMAL)
        fb = gui_app.font(FontWeight.BOLD)

        if self._last_drive and self._last_drive.get('reasons'):
            r = self._last_drive['reasons']
            if sum(r.values()) == 0:
                rl.draw_text_ex(fn, "No interruptions recorded", rl.Vector2(x, y + 12), 40, 0, _DIM)
                return
            overrides  = [("Accel",    f"{r.get('gas', 0.0):.0f}%"),
                          ("Steering", f"{r.get('steer', 0.0):.0f}%")]
            disengages = [("Braking", f"{r.get('brake', 0.0):.0f}%"),
                          ("Cancel",  f"{r.get('cancel', 0.0):.0f}%")]
        else:
            rl.draw_text_ex(fn, "No drive data yet", rl.Vector2(x, y + 12), 40, 0, _DIM)
            return

        half = w // 2
        quarter = w // 4
        divider_x = x + half

        rl.draw_line_ex(rl.Vector2(divider_x, y), rl.Vector2(divider_x, y + 116), 1, _DIVIDER)

        ow = int(rl.measure_text_ex(fn, "Overrides", 30, 0).x)
        dw = int(rl.measure_text_ex(fn, "Disengagements", 30, 0).x)
        rl.draw_text_ex(fn, "Overrides",      rl.Vector2(x + (half - ow) // 2,         y), 30, 0, _DIM)
        rl.draw_text_ex(fn, "Disengagements", rl.Vector2(divider_x + (half - dw) // 2, y), 30, 0, _DIM)

        row_y = y + 38
        for i, (label, val) in enumerate(overrides + disengages):
            col_center = x + i * quarter + quarter // 2
            lw = int(rl.measure_text_ex(fn, label, 34, 0).x)
            vw = int(rl.measure_text_ex(fb, val,   50, 0).x)
            rl.draw_text_ex(fn, label, rl.Vector2(col_center - lw // 2, row_y),      34, 0, _DIM)
            rl.draw_text_ex(fb, val,   rl.Vector2(col_center - vw // 2, row_y + 42), 50, 0, rl.WHITE)

import json
import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.selfdrive.ui.widgets.stats_common import StatsPageWidget, DIM, DIVIDER

_GREEN    = rl.Color(70, 200, 100, 255)
_ORANGE   = rl.Color(234, 160, 50, 255)
_AOL      = rl.Color(180, 130, 255, 255)


class DriveStatsWidget(StatsPageWidget):
    def __init__(self):
        super().__init__()
        self._lifetime: dict | None = None
        self._last_drive: dict | None = None

    def _render(self, rect: rl.Rectangle):
        self._maybe_reload()
        x, y, w = self._draw_frame(rect)

        y = self._draw_section("LIFETIME DRIVING", x, y, w)
        y = self._draw_stat_triple(
            x, y, w,
            "Engaged",    self._lifetime_pct("engaged"),    self._fmt_lifetime_mi("engaged_mi"),    _GREEN,
            "AOL",        self._lifetime_pct("aol"),        self._fmt_lifetime_mi("aol_mi"),        _AOL,
            "Disengaged", self._lifetime_pct("disengaged"), self._fmt_lifetime_mi("disengaged_mi"), _ORANGE,
        )
        y += 25

        y = self._draw_section("LAST DRIVE", x, y, w)
        y = self._draw_stat_triple(
            x, y, w,
            "Engaged",    self._fmt_pct("engaged_pct"),    self._fmt_drive_mi("engaged_mi"),    _GREEN,
            "AOL",        self._fmt_pct("aol_pct"),        self._fmt_drive_mi("aol_mi"),        _AOL,
            "Disengaged", self._fmt_pct("disengaged_pct"), self._fmt_drive_mi("disengaged_mi"), _ORANGE,
        )
        y += 25

        y = self._draw_section("OVERRIDES & DISENGAGEMENTS", x, y, w)
        self._draw_reasons(x, y, w)

    def _reload(self):
        for attr, key in (("_lifetime", "SpysyLifetimeStats"), ("_last_drive", "SpysyLastDriveStats")):
            raw = self._params.get(key)
            try:
                setattr(self, attr, json.loads(raw) if raw else None)
            except Exception:
                setattr(self, attr, None)

    def _fmt_lifetime_mi(self, field: str) -> str:
        if self._lifetime is None or field not in self._lifetime:
            return "N/A"
        return f"{self._lifetime.get(field, 0.0):,.1f} mi"

    def _fmt_drive_mi(self, field: str) -> str:
        if self._last_drive is None or field not in self._last_drive:
            return "N/A"
        return f"{self._last_drive.get(field, 0.0):,.1f} mi"

    def _fmt_pct(self, field: str) -> str:
        if self._last_drive is None or field not in self._last_drive:
            return "N/A"
        return f"{self._last_drive.get(field, 0.0):.1f}%"

    def _lifetime_pct(self, side: str) -> str:
        if self._lifetime is None:
            return "N/A"
        eng = self._lifetime.get("engaged_mi", 0.0)
        aol = self._lifetime.get("aol_mi", 0.0)
        dis = self._lifetime.get("disengaged_mi", 0.0)
        total = eng + aol + dis
        if total == 0:
            return "N/A"
        value = {"engaged": eng, "aol": aol, "disengaged": dis}[side]
        return f"{value / total * 100:.1f}%"

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

    def _draw_reasons(self, x: int, y: int, w: int):
        fn = gui_app.font(FontWeight.NORMAL)
        fb = gui_app.font(FontWeight.BOLD)

        if self._last_drive and self._last_drive.get("reasons"):
            r = self._last_drive["reasons"]
            if sum(r.values()) == 0:
                rl.draw_text_ex(fn, "No interruptions recorded", rl.Vector2(x, y + 12), 40, 0, DIM)
                return
            # AOL counts as an override reason: it's steering-active middle ground, not a
            # full disengagement, so it belongs alongside Accel/Steering rather than Braking/Cancel.
            overrides  = [("Accel",    f"{r.get('gas', 0.0):.0f}%"),
                          ("Steering", f"{r.get('steer', 0.0):.0f}%"),
                          ("AOL",      f"{r.get('aol', 0.0):.0f}%")]
            disengages = [("Braking", f"{r.get('brake', 0.0):.0f}%"),
                          ("Cancel",  f"{r.get('cancel', 0.0):.0f}%")]
        else:
            rl.draw_text_ex(fn, "No drive data yet", rl.Vector2(x, y + 12), 40, 0, DIM)
            return

        half = w // 2
        divider_x = x + half

        rl.draw_line_ex(rl.Vector2(divider_x, y), rl.Vector2(divider_x, y + 116), 1, DIVIDER)

        ow = int(rl.measure_text_ex(fn, "Overrides", 30, 0).x)
        dw = int(rl.measure_text_ex(fn, "Disengagements", 30, 0).x)
        rl.draw_text_ex(fn, "Overrides",      rl.Vector2(x + (half - ow) // 2,         y), 30, 0, DIM)
        rl.draw_text_ex(fn, "Disengagements", rl.Vector2(divider_x + (half - dw) // 2, y), 30, 0, DIM)

        row_y = y + 38
        self._draw_reason_cells(x, half, overrides, row_y, fn, fb)
        self._draw_reason_cells(divider_x, half, disengages, row_y, fn, fb)

    def _draw_reason_cells(self, section_x: int, section_w: int, items: list, row_y: int, fn, fb):
        col_w = section_w // len(items)
        for i, (label, val) in enumerate(items):
            col_center = section_x + i * col_w + col_w // 2
            lw = int(rl.measure_text_ex(fn, label, 34, 0).x)
            vw = int(rl.measure_text_ex(fb, val,   50, 0).x)
            rl.draw_text_ex(fn, label, rl.Vector2(col_center - lw // 2, row_y),      34, 0, DIM)
            rl.draw_text_ex(fb, val,   rl.Vector2(col_center - vw // 2, row_y + 42), 50, 0, rl.WHITE)

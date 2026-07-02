import json
import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.selfdrive.ui.widgets.stats_common import StatsPageWidget, DIM, DIVIDER

_GREEN    = rl.Color(70, 200, 100, 255)
_ORANGE   = rl.Color(234, 160, 50, 255)
_AOL      = rl.Color(180, 130, 255, 255)
_TURN     = rl.Color(234, 160, 50, 255)
_STRAIGHT = rl.Color(70, 200, 180, 255)
_CURVE    = rl.Color(100, 170, 255, 255)
_LANECHG  = rl.Color(180, 130, 255, 255)


class OverrideStatsWidget(StatsPageWidget):
    """Everything about the most recent drive: Engaged / AOL / Disengaged split, a
    breakdown of override time by driving context (turn / straight line / curve / lane
    change), and why (accel/steer/AOL override vs. braking/cancel disengagement). See the
    Turn / Curve / Straight Line pages (next in the cycle) for detail on each context. The
    context breakdown is inferred from measurable conditions at override time, not the
    driver's actual intent - it's a "under what conditions" view, not a literal "why"."""

    def __init__(self):
        super().__init__()
        self._last_drive: dict | None = None

    def _render(self, rect: rl.Rectangle):
        self._maybe_reload()
        x, y, w = self._draw_frame(rect)

        y = self._draw_section("LAST DRIVE", x, y, w)
        y = self._draw_stat_triple(
            x, y, w,
            "Engaged",    self._fmt_pct("engaged_pct"),    self._fmt_mi("engaged_mi"),    _GREEN,
            "AOL",        self._fmt_pct("aol_pct"),        self._fmt_mi("aol_mi"),        _AOL,
            "Disengaged", self._fmt_pct("disengaged_pct"), self._fmt_mi("disengaged_mi"), _ORANGE,
        )
        y += 25

        y = self._draw_section("OVERRIDE CONTEXT", x, y, w)
        y = self._draw_context_breakdown(x, y, w)
        y += 25

        y = self._draw_section("OVERRIDES & DISENGAGEMENTS", x, y, w)
        self._draw_reasons(x, y, w)

    def _reload(self):
        raw = self._params.get("SpysyLastDriveStats")
        try:
            self._last_drive = json.loads(raw) if raw else None
        except Exception:
            self._last_drive = None

    def _fmt_mi(self, field: str) -> str:
        if self._last_drive is None or field not in self._last_drive:
            return "N/A"
        return f"{self._last_drive.get(field, 0.0):,.1f} mi"

    def _fmt_pct(self, field: str) -> str:
        if self._last_drive is None or field not in self._last_drive:
            return "N/A"
        return f"{self._last_drive.get(field, 0.0):.1f}%"

    def _draw_context_breakdown(self, x: int, y: int, w: int) -> int:
        fn = gui_app.font(FontWeight.NORMAL)
        fb = gui_app.font(FontWeight.BOLD)

        if self._last_drive is None or "override_pct" not in self._last_drive:
            rl.draw_text_ex(fn, "No drive data yet", rl.Vector2(x, y + 12), 40, 0, DIM)
            return y + 60
        if self._last_drive.get("override_mi", 0.0) == 0:
            rl.draw_text_ex(fn, "No overrides on the last drive", rl.Vector2(x, y + 12), 40, 0, DIM)
            return y + 60

        cells = [
            ("Turns",         self._fmt_pct("turn_pct"),        _TURN),
            ("Straight Line", self._fmt_pct("straight_pct"),    _STRAIGHT),
            ("Curve",         self._fmt_pct("curve_pct"),       _CURVE),
            ("Lane Change",   self._fmt_pct("lane_change_pct"), _LANECHG),
        ]
        quarter = w // 4
        for i, (label, val, color) in enumerate(cells):
            col_x = x + i * quarter
            rl.draw_text_ex(fn, label, rl.Vector2(col_x, y),      32, 0, DIM)
            rl.draw_text_ex(fb, val,   rl.Vector2(col_x, y + 44), 48, 0, color)

        note = "Buckets reflect driving context at override time, not confirmed intent"
        rl.draw_text_ex(fn, note, rl.Vector2(x, y + 116), 26, 0, DIM)
        return y + 150

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

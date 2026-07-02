import json
import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.selfdrive.ui.widgets.stats_common import StatsPageWidget, DIM

_GREEN    = rl.Color(70, 200, 100, 255)
_ORANGE   = rl.Color(234, 160, 50, 255)
_AOL      = rl.Color(180, 130, 255, 255)
_TURN     = rl.Color(234, 160, 50, 255)
_STRAIGHT = rl.Color(70, 200, 180, 255)
_CURVE    = rl.Color(100, 170, 255, 255)
_LANECHG  = rl.Color(180, 130, 255, 255)


class DriveStatsWidget(StatsPageWidget):
    """Lifetime driving summary: Engaged / AOL / Disengaged split, and - of all the times
    the driver has intervened - how much of that was in turns, on straight roads, in
    curves, or lane changes. See the Last Drive page (next in the cycle) for this same
    breakdown scoped to just the most recent drive."""

    def __init__(self):
        super().__init__()
        self._lifetime: dict | None = None

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

        y = self._draw_section("LIFETIME OVERRIDE CONTEXT", x, y, w)
        self._draw_override_context(x, y, w)

    def _reload(self):
        raw = self._params.get("SpysyLifetimeStats")
        try:
            self._lifetime = json.loads(raw) if raw else None
        except Exception:
            self._lifetime = None

    def _fmt_lifetime_mi(self, field: str) -> str:
        if self._lifetime is None or field not in self._lifetime:
            return "N/A"
        return f"{self._lifetime.get(field, 0.0):,.1f} mi"

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

    def _override_context_pct(self, field: str) -> str:
        if self._lifetime is None:
            return "N/A"
        override_mi = self._lifetime.get("override_mi", 0.0)
        if override_mi == 0:
            return "N/A"
        return f"{self._lifetime.get(field, 0.0) / override_mi * 100:.1f}%"

    def _draw_override_context(self, x: int, y: int, w: int):
        fn = gui_app.font(FontWeight.NORMAL)
        fb = gui_app.font(FontWeight.BOLD)

        if self._lifetime is None or self._lifetime.get("override_mi", 0.0) == 0:
            rl.draw_text_ex(fn, "No overrides recorded yet", rl.Vector2(x, y + 12), 40, 0, DIM)
            return

        cells = [
            ("Turns",         self._override_context_pct("turn_mi"),        _TURN),
            ("Straight Line", self._override_context_pct("straight_mi"),    _STRAIGHT),
            ("Curve",         self._override_context_pct("curve_mi"),       _CURVE),
            ("Lane Change",   self._override_context_pct("lane_change_mi"), _LANECHG),
        ]
        quarter = w // 4
        for i, (label, val, color) in enumerate(cells):
            col_x = x + i * quarter
            rl.draw_text_ex(fn, label, rl.Vector2(col_x, y),      32, 0, DIM)
            rl.draw_text_ex(fb, val,   rl.Vector2(col_x, y + 44), 48, 0, color)

        note = f"{self._fmt_lifetime_mi('override_mi')} of lifetime driving has been overridden"
        rl.draw_text_ex(fn, note, rl.Vector2(x, y + 116), 28, 0, DIM)

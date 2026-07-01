import json
import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.selfdrive.ui.widgets.stats_common import StatsPageWidget, DIM

_WHITE   = rl.Color(255, 255, 255, 255)
_TURN    = rl.Color(234, 160, 50, 255)
_LANEPOS = rl.Color(70, 200, 180, 255)
_LANECHG = rl.Color(180, 130, 255, 255)


class OverrideStatsWidget(StatsPageWidget):
    """Shows how much of your driving is spent overriding the model's steering, and
    breaks that override time down by driving context (turn / lane position / lane
    change). The breakdown is inferred from measurable conditions at override time,
    not the driver's actual intent — it's a "under what conditions" view, not a
    literal "why"."""

    def __init__(self):
        super().__init__()
        self._lifetime: dict | None = None
        self._last_drive: dict | None = None

    def _render(self, rect: rl.Rectangle):
        self._maybe_reload()
        x, y, w = self._draw_frame(rect)

        y = self._draw_section("STEERING OVERRIDES", x, y, w)
        y = self._draw_stat_pair(
            x, y, w,
            "Override Time (Lifetime)", self._lifetime_override_pct(), _WHITE,
            "Avg. Angle Disagreement",  self._lifetime_avg_divergence(), _WHITE,
        )
        y += 25

        y = self._draw_section("LAST DRIVE — OVERRIDE CONTEXT", x, y, w)
        self._draw_context_breakdown(x, y, w)

    def _reload(self):
        for attr, key in (("_lifetime", "SpysyLifetimeStats"), ("_last_drive", "SpysyLastDriveStats")):
            raw = self._params.get(key)
            try:
                setattr(self, attr, json.loads(raw) if raw else None)
            except Exception:
                setattr(self, attr, None)

    def _lifetime_override_pct(self) -> str:
        if self._lifetime is None:
            return "—"
        controlled = self._lifetime.get("engaged_mi", 0.0) + self._lifetime.get("aol_mi", 0.0)
        if controlled == 0:
            return "—"
        return f"{self._lifetime.get('override_mi', 0.0) / controlled * 100:.1f}%"

    def _lifetime_avg_divergence(self) -> str:
        if self._lifetime is None or "avg_divergence_deg" not in self._lifetime:
            return "—"
        return f"{self._lifetime.get('avg_divergence_deg', 0.0):.1f}°"

    def _fmt_pct(self, field: str) -> str:
        if self._last_drive is None or field not in self._last_drive:
            return "—"
        return f"{self._last_drive.get(field, 0.0):.1f}%"

    def _fmt_deg(self, field: str) -> str:
        if self._last_drive is None or field not in self._last_drive:
            return "—"
        return f"{self._last_drive.get(field, 0.0):.1f}°"

    def _draw_stat_pair(self, x: int, y: int, w: int,
                        l1: str, v1: str, c1: rl.Color,
                        l2: str, v2: str, c2: rl.Color) -> int:
        fn = gui_app.font(FontWeight.NORMAL)
        fb = gui_app.font(FontWeight.BOLD)
        half = w // 2
        for i, (label, val, color) in enumerate([(l1, v1, c1), (l2, v2, c2)]):
            col_x = x + i * half
            rl.draw_text_ex(fn, label, rl.Vector2(col_x, y),      32, 0, DIM)
            rl.draw_text_ex(fb, val,   rl.Vector2(col_x, y + 44), 56, 0, color)
        return y + 120

    def _draw_context_breakdown(self, x: int, y: int, w: int):
        fn = gui_app.font(FontWeight.NORMAL)
        fb = gui_app.font(FontWeight.BOLD)

        if self._last_drive is None or "override_pct" not in self._last_drive:
            rl.draw_text_ex(fn, "No drive data yet", rl.Vector2(x, y + 12), 40, 0, DIM)
            return
        if self._last_drive.get("override_mi", 0.0) == 0:
            rl.draw_text_ex(fn, "No overrides on the last drive", rl.Vector2(x, y + 12), 40, 0, DIM)
            return

        cells = [
            ("Turns",         self._fmt_pct("turn_pct"),        _TURN),
            ("Lane Position", self._fmt_pct("lane_pos_pct"),    _LANEPOS),
            ("Lane Change",   self._fmt_pct("lane_change_pct"), _LANECHG),
        ]
        third = w // 3
        for i, (label, val, color) in enumerate(cells):
            col_x = x + i * third
            rl.draw_text_ex(fn, label, rl.Vector2(col_x, y),      36, 0, DIM)
            rl.draw_text_ex(fb, val,   rl.Vector2(col_x, y + 46), 52, 0, color)

        note = (f"{self._fmt_pct('override_pct')} of this drive overridden, "
                f"avg. {self._fmt_deg('avg_divergence_deg')} off the model's commanded angle")
        rl.draw_text_ex(fn, note, rl.Vector2(x, y + 118), 28, 0, DIM)
        rl.draw_text_ex(fn, "Buckets reflect driving context at override time, not confirmed intent",
                         rl.Vector2(x, y + 152), 26, 0, DIM)

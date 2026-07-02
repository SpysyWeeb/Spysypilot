import json
import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.selfdrive.ui.widgets.stats_common import StatsPageWidget, DIM

_TURN = rl.Color(234, 160, 50, 255)
_TEAL = rl.Color(70, 200, 180, 255)
_BLUE = rl.Color(100, 170, 255, 255)


class TurnStatsWidget(StatsPageWidget):
    """Detail on low-speed, large-angle turns (the kind you'd make at an intersection).
    Everything here is signed: positive means the driver did more of that thing than the
    model (held more angle, acted sooner); negative means less. Split left vs right since
    a car's real-world turn dynamics aren't symmetric."""

    def __init__(self):
        super().__init__()
        self._last_drive: dict | None = None

    def _render(self, rect: rl.Rectangle):
        self._maybe_reload()
        x, y, w = self._draw_frame(rect)

        if self._last_drive is None or "turn_pct" not in self._last_drive:
            y = self._draw_section("TURN ANALYSIS - LAST DRIVE", x, y, w)
            rl.draw_text_ex(gui_app.font(FontWeight.NORMAL), "No drive data yet", rl.Vector2(x, y + 12), 40, 0, DIM)
            return
        if self._last_drive.get("turn_pct", 0.0) == 0.0:
            y = self._draw_section("TURN ANALYSIS - LAST DRIVE", x, y, w)
            rl.draw_text_ex(gui_app.font(FontWeight.NORMAL), "No turns on the last drive", rl.Vector2(x, y + 12), 40, 0, DIM)
            return

        y = self._draw_section("DRIVER ANGLE vs MODEL", x, y, w)
        y = self._draw_row_group(
            x, y, w,
            ("Extra Angle Held",     lambda side: self._fmt_deg(side),  _TURN),
            ("Time Steering Harder", lambda side: self._fmt_soft(side), _TURN),
        )
        rl.draw_text_ex(gui_app.font(FontWeight.NORMAL), "+ = driver held more angle than the model; - = held less",
                         rl.Vector2(x, y), 26, 0, DIM)
        y += 56

        y = self._draw_section("DRIVER TIMING vs MODEL", x, y, w)
        y = self._draw_row_group(
            x, y, w,
            ("Turn-In Lead", lambda side: self._fmt_lead(f"turn_{side}_turnin_lead_s", f"turn_{side}_turnin_count"), _BLUE),
            ("Unwind Lead",  lambda side: self._fmt_lead(f"turn_{side}_unwind_lead_s", f"turn_{side}_unwind_count"), _TEAL),
        )
        n_turnin = self._last_drive.get("turn_left_turnin_count", 0) + self._last_drive.get("turn_right_turnin_count", 0)
        n_unwind = self._last_drive.get("turn_left_unwind_count", 0) + self._last_drive.get("turn_right_unwind_count", 0)
        fn = gui_app.font(FontWeight.NORMAL)
        rl.draw_text_ex(fn, "+ = driver acted before the model's plan; - = after",
                         rl.Vector2(x, y), 26, 0, DIM)
        rl.draw_text_ex(fn, f"{n_turnin} turn-in / {n_unwind} unwind sample(s) over 90° analyzed",
                         rl.Vector2(x, y + 32), 26, 0, DIM)

    def _reload(self):
        raw = self._params.get("SpysyLastDriveStats")
        try:
            self._last_drive = json.loads(raw) if raw else None
        except Exception:
            self._last_drive = None

    def _fmt_deg(self, side: str) -> str:
        deg = self._last_drive.get(f"turn_{side}_agg_deg")
        if deg is None:
            return "N/A"
        return f"{deg:+.1f}°"

    def _fmt_soft(self, side: str) -> str:
        pct = self._last_drive.get(f"turn_{side}_soft_pct")
        if pct is None:
            return "N/A"
        return f"{pct:.0f}%"

    def _fmt_lead(self, field: str, count_field: str) -> str:
        n = self._last_drive.get(count_field, 0)
        if not n:
            return "N/A"
        return f"{self._last_drive.get(field, 0.0):+.1f}s"

    def _draw_row_group(self, x: int, y: int, w: int, *rows: tuple) -> int:
        """rows: (label, side -> str, color) tuples, drawn as Left/Right columns."""
        fn = gui_app.font(FontWeight.NORMAL)
        fb = gui_app.font(FontWeight.BOLD)

        label_w = int(w * 0.36)
        left_x = x + label_w
        right_x = left_x + (w - label_w) // 2

        rl.draw_text_ex(fn, "Left Turns",  rl.Vector2(left_x, y),  32, 0, DIM)
        rl.draw_text_ex(fn, "Right Turns", rl.Vector2(right_x, y), 32, 0, DIM)

        row_y = y + 42
        for label, fmt, color in rows:
            rl.draw_text_ex(fn, label, rl.Vector2(x, row_y), 32, 0, DIM)
            rl.draw_text_ex(fb, fmt("left"),  rl.Vector2(left_x, row_y),  40, 0, color)
            rl.draw_text_ex(fb, fmt("right"), rl.Vector2(right_x, row_y), 40, 0, color)
            row_y += 62

        return row_y

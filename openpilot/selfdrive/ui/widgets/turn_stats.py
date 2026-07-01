import json
import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.selfdrive.ui.widgets.stats_common import StatsPageWidget, DIM

_TURN    = rl.Color(234, 160, 50, 255)
_TEAL    = rl.Color(70, 200, 180, 255)
_BLUE    = rl.Color(100, 170, 255, 255)


class TurnStatsWidget(StatsPageWidget):
    """Detail on low-speed, large-angle turns (the kind you'd make at an intersection):
    how much more (or less) angle the driver holds than the model commands, and how much
    sooner the driver starts turning into and straightening back out of the turn than the
    model's own plan would."""

    def __init__(self):
        super().__init__()
        self._last_drive: dict | None = None

    def _render(self, rect: rl.Rectangle):
        self._maybe_reload()
        x, y, w = self._draw_frame(rect)

        y = self._draw_section("TURN ANALYSIS - LAST DRIVE", x, y, w)

        if self._last_drive is None or "turn_pct" not in self._last_drive:
            rl.draw_text_ex(gui_app.font(FontWeight.NORMAL), "No drive data yet", rl.Vector2(x, y + 12), 40, 0, DIM)
            return
        if self._last_drive.get("turn_pct", 0.0) == 0.0:
            rl.draw_text_ex(gui_app.font(FontWeight.NORMAL), "No turns on the last drive", rl.Vector2(x, y + 12), 40, 0, DIM)
            return

        self._draw_detail(x, y, w)

    def _reload(self):
        raw = self._params.get("SpysyLastDriveStats")
        try:
            self._last_drive = json.loads(raw) if raw else None
        except Exception:
            self._last_drive = None

    def _fmt_soft(self, side: str) -> str:
        pct = self._last_drive.get(f"turn_{side}_soft_pct")
        deg = self._last_drive.get(f"turn_{side}_agg_deg")
        if pct is None or deg is None:
            return "N/A"
        return f"{pct:.0f}% ({deg:+.1f}°)"

    def _fmt_lead(self, field: str, count_field: str) -> str:
        n = self._last_drive.get(count_field, 0)
        if not n:
            return "N/A"
        return f"{self._last_drive.get(field, 0.0):+.1f}s"

    def _draw_detail(self, x: int, y: int, w: int):
        fn = gui_app.font(FontWeight.NORMAL)
        fb = gui_app.font(FontWeight.BOLD)

        label_w = int(w * 0.36)
        left_x = x + label_w
        right_x = left_x + (w - label_w) // 2

        rl.draw_text_ex(fn, "Left Turns",  rl.Vector2(left_x, y),  32, 0, DIM)
        rl.draw_text_ex(fn, "Right Turns", rl.Vector2(right_x, y), 32, 0, DIM)

        row_y = y + 42
        rl.draw_text_ex(fn, "Model too soft", rl.Vector2(x, row_y), 32, 0, DIM)
        rl.draw_text_ex(fb, self._fmt_soft("left"),  rl.Vector2(left_x, row_y),  40, 0, _TURN)
        rl.draw_text_ex(fb, self._fmt_soft("right"), rl.Vector2(right_x, row_y), 40, 0, _TURN)

        row_y += 62
        rl.draw_text_ex(fn, "Turns in sooner", rl.Vector2(x, row_y), 32, 0, DIM)
        rl.draw_text_ex(fb, self._fmt_lead("turn_left_turnin_lead_s", "turn_left_turnin_count"),
                         rl.Vector2(left_x, row_y), 40, 0, _BLUE)
        rl.draw_text_ex(fb, self._fmt_lead("turn_right_turnin_lead_s", "turn_right_turnin_count"),
                         rl.Vector2(right_x, row_y), 40, 0, _BLUE)

        row_y += 62
        rl.draw_text_ex(fn, "Unwinds sooner", rl.Vector2(x, row_y), 32, 0, DIM)
        rl.draw_text_ex(fb, self._fmt_lead("turn_left_unwind_lead_s", "turn_left_unwind_count"),
                         rl.Vector2(left_x, row_y), 40, 0, _TEAL)
        rl.draw_text_ex(fb, self._fmt_lead("turn_right_unwind_lead_s", "turn_right_unwind_count"),
                         rl.Vector2(right_x, row_y), 40, 0, _TEAL)

        n_turnin = self._last_drive.get("turn_left_turnin_count", 0) + self._last_drive.get("turn_right_turnin_count", 0)
        n_unwind = self._last_drive.get("turn_left_unwind_count", 0) + self._last_drive.get("turn_right_unwind_count", 0)
        row_y += 56
        rl.draw_text_ex(fn, f"{n_turnin} turn-in / {n_unwind} unwind sample(s) over 90° analyzed for timing",
                         rl.Vector2(x, row_y), 26, 0, DIM)
        rl.draw_text_ex(fn, "\"Sooner\" = positive; negative means the driver was slower than the model",
                         rl.Vector2(x, row_y + 34), 26, 0, DIM)

import json
import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.selfdrive.ui.widgets.stats_common import StatsPageWidget, DIM

_CURVE = rl.Color(100, 170, 255, 255)


class CurveStatsWidget(StatsPageWidget):
    """Detail on curves (real road curvature below the low-speed/large-angle Turn
    threshold - highway bends, winding roads): how much the driver tugs on the wheel
    relative to what the model commands, for left and right curves separately."""

    def __init__(self):
        super().__init__()
        self._last_drive: dict | None = None

    def _render(self, rect: rl.Rectangle):
        self._maybe_reload()
        x, y, w = self._draw_frame(rect)

        y = self._draw_section("CURVE ANALYSIS - LAST DRIVE", x, y, w)

        if self._last_drive is None or "curve_pct" not in self._last_drive:
            rl.draw_text_ex(gui_app.font(FontWeight.NORMAL), "No drive data yet", rl.Vector2(x, y + 12), 40, 0, DIM)
            return
        if self._last_drive.get("curve_pct", 0.0) == 0.0:
            rl.draw_text_ex(gui_app.font(FontWeight.NORMAL), "No curve overrides on the last drive", rl.Vector2(x, y + 12), 40, 0, DIM)
            return

        self._draw_detail(x, y, w)

    def _reload(self):
        raw = self._params.get("SpysyLastDriveStats")
        try:
            self._last_drive = json.loads(raw) if raw else None
        except Exception:
            self._last_drive = None

    def _fmt_tug(self, side: str) -> str:
        pct = self._last_drive.get(f"curve_{side}_soft_pct")
        deg = self._last_drive.get(f"curve_{side}_agg_deg")
        if pct is None or deg is None:
            return "N/A"
        return f"{pct:.0f}% ({deg:+.1f}°)"

    def _draw_detail(self, x: int, y: int, w: int):
        fn = gui_app.font(FontWeight.NORMAL)
        fb = gui_app.font(FontWeight.BOLD)

        half = w // 2
        rl.draw_text_ex(fn, "Left Curves",  rl.Vector2(x,            y), 32, 0, DIM)
        rl.draw_text_ex(fn, "Right Curves", rl.Vector2(x + half,     y), 32, 0, DIM)

        row_y = y + 44
        rl.draw_text_ex(fb, self._fmt_tug("left"),  rl.Vector2(x,        row_y), 56, 0, _CURVE)
        rl.draw_text_ex(fb, self._fmt_tug("right"), rl.Vector2(x + half, row_y), 56, 0, _CURVE)

        note_y = row_y + 90
        rl.draw_text_ex(fn, "% of that curve's override time spent tugging harder than the model",
                         rl.Vector2(x, note_y), 26, 0, DIM)
        rl.draw_text_ex(fn, "commanded, with the average angle by which the driver pulled harder (+) or eased off (-)",
                         rl.Vector2(x, note_y + 32), 26, 0, DIM)

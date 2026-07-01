import json
import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.selfdrive.ui.widgets.stats_common import StatsPageWidget, DIM

_STRAIGHT = rl.Color(70, 200, 180, 255)


class StraightStatsWidget(StatsPageWidget):
    """Detail on straight-road overrides: the model has essentially no curvature commanded,
    so any override is purely a lane-position preference. Shows which way the driver tends
    to pull and by how much, split left vs right."""

    def __init__(self):
        super().__init__()
        self._last_drive: dict | None = None

    def _render(self, rect: rl.Rectangle):
        self._maybe_reload()
        x, y, w = self._draw_frame(rect)

        y = self._draw_section("STRAIGHT LINE ANALYSIS - LAST DRIVE", x, y, w)

        if self._last_drive is None or "straight_pct" not in self._last_drive:
            rl.draw_text_ex(gui_app.font(FontWeight.NORMAL), "No drive data yet", rl.Vector2(x, y + 12), 40, 0, DIM)
            return
        if self._last_drive.get("straight_pct", 0.0) == 0.0:
            rl.draw_text_ex(gui_app.font(FontWeight.NORMAL), "No straight-line overrides on the last drive",
                             rl.Vector2(x, y + 12), 40, 0, DIM)
            return

        self._draw_detail(x, y, w)

    def _reload(self):
        raw = self._params.get("SpysyLastDriveStats")
        try:
            self._last_drive = json.loads(raw) if raw else None
        except Exception:
            self._last_drive = None

    def _fmt_pull(self, side: str) -> str:
        deg = self._last_drive.get(f"straight_{side}_pull_deg")
        if deg is None:
            return "N/A"
        return f"{deg:.1f}°"

    def _draw_detail(self, x: int, y: int, w: int):
        fn = gui_app.font(FontWeight.NORMAL)
        fb = gui_app.font(FontWeight.BOLD)

        half = w // 2
        rl.draw_text_ex(fn, "Pulling Left",  rl.Vector2(x,        y), 32, 0, DIM)
        rl.draw_text_ex(fn, "Pulling Right", rl.Vector2(x + half, y), 32, 0, DIM)

        row_y = y + 44
        rl.draw_text_ex(fb, self._fmt_pull("left"),  rl.Vector2(x,        row_y), 56, 0, _STRAIGHT)
        rl.draw_text_ex(fb, self._fmt_pull("right"), rl.Vector2(x + half, row_y), 56, 0, _STRAIGHT)

        note_y = row_y + 90
        rl.draw_text_ex(fn, "Average angle off the model's ~0° commanded angle while overriding on a straight road",
                         rl.Vector2(x, note_y), 26, 0, DIM)

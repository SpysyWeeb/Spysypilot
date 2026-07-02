import json
import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.selfdrive.ui.widgets.stats_common import StatsPageWidget, DIM

_STRAIGHT = rl.Color(70, 200, 180, 255)


class StraightStatsWidget(StatsPageWidget):
    """Straight-road overrides: the model has essentially no curvature commanded, so any
    override there is a pure lane-position preference. Shows, out of all engaged+AOL
    (controlled) time, how often the driver is pulling on the wheel on a straight road and
    by how many degrees off the model's ~0 commanded angle - lifetime and last drive."""

    def __init__(self):
        super().__init__()
        self._lifetime: dict | None = None
        self._last_drive: dict | None = None

    def _render(self, rect: rl.Rectangle):
        self._maybe_reload()
        x, y, w = self._draw_frame(rect)

        y = self._draw_section("STRAIGHT LINE ANALYSIS", x, y, w)
        y = self._draw_row_group(x, y, w, "LIFETIME", lifetime=True)
        y += 20
        self._draw_row_group(x, y, w, "LAST DRIVE", lifetime=False)

    def _reload(self):
        for attr, key in (("_lifetime", "SpysyLifetimeStats"), ("_last_drive", "SpysyLastDriveStats")):
            raw = self._params.get(key)
            try:
                setattr(self, attr, json.loads(raw) if raw else None)
            except Exception:
                setattr(self, attr, None)

    def _pct(self, side: str, lifetime: bool) -> str:
        source = self._lifetime if lifetime else self._last_drive
        if source is None:
            return "N/A"
        controlled = source.get("engaged_mi", 0.0) + source.get("aol_mi", 0.0)
        side_mi = source.get(f"straight_{side}_mi")
        if controlled == 0 or side_mi is None:
            return "N/A"
        return f"{side_mi / controlled * 100:.1f}%"

    def _deg(self, side: str, lifetime: bool) -> str:
        source = self._lifetime if lifetime else self._last_drive
        if source is None:
            return "N/A"
        key = f"avg_pull_{side}_deg" if lifetime else f"straight_{side}_pull_deg"
        deg = source.get(key)
        side_mi = source.get(f"straight_{side}_mi")
        if deg is None or not side_mi:
            return "N/A"
        return f"{deg:.1f}°"

    def _draw_row_group(self, x: int, y: int, w: int, title: str, lifetime: bool) -> int:
        fn = gui_app.font(FontWeight.NORMAL)
        fb = gui_app.font(FontWeight.BOLD)

        rl.draw_text_ex(fn, title, rl.Vector2(x, y), 28, 0, DIM)
        y += 36

        label_w = int(w * 0.5)
        left_x = x + label_w
        right_x = left_x + (w - label_w) // 2

        rl.draw_text_ex(fn, "Left", rl.Vector2(left_x, y), 30, 0, DIM)
        rl.draw_text_ex(fn, "Right", rl.Vector2(right_x, y), 30, 0, DIM)

        row_y = y + 38
        rl.draw_text_ex(fn, "% of controlled time pulling", rl.Vector2(x, row_y), 30, 0, DIM)
        rl.draw_text_ex(fb, self._pct("left", lifetime),  rl.Vector2(left_x, row_y),  38, 0, _STRAIGHT)
        rl.draw_text_ex(fb, self._pct("right", lifetime), rl.Vector2(right_x, row_y), 38, 0, _STRAIGHT)

        row_y += 54
        rl.draw_text_ex(fn, "Avg. degrees off commanded", rl.Vector2(x, row_y), 30, 0, DIM)
        rl.draw_text_ex(fb, self._deg("left", lifetime),  rl.Vector2(left_x, row_y),  38, 0, _STRAIGHT)
        rl.draw_text_ex(fb, self._deg("right", lifetime), rl.Vector2(right_x, row_y), 38, 0, _STRAIGHT)

        return row_y + 60

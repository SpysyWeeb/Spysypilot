"""
Route listing for the Clip Viewer feature.

Wraps drive_statsd's existing route-grouping helper (_list_routes) instead of re-walking
Paths.log_root() ourselves, and adds the two extra bits a route picker UI needs -- a parsed
human-readable date and an approximate duration -- both derivable from the route directory name
alone, without opening any log files.
"""
import re
from dataclasses import dataclass, field

from openpilot.common.hardware.hw import Paths
from openpilot.selfdrive.spysypilot.drive_statsd import _list_routes

# Segments are ~60s each (the last one in a route may be shorter); good enough for a list-view
# duration estimate without parsing any logs.
SEGMENT_APPROX_SECONDS = 60

# On-disk route name: "<16-hex-char dongle_id>_<YYYY-MM-DD>--<HH-MM-SS>"
_ROUTE_NAME_RE = re.compile(r'^[0-9a-f]{16}_(?P<date>\d{4}-\d{2}-\d{2})--(?P<time>\d{2}-\d{2}-\d{2})$')


@dataclass
class RouteSummary:
  name: str  # on-disk route name, e.g. "3b58edf675a3d17f_2023-01-01--10-11-12"
  segments: list[str] = field(default_factory=list)  # sorted segment directory names
  date_str: str = "unknown date"
  approx_duration_s: int = 0

  @property
  def num_segments(self) -> int:
    return len(self.segments)


def _format_date(route_name: str) -> str:
  m = _ROUTE_NAME_RE.match(route_name)
  if not m:
    return "unknown date"
  return f"{m.group('date')} {m.group('time').replace('-', ':')}"


def list_route_summaries(log_root: str | None = None) -> list[RouteSummary]:
  """Return every route stored on-device, newest first."""
  root = log_root if log_root is not None else Paths.log_root()
  routes = _list_routes(root)

  summaries = [
    RouteSummary(
      name=name,
      segments=segs,
      date_str=_format_date(name),
      approx_duration_s=len(segs) * SEGMENT_APPROX_SECONDS,
    )
    for name, segs in routes.items()
  ]

  # Route names are "<dongle_id>_<YYYY-MM-DD--HH-MM-SS>": zero-padded, fixed-width, and (on a
  # single device) sharing the same dongle_id prefix, so a plain lexicographic sort on the name
  # doubles as a chronological sort -- newest first.
  summaries.sort(key=lambda r: r.name, reverse=True)
  return summaries

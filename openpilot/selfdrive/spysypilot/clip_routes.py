"""
Route listing for the Clip Viewer feature.

Wraps drive_statsd's existing route-grouping helper (_list_routes) instead of re-walking
Paths.log_root() ourselves, and adds the two extra bits a route picker UI needs -- a
human-readable date and an approximate duration.
"""
import os
import time
from dataclasses import dataclass, field

from openpilot.common.hardware.hw import Paths
from openpilot.selfdrive.spysypilot.drive_statsd import _list_routes

# Segments are ~60s each (the last one in a route may be shorter); good enough for a list-view
# duration estimate without parsing any logs.
SEGMENT_APPROX_SECONDS = 60


@dataclass
class RouteSummary:
  name: str  # on-disk route name, e.g. "00000001--4fb45ad73b"
  segments: list[str] = field(default_factory=list)  # sorted segment directory names
  date_str: str = "unknown date"
  approx_duration_s: int = 0

  @property
  def num_segments(self) -> int:
    return len(self.segments)


def _format_date(log_root: str, first_segment: str) -> str:
  """Route directory names on-device do NOT reliably embed a usable date -- verified against a
  real device: names look like "00000001--4fb45ad73b--0" (an incrementing counter + short id, no
  timestamp at all). Root cause, also confirmed on-device: the RTC reads ~1970-01-01 on every
  power-cycle (no working battery backup) and is only corrected by NTP sometime after boot;
  openpilot's logger falls back to a counter-based name whenever the clock isn't known-valid yet
  at logging time, which on this device is apparently always. So: use the first segment
  directory's filesystem mtime instead -- it reflects the (by-then NTP-corrected) clock at the
  time that segment was written, independent of whatever naming scheme produced the directory."""
  try:
    mtime = os.path.getmtime(os.path.join(log_root, first_segment))
  except OSError:
    return "unknown date"
  return time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))


def list_route_summaries(log_root: str | None = None) -> list[RouteSummary]:
  """Return every route stored on-device, newest first."""
  root = log_root if log_root is not None else Paths.log_root()
  routes = _list_routes(root)

  summaries = [
    RouteSummary(
      name=name,
      segments=segs,
      date_str=_format_date(root, segs[0]) if segs else "unknown date",
      approx_duration_s=len(segs) * SEGMENT_APPROX_SECONDS,
    )
    for name, segs in routes.items()
  ]

  # Route names are "<counter>--<id>": the leading counter is a zero-padded, monotonically
  # increasing sequence number assigned at logging time (confirmed on-device: "00000000--...",
  # "00000001--...", ...), so it sorts chronologically regardless of the RTC/date situation above
  # -- a plain lexicographic sort on the name is newest-first when reversed.
  summaries.sort(key=lambda r: r.name, reverse=True)
  return summaries

#!/usr/bin/env python3
"""
drive_statsd - Off-road service that parses completed drive logs and
writes per-drive and lifetime engagement stats to Params for the UI.
"""
import json
import os
import time
from typing import Optional

from openpilot.common.params import Params
from openpilot.common.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog

METERS_PER_MILE = 1609.344
POLL_INTERVAL = 30.0

# OnroadEvent.EventName ordinals from cereal/log.capnp
_BUTTON_CANCEL   = 9   # buttonCancel
_PEDAL_PRESSED   = 11  # pedalPressed (brake disengagement)
_GAS_OVERRIDE    = 13  # gasPressedOverride
_STEER_OVERRIDE  = 14  # steerOverride
_STEER_DISENGAGE = 94  # steerDisengage


def _find_rlog(seg_path: str) -> Optional[str]:
    for name in ('rlog.zst', 'rlog.bz2', 'rlog'):
        p = os.path.join(seg_path, name)
        if os.path.exists(p):
            return p
    return None


def _list_routes(log_root: str) -> dict[str, list[str]]:
    """Return {route_name: [seg_dir, ...]} for all routes in log_root."""
    routes: dict[str, list[str]] = {}
    try:
        for entry in os.listdir(log_root):
            if '--' not in entry:
                continue
            if not os.path.isdir(os.path.join(log_root, entry)):
                continue
            route_name = entry.rsplit('--', 1)[0]
            routes.setdefault(route_name, []).append(entry)
    except OSError:
        return {}

    for segs in routes.values():
        segs.sort(key=lambda d: int(d.rsplit('--', 1)[1]) if d.rsplit('--', 1)[1].isdigit() else 0)
    return routes


def _most_recent_route(log_root: str) -> Optional[tuple[str, list[str]]]:
    routes = _list_routes(log_root)
    if not routes:
        return None

    def _mtime(item: tuple[str, list[str]]) -> float:
        _, segs = item
        times = []
        for seg in segs:
            try:
                times.append(os.path.getmtime(os.path.join(log_root, seg)))
            except OSError:
                pass
        return max(times) if times else 0.0

    return max(routes.items(), key=_mtime)


def _parse_segment(seg_path: str) -> dict:
    rlog = _find_rlog(seg_path)
    if rlog is None:
        return {}

    try:
        from openpilot.tools.lib.logreader import LogReader
    except ImportError:
        return {}

    engaged_m = 0.0
    disengaged_m = 0.0
    gas = steer = brake = cancel = 0

    enabled = False
    last_t: Optional[float] = None
    last_vego = 0.0
    prev_event_names: set[int] = set()

    try:
        for msg in LogReader(rlog):
            t = msg.logMonoTime / 1e9
            w = msg.which()

            if w == 'carState':
                vego = msg.carState.vEgo
                if last_t is not None:
                    dt = min(t - last_t, 0.5)
                    m = last_vego * dt
                    if enabled:
                        engaged_m += m
                    else:
                        disengaged_m += m
                last_t = t
                last_vego = vego

            elif w == 'selfdriveState':
                enabled = msg.selfdriveState.enabled

            elif w == 'onroadEvents':
                names: set[int] = {ev.name.raw for ev in msg.onroadEvents}
                new = names - prev_event_names
                if _GAS_OVERRIDE in new:
                    gas += 1
                if _STEER_OVERRIDE in new:
                    steer += 1
                if _PEDAL_PRESSED in new or _STEER_DISENGAGE in new:
                    brake += 1
                if _BUTTON_CANCEL in new:
                    cancel += 1
                prev_event_names = names

    except Exception as e:
        cloudlog.warning(f"drive_statsd: error parsing {rlog}: {e}")

    return {
        'engaged_m': engaged_m,
        'disengaged_m': disengaged_m,
        'events': {'gas': gas, 'steer': steer, 'brake': brake, 'cancel': cancel},
    }


def _parse_route(log_root: str, seg_dirs: list[str], params: Params, route_name: str) -> dict:
    total: dict = {'engaged_m': 0.0, 'disengaged_m': 0.0,
                   'events': {'gas': 0, 'steer': 0, 'brake': 0, 'cancel': 0}}
    n = len(seg_dirs)
    for i, seg_dir in enumerate(seg_dirs, 1):
        params.put("SpysyStatsStatus", f"Analyzing {route_name} · seg {i}/{n}")
        seg = _parse_segment(os.path.join(log_root, seg_dir))
        if not seg:
            continue
        total['engaged_m'] += seg['engaged_m']
        total['disengaged_m'] += seg['disengaged_m']
        for k in total['events']:
            total['events'][k] += seg['events'].get(k, 0)
    return total


def _reason_pcts(events: dict) -> dict:
    total = sum(events.values())
    if total == 0:
        return {k: 0.0 for k in events}
    return {k: round(v / total * 100, 1) for k, v in events.items()}


def main():
    params = Params()
    log_root = Paths.log_root()

    try:
        raw = params.get("SpysyLifetimeStats")
        lifetime = json.loads(raw) if raw else {'engaged_m': 0.0, 'disengaged_m': 0.0}
    except Exception:
        lifetime = {'engaged_m': 0.0, 'disengaged_m': 0.0}

    last_processed = params.get("SpysyLastProcessedRoute") or ""

    while True:
        result = _most_recent_route(log_root)
        if result is None or result[0] == last_processed:
            time.sleep(POLL_INTERVAL)
            continue

        route_name, seg_dirs = result
        cloudlog.info(f"drive_statsd: processing route {route_name} ({len(seg_dirs)} segments)")

        # Allow loggerd a moment to finish flushing the final segment
        time.sleep(5.0)

        # Re-fetch in case more segments appeared during the wait
        result = _most_recent_route(log_root)
        if result is None:
            time.sleep(POLL_INTERVAL)
            continue
        route_name, seg_dirs = result

        if route_name == last_processed:
            time.sleep(POLL_INTERVAL)
            continue

        drive = _parse_route(log_root, seg_dirs, params, route_name)

        lifetime['engaged_m'] += drive['engaged_m']
        lifetime['disengaged_m'] += drive['disengaged_m']

        total_m = drive['engaged_m'] + drive['disengaged_m']
        eng_pct = round(drive['engaged_m'] / total_m * 100, 1) if total_m > 0 else 0.0

        last_drive = {
            'engaged_mi': round(drive['engaged_m'] / METERS_PER_MILE, 2),
            'disengaged_mi': round(drive['disengaged_m'] / METERS_PER_MILE, 2),
            'engaged_pct': eng_pct,
            'disengaged_pct': round(100.0 - eng_pct, 1),
            'reasons': _reason_pcts(drive['events']),
        }

        params.put("SpysyStatsStatus", "")
        # Write route name first to prevent double-counting if killed mid-write
        params.put("SpysyLastProcessedRoute", route_name)
        params.put("SpysyLastDriveStats", json.dumps(last_drive))
        params.put("SpysyLifetimeStats", json.dumps({
            'engaged_m': lifetime['engaged_m'],
            'disengaged_m': lifetime['disengaged_m'],
        }))
        last_processed = route_name
        cloudlog.info(f"drive_statsd: done — {eng_pct:.1f}% engaged last drive")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()

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

# All fields the current analyzer produces in SpysyLifetimeStats.
# If stored data is missing any of these, all routes are reanalyzed.
REQUIRED_LIFETIME_FIELDS = {"engaged_mi", "disengaged_mi", "aol_mi"}


def _find_rlog(seg_path: str) -> Optional[str]:
    for name in ('rlog.zst', 'rlog.bz2', 'rlog'):
        p = os.path.join(seg_path, name)
        if os.path.exists(p):
            return p
    return None


def _list_routes(log_root: str) -> dict[str, list[str]]:
    """Return {route_name: [seg_dir, ...]} sorted within each route."""
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
    aol_m = 0.0
    gas = steer = brake = cancel = aol = 0

    enabled = False
    aol_active = False
    aol_middle_prev = False
    last_t: Optional[float] = None
    last_vego = 0.0
    prev_event_names: set[int] = set()

    def _mark_aol_edge():
        # AOL's "middle ground": steering is active (aol_active) but the car isn't
        # fully engaged (enabled). Counts one event per rising edge into that state.
        nonlocal aol, aol_middle_prev
        aol_middle = aol_active and not enabled
        if aol_middle and not aol_middle_prev:
            aol += 1
        aol_middle_prev = aol_middle

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
                    elif aol_active:
                        aol_m += m
                    else:
                        disengaged_m += m
                last_t = t
                last_vego = vego

            elif w == 'selfdriveState':
                enabled = msg.selfdriveState.enabled
                _mark_aol_edge()

            elif w == 'spysydriveStateSP':
                aol_active = bool(msg.spysydriveStateSP.aol.active)
                _mark_aol_edge()

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
        'aol_m': aol_m,
        'events': {'gas': gas, 'steer': steer, 'brake': brake, 'cancel': cancel, 'aol': aol},
    }


def _parse_route(log_root: str, seg_dirs: list[str], params: Params,
                 route_name: str, route_idx: int, route_total: int) -> dict:
    total: dict = {'engaged_m': 0.0, 'disengaged_m': 0.0, 'aol_m': 0.0,
                   'events': {'gas': 0, 'steer': 0, 'brake': 0, 'cancel': 0, 'aol': 0}}
    n = len(seg_dirs)
    for i, seg_dir in enumerate(seg_dirs, 1):
        params.put("SpysyStatsStatus",
                   f"Analyzing route {route_idx}/{route_total} - seg {i}/{n}")
        seg = _parse_segment(os.path.join(log_root, seg_dir))
        if not seg:
            continue
        total['engaged_m'] += seg['engaged_m']
        total['disengaged_m'] += seg['disengaged_m']
        total['aol_m'] += seg.get('aol_m', 0.0)
        for k in total['events']:
            total['events'][k] += seg['events'].get(k, 0)
    return total


def _reason_pcts(events: dict) -> dict:
    total = sum(events.values())
    if total == 0:
        return {k: 0.0 for k in events}
    return {k: round(v / total * 100, 1) for k, v in events.items()}


def _startup_verify(log_root: str, processed: set[str], lifetime: dict,
                    params: Params) -> tuple[dict, set[str]]:
    """
    Schema check then disk presence check.

    If stored lifetime data is missing any field from REQUIRED_LIFETIME_FIELDS,
    all routes are cleared for full reanalysis — this automatically triggers
    whenever a new data field is added to the analyzer.

    Returns (lifetime, processed), either of which may have been reset.
    """
    if processed:
        raw = params.get("SpysyLifetimeStats")
        needs_reanalysis = False
        if not raw:
            # Processed set is non-empty but lifetime data is gone — inconsistent state.
            needs_reanalysis = True
        else:
            try:
                stored = json.loads(raw)
                missing = REQUIRED_LIFETIME_FIELDS - set(stored.keys())
                if missing:
                    cloudlog.info(f"drive_statsd: lifetime data missing fields {missing}, clearing for full reanalysis")
                    needs_reanalysis = True
            except Exception:
                needs_reanalysis = True

        if needs_reanalysis:
            params.put("SpysyStatsStatus", "New data fields detected — reanalyzing all routes...")
            processed = set()
            lifetime = {'engaged_m': 0.0, 'disengaged_m': 0.0, 'aol_m': 0.0}
            _save_lifetime(params, lifetime)
            _save_processed(params, processed)
            return lifetime, processed

    if not processed:
        return lifetime, processed

    # Disk presence check — confirm each tracked route still exists
    all_routes_disk = _list_routes(log_root)
    route_list = sorted(processed)
    total = len(route_list)
    cloudlog.info(f"drive_statsd: verifying {total} tracked route(s) at startup")
    for idx, route_name in enumerate(route_list, 1):
        params.put("SpysyStatsStatus", f"Verifying route {idx}/{total}...")
        time.sleep(0.3)
        if route_name not in all_routes_disk:
            cloudlog.warning(f"drive_statsd: tracked route missing from disk: {route_name}")
    params.put("SpysyStatsStatus", f"Verified {total} route(s)")
    cloudlog.info("drive_statsd: startup verification complete")
    return lifetime, processed


def _load_lifetime(params: Params) -> dict:
    try:
        raw = params.get("SpysyLifetimeStats")
        if raw:
            stored = json.loads(raw)
            return {
                'engaged_m': stored.get('engaged_mi', 0.0) * METERS_PER_MILE,
                'disengaged_m': stored.get('disengaged_mi', 0.0) * METERS_PER_MILE,
                'aol_m': stored.get('aol_mi', 0.0) * METERS_PER_MILE,
            }
    except Exception:
        pass
    return {'engaged_m': 0.0, 'disengaged_m': 0.0, 'aol_m': 0.0}


def _save_lifetime(params: Params, lifetime: dict):
    params.put("SpysyLifetimeStats", json.dumps({
        'engaged_mi': round(lifetime['engaged_m'] / METERS_PER_MILE, 2),
        'disengaged_mi': round(lifetime['disengaged_m'] / METERS_PER_MILE, 2),
        'aol_mi': round(lifetime['aol_m'] / METERS_PER_MILE, 2),
    }))


def _load_processed(params: Params) -> set[str]:
    try:
        raw = params.get("SpysyProcessedRoutes")
        return set(json.loads(raw)) if raw else set()
    except Exception:
        return set()


def _save_processed(params: Params, processed: set[str]):
    params.put("SpysyProcessedRoutes", json.dumps(sorted(processed)))


def main():
    params = Params()
    log_root = Paths.log_root()

    lifetime = _load_lifetime(params)
    processed = _load_processed(params)

    lifetime, processed = _startup_verify(log_root, processed, lifetime, params)

    while True:
        # Force refresh: wipe accumulated stats and reprocess everything
        if params.get_bool("SpysyForceStatsRefresh"):
            params.put_bool("SpysyForceStatsRefresh", False)
            lifetime = {'engaged_m': 0.0, 'disengaged_m': 0.0, 'aol_m': 0.0}
            processed = set()
            _save_lifetime(params, lifetime)
            _save_processed(params, processed)
            cloudlog.info("drive_statsd: force refresh - reprocessing all routes")

        all_routes = _list_routes(log_root)

        # Prune stale entries for routes the deleter has already removed
        processed &= set(all_routes.keys())

        unprocessed = sorted(name for name in all_routes if name not in processed)

        if not unprocessed:
            params.put("SpysyStatsStatus", "Nothing to analyze")
            time.sleep(POLL_INTERVAL)
            continue

        cloudlog.info(f"drive_statsd: {len(unprocessed)} unprocessed route(s)")

        last_drive: Optional[dict] = None
        total_routes = len(unprocessed)

        for idx, route_name in enumerate(unprocessed, 1):
            seg_dirs = all_routes[route_name]
            cloudlog.info(f"drive_statsd: processing {route_name} ({len(seg_dirs)} seg(s))")

            drive = _parse_route(log_root, seg_dirs, params, route_name, idx, total_routes)

            lifetime['engaged_m'] += drive['engaged_m']
            lifetime['disengaged_m'] += drive['disengaged_m']
            lifetime['aol_m'] += drive['aol_m']

            # Engaged / AOL / Disengaged are mutually exclusive and always sum to 100%;
            # disengaged_pct absorbs the rounding remainder so the three stay exact.
            total_m = drive['engaged_m'] + drive['aol_m'] + drive['disengaged_m']
            eng_pct = round(drive['engaged_m'] / total_m * 100, 1) if total_m > 0 else 0.0
            aol_pct = round(drive['aol_m'] / total_m * 100, 1) if total_m > 0 else 0.0

            last_drive = {
                'engaged_mi': round(drive['engaged_m'] / METERS_PER_MILE, 2),
                'disengaged_mi': round(drive['disengaged_m'] / METERS_PER_MILE, 2),
                'aol_mi': round(drive['aol_m'] / METERS_PER_MILE, 2),
                'engaged_pct': eng_pct,
                'disengaged_pct': round(100.0 - eng_pct - aol_pct, 1),
                'aol_pct': aol_pct,
                'reasons': _reason_pcts(drive['events']),
            }

            processed.add(route_name)
            # Persist after every route so a crash mid-run does not lose progress
            _save_processed(params, processed)
            _save_lifetime(params, lifetime)

        if last_drive:
            params.put("SpysyLastDriveStats", json.dumps(last_drive))

        params.put("SpysyStatsStatus", f"Done analyzing - {unprocessed[-1]}")
        cloudlog.info(f"drive_statsd: done - processed {total_routes} route(s)")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()

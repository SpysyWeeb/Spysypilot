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
from openpilot.common.constants import CV
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

# A "turn" (as opposed to a highway curve or lane-position nudge) is a low-speed,
# large-angle maneuver - the kind you'd make at an intersection. Positive
# steeringAngleDeg is left (ISO convention), so sign(angle) gives direction throughout.
TURN_SPEED_MAX_MS = 15.0 * CV.MPH_TO_MS
TURN_PEAK_MIN_DEG = 90.0

# Below this |commanded angle|, an override is "straight line" (lane position preference);
# at/above it (but not a Turn), it's a "curve" (the model is actually steering for road
# geometry, e.g. a highway bend).
STRAIGHT_LINE_MAX_DEG = 10.0

# Turn-episode detection (for turn-in/unwind timing) brackets a turn between crossing this
# onset angle on the way in and back out; UNWIND_FRACTION of the episode's peak angle is the
# "starting to straighten out" proxy compared between commanded and actual.
TURN_ONSET_DEG = 20.0
UNWIND_FRACTION = 0.5

# All fields the current analyzer produces in SpysyLifetimeStats.
# If stored data is missing any of these, all routes are reanalyzed.
REQUIRED_LIFETIME_FIELDS = {
    "engaged_mi", "disengaged_mi", "aol_mi", "override_mi",
    "turn_mi", "straight_mi", "curve_mi", "lane_change_mi", "avg_divergence_deg",
    "turn_left_mi", "turn_right_mi",
    "avg_agg_left_deg", "avg_agg_right_deg",
    "soft_pct_left", "soft_pct_right",
    "avg_unwind_lead_left_s", "avg_unwind_lead_right_s",
    "unwind_count_left", "unwind_count_right",
    "avg_turnin_lead_left_s", "avg_turnin_lead_right_s",
    "turnin_count_left", "turnin_count_right",
    "straight_left_mi", "straight_right_mi",
    "avg_pull_left_deg", "avg_pull_right_deg",
    "curve_left_mi", "curve_right_mi",
    "avg_curve_agg_left_deg", "avg_curve_agg_right_deg",
    "curve_soft_pct_left", "curve_soft_pct_right",
}

_LIFETIME_ZERO = {
    'engaged_m': 0.0, 'disengaged_m': 0.0, 'aol_m': 0.0, 'override_m': 0.0,
    'turn_m': 0.0, 'straight_m': 0.0, 'curve_m': 0.0, 'lane_change_m': 0.0, 'avg_divergence_deg': 0.0,
    'turn_left_m': 0.0, 'turn_right_m': 0.0,
    'avg_agg_left_deg': 0.0, 'avg_agg_right_deg': 0.0,
    'soft_pct_left': 0.0, 'soft_pct_right': 0.0,
    'avg_unwind_lead_left_s': 0.0, 'avg_unwind_lead_right_s': 0.0,
    'unwind_count_left': 0, 'unwind_count_right': 0,
    'avg_turnin_lead_left_s': 0.0, 'avg_turnin_lead_right_s': 0.0,
    'turnin_count_left': 0, 'turnin_count_right': 0,
    'straight_left_m': 0.0, 'straight_right_m': 0.0,
    'avg_pull_left_deg': 0.0, 'avg_pull_right_deg': 0.0,
    'curve_left_m': 0.0, 'curve_right_m': 0.0,
    'avg_curve_agg_left_deg': 0.0, 'avg_curve_agg_right_deg': 0.0,
    'curve_soft_pct_left': 0.0, 'curve_soft_pct_right': 0.0,
}


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
    override_m = 0.0
    turn_m = 0.0
    straight_m = 0.0
    curve_m = 0.0
    lane_change_m = 0.0
    divergence_wsum = 0.0  # degrees * meters, distance-weighted sum for averaging later
    gas = steer = brake = cancel = aol = 0

    # Turn detail, split by direction (left/right). *_m and *_agg_wsum are the override-time
    # distance and signed-divergence-weighted-sum used to compute an "is the model too soft or
    # too aggressive" average; *_soft_m is the subset of that where the driver pushed harder
    # than commanded (model undershooting). Same pattern reused for curves.
    turn_side_m = {'left': 0.0, 'right': 0.0}
    turn_side_agg_wsum = {'left': 0.0, 'right': 0.0}
    turn_side_soft_m = {'left': 0.0, 'right': 0.0}

    curve_side_m = {'left': 0.0, 'right': 0.0}
    curve_side_agg_wsum = {'left': 0.0, 'right': 0.0}
    curve_side_soft_m = {'left': 0.0, 'right': 0.0}

    # Straight-line: side is which way the driver is pulling (sign of actual angle, since
    # commanded is ~0 here and its sign isn't meaningful). wsum is pull magnitude.
    straight_side_m = {'left': 0.0, 'right': 0.0}
    straight_side_wsum = {'left': 0.0, 'right': 0.0}

    # Turn-in / unwind timing: how much sooner the driver starts turning into / straightening
    # out of a turn than the model's own plan would, for turns whose peak exceeds
    # TURN_PEAK_MIN_DEG. Tracked independently of steeringPressed - even a turn the driver
    # never touches is a valid (near-zero-lead) data point.
    turnin_lead_sum = {'left': 0.0, 'right': 0.0}
    turnin_count = {'left': 0, 'right': 0}
    unwind_lead_sum = {'left': 0.0, 'right': 0.0}
    unwind_count = {'left': 0, 'right': 0}

    enabled = False
    aol_active = False
    aol_middle_prev = False
    commanded_angle = 0.0
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

    # Turn-episode state.
    in_turn = False
    turn_side = 0  # +1 left, -1 right
    turn_peak_cmd = 0.0
    turn_peak_cmd_t = 0.0
    turn_peak_act = 0.0
    turn_peak_act_t = 0.0
    cmd_onset_t: Optional[float] = None
    act_onset_t: Optional[float] = None
    cmd_unwind_t: Optional[float] = None
    act_unwind_t: Optional[float] = None

    def _turn_episode_tick(t: float, ang: float, act: float):
        nonlocal in_turn, turn_side, turn_peak_cmd, turn_peak_cmd_t, turn_peak_act, turn_peak_act_t, \
            cmd_onset_t, act_onset_t, cmd_unwind_t, act_unwind_t

        if not in_turn:
            # Episode starts on whichever signal (commanded or actual) crosses onset first,
            # so a driver who leads the model into a turn doesn't get missed.
            triggered = ang if abs(ang) >= TURN_ONSET_DEG else act if abs(act) >= TURN_ONSET_DEG else None
            if triggered is not None:
                in_turn = True
                turn_side = 1 if triggered > 0 else -1
                turn_peak_cmd, turn_peak_cmd_t = ang, t
                turn_peak_act, turn_peak_act_t = act, t
                cmd_onset_t = t if turn_side * ang >= TURN_ONSET_DEG else None
                act_onset_t = t if turn_side * act >= TURN_ONSET_DEG else None
                cmd_unwind_t = act_unwind_t = None
            return

        if cmd_onset_t is None and turn_side * ang >= TURN_ONSET_DEG:
            cmd_onset_t = t
        if act_onset_t is None and turn_side * act >= TURN_ONSET_DEG:
            act_onset_t = t

        if turn_side * ang > turn_side * turn_peak_cmd:
            turn_peak_cmd, turn_peak_cmd_t = ang, t
        if turn_side * act > turn_side * turn_peak_act:
            turn_peak_act, turn_peak_act_t = act, t

        # Only evaluate an unwind crossing once the tracked peak is a real maneuver peak
        # (>= onset), not the trivial near-zero value right at episode entry - otherwise
        # "0 <= 0 * UNWIND_FRACTION" trivially fires before the angle has even started rising.
        if (cmd_unwind_t is None and abs(turn_peak_cmd) >= TURN_ONSET_DEG and t > turn_peak_cmd_t
                and turn_side * ang <= turn_side * turn_peak_cmd * UNWIND_FRACTION):
            cmd_unwind_t = t
        if (act_unwind_t is None and abs(turn_peak_act) >= TURN_ONSET_DEG and t > turn_peak_act_t
                and turn_side * act <= turn_side * turn_peak_act * UNWIND_FRACTION):
            act_unwind_t = t

        # Only exit once BOTH signals are past their own peak - otherwise a driver who leads
        # the model into the turn causes a premature exit while commanded is still ramping up
        # (commanded is briefly still below onset, wiping the onset timestamps just recorded).
        if abs(ang) < TURN_ONSET_DEG and t > turn_peak_cmd_t and t > turn_peak_act_t:
            if abs(turn_peak_cmd) >= TURN_PEAK_MIN_DEG:
                side_key = 'left' if turn_side > 0 else 'right'
                if cmd_onset_t is not None and act_onset_t is not None:
                    turnin_lead_sum[side_key] += cmd_onset_t - act_onset_t
                    turnin_count[side_key] += 1
                if cmd_unwind_t is not None and act_unwind_t is not None:
                    unwind_lead_sum[side_key] += cmd_unwind_t - act_unwind_t
                    unwind_count[side_key] += 1
            in_turn = False

    try:
        for msg in LogReader(rlog):
            t = msg.logMonoTime / 1e9
            w = msg.which()

            if w == 'carState':
                cs = msg.carState
                vego = cs.vEgo
                if last_t is not None:
                    dt = min(t - last_t, 0.5)
                    m = last_vego * dt
                    if enabled:
                        engaged_m += m
                    elif aol_active:
                        aol_m += m
                    else:
                        disengaged_m += m

                    if enabled or aol_active:
                        _turn_episode_tick(t, commanded_angle, cs.steeringAngleDeg)

                    # Override analysis: only meaningful while the model is actually in
                    # control (engaged or AOL) and the driver is fighting the wheel.
                    if (enabled or aol_active) and cs.steeringPressed:
                        override_m += m
                        divergence_wsum += abs(commanded_angle - cs.steeringAngleDeg) * m
                        is_turn = vego < TURN_SPEED_MAX_MS and abs(commanded_angle) > TURN_PEAK_MIN_DEG

                        if cs.leftBlinker or cs.rightBlinker:
                            lane_change_m += m
                        elif is_turn:
                            turn_m += m
                            side_key = 'left' if commanded_angle > 0 else 'right'
                            aggression = (1 if commanded_angle > 0 else -1) * (cs.steeringAngleDeg - commanded_angle)
                            turn_side_m[side_key] += m
                            turn_side_agg_wsum[side_key] += aggression * m
                            if aggression > 0:
                                turn_side_soft_m[side_key] += m
                        elif abs(commanded_angle) < STRAIGHT_LINE_MAX_DEG:
                            straight_m += m
                            side_key = 'left' if cs.steeringAngleDeg > 0 else 'right'
                            pull = abs(cs.steeringAngleDeg - commanded_angle)
                            straight_side_m[side_key] += m
                            straight_side_wsum[side_key] += pull * m
                        else:
                            curve_m += m
                            side_key = 'left' if commanded_angle > 0 else 'right'
                            aggression = (1 if commanded_angle > 0 else -1) * (cs.steeringAngleDeg - commanded_angle)
                            curve_side_m[side_key] += m
                            curve_side_agg_wsum[side_key] += aggression * m
                            if aggression > 0:
                                curve_side_soft_m[side_key] += m
                last_t = t
                last_vego = vego

            elif w == 'carControl':
                # Angle-based commanded steering (this fork targets an angle-control car);
                # compared directly against carState.steeringAngleDeg, the actual angle.
                commanded_angle = msg.carControl.actuators.steeringAngleDeg

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
        'override_m': override_m,
        'turn_m': turn_m,
        'straight_m': straight_m,
        'curve_m': curve_m,
        'lane_change_m': lane_change_m,
        'divergence_wsum': divergence_wsum,
        'turn_side_m': turn_side_m,
        'turn_side_agg_wsum': turn_side_agg_wsum,
        'turn_side_soft_m': turn_side_soft_m,
        'curve_side_m': curve_side_m,
        'curve_side_agg_wsum': curve_side_agg_wsum,
        'curve_side_soft_m': curve_side_soft_m,
        'straight_side_m': straight_side_m,
        'straight_side_wsum': straight_side_wsum,
        'turnin_lead_sum': turnin_lead_sum,
        'turnin_count': turnin_count,
        'unwind_lead_sum': unwind_lead_sum,
        'unwind_count': unwind_count,
        'events': {'gas': gas, 'steer': steer, 'brake': brake, 'cancel': cancel, 'aol': aol},
    }


def _parse_route(log_root: str, seg_dirs: list[str], params: Params,
                 route_name: str, route_idx: int, route_total: int) -> dict:
    total: dict = {
        'engaged_m': 0.0, 'disengaged_m': 0.0, 'aol_m': 0.0,
        'override_m': 0.0, 'turn_m': 0.0, 'straight_m': 0.0, 'curve_m': 0.0, 'lane_change_m': 0.0,
        'divergence_wsum': 0.0,
        'turn_side_m': {'left': 0.0, 'right': 0.0},
        'turn_side_agg_wsum': {'left': 0.0, 'right': 0.0},
        'turn_side_soft_m': {'left': 0.0, 'right': 0.0},
        'curve_side_m': {'left': 0.0, 'right': 0.0},
        'curve_side_agg_wsum': {'left': 0.0, 'right': 0.0},
        'curve_side_soft_m': {'left': 0.0, 'right': 0.0},
        'straight_side_m': {'left': 0.0, 'right': 0.0},
        'straight_side_wsum': {'left': 0.0, 'right': 0.0},
        'turnin_lead_sum': {'left': 0.0, 'right': 0.0},
        'turnin_count': {'left': 0, 'right': 0},
        'unwind_lead_sum': {'left': 0.0, 'right': 0.0},
        'unwind_count': {'left': 0, 'right': 0},
        'events': {'gas': 0, 'steer': 0, 'brake': 0, 'cancel': 0, 'aol': 0},
    }
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
        total['override_m'] += seg.get('override_m', 0.0)
        total['turn_m'] += seg.get('turn_m', 0.0)
        total['straight_m'] += seg.get('straight_m', 0.0)
        total['curve_m'] += seg.get('curve_m', 0.0)
        total['lane_change_m'] += seg.get('lane_change_m', 0.0)
        total['divergence_wsum'] += seg.get('divergence_wsum', 0.0)
        for side in ('left', 'right'):
            total['turn_side_m'][side] += seg.get('turn_side_m', {}).get(side, 0.0)
            total['turn_side_agg_wsum'][side] += seg.get('turn_side_agg_wsum', {}).get(side, 0.0)
            total['turn_side_soft_m'][side] += seg.get('turn_side_soft_m', {}).get(side, 0.0)
            total['curve_side_m'][side] += seg.get('curve_side_m', {}).get(side, 0.0)
            total['curve_side_agg_wsum'][side] += seg.get('curve_side_agg_wsum', {}).get(side, 0.0)
            total['curve_side_soft_m'][side] += seg.get('curve_side_soft_m', {}).get(side, 0.0)
            total['straight_side_m'][side] += seg.get('straight_side_m', {}).get(side, 0.0)
            total['straight_side_wsum'][side] += seg.get('straight_side_wsum', {}).get(side, 0.0)
            total['turnin_lead_sum'][side] += seg.get('turnin_lead_sum', {}).get(side, 0.0)
            total['turnin_count'][side] += seg.get('turnin_count', {}).get(side, 0)
            total['unwind_lead_sum'][side] += seg.get('unwind_lead_sum', {}).get(side, 0.0)
            total['unwind_count'][side] += seg.get('unwind_count', {}).get(side, 0)
        for k in total['events']:
            total['events'][k] += seg['events'].get(k, 0)
    return total


def _reason_pcts(events: dict) -> dict:
    total = sum(events.values())
    if total == 0:
        return {k: 0.0 for k in events}
    return {k: round(v / total * 100, 1) for k, v in events.items()}


def _merge_avg(old_avg: float, old_weight: float, new_avg: float, new_weight: float) -> float:
    """Distance-weighted merge of two averages, e.g. combining a route's average
    divergence into the running lifetime average without needing to keep the raw sum."""
    total_weight = old_weight + new_weight
    if total_weight == 0:
        return 0.0
    return (old_avg * old_weight + new_avg * new_weight) / total_weight


def _startup_verify(log_root: str, processed: set[str], lifetime: dict,
                    params: Params) -> tuple[dict, set[str]]:
    """
    Schema check then disk presence check.

    If stored lifetime data is missing any field from REQUIRED_LIFETIME_FIELDS,
    all routes are cleared for full reanalysis - this automatically triggers
    whenever a new data field is added to the analyzer.

    Returns (lifetime, processed), either of which may have been reset.
    """
    if processed:
        raw = params.get("SpysyLifetimeStats")
        needs_reanalysis = False
        if not raw:
            # Processed set is non-empty but lifetime data is gone - inconsistent state.
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
            params.put("SpysyStatsStatus", "New data fields detected - reanalyzing all routes...")
            processed = set()
            lifetime = dict(_LIFETIME_ZERO)
            _save_lifetime(params, lifetime)
            _save_processed(params, processed)
            return lifetime, processed

    if not processed:
        return lifetime, processed

    # Disk presence check - confirm each tracked route still exists
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
                'override_m': stored.get('override_mi', 0.0) * METERS_PER_MILE,
                'turn_m': stored.get('turn_mi', 0.0) * METERS_PER_MILE,
                'straight_m': stored.get('straight_mi', 0.0) * METERS_PER_MILE,
                'curve_m': stored.get('curve_mi', 0.0) * METERS_PER_MILE,
                'lane_change_m': stored.get('lane_change_mi', 0.0) * METERS_PER_MILE,
                'avg_divergence_deg': stored.get('avg_divergence_deg', 0.0),
                'turn_left_m': stored.get('turn_left_mi', 0.0) * METERS_PER_MILE,
                'turn_right_m': stored.get('turn_right_mi', 0.0) * METERS_PER_MILE,
                'avg_agg_left_deg': stored.get('avg_agg_left_deg', 0.0),
                'avg_agg_right_deg': stored.get('avg_agg_right_deg', 0.0),
                'soft_pct_left': stored.get('soft_pct_left', 0.0),
                'soft_pct_right': stored.get('soft_pct_right', 0.0),
                'avg_unwind_lead_left_s': stored.get('avg_unwind_lead_left_s', 0.0),
                'avg_unwind_lead_right_s': stored.get('avg_unwind_lead_right_s', 0.0),
                'unwind_count_left': stored.get('unwind_count_left', 0),
                'unwind_count_right': stored.get('unwind_count_right', 0),
                'avg_turnin_lead_left_s': stored.get('avg_turnin_lead_left_s', 0.0),
                'avg_turnin_lead_right_s': stored.get('avg_turnin_lead_right_s', 0.0),
                'turnin_count_left': stored.get('turnin_count_left', 0),
                'turnin_count_right': stored.get('turnin_count_right', 0),
                'straight_left_m': stored.get('straight_left_mi', 0.0) * METERS_PER_MILE,
                'straight_right_m': stored.get('straight_right_mi', 0.0) * METERS_PER_MILE,
                'avg_pull_left_deg': stored.get('avg_pull_left_deg', 0.0),
                'avg_pull_right_deg': stored.get('avg_pull_right_deg', 0.0),
                'curve_left_m': stored.get('curve_left_mi', 0.0) * METERS_PER_MILE,
                'curve_right_m': stored.get('curve_right_mi', 0.0) * METERS_PER_MILE,
                'avg_curve_agg_left_deg': stored.get('avg_curve_agg_left_deg', 0.0),
                'avg_curve_agg_right_deg': stored.get('avg_curve_agg_right_deg', 0.0),
                'curve_soft_pct_left': stored.get('curve_soft_pct_left', 0.0),
                'curve_soft_pct_right': stored.get('curve_soft_pct_right', 0.0),
            }
    except Exception:
        pass
    return dict(_LIFETIME_ZERO)


def _save_lifetime(params: Params, lifetime: dict):
    params.put("SpysyLifetimeStats", json.dumps({
        'engaged_mi': round(lifetime['engaged_m'] / METERS_PER_MILE, 2),
        'disengaged_mi': round(lifetime['disengaged_m'] / METERS_PER_MILE, 2),
        'aol_mi': round(lifetime['aol_m'] / METERS_PER_MILE, 2),
        'override_mi': round(lifetime['override_m'] / METERS_PER_MILE, 2),
        'turn_mi': round(lifetime['turn_m'] / METERS_PER_MILE, 2),
        'straight_mi': round(lifetime['straight_m'] / METERS_PER_MILE, 2),
        'curve_mi': round(lifetime['curve_m'] / METERS_PER_MILE, 2),
        'lane_change_mi': round(lifetime['lane_change_m'] / METERS_PER_MILE, 2),
        'avg_divergence_deg': round(lifetime['avg_divergence_deg'], 2),
        'turn_left_mi': round(lifetime['turn_left_m'] / METERS_PER_MILE, 2),
        'turn_right_mi': round(lifetime['turn_right_m'] / METERS_PER_MILE, 2),
        'avg_agg_left_deg': round(lifetime['avg_agg_left_deg'], 2),
        'avg_agg_right_deg': round(lifetime['avg_agg_right_deg'], 2),
        'soft_pct_left': round(lifetime['soft_pct_left'], 1),
        'soft_pct_right': round(lifetime['soft_pct_right'], 1),
        'avg_unwind_lead_left_s': round(lifetime['avg_unwind_lead_left_s'], 2),
        'avg_unwind_lead_right_s': round(lifetime['avg_unwind_lead_right_s'], 2),
        'unwind_count_left': lifetime['unwind_count_left'],
        'unwind_count_right': lifetime['unwind_count_right'],
        'avg_turnin_lead_left_s': round(lifetime['avg_turnin_lead_left_s'], 2),
        'avg_turnin_lead_right_s': round(lifetime['avg_turnin_lead_right_s'], 2),
        'turnin_count_left': lifetime['turnin_count_left'],
        'turnin_count_right': lifetime['turnin_count_right'],
        'straight_left_mi': round(lifetime['straight_left_m'] / METERS_PER_MILE, 2),
        'straight_right_mi': round(lifetime['straight_right_m'] / METERS_PER_MILE, 2),
        'avg_pull_left_deg': round(lifetime['avg_pull_left_deg'], 2),
        'avg_pull_right_deg': round(lifetime['avg_pull_right_deg'], 2),
        'curve_left_mi': round(lifetime['curve_left_m'] / METERS_PER_MILE, 2),
        'curve_right_mi': round(lifetime['curve_right_m'] / METERS_PER_MILE, 2),
        'avg_curve_agg_left_deg': round(lifetime['avg_curve_agg_left_deg'], 2),
        'avg_curve_agg_right_deg': round(lifetime['avg_curve_agg_right_deg'], 2),
        'curve_soft_pct_left': round(lifetime['curve_soft_pct_left'], 1),
        'curve_soft_pct_right': round(lifetime['curve_soft_pct_right'], 1),
    }))


def _load_processed(params: Params) -> set[str]:
    try:
        raw = params.get("SpysyProcessedRoutes")
        return set(json.loads(raw)) if raw else set()
    except Exception:
        return set()


def _save_processed(params: Params, processed: set[str]):
    params.put("SpysyProcessedRoutes", json.dumps(sorted(processed)))


def _merge_side_detail(lifetime: dict, route_side_m: dict, route_agg_wsum: dict, route_soft_m: dict,
                       m_key_fmt: str, agg_key_fmt: str, soft_key_fmt: str) -> dict:
    """Shared merge for turn/curve side detail (aggression avg + soft-time %).
    Returns this route's per-side values for the caller's last_drive dict."""
    route_detail = {}
    for side in ('left', 'right'):
        side_m = route_side_m[side]
        route_agg = route_agg_wsum[side] / side_m if side_m > 0 else 0.0
        route_soft_pct = round(route_soft_m[side] / side_m * 100, 1) if side_m > 0 else 0.0

        m_key, agg_key, soft_key = m_key_fmt.format(side), agg_key_fmt.format(side), soft_key_fmt.format(side)
        lifetime[agg_key] = _merge_avg(lifetime[agg_key], lifetime[m_key], route_agg, side_m)
        lifetime[soft_key] = _merge_avg(lifetime[soft_key], lifetime[m_key], route_soft_pct, side_m)
        lifetime[m_key] += side_m

        route_detail[side] = {'agg_deg': round(route_agg, 2), 'soft_pct': route_soft_pct}
    return route_detail


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
            lifetime = dict(_LIFETIME_ZERO)
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

            # Override analysis: override_pct is share of controlled (engaged+AOL) distance
            # spent overriding; turn/straight/curve/lane_change_pct break that override time
            # down by context, so they sum to 100% of override_pct. curve_pct absorbs the
            # rounding remainder, same trick as disengaged_pct above.
            controlled_m = drive['engaged_m'] + drive['aol_m']
            override_pct = round(drive['override_m'] / controlled_m * 100, 1) if controlled_m > 0 else 0.0
            route_avg_divergence = drive['divergence_wsum'] / drive['override_m'] if drive['override_m'] > 0 else 0.0
            turn_pct = round(drive['turn_m'] / drive['override_m'] * 100, 1) if drive['override_m'] > 0 else 0.0
            straight_pct = round(drive['straight_m'] / drive['override_m'] * 100, 1) if drive['override_m'] > 0 else 0.0
            lane_change_pct = round(drive['lane_change_m'] / drive['override_m'] * 100, 1) if drive['override_m'] > 0 else 0.0
            curve_pct = round(100.0 - turn_pct - straight_pct - lane_change_pct, 1) if drive['override_m'] > 0 else 0.0

            lifetime['avg_divergence_deg'] = _merge_avg(
                lifetime['avg_divergence_deg'], lifetime['override_m'], route_avg_divergence, drive['override_m'])
            lifetime['override_m'] += drive['override_m']
            lifetime['turn_m'] += drive['turn_m']
            lifetime['straight_m'] += drive['straight_m']
            lifetime['curve_m'] += drive['curve_m']
            lifetime['lane_change_m'] += drive['lane_change_m']

            # Turn detail, split left/right: aggression is a signed avg (+ = model too soft,
            # driver pushes harder; - = model too aggressive, driver backs off).
            turn_detail = _merge_side_detail(
                lifetime, drive['turn_side_m'], drive['turn_side_agg_wsum'], drive['turn_side_soft_m'],
                'turn_{}_m', 'avg_agg_{}_deg', 'soft_pct_{}')

            # Curve detail: same aggression concept, scoped to curves instead of sharp turns.
            curve_detail = _merge_side_detail(
                lifetime, drive['curve_side_m'], drive['curve_side_agg_wsum'], drive['curve_side_soft_m'],
                'curve_{}_m', 'avg_curve_agg_{}_deg', 'curve_soft_pct_{}')

            # Straight-line detail: which way the driver tends to pull, and by how much.
            straight_detail = {}
            for side in ('left', 'right'):
                side_m = drive['straight_side_m'][side]
                route_pull = drive['straight_side_wsum'][side] / side_m if side_m > 0 else 0.0
                lifetime[f'avg_pull_{side}_deg'] = _merge_avg(
                    lifetime[f'avg_pull_{side}_deg'], lifetime[f'straight_{side}_m'], route_pull, side_m)
                lifetime[f'straight_{side}_m'] += side_m
                straight_detail[side] = round(route_pull, 2)

            # Turn-in / unwind timing, split left/right.
            timing_detail: dict = {}
            for kind, lead_sum, count, lifetime_avg_fmt, lifetime_count_fmt in (
                ('turnin', drive['turnin_lead_sum'], drive['turnin_count'],
                 'avg_turnin_lead_{}_s', 'turnin_count_{}'),
                ('unwind', drive['unwind_lead_sum'], drive['unwind_count'],
                 'avg_unwind_lead_{}_s', 'unwind_count_{}'),
            ):
                timing_detail[kind] = {}
                for side in ('left', 'right'):
                    n = count[side]
                    route_lead = lead_sum[side] / n if n > 0 else 0.0
                    avg_key, count_key = lifetime_avg_fmt.format(side), lifetime_count_fmt.format(side)
                    lifetime[avg_key] = _merge_avg(lifetime[avg_key], lifetime[count_key], route_lead, n)
                    lifetime[count_key] += n
                    timing_detail[kind][side] = {'lead_s': round(route_lead, 2), 'count': n}

            last_drive = {
                'engaged_mi': round(drive['engaged_m'] / METERS_PER_MILE, 2),
                'disengaged_mi': round(drive['disengaged_m'] / METERS_PER_MILE, 2),
                'aol_mi': round(drive['aol_m'] / METERS_PER_MILE, 2),
                'engaged_pct': eng_pct,
                'disengaged_pct': round(100.0 - eng_pct - aol_pct, 1),
                'aol_pct': aol_pct,
                'reasons': _reason_pcts(drive['events']),
                'override_mi': round(drive['override_m'] / METERS_PER_MILE, 2),
                'override_pct': override_pct,
                'avg_divergence_deg': round(route_avg_divergence, 2),
                'turn_pct': turn_pct,
                'straight_pct': straight_pct,
                'curve_pct': curve_pct,
                'lane_change_pct': lane_change_pct,
                'turn_left_agg_deg': turn_detail['left']['agg_deg'],
                'turn_left_soft_pct': turn_detail['left']['soft_pct'],
                'turn_right_agg_deg': turn_detail['right']['agg_deg'],
                'turn_right_soft_pct': turn_detail['right']['soft_pct'],
                'turn_left_turnin_lead_s': timing_detail['turnin']['left']['lead_s'],
                'turn_left_turnin_count': timing_detail['turnin']['left']['count'],
                'turn_right_turnin_lead_s': timing_detail['turnin']['right']['lead_s'],
                'turn_right_turnin_count': timing_detail['turnin']['right']['count'],
                'turn_left_unwind_lead_s': timing_detail['unwind']['left']['lead_s'],
                'turn_left_unwind_count': timing_detail['unwind']['left']['count'],
                'turn_right_unwind_lead_s': timing_detail['unwind']['right']['lead_s'],
                'turn_right_unwind_count': timing_detail['unwind']['right']['count'],
                'curve_left_agg_deg': curve_detail['left']['agg_deg'],
                'curve_left_soft_pct': curve_detail['left']['soft_pct'],
                'curve_right_agg_deg': curve_detail['right']['agg_deg'],
                'curve_right_soft_pct': curve_detail['right']['soft_pct'],
                'straight_left_pull_deg': straight_detail['left'],
                'straight_right_pull_deg': straight_detail['right'],
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

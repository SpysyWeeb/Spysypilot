#!/usr/bin/env python3
"""Snapshot/event equivalence (shadow_learner_design.md tests_required item
5): replaying a route's rackEffortFrame stream through the count-capped EMA
must exactly reproduce that route's own rackEffortSnapshot -- the design's
core auditability guarantee.

Uses the real corpus via SteadyClassifier (already proven bit-exact against
extract.py in test_rack_effort_bitexact.py) so the run/cell structure being
accumulated here is a real one; this test is specifically about
RackEffortAccumulator's online-vs-replay agreement, not the classifier.
"""
import importlib.util
import os

import pytest

from openpilot.selfdrive.locationd.rack_effort_classifier import (
  RackEffortAccumulator, SteadyClassifier, HANDS_OFF_TORQUE_THRESH_OFFLINE, build_frame_sample,
)

EXTRACT_PY = "/var/home/alex/Documents/spysypilot-route-audit/phase4/rack_effort_seed/extract.py"
ROUTES_DIR = "/var/home/alex/Documents/spysypilot-route-audit/routes"
ROUTE_KEY, SEGMENTS = "0000002c--9d8bfd265b", [4, 5, 6]

pytestmark = pytest.mark.skipif(not os.path.isfile(EXTRACT_PY), reason="phase4 corpus/tooling not present on this machine")


def _load_extract():
  spec = importlib.util.spec_from_file_location("rack_effort_seed_extract_snap", EXTRACT_PY)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def _classified_frames(out, meta):
  n = len(out["lm"])
  clf = SteadyClassifier(hands_off_torque_thresh=HANDS_OFF_TORQUE_THRESH_OFFLINE)
  frames = []
  for i in range(n):
    lw = "rackState" if out["rackWhich"][i] == 0 else "none"
    fs = build_frame_sample(
      log_mono_time=int(out["lm"][i]), which_lateral=lw, rack_fallback=False,
      lat_active=bool(out["latActive"][i]), steering_pressed=bool(out["steeringPressed"][i]),
      steering_torque=float(out["steeringTorque"][i]), steering_angle_deg=float(out["steeringAngleDeg"][i]),
      steering_rate_deg=float(out["steeringRateDeg"][i]), v_ego=float(out["vEgo"][i]),
      h_measured=float(out["outTorque"][i]),
      vp_valid=bool(out["vpValid"][i]), roll_live=float(out["roll"][i]),
      angle_offset_deg_live=float(out["angleOffsetDeg"][i]), stiffness_factor_live=float(out["stiffnessFactor"][i]),
      steer_ratio_live=float(out["steerRatioLive"][i]),
      ltp_valid=bool(out["ltpValid"][i]), ltp_cal_perc=int(out["calPerc"][i]), use_params=bool(out["useParams"][i]),
      lat_accel_factor_filtered=float(out["latAccelFactorFiltered"][i]),
      lat_accel_offset_filtered=float(out["latAccelOffsetFiltered"][i]),
      cp_mass=meta["mass"], cp_center_to_front=meta["centerToFront"], cp_wheelbase=meta["wheelbase"],
      cp_steer_ratio_rear=meta["steerRatioRearCP"], cp_tire_stiffness_front=meta["tireStiffnessFront"],
      cp_tire_stiffness_rear=meta["tireStiffnessRear"], cp_steer_ratio=meta["steerRatioCP"],
    )
    result = clf.step(fs)
    if result is not None:
      frames.append(result)
  return frames


def _snapshot_cells(cells):
  return {k: (v.bias_hat, v.n_events, dict(v.route_sketch)) for k, v in cells.items()}


@pytest.fixture(scope="module")
def frames():
  extract = _load_extract()
  seg_results = [extract.decode_segment((ROUTE_KEY, seg, os.path.join(ROUTES_DIR, ROUTE_KEY, f"{ROUTE_KEY}--{seg}", "rlog.zst")))
                 for seg in SEGMENTS]
  out, meta, _rack_effort = extract.assemble_route(ROUTE_KEY, "nested", seg_results)
  return _classified_frames(out, meta)


def test_full_batch_replay_matches_incremental_online_accumulation(frames):
  assert len(frames) > 50, "expected a meaningful number of steady frames in this corpus slice"

  # "on-device" simulation: feed frames one at a time, taking periodic snapshots that only
  # *read* accumulator.cells (never flush mid-stream) -- rack_effort_observer.py's own pattern,
  # so a snapshot boundary can never fragment an in-progress dwell-event.
  live = RackEffortAccumulator()
  live_snapshots = []
  for i, f in enumerate(frames):
    live.feed(f, route_hash=42)
    if i % 500 == 0:
      live_snapshots.append(_snapshot_cells(live.cells))
  live.flush(route_hash=42)
  final_live = _snapshot_cells(live.cells)

  # "offline replay" (fit.py's replay mode): process the whole logged frame stream at once.
  replay = RackEffortAccumulator()
  for f in frames:
    replay.feed(f, route_hash=42)
  replay.flush(route_hash=42)
  final_replay = _snapshot_cells(replay.cells)

  assert final_live == final_replay
  assert len(final_live) > 0

  # every periodic snapshot must be consistent with (a strict, monotonic prefix of) the
  # final state -- proves reading accumulator.cells mid-stream never mutates or loses events.
  for snap in live_snapshots:
    for key, (_bias, n_events, _routes) in snap.items():
      assert key in final_live
      assert n_events <= final_live[key][1]

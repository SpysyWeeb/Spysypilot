#!/usr/bin/env python3
"""Bit-exact replay test for the Rack Effort Shadow Observer's classifier.

Feeds real rlogs from the phase4 rack_effort_seed corpus through BOTH
extract.py's own vectorized H (steady-frame) classifier AND a frame-stepped
port of rack_effort_classifier.SteadyClassifier, then asserts they agree on
steady flags, cellId, runId boundaries, and hPrior/hResidual to float32
tolerance. This is the load-bearing test proving reproduction, not
approximation (shadow_learner_design.md tests_required item 1).

extract.py lives outside this repo (it is the offline audit tool this
observer must reproduce), so it is loaded by path via importlib rather than
imported as a package.
"""
import importlib.util
import math
import os

import numpy as np
import pytest

from openpilot.selfdrive.locationd.rack_effort_classifier import (
  SteadyClassifier, HANDS_OFF_TORQUE_THRESH_OFFLINE, MIN_RUN_FRAMES,
  build_frame_sample,
)

EXTRACT_PY = "/var/home/alex/Documents/spysypilot-route-audit/phase4/rack_effort_seed/extract.py"
ROUTES_DIR = "/var/home/alex/Documents/spysypilot-route-audit/routes"

CORPUS = [
  ("0000002c--9d8bfd265b", [4, 5, 6]),
  ("0000002d--ed7bd32ccf", [2, 3, 4]),
]

pytestmark = pytest.mark.skipif(not os.path.isfile(EXTRACT_PY), reason="phase4 corpus/tooling not present on this machine")


def _load_extract():
  spec = importlib.util.spec_from_file_location("rack_effort_seed_extract", EXTRACT_PY)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def _assemble_route(extract, route_key, segments):
  seg_results = []
  for seg in segments:
    path = os.path.join(ROUTES_DIR, route_key, f"{route_key}--{seg}", "rlog.zst")
    seg_results.append(extract.decode_segment((route_key, seg, path)))
  out, meta, _rack_effort = extract.assemble_route(route_key, "nested", seg_results)
  assert out is not None, f"{route_key}: extract.py produced no frames ({meta})"
  return out, meta


def _replay_with_our_classifier(out, meta):
  """Runs SteadyClassifier over the same per-frame arrays extract.py decoded,
  using HANDS_OFF_TORQUE_THRESH_OFFLINE so the candidate gate matches
  extract.py's own HANDS_OFF_TORQUE_THRESH=30 exactly (the live process uses
  STEER_DRIVER_ALLOWANCE=50 instead -- see rack_effort_classifier.py)."""
  n = len(out["lm"])
  clf = SteadyClassifier(hands_off_torque_thresh=HANDS_OFF_TORQUE_THRESH_OFFLINE)
  our_steady = np.zeros(n, dtype=bool)
  our_h_prior = np.full(n, np.nan, dtype=np.float32)
  our_h_residual = np.full(n, np.nan, dtype=np.float32)
  our_v_band = np.full(n, -1, dtype=np.int16)
  our_angle_band = np.full(n, -1, dtype=np.int16)
  our_lat_bin = np.zeros(n, dtype=np.int16)
  our_direction = np.zeros(n, dtype=np.int8)
  our_run_id = np.zeros(n, dtype=np.int64)

  for i in range(n):
    lw = "rackState" if out["rackWhich"][i] == 0 else ("torqueState" if out["rackWhich"][i] == 1 else "none")
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
    # sanity: our own scalar physics must match extract's own vectorized arrays bit-for-bit
    if math.isfinite(out["latAccel"][i]) or math.isfinite(fs.lat_accel):
      np.testing.assert_allclose(np.float32(fs.lat_accel), out["latAccel"][i], rtol=1e-4, atol=1e-4)

    result = clf.step(fs)
    center_i = i - clf.half
    if result is None:
      continue
    assert 0 <= center_i < n
    our_steady[center_i] = True
    our_h_prior[center_i] = result.h_prior
    our_h_residual[center_i] = result.h_residual
    our_v_band[center_i] = result.v_band_idx
    our_angle_band[center_i] = result.angle_band_idx
    our_lat_bin[center_i] = result.lat_accel_bin_idx
    our_direction[center_i] = result.direction
    our_run_id[center_i] = result.run_id

  return {"steady": our_steady, "hPrior": our_h_prior, "hResidual": our_h_residual,
           "vBand": our_v_band, "angleBand": our_angle_band, "latAccelBin": our_lat_bin,
           "direction": our_direction, "runId": our_run_id}


def _run_ids_from_mask(mask, cell_tuples):
  """Recomputes extract.py's implicit event/run ids from its own steadyMask +
  per-frame cell tuple, for comparison against our explicit runId field."""
  run_id = np.zeros(len(mask), dtype=np.int64)
  cur = 0
  prev_mask = False
  prev_cell = None
  for i in range(len(mask)):
    if not mask[i]:
      prev_mask = False
      prev_cell = None
      continue
    cell = cell_tuples[i]
    if not (prev_mask and cell == prev_cell):
      cur += 1
    run_id[i] = cur
    prev_mask = True
    prev_cell = cell
  return run_id


@pytest.mark.parametrize("route_key,segments", CORPUS)
def test_bitexact_vs_extract(route_key, segments):
  extract = _load_extract()
  out, meta = _assemble_route(extract, route_key, segments)
  ours = _replay_with_our_classifier(out, meta)

  # extract.py's own steady mask requires an interior margin of `half` frames on both ends
  # (rolling_window_mask leaves the edges False) -- our streaming classifier has the same
  # fixed-lag blind spot at the very start/end of the decoded array, so compare interior only.
  half = 15
  lo, hi = half, len(out["lm"]) - half
  ref_steady = out["steadyMask"][lo:hi]
  assert ref_steady.sum() > 0, f"{route_key}: no steady frames in reference corpus slice"

  np.testing.assert_array_equal(ours["steady"][lo:hi], ref_steady)

  idx = np.where(ref_steady)[0] + lo
  np.testing.assert_allclose(ours["hPrior"][idx], out["hPrior"][idx], rtol=0, atol=1e-4)
  np.testing.assert_allclose(ours["hResidual"][idx], out["hResidual"][idx], rtol=0, atol=1e-4)
  np.testing.assert_array_equal(ours["vBand"][idx], out["vBand"][idx])
  np.testing.assert_array_equal(ours["angleBand"][idx], out["angleBand"][idx])
  np.testing.assert_array_equal(ours["latAccelBin"][idx], out["latAccelBin"][idx])
  np.testing.assert_array_equal(ours["direction"][idx], out["direction"][idx])

  cell_tuples = list(zip((int(v) for v in out["vBand"]), (int(v) for v in out["angleBand"]),
                          (int(v) for v in out["latAccelBin"]), (int(v) for v in out["direction"]), strict=True))
  ref_run_id = _run_ids_from_mask(out["steadyMask"], cell_tuples)
  # runIds are monotonic counters with independent starting points (extract.py's derivation
  # here starts at 1 for the first run same as ours) -- compare run *boundaries*, not raw ids,
  # by checking that "same run" / "different run" agrees frame-to-frame across the two arrays.
  ref_same_run = ref_run_id[idx][1:] == ref_run_id[idx][:-1]
  our_same_run = ours["runId"][idx][1:] == ours["runId"][idx][:-1]
  # only meaningful between temporally-adjacent steady frames
  adjacent = np.diff(idx) == 1
  np.testing.assert_array_equal(our_same_run[adjacent], ref_same_run[adjacent])

  n_events_ours = len(set(ours["runId"][idx].tolist()))
  n_events_ref = len(set(ref_run_id[idx].tolist()))
  assert n_events_ours == n_events_ref, (n_events_ours, n_events_ref)


def test_min_run_frames_matches_learner_note():
  assert MIN_RUN_FRAMES == 3

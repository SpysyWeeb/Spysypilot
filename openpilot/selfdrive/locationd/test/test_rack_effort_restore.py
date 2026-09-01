#!/usr/bin/env python3
"""Restore/invalidate roundtrip (shadow_learner_design.md tests_required
item 4), mirroring test_paramsd.py's fingerprint-reset pattern: a matching
restore key restores state, a mismatch leaves the cache alone and starts
empty, and a corrupted blob is discarded via Params.remove() -- the exact
try/except-and-remove() shape torqued.py uses, per owner default (3)."""
import pytest
from opendbc.car.structs import car

from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.locationd.rack_effort_classifier import CellState
from openpilot.selfdrive.locationd.rack_effort_observer import restore_state, snapshot_msg


def _cp(fingerprint="HYUNDAI_PALISADE"):
  cp = car.CarParams.new_message()
  cp.carFingerprint = fingerprint
  return cp


class TestRackEffortRestore(OpenpilotTestCase):
  def test_no_cache_present_starts_empty(self):
    assert restore_state(_cp()) == {}

  def test_matching_restore_key_restores_cells(self):
    params = Params()
    CP = _cp("HYUNDAI_PALISADE")
    params.put("CarParamsPrevRoute", CP.to_bytes(), block=True)

    cells = {(1, 2, -3, 1): CellState(bias_hat=0.42, n_events=7, route_sketch={123: 3}, last_update_mono_time=999)}
    params.put("LiveRackEffortObserver", snapshot_msg(cells, 12345).to_bytes(), block=True)

    restored = restore_state(CP)
    assert set(restored.keys()) == {(1, 2, -3, 1)}
    got = restored[(1, 2, -3, 1)]
    assert got.bias_hat == pytest.approx(0.42, abs=1e-5)
    assert got.n_events == 7
    assert got.route_sketch == {123: 3}
    assert got.last_update_mono_time == 999

  def test_mismatched_fingerprint_starts_empty_but_leaves_cache_alone(self):
    params = Params()
    params.put("CarParamsPrevRoute", _cp("HYUNDAI_PALISADE").to_bytes(), block=True)
    params.put("LiveRackEffortObserver",
               snapshot_msg({(0, 0, 0, 1): CellState(bias_hat=1.0, n_events=5)}, 1).to_bytes(), block=True)

    restored = restore_state(_cp("HYUNDAI_SANTAFE"))
    assert restored == {}
    # a version/fingerprint mismatch is not corruption -- torqued.py's own pattern only
    # removes the cache on a decode exception, never on a clean mismatch.
    assert params.get("LiveRackEffortObserver") is not None

  def test_corrupted_blob_is_removed(self):
    params = Params()
    CP = _cp()
    params.put("CarParamsPrevRoute", CP.to_bytes(), block=True)
    params.put("LiveRackEffortObserver", b"not a valid capnp message", block=True)

    restored = restore_state(CP)
    assert restored == {}
    assert params.get("LiveRackEffortObserver") is None

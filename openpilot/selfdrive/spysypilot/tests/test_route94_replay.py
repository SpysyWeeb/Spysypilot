"""Committed route-94 regression fixture for lateral detector v6.

Ground truth for every assertion in this file is the prior-agent validation
report at
`/home/alex/Documents/spysypilot-route-audit/routes/00000094--8cec74a749/analysis/route94_v6_validation_report.md`
(branch driving-event-platform @ f93ec779f0, Stages A+B). This module
replays COMMITTED, compact windows of real route-94 telemetry (extracted
from that analysis directory's validated route94_lat_samples_v6.csv,
unmodified except for column trimming, 5-decimal rounding, and windowing)
through the CURRENT `LateralEventDetector`, matching the report's findings
exactly -- including where those findings differ from DESIGN.md's original,
more optimistic wording. Honest caveats baked into the assertions below
(see also the per-window "assertion"/"mechanism" documentation in
route94_lat_windows_meta.json):

  * Two pre-existing turnStopTurn "reject" anchors (mono ~141.79, ~211.68)
    are deliberately NOT in this fixture: both were live-daemon-only
    artifacts never reproducible in ANY offline replay (v5 or v6), so
    testing their absence would not exercise any v6-specific logic.
  * Of the three turn-stop reject anchors that ARE included, each fails via
    a DIFFERENT mechanism (never-enters-dwell twice over, center-band-drift
    reset once, state fragmentation once) -- not a uniform max-dwell
    timeout. Each gets a tailored anti-vacuity probe.
  * Of 20 short-dwell turnStopTurn keeps, 19 fire and 1 (v5 anchor mono
    735.289) does not -- not because of the new progress gates, but because
    a genuinely new v6-only turnStopTurn (mono 731.977) fires 5.4s earlier
    and consumes the shared 8s per-type EVENT_COOLDOWN (amendment A10,
    already-accepted tradeoff).
  * The 4 driver-rescue anchors classify `driver_causation == "driverCreated"`,
    NOT "interventionBacked" as DESIGN.md S6 hoped -- this route is
    hands-on-heavy (continuous driver presence through arm/deficit-start/
    evaluation), so the "clean autonomous deficit + later rescue" path
    structurally never occurs here. Reviewed and accepted by the user.
    `driver_assisted_unwind=True` and confidence 0.95 ARE preserved
    correctly for all 4 (the v5-era rescue mechanism still works).
  * Crown-neutral timestamp presence is data-dependent, not universal: only
    5 of 7 unwind keeps have `requested_crown_neutral_mono_time` present,
    and only 1 of 4 no-assist keeps has `applied_crown_neutral_mono_time`
    present -- both are physically correct (saturated/oscillating torque
    near emission never sustains the 100ms sub-0.05 hold), not bugs.

Replay style deliberately follows this suite's established 100 Hz-builder
convention (test_lat_event_detector.py) as closely as a real-telemetry
replay allows: no runner-specific fixtures, no wall-clock, one small `_replay` helper
stepping `detector.update(...)` over real ordered samples instead of a
synthetic waveform. Unlike the rest of the suite, replay uses the DEFAULT
`LateralEventDetector()` (EVENT_COOLDOWN=8.0s active, not cooldown=0) --
required to reproduce the real cooldown-consumption behavior above.
"""
import csv
import io
import json
import math
from pathlib import Path

import zstandard

from openpilot.cereal import messaging
from openpilot.common.parameterized import parameterized
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.spysypilot.driving_event_indexer import _assign_group_role, event_to_record
from openpilot.selfdrive.spysypilot.lat_event_detector import LateralEventDetector, LateralSample


DATA_DIR = Path(__file__).parent / "data"
META = json.loads((DATA_DIR / "route94_lat_windows_meta.json").read_text())
WINDOWS_BY_LABEL = {window["label"]: window for window in META["windows"]}


class _Approx:
  def __init__(self, expected: float, rel: float, abs_tol: float):
    self.expected = expected
    self.rel = rel
    self.abs_tol = abs_tol

  def __eq__(self, actual):
    return math.isclose(actual, self.expected, rel_tol=self.rel, abs_tol=self.abs_tol)


def approx(expected: float, *, rel: float = 1e-6, abs_tol: float = 1e-12):
  return _Approx(expected, rel, abs_tol)


def _load_fixture_rows() -> list[dict[str, str]]:
  compressed = (DATA_DIR / META["fixture_csv"]).read_bytes()
  # Bounded decompression (matches the sibling extraction-script convention
  # elsewhere in this project family); harmless here since this is trusted,
  # committed first-party test data, but free to keep consistent.
  csv_bytes = zstandard.ZstdDecompressor().decompress(compressed, max_output_size=2 * 1024**3)
  return list(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8"))))


_ROWS = _load_fixture_rows()


def _number(row: dict[str, str], name: str, default: float = 0.0) -> float:
  text = row.get(name, "")
  return float(text) if text else default


def _number_or_none(row: dict[str, str], name: str) -> float | None:
  text = row.get(name, "")
  return float(text) if text else None


def _boolean(row: dict[str, str], name: str) -> bool:
  return row.get(name) == "True"


def _make_sample(row: dict[str, str]) -> LateralSample:
  """CSV -> LateralSample, matching the analysis directory's validated
  replay_route94_lat_v6.py field mapping exactly (map-replaykit.md S4)."""
  return LateralSample(
    mono_time=_number(row, "mono_s"),
    active=_boolean(row, "active") and _boolean(row, "lat_active"),
    v_ego=_number(row, "v_ego"),
    steering_angle_deg=_number(row, "steering_angle_deg"),
    steering_rate_deg=_number(row, "filtered_steering_rate_deg"),
    driver_torque=_number(row, "steering_torque"),
    steering_pressed=_boolean(row, "steering_pressed"),
    desired_lateral_accel=_number(row, "desired_lateral_accel"),
    actual_lateral_accel=_number(row, "actual_lateral_accel"),
    request_torque=_number(row, "requested_torque"),
    applied_torque=_number(row, "applied_torque"),
    reference_rate=_number(row, "reference_rate"),
    measurement_rate=_number(row, "tracking_measurement_rate"),
    reference_target_torque=_number(row, "reference_target_torque"),
    reference_unwind_scale=_number(row, "reference_unwind_scale"),
    reference_sustained_unwind_scale=_number(row, "reference_sustained_unwind_scale"),
    unwind_effective_phase=_number(row, "unwind_effective_phase"),
    unwind_overspeed=_number(row, "unwind_overspeed"),
    unwind_same_episode=_boolean(row, "unwind_same_episode"),
    road_confounded=_boolean(row, "road_confounded"),
    vertical_accel_deviation=_number(row, "vertical_accel_deviation"),
    actual_damping_amount=_number(row, "damping_applied"),
    actual_damping_state=row.get("damping_state", "unavailable"),
    turn_in_blocked=_boolean(row, "damping_blocked"),
    breakaway_latch=_number(row, "breakaway_latch"),
    sustain_floor_contribution=_number(row, "damping_floor"),
    damping_version=int(_number(row, "damping_version")),
    unwind_neutral_torque=_number(row, "unwind_neutral_torque_v6"),
    unwind_phase_direction=_number(row, "unwind_phase_direction"),
    high_angle_unwind_scale=_number_or_none(row, "high_angle_unwind_scale"),
    torque_command_before_high_angle_exit=_number_or_none(row, "torque_command_before_high_angle_exit"),
    high_angle_unwind_old_torque_correction=_number_or_none(row, "high_angle_unwind_old_torque_correction"),
    high_angle_unwind_old_direction_torque=_number_or_none(row, "high_angle_unwind_old_direction_torque"),
  )


def _replay(start: float, end: float, *, track_dwell: bool = False):
  """Replay committed fixture rows in [start, end] through a FRESH
  LateralEventDetector (default cooldown -- see module docstring)."""
  detector = LateralEventDetector()
  events = []
  dwell_trace = []
  previous = (None, None)
  for row in _ROWS:
    mono = float(row["mono_s"])
    if mono < start or mono > end:
      continue
    sample = _make_sample(row)
    detected = detector.update(sample)
    if track_dwell:
      episode = detector.turn_stop_episode
      state = episode.state if episode is not None else None
      episode_id = id(episode) if episode is not None else None
      if (state, episode_id) != previous:
        dwell_trace.append((mono, state))
        previous = (state, episode_id)
    if detected is None:
      continue
    batch = detected if isinstance(detected, (list, tuple)) else [detected]
    events.extend(batch)
  return events, dwell_trace


# --------------------------------------------------------------------------
# A. Seven unwind-deficit keeps + the 10:07.56 driver-rescue extra (8 total)
# --------------------------------------------------------------------------

UNWIND_WINDOWS = [window for window in META["windows"] if window["kind"] == "unwind_keep"]


def _test_unwind_deficit_keeps_fire_with_expected_evidence(window):
  events, _ = _replay(window["mono_start"], window["mono_end"])
  matches = [
    event for event in events
    if event.event_type == "unwindProgressDeficit"
    and event.occurred_mono_time == approx(window["anchor_mono"], abs_tol=0.01)
  ]
  assert len(matches) == 1, f"expected exactly one unwindProgressDeficit near {window['anchor_mono']}, got {events}"
  event = matches[0]
  expect = window["expect"]
  ev = event.evidence

  assert event.confidence == approx(expect["confidence"])
  assert ev.driver_causation == expect["driver_causation"]
  assert ev.driver_assisted_unwind == expect["driver_assisted_unwind"]
  assert ev.driver_assist_raw_torque_only == expect["driver_assist_raw_torque_only"]

  # Crown-neutral presence/absence must match the report exactly per anchor
  # (amendment A13 scoping) -- NOT a blanket "present" expectation.
  if expect["requested_crown_neutral_mono_time"] is None:
    assert ev.requested_crown_neutral_mono_time is None
  else:
    assert ev.requested_crown_neutral_mono_time == approx(expect["requested_crown_neutral_mono_time"], abs_tol=0.01)
  if expect["applied_crown_neutral_mono_time"] is None:
    assert ev.applied_crown_neutral_mono_time is None
  else:
    assert ev.applied_crown_neutral_mono_time == approx(expect["applied_crown_neutral_mono_time"], abs_tol=0.01)

  # Meaningful crown-vs-literal-zero deltas (report section E), where the
  # report found one -- demonstrates the crown-aware fix actually changes
  # the answer on these specific episodes.
  if "requested_vs_literal_delta_s" in expect:
    delta = ev.requested_crown_neutral_mono_time - ev.requested_torque_neutral_cross_mono_time
    assert delta == approx(expect["requested_vs_literal_delta_s"], abs_tol=0.01)

  # Rebound tracker: pinned exactly for the 2 confirmed rebounds (report
  # section F) and asserted zero/absent for the anchors that have none.
  assert ev.unwind_rebound_max_magnitude == approx(expect["rebound_max_magnitude"], abs_tol=0.001)
  if expect["rebound_start_mono_time"] is None:
    assert ev.unwind_rebound_start_mono_time is None
  else:
    assert ev.unwind_rebound_start_mono_time == approx(expect["rebound_start_mono_time"], abs_tol=0.01)
  assert ev.unwind_rebound_duration_s == approx(expect["rebound_duration_s"], abs_tol=0.001)

  if "detected_mono_time" in expect:
    assert event.detected_mono_time == approx(expect["detected_mono_time"], abs_tol=0.1)


# --------------------------------------------------------------------------
# B. Three turn-stop reject anchors that genuinely exercise v6 mechanisms
#    (the other 2 named-in-DESIGN long-dwell rejects, mono ~141.79/~211.68,
#    are pre-existing live-only artifacts never reproducible in ANY replay
#    and are deliberately NOT part of this fixture -- see meta.json).
# --------------------------------------------------------------------------

def _test_reject_06_38_9_never_enters_dwell_near_anchor():
  """v6-specific mechanism: the episode DOES become active (moving/slowing)
  essentially at the anchor, but never forms a 'dwell' there at all -- the
  pre-dwell-progress/episode-identity gates suppress candidate FORMATION,
  not a post-dwell discard. Anti-vacuity: prove activity started near the
  anchor (not silently absent) yet no dwell followed within 3s of it."""
  window = WINDOWS_BY_LABEL["reject_06_38_9_never_enters_dwell"]
  events, trace = _replay(window["mono_start"], window["mono_end"], track_dwell=True)
  assert not any(event.event_type == "turnStopTurn" for event in events)

  probe = window["anti_vacuity_probe"]
  moving_entries = [t for t, state in trace if state == "moving"]
  assert any(t == approx(probe["expected_moving_entry_mono"], abs_tol=0.05) for t in moving_entries), (
    "anti-vacuity guard failed: the detector never even entered 'moving' near the anchor"
  )
  dwell_entries = [t for t, state in trace if state == "dwell"]
  assert dwell_entries, "anti-vacuity guard failed: no dwell entries observed at all in the window"
  assert all(abs(t - window["anchor_mono"]) > probe["no_dwell_within_s_of_anchor"] for t in dwell_entries), (
    "a dwell was entered too close to the anchor -- this would falsify the never-enters-dwell mechanism"
  )
  assert dwell_entries[0] == approx(probe["nearest_dwell_entry_mono"], abs_tol=0.05)


def _test_reject_08_33_7_center_band_reset_never_releases():
  """v6-specific mechanism: a dwell IS entered (state machine reached
  'dwell', not silently absent), but resets DIRECTLY to no-episode after
  ~1.1s when the wheel drifts back inside the center band with no
  meaningful demand (DESIGN S2 rule 4) -- it never passes through
  'released', so no PendingTurnStopCandidate is ever formed."""
  window = WINDOWS_BY_LABEL["reject_08_33_7_center_band_reset"]
  events, trace = _replay(window["mono_start"], window["mono_end"], track_dwell=True)
  assert not any(event.event_type == "turnStopTurn" for event in events)

  probe = window["anti_vacuity_probe"]
  dwell_entries = [(t, state) for t, state in trace if state == "dwell"]
  assert len(dwell_entries) == 1, "anti-vacuity guard failed: expected exactly one dwell entry"
  dwell_mono, _ = dwell_entries[0]
  assert dwell_mono == approx(probe["expected_dwell_entry_mono"], abs_tol=0.01)

  index = trace.index((dwell_mono, "dwell"))
  next_mono, next_state = trace[index + 1]
  assert next_state is None, "dwell must reset directly to no-episode, never to 'released'"
  assert next_mono == approx(probe["expected_dwell_exit_mono"], abs_tol=0.01)
  assert (next_mono - dwell_mono) == approx(probe["expected_dwell_duration_s"], abs_tol=0.01)


def _test_reject_seg9_649_0_fragmentation_and_byproduct_event():
  """v6-specific mechanism: instead of one sustained ~7.1s dwell, the state
  machine cycles rapidly through moving/slowing/short-dwell/released
  fragments. The anchor's OWN attempted dwell never survives as itself, but
  one of the fragments (the second) DOES clear every gate and produces a
  genuinely new, real turnStopTurn -- asserted explicitly here rather than
  silently passed through, since it is not named in the validation report's
  anchor tables."""
  window = WINDOWS_BY_LABEL["reject_seg9_649_0_fragmentation"]
  events, trace = _replay(window["mono_start"], window["mono_end"], track_dwell=True)
  turn_stop_events = [event for event in events if event.event_type == "turnStopTurn"]

  assert not any(
    abs(event.occurred_mono_time - window["anchor_mono"]) < window["expect_no_event_near_anchor_s"]
    for event in turn_stop_events
  ), "an event fired too close to the anchor -- this would falsify the fragmentation mechanism"

  probe = window["anti_vacuity_probe"]
  dwell_entries = [t for t, state in trace if state == "dwell"]
  assert len(dwell_entries) >= probe["min_fragment_count"], "anti-vacuity guard failed: not enough fragmented dwell cycles observed"
  for expected_mono in probe["expected_dwell_entries_mono"]:
    assert any(t == approx(expected_mono, abs_tol=0.01) for t in dwell_entries)

  byproduct = window["byproduct_event"]
  assert len(turn_stop_events) == 1, "expected exactly the one documented byproduct turnStopTurn, nothing else"
  event = turn_stop_events[0]
  assert event.occurred_mono_time == approx(byproduct["occurred_mono_time"], abs_tol=0.005)
  assert event.detected_mono_time == approx(byproduct["detected_mono_time"], abs_tol=0.1)
  ev = event.evidence
  assert ev.dwell_duration_s == approx(byproduct["dwell_duration_s"], abs_tol=0.005)
  assert ev.dwell_steering_angle_deg == approx(byproduct["dwell_steering_angle_deg"], abs_tol=0.5)
  assert ev.restart_classification == byproduct["restart_classification"]


# --------------------------------------------------------------------------
# C. 19-of-20 short-dwell turnStopTurn keeps, plus the dedicated
#    731.977/735.289 cooldown-consumption pair (amendment A10).
# --------------------------------------------------------------------------

KEEP_WINDOWS = [window for window in META["windows"] if window["kind"] == "turn_stop_keep"]


def _test_short_dwell_turn_stop_keeps_fire(window):
  events, _ = _replay(window["mono_start"], window["mono_end"])
  matches = [
    event for event in events
    if event.event_type == "turnStopTurn"
    and event.occurred_mono_time == approx(window["expect_occurred_mono_time"], abs_tol=0.01)
  ]
  assert len(matches) == 1, f"expected exactly one turnStopTurn near {window['expect_occurred_mono_time']}, got {events}"


def _test_cooldown_pair_731_977_fires_735_289_does_not():
  """Amendment A10, documented not as a bug: a genuinely NEW v6-only
  turnStopTurn at mono ~731.977 fires and consumes the shared 8s per-type
  EVENT_COOLDOWN. The 20th short-dwell keep (v5 anchor mono 735.289) then
  does NOT fire -- its dwell/release physically happens right on schedule
  and clears BOTH the pre- and post-dwell progress gates comfortably, but
  737.389 (its own 2.1s cluster deadline) - 731.977 = 5.41s < 8.0s, so
  _emit's cooldown gate silently returns None. This is the accepted
  cooldown tradeoff, not a new-discard-rule loss -- assert both halves
  explicitly rather than letting the absence pass unexamined."""
  window = WINDOWS_BY_LABEL["cooldown_pair_731_977_735_289"]
  events, _ = _replay(window["mono_start"], window["mono_end"])
  turn_stop_events = [event for event in events if event.event_type == "turnStopTurn"]

  fires = window["fires"]
  does_not_fire = window["does_not_fire"]
  assert not any(
    abs(event.occurred_mono_time - does_not_fire["anchor_mono"]) < 1.0 for event in turn_stop_events
  ), "735.289 correctly must not fire -- if it did, the cooldown-consumption tradeoff would no longer hold"

  matches = [event for event in turn_stop_events if event.occurred_mono_time == approx(fires["anchor_mono"], abs_tol=0.01)]
  assert len(matches) == 1
  event = matches[0]
  assert event.detected_mono_time == approx(fires["detected_mono_time"], abs_tol=0.1)
  ev = event.evidence
  assert ev.dwell_duration_s == approx(fires["dwell_duration_s"], abs_tol=0.005)
  assert ev.dwell_steering_angle_deg == approx(fires["dwell_steering_angle_deg"], abs_tol=0.5)
  assert ev.turn_stop_pre_dwell_progress_deg == approx(fires["turn_stop_pre_dwell_progress_deg"], abs_tol=0.5)
  assert ev.turn_stop_post_dwell_progress_deg == approx(fires["turn_stop_post_dwell_progress_deg"], abs_tol=0.5)
  assert ev.restart_classification == fires["restart_classification"]


# --------------------------------------------------------------------------
# D. Old (pre-v6) manifest lines must keep decoding safely (append-only).
# --------------------------------------------------------------------------

def _test_old_v5_manifest_lines_still_decode_safely():
  """Synthesized (NOT real-device) envelope-version</detector-version-2-5
  DrivingEvent messages, shaped like a real v5 manifest line (device ran
  branch combo, detector_version 5, detector blatLateralEventDetector --
  see map-replaykit.md) but with route/event/git identifiers replaced by
  fixed test strings, per amendment A11 ("do NOT commit real device
  identifiers"). Exercises the REAL `event_to_record` + `_assign_group_role`
  driving_event_indexer.py functions this fixture's own v6 records rely on
  for append-only decoding -- not a hand-rolled schema check -- confirming
  version<3 / detector_version 2-5 style records decode safely: group_role
  is always "primary" (DESIGNATION_MIN_VERSION=3 gate, amendment A4) and
  every v6-only payload field (driver_causation, crown-neutral timestamps,
  rebound fields, ...) serializes to a safe zero/false/"" default rather
  than raising. A full segment-scanning indexer integration (tmp_path
  segment trees + scan_once/rebuild, as in test_driving_event_indexer.py)
  was judged too heavy for this fixture-focused test file; calling
  event_to_record/_assign_group_role directly is the same real code path
  minus the filesystem/segment-tree scaffolding.
  """
  cases = [
    (1, 2, "test-route--0", "test-event-0", "test-group-0"),
    (2, 3, "test-route--0", "test-event-1", "test-group-1"),
    (2, 4, "test-route--0", "test-event-2", "test-group-2"),
    (2, 5, "test-route--0", "test-event-3", "test-group-3"),
  ]
  for index, (envelope_version, detector_version, route, event_id, group_id) in enumerate(cases):
    occurred = 100_000_000_000 + index * 1_000_000_000
    msg = messaging.new_message("drivingEvent", valid=True)
    event = msg.drivingEvent
    event.version = envelope_version
    event.eventId = event_id
    event.groupId = group_id
    event.occurredMonoTime = occurred
    event.domain = "lateral"
    event.source = "automatic"
    event.eventType = "lat.unwindProgressDeficit"
    event.detector = "blatLateralEventDetector"
    event.detectorVersion = detector_version
    event.severity = "warning"
    event.confidence = 0.9
    event.attribution = "controller"
    event.gitCommit = "test-branch"
    event.gitBranch = "test-branch"
    payload = event.payload.init("lateral")
    payload.unwindEpisodeStartMonoTime = occurred - 2_000_000_000
    payload.unwindDeficitStartMonoTime = occurred - 500_000_000
    payload.driverAssistedUnwind = detector_version >= 5

    record = event_to_record(
      msg.as_reader().drivingEvent, None, route, 0, occurred,
      occurred - 10_000_000_000, occurred - 20_000_000_000,
    )
    assert record["route"] == route
    assert record["event_id"] == event_id
    assert record["detector_version"] == detector_version

    # v6-only fields must all be safe append-only defaults, never raise.
    assert record["payload"]["driver_causation"] == ""
    assert record["payload"]["requested_crown_neutral_present"] is False
    assert record["payload"]["applied_crown_neutral_present"] is False
    assert record["payload"]["unwind_rebound_max_magnitude"] == 0.0
    assert record["payload"]["unwind_rebound_start_present"] is False
    assert record["payload"]["unwind_neutral_torque"] == 0.0
    assert record["payload"]["high_angle_evidence_valid"] is False
    assert record["payload"]["driver_assist_raw_torque_only"] is False
    assert record["payload"]["turn_stop_pre_dwell_progress_deg"] == 0.0
    assert record["payload"]["turn_stop_post_dwell_progress_deg"] == 0.0

    _assign_group_role(record, {}, {})
    assert record["group_role"] == "primary"
    assert record["primary_event_id"] == event_id


class TestRoute94Replay(OpenpilotTestCase):
  @parameterized.expand(UNWIND_WINDOWS, ids=lambda window: window["label"])
  def test_unwind_deficit_keeps_fire_with_expected_evidence(self, window):
    _test_unwind_deficit_keeps_fire_with_expected_evidence(window)

  test_reject_06_38_9_never_enters_dwell_near_anchor = staticmethod(
    _test_reject_06_38_9_never_enters_dwell_near_anchor)
  test_reject_08_33_7_center_band_reset_never_releases = staticmethod(
    _test_reject_08_33_7_center_band_reset_never_releases)
  test_reject_seg9_649_0_fragmentation_and_byproduct_event = staticmethod(
    _test_reject_seg9_649_0_fragmentation_and_byproduct_event)

  @parameterized.expand(KEEP_WINDOWS, ids=lambda window: window["label"])
  def test_short_dwell_turn_stop_keeps_fire(self, window):
    _test_short_dwell_turn_stop_keeps_fire(window)

  test_cooldown_pair_731_977_fires_735_289_does_not = staticmethod(
    _test_cooldown_pair_731_977_fires_735_289_does_not)
  test_old_v5_manifest_lines_still_decode_safely = staticmethod(_test_old_v5_manifest_lines_still_decode_safely)

from __future__ import annotations

from dataclasses import replace
import gc
from pathlib import Path
import random
import struct
import tracemalloc

import pytest  # noqa: TID251

from openpilot.selfdrive.controls.lib.blatv2 import behavior_segmentation
from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import (
  BehaviorSample,
  BehaviorSourceIdentity,
  EventLocator,
  ManeuverPhase,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_filebacked import (
  BehaviorScratchError,
  BehaviorSampleScratch,
  score_file_backed_window,
  segment_file_backed_behavior_route,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_metrics import (
  aggregate_behavior_metrics,
  BehaviorMetricConfig,
  score_behavior,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_segmentation import (
  _Run,
  _Span,
  _SpanScratch,
  BoundaryReason,
  SegmentationResourceError,
  SegmentationConfig,
  segment_behavior_route,
)


SOURCE = BehaviorSourceIdentity(
  controller_name="file-backed-test",
  controller_artifact_sha256="a" * 64,
  source_openpilot_commit="b" * 40,
  opendbc_commit="c" * 40,
  panda_commit="d" * 40,
  evidence_schema_version=1,
)


def segmentation_config() -> SegmentationConfig:
  return SegmentationConfig(
    schema_version=1,
    reference_zero_threshold_1pm=0.0005,
    quasi_steady_rate_threshold_1pm_s=0.005,
    monotonic_progress_epsilon_1pm_s=0.0001,
    turn_class_curvature_threshold_1pm=0.02,
    direct_handoff_min_peak_curvature_1pm=0.002,
    direct_handoff_max_neutral_duration_s=0.35,
    minimum_phase_duration_s=0.09,
    minimum_phase_samples=2,
    maximum_phase_extension_s=1.0,
    maximum_sample_gap_s=0.15,
    turn_in_crossing_fraction=0.5,
    release_onset_fraction=0.9,
    maximum_raw_phase_spans=65_536,
    maximum_phase_windows=4_096,
    maximum_event_locators=4_096,
    maximum_event_phase_attachments=65_536,
  )


def metric_config() -> BehaviorMetricConfig:
  return BehaviorMetricConfig(
    burst_window_s=0.3,
    chatter_torque_rate_threshold_per_s=0.05,
    turn_in_crossing_fraction=0.5,
    release_crossing_fraction=0.9,
    correction_curvature_threshold_1pm=0.002,
    unused_headroom_threshold=0.1,
    growing_error_epsilon_1pm=0.0001,
    completion_delivered_fraction=0.95,
    minimum_samples=2,
    speed_nodes_mps=(0.0, 5.0, 10.0, 20.0, 30.0),
    maximum_route_windows_per_stratum=4,
  )


def sample(
  index: int,
  curvature: float,
  *,
  intervention: bool = False,
  lateral_active: bool = True,
) -> BehaviorSample:
  route_time_s = index * 0.1
  return BehaviorSample(
    mono_time_ns=1_000_000_000 + round(route_time_s * 1e9),
    route_time_s=route_time_s,
    speed_mps=4.0 + (index % 7) * 0.25,
    scalar_curvature_1pm=curvature,
    desired_curvature_1pm=curvature,
    anchored_curvature_1pm=curvature,
    desired_rack_angle_deg=curvature * 1_000.0,
    desired_rack_rate_deg_s=curvature * 50.0,
    desired_rack_accel_deg_s2=curvature * -20.0,
    measured_curvature_1pm=curvature * 0.9 + (index % 3) * 0.0001,
    measured_rack_angle_deg=curvature * 900.0,
    measured_rack_rate_deg_s=curvature * 40.0,
    measured_rack_accel_deg_s2=curvature * -15.0,
    raw_requested_torque=curvature * 10.0 + (index % 2) * 0.01,
    envelope_applied_torque=curvature * 9.0,
    torque_headroom=0.5,
    actuator_constrained=index % 9 == 0,
    lateral_active=lateral_active,
    inputs_valid=lateral_active,
    steering_pressed=intervention,
    controller_fault=False,
    driver_intervention_onset=intervention,
  )


def shape() -> tuple[BehaviorSample, ...]:
  curvatures = (
    0.0, 0.0, 0.0,
    0.001, 0.004, 0.008, 0.012,
    0.012, 0.012, 0.012,
    0.009, 0.005, 0.002,
    0.0, 0.0, 0.0,
    -0.004, -0.009, -0.012, -0.012,
    -0.007, -0.002, 0.0, 0.0,
  )
  values = tuple(sample(index, curvature) for index, curvature in enumerate(curvatures))
  return values[:18] + (replace(values[18], driver_intervention_onset=True, steering_pressed=True),) + values[19:]


def event(samples: tuple[BehaviorSample, ...]) -> EventLocator:
  return EventLocator(
    event_type="lat.turnStopTurn",
    occurred_mono_time_ns=samples[8].mono_time_ns,
    analysis_window_before_s=0.2,
    analysis_window_after_s=0.1,
    severity="warning",
  )


def test_fixed_records_round_trip_every_sample_field_and_signed_zero(tmp_path: Path) -> None:
  expected = replace(sample(0, -0.0), raw_requested_torque=-0.0)
  with BehaviorSampleScratch(tmp_path) as scratch:
    scratch.append(expected)
    reader = scratch.finish()
    actual = reader[0]
    assert actual == expected
    assert struct.pack("<d", actual.anchored_curvature_1pm) == struct.pack("<d", -0.0)
    assert struct.pack("<d", actual.raw_requested_torque) == struct.pack("<d", -0.0)
    assert tuple(reader) == (expected,)


def test_file_backed_segmentation_and_metrics_match_eager_exactly(tmp_path: Path) -> None:
  samples = shape()
  events = (event(samples),)
  eager = segment_behavior_route("route-file", SOURCE, samples, events, segmentation_config())

  with BehaviorSampleScratch(tmp_path) as scratch:
    scratch.extend(iter(samples))
    reader = scratch.finish()
    bounded = segment_file_backed_behavior_route(
      "route-file", SOURCE, reader, events, segmentation_config(),
    )
    assert bounded.sha256 == eager.sha256
    assert tuple(
      (
        item.window_id,
        item.maneuver_class,
        item.phase,
        item.descriptor.start_sample_index,
        item.descriptor.end_sample_index_exclusive,
        item.observability,
        item.event_locators,
      )
      for item in bounded.windows
    ) == tuple(
      (
        item.window.window_id,
        item.window.maneuver_class,
        item.window.phase,
        item.start_sample_index,
        item.end_sample_index_exclusive,
        item.observability,
        item.window.event_locators,
      )
      for item in eager.windows
    )
    assert bounded.event_coverage == eager.event_coverage
    assert tuple(bounded.unassigned_sample_indices) == eager.unassigned_sample_indices

    bounded_metrics = tuple(score_file_backed_window(item, metric_config()) for item in bounded.windows)
    bounded_scorecard = aggregate_behavior_metrics(bounded_metrics, metric_config())
    eager_scorecard = score_behavior(eager.behavior_windows, metric_config())
    assert bounded_metrics == eager_scorecard.windows
    assert bounded_scorecard == eager_scorecard
    assert bounded_scorecard.to_json() == eager_scorecard.to_json()


@pytest.mark.parametrize(
  "curvatures",
  (
    (0.012, 0.009, 0.005, 0.0001, -0.005, -0.009, -0.012, -0.012),
    (0.0, 0.004, 0.008, 0.012, 0.012, 0.008, 0.003, 0.0),
    (-0.0, -0.003, -0.008, -0.014, -0.014, -0.009, -0.002, -0.0),
  ),
)
def test_file_backed_phase_shapes_preserve_eager_hashes_and_scores(
  tmp_path: Path,
  curvatures: tuple[float, ...],
) -> None:
  samples = tuple(sample(index, curvature) for index, curvature in enumerate(curvatures))
  eager = segment_behavior_route("route-shape", SOURCE, samples, (), segmentation_config())
  with BehaviorSampleScratch(tmp_path) as scratch:
    scratch.extend(samples)
    bounded = segment_file_backed_behavior_route(
      "route-shape", SOURCE, scratch.finish(), (), segmentation_config(),
    )
    assert bounded.sha256 == eager.sha256
    assert tuple(score_file_backed_window(item, metric_config()) for item in bounded.windows) == tuple(
      score_behavior(eager.behavior_windows, metric_config()).windows
    )


def test_randomized_file_backed_segmentation_preserves_eager_authority(
  tmp_path: Path,
) -> None:
  generator = random.Random(0xB1A7)
  curvature_values = (-0.018, -0.008, -0.0001, 0.0, 0.0001, 0.008, 0.018)
  for case_index in range(24):
    values: list[BehaviorSample] = []
    for index in range(80):
      lateral_active = generator.random() >= 0.04
      intervention = lateral_active and generator.random() < 0.025
      values.append(sample(
        index,
        generator.choice(curvature_values),
        intervention=intervention,
        lateral_active=lateral_active,
      ))
    samples = tuple(values)
    route_id = f"route-random-{case_index}"
    eager = segment_behavior_route(route_id, SOURCE, samples, (), segmentation_config())
    with BehaviorSampleScratch(tmp_path) as scratch:
      scratch.extend(samples)
      bounded = segment_file_backed_behavior_route(
        route_id, SOURCE, scratch.finish(), (), segmentation_config(),
      )
      assert bounded.sha256 == eager.sha256
      assert tuple(bounded.unassigned_sample_indices) == eager.unassigned_sample_indices


def test_adversarial_short_span_population_preserves_committed_hash(
  tmp_path: Path,
) -> None:
  with BehaviorSampleScratch(tmp_path) as scratch:
    for index in range(10_000):
      scratch.append(sample(index, 0.005 if index % 2 == 0 else 0.015))
    segmented = segment_file_backed_behavior_route(
      "route-adversarial", SOURCE, scratch.finish(), (), segmentation_config(),
    )

    assert len(segmented.windows) == 1
    assert segmented.windows[0].descriptor.start_sample_index == 0
    assert segmented.windows[0].descriptor.end_sample_index_exclusive == 2
    assert len(segmented.unassigned_sample_indices) == 9_998
    assert segmented.sha256 == "ee0ec76d087d014c5041ee6de038f1561c2680f1c00fbd424b73c78901f2bf99"


def test_many_events_do_not_rescan_every_adversarial_span(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  span_reads = 0
  original_getitem = _SpanScratch.__getitem__

  def counted_getitem(store: _SpanScratch, index: int):
    nonlocal span_reads
    span_reads += 1
    return original_getitem(store, index)

  monkeypatch.setattr(_SpanScratch, "__getitem__", counted_getitem)
  with BehaviorSampleScratch(tmp_path) as scratch:
    for index in range(10_000):
      scratch.append(sample(index, 0.005 if index % 2 == 0 else 0.015))
    reader = scratch.finish()
    events = tuple(
      EventLocator(
        "lat.turnStopTurn",
        reader[index].mono_time_ns,
        0.05,
        0.05,
        "warning",
      )
      for index in range(50, 10_000, 100)
    )
    segmented = segment_file_backed_behavior_route(
      "route-adversarial-events", SOURCE, reader, events, segmentation_config(),
    )

    assert len(segmented.event_coverage) == 100
    assert segmented.sha256 == "9c39b1a605b450c1489c4152be124a98beeaf190f8a5ffbdaf1e83a436303116"
    # One canonical descriptor pass is O(spans). Every event then performs a
    # binary search plus a local overlap scan, not another full-route scan.
    assert span_reads < 25_000


def test_span_scratch_reads_fixed_blocks_instead_of_one_record_syscalls(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  pread_count = 0
  original_pread = behavior_segmentation.os.pread

  def counted_pread(descriptor: int, count: int, offset: int) -> bytes:
    nonlocal pread_count
    pread_count += 1
    return original_pread(descriptor, count, offset)

  monkeypatch.setattr(behavior_segmentation.os, "pread", counted_pread)
  run = _Run(0, 10_000, None, BoundaryReason.PHASE_INCOMPLETE_AT_ROUTE_END)
  with _SpanScratch() as spans:
    for index in range(10_000):
      spans.append(_Span(
        index,
        index + 1,
        ManeuverPhase.TURN_IN if index % 2 == 0 else ManeuverPhase.RELEASE_UNWIND,
        run,
      ))
    spans.finish()

    assert sum(1 for _ in spans) == 10_000
    assert spans[0].start == 0
    assert spans[1].start == 1
    assert spans[-1].start == 9_999
    assert pread_count < 40


def test_scratch_is_removed_after_success_and_error(tmp_path: Path) -> None:
  with BehaviorSampleScratch(tmp_path) as scratch:
    path = scratch.path
    scratch.append(sample(0, 0.0))
    scratch.finish()
    assert path.exists()
  assert not path.exists()

  with pytest.raises(RuntimeError, match="abort"):
    with BehaviorSampleScratch(tmp_path) as scratch:
      aborted_path = scratch.path
      scratch.append(sample(0, 0.0))
      raise RuntimeError("abort")
  assert not aborted_path.exists()


def test_reader_rejects_same_size_scratch_tamper(tmp_path: Path) -> None:
  with BehaviorSampleScratch(tmp_path) as scratch:
    scratch.append(sample(0, 0.0))
    reader = scratch.finish()
    with scratch.path.open("r+b") as output:
      output.seek(-1, 2)
      previous = output.read(1)
      output.seek(-1, 2)
      output.write(bytes((previous[0] ^ 1,)))
      output.flush()
    with pytest.raises(BehaviorScratchError, match="changed"):
      _ = reader[0]


def test_file_backed_segmentation_and_scoring_heap_is_route_length_bounded(tmp_path: Path) -> None:
  def peak(count: int) -> int:
    gc.collect()
    tracemalloc.start()
    try:
      with BehaviorSampleScratch(tmp_path) as scratch:
        for index in range(count):
          scratch.append(sample(index, 0.0))
        reader = scratch.finish()
        segmented = segment_file_backed_behavior_route(
          f"route-{count}", SOURCE, reader, (), segmentation_config(),
        )
        assert len(segmented.windows) == 1
        result = score_file_backed_window(segmented.windows[0], metric_config())
        assert result.clean_sample_count == count
      return tracemalloc.get_traced_memory()[1]
    finally:
      tracemalloc.stop()

  small_peak = peak(500)
  large_peak = peak(5_000)
  assert large_peak <= small_peak + 512 * 1024


def test_adversarial_short_span_heap_is_route_length_bounded(tmp_path: Path) -> None:
  def peak(count: int) -> int:
    gc.collect()
    tracemalloc.start()
    try:
      with BehaviorSampleScratch(tmp_path) as scratch:
        for index in range(count):
          scratch.append(sample(index, 0.005 if index % 2 == 0 else 0.015))
        segmented = segment_file_backed_behavior_route(
          f"route-adversarial-{count}", SOURCE, scratch.finish(), (), segmentation_config(),
        )
        assert len(segmented.windows) == 1
        assert len(segmented.unassigned_sample_indices) == count - 2
        assert len(segmented.sha256) == 64
      return tracemalloc.get_traced_memory()[1]
    finally:
      tracemalloc.stop()

  small_peak = peak(1_000)
  large_peak = peak(10_000)
  assert large_peak <= small_peak + 512 * 1024


def test_versioned_span_work_limits_reject_without_unbounded_output(tmp_path: Path) -> None:
  raw_limited = replace(
    segmentation_config(),
    maximum_raw_phase_spans=64,
    maximum_phase_windows=64,
  )
  with BehaviorSampleScratch(tmp_path) as scratch:
    for index in range(10_000):
      scratch.append(sample(index, 0.005 if index % 2 == 0 else 0.015))
    with pytest.raises(SegmentationResourceError, match="raw phase-span"):
      segment_file_backed_behavior_route(
        "route-raw-budget", SOURCE, scratch.finish(), (), raw_limited,
      )

  window_limited = replace(segmentation_config(), maximum_phase_windows=8)
  with BehaviorSampleScratch(tmp_path) as scratch:
    for index in range(500):
      position = index % 12
      curvature = 0.002 + 0.001 * (position if position <= 6 else 12 - position)
      scratch.append(sample(index, curvature))
    with pytest.raises(SegmentationResourceError, match="phase-window"):
      segment_file_backed_behavior_route(
        "route-window-budget", SOURCE, scratch.finish(), (), window_limited,
      )


def test_event_density_is_indexed_and_attachment_work_fails_closed(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  span_reads = 0
  original_getitem = _SpanScratch.__getitem__

  def counted_getitem(store: _SpanScratch, index: int):
    nonlocal span_reads
    span_reads += 1
    return original_getitem(store, index)

  monkeypatch.setattr(_SpanScratch, "__getitem__", counted_getitem)
  with BehaviorSampleScratch(tmp_path) as scratch:
    for index in range(10_000):
      scratch.append(sample(index, 0.005 if index % 2 == 0 else 0.015))
    reader = scratch.finish()
    events = tuple(
      EventLocator("lat.turnStopTurn", reader[index].mono_time_ns, 5.0, 5.0, "warning")
      for index in range(10, 10_000, 10)
    )
    segmented = segment_file_backed_behavior_route(
      "route-dense-events", SOURCE, reader, events, segmentation_config(),
    )
    assert len(segmented.event_coverage) == 999
    # One span-index construction and one descriptor pass dominate; event
    # count no longer multiplies the raw span population.
    assert span_reads < 23_000

  attachment_limited = replace(
    segmentation_config(),
    maximum_event_phase_attachments=1,
  )
  with BehaviorSampleScratch(tmp_path) as scratch:
    for index in range(120):
      position = index % 12
      curvature = 0.002 + 0.001 * (position if position <= 6 else 12 - position)
      scratch.append(sample(index, curvature))
    reader = scratch.finish()
    locator = EventLocator(
      "lat.turnStopTurn",
      reader[60].mono_time_ns,
      10.0,
      10.0,
      "warning",
    )
    with pytest.raises(SegmentationResourceError, match="attachment"):
      segment_file_backed_behavior_route(
        "route-attachment-budget", SOURCE, reader, (locator,), attachment_limited,
      )

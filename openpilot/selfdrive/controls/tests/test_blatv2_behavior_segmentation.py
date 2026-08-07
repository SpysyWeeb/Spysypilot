from __future__ import annotations

from dataclasses import replace
import unittest

from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import (
  BehaviorSample,
  BehaviorSourceIdentity,
  EventLocator,
  ManeuverPhase,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_segmentation import (
  BoundaryReason,
  EventCoverageStop,
  SegmentationConfig,
  segment_behavior_route,
)


def source_identity() -> BehaviorSourceIdentity:
  return BehaviorSourceIdentity(
    controller_name="segmentation-test",
    controller_artifact_sha256="a" * 64,
    source_openpilot_commit="b" * 40,
    opendbc_commit="c" * 40,
    panda_commit="d" * 40,
    evidence_schema_version=1,
  )


def config() -> SegmentationConfig:
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


def sample(index: int, curvature: float, *, time_s: float | None = None, valid: bool = True,
           intervention: bool = False, steering_pressed: bool = False) -> BehaviorSample:
  route_time = index * 0.1 if time_s is None else time_s
  return BehaviorSample(
    mono_time_ns=1_000_000_000 + round(route_time * 1e9),
    route_time_s=route_time,
    speed_mps=5.0,
    scalar_curvature_1pm=curvature,
    desired_curvature_1pm=curvature,
    anchored_curvature_1pm=curvature,
    desired_rack_angle_deg=curvature * 1_000.0,
    desired_rack_rate_deg_s=0.0,
    desired_rack_accel_deg_s2=0.0,
    measured_curvature_1pm=curvature * 0.9,
    measured_rack_angle_deg=curvature * 900.0,
    measured_rack_rate_deg_s=0.0,
    measured_rack_accel_deg_s2=0.0,
    raw_requested_torque=curvature * 10.0,
    envelope_applied_torque=curvature * 9.0,
    torque_headroom=0.5,
    actuator_constrained=False,
    lateral_active=valid,
    inputs_valid=valid,
    steering_pressed=steering_pressed,
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
  )
  return tuple(sample(index, curvature) for index, curvature in enumerate(curvatures))


class TestBehaviorSegmentation(unittest.TestCase):
  def test_route_wide_discovery_finds_each_ordinary_phase_without_events(self):
    result = segment_behavior_route("route-a", source_identity(), shape(), (), config())
    phases = {window.window.phase for window in result.windows}

    self.assertIn(ManeuverPhase.STRAIGHT_QUASI_STEADY, phases)
    self.assertIn(ManeuverPhase.TURN_IN, phases)
    self.assertIn(ManeuverPhase.HOLD, phases)
    self.assertIn(ManeuverPhase.RELEASE_UNWIND, phases)
    self.assertEqual(result.event_coverage, ())
    timed = {
      window.window.phase: window.observability.metric_crossing_observed
      for window in result.windows
      if window.window.phase in (ManeuverPhase.TURN_IN, ManeuverPhase.RELEASE_UNWIND)
    }
    self.assertTrue(timed[ManeuverPhase.TURN_IN])
    self.assertTrue(timed[ManeuverPhase.RELEASE_UNWIND])

  def test_direct_sign_handoff_rewrites_release_neutral_turn_in_as_one_phase(self):
    values = (0.012, 0.009, 0.005, 0.0001, -0.005, -0.009, -0.012, -0.012)
    samples = tuple(sample(index, value) for index, value in enumerate(values))
    result = segment_behavior_route("route-handoff", source_identity(), samples, (), config())
    handoffs = [window for window in result.windows if window.window.phase is ManeuverPhase.DIRECT_HANDOFF]

    self.assertEqual(len(handoffs), 1)
    self.assertGreater(handoffs[0].window.samples[0].anchored_curvature_1pm, 0.0)
    self.assertLess(handoffs[0].window.samples[-1].anchored_curvature_1pm, 0.0)

  def test_event_context_extends_past_committed_after_window_to_physical_completion(self):
    samples = shape()
    event = EventLocator(
      event_type="lat.turnStopTurn",
      occurred_mono_time_ns=samples[7].mono_time_ns,
      analysis_window_before_s=0.2,
      analysis_window_after_s=0.1,
      severity="warning",
    )
    result = segment_behavior_route("route-event", source_identity(), samples, (event,), config())
    coverage = result.event_coverage[0]

    self.assertTrue(coverage.extended_beyond_nominal_end)
    self.assertEqual(coverage.stop_reason, EventCoverageStop.PHASE_COMPLETE)
    self.assertFalse(coverage.phase_incomplete_at_boundary)
    self.assertGreater(coverage.physical_end_mono_time_ns, coverage.nominal_end_mono_time_ns)
    self.assertTrue(any(event in window.window.event_locators for window in result.windows))

  def test_event_extension_stops_at_next_event_window(self):
    samples = shape()
    first = EventLocator("lat.turnStopTurn", samples[6].mono_time_ns, 0.1, 0.05, "warning")
    second = EventLocator("lat.committedHandoffHarshness", samples[10].mono_time_ns, 0.0, 0.1, "warning")
    result = segment_behavior_route("route-events", source_identity(), samples, (first, second), config())
    coverage = result.event_coverage[0]

    self.assertEqual(coverage.stop_reason, EventCoverageStop.NEXT_EVENT_WINDOW)
    self.assertEqual(coverage.physical_end_mono_time_ns, second.occurred_mono_time_ns)
    self.assertTrue(coverage.phase_incomplete_at_boundary)

  def test_driver_contact_censors_current_and_following_maneuver_phases(self):
    samples = list(shape())
    samples[8] = replace(samples[8], driver_intervention_onset=True, steering_pressed=True)
    result = segment_behavior_route("route-driver", source_identity(), samples, (), config())
    onset_window = next(
      window for window in result.windows
      if window.observability.driver_censor_mono_time_ns == samples[8].mono_time_ns
    )

    self.assertEqual(onset_window.window.intervention_mono_time_ns, samples[8].mono_time_ns)
    self.assertLess(len(onset_window.window.clean_pre_intervention_samples), len(onset_window.window.samples))
    self.assertIn(BoundaryReason.DRIVER_INTERVENTION_CENSOR, onset_window.observability.reasons)
    self.assertFalse(any(
      window.start_sample_index > 8 and window.window.phase is ManeuverPhase.RELEASE_UNWIND
      for window in result.windows
    ))
    self.assertFalse(onset_window.window.intervention_is_quality_vote)

  def test_straight_motion_does_not_clear_censor_but_reengagement_does(self):
    values = (
      0.0, 0.002, 0.006, 0.010, 0.010, 0.006, 0.002,
      0.0, 0.0, 0.0, 0.0, 0.0,  # long straight, still active
      0.0,  # explicit inactive lifecycle boundary
      0.0, 0.003, 0.007, 0.011, 0.011,
    )
    samples = [sample(index, value) for index, value in enumerate(values)]
    samples[4] = replace(
      samples[4],
      driver_intervention_onset=True,
      steering_pressed=True,
    )
    samples[12] = replace(
      samples[12],
      lateral_active=False,
      inputs_valid=False,
    )
    result = segment_behavior_route(
      "route-episode-censor",
      source_identity(),
      samples,
      (),
      config(),
    )

    self.assertFalse(any(
      window.start_sample_index > 4 and window.end_sample_index_exclusive <= 12
      for window in result.windows
    ))
    self.assertTrue(any(
      window.start_sample_index >= 13
      and window.observability.driver_censor_mono_time_ns is None
      for window in result.windows
    ))

  def test_gap_and_invalid_frame_split_runs_and_preserve_boundary_reason(self):
    samples = (
      sample(0, 0.0, time_s=0.0),
      sample(1, 0.004, time_s=0.1),
      sample(2, 0.008, time_s=0.2),
      sample(3, 0.009, time_s=0.5),
      sample(4, 0.011, time_s=0.6),
      sample(5, 0.012, time_s=0.7, valid=False),
      sample(6, 0.010, time_s=0.8),
      sample(7, 0.008, time_s=0.9),
    )
    result = segment_behavior_route("route-gap", source_identity(), samples, (), config())

    self.assertIn(5, result.unassigned_sample_indices)
    self.assertTrue(any(
      BoundaryReason.PHASE_ONSET_PRECEDES_AVAILABLE_EVIDENCE in window.observability.reasons
      for window in result.windows
      if window.start_sample_index in (3, 6)
    ))

  def test_ids_order_hash_and_phase_ownership_are_deterministic(self):
    first = segment_behavior_route("route-stable", source_identity(), shape(), (), config())
    second = segment_behavior_route("route-stable", source_identity(), tuple(shape()), (), config())

    self.assertEqual(first.to_json(), second.to_json())
    self.assertEqual(first.sha256, second.sha256)
    ids = tuple(window.window.window_id for window in first.windows)
    self.assertEqual(ids, tuple(sorted(ids)))
    owned = [
      index
      for window in first.windows
      for index in range(window.start_sample_index, window.end_sample_index_exclusive)
    ]
    self.assertEqual(len(owned), len(set(owned)))

  def test_observability_distinguishes_missing_onset_from_incomplete_end(self):
    values = (0.010, 0.010, 0.009, 0.007, 0.005, 0.003)
    samples = tuple(sample(index, value) for index, value in enumerate(values))
    result = segment_behavior_route("route-boundary", source_identity(), samples, (), config())

    first = result.windows[0]
    last = result.windows[-1]
    self.assertFalse(first.observability.onset_observed)
    self.assertIn(BoundaryReason.PHASE_ONSET_PRECEDES_AVAILABLE_EVIDENCE, first.observability.reasons)
    self.assertFalse(last.observability.completion_observed)
    self.assertIn(BoundaryReason.PHASE_INCOMPLETE_AT_ROUTE_END, last.observability.reasons)

  def test_invalid_order_and_invalid_config_fail_closed(self):
    ordered = shape()
    with self.assertRaisesRegex(ValueError, "strictly ordered"):
      segment_behavior_route("route-order", source_identity(), (ordered[1], ordered[0]), (), config())
    with self.assertRaisesRegex(ValueError, "canonical timestamp order"):
      late = EventLocator("late", ordered[8].mono_time_ns, 0.0, 0.0, "info")
      early = EventLocator("early", ordered[2].mono_time_ns, 0.0, 0.0, "info")
      segment_behavior_route("route-events", source_identity(), ordered, (late, early), config())
    with self.assertRaisesRegex(ValueError, "positive"):
      replace(config(), maximum_sample_gap_s=0.0)


if __name__ == "__main__":
  unittest.main()

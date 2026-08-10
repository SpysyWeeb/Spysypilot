from types import SimpleNamespace

import pytest

from openpilot.selfdrive.spysypilot.driving_eventd import (
  AcceptedEvent,
  DrivingEventPlatform,
  EventSubmitter,
  EventCandidate,
  EventRecorder,
  GROUP_KEY_PRUNE_NS,
  GROUP_MAX_LIFETIME_NS,
  ROAD_CONFIDENCE_FLOOR,
  ROAD_SUBSTANTIAL_CONFIDENCE_PENALTY,
  build_message,
  lateral_candidate,
  lateral_sample,
  live_pose_sample,
  log_retry_warning,
  longitudinal_sample,
  manual_candidate,
  rolling_lead_candidate,
  rolling_lead_sample,
  stop_jolt_candidate,
)
from openpilot.selfdrive.spysypilot.lat_event_detector import (
  DETECTOR_VERSION as LAT_DETECTOR_VERSION,
  LateralDetection,
  LateralEvidence,
  LateralSample,
  SteeringRateFilter,
)
from openpilot.selfdrive.spysypilot.long_event_detector import LaunchSample, LongEvent
from openpilot.selfdrive.spysypilot.rolling_lead_response_detector import (
  RollingLeadOnsetSnapshot,
  RollingLeadResponseEvent,
  RollingLeadSample,
)
from openpilot.selfdrive.spysypilot.stop_jolt_detector import StopJoltCarSample, StopJoltEvent, StopJoltImuSample


def launch_sample(t: float = 1.0) -> LaunchSample:
  return LaunchSample(t, True, True, 0.0, True, True, 5.0, 0.0, 0.0, 1, True, True, True, 0.0, False, 0.0)


class FixedDetector:
  def __init__(self, result=None, error: Exception | None = None):
    self.result = result
    self.error = error
    self.calls = 0

  def update(self, _sample):
    self.calls += 1
    if self.error is not None:
      raise self.error
    return self.result


class FixedStopDetector:
  def __init__(self, result=None, car_error: Exception | None = None, imu_error: Exception | None = None):
    self.result = result
    self.car_error = car_error
    self.imu_error = imu_error
    self.car_calls = 0
    self.imu_calls = 0

  def update_car(self, _sample):
    self.car_calls += 1
    if self.car_error is not None:
      raise self.car_error
    return self.result

  def update_imu(self, _sample):
    self.imu_calls += 1
    if self.imu_error is not None:
      raise self.imu_error


def stop_car_sample(t: float = 2.0) -> StopJoltCarSample:
  return StopJoltCarSample(
    t, True, True, True, 0.0, -1.0, True, False, False, True,
    -0.5, -0.6, -0.7, "stopping", True, 4.0, 0.0, True,
  )


def stop_event() -> StopJoltEvent:
  return StopJoltEvent(
    classification="grabAndRebound",
    severity="warning",
    confidence=0.91,
    reason="bad landing",
    attribution="mixed",
    episode_start_mono_time=1.0,
    standstill_mono_time=2.0,
    peak_jolt_mono_time=1.8,
    detection_mono_time=2.45,
    imu_jerk=-3.5,
    a_ego_jerk=-3.0,
    imu_accel_before=-0.2,
    imu_accel_after=-1.0,
    a_ego_accel_before=-0.3,
    a_ego_accel_after=-1.1,
    accel_change=-0.8,
    accel_at_02_mps=-0.9,
    v_ego_at_peak=0.15,
    plan_a_target=-0.5,
    requested_accel=-0.6,
    applied_accel=-0.7,
    plan_accel_change=0.1,
    requested_accel_change=0.2,
    applied_accel_change=0.3,
    should_stop_before=True,
    should_stop_at_peak=True,
    should_stop_after=True,
    long_control_state_before="stopping",
    long_control_state_at_peak="stopping",
    long_control_state_after="pid",
    lead_present=True,
    d_rel=4.0,
    v_lead_k=0.0,
    brake_pressed=False,
    gas_pressed=False,
    brake_hold_active=True,
    radar_valid=True,
    imu_valid=True,
    road_confounded=False,
  )


def rolling_sample(t: float = 3.0) -> RollingLeadSample:
  return RollingLeadSample(
    t, True, 8.0, 0.0, False, False, True, True, True, 7,
    20.0, 10.0, 10.0, 2.0, True, -0.5, False, True, -0.5,
  )


def rolling_event() -> RollingLeadResponseEvent:
  onset = RollingLeadOnsetSnapshot(
    "leadCommit", 3.0, 20.0, 10.0, 10.0, 2.0, 8.0, 0.0, -0.5, -0.5, False, False, False,
  )
  return RollingLeadResponseEvent(
    "late_rolling_lead_response_planner", "planner", "late planner", 2, 0.95,
    3.0, 5.5, 3.0, 4.0, 4.1, None, 1.0, 1.1, None,
    8.0, 13.0, 8.0, 9.0, 4.0, 20.0, 25.0, 5.0,
    0.0, 2.5, -0.5, 1.0, -0.5, 1.0, 0.0, 0.4,
    7, False, False, (onset,), 3.0,
  )


def test_unique_ids_and_group_correlation():
  values = iter(("group", "event-1", "event-2", "next-group", "event-3"))
  recorder = EventRecorder(values.__next__, group_window_ns=100)
  first = recorder.accept(manual_candidate(1000))
  second = recorder.accept(manual_candidate(1050))
  third = recorder.accept(manual_candidate(1200))
  assert first.event_id != second.event_id
  assert first.group_id == second.group_id == "group"
  assert third.group_id == "next-group"


def test_semantic_episode_group_survives_generic_time_window():
  values = iter(("episode-group", "event-1", "event-2"))
  recorder = EventRecorder(values.__next__, group_window_ns=100)
  first_candidate = EventCandidate(
    1_000, "lateral", "automatic", "lat.stallRelease", "test", 3,
    "warning", 0.8, "first", episode_key="lat:episode",
  )
  second_candidate = EventCandidate(
    10_000, "lateral", "automatic", "lat.handoffMismatch", "test", 3,
    "warning", 0.9, "second", episode_key="lat:episode",
  )
  first = recorder.accept(first_candidate)
  second = recorder.accept(second_candidate)
  assert first.group_id == second.group_id == "episode-group"


def test_distinct_keys_ten_seconds_apart_get_different_groups():
  # A3: unlike an exact-key match, two genuinely different physical
  # maneuvers (distinct episode keys) more than the keyless chain window
  # apart must NOT share a group.
  recorder = EventRecorder()
  first = recorder.accept(EventCandidate(
    0, "lateral", "automatic", "lat.stallRelease", "test", 6,
    "warning", 0.8, "first", episode_key="lat:first",
  ))
  second = recorder.accept(EventCandidate(
    10_000_000_000, "lateral", "automatic", "lat.stallRelease", "test", 6,
    "warning", 0.8, "second", episode_key="lat:second",
  ))
  assert second.group_id != first.group_id


def test_group_key_last_seen_refresh_is_monotone_under_out_of_order_occurrence():
  # A3: candidates within one update() call are only sorted by occurred
  # time; across separate accept() calls, detection lag differs per
  # detector, so a later accept() can carry an EARLIER occurred_mono_time.
  # The key's last_seen bookkeeping must never rewind, or a still-related
  # key could be pruned early.
  recorder = EventRecorder()
  key = "lat:episode"
  first = recorder.accept(EventCandidate(
    20_000_000_000, "lateral", "automatic", "lat.stallRelease", "test", 6,
    "warning", 0.8, "first", episode_key=key,
  ))
  second = recorder.accept(EventCandidate(
    5_000_000_000, "lateral", "automatic", "lat.handoffMismatch", "test", 6,
    "warning", 0.8, "second", episode_key=key,
  ))
  assert second.group_id == first.group_id
  # An unrelated key far enough away triggers a memory-only prune sweep
  # (GROUP_KEY_PRUNE_NS after the correct, monotone last_seen of 20s): if
  # last_seen had wrongly rewound to 5s here, this sweep would evict `key`.
  assert 40_000_000_000 - 20_000_000_000 <= GROUP_KEY_PRUNE_NS
  assert 40_000_000_000 - 5_000_000_000 > GROUP_KEY_PRUNE_NS
  unrelated = recorder.accept(EventCandidate(
    40_000_000_000, "lateral", "automatic", "lat.stallRelease", "test", 6,
    "warning", 0.8, "unrelated", episode_key="lat:other",
  ))
  assert unrelated.group_id != first.group_id
  third = recorder.accept(EventCandidate(
    40_100_000_000, "lateral", "automatic", "lat.stallRelease", "test", 6,
    "warning", 0.8, "third", episode_key=key,
  ))
  assert third.group_id == first.group_id


def test_brand_new_key_far_in_the_past_mints_a_new_group():
  # Fix 1 (major, empirically reproduced): the brand-new-key chain check
  # compared occurred_mono_time - group_last_time <= group_window_ns with NO
  # lower bound. A candidate whose occurred_mono_time is far EARLIER than the
  # running group's last-seen time produces a large NEGATIVE delta, which is
  # trivially <= any positive window -- so it always "passed" and incorrectly
  # joined the running group, even when the two candidates are a genuinely
  # separate maneuver 50s apart. Bug reproduced: under the pre-fix one-sided
  # comparison, `second.group_id` would wrongly equal `first.group_id` here.
  recorder = EventRecorder()
  first = recorder.accept(EventCandidate(
    200_000_000_000, "lateral", "automatic", "lat.stallRelease", "test", 6,
    "warning", 0.8, "first", episode_key="lat:first",
  ))
  second = recorder.accept(EventCandidate(
    150_000_000_000, "lateral", "automatic", "lat.stallRelease", "test", 6,
    "warning", 0.8, "second", episode_key="lat:second",
  ))
  assert second.group_id != first.group_id


def test_keyless_candidate_far_in_the_past_mints_a_new_group():
  # Fix 1, keyless path: the same one-sided comparison bug (as an inverted OR
  # condition) let a far-in-the-past keyless (manual) candidate silently join
  # the running group instead of minting a new one.
  recorder = EventRecorder()
  first = recorder.accept(manual_candidate(200_000_000_000))
  second = recorder.accept(manual_candidate(150_000_000_000))
  assert second.group_id != first.group_id


def test_group_hard_lifetime_splits_a_pathological_same_key_chain():
  # A3: exact-key reuse alone would chain a same-key stream forever; the
  # hard GROUP_MAX_LIFETIME_NS cap must split it and re-bind the key.
  recorder = EventRecorder()
  key = "lat:episode"
  first = recorder.accept(EventCandidate(
    0, "lateral", "automatic", "lat.stallRelease", "test", 6,
    "warning", 0.8, "first", episode_key=key,
  ))
  second = recorder.accept(EventCandidate(
    GROUP_MAX_LIFETIME_NS - 1_000_000_000, "lateral", "automatic", "lat.stallRelease", "test", 6,
    "warning", 0.8, "second", episode_key=key,
  ))
  assert second.group_id == first.group_id
  third = recorder.accept(EventCandidate(
    GROUP_MAX_LIFETIME_NS + 1_000_000_000, "lateral", "automatic", "lat.stallRelease", "test", 6,
    "warning", 0.8, "third", episode_key=key,
  ))
  assert third.group_id != first.group_id
  # The key is now bound to the new group; a later same-key candidate still
  # within THAT group's own hard lifetime reuses it.
  fourth = recorder.accept(EventCandidate(
    GROUP_MAX_LIFETIME_NS + 2_000_000_000, "lateral", "automatic", "lat.stallRelease", "test", 6,
    "warning", 0.8, "fourth", episode_key=key,
  ))
  assert fourth.group_id == third.group_id


def test_retrying_an_accepted_event_retains_ids():
  accepted = AcceptedEvent("event", "group", manual_candidate(123))
  first = build_message(accepted).drivingEvent
  second = build_message(accepted).drivingEvent
  assert first.eventId == second.eventId == "event"
  assert first.groupId == second.groupId == "group"


def test_manual_lateral_and_longitudinal_coexist():
  lateral = FixedDetector(LateralDetection("centerOvershoot", "warning", 0.9, "lateral reason"))
  longitudinal = FixedDetector(LongEvent(
    "late_lead_launch_vehicle", "long reason", 2, 0.95,
  ))
  ids = iter(("group", "lat", "long", "manual"))
  platform = DrivingEventPlatform(
    recorder=EventRecorder(ids.__next__),
    lateral_detector=lateral,
    longitudinal_detector=longitudinal,
  )
  events = platform.update(LateralSample(1.0), launch_sample(), manual_pressed=True, manual_time_ns=1_000_000_000)
  assert {event.candidate.domain for event in events} == {"manual", "lateral", "longitudinal"}
  assert {event.group_id for event in events} == {"group"}
  assert next(event for event in events if event.candidate.domain == "longitudinal").candidate.attribution == "mixed"


def test_simultaneous_lateral_detections_are_all_retained():
  evidence = LateralEvidence(
    0.0, 1.0, 0.0, 0.0, False, 0.0, 0, 0.0, "lat:episode", 3.0, 2.0,
  )
  lateral = FixedDetector((
    LateralDetection(
      "torqueUnderDelivery", "critical", 0.9, "authority unused", evidence,
      occurred_mono_time=0.0, detected_mono_time=1.0,
    ),
    LateralDetection(
      "driverTakeover", "critical", 0.95, "driver override", evidence,
      occurred_mono_time=0.5, detected_mono_time=1.0,
    ),
  ))
  platform = DrivingEventPlatform(
    recorder=EventRecorder(iter(("group", "under", "takeover")).__next__),
    lateral_detector=lateral,
  )
  events = platform.update(lateral_sample=LateralSample(1.0))
  assert [event.candidate.event_type for event in events] == [
    "lat.torqueUnderDelivery", "lat.driverTakeover",
  ]
  assert {event.group_id for event in events} == {"group"}


def test_detector_exception_isolation():
  errors = []
  platform = DrivingEventPlatform(
    recorder=EventRecorder(iter(("group", "long")).__next__),
    lateral_detector=FixedDetector(error=RuntimeError("broken")),
    longitudinal_detector=FixedDetector(LongEvent(
      "late_lead_launch_controller", "long reason", 1, 0.95,
    )),
    on_error=errors.append,
  )
  events = platform.update(LateralSample(1.0), launch_sample())
  assert errors == ["lateral"]
  assert [event.candidate.domain for event in events] == ["longitudinal"]


def test_lead_launch_and_stop_jolt_failure_isolation():
  errors = []
  stop_detector = FixedStopDetector(stop_event())
  platform = DrivingEventPlatform(
    recorder=EventRecorder(iter(("group", "stop")).__next__),
    lateral_detector=FixedDetector(),
    longitudinal_detector=FixedDetector(error=RuntimeError("lead broken")),
    stop_jolt_detector=stop_detector,
    on_error=errors.append,
  )
  events = platform.update(
    longitudinal_sample=launch_sample(),
    stop_car_sample=stop_car_sample(),
    stop_imu_sample=StopJoltImuSample(2.0, -1.0, True),
  )
  assert errors == ["longitudinal"]
  assert stop_detector.imu_calls == 1
  assert [event.candidate.event_type for event in events] == ["long.stopJolt"]

  errors.clear()
  platform = DrivingEventPlatform(
    recorder=EventRecorder(iter(("group", "lead")).__next__),
    lateral_detector=FixedDetector(),
    longitudinal_detector=FixedDetector(LongEvent(
      "late_lead_launch_controller", "long reason", 1, 0.95,
    )),
    stop_jolt_detector=FixedStopDetector(car_error=RuntimeError("stop broken")),
    on_error=errors.append,
  )
  events = platform.update(longitudinal_sample=launch_sample(), stop_car_sample=stop_car_sample())
  assert errors == ["stop_jolt"]
  assert [event.candidate.event_type for event in events] == ["long.lateLeadLaunchController"]


def test_rolling_lead_and_standstill_launch_failure_isolation():
  errors = []
  platform = DrivingEventPlatform(
    recorder=EventRecorder(iter(("group", "rolling")).__next__),
    longitudinal_detector=FixedDetector(error=RuntimeError("standstill detector broken")),
    rolling_lead_detector=FixedDetector(rolling_event()),
    on_error=errors.append,
  )
  events = platform.update(longitudinal_sample=launch_sample(), rolling_lead_sample=rolling_sample())
  assert errors == ["longitudinal"]
  assert [event.candidate.event_type for event in events] == ["long.lateRollingLeadResponsePlanner"]

  errors.clear()
  platform = DrivingEventPlatform(
    recorder=EventRecorder(iter(("group", "launch")).__next__),
    longitudinal_detector=FixedDetector(LongEvent(
      "late_lead_launch_controller", "long reason", 1, 0.95,
    )),
    rolling_lead_detector=FixedDetector(error=RuntimeError("rolling detector broken")),
    on_error=errors.append,
  )
  events = platform.update(longitudinal_sample=launch_sample(), rolling_lead_sample=rolling_sample())
  assert errors == ["rolling_lead_response"]
  assert [event.candidate.event_type for event in events] == ["long.lateLeadLaunchController"]


def test_manual_input_does_not_suppress_or_replace_detector_output():
  lateral = FixedDetector(LateralDetection("centerOvershoot", "warning", 0.9, "reason"))
  platform = DrivingEventPlatform(
    recorder=EventRecorder(iter(("group", "lat", "manual")).__next__),
    lateral_detector=lateral,
    longitudinal_detector=FixedDetector(),
  )
  events = platform.update(LateralSample(1.0), launch_sample(), manual_pressed=True, manual_time_ns=1_000_000_000)
  assert lateral.calls == 1
  assert len(events) == 2


def test_detectors_run_only_for_their_updated_sample():
  lateral = FixedDetector()
  longitudinal = FixedDetector()
  platform = DrivingEventPlatform(lateral_detector=lateral, longitudinal_detector=longitudinal)

  platform.update(lateral_sample=LateralSample(1.0))
  assert lateral.calls == 1
  assert longitudinal.calls == 0

  platform.update(longitudinal_sample=launch_sample())
  assert lateral.calls == 1
  assert longitudinal.calls == 1

  platform.update(manual_pressed=True, manual_time_ns=123)
  assert lateral.calls == 1
  assert longitudinal.calls == 1


def test_submitter_retries_stable_id_until_acknowledged():
  class FakePubMaster:
    def __init__(self):
      self.messages = []

    def send(self, service, msg):
      self.messages.append((service, msg.drivingEvent.eventId, msg.drivingEvent.groupId))

  pm = FakePubMaster()
  submitter = EventSubmitter(pm, retry_interval_ns=100)
  accepted = AcceptedEvent("event", "group", manual_candidate(123))
  submitter.submit(accepted, now_ns=1_000)
  assert submitter.retry_due(now_ns=1_099) == []
  assert submitter.retry_due(now_ns=1_100) == ["event"]
  assert pm.messages == [
    ("drivingEvent", "event", "group"),
    ("drivingEvent", "event", "group"),
  ]
  assert submitter.pending["event"].attempts == 2
  assert submitter.acknowledge("event")
  assert submitter.retry_due(now_ns=2_000) == []


def test_retry_warning_coerces_dynamic_id_at_log_boundary(monkeypatch):
  class DynamicId:
    def __str__(self):
      return "event-from-dynamic-struct"

  messages = []
  monkeypatch.setattr(
    log_retry_warning.__globals__["cloudlog"], "warning", messages.append,
  )
  log_retry_warning(DynamicId())
  assert messages == [
    "driving_eventd: retrying unacknowledged event "
    "event-from-dynamic-struct",
  ]


def test_driving_eventd_is_restartable_after_passive_logger_failure():
  from openpilot.system.manager.process_config import managed_processes

  assert managed_processes["driving_eventd"].restart_if_crash


def test_submit_failure_remains_pending_for_retry():
  class FlakyPubMaster:
    def __init__(self):
      self.calls = 0

    def send(self, _service, _msg):
      self.calls += 1
      if self.calls == 1:
        raise RuntimeError("logger unavailable")

  pm = FlakyPubMaster()
  submitter = EventSubmitter(pm, retry_interval_ns=100)
  submitter.submit(AcceptedEvent("event", "group", manual_candidate(123)), now_ns=1_000)
  assert "event" in submitter.pending
  assert submitter.retry_due(now_ns=1_100) == ["event"]
  assert pm.calls == 2


def test_generic_domain_message_has_typed_empty_payload():
  candidate = EventCandidate(
    occurred_mono_time=123,
    domain="system",
    source="automatic",
    event_type="system.test",
    detector="test",
    detector_version=1,
    severity="info",
    confidence=1.0,
    reason="test",
  )
  msg = build_message(AcceptedEvent("event", "group", candidate))
  assert msg.drivingEvent.domain == "system"
  assert msg.drivingEvent.payload.which() == "none"


def test_stop_jolt_schema_round_trip_preserves_peak_and_detection_times():
  candidate = stop_jolt_candidate(stop_event())
  msg = build_message(AcceptedEvent("stop-event", "stop-group", candidate)).as_reader().drivingEvent
  assert msg.eventType == "long.stopJolt"
  assert msg.detector == "smoothStopJoltDetector"
  assert msg.detectorVersion == 1
  assert msg.occurredMonoTime == 1_800_000_000
  assert msg.detectedMonoTime == 2_450_000_000
  assert msg.episodeKey == "stop:1000000000"
  assert msg.analysisWindowBeforeS == 5.0
  assert msg.analysisWindowAfterS == 2.0
  assert msg.payload.which() == "stopJolt"
  payload = msg.payload.stopJolt
  assert payload.peakJoltMonoTime == msg.occurredMonoTime
  assert payload.detectionMonoTime == msg.detectedMonoTime
  assert payload.classification == "grabAndRebound"
  assert payload.imuJerk == pytest.approx(-3.5)
  assert payload.aEgoJerk == pytest.approx(-3.0)
  assert payload.shouldStopBefore and payload.shouldStopAtPeak and payload.shouldStopAfter
  assert payload.longControlStateAfter == "pid"


def test_rolling_lead_schema_preserves_commit_and_later_detection():
  candidate = rolling_lead_candidate(rolling_event())
  msg = build_message(AcceptedEvent("rolling", "group", candidate)).as_reader().drivingEvent
  assert msg.eventType == "long.lateRollingLeadResponsePlanner"
  assert msg.detector == "rollingLeadResponseDetector"
  assert msg.detectorVersion == 1
  assert msg.occurredMonoTime == 3_000_000_000
  assert msg.detectedMonoTime == 5_500_000_000
  assert msg.analysisWindowBeforeS == 5.0
  assert msg.analysisWindowAfterS == 8.0
  assert msg.payload.which() == "rollingLeadResponse"
  payload = msg.payload.rollingLeadResponse
  assert payload.leadCommitMonoTime == msg.occurredMonoTime
  assert payload.detectedMonoTime == msg.detectedMonoTime
  assert payload.plannerResponsePresent
  assert payload.controllerResponsePresent
  assert not payload.egoResponsePresent
  assert payload.radarTrackId == 7
  assert payload.onsets[0].kind == "leadCommit"
  assert payload.onsets[0].monoTime == msg.occurredMonoTime


def test_lateral_candidate_uses_physical_crossing_and_later_confirmation_times():
  evidence = LateralEvidence(
    1.0, 2.0, 0.0, 0.0, False, 0.0, 0, 0.5, "lat:episode", 2.0, 2.0,
    center_crossing_mono_time=1.25,
    committed_reversal=True,
  )
  detection = LateralDetection(
    "committedHandoffHarshness", "warning", 0.9, "harsh",
    evidence, occurred_mono_time=1.25, detected_mono_time=1.50,
  )
  candidate = lateral_candidate(LateralSample(1.5), detection)
  assert candidate.occurred_mono_time == 1_250_000_000
  assert candidate.detected_mono_time == 1_500_000_000
  assert candidate.event_type == "lat.committedHandoffHarshness"


def test_driver_takeover_is_driver_ground_truth_not_driver_or_road_confounded():
  evidence = LateralEvidence(
    1.0, 2.0, 1.0, 250.0, True, 1.0, 3, 0.5, "lat:takeover", 3.0, 2.0,
    driver_interaction="confirmedSteeringPressed",
    road_interaction="substantial",
    demanded_curvature=0.12,
    delivered_curvature_fraction=0.65,
    torque_headroom=0.384,
    signed_tracking_deficit=0.42,
    takeover_confirmation_duration_s=0.30,
  )
  detection = LateralDetection(
    "driverTakeover", "critical", 0.95, "driver override", evidence,
    occurred_mono_time=1.5, detected_mono_time=1.8,
  )
  candidate = lateral_candidate(
    LateralSample(1.8, steering_pressed=True, road_confounded=True),
    detection,
  )
  assert candidate.event_type == "lat.driverTakeover"
  assert candidate.attribution == "driver"
  assert not candidate.driver_confounded
  assert candidate.road_confounded
  assert candidate.confidence == 0.95

  msg = build_message(AcceptedEvent("takeover", "group", candidate)).as_reader().drivingEvent
  assert msg.version == 4
  assert msg.attribution == "driver"
  assert not msg.driverConfounded
  assert msg.payload.lateral.demandedCurvature == pytest.approx(0.12)
  assert msg.payload.lateral.deliveredCurvatureFraction == pytest.approx(0.65)
  assert msg.payload.lateral.torqueHeadroom == pytest.approx(0.384)
  assert msg.payload.lateral.signedTrackingDeficit == pytest.approx(0.42)
  assert msg.payload.lateral.takeoverConfirmationDurationS == pytest.approx(0.30)


def test_road_confidence_penalty_applies_to_non_rescued_substantial_road_events():
  evidence = LateralEvidence(
    1.0, 2.0, 0.0, 0.0, False, 0.0, 0, 0.5, "lat:episode", 2.0, 2.0,
    road_interaction="substantial",
  )
  detection = LateralDetection(
    "centerOvershoot", "warning", 0.90, "road bump", evidence,
    occurred_mono_time=1.0, detected_mono_time=1.0,
  )
  candidate = lateral_candidate(LateralSample(1.0), detection)
  # A7: confidence_out == max(confidence_in - PENALTY, FLOOR), an honest
  # invariant — not a claim that the 0.50 floor is reached in practice.
  assert candidate.confidence == pytest.approx(
    max(0.90 - ROAD_SUBSTANTIAL_CONFIDENCE_PENALTY, ROAD_CONFIDENCE_FLOOR),
  )
  assert candidate.confidence == pytest.approx(0.85)


def test_road_confidence_penalty_never_applies_to_intervention_backed_or_assisted_unwind():
  rescued_by_causation = LateralEvidence(
    1.0, 2.0, 0.0, 0.0, False, 0.0, 0, 0.5, "lat:episode", 2.0, 2.0,
    road_interaction="substantial", driver_causation="interventionBacked",
  )
  detection = LateralDetection(
    "unwindProgressDeficit", "info", 0.95, "assisted", rescued_by_causation,
    occurred_mono_time=1.0, detected_mono_time=1.0,
  )
  candidate = lateral_candidate(LateralSample(1.0), detection)
  assert candidate.confidence == 0.95

  # driver_assisted_unwind alone (independent of driver_causation) also
  # exempts the event from the penalty.
  rescued_by_flag = LateralEvidence(
    1.0, 2.0, 0.0, 0.0, False, 0.0, 0, 0.5, "lat:episode", 2.0, 2.0,
    road_interaction="substantial", driver_assisted_unwind=True,
  )
  flag_detection = LateralDetection(
    "unwindProgressDeficit", "info", 0.95, "assisted", rescued_by_flag,
    occurred_mono_time=1.0, detected_mono_time=1.0,
  )
  flag_candidate = lateral_candidate(LateralSample(1.0), flag_detection)
  assert flag_candidate.confidence == 0.95


def test_road_confidence_penalty_not_applied_when_road_interaction_not_substantial():
  evidence = LateralEvidence(
    1.0, 2.0, 0.0, 0.0, False, 0.0, 0, 0.5, "lat:episode", 2.0, 2.0,
  )
  detection = LateralDetection(
    "centerOvershoot", "warning", 0.90, "clean", evidence,
    occurred_mono_time=1.0, detected_mono_time=1.0,
  )
  candidate = lateral_candidate(LateralSample(1.0), detection)
  assert candidate.confidence == 0.90


def test_road_confidence_floor_is_an_invariant_not_a_reachable_branch():
  # The lowest real emit-site confidence literal is 0.80, and
  # 0.80 - 0.05 = 0.75 never reaches the 0.50 floor in practice; assert the
  # formula directly rather than claiming the floor engages.
  for confidence_in in (0.80, 0.90, 0.95, 1.0):
    evidence = LateralEvidence(
      1.0, 2.0, 0.0, 0.0, False, 0.0, 0, 0.5, "lat:episode", 2.0, 2.0,
      road_interaction="substantial",
    )
    detection = LateralDetection(
      "centerOvershoot", "warning", confidence_in, "road bump", evidence,
      occurred_mono_time=1.0, detected_mono_time=1.0,
    )
    candidate = lateral_candidate(LateralSample(1.0), detection)
    assert candidate.confidence == pytest.approx(
      max(confidence_in - ROAD_SUBSTANTIAL_CONFIDENCE_PENALTY, ROAD_CONFIDENCE_FLOOR),
    )
    assert candidate.confidence > ROAD_CONFIDENCE_FLOOR


def test_lateral_build_message_wires_v6_present_bit_and_causation_fields():
  # v6 monoTime+Present pairs use `is not None` as the presence test, NOT
  # `> 0.0` like the legacy torque-neutral-crossing fields: a real mono time
  # can legitimately land on exactly 0.0 and must still serialize Present.
  evidence = LateralEvidence(
    1.0, 2.0, 0.0, 0.0, False, 0.0, 0, 0.5, "lat:episode", 2.0, 2.0,
    requested_crown_neutral_mono_time=0.0,
    wheel_progress_5_mono_time=0.0,
    road_evidence_window_start_mono_time=0.0,
    unwind_rebound_start_mono_time=None,
    future_unwind_commit_mono_time=1.0,
    high_angle_evidence_valid=True,
    high_angle_unwind_scale=0.42,
    torque_command_before_high_angle_exit=0.33,
    high_angle_unwind_old_torque_correction=0.11,
    high_angle_unwind_old_direction_torque=0.09,
    old_turn_sign=-1.0,
    driver_causation="interventionBacked",
    driver_assist_raw_torque_only=True,
    unwind_neutral_torque=0.18,
    unwind_phase_direction=-1.0,
  )
  detection = LateralDetection(
    "unwindProgressDeficit", "info", 0.95, "reason", evidence,
    occurred_mono_time=1.0, detected_mono_time=1.0,
  )
  candidate = lateral_candidate(LateralSample(1.0), detection)
  msg = build_message(AcceptedEvent("event", "group", candidate)).as_reader().drivingEvent
  payload = msg.payload.lateral
  assert payload.requestedCrownNeutralPresent
  assert payload.requestedCrownNeutralMonoTime == 0
  assert payload.wheelProgress5Present
  assert payload.wheelProgress5MonoTime == 0
  assert payload.roadEvidenceWindowStartPresent
  assert payload.roadEvidenceWindowStartMonoTime == 0
  assert not payload.unwindReboundStartPresent
  assert payload.futureUnwindCommitPresent
  assert payload.futureUnwindCommitMonoTime == 1_000_000_000
  assert payload.highAngleEvidenceValid
  assert payload.highAngleUnwindScale == pytest.approx(0.42)
  assert payload.torqueCommandBeforeHighAngleExit == pytest.approx(0.33)
  assert payload.highAngleUnwindOldTorqueCorrection == pytest.approx(0.11)
  assert payload.highAngleUnwindOldDirectionTorque == pytest.approx(0.09)
  assert payload.oldTurnSign == pytest.approx(-1.0)
  assert payload.driverCausation == "interventionBacked"
  assert payload.driverAssistRawTorqueOnly
  assert payload.unwindNeutralTorque == pytest.approx(0.18)
  assert payload.unwindPhaseDirection == pytest.approx(-1.0)


def test_lateral_build_message_high_angle_scalars_zero_when_invalid():
  evidence = LateralEvidence(
    1.0, 2.0, 0.0, 0.0, False, 0.0, 0, 0.5, "lat:episode", 2.0, 2.0,
    high_angle_evidence_valid=False,
    high_angle_unwind_scale=0.9,
    torque_command_before_high_angle_exit=0.9,
    high_angle_unwind_old_torque_correction=0.9,
    high_angle_unwind_old_direction_torque=0.9,
  )
  detection = LateralDetection(
    "centerOvershoot", "warning", 0.9, "reason", evidence,
    occurred_mono_time=1.0, detected_mono_time=1.0,
  )
  candidate = lateral_candidate(LateralSample(1.0), detection)
  msg = build_message(AcceptedEvent("event", "group", candidate)).as_reader().drivingEvent
  payload = msg.payload.lateral
  assert not payload.highAngleEvidenceValid
  assert payload.highAngleUnwindScale == 0.0
  assert payload.torqueCommandBeforeHighAngleExit == 0.0
  assert payload.highAngleUnwindOldTorqueCorrection == 0.0
  assert payload.highAngleUnwindOldDirectionTorque == 0.0


@pytest.mark.parametrize(
  ("internal_name", "public_name"),
  [
    ("unwind_progress_deficit", "lat.unwindProgressDeficit"),
    ("turn_stop_turn", "lat.turnStopTurn"),
  ],
)
def test_lateral_names_times_windows_and_missing_optional_evidence_are_safe(internal_name, public_name):
  evidence = LateralEvidence(
    1.0, 3.0, 0.0, 0.0, False, 0.0, 0, 0.5, "lat:v5", 6.0, 4.0,
  )
  detection = LateralDetection(
    internal_name, "warning", 0.9, "v5 evidence",
    evidence, occurred_mono_time=1.25, detected_mono_time=2.50,
  )
  candidate = lateral_candidate(LateralSample(2.5), detection)
  msg = build_message(AcceptedEvent("v5", "group", candidate)).as_reader().drivingEvent

  assert LAT_DETECTOR_VERSION == 7
  assert candidate.event_type == public_name
  assert msg.eventType == public_name
  assert msg.detectorVersion == 7
  assert msg.occurredMonoTime == 1_250_000_000
  assert msg.detectedMonoTime == 2_500_000_000
  assert msg.analysisWindowBeforeS == 6.0
  assert msg.analysisWindowAfterS == 4.0
  assert not msg.payload.lateral.requestedTorqueNeutralCrossPresent
  assert msg.payload.lateral.requestedTorqueNeutralCrossMonoTime == 0
  assert not msg.payload.lateral.appliedTorqueNeutralCrossPresent
  assert msg.payload.lateral.appliedTorqueNeutralCrossMonoTime == 0
  assert not msg.payload.lateral.turnStopDampingValid
  assert msg.payload.lateral.turnStopActualDampingState == "unavailable"
  assert msg.payload.lateral.unwindRequestedTorqueAtTrigger == 0.0
  assert not msg.payload.lateral.driverInvolvedDuringDwell
  # v6 optional evidence, absent on this legacy-shaped evidence object.
  assert not msg.payload.lateral.highAngleEvidenceValid
  assert msg.payload.lateral.highAngleUnwindScale == 0.0
  assert not msg.payload.lateral.futureUnwindCommitPresent
  assert msg.payload.lateral.futureUnwindCommitMonoTime == 0
  assert not msg.payload.lateral.requestedCrownNeutralPresent
  assert msg.payload.lateral.requestedCrownNeutralMonoTime == 0
  assert not msg.payload.lateral.appliedCrownNeutralPresent
  assert not msg.payload.lateral.wheelProgress5Present
  assert not msg.payload.lateral.wheelProgress20Present
  assert not msg.payload.lateral.wheelProgress50Present
  assert not msg.payload.lateral.unwindReboundStartPresent
  assert msg.payload.lateral.unwindReboundMaxMagnitude == 0.0
  assert msg.payload.lateral.driverCausation == ""
  assert not msg.payload.lateral.driverAssistRawTorqueOnly
  # v7 optional context is safe for legacy-shaped evidence.
  assert msg.payload.lateral.demandedCurvature == 0.0
  assert msg.payload.lateral.deliveredCurvatureFraction == 0.0
  assert msg.payload.lateral.torqueHeadroom == 0.0
  assert msg.payload.lateral.signedTrackingDeficit == 0.0
  assert msg.payload.lateral.authorityUnderDeliveryDurationS == 0.0
  assert msg.payload.lateral.takeoverConfirmationDurationS == 0.0
  assert msg.payload.lateral.demandedCurvature == 0.0
  assert msg.payload.lateral.deliveredCurvatureFraction == 0.0
  assert msg.payload.lateral.torqueHeadroom == 0.0
  assert msg.payload.lateral.signedTrackingDeficit == 0.0
  assert msg.payload.lateral.authorityUnderDeliveryDurationS == 0.0
  assert msg.payload.lateral.takeoverConfirmationDurationS == 0.0
  assert not msg.payload.lateral.roadEvidenceWindowStartPresent


def test_live_pose_sampler_uses_its_own_timestamp_and_updates_bump_once():
  class CountingBumpClassifier:
    def __init__(self):
      self.calls = []

    def update(self, z_accel, now):
      self.calls.append((z_accel, now))
      return True

  class FakeSubMaster:
    def __init__(self):
      self.data = {
        "deviceMotion": SimpleNamespace(
          accelerationDevice=SimpleNamespace(x=-0.8, z=9.9, valid=True),
          inputsOK=True,
          sensorsOK=True,
        ),
      }
      self.valid = {"deviceMotion": True}
      self.logMonoTime = {"deviceMotion": 2_500_000_000, "carState": 9_000_000_000}

    def __getitem__(self, name):
      return self.data[name]

  bump_classifier = CountingBumpClassifier()
  sample = live_pose_sample(FakeSubMaster(), bump_classifier)
  assert sample.t == 2.5
  assert sample.accel_x == -0.8
  assert sample.valid
  assert sample.road_confounded
  assert bump_classifier.calls == [(9.9, 2.5)]


def test_longitudinal_sampler_uses_radar_present_field():
  class FakeSubMaster:
    def __init__(self):
      self.data = {
        "radarState": SimpleNamespace(leadOne=SimpleNamespace(
          present=True, dRel=5.0, vLead=1.0, vLeadK=1.0, radarTrackId=7,
        )),
        "modelV2": SimpleNamespace(leadsV3=[]),
        "carState": SimpleNamespace(standstill=True, vEgo=0.0),
        "carControl": SimpleNamespace(longActive=True),
        "longitudinalPlan": SimpleNamespace(shouldStop=True),
        "carOutput": SimpleNamespace(actuatorsOutput=SimpleNamespace(accel=0.0)),
      }
      self.valid = dict.fromkeys(self.data, True)
      self.logMonoTime = {"carState": 1_000_000_000}

    def __getitem__(self, name):
      return self.data[name]

  sample = longitudinal_sample(FakeSubMaster())
  assert sample.lead_present
  assert sample.radar_track_id == 7


def test_rolling_sampler_requires_a_radar_matched_valid_track():
  class FakeSubMaster:
    def __init__(self):
      self.data = {
        "radarState": SimpleNamespace(leadOne=SimpleNamespace(
          present=True, radar=True, dRel=20.0, vLead=10.0, vLeadK=10.0, aLeadK=1.5, radarTrackId=7,
        )),
        "carState": SimpleNamespace(vEgo=8.0, aEgo=-0.2, gasPressed=False, brakePressed=False),
        "carControl": SimpleNamespace(longActive=True),
        "longitudinalPlan": SimpleNamespace(aTarget=-0.4, shouldStop=False),
        "carOutput": SimpleNamespace(actuatorsOutput=SimpleNamespace(accel=-0.4)),
      }
      self.valid = dict.fromkeys(self.data, True)
      self.logMonoTime = {"carState": 3_000_000_000}

    def __getitem__(self, name):
      return self.data[name]

  sm = FakeSubMaster()
  sample = rolling_lead_sample(sm)
  assert sample.t == 3.0
  assert sample.long_active and sample.lead_valid and sample.radar_valid
  assert sample.radar_track_id == 7
  sm.data["radarState"].leadOne.radar = False
  assert not rolling_lead_sample(sm).lead_valid


def test_lateral_sampler_uses_actual_caroutput_damping_not_torque_d_term():
  torque_state = SimpleNamespace(
    active=True, desiredLateralAccel=0.2, actualLateralAccel=0.1, p=0.3, d=-0.72,
    version=4, referenceVersion=4, referenceRate=0.5, referenceReachableTargetTorque=0.1,
    trackingMeasurementRate=-0.22,
    referenceUnwindScale=0.5, referenceSustainedUnwindScale=0.6, unwindEffectivePhase=0.7,
    unwindPhaseOverspeed=0.8, unwindSameEpisode=True,
  )

  class FakeSubMaster:
    def __init__(self):
      self.data = {
        "controlsState": SimpleNamespace(
          lateralControlState=SimpleNamespace(
            which=lambda: "torqueState",
            torqueState=torque_state,
          ),
        ),
        "carState": SimpleNamespace(
          steeringAngleDeg=30.0, vEgo=4.0, steeringTorque=0.0,
          steeringPressed=False, steeringTorqueEps=20.0,
        ),
        "carControl": SimpleNamespace(
          latActive=True, actuators=SimpleNamespace(torque=0.6),
        ),
        "carOutput": SimpleNamespace(actuatorsOutput=SimpleNamespace(
          torque=0.5,
          torqueDampingApplied=0.031,
          torqueDampingState="damping",
          torqueDampingVersion=2,
          torqueDampingBlocked=False,
          torqueBreakawayLatch=0.895,
          torqueDampingFloor=0.805,
        )),
      }
      self.logMonoTime = {"controlsState": 2_000_000_000}
      self.valid = {"carOutput": True}

    def __getitem__(self, name):
      return self.data[name]

  sm = FakeSubMaster()
  sample = lateral_sample(sm, SteeringRateFilter(), False, 2.1)
  assert sample.damping_applied == 0.0
  assert sample.damping_state == "legacyUnavailable"
  assert sample.actual_damping_amount == pytest.approx(0.031)
  assert sample.actual_damping_state == "damping"
  assert sample.breakaway_latch == pytest.approx(0.895)
  assert sample.sustain_floor_contribution == pytest.approx(0.805)
  assert sample.damping_version == 2
  assert sample.vertical_accel_deviation == pytest.approx(2.1)
  assert sample.measurement_rate == pytest.approx(-0.22)
  sm.valid["carOutput"] = False
  invalid = lateral_sample(sm, SteeringRateFilter(), False)
  assert invalid.actual_damping_amount is None
  assert invalid.actual_damping_state is None
  assert invalid.damping_version is None


def test_lateral_sampler_wires_crown_neutral_and_high_angle_fields_together():
  torque_state = SimpleNamespace(
    active=True, desiredLateralAccel=0.2, actualLateralAccel=0.1, p=0.3, d=-0.72,
    version=4, referenceVersion=4, referenceRate=0.5, referenceReachableTargetTorque=0.1,
    trackingMeasurementRate=-0.22,
    referenceUnwindScale=0.5, referenceSustainedUnwindScale=0.6, unwindEffectivePhase=0.7,
    unwindPhaseOverspeed=0.8, unwindSameEpisode=True,
    unwindNeutralTorque=0.18, unwindPhaseDirection=-1.0,
    highAngleUnwindScale=0.42, torqueCommandBeforeHighAngleExit=0.33,
    highAngleUnwindOldTorqueCorrection=0.11, highAngleUnwindOldDirectionTorque=0.09,
  )

  class FakeSubMaster:
    def __init__(self):
      self.data = {
        "controlsState": SimpleNamespace(
          lateralControlState=SimpleNamespace(which=lambda: "torqueState", torqueState=torque_state),
        ),
        "carState": SimpleNamespace(
          steeringAngleDeg=30.0, vEgo=4.0, steeringTorque=0.0,
          steeringPressed=False, steeringTorqueEps=20.0,
        ),
        "carControl": SimpleNamespace(latActive=True, actuators=SimpleNamespace(torque=0.6)),
        "carOutput": SimpleNamespace(actuatorsOutput=SimpleNamespace(torque=0.5)),
      }
      self.logMonoTime = {"controlsState": 2_000_000_000}
      self.valid = {"carOutput": False}

    def __getitem__(self, name):
      return self.data[name]

  sample = lateral_sample(FakeSubMaster(), SteeringRateFilter(), False)
  assert sample.unwind_neutral_torque == pytest.approx(0.18)
  assert sample.unwind_phase_direction == pytest.approx(-1.0)
  assert sample.high_angle_unwind_scale == pytest.approx(0.42)
  assert sample.torque_command_before_high_angle_exit == pytest.approx(0.33)
  assert sample.high_angle_unwind_old_torque_correction == pytest.approx(0.11)
  assert sample.high_angle_unwind_old_direction_torque == pytest.approx(0.09)


def test_lateral_sampler_high_angle_fields_all_none_together_when_schema_lacks_them():
  # DESIGN §1.1: the four high-angle fields are grouped under one
  # availability check so they are all-None (schema lacks them) or
  # all-float together — never gated on carOutput's damping_valid.
  torque_state = SimpleNamespace(
    active=True, desiredLateralAccel=0.2, actualLateralAccel=0.1, p=0.3, d=-0.72,
    version=4, unwindNeutralTorque=0.05, unwindPhaseDirection=1.0,
  )

  class FakeSubMaster:
    def __init__(self):
      self.data = {
        "controlsState": SimpleNamespace(
          lateralControlState=SimpleNamespace(which=lambda: "torqueState", torqueState=torque_state),
        ),
        "carState": SimpleNamespace(
          steeringAngleDeg=10.0, vEgo=4.0, steeringTorque=0.0,
          steeringPressed=False, steeringTorqueEps=20.0,
        ),
        "carControl": SimpleNamespace(latActive=True, actuators=SimpleNamespace(torque=0.1)),
        "carOutput": SimpleNamespace(actuatorsOutput=SimpleNamespace(torque=0.1)),
      }
      self.logMonoTime = {"controlsState": 1_000_000_000}
      self.valid = {"carOutput": False}

    def __getitem__(self, name):
      return self.data[name]

  sample = lateral_sample(FakeSubMaster(), SteeringRateFilter(), False)
  assert sample.unwind_neutral_torque == pytest.approx(0.05)
  assert sample.unwind_phase_direction == pytest.approx(1.0)
  assert sample.high_angle_unwind_scale is None
  assert sample.torque_command_before_high_angle_exit is None
  assert sample.high_angle_unwind_old_torque_correction is None
  assert sample.high_angle_unwind_old_direction_torque is None


def test_lateral_sampler_defaults_crown_and_high_angle_fields_without_torque_state():
  class FakeSubMaster:
    def __init__(self):
      self.data = {
        "controlsState": SimpleNamespace(
          lateralControlState=SimpleNamespace(which=lambda: "other"),
        ),
        "carState": SimpleNamespace(
          steeringAngleDeg=0.0, vEgo=0.0, steeringTorque=0.0,
          steeringPressed=False, steeringTorqueEps=0.0,
        ),
        "carControl": SimpleNamespace(latActive=False, actuators=SimpleNamespace(torque=0.0)),
        "carOutput": SimpleNamespace(actuatorsOutput=SimpleNamespace(torque=0.0)),
      }
      self.logMonoTime = {"controlsState": 500_000_000}
      self.valid = {"carOutput": False}

    def __getitem__(self, name):
      return self.data[name]

  sample = lateral_sample(FakeSubMaster(), SteeringRateFilter(), False)
  assert sample.unwind_neutral_torque == 0.0
  assert sample.unwind_phase_direction == 0.0
  assert sample.high_angle_unwind_scale is None
  assert sample.torque_command_before_high_angle_exit is None
  assert sample.high_angle_unwind_old_torque_correction is None
  assert sample.high_angle_unwind_old_direction_torque is None

from types import SimpleNamespace

from openpilot.selfdrive.spysypilot.driving_eventd import (
  AcceptedEvent,
  DrivingEventPlatform,
  EventSubmitter,
  EventCandidate,
  EventRecorder,
  build_message,
  longitudinal_sample,
  manual_candidate,
)
from openpilot.selfdrive.spysypilot.lat_event_detector import LateralDetection, LateralSample
from openpilot.selfdrive.spysypilot.long_event_detector import LaunchSample, LongEvent


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


def test_longitudinal_sampler_uses_fork_radar_status_field():
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

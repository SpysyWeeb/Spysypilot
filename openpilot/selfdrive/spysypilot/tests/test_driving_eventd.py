from types import SimpleNamespace

from openpilot.selfdrive.spysypilot.driving_eventd import (
  AcceptedEvent,
  DrivingEventPlatform,
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


def test_retrying_an_accepted_event_retains_ids():
  accepted = AcceptedEvent("event", "group", manual_candidate(123))
  first = build_message(accepted).drivingEvent
  second = build_message(accepted).drivingEvent
  assert first.eventId == second.eventId == "event"
  assert first.groupId == second.groupId == "group"


def test_manual_lateral_and_longitudinal_coexist():
  lateral = FixedDetector(LateralDetection("centerOvershoot", "warning", 0.9, "lateral reason"))
  longitudinal = FixedDetector(LongEvent(
    "late_lead_launch_vehicle", "Long Event Logged", "long reason", 2, 0.95,
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


def test_detector_exception_isolation():
  errors = []
  platform = DrivingEventPlatform(
    recorder=EventRecorder(iter(("group", "long")).__next__),
    lateral_detector=FixedDetector(error=RuntimeError("broken")),
    longitudinal_detector=FixedDetector(LongEvent(
      "late_lead_launch_controller", "Long Event Logged", "long reason", 1, 0.95,
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
          status=True, dRel=5.0, vLead=1.0, vLeadK=1.0, radarTrackId=7,
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

from types import SimpleNamespace

from openpilot.selfdrive.ui.onroad.driving_event_notification import EventNotificationQueue, notification_text


def ack(domain="lateral", source="automatic", group="group", event="event",
        marker=True, preserved=True):
  return SimpleNamespace(
    domain=domain,
    source=source,
    groupId=group,
    eventId=event,
    markerWritten=marker,
    currentSegmentPreserved=preserved,
  )


def test_success_requires_marker_and_current_preservation():
  assert notification_text(ack()) == ("Lat Event Logged", False)
  assert notification_text(ack(marker=False)) == ("Event Save Failed", True)
  assert notification_text(ack(preserved=False)) == ("Event Save Failed", True)


def test_domain_titles_are_owned_by_shared_renderer():
  assert notification_text(ack(domain="longitudinal")) == ("Long Event Logged", False)
  assert notification_text(ack(domain="manual", source="user")) == ("Event Logged", False)


def test_group_is_coalesced_without_hiding_failure():
  queue = EventNotificationQueue()
  queue.push(ack(domain="manual", source="user"), 1.0)
  queue.push(ack(domain="lateral"), 1.1)
  assert len(queue.pending) == 1
  assert queue.pending[0].text == "Lat Event Logged"
  queue.push(ack(domain="longitudinal", marker=False), 1.2)
  assert len(queue.pending) == 1
  assert queue.pending[0].text == "Event Save Failed"


def test_any_safety_alert_delays_then_releases_notice():
  queue = EventNotificationQueue()
  queue.push(ack(), 1.0)
  assert queue.current(safety_alert_visible=True, now=1.1) is None
  notice = queue.current(safety_alert_visible=False, now=2.0)
  assert notice is not None
  assert notice.text == "Lat Event Logged"


def test_corrected_acknowledgment_replaces_failure_for_same_event():
  queue = EventNotificationQueue()
  queue.push(ack(marker=False), 1.0)
  queue.push(ack(marker=True, preserved=True), 1.1)
  notice = queue.current(safety_alert_visible=False, now=1.2)
  assert notice is not None
  assert notice.text == "Lat Event Logged"
  assert not notice.failed

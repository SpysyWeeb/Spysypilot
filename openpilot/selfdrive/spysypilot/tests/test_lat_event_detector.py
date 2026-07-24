from openpilot.selfdrive.spysypilot.lat_event_detector import DETECTOR_VERSION, LateralEventDetector, LateralSample


def sample(t: float, **kwargs) -> LateralSample:
  defaults = {
    "active": True,
    "reference_rate": 0.3,
    "unwind_same_episode": True,
  }
  defaults.update(kwargs)
  return LateralSample(t, **defaults)


def test_per_type_cooldown_is_detector_version_two():
  assert DETECTOR_VERSION == 2


def test_inactive_never_triggers():
  detector = LateralEventDetector(cooldown=0.0)
  for i in range(200):
    event = detector.update(sample(i * 0.01, active=False, steering_rate_deg=200.0, request_torque=1.0, applied_torque=1.0))
    assert event is None


def test_late_unwind():
  detector = LateralEventDetector(cooldown=0.0)
  event = None
  for i in range(120):
    event = detector.update(sample(
      i * 0.01,
      steering_angle_deg=90.0,
      steering_rate_deg=0.0,
      reference_sustained_unwind_scale=0.9,
      unwind_effective_phase=0.9,
    ))
    if event is not None:
      break
  assert event is not None
  assert event.event_type == "lateUnwind"


def test_stall_release_cycles():
  detector = LateralEventDetector(cooldown=0.0)
  event = None
  now = 0.0
  for _ in range(3):
    for _ in range(20):
      event = detector.update(sample(now, steering_rate_deg=0.0))
      now += 0.01
    event = detector.update(sample(now, steering_rate_deg=40.0))
    now += 0.01
  assert event is not None
  assert event.event_type == "stallRelease"


def test_handoff_uses_symmetric_applied_target_gap():
  detector = LateralEventDetector(cooldown=0.0)
  detector.update(sample(0.0, unwind_effective_phase=1.0, applied_torque=-0.7, reference_target_torque=-0.7))
  event = detector.update(sample(
    0.01,
    unwind_effective_phase=0.0,
    unwind_same_episode=False,
    steering_rate_deg=80.0,
    applied_torque=-0.6,
    reference_target_torque=0.2,
  ))
  assert event is not None
  assert event.event_type == "handoffMismatch"


def test_center_overshoot():
  detector = LateralEventDetector(cooldown=0.0)
  event = detector.update(sample(
    0.0,
    steering_angle_deg=5.0,
    steering_rate_deg=150.0,
    unwind_effective_phase=0.8,
    unwind_overspeed=0.5,
    applied_torque=0.5,
    reference_target_torque=0.0,
  ))
  assert event is not None
  assert event.event_type == "centerOvershoot"


def test_cooldown_suppresses_repeated_events():
  detector = LateralEventDetector(cooldown=8.0)
  first = detector.update(sample(
    1.0,
    steering_angle_deg=5.0,
    steering_rate_deg=150.0,
    unwind_effective_phase=0.8,
    unwind_overspeed=0.5,
    applied_torque=0.5,
  ))
  second = detector.update(sample(
    2.0,
    steering_angle_deg=5.0,
    steering_rate_deg=150.0,
    unwind_effective_phase=0.8,
    unwind_overspeed=0.5,
    applied_torque=0.5,
  ))
  assert first is not None
  assert second is None


def test_different_event_types_do_not_share_a_cooldown():
  detector = LateralEventDetector(cooldown=8.0)
  first = detector.update(sample(
    1.0,
    steering_angle_deg=5.0,
    steering_rate_deg=150.0,
    unwind_effective_phase=0.8,
    unwind_overspeed=0.5,
    applied_torque=0.5,
  ))
  detector.update(sample(
    1.01,
    unwind_effective_phase=1.0,
    applied_torque=-0.7,
    reference_target_torque=-0.7,
  ))
  second = detector.update(sample(
    1.02,
    unwind_effective_phase=0.0,
    unwind_same_episode=False,
    steering_rate_deg=80.0,
    applied_torque=-0.6,
    reference_target_torque=0.2,
  ))
  assert first is not None and first.event_type == "centerOvershoot"
  assert second is not None and second.event_type == "handoffMismatch"

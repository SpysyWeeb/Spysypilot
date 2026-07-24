import pytest

from openpilot.selfdrive.spysypilot.lat_event_detector import DETECTOR_VERSION, LateralEventDetector, LateralSample


def sample(t: float, **kwargs) -> LateralSample:
  defaults = {
    "active": True,
    "reference_rate": 0.3,
    "unwind_same_episode": True,
  }
  defaults.update(kwargs)
  return LateralSample(t, **defaults)


def test_stall_lifetime_fix_is_detector_version_three():
  assert DETECTOR_VERSION == 3


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
  assert event.evidence is not None
  assert event.evidence.stall_release_count == 3
  assert len(event.evidence.stall_durations_s) == 3


@pytest.mark.parametrize(("inactive_s", "inactive_samples"), [
  (7.801, 638),
  (31.578, 2781),
  (40.232, 3451),
])
def test_tracking_inactive_cannot_release_a_stale_stall(inactive_s, inactive_samples):
  detector = LateralEventDetector(cooldown=0.0)
  now = 0.0
  for _ in range(20):
    detector.update(sample(now, steering_rate_deg=0.0))
    now += 0.01
  for index in range(inactive_samples):
    now = 0.2 + inactive_s * (index + 1) / inactive_samples
    detector.update(sample(
      now,
      steering_rate_deg=0.0,
      reference_rate=0.0,
      desired_lateral_accel=0.0,
      actual_lateral_accel=0.0,
    ))
  event = detector.update(sample(now + 0.01, steering_rate_deg=40.0))
  assert event is None
  assert len(detector.releases) == 0


def test_short_rate_transition_hysteresis_retains_real_release():
  detector = LateralEventDetector(cooldown=0.0)
  now = 0.0
  for _ in range(20):
    detector.update(sample(now, steering_rate_deg=0.0))
    now += 0.01
  detector.update(sample(now, steering_rate_deg=15.0))
  detector.update(sample(now + 0.1, steering_rate_deg=40.0))
  assert len(detector.releases) == 1


def test_evidence_aggregates_driver_and_road_confounders():
  detector = LateralEventDetector(cooldown=0.0)
  event = None
  now = 0.0
  for cycle in range(3):
    for _ in range(20):
      event = detector.update(sample(
        now,
        steering_rate_deg=0.0,
        driver_torque=80.0 if cycle == 0 else 0.0,
        road_confounded=cycle == 1,
      ))
      now += 0.01
    event = detector.update(sample(now, steering_rate_deg=40.0))
    now += 0.01
  assert event is not None and event.evidence is not None
  assert event.evidence.driver_confounded_any
  assert event.evidence.road_confounded_any
  assert event.evidence.max_abs_driver_torque == 80.0
  assert 0.0 < event.evidence.driver_confounded_fraction < 1.0
  assert 0.0 < event.evidence.road_confounded_fraction < 1.0


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


def test_extended_maneuver_retains_semantic_episode_key():
  detector = LateralEventDetector(cooldown=0.0)
  first = detector.update(sample(
    0.0,
    steering_angle_deg=5.0,
    steering_rate_deg=150.0,
    unwind_effective_phase=0.8,
    unwind_overspeed=0.5,
    applied_torque=0.5,
  ))
  assert first is not None and first.evidence is not None
  for second in range(1, 7):
    detector.update(sample(
      float(second),
      steering_angle_deg=60.0,
      steering_rate_deg=20.0,
      unwind_effective_phase=1.0,
      applied_torque=-0.7,
      reference_target_torque=-0.7,
    ))
  last = detector.update(sample(
    7.0,
    steering_angle_deg=40.0,
    steering_rate_deg=80.0,
    unwind_effective_phase=0.9,
    unwind_same_episode=False,
    applied_torque=-0.6,
    reference_target_torque=0.2,
  ))
  assert last is not None and last.evidence is not None
  assert first.evidence.episode_key == last.evidence.episode_key

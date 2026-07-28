from collections import deque

import pytest

from openpilot.selfdrive.spysypilot import lat_event_detector
from openpilot.selfdrive.spysypilot.lat_event_detector import (
  DETECTOR_VERSION,
  DRIVER_AT_START_WINDOW_S,
  DRIVER_CAUSATION_AUTONOMOUS_ONLY,
  DRIVER_CAUSATION_DRIVER_CREATED,
  DRIVER_CAUSATION_INTERVENTION_BACKED,
  DRIVER_CONFOUND_STEERING_PRESSED,
  DRIVER_CONFOUND_TORQUE,
  DRIVER_INTERACTION_CONFIRMED_STEERING_PRESSED,
  DRIVER_INTERACTION_POSSIBLE_RAW_TORQUE,
  EPISODE_CENTER_CLOSE_S,
  EPISODE_MAX_LIFETIME_S,
  LateralEventDetector,
  LateralSample,
  PendingHandoff,
  ROAD_INTERACTION_SUBSTANTIAL,
  ROAD_INTERACTION_TRANSIENT,
  TURN_STOP_CLUSTER_SELECTION_S,
  TURN_STOP_MAX_DWELL_S,
  TURN_STOP_MIN_POST_PROGRESS_DEG,
  UnwindProgressEpisode,
  UNWIND_REBOUND_MIN_COMPONENT,
)


def sample(t: float, **kwargs) -> LateralSample:
  defaults = {
    "active": True,
    "reference_rate": 0.3,
    "unwind_same_episode": True,
  }
  defaults.update(kwargs)
  return LateralSample(t, **defaults)


def settle(detector: LateralEventDetector, start: float, desired: float,
           actual: float | None = None, end: float | None = None, **kwargs):
  event = None
  actual = desired if actual is None else actual
  end = start + 0.6 if end is None else end
  tick = int(round(start * 100.0)) + 1
  while tick / 100.0 <= end + 1e-9:
    event = detector.update(sample(
      tick / 100.0,
      steering_angle_deg=-10.0,
      steering_rate_deg=0.0,
      desired_lateral_accel=desired,
      actual_lateral_accel=actual,
      **kwargs,
    )) or event
    tick += 1
  return event


def fast_crossing(detector: LateralEventDetector, desired_before: float = 0.5,
                  desired_after: float = 0.5, rate: float = -150.0,
                  gap: float = 0.5, start: float = 0.0):
  common = {
    "steering_rate_deg": rate,
    "unwind_effective_phase": 0.8,
    "unwind_overspeed": 0.5,
    "applied_torque": gap,
    "reference_target_torque": 0.0,
  }
  detector.update(sample(
    start,
    steering_angle_deg=30.0,
    desired_lateral_accel=desired_before,
    actual_lateral_accel=desired_before,
    **common,
  ))
  detector.update(sample(
    start + 0.15,
    steering_angle_deg=25.0,
    desired_lateral_accel=desired_before,
    actual_lateral_accel=desired_before,
    **common,
  ))
  detector.update(sample(
    start + 0.20,
    steering_angle_deg=20.0,
    desired_lateral_accel=desired_after,
    actual_lateral_accel=desired_after,
    **common,
  ))
  detector.update(sample(
    start + 0.25,
    steering_angle_deg=5.0,
    desired_lateral_accel=desired_after,
    actual_lateral_accel=desired_after,
    **common,
  ))
  event = detector.update(sample(
    start + 0.30,
    steering_angle_deg=-5.0,
    desired_lateral_accel=desired_after,
    actual_lateral_accel=desired_after,
    **common,
  ))
  return event or settle(
    detector,
    start + 0.30,
    desired_after,
    desired_after,
    start + 0.90,
    unwind_effective_phase=0.8,
    unwind_overspeed=0.5,
    applied_torque=gap,
  )


def stall_release_event(detector: LateralEventDetector, *,
                        release_kwargs: list[dict] | None = None,
                        stationary_kwargs: list[dict] | None = None):
  now = 0.0
  event = None
  release_kwargs = release_kwargs or [{}, {}, {}]
  stationary_kwargs = stationary_kwargs or [{}, {}, {}]
  for cycle in range(3):
    for _ in range(20):
      event = detector.update(sample(
        now,
        steering_rate_deg=0.0,
        **stationary_kwargs[cycle],
      )) or event
      now += 0.01
    event = detector.update(sample(
      now,
      steering_rate_deg=40.0,
      **release_kwargs[cycle],
    )) or event
    now += 0.01
  return event


def test_lateral_detector_version_seven():
  assert DETECTOR_VERSION == 7


def test_inactive_never_triggers():
  detector = LateralEventDetector(cooldown=0.0)
  for i in range(200):
    event = detector.update(sample(i * 0.01, active=False, steering_rate_deg=200.0, request_torque=1.0, applied_torque=1.0))
    assert event is None


def test_torque_under_delivery_requires_sustained_error_and_unused_headroom():
  detector = LateralEventDetector(cooldown=0.0)
  event = None
  for index in range(102):
    result = detector.update(sample(
      index * 0.01,
      v_ego=4.0,
      desired_lateral_accel=1.0,
      actual_lateral_accel=0.5,
      request_torque=0.6,
      applied_torque=0.6,
    ))
    if result is not None:
      batch = result if isinstance(result, (list, tuple)) else (result,)
      event = next(
        (item for item in batch if item.event_type == "torqueUnderDelivery"),
        event,
      )
  assert event is not None and event.evidence is not None
  assert event.occurred_mono_time == pytest.approx(0.0)
  assert event.detected_mono_time == pytest.approx(1.0)
  assert event.evidence.authority_under_delivery_duration_s == pytest.approx(1.0)
  assert event.evidence.demanded_curvature == pytest.approx(0.0625)
  assert event.evidence.delivered_curvature_fraction == pytest.approx(0.5)
  assert event.evidence.torque_headroom == pytest.approx(0.4)
  assert event.evidence.signed_tracking_deficit == pytest.approx(0.5)


@pytest.mark.parametrize("changes", [
  {"request_torque": 0.76},       # 24% headroom, below the b7-derived 25%.
  {"actual_lateral_accel": 0.81}, # More than 80% delivered.
  {"desired_lateral_accel": 0.34, "actual_lateral_accel": 0.0},
])
def test_torque_under_delivery_each_envelope_boundary_is_required(changes):
  detector = LateralEventDetector(cooldown=0.0)
  kwargs = {
    "v_ego": 4.0,
    "desired_lateral_accel": 1.0,
    "actual_lateral_accel": 0.5,
    "request_torque": 0.6,
    "applied_torque": 0.6,
  }
  kwargs.update(changes)
  for index in range(130):
    result = detector.update(sample(index * 0.01, **kwargs))
    batch = result if isinstance(result, (list, tuple)) else (result,)
    assert not any(
      item is not None and item.event_type == "torqueUnderDelivery"
      for item in batch
    )


def test_takeover_requires_300ms_and_emits_once_per_confirmed_press():
  detector = LateralEventDetector(cooldown=0.0)
  events = []
  for index in range(80):
    pressed = 10 <= index <= 55
    result = detector.update(sample(
      index * 0.01,
      v_ego=4.0,
      steering_pressed=pressed,
      steering_angle_deg=80.0,
      desired_lateral_accel=1.0,
      actual_lateral_accel=0.5,
      request_torque=0.6,
    ))
    if result is not None:
      batch = result if isinstance(result, (list, tuple)) else (result,)
      events.extend(item for item in batch if item.event_type == "driverTakeover")
  assert len(events) == 1
  event = events[0]
  assert event.occurred_mono_time == pytest.approx(0.10)
  assert event.detected_mono_time == pytest.approx(0.40)
  assert event.severity == "critical"
  assert event.evidence is not None
  assert event.evidence.takeover_confirmation_duration_s == pytest.approx(0.30)
  assert event.evidence.demanded_curvature == pytest.approx(0.0625)
  assert event.evidence.torque_headroom == pytest.approx(0.4)


def test_short_or_inactive_steering_press_is_not_a_takeover():
  detector = LateralEventDetector(cooldown=0.0)
  for index in range(100):
    result = detector.update(sample(
      index * 0.01,
      active=index >= 50,
      steering_pressed=(10 <= index < 40) or (60 <= index < 89),
    ))
    batch = result if isinstance(result, (list, tuple)) else (result,)
    assert not any(
      item is not None and item.event_type == "driverTakeover"
      for item in batch
    )


def test_takeover_and_under_delivery_on_same_frame_are_both_retained():
  detector = LateralEventDetector(cooldown=0.0)
  simultaneous = None
  for index in range(102):
    result = detector.update(sample(
      index * 0.01,
      v_ego=4.0,
      # Start one frame before the mathematically exact 0.70 s boundary so
      # binary float representation cannot defer the 0.30 s confirmation by
      # one frame; both detectors must mature on the 1.00 s sample.
      steering_pressed=index >= 69,
      desired_lateral_accel=1.0,
      actual_lateral_accel=0.4,
      request_torque=0.5,
    ))
    if isinstance(result, tuple) and len(result) > 1:
      simultaneous = result
  assert simultaneous is not None
  assert {event.event_type for event in simultaneous} == {
    "torqueUnderDelivery", "driverTakeover",
  }


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
  event = stall_release_event(LateralEventDetector(cooldown=0.0))
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


def test_fast_center_crossing_without_reversal_is_center_overshoot():
  event = fast_crossing(LateralEventDetector(cooldown=0.0))
  assert event is not None and event.evidence is not None
  assert event.event_type == "centerOvershoot"
  assert not event.evidence.committed_reversal


def test_fast_center_crossing_after_committed_reversal_is_harsh_handoff():
  event = fast_crossing(
    LateralEventDetector(cooldown=0.0),
    desired_before=0.5,
    desired_after=-0.5,
  )
  assert event is not None and event.evidence is not None
  assert event.event_type == "committedHandoffHarshness"
  assert event.evidence.committed_reversal


def test_desired_sign_blip_is_not_a_committed_reversal():
  detector = LateralEventDetector(cooldown=0.0)
  common = {
    "steering_rate_deg": -150.0,
    "unwind_effective_phase": 0.8,
    "unwind_overspeed": 0.5,
    "applied_torque": 0.5,
  }
  detector.update(sample(-0.2, steering_angle_deg=30.0, desired_lateral_accel=0.5, **common))
  detector.update(sample(0.0, steering_angle_deg=30.0, desired_lateral_accel=0.5, **common))
  detector.update(sample(0.05, steering_angle_deg=20.0, desired_lateral_accel=-0.5, **common))
  detector.update(sample(0.10, steering_angle_deg=5.0, desired_lateral_accel=0.5, **common))
  detector.update(sample(0.15, steering_angle_deg=-5.0, desired_lateral_accel=0.5, **common))
  event = settle(
    detector,
    0.15,
    0.5,
    0.5,
    0.75,
    unwind_effective_phase=0.8,
    unwind_overspeed=0.5,
    applied_torque=0.5,
  )
  assert event is not None
  assert event.event_type == "centerOvershoot"


def test_one_meaningful_reversal_sample_followed_by_zero_is_not_committed():
  detector = LateralEventDetector(cooldown=0.0)
  common = {
    "steering_rate_deg": -150.0,
    "unwind_effective_phase": 0.8,
    "unwind_overspeed": 0.5,
    "applied_torque": 0.5,
  }
  detector.update(sample(-0.2, steering_angle_deg=30.0, desired_lateral_accel=0.5, **common))
  detector.update(sample(0.0, steering_angle_deg=30.0, desired_lateral_accel=0.5, **common))
  detector.update(sample(0.05, steering_angle_deg=20.0, desired_lateral_accel=-0.5, **common))
  detector.update(sample(0.10, steering_angle_deg=5.0, desired_lateral_accel=0.0, **common))
  detector.update(sample(0.15, steering_angle_deg=-5.0, desired_lateral_accel=0.0, **common))
  event = settle(
    detector,
    0.15,
    0.0,
    0.0,
    0.75,
    unwind_effective_phase=0.8,
    unwind_overspeed=0.5,
    applied_torque=0.5,
  )
  assert event is not None
  assert event.event_type == "centerOvershoot"


def test_normal_smooth_committed_reversal_is_silent():
  event = fast_crossing(
    LateralEventDetector(cooldown=0.0),
    desired_before=0.5,
    desired_after=-0.5,
    rate=-60.0,
    gap=0.05,
  )
  assert event is None


def test_brief_torque_gap_does_not_make_smooth_committed_reversal_harsh():
  detector = LateralEventDetector(cooldown=0.0)
  detector.update(sample(
    -0.2, steering_angle_deg=30.0, steering_rate_deg=-60.0,
    desired_lateral_accel=0.5, actual_lateral_accel=0.5,
    applied_torque=0.05,
  ))
  detector.update(sample(
    0.0, steering_angle_deg=30.0, steering_rate_deg=-60.0,
    desired_lateral_accel=0.5, actual_lateral_accel=0.5,
    applied_torque=0.05,
  ))
  detector.update(sample(
    0.05, steering_angle_deg=20.0, steering_rate_deg=-60.0,
    desired_lateral_accel=-0.5, actual_lateral_accel=-0.5,
    applied_torque=0.05,
  ))
  detector.update(sample(
    0.10, steering_angle_deg=5.0, steering_rate_deg=-60.0,
    desired_lateral_accel=-0.5, actual_lateral_accel=-0.5,
    applied_torque=0.20,
  ))
  detector.update(sample(
    0.15, steering_angle_deg=-5.0, steering_rate_deg=-60.0,
    desired_lateral_accel=-0.5, actual_lateral_accel=-0.5,
    applied_torque=0.20,
  ))
  event = settle(
    detector,
    0.15,
    -0.5,
    -0.5,
    0.80,
    applied_torque=0.05,
  )
  assert event is None


def test_tracking_error_persisting_before_crossing_counts_as_harshness():
  detector = LateralEventDetector(cooldown=0.0)
  for tick in range(-20, 1):
    detector.update(sample(
      tick / 100.0,
      steering_angle_deg=30.0,
      steering_rate_deg=-60.0,
      desired_lateral_accel=0.5,
      actual_lateral_accel=0.0,
      applied_torque=0.05,
    ))
  detector.update(sample(
    0.05, steering_angle_deg=20.0, steering_rate_deg=-60.0,
    desired_lateral_accel=-0.5, actual_lateral_accel=0.0,
    applied_torque=0.05,
  ))
  detector.update(sample(
    0.10, steering_angle_deg=5.0, steering_rate_deg=-60.0,
    desired_lateral_accel=-0.5, actual_lateral_accel=-0.5,
    applied_torque=0.05,
  ))
  detector.update(sample(
    0.15, steering_angle_deg=-5.0, steering_rate_deg=-60.0,
    desired_lateral_accel=-0.5, actual_lateral_accel=-0.5,
    applied_torque=0.05,
  ))
  event = settle(
    detector,
    0.15,
    -0.5,
    -0.5,
    0.80,
    applied_torque=0.05,
  )
  assert event is not None
  assert event.event_type == "committedHandoffHarshness"


def test_handoff_and_crossing_within_half_second_are_one_event():
  detector = LateralEventDetector(cooldown=0.0)
  detector.update(sample(
    -0.2,
    steering_angle_deg=30.0,
    steering_rate_deg=-150.0,
    desired_lateral_accel=0.5,
    actual_lateral_accel=0.5,
    unwind_effective_phase=1.0,
    applied_torque=0.5,
  ))
  detector.update(sample(
    0.0,
    steering_angle_deg=30.0,
    steering_rate_deg=-150.0,
    desired_lateral_accel=0.5,
    actual_lateral_accel=0.5,
    unwind_effective_phase=1.0,
    applied_torque=0.5,
  ))
  detector.update(sample(
    0.05,
    steering_angle_deg=20.0,
    steering_rate_deg=-150.0,
    desired_lateral_accel=-0.5,
    actual_lateral_accel=0.5,
    unwind_effective_phase=0.0,
    unwind_same_episode=False,
    applied_torque=0.5,
  ))
  detector.update(sample(
    0.10,
    steering_angle_deg=5.0,
    steering_rate_deg=-150.0,
    desired_lateral_accel=-0.5,
    actual_lateral_accel=0.5,
    unwind_effective_phase=0.8,
    unwind_overspeed=0.5,
    applied_torque=0.5,
  ))
  detector.update(sample(
    0.15,
    steering_angle_deg=-5.0,
    steering_rate_deg=-150.0,
    desired_lateral_accel=-0.5,
    actual_lateral_accel=0.5,
    unwind_effective_phase=0.8,
    unwind_overspeed=0.5,
    applied_torque=0.5,
  ))
  events = []
  for tick in range(16, 90):
    event = detector.update(sample(
      tick / 100.0,
      steering_angle_deg=-10.0,
      desired_lateral_accel=-0.5,
      actual_lateral_accel=0.5,
      unwind_effective_phase=0.8,
      unwind_overspeed=0.5,
      applied_torque=0.5,
    ))
    if event is not None:
      events.append(event)
  assert [event.event_type for event in events] == ["committedHandoffHarshness"]
  assert events[0].evidence is not None
  assert events[0].evidence.phase_handoff_mono_time == pytest.approx(0.05)


def test_committed_reversal_does_not_assume_wheel_and_trajectory_sign_mapping():
  """Segment-52 shape: wheel crosses -/+ while desired acceleration reverses +/-."""
  detector = LateralEventDetector(cooldown=0.0)
  detector.update(sample(
    -0.2,
    steering_angle_deg=-30.0,
    steering_rate_deg=315.0,
    desired_lateral_accel=0.236,
    actual_lateral_accel=0.1,
    unwind_effective_phase=1.0,
    applied_torque=0.307,
  ))
  detector.update(sample(
    0.0,
    steering_angle_deg=-30.0,
    steering_rate_deg=315.0,
    desired_lateral_accel=0.236,
    actual_lateral_accel=0.1,
    unwind_effective_phase=1.0,
    applied_torque=0.307,
  ))
  detector.update(sample(
    0.10,
    steering_angle_deg=-20.0,
    steering_rate_deg=315.0,
    desired_lateral_accel=-0.20,
    actual_lateral_accel=0.1,
    unwind_effective_phase=0.0,
    unwind_same_episode=False,
    applied_torque=0.307,
  ))
  detector.update(sample(
    0.30,
    steering_angle_deg=-5.0,
    steering_rate_deg=315.0,
    desired_lateral_accel=-0.359,
    actual_lateral_accel=0.4,
    unwind_effective_phase=0.8,
    unwind_overspeed=0.5,
    applied_torque=0.307,
  ))
  detector.update(sample(
    0.40,
    steering_angle_deg=5.0,
    steering_rate_deg=315.0,
    desired_lateral_accel=-0.359,
    actual_lateral_accel=0.4,
    unwind_effective_phase=0.8,
    unwind_overspeed=0.5,
    applied_torque=0.307,
  ))
  event = None
  for tick in range(41, 105):
    event = detector.update(sample(
      tick / 100.0,
      steering_angle_deg=10.0,
      desired_lateral_accel=-0.359,
      actual_lateral_accel=0.4,
      unwind_effective_phase=0.8,
      unwind_overspeed=0.5,
      applied_torque=0.307,
    )) or event
  assert event is not None and event.evidence is not None
  assert event.event_type == "committedHandoffHarshness"
  assert event.evidence.handoff_consolidated
  assert event.evidence.handoff_center_delta_s == pytest.approx(0.25)
  assert event.evidence.signed_center_rate_deg == 315.0
  assert event.evidence.desired_lateral_accel_before_crossing == pytest.approx(0.236)
  assert event.evidence.desired_lateral_accel_after_crossing == pytest.approx(-0.20)


def test_handoff_and_unrelated_later_crossing_are_separate_events():
  detector = LateralEventDetector(cooldown=0.0)
  detector.update(sample(0.0, steering_angle_deg=30.0, unwind_effective_phase=1.0))
  detector.update(sample(
    0.01,
    steering_angle_deg=30.0,
    steering_rate_deg=80.0,
    unwind_effective_phase=0.0,
    unwind_same_episode=False,
    applied_torque=0.5,
  ))
  events = []
  for tick in range(2, 70):
    event = detector.update(sample(tick / 100.0, steering_angle_deg=30.0))
    if event is not None:
      events.append(event)
  crossing = fast_crossing(detector, start=0.70)
  if crossing is not None:
    events.append(crossing)
  assert [event.event_type for event in events] == ["handoffMismatch", "centerOvershoot"]


def test_crossing_occurrence_is_interpolated_and_detection_is_later():
  event = fast_crossing(LateralEventDetector(cooldown=0.0))
  assert event is not None
  assert event.occurred_mono_time == pytest.approx(0.275)
  assert event.detected_mono_time is not None
  assert event.detected_mono_time >= event.occurred_mono_time + 0.5


def test_exact_zero_sample_does_not_hide_physical_crossing():
  detector = LateralEventDetector(cooldown=0.0)
  common = {
    "steering_rate_deg": -150.0,
    "desired_lateral_accel": 0.5,
    "actual_lateral_accel": 0.5,
    "unwind_effective_phase": 0.8,
    "unwind_overspeed": 0.5,
    "applied_torque": 0.5,
  }
  detector.update(sample(0.0, steering_angle_deg=30.0, **common))
  detector.update(sample(0.10, steering_angle_deg=5.0, **common))
  detector.update(sample(0.15, steering_angle_deg=0.0, **common))
  detector.update(sample(0.20, steering_angle_deg=-5.0, **common))
  event = settle(
    detector,
    0.20,
    0.5,
    0.5,
    0.80,
    unwind_effective_phase=0.8,
    unwind_overspeed=0.5,
    applied_torque=0.5,
  )
  assert event is not None
  assert event.event_type == "centerOvershoot"
  assert event.occurred_mono_time == pytest.approx(0.15)


def test_raw_torque_spike_is_possible_interaction_not_confirmed_press():
  event = stall_release_event(
    LateralEventDetector(cooldown=0.0),
    stationary_kwargs=[
      {"driver_torque": 80.0},
      {},
      {},
    ],
  )
  assert event is not None and event.evidence is not None
  assert event.evidence.driver_interaction == DRIVER_INTERACTION_POSSIBLE_RAW_TORQUE
  assert event.evidence.driver_confound_reason == DRIVER_CONFOUND_TORQUE
  assert 0.0 < event.evidence.raw_torque_exceeded_fraction < 1.0
  assert not event.evidence.steering_pressed_any


def test_sustained_steering_pressed_is_confirmed_and_does_not_suppress():
  event = stall_release_event(
    LateralEventDetector(cooldown=0.0),
    stationary_kwargs=[
      {"steering_pressed": True},
      {"steering_pressed": True},
      {"steering_pressed": True},
    ],
    release_kwargs=[
      {"steering_pressed": True},
      {"steering_pressed": True},
      {"steering_pressed": True},
    ],
  )
  assert event is not None and event.evidence is not None
  assert event.evidence.driver_interaction == DRIVER_INTERACTION_CONFIRMED_STEERING_PRESSED
  assert event.evidence.driver_confound_reason & DRIVER_CONFOUND_STEERING_PRESSED
  assert event.evidence.steering_pressed_fraction > 0.9


def test_trigger_road_bump_is_distinct_from_transient_window_bump():
  trigger_bump = stall_release_event(
    LateralEventDetector(cooldown=0.0),
    release_kwargs=[{}, {}, {"road_confounded": True, "vertical_accel_deviation": 2.0}],
  )
  transient_detector = LateralEventDetector(cooldown=0.0)
  transient = None
  now = 0.0
  for cycle in range(3):
    for stationary_index in range(20):
      one_sample_bump = cycle == 0 and stationary_index == 0
      transient = transient_detector.update(sample(
        now,
        steering_rate_deg=0.0,
        road_confounded=one_sample_bump,
        vertical_accel_deviation=1.5 if one_sample_bump else 0.0,
      )) or transient
      now += 0.01
    transient = transient_detector.update(sample(now, steering_rate_deg=40.0)) or transient
    now += 0.01
  assert trigger_bump is not None and trigger_bump.evidence is not None
  assert transient is not None and transient.evidence is not None
  assert trigger_bump.evidence.road_confounded_at_trigger
  assert not transient.evidence.road_confounded_at_trigger
  assert transient.evidence.road_interaction == ROAD_INTERACTION_TRANSIENT
  assert transient.evidence.max_vertical_accel_deviation == 1.5


def test_substantial_road_bump_remains_emitted_and_marked():
  event = stall_release_event(
    LateralEventDetector(cooldown=0.0),
    stationary_kwargs=[
      {"road_confounded": True, "vertical_accel_deviation": 2.5},
      {"road_confounded": True, "vertical_accel_deviation": 2.0},
      {},
    ],
  )
  assert event is not None and event.evidence is not None
  assert event.evidence.road_interaction == ROAD_INTERACTION_SUBSTANTIAL
  assert event.evidence.road_confounded_fraction > 0.25


def test_stall_release_contains_three_distinct_actual_damping_snapshots():
  event = stall_release_event(
    LateralEventDetector(cooldown=0.0),
    release_kwargs=[
      {
        "actual_damping_amount": 0.0,
        "actual_damping_state": "turnInAuthority",
        "turn_in_blocked": True,
        "breakaway_latch": 0.0,
        "sustain_floor_contribution": 0.0,
        "damping_version": 2,
      },
      {
        "actual_damping_amount": 0.0,
        "actual_damping_state": "turnInAuthority",
        "turn_in_blocked": True,
        "breakaway_latch": 0.0,
        "sustain_floor_contribution": 0.0,
        "damping_version": 2,
      },
      {
        "actual_damping_amount": 0.031785,
        "actual_damping_state": "damping",
        "turn_in_blocked": False,
        "breakaway_latch": 0.894866,
        "sustain_floor_contribution": 0.805379,
        "damping_version": 2,
      },
    ],
  )
  assert event is not None and event.evidence is not None
  releases = event.evidence.stall_releases
  assert len(releases) == 3
  assert [release.actual_damping_amount for release in releases] == [0.0, 0.0, 0.031785]
  assert [release.turn_in_blocked for release in releases] == [True, True, False]
  assert releases[0].breakaway_latch == 0.0
  assert releases[1].breakaway_latch == 0.0
  assert releases[2].breakaway_latch == 0.894866
  assert releases[2].sustain_floor_contribution == 0.805379


def test_optional_vehicle_damping_fields_can_be_absent():
  event = stall_release_event(LateralEventDetector(cooldown=0.0))
  assert event is not None and event.evidence is not None
  assert all(release.actual_damping_amount is None for release in event.evidence.stall_releases)
  assert all(release.actual_damping_state is None for release in event.evidence.stall_releases)


def test_non_hyundai_does_not_reuse_legacy_torque_state_d_as_actual_damping():
  event = stall_release_event(
    LateralEventDetector(cooldown=0.0),
    release_kwargs=[
      {"damping_applied": 4.0},
      {"damping_applied": 5.0},
      {"damping_applied": 6.0},
    ],
  )
  assert event is not None and event.evidence is not None
  assert all(release.actual_damping_amount is None for release in event.evidence.stall_releases)


def unwind_progress_event(*, angle: float = 300.0, driver_assist: bool = False,
                          road_confounded: bool = False, assist_mode: str | None = None,
                          assist_at: float = 0.75, sustained_driver_torque: float = 0.0):
  detector = LateralEventDetector(cooldown=0.0)
  expected_direction = -1.0 if angle > 0.0 else 1.0
  reference_rate = -expected_direction
  if assist_mode is None and driver_assist:
    assist_mode = "pressed"
  detector.update(sample(
    0.0,
    steering_angle_deg=angle - expected_direction * 10.0,
    steering_rate_deg=-expected_direction * 300.0,
    reference_rate=0.0,
    driver_torque=sustained_driver_torque,
  ))
  event = None
  for index in range(1, 321):
    now = index * 0.01
    kwargs = {
      "steering_angle_deg": angle,
      "steering_rate_deg": expected_direction * 100.0,
      "reference_rate": reference_rate,
      "measurement_rate": -0.2 * reference_rate,
      "reference_sustained_unwind_scale": 0.9,
      "unwind_effective_phase": 0.9,
      "request_torque": 0.8 if now < 0.6 else -0.8,
      "applied_torque": 0.8 if now < 0.7 else -0.8,
      "reference_target_torque": -0.9,
      "road_confounded": road_confounded,
      "vertical_accel_deviation": 1.8 if road_confounded else 0.0,
      "driver_torque": sustained_driver_torque,
    }
    if assist_mode == "pressed" and now >= assist_at:
      kwargs["driver_torque"] = expected_direction * 100.0
      kwargs["steering_pressed"] = True
      kwargs["steering_rate_deg"] = expected_direction * 180.0
    elif assist_mode == "raw" and now >= assist_at:
      kwargs["driver_torque"] = expected_direction * 250.0
      kwargs["steering_rate_deg"] = expected_direction * 180.0
    event = detector.update(sample(now, **kwargs))
    if event is not None:
      return event
  return event


def turn_stop_turn_event(*, movement_rate: float = 120.0, dwell_rate: float = 0.0,
                         release_rate: float = 60.0, angle: float = 120.0,
                         dwell_s: float = 0.6, movement_s: float = 0.15,
                         post_sweep_ticks: int = 30, driver_before: bool = False,
                         driver_during: bool = False, driver_after: bool = False,
                         road_confounded: bool = False, demand: bool = True,
                         damping: bool = False):
  detector = LateralEventDetector(cooldown=0.0)
  reference_rate = 0.5 if demand else 0.0
  desired = 0.5 if demand else 0.0
  actual = 0.0 if demand else 0.0
  movement_count = int(round(movement_s / 0.01))
  for index in range(movement_count):
    # The wheel sweeps kinematically into the dwell angle at movement_rate,
    # so the pre-dwell travel equals movement_rate * movement_s.
    detector.update(sample(
      index * 0.01,
      steering_angle_deg=angle - movement_rate * (movement_count - index) * 0.01,
      steering_rate_deg=movement_rate,
      reference_rate=reference_rate,
      desired_lateral_accel=desired,
      actual_lateral_accel=actual,
      driver_torque=80.0 if driver_before else 0.0,
      steering_pressed=driver_before,
    ))
  count = int(round(dwell_s / 0.01))
  for index in range(count):
    detector.update(sample(
      (movement_count + index) * 0.01,
      steering_angle_deg=angle,
      steering_rate_deg=dwell_rate,
      reference_rate=reference_rate,
      desired_lateral_accel=desired,
      actual_lateral_accel=actual,
      request_torque=0.9 if demand else 0.0,
      applied_torque=0.85 if demand else 0.0,
      reference_target_torque=0.8 if demand else 0.0,
      driver_torque=80.0 if driver_during else 0.0,
      steering_pressed=driver_during,
      road_confounded=road_confounded,
      vertical_accel_deviation=1.7 if road_confounded else 0.0,
    ))
  release_time = (movement_count + count) * 0.01
  event = detector.update(sample(
    release_time,
    steering_angle_deg=angle,
    steering_rate_deg=release_rate,
    reference_rate=reference_rate,
    desired_lateral_accel=desired,
    actual_lateral_accel=actual,
    request_torque=0.7 if demand else 0.0,
    applied_torque=0.7 if demand else 0.0,
    reference_target_torque=0.75 if demand else 0.0,
    driver_torque=80.0 if driver_after else 0.0,
    steering_pressed=driver_after,
    actual_damping_amount=0.03 if damping else None,
    actual_damping_state="damping" if damping else None,
    turn_in_blocked=False if damping else None,
    breakaway_latch=0.8 if damping else None,
    sustain_floor_contribution=0.6 if damping else None,
    damping_version=2 if damping else None,
    road_confounded=road_confounded,
    vertical_accel_deviation=1.7 if road_confounded else 0.0,
  ))
  current_angle = angle
  for index in range(1, 291):
    rate = release_rate * (1.5 if index <= 10 else 1.0)
    if index <= post_sweep_ticks:
      # The post-release tail sweeps consistently with the release rate for
      # post_sweep_ticks samples, then holds the reached angle.
      current_angle += rate * 0.01
    event = detector.update(sample(
      release_time + index * 0.01,
      steering_angle_deg=current_angle,
      steering_rate_deg=rate,
      reference_rate=reference_rate,
      desired_lateral_accel=desired,
      actual_lateral_accel=actual,
      driver_torque=80.0 if driver_after else 0.0,
      steering_pressed=driver_after,
      road_confounded=road_confounded,
      vertical_accel_deviation=1.7 if road_confounded else 0.0,
    )) or event
  return event


def test_unwind_progress_deficit_detects_moving_but_lagging_wheel():
  event = unwind_progress_event()
  assert event is not None and event.evidence is not None
  assert event.event_type == "unwindProgressDeficit"
  assert event.evidence.expected_angle_progress_deg > 0.0
  assert event.evidence.unwind_progress_ratio < 0.35
  assert abs(event.evidence.measurement_rate_at_trigger) > 0.0
  assert event.evidence.analysis_window_before_s == 6.0
  assert event.evidence.analysis_window_after_s == 2.0


@pytest.mark.parametrize(("angle", "minimum_duration"), [
  (80.0, 0.8),
  (180.0, 0.6),
  (300.0, 0.4),
])
def test_unwind_progress_uses_angle_dependent_duration(angle, minimum_duration):
  event = unwind_progress_event(angle=angle)
  assert event is not None and event.evidence is not None
  # Normal-confidence events retain a two-second confirmation window after
  # the band-specific deficit duration.
  assert event.evidence.unwind_deficit_duration_s >= minimum_duration + 1.99


def test_driver_assist_immediately_confirms_unwind_deficit():
  event = unwind_progress_event(driver_assist=True)
  assert event is not None and event.evidence is not None
  assert event.event_type == "unwindProgressDeficit"
  assert event.confidence >= 0.95
  assert event.reason == "driver intervention accelerated an unwind that was behind the requested trajectory"
  assert event.evidence.driver_assisted_unwind
  assert event.evidence.driver_assist_mono_time == pytest.approx(event.detected_mono_time)
  assert event.evidence.driver_assist_torque < 0.0
  assert event.evidence.wheel_rate_increase_after_assist_deg_s > 0.0


def test_unwind_neutral_crossings_are_interpolated_and_trigger_torque_is_retained():
  event = unwind_progress_event(driver_assist=True)
  assert event is not None and event.evidence is not None
  evidence = event.evidence
  assert evidence.requested_torque_neutral_cross_mono_time == pytest.approx(0.595)
  assert evidence.applied_torque_neutral_cross_mono_time == pytest.approx(0.695)
  assert evidence.unwind_command_delay_s == pytest.approx(
    evidence.requested_torque_neutral_cross_mono_time - evidence.unwind_episode_start_mono_time,
  )
  assert evidence.unwind_requested_torque_at_trigger != event.evidence.driver_assist_torque


def test_reference_recommit_cancels_apparent_unwind():
  detector = LateralEventDetector(cooldown=0.0)
  events = []
  for index in range(350):
    now = index * 0.01
    reference_rate = 0.8 if now < 0.35 else -0.8
    event = detector.update(sample(
      now,
      steering_angle_deg=180.0,
      steering_rate_deg=-40.0,
      reference_rate=reference_rate,
      reference_sustained_unwind_scale=0.9,
      unwind_effective_phase=0.9,
    ))
    if event is not None:
      events.append(event)
  assert all(event.event_type != "unwindProgressDeficit" for event in events)


def test_unwind_road_bump_is_attribution_not_suppression():
  event = unwind_progress_event(driver_assist=True, road_confounded=True)
  assert event is not None and event.evidence is not None
  assert event.evidence.road_confounded_any


def test_legacy_stationary_late_unwind_does_not_duplicate_progress_event():
  detector = LateralEventDetector(cooldown=0.0)
  events = []
  for index in range(400):
    event = detector.update(sample(
      index * 0.01,
      steering_angle_deg=90.0,
      steering_rate_deg=0.0,
      reference_rate=0.5,
      reference_sustained_unwind_scale=0.9,
      unwind_effective_phase=0.9,
    ))
    if event is not None:
      events.append(event.event_type)
  assert events.count("lateUnwind") == 1
  assert "unwindProgressDeficit" not in events


def test_turn_stop_turn_detects_one_strong_release():
  event = turn_stop_turn_event()
  assert event is not None and event.evidence is not None
  assert event.event_type == "turnStopTurn"
  assert event.evidence.dwell_duration_s >= 0.5
  assert event.evidence.restart_classification == "sameDirectionTurn"
  assert event.evidence.analysis_window_before_s == 6.0
  assert event.evidence.analysis_window_after_s == 2.0
  assert event.occurred_mono_time == pytest.approx(event.evidence.release_mono_time)
  assert event.detected_mono_time > event.occurred_mono_time
  assert event.evidence.turn_stop_pre_dwell_progress_deg == pytest.approx(18.0)
  assert event.evidence.turn_stop_post_dwell_progress_deg == pytest.approx(21.0)
  assert event.evidence.driver_causation == DRIVER_CAUSATION_AUTONOMOUS_ONLY


def test_turn_stop_turn_detects_major_slowdown_without_sub_eight_rate():
  event = turn_stop_turn_event(
    movement_rate=200.0,
    dwell_rate=40.0,
    release_rate=90.0,
  )
  assert event is not None and event.evidence is not None
  assert event.evidence.minimum_dwell_rate_deg_s == pytest.approx(40.0)
  assert event.evidence.rate_reduction_ratio == pytest.approx(0.8)


@pytest.mark.parametrize(("movement_rate", "release_rate", "classification"), [
  (120.0, 60.0, "sameDirectionTurn"),
  (120.0, -60.0, "unwind"),
  (-120.0, 60.0, "reversal"),
])
def test_turn_stop_turn_restart_classification(movement_rate, release_rate, classification):
  event = turn_stop_turn_event(
    movement_rate=movement_rate,
    release_rate=release_rate,
  )
  assert event is not None and event.evidence is not None
  assert event.evidence.restart_classification == classification


def test_turn_stop_turn_tracks_driver_phases_and_actual_damping():
  event = turn_stop_turn_event(
    driver_before=True,
    driver_during=False,
    driver_after=True,
    damping=True,
  )
  assert event is not None and event.evidence is not None
  evidence = event.evidence
  assert evidence.driver_involved_before_dwell
  assert not evidence.driver_involved_during_dwell
  assert evidence.driver_involved_after_dwell
  assert evidence.turn_stop_actual_damping_amount == pytest.approx(0.03)
  assert evidence.turn_stop_actual_damping_state == "damping"
  assert evidence.turn_stop_breakaway_latch == pytest.approx(0.8)
  assert evidence.turn_stop_damping_version == 2


def test_turn_stop_turn_non_hyundai_optional_damping_is_absent():
  event = turn_stop_turn_event()
  assert event is not None and event.evidence is not None
  assert event.evidence.turn_stop_actual_damping_amount is None
  assert event.evidence.turn_stop_actual_damping_state is None
  assert event.evidence.turn_stop_damping_version is None


def test_turn_stop_turn_road_bump_is_retained_as_attribution():
  event = turn_stop_turn_event(road_confounded=True)
  assert event is not None and event.evidence is not None
  assert event.evidence.road_confounded_any


def test_intentional_driver_hold_without_reference_demand_is_silent():
  event = turn_stop_turn_event(
    demand=False,
    driver_before=True,
    driver_during=True,
    driver_after=True,
  )
  assert event is None


def test_steady_constant_radius_hold_does_not_emit_without_release():
  detector = LateralEventDetector(cooldown=0.0)
  detector.update(sample(
    0.0,
    steering_angle_deg=110.0,
    steering_rate_deg=80.0,
    reference_rate=0.0,
    desired_lateral_accel=0.5,
    actual_lateral_accel=0.5,
  ))
  for index in range(1, 501):
    event = detector.update(sample(
      index * 0.01,
      steering_angle_deg=110.0,
      steering_rate_deg=0.0,
      reference_rate=0.0,
      desired_lateral_accel=0.5,
      actual_lateral_accel=0.5,
    ))
    assert event is None


def test_constant_radius_pause_then_release_without_reference_motion_is_silent():
  assert turn_stop_turn_event(demand=False) is None


def test_near_center_pause_and_tiny_corrections_are_silent():
  for angle in (10.0, 24.0):
    detector = LateralEventDetector(cooldown=0.0)
    detector.update(sample(
      0.0,
      steering_angle_deg=angle,
      steering_rate_deg=60.0,
      reference_rate=0.0,
      desired_lateral_accel=0.0,
      actual_lateral_accel=0.0,
    ))
    for index in range(1, 80):
      assert detector.update(sample(
        index * 0.01,
        steering_angle_deg=angle,
        steering_rate_deg=0.0,
        reference_rate=0.0,
        desired_lateral_accel=0.0,
        actual_lateral_accel=0.0,
      )) is None
    assert detector.update(sample(
      0.81,
      steering_angle_deg=angle,
      steering_rate_deg=50.0,
      reference_rate=0.0,
      desired_lateral_accel=0.0,
      actual_lateral_accel=0.0,
    )) is None


def test_short_weak_turn_stop_dwell_is_silent():
  assert turn_stop_turn_event(
    angle=40.0,
    dwell_s=0.2,
    demand=False,
  ) is None


# --- v6: turn-stop max-dwell lifetime (design.md section 2 rule 1) ---------


def test_turn_stop_dwell_past_max_never_emits():
  event = turn_stop_turn_event(dwell_s=TURN_STOP_MAX_DWELL_S + 0.45)
  batch = event if isinstance(event, (list, tuple)) else (event,)
  assert not any(item is not None and item.event_type == "turnStopTurn" for item in batch)


def test_turn_stop_dwell_just_under_max_with_valid_release_emits():
  event = turn_stop_turn_event(dwell_s=TURN_STOP_MAX_DWELL_S - 0.05)
  assert event is not None and event.evidence is not None
  assert event.event_type == "turnStopTurn"
  assert event.evidence.dwell_duration_s == pytest.approx(TURN_STOP_MAX_DWELL_S - 0.05)
  assert event.evidence.turn_stop_post_dwell_progress_deg >= TURN_STOP_MIN_POST_PROGRESS_DEG


# --- v6: crown-neutral-aware unwind timing (design.md section 1) ----------


def crown_offset_unwind_event(*, angle: float, unwind_neutral_torque: float = 0.0):
  """Like unwind_progress_event, but request/applied torque ramp linearly
  through the old-turn direction instead of stepping, so a nonzero crown
  offset measurably shifts the crown-neutral-reach timestamp away from the
  literal-zero torque crossing (which stays fixed at the ramp midpoint)."""
  detector = LateralEventDetector(cooldown=0.0)
  expected_direction = -1.0 if angle > 0.0 else 1.0
  reference_rate = -expected_direction
  sign = 1.0 if angle > 0.0 else -1.0
  detector.update(sample(
    0.0,
    steering_angle_deg=angle - expected_direction * 10.0,
    steering_rate_deg=-expected_direction * 300.0,
    reference_rate=0.0,
  ))
  event = None
  for index in range(1, 321):
    now = round(index * 0.01, 2)
    ramp = min(1.0, max(0.0, now - 0.2))
    torque = (1.0 - 2.0 * ramp) * sign
    event = detector.update(sample(
      now,
      steering_angle_deg=angle,
      steering_rate_deg=expected_direction * 100.0,
      reference_rate=reference_rate,
      measurement_rate=-0.2 * reference_rate,
      reference_sustained_unwind_scale=0.9,
      unwind_effective_phase=0.9,
      request_torque=torque,
      applied_torque=torque,
      reference_target_torque=-0.9 * sign,
      unwind_neutral_torque=unwind_neutral_torque * sign,
    ))
    if event is not None:
      return event
  return event


def test_crown_neutral_timestamps_symmetric_across_mirrored_turns():
  left = crown_offset_unwind_event(angle=300.0, unwind_neutral_torque=0.2)
  right = crown_offset_unwind_event(angle=-300.0, unwind_neutral_torque=0.2)
  assert left is not None and left.evidence is not None
  assert right is not None and right.evidence is not None
  assert left.evidence.requested_crown_neutral_mono_time == pytest.approx(
    right.evidence.requested_crown_neutral_mono_time,
  )
  assert left.evidence.applied_crown_neutral_mono_time == pytest.approx(
    right.evidence.applied_crown_neutral_mono_time,
  )
  # A nonzero crown offset pulls the crown-reach point away from the
  # literal-zero torque crossing (fixed at the ramp midpoint, ~0.70).
  assert abs(
    left.evidence.requested_crown_neutral_mono_time
    - left.evidence.requested_torque_neutral_cross_mono_time,
  ) > 0.05
  assert abs(
    left.evidence.applied_crown_neutral_mono_time
    - left.evidence.applied_torque_neutral_cross_mono_time,
  ) > 0.05


def test_crown_neutral_matches_literal_zero_when_neutral_is_zero():
  # No-assist path (assist would fire at 0.75, before the applied crown's
  # 0.10s hold completes at ~0.80 -- A13: applied-crown presence is only
  # guaranteed for no-assist keeps).
  event = unwind_progress_event()
  assert event is not None and event.evidence is not None
  evidence = event.evidence
  assert evidence.requested_crown_neutral_mono_time == pytest.approx(
    evidence.requested_torque_neutral_cross_mono_time, abs=0.015,
  )
  assert evidence.applied_crown_neutral_mono_time == pytest.approx(
    evidence.applied_torque_neutral_cross_mono_time, abs=0.015,
  )


# --- v6: old-torque rebound tracker (design.md section 1.5) ----------------


def unwind_rebound_event(*, angle: float = 300.0, rebound: bool = True):
  """Requested torque releases past crown-neutral (old_turn_sign=+1, neutral=0
  for angle=300.0) and, when rebound=True, re-excurses into the old-turn
  direction in two damped cycles (~0.5 then ~0.2 normalized, ~0.35s apart)
  before settling for good."""
  detector = LateralEventDetector(cooldown=0.0)
  expected_direction = -1.0 if angle > 0.0 else 1.0
  reference_rate = -expected_direction
  detector.update(sample(
    0.0,
    steering_angle_deg=angle - expected_direction * 10.0,
    steering_rate_deg=-expected_direction * 300.0,
    reference_rate=0.0,
  ))
  event = None
  for index in range(1, 321):
    now = round(index * 0.01, 2)
    if now < 0.60:
      torque = 0.8
    elif now < 0.62:
      torque = -0.30
    elif rebound and now < 0.66:
      torque = 0.50
    elif now < 0.95:
      torque = -0.30
    elif rebound and now < 0.98:
      torque = 0.20
    else:
      torque = -0.8
    event = detector.update(sample(
      now,
      steering_angle_deg=angle,
      steering_rate_deg=expected_direction * 100.0,
      reference_rate=reference_rate,
      measurement_rate=-0.2 * reference_rate,
      reference_sustained_unwind_scale=0.9,
      unwind_effective_phase=0.9,
      request_torque=torque,
      applied_torque=torque,
      reference_target_torque=-0.9,
    ))
    if event is not None:
      return event
  return event


def test_unwind_rebound_cycles_are_tracked():
  event = unwind_rebound_event(rebound=True)
  assert event is not None and event.evidence is not None
  evidence = event.evidence
  assert evidence.unwind_rebound_start_mono_time == pytest.approx(0.62)
  assert evidence.unwind_rebound_max_magnitude == pytest.approx(0.50)
  assert evidence.unwind_rebound_max_magnitude > UNWIND_REBOUND_MIN_COMPONENT
  assert evidence.unwind_rebound_duration_s > 0.0
  assert evidence.unwind_rebound_same_episode


def test_unwind_clean_release_records_no_rebound():
  event = unwind_rebound_event(rebound=False)
  assert event is not None and event.evidence is not None
  evidence = event.evidence
  assert evidence.unwind_rebound_start_mono_time is None
  assert evidence.unwind_rebound_max_magnitude == 0.0
  assert evidence.unwind_rebound_duration_s == 0.0


def unwind_rebound_without_genuine_release_event(*, angle: float = 300.0):
  """Episode arms already near crown-neutral (requested torque starts at
  0.0, well inside CROWN_NEUTRAL_TOLERANCE) -- there is no genuine prior
  old-direction hold to release from. A single transient torque bump
  (ordinary control-loop chatter, not a real hold-then-release) crosses
  UNWIND_REBOUND_MIN_COMPONENT briefly around t=0.60-0.62 before subsiding
  back to zero for the rest of the episode."""
  detector = LateralEventDetector(cooldown=0.0)
  expected_direction = -1.0 if angle > 0.0 else 1.0
  reference_rate = -expected_direction
  detector.update(sample(
    0.0,
    steering_angle_deg=angle - expected_direction * 10.0,
    steering_rate_deg=-expected_direction * 300.0,
    reference_rate=0.0,
    request_torque=0.0,
    applied_torque=0.0,
  ))
  event = None
  for index in range(1, 321):
    now = round(index * 0.01, 2)
    torque = 0.15 if 0.60 <= now < 0.62 else 0.0
    event = detector.update(sample(
      now,
      steering_angle_deg=angle,
      steering_rate_deg=expected_direction * 100.0,
      reference_rate=reference_rate,
      measurement_rate=-0.2 * reference_rate,
      reference_sustained_unwind_scale=0.9,
      unwind_effective_phase=0.9,
      request_torque=torque,
      applied_torque=torque,
      reference_target_torque=-0.9,
    ))
    if event is not None:
      return event
  return event


def test_rebound_does_not_arm_without_a_genuine_prior_hold():
  # Fix 3 (minor, confirmed): rebound_armed_mono_time used to be set the
  # FIRST time requested_component dropped below CROWN_NEUTRAL_TOLERANCE,
  # with no check that old-direction torque was ever genuinely held above
  # tolerance first. An episode that arms already near crown-neutral would
  # then misrepresent a later transient chatter bump (never a real hold then
  # release) as a "rebound". Pre-fix this scenario would report
  # unwind_rebound_start_mono_time ~= 0.60 and unwind_rebound_max_magnitude
  # ~= 0.15; post-fix, since old-direction torque was never held above
  # tolerance before that bump, no rebound is armed in time to record it.
  event = unwind_rebound_without_genuine_release_event()
  assert event is not None and event.evidence is not None
  evidence = event.evidence
  assert evidence.unwind_rebound_start_mono_time is None
  assert evidence.unwind_rebound_max_magnitude == 0.0
  assert evidence.unwind_rebound_duration_s == 0.0


# --- v6 fix: crown/rebound bookkeeping must not be skipped when a
# higher-priority detector (crossing/handoff) claims the emission slot ------


def test_crown_rebound_bookkeeping_advances_on_a_sample_claimed_by_handoff():
  # Fix 2 (major, empirically reproduced): update() ran a priority chain
  # (_update_pending_crossing, then _update_pending_handoff, then
  # _update_unwind_progress) where at most one detector's update path ran
  # per sample. The crown-neutral/rebound tracker lives entirely inside
  # _update_unwind_progress, so whenever a pending crossing/handoff resolved
  # on a sample where an unwind episode was ALSO active, the crown/rebound
  # bookkeeping (and the previous_mono_time/previous_request_torque/
  # previous_applied_torque advance it depends on) silently skipped that
  # sample -- corrupting the next real invocation's delta.
  #
  # This constructs the shared-sample scenario directly (an already-mid-
  # flight UnwindProgressEpisode plus a PendingHandoff about to resolve) so
  # the exact sample-skip can be pinned deterministically. Pre-fix, this
  # test fails: nothing below ever runs `_update_unwind_crown_tracking` for
  # the handoff-claimed sample, so `previous_mono_time` stays stuck at 10.00
  # and `rebound_duration_s` stays 0.0 (the genuine 0.01s the signal spent
  # above UNWIND_REBOUND_MIN_COMPONENT between 10.00 and 10.01 is dropped
  # entirely, on both the shared sample and the following real invocation).
  detector = LateralEventDetector(cooldown=0.0)
  episode = UnwindProgressEpisode(
    start_mono_time=9.0,
    initial_steering_angle_deg=300.0,
    peak_steering_angle_deg=300.0,
    expected_direction=-1,
    expected_rate_deg_s=112.5,
    episode_start_snapshot=9.0,
    old_turn_sign=1.0,
    previous_request_torque=0.15,
    previous_applied_torque=0.15,
    previous_mono_time=10.00,
    recent_toward_rates=deque(),
    rebound_armed_mono_time=9.50,
    rebound_start_mono_time=9.55,
    rebound_max_magnitude=0.15,
    rebound_same_episode=True,
    rebound_previous_above=True,
    held_old_direction_before_release=True,
  )
  detector.unwind_progress_episode = episode
  detector.pending_handoff = PendingHandoff(
    mono_time=9.50,
    episode_start=9.0,
    previous_phase=0.0,
    previous_same_episode=False,
    peak_abs_rate_deg=100.0,
    peak_tracking_error=0.5,
    peak_applied_target_gap=0.3,
  )

  # Shared sample: a handoff resolves (claims the emission slot) while the
  # unwind episode is genuinely still mid-rebound (component 0.15 stays
  # above UNWIND_REBOUND_MIN_COMPONENT=0.10).
  shared = sample(
    10.01,
    steering_angle_deg=300.0,
    steering_rate_deg=-100.0,
    reference_rate=1.0,
    unwind_effective_phase=0.9,
    reference_sustained_unwind_scale=0.9,
    request_torque=0.15,
    applied_torque=0.15,
    unwind_neutral_torque=0.0,
  )
  detection = detector.update(shared)
  assert detection is not None and detection.event_type == "handoffMismatch"

  assert episode.previous_mono_time == pytest.approx(10.01)
  assert episode.rebound_duration_s == pytest.approx(0.01)
  assert episode.rebound_previous_above

  # The next (uncontested) invocation drops back below the rebound
  # threshold. The total rebound_duration_s must already reflect the shared
  # sample's contribution -- it must NOT change here, and previous_mono_time
  # must advance from the shared sample's time, not from the stale 10.00.
  followup = sample(
    10.02,
    steering_angle_deg=300.0,
    steering_rate_deg=-100.0,
    reference_rate=1.0,
    unwind_effective_phase=0.9,
    reference_sustained_unwind_scale=0.9,
    request_torque=0.02,
    applied_torque=0.02,
    unwind_neutral_torque=0.0,
  )
  detector.update(followup)
  assert episode.rebound_duration_s == pytest.approx(0.01)
  assert not episode.rebound_previous_above
  assert episode.previous_mono_time == pytest.approx(10.02)


# --- v6: driver causation vs rescue (design.md section 6, amendment A5) ---


def test_pressed_assist_rescue_is_intervention_backed():
  event = unwind_progress_event(driver_assist=True)
  assert event is not None and event.evidence is not None
  assert event.evidence.driver_causation == DRIVER_CAUSATION_INTERVENTION_BACKED
  assert event.evidence.driver_assisted_unwind
  assert event.confidence == pytest.approx(0.95)
  assert not event.evidence.driver_assist_raw_torque_only


def test_raw_torque_assist_rescue_is_intervention_backed_raw_only():
  event = unwind_progress_event(assist_mode="raw", assist_at=0.75)
  assert event is not None and event.evidence is not None
  assert event.evidence.driver_causation == DRIVER_CAUSATION_INTERVENTION_BACKED
  assert event.evidence.driver_assist_raw_torque_only


def test_sustained_driver_torque_throughout_is_driver_created():
  event = unwind_progress_event(sustained_driver_torque=80.0)
  assert event is not None and event.evidence is not None
  assert event.evidence.driver_causation == DRIVER_CAUSATION_DRIVER_CREATED
  assert event.evidence.driver_confounded_any


def test_rescue_shortly_after_deficit_start_is_still_intervention_backed():
  # A5: driver_active_at_deficit_start only looks BACKWARD from deficit
  # start. Find where the deficit starts for the default no-assist profile,
  # then land the press just after it (well inside DRIVER_AT_START_WINDOW_S)
  # to prove the one-sided window does not misclassify a near-immediate
  # rescue as "active at start".
  probe = unwind_progress_event()
  assert probe is not None and probe.evidence is not None
  deficit_start = probe.evidence.unwind_deficit_start_mono_time
  assist_at = round(deficit_start + DRIVER_AT_START_WINDOW_S / 2.0, 2)
  assert deficit_start < assist_at <= deficit_start + DRIVER_AT_START_WINDOW_S

  event = unwind_progress_event(assist_mode="pressed", assist_at=assist_at)
  assert event is not None and event.evidence is not None
  assert not event.evidence.driver_active_at_deficit_start
  assert event.evidence.driver_causation == DRIVER_CAUSATION_INTERVENTION_BACKED


# --- v6: road confounding placement (design.md section 7, amendment A7) ---


def test_road_confounded_assist_deficit_still_emits_at_full_confidence():
  event = unwind_progress_event(driver_assist=True, road_confounded=True)
  assert event is not None and event.evidence is not None
  assert event.event_type == "unwindProgressDeficit"
  assert event.confidence == pytest.approx(0.95)
  assert event.evidence.road_confounded_fraction > 0.0
  assert event.evidence.max_vertical_accel_deviation > 0.0


def test_road_confounded_non_assist_deficit_records_metrics_without_penalty():
  # The road confidence penalty (A7) is applied daemon-side in Stage B, not
  # in the detector; the no-assist literal confidence (0.90) is unchanged.
  event = unwind_progress_event(road_confounded=True)
  assert event is not None and event.evidence is not None
  assert event.confidence == pytest.approx(0.90)
  assert event.evidence.road_confounded_fraction > 0.0
  assert event.evidence.max_vertical_accel_deviation > 0.0


# --- v6: episode identity, re-arm guard, lifetime cap (section 3, A2) -----


def test_turn_stop_event_keeps_snapshot_key_after_wheel_recenters():
  # The dwell/release is identical to the default turn_stop_turn_event
  # scenario (movement_rate=120, angle=120, dwell_s=0.6) up to release, then
  # the wheel is swept straight back through center and held there long
  # enough to close the detector's live episode BEFORE the pending
  # candidate's 2.1s cluster deadline elapses. The emitted evidence must
  # still carry the key that was live when the turn-stop episode was
  # created (occurrence time), not the live (closed) state at emission.
  detector = LateralEventDetector(cooldown=0.0)
  angle = 120.0
  movement_rate = 120.0
  movement_count = 15
  for index in range(movement_count):
    detector.update(sample(
      index * 0.01,
      steering_angle_deg=angle - movement_rate * (movement_count - index) * 0.01,
      steering_rate_deg=movement_rate,
      reference_rate=0.5,
      desired_lateral_accel=0.5,
      actual_lateral_accel=0.0,
    ))
  dwell_count = 60
  for index in range(dwell_count):
    detector.update(sample(
      (movement_count + index) * 0.01,
      steering_angle_deg=angle,
      steering_rate_deg=0.0,
      reference_rate=0.5,
      desired_lateral_accel=0.5,
      actual_lateral_accel=0.0,
      request_torque=0.9,
      applied_torque=0.85,
      reference_target_torque=0.8,
    ))
  release_time = (movement_count + dwell_count) * 0.01
  event = detector.update(sample(
    release_time,
    steering_angle_deg=angle,
    steering_rate_deg=60.0,
    reference_rate=0.5,
    desired_lateral_accel=0.5,
    actual_lateral_accel=0.0,
    request_torque=0.7,
    applied_torque=0.7,
    reference_target_torque=0.75,
  ))
  assert event is None
  assert detector.episode_start == pytest.approx(0.0)

  # Sweep straight back through center (large post-release progress, easily
  # clearing TURN_STOP_MIN_POST_PROGRESS_DEG) then hold at 0 well past
  # EPISODE_CENTER_CLOSE_S so the live episode closes.
  t = release_time
  for index in range(1, 41):
    t = round(release_time + index * 0.01, 2)
    event = detector.update(sample(
      t,
      steering_angle_deg=angle - 3.0 * index,
      steering_rate_deg=-300.0,
      reference_rate=0.0,
      desired_lateral_accel=0.0,
      actual_lateral_accel=0.0,
    )) or event
  hold_ticks = int(round((EPISODE_CENTER_CLOSE_S + 0.3) / 0.01))
  for _ in range(hold_ticks):
    t = round(t + 0.01, 2)
    event = detector.update(sample(
      t,
      steering_angle_deg=0.0,
      steering_rate_deg=0.0,
      reference_rate=0.0,
      desired_lateral_accel=0.0,
      actual_lateral_accel=0.0,
    )) or event
  assert detector.episode_start is None  # live episode closed before deadline

  deadline = release_time + TURN_STOP_CLUSTER_SELECTION_S
  while t < deadline + 0.05:
    t = round(t + 0.01, 2)
    event = detector.update(sample(
      t,
      steering_angle_deg=0.0,
      steering_rate_deg=0.0,
      reference_rate=0.0,
      desired_lateral_accel=0.0,
      actual_lateral_accel=0.0,
    )) or event

  assert event is not None and event.evidence is not None
  assert event.event_type == "turnStopTurn"
  assert event.evidence.episode_start_mono_time == pytest.approx(0.0)
  assert event.evidence.episode_key == "lat:0"
  assert detector.episode_start is None  # confirms the live state had moved on


def test_episode_rearm_guard_blocks_tracking_active_alone_near_center():
  detector = LateralEventDetector(cooldown=0.0)
  detector.update(sample(0.0, steering_angle_deg=30.0, steering_rate_deg=60.0, reference_rate=0.0))
  assert detector.episode_start == pytest.approx(0.0)

  close_ticks = int(round((EPISODE_CENTER_CLOSE_S + 0.1) / 0.01))
  now = 0.0
  for index in range(1, close_ticks + 1):
    now = round(index * 0.01, 2)
    detector.update(sample(now, steering_angle_deg=0.0, steering_rate_deg=0.0, reference_rate=0.0))
  assert detector.episode_start is None

  # tracking_active alone (reference_rate spike) must NOT re-arm while the
  # wheel stays centered (A2 churn guard) -- true on every sample, not just
  # the last one.
  for index in range(close_ticks + 1, close_ticks + 61):
    now = round(index * 0.01, 2)
    detector.update(sample(now, steering_angle_deg=0.0, steering_rate_deg=0.0, reference_rate=0.5))
    assert detector.episode_start is None

  # Real motion resumes: a fresh episode identity is stamped.
  now = round(now + 0.01, 2)
  detector.update(sample(now, steering_angle_deg=30.0, steering_rate_deg=60.0, reference_rate=0.0))
  assert detector.episode_start == pytest.approx(now)


def test_episode_lifetime_cap_forces_reset_after_max_lifetime():
  detector = LateralEventDetector(cooldown=0.0)
  detector.update(sample(0.0, steering_angle_deg=30.0, steering_rate_deg=0.0, reference_rate=0.0))
  first_start = detector.episode_start
  assert first_start == pytest.approx(0.0)

  now = 0.0
  ticks_to_boundary = int(round(EPISODE_MAX_LIFETIME_S / 0.1))
  for index in range(1, ticks_to_boundary + 1):
    now = round(index * 0.1, 1)
    detector.update(sample(now, steering_angle_deg=30.0, steering_rate_deg=0.0, reference_rate=0.0))
    assert detector.episode_start == pytest.approx(first_start)

  now = round(now + 0.1, 1)
  assert now > EPISODE_MAX_LIFETIME_S
  detector.update(sample(now, steering_angle_deg=30.0, steering_rate_deg=0.0, reference_rate=0.0))
  assert detector.episode_start == pytest.approx(now)
  assert detector.episode_start != pytest.approx(first_start)


# --- v6: route discontinuity reset (section 2 rule 5, amendment A8) -------


def test_backwards_discontinuity_resets_cooldown_and_history_with_default_cooldown():
  # Uses the DEFAULT cooldown (not cooldown=0.0): without reset_discontinuity()
  # clearing last_event_times, a backwards jump leaves a future-stamped
  # cooldown entry that would suppress this event type forever (every later
  # delta stays negative, which always satisfies "< cooldown").
  detector = LateralEventDetector()
  event = None
  for index in range(102):
    t = round(699.0 + index * 0.01, 2)
    event = detector.update(sample(
      t,
      steering_angle_deg=90.0,
      steering_rate_deg=0.0,
      reference_rate=0.5,
      reference_sustained_unwind_scale=0.9,
      unwind_effective_phase=0.9,
    )) or event
  assert event is not None and event.event_type == "lateUnwind"
  assert event.occurred_mono_time == pytest.approx(700.01)

  event2 = None
  for index in range(102):
    t = round(60.0 + index * 0.01, 2)
    event2 = detector.update(sample(
      t,
      steering_angle_deg=90.0,
      steering_rate_deg=0.0,
      reference_rate=0.5,
      reference_sustained_unwind_scale=0.9,
      unwind_effective_phase=0.9,
    )) or event2
  assert event2 is not None and event2.event_type == "lateUnwind"
  assert event2.occurred_mono_time == pytest.approx(61.01)
  # No pre-gap history leaked into the post-gap evidence window.
  assert event2.evidence is not None
  assert event2.evidence.evidence_start_mono_time == pytest.approx(60.0)
  assert event2.evidence.evidence_start_mono_time >= 60.0


# --- v6: road-evidence window clamped to retained history (section 7, A6) -


def test_road_evidence_window_clamped_to_trimmed_history(monkeypatch):
  # Fix 4 / amendment A6 (test-coverage gap): road_evidence_window_start_
  # mono_time must be clamped to max(min(episode_start_snapshot,
  # evidence_start), history[0][0]) -- the serialized window must never
  # claim coverage the retained history ring didn't actually have. Shrinking
  # EVIDENCE_HISTORY_S makes the ring actually TRIM mid-scenario (not merely
  # run out from route start, which every early-route test already exercises
  # trivially): the naive window start here is episode_start_mono_time=0.0,
  # but by the time the no-assist deficit fires (~2.5s later) the trimmed
  # history ring only goes back to ~1.52s, so the clamp must win.
  monkeypatch.setattr(lat_event_detector, "EVIDENCE_HISTORY_S", 1.0)
  event = unwind_progress_event()
  assert event is not None and event.evidence is not None
  evidence = event.evidence
  assert evidence.episode_start_mono_time == pytest.approx(0.0)

  naive_window_start = min(evidence.episode_start_mono_time, event.occurred_mono_time - 6.0)
  assert naive_window_start < evidence.episode_start_mono_time
  assert evidence.road_evidence_window_start_mono_time is not None
  assert evidence.road_evidence_window_start_mono_time > evidence.episode_start_mono_time
  assert evidence.road_evidence_window_start_mono_time != pytest.approx(naive_window_start)
  assert evidence.road_evidence_window_start_mono_time == pytest.approx(1.52)

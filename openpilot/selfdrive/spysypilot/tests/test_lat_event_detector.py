import pytest

from openpilot.selfdrive.spysypilot.lat_event_detector import (
  DETECTOR_VERSION,
  DRIVER_CONFOUND_STEERING_PRESSED,
  DRIVER_CONFOUND_TORQUE,
  DRIVER_INTERACTION_CONFIRMED_STEERING_PRESSED,
  DRIVER_INTERACTION_POSSIBLE_RAW_TORQUE,
  LateralEventDetector,
  LateralSample,
  ROAD_INTERACTION_SUBSTANTIAL,
  ROAD_INTERACTION_TRANSIENT,
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


def test_lateral_detector_version_four():
  assert DETECTOR_VERSION == 4


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

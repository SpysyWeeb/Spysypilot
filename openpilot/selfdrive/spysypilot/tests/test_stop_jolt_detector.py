from collections.abc import Callable

import pytest

from openpilot.selfdrive.spysypilot.stop_jolt_detector import (
  StopJoltCarSample,
  StopJoltDetector,
  StopJoltEvent,
  StopJoltImuSample,
)


AccelProfile = Callable[[float], float]


def smooth_accel(t: float) -> float:
  if t < 2.4:
    return -0.65
  if t < 3.0:
    return -0.65 * (3.0 - t) / 0.6
  return 0.0


def brake_grab_accel(t: float) -> float:
  if t < 2.35:
    return -0.4
  if t < 2.75:
    return -1.65
  if t < 3.0:
    return -0.55
  return 0.0


def release_snap_accel(t: float) -> float:
  if t < 3.0:
    return -0.55
  if t < 3.25:
    return 0.85
  return 0.0


def run_stop(
  car_accel: AccelProfile,
  imu_accel: AccelProfile | None = None,
  *,
  end: float = 4.0,
  driver_brake_at: float | None = None,
  road_at: float | None = None,
  imu_valid: bool = True,
  stop_at: float = 3.0,
  should_stop_at: float = 0.0,
  invalid_at: float | None = None,
) -> list[StopJoltEvent]:
  detector = StopJoltDetector()
  events: list[StopJoltEvent] = []
  imu_accel = car_accel if imu_accel is None else imu_accel
  for frame in range(round(end * 100) + 1):
    t = frame / 100.0
    v_ego = max(0.0, 2.0 * (1.0 - min(t, stop_at) / stop_at))
    stopped = t >= stop_at
    road_confounded = road_at is not None and abs(t - road_at) <= 0.35
    if frame % 5 == 0:
      detector.update_imu(StopJoltImuSample(
        t=t,
        accel_x=imu_accel(t),
        valid=imu_valid,
        road_confounded=road_confounded,
      ))
    event = detector.update_car(StopJoltCarSample(
      t=t,
      required_valid=invalid_at is None or t < invalid_at,
      long_active=True,
      should_stop=t >= should_stop_at,
      v_ego=v_ego,
      a_ego=car_accel(t),
      standstill=stopped,
      brake_pressed=driver_brake_at is not None and t >= driver_brake_at,
      gas_pressed=False,
      brake_hold_active=stopped,
      plan_a_target=car_accel(t),
      requested_accel=car_accel(t),
      applied_accel=car_accel(t),
      long_control_state="stopping" if not stopped else "pid",
      lead_present=True,
      d_rel=4.0,
      v_lead_k=0.0,
      radar_valid=True,
      road_confounded=road_confounded,
    ))
    if event is not None:
      events.append(event)
  return events


def test_smooth_tapered_stop_remains_silent():
  assert run_stop(smooth_accel) == []


def test_abrupt_final_brake_grab_triggers():
  events = run_stop(brake_grab_accel)
  assert len(events) == 1
  assert events[0].classification in ("brakeGrab", "grabAndRebound")
  assert events[0].imu_jerk < 0.0
  assert events[0].a_ego_jerk < 0.0


def test_approach_history_survives_until_late_should_stop_arming():
  events = run_stop(brake_grab_accel, should_stop_at=2.7)
  assert len(events) == 1
  assert events[0].episode_start_mono_time < 1.0


def test_abrupt_release_rebound_triggers():
  events = run_stop(release_snap_accel)
  assert len(events) == 1
  assert events[0].classification == "releaseSnap"
  assert events[0].imu_jerk > 0.0
  assert events[0].a_ego_jerk > 0.0


def test_imu_dominant_delayed_release_shape_from_replay_triggers():
  def delayed_a_ego(t: float) -> float:
    return -0.3 if t < 3.1 else 0.1

  def imu_rebound(t: float) -> float:
    return -0.5 if t < 3.0 else 0.7

  events = run_stop(delayed_a_ego, imu_rebound)
  assert len(events) == 1
  assert events[0].classification == "releaseSnap"
  assert events[0].imu_jerk >= 3.0
  assert 0.75 <= events[0].a_ego_jerk < 2.5
  assert events[0].confidence == pytest.approx(0.82)


def test_one_frame_acceleration_spike_remains_silent():
  def spike(t: float) -> float:
    return -2.5 if abs(t - 2.5) < 0.005 else -0.4

  assert run_stop(spike, spike) == []


def test_wheel_speed_estimator_snap_without_imu_confirmation_remains_silent():
  assert run_stop(brake_grab_accel, smooth_accel) == []


def test_strict_a_ego_fallback_works_when_imu_unavailable():
  events = run_stop(brake_grab_accel, imu_valid=False)
  assert len(events) == 1
  assert not events[0].imu_valid
  assert events[0].confidence < 0.8


def test_driver_braking_does_not_create_smooth_stops_event():
  assert run_stop(brake_grab_accel, driver_brake_at=2.3) == []


def test_required_message_invalidity_blocks_the_episode():
  assert run_stop(brake_grab_accel, invalid_at=2.3) == []


def test_vertical_road_bump_suppresses_warning_level_event():
  def warning_jolt(t: float) -> float:
    return -0.3 if t < 2.0 else (-1.5 if t < 2.35 else -0.2)

  unconfounded = run_stop(warning_jolt)
  assert len(unconfounded) == 1
  assert unconfounded[0].severity == "warning"
  assert run_stop(warning_jolt, road_at=2.5) == []


def test_severe_road_confounded_event_is_retained_with_reduced_confidence():
  def severe(t: float) -> float:
    return -0.4 if t < 2.35 else (-2.4 if t < 2.75 else 0.0)

  events = run_stop(severe, road_at=2.5)
  assert len(events) == 1
  assert events[0].road_confounded
  assert events[0].severity == "critical"
  assert events[0].confidence < 0.9


def test_hard_braking_above_landing_speed_window_remains_silent():
  def high_speed_brake(t: float) -> float:
    if 0.2 <= t < 0.6:
      return -2.5
    return smooth_accel(t)

  assert run_stop(high_speed_brake) == []


def test_creeping_queue_does_not_finalize_before_true_standstill():
  detector = StopJoltDetector()
  event = None
  for frame in range(500):
    t = frame / 100.0
    v_ego = max(0.12, 2.0 - t)
    accel = brake_grab_accel(min(t, 2.7))
    if frame % 5 == 0:
      detector.update_imu(StopJoltImuSample(t, accel, True))
    event = detector.update_car(StopJoltCarSample(
      t, True, True, True, v_ego, accel, False, False, False, False,
      accel, accel, accel, "stopping", True, 3.0, 0.0, True,
    ))
  assert event is None

  events = []
  for frame in range(500, 580):
    t = frame / 100.0
    accel = 0.0
    if frame % 5 == 0:
      detector.update_imu(StopJoltImuSample(t, accel, True))
    event = detector.update_car(StopJoltCarSample(
      t, True, True, True, 0.0, accel, True, False, False, True,
      accel, accel, accel, "pid", True, 3.0, 0.0, True,
    ))
    if event is not None:
      events.append(event)
  assert len(events) == 1


def test_only_one_event_is_emitted_per_stop():
  assert len(run_stop(brake_grab_accel, end=6.0)) == 1


def test_peak_occurrence_precedes_later_detection():
  event = run_stop(brake_grab_accel)[0]
  assert event.peak_jolt_mono_time < event.standstill_mono_time
  assert event.detection_mono_time >= event.standstill_mono_time + 0.45


def test_duplicate_held_imu_timestamp_is_ignored():
  detector = StopJoltDetector()
  detector.update_imu(StopJoltImuSample(1.0, -0.5, True))
  detector.update_imu(StopJoltImuSample(1.0, -9.0, True))
  assert len(detector.imu_history) == 1
  assert detector.imu_history[0].accel_x == pytest.approx(-0.5)

from __future__ import annotations

from dataclasses import replace

from openpilot.common.parameterized import parameterized
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.spysypilot.rolling_lead_response_detector import (
  EPISODE_COOLDOWN_S,
  RollingLeadResponseDetector,
  RollingLeadSample,
)


DT = 0.05


def sample(t: float, **kwargs) -> RollingLeadSample:
  defaults = {
    "long_active": True,
    "v_ego": 8.0,
    "a_ego": 0.0,
    "gas_pressed": False,
    "brake_pressed": False,
    "lead_present": True,
    "lead_valid": True,
    "radar_valid": True,
    "radar_track_id": 17,
    "d_rel": 20.0,
    "v_lead": 8.0,
    "v_lead_k": 8.0,
    "a_lead_k": 0.0,
    "plan_valid": True,
    "a_target": -0.5,
    "should_stop": False,
    "output_valid": True,
    "output_accel": -0.5,
  }
  defaults.update(kwargs)
  return RollingLeadSample(t=t, **defaults)


def accel_motion(t: float, onset: float, accel: float) -> tuple[float, float]:
  elapsed = max(0.0, t - onset)
  return accel * elapsed, 0.5 * accel * elapsed * elapsed


def pullaway_sample(
  t: float,
  *,
  lead_onset: float = 1.5,
  lead_accel: float = 2.5,
  plan_onset: float | None = 2.8,
  output_onset: float | None = 2.9,
  ego_onset: float | None = 3.0,
  ego_accel: float = 1.0,
  track_id: int = 17,
  **kwargs,
) -> RollingLeadSample:
  lead_dv, lead_dx = accel_motion(t, lead_onset, lead_accel)
  ego_dv, ego_dx = (0.0, 0.0) if ego_onset is None else accel_motion(t, ego_onset, ego_accel)
  values = {
    "v_ego": 8.0 + ego_dv,
    "a_ego": ego_accel if ego_onset is not None and t >= ego_onset else 0.0,
    "radar_track_id": track_id,
    "d_rel": 20.0 + lead_dx - ego_dx,
    "v_lead": 8.0 + lead_dv,
    "v_lead_k": 8.0 + lead_dv,
    "a_lead_k": lead_accel if t >= lead_onset else 0.0,
    "a_target": 0.6 if plan_onset is not None and t >= plan_onset else -0.5,
    "output_accel": 0.6 if output_onset is not None and t >= output_onset else -0.5,
  }
  values.update(kwargs)
  return sample(t, **values)


def run(detector: RollingLeadResponseDetector, end: float, state_fn, start: float = 0.0):
  events = []
  count = round((end - start) / DT)
  for index in range(count + 1):
    t = start + index * DT
    event = detector.update(state_fn(t))
    if event is not None:
      events.append(event)
  return events


class TestRollingLeadResponseDetector(OpenpilotTestCase):
  def test_planner_late_while_gap_opens(self):
    detector = RollingLeadResponseDetector()
    events = run(detector, 5.0, lambda t: pullaway_sample(t, plan_onset=2.8, output_onset=2.9, ego_onset=3.0))

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "late_rolling_lead_response_planner"
    assert "planner response" in event.attribution_detail
    assert event.lead_to_plan_s is not None and event.lead_to_plan_s > 1.0
    assert event.max_gap_growth >= 3.0
    assert event.detected_mono_time > event.occurred_mono_time
    # Commitment is confirmed later by relative speed and gap growth, but the
    # preserved occurrence is the start of the sustained lead pull-away.
    assert event.occurred_mono_time <= 1.7
    assert event.detected_mono_time - event.occurred_mono_time >= 2.5
    assert event.occurred_mono_time == event.lead_commit_mono_time
    assert event.analysis_window_before_s == 5.0
    assert event.analysis_window_after_s == 8.0
    assert [onset.kind for onset in event.onsets][0] == "leadCommit"

  def test_controller_late_after_prompt_plan(self):
    detector = RollingLeadResponseDetector()
    events = run(detector, 5.0, lambda t: pullaway_sample(
      t, plan_onset=1.65, output_onset=2.2, ego_onset=2.8,
    ))

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "late_rolling_lead_response_controller"
    assert event.lead_to_plan_s is not None and event.lead_to_plan_s <= 0.75
    assert event.controller_response_mono_time is not None
    assert event.planner_response_mono_time is not None
    assert event.controller_response_mono_time - event.planner_response_mono_time > 0.25

  def test_vehicle_late_after_prompt_command(self):
    detector = RollingLeadResponseDetector()
    events = run(detector, 5.0, lambda t: pullaway_sample(
      t, plan_onset=1.65, output_onset=1.75, ego_onset=2.7,
    ))

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "late_rolling_lead_response_vehicle"
    assert event.ego_response_mono_time is not None
    assert event.controller_response_mono_time is not None
    assert event.ego_response_mono_time - event.controller_response_mono_time > 0.5

  def test_lead_and_ego_accelerate_together_is_silent(self):
    detector = RollingLeadResponseDetector()
    events = run(detector, 5.0, lambda t: pullaway_sample(
      t, plan_onset=1.5, output_onset=1.5, ego_onset=1.5, ego_accel=2.5,
    ))
    assert events == []

  def test_rapid_gap_with_nominal_onsets_has_explicit_downstream_fallback(self):
    detector = RollingLeadResponseDetector()
    events = run(detector, 5.0, lambda t: pullaway_sample(
      t,
      lead_accel=4.0,
      plan_onset=1.5,
      output_onset=1.5,
      ego_onset=1.5,
      ego_accel=1.5,
    ))
    assert len(events) == 1
    event = events[0]
    assert event.event_type == "late_rolling_lead_response_vehicle"
    assert "conservative downstream fallback" in event.attribution_detail
    assert "0.00 s" not in event.detail

  def test_slow_stop_and_go_creep_is_silent(self):
    detector = RollingLeadResponseDetector()

    def state(t: float) -> RollingLeadSample:
      elapsed = min(max(t - 1.5, 0.0), 1.2)
      lead_speed = 8.0 + 0.5 * elapsed
      lead_dx = 0.25 * elapsed * elapsed
      return sample(
        t,
        d_rel=20.0 + lead_dx,
        v_lead=lead_speed,
        v_lead_k=lead_speed,
        a_lead_k=0.5 if 1.5 <= t <= 2.7 else 0.0,
      )

    assert run(detector, 5.0, state) == []

  @parameterized.expand(["track", "cutin"])
  def test_track_change_or_cutin_is_silent(self, failure):
    detector = RollingLeadResponseDetector()

    def state(t: float) -> RollingLeadSample:
      current = pullaway_sample(t)
      if t >= 1.9:
        if failure == "track":
          return replace(current, radar_track_id=99)
        return replace(current, lead_present=False, lead_valid=False)
      return current

    assert run(detector, 5.0, state) == []

  def test_radar_distance_jump_is_silent(self):
    detector = RollingLeadResponseDetector()

    def state(t: float) -> RollingLeadSample:
      current = pullaway_sample(t)
      if t >= 1.9:
        return replace(current, d_rel=current.d_rel + 2.0)
      return current

    assert run(detector, 5.0, state) == []

  @parameterized.expand(["gas_pressed", "brake_pressed"])
  def test_driver_input_cancels(self, pedal):
    detector = RollingLeadResponseDetector()

    def state(t: float) -> RollingLeadSample:
      current = pullaway_sample(t)
      return replace(current, **{pedal: 1.9 <= t <= 2.1})

    assert run(detector, 5.0, state) == []

  def test_longitudinal_disabled_is_silent(self):
    detector = RollingLeadResponseDetector()
    assert run(detector, 5.0, lambda t: pullaway_sample(t, long_active=False)) == []

  def test_bad_episode_emits_only_once(self):
    detector = RollingLeadResponseDetector()
    events = run(detector, 12.0, lambda t: pullaway_sample(t, ego_onset=None))
    assert len(events) == 1

  def test_later_independent_episode_can_emit_after_cooldown(self):
    detector = RollingLeadResponseDetector()

    def state(t: float) -> RollingLeadSample:
      if t < 5.0:
        return pullaway_sample(t, ego_onset=None)
      if t < 31.0:
        return sample(t, v_ego=14.0, v_lead=14.0, v_lead_k=14.0, d_rel=30.0)
      current = pullaway_sample(
        t,
        lead_onset=32.0,
        plan_onset=33.3,
        output_onset=33.4,
        ego_onset=None,
        v_ego=14.0,
      )
      # Shift the helper's eight-metre-per-second and 20-metre baselines.
      return replace(
        current,
        v_lead=current.v_lead + 6.0,
        v_lead_k=current.v_lead_k + 6.0,
        d_rel=current.d_rel + 10.0,
      )

    events = run(detector, 36.5, state)
    assert len(events) == 2
    assert events[1].occurred_mono_time - events[0].detected_mono_time >= EPISODE_COOLDOWN_S

  @parameterized.expand(
    [
      ("route92_09_06_49", 2.2, 2.85),
      ("route92_09_17_10", 2.8, 2.75),
    ],
  )
  def test_route92_planner_late_synthetic_profiles(self, label, lead_accel, plan_onset):
    detector = RollingLeadResponseDetector()
    events = run(detector, 5.2, lambda t: pullaway_sample(
      t,
      lead_accel=lead_accel,
      plan_onset=plan_onset,
      output_onset=plan_onset + 0.1,
      ego_onset=plan_onset + 0.3,
    ))

    assert events, label
    assert events[0].event_type == "late_rolling_lead_response_planner"

from dataclasses import replace

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.spysypilot.long_event_detector import LaunchSample, LeadLaunchDetector


DT = 0.05


def sample(t: float, **kwargs) -> LaunchSample:
  defaults = {
    "active": True,
    "standstill": True,
    "v_ego": 0.0,
    "lead_present": True,
    "radar_valid": True,
    "d_rel": 6.0,
    "v_lead": 0.0,
    "v_lead_k": 0.0,
    "radar_track_id": 1,
    "plan_valid": True,
    "plan_should_stop": True,
    "output_valid": True,
    "output_accel": -2.0,
    "forecast_valid": True,
    "predicted_lead_v_2s": 0.0,
  }
  defaults.update(kwargs)
  return LaunchSample(t=t, **defaults)


def run_scenario(detector: LeadLaunchDetector, end: float, state_fn):
  event = None
  steps = round(end / DT) + 1
  for index in range(steps):
    t = index * DT
    event = detector.update(state_fn(t))
    if event is not None:
      return event
  return None


class TestLeadLaunchDetector(OpenpilotTestCase):
  def test_attributes_early_command_late_wheel_motion_to_vehicle(self):
    detector = LeadLaunchDetector()

    def state(t):
      return sample(
        t,
        plan_should_stop=t < 1.0,
        output_accel=0.6 if t >= 1.0 else -2.0,
        predicted_lead_v_2s=1.0 if t >= 0.8 else 0.0,
        v_lead=0.5 if t >= 2.0 else 0.0,
        v_lead_k=0.5 if t >= 2.0 else 0.0,
        standstill=t < 3.2,
        v_ego=0.2 if t >= 3.2 else 0.0,
        a_ego=0.2 if t >= 3.0 else 0.0,
      )

    event = run_scenario(detector, 4.0, state)
    assert event is not None
    assert event.event_type == "late_lead_launch_vehicle"
    self.assertAlmostEqual(event.lead_to_ego_s, 1.2)
    self.assertAlmostEqual(event.command_to_ego_s, 2.2)
    self.assertAlmostEqual(event.plan_to_lead_s, -1.0)
    self.assertAlmostEqual(event.forecast_to_lead_s, -1.2)
    self.assertAlmostEqual(event.confidence, 0.95)
    assert event.attribution_detail == "downstream/vehicle response"
    assert event.episode_start_mono_time == 0.0
    assert {onset.kind for onset in event.onsets} >= {
      "candidate", "forecast", "plan", "command", "lead", "ego", "egoAcceleration",
    }

  def test_attributes_late_plan(self):
    detector = LeadLaunchDetector()

    def state(t):
      return sample(
        t,
        v_lead=0.5 if t >= 2.0 else 0.0,
        v_lead_k=0.5 if t >= 2.0 else 0.0,
        plan_should_stop=t < 2.5,
        output_accel=0.6 if t >= 2.6 else -2.0,
        standstill=t < 3.0,
        v_ego=0.2 if t >= 3.0 else 0.0,
      )

    event = run_scenario(detector, 4.0, state)
    assert event is not None
    assert event.event_type == "late_lead_launch_planner"
    self.assertAlmostEqual(event.plan_to_lead_s, 0.5)

  def test_does_not_flag_a_swift_launch(self):
    detector = LeadLaunchDetector()

    def state(t):
      return sample(
        t,
        plan_should_stop=t < 1.5,
        output_accel=0.6 if t >= 1.6 else -2.0,
        v_lead=0.5 if t >= 2.0 else 0.0,
        v_lead_k=0.5 if t >= 2.0 else 0.0,
        standstill=t < 2.15,
        v_ego=0.2 if t >= 2.15 else 0.0,
      )

    assert run_scenario(detector, 3.0, state) is None

  def test_radar_jump_lowers_confidence_without_hiding_event(self):
    detector = LeadLaunchDetector()

    def state(t):
      d_rel = 8.0 if 1.0 <= t < 1.1 else 6.0
      return sample(
        t,
        d_rel=d_rel,
        plan_should_stop=t < 1.0,
        output_accel=0.6 if t >= 1.0 else -2.0,
        v_lead=0.5 if t >= 2.0 else 0.0,
        v_lead_k=0.5 if t >= 2.0 else 0.0,
        standstill=t < 3.0,
        v_ego=0.2 if t >= 3.0 else 0.0,
      )

    event = run_scenario(detector, 4.0, state)
    assert event is not None
    self.assertAlmostEqual(event.confidence, 0.55)

  def test_logs_stall_once(self):
    detector = LeadLaunchDetector()

    def state(t):
      return sample(
        t,
        plan_should_stop=t < 1.0,
        output_accel=0.6 if t >= 1.0 else -2.0,
        v_lead=0.5 if t >= 2.0 else 0.0,
        v_lead_k=0.5 if t >= 2.0 else 0.0,
      )

    event = run_scenario(detector, 6.0, state)
    assert event is not None
    assert event.event_type == "lead_launch_stall"
    assert event.severity == 3

  def test_disengagement_resets_candidate(self):
    detector = LeadLaunchDetector()
    for index in range(30):
      detector.update(sample(index * DT))
    detector.update(replace(sample(1.5), active=False))

    # A moving lead immediately after re-engagement must wait for a fresh standstill arm.
    event = detector.update(sample(1.55, v_lead=1.0, v_lead_k=1.0))
    assert event is None

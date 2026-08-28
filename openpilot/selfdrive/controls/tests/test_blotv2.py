from types import SimpleNamespace
import math
import unittest

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.blotv2 import (
  BLoTv2Supervisor,
  DebouncedTrigger,
  JERK_SCALE_MIN,
  JERK_SCALE_RATE,
  LEAD_DEPARTURE_CANCEL,
  LEAD_DEPARTURE_CONFIRM,
  LeadDeparturePreRelease,
  ONSET_PAD_MAX,
  ONSET_RATE_DOWN,
  ONSET_RATE_UP,
  STOPPED_LEAD_PAD_MAX,
  model_predicted_acceleration,
  model_predicted_speed,
)
from openpilot.selfdrive.controls.lib.longitudinal_lead import LeadObservation

T_FOLLOW_BASE = 1.45


def frames(seconds):
  return round(seconds / DT_MDL)


def lead(v=15.0, d=30.0, a=0.0):
  return LeadObservation(True, distance=d, speed=v, acceleration=a, model_prob=1.0)


def model_lead(v=None, prob=1.0):
  return SimpleNamespace(
    prob=prob,
    v=[15.0] * 6 if v is None else v,
  )


def run(supervisor, observation, v_ego, a_mpc, seconds):
  policy = None
  for _ in range(frames(seconds)):
    policy = supervisor.update(
      observation,
      v_ego,
      a_mpc,
      T_FOLLOW_BASE,
    )
  return policy


class TestDebouncedTrigger(unittest.TestCase):
  def test_arms_at_configured_elapsed_time(self):
    trigger = DebouncedTrigger(0.4, DT_MDL)
    for _ in range(frames(0.4) - 1):
      self.assertFalse(trigger.step(True, False))
    self.assertTrue(trigger.step(True, False))

  def test_holds_in_hysteresis_band_and_resets(self):
    trigger = DebouncedTrigger(0.4, DT_MDL)
    for _ in range(frames(0.4)):
      trigger.step(True, False)
    self.assertTrue(trigger.step(False, False))
    self.assertFalse(trigger.step(False, True))


class TestModelLeadHelpers(unittest.TestCase):
  def test_predicted_acceleration_uses_real_model_horizon(self):
    self.assertAlmostEqual(
      model_predicted_acceleration(model_lead([15.0, 13.0, 12.0])),
      -1.0,
    )

  def test_invalid_model_forecast_is_silent(self):
    self.assertIsNone(model_predicted_acceleration(model_lead([15.0], prob=1.0)))
    self.assertIsNone(model_predicted_acceleration(model_lead([15.0, 10.0], prob=0.2)))
    self.assertIsNone(model_predicted_acceleration(None))

    observation = lead(v=0.1)
    for probability in (0.5, 1.1, math.nan, math.inf, -math.inf):
      self.assertIsNone(model_predicted_acceleration(model_lead([15.0, 10.0], prob=probability)))
      self.assertIsNone(model_predicted_speed(model_lead([2.0, 2.4], prob=probability), observation))

    for radar_probability in (0.5, 1.1, math.nan, math.inf, -math.inf):
      self.assertIsNone(model_predicted_speed(
        model_lead([2.0, 2.4]), LeadObservation(True, 30.0, 0.1, 0.0, radar_probability),
      ))

  def test_predicted_speed_is_radar_anchored(self):
    observation = lead(v=0.1)
    self.assertAlmostEqual(
      model_predicted_speed(model_lead([2.0, 2.4]), observation),
      0.5,
    )


class TestSupervisorScope(unittest.TestCase):
  def test_no_lead_is_inert(self):
    policy = run(BLoTv2Supervisor(), LeadObservation(), 15.0, -3.0, 2.0)
    self.assertEqual(policy.jerk_scale, 1.0)
    self.assertEqual(policy.t_follow, T_FOLLOW_BASE)

  def test_crawl_keeps_stock_mpc_policy(self):
    policy = run(BLoTv2Supervisor(), lead(v=0.0, d=8.0), 0.5, -3.0, 2.0)
    self.assertEqual(policy.jerk_scale, 1.0)
    self.assertEqual(policy.t_follow, T_FOLLOW_BASE)


class TestRecoveryAndLaunch(unittest.TestCase):
  def test_recovery_relaxes_stale_mpc_braking(self):
    supervisor = BLoTv2Supervisor()
    policy = run(supervisor, lead(), 15.0, -2.0, 1.0)
    self.assertAlmostEqual(policy.jerk_scale, JERK_SCALE_MIN)
    self.assertTrue(policy.recovery_active)

  def test_recovery_ignores_small_trim(self):
    policy = run(BLoTv2Supervisor(), lead(), 15.0, -0.5, 2.0)
    self.assertEqual(policy.jerk_scale, 1.0)

  def test_responsive_policy_survives_low_speed_lead_transition(self):
    supervisor = BLoTv2Supervisor()
    run(supervisor, lead(), 15.0, -2.0, 1.0)

    policy = run(supervisor, lead(v=0.5, d=8.0), 0.5, 0.0, 1.0)

    self.assertEqual(policy.jerk_scale, JERK_SCALE_MIN)

  def test_model_forecast_arms_response(self):
    supervisor = BLoTv2Supervisor()
    predicted_accel = model_predicted_acceleration(
      model_lead([15.0, 11.0, 8.0]),
    )
    policy = None
    for _ in range(frames(1.0)):
      policy = supervisor.update(
        lead(),
        15.0,
        0.0,
        T_FOLLOW_BASE,
        predicted_accel,
      )
    self.assertAlmostEqual(policy.jerk_scale, JERK_SCALE_MIN)
    self.assertTrue(policy.model_active)

  def test_model_forecast_uses_existing_onset_pad(self):
    supervisor = BLoTv2Supervisor()
    policy = None
    for _ in range(frames(1.0)):
      policy = supervisor.update(
        lead(v=14.0, d=40.0),
        15.0,
        -0.5,
        T_FOLLOW_BASE,
        -0.75,
      )
    assert policy is not None
    self.assertTrue(policy.model_active)
    self.assertAlmostEqual(policy.t_follow - T_FOLLOW_BASE, ONSET_PAD_MAX * 0.5)

  def test_launch_relaxes_lagging_mpc(self):
    observation = lead(v=5.0, d=10.0, a=2.0)
    policy = run(BLoTv2Supervisor(), observation, 3.0, 0.2, 1.0)
    self.assertAlmostEqual(policy.jerk_scale, JERK_SCALE_MIN)
    self.assertTrue(policy.launch_active)

  def test_constant_speed_pullaway_does_not_arm(self):
    observation = lead(v=8.0, d=15.0, a=0.0)
    policy = run(BLoTv2Supervisor(), observation, 3.0, 0.0, 2.0)
    self.assertEqual(policy.jerk_scale, 1.0)


class TestSafetyAndContinuity(unittest.TestCase):
  def test_matched_mpc_braking_is_not_emergency(self):
    policy = BLoTv2Supervisor().update(
      lead(v=0.0, d=24.0, a=-0.15),
      8.0,
      -1.5,
      T_FOLLOW_BASE,
    )
    self.assertGreater(policy.required_decel, 1.5)
    self.assertFalse(policy.emergency)

  def test_nonfinite_mpc_target_keeps_emergency(self):
    policy = BLoTv2Supervisor().update(
      lead(v=0.0, d=24.0, a=-0.15),
      8.0,
      math.nan,
      T_FOLLOW_BASE,
    )
    self.assertTrue(policy.emergency)

  def test_emergency_returns_toward_stock_policy(self):
    supervisor = BLoTv2Supervisor()
    run(supervisor, lead(), 15.0, -2.0, 1.0)
    emergency_lead = lead(v=2.0, d=15.0, a=-2.0)
    policy = run(supervisor, emergency_lead, 10.0, -1.0, 1.0)
    self.assertTrue(policy.emergency)
    self.assertEqual(policy.jerk_scale, 1.0)

  def test_low_speed_transition_does_not_freeze_emergency_release(self):
    supervisor = BLoTv2Supervisor()
    run(supervisor, lead(), 15.0, -2.0, 1.0)
    emergency_lead = lead(v=0.0, d=5.0, a=-2.0)
    policy = supervisor.update(emergency_lead, 2.0, 0.0, T_FOLLOW_BASE)
    self.assertTrue(policy.emergency)

    policy = run(supervisor, emergency_lead, 0.5, 0.0, 1.0)

    self.assertEqual(policy.jerk_scale, 1.0)

  def test_low_speed_hold_releases_after_lead_loss(self):
    supervisor = BLoTv2Supervisor()
    run(supervisor, lead(), 15.0, -2.0, 1.0)
    self.assertEqual(run(supervisor, lead(), 0.5, 0.0, 1.0).jerk_scale, JERK_SCALE_MIN)

    released = supervisor.update(LeadObservation(), 0.5, 0.0, T_FOLLOW_BASE)
    self.assertGreater(released.jerk_scale, JERK_SCALE_MIN)
    self.assertEqual(run(supervisor, lead(), 0.5, 0.0, 1.0).jerk_scale, 1.0)

    run(supervisor, lead(), 15.0, -2.0, 1.0)
    self.assertEqual(run(supervisor, lead(), 0.5, 0.0, 1.0).jerk_scale, JERK_SCALE_MIN)
    supervisor.reset()
    after_reset = supervisor.update(lead(), 0.5, 0.0, T_FOLLOW_BASE)
    self.assertEqual(after_reset.jerk_scale, 1.0)

  def test_low_speed_hold_requires_exact_minimum_state(self):
    supervisor = BLoTv2Supervisor()
    supervisor.jerk_scale = JERK_SCALE_MIN + 1e-10

    policy = run(supervisor, lead(), 0.5, 0.0, 1.0)

    self.assertEqual(policy.jerk_scale, 1.0)

  def test_jerk_scale_never_steps(self):
    supervisor = BLoTv2Supervisor()
    previous = 1.0
    max_step = JERK_SCALE_RATE * DT_MDL + 1e-9
    for frame in range(frames(3.0)):
      observation = lead() if frame < frames(1.5) else LeadObservation()
      policy = supervisor.update(
        observation,
        15.0,
        -2.0,
        T_FOLLOW_BASE,
      )
      self.assertLessEqual(abs(policy.jerk_scale - previous), max_step)
      previous = policy.jerk_scale

  def test_onset_pad_is_proportional(self):
    observation = lead(v=14.0, d=40.0, a=-0.75)
    policy = run(BLoTv2Supervisor(), observation, 15.0, -0.5, 2.0)
    self.assertAlmostEqual(
      policy.t_follow - T_FOLLOW_BASE,
      ONSET_PAD_MAX * 0.5,
    )

  def test_stopped_lead_uses_larger_onset_pad(self):
    policy = run(BLoTv2Supervisor(), lead(v=0.0, d=83.0), 14.0, -0.4, 2.0)
    assert policy is not None
    self.assertAlmostEqual(policy.t_follow - T_FOLLOW_BASE, STOPPED_LEAD_PAD_MAX)

  def test_onset_pad_respects_slew_rates(self):
    supervisor = BLoTv2Supervisor()
    observation = lead(v=14.0, d=40.0, a=-0.75)
    previous = 0.0
    for _ in range(frames(2.0)):
      policy = supervisor.update(
        observation,
        15.0,
        -0.5,
        T_FOLLOW_BASE,
      )
      pad = policy.t_follow - T_FOLLOW_BASE
      self.assertLessEqual(pad - previous, ONSET_RATE_UP * DT_MDL + 1e-9)
      self.assertLessEqual(previous - pad, ONSET_RATE_DOWN * DT_MDL + 1e-9)
      previous = pad

  def test_personality_base_changes_without_delay(self):
    supervisor = BLoTv2Supervisor()
    supervisor.update(lead(), 15.0, 0.0, T_FOLLOW_BASE)
    policy = supervisor.update(lead(), 15.0, 0.0, 1.25)
    self.assertEqual(policy.t_follow, 1.25)


class TestLeadDeparturePreRelease(unittest.TestCase):
  def test_requires_sustained_prediction(self):
    release = LeadDeparturePreRelease()
    observation = lead(v=0.0, d=6.0)
    for _ in range(frames(LEAD_DEPARTURE_CONFIRM) - 1):
      self.assertFalse(release.update(True, True, observation, 1.0))
    self.assertTrue(release.update(True, True, observation, 1.0))

  def test_measured_motion_requires_sustained_departure(self):
    release = LeadDeparturePreRelease()
    route_noise = (0.262, 0.259, 0.250, 0.241)
    for speed in route_noise:
      self.assertFalse(release.update(True, True, lead(v=speed, d=4.3), None))

  def test_strong_measured_motion_releases_immediately(self):
    release = LeadDeparturePreRelease()
    self.assertTrue(release.update(True, True, lead(v=0.4, d=4.3), None))

  def test_collapsed_prediction_reapplies_hold(self):
    release = LeadDeparturePreRelease()
    observation = lead(v=0.0, d=6.0)
    for _ in range(frames(LEAD_DEPARTURE_CONFIRM)):
      release.update(True, True, observation, 1.0)
    for _ in range(frames(LEAD_DEPARTURE_CANCEL) - 1):
      self.assertTrue(release.update(True, True, observation, 0.0))
    self.assertFalse(release.update(True, True, observation, 0.0))

  def test_scope_loss_fails_closed(self):
    release = LeadDeparturePreRelease()
    observation = lead(v=0.0, d=6.0)
    for _ in range(frames(LEAD_DEPARTURE_CONFIRM)):
      release.update(True, True, observation, 1.0)
    self.assertFalse(release.update(True, True, LeadObservation(), 1.0))
    self.assertFalse(release.update(False, True, observation, 1.0))
    self.assertFalse(release.update(True, False, observation, 1.0))


if __name__ == "__main__":
  unittest.main()

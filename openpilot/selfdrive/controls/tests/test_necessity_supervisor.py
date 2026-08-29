import math


from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_lead import LeadObservation
from openpilot.selfdrive.controls.lib.necessity_supervisor import (DebouncedTrigger, JERK_SCALE_MIN, JERK_SCALE_RATE, LEAD_DEPARTURE_CANCEL,
                                                                   LEAD_DEPARTURE_CONFIRM, LeadDeparturePreRelease, NecessitySupervisor,
                                                                   ONSET_PAD_MAX, ONSET_RATE_DOWN, ONSET_RATE_UP, STOPPED_LEAD_PAD_MAX)


def frames(seconds):
  return round(seconds / DT_MDL)


def lead(v=15.0, d=30.0, a=0.0):
  return LeadObservation(True, distance=d, speed=v, acceleration=a, model_prob=1.0)


def run(supervisor, observation, v_ego, a_mpc, seconds, predicted_lead_accel=None):
  policy = None
  for _ in range(frames(seconds)):
    policy = supervisor.update(observation, v_ego, a_mpc, predicted_lead_accel)
  return policy


class TestDebouncedTrigger:
  def test_arms_after_the_debounce(self):
    trigger = DebouncedTrigger(0.4, DT_MDL)
    for _ in range(frames(0.4) - 1):
      assert not trigger.step(True, False)
    assert trigger.step(True, False)

  def test_holds_in_the_hysteresis_band_and_resets(self):
    trigger = DebouncedTrigger(0.4, DT_MDL)
    for _ in range(frames(0.4)):
      trigger.step(True, False)
    assert trigger.step(False, False)
    assert not trigger.step(False, True)


class TestScope:
  def test_no_lead_is_inert(self):
    policy = run(NecessitySupervisor(), LeadObservation(), 15.0, -3.0, 2.0)
    assert policy.jerk_scale == 1.0
    assert policy.t_follow_pad == 0.0

  def test_a_crawl_starts_with_the_stock_policy(self):
    policy = run(NecessitySupervisor(), lead(v=0.0, d=8.0), 0.5, -3.0, 2.0)
    assert policy.jerk_scale == 1.0
    assert policy.t_follow_pad == 0.0


class TestTriggers:
  def test_recovery_relaxes_stale_mpc_braking(self):
    assert math.isclose(run(NecessitySupervisor(), lead(), 15.0, -2.0, 1.0).jerk_scale, JERK_SCALE_MIN, rel_tol=1e-6, abs_tol=1e-9)

  def test_recovery_ignores_a_small_trim(self):
    assert run(NecessitySupervisor(), lead(), 15.0, -0.5, 2.0).jerk_scale == 1.0

  def test_model_forecast_arms_the_response(self):
    assert math.isclose(run(NecessitySupervisor(), lead(), 15.0, 0.0, 1.0, predicted_lead_accel=-1.0).jerk_scale, JERK_SCALE_MIN, rel_tol=1e-6, abs_tol=1e-9)

  def test_model_forecast_opens_the_onset_pad(self):
    policy = run(NecessitySupervisor(), lead(v=14.0, d=40.0), 15.0, -0.5, 1.0, predicted_lead_accel=-0.75)
    assert math.isclose(policy.t_follow_pad, ONSET_PAD_MAX * 0.5, rel_tol=1e-6, abs_tol=1e-9)

  def test_launch_relaxes_a_lagging_mpc(self):
    assert math.isclose(run(NecessitySupervisor(), lead(v=5.0, d=10.0, a=2.0), 3.0, 0.2, 1.0).jerk_scale, JERK_SCALE_MIN, rel_tol=1e-6, abs_tol=1e-9)

  def test_constant_speed_pull_away_does_not_arm(self):
    assert run(NecessitySupervisor(), lead(v=8.0, d=15.0, a=0.0), 3.0, 0.0, 2.0).jerk_scale == 1.0

  def test_jerk_scale_never_steps(self):
    supervisor = NecessitySupervisor()
    previous = 1.0
    for frame in range(frames(3.0)):
      policy = supervisor.update(lead() if frame < frames(1.5) else LeadObservation(), 15.0, -2.0)
      assert abs(policy.jerk_scale - previous) <= JERK_SCALE_RATE * DT_MDL + 1e-9
      previous = policy.jerk_scale


class TestPads:
  def test_onset_pad_is_proportional(self):
    pad = run(NecessitySupervisor(), lead(v=14.0, d=40.0, a=-0.75), 15.0, -0.5, 2.0).t_follow_pad
    assert math.isclose(pad, ONSET_PAD_MAX * 0.5, rel_tol=1e-6, abs_tol=1e-9)

  def test_stopped_lead_gets_the_larger_pad(self):
    assert math.isclose(run(NecessitySupervisor(), lead(v=0.0, d=83.0), 14.0, -0.4, 2.0).t_follow_pad, STOPPED_LEAD_PAD_MAX, rel_tol=1e-6, abs_tol=1e-9)

  def test_pads_saturate_instead_of_vanishing_above_the_onset_limit(self):
    # 10 m/s toward a stopped lead 30 m out needs 1.9 m/s^2, past ONSET_MAX_A_REQ; the pad must stay at its ceiling
    policy = run(NecessitySupervisor(), lead(v=0.0, d=30.0), 10.0, -1.9, 2.0)
    assert not policy.stand_down
    assert math.isclose(policy.t_follow_pad, STOPPED_LEAD_PAD_MAX, rel_tol=1e-6, abs_tol=1e-9)
    policy = run(NecessitySupervisor(), lead(v=10.0, d=20.0, a=-3.0), 15.0, -3.4, 2.0)
    assert not policy.stand_down
    assert math.isclose(policy.t_follow_pad, ONSET_PAD_MAX, rel_tol=1e-6, abs_tol=1e-9)

  def test_pads_respect_their_slew_rates(self):
    supervisor = NecessitySupervisor()
    previous = 0.0
    for _ in range(frames(2.0)):
      pad = supervisor.update(lead(v=14.0, d=40.0, a=-0.75), 15.0, -0.5).t_follow_pad
      assert pad - previous <= ONSET_RATE_UP * DT_MDL + 1e-9
      assert previous - pad <= ONSET_RATE_DOWN * DT_MDL + 1e-9
      previous = pad


class TestStandDown:
  def test_matched_mpc_braking_is_not_a_stand_down(self):
    assert not NecessitySupervisor().update(lead(v=0.0, d=24.0, a=-0.15), 8.0, -1.5).stand_down

  def test_a_nonfinite_mpc_target_stands_down(self):
    assert NecessitySupervisor().update(lead(v=0.0, d=24.0, a=-0.15), 8.0, math.nan).stand_down

  def test_stand_down_returns_toward_the_stock_policy(self):
    supervisor = NecessitySupervisor()
    run(supervisor, lead(), 15.0, -2.0, 1.0)
    policy = run(supervisor, lead(v=2.0, d=15.0, a=-2.0), 10.0, -1.0, 1.0)
    assert policy.stand_down
    assert policy.jerk_scale == 1.0


class TestLowSpeedHold:
  def test_responsive_policy_survives_the_crawl(self):
    supervisor = NecessitySupervisor()
    run(supervisor, lead(), 15.0, -2.0, 1.0)
    assert math.isclose(run(supervisor, lead(v=0.5, d=8.0), 0.5, 0.0, 1.0).jerk_scale, JERK_SCALE_MIN, rel_tol=1e-6, abs_tol=1e-9)

  def test_partial_softening_is_kept_too(self):
    supervisor = NecessitySupervisor()
    partial = run(supervisor, lead(), 15.0, -2.0, 0.6).jerk_scale
    assert JERK_SCALE_MIN < partial < 1.0
    assert math.isclose(run(supervisor, lead(v=0.5, d=8.0), 0.5, 0.0, 1.0).jerk_scale, partial, rel_tol=1e-6, abs_tol=1e-9)

  def test_a_stand_down_release_is_not_frozen_by_the_crawl(self):
    supervisor = NecessitySupervisor()
    run(supervisor, lead(), 15.0, -2.0, 1.0)
    assert supervisor.update(lead(v=0.0, d=5.0, a=-2.0), 2.0, 0.0).stand_down
    assert run(supervisor, lead(v=0.0, d=5.0, a=-2.0), 0.5, 0.0, 1.0).jerk_scale == 1.0

  def test_the_hold_releases_after_lead_loss_and_reset(self):
    supervisor = NecessitySupervisor()
    run(supervisor, lead(), 15.0, -2.0, 1.0)
    assert math.isclose(run(supervisor, lead(), 0.5, 0.0, 1.0).jerk_scale, JERK_SCALE_MIN, rel_tol=1e-6, abs_tol=1e-9)
    assert supervisor.update(LeadObservation(), 0.5, 0.0).jerk_scale > JERK_SCALE_MIN
    assert run(supervisor, lead(), 0.5, 0.0, 1.0).jerk_scale == 1.0
    run(supervisor, lead(), 15.0, -2.0, 1.0)
    assert math.isclose(run(supervisor, lead(), 0.5, 0.0, 1.0).jerk_scale, JERK_SCALE_MIN, rel_tol=1e-6, abs_tol=1e-9)
    supervisor.reset()
    assert supervisor.update(lead(), 0.5, 0.0).jerk_scale == 1.0

  def test_the_whiplash_ratchet_and_the_hold_are_separate(self):
    supervisor = NecessitySupervisor()
    run(supervisor, lead(), 15.0, -2.0, 1.0)
    # a braking lead we are still closing on must not stiffen the solution, at speed or at the crawl
    assert math.isclose(run(supervisor, lead(v=10.0, d=30.0, a=-0.5), 12.0, -0.5, 0.5).jerk_scale, JERK_SCALE_MIN, rel_tol=1e-6, abs_tol=1e-9)
    assert math.isclose(run(supervisor, lead(v=0.2, d=8.0, a=-0.5), 0.8, -0.5, 0.5).jerk_scale, JERK_SCALE_MIN, rel_tol=1e-6, abs_tol=1e-9)


class TestLeadDeparturePreRelease:
  def test_requires_a_sustained_prediction(self):
    release = LeadDeparturePreRelease()
    for _ in range(frames(LEAD_DEPARTURE_CONFIRM) - 1):
      assert not release.update(True, True, lead(v=0.0, d=6.0), 1.0)
    assert release.update(True, True, lead(v=0.0, d=6.0), 1.0)

  def test_measured_noise_is_not_a_departure(self):
    release = LeadDeparturePreRelease()
    for speed in (0.262, 0.259, 0.250, 0.241):
      assert not release.update(True, True, lead(v=speed, d=4.3), None)

  def test_strong_measured_motion_releases_immediately(self):
    assert LeadDeparturePreRelease().update(True, True, lead(v=0.4, d=4.3), None)

  def test_a_collapsed_prediction_reapplies_the_hold(self):
    release = LeadDeparturePreRelease()
    for _ in range(frames(LEAD_DEPARTURE_CONFIRM)):
      release.update(True, True, lead(v=0.0, d=6.0), 1.0)
    for _ in range(frames(LEAD_DEPARTURE_CANCEL) - 1):
      assert release.update(True, True, lead(v=0.0, d=6.0), 0.0)
    assert not release.update(True, True, lead(v=0.0, d=6.0), 0.0)

  def test_scope_loss_fails_closed(self):
    release = LeadDeparturePreRelease()
    for _ in range(frames(LEAD_DEPARTURE_CONFIRM)):
      release.update(True, True, lead(v=0.0, d=6.0), 1.0)
    assert not release.update(True, True, LeadObservation(), 1.0)
    assert not release.update(False, True, lead(v=0.0, d=6.0), 1.0)
    assert not release.update(True, False, lead(v=0.0, d=6.0), 1.0)

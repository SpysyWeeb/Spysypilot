import math


from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_lead import LeadObservation
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LEAD_ABSENCE_FRAMES
from openpilot.selfdrive.controls.lib.necessity_supervisor import (DebouncedTrigger, JERK_SCALE_MIN, JERK_SCALE_RATE, LEAD_DEPARTURE_CANCEL,
                                                                   LEAD_DEPARTURE_CONFIRM, LEAD_SAME_CAR_JUMP, LeadDeparturePreRelease,
                                                                   NecessitySupervisor,
                                                                   ONSET_MAX_A_REQ, ONSET_PAD_MAX, ONSET_RATE_DOWN, ONSET_RATE_UP, PURSUIT_TAIL_S,
                                                                   STOPPED_LEAD_FULL_DECEL, STOPPED_LEAD_PAD_MAX)


def frames(seconds):
  return round(seconds / DT_MDL)


def lead(v=15.0, d=30.0, a=0.0, model_prob=1.0, track_id=7):
  # a radar track the model confirms as its lead, unless model_prob says otherwise
  return LeadObservation(True, distance=d, speed=v, acceleration=a, model_prob=model_prob, track_id=track_id)


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

  def test_the_low_jerk_cost_outlasts_the_braking_behind_a_lead_pulling_away(self):
    supervisor = NecessitySupervisor()
    run(supervisor, lead(v=0.5, d=8.0, a=1.0), 4.0, -2.0, 1.0)              # excess braking while the lead is already leaving
    assert math.isclose(supervisor.jerk_scale, JERK_SCALE_MIN, rel_tol=1e-6, abs_tol=1e-9)
    policy = run(supervisor, lead(v=2.0, d=9.0, a=2.0), 3.0, -0.2, 1.0)     # our braking has eased: the recovery trigger is off ...
    assert math.isclose(policy.jerk_scale, JERK_SCALE_MIN, rel_tol=1e-6, abs_tol=1e-9)   # ... and the tail keeps the low cost
    policy = run(supervisor, lead(v=6.0, d=14.0, a=2.0), 4.0, 1.8, PURSUIT_TAIL_S)
    assert policy.jerk_scale == 1.0                                          # the tail ends on its own

  def test_the_tail_ends_when_the_lead_stops_pulling_away(self):
    supervisor = NecessitySupervisor()
    run(supervisor, lead(v=0.5, d=8.0, a=1.0), 4.0, -2.0, 1.0)
    policy = run(supervisor, lead(v=2.0, d=9.0, a=0.0), 3.0, -0.2, 1.0)
    assert policy.jerk_scale == 1.0

  def test_easing_behind_a_lead_that_never_pulled_away_returns_to_stock(self):
    supervisor = NecessitySupervisor()
    run(supervisor, lead(v=0.5, d=8.0, a=0.0), 4.0, -2.0, 1.0)
    policy = run(supervisor, lead(v=0.5, d=8.0, a=0.0), 2.0, -0.2, 1.0)
    assert policy.jerk_scale == 1.0

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
    assert math.isclose(policy.t_follow_pad, STOPPED_LEAD_PAD_MAX, rel_tol=1e-6, abs_tol=1e-9)
    policy = run(NecessitySupervisor(), lead(v=10.0, d=20.0, a=-3.0), 15.0, -3.4, 2.0)
    assert math.isclose(policy.t_follow_pad, ONSET_PAD_MAX, rel_tol=1e-6, abs_tol=1e-9)

  def test_the_stopped_lead_pad_tops_out_below_the_stand_down_gate(self):
    # the two 1.5s are different things: ONSET_MAX_A_REQ is the stand-down gate, this pad's ramp ends at BLoTv2's 1.2.
    # 9 m/s toward a stopped lead 34 m out needs 1.35 m/s^2, inside that window, and the pad is already at its ceiling
    # (audit 2026-09-17 F006: the docs said the ramp ran to 1.5, the field value it inherited is 1.2)
    assert STOPPED_LEAD_FULL_DECEL < 1.35 < ONSET_MAX_A_REQ
    pad = run(NecessitySupervisor(), lead(v=0.0, d=34.0), 9.0, -1.35, 2.0).t_follow_pad
    assert math.isclose(pad, STOPPED_LEAD_PAD_MAX, rel_tol=1e-6, abs_tol=1e-9)

  def test_pads_respect_their_slew_rates(self):
    supervisor = NecessitySupervisor()
    previous = 0.0
    for _ in range(frames(2.0)):
      pad = supervisor.update(lead(v=14.0, d=40.0, a=-0.75), 15.0, -0.5).t_follow_pad
      assert pad - previous <= ONSET_RATE_UP * DT_MDL + 1e-9
      assert previous - pad <= ONSET_RATE_DOWN * DT_MDL + 1e-9
      previous = pad


class TestStandDown:
  # a stand-down is observable only as the stock policy: no pad, no softening
  def test_matched_mpc_braking_is_not_a_stand_down(self):
    assert NecessitySupervisor().update(lead(v=0.0, d=24.0, a=-0.15), 8.0, -1.5).t_follow_pad > 0.0

  def test_a_nonfinite_mpc_target_stands_down(self):
    policy = NecessitySupervisor().update(lead(v=0.0, d=24.0, a=-0.15), 8.0, math.nan)
    assert policy.t_follow_pad == 0.0 and policy.jerk_scale == 1.0

  def test_stand_down_returns_toward_the_stock_policy(self):
    supervisor = NecessitySupervisor()
    run(supervisor, lead(), 15.0, -2.0, 1.0)
    policy = run(supervisor, lead(v=2.0, d=15.0, a=-2.0), 10.0, -1.0, 1.0)
    assert policy.jerk_scale == 1.0 and policy.t_follow_pad == 0.0


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
    assert supervisor.update(lead(v=0.0, d=5.0, a=-2.0), 2.0, 0.0).jerk_scale > JERK_SCALE_MIN
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
    assert LeadDeparturePreRelease().update(True, True, lead(v=0.6, d=4.3), None)
    # a creep is not a departure: 0.65 m/s that stopped again cycled the hold under a standing car (route 0x2b t=1540);
    # below the immediate threshold the confirmed path decides
    assert not LeadDeparturePreRelease().update(True, True, lead(v=0.4, d=4.3), None)

  def test_a_collapsed_prediction_reapplies_the_hold(self):
    release = LeadDeparturePreRelease()
    for _ in range(frames(LEAD_DEPARTURE_CONFIRM)):
      release.update(True, True, lead(v=0.0, d=6.0), 1.0)
    for _ in range(frames(LEAD_DEPARTURE_CANCEL) - 1):
      assert release.update(True, True, lead(v=0.0, d=6.0), 0.0)
    assert not release.update(True, True, lead(v=0.0, d=6.0), 0.0)

  def test_a_measured_speed_from_radards_low_speed_override_is_not_a_departure(self):
    # the override return of a standing car reads 0.3 m/s at a flat distance; the model does not confirm it
    release = LeadDeparturePreRelease()
    release.update(True, True, lead(v=0.0, d=4.4), None)
    for _ in range(frames(1.0)):
      assert not release.update(True, True, lead(v=0.3, d=3.9, model_prob=0.0, track_id=8), None)

  def test_a_frame_the_model_does_not_confirm_restarts_the_measured_confirmation(self):
    release = LeadDeparturePreRelease()
    for _ in range(frames(LEAD_DEPARTURE_CONFIRM) - 1):
      assert not release.update(True, True, lead(v=0.3, d=6.0), None)
    assert not release.update(True, True, lead(v=0.3, d=6.0, model_prob=0.0), None)
    for _ in range(frames(LEAD_DEPARTURE_CONFIRM) - 1):
      assert not release.update(True, True, lead(v=0.3, d=6.0), None)
    assert release.update(True, True, lead(v=0.3, d=6.0), None)

  def test_a_track_switch_restarts_the_measured_confirmation(self):
    # another return takes the lead slot: its reading on the frame it appears is a fresh filter's, and confirms only from then on
    release = LeadDeparturePreRelease()
    for _ in range(frames(LEAD_DEPARTURE_CONFIRM) - 1):
      assert not release.update(True, True, lead(v=0.3, d=6.0), None)
    for _ in range(frames(LEAD_DEPARTURE_CONFIRM)):
      assert not release.update(True, True, lead(v=0.3, d=5.8, track_id=9), None)
    assert release.update(True, True, lead(v=0.3, d=5.8, track_id=9), None)

  def test_two_returns_trading_the_lead_slot_never_confirm(self):
    release = LeadDeparturePreRelease()
    for i in range(frames(1.0)):
      assert not release.update(True, True, lead(v=0.3, d=6.0, track_id=7 + i % 2), None)

  def test_once_the_mpc_stops_the_car_for_the_lead_only_a_departure_releases_it(self):
    release = LeadDeparturePreRelease()
    release.update(True, True, lead(v=0.0, d=8.0), None)
    assert not release.holding
    release.update(True, True, lead(v=0.0, d=8.0), None, True)
    for _ in range(frames(2.0)):
      assert not release.update(True, True, lead(v=0.0, d=8.0), None, False)
      assert release.holding
    assert release.update(True, True, lead(v=0.6, d=8.0), None) and not release.holding

  def test_a_release_taken_back_is_reported_once(self):
    release = LeadDeparturePreRelease()
    release.update(True, True, lead(v=0.0, d=6.0), None, True)
    for _ in range(frames(LEAD_DEPARTURE_CONFIRM)):
      release.update(True, True, lead(v=0.0, d=6.0), 1.0)
    cancelled = []
    for _ in range(frames(LEAD_DEPARTURE_CANCEL) + 3):
      release.update(True, True, lead(v=0.0, d=6.0), 0.0)
      cancelled.append(release.cancelled)
    assert cancelled == [False] * (frames(LEAD_DEPARTURE_CANCEL) - 1) + [True] + [False] * 3
    assert release.holding
    # the car leaving standstill ends the scope: no hold and nothing taken back
    for _ in range(frames(LEAD_DEPARTURE_CONFIRM)):
      release.update(True, True, lead(v=0.0, d=6.0), 1.0)
    assert not release.update(True, False, lead(v=0.0, d=6.0), 0.0)
    assert not release.cancelled and not release.holding

  def test_a_release_taken_back_restores_only_a_hold(self):
    # a car at rest behind a lead the MPC never stopped it for: a release on its forecast taken back holds nothing
    release = LeadDeparturePreRelease()
    for _ in range(frames(LEAD_DEPARTURE_CONFIRM)):
      release.update(True, True, lead(v=0.0, d=20.0), 1.0)
    for _ in range(frames(LEAD_DEPARTURE_CANCEL) + 3):
      release.update(True, True, lead(v=0.0, d=20.0), 0.0)
      assert not release.cancelled and not release.holding
    assert not release.update(True, True, lead(v=0.0, d=20.0), 0.0)

  def test_the_lead_missing_keeps_the_hold(self):
    release = LeadDeparturePreRelease(absence_frames=LEAD_ABSENCE_FRAMES)
    release.update(True, True, lead(v=0.0, d=7.5), None, True)
    # no lead for a second, far beyond the MPC's own absence hold: nothing says the lead has left
    for _ in range(frames(1.0)):
      assert not release.update(True, True, LeadObservation(), 1.0)
      assert release.holding and not release.handed_back
    # back where it stood it is the same lead, and only its departure releases the car
    for _ in range(frames(1.0)):
      assert not release.update(True, True, lead(v=0.0, d=7.6, track_id=8), None)
      assert release.holding and not release.handed_back
    assert release.update(True, True, lead(v=0.6, d=7.6, track_id=8), None) and not release.holding

  def test_a_car_standing_beyond_the_lead_ends_the_hold(self):
    # the slot passes to a car 5 m beyond the held lead, at once or after the lead went missing: the lead has left
    for missing in (0, frames(1.0)):
      release = LeadDeparturePreRelease(absence_frames=LEAD_ABSENCE_FRAMES)
      release.update(True, True, lead(v=0.0, d=7.5), None, True)
      for _ in range(missing):
        release.update(True, True, LeadObservation(), None)
      assert not release.update(True, True, lead(v=0.0, d=12.5, track_id=9), None, True)
      assert not release.holding and release.handed_back
      release.update(True, True, lead(v=0.0, d=12.5, track_id=9), None, True)
      assert not release.holding and not release.handed_back

  def test_a_nearer_car_in_the_slot_keeps_the_hold(self):
    # a car cuts in between and the radar trades the slot between it and the held lead: nothing has left
    release = LeadDeparturePreRelease(absence_frames=LEAD_ABSENCE_FRAMES)
    release.update(True, True, lead(v=0.0, d=8.9), None, True)
    for i in range(frames(1.0)):
      release.update(True, True, lead(v=0.0, d=3.1, track_id=9) if i % 3 else lead(v=0.0, d=8.9), None)
      assert release.holding and not release.handed_back
    # the car that cut in leaves: its departure releases the car
    assert release.update(True, True, lead(v=0.6, d=3.2, track_id=9), None) and not release.holding

  def test_a_car_read_past_the_cut_in_is_the_held_lead_within_its_standing_band(self):
    # past a car that cut in, the slot reads a car where the held lead stood: within two range steps of that it is the lead;
    # a third step beyond is a car standing farther ahead, so the lead has left
    for beyond, kept in ((0.2, True), (0.3, False)):
      release = LeadDeparturePreRelease(absence_frames=LEAD_ABSENCE_FRAMES)
      release.update(True, True, lead(v=0.0, d=8.9), None, True)
      release.update(True, True, lead(v=0.0, d=3.1, track_id=9), None)
      release.update(True, True, lead(v=0.0, d=8.9 + beyond, track_id=10), None)
      assert release.holding == kept and release.handed_back != kept, beyond

  def test_the_models_path_opening_ends_the_hold_on_a_missing_lead(self):
    release = LeadDeparturePreRelease(absence_frames=LEAD_ABSENCE_FRAMES)
    release.update(True, True, lead(v=0.0, d=7.5), None, True)
    # with the lead in sight its own departure is the evidence, not the model's path, however long that stays open
    for _ in range(frames(1.0)):
      release.update(True, True, lead(v=0.0, d=7.5), None, False, True)
      assert release.holding
    # the lead missing while the path is not open says nothing; the first frame both hold, it has left
    for _ in range(frames(1.0)):
      release.update(True, True, LeadObservation(), None)
      assert release.holding and not release.handed_back
    release.update(True, True, LeadObservation(), None, False, True)
    assert not release.holding and release.handed_back

  def test_a_release_taken_back_restores_the_hold_only_while_the_lead_stands_where_it_was_released(self):
    # the forecast releases a car held behind a lead 6 m ahead and then fades; a standing car reads up to two of the radar's
    # 0.1 m range steps from where it stood, and a lead read nearer has not left
    for d_cancel, rehold in ((6.2, True), (5.6, True), (6.5, False)):
      release = LeadDeparturePreRelease(absence_frames=LEAD_ABSENCE_FRAMES)
      release.update(True, True, lead(v=0.0, d=6.0), None, True)
      for _ in range(frames(LEAD_DEPARTURE_CONFIRM)):
        release.update(True, True, lead(v=0.0, d=6.0), 1.0)
      for _ in range(frames(LEAD_DEPARTURE_CANCEL)):
        assert not release.cancelled
        release.update(True, True, lead(v=0.0, d=d_cancel), 0.0)
      assert release.cancelled == rehold and release.holding == rehold and not release.handed_back
      # a lead that has moved is leaving: nothing holds the car until the MPC's own plan stops it for the lead again
      release.update(True, True, lead(v=0.0, d=d_cancel), 0.0)
      assert release.holding == rehold
      release.update(True, True, lead(v=0.0, d=d_cancel), 0.0, True)
      assert release.holding

  def test_the_hold_follows_its_lead_and_ends_with_it(self):
    release = LeadDeparturePreRelease(absence_frames=LEAD_ABSENCE_FRAMES)
    release.update(True, True, lead(v=0.0, d=7.3), None, True)
    # radard re-associates the same car where it stands, on another track or from vision alone
    for track_id in (8, -1, 9):
      release.update(True, True, lead(v=0.0, d=7.3 + 0.5 * LEAD_SAME_CAR_JUMP, track_id=track_id), None)
      assert release.holding and not release.handed_back
    # another car takes the slot: the hold ends with a hand-back, and the car did not come to rest behind that car, so the
    # MPC's own bit is the MPC's again -- its plan still stopping the car on the frame the slot changes is not a hold
    release.update(True, True, lead(v=0.0, d=25.0, track_id=10), None, True)
    assert not release.holding and release.handed_back
    release.update(True, True, lead(v=0.0, d=25.0, track_id=10), None, True)
    assert not release.holding and not release.handed_back
    # the lead's own track taking the slot back (a car that cut in has left) is that lead again
    release.update(True, True, lead(v=0.0, d=7.5, track_id=8), None, True)
    assert release.holding and not release.handed_back
    release.update(True, True, lead(v=0.0, d=25.0, track_id=10), None, True)
    # the car comes to rest again, behind that car
    release.update(True, False, lead(v=0.0, d=7.0, track_id=10), None, True)
    release.update(True, True, lead(v=0.0, d=7.0, track_id=10), None, True)
    assert release.holding

  def test_only_the_lead_the_car_came_to_rest_behind_is_held(self):
    # at rest with no lead (a red light's hold) past the MPC's absence hold, a lead then appears: the MPC's plan stopping the
    # car for it is its own bit, not a hold, and a release on it taken back holds nothing
    release = LeadDeparturePreRelease(absence_frames=LEAD_ABSENCE_FRAMES)
    for _ in range(LEAD_ABSENCE_FRAMES):
      release.update(True, True, LeadObservation(), None)
    release.update(True, True, lead(v=0.0, d=12.0), None, True)
    assert not release.holding
    for _ in range(frames(LEAD_DEPARTURE_CONFIRM)):
      release.update(True, True, lead(v=0.0, d=12.0), 1.0, True)
    for _ in range(frames(LEAD_DEPARTURE_CANCEL) + 1):
      release.update(True, True, lead(v=0.0, d=12.0), 0.0, True)
      assert not release.cancelled and not release.holding
    # a lead the radar reports only a few frames after the car came to rest is the one it came to rest behind
    release = LeadDeparturePreRelease(absence_frames=LEAD_ABSENCE_FRAMES)
    for _ in range(LEAD_ABSENCE_FRAMES - 1):
      release.update(True, True, LeadObservation(), None)
    release.update(True, True, lead(v=0.0, d=6.0), None, True)
    assert release.holding

  def test_a_release_is_taken_back_only_on_its_lead(self):
    # the lead leaves and a standing car farther ahead takes the slot while the car still stands: nothing holds the car
    release = LeadDeparturePreRelease(absence_frames=LEAD_ABSENCE_FRAMES)
    release.update(True, True, lead(v=0.0, d=7.3), None, True)
    for _ in range(frames(LEAD_DEPARTURE_CONFIRM)):
      release.update(True, True, lead(v=0.0, d=7.3), 1.0)
    assert release.update(True, True, lead(v=0.6, d=7.4), 1.0)
    for _ in range(frames(1.0)):
      assert not release.update(True, True, lead(v=0.0, d=20.0, track_id=9), 0.0)
      assert not (release.cancelled or release.holding or release.handed_back)

  def test_scope_loss_fails_closed(self):
    release = LeadDeparturePreRelease()
    for _ in range(frames(LEAD_DEPARTURE_CONFIRM)):
      release.update(True, True, lead(v=0.0, d=6.0), 1.0)
    assert not release.update(True, True, LeadObservation(), 1.0)
    assert not release.update(False, True, lead(v=0.0, d=6.0), 1.0)
    assert not release.update(True, False, lead(v=0.0, d=6.0), 1.0)

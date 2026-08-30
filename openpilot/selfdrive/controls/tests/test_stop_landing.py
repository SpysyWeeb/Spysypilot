import math

from opendbc.car.interfaces import ACCEL_MIN
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_lead import LeadObservation
from openpilot.selfdrive.controls.lib.stop_landing import (CREEP_DECEL, CREEP_FADE_SPEED, CREEP_SPEED, KISS_DECEL, KISS_SPEED, LANDING_SPEED,
                                                            LAUNCH_FRAMES, LEAD_FULL_AUTHORITY, LEAD_LANDING_GAP, RELEASE_DEADBAND, RELEASE_GAIN,
                                                            RELEASE_LIFT_MAX, STALL_RELEASE_RATE, STALL_S, StopLanding, landing_bound, landing_floor)

NO_LEAD = LeadObservation()


def lead(distance, speed=0.0, acceleration=0.0):
  return LeadObservation(True, distance, speed, acceleration, 1.0)


def frames(seconds):
  return round(seconds / DT_MDL)


def landing(v_ego=1.5, a_target=-2.0):
  law = StopLanding()
  law.update(a_target, v_ego, NO_LEAD, True)
  assert law.landing
  return law


class TestCorridor:
  def test_the_bound_falls_linearly_with_speed_and_only_removes_surplus_braking(self):
    law = StopLanding()
    assert math.isclose(landing_bound(1.5), 1.35, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(landing_bound(0.9), 0.93, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(landing_bound(KISS_SPEED), KISS_DECEL, rel_tol=1e-9, abs_tol=1e-9)
    # at the top of the window the bound sits above any comfort approach
    assert law.update(-2.0, 3.0, NO_LEAD, True) == -2.0 and not law.active
    assert math.isclose(law.update(-2.0, 1.5, NO_LEAD, True), -1.35, rel_tol=1e-9, abs_tol=1e-9)
    assert law.active
    assert math.isclose(law.update(-2.0, 0.5, NO_LEAD, True), -landing_bound(0.5), rel_tol=1e-9, abs_tol=1e-9)
    assert landing_bound(0.5) < 0.35    # the kiss arrives early enough for the ESP's ~0.7 s release to land by the wheel stop
    # a plan inside the corridor passes untouched
    assert law.update(-0.5, 1.5, NO_LEAD, True) == -0.5 and not law.active

  def test_the_floor_keeps_a_landing_braking_and_fades_above_creep_speed(self):
    assert math.isclose(landing_floor(CREEP_SPEED), CREEP_DECEL, rel_tol=1e-9, abs_tol=1e-9)
    assert landing_floor(CREEP_FADE_SPEED) == 0.0
    assert math.isclose(landing_floor(KISS_SPEED), KISS_DECEL, rel_tol=1e-9, abs_tol=1e-9)
    law = landing()
    assert math.isclose(law.update(-0.05, 0.8, NO_LEAD, True), -landing_floor(0.8), rel_tol=1e-9, abs_tol=1e-9)
    assert law.active
    # above the fade the floor is gone: an easing plan in a queue is not dragged to a stop
    assert law.update(-0.05, 2.0, NO_LEAD, True) == -0.05 and not law.active
    # the floor never starts a landing on its own
    assert StopLanding().update(-0.05, 0.8, NO_LEAD, False) == -0.05

  def test_both_edges_meet_at_the_kiss_by_walking_pace(self):
    law = landing()
    for v_ego in (KISS_SPEED, 0.1, 0.05, 0.0):
      assert math.isclose(law.update(-2.0, v_ego, NO_LEAD, True), -KISS_DECEL, rel_tol=1e-9, abs_tol=1e-9)
      assert math.isclose(law.update(-0.02, v_ego, NO_LEAD, True), -KISS_DECEL, rel_tol=1e-9, abs_tol=1e-9)
      assert math.isclose(law.update(0.05, v_ego, NO_LEAD, True), -KISS_DECEL, rel_tol=1e-9, abs_tol=1e-9)
    assert law.landing

  def test_the_law_lives_below_landing_speed_and_starts_only_on_braking_intent(self):
    law = StopLanding()
    assert law.update(-2.0, LANDING_SPEED + 0.1, NO_LEAD, True) == -2.0 and not law.landing
    assert law.update(-2.0, 1.5, NO_LEAD, False) == -2.0 and not law.landing
    assert law.update(0.5, 1.0, NO_LEAD, True) == 0.5 and not law.landing
    # a hover frame with intent is not a stop: the plan must brake more than the kiss to start one
    assert law.update(-KISS_DECEL / 2.0, 0.2, NO_LEAD, True) == -KISS_DECEL / 2.0 and not law.landing
    assert law.update(-2.0, 1.5, NO_LEAD, True) < -1.0 and law.landing
    # intent flickering off does not end it
    assert math.isclose(law.update(-2.0, 1.4, NO_LEAD, False), -landing_bound(1.4), rel_tol=1e-9, abs_tol=1e-9)
    assert law.landing
    # leaving the window does
    assert law.update(-2.0, LANDING_SPEED, NO_LEAD, True) == -2.0 and not law.landing


class TestLatchAndLaunch:
  def test_a_hover_around_zero_is_held_at_the_floor_and_never_flickers(self):
    # route 28: the MPC column lets go of the brake by 0.2 m/s and alternates +-0.1 around zero
    law = landing(0.3, -0.2)
    outputs = [law.update(a, 0.2, NO_LEAD, True) for a in (0.03, -0.2, 0.09, -0.2, 0.13, -0.05, 0.16, 0.02)]
    # every frame braking, inside the corridor: the plan's own braking passes, the rest sits on the floor
    assert all(-landing_bound(0.2) <= o <= -landing_floor(0.2) for o in outputs), outputs
    assert max(outputs) - min(outputs) < 0.05
    assert law.landing

  def test_a_climbing_plan_ends_the_landing_after_launch_frames_only_while_rolling(self):
    law = landing(0.6, -0.3)
    for i in range(LAUNCH_FRAMES - 1):
      assert law.update(0.1 * (i + 1), 0.6, NO_LEAD, True) == -landing_floor(0.6) and law.landing
    assert law.update(0.1 * LAUNCH_FRAMES, 0.6, NO_LEAD, True) == 0.1 * LAUNCH_FRAMES and not law.landing
    # and a fresh landing needs braking intent again
    assert law.update(0.3, 0.6, NO_LEAD, True) == 0.3 and not law.landing
    # at standstill the hover may drift positive without ending the landing: the planner's release is the authority there
    law = landing(0.6, -0.3)
    for _ in range(LAUNCH_FRAMES * 3):
      assert law.update(0.12, 0.05, NO_LEAD, True) == -KISS_DECEL and law.landing
    assert law.update(0.12, 0.05, NO_LEAD, True, launch=True) == 0.12 and not law.landing

  def test_the_planners_own_release_ends_the_landing_at_once(self):
    law = landing(0.2, -0.2)
    assert law.update(0.02, 0.0, NO_LEAD, True) == -KISS_DECEL
    assert law.update(0.02, 0.0, NO_LEAD, True, launch=True) == 0.02 and not law.landing

  def test_reset_forgets_the_landing(self):
    law = landing()
    law.reset()
    assert not law.landing and not law.active
    assert law.update(-2.0, 1.5, NO_LEAD, False) == -2.0


class TestLead:
  def test_a_close_lead_lifts_the_bound_but_keeps_the_floor(self):
    law = landing()
    assert law.update(-2.5, 1.0, lead(LEAD_FULL_AUTHORITY - 0.5), True) == -2.5 and not law.active
    assert math.isclose(law.update(0.05, 0.5, lead(LEAD_FULL_AUTHORITY - 0.5), True), -landing_floor(0.5), rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(law.update(-2.5, 1.0, lead(LEAD_FULL_AUTHORITY + 0.5), True), -landing_bound(1.0), rel_tol=1e-9, abs_tol=1e-9)

  def test_the_braking_needed_to_stop_behind_the_lead_always_passes(self):
    law = landing(3.4, -3.5)
    # a stopped lead 5.5 m out at 3.4 m/s needs 11.56 / (2 * 1.5) = 3.85 m/s^2 to stop LEAD_LANDING_GAP behind it: more than the law
    v_ego, distance = 3.4, LEAD_FULL_AUTHORITY + 0.5
    needed = v_ego ** 2 / (2.0 * (distance - LEAD_LANDING_GAP))
    assert needed > landing_bound(v_ego)
    assert law.update(-3.5, v_ego, lead(distance), True) == max(-needed, ACCEL_MIN)
    # the same lead far enough away needs less than the law: the law rules
    assert math.isclose(law.update(-3.5, v_ego, lead(20.0), True), -landing_bound(v_ego), rel_tol=1e-9, abs_tol=1e-9)
    # a lead moving with the car needs nothing
    assert math.isclose(law.update(-3.5, v_ego, lead(distance, speed=v_ego), True), -landing_bound(v_ego), rel_tol=1e-9, abs_tol=1e-9)


class TestWatchdog:
  def test_a_stall_shifts_the_corridor_toward_more_braking_while_rolling(self):
    law = landing()
    v_ego = 1.2
    for _ in range(frames(STALL_S)):
      out = law.update(-2.5, v_ego, NO_LEAD, True)
    assert math.isclose(out, -landing_bound(v_ego), rel_tol=1e-6, abs_tol=1e-9)
    for _ in range(frames(2.0)):
      out = law.update(-2.5, v_ego, NO_LEAD, True)
      floor = law.update(-0.05, v_ego, NO_LEAD, True)
    stalled = (frames(STALL_S) + 2 * frames(2.0) - 1) * DT_MDL - STALL_S    # the first frame under the corridor is its own low mark
    assert math.isclose(out, -(landing_bound(v_ego) + (stalled - DT_MDL) * STALL_RELEASE_RATE), rel_tol=1e-6, abs_tol=1e-9)
    assert math.isclose(floor, -(landing_floor(v_ego) + stalled * STALL_RELEASE_RATE), rel_tol=1e-6, abs_tol=1e-9)
    # slowing again is progress: the release resets
    assert math.isclose(law.update(-2.5, v_ego - 0.1, NO_LEAD, True), -landing_bound(v_ego - 0.1), rel_tol=1e-6, abs_tol=1e-9)

  def test_the_watchdog_sleeps_at_walking_pace(self):
    # a stopped car does not slow: the kiss must not drift, it is the next launch's starting point
    law = landing()
    for _ in range(frames(5.0)):
      out = law.update(-2.0, 0.05, NO_LEAD, True)
    assert math.isclose(out, -KISS_DECEL, rel_tol=1e-9, abs_tol=1e-9)


class TestReleaseLift:
  # route 0x2a (2026-08-30): the ESP follows a braking increase with ~0.2 s and a release with ~0.7 s, so the car brakes
  # harder than the plan asks through every landing; the lift asks for less in proportion to the measured surplus
  def test_the_lift_follows_the_measured_surplus_one_way_only(self):
    law = landing(2.0, -1.0)
    # the car brakes as asked, or less: nothing changes
    assert law.update(-1.0, 2.0, NO_LEAD, True, a_ego=-1.0) == -1.0
    assert law.update(-1.0, 2.0, NO_LEAD, True, a_ego=-0.6) == -1.0
    # the car brakes 0.8 harder than asked: the request is lifted by gain * surplus less the deadband
    expected = -1.0 + RELEASE_GAIN * 0.8 - RELEASE_DEADBAND
    assert math.isclose(law.update(-1.0, 2.0, NO_LEAD, True, a_ego=-1.8), expected, rel_tol=1e-9, abs_tol=1e-9)
    # capped
    assert math.isclose(law.update(-2.5, 2.0, NO_LEAD, True, a_ego=-5.0), -2.5 + RELEASE_LIFT_MAX, rel_tol=1e-9, abs_tol=1e-9)
    # no measurement, no lift
    assert law.update(-1.0, 2.0, NO_LEAD, True) == -1.0

  def test_the_lift_never_crosses_the_floor_or_the_leads_requirement(self):
    law = landing(0.8, -0.6)
    # a surplus at walking pace: the floor still holds the landing
    assert math.isclose(law.update(-0.6, 0.8, NO_LEAD, True, a_ego=-1.6), -landing_floor(0.8), rel_tol=1e-9, abs_tol=1e-9)
    # a stopped lead 8 m out at 3 m/s needs 9 / (2 * 4) = 1.125 m/s^2: the lift stops there
    law = landing(3.0, -1.5)
    needed = 3.0 ** 2 / (2.0 * (8.0 - LEAD_LANDING_GAP))
    assert math.isclose(law.update(-1.5, 3.0, lead(8.0), True, a_ego=-2.5), -needed, rel_tol=1e-9, abs_tol=1e-9)
    # a plan already braking less than the requirement is left alone, never pushed down to it (route 0x2a replay: a radar
    # return at 1.1 m once turned this clamp into -3.5 m/s^2)
    assert law.update(-0.8, 3.0, lead(8.0), True, a_ego=-0.8) == -0.8
    assert law.update(-0.8, 3.0, lead(8.0), True, a_ego=-2.0) == -0.8

  def test_the_lift_sleeps_at_walking_pace_and_on_a_positive_plan(self):
    law = landing(0.2, -0.2)
    assert math.isclose(law.update(-0.2, 0.1, NO_LEAD, True, a_ego=-1.0), -KISS_DECEL, rel_tol=1e-9, abs_tol=1e-9)
    law = landing(1.0, -0.5)
    assert law.update(0.05, 1.0, NO_LEAD, True, a_ego=-1.0) == -landing_floor(1.0)

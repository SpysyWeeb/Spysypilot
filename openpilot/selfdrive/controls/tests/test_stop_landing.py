import math

from opendbc.car.interfaces import ACCEL_MIN
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_lead import LeadObservation
from openpilot.selfdrive.controls.lib.stop_landing import (CREEP_DECEL, CREEP_SPEED, LANDING_C, LANDING_K, LANDING_SPEED, LEAD_FULL_AUTHORITY,
                                                            LEAD_LANDING_GAP, STALL_RELEASE_RATE, STALL_S, STANDSTILL_SPEED, StopLanding,
                                                            landing_bound)

NO_LEAD = LeadObservation()


def lead(distance, speed=0.0, acceleration=0.0):
  return LeadObservation(True, distance, speed, acceleration, 1.0)


def frames(seconds):
  return round(seconds / DT_MDL)


class TestLandingBound:
  def test_the_bound_falls_linearly_with_speed_and_only_removes_surplus_braking(self):
    law = StopLanding()
    assert math.isclose(landing_bound(1.5), 1.35, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(landing_bound(0.5), 0.65, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(landing_bound(0.0), LANDING_C, rel_tol=1e-9, abs_tol=1e-9)
    # at the top of the window the bound sits above any comfort approach
    assert law.update(-2.0, 3.0, NO_LEAD, True) == -2.0 and not law.active
    assert math.isclose(law.update(-2.0, 1.5, NO_LEAD, True), -(LANDING_K * 1.5 + LANDING_C), rel_tol=1e-9, abs_tol=1e-9)
    assert law.active
    assert math.isclose(law.update(-2.0, 0.5, NO_LEAD, True), -0.65, rel_tol=1e-9, abs_tol=1e-9)
    # a plan already inside the law passes untouched
    assert law.update(-0.5, 1.5, NO_LEAD, True) == -0.5 and not law.active

  def test_the_law_lives_only_in_the_window(self):
    law = StopLanding()
    assert law.update(-2.0, LANDING_SPEED + 0.1, NO_LEAD, True) == -2.0
    assert law.update(-2.0, STANDSTILL_SPEED / 2.0, NO_LEAD, True) == -2.0
    assert law.update(0.5, 1.0, NO_LEAD, True) == 0.5
    assert not law.landing

  def test_a_landing_needs_stop_intent_and_then_survives_its_flicker(self):
    law = StopLanding()
    assert law.update(-2.0, 1.5, NO_LEAD, False) == -2.0 and not law.landing
    assert law.update(-2.0, 1.5, NO_LEAD, True) < -1.0 and law.landing
    # the model's stop bit flickers off for a frame: the landing holds
    assert math.isclose(law.update(-2.0, 1.4, NO_LEAD, False), -landing_bound(1.4), rel_tol=1e-9, abs_tol=1e-9)
    # the plan lifting ends the landing, and intent is needed again afterwards
    assert law.update(0.2, 1.4, NO_LEAD, False) == 0.2 and not law.landing
    assert law.update(-2.0, 1.4, NO_LEAD, False) == -2.0 and not law.landing


class TestLead:
  def test_a_close_lead_keeps_full_authority(self):
    law = StopLanding()
    assert law.update(-2.5, 1.0, lead(LEAD_FULL_AUTHORITY - 0.5), True) == -2.5 and not law.active
    assert law.update(-2.5, 1.0, lead(LEAD_FULL_AUTHORITY + 0.5), True) == -landing_bound(1.0)

  def test_the_braking_needed_to_stop_behind_the_lead_always_passes(self):
    law = StopLanding()
    # a stopped lead 5.5 m out at 3.4 m/s needs 11.56 / (2 * 1.5) = 3.85 m/s^2 to stop LEAD_LANDING_GAP behind it: more than the law
    v_ego, distance = 3.4, LEAD_FULL_AUTHORITY + 0.5
    needed = v_ego ** 2 / (2.0 * (distance - LEAD_LANDING_GAP))
    assert needed > landing_bound(v_ego)
    assert law.update(-3.5, v_ego, lead(distance), True) == max(-needed, ACCEL_MIN)
    # the same lead far enough away needs less than the law: the law rules
    assert math.isclose(law.update(-3.5, v_ego, lead(20.0), True), -landing_bound(v_ego), rel_tol=1e-9, abs_tol=1e-9)
    # a lead moving with the car needs nothing
    assert math.isclose(law.update(-3.5, v_ego, lead(distance, speed=v_ego), True), -landing_bound(v_ego), rel_tol=1e-9, abs_tol=1e-9)


class TestGuards:
  def test_the_watchdog_releases_a_bound_that_holds_speed(self):
    law = StopLanding()
    v_ego = 1.2
    for _ in range(frames(STALL_S)):
      out = law.update(-2.5, v_ego, NO_LEAD, True)
    assert math.isclose(out, -landing_bound(v_ego), rel_tol=1e-6, abs_tol=1e-9)
    for _ in range(frames(2.0)):
      out = law.update(-2.5, v_ego, NO_LEAD, True)
    stalled = (frames(STALL_S) + frames(2.0) - 1) * DT_MDL - STALL_S    # the first frame under the bound is its own low mark
    assert math.isclose(out, -(landing_bound(v_ego) + stalled * STALL_RELEASE_RATE), rel_tol=1e-6, abs_tol=1e-9)
    # slowing again is progress: the release resets
    out = law.update(-2.5, v_ego - 0.1, NO_LEAD, True)
    assert math.isclose(out, -landing_bound(v_ego - 0.1), rel_tol=1e-6, abs_tol=1e-9)

  def test_the_creep_floor_keeps_a_landing_plan_braking(self):
    law = StopLanding()
    assert law.update(-0.2, CREEP_SPEED - 0.2, NO_LEAD, True) == -CREEP_DECEL and law.active
    assert law.update(-0.2, CREEP_SPEED + 0.2, NO_LEAD, True) == -0.2 and not law.active
    # the floor is a landing matter: without intent a gentle plan at walking pace is left alone
    fresh = StopLanding()
    assert fresh.update(-0.2, CREEP_SPEED - 0.2, NO_LEAD, False) == -0.2

  def test_reset_forgets_the_landing(self):
    law = StopLanding()
    law.update(-2.0, 1.5, NO_LEAD, True)
    assert law.landing
    law.reset()
    assert not law.landing and not law.active
    assert law.update(-2.0, 1.5, NO_LEAD, False) == -2.0

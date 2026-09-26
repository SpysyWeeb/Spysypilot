import math

import numpy as np

from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR
from opendbc.car.interfaces import ACCEL_MIN
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_lead import LeadObservation
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import STOP_DISTANCE
from openpilot.selfdrive.controls.lib.longitudinal_planner import J_CRUISE_BP, J_CRUISE_VALS, get_cruise_accel
from openpilot.selfdrive.controls.lib.stop_landing import (CREEP_FADE_SPEED, CREEP_SPEED, KISS_DECEL, KISS_SPEED, LANDING_SPEED,
                                                            LAUNCH_FRAMES, LEAD_MIN_GAP_BUDGET, STALL_MAX, STANDSTILL_SPEED, StopLanding,
                                                            landing_bound, landing_floor)

NO_LEAD = LeadObservation()
# the planner's action_t on the Palisade: the car's actuator delay plus one model frame
ACTION_T = CarInterface.get_non_essential_params(CAR.HYUNDAI_PALISADE).longitudinalActuatorDelay + DT_MDL
# and its cruise jerk, which a landing hands the output back at
RELEASE_JERK = (J_CRUISE_BP, J_CRUISE_VALS)


def lead(distance, speed=0.0, acceleration=0.0):
  return LeadObservation(True, distance, speed, acceleration)


def frames(seconds):
  return round(seconds / DT_MDL)


def landing(v_ego=1.5, a_target=-2.0, action_t=ACTION_T):
  law = StopLanding(DT_MDL, action_t, RELEASE_JERK)
  law.update(a_target, v_ego, NO_LEAD, True)
  assert law.landing
  return law


def corridor_ride(edge, v0, h=1e-4):
  # a car riding a corridor edge exactly, v' = -edge(v), integrated forward in time from v0
  ts, vs = [0.0], [v0]
  while vs[-1] > KISS_SPEED - 0.1:
    vs.append(vs[-1] - h * edge(vs[-1]))
    ts.append(ts[-1] + h)
  return np.array(ts), np.array(vs)


def cruise_carry(a_prev, v_ego):
  # the planner's cruise candidate carrying a hand-over up from a_prev toward a set speed far above
  return get_cruise_accel(True, v_ego + 10.0, v_ego, a_prev, DT_MDL, 0.0, True)


def stopping_needs(v_ego, distance, speed=0.0, acceleration=0.0):
  # the constant deceleration that still stops a car coasting through ACTION_T LEAD_MIN_GAP_BUDGET short of where the lead
  # comes to rest
  rest = distance + (speed * speed / (-2.0 * acceleration) if acceleration < 0.0 else 0.0)
  room = rest - v_ego * ACTION_T - LEAD_MIN_GAP_BUDGET
  return v_ego * v_ego / (2.0 * room) if room > 0.0 else math.inf


class TestCorridor:
  def test_the_bound_only_removes_braking_and_the_floor_only_adds_it_up_to_itself(self):
    law = landing()
    for v_ego in np.linspace(LANDING_SPEED - 0.01, 0.0, 36):
      for a_target in np.linspace(-3.5, -0.1, 18):
        out = law.update(a_target, v_ego, NO_LEAD, True)
        assert out >= min(a_target, -law.floor(v_ego)) - 1e-9, (v_ego, a_target, out)
        assert out <= a_target + 1e-9 or math.isclose(out, -law.bound(v_ego), rel_tol=1e-9, abs_tol=1e-9), (v_ego, a_target, out)

  def test_without_a_delay_the_request_is_the_car_side_corridor(self):
    law = landing(LANDING_SPEED - 0.01, -3.0, action_t=0.0)
    for v_ego in np.linspace(LANDING_SPEED - 0.01, 0.0, 50):
      assert math.isclose(law.update(-3.0, v_ego, NO_LEAD, True), -landing_bound(v_ego), rel_tol=1e-9, abs_tol=1e-9)
      if v_ego < CREEP_FADE_SPEED:
        assert math.isclose(law.update(0.0, v_ego, NO_LEAD, True), -landing_floor(v_ego), rel_tol=1e-9, abs_tol=1e-9)

  def test_each_edge_is_requested_one_actuator_delay_ahead_of_the_car(self):
    # the request made at a speed is what a car riding the corridor carries action_t later: where the car reaches a
    # breakpoint, the request made action_t earlier asks for that breakpoint's deceleration
    for edge, v0, breakpoints, plan in ((landing_bound, 1.8, (0.9, KISS_SPEED), -3.0), (landing_floor, 1.4, (CREEP_SPEED, KISS_SPEED), 0.0)):
      ts, vs = corridor_ride(edge, v0)
      law = landing(v0, -3.0)
      for v_car in breakpoints:
        t_car = np.interp(-v_car, -vs, ts)
        v_request = float(np.interp(t_car - ACTION_T, ts, vs))
        assert v_request > v_car + 0.05
        assert math.isclose(law.update(plan, v_request, NO_LEAD, True), -edge(v_car), abs_tol=2e-3), (edge.__name__, v_car)
    # the ceiling is left where it is: the plan is below it at the top of the window
    law = landing(LANDING_SPEED - 0.01, -3.0)
    assert math.isclose(law.update(-3.0, LANDING_SPEED - 1e-6, NO_LEAD, True), -landing_bound(LANDING_SPEED), abs_tol=1e-4)

  def test_a_plan_inside_the_corridor_passes_untouched(self):
    law = landing()
    # at the top of the window the request sits above any comfort approach
    assert law.update(-2.0, 3.0, NO_LEAD, True) == -2.0
    assert law.update(-0.5, 1.5, NO_LEAD, True) == -0.5
    assert law.update(-2.0, 1.5, NO_LEAD, True) == -law.bound(1.5)

  def test_the_floor_keeps_a_landing_braking_and_fades_above_creep_speed(self):
    law = landing()
    assert law.update(-0.05, 0.8, NO_LEAD, True) == -law.floor(0.8)
    # above the fade the floor is gone: an easing plan in a queue is not dragged to a stop
    assert law.update(-0.05, 2.0, NO_LEAD, True) == -0.05
    # the floor never starts a landing on its own
    assert StopLanding(DT_MDL, ACTION_T, RELEASE_JERK).update(-0.05, 0.8, NO_LEAD, False) == -0.05

  def test_both_edges_meet_at_the_kiss_by_walking_pace(self):
    law = landing()
    for v_ego in (KISS_SPEED, 0.1, 0.05, 0.0):
      assert math.isclose(law.update(-2.0, v_ego, NO_LEAD, True), -KISS_DECEL, rel_tol=1e-9, abs_tol=1e-9)
      assert math.isclose(law.update(-0.02, v_ego, NO_LEAD, True), -KISS_DECEL, rel_tol=1e-9, abs_tol=1e-9)
      assert math.isclose(law.update(0.05, v_ego, NO_LEAD, True), -KISS_DECEL, rel_tol=1e-9, abs_tol=1e-9)
    assert law.landing

  def test_the_law_lives_below_landing_speed_and_starts_only_on_braking_intent(self):
    law = StopLanding(DT_MDL, ACTION_T, RELEASE_JERK)
    assert law.update(-2.0, LANDING_SPEED + 0.1, NO_LEAD, True) == -2.0 and not law.landing
    assert law.update(-2.0, 1.5, NO_LEAD, False) == -2.0 and not law.landing
    assert law.update(0.5, 1.0, NO_LEAD, True) == 0.5 and not law.landing
    # a hover frame with intent is not a stop: the plan must brake more than the kiss to start one
    assert law.update(-KISS_DECEL / 2.0, 0.2, NO_LEAD, True) == -KISS_DECEL / 2.0 and not law.landing
    assert law.update(-2.0, 1.5, NO_LEAD, True) < -0.5 and law.landing
    # intent flickering off does not end it
    assert law.update(-2.0, 1.4, NO_LEAD, False) == -law.bound(1.4)
    assert law.landing
    # leaving the window does
    assert law.update(-2.0, LANDING_SPEED, NO_LEAD, True) == -2.0 and not law.landing


class TestNoLoopOnTheMeasuredDeceleration:
  def test_the_request_does_not_depend_on_a_car_braking_beyond_the_floor(self):
    for v_ego in (0.6, 1.0, 1.5, 2.0, 3.0):
      for a_target in (-0.5, -1.0, -2.5):
        outputs = set()
        for surplus in (0.0, 0.3, 1.0, 2.5):
          law = landing(v_ego, a_target)
          a_ego = -landing_floor(v_ego) - surplus
          outputs.add(tuple(law.update(a_target, v_ego, NO_LEAD, True, a_ego=a_ego) for _ in range(frames(1.0))))
        assert len(outputs) == 1, (v_ego, a_target, outputs)


class TestLatchAndLaunch:
  def test_a_nonfinite_speed_leaves_the_plan_alone(self):
    law = landing()
    assert law.update(-2.0, math.nan, NO_LEAD, True) == -2.0 and not law.landing
    assert law.update(-2.0, math.inf, NO_LEAD, True) == -2.0
    assert math.isclose(law.update(-2.0, -0.01, NO_LEAD, True), -2.0, rel_tol=1e-9, abs_tol=1e-9) or law.landing

  def test_a_nonfinite_target_leaves_the_plan_alone_and_does_not_corrupt_the_launch_count(self):
    # a solver-divergence frame (get_accel_from_plan has no isfinite guard) must not spend one of the LAUNCH_FRAMES
    # a climbing plan needs to end a landing -- the bad frame is skipped outright, not counted as a non-positive one
    law = landing(0.6, -0.3)
    assert law.update(0.1, 0.6, NO_LEAD, True) == -law.floor(0.6) and law.landing     # 1 of LAUNCH_FRAMES
    assert law.update(0.2, 0.6, NO_LEAD, True) == -law.floor(0.6) and law.landing     # 2 of LAUNCH_FRAMES
    assert math.isnan(law.update(math.nan, 0.6, NO_LEAD, True)) and law.landing       # the bad frame passes through untouched
    law.update(0.3, 0.6, NO_LEAD, True)
    assert not law.landing                                                            # the 3rd positive frame still ends it
    fresh = StopLanding(DT_MDL, ACTION_T, RELEASE_JERK)
    assert math.isnan(fresh.update(math.nan, 1.5, NO_LEAD, True))
    assert not fresh.landing                                                          # nor does a bad frame start one

  def test_a_hover_around_zero_is_held_at_the_floor_and_never_flickers(self):
    # the MPC column lets go of the brake by 0.2 m/s and alternates +-0.1 around zero
    law = landing(0.3, -0.2)
    outputs = [law.update(a, 0.2, NO_LEAD, True) for a in (0.03, -0.2, 0.09, -0.2, 0.13, -0.05, 0.16, 0.02)]
    # every frame braking, inside the corridor: the plan's own braking passes, the rest sits on the floor
    assert all(-law.bound(0.2) <= o <= -law.floor(0.2) for o in outputs), outputs
    assert max(outputs) - min(outputs) < 0.05
    assert law.landing

  def test_a_launch_frame_never_starts_a_landing(self):
    law = StopLanding(DT_MDL, ACTION_T, RELEASE_JERK)
    for _ in range(frames(1.0)):
      out = law.update(-0.3, 0.05, NO_LEAD, True, launch=True)
      assert not law.landing
      assert out == -0.3                                                    # the plan passes through untouched, every frame
    law.update(-0.3, 0.05, NO_LEAD, True, launch=False)
    assert law.landing                                                      # the next stop intent without a launch latches as before

  def test_a_climbing_plan_ends_the_landing_after_launch_frames_only_while_rolling(self):
    law = landing(0.6, -0.3)
    for i in range(LAUNCH_FRAMES - 1):
      assert law.update(0.1 * (i + 1), 0.6, NO_LEAD, True) == -law.floor(0.6) and law.landing
    law.update(0.1 * LAUNCH_FRAMES, 0.6, NO_LEAD, True)
    assert not law.landing
    # and a fresh landing needs braking intent again
    law.update(0.3, 0.6, NO_LEAD, True)
    assert not law.landing
    # at standstill the hover may drift positive without ending the landing: the planner's release is the authority there
    law = landing(0.6, -0.3)
    for _ in range(LAUNCH_FRAMES * 3):
      assert law.update(0.12, 0.05, NO_LEAD, True) == -KISS_DECEL and law.landing
    law.update(0.12, 0.05, NO_LEAD, True, launch=True)
    assert not law.landing

  def test_the_planners_own_release_ends_the_landing_at_once(self):
    law = landing(0.2, -0.2)
    assert law.update(0.02, 0.0, NO_LEAD, True) == -KISS_DECEL
    law.update(0.02, 0.0, NO_LEAD, True, launch=True)
    assert not law.landing

  def test_a_launch_taken_back_at_standstill_latches_the_landing_again(self):
    # the release ended the landing and the plan hovers above the kiss, where no landing would start: the planner takes the
    # launch back while the car still stands, and the landing it ended holds the car on the kiss again, braking passing at once
    law = landing(0.2, -0.2)
    law.update(0.13, 0.0, NO_LEAD, True, launch=True)
    law.update(0.13, 0.0, NO_LEAD, True, launch=True)
    assert law.update(0.13, 0.0, NO_LEAD, True) > -KISS_DECEL and not law.landing
    assert law.update(0.13, 0.0, NO_LEAD, True, launch_cancelled=True) == -KISS_DECEL and law.landing and not law.releasing
    assert law.update(0.13, 0.0, NO_LEAD, True) == -KISS_DECEL and law.landing
    # a launch starting on the frame it is taken back, or a car already rolling, is not the landing's to restore
    law = landing(0.2, -0.2)
    law.update(0.13, 0.0, NO_LEAD, True, launch=True)
    law.update(0.13, 0.0, NO_LEAD, True, launch=True, launch_cancelled=True)
    assert not law.landing
    law.update(0.13, STANDSTILL_SPEED + 0.05, NO_LEAD, True, launch_cancelled=True)
    assert not law.landing

  def test_reset_forgets_the_landing_and_its_release(self):
    law = landing(0.2, -0.2)
    law.update(0.02, 0.0, NO_LEAD, True)
    law.update(0.5, 0.0, NO_LEAD, True, launch=True)
    assert law.releasing
    law.reset()
    assert not law.landing and not law.releasing
    assert law.update(-2.0, 1.5, NO_LEAD, False) == -2.0
    assert law.update(0.5, 1.5, NO_LEAD, False) == 0.5


class TestLead:
  def test_the_output_is_continuous_as_a_close_lead_crosses_5_m(self):
    law = landing(0.33, -0.8)
    outputs = [law.update(-0.8, 0.33, lead(d), True, a_ego=-0.6) for d in np.arange(5.1, 4.9, -0.001)]
    outputs.append(law.update(-0.8, 0.33, lead(5.0), True, a_ego=-0.6))
    assert max(outputs) - min(outputs) < 1e-6, (min(outputs), max(outputs))

  def test_the_output_is_continuous_as_a_stopped_lead_closes(self):
    law = landing(2.0, -3.0)
    outputs = [law.update(-3.0, 2.0, lead(d), True) for d in np.arange(12.0, 1.0, -0.001)]
    assert max(abs(b - a) for a, b in zip(outputs, outputs[1:], strict=False)) < 0.01

  def test_the_braking_passed_grows_without_limit_as_the_room_left_after_the_delay_closes(self):
    # at walking pace the lead's requirement at STOP_DISTANCE's geometry is below this plan; closer in, the room the car has
    # once its requests in flight have played out takes over, continuously, until the plan's whole braking passes
    law = landing(0.5, -1.0)
    outputs = [law.update(-1.0, 0.5, lead(d), True) for d in np.arange(3.0, 0.05, -0.001)]
    assert all(b <= a + 1e-12 for a, b in zip(outputs, outputs[1:], strict=False))
    assert max(abs(b - a) for a, b in zip(outputs, outputs[1:], strict=False)) < 0.01
    assert outputs[0] > -0.5 and outputs[-1] == -1.0

  def test_a_lead_inside_the_gap_budget_gets_what_stops_the_car_short_of_it_after_the_delay(self):
    # returns and stopped cars that appear inside a latched landing at walking pace: whatever the plan asks passes, up to
    # what stops a car coasting through the actuator delay short of the lead
    for v_ego, obstacle, plan in ((0.37, (0.32,), -0.6), (1.0, (1.0,), -2.0), (0.7, (0.6,), -1.5), (1.5, (1.1,), -3.5),
                                  (3.0, (1.0, 3.0, -4.0), -3.5), (0.5, (1.2,), -1.0), (0.5, (0.8,), -1.0)):
      law = landing(v_ego, -2.0)
      out = law.update(plan, v_ego, lead(*obstacle), True, a_ego=-0.5)
      assert -out >= min(stopping_needs(v_ego, *obstacle), -plan) - 1e-9, (v_ego, obstacle, out)

  def test_the_braking_needed_to_stop_behind_the_lead_always_passes(self):
    law = landing(3.4, -3.5)
    # a stopped lead 1.5 m beyond STOP_DISTANCE at 3.4 m/s needs 3.4^2 / (2 * 1.5) = 3.85 m/s^2: more than the law
    v_ego, distance = 3.4, STOP_DISTANCE + 1.5
    needed = v_ego ** 2 / (2.0 * (distance - STOP_DISTANCE))
    assert needed > law.bound(v_ego)
    assert law.update(-3.5, v_ego, lead(distance), True) == max(-needed, ACCEL_MIN)
    # the same lead far enough away needs less than the law: the law rules
    assert law.update(-3.5, v_ego, lead(40.0), True) == -law.bound(v_ego)
    # a lead moving with the car needs nothing
    assert law.update(-3.5, v_ego, lead(distance, speed=v_ego), True) == -law.bound(v_ego)
    # inside STOP_DISTANCE the requirement is planned in no less than LEAD_MIN_GAP_BUDGET of room, and the floor stays
    assert math.isclose(law.update(-3.0, 1.5, lead(3.0), True), -1.5 ** 2 / (2.0 * LEAD_MIN_GAP_BUDGET), rel_tol=1e-9)
    assert law.update(0.05, 0.5, lead(4.5), True) == -law.floor(0.5)

  def test_a_plan_braking_less_than_the_leads_requirement_is_left_alone(self):
    # the requirement lets braking through; it never adds any
    for v_ego, distance, plan in ((1.0, 4.0, -0.8), (0.37, 0.32, -0.3), (1.5, 1.1, -0.2)):
      law = landing(v_ego, -2.0)
      assert law.update(plan, v_ego, lead(distance), True) == plan


class TestAntiStall:
  def test_a_car_not_slowing_presses_both_edges_more_every_frame_up_to_the_cap(self):
    # creep torque beating the kiss: the car holds its speed or re-accelerates while the landing asks for braking
    law = landing(0.3, -0.3)
    kiss = [law.update(-0.3, 0.3, NO_LEAD, True, a_ego=0.1) for _ in range(frames(3.0))]
    assert kiss[0] < -KISS_DECEL
    assert all(b < a for a, b in zip(kiss, kiss[1:frames(1.0)], strict=False))
    assert math.isclose(kiss[-1], -(KISS_DECEL + STALL_MAX), rel_tol=1e-9)
    # the same shortfall above the kiss speed opens the bound too
    law = landing(1.0, -3.0)
    bound = [law.update(-3.0, 1.0, NO_LEAD, True, a_ego=0.0) for _ in range(frames(1.0))]
    assert bound[0] < -law.bound(1.0) and all(b < a for a, b in zip(bound, bound[1:], strict=False))

  def test_the_press_drains_as_the_car_slows_again(self):
    law = landing(0.3, -0.3)
    for _ in range(frames(1.0)):
      pressed = law.update(-0.3, 0.3, NO_LEAD, True, a_ego=0.1)
    drained = [law.update(-0.3, 0.3, NO_LEAD, True, a_ego=-0.6) for _ in range(frames(2.0))]
    assert drained[0] > pressed
    assert all(b >= a for a, b in zip(drained, drained[1:], strict=False))
    assert drained[-1] == -KISS_DECEL

  def test_nothing_is_pressed_at_standstill_and_the_press_is_held_for_the_next_roll(self):
    law = landing(0.3, -0.3)
    for _ in range(frames(1.0)):
      pressed = law.update(-0.3, 0.3, NO_LEAD, True, a_ego=0.1)
    for _ in range(frames(2.0)):
      assert law.update(-0.9, STANDSTILL_SPEED, NO_LEAD, True, a_ego=0.2) == -KISS_DECEL
      assert law.update(-0.3, 0.0, NO_LEAD, True, a_ego=0.0) == -KISS_DECEL
    rolled = law.update(-0.3, 0.3, NO_LEAD, True, a_ego=-KISS_DECEL)
    assert pressed < rolled < -KISS_DECEL - 0.9 * (-KISS_DECEL - pressed)

  def test_nothing_winds_it_up_where_the_floor_has_faded_out(self):
    # the floor is zero here, so the request is the plan inside the bound whatever the car does -- a landing latched while
    # the car still carries a queue launch's acceleration included
    for a_ego in (0.0, -0.3, -1.5, 0.4, 1.8):
      law = landing(3.4, -2.5)
      for v_ego in (3.4, 2.5, CREEP_FADE_SPEED):
        for i in range(frames(2.0)):
          a_target = -2.5 if i % 2 else -0.4
          assert law.update(a_target, v_ego, NO_LEAD, True, a_ego=a_ego) == max(a_target, -law.bound(v_ego))
          assert law.integral == 0.0
      assert law.update(0.0, 1.2, NO_LEAD, True, a_ego=-landing_floor(1.2)) == -law.floor(1.2)

  def test_a_press_fades_in_and_out_with_the_floor(self):
    # a queue launch the plan turns into a stop: latched while the car still accelerates, it climbs through the floor's fade
    # and brakes back down through it. The press never steps the request by more than the plan does
    for v_ego, launch, seconds, braking, easing in ((1.2, 1.0, 0.4, -0.4, -0.2), (1.0, 1.5, 0.5, -0.4, -0.1)):
      law = StopLanding(DT_MDL, ACTION_T, RELEASE_JERK)
      plans, outputs, pressed = [], [], 0.0
      for a_ego, plan, duration in ((launch, -0.3, seconds), (braking, easing, 2.0)):
        for _ in range(frames(duration)):
          plans.append(plan)
          outputs.append(law.update(plan, v_ego, NO_LEAD, True, a_ego=a_ego))
          pressed = max(pressed, law.integral)
          v_ego += a_ego * DT_MDL
      assert pressed > 0.1
      assert np.abs(np.diff(outputs)).max() <= np.abs(np.diff(plans)).max() + 1e-9, np.abs(np.diff(outputs)).max()
    # a press built by creep just below the fade, then the speed wandering across it: no step on the way
    law = landing(1.4, -0.5)
    for _ in range(frames(1.0)):
      law.update(-0.1, 1.4, NO_LEAD, True, a_ego=0.2)
    assert law.integral > 0.0
    speeds = np.concatenate((np.arange(1.4, 1.6, 0.001), np.arange(1.6, 1.4, -0.001)))
    outputs = [law.update(-0.1, v, NO_LEAD, True) for v in speeds]
    assert np.abs(np.diff(outputs)).max() < 0.01

  def test_a_car_creeping_under_a_hovering_plan_is_pressed(self):
    # the MPC hovers around zero behind the stop while creep torque beats the kiss: the press follows the car, not the plan
    law = landing(0.3, -0.3)
    hover = (0.03, -0.05, 0.09, 0.8)
    outputs = [law.update(hover[i % len(hover)], 0.3, NO_LEAD, True, a_ego=0.1) for i in range(frames(1.0))]
    assert outputs[0] < -KISS_DECEL
    assert all(b < a for a, b in zip(outputs, outputs[1:], strict=False))
    assert law.landing


class TestRelease:
  def assert_handed_back(self, law, held, frames_):
    # from where the landing held it, the output rises as the planner's cruise carry would until it meets the plan, then
    # it is the plan
    prev, met = held, False
    for plan, v_ego, launch, a_ego in frames_:
      out = law.update(plan, v_ego, NO_LEAD, True, launch, a_ego)
      carried = cruise_carry(prev, v_ego)
      met = met or plan <= carried
      assert math.isclose(out, plan if met else carried, rel_tol=0.0, abs_tol=1e-9), (plan, v_ego, prev, out)
      prev = out
    assert met and not law.landing and not law.releasing

  def test_a_launch_from_the_kiss_is_handed_back_at_the_cruise_carrys_rate(self):
    # the lead-departure pre-release at standstill, the MPC already launching
    law = landing(0.2, -0.2)
    for _ in range(frames(1.0)):
      law.update(0.05, 0.0, NO_LEAD, True)
    held = law.output
    assert held == -KISS_DECEL
    self.assert_handed_back(law, held, [(0.9, 0.0, True, 0.0)] + [(0.9, 0.01 * i, True, 0.3) for i in range(1, frames(1.5))])

  def test_a_launch_from_the_floor_while_rolling_is_handed_back_at_the_cruise_carrys_rate(self):
    # a Force Stops release while the floor holds a car still at walking pace
    law = landing(1.0, -0.5)
    for _ in range(frames(0.5)):
      law.update(-0.1, 1.0, NO_LEAD, True, a_ego=-0.5)
    held = law.output
    assert held == -law.floor(1.0)
    self.assert_handed_back(law, held, [(0.6, 1.0, True, -0.3)] * frames(1.5))

  def test_a_plan_climbing_out_of_a_pressed_landing_is_handed_back_at_the_cruise_carrys_rate(self):
    # following a queue creeping ahead: the car holds its speed, the press builds on the floor, and the plan climbs out
    law = landing(0.93, -0.19)
    for _ in range(frames(1.0)):
      law.update(-0.19, 0.93, NO_LEAD, True, a_ego=0.0)
    for plan in (0.04, 0.1):
      law.update(plan, 0.93, NO_LEAD, True, a_ego=0.0)
    assert law.landing and law.integral > 0.3
    held = law.output
    self.assert_handed_back(law, held, [(0.3, 0.93, False, 0.0)] * frames(1.5))

  def test_the_press_adds_nothing_to_the_exit_step(self):
    # braking beyond the floor, holding speed, creeping forward: however much press the landing held when the plan climbed
    # out of it, the output comes back at the one release rate
    for a_ego, pressed in ((-0.5, False), (0.0, True), (0.2, True)):
      law = landing(0.93, -0.19)
      for _ in range(frames(1.0)):
        law.update(-0.19, 0.93, NO_LEAD, True, a_ego=a_ego)
      assert (law.integral > 0.3) == pressed
      outs = [law.output] + [law.update(p, 0.93, NO_LEAD, True, a_ego=a_ego) for p in (0.04, 0.1, 0.3, 0.3)]
      assert not law.landing
      assert math.isclose(max(np.diff(outs)), cruise_carry(0.0, 0.93), rel_tol=0.0, abs_tol=1e-9), (a_ego, outs)

  def test_a_nonfinite_target_during_a_release_passes_and_the_release_carries_on(self):
    # a solver-divergence frame mid-release touches no state: the next frame goes on from the last output the car was told
    law = landing(0.93, -0.19)
    for _ in range(frames(1.0)):
      law.update(-0.19, 0.93, NO_LEAD, True, a_ego=0.0)
    outs = [law.update(p, 0.93, NO_LEAD, True, a_ego=0.0) for p in (0.04, 0.1, 0.8, 0.8, 0.8)]
    assert not law.landing and law.releasing
    assert math.isnan(law.update(math.nan, 0.93, NO_LEAD, True, a_ego=0.0)) and law.releasing
    assert math.isclose(law.update(0.8, 0.93, NO_LEAD, True, a_ego=0.0), cruise_carry(outs[-1], 0.93), rel_tol=0.0, abs_tol=1e-9)
    assert law.releasing

  def test_only_what_the_landing_held_is_carried(self):
    # a landing that passed the plan hands nothing back: the plan's own climb out of it passes as it is
    law = landing(2.0, -0.5)
    plans = [-0.5, 0.2, 0.7, 1.2, 1.6]
    assert [law.update(p, 2.0, NO_LEAD, True) for p in plans] == plans
    assert not law.landing and not law.releasing
    # nor does one whose bound was taking braking off the plan: the braking the plan asks passes at once, and so does its climb
    law = landing(1.0, -3.0)
    assert law.update(-3.0, 1.0, NO_LEAD, True, launch=True) == -3.0 and not law.releasing
    law = landing(1.0, -3.0)
    assert law.update(0.8, 1.0, NO_LEAD, True, launch=True) == 0.8 and not law.releasing
    # the same holds for a landing starting again during a release with its bound lifting the plan above the running ceiling
    law = landing(1.0, -0.5)
    for _ in range(frames(1.0)):
      law.update(-0.1, 1.0, NO_LEAD, True, a_ego=0.2)
    for p in (0.1, 0.2, 0.3, 0.8):
      law.update(p, 1.0, NO_LEAD, True, a_ego=0.0)
    assert law.update(-3.0, 1.0, NO_LEAD, True) > -3.0 and law.landing and law.releasing
    assert law.update(0.9, 1.0, NO_LEAD, True, launch=True) == 0.9 and not law.releasing
    # braking the plan asks during a release passes at once, and ends the release: the plan's next climb is its own
    law = landing(0.2, -0.2)
    law.update(0.02, 0.0, NO_LEAD, True)
    assert law.update(0.9, 0.0, NO_LEAD, True, launch=True) < 0.0 and law.releasing
    assert law.update(-1.0, 0.05, NO_LEAD, True, launch=True) == -1.0 and not law.releasing
    assert law.update(0.9, 0.05, NO_LEAD, False) == 0.9

  def test_a_landing_starting_again_during_a_release_does_not_drop_it(self):
    # the plan climbs out of a pressed landing and turns straight back to braking: the new landing starts without the press,
    # and the output still rises no faster than the release until it meets the new landing's corridor
    law = landing(1.0, -0.5)
    for _ in range(frames(1.0)):
      law.update(-0.1, 1.0, NO_LEAD, True, a_ego=0.2)
    assert law.integral > 0.45
    outs = [law.output] + [law.update(p, 1.0, NO_LEAD, True, a_ego=-0.5) for p in (0.1, 0.2, 0.3) + (-2.0,) * frames(0.5)]
    assert law.landing
    step = cruise_carry(0.0, 1.0)
    assert -law.bound(1.0) - outs[3] > 2.0 * step
    assert max(np.diff(outs)) <= step + 1e-9
    assert outs[-1] == -law.bound(1.0)

  def test_the_speed_leaving_the_window_does_not_cut_a_release_short(self):
    law = landing(1.2, -0.5)
    law.update(-0.1, 1.2, NO_LEAD, True, a_ego=-0.5)
    v_ego, outs, beyond = 1.2, [law.output], []
    for _ in range(frames(3.0)):
      outs.append(law.update(3.5, v_ego, NO_LEAD, True, launch=True))
      if v_ego >= LANDING_SPEED:
        beyond.append(outs[-1])
      v_ego += outs[-1] * DT_MDL
    assert beyond[0] < 3.5 and outs[-1] == 3.5
    assert max(np.diff(outs)) <= max(RELEASE_JERK[1]) * DT_MDL + 1e-9

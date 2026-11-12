import math
import subprocess
import sys

import openpilot.cereal.messaging as messaging
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib import conditional_experimental_mode as cem_module
from openpilot.selfdrive.controls.lib.conditional_experimental_mode import (ConditionalExperimentalMode, DRIVER_OVERRIDE_SUPPRESS_S,
                                                                            GO_ACCEL, GO_CONFIRM_S, GO_ROLL_SPEED, LEAD_BEYOND_STOP_MARGIN,
                                                                            LEAD_BEYOND_STOP_MEMORY_S, LEAD_RELEASE_HYSTERESIS_S,
                                                                            MODE_MIN_LATCH_S, POST_STOP_SUPPRESS_S, STANDSTILL_MIN_LATCH_S,
                                                                            STOP_INTENT_HOLD_S, STOP_RELEASE_HYSTERESIS_S, CLEAR_CANCEL_S)
from openpilot.selfdrive.controls.lib.stop_helpers import (MODEL_INVALID_RELEASE_S, STOP_EARLY_HINT_DESIRED_ACCEL_MAX, STOP_EARLY_MIN_SPEED,
                                                           StopObservation)
from openpilot.selfdrive.modeld.constants import ModelConstants

N = ModelConstants.IDX_N
TICKS_PER_FRAME = round(DT_MDL / DT_CTRL)


def model(path_end=90.0, terminal_speed=10.0, should_stop=False, desired_accel=0.0, leads=()):
  md = messaging.new_message('modelV2').modelV2
  md.position.x = [path_end * i / (N - 1) for i in range(N)]
  md.position.y = [0.0] * N
  md.velocity.x = [10.0 + (terminal_speed - 10.0) * i / (N - 1) for i in range(N)]
  md.orientation.z = [0.0] * N
  md.action.shouldStop = should_stop
  md.action.desiredAcceleration = desired_accel
  md.init('leadsV3', 3)
  for lead, (prob, x, y) in zip(md.leadsV3, leads or ((0.0, 200.0, 0.0),) * 3, strict=True):
    lead.prob = prob
    lead.x = [x] * len(ModelConstants.LEAD_T_IDXS)
    lead.y = [y] * len(ModelConstants.LEAD_T_IDXS)
  return md


def stop_model(v_ego=10.0):
  return model(path_end=v_ego * 3.0, terminal_speed=0.2, desired_accel=-0.3)


def car_state(v_ego=10.0, standstill=False, gas=False, brake=False, blinker=False, steering_angle=0.0):
  cs = messaging.new_message('carState').carState
  cs.vEgo = v_ego
  cs.standstill = standstill
  cs.gasPressed = gas
  cs.brakePressed = brake
  cs.leftBlinker = blinker
  cs.steeringAngleDeg = steering_angle
  return cs


def radar_state(d_one=None, d_two=None, v_one=0.0, v_two=0.0):
  rs = messaging.new_message('radarState').radarState
  for lead, d, v in ((rs.leadOne, d_one, v_one), (rs.leadTwo, d_two, v_two)):
    if d is not None:
      lead.present = True
      lead.dRel = d
      lead.vLead = v
  return rs


def run(cem, seconds, md=None, cs=None, rs=None, *, enabled=True, model_valid=True, radar_valid=True, model_updates=True):
  # real cadence: 100 Hz control ticks, a new model frame every fifth tick
  md = md if md is not None else model()
  cs = cs if cs is not None else car_state()
  rs = rs if rs is not None else radar_state()
  out = None
  for tick in range(round(seconds / DT_CTRL)):
    out = cem.update(md, cs, rs, controls_enabled=enabled, model_updated=model_updates and tick % TICKS_PER_FRAME == 0,
                     model_valid=model_valid, radar_valid=radar_valid)
  return out


class TestEntry:
  def test_ordinary_driving_stays_chill(self):
    assert not run(ConditionalExperimentalMode(), 5.0)

  def test_a_confirmed_stop_enters_after_filter_and_debounce(self):
    cem = ConditionalExperimentalMode()
    assert not run(cem, 0.2, stop_model())
    assert run(cem, 0.5, stop_model())

  def test_a_brief_prediction_does_not_enter(self):
    cem = ConditionalExperimentalMode()
    assert not run(cem, 0.1, stop_model())
    assert not run(cem, 1.0)

  def test_a_far_lead_in_view_does_not_block_entry(self):
    assert run(ConditionalExperimentalMode(), 1.0, stop_model(), rs=radar_state(d_one=120.0))
    assert run(ConditionalExperimentalMode(), 1.0, stop_model(), rs=radar_state(d_two=300.0))

  def test_a_relevant_lead_blocks_entry_through_a_short_dropout(self):
    # a direct shouldStop cannot mint the early release, only a strict stop with a clear corridor can
    cem = ConditionalExperimentalMode()
    direct = model(should_stop=True, desired_accel=-0.3)
    assert not run(cem, 1.0, direct, rs=radar_state(d_one=20.0))
    # the lead gone for less than the guard: its evidence is still zeroed, and the lead coming back re-arms the guard
    assert not run(cem, LEAD_RELEASE_HYSTERESIS_S - DT_MDL, direct)
    assert cem.intent_filter.x == 0.0
    assert not run(cem, 0.5, direct, rs=radar_state(d_one=20.0))
    assert not run(cem, LEAD_RELEASE_HYSTERESIS_S - DT_MDL, direct)
    assert run(cem, 1.5, direct)

  def test_a_strict_stop_with_a_clear_corridor_releases_the_lead_guard_early(self):
    cem = ConditionalExperimentalMode()
    run(cem, 1.0, stop_model(), rs=radar_state(d_one=20.0))
    # inside the guard the release already lets the evidence through
    assert not run(cem, LEAD_RELEASE_HYSTERESIS_S - DT_MDL, stop_model())
    assert cem._lead_release_active and cem.intent_filter.x > 0.0
    assert run(cem, 1.0, stop_model())

  def test_an_in_corridor_hypothesis_keeps_the_lead_guard(self):
    cem = ConditionalExperimentalMode()
    run(cem, 1.0, stop_model(), rs=radar_state(d_one=20.0))
    replacement = model(path_end=30.0, terminal_speed=0.2, desired_accel=-0.3, leads=((0.9, 200.0, 0.0), (0.2, 27.0, 0.3), (0.0, 20.0, 0.0)))
    assert not run(cem, LEAD_RELEASE_HYSTERESIS_S - DT_MDL, replacement)
    assert not cem._lead_release_active and cem.intent_filter.x == 0.0

  def test_a_raw_lead_on_a_control_tick_revokes_a_pending_release_only(self):
    cem = ConditionalExperimentalMode()
    run(cem, 1.0, stop_model(), rs=radar_state(d_one=20.0))
    run(cem, 0.1, stop_model())
    assert cem._lead_release_active
    cem.update(stop_model(), car_state(), radar_state(d_one=200.0), controls_enabled=True, model_updated=False, model_valid=True)
    assert not cem._lead_release_active
    assert cem.intent_filter.x > 0.0

  def test_a_committed_turn_does_not_trigger(self):
    assert not run(ConditionalExperimentalMode(), 2.0, stop_model(5.0), car_state(5.0, blinker=True, steering_angle=45.0))
    assert run(ConditionalExperimentalMode(), 2.0, stop_model(), car_state(blinker=True, steering_angle=5.0))


class TestExitAndHold:
  def entered(self):
    cem = ConditionalExperimentalMode()
    assert run(cem, 1.0, stop_model())
    return cem

  def test_a_confirmed_stop_holds_through_prediction_flicker(self):
    # the flicker is evidence dropping out, not the road opening: a short, slow path (an open road now cancels the hold)
    cem = self.entered()
    for _ in range(3):
      assert run(cem, 0.3, model(path_end=15.0, terminal_speed=2.0))
      assert run(cem, 0.3, stop_model())
    assert run(cem, STOP_INTENT_HOLD_S - 0.5, model(path_end=15.0, terminal_speed=2.0))

  def test_stable_clear_returns_to_chill(self):
    cem = self.entered()
    assert not run(cem, STOP_INTENT_HOLD_S + 1.5)

  def test_standstill_latch_then_resume_releases_and_suppresses_reentry(self):
    cem = self.entered()
    assert run(cem, 2.0, stop_model(0.0), car_state(0.0, standstill=True))
    assert not run(cem, 0.05, model(), car_state(1.0))
    assert not run(cem, POST_STOP_SUPPRESS_S - 0.5, stop_model())
    assert run(cem, 1.5, stop_model())

  def test_pedals_return_to_chill_and_suppress_reentry(self):
    cem = self.entered()
    assert not run(cem, 0.05, stop_model(), car_state(gas=True))
    assert not run(cem, DRIVER_OVERRIDE_SUPPRESS_S - 0.5, stop_model())
    assert run(cem, 1.5, stop_model())
    cem = self.entered()
    assert not run(cem, 0.05, stop_model(), car_state(brake=True))

  def test_a_hung_model_releases_the_mode_within_the_invalid_window(self):
    cem = self.entered()
    assert run(cem, MODEL_INVALID_RELEASE_S - 0.1, stop_model(), model_valid=False, model_updates=False)
    assert not run(cem, 0.2, stop_model(), model_valid=False, model_updates=False)

  def test_a_nonfinite_trajectory_counts_as_an_invalid_model(self):
    cem = self.entered()
    broken = stop_model()
    broken.velocity.x = [float('nan')] * N
    assert run(cem, MODEL_INVALID_RELEASE_S - 0.1, broken)
    assert not run(cem, 0.2, broken)

  def test_a_nonfinite_trajectory_still_releases_the_mode_while_the_radar_is_down(self):
    # the release is the model's own: a radar dropout used to certify the garbage trajectory as complete and the
    # 0.5 s window never opened, so a double fault held the mode until the 4 s intent hold ran out
    cem = self.entered()
    broken = stop_model()
    broken.velocity.x = [float('nan')] * N
    assert run(cem, MODEL_INVALID_RELEASE_S - 0.1, broken, radar_valid=False)
    assert not run(cem, 0.2, broken, radar_valid=False)

  def test_a_radar_dropout_alone_is_not_an_invalid_model(self):
    # the radar only decides whether a frame carries stop evidence; a complete model frame keeps the mode past the
    # invalid window, and the ordinary clear hysteresis is what ends the search
    cem = self.entered()
    assert run(cem, MODEL_INVALID_RELEASE_S + 0.3, stop_model(), radar_valid=False)

  def test_disable_resets_to_chill(self):
    cem = self.entered()
    assert not run(cem, 0.05, stop_model(), enabled=False)
    assert not cem.experimental_mode and cem.intent_filter.x == 0.0


def hint_model(v_ego=15.0):
  # borderline evidence: the hint tier alone, confidence exactly 0.70, the plan keeping over a third of the speed
  # (route 0x2b's sustaining frames)
  return model(path_end=v_ego * 6.0, terminal_speed=6.5, desired_accel=-0.3)


def early_model(v_ego=15.0):
  # committed evidence: the early tier, confidence 0.80
  return model(path_end=v_ego * 6.0, terminal_speed=5.0, desired_accel=-0.6)


def murky_model():
  # neither stop evidence nor an open road: a short path that keeps over a third of the speed
  return model(path_end=15.0, terminal_speed=6.5)


class TestSearchRelease:
  # route 0x2b t=1291-1317: the mode engaged for a real red light, the car passed it, and lone 0.70-confidence hint
  # frames kept refreshing the 4 s hold for 17.5 s while the model's own path showed the road open twice

  def enter_on_hints(self):
    cem = ConditionalExperimentalMode()
    assert run(cem, 1.0, hint_model(), car_state(15.0))
    return cem

  def test_a_lone_hint_cannot_sustain_the_search(self):
    cem = self.enter_on_hints()
    # after entry: murky frames with a single hint frame every 1.5 s -- the old behavior stayed latched indefinitely
    for _ in range(4):
      run(cem, 1.5, murky_model(), car_state(15.0))
      run(cem, DT_MDL, hint_model(), car_state(15.0))
    assert not cem.experimental_mode

  def test_committed_evidence_still_sustains(self):
    cem = self.enter_on_hints()
    for _ in range(4):
      run(cem, 1.5, murky_model(), car_state(15.0))
      run(cem, DT_MDL, early_model(), car_state(15.0))
    assert cem.experimental_mode

  def test_an_open_road_cancels_the_hold(self):
    cem = ConditionalExperimentalMode()
    assert run(cem, 1.0, stop_model(), car_state(10.0))
    # the road opens (a long, moving path): the hold is canceled and the ordinary clear hysteresis exits well
    # before the 4 s the last strict frame would have held
    assert not run(cem, CLEAR_CANCEL_S + 1.0, model(path_end=150.0, terminal_speed=12.0), car_state(10.0))

  def test_a_short_open_flash_does_not_cancel(self):
    cem = ConditionalExperimentalMode()
    assert run(cem, 1.0, stop_model(), car_state(10.0))
    run(cem, CLEAR_CANCEL_S * 0.6, model(path_end=150.0, terminal_speed=12.0), car_state(10.0))
    assert run(cem, DT_MDL, stop_model(), car_state(10.0))


def dip_model(v_ego, low=1.0, terminal_speed=6.0, path_end=40.0, desired_accel=-1.5):
  # the model anticipating a green: the plan sheds the speed toward the line, then relaunches before its end
  md = model(path_end=path_end, terminal_speed=terminal_speed, desired_accel=desired_accel)
  mid = (N - 1) // 2
  md.velocity.x = [v_ego + (low - v_ego) * i / mid if i <= mid else low + (terminal_speed - low) * (i - mid) / (N - 1 - mid) for i in range(N)]
  return md


def held_stop_model():
  # the model holding the car at a red: a stub path and its stop bit
  return model(path_end=3.0, terminal_speed=0.2, should_stop=True)


def launch_model(v_ego=0.0, terminal_speed=8.0, path_end=40.0, desired_accel=0.3):
  # an open road, the plan climbing from the car's own speed
  md = model(path_end=path_end, terminal_speed=terminal_speed, desired_accel=desired_accel)
  md.velocity.x = [v_ego + (terminal_speed - v_ego) * i / (N - 1) for i in range(N)]
  return md


class TestSlowingAndGo:
  def entered(self):
    cem = ConditionalExperimentalMode()
    assert run(cem, 1.0, stop_model())
    return cem

  def test_an_anticipated_green_while_moving_keeps_the_mode(self):
    # the plan's end reads as an open road, its low point does not; entry still needs stop evidence, which below 13 m/s a slowing plan
    # on a straight approach is only with the model braking at least 0.25 m/s^2 (D47)
    assert not run(ConditionalExperimentalMode(), 2.0, dip_model(8.0, desired_accel=-0.2), car_state(8.0))
    assert run(ConditionalExperimentalMode(), 2.0, dip_model(8.0), car_state(8.0))
    cem = ConditionalExperimentalMode()
    assert run(cem, 1.0, stop_model(8.0), car_state(8.0))
    assert run(cem, STOP_INTENT_HOLD_S + 1.5, dip_model(8.0), car_state(8.0))

  def test_the_plan_still_shedding_speed_keeps_the_mode_below_the_early_tier(self):
    # entered on the early tier; below its speed, with a blinker on, no tier fires while the plan still ends far below the car
    cem = ConditionalExperimentalMode()
    assert run(cem, 1.0, early_model(15.0), car_state(15.0))
    shedding = model(path_end=85.0, terminal_speed=3.5, desired_accel=-0.7)
    assert run(cem, STOP_INTENT_HOLD_S + 1.5, shedding, car_state(12.5, blinker=True))

  def test_a_green_at_rest_exits_when_the_model_goes(self):
    cem = self.entered()
    rest = car_state(0.0, standstill=True)
    assert run(cem, STANDSTILL_MIN_LATCH_S + 0.5, stop_model(0.0), rest)
    assert run(cem, GO_CONFIRM_S - 0.1, launch_model(), rest)
    assert not run(cem, 0.1, launch_model(), rest)

  def test_a_stop_frame_restarts_the_go_count(self):
    cem = self.entered()
    rest = car_state(0.0, standstill=True)
    assert run(cem, STANDSTILL_MIN_LATCH_S + 0.5, stop_model(0.0), rest)
    # the model calls the stop again for one frame between two short open-road runs: neither run is a go
    assert run(cem, GO_CONFIRM_S - 0.1, launch_model(), rest)
    assert run(cem, DT_MDL, model(path_end=3.0, terminal_speed=0.2, should_stop=True), rest)
    assert run(cem, GO_CONFIRM_S - 0.1, launch_model(), rest)
    assert not run(cem, 0.1, launch_model(), rest)

  def test_an_open_road_flash_at_rest_keeps_the_mode(self):
    cem = self.entered()
    rest = car_state(0.0, standstill=True)
    assert run(cem, STANDSTILL_MIN_LATCH_S + 0.5, stop_model(0.0), rest)
    # the longest open-road flash the model has shown in the middle of a red
    assert run(cem, 0.35, launch_model(), rest)
    assert run(cem, 1.0, held_stop_model(), rest)

  def test_the_model_calling_the_stop_again_takes_back_an_exit_at_rest(self):
    cem = self.entered()
    rest = car_state(0.0, standstill=True)
    assert run(cem, STANDSTILL_MIN_LATCH_S + 0.5, stop_model(0.0), rest)
    assert not run(cem, GO_CONFIRM_S + 0.1, launch_model(), rest)
    # well inside the re-entry suppression: the car has not moved, so the stop is the model's to take back
    assert run(cem, POST_STOP_SUPPRESS_S / 2, held_stop_model(), rest)

  def test_an_exit_is_final_once_the_car_moves_off(self):
    cem = self.entered()
    rest = car_state(0.0, standstill=True)
    assert run(cem, STANDSTILL_MIN_LATCH_S + 0.5, stop_model(0.0), rest)
    assert not run(cem, GO_CONFIRM_S + 0.1, launch_model(), rest)
    assert not run(cem, 0.2, launch_model(1.0), car_state(1.0))
    assert not run(cem, POST_STOP_SUPPRESS_S - 1.0, held_stop_model(), car_state(1.0))

  def test_a_green_at_rest_waits_for_the_standstill_latch(self):
    cem = self.entered()
    rest = car_state(0.0, standstill=True)
    run(cem, 0.4, stop_model(0.0), rest)
    # the go is confirmed long before the car has stood STANDSTILL_MIN_LATCH_S
    assert run(cem, STANDSTILL_MIN_LATCH_S - 0.5, launch_model(), rest)
    assert not run(cem, 0.15, launch_model(), rest)

  def test_strong_acceleration_while_moving_exits(self):
    cem = self.entered()
    run(cem, MODE_MIN_LATCH_S, stop_model())
    cs = car_state(6.0)
    # an open road with a mild request is not a go: the clear hysteresis decides that one
    assert run(cem, GO_CONFIRM_S + 0.2, launch_model(6.0, 12.0, 80.0, GO_ACCEL - 0.3), cs)
    assert not run(cem, GO_CONFIRM_S, launch_model(6.0, 12.0, 80.0, GO_ACCEL + 0.3), cs)

  def test_a_launch_request_while_the_plan_still_slows_is_not_a_go(self):
    # the end reads as an open road and the model asks for acceleration now, but the plan still sheds the speed further on
    cem = ConditionalExperimentalMode()
    assert run(cem, 1.0, stop_model(8.0), car_state(8.0))
    assert run(cem, STOP_INTENT_HOLD_S + 1.5, dip_model(8.0, desired_accel=GO_ACCEL + 0.3), car_state(8.0))

  def test_the_go_exit_keeps_the_mode_latch_and_suppresses_reentry(self):
    cem = ConditionalExperimentalMode()
    assert run(cem, 0.6, stop_model())
    remaining = MODE_MIN_LATCH_S - cem._active_elapsed
    assert remaining - 0.1 > GO_CONFIRM_S
    go = launch_model(6.0, 12.0, 80.0, GO_ACCEL + 0.3)
    assert run(cem, remaining - 0.1, go, car_state(6.0))
    assert not run(cem, 0.15, go, car_state(6.0))
    assert not run(cem, POST_STOP_SUPPRESS_S - 0.5, stop_model())
    assert run(cem, 1.5, stop_model())


def through_lead():
  # a lead going through the stop: 25 m ahead at 10 m/s
  return radar_state(d_one=25.0, v_one=10.0)


class TestLeadOwnership:
  # route 0x158 t=5266-5286: the model stopped for a yellow light while the SUV ahead braked, then went through it; the
  # relevant-lead veto on every frame kept the mode in Chill until the driver braked. Since D46 only a queue owns the stop

  def test_a_lead_going_through_a_near_stop_does_not_block_entry(self):
    assert run(ConditionalExperimentalMode(), 1.0, stop_model(), rs=through_lead())
    assert run(ConditionalExperimentalMode(), 1.0, model(should_stop=True, desired_accel=-0.3), rs=radar_state(d_one=30.0, v_one=25.0))

  def test_a_moving_lead_that_would_rest_short_of_the_stop_does_not_block_entry(self):
    # D42 read this lead's rest point (25 + 5^2 / 5 - 7 = 23 m) short of the 30 m stop and blocked; a moving lead is no queue
    assert run(ConditionalExperimentalMode(), 1.0, stop_model(), rs=radar_state(d_one=25.0, v_one=5.0))

  def test_the_models_likely_lead_ending_short_of_the_stop_does_not_block_entry(self):
    md = model(path_end=30.0, terminal_speed=0.2, desired_accel=-0.3, leads=((0.8, 35.0, 0.0), (0.0, 200.0, 0.0), (0.0, 200.0, 0.0)))
    assert run(ConditionalExperimentalMode(), 1.0, md, rs=through_lead())

  def test_a_queue_past_the_stop_point_blocks_entry(self):
    # the model stops a few meters farther back from a stopped car than the MPC does, so the car's rest point lies past
    # the model's; it is the queue the model stops for, and its creep and launch are the lead logic's
    md, cs = model(path_end=60.0, terminal_speed=0.2, desired_accel=-0.3), car_state(15.0)
    for v_lead in (-0.5, 0.0, 0.9):
      assert not run(ConditionalExperimentalMode(), 2.0, md, cs, rs=radar_state(d_one=68.0, v_one=v_lead))
    assert run(ConditionalExperimentalMode(), 2.0, md, cs, rs=radar_state(d_one=68.0, v_one=14.0))

  def test_a_tracked_lead_no_longer_owning_the_stop_ends_the_veto_at_once(self):
    cem = ConditionalExperimentalMode()
    assert not run(cem, 1.0, stop_model(), rs=radar_state(d_one=25.0, v_one=0.5))
    # the queue moves off through the stop: positive evidence, so no grace runs (a lead the radar loses keeps it, see TestEntry)
    assert run(cem, 1.0, stop_model(), rs=through_lead())

  def test_a_far_stop_keeps_the_lead_veto(self):
    # the early and hint tiers read a stop far ahead; a lead there going through it is no evidence the stop is the model's
    # (route 0xa2 t=459.8: a lead 107 m ahead slowing, a hint-tier stop at 147 m, then e2e out-braking Chill with no stop)
    lead = radar_state(d_one=60.0, v_one=15.0)
    assert not run(ConditionalExperimentalMode(), 2.0, early_model(), car_state(15.0), rs=lead)
    assert not run(ConditionalExperimentalMode(), 2.0, hint_model(), car_state(15.0), rs=lead)
    assert run(ConditionalExperimentalMode(), 2.0, early_model(), car_state(15.0))

  def test_at_a_crawl_a_relevant_lead_always_blocks_entry(self):
    # route 0x158 t=5333.2: at rest behind a creeping queue; the queue's creep and launch are the lead logic's
    creeping = radar_state(d_one=9.0, v_one=1.2)
    stub = model(path_end=1.5, terminal_speed=0.2, should_stop=True)
    assert not run(ConditionalExperimentalMode(), 2.0, stub, car_state(0.0, standstill=True), rs=creeping)
    assert not run(ConditionalExperimentalMode(), 2.0, stub, car_state(GO_ROLL_SPEED - 0.5), rs=creeping)
    assert run(ConditionalExperimentalMode(), 2.0, stub, car_state(GO_ROLL_SPEED + 0.5), rs=creeping)


YELLOW_LIGHT_SPEED = 13.5


def yellow_light_stop(frame, should_stop=False, leads=()):
  # route 0x1f1 t=544.9: the model's strict stop point moving between 47 and 57 m from frame to frame
  return model(path_end=47.0 if frame % 2 else 57.0, terminal_speed=0.2, should_stop=should_stop, desired_accel=-1.3, leads=leads)


def yellow_light_lead(v_lead=YELLOW_LIGHT_SPEED):
  # the car ahead, 20 m out and holding 30 mph through the yellow. Its rest point by the MPC's geometry, 20 + 13.5^2 / 5 - 7 =
  # 49.5 m, lay between the model's stop points, so D42's ownership flickered and every flicker re-armed the 3 s guard
  return radar_state(d_one=20.0, v_one=v_lead)


def run_frames(cem, frames, cs, rs):
  # one model message per model frame, held for its control ticks; the control tick of entry, counted from the first frame, or None
  for i, md in enumerate(frames):
    for k in range(TICKS_PER_FRAME):
      if cem.update(md, cs, rs, controls_enabled=True, model_updated=k == 0, model_valid=True):
        return i * TICKS_PER_FRAME + k
  return None


class TestYellowLight:
  # route 0x1f1 t=544.9-547.1 (2026-11-09): the model planned the stop for a yellow light the car ahead went through; the mode never
  # entered and the driver braked with 22 m left. A moving lead no longer blocks a near stop (D46)

  def entry_tick(self, rs, frames=None, cs=None):
    cem = ConditionalExperimentalMode()
    cs = cs if cs is not None else car_state(YELLOW_LIGHT_SPEED)
    # a second of cruising behind the lead first, so its guard is running when the stop appears, as it was on the road
    run(cem, 1.0, model(path_end=140.0, terminal_speed=YELLOW_LIGHT_SPEED), cs, rs)
    assert not cem.experimental_mode
    return run_frames(cem, frames if frames is not None else [yellow_light_stop(i) for i in range(40)], cs, rs)

  def test_a_lead_going_through_a_flickering_stop_costs_no_entry_time(self):
    no_lead = self.entry_tick(radar_state())
    # the entry filter and the debounce: the tenth strict frame
    assert no_lead is not None and no_lead * DT_CTRL < 0.5
    assert self.entry_tick(yellow_light_lead()) == no_lead

  def test_the_models_lead_forecast_ending_short_of_the_stop_costs_no_entry_time(self):
    forecast = ((0.9, 25.0, 0.0), (0.0, 200.0, 0.0), (0.0, 200.0, 0.0))
    frames = [yellow_light_stop(i, leads=forecast) for i in range(40)]
    assert self.entry_tick(yellow_light_lead(), frames) == self.entry_tick(radar_state(), frames) is not None

  def test_a_queue_still_blocks(self):
    for v_lead in (-0.3, 0.0, 0.9):
      assert self.entry_tick(yellow_light_lead(v_lead)) is None

  def test_the_car_below_go_roll_speed_still_blocks(self):
    # at a crawl only the model's stop bit makes a near stop; a lead pulling away still keeps the creep and launch with the lead logic
    frames, crawl = [yellow_light_stop(i, should_stop=True) for i in range(40)], car_state(GO_ROLL_SPEED - 0.5)
    assert self.entry_tick(yellow_light_lead(2.0), frames, crawl) is None
    assert self.entry_tick(radar_state(), frames, crawl) is not None

  def test_early_and_hint_tiers_behind_the_lead_still_block(self):
    cs = car_state(15.0)
    for md in (early_model(), hint_model()):
      assert self.entry_tick(yellow_light_lead(), [md] * 40, cs) is None
      assert self.entry_tick(radar_state(), [md] * 40, cs) is not None

  def test_a_lead_whose_speed_or_place_cannot_be_read_blocks(self):
    nan = float('nan')
    assert self.entry_tick(yellow_light_lead(nan)) is None
    assert self.entry_tick(radar_state(d_one=20.0, v_one=YELLOW_LIGHT_SPEED, d_two=nan, v_two=YELLOW_LIGHT_SPEED)) is None


LOST_LEAD_SPEED = 19.7


class TestLostLead:
  # route 0x207 t=346.1-350.4 (2026-11-11): 44 mph toward a red light after a bend, the lead gone through it and lost by the radar at
  # t=346.9; the model planned the stop on the early tier from t=347.7, but the 3 s guard the lost lead left zeroed that evidence, the
  # corridor release needs a strict frame, and the driver braked at t=350.4 (0x201 t=115.7-120.3 the same). The guard is 0.5 s (D48)

  def entry_time(self, rs_before, frames=60):
    # a second behind a moving lead on an ordinary plan (each frame re-arms the guard), then the lead is gone and the stop is planned
    cem, cs = ConditionalExperimentalMode(), car_state(LOST_LEAD_SPEED)
    run(cem, 1.0, model(path_end=150.0, terminal_speed=LOST_LEAD_SPEED), cs, rs_before)
    assert not cem.experimental_mode
    tick = run_frames(cem, [early_model(LOST_LEAD_SPEED)] * frames, cs, radar_state())
    return None if tick is None else tick * DT_CTRL

  def test_a_stop_planned_as_the_lead_is_lost_enters_after_the_short_guard(self):
    no_lead = self.entry_time(radar_state())
    lost = self.entry_time(radar_state(d_one=40.0, v_one=LOST_LEAD_SPEED))
    assert no_lead is not None and lost is not None
    # the guard still runs, and only the guard: entry comes its length after the time it takes with no lead
    assert math.isclose(lost - no_lead, LEAD_RELEASE_HYSTERESIS_S, abs_tol=DT_MDL + 1e-9)
    assert lost < 1.5

  def test_the_lead_coming_back_inside_the_guard_still_blocks(self):
    cem, cs = ConditionalExperimentalMode(), car_state(LOST_LEAD_SPEED)
    lead = radar_state(d_one=40.0, v_one=LOST_LEAD_SPEED)
    for _ in range(4):
      assert not run(cem, 0.5, early_model(LOST_LEAD_SPEED), cs, lead)
      assert not run(cem, LEAD_RELEASE_HYSTERESIS_S - DT_MDL, early_model(LOST_LEAD_SPEED), cs)


JUNCTION_SPEED = 14.3
JUNCTION_STOP = 48.0


def junction_frames(n, pattern=(0, 0, 1)):
  # route 0x1fc t=1451.9-1455.8: the plan alternating between a strict stop 42-55 m out and a longer, slowing plan 79-103 m out
  # (the hint tier), the longer one putting the relevant-lead window past a lead 80-86 m ahead
  return [model(path_end=95.0, terminal_speed=3.0, desired_accel=-0.5) if pattern[i % len(pattern)] else
          model(path_end=JUNCTION_STOP, terminal_speed=0.2, desired_accel=-1.0) for i in range(n)]


def through_the_junction(d_two=None, v_two=14.0):
  # the car ahead went through the yellow: 83 m out at 14 m/s, past the model's stop point
  return radar_state(d_one=83.0, v_one=14.0, d_two=d_two, v_two=v_two)


class TestLeadBeyondTheStop:
  # route 0x1fc t=1451.9-1455.8 (2026-11-11): 32 mph toward a light turning yellow, the lead 80-86 m out already past the junction. Every
  # longer plan made that lead relevant and re-armed the guard, the mode never entered and the car crossed the line at the end of the
  # yellow. A moving lead past the model's nearest stop point of the last second is no relevant lead for the veto (D49)

  def approach(self, rs, frames=None, cs=None):
    # a second behind the lead on an ordinary plan (its guard running), then the light: the control tick of entry and the guard per frame
    cem, cs = ConditionalExperimentalMode(), cs if cs is not None else car_state(JUNCTION_SPEED)
    run(cem, 1.0, model(path_end=140.0, terminal_speed=JUNCTION_SPEED), cs, rs)
    assert not cem.experimental_mode
    entry, guard = None, []
    for i, md in enumerate(frames if frames is not None else junction_frames(60)):
      for k in range(TICKS_PER_FRAME):
        if cem.update(md, cs, rs, controls_enabled=True, model_updated=k == 0, model_valid=True) and entry is None:
          entry = i * TICKS_PER_FRAME + k
      guard.append(cem._lead_veto_remaining)
    return entry, guard

  def test_a_lead_through_the_junction_no_longer_re_arms_the_guard(self):
    entry, guard = self.approach(through_the_junction())
    assert all(b <= a for a, b in zip(guard, guard[1:], strict=False))
    # the guard the lead left before the light runs out, then the filter and the debounce
    assert entry is not None and entry * DT_CTRL < LEAD_RELEASE_HYSTERESIS_S + 1.0

  def test_the_rule_switched_off_keeps_the_guard_armed(self, monkeypatch):
    monkeypatch.setattr(cem_module, 'LEAD_BEYOND_STOP_MEMORY_S', 0.0)
    entry, guard = self.approach(through_the_junction())
    assert entry is None and max(guard) == LEAD_RELEASE_HYSTERESIS_S

  def test_a_lead_nearer_than_beyond_the_stop_point_vetoes_as_before(self, monkeypatch):
    # a moving lead short of the stop, a car that will stop short of the line, a queue past the model's stop point (D42), a far queue,
    # a nearer lead in the second slot, and a lead whose speed or place cannot be read: the same frames with the rule on and off
    nan = float('nan')
    cases = (radar_state(d_one=40.0, v_one=14.0), radar_state(d_one=JUNCTION_STOP - 5.0, v_one=4.0),
             radar_state(d_one=JUNCTION_STOP + 12.0, v_one=0.0), radar_state(d_one=83.0, v_one=0.5), through_the_junction(d_two=45.0),
             through_the_junction(d_two=45.0, v_two=0.0), radar_state(d_one=83.0, v_one=nan), through_the_junction(d_two=nan),
             radar_state(d_one=JUNCTION_STOP + LEAD_BEYOND_STOP_MARGIN, v_one=14.0))
    on = [self.approach(rs) for rs in cases]
    monkeypatch.setattr(cem_module, 'LEAD_BEYOND_STOP_MEMORY_S', 0.0)
    assert on == [self.approach(rs) for rs in cases]
    assert all(entry is None for entry, _ in on[:4])

  def test_at_a_crawl_the_lead_keeps_its_veto(self):
    stub = [model(path_end=1.5, terminal_speed=0.2, should_stop=True)] * 40
    for v_ego in (0.0, GO_ROLL_SPEED - 0.5):
      assert self.approach(radar_state(d_one=9.0, v_one=1.2), stub, car_state(v_ego, standstill=v_ego == 0.0))[0] is None

  def test_the_stop_point_is_remembered_for_a_second_only(self):
    cem, cs, rs = ConditionalExperimentalMode(), car_state(JUNCTION_SPEED), through_the_junction()
    run(cem, 0.5, model(path_end=JUNCTION_STOP, terminal_speed=0.2, desired_accel=-1.0), cs, rs)
    ordinary = model(path_end=140.0, terminal_speed=JUNCTION_SPEED)
    # the lead is in the window of the ordinary plan; while the strict stop point is remembered it is past it
    run(cem, LEAD_BEYOND_STOP_MEMORY_S - 2 * DT_MDL, ordinary, cs, rs)
    assert cem._lead_veto_remaining < LEAD_RELEASE_HYSTERESIS_S
    run(cem, 2 * DT_MDL, ordinary, cs, rs)
    assert cem._lead_veto_remaining == LEAD_RELEASE_HYSTERESIS_S

  def test_the_stop_point_is_forgotten_on_disengagement_reset_and_cleared_evidence(self):
    cs, rs, ordinary = car_state(JUNCTION_SPEED), through_the_junction(), model(path_end=140.0, terminal_speed=JUNCTION_SPEED)
    strict = model(path_end=JUNCTION_STOP, terminal_speed=0.2, desired_accel=-1.0)
    for forget in (lambda cem: run(cem, DT_CTRL, strict, cs, rs, enabled=False), lambda cem: cem.reset(),
                   lambda cem: run(cem, DT_CTRL, strict, car_state(JUNCTION_SPEED, brake=True), rs),
                   lambda cem: cem._clear_evidence()):
      cem = ConditionalExperimentalMode()
      run(cem, 0.3, strict, cs, rs)
      assert cem._stop_points
      forget(cem)
      assert cem._stop_points == []
      cem._override_remaining = 0.0
      run(cem, DT_MDL, ordinary, cs, rs)
      assert cem._lead_veto_remaining == LEAD_RELEASE_HYSTERESIS_S


LOW_SPEED_APPROACH = 10.0


def creeping_stop(frame=0, v_ego=LOW_SPEED_APPROACH, desired_accel=-0.5, low=None, end=None, path_end=None, curvature=0.0, heading=0.0):
  # route 0x1ef t=209.3-213.9: a red light with no car ahead. The plan sheds the speed to a low point of 0.3-0.9 m/s near its end and
  # creeps on to 1.1-2.5 m/s, so the strict tier (an end at or below 1.0 m/s) never reads it, and below 13 m/s no early tier can
  low = (0.3, 0.6, 0.9, 0.5)[frame % 4] if low is None else low
  end = (1.1, 1.8, 2.5, 1.4)[frame % 4] if end is None else end
  md = model(path_end=60.0 - 0.5 * frame if path_end is None else path_end, terminal_speed=end, desired_accel=desired_accel)
  dip = N - 5
  md.velocity.x = [v_ego + (low - v_ego) * i / dip if i <= dip else low + (end - low) * (i - dip) / (N - 1 - dip) for i in range(N)]
  md.action.desiredCurvature = curvature
  md.orientation.z = [heading * i / (N - 1) for i in range(N)]
  return md


def red_light_request(frame):
  # the model's request falling through the hint tier's -0.25 on the ninth frame (0x1ef: -0.25 at t=210.5, -0.53 at t=211.3)
  return -0.10 - 0.02 * frame


class TestLowSpeedApproach:
  # route 0x1ef t=207.7-213.9 (2026-11-09): engaged at 7.8 m/s toward a light red for 8 s, no car ahead. The plan slowed from t=209.3
  # and the model's request passed -0.25 at t=210.5, but no tier read the stop, Chill took the car to 10.8 m/s and the driver braked
  # with 21.6 m left. Below 13 m/s a slowing, straight, braking plan is now entry evidence at the hint tier's 0.70 (D47)

  def red_light(self, n=60, **kwargs):
    return [creeping_stop(i, desired_accel=red_light_request(i), **kwargs) for i in range(n)]

  def entry_tick(self, frames, cs=None, rs=None):
    return run_frames(ConditionalExperimentalMode(), frames, cs if cs is not None else car_state(LOW_SPEED_APPROACH),
                      rs if rs is not None else radar_state())

  def hint_entry_tick(self):
    # the hint tier's own 0.70 through the filter and the debounce, from a mode with no evidence yet
    return run_frames(ConditionalExperimentalMode(), [hint_model()] * 40, car_state(15.0), radar_state())

  def test_a_creeping_red_light_enters_like_the_hint_tier_once_the_model_brakes(self):
    first = next(i for i in range(60) if red_light_request(i) <= STOP_EARLY_HINT_DESIRED_ACCEL_MAX)
    tick = self.entry_tick(self.red_light())
    assert tick is not None and tick - first * TICKS_PER_FRAME == self.hint_entry_tick()
    assert (tick - first * TICKS_PER_FRAME) * DT_CTRL < 0.7
    # anywhere below the early tiers' speed
    for v_ego in (5.0, STOP_EARLY_MIN_SPEED - 0.1):
      assert self.entry_tick([creeping_stop(i, v_ego=v_ego) for i in range(40)], car_state(v_ego)) is not None

  def test_a_single_frame_or_a_run_shorter_than_the_debounce_does_not_enter(self):
    frames = [creeping_stop(i) for i in range(40)]
    needed = self.hint_entry_tick() // TICKS_PER_FRAME + 1
    assert self.entry_tick(frames[:1] + [model()] * 40) is None
    assert self.entry_tick(frames[:needed - 1] + [model()] * 40) is None
    assert self.entry_tick(frames[:needed] + [model()] * 40) is not None

  def test_no_entry_while_the_model_asks_for_less_braking_than_the_hint_tier(self):
    for request in (STOP_EARLY_HINT_DESIRED_ACCEL_MAX + 0.01, 0.0, float('nan')):
      assert self.entry_tick([creeping_stop(i, desired_accel=request) for i in range(40)]) is None

  def test_no_entry_off_a_straight_approach(self):
    frames = [creeping_stop(i) for i in range(40)]
    assert self.entry_tick(frames, car_state(LOW_SPEED_APPROACH, blinker=True)) is None
    # the early tier's own test: lateral acceleration above 1 m/s^2, or the plan turning more than 20 degrees
    assert self.entry_tick([creeping_stop(i, curvature=1.1 / LOW_SPEED_APPROACH ** 2) for i in range(40)]) is None
    assert self.entry_tick([creeping_stop(i, heading=math.radians(25.0)) for i in range(40)]) is None

  def test_no_entry_when_the_plan_does_not_slow_or_ends_behind_the_car(self):
    # the low point must sit at or below 35 % of the car's speed; a plan that keeps 4 m/s of 10 is a slowdown, not a stop
    assert self.entry_tick([creeping_stop(i, low=4.0, end=4.5) for i in range(40)]) is None
    assert self.entry_tick([creeping_stop(i, path_end=-1.0) for i in range(40)]) is None

  def test_at_or_above_the_early_tiers_speed_the_existing_tiers_alone_decide(self):
    # the same creeping plan ending beyond the early and hint windows: no tier reads it there, so no entry
    for v_ego in (STOP_EARLY_MIN_SPEED, 14.0):
      frames = [creeping_stop(i, v_ego=v_ego, path_end=120.0) for i in range(40)]
      assert self.entry_tick(frames, car_state(v_ego)) is None

  def test_a_relevant_lead_vetoes_it_as_it_does_the_early_and_hint_tiers(self):
    # behind a relevant lead only the strict and direct tiers are a near stop (D42, D46); this reading is neither
    for lead in (radar_state(d_one=25.0, v_one=8.0), radar_state(d_one=25.0, v_one=0.5), radar_state(d_two=40.0, v_two=6.0)):
      assert self.entry_tick(self.red_light(), rs=lead) is None

  def test_it_cannot_refresh_the_hold_of_an_active_mode(self, monkeypatch):
    # D26: while active only the 0.8 tiers refresh the 4 s hold. The reading is entry evidence only; what holds a mode it entered is
    # D41's slowing term, which every real frame carrying the reading also carries. A frame with the reading alone adds nothing
    cem = ConditionalExperimentalMode()
    assert run(cem, 1.0, stop_model())
    lone = StopObservation(path_end=40.0, complete=True, slowing_approach=True)
    monkeypatch.setattr('openpilot.selfdrive.controls.lib.conditional_experimental_mode.observe_model_stop', lambda *args: lone)
    assert not run(cem, STOP_INTENT_HOLD_S + STOP_RELEASE_HYSTERESIS_S + 0.5, model())
    cem = ConditionalExperimentalMode()
    monkeypatch.undo()
    assert run(cem, 1.0, stop_model())
    slowing = StopObservation(path_end=40.0, complete=True, slowing=True, slowing_approach=True)
    monkeypatch.setattr('openpilot.selfdrive.controls.lib.conditional_experimental_mode.observe_model_stop', lambda *args: slowing)
    assert run(cem, STOP_INTENT_HOLD_S + STOP_RELEASE_HYSTERESIS_S + 0.5, model())

  def test_the_plan_still_slowing_holds_a_mode_it_entered(self):
    cem, cs = ConditionalExperimentalMode(), car_state(LOW_SPEED_APPROACH)
    assert run_frames(cem, self.red_light(), cs, radar_state()) is not None
    # the model eases its request, the plan still sheds the speed: D41's term holds the mode past the 4 s hold
    assert run(cem, STOP_INTENT_HOLD_S + 1.5, creeping_stop(desired_accel=-0.1), cs)
    # no longer slowing and no other evidence: the hold runs out and the clear hysteresis ends the mode
    assert not run(cem, STOP_INTENT_HOLD_S + 1.5, murky_model(), cs)


class TestFootprint:
  def test_the_mode_does_not_load_the_long_mpc(self):
    # selfdrived runs the mode: it must not load casadi or the generated solver
    heavy = ('casadi', 'openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc')
    code = f"import sys; import openpilot.selfdrive.controls.lib.conditional_experimental_mode; print([m for m in {heavy} if m in sys.modules])"
    assert subprocess.check_output([sys.executable, '-c', code], text=True).strip() == '[]'

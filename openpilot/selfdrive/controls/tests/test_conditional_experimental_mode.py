import openpilot.cereal.messaging as messaging
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.conditional_experimental_mode import (ConditionalExperimentalMode, DRIVER_OVERRIDE_SUPPRESS_S,
                                                                            GO_ACCEL, GO_CONFIRM_S, LEAD_RELEASE_HYSTERESIS_S,
                                                                            MODE_MIN_LATCH_S, POST_STOP_SUPPRESS_S, STANDSTILL_MIN_LATCH_S,
                                                                            STOP_INTENT_HOLD_S, CLEAR_CANCEL_S)
from openpilot.selfdrive.controls.lib.stop_helpers import MODEL_INVALID_RELEASE_S
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


def radar_state(d_one=None, d_two=None):
  rs = messaging.new_message('radarState').radarState
  for lead, d in ((rs.leadOne, d_one), (rs.leadTwo, d_two)):
    if d is not None:
      lead.present = True
      lead.dRel = d
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
    assert not run(cem, LEAD_RELEASE_HYSTERESIS_S - 0.5, direct)
    assert run(cem, 1.5, direct)

  def test_a_strict_stop_with_a_clear_corridor_releases_the_lead_guard_early(self):
    cem = ConditionalExperimentalMode()
    run(cem, 1.0, stop_model(), rs=radar_state(d_one=20.0))
    assert run(cem, 1.0, stop_model())

  def test_an_in_corridor_hypothesis_keeps_the_lead_guard(self):
    cem = ConditionalExperimentalMode()
    run(cem, 1.0, stop_model(), rs=radar_state(d_one=20.0))
    replacement = model(path_end=30.0, terminal_speed=0.2, desired_accel=-0.3, leads=((0.9, 200.0, 0.0), (0.2, 27.0, 0.3), (0.0, 20.0, 0.0)))
    assert not run(cem, 1.0, replacement)

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
    # the plan's end reads as an open road, its low point does not; entry still needs stop evidence
    assert not run(ConditionalExperimentalMode(), 2.0, dip_model(8.0), car_state(8.0))
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

import openpilot.cereal.messaging as messaging
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.conditional_experimental_mode import (ConditionalExperimentalMode, DRIVER_OVERRIDE_SUPPRESS_S,
                                                                            LEAD_RELEASE_HYSTERESIS_S, POST_STOP_SUPPRESS_S,
                                                                            STOP_INTENT_HOLD_S)
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
    cem = self.entered()
    for _ in range(3):
      assert run(cem, 0.3)
      assert run(cem, 0.3, stop_model())
    assert run(cem, STOP_INTENT_HOLD_S - 0.5)

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

  def test_disable_resets_to_chill(self):
    cem = self.entered()
    assert not run(cem, 0.05, stop_model(), enabled=False)
    assert not cem.experimental_mode and cem.intent_filter.x == 0.0

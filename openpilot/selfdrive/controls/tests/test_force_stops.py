import math


import openpilot.cereal.messaging as messaging
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.force_stops import (A_STOP_ENVELOPE, CLEAR_WINDOW_S, DV_MAX, ForceStops, GAS_OVERRIDE_S,
                                                           LATCH_SETBACK, MPC_PROFILE_OFFSET, NO_CAP, PROFILE_HANDOVER_SPEED, PROFILE_JERK,
                                                           PROFILE_LANDING, PROFILE_MAX_DECEL, PROFILE_MIN_TIME, QUALIFY_S, REARM_S,
                                                           RELEASE_OPEN_FRAMES, RELEASE_OPEN_LENGTH)
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import STOP_DISTANCE
from openpilot.selfdrive.controls.lib.stop_helpers import MODEL_INVALID_RELEASE_S, StopObservation


def frames(seconds):
  return round(seconds / DT_MDL)


def obs(path_end=20.0, should_stop=True, braking=True, strict=False, early=False, lead=False, relevant=False, turn=False,
        release_open=False, moving=False, corridor_clear=True, lane_change=False):
  return StopObservation(1.0 if should_stop else 0.0, path_end, should_stop, strict, early, True, braking, moving, relevant, lead, turn,
                         release_open, corridor_clear, lane_change)


def car_state(v_ego=10.0, standstill=False, gas=False, brake=False):
  cs = messaging.new_message('carState').carState
  cs.vEgo = v_ego
  cs.standstill = standstill
  cs.gasPressed = gas
  cs.brakePressed = brake
  return cs


def run(fs, seconds, observation, cs, experimental=True, enabled=True, valid=True):
  result = None
  for _ in range(frames(seconds)):
    result = fs.update(observation, cs, experimental, enabled, valid)
  return result


def committed(fs=None, v_ego=10.0, path_end=20.0):
  fs = fs or ForceStops()
  result = run(fs, 1.5, obs(path_end=path_end), car_state(v_ego))
  assert fs.forcing
  return fs, result


def car_state_accel(v_ego, a_ego):
  cs = car_state(v_ego)
  cs.aEgo = a_ego
  return cs


def commit_on_strict_evidence(v_ego, path_end, a_ego=0.0):
  # a world-fixed endpoint under strict evidence commits through the qualify path, as a red light does
  fs = ForceStops()
  for i in range(frames(QUALIFY_S) + 2):
    result = fs.update(obs(path_end=path_end - v_ego * DT_MDL * i, should_stop=False, strict=True, braking=False),
                       car_state_accel(v_ego, a_ego), True, True, True)
  assert fs.forcing
  return fs, result


def holding(fs=None):
  fs, _ = committed(fs)
  result = run(fs, 0.1, obs(path_end=4.0), car_state(0.0, standstill=True))
  assert fs.holding and result.holding
  return fs, result


class TestEntry:
  def test_classic_latch_commits_the_model_endpoint_with_its_setback(self):
    fs, result = committed()
    assert abs((result.stop_x) - (20.0 - LATCH_SETBACK - 10.0 * DT_MDL * (frames(1.5) - frames(1.5) + 0))) <= 10.0 * DT_MDL * frames(1.5)
    assert result.v_cruise_cap == NO_CAP                            # no speed cap once committed (D19)

  def test_widened_window_needs_braking_evidence(self):
    slow_brake = ForceStops()
    run(slow_brake, 3.0, obs(path_end=38.0, should_stop=False, braking=False), car_state(10.0))
    assert not slow_brake.forcing
    braking = ForceStops()
    run(braking, 3.0, obs(path_end=32.0, should_stop=True, braking=True), car_state(10.0))
    assert braking.forcing

  def test_shaping_caps_the_approach_on_the_live_endpoint(self):
    fs = ForceStops()
    result = fs.update(obs(path_end=60.0, should_stop=False, early=True), car_state(15.0), True, True, True)
    assert not fs.forcing
    assert math.isclose(result.v_cruise_cap, max(math.sqrt(2.0 * A_STOP_ENVELOPE * (60.0 - MPC_PROFILE_OFFSET)), 15.0 - DV_MAX), rel_tol=1e-6, abs_tol=1e-9)
    assert result.stop_x is None

  def test_a_short_window_of_strict_world_fixed_evidence_commits_before_the_classic_window(self):
    fs = ForceStops()
    path_end = 60.0
    for i in range(frames(QUALIFY_S) + 2):
      fs.update(obs(path_end=path_end - 10.0 * DT_MDL * i, should_stop=False, strict=True, braking=False), car_state(10.0), True, True, True)
    assert fs.forcing
    drifting = ForceStops()
    for i in range(frames(QUALIFY_S) + 2):
      drifting.update(obs(path_end=60.0 + 8.0 * (i % 2), should_stop=False, strict=True, braking=False), car_state(10.0), True, True, True)
    assert not drifting.forcing

  def test_a_committed_turn_never_commits(self):
    fs = ForceStops()
    run(fs, 2.0, obs(turn=True), car_state(5.0))
    assert not fs.forcing

  def test_mode_gates_entry_only(self):
    fs = ForceStops()
    assert run(fs, 2.0, obs(), car_state(10.0), experimental=False).v_cruise_cap == NO_CAP and not fs.forcing
    fs, _ = committed()
    assert run(fs, 1.0, obs(), car_state(10.0), experimental=False).stop_x is not None and fs.forcing


class TestMovingReleases:
  def test_a_tracked_lead_hands_the_stop_to_the_lead_logic_and_the_hold_can_re_form(self):
    fs, _ = committed()
    # one radar frame is not a lead: the commitment survives it (route 24 lost a red-light commitment to a single frame)
    assert run(fs, DT_MDL, obs(lead=True), car_state(10.0)).stop_x is not None and fs.forcing
    blocked = ForceStops()
    for i in range(frames(QUALIFY_S) + 4):
      blocked.update(obs(path_end=60.0 - 10.0 * DT_MDL * i, should_stop=False, strict=True, braking=False, lead=(i % 3 == 0)),
                     car_state(10.0), True, True, True)
    assert not blocked.forcing                                   # a raw lead, even a flickering one, blocks a new commitment
    assert run(fs, 0.8, obs(lead=True), car_state(10.0)).stop_x is None and not fs.forcing
    assert run(fs, 3.0, obs(path_end=30.0, should_stop=False, braking=False), car_state(2.0)).stop_x is None
    assert run(fs, 0.05, obs(path_end=4.0), car_state(0.0, standstill=True)).holding

  def test_brake_releases_and_gas_suppresses_shaping_for_the_grace(self):
    fs, _ = committed()
    assert not run(fs, 0.05, obs(), car_state(10.0, brake=True)).holding and not fs.forcing
    fs, _ = committed()
    run(fs, 0.05, obs(), car_state(10.0, gas=True))
    assert not fs.forcing
    assert run(fs, GAS_OVERRIDE_S - 1.0, obs(path_end=60.0, should_stop=False, early=True), car_state(10.0)).v_cruise_cap == NO_CAP
    assert run(fs, 1.5, obs(path_end=60.0, should_stop=False, early=True), car_state(10.0)).v_cruise_cap < NO_CAP

  def test_model_invalid_is_debounced(self):
    fs, _ = committed()
    assert run(fs, MODEL_INVALID_RELEASE_S - 0.1, obs(), car_state(10.0), valid=False).stop_x is not None
    assert run(fs, 0.2, obs(), car_state(10.0), valid=False).stop_x is None and not fs.forcing

  def test_an_open_clear_path_releases_at_once_and_an_ambiguous_one_waits_out_the_position_hold(self):
    # D28: a long, evidence-free path is a green and releases within RELEASE_OPEN_FRAMES; a clear but SHORT path
    # (nothing to drive toward yet) still goes the slow way -- filtered detector decay plus the position hold
    fs, _ = committed()
    open_road = obs(path_end=90.0, should_stop=False, braking=False, moving=True)
    assert run(fs, (RELEASE_OPEN_FRAMES + 1) * DT_MDL, open_road, car_state(5.0)).stop_x is None
    fs2, _ = committed()
    short_clear = obs(path_end=25.0, should_stop=False, braking=False, moving=True)
    assert run(fs2, 3.0, short_clear, car_state(5.0)).stop_x is not None
    assert run(fs2, 3.0, short_clear, car_state(5.0)).stop_x is None

  def test_latched_point_follows_the_model_forward_and_down_at_bounded_rates(self):
    fs, _ = committed(path_end=20.0)
    before = fs.remaining
    fs.update(obs(path_end=30.0), car_state(10.0), True, True, True)
    assert math.isclose(fs.remaining, before - 10.0 * DT_MDL + 3.0 * DT_MDL, rel_tol=1e-6, abs_tol=1e-9)
    fs.remaining = 12.0
    fs.update(obs(path_end=8.0), car_state(2.0), True, True, True)
    assert math.isclose(fs.remaining, 12.0 - 2.0 * DT_MDL - 2.0 * DT_MDL, rel_tol=1e-6, abs_tol=1e-9)

  def test_the_latched_point_follows_a_far_drifting_endpoint_while_the_model_still_calls_the_stop(self):
    fs, _ = committed(path_end=20.0)
    before = fs.remaining
    fs.update(obs(path_end=60.0), car_state(10.0), True, True, True)   # 6 s out: beyond the latch window, still a stop
    assert math.isclose(fs.remaining, before - 10.0 * DT_MDL + 3.0 * DT_MDL, rel_tol=1e-6, abs_tol=1e-9)
    before = fs.remaining
    fs.update(obs(path_end=60.0, should_stop=False, braking=False), car_state(10.0), True, True, True)   # a green: no stop call, no extension
    assert math.isclose(fs.remaining, before - 10.0 * DT_MDL, rel_tol=1e-6, abs_tol=1e-9)


class TestHold:
  def test_a_hold_needs_a_commitment_or_a_recent_release(self):
    fs = ForceStops()
    assert not run(fs, 2.0, obs(path_end=4.0), car_state(0.0, standstill=True)).holding
    fs, _ = holding()
    run(fs, 0.05, obs(path_end=4.0, lead=True, relevant=True), car_state(0.0, standstill=True))
    run(fs, REARM_S + 0.5, obs(path_end=90.0, should_stop=False, braking=False, moving=True), car_state(0.0, standstill=True))
    assert not run(fs, 0.5, obs(path_end=4.0), car_state(0.0, standstill=True)).holding

  def test_standstill_turns_a_commitment_into_a_hold(self):
    _, result = holding()
    assert result.holding and result.v_cruise_cap == 0.0 and result.stop_x is not None and result.stop_x >= -STOP_DISTANCE

  def test_the_hold_ignores_a_flickering_stop_signal(self):
    fs, _ = holding()
    for i in range(frames(3.0)):
      result = fs.update(obs(path_end=4.0, should_stop=(i % 2 == 0)), car_state(0.0, standstill=True), True, True, True)
      assert result.holding

  def test_launch_evidence_releases_the_hold(self):
    fs, _ = holding()
    result = run(fs, 0.5, obs(path_end=60.0, should_stop=False, braking=False, release_open=True, moving=True), car_state(0.0, standstill=True))
    assert not result.holding and result.stop_x is None

  def test_a_mode_exit_does_not_release_the_hold(self):
    fs, _ = holding()
    assert run(fs, 2.0, obs(path_end=4.0), car_state(0.0, standstill=True), experimental=False).holding

  def test_a_relevant_lead_releases_and_the_hold_re_enters_when_it_leaves(self):
    fs, _ = holding()
    assert not run(fs, 0.05, obs(path_end=4.0, lead=True, relevant=True), car_state(0.0, standstill=True)).holding
    assert run(fs, 0.05, obs(path_end=4.0), car_state(0.0, standstill=True)).holding

  def test_a_far_lead_does_not_break_the_hold(self):
    fs, _ = holding()
    assert run(fs, 1.0, obs(path_end=4.0, lead=True, relevant=False), car_state(0.0, standstill=True)).holding

  def test_a_gas_tap_re_stop_re_enters_the_hold_inside_the_grace(self):
    fs, _ = holding()
    assert not run(fs, 0.2, obs(path_end=4.0), car_state(0.5, gas=True)).holding
    assert run(fs, 0.5, obs(path_end=4.0), car_state(0.0, standstill=True)).holding

  def test_rollback_flicker_keeps_the_latch_and_creep_returns_to_a_commitment(self):
    fs, _ = holding()
    assert run(fs, 0.5, obs(path_end=4.0), car_state(0.5, standstill=False)).holding
    result = run(fs, 0.05, obs(path_end=4.0), car_state(0.9, standstill=False))
    assert not result.holding and fs.forcing and result.stop_x is not None
    assert run(fs, 0.05, obs(path_end=4.0), car_state(0.0, standstill=True)).holding

  def test_model_invalid_while_holding_is_debounced(self):
    fs, _ = holding()
    assert run(fs, MODEL_INVALID_RELEASE_S - 0.1, obs(path_end=4.0), car_state(0.0, standstill=True), valid=False).holding
    assert not run(fs, 0.2, obs(path_end=4.0), car_state(0.0, standstill=True), valid=False).holding

  def test_fallback_release_needs_mostly_clear_moving_frames(self):
    fs, _ = holding()
    ambiguous = obs(path_end=30.0, should_stop=False, braking=False, moving=True, corridor_clear=True)
    assert run(fs, CLEAR_WINDOW_S - 0.2, ambiguous, car_state(0.0, standstill=True)).holding
    assert not run(fs, 0.3, ambiguous, car_state(0.0, standstill=True)).holding
    fs, _ = holding()
    for i in range(frames(CLEAR_WINDOW_S + 1.0)):
      result = fs.update(obs(path_end=30.0, should_stop=(i % 3 != 0), braking=False, moving=True), car_state(0.0, standstill=True), True, True, True)
    assert result.holding


class TestApproachProfile:
  def test_the_profile_is_the_constant_deceleration_to_the_landing_entered_at_the_jerk_limit(self):
    fs, result = commit_on_strict_evidence(13.0, 60.0, a_ego=-1.2)
    world = fs.remaining
    previous = result.a_target
    assert -1.2 - PROFILE_JERK * DT_MDL * 3 <= previous <= -1.2   # entered from the car's own deceleration, not from zero
    for _ in range(frames(1.0)):
      world -= 13.0 * DT_MDL
      result = fs.update(obs(path_end=world + LATCH_SETBACK, should_stop=False, strict=True, braking=True), car_state_accel(13.0, previous), True, True, True)
      assert result.a_target is not None and result.a_target <= 0.0
      assert previous - result.a_target <= PROFILE_JERK * DT_MDL + 1e-9
      previous = result.a_target
    need = 13.0 ** 2 / (2.0 * (fs.remaining - PROFILE_LANDING))
    assert math.isclose(result.a_target, -need, rel_tol=1e-6, abs_tol=1e-9)

  def test_following_the_profile_holds_it_flat_and_hands_over_short_of_the_point(self):
    fs, _ = commit_on_strict_evidence(13.0, 60.0, a_ego=-1.0)
    v = 13.0
    a = -1.0
    world = fs.remaining
    history = []
    for _ in range(frames(15.0)):
      result = fs.update(obs(path_end=world + LATCH_SETBACK, should_stop=False, strict=True, braking=True), car_state_accel(v, a), True, True, True)
      if result.a_target is None:
        break
      a = result.a_target
      history.append((v, a, fs.remaining))
      v = max(v + a * DT_MDL, 0.0)
      world -= v * DT_MDL
    flat = [a for v, a, _ in history if 9.0 < v < 12.5]
    assert max(flat) - min(flat) < 0.1                           # constant deceleration once entered ...
    easing = [a for v, a, _ in history if 3.0 < v < 9.0]
    assert all(later >= earlier - 1e-6 for earlier, later in zip(easing, easing[1:], strict=False))   # ... then only ever easing off
    assert easing[-1] - easing[0] < 0.6                          # gently: the landing margin shrinks with the remaining distance
    assert min(a for _, a, _ in history) > -2.2                  # a 13 m/s stop seen 60 m out never needs more than ~2 m/s^2
    assert history[-1][0] <= PROFILE_HANDOVER_SPEED               # the profile fades out below the handover speed ...
    assert history[-1][2] >= 0.0                                 # ... never past the committed point: the column and the hold land

  def test_the_profile_is_capped_and_absent_without_a_moving_commitment(self):
    fs, _ = commit_on_strict_evidence(20.0, 60.0)
    world = fs.remaining
    for _ in range(frames(1.6)):
      world -= 20.0 * DT_MDL
      result = fs.update(obs(path_end=world + LATCH_SETBACK, should_stop=False, strict=True, braking=True), car_state_accel(20.0, 0.0), True, True, True)
    assert math.isclose(result.a_target, -PROFILE_MAX_DECEL, rel_tol=1e-6, abs_tol=1e-9)
    shaping = ForceStops()
    assert run(shaping, 0.5, obs(path_end=80.0, should_stop=False, early=True, braking=True), car_state(15.0)).a_target is None
    _, held = holding()
    assert held.a_target is None


class TestFieldTest4:
  def test_the_profile_tapers_with_the_speed_as_the_landing_closes(self):
    fs, _ = commit_on_strict_evidence(13.0, 60.0, a_ego=-1.0)
    v, a, world = 13.0, -1.0, fs.remaining
    history = []
    for _ in range(frames(15.0)):
      result = fs.update(obs(path_end=world + LATCH_SETBACK, should_stop=False, strict=True, braking=True), car_state_accel(v, a), True, True, True)
      if result.a_target is None:
        break
      a = result.a_target
      history.append((v, a, fs.remaining))
      v = max(v + a * DT_MDL, 0.0)
      world -= v * DT_MDL
    tail = [(v, a) for v, a, _ in history if v < 3.0]
    assert all(-a <= v / (2.0 * PROFILE_MIN_TIME) + 1e-6 for v, a in tail)          # never harder than v/2 near the end ...
    assert all(later >= earlier - 1e-3 for (_, earlier), (_, later) in zip(tail, tail[1:], strict=False))   # ... and only easing

  def test_no_speed_cap_once_committed(self):
    fs, result = committed()
    assert result.v_cruise_cap == NO_CAP and result.stop_x is not None
    shaping = ForceStops()
    assert run(shaping, 0.5, obs(path_end=80.0, should_stop=False, early=True, braking=True), car_state(15.0)).v_cruise_cap < NO_CAP

  def test_a_lane_change_drops_the_commitment_and_the_shaping(self):
    fs, _ = committed()
    assert run(fs, DT_MDL, obs(lane_change=True), car_state(10.0)).stop_x is None and not fs.forcing
    shaping = ForceStops()
    assert run(shaping, 0.5, obs(path_end=80.0, should_stop=False, early=True, braking=True, lane_change=True), car_state(15.0)).v_cruise_cap == NO_CAP
    fs, held = holding()
    assert run(fs, DT_MDL, obs(path_end=4.0, lane_change=True), car_state(0.0, standstill=True)).holding   # a hold is not a lane

  def test_an_open_path_releases_the_hold_in_three_frames_but_a_flash_does_not(self):
    fs, _ = holding()
    for _ in range(RELEASE_OPEN_FRAMES - 1):
      assert run(fs, DT_MDL, obs(path_end=RELEASE_OPEN_LENGTH + 20.0, should_stop=False), car_state(0.0, standstill=True)).holding
    assert run(fs, DT_MDL, obs(path_end=4.0), car_state(0.0, standstill=True)).holding                       # the flash ends: still held
    for _ in range(RELEASE_OPEN_FRAMES):
      result = run(fs, DT_MDL, obs(path_end=RELEASE_OPEN_LENGTH + 20.0, should_stop=False), car_state(0.0, standstill=True))
    assert not result.holding and not fs.holding


class TestMovingGreenRelease:
  # route 0x2c t=1105/1135: the light turned green mid-approach; the commitment must let go with the road, not 4 s later
  def test_an_open_path_releases_a_moving_commitment_in_three_frames(self):
    fs, _ = committed()
    for _ in range(RELEASE_OPEN_FRAMES - 1):
      result = fs.update(obs(path_end=RELEASE_OPEN_LENGTH + 10.0, should_stop=False, braking=False, moving=True), car_state(8.0), True, True, True)
      assert result.a_target is not None or result.stop_x is not None or fs.forcing
    result = fs.update(obs(path_end=RELEASE_OPEN_LENGTH + 10.0, should_stop=False, braking=False, moving=True), car_state(8.0), True, True, True)
    assert not fs.forcing and result.stop_x is None

  def test_a_noisy_dip_or_lingering_stop_evidence_resets_the_release(self):
    fs, _ = committed()
    fs.update(obs(path_end=RELEASE_OPEN_LENGTH + 10.0, should_stop=False, braking=False, moving=True), car_state(8.0), True, True, True)
    # one short-path frame between open frames: the counter starts over
    fs.update(obs(path_end=10.0), car_state(8.0), True, True, True)
    for _ in range(RELEASE_OPEN_FRAMES - 1):
      fs.update(obs(path_end=RELEASE_OPEN_LENGTH + 10.0, should_stop=False, braking=False, moving=True), car_state(8.0), True, True, True)
    assert fs.forcing
    # a long path that still carries strict stop evidence is not a green
    fs2, _ = committed()
    for _ in range(RELEASE_OPEN_FRAMES + 2):
      fs2.update(obs(path_end=RELEASE_OPEN_LENGTH + 10.0, should_stop=False, strict=True, moving=True), car_state(8.0), True, True, True)
    assert fs2.forcing

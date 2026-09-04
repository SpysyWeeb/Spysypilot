import math

import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.cereal import log
from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.values import CAR as HYUNDAI
from opendbc.car.structs import car
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.latcontrol_rack import FALLBACK_HOLD_S
from openpilot.selfdrive.controls.lib.rack_trajectory import (
  HOLD_TOPUP_ERROR_CAP_DEG,
  HOLD_TOPUP_LEAK_RC_S,
  HOLD_TOPUP_OVERRIDE_DECAY_S,
  HOLD_TOPUP_RATE,
  HOLD_TOPUP_RELEASE_COOLDOWN_S,
  HOLD_TOPUP_ZERO_EPS_TORQUE,
  INACTIVE_HOLD_FRAMES,
  MAX_HOLD_TOPUP_TORQUE,
  R7_MAX_TORQUE_STEP,
  RackTrajectoryController,
  STALE_MODEL_S,
  _hold_topup_step,
)
import openpilot.selfdrive.controls.lib.rack_trajectory as rack_trajectory
from openpilot.selfdrive.controls.tests.test_latcontrol_rack import get_rack_controller, horizon_model


def hold_fixture(speed=8.0, curvature=0.0156, initial_angle_deg=0.0):
  """A constant-curvature model at `speed`: the plan settles at the matching steering angle and the
  test pins the measured wheel wherever it likes relative to that plan (the plan never follows the
  wheel while unpressed), which is exactly the standing-shortfall geometry of FM3.14."""
  CarInterface = interfaces[HYUNDAI.HYUNDAI_PALISADE]
  CP = CarInterface.get_non_essential_params(HYUNDAI.HYUNDAI_PALISADE)
  CI = CarInterface(CP)
  VM = VehicleModel(CP)
  controller = RackTrajectoryController()
  CS = car.CarState.new_message()
  CS.vEgo = speed
  CS.steeringAngleDeg = initial_angle_deg
  params = log.VehicleParameters.new_message()
  model = horizon_model([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], [curvature * speed] * 6, [speed] * 6)
  model.action.desiredCurvature = curvature
  torque_from_lateral_accel = CI.torque_from_lateral_accel()

  def step(frame):
    model.timestampEof = 1_000_000_000 + (frame // 5) * 50_000_000
    controller.set_model(model, model.timestampEof + 30_000_000)
    output = controller.update(True, CS, VM, params, CP.lateralTuning.torque, torque_from_lateral_accel, .2, curvature)
    assert output is not None
    return output

  return controller, CS, step


def pin_short_of_plan(CS, output, offset_deg):
  """Hold the wheel `offset_deg` short of the plan on the plan's own side (negative offset: beyond it)."""
  plan = output.planned_angle_deg
  CS.steeringAngleDeg = plan - offset_deg * (math.copysign(1.0, plan) if plan != 0.0 else -1.0)


def settle(controller, CS, step, frames=400):
  """Run the plan out to its target with the wheel exactly on it: the term stays exactly 0.0."""
  output = step(0)
  for frame in range(1, frames):
    pin_short_of_plan(CS, output, 0.0)
    output = step(frame)
  assert abs(output.planned_rate_deg_s) < 0.5, output.planned_rate_deg_s
  # the wheel trailed the moving plan by one frame on the way here; start the term from a clean zero
  controller.hold_topup_torque = 0.0
  pin_short_of_plan(CS, output, 0.0)
  output = step(frames)
  assert output.hold_topup_torque == 0.0
  return output, frames + 1


class TestHoldTopup(OpenpilotTestCase):
  """FM3.14: the third torque term (predict / correct position / make up the standing shortfall)."""

  # ---- the pure step ----

  def test_step_never_grows_when_not_accumulating(self):
    state = 0.1
    for _ in range(300):
      new = _hold_topup_step(state, 5.0, 1.0, False, False, 0.01)
      assert abs(new) <= abs(state)
      state = new
    assert 0.0 <= state < 0.1

  def test_step_matches_the_closed_form(self):
    rc = HOLD_TOPUP_LEAK_RC_S
    for error in (1.5, 2.0):
      state = 0.0
      for frame in range(1, 51):
        state = _hold_topup_step(state, error, 1.0, True, False, 0.01)
        t = frame * 0.01
        if t in (0.3, 0.4, 0.5):
          expected = HOLD_TOPUP_RATE * error * rc * (1.0 - math.exp(-t / rc))
          assert math.isclose(state, expected, rel_tol=0.03), (error, t, state, expected)

  def test_step_is_bounded_and_the_error_is_capped(self):
    state = 0.0
    fast = 0.0
    for _ in range(3000):
      state = _hold_topup_step(state, 40.0, 1.0, True, False, 0.01)
      fast = _hold_topup_step(fast, HOLD_TOPUP_ERROR_CAP_DEG, 1.0, True, False, 0.01)
      assert abs(state) <= MAX_HOLD_TOPUP_TORQUE + 1e-12
      assert state == fast  # 40 deg and the cap grow identically
    assert state == MAX_HOLD_TOPUP_TORQUE

  def test_step_leaks_at_the_passive_and_the_fast_rate(self):
    peak = 0.15
    state = peak
    for _ in range(int(3 * HOLD_TOPUP_LEAK_RC_S / 0.01)):
      state = _hold_topup_step(state, 0.0, 1.0, False, False, 0.01)
    assert math.isclose(state, peak * math.exp(-3.0), rel_tol=0.05)
    state = peak
    for _ in range(int(2 * HOLD_TOPUP_OVERRIDE_DECAY_S / 0.01)):
      state = _hold_topup_step(state, 5.0, 1.0, False, True, 0.01)
    assert state <= 0.14 * peak

  def test_step_reaches_exact_zero(self):
    state = 0.01
    frames = 0
    while state != 0.0:
      state = _hold_topup_step(state, 0.0, 1.0, False, False, 0.01)
      frames += 1
      assert frames < 5000
    assert _hold_topup_step(0.0, 0.0, 1.0, False, False, 0.01) == 0.0
    assert HOLD_TOPUP_ZERO_EPS_TORQUE < 1e-3

  def test_step_per_frame_bound_under_gate_thrashing(self):
    rng = np.random.default_rng(7)
    bound = 0.01 * (HOLD_TOPUP_RATE * HOLD_TOPUP_ERROR_CAP_DEG + MAX_HOLD_TOPUP_TORQUE / HOLD_TOPUP_OVERRIDE_DECAY_S)
    state = 0.0
    for _ in range(5000):
      fast = bool(rng.integers(2))
      accumulating = bool(rng.integers(2))
      new = _hold_topup_step(state, float(rng.choice([-40.0, -5.0, -1.0, 1.0, 5.0, 40.0])), float(rng.random()), accumulating, fast, 0.01)
      assert abs(new - state) <= bound + 1e-9
      state = new
    assert bound < R7_MAX_TORQUE_STEP / 4

  # ---- through the controller ----

  def test_zero_on_engage_and_bit_identical_while_gated(self):
    # a fresh engage is exactly 0.0, and with the driver pressing throughout (growth gated off from
    # the first frame) every output is bit-identical to a controller with the term pinned to zero
    outputs = []
    for rate in (HOLD_TOPUP_RATE, 0.0):
      rack_trajectory.HOLD_TOPUP_RATE = rate
      try:
        controller, CS, step = hold_fixture()
        CS.steeringPressed = True
        run = []
        output = step(0)
        assert output.hold_topup_torque == 0.0
        run.append(output.torque)
        for frame in range(1, 300):
          pin_short_of_plan(CS, output, 1.5)
          output = step(frame)
          assert output.hold_topup_torque == 0.0
          assert not output.hold_topup_growing
          run.append(output.torque)
        outputs.append(run)
      finally:
        rack_trajectory.HOLD_TOPUP_RATE = HOLD_TOPUP_RATE
    assert outputs[0] == outputs[1]

  def test_grows_toward_the_feedforward_under_a_standing_error_and_stays_bounded(self):
    controller, CS, step = hold_fixture()
    output, frame = settle(controller, CS, step)
    start = output.hold_topup_torque
    values = []
    for i in range(1, 2001):
      pin_short_of_plan(CS, output, 1.5)
      output = step(frame + i)
      values.append(output.hold_topup_torque)
      # the term pushes the way the feedforward does: toward more angle, the plan's own direction
      if output.hold_topup_torque != start:
        assert math.copysign(1.0, output.hold_topup_torque - start) == math.copysign(1.0, output.feedforward_torque)
      assert abs(output.hold_topup_torque) <= MAX_HOLD_TOPUP_TORQUE + 1e-12
    growth_half_s = abs(values[49] - start)
    assert 0.02 <= growth_half_s <= 0.08, growth_half_s
    assert max(abs(v) for v in values) == MAX_HOLD_TOPUP_TORQUE
    # monotonic while the error stands (the leak never wins against a held error under the cap)
    assert all(abs(b) >= abs(a) - 1e-9 for a, b in zip(values, values[1:], strict=False))

  def test_no_speed_term(self):
    growth = {}
    for speed in (8.0, 16.0, 30.0):
      controller, CS, step = hold_fixture(speed=speed, curvature=0.3 / speed ** 2)
      output, frame = settle(controller, CS, step)
      start = output.hold_topup_torque
      for i in range(1, 51):
        pin_short_of_plan(CS, output, 1.5)
        output = step(frame + i)
      growth[speed] = abs(output.hold_topup_torque - start)
    assert max(growth.values()) < 1.5 * min(growth.values()), growth

  def test_disturbance_plant_closes_the_shortfall_without_overshoot(self):
    # a rack that needs more torque to hold than the lateral-accel feedforward predicts (the route 0x3e
    # geometry): wheel rate = K * (applied torque - hold torque the rack really needs at this angle)
    def run(rate):
      rack_trajectory.HOLD_TOPUP_RATE = rate
      try:
        controller, CS, step = hold_fixture()
        angle, plans, angles, topups = 0.0, [], [], []
        sign_relation = None
        for frame in range(1500):
          CS.steeringAngleDeg = angle
          output = step(frame)
          if sign_relation is None and abs(output.planned_angle_deg) > 5.0:
            sign_relation = math.copysign(1.0, output.feedforward_torque * output.planned_angle_deg)
          need = 0.011 * angle * (sign_relation or 1.0)  # 0.54 at the 49 deg plan: ~30 % above the feedforward
          rate_deg_s = 20.0 * (output.torque - need)
          angle += 0.01 * rate_deg_s
          CS.steeringRateDeg = rate_deg_s
          plans.append(output.planned_angle_deg)
          angles.append(angle)
          topups.append(output.hold_topup_torque)
        return np.asarray(plans), np.asarray(angles), np.asarray(topups)
      finally:
        rack_trajectory.HOLD_TOPUP_RATE = HOLD_TOPUP_RATE

    plans, angles, topups = run(HOLD_TOPUP_RATE)
    plans0, angles0, _ = run(0.0)
    gap = np.abs(plans[-200:] - angles[-200:]).mean()
    gap0 = np.abs(plans0[-200:] - angles0[-200:]).mean()
    assert gap0 > 1.0, gap0  # the shortfall is real without the term
    assert gap < 0.5 * gap0, (gap, gap0)
    overshoot = (np.abs(angles) - np.abs(plans))[500:].max()
    assert overshoot < 0.3, overshoot
    assert abs(topups[-1]) > 0.05

  def test_freezes_on_a_press_decays_fast_and_cools_down_before_growing_again(self):
    controller, CS, step = hold_fixture()
    output, frame = settle(controller, CS, step)
    for i in range(1, 301):
      pin_short_of_plan(CS, output, 1.5)
      output = step(frame + i)
    frame += 300
    before_press = output.hold_topup_torque
    assert abs(before_press) > 0.1
    CS.steeringPressed = True
    for i in range(1, 16):
      pin_short_of_plan(CS, output, 1.5)
      previous = output.hold_topup_torque
      output = step(frame + i)
      assert not output.hold_topup_growing
      assert abs(output.hold_topup_torque) <= abs(previous)
    frame += 15
    CS.steeringPressed = False
    at_release = output.hold_topup_torque
    cooldown = round(HOLD_TOPUP_RELEASE_COOLDOWN_S / 0.01)
    for i in range(1, cooldown + 1):
      pin_short_of_plan(CS, output, 1.5)
      output = step(frame + i)
      assert not output.hold_topup_growing
    frame += cooldown
    # the fast leak ran through the whole cooldown: one time constant gone, not a tenth of one
    assert math.isclose(abs(output.hold_topup_torque), abs(at_release) * math.exp(-HOLD_TOPUP_RELEASE_COOLDOWN_S / HOLD_TOPUP_OVERRIDE_DECAY_S), rel_tol=0.1)
    after_cooldown = output.hold_topup_torque
    for i in range(1, 51):
      pin_short_of_plan(CS, output, 1.5)
      output = step(frame + i)
    assert output.hold_topup_growing
    assert abs(output.hold_topup_torque) > abs(after_cooldown)

  def test_no_growth_into_a_feedback_cap_in_the_error_direction(self):
    # the wheel 10 deg beyond the plan: the proportional return already saturates its cap, so the
    # term must not stack on top of it -- and resumes once the error is small enough for P alone
    controller, CS, step = hold_fixture()
    output, frame = settle(controller, CS, step)
    for i in range(1, 101):
      pin_short_of_plan(CS, output, -10.0)
      output = step(frame + i)
      assert output.feedback_limited
      assert not output.hold_topup_growing
      assert output.hold_topup_torque == 0.0
    frame += 100
    for i in range(1, 101):
      pin_short_of_plan(CS, output, -2.0)
      output = step(frame + i)
    assert output.hold_topup_growing
    # beyond the plan the term unwinds: opposite to the feedforward, toward the plan
    assert math.copysign(1.0, output.hold_topup_torque) == -math.copysign(1.0, output.feedforward_torque)

  def test_no_growth_while_the_plan_moves_or_the_guard_is_active(self):
    controller, CS, step = hold_fixture()
    output = step(0)
    for frame in range(1, 200):
      pin_short_of_plan(CS, output, 1.5)
      output = step(frame)
      if abs(output.planned_rate_deg_s) > 8.0:
        assert abs(output.hold_topup_torque) < 0.005
    output, frame = settle(controller, CS, step)
    for i in range(1, 201):
      pin_short_of_plan(CS, output, 1.5)
      output = step(frame + i)
    frame += 200
    grown = output.hold_topup_torque
    controller.direction_guard_scale = 1.0  # a conflict the guard is still blending out of
    guarded_frames = 0
    for i in range(1, 201):
      pin_short_of_plan(CS, output, 1.5)
      previous = output.hold_topup_torque
      output = step(frame + i)
      if controller.direction_guard_scale > 0.0:
        guarded_frames += 1
        assert not output.hold_topup_growing
        assert abs(output.hold_topup_torque) <= abs(previous)
      elif guarded_frames:
        break
    assert guarded_frames > 5
    assert abs(output.hold_topup_torque) < abs(grown)

  def test_no_growth_against_a_wanted_unwind(self):
    # target straight ahead, wheel well beyond the plan on the plan's side: a pure unwind the rack's
    # own self-aligning torque produces -- the term must not push it
    controller, CS, step = hold_fixture(curvature=0.0, initial_angle_deg=20.0)
    output = step(0)
    for frame in range(1, 600):
      output = step(frame)
      assert abs(output.hold_topup_torque) < 0.01, (frame, output.hold_topup_torque, output.direction_fraction)
    assert abs(output.planned_angle_deg) < 0.5
    assert output.direction_fraction > 0.95

  def test_persists_through_a_blip_and_is_zeroed_by_reset(self):
    controller, CS, step = hold_fixture()
    output, frame = settle(controller, CS, step)
    for i in range(1, 201):
      pin_short_of_plan(CS, output, 1.5)
      output = step(frame + i)
    frame += 200
    held = controller.hold_topup_torque
    assert abs(held) > 0.05
    for _ in range(3):
      controller.hold()
    assert controller.hold_topup_torque == held
    pin_short_of_plan(CS, output, 1.5)
    output = step(frame + 1)
    assert output.hold_topup_torque == held
    controller.reset()
    assert controller.hold_topup_torque == 0.0
    assert controller.hold_topup_cooldown_frames == 0

  def test_logged_through_the_rack_controller_and_defaulted_by_stock(self):
    controller, stock, VM = get_rack_controller()
    CS = car.CarState.new_message()
    CS.vEgo = 8.0
    params = log.VehicleParameters.new_message()
    _, _, rack_log = controller.update(True, CS, VM, params, False, 0.002, False, 0.2)
    assert rack_log.fallback
    assert rack_log.holdTopupTorque == 0.0
    assert not rack_log.holdTopupGrowing
    model = horizon_model([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], [0.0156 * 8.0] * 6, [8.0] * 6)
    model.action.desiredCurvature = 0.0156
    planned = 0.0
    for frame in range(600):
      CS.steeringAngleDeg = planned - math.copysign(1.5, planned) if planned != 0.0 else 0.0
      model.timestampEof = 1_000_000_000 + frame * 10_000_000
      _, planned, rack_log = controller.update(True, CS, VM, params, False, 0.0156, False, 0.2, model=model, mono_time_ns=model.timestampEof + 50_000_000)
      assert not rack_log.fallback
    assert abs(rack_log.holdTopupTorque) > 0.02
    assert rack_log.holdTopupGrowing

  def test_no_growth_when_the_target_disagrees_with_the_plan_and_a_reversed_residual_drains_fast(self):
    controller, CS, step = hold_fixture()
    output, frame = settle(controller, CS, step)
    for i in range(1, 301):
      pin_short_of_plan(CS, output, 1.5)
      output = step(frame + i)
    frame += 300
    grown = output.hold_topup_torque
    assert abs(grown) > 0.1
    # the wheel now beyond the plan: the accumulated push opposes the error and must drain at the fast
    # rate (gone within ~0.3 s), not linger for seconds at the passive leak
    for i in range(1, 31):
      pin_short_of_plan(CS, output, -1.5)
      output = step(frame + i)
    frame += 30
    assert abs(output.hold_topup_torque) < 0.45 * abs(grown), (output.hold_topup_torque, grown)
    # plan and target on opposite sides of the wheel: no growth at all, whatever the plan error
    controller.hold_topup_torque = 0.0
    for i in range(1, 201):
      # a wheel past the served target on the target's side while the plan is still short of it
      plan_short = abs(output.planned_angle_deg) < abs(output.near_target_angle_deg)
      CS.steeringAngleDeg = output.near_target_angle_deg * 1.02 if plan_short else output.planned_angle_deg
      output = step(frame + i)
      if abs(output.planned_angle_deg) < abs(output.near_target_angle_deg) and abs(CS.steeringAngleDeg) > abs(output.planned_angle_deg):
        assert not output.hold_topup_growing

  def test_composed_torque_r7_bound_across_a_press_and_release(self):
    # the review's finding: the assist branch slews the committed torque toward its cap during an
    # opposing press, and on release the composed request -- the term's carried-in value included --
    # must not come back in one step; the slew carries on until the request is within a step
    controller, CS, step = hold_fixture()
    output, frame = settle(controller, CS, step)
    for i in range(1, 1201):
      pin_short_of_plan(CS, output, 4.5)
      output = step(frame + i)
    frame += 1200
    assert abs(output.hold_topup_torque) == MAX_HOLD_TOPUP_TORQUE
    assert abs(output.torque) > 0.6
    previous = output.torque
    CS.steeringPressed = True
    CS.steeringTorque = -math.copysign(200.0, output.torque)  # an opposing hand above the pressed threshold
    trace = []
    for i in range(1, 16):
      pin_short_of_plan(CS, output, 4.5)
      output = step(frame + i)
      trace.append(output.torque - previous)
      previous = output.torque
    frame += 15
    assert abs(output.torque) <= 0.5 + 1e-9  # slewed down to the opposing-driver floor
    CS.steeringPressed = False
    CS.steeringTorque = 0.0
    for i in range(1, 61):
      pin_short_of_plan(CS, output, 4.5)
      output = step(frame + i)
      trace.append(output.torque - previous)
      previous = output.torque
    assert max(abs(d) for d in trace) <= R7_MAX_TORQUE_STEP + 1e-9, max(abs(d) for d in trace)
    assert abs(output.torque) > 0.55  # and the request did come back once reconciled

  def test_creeping_wheel_grows_and_an_approaching_wheel_fades(self):
    # the route 0x3e geometry: the plan static, the wheel leaving it at 2 deg/s -- the term must grow;
    # a wheel already closing on the plan at 1 deg/s -- growth must fade (the approach gate)
    def run(rate_deg_s, offset_deg, frames=100):
      controller, CS, step = hold_fixture()
      output, frame = settle(controller, CS, step)
      plan = output.planned_angle_deg
      away = -math.copysign(1.0, plan)  # toward center, i.e. short of the plan
      angle = plan + away * offset_deg
      for i in range(1, frames + 1):
        angle += away * rate_deg_s * 0.01
        CS.steeringAngleDeg = angle
        CS.steeringRateDeg = away * rate_deg_s
        output = step(frame + i)
      return abs(output.hold_topup_torque)
    departing = run(2.0, 0.5)
    approaching = run(-1.0, 2.5)
    assert departing > 0.02, departing
    assert approaching < 0.3 * departing, (approaching, departing)

  def test_growth_resumes_on_the_first_frame_after_the_cooldown(self):
    controller, CS, step = hold_fixture()
    output, frame = settle(controller, CS, step)
    for i in range(1, 201):
      pin_short_of_plan(CS, output, 1.5)
      output = step(frame + i)
    frame += 200
    CS.steeringPressed = True
    for i in range(1, 6):
      pin_short_of_plan(CS, output, 1.5)
      output = step(frame + i)
    frame += 5
    CS.steeringPressed = False
    cooldown = round(HOLD_TOPUP_RELEASE_COOLDOWN_S / 0.01)
    for i in range(1, cooldown + 1):
      pin_short_of_plan(CS, output, 1.5)
      output = step(frame + i)
      assert not output.hold_topup_growing, i
    pin_short_of_plan(CS, output, 1.5)
    output = step(frame + cooldown + 1)
    assert output.hold_topup_growing

  def test_six_inactive_frames_reset_the_term_through_hold(self):
    controller, CS, step = hold_fixture()
    output, frame = settle(controller, CS, step)
    for i in range(1, 201):
      pin_short_of_plan(CS, output, 1.5)
      output = step(frame + i)
    assert abs(controller.hold_topup_torque) > 0.05
    for _ in range(INACTIVE_HOLD_FRAMES + 1):
      controller.hold()
    assert controller.hold_topup_torque == 0.0
    assert controller.hold_topup_cooldown_frames == 0
    assert not controller.release_reconcile

  def test_stale_model_fallback_zeroes_a_grown_term_and_resumes_from_zero(self):
    controller, stock, VM = get_rack_controller()
    CS = car.CarState.new_message()
    CS.vEgo = 8.0
    params = log.VehicleParameters.new_message()
    model = horizon_model([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], [0.0156 * 8.0] * 6, [8.0] * 6)
    model.action.desiredCurvature = 0.0156
    planned = 0.0
    for frame in range(600):
      CS.steeringAngleDeg = planned - math.copysign(1.5, planned) if planned != 0.0 else 0.0
      model.timestampEof = 1_000_000_000 + frame * 10_000_000
      _, planned, rack_log = controller.update(True, CS, VM, params, False, 0.0156, False, 0.2, model=model, mono_time_ns=model.timestampEof + 50_000_000)
    assert abs(rack_log.holdTopupTorque) > 0.02
    stale_ns = model.timestampEof + int(STALE_MODEL_S * 1e9) + 10_000_000
    _, _, rack_log = controller.update(True, CS, VM, params, False, 0.0156, False, 0.2, model=model, mono_time_ns=stale_ns)
    assert rack_log.fallback
    assert controller.rack.hold_topup_torque == 0.0
    assert rack_log.holdTopupTorque == 0.0
    hold_frames = int(FALLBACK_HOLD_S / DT_CTRL)
    for frame in range(hold_frames):
      model.timestampEof = stale_ns + frame * 10_000_000
      _, _, rack_log = controller.update(True, CS, VM, params, False, 0.0156, False, 0.2, model=model, mono_time_ns=model.timestampEof)
      assert rack_log.fallback
      assert rack_log.holdTopupTorque == 0.0
    model.timestampEof = stale_ns + hold_frames * 10_000_000
    _, _, rack_log = controller.update(True, CS, VM, params, False, 0.0156, False, 0.2, model=model, mono_time_ns=model.timestampEof)
    assert not rack_log.fallback
    assert rack_log.holdTopupTorque == 0.0

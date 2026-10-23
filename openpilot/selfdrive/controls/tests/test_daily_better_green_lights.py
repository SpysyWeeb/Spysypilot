"""Better Green Lights through the real daily MPC, serialized plan and LongControl."""
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import openpilot.cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan
from openpilot.selfdrive.controls.tests.test_daily_blotv3_integration import PlannerMessages, car_params, set_lead
from openpilot.selfdrive.controls.tests.test_daily_curve_integration import curve_messages
from openpilot.selfdrive.modeld.constants import ModelConstants


class Publisher:
  def __init__(self):
    self.messages = {}

  def send(self, service, message):
    self.messages[service] = messaging.log_from_bytes(message.to_bytes())


def set_model(sm, length=5.0, accel=-0.2, should_stop=True, moving=False):
  md = sm['modelV2']
  times = np.asarray(ModelConstants.T_IDXS)
  md.position.x = (times / times[-1] * length).tolist()
  md.position.y = np.zeros_like(times).tolist()
  md.position.z = np.zeros_like(times).tolist()
  md.velocity.x = (np.maximum(times - 1.0, 0.0) if moving else np.zeros_like(times)).tolist()
  md.acceleration.x = (np.where(times > 1.0, 1.0, 0.0) if moving else np.zeros_like(times)).tolist()
  md.orientation.z = np.zeros_like(times).tolist()
  md.action.desiredAcceleration = accel
  md.action.shouldStop = should_stop


def messages():
  sm = PlannerMessages(v_ego=0.0)
  sm['carState'].standstill = True
  sm['controlsState'].longControlState = LongCtrlState.stopping
  set_model(sm)
  return sm


def step(planner, sm, control=None, frames=1):
  for _ in range(frames):
    planner.update(sm)
    pm = Publisher()
    planner.publish(sm, pm)
    plan = pm.messages['longitudinalPlan'].longitudinalPlan
    assert np.isfinite([plan.aTarget, *plan.speeds, *plan.accels, *plan.jerks]).all()
    assert ACCEL_MIN <= plan.aTarget <= ACCEL_MAX
    if control is not None:
      control.update(True, sm['carState'], plan.aTarget, plan.shouldStop, (ACCEL_MIN, ACCEL_MAX))
      sm['controlsState'].longControlState = control.long_control_state
  return plan


def held_model_stop():
  cp = car_params()
  planner, sm, control = LongitudinalPlanner(cp), messages(), LongControl(cp)
  plan = step(planner, sm, control, frames=5)
  assert plan.shouldStop and planner.stop_landing.landing
  assert control.long_control_state == LongCtrlState.stopping
  assert not planner.force_stops.holding and not planner.force_stops.forcing
  return planner, sm, control


def test_held_model_stop_opens_before_lagging_stop_bit_and_releases_landing():
  planner, sm, control = held_model_stop()
  set_model(sm, length=25.0, accel=0.05)
  # The source RC=.3 confidence>.7 requires eight .05 s samples, not a path flash.
  plan = step(planner, sm, control, frames=7)
  assert plan.shouldStop and control.long_control_state == LongCtrlState.stopping
  plan = step(planner, sm, control)
  assert sm['modelV2'].action.shouldStop  # no mutation of the model or the other stop owners
  assert not plan.shouldStop
  assert not planner.stop_landing.landing
  assert control.long_control_state == LongCtrlState.pid
  # Release the landing floor at its existing cruise jerk, not a new acceleration step.
  assert -0.15 < plan.aTarget <= -0.15 + 2.0 * planner.dt + 1e-6
  plan = step(planner, sm, control)
  assert not plan.shouldStop and plan.aTarget > 0.0
  assert control.last_output_accel > 0.0


def test_committed_velocity_plan_is_time_shifted_into_the_published_launch_target():
  cp = car_params()
  planner, sm, control = LongitudinalPlanner(cp, init_a=1.5), messages(), LongControl(cp)
  set_model(sm, length=30.0, accel=0.05, should_stop=False, moving=True)
  plan = step(planner, sm, control, frames=20)
  assert str(plan.longitudinalPlanSource) == 'e2e'
  assert 0.9 < plan.aTarget <= 1.5
  assert not plan.shouldStop
  assert control.long_control_state == LongCtrlState.pid
  assert control.last_output_accel > 0.9


@pytest.mark.parametrize('event', ['off', 'disabled', 'cruise_unset', 'gas', 'brake', 'force_decel', 'mode_exit',
                                 'modelV2', 'radarState', 'carState', 'selfdriveState'])
def test_inactive_or_stale_frames_clear_all_launch_state_and_cannot_precharge_reentry(event):
  planner, sm, control = held_model_stop()
  set_model(sm, length=25.0, accel=0.05)
  step(planner, sm, control, frames=10)
  assert planner.anticipating
  if event == 'off':
    sm['controlsState'].longControlState = LongCtrlState.off
  elif event == 'disabled':
    sm['selfdriveState'].enabled = False
  elif event == 'cruise_unset':
    sm['carState'].vCruise = V_CRUISE_UNSET
  elif event in ('gas', 'brake'):
    setattr(sm['carState'], event + 'Pressed', True)
  elif event == 'force_decel':
    sm['controlsState'].forceDecel = True
  elif event == 'mode_exit':
    sm['selfdriveState'].experimentalMode = False
  else:
    sm.invalid.add(event)
  step(planner, sm, frames=20)
  assert not planner.anticipating
  assert not planner.launch_armed
  assert planner.launch_open.x == 0.0
  fresh = messages()
  set_model(fresh, length=25.0, accel=0.05)
  step(planner, fresh, frames=7)
  assert not planner.anticipating
  step(planner, fresh)
  assert planner.anticipating


@pytest.mark.parametrize('field,damage', [
  ('position.x', 'empty'), ('position.x', 'short'), ('position.x', 'reversed'),
  ('position.y', 'empty'), ('position.z', 'short'), ('velocity.x', 'short'), ('acceleration.x', 'empty'),
  *[(field, damage) for field in ('position.x', 'position.y', 'position.z', 'velocity.x', 'acceleration.x')
    for damage in ('nan', 'inf')],
])
def test_malformed_model_cannot_retain_anticipation_or_apply_launch_assist(field, damage):
  planner, sm, control = held_model_stop()
  set_model(sm, length=25.0, accel=0.05, moving=True)
  step(planner, sm, control, frames=10)
  assert planner.anticipating
  section, axis = field.split('.')
  owner = getattr(sm['modelV2'], section)
  values = list(getattr(owner, axis))
  if damage == 'empty':
    values = []
  elif damage == 'short':
    values = values[:-1]
  elif damage == 'reversed':
    values[4] = values[3] - 1.0
  else:
    values[-1] = float(damage)
  setattr(owner, axis, values)
  plan = step(planner, sm)
  assert not planner.anticipating
  assert not planner.launch_armed
  assert planner.launch_open.x == 0.0
  # Even if the raw model drops its stop bit, bad geometry/kinematics cannot add an accel floor.
  sm['modelV2'].action.shouldStop = False
  plan = step(planner, sm, frames=3)
  assert plan.aTarget <= 0.05 + 1e-6


def step_back(sm, steps):
  # move point i + 1 to a given distance behind point i
  xs = list(sm['modelV2'].position.x)
  for i, back in steps:
    xs[i + 1] = xs[i] - back
  sm['modelV2'].position.x = xs


def test_path_jitter_at_rest_still_arms_and_applies_the_launch_assist():
  cp = car_params()
  planner, sm, control = LongitudinalPlanner(cp), messages(), LongControl(cp)
  # at rest the model's stub steps back by a few cm and its opened path by a few mm
  step_back(sm, [(13, 0.03), (21, 0.08)])
  plan = step(planner, sm, control, frames=5)
  assert plan.shouldStop and planner.stop_landing.landing
  assert planner.launch_armed
  set_model(sm, length=25.0, accel=0.05, moving=True)
  step_back(sm, [(3, 0.0035), (6, 0.002)])
  plan = step(planner, sm, control, frames=8)
  assert planner.anticipating and not plan.shouldStop
  plan = step(planner, sm, control, frames=30)
  expected = source_launch_floor(sm['modelV2'], 0.0, cp.longitudinalActuatorDelay + planner.dt)
  assert expected > 0.7
  assert plan.aTarget == pytest.approx(expected, abs=1e-6)
  assert control.last_output_accel > 0.7


@pytest.mark.parametrize('index', [0, 31])
@pytest.mark.parametrize('speed,accepted', [(4.9, True), (5.1, False)])
def test_backward_path_step_is_bounded_as_a_speed_over_the_point_spacing(index, speed, accepted):
  planner, sm, control = held_model_stop()
  set_model(sm, length=30.0, accel=0.05, moving=True)
  times = ModelConstants.T_IDXS
  step_back(sm, [(index, speed * (times[index + 1] - times[index]))])
  step(planner, sm, control, frames=10)
  assert planner.anticipating == accepted
  assert planner.launch_armed == accepted


@pytest.mark.parametrize('speed', [0.0, 0.2])
def test_path_recollapse_cancels_launch_and_restores_the_hold_even_with_hover_accel(speed):
  planner, sm, control = held_model_stop()
  set_model(sm, length=25.0, accel=0.05)
  step(planner, sm, control, frames=10)
  assert not planner.stop_landing.landing and control.long_control_state == LongCtrlState.pid
  sm['carState'].vEgo = speed
  sm['carState'].standstill = speed == 0.0
  set_model(sm, length=9.99, accel=0.05)
  plan = step(planner, sm, control)
  assert not planner.anticipating
  assert plan.shouldStop
  assert control.long_control_state == LongCtrlState.stopping
  assert control.last_output_accel < 0.0
  if speed == 0.0:
    assert planner.stop_landing.landing
    assert plan.aTarget < 0.0


def test_a_launch_cancelled_at_creep_restores_landing_when_the_wheels_stop():
  planner, sm, control = held_model_stop()
  set_model(sm, length=25.0, accel=0.05)
  step(planner, sm, control, frames=10)
  sm['carState'].vEgo, sm['carState'].standstill = 0.2, False
  set_model(sm, length=5.0, accel=0.05)
  step(planner, sm, control, frames=3)
  assert control.long_control_state == LongCtrlState.stopping
  sm['carState'].vEgo, sm['carState'].standstill = 0.0, True
  # The stop bit may flicker after the cancellation. A hover must not release the hold again.
  sm['modelV2'].action.shouldStop = False
  plan = step(planner, sm, control)
  assert planner.stop_landing.landing
  assert plan.shouldStop
  assert control.long_control_state == LongCtrlState.stopping


@pytest.mark.parametrize('owner', ['force_stops', 'leadOne', 'leadTwo'])
def test_green_cannot_release_another_stop_owners_landing_or_stop_bit(owner):
  planner, sm, control = LongitudinalPlanner(car_params()), messages(), LongControl(car_params())
  if owner == 'force_stops':
    set_model(sm, length=2.0)
  else:
    set_lead(sm, owner, 3.5)
  plan = step(planner, sm, control, frames=40)
  assert plan.shouldStop and planner.stop_landing.landing
  if owner == 'force_stops':
    assert planner.force_stops.holding
  else:
    assert planner.lead_departure.holding
  set_model(sm, length=25.0, accel=0.05, moving=True)
  for _ in range(12):
    plan = step(planner, sm, control)
    assert plan.shouldStop
    assert planner.stop_landing.landing
    assert control.long_control_state == LongCtrlState.stopping
  assert planner.anticipating  # evidence exists; it has no authority over the blocker


def test_green_waits_for_urgent_min_then_releases_without_needing_a_second_path_opening():
  planner, sm, control = held_model_stop()
  set_model(sm, length=25.0, accel=-0.3)
  plan = step(planner, sm, control, frames=10)
  assert planner.anticipating and planner.stop_landing.landing and plan.shouldStop
  sm['modelV2'].action.desiredAcceleration = 0.05
  plan = step(planner, sm, control, frames=3)
  assert not planner.stop_landing.landing
  assert not plan.shouldStop and control.long_control_state == LongCtrlState.pid


def test_curve_limited_gas_override_is_not_learned_as_a_model_limitation():
  sm = curve_messages(curvature=0.04, v_ego=15.0, experimental=True, e2e_accel=-0.2)
  planner = LongitudinalPlanner(car_params(), init_v=15.0)
  plan = step(planner, sm, frames=30)
  assert planner.curve.a_target is not None and planner.curve.a_target < -0.3
  assert str(plan.longitudinalPlanSource) == 'curve'
  assert not planner.accel_boost.boost_eligible
  sm['carState'].gasPressed = True
  sm['controlsState'].longControlState = LongCtrlState.off
  step(planner, sm, frames=10)
  assert planner.accel_boost.total_boost == 0.0


@pytest.mark.parametrize('length,expected', [(19.99, False), (20.0, False), (20.01, True)])
def test_path_open_threshold_is_strictly_twenty_meters(length, expected):
  planner, sm, control = held_model_stop()
  set_model(sm, length=length, accel=0.05)
  step(planner, sm, control, frames=8)
  assert planner.anticipating == expected
  assert planner.launch_open.x == pytest.approx(1.0 - (0.3 / 0.35) ** 8 if expected else 0.0)


@pytest.mark.parametrize('length,speed,expected', [(10.0, 0.2, True), (9.99, 0.2, False),
                                                 (25.0, 2.0, True), (25.0, 2.01, False)])
def test_anticipation_close_and_speed_boundaries(length, speed, expected):
  planner, sm, control = held_model_stop()
  set_model(sm, length=25.0, accel=0.05)
  step(planner, sm, control, frames=10)
  sm['carState'].vEgo, sm['carState'].standstill = speed, False
  set_model(sm, length=length, accel=0.05)
  step(planner, sm)
  assert planner.anticipating == expected


@pytest.mark.parametrize('held,stop_bit', [(False, True), (True, False)])
def test_anticipation_requires_both_standstill_and_the_model_stop_bit(held, stop_bit):
  planner, sm = LongitudinalPlanner(car_params()), messages()
  sm['carState'].standstill = held
  sm['carState'].vEgo = 0.0 if held else 0.2
  set_model(sm, length=25.0, accel=0.05, should_stop=stop_bit)
  step(planner, sm, frames=20)
  assert not planner.anticipating


def source_launch_floor(md, speed, action_t):
  # Oracle from better-green-lights cec0e040: interpolate twice, first strictly >1.2,
  # shift by at most 3.5 s, get_accel_from_plan, cap 1.5 and fade to zero at 2 m/s.
  v = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, md.velocity.x)
  a = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, md.acceleration.x)
  if np.interp(3.5, T_IDXS_MPC, v) <= 2.0:
    return md.action.desiredAcceleration
  cut = min(float(T_IDXS_MPC[np.argmax(v > 1.2)]), 3.5)
  target = get_accel_from_plan(np.interp(T_IDXS_MPC + cut, T_IDXS_MPC, v),
                               np.interp(T_IDXS_MPC + cut, T_IDXS_MPC, a), T_IDXS_MPC, action_t=action_t)
  return max(md.action.desiredAcceleration, min(target, np.interp(speed, [1.2, 2.0], [1.5, 0.0])))


@pytest.mark.parametrize('speed', [0.0, 0.8, 1.2, 1.6, 1.99, 2.0])
@pytest.mark.parametrize('slope', [1.0, 2.0])
def test_source_time_shift_parity_cap_and_fade_reach_the_wire(speed, slope):
  planner, sm = LongitudinalPlanner(car_params(), init_a=1.5), messages()
  set_model(sm, length=40.0, accel=0.05, should_stop=False, moving=True)
  md = sm['modelV2']
  md.velocity.x = (np.asarray(list(md.velocity.x)) * slope).tolist()
  md.acceleration.x = (np.asarray(list(md.acceleration.x)) * slope).tolist()
  step(planner, sm)
  sm['carState'].vEgo, sm['carState'].standstill = speed, speed == 0.0
  expected = source_launch_floor(md, sm['carState'].vEgo, planner.CP.longitudinalActuatorDelay + planner.dt)
  plan = step(planner, sm, frames=30)
  assert plan.aTarget == pytest.approx(expected, abs=1e-6)
  assert str(plan.longitudinalPlanSource) == 'e2e'
  assert not plan.shouldStop


@pytest.mark.parametrize('commit_speed', [1.999, 2.0, 2.001])
def test_committed_motion_requires_strictly_more_than_two_at_three_point_five_seconds(commit_speed):
  planner, sm = LongitudinalPlanner(car_params(), init_a=1.5), messages()
  set_model(sm, length=40.0, accel=0.05, should_stop=False, moving=True)
  times = np.asarray(ModelConstants.T_IDXS)
  sm['modelV2'].velocity.x = np.minimum(2.0 * np.maximum(times - 1.0, 0.0), commit_speed).tolist()
  sm['modelV2'].acceleration.x = np.zeros_like(times).tolist()
  expected = source_launch_floor(sm['modelV2'], 0.0, planner.CP.longitudinalActuatorDelay + planner.dt)
  plan = step(planner, sm, frames=30)
  assert plan.aTarget == pytest.approx(expected)
  assert (plan.aTarget > 0.1) == (commit_speed > 2.0)


@pytest.mark.parametrize('slot', ['leadOne', 'leadTwo'])
def test_launch_assist_cannot_override_close_lead_braking_or_fcw(slot):
  planner, sm = LongitudinalPlanner(car_params(), init_a=1.5), messages()
  set_model(sm, length=40.0, accel=0.05, should_stop=False, moving=True)
  step(planner, sm, frames=5)
  sm['carState'].vEgo, sm['carState'].standstill = 1.0, False
  set_lead(sm, 'leadOne', 50.0)
  set_lead(sm, slot, 0.05)
  plan = step(planner, sm, frames=10)
  assert planner.launch_armed
  # Preserve the exact native minimum, not an invented minimum braking magnitude.
  assert plan.aTarget < 0.0
  assert plan.aTarget == pytest.approx(planner.mpc_a_target)
  assert planner.stop_landing.required_decel > 3.0
  # Daily's FCW is leadOne-owned; leadTwo still participates in braking/landing.
  assert (planner.mpc.crash_cnt > 2) == (slot == 'leadOne')
  assert plan.fcw == (slot == 'leadOne')


@pytest.mark.parametrize('speed,standstill', [(2.0, False), (2.01, False), (2.01, True)])
def test_acceleration_floor_is_gone_at_disarm_speed_even_for_a_negative_model_target(speed, standstill):
  planner, sm = LongitudinalPlanner(car_params(), init_a=1.5), messages()
  set_model(sm, length=40.0, accel=0.05, should_stop=False, moving=True)
  step(planner, sm)
  sm['carState'].vEgo, sm['carState'].standstill = speed, standstill
  sm['modelV2'].action.desiredAcceleration = -0.05
  plan = step(planner, sm, frames=15)
  assert plan.aTarget == pytest.approx(-0.05)
  if speed > 2.0:
    assert not planner.launch_armed


@pytest.mark.parametrize('manual', [False, True])
def test_real_effective_manual_or_cem_mode_drives_the_same_green_launch(manual):
  from openpilot.selfdrive.controls.lib.conditional_experimental_mode import ConditionalExperimentalMode
  from openpilot.selfdrive.selfdrived.selfdrived import SelfdriveD

  planner, sm, control = LongitudinalPlanner(car_params()), messages(), LongControl(car_params())
  sm.updated = dict.fromkeys(sm, True)
  sm.valid = dict.fromkeys(sm, True)
  sm.alive = dict.fromkeys(sm, True)
  sm.freq_ok = dict.fromkeys(sm, True)
  # The small input double deliberately avoids constructing live messaging sockets.
  sd = cast(SelfdriveD, SimpleNamespace(CP=planner.CP, enabled=True, sm=sm, manual_experimental_mode=manual,
                                      conditional_experimental_mode=ConditionalExperimentalMode(control_dt=planner.dt)))
  for i in range(20):
    SelfdriveD.update_experimental_mode(sd, sm['carState'])
    if i == 0:
      assert not sd.conditional_experimental_mode.experimental_mode
      assert sd.experimental_mode == manual
    sm['selfdriveState'].experimentalMode = sd.experimental_mode
    step(planner, sm, control)
  assert sd.conditional_experimental_mode.experimental_mode
  assert control.long_control_state == LongCtrlState.stopping
  set_model(sm, length=25.0, accel=0.05, moving=True)
  for _ in range(15):
    SelfdriveD.update_experimental_mode(sd, sm['carState'])
    sm['selfdriveState'].experimentalMode = sd.experimental_mode
    plan = step(planner, sm, control)
  assert planner.anticipating
  assert not plan.shouldStop and plan.aTarget > 0.0
  assert control.long_control_state == LongCtrlState.pid

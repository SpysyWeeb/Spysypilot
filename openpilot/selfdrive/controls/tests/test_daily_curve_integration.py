"""Exercise the real daily planner/MPC and serialized outputs, not a policy-only min()."""
import numpy as np
import pytest

import openpilot.cereal.messaging as messaging
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.lib.model_curve_speed import ModelCurveSpeedLimiter
from openpilot.selfdrive.controls.tests.test_model_curve_speed import make_cp, make_model, settle, tracking
from openpilot.selfdrive.controls.tests.test_daily_blotv3_integration import PlannerMessages, car_params
from openpilot.selfdrive.modeld.constants import ModelConstants


class Publisher:
  def __init__(self):
    self.messages = {}

  def send(self, service, message):
    self.messages[service] = messaging.log_from_bytes(message.to_bytes())


def curve_messages(curvature=0.02, v_ego=20.0, experimental=False, e2e_accel=0.0):
  sm = PlannerMessages(v_ego=v_ego, experimental=experimental, e2e_accel=e2e_accel)
  sm['carControl'].latActive = True
  sm['carControl'].orientationNED = [0.0, 0.0, 0.0]
  st = sm['controlsState'].lateralControlState.init('torqueState')
  st.active = True
  st.output = 0.3
  st.actualLateralAccel = 1.0
  st.desiredLateralAccel = 1.0
  set_path(sm, curvature, v_ego)
  return sm


def set_path(sm, curvature, v_ego):
  times = np.asarray(ModelConstants.T_IDXS)
  md = sm['modelV2']
  md.position.x = (times * v_ego).tolist()
  md.position.y = np.zeros_like(times).tolist()
  md.velocity.x = np.full_like(times, v_ego).tolist()
  md.orientationRate.z = np.full_like(times, v_ego * curvature).tolist()


def run_plan(sm, frames=30):
  planner = LongitudinalPlanner(car_params(), init_v=sm['carState'].vEgo)
  for _ in range(frames):
    planner.update(sm)
  pm = Publisher()
  planner.publish(sm, pm)
  return planner, pm.messages


def test_curve_lowers_the_actual_published_daily_target_without_changing_cruise_or_stop():
  straight, straight_messages = run_plan(curve_messages(0.0))
  sm = curve_messages()
  curve, messages = run_plan(sm)
  target = messages['longitudinalPlan'].longitudinalPlan
  assert target.aTarget < straight_messages['longitudinalPlan'].longitudinalPlan.aTarget - 0.5
  assert target.aTarget <= -1.0
  assert str(target.longitudinalPlanSource) == 'curve'
  assert not target.shouldStop
  assert sm['carState'].vCruise == 100.0
  assert curve.plan_winner == 'curve'
  assert straight.plan_winner == 'cruise'


@pytest.mark.parametrize('roll', [float('nan'), float('inf'), -float('inf'), None, 'invalid'])
def test_invalid_auxiliary_roll_keeps_limiter_protection_and_neutral_roll_recovery(roll):
  limiter, neutral = ModelCurveSpeedLimiter(make_cp()), ModelCurveSpeedLimiter(make_cp())
  bend = make_model(20.0, np.full(ModelConstants.IDX_N, 0.03))
  kwargs = {'v_ego': 20.0, 'a_ego': -2.0, 'lateral_active': True, 'lateral_state': tracking()}
  for policy in (limiter, neutral):
    result = settle(policy, bend, 40, **kwargs)
    assert result.a_target == -2.0
  for _ in range(40):
    result = limiter.update(bend, roll=roll, **kwargs)
    assert result == neutral.update(bend, roll=0.0, **kwargs)
    assert result.a_target is not None and np.isfinite(result.a_target) and result.a_target <= -1.0
  # A bad roll on the first straight frame is still a usable path frame. It must
  # advance the median/release ramp, not disappear then revive the old -2 candidate.
  straight = make_model(20.0)
  assert limiter.update(straight, roll=roll, **kwargs) == neutral.update(straight, roll=0.0, **kwargs)
  recovered = limiter.update(straight, roll=0.0, **kwargs)
  assert recovered == neutral.update(straight, roll=0.0, **kwargs)
  assert recovered.a_target is not None and recovered.a_target > -2.0
  for _ in range(50):
    recovered = limiter.update(straight, **kwargs)
    assert recovered == neutral.update(straight, **kwargs)
  assert recovered.a_target is None


@pytest.mark.parametrize('roll', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_roll_keeps_actual_planner_curve_protection_and_straight_recovery(roll):
  sm, neutral_sm = curve_messages(0.03), curve_messages(0.03)
  planner, _ = run_plan(sm, 40)
  neutral, _ = run_plan(neutral_sm, 40)
  sm['vehicleParameters'].roll = roll
  for _ in range(40):
    planner.update(sm)
    neutral.update(neutral_sm)
    assert planner.curve == neutral.curve
    assert planner.curve.a_target is not None and planner.curve.a_target <= -1.0
    assert planner.plan_winner == 'curve'
    assert planner.output_a_target == neutral.output_a_target
    assert planner.output_a_target <= -1.0
  set_path(sm, 0.0, 20.0)
  set_path(neutral_sm, 0.0, 20.0)
  planner.update(sm)
  neutral.update(neutral_sm)
  assert planner.curve == neutral.curve
  assert planner.curve.a_target is not None
  sm['vehicleParameters'].roll = 0.0
  planner.update(sm)
  neutral.update(neutral_sm)
  assert planner.curve == neutral.curve
  assert planner.curve.a_target is not None and planner.curve.a_target > -2.0
  for _ in range(50):
    planner.update(sm)
    neutral.update(neutral_sm)
    assert planner.curve == neutral.curve
    assert planner.output_a_target == neutral.output_a_target
  assert planner.curve.a_target is None and planner.plan_winner == 'cruise'
  pm = Publisher()
  planner.publish(sm, pm)
  assert np.isfinite(pm.messages['longitudinalPlan'].longitudinalPlan.aTarget)
  assert not pm.messages['curvePolicyState'].curvePolicyState.active


def test_auxiliary_roll_fallback_does_not_learn_authority_from_an_unknown_bank():
  limiter = ModelCurveSpeedLimiter(make_cp())
  bend = make_model(20.0, np.full(ModelConstants.IDX_N, 0.03))
  state = tracking(torque=0.8, error=0.05, lat=3.0)
  kwargs = {'v_ego': 20.0, 'a_ego': -2.0, 'lateral_active': True, 'lateral_state': state}
  settle(limiter, bend, 40, roll=0.05, **kwargs)
  authority = limiter.authority.x
  for _ in range(40):
    result = limiter.update(bend, roll=float('nan'), **kwargs)
    assert result.a_target is not None and result.a_target <= -1.0
    assert limiter.authority.x == authority
  limiter.update(bend, roll=0.0, **kwargs)
  assert limiter.authority.x != authority  # real bank measurements still permit learning


@pytest.mark.parametrize('field', ['v_ego', 'a_ego', 'accel_coast'])
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf'), None, 'invalid'])
@pytest.mark.parametrize('braking', [False, True])
def test_abandoned_scalar_frame_clears_dynamic_curve_state_before_straight_recovery(field, bad, braking):
  limiter = ModelCurveSpeedLimiter(make_cp())
  bend = make_model(20.0, np.full(ModelConstants.IDX_N, 0.03))
  state = tracking(torque=1.0, error=1.0, lat=2.0, desired=3.0) if braking else tracking()
  kwargs = {'v_ego': 20.0, 'a_ego': -2.0, 'accel_coast': -0.3, 'lateral_active': True, 'lateral_state': state}
  assert settle(limiter, bend, 40, **kwargs).a_target == -2.0
  authority = limiter.authority.x
  result = limiter.update(bend, **{**kwargs, field: bad})
  assert result.a_target is None
  assert not limiter.active and limiter.regime == 'free'
  assert limiter.authority.x == authority  # car calibration survives; frame dynamics do not
  recovered = limiter.update(make_model(20.0), **{**kwargs, 'lateral_state': tracking()})
  assert recovered.a_target is None and recovered.regime == 'free' and not recovered.holding


def test_invalid_model_cannot_keep_a_stale_curve_candidate_or_break_handover():
  sm = curve_messages()
  planner, _ = run_plan(sm)
  previous = planner.output_a_target
  sm.invalid.add('modelV2')
  planner.update(sm)
  assert planner.plan_winner == 'cruise'
  assert planner.curve.a_target is None
  assert 0.0 <= planner.output_a_target - previous <= 0.1
  assert not planner.output_should_stop
  # Fresh straight data must not revive the old curve's median or jerk anchor.
  sm.invalid.clear()
  set_path(sm, 0.0, 20.0)
  planner.update(sm)
  assert planner.curve.a_target is None


@pytest.mark.parametrize('broken', ['shape', 'nonfinite'])
def test_malformed_path_releases_stale_anticipation_but_live_understeer_still_brakes(broken):
  sm = curve_messages()
  planner, _ = run_plan(sm)
  sm['modelV2'].orientationRate.z = [] if broken == 'shape' else [float('nan')] * ModelConstants.IDX_N
  previous = planner.output_a_target
  planner.update(sm)
  assert planner.curve.a_target is None
  assert planner.plan_winner == 'cruise'
  assert 0.0 <= planner.output_a_target - previous <= 0.1
  # A fresh measured loss of steering authority must still react without a usable path.
  st = sm['controlsState'].lateralControlState.torqueState
  st.output = 1.0
  st.desiredLateralAccel = 3.0
  st.actualLateralAccel = 2.0
  st.error = 1.0
  for _ in range(30):
    planner.update(sm)
  assert planner.curve.regime == 'brake'
  assert planner.curve.a_target is not None and planner.curve.a_target <= -1.0


def test_disengaged_curve_telemetry_does_not_claim_an_authoritative_candidate():
  from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
  sm = curve_messages()
  planner, _ = run_plan(sm)
  sm['controlsState'].longControlState = LongCtrlState.off
  sm['selfdriveState'].enabled = False
  planner.update(sm)
  pm = Publisher()
  planner.publish(sm, pm)
  assert not pm.messages['curvePolicyState'].curvePolicyState.active
  assert planner.plan_winner != 'curve'


@pytest.mark.parametrize('owner', ['profile', 'e2e', 'lead'])
def test_curve_never_eases_a_more_urgent_stop_request(monkeypatch, owner):
  from openpilot.selfdrive.controls.lib.force_stops import ForceStopsResult
  from openpilot.selfdrive.controls.tests.test_daily_blotv3_integration import set_lead
  sm = curve_messages(experimental=owner == 'e2e', e2e_accel=-3.0)
  planner = LongitudinalPlanner(car_params(), init_v=20.0)
  if owner == 'profile':
    monkeypatch.setattr(planner.force_stops, 'update', lambda *args: ForceStopsResult(a_target=-3.0))
  elif owner == 'lead':
    set_lead(sm, 'leadOne', 5.0)
  for _ in range(30):
    planner.update(sm)
  pm = Publisher()
  planner.publish(sm, pm)
  target = pm.messages['longitudinalPlan'].longitudinalPlan
  assert planner.curve.a_target is not None
  assert target.aTarget < planner.curve.a_target
  assert target.aTarget <= -3.0
  assert str(target.longitudinalPlanSource) == {'profile': 'stop', 'e2e': 'e2e', 'lead': 'lead0'}[owner]
  if owner == 'lead':
    assert target.fcw


def test_optional_torque_service_does_not_invalidate_plans_and_needs_its_own_health():
  from openpilot.selfdrive.controls import plannerd
  from openpilot.selfdrive.controls.lib.longitudinal_planner import get_live_torque_params
  sm = plannerd.get_submaster()
  healthy = curve_messages()
  for service, payload in healthy.items():
    sm.data[service] = payload
    sm.alive[service] = sm.freq_ok[service] = sm.valid[service] = True
  optional = 'lateralTorqueParameters'
  assert optional in sm.services
  sm.data[optional] = messaging.new_message(optional).lateralTorqueParameters
  sm[optional].useParams = True
  sm[optional].latAccelFactorFiltered = 2.7
  sm[optional].frictionCoefficientFiltered = 0.11
  for health in ('alive', 'freq_ok', 'valid'):
    sm.alive[optional] = sm.freq_ok[optional] = sm.valid[optional] = True
    getattr(sm, health)[optional] = False
    assert sm.all_checks()
    assert get_live_torque_params(sm) is None
    planner = LongitudinalPlanner(car_params(), init_v=20.0)
    planner.update(sm)
    pm = Publisher()
    planner.publish(sm, pm)
    assert pm.messages['longitudinalPlan'].valid
  sm.alive[optional] = sm.freq_ok[optional] = sm.valid[optional] = True
  assert get_live_torque_params(sm) is sm[optional]


def test_curve_service_schema_and_replay_contract_preserves_daily_abi():
  from openpilot.cereal import custom, log
  from openpilot.cereal.services import SERVICE_LIST
  from openpilot.selfdrive.test.process_replay.process_replay import CONFIGS
  source = log.LongitudinalPlan.LongitudinalPlanSource
  assert source.stop == 5
  assert source.curve == 6
  assert log.Event.schema.fields['spysydriveStateSP'].proto.ordinal.explicit == 107
  assert log.Event.schema.fields['curvePolicyState'].proto.ordinal.explicit == 108
  assert custom.CurvePolicyState.schema.node.id == 0xaedffd8f31e7b55d
  assert SERVICE_LIST['curvePolicyState'].frequency == 20.0
  assert SERVICE_LIST['curvePolicyState'].should_log
  replay = next(c for c in CONFIGS if c.proc_name == 'plannerd')
  assert 'curvePolicyState' in replay.subs
  assert 'lateralTorqueParameters' in replay.pubs


def test_gas_override_preserves_reaction_then_resumes_without_stale_reset():
  from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
  sm = curve_messages(curvature=0.0)
  st = sm['controlsState'].lateralControlState.torqueState
  st.output = 1.0
  st.desiredLateralAccel = 3.0
  st.actualLateralAccel = 2.0
  st.error = 1.0
  planner, _ = run_plan(sm)
  assert planner.curve.regime == 'brake'
  authority = planner.curve.authority_factor
  sm['controlsState'].longControlState = LongCtrlState.off
  sm['carState'].gasPressed = True
  for _ in range(12):
    planner.update(sm)
    assert planner.curve_speed_limiter.regime == 'brake'
    assert planner.curve.a_target is None  # not admitted during override
  sm['controlsState'].longControlState = LongCtrlState.pid
  sm['carState'].gasPressed = False
  planner.update(sm)
  assert planner.curve.regime == 'brake'
  assert planner.curve.authority_factor == authority
  assert planner.output_a_target < 0.0


def test_a_curve_release_hands_cruise_the_published_target_not_its_hidden_ramp():
  sm = curve_messages()
  planner, _ = run_plan(sm)
  assert planner.output_a_target <= -1.0
  set_path(sm, 0.0, 20.0)
  sm['carState'].steeringPressed = True  # remove authority immediately once the spatial median clears
  for _ in range(10):
    previous = planner.output_a_target
    planner.update(sm)
    assert 0.0 <= planner.output_a_target - previous <= 0.151
  assert planner.plan_winner == 'cruise'
  assert not planner.output_should_stop

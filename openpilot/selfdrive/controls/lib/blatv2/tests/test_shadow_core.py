import gc
import math
import tracemalloc
from types import SimpleNamespace

import openpilot.cereal.messaging as messaging
from openpilot.selfdrive.controls.lib.blatv2.controller import ControllerParams
from openpilot.selfdrive.controls.lib.blatv2.plant import PlantParams, PlantState
from openpilot.selfdrive.controls.lib.blatv2.reference import (
  horizon,
  model_action_time,
)
from openpilot.selfdrive.controls.lib.blatv2.shadow import ShadowCore
from openpilot.selfdrive.controls.blatv2_shadowd import populate_shadow_message
from openpilot.selfdrive.controls.lib.blatv2.v14_shadow import V14ShadowResult


def plant_params() -> PlantParams:
  return PlantParams(
    4000.0,
    10.0,
    0.09,
    0.12,
    409,
    4,
    7,
    1,
    True,
    (2.5, 5.5, 8.5, 12.0, 16.5, 21.0),
    (0.85, 0.39, 0.38, 0.36, 0.286, 0.288),
  )


def namespace(**kwargs):
  return SimpleNamespace(**kwargs)


def model(scalar: float = 0.01):
  return namespace(
    action=namespace(desiredCurvature=scalar),
    orientationRate=namespace(t=(0.0, 0.2, 0.4, 0.8), z=(0.1, 0.1, 0.1, 0.1)),
    velocity=namespace(t=(0.0, 0.2, 0.4, 0.8), x=(10.0, 10.0, 10.0, 10.0)),
  )


def car_params():
  return namespace(
    mass=2000.0,
    wheelbase=3.0,
    centerToFront=1.2,
    tireStiffnessFront=100000.0,
    tireStiffnessRear=110000.0,
    steerRatio=15.0,
    steerRatioRear=0.0,
  )


def controller_params():
  return ControllerParams(0.05, 0.01, 0.5, 0.5, True)


def car_control(torque: float = -0.1):
  return namespace(latActive=True, actuators=namespace(torque=torque))


def result_values(result):
  return (
    result.valid,
    result.reference_curvature,
    result.torque_demand,
    result.feasible_torque,
    result.plant_residual,
    result.scalar_plan_disagreement,
    result.horizon,
    result.v_ego,
    result.aligning_torque,
    result.align_inputs_valid,
    result.disturbance_estimate,
    result.observer_status,
    result.observer_unconstrained_update,
    result.mpc_command_torque,
    result.mpc_status,
    result.mpc_candidate_count,
    result.mpc_available_schedule_count,
    result.mpc_optimality_residual,
    result.fallback_command_torque,
    result.fallback_status,
    result.fallback_candidate_count,
    result.fallback_optimality_residual,
  )


def test_shared_core_is_deterministic_and_residual_valid_after_bootstrap():
  params = plant_params()
  torque = namespace(latAccelFactor=2.5, latAccelOffset=0.0, friction=0.1)
  car_state = namespace(vEgo=10.0, steeringAngleDeg=2.0, steeringRateDeg=0.5, steeringPressed=False, standstill=False)
  car_output = namespace(actuatorsOutput=namespace(torque=-0.1))
  live = namespace(roll=0.01, angleOffsetDeg=0.2, stiffnessFactor=0.9, steerRatio=15.5)

  left = ShadowCore(params, torque, car_params(), controller_params())
  right = ShadowCore(params, torque, car_params(), controller_params())
  left_first_valid = left.compute(model(), car_state, car_control(), car_output, live, True, 0.12, True).valid
  left_result = left.compute(model(), car_state, car_control(), car_output, live, True, 0.12, True)
  left_values = result_values(left_result)
  right.compute(model(), car_state, car_control(), car_output, live, True, 0.12, True)
  right_values = result_values(right.compute(model(), car_state, car_control(), car_output, live, True, 0.12, True))

  assert left_values == right_values
  assert not left_first_valid
  assert left_result.valid
  assert all(math.isfinite(value) for value in (
    left_result.reference_curvature,
    left_result.torque_demand,
    left_result.feasible_torque,
    left_result.plant_residual,
    left_result.scalar_plan_disagreement,
    left_result.horizon,
    left_result.v_ego,
    left_result.aligning_torque,
    left_result.disturbance_estimate,
    left_result.observer_unconstrained_update,
    left_result.mpc_command_torque,
    left_result.mpc_optimality_residual,
    left_result.fallback_command_torque,
    left_result.fallback_optimality_residual,
  ))


def test_learned_reference_lag_never_becomes_rack_prediction_delay():
  params = plant_params()
  torque = namespace(
    latAccelFactor=2.5, latAccelOffset=0.0, friction=0.1,
  )
  car_state = namespace(
    vEgo=10.0,
    steeringAngleDeg=2.0,
    steeringRateDeg=0.5,
    steeringPressed=False,
    standstill=False,
  )
  car_output = namespace(actuatorsOutput=namespace(torque=-0.1))
  live = namespace(
    roll=0.01,
    angleOffsetDeg=0.2,
    stiffnessFactor=0.9,
    steerRatio=15.5,
  )
  learned_end_to_end_lag = 0.40
  core = ShadowCore(params, torque, car_params(), controller_params())

  # Bootstrap the one-step residual, then inspect a valid controller frame.
  core.compute(
    model(), car_state, car_control(), car_output, live, True,
    learned_end_to_end_lag, True,
  )
  result = core.compute(
    model(), car_state, car_control(), car_output, live, True,
    learned_end_to_end_lag, True,
  )
  candidate = core.fallback.result

  assert result.valid
  assert core.reference_delay == learned_end_to_end_lag
  assert core.action_time == model_action_time(learned_end_to_end_lag)
  assert core.actuation_delay == params.actuation_delay
  assert result.horizon == horizon(params)
  assert candidate.action_time_seconds == model_action_time(
    learned_end_to_end_lag
  )
  assert candidate.prediction_delay_seconds == params.actuation_delay

  expected = PlantState(0.0, 0.0, 0.0, 0.0)
  core.twin.predict_held_state_into(
    PlantState(2.0, 0.5, -0.1, 10.0),
    params.actuation_delay,
    core.previous_align_inputs,
    result.disturbance_estimate,
    expected,
    0.05,
  )
  assert candidate.predicted_angle_deg == expected.angle_deg
  assert candidate.predicted_rate_deg_s == expected.rate_deg_s


def test_valid_core_result_serializes_at_capnp_publish_boundary():
  params = plant_params()
  torque = namespace(
    latAccelFactor=2.5,
    latAccelOffset=0.0,
    friction=0.1,
  )
  car_state = namespace(
    vEgo=10.0,
    steeringAngleDeg=2.0,
    steeringRateDeg=0.5,
    steeringPressed=False,
    standstill=False,
  )
  car_output = namespace(actuatorsOutput=namespace(torque=-0.1))
  live = namespace(
    roll=0.01,
    angleOffsetDeg=0.2,
    stiffnessFactor=0.9,
    steerRatio=15.5,
  )
  core = ShadowCore(params, torque, car_params(), controller_params())

  core.compute(
    model(), car_state, car_control(), car_output, live, True, 0.12, True,
  )
  result = core.compute(
    model(), car_state, car_control(), car_output, live, True, 0.12, True,
  )
  assert result.valid

  message = messaging.new_message("blatV2Shadow")
  populate_shadow_message(
    message,
    message.blatV2Shadow,
    result,
    log_mono_time_ns=123,
    message_valid=True,
    compute_seconds=0.001,
    shared_compute_seconds=0.0001,
    live_lqi_command_torque=-0.08,
    live_lqi_status=0,
    live_lqi_compute_seconds=0.0008,
    live_lqi_output_valid=True,
    live_lqi_invalid_frames=0,
    live_lqi_recovery_ok_frames=10,
    live_lqi_controller_version=207,
    live_action_raw_command_torque=-0.09,
    live_action_feedforward_torque=-0.04,
    live_action_feedback_torque=-0.05,
    live_action_desired_angle_deg=12.0,
    live_action_desired_rate_deg_s=3.0,
    live_action_desired_acceleration_deg_s2=4.0,
    live_action_predicted_angle_deg=10.0,
    live_action_predicted_rate_deg_s=2.0,
    live_action_required_acceleration_deg_s2=5.0,
    live_action_speed_mps=6.0,
    live_action_aligning_torque=-0.03,
    live_action_friction_torque=-0.02,
    live_action_dynamic_torque=-0.04,
    live_action_time_seconds=0.295,
    live_action_prediction_delay_seconds=0.12,
    live_action_slew_constrained=True,
    live_action_breakaway_active=True,
    live_action_breakaway_persistence_frames=5,
    live_action_horizon_assist_active=True,
    live_action_horizon_torque_demand=-0.8,
    live_action_horizon_demand_time_seconds=0.45,
    live_action_no_lead_limited=False,
    v14_result=V14ShadowResult(-0.07, 0.01, True, 14),
    v14_compute_seconds=0.0002,
  )
  serialized = message.to_bytes()

  assert serialized
  assert message.blatV2Shadow.sharedComputeTimeSeconds == 0.0001
  assert message.blatV2Shadow.liveLqiCommandTorque == -0.08
  assert message.blatV2Shadow.liveLqiComputeTimeSeconds == 0.0008
  assert message.blatV2Shadow.liveLqiControllerVersion == 207
  assert message.blatV2Shadow.liveActionRawCommandTorque == -0.09
  assert message.blatV2Shadow.liveActionDesiredAngleDeg == 12.0
  assert message.blatV2Shadow.liveActionPredictionDelaySeconds == 0.12
  assert message.blatV2Shadow.liveActionSlewConstrained
  assert message.blatV2Shadow.liveActionBreakawayActive
  assert message.blatV2Shadow.liveActionBreakawayPersistenceFrames == 5
  assert message.blatV2Shadow.liveActionHorizonAssistActive
  assert message.blatV2Shadow.liveActionHorizonTorqueDemand == -0.8
  assert message.blatV2Shadow.liveActionHorizonDemandTimeSeconds == 0.45
  assert not message.blatV2Shadow.liveActionNoLeadLimited
  assert message.blatV2Shadow.v14CommandTorque == -0.07
  assert message.blatV2Shadow.v14ControllerVersion == 14


def test_shared_core_has_no_unbounded_allocation_growth():
  params = plant_params()
  torque = namespace(latAccelFactor=2.5, latAccelOffset=0.0, friction=0.1)
  car_state = namespace(vEgo=10.0, steeringAngleDeg=2.0, steeringRateDeg=0.5, steeringPressed=False, standstill=False)
  car_output = namespace(actuatorsOutput=namespace(torque=-0.1))
  live = namespace(roll=0.01, angleOffsetDeg=0.2, stiffnessFactor=0.9, steerRatio=15.5)
  core = ShadowCore(params, torque, car_params(), controller_params())
  twin_identity = id(core.twin)
  result_identity = id(core.result)
  observer_identity = id(core.observer)
  mpc_identity = id(core.mpc)
  fallback_identity = id(core.fallback)
  mpc_workspace_identity = id(core.mpc.workspace)
  fallback_workspace_identity = id(core.fallback.workspace)
  shared_workspace_identity = id(core.candidate_workspace)

  tracemalloc.start()
  try:
    for _ in range(100):
      core.compute(model(), car_state, car_control(), car_output, live, True, 0.12, True)
    gc.collect()
    baseline_current, _ = tracemalloc.get_traced_memory()

    for _ in range(1_000):
      core.compute(model(), car_state, car_control(), car_output, live, True, 0.12, True)
    gc.collect()
    final_current, _ = tracemalloc.get_traced_memory()
  finally:
    tracemalloc.stop()

  assert final_current - baseline_current <= 4096
  assert id(core.twin) == twin_identity
  assert id(core.result) == result_identity
  assert id(core.observer) == observer_identity
  assert id(core.mpc) == mpc_identity
  assert id(core.fallback) == fallback_identity
  assert id(core.candidate_workspace) == shared_workspace_identity
  assert id(core.mpc.workspace) == mpc_workspace_identity == shared_workspace_identity
  assert id(core.fallback.workspace) == fallback_workspace_identity == shared_workspace_identity


def test_invalid_live_parameters_use_zero_inputs_without_stale_carryover():
  params = plant_params()
  torque = namespace(latAccelFactor=2.5, latAccelOffset=0.0, friction=0.1)
  car_state = namespace(vEgo=10.0, steeringAngleDeg=2.0, steeringRateDeg=0.5, steeringPressed=False, standstill=False)
  car_output = namespace(actuatorsOutput=namespace(torque=-0.1))
  live = namespace(roll=0.01, angleOffsetDeg=0.2, stiffnessFactor=0.9, steerRatio=15.5)
  core = ShadowCore(params, torque, car_params(), controller_params())

  assert core.compute(model(), car_state, car_control(), car_output, live, True, 0.12, True).align_inputs_valid
  core.compute(model(), car_state, car_control(), car_output, live, False, 0.12, True)
  fallback = core.compute(model(), car_state, car_control(), car_output, live, False, 0.12, True)
  fallback_demand = fallback.torque_demand
  fallback_aligning = fallback.aligning_torque

  fresh = ShadowCore(params, torque, car_params(), controller_params())
  fresh.compute(model(), car_state, car_control(), car_output, live, False, 0.12, True)
  expected = fresh.compute(model(), car_state, car_control(), car_output, live, False, 0.12, True)
  assert not fallback.align_inputs_valid
  assert fallback_demand == expected.torque_demand
  assert fallback_aligning == expected.aligning_torque

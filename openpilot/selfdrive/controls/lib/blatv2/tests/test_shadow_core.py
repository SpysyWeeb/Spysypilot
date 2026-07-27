import gc
import math
import tracemalloc
from types import SimpleNamespace

import openpilot.cereal.messaging as messaging
from openpilot.selfdrive.controls.lib.blatv2.controller import ControllerParams
from openpilot.selfdrive.controls.lib.blatv2.plant import PlantParams
from openpilot.selfdrive.controls.lib.blatv2.shadow import ShadowCore
from openpilot.selfdrive.controls.blatv2_shadowd import populate_shadow_message


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
    result.mpc_optimality_residual,
    result.fallback_command_torque,
    result.fallback_status,
    result.fallback_candidate_count,
    result.fallback_optimality_residual,
  )


def test_shared_core_is_deterministic_and_residual_valid_after_bootstrap():
  params = PlantParams(4000.0, 10.0, 0.05, 0.12, 409, 4, 7, 1, True)
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


def test_valid_core_result_serializes_at_capnp_publish_boundary():
  params = PlantParams(4000.0, 10.0, 0.05, 0.12, 409, 4, 7, 1, True)
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
    mpc_compute_seconds=0.0008,
    fallback_compute_seconds=0.0002,
  )
  serialized = message.to_bytes()

  assert serialized
  assert message.blatV2Shadow.sharedComputeTimeSeconds == 0.0001
  assert message.blatV2Shadow.mpcOptimalityResidual == float(
    result.mpc_optimality_residual
  )
  assert message.blatV2Shadow.fallbackOptimalityResidual == float(
    result.fallback_optimality_residual
  )


def test_shared_core_has_no_unbounded_allocation_growth():
  params = PlantParams(4000.0, 10.0, 0.05, 0.12, 409, 4, 7, 1, True)
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
  assert id(core.mpc.workspace) == mpc_workspace_identity
  assert id(core.fallback.workspace) == fallback_workspace_identity


def test_invalid_live_parameters_use_zero_inputs_without_stale_carryover():
  params = PlantParams(4000.0, 10.0, 0.05, 0.12, 409, 4, 7, 1, True)
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

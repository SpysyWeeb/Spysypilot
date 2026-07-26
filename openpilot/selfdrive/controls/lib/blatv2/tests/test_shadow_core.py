import gc
import math
import tracemalloc
from types import SimpleNamespace

from openpilot.selfdrive.controls.lib.blatv2.plant import PlantParams
from openpilot.selfdrive.controls.lib.blatv2.shadow import ShadowCore


def namespace(**kwargs):
  return SimpleNamespace(**kwargs)


def model(scalar: float = 0.01):
  return namespace(
    action=namespace(desiredCurvature=scalar),
    orientationRate=namespace(t=(0.0, 0.2, 0.4, 0.8), z=(0.1, 0.1, 0.1, 0.1)),
    velocity=namespace(t=(0.0, 0.2, 0.4, 0.8), x=(10.0, 10.0, 10.0, 10.0)),
  )


def test_shared_core_is_deterministic_and_residual_valid_after_bootstrap():
  params = PlantParams(4000.0, 10.0, 0.05, 0.12, 409, 4, 7, 1, True)
  torque = namespace(latAccelFactor=2.5, latAccelOffset=0.0, friction=0.1)
  car_state = namespace(vEgo=10.0, steeringAngleDeg=2.0, steeringRateDeg=0.5)
  car_output = namespace(actuatorsOutput=namespace(torque=-0.1))
  live = namespace(roll=0.01)

  left = ShadowCore(params, torque)
  right = ShadowCore(params, torque)
  left_results = [
    left.compute(model(), car_state, car_output, live, 0.12, True),
    left.compute(model(), car_state, car_output, live, 0.12, True),
  ]
  right_results = [
    right.compute(model(), car_state, car_output, live, 0.12, True),
    right.compute(model(), car_state, car_output, live, 0.12, True),
  ]

  assert left_results == right_results
  assert not left_results[0].valid
  assert left_results[1].valid
  assert all(math.isfinite(value) for value in (
    left_results[1].reference_curvature,
    left_results[1].torque_demand,
    left_results[1].feasible_torque,
    left_results[1].plant_residual,
    left_results[1].scalar_plan_disagreement,
    left_results[1].horizon,
  ))


def test_shared_core_has_no_unbounded_allocation_growth():
  params = PlantParams(4000.0, 10.0, 0.05, 0.12, 409, 4, 7, 1, True)
  torque = namespace(latAccelFactor=2.5, latAccelOffset=0.0, friction=0.1)
  car_state = namespace(vEgo=10.0, steeringAngleDeg=2.0, steeringRateDeg=0.5)
  car_output = namespace(actuatorsOutput=namespace(torque=-0.1))
  live = namespace(roll=0.01)
  core = ShadowCore(params, torque)

  tracemalloc.start()
  try:
    for _ in range(1_000):
      core.compute(model(), car_state, car_output, live, 0.12, True)
    gc.collect()
    baseline_current, _ = tracemalloc.get_traced_memory()

    for _ in range(10_000):
      core.compute(model(), car_state, car_output, live, 0.12, True)
    gc.collect()
    final_current, _ = tracemalloc.get_traced_memory()
  finally:
    tracemalloc.stop()

  assert final_current - baseline_current <= 4096

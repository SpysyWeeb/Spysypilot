from openpilot.selfdrive.controls.lib.blatv2.controller import ControllerParams, ObserverStatus
from openpilot.selfdrive.controls.lib.blatv2.observer import DisturbanceObserver
from openpilot.selfdrive.controls.lib.blatv2.plant import PlantParams


PLANT_PARAMS = PlantParams(
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
CONTROLLER_PARAMS = ControllerParams(0.05, 0.01, 0.5, 0.5, True)


def update(observer: DisturbanceObserver, **overrides) -> float:
  inputs = {
    "rate_residual": -0.2,
    "residual_valid": True,
    "lateral_active": True,
    "steering_pressed": False,
    "standstill": False,
    "model_valid": True,
    "recorded_constraint_active": False,
  }
  inputs.update(overrides)
  return observer.update(**inputs)


def test_observer_uses_recorded_response_and_respects_lifecycle():
  observer = DisturbanceObserver(PLANT_PARAMS, CONTROLLER_PARAMS)

  assert update(observer) == 0.0
  assert observer.status == ObserverStatus.RESET_ENGAGEMENT
  learned = update(observer)
  assert learned > 0.0
  assert observer.status == ObserverStatus.ACTIVE

  frozen = update(observer, recorded_constraint_active=True)
  assert frozen == learned
  assert observer.status == ObserverStatus.FROZEN_RECORDED_CONSTRAINT
  assert observer.unconstrained_update > frozen

  assert update(observer, steering_pressed=True) == 0.0
  assert observer.status == ObserverStatus.RESET_STEERING_PRESSED
  assert update(observer, steering_pressed=False) > 0.0
  assert update(observer, standstill=True) == 0.0
  assert observer.status == ObserverStatus.RESET_STANDSTILL
  assert update(observer, lateral_active=False) == 0.0
  assert observer.status == ObserverStatus.RESET_LATERAL_INVALID


def test_observer_is_bounded_by_the_shared_breakaway_envelope():
  observer = DisturbanceObserver(PLANT_PARAMS, CONTROLLER_PARAMS)
  update(observer)
  for _ in range(100):
    update(observer, rate_residual=-1e6)
  assert observer.estimate == PLANT_PARAMS.t_breakaway
  assert observer.unconstrained_update > observer.estimate

from types import SimpleNamespace

from openpilot.cereal import log
from openpilot.selfdrive.controls.controlsd import force_decel_requested


def selfdrive_state(state):
  return SimpleNamespace(state=state)


def driver_monitoring_state(force_decel=False):
  return SimpleNamespace(noResponseForceDecel=force_decel)


def test_force_decel_does_not_require_controls_state_subscription():
  # controlsd publishes controlsState itself; the emergency input must be
  # derived entirely from services already present in its SubMaster.
  assert not force_decel_requested(
    selfdrive_state(log.SelfdriveState.OpenpilotState.enabled),
    driver_monitoring_state(),
  )


def test_force_decel_preserves_driver_monitoring_override():
  assert force_decel_requested(
    selfdrive_state(log.SelfdriveState.OpenpilotState.enabled),
    driver_monitoring_state(True),
  )


def test_force_decel_preserves_soft_disable_override():
  assert force_decel_requested(
    selfdrive_state(log.SelfdriveState.OpenpilotState.softDisabling),
    driver_monitoring_state(),
  )

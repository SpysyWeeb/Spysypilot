"""Exact stock-controller construction at modular architecture boundaries."""

from __future__ import annotations

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.latcontrol_torque import (
  LatControlTorque,
)


def fresh_stock_torque_controller(CP, CI, live_torque_params=None):
  """Construct fully reset stock hidden state, optionally with one live tune."""
  controller = LatControlTorque(CP, CI, DT_CTRL)
  if live_torque_params is not None and live_torque_params.useParams:
    controller.update_live_torque_params(
      live_torque_params.latAccelFactorFiltered,
      live_torque_params.latAccelOffsetFiltered,
      live_torque_params.frictionCoefficientFiltered,
    )
  return controller

"""Exact stock steering-controller construction at modular boundaries.

"Exact stock" here means the unmodified :class:`LatControlTorque` algorithm
on the same reconstructed model-intent stream as every candidate. The detected
vehicle's runtime ``CarControllerParams`` owns its command envelope; replay
does not substitute a literal end-to-end stock planning stack or hard-code one
vehicle's limits.
"""

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

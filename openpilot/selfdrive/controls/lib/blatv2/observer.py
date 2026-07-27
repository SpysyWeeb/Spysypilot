"""Shared recorded-response disturbance observer for BLaTv2 A/B shadows.

The observer's input contract is deliberately one-way: recorded applied torque
and measured steering state produce one shared estimate; MPC and fallback may
read that estimate but can never write it. Candidate commands therefore cannot
pollute the other candidate's state.
"""

from __future__ import annotations

import math

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.blatv2.controller import ControllerParams, ObserverStatus
from openpilot.selfdrive.controls.lib.blatv2.plant import PlantParams


class DisturbanceObserver:
  def __init__(self, plant_params: PlantParams, controller_params: ControllerParams):
    self.plant_params = plant_params
    self.controller_params = controller_params
    self.estimate = 0.0
    self.unconstrained_update = 0.0
    self.status = ObserverStatus.RESET_LATERAL_INVALID
    self._previous_lateral_active = False

  def reset(self) -> None:
    self.estimate = 0.0
    self.unconstrained_update = 0.0
    self.status = ObserverStatus.RESET_LATERAL_INVALID
    self._previous_lateral_active = False

  def update(
    self,
    rate_residual: float,
    residual_valid: bool,
    lateral_active: bool,
    steering_pressed: bool,
    standstill: bool,
    model_valid: bool,
    recorded_constraint_active: bool,
  ) -> float:
    """Update from recorded response only, applying the pinned lifecycle.

    The inferred disturbance is ``-rate_residual / (k_t * DT_CTRL)`` in
    normalized-torque units. It is low-pass learned with the physical
    ``tau_disturbance`` and clamped to ``±t_breakaway``. This clamp is shared
    with the plant's identified/provisional friction envelope; it introduces no
    independent observer authority.
    """
    engaged_now = bool(lateral_active and not self._previous_lateral_active)
    self._previous_lateral_active = bool(lateral_active)

    if not lateral_active:
      status = ObserverStatus.RESET_LATERAL_INVALID
    elif steering_pressed:
      status = ObserverStatus.RESET_STEERING_PRESSED
    elif standstill:
      status = ObserverStatus.RESET_STANDSTILL
    elif engaged_now:
      status = ObserverStatus.RESET_ENGAGEMENT
    elif not model_valid or not residual_valid:
      status = ObserverStatus.RESET_MODEL_INVALID
    elif recorded_constraint_active:
      status = ObserverStatus.FROZEN_RECORDED_CONSTRAINT
    else:
      status = ObserverStatus.ACTIVE

    self.status = status
    if status.reset:
      self.estimate = 0.0
      self.unconstrained_update = 0.0
      return self.estimate

    residual = float(rate_residual)
    if not math.isfinite(residual):
      self.status = ObserverStatus.RESET_MODEL_INVALID
      self.estimate = 0.0
      self.unconstrained_update = 0.0
      return self.estimate

    inferred = -residual / (self.plant_params.k_t * DT_CTRL)
    alpha = DT_CTRL / (self.controller_params.tau_disturbance + DT_CTRL)
    unconstrained = self.estimate + alpha * (inferred - self.estimate)
    self.unconstrained_update = unconstrained
    if status.frozen:
      return self.estimate

    bound = self.plant_params.t_breakaway
    self.estimate = min(max(unconstrained, -bound), bound)
    return self.estimate

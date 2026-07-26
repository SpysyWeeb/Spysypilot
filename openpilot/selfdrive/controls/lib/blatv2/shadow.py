"""Shared deterministic numerical core for BLaTv2 shadow telemetry.

The on-device daemon and the route-audit harness import this file directly.
Environment measurements, including wall-clock compute time and messaging
transport health, deliberately live outside this core.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from openpilot.selfdrive.controls.lib.blatv2.plant import PlantParams, PlantState, PlantTwin
from openpilot.selfdrive.controls.lib.blatv2.reference import (
  build_reference,
  horizon,
  interpolate,
  model_action_time,
  plan_curvatures_from_model,
  torque_demand,
)


@dataclass(frozen=True)
class ShadowResult:
  valid: bool
  reference_curvature: float
  torque_demand: float
  feasible_torque: float
  plant_residual: float
  scalar_plan_disagreement: float
  horizon: float


class ShadowCore:
  """Stateful only for the one-step plant residual.

  Reference construction, inverse statics, and feasible-torque projection are
  pure functions of the current recorded frame. The prior frame is retained
  solely to evaluate the requested one-step plant residual.
  """

  def __init__(self, seed_params: PlantParams, torque_params: Any):
    self.seed_params = seed_params
    self.torque_params = torque_params
    self.previous_state: PlantState | None = None
    self.previous_applied_torque = 0.0

  def reset(self) -> None:
    self.previous_state = None
    self.previous_applied_torque = 0.0

  def frame_params(self, lateral_delay: float, lateral_delay_valid: bool) -> PlantParams:
    delay = float(lateral_delay)
    if lateral_delay_valid and math.isfinite(delay) and delay >= 0.0:
      return self.seed_params.with_actuation_delay(delay)
    return self.seed_params

  def compute(
    self,
    model: Any,
    car_state: Any,
    car_output: Any,
    live_parameters: Any,
    lateral_delay: float,
    lateral_delay_valid: bool,
  ) -> ShadowResult:
    frame_params = self.frame_params(lateral_delay, lateral_delay_valid)
    twin = PlantTwin(frame_params)

    scalar = float(model.action.desiredCurvature)
    plan_times, plan_curvatures, plan_valid = plan_curvatures_from_model(model, scalar)
    action_time = model_action_time(frame_params.actuation_delay)
    horizon_seconds = horizon(frame_params)
    ref_times, ref_curvatures = build_reference(
      scalar,
      plan_times,
      plan_curvatures,
      action_time,
      horizon_seconds,
    )
    reference_now_delay = interpolate(ref_times, ref_curvatures, action_time)
    plan_at_action = interpolate(plan_times, plan_curvatures, action_time)
    disagreement = plan_at_action - scalar

    demand = torque_demand(
      reference_now_delay,
      float(car_state.vEgo),
      float(live_parameters.roll),
      self.torque_params,
    )
    applied = float(car_output.actuatorsOutput.torque)
    feasible = twin.apply_slew(applied, demand)
    state = PlantState(
      angle_deg=float(car_state.steeringAngleDeg),
      rate_deg_s=float(car_state.steeringRateDeg),
      applied_torque=applied,
    )

    residual_valid = self.previous_state is not None
    residual = 0.0
    if self.previous_state is not None:
      residual = twin.one_step_residual(self.previous_state, self.previous_applied_torque, state)
    self.previous_state = state
    self.previous_applied_torque = applied

    values = (
      reference_now_delay,
      demand,
      feasible,
      residual,
      disagreement,
      horizon_seconds,
    )
    valid = bool(plan_valid and residual_valid and all(math.isfinite(value) for value in values))
    return ShadowResult(
      valid=valid,
      reference_curvature=float(reference_now_delay),
      torque_demand=float(demand),
      feasible_torque=float(feasible),
      plant_residual=float(residual),
      scalar_plan_disagreement=float(disagreement),
      horizon=float(horizon_seconds),
    )

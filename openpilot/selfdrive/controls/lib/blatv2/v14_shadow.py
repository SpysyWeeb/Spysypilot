"""Passive adapter around the byte-identical frozen-v14 controller pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.blatv2.v14.latcontrol_torque import (
  LatControlTorque,
  VERSION as V14_VERSION,
)
from openpilot.selfdrive.controls.lib.blatv2.v14.lateral_reference_planner import (
  ActuatorPreviewConfig,
  LateralReferencePlanner,
)
from openpilot.selfdrive.controls.lib.drive_helpers import clip_curvature
from openpilot.selfdrive.modeld.modeld import LAT_SMOOTH_SECONDS


@dataclass(slots=True)
class V14ShadowResult:
  command_torque: float = 0.0
  desired_curvature: float = 0.0
  valid: bool = False
  controller_version: int = V14_VERSION


class FrozenV14ShadowController:
  """Run the complete frozen-v14 planner/controller without actuation access.

  The controller and planner implementations are frozen source blobs, not
  ports.  This adapter mirrors their controlsd call boundary: model-plan
  updates feed the reference planner, the resulting curvature passes through
  ``clip_curvature``, and every reference/cascade/unwind/episode input reaches
  ``LatControlTorque.update`` unchanged.
  """

  def __init__(self, car_params: Any, car_interface: Any) -> None:
    self.CP = car_params
    self.CI = car_interface
    self.VM = VehicleModel(car_params)
    self.controller = LatControlTorque(car_params, car_interface, DT_CTRL)
    self.reference = LateralReferencePlanner(DT_CTRL)
    actuator = car_interface.CC.params
    self.reference.configure_actuator(
      ActuatorPreviewConfig(
        max_torque=actuator.STEER_MAX,
        delta_up=actuator.STEER_DELTA_UP,
        delta_down=actuator.STEER_DELTA_DOWN,
        steer_step=actuator.STEER_STEP,
      )
    )
    self.desired_curvature = 0.0
    self.measured_curvature = 0.0
    self.steer_limited_by_safety = False
    self.result = V14ShadowResult()

  def reset(self) -> None:
    self.controller.reset()
    self.reference.reset()
    self.desired_curvature = self.measured_curvature
    self.steer_limited_by_safety = False
    self.result.command_torque = 0.0
    self.result.desired_curvature = float(self.desired_curvature)
    self.result.valid = False

  def step(
    self,
    model: Any,
    model_valid: bool,
    model_updated: bool,
    car_state: Any,
    car_output: Any,
    lateral_active: bool,
    live_parameters: Any,
    live_torque_parameters: Any,
    live_torque_parameters_valid: bool,
    lateral_delay: float,
    lateral_maneuver_plan: Any,
    lateral_maneuver_plan_valid: bool,
  ) -> V14ShadowResult:
    stiffness = max(float(live_parameters.stiffnessFactor), 0.1)
    steer_ratio = max(float(live_parameters.steerRatio), 0.1)
    self.VM.update_params(stiffness, steer_ratio)
    steer_angle = math.radians(
      float(car_state.steeringAngleDeg)
      - float(live_parameters.angleOffsetDeg)
    )
    self.measured_curvature = -self.VM.calc_curvature(
      steer_angle, float(car_state.vEgo), float(live_parameters.roll),
    )

    if (
      live_torque_parameters_valid
      and bool(live_torque_parameters.useParams)
    ):
      self.controller.update_live_torque_params(
        float(live_torque_parameters.latAccelFactorFiltered),
        float(live_torque_parameters.latAccelOffsetFiltered),
        float(live_torque_parameters.frictionCoefficientFiltered),
      )

    if not lateral_active:
      self.controller.reset()

    applied_torque = float(car_output.actuatorsOutput.torque)
    delay = float(lateral_delay) + LAT_SMOOTH_SECONDS
    if lateral_maneuver_plan_valid:
      new_curvature = (
        float(lateral_maneuver_plan.desiredCurvature)
        if lateral_active else self.measured_curvature
      )
      self.reference.reset()
    else:
      raw_curvature = (
        float(model.action.desiredCurvature)
        if lateral_active else self.measured_curvature
      )
      if not lateral_active or not model_valid:
        self.reference.reset()
      elif model_updated:
        self.reference.update(
          model, self.measured_curvature, float(car_state.vEgo),
        )
      new_curvature = self.reference.get_curvature(
        raw_curvature,
        float(car_state.vEgo),
        delay,
        applied_torque,
        float(self.controller.torque_params.latAccelFactor),
        float(self.controller.torque_params.friction),
        float(live_parameters.roll),
        float(self.controller.torque_params.latAccelOffset),
      )

    self.desired_curvature, curvature_limited = clip_curvature(
      float(car_state.vEgo),
      self.desired_curvature,
      new_curvature,
      float(live_parameters.roll),
    )
    diagnostics = self.reference.diagnostics
    torque, _, _ = self.controller.update(
      lateral_active,
      car_state,
      self.VM,
      live_parameters,
      self.steer_limited_by_safety,
      self.desired_curvature,
      curvature_limited,
      delay,
      applied_torque,
      diagnostics.unwind_scale,
      diagnostics.target_torque,
      diagnostics.geometric_target_torque,
      diagnostics.episode_target_torque,
      diagnostics.output_curvature * float(car_state.vEgo) ** 2,
      diagnostics.episode_lateral_accel,
      (
        diagnostics.trajectory_curvature_rate
        if diagnostics.trajectory_rate_valid else None
      ),
    )
    command = float(torque)
    valid = bool(
      math.isfinite(command)
      and math.isfinite(self.desired_curvature)
      and math.isfinite(self.measured_curvature)
    )
    if lateral_active:
      self.steer_limited_by_safety = (
        abs(command - applied_torque) > 1e-2
      )
    else:
      self.steer_limited_by_safety = False

    result = self.result
    result.command_torque = command if valid else 0.0
    result.desired_curvature = float(self.desired_curvature)
    result.valid = valid
    result.controller_version = V14_VERSION
    return result

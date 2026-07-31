#!/usr/bin/env python3
"""Passive modular-BLaTv2 shadow runner for offline/harness validation.

The stock-only field manager does not launch this module. When invoked by
replay or a harness, it publishes telemetry only. It owns no actuator
publisher and the one-step actuator projection is diagnostic: the projected
torque is never fed back into :class:`ModularControllerCore`.

Model timing is consumed without controller-side reconstruction:

* ``modelV2.timestampEof`` is the native-plan origin;
* ``modelV2``'s event ``logMonoTime`` is publication time;
* ``modelV2.action.desiredCurvatureTime`` is the exact published scalar time;
* ``modelV2.orientationRate.t`` is the current published native grid; and
* the ``carState`` event ``logMonoTime`` is the sampled rack-state time; and
* a direct monotonic capture immediately before core computation is the
  controller witness.

No ``LAT_SMOOTH_SECONDS``, ``DT_MDL``, or ``liveDelay`` value participates in
the timing contract.
"""

from __future__ import annotations

import importlib
import inspect
import math
from pathlib import Path
import time
from typing import Any

from opendbc.car.structs import car
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.realtime import (
  DT_CTRL,
  Priority,
  config_realtime_process,
)
from openpilot.selfdrive.controls.lib.blatv2.actuator import (
  apply_torque_envelope,
)
from openpilot.selfdrive.controls.lib.blatv2.bootstrap import profile_sha256
from openpilot.selfdrive.controls.lib.blatv2.core import (
  CoreResult,
  ModularControllerCore,
)
from openpilot.selfdrive.controls.lib.blatv2.intent import (
  INTENT_CAPACITY,
  IntentAdaptation,
  IntentBuildStatus,
  adapt_model_intent_into,
)
from openpilot.selfdrive.controls.lib.blatv2.policy import ControllerPolicy
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import (
  RackMappingSnapshot,
)
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  ProvisionalRackDynamics,
  RuntimeVehicleBundle,
  build_runtime_vehicle_bundle,
)


MODULAR_SHADOW_SCHEMA_VERSION = 2
PUBLISHED_SERVICES = ("blatV2Shadow",)
SUBSCRIBED_SERVICES = (
  "modelV2",
  "carState",
  "carControl",
  "carOutput",
  "selfdriveState",
  "controlsState",
  "liveParameters",
)
PROVISIONAL_RACK_DYNAMICS_PATH = Path(__file__).resolve().parent / "lib" / "blatv2" / "provisional_rack_dynamics.json"
PROVISIONAL_POLICY_PATH = Path(__file__).resolve().parent / "lib" / "blatv2" / "provisional_controller_policy.json"
MAX_RACK_DERIVATIVE_GAP_S = 0.015
RECORDED_ACTUATOR_CONSTRAINT_TOLERANCE = 1e-2
_CONTROLLER_LIMIT_FIELDS = (
  "STEER_MAX",
  "STEER_DELTA_UP",
  "STEER_DELTA_DOWN",
  "STEER_STEP",
  "STEER_DRIVER_ALLOWANCE",
  "STEER_DRIVER_MULTIPLIER",
  "STEER_DRIVER_FACTOR",
)


def _control_witness_mono_ns() -> int:
  """Capture the same monotonic boot-time clock used by Event.logMonoTime."""
  clock_id = getattr(time, "CLOCK_BOOTTIME", time.CLOCK_MONOTONIC)
  return time.clock_gettime_ns(clock_id)


def assert_no_actuation_publishers(
  services: tuple[str, ...] = PUBLISHED_SERVICES,
) -> None:
  """Pin the process's structural inability to publish an actuator command."""
  assert services == ("blatV2Shadow",)
  assert "carControl" not in services


def _messaging_module() -> Any:
  # Import lazily so the deterministic numerical/publisher tests do not
  # require the native msgq extension. The managed process always has it.
  import openpilot.cereal.messaging as messaging

  return messaging


def _has_controller_limits(candidate: object) -> bool:
  return all(hasattr(candidate, name) for name in _CONTROLLER_LIMIT_FIELDS)


def controller_params_from_interface(
  car_interface: object,
  car_params: car.CarParams,
) -> object:
  """Construct the detected platform's controller parameters generically.

  Opendbc's detected ``CarInterface`` owns the production ``CarController``
  class. Its controller module imports the matching ``CarControllerParams``;
  resolving that class through the detected controller is the same dynamic
  platform selection used by ``interfaces[CP.carFingerprint]``. No platform
  package is imported or named here.
  """
  controller_class = getattr(car_interface, "CarController", None)
  if controller_class is None:
    raise RuntimeError("detected CarInterface has no CarController class")
  controller_module = importlib.import_module(controller_class.__module__)
  controller_params_class = getattr(
    controller_module,
    "CarControllerParams",
    None,
  )
  if controller_params_class is None:
    raise RuntimeError(
      "detected CarController does not expose CarControllerParams",
    )

  signature = inspect.signature(controller_params_class)
  positional = tuple(
    parameter
    for parameter in signature.parameters.values()
    if parameter.kind
    in (
      inspect.Parameter.POSITIONAL_ONLY,
      inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
  )
  if len(positional) == 0:
    controller_params = controller_params_class()
  elif len(positional) == 1:
    controller_params = controller_params_class(car_params)
  else:
    raise RuntimeError(
      "detected CarControllerParams has an unsupported constructor",
    )
  if not _has_controller_limits(controller_params):
    raise RuntimeError(
      "detected CarControllerParams lacks the torque-envelope contract",
    )
  return controller_params


class _RackAcceleration:
  """Timestamped measured-rate derivative with explicit gap invalidation."""

  __slots__ = ("_previous_time_s", "_previous_rate_deg_s")

  def __init__(self) -> None:
    self._previous_time_s: float | None = None
    self._previous_rate_deg_s = 0.0

  def reset(self) -> None:
    self._previous_time_s = None
    self._previous_rate_deg_s = 0.0

  def update(
    self,
    *,
    sample_mono_ns: int,
    rack_rate_deg_s: float,
    inputs_valid: bool,
  ) -> tuple[float, bool]:
    try:
      sample_time_s = int(sample_mono_ns) * 1e-9
      rate_deg_s = float(rack_rate_deg_s)
    except (TypeError, ValueError, OverflowError):
      self.reset()
      return 0.0, False
    if not inputs_valid or sample_mono_ns < 0 or not math.isfinite(sample_time_s) or not math.isfinite(rate_deg_s):
      self.reset()
      return 0.0, False

    acceleration = 0.0
    valid = False
    if self._previous_time_s is not None:
      dt_s = sample_time_s - self._previous_time_s
      valid = 0.0 < dt_s <= MAX_RACK_DERIVATIVE_GAP_S
      if valid:
        acceleration = (rate_deg_s - self._previous_rate_deg_s) / dt_s
    self._previous_time_s = sample_time_s
    self._previous_rate_deg_s = rate_deg_s
    return acceleration, valid


class _CanonicalModelSelector:
  """Resolve the exact model snapshot witnessed by controlsd.

  ``SubMaster`` drains its non-polled model socket after receiving the
  controlsState poll witness. The current model can therefore be one
  publication newer than the message controlsd consumed. controlsState's
  ``lateralPlanMonoTime`` is the exact model event timestamp; only an exact
  current or one-snapshot cached match is accepted.
  """

  __slots__ = (
    "_cached_message",
    "_cached_mono_ns",
    "_cached_valid",
    "_cached_alive",
    "selected_message",
    "selected_mono_ns",
    "selected_valid",
    "selected_alive",
  )

  def __init__(self) -> None:
    self._cached_message: Any | None = None
    self._cached_mono_ns = 0
    self._cached_valid = False
    self._cached_alive = False
    self.selected_message: Any | None = None
    self.selected_mono_ns = 0
    self.selected_valid = False
    self.selected_alive = False

  def select(
    self,
    *,
    controls_model_mono_ns: int,
    current_message: Any,
    current_mono_ns: int,
    current_valid: bool,
    current_alive: bool,
    current_available: bool,
  ) -> bool:
    target = int(controls_model_mono_ns)
    current_time = int(current_mono_ns)
    self.selected_message = None
    self.selected_mono_ns = 0
    self.selected_valid = False
    self.selected_alive = False

    matched = False
    if target > 0 and current_available and current_time == target:
      self.selected_message = current_message
      self.selected_mono_ns = current_time
      self.selected_valid = bool(current_valid)
      self.selected_alive = bool(current_alive)
      matched = True
    elif target > 0 and self._cached_message is not None and self._cached_mono_ns == target:
      self.selected_message = self._cached_message
      self.selected_mono_ns = self._cached_mono_ns
      self.selected_valid = self._cached_valid
      self.selected_alive = self._cached_alive
      matched = True

    if current_available and current_time > 0 and current_time != self._cached_mono_ns:
      # A model publication is only 20 Hz. One retained builder is enough to
      # resolve the known one-update race without retaining an arbitrary plan.
      self._cached_message = current_message.as_builder()
      self._cached_mono_ns = current_time
      self._cached_valid = bool(current_valid)
      self._cached_alive = bool(current_alive)
    return matched


class ModularShadowRunner:
  """Preconstructed numerical artifacts and reusable per-frame storage."""

  def __init__(
    self,
    *,
    car_params: car.CarParams,
    car_interface: object,
    controller_params: object,
    runtime_bundle: RuntimeVehicleBundle,
    policy: ControllerPolicy,
  ) -> None:
    self.car_params = car_params
    self.car_interface = car_interface
    self.controller_params = controller_params
    self.runtime_bundle = runtime_bundle
    self.policy = policy
    self.profile = runtime_bundle.seed_profile
    self.vehicle_model = VehicleModel(car_params)
    self.core = ModularControllerCore(
      fixed_dt_s=DT_CTRL,
      profile=self.profile,
      tracking_policy=policy.tracking_policy,
      observer_policy=policy.observer_policy,
      nominal_mapping=runtime_bundle.nominal_rack_mapping,
      plan_capacity=INTENT_CAPACITY,
    )

    self.plan_times_s = [0.0] * INTENT_CAPACITY
    self.orientation_rates_z = [0.0] * INTENT_CAPACITY
    self.velocities_x = [0.0] * INTENT_CAPACITY
    self.plan_curvatures = [0.0] * INTENT_CAPACITY
    self.rack_acceleration = _RackAcceleration()

    self.runtime_vehicle_identity_hash = runtime_bundle.identity_sha256
    self.policy_hash = policy.sha256
    self.profile_hash = profile_sha256(self.profile)
    self.intent_adaptation: IntentAdaptation | None = None
    self.core_result: CoreResult = self.core.result
    self.measured_acceleration_deg_s2 = 0.0
    self.measured_previous_applied_torque = 0.0
    self.measured_driver_torque = 0.0
    self.feasible_torque = 0.0
    self.unmet_torque = 0.0
    self.feasibility_constrained = False
    self.recorded_actuator_constrained = False
    self.live_parameters_valid = False
    self.model_input_valid = False
    self.vehicle_state_valid = False
    self.lateral_active = False
    self.lateral_valid = False
    self.profile_lower_node_speed_mps = self.profile.nodes[0].speed_mps
    self.profile_upper_node_speed_mps = self.profile.nodes[0].speed_mps
    self.profile_confidence = 0.0
    self.state_sample_mono_ns = 0
    self.control_witness_mono_ns = 0
    self.valid = False
    self._previous_lateral_active = False

  @classmethod
  def from_car_params(
    cls,
    car_params: car.CarParams,
  ) -> ModularShadowRunner:
    from opendbc.car.car_helpers import interfaces

    interface_class = interfaces[str(car_params.carFingerprint)]
    car_interface = interface_class(car_params)
    controller_params = controller_params_from_interface(
      car_interface,
      car_params,
    )
    dynamics = ProvisionalRackDynamics.from_json_file(
      PROVISIONAL_RACK_DYNAMICS_PATH,
    )
    policy = ControllerPolicy.from_json_file(PROVISIONAL_POLICY_PATH)
    runtime_bundle = build_runtime_vehicle_bundle(
      car_params=car_params,
      car_interface_or_callback=car_interface,
      controller_params=controller_params,
      vehicle_identity=str(car_params.carFingerprint),
      provisional_rack_dynamics=dynamics,
    )
    return cls(
      car_params=car_params,
      car_interface=car_interface,
      controller_params=controller_params,
      runtime_bundle=runtime_bundle,
      policy=policy,
    )

  def _live_mapping(
    self,
    live_parameters: Any,
    *,
    inputs_valid: bool,
  ) -> RackMappingSnapshot | None:
    self.live_parameters_valid = False
    if not inputs_valid:
      return None
    try:
      valid = (
        bool(live_parameters.valid)
        and bool(live_parameters.angleOffsetValid)
        and bool(live_parameters.steerRatioValid)
        and bool(live_parameters.stiffnessFactorValid)
      )
      if not valid:
        return None
      stiffness_factor = float(live_parameters.stiffnessFactor)
      steer_ratio = float(live_parameters.steerRatio)
      roll = float(live_parameters.roll)
      angle_offset = float(live_parameters.angleOffsetDeg)
      if not all(
        math.isfinite(value)
        for value in (
          stiffness_factor,
          steer_ratio,
          roll,
          angle_offset,
        )
      ):
        return None
      self.vehicle_model.update_params(
        max(stiffness_factor, 0.1),
        max(steer_ratio, 0.1),
      )
      mapping = RackMappingSnapshot.from_vehicle_model(
        self.vehicle_model,
        roll_rad=roll,
        angle_offset_deg=angle_offset,
        valid=True,
      )
    except (AttributeError, TypeError, ValueError, OverflowError):
      return None
    self.live_parameters_valid = True
    return mapping

  def _adapt_intent(
    self,
    *,
    state_sample_mono_ns: int,
    control_witness_mono_ns: int,
    model_publication_mono_ns: int,
    model_message: Any,
    model_message_valid: bool,
    model_message_alive: bool,
    current_v_ego_m_s: float,
    transport_delay_s: float,
  ) -> tuple[IntentAdaptation, float]:
    try:
      native_plan_times = model_message.orientationRate.t
      native_orientation_rates = model_message.orientationRate.z
      native_velocities = model_message.velocity.x
      velocity_times = model_message.velocity.t
      scalar_curvature = float(
        model_message.action.desiredCurvature,
      )
      same_native_grid = len(native_plan_times) == len(velocity_times) and all(
        float(left) == float(right)
        for left, right in zip(
          native_plan_times,
          velocity_times,
          strict=True,
        )
      )
      if not same_native_grid:
        native_velocities = ()
      adaptation = adapt_model_intent_into(
        state_sample_mono_ns=state_sample_mono_ns,
        control_witness_mono_ns=control_witness_mono_ns,
        model_publication_mono_ns=model_publication_mono_ns,
        plan_origin_mono_ns=int(model_message.timestampEof),
        model_frame_id=int(model_message.frameId),
        message_valid=model_message_valid,
        message_alive=model_message_alive,
        scalar_desired_curvature=scalar_curvature,
        published_desired_curvature_time_s=float(
          model_message.action.desiredCurvatureTime,
        ),
        native_plan_times_s=native_plan_times,
        native_orientation_rates_z=native_orientation_rates,
        native_velocities_x=native_velocities,
        current_v_ego_m_s=current_v_ego_m_s,
        physical_transport_delay_s=transport_delay_s,
        output_plan_times_s=self.plan_times_s,
        output_orientation_rates_z=self.orientation_rates_z,
        output_velocities_x=self.velocities_x,
        output_plan_curvatures=self.plan_curvatures,
      )
      return adaptation, scalar_curvature
    except (AttributeError, TypeError, ValueError, OverflowError):
      # The adapter clears every output buffer before inspecting inputs. This
      # second call pins the same no-stale-plan property when message field
      # extraction itself failed before the adapter could run.
      adaptation = adapt_model_intent_into(
        state_sample_mono_ns=int(state_sample_mono_ns),
        control_witness_mono_ns=int(control_witness_mono_ns),
        model_publication_mono_ns=0,
        plan_origin_mono_ns=0,
        model_frame_id=0,
        message_valid=False,
        message_alive=False,
        scalar_desired_curvature=math.nan,
        published_desired_curvature_time_s=0.0,
        native_plan_times_s=(),
        native_orientation_rates_z=(),
        native_velocities_x=(),
        current_v_ego_m_s=math.nan,
        physical_transport_delay_s=float(transport_delay_s),
        output_plan_times_s=self.plan_times_s,
        output_orientation_rates_z=self.orientation_rates_z,
        output_velocities_x=self.velocities_x,
        output_plan_curvatures=self.plan_curvatures,
      )
      return adaptation, math.nan

  def update(
    self,
    *,
    state_sample_mono_ns: int,
    control_witness_mono_ns: int,
    model_publication_mono_ns: int,
    model_message: Any,
    car_state: Any,
    car_control: Any,
    car_output: Any,
    selfdrive_state: Any,
    live_parameters: Any,
    model_message_valid: bool,
    model_message_alive: bool,
    vehicle_inputs_valid: bool,
    live_parameters_inputs_valid: bool,
  ) -> CoreResult:
    """Compute one passive frame from recorded response and model intent."""
    self.state_sample_mono_ns = int(state_sample_mono_ns)
    self.control_witness_mono_ns = int(control_witness_mono_ns)
    try:
      current_speed = float(car_state.vEgo)
      measured_angle = float(car_state.steeringAngleDeg)
      measured_rate = float(car_state.steeringRateDeg)
      applied_torque = float(car_output.actuatorsOutput.torque)
      requested_torque = float(car_control.actuators.torque)
      driver_torque = float(car_state.steeringTorque)
      steering_pressed = bool(car_state.steeringPressed)
      standstill = bool(car_state.standstill)
      lateral_active = bool(car_control.latActive)
      selfdrive_active = bool(selfdrive_state.active)
    except (AttributeError, TypeError, ValueError, OverflowError):
      current_speed = math.nan
      measured_angle = math.nan
      measured_rate = math.nan
      applied_torque = math.nan
      requested_torque = math.nan
      driver_torque = math.nan
      steering_pressed = False
      standstill = False
      lateral_active = False
      selfdrive_active = False
      vehicle_inputs_valid = False

    numeric_vehicle_inputs_valid = bool(vehicle_inputs_valid) and all(
      math.isfinite(value)
      for value in (
        current_speed,
        measured_angle,
        measured_rate,
        applied_torque,
        requested_torque,
        driver_torque,
      )
    )
    self.vehicle_state_valid = numeric_vehicle_inputs_valid and current_speed >= 0.0
    self.lateral_active = self.vehicle_state_valid and lateral_active and selfdrive_active
    measured_acceleration, acceleration_valid = self.rack_acceleration.update(
      sample_mono_ns=state_sample_mono_ns,
      rack_rate_deg_s=measured_rate,
      inputs_valid=self.vehicle_state_valid,
    )
    self.measured_acceleration_deg_s2 = measured_acceleration
    self.lateral_valid = self.vehicle_state_valid and acceleration_valid
    engagement_boundary = self.lateral_active != self._previous_lateral_active
    self._previous_lateral_active = self.lateral_active

    live_mapping = self._live_mapping(
      live_parameters,
      inputs_valid=live_parameters_inputs_valid,
    )
    profile_speed = current_speed if self.vehicle_state_valid else self.profile.nodes[0].speed_mps
    transport_delay = self.profile.parameters_at(
      profile_speed,
    ).parameters.transport_delay_s
    adaptation, scalar_curvature = self._adapt_intent(
      state_sample_mono_ns=state_sample_mono_ns,
      control_witness_mono_ns=control_witness_mono_ns,
      model_publication_mono_ns=model_publication_mono_ns,
      model_message=model_message,
      model_message_valid=model_message_valid,
      model_message_alive=model_message_alive,
      current_v_ego_m_s=(current_speed if self.vehicle_state_valid else math.nan),
      transport_delay_s=transport_delay,
    )
    self.intent_adaptation = adaptation
    self.model_input_valid = bool(adaptation.frame is not None and adaptation.frame.validity.model_valid)

    self.recorded_actuator_constrained = self.vehicle_state_valid and abs(requested_torque - applied_torque) > RECORDED_ACTUATOR_CONSTRAINT_TOLERANCE
    lateral_accel_offset = float(
      self.car_params.lateralTuning.torque.latAccelOffset,
    )
    result = self.core.update(
      frame=adaptation.frame,
      intent_status=adaptation.status,
      intent_plan_times_s=self.plan_times_s,
      intent_orientation_rates_z=self.orientation_rates_z,
      intent_velocities_x=self.velocities_x,
      scalar_curvature=(scalar_curvature if self.model_input_valid else math.nan),
      current_v_ego_m_s=(current_speed if self.vehicle_state_valid else math.nan),
      measured_rack_angle_deg=measured_angle,
      measured_rack_rate_deg_s=measured_rate,
      measured_rack_acceleration_deg_s2=measured_acceleration,
      recorded_applied_torque=applied_torque,
      lateral_accel_offset=lateral_accel_offset,
      live_mapping=live_mapping,
      lateral_active=self.lateral_active,
      lateral_valid=self.lateral_valid,
      engagement_boundary=engagement_boundary,
      live_parameters_valid=self.live_parameters_valid,
      steering_pressed=steering_pressed,
      actuator_constrained=self.recorded_actuator_constrained,
      # A passive shadow has no constrained output. The one-step feasibility
      # projection below is diagnostic and must never influence the observer.
      output_constrained=False,
      standstill=standstill,
    )
    self.core_result = result
    self.measured_previous_applied_torque = applied_torque if math.isfinite(applied_torque) else 0.0
    self.measured_driver_torque = driver_torque if math.isfinite(driver_torque) else 0.0
    self.feasible_torque = 0.0
    self.unmet_torque = 0.0
    self.feasibility_constrained = False
    feasibility_valid = result.valid and math.isfinite(result.raw_torque) and math.isfinite(applied_torque) and math.isfinite(driver_torque)
    if feasibility_valid:
      try:
        envelope = apply_torque_envelope(
          self.runtime_bundle.torque_limits,
          result.raw_torque,
          applied_torque,
          driver_torque,
        )
        self.feasible_torque = envelope.applied_torque
        self.unmet_torque = result.raw_torque - envelope.applied_torque
        self.feasibility_constrained = envelope.constrained
      except (TypeError, ValueError, OverflowError):
        feasibility_valid = False

    lower_index = min(
      max(int(result.profile_lower_node), 0),
      len(self.profile.nodes) - 1,
    )
    upper_index = min(
      max(int(result.profile_upper_node), 0),
      len(self.profile.nodes) - 1,
    )
    lower_node = self.profile.nodes[lower_index]
    upper_node = self.profile.nodes[upper_index]
    self.profile_lower_node_speed_mps = lower_node.speed_mps
    self.profile_upper_node_speed_mps = upper_node.speed_mps
    self.profile_confidence = lower_node.parameters.confidence + result.profile_upper_weight * (
      upper_node.parameters.confidence - lower_node.parameters.confidence
    )
    self.valid = bool(result.valid and feasibility_valid and self.vehicle_state_valid and self.model_input_valid)
    return result


def _finite_or_zero(value: object) -> float:
  try:
    converted = float(value)
  except (TypeError, ValueError, OverflowError):
    return 0.0
  return converted if math.isfinite(converted) else 0.0


def _uint64_or_zero(value: object) -> int:
  try:
    converted = int(value)
  except (TypeError, ValueError, OverflowError):
    return 0
  return converted if 0 <= converted <= (1 << 64) - 1 else 0


def populate_shadow_message(
  message: Any,
  shadow: Any,
  runner: ModularShadowRunner,
  *,
  log_mono_time_ns: int,
  compute_time_seconds: float,
) -> None:
  """Populate and normalize the raw-core-to-Cap'n-Proto boundary."""
  result = runner.core_result
  adaptation = runner.intent_adaptation
  intent_status: IntentBuildStatus | None = None if adaptation is None else adaptation.status

  message.logMonoTime = int(log_mono_time_ns)
  message.valid = bool(runner.valid)
  shadow.modularSchemaVersion = int(MODULAR_SHADOW_SCHEMA_VERSION)
  shadow.modularRuntimeVehicleIdentityHash = str(
    runner.runtime_vehicle_identity_hash,
  )
  shadow.modularPolicyHash = str(runner.policy_hash)
  shadow.modularProfileHash = str(runner.profile_hash)
  shadow.modularModelFrameId = int(result.model_frame_id)
  shadow.modularIntentStatus = int(result.intent_code)
  shadow.modularCoreStatus = int(result.status)
  shadow.modularValid = bool(runner.valid)
  shadow.modularIntentUsable = bool(intent_status is not None and intent_status.usable)
  shadow.modularProfileQualified = bool(result.profile_qualified)
  shadow.modularReferenceValid = bool(result.reference_valid)
  shadow.modularScalarOnly = bool(result.reference_scalar_only)
  shadow.modularNominalMappingUsed = bool(result.reference_valid and not result.rack_mapping_valid)
  shadow.modularLiveParametersValid = bool(
    runner.live_parameters_valid,
  )
  shadow.modularRecordedActuatorConstrained = bool(
    runner.recorded_actuator_constrained,
  )
  shadow.modularFeasibilityConstrained = bool(
    runner.feasibility_constrained,
  )
  shadow.modularObserverSaturated = bool(result.observer_saturated)
  shadow.modularRawTorque = _finite_or_zero(result.raw_torque)
  shadow.modularFeasibleTorque = _finite_or_zero(
    runner.feasible_torque,
  )
  shadow.modularUnmetTorque = _finite_or_zero(runner.unmet_torque)
  shadow.modularAligningTorque = _finite_or_zero(
    result.aligning_torque,
  )
  shadow.modularFrictionTorque = _finite_or_zero(
    result.friction_torque,
  )
  shadow.modularMotionFeedforwardTorque = _finite_or_zero(
    result.motion_feedforward_torque,
  )
  shadow.modularPositionFeedbackTorque = _finite_or_zero(
    result.position_feedback_torque,
  )
  shadow.modularRateFeedbackTorque = _finite_or_zero(
    result.rate_feedback_torque,
  )
  shadow.modularDisturbanceTorque = _finite_or_zero(
    result.disturbance_torque,
  )
  shadow.modularDesiredCurvature = _finite_or_zero(
    result.desired_curvature,
  )
  shadow.modularDesiredCurvatureRate = _finite_or_zero(
    result.desired_curvature_rate,
  )
  shadow.modularDesiredCurvatureAcceleration = _finite_or_zero(
    result.desired_curvature_acceleration,
  )
  shadow.modularDesiredAngleDeg = _finite_or_zero(
    result.desired_angle_deg,
  )
  shadow.modularDesiredRateDegS = _finite_or_zero(
    result.desired_rate_deg_s,
  )
  shadow.modularDesiredAccelerationDegS2 = _finite_or_zero(
    result.desired_acceleration_deg_s2,
  )
  shadow.modularMeasuredAngleDeg = _finite_or_zero(
    result.measured_angle_deg,
  )
  shadow.modularMeasuredRateDegS = _finite_or_zero(
    result.measured_rate_deg_s,
  )
  shadow.modularMeasuredAccelerationDegS2 = _finite_or_zero(
    runner.measured_acceleration_deg_s2,
  )
  shadow.modularPredictedAngleDeg = _finite_or_zero(
    result.predicted_angle_deg,
  )
  shadow.modularPredictedRateDegS = _finite_or_zero(
    result.predicted_rate_deg_s,
  )
  shadow.modularPositionErrorDeg = _finite_or_zero(
    result.position_error_deg,
  )
  shadow.modularRateErrorDegS = _finite_or_zero(
    result.rate_error_deg_s,
  )
  shadow.modularRequiredAccelerationDegS2 = _finite_or_zero(
    result.required_acceleration_deg_s2,
  )
  shadow.modularObserverEstimateTorque = _finite_or_zero(
    result.observer_estimated_disturbance_torque,
  )
  shadow.modularObserverInstantaneousTorque = _finite_or_zero(
    result.observer_instantaneous_disturbance_torque,
  )
  shadow.modularObserverStatus = int(result.observer_status)
  shadow.modularProfileLowerNodeSpeedMps = _finite_or_zero(
    runner.profile_lower_node_speed_mps,
  )
  shadow.modularProfileUpperNodeSpeedMps = _finite_or_zero(
    runner.profile_upper_node_speed_mps,
  )
  shadow.modularProfileUpperWeight = _finite_or_zero(
    result.profile_upper_weight,
  )
  shadow.modularTorquePerLateralAccel = _finite_or_zero(
    result.torque_per_lateral_accel,
  )
  shadow.modularRackGainDegS2PerTorque = _finite_or_zero(
    result.rack_gain_deg_s2_per_torque,
  )
  shadow.modularRackDampingPerS = _finite_or_zero(
    result.rack_damping_per_s,
  )
  shadow.modularTransportDelaySeconds = _finite_or_zero(
    result.transport_delay_s,
  )
  shadow.modularStaticFrictionTorque = _finite_or_zero(
    result.static_friction_torque,
  )
  shadow.modularKineticFrictionTorque = _finite_or_zero(
    result.kinetic_friction_torque,
  )
  shadow.modularRackRateResolutionDegS = _finite_or_zero(
    result.rack_rate_resolution_deg_s,
  )
  shadow.modularProfileConfidence = _finite_or_zero(
    runner.profile_confidence,
  )
  shadow.modularPlanAgeSeconds = _finite_or_zero(
    0.0 if intent_status is None else intent_status.publication_age_s,
  )
  shadow.modularDesiredCurvatureTimeSeconds = _finite_or_zero(
    0.0 if adaptation is None or adaptation.frame is None else adaptation.frame.timing.scalar_action_plan_s,
  )
  shadow.modularPlanTimeNowSeconds = _finite_or_zero(
    result.plan_time_now_s,
  )
  shadow.modularPhysicalEffectPlanSeconds = _finite_or_zero(
    result.physical_effect_plan_s,
  )
  shadow.modularCurrentSpeedMps = _finite_or_zero(
    result.current_speed_mps,
  )
  shadow.modularEffectSpeedMps = _finite_or_zero(
    result.effect_speed_mps,
  )
  shadow.modularMeasuredPreviousAppliedTorque = _finite_or_zero(
    runner.measured_previous_applied_torque,
  )
  shadow.modularMeasuredDriverTorque = _finite_or_zero(
    runner.measured_driver_torque,
  )
  shadow.modularComputeTimeSeconds = _finite_or_zero(
    compute_time_seconds,
  )
  shadow.modularModelInputValid = bool(runner.model_input_valid)
  shadow.modularVehicleStateValid = bool(
    runner.vehicle_state_valid,
  )
  shadow.modularLateralActive = bool(runner.lateral_active)
  shadow.modularLateralValid = bool(runner.lateral_valid)
  shadow.modularActuationEnvelopeVerified = bool(
    runner.runtime_bundle.torque_limits.production_envelope_verified,
  )
  shadow.modularStateSampleMonoTime = _uint64_or_zero(
    runner.state_sample_mono_ns,
  )
  shadow.modularControlWitnessMonoTime = _uint64_or_zero(
    runner.control_witness_mono_ns,
  )
  shadow.modularStateAgeSeconds = _finite_or_zero(
    result.state_age_s,
  )
  shadow.modularTotalPredictionHorizonSeconds = _finite_or_zero(
    result.total_prediction_horizon_s,
  )


class BlatV2Shadow:
  """Messaging wrapper around the preconstructed passive runner."""

  def __init__(self) -> None:
    assert_no_actuation_publishers()
    messaging = _messaging_module()
    from openpilot.common.params import Params

    params = Params()
    car_params = messaging.log_from_bytes(
      params.get("CarParams", block=True),
      car.CarParams,
    )
    self.runner = ModularShadowRunner.from_car_params(car_params)
    self.sm = messaging.SubMaster(
      list(SUBSCRIBED_SERVICES),
      poll="controlsState",
    )
    self.pm = messaging.PubMaster(list(PUBLISHED_SERVICES))
    assert set(self.pm.sock) == {"blatV2Shadow"}
    assert "carControl" not in self.pm.sock
    self.message = messaging.new_message("blatV2Shadow")
    self.shadow = self.message.blatV2Shadow
    self.model_selector = _CanonicalModelSelector()

  @staticmethod
  def _checks(sm: Any, services: tuple[str, ...]) -> bool:
    return bool(all(sm.seen[service] for service in services) and sm.all_checks(list(services)))

  def step(self) -> None:
    self.sm.update()
    if not self.sm.updated["controlsState"]:
      return

    vehicle_services = (
      "carState",
      "carControl",
      "carOutput",
      "selfdriveState",
      "controlsState",
    )
    controls_model_mono_ns = int(
      self.sm["controlsState"].lateralPlanMonoTime,
    )
    model_resolved = self.model_selector.select(
      controls_model_mono_ns=controls_model_mono_ns,
      current_message=self.sm["modelV2"],
      current_mono_ns=int(self.sm.logMonoTime["modelV2"]),
      current_valid=bool(self.sm.valid["modelV2"]),
      current_alive=bool(self.sm.alive["modelV2"]),
      current_available=bool(self.sm.seen["modelV2"]),
    )
    selected_model = self.model_selector.selected_message
    # This witness shares the monotonic log clock and is captured immediately
    # before numerical computation. controlsState.logMonoTime is only a poll
    # event timestamp and can predate work performed by this process.
    control_witness_mono_ns = _control_witness_mono_ns()
    start = time.perf_counter()
    self.runner.update(
      state_sample_mono_ns=int(self.sm.logMonoTime["carState"]),
      control_witness_mono_ns=control_witness_mono_ns,
      model_publication_mono_ns=(self.model_selector.selected_mono_ns if model_resolved else 0),
      model_message=selected_model,
      car_state=self.sm["carState"],
      car_control=self.sm["carControl"],
      car_output=self.sm["carOutput"],
      selfdrive_state=self.sm["selfdriveState"],
      live_parameters=self.sm["liveParameters"],
      model_message_valid=bool(
        model_resolved and self.model_selector.selected_valid,
      ),
      model_message_alive=bool(
        model_resolved and self.model_selector.selected_alive,
      ),
      vehicle_inputs_valid=self._checks(self.sm, vehicle_services),
      live_parameters_inputs_valid=self._checks(
        self.sm,
        ("liveParameters",),
      ),
    )
    compute_time_seconds = time.perf_counter() - start
    populate_shadow_message(
      self.message,
      self.shadow,
      self.runner,
      log_mono_time_ns=control_witness_mono_ns,
      compute_time_seconds=compute_time_seconds,
    )
    self.pm.send("blatV2Shadow", self.message)

  def run(self) -> None:
    while True:
      self.step()


def main() -> None:
  # Roam only on cores 0-4. Core 5 hosts equal-priority planning work and
  # cores 6-7 host camera/model workloads; CTRL_LOW always yields to controlsd.
  config_realtime_process([0, 1, 2, 3, 4], Priority.CTRL_LOW)
  BlatV2Shadow().run()


if __name__ == "__main__":
  main()

"""Deterministic live/replay adapter for one modular-controller frame.

This module is the only translation layer between cereal/opendbc messages and
the numerical BLaTv2 artifacts.  It retains no steering policy:

* the raw model scalar and its published action timestamp are consumed as-is;
* model and velocity trajectories must share the exact native time grid;
* live rack mapping is built from one ``liveParameters`` snapshot;
* unsigned rack-rate magnitude is signed from measured angle motion;
* rack acceleration is differentiated only across consecutive, signed
  carState sample gaps no larger than 15 ms; and
* the previous command is accepted only when ``torqueOutputCan`` is an exact
  integer count; and
* physical-history eligibility separately requires the platform steering
  request bit and its validity witness.

All trajectory storage and the output object are allocated once.  Callers
must snapshot values they need after the next :meth:`prepare` call.
"""

from __future__ import annotations

from enum import IntEnum
import math
from typing import Any

from opendbc.car.structs import car
from opendbc.car.vehicle_model import VehicleModel

from openpilot.selfdrive.controls.lib.blatv2.actuator import (
  RuntimeTorqueLimits,
)
from openpilot.selfdrive.controls.lib.blatv2.intent import (
  INTENT_CAPACITY,
  IntentAdaptation,
  adapt_model_intent_into,
)
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import (
  RackMappingSnapshot,
)
from openpilot.selfdrive.controls.lib.blatv2.rack_motion import (
  SignedRackMotionNormalizer,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  VehicleProfile,
)


# One recorded-frame quality bound is shared by rack differentiation and the
# live controller's control-witness cadence check.  Neither path may bridge a
# missing 100 Hz controller frame.
MAX_RECORDED_FRAME_GAP_NS = 15_000_000


def exact_nonnegative_int(value: Any, description: str) -> int:
  if isinstance(value, bool):
    raise ValueError(f"{description} must be an exact nonnegative integer")
  result = int(value)
  if result != value or result < 0:
    raise ValueError(f"{description} must be an exact nonnegative integer")
  return result


class LiveAdapterStatus(IntEnum):
  OK = 0
  INVALID_VEHICLE_STATE = 1
  INVALID_LIVE_PARAMETERS = 2
  INVALID_RACK_DERIVATIVE = 3
  INVALID_MODEL_FIELDS = 4


class PreparedLiveInput:
  """Reused primitive/message-free result from :class:`LiveInputAdapter`."""

  __slots__ = (
    "status",
    "adaptation",
    "scalar_curvature",
    "current_v_ego_m_s",
    "measured_rack_angle_deg",
    "measured_rack_rate_deg_s",
    "measured_rack_acceleration_deg_s2",
    "driver_torque",
    "steering_pressed",
    "standstill",
    "vehicle_state_valid",
    "rack_derivative_valid",
    "live_parameters_valid",
    "lateral_valid",
    "live_mapping",
    "state_sample_mono_ns",
    "control_witness_mono_ns",
    "model_publication_mono_ns",
    "model_timestamp_eof_ns",
    "desired_curvature_time_s",
  )

  def __init__(self) -> None:
    self.clear()

  def clear(self) -> None:
    self.status = LiveAdapterStatus.INVALID_VEHICLE_STATE
    self.adaptation: IntentAdaptation | None = None
    self.scalar_curvature = math.nan
    self.current_v_ego_m_s = math.nan
    self.measured_rack_angle_deg = math.nan
    self.measured_rack_rate_deg_s = math.nan
    self.measured_rack_acceleration_deg_s2 = math.nan
    self.driver_torque = math.nan
    self.steering_pressed = False
    self.standstill = False
    self.vehicle_state_valid = False
    self.rack_derivative_valid = False
    self.live_parameters_valid = False
    self.lateral_valid = False
    self.live_mapping: RackMappingSnapshot | None = None
    self.state_sample_mono_ns = 0
    self.control_witness_mono_ns = 0
    self.model_publication_mono_ns = 0
    self.model_timestamp_eof_ns = 0
    self.desired_curvature_time_s = 0.0


def exact_applied_torque_counts(
  car_output: Any,
  limits: RuntimeTorqueLimits,
) -> int | None:
  """Return the exact prior command-layer CAN count; never re-quantize torque.

  ``torqueOutputCan`` is the controller value emitted toward the platform. It
  is the exact discrete Markov state needed by the command envelope, but it is
  not proof of measured EPS torque or of the platform request bit remaining
  asserted during a safety-controller cut. Those response/constraint signals
  must remain separate observer inputs when the platform exposes them.
  """
  try:
    raw = car_output.actuatorsOutput.torqueOutputCan
    if isinstance(raw, bool):
      return None
    numeric = float(raw)
  except (AttributeError, TypeError, ValueError, OverflowError):
    return None
  if not math.isfinite(numeric) or not numeric.is_integer():
    return None
  counts = int(numeric)
  if abs(counts) > limits.steer_max:
    return None
  return counts


def exact_steering_request_state(
  car_output: Any,
) -> tuple[bool, int] | None:
  """Return the emitted request bit and platform fault-avoidance counter.

  This authenticates command-layer telemetry only. An asserted request is not
  proof of measured EPS torque, while a deasserted request means the carried
  ``torqueOutputCan`` count must not be entered as physical applied history.
  """
  try:
    actuators = car_output.actuatorsOutput
    active = actuators.steeringRequestActive
    valid = actuators.steeringRequestActiveValid
    counter = actuators.steeringRequestFaultAvoidanceCounter
  except (AttributeError, TypeError, ValueError, OverflowError):
    return None
  if type(active) is not bool or valid is not True or type(counter) is not int:
    return None
  if not 0 <= counter <= 255:
    return None
  return active, counter


class LiveInputAdapter:
  """Preallocated conversion of one frozen live/replay frame snapshot."""

  __slots__ = (
    "car_params",
    "profile",
    "vehicle_model",
    "plan_times_s",
    "orientation_rates_z",
    "velocities_x",
    "plan_curvatures",
    "result",
    "_rack_motion",
  )

  def __init__(
    self,
    *,
    car_params: car.CarParams,
    profile: VehicleProfile,
  ) -> None:
    if not isinstance(profile, VehicleProfile):
      raise TypeError("live adapter requires an exact VehicleProfile")
    self.car_params = car_params
    self.profile = profile
    self.vehicle_model = VehicleModel(car_params)
    self.plan_times_s = [0.0] * INTENT_CAPACITY
    self.orientation_rates_z = [0.0] * INTENT_CAPACITY
    self.velocities_x = [0.0] * INTENT_CAPACITY
    self.plan_curvatures = [0.0] * INTENT_CAPACITY
    self.result = PreparedLiveInput()
    self._rack_motion = SignedRackMotionNormalizer()

  def reset_derivative(self) -> None:
    self._rack_motion.reset()

  def observe_inactive_state(
    self,
    *,
    state_sample_mono_ns: int,
    car_state: Any,
    live_parameters: Any,
    inputs_valid: bool,
    live_parameters_inputs_valid: bool,
  ) -> None:
    """Warm signed rack motion while no engagement is bound."""
    try:
      sample_ns = int(state_sample_mono_ns)
      speed = float(car_state.vEgo)
      angle = float(car_state.steeringAngleDeg)
      rate = float(car_state.steeringRateDeg)
      standstill = bool(car_state.standstill)
      resolution = self.profile.parameters_at(
        speed,
      ).parameters.rack_rate_resolution_deg_s
      timestamp_valid = (
        not isinstance(state_sample_mono_ns, bool)
        and sample_ns == state_sample_mono_ns
        and sample_ns >= 0
      )
      mapping = self._live_mapping(
        live_parameters,
        inputs_valid=live_parameters_inputs_valid,
      )
    except (AttributeError, TypeError, ValueError, OverflowError):
      self._rack_motion.reset()
      return
    self._rack_motion.update(
      sample_mono_ns=sample_ns,
      steering_angle_deg=(
        angle - mapping.angle_offset_deg
        if mapping is not None
        else angle
      ),
      raw_rate_deg_s=rate,
      rate_resolution_deg_s=resolution,
      lifecycle_valid=(
        bool(inputs_valid)
        and timestamp_valid
        and speed >= 0.0
        and not standstill
        and mapping is not None
      ),
      maximum_gap_ns=MAX_RECORDED_FRAME_GAP_NS,
    )

  def _live_mapping(
    self,
    live_parameters: Any,
    *,
    inputs_valid: bool,
  ) -> RackMappingSnapshot | None:
    if not inputs_valid:
      return None
    try:
      valid = (
        bool(live_parameters.valid)
        and bool(live_parameters.angleOffsetValid)
        and bool(live_parameters.steerRatioValid)
        and bool(live_parameters.stiffnessFactorValid)
      )
      stiffness = float(live_parameters.stiffnessFactor)
      steer_ratio = float(live_parameters.steerRatio)
      roll = float(live_parameters.roll)
      angle_offset = float(live_parameters.angleOffsetDeg)
      if (
        not valid
        or not all(math.isfinite(value) for value in (
          stiffness, steer_ratio, roll, angle_offset,
        ))
        or stiffness <= 0.0
        or steer_ratio <= 0.0
      ):
        return None
      self.vehicle_model.update_params(stiffness, steer_ratio)
      return RackMappingSnapshot.from_vehicle_model(
        self.vehicle_model,
        roll_rad=roll,
        angle_offset_deg=angle_offset,
        valid=True,
      )
    except (
      AttributeError,
      TypeError,
      ValueError,
      OverflowError,
      ZeroDivisionError,
    ):
      return None

  @staticmethod
  def _same_native_grid(model_message: Any) -> bool:
    try:
      orientation_times = model_message.orientationRate.t
      velocity_times = model_message.velocity.t
      if len(orientation_times) != len(velocity_times):
        return False
      for left, right in zip(
        orientation_times,
        velocity_times,
        strict=True,
      ):
        if float(left) != float(right):
          return False
      return True
    except (AttributeError, TypeError, ValueError, OverflowError):
      return False

  def _invalid_adaptation(
    self,
    *,
    state_sample_mono_ns: int,
    control_witness_mono_ns: int,
    transport_delay_s: float,
  ) -> IntentAdaptation:
    return adapt_model_intent_into(
      state_sample_mono_ns=state_sample_mono_ns,
      control_witness_mono_ns=control_witness_mono_ns,
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
      physical_transport_delay_s=transport_delay_s,
      output_plan_times_s=self.plan_times_s,
      output_orientation_rates_z=self.orientation_rates_z,
      output_velocities_x=self.velocities_x,
      output_plan_curvatures=self.plan_curvatures,
    )

  def prepare(
    self,
    *,
    state_sample_mono_ns: int,
    control_witness_mono_ns: int,
    model_publication_mono_ns: int,
    model_message: Any,
    car_state: Any,
    live_parameters: Any,
    model_message_valid: bool,
    model_message_alive: bool,
    vehicle_inputs_valid: bool,
    live_parameters_inputs_valid: bool,
  ) -> PreparedLiveInput:
    """Consume one frozen snapshot and return the reused canonical result."""
    output = self.result
    output.clear()
    try:
      output.state_sample_mono_ns = exact_nonnegative_int(
        state_sample_mono_ns,
        "state sample timestamp",
      )
      output.control_witness_mono_ns = exact_nonnegative_int(
        control_witness_mono_ns,
        "control witness timestamp",
      )
      output.model_publication_mono_ns = exact_nonnegative_int(
        model_publication_mono_ns,
        "model publication timestamp",
      )
      output.current_v_ego_m_s = float(car_state.vEgo)
      output.measured_rack_angle_deg = float(
        car_state.steeringAngleDeg,
      )
      raw_rack_rate_deg_s = float(car_state.steeringRateDeg)
      output.driver_torque = float(car_state.steeringTorque)
      output.steering_pressed = bool(car_state.steeringPressed)
      output.standstill = bool(car_state.standstill)
    except (AttributeError, TypeError, ValueError, OverflowError):
      self._rack_motion.reset()
      output.adaptation = self._invalid_adaptation(
        state_sample_mono_ns=max(output.state_sample_mono_ns, 0),
        control_witness_mono_ns=max(output.control_witness_mono_ns, 0),
        transport_delay_s=0.0,
      )
      return output

    output.vehicle_state_valid = (
      bool(vehicle_inputs_valid)
      and output.control_witness_mono_ns >= output.state_sample_mono_ns
      and all(math.isfinite(value) for value in (
        output.current_v_ego_m_s,
        output.measured_rack_angle_deg,
        raw_rack_rate_deg_s,
        output.driver_torque,
      ))
      and output.current_v_ego_m_s >= 0.0
    )
    output.live_mapping = self._live_mapping(
      live_parameters,
      inputs_valid=live_parameters_inputs_valid,
    )
    output.live_parameters_valid = output.live_mapping is not None
    profile_speed = (
      output.current_v_ego_m_s
      if output.vehicle_state_valid
      else self.profile.nodes[0].speed_mps
    )
    profile_parameters = self.profile.parameters_at(
      profile_speed,
    ).parameters
    motion = self._rack_motion.update(
      sample_mono_ns=output.state_sample_mono_ns,
      steering_angle_deg=(
        output.measured_rack_angle_deg - output.live_mapping.angle_offset_deg
        if output.live_mapping is not None
        else output.measured_rack_angle_deg
      ),
      raw_rate_deg_s=raw_rack_rate_deg_s,
      rate_resolution_deg_s=profile_parameters.rack_rate_resolution_deg_s,
      lifecycle_valid=(
        output.vehicle_state_valid
        and output.live_mapping is not None
        and not output.standstill
      ),
      maximum_gap_ns=MAX_RECORDED_FRAME_GAP_NS,
    )
    output.measured_rack_rate_deg_s = (
      motion.signed_rate_deg_s if motion.sign_valid else math.nan
    )
    output.measured_rack_acceleration_deg_s2 = (
      motion.rack_acceleration_deg_s2
      if motion.derivative_continuous or motion.direction_reversal
      else math.nan
    )
    output.rack_derivative_valid = (
      motion.derivative_continuous or motion.direction_reversal
    )
    # Nominal mapping remains useful for passive core diagnostics, but live
    # actuation cannot manufacture rack-position error from missing/stale
    # angle offset, roll, stiffness, or steer ratio.
    output.lateral_valid = (
      output.vehicle_state_valid
      and output.rack_derivative_valid
      and output.live_parameters_valid
    )
    transport_delay = profile_parameters.transport_delay_s

    try:
      output.model_timestamp_eof_ns = exact_nonnegative_int(
        model_message.timestampEof,
        "model plan-origin timestamp",
      )
      model_frame_id = exact_nonnegative_int(
        model_message.frameId,
        "model frame id",
      )
      output.desired_curvature_time_s = float(
        model_message.action.desiredCurvatureTime,
      )
      output.scalar_curvature = float(
        model_message.action.desiredCurvature,
      )
      native_plan_times = model_message.orientationRate.t
      native_orientation_rates = model_message.orientationRate.z
      native_velocities = (
        model_message.velocity.x
        if self._same_native_grid(model_message)
        else ()
      )
      output.adaptation = adapt_model_intent_into(
        state_sample_mono_ns=output.state_sample_mono_ns,
        control_witness_mono_ns=output.control_witness_mono_ns,
        model_publication_mono_ns=output.model_publication_mono_ns,
        plan_origin_mono_ns=output.model_timestamp_eof_ns,
        model_frame_id=model_frame_id,
        message_valid=model_message_valid,
        message_alive=model_message_alive,
        scalar_desired_curvature=output.scalar_curvature,
        published_desired_curvature_time_s=(
          output.desired_curvature_time_s
        ),
        native_plan_times_s=native_plan_times,
        native_orientation_rates_z=native_orientation_rates,
        native_velocities_x=native_velocities,
        current_v_ego_m_s=output.current_v_ego_m_s,
        physical_transport_delay_s=transport_delay,
        output_plan_times_s=self.plan_times_s,
        output_orientation_rates_z=self.orientation_rates_z,
        output_velocities_x=self.velocities_x,
        output_plan_curvatures=self.plan_curvatures,
      )
    except (AttributeError, TypeError, ValueError, OverflowError):
      output.status = LiveAdapterStatus.INVALID_MODEL_FIELDS
      output.adaptation = self._invalid_adaptation(
        state_sample_mono_ns=max(output.state_sample_mono_ns, 0),
        control_witness_mono_ns=max(output.control_witness_mono_ns, 0),
        transport_delay_s=transport_delay,
      )
      return output

    if not output.vehicle_state_valid:
      output.status = LiveAdapterStatus.INVALID_VEHICLE_STATE
    elif not output.live_parameters_valid:
      output.status = LiveAdapterStatus.INVALID_LIVE_PARAMETERS
    elif not output.rack_derivative_valid:
      output.status = LiveAdapterStatus.INVALID_RACK_DERIVATIVE
    else:
      output.status = LiveAdapterStatus.OK
    return output

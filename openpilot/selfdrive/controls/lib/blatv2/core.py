"""Physical-effect-time composition for the modular BLaTv2 controller.

The core samples the model-authored path on the fixed two-second control grid,
advances the measured rack state through request-aware applied history, and
asks one :class:`HorizonController` for one command. The initial inverse-torque
decomposition remains the raw authored demand used by telemetry; the planned
command is kept separate and is the only value eligible for the outer safety
guard and production actuator envelope.

The core owns fixed storage for reference sampling and the recorded applied
torque history. The hot method returns one reused mutable result. Its
``snapshot`` helper allocates only when tests or telemetry explicitly request
an immutable copy.

Several leaf APIs intentionally return small immutable dataclasses
(``ReferenceBuildStatus``, ``RackMappingStatus``, ``InterpolatedProfile``,
``ObserverResult``, ``RackState``/``PlantStep``, and ``ComputedTorque``).
Those are the current shared-artifact boundaries; this composer does not fork
their arithmetic merely to avoid those bounded allocations.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
import math
from numbers import Integral

from opendbc.car.hyundai.steering_request import (
  steering_request_fault_avoidance_counter_valid,
)

from openpilot.selfdrive.controls.lib.blatv2.actuator import RuntimeTorqueLimits
from openpilot.selfdrive.controls.lib.blatv2.contracts import CanonicalFrame
from openpilot.selfdrive.controls.lib.blatv2.horizon import (
  HORIZON_SAMPLE_COUNT,
  HorizonController,
  HorizonPolicy,
  HorizonStatus,
)
from openpilot.selfdrive.controls.lib.blatv2.intent import IntentBuildStatus
from openpilot.selfdrive.controls.lib.blatv2.observer import (
  DisturbanceObserver,
  ObserverMeasurement,
  ObserverPolicy,
  ObserverStatus,
)
from openpilot.selfdrive.controls.lib.blatv2.plant import (
  RackState,
  TrackingPolicy,
  compute_inverse_torque,
  departure_friction_torque,
  predict_applied_history,
  steady_road_load_torque,
  step_plant,
)
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import (
  RackMappingSnapshot,
  curvature_from_measured_angle,
  map_reference_into,
)
from openpilot.selfdrive.controls.lib.blatv2.reference import (
  sample_reference_into,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  PhysicalParameters,
  VehicleProfile,
)


class CoreStatus(IntEnum):
  OK = 0
  DEGRADED_SCALAR_ONLY = 1
  DEGRADED_NOMINAL_MAPPING = 2
  SHADOW_UNQUALIFIED_PROFILE = 3
  INVALID_MEASUREMENT = 4
  INVALID_INTENT = 5
  TRANSPORT_TIME_MISMATCH = 6
  REFERENCE_FAILURE = 7
  RACK_MAPPING_FAILURE = 8
  INSUFFICIENT_APPLIED_HISTORY = 9
  PLANT_FAILURE = 10
  HORIZON_FAILURE = 11


class CoreResult:
  """Reused no-growth output buffer; call ``snapshot`` before the next frame."""

  __slots__ = (
    "status",
    "valid",
    "profile_qualified",
    "intent_code",
    "model_frame_id",
    "reference_valid",
    "reference_scalar_only",
    "rack_mapping_valid",
    "horizon_status",
    "horizon_valid",
    "observer_status",
    "observer_saturated",
    "plan_time_now_s",
    "physical_effect_plan_s",
    "state_age_s",
    "transport_delay_s",
    "total_prediction_horizon_s",
    "current_speed_mps",
    "effect_speed_mps",
    "profile_lower_node",
    "profile_upper_node",
    "profile_upper_weight",
    "torque_per_lateral_accel",
    "lateral_accel_offset_correction_mps2",
    "effective_lateral_accel_offset_mps2",
    "rack_gain_deg_s2_per_torque",
    "rack_damping_per_s",
    "static_friction_torque",
    "kinetic_friction_torque",
    "rack_rate_resolution_deg_s",
    "desired_curvature",
    "desired_curvature_rate",
    "desired_curvature_acceleration",
    "desired_angle_deg",
    "desired_rate_deg_s",
    "desired_acceleration_deg_s2",
    "measured_angle_deg",
    "measured_rate_deg_s",
    "predicted_angle_deg",
    "predicted_rate_deg_s",
    "prediction_history_count",
    "prediction_fractional_dt_s",
    "prediction_first_applied_torque",
    "prediction_last_applied_torque",
    "observer_aligning_torque",
    "observer_friction_torque",
    "observer_instantaneous_disturbance_torque",
    "observer_estimated_disturbance_torque",
    "raw_torque",
    "raw_requested_counts",
    "planned_torque",
    "planned_counts",
    "reactive_torque",
    "reactive_counts",
    "raw_to_planned_constrained",
    "raw_to_planned_unmet_torque",
    "raw_to_planned_residual_counts",
    "preparation_active",
    "preparation_scheduled",
    "driver_suppressed",
    "future_band_reachable",
    "first_unreachable_index",
    "first_unreachable_time_s",
    "maximum_band_residual_counts",
    "maximum_path_lead_deg",
    "maximum_path_rate_lead_deg_s",
    "path_lead_constrained_samples",
    "maximum_authority_required",
    "maximum_authority_active",
    "maximum_urgency",
    "first_request_suppression_index",
    "aligning_torque",
    "friction_torque",
    "motion_feedforward_torque",
    "position_feedback_torque",
    "rate_feedback_torque",
    "disturbance_torque",
    "position_error_deg",
    "rate_error_deg_s",
    "required_acceleration_deg_s2",
  )

  def __init__(self) -> None:
    self.clear(
      CoreStatus.INVALID_MEASUREMENT,
      profile_qualified=False,
      intent_code=0,
      model_frame_id=0,
    )

  def clear(
    self,
    status: CoreStatus,
    *,
    profile_qualified: bool,
    intent_code: int,
    model_frame_id: int,
  ) -> None:
    self.status = status
    self.valid = False
    self.profile_qualified = profile_qualified
    self.intent_code = intent_code
    self.model_frame_id = model_frame_id
    self.reference_valid = False
    self.reference_scalar_only = False
    self.rack_mapping_valid = False
    self.horizon_status = int(HorizonStatus.INVALID_INPUT)
    self.horizon_valid = False
    self.observer_status = int(ObserverStatus.DISABLED_NO_POLICY)
    self.observer_saturated = False
    self.plan_time_now_s = 0.0
    self.physical_effect_plan_s = 0.0
    self.state_age_s = 0.0
    self.transport_delay_s = 0.0
    self.total_prediction_horizon_s = 0.0
    self.current_speed_mps = 0.0
    self.effect_speed_mps = 0.0
    self.profile_lower_node = 0
    self.profile_upper_node = 0
    self.profile_upper_weight = 0.0
    self.torque_per_lateral_accel = 0.0
    self.lateral_accel_offset_correction_mps2 = 0.0
    self.effective_lateral_accel_offset_mps2 = 0.0
    self.rack_gain_deg_s2_per_torque = 0.0
    self.rack_damping_per_s = 0.0
    self.static_friction_torque = 0.0
    self.kinetic_friction_torque = 0.0
    self.rack_rate_resolution_deg_s = 0.0
    self.desired_curvature = 0.0
    self.desired_curvature_rate = 0.0
    self.desired_curvature_acceleration = 0.0
    self.desired_angle_deg = 0.0
    self.desired_rate_deg_s = 0.0
    self.desired_acceleration_deg_s2 = 0.0
    self.measured_angle_deg = 0.0
    self.measured_rate_deg_s = 0.0
    self.predicted_angle_deg = 0.0
    self.predicted_rate_deg_s = 0.0
    self.prediction_history_count = 0
    self.prediction_fractional_dt_s = 0.0
    self.prediction_first_applied_torque = 0.0
    self.prediction_last_applied_torque = 0.0
    self.observer_aligning_torque = 0.0
    self.observer_friction_torque = 0.0
    self.observer_instantaneous_disturbance_torque = 0.0
    self.observer_estimated_disturbance_torque = 0.0
    self.raw_torque = 0.0
    self.raw_requested_counts = 0
    self.planned_torque = 0.0
    self.planned_counts = 0
    self.reactive_torque = 0.0
    self.reactive_counts = 0
    self.raw_to_planned_constrained = False
    self.raw_to_planned_unmet_torque = 0.0
    self.raw_to_planned_residual_counts = 0
    self.preparation_active = False
    self.preparation_scheduled = False
    self.driver_suppressed = False
    self.future_band_reachable = False
    self.first_unreachable_index = -1
    self.first_unreachable_time_s = -1.0
    self.maximum_band_residual_counts = 0
    self.maximum_path_lead_deg = 0.0
    self.maximum_path_rate_lead_deg_s = 0.0
    self.path_lead_constrained_samples = 0
    self.maximum_authority_required = False
    self.maximum_authority_active = False
    self.maximum_urgency = 0.0
    self.first_request_suppression_index = -1
    self.aligning_torque = 0.0
    self.friction_torque = 0.0
    self.motion_feedforward_torque = 0.0
    self.position_feedback_torque = 0.0
    self.rate_feedback_torque = 0.0
    self.disturbance_torque = 0.0
    self.position_error_deg = 0.0
    self.rate_error_deg_s = 0.0
    self.required_acceleration_deg_s2 = 0.0

  def snapshot(self) -> tuple[object, ...]:
    """Allocate an immutable value copy for tests or telemetry only."""
    return tuple(getattr(self, name) for name in self.__slots__)


class _PrefixSequence(Sequence[float]):
  """Reusable non-allocating prefix view over a caller-owned sequence."""

  __slots__ = ("source", "count", "offset")

  def __init__(self) -> None:
    self.source: Sequence[float] = ()
    self.count = 0
    self.offset = 0

  def configure(
    self,
    source: Sequence[float],
    count: int,
    offset: int = 0,
  ) -> None:
    if count < 0 or offset < 0 or offset + count > len(source):
      raise ValueError("prefix view exceeds its source")
    self.source = source
    self.count = count
    self.offset = offset

  def __len__(self) -> int:
    return self.count

  def __getitem__(self, index: int) -> float:
    selected = int(index)
    if selected < 0:
      selected += self.count
    if selected < 0 or selected >= self.count:
      raise IndexError(index)
    return float(self.source[self.offset + selected])


class ModularControllerCore:
  """Deterministic composition of the fixed modular controller artifacts."""

  def __init__(
    self,
    *,
    fixed_dt_s: float,
    profile: VehicleProfile,
    tracking_policy: TrackingPolicy,
    observer_policy: ObserverPolicy | None,
    nominal_mapping: RackMappingSnapshot,
    runtime_limits: RuntimeTorqueLimits,
    horizon_policy: HorizonPolicy,
    plan_capacity: int,
  ) -> None:
    dt = float(fixed_dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
      raise ValueError("core fixed dt must be finite and positive")
    if not isinstance(profile, VehicleProfile):
      raise TypeError("core profile must use the modular VehicleProfile")
    if not isinstance(tracking_policy, TrackingPolicy):
      raise TypeError("core requires an explicit TrackingPolicy")
    if observer_policy is not None and not isinstance(
      observer_policy, ObserverPolicy,
    ):
      raise TypeError("core observer policy must be explicit or None")
    if not isinstance(nominal_mapping, RackMappingSnapshot):
      raise TypeError("core nominal mapping has the wrong type")
    if not nominal_mapping.valid:
      raise ValueError("core nominal mapping must be valid")
    if plan_capacity < 2:
      raise ValueError("core plan capacity must hold at least two samples")
    if not isinstance(runtime_limits, RuntimeTorqueLimits):
      raise TypeError("core requires explicit RuntimeTorqueLimits")
    if not isinstance(horizon_policy, HorizonPolicy):
      raise TypeError("core requires an explicit HorizonPolicy")

    self.fixed_dt_s = dt
    self.profile = profile
    self.tracking_policy = tracking_policy
    self.nominal_mapping = nominal_mapping
    self.runtime_limits = runtime_limits
    self.horizon_policy = horizon_policy
    self.plan_capacity = int(plan_capacity)
    self.observer = DisturbanceObserver(observer_policy, dt)
    self.horizon = HorizonController(
      fixed_dt_s=dt,
      limits=runtime_limits,
      profile=profile,
      tracking_policy=tracking_policy,
      horizon_policy=horizon_policy,
      nominal_mapping=nominal_mapping,
    )
    self.result = CoreResult()

    self._plan_times = _PrefixSequence()
    self._plan_rates = _PrefixSequence()
    self._plan_speeds = _PrefixSequence()
    self._prediction_view = _PrefixSequence()

    self._query_times = [0.0] * HORIZON_SAMPLE_COUNT
    self._reference_times = [0.0] * HORIZON_SAMPLE_COUNT
    self._reference_curvatures = [0.0] * HORIZON_SAMPLE_COUNT
    self._reference_curvature_rates = [0.0] * HORIZON_SAMPLE_COUNT
    self._reference_curvature_accelerations = [0.0] * HORIZON_SAMPLE_COUNT
    self._reference_speeds = [0.0] * HORIZON_SAMPLE_COUNT
    self._reference_speed_rates = [0.0] * HORIZON_SAMPLE_COUNT
    self._reference_speed_accelerations = [0.0] * HORIZON_SAMPLE_COUNT
    self._scratch_curvatures = [0.0] * self.plan_capacity
    self._scratch_curvature_tangents = [0.0] * self.plan_capacity
    self._scratch_speed_tangents = [0.0] * self.plan_capacity
    self._rack_angles = [0.0] * HORIZON_SAMPLE_COUNT
    self._rack_rates = [0.0] * HORIZON_SAMPLE_COUNT
    self._rack_accelerations = [0.0] * HORIZON_SAMPLE_COUNT
    self._committed_command_time_angles = [0.0] * HORIZON_SAMPLE_COUNT

    maximum_delay = max(
      node.parameters.transport_delay_s for node in profile.nodes
    )
    # The rack-state sample can precede the control witness. Keep fixed
    # storage beyond the transport-only minimum so a delayed, but correctly
    # ordered, state sample can be advanced to the same physical effect time
    # without allocating in the hot path. An older sample fails legibly when
    # the required history exceeds this fixed canonical-plan capacity.
    self.history_capacity = max(
      int(math.ceil(maximum_delay / dt)),
      self.plan_capacity,
    )
    self._applied_history = [0.0] * self.history_capacity
    self._prediction_torques = [0.0] * self.history_capacity
    self._history_write_index = 0
    self._history_count = 0
    self._selected_history_count = 0
    self._selected_full_steps = 0
    self._selected_fractional_dt_s = 0.0

  def _record_applied_torque(self, applied_torque: float) -> None:
    if self.history_capacity == 0:
      return
    self._applied_history[self._history_write_index] = applied_torque
    self._history_write_index = (
      self._history_write_index + 1
    ) % self.history_capacity
    self._history_count = min(
      self._history_count + 1, self.history_capacity,
    )

  def prime_applied_history(self, current_applied_torque: float) -> None:
    """Initialize transport state with measured zero-order-hold prehistory.

    An engagement starts without controller-frame history, while the rack has
    a physical torque state across its transport delay. Repeating the exact
    request-aware value across the fixed history ring is the deterministic
    initialization for that unknown pre-engagement interval. It predicts
    neither path nor future torque and is replaced one sample at a time by
    recorded response after engagement.
    """
    applied = float(current_applied_torque)
    if not math.isfinite(applied) or abs(applied) > 1.0:
      raise ValueError("applied-history prime must be finite and normalized")
    for index in range(self.history_capacity):
      self._applied_history[index] = applied
      self._prediction_torques[index] = applied
    self._history_write_index = 0
    self._history_count = self.history_capacity
    self._selected_history_count = 0
    self._selected_full_steps = 0
    self._selected_fractional_dt_s = 0.0

  def _select_applied_history(
    self,
    delay_s: float,
  ) -> bool:
    """Select oldest-to-newest committed torque for exactly ``delay_s``."""
    self._selected_history_count = 0
    self._selected_full_steps = 0
    self._selected_fractional_dt_s = 0.0
    if delay_s == 0.0:
      return True

    full_steps = int(math.floor(delay_s / self.fixed_dt_s))
    fractional_dt = delay_s - full_steps * self.fixed_dt_s
    floating_tolerance = 8.0 * math.ulp(
      max(delay_s, self.fixed_dt_s),
    )
    if fractional_dt <= floating_tolerance:
      fractional_dt = 0.0
    elif self.fixed_dt_s - fractional_dt <= floating_tolerance:
      full_steps += 1
      fractional_dt = 0.0

    required_count = full_steps + (1 if fractional_dt > 0.0 else 0)
    if required_count > self._history_count:
      return False
    oldest_index = (
      self._history_write_index - required_count
    ) % self.history_capacity
    for index in range(required_count):
      self._prediction_torques[index] = self._applied_history[
        (oldest_index + index) % self.history_capacity
      ]
    self._selected_history_count = required_count
    self._selected_full_steps = full_steps
    self._selected_fractional_dt_s = fractional_dt
    return True

  def _fill_committed_command_time_angles(
    self,
    *,
    initial_state: RackState,
    state_age_s: float,
    transport_delay_s: float,
    speed_mps: float,
    mapping: RackMappingSnapshot,
    lateral_accel_offset: float,
    parameters: PhysicalParameters,
    disturbance_torque: float,
  ) -> bool:
    """Sample rack angle at command time through already-committed torque."""
    for index in range(HORIZON_SAMPLE_COUNT):
      self._committed_command_time_angles[index] = 0.0
    tolerance = 8.0 * math.ulp(max(
      state_age_s + transport_delay_s,
      self.fixed_dt_s,
    ))
    last_index = min(
      int(math.floor((transport_delay_s + tolerance) / self.fixed_dt_s)),
      HORIZON_SAMPLE_COUNT - 1,
    )
    state = initial_state
    elapsed_s = 0.0
    segment_index = 0
    segment_remaining_s = (
      self._selected_fractional_dt_s
      if self._selected_fractional_dt_s > 0.0
      else self.fixed_dt_s
    )
    for output_index in range(last_index + 1):
      target_s = state_age_s + output_index * self.fixed_dt_s
      while target_s - elapsed_s > tolerance:
        if segment_index >= self._selected_history_count:
          return False
        step_s = min(segment_remaining_s, target_s - elapsed_s)
        state = step_plant(
          state,
          self._prediction_torques[segment_index],
          speed_mps,
          mapping,
          self.nominal_mapping,
          lateral_accel_offset,
          parameters,
          disturbance_torque,
          step_s,
        ).state
        elapsed_s += step_s
        segment_remaining_s -= step_s
        if segment_remaining_s <= tolerance:
          segment_index += 1
          segment_remaining_s = self.fixed_dt_s
      self._committed_command_time_angles[output_index] = state.angle_deg
    return True

  def _copy_observer_result(self, observer_result: object) -> None:
    self.result.observer_status = int(observer_result.status)
    self.result.observer_saturated = observer_result.saturated
    self.result.observer_instantaneous_disturbance_torque = (
      observer_result.instantaneous_disturbance_torque
    )
    self.result.observer_estimated_disturbance_torque = (
      observer_result.estimated_disturbance_torque
    )

  def _update_observer(
    self,
    *,
    applied_torque: float,
    rack_rate_deg_s: float,
    rack_acceleration_deg_s2: float,
    aligning_torque: float,
    friction_torque: float,
    parameters: PhysicalParameters,
    lateral_active: bool,
    lateral_valid: bool,
    engagement_boundary: bool,
    model_valid: bool,
    vehicle_state_valid: bool,
    live_parameters_valid: bool,
    steering_pressed: bool,
    actuator_constrained: bool,
    output_constrained: bool,
    standstill: bool,
  ) -> None:
    # ObserverMeasurement is an existing immutable leaf-API boundary.
    observer_result = self.observer.update(
      ObserverMeasurement(
        applied_torque=applied_torque,
        rack_rate_deg_s=rack_rate_deg_s,
        rack_acceleration_deg_s2=rack_acceleration_deg_s2,
        aligning_torque=aligning_torque,
        friction_torque=friction_torque,
        lateral_active=lateral_active,
        lateral_valid=lateral_valid,
        engagement_boundary=engagement_boundary,
        model_valid=model_valid,
        vehicle_state_valid=vehicle_state_valid,
        live_parameters_valid=live_parameters_valid,
        steering_pressed=steering_pressed,
        actuator_constrained=actuator_constrained,
        output_constrained=output_constrained,
        standstill=standstill,
      ),
      parameters,
    )
    self._copy_observer_result(observer_result)

  def update(
    self,
    *,
    frame: CanonicalFrame | None,
    intent_status: IntentBuildStatus,
    intent_plan_times_s: Sequence[float],
    intent_orientation_rates_z: Sequence[float],
    intent_velocities_x: Sequence[float],
    scalar_curvature: float,
    current_v_ego_m_s: float,
    measured_rack_angle_deg: float,
    measured_rack_rate_deg_s: float,
    measured_rack_acceleration_deg_s2: float,
    previous_command_counts: int,
    recorded_applied_torque: float,
    driver_torque: float,
    lateral_accel_offset: float,
    live_mapping: RackMappingSnapshot | None,
    lateral_active: bool,
    lateral_valid: bool,
    engagement_boundary: bool,
    live_parameters_valid: bool,
    steering_pressed: bool,
    steering_request_fault_avoidance_counter: int,
    steering_request_state_valid: bool,
    actuator_constrained: bool,
    output_constrained: bool,
    standstill: bool,
  ) -> CoreResult:
    """Compose one horizon frame and return its sole planned command."""
    if (
      intent_status.count < 0
      or intent_status.count > self.plan_capacity
      or len(intent_plan_times_s) < intent_status.count
      or len(intent_orientation_rates_z) < intent_status.count
      or len(intent_velocities_x) < intent_status.count
    ):
      raise ValueError("adapted intent buffers violate core capacity")

    self.result.clear(
      CoreStatus.INVALID_MEASUREMENT,
      profile_qualified=self.profile.qualified,
      intent_code=int(intent_status.code),
      model_frame_id=intent_status.model_frame_id,
    )
    self.result.plan_time_now_s = (
      intent_status.plan_time_now_s
      if math.isfinite(intent_status.plan_time_now_s)
      else 0.0
    )
    self.result.state_age_s = (
      intent_status.state_age_s
      if math.isfinite(intent_status.state_age_s)
      else 0.0
    )
    self.result.measured_angle_deg = (
      measured_rack_angle_deg
      if math.isfinite(measured_rack_angle_deg)
      else 0.0
    )
    self.result.measured_rate_deg_s = (
      measured_rack_rate_deg_s
      if math.isfinite(measured_rack_rate_deg_s)
      else 0.0
    )
    command_state_valid = (
      not isinstance(previous_command_counts, bool)
      and isinstance(previous_command_counts, Integral)
      and abs(int(previous_command_counts))
      <= self.runtime_limits.steer_max
    )

    measurement_finite = (
      math.isfinite(scalar_curvature)
      and math.isfinite(current_v_ego_m_s)
      and current_v_ego_m_s >= 0.0
      and math.isfinite(measured_rack_angle_deg)
      and math.isfinite(measured_rack_rate_deg_s)
      and math.isfinite(measured_rack_acceleration_deg_s2)
      and math.isfinite(recorded_applied_torque)
      and abs(recorded_applied_torque) <= 1.0
      and math.isfinite(driver_torque)
      and math.isfinite(lateral_accel_offset)
      and command_state_valid
      and steering_request_state_valid is True
      and steering_request_fault_avoidance_counter_valid(
        steering_request_fault_avoidance_counter,
      )
    )
    if not measurement_finite:
      fallback_parameters = self.profile.nodes[0].parameters
      self._update_observer(
        applied_torque=(
          recorded_applied_torque
          if math.isfinite(recorded_applied_torque)
          else math.nan
        ),
        rack_rate_deg_s=measured_rack_rate_deg_s,
        rack_acceleration_deg_s2=measured_rack_acceleration_deg_s2,
        aligning_torque=0.0,
        friction_torque=0.0,
        parameters=fallback_parameters,
        lateral_active=lateral_active,
        lateral_valid=False,
        engagement_boundary=engagement_boundary,
        model_valid=False,
        vehicle_state_valid=False,
        live_parameters_valid=False,
        steering_pressed=steering_pressed,
        actuator_constrained=actuator_constrained,
        output_constrained=output_constrained,
        standstill=standstill,
      )
      return self.result

    self.result.current_speed_mps = current_v_ego_m_s
    self._record_applied_torque(recorded_applied_torque)
    current_profile = self.profile.parameters_at(current_v_ego_m_s)
    current_parameters = current_profile.parameters
    current_lateral_accel_offset = (
      lateral_accel_offset
      + current_parameters.lateral_accel_offset_correction_mps2
    )
    selected_mapping = (
      live_mapping
      if live_mapping is not None and live_mapping.valid
      else self.nominal_mapping
    )
    observer_mapping_valid = (
      live_parameters_valid
      and live_mapping is not None
      and live_mapping.valid
    )

    model_valid = (
      frame is not None
      and frame.validity.model_valid
      and intent_status.usable
    )
    vehicle_state_valid = (
      frame is not None and frame.validity.vehicle_state_valid
    )
    observer_aligning = 0.0
    observer_friction = 0.0
    if model_valid and vehicle_state_valid:
      try:
        measured_curvature = curvature_from_measured_angle(
          measured_rack_angle_deg,
          current_v_ego_m_s,
          live_mapping,
          self.nominal_mapping,
        ).curvature
        observer_aligning = steady_road_load_torque(
          measured_curvature,
          current_v_ego_m_s,
          selected_mapping.roll_rad,
          current_lateral_accel_offset,
          current_parameters.torque_per_lateral_accel,
        )
        observer_friction = departure_friction_torque(
          measured_rack_rate_deg_s,
          measured_rack_rate_deg_s,
          current_parameters,
        )
      except (ValueError, OverflowError):
        vehicle_state_valid = False
        observer_aligning = 0.0
        observer_friction = 0.0
    self.result.observer_aligning_torque = observer_aligning
    self.result.observer_friction_torque = observer_friction
    self._update_observer(
      applied_torque=recorded_applied_torque,
      rack_rate_deg_s=measured_rack_rate_deg_s,
      rack_acceleration_deg_s2=measured_rack_acceleration_deg_s2,
      aligning_torque=observer_aligning,
      friction_torque=observer_friction,
      parameters=current_parameters,
      lateral_active=lateral_active,
      lateral_valid=lateral_valid,
      engagement_boundary=engagement_boundary,
      model_valid=model_valid,
      vehicle_state_valid=vehicle_state_valid,
      live_parameters_valid=observer_mapping_valid,
      steering_pressed=steering_pressed,
      actuator_constrained=actuator_constrained,
      output_constrained=output_constrained,
      standstill=standstill,
    )

    if frame is None or not intent_status.usable:
      self.result.status = CoreStatus.INVALID_INTENT
      return self.result

    transport_delay = current_parameters.transport_delay_s
    physical_effect_plan_s = (
      intent_status.plan_time_now_s + transport_delay
    )
    total_prediction_horizon_s = (
      frame.timing.state_age_s + transport_delay
    )
    self.result.transport_delay_s = transport_delay
    self.result.physical_effect_plan_s = physical_effect_plan_s
    self.result.total_prediction_horizon_s = (
      total_prediction_horizon_s
    )
    if (
      frame.timing.transport_delay_s != transport_delay
      or intent_status.plan_time_now_s != frame.timing.plan_time_now_s
      or intent_status.state_age_s != frame.timing.state_age_s
      or intent_status.physical_effect_plan_s != physical_effect_plan_s
      or (
        intent_status.total_prediction_horizon_s
        != total_prediction_horizon_s
      )
    ):
      self.result.status = CoreStatus.TRANSPORT_TIME_MISMATCH
      return self.result

    self._plan_times.configure(
      intent_plan_times_s, intent_status.count,
    )
    self._plan_rates.configure(
      intent_orientation_rates_z, intent_status.count,
    )
    self._plan_speeds.configure(
      intent_velocities_x, intent_status.count,
    )
    for index in range(HORIZON_SAMPLE_COUNT):
      self._query_times[index] = (
        physical_effect_plan_s + index * self.fixed_dt_s
      )
    try:
      reference_status = sample_reference_into(
        self._plan_times,
        self._plan_rates,
        self._plan_speeds,
        scalar_curvature,
        frame.timing.scalar_action_plan_s,
        intent_status.plan_time_now_s,
        current_v_ego_m_s,
        self._query_times,
        HORIZON_SAMPLE_COUNT,
        self._reference_times,
        self._reference_curvatures,
        self._reference_curvature_rates,
        self._reference_curvature_accelerations,
        self._reference_speeds,
        self._reference_speed_rates,
        self._reference_speed_accelerations,
        self._scratch_curvatures,
        self._scratch_curvature_tangents,
        self._scratch_speed_tangents,
      )
    except (TypeError, ValueError, OverflowError):
      self.result.status = CoreStatus.REFERENCE_FAILURE
      return self.result

    self.result.reference_valid = reference_status.valid
    self.result.reference_scalar_only = reference_status.scalar_only
    if (
      reference_status.count != HORIZON_SAMPLE_COUNT
      or not reference_status.valid
      or reference_status.scalar_only
    ):
      self.result.status = CoreStatus.REFERENCE_FAILURE
      return self.result
    self.result.desired_curvature = self._reference_curvatures[0]
    self.result.desired_curvature_rate = (
      self._reference_curvature_rates[0]
    )
    self.result.desired_curvature_acceleration = (
      self._reference_curvature_accelerations[0]
    )
    self.result.effect_speed_mps = self._reference_speeds[0]

    effect_profile = self.profile.parameters_at(
      self._reference_speeds[0],
    )
    effect_parameters = effect_profile.parameters
    self.result.profile_lower_node = effect_profile.lower_node
    self.result.profile_upper_node = effect_profile.upper_node
    self.result.profile_upper_weight = effect_profile.upper_weight
    self.result.torque_per_lateral_accel = (
      effect_parameters.torque_per_lateral_accel
    )
    self.result.lateral_accel_offset_correction_mps2 = (
      effect_parameters.lateral_accel_offset_correction_mps2
    )
    effect_lateral_accel_offset = (
      lateral_accel_offset
      + effect_parameters.lateral_accel_offset_correction_mps2
    )
    self.result.effective_lateral_accel_offset_mps2 = (
      effect_lateral_accel_offset
    )
    self.result.rack_gain_deg_s2_per_torque = (
      effect_parameters.rack_gain_deg_s2_per_torque
    )
    self.result.rack_damping_per_s = (
      effect_parameters.rack_damping_per_s
    )
    self.result.static_friction_torque = (
      effect_parameters.static_friction_torque
    )
    self.result.kinetic_friction_torque = (
      effect_parameters.kinetic_friction_torque
    )
    self.result.rack_rate_resolution_deg_s = (
      effect_parameters.rack_rate_resolution_deg_s
    )

    try:
      rack_status = map_reference_into(
        self._reference_curvatures,
        self._reference_curvature_rates,
        self._reference_curvature_accelerations,
        self._reference_speeds,
        self._reference_speed_rates,
        self._reference_speed_accelerations,
        HORIZON_SAMPLE_COUNT,
        live_mapping,
        self.nominal_mapping,
        self._rack_angles,
        self._rack_rates,
        self._rack_accelerations,
      )
    except (TypeError, ValueError, OverflowError):
      self.result.status = CoreStatus.RACK_MAPPING_FAILURE
      return self.result

    self.result.rack_mapping_valid = rack_status.valid
    if rack_status.count != HORIZON_SAMPLE_COUNT:
      self.result.status = CoreStatus.RACK_MAPPING_FAILURE
      return self.result
    self.result.desired_angle_deg = self._rack_angles[0]
    self.result.desired_rate_deg_s = self._rack_rates[0]
    self.result.desired_acceleration_deg_s2 = (
      self._rack_accelerations[0]
    )

    if not self._select_applied_history(total_prediction_horizon_s):
      self.result.status = CoreStatus.INSUFFICIENT_APPLIED_HISTORY
      return self.result
    history_count = self._selected_history_count
    full_steps = self._selected_full_steps
    fractional_dt = self._selected_fractional_dt_s
    self.result.prediction_history_count = history_count
    self.result.prediction_fractional_dt_s = fractional_dt
    if history_count > 0:
      self.result.prediction_first_applied_torque = (
        self._prediction_torques[0]
      )
      self.result.prediction_last_applied_torque = (
        self._prediction_torques[history_count - 1]
      )

    predicted_state = RackState(
      measured_rack_angle_deg,
      measured_rack_rate_deg_s,
      recorded_applied_torque,
    )
    try:
      if not self._fill_committed_command_time_angles(
        initial_state=predicted_state,
        state_age_s=frame.timing.state_age_s,
        transport_delay_s=transport_delay,
        speed_mps=self._reference_speeds[0],
        mapping=selected_mapping,
        lateral_accel_offset=effect_lateral_accel_offset,
        parameters=effect_parameters,
        disturbance_torque=self.observer.estimate_torque,
      ):
        raise ValueError("command-time transport history is incomplete")
      full_offset = 0
      if fractional_dt > 0.0:
        predicted_state = step_plant(
          predicted_state,
          self._prediction_torques[0],
          self._reference_speeds[0],
          selected_mapping,
          self.nominal_mapping,
          effect_lateral_accel_offset,
          effect_parameters,
          self.observer.estimate_torque,
          fractional_dt,
        ).state
        full_offset = 1
      self._prediction_view.configure(
        self._prediction_torques,
        full_steps,
        full_offset,
      )
      predicted_state = predict_applied_history(
        predicted_state,
        self._prediction_view,
        self._reference_speeds[0],
        selected_mapping,
        self.nominal_mapping,
        effect_lateral_accel_offset,
        effect_parameters,
        self.observer.estimate_torque,
        self.fixed_dt_s,
      )
      inverse = compute_inverse_torque(
        predicted_state,
        self._reference_curvatures[0],
        self._rack_angles[0],
        self._rack_rates[0],
        self._rack_accelerations[0],
        self._reference_speeds[0],
        selected_mapping.roll_rad,
        effect_lateral_accel_offset,
        effect_parameters,
        self.tracking_policy,
        self.observer.estimate_torque,
      )
    except (TypeError, ValueError, OverflowError):
      self.result.status = CoreStatus.PLANT_FAILURE
      return self.result

    self.result.predicted_angle_deg = predicted_state.angle_deg
    self.result.predicted_rate_deg_s = predicted_state.rate_deg_s
    self.result.raw_torque = inverse.raw_torque
    self.result.aligning_torque = inverse.aligning_torque
    self.result.friction_torque = inverse.friction_torque
    self.result.motion_feedforward_torque = (
      inverse.motion_feedforward_torque
    )
    self.result.position_feedback_torque = (
      inverse.position_feedback_torque
    )
    self.result.rate_feedback_torque = inverse.rate_feedback_torque
    self.result.disturbance_torque = inverse.disturbance_torque
    self.result.position_error_deg = inverse.position_error_deg
    self.result.rate_error_deg_s = inverse.rate_error_deg_s
    self.result.required_acceleration_deg_s2 = (
      inverse.required_acceleration_deg_s2
    )
    if not math.isfinite(self.result.raw_torque):
      self.result.status = CoreStatus.PLANT_FAILURE
      self.result.raw_torque = 0.0
      return self.result

    try:
      horizon_result = self.horizon.update(
        desired_curvatures=self._reference_curvatures,
        desired_angles_deg=self._rack_angles,
        desired_rates_deg_s=self._rack_rates,
        desired_accelerations_deg_s2=self._rack_accelerations,
        planned_speeds_mps=self._reference_speeds,
        initial_state=predicted_state,
        previous_applied_counts=int(previous_command_counts),
        driver_torque=driver_torque,
        steering_pressed=steering_pressed,
        lateral_active=lateral_active,
        current_steering_angle_deg=measured_rack_angle_deg,
        steering_request_fault_avoidance_counter=(
          steering_request_fault_avoidance_counter
        ),
        steering_request_state_valid=steering_request_state_valid,
        live_mapping=live_mapping,
        lateral_accel_offset_mps2=lateral_accel_offset,
        disturbance_torque=self.observer.estimate_torque,
        transport_delay_s=transport_delay,
        committed_command_time_angles_deg=(
          self._committed_command_time_angles
        ),
      )
    except (TypeError, ValueError, OverflowError):
      self.result.status = CoreStatus.HORIZON_FAILURE
      return self.result

    self.result.horizon_status = int(horizon_result.status)
    self.result.horizon_valid = horizon_result.valid
    if (
      not horizon_result.valid
      or not math.isfinite(horizon_result.raw_torque)
      or horizon_result.raw_torque != inverse.raw_torque
      or not math.isfinite(horizon_result.planned_torque)
      or abs(horizon_result.planned_counts) > self.runtime_limits.steer_max
      or horizon_result.planned_torque
      != horizon_result.planned_counts / self.runtime_limits.steer_max
    ):
      self.result.status = CoreStatus.HORIZON_FAILURE
      return self.result

    raw_requested_counts = int(round(
      inverse.raw_torque * self.runtime_limits.steer_max,
    ))
    self.result.raw_requested_counts = raw_requested_counts
    self.result.planned_torque = horizon_result.planned_torque
    self.result.planned_counts = horizon_result.planned_counts
    self.result.reactive_torque = horizon_result.reactive_torque
    self.result.reactive_counts = horizon_result.reactive_counts
    self.result.raw_to_planned_constrained = (
      raw_requested_counts != horizon_result.planned_counts
    )
    self.result.raw_to_planned_unmet_torque = (
      inverse.raw_torque - horizon_result.planned_torque
    )
    self.result.raw_to_planned_residual_counts = (
      raw_requested_counts - horizon_result.planned_counts
    )
    self.result.preparation_active = horizon_result.preparation_active
    self.result.preparation_scheduled = (
      horizon_result.preparation_scheduled
    )
    self.result.driver_suppressed = horizon_result.driver_suppressed
    self.result.future_band_reachable = (
      horizon_result.future_band_reachable
    )
    self.result.first_unreachable_index = (
      horizon_result.first_unreachable_index
    )
    self.result.first_unreachable_time_s = (
      horizon_result.first_unreachable_time_s
    )
    self.result.maximum_band_residual_counts = (
      horizon_result.maximum_band_residual_counts
    )
    self.result.maximum_path_lead_deg = (
      horizon_result.maximum_path_lead_deg
    )
    self.result.maximum_path_rate_lead_deg_s = (
      horizon_result.maximum_path_rate_lead_deg_s
    )
    self.result.path_lead_constrained_samples = (
      horizon_result.path_lead_constrained_samples
    )
    self.result.maximum_authority_required = (
      horizon_result.maximum_authority_required
    )
    self.result.maximum_authority_active = (
      horizon_result.maximum_authority_active
    )
    self.result.maximum_urgency = horizon_result.maximum_urgency
    self.result.first_request_suppression_index = (
      horizon_result.first_request_suppression_index
    )

    self.result.valid = True
    if not self.profile.qualified:
      self.result.status = CoreStatus.SHADOW_UNQUALIFIED_PROFILE
    elif not rack_status.valid:
      self.result.status = CoreStatus.DEGRADED_NOMINAL_MAPPING
    else:
      self.result.status = CoreStatus.OK
    return self.result

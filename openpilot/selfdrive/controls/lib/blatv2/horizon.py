"""Plant-aware selection around an exact two-second count projection.

The authored model path is the objective. A reactive inverse-torque rollout
provides a minimum-progress reference, while ``ReachableCountProjector``
supplies an exact count-space witness for future Hyundai rate limits. Only the
first reachable command is selected. Future commands are diagnostic and are
recomputed at the next control frame.

Preview may move the command before a path transition only when the shared
rack plant predicts no corresponding early rack motion. Once the authored path
is moving, or a meaningful tracking error exists, a candidate may not worsen
the next-state tracking cost relative to the reactive command. This prevents
future reachability from trading away the immediate path, while still allowing
motionless friction preload and release preparation.

Hyundai request-fault avoidance advances from the current observed angle and
then the predicted rack angle. Request-off frames retain transmitted count
state while contributing zero rack-effective torque.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Any

from opendbc.car.hyundai.steering_request import (
  apply_steering_request_fault_avoidance,
  steering_request_fault_avoidance_counter_valid,
)

from openpilot.selfdrive.controls.lib.blatv2.actuator import (
  RuntimeTorqueLimits,
  apply_torque_envelope_counts,
)
from openpilot.selfdrive.controls.lib.blatv2.plant import (
  PlantStep,
  RackState,
  TrackingPolicy,
  compute_inverse_torque,
  step_plant,
)
from openpilot.selfdrive.controls.lib.blatv2.preview import (
  REACHABILITY_SAMPLE_COUNT,
  ReachabilityStatus,
  ReachableCountProjector,
)
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import (
  RackMappingSnapshot,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  PhysicalParameters,
  VehicleProfile,
)


HORIZON_SECONDS = 2.0
IMMEDIATE_SECONDS = 0.5
PREPARATION_SECONDS = 1.2
CONTROL_DT_SECONDS = 0.01
HORIZON_SAMPLE_COUNT = 201
HORIZON_POLICY_SCHEMA_VERSION = 1
_NUMERICAL_POSITION_TOLERANCE_DEG = 1e-12
_NUMERICAL_RATE_TOLERANCE_DEG_S = 1e-10


class HorizonStatus(IntEnum):
  OK = 0
  FUTURE_CONSTRAINED = 1
  DRIVER_OVERRIDE = 2
  INVALID_INPUT = 3
  REACTIVE_ONLY = 4
  PLANT_FAILURE = 5


@dataclass(frozen=True, slots=True)
class HorizonPolicy:
  """Immutable response tolerances; vehicle dynamics remain in the profile."""

  revision: int
  provenance: str
  provisional: bool
  immediate_confidence: float
  preparation_confidence: float
  reserve_confidence: float
  smooth_position_tolerance_s: float
  smooth_rate_tolerance_quanta: float
  no_lead_position_tolerance_s: float
  no_lead_rate_tolerance_quanta: float
  maximum_torque_slack: float
  schema_version: int = HORIZON_POLICY_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if (
      type(self.revision) is not int
      or self.revision < 0
      or type(self.provenance) is not str
      or not self.provenance.strip()
      or type(self.provisional) is not bool
      or type(self.schema_version) is not int
      or self.schema_version != HORIZON_POLICY_SCHEMA_VERSION
    ):
      raise ValueError("horizon policy identity is invalid")
    confidence = (
      self.immediate_confidence,
      self.preparation_confidence,
      self.reserve_confidence,
    )
    values = (
      *confidence,
      self.smooth_position_tolerance_s,
      self.smooth_rate_tolerance_quanta,
      self.no_lead_position_tolerance_s,
      self.no_lead_rate_tolerance_quanta,
      self.maximum_torque_slack,
    )
    if not all(math.isfinite(value) for value in values):
      raise ValueError("horizon policy values must be finite")
    if not (1.0 == self.immediate_confidence >= self.preparation_confidence >= self.reserve_confidence > 0.0):
      raise ValueError("horizon confidence must be positive and monotone")
    if (
      self.smooth_position_tolerance_s <= 0.0
      or self.smooth_rate_tolerance_quanta <= 0.0
      or self.no_lead_position_tolerance_s <= 0.0
      or self.no_lead_rate_tolerance_quanta <= 0.0
      or not 0.0 <= self.maximum_torque_slack <= 1.0
    ):
      raise ValueError("horizon tolerances are outside their domain")

  def to_dict(self) -> dict[str, Any]:
    return {
      "immediate_confidence": self.immediate_confidence,
      "maximum_torque_slack": self.maximum_torque_slack,
      "no_lead_position_tolerance_s": self.no_lead_position_tolerance_s,
      "no_lead_rate_tolerance_quanta": self.no_lead_rate_tolerance_quanta,
      "preparation_confidence": self.preparation_confidence,
      "provenance": self.provenance,
      "provisional": self.provisional,
      "reserve_confidence": self.reserve_confidence,
      "revision": self.revision,
      "schema_version": self.schema_version,
      "smooth_position_tolerance_s": self.smooth_position_tolerance_s,
      "smooth_rate_tolerance_quanta": self.smooth_rate_tolerance_quanta,
    }

  def to_json(self) -> str:
    return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

  @property
  def sha256(self) -> str:
    return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

  @classmethod
  def from_json_file(cls, path: str | Path) -> HorizonPolicy:
    try:
      payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
      raise ValueError("horizon policy file is unreadable") from exc
    expected_keys = {
      "immediate_confidence",
      "maximum_torque_slack",
      "no_lead_position_tolerance_s",
      "no_lead_rate_tolerance_quanta",
      "preparation_confidence",
      "provenance",
      "provisional",
      "reserve_confidence",
      "revision",
      "schema_version",
      "smooth_position_tolerance_s",
      "smooth_rate_tolerance_quanta",
    }
    if type(payload) is not dict or set(payload) != expected_keys:
      raise ValueError("horizon policy keys do not match the schema")
    if (
      type(payload["revision"]) is not int
      or type(payload["provisional"]) is not bool
      or type(payload["provenance"]) is not str
      or type(payload["schema_version"]) is not int
    ):
      raise ValueError("horizon policy identity fields have wrong types")
    numeric_fields = expected_keys - {
      "provenance",
      "provisional",
      "revision",
      "schema_version",
    }
    if any(type(payload[field]) not in (int, float) for field in numeric_fields):
      raise ValueError("horizon policy numeric field has wrong type")
    return cls(
      revision=payload["revision"],
      provenance=payload["provenance"],
      provisional=payload["provisional"],
      immediate_confidence=float(payload["immediate_confidence"]),
      preparation_confidence=float(payload["preparation_confidence"]),
      reserve_confidence=float(payload["reserve_confidence"]),
      smooth_position_tolerance_s=float(payload["smooth_position_tolerance_s"]),
      smooth_rate_tolerance_quanta=float(payload["smooth_rate_tolerance_quanta"]),
      no_lead_position_tolerance_s=float(payload["no_lead_position_tolerance_s"]),
      no_lead_rate_tolerance_quanta=float(payload["no_lead_rate_tolerance_quanta"]),
      maximum_torque_slack=float(payload["maximum_torque_slack"]),
      schema_version=payload["schema_version"],
    )


class HorizonResult:
  """Reused scalar result; diagnostic trajectories remain planner-owned."""

  __slots__ = (
    "status",
    "valid",
    "raw_torque",
    "planned_torque",
    "planned_counts",
    "reactive_torque",
    "reactive_counts",
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
  )

  def __init__(self) -> None:
    self.clear(HorizonStatus.INVALID_INPUT)

  def clear(self, status: HorizonStatus) -> None:
    self.status = status
    self.valid = False
    self.raw_torque = 0.0
    self.planned_torque = 0.0
    self.planned_counts = 0
    self.reactive_torque = 0.0
    self.reactive_counts = 0
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

  def snapshot(self) -> tuple[object, ...]:
    return tuple(getattr(self, name) for name in self.__slots__)


def horizon_confidence(time_s: float, policy: HorizonPolicy) -> float:
  """Return continuous confidence over the immediate/preparation/reserve zones."""
  time = float(time_s)
  if not math.isfinite(time) or time < 0.0 or time > HORIZON_SECONDS:
    raise ValueError("horizon time is outside the fixed preview")
  if time <= IMMEDIATE_SECONDS:
    return policy.immediate_confidence
  if time <= PREPARATION_SECONDS:
    fraction = (time - IMMEDIATE_SECONDS) / (PREPARATION_SECONDS - IMMEDIATE_SECONDS)
    return policy.immediate_confidence + fraction * (policy.preparation_confidence - policy.immediate_confidence)
  fraction = (time - PREPARATION_SECONDS) / (HORIZON_SECONDS - PREPARATION_SECONDS)
  return policy.preparation_confidence + fraction * (policy.reserve_confidence - policy.preparation_confidence)


def _finite_float(value: object) -> float | None:
  if isinstance(value, (bool, str, bytes, bytearray)):
    return None
  try:
    numeric = float(value)
  except (TypeError, ValueError, OverflowError):
    return None
  return numeric if math.isfinite(numeric) else None


def _sequence_is_finite(values: Sequence[float], count: int) -> bool:
  if isinstance(values, (str, bytes, bytearray)):
    return False
  try:
    return len(values) == count and all(_finite_float(values[index]) is not None for index in range(count))
  except Exception:
    return False


def _sequence_prefix_is_finite(
  values: Sequence[float] | None,
  count: int,
) -> bool:
  if values is None or isinstance(values, (str, bytes, bytearray)):
    return False
  try:
    return len(values) >= count and all(_finite_float(values[index]) is not None for index in range(count))
  except Exception:
    return False


class HorizonController:
  """Bounded one-command selector around a coherent reactive plant rollout."""

  def __init__(
    self,
    *,
    fixed_dt_s: float,
    limits: RuntimeTorqueLimits,
    profile: VehicleProfile,
    tracking_policy: TrackingPolicy,
    horizon_policy: HorizonPolicy,
    nominal_mapping: RackMappingSnapshot,
  ) -> None:
    dt = _finite_float(fixed_dt_s)
    if dt is None or abs(dt - CONTROL_DT_SECONDS) > 8.0 * math.ulp(CONTROL_DT_SECONDS):
      raise ValueError("horizon controller requires the 100 Hz control grid")
    if not isinstance(limits, RuntimeTorqueLimits) or not limits.production_envelope_verified or limits.steer_step != 1:
      raise ValueError("horizon controller requires a verified 100 Hz envelope")
    if not isinstance(profile, VehicleProfile):
      raise TypeError("horizon controller requires a VehicleProfile")
    if not isinstance(tracking_policy, TrackingPolicy):
      raise TypeError("horizon controller requires a TrackingPolicy")
    if not isinstance(horizon_policy, HorizonPolicy):
      raise TypeError("horizon controller requires a HorizonPolicy")
    if not isinstance(nominal_mapping, RackMappingSnapshot) or not nominal_mapping.valid:
      raise ValueError("horizon controller requires a valid nominal mapping")
    if any(node.parameters.rack_rate_resolution_deg_s <= 0.0 for node in profile.nodes):
      raise ValueError("horizon controller requires measured rack-rate resolution")
    if HORIZON_SAMPLE_COUNT != REACHABILITY_SAMPLE_COUNT:
      raise ValueError("horizon and count projection grids disagree")

    self.fixed_dt_s = dt
    self.limits = limits
    self.profile = profile
    self.tracking_policy = tracking_policy
    self.horizon_policy = horizon_policy
    self.nominal_mapping = nominal_mapping
    self.projector = ReachableCountProjector(
      fixed_dt_s=dt,
      limits=limits,
    )
    self.result = HorizonResult()

    count = HORIZON_SAMPLE_COUNT
    self.raw_torques = [0.0] * count
    self.reachability_torques = [0.0] * count
    self.reactive_counts = [0] * count
    self.band_lower_counts = [0] * count
    self.band_upper_counts = [0] * count
    self.confidences = [horizon_confidence(index * dt, horizon_policy) for index in range(count)]
    self.reactive_angles_deg = [0.0] * count
    self.reactive_rates_deg_s = [0.0] * count
    self.steering_request_active = [False] * count
    self.steering_request_counters = [0] * count

  def _clear_arrays(self) -> None:
    for index in range(HORIZON_SAMPLE_COUNT):
      self.raw_torques[index] = 0.0
      self.reachability_torques[index] = 0.0
      self.reactive_counts[index] = 0
      self.band_lower_counts[index] = 0
      self.band_upper_counts[index] = 0
      self.reactive_angles_deg[index] = 0.0
      self.reactive_rates_deg_s[index] = 0.0
      self.steering_request_active[index] = False
      self.steering_request_counters[index] = 0

  def _bounded_request_counts(self, raw_torque: float) -> int | None:
    scaled = raw_torque * self.limits.steer_max
    if not math.isfinite(scaled):
      return None
    return min(
      max(int(round(scaled)), -self.limits.steer_max),
      self.limits.steer_max,
    )

  def _apply_neutral_envelope(self, requested: int, previous: int) -> int:
    return self.projector.project_neutral_counts(requested, previous)

  def _step(
    self,
    state: RackState,
    transmitted_counts: int,
    steering_request_active: bool,
    speed_mps: float,
    mapping: RackMappingSnapshot,
    lateral_accel_offset: float,
    parameters: PhysicalParameters,
    disturbance_torque: float,
    dt_s: float | None = None,
  ) -> PlantStep:
    return step_plant(
      state,
      (transmitted_counts / self.limits.steer_max if steering_request_active else 0.0),
      speed_mps,
      mapping,
      self.nominal_mapping,
      lateral_accel_offset,
      parameters,
      disturbance_torque,
      self.fixed_dt_s if dt_s is None else dt_s,
    )

  def _command_time_angle(
    self,
    *,
    index: int,
    current_steering_angle_deg: float,
    transport_delay_s: float,
    committed_command_time_angles_deg: Sequence[float] | None,
    planned_speeds_mps: Sequence[float],
    mapping: RackMappingSnapshot,
    lateral_accel_offset_mps2: float,
    disturbance_torque: float,
  ) -> float:
    if index == 0:
      return current_steering_angle_deg
    command_time_s = index * self.fixed_dt_s
    tolerance = 8.0 * math.ulp(max(command_time_s, transport_delay_s, self.fixed_dt_s))
    if command_time_s <= transport_delay_s + tolerance:
      if committed_command_time_angles_deg is None:
        raise ValueError("command-time transport history is unavailable")
      return float(committed_command_time_angles_deg[index])

    effect_offset_s = command_time_s - transport_delay_s
    base_index = int(math.floor(effect_offset_s / self.fixed_dt_s))
    fractional_dt_s = effect_offset_s - base_index * self.fixed_dt_s
    if fractional_dt_s <= tolerance:
      return self.reactive_angles_deg[base_index]
    if self.fixed_dt_s - fractional_dt_s <= tolerance:
      return self.reactive_angles_deg[base_index + 1]

    speed = float(planned_speeds_mps[base_index])
    parameters = self.profile.parameters_at(speed).parameters
    state = RackState(
      self.reactive_angles_deg[base_index],
      self.reactive_rates_deg_s[base_index],
      self.reactive_counts[base_index] / self.limits.steer_max,
    )
    return self._step(
      state,
      self.reactive_counts[base_index],
      self.steering_request_active[base_index],
      speed,
      mapping,
      lateral_accel_offset_mps2 + parameters.lateral_accel_offset_correction_mps2,
      parameters,
      disturbance_torque,
      fractional_dt_s,
    ).state.angle_deg

  def _movement_direction(
    self,
    desired_angles_deg: Sequence[float],
    desired_rates_deg_s: Sequence[float],
    initial_state: RackState,
    position_scale_deg: float,
  ) -> float:
    count = len(desired_angles_deg)
    if count < 2 or len(desired_rates_deg_s) < count:
      raise ValueError("movement direction requires at least two path samples")
    immediate_delta = float(desired_angles_deg[1]) - float(desired_angles_deg[0])
    if abs(immediate_delta) > _NUMERICAL_POSITION_TOLERANCE_DEG:
      return math.copysign(1.0, immediate_delta)
    for index in range(2, count):
      delta = float(desired_angles_deg[index]) - float(desired_angles_deg[0])
      if abs(delta) > _NUMERICAL_POSITION_TOLERANCE_DEG:
        return math.copysign(1.0, delta)
    current_error = float(desired_angles_deg[0]) - initial_state.angle_deg
    if abs(current_error) > position_scale_deg:
      return math.copysign(1.0, current_error)
    desired_rate = float(desired_rates_deg_s[0])
    if abs(desired_rate) > _NUMERICAL_RATE_TOLERANCE_DEG_S:
      return math.copysign(1.0, desired_rate)
    return 0.0

  @staticmethod
  def _path_leads(
    state: RackState,
    *,
    direction: float,
    desired_angle_deg: float,
    desired_rate_deg_s: float,
  ) -> tuple[float, float]:
    position_error = state.angle_deg - desired_angle_deg
    rate_error = state.rate_deg_s - desired_rate_deg_s
    if direction != 0.0:
      return (
        max(direction * position_error, 0.0),
        max(direction * rate_error, 0.0),
      )
    return abs(position_error), abs(rate_error)

  @staticmethod
  def _tracking_cost(
    state: RackState,
    desired_angle_deg: float,
    desired_rate_deg_s: float,
    position_scale_deg: float,
    rate_scale_deg_s: float,
  ) -> float:
    position_error = (desired_angle_deg - state.angle_deg) / position_scale_deg
    rate_error = (desired_rate_deg_s - state.rate_deg_s) / rate_scale_deg_s
    return position_error * position_error + 0.25 * rate_error * rate_error

  def _path_safe(
    self,
    *,
    initial: RackState,
    candidate: RackState,
    reactive: RackState,
    direction: float,
    desired_angle_now_deg: float,
    desired_angle_next_deg: float,
    desired_rate_now_deg_s: float,
    desired_rate_next_deg_s: float,
    position_tolerance_deg: float,
    rate_tolerance_deg_s: float,
  ) -> tuple[bool, float, float]:
    authored_moving = abs(desired_angle_next_deg - desired_angle_now_deg) > _NUMERICAL_POSITION_TOLERANCE_DEG
    initial_lead, _ = self._path_leads(
      initial,
      direction=direction,
      desired_angle_deg=desired_angle_now_deg,
      desired_rate_deg_s=desired_rate_now_deg_s,
    )
    candidate_lead, candidate_rate_lead = self._path_leads(
      candidate,
      direction=direction,
      desired_angle_deg=desired_angle_next_deg,
      desired_rate_deg_s=desired_rate_next_deg_s,
    )
    reactive_lead, reactive_rate_lead = self._path_leads(
      reactive,
      direction=direction,
      desired_angle_deg=desired_angle_next_deg,
      desired_rate_deg_s=desired_rate_next_deg_s,
    )
    if authored_moving:
      position_limit = position_tolerance_deg
    else:
      position_limit = max(initial_lead, reactive_lead)
      initial_error = abs(initial.angle_deg - desired_angle_now_deg)
      if initial_error > position_tolerance_deg:
        reactive_error = abs(reactive.angle_deg - desired_angle_next_deg)
        candidate_error = abs(candidate.angle_deg - desired_angle_next_deg)
        tracking_limit = max(
          position_tolerance_deg,
          min(initial_error, reactive_error),
        )
        if candidate_error > tracking_limit + _NUMERICAL_POSITION_TOLERANCE_DEG:
          return False, candidate_lead, candidate_rate_lead
    if initial_lead > position_tolerance_deg:
      position_limit = max(
        position_tolerance_deg,
        min(initial_lead, reactive_lead),
      )
    if candidate_lead > position_limit + _NUMERICAL_POSITION_TOLERANCE_DEG:
      return False, candidate_lead, 0.0

    rate_limit = max(reactive_rate_lead, rate_tolerance_deg_s, 0.0)
    if candidate_rate_lead > rate_limit + _NUMERICAL_RATE_TOLERANCE_DEG_S:
      return False, candidate_lead, candidate_rate_lead
    return True, candidate_lead, candidate_rate_lead

  @staticmethod
  def _unit_urgency(value: float) -> float:
    magnitude = abs(value)
    return magnitude / (1.0 + magnitude)

  def _select_first_count(
    self,
    *,
    initial_state: RackState,
    reactive_counts: int,
    witness_counts: int,
    previous_counts: int,
    driver_torque: float,
    raw_torque: float,
    desired_angles_deg: Sequence[float],
    desired_rates_deg_s: Sequence[float],
    speed_mps: float,
    mapping: RackMappingSnapshot,
    lateral_accel_offset: float,
    parameters: PhysicalParameters,
    disturbance_torque: float,
    steering_request_active: bool,
  ) -> tuple[int, float, float, float, bool]:
    reactive_step = self._step(
      initial_state,
      reactive_counts,
      steering_request_active,
      speed_mps,
      mapping,
      lateral_accel_offset,
      parameters,
      disturbance_torque,
    )
    requested = self._bounded_request_counts(raw_torque)
    if requested is None:
      raise ValueError("reactive torque is not representable")

    resolution = parameters.rack_rate_resolution_deg_s
    smooth_position_scale = max(
      resolution * self.horizon_policy.smooth_position_tolerance_s,
      _NUMERICAL_POSITION_TOLERANCE_DEG,
    )
    smooth_rate_scale = max(
      resolution * self.horizon_policy.smooth_rate_tolerance_quanta,
      _NUMERICAL_RATE_TOLERANCE_DEG_S,
    )
    no_lead_position_scale = max(
      resolution * self.horizon_policy.no_lead_position_tolerance_s,
      _NUMERICAL_POSITION_TOLERANCE_DEG,
    )
    no_lead_rate_scale = max(
      resolution * self.horizon_policy.no_lead_rate_tolerance_quanta,
      _NUMERICAL_RATE_TOLERANCE_DEG_S,
    )
    direction = self._movement_direction(
      desired_angles_deg,
      desired_rates_deg_s,
      initial_state,
      no_lead_position_scale,
    )
    desired_angle_now = float(desired_angles_deg[0])
    desired_angle_next = float(desired_angles_deg[1])
    desired_rate_now = float(desired_rates_deg_s[0])
    desired_rate_next = float(desired_rates_deg_s[1])
    reactive_cost = self._tracking_cost(
      reactive_step.state,
      desired_angle_next,
      desired_rate_next,
      smooth_position_scale,
      smooth_rate_scale,
    )
    position_error_now = (desired_angle_now - initial_state.angle_deg) / smooth_position_scale
    rate_error_now = (desired_rate_now - initial_state.rate_deg_s) / smooth_rate_scale
    tracking_urgency = self._unit_urgency(
      math.hypot(position_error_now, rate_error_now),
    )
    reachability_urgency = min(
      abs(witness_counts - reactive_counts) / max(self.limits.delta_up + self.limits.delta_down, 1),
      1.0,
    )
    authority_urgency = min(abs(raw_torque), 1.0)
    urgency = 1.0 - ((1.0 - tracking_urgency) * (1.0 - reachability_urgency) * (1.0 - authority_urgency))

    authored_active = (
      abs(desired_angle_next - desired_angle_now) > _NUMERICAL_POSITION_TOLERANCE_DEG
      or abs(desired_angle_now - initial_state.angle_deg) > no_lead_position_scale
      or abs(desired_rate_now - initial_state.rate_deg_s) > 0.25 * no_lead_rate_scale
    )
    raw_direction = (
      0.0
      if requested == previous_counts
      else math.copysign(
        1.0,
        requested - previous_counts,
      )
    )
    lower = apply_torque_envelope_counts(
      self.limits,
      -self.limits.steer_max,
      previous_counts,
      driver_torque,
    )
    upper = apply_torque_envelope_counts(
      self.limits,
      self.limits.steer_max,
      previous_counts,
      driver_torque,
    )
    span = max(upper - lower, 1)
    tracking_weight = 2.0 + 8.0 * tracking_urgency
    future_weight = 1.0 + 3.0 * reachability_urgency
    smooth_weight = 0.25 * (1.0 - urgency)
    maximum_authority_requested = abs(raw_torque) >= 1.0
    best: tuple[int, float, int, int, int, int, float, float] | None = None
    lead_constrained = False
    for counts in range(lower, upper + 1):
      stepped = self._step(
        initial_state,
        counts,
        steering_request_active,
        speed_mps,
        mapping,
        lateral_accel_offset,
        parameters,
        disturbance_torque,
      ).state
      safe, lead, rate_lead = self._path_safe(
        initial=initial_state,
        candidate=stepped,
        reactive=reactive_step.state,
        direction=direction,
        desired_angle_now_deg=desired_angle_now,
        desired_angle_next_deg=desired_angle_next,
        desired_rate_now_deg_s=desired_rate_now,
        desired_rate_next_deg_s=desired_rate_next,
        position_tolerance_deg=no_lead_position_scale,
        rate_tolerance_deg_s=no_lead_rate_scale,
      )
      if not safe:
        lead_constrained = True
        continue
      tracking_cost = self._tracking_cost(
        stepped,
        desired_angle_next,
        desired_rate_next,
        smooth_position_scale,
        smooth_rate_scale,
      )
      if authored_active and tracking_cost > reactive_cost + 1e-12:
        continue
      if authored_active and not maximum_authority_requested and raw_direction == direction and raw_direction * (counts - reactive_counts) < 0.0:
        continue
      score = tracking_weight * tracking_cost + future_weight * abs(counts - witness_counts) / span + smooth_weight * abs(counts - previous_counts) / span
      key = (
        abs(counts - reactive_counts) if maximum_authority_requested else 0,
        score,
        abs(counts - witness_counts),
        abs(counts - reactive_counts),
        abs(counts - previous_counts),
        counts,
        lead,
        rate_lead,
      )
      if best is None or key[:6] < best[:6]:
        best = key
    if best is None:
      return (
        reactive_counts,
        urgency,
        reactive_step.state.angle_deg,
        reactive_step.state.rate_deg_s,
        lead_constrained,
      )
    selected = int(best[5])
    selected_state = self._step(
      initial_state,
      selected,
      steering_request_active,
      speed_mps,
      mapping,
      lateral_accel_offset,
      parameters,
      disturbance_torque,
    ).state
    return selected, urgency, selected_state.angle_deg, selected_state.rate_deg_s, lead_constrained

  def update(
    self,
    *,
    desired_curvatures: Sequence[float],
    desired_angles_deg: Sequence[float],
    desired_rates_deg_s: Sequence[float],
    desired_accelerations_deg_s2: Sequence[float],
    planned_speeds_mps: Sequence[float],
    initial_state: RackState,
    previous_applied_counts: int,
    driver_torque: float,
    steering_pressed: bool,
    lateral_active: bool,
    current_steering_angle_deg: float,
    steering_request_fault_avoidance_counter: int,
    steering_request_state_valid: bool,
    live_mapping: RackMappingSnapshot | None,
    lateral_accel_offset_mps2: float,
    disturbance_torque: float,
    transport_delay_s: float = 0.0,
    committed_command_time_angles_deg: Sequence[float] | None = None,
  ) -> HorizonResult:
    """Select one command for a reference beginning at physical-effect time."""
    self.result.clear(HorizonStatus.INVALID_INPUT)
    self._clear_arrays()
    sequences = (
      desired_curvatures,
      desired_angles_deg,
      desired_rates_deg_s,
      desired_accelerations_deg_s2,
      planned_speeds_mps,
    )
    if (
      not isinstance(initial_state, RackState)
      or isinstance(previous_applied_counts, bool)
      or not isinstance(previous_applied_counts, Integral)
      or type(steering_pressed) is not bool
      or type(lateral_active) is not bool
      or type(steering_request_state_valid) is not bool
      or not steering_request_state_valid
      or not steering_request_fault_avoidance_counter_valid(
        steering_request_fault_avoidance_counter,
      )
      or (live_mapping is not None and not isinstance(live_mapping, RackMappingSnapshot))
      or any(not _sequence_is_finite(sequence, HORIZON_SAMPLE_COUNT) for sequence in sequences)
    ):
      return self.result
    previous_counts = int(previous_applied_counts)
    driver = _finite_float(driver_torque)
    offset = _finite_float(lateral_accel_offset_mps2)
    disturbance = _finite_float(disturbance_torque)
    current_steering_angle = _finite_float(current_steering_angle_deg)
    transport_delay = _finite_float(transport_delay_s)
    if (
      abs(previous_counts) > self.limits.steer_max
      or driver is None
      or offset is None
      or disturbance is None
      or current_steering_angle is None
      or transport_delay is None
      or transport_delay < 0.0
      or any(float(speed) < 0.0 for speed in planned_speeds_mps)
    ):
      return self.result
    committed_count = min(
      int(math.floor((transport_delay + 8.0 * math.ulp(max(transport_delay, self.fixed_dt_s))) / self.fixed_dt_s)) + 1,
      HORIZON_SAMPLE_COUNT,
    )
    if transport_delay > 0.0 and not _sequence_prefix_is_finite(
      committed_command_time_angles_deg,
      committed_count,
    ):
      return self.result
    mapping = live_mapping if live_mapping is not None and live_mapping.valid else self.nominal_mapping

    try:
      reactive_state = initial_state
      reactive_previous = previous_counts
      steering_request_counter = steering_request_fault_avoidance_counter
      for index in range(HORIZON_SAMPLE_COUNT):
        speed = float(planned_speeds_mps[index])
        parameters = self.profile.parameters_at(speed).parameters
        effective_offset = offset + parameters.lateral_accel_offset_correction_mps2
        self.reactive_angles_deg[index] = reactive_state.angle_deg
        self.reactive_rates_deg_s[index] = reactive_state.rate_deg_s
        (
          steering_request_counter,
          steering_request_active,
        ) = apply_steering_request_fault_avoidance(
          self._command_time_angle(
            index=index,
            current_steering_angle_deg=current_steering_angle,
            transport_delay_s=transport_delay,
            committed_command_time_angles_deg=(committed_command_time_angles_deg),
            planned_speeds_mps=planned_speeds_mps,
            mapping=mapping,
            lateral_accel_offset_mps2=offset,
            disturbance_torque=disturbance,
          ),
          lateral_active,
          steering_request_counter,
        )
        self.steering_request_active[index] = steering_request_active
        self.steering_request_counters[index] = steering_request_counter
        if not steering_request_active and self.result.first_request_suppression_index < 0:
          self.result.first_request_suppression_index = index
        inverse = compute_inverse_torque(
          reactive_state,
          float(desired_curvatures[index]),
          float(desired_angles_deg[index]),
          float(desired_rates_deg_s[index]),
          float(desired_accelerations_deg_s2[index]),
          speed,
          mapping.roll_rad,
          effective_offset,
          parameters,
          self.tracking_policy,
          disturbance,
        )
        self.raw_torques[index] = inverse.raw_torque
        requested = self._bounded_request_counts(inverse.raw_torque)
        if requested is None:
          raise ValueError("reactive request is not representable")
        slack_counts = round((1.0 - self.confidences[index]) * self.horizon_policy.maximum_torque_slack * self.limits.steer_max)
        slack_counts = min(max(slack_counts, 0), self.limits.steer_max)
        if requested > 0:
          self.band_lower_counts[index] = max(0, requested - slack_counts)
          self.band_upper_counts[index] = requested
          reachability_counts = self.band_lower_counts[index]
        elif requested < 0:
          self.band_lower_counts[index] = requested
          self.band_upper_counts[index] = min(0, requested + slack_counts)
          reachability_counts = self.band_upper_counts[index]
        else:
          reachability_counts = 0
        self.reachability_torques[index] = reachability_counts / self.limits.steer_max
        if index == 0:
          reactive = apply_torque_envelope_counts(
            self.limits,
            requested,
            reactive_previous,
            driver,
          )
        else:
          reactive = self._apply_neutral_envelope(requested, reactive_previous)
        self.reactive_counts[index] = reactive
        reactive_previous = reactive
        if index + 1 < HORIZON_SAMPLE_COUNT:
          reactive_state = self._step(
            reactive_state,
            reactive,
            steering_request_active,
            speed,
            mapping,
            effective_offset,
            parameters,
            disturbance,
          ).state

      projection = self.projector.update(
        ideal_torques=self.reachability_torques,
        previous_applied_counts=previous_counts,
        driver_torque=driver,
        steering_pressed=steering_pressed,
      )
      if not projection.valid or projection.status in (
        ReachabilityStatus.INVALID_INPUT,
        ReachabilityStatus.ENVELOPE_MISMATCH,
      ):
        self.result.status = HorizonStatus.PLANT_FAILURE
        return self.result
      driver_suppressed = projection.status == ReachabilityStatus.DRIVER_SUPPRESSED
      selected_first = self.reactive_counts[0]
      first_urgency = 0.0
      selected_next_angle = self.reactive_angles_deg[1]
      selected_next_rate = self.reactive_rates_deg_s[1]
      selection_lead_constrained = False
      first_parameters = self.profile.parameters_at(float(planned_speeds_mps[0])).parameters
      if not driver_suppressed:
        effective_offset = offset + first_parameters.lateral_accel_offset_correction_mps2
        selected_first, first_urgency, selected_next_angle, selected_next_rate, selection_lead_constrained = self._select_first_count(
          initial_state=initial_state,
          reactive_counts=self.reactive_counts[0],
          witness_counts=self.projector.witness_counts[0],
          previous_counts=previous_counts,
          driver_torque=driver,
          raw_torque=self.raw_torques[0],
          desired_angles_deg=desired_angles_deg,
          desired_rates_deg_s=desired_rates_deg_s,
          speed_mps=float(planned_speeds_mps[0]),
          mapping=mapping,
          lateral_accel_offset=effective_offset,
          parameters=first_parameters,
          disturbance_torque=disturbance,
          steering_request_active=self.steering_request_active[0],
        )
      position_tolerance = max(
        first_parameters.rack_rate_resolution_deg_s * self.horizon_policy.no_lead_position_tolerance_s,
        _NUMERICAL_POSITION_TOLERANCE_DEG,
      )
      rate_tolerance = max(
        first_parameters.rack_rate_resolution_deg_s * self.horizon_policy.no_lead_rate_tolerance_quanta,
        _NUMERICAL_RATE_TOLERANCE_DEG_S,
      )
      direction = self._movement_direction(
        desired_angles_deg,
        desired_rates_deg_s,
        initial_state,
        position_tolerance,
      )
      maximum_lead, maximum_rate_lead = self._path_leads(
        RackState(
          selected_next_angle,
          selected_next_rate,
          selected_first / self.limits.steer_max,
        ),
        direction=direction,
        desired_angle_deg=float(desired_angles_deg[1]),
        desired_rate_deg_s=float(desired_rates_deg_s[1]),
      )
      lead_constrained = int(
        selection_lead_constrained
        or maximum_lead > position_tolerance + _NUMERICAL_POSITION_TOLERANCE_DEG
        or maximum_rate_lead > rate_tolerance + _NUMERICAL_RATE_TOLERANCE_DEG_S
      )
    except (TypeError, ValueError, OverflowError, IndexError):
      # All configuration objects were validated at construction. Failures in
      # this block therefore originate in the runtime path/reference payload.
      self.result.status = HorizonStatus.INVALID_INPUT
      return self.result

    self.result.raw_torque = self.raw_torques[0]
    self.result.reactive_counts = self.reactive_counts[0]
    self.result.reactive_torque = self.reactive_counts[0] / self.limits.steer_max
    self.result.planned_counts = selected_first
    self.result.planned_torque = selected_first / self.limits.steer_max
    self.result.preparation_active = selected_first != self.reactive_counts[0]
    self.result.preparation_scheduled = projection.preparation_active_now or projection.preparation_scheduled_later
    self.result.driver_suppressed = driver_suppressed
    self.result.future_band_reachable = projection.witness_sequence_reachable and projection.authored_sequence_exactly_reachable
    self.result.first_unreachable_index = projection.first_authored_miss_index
    self.result.first_unreachable_time_s = -1.0 if projection.first_authored_miss_time_s is None else projection.first_authored_miss_time_s
    self.result.maximum_band_residual_counts = projection.maximum_absolute_authored_residual_counts
    self.result.maximum_path_lead_deg = maximum_lead
    self.result.maximum_path_rate_lead_deg_s = maximum_rate_lead
    self.result.path_lead_constrained_samples = lead_constrained
    self.result.maximum_authority_required = any(abs(raw) >= 1.0 for raw in self.raw_torques)
    raw_first = self.raw_torques[0]
    self.result.maximum_authority_active = abs(raw_first) >= 1.0 and selected_first == apply_torque_envelope_counts(
      self.limits,
      int(math.copysign(self.limits.steer_max, raw_first)),
      previous_counts,
      driver,
    )
    self.result.maximum_urgency = first_urgency
    self.result.valid = True
    self.result.status = (
      HorizonStatus.DRIVER_OVERRIDE if driver_suppressed else (HorizonStatus.OK if self.result.future_band_reachable else HorizonStatus.FUTURE_CONSTRAINED)
    )
    return self.result

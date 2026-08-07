"""Immutable offroad evidence contract for BLaTv2 behavior learning.

This module describes what happened; it does not judge the controller and it
cannot actuate or persist anything.  In particular, lane-line placement is
deliberately absent.  The immutable model reference is the target, while the
measured rack and curvature fields describe delivery of that target.

Auto-logger events are locators for useful windows, never truth labels.  A
confirmed driver intervention bookmarks the clean context preceding contact;
the contact frame and every later frame in that window are ineligible for
delivered-behavior scoring.  The intervention itself is not a positive or
negative vote.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import math
import re
from typing import Any

from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import (
  RackMappingSnapshot,
  map_reference,
)
from openpilot.selfdrive.controls.lib.blatv2.reference import sample_reference


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
BEHAVIOR_SCENARIO_PROVENANCE_SCHEMA_VERSION = 1


class ManeuverPhase(StrEnum):
  STRAIGHT_QUASI_STEADY = "straight_quasi_steady"
  TURN_IN = "turn_in"
  HOLD = "hold"
  RELEASE_UNWIND = "release_unwind"
  DIRECT_HANDOFF = "direct_handoff"


class ManeuverClass(StrEnum):
  STRAIGHT = "straight"
  CURVE = "curve"
  TURN = "turn"
  LANE_CHANGE = "lane_change"
  DIRECT_HANDOFF = "direct_handoff"


def _finite(name: str, value: float) -> None:
  if not math.isfinite(value):
    raise ValueError(f"{name} must be finite")


def _canonical_number(value: float) -> float:
  """Normalize signed zero before canonical JSON encoding."""
  return 0.0 if value == 0.0 else value


def canonical_json(payload: object) -> str:
  """Return deterministic JSON; non-finite floats are rejected."""
  return json.dumps(
    payload,
    allow_nan=False,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
  )


@dataclass(frozen=True, slots=True)
class BehaviorSourceIdentity:
  """Exact controller and source identity for one replay population."""

  controller_name: str
  controller_artifact_sha256: str
  source_openpilot_commit: str
  opendbc_commit: str
  panda_commit: str
  evidence_schema_version: int

  def __post_init__(self) -> None:
    if not self.controller_name.strip():
      raise ValueError("controller_name must not be empty")
    if _SHA256_RE.fullmatch(self.controller_artifact_sha256) is None:
      raise ValueError("controller_artifact_sha256 must be lowercase SHA-256")
    if _COMMIT_RE.fullmatch(self.source_openpilot_commit) is None:
      raise ValueError("source_openpilot_commit must be a full lowercase commit")
    if _COMMIT_RE.fullmatch(self.opendbc_commit) is None:
      raise ValueError("opendbc_commit must be a full lowercase commit")
    if _COMMIT_RE.fullmatch(self.panda_commit) is None:
      raise ValueError("panda_commit must be a full lowercase commit")
    if self.evidence_schema_version <= 0:
      raise ValueError("evidence_schema_version must be positive")

  def to_dict(self) -> dict[str, Any]:
    return {
      "controllerArtifactSha256": self.controller_artifact_sha256,
      "controllerName": self.controller_name,
      "evidenceSchemaVersion": self.evidence_schema_version,
      "opendbcCommit": self.opendbc_commit,
      "pandaCommit": self.panda_commit,
      "sourceOpenpilotCommit": self.source_openpilot_commit,
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())

  @property
  def sha256(self) -> str:
    return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class BehaviorScenarioProvenance:
  """Recorded source and immutable input identity for one replay scenario.

  A scenario supplies authenticated model, vehicle-state, calibration, and
  applied-torque context to counterfactual controllers.  Its recorded
  controller is provenance only: it is neither a quality label nor silently
  relabeled as the exact-stock opponent.  Compatibility is established by
  the strict scenario decoder before this object is emitted.
  """

  schema_version: int
  route_id: str
  route_evidence_sha256: str
  recorded_source: BehaviorSourceIdentity
  recorded_behavior_eligible: bool
  recorded_behavior_ineligible_reason: str
  vehicle_identity: str
  runtime_identity: str
  preparation_cache_key: str

  def __post_init__(self) -> None:
    if self.schema_version != BEHAVIOR_SCENARIO_PROVENANCE_SCHEMA_VERSION:
      raise ValueError("behavior scenario provenance schema is incompatible")
    if not self.route_id.strip():
      raise ValueError("scenario route_id must not be empty")
    for name, value in (
      ("route_evidence_sha256", self.route_evidence_sha256),
      ("runtime_identity", self.runtime_identity),
      ("preparation_cache_key", self.preparation_cache_key),
    ):
      if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    if not isinstance(self.recorded_source, BehaviorSourceIdentity):
      raise TypeError("recorded_source must be a BehaviorSourceIdentity")
    if type(self.recorded_behavior_eligible) is not bool:
      raise TypeError("recorded_behavior_eligible must be boolean")
    if not self.recorded_behavior_ineligible_reason.strip():
      raise ValueError("recorded behavior reason must not be empty")
    if self.recorded_behavior_eligible != (
      self.recorded_behavior_ineligible_reason == "eligible"
    ):
      raise ValueError("recorded behavior eligibility and reason disagree")
    if not self.vehicle_identity.strip():
      raise ValueError("scenario vehicle identity must not be empty")

  def to_dict(self) -> dict[str, Any]:
    return {
      "preparationCacheKey": self.preparation_cache_key,
      "recordedBehaviorEligible": self.recorded_behavior_eligible,
      "recordedBehaviorIneligibleReason": self.recorded_behavior_ineligible_reason,
      "recordedSource": self.recorded_source.to_dict(),
      "routeEvidenceSha256": self.route_evidence_sha256,
      "routeId": self.route_id,
      "runtimeIdentity": self.runtime_identity,
      "schemaVersion": self.schema_version,
      "vehicleIdentity": self.vehicle_identity,
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())

  @property
  def sha256(self) -> str:
    return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class BehaviorScenarioSetIdentity:
  """Order-sensitive experiment identity over scenario provenance."""

  sources: tuple[BehaviorScenarioProvenance, ...]

  def __post_init__(self) -> None:
    if type(self.sources) is not tuple or not self.sources:
      raise ValueError("behavior scenario set must not be empty")
    if any(not isinstance(source, BehaviorScenarioProvenance) for source in self.sources):
      raise TypeError("behavior scenario sources have the wrong type")
    route_ids = tuple(source.route_id for source in self.sources)
    if len(set(route_ids)) != len(route_ids):
      raise ValueError("behavior scenario route IDs must be unique")

  def to_dict(self) -> dict[str, Any]:
    return {
      "scenarioSources": [source.to_dict() for source in self.sources],
      "schemaVersion": BEHAVIOR_SCENARIO_PROVENANCE_SCHEMA_VERSION,
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())

  @property
  def sha256(self) -> str:
    return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SparseModelBehaviorIntent:
  """One sparse, raw model publication on its native time grid.

  This object deliberately contains no rack-space values.  Reference and rack
  mapping depend on the exact 100-Hz witness speed, delay, and live vehicle
  mapping; precomputing them once at model publication time would make replay
  physically wrong and controller-dependent.
  """

  plan_origin_mono_time_ns: int
  publication_mono_time_ns: int
  model_frame_id: int
  plan_valid: bool
  scalar_curvature_1pm: float
  scalar_action_plan_s: float
  native_times_s: tuple[float, ...]
  orientation_rates_z: tuple[float, ...]
  velocities_x: tuple[float, ...]

  def __post_init__(self) -> None:
    if (
      self.plan_origin_mono_time_ns < 0
      or self.publication_mono_time_ns < 0
      or self.model_frame_id < 0
    ):
      raise ValueError("model timestamps must be non-negative")
    if self.plan_origin_mono_time_ns > self.publication_mono_time_ns:
      raise ValueError("model plan origin cannot follow publication")
    if type(self.plan_valid) is not bool:
      raise TypeError("plan_valid must be boolean")
    scalar_values = (
      self.scalar_curvature_1pm,
      self.scalar_action_plan_s,
    )
    if not all(math.isfinite(value) for value in scalar_values):
      raise ValueError("sparse model scalar intent values must be finite")
    if self.scalar_action_plan_s < 0.0:
      raise ValueError("scalar action time must be non-negative")
    if self.plan_valid:
      count = len(self.native_times_s)
      if count == 0 or any(
        len(values) != count
        for values in (
          self.orientation_rates_z,
          self.velocities_x,
        )
      ):
        raise ValueError("valid sparse model arrays must be non-empty and equal length")
      values = (
        *self.native_times_s,
        *self.orientation_rates_z,
        *self.velocities_x,
      )
      if not all(math.isfinite(value) for value in values):
        raise ValueError("valid sparse model arrays must be finite")
      if self.native_times_s[0] < 0.0 or any(
        right <= left
        for left, right in zip(
          self.native_times_s,
          self.native_times_s[1:],
          strict=False,
        )
      ):
        raise ValueError("valid model sample times must be non-negative and increasing")


@dataclass(frozen=True, slots=True)
class BehaviorReferenceAtControl:
  """Exact target derived at one control witness from raw model intent."""

  model_publication_mono_time_ns: int
  plan_time_now_s: float
  physical_effect_plan_s: float
  scalar_curvature_1pm: float
  anchored_curvature_1pm: float
  anchored_curvature_rate_1pm_s: float
  anchored_curvature_accel_1pm_s2: float
  desired_rack_angle_deg: float
  desired_rack_rate_deg_s: float
  desired_rack_accel_deg_s2: float
  valid: bool

  def __post_init__(self) -> None:
    if self.model_publication_mono_time_ns < 0:
      raise ValueError("model publication time must be non-negative")
    values = (
      self.plan_time_now_s,
      self.physical_effect_plan_s,
      self.scalar_curvature_1pm,
      self.anchored_curvature_1pm,
      self.anchored_curvature_rate_1pm_s,
      self.anchored_curvature_accel_1pm_s2,
      self.desired_rack_angle_deg,
      self.desired_rack_rate_deg_s,
      self.desired_rack_accel_deg_s2,
    )
    if not all(math.isfinite(value) for value in values):
      raise ValueError("per-control behavior reference must be finite")
    if self.plan_time_now_s < 0.0 or self.physical_effect_plan_s < self.plan_time_now_s:
      raise ValueError("per-control behavior reference times are invalid")


@dataclass(frozen=True, slots=True)
class EventLocator:
  """Committed logger context used only to locate physical behavior."""

  event_type: str
  occurred_mono_time_ns: int
  analysis_window_before_s: float
  analysis_window_after_s: float
  severity: str

  def __post_init__(self) -> None:
    if not self.event_type.strip() or not self.severity.strip():
      raise ValueError("event locator text fields must not be empty")
    if self.occurred_mono_time_ns < 0:
      raise ValueError("occurred_mono_time_ns must be non-negative")
    for name, value in (
      ("analysis_window_before_s", self.analysis_window_before_s),
      ("analysis_window_after_s", self.analysis_window_after_s),
    ):
      _finite(name, value)
      if value < 0.0:
        raise ValueError(f"{name} must be non-negative")

  def to_dict(self) -> dict[str, Any]:
    return {
      "analysisWindowAfterS": _canonical_number(self.analysis_window_after_s),
      "analysisWindowBeforeS": _canonical_number(self.analysis_window_before_s),
      "eventType": self.event_type,
      "occurredMonoTimeNs": self.occurred_mono_time_ns,
      "severity": self.severity,
    }


@dataclass(frozen=True, slots=True)
class BehaviorSample:
  """One canonical controller-response sample.

  The scalar is the model's validated placement action.  ``desired`` and
  ``anchored`` deliberately name the same scalar-anchored target supplied to
  the controller; the raw planner trajectory remains only in
  :class:`SparseModelBehaviorIntent` as shape input to the stateless reference
  transform.  Keeping those two sample fields equal prevents metric code from
  accidentally promoting the raw planner into a second placement authority.
  None of these values may be reconstructed from lane lines.
  """

  mono_time_ns: int
  route_time_s: float
  speed_mps: float
  scalar_curvature_1pm: float
  desired_curvature_1pm: float
  anchored_curvature_1pm: float
  desired_rack_angle_deg: float
  desired_rack_rate_deg_s: float
  desired_rack_accel_deg_s2: float
  measured_curvature_1pm: float
  measured_rack_angle_deg: float
  measured_rack_rate_deg_s: float
  measured_rack_accel_deg_s2: float
  raw_requested_torque: float
  envelope_applied_torque: float
  torque_headroom: float
  actuator_constrained: bool
  lateral_active: bool
  inputs_valid: bool
  steering_pressed: bool
  controller_fault: bool
  driver_intervention_onset: bool

  def __post_init__(self) -> None:
    if self.mono_time_ns < 0:
      raise ValueError("mono_time_ns must be non-negative")
    numeric_fields = (
      "route_time_s",
      "speed_mps",
      "scalar_curvature_1pm",
      "desired_curvature_1pm",
      "anchored_curvature_1pm",
      "desired_rack_angle_deg",
      "desired_rack_rate_deg_s",
      "desired_rack_accel_deg_s2",
      "measured_curvature_1pm",
      "measured_rack_angle_deg",
      "measured_rack_rate_deg_s",
      "measured_rack_accel_deg_s2",
      "raw_requested_torque",
      "envelope_applied_torque",
      "torque_headroom",
    )
    for name in numeric_fields:
      _finite(name, getattr(self, name))
    if self.route_time_s < 0.0 or self.speed_mps < 0.0:
      raise ValueError("time and speed must be non-negative")
    if self.torque_headroom < 0.0:
      raise ValueError("torque_headroom must be non-negative")

  @property
  def physically_valid(self) -> bool:
    return self.lateral_active and self.inputs_valid and not self.controller_fault

  @property
  def delivered_frame_eligible(self) -> bool:
    return self.physically_valid and not self.steering_pressed and not self.driver_intervention_onset

  def to_dict(self) -> dict[str, Any]:
    return {
      "actuatorConstrained": self.actuator_constrained,
      "anchoredCurvature1pm": _canonical_number(self.anchored_curvature_1pm),
      "controllerFault": self.controller_fault,
      "desiredCurvature1pm": _canonical_number(self.desired_curvature_1pm),
      "desiredRackAccelDegS2": _canonical_number(self.desired_rack_accel_deg_s2),
      "desiredRackAngleDeg": _canonical_number(self.desired_rack_angle_deg),
      "desiredRackRateDegS": _canonical_number(self.desired_rack_rate_deg_s),
      "driverInterventionOnset": self.driver_intervention_onset,
      "envelopeAppliedTorque": _canonical_number(self.envelope_applied_torque),
      "inputsValid": self.inputs_valid,
      "lateralActive": self.lateral_active,
      "measuredCurvature1pm": _canonical_number(self.measured_curvature_1pm),
      "measuredRackAccelDegS2": _canonical_number(self.measured_rack_accel_deg_s2),
      "measuredRackAngleDeg": _canonical_number(self.measured_rack_angle_deg),
      "measuredRackRateDegS": _canonical_number(self.measured_rack_rate_deg_s),
      "monoTimeNs": self.mono_time_ns,
      "rawRequestedTorque": _canonical_number(self.raw_requested_torque),
      "routeTimeS": _canonical_number(self.route_time_s),
      "scalarCurvature1pm": _canonical_number(self.scalar_curvature_1pm),
      "speedMps": _canonical_number(self.speed_mps),
      "steeringPressed": self.steering_pressed,
      "torqueHeadroom": _canonical_number(self.torque_headroom),
    }


@dataclass(frozen=True, slots=True)
class BehaviorControlResponse:
  """100-Hz measured/requested fields paired with a sparse model index."""

  mono_time_ns: int
  route_time_s: float
  speed_mps: float
  transport_delay_s: float
  live_rack_mapping: RackMappingSnapshot | None
  nominal_rack_mapping: RackMappingSnapshot
  measured_curvature_1pm: float
  measured_rack_angle_deg: float
  measured_rack_rate_deg_s: float
  measured_rack_accel_deg_s2: float
  raw_requested_torque: float
  envelope_applied_torque: float
  torque_headroom: float
  actuator_constrained: bool
  lateral_active: bool
  inputs_valid: bool
  steering_pressed: bool
  controller_fault: bool
  driver_intervention_onset: bool

  def __post_init__(self) -> None:
    if self.mono_time_ns < 0:
      raise ValueError("control witness must be non-negative")
    for name in (
      "route_time_s",
      "speed_mps",
      "transport_delay_s",
      "measured_curvature_1pm",
      "measured_rack_angle_deg",
      "measured_rack_rate_deg_s",
      "measured_rack_accel_deg_s2",
      "raw_requested_torque",
      "envelope_applied_torque",
      "torque_headroom",
    ):
      _finite(name, getattr(self, name))
    if (
      self.route_time_s < 0.0
      or self.speed_mps < 0.0
      or self.transport_delay_s < 0.0
      or self.torque_headroom < 0.0
    ):
      raise ValueError("control time, speed, delay, and headroom must be non-negative")
    if not isinstance(self.nominal_rack_mapping, RackMappingSnapshot):
      raise TypeError("nominal_rack_mapping must be a RackMappingSnapshot")
    if self.live_rack_mapping is not None and not isinstance(
      self.live_rack_mapping,
      RackMappingSnapshot,
    ):
      raise TypeError("live_rack_mapping must be a RackMappingSnapshot or None")


def derive_behavior_reference(
  model: SparseModelBehaviorIntent,
  response: BehaviorControlResponse,
) -> BehaviorReferenceAtControl:
  """Build the exact anchored/rack target at this 100-Hz witness.

  The query uses the same stateless reference and rack-mapping artifacts as
  live control.  The raw plan remains immutable, while speed, delay, roll,
  offset, steer ratio, and stiffness are sampled from this response frame.
  """
  if model.publication_mono_time_ns > response.mono_time_ns:
    raise ValueError("model publication cannot follow the controls witness")
  plan_time_now_s = (
    response.mono_time_ns - model.plan_origin_mono_time_ns
  ) * 1e-9
  if plan_time_now_s < 0.0:
    raise ValueError("model plan origin cannot follow the controls witness")
  physical_effect_plan_s = plan_time_now_s + response.transport_delay_s
  # Preserve an explicitly invalid publication as scalar-only evidence.  The
  # route artifact retains its native arrays for audit, but a downstream
  # consumer may never reinterpret them as a valid future path.
  native_times = model.native_times_s if model.plan_valid else ()
  orientation_rates = model.orientation_rates_z if model.plan_valid else ()
  velocities = model.velocities_x if model.plan_valid else ()
  reference = sample_reference(
    native_times_s=native_times,
    orientation_rates_z=orientation_rates,
    velocities_x=velocities,
    scalar_curvature=model.scalar_curvature_1pm,
    scalar_action_plan_s=model.scalar_action_plan_s,
    plan_time_now_s=plan_time_now_s,
    measured_v_ego=response.speed_mps,
    query_times_s=(physical_effect_plan_s,),
  )
  rack = map_reference(
    curvature=reference.curvatures[0],
    curvature_rate=reference.curvature_rates[0],
    curvature_acceleration=reference.curvature_accelerations[0],
    speed=reference.planned_speeds[0],
    speed_rate=reference.planned_speed_rates[0],
    speed_acceleration=reference.planned_speed_accelerations[0],
    live_snapshot=response.live_rack_mapping,
    nominal_snapshot=response.nominal_rack_mapping,
  )
  return BehaviorReferenceAtControl(
    model_publication_mono_time_ns=model.publication_mono_time_ns,
    plan_time_now_s=plan_time_now_s,
    physical_effect_plan_s=physical_effect_plan_s,
    scalar_curvature_1pm=model.scalar_curvature_1pm,
    anchored_curvature_1pm=reference.curvatures[0],
    anchored_curvature_rate_1pm_s=reference.curvature_rates[0],
    anchored_curvature_accel_1pm_s2=reference.curvature_accelerations[0],
    desired_rack_angle_deg=rack.angle_deg,
    desired_rack_rate_deg_s=rack.rate_deg_s,
    desired_rack_accel_deg_s2=rack.acceleration_deg_s2,
    valid=reference.valid and rack.valid,
  )


def assemble_behavior_sample(
  reference: BehaviorReferenceAtControl,
  response: BehaviorControlResponse,
) -> BehaviorSample:
  """Associate an exact per-control reference with its response witness."""
  if reference.model_publication_mono_time_ns > response.mono_time_ns:
    raise ValueError("model publication cannot follow the controls witness")
  return BehaviorSample(
    mono_time_ns=response.mono_time_ns,
    route_time_s=response.route_time_s,
    speed_mps=response.speed_mps,
    scalar_curvature_1pm=reference.scalar_curvature_1pm,
    # The controller's desired curvature is the scalar-anchored target.  Raw
    # planner shape remains available in SparseModelBehaviorIntent and is not
    # promoted to a second path-placement authority here.
    desired_curvature_1pm=reference.anchored_curvature_1pm,
    anchored_curvature_1pm=reference.anchored_curvature_1pm,
    desired_rack_angle_deg=reference.desired_rack_angle_deg,
    desired_rack_rate_deg_s=reference.desired_rack_rate_deg_s,
    desired_rack_accel_deg_s2=reference.desired_rack_accel_deg_s2,
    measured_curvature_1pm=response.measured_curvature_1pm,
    measured_rack_angle_deg=response.measured_rack_angle_deg,
    measured_rack_rate_deg_s=response.measured_rack_rate_deg_s,
    measured_rack_accel_deg_s2=response.measured_rack_accel_deg_s2,
    raw_requested_torque=response.raw_requested_torque,
    envelope_applied_torque=response.envelope_applied_torque,
    torque_headroom=response.torque_headroom,
    actuator_constrained=response.actuator_constrained,
    lateral_active=response.lateral_active,
    inputs_valid=response.inputs_valid and reference.valid,
    steering_pressed=response.steering_pressed,
    controller_fault=response.controller_fault,
    driver_intervention_onset=response.driver_intervention_onset,
  )


@dataclass(frozen=True, slots=True)
class BehaviorWindow:
  """One phase window with deterministic intervention censoring."""

  route_id: str
  window_id: str
  source: BehaviorSourceIdentity
  maneuver_class: ManeuverClass
  phase: ManeuverPhase
  samples: tuple[BehaviorSample, ...]
  event_locators: tuple[EventLocator, ...] = ()

  def __post_init__(self) -> None:
    if not self.route_id.strip() or not self.window_id.strip():
      raise ValueError("route_id and window_id must not be empty")
    if not self.samples:
      raise ValueError("behavior window must contain samples")
    if any(
      right.mono_time_ns <= left.mono_time_ns
      or right.route_time_s <= left.route_time_s
      for left, right in zip(self.samples, self.samples[1:], strict=False)
    ):
      raise ValueError("behavior samples must be strictly time ordered")
    event_keys = tuple(
      (event.occurred_mono_time_ns, event.event_type, event.severity)
      for event in self.event_locators
    )
    if tuple(sorted(event_keys)) != event_keys:
      raise ValueError("event locators must be in canonical timestamp order")

  @property
  def intervention_mono_time_ns(self) -> int | None:
    for sample in self.samples:
      if sample.driver_intervention_onset:
        return sample.mono_time_ns
    return None

  @property
  def clean_pre_intervention_samples(self) -> tuple[BehaviorSample, ...]:
    """Eligible response frames strictly before first driver contact."""
    clean: list[BehaviorSample] = []
    for sample in self.samples:
      if sample.driver_intervention_onset:
        break
      if sample.delivered_frame_eligible:
        clean.append(sample)
    return tuple(clean)

  @property
  def intervention_is_quality_vote(self) -> bool:
    """Always false: context determines quality, not contact alone."""
    return False

  def to_dict(self) -> dict[str, Any]:
    return {
      "eventLocators": [event.to_dict() for event in self.event_locators],
      "maneuverClass": self.maneuver_class.value,
      "phase": self.phase.value,
      "routeId": self.route_id,
      "samples": [sample.to_dict() for sample in self.samples],
      "source": self.source.to_dict(),
      "windowId": self.window_id,
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())

  @property
  def sha256(self) -> str:
    return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

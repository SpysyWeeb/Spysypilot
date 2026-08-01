"""Exact, self-anchored behavior replay for the modular learning transaction.

This module is the production numerical adapter between immutable
``RouteEvidenceArtifact`` bytes and :mod:`behavior_transaction`.  It has two
controller request producers, but deliberately only one episode simulator:

* the modular producer calls the existing :class:`ModularControllerCore`;
* the stock producer constructs the source :class:`LatControlTorque` through
  :func:`fresh_stock_torque_controller` and preserves controlsd's
  ``clip_curvature``/VehicleModel/live-torque state; and
* both requests pass through the exact count-space opendbc envelope and the
  same :func:`step_plant` implementation.

Recorded rack state and the exact ``torqueOutputCan`` count initialize each
inactive-to-active lateral episode.  They are never consumed again during
that active episode.  Inactive/manual frames are context only.  A confirmed
driver intervention censors the remainder of the active episode, without
re-anchoring it, and eligibility resumes only at a later inactive-to-active
boundary.

Counterfactual observer learning is intentionally disabled.  A twin learning
from its own prediction has identically zero innovation and would merely make
the replay look stateful without adding physical information.  This choice is
explicit and shared by every modular candidate.

There is no Params access, activation, persistence, or actuation path here.
Every callback constructs fresh whole-route/controller state and is safe to
invoke concurrently.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import base64
import json
import math
from types import SimpleNamespace
from typing import Any

from opendbc.car.structs import car
from opendbc.car.vehicle_model import VehicleModel

from openpilot.cereal import messaging
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.blatv2.actuator import (
  RuntimeTorqueLimits,
  apply_torque_envelope_counts,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_coordinator import (
  ReplayCoreIdentity,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import (
  BehaviorSourceIdentity,
  EventLocator,
  SparseModelBehaviorIntent,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_transaction import (
  BehaviorReplayCore,
  CanonicalBehaviorControlInput,
  ControllerFrameOutput,
  ControllerReplayRequest,
  DecodedBehaviorRoute,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_profile import (
  VehicleCalibrationProfile,
)
from openpilot.selfdrive.controls.lib.blatv2.core import (
  CoreStatus,
  ModularControllerCore,
)
from openpilot.selfdrive.controls.lib.blatv2.intent import (
  INTENT_CAPACITY,
  adapt_model_intent_into,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_runtime import (
  build_detected_runtime_bundle,
)
from openpilot.selfdrive.controls.lib.blatv2.plant import (
  RackState,
  TrackingPolicy,
  step_plant,
)
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import (
  RackMappingSnapshot,
  curvature_from_measured_angle,
)
from openpilot.selfdrive.controls.lib.blatv2.route_evidence import (
  ROUTE_EVIDENCE_VERSION,
  RouteEvidenceArtifact,
)
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  ProvisionalRackDynamics,
  RuntimeVehicleBundle,
)
from openpilot.selfdrive.controls.lib.blatv2.stock_bootstrap import (
  fresh_stock_torque_controller,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  VehicleProfile,
  compose_controller_profile,
)
from openpilot.selfdrive.controls.lib.drive_helpers import clip_curvature


BEHAVIOR_REPLAY_INPUT_SCHEMA_VERSION = 1
# Importing modeld just to obtain this scalar also imports the device-only
# vision IPC extension.  Keep the source value local so replay remains usable
# in the off-device harness; the exact-stock tests pin it against modeld.py,
# and the replay identity pins the source commit.
SOURCE_LAT_SMOOTH_SECONDS = 0.0
_CORE_INPUT_KEYS = frozenset({
  "appliedCountValid",
  "carParamsBase64",
  "controlCadenceValid",
  "driverTorque",
  "lateralDelayInputsValid",
  "lateralDelayPublicationIndex",
  "lateralDelaySeconds",
  "lateralManeuverDesiredCurvature",
  "lateralManeuverPlanPublicationIndex",
  "lateralManeuverPlanValid",
  "interventionOnsetUncertain",
  "liveParametersInputsValid",
  "liveTorqueHealthExact",
  "liveTorqueFriction",
  "liveTorqueInputsValid",
  "liveTorqueLatAccelFactor",
  "liveTorqueLatAccelOffset",
  "liveTorqueParametersPublicationIndex",
  "liveTorqueUseParams",
  "modelFrameId",
  "modelMessageAlive",
  "modelMessageValid",
  "physicalRecordIndex",
  "recordedAppliedCounts",
  "recordedAppliedTorque",
  "recordedRackAccelerationDegS2",
  "recordedRackAngleDeg",
  "recordedRackRateDegS",
  "recordedRawRequestTorque",
  "schemaVersion",
  "stateSampleMonoTimeNs",
  "standstill",
  "steerRatio",
  "stiffnessFactor",
  "witnessResolved",
})
_UINT64_MAX = (1 << 64) - 1
_LIVE_CORE_STATUSES = (
  CoreStatus.OK,
  CoreStatus.DEGRADED_SCALAR_ONLY,
  CoreStatus.DEGRADED_NOMINAL_MAPPING,
)


class BehaviorReplayError(RuntimeError):
  """Route evidence cannot produce an exact comparable replay."""


def _canonical_json_bytes(payload: object) -> bytes:
  return json.dumps(
    payload,
    allow_nan=False,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
  ).encode("ascii")


def _require_exact_bool(name: str, value: object) -> bool:
  if type(value) is not bool:
    raise BehaviorReplayError(f"{name} must be a boolean")
  return value


def _require_exact_int(
  name: str,
  value: object,
  *,
  minimum: int,
  maximum: int,
) -> int:
  if type(value) is not int or not minimum <= value <= maximum:
    raise BehaviorReplayError(f"{name} is outside its integer domain")
  return value


def _require_finite(name: str, value: object) -> float:
  if type(value) not in (int, float):
    raise BehaviorReplayError(f"{name} must be numeric")
  selected = float(value)
  if not math.isfinite(selected):
    raise BehaviorReplayError(f"{name} must be finite")
  return selected


def _require_index(name: str, value: object) -> int:
  return _require_exact_int(
    name,
    value,
    minimum=-1,
    maximum=(1 << 31) - 1,
  )


@dataclass(frozen=True, slots=True)
class ReplayFrameInput:
  """Strict decoder-owned payload consumed by both controller factories."""

  physical_record_index: int
  state_sample_mono_time_ns: int
  model_frame_id: int
  recorded_rack_angle_deg: float
  recorded_rack_rate_deg_s: float
  recorded_rack_acceleration_deg_s2: float
  recorded_applied_torque: float
  recorded_applied_counts: int
  recorded_raw_request_torque: float
  driver_torque: float
  stiffness_factor: float
  steer_ratio: float
  live_torque_parameters_publication_index: int
  live_torque_lat_accel_factor: float
  live_torque_lat_accel_offset: float
  live_torque_friction: float
  lateral_delay_publication_index: int
  lateral_delay_seconds: float
  lateral_maneuver_plan_publication_index: int
  lateral_maneuver_desired_curvature: float
  applied_count_valid: bool
  witness_resolved: bool
  control_cadence_valid: bool
  model_message_valid: bool
  model_message_alive: bool
  live_parameters_inputs_valid: bool
  live_torque_health_exact: bool
  live_torque_inputs_valid: bool
  live_torque_use_params: bool
  lateral_delay_inputs_valid: bool
  lateral_maneuver_plan_valid: bool
  intervention_onset_uncertain: bool
  standstill: bool
  car_params_bytes: bytes | None = None

  def __post_init__(self) -> None:
    if self.physical_record_index < 0:
      raise ValueError("physical record index must be non-negative")
    if not 0 <= self.state_sample_mono_time_ns <= _UINT64_MAX:
      raise ValueError("state sample timestamp is outside UInt64")
    if self.model_frame_id < 0:
      raise ValueError("model frame ID must be non-negative")
    if not all(math.isfinite(value) for value in (
      self.recorded_rack_angle_deg,
      self.recorded_rack_rate_deg_s,
      self.recorded_rack_acceleration_deg_s2,
      self.recorded_applied_torque,
      self.recorded_raw_request_torque,
      self.driver_torque,
      self.stiffness_factor,
      self.steer_ratio,
      self.live_torque_lat_accel_factor,
      self.live_torque_lat_accel_offset,
      self.live_torque_friction,
      self.lateral_delay_seconds,
      self.lateral_maneuver_desired_curvature,
    )):
      raise ValueError("replay input values must be finite")
    if self.live_parameters_inputs_valid and (
      self.stiffness_factor <= 0.0 or self.steer_ratio <= 0.0
    ):
      raise ValueError("live rack mapping inputs must be positive")
    if (
      self.live_torque_inputs_valid
      and self.live_torque_use_params
      and self.live_torque_lat_accel_factor <= 0.0
    ):
      raise ValueError("live torque factor must be positive")
    if self.live_torque_friction < 0.0 or self.lateral_delay_seconds < 0.0:
      raise ValueError("friction and lateral delay must be non-negative")
    if any(index < -1 for index in (
      self.live_torque_parameters_publication_index,
      self.lateral_delay_publication_index,
      self.lateral_maneuver_plan_publication_index,
    )):
      raise ValueError("publication indices must use -1 or a valid index")
    if self.car_params_bytes is not None and type(self.car_params_bytes) is not bytes:
      raise TypeError("CarParams payload must be immutable bytes or None")

  def to_bytes(self) -> bytes:
    payload = {
      "appliedCountValid": self.applied_count_valid,
      "carParamsBase64": (
        None
        if self.car_params_bytes is None
        else base64.b64encode(self.car_params_bytes).decode("ascii")
      ),
      "controlCadenceValid": self.control_cadence_valid,
      "driverTorque": self.driver_torque,
      "lateralDelayInputsValid": self.lateral_delay_inputs_valid,
      "lateralDelayPublicationIndex": self.lateral_delay_publication_index,
      "lateralDelaySeconds": self.lateral_delay_seconds,
      "lateralManeuverDesiredCurvature": self.lateral_maneuver_desired_curvature,
      "lateralManeuverPlanPublicationIndex": self.lateral_maneuver_plan_publication_index,
      "lateralManeuverPlanValid": self.lateral_maneuver_plan_valid,
      "interventionOnsetUncertain": self.intervention_onset_uncertain,
      "liveParametersInputsValid": self.live_parameters_inputs_valid,
      "liveTorqueHealthExact": self.live_torque_health_exact,
      "liveTorqueFriction": self.live_torque_friction,
      "liveTorqueInputsValid": self.live_torque_inputs_valid,
      "liveTorqueLatAccelFactor": self.live_torque_lat_accel_factor,
      "liveTorqueLatAccelOffset": self.live_torque_lat_accel_offset,
      "liveTorqueParametersPublicationIndex": self.live_torque_parameters_publication_index,
      "liveTorqueUseParams": self.live_torque_use_params,
      "modelFrameId": self.model_frame_id,
      "modelMessageAlive": self.model_message_alive,
      "modelMessageValid": self.model_message_valid,
      "physicalRecordIndex": self.physical_record_index,
      "recordedAppliedCounts": self.recorded_applied_counts,
      "recordedAppliedTorque": self.recorded_applied_torque,
      "recordedRackAccelerationDegS2": self.recorded_rack_acceleration_deg_s2,
      "recordedRackAngleDeg": self.recorded_rack_angle_deg,
      "recordedRackRateDegS": self.recorded_rack_rate_deg_s,
      "recordedRawRequestTorque": self.recorded_raw_request_torque,
      "schemaVersion": BEHAVIOR_REPLAY_INPUT_SCHEMA_VERSION,
      "stateSampleMonoTimeNs": self.state_sample_mono_time_ns,
      "standstill": self.standstill,
      "steerRatio": self.steer_ratio,
      "stiffnessFactor": self.stiffness_factor,
      "witnessResolved": self.witness_resolved,
    }
    return _canonical_json_bytes(payload)

  @classmethod
  def from_bytes(cls, encoded: bytes) -> ReplayFrameInput:
    if type(encoded) is not bytes:
      raise TypeError("replay core input must be immutable bytes")
    try:
      payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
      raise BehaviorReplayError("replay core input is not valid JSON") from exc
    if type(payload) is not dict or set(payload) != _CORE_INPUT_KEYS:
      raise BehaviorReplayError("replay core input keys do not match the schema")
    if _canonical_json_bytes(payload) != encoded:
      raise BehaviorReplayError("replay core input is not canonical")
    version = _require_exact_int(
      "schemaVersion",
      payload["schemaVersion"],
      minimum=1,
      maximum=BEHAVIOR_REPLAY_INPUT_SCHEMA_VERSION,
    )
    if version != BEHAVIOR_REPLAY_INPUT_SCHEMA_VERSION:
      raise BehaviorReplayError("replay core input schema is incompatible")
    encoded_cp = payload["carParamsBase64"]
    if encoded_cp is not None and type(encoded_cp) is not str:
      raise BehaviorReplayError("CarParams payload must be base64 text or null")
    try:
      car_params_bytes = (
        None
        if encoded_cp is None
        else base64.b64decode(encoded_cp, validate=True)
      )
    except (ValueError, base64.binascii.Error) as exc:
      raise BehaviorReplayError("CarParams payload is not canonical base64") from exc
    if car_params_bytes is not None and base64.b64encode(car_params_bytes).decode("ascii") != encoded_cp:
      raise BehaviorReplayError("CarParams payload is not canonical base64")
    try:
      result = cls(
        physical_record_index=_require_exact_int(
          "physicalRecordIndex", payload["physicalRecordIndex"],
          minimum=0, maximum=(1 << 31) - 1,
        ),
        state_sample_mono_time_ns=_require_exact_int(
          "stateSampleMonoTimeNs", payload["stateSampleMonoTimeNs"],
          minimum=0, maximum=_UINT64_MAX,
        ),
        model_frame_id=_require_exact_int(
          "modelFrameId", payload["modelFrameId"],
          minimum=0, maximum=_UINT64_MAX,
        ),
        recorded_rack_angle_deg=_require_finite(
          "recordedRackAngleDeg", payload["recordedRackAngleDeg"],
        ),
        recorded_rack_rate_deg_s=_require_finite(
          "recordedRackRateDegS", payload["recordedRackRateDegS"],
        ),
        recorded_rack_acceleration_deg_s2=_require_finite(
          "recordedRackAccelerationDegS2", payload["recordedRackAccelerationDegS2"],
        ),
        recorded_applied_torque=_require_finite(
          "recordedAppliedTorque", payload["recordedAppliedTorque"],
        ),
        recorded_applied_counts=_require_exact_int(
          "recordedAppliedCounts", payload["recordedAppliedCounts"],
          minimum=-(1 << 31), maximum=(1 << 31) - 1,
        ),
        recorded_raw_request_torque=_require_finite(
          "recordedRawRequestTorque", payload["recordedRawRequestTorque"],
        ),
        driver_torque=_require_finite("driverTorque", payload["driverTorque"]),
        stiffness_factor=_require_finite("stiffnessFactor", payload["stiffnessFactor"]),
        steer_ratio=_require_finite("steerRatio", payload["steerRatio"]),
        live_torque_parameters_publication_index=_require_index(
          "liveTorqueParametersPublicationIndex",
          payload["liveTorqueParametersPublicationIndex"],
        ),
        live_torque_lat_accel_factor=_require_finite(
          "liveTorqueLatAccelFactor", payload["liveTorqueLatAccelFactor"],
        ),
        live_torque_lat_accel_offset=_require_finite(
          "liveTorqueLatAccelOffset", payload["liveTorqueLatAccelOffset"],
        ),
        live_torque_friction=_require_finite(
          "liveTorqueFriction", payload["liveTorqueFriction"],
        ),
        lateral_delay_publication_index=_require_index(
          "lateralDelayPublicationIndex", payload["lateralDelayPublicationIndex"],
        ),
        lateral_delay_seconds=_require_finite(
          "lateralDelaySeconds", payload["lateralDelaySeconds"],
        ),
        lateral_maneuver_plan_publication_index=_require_index(
          "lateralManeuverPlanPublicationIndex",
          payload["lateralManeuverPlanPublicationIndex"],
        ),
        lateral_maneuver_desired_curvature=_require_finite(
          "lateralManeuverDesiredCurvature",
          payload["lateralManeuverDesiredCurvature"],
        ),
        applied_count_valid=_require_exact_bool(
          "appliedCountValid", payload["appliedCountValid"],
        ),
        witness_resolved=_require_exact_bool(
          "witnessResolved", payload["witnessResolved"],
        ),
        control_cadence_valid=_require_exact_bool(
          "controlCadenceValid", payload["controlCadenceValid"],
        ),
        model_message_valid=_require_exact_bool(
          "modelMessageValid", payload["modelMessageValid"],
        ),
        model_message_alive=_require_exact_bool(
          "modelMessageAlive", payload["modelMessageAlive"],
        ),
        live_parameters_inputs_valid=_require_exact_bool(
          "liveParametersInputsValid", payload["liveParametersInputsValid"],
        ),
        live_torque_health_exact=_require_exact_bool(
          "liveTorqueHealthExact", payload["liveTorqueHealthExact"],
        ),
        live_torque_inputs_valid=_require_exact_bool(
          "liveTorqueInputsValid", payload["liveTorqueInputsValid"],
        ),
        live_torque_use_params=_require_exact_bool(
          "liveTorqueUseParams", payload["liveTorqueUseParams"],
        ),
        lateral_delay_inputs_valid=_require_exact_bool(
          "lateralDelayInputsValid", payload["lateralDelayInputsValid"],
        ),
        lateral_maneuver_plan_valid=_require_exact_bool(
          "lateralManeuverPlanValid", payload["lateralManeuverPlanValid"],
        ),
        intervention_onset_uncertain=_require_exact_bool(
          "interventionOnsetUncertain", payload["interventionOnsetUncertain"],
        ),
        standstill=_require_exact_bool("standstill", payload["standstill"]),
        car_params_bytes=car_params_bytes,
      )
    except (TypeError, ValueError) as exc:
      if isinstance(exc, BehaviorReplayError):
        raise
      raise BehaviorReplayError("replay core input values are invalid") from exc
    if result.to_bytes() != encoded:
      raise BehaviorReplayError("replay core input scalar encodings are not canonical")
    return result


@dataclass(frozen=True, slots=True)
class _RouteRuntime:
  car_params: Any
  car_interface: Any
  runtime_bundle: RuntimeVehicleBundle
  controller_profile: VehicleProfile


def _linked_publication(
  values: Sequence[Any],
  index: int,
  available: bool,
  *,
  name: str,
) -> Any | None:
  if available != (index >= 0):
    raise BehaviorReplayError(f"{name} availability and index disagree")
  if index < 0:
    return None
  if index >= len(values):
    raise BehaviorReplayError(f"{name} publication index is out of range")
  return values[index]


def _mapping_from_physical_frame(
  vehicle_model: VehicleModel,
  physical: Any,
) -> RackMappingSnapshot | None:
  valid = (
    physical.live_parameters_valid
    and physical.angle_offset_valid
    and physical.steer_ratio_valid
    and physical.stiffness_factor_valid
    and math.isfinite(physical.stiffness_factor)
    and physical.stiffness_factor > 0.0
    and math.isfinite(physical.steer_ratio)
    and physical.steer_ratio > 0.0
    and math.isfinite(physical.roll_rad)
    and math.isfinite(physical.angle_offset_deg)
  )
  if not valid:
    return None
  try:
    vehicle_model.update_params(
      max(physical.stiffness_factor, 0.1),
      max(physical.steer_ratio, 0.1),
    )
    return RackMappingSnapshot.from_vehicle_model(
      vehicle_model,
      roll_rad=physical.roll_rad,
      angle_offset_deg=physical.angle_offset_deg,
      valid=True,
    )
  except (TypeError, ValueError, OverflowError, ZeroDivisionError):
    return None


def _rack_accelerations(
  frames: tuple[Any, ...],
) -> tuple[tuple[float, bool], ...]:
  output: list[tuple[float, bool]] = []
  previous_time: int | None = None
  previous_rate = 0.0
  for frame in frames:
    acceleration = 0.0
    valid = False
    if previous_time is not None:
      gap_ns = frame.response_mono_ns - previous_time
      valid = 0 < gap_ns <= 15_000_000
      if valid:
        acceleration = (
          frame.steering_rate_deg_s - previous_rate
        ) / (gap_ns * 1e-9)
    output.append((acceleration, valid))
    previous_time = frame.response_mono_ns
    previous_rate = frame.steering_rate_deg_s
  return tuple(output)


def behavior_source_identity_from_route_artifact(
  artifact: RouteEvidenceArtifact,
) -> BehaviorSourceIdentity:
  """Project the one canonical behavioral source identity from route bytes.

  Cohort selection and route decoding must bind the exact same source.  Keep
  that projection here, next to the sole route-evidence decoder, so an
  offroad coordinator never grows a subtly different interpretation of the
  controller/build fields recorded in the artifact.
  """
  if type(artifact) is not RouteEvidenceArtifact:
    raise TypeError("behavior source projection requires RouteEvidenceArtifact")
  source = artifact.source_identity
  if not source.behavior_eligible:
    raise BehaviorReplayError(
      f"route evidence is behavior-ineligible: {source.behavior_ineligible_reason}",
    )
  return BehaviorSourceIdentity(
    controller_name=source.controller_source_kind,
    controller_artifact_sha256=source.controller_artifact_sha256,
    source_openpilot_commit=source.source_superproject_commit,
    opendbc_commit=source.source_opendbc_commit,
    panda_commit=source.source_panda_commit,
    evidence_schema_version=ROUTE_EVIDENCE_VERSION,
  )


def make_behavior_route_evidence_decoder(
  *,
  provisional_dynamics: ProvisionalRackDynamics,
  interface_registry: Mapping[str, type] | None = None,
) -> Any:
  """Return the sole strict decoder for the shared route artifact.

  The detected runtime is rebuilt from the artifact's exact CarParams and
  compared with the frozen physical profile.  No current-device CarParams,
  Params key, or mutable runtime bundle participates.
  """
  if not isinstance(provisional_dynamics, ProvisionalRackDynamics):
    raise TypeError("behavior decoder requires explicit provisional dynamics")
  registry = None if interface_registry is None else dict(interface_registry)

  def decode(
    artifact: object,
    physical_profile: VehicleCalibrationProfile,
  ) -> DecodedBehaviorRoute:
    if type(artifact) is not RouteEvidenceArtifact:
      raise TypeError("behavior decoder requires RouteEvidenceArtifact")
    if not isinstance(physical_profile, VehicleCalibrationProfile):
      raise TypeError("behavior decoder requires VehicleCalibrationProfile")
    source = artifact.source_identity
    recorded_source = behavior_source_identity_from_route_artifact(artifact)
    params = _decode_car_params(bytes(artifact.car_params_bytes))
    try:
      bundle, _, _ = build_detected_runtime_bundle(
        car_params=params,
        provisional_rack_dynamics=provisional_dynamics,
        interface_registry=registry,
      )
      controller_profile = compose_controller_profile(
        physical_profile,
        bundle.seed_profile,
      )
      physical_frames = tuple(artifact.iter_physical_frames())
    except Exception as exc:
      raise BehaviorReplayError(
        "route artifact cannot bind a verified physical runtime",
      ) from exc
    if (
      source.vehicle_identity != bundle.vehicle_identity
      or physical_profile.vehicle_identity != bundle.vehicle_identity
      or controller_profile.vehicle_identity != bundle.vehicle_identity
    ):
      raise BehaviorReplayError("route/profile/runtime vehicle identities differ")
    if len(physical_frames) != len(artifact.control_witnesses):
      raise BehaviorReplayError("physical/control evidence populations disagree")

    models = tuple(
      SparseModelBehaviorIntent(
        plan_origin_mono_time_ns=model.timestamp_eof_ns,
        publication_mono_time_ns=model.mono_time_ns,
        model_frame_id=model.frame_id,
        plan_valid=model.native_grid_valid,
        scalar_curvature_1pm=model.scalar_curvature,
        scalar_action_plan_s=model.desired_curvature_time_s,
        native_times_s=model.plan_times,
        orientation_rates_z=model.orientation_rate_z,
        velocities_x=model.velocity_x,
      )
      for model in artifact.model_publications
    )
    nominal_mapping = bundle.nominal_rack_mapping
    mapping_model = VehicleModel(params)
    accelerations = _rack_accelerations(physical_frames)
    controls: list[CanonicalBehaviorControlInput] = []
    cp_bytes = bytes(artifact.car_params_bytes)
    for index, (witness, physical, acceleration_fact) in enumerate(zip(
      artifact.control_witnesses,
      physical_frames,
      accelerations,
      strict=True,
    )):
      if witness.physical_record_index != index:
        raise BehaviorReplayError("control/physical link is non-canonical")
      model = _linked_publication(
        artifact.model_publications,
        witness.model_publication_index,
        witness.model_link_valid,
        name="model",
      )
      torque = _linked_publication(
        artifact.live_torque_parameters,
        witness.live_torque_parameters_index,
        witness.live_torque_parameters_available,
        name="live torque",
      )
      delay = _linked_publication(
        artifact.live_delays,
        witness.live_delay_index,
        witness.live_delay_available,
        name="live delay",
      )
      maneuver = _linked_publication(
        artifact.lateral_maneuver_plans,
        witness.lateral_maneuver_plan_index,
        witness.maneuver_plan_available,
        name="lateral maneuver",
      )
      for name, publication in (
        ("model", model),
        ("live torque", torque),
        ("live delay", delay),
        ("lateral maneuver", maneuver),
      ):
        if publication is not None and publication.mono_time_ns > witness.mono_time_ns:
          raise BehaviorReplayError(f"{name} link points to a future publication")
      live_mapping = _mapping_from_physical_frame(mapping_model, physical)
      live_parameters_inputs_valid = (
        live_mapping is not None
        and witness.live_parameters_mono_ns >= 0
      )
      torque_factor = (
        float(params.lateralTuning.torque.latAccelFactor)
        if torque is None
        else torque.lat_accel_factor
      )
      torque_offset = (
        float(params.lateralTuning.torque.latAccelOffset)
        if torque is None
        else torque.lat_accel_offset
      )
      torque_friction = (
        float(params.lateralTuning.torque.friction)
        if torque is None
        else torque.friction
      )
      frame_input = ReplayFrameInput(
        physical_record_index=index,
        state_sample_mono_time_ns=witness.state_sample_mono_ns,
        model_frame_id=0 if model is None else model.frame_id,
        recorded_rack_angle_deg=physical.steering_angle_deg,
        recorded_rack_rate_deg_s=physical.steering_rate_deg_s,
        recorded_rack_acceleration_deg_s2=acceleration_fact[0],
        recorded_applied_torque=physical.applied_torque,
        recorded_applied_counts=witness.torque_output_can_count,
        recorded_raw_request_torque=witness.raw_request_torque,
        driver_torque=physical.steering_torque,
        stiffness_factor=physical.stiffness_factor,
        steer_ratio=physical.steer_ratio,
        live_torque_parameters_publication_index=(
          witness.live_torque_parameters_index
        ),
        live_torque_lat_accel_factor=torque_factor,
        live_torque_lat_accel_offset=torque_offset,
        live_torque_friction=torque_friction,
        lateral_delay_publication_index=witness.live_delay_index,
        lateral_delay_seconds=(0.0 if delay is None else delay.lateral_delay_s),
        lateral_maneuver_plan_publication_index=(
          witness.lateral_maneuver_plan_index
        ),
        lateral_maneuver_desired_curvature=(
          0.0 if maneuver is None else maneuver.desired_curvature
        ),
        applied_count_valid=witness.torque_output_can_valid,
        witness_resolved=not witness.race_unresolved,
        control_cadence_valid=(
          not witness.gap_from_previous and (index == 0 or acceleration_fact[1])
        ),
        model_message_valid=(False if model is None else model.message_valid),
        model_message_alive=(model is not None and witness.model_message_alive),
        live_parameters_inputs_valid=live_parameters_inputs_valid,
        live_torque_health_exact=(
          witness.live_torque_parameters_health_exact
        ),
        # This is the reconstructed witness-time SubMaster all_checks result,
        # not an inference from the latest publication's payload bits.
        live_torque_inputs_valid=(
          witness.live_torque_parameters_checks_passed
        ),
        live_torque_use_params=(torque is not None and torque.use_params),
        lateral_delay_inputs_valid=(delay is not None and delay.message_valid),
        lateral_maneuver_plan_valid=(
          maneuver is not None and maneuver.message_valid
        ),
        intervention_onset_uncertain=witness.intervention_onset_uncertain,
        standstill=physical.standstill,
        car_params_bytes=cp_bytes if index == 0 else None,
      )
      physically_valid = (
        source.behavior_eligible
        and witness.message_valid
        and witness.inputs_valid
        and witness.model_link_valid
        and model is not None
        and model.message_valid
        and physical.inputs_valid
        and physical.can_valid
        and not physical.can_timeout
        and not witness.race_unresolved
        and not witness.gap_from_previous
      )
      controls.append(CanonicalBehaviorControlInput(
        mono_time_ns=witness.mono_time_ns,
        route_time_s=(
          witness.mono_time_ns - source.route_time_origin_mono_ns
        ) * 1e-9,
        speed_mps=physical.speed_mps,
        model_publication_index=(
          None if model is None else witness.model_publication_index
        ),
        live_rack_mapping=live_mapping,
        nominal_rack_mapping=nominal_mapping,
        core_input=frame_input.to_bytes(),
        inputs_valid=physically_valid,
        lateral_active=witness.lateral_active,
        steering_pressed=physical.steering_pressed,
        platform_fault=(
          witness.steer_fault
          or physical.steer_fault_temporary
          or physical.steer_fault_permanent
          or not physical.can_valid
          or physical.can_timeout
        ),
        driver_intervention_onset=witness.intervention_onset,
      ))
    events = tuple(sorted(
      (
        EventLocator(
          event_type=event.event_type,
          occurred_mono_time_ns=event.occurred_mono_time_ns,
          analysis_window_before_s=event.analysis_window_before_s,
          analysis_window_after_s=event.analysis_window_after_s,
          severity=event.severity,
        )
        for event in artifact.event_locators
        if event.message_valid
      ),
      key=lambda event: (
        event.occurred_mono_time_ns,
        event.event_type,
        event.severity,
      ),
    ))
    return DecodedBehaviorRoute(
      route_id=source.route_id,
      route_evidence_sha256=artifact.sha256,
      vehicle_identity=source.vehicle_identity,
      recorded_source=recorded_source,
      model_publications=models,
      control_inputs=tuple(controls),
      event_locators=events,
    )

  return decode


def _decode_car_params(encoded: bytes) -> Any:
  if not encoded:
    raise BehaviorReplayError("route replay lacks canonical CarParams bytes")
  try:
    params = messaging.log_from_bytes(encoded, car.CarParams)
    # Force the fields used during construction while the capnp reader is in
    # scope; malformed/non-CarParams payloads fail here rather than later.
    str(params.carFingerprint)
    float(params.mass)
    float(params.wheelbase)
    return params
  except Exception as exc:
    raise BehaviorReplayError("route CarParams cannot be decoded") from exc


def _route_runtime(
  request: ControllerReplayRequest,
  first_input: ReplayFrameInput,
  provisional_dynamics: ProvisionalRackDynamics,
  interface_registry: Mapping[str, type] | None,
) -> _RouteRuntime:
  if first_input.car_params_bytes is None:
    raise BehaviorReplayError("first replay frame must carry canonical CarParams")
  params = _decode_car_params(first_input.car_params_bytes)
  try:
    bundle, interface, _ = build_detected_runtime_bundle(
      car_params=params,
      provisional_rack_dynamics=provisional_dynamics,
      interface_registry=interface_registry,
    )
    profile = compose_controller_profile(
      request.physical_profile,
      bundle.seed_profile,
    )
  except Exception as exc:
    raise BehaviorReplayError("route runtime bundle cannot be reconstructed") from exc
  if bundle.vehicle_identity != request.route.vehicle_identity:
    raise BehaviorReplayError("route runtime identity differs from decoded route")
  if profile.vehicle_identity != request.route.vehicle_identity:
    raise BehaviorReplayError("physical profile belongs to a different vehicle")
  for control in request.route.control_inputs:
    if control.nominal_rack_mapping != bundle.nominal_rack_mapping:
      raise BehaviorReplayError("decoded nominal geometry differs from CarParams")
  return _RouteRuntime(params, interface, bundle, profile)


def _decode_inputs(route: DecodedBehaviorRoute) -> tuple[ReplayFrameInput, ...]:
  decoded = tuple(ReplayFrameInput.from_bytes(control.core_input) for control in route.control_inputs)
  if not decoded:
    raise BehaviorReplayError("behavior replay route contains no control inputs")
  if decoded[0].car_params_bytes is None:
    raise BehaviorReplayError("first replay input omits route CarParams")
  if any(value.car_params_bytes is not None for value in decoded[1:]):
    raise BehaviorReplayError("CarParams must appear exactly once at route start")
  if any(
    frame.physical_record_index <= previous.physical_record_index
    for previous, frame in zip(decoded, decoded[1:], strict=False)
  ):
    raise BehaviorReplayError("physical record links must be strictly increasing")
  return decoded


def _headroom(limits: RuntimeTorqueLimits, requested_counts: int) -> float:
  used = min(abs(int(requested_counts)), limits.steer_max)
  return (limits.steer_max - used) / limits.steer_max


def _measured_curvature(
  state: RackState,
  speed_mps: float,
  live_mapping: RackMappingSnapshot | None,
  nominal_mapping: RackMappingSnapshot,
) -> float:
  return curvature_from_measured_angle(
    state.angle_deg,
    speed_mps,
    live_mapping,
    nominal_mapping,
  ).curvature


def _output(
  *,
  mono_time_ns: int,
  state: RackState,
  acceleration_deg_s2: float,
  measured_curvature: float,
  raw_requested_torque: float,
  applied_counts: int,
  requested_counts: int,
  limits: RuntimeTorqueLimits,
  response_eligible: bool,
  controller_fault: bool,
) -> ControllerFrameOutput:
  raw = float(raw_requested_torque)
  if not math.isfinite(raw):
    raw = 0.0
    controller_fault = True
    response_eligible = False
  return ControllerFrameOutput(
    mono_time_ns=mono_time_ns,
    measured_curvature_1pm=float(measured_curvature),
    measured_rack_angle_deg=float(state.angle_deg),
    measured_rack_rate_deg_s=float(state.rate_deg_s),
    measured_rack_accel_deg_s2=float(acceleration_deg_s2),
    raw_requested_torque=raw,
    envelope_applied_torque=applied_counts / limits.steer_max,
    torque_headroom=_headroom(limits, requested_counts),
    actuator_constrained=applied_counts != requested_counts,
    controller_fault=bool(controller_fault),
    response_eligible=bool(response_eligible),
  )


class _RequestProducer:
  """Episode-local controller interface; never owns the plant or envelope."""

  def reset(
    self,
    *,
    initial_state: RackState,
    initial_applied_counts: int,
    initial_raw_request_torque: float,
  ) -> None:
    raise NotImplementedError

  def request_torque(
    self,
    *,
    request: ControllerReplayRequest,
    control_index: int,
    frame_input: ReplayFrameInput,
    state: RackState,
    acceleration_deg_s2: float,
    previous_applied_counts: int,
    previous_output_constrained: bool,
    engagement_boundary: bool,
  ) -> tuple[float, bool]:
    raise NotImplementedError


class _ModularRequestProducer(_RequestProducer):
  def __init__(
    self,
    request: ControllerReplayRequest,
    runtime: _RouteRuntime,
  ) -> None:
    if request.policy is None:
      raise BehaviorReplayError("modular replay requires an explicit behavior policy")
    self.core = ModularControllerCore(
      fixed_dt_s=DT_CTRL,
      profile=runtime.controller_profile,
      tracking_policy=TrackingPolicy(
        natural_frequency_per_s=request.policy.natural_frequency_per_s,
        damping_ratio=request.policy.damping_ratio,
      ),
      # A twin cannot produce a non-degenerate response innovation against
      # itself. Counterfactual observer learning is therefore explicitly off.
      observer_policy=None,
      nominal_mapping=runtime.runtime_bundle.nominal_rack_mapping,
      plan_capacity=INTENT_CAPACITY,
    )
    self.params = runtime.car_params
    self.profile = runtime.controller_profile
    self.plan_times = [0.0] * INTENT_CAPACITY
    self.orientation_rates = [0.0] * INTENT_CAPACITY
    self.velocities = [0.0] * INTENT_CAPACITY
    self.plan_curvatures = [0.0] * INTENT_CAPACITY

  def reset(
    self,
    *,
    initial_state: RackState,
    initial_applied_counts: int,
    initial_raw_request_torque: float,
  ) -> None:
    # A producer object is created once per route callback.  Reconstructing
    # its core at a later episode boundary would require retaining constructor
    # inputs and risks an accidental state leak. The whole producer is instead
    # replaced by `_EpisodeReplay`'s factory at each boundary.
    del initial_applied_counts, initial_raw_request_torque
    self.core.prime_applied_history(initial_state.applied_torque)

  def request_torque(
    self,
    *,
    request: ControllerReplayRequest,
    control_index: int,
    frame_input: ReplayFrameInput,
    state: RackState,
    acceleration_deg_s2: float,
    previous_applied_counts: int,
    previous_output_constrained: bool,
    engagement_boundary: bool,
  ) -> tuple[float, bool]:
    control = request.route.control_inputs[control_index]
    if control.model_publication_index is None:
      return 0.0, True
    model = request.route.model_publications[control.model_publication_index]
    if model.model_frame_id != frame_input.model_frame_id:
      raise BehaviorReplayError("model frame identity differs from the exact link")
    parameters = self.profile.parameters_at(control.speed_mps).parameters
    adaptation = adapt_model_intent_into(
      state_sample_mono_ns=frame_input.state_sample_mono_time_ns,
      control_witness_mono_ns=control.mono_time_ns,
      model_publication_mono_ns=model.publication_mono_time_ns,
      plan_origin_mono_ns=model.plan_origin_mono_time_ns,
      model_frame_id=frame_input.model_frame_id,
      message_valid=frame_input.model_message_valid,
      message_alive=frame_input.model_message_alive,
      scalar_desired_curvature=model.scalar_curvature_1pm,
      published_desired_curvature_time_s=model.scalar_action_plan_s,
      native_plan_times_s=model.native_times_s,
      native_orientation_rates_z=model.orientation_rates_z,
      native_velocities_x=model.velocities_x,
      current_v_ego_m_s=control.speed_mps,
      physical_transport_delay_s=parameters.transport_delay_s,
      output_plan_times_s=self.plan_times,
      output_orientation_rates_z=self.orientation_rates,
      output_velocities_x=self.velocities,
      output_plan_curvatures=self.plan_curvatures,
    )
    result = self.core.update(
      frame=adaptation.frame,
      intent_status=adaptation.status,
      intent_plan_times_s=self.plan_times,
      intent_orientation_rates_z=self.orientation_rates,
      intent_velocities_x=self.velocities,
      scalar_curvature=model.scalar_curvature_1pm,
      current_v_ego_m_s=control.speed_mps,
      measured_rack_angle_deg=state.angle_deg,
      measured_rack_rate_deg_s=state.rate_deg_s,
      measured_rack_acceleration_deg_s2=acceleration_deg_s2,
      recorded_applied_torque=state.applied_torque,
      lateral_accel_offset=float(self.params.lateralTuning.torque.latAccelOffset),
      live_mapping=control.live_rack_mapping,
      lateral_active=True,
      lateral_valid=(
        control.inputs_valid
        and frame_input.witness_resolved
        and frame_input.control_cadence_valid
        and frame_input.live_parameters_inputs_valid
      ),
      engagement_boundary=engagement_boundary,
      live_parameters_valid=frame_input.live_parameters_inputs_valid,
      steering_pressed=control.steering_pressed,
      actuator_constrained=previous_output_constrained,
      output_constrained=previous_output_constrained,
      standstill=frame_input.standstill,
    )
    if result.valid and result.status in _LIVE_CORE_STATUSES:
      reference = request.references[control_index]
      exact_fields = (
        (result.desired_curvature, reference.anchored_curvature_1pm),
        (result.desired_curvature_rate, reference.anchored_curvature_rate_1pm_s),
        (result.desired_curvature_acceleration, reference.anchored_curvature_accel_1pm_s2),
        (result.desired_angle_deg, reference.desired_rack_angle_deg),
        (result.desired_rate_deg_s, reference.desired_rack_rate_deg_s),
        (result.desired_acceleration_deg_s2, reference.desired_rack_accel_deg_s2),
      )
      if any(left != right for left, right in exact_fields):
        raise BehaviorReplayError(
          "modular core reference differs from transaction reference",
        )
      return float(result.raw_torque), False
    return 0.0, True


class _StockRequestProducer(_RequestProducer):
  def __init__(self, runtime: _RouteRuntime) -> None:
    self.params = runtime.car_params
    self.interface = runtime.car_interface
    self.controller = fresh_stock_torque_controller(
      self.params,
      self.interface,
    )
    self.vehicle_model = VehicleModel(self.params)
    self.desired_curvature = 0.0
    self.desired_curvature_initialized = False
    self.previous_raw_request = 0.0
    self.previous_applied = 0.0

  def reset(
    self,
    *,
    initial_state: RackState,
    initial_applied_counts: int,
    initial_raw_request_torque: float,
  ) -> None:
    self.desired_curvature = 0.0
    self.desired_curvature_initialized = False
    self.previous_raw_request = float(initial_raw_request_torque)
    self.previous_applied = initial_state.applied_torque

  def request_torque(
    self,
    *,
    request: ControllerReplayRequest,
    control_index: int,
    frame_input: ReplayFrameInput,
    state: RackState,
    acceleration_deg_s2: float,
    previous_applied_counts: int,
    previous_output_constrained: bool,
    engagement_boundary: bool,
  ) -> tuple[float, bool]:
    del acceleration_deg_s2, previous_applied_counts, engagement_boundary
    control = request.route.control_inputs[control_index]
    if control.model_publication_index is None:
      return 0.0, True
    model = request.route.model_publications[control.model_publication_index]
    if not frame_input.live_torque_health_exact:
      # Stock conditionally updates its stateful calibration from
      # sm.all_checks(['liveTorqueParameters']).  If that witness-time result
      # cannot be proved, there is no exact stock replay for this frame.
      return 0.0, True
    try:
      self.vehicle_model.update_params(
        max(frame_input.stiffness_factor, 0.1),
        max(frame_input.steer_ratio, 0.1),
      )
      live_params = SimpleNamespace(
        angleOffsetDeg=(
          control.live_rack_mapping.angle_offset_deg
          if control.live_rack_mapping is not None
          else 0.0
        ),
        roll=(
          control.live_rack_mapping.roll_rad
          if control.live_rack_mapping is not None
          else 0.0
        ),
      )
      current_curvature = -self.vehicle_model.calc_curvature(
        math.radians(state.angle_deg - live_params.angleOffsetDeg),
        control.speed_mps,
        live_params.roll,
      )
      if frame_input.live_torque_inputs_valid and frame_input.live_torque_use_params:
        self.controller.update_live_torque_params(
          frame_input.live_torque_lat_accel_factor,
          frame_input.live_torque_lat_accel_offset,
          frame_input.live_torque_friction,
        )
      selected_curvature = (
        frame_input.lateral_maneuver_desired_curvature
        if frame_input.lateral_maneuver_plan_valid
        else model.scalar_curvature_1pm
      )
      if not self.desired_curvature_initialized:
        self.desired_curvature = current_curvature
        self.desired_curvature_initialized = True
      self.desired_curvature, curvature_limited = clip_curvature(
        control.speed_mps,
        self.desired_curvature,
        selected_curvature,
        live_params.roll,
      )
      car_state = SimpleNamespace(
        vEgo=control.speed_mps,
        steeringAngleDeg=state.angle_deg,
        steeringRateDeg=state.rate_deg_s,
        steeringTorque=frame_input.driver_torque,
        steeringPressed=control.steering_pressed,
        standstill=frame_input.standstill,
      )
      raw, _, _ = self.controller.update(
        True,
        car_state,
        self.vehicle_model,
        live_params,
        previous_output_constrained,
        self.desired_curvature,
        curvature_limited,
        frame_input.lateral_delay_seconds + SOURCE_LAT_SMOOTH_SECONDS,
      )
      raw_float = float(raw)
      if not math.isfinite(raw_float):
        return 0.0, True
      self.previous_raw_request = raw_float
      self.previous_applied = state.applied_torque
      return raw_float, False
    except (
      AttributeError,
      TypeError,
      ValueError,
      OverflowError,
      ZeroDivisionError,
    ):
      return 0.0, True


class _EpisodeReplay:
  """One shared count-envelope/plant loop for either request producer."""

  def __init__(
    self,
    *,
    request: ControllerReplayRequest,
    inputs: tuple[ReplayFrameInput, ...],
    runtime: _RouteRuntime,
    modular: bool,
  ) -> None:
    self.request = request
    self.inputs = inputs
    self.runtime = runtime
    self.modular = bool(modular)
    self.limits = runtime.runtime_bundle.torque_limits
    self.state: RackState | None = None
    self.acceleration_deg_s2 = 0.0
    self.previous_applied_counts = 0
    self.previous_requested_counts = 0
    self.previous_active = False
    self.censored = False
    self.episode_faulted = False
    self.producer: _RequestProducer | None = None

  def _fresh_producer(self) -> _RequestProducer:
    return (
      _ModularRequestProducer(self.request, self.runtime)
      if self.modular
      else _StockRequestProducer(self.runtime)
    )

  def _bootstrap(self, frame_input: ReplayFrameInput) -> bool:
    if (
      not frame_input.applied_count_valid
      or abs(frame_input.recorded_applied_counts) > self.limits.steer_max
      or not frame_input.witness_resolved
    ):
      self.state = None
      self.producer = None
      self.episode_faulted = True
      return False
    applied = frame_input.recorded_applied_counts / self.limits.steer_max
    self.state = RackState(
      frame_input.recorded_rack_angle_deg,
      frame_input.recorded_rack_rate_deg_s,
      applied,
    )
    self.acceleration_deg_s2 = frame_input.recorded_rack_acceleration_deg_s2
    self.previous_applied_counts = frame_input.recorded_applied_counts
    self.previous_requested_counts = int(round(
      frame_input.recorded_raw_request_torque * self.limits.steer_max,
    ))
    self.censored = False
    self.episode_faulted = False
    self.producer = self._fresh_producer()
    self.producer.reset(
      initial_state=self.state,
      initial_applied_counts=self.previous_applied_counts,
      initial_raw_request_torque=frame_input.recorded_raw_request_torque,
    )
    return True

  def _inactive_output(
    self,
    control: CanonicalBehaviorControlInput,
    frame_input: ReplayFrameInput,
  ) -> ControllerFrameOutput:
    counts = (
      frame_input.recorded_applied_counts
      if frame_input.applied_count_valid
      and abs(frame_input.recorded_applied_counts) <= self.limits.steer_max
      else 0
    )
    state = RackState(
      frame_input.recorded_rack_angle_deg,
      frame_input.recorded_rack_rate_deg_s,
      counts / self.limits.steer_max,
    )
    try:
      curvature = _measured_curvature(
        state,
        control.speed_mps,
        control.live_rack_mapping,
        control.nominal_rack_mapping,
      )
    except (ValueError, OverflowError):
      curvature = 0.0
    return _output(
      mono_time_ns=control.mono_time_ns,
      state=state,
      acceleration_deg_s2=frame_input.recorded_rack_acceleration_deg_s2,
      measured_curvature=curvature,
      raw_requested_torque=0.0,
      applied_counts=counts,
      requested_counts=counts,
      limits=self.limits,
      response_eligible=False,
      controller_fault=False,
    )

  def replay(self) -> tuple[ControllerFrameOutput, ...]:
    outputs: list[ControllerFrameOutput] = []
    for index, (control, frame_input) in enumerate(zip(
      self.request.route.control_inputs,
      self.inputs,
      strict=True,
    )):
      active = bool(control.lateral_active)
      engagement_boundary = active and not self.previous_active
      if not active:
        outputs.append(self._inactive_output(control, frame_input))
        self.state = None
        self.producer = None
        self.censored = False
        self.episode_faulted = False
        self.previous_active = False
        continue

      if engagement_boundary and not self._bootstrap(frame_input):
        outputs.append(self._inactive_output(control, frame_input))
        self.previous_active = True
        continue
      self.previous_active = True
      if (
        control.driver_intervention_onset
        or frame_input.intervention_onset_uncertain
      ):
        self.censored = True
      if self.state is None or self.producer is None:
        outputs.append(self._inactive_output(control, frame_input))
        continue

      runtime_valid = (
        control.inputs_valid
        and frame_input.witness_resolved
        and frame_input.control_cadence_valid
        and frame_input.live_parameters_inputs_valid
      )
      if not runtime_valid:
        self.episode_faulted = True
      previous_output_constrained = (
        abs(
          self.previous_requested_counts - self.previous_applied_counts,
        ) / self.limits.steer_max > 1e-2
      )
      raw = 0.0
      controller_fault = self.episode_faulted
      if not self.episode_faulted:
        raw, controller_fault = self.producer.request_torque(
          request=self.request,
          control_index=index,
          frame_input=frame_input,
          state=self.state,
          acceleration_deg_s2=self.acceleration_deg_s2,
          previous_applied_counts=self.previous_applied_counts,
          previous_output_constrained=previous_output_constrained,
          engagement_boundary=engagement_boundary,
        )
      if not math.isfinite(raw):
        raw = 0.0
        controller_fault = True
      requested_counts = int(round(raw * self.limits.steer_max))
      try:
        applied_counts = apply_torque_envelope_counts(
          self.limits,
          requested_counts,
          self.previous_applied_counts,
          frame_input.driver_torque,
        )
        curvature = _measured_curvature(
          self.state,
          control.speed_mps,
          control.live_rack_mapping,
          control.nominal_rack_mapping,
        )
      except (TypeError, ValueError, OverflowError):
        applied_counts = self.previous_applied_counts
        requested_counts = self.previous_applied_counts
        curvature = 0.0
        raw = applied_counts / self.limits.steer_max
        controller_fault = True
        self.episode_faulted = True
      response_eligible = (
        runtime_valid
        and not self.censored
        and not controller_fault
        and not control.platform_fault
      )
      outputs.append(_output(
        mono_time_ns=control.mono_time_ns,
        state=self.state,
        acceleration_deg_s2=self.acceleration_deg_s2,
        measured_curvature=curvature,
        raw_requested_torque=raw,
        applied_counts=applied_counts,
        requested_counts=requested_counts,
        limits=self.limits,
        response_eligible=response_eligible,
        controller_fault=controller_fault,
      ))

      selected_mapping = (
        control.live_rack_mapping
        if control.live_rack_mapping is not None and control.live_rack_mapping.valid
        else control.nominal_rack_mapping
      )
      parameters = self.runtime.controller_profile.parameters_at(
        control.speed_mps,
      ).parameters
      try:
        plant_step = step_plant(
          self.state,
          applied_counts / self.limits.steer_max,
          control.speed_mps,
          selected_mapping,
          control.nominal_rack_mapping,
          float(self.runtime.car_params.lateralTuning.torque.latAccelOffset)
          + parameters.lateral_accel_offset_correction_mps2,
          parameters,
          # Counterfactual observer learning is disabled; the plant and core
          # therefore share the same explicit zero disturbance.
          0.0,
          DT_CTRL,
        )
        self.state = plant_step.state
        self.acceleration_deg_s2 = plant_step.acceleration_deg_s2
      except (TypeError, ValueError, OverflowError):
        self.episode_faulted = True
      self.previous_requested_counts = requested_counts
      self.previous_applied_counts = applied_counts
    return tuple(outputs)


def _make_replay_core(
  identity: ReplayCoreIdentity,
  *,
  provisional_dynamics: ProvisionalRackDynamics,
  interface_registry: Mapping[str, type] | None,
  modular: bool,
) -> BehaviorReplayCore:
  if not isinstance(identity, ReplayCoreIdentity):
    raise TypeError("behavior replay core identity has the wrong type")
  if not isinstance(provisional_dynamics, ProvisionalRackDynamics):
    raise TypeError("behavior replay requires explicit provisional rack dynamics")
  registry = None if interface_registry is None else dict(interface_registry)

  def replay_route(request: ControllerReplayRequest) -> Iterable[ControllerFrameOutput]:
    if not isinstance(request, ControllerReplayRequest):
      raise TypeError("behavior replay request has the wrong type")
    if request.artifact_identity.core != identity:
      raise BehaviorReplayError(
        "behavior replay request/core identities do not match",
      )
    if modular != (request.policy is not None):
      raise BehaviorReplayError(
        "behavior replay core and policy presence do not match",
      )
    inputs = _decode_inputs(request.route)
    runtime = _route_runtime(
      request,
      inputs[0],
      provisional_dynamics,
      registry,
    )
    return _EpisodeReplay(
      request=request,
      inputs=inputs,
      runtime=runtime,
      modular=modular,
    ).replay()

  return BehaviorReplayCore(identity=identity, replay_route=replay_route)


def make_modular_behavior_replay_core(
  identity: ReplayCoreIdentity,
  *,
  provisional_dynamics: ProvisionalRackDynamics,
  interface_registry: Mapping[str, type] | None = None,
) -> BehaviorReplayCore:
  """Build a thread-safe factory callback over exact modular core bytes."""
  return _make_replay_core(
    identity,
    provisional_dynamics=provisional_dynamics,
    interface_registry=interface_registry,
    modular=True,
  )


def make_exact_stock_behavior_replay_core(
  identity: ReplayCoreIdentity,
  *,
  provisional_dynamics: ProvisionalRackDynamics,
  interface_registry: Mapping[str, type] | None = None,
) -> BehaviorReplayCore:
  """Build full source-stock replay; no modular-reference hybrid exists."""
  return _make_replay_core(
    identity,
    provisional_dynamics=provisional_dynamics,
    interface_registry=interface_registry,
    modular=False,
  )

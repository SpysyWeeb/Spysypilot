"""Exact, self-anchored behavior replay for the modular learning transaction.

This module is the production numerical adapter between immutable
``RouteEvidenceArtifact`` bytes and :mod:`behavior_transaction`.  It has two
controller request producers, but deliberately only one episode simulator:

* the modular producer calls the existing :class:`ModularControllerCore`;
* the stock producer constructs the source :class:`LatControlTorque` through
  :func:`fresh_stock_torque_controller` and preserves controlsd's
  ``clip_curvature``/VehicleModel/live-torque state; and
* both requests pass through the exact count-space opendbc envelope and the
  same independently identified counterfactual plant member.

The strict behavior decoder retains the original homogeneous-controller
qualification contract.  A separate scenario-only decoder admits an older or
unverified recorded controller only as authenticated route context.  It
preserves that source and rejection reason verbatim, validates the planes
needed by both counterfactual opponents, and emits an order-sensitive scenario
identity.  Recorded commands never become targets or controller state.

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
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from opendbc.car.hyundai.steering_request import (
  apply_steering_request_fault_avoidance,
)
from opendbc.car.vehicle_model import VehicleModel

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.blatv2.actuator import (
  RuntimeTorqueLimits,
  apply_torque_envelope_counts,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_coordinator import (
  ReplayCoreIdentity,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import (
  BEHAVIOR_SCENARIO_PROVENANCE_SCHEMA_VERSION,
  BehaviorReferenceAtControl,
  BehaviorScenarioProvenance,
  BehaviorScenarioSetIdentity,
  BehaviorSourceIdentity,
  EventLocator,
  SparseModelBehaviorIntent,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_policy import BehaviorPolicy
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
from openpilot.selfdrive.controls.lib.blatv2.counterfactual_plant import (
  AppliedTorqueDelayLine,
  CounterfactualPlantMember,
  step_counterfactual_plant,
)
from openpilot.selfdrive.controls.lib.blatv2.intent import (
  INTENT_CAPACITY,
  adapt_model_intent_into,
)
from openpilot.selfdrive.controls.lib.blatv2.horizon import HorizonPolicy
from openpilot.selfdrive.controls.lib.blatv2.learning_runtime import (
  build_detected_runtime_bundle,
)
from openpilot.selfdrive.controls.lib.blatv2.plant import (
  RackState,
  TrackingPolicy,
)
from openpilot.selfdrive.controls.lib.blatv2.preparation_frame import (
  MeasuredLearningFrame,
)
from openpilot.selfdrive.controls.lib.blatv2.preparation_contract import decode_car_params
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import (
  RackMappingSnapshot,
  curvature_from_measured_angle,
)
from openpilot.selfdrive.controls.lib.blatv2.route_evidence import (
  ROUTE_EVIDENCE_VERSION,
  ControlsWitness,
  LateralManeuverPlanPublication,
  LiveDelayPublication,
  LiveTorqueParametersPublication,
  ModelPublication,
  RouteEvidenceArtifact,
  RouteEvidenceSourceIdentity,
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
EXACT_STOCK_REPLAY_CONTROLLER_NAME = "openpilot.LatControlTorque.exact-stock"
EXACT_STOCK_REPLAY_IMPLEMENTATION_CONTRACT = "behavior-replay-full-stock-v1"
MODULAR_REPLAY_CONTROLLER_NAME = "blatv2.ModularControllerCore"
MODULAR_REPLAY_IMPLEMENTATION_CONTRACT = "behavior-replay-modular-core-v2"
PROVISIONAL_HORIZON_POLICY_PATH = Path(__file__).resolve().parent / "provisional_horizon_policy.json"
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


def reviewed_replay_core_identity(
  *,
  exact_stock: bool,
  source_openpilot_commit: str,
  opendbc_commit: str,
  panda_commit: str,
) -> ReplayCoreIdentity:
  """Compose one reviewed replay adapter identity from clean source commits.

  The controller name and implementation contract are intentionally not
  caller inputs.  Otherwise an arbitrary callback could choose an attractive
  label, calculate a self-consistent digest, and masquerade as the stock floor.
  The clean-checkout/source-pin boundary remains responsible for proving that
  the supplied commits identify the bytes executing this adapter.
  """
  return ReplayCoreIdentity.compose(
    controller_name=(
      EXACT_STOCK_REPLAY_CONTROLLER_NAME
      if exact_stock
      else MODULAR_REPLAY_CONTROLLER_NAME
    ),
    implementation_contract=(
      EXACT_STOCK_REPLAY_IMPLEMENTATION_CONTRACT
      if exact_stock
      else MODULAR_REPLAY_IMPLEMENTATION_CONTRACT
    ),
    replay_input_schema_version=BEHAVIOR_REPLAY_INPUT_SCHEMA_VERSION,
    source_openpilot_commit=source_openpilot_commit,
    opendbc_commit=opendbc_commit,
    panda_commit=panda_commit,
  )


def validate_reviewed_replay_core_identity(
  identity: ReplayCoreIdentity,
  *,
  exact_stock: bool,
) -> None:
  """Reject a role label whose core digest is not the reviewed adapter."""
  if not isinstance(identity, ReplayCoreIdentity):
    raise TypeError("behavior replay core identity has the wrong type")
  expected = reviewed_replay_core_identity(
    exact_stock=exact_stock,
    source_openpilot_commit=identity.source_openpilot_commit,
    opendbc_commit=identity.opendbc_commit,
    panda_commit=identity.panda_commit,
  )
  if identity != expected:
    role = "exact-stock" if exact_stock else "modular"
    raise BehaviorReplayError(
      f"{role} replay identity differs from the reviewed implementation",
    )


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
    if (
      self.live_torque_inputs_valid
      and self.live_torque_use_params
      and self.live_torque_friction < 0.0
    ):
      raise ValueError("consumed live torque friction must be non-negative")
    if self.lateral_delay_seconds < 0.0:
      raise ValueError("lateral delay must be non-negative")
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


def validate_behavior_scenario_active_frame(
  *,
  physical_record_index: int,
  rack_acceleration_valid: bool,
  witness: ControlsWitness,
  physical: Any,
  model: ModelPublication | None,
  live_torque: LiveTorqueParametersPublication | None,
  live_delay: LiveDelayPublication | None,
  maneuver: LateralManeuverPlanPublication | None,
) -> None:
  """Validate every input consumed by one lateral-active scenario frame.

  This is the full-route counterpart of certification-vector v5's bounded
  scenario plane.  It runs before reference derivation or either controller is
  advanced. Exact-stock calibration may legitimately retain its initialized
  or last accepted value when the witnessed live-torque checks did not pass;
  missing context and synthetic delay fallback remain forbidden. Live
  rack-mapping validity remains the separate pinned measured-frame contract.
  """
  if not isinstance(witness, ControlsWitness):
    raise TypeError("behavior scenario validation requires a controls witness")
  if not witness.lateral_active:
    return
  if witness.race_unresolved:
    raise BehaviorReplayError("active lateral scenario contains unresolved evidence")
  if witness.gap_from_previous:
    raise BehaviorReplayError("active lateral scenario lacks valid control cadence")
  if not witness.car_control_paired:
    raise BehaviorReplayError("active lateral scenario lacks carControl context")
  if not isinstance(physical, MeasuredLearningFrame):
    raise TypeError("active lateral scenario physical frame is malformed")
  if witness.inputs_valid != physical.inputs_valid:
    raise BehaviorReplayError(
      "active lateral scenario invalid evidence: physical validity disagrees",
    )
  if not witness.message_valid or not physical.inputs_valid:
    raise BehaviorReplayError("active lateral scenario contains invalid evidence")
  if physical_record_index != 0 and not rack_acceleration_valid:
    raise BehaviorReplayError("active lateral scenario lacks valid control cadence")
  if (
    not witness.model_link_valid
    or witness.model_publication_index < 0
    or not isinstance(model, ModelPublication)
  ):
    raise BehaviorReplayError("active lateral scenario lacks exact model intent")
  if model.mono_time_ns > witness.mono_time_ns:
    raise BehaviorReplayError("active lateral scenario model link points to the future")
  if not model.message_valid or not witness.model_message_alive:
    raise BehaviorReplayError("active lateral scenario model intent is not valid and alive")
  model_payload_valid = (
    model.frame_id >= 0
    and 0 <= model.timestamp_eof_ns <= model.mono_time_ns
    and math.isfinite(model.desired_curvature_time_s)
    and model.desired_curvature_time_s >= 0.0
    and model.native_grid_valid
    and bool(model.plan_times)
    and math.isfinite(model.plan_times[0])
    and model.plan_times[0] >= 0.0
  )
  if not model_payload_valid:
    raise BehaviorReplayError("active lateral scenario model intent payload is invalid")
  if (
    not witness.torque_output_can_valid
    or type(witness.torque_output_can_count) is not int
    or not math.isfinite(physical.applied_torque)
  ):
    raise BehaviorReplayError("active lateral scenario lacks exact applied torque")

  if (
    not witness.live_torque_parameters_available
    or witness.live_torque_parameters_index < 0
    or not isinstance(live_torque, LiveTorqueParametersPublication)
  ):
    raise BehaviorReplayError("active lateral scenario lacks live torque context")
  if live_torque.mono_time_ns > witness.mono_time_ns:
    raise BehaviorReplayError("active lateral scenario live torque link points to the future")
  if not live_torque.message_valid:
    raise BehaviorReplayError("active lateral scenario live torque publication is invalid")
  if not witness.live_torque_parameters_health_exact:
    raise BehaviorReplayError(
      "active lateral scenario lacks exact stock calibration health",
    )
  # These payload scalars are consumed only when the exact witness-time health
  # check passed and the publication selected parameter use.  Failed checks or
  # useParams=false deliberately leave even negative sentinel values inert.
  if witness.live_torque_parameters_checks_passed and live_torque.use_params:
    if (
      not math.isfinite(live_torque.lat_accel_factor)
      or live_torque.lat_accel_factor <= 0.0
      or not math.isfinite(live_torque.lat_accel_offset)
      or not math.isfinite(live_torque.friction)
      or live_torque.friction < 0.0
    ):
      raise BehaviorReplayError("active lateral scenario live torque payload is invalid")

  if (
    not witness.live_delay_available
    or witness.live_delay_index < 0
    or not isinstance(live_delay, LiveDelayPublication)
  ):
    raise BehaviorReplayError("active lateral scenario lacks live delay context")
  if live_delay.mono_time_ns > witness.mono_time_ns:
    raise BehaviorReplayError("active lateral scenario live delay link points to the future")
  if (
    not live_delay.message_valid
    or not math.isfinite(live_delay.lateral_delay_s)
    or live_delay.lateral_delay_s < 0.0
  ):
    raise BehaviorReplayError("active lateral scenario live delay payload is invalid")

  maneuver_link_present = (
    witness.maneuver_plan_available
    or witness.lateral_maneuver_plan_index >= 0
  )
  if maneuver_link_present and not isinstance(
    maneuver,
    LateralManeuverPlanPublication,
  ):
    raise BehaviorReplayError("active lateral scenario maneuver context is invalid")
  if maneuver is not None:
    if maneuver.mono_time_ns > witness.mono_time_ns:
      raise BehaviorReplayError("active lateral scenario maneuver link points to the future")
    if maneuver.message_valid:
      raise BehaviorReplayError(
        "active lateral scenario uses a lateral maneuver plan override",
      )


def behavior_rack_mapping_from_physical_frame(
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


def sparse_model_behavior_intent(
  publication: ModelPublication,
) -> SparseModelBehaviorIntent:
  """Project one authenticated model publication into replay intent."""
  if not isinstance(publication, ModelPublication):
    raise TypeError("behavior intent requires a model publication")
  return SparseModelBehaviorIntent(
    plan_origin_mono_time_ns=publication.timestamp_eof_ns,
    publication_mono_time_ns=publication.mono_time_ns,
    model_frame_id=publication.frame_id,
    plan_valid=publication.native_grid_valid,
    scalar_curvature_1pm=publication.scalar_curvature,
    scalar_action_plan_s=publication.desired_curvature_time_s,
    native_times_s=publication.plan_times,
    orientation_rates_z=publication.orientation_rate_z,
    velocities_x=publication.velocity_x,
  )


def build_canonical_behavior_frame(
  *,
  index: int,
  source: RouteEvidenceSourceIdentity,
  car_params: Any,
  car_params_bytes: bytes,
  nominal_rack_mapping: RackMappingSnapshot,
  mapping_model: VehicleModel,
  witness: ControlsWitness,
  physical: Any,
  rack_acceleration_deg_s2: float,
  rack_acceleration_valid: bool,
  model: ModelPublication | None,
  live_torque: LiveTorqueParametersPublication | None,
  live_delay: LiveDelayPublication | None,
  maneuver: LateralManeuverPlanPublication | None,
  scenario_only: bool,
) -> tuple[CanonicalBehaviorControlInput, ReplayFrameInput, SparseModelBehaviorIntent | None]:
  """Build the sole canonical control/frame pair for eager and streamed replay."""
  if witness.physical_record_index != index:
    raise BehaviorReplayError("control/physical link is non-canonical")
  for name, publication in (
    ("model", model),
    ("live torque", live_torque),
    ("live delay", live_delay),
    ("lateral maneuver", maneuver),
  ):
    if publication is not None and publication.mono_time_ns > witness.mono_time_ns:
      raise BehaviorReplayError(f"{name} link points to a future publication")
  validate_behavior_scenario_active_frame(
    physical_record_index=index,
    rack_acceleration_valid=rack_acceleration_valid,
    witness=witness,
    physical=physical,
    model=model,
    live_torque=live_torque,
    live_delay=live_delay,
    maneuver=maneuver,
  )
  live_mapping = behavior_rack_mapping_from_physical_frame(mapping_model, physical)
  live_parameters_inputs_valid = (
    live_mapping is not None
    and witness.live_parameters_mono_ns >= 0
  )
  torque_factor = (
    float(car_params.lateralTuning.torque.latAccelFactor)
    if live_torque is None
    else live_torque.lat_accel_factor
  )
  torque_offset = (
    float(car_params.lateralTuning.torque.latAccelOffset)
    if live_torque is None
    else live_torque.lat_accel_offset
  )
  torque_friction = (
    float(car_params.lateralTuning.torque.friction)
    if live_torque is None
    else live_torque.friction
  )
  frame_input = ReplayFrameInput(
    physical_record_index=index,
    state_sample_mono_time_ns=witness.state_sample_mono_ns,
    model_frame_id=0 if model is None else model.frame_id,
    recorded_rack_angle_deg=physical.steering_angle_deg,
    recorded_rack_rate_deg_s=physical.steering_rate_deg_s,
    recorded_rack_acceleration_deg_s2=rack_acceleration_deg_s2,
    recorded_applied_torque=physical.applied_torque,
    recorded_applied_counts=witness.torque_output_can_count,
    # Recorded request is retained only as authenticated provenance.  The
    # replay stepper bootstraps from the applied count and never consumes it.
    recorded_raw_request_torque=witness.raw_request_torque,
    driver_torque=physical.steering_torque,
    stiffness_factor=physical.stiffness_factor,
    steer_ratio=physical.steer_ratio,
    live_torque_parameters_publication_index=witness.live_torque_parameters_index,
    live_torque_lat_accel_factor=torque_factor,
    live_torque_lat_accel_offset=torque_offset,
    live_torque_friction=torque_friction,
    lateral_delay_publication_index=witness.live_delay_index,
    lateral_delay_seconds=(0.0 if live_delay is None else live_delay.lateral_delay_s),
    lateral_maneuver_plan_publication_index=witness.lateral_maneuver_plan_index,
    lateral_maneuver_desired_curvature=(
      0.0 if maneuver is None else maneuver.desired_curvature
    ),
    applied_count_valid=witness.torque_output_can_valid,
    witness_resolved=not witness.race_unresolved,
    control_cadence_valid=(
      not witness.gap_from_previous and (index == 0 or rack_acceleration_valid)
    ),
    model_message_valid=(False if model is None else model.message_valid),
    model_message_alive=(model is not None and witness.model_message_alive),
    live_parameters_inputs_valid=live_parameters_inputs_valid,
    live_torque_health_exact=witness.live_torque_parameters_health_exact,
    # This is the reconstructed witness-time SubMaster all_checks result,
    # not an inference from the latest publication's payload bits.
    live_torque_inputs_valid=witness.live_torque_parameters_checks_passed,
    live_torque_use_params=(live_torque is not None and live_torque.use_params),
    lateral_delay_inputs_valid=(live_delay is not None and live_delay.message_valid),
    lateral_maneuver_plan_valid=(maneuver is not None and maneuver.message_valid),
    intervention_onset_uncertain=witness.intervention_onset_uncertain,
    standstill=physical.standstill,
    car_params_bytes=car_params_bytes if index == 0 else None,
  )
  physically_valid = (
    (source.behavior_eligible or scenario_only)
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
  control = CanonicalBehaviorControlInput(
    mono_time_ns=witness.mono_time_ns,
    route_time_s=(
      witness.mono_time_ns - source.route_time_origin_mono_ns
    ) * 1e-9,
    speed_mps=physical.speed_mps,
    model_publication_index=(
      None if model is None else witness.model_publication_index
    ),
    live_rack_mapping=live_mapping,
    nominal_rack_mapping=nominal_rack_mapping,
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
  )
  return control, frame_input, None if model is None else sparse_model_behavior_intent(model)


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


def behavior_source_identity_from_route_source(
  source: RouteEvidenceSourceIdentity,
) -> BehaviorSourceIdentity:
  """Project one canonical behavioral identity from authenticated metadata.

  Cohort selection and route decoding must bind the exact same source.  Keep
  that projection here, next to the sole route-evidence decoder, so an
  offroad coordinator never grows a subtly different interpretation of the
  controller/build fields recorded in the artifact.
  """
  if type(source) is not RouteEvidenceSourceIdentity:
    raise TypeError("behavior source projection requires RouteEvidenceSourceIdentity")
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


def recorded_behavior_source_identity_from_route_source(
  source: RouteEvidenceSourceIdentity,
) -> BehaviorSourceIdentity:
  """Project recorded provenance without granting it behavioral authority."""
  if type(source) is not RouteEvidenceSourceIdentity:
    raise TypeError("recorded source projection requires RouteEvidenceSourceIdentity")
  return BehaviorSourceIdentity(
    controller_name=source.controller_source_kind,
    controller_artifact_sha256=source.controller_artifact_sha256,
    source_openpilot_commit=source.source_superproject_commit,
    opendbc_commit=source.source_opendbc_commit,
    panda_commit=source.source_panda_commit,
    evidence_schema_version=ROUTE_EVIDENCE_VERSION,
  )


def behavior_scenario_provenance_from_route_source(
  source: RouteEvidenceSourceIdentity,
  route_evidence_sha256: str,
) -> BehaviorScenarioProvenance:
  """Validate and preserve an authenticated recorded source as a scenario."""
  return BehaviorScenarioProvenance(
    schema_version=BEHAVIOR_SCENARIO_PROVENANCE_SCHEMA_VERSION,
    route_id=source.route_id,
    route_evidence_sha256=route_evidence_sha256,
    recorded_source=recorded_behavior_source_identity_from_route_source(source),
    recorded_behavior_eligible=source.behavior_eligible,
    recorded_behavior_ineligible_reason=source.behavior_ineligible_reason,
    vehicle_identity=source.vehicle_identity,
    runtime_identity=source.runtime_identity,
    preparation_cache_key=source.preparation_cache_key,
  )


def behavior_scenario_set_identity(
  routes: Sequence[DecodedBehaviorRoute],
) -> BehaviorScenarioSetIdentity:
  """Bind the caller's ordered scenario/source population to one identity."""
  if not routes:
    raise BehaviorReplayError("scenario experiment contains no routes")
  provenances: list[BehaviorScenarioProvenance] = []
  for route in routes:
    if not isinstance(route, DecodedBehaviorRoute):
      raise TypeError("scenario experiment requires decoded behavior routes")
    if route.scenario_provenance is None:
      raise BehaviorReplayError("decoded route lacks scenario-only provenance")
    provenances.append(route.scenario_provenance)
  return BehaviorScenarioSetIdentity(tuple(provenances))


def behavior_source_identity_from_route_artifact(
  artifact: RouteEvidenceArtifact,
) -> BehaviorSourceIdentity:
  """Project the exact source identity from a fully decoded legacy artifact."""
  if type(artifact) is not RouteEvidenceArtifact:
    raise TypeError("behavior source projection requires RouteEvidenceArtifact")
  return behavior_source_identity_from_route_source(artifact.source_identity)


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
  return _make_behavior_route_evidence_decoder(
    provisional_dynamics=provisional_dynamics,
    interface_registry=interface_registry,
    scenario_only=False,
  )


def make_behavior_scenario_route_evidence_decoder(
  *,
  provisional_dynamics: ProvisionalRackDynamics,
  interface_registry: Mapping[str, type] | None = None,
) -> Any:
  """Return a strict decoder for controller-independent route scenarios.

  This admits routes whose *recorded controller* was not behavior-authorized,
  but only when the authenticated planes required to replay exact stock and a
  modular candidate are complete. Recorded-controller identity and its
  original ineligibility reason remain immutable provenance.
  """
  return _make_behavior_route_evidence_decoder(
    provisional_dynamics=provisional_dynamics,
    interface_registry=interface_registry,
    scenario_only=True,
  )


def _make_behavior_route_evidence_decoder(
  *,
  provisional_dynamics: ProvisionalRackDynamics,
  interface_registry: Mapping[str, type] | None,
  scenario_only: bool,
) -> Any:
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
    if scenario_only:
      scenario_provenance = behavior_scenario_provenance_from_route_source(
        source,
        artifact.sha256,
      )
      recorded_source = scenario_provenance.recorded_source
    else:
      scenario_provenance = None
      recorded_source = behavior_source_identity_from_route_artifact(artifact)
    params = decode_behavior_car_params(bytes(artifact.car_params_bytes))
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
      sparse_model_behavior_intent(model)
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
      control, _, _ = build_canonical_behavior_frame(
        index=index,
        source=source,
        car_params=params,
        car_params_bytes=cp_bytes,
        nominal_rack_mapping=nominal_mapping,
        mapping_model=mapping_model,
        witness=witness,
        physical=physical,
        rack_acceleration_deg_s2=acceleration_fact[0],
        rack_acceleration_valid=acceleration_fact[1],
        model=model,
        live_torque=torque,
        live_delay=delay,
        maneuver=maneuver,
        scenario_only=scenario_only,
      )
      controls.append(control)
    if scenario_only:
      if not any(witness.lateral_active for witness in artifact.control_witnesses):
        raise BehaviorReplayError("route evidence has no active lateral scenario")
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
      scenario_provenance=scenario_provenance,
    )

  return decode


def decode_behavior_car_params(encoded: bytes) -> Any:
  """Decode and minimally force the route's authenticated CarParams."""
  if not encoded:
    raise BehaviorReplayError("route replay lacks canonical CarParams bytes")
  try:
    params = decode_car_params(encoded)
    # Force the fields used during construction so malformed/non-CarParams
    # payloads fail here rather than later.
    str(params.carFingerprint)
    float(params.mass)
    float(params.wheelbase)
    return params
  except Exception as exc:
    raise BehaviorReplayError("route CarParams cannot be decoded") from exc


def _route_runtime(
  *,
  vehicle_identity: str,
  physical_profile: VehicleCalibrationProfile,
  first_input: ReplayFrameInput,
  nominal_rack_mapping: RackMappingSnapshot,
  provisional_dynamics: ProvisionalRackDynamics,
  interface_registry: Mapping[str, type] | None,
) -> _RouteRuntime:
  if first_input.car_params_bytes is None:
    raise BehaviorReplayError("first replay frame must carry canonical CarParams")
  params = decode_behavior_car_params(first_input.car_params_bytes)
  try:
    bundle, interface, _ = build_detected_runtime_bundle(
      car_params=params,
      provisional_rack_dynamics=provisional_dynamics,
      interface_registry=interface_registry,
    )
    profile = compose_controller_profile(
      physical_profile,
      bundle.seed_profile,
    )
  except Exception as exc:
    raise BehaviorReplayError("route runtime bundle cannot be reconstructed") from exc
  if bundle.vehicle_identity != vehicle_identity:
    raise BehaviorReplayError("route runtime identity differs from decoded route")
  if profile.vehicle_identity != vehicle_identity:
    raise BehaviorReplayError("physical profile belongs to a different vehicle")
  if nominal_rack_mapping != bundle.nominal_rack_mapping:
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
  planned_requested_torque: float,
  reachable_counts: int,
  applied_counts: int,
  requested_counts: int,
  limits: RuntimeTorqueLimits,
  steering_request_active: bool,
  maximum_authority_required: bool,
  response_eligible: bool,
  controller_fault: bool,
) -> ControllerFrameOutput:
  raw = float(raw_requested_torque)
  planned = float(planned_requested_torque)
  if not math.isfinite(raw) or not math.isfinite(planned):
    raw = 0.0
    planned = 0.0
    controller_fault = True
    response_eligible = False
  return ControllerFrameOutput(
    mono_time_ns=mono_time_ns,
    measured_curvature_1pm=float(measured_curvature),
    measured_rack_angle_deg=float(state.angle_deg),
    measured_rack_rate_deg_s=float(state.rate_deg_s),
    measured_rack_accel_deg_s2=float(acceleration_deg_s2),
    raw_requested_torque=raw,
    planned_requested_torque=planned,
    reachable_envelope_torque=reachable_counts / limits.steer_max,
    envelope_applied_torque=applied_counts / limits.steer_max,
    torque_headroom=_headroom(limits, requested_counts),
    actuator_constrained=applied_counts != requested_counts,
    steering_request_active=bool(steering_request_active),
    maximum_authority_required=bool(maximum_authority_required),
    controller_fault=bool(controller_fault),
    response_eligible=bool(response_eligible),
  )


class _RequestProducer:
  """Episode-local controller interface; never owns the plant or envelope."""

  def reset(
    self,
    *,
    initial_state: RackState,
  ) -> None:
    raise NotImplementedError

  def request_torque(
    self,
    *,
    control: CanonicalBehaviorControlInput,
    frame_input: ReplayFrameInput,
    model_intent: SparseModelBehaviorIntent | None,
    reference: BehaviorReferenceAtControl | None,
    state: RackState,
    acceleration_deg_s2: float,
    previous_applied_counts: int,
    previous_steering_request_active: bool,
    steering_request_fault_avoidance_counter: int,
    steering_request_state_valid: bool,
    previous_output_constrained: bool,
    engagement_boundary: bool,
  ) -> tuple[float, float, bool]:
    raise NotImplementedError


class _ModularRequestProducer(_RequestProducer):
  def __init__(
    self,
    policy: BehaviorPolicy,
    runtime: _RouteRuntime,
  ) -> None:
    horizon_policy = HorizonPolicy.from_json_file(
      PROVISIONAL_HORIZON_POLICY_PATH,
    )
    self.core = ModularControllerCore(
      fixed_dt_s=DT_CTRL,
      profile=runtime.controller_profile,
      tracking_policy=TrackingPolicy(
        natural_frequency_per_s=policy.natural_frequency_per_s,
        damping_ratio=policy.damping_ratio,
      ),
      # A twin cannot produce a non-degenerate response innovation against
      # itself. Counterfactual observer learning is therefore explicitly off.
      observer_policy=None,
      nominal_mapping=runtime.runtime_bundle.nominal_rack_mapping,
      runtime_limits=runtime.runtime_bundle.torque_limits,
      horizon_policy=horizon_policy,
      plan_capacity=INTENT_CAPACITY,
    )
    self.params = runtime.car_params
    self.profile = runtime.controller_profile
    self.steer_max = runtime.runtime_bundle.torque_limits.steer_max
    self.plan_times = [0.0] * INTENT_CAPACITY
    self.orientation_rates = [0.0] * INTENT_CAPACITY
    self.velocities = [0.0] * INTENT_CAPACITY
    self.plan_curvatures = [0.0] * INTENT_CAPACITY

  def reset(
    self,
    *,
    initial_state: RackState,
  ) -> None:
    # The stepper constructs a fresh producer at every active episode boundary;
    # this reset only primes the episode's exact applied-torque anchor.
    self.core.prime_applied_history(initial_state.applied_torque)

  def request_torque(
    self,
    *,
    control: CanonicalBehaviorControlInput,
    frame_input: ReplayFrameInput,
    model_intent: SparseModelBehaviorIntent | None,
    reference: BehaviorReferenceAtControl | None,
    state: RackState,
    acceleration_deg_s2: float,
    previous_applied_counts: int,
    previous_steering_request_active: bool,
    steering_request_fault_avoidance_counter: int,
    steering_request_state_valid: bool,
    previous_output_constrained: bool,
    engagement_boundary: bool,
  ) -> tuple[float, float, bool]:
    if model_intent is None or reference is None:
      return 0.0, 0.0, True
    if model_intent.model_frame_id != frame_input.model_frame_id:
      raise BehaviorReplayError("model frame identity differs from the exact link")
    parameters = self.profile.parameters_at(control.speed_mps).parameters
    adaptation = adapt_model_intent_into(
      state_sample_mono_ns=frame_input.state_sample_mono_time_ns,
      control_witness_mono_ns=control.mono_time_ns,
      model_publication_mono_ns=model_intent.publication_mono_time_ns,
      plan_origin_mono_ns=model_intent.plan_origin_mono_time_ns,
      model_frame_id=frame_input.model_frame_id,
      message_valid=frame_input.model_message_valid,
      message_alive=frame_input.model_message_alive,
      scalar_desired_curvature=model_intent.scalar_curvature_1pm,
      published_desired_curvature_time_s=model_intent.scalar_action_plan_s,
      native_plan_times_s=model_intent.native_times_s,
      native_orientation_rates_z=model_intent.orientation_rates_z,
      native_velocities_x=model_intent.velocities_x,
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
      scalar_curvature=model_intent.scalar_curvature_1pm,
      current_v_ego_m_s=control.speed_mps,
      measured_rack_angle_deg=state.angle_deg,
      measured_rack_rate_deg_s=state.rate_deg_s,
      measured_rack_acceleration_deg_s2=acceleration_deg_s2,
      # The controller receives the previous torque actually placed on CAN.
      # Rack-effective torque is delay-line/plant state and must never be fed
      # back as if the actuator had emitted it this frame.
      previous_command_counts=previous_applied_counts,
      recorded_applied_torque=(
        previous_applied_counts / self.steer_max
        if previous_steering_request_active
        else 0.0
      ),
      driver_torque=frame_input.driver_torque,
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
      steering_request_fault_avoidance_counter=(
        steering_request_fault_avoidance_counter
      ),
      steering_request_state_valid=steering_request_state_valid,
      actuator_constrained=previous_output_constrained,
      output_constrained=previous_output_constrained,
      standstill=frame_input.standstill,
    )
    if result.valid and result.status in _LIVE_CORE_STATUSES:
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
      return float(result.raw_torque), float(result.planned_torque), False
    return 0.0, 0.0, True


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

  def reset(
    self,
    *,
    initial_state: RackState,
  ) -> None:
    self.desired_curvature = 0.0
    self.desired_curvature_initialized = False
    del initial_state

  def request_torque(
    self,
    *,
    control: CanonicalBehaviorControlInput,
    frame_input: ReplayFrameInput,
    model_intent: SparseModelBehaviorIntent | None,
    reference: BehaviorReferenceAtControl | None,
    state: RackState,
    acceleration_deg_s2: float,
    previous_applied_counts: int,
    previous_steering_request_active: bool,
    steering_request_fault_avoidance_counter: int,
    steering_request_state_valid: bool,
    previous_output_constrained: bool,
    engagement_boundary: bool,
  ) -> tuple[float, float, bool]:
    del (
      acceleration_deg_s2,
      previous_applied_counts,
      previous_steering_request_active,
      steering_request_fault_avoidance_counter,
      steering_request_state_valid,
      engagement_boundary,
      reference,
    )
    if model_intent is None:
      return 0.0, 0.0, True
    if model_intent.model_frame_id != frame_input.model_frame_id:
      raise BehaviorReplayError("model frame identity differs from the exact link")
    if not frame_input.live_torque_health_exact:
      # Stock conditionally updates its stateful calibration from
      # sm.all_checks(['liveTorqueParameters']).  If that witness-time result
      # cannot be proved, there is no exact stock replay for this frame.
      return 0.0, 0.0, True
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
        else model_intent.scalar_curvature_1pm
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
        return 0.0, 0.0, True
      return raw_float, raw_float, False
    except (
      AttributeError,
      TypeError,
      ValueError,
      OverflowError,
      ZeroDivisionError,
    ):
      return 0.0, 0.0, True


class BehaviorReplayStepper:
  """Route-bounded, constant-memory counterfactual episode simulator.

  ``step`` consumes one canonical control witness at a time together with its
  already-linked model intent and derived reference.  The stepper owns only
  controller/plant episode state and fixed-capacity intent buffers; it never
  retains route controls, model publications, references, or output arrays.

  A non-``None`` policy selects the modular controller.  ``None`` selects the
  exact source-stock torque controller. Both paths share this exact envelope,
  censoring, lifecycle, and plant implementation.
  """

  def __init__(
    self,
    *,
    vehicle_identity: str,
    physical_profile: VehicleCalibrationProfile,
    policy: BehaviorPolicy | None,
    first_frame_input: ReplayFrameInput,
    nominal_rack_mapping: RackMappingSnapshot,
    provisional_dynamics: ProvisionalRackDynamics,
    plant_member: CounterfactualPlantMember | None = None,
    interface_registry: Mapping[str, type] | None = None,
  ) -> None:
    if not vehicle_identity.strip():
      raise ValueError("route vehicle identity must not be empty")
    if not isinstance(physical_profile, VehicleCalibrationProfile):
      raise TypeError("behavior stepper requires a physical profile")
    if policy is not None and not isinstance(policy, BehaviorPolicy):
      raise TypeError("behavior stepper policy has the wrong type")
    if not isinstance(first_frame_input, ReplayFrameInput):
      raise TypeError("behavior stepper requires a canonical first frame")
    if not isinstance(nominal_rack_mapping, RackMappingSnapshot):
      raise TypeError("behavior stepper requires a nominal rack mapping")
    if not isinstance(provisional_dynamics, ProvisionalRackDynamics):
      raise TypeError("behavior stepper requires explicit provisional dynamics")
    registry = None if interface_registry is None else dict(interface_registry)
    self.runtime = _route_runtime(
      vehicle_identity=vehicle_identity,
      physical_profile=physical_profile,
      first_input=first_frame_input,
      nominal_rack_mapping=nominal_rack_mapping,
      provisional_dynamics=provisional_dynamics,
      interface_registry=registry,
    )
    self.policy = policy
    self.limits = self.runtime.runtime_bundle.torque_limits
    if plant_member is None:
      # Compatibility-only diagnostic member for legacy callers. Selection
      # authorities pass an independently identified member explicitly.
      plant_member = CounterfactualPlantMember.create(
        rack_gain_deg_s2_per_torque=(
          provisional_dynamics.rack_gain_deg_s2_per_torque
        ),
        rack_damping_per_s=provisional_dynamics.rack_damping_per_s,
        delay_offset_s=0.0,
        unresolved_load_torque=0.0,
      )
    if not isinstance(plant_member, CounterfactualPlantMember):
      raise TypeError("behavior stepper requires a counterfactual plant member")
    self.plant_member = plant_member
    maximum_delay = max(
      self.plant_member.effective_delay_s(
        node.parameters.transport_delay_s,
      )
      for node in self.runtime.controller_profile.nodes
    )
    self.delay_line = AppliedTorqueDelayLine(
      fixed_dt_s=DT_CTRL,
      maximum_delay_s=maximum_delay,
    )
    self._car_params_bytes = first_frame_input.car_params_bytes
    self.reset()

  def reset(self) -> None:
    """Start a fresh traversal of the route bound at construction."""
    self.state: RackState | None = None
    self.acceleration_deg_s2 = 0.0
    self.previous_applied_counts = 0
    self.previous_requested_counts = 0
    self.previous_steering_request_active = False
    self.steering_request_fault_avoidance_counter = 0
    self.previous_active = False
    self.censored = False
    self.episode_faulted = False
    self.producer: _RequestProducer | None = None
    self._previous_physical_record_index: int | None = None
    self._previous_control_mono_time_ns: int | None = None

  def _fresh_producer(self) -> _RequestProducer:
    return (
      _ModularRequestProducer(self.policy, self.runtime)
      if self.policy is not None
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
    # The route's recorded request belongs to the controller which happened
    # to drive it. It cannot perturb stock or candidate counterfactual state.
    self.previous_requested_counts = frame_input.recorded_applied_counts
    self.previous_steering_request_active = True
    self.steering_request_fault_avoidance_counter = 0
    self.delay_line.reset(applied)
    self.censored = False
    self.episode_faulted = False
    self.producer = self._fresh_producer()
    self.producer.reset(
      initial_state=self.state,
    )
    return True

  def _inactive_output(
    self,
    control: CanonicalBehaviorControlInput,
    frame_input: ReplayFrameInput,
    *,
    controller_fault: bool = False,
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
      planned_requested_torque=0.0,
      reachable_counts=counts,
      applied_counts=counts,
      requested_counts=counts,
      limits=self.limits,
      steering_request_active=False,
      maximum_authority_required=False,
      response_eligible=False,
      controller_fault=controller_fault,
    )

  def _faulted_output(
    self,
    control: CanonicalBehaviorControlInput,
  ) -> ControllerFrameOutput:
    assert self.state is not None
    try:
      curvature = _measured_curvature(
        self.state,
        control.speed_mps,
        control.live_rack_mapping,
        control.nominal_rack_mapping,
      )
    except (TypeError, ValueError, OverflowError):
      curvature = 0.0
    return _output(
      mono_time_ns=control.mono_time_ns,
      state=self.state,
      acceleration_deg_s2=self.acceleration_deg_s2,
      measured_curvature=curvature,
      raw_requested_torque=self.previous_applied_counts / self.limits.steer_max,
      planned_requested_torque=self.previous_applied_counts / self.limits.steer_max,
      reachable_counts=self.previous_applied_counts,
      applied_counts=self.previous_applied_counts,
      requested_counts=self.previous_applied_counts,
      limits=self.limits,
      steering_request_active=False,
      maximum_authority_required=False,
      response_eligible=False,
      controller_fault=True,
    )

  def _validate_frame(
    self,
    control: CanonicalBehaviorControlInput,
    frame_input: ReplayFrameInput,
    model_intent: SparseModelBehaviorIntent | None,
    reference: BehaviorReferenceAtControl | None,
  ) -> None:
    if not isinstance(control, CanonicalBehaviorControlInput):
      raise TypeError("behavior step requires canonical control context")
    if not isinstance(frame_input, ReplayFrameInput):
      raise TypeError("behavior step requires canonical frame input")
    if model_intent is not None and not isinstance(model_intent, SparseModelBehaviorIntent):
      raise TypeError("behavior step model context has the wrong type")
    if reference is not None and not isinstance(reference, BehaviorReferenceAtControl):
      raise TypeError("behavior step reference context has the wrong type")
    if control.core_input != frame_input.to_bytes():
      raise BehaviorReplayError("control context and canonical frame input differ")
    if control.nominal_rack_mapping != self.runtime.runtime_bundle.nominal_rack_mapping:
      raise BehaviorReplayError("decoded nominal geometry differs from CarParams")
    linked = control.model_publication_index is not None
    if linked != (model_intent is not None):
      raise BehaviorReplayError("control/model context availability differs")
    if model_intent is not None:
      if model_intent.publication_mono_time_ns > control.mono_time_ns:
        raise BehaviorReplayError("control context references a future model")
      if reference is not None and (
        reference.model_publication_mono_time_ns
        != model_intent.publication_mono_time_ns
      ):
        raise BehaviorReplayError("reference/model context identity differs")
    if self._previous_physical_record_index is None:
      if frame_input.car_params_bytes != self._car_params_bytes:
        raise BehaviorReplayError("first replay step differs from route CarParams")
    else:
      if frame_input.car_params_bytes is not None:
        raise BehaviorReplayError("CarParams may appear only at route start")
      if frame_input.physical_record_index <= self._previous_physical_record_index:
        raise BehaviorReplayError("physical record links must be strictly increasing")
      assert self._previous_control_mono_time_ns is not None
      if control.mono_time_ns <= self._previous_control_mono_time_ns:
        raise BehaviorReplayError("control inputs must be strictly time ordered")

  def step(
    self,
    *,
    control: CanonicalBehaviorControlInput,
    frame_input: ReplayFrameInput,
    model_intent: SparseModelBehaviorIntent | None,
    reference: BehaviorReferenceAtControl | None,
  ) -> ControllerFrameOutput:
    """Advance one frame without retaining any caller-owned route data."""
    self._validate_frame(control, frame_input, model_intent, reference)
    self._previous_physical_record_index = frame_input.physical_record_index
    self._previous_control_mono_time_ns = control.mono_time_ns

    active = bool(control.lateral_active)
    engagement_boundary = active and not self.previous_active
    if not active:
      output = self._inactive_output(control, frame_input)
      self.state = None
      self.producer = None
      self.censored = False
      self.episode_faulted = False
      self.previous_active = False
      return output

    if engagement_boundary and not self._bootstrap(frame_input):
      self.previous_active = True
      return self._inactive_output(control, frame_input, controller_fault=True)
    self.previous_active = True
    if control.driver_intervention_onset or frame_input.intervention_onset_uncertain:
      self.censored = True
    if self.state is None or self.producer is None:
      self.episode_faulted = True
      self.censored = True
      return self._inactive_output(control, frame_input, controller_fault=True)

    runtime_valid = (
      control.inputs_valid
      and frame_input.witness_resolved
      and frame_input.control_cadence_valid
      and frame_input.live_parameters_inputs_valid
    )
    if not runtime_valid:
      self.episode_faulted = True
    if control.platform_fault:
      self.episode_faulted = True
      self.censored = True
    previous_output_constrained = (
      abs(
        self.previous_requested_counts - self.previous_applied_counts,
      ) / self.limits.steer_max > 1e-2
    )
    if self.episode_faulted:
      return self._faulted_output(control)
    raw = 0.0
    planned = 0.0
    controller_fault = self.episode_faulted
    if not self.episode_faulted:
      raw, planned, controller_fault = self.producer.request_torque(
        control=control,
        frame_input=frame_input,
        model_intent=model_intent,
        reference=reference,
        state=self.state,
        acceleration_deg_s2=self.acceleration_deg_s2,
        previous_applied_counts=self.previous_applied_counts,
        previous_steering_request_active=(
          self.previous_steering_request_active
        ),
        steering_request_fault_avoidance_counter=(
          self.steering_request_fault_avoidance_counter
        ),
        steering_request_state_valid=True,
        previous_output_constrained=previous_output_constrained,
        engagement_boundary=engagement_boundary,
      )
    if not math.isfinite(raw) or not math.isfinite(planned):
      raw = 0.0
      planned = 0.0
      controller_fault = True
    if controller_fault:
      self.episode_faulted = True
      self.censored = True
      return self._faulted_output(control)
    raw_counts = int(round(raw * self.limits.steer_max))
    requested_counts = int(round(planned * self.limits.steer_max))
    try:
      applied_counts = apply_torque_envelope_counts(
        self.limits,
        requested_counts,
        self.previous_applied_counts,
        frame_input.driver_torque,
      )
      raw_direction = (raw_counts > 0) - (raw_counts < 0)
      reachable_counts = apply_torque_envelope_counts(
        self.limits,
        raw_direction * self.limits.steer_max,
        self.previous_applied_counts,
        frame_input.driver_torque,
      )
      maximum_authority_required = (
        raw_direction != 0
        and raw_direction * raw_counts >= raw_direction * reachable_counts
      )
      curvature = _measured_curvature(
        self.state,
        control.speed_mps,
        control.live_rack_mapping,
        control.nominal_rack_mapping,
      )
      (
        next_steering_request_counter,
        steering_request_active,
      ) = apply_steering_request_fault_avoidance(
        self.state.angle_deg,
        True,
        self.steering_request_fault_avoidance_counter,
      )
    except (TypeError, ValueError, OverflowError):
      self.episode_faulted = True
      self.censored = True
      return self._faulted_output(control)
    response_eligible = (
      runtime_valid
      and not self.censored
      and not controller_fault
    )
    output = _output(
      mono_time_ns=control.mono_time_ns,
      state=self.state,
      acceleration_deg_s2=self.acceleration_deg_s2,
      measured_curvature=curvature,
      raw_requested_torque=raw,
      planned_requested_torque=planned,
      reachable_counts=reachable_counts,
      applied_counts=applied_counts,
      requested_counts=requested_counts,
      limits=self.limits,
      steering_request_active=steering_request_active,
      maximum_authority_required=maximum_authority_required,
      response_eligible=response_eligible,
      controller_fault=controller_fault,
    )

    selected_mapping = (
      control.live_rack_mapping
      if control.live_rack_mapping is not None and control.live_rack_mapping.valid
      else control.nominal_rack_mapping
    )
    parameters = self.runtime.controller_profile.parameters_at(
      control.speed_mps,
    ).parameters
    try:
      can_applied_torque = applied_counts / self.limits.steer_max
      rack_effective_torque = self.delay_line.commit_and_sample(
        can_applied_torque if steering_request_active else 0.0,
        self.plant_member.effective_delay_s(parameters.transport_delay_s),
      )
      plant_step = step_counterfactual_plant(
        state=self.state,
        rack_effective_torque=rack_effective_torque,
        speed_mps=control.speed_mps,
        mapping=selected_mapping,
        nominal_mapping=control.nominal_rack_mapping,
        lateral_accel_offset=(
          float(self.runtime.car_params.lateralTuning.torque.latAccelOffset)
          + parameters.lateral_accel_offset_correction_mps2
        ),
        base_parameters=parameters,
        member=self.plant_member,
        dt=DT_CTRL,
      )
      # RackState remains the controller-facing state: its applied torque is
      # the latest command placed on CAN. The delayed rack-effective value is
      # owned exclusively by the delay line.
      self.state = RackState(
        plant_step.state.angle_deg,
        plant_step.state.rate_deg_s,
        can_applied_torque,
      )
      self.acceleration_deg_s2 = plant_step.acceleration_deg_s2
    except (TypeError, ValueError, OverflowError):
      self.episode_faulted = True
      self.censored = True
      return self._faulted_output(control)
    self.previous_requested_counts = requested_counts
    self.previous_applied_counts = applied_counts
    self.previous_steering_request_active = steering_request_active
    self.steering_request_fault_avoidance_counter = (
      next_steering_request_counter
    )
    return output


@dataclass(frozen=True, slots=True)
class _ReviewedBehaviorRouteReplay:
  """Fixed execution adapter paired with a reviewed core identity.

  Keeping this as an exact private type, instead of annotating an arbitrary
  callback, lets production orchestration distinguish the reviewed adapter
  from test doubles.  Python constructors are not a security boundary; the
  clean source commit is.  The useful invariant is that every instance runs
  this implementation rather than caller-supplied controller code.
  """

  identity: ReplayCoreIdentity
  provisional_dynamics: ProvisionalRackDynamics
  interface_registry: Mapping[str, type] | None
  modular: bool

  def __call__(
    self,
    request: ControllerReplayRequest,
  ) -> Iterable[ControllerFrameOutput]:
    if not isinstance(request, ControllerReplayRequest):
      raise TypeError("behavior replay request has the wrong type")
    if request.artifact_identity.core != self.identity:
      raise BehaviorReplayError(
        "behavior replay request/core identities do not match",
      )
    if self.modular != (request.policy is not None):
      raise BehaviorReplayError(
        "behavior replay core and policy presence do not match",
      )
    inputs = _decode_inputs(request.route)
    stepper = BehaviorReplayStepper(
      vehicle_identity=request.route.vehicle_identity,
      physical_profile=request.physical_profile,
      policy=request.policy,
      first_frame_input=inputs[0],
      nominal_rack_mapping=request.route.control_inputs[0].nominal_rack_mapping,
      provisional_dynamics=self.provisional_dynamics,
      interface_registry=self.interface_registry,
    )
    # Keep this legacy whole-route callback as a zero-semantic-change adapter
    # over the public streaming core.
    outputs: list[ControllerFrameOutput] = []
    for control, frame_input, reference in zip(
      request.route.control_inputs,
      inputs,
      request.references,
      strict=True,
    ):
      model_intent = (
        None
        if control.model_publication_index is None
        else request.route.model_publications[control.model_publication_index]
      )
      outputs.append(stepper.step(
        control=control,
        frame_input=frame_input,
        model_intent=model_intent,
        reference=reference,
      ))
    return tuple(outputs)


def _make_replay_core(
  identity: ReplayCoreIdentity,
  *,
  provisional_dynamics: ProvisionalRackDynamics,
  interface_registry: Mapping[str, type] | None,
  modular: bool,
) -> BehaviorReplayCore:
  validate_reviewed_replay_core_identity(identity, exact_stock=not modular)
  if not isinstance(provisional_dynamics, ProvisionalRackDynamics):
    raise TypeError("behavior replay requires explicit provisional rack dynamics")
  adapter = _ReviewedBehaviorRouteReplay(
    identity=identity,
    provisional_dynamics=provisional_dynamics,
    interface_registry=(
      None if interface_registry is None else dict(interface_registry)
    ),
    modular=modular,
  )
  return BehaviorReplayCore(identity=identity, replay_route=adapter)


def validate_reviewed_behavior_replay_core(
  core: BehaviorReplayCore,
  *,
  exact_stock: bool,
) -> None:
  """Require the fixed adapter as well as its reviewed identity digest."""
  if not isinstance(core, BehaviorReplayCore):
    raise TypeError("behavior replay core has the wrong type")
  validate_reviewed_replay_core_identity(core.identity, exact_stock=exact_stock)
  adapter = core.replay_route
  if (
    type(adapter) is not _ReviewedBehaviorRouteReplay
    or adapter.identity != core.identity
    or adapter.modular == exact_stock
    or adapter.interface_registry is not None
  ):
    raise BehaviorReplayError(
      "behavior replay core does not use the reviewed execution adapter",
    )


def reviewed_behavior_replay_dynamics(
  core: BehaviorReplayCore,
  *,
  exact_stock: bool,
) -> ProvisionalRackDynamics:
  """Return the frozen dynamics after admitting the reviewed replay adapter."""
  validate_reviewed_behavior_replay_core(core, exact_stock=exact_stock)
  adapter = core.replay_route
  assert type(adapter) is _ReviewedBehaviorRouteReplay
  return adapter.provisional_dynamics


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

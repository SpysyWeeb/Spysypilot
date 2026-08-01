"""Canonical shared route evidence for the two offroad BLaTv2 authorities.

The artifact is deliberately controller independent.  A route is decoded once
per independent authority into one immutable container: the existing physical
``MeasuredLearningFrame`` wire plane is stored exactly once, while the compact
behavior planes preserve the clocks and source values needed for exact replay.

Version 2 intentionally has no compatibility path for the old physical-only
``BLATSP01`` spool.  Treating an old spool as complete behavior evidence would
silently train on missing model/controller context, so it fails closed.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import tempfile
from typing import Any


ROUTE_EVIDENCE_MAGIC = b"BLATRE02"
ROUTE_EVIDENCE_VERSION = 2
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_CAR_PARAMS_BYTES = 1024 * 1024
MAX_PHYSICAL_RECORDS = 1_000_000
MAX_MODEL_PUBLICATIONS = 250_000
MAX_CONTROL_WITNESSES = 1_000_000
MAX_SPARSE_PUBLICATIONS = 250_000
MAX_EVENT_LOCATORS = 100_000
MAX_NATIVE_PLAN_SAMPLES = 1024
MAX_EVENT_ID_BYTES = 256
MAX_EVENT_TYPE_BYTES = 128
MAX_EVENT_SEVERITY_BYTES = 64
MAX_STATUS_BYTES = 128

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
# manifest, CarParams, physical, model, controls, torque, delay, maneuver, event
_HEADER = struct.Struct("<8sHH9Q")
_MODEL = struct.Struct("<IIQqqddBBBBH")
_CONTROL = struct.Struct("<IIQIiiiiQqqQqqddddi18B")
_TORQUE = struct.Struct("<IIQdddqBBB")
_DELAY = struct.Struct("<IIQdqBBH")
_MANEUVER = struct.Struct("<IIQdBB")
_EVENT = struct.Struct("<IIQQddBHHH")


class RouteEvidenceError(RuntimeError):
  """An evidence artifact or store operation violates its closed contract."""


def _finite(value: object, field: str) -> float:
  if type(value) not in (int, float) or not math.isfinite(float(value)):
    raise RouteEvidenceError(f"{field} must be finite")
  return float(value)


def _uint(value: object, field: str, maximum: int = (1 << 64) - 1) -> int:
  if type(value) is not int or value < 0 or value > maximum:
    raise RouteEvidenceError(f"{field} is out of range")
  return value


def _sint(value: object, field: str, minimum: int, maximum: int) -> int:
  if type(value) is not int or value < minimum or value > maximum:
    raise RouteEvidenceError(f"{field} is out of range")
  return value


def _boolean(value: object, field: str) -> bool:
  if type(value) is not bool:
    raise RouteEvidenceError(f"{field} must be bool")
  return value


def _text(value: object, field: str, maximum: int) -> str:
  if type(value) is not str or not value or "\x00" in value:
    raise RouteEvidenceError(f"{field} is invalid")
  try:
    encoded = value.encode("utf-8")
  except UnicodeEncodeError as error:
    raise RouteEvidenceError(f"{field} is invalid UTF-8") from error
  if len(encoded) > maximum:
    raise RouteEvidenceError(f"{field} exceeds its bound")
  return value


def _hash(value: object, field: str) -> str:
  if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
    raise RouteEvidenceError(f"{field} must be a SHA-256")
  return value


def _commit(value: object, field: str) -> str:
  if type(value) is not str or _COMMIT_RE.fullmatch(value) is None:
    raise RouteEvidenceError(f"{field} must be a full commit")
  return value


def _canonical_json(value: object) -> bytes:
  try:
    return json.dumps(
      value, allow_nan=False, separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
  except (TypeError, ValueError) as error:
    raise RouteEvidenceError("manifest is not canonical JSON") from error


@dataclass(frozen=True, slots=True)
class RouteEvidenceSourceIdentity:
  route_id: str
  route_time_origin_mono_ns: int
  route_segment_sha256: tuple[str, ...]
  route_segment_size_bytes: tuple[int, ...]
  source_superproject_commit: str
  source_opendbc_commit: str
  source_panda_commit: str
  controller_source_kind: str
  controller_artifact_sha256: str
  behavior_eligible: bool
  behavior_ineligible_reason: str
  vehicle_identity: str
  runtime_identity: str
  schema_versions: Mapping[str, int]
  preparation_provenance: Mapping[str, object]
  physical_plane_encoding_id: str
  physical_record_count: int
  preparation_cache_key: str
  controls_witness_count: int
  unresolved_witness_count: int
  gap_count: int
  model_link_failure_count: int
  pre_poll_dropped_timestamps_ns: tuple[int, ...] = ()

  def __post_init__(self) -> None:
    _text(self.route_id, "route id", 1024)
    _uint(self.route_time_origin_mono_ns, "route time origin")
    if (
      type(self.route_segment_sha256) is not tuple
      or not self.route_segment_sha256
      or len(self.route_segment_sha256) > 128
      or type(self.route_segment_size_bytes) is not tuple
      or len(self.route_segment_sha256) != len(self.route_segment_size_bytes)
    ):
      raise RouteEvidenceError("route segment identity is invalid")
    for value in self.route_segment_sha256:
      _hash(value, "route segment hash")
    for value in self.route_segment_size_bytes:
      _uint(value, "route segment size", 1 << 40)
    _commit(self.source_superproject_commit, "source superproject commit")
    _commit(self.source_opendbc_commit, "source opendbc commit")
    _commit(self.source_panda_commit, "source panda commit")
    if self.controller_source_kind not in {
      "stock_canonical", "modular_artifact", "ineligible",
    }:
      raise RouteEvidenceError("controller source kind is invalid")
    _hash(self.controller_artifact_sha256, "controller artifact hash")
    _boolean(self.behavior_eligible, "behavior eligible")
    _text(self.behavior_ineligible_reason, "behavior reason", 256)
    if self.behavior_eligible != (self.behavior_ineligible_reason == "eligible"):
      raise RouteEvidenceError("behavior eligibility and reason disagree")
    if self.behavior_eligible and self.controller_source_kind == "ineligible":
      raise RouteEvidenceError("eligible evidence lacks a controller source")
    _text(self.vehicle_identity, "vehicle identity", 4096)
    _text(self.runtime_identity, "runtime identity", 4096)
    if type(self.schema_versions) is not dict or not self.schema_versions:
      raise RouteEvidenceError("schema versions are invalid")
    for key, value in self.schema_versions.items():
      _text(key, "schema version key", 128)
      _uint(value, "schema version", (1 << 31) - 1)
    if type(self.preparation_provenance) is not dict:
      raise RouteEvidenceError("preparation provenance is invalid")
    _canonical_json(self.preparation_provenance)
    _text(self.physical_plane_encoding_id, "physical encoding", 128)
    _uint(self.physical_record_count, "physical records", MAX_PHYSICAL_RECORDS)
    _hash(self.preparation_cache_key, "preparation cache key")
    _uint(self.controls_witness_count, "controls witnesses", MAX_CONTROL_WITNESSES)
    _uint(self.unresolved_witness_count, "unresolved witnesses", self.controls_witness_count)
    _uint(self.gap_count, "gap count", self.controls_witness_count)
    _uint(self.model_link_failure_count, "model link failures", self.controls_witness_count)
    if type(self.pre_poll_dropped_timestamps_ns) is not tuple:
      raise RouteEvidenceError("pre-poll timestamps must be a tuple")
    previous = -1
    for value in self.pre_poll_dropped_timestamps_ns:
      _uint(value, "pre-poll timestamp")
      if value <= previous:
        raise RouteEvidenceError("pre-poll timestamps are not ordered")
      previous = value

  def manifest_dict(self) -> dict[str, object]:
    payload = asdict(self)
    payload["schema_versions"] = dict(sorted(self.schema_versions.items()))
    payload["preparation_provenance"] = dict(self.preparation_provenance)
    return payload


@dataclass(frozen=True, slots=True)
class ModelPublication:
  segment_index: int
  ordinal: int
  mono_time_ns: int
  frame_id: int
  timestamp_eof_ns: int
  scalar_curvature: float
  desired_curvature_time_s: float
  plan_times: tuple[float, ...]
  orientation_rate_z: tuple[float, ...]
  velocity_x: tuple[float, ...]
  message_valid: bool
  native_grid_valid: bool

  def __post_init__(self) -> None:
    _uint(self.segment_index, "model segment", (1 << 32) - 1)
    _uint(self.ordinal, "model ordinal", (1 << 32) - 1)
    _uint(self.mono_time_ns, "model mono time")
    _sint(self.frame_id, "model frame id", -(1 << 63), (1 << 63) - 1)
    _sint(self.timestamp_eof_ns, "model timestamp EOF", -(1 << 63), (1 << 63) - 1)
    _finite(self.scalar_curvature, "scalar curvature")
    _finite(self.desired_curvature_time_s, "desired curvature time")
    _boolean(self.message_valid, "model message valid")
    _boolean(self.native_grid_valid, "native grid valid")
    arrays = (self.plan_times, self.orientation_rate_z, self.velocity_x)
    if any(type(values) is not tuple for values in arrays):
      raise RouteEvidenceError("model native arrays must be tuples")
    if not self.native_grid_valid:
      if any(values for values in arrays):
        raise RouteEvidenceError("invalid native model grid must be empty")
      return
    if not self.plan_times or len(self.plan_times) > MAX_NATIVE_PLAN_SAMPLES:
      raise RouteEvidenceError("model native grid length is invalid")
    if not (len(self.plan_times) == len(self.orientation_rate_z) == len(self.velocity_x)):
      raise RouteEvidenceError("model native grid lengths disagree")
    for name, values in zip(("model time", "model orientation rate", "model velocity"), arrays, strict=True):
      for value in values:
        _finite(value, name)
    if any(right <= left for left, right in zip(self.plan_times, self.plan_times[1:], strict=False)):
      raise RouteEvidenceError("model times are not strictly increasing")


@dataclass(frozen=True, slots=True)
class ControlsWitness:
  segment_index: int
  ordinal: int
  mono_time_ns: int
  physical_record_index: int
  model_publication_index: int
  live_torque_parameters_index: int
  live_delay_index: int
  lateral_maneuver_plan_index: int
  poll_mono_time_ns: int
  state_sample_mono_ns: int
  live_parameters_mono_ns: int
  car_output_report_mono_ns: int
  car_output_effective_mono_ns: int
  car_control_mono_ns: int
  raw_request_torque: float
  measured_curvature: float
  desired_curvature: float
  envelope_headroom: float
  torque_output_can_count: int
  message_valid: bool
  model_message_alive: bool
  model_link_valid: bool
  inputs_valid: bool
  lateral_active: bool
  driver_intervening: bool
  steer_fault: bool
  intervention_onset: bool
  intervention_onset_uncertain: bool
  race_unresolved: bool
  gap_from_previous: bool
  car_control_paired: bool
  torque_output_can_valid: bool
  maneuver_plan_available: bool
  live_torque_parameters_available: bool
  live_delay_available: bool
  live_torque_parameters_checks_passed: bool
  live_torque_parameters_health_exact: bool

  def __post_init__(self) -> None:
    _uint(self.segment_index, "control segment", (1 << 32) - 1)
    _uint(self.ordinal, "control ordinal", (1 << 32) - 1)
    _uint(self.mono_time_ns, "control mono time")
    _uint(self.physical_record_index, "physical record index", MAX_PHYSICAL_RECORDS - 1)
    for field, value in (
      ("model publication index", self.model_publication_index),
      ("live torque index", self.live_torque_parameters_index),
      ("live delay index", self.live_delay_index),
      ("maneuver plan index", self.lateral_maneuver_plan_index),
    ):
      _sint(value, field, -1, (1 << 31) - 1)
    for field, value in (
      ("poll mono time", self.poll_mono_time_ns),
      ("state mono time", self.state_sample_mono_ns),
      ("live parameters mono time", self.live_parameters_mono_ns),
      ("car output report mono time", self.car_output_report_mono_ns),
      ("car output effective mono time", self.car_output_effective_mono_ns),
      ("car control mono time", self.car_control_mono_ns),
    ):
      _sint(value, field, -1, (1 << 63) - 1)
    _finite(self.raw_request_torque, "raw request torque")
    _finite(self.measured_curvature, "measured curvature")
    _finite(self.desired_curvature, "desired curvature")
    headroom = _finite(self.envelope_headroom, "envelope headroom")
    if headroom < 0.0 or headroom > 1.0:
      raise RouteEvidenceError("envelope headroom is outside [0,1]")
    _sint(self.torque_output_can_count, "torque output CAN count", -(1 << 31), (1 << 31) - 1)
    for field in (
      "message_valid", "model_message_alive", "model_link_valid",
      "inputs_valid", "lateral_active", "driver_intervening",
      "steer_fault", "intervention_onset", "intervention_onset_uncertain",
      "race_unresolved", "gap_from_previous", "car_control_paired",
      "torque_output_can_valid", "maneuver_plan_available",
      "live_torque_parameters_available", "live_delay_available",
      "live_torque_parameters_checks_passed",
      "live_torque_parameters_health_exact",
    ):
      _boolean(getattr(self, field), field)
    if self.model_link_valid != (self.model_publication_index >= 0):
      raise RouteEvidenceError("model link flag and index disagree")
    if self.car_control_paired != (self.car_control_mono_ns >= 0):
      raise RouteEvidenceError("carControl pairing flag and clock disagree")
    if self.torque_output_can_valid is False and self.torque_output_can_count != 0:
      raise RouteEvidenceError("invalid CAN count must be canonical zero")


@dataclass(frozen=True, slots=True)
class LiveTorqueParametersPublication:
  segment_index: int
  ordinal: int
  mono_time_ns: int
  lat_accel_factor: float
  lat_accel_offset: float
  friction: float
  version: int
  message_valid: bool
  live_valid: bool
  use_params: bool


@dataclass(frozen=True, slots=True)
class LiveDelayPublication:
  segment_index: int
  ordinal: int
  mono_time_ns: int
  lateral_delay_s: float
  version: int
  message_valid: bool
  status: str


@dataclass(frozen=True, slots=True)
class LateralManeuverPlanPublication:
  segment_index: int
  ordinal: int
  mono_time_ns: int
  desired_curvature: float
  message_valid: bool


@dataclass(frozen=True, slots=True)
class DrivingEventLocator:
  segment_index: int
  ordinal: int
  publication_mono_time_ns: int
  occurred_mono_time_ns: int
  analysis_window_before_s: float
  analysis_window_after_s: float
  event_id: str
  event_type: str
  severity: str
  message_valid: bool


def _validate_sparse(records: Sequence[Any], cls: type, name: str, maximum: int) -> None:
  if type(records) is not tuple or len(records) > maximum:
    raise RouteEvidenceError(f"{name} population exceeds its bound")
  previous: tuple[int, int, int] | None = None
  for record in records:
    if type(record) is not cls:
      raise RouteEvidenceError(f"{name} record type is invalid")
    _uint(record.segment_index, f"{name} segment", (1 << 32) - 1)
    _uint(record.ordinal, f"{name} ordinal", (1 << 32) - 1)
    _uint(record.mono_time_ns, f"{name} mono time")
    key = (record.mono_time_ns, record.segment_index, record.ordinal)
    if previous is not None and key <= previous:
      raise RouteEvidenceError(f"{name} records are not ordered")
    previous = key


def _encode_models(records: tuple[ModelPublication, ...]) -> bytes:
  output = bytearray()
  for value in records:
    output.extend(_MODEL.pack(
      value.segment_index, value.ordinal, value.mono_time_ns, value.frame_id,
      value.timestamp_eof_ns, value.scalar_curvature,
      value.desired_curvature_time_s, value.message_valid,
      value.native_grid_valid, 0, 0, len(value.plan_times),
    ))
    for array in (value.plan_times, value.orientation_rate_z, value.velocity_x):
      if array:
        output.extend(struct.pack(f"<{len(array)}d", *array))
  return bytes(output)


def _decode_models(encoded: memoryview, count: int) -> tuple[ModelPublication, ...]:
  values: list[ModelPublication] = []
  offset = 0
  for _ in range(count):
    if offset + _MODEL.size > len(encoded):
      raise RouteEvidenceError("model section is truncated")
    row = _MODEL.unpack_from(encoded, offset)
    offset += _MODEL.size
    if row[9] != 0 or row[10] != 0 or row[7] not in (0, 1) or row[8] not in (0, 1):
      raise RouteEvidenceError("model boolean/reserved byte is non-canonical")
    length = row[11]
    byte_count = length * 8
    arrays: list[tuple[float, ...]] = []
    for _array in range(3):
      if offset + byte_count > len(encoded):
        raise RouteEvidenceError("model native grid is truncated")
      arrays.append(tuple(struct.unpack_from(f"<{length}d", encoded, offset)) if length else ())
      offset += byte_count
    values.append(ModelPublication(
      segment_index=row[0], ordinal=row[1], mono_time_ns=row[2],
      frame_id=row[3], timestamp_eof_ns=row[4], scalar_curvature=row[5],
      desired_curvature_time_s=row[6], plan_times=arrays[0],
      orientation_rate_z=arrays[1], velocity_x=arrays[2],
      message_valid=bool(row[7]), native_grid_valid=bool(row[8]),
    ))
  if offset != len(encoded):
    raise RouteEvidenceError("model section contains trailing bytes")
  return tuple(values)


def _encode_controls(records: tuple[ControlsWitness, ...]) -> bytes:
  output = bytearray(len(records) * _CONTROL.size)
  offset = 0
  for value in records:
    _CONTROL.pack_into(output, offset,
      value.segment_index, value.ordinal, value.mono_time_ns,
      value.physical_record_index, value.model_publication_index,
      value.live_torque_parameters_index, value.live_delay_index,
      value.lateral_maneuver_plan_index, value.poll_mono_time_ns,
      value.state_sample_mono_ns, value.live_parameters_mono_ns,
      value.car_output_report_mono_ns, value.car_output_effective_mono_ns,
      value.car_control_mono_ns, value.raw_request_torque,
      value.measured_curvature, value.desired_curvature,
      value.envelope_headroom, value.torque_output_can_count,
      value.message_valid, value.model_message_alive, value.model_link_valid,
      value.inputs_valid, value.lateral_active, value.driver_intervening,
      value.steer_fault, value.intervention_onset,
      value.intervention_onset_uncertain, value.race_unresolved,
      value.gap_from_previous, value.car_control_paired,
      value.torque_output_can_valid, value.maneuver_plan_available,
      value.live_torque_parameters_available, value.live_delay_available,
      value.live_torque_parameters_checks_passed,
      value.live_torque_parameters_health_exact,
    )
    offset += _CONTROL.size
  return bytes(output)


def _decode_controls(encoded: memoryview, count: int) -> tuple[ControlsWitness, ...]:
  if len(encoded) != count * _CONTROL.size:
    raise RouteEvidenceError("control section size/count disagree")
  values: list[ControlsWitness] = []
  for offset in range(0, len(encoded), _CONTROL.size):
    row = _CONTROL.unpack_from(encoded, offset)
    flags = row[19:]
    if any(flag not in (0, 1) for flag in flags):
      raise RouteEvidenceError("control boolean is non-canonical")
    values.append(ControlsWitness(
      segment_index=row[0], ordinal=row[1], mono_time_ns=row[2],
      physical_record_index=row[3], model_publication_index=row[4],
      live_torque_parameters_index=row[5], live_delay_index=row[6],
      lateral_maneuver_plan_index=row[7], poll_mono_time_ns=row[8],
      state_sample_mono_ns=row[9], live_parameters_mono_ns=row[10],
      car_output_report_mono_ns=row[11], car_output_effective_mono_ns=row[12],
      car_control_mono_ns=row[13], raw_request_torque=row[14],
      measured_curvature=row[15], desired_curvature=row[16],
      envelope_headroom=row[17], torque_output_can_count=row[18],
      message_valid=bool(row[19]), model_message_alive=bool(row[20]),
      model_link_valid=bool(row[21]), inputs_valid=bool(row[22]),
      lateral_active=bool(row[23]), driver_intervening=bool(row[24]),
      steer_fault=bool(row[25]), intervention_onset=bool(row[26]),
      intervention_onset_uncertain=bool(row[27]), race_unresolved=bool(row[28]),
      gap_from_previous=bool(row[29]), car_control_paired=bool(row[30]),
      torque_output_can_valid=bool(row[31]), maneuver_plan_available=bool(row[32]),
      live_torque_parameters_available=bool(row[33]),
      live_delay_available=bool(row[34]),
      live_torque_parameters_checks_passed=bool(row[35]),
      live_torque_parameters_health_exact=bool(row[36]),
    ))
  return tuple(values)


def _encode_torque(records: tuple[LiveTorqueParametersPublication, ...]) -> bytes:
  return b"".join(_TORQUE.pack(
    item.segment_index, item.ordinal, item.mono_time_ns,
    _finite(item.lat_accel_factor, "lat accel factor"),
    _finite(item.lat_accel_offset, "lat accel offset"),
    _finite(item.friction, "friction"),
    _sint(item.version, "torque version", -(1 << 63), (1 << 63) - 1),
    _boolean(item.message_valid, "torque message valid"),
    _boolean(item.live_valid, "torque live valid"),
    _boolean(item.use_params, "torque use params"),
  ) for item in records)


def _decode_torque(encoded: memoryview, count: int) -> tuple[LiveTorqueParametersPublication, ...]:
  if len(encoded) != count * _TORQUE.size:
    raise RouteEvidenceError("torque section size/count disagree")
  result = []
  for offset in range(0, len(encoded), _TORQUE.size):
    row = _TORQUE.unpack_from(encoded, offset)
    if any(flag not in (0, 1) for flag in row[7:]):
      raise RouteEvidenceError("torque boolean is non-canonical")
    result.append(LiveTorqueParametersPublication(*row[:7], *(bool(value) for value in row[7:])))
  return tuple(result)


def _encode_delay(records: tuple[LiveDelayPublication, ...]) -> bytes:
  output = bytearray()
  for item in records:
    status = _text(item.status, "live delay status", MAX_STATUS_BYTES).encode()
    output.extend(_DELAY.pack(
      item.segment_index, item.ordinal, item.mono_time_ns,
      _finite(item.lateral_delay_s, "lateral delay"),
      _sint(item.version, "delay version", -(1 << 63), (1 << 63) - 1),
      _boolean(item.message_valid, "delay message valid"), 0, len(status),
    ))
    output.extend(status)
  return bytes(output)


def _decode_delay(encoded: memoryview, count: int) -> tuple[LiveDelayPublication, ...]:
  output = []
  offset = 0
  for _ in range(count):
    if offset + _DELAY.size > len(encoded):
      raise RouteEvidenceError("delay section is truncated")
    row = _DELAY.unpack_from(encoded, offset)
    offset += _DELAY.size
    if row[5] not in (0, 1) or row[6] != 0 or offset + row[7] > len(encoded):
      raise RouteEvidenceError("delay record is non-canonical")
    try:
      status = bytes(encoded[offset:offset + row[7]]).decode()
    except UnicodeDecodeError as error:
      raise RouteEvidenceError("delay status is invalid UTF-8") from error
    offset += row[7]
    output.append(LiveDelayPublication(row[0], row[1], row[2], row[3], row[4], bool(row[5]), status))
  if offset != len(encoded):
    raise RouteEvidenceError("delay section contains trailing bytes")
  return tuple(output)


def _encode_maneuvers(records: tuple[LateralManeuverPlanPublication, ...]) -> bytes:
  return b"".join(_MANEUVER.pack(
    item.segment_index, item.ordinal, item.mono_time_ns,
    _finite(item.desired_curvature, "maneuver desired curvature"),
    _boolean(item.message_valid, "maneuver message valid"), 0,
  ) for item in records)


def _decode_maneuvers(encoded: memoryview, count: int) -> tuple[LateralManeuverPlanPublication, ...]:
  if len(encoded) != count * _MANEUVER.size:
    raise RouteEvidenceError("maneuver section size/count disagree")
  result = []
  for offset in range(0, len(encoded), _MANEUVER.size):
    row = _MANEUVER.unpack_from(encoded, offset)
    if row[4] not in (0, 1) or row[5] != 0:
      raise RouteEvidenceError("maneuver record is non-canonical")
    result.append(LateralManeuverPlanPublication(*row[:4], bool(row[4])))
  return tuple(result)


def _encode_events(records: tuple[DrivingEventLocator, ...]) -> bytes:
  output = bytearray()
  for item in records:
    strings = (
      _text(item.event_id, "event id", MAX_EVENT_ID_BYTES).encode(),
      _text(item.event_type, "event type", MAX_EVENT_TYPE_BYTES).encode(),
      _text(item.severity, "event severity", MAX_EVENT_SEVERITY_BYTES).encode(),
    )
    output.extend(_EVENT.pack(
      item.segment_index, item.ordinal, item.publication_mono_time_ns,
      item.occurred_mono_time_ns,
      _finite(item.analysis_window_before_s, "event before window"),
      _finite(item.analysis_window_after_s, "event after window"),
      _boolean(item.message_valid, "event message valid"),
      *(len(value) for value in strings),
    ))
    for value in strings:
      output.extend(value)
  return bytes(output)


def _decode_events(encoded: memoryview, count: int) -> tuple[DrivingEventLocator, ...]:
  result = []
  offset = 0
  for _ in range(count):
    if offset + _EVENT.size > len(encoded):
      raise RouteEvidenceError("event section is truncated")
    row = _EVENT.unpack_from(encoded, offset)
    offset += _EVENT.size
    if row[6] not in (0, 1):
      raise RouteEvidenceError("event boolean is non-canonical")
    strings = []
    for length in row[7:10]:
      if offset + length > len(encoded):
        raise RouteEvidenceError("event text is truncated")
      try:
        strings.append(bytes(encoded[offset:offset + length]).decode())
      except UnicodeDecodeError as error:
        raise RouteEvidenceError("event text is invalid UTF-8") from error
      offset += length
    result.append(DrivingEventLocator(
      segment_index=row[0], ordinal=row[1], publication_mono_time_ns=row[2],
      occurred_mono_time_ns=row[3], analysis_window_before_s=row[4],
      analysis_window_after_s=row[5], event_id=strings[0],
      event_type=strings[1], severity=strings[2], message_valid=bool(row[6]),
    ))
  if offset != len(encoded):
    raise RouteEvidenceError("event section contains trailing bytes")
  return tuple(result)


class RouteEvidenceArtifact:
  """One canonical, hash-addressed, shared route preparation artifact."""

  __slots__ = (
    "_canonical_bytes", "_physical_offset", "_physical_size", "source_identity",
    "car_params_bytes", "model_publications", "control_witnesses",
    "live_torque_parameters", "live_delays", "lateral_maneuver_plans",
    "event_locators", "manifest", "sha256",
  )

  def __init__(
    self,
    source_identity: RouteEvidenceSourceIdentity,
    car_params_bytes: bytes | bytearray | memoryview,
    physical_bytes: bytes | bytearray | memoryview,
    model_publications: tuple[ModelPublication, ...],
    control_witnesses: tuple[ControlsWitness, ...],
    live_torque_parameters: tuple[LiveTorqueParametersPublication, ...] = (),
    live_delays: tuple[LiveDelayPublication, ...] = (),
    lateral_maneuver_plans: tuple[LateralManeuverPlanPublication, ...] = (),
    event_locators: tuple[DrivingEventLocator, ...] = (),
  ) -> None:
    if type(source_identity) is not RouteEvidenceSourceIdentity:
      raise RouteEvidenceError("source identity type is invalid")
    cp = bytes(car_params_bytes)
    physical = bytes(physical_bytes)
    if not cp or len(cp) > MAX_CAR_PARAMS_BYTES:
      raise RouteEvidenceError("canonical CarParams exceeds its bound")
    if type(model_publications) is not tuple or len(model_publications) > MAX_MODEL_PUBLICATIONS:
      raise RouteEvidenceError("model population exceeds its bound")
    if type(control_witnesses) is not tuple or len(control_witnesses) > MAX_CONTROL_WITNESSES:
      raise RouteEvidenceError("control population exceeds its bound")
    if len(control_witnesses) != source_identity.physical_record_count:
      raise RouteEvidenceError("control and physical-record populations disagree")
    # Imported lazily so the scratch-spool facade may itself wrap this module
    # without a top-level import cycle.  There remains one physical frame wire
    # implementation and behavior consumers never duplicate its layout.
    from openpilot.selfdrive.controls.lib.blatv2.learning_backfill_spool import (
      SPOOL_RECORD_SIZE,
    )
    if len(physical) != source_identity.physical_record_count * SPOOL_RECORD_SIZE:
      raise RouteEvidenceError("physical plane size/count/encoding disagree")
    _validate_sparse(live_torque_parameters, LiveTorqueParametersPublication, "live torque", MAX_SPARSE_PUBLICATIONS)
    _validate_sparse(live_delays, LiveDelayPublication, "live delay", MAX_SPARSE_PUBLICATIONS)
    _validate_sparse(lateral_maneuver_plans, LateralManeuverPlanPublication, "maneuver plan", MAX_SPARSE_PUBLICATIONS)
    if type(event_locators) is not tuple or len(event_locators) > MAX_EVENT_LOCATORS:
      raise RouteEvidenceError("event population exceeds its bound")
    previous_model: tuple[int, int, int] | None = None
    for value in model_publications:
      if type(value) is not ModelPublication:
        raise RouteEvidenceError("model record type is invalid")
      key = (value.mono_time_ns, value.segment_index, value.ordinal)
      if previous_model is not None and key <= previous_model:
        raise RouteEvidenceError("model records are not ordered")
      previous_model = key
    previous_control: tuple[int, int, int] | None = None
    for index, value in enumerate(control_witnesses):
      if type(value) is not ControlsWitness:
        raise RouteEvidenceError("control record type is invalid")
      if value.physical_record_index != index:
        raise RouteEvidenceError("physical record indices are not canonical")
      if value.model_publication_index >= len(model_publications):
        raise RouteEvidenceError("model publication index is out of range")
      if value.live_torque_parameters_index >= len(live_torque_parameters):
        raise RouteEvidenceError("live torque index is out of range")
      if value.live_delay_index >= len(live_delays):
        raise RouteEvidenceError("live delay index is out of range")
      if value.lateral_maneuver_plan_index >= len(lateral_maneuver_plans):
        raise RouteEvidenceError("maneuver plan index is out of range")
      key = (value.mono_time_ns, value.segment_index, value.ordinal)
      if previous_control is not None and key <= previous_control:
        raise RouteEvidenceError("control records are not ordered")
      previous_control = key
    event_ids: set[str] = set()
    previous_event: tuple[int, int, int] | None = None
    for value in event_locators:
      if type(value) is not DrivingEventLocator:
        raise RouteEvidenceError("event record type is invalid")
      _uint(value.segment_index, "event segment", (1 << 32) - 1)
      _uint(value.ordinal, "event ordinal", (1 << 32) - 1)
      _uint(value.publication_mono_time_ns, "event publication time")
      _uint(value.occurred_mono_time_ns, "event occurred time")
      if _finite(value.analysis_window_before_s, "event before window") < 0.0 or _finite(value.analysis_window_after_s, "event after window") < 0.0:
        raise RouteEvidenceError("event windows must be nonnegative")
      _boolean(value.message_valid, "event message valid")
      if value.event_id in event_ids:
        raise RouteEvidenceError("event IDs are not unique")
      event_ids.add(value.event_id)
      key = (value.publication_mono_time_ns, value.segment_index, value.ordinal)
      if previous_event is not None and key <= previous_event:
        raise RouteEvidenceError("event records are not ordered")
      previous_event = key
    sections = (
      cp, physical, _encode_models(model_publications),
      _encode_controls(control_witnesses), _encode_torque(live_torque_parameters),
      _encode_delay(live_delays), _encode_maneuvers(lateral_maneuver_plans),
      _encode_events(event_locators),
    )
    manifest = {
      "artifact_schema_version": ROUTE_EVIDENCE_VERSION,
      "car_params_sha256": hashlib.sha256(cp).hexdigest(),
      "car_params_size_bytes": len(cp),
      "control_witness_count": len(control_witnesses),
      "driving_event_locator_count": len(event_locators),
      "lateral_maneuver_plan_count": len(lateral_maneuver_plans),
      "live_delay_count": len(live_delays),
      "live_torque_parameters_count": len(live_torque_parameters),
      "model_publication_count": len(model_publications),
      "physical_plane_sha256": hashlib.sha256(physical).hexdigest(),
      "physical_plane_size_bytes": len(physical),
      "section_sha256": {
        name: hashlib.sha256(section).hexdigest()
        for name, section in zip(
          ("car_params", "physical", "models", "controls", "live_torque", "live_delay", "maneuvers", "events"),
          sections,
          strict=True,
        )
      },
      "source_identity": source_identity.manifest_dict(),
    }
    manifest_bytes = _canonical_json(manifest)
    header = _HEADER.pack(
      ROUTE_EVIDENCE_MAGIC, ROUTE_EVIDENCE_VERSION, 0,
      len(manifest_bytes), *(len(section) for section in sections),
    )
    canonical = b"".join((header, manifest_bytes, *sections))
    if len(canonical) > MAX_ARTIFACT_BYTES:
      raise RouteEvidenceError("route evidence exceeds bridge/artifact bound")
    self._canonical_bytes = canonical
    self._physical_offset = _HEADER.size + len(manifest_bytes) + len(cp)
    self._physical_size = len(physical)
    self.source_identity = source_identity
    self.car_params_bytes = memoryview(canonical)[_HEADER.size + len(manifest_bytes):self._physical_offset]
    self.model_publications = model_publications
    self.control_witnesses = control_witnesses
    self.live_torque_parameters = live_torque_parameters
    self.live_delays = live_delays
    self.lateral_maneuver_plans = lateral_maneuver_plans
    self.event_locators = event_locators
    self.manifest = manifest
    self.sha256 = hashlib.sha256(canonical).hexdigest()

  @property
  def canonical_bytes(self) -> bytes:
    return self._canonical_bytes

  @property
  def physical_bytes(self) -> memoryview:
    return memoryview(self._canonical_bytes)[self._physical_offset:self._physical_offset + self._physical_size]

  @property
  def source_key(self) -> str:
    return self.source_identity.preparation_cache_key

  def __eq__(self, other: object) -> bool:
    return (
      type(other) is RouteEvidenceArtifact
      and self.canonical_bytes == other.canonical_bytes
    )

  def __hash__(self) -> int:
    return hash(self.sha256)

  def iter_model_publications(self) -> Iterator[ModelPublication]:
    return iter(self.model_publications)

  def iter_physical_frames(self) -> Iterator[Any]:
    """Decode the single canonical physical plane with its owning codec."""
    from openpilot.selfdrive.controls.lib.blatv2.learning_backfill_spool import (
      SPOOL_RECORD_SIZE,
      _decode_frame,
    )
    encoded = self.physical_bytes
    for offset in range(0, len(encoded), SPOOL_RECORD_SIZE):
      yield _decode_frame(bytes(encoded[offset:offset + SPOOL_RECORD_SIZE]))

  def iter_control_witnesses(self) -> Iterator[ControlsWitness]:
    return iter(self.control_witnesses)

  def iter_event_locators(self) -> Iterator[DrivingEventLocator]:
    return iter(self.event_locators)

  @classmethod
  def from_bytes(cls, value: bytes | bytearray | memoryview) -> RouteEvidenceArtifact:
    encoded = bytes(value)
    if len(encoded) > MAX_ARTIFACT_BYTES or len(encoded) < _HEADER.size:
      raise RouteEvidenceError("route evidence size is invalid")
    header = _HEADER.unpack_from(encoded)
    if header[0] != ROUTE_EVIDENCE_MAGIC or header[1] != ROUTE_EVIDENCE_VERSION or header[2] != 0:
      raise RouteEvidenceError("route evidence version/magic is unsupported")
    sizes = header[3:]
    if _HEADER.size + sum(sizes) != len(encoded):
      raise RouteEvidenceError("route evidence sections are truncated or trailing")
    offsets = []
    cursor = _HEADER.size
    for size in sizes:
      offsets.append((cursor, cursor + size))
      cursor += size
    sections = tuple(memoryview(encoded)[left:right] for left, right in offsets)
    try:
      manifest: Any = json.loads(bytes(sections[0]))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
      raise RouteEvidenceError("manifest is invalid JSON") from error
    if type(manifest) is not dict or bytes(sections[0]) != _canonical_json(manifest):
      raise RouteEvidenceError("manifest is non-canonical")
    expected_manifest_keys = {
      "artifact_schema_version", "car_params_sha256", "car_params_size_bytes",
      "control_witness_count", "driving_event_locator_count", "lateral_maneuver_plan_count",
      "live_delay_count", "live_torque_parameters_count", "model_publication_count",
      "physical_plane_sha256", "physical_plane_size_bytes", "section_sha256",
      "source_identity",
    }
    if set(manifest) != expected_manifest_keys or manifest["artifact_schema_version"] != ROUTE_EVIDENCE_VERSION:
      raise RouteEvidenceError("manifest shape/version is invalid")
    source_payload = manifest["source_identity"]
    if type(source_payload) is not dict:
      raise RouteEvidenceError("source identity manifest is invalid")
    try:
      source_payload = dict(source_payload)
      source_payload["route_segment_sha256"] = tuple(source_payload["route_segment_sha256"])
      source_payload["route_segment_size_bytes"] = tuple(source_payload["route_segment_size_bytes"])
      source_payload["pre_poll_dropped_timestamps_ns"] = tuple(source_payload["pre_poll_dropped_timestamps_ns"])
      source = RouteEvidenceSourceIdentity(**source_payload)
    except (KeyError, TypeError, ValueError) as error:
      raise RouteEvidenceError("source identity manifest is invalid") from error
    cp = bytes(sections[1])
    physical = bytes(sections[2])
    section_hashes = manifest["section_sha256"]
    section_names = ("car_params", "physical", "models", "controls", "live_torque", "live_delay", "maneuvers", "events")
    if (
      type(section_hashes) is not dict
      or set(section_hashes) != set(section_names)
      or any(
        section_hashes[name] != hashlib.sha256(bytes(section)).hexdigest()
        for name, section in zip(section_names, sections[1:], strict=True)
      )
    ):
      raise RouteEvidenceError("route evidence section hash mismatch")
    if (
      manifest["car_params_size_bytes"] != len(cp)
      or manifest["car_params_sha256"] != hashlib.sha256(cp).hexdigest()
      or manifest["physical_plane_size_bytes"] != len(physical)
      or manifest["physical_plane_sha256"] != hashlib.sha256(physical).hexdigest()
    ):
      raise RouteEvidenceError("CarParams/physical section hash or size mismatch")
    models = _decode_models(sections[3], manifest["model_publication_count"])
    controls = _decode_controls(sections[4], manifest["control_witness_count"])
    torque = _decode_torque(sections[5], manifest["live_torque_parameters_count"])
    delays = _decode_delay(sections[6], manifest["live_delay_count"])
    maneuvers = _decode_maneuvers(sections[7], manifest["lateral_maneuver_plan_count"])
    events = _decode_events(sections[8], manifest["driving_event_locator_count"])
    rebuilt = cls(source, cp, physical, models, controls, torque, delays, maneuvers, events)
    if rebuilt.canonical_bytes != encoded:
      raise RouteEvidenceError("route evidence is not canonical")
    return rebuilt


def _regular(path: Path, purpose: str) -> os.stat_result:
  try:
    result = path.lstat()
  except FileNotFoundError as error:
    raise RouteEvidenceError(f"{purpose} is missing") from error
  if path.is_symlink() or not stat.S_ISREG(result.st_mode):
    raise RouteEvidenceError(f"{purpose} is not a regular file")
  return result


def _safe_directory(path: Path, create: bool) -> None:
  if create:
    path.mkdir(parents=True, exist_ok=True)
  try:
    result = path.lstat()
  except FileNotFoundError as error:
    raise RouteEvidenceError("evidence directory is missing") from error
  if path.is_symlink() or not stat.S_ISDIR(result.st_mode):
    raise RouteEvidenceError("unsafe evidence directory")


def _atomic_write(path: Path, data: bytes) -> None:
  descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".partial")
  temporary = Path(name)
  try:
    os.fchmod(descriptor, 0o600)
    view = memoryview(data)
    while view:
      count = os.write(descriptor, view)
      if count <= 0:
        raise OSError("short evidence write")
      view = view[count:]
    os.fsync(descriptor)
    os.close(descriptor)
    descriptor = -1
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
      os.fsync(directory)
    finally:
      os.close(directory)
  finally:
    if descriptor >= 0:
      os.close(descriptor)
    try:
      temporary.unlink()
    except FileNotFoundError:
      pass


class RouteEvidenceStore:
  """Immutable content store; publication is permitted only after exact A/A."""

  def __init__(self, root: str | Path) -> None:
    self.root = Path(root)

  def publish(self, first: RouteEvidenceArtifact, second: RouteEvidenceArtifact) -> RouteEvidenceArtifact:
    if type(first) is not RouteEvidenceArtifact or type(second) is not RouteEvidenceArtifact:
      raise RouteEvidenceError("A/A evidence types are invalid")
    if first.canonical_bytes != second.canonical_bytes:
      raise RouteEvidenceError("A/A mismatch; evidence was not published")
    _safe_directory(self.root, create=True)
    objects = self.root / "objects"
    sources = self.root / "sources"
    _safe_directory(objects, create=True)
    _safe_directory(sources, create=True)
    object_path = objects / f"{first.sha256}.route-evidence"
    index_path = sources / f"{first.source_key}.index"
    if object_path.exists() or object_path.is_symlink():
      _regular(object_path, "immutable evidence object")
      if object_path.read_bytes() != first.canonical_bytes:
        raise RouteEvidenceError("immutable evidence object bytes disagree")
    else:
      _atomic_write(object_path, first.canonical_bytes)
    index = f"{first.sha256}\n".encode("ascii")
    if index_path.exists() or index_path.is_symlink():
      _regular(index_path, "evidence source index")
      if index_path.read_bytes() != index:
        raise RouteEvidenceError("immutable evidence source index disagrees")
    else:
      _atomic_write(index_path, index)
    return first

  def load(self, sha256: str) -> RouteEvidenceArtifact:
    _hash(sha256, "route evidence sha256")
    _safe_directory(self.root, create=False)
    path = self.root / "objects" / f"{sha256}.route-evidence"
    info = _regular(path, "route evidence object")
    if info.st_size > MAX_ARTIFACT_BYTES:
      raise RouteEvidenceError("route evidence object exceeds its bound")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
      raise RouteEvidenceError("route evidence object hash mismatch")
    return RouteEvidenceArtifact.from_bytes(data)

  def lookup(self, source_key: str) -> RouteEvidenceArtifact | None:
    _hash(source_key, "route evidence source key")
    try:
      _safe_directory(self.root, create=False)
    except RouteEvidenceError:
      return None
    path = self.root / "sources" / f"{source_key}.index"
    if not path.exists() and not path.is_symlink():
      return None
    _regular(path, "route evidence source index")
    try:
      sha = path.read_text(encoding="ascii").removesuffix("\n")
    except UnicodeDecodeError as error:
      raise RouteEvidenceError("route evidence source index is invalid") from error
    _hash(sha, "route evidence source index hash")
    return self.load(sha)

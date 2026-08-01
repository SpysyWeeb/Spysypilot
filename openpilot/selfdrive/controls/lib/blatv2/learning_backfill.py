"""Offroad-only, full-rlog owner for durable BLaTv2 learning evidence.

The live learner is preview-only. This module is the sole durable evidence
writer once shipped: it accepts only complete local full-rlog routes, replays
the existing measured-frame/runtime path twice, records every decision in a
canonical SHA-bound ledger, and publishes an immutable generation by one
atomic CURRENT-pointer replacement.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import select
import shutil
import signal
import stat
import statistics
import struct
import subprocess
import tempfile
import threading
import time
from typing import Any

from openpilot.cereal.services import SERVICE_LIST
from openpilot.selfdrive.controls.lib.blatv2.calibration_coordinator import (
  CalibrationLearningFinalization,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill_progress import (
  BackfillProgressPhase,
  BackfillProgressPublisher,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill_spool import (
  PreparedRouteSpool,
  PreparedRouteSpoolDescriptor,
  SpoolFormatError,
  open_prepared_route_spool,
  write_prepared_route_spool,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_operation_status import (
  LearningOperationState,
  LearningOperationStatusPublisher,
  route_identity_sha256,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_runtime import (
  LearningArtifactPaths,
  PersistentLearningRuntime,
)
from openpilot.selfdrive.controls.lib.blatv2.preparation_frame import (
  MeasuredLearningFrame,
)
from openpilot.selfdrive.controls.lib.blatv2.route_evidence import (
  ControlsWitness,
  DrivingEventLocator,
  LateralManeuverPlanPublication,
  LiveDelayPublication,
  LiveTorqueParametersPublication,
  ModelPublication,
  RouteEvidenceArtifact,
  RouteEvidenceError,
  RouteEvidenceSourceIdentity,
  RouteEvidenceStore,
)
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  RuntimeVehicleBundle,
)


BACKFILL_LEDGER_SCHEMA_VERSION = 2
BACKFILL_PROVENANCE_SCHEMA_VERSION = 1
BACKFILL_COMMIT_SCHEMA_VERSION = 2
BACKFILL_POINTER_SCHEMA_VERSION = 1
NATIVE_EXTRACTOR_SCHEMA_VERSION = 3
CANONICAL_JOIN_SCHEMA_VERSION = 3
MAXIMUM_EVENT_BYTES = 64 * 1024 * 1024
MAXIMUM_EVENT_TRAVERSAL_WORDS = MAXIMUM_EVENT_BYTES // 8
MAXIMUM_SELECTED_RECORDS_PER_SEGMENT = 100_000
MAXIMUM_SELECTED_BYTES_PER_SEGMENT = 256 * 1024 * 1024
MAXIMUM_ROUTE_SEGMENTS = 128
MAXIMUM_ROUTE_FRAMES = 1_000_000
MAXIMUM_UNRESOLVED_FRACTION = 0.01
EXTRACTOR_IO_TIMEOUT_S = 60.0
EXTRACTOR_EXIT_TIMEOUT_S = 5.0
MAXIMUM_CONTROL_GAP_NS = 15_000_000
# Two workers are the independent A/A replay authorities. Four adds one
# private, one-route-ahead preparation lane to each authority; application
# remains serial within each pass and prepared data is never shared across it.
BACKFILL_REPLAY_WORKER_COUNT = 4
SUPPORTED_BACKFILL_REPLAY_WORKER_COUNTS = (1, 2, 4)
BACKFILL_SPOOL_DIRECTORY_PREFIX = ".blatv2-backfill-prepare-"
_BACKFILL_SPOOL_DIRECTORY_RE = re.compile(
  rf"{re.escape(BACKFILL_SPOOL_DIRECTORY_PREFIX)}[12]-[a-z0-9_]{{8}}\Z",
)
REPLAY_WORKER_STARTUP_TIMEOUT_S = 1.0
REPLAY_WORKER_COOPERATIVE_STOP_S = 0.75
REPLAY_WORKER_SIGNAL_STOP_S = 0.5
NOMINAL_CONTROL_PERIOD_NS = int(
  round(1e9 / SERVICE_LIST["controlsState"].frequency),
)
_ROUTE_DIRECTORY_RE = re.compile(
  r"(?P<route>[0-9a-f]{8}--[0-9a-f]{10})--(?P<segment>[0-9]+)",
)
_FULL_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_STREAM_MAGIC = b"BLATV2R1"
_STREAM_HEADER = struct.Struct("<8sII")
_RECORD_HEADER = struct.Struct("<IIQ")
_END_RECORD = 0xffffffff
_EVENT_WHICH = {
  "initData": 0,
  "controlsState": 6,
  "carState": 21,
  "carControl": 22,
  "liveParameters": 60,
  "carParams": 67,
  "sentinel": 71,
  "carOutput": 125,
  # Event::Which values are union discriminants, not Cap'n Proto field
  # ordinals; the two deprecated union slots preceding these fields are not
  # discriminants in the generated enum.
  "modelV2": 73,
  "liveTorqueParameters": 92,
  "selfdriveState": 128,
  "liveDelay": 144,
  "lateralManeuverPlan": 148,
  "drivingEvent": 151,
}
_SOURCE_SERVICES = (
  "carControl",
  "carState",
  "carOutput",
  "liveParameters",
)
_BEHAVIOR_CONTEXT_SERVICES = (
  "modelV2",
  "liveTorqueParameters",
  "liveDelay",
  "lateralManeuverPlan",
  "drivingEvent",
)
_CANONICAL_WITNESS_SERVICES = (
  "controlsState",
  "selfdriveState",
)
_LEDGER_PROVENANCE_KEYS = {
  "canonical_join_schema_version",
  "car_params_sha256",
  "dongle_id_sha256",
  "extractor_schema_version",
  "log_schema_blob",
  "opendbc_commit",
  "panda_commit",
  "physical_compatibility_sha256",
  "route_version",
  "selected_event_stream_sha256",
  "superproject_commit",
}


class BackfillError(RuntimeError):
  """Whole-operation failure with a stable UI diagnostic."""

  def __init__(self, diagnostic: str, message: str) -> None:
    super().__init__(message)
    self.diagnostic = diagnostic


class RouteRejected(RuntimeError):
  """One route is ineligible; later routes may still be replayed."""

  def __init__(self, reason: str, message: str) -> None:
    super().__init__(message)
    self.reason = reason


@dataclass(frozen=True, slots=True)
class BuildDescriptor:
  superproject_commit: str
  opendbc_commit: str
  panda_commit: str
  log_schema_blob: str
  supported_vehicle_identity: str
  steer_max: int
  steer_delta_up: int
  steer_delta_down: int
  steer_step: int
  driver_allowance: int
  driver_multiplier: int
  driver_factor: int
  production_envelope_verified: bool
  rack_rate_resolution_deg_s: float

  def __post_init__(self) -> None:
    commits = (
      self.superproject_commit,
      self.opendbc_commit,
      self.panda_commit,
      self.log_schema_blob,
    )
    integer_limits = (
      self.steer_max,
      self.steer_delta_up,
      self.steer_delta_down,
      self.steer_step,
      self.driver_allowance,
      self.driver_multiplier,
      self.driver_factor,
    )
    if (
      any(
        type(value) is not str
        or _FULL_COMMIT_RE.fullmatch(value) is None
        for value in commits
      )
      or type(self.supported_vehicle_identity) is not str
      or not self.supported_vehicle_identity
      or any(type(value) is not int for value in integer_limits)
      or any(value <= 0 for value in integer_limits[:4])
      or any(value < 0 for value in integer_limits[4:])
      or self.production_envelope_verified is not True
      or type(self.rack_rate_resolution_deg_s) not in (int, float)
      or not math.isfinite(float(self.rack_rate_resolution_deg_s))
      or float(self.rack_rate_resolution_deg_s) <= 0.0
    ):
      raise ValueError("build descriptor provenance/envelope is invalid")
    object.__setattr__(
      self,
      "rack_rate_resolution_deg_s",
      float(self.rack_rate_resolution_deg_s),
    )

  @classmethod
  def from_dict(cls, payload: object) -> BuildDescriptor:
    expected = {
      "log_schema_blob",
      "opendbc_commit",
      "panda_commit",
      "driver_allowance",
      "driver_factor",
      "driver_multiplier",
      "production_envelope_verified",
      "rack_rate_resolution_deg_s",
      "steer_delta_down",
      "steer_delta_up",
      "steer_max",
      "steer_step",
      "superproject_commit",
      "supported_vehicle_identity",
    }
    if type(payload) is not dict or set(payload) != expected:
      raise ValueError("build descriptor keys do not match")
    commits = (
      payload["superproject_commit"],
      payload["opendbc_commit"],
      payload["panda_commit"],
      payload["log_schema_blob"],
    )
    if any(
      type(value) is not str or _FULL_COMMIT_RE.fullmatch(value) is None
      for value in commits
    ):
      raise ValueError("build descriptor requires full commit identities")
    vehicle = payload["supported_vehicle_identity"]
    limits = (
      payload["steer_max"],
      payload["steer_delta_up"],
      payload["steer_delta_down"],
      payload["steer_step"],
      payload["driver_allowance"],
      payload["driver_multiplier"],
      payload["driver_factor"],
    )
    rack_rate_resolution = payload["rack_rate_resolution_deg_s"]
    if (
      type(vehicle) is not str
      or not vehicle
      or any(type(value) is not int for value in limits)
      or any(value <= 0 for value in limits[:4])
      or any(value < 0 for value in limits[4:])
      or type(payload["production_envelope_verified"]) is not bool
      or type(rack_rate_resolution) not in (int, float)
      or not math.isfinite(float(rack_rate_resolution))
      or float(rack_rate_resolution) <= 0.0
    ):
      raise ValueError("build descriptor vehicle/limits are invalid")
    return cls(
      superproject_commit=payload["superproject_commit"],
      opendbc_commit=payload["opendbc_commit"],
      panda_commit=payload["panda_commit"],
      log_schema_blob=payload["log_schema_blob"],
      supported_vehicle_identity=vehicle,
      steer_max=payload["steer_max"],
      steer_delta_up=payload["steer_delta_up"],
      steer_delta_down=payload["steer_delta_down"],
      steer_step=payload["steer_step"],
      driver_allowance=payload["driver_allowance"],
      driver_multiplier=payload["driver_multiplier"],
      driver_factor=payload["driver_factor"],
      production_envelope_verified=(
        payload["production_envelope_verified"]
      ),
      rack_rate_resolution_deg_s=float(rack_rate_resolution),
    )

  def to_dict(self) -> dict[str, object]:
    return {
      "log_schema_blob": self.log_schema_blob,
      "opendbc_commit": self.opendbc_commit,
      "panda_commit": self.panda_commit,
      "driver_allowance": self.driver_allowance,
      "driver_factor": self.driver_factor,
      "driver_multiplier": self.driver_multiplier,
      "production_envelope_verified": (
        self.production_envelope_verified
      ),
      "rack_rate_resolution_deg_s": (
        self.rack_rate_resolution_deg_s
      ),
      "steer_delta_down": self.steer_delta_down,
      "steer_delta_up": self.steer_delta_up,
      "steer_max": self.steer_max,
      "steer_step": self.steer_step,
      "superproject_commit": self.superproject_commit,
      "supported_vehicle_identity": self.supported_vehicle_identity,
    }

  def controller_params_proxy(self) -> object:
    """Reviewed recorded envelope, independent of current route CP flags."""
    from types import SimpleNamespace

    return SimpleNamespace(
      STEER_MAX=self.steer_max,
      STEER_DELTA_UP=self.steer_delta_up,
      STEER_DELTA_DOWN=self.steer_delta_down,
      STEER_STEP=self.steer_step,
      STEER_DRIVER_ALLOWANCE=self.driver_allowance,
      STEER_DRIVER_MULTIPLIER=self.driver_multiplier,
      STEER_DRIVER_FACTOR=self.driver_factor,
      BLATV2_RUNTIME_ENVELOPE_COMPATIBLE=(
        self.production_envelope_verified
      ),
      BLATV2_RACK_RATE_RESOLUTION_DEG_S=(
        self.rack_rate_resolution_deg_s
      ),
    )


class BuildDescriptorRegistry:
  def __init__(self, descriptors: tuple[BuildDescriptor, ...]) -> None:
    by_commit: dict[str, BuildDescriptor] = {}
    for descriptor in descriptors:
      if descriptor.superproject_commit in by_commit:
        raise ValueError("duplicate superproject build descriptor")
      by_commit[descriptor.superproject_commit] = descriptor
    self._by_commit = by_commit

  @classmethod
  def from_json_file(cls, path: str | Path) -> BuildDescriptorRegistry:
    encoded = Path(path).read_bytes()
    payload = json.loads(encoded)
    if (
      type(payload) is not dict
      or set(payload) != {"descriptors", "schema_version"}
      or payload["schema_version"] != 1
      or type(payload["descriptors"]) is not list
      or encoded != _canonical_json_bytes(payload)
    ):
      raise ValueError("historical build registry is not canonical")
    descriptors = tuple(
      BuildDescriptor.from_dict(item)
      for item in payload["descriptors"]
    )
    if tuple(
      descriptor.superproject_commit
      for descriptor in descriptors
    ) != tuple(sorted(
      descriptor.superproject_commit
      for descriptor in descriptors
    )):
      raise ValueError("historical build descriptors are not sorted")
    return cls(descriptors)

  def resolve(self, superproject_commit: str) -> BuildDescriptor | None:
    return self._by_commit.get(superproject_commit)

  def with_descriptor(
    self,
    descriptor: BuildDescriptor,
  ) -> BuildDescriptorRegistry:
    existing = self._by_commit.get(descriptor.superproject_commit)
    if existing is not None:
      if existing != descriptor:
        raise ValueError("current build conflicts with reviewed descriptor")
      return self
    return BuildDescriptorRegistry(
      tuple(self._by_commit.values()) + (descriptor,),
    )

  @property
  def identity_sha256(self) -> str:
    return _sha256(_canonical_json_bytes({
      "descriptors": [
        descriptor.to_dict()
        for descriptor in sorted(
          self._by_commit.values(),
          key=lambda item: item.superproject_commit,
        )
      ],
      "schema_version": 1,
    }))


def build_current_historical_descriptor(
  *,
  source_commit: str,
  opendbc_commit: str,
  panda_commit: str,
  log_schema_path: str | Path,
  current_car_params: Any,
  current_runtime_bundle: RuntimeVehicleBundle,
) -> BuildDescriptor:
  """Build the effective current-build descriptor from runtime facts.

  The device and an authenticated off-device preparation worker must perform
  exactly the same construction before extending the reviewed, file-backed
  historical registry.  Keep this helper free of Params and device state so
  both architectures import these exact bytes and arithmetic.
  """
  rack_resolutions = {
    float(node.parameters.rack_rate_resolution_deg_s)
    for node in current_runtime_bundle.calibration_seed_profile.nodes
  }
  if len(rack_resolutions) != 1:
    raise ValueError("current runtime has inconsistent rack-rate resolution")
  limits = current_runtime_bundle.torque_limits
  return BuildDescriptor(
    superproject_commit=source_commit,
    opendbc_commit=opendbc_commit,
    panda_commit=panda_commit,
    log_schema_blob=git_blob_sha1(log_schema_path),
    supported_vehicle_identity=str(current_car_params.carFingerprint),
    steer_max=limits.steer_max,
    steer_delta_up=limits.delta_up,
    steer_delta_down=limits.delta_down,
    steer_step=limits.steer_step,
    driver_allowance=limits.driver_allowance,
    driver_multiplier=limits.driver_multiplier,
    driver_factor=limits.driver_factor,
    production_envelope_verified=limits.production_envelope_verified,
    rack_rate_resolution_deg_s=rack_resolutions.pop(),
  )


@dataclass(frozen=True, slots=True)
class ExtractedEvent:
  which: int
  mono_ns: int
  ordinal: int
  encoded: bytes


@dataclass(frozen=True, slots=True)
class VerifiedExtractor:
  """One executable inode whose hash and child execution are inseparable."""

  path: Path
  descriptor: int
  opened_stat: os.stat_result
  sha256: str


@dataclass(frozen=True, slots=True)
class _DecodedExtractedEvent:
  which: str
  mono_ns: int
  valid: bool
  payload: Any


@dataclass(frozen=True, slots=True)
class _RecordedCarState:
  v_ego: float
  steering_angle_deg: float
  steering_rate_deg_s: float
  steering_torque: float
  steering_pressed: bool
  standstill: bool
  steer_fault_temporary: bool
  steer_fault_permanent: bool
  can_valid: bool
  can_timeout: bool


@dataclass(frozen=True, slots=True)
class _RecordedCarControl:
  lateral_active: bool
  request_torque: float


@dataclass(frozen=True, slots=True)
class _RecordedCarOutput:
  applied_torque: float
  torque_output_can_count: int
  torque_output_can_valid: bool


@dataclass(frozen=True, slots=True)
class _RecordedLiveParameters:
  valid: bool
  angle_offset_valid: bool
  steer_ratio_valid: bool
  stiffness_factor_valid: bool
  angle_offset_deg: float
  steer_ratio: float
  stiffness_factor: float
  roll_rad: float


@dataclass(frozen=True, slots=True)
class _RecordedControlsState:
  lateral_plan_mono_ns: int
  measured_curvature: float
  desired_curvature: float
  modular_architecture: str
  modular_selection: int
  modular_artifact_sha256: str
  modular_source_openpilot_commit: str
  modular_opendbc_commit: str
  modular_selection_bound: bool


@dataclass(frozen=True, slots=True)
class _RecordedModel:
  frame_id: int
  timestamp_eof_ns: int
  scalar_curvature: float
  desired_curvature_time_s: float
  plan_times_s: tuple[float, ...]
  orientation_rate_z: tuple[float, ...]
  velocity_x: tuple[float, ...]
  native_grid_valid: bool


@dataclass(frozen=True, slots=True)
class _RecordedLiveTorqueParameters:
  live_valid: bool
  use_params: bool
  version: int
  lat_accel_factor: float
  lat_accel_offset: float
  friction: float


@dataclass(frozen=True, slots=True)
class _RecordedLiveDelay:
  lateral_delay_s: float
  status: str
  version: int


@dataclass(frozen=True, slots=True)
class _RecordedLateralManeuverPlan:
  desired_curvature: float


@dataclass(frozen=True, slots=True)
class _RecordedDrivingEvent:
  event_id: str
  occurred_mono_ns: int
  analysis_window_before_s: float
  analysis_window_after_s: float
  event_type: str
  severity: str


@dataclass(frozen=True, slots=True)
class RouteSegment:
  index: int
  path: Path
  sha256: str
  size_bytes: int

  def to_ledger_dict(self) -> dict[str, object]:
    return {
      "index": self.index,
      "sha256": self.sha256,
      "size_bytes": self.size_bytes,
    }


@dataclass(frozen=True, slots=True)
class RouteCandidate:
  route_name: str
  route_counter: int
  segments: tuple[RouteSegment, ...]

  @property
  def display_identity(self) -> str:
    return route_identity_sha256(self.route_name)


@dataclass(frozen=True, slots=True)
class _TimedRouteRecord:
  mono_ns: int
  segment_index: int
  ordinal: int
  source_order: int
  valid: bool
  payload: Any


@dataclass(frozen=True, slots=True)
class _CanonicalControlJoin:
  witness: _TimedRouteRecord
  poll_mono_ns: int
  car_state: _TimedRouteRecord
  live_parameters: _TimedRouteRecord
  car_output: _TimedRouteRecord
  previous_car_output_mono_ns: int | None
  car_control: _TimedRouteRecord | None
  curvature_unresolved: bool
  gap_from_previous: bool


@dataclass(frozen=True, slots=True)
class PreparedRoute:
  frames: tuple[Any, ...]
  controls_witness_count: int
  unresolved_witness_count: int
  gap_count: int
  provenance: dict[str, object]
  route_evidence: Any | None = None
  pre_poll_dropped_count: int = 0
  behavior_eligible: bool = False
  behavior_ineligible_reason: str = "shared_evidence_unavailable"


@dataclass(frozen=True, slots=True)
class ReplayResult:
  route: RouteCandidate
  disposition: str
  diagnostic: str
  provenance: dict[str, object] | None
  accepted_sample_count: int
  rejected_sample_count: int
  controls_witness_count: int
  unresolved_witness_count: int
  route_evidence_sha256: str | None = None
  route_evidence_model_publication_count: int = 0
  route_evidence_control_witness_count: int = 0
  route_evidence_event_locator_count: int = 0
  route_evidence_source_key: str | None = None

  def ledger_entry(self) -> dict[str, object]:
    return {
      "accepted_sample_count": self.accepted_sample_count,
      "controls_witness_count": self.controls_witness_count,
      "diagnostic": self.diagnostic,
      "disposition": self.disposition,
      "provenance": self.provenance,
      "rejected_sample_count": self.rejected_sample_count,
      "route_counter": self.route.route_counter,
      "route_identity_sha256": self.route.display_identity,
      "route_evidence_control_witness_count": (
        self.route_evidence_control_witness_count
      ),
      "route_evidence_event_locator_count": (
        self.route_evidence_event_locator_count
      ),
      "route_evidence_model_publication_count": (
        self.route_evidence_model_publication_count
      ),
      "route_evidence_sha256": self.route_evidence_sha256,
      "route_name": self.route.route_name,
      "segments": [
        segment.to_ledger_dict()
        for segment in self.route.segments
      ],
      "unresolved_witness_count": self.unresolved_witness_count,
    }


@dataclass(frozen=True, slots=True)
class ReplayPass:
  finalization: CalibrationLearningFinalization
  results: tuple[ReplayResult, ...]
  accepted_sample_count: int
  rejected_sample_count: int


@dataclass(frozen=True, slots=True)
class BackfillPublication:
  generation_sha256: str
  ledger_sha256: str
  finalization: CalibrationLearningFinalization
  accepted_sample_count: int
  rejected_sample_count: int
  diagnostic: str


@dataclass(frozen=True, slots=True)
class BackfillRunResult:
  publication: BackfillPublication | None
  pending_logger_close: bool


@dataclass(frozen=True, slots=True)
class BehaviorEvidenceCohortSelection:
  """Newest complete, contiguous, exact-source behavior population."""

  status: str
  reason: str
  blocking_route_name: str | None
  source_identity_sha256: str | None
  artifacts: tuple[RouteEvidenceArtifact, ...]

  @property
  def ready(self) -> bool:
    return self.status == "ready"


@dataclass(frozen=True, slots=True)
class FullRlogDiscovery:
  candidates: tuple[RouteCandidate, ...]
  pending_logger_close: bool


def _canonical_json_bytes(payload: object) -> bytes:
  return json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
  ).encode("utf-8")


def _sha256(encoded: bytes) -> str:
  return hashlib.sha256(encoded).hexdigest()


def _abort_if_requested(
  abort_requested: Callable[[], bool],
  message: str,
) -> None:
  if abort_requested():
    raise BackfillError("unexpected_error", message)


def _sha256_file(
  path: Path,
  *,
  abort_requested: Callable[[], bool] = lambda: False,
) -> str:
  digest = hashlib.sha256()
  _abort_if_requested(
    abort_requested,
    "backfill aborted before hashing a file",
  )
  with path.open("rb") as stream:
    while True:
      _abort_if_requested(
        abort_requested,
        "backfill aborted while hashing a file",
      )
      chunk = stream.read(1024 * 1024)
      _abort_if_requested(
        abort_requested,
        "backfill aborted while hashing a file",
      )
      if not chunk:
        break
      digest.update(chunk)
  return digest.hexdigest()


def _sha256_open_file(
  descriptor: int,
  *,
  abort_requested: Callable[[], bool],
) -> str:
  """Hash one already-open regular file without changing its file offset."""
  digest = hashlib.sha256()
  offset = 0
  while True:
    _abort_if_requested(
      abort_requested,
      "backfill aborted while hashing an open route segment",
    )
    chunk = os.pread(descriptor, 1024 * 1024, offset)
    _abort_if_requested(
      abort_requested,
      "backfill aborted while hashing an open route segment",
    )
    if not chunk:
      break
    digest.update(chunk)
    offset += len(chunk)
  return digest.hexdigest()


def _extractor_stat_is_executable(
  file_stat: os.stat_result,
  descriptor: int,
) -> bool:
  return (
    stat.S_ISREG(file_stat.st_mode)
    and stat.S_IMODE(file_stat.st_mode) & 0o111 != 0
    and os.access(f"/proc/self/fd/{descriptor}", os.X_OK)
  )


def open_verified_extractor(
  extractor_path: str | Path,
  *,
  expected_sha256: str | None = None,
  abort_requested: Callable[[], bool] = lambda: False,
) -> VerifiedExtractor:
  """Open and hash the exact executable inode later inherited by children."""
  if (
    expected_sha256 is not None
    and (
      type(expected_sha256) is not str
      or _SHA256_RE.fullmatch(expected_sha256) is None
    )
  ):
    raise ValueError("expected extractor SHA-256 is invalid")
  path = Path(extractor_path)
  descriptor = -1
  try:
    _abort_if_requested(
      abort_requested,
      "backfill aborted before opening the native extractor",
    )
    path_stat = path.lstat()
    if not stat.S_ISREG(path_stat.st_mode) or path.is_symlink():
      raise OSError("native extractor is not a regular file")
    descriptor = os.open(
      path,
      os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    opened_stat = os.fstat(descriptor)
    observed_sha256 = _sha256_open_file(
      descriptor,
      abort_requested=abort_requested,
    )
    if (
      not _same_open_segment_identity(path_stat, opened_stat)
      or path_stat.st_mode != opened_stat.st_mode
      or not _extractor_stat_is_executable(opened_stat, descriptor)
      or (
        expected_sha256 is not None
        and observed_sha256 != expected_sha256
      )
    ):
      raise OSError("native extractor identity does not match its contract")
    return VerifiedExtractor(
      path=path,
      descriptor=descriptor,
      opened_stat=opened_stat,
      sha256=observed_sha256,
    )
  except BackfillError:
    if descriptor >= 0:
      os.close(descriptor)
    raise
  except OSError as exc:
    if descriptor >= 0:
      os.close(descriptor)
    raise BackfillError(
      "backfill_reader_unavailable",
      "native rlog extractor is unavailable or changed",
    ) from exc


def verify_open_extractor(
  extractor: VerifiedExtractor,
  *,
  abort_requested: Callable[[], bool] = lambda: False,
) -> None:
  """Verify the held executable and its pathname after route preparation."""
  try:
    after = os.fstat(extractor.descriptor)
    path_after = extractor.path.lstat()
    if (
      not _same_open_segment_identity(extractor.opened_stat, after)
      or not _same_open_segment_identity(extractor.opened_stat, path_after)
      or extractor.opened_stat.st_mode != after.st_mode
      or extractor.opened_stat.st_mode != path_after.st_mode
      or not _extractor_stat_is_executable(after, extractor.descriptor)
      or _sha256_open_file(
        extractor.descriptor,
        abort_requested=abort_requested,
      ) != extractor.sha256
    ):
      raise OSError("native extractor changed during route preparation")
  except BackfillError:
    raise
  except OSError as exc:
    raise BackfillError(
      "backfill_reader_unavailable",
      "native rlog extractor changed during route preparation",
    ) from exc


def _same_open_segment_identity(
  left: os.stat_result,
  right: os.stat_result,
) -> bool:
  return (
    stat.S_ISREG(right.st_mode)
    and left.st_dev == right.st_dev
    and left.st_ino == right.st_ino
    and left.st_size == right.st_size
    and left.st_mtime_ns == right.st_mtime_ns
    and left.st_ctime_ns == right.st_ctime_ns
  )


def _open_verified_route_segment(
  segment: RouteSegment,
  *,
  abort_requested: Callable[[], bool],
) -> tuple[int, os.stat_result]:
  """Open, identify, and hash the exact inode the extractor will consume."""
  _abort_if_requested(
    abort_requested,
    "backfill aborted before opening a route segment",
  )
  descriptor = -1
  try:
    path_stat = segment.path.lstat()
    if not stat.S_ISREG(path_stat.st_mode) or segment.path.is_symlink():
      raise OSError("route segment is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(
      os,
      "O_CLOEXEC",
      0,
    )
    descriptor = os.open(segment.path, flags)
    opened_stat = os.fstat(descriptor)
    if (
      not _same_open_segment_identity(path_stat, opened_stat)
      or opened_stat.st_size != segment.size_bytes
      or _sha256_open_file(
        descriptor,
        abort_requested=abort_requested,
      ) != segment.sha256
    ):
      raise OSError("route segment identity does not match discovery")
    return descriptor, opened_stat
  except BackfillError:
    if descriptor >= 0:
      os.close(descriptor)
    raise
  except OSError as exc:
    if descriptor >= 0:
      os.close(descriptor)
    raise RouteRejected(
      "segment_changed",
      "route segment changed after discovery",
    ) from exc


def _verify_open_route_segment(
  segment: RouteSegment,
  descriptor: int,
  opened_stat: os.stat_result,
  *,
  abort_requested: Callable[[], bool],
) -> None:
  """Reject mutation/replacement of the held inode during extraction."""
  try:
    after = os.fstat(descriptor)
    path_after = segment.path.lstat()
    if (
      not _same_open_segment_identity(opened_stat, after)
      or not _same_open_segment_identity(opened_stat, path_after)
      or _sha256_open_file(
        descriptor,
        abort_requested=abort_requested,
      ) != segment.sha256
    ):
      raise OSError("route segment changed during extraction")
  except BackfillError:
    raise
  except OSError as exc:
    raise RouteRejected(
      "segment_changed",
      "route segment changed during extraction",
    ) from exc


def git_blob_sha1(path: str | Path) -> str:
  encoded = Path(path).read_bytes()
  digest = hashlib.sha1()
  digest.update(f"blob {len(encoded)}\0".encode("ascii"))
  digest.update(encoded)
  return digest.hexdigest()


def bounded_event_reader(encoded: bytes) -> AbstractContextManager[Any]:
  """Decode one helper-framed Event with finite traversal/nesting limits."""
  from openpilot.cereal import log

  return log.Event.from_bytes(
    encoded,
    traversal_limit_in_words=MAXIMUM_EVENT_TRAVERSAL_WORDS,
    nesting_limit=64,
  )


def discover_complete_route_candidates(
  log_root: str | Path,
  *,
  abort_requested: Callable[[], bool] = lambda: False,
) -> tuple[RouteCandidate, ...]:
  """Discover preliminarily complete full-rlog routes, oldest first."""
  return discover_full_rlog_state(
    log_root,
    abort_requested=abort_requested,
  ).candidates


def discover_full_rlog_state(
  log_root: str | Path,
  *,
  abort_requested: Callable[[], bool] = lambda: False,
) -> FullRlogDiscovery:
  """Snapshot complete candidates and logger ownership in one scan."""
  root = Path(log_root)
  grouped: dict[str, dict[int, Path]] = {}
  locked_directories: set[Path] = set()
  _abort_if_requested(
    abort_requested,
    "backfill aborted before route discovery",
  )
  if not root.is_dir():
    return FullRlogDiscovery((), False)
  for child in root.iterdir():
    _abort_if_requested(
      abort_requested,
      "backfill aborted during route discovery",
    )
    if not child.is_dir():
      continue
    match = _ROUTE_DIRECTORY_RE.fullmatch(child.name)
    if match is None:
      continue
    try:
      if any(child.glob("*.lock")):
        locked_directories.add(child)
    except OSError:
      # A directory pruned during the snapshot is no longer pending.
      continue
    grouped.setdefault(match["route"], {})[int(match["segment"])] = child

  candidates = []
  for route_name, directories in grouped.items():
    _abort_if_requested(
      abort_requested,
      "backfill aborted during route discovery",
    )
    indexes = sorted(directories)
    if (
      not indexes
      or indexes[0] != 0
      or indexes != list(range(indexes[-1] + 1))
      or len(indexes) > MAXIMUM_ROUTE_SEGMENTS
      or any(
        directories[index] in locked_directories
        for index in indexes
      )
    ):
      continue
    segments = []
    complete = True
    for index in indexes:
      _abort_if_requested(
        abort_requested,
        "backfill aborted during route discovery",
      )
      directory = directories[index]
      compressed = directory / "rlog.zst"
      raw = directory / "rlog"
      path = compressed if compressed.is_file() else raw
      if not path.is_file():
        complete = False
        break
      try:
        segment_sha256 = _sha256_file(
          path,
          abort_requested=abort_requested,
        )
        segment_size = path.stat().st_size
      except OSError:
        # Route pruning is allowed to race an offroad scan. A disappearing
        # candidate is simply no longer locally available; it must not fail
        # unrelated complete routes or the whole backfill operation.
        complete = False
        break
      segments.append(RouteSegment(
        index=index,
        path=path,
        sha256=segment_sha256,
        size_bytes=segment_size,
      ))
    if complete:
      candidates.append(RouteCandidate(
        route_name=route_name,
        route_counter=int(route_name[:8], 16),
        segments=tuple(segments),
      ))
  return FullRlogDiscovery(
    candidates=tuple(sorted(
      candidates,
      key=lambda route: (route.route_counter, route.route_name),
    )),
    pending_logger_close=bool(locked_directories),
  )


def has_pending_full_rlog(
  log_root: str | Path,
  *,
  abort_requested: Callable[[], bool] = lambda: False,
) -> bool:
  root = Path(log_root)
  _abort_if_requested(
    abort_requested,
    "backfill aborted before checking logger finalization",
  )
  if not root.is_dir():
    return False
  for child in root.iterdir():
    _abort_if_requested(
      abort_requested,
      "backfill aborted while checking logger finalization",
    )
    if (
      child.is_dir()
      and _ROUTE_DIRECTORY_RE.fullmatch(child.name) is not None
      and any(child.glob("*.lock"))
    ):
      return True
  return False


def _read_process_exact(
  process: subprocess.Popen,
  stream: Any,
  size: int,
  *,
  abort_requested: Callable[[], bool],
) -> bytes:
  chunks = bytearray()
  deadline = time.monotonic() + EXTRACTOR_IO_TIMEOUT_S
  while len(chunks) < size:
    if abort_requested():
      raise BackfillError(
        "unexpected_error",
        "backfill aborted while extracting a route",
      )
    ready, _, _ = select.select((stream,), (), (), 0.1)
    if not ready:
      if process.poll() is not None or time.monotonic() >= deadline:
        break
      continue
    chunk = os.read(stream.fileno(), size - len(chunks))
    if not chunk:
      break
    chunks.extend(chunk)
    deadline = time.monotonic() + EXTRACTOR_IO_TIMEOUT_S
  return bytes(chunks)


def _wait_process(
  process: subprocess.Popen,
  *,
  abort_requested: Callable[[], bool],
) -> int:
  deadline = time.monotonic() + EXTRACTOR_EXIT_TIMEOUT_S
  while process.poll() is None:
    if abort_requested():
      raise BackfillError(
        "unexpected_error",
        "backfill aborted while waiting for extractor",
      )
    if time.monotonic() >= deadline:
      raise RouteRejected(
        "extractor_timeout",
        "native extractor did not exit after its trailer",
      )
    time.sleep(0.05)
  return int(process.returncode)


def extract_segment_events(
  extractor_path: str | Path,
  segment_path: str | Path,
  *,
  abort_requested: Callable[[], bool] = lambda: False,
  extractor_fd: int | None = None,
  segment_fd: int | None = None,
) -> tuple[ExtractedEvent, ...]:
  """Buffer one selected segment only after a verified native trailer/exit.

  ``prepare_route`` supplies a held, pre-hashed descriptor. ``/proc/self/fd``
  makes the native child open that exact inode instead of resolving a mutable
  pathname between the parent's before/after checks.
  """
  if extractor_fd is not None:
    if type(extractor_fd) is not int or extractor_fd < 0:
      raise ValueError("extractor descriptor is invalid")
    extractor_stat = os.fstat(extractor_fd)
    if not _extractor_stat_is_executable(extractor_stat, extractor_fd):
      raise ValueError("extractor descriptor is not executable")
    extractor_argument_path = f"/proc/self/fd/{extractor_fd}"
    inherited_fds = [extractor_fd]
  else:
    # Direct pathname mode remains for isolated protocol tests. Production
    # route preparation always supplies the verified held descriptor above.
    extractor = Path(extractor_path)
    if not extractor.is_file() or not os.access(extractor, os.X_OK):
      raise BackfillError(
        "backfill_reader_unavailable",
        "native rlog extractor is unavailable",
      )
    extractor_argument_path = str(extractor)
    inherited_fds = []
  if segment_fd is not None:
    if type(segment_fd) is not int or segment_fd < 0:
      raise ValueError("segment descriptor is invalid")
    if not stat.S_ISREG(os.fstat(segment_fd).st_mode):
      raise ValueError("segment descriptor is not a regular file")
    extractor_argument = f"/proc/self/fd/{segment_fd}"
    inherited_fds.append(segment_fd)
  else:
    extractor_argument = str(segment_path)
  with tempfile.TemporaryFile() as errors:
    process = subprocess.Popen(
      (extractor_argument_path, extractor_argument),
      stdin=subprocess.DEVNULL,
      stdout=subprocess.PIPE,
      stderr=errors,
      pass_fds=tuple(inherited_fds),
    )
    if process.stdout is None:
      process.kill()
      process.wait()
      raise BackfillError(
        "backfill_reader_unavailable",
        "native rlog extractor has no stdout",
      )
    records: list[ExtractedEvent] = []
    selected_bytes = 0
    try:
      header = _read_process_exact(
        process,
        process.stdout,
        _STREAM_HEADER.size,
        abort_requested=abort_requested,
      )
      if len(header) != _STREAM_HEADER.size:
        raise RouteRejected("extractor_truncated", "missing stream header")
      magic, version, reserved = _STREAM_HEADER.unpack(header)
      if (
        magic != _STREAM_MAGIC
        or version != NATIVE_EXTRACTOR_SCHEMA_VERSION
        or reserved != 0
      ):
        raise RouteRejected(
          "extractor_schema_mismatch",
          "native extraction schema is incompatible",
        )
      while True:
        if abort_requested():
          raise BackfillError(
            "unexpected_error",
            "backfill aborted for onroad transition",
          )
        raw_header = _read_process_exact(
          process,
          process.stdout,
          _RECORD_HEADER.size,
          abort_requested=abort_requested,
        )
        if len(raw_header) != _RECORD_HEADER.size:
          raise RouteRejected(
            "extractor_truncated",
            "native extraction record is truncated",
          )
        size, which, mono_ns = _RECORD_HEADER.unpack(raw_header)
        if size == 0:
          if which != _END_RECORD or mono_ns != len(records):
            raise RouteRejected(
              "extractor_trailer_mismatch",
              "native extraction trailer does not match records",
            )
          return_code = _wait_process(
            process,
            abort_requested=abort_requested,
          )
          if process.stdout.read() != b"":
            raise RouteRejected(
              "extractor_trailing_output",
              "native extractor emitted trailing output",
            )
          break
        if size > MAXIMUM_EVENT_BYTES:
          raise RouteRejected(
            "event_too_large",
            "native event exceeds traversal bound",
          )
        encoded = _read_process_exact(
          process,
          process.stdout,
          size,
          abort_requested=abort_requested,
        )
        if len(encoded) != size:
          raise RouteRejected(
            "extractor_truncated",
            "native extraction payload is truncated",
          )
        records.append(ExtractedEvent(
          which=which,
          mono_ns=mono_ns,
          ordinal=len(records),
          encoded=encoded,
        ))
        selected_bytes += len(encoded)
        if (
          len(records) > MAXIMUM_SELECTED_RECORDS_PER_SEGMENT
          or selected_bytes > MAXIMUM_SELECTED_BYTES_PER_SEGMENT
        ):
          raise RouteRejected(
            "extractor_output_too_large",
            "selected segment stream exceeds bounded size",
          )
      errors.seek(0)
      error_text = errors.read().decode("utf-8", errors="replace").strip()
      if return_code != 0 or error_text:
        raise RouteRejected(
          "extractor_failed",
          f"native extraction failed ({return_code}): {error_text[:256]}",
        )
      return tuple(records)
    except BaseException:
      if process.poll() is None:
        process.kill()
      process.wait()
      raise
    finally:
      process.stdout.close()


def _bundle_physical_projection(
  bundle: RuntimeVehicleBundle,
) -> dict[str, object]:
  payload = bundle.to_dict()
  rack_resolutions = {
    float(node.parameters.rack_rate_resolution_deg_s)
    for node in bundle.calibration_seed_profile.nodes
  }
  if len(rack_resolutions) != 1:
    raise ValueError("runtime seed has inconsistent rack-rate resolution")
  return {
    "car_fingerprint": payload["car_fingerprint"],
    "nominal_rack_mapping": payload["nominal_rack_mapping"],
    "rack_rate_resolution_deg_s": rack_resolutions.pop(),
    "calibration_seed_profile": (
      bundle.calibration_seed_profile.to_dict()
    ),
    "stock_lateral_accel_offset_mps2": (
      bundle.stock_lateral_accel_offset_mps2
    ),
    "torque_callback_slope": payload["torque_callback_slope"],
    "torque_limits": payload["torque_limits"],
    "vehicle_identity": payload["vehicle_identity"],
  }


def _validate_route_bundle(
  *,
  route_car_params: Any,
  route_bundle: RuntimeVehicleBundle,
  current_car_params: Any,
  current_bundle: RuntimeVehicleBundle,
  descriptor: BuildDescriptor,
) -> str:
  if (
    str(route_car_params.carFingerprint)
    != descriptor.supported_vehicle_identity
    or str(route_car_params.carFingerprint)
    != str(current_car_params.carFingerprint)
  ):
    raise RouteRejected(
      "vehicle_identity_mismatch",
      "route belongs to another vehicle platform",
    )
  route_vin = str(route_car_params.carVin).strip()
  current_vin = str(current_car_params.carVin).strip()
  route_vin_available = route_vin and set(route_vin) != {"0"}
  current_vin_available = current_vin and set(current_vin) != {"0"}
  if (
    route_vin_available
    and current_vin_available
    and route_vin != current_vin
  ):
    raise RouteRejected(
      "vehicle_vin_mismatch",
      "route VIN belongs to another physical vehicle",
    )
  limits = route_bundle.torque_limits
  current_limits = current_bundle.torque_limits
  if (
    (
      limits.steer_max,
      limits.delta_up,
      limits.delta_down,
      limits.steer_step,
      limits.driver_allowance,
      limits.driver_multiplier,
      limits.driver_factor,
      limits.production_envelope_verified,
    )
    != (
      descriptor.steer_max,
      descriptor.steer_delta_up,
      descriptor.steer_delta_down,
      descriptor.steer_step,
      descriptor.driver_allowance,
      descriptor.driver_multiplier,
      descriptor.driver_factor,
      descriptor.production_envelope_verified,
    )
    or (
      limits.steer_max,
      limits.delta_up,
      limits.delta_down,
      limits.steer_step,
      limits.driver_allowance,
      limits.driver_multiplier,
      limits.driver_factor,
      limits.production_envelope_verified,
    )
    != (
      current_limits.steer_max,
      current_limits.delta_up,
      current_limits.delta_down,
      current_limits.steer_step,
      current_limits.driver_allowance,
      current_limits.driver_multiplier,
      current_limits.driver_factor,
      current_limits.production_envelope_verified,
    )
  ):
    raise RouteRejected(
      "controller_limits_mismatch",
      "normalized route torque uses a different physical envelope",
    )
  projection = _bundle_physical_projection(route_bundle)
  if (
    projection["rack_rate_resolution_deg_s"]
    != descriptor.rack_rate_resolution_deg_s
  ):
    raise RouteRejected(
      "controller_limits_mismatch",
      "route rack-rate resolution differs from its reviewed envelope",
    )
  if projection != _bundle_physical_projection(current_bundle):
    raise RouteRejected(
      "physical_runtime_mismatch",
      "route physical runtime does not match current vehicle",
    )
  return _sha256(_canonical_json_bytes(projection))


def _copy_car_state(message: Any) -> _RecordedCarState:
  return _RecordedCarState(
    v_ego=float(message.vEgo),
    steering_angle_deg=float(message.steeringAngleDeg),
    steering_rate_deg_s=float(message.steeringRateDeg),
    steering_torque=float(message.steeringTorque),
    steering_pressed=bool(message.steeringPressed),
    standstill=bool(message.standstill),
    steer_fault_temporary=bool(message.steerFaultTemporary),
    steer_fault_permanent=bool(message.steerFaultPermanent),
    can_valid=bool(message.canValid),
    can_timeout=bool(message.canTimeout),
  )


def _copy_car_control(message: Any) -> _RecordedCarControl:
  return _RecordedCarControl(
    lateral_active=bool(message.latActive),
    request_torque=float(message.actuators.torque),
  )


def _copy_car_output(message: Any) -> _RecordedCarOutput:
  raw_can_count = float(message.actuatorsOutput.torqueOutputCan)
  can_count_valid = (
    math.isfinite(raw_can_count)
    and raw_can_count.is_integer()
    and -(1 << 31) <= raw_can_count <= (1 << 31) - 1
  )
  return _RecordedCarOutput(
    applied_torque=float(message.actuatorsOutput.torque),
    torque_output_can_count=(int(raw_can_count) if can_count_valid else 0),
    torque_output_can_valid=can_count_valid,
  )


def _copy_live_parameters(message: Any) -> _RecordedLiveParameters:
  return _RecordedLiveParameters(
    valid=bool(message.valid),
    angle_offset_valid=bool(message.angleOffsetValid),
    steer_ratio_valid=bool(message.steerRatioValid),
    stiffness_factor_valid=bool(message.stiffnessFactorValid),
    angle_offset_deg=float(message.angleOffsetDeg),
    steer_ratio=float(message.steerRatio),
    stiffness_factor=float(message.stiffnessFactor),
    roll_rad=float(message.roll),
  )


def _copy_controls_state(message: Any) -> _RecordedControlsState:
  artifact = ""
  architecture = ""
  selection = -1
  source_commit = ""
  opendbc_commit = ""
  selection_bound = False
  try:
    lateral = message.lateralControlState
    if lateral.which() == "torqueState":
      torque_state = lateral.torqueState
      architecture = str(torque_state.modularArchitecture)
      selection = int(torque_state.modularSelection)
      artifact = str(torque_state.modularArtifactHash)
      source_commit = str(torque_state.modularSourceOpenpilotCommit)
      opendbc_commit = str(torque_state.modularOpendbcCommit)
      selection_bound = bool(torque_state.modularSelectionBound)
  except Exception:
    # Historical schemas have no modular telemetry.  They remain valid for
    # physical calibration but cannot claim a verified behavior source.
    pass
  return _RecordedControlsState(
    lateral_plan_mono_ns=int(message.lateralPlanMonoTime),
    measured_curvature=float(message.curvature),
    desired_curvature=float(message.desiredCurvature),
    modular_architecture=architecture,
    modular_selection=selection,
    modular_artifact_sha256=artifact,
    modular_source_openpilot_commit=source_commit,
    modular_opendbc_commit=opendbc_commit,
    modular_selection_bound=selection_bound,
  )


def _float_tuple(values: Any) -> tuple[float, ...]:
  return tuple(float(value) for value in values)


def _copy_model(message: Any) -> _RecordedModel:
  plan_times = _float_tuple(message.orientationRate.t)
  orientation_rate_z = _float_tuple(message.orientationRate.z)
  velocity_times = _float_tuple(message.velocity.t)
  velocity_x = _float_tuple(message.velocity.x)
  native_grid_valid = (
    bool(plan_times)
    and plan_times == velocity_times
    and len(plan_times) == len(orientation_rate_z) == len(velocity_x)
    and all(math.isfinite(value) for value in (
      *plan_times,
      *orientation_rate_z,
      *velocity_x,
    ))
    and all(
      right > left
      for left, right in zip(plan_times, plan_times[1:], strict=False)
    )
  )
  return _RecordedModel(
    frame_id=int(message.frameId),
    timestamp_eof_ns=int(message.timestampEof),
    scalar_curvature=float(message.action.desiredCurvature),
    desired_curvature_time_s=float(
      message.action.desiredCurvatureTime,
    ),
    plan_times_s=plan_times if native_grid_valid else (),
    orientation_rate_z=(
      orientation_rate_z if native_grid_valid else ()
    ),
    velocity_x=velocity_x if native_grid_valid else (),
    native_grid_valid=native_grid_valid,
  )


def _copy_live_torque_parameters(
  message: Any,
) -> _RecordedLiveTorqueParameters:
  use_filtered = bool(message.useParams)
  return _RecordedLiveTorqueParameters(
    live_valid=bool(message.liveValid),
    use_params=use_filtered,
    version=int(message.version),
    lat_accel_factor=float(
      message.latAccelFactorFiltered
      if use_filtered
      else message.latAccelFactorRaw
    ),
    lat_accel_offset=float(
      message.latAccelOffsetFiltered
      if use_filtered
      else message.latAccelOffsetRaw
    ),
    friction=float(
      message.frictionCoefficientFiltered
      if use_filtered
      else message.frictionCoefficientRaw
    ),
  )


def _copy_live_delay(message: Any) -> _RecordedLiveDelay:
  return _RecordedLiveDelay(
    lateral_delay_s=float(message.lateralDelay),
    status=str(message.status),
    version=int(message.version),
  )


def _copy_lateral_maneuver_plan(
  message: Any,
) -> _RecordedLateralManeuverPlan:
  return _RecordedLateralManeuverPlan(
    desired_curvature=float(message.desiredCurvature),
  )


def _copy_driving_event(message: Any) -> _RecordedDrivingEvent:
  return _RecordedDrivingEvent(
    event_id=str(message.eventId),
    occurred_mono_ns=int(message.occurredMonoTime),
    analysis_window_before_s=float(message.analysisWindowBeforeS),
    analysis_window_after_s=float(message.analysisWindowAfterS),
    event_type=str(message.eventType),
    severity=str(message.severity),
  )


def _copy_selected_payload(which: str, message: Any) -> Any:
  copiers: dict[str, Callable[[Any], Any]] = {
    "carState": _copy_car_state,
    "carControl": _copy_car_control,
    "carOutput": _copy_car_output,
    "liveParameters": _copy_live_parameters,
    "controlsState": _copy_controls_state,
    "modelV2": _copy_model,
    "liveTorqueParameters": _copy_live_torque_parameters,
    "liveDelay": _copy_live_delay,
    "lateralManeuverPlan": _copy_lateral_maneuver_plan,
    "drivingEvent": _copy_driving_event,
  }
  copier = copiers.get(which)
  return None if copier is None else copier(message)


def _decode_extracted_event(
  record: ExtractedEvent,
  event_reader: Callable[[bytes], AbstractContextManager[Any]],
) -> _DecodedExtractedEvent:
  """Copy one bounded Event into route-local, context-independent values."""
  try:
    with event_reader(record.encoded) as event:
      which = event.which()
      if which not in _EVENT_WHICH or _EVENT_WHICH[which] != record.which:
        raise RouteRejected(
          "event_kind_mismatch",
          "native event kind does not match bounded decoder",
        )
      mono_ns = int(event.logMonoTime)
      if mono_ns != record.mono_ns:
        raise RouteRejected(
          "event_time_mismatch",
          "native event timestamp does not match bounded decoder",
        )
      valid = bool(event.valid)
      if which == "sentinel":
        payload: Any = str(event.sentinel.type)
      elif which == "initData":
        init = event.initData
        payload = (
          str(init.gitCommit),
          bool(init.dirty),
          str(init.dongleId),
          str(init.version),
        )
      elif which == "carParams":
        payload = bytes(event.carParams.as_builder().to_bytes())
      elif (
        which in _SOURCE_SERVICES
        or which in _BEHAVIOR_CONTEXT_SERVICES
        or which in _CANONICAL_WITNESS_SERVICES
      ):
        payload = _copy_selected_payload(
          which,
          getattr(event, which),
        )
      else:
        payload = None
      return _DecodedExtractedEvent(
        which=which,
        mono_ns=mono_ns,
        valid=valid,
        payload=payload,
      )
  except (BackfillError, RouteRejected):
    raise
  except Exception as exc:
    raise RouteRejected(
      "event_decode_failed",
      "bounded route event could not be decoded",
    ) from exc


def _float32_bytes(value: float) -> bytes:
  return struct.pack("<f", float(value))


def _evaluate_recorded_curvature(
  vehicle_model: Any,
  car_state: _RecordedCarState,
  live_parameters: _RecordedLiveParameters,
) -> float:
  vehicle_model.update_params(
    max(live_parameters.stiffness_factor, 0.1),
    max(live_parameters.steer_ratio, 0.1),
  )
  return -float(vehicle_model.calc_curvature(
    math.radians(
      car_state.steering_angle_deg
      - live_parameters.angle_offset_deg
    ),
    car_state.v_ego,
    live_parameters.roll_rad,
  ))


def _latest_record_at_or_before(
  records: tuple[_TimedRouteRecord, ...],
  timestamps: tuple[int, ...],
  mono_ns: int,
) -> _TimedRouteRecord | None:
  index = bisect_right(timestamps, mono_ns) - 1
  return None if index < 0 else records[index]


def _pair_same_cycle_car_controls(
  controls: tuple[_TimedRouteRecord, ...],
  car_controls: tuple[_TimedRouteRecord, ...],
) -> dict[int, _TimedRouteRecord]:
  """Pair the first unmatched command in each controlsState cycle.

  controlsd publishes ``controlsState`` before ``carControl``.  Timestamp-only
  latest-before joining therefore selects the preceding cycle.  This bounded
  forward pairing reconstructs the actual cycle without allowing a command
  to be reused by two witnesses.
  """
  paired: dict[int, _TimedRouteRecord] = {}
  command_index = 0
  for index, witness in enumerate(controls):
    next_witness_ns = (
      controls[index + 1].mono_ns
      if index + 1 < len(controls)
      else witness.mono_ns + MAXIMUM_CONTROL_GAP_NS + 1
    )
    while (
      command_index < len(car_controls)
      and car_controls[command_index].mono_ns < witness.mono_ns
    ):
      command_index += 1
    if command_index >= len(car_controls):
      continue
    candidate = car_controls[command_index]
    if (
      candidate.mono_ns < next_witness_ns
      and candidate.mono_ns - witness.mono_ns
      <= MAXIMUM_CONTROL_GAP_NS
    ):
      paired[witness.source_order] = candidate
      command_index += 1
  return paired


def _build_canonical_control_joins(
  *,
  route_name: str,
  controls: tuple[_TimedRouteRecord, ...],
  polls: tuple[_TimedRouteRecord, ...],
  car_states: tuple[_TimedRouteRecord, ...],
  live_parameters: tuple[_TimedRouteRecord, ...],
  car_outputs: tuple[_TimedRouteRecord, ...],
  car_controls: tuple[_TimedRouteRecord, ...],
  vehicle_model: Any,
) -> tuple[tuple[_CanonicalControlJoin, ...], tuple[int, ...]]:
  """Freeze the controller-independent SubMaster input race once.

  ``controlsState.curvature`` is a Float32 selection oracle.  It is computed
  upstream of every controller candidate, so matching its exact Float32 bits
  reconstructs the recording's source scheduling without tailoring inputs to
  the learner or a replayed controller.
  """
  if not controls:
    raise RouteRejected(
      "missing_controls_witness",
      "route contains no controlsState witnesses",
    )
  if not polls:
    raise RouteRejected(
      "missing_selfdrive_poll",
      "route contains no selfdriveState poll witnesses",
    )
  if not car_states or not car_outputs:
    raise RouteRejected(
      "missing_measurement_service",
      "route lacks carState or carOutput measurements",
    )

  ordered_controls = tuple(sorted(
    controls,
    key=lambda record: (record.mono_ns, record.source_order),
  ))
  ordered_polls = tuple(sorted(
    polls,
    key=lambda record: (record.mono_ns, record.source_order),
  ))
  ordered_states = tuple(sorted(
    car_states,
    key=lambda record: (record.mono_ns, record.source_order),
  ))
  ordered_parameters = tuple(sorted(
    live_parameters,
    key=lambda record: (record.mono_ns, record.source_order),
  ))
  ordered_outputs = tuple(sorted(
    car_outputs,
    key=lambda record: (record.mono_ns, record.source_order),
  ))
  ordered_commands = tuple(sorted(
    car_controls,
    key=lambda record: (record.mono_ns, record.source_order),
  ))
  poll_times = tuple(record.mono_ns for record in ordered_polls)
  state_times = tuple(record.mono_ns for record in ordered_states)
  parameter_times = tuple(record.mono_ns for record in ordered_parameters)
  output_times = tuple(record.mono_ns for record in ordered_outputs)

  first_poll = poll_times[0]
  first_scoreable = 0
  while (
    first_scoreable < len(ordered_controls)
    and ordered_controls[first_scoreable].mono_ns < first_poll
  ):
    first_scoreable += 1
  dropped = tuple(
    record.mono_ns for record in ordered_controls[:first_scoreable]
  )
  scoreable = ordered_controls[first_scoreable:]
  if not scoreable:
    raise RouteRejected(
      "missing_controls_witness",
      "all controlsState witnesses precede the first selfdriveState poll",
    )
  paired_commands = _pair_same_cycle_car_controls(
    scoreable,
    ordered_commands,
  )
  default_parameters = _TimedRouteRecord(
    mono_ns=-1,
    segment_index=0,
    ordinal=0,
    source_order=-1,
    valid=True,
    payload=_RecordedLiveParameters(
      valid=False,
      angle_offset_valid=False,
      steer_ratio_valid=False,
      stiffness_factor_valid=False,
      angle_offset_deg=0.0,
      steer_ratio=0.0,
      stiffness_factor=0.0,
      roll_rad=0.0,
    ),
  )

  joins: list[_CanonicalControlJoin] = []
  previous_witness_ns: int | None = None
  for witness in scoreable:
    poll_index = bisect_right(poll_times, witness.mono_ns) - 1
    if poll_index < 0:
      raise AssertionError("pre-poll controls prefix was not removed")
    poll_ns = poll_times[poll_index]
    baseline_state = _latest_record_at_or_before(
      ordered_states,
      state_times,
      poll_ns,
    )
    if baseline_state is None:
      raise RouteRejected(
        "measurement_race_unreconstructable",
        f"{route_name}: selfdriveState poll precedes the first carState",
      )
    baseline_parameters = (
      _latest_record_at_or_before(
        ordered_parameters,
        parameter_times,
        poll_ns,
      )
      or default_parameters
    )
    car_output = _latest_record_at_or_before(
      ordered_outputs,
      output_times,
      poll_ns,
    )
    if car_output is None:
      raise RouteRejected(
        "measurement_race_unreconstructable",
        f"{route_name}: selfdriveState poll precedes the first carOutput",
      )
    output_index = bisect_right(output_times, car_output.mono_ns) - 1
    previous_output_ns = (
      None
      if output_index <= 0
      else ordered_outputs[output_index - 1].mono_ns
    )

    state_left = bisect_right(state_times, poll_ns)
    state_right = bisect_right(state_times, witness.mono_ns)
    parameter_left = bisect_right(parameter_times, poll_ns)
    parameter_right = bisect_right(
      parameter_times,
      witness.mono_ns,
    )
    state_candidates = (
      baseline_state,
      *ordered_states[state_left:state_right],
    )
    parameter_candidates = (
      baseline_parameters,
      *ordered_parameters[parameter_left:parameter_right],
    )
    controls_payload = witness.payload
    if not isinstance(controls_payload, _RecordedControlsState):
      raise RouteRejected(
        "measurement_race_unreconstructable",
        "controlsState payload has an invalid compact type",
      )
    logged_bits = _float32_bytes(controls_payload.measured_curvature)
    matches: list[tuple[object, ...]] = []
    for state_record in state_candidates:
      state = state_record.payload
      if not isinstance(state, _RecordedCarState):
        raise RouteRejected(
          "measurement_race_unreconstructable",
          "carState compact payload has an invalid type",
        )
      for parameter_record in parameter_candidates:
        parameters = parameter_record.payload
        if not isinstance(parameters, _RecordedLiveParameters):
          raise RouteRejected(
            "measurement_race_unreconstructable",
            "liveParameters compact payload has an invalid type",
          )
        calculated = _evaluate_recorded_curvature(
          vehicle_model,
          state,
          parameters,
        )
        if _float32_bytes(calculated) != logged_bits:
          continue
        baseline = (
          state_record is baseline_state
          and parameter_record is baseline_parameters
        )
        matches.append((
          abs(calculated - controls_payload.measured_curvature),
          0 if baseline else 1,
          state_record.mono_ns,
          parameter_record.mono_ns,
          state_record.source_order,
          parameter_record.source_order,
          state_record,
          parameter_record,
        ))
    unresolved = not matches
    if unresolved:
      selected_state = baseline_state
      selected_parameters = baseline_parameters
    else:
      selected = min(matches, key=lambda item: item[:6])
      selected_state = selected[6]
      selected_parameters = selected[7]
    joins.append(_CanonicalControlJoin(
      witness=witness,
      poll_mono_ns=poll_ns,
      car_state=selected_state,
      live_parameters=selected_parameters,
      car_output=car_output,
      previous_car_output_mono_ns=previous_output_ns,
      car_control=paired_commands.get(witness.source_order),
      curvature_unresolved=unresolved,
      gap_from_previous=(
        previous_witness_ns is not None
        and witness.mono_ns - previous_witness_ns
        > MAXIMUM_CONTROL_GAP_NS
      ),
    ))
    previous_witness_ns = witness.mono_ns
  return tuple(joins), dropped


def _measured_frame_from_join(
  join: _CanonicalControlJoin,
) -> MeasuredLearningFrame:
  car_state = join.car_state.payload
  live_parameters = join.live_parameters.payload
  car_output = join.car_output.payload
  car_control = (
    None if join.car_control is None else join.car_control.payload
  )
  if (
    not isinstance(car_state, _RecordedCarState)
    or not isinstance(live_parameters, _RecordedLiveParameters)
    or not isinstance(car_output, _RecordedCarOutput)
    or (
      car_control is not None
      and not isinstance(car_control, _RecordedCarControl)
    )
  ):
    raise RouteRejected(
      "measured_frame_invalid",
      "canonical join contains an invalid compact payload",
    )
  applied_effective_ns = (
    0
    if join.previous_car_output_mono_ns is None
    else join.previous_car_output_mono_ns
  )
  source_valid = (
    join.witness.valid
    and join.car_state.valid
    and join.live_parameters.valid
    and join.car_output.valid
    and join.car_control is not None
    and join.car_control.valid
  )
  return MeasuredLearningFrame(
    sample_mono_ns=join.witness.mono_ns,
    response_mono_ns=join.car_state.mono_ns,
    applied_report_mono_ns=join.car_output.mono_ns,
    applied_effective_mono_ns=applied_effective_ns,
    speed_mps=car_state.v_ego,
    steering_angle_deg=car_state.steering_angle_deg,
    steering_rate_deg_s=car_state.steering_rate_deg_s,
    steering_torque=car_state.steering_torque,
    steering_pressed=car_state.steering_pressed,
    standstill=car_state.standstill,
    steer_fault_temporary=car_state.steer_fault_temporary,
    steer_fault_permanent=car_state.steer_fault_permanent,
    can_valid=car_state.can_valid,
    can_timeout=car_state.can_timeout,
    applied_torque=car_output.applied_torque,
    lateral_active=(
      False if car_control is None else car_control.lateral_active
    ),
    live_parameters_valid=live_parameters.valid,
    angle_offset_valid=live_parameters.angle_offset_valid,
    steer_ratio_valid=live_parameters.steer_ratio_valid,
    stiffness_factor_valid=live_parameters.stiffness_factor_valid,
    angle_offset_deg=live_parameters.angle_offset_deg,
    steer_ratio=live_parameters.steer_ratio,
    stiffness_factor=live_parameters.stiffness_factor,
    roll_rad=live_parameters.roll_rad,
    inputs_valid=(
      source_valid
      and not join.curvature_unresolved
      and 0 < join.car_state.mono_ns <= join.witness.mono_ns
      and 0 < join.car_output.mono_ns <= join.poll_mono_ns
      and (
        applied_effective_ns == 0
        or 0 < applied_effective_ns < join.car_output.mono_ns
      )
    ),
  )


def _exact_model_indices(
  joins: tuple[_CanonicalControlJoin, ...],
  model_records: tuple[_TimedRouteRecord, ...],
) -> tuple[int | None, ...]:
  """Resolve only the model publication named by lateralPlanMonoTime."""
  by_publication: dict[int, int | None] = {}
  for index, record in enumerate(model_records):
    if record.mono_ns in by_publication:
      by_publication[record.mono_ns] = None
    else:
      by_publication[record.mono_ns] = index
  indices: list[int | None] = []
  for join in joins:
    controls = join.witness.payload
    if not isinstance(controls, _RecordedControlsState):
      indices.append(None)
      continue
    linked = by_publication.get(controls.lateral_plan_mono_ns)
    indices.append(linked if isinstance(linked, int) else None)
  return tuple(indices)


def _behavior_controller_source(
  joins: tuple[_CanonicalControlJoin, ...],
  *,
  source_superproject_commit: str,
  source_opendbc_commit: str,
  source_panda_commit: str,
) -> tuple[bool, str, str, str]:
  """Bind all active behavior frames to one independently verifiable source."""
  source_payloads: set[bytes] = set()
  active_count = 0
  for join in joins:
    command = None if join.car_control is None else join.car_control.payload
    if (
      not isinstance(command, _RecordedCarControl)
      or not command.lateral_active
    ):
      continue
    active_count += 1
    controls = join.witness.payload
    if not isinstance(controls, _RecordedControlsState):
      return False, "invalid_controls_source", "0" * 64, "ineligible"
    if controls.modular_selection == 0:
      if not controls.modular_architecture:
        return False, "unverified_stock_composition", "0" * 64, "ineligible"
      payload = _canonical_json_bytes({
        "architecture": controls.modular_architecture,
        "kind": "stock_canonical_composition",
        "opendbc_commit": source_opendbc_commit,
        "panda_commit": source_panda_commit,
        "superproject_commit": source_superproject_commit,
      })
    elif controls.modular_selection == 1:
      if (
        not controls.modular_selection_bound
        or _SHA256_RE.fullmatch(
          controls.modular_artifact_sha256,
        ) is None
        or controls.modular_source_openpilot_commit
        != source_superproject_commit
        or controls.modular_opendbc_commit != source_opendbc_commit
      ):
        return False, "unverified_modular_artifact", "0" * 64, "ineligible"
      payload = _canonical_json_bytes({
        "artifact_sha256": controls.modular_artifact_sha256,
        "kind": "verified_modular_artifact",
        "opendbc_commit": source_opendbc_commit,
        "panda_commit": source_panda_commit,
        "superproject_commit": source_superproject_commit,
      })
    else:
      return False, "unverified_controller_source", "0" * 64, "ineligible"
    source_payloads.add(payload)
  if active_count == 0:
    return False, "no_active_lateral_frames", "0" * 64, "ineligible"
  if len(source_payloads) != 1:
    return False, "mixed_controller_sources", "0" * 64, "ineligible"
  payload = source_payloads.pop()
  source_kind = (
    "modular_artifact"
    if b'"kind":"verified_modular_artifact"' in payload
    else "stock_canonical"
  )
  return True, "eligible", _sha256(payload), source_kind


def _latest_sparse_indices(
  joins: tuple[_CanonicalControlJoin, ...],
  records: tuple[_TimedRouteRecord, ...],
) -> tuple[int | None, ...]:
  """Return the last publication owned by each reconstructed poll cycle."""
  timestamps = tuple(record.mono_ns for record in records)
  return tuple(
    (index if (index := bisect_right(timestamps, join.poll_mono_ns) - 1) >= 0 else None)
    for join in joins
  )


def _reconstruct_live_torque_health(
  *,
  poll_mono_ns: int,
  publication_index: int | None,
  records: tuple[_TimedRouteRecord, ...],
) -> tuple[bool, bool]:
  """Conservatively prove witness-time SubMaster ``all_checks``.

  controlsd receives this 4 Hz non-polled service on the canonical 100 Hz
  poll.  When every observed publication interval is comfortably inside the
  FrequencyTracker band, its result is invariant to one-cycle receive jitter.
  Boundary cases remain explicitly inexact; they are never guessed.
  """
  if publication_index is None:
    return False, True  # unseen static-frequency service is exactly unhealthy
  latest = records[publication_index]
  if not latest.valid:
    return False, True  # all_valid is false regardless of alive/frequency
  if publication_index == 0:
    return False, True  # FrequencyTracker has no interval yet
  nominal_poll_jitter_ns = MAXIMUM_CONTROL_GAP_NS
  alive_limit_ns = int(10e9 / SERVICE_LIST["liveTorqueParameters"].frequency)
  age_ns = poll_mono_ns - latest.mono_ns
  if age_ns < 0:
    return False, False
  if age_ns >= alive_limit_ns + nominal_poll_jitter_ns:
    return False, True
  if age_ns >= alive_limit_ns - nominal_poll_jitter_ns:
    return False, False
  # FrequencyTracker's 4 Hz acceptable range is 3.2..4.8 Hz.  Requiring each
  # source interval to stay inside the band after +/- one poll proves both its
  # full and recent moving averages valid without reconstructing wall time.
  minimum_dt_ns = int(1e9 / (4.8)) + 2 * nominal_poll_jitter_ns
  maximum_dt_ns = int(1e9 / (3.2)) - 2 * nominal_poll_jitter_ns
  intervals = (
    records[index].mono_ns - records[index - 1].mono_ns
    for index in range(1, publication_index + 1)
  )
  if all(minimum_dt_ns <= value <= maximum_dt_ns for value in intervals):
    return True, True
  return False, False


def _active_witness_missing_exact_model_link(
  joins: tuple[_CanonicalControlJoin, ...],
  model_indices: tuple[int | None, ...],
) -> bool:
  """Return whether behavior replay lacks a model link while lateral is live.

  Missing model publications before lateral activation are valid startup
  context and remain in the shared control plane.  Once lateral is active,
  however, the exact ``lateralPlanMonoTime`` publication is authority data:
  synthesizing or carrying a nearby plan would make replay non-canonical.
  """
  if len(joins) != len(model_indices):
    raise ValueError("model-link population does not match controls witnesses")
  return any(
    model_indices[index] is None
    and isinstance(join.car_control.payload, _RecordedCarControl)
    and join.car_control.payload.lateral_active
    for index, join in enumerate(joins)
    if join.car_control is not None
  )


def _route_evidence_artifact(
  *,
  route: RouteCandidate,
  route_time_origin_mono_ns: int,
  route_car_params_bytes: bytes,
  route_bundle: RuntimeVehicleBundle,
  route_descriptor: BuildDescriptor,
  route_records: Mapping[str, list[_TimedRouteRecord]],
  joins: tuple[_CanonicalControlJoin, ...],
  frames: tuple[MeasuredLearningFrame, ...],
  pre_poll_dropped: tuple[int, ...],
  gap_count: int,
  provenance: Mapping[str, object],
) -> RouteEvidenceArtifact:
  """Encode the one shared physical/behavior preparation result."""
  from openpilot.selfdrive.controls.lib.blatv2.learning_backfill_spool import (
    _encode_frame,
  )

  models_raw = tuple(sorted(
    route_records["modelV2"],
    key=lambda record: (record.mono_ns, record.segment_index, record.ordinal),
  ))
  torque_raw = tuple(sorted(
    route_records["liveTorqueParameters"],
    key=lambda record: (record.mono_ns, record.segment_index, record.ordinal),
  ))
  delay_raw = tuple(sorted(
    route_records["liveDelay"],
    key=lambda record: (record.mono_ns, record.segment_index, record.ordinal),
  ))
  maneuver_raw = tuple(sorted(
    route_records["lateralManeuverPlan"],
    key=lambda record: (record.mono_ns, record.segment_index, record.ordinal),
  ))
  model_indices = _exact_model_indices(joins, models_raw)
  torque_indices = _latest_sparse_indices(joins, torque_raw)
  delay_indices = _latest_sparse_indices(joins, delay_raw)
  maneuver_indices = _latest_sparse_indices(joins, maneuver_raw)
  torque_health = tuple(
    _reconstruct_live_torque_health(
      poll_mono_ns=join.poll_mono_ns,
      publication_index=torque_indices[index],
      records=torque_raw,
    )
    for index, join in enumerate(joins)
  )

  models = tuple(
    ModelPublication(
      segment_index=record.segment_index,
      ordinal=record.ordinal,
      mono_time_ns=record.mono_ns,
      frame_id=payload.frame_id,
      timestamp_eof_ns=payload.timestamp_eof_ns,
      scalar_curvature=payload.scalar_curvature,
      desired_curvature_time_s=payload.desired_curvature_time_s,
      plan_times=payload.plan_times_s,
      orientation_rate_z=payload.orientation_rate_z,
      velocity_x=payload.velocity_x,
      message_valid=record.valid,
      native_grid_valid=payload.native_grid_valid,
    )
    for record in models_raw
    if isinstance((payload := record.payload), _RecordedModel)
  )
  if len(models) != len(models_raw):
    raise RouteRejected(
      "behavior_context_invalid",
      "model behavior plane contains an invalid compact payload",
    )
  torque = tuple(
    LiveTorqueParametersPublication(
      segment_index=record.segment_index,
      ordinal=record.ordinal,
      mono_time_ns=record.mono_ns,
      lat_accel_factor=payload.lat_accel_factor,
      lat_accel_offset=payload.lat_accel_offset,
      friction=payload.friction,
      version=payload.version,
      message_valid=record.valid,
      live_valid=payload.live_valid,
      use_params=payload.use_params,
    )
    for record in torque_raw
    if isinstance((payload := record.payload), _RecordedLiveTorqueParameters)
  )
  delays = tuple(
    LiveDelayPublication(
      segment_index=record.segment_index,
      ordinal=record.ordinal,
      mono_time_ns=record.mono_ns,
      lateral_delay_s=payload.lateral_delay_s,
      version=payload.version,
      message_valid=record.valid,
      status=payload.status,
    )
    for record in delay_raw
    if isinstance((payload := record.payload), _RecordedLiveDelay)
  )
  maneuvers = tuple(
    LateralManeuverPlanPublication(
      segment_index=record.segment_index,
      ordinal=record.ordinal,
      mono_time_ns=record.mono_ns,
      desired_curvature=payload.desired_curvature,
      message_valid=record.valid,
    )
    for record in maneuver_raw
    if isinstance((payload := record.payload), _RecordedLateralManeuverPlan)
  )
  if len(torque) != len(torque_raw) or len(delays) != len(delay_raw) or len(maneuvers) != len(maneuver_raw):
    raise RouteRejected(
      "behavior_context_invalid",
      "sparse behavior plane contains an invalid compact payload",
    )

  source_eligible, source_reason, source_hash, source_kind = (
    _behavior_controller_source(
      joins,
      source_superproject_commit=route_descriptor.superproject_commit,
      source_opendbc_commit=route_descriptor.opendbc_commit,
      source_panda_commit=route_descriptor.panda_commit,
    )
  )
  model_failure_count = sum(index is None for index in model_indices)
  active_exact_count_failure = any(
    isinstance(join.car_control.payload, _RecordedCarControl)
    and join.car_control.payload.lateral_active
    and (
      not isinstance(join.car_output.payload, _RecordedCarOutput)
      or not join.car_output.payload.torque_output_can_valid
    )
    for join in joins
    if join.car_control is not None
  )
  behavior_eligible = source_eligible
  behavior_reason = source_reason
  if behavior_eligible and _active_witness_missing_exact_model_link(
    joins,
    model_indices,
  ):
    behavior_eligible = False
    behavior_reason = "exact_model_link_missing"
  elif behavior_eligible and not maneuver_raw:
    behavior_eligible = False
    behavior_reason = "lateral_maneuver_plan_missing"
  elif behavior_eligible and active_exact_count_failure:
    behavior_eligible = False
    behavior_reason = "exact_applied_can_count_missing"
  elif behavior_eligible and source_kind == "stock_canonical" and any(
    isinstance(join.car_control.payload, _RecordedCarControl)
    and join.car_control.payload.lateral_active
    and not torque_health[index][1]
    for index, join in enumerate(joins)
    if join.car_control is not None
  ):
    behavior_eligible = False
    behavior_reason = "live_torque_submaster_health_unreconstructable"

  event_by_id: dict[str, DrivingEventLocator] = {}
  event_conflict = False
  for record in sorted(
    route_records["drivingEvent"],
    key=lambda item: (item.mono_ns, item.segment_index, item.ordinal),
  ):
    payload = record.payload
    if not isinstance(payload, _RecordedDrivingEvent) or not payload.event_id:
      event_conflict = True
      continue
    locator = DrivingEventLocator(
      segment_index=record.segment_index,
      ordinal=record.ordinal,
      publication_mono_time_ns=record.mono_ns,
      occurred_mono_time_ns=payload.occurred_mono_ns,
      analysis_window_before_s=payload.analysis_window_before_s,
      analysis_window_after_s=payload.analysis_window_after_s,
      event_id=payload.event_id,
      event_type=payload.event_type,
      severity=payload.severity,
      message_valid=record.valid,
    )
    existing = event_by_id.get(payload.event_id)
    if existing is None:
      event_by_id[payload.event_id] = locator
    elif existing != locator:
      event_conflict = True
  if behavior_eligible and event_conflict:
    behavior_eligible = False
    behavior_reason = "driving_event_identity_conflict"

  controls: list[ControlsWitness] = []
  previous_active = False
  previous_intervening = False
  for index, join in enumerate(joins):
    state = join.car_state.payload
    output = join.car_output.payload
    control = None if join.car_control is None else join.car_control.payload
    controls_state = join.witness.payload
    if (
      not isinstance(state, _RecordedCarState)
      or not isinstance(output, _RecordedCarOutput)
      or not isinstance(controls_state, _RecordedControlsState)
      or (control is not None and not isinstance(control, _RecordedCarControl))
    ):
      raise RouteRejected(
        "behavior_context_invalid",
        "control witness contains an invalid compact payload",
      )
    active = control is not None and control.lateral_active
    intervening = active and state.steering_pressed
    observed_onset = active and intervening and previous_active and not previous_intervening
    uncertain_onset = intervening and (
      not previous_active or join.gap_from_previous
    )
    model_index = model_indices[index]
    model_alive = (
      model_index is not None
      and models_raw[model_index].mono_ns <= join.witness.mono_ns
      and join.witness.mono_ns - models_raw[model_index].mono_ns
      < int(10e9 / SERVICE_LIST["modelV2"].frequency)
    )
    raw_request = 0.0 if control is None else control.request_torque
    controls.append(ControlsWitness(
      segment_index=join.witness.segment_index,
      ordinal=join.witness.ordinal,
      mono_time_ns=join.witness.mono_ns,
      physical_record_index=index,
      model_publication_index=(-1 if model_index is None else model_index),
      live_torque_parameters_index=(-1 if torque_indices[index] is None else torque_indices[index]),
      live_delay_index=(-1 if delay_indices[index] is None else delay_indices[index]),
      lateral_maneuver_plan_index=(-1 if maneuver_indices[index] is None else maneuver_indices[index]),
      poll_mono_time_ns=join.poll_mono_ns,
      state_sample_mono_ns=join.car_state.mono_ns,
      live_parameters_mono_ns=join.live_parameters.mono_ns,
      car_output_report_mono_ns=join.car_output.mono_ns,
      car_output_effective_mono_ns=(0 if join.previous_car_output_mono_ns is None else join.previous_car_output_mono_ns),
      car_control_mono_ns=(-1 if join.car_control is None else join.car_control.mono_ns),
      raw_request_torque=raw_request,
      measured_curvature=controls_state.measured_curvature,
      desired_curvature=controls_state.desired_curvature,
      envelope_headroom=max(0.0, min(1.0, 1.0 - abs(raw_request))),
      torque_output_can_count=output.torque_output_can_count,
      message_valid=join.witness.valid,
      model_message_alive=model_alive,
      model_link_valid=model_index is not None,
      inputs_valid=frames[index].inputs_valid,
      lateral_active=active,
      driver_intervening=intervening,
      steer_fault=state.steer_fault_temporary or state.steer_fault_permanent,
      intervention_onset=observed_onset,
      intervention_onset_uncertain=uncertain_onset,
      race_unresolved=join.curvature_unresolved,
      gap_from_previous=join.gap_from_previous,
      car_control_paired=join.car_control is not None,
      torque_output_can_valid=output.torque_output_can_valid,
      maneuver_plan_available=maneuver_indices[index] is not None,
      live_torque_parameters_available=torque_indices[index] is not None,
      live_delay_available=delay_indices[index] is not None,
      live_torque_parameters_checks_passed=torque_health[index][0],
      live_torque_parameters_health_exact=torque_health[index][1],
    ))
    previous_active = active
    previous_intervening = intervening

  cache_payload = {
    "canonical_join_schema_version": CANONICAL_JOIN_SCHEMA_VERSION,
    "extractor_schema_version": NATIVE_EXTRACTOR_SCHEMA_VERSION,
    "log_schema_blob": route_descriptor.log_schema_blob,
    "opendbc_commit": route_descriptor.opendbc_commit,
    "route_segments": [
      {"index": segment.index, "sha256": segment.sha256, "size_bytes": segment.size_bytes}
      for segment in route.segments
    ],
    "runtime_bundle_sha256": _sha256(route_bundle.to_json().encode()),
    "superproject_commit": route_descriptor.superproject_commit,
  }
  source = RouteEvidenceSourceIdentity(
    route_id=route.route_name,
    route_time_origin_mono_ns=route_time_origin_mono_ns,
    route_segment_sha256=tuple(segment.sha256 for segment in route.segments),
    route_segment_size_bytes=tuple(segment.size_bytes for segment in route.segments),
    source_superproject_commit=route_descriptor.superproject_commit,
    source_opendbc_commit=route_descriptor.opendbc_commit,
    source_panda_commit=route_descriptor.panda_commit,
    controller_source_kind=(source_kind if behavior_eligible else "ineligible"),
    controller_artifact_sha256=source_hash,
    behavior_eligible=behavior_eligible,
    behavior_ineligible_reason=behavior_reason,
    vehicle_identity=route_bundle.vehicle_identity,
    runtime_identity=_sha256(route_bundle.to_json().encode()),
    schema_versions={
      "canonical_join": CANONICAL_JOIN_SCHEMA_VERSION,
      "extractor": NATIVE_EXTRACTOR_SCHEMA_VERSION,
      "route_evidence": 2,
      "live_torque_health_reconstruction": 1,
    },
    preparation_provenance=dict(provenance),
    physical_plane_encoding_id="blatv2-measured-learning-frame-v1",
    physical_record_count=len(frames),
    preparation_cache_key=_sha256(_canonical_json_bytes(cache_payload)),
    controls_witness_count=len(joins) + len(pre_poll_dropped),
    unresolved_witness_count=(
      sum(join.curvature_unresolved or join.car_control is None for join in joins)
      + len(pre_poll_dropped)
    ),
    gap_count=gap_count,
    model_link_failure_count=model_failure_count,
    pre_poll_dropped_timestamps_ns=pre_poll_dropped,
  )
  physical = b"".join(_encode_frame(frame) for frame in frames)
  return RouteEvidenceArtifact(
    source,
    route_car_params_bytes,
    physical,
    models,
    tuple(controls),
    torque,
    delays,
    maneuvers,
    tuple(event_by_id.values()),
  )


def _prepare_route_with_extractor(
  route: RouteCandidate,
  *,
  extractor_path: str | Path,
  extractor_fd: int,
  event_reader: Callable[[bytes], AbstractContextManager[Any]],
  car_params_decoder: Callable[[bytes], Any],
  descriptor_registry: BuildDescriptorRegistry,
  route_bundle_factory: Callable[
    [Any, BuildDescriptor],
    RuntimeVehicleBundle,
  ],
  current_car_params: Any,
  current_bundle: RuntimeVehicleBundle,
  expected_dongle_id: str,
  abort_requested: Callable[[], bool] = lambda: False,
  segment_started: Callable[[RouteSegment, int, int], None] | None = None,
  segment_completed: Callable[[RouteSegment, int, int], None] | None = None,
  structural_first_segment_index: int | None = None,
  structural_last_segment_index: int | None = None,
  maximum_controls_witnesses: int = MAXIMUM_ROUTE_FRAMES,
  route_car_params_seed: bytes | None = None,
) -> PreparedRoute:
  """Validate one complete route before exposing any frame to the learner."""
  if (
    type(maximum_controls_witnesses) is not int
    or maximum_controls_witnesses <= 0
    or maximum_controls_witnesses > MAXIMUM_ROUTE_FRAMES
  ):
    raise ValueError("maximum controls witness bound is invalid")
  first_segment_index = (
    route.segments[0].index
    if structural_first_segment_index is None
    else structural_first_segment_index
  )
  last_segment_index = (
    route.segments[-1].index
    if structural_last_segment_index is None
    else structural_last_segment_index
  )
  if (
    type(first_segment_index) is not int
    or type(last_segment_index) is not int
    or first_segment_index < 0
    or last_segment_index < first_segment_index
    or any(
      segment.index < first_segment_index or segment.index > last_segment_index
      for segment in route.segments
    )
  ):
    raise ValueError("route structural boundary indices are invalid")
  route_records: dict[str, list[_TimedRouteRecord]] = {
    service: []
    for service in (
      *_SOURCE_SERVICES,
      *_BEHAVIOR_CONTEXT_SERVICES,
      *_CANONICAL_WITNESS_SERVICES,
    )
  }
  first_source_time: dict[str, int] = {}
  decoded_controls_count = 0
  if route_car_params_seed is not None and type(route_car_params_seed) is not bytes:
    raise ValueError("route CarParams seed must be immutable bytes")
  route_car_params_bytes: bytes | None = route_car_params_seed
  route_car_params: Any | None = None
  if route_car_params_bytes is not None:
    try:
      route_car_params = car_params_decoder(route_car_params_bytes)
    except BackfillError:
      raise
    except Exception as exc:
      raise RouteRejected(
        "car_params_decode_failed",
        "route CarParams seed could not be decoded",
      ) from exc
  route_descriptor: BuildDescriptor | None = None
  route_init_identity: tuple[object, ...] | None = None
  route_time_origin_mono_ns: int | None = None
  physical_compatibility_sha256: str | None = None
  last_service_time: dict[str, int] = {}
  extraction_digest = hashlib.sha256()
  source_order = 0

  segment_count = len(route.segments)
  for segment_position, segment in enumerate(route.segments, start=1):
    if abort_requested():
      raise BackfillError(
        "unexpected_error",
        "backfill aborted for onroad transition",
      )
    if segment_started is not None:
      segment_started(segment, segment_position, segment_count)
    segment_descriptor = -1
    try:
      segment_descriptor, opened_stat = _open_verified_route_segment(
        segment,
        abort_requested=abort_requested,
      )
      records = extract_segment_events(
        extractor_path,
        segment.path,
        abort_requested=abort_requested,
        extractor_fd=extractor_fd,
        segment_fd=segment_descriptor,
      )
      _verify_open_route_segment(
        segment,
        segment_descriptor,
        opened_stat,
        abort_requested=abort_requested,
      )
    finally:
      if segment_descriptor >= 0:
        os.close(segment_descriptor)

    sentinels = []
    segment_init_seen = False
    segment_payload_started = False
    segment_ended = False
    for selected_index, record in enumerate(records):
      if selected_index % 256 == 0:
        _abort_if_requested(
          abort_requested,
          "backfill aborted while decoding a route",
        )
      extraction_digest.update(struct.pack(
        "<IIQI",
        segment.index,
        record.which,
        record.mono_ns,
        len(record.encoded),
      ))
      extraction_digest.update(record.encoded)
      decoded = _decode_extracted_event(record, event_reader)
      which = decoded.which
      if (
        which in {"sentinel", "initData", "carParams"}
        and not decoded.valid
      ):
        raise RouteRejected(
          "invalid_provenance_event",
          "route structural/provenance event is invalid",
        )
      if which == "initData":
        if (
          selected_index != 0
          or segment_init_seen
          or segment_payload_started
          or segment_ended
        ):
          raise RouteRejected(
            "route_structure_mismatch",
            "segment InitData is missing, duplicated, or misordered",
          )
        segment_init_seen = True
        (
          superproject_commit,
          dirty,
          dongle_id,
          route_version,
        ) = decoded.payload
        if (
          type(route_version) is not str
          or not route_version
          or len(route_version) > 256
          or route_version.strip() != route_version
          or not route_version.isprintable()
        ):
          raise RouteRejected(
            "invalid_route_version",
            "route version is empty or malformed",
          )
        descriptor = descriptor_registry.resolve(superproject_commit)
        if descriptor is None:
          raise RouteRejected(
            "unreviewed_build",
            "route build has no reviewed provenance descriptor",
          )
        if dirty:
          raise RouteRejected(
            "dirty_build",
            "route was recorded by a dirty build",
          )
        if not dongle_id or dongle_id != expected_dongle_id:
          raise RouteRejected(
            "dongle_mismatch",
            "route belongs to another device",
          )
        identity = (
          superproject_commit,
          descriptor.opendbc_commit,
          descriptor.panda_commit,
          descriptor.log_schema_blob,
          route_version,
          dongle_id,
        )
        if route_init_identity is None:
          route_init_identity = identity
          route_descriptor = descriptor
          route_time_origin_mono_ns = decoded.mono_ns
        elif route_init_identity != identity:
          raise RouteRejected(
            "build_provenance_changed",
            "route segments disagree on build provenance",
          )
        continue

      if which == "sentinel":
        sentinel_type = decoded.payload
        if sentinel_type in {"startOfRoute", "startOfSegment"}:
          if (
            not segment_init_seen
            or segment_payload_started
            or segment_ended
          ):
            raise RouteRejected(
              "route_structure_mismatch",
              "segment start sentinel is missing, duplicated, or misordered",
            )
          segment_payload_started = True
        elif sentinel_type in {"endOfRoute", "endOfSegment"}:
          if not segment_payload_started or segment_ended:
            raise RouteRejected(
              "route_structure_mismatch",
              "segment end sentinel is duplicated or misordered",
            )
          segment_ended = True
        else:
          raise RouteRejected(
            "route_structure_mismatch",
            "segment sentinel type is unknown",
          )
        sentinels.append(sentinel_type)
        continue

      if not segment_payload_started or segment_ended:
        raise RouteRejected(
          "route_structure_mismatch",
          "learner payload appears outside segment sentinels",
        )

      if which == "carParams":
        canonical = decoded.payload
        if (
          route_car_params_bytes is not None
          and route_car_params_bytes != canonical
        ):
          raise RouteRejected(
            "car_params_changed",
            "route contains inconsistent serialized CarParams",
          )
        if route_car_params_bytes is None:
          route_car_params_bytes = canonical
          try:
            route_car_params = car_params_decoder(canonical)
          except BackfillError:
            raise
          except Exception as exc:
            raise RouteRejected(
              "car_params_decode_failed",
              "route CarParams could not be decoded",
            ) from exc
      elif which in route_records:
        previous_service_time = last_service_time.get(which, 0)
        if (
          record.mono_ns < previous_service_time
          or (
            which == "controlsState"
            and record.mono_ns <= previous_service_time
          )
        ):
          raise RouteRejected(
            "service_time_regression",
            "measurement service timestamps move backwards",
          )
        last_service_time[which] = record.mono_ns
        route_records[which].append(_TimedRouteRecord(
          mono_ns=record.mono_ns,
          segment_index=segment.index,
          ordinal=record.ordinal,
          source_order=source_order,
          valid=decoded.valid,
          payload=decoded.payload,
        ))
        source_order += 1
        if which in _SOURCE_SERVICES:
          first_source_time.setdefault(which, record.mono_ns)
        if which == "controlsState":
          decoded_controls_count += 1
          if decoded_controls_count > maximum_controls_witnesses:
            raise RouteRejected(
              "route_too_large",
              "route exceeds bounded controls witness count",
            )

    expected_start = (
      "startOfRoute"
      if segment.index == first_segment_index
      else "startOfSegment"
    )
    expected_end = (
      "endOfRoute"
      if segment.index == last_segment_index
      else "endOfSegment"
    )
    if (
      not segment_init_seen
      or not segment_payload_started
      or not segment_ended
      or sentinels != [expected_start, expected_end]
    ):
      raise RouteRejected(
        "route_sentinel_mismatch",
        "full-rlog route is incomplete or has invalid sentinels",
      )

    if segment_completed is not None:
      segment_completed(segment, segment_position, segment_count)
  if (
    route_init_identity is None
    or route_descriptor is None
    or route_time_origin_mono_ns is None
    or route_car_params is None
    or route_car_params_bytes is None
  ):
    raise RouteRejected(
      "missing_route_provenance",
      "route lacks InitData or a consistent CarParams event",
    )
  try:
    route_bundle = route_bundle_factory(
      route_car_params,
      route_descriptor,
    )
    physical_compatibility_sha256 = _validate_route_bundle(
      route_car_params=route_car_params,
      route_bundle=route_bundle,
      current_car_params=current_car_params,
      current_bundle=current_bundle,
      descriptor=route_descriptor,
    )
  except (BackfillError, RouteRejected):
    raise
  except Exception as exc:
    raise RouteRejected(
      "route_runtime_unsupported",
      "route runtime compatibility could not be reconstructed",
    ) from exc

  try:
    from opendbc.car.vehicle_model import VehicleModel

    canonical_joins, pre_poll_dropped = _build_canonical_control_joins(
      route_name=route.route_name,
      controls=tuple(route_records["controlsState"]),
      polls=tuple(route_records["selfdriveState"]),
      car_states=tuple(route_records["carState"]),
      live_parameters=tuple(route_records["liveParameters"]),
      car_outputs=tuple(route_records["carOutput"]),
      car_controls=tuple(route_records["carControl"]),
      vehicle_model=VehicleModel(route_car_params),
    )
    all_frames = tuple(
      _measured_frame_from_join(join)
      for join in canonical_joins
    )
  except (BackfillError, RouteRejected):
    raise
  except Exception as exc:
    raise RouteRejected(
      "measurement_race_unreconstructable",
      "route canonical input race could not be reconstructed",
    ) from exc

  if set(first_source_time) != set(_SOURCE_SERVICES):
    raise RouteRejected(
      "missing_measurement_service",
      "route lacks a required measurement service",
    )
  # Gate the complete controls population. A late-starting or early-stopping
  # required source must not silently shrink the quality denominator.
  eligible_controls = tuple(
    join.witness.mono_ns for join in canonical_joins
  )
  unresolved_in_coverage = sum(
    join.curvature_unresolved or join.car_control is None
    for join in canonical_joins
  )
  gap_count = 0
  for left, right in zip(
    eligible_controls,
    eligible_controls[1:],
    strict=False,
  ):
    difference = right - left
    if difference > MAXIMUM_CONTROL_GAP_NS:
      gap_count += max(
        1,
        int(round(difference / NOMINAL_CONTROL_PERIOD_NS)) - 1,
      )
  defect_count = unresolved_in_coverage + gap_count
  if (
    not eligible_controls
    or defect_count / len(eligible_controls)
    > MAXIMUM_UNRESOLVED_FRACTION
  ):
    raise RouteRejected(
      "measurement_continuity_failed",
      "route has more than one percent unresolved/gapped witnesses",
    )

  (
    superproject_commit,
    opendbc_commit,
    panda_commit,
    log_schema_blob,
    version,
    dongle_id,
  ) = route_init_identity
  provenance = {
    "canonical_join_schema_version": CANONICAL_JOIN_SCHEMA_VERSION,
    "car_params_sha256": _sha256(route_car_params_bytes),
    "dongle_id_sha256": _sha256(str(dongle_id).encode("utf-8")),
    "extractor_schema_version": NATIVE_EXTRACTOR_SCHEMA_VERSION,
    "log_schema_blob": log_schema_blob,
    "opendbc_commit": opendbc_commit,
    "panda_commit": panda_commit,
    "physical_compatibility_sha256": physical_compatibility_sha256,
    "route_version": version,
    "selected_event_stream_sha256": extraction_digest.hexdigest(),
    "superproject_commit": superproject_commit,
  }
  route_evidence = _route_evidence_artifact(
    route=route,
    route_time_origin_mono_ns=route_time_origin_mono_ns,
    route_car_params_bytes=route_car_params_bytes,
    route_bundle=route_bundle,
    route_descriptor=route_descriptor,
    route_records=route_records,
    joins=canonical_joins,
    frames=all_frames,
    pre_poll_dropped=pre_poll_dropped,
    gap_count=gap_count,
    provenance=provenance,
  )
  return PreparedRoute(
    frames=all_frames,
    controls_witness_count=decoded_controls_count,
    unresolved_witness_count=(
      unresolved_in_coverage + len(pre_poll_dropped)
    ),
    gap_count=gap_count,
    provenance=provenance,
    route_evidence=route_evidence,
    pre_poll_dropped_count=len(pre_poll_dropped),
    behavior_eligible=route_evidence.source_identity.behavior_eligible,
    behavior_ineligible_reason=(
      route_evidence.source_identity.behavior_ineligible_reason
    ),
  )


def prepare_route(
  route: RouteCandidate,
  *,
  extractor_path: str | Path,
  event_reader: Callable[[bytes], AbstractContextManager[Any]],
  car_params_decoder: Callable[[bytes], Any],
  descriptor_registry: BuildDescriptorRegistry,
  route_bundle_factory: Callable[
    [Any, BuildDescriptor],
    RuntimeVehicleBundle,
  ],
  current_car_params: Any,
  current_bundle: RuntimeVehicleBundle,
  expected_dongle_id: str,
  expected_extractor_sha256: str | None = None,
  abort_requested: Callable[[], bool] = lambda: False,
  segment_started: Callable[[RouteSegment, int, int], None] | None = None,
  segment_completed: Callable[[RouteSegment, int, int], None] | None = None,
  structural_first_segment_index: int | None = None,
  structural_last_segment_index: int | None = None,
  maximum_controls_witnesses: int = MAXIMUM_ROUTE_FRAMES,
  route_car_params_seed: bytes | None = None,
) -> PreparedRoute:
  """Prepare one route with one hash-bound extractor inode held throughout."""
  extractor = open_verified_extractor(
    extractor_path,
    expected_sha256=expected_extractor_sha256,
    abort_requested=abort_requested,
  )
  try:
    return _prepare_route_with_extractor(
      route,
      extractor_path=extractor_path,
      extractor_fd=extractor.descriptor,
      event_reader=event_reader,
      car_params_decoder=car_params_decoder,
      descriptor_registry=descriptor_registry,
      route_bundle_factory=route_bundle_factory,
      current_car_params=current_car_params,
      current_bundle=current_bundle,
      expected_dongle_id=expected_dongle_id,
      abort_requested=abort_requested,
      segment_started=segment_started,
      segment_completed=segment_completed,
      structural_first_segment_index=structural_first_segment_index,
      structural_last_segment_index=structural_last_segment_index,
      maximum_controls_witnesses=maximum_controls_witnesses,
      route_car_params_seed=route_car_params_seed,
    )
  finally:
    try:
      verify_open_extractor(
        extractor,
        abort_requested=abort_requested,
      )
    finally:
      os.close(extractor.descriptor)


def _empty_ledger(runtime_identity_sha256: str) -> dict[str, object]:
  return {
    "entries": [],
    "runtime_identity_sha256": runtime_identity_sha256,
    "schema_version": BACKFILL_LEDGER_SCHEMA_VERSION,
    "watermark_route_counter": None,
  }


def validate_ledger(
  payload: object,
  *,
  runtime_identity_sha256: str,
) -> dict[str, object]:
  if (
    type(payload) is not dict
    or set(payload) != {
      "entries",
      "runtime_identity_sha256",
      "schema_version",
      "watermark_route_counter",
    }
    or payload["schema_version"] != BACKFILL_LEDGER_SCHEMA_VERSION
    or payload["runtime_identity_sha256"] != runtime_identity_sha256
    or type(payload["entries"]) is not list
  ):
    raise BackfillError(
      "backfill_untracked_evidence",
      "backfill ledger schema/runtime is incompatible",
    )
  watermark = payload["watermark_route_counter"]
  if watermark is not None and (
    type(watermark) is not int or watermark < 0
  ):
    raise BackfillError(
      "backfill_untracked_evidence",
      "backfill ledger watermark is invalid",
    )
  seen = set()
  maximum = None
  for entry in payload["entries"]:
    if type(entry) is not dict or set(entry) != {
      "accepted_sample_count",
      "controls_witness_count",
      "diagnostic",
      "disposition",
      "provenance",
      "rejected_sample_count",
      "route_counter",
      "route_identity_sha256",
      "route_evidence_control_witness_count",
      "route_evidence_event_locator_count",
      "route_evidence_model_publication_count",
      "route_evidence_sha256",
      "route_name",
      "segments",
      "unresolved_witness_count",
    }:
      raise BackfillError(
        "backfill_untracked_evidence",
        "backfill ledger entry keys are invalid",
      )
    route_name = entry["route_name"]
    if (
      type(route_name) is not str
      or _ROUTE_DIRECTORY_RE.fullmatch(f"{route_name}--0") is None
      or route_name in seen
      or entry["route_identity_sha256"] != route_identity_sha256(route_name)
      or type(entry["route_counter"]) is not int
      or entry["route_counter"] != int(route_name[:8], 16)
      or entry["disposition"] not in {
        "ingested",
        "rejected",
        "late_older_skipped",
      }
      or type(entry["segments"]) is not list
      or not entry["segments"]
      or type(entry["diagnostic"]) is not str
      or not entry["diagnostic"]
    ):
      raise BackfillError(
        "backfill_untracked_evidence",
        "backfill ledger route identity is invalid",
      )
    for name in (
      "accepted_sample_count",
      "controls_witness_count",
      "rejected_sample_count",
      "route_evidence_control_witness_count",
      "route_evidence_event_locator_count",
      "route_evidence_model_publication_count",
      "unresolved_witness_count",
    ):
      if type(entry[name]) is not int or entry[name] < 0:
        raise BackfillError(
          "backfill_untracked_evidence",
          "backfill ledger counters are invalid",
        )
    disposition = entry["disposition"]
    accepted = entry["accepted_sample_count"]
    controls = entry["controls_witness_count"]
    rejected = entry["rejected_sample_count"]
    unresolved = entry["unresolved_witness_count"]
    provenance = entry["provenance"]
    route_evidence_sha256 = entry["route_evidence_sha256"]
    if disposition == "ingested":
      if (
        entry["diagnostic"] != "ingested"
        or type(provenance) is not dict
        or set(provenance) != _LEDGER_PROVENANCE_KEYS
        or type(provenance["canonical_join_schema_version"]) is not int
        or provenance["canonical_join_schema_version"]
        != CANONICAL_JOIN_SCHEMA_VERSION
        or type(provenance["extractor_schema_version"]) is not int
        or provenance["extractor_schema_version"]
        != NATIVE_EXTRACTOR_SCHEMA_VERSION
        or any(
          type(provenance[name]) is not str
          or _FULL_COMMIT_RE.fullmatch(provenance[name]) is None
          for name in (
            "log_schema_blob",
            "opendbc_commit",
            "panda_commit",
            "superproject_commit",
          )
        )
        or any(
          type(provenance[name]) is not str
          or _SHA256_RE.fullmatch(provenance[name]) is None
          for name in (
            "car_params_sha256",
            "dongle_id_sha256",
            "physical_compatibility_sha256",
            "selected_event_stream_sha256",
          )
        )
        or type(provenance["route_version"]) is not str
        or not provenance["route_version"]
        or len(provenance["route_version"]) > 256
        or provenance["route_version"].strip()
        != provenance["route_version"]
        or not provenance["route_version"].isprintable()
        or controls <= 0
        or unresolved > controls
        or accepted > controls - unresolved
        or rejected < controls - accepted
        or (
          route_evidence_sha256 is not None
          and (
            type(route_evidence_sha256) is not str
            or _SHA256_RE.fullmatch(route_evidence_sha256) is None
          )
        )
      ):
        raise BackfillError(
          "backfill_untracked_evidence",
          "backfill ledger ingested result is incoherent",
        )
    elif (
      provenance is not None
      or accepted != 0
      or controls != 0
      or rejected != 0
      or unresolved != 0
      or route_evidence_sha256 is not None
      or entry["route_evidence_control_witness_count"] != 0
      or entry["route_evidence_event_locator_count"] != 0
      or entry["route_evidence_model_publication_count"] != 0
      or (
        disposition == "late_older_skipped"
        and entry["diagnostic"] != "late_older_skipped"
      )
      or (
        disposition == "rejected"
        and entry["diagnostic"] in {"ingested", "late_older_skipped"}
      )
    ):
      raise BackfillError(
        "backfill_untracked_evidence",
        "backfill ledger skipped/rejected result is incoherent",
      )
    if (
      disposition == "late_older_skipped"
      and (
        maximum is None
        or entry["route_counter"] > maximum
      )
    ):
      raise BackfillError(
        "backfill_untracked_evidence",
        "late route does not predate the durable watermark",
      )
    for index, segment in enumerate(entry["segments"]):
      if (
        type(segment) is not dict
        or set(segment) != {"index", "sha256", "size_bytes"}
        or segment["index"] != index
        or type(segment["sha256"]) is not str
        or _SHA256_RE.fullmatch(segment["sha256"]) is None
        or type(segment["size_bytes"]) is not int
        or segment["size_bytes"] <= 0
      ):
        raise BackfillError(
          "backfill_untracked_evidence",
          "backfill ledger segment identity is invalid",
        )
    seen.add(route_name)
    maximum = (
      entry["route_counter"]
      if maximum is None
      else max(maximum, entry["route_counter"])
    )
  if watermark != maximum:
    raise BackfillError(
      "backfill_untracked_evidence",
      "backfill ledger watermark does not match entries",
    )
  return payload


def load_ledger(
  artifact_paths: LearningArtifactPaths,
  *,
  runtime_identity_sha256: str,
) -> dict[str, object]:
  if not artifact_paths.backfill_pointer.is_file():
    legacy_evidence = artifact_paths.root / "evidence.json"
    legacy_manifest = artifact_paths.root / "manifest.json"
    if legacy_evidence.is_file() or legacy_manifest.is_file():
      raise BackfillError(
        "backfill_untracked_evidence",
        "legacy evidence has no exactly-once route ledger",
      )
    return _empty_ledger(runtime_identity_sha256)
  encoded = artifact_paths.backfill_ledger.read_bytes()
  try:
    payload = json.loads(encoded)
  except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise BackfillError(
      "backfill_untracked_evidence",
      "backfill ledger is not JSON",
    ) from exc
  if encoded != _canonical_json_bytes(payload):
    raise BackfillError(
      "backfill_untracked_evidence",
      "backfill ledger is not canonical",
    )
  return validate_ledger(
    payload,
    runtime_identity_sha256=runtime_identity_sha256,
  )


def _behavior_cohort_source_payload(
  artifact: RouteEvidenceArtifact,
) -> dict[str, object]:
  source = artifact.source_identity
  return {
    "controller_artifact_sha256": source.controller_artifact_sha256,
    "controller_source_kind": source.controller_source_kind,
    "evidence_schema_version": source.schema_versions.get("route_evidence"),
    "runtime_identity": source.runtime_identity,
    "source_opendbc_commit": source.source_opendbc_commit,
    "source_panda_commit": source.source_panda_commit,
    "source_superproject_commit": source.source_superproject_commit,
    "vehicle_identity": source.vehicle_identity,
  }


def _artifact_matches_ledger_entry(
  artifact: RouteEvidenceArtifact,
  entry: Mapping[str, object],
) -> bool:
  source = artifact.source_identity
  segments = entry["segments"]
  if type(segments) is not list:
    return False
  return (
    source.route_id == entry["route_name"]
    and source.route_segment_sha256
    == tuple(segment["sha256"] for segment in segments)
    and source.route_segment_size_bytes
    == tuple(segment["size_bytes"] for segment in segments)
    and source.controls_witness_count == entry["controls_witness_count"]
    and source.unresolved_witness_count
    == entry["unresolved_witness_count"]
    and artifact.sha256 == entry["route_evidence_sha256"]
    and len(artifact.model_publications)
    == entry["route_evidence_model_publication_count"]
    and len(artifact.control_witnesses)
    == entry["route_evidence_control_witness_count"]
    and len(artifact.event_locators)
    == entry["route_evidence_event_locator_count"]
  )


def select_homogeneous_behavior_cohort(
  *,
  ledger: dict[str, object],
  store: RouteEvidenceStore,
) -> BehaviorEvidenceCohortSelection:
  """Select the newest contiguous exact-source route population.

  This is intentionally stricter than "find enough good routes".  Starting
  from the newest normal ledger entry, every route must be accounted for
  until an older, eligible artifact proves an exact controller-source
  boundary.  Missing, rejected, corrupt, or behavior-ineligible evidence in
  that current-source population blocks qualification instead of being
  cherry-picked around.  Explicit late-older skips predate the append-only
  watermark and are the sole ignored entry class.
  """
  validated = validate_ledger(
    ledger,
    runtime_identity_sha256=str(ledger.get("runtime_identity_sha256", "")),
  )
  entries = sorted(
    (
      entry
      for entry in validated["entries"]
      if entry["disposition"] != "late_older_skipped"
    ),
    key=lambda entry: entry["route_counter"],
    reverse=True,
  )
  if not entries:
    return BehaviorEvidenceCohortSelection(
      status="empty",
      reason="no_ingested_routes",
      blocking_route_name=None,
      source_identity_sha256=None,
      artifacts=(),
    )

  selected: list[RouteEvidenceArtifact] = []
  source_payload: dict[str, object] | None = None
  source_identity: str | None = None
  for entry in entries:
    route_name = str(entry["route_name"])
    if entry["disposition"] != "ingested":
      return BehaviorEvidenceCohortSelection(
        status="blocked",
        reason=(
          "newest_route_rejected"
          if source_payload is None
          else "interleaved_route_rejected"
        ),
        blocking_route_name=route_name,
        source_identity_sha256=source_identity,
        artifacts=(),
      )
    evidence_sha256 = entry["route_evidence_sha256"]
    if type(evidence_sha256) is not str:
      return BehaviorEvidenceCohortSelection(
        status="blocked",
        reason="route_evidence_missing",
        blocking_route_name=route_name,
        source_identity_sha256=source_identity,
        artifacts=(),
      )
    try:
      artifact = store.load(evidence_sha256)
    except (OSError, RouteEvidenceError):
      return BehaviorEvidenceCohortSelection(
        status="blocked",
        reason="route_evidence_corrupt",
        blocking_route_name=route_name,
        source_identity_sha256=source_identity,
        artifacts=(),
      )
    if not _artifact_matches_ledger_entry(artifact, entry):
      return BehaviorEvidenceCohortSelection(
        status="blocked",
        reason="route_evidence_ledger_mismatch",
        blocking_route_name=route_name,
        source_identity_sha256=source_identity,
        artifacts=(),
      )
    if not artifact.source_identity.behavior_eligible:
      return BehaviorEvidenceCohortSelection(
        status="blocked",
        reason=f"route_evidence_ineligible:{artifact.source_identity.behavior_ineligible_reason}",
        blocking_route_name=route_name,
        source_identity_sha256=source_identity,
        artifacts=(),
      )
    artifact_source = _behavior_cohort_source_payload(artifact)
    artifact_source_identity = _sha256(_canonical_json_bytes(artifact_source))
    if source_payload is None:
      source_payload = artifact_source
      source_identity = artifact_source_identity
    elif artifact_source != source_payload:
      break
    selected.append(artifact)

  if not selected or source_identity is None:
    raise AssertionError("ready behavior cohort lost its source population")
  selected.reverse()
  return BehaviorEvidenceCohortSelection(
    status="ready",
    reason="ready",
    blocking_route_name=None,
    source_identity_sha256=source_identity,
    artifacts=tuple(selected),
  )


class ExclusiveBackfillWriter(AbstractContextManager["ExclusiveBackfillWriter"]):
  def __init__(self, artifact_root: Path) -> None:
    self.artifact_root = artifact_root
    self._stream: Any | None = None

  def __enter__(self) -> ExclusiveBackfillWriter:
    self.artifact_root.mkdir(parents=True, exist_ok=True)
    self._stream = (self.artifact_root / "backfill.lock").open("a+b")
    try:
      fcntl.flock(
        self._stream.fileno(),
        fcntl.LOCK_EX | fcntl.LOCK_NB,
      )
    except BlockingIOError as exc:
      self._stream.close()
      self._stream = None
      raise BackfillError(
        "unexpected_error",
        "another backfill writer owns this runtime",
      ) from exc
    return self

  def __exit__(self, *exc_info: object) -> None:
    if self._stream is not None:
      fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
      self._stream.close()
      self._stream = None

  @property
  def fileno(self) -> int:
    if self._stream is None:
      raise RuntimeError("backfill writer lock is not active")
    return int(self._stream.fileno())


def cleanup_stale_prepared_route_spools(artifact_root: Path) -> None:
  """Remove only abandoned four-lane scratch dirs under the writer lock."""
  removed = False
  try:
    entries = tuple(artifact_root.iterdir())
  except OSError as exc:
    raise BackfillError(
      "backfill_reader_unavailable",
      "prepared-route scratch inventory is unavailable",
    ) from exc
  for entry in entries:
    if _BACKFILL_SPOOL_DIRECTORY_RE.fullmatch(entry.name) is None:
      continue
    try:
      entry_stat = entry.lstat()
      if not stat.S_ISDIR(entry_stat.st_mode) or entry.is_symlink():
        raise BackfillError(
          "backfill_spool_invalid",
          "prepared-route scratch path has an unsafe type",
        )
      shutil.rmtree(entry)
      removed = True
    except BackfillError:
      raise
    except OSError as exc:
      raise BackfillError(
        "backfill_spool_invalid",
        "abandoned prepared-route scratch cleanup failed",
      ) from exc
  if removed:
    _fsync_directory(artifact_root)


def _write_fsynced(path: Path, encoded: bytes) -> None:
  with path.open("xb") as stream:
    stream.write(encoded)
    stream.flush()
    os.fsync(stream.fileno())


def _atomic_write_bytes(
  path: Path,
  encoded: bytes,
  *,
  abort_requested: Callable[[], bool],
) -> None:
  """Fsync a pointer temp, recheck offroad state, then atomically replace."""
  temporary_fd, temporary_name = tempfile.mkstemp(
    dir=path.parent,
    prefix=f".{path.name}.",
    suffix=".tmp",
  )
  try:
    with os.fdopen(temporary_fd, "wb") as temporary:
      temporary_fd = -1
      temporary.write(encoded)
      temporary.flush()
      os.fsync(temporary.fileno())
    _abort_if_requested(
      abort_requested,
      "onroad transition aborted backfill during CURRENT staging",
    )
    os.replace(temporary_name, path)
    _fsync_directory(path.parent)
  except BaseException:
    if temporary_fd >= 0:
      os.close(temporary_fd)
    try:
      os.unlink(temporary_name)
    except FileNotFoundError:
      pass
    raise


def _fsync_directory(path: Path) -> None:
  descriptor = os.open(
    path,
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
  )
  try:
    os.fsync(descriptor)
  finally:
    os.close(descriptor)


def publish_generation(
  *,
  artifact_paths: LearningArtifactPaths,
  runtime_identity_sha256: str,
  finalization: CalibrationLearningFinalization,
  ledger: dict[str, object],
  descriptor_registry_sha256: str,
  extractor_sha256: str,
  abort_requested: Callable[[], bool],
) -> tuple[str, str]:
  """Publish all artifacts as one immutable generation + atomic pointer."""
  _abort_if_requested(
    abort_requested,
    "onroad transition aborted backfill before staging",
  )
  ledger_bytes = _canonical_json_bytes(ledger)
  ledger_sha256 = _sha256(ledger_bytes)
  # Schema-8 separates the profile selected for behavior replay from the
  # optional learned candidate.  An all-seed qualification therefore still
  # has a selected profile even though it has no candidate change.  The
  # getattr fallback keeps older audit test doubles honest: their candidate
  # was also their only possible selection.
  selected_profile_json = getattr(
    finalization,
    "selected_profile_json",
    finalization.candidate_profile_json,
  )
  selected_profile_sha256 = getattr(
    finalization,
    "selected_profile_sha256",
    finalization.candidate_profile_sha256,
  )
  if (selected_profile_json is None) != (selected_profile_sha256 is None):
    raise BackfillError(
      "backfill_publish_failed",
      "selected profile JSON and identity are incomplete",
    )
  if finalization.all_nodes_qualified != (selected_profile_json is not None):
    raise BackfillError(
      "backfill_publish_failed",
      "fully qualified calibration must commit exactly one selected profile",
    )
  if (
    selected_profile_json is not None
    and _sha256(selected_profile_json) != selected_profile_sha256
  ):
    raise BackfillError(
      "backfill_publish_failed",
      "selected profile identity does not match its canonical bytes",
    )
  if (
    finalization.candidate_profile_json is not None
    and (
      finalization.candidate_profile_sha256 is None
      or _sha256(finalization.candidate_profile_json)
      != finalization.candidate_profile_sha256
      or finalization.candidate_profile_json != selected_profile_json
    )
  ):
    raise BackfillError(
      "backfill_publish_failed",
      "learned candidate is not the committed selected profile",
    )
  previous_generation = None
  if artifact_paths.backfill_pointer.is_file():
    previous_generation = json.loads(
      artifact_paths.backfill_pointer.read_bytes(),
    )["generation_sha256"]
  provenance_bytes = _canonical_json_bytes({
    "canonical_join_schema_version": CANONICAL_JOIN_SCHEMA_VERSION,
    "descriptor_registry_sha256": descriptor_registry_sha256,
    "extractor_sha256": extractor_sha256,
    "extractor_schema_version": NATIVE_EXTRACTOR_SCHEMA_VERSION,
    "ledger_sha256": ledger_sha256,
    "previous_generation_sha256": previous_generation,
    "runtime_identity_sha256": runtime_identity_sha256,
    "schema_version": BACKFILL_PROVENANCE_SCHEMA_VERSION,
    "source": "complete_full_rlog_only",
  })
  commit_bytes = _canonical_json_bytes({
    "candidate_profile_sha256": (
      finalization.candidate_profile_sha256
    ),
    "evidence_sha256": finalization.evidence_sha256,
    "ledger_sha256": ledger_sha256,
    "manifest_sha256": finalization.manifest_sha256,
    "provenance_sha256": _sha256(provenance_bytes),
    "runtime_identity_sha256": runtime_identity_sha256,
    "schema_version": BACKFILL_COMMIT_SCHEMA_VERSION,
    "selected_profile_sha256": selected_profile_sha256,
  })
  generation_sha256 = _sha256(commit_bytes)
  generations = artifact_paths.backfill_generations
  generations.mkdir(parents=True, exist_ok=True)
  staging = Path(tempfile.mkdtemp(
    dir=generations,
    prefix=".staging-",
  ))
  try:
    _write_fsynced(staging / "evidence.json", finalization.evidence_bytes)
    _write_fsynced(staging / "manifest.json", finalization.manifest_bytes)
    _write_fsynced(staging / "ledger.json", ledger_bytes)
    _write_fsynced(staging / "provenance.json", provenance_bytes)
    if selected_profile_json is not None:
      selected_profiles = staging / "selected_profiles"
      selected_profiles.mkdir()
      if selected_profile_sha256 is None:
        raise AssertionError("selected profile JSON lacks identity")
      _write_fsynced(
        selected_profiles / f"{selected_profile_sha256}.json",
        selected_profile_json,
      )
      _fsync_directory(selected_profiles)
    if finalization.candidate_profile_json is not None:
      candidates = staging / "candidates"
      candidates.mkdir()
      candidate_identity = finalization.candidate_profile_sha256
      if candidate_identity is None:
        raise AssertionError("candidate JSON lacks identity")
      _write_fsynced(
        candidates / f"{candidate_identity}.json",
        finalization.candidate_profile_json,
      )
      _fsync_directory(candidates)
    _write_fsynced(staging / "commit.json", commit_bytes)
    _fsync_directory(staging)
    _abort_if_requested(
      abort_requested,
      "onroad transition aborted backfill before publication",
    )
    generation = generations / generation_sha256
    try:
      os.rename(staging, generation)
    except OSError as exc:
      if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
        raise
      # Content-address collision is safe only when every committed byte is
      # already identical. The existing immutable generation wins.
      existing_commit = generation / "commit.json"
      try:
        generation_matches = (
          existing_commit.is_file()
          and existing_commit.read_bytes() == commit_bytes
          and (generation / "evidence.json").read_bytes()
          == finalization.evidence_bytes
          and (generation / "manifest.json").read_bytes()
          == finalization.manifest_bytes
          and (generation / "ledger.json").read_bytes() == ledger_bytes
          and (generation / "provenance.json").read_bytes()
          == provenance_bytes
        )
      except OSError:
        generation_matches = False
      if not generation_matches:
        raise BackfillError(
          "backfill_publish_failed",
          "generation identity collision is not byte-identical",
        ) from exc
      selected_directory = generation / "selected_profiles"
      expected_selected_names = (
        set()
        if selected_profile_sha256 is None
        else {f"{selected_profile_sha256}.json"}
      )
      try:
        observed_selected_names = (
          {path.name for path in selected_directory.iterdir()}
          if selected_directory.is_dir()
          and not selected_directory.is_symlink()
          else set()
        )
      except OSError:
        observed_selected_names = set()
      if observed_selected_names != expected_selected_names or (
        (selected_directory.exists() or selected_directory.is_symlink())
        and not expected_selected_names
      ):
        raise BackfillError(
          "backfill_publish_failed",
          "existing generation selected profile set is corrupt",
        ) from exc
      if selected_profile_sha256 is not None:
        try:
          selected_matches = (
            generation
            / "selected_profiles"
            / f"{selected_profile_sha256}.json"
          ).read_bytes() == selected_profile_json
        except OSError:
          selected_matches = False
        if not selected_matches:
          raise BackfillError(
            "backfill_publish_failed",
            "existing generation selected profile is corrupt",
          ) from exc
      candidate_identity = finalization.candidate_profile_sha256
      if candidate_identity is not None:
        try:
          candidate_matches = (
            generation
            / "candidates"
            / f"{candidate_identity}.json"
          ).read_bytes() == finalization.candidate_profile_json
        except OSError:
          candidate_matches = False
        if not candidate_matches:
          raise BackfillError(
            "backfill_publish_failed",
            "existing generation candidate is corrupt",
          ) from exc
      shutil.rmtree(staging)
    _fsync_directory(generations)
    _abort_if_requested(
      abort_requested,
      "onroad transition aborted backfill before CURRENT",
    )
    pointer_bytes = _canonical_json_bytes({
      "generation_sha256": generation_sha256,
      "schema_version": BACKFILL_POINTER_SCHEMA_VERSION,
    })
    _atomic_write_bytes(
      artifact_paths.backfill_pointer,
      pointer_bytes,
      abort_requested=abort_requested,
    )
    _fsync_directory(artifact_paths.root)
    return generation_sha256, ledger_sha256
  except BaseException as exc:
    if staging.exists():
      shutil.rmtree(staging)
    if isinstance(exc, BackfillError):
      raise
    if isinstance(exc, OSError):
      raise BackfillError(
        "backfill_publish_failed",
        "immutable backfill publication failed",
      ) from exc
    raise


def _prepared_route_evidence_summary(
  prepared: PreparedRoute | PreparedRouteSpool,
) -> tuple[str | None, int, int, int, str | None]:
  artifact = getattr(prepared, "route_evidence", None)
  if artifact is None:
    return None, 0, 0, 0, None
  manifest = artifact.manifest
  return (
    str(artifact.sha256),
    int(manifest["model_publication_count"]),
    int(manifest["control_witness_count"]),
    int(manifest["driving_event_locator_count"]),
    str(artifact.source_key),
  )


def replay_routes(
  *,
  runtime: PersistentLearningRuntime,
  routes: tuple[RouteCandidate, ...],
  prepare: Callable[
    [RouteCandidate],
    PreparedRoute | PreparedRouteSpool,
  ],
  abort_requested: Callable[[], bool] = lambda: False,
  route_completed: Callable[
    [RouteCandidate, int, int],
    None,
  ] | None = None,
  route_applying: Callable[[RouteCandidate], None] | None = None,
) -> ReplayPass:
  results = []
  accepted_total = 0
  rejected_total = 0
  for route in routes:
    prepared: PreparedRoute | PreparedRouteSpool | None = None
    _abort_if_requested(
      abort_requested,
      "backfill aborted before replaying a route",
    )
    try:
      prepared = prepare(route)
    except RouteRejected as exc:
      results.append(ReplayResult(
        route=route,
        disposition="rejected",
        diagnostic=exc.reason,
        provenance=None,
        accepted_sample_count=0,
        rejected_sample_count=0,
        controls_witness_count=0,
        unresolved_witness_count=0,
      ))
      if route_completed is not None:
        route_completed(route, accepted_total, rejected_total)
      continue
    try:
      before_accepted = runtime.coordinator.accepted_sample_count
      if route_applying is not None:
        route_applying(route)
      route_evidence = getattr(prepared, "route_evidence", None)
      route_content_sha256 = (
        None
        if route_evidence is None
        else str(route_evidence.sha256)
      )
      runtime.transition_onroad(
        route.display_identity,
        route_content_sha256,
      )
      frames = (
        prepared.frames
        if isinstance(prepared, PreparedRoute)
        else prepared.iter_frames()
      )
      for frame_index, frame in enumerate(frames):
        if frame_index % 256 == 0:
          _abort_if_requested(
            abort_requested,
            "backfill aborted while replaying route frames",
          )
        runtime.ingest(frame)
      _abort_if_requested(
        abort_requested,
        "backfill aborted after replaying route frames",
      )
      runtime.transition_offroad_without_persist()
      accepted = (
        runtime.coordinator.accepted_sample_count - before_accepted
      )
      # Every recorded witness is either accepted or rejected exactly once;
      # startup-prefix witnesses have no frame to ingest but still belong to
      # the route's reported denominator. Gaps remain explicit missing-frame
      # defects in addition to the recorded-witness population.
      rejected = (
        prepared.controls_witness_count
        - accepted
        + prepared.gap_count
      )
      (
        route_evidence_sha256,
        route_evidence_model_count,
        route_evidence_control_count,
        route_evidence_event_count,
        route_evidence_source_key,
      ) = _prepared_route_evidence_summary(prepared)
      accepted_total += accepted
      rejected_total += rejected
      results.append(ReplayResult(
        route=route,
        disposition="ingested",
        diagnostic="ingested",
        provenance=prepared.provenance,
        accepted_sample_count=accepted,
        rejected_sample_count=rejected,
        controls_witness_count=prepared.controls_witness_count,
        unresolved_witness_count=prepared.unresolved_witness_count,
        route_evidence_sha256=route_evidence_sha256,
        route_evidence_model_publication_count=(
          route_evidence_model_count
        ),
        route_evidence_control_witness_count=(
          route_evidence_control_count
        ),
        route_evidence_event_locator_count=route_evidence_event_count,
        route_evidence_source_key=route_evidence_source_key,
      ))
      if route_completed is not None:
        route_completed(route, accepted_total, rejected_total)
    except SpoolFormatError as exc:
      raise BackfillError(
        "backfill_spool_invalid",
        f"prepared route spool is invalid: {exc}",
      ) from exc
    finally:
      if isinstance(prepared, PreparedRouteSpool):
        try:
          prepared.cleanup()
        except SpoolFormatError as exc:
          raise BackfillError(
            "backfill_spool_invalid",
            f"prepared route spool cleanup failed: {exc}",
          ) from exc
  return ReplayPass(
    finalization=runtime.coordinator.finalize(),
    results=tuple(results),
    accepted_sample_count=accepted_total,
    rejected_sample_count=rejected_total,
  )


def verify_replay_passes(first: ReplayPass, second: ReplayPass) -> None:
  if (
    first.finalization.evidence_bytes
    != second.finalization.evidence_bytes
    or first.finalization.manifest_bytes
    != second.finalization.manifest_bytes
    or getattr(
      first.finalization,
      "selected_profile_json",
      first.finalization.candidate_profile_json,
    )
    != getattr(
      second.finalization,
      "selected_profile_json",
      second.finalization.candidate_profile_json,
    )
    or getattr(
      first.finalization,
      "selected_profile_sha256",
      first.finalization.candidate_profile_sha256,
    )
    != getattr(
      second.finalization,
      "selected_profile_sha256",
      second.finalization.candidate_profile_sha256,
    )
    or first.finalization.candidate_profile_json
    != second.finalization.candidate_profile_json
    or tuple(result.ledger_entry() for result in first.results)
    != tuple(result.ledger_entry() for result in second.results)
    or first.accepted_sample_count != second.accepted_sample_count
    or first.rejected_sample_count != second.rejected_sample_count
  ):
    raise BackfillError(
      "backfill_nondeterministic",
      "independent historical replays were not byte-identical",
    )


def _route_evidence_staging_path(
  root: Path,
  authority_index: int,
  sha256: str,
) -> Path:
  if authority_index not in (1, 2) or _SHA256_RE.fullmatch(sha256) is None:
    raise BackfillError(
      "backfill_route_incompatible",
      "route-evidence staging identity is invalid",
    )
  return root / ".route-evidence-staging-v2" / f"authority-{authority_index}" / f"{sha256}.route-evidence"


def _stage_route_evidence(
  *,
  root: Path,
  authority_index: int,
  artifact: RouteEvidenceArtifact,
) -> None:
  path = _route_evidence_staging_path(root, authority_index, artifact.sha256)
  path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
  if path.exists():
    digest = hashlib.sha256()
    try:
      with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
          digest.update(block)
    except OSError as error:
      raise BackfillError(
        "backfill_nondeterministic",
        "route-evidence staging object cannot be inspected",
      ) from error
    if (
      path.is_symlink()
      or not path.is_file()
      or digest.hexdigest() != artifact.sha256
    ):
      raise BackfillError(
        "backfill_nondeterministic",
        "route-evidence staging object is not immutable",
      )
    return
  descriptor, temporary_name = tempfile.mkstemp(
    dir=path.parent, prefix=f".{path.name}.", suffix=".partial",
  )
  temporary = Path(temporary_name)
  try:
    os.fchmod(descriptor, 0o600)
    view = memoryview(artifact.canonical_bytes)
    while view:
      count = os.write(descriptor, view)
      if count <= 0:
        raise OSError("short route-evidence staging write")
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


def _stage_route_evidence_spool(
  *,
  root: Path,
  authority_index: int,
  spool: PreparedRouteSpool,
) -> None:
  """Stage a complete remote artifact with constant-size copy buffers."""
  artifact = spool.route_evidence
  path = _route_evidence_staging_path(root, authority_index, artifact.sha256)
  path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
  source = spool.canonical_path
  if path.exists():
    digest = hashlib.sha256()
    with path.open("rb") as stream:
      while block := stream.read(1024 * 1024):
        digest.update(block)
    if path.is_symlink() or not path.is_file() or digest.hexdigest() != artifact.sha256:
      raise BackfillError(
        "backfill_nondeterministic",
        "route-evidence staging object is not immutable",
      )
    return
  source_info = source.lstat()
  expected_source = (
    artifact.st_dev, artifact.st_ino, artifact.st_size,
    artifact.st_mtime_ns, artifact.st_ctime_ns,
  )
  if (
    source_info.st_dev, source_info.st_ino, source_info.st_size,
    source_info.st_mtime_ns, source_info.st_ctime_ns,
  ) != expected_source or source.is_symlink():
    raise BackfillError(
      "backfill_nondeterministic",
      "remote route-evidence source changed before staging",
    )
  # Do not hard-link the downloaded spool: link/unlink changes inode ctime and
  # would invalidate the held PreparedRouteSpool identity before its physical
  # plane is streamed.  A bounded-buffer copy keeps the source immutable and
  # preserves the exact A/A application contract.
  descriptor, temporary_name = tempfile.mkstemp(
    dir=path.parent, prefix=f".{path.name}.", suffix=".partial",
  )
  temporary = Path(temporary_name)
  source_descriptor = -1
  try:
    os.fchmod(descriptor, 0o600)
    source_descriptor = os.open(
      source,
      os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    opened_source = os.fstat(source_descriptor)
    if (
      opened_source.st_dev, opened_source.st_ino, opened_source.st_size,
      opened_source.st_mtime_ns, opened_source.st_ctime_ns,
    ) != expected_source:
      raise BackfillError(
        "backfill_nondeterministic",
        "remote route-evidence source changed during staging",
      )
    while block := os.read(source_descriptor, 1024 * 1024):
      view = memoryview(block)
      while view:
        count = os.write(descriptor, view)
        if count <= 0:
          raise OSError("short route-evidence staging write")
        view = view[count:]
    after_source = os.fstat(source_descriptor)
    if (
      after_source.st_dev, after_source.st_ino, after_source.st_size,
      after_source.st_mtime_ns, after_source.st_ctime_ns,
    ) != expected_source:
      raise BackfillError(
        "backfill_nondeterministic",
        "remote route-evidence source changed during staging",
      )
    os.fsync(descriptor)
    os.close(descriptor)
    descriptor = -1
    os.replace(temporary, path)
    _fsync_directory(path.parent)
  finally:
    if source_descriptor >= 0:
      os.close(source_descriptor)
    if descriptor >= 0:
      os.close(descriptor)
    try:
      temporary.unlink()
    except FileNotFoundError:
      pass


def _publish_route_evidence_after_aa(
  *,
  root: Path,
  first: ReplayPass,
  second: ReplayPass,
) -> None:
  """Publish complete route artifacts only after replay-level A/A passes."""
  store = RouteEvidenceStore(root / "route_evidence_v2")
  second_by_route = {
    result.route.route_name: result
    for result in second.results
  }
  for first_result in first.results:
    if first_result.disposition != "ingested":
      continue
    second_result = second_by_route.get(first_result.route.route_name)
    if (
      second_result is None
      or second_result.disposition != "ingested"
      or first_result.route_evidence_sha256 is None
      or second_result.route_evidence_sha256 != first_result.route_evidence_sha256
      or first_result.route_evidence_source_key is None
      or second_result.route_evidence_source_key
      != first_result.route_evidence_source_key
    ):
      raise BackfillError(
        "backfill_nondeterministic",
        "route-evidence A/A identities disagree",
      )
    sha = first_result.route_evidence_sha256
    first_path = _route_evidence_staging_path(root, 1, sha)
    second_path = _route_evidence_staging_path(root, 2, sha)
    try:
      store.publish_files(
        first_path,
        second_path,
        sha256=sha,
        source_key=first_result.route_evidence_source_key,
      )
      # The authority files are transaction scratch, not a third durable copy
      # of every route.  Remove them only after the content-addressed store has
      # fsynced its exact A/A result.
      for path in (first_path, second_path):
        path.unlink()
        _fsync_directory(path.parent)
    except (OSError, RouteEvidenceError) as error:
      raise BackfillError(
        "backfill_nondeterministic",
        "route-evidence A/A bytes are missing or disagree",
      ) from error
  staging_root = root / ".route-evidence-staging-v2"
  try:
    for authority_index in (1, 2):
      (staging_root / f"authority-{authority_index}").rmdir()
    staging_root.rmdir()
    _fsync_directory(root)
  except FileNotFoundError:
    pass
  except OSError as error:
    # A non-empty directory means an unaccounted authority artifact exists;
    # silently retaining it could mask a later route-population mismatch.
    raise BackfillError(
      "backfill_nondeterministic",
      "route-evidence authority staging was not fully consumed",
    ) from error


_FORK_CHILD_ONLY_IPC_FDS: tuple[int, ...] = ()


def _forked_task_entry(
  connection: Any,
  task: Callable[[Callable[[], bool]], object],
  cancel_requested: Any,
  start_requested: Any,
  transaction_abort_requested: Callable[[], bool],
  inherited_close_fds: tuple[int, ...],
  expected_parent_pid: int,
  isolate_process_group: bool,
  task_description: str,
) -> None:
  """Run one cancellable task without inheriting another worker's IPC."""

  global _FORK_CHILD_ONLY_IPC_FDS

  def parent_lost() -> bool:
    return os.getppid() != expected_parent_pid

  def worker_abort_requested() -> bool:
    return (
      bool(cancel_requested.is_set())
      or parent_lost()
      or bool(transaction_abort_requested())
    )

  try:
    for descriptor in inherited_close_fds:
      os.close(descriptor)
    # These descriptors belonged to an enclosing worker in the address space
    # from which this process was forked. They are closed here and must not be
    # carried into a still-deeper fork after their descriptor numbers can be
    # reused. Only this worker's own result channel is registered below.
    _FORK_CHILD_ONLY_IPC_FDS = ()
    if isolate_process_group:
      os.setsid()
    connection.send((
      "ready",
      os.getpid(),
      os.getsid(0),
      os.getpgrp(),
    ))
    # Do not spawn the native extractor until the parent knows this PID owns
    # the intended process group. This closes the startup cancellation race in
    # which only the Python worker, but not a new grandchild, could be killed.
    while not start_requested.wait(timeout=0.05):
      if worker_abort_requested():
        raise BackfillError(
          "unexpected_error",
          f"{task_description} worker aborted before startup acknowledgement",
        )
    if worker_abort_requested():
      raise BackfillError(
        "unexpected_error",
        f"{task_description} worker aborted at startup acknowledgement",
      )
    _FORK_CHILD_ONLY_IPC_FDS = (connection.fileno(),)
    try:
      result = task(worker_abort_requested)
      connection.send(("ok", result))
    finally:
      _FORK_CHILD_ONLY_IPC_FDS = ()
  except BackfillError as exc:
    connection.send(("backfill_error", exc.diagnostic, str(exc)))
  except BaseException as exc:
    connection.send((
      "unexpected_error",
      type(exc).__name__,
      str(exc),
    ))
  finally:
    connection.close()


def _forked_replay_entry(
  connection: Any,
  replay: Callable[[Callable[[], bool]], ReplayPass],
  cancel_requested: Any,
  start_requested: Any,
  transaction_abort_requested: Callable[[], bool],
  inherited_close_fds: tuple[int, ...],
  expected_parent_pid: int,
) -> None:
  """Backward-compatible entry point for one isolated replay pass."""

  _forked_task_entry(
    connection,
    replay,
    cancel_requested,
    start_requested,
    transaction_abort_requested,
    inherited_close_fds,
    expected_parent_pid,
    True,
    "verification replay",
  )


class _ForkedTaskWorker:
  """One cancellable fork task with explicit process-group ownership."""

  def __init__(
    self,
    *,
    task: Callable[[Callable[[], bool]], object],
    expected_result_type: type[Any],
    abort_requested: Callable[[], bool],
    inherited_close_fds: tuple[int, ...] = (),
    isolate_process_group: bool = True,
    process_name: str = "blatv2-task-worker",
    task_description: str = "background task",
  ) -> None:
    current_process = multiprocessing.current_process()
    if current_process.daemon:
      raise BackfillError(
        "unexpected_error",
        f"{task_description} worker cannot fork from a daemon process",
      )
    if threading.active_count() != 1:
      raise BackfillError(
        "unexpected_error",
        f"{task_description} worker cannot fork from a multithreaded process",
      )
    if not isinstance(expected_result_type, type):
      raise TypeError("expected_result_type must be a type")
    try:
      context = multiprocessing.get_context("fork")
    except ValueError as exc:
      raise BackfillError(
        "backfill_reader_unavailable",
        f"{task_description} requires the Linux fork process context",
      ) from exc
    receive = None
    send = None
    close_fds = tuple(dict.fromkeys((
      *(int(descriptor) for descriptor in inherited_close_fds),
      *_FORK_CHILD_ONLY_IPC_FDS,
    )))
    expected_session_id = os.getsid(0)
    expected_process_group_id = os.getpgrp()
    try:
      receive, send = context.Pipe(duplex=False)
      cancel_requested = context.Event()
      start_requested = context.Event()
      process = context.Process(
        target=_forked_task_entry,
        args=(
          send,
          task,
          cancel_requested,
          start_requested,
          abort_requested,
          close_fds,
          os.getpid(),
          isolate_process_group,
          task_description,
        ),
        name=process_name,
      )
    except BaseException as exc:
      if receive is not None:
        receive.close()
      if send is not None:
        send.close()
      raise BackfillError(
        "unexpected_error",
        f"{task_description} worker resources could not be created",
      ) from exc
    self._receive = receive
    self._cancel_requested = cancel_requested
    self._start_requested = start_requested
    self._abort_requested = abort_requested
    self._expected_result_type = expected_result_type
    self._expected_session_id = expected_session_id
    self._expected_process_group_id = expected_process_group_id
    self._isolate_process_group = isolate_process_group
    self._task_description = task_description
    self._closed = False
    self._started = False
    self._group_ready = False
    self._process = process
    try:
      self._process.start()
      self._started = True
      send.close()
      deadline = time.monotonic() + REPLAY_WORKER_STARTUP_TIMEOUT_S
      while not self._receive.poll(0.1):
        _abort_if_requested(
          self._abort_requested,
          f"backfill aborted while starting {self._task_description}",
        )
        if not self._process.is_alive():
          raise BackfillError(
            "unexpected_error",
            f"{self._task_description} worker exited during startup",
          )
        if time.monotonic() >= deadline:
          raise BackfillError(
            "unexpected_error",
            f"{self._task_description} worker startup timed out",
          )
      try:
        ready = self._receive.recv()
      except (EOFError, OSError) as exc:
        raise BackfillError(
          "unexpected_error",
          f"{self._task_description} worker startup channel failed",
        ) from exc
      if (
        type(ready) is not tuple
        or len(ready) != 4
        or ready[0] != "ready"
        or ready[1] != self._process.pid
        or type(ready[2]) is not int
        or type(ready[3]) is not int
      ):
        raise BackfillError(
          "unexpected_error",
          f"{self._task_description} worker returned an invalid startup identity",
        )
      expected_isolation = (
        ready[2] == self._process.pid
        and ready[3] == self._process.pid
        if self._isolate_process_group
        else (
          ready[2] == self._expected_session_id
          and ready[3] == self._expected_process_group_id
        )
      )
      if not expected_isolation:
        raise BackfillError(
          "unexpected_error",
          f"{self._task_description} worker did not establish its expected process-group isolation",
        )
      self._group_ready = self._isolate_process_group
      self._start_requested.set()
    except BaseException as exc:
      try:
        send.close()
      except OSError:
        pass
      try:
        self._stop()
      except BaseException as cleanup_exc:
        self._receive.close()
        self._closed = True
        raise BackfillError(
          "unexpected_error",
          f"{self._task_description} worker failed startup cleanup",
        ) from cleanup_exc
      self._receive.close()
      self._closed = True
      if isinstance(exc, BackfillError):
        raise
      raise BackfillError(
        "unexpected_error",
        f"{self._task_description} worker could not start",
      ) from exc

  def _signal_worker(self, signal_number: int) -> None:
    if not self._started:
      return
    if self._isolate_process_group and self._group_ready:
      try:
        os.killpg(self._process.pid, signal_number)
      except ProcessLookupError:
        pass
      return
    if not self._process.is_alive():
      return
    if signal_number == signal.SIGTERM:
      self._process.terminate()
    else:
      self._process.kill()

  def _stop(self) -> None:
    self._cancel_requested.set()
    if not self._started:
      return
    self._process.join(timeout=REPLAY_WORKER_COOPERATIVE_STOP_S)
    # Isolated workers own the whole process group, including their native
    # extractor. A nested worker inherits its parent's group and must signal
    # only its own Python process; killing that group would kill its parent.
    self._signal_worker(signal.SIGTERM)
    self._process.join(timeout=REPLAY_WORKER_SIGNAL_STOP_S)
    self._signal_worker(signal.SIGKILL)
    self._process.join(timeout=REPLAY_WORKER_SIGNAL_STOP_S)
    if self._process.is_alive():
      raise BackfillError(
        "unexpected_error",
        f"{self._task_description} worker could not be reaped",
      )

  def cancel(self) -> None:
    if self._closed:
      return
    self._stop()
    self._receive.close()
    self._closed = True

  def result(self) -> object:
    if self._closed:
      raise RuntimeError(
        f"{self._task_description} worker result was already consumed",
      )
    try:
      while not self._receive.poll(0.1):
        _abort_if_requested(
          self._abort_requested,
          f"backfill aborted while waiting for {self._task_description}",
        )
        if not self._process.is_alive():
          raise BackfillError(
            "unexpected_error",
            f"{self._task_description} worker exited without a result",
          )
      try:
        payload = self._receive.recv()
      except (EOFError, OSError) as exc:
        raise BackfillError(
          "unexpected_error",
          f"{self._task_description} worker result channel failed",
        ) from exc
      self._process.join(timeout=REPLAY_WORKER_SIGNAL_STOP_S)
      if self._process.is_alive():
        raise BackfillError(
          "unexpected_error",
          f"{self._task_description} worker did not exit after its result",
        )
      if self._process.exitcode != 0:
        raise BackfillError(
          "unexpected_error",
          f"{self._task_description} worker exited abnormally",
        )
      if (
        type(payload) is not tuple
        or not payload
        or type(payload[0]) is not str
      ):
        raise BackfillError(
          "unexpected_error",
          f"{self._task_description} worker returned a malformed result",
        )
      if payload[0] == "ok" and len(payload) == 2:
        result = payload[1]
        if not isinstance(result, self._expected_result_type):
          raise BackfillError(
            "unexpected_error",
            f"{self._task_description} worker result has the wrong type",
          )
        return result
      if (
        payload[0] == "backfill_error"
        and len(payload) == 3
        and type(payload[1]) is str
        and type(payload[2]) is str
      ):
        raise BackfillError(payload[1], payload[2])
      if (
        payload[0] == "unexpected_error"
        and len(payload) == 3
        and type(payload[1]) is str
        and type(payload[2]) is str
      ):
        raise BackfillError(
          "unexpected_error",
          f"{self._task_description} worker failed: {payload[1]}: {payload[2]}",
        )
      raise BackfillError(
        "unexpected_error",
        f"{self._task_description} worker returned an unknown result",
      )
    except BaseException:
      self._stop()
      raise
    finally:
      self._receive.close()
      self._closed = True


class _ForkedReplayWorker(_ForkedTaskWorker):
  """Backward-compatible isolated verification replay worker."""

  def __init__(
    self,
    *,
    replay: Callable[[Callable[[], bool]], ReplayPass],
    abort_requested: Callable[[], bool],
    inherited_close_fds: tuple[int, ...] = (),
  ) -> None:
    super().__init__(
      task=replay,
      expected_result_type=ReplayPass,
      abort_requested=abort_requested,
      inherited_close_fds=inherited_close_fds,
      isolate_process_group=True,
      process_name="blatv2-replay-2",
      task_description="verification replay",
    )

  def result(self) -> ReplayPass:
    result = super().result()
    if not isinstance(result, ReplayPass):
      raise AssertionError("generic replay worker type check was bypassed")
    return result


@dataclass(frozen=True, slots=True)
class _RoutePreparationOutcome:
  """Small IPC result for one independently prepared route."""

  route_name: str
  descriptor: PreparedRouteSpoolDescriptor | None
  rejection_reason: str | None
  rejection_message: str | None


def _prepare_route_to_spool(
  *,
  route: RouteCandidate,
  prepare: Callable[
    [RouteCandidate, Callable[[], bool]],
    PreparedRoute,
  ],
  spool_directory: Path,
  abort_requested: Callable[[], bool],
) -> _RoutePreparationOutcome:
  """Prepare one route completely and expose only its bounded descriptor."""
  try:
    prepared = prepare(route, abort_requested)
  except RouteRejected as exc:
    return _RoutePreparationOutcome(
      route_name=route.route_name,
      descriptor=None,
      rejection_reason=exc.reason,
      rejection_message=str(exc),
    )
  try:
    descriptor = write_prepared_route_spool(
      spool_directory,
      route.route_name,
      prepared.frames,
      controls_witness_count=prepared.controls_witness_count,
      unresolved_witness_count=prepared.unresolved_witness_count,
      gap_count=prepared.gap_count,
      provenance=prepared.provenance,
      max_frames=MAXIMUM_ROUTE_FRAMES,
      abort_requested=abort_requested,
      route_evidence=prepared.route_evidence,
    )
  except SpoolFormatError as exc:
    if abort_requested():
      raise BackfillError(
        "unexpected_error",
        "backfill aborted while writing a prepared route spool",
      ) from exc
    raise BackfillError(
      "backfill_spool_invalid",
      f"prepared route spool could not be encoded: {exc}",
    ) from exc
  except OSError as exc:
    raise BackfillError(
      "backfill_reader_unavailable",
      "prepared route spool could not be written",
    ) from exc
  return _RoutePreparationOutcome(
    route_name=route.route_name,
    descriptor=descriptor,
    rejection_reason=None,
    rejection_message=None,
  )


class _PrefetchingRoutePreparer:
  """One-route-ahead private preparation lane for one replay authority.

  The first route is prepared by the authority itself. While that route is
  applied, one helper prepares the next route into a fixed-record scratch
  spool. No prepared bytes cross authorities and application order remains
  canonical. A single-route operation starts no helper at all.
  """

  def __init__(
    self,
    *,
    authority_index: int,
    routes: tuple[RouteCandidate, ...],
    local_prepare: Callable[[RouteCandidate], PreparedRoute],
    helper_prepare: Callable[
      [RouteCandidate, Callable[[], bool]],
      PreparedRoute,
    ],
    scratch_parent: Path,
    abort_requested: Callable[[], bool],
    helper_transaction_abort_requested: Callable[[], bool] | None = None,
    inherited_close_fds: tuple[int, ...] = (),
    prefetched_route_ready: Callable[[RouteCandidate], None] | None = None,
  ) -> None:
    if authority_index not in (1, 2):
      raise ValueError("preparation authority index must be one or two")
    self.authority_index = authority_index
    self._routes = routes
    self._local_prepare = local_prepare
    self._helper_prepare = helper_prepare
    self._scratch_parent = scratch_parent
    self._abort_requested = abort_requested
    self._helper_transaction_abort_requested = (
      abort_requested
      if helper_transaction_abort_requested is None
      else helper_transaction_abort_requested
    )
    self._inherited_close_fds = inherited_close_fds
    self._prefetched_route_ready = prefetched_route_ready
    self._position = 0
    self._pending_route: RouteCandidate | None = None
    self._pending_worker: _ForkedTaskWorker | None = None
    self._scratch_directory: Path | None = None
    self._closed = False

  def _ensure_scratch_directory(self) -> Path:
    if self._scratch_directory is None:
      try:
        self._scratch_directory = Path(tempfile.mkdtemp(
          dir=self._scratch_parent,
          prefix=(
            f"{BACKFILL_SPOOL_DIRECTORY_PREFIX}{self.authority_index}-"
          ),
        ))
      except OSError as exc:
        raise BackfillError(
          "backfill_reader_unavailable",
          "private prepared-route scratch directory is unavailable",
        ) from exc
    return self._scratch_directory

  def _start_helper(self, route: RouteCandidate) -> None:
    if self._pending_worker is not None or self._pending_route is not None:
      raise RuntimeError("prepared-route helper already owns a route")
    spool_directory = self._ensure_scratch_directory()
    prepare = self._helper_prepare

    def task(
      worker_abort_requested: Callable[[], bool],
    ) -> _RoutePreparationOutcome:
      return _prepare_route_to_spool(
        route=route,
        prepare=prepare,
        spool_directory=spool_directory,
        abort_requested=worker_abort_requested,
      )

    # Authority 1's helper owns an isolated group so parent cancellation also
    # kills its native extractor. Authority 2's helper stays in the isolated
    # verification group; the root worker's killpg then reaps both together.
    self._pending_worker = _ForkedTaskWorker(
      task=task,
      expected_result_type=_RoutePreparationOutcome,
      abort_requested=self._helper_transaction_abort_requested,
      inherited_close_fds=self._inherited_close_fds,
      isolate_process_group=self.authority_index == 1,
      process_name=f"blatv2-prepare-{self.authority_index}",
      task_description=(
        f"replay authority {self.authority_index} route preparation"
      ),
    )
    self._pending_route = route

  def _receive_prefetched(
    self,
    route: RouteCandidate,
  ) -> PreparedRouteSpool:
    worker = self._pending_worker
    if worker is None or self._pending_route != route:
      raise RuntimeError("prepared-route helper result is out of order")
    try:
      outcome = worker.result()
    finally:
      self._pending_worker = None
      self._pending_route = None
    if not isinstance(outcome, _RoutePreparationOutcome):
      raise AssertionError("generic preparation worker type check was bypassed")
    if outcome.route_name != route.route_name:
      raise BackfillError(
        "backfill_spool_invalid",
        "prepared route helper returned a different route",
      )
    if self._prefetched_route_ready is not None:
      self._prefetched_route_ready(route)
    if outcome.rejection_reason is not None:
      if (
        outcome.descriptor is not None
        or outcome.rejection_message is None
      ):
        raise BackfillError(
          "backfill_spool_invalid",
          "prepared route rejection result is malformed",
        )
      raise RouteRejected(
        outcome.rejection_reason,
        outcome.rejection_message,
      )
    if (
      outcome.descriptor is None
      or outcome.rejection_message is not None
    ):
      raise BackfillError(
        "backfill_spool_invalid",
        "prepared route success result is malformed",
      )
    try:
      return open_prepared_route_spool(
        self._ensure_scratch_directory(),
        outcome.descriptor,
        expected_route_name=route.route_name,
        max_frames=MAXIMUM_ROUTE_FRAMES,
      )
    except SpoolFormatError as exc:
      raise BackfillError(
        "backfill_spool_invalid",
        f"prepared route helper result is invalid: {exc}",
      ) from exc

  def __call__(
    self,
    route: RouteCandidate,
  ) -> PreparedRoute | PreparedRouteSpool:
    if self._closed:
      raise RuntimeError("prepared-route helper lane is closed")
    if (
      self._position >= len(self._routes)
      or self._routes[self._position] != route
    ):
      raise RuntimeError("prepared-route helper call is out of order")

    # Fill the pipeline before the first route's local read. This is the only
    # point where both preparation lanes can overlap immediately; subsequent
    # calls already have their one-route-ahead helper in flight.
    if (
      self._position == 0
      and len(self._routes) > 1
      and self._pending_worker is None
    ):
      self._start_helper(self._routes[1])

    rejection: RouteRejected | None = None
    prepared: PreparedRoute | PreparedRouteSpool | None = None
    if self._pending_route != route:
      try:
        prepared = self._local_prepare(route)
      except RouteRejected as exc:
        rejection = exc
    else:
      try:
        prepared = self._receive_prefetched(route)
      except RouteRejected as exc:
        rejection = exc

    self._position += 1
    if (
      self._position < len(self._routes)
      and self._pending_worker is None
    ):
      try:
        self._start_helper(self._routes[self._position])
      except BaseException:
        if isinstance(prepared, PreparedRouteSpool):
          prepared.cleanup()
        raise

    if rejection is not None:
      raise rejection
    if prepared is None:
      raise AssertionError("route preparation produced no result")
    return prepared

  def close(self) -> None:
    if self._closed:
      return
    worker = self._pending_worker
    self._pending_worker = None
    self._pending_route = None
    cleanup_error: BaseException | None = None
    try:
      if worker is not None:
        worker.cancel()
    except BaseException as exc:
      cleanup_error = exc
    try:
      if (
        self._scratch_directory is not None
        and self._scratch_directory.exists()
      ):
        shutil.rmtree(self._scratch_directory)
    except OSError as exc:
      if cleanup_error is None:
        cleanup_error = BackfillError(
          "backfill_spool_invalid",
          "prepared route scratch cleanup failed",
        )
        cleanup_error.__cause__ = exc
    self._closed = True
    if cleanup_error is not None:
      raise cleanup_error

  def __enter__(self) -> _PrefetchingRoutePreparer:
    return self

  def __exit__(self, *exc_info: object) -> None:
    self.close()


def ledger_routes(
  ledger: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
  return {
    entry["route_name"]: entry
    for entry in ledger["entries"]  # type: ignore[index]
  }


def verify_known_route_hashes(
  ledger: Mapping[str, object],
  discovered: tuple[RouteCandidate, ...],
) -> None:
  known = ledger_routes(ledger)
  for route in discovered:
    entry = known.get(route.route_name)
    if entry is None:
      continue
    expected = [
      segment.to_ledger_dict()
      for segment in route.segments
    ]
    if entry["segments"] != expected:
      raise BackfillError(
        "backfill_untracked_evidence",
        "previously ledgered route content changed",
      )


def extend_ledger(
  ledger: dict[str, object],
  *,
  late_routes: tuple[RouteCandidate, ...],
  replay_results: tuple[ReplayResult, ...],
) -> dict[str, object]:
  entries = list(ledger["entries"])
  for route in late_routes:
    entries.append(ReplayResult(
      route=route,
      disposition="late_older_skipped",
      diagnostic="late_older_skipped",
      provenance=None,
      accepted_sample_count=0,
      rejected_sample_count=0,
      controls_witness_count=0,
      unresolved_witness_count=0,
    ).ledger_entry())
  entries.extend(result.ledger_entry() for result in replay_results)
  counters = [entry["route_counter"] for entry in entries]
  return validate_ledger({
    "entries": entries,
    "runtime_identity_sha256": ledger["runtime_identity_sha256"],
    "schema_version": BACKFILL_LEDGER_SCHEMA_VERSION,
    "watermark_route_counter": max(counters) if counters else None,
  }, runtime_identity_sha256=ledger["runtime_identity_sha256"])


class _BackfillProgressTracker:
  """Track display-only replay work without entering replay artifacts."""

  def __init__(
    self,
    *,
    routes: tuple[RouteCandidate, ...],
    operation_status: LearningOperationStatusPublisher,
    publisher: BackfillProgressPublisher,
    abort_requested: Callable[[], bool],
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
  ) -> None:
    if not routes:
      raise ValueError("progress tracking requires at least one route")
    self.routes = routes
    self.operation_status = operation_status
    self.publisher = publisher
    self.abort_requested = abort_requested
    self.monotonic_ns = monotonic_ns
    self.route_indexes = {
      route.route_name: index
      for index, route in enumerate(routes, start=1)
    }
    self.route_byte_prefixes: dict[str, int] = {}
    self.route_segment_prefixes: dict[str, int] = {}
    byte_prefix = 0
    segment_prefix = 0
    for route in routes:
      self.route_byte_prefixes[route.route_name] = byte_prefix
      self.route_segment_prefixes[route.route_name] = segment_prefix
      byte_prefix += sum(segment.size_bytes for segment in route.segments)
      segment_prefix += len(route.segments)
    self.source_bytes_per_pass = byte_prefix
    self.segments_per_pass = segment_prefix
    if self.source_bytes_per_pass <= 0 or self.segments_per_pass <= 0:
      raise ValueError("progress inventory must contain nonempty segments")

    # Each pass reads every compressed segment and then applies that route's
    # prepared frames. The equal-byte apply proxy prevents the bar from
    # reaching a pass boundary while route evidence is still being ingested.
    self.total_work_units = 4 * self.source_bytes_per_pass
    self.total_replay_segment_count = 2 * self.segments_per_pass
    self._completed_read_units = 0
    self._completed_apply_units = 0
    self._completed_segments = 0
    self._read_rates: list[float] = []
    self._apply_rates: list[float] = []
    self._active_kind: str | None = None
    self._active_started_ns = 0
    self._active_work_units = 0
    self._pass_index = 1
    self._route: RouteCandidate | None = None
    self._segment_index = 0

  def _operation_payload(self) -> dict[str, object]:
    payload = self.operation_status.last_payload
    if payload is None:
      raise RuntimeError("progress has no operation-status identity")
    return payload

  def _remaining_seconds(self) -> int | None:
    # Reading and application have materially different costs. Do not show
    # an ETA until both rates have independent support.
    if len(self._read_rates) < 3 or len(self._apply_rates) < 3:
      return None
    read_rate = statistics.median(self._read_rates)
    apply_rate = statistics.median(self._apply_rates)
    remaining_read = (
      2 * self.source_bytes_per_pass - self._completed_read_units
    )
    remaining_apply = (
      2 * self.source_bytes_per_pass - self._completed_apply_units
    )
    estimate = read_rate * remaining_read + apply_rate * remaining_apply
    return max(0, int(math.ceil(estimate)))

  def _publish(
    self,
    phase: BackfillProgressPhase,
    *,
    coordinate: bool,
  ) -> None:
    _abort_if_requested(
      self.abort_requested,
      "backfill progress publication aborted for onroad transition",
    )
    route = self._route if coordinate else None
    segment_index = self._segment_index if coordinate else None
    self.publisher.publish(
      operation_status=self._operation_payload(),
      phase=phase,
      pass_index=self._pass_index,
      pass_count=2,
      current_route_identity=(
        None if route is None else route.display_identity
      ),
      current_route_index=(
        None if route is None else self.route_indexes[route.route_name]
      ),
      total_route_count=len(self.routes),
      current_segment_index=segment_index,
      current_route_segment_count=(
        None if route is None else len(route.segments)
      ),
      completed_replay_segment_count=self._completed_segments,
      total_replay_segment_count=self.total_replay_segment_count,
      completed_work_units=(
        self._completed_read_units + self._completed_apply_units
      ),
      total_work_units=self.total_work_units,
      approximate_remaining_seconds=(
        self._remaining_seconds() if coordinate else None
      ),
    )

  def segment_started(
    self,
    *,
    pass_index: int,
    route: RouteCandidate,
    segment: RouteSegment,
    segment_index: int,
    segment_count: int,
  ) -> None:
    if (
      pass_index not in (1, 2)
      or segment_count != len(route.segments)
      or not 1 <= segment_index <= segment_count
      or route.segments[segment_index - 1] != segment
      or self._active_kind is not None
    ):
      raise RuntimeError("backfill segment progress is incoherent")
    self._pass_index = pass_index
    self._route = route
    self._segment_index = segment_index
    self._active_kind = "read"
    self._active_started_ns = int(self.monotonic_ns())
    self._active_work_units = segment.size_bytes
    self._publish(BackfillProgressPhase.READING_SEGMENT, coordinate=True)

  def segment_completed(
    self,
    *,
    pass_index: int,
    route: RouteCandidate,
    segment: RouteSegment,
    segment_index: int,
    segment_count: int,
  ) -> None:
    if (
      self._active_kind != "read"
      or pass_index != self._pass_index
      or route is not self._route
      or segment_index != self._segment_index
      or segment_count != len(route.segments)
      or self._active_work_units != segment.size_bytes
    ):
      raise RuntimeError("backfill segment completion is incoherent")
    elapsed_s = max(
      0.0,
      (int(self.monotonic_ns()) - self._active_started_ns) / 1e9,
    )
    if elapsed_s > 0.0:
      self._read_rates.append(elapsed_s / segment.size_bytes)
    self._completed_read_units += segment.size_bytes
    pass_segment_base = (pass_index - 1) * self.segments_per_pass
    route_segment_prefix = self.route_segment_prefixes[route.route_name]
    self._completed_segments = max(
      self._completed_segments,
      pass_segment_base + route_segment_prefix + segment_index,
    )
    self._active_kind = None
    self._active_work_units = 0

  def route_applying(
    self,
    *,
    pass_index: int,
    route: RouteCandidate,
  ) -> None:
    if (
      pass_index not in (1, 2)
      or self._active_kind is not None
      or self._route is not route
      or self._segment_index != len(route.segments)
    ):
      raise RuntimeError("backfill route application is incoherent")
    self._pass_index = pass_index
    self._active_kind = "apply"
    self._active_started_ns = int(self.monotonic_ns())
    self._active_work_units = sum(
      segment.size_bytes for segment in route.segments
    )
    self._publish(BackfillProgressPhase.APPLYING_ROUTE, coordinate=True)

  def route_prepared(
    self,
    *,
    pass_index: int,
    route: RouteCandidate,
  ) -> None:
    """Account for a private helper's completed canonical route read.

    Helpers deliberately have no Params/progress authority. The parent calls
    this immediately before applying the returned spool so the existing
    two-pass projection remains ordered and monotonic without exposing helper
    scheduling in the UI or changing the progress schema.
    """
    if pass_index not in (1, 2) or self._active_kind is not None:
      raise RuntimeError("backfill prepared-route progress is incoherent")
    route_bytes = sum(segment.size_bytes for segment in route.segments)
    pass_byte_base = (pass_index - 1) * self.source_bytes_per_pass
    route_byte_end = self.route_byte_prefixes[route.route_name] + route_bytes
    pass_segment_base = (pass_index - 1) * self.segments_per_pass
    route_segment_end = (
      self.route_segment_prefixes[route.route_name] + len(route.segments)
    )
    self._completed_read_units = max(
      self._completed_read_units,
      pass_byte_base + route_byte_end,
    )
    self._completed_segments = max(
      self._completed_segments,
      pass_segment_base + route_segment_end,
    )
    self._pass_index = pass_index
    self._route = route
    self._segment_index = len(route.segments)

  def route_completed(
    self,
    *,
    pass_index: int,
    route: RouteCandidate,
  ) -> None:
    if pass_index not in (1, 2):
      raise RuntimeError("backfill completion pass is invalid")
    route_bytes = sum(segment.size_bytes for segment in route.segments)
    if self._active_kind == "apply":
      if (
        pass_index != self._pass_index
        or route is not self._route
        or self._active_work_units != route_bytes
      ):
        raise RuntimeError("backfill route completion is incoherent")
      elapsed_s = max(
        0.0,
        (int(self.monotonic_ns()) - self._active_started_ns) / 1e9,
      )
      if elapsed_s > 0.0:
        self._apply_rates.append(elapsed_s / route_bytes)
    elif self._active_kind == "read":
      # A rejected segment never completed; its partial timing is not an ETA
      # observation. The resolved route is skipped in both replay passes.
      pass
    elif self._active_kind is not None:
      raise RuntimeError("backfill route completion has an unknown task")

    pass_byte_base = (pass_index - 1) * self.source_bytes_per_pass
    route_byte_end = (
      self.route_byte_prefixes[route.route_name] + route_bytes
    )
    pass_segment_base = (pass_index - 1) * self.segments_per_pass
    route_segment_end = (
      self.route_segment_prefixes[route.route_name] + len(route.segments)
    )
    self._completed_read_units = max(
      self._completed_read_units,
      pass_byte_base + route_byte_end,
    )
    self._completed_apply_units = max(
      self._completed_apply_units,
      pass_byte_base + route_byte_end,
    )
    self._completed_segments = max(
      self._completed_segments,
      pass_segment_base + route_segment_end,
    )
    self._pass_index = pass_index
    self._route = route
    self._segment_index = len(route.segments)
    self._active_kind = None
    self._active_work_units = 0

  def comparing(self) -> None:
    if (
      self._active_kind is not None
      or self._completed_segments != self.total_replay_segment_count
      or self._completed_read_units + self._completed_apply_units
      != self.total_work_units
    ):
      raise RuntimeError("comparison began before replay completed")
    self._pass_index = 2
    self._route = None
    self._segment_index = 0
    self._publish(BackfillProgressPhase.COMPARING, coordinate=False)

  def parallel_verification_completed(self) -> None:
    """Advance display-only accounting for the unprojected worker pass."""
    if (
      self._active_kind is not None
      or self._completed_segments != self.segments_per_pass
      or self._completed_read_units != self.source_bytes_per_pass
      or self._completed_apply_units != self.source_bytes_per_pass
    ):
      raise RuntimeError(
        "parallel verification completed before primary replay",
      )
    self._completed_read_units = 2 * self.source_bytes_per_pass
    self._completed_apply_units = 2 * self.source_bytes_per_pass
    self._completed_segments = self.total_replay_segment_count
    self._pass_index = 2
    self._route = None
    self._segment_index = 0
    self._active_kind = None
    self._active_work_units = 0

  def publishing(self) -> None:
    self._publish(BackfillProgressPhase.PUBLISHING, coordinate=False)


class HistoricalLearningBackfill:
  """One cancellable, exclusive offroad scan/replay/publication transaction."""

  def __init__(
    self,
    *,
    log_root: str | Path,
    extractor_path: str | Path,
    current_car_params: Any,
    runtime_factory: Callable[[], PersistentLearningRuntime],
    route_bundle_factory: Callable[
      [Any, BuildDescriptor],
      RuntimeVehicleBundle,
    ],
    car_params_decoder: Callable[[bytes], Any],
    descriptor_registry: BuildDescriptorRegistry,
    expected_dongle_id: str,
    operation_status: LearningOperationStatusPublisher,
    abort_requested: Callable[[], bool],
    backfill_progress: BackfillProgressPublisher | None = None,
    progress_monotonic_ns: Callable[[], int] = time.monotonic_ns,
    pending_route_identity: str | None = None,
    replay_worker_count: int = BACKFILL_REPLAY_WORKER_COUNT,
    route_discovery: Callable[
      [Callable[[], bool]], FullRlogDiscovery
    ] | None = None,
    prepared_route_source: Callable[
      [int, RouteCandidate, Callable[[], bool]],
      PreparedRoute | PreparedRouteSpool,
    ] | None = None,
    preparation_extractor_sha256: str | None = None,
    event_reader: Callable[
      [bytes],
      AbstractContextManager[Any],
    ] = bounded_event_reader,
  ) -> None:
    self.log_root = Path(log_root)
    self.extractor_path = Path(extractor_path)
    self.current_car_params = current_car_params
    self.runtime_factory = runtime_factory
    self.route_bundle_factory = route_bundle_factory
    self.car_params_decoder = car_params_decoder
    self.descriptor_registry = descriptor_registry
    self.expected_dongle_id = str(expected_dongle_id)
    self.operation_status = operation_status
    self.backfill_progress = backfill_progress
    self.progress_monotonic_ns = progress_monotonic_ns
    self.abort_requested = abort_requested
    self.pending_route_identity = pending_route_identity
    self._pending_route_quiescence_observed = False
    if (
      type(replay_worker_count) is not int
      or replay_worker_count not in SUPPORTED_BACKFILL_REPLAY_WORKER_COUNTS
    ):
      raise ValueError("backfill replay worker count is outside its bound")
    self.replay_worker_count = replay_worker_count
    # These two seams let an authenticated off-device worker replace only
    # immutable route discovery and preparation.  Learner application,
    # A/A verification, ledger mutation, finalization, and atomic publication
    # remain in this process on the device.  The default path below is the
    # original local-rlog behavior byte-for-byte.
    self.route_discovery = route_discovery
    self.prepared_route_source = prepared_route_source
    if (
      preparation_extractor_sha256 is not None
      and (
        type(preparation_extractor_sha256) is not str
        or _SHA256_RE.fullmatch(preparation_extractor_sha256) is None
      )
    ):
      raise ValueError("preparation extractor identity is invalid")
    # A remote preparation authority runs a different architecture-specific
    # extractor binary. Publication must name the binary that actually decoded
    # the route, never the unused local executable merely because it publishes.
    self.preparation_extractor_sha256 = preparation_extractor_sha256
    self._active_local_extractor_sha256: str | None = None
    self.event_reader = event_reader

  @property
  def pending_route_quiescence_observed(self) -> bool:
    return self._pending_route_quiescence_observed

  def preserve_pending_route_quiescence(self, observed: bool) -> None:
    """Carry the one-poll logger-close guard across backend transactions."""
    if type(observed) is not bool:
      raise TypeError("pending-route quiescence state must be boolean")
    self._pending_route_quiescence_observed = (
      self._pending_route_quiescence_observed or observed
    )

  @staticmethod
  def _runtime_context(
    runtime: PersistentLearningRuntime,
  ) -> dict[str, object]:
    return {
      "runtime_identity_sha256": (
        runtime.runtime_bundle.calibration_identity_sha256
      ),
      "vehicle_identity": runtime.runtime_bundle.vehicle_identity,
    }

  def _publish(
    self,
    runtime: PersistentLearningRuntime,
    *,
    state: LearningOperationState,
    diagnostic: str,
    new_operation: bool = False,
    accepted_sample_count: int = 0,
    rejected_sample_count: int = 0,
    retry_count: int = 0,
    evidence_sha256: str | None = None,
    ledger_sha256: str | None = None,
    current_route_identity: str | None = None,
    current_route_index: int | None = None,
    total_route_count: int | None = None,
    last_route_identity: str | None = None,
  ) -> None:
    if self.abort_requested():
      # Manager's IsOffroad transition is the ownership handoff to the live
      # learner. Treat a stale offroad write as transaction cancellation.
      raise BackfillError(
        "unexpected_error",
        "backfill status publication aborted for onroad transition",
      )
    self.operation_status.publish(
      state=state,
      diagnostic=diagnostic,
      new_operation=new_operation,
      accepted_sample_count=accepted_sample_count,
      rejected_sample_count=rejected_sample_count,
      retry_count=retry_count,
      evidence_sha256=evidence_sha256,
      ledger_sha256=ledger_sha256,
      current_route_identity=current_route_identity,
      current_route_index=current_route_index,
      total_route_count=total_route_count,
      last_route_identity=last_route_identity,
      **self._runtime_context(runtime),
    )

  def _prepare(
    self,
    runtime: PersistentLearningRuntime,
    route: RouteCandidate,
    *,
    authority_index: int = 1,
    expected_extractor_sha256: str | None = None,
    abort_requested: Callable[[], bool] | None = None,
    segment_started: Callable[[RouteSegment, int, int], None] | None = None,
    segment_completed: Callable[[RouteSegment, int, int], None] | None = None,
  ) -> PreparedRoute | PreparedRouteSpool:
    selected_abort = (
      self.abort_requested
      if abort_requested is None
      else abort_requested
    )
    if authority_index not in (1, 2):
      raise ValueError("backfill preparation authority must be one or two")
    if self.prepared_route_source is not None:
      _abort_if_requested(
        selected_abort,
        "backfill aborted before prepared-route retrieval",
      )
      prepared = self.prepared_route_source(
        authority_index,
        route,
        selected_abort,
      )
      if type(prepared) not in (PreparedRoute, PreparedRouteSpool):
        raise BackfillError(
          "backfill_route_incompatible",
          "prepared-route source returned an invalid result",
        )
      _abort_if_requested(
        selected_abort,
        "backfill aborted after prepared-route retrieval",
      )
    else:
      prepared = prepare_route(
        route,
        extractor_path=self.extractor_path,
        event_reader=self.event_reader,
        car_params_decoder=self.car_params_decoder,
        descriptor_registry=self.descriptor_registry,
        route_bundle_factory=self.route_bundle_factory,
        current_car_params=self.current_car_params,
        current_bundle=runtime.runtime_bundle,
        expected_dongle_id=self.expected_dongle_id,
        expected_extractor_sha256=(
          self._active_local_extractor_sha256
          if expected_extractor_sha256 is None
          else expected_extractor_sha256
        ),
        abort_requested=selected_abort,
        segment_started=segment_started,
        segment_completed=segment_completed,
      )
    artifact = getattr(prepared, "route_evidence", None)
    if isinstance(prepared, PreparedRouteSpool):
      _stage_route_evidence_spool(
        root=runtime.artifact_paths.root,
        authority_index=authority_index,
        spool=prepared,
      )
    elif type(artifact) is RouteEvidenceArtifact:
      _stage_route_evidence(
        root=runtime.artifact_paths.root,
        authority_index=authority_index,
        artifact=artifact,
      )
    else:
      raise BackfillError(
        "backfill_route_incompatible",
        "prepared route lacks complete shared route evidence v2",
      )
    return prepared

  def _run_once_with_extractor_identity(self) -> BackfillRunResult:
    if self.abort_requested():
      raise BackfillError(
        "unexpected_error",
        "backfill cannot start onroad",
      )
    progress_enabled = self.backfill_progress is not None

    def disable_progress() -> None:
      nonlocal progress_enabled
      progress_enabled = False
      if self.backfill_progress is not None:
        try:
          self.backfill_progress.clear()
        except Exception:
          pass

    def project_progress(action: Callable[[], None]) -> None:
      """Run display-only work without coupling it to learning liveness."""
      if not progress_enabled:
        return
      try:
        action()
      except BackfillError:
        # Tracker publication checks the same onroad-cancellation source as
        # replay. That handoff remains authoritative and must not be hidden.
        raise
      except Exception:
        disable_progress()

    if self.backfill_progress is not None:
      project_progress(self.backfill_progress.clear)
    initial_runtime = self.runtime_factory()
    self._publish(
      initial_runtime,
      state=LearningOperationState.BACKFILLING,
      diagnostic="scanning_routes",
      new_operation=True,
    )
    artifact_paths = initial_runtime.artifact_paths
    runtime_identity = (
      initial_runtime.runtime_bundle.calibration_identity_sha256
    )
    with ExclusiveBackfillWriter(artifact_paths.root) as writer:
      # Re-resolve under the writer lock in case CURRENT changed between
      # process startup and lock acquisition.
      initial_runtime = self.runtime_factory()
      artifact_paths = initial_runtime.artifact_paths
      runtime_identity = (
        initial_runtime.runtime_bundle.calibration_identity_sha256
      )
      cleanup_stale_prepared_route_spools(artifact_paths.root)
      ledger = load_ledger(
        artifact_paths,
        runtime_identity_sha256=runtime_identity,
      )
      if self.route_discovery is None:
        discovery = discover_full_rlog_state(
          self.log_root,
          abort_requested=self.abort_requested,
        )
      else:
        _abort_if_requested(
          self.abort_requested,
          "backfill aborted before remote route discovery",
        )
        discovery = self.route_discovery(self.abort_requested)
        _abort_if_requested(
          self.abort_requested,
          "backfill aborted after remote route discovery",
        )
      if type(discovery) is not FullRlogDiscovery:
        raise BackfillError(
          "backfill_route_incompatible",
          "route discovery source returned an invalid inventory",
        )
      discovered = discovery.candidates
      pending_close = discovery.pending_logger_close
      verify_known_route_hashes(ledger, discovered)
      known = ledger_routes(ledger)
      unprocessed = tuple(
        route
        for route in discovered
        if route.route_name not in known
      )
      if (
        self.pending_route_identity is not None
        and not self._pending_route_quiescence_observed
      ):
        current_route = tuple(
          route
          for route in unprocessed
          if route.display_identity == self.pending_route_identity
        )
        if current_route:
          # loggerd removes its marker immediately before its zstd writer
          # destructor flushes/closes. A locked snapshot does not consume this
          # guard: first discover the just-finished route unlocked, then defer
          # it for one complete daemon poll before any hashing/replay.
          self._pending_route_quiescence_observed = True
          pending_close = True
          current_names = {
            route.route_name
            for route in current_route
          }
          unprocessed = tuple(
            route
            for route in unprocessed
            if route.route_name not in current_names
          )
      watermark = ledger["watermark_route_counter"]
      late_routes = tuple(
        route
        for route in unprocessed
        if watermark is not None and route.route_counter <= watermark
      )
      replay_candidates = tuple(
        route
        for route in unprocessed
        if watermark is None or route.route_counter > watermark
      )
      progress = None
      if (
        progress_enabled
        and self.backfill_progress is not None
        and replay_candidates
      ):
        try:
          progress = _BackfillProgressTracker(
            routes=replay_candidates,
            operation_status=self.operation_status,
            publisher=self.backfill_progress,
            abort_requested=self.abort_requested,
            monotonic_ns=self.progress_monotonic_ns,
          )
        except BackfillError:
          raise
        except Exception:
          disable_progress()

      if not unprocessed:
        if pending_close:
          self._publish(
            initial_runtime,
            state=LearningOperationState.FINALIZING,
            diagnostic="finalizing_drive",
            last_route_identity=self.pending_route_identity,
          )
          return BackfillRunResult(
            publication=None,
            pending_logger_close=True,
          )
        if artifact_paths.backfill_pointer.is_file():
          finalization = initial_runtime.coordinator.finalize()
          ledger_sha256 = _sha256(
            artifact_paths.backfill_ledger.read_bytes(),
          )
          self._publish(
            initial_runtime,
            state=LearningOperationState.IDLE,
            diagnostic="evidence_ready",
            evidence_sha256=finalization.evidence_sha256,
            ledger_sha256=ledger_sha256,
          )
        else:
          self._publish(
            initial_runtime,
            state=LearningOperationState.READY_NO_EVIDENCE,
            diagnostic="ready_for_first_drive",
          )
        return BackfillRunResult(
          publication=None,
          pending_logger_close=False,
        )

      route_indexes = {
        route.route_name: index
        for index, route in enumerate(replay_candidates, start=1)
      }
      first_progress_accepted = 0
      first_progress_rejected = 0

      def first_prepare(route: RouteCandidate) -> PreparedRoute:
        self._publish(
          initial_runtime,
          state=LearningOperationState.BACKFILLING,
          diagnostic="replaying_route",
          current_route_identity=route.display_identity,
          current_route_index=route_indexes[route.route_name],
          total_route_count=len(replay_candidates),
          accepted_sample_count=first_progress_accepted,
          rejected_sample_count=first_progress_rejected,
        )
        return self._prepare(
          initial_runtime,
          route,
          authority_index=1,
          segment_started=(
            None
            if progress is None
            else lambda segment, segment_index, segment_count: project_progress(
              lambda: progress.segment_started(
                pass_index=1,
                route=route,
                segment=segment,
                segment_index=segment_index,
                segment_count=segment_count,
              )
            )
          ),
          segment_completed=(
            None
            if progress is None
            else lambda segment, segment_index, segment_count: project_progress(
              lambda: progress.segment_completed(
                pass_index=1,
                route=route,
                segment=segment,
                segment_index=segment_index,
                segment_count=segment_count,
              )
            )
          ),
        )

      def first_route_completed(
        route: RouteCandidate,
        accepted: int,
        rejected: int,
      ) -> None:
        nonlocal first_progress_accepted, first_progress_rejected
        first_progress_accepted = accepted
        first_progress_rejected = rejected
        self._publish(
          initial_runtime,
          state=LearningOperationState.BACKFILLING,
          diagnostic="replaying_route",
          current_route_identity=route.display_identity,
          current_route_index=route_indexes[route.route_name],
          total_route_count=len(replay_candidates),
          accepted_sample_count=accepted,
          rejected_sample_count=rejected,
        )
        if progress is not None:
          project_progress(
            lambda: progress.route_completed(pass_index=1, route=route),
          )

      def first_prefetched_route_ready(route: RouteCandidate) -> None:
        self._publish(
          initial_runtime,
          state=LearningOperationState.BACKFILLING,
          diagnostic="replaying_route",
          current_route_identity=route.display_identity,
          current_route_index=route_indexes[route.route_name],
          total_route_count=len(replay_candidates),
          accepted_sample_count=first_progress_accepted,
          rejected_sample_count=first_progress_rejected,
        )
        if progress is not None:
          project_progress(
            lambda: progress.route_prepared(
              pass_index=1,
              route=route,
            ),
          )

      def parallel_verification_replay(
        worker_abort_requested: Callable[[], bool],
      ) -> ReplayPass:
        worker_runtime = self.runtime_factory()
        def verification_prepare(route: RouteCandidate) -> PreparedRoute:
          return self._prepare(
            worker_runtime,
            route,
            authority_index=2,
            abort_requested=worker_abort_requested,
          )

        if self.replay_worker_count == 4:
          with _PrefetchingRoutePreparer(
            authority_index=2,
            routes=replay_candidates,
            local_prepare=verification_prepare,
            helper_prepare=(
              lambda route, helper_abort_requested: self._prepare(
                worker_runtime,
                route,
                authority_index=2,
                abort_requested=helper_abort_requested,
              )
            ),
            scratch_parent=artifact_paths.root,
            abort_requested=worker_abort_requested,
            helper_transaction_abort_requested=self.abort_requested,
          ) as verification_preparer:
            return replay_routes(
              runtime=worker_runtime,
              routes=replay_candidates,
              prepare=verification_preparer,
              abort_requested=worker_abort_requested,
            )
        return replay_routes(
          runtime=worker_runtime,
          routes=replay_candidates,
          prepare=verification_prepare,
          abort_requested=worker_abort_requested,
        )

      # Restore the parent runtime before creating any child. A restore
      # failure must leave no verification worker or inherited resources.
      first_runtime = self.runtime_factory()
      verification_worker = (
        None
        if self.replay_worker_count == 1
        else _ForkedReplayWorker(
          replay=parallel_verification_replay,
          abort_requested=self.abort_requested,
          inherited_close_fds=(writer.fileno,),
        )
      )
      try:
        first_route_applying = (
          None
          if progress is None
          else lambda route: project_progress(
            lambda: progress.route_applying(
              pass_index=1,
              route=route,
            ),
          )
        )
        if self.replay_worker_count == 4:
          with _PrefetchingRoutePreparer(
            authority_index=1,
            routes=replay_candidates,
            local_prepare=first_prepare,
            helper_prepare=(
              lambda route, helper_abort_requested: self._prepare(
                first_runtime,
                route,
                authority_index=1,
                abort_requested=helper_abort_requested,
              )
            ),
            scratch_parent=artifact_paths.root,
            abort_requested=self.abort_requested,
            inherited_close_fds=(writer.fileno,),
            prefetched_route_ready=first_prefetched_route_ready,
          ) as first_preparer:
            first = replay_routes(
              runtime=first_runtime,
              routes=replay_candidates,
              prepare=first_preparer,
              abort_requested=self.abort_requested,
              route_completed=first_route_completed,
              route_applying=first_route_applying,
            )
        else:
          first = replay_routes(
            runtime=first_runtime,
            routes=replay_candidates,
            prepare=first_prepare,
            abort_requested=self.abort_requested,
            route_completed=first_route_completed,
            route_applying=first_route_applying,
          )
      except BaseException:
        if verification_worker is not None:
          verification_worker.cancel()
        raise
      last_route_identity = (
        replay_candidates[-1].display_identity
        if replay_candidates
        else None
      )
      try:
        self._publish(
          initial_runtime,
          state=LearningOperationState.FINALIZING,
          diagnostic="verifying_backfill",
          accepted_sample_count=first.accepted_sample_count,
          rejected_sample_count=first.rejected_sample_count,
          last_route_identity=last_route_identity,
        )
      except BaseException:
        if verification_worker is not None:
          verification_worker.cancel()
        raise
      if verification_worker is not None:
        second = verification_worker.result()
        if progress is not None:
          project_progress(progress.parallel_verification_completed)
      else:
        second_runtime = self.runtime_factory()

        def second_prepare(route: RouteCandidate) -> PreparedRoute:
          return self._prepare(
            second_runtime,
            route,
            authority_index=2,
            segment_started=(
              None
              if progress is None
              else lambda segment, segment_index, segment_count: project_progress(
                lambda: progress.segment_started(
                  pass_index=2,
                  route=route,
                  segment=segment,
                  segment_index=segment_index,
                  segment_count=segment_count,
                )
              )
            ),
            segment_completed=(
              None
              if progress is None
              else lambda segment, segment_index, segment_count: project_progress(
                lambda: progress.segment_completed(
                  pass_index=2,
                  route=route,
                  segment=segment,
                  segment_index=segment_index,
                  segment_count=segment_count,
                )
              )
            ),
          )

        def second_route_completed(
          route: RouteCandidate,
          _accepted: int,
          _rejected: int,
        ) -> None:
          if progress is not None:
            project_progress(
              lambda: progress.route_completed(pass_index=2, route=route),
            )

        second = replay_routes(
          runtime=second_runtime,
          routes=replay_candidates,
          prepare=second_prepare,
          abort_requested=self.abort_requested,
          route_completed=second_route_completed,
          route_applying=(
            None
            if progress is None
            else lambda route: project_progress(
              lambda: progress.route_applying(
                pass_index=2,
                route=route,
              ),
            )
          ),
        )
      if progress is not None:
        project_progress(progress.comparing)
      verify_replay_passes(first, second)
      _publish_route_evidence_after_aa(
        root=artifact_paths.root,
        first=first,
        second=second,
      )
      new_ledger = extend_ledger(
        ledger,
        late_routes=late_routes,
        replay_results=first.results,
      )
      self._publish(
        initial_runtime,
        state=LearningOperationState.FINALIZING,
        diagnostic="publishing_backfill",
        accepted_sample_count=first.accepted_sample_count,
        rejected_sample_count=first.rejected_sample_count,
        last_route_identity=last_route_identity,
      )
      if progress is not None:
        project_progress(progress.publishing)
      extractor_sha256 = self.preparation_extractor_sha256
      if extractor_sha256 is None:
        extractor_sha256 = self._active_local_extractor_sha256
      if extractor_sha256 is None:
        raise BackfillError(
          "backfill_reader_unavailable",
          "native extractor transaction identity is unavailable",
        )
      generation_sha256, ledger_sha256 = publish_generation(
        artifact_paths=artifact_paths,
        runtime_identity_sha256=runtime_identity,
        finalization=first.finalization,
        ledger=new_ledger,
        descriptor_registry_sha256=(
          self.descriptor_registry.identity_sha256
        ),
        extractor_sha256=extractor_sha256,
        abort_requested=self.abort_requested,
      )
      rejected = any(
        result.disposition == "rejected"
        for result in first.results
      )
      if rejected:
        diagnostic = "backfill_complete_with_rejections"
      elif late_routes:
        diagnostic = "backfill_complete_late_older_skipped"
      else:
        diagnostic = "backfill_complete"
      self._publish(
        initial_runtime,
        state=LearningOperationState.IDLE,
        diagnostic=diagnostic,
        accepted_sample_count=first.accepted_sample_count,
        rejected_sample_count=first.rejected_sample_count,
        evidence_sha256=first.finalization.evidence_sha256,
        ledger_sha256=ledger_sha256,
        last_route_identity=last_route_identity,
      )
      if self.backfill_progress is not None:
        project_progress(self.backfill_progress.clear)
      if pending_close:
        # The committed generation is complete, but loggerd is still closing
        # another route. Replace the terminal publication operation with a
        # fresh stable FINALIZING operation before the daemon begins lock-only
        # polling, while retaining the authenticated committed snapshot.
        self._publish(
          initial_runtime,
          state=LearningOperationState.FINALIZING,
          diagnostic="finalizing_drive",
          new_operation=True,
          evidence_sha256=first.finalization.evidence_sha256,
          ledger_sha256=ledger_sha256,
          last_route_identity=self.pending_route_identity,
        )
      return BackfillRunResult(
        publication=BackfillPublication(
          generation_sha256=generation_sha256,
          ledger_sha256=ledger_sha256,
          finalization=first.finalization,
          accepted_sample_count=first.accepted_sample_count,
          rejected_sample_count=first.rejected_sample_count,
          diagnostic=diagnostic,
        ),
        pending_logger_close=pending_close,
      )

  def run_once(self) -> BackfillRunResult:
    """Run one transaction under one pinned local extractor identity."""
    if (
      self.prepared_route_source is not None
      and self.preparation_extractor_sha256 is not None
    ):
      return self._run_once_with_extractor_identity()
    if self._active_local_extractor_sha256 is not None:
      raise BackfillError(
        "unexpected_error",
        "native extractor transaction is already active",
      )
    extractor = open_verified_extractor(
      self.extractor_path,
      abort_requested=self.abort_requested,
    )
    self._active_local_extractor_sha256 = extractor.sha256
    try:
      return self._run_once_with_extractor_identity()
    finally:
      try:
        verify_open_extractor(
          extractor,
          abort_requested=self.abort_requested,
        )
      finally:
        self._active_local_extractor_sha256 = None
        os.close(extractor.descriptor)

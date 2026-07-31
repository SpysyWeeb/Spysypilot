"""Offroad-only, full-rlog owner for durable BLaTv2 learning evidence.

The live learner is preview-only. This module is the sole durable evidence
writer once shipped: it accepts only complete local full-rlog routes, replays
the existing measured-frame/runtime path twice, records every decision in a
canonical SHA-bound ledger, and publishes an immutable generation by one
atomic CURRENT-pointer replacement.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import shutil
import statistics
import struct
import subprocess
import tempfile
import time
from typing import Any

from openpilot.cereal.services import SERVICE_LIST
from openpilot.selfdrive.controls.lib.blatv2.learning_coordinator import (
  LearningFinalization,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_frame import (
  CanonicalSourceHistory,
  maximum_source_age_ns,
  measured_learning_frame,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill_progress import (
  BackfillProgressPhase,
  BackfillProgressPublisher,
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
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  RuntimeVehicleBundle,
)


BACKFILL_LEDGER_SCHEMA_VERSION = 1
BACKFILL_PROVENANCE_SCHEMA_VERSION = 1
BACKFILL_COMMIT_SCHEMA_VERSION = 1
BACKFILL_POINTER_SCHEMA_VERSION = 1
NATIVE_EXTRACTOR_SCHEMA_VERSION = 1
CANONICAL_JOIN_SCHEMA_VERSION = 1
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
}
_SOURCE_SERVICES = (
  "carControl",
  "carState",
  "carOutput",
  "liveParameters",
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


@dataclass(frozen=True, slots=True)
class ExtractedEvent:
  which: int
  mono_ns: int
  ordinal: int
  encoded: bytes


@dataclass(frozen=True, slots=True)
class _DecodedExtractedEvent:
  which: str
  mono_ns: int
  valid: bool
  payload: Any


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
class PreparedRoute:
  frames: tuple[Any, ...]
  controls_witness_count: int
  unresolved_witness_count: int
  gap_count: int
  provenance: dict[str, object]


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
      "route_name": self.route.route_name,
      "segments": [
        segment.to_ledger_dict()
        for segment in self.route.segments
      ],
      "unresolved_witness_count": self.unresolved_witness_count,
    }


@dataclass(frozen=True, slots=True)
class ReplayPass:
  finalization: LearningFinalization
  results: tuple[ReplayResult, ...]
  accepted_sample_count: int
  rejected_sample_count: int


@dataclass(frozen=True, slots=True)
class BackfillPublication:
  generation_sha256: str
  ledger_sha256: str
  finalization: LearningFinalization
  accepted_sample_count: int
  rejected_sample_count: int
  diagnostic: str


@dataclass(frozen=True, slots=True)
class BackfillRunResult:
  publication: BackfillPublication | None
  pending_logger_close: bool


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
) -> tuple[ExtractedEvent, ...]:
  """Buffer one selected segment only after a verified native trailer/exit."""
  extractor = Path(extractor_path)
  if not extractor.is_file() or not os.access(extractor, os.X_OK):
    raise BackfillError(
      "backfill_reader_unavailable",
      "native rlog extractor is unavailable",
    )
  with tempfile.TemporaryFile() as errors:
    process = subprocess.Popen(
      (str(extractor), str(segment_path)),
      stdin=subprocess.DEVNULL,
      stdout=subprocess.PIPE,
      stderr=errors,
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
    for node in bundle.seed_profile.nodes
  }
  if len(rack_resolutions) != 1:
    raise ValueError("runtime seed has inconsistent rack-rate resolution")
  return {
    "car_fingerprint": payload["car_fingerprint"],
    "nominal_rack_mapping": payload["nominal_rack_mapping"],
    "rack_rate_resolution_deg_s": rack_resolutions.pop(),
    "seed_profile": payload["seed_profile"],
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


def _copy_message(message: Any) -> Any:
  builder = getattr(message, "as_builder", None)
  return builder() if callable(builder) else message


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
      elif which in _SOURCE_SERVICES:
        payload = _copy_message(getattr(event, which))
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
  abort_requested: Callable[[], bool] = lambda: False,
  segment_started: Callable[[RouteSegment, int, int], None] | None = None,
  segment_completed: Callable[[RouteSegment, int, int], None] | None = None,
) -> PreparedRoute:
  """Validate one complete route before exposing any frame to the learner."""
  all_frames = []
  histories = {
    service: CanonicalSourceHistory()
    for service in _SOURCE_SERVICES
  }
  first_source_time: dict[str, int] = {}
  unresolved_times: list[int] = []
  control_times: list[int] = []
  controls_count = 0
  decoded_controls_count = 0
  route_car_params_bytes: bytes | None = None
  route_car_params: Any | None = None
  route_descriptor: BuildDescriptor | None = None
  route_init_identity: tuple[object, ...] | None = None
  physical_compatibility_sha256: str | None = None
  last_service_time: dict[str, int] = {}
  extraction_digest = hashlib.sha256()

  segment_count = len(route.segments)
  for segment_position, segment in enumerate(route.segments, start=1):
    if abort_requested():
      raise BackfillError(
        "unexpected_error",
        "backfill aborted for onroad transition",
      )
    if segment_started is not None:
      segment_started(segment, segment_position, segment_count)
    try:
      before_sha = _sha256_file(
        segment.path,
        abort_requested=abort_requested,
      )
    except OSError as exc:
      raise RouteRejected(
        "segment_changed",
        "route segment disappeared after discovery",
      ) from exc
    if before_sha != segment.sha256:
      raise RouteRejected(
        "segment_changed",
        "route segment changed after discovery",
      )
    records = extract_segment_events(
      extractor_path,
      segment.path,
      abort_requested=abort_requested,
    )
    try:
      after_sha = _sha256_file(
        segment.path,
        abort_requested=abort_requested,
      )
    except OSError as exc:
      raise RouteRejected(
        "segment_changed",
        "route segment disappeared during extraction",
      ) from exc
    if after_sha != segment.sha256:
      raise RouteRejected(
        "segment_changed",
        "route segment changed during extraction",
      )

    sentinels = []
    segment_init_seen = False
    segment_payload_started = False
    segment_ended = False
    decoded_records = []
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
      elif which in _SOURCE_SERVICES:
        if record.mono_ns < last_service_time.get(which, 0):
          raise RouteRejected(
            "service_time_regression",
            "measurement service timestamps move backwards",
          )
        last_service_time[which] = record.mono_ns
        decoded_records.append((
          record.mono_ns,
          segment.index,
          record.ordinal,
          which,
          decoded.valid,
          decoded.payload,
        ))
        first_source_time.setdefault(which, record.mono_ns)
      elif which == "controlsState":
        decoded_controls_count += 1
        if decoded_controls_count > MAXIMUM_ROUTE_FRAMES:
          raise RouteRejected(
            "route_too_large",
            "route exceeds bounded controls witness count",
          )
        if record.mono_ns <= last_service_time.get(which, 0):
          raise RouteRejected(
            "control_time_regression",
            "controls witness timestamps are not strictly increasing",
          )
        last_service_time[which] = record.mono_ns
        decoded_records.append((
          record.mono_ns,
          segment.index,
          record.ordinal,
          which,
          decoded.valid,
          None,
        ))

    expected_start = (
      "startOfRoute" if segment.index == 0 else "startOfSegment"
    )
    expected_end = (
      "endOfRoute"
      if segment.index == route.segments[-1].index
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

    for join_index, (
      mono_ns,
      _,
      _ordinal,
      which,
      valid,
      message,
    ) in enumerate(sorted(decoded_records)):
      if join_index % 256 == 0:
        _abort_if_requested(
          abort_requested,
          "backfill aborted while joining route measurements",
        )
      if which in _SOURCE_SERVICES:
        histories[which].update(
          message=message,
          mono_ns=mono_ns,
          valid=valid,
          alive=True,
        )
        continue
      controls_count += 1
      control_times.append(mono_ns)
      selected = {}
      if valid and mono_ns > 0:
        for service, history in histories.items():
          snapshot = history.select(
            witness_mono_ns=mono_ns,
            maximum_age_ns=maximum_source_age_ns(service),
          )
          if snapshot is None:
            selected = {}
            break
          selected[service] = snapshot
      if not selected:
        unresolved_times.append(mono_ns)
        continue
      try:
        all_frames.append(measured_learning_frame(
          witness_mono_ns=mono_ns,
          car_state=selected["carState"].message,
          car_control=selected["carControl"].message,
          car_output=selected["carOutput"].message,
          live_parameters=selected["liveParameters"].message,
        ))
      except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise RouteRejected(
          "measured_frame_invalid",
          "route measured frame is not representable",
        ) from exc
    if segment_completed is not None:
      segment_completed(segment, segment_position, segment_count)
  if (
    route_init_identity is None
    or route_descriptor is None
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

  if set(first_source_time) != set(_SOURCE_SERVICES):
    raise RouteRejected(
      "missing_measurement_service",
      "route lacks a required measurement service",
    )
  # Gate the complete controls population. A late-starting or early-stopping
  # required source must not silently shrink the quality denominator.
  eligible_controls = control_times
  unresolved_in_coverage = len(unresolved_times)
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
  return PreparedRoute(
    frames=tuple(all_frames),
    controls_witness_count=controls_count,
    unresolved_witness_count=len(unresolved_times),
    gap_count=gap_count,
    provenance=provenance,
  )


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
  finalization: LearningFinalization,
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


def replay_routes(
  *,
  runtime: PersistentLearningRuntime,
  routes: tuple[RouteCandidate, ...],
  prepare: Callable[[RouteCandidate], PreparedRoute],
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
    before_ingested = runtime.coordinator.ingested_sample_count
    before_accepted = runtime.coordinator.accepted_sample_count
    if route_applying is not None:
      route_applying(route)
    runtime.transition_onroad()
    for frame_index, frame in enumerate(prepared.frames):
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
    ingested = (
      runtime.coordinator.ingested_sample_count - before_ingested
    )
    accepted = (
      runtime.coordinator.accepted_sample_count - before_accepted
    )
    rejected = (
      ingested - accepted
      + prepared.unresolved_witness_count
      + prepared.gap_count
    )
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
    ))
    if route_completed is not None:
      route_completed(route, accepted_total, rejected_total)
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
    self.event_reader = event_reader

  @staticmethod
  def _runtime_context(
    runtime: PersistentLearningRuntime,
  ) -> dict[str, object]:
    return {
      "runtime_identity_sha256": (
        runtime.runtime_bundle.identity_sha256
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
    segment_started: Callable[[RouteSegment, int, int], None] | None = None,
    segment_completed: Callable[[RouteSegment, int, int], None] | None = None,
  ) -> PreparedRoute:
    return prepare_route(
      route,
      extractor_path=self.extractor_path,
      event_reader=self.event_reader,
      car_params_decoder=self.car_params_decoder,
      descriptor_registry=self.descriptor_registry,
      route_bundle_factory=self.route_bundle_factory,
      current_car_params=self.current_car_params,
      current_bundle=runtime.runtime_bundle,
      expected_dongle_id=self.expected_dongle_id,
      abort_requested=self.abort_requested,
      segment_started=segment_started,
      segment_completed=segment_completed,
    )

  def run_once(self) -> BackfillRunResult:
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
    runtime_identity = initial_runtime.runtime_bundle.identity_sha256
    with ExclusiveBackfillWriter(artifact_paths.root):
      # Re-resolve under the writer lock in case CURRENT changed between
      # process startup and lock acquisition.
      initial_runtime = self.runtime_factory()
      artifact_paths = initial_runtime.artifact_paths
      runtime_identity = initial_runtime.runtime_bundle.identity_sha256
      ledger = load_ledger(
        artifact_paths,
        runtime_identity_sha256=runtime_identity,
      )
      discovery = discover_full_rlog_state(
        self.log_root,
        abort_requested=self.abort_requested,
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

      first_runtime = self.runtime_factory()
      first = replay_routes(
        runtime=first_runtime,
        routes=replay_candidates,
        prepare=first_prepare,
        abort_requested=self.abort_requested,
        route_completed=first_route_completed,
        route_applying=(
          None
          if progress is None
          else lambda route: project_progress(
            lambda: progress.route_applying(
              pass_index=1,
              route=route,
            ),
          )
        ),
      )
      last_route_identity = (
        replay_candidates[-1].display_identity
        if replay_candidates
        else None
      )
      self._publish(
        initial_runtime,
        state=LearningOperationState.FINALIZING,
        diagnostic="verifying_backfill",
        accepted_sample_count=first.accepted_sample_count,
        rejected_sample_count=first.rejected_sample_count,
        last_route_identity=last_route_identity,
      )
      second_runtime = self.runtime_factory()

      def second_prepare(route: RouteCandidate) -> PreparedRoute:
        return self._prepare(
          second_runtime,
          route,
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
      extractor_sha256 = _sha256_file(
        self.extractor_path,
        abort_requested=self.abort_requested,
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

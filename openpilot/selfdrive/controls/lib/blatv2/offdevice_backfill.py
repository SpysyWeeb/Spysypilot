"""Device-authoritative orchestration for PC-assisted route preparation.

The remote worker may decode and canonicalize immutable full rlogs.  It never
runs the learner, mutates the ledger, finalizes a profile, writes Params, or
publishes CURRENT.  This module downloads two independently prepared spool
sets, then feeds them to the unchanged device-side ``HistoricalLearningBackfill``
transaction.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import hashlib
import hmac
import multiprocessing
import os
from pathlib import Path
import re
import resource
import secrets
import signal
import stat
import tempfile
import time
from typing import Final

from openpilot.selfdrive.controls.lib.blatv2.certification_vector import (
  CERTIFICATION_VECTOR_MAX_BYTES,
  CertificationVector,
  CertificationVectorError,
  build_certification_vector,
  certification_vector_selection,
)

from openpilot.selfdrive.controls.lib.blatv2.learning_backfill import (
  BACKFILL_PROVENANCE_SCHEMA_VERSION,
  MAXIMUM_ROUTE_FRAMES,
  BackfillError,
  FullRlogDiscovery,
  HistoricalLearningBackfill,
  PreparedRoute,
  RouteCandidate,
  RouteRejected,
  RouteSegment,
  discover_full_rlog_state,
  ledger_routes,
  load_ledger,
  verify_known_route_hashes,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill_progress import (
  BackfillProgressPhase,
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
  route_identity_sha256,
)
from openpilot.selfdrive.controls.lib.blatv2.offdevice_client import (
  OffdeviceBridgeClient,
)
from openpilot.selfdrive.controls.lib.blatv2.offdevice_progress import (
  OffdeviceFallbackReason,
  OffdeviceProgressPhase,
  OffdeviceProgressPublisher,
)
from openpilot.selfdrive.controls.lib.blatv2.offdevice_protocol import (
  MAX_ROUTE_COUNT,
  BridgeCorruptError,
  BridgeIncompatibleError,
  BridgeRemoteError,
  BridgeUnavailableError,
  canonical_json_bytes,
  check_abort,
  decode_canonical_json,
)


REMOTE_JOB_POLL_SECONDS: Final = 0.25
REMOTE_SCRATCH_PREFIX: Final = ".blatv2-remote-prepare-"
CERTIFICATION_SOURCE_SNAPSHOT_PREFIX: Final = ".blatv2-cert-vector-source-"
PRIVATE_SCRATCH_MARKER: Final = ".blatv2-private-scratch-v1"
PRIVATE_SCRATCH_MARKER_BYTES: Final = b"blatv2-private-scratch-v1\n"
PRIVATE_SCRATCH_MAX_CHILDREN: Final = 1024
PRIVATE_SCRATCH_DISK_MARGIN_BYTES: Final = 16 * 1024 * 1024
PRIVATE_SCRATCH_QUARANTINE_PREFIX: Final = ".blatv2-private-quarantine-"
_REMOTE_PLACEHOLDER_DIRECTORY: Final = ".blatv2-remote-inventory"
REMOTE_CERTIFICATION_DIRECTORY: Final = ".blatv2-offdevice-certifications"
REMOTE_CERTIFICATION_SCHEMA_VERSION: Final = 5
ARCHITECTURE_VERIFICATION_TIMEOUT_SECONDS: Final = 120.0
ARCHITECTURE_VERIFICATION_MAX_RSS_BYTES: Final = 450 * 1024 * 1024
ARCHITECTURE_VERIFICATION_MAX_COMBINED_RSS_BYTES: Final = 600 * 1024 * 1024
ARCHITECTURE_VERIFICATION_MIN_CHILD_BUDGET_BYTES: Final = 128 * 1024 * 1024
ARCHITECTURE_VERIFICATION_MAX_IPC_BYTES: Final = (
  CERTIFICATION_VECTOR_MAX_BYTES + 4096
)
REMOTE_UNAVAILABLE_ERROR_CODES: Final = frozenset({
  "artifact_not_found",
  "busy",
  "internal_error",
  "job_failed",
  "job_not_found",
  "route_unavailable",
})
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE: Final = re.compile(r"[0-9a-f]{40}\Z")
_ROUTE_RE: Final = re.compile(r"[0-9a-f]{8}--[0-9a-f]{10}\Z")
_CERTIFICATION_DOMAIN_KEYS: Final = (
  "canonical_join_schema_version",
  "extractor_schema_version",
  "log_schema_blob",
  "physical_compatibility_sha256",
)
# Full recorded CarParams bytes remain content-addressed in each prepared
# artifact and must agree across the two PC authorities. They are deliberately
# absent here. The complete RuntimeVehicleBundle identity is added by
# ``_certification_domain`` alongside the independently validated physical
# projection, so every accepted numerical dependency remains covered without
# letting unrelated serialized fields create near-one-route domains.


class BridgeFallbackUnavailableError(BridgeUnavailableError):
  """Transient fallback carrying a stable, display-only reason code."""

  def __init__(
    self,
    message: str,
    fallback_reason: OffdeviceFallbackReason,
  ) -> None:
    super().__init__(message)
    self.fallback_reason = fallback_reason


class ArchitectureVerificationError(BridgeIncompatibleError):
  """Fail-closed bounded ARM proof failure; local replay is forbidden."""

  def __init__(self, reason: str) -> None:
    super().__init__(reason)
    self.reason = reason


@dataclass(frozen=True, slots=True)
class _PrivateScratchChild:
  st_dev: int
  st_ino: int
  file_type: int
  stable_size: int | None
  stable_mtime_ns: int | None


@dataclass(slots=True)
class _PrivateScratchRoot:
  """One inode-bound, flat private scratch directory owned by this process."""

  path: Path
  st_dev: int
  st_ino: int
  prefix: str
  children: dict[str, _PrivateScratchChild] = field(default_factory=dict)


def _register_private_scratch_file(
  owner: _PrivateScratchRoot,
  path: Path,
  *,
  stable: bool,
) -> None:
  if path.parent != owner.path or not path.name or "/" in path.name:
    raise BridgeUnavailableError("private scratch child path is invalid")
  try:
    before = path.lstat()
    descriptor = os.open(
      path,
      os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
  except OSError as exc:
    raise BridgeUnavailableError("private scratch child is unavailable") from exc
  try:
    opened = os.fstat(descriptor)
    if (
      path.is_symlink()
      or not stat.S_ISREG(before.st_mode)
      or opened.st_dev != before.st_dev
      or opened.st_ino != before.st_ino
      or opened.st_uid != os.geteuid()
      or stat.S_IMODE(opened.st_mode) & 0o077
    ):
      raise BridgeUnavailableError("private scratch child is unsafe")
    observed = _PrivateScratchChild(
      st_dev=opened.st_dev,
      st_ino=opened.st_ino,
      file_type=stat.S_IFMT(opened.st_mode),
      stable_size=opened.st_size if stable else None,
      stable_mtime_ns=opened.st_mtime_ns if stable else None,
    )
    prior = owner.children.get(path.name)
    if prior is not None and (
      prior.st_dev != observed.st_dev or prior.st_ino != observed.st_ino
    ):
      raise BridgeUnavailableError("private scratch child identity changed")
    owner.children[path.name] = observed
  finally:
    os.close(descriptor)


def _rename_private_scratch_file(
  owner: _PrivateScratchRoot,
  source: Path,
  destination: Path,
  *,
  stable: bool,
) -> None:
  prior = owner.children.get(source.name)
  if prior is None or source.parent != owner.path or destination.parent != owner.path:
    raise BridgeUnavailableError("private scratch rename is unowned")
  os.replace(source, destination)
  owner.children.pop(source.name)
  _register_private_scratch_file(owner, destination, stable=stable)


def _create_private_scratch(
  parent: Path,
  *,
  prefix: str,
  required_bytes: int,
) -> _PrivateScratchRoot:
  if required_bytes < 0:
    raise BridgeUnavailableError("private scratch byte requirement is invalid")
  path: Path | None = None
  try:
    parent.mkdir(parents=True, exist_ok=True)
    parent_info = parent.lstat()
    if parent.is_symlink() or not stat.S_ISDIR(parent_info.st_mode):
      raise BridgeUnavailableError("private scratch parent is unsafe")
    filesystem = os.statvfs(parent)
    available = filesystem.f_bavail * filesystem.f_frsize
    if available < required_bytes + PRIVATE_SCRATCH_DISK_MARGIN_BYTES:
      raise BridgeUnavailableError("private scratch filesystem is full")
    path = Path(tempfile.mkdtemp(dir=parent, prefix=prefix))
    os.chmod(path, 0o700)
    directory_fd = os.open(
      path,
      os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
      | getattr(os, "O_NOFOLLOW", 0),
    )
  except BridgeUnavailableError:
    raise
  except OSError as exc:
    if path is not None:
      try:
        path.rmdir()
      except OSError:
        pass
    raise BridgeUnavailableError("private scratch is unavailable") from exc
  assert path is not None
  marker_fd = -1
  try:
    opened = os.fstat(directory_fd)
    if (
      not stat.S_ISDIR(opened.st_mode)
      or opened.st_uid != os.geteuid()
      or stat.S_IMODE(opened.st_mode) != 0o700
    ):
      raise BridgeUnavailableError("private scratch ownership is invalid")
    marker_fd = os.open(
      PRIVATE_SCRATCH_MARKER,
      os.O_WRONLY | os.O_CREAT | os.O_EXCL
      | getattr(os, "O_NOFOLLOW", 0),
      0o600,
      dir_fd=directory_fd,
    )
    view = memoryview(PRIVATE_SCRATCH_MARKER_BYTES)
    while view:
      written = os.write(marker_fd, view)
      if written <= 0:
        raise OSError("short private scratch marker write")
      view = view[written:]
    os.fsync(marker_fd)
    os.fsync(directory_fd)
    owner = _PrivateScratchRoot(
      path=path,
      st_dev=opened.st_dev,
      st_ino=opened.st_ino,
      prefix=prefix,
    )
    _register_private_scratch_file(
      owner,
      path / PRIVATE_SCRATCH_MARKER,
      stable=True,
    )
    return owner
  except BaseException:
    if marker_fd >= 0:
      os.close(marker_fd)
      marker_fd = -1
    try:
      os.unlink(PRIVATE_SCRATCH_MARKER, dir_fd=directory_fd)
    except FileNotFoundError:
      pass
    os.close(directory_fd)
    directory_fd = -1
    try:
      path.rmdir()
    except OSError:
      pass
    raise
  finally:
    if marker_fd >= 0:
      os.close(marker_fd)
    if directory_fd >= 0:
      os.close(directory_fd)


def _private_scratch_from_path(
  path: Path,
  *,
  prefix: str,
) -> _PrivateScratchRoot:
  try:
    info = path.lstat()
  except FileNotFoundError as exc:
    raise BridgeUnavailableError("private scratch is missing") from exc
  if (
    not path.name.startswith(prefix)
    or path.is_symlink()
    or not stat.S_ISDIR(info.st_mode)
    or info.st_uid != os.geteuid()
    or stat.S_IMODE(info.st_mode) != 0o700
  ):
    raise BridgeUnavailableError("private scratch ownership is invalid")
  owner = _PrivateScratchRoot(path, info.st_dev, info.st_ino, prefix)
  _register_private_scratch_file(
    owner,
    path / PRIVATE_SCRATCH_MARKER,
    stable=True,
  )
  return owner


def _cleanup_private_scratch(owner: _PrivateScratchRoot) -> None:
  """Delete only direct authenticated files from the exact owned inode."""
  directory_fd = -1
  try:
    directory_fd = os.open(
      owner.path,
      os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
      | getattr(os, "O_NOFOLLOW", 0),
    )
    opened = os.fstat(directory_fd)
    if (
      opened.st_dev != owner.st_dev
      or opened.st_ino != owner.st_ino
      or not stat.S_ISDIR(opened.st_mode)
      or opened.st_uid != os.geteuid()
      or stat.S_IMODE(opened.st_mode) != 0o700
    ):
      raise BridgeUnavailableError("private scratch inode changed")
    names = os.listdir(directory_fd)
    if (
      len(names) > PRIVATE_SCRATCH_MAX_CHILDREN
      or PRIVATE_SCRATCH_MARKER not in names
      or not set(names).issubset(owner.children)
    ):
      raise BridgeUnavailableError("private scratch contents are unknown")
    marker_fd = os.open(
      PRIVATE_SCRATCH_MARKER,
      os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
      dir_fd=directory_fd,
    )
    try:
      marker_info = os.fstat(marker_fd)
      marker = os.read(marker_fd, len(PRIVATE_SCRATCH_MARKER_BYTES) + 1)
      if (
        not stat.S_ISREG(marker_info.st_mode)
        or marker_info.st_uid != os.geteuid()
        or stat.S_IMODE(marker_info.st_mode) & 0o077
        or marker != PRIVATE_SCRATCH_MARKER_BYTES
      ):
        raise BridgeUnavailableError("private scratch marker is invalid")
    finally:
      os.close(marker_fd)
    for name in names:
      if not name or "/" in name or name in {".", ".."}:
        raise BridgeUnavailableError("private scratch child name is invalid")
      child = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
      expected = owner.children[name]
      if (
        not stat.S_ISREG(child.st_mode)
        or child.st_uid != os.geteuid()
        or stat.S_IMODE(child.st_mode) & 0o077
        or child.st_dev != expected.st_dev
        or child.st_ino != expected.st_ino
        or stat.S_IFMT(child.st_mode) != expected.file_type
        or (
          expected.stable_size is not None
          and child.st_size != expected.stable_size
        )
        or (
          expected.stable_mtime_ns is not None
          and child.st_mtime_ns != expected.stable_mtime_ns
        )
      ):
        raise BridgeUnavailableError("private scratch child identity changed")
    # Validate the complete flat population before deleting any member.
    for name in names:
      os.unlink(name, dir_fd=directory_fd)
    os.fsync(directory_fd)
  except FileNotFoundError as exc:
    if directory_fd < 0:
      return
    raise BridgeUnavailableError(
      "private scratch child changed during cleanup",
    ) from exc
  except BridgeUnavailableError:
    raise
  except OSError as exc:
    raise BridgeUnavailableError("private scratch cleanup failed") from exc
  finally:
    if directory_fd >= 0:
      os.close(directory_fd)
  try:
    rebound = owner.path.lstat()
    if rebound.st_dev != owner.st_dev or rebound.st_ino != owner.st_ino:
      raise BridgeUnavailableError("private scratch path changed")
    owner.path.rmdir()
  except FileNotFoundError:
    return
  except BridgeUnavailableError:
    raise
  except OSError as exc:
    raise BridgeUnavailableError("private scratch root cleanup failed") from exc


@dataclass(frozen=True, slots=True)
class RemoteRoutePlan:
  discovery: FullRlogDiscovery
  replay_candidates: tuple[RouteCandidate, ...]
  late_candidates: tuple[RouteCandidate, ...]
  upload_candidates: tuple[RouteCandidate, ...]
  locally_available_route_names: frozenset[str]
  unverified_exclusions: tuple[RemoteRouteExclusion, ...] = ()


@dataclass(frozen=True, slots=True)
class RemoteRouteExclusion:
  """Display-only record of one unconsumed PC-only rejection.

  This record is deliberately absent from the learning ledger and every
  qualification input.  It says only that the PC authorities agreed to reject
  archive bytes which the device could not independently inspect; it never
  turns that rejection into device-authoritative evidence.
  """

  route_identity_sha256: str
  rejection_reason: str
  rejection_message: str


@dataclass(frozen=True, slots=True)
class _PreparedOutcome:
  descriptor: PreparedRouteSpoolDescriptor | None
  certification_vector: _CertificationVectorDescriptor | None = None
  rejection_reason: str | None = None
  rejection_message: str | None = None
  certification_identity_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class _CertificationVectorDescriptor:
  filename: str
  sha256: str
  size_bytes: int
  selected_controls_witnesses: int
  selected_segment_count: int
  selection_identity_sha256: str


@dataclass(frozen=True, slots=True)
class _CertificationBinding:
  """Durable numerical compatibility plus non-keyed provenance metadata."""

  compatibility: dict[str, object]
  audit: dict[str, object]


def _candidate_from_inventory(
  route_payload: Mapping[str, object],
  *,
  placeholder_root: Path,
) -> RouteCandidate:
  route_name = str(route_payload["route_name"])
  segments_payload = route_payload["segments"]
  if type(segments_payload) is not list:
    raise BridgeCorruptError("inventory route segments are malformed")
  segments = tuple(
    RouteSegment(
      index=int(segment["index"]),
      path=(
        placeholder_root
        / route_name
        / f"rlog_{int(segment['index']):03d}.zst"
      ),
      sha256=str(segment["sha256"]),
      size_bytes=int(segment["size_bytes"]),
    )
    for segment in segments_payload
  )
  return RouteCandidate(
    route_name=route_name,
    route_counter=int(route_name[:8], 16),
    segments=segments,
  )


def _same_route_bytes(left: RouteCandidate, right: RouteCandidate) -> bool:
  return (
    left.route_name == right.route_name
    and tuple(
      (segment.index, segment.sha256, segment.size_bytes)
      for segment in left.segments
    )
    == tuple(
      (segment.index, segment.sha256, segment.size_bytes)
      for segment in right.segments
    )
  )


def build_remote_route_plan(
  *,
  local_discovery: FullRlogDiscovery,
  inventory_payload: Mapping[str, object],
  expected_dongle_id: str,
  placeholder_root: str | Path,
  engine: HistoricalLearningBackfill,
) -> RemoteRoutePlan:
  """Freeze one canonical route manifest selected by the device.

  The worker offers inventory; it never injects routes itself.  This policy
  explicitly selects every complete route for the current dongle, preserves
  local logger ownership, and rejects a same-name byte disagreement.
  """
  routes_payload = inventory_payload.get("routes")
  if type(routes_payload) is not list:
    raise BridgeCorruptError("worker inventory is malformed")
  remote_by_name: dict[str, RouteCandidate] = {}
  placeholder = Path(placeholder_root) / _REMOTE_PLACEHOLDER_DIRECTORY
  for payload in routes_payload:
    if type(payload) is not dict:
      raise BridgeCorruptError("worker inventory route is malformed")
    if payload["dongle_id"] != expected_dongle_id:
      continue
    candidate = _candidate_from_inventory(payload, placeholder_root=placeholder)
    if candidate.route_name in remote_by_name:
      raise BridgeCorruptError("worker inventory repeats a route")
    remote_by_name[candidate.route_name] = candidate

  local_by_name = {
    route.route_name: route
    for route in local_discovery.candidates
  }
  uploads: list[RouteCandidate] = []
  selected: dict[str, RouteCandidate] = dict(remote_by_name)
  for route_name, local in local_by_name.items():
    remote = remote_by_name.get(route_name)
    if remote is not None and not _same_route_bytes(local, remote):
      raise BridgeCorruptError(
        f"worker and device disagree on route bytes: {route_name}",
      )
    if remote is None:
      uploads.append(local)
      selected[route_name] = local
    else:
      # Preserve the real local path for upload diagnostics while keeping the
      # immutable byte identity supplied by both sides.
      selected[route_name] = local

  candidates = tuple(sorted(
    selected.values(),
    key=lambda route: (route.route_counter, route.route_name),
  ))
  discovery = FullRlogDiscovery(
    candidates=candidates,
    pending_logger_close=local_discovery.pending_logger_close,
  )

  runtime = engine.runtime_factory()
  runtime_identity = runtime.runtime_bundle.calibration_identity_sha256
  ledger = load_ledger(
    runtime.artifact_paths,
    runtime_identity_sha256=runtime_identity,
  )
  verify_known_route_hashes(ledger, candidates)
  known = ledger_routes(ledger)
  watermark = ledger["watermark_route_counter"]
  eligible_unprocessed = tuple(
    route
    for route in candidates
    if route.route_name not in known
    and not (
      engine.pending_route_identity is not None
      and not engine.pending_route_quiescence_observed
      and route.display_identity == engine.pending_route_identity
    )
  )
  late_candidates = tuple(
    route
    for route in eligible_unprocessed
    if watermark is not None and route.route_counter <= watermark
  )
  replay_candidates = tuple(
    route
    for route in eligible_unprocessed
    if watermark is None or route.route_counter > watermark
  )
  if len(replay_candidates) > MAX_ROUTE_COUNT:
    # Protocol v1 deliberately has no batching: partial remote transactions
    # would make one learner publication depend on several worker jobs. Keep
    # the original complete local replay transaction instead.
    raise BridgeFallbackUnavailableError(
      "remote replay candidate count exceeds protocol-v1 bound",
      OffdeviceFallbackReason.REMOTE_ROUTE_LIMIT,
    )
  replay_names = {route.route_name for route in replay_candidates}
  pending_identity = (
    engine.pending_route_identity
    if (
      engine.pending_route_identity is not None
      and not engine.pending_route_quiescence_observed
    )
    else None
  )
  quiescent_uploads = tuple(
    route
    for route in uploads
    if route.display_identity != pending_identity
  )
  # Archive synchronization has no learner/publication authority. Missing
  # routes needed by this job go first; every other immutable, quiescent local
  # route follows so a drive processed during PC absence is not permanently
  # absent from the user's durable workstation archive.
  upload_candidates = tuple(
    route for route in quiescent_uploads if route.route_name in replay_names
  ) + tuple(
    route for route in quiescent_uploads if route.route_name not in replay_names
  )
  return RemoteRoutePlan(
    discovery=discovery,
    replay_candidates=replay_candidates,
    late_candidates=late_candidates,
    upload_candidates=upload_candidates,
    locally_available_route_names=frozenset(local_by_name),
  )


def _hash_regular_file(
  path: Path,
  *,
  abort_requested: Callable[[], bool],
) -> str:
  """Hash one immutable executable without following a replacement link."""
  check_abort(abort_requested)
  try:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
      raise BridgeUnavailableError("device extractor is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
  except BridgeUnavailableError:
    raise
  except OSError as exc:
    raise BridgeUnavailableError("device extractor is unavailable") from exc
  digest = hashlib.sha256()
  try:
    opened = os.fstat(descriptor)
    if opened.st_dev != before.st_dev or opened.st_ino != before.st_ino:
      raise BridgeUnavailableError("device extractor changed during open")
    while True:
      check_abort(abort_requested)
      chunk = os.read(descriptor, 1024 * 1024)
      if not chunk:
        break
      digest.update(chunk)
    after = os.fstat(descriptor)
    if (
      after.st_dev != opened.st_dev
      or after.st_ino != opened.st_ino
      or after.st_size != opened.st_size
      or after.st_mtime_ns != opened.st_mtime_ns
    ):
      raise BridgeUnavailableError("device extractor changed while hashing")
  finally:
    os.close(descriptor)
  return digest.hexdigest()


def _spool_files_equal(
  root: Path,
  left_name: str,
  right_name: str,
  *,
  abort_requested: Callable[[], bool],
) -> bool:
  """Stream-compare two regular spool files inside the private scratch root."""
  if any(
    Path(name).name != name or name in {"", ".", ".."}
    for name in (left_name, right_name)
  ):
    raise BridgeUnavailableError("certification spool name is unsafe")
  directory_fd = -1
  descriptors: list[int] = []
  try:
    directory_fd = os.open(
      root,
      os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    for name in (left_name, right_name):
      descriptors.append(os.open(name, flags, dir_fd=directory_fd))
    initial = [os.fstat(descriptor) for descriptor in descriptors]
    if (
      any(not stat.S_ISREG(item.st_mode) for item in initial)
      or initial[0].st_size != initial[1].st_size
    ):
      return False
    while True:
      check_abort(abort_requested)
      chunks = [os.read(descriptor, 1024 * 1024) for descriptor in descriptors]
      if chunks[0] != chunks[1]:
        return False
      if not chunks[0]:
        break
    after = [os.fstat(descriptor) for descriptor in descriptors]
    for before, current in zip(initial, after, strict=True):
      if (
        before.st_dev != current.st_dev
        or before.st_ino != current.st_ino
        or before.st_size != current.st_size
        or before.st_mtime_ns != current.st_mtime_ns
        or before.st_ctime_ns != current.st_ctime_ns
      ):
        raise BridgeUnavailableError(
          "certification spool changed during byte comparison",
        )
    return True
  except BridgeUnavailableError:
    raise
  except OSError as exc:
    raise BridgeUnavailableError(
      "certification spool could not be compared",
    ) from exc
  finally:
    for descriptor in descriptors:
      os.close(descriptor)
    if directory_fd >= 0:
      os.close(directory_fd)


def _certification_domain(
  provenance: Mapping[str, object],
  *,
  runtime_vehicle_bundle_sha256: str,
) -> dict[str, object]:
  try:
    domain = {
      key: provenance[key]
      for key in _CERTIFICATION_DOMAIN_KEYS
    }
  except KeyError as exc:
    raise BridgeCorruptError(
      "prepared route lacks certification provenance",
    ) from exc
  domain["runtime_vehicle_bundle_sha256"] = (
    runtime_vehicle_bundle_sha256
  )
  if (
    type(domain["canonical_join_schema_version"]) is not int
    or int(domain["canonical_join_schema_version"]) <= 0
    or type(domain["extractor_schema_version"]) is not int
    or int(domain["extractor_schema_version"]) <= 0
    or type(domain["log_schema_blob"]) is not str
    or _COMMIT_RE.fullmatch(str(domain["log_schema_blob"])) is None
    or type(domain["physical_compatibility_sha256"]) is not str
    or _SHA256_RE.fullmatch(
      str(domain["physical_compatibility_sha256"]),
    ) is None
    or type(domain["runtime_vehicle_bundle_sha256"]) is not str
    or _SHA256_RE.fullmatch(
      str(domain["runtime_vehicle_bundle_sha256"]),
    ) is None
  ):
    raise BridgeCorruptError("prepared route certification domain is invalid")
  return domain


def _load_downloaded_certification_vector(
  root: Path,
  descriptor: _CertificationVectorDescriptor,
  *,
  route_name: str,
  abort_requested: Callable[[], bool],
) -> CertificationVector:
  if Path(descriptor.filename).name != descriptor.filename:
    raise BridgeCorruptError("certification vector filename is unsafe")
  path = root / descriptor.filename
  try:
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
      raise BridgeCorruptError("certification vector file is unsafe")
    if info.st_size != descriptor.size_bytes:
      raise BridgeCorruptError("certification vector size changed")
    check_abort(abort_requested)
    encoded = path.read_bytes()
  except BridgeCorruptError:
    raise
  except OSError as exc:
    raise BridgeCorruptError("certification vector is unavailable") from exc
  if hashlib.sha256(encoded).hexdigest() != descriptor.sha256:
    raise BridgeCorruptError("certification vector hash changed")
  try:
    vector = CertificationVector.from_bytes(encoded)
  except CertificationVectorError as exc:
    raise BridgeCorruptError("certification vector is malformed") from exc
  if vector.manifest["route_name"] != route_name:
    raise BridgeCorruptError("certification vector route changed")
  return vector


def _certification_domain_from_vector(
  vector: CertificationVector,
) -> dict[str, object]:
  results = vector.manifest.get("segment_results")
  if type(results) is not list or not results:
    raise BridgeCorruptError("certification vector lacks segment results")
  domains: list[object] = []
  for result in results:
    if type(result) is not dict:
      raise BridgeCorruptError("certification vector segment is malformed")
    domains.append(result.get("preparation_domain"))
  if type(domains[0]) is not dict or any(value != domains[0] for value in domains[1:]):
    raise BridgeCorruptError("certification vector domains disagree")
  source = domains[0]
  assert type(source) is dict
  try:
    provenance = {
      key: source[key]
      for key in _CERTIFICATION_DOMAIN_KEYS
    }
    runtime_identity = source["runtime_vehicle_bundle_sha256"]
  except KeyError as exc:
    raise BridgeCorruptError(
      "certification vector domain is incomplete",
    ) from exc
  return _certification_domain(
    provenance,
    runtime_vehicle_bundle_sha256=str(runtime_identity),
  )


def _architecture_vector_child(
  connection: object,
  engine: HistoricalLearningBackfill,
  route: RouteCandidate,
  expected_extractor_sha256: str,
  allocation_headroom_bytes: int,
) -> None:
  """Child-only bounded production-vector execution."""
  try:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    vm_size = None
    for line in Path("/proc/self/status").read_text().splitlines():
      if line.startswith("VmSize:"):
        vm_size = int(line.split()[1]) * 1024
        break
    if vm_size is None:
      raise RuntimeError("child virtual-memory baseline is unavailable")
    child_rss = _process_rss_bytes(os.getpid())
    if child_rss is None or child_rss >= allocation_headroom_bytes:
      raise RuntimeError("child resident-memory baseline exceeds its budget")
    address_limit = vm_size + (allocation_headroom_bytes - child_rss)
    resource.setrlimit(resource.RLIMIT_AS, (address_limit, address_limit))
    resource.setrlimit(
      resource.RLIMIT_FSIZE,
      (
        ARCHITECTURE_VERIFICATION_MAX_IPC_BYTES,
        ARCHITECTURE_VERIFICATION_MAX_IPC_BYTES,
      ),
    )
    cpu_limit = max(1, int(ARCHITECTURE_VERIFICATION_TIMEOUT_SECONDS))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 1))
    runtime = engine.runtime_factory()

    def started(_segment: RouteSegment, position: int, count: int) -> None:
      connection.send_bytes(canonical_json_bytes({
        "count": count,
        "position": position,
        "type": "progress",
      }))

    vector = build_certification_vector(
      route,
      extractor_path=engine.extractor_path,
      event_reader=engine.event_reader,
      car_params_decoder=engine.car_params_decoder,
      descriptor_registry=engine.descriptor_registry,
      route_bundle_factory=engine.route_bundle_factory,
      current_car_params=engine.current_car_params,
      current_bundle=runtime.runtime_bundle,
      expected_dongle_id=engine.expected_dongle_id,
      expected_extractor_sha256=expected_extractor_sha256,
      segment_started=started,
    )
    connection.send_bytes(b"vector\0" + vector.canonical_bytes)
  except BaseException as exc:
    # Exception text can contain paths or architecture-specific details.  The
    # parent publishes one stable diagnostic while logs retain only the class.
    try:
      connection.send_bytes(canonical_json_bytes({
        "error_type": type(exc).__name__,
        "type": "error",
      }))
    except BaseException:
      pass
  finally:
    try:
      connection.close()
    except BaseException:
      pass


def _process_rss_bytes(pid: int) -> int | None:
  try:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
      if line.startswith("VmRSS:"):
        fields = line.split()
        return int(fields[1]) * 1024
  except (FileNotFoundError, OSError, ValueError, IndexError):
    return None
  return None


def _copy_certification_source_segment(
  source: RouteSegment,
  destination: Path,
  *,
  owner: _PrivateScratchRoot,
  abort_requested: Callable[[], bool],
) -> RouteSegment:
  """Copy one immutable whole rlog into private certification scratch.

  The raw route may be purged independently of the learner.  The child never
  receives that pathname: this parent-side copy authenticates the selected
  inode, bytes, and manifest identity first, then fsyncs a private snapshot.
  """
  source_fd = -1
  destination_fd = -1
  temporary = destination.with_name(f".{destination.name}.partial")
  digest = hashlib.sha256()
  copied = 0
  try:
    before_path = source.path.lstat()
    if source.path.is_symlink() or not stat.S_ISREG(before_path.st_mode):
      raise ArchitectureVerificationError(
        "architecture_verification_source_changed",
      )
    if before_path.st_size != source.size_bytes:
      raise ArchitectureVerificationError(
        "architecture_verification_source_changed",
      )
    source_fd = os.open(
      source.path,
      os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    opened = os.fstat(source_fd)
    if (
      opened.st_dev != before_path.st_dev
      or opened.st_ino != before_path.st_ino
      or opened.st_size != source.size_bytes
      or not stat.S_ISREG(opened.st_mode)
    ):
      raise ArchitectureVerificationError(
        "architecture_verification_source_changed",
      )
    destination_fd = os.open(
      temporary,
      os.O_WRONLY | os.O_CREAT | os.O_EXCL
      | getattr(os, "O_NOFOLLOW", 0),
      0o600,
    )
    _register_private_scratch_file(owner, temporary, stable=False)
    while copied < source.size_bytes:
      check_abort(abort_requested)
      block = os.read(source_fd, min(1024 * 1024, source.size_bytes - copied))
      if not block:
        raise ArchitectureVerificationError(
          "architecture_verification_source_changed",
        )
      copied += len(block)
      digest.update(block)
      view = memoryview(block)
      while view:
        written = os.write(destination_fd, view)
        if written <= 0:
          raise ArchitectureVerificationError(
            "architecture_verification_snapshot_failed",
          )
        view = view[written:]
    if os.read(source_fd, 1):
      raise ArchitectureVerificationError(
        "architecture_verification_source_changed",
      )
    after_fd = os.fstat(source_fd)
    identity_before = (
      before_path.st_dev, before_path.st_ino, before_path.st_size,
      stat.S_IFMT(before_path.st_mode), before_path.st_mtime_ns,
    )
    identity_after = (
      after_fd.st_dev, after_fd.st_ino, after_fd.st_size,
      stat.S_IFMT(after_fd.st_mode), after_fd.st_mtime_ns,
    )
    if (
      identity_before != identity_after
      or digest.hexdigest() != source.sha256
    ):
      raise ArchitectureVerificationError(
        "architecture_verification_source_changed",
      )
    os.fsync(destination_fd)
    os.close(destination_fd)
    destination_fd = -1
    _rename_private_scratch_file(
      owner,
      temporary,
      destination,
      stable=True,
    )
    directory_fd = os.open(
      destination.parent,
      os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
      os.fsync(directory_fd)
    finally:
      os.close(directory_fd)
    snapshot_fd = os.open(
      destination,
      os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
      snapshot_info = os.fstat(snapshot_fd)
      snapshot_digest = hashlib.sha256()
      snapshot_size = 0
      while block := os.read(snapshot_fd, 1024 * 1024):
        snapshot_size += len(block)
        snapshot_digest.update(block)
      snapshot_after = os.fstat(snapshot_fd)
      if (
        not stat.S_ISREG(snapshot_info.st_mode)
        or snapshot_info.st_uid != os.geteuid()
        or stat.S_IMODE(snapshot_info.st_mode) & 0o077
        or snapshot_size != source.size_bytes
        or snapshot_digest.hexdigest() != source.sha256
        or snapshot_after.st_size != snapshot_info.st_size
        or snapshot_after.st_mtime_ns != snapshot_info.st_mtime_ns
      ):
        raise ArchitectureVerificationError(
          "architecture_verification_snapshot_failed",
        )
    finally:
      os.close(snapshot_fd)
  except ArchitectureVerificationError:
    raise
  except (OSError, ValueError) as exc:
    raise ArchitectureVerificationError(
      "architecture_verification_snapshot_failed",
    ) from exc
  finally:
    if destination_fd >= 0:
      os.close(destination_fd)
    if source_fd >= 0:
      os.close(source_fd)
    try:
      temporary.unlink()
    except FileNotFoundError:
      pass
  return RouteSegment(
    index=source.index,
    path=destination,
    sha256=source.sha256,
    size_bytes=source.size_bytes,
  )


def _snapshot_certification_route(
  route: RouteCandidate,
  *,
  parent: Path,
  abort_requested: Callable[[], bool],
) -> tuple[_PrivateScratchRoot, RouteCandidate]:
  """Snapshot only deterministically selected segments for the ARM child."""
  try:
    selected = certification_vector_selection(route)
  except CertificationVectorError as exc:
    raise ArchitectureVerificationError(
      "architecture_verification_snapshot_failed",
    ) from exc
  selected_indices = {segment.index for segment in selected}
  try:
    owner = _create_private_scratch(
      parent,
      prefix=CERTIFICATION_SOURCE_SNAPSHOT_PREFIX,
      required_bytes=sum(segment.size_bytes for segment in selected),
    )
  except BridgeUnavailableError as exc:
    raise ArchitectureVerificationError(
      "architecture_verification_snapshot_failed",
    ) from exc
  snapshots: dict[int, RouteSegment] = {}
  try:
    for segment in selected:
      destination = owner.path / f"rlog-{segment.index:03d}"
      snapshots[segment.index] = _copy_certification_source_segment(
        segment,
        destination,
        owner=owner,
        abort_requested=abort_requested,
      )
    cloned = tuple(
      snapshots.get(
        segment.index,
        RouteSegment(
          index=segment.index,
          # An absent private path makes accidental reads of unselected source
          # segments fail closed and is directly covered by the opener audit.
          path=owner.path / f"unselected-{segment.index:03d}.not-present",
          sha256=segment.sha256,
          size_bytes=segment.size_bytes,
        ),
      )
      for segment in route.segments
    )
    if {segment.index for segment in snapshots.values()} != selected_indices:
      raise AssertionError("certification snapshot selection changed")
    return owner, RouteCandidate(
      route_name=route.route_name,
      route_counter=route.route_counter,
      segments=cloned,
    )
  except BaseException:
    _cleanup_private_scratch(owner)
    raise


def _run_architecture_verification(
  *,
  engine: HistoricalLearningBackfill,
  route: RouteCandidate,
  expected_extractor_sha256: str,
  abort_requested: Callable[[], bool],
  progress: Callable[[int, int], None] | None = None,
  timeout_seconds: float = ARCHITECTURE_VERIFICATION_TIMEOUT_SECONDS,
  maximum_rss_bytes: int = ARCHITECTURE_VERIFICATION_MAX_RSS_BYTES,
) -> CertificationVector:
  """Run bounded ARM verification without risking the daemon process."""
  if timeout_seconds <= 0.0 or maximum_rss_bytes <= 0:
    raise ValueError("architecture-verification limits are invalid")
  context = multiprocessing.get_context("fork")
  parent_rss = _process_rss_bytes(os.getpid())
  if parent_rss is None:
    raise ArchitectureVerificationError(
      "architecture_verification_resource_accounting_unavailable",
    )
  child_budget = min(
    maximum_rss_bytes,
    ARCHITECTURE_VERIFICATION_MAX_COMBINED_RSS_BYTES - parent_rss,
  )
  if child_budget < ARCHITECTURE_VERIFICATION_MIN_CHILD_BUDGET_BYTES:
    raise ArchitectureVerificationError(
      "architecture_verification_resource_limit",
    )
  snapshot_parent = Path(engine.runtime_factory().artifact_paths.root)
  snapshot_owner, snapshot_route = _snapshot_certification_route(
    route,
    parent=snapshot_parent,
    abort_requested=abort_requested,
  )
  parent_connection = None
  child_connection = None
  process = None
  child_started = False
  try:
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
      target=_architecture_vector_child,
      args=(
        child_connection,
        engine,
        snapshot_route,
        expected_extractor_sha256,
        child_budget,
      ),
      daemon=True,
    )
    process.start()
    child_started = True
    child_connection.close()
    child_connection = None
    deadline = time.monotonic() + timeout_seconds
    final: bytes | None = None
    failure: str | None = None
    while process.is_alive() or parent_connection.poll():
      check_abort(abort_requested)
      if time.monotonic() >= deadline:
        failure = "architecture_verification_timeout"
        break
      rss = _process_rss_bytes(process.pid)
      current_parent_rss = _process_rss_bytes(os.getpid())
      if (
        rss is not None
        and current_parent_rss is not None
        and (
          rss > child_budget
          or rss + current_parent_rss
          > ARCHITECTURE_VERIFICATION_MAX_COMBINED_RSS_BYTES
        )
      ):
        failure = "architecture_verification_resource_limit"
        break
      if parent_connection.poll(0.05):
        try:
          payload = parent_connection.recv_bytes(
            ARCHITECTURE_VERIFICATION_MAX_IPC_BYTES,
          )
        except (EOFError, OSError):
          break
        if payload.startswith(b"vector\0"):
          if final is not None:
            failure = "architecture_verification_corrupt_output"
            break
          final = payload[len(b"vector\0"):]
          continue
        try:
          message = decode_canonical_json(payload, maximum_bytes=4096)
        except BridgeCorruptError:
          failure = "architecture_verification_corrupt_output"
          break
        if type(message) is not dict or message.get("type") not in {
          "progress", "error",
        }:
          failure = "architecture_verification_corrupt_output"
          break
        if message["type"] == "progress":
          if set(message) != {"count", "position", "type"}:
            failure = "architecture_verification_corrupt_output"
            break
          position = message["position"]
          count = message["count"]
          if (
            type(position) is not int
            or type(count) is not int
            or not 1 <= position <= count <= 3
          ):
            failure = "architecture_verification_corrupt_output"
            break
          if progress is not None:
            progress(position, count)
        else:
          if set(message) != {"error_type", "type"} or type(
            message.get("error_type")
          ) is not str:
            failure = "architecture_verification_corrupt_output"
          elif message["error_type"] == "MemoryError":
            failure = "architecture_verification_resource_limit"
          else:
            failure = "architecture_verification_child_failed"
          break
    if failure is not None:
      if process.is_alive():
        process.kill()
      process.join(timeout=2.0)
      raise ArchitectureVerificationError(failure)
    process.join(timeout=2.0)
    if process.is_alive():
      process.kill()
      process.join(timeout=2.0)
      raise ArchitectureVerificationError(
        "architecture_verification_timeout",
      )
    if final is None and process.exitcode in {
      -signal.SIGKILL,
      -signal.SIGXCPU,
      -signal.SIGXFSZ,
    }:
      raise ArchitectureVerificationError(
        "architecture_verification_resource_limit",
      )
    if process.exitcode != 0 or final is None:
      raise ArchitectureVerificationError(
        "architecture_verification_child_failed",
      )
    try:
      return CertificationVector.from_bytes(final)
    except CertificationVectorError as exc:
      raise ArchitectureVerificationError(
        "architecture_verification_corrupt_output",
      ) from exc
  finally:
    # Abort and unexpected parent failures are just as important as the named
    # failure paths: no extractor child may outlive its owning daemon request.
    if child_started and process is not None:
      if process.is_alive():
        process.terminate()
      process.join(timeout=2.0)
      if process.is_alive():
        process.kill()
        process.join(timeout=2.0)
    if child_connection is not None:
      child_connection.close()
    if parent_connection is not None:
      parent_connection.close()
    _cleanup_private_scratch(snapshot_owner)


def _select_certification_canary(
  domain_routes: list[tuple[RouteCandidate, _PreparedOutcome]],
  *,
  locally_available_route_names: frozenset[str],
) -> tuple[RouteCandidate, _PreparedOutcome] | None:
  """Choose the cheapest deterministic locally reproducible domain witness."""
  candidates = [
    item for item in domain_routes
    if item[0].route_name in locally_available_route_names
  ]
  if not candidates:
    return None

  def cost(
    item: tuple[RouteCandidate, _PreparedOutcome],
  ) -> tuple[int, int, int, str]:
    route, outcome = item
    descriptor = outcome.descriptor
    if descriptor is None:
      raise BridgeCorruptError("accepted canary candidate lacks a descriptor")
    return (
      descriptor.frame_count,
      sum(segment.size_bytes for segment in route.segments),
      len(route.segments),
      route.route_name,
    )

  return min(candidates, key=cost)


def _certification_compatibility(
  *,
  contract: Mapping[str, object],
  device_extractor_sha256: str,
  device_numerical_environment_sha256: str,
  preparation_implementation_sha256: str,
  worker_extractor_sha256: str,
  worker_implementation_commit: str,
  worker_implementation_sha256: str,
  worker_numerical_environment_sha256: str,
  worker_preparation_implementation_sha256: str,
  worker_instance_id: str,
) -> _CertificationBinding:
  # A worker instance identifies one authenticated transport session, not a
  # numerical implementation. Job status and artifact requests still verify
  # it on every transaction, but an otherwise identical service restart must
  # not invalidate a durable ARM implementation certificate.
  if (
    type(worker_instance_id) is not str
    or _SHA256_RE.fullmatch(worker_instance_id) is None
  ):
    raise BridgeIncompatibleError(
      "certification worker_instance_id is invalid",
    )
  compatibility: dict[str, object] = {
    "device_extractor_sha256": device_extractor_sha256,
    "device_numerical_environment_sha256": (
      device_numerical_environment_sha256
    ),
    "historical_descriptor_registry_sha256": (
      contract["historical_descriptor_registry_sha256"]
    ),
    "opendbc_commit": contract["opendbc_commit"],
    "panda_commit": contract["panda_commit"],
    "preparation_implementation_sha256": (
      preparation_implementation_sha256
    ),
    "runtime_identity_sha256": contract["runtime_identity_sha256"],
    "worker_extractor_sha256": worker_extractor_sha256,
    "worker_numerical_environment_sha256": (
      worker_numerical_environment_sha256
    ),
    "worker_preparation_implementation_sha256": (
      worker_preparation_implementation_sha256
    ),
  }
  # Commits and the effective current-build descriptor registry remain exact
  # live-session checks. They are retained for audit, but do not invalidate a
  # durable certificate when the content-addressed preparation core and every
  # numerical compatibility field are unchanged.
  audit: dict[str, object] = {
    "descriptor_registry_sha256": contract["descriptor_registry_sha256"],
    "source_commit": contract["source_commit"],
    "worker_implementation_commit": worker_implementation_commit,
    "worker_implementation_sha256": worker_implementation_sha256,
  }
  for key in (
    "opendbc_commit",
    "panda_commit",
  ):
    if type(compatibility[key]) is not str or _COMMIT_RE.fullmatch(
      str(compatibility[key]),
    ) is None:
      raise BridgeIncompatibleError(f"certification {key} is invalid")
  for key in (
    "device_extractor_sha256",
    "device_numerical_environment_sha256",
    "historical_descriptor_registry_sha256",
    "preparation_implementation_sha256",
    "runtime_identity_sha256",
    "worker_extractor_sha256",
    "worker_numerical_environment_sha256",
    "worker_preparation_implementation_sha256",
  ):
    if type(compatibility[key]) is not str or _SHA256_RE.fullmatch(
      str(compatibility[key]),
    ) is None:
      raise BridgeIncompatibleError(f"certification {key} is invalid")
  for key in ("source_commit", "worker_implementation_commit"):
    if type(audit[key]) is not str or _COMMIT_RE.fullmatch(
      str(audit[key]),
    ) is None:
      raise BridgeIncompatibleError(f"certification audit {key} is invalid")
  if (
    type(audit["descriptor_registry_sha256"]) is not str
    or _SHA256_RE.fullmatch(str(audit["descriptor_registry_sha256"])) is None
  ):
    raise BridgeIncompatibleError(
      "certification audit descriptor_registry_sha256 is invalid",
    )
  if (
    type(audit["worker_implementation_sha256"]) is not str
    or _SHA256_RE.fullmatch(str(audit["worker_implementation_sha256"])) is None
  ):
    raise BridgeIncompatibleError(
      "certification audit worker_implementation_sha256 is invalid",
    )
  return _CertificationBinding(compatibility=compatibility, audit=audit)


def _certification_identity(
  compatibility: Mapping[str, object],
  domain: Mapping[str, object],
  test_vector: Mapping[str, object],
) -> str:
  return hashlib.sha256(canonical_json_bytes({
    "compatibility": dict(compatibility),
    "domain": dict(domain),
    "schema_version": REMOTE_CERTIFICATION_SCHEMA_VERSION,
    "test_vector": dict(test_vector),
  })).hexdigest()


def _certification_root(artifact_root: Path) -> Path:
  root = artifact_root / REMOTE_CERTIFICATION_DIRECTORY
  try:
    root.mkdir(mode=0o700)
  except FileExistsError:
    pass
  except OSError as exc:
    raise BridgeUnavailableError(
      "off-device certification directory is unavailable",
    ) from exc
  try:
    root_stat = root.lstat()
  except OSError as exc:
    raise BridgeUnavailableError(
      "off-device certification directory cannot be inspected",
    ) from exc
  if (
    not stat.S_ISDIR(root_stat.st_mode)
    or root.is_symlink()
    or stat.S_IMODE(root_stat.st_mode) != 0o700
    or root_stat.st_uid != os.geteuid()
  ):
    raise BridgeUnavailableError(
      "off-device certification directory is not private",
    )
  return root


def _certificate_hmac(
  *,
  secret: bytes,
  unsigned: Mapping[str, object],
) -> str:
  return hmac.new(
    secret,
    canonical_json_bytes(dict(unsigned)),
    hashlib.sha256,
  ).hexdigest()


def _validate_certification_audit(payload: object) -> dict[str, object]:
  if type(payload) is not dict or set(payload) != {
    "descriptor_registry_sha256",
    "source_commit",
    "worker_implementation_commit",
    "worker_implementation_sha256",
  }:
    raise BridgeUnavailableError("cached certification audit is invalid")
  if any(
    type(payload[key]) is not str
    or _COMMIT_RE.fullmatch(str(payload[key])) is None
    for key in ("source_commit", "worker_implementation_commit")
  ):
    raise BridgeUnavailableError("cached certification audit is invalid")
  descriptor = payload["descriptor_registry_sha256"]
  if type(descriptor) is not str or _SHA256_RE.fullmatch(descriptor) is None:
    raise BridgeUnavailableError("cached certification audit is invalid")
  worker_tree = payload["worker_implementation_sha256"]
  if type(worker_tree) is not str or _SHA256_RE.fullmatch(worker_tree) is None:
    raise BridgeUnavailableError("cached certification audit is invalid")
  return dict(payload)


def _load_certification(
  *,
  root: Path,
  identity: str,
  compatibility: Mapping[str, object],
  domain: Mapping[str, object],
  expected_test_vector: Mapping[str, object],
  secret: bytes,
) -> bool:
  path = root / f"{identity}.json"
  try:
    file_stat = path.lstat()
  except FileNotFoundError:
    return False
  except OSError as exc:
    raise BridgeUnavailableError(
      "off-device certification cannot be inspected",
    ) from exc
  if (
    not stat.S_ISREG(file_stat.st_mode)
    or path.is_symlink()
    or stat.S_IMODE(file_stat.st_mode) != 0o600
    or file_stat.st_uid != os.geteuid()
    or file_stat.st_size > 64 * 1024
  ):
    raise BridgeUnavailableError("cached off-device certification is unsafe")
  try:
    encoded = path.read_bytes()
    payload = decode_canonical_json(encoded, maximum_bytes=64 * 1024)
  except (OSError, BridgeCorruptError) as exc:
    raise BridgeUnavailableError(
      "cached off-device certification is unreadable",
    ) from exc
  test_vector = payload.get("test_vector") if type(payload) is dict else None
  if (
    type(payload) is not dict
    or set(payload) != {
      "audit",
      "compatibility",
      "domain",
      "hmac_sha256",
      "schema_version",
      "test_vector",
    }
    or encoded != canonical_json_bytes(payload)
    or payload["schema_version"] != REMOTE_CERTIFICATION_SCHEMA_VERSION
    or payload["compatibility"] != dict(compatibility)
    or payload["domain"] != dict(domain)
    or type(payload["hmac_sha256"]) is not str
    or _SHA256_RE.fullmatch(payload["hmac_sha256"]) is None
    or type(test_vector) is not dict
  ):
    raise BridgeUnavailableError("cached off-device certification is invalid")
  _validate_certification_audit(payload["audit"])
  if test_vector != dict(expected_test_vector) or set(test_vector) != {
    "certification_vector_sha256",
    "route_name",
    "selected_controls_witnesses",
    "selected_segment_count",
    "selection_identity_sha256",
    "vector_identity_sha256",
  }:
    raise BridgeUnavailableError(
      "cached off-device certification test vector is malformed",
    )
  route_name = test_vector["route_name"]
  if (
    type(route_name) is not str
    or _ROUTE_RE.fullmatch(route_name) is None
    or type(test_vector["selected_controls_witnesses"]) is not int
    or not 1 <= test_vector["selected_controls_witnesses"] <= 30_000
    or type(test_vector["selected_segment_count"]) is not int
    or not 1 <= test_vector["selected_segment_count"] <= 3
  ):
    raise BridgeUnavailableError(
      "cached off-device certification test vector is invalid",
    )
  for key in (
    "certification_vector_sha256",
    "selection_identity_sha256",
    "vector_identity_sha256",
  ):
    if type(test_vector[key]) is not str or _SHA256_RE.fullmatch(
      test_vector[key],
    ) is None:
      raise BridgeUnavailableError(
        "cached off-device certification content hash is invalid",
      )
  unsigned = dict(payload)
  signature = str(unsigned.pop("hmac_sha256"))
  if not hmac.compare_digest(
    signature,
    _certificate_hmac(secret=secret, unsigned=unsigned),
  ):
    raise BridgeUnavailableError(
      "cached off-device certification authentication failed",
    )
  if _certification_identity(
    compatibility,
    domain,
    expected_test_vector,
  ) != identity:
    raise BridgeUnavailableError("cached off-device certification is misbound")
  return True


def _write_certification(
  *,
  root: Path,
  identity: str,
  compatibility: Mapping[str, object],
  audit: Mapping[str, object],
  domain: Mapping[str, object],
  test_vector: Mapping[str, object],
  secret: bytes,
  abort_requested: Callable[[], bool],
) -> None:
  unsigned: dict[str, object] = {
    "audit": dict(audit),
    "compatibility": dict(compatibility),
    "domain": dict(domain),
    "schema_version": REMOTE_CERTIFICATION_SCHEMA_VERSION,
    "test_vector": dict(test_vector),
  }
  _validate_certification_audit(unsigned["audit"])
  payload = dict(unsigned)
  payload["hmac_sha256"] = _certificate_hmac(
    secret=secret,
    unsigned=unsigned,
  )
  encoded = canonical_json_bytes(payload)
  path = root / f"{identity}.json"
  if path.exists():
    if not _load_certification(
      root=root,
      identity=identity,
      compatibility=compatibility,
      domain=domain,
      expected_test_vector=test_vector,
      secret=secret,
    ):
      raise AssertionError("existing certification disappeared")
    return
  descriptor = -1
  temporary: Path | None = None
  try:
    descriptor, name = tempfile.mkstemp(
      dir=root,
      prefix=f".{identity}.",
      suffix=".tmp",
    )
    temporary = Path(name)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
      descriptor = -1
      stream.write(encoded)
      stream.flush()
      os.fsync(stream.fileno())
    check_abort(abort_requested)
    os.replace(temporary, path)
    temporary = None
    directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
      os.fsync(directory_fd)
    finally:
      os.close(directory_fd)
  except OSError as exc:
    raise BridgeUnavailableError(
      "off-device certification could not be published",
    ) from exc
  finally:
    if descriptor >= 0:
      os.close(descriptor)
    if temporary is not None:
      try:
        temporary.unlink()
      except FileNotFoundError:
        pass


def _rejection_test_route(route: RouteCandidate) -> dict[str, object]:
  return {
    "route_name": route.route_name,
    "segments": [segment.to_ledger_dict() for segment in route.segments],
  }


def _rejection_certification_identity(
  compatibility: Mapping[str, object],
  route: RouteCandidate,
) -> str:
  return hashlib.sha256(canonical_json_bytes({
    "compatibility": dict(compatibility),
    "route": _rejection_test_route(route),
    "schema_version": REMOTE_CERTIFICATION_SCHEMA_VERSION,
    "type": "rejection",
  })).hexdigest()


def _load_rejection_certification(
  *,
  root: Path,
  identity: str,
  compatibility: Mapping[str, object],
  route: RouteCandidate,
  reason: str,
  message: str,
  secret: bytes,
) -> bool:
  path = root / f"{identity}.json"
  try:
    file_stat = path.lstat()
  except FileNotFoundError:
    return False
  except OSError as exc:
    raise BridgeUnavailableError(
      "off-device rejection certification cannot be inspected",
    ) from exc
  if (
    not stat.S_ISREG(file_stat.st_mode)
    or path.is_symlink()
    or stat.S_IMODE(file_stat.st_mode) != 0o600
    or file_stat.st_uid != os.geteuid()
    or file_stat.st_size > 64 * 1024
  ):
    raise BridgeUnavailableError(
      "cached off-device rejection certification is unsafe",
    )
  try:
    encoded = path.read_bytes()
    payload = decode_canonical_json(encoded, maximum_bytes=64 * 1024)
  except (OSError, BridgeCorruptError) as exc:
    raise BridgeUnavailableError(
      "cached off-device rejection certification is unreadable",
    ) from exc
  expected_rejection = {"message": message, "reason": reason}
  if (
    type(payload) is not dict
    or set(payload) != {
      "audit",
      "compatibility",
      "hmac_sha256",
      "rejection",
      "route",
      "schema_version",
      "type",
    }
    or encoded != canonical_json_bytes(payload)
    or payload["schema_version"] != REMOTE_CERTIFICATION_SCHEMA_VERSION
    or payload["type"] != "rejection"
    or payload["compatibility"] != dict(compatibility)
    or payload["route"] != _rejection_test_route(route)
    or payload["rejection"] != expected_rejection
    or type(payload["hmac_sha256"]) is not str
    or _SHA256_RE.fullmatch(payload["hmac_sha256"]) is None
  ):
    raise BridgeUnavailableError(
      "cached off-device rejection certification is invalid",
    )
  _validate_certification_audit(payload["audit"])
  unsigned = dict(payload)
  signature = str(unsigned.pop("hmac_sha256"))
  if not hmac.compare_digest(
    signature,
    _certificate_hmac(secret=secret, unsigned=unsigned),
  ):
    raise BridgeUnavailableError(
      "cached off-device rejection certification authentication failed",
    )
  return True


def _write_rejection_certification(
  *,
  root: Path,
  identity: str,
  compatibility: Mapping[str, object],
  audit: Mapping[str, object],
  route: RouteCandidate,
  reason: str,
  message: str,
  secret: bytes,
  abort_requested: Callable[[], bool],
) -> None:
  unsigned: dict[str, object] = {
    "audit": dict(audit),
    "compatibility": dict(compatibility),
    "rejection": {"message": message, "reason": reason},
    "route": _rejection_test_route(route),
    "schema_version": REMOTE_CERTIFICATION_SCHEMA_VERSION,
    "type": "rejection",
  }
  _validate_certification_audit(unsigned["audit"])
  payload = dict(unsigned)
  payload["hmac_sha256"] = _certificate_hmac(
    secret=secret,
    unsigned=unsigned,
  )
  encoded = canonical_json_bytes(payload)
  path = root / f"{identity}.json"
  if path.exists():
    if not _load_rejection_certification(
      root=root,
      identity=identity,
      compatibility=compatibility,
      route=route,
      reason=reason,
      message=message,
      secret=secret,
    ):
      raise AssertionError("existing rejection certification disappeared")
    return
  descriptor = -1
  temporary: Path | None = None
  try:
    descriptor, name = tempfile.mkstemp(
      dir=root,
      prefix=f".{identity}.",
      suffix=".tmp",
    )
    temporary = Path(name)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
      descriptor = -1
      stream.write(encoded)
      stream.flush()
      os.fsync(stream.fileno())
    check_abort(abort_requested)
    os.replace(temporary, path)
    temporary = None
    directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
      os.fsync(directory_fd)
    finally:
      os.close(directory_fd)
  except OSError as exc:
    raise BridgeUnavailableError(
      "off-device rejection certification could not be published",
    ) from exc
  finally:
    if descriptor >= 0:
      os.close(descriptor)
    if temporary is not None:
      try:
        temporary.unlink()
      except FileNotFoundError:
        pass


def _certify_preparation_domains(
  *,
  engine: HistoricalLearningBackfill,
  plan: RemoteRoutePlan,
  scratch_directory: Path,
  outcomes: Mapping[tuple[int, str], _PreparedOutcome],
  contract: Mapping[str, object],
  device_numerical_environment_sha256: str,
  preparation_implementation_sha256: str,
  worker_extractor_sha256: str,
  worker_implementation_commit: str,
  worker_implementation_sha256: str,
  worker_numerical_environment_sha256: str,
  worker_preparation_implementation_sha256: str,
  worker_instance_id: str,
  secret: bytes,
  abort_requested: Callable[[], bool],
  progress: Callable[
    [int, int, int, int, int, str | None, int | None, int | None],
    None,
  ] | None = None,
  architecture_verifier: Callable[..., CertificationVector] = (
    _run_architecture_verification
  ),
) -> dict[tuple[int, str], _PreparedOutcome]:
  """Certify each accepted preparation-compatibility domain on ARM once."""
  device_extractor_sha256 = _hash_regular_file(
    engine.extractor_path,
    abort_requested=abort_requested,
  )
  binding = _certification_compatibility(
    contract=contract,
    device_extractor_sha256=device_extractor_sha256,
    device_numerical_environment_sha256=(
      device_numerical_environment_sha256
    ),
    preparation_implementation_sha256=(
      preparation_implementation_sha256
    ),
    worker_extractor_sha256=worker_extractor_sha256,
    worker_implementation_commit=worker_implementation_commit,
    worker_implementation_sha256=worker_implementation_sha256,
    worker_numerical_environment_sha256=(
      worker_numerical_environment_sha256
    ),
    worker_preparation_implementation_sha256=(
      worker_preparation_implementation_sha256
    ),
    worker_instance_id=worker_instance_id,
  )
  compatibility = binding.compatibility
  certification_root = _certification_root(
    engine.runtime_factory().artifact_paths.root,
  )
  certified = dict(outcomes)
  grouped: dict[
    str,
    tuple[dict[str, object], list[tuple[RouteCandidate, _PreparedOutcome]]],
  ] = {}
  rejected: list[tuple[RouteCandidate, _PreparedOutcome]] = []
  certified_route_count = 0
  for route in plan.replay_candidates:
    outcome = outcomes[(1, route.route_name)]
    if outcome.rejection_reason is not None:
      if route.route_name not in plan.locally_available_route_names:
        raise BridgeUnavailableError(
          "rejected remote route has no local certification bytes",
        )
      rejection_message = outcome.rejection_message
      if rejection_message is None:
        raise BridgeCorruptError("rejected outcome lacks a message")
      rejected.append((route, outcome))
      continue
    descriptor = outcome.descriptor
    if descriptor is None:
      raise BridgeCorruptError("accepted remote outcome lacks a descriptor")
    verification_descriptor = outcomes[(2, route.route_name)].descriptor
    if (
      verification_descriptor is None
      or descriptor.sha256 != verification_descriptor.sha256
      or descriptor.size_bytes != verification_descriptor.size_bytes
      or descriptor.frame_count != verification_descriptor.frame_count
    ):
      raise BridgeCorruptError(
        "accepted preparation authorities disagree before certification",
      )
    vector_descriptor = outcome.certification_vector
    verification_vector_descriptor = outcomes[
      (2, route.route_name)
    ].certification_vector
    if (
      vector_descriptor is None
      or verification_vector_descriptor is None
      or vector_descriptor.sha256 != verification_vector_descriptor.sha256
      or vector_descriptor.size_bytes != verification_vector_descriptor.size_bytes
      or vector_descriptor.selection_identity_sha256
      != verification_vector_descriptor.selection_identity_sha256
    ):
      raise BridgeCorruptError(
        "certification-vector authorities disagree before certification",
      )
    vector = _load_downloaded_certification_vector(
      scratch_directory,
      vector_descriptor,
      route_name=route.route_name,
      abort_requested=abort_requested,
    )
    domain = _certification_domain_from_vector(vector)
    identity = hashlib.sha256(canonical_json_bytes({
      "domain": domain,
      "type": "certification-domain-group-v1",
    })).hexdigest()
    existing = grouped.get(identity)
    if existing is None:
      grouped[identity] = (domain, [(route, outcome)])
    else:
      if existing[0] != domain:
        raise BridgeCorruptError("certification domain identity collided")
      existing[1].append((route, outcome))

  certified_identities: dict[str, str] = {}
  if progress is not None:
    progress(
      0,
      certified_route_count,
      len(grouped),
      len(plan.replay_candidates),
      0,
      None,
      None,
      None,
    )
  if rejected:
    # A rejected complete PC artifact has no bounded numerical output to
    # compare.  Re-running the full route on ARM is the exact OOM path this
    # protocol replaces, so rejection certification is explicitly fail-closed
    # rather than a hidden heavyweight local fallback.
    raise ArchitectureVerificationError(
      "architecture_verification_rejection_unprovable",
    )

  for domain_index, (
    group_identity,
    (domain, domain_routes),
  ) in enumerate(grouped.items(), start=1):
    selected = _select_certification_canary(
      domain_routes,
      locally_available_route_names=plan.locally_available_route_names,
    )
    if selected is None:
      raise BridgeUnavailableError(
        "accepted remote preparation domain has no local certification route",
      )
    route, remote_outcome = selected
    route_display_identity = route_identity_sha256(route.route_name)
    remote_vector_descriptor = remote_outcome.certification_vector
    if remote_vector_descriptor is None:
      raise BridgeCorruptError("accepted canary lacks a certification vector")
    remote_vector = _load_downloaded_certification_vector(
      scratch_directory,
      remote_vector_descriptor,
      route_name=route.route_name,
      abort_requested=abort_requested,
    )
    vector_identity = remote_vector.manifest.get("vector_identity_sha256")
    if type(vector_identity) is not str or _SHA256_RE.fullmatch(vector_identity) is None:
      raise BridgeCorruptError("certification vector identity is invalid")
    test_vector = {
      "certification_vector_sha256": remote_vector.sha256,
      "route_name": route.route_name,
      "selected_controls_witnesses": remote_vector_descriptor.selected_controls_witnesses,
      "selected_segment_count": remote_vector_descriptor.selected_segment_count,
      "selection_identity_sha256": remote_vector_descriptor.selection_identity_sha256,
      "vector_identity_sha256": vector_identity,
    }
    identity = _certification_identity(
      compatibility,
      domain,
      test_vector,
    )
    if progress is not None:
      progress(
        len(certified_identities),
        certified_route_count,
        len(grouped),
        len(plan.replay_candidates),
        domain_index,
        route_display_identity,
        None,
        None,
      )
    if _load_certification(
      root=certification_root,
      identity=identity,
      compatibility=compatibility,
      domain=domain,
      expected_test_vector=test_vector,
      secret=secret,
    ):
      certified_identities[group_identity] = identity
      certified_route_count += len(domain_routes)
      if progress is not None:
        progress(
          len(certified_identities),
          certified_route_count,
          len(grouped),
          len(plan.replay_candidates),
          len(certified_identities),
          None,
          None,
          None,
        )
      continue
    check_abort(abort_requested)

    def vector_progress(position: int, count: int) -> None:
      if progress is not None:
        progress(
          len(certified_identities),
          certified_route_count,
          len(grouped),
          len(plan.replay_candidates),
          domain_index,
          route_display_identity,
          position,
          count,
        )

    local_vector = architecture_verifier(
      engine=engine,
      route=route,
      expected_extractor_sha256=device_extractor_sha256,
      abort_requested=abort_requested,
      progress=vector_progress,
    )
    if local_vector.canonical_bytes != remote_vector.canonical_bytes:
      raise ArchitectureVerificationError(
        "architecture_verification_byte_mismatch",
      )
    _write_certification(
      root=certification_root,
      identity=identity,
      compatibility=compatibility,
      audit=binding.audit,
      domain=domain,
      test_vector=test_vector,
      secret=secret,
      abort_requested=abort_requested,
    )
    certified_identities[group_identity] = identity
    certified_route_count += len(domain_routes)
    if progress is not None:
      progress(
        len(certified_identities),
        certified_route_count,
        len(grouped),
        len(plan.replay_candidates),
        len(certified_identities),
        None,
        None,
        None,
      )

  for group_identity, (_domain, domain_routes) in grouped.items():
    identity = certified_identities.get(group_identity)
    if identity is None:
      raise BridgeUnavailableError(
        "accepted remote preparation domain is not certified",
      )
    route_names = {route.route_name for route, _outcome in domain_routes}
    for authority in (1, 2):
      for route_name in route_names:
        outcome = certified[(authority, route_name)]
        certified[(authority, route_name)] = _PreparedOutcome(
          descriptor=outcome.descriptor,
          certification_vector=outcome.certification_vector,
          rejection_reason=outcome.rejection_reason,
          rejection_message=outcome.rejection_message,
          certification_identity_sha256=identity,
        )
  if certified_route_count != len(plan.replay_candidates):
    raise BridgeUnavailableError(
      "remote preparation certification coverage is incomplete",
    )
  return certified


def _prior_generation_extractor_sha256(
  engine: HistoricalLearningBackfill,
) -> str:
  """Return the authenticated extractor named by the active generation.

  A late-only ledger update decodes no route. It must carry forward the
  extractor that produced the evidence it republishes, rather than naming an
  unused ARM or PC executable. Any ambiguity abandons remote orchestration and
  lets the complete local transaction apply its normal restore diagnostics.
  """
  try:
    runtime = engine.runtime_factory()
    paths = runtime.artifact_paths.resolved()
    encoded = paths.backfill_provenance.read_bytes()
    payload = decode_canonical_json(encoded, maximum_bytes=64 * 1024)
  except (OSError, ValueError, BridgeCorruptError) as exc:
    raise BridgeUnavailableError(
      "prior generation extractor provenance is unavailable",
    ) from exc
  expected_keys = {
    "canonical_join_schema_version",
    "descriptor_registry_sha256",
    "extractor_schema_version",
    "extractor_sha256",
    "ledger_sha256",
    "previous_generation_sha256",
    "runtime_identity_sha256",
    "schema_version",
    "source",
  }
  extractor = payload.get("extractor_sha256") if type(payload) is dict else None
  if (
    type(payload) is not dict
    or set(payload) != expected_keys
    or encoded != canonical_json_bytes(payload)
    or payload["schema_version"] != BACKFILL_PROVENANCE_SCHEMA_VERSION
    or payload["source"] != "complete_full_rlog_only"
    or payload["runtime_identity_sha256"]
    != runtime.runtime_bundle.calibration_identity_sha256
    or type(extractor) is not str
    or _SHA256_RE.fullmatch(extractor) is None
  ):
    raise BridgeUnavailableError(
      "prior generation extractor provenance is invalid",
    )
  return extractor


class _RemoteProgressProjector:
  """Restamp signed PC progress onto the device's display-only clocks."""

  def __init__(
    self,
    *,
    engine: HistoricalLearningBackfill,
    routes: tuple[RouteCandidate, ...],
    offdevice_progress: OffdeviceProgressPublisher | None = None,
  ) -> None:
    self.engine = engine
    self.routes = routes
    self.offdevice_progress = offdevice_progress
    self.route_indexes = {
      route.route_name: index
      for index, route in enumerate(routes, start=1)
    }
    self.route_segment_prefixes: dict[str, int] = {}
    self.route_byte_prefixes: dict[str, int] = {}
    segments = 0
    source_bytes = 0
    for route in routes:
      self.route_segment_prefixes[route.route_name] = segments
      self.route_byte_prefixes[route.route_name] = source_bytes
      segments += len(route.segments)
      source_bytes += sum(segment.size_bytes for segment in route.segments)
    self.segments_per_authority = segments
    self.bytes_per_authority = source_bytes
    self._last_coordinate: tuple[int, int, int] = (0, 0, 0)
    self._status_authority = 0

  def _new_device_operation(self) -> None:
    runtime = self.engine.runtime_factory()
    self.engine.operation_status.publish(
      state=LearningOperationState.BACKFILLING,
      diagnostic="scanning_routes",
      new_operation=True,
      runtime_identity_sha256=(
        runtime.runtime_bundle.calibration_identity_sha256
      ),
      vehicle_identity=runtime.runtime_bundle.vehicle_identity,
    )
    if self.engine.backfill_progress is not None:
      self.engine.backfill_progress.clear()

  def start(self) -> None:
    self._new_device_operation()
    if self.offdevice_progress is not None:
      self.offdevice_progress.publish(
        phase=OffdeviceProgressPhase.REMOTE_PROCESSING,
        new_session=True,
        remote_authority_count=2,
        remote_authority_index=0,
        remote_route_count=len(self.routes),
        remote_route_index=0,
      )

  def update(self, progress: Mapping[str, object]) -> None:
    authority = int(progress["authority_index"])
    route_index = int(progress["route_index"])
    segment_index = int(progress["segment_index"])
    if authority == 0 or route_index == 0 or segment_index == 0:
      return
    route_name = str(progress["route_name"])
    if (
      authority not in (1, 2)
      or route_index != self.route_indexes.get(route_name)
      or route_index > len(self.routes)
    ):
      raise BridgeCorruptError("worker progress does not match frozen routes")
    route = self.routes[route_index - 1]
    if (
      int(progress["route_count"]) != len(self.routes)
      or int(progress["segment_count"]) != len(route.segments)
      or not 1 <= segment_index <= len(route.segments)
    ):
      raise BridgeCorruptError("worker progress dimensions changed")
    coordinate = (authority, route_index, segment_index)
    if coordinate < self._last_coordinate:
      raise BridgeCorruptError("worker progress moved backward")
    self._last_coordinate = coordinate
    if self.offdevice_progress is not None:
      self.offdevice_progress.publish(
        phase=OffdeviceProgressPhase.REMOTE_PROCESSING,
        remote_authority_count=2,
        remote_authority_index=authority,
        remote_route_count=len(self.routes),
        remote_route_index=route_index,
      )
    if authority != self._status_authority:
      if self._status_authority not in (0, authority - 1):
        raise BridgeCorruptError("worker preparation authority jumped")
      if self._status_authority != 0:
        # LearningOperationStatus route counters are operation-local and do
        # not know about replay passes. Bind pass two to a fresh display-only
        # operation so route 1 never appears to move a prior route counter
        # backward. This changes no evidence or job ordering.
        self._new_device_operation()
      self._status_authority = authority

    runtime = self.engine.runtime_factory()
    self.engine.operation_status.publish(
      state=LearningOperationState.BACKFILLING,
      diagnostic="replaying_route",
      current_route_identity=route_identity_sha256(route_name),
      current_route_index=route_index,
      total_route_count=len(self.routes),
      runtime_identity_sha256=(
        runtime.runtime_bundle.calibration_identity_sha256
      ),
      vehicle_identity=runtime.runtime_bundle.vehicle_identity,
    )
    if self.engine.backfill_progress is None:
      return
    segment_prefix = self.route_segment_prefixes[route_name]
    byte_prefix = self.route_byte_prefixes[route_name]
    completed_segments = (
      (authority - 1) * self.segments_per_authority
      + segment_prefix
      + segment_index - 1
    )
    completed_bytes = (
      (authority - 1) * self.bytes_per_authority
      + byte_prefix
      + sum(
        segment.size_bytes
        for segment in route.segments[:segment_index - 1]
      )
    )
    status = self.engine.operation_status.last_payload
    if status is None:
      raise BridgeCorruptError("remote progress has no device operation")
    self.engine.backfill_progress.publish(
      operation_status=status,
      phase=BackfillProgressPhase.READING_SEGMENT,
      pass_index=authority,
      pass_count=2,
      current_route_identity=route.display_identity,
      current_route_index=route_index,
      total_route_count=len(self.routes),
      current_segment_index=segment_index,
      current_route_segment_count=len(route.segments),
      completed_replay_segment_count=completed_segments,
      total_replay_segment_count=2 * self.segments_per_authority,
      completed_work_units=completed_bytes,
      total_work_units=2 * self.bytes_per_authority,
      approximate_remaining_seconds=None,
    )

  def complete(self) -> None:
    if self.engine.backfill_progress is None or not self.routes:
      return
    status = self.engine.operation_status.last_payload
    if status is None:
      raise BridgeCorruptError("remote completion has no device operation")
    self.engine.backfill_progress.publish(
      operation_status=status,
      phase=BackfillProgressPhase.COMPARING,
      pass_index=2,
      pass_count=2,
      current_route_identity=None,
      current_route_index=None,
      total_route_count=len(self.routes),
      current_segment_index=None,
      current_route_segment_count=None,
      completed_replay_segment_count=2 * self.segments_per_authority,
      total_replay_segment_count=2 * self.segments_per_authority,
      completed_work_units=2 * self.bytes_per_authority,
      total_work_units=2 * self.bytes_per_authority,
      approximate_remaining_seconds=None,
    )


class RemotePreparationSession:
  """Validated local spool set consumed once by the device replay engine."""

  def __init__(
    self,
    *,
    base_engine: HistoricalLearningBackfill,
    plan: RemoteRoutePlan,
    scratch_directory: Path | None,
    outcomes: Mapping[tuple[int, str], _PreparedOutcome],
    worker_extractor_sha256: str | None,
    scratch_owner: _PrivateScratchRoot | None = None,
  ) -> None:
    self.base_engine = base_engine
    self.plan = plan
    self.scratch_directory = scratch_directory
    if scratch_directory is None:
      if scratch_owner is not None:
        raise BackfillError(
          "backfill_spool_invalid",
          "remote scratch owner exists without a directory",
        )
      self._scratch_owner = None
    else:
      try:
        owner = (
          _private_scratch_from_path(
            scratch_directory,
            prefix=REMOTE_SCRATCH_PREFIX,
          )
          if scratch_owner is None
          else scratch_owner
        )
      except BridgeUnavailableError as exc:
        raise BackfillError(
          "backfill_spool_invalid",
          str(exc),
        ) from exc
      if owner.path != scratch_directory:
        raise BackfillError(
          "backfill_spool_invalid",
          "remote scratch owner path disagrees",
        )
      self._scratch_owner = owner
    self.outcomes = dict(outcomes)
    self.worker_extractor_sha256 = worker_extractor_sha256
    self.unverified_exclusions = tuple(
      getattr(plan, "unverified_exclusions", ()),
    )
    self._closed = False

  def _source(
    self,
    authority_index: int,
    route: RouteCandidate,
    abort_requested: Callable[[], bool],
  ) -> PreparedRouteSpool:
    check_abort(abort_requested)
    outcome = self.outcomes.get((authority_index, route.route_name))
    if outcome is None:
      raise BackfillError(
        "backfill_spool_invalid",
        "remote preparation outcome is absent",
      )
    if outcome.rejection_reason is not None:
      if (
        outcome.certification_identity_sha256 is None
        or _SHA256_RE.fullmatch(
          outcome.certification_identity_sha256,
        ) is None
      ):
        raise BackfillError(
          "backfill_spool_invalid",
          "remote route rejection is not ARM-certified",
        )
      assert outcome.rejection_message is not None
      raise RouteRejected(
        outcome.rejection_reason,
        outcome.rejection_message,
      )
    descriptor = outcome.descriptor
    if (
      descriptor is None
      or self.scratch_directory is None
      or outcome.certification_identity_sha256 is None
      or _SHA256_RE.fullmatch(
        outcome.certification_identity_sha256,
      ) is None
    ):
      raise BackfillError(
        "backfill_spool_invalid",
        "remote preparation descriptor or certification is absent",
      )
    try:
      return open_prepared_route_spool(
        self.scratch_directory,
        descriptor,
        expected_route_name=route.route_name,
        max_frames=MAXIMUM_ROUTE_FRAMES,
      )
    except SpoolFormatError as exc:
      raise BackfillError(
        "backfill_spool_invalid",
        f"remote prepared route is invalid: {exc}",
      ) from exc

  def build_engine(self) -> HistoricalLearningBackfill:
    engine = self.base_engine
    remote_engine = HistoricalLearningBackfill(
      log_root=engine.log_root,
      extractor_path=engine.extractor_path,
      current_car_params=engine.current_car_params,
      runtime_factory=engine.runtime_factory,
      route_bundle_factory=engine.route_bundle_factory,
      car_params_decoder=engine.car_params_decoder,
      descriptor_registry=engine.descriptor_registry,
      expected_dongle_id=engine.expected_dongle_id,
      operation_status=engine.operation_status,
      backfill_progress=engine.backfill_progress,
      progress_monotonic_ns=engine.progress_monotonic_ns,
      abort_requested=engine.abort_requested,
      pending_route_identity=engine.pending_route_identity,
      # Only the two causal A/A authorities run on ARM. Route preparation has
      # already happened twice on the four-worker PC and is never shared.
      replay_worker_count=2,
      route_discovery=lambda abort: self.plan.discovery,
      prepared_route_source=self._source,
      preparation_extractor_sha256=self.worker_extractor_sha256,
      event_reader=engine.event_reader,
    )
    remote_engine.preserve_pending_route_quiescence(
      engine.pending_route_quiescence_observed,
    )
    return remote_engine

  def preserve_transaction_state(
    self,
    remote_engine: HistoricalLearningBackfill,
  ) -> None:
    self.base_engine.preserve_pending_route_quiescence(
      remote_engine.pending_route_quiescence_observed,
    )

  def close(self) -> None:
    if self._closed:
      return
    self._closed = True
    owner = self._scratch_owner
    if owner is None:
      return
    try:
      _cleanup_private_scratch(owner)
    except BridgeUnavailableError as exc:
      raise BackfillError(
        "backfill_spool_invalid",
        str(exc),
      ) from exc

  def __enter__(self) -> RemotePreparationSession:
    return self

  def __exit__(self, *_args: object) -> None:
    self.close()


def _upload_missing_routes(
  *,
  client: OffdeviceBridgeClient,
  routes: tuple[RouteCandidate, ...],
  dongle_id: str,
) -> None:
  for route in routes:
    for segment in route.segments:
      client.upload_segment(
        dongle_id=dongle_id,
        route_name=route.route_name,
        segment_index=segment.index,
        segment_path=segment.path,
        segment_size_bytes=segment.size_bytes,
        segment_sha256=segment.sha256,
      )
    # A completed segment is still private staging. Only this exact ordered
    # route manifest lets the worker atomically publish the whole route into
    # its durable inventory, so a network/onroad interruption can never turn a
    # contiguous prefix into a falsely complete route.
    client.commit_route(
      dongle_id=dongle_id,
      route_name=route.route_name,
      segments=[{
        "index": segment.index,
        "sha256": segment.sha256,
        "size_bytes": segment.size_bytes,
      } for segment in route.segments],
    )


def _validated_outcomes(
  *,
  status: Mapping[str, object],
  routes: tuple[RouteCandidate, ...],
) -> dict[tuple[int, str], dict[str, object]]:
  payload = status.get("outcomes")
  if type(payload) is not list:
    raise BridgeCorruptError("completed job outcomes are malformed")
  by_key: dict[tuple[int, str], dict[str, object]] = {}
  for raw in payload:
    if type(raw) is not dict:
      raise BridgeCorruptError("completed job outcome is malformed")
    key = (int(raw["authority_index"]), str(raw["route_name"]))
    if key in by_key:
      raise BridgeCorruptError("completed job repeats an outcome")
    by_key[key] = raw
  expected = {
    (authority, route.route_name)
    for authority in (1, 2)
    for route in routes
  }
  if set(by_key) != expected:
    raise BridgeCorruptError("completed job outcome set is incomplete")
  for route in routes:
    first = by_key[(1, route.route_name)]
    second = by_key[(2, route.route_name)]
    if first["disposition"] != second["disposition"]:
      raise BridgeCorruptError("preparation authorities disagree on disposition")
    if first["disposition"] == "rejected":
      if (
        first["reason"] != second["reason"]
        or first["message"] != second["message"]
      ):
        raise BridgeCorruptError("preparation authorities disagree on rejection")
    else:
      left = first["descriptor"]
      right = second["descriptor"]
      for key in ("sha256", "size_bytes", "frame_count", "provenance"):
        if left[key] != right[key]:
          raise BridgeCorruptError("preparation authority artifacts differ")
      left_vector = first["certification_vector"]
      right_vector = second["certification_vector"]
      for key in (
        "sha256",
        "size_bytes",
        "selected_controls_witnesses",
        "selected_segment_count",
        "selection_identity_sha256",
      ):
        if left_vector[key] != right_vector[key]:
          raise BridgeCorruptError(
            "certification-vector authority artifacts differ",
          )
  return by_key


def _exclude_unverified_remote_rejections(
  *,
  plan: RemoteRoutePlan,
  outcomes: Mapping[tuple[int, str], dict[str, object]],
) -> tuple[
  RemoteRoutePlan,
  dict[tuple[int, str], dict[str, object]],
]:
  """Remove only A/A-agreed, PC-only rejections from learner discovery.

  ``_validated_outcomes`` must run first over the original frozen manifest.
  The resulting exclusion has no ledger disposition, watermark effect, route
  evidence, learner count, or behavior-cohort vote.  A locally retained route
  is never eligible: its rejection continues through ARM certification.
  """
  expected = {
    (authority, route.route_name)
    for authority in (1, 2)
    for route in plan.replay_candidates
  }
  if set(outcomes) != expected:
    raise BridgeCorruptError(
      "remote rejection partition requires the complete validated outcome set",
    )

  excluded_names: set[str] = set()
  exclusions: list[RemoteRouteExclusion] = []
  for route in plan.replay_candidates:
    first = outcomes[(1, route.route_name)]
    second = outcomes[(2, route.route_name)]
    if (
      route.route_name not in plan.locally_available_route_names
      and first["disposition"] == "rejected"
    ):
      # Defense in depth: callers must validate the original A/A result set
      # before partitioning. Never let a direct helper call weaken that rule.
      if (
        second["disposition"] != "rejected"
        or first["reason"] != second["reason"]
        or first["message"] != second["message"]
      ):
        raise BridgeCorruptError(
          "preparation authorities disagree on remote-only rejection",
        )
      excluded_names.add(route.route_name)
      exclusions.append(RemoteRouteExclusion(
        route_identity_sha256=route.display_identity,
        rejection_reason=str(first["reason"]),
        rejection_message=str(first["message"]),
      ))

  if not excluded_names:
    return plan, dict(outcomes)
  if (
    excluded_names & plan.locally_available_route_names
    or excluded_names & {route.route_name for route in plan.late_candidates}
    or excluded_names & {route.route_name for route in plan.upload_candidates}
  ):
    raise BridgeCorruptError("remote-only exclusion escaped its replay scope")

  effective = RemoteRoutePlan(
    discovery=FullRlogDiscovery(
      candidates=tuple(
        route
        for route in plan.discovery.candidates
        if route.route_name not in excluded_names
      ),
      pending_logger_close=plan.discovery.pending_logger_close,
    ),
    replay_candidates=tuple(
      route
      for route in plan.replay_candidates
      if route.route_name not in excluded_names
    ),
    late_candidates=plan.late_candidates,
    upload_candidates=plan.upload_candidates,
    locally_available_route_names=plan.locally_available_route_names,
    unverified_exclusions=(
      *plan.unverified_exclusions,
      *exclusions,
    ),
  )
  effective_names = {
    route.route_name for route in effective.replay_candidates
  }
  filtered = {
    key: value
    for key, value in outcomes.items()
    if key[1] in effective_names
  }
  if set(filtered) != {
    (authority, route.route_name)
    for authority in (1, 2)
    for route in effective.replay_candidates
  }:
    raise BridgeCorruptError("effective remote outcome set is incomplete")
  return effective, filtered


def _download_outcomes(
  *,
  client: OffdeviceBridgeClient,
  job_id: str,
  routes: tuple[RouteCandidate, ...],
  outcomes: Mapping[tuple[int, str], dict[str, object]],
  scratch_parent: Path,
  abort_requested: Callable[[], bool],
  progress: Callable[[int, int, int, int], None] | None = None,
) -> tuple[_PrivateScratchRoot, dict[tuple[int, str], _PreparedOutcome]]:
  resolved: dict[tuple[int, str], _PreparedOutcome] = {}
  accepted = tuple(
    (authority, route)
    for authority in (1, 2)
    for route in routes
    if outcomes[(authority, route.route_name)]["disposition"] != "rejected"
  )
  total_artifacts = 2 * len(accepted)
  total_bytes = sum(
    int(outcomes[(authority, route.route_name)]["descriptor"]["size_bytes"])
    + int(
      outcomes[(authority, route.route_name)]["certification_vector"][
        "size_bytes"
      ]
    )
    for authority, route in accepted
  )
  owner = _create_private_scratch(
    scratch_parent,
    prefix=REMOTE_SCRATCH_PREFIX,
    required_bytes=total_bytes,
  )
  root = owner.path
  completed_artifacts = 0
  completed_bytes = 0
  if progress is not None and total_artifacts:
    progress(0, total_artifacts, 0, total_bytes)
  try:
    for authority in (1, 2):
      for route in routes:
        check_abort(abort_requested)
        raw = outcomes[(authority, route.route_name)]
        if raw["disposition"] == "rejected":
          resolved[(authority, route.route_name)] = _PreparedOutcome(
            descriptor=None,
            certification_vector=None,
            rejection_reason=str(raw["reason"]),
            rejection_message=str(raw["message"]),
            certification_identity_sha256=None,
          )
          continue
        remote = raw["descriptor"]
        filename = f"a{authority}-{route.route_name}.spool"
        partial = root / f".{filename}.partial"
        final = root / filename
        artifact_base_bytes = completed_bytes
        artifact_count_before = completed_artifacts
        expected_artifact_bytes = int(remote["size_bytes"])

        def artifact_progress(
          artifact_bytes: int,
          artifact_total_bytes: int,
          *,
          artifact_base_bytes: int = artifact_base_bytes,
          artifact_count_before: int = artifact_count_before,
          expected_artifact_bytes: int = expected_artifact_bytes,
        ) -> None:
          if artifact_total_bytes != expected_artifact_bytes:
            raise BridgeCorruptError("artifact progress total changed")
          if progress is not None:
            progress(
              artifact_count_before,
              total_artifacts,
              artifact_base_bytes + artifact_bytes,
              total_bytes,
            )

        with partial.open("xb") as sink:
          os.fchmod(sink.fileno(), 0o600)
          _register_private_scratch_file(owner, partial, stable=False)
          client.download_artifact(
            job_id=job_id,
            artifact_id=str(remote["artifact_id"]),
            expected_size_bytes=int(remote["size_bytes"]),
            expected_sha256=str(remote["sha256"]),
            sink=sink,
            progress=artifact_progress,
          )
          sink.flush()
          os.fsync(sink.fileno())
        _rename_private_scratch_file(
          owner,
          partial,
          final,
          stable=True,
        )
        descriptor = PreparedRouteSpoolDescriptor(
          route_name=route.route_name,
          filename=filename,
          sha256=str(remote["sha256"]),
          size_bytes=int(remote["size_bytes"]),
          frame_count=int(remote["frame_count"]),
        )
        completed_artifacts += 1
        completed_bytes += int(remote["size_bytes"])
        if progress is not None:
          progress(
            completed_artifacts,
            total_artifacts,
            completed_bytes,
            total_bytes,
          )

        remote_vector = raw["certification_vector"]
        vector_filename = f"a{authority}-{route.route_name}.cert-vector"
        vector_partial = root / f".{vector_filename}.partial"
        vector_final = root / vector_filename
        vector_base_bytes = completed_bytes
        vector_count_before = completed_artifacts
        expected_vector_bytes = int(remote_vector["size_bytes"])

        def vector_progress(
          artifact_bytes: int,
          artifact_total_bytes: int,
          *,
          artifact_base_bytes: int = vector_base_bytes,
          artifact_count_before: int = vector_count_before,
          expected_artifact_bytes: int = expected_vector_bytes,
        ) -> None:
          if artifact_total_bytes != expected_artifact_bytes:
            raise BridgeCorruptError("vector progress total changed")
          if progress is not None:
            progress(
              artifact_count_before,
              total_artifacts,
              artifact_base_bytes + artifact_bytes,
              total_bytes,
            )

        with vector_partial.open("xb") as sink:
          os.fchmod(sink.fileno(), 0o600)
          _register_private_scratch_file(owner, vector_partial, stable=False)
          client.download_artifact(
            job_id=job_id,
            artifact_id=str(remote_vector["artifact_id"]),
            expected_size_bytes=expected_vector_bytes,
            expected_sha256=str(remote_vector["sha256"]),
            sink=sink,
            progress=vector_progress,
          )
          sink.flush()
          os.fsync(sink.fileno())
        _rename_private_scratch_file(
          owner,
          vector_partial,
          vector_final,
          stable=True,
        )
        try:
          vector = CertificationVector.from_bytes(vector_final.read_bytes())
        except (CertificationVectorError, OSError) as exc:
          raise BridgeCorruptError(
            f"downloaded certification vector is invalid: {exc}",
          ) from exc
        if (
          vector.sha256 != remote_vector["sha256"]
          or vector.manifest["route_name"] != route.route_name
          or vector.manifest["selection_identity_sha256"]
          != remote_vector["selection_identity_sha256"]
          or vector.manifest["bounds"]["selected_controls_witnesses"]
          != remote_vector["selected_controls_witnesses"]
          or len(vector.manifest["segment_results"])
          != remote_vector["selected_segment_count"]
        ):
          raise BridgeCorruptError(
            "downloaded certification vector descriptor changed",
          )
        vector_descriptor = _CertificationVectorDescriptor(
          filename=vector_filename,
          sha256=str(remote_vector["sha256"]),
          size_bytes=expected_vector_bytes,
          selected_controls_witnesses=int(
            remote_vector["selected_controls_witnesses"],
          ),
          selected_segment_count=int(remote_vector["selected_segment_count"]),
          selection_identity_sha256=str(
            remote_vector["selection_identity_sha256"],
          ),
        )
        resolved[(authority, route.route_name)] = _PreparedOutcome(
          descriptor=descriptor,
          certification_vector=vector_descriptor,
          rejection_reason=None,
          rejection_message=None,
          certification_identity_sha256=None,
        )
        completed_artifacts += 1
        completed_bytes += expected_vector_bytes
        if progress is not None:
          progress(
            completed_artifacts,
            total_artifacts,
            completed_bytes,
            total_bytes,
          )
    directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
      os.fsync(directory_fd)
    finally:
      os.close(directory_fd)
    return owner, resolved
  except BaseException:
    _cleanup_private_scratch(owner)
    raise


def remote_error_is_unavailable(error: BridgeRemoteError) -> bool:
  """Classify failures that can be completed by the unchanged local path.

  These codes describe transient worker state or resources. Contract/source
  mismatches, malformed transfers, bounds failures, and conflicts remain
  fail-closed because retrying locally must not hide an integrity failure.
  """
  return error.code in REMOTE_UNAVAILABLE_ERROR_CODES


def remote_error_fallback_reason(
  error: BridgeRemoteError,
) -> OffdeviceFallbackReason:
  """Map authenticated worker failures to stable display-only diagnostics."""
  if error.code == "busy":
    return OffdeviceFallbackReason.WORKER_BUSY
  if error.code in {"artifact_not_found", "route_unavailable"}:
    return OffdeviceFallbackReason.REMOTE_ARTIFACT_UNAVAILABLE
  if error.code in {"internal_error", "job_failed"}:
    return OffdeviceFallbackReason.REMOTE_JOB_FAILED
  if error.code == "job_not_found":
    return OffdeviceFallbackReason.NETWORK_INTERRUPTED
  return OffdeviceFallbackReason.REMOTE_PREPARATION_UNAVAILABLE


def cleanup_stale_remote_scratch(parent: str | Path) -> int:
  """Quarantine marked stale scratch without deleting unauthenticated files."""
  root = Path(parent)
  try:
    root.mkdir(parents=True, exist_ok=True)
    entries = tuple(root.iterdir())
  except OSError as exc:
    raise BridgeUnavailableError("remote scratch root is unavailable") from exc
  removed = 0
  owned_prefixes = (
    REMOTE_SCRATCH_PREFIX,
    CERTIFICATION_SOURCE_SNAPSHOT_PREFIX,
  )
  for entry in entries:
    prefix = next(
      (value for value in owned_prefixes if entry.name.startswith(value)),
      None,
    )
    if prefix is None:
      continue
    owner = _private_scratch_from_path(entry, prefix=prefix)
    directory_fd = os.open(
      entry,
      os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
      | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
      opened = os.fstat(directory_fd)
      names = os.listdir(directory_fd)
      if (
        opened.st_dev != owner.st_dev
        or opened.st_ino != owner.st_ino
        or len(names) > PRIVATE_SCRATCH_MAX_CHILDREN
        or PRIVATE_SCRATCH_MARKER not in names
      ):
        raise BridgeUnavailableError("stale private scratch is unsafe")
      for name in names:
        child = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
          not stat.S_ISREG(child.st_mode)
          or child.st_uid != os.geteuid()
          or stat.S_IMODE(child.st_mode) & 0o077
        ):
          raise BridgeUnavailableError("stale private scratch child is unsafe")
    finally:
      os.close(directory_fd)
    rebound = entry.lstat()
    if rebound.st_dev != owner.st_dev or rebound.st_ino != owner.st_ino:
      raise BridgeUnavailableError("stale private scratch path changed")
    quarantine = root / (
      f"{PRIVATE_SCRATCH_QUARANTINE_PREFIX}{secrets.token_hex(16)}"
    )
    os.rename(entry, quarantine)
    quarantined = quarantine.lstat()
    if quarantined.st_dev != owner.st_dev or quarantined.st_ino != owner.st_ino:
      raise BridgeUnavailableError("stale scratch quarantine identity changed")
    parent_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
      os.fsync(parent_fd)
    finally:
      os.close(parent_fd)
    removed += 1
  return removed


def prepare_remote_session(
  *,
  engine: HistoricalLearningBackfill,
  client: OffdeviceBridgeClient,
  contract: dict[str, object],
  device_numerical_environment_sha256: str,
  preparation_implementation_sha256: str,
  scratch_parent: str | Path,
  abort_requested: Callable[[], bool],
  sleep: Callable[[float], None] = time.sleep,
  offdevice_progress: OffdeviceProgressPublisher | None = None,
) -> RemotePreparationSession:
  """Prepare a complete remote spool transaction or raise a stable error."""
  check_abort(abort_requested)
  cleanup_stale_remote_scratch(scratch_parent)
  health = client.health()
  for key in (
    "source_commit",
    "opendbc_commit",
    "panda_commit",
    "historical_descriptor_registry_sha256",
  ):
    if health[key] != contract[key]:
      raise BridgeIncompatibleError(f"worker health disagrees on {key}")
  if (
    health["preparation_implementation_sha256"]
    != preparation_implementation_sha256
  ):
    raise BridgeIncompatibleError(
      "worker route-preparation implementation differs from the device",
    )

  local_discovery = discover_full_rlog_state(
    engine.log_root,
    abort_requested=abort_requested,
  )
  inventory = client.route_inventory()
  plan = build_remote_route_plan(
    local_discovery=local_discovery,
    inventory_payload=inventory,
    expected_dongle_id=engine.expected_dongle_id,
    placeholder_root=scratch_parent,
    engine=engine,
  )
  _upload_missing_routes(
    client=client,
    routes=plan.upload_candidates,
    dongle_id=engine.expected_dongle_id,
  )
  if not plan.replay_candidates:
    prior_extractor = (
      _prior_generation_extractor_sha256(engine)
      if plan.late_candidates
      else None
    )
    return RemotePreparationSession(
      base_engine=engine,
      plan=plan,
      scratch_directory=None,
      outcomes={},
      worker_extractor_sha256=prior_extractor,
    )

  projector = _RemoteProgressProjector(
    engine=engine,
    routes=plan.replay_candidates,
    offdevice_progress=offdevice_progress,
  )
  projector.start()
  client_job_id = secrets.token_hex(16)
  created = client.create_job(
    client_job_id=client_job_id,
    routes=[route.route_name for route in plan.replay_candidates],
    contract=contract,
  )
  worker_identity_keys = (
    "preparation_implementation_sha256",
    "worker_instance_id",
    "worker_implementation_commit",
    "worker_implementation_sha256",
    "worker_numerical_environment_sha256",
    "worker_preparation_implementation_sha256",
  )
  if any(created[key] != health[key] for key in worker_identity_keys):
    raise BridgeUnavailableError(
      "worker instance or implementation changed before job creation",
    )
  job_id = str(created["job_id"])
  final_status: dict[str, object] | None = None
  try:
    while True:
      check_abort(abort_requested)
      try:
        status = client.job_status(job_id)
      except BridgeRemoteError as remote_error:
        if remote_error_is_unavailable(remote_error):
          raise BridgeFallbackUnavailableError(
            str(remote_error),
            remote_error_fallback_reason(remote_error),
          ) from remote_error
        raise
      if any(status[key] != health[key] for key in worker_identity_keys):
        raise BridgeUnavailableError(
          "worker instance or implementation changed during preparation",
        )
      projector.update(status["progress"])
      state = status["state"]
      if state == "completed":
        final_status = status
        break
      if state == "failed":
        error = status["error"]
        assert type(error) is dict
        remote_error = BridgeRemoteError(str(error["code"]), str(error["message"]))
        if remote_error_is_unavailable(remote_error):
          raise BridgeFallbackUnavailableError(
            str(remote_error),
            remote_error_fallback_reason(remote_error),
          )
        raise remote_error
      if state == "canceled":
        raise BridgeFallbackUnavailableError(
          "remote preparation job was canceled",
          OffdeviceFallbackReason.REMOTE_JOB_CANCELED,
        )
      sleep(REMOTE_JOB_POLL_SECONDS)
  except BaseException:
    try:
      client.cancel_job(job_id)
    except Exception:
      pass
    raise
  assert final_status is not None
  if final_status["worker_extractor_sha256"] != health["worker_extractor_sha256"]:
    raise BridgeCorruptError("worker extractor identity changed during the job")
  outcomes = _validated_outcomes(
    status=final_status,
    routes=plan.replay_candidates,
  )
  plan, outcomes = _exclude_unverified_remote_rejections(
    plan=plan,
    outcomes=outcomes,
  )
  excluded_count = len(plan.unverified_exclusions)
  total_certification_routes = len(plan.replay_candidates) + excluded_count
  if not plan.replay_candidates:
    if offdevice_progress is not None:
      offdevice_progress.publish(
        phase=OffdeviceProgressPhase.ARM_CERTIFYING,
        architecture_domain_count=0,
        architecture_domain_index=0,
        certified_domain_count=0,
        certified_route_count=0,
        remote_only_rejection_excluded_count=excluded_count,
        total_certification_domain_count=0,
        total_certification_route_count=total_certification_routes,
      )
    prior_extractor = (
      _prior_generation_extractor_sha256(engine)
      if plan.late_candidates
      else None
    )
    projector.complete()
    if offdevice_progress is not None:
      offdevice_progress.publish(
        phase=OffdeviceProgressPhase.REMOTE_READY,
        certified_domain_count=0,
        certified_route_count=0,
        remote_only_rejection_excluded_count=excluded_count,
        total_certification_domain_count=0,
        total_certification_route_count=total_certification_routes,
      )
    return RemotePreparationSession(
      base_engine=engine,
      plan=plan,
      scratch_directory=None,
      outcomes={},
      worker_extractor_sha256=prior_extractor,
    )

  def download_progress(
    completed_artifacts: int,
    total_artifacts: int,
    completed_bytes: int,
    total_bytes: int,
  ) -> None:
    if offdevice_progress is None:
      return
    offdevice_progress.publish(
      phase=OffdeviceProgressPhase.DOWNLOADING,
      completed_artifact_count=completed_artifacts,
      completed_bytes=completed_bytes,
      total_artifact_count=total_artifacts,
      total_bytes=total_bytes,
    )

  scratch_owner, downloaded = _download_outcomes(
    client=client,
    job_id=job_id,
    routes=plan.replay_candidates,
    outcomes=outcomes,
    scratch_parent=Path(scratch_parent),
    abort_requested=abort_requested,
    progress=download_progress,
  )
  root = scratch_owner.path
  try:
    last_certification_progress = [0, 0, 0, len(plan.replay_candidates)]

    def certification_progress(
      certified_domains: int,
      certified_routes: int,
      total_domains: int,
      total_routes: int,
      architecture_domain_index: int,
      architecture_route_identity_sha256: str | None,
      architecture_segment_index: int | None,
      architecture_segment_count: int | None,
    ) -> None:
      last_certification_progress[:] = [
        certified_domains,
        certified_routes,
        total_domains,
        total_routes,
      ]
      if offdevice_progress is None:
        return
      offdevice_progress.publish(
        phase=OffdeviceProgressPhase.ARM_CERTIFYING,
        architecture_domain_count=total_domains,
        architecture_domain_index=architecture_domain_index,
        architecture_route_identity_sha256=(
          architecture_route_identity_sha256
        ),
        architecture_segment_count=architecture_segment_count,
        architecture_segment_index=architecture_segment_index,
        certified_domain_count=certified_domains,
        certified_route_count=certified_routes,
        remote_only_rejection_excluded_count=excluded_count,
        total_certification_domain_count=total_domains,
        total_certification_route_count=total_routes + excluded_count,
      )

    certified = _certify_preparation_domains(
      engine=engine,
      plan=plan,
      scratch_directory=root,
      outcomes=downloaded,
      contract=contract,
      device_numerical_environment_sha256=(
        device_numerical_environment_sha256
      ),
      preparation_implementation_sha256=(
        preparation_implementation_sha256
      ),
      worker_extractor_sha256=str(
        final_status["worker_extractor_sha256"],
      ),
      worker_implementation_commit=str(
        final_status["worker_implementation_commit"],
      ),
      worker_implementation_sha256=str(
        final_status["worker_implementation_sha256"],
      ),
      worker_numerical_environment_sha256=str(
        final_status["worker_numerical_environment_sha256"],
      ),
      worker_preparation_implementation_sha256=str(
        final_status["worker_preparation_implementation_sha256"],
      ),
      worker_instance_id=str(final_status["worker_instance_id"]),
      secret=client.secret,
      abort_requested=abort_requested,
      progress=certification_progress,
    )
    projector.complete()
    if offdevice_progress is not None:
      certified_domains, certified_routes, total_domains, total_routes = (
        last_certification_progress
      )
      offdevice_progress.publish(
        phase=OffdeviceProgressPhase.REMOTE_READY,
        certified_domain_count=certified_domains,
        certified_route_count=certified_routes,
        remote_only_rejection_excluded_count=excluded_count,
        total_certification_domain_count=total_domains,
        total_certification_route_count=total_routes + excluded_count,
      )
    return RemotePreparationSession(
      base_engine=engine,
      plan=plan,
      scratch_directory=root,
      outcomes=certified,
      worker_extractor_sha256=str(
        final_status["worker_extractor_sha256"],
      ),
      scratch_owner=scratch_owner,
    )
  except BaseException:
    _cleanup_private_scratch(scratch_owner)
    raise

"""Scratch facade over the complete shared route-evidence v2 artifact.

The former BLATSP01 physical-only spool is intentionally unsupported.  A
worker now transports the exact RouteEvidenceArtifact bytes; calibration sees
the lazy physical iterator while behavior replay sees the compact context
planes from the same hash-bound object.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat
import struct
import tempfile

from openpilot.selfdrive.controls.lib.blatv2.preparation_frame import MeasuredLearningFrame
from openpilot.selfdrive.controls.lib.blatv2.route_evidence import (
  MAX_ARTIFACT_BYTES,
  ROUTE_EVIDENCE_MAGIC,
  ROUTE_EVIDENCE_VERSION,
  RouteEvidenceArtifact,
  RouteEvidenceError,
  RouteEvidenceFileSummary,
  inspect_route_evidence_file,
)


SPOOL_MAGIC = ROUTE_EVIDENCE_MAGIC
SPOOL_VERSION = ROUTE_EVIDENCE_VERSION
_FRAME = struct.Struct("<4q9d12?")
SPOOL_RECORD_SIZE = _FRAME.size
SPOOL_HEADER_SIZE = 0
MAX_METADATA_BYTES = 0
_SAFE_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class SpoolFormatError(RuntimeError):
  """A scratch route-evidence artifact is unsafe, corrupt, or mismatched."""


@dataclass(frozen=True, slots=True)
class PreparedRouteSpoolDescriptor:
  route_name: str
  filename: str
  sha256: str
  size_bytes: int
  frame_count: int

  @property
  def route(self) -> str:
    return self.route_name

  def cleanup(self, directory: str | Path) -> None:
    root = _private_directory(directory, create=False)
    _validate_descriptor(self)
    path = root / self.filename
    info = _regular_file_lstat(path)
    if info.st_size != self.size_bytes or _hash_regular_file(path, info) != self.sha256:
      raise SpoolFormatError("spool cleanup identity mismatch")
    path.unlink()
    _fsync_directory(root)


class PreparedRouteSpool:
  __slots__ = (
    "_artifact", "_descriptor", "_directory", "controls_witness_count",
    "gap_count", "provenance", "route_name", "unresolved_witness_count",
  )

  def __init__(self, directory: Path, descriptor: PreparedRouteSpoolDescriptor, artifact: RouteEvidenceFileSummary) -> None:
    self._directory = directory
    self._descriptor = descriptor
    self._artifact = artifact
    source = artifact.source_identity
    self.route_name = source.route_id
    self.controls_witness_count = source.controls_witness_count
    self.unresolved_witness_count = source.unresolved_witness_count
    self.gap_count = source.gap_count
    self.provenance = dict(source.preparation_provenance)

  @property
  def frame_count(self) -> int:
    return self._descriptor.frame_count

  @property
  def descriptor(self) -> PreparedRouteSpoolDescriptor:
    return self._descriptor

  @property
  def route_evidence(self) -> RouteEvidenceFileSummary:
    return self._artifact

  @property
  def canonical_path(self) -> Path:
    """Private bridge seam for constant-memory A/A staging."""
    return self._directory / self._descriptor.filename

  def iter_frames(self) -> Iterator[MeasuredLearningFrame]:
    path = self.canonical_path
    summary = self._artifact
    descriptor = os.open(
      path,
      os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
      opened = os.fstat(descriptor)
      expected_stat = (
        summary.st_dev, summary.st_ino, summary.st_size,
        summary.st_mtime_ns, summary.st_ctime_ns,
      )
      if (
        opened.st_dev, opened.st_ino, opened.st_size,
        opened.st_mtime_ns, opened.st_ctime_ns,
      ) != expected_stat:
        raise SpoolFormatError("spool changed after validation")
      os.lseek(descriptor, summary.physical_offset, os.SEEK_SET)
      remaining = summary.physical_size
      records_per_chunk = 1024
      while remaining:
        requested = min(
          remaining,
          records_per_chunk * SPOOL_RECORD_SIZE,
        )
        encoded = os.read(descriptor, requested)
        if len(encoded) != requested or len(encoded) % SPOOL_RECORD_SIZE:
          raise SpoolFormatError("spool physical plane is truncated")
        remaining -= len(encoded)
        view = memoryview(encoded)
        for offset in range(0, len(view), SPOOL_RECORD_SIZE):
          yield _decode_frame(view[offset:offset + SPOOL_RECORD_SIZE])
      after = os.fstat(descriptor)
      if (
        after.st_dev, after.st_ino, after.st_size,
        after.st_mtime_ns, after.st_ctime_ns,
      ) != expected_stat:
        raise SpoolFormatError("spool changed during physical replay")
    finally:
      os.close(descriptor)

  def __iter__(self) -> Iterator[MeasuredLearningFrame]:
    return self.iter_frames()

  def cleanup(self) -> None:
    path = self.canonical_path
    info = _regular_file_lstat(path)
    summary = self._artifact
    if (
      info.st_dev != summary.st_dev
      or info.st_ino != summary.st_ino
      or info.st_size != summary.st_size
      or info.st_mtime_ns != summary.st_mtime_ns
      or info.st_ctime_ns != summary.st_ctime_ns
    ):
      raise SpoolFormatError("spool cleanup identity mismatch")
    path.unlink()
    _fsync_directory(self._directory)


def write_prepared_route_spool(
  directory: str | Path,
  route_name: str,
  frames: tuple[MeasuredLearningFrame, ...],
  *,
  controls_witness_count: int,
  unresolved_witness_count: int,
  gap_count: int,
  provenance: Mapping[str, object],
  max_frames: int,
  abort_requested: Callable[[], bool],
  filename: str | None = None,
  route_evidence: RouteEvidenceArtifact | None = None,
) -> PreparedRouteSpoolDescriptor:
  """Publish exact v2 artifact bytes; physical-only calls fail closed."""
  root = _private_directory(directory, create=True)
  _validate_route_name(route_name)
  if route_evidence is None or type(route_evidence) is not RouteEvidenceArtifact:
    raise SpoolFormatError("complete shared route evidence v2 is required")
  if type(frames) is not tuple or any(type(frame) is not MeasuredLearningFrame for frame in frames):
    raise SpoolFormatError("spool frames are invalid")
  _validate_frame_bound(len(frames), max_frames)
  source = route_evidence.source_identity
  if (
    source.route_id != route_name
    or source.physical_record_count != len(frames)
    or source.controls_witness_count != controls_witness_count
    or source.unresolved_witness_count != unresolved_witness_count
    or source.gap_count != gap_count
    or dict(source.preparation_provenance) != dict(provenance)
    or bytes(route_evidence.physical_bytes) != b"".join(_encode_frame(frame) for frame in frames)
  ):
    raise SpoolFormatError("route evidence and preparation facts disagree")
  selected = _default_filename(route_name) if filename is None else filename
  _validate_filename(selected)
  data = route_evidence.canonical_bytes
  if len(data) > MAX_ARTIFACT_BYTES:
    raise SpoolFormatError("route evidence exceeds bridge bound")
  _abort_if_requested(abort_requested)
  final_path = root / selected
  partial_fd = -1
  partial_path: Path | None = None
  try:
    partial_fd, name = tempfile.mkstemp(dir=root, prefix=f".{selected}.", suffix=".partial")
    partial_path = Path(name)
    os.fchmod(partial_fd, 0o600)
    view = memoryview(data)
    while view:
      _abort_if_requested(abort_requested)
      count = os.write(partial_fd, view)
      if count <= 0:
        raise OSError("short spool write")
      view = view[count:]
    os.fsync(partial_fd)
    os.close(partial_fd)
    partial_fd = -1
    os.replace(partial_path, final_path)
    partial_path = None
    _fsync_directory(root)
  except BaseException:
    if partial_fd >= 0:
      os.close(partial_fd)
    if partial_path is not None:
      try:
        partial_path.unlink()
      except FileNotFoundError:
        pass
    raise
  return PreparedRouteSpoolDescriptor(
    route_name=route_name, filename=selected,
    sha256=route_evidence.sha256, size_bytes=len(data), frame_count=len(frames),
  )


def open_prepared_route_spool(
  directory: str | Path,
  descriptor: PreparedRouteSpoolDescriptor,
  *,
  expected_route_name: str,
  max_frames: int,
) -> PreparedRouteSpool:
  root = _private_directory(directory, create=False)
  _validate_descriptor(descriptor)
  _validate_route_name(expected_route_name)
  _validate_frame_bound(descriptor.frame_count, max_frames)
  if descriptor.route_name != expected_route_name or descriptor.size_bytes > MAX_ARTIFACT_BYTES:
    raise SpoolFormatError("spool descriptor route/size mismatch")
  path = root / descriptor.filename
  info = _regular_file_lstat(path)
  if info.st_size != descriptor.size_bytes:
    raise SpoolFormatError("spool descriptor identity mismatch")
  try:
    artifact = inspect_route_evidence_file(path)
  except RouteEvidenceError as error:
    raise SpoolFormatError("old/incomplete/corrupt route spool is unsupported") from error
  if (
    artifact.sha256 != descriptor.sha256
    or artifact.source_identity.route_id != expected_route_name
    or artifact.source_identity.physical_record_count != descriptor.frame_count
  ):
    raise SpoolFormatError("route evidence identity/count mismatch")
  return PreparedRouteSpool(root, descriptor, artifact)


def _encode_frame(frame: MeasuredLearningFrame) -> bytes:
  return _FRAME.pack(
    frame.sample_mono_ns, frame.response_mono_ns,
    frame.applied_report_mono_ns, frame.applied_effective_mono_ns,
    frame.speed_mps, frame.steering_angle_deg, frame.steering_rate_deg_s,
    frame.steering_torque, frame.applied_torque, frame.angle_offset_deg,
    frame.steer_ratio, frame.stiffness_factor, frame.roll_rad,
    frame.steering_pressed, frame.standstill, frame.steer_fault_temporary,
    frame.steer_fault_permanent, frame.can_valid, frame.can_timeout,
    frame.lateral_active, frame.live_parameters_valid,
    frame.angle_offset_valid, frame.steer_ratio_valid,
    frame.stiffness_factor_valid, frame.inputs_valid,
  )


def _decode_frame(encoded: bytes) -> MeasuredLearningFrame:
  values = _FRAME.unpack(encoded)
  return MeasuredLearningFrame(
    sample_mono_ns=values[0], response_mono_ns=values[1],
    applied_report_mono_ns=values[2], applied_effective_mono_ns=values[3],
    speed_mps=values[4], steering_angle_deg=values[5],
    steering_rate_deg_s=values[6], steering_torque=values[7],
    applied_torque=values[8], angle_offset_deg=values[9],
    steer_ratio=values[10], stiffness_factor=values[11], roll_rad=values[12],
    steering_pressed=values[13], standstill=values[14],
    steer_fault_temporary=values[15], steer_fault_permanent=values[16],
    can_valid=values[17], can_timeout=values[18], lateral_active=values[19],
    live_parameters_valid=values[20], angle_offset_valid=values[21],
    steer_ratio_valid=values[22], stiffness_factor_valid=values[23],
    inputs_valid=values[24],
  )


def _default_filename(route_name: str) -> str:
  identity = hashlib.sha256(route_name.encode()).hexdigest()[:24]
  return f"prepared-route-{identity}.route-evidence"


def _validate_filename(value: object) -> None:
  if type(value) is not str or _SAFE_FILENAME.fullmatch(value) is None or value in {".", ".."}:
    raise SpoolFormatError("spool filename is unsafe")


def _validate_route_name(value: object) -> None:
  if type(value) is not str or not value or len(value.encode()) > 1024 or "\0" in value:
    raise SpoolFormatError("spool route name is invalid")


def _validate_descriptor(value: object) -> None:
  if type(value) is not PreparedRouteSpoolDescriptor:
    raise SpoolFormatError("spool descriptor type is invalid")
  _validate_route_name(value.route_name)
  _validate_filename(value.filename)
  if type(value.sha256) is not str or _SHA256.fullmatch(value.sha256) is None:
    raise SpoolFormatError("spool descriptor hash is invalid")
  if type(value.size_bytes) is not int or value.size_bytes < 0 or type(value.frame_count) is not int or value.frame_count < 0:
    raise SpoolFormatError("spool descriptor counts are invalid")


def _validate_frame_bound(count: object, maximum: object) -> None:
  if type(maximum) is not int or maximum < 0 or type(count) is not int or count < 0 or count > maximum:
    raise SpoolFormatError("spool frame count exceeds caller bound")


def _private_directory(directory: str | Path, *, create: bool) -> Path:
  root = Path(directory)
  if create:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
  try:
    info = root.lstat()
  except FileNotFoundError as error:
    raise SpoolFormatError("spool directory does not exist") from error
  if root.is_symlink() or not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
    raise SpoolFormatError("spool directory is not private")
  return root


def _regular_file_lstat(path: Path) -> os.stat_result:
  try:
    info = path.lstat()
  except FileNotFoundError as error:
    raise SpoolFormatError("spool file is missing") from error
  if path.is_symlink() or not stat.S_ISREG(info.st_mode):
    raise SpoolFormatError("spool file is not a regular non-symlink file")
  return info


def _hash_regular_file(path: Path, expected: os.stat_result) -> str:
  descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
  try:
    actual = os.fstat(descriptor)
    if actual.st_dev != expected.st_dev or actual.st_ino != expected.st_ino or actual.st_size != expected.st_size:
      raise SpoolFormatError("spool changed while opening")
    digest = hashlib.sha256()
    while block := os.read(descriptor, 1024 * 1024):
      digest.update(block)
    return digest.hexdigest()
  finally:
    os.close(descriptor)


def _abort_if_requested(callback: Callable[[], bool]) -> None:
  if callback():
    raise SpoolFormatError("spool write cancelled")


def _fsync_directory(path: Path) -> None:
  descriptor = os.open(path, os.O_RDONLY)
  try:
    os.fsync(descriptor)
  finally:
    os.close(descriptor)

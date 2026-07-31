"""Private, deterministic scratch spools for prepared learning routes.

The spool is deliberately narrower than :mod:`learning_backfill`: it knows
only the measured frame wire format and the route facts needed to bind those
frames to a preparation result.  It is therefore safe for worker processes to
exchange a small pickleable descriptor instead of copying a complete tuple of
frames through a multiprocessing pipe.

Files are scratch artifacts, not durable learning state.  The caller chooses a
private directory, the writer publishes one complete file by atomic rename,
and cleanup verifies the descriptor identity before deleting that exact file.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import tempfile
from typing import Any

from openpilot.selfdrive.controls.lib.blatv2.learning_runtime import (
  MeasuredLearningFrame,
)


SPOOL_MAGIC = b"BLATSP01"
SPOOL_VERSION = 1
_HEADER = struct.Struct("<8sHHQI")
_FRAME = struct.Struct("<4q9d12?")
SPOOL_HEADER_SIZE = _HEADER.size
SPOOL_RECORD_SIZE = _FRAME.size
MAX_METADATA_BYTES = 64 * 1024
_SAFE_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_METADATA_KEYS = frozenset({
  "controls_witness_count",
  "gap_count",
  "provenance",
  "route_name",
  "unresolved_witness_count",
})


class SpoolFormatError(RuntimeError):
  """A scratch spool is unsafe, corrupt, non-canonical, or out of bounds."""


@dataclass(frozen=True, slots=True)
class PreparedRouteSpoolDescriptor:
  """Small process-safe identity for one completely published spool."""

  route_name: str
  filename: str
  sha256: str
  size_bytes: int
  frame_count: int

  @property
  def route(self) -> str:
    return self.route_name

  def cleanup(self, directory: str | Path) -> None:
    """Delete only the regular file whose bytes match this descriptor."""
    root = _private_directory(directory, create=False)
    _validate_descriptor(self)
    path = root / self.filename
    file_stat = _regular_file_lstat(path)
    if file_stat.st_size != self.size_bytes:
      raise SpoolFormatError("spool cleanup identity size mismatch")
    if _hash_regular_file(path, file_stat) != self.sha256:
      raise SpoolFormatError("spool cleanup identity hash mismatch")
    path.unlink()
    _fsync_directory(root)


class PreparedRouteSpool:
  """Validated route facts plus a lazy, independently checked frame stream."""

  __slots__ = (
    "_descriptor",
    "_directory",
    "_metadata_bytes",
    "controls_witness_count",
    "gap_count",
    "provenance",
    "route_name",
    "unresolved_witness_count",
  )

  def __init__(
    self,
    *,
    directory: Path,
    descriptor: PreparedRouteSpoolDescriptor,
    metadata: dict[str, object],
    metadata_bytes: bytes,
  ) -> None:
    self._directory = directory
    self._descriptor = descriptor
    self._metadata_bytes = metadata_bytes
    self.route_name = str(metadata["route_name"])
    self.controls_witness_count = int(metadata["controls_witness_count"])
    self.unresolved_witness_count = int(
      metadata["unresolved_witness_count"],
    )
    self.gap_count = int(metadata["gap_count"])
    # Isolate callers from the dictionary validated above.
    self.provenance = json.loads(_canonical_json_bytes(
      metadata["provenance"],
    ))

  @property
  def frame_count(self) -> int:
    return self._descriptor.frame_count

  @property
  def descriptor(self) -> PreparedRouteSpoolDescriptor:
    return self._descriptor

  def iter_frames(self) -> Iterator[MeasuredLearningFrame]:
    """Yield frames lazily and reject any post-open file mutation."""
    path = self._directory / self._descriptor.filename
    file_stat = _regular_file_lstat(path)
    if file_stat.st_size != self._descriptor.size_bytes:
      raise SpoolFormatError("spool changed size after validation")

    descriptor = os.open(
      path,
      os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
      opened_stat = os.fstat(descriptor)
      if not stat.S_ISREG(opened_stat.st_mode):
        raise SpoolFormatError("spool iterator source is not a regular file")
      if (
        opened_stat.st_dev != file_stat.st_dev
        or opened_stat.st_ino != file_stat.st_ino
      ):
        raise SpoolFormatError("spool changed during iterator open")

      digest = hashlib.sha256()
      header = _read_exact(descriptor, SPOOL_HEADER_SIZE, "header")
      digest.update(header)
      magic, version, record_size, frame_count, metadata_size = (
        _HEADER.unpack(header)
      )
      if (
        magic != SPOOL_MAGIC
        or version != SPOOL_VERSION
        or record_size != SPOOL_RECORD_SIZE
        or frame_count != self.frame_count
        or metadata_size != len(self._metadata_bytes)
      ):
        raise SpoolFormatError("spool header changed after validation")
      metadata_bytes = _read_exact(descriptor, metadata_size, "metadata")
      digest.update(metadata_bytes)
      if metadata_bytes != self._metadata_bytes:
        raise SpoolFormatError("spool metadata changed after validation")

      for _ in range(self.frame_count):
        encoded = _read_exact(descriptor, SPOOL_RECORD_SIZE, "frame")
        digest.update(encoded)
        yield _decode_frame(encoded)

      trailing = os.read(descriptor, 1)
      if trailing:
        raise SpoolFormatError("spool contains trailing bytes")
      if digest.hexdigest() != self._descriptor.sha256:
        raise SpoolFormatError("spool hash changed after validation")
    finally:
      os.close(descriptor)

  def __iter__(self) -> Iterator[MeasuredLearningFrame]:
    return self.iter_frames()

  def cleanup(self) -> None:
    """Remove this validated scratch artifact after canonical consumption."""
    self._descriptor.cleanup(self._directory)


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
) -> PreparedRouteSpoolDescriptor:
  """Encode and atomically publish one bounded, deterministic route spool."""
  root = _private_directory(directory, create=True)
  _validate_route_name(route_name)
  if type(frames) is not tuple:
    raise TypeError("spool frames must be a tuple")
  frame_count = len(frames)
  _validate_frame_bound(frame_count, max_frames)
  counts = _validate_counts(
    controls_witness_count=controls_witness_count,
    unresolved_witness_count=unresolved_witness_count,
    gap_count=gap_count,
    frame_count=frame_count,
  )
  if not isinstance(provenance, Mapping):
    raise SpoolFormatError("spool provenance must be a JSON object")

  metadata: dict[str, object] = {
    "controls_witness_count": counts[0],
    "gap_count": counts[2],
    "provenance": dict(provenance),
    "route_name": route_name,
    "unresolved_witness_count": counts[1],
  }
  metadata_bytes = _canonical_json_bytes(metadata)
  if len(metadata_bytes) > MAX_METADATA_BYTES:
    raise SpoolFormatError("spool metadata exceeds its format bound")

  selected_filename = (
    _default_filename(route_name)
    if filename is None
    else filename
  )
  _validate_filename(selected_filename)
  final_path = root / selected_filename
  header = _HEADER.pack(
    SPOOL_MAGIC,
    SPOOL_VERSION,
    SPOOL_RECORD_SIZE,
    frame_count,
    len(metadata_bytes),
  )
  expected_size = SPOOL_HEADER_SIZE + len(metadata_bytes) + (
    frame_count * SPOOL_RECORD_SIZE
  )
  maximum_size = _maximum_file_size(max_frames)
  if expected_size > maximum_size:
    raise SpoolFormatError("spool exceeds caller frame bound")

  partial_fd = -1
  partial_path: Path | None = None
  try:
    partial_fd, partial_name = tempfile.mkstemp(
      dir=root,
      prefix=f".{selected_filename}.",
      suffix=".partial",
    )
    partial_path = Path(partial_name)
    os.fchmod(partial_fd, 0o600)
    digest = hashlib.sha256()
    _abort_if_requested(abort_requested)
    _write_hashed(partial_fd, header, digest)
    _write_hashed(partial_fd, metadata_bytes, digest)
    for frame in frames:
      _abort_if_requested(abort_requested)
      if type(frame) is not MeasuredLearningFrame:
        raise SpoolFormatError("spool contains an invalid frame type")
      try:
        encoded = _encode_frame(frame)
      except (OverflowError, struct.error) as error:
        raise SpoolFormatError("spool frame is not encodable") from error
      _write_hashed(partial_fd, encoded, digest)
    _abort_if_requested(abort_requested)
    os.fsync(partial_fd)
    os.close(partial_fd)
    partial_fd = -1
    os.replace(partial_path, final_path)
    partial_path = None
    _fsync_directory(root)
    return PreparedRouteSpoolDescriptor(
      route_name=route_name,
      filename=selected_filename,
      sha256=digest.hexdigest(),
      size_bytes=expected_size,
      frame_count=frame_count,
    )
  except BaseException:
    if partial_fd >= 0:
      os.close(partial_fd)
    if partial_path is not None:
      try:
        partial_path.unlink()
      except FileNotFoundError:
        pass
    raise


def open_prepared_route_spool(
  directory: str | Path,
  descriptor: PreparedRouteSpoolDescriptor,
  *,
  expected_route_name: str,
  max_frames: int,
) -> PreparedRouteSpool:
  """Validate a complete spool before exposing its route facts and iterator."""
  root = _private_directory(directory, create=False)
  _validate_descriptor(descriptor)
  _validate_route_name(expected_route_name)
  if descriptor.route_name != expected_route_name:
    raise SpoolFormatError("spool descriptor route mismatch")
  _validate_frame_bound(descriptor.frame_count, max_frames)
  if descriptor.size_bytes > _maximum_file_size(max_frames):
    raise SpoolFormatError("spool descriptor exceeds caller frame bound")

  path = root / descriptor.filename
  file_stat = _regular_file_lstat(path)
  if file_stat.st_size != descriptor.size_bytes:
    raise SpoolFormatError("spool descriptor size mismatch")
  if _hash_regular_file(path, file_stat) != descriptor.sha256:
    raise SpoolFormatError("spool descriptor hash mismatch")

  with path.open("rb") as stream:
    header = stream.read(SPOOL_HEADER_SIZE)
    if len(header) != SPOOL_HEADER_SIZE:
      raise SpoolFormatError("spool header is truncated")
    magic, version, record_size, frame_count, metadata_size = (
      _HEADER.unpack(header)
    )
    if magic != SPOOL_MAGIC:
      raise SpoolFormatError("spool magic is invalid")
    if version != SPOOL_VERSION:
      raise SpoolFormatError("spool version is unsupported")
    if record_size != SPOOL_RECORD_SIZE:
      raise SpoolFormatError("spool record size is incompatible")
    if frame_count != descriptor.frame_count:
      raise SpoolFormatError("spool header frame count mismatch")
    _validate_frame_bound(frame_count, max_frames)
    if metadata_size > MAX_METADATA_BYTES:
      raise SpoolFormatError("spool metadata exceeds its format bound")
    expected_size = SPOOL_HEADER_SIZE + metadata_size + (
      frame_count * SPOOL_RECORD_SIZE
    )
    if expected_size != descriptor.size_bytes:
      raise SpoolFormatError("spool size does not match its header")
    metadata_bytes = stream.read(metadata_size)
    if len(metadata_bytes) != metadata_size:
      raise SpoolFormatError("spool metadata is truncated")
    metadata = _decode_metadata(metadata_bytes)
    if metadata["route_name"] != expected_route_name:
      raise SpoolFormatError("spool metadata route mismatch")
    _validate_counts(
      controls_witness_count=metadata["controls_witness_count"],
      unresolved_witness_count=metadata["unresolved_witness_count"],
      gap_count=metadata["gap_count"],
      frame_count=frame_count,
    )

  return PreparedRouteSpool(
    directory=root,
    descriptor=descriptor,
    metadata=metadata,
    metadata_bytes=metadata_bytes,
  )


def _encode_frame(frame: MeasuredLearningFrame) -> bytes:
  return _FRAME.pack(
    frame.sample_mono_ns,
    frame.response_mono_ns,
    frame.applied_report_mono_ns,
    frame.applied_effective_mono_ns,
    frame.speed_mps,
    frame.steering_angle_deg,
    frame.steering_rate_deg_s,
    frame.steering_torque,
    frame.applied_torque,
    frame.angle_offset_deg,
    frame.steer_ratio,
    frame.stiffness_factor,
    frame.roll_rad,
    frame.steering_pressed,
    frame.standstill,
    frame.steer_fault_temporary,
    frame.steer_fault_permanent,
    frame.can_valid,
    frame.can_timeout,
    frame.lateral_active,
    frame.live_parameters_valid,
    frame.angle_offset_valid,
    frame.steer_ratio_valid,
    frame.stiffness_factor_valid,
    frame.inputs_valid,
  )


def _decode_frame(encoded: bytes) -> MeasuredLearningFrame:
  values = _FRAME.unpack(encoded)
  return MeasuredLearningFrame(
    sample_mono_ns=values[0],
    response_mono_ns=values[1],
    applied_report_mono_ns=values[2],
    applied_effective_mono_ns=values[3],
    speed_mps=values[4],
    steering_angle_deg=values[5],
    steering_rate_deg_s=values[6],
    steering_torque=values[7],
    applied_torque=values[8],
    angle_offset_deg=values[9],
    steer_ratio=values[10],
    stiffness_factor=values[11],
    roll_rad=values[12],
    steering_pressed=values[13],
    standstill=values[14],
    steer_fault_temporary=values[15],
    steer_fault_permanent=values[16],
    can_valid=values[17],
    can_timeout=values[18],
    lateral_active=values[19],
    live_parameters_valid=values[20],
    angle_offset_valid=values[21],
    steer_ratio_valid=values[22],
    stiffness_factor_valid=values[23],
    inputs_valid=values[24],
  )


def _canonical_json_bytes(value: object) -> bytes:
  try:
    return json.dumps(
      value,
      allow_nan=False,
      separators=(",", ":"),
      sort_keys=True,
    ).encode("utf-8")
  except (TypeError, ValueError) as error:
    raise SpoolFormatError("spool metadata is not canonical JSON") from error


def _decode_metadata(encoded: bytes) -> dict[str, object]:
  try:
    payload: Any = json.loads(encoded)
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise SpoolFormatError("spool metadata is invalid JSON") from error
  if type(payload) is not dict or set(payload) != _METADATA_KEYS:
    raise SpoolFormatError("spool metadata shape is invalid")
  if encoded != _canonical_json_bytes(payload):
    raise SpoolFormatError("spool metadata is not canonical")
  _validate_route_name(payload["route_name"])
  if type(payload["provenance"]) is not dict:
    raise SpoolFormatError("spool provenance must be a JSON object")
  return payload


def _default_filename(route_name: str) -> str:
  identity = hashlib.sha256(route_name.encode("utf-8")).hexdigest()[:24]
  return f"prepared-route-{identity}.blatspool"


def _validate_filename(filename: object) -> None:
  if type(filename) is not str or _SAFE_FILENAME.fullmatch(filename) is None:
    raise SpoolFormatError("spool filename is unsafe")
  if filename in (".", ".."):
    raise SpoolFormatError("spool filename is unsafe")


def _validate_route_name(route_name: object) -> None:
  if (
    type(route_name) is not str
    or not route_name
    or len(route_name.encode("utf-8")) > 1024
    or "\x00" in route_name
  ):
    raise SpoolFormatError("spool route name is invalid")


def _validate_descriptor(descriptor: object) -> None:
  if type(descriptor) is not PreparedRouteSpoolDescriptor:
    raise SpoolFormatError("spool descriptor type is invalid")
  _validate_route_name(descriptor.route_name)
  _validate_filename(descriptor.filename)
  if type(descriptor.sha256) is not str or _SHA256.fullmatch(
    descriptor.sha256,
  ) is None:
    raise SpoolFormatError("spool descriptor hash is invalid")
  if type(descriptor.size_bytes) is not int or descriptor.size_bytes < 0:
    raise SpoolFormatError("spool descriptor size is invalid")
  if type(descriptor.frame_count) is not int or descriptor.frame_count < 0:
    raise SpoolFormatError("spool descriptor frame count is invalid")


def _validate_frame_bound(frame_count: object, max_frames: object) -> None:
  if type(max_frames) is not int or max_frames < 0:
    raise SpoolFormatError("spool max_frames is invalid")
  if (
    type(frame_count) is not int
    or frame_count < 0
    or frame_count > max_frames
  ):
    raise SpoolFormatError("spool frame count exceeds caller bound")


def _validate_counts(
  *,
  controls_witness_count: object,
  unresolved_witness_count: object,
  gap_count: object,
  frame_count: int,
) -> tuple[int, int, int]:
  values = (
    controls_witness_count,
    unresolved_witness_count,
    gap_count,
  )
  if any(type(value) is not int or value < 0 for value in values):
    raise SpoolFormatError("spool route counts are invalid")
  controls = int(controls_witness_count)
  unresolved = int(unresolved_witness_count)
  gaps = int(gap_count)
  if frame_count > controls or unresolved > controls or gaps > controls:
    raise SpoolFormatError("spool route counts are inconsistent")
  return controls, unresolved, gaps


def _maximum_file_size(max_frames: int) -> int:
  _validate_frame_bound(0, max_frames)
  return SPOOL_HEADER_SIZE + MAX_METADATA_BYTES + (
    max_frames * SPOOL_RECORD_SIZE
  )


def _private_directory(directory: str | Path, *, create: bool) -> Path:
  root = Path(directory)
  if create:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
  try:
    directory_stat = root.lstat()
  except FileNotFoundError as error:
    raise SpoolFormatError("spool directory does not exist") from error
  if root.is_symlink() or not stat.S_ISDIR(directory_stat.st_mode):
    raise SpoolFormatError("spool directory is not a real directory")
  if stat.S_IMODE(directory_stat.st_mode) & 0o077:
    raise SpoolFormatError("spool directory is not private")
  return root


def _regular_file_lstat(path: Path) -> os.stat_result:
  try:
    file_stat = path.lstat()
  except FileNotFoundError as error:
    raise SpoolFormatError("spool file is missing") from error
  if path.is_symlink() or not stat.S_ISREG(file_stat.st_mode):
    raise SpoolFormatError("spool file is not a regular non-symlink file")
  return file_stat


def _hash_regular_file(path: Path, expected_stat: os.stat_result) -> str:
  descriptor = os.open(
    path,
    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
  )
  try:
    opened_stat = os.fstat(descriptor)
    if (
      not stat.S_ISREG(opened_stat.st_mode)
      or opened_stat.st_dev != expected_stat.st_dev
      or opened_stat.st_ino != expected_stat.st_ino
      or opened_stat.st_size != expected_stat.st_size
    ):
      raise SpoolFormatError("spool changed while opening")
    digest = hashlib.sha256()
    while True:
      block = os.read(descriptor, 1024 * 1024)
      if not block:
        break
      digest.update(block)
    return digest.hexdigest()
  finally:
    os.close(descriptor)


def _read_exact(descriptor: int, size: int, section: str) -> bytes:
  encoded = bytearray()
  while len(encoded) < size:
    block = os.read(descriptor, size - len(encoded))
    if not block:
      raise SpoolFormatError(f"spool {section} is truncated")
    encoded.extend(block)
  return bytes(encoded)


def _write_hashed(
  descriptor: int,
  encoded: bytes,
  digest: Any,
) -> None:
  view = memoryview(encoded)
  while view:
    written = os.write(descriptor, view)
    if written <= 0:
      raise OSError("short spool write")
    digest.update(view[:written])
    view = view[written:]


def _abort_if_requested(abort_requested: Callable[[], bool]) -> None:
  if abort_requested():
    raise SpoolFormatError("spool write cancelled")


def _fsync_directory(directory: Path) -> None:
  descriptor = os.open(directory, os.O_RDONLY)
  try:
    os.fsync(descriptor)
  finally:
    os.close(descriptor)

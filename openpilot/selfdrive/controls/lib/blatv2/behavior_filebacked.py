"""Private bounded-memory preparation views for offline behavior replay.

The wire format in this module is scratch space, never an activation or
persistence artifact.  It exists to keep a route's 100-Hz samples out of the
Python heap while the shared segmentation and metric authorities make several
deterministic passes over them.  Records are fixed-width little-endian binary;
all numerical fields retain their IEEE-754 float64 representation.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Any, overload

from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import (
  BehaviorSample,
  BehaviorSourceIdentity,
  EventLocator,
  ManeuverClass,
  ManeuverPhase,
  canonical_json,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_segmentation import (
  BehaviorPhaseSpan,
  EventCoverage,
  SegmentationConfig,
  WindowObservability,
  segment_behavior_spans,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_metrics import (
  BehaviorMetricConfig,
  WindowMetricSet,
  score_sample_view,
)


_MAGIC = b"BLATBS01"
_VERSION = 1
_HEADER = struct.Struct("<8sIIQ")
_RECORD = struct.Struct("<Q15dB7x")
_FLAG_ACTUATOR_CONSTRAINED = 1 << 0
_FLAG_LATERAL_ACTIVE = 1 << 1
_FLAG_INPUTS_VALID = 1 << 2
_FLAG_STEERING_PRESSED = 1 << 3
_FLAG_CONTROLLER_FAULT = 1 << 4
_FLAG_DRIVER_INTERVENTION_ONSET = 1 << 5
_KNOWN_FLAGS = (
  _FLAG_ACTUATOR_CONSTRAINED
  | _FLAG_LATERAL_ACTIVE
  | _FLAG_INPUTS_VALID
  | _FLAG_STEERING_PRESSED
  | _FLAG_CONTROLLER_FAULT
  | _FLAG_DRIVER_INTERVENTION_ONSET
)
_READ_CHUNK_RECORDS = 512


class BehaviorScratchError(RuntimeError):
  pass


def _encode_sample(sample: BehaviorSample) -> bytes:
  flags = (
    (_FLAG_ACTUATOR_CONSTRAINED if sample.actuator_constrained else 0)
    | (_FLAG_LATERAL_ACTIVE if sample.lateral_active else 0)
    | (_FLAG_INPUTS_VALID if sample.inputs_valid else 0)
    | (_FLAG_STEERING_PRESSED if sample.steering_pressed else 0)
    | (_FLAG_CONTROLLER_FAULT if sample.controller_fault else 0)
    | (_FLAG_DRIVER_INTERVENTION_ONSET if sample.driver_intervention_onset else 0)
  )
  return _RECORD.pack(
    sample.mono_time_ns,
    sample.route_time_s,
    sample.speed_mps,
    sample.scalar_curvature_1pm,
    sample.desired_curvature_1pm,
    sample.anchored_curvature_1pm,
    sample.desired_rack_angle_deg,
    sample.desired_rack_rate_deg_s,
    sample.desired_rack_accel_deg_s2,
    sample.measured_curvature_1pm,
    sample.measured_rack_angle_deg,
    sample.measured_rack_rate_deg_s,
    sample.measured_rack_accel_deg_s2,
    sample.raw_requested_torque,
    sample.envelope_applied_torque,
    sample.torque_headroom,
    flags,
  )


def _decode_sample(encoded: bytes | memoryview) -> BehaviorSample:
  values = _RECORD.unpack(encoded)
  flags = values[-1]
  if flags & ~_KNOWN_FLAGS:
    raise BehaviorScratchError("behavior scratch record contains unknown flags")
  return BehaviorSample(
    mono_time_ns=values[0],
    route_time_s=values[1],
    speed_mps=values[2],
    scalar_curvature_1pm=values[3],
    desired_curvature_1pm=values[4],
    anchored_curvature_1pm=values[5],
    desired_rack_angle_deg=values[6],
    desired_rack_rate_deg_s=values[7],
    desired_rack_accel_deg_s2=values[8],
    measured_curvature_1pm=values[9],
    measured_rack_angle_deg=values[10],
    measured_rack_rate_deg_s=values[11],
    measured_rack_accel_deg_s2=values[12],
    raw_requested_torque=values[13],
    envelope_applied_torque=values[14],
    torque_headroom=values[15],
    actuator_constrained=bool(flags & _FLAG_ACTUATOR_CONSTRAINED),
    lateral_active=bool(flags & _FLAG_LATERAL_ACTIVE),
    inputs_valid=bool(flags & _FLAG_INPUTS_VALID),
    steering_pressed=bool(flags & _FLAG_STEERING_PRESSED),
    controller_fault=bool(flags & _FLAG_CONTROLLER_FAULT),
    driver_intervention_onset=bool(flags & _FLAG_DRIVER_INTERVENTION_ONSET),
  )


class BehaviorSampleReader(Sequence[BehaviorSample]):
  """Random-access, re-iterable view over one immutable scratch file."""

  __slots__ = (
    "_cache", "_cache_start", "_descriptor", "_identity", "_path", "_record_count",
  )

  def __init__(self, path: Path) -> None:
    self._path = path
    self._descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
      identity = os.fstat(self._descriptor)
      encoded = os.pread(self._descriptor, _HEADER.size, 0)
      if len(encoded) != _HEADER.size:
        raise BehaviorScratchError("behavior scratch header is truncated")
      magic, version, record_size, record_count = _HEADER.unpack(encoded)
      if magic != _MAGIC or version != _VERSION or record_size != _RECORD.size:
        raise BehaviorScratchError("behavior scratch header is incompatible")
      if identity.st_size != _HEADER.size + record_count * _RECORD.size:
        raise BehaviorScratchError("behavior scratch size and record count disagree")
      self._identity = (
        identity.st_dev,
        identity.st_ino,
        identity.st_size,
        identity.st_mtime_ns,
        identity.st_ctime_ns,
      )
      self._record_count = record_count
      self._cache_start = 0
      self._cache = b""
    except BaseException:
      os.close(self._descriptor)
      self._descriptor = -1
      raise

  def close(self) -> None:
    if self._descriptor >= 0:
      os.close(self._descriptor)
      self._descriptor = -1

  def _check_open(self) -> None:
    if self._descriptor < 0:
      raise BehaviorScratchError("behavior scratch reader is closed")
    observed = os.fstat(self._descriptor)
    if (
      observed.st_dev,
      observed.st_ino,
      observed.st_size,
      observed.st_mtime_ns,
      observed.st_ctime_ns,
    ) != self._identity:
      raise BehaviorScratchError("behavior scratch changed while open")

  def __len__(self) -> int:
    return self._record_count

  @overload
  def __getitem__(self, index: int) -> BehaviorSample: ...

  @overload
  def __getitem__(self, index: slice) -> BehaviorSampleSpan: ...

  def __getitem__(self, index: int | slice) -> BehaviorSample | BehaviorSampleSpan:
    if isinstance(index, slice):
      start, stop, step = index.indices(self._record_count)
      if step != 1:
        raise ValueError("behavior sample spans require a unit stride")
      return BehaviorSampleSpan(self, start, stop)
    resolved = index + self._record_count if index < 0 else index
    if not 0 <= resolved < self._record_count:
      raise IndexError(index)
    cache_count = len(self._cache) // _RECORD.size
    if not self._cache_start <= resolved < self._cache_start + cache_count:
      self._check_open()
      self._cache_start = resolved
      count = min(_READ_CHUNK_RECORDS, self._record_count - resolved)
      offset = _HEADER.size + resolved * _RECORD.size
      self._cache = os.pread(self._descriptor, count * _RECORD.size, offset)
      if len(self._cache) != count * _RECORD.size:
        raise BehaviorScratchError("behavior scratch records are truncated")
      self._check_open()
    left = (resolved - self._cache_start) * _RECORD.size
    return _decode_sample(memoryview(self._cache)[left:left + _RECORD.size])

  def __iter__(self) -> Iterator[BehaviorSample]:
    yield from self.iter_range(0, self._record_count)

  def iter_range(self, start: int, end: int) -> Iterator[BehaviorSample]:
    """Read a bounded record block at a time for repeated metric passes."""
    if start < 0 or end < start or end > self._record_count:
      raise ValueError("behavior sample range bounds are invalid")
    self._check_open()
    try:
      cursor = start
      while cursor < end:
        count = min(_READ_CHUNK_RECORDS, end - cursor)
        offset = _HEADER.size + cursor * _RECORD.size
        encoded = os.pread(self._descriptor, count * _RECORD.size, offset)
        if len(encoded) != count * _RECORD.size:
          raise BehaviorScratchError("behavior scratch records are truncated")
        view = memoryview(encoded)
        for local_index in range(count):
          left = local_index * _RECORD.size
          yield _decode_sample(view[left:left + _RECORD.size])
        cursor += count
    finally:
      self._check_open()


@dataclass(frozen=True, slots=True)
class BehaviorSampleSpan(Sequence[BehaviorSample]):
  reader: BehaviorSampleReader
  start: int
  end: int

  def __post_init__(self) -> None:
    if self.start < 0 or self.end < self.start or self.end > len(self.reader):
      raise ValueError("behavior sample span bounds are invalid")

  def __len__(self) -> int:
    return self.end - self.start

  @overload
  def __getitem__(self, index: int) -> BehaviorSample: ...

  @overload
  def __getitem__(self, index: slice) -> BehaviorSampleSpan: ...

  def __getitem__(self, index: int | slice) -> BehaviorSample | BehaviorSampleSpan:
    if isinstance(index, slice):
      start, stop, step = index.indices(len(self))
      if step != 1:
        raise ValueError("behavior sample spans require a unit stride")
      return BehaviorSampleSpan(self.reader, self.start + start, self.start + stop)
    resolved = index + len(self) if index < 0 else index
    if not 0 <= resolved < len(self):
      raise IndexError(index)
    return self.reader[self.start + resolved]

  def __iter__(self) -> Iterator[BehaviorSample]:
    yield from self.reader.iter_range(self.start, self.end)


class BehaviorSampleScratch:
  """Owned temporary sample store, removed on every exit path."""

  __slots__ = ("_count", "_directory", "_output", "path", "reader")

  def __init__(self, parent: Path | None = None) -> None:
    self._directory = Path(tempfile.mkdtemp(prefix="blatv2-behavior-", dir=parent))
    self.path = self._directory / "samples.bin"
    self.reader: BehaviorSampleReader | None = None
    self._output: Any = None
    self._count = 0
    try:
      self._output = self.path.open("xb")
      self._output.write(_HEADER.pack(_MAGIC, _VERSION, _RECORD.size, 0))
    except BaseException:
      self.close()
      raise

  def __enter__(self) -> BehaviorSampleScratch:
    if self._output is None and self.reader is None:
      raise BehaviorScratchError("behavior scratch is closed")
    return self

  def __exit__(self, *_: object) -> None:
    self.close()

  def append(self, sample: BehaviorSample) -> None:
    """Append exactly one replay result without retaining it in Python."""
    if self._output is None:
      raise BehaviorScratchError("behavior scratch is already finalized")
    self._output.write(_encode_sample(sample))
    self._count += 1

  def extend(self, samples: Iterable[BehaviorSample]) -> None:
    for sample in samples:
      self.append(sample)

  def finish(self) -> BehaviorSampleReader:
    if self.reader is not None:
      return self.reader
    if self._output is None:
      raise BehaviorScratchError("behavior scratch is closed")
    try:
      self._output.flush()
      os.fsync(self._output.fileno())
      self._output.seek(0)
      self._output.write(_HEADER.pack(_MAGIC, _VERSION, _RECORD.size, self._count))
      self._output.flush()
      os.fsync(self._output.fileno())
    finally:
      self._output.close()
      self._output = None
    self.reader = BehaviorSampleReader(self.path)
    return self.reader

  def close(self) -> None:
    if self._output is not None:
      self._output.close()
      self._output = None
    if self.reader is not None:
      self.reader.close()
      self.reader = None
    directory = getattr(self, "_directory", None)
    if directory is not None and directory.exists():
      shutil.rmtree(directory)


@dataclass(frozen=True, slots=True)
class FileBackedBehaviorWindow:
  route_id: str
  source: BehaviorSourceIdentity
  descriptor: BehaviorPhaseSpan
  samples: BehaviorSampleSpan

  @property
  def window_id(self) -> str:
    return self.descriptor.window_id

  @property
  def maneuver_class(self) -> ManeuverClass:
    return self.descriptor.maneuver_class

  @property
  def phase(self) -> ManeuverPhase:
    return self.descriptor.phase

  @property
  def event_locators(self) -> tuple[EventLocator, ...]:
    return self.descriptor.event_locators

  @property
  def observability(self) -> WindowObservability:
    return self.descriptor.observability


@dataclass(frozen=True, slots=True)
class FileBackedSegmentationResult:
  route_id: str
  source: BehaviorSourceIdentity
  config_sha256: str
  windows: tuple[FileBackedBehaviorWindow, ...]
  event_coverage: tuple[EventCoverage, ...]
  unassigned_sample_indices: Sequence[int]

  def _canonical_chunks(self) -> Iterator[str]:
    yield '{"configSha256":'
    yield canonical_json(self.config_sha256)
    yield ',"eventCoverage":'
    yield canonical_json([value.to_dict() for value in self.event_coverage])
    yield ',"routeId":'
    yield canonical_json(self.route_id)
    yield ',"sourceIdentitySha256":'
    yield canonical_json(self.source.sha256)
    yield ',"unassignedSampleIndices":['
    for index, sample_index in enumerate(self.unassigned_sample_indices):
      if index:
        yield ","
      yield str(sample_index)
    yield "]"
    yield ',"windows":['
    for window_index, item in enumerate(self.windows):
      if window_index:
        yield ","
      descriptor = item.descriptor
      yield '{"endSampleIndexExclusive":'
      yield str(descriptor.end_sample_index_exclusive)
      yield ',"observability":'
      yield canonical_json(descriptor.observability.to_dict())
      yield ',"startSampleIndex":'
      yield str(descriptor.start_sample_index)
      yield ',"window":{"eventLocators":'
      yield canonical_json([event.to_dict() for event in descriptor.event_locators])
      yield ',"maneuverClass":'
      yield canonical_json(descriptor.maneuver_class.value)
      yield ',"phase":'
      yield canonical_json(descriptor.phase.value)
      yield ',"routeId":'
      yield canonical_json(self.route_id)
      yield ',"samples":['
      for sample_index, sample in enumerate(item.samples):
        if sample_index:
          yield ","
        yield canonical_json(sample.to_dict())
      yield '],"source":'
      yield canonical_json(self.source.to_dict())
      yield ',"windowId":'
      yield canonical_json(descriptor.window_id)
      yield "}}"
    yield "]}"

  @property
  def sha256(self) -> str:
    digest = hashlib.sha256()
    for chunk in self._canonical_chunks():
      digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()

  def descriptor_dict(self) -> dict[str, Any]:
    return {
      "configSha256": self.config_sha256,
      "eventCoverage": [value.to_dict() for value in self.event_coverage],
      "routeId": self.route_id,
      "sourceIdentitySha256": self.source.sha256,
      "unassignedSampleIndices": list(self.unassigned_sample_indices),
      "windows": [
        {
          "endSampleIndexExclusive": item.descriptor.end_sample_index_exclusive,
          "observability": item.observability.to_dict(),
          "startSampleIndex": item.descriptor.start_sample_index,
          "windowId": item.window_id,
        }
        for item in self.windows
      ],
    }


def segment_file_backed_behavior_route(
  route_id: str,
  source: BehaviorSourceIdentity,
  samples: BehaviorSampleReader,
  event_locators: Iterable[EventLocator],
  config: SegmentationConfig,
) -> FileBackedSegmentationResult:
  segmented = segment_behavior_spans(route_id, source, samples, event_locators, config)
  return FileBackedSegmentationResult(
    route_id=route_id,
    source=source,
    config_sha256=segmented.config_sha256,
    windows=tuple(
      FileBackedBehaviorWindow(
        route_id=route_id,
        source=source,
        descriptor=span,
        samples=BehaviorSampleSpan(
          samples,
          span.start_sample_index,
          span.end_sample_index_exclusive,
        ),
      )
      for span in segmented.spans
    ),
    event_coverage=segmented.event_coverage,
    unassigned_sample_indices=segmented.unassigned_sample_indices,
  )


def score_file_backed_window(
  window: FileBackedBehaviorWindow,
  config: BehaviorMetricConfig,
) -> WindowMetricSet:
  """Delegate file-backed scoring to the sole shared metric authority."""
  return score_sample_view(
    route_id=window.route_id,
    window_id=window.window_id,
    source_identity_sha256=window.source.sha256,
    maneuver_class=window.maneuver_class,
    phase=window.phase,
    samples=window.samples,
    config=config,
  )

"""Deterministic, non-actuating segmentation of BLaTv2 behavior evidence.

The driving model's anchored curvature is the only phase signal.  Logger
events merely locate interesting context; they never label a phase or vote on
controller quality.  No lane-line signal is accepted by this interface.

All thresholds and timing rules live in :class:`SegmentationConfig`.  The
``provisional_offline_gate`` values are deliberately named and versioned: they
are reproducibility constants for the offline gate, not controller tuning.
Physical phase windows never overlap.  Event coverage is reported separately,
so extending a logger window cannot duplicate samples in metric input.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Iterator, Sequence
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from enum import StrEnum
import hashlib
import math
import mmap
import os
import struct
import tempfile
from typing import Any

from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import (
  BehaviorSample,
  BehaviorSourceIdentity,
  BehaviorWindow,
  EventLocator,
  ManeuverClass,
  ManeuverPhase,
  canonical_json,
)


class BoundaryReason(StrEnum):
  """Why a physical phase boundary is not directly observable."""

  PHASE_ONSET_PRECEDES_AVAILABLE_EVIDENCE = "phase_onset_precedes_available_evidence"
  PHASE_INCOMPLETE_AT_ROUTE_END = "phase_incomplete_at_route_end"
  PHASE_INCOMPLETE_AT_DATA_GAP = "phase_incomplete_at_data_gap"
  PHASE_INCOMPLETE_AT_INVALID_INPUT = "phase_incomplete_at_invalid_input"
  DRIVER_INTERVENTION_CENSOR = "driver_intervention_censor"


class EventCoverageStop(StrEnum):
  PHASE_COMPLETE = "phase_complete"
  MAXIMUM_EXTENSION = "maximum_extension"
  NEXT_EVENT_WINDOW = "next_event_window"
  ROUTE_BOUNDARY = "route_boundary"
  DATA_GAP_OR_INVALID = "data_gap_or_invalid"
  NO_MANEUVER = "no_maneuver"


class SegmentationResourceError(ValueError):
  """A route exceeds its versioned offline segmentation work authority."""


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
  """Every deterministic phase, coverage, and work limit used here.

  Work limits reject an adversarial route; they never truncate, sample, or
  reinterpret it. They are part of the hashed segmentation authority so a
  limit change cannot silently alter the population admitted to training.
  """

  schema_version: int
  reference_zero_threshold_1pm: float
  quasi_steady_rate_threshold_1pm_s: float
  monotonic_progress_epsilon_1pm_s: float
  turn_class_curvature_threshold_1pm: float
  direct_handoff_min_peak_curvature_1pm: float
  direct_handoff_max_neutral_duration_s: float
  minimum_phase_duration_s: float
  minimum_phase_samples: int
  maximum_phase_extension_s: float
  maximum_sample_gap_s: float
  turn_in_crossing_fraction: float
  release_onset_fraction: float
  maximum_raw_phase_spans: int
  maximum_phase_windows: int
  maximum_event_locators: int
  maximum_event_phase_attachments: int

  def __post_init__(self) -> None:
    if self.schema_version <= 0:
      raise ValueError("schema_version must be positive")
    positive = (
      self.reference_zero_threshold_1pm,
      self.quasi_steady_rate_threshold_1pm_s,
      self.monotonic_progress_epsilon_1pm_s,
      self.turn_class_curvature_threshold_1pm,
      self.direct_handoff_min_peak_curvature_1pm,
      self.direct_handoff_max_neutral_duration_s,
      self.minimum_phase_duration_s,
      self.maximum_phase_extension_s,
      self.maximum_sample_gap_s,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in positive):
      raise ValueError("segmentation thresholds and durations must be finite and positive")
    if self.minimum_phase_samples <= 0:
      raise ValueError("minimum_phase_samples must be positive")
    work_limits = (
      self.maximum_raw_phase_spans,
      self.maximum_phase_windows,
      self.maximum_event_locators,
      self.maximum_event_phase_attachments,
    )
    if any(type(value) is not int or value <= 0 for value in work_limits):
      raise ValueError("segmentation work limits must be positive integers")
    if self.maximum_phase_windows > self.maximum_raw_phase_spans:
      raise ValueError("phase-window limit cannot exceed raw-span limit")
    if not 0.0 < self.turn_in_crossing_fraction < 1.0:
      raise ValueError("turn_in_crossing_fraction must be in (0, 1)")
    if not 0.0 < self.release_onset_fraction < 1.0:
      raise ValueError("release_onset_fraction must be in (0, 1)")

  @classmethod
  def provisional_offline_gate(cls) -> SegmentationConfig:
    """Version-2 reproducibility constants, not controller feel dials.

    The 50% turn-in and 90% release fractions are the committed metric
    conventions.  Remaining values are conservative evidence-discrimination
    constants and must be revised by a version bump, never silently.
    """
    return cls(
      schema_version=2,
      reference_zero_threshold_1pm=0.0005,
      quasi_steady_rate_threshold_1pm_s=0.001,
      monotonic_progress_epsilon_1pm_s=0.00001,
      turn_class_curvature_threshold_1pm=0.02,
      direct_handoff_min_peak_curvature_1pm=0.002,
      direct_handoff_max_neutral_duration_s=0.30,
      minimum_phase_duration_s=0.05,
      minimum_phase_samples=3,
      maximum_phase_extension_s=8.0,
      maximum_sample_gap_s=0.015,
      turn_in_crossing_fraction=0.5,
      release_onset_fraction=0.9,
      maximum_raw_phase_spans=65_536,
      maximum_phase_windows=4_096,
      maximum_event_locators=4_096,
      maximum_event_phase_attachments=65_536,
    )

  def to_dict(self) -> dict[str, Any]:
    return {
      "directHandoffMaxNeutralDurationS": self.direct_handoff_max_neutral_duration_s,
      "directHandoffMinPeakCurvature1pm": self.direct_handoff_min_peak_curvature_1pm,
      "maximumEventLocators": self.maximum_event_locators,
      "maximumEventPhaseAttachments": self.maximum_event_phase_attachments,
      "maximumPhaseExtensionS": self.maximum_phase_extension_s,
      "maximumPhaseWindows": self.maximum_phase_windows,
      "maximumRawPhaseSpans": self.maximum_raw_phase_spans,
      "maximumSampleGapS": self.maximum_sample_gap_s,
      "minimumPhaseDurationS": self.minimum_phase_duration_s,
      "minimumPhaseSamples": self.minimum_phase_samples,
      "monotonicProgressEpsilon1pmS": self.monotonic_progress_epsilon_1pm_s,
      "quasiSteadyRateThreshold1pmS": self.quasi_steady_rate_threshold_1pm_s,
      "referenceZeroThreshold1pm": self.reference_zero_threshold_1pm,
      "releaseOnsetFraction": self.release_onset_fraction,
      "schemaVersion": self.schema_version,
      "turnClassCurvatureThreshold1pm": self.turn_class_curvature_threshold_1pm,
      "turnInCrossingFraction": self.turn_in_crossing_fraction,
    }

  @property
  def sha256(self) -> str:
    return hashlib.sha256(canonical_json(self.to_dict()).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class WindowObservability:
  onset_observed: bool
  completion_observed: bool
  metric_crossing_observed: bool | None
  reasons: tuple[BoundaryReason, ...]
  driver_censor_mono_time_ns: int | None

  def to_dict(self) -> dict[str, Any]:
    return {
      "completionObserved": self.completion_observed,
      "driverCensorMonoTimeNs": self.driver_censor_mono_time_ns,
      "metricCrossingObserved": self.metric_crossing_observed,
      "onsetObserved": self.onset_observed,
      "reasons": [reason.value for reason in self.reasons],
    }


@dataclass(frozen=True, slots=True)
class SegmentedBehaviorWindow:
  """Metric-ready physical phase plus non-judgmental context metadata."""

  window: BehaviorWindow
  start_sample_index: int
  end_sample_index_exclusive: int
  observability: WindowObservability

  def __post_init__(self) -> None:
    if self.start_sample_index < 0 or self.end_sample_index_exclusive <= self.start_sample_index:
      raise ValueError("segmented sample bounds must be a non-empty half-open interval")

  def to_dict(self) -> dict[str, Any]:
    return {
      "endSampleIndexExclusive": self.end_sample_index_exclusive,
      "observability": self.observability.to_dict(),
      "startSampleIndex": self.start_sample_index,
      "window": self.window.to_dict(),
    }


@dataclass(frozen=True, slots=True)
class EventCoverage:
  locator: EventLocator
  physical_start_mono_time_ns: int | None
  physical_end_mono_time_ns: int | None
  nominal_start_mono_time_ns: int
  nominal_end_mono_time_ns: int
  extended_beyond_nominal_end: bool
  onset_precedes_available_evidence: bool
  phase_incomplete_at_boundary: bool
  stop_reason: EventCoverageStop

  def to_dict(self) -> dict[str, Any]:
    return {
      "extendedBeyondNominalEnd": self.extended_beyond_nominal_end,
      "locator": self.locator.to_dict(),
      "nominalEndMonoTimeNs": self.nominal_end_mono_time_ns,
      "nominalStartMonoTimeNs": self.nominal_start_mono_time_ns,
      "onsetPrecedesAvailableEvidence": self.onset_precedes_available_evidence,
      "phaseIncompleteAtBoundary": self.phase_incomplete_at_boundary,
      "physicalEndMonoTimeNs": self.physical_end_mono_time_ns,
      "physicalStartMonoTimeNs": self.physical_start_mono_time_ns,
      "stopReason": self.stop_reason.value,
    }


@dataclass(frozen=True, slots=True)
class SegmentationResult:
  route_id: str
  source_identity_sha256: str
  config_sha256: str
  windows: tuple[SegmentedBehaviorWindow, ...]
  event_coverage: tuple[EventCoverage, ...]
  unassigned_sample_indices: tuple[int, ...]

  @property
  def behavior_windows(self) -> tuple[BehaviorWindow, ...]:
    """The direct input expected by ``behavior_metrics.score_behavior``."""
    return tuple(item.window for item in self.windows)

  def to_dict(self) -> dict[str, Any]:
    return {
      "configSha256": self.config_sha256,
      "eventCoverage": [coverage.to_dict() for coverage in self.event_coverage],
      "routeId": self.route_id,
      "sourceIdentitySha256": self.source_identity_sha256,
      "unassignedSampleIndices": list(self.unassigned_sample_indices),
      "windows": [window.to_dict() for window in self.windows],
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())

  @property
  def sha256(self) -> str:
    return hashlib.sha256(self.to_json().encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class BehaviorPhaseSpan:
  """Bounded phase descriptor shared by eager and file-backed consumers."""

  window_id: str
  maneuver_class: ManeuverClass
  phase: ManeuverPhase
  start_sample_index: int
  end_sample_index_exclusive: int
  event_locators: tuple[EventLocator, ...]
  observability: WindowObservability

  def __post_init__(self) -> None:
    if not self.window_id.strip():
      raise ValueError("window_id must not be empty")
    if self.start_sample_index < 0 or self.end_sample_index_exclusive <= self.start_sample_index:
      raise ValueError("phase span bounds must be a non-empty half-open interval")


@dataclass(frozen=True, slots=True)
class BehaviorSegmentationSpans:
  """Segmentation metadata without copied behavior sample tuples."""

  route_id: str
  source_identity_sha256: str
  config_sha256: str
  spans: tuple[BehaviorPhaseSpan, ...]
  event_coverage: tuple[EventCoverage, ...]
  unassigned_sample_indices: Sequence[int]


@dataclass(frozen=True, slots=True)
class _UnassignedSampleIndices(Sequence[int]):
  """Compact complement of the non-overlapping assigned phase ranges."""

  ranges: tuple[tuple[int, int], ...]
  count: int

  @classmethod
  def from_assigned_ranges(
    cls,
    sample_count: int,
    assigned_ranges: tuple[tuple[int, int], ...],
  ) -> _UnassignedSampleIndices:
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for start, end in assigned_ranges:
      if start < cursor or end < start or end > sample_count:
        raise ValueError("assigned segmentation ranges are invalid")
      if cursor < start:
        ranges.append((cursor, start))
      cursor = end
    if cursor < sample_count:
      ranges.append((cursor, sample_count))
    return cls(tuple(ranges), sum(end - start for start, end in ranges))

  def __len__(self) -> int:
    return self.count

  def __iter__(self) -> Iterator[int]:
    for start, end in self.ranges:
      yield from range(start, end)

  def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
    if isinstance(index, slice):
      return tuple(self)[index]
    selected = index + self.count if index < 0 else index
    if not 0 <= selected < self.count:
      raise IndexError(index)
    for start, end in self.ranges:
      length = end - start
      if selected < length:
        return start + selected
      selected -= length
    raise AssertionError("unassigned sample index count is inconsistent")


@dataclass(frozen=True, slots=True)
class _Run:
  start: int
  end: int
  start_reason: BoundaryReason | None
  end_reason: BoundaryReason | None


@dataclass(frozen=True, slots=True)
class _Span:
  start: int
  end: int
  phase: ManeuverPhase
  run: _Run
  driver_censor_ns: int | None = None


_BOUNDARY_REASONS = tuple(BoundaryReason)
_BOUNDARY_REASON_TO_CODE = {reason: index for index, reason in enumerate(_BOUNDARY_REASONS)}
_NO_BOUNDARY_REASON = 0xff
_PHASE_CODES = tuple(ManeuverPhase)
_PHASE_TO_CODE = {phase: index for index, phase in enumerate(_PHASE_CODES)}
_SPAN_RECORD = struct.Struct("<QQBQQBBBQ")
_EPISODE_RECORD = struct.Struct("<QQ")
_SPAN_BLOCK_BYTES = 16 * 1024
_SPAN_RECORDS_PER_BLOCK = max(1, _SPAN_BLOCK_BYTES // _SPAN_RECORD.size)
_EPISODE_RECORDS_PER_BLOCK = max(1, _SPAN_BLOCK_BYTES // _EPISODE_RECORD.size)


class _SpanScratch:
  """Owned fixed-record store for canonical raw phase spans.

  Raw phases can alternate every sample while nearly all fail the committed
  minimum-duration rule. Keeping those transient descriptors as Python
  objects makes heap use route-length-dependent. This store retains the exact
  pre-filter population on disk because event coverage and completion depend
  on it, while exposing the same ordered random-access view to both eager and
  file-backed segmentation.
  """

  __slots__ = (
    "_cache",
    "_cache_start",
    "_count",
    "_episode_cache",
    "_episode_cache_start",
    "_episode_count",
    "_episode_file",
    "_file",
    "_finished",
    "_open_episode_run",
    "_open_episode_start",
  )

  def __init__(self) -> None:
    self._file = tempfile.TemporaryFile(prefix="blatv2-segmentation-spans-", buffering=0)
    self._episode_file = tempfile.TemporaryFile(prefix="blatv2-segmentation-episodes-", buffering=0)
    self._count = 0
    self._episode_count = 0
    self._cache_start = -1
    self._cache: tuple[_Span, ...] = ()
    self._episode_cache_start = -1
    self._episode_cache: tuple[tuple[int, int], ...] = ()
    self._open_episode_start: int | None = None
    self._open_episode_run: _Run | None = None
    self._finished = False

  def __enter__(self) -> _SpanScratch:
    return self

  def __exit__(self, *_: object) -> None:
    self._file.close()
    self._episode_file.close()

  def __len__(self) -> int:
    return self._count

  def __iter__(self) -> Iterator[_Span]:
    for index in range(self._count):
      yield self[index]

  def __getitem__(self, index: int) -> _Span:
    if type(index) is not int:
      raise TypeError("span scratch indices must be integers")
    if not self._finished:
      raise RuntimeError("temporary segmentation spans are not finalized")
    resolved = index + self._count if index < 0 else index
    if not 0 <= resolved < self._count:
      raise IndexError(index)
    block_start = (resolved // _SPAN_RECORDS_PER_BLOCK) * _SPAN_RECORDS_PER_BLOCK
    if block_start != self._cache_start:
      count = min(_SPAN_RECORDS_PER_BLOCK, self._count - block_start)
      encoded = os.pread(
        self._file.fileno(),
        count * _SPAN_RECORD.size,
        block_start * _SPAN_RECORD.size,
      )
      if len(encoded) != count * _SPAN_RECORD.size:
        raise RuntimeError("temporary segmentation span is truncated")
      self._cache = tuple(
        self._decode_span(encoded[offset:offset + _SPAN_RECORD.size])
        for offset in range(0, len(encoded), _SPAN_RECORD.size)
      )
      self._cache_start = block_start
    return self._cache[resolved - block_start]

  @staticmethod
  def _decode_span(encoded: bytes) -> _Span:
    (
      start,
      end,
      phase_code,
      run_start,
      run_end,
      start_reason_code,
      end_reason_code,
      driver_censor_present,
      driver_censor_ns,
    ) = _SPAN_RECORD.unpack(encoded)
    try:
      phase = _PHASE_CODES[phase_code]
      start_reason = (
        None
        if start_reason_code == _NO_BOUNDARY_REASON
        else _BOUNDARY_REASONS[start_reason_code]
      )
      end_reason = (
        None
        if end_reason_code == _NO_BOUNDARY_REASON
        else _BOUNDARY_REASONS[end_reason_code]
      )
    except IndexError as exc:
      raise RuntimeError("temporary segmentation span is invalid") from exc
    if driver_censor_present not in (0, 1):
      raise RuntimeError("temporary segmentation span is invalid")
    return _Span(
      start,
      end,
      phase,
      _Run(run_start, run_end, start_reason, end_reason),
      driver_censor_ns if driver_censor_present else None,
    )

  def append(self, span: _Span) -> None:
    if self._finished:
      raise RuntimeError("temporary segmentation spans are already finalized")
    if self._open_episode_start is not None and (
      span.phase is ManeuverPhase.STRAIGHT_QUASI_STEADY
      or span.run != self._open_episode_run
    ):
      self._append_episode(self._open_episode_start, self._count)
      self._open_episode_start = None
      self._open_episode_run = None
    if (
      span.phase is not ManeuverPhase.STRAIGHT_QUASI_STEADY
      and self._open_episode_start is None
    ):
      self._open_episode_start = self._count
      self._open_episode_run = span.run
    encoded = _SPAN_RECORD.pack(
      span.start,
      span.end,
      _PHASE_TO_CODE[span.phase],
      span.run.start,
      span.run.end,
      (
        _NO_BOUNDARY_REASON
        if span.run.start_reason is None
        else _BOUNDARY_REASON_TO_CODE[span.run.start_reason]
      ),
      (
        _NO_BOUNDARY_REASON
        if span.run.end_reason is None
        else _BOUNDARY_REASON_TO_CODE[span.run.end_reason]
      ),
      int(span.driver_censor_ns is not None),
      0 if span.driver_censor_ns is None else span.driver_censor_ns,
    )
    written = self._file.write(encoded)
    if written != len(encoded):
      raise RuntimeError("temporary segmentation span write made no progress")
    self._count += 1

  def _append_episode(self, start: int, end: int) -> None:
    encoded = _EPISODE_RECORD.pack(start, end)
    if self._episode_file.write(encoded) != len(encoded):
      raise RuntimeError("temporary segmentation episode write made no progress")
    self._episode_count += 1

  def finish(self) -> None:
    if self._finished:
      return
    if self._open_episode_start is not None:
      self._append_episode(self._open_episode_start, self._count)
      self._open_episode_start = None
      self._open_episode_run = None
    self._finished = True

  def _episode(self, index: int) -> tuple[int, int]:
    block_start = (index // _EPISODE_RECORDS_PER_BLOCK) * _EPISODE_RECORDS_PER_BLOCK
    if block_start != self._episode_cache_start:
      count = min(_EPISODE_RECORDS_PER_BLOCK, self._episode_count - block_start)
      encoded = os.pread(
        self._episode_file.fileno(),
        count * _EPISODE_RECORD.size,
        block_start * _EPISODE_RECORD.size,
      )
      if len(encoded) != count * _EPISODE_RECORD.size:
        raise RuntimeError("temporary segmentation episode is truncated")
      self._episode_cache = tuple(
        _EPISODE_RECORD.unpack(encoded[offset:offset + _EPISODE_RECORD.size])
        for offset in range(0, len(encoded), _EPISODE_RECORD.size)
      )
      self._episode_cache_start = block_start
    return self._episode_cache[index - block_start]

  def episode_bounds(self, anchor_index: int) -> tuple[int, int]:
    """Find the non-straight episode containing one known non-straight span."""
    low = 0
    high = self._episode_count
    while low < high:
      middle = (low + high) // 2
      start, end = self._episode(middle)
      if anchor_index < start:
        high = middle
      elif anchor_index >= end:
        low = middle + 1
      else:
        return start, end
    raise RuntimeError("temporary segmentation episode index is inconsistent")


def _validate_inputs(
  route_id: str,
  samples: Sequence[BehaviorSample],
  events: tuple[EventLocator, ...],
) -> None:
  if not route_id.strip():
    raise ValueError("route_id must not be empty")
  if not samples:
    raise ValueError("segmentation requires at least one sample")
  for index in range(1, len(samples)):
    left = samples[index - 1]
    right = samples[index]
    if right.mono_time_ns <= left.mono_time_ns or right.route_time_s <= left.route_time_s:
      raise ValueError("samples must be strictly ordered by mono and route time")
  keys = tuple((event.occurred_mono_time_ns, event.event_type, event.severity) for event in events)
  if keys != tuple(sorted(keys)):
    raise ValueError("event locators must be in canonical timestamp order")


def _valid_runs(samples: Sequence[BehaviorSample], config: SegmentationConfig) -> Iterator[_Run]:
  start: int | None = None
  start_reason: BoundaryReason | None = None
  end_reason: BoundaryReason | None = None
  for index, sample in enumerate(samples):
    valid = sample.physically_valid
    gap = index > 0 and sample.route_time_s - samples[index - 1].route_time_s > config.maximum_sample_gap_s
    if start is not None and (not valid or gap):
      end_reason = (
        BoundaryReason.PHASE_INCOMPLETE_AT_DATA_GAP
        if gap
        else BoundaryReason.PHASE_INCOMPLETE_AT_INVALID_INPUT
      )
      yield _Run(start, index, start_reason, end_reason)
      start = None
    if valid and start is None:
      if gap:
        start_reason = BoundaryReason.PHASE_ONSET_PRECEDES_AVAILABLE_EVIDENCE
      elif index > 0 and not samples[index - 1].physically_valid:
        start_reason = BoundaryReason.PHASE_ONSET_PRECEDES_AVAILABLE_EVIDENCE
      else:
        start_reason = None
      start = index
  if start is not None:
    yield _Run(start, len(samples), start_reason, BoundaryReason.PHASE_INCOMPLETE_AT_ROUTE_END)


def _reference_rate(samples: Sequence[BehaviorSample], run: _Run, index: int) -> float:
  if index > run.start:
    left = samples[index - 1]
    right = samples[index]
  elif index + 1 < run.end:
    left = samples[index]
    right = samples[index + 1]
  else:
    return 0.0
  return (right.anchored_curvature_1pm - left.anchored_curvature_1pm) / (right.route_time_s - left.route_time_s)


def _primitive_phase(sample: BehaviorSample, rate: float, config: SegmentationConfig) -> ManeuverPhase:
  curvature = sample.anchored_curvature_1pm
  magnitude = abs(curvature)
  if magnitude <= config.reference_zero_threshold_1pm:
    return ManeuverPhase.STRAIGHT_QUASI_STEADY
  directional_progress = math.copysign(1.0, curvature) * rate
  if abs(rate) <= config.quasi_steady_rate_threshold_1pm_s:
    return ManeuverPhase.HOLD
  if directional_progress > config.monotonic_progress_epsilon_1pm_s:
    return ManeuverPhase.TURN_IN
  if directional_progress < -config.monotonic_progress_epsilon_1pm_s:
    return ManeuverPhase.RELEASE_UNWIND
  return ManeuverPhase.HOLD


_LABEL_WRITE_CHUNK = 64 * 1024


def _read_label(labels: mmap.mmap, index: int) -> ManeuverPhase:
  encoded = labels[index]
  if encoded >= len(_PHASE_CODES):
    raise RuntimeError("temporary segmentation label is invalid")
  return _PHASE_CODES[encoded]


def _write_label_range(
  labels: mmap.mmap,
  start: int,
  end: int,
  phase: ManeuverPhase,
) -> None:
  encoded = bytes((_PHASE_TO_CODE[phase],))
  cursor = start
  while cursor < end:
    chunk_end = min(cursor + _LABEL_WRITE_CHUNK, end)
    labels[cursor:chunk_end] = encoded * (chunk_end - cursor)
    cursor = chunk_end


def _apply_direct_handoffs(
  samples: Sequence[BehaviorSample],
  run: _Run,
  labels: mmap.mmap,
  config: SegmentationConfig,
) -> None:
  previous_meaningful: int | None = None
  for right in range(run.start, run.end):
    right_curvature = samples[right].anchored_curvature_1pm
    if abs(right_curvature) < config.direct_handoff_min_peak_curvature_1pm:
      continue
    left = previous_meaningful
    previous_meaningful = right
    if left is None:
      continue
    left_curvature = samples[left].anchored_curvature_1pm
    if left_curvature * right_curvature >= 0.0:
      continue
    if samples[right].route_time_s - samples[left].route_time_s > config.direct_handoff_max_neutral_duration_s:
      continue
    start = left
    while start > run.start and _read_label(labels, start - 1) is ManeuverPhase.RELEASE_UNWIND:
      start -= 1
    end = right + 1
    while end < run.end and _read_label(labels, end) is ManeuverPhase.TURN_IN:
      end += 1
    _write_label_range(labels, start, end, ManeuverPhase.DIRECT_HANDOFF)


def _raw_spans(
  samples: Sequence[BehaviorSample],
  runs: Iterable[_Run],
  config: SegmentationConfig,
) -> Iterator[_Span]:
  with tempfile.TemporaryFile(prefix="blatv2-segmentation-") as labels:
    labels.truncate(len(samples))
    with mmap.mmap(labels.fileno(), len(samples), access=mmap.ACCESS_WRITE) as mapped:
      for run in runs:
        for index in range(run.start, run.end):
          _write_label_range(
            mapped,
            index,
            index + 1,
            _primitive_phase(samples[index], _reference_rate(samples, run, index), config),
          )
        _apply_direct_handoffs(samples, run, mapped, config)
        start = run.start
        start_phase = _read_label(mapped, start)
        for index in range(run.start + 1, run.end + 1):
          if index == run.end or _read_label(mapped, index) is not start_phase:
            yield _Span(start, index, start_phase, run)
            start = index
            if index < run.end:
              start_phase = _read_label(mapped, index)


def _phase_completed(left: _Span, next_span: _Span | None) -> bool:
  if left.phase is ManeuverPhase.STRAIGHT_QUASI_STEADY:
    return True
  if next_span is None or next_span.run != left.run:
    return False
  allowed = {
    ManeuverPhase.TURN_IN: (ManeuverPhase.HOLD, ManeuverPhase.RELEASE_UNWIND, ManeuverPhase.DIRECT_HANDOFF),
    ManeuverPhase.HOLD: (ManeuverPhase.RELEASE_UNWIND, ManeuverPhase.DIRECT_HANDOFF, ManeuverPhase.STRAIGHT_QUASI_STEADY),
    ManeuverPhase.RELEASE_UNWIND: (ManeuverPhase.STRAIGHT_QUASI_STEADY, ManeuverPhase.TURN_IN, ManeuverPhase.DIRECT_HANDOFF),
    ManeuverPhase.DIRECT_HANDOFF: (ManeuverPhase.HOLD, ManeuverPhase.RELEASE_UNWIND, ManeuverPhase.STRAIGHT_QUASI_STEADY),
  }
  return next_span.phase in allowed[left.phase]


def _censor_spans(
  samples: Sequence[BehaviorSample],
  spans: Iterable[_Span],
) -> Iterator[_Span]:
  """Censor from intervention onset through the whole active episode.

  Straight curvature is not a lifecycle boundary.  Neither an invalid input
  nor a recorded gap can prove that the driver released control.  Only a real
  lateral-inactive sample clears the censor; evidence may then resume on the
  next inactive-to-active transition.
  """
  censor_ns: int | None = None
  cursor = 0
  for span in spans:
    while cursor < span.start:
      sample = samples[cursor]
      if not sample.lateral_active:
        censor_ns = None
      elif sample.driver_intervention_onset and censor_ns is None:
        censor_ns = sample.mono_time_ns
      cursor += 1
    active_censor_ns: int | None = None
    while cursor < span.end:
      sample = samples[cursor]
      if not sample.lateral_active:
        censor_ns = None
      elif sample.driver_intervention_onset and censor_ns is None:
        censor_ns = sample.mono_time_ns
      if active_censor_ns is None and censor_ns is not None:
        active_censor_ns = censor_ns
      cursor += 1
    yield _Span(
      span.start,
      span.end,
      span.phase,
      span.run,
      active_censor_ns,
    )


def _qualifies(span: _Span, samples: Sequence[BehaviorSample], config: SegmentationConfig) -> bool:
  count = span.end - span.start
  duration = samples[span.end - 1].route_time_s - samples[span.start].route_time_s
  return count >= config.minimum_phase_samples and duration >= config.minimum_phase_duration_s


def _maneuver_class(span: _Span, samples: Sequence[BehaviorSample], config: SegmentationConfig) -> ManeuverClass:
  if span.phase is ManeuverPhase.STRAIGHT_QUASI_STEADY:
    return ManeuverClass.STRAIGHT
  if span.phase is ManeuverPhase.DIRECT_HANDOFF:
    return ManeuverClass.DIRECT_HANDOFF
  peak = max(abs(sample.anchored_curvature_1pm) for sample in samples[span.start:span.end])
  return ManeuverClass.TURN if peak >= config.turn_class_curvature_threshold_1pm else ManeuverClass.CURVE


def _metric_crossing_observed(
  span: _Span,
  samples: Sequence[BehaviorSample],
  config: SegmentationConfig,
) -> bool | None:
  """Report whether the committed timing crossing exists inside the phase."""
  peak = max(abs(samples[index].anchored_curvature_1pm) for index in range(span.start, span.end))
  def crosses(threshold: float, rising: bool) -> bool:
    previous = abs(samples[span.start].anchored_curvature_1pm)
    for index in range(span.start + 1, span.end):
      current = abs(samples[index].anchored_curvature_1pm)
      if (previous < threshold <= current) if rising else (previous >= threshold > current):
        return True
      previous = current
    return False
  if span.phase is ManeuverPhase.TURN_IN:
    threshold = peak * config.turn_in_crossing_fraction
    return crosses(threshold, True)
  if span.phase is ManeuverPhase.RELEASE_UNWIND:
    threshold = peak * config.release_onset_fraction
    return crosses(threshold, False)
  return None


def _event_coverage(
  samples: Sequence[BehaviorSample],
  spans: _SpanScratch,
  events: tuple[EventLocator, ...],
  config: SegmentationConfig,
) -> tuple[EventCoverage, ...]:
  coverages: list[EventCoverage] = []
  route_start = samples[0].mono_time_ns
  route_end = samples[-1].mono_time_ns
  extension_ns = round(config.maximum_phase_extension_s * 1e9)
  nonstraight_indices = array("Q")
  nonstraight_starts = array("Q")
  nonstraight_ends = array("Q")
  for span_index, span in enumerate(spans):
    if span.phase is ManeuverPhase.STRAIGHT_QUASI_STEADY:
      continue
    nonstraight_indices.append(span_index)
    nonstraight_starts.append(samples[span.start].mono_time_ns)
    nonstraight_ends.append(samples[span.end - 1].mono_time_ns)
  for event_index, event in enumerate(events):
    before_ns = round(event.analysis_window_before_s * 1e9)
    after_ns = round(event.analysis_window_after_s * 1e9)
    nominal_start = max(route_start, event.occurred_mono_time_ns - before_ns)
    nominal_end = min(route_end, event.occurred_mono_time_ns + after_ns)
    anchor_index: int | None = None
    anchor_key: tuple[int, int, int] | None = None
    insertion = bisect_right(nonstraight_starts, event.occurred_mono_time_ns)
    for candidate in (insertion - 1, insertion):
      if not 0 <= candidate < len(nonstraight_indices):
        continue
      index = nonstraight_indices[candidate]
      span_start_ns = nonstraight_starts[candidate]
      span_end_ns = nonstraight_ends[candidate]
      if span_end_ns < nominal_start or span_start_ns > nominal_end:
        continue
      key = (
        0
        if span_start_ns <= event.occurred_mono_time_ns <= span_end_ns
        else 1,
        abs(span_start_ns - event.occurred_mono_time_ns),
        index,
      )
      if anchor_key is None or key < anchor_key:
        anchor_index = index
        anchor_key = key
    if anchor_index is None:
      coverages.append(EventCoverage(
        locator=event,
        physical_start_mono_time_ns=None,
        physical_end_mono_time_ns=None,
        nominal_start_mono_time_ns=nominal_start,
        nominal_end_mono_time_ns=nominal_end,
        extended_beyond_nominal_end=False,
        onset_precedes_available_evidence=False,
        phase_incomplete_at_boundary=False,
        stop_reason=EventCoverageStop.NO_MANEUVER,
      ))
      continue
    episode_start, episode_end = spans.episode_bounds(anchor_index)
    physical_start = samples[spans[episode_start].start].mono_time_ns
    desired_end = samples[spans[episode_end - 1].end - 1].mono_time_ns
    maximum_end = nominal_end + extension_ns
    hard_end = min(route_end, maximum_end)
    stop_reason = EventCoverageStop.PHASE_COMPLETE
    if event_index + 1 < len(events):
      next_event = events[event_index + 1]
      next_start = max(route_start, next_event.occurred_mono_time_ns - round(next_event.analysis_window_before_s * 1e9))
      # Overlapping detection contexts do not move the current boundary before
      # its own committed occurrence; the next event still owns later context.
      next_boundary = max(event.occurred_mono_time_ns, next_start)
      if next_boundary < hard_end:
        hard_end = next_boundary
        stop_reason = EventCoverageStop.NEXT_EVENT_WINDOW
    if desired_end > hard_end:
      if stop_reason is EventCoverageStop.PHASE_COMPLETE:
        stop_reason = (
          EventCoverageStop.ROUTE_BOUNDARY
          if hard_end == route_end
          else EventCoverageStop.MAXIMUM_EXTENSION
        )
      physical_end = hard_end
      incomplete = True
    else:
      physical_end = desired_end
      incomplete = not _phase_completed(
        spans[episode_end - 1],
        spans[episode_end] if episode_end < len(spans) else None,
      )
      if incomplete:
        stop_reason = (
          EventCoverageStop.ROUTE_BOUNDARY
          if spans[episode_end - 1].run.end == len(samples)
          else EventCoverageStop.DATA_GAP_OR_INVALID
        )
    earliest_start = max(route_start, nominal_start - extension_ns)
    onset_precedes = physical_start < earliest_start or (
      spans[episode_start].run.start_reason is not None
      and spans[episode_start].start == spans[episode_start].run.start
    )
    physical_start = max(physical_start, earliest_start)
    coverages.append(EventCoverage(
      locator=event,
      physical_start_mono_time_ns=physical_start,
      physical_end_mono_time_ns=physical_end,
      nominal_start_mono_time_ns=nominal_start,
      nominal_end_mono_time_ns=nominal_end,
      extended_beyond_nominal_end=physical_end > nominal_end,
      onset_precedes_available_evidence=onset_precedes,
      phase_incomplete_at_boundary=incomplete,
      stop_reason=stop_reason,
    ))
  return tuple(coverages)


def _attach_events_to_descriptors(
  samples: Sequence[BehaviorSample],
  descriptors: list[BehaviorPhaseSpan],
  coverage: tuple[EventCoverage, ...],
  config: SegmentationConfig,
) -> tuple[BehaviorPhaseSpan, ...]:
  """Attach only interval overlaps, with work proportional to real output."""
  if not descriptors or not coverage:
    return tuple(descriptors)
  starts = array(
    "Q",
    (samples[item.start_sample_index].mono_time_ns for item in descriptors),
  )
  ends = array(
    "Q",
    (samples[item.end_sample_index_exclusive - 1].mono_time_ns for item in descriptors),
  )
  attachments: list[list[EventLocator]] = [[] for _ in descriptors]
  attachment_count = 0
  for item in coverage:
    if item.physical_start_mono_time_ns is None or item.physical_end_mono_time_ns is None:
      continue
    index = bisect_left(ends, item.physical_start_mono_time_ns)
    while index < len(descriptors) and starts[index] <= item.physical_end_mono_time_ns:
      attachments[index].append(item.locator)
      attachment_count += 1
      if attachment_count > config.maximum_event_phase_attachments:
        raise SegmentationResourceError(
          "route exceeds maximum event-to-phase attachment work",
        )
      index += 1
  return tuple(
    replace(descriptor, event_locators=tuple(events))
    for descriptor, events in zip(descriptors, attachments, strict=True)
  )


def _bounded_event_locators(
  event_locators: Iterable[EventLocator],
  maximum: int,
) -> tuple[EventLocator, ...]:
  values: list[EventLocator] = []
  for value in event_locators:
    if len(values) >= maximum:
      raise SegmentationResourceError("route exceeds maximum event-locator work")
    values.append(value)
  return tuple(values)


def segment_behavior_spans(
  route_id: str,
  source: BehaviorSourceIdentity,
  samples: Sequence[BehaviorSample],
  event_locators: Iterable[EventLocator],
  config: SegmentationConfig,
) -> BehaviorSegmentationSpans:
  """Segment a re-iterable sample view without copying its sample payload."""
  event_values = _bounded_event_locators(
    event_locators,
    config.maximum_event_locators,
  )
  _validate_inputs(route_id, samples, event_values)
  runs = _valid_runs(samples, config)
  raw_spans = _raw_spans(samples, runs, config)
  descriptors: list[BehaviorPhaseSpan] = []
  assigned_ranges: list[tuple[int, int]] = []
  with _SpanScratch() as censored_spans:
    for span in _censor_spans(samples, raw_spans):
      if len(censored_spans) >= config.maximum_raw_phase_spans:
        raise SegmentationResourceError("route exceeds maximum raw phase-span work")
      censored_spans.append(span)
    censored_spans.finish()
    coverage = _event_coverage(samples, censored_spans, event_values, config)
    for span_index, span in enumerate(censored_spans):
      if not _qualifies(span, samples, config):
        continue
      # The onset-containing phase retains post-contact samples as diagnostic
      # context; BehaviorWindow censors them.  Later phases in the same maneuver
      # are omitted through the remainder of the same lateral-active episode.
      # Only a real inactive-to-active boundary can begin independent evidence.
      if span.driver_censor_ns is not None and not any(
        samples[index].driver_intervention_onset for index in range(span.start, span.end)
      ):
        continue
      next_span = censored_spans[span_index + 1] if span_index + 1 < len(censored_spans) else None
      onset_observed = not (
        span.start == span.run.start
        and span.phase is not ManeuverPhase.STRAIGHT_QUASI_STEADY
        and (span.run.start == 0 or span.run.start_reason is not None)
      )
      completion_observed = _phase_completed(span, next_span)
      reasons: list[BoundaryReason] = []
      if not onset_observed:
        reasons.append(BoundaryReason.PHASE_ONSET_PRECEDES_AVAILABLE_EVIDENCE)
      if not completion_observed and span.phase is not ManeuverPhase.STRAIGHT_QUASI_STEADY:
        if span.run.end_reason is not None:
          reasons.append(span.run.end_reason)
      if span.driver_censor_ns is not None:
        reasons.append(BoundaryReason.DRIVER_INTERVENTION_CENSOR)
      if len(descriptors) >= config.maximum_phase_windows:
        raise SegmentationResourceError("route exceeds maximum retained phase-window work")
      window_id = f"{route_id}:{len(descriptors):06d}:{span.phase.value}:{samples[span.start].mono_time_ns}-{samples[span.end - 1].mono_time_ns}"
      observability = WindowObservability(
        onset_observed=onset_observed,
        completion_observed=completion_observed,
        metric_crossing_observed=_metric_crossing_observed(span, samples, config),
        reasons=tuple(dict.fromkeys(reasons)),
        driver_censor_mono_time_ns=span.driver_censor_ns,
      )
      descriptors.append(BehaviorPhaseSpan(
        window_id=window_id,
        maneuver_class=_maneuver_class(span, samples, config),
        phase=span.phase,
        start_sample_index=span.start,
        end_sample_index_exclusive=span.end,
        event_locators=(),
        observability=observability,
      ))
      assigned_ranges.append((span.start, span.end))
    retained_descriptors = _attach_events_to_descriptors(
      samples,
      descriptors,
      coverage,
      config,
    )
    unassigned = _UnassignedSampleIndices.from_assigned_ranges(
      len(samples),
      tuple(assigned_ranges),
    )
  return BehaviorSegmentationSpans(
    route_id=route_id,
    source_identity_sha256=source.sha256,
    config_sha256=config.sha256,
    spans=retained_descriptors,
    event_coverage=coverage,
    unassigned_sample_indices=unassigned,
  )


def segment_behavior_route(
  route_id: str,
  source: BehaviorSourceIdentity,
  samples: Iterable[BehaviorSample],
  event_locators: Iterable[EventLocator],
  config: SegmentationConfig,
) -> SegmentationResult:
  """Segment one route into canonical, non-overlapping physical phases."""
  sample_values = tuple(samples)
  segmented = segment_behavior_spans(route_id, source, sample_values, event_locators, config)
  windows = tuple(
    SegmentedBehaviorWindow(
      window=BehaviorWindow(
        route_id=route_id,
        window_id=span.window_id,
        source=source,
        maneuver_class=span.maneuver_class,
        phase=span.phase,
        samples=sample_values[span.start_sample_index:span.end_sample_index_exclusive],
        event_locators=span.event_locators,
      ),
      start_sample_index=span.start_sample_index,
      end_sample_index_exclusive=span.end_sample_index_exclusive,
      observability=span.observability,
    )
    for span in segmented.spans
  )
  return SegmentationResult(
    route_id=segmented.route_id,
    source_identity_sha256=segmented.source_identity_sha256,
    config_sha256=segmented.config_sha256,
    windows=windows,
    event_coverage=segmented.event_coverage,
    unassigned_sample_indices=tuple(segmented.unassigned_sample_indices),
  )

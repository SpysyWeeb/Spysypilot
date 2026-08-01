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

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import math
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


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
  """Every deterministic phase/coverage constant used by this module."""

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
    if not 0.0 < self.turn_in_crossing_fraction < 1.0:
      raise ValueError("turn_in_crossing_fraction must be in (0, 1)")
    if not 0.0 < self.release_onset_fraction < 1.0:
      raise ValueError("release_onset_fraction must be in (0, 1)")

  @classmethod
  def provisional_offline_gate(cls) -> SegmentationConfig:
    """Version-1 reproducibility constants, not controller feel dials.

    The 50% turn-in and 90% release fractions are the committed metric
    conventions.  Remaining values are conservative evidence-discrimination
    constants and must be revised by a version bump, never silently.
    """
    return cls(
      schema_version=1,
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
    )

  def to_dict(self) -> dict[str, Any]:
    return {
      "directHandoffMaxNeutralDurationS": self.direct_handoff_max_neutral_duration_s,
      "directHandoffMinPeakCurvature1pm": self.direct_handoff_min_peak_curvature_1pm,
      "maximumPhaseExtensionS": self.maximum_phase_extension_s,
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


def _validate_inputs(
  route_id: str,
  samples: tuple[BehaviorSample, ...],
  events: tuple[EventLocator, ...],
) -> None:
  if not route_id.strip():
    raise ValueError("route_id must not be empty")
  if not samples:
    raise ValueError("segmentation requires at least one sample")
  if any(
    right.mono_time_ns <= left.mono_time_ns or right.route_time_s <= left.route_time_s
    for left, right in zip(samples, samples[1:], strict=False)
  ):
    raise ValueError("samples must be strictly ordered by mono and route time")
  keys = tuple((event.occurred_mono_time_ns, event.event_type, event.severity) for event in events)
  if keys != tuple(sorted(keys)):
    raise ValueError("event locators must be in canonical timestamp order")


def _valid_runs(samples: tuple[BehaviorSample, ...], config: SegmentationConfig) -> tuple[_Run, ...]:
  runs: list[_Run] = []
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
      runs.append(_Run(start, index, start_reason, end_reason))
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
    runs.append(_Run(start, len(samples), start_reason, BoundaryReason.PHASE_INCOMPLETE_AT_ROUTE_END))
  return tuple(runs)


def _reference_rate(samples: tuple[BehaviorSample, ...], run: _Run, index: int) -> float:
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


def _apply_direct_handoffs(
  samples: tuple[BehaviorSample, ...],
  run: _Run,
  labels: list[ManeuverPhase],
  config: SegmentationConfig,
) -> None:
  meaningful = [
    index
    for index in range(run.start, run.end)
    if abs(samples[index].anchored_curvature_1pm) >= config.direct_handoff_min_peak_curvature_1pm
  ]
  for left, right in zip(meaningful, meaningful[1:], strict=False):
    left_curvature = samples[left].anchored_curvature_1pm
    right_curvature = samples[right].anchored_curvature_1pm
    if left_curvature * right_curvature >= 0.0:
      continue
    if samples[right].route_time_s - samples[left].route_time_s > config.direct_handoff_max_neutral_duration_s:
      continue
    start = left
    while start > run.start and labels[start - 1] is ManeuverPhase.RELEASE_UNWIND:
      start -= 1
    end = right + 1
    while end < run.end and labels[end] is ManeuverPhase.TURN_IN:
      end += 1
    for index in range(start, end):
      labels[index] = ManeuverPhase.DIRECT_HANDOFF


def _raw_spans(
  samples: tuple[BehaviorSample, ...],
  runs: tuple[_Run, ...],
  config: SegmentationConfig,
) -> tuple[_Span, ...]:
  spans: list[_Span] = []
  for run in runs:
    labels = [ManeuverPhase.STRAIGHT_QUASI_STEADY] * len(samples)
    for index in range(run.start, run.end):
      labels[index] = _primitive_phase(samples[index], _reference_rate(samples, run, index), config)
    _apply_direct_handoffs(samples, run, labels, config)
    start = run.start
    for index in range(run.start + 1, run.end + 1):
      if index == run.end or labels[index] is not labels[start]:
        spans.append(_Span(start, index, labels[start], run))
        start = index
  return tuple(spans)


def _phase_completed(left: _Span, next_span: _Span | None) -> bool:
  if left.phase is ManeuverPhase.STRAIGHT_QUASI_STEADY:
    return True
  if next_span is None or next_span.run is not left.run:
    return False
  allowed = {
    ManeuverPhase.TURN_IN: (ManeuverPhase.HOLD, ManeuverPhase.RELEASE_UNWIND, ManeuverPhase.DIRECT_HANDOFF),
    ManeuverPhase.HOLD: (ManeuverPhase.RELEASE_UNWIND, ManeuverPhase.DIRECT_HANDOFF, ManeuverPhase.STRAIGHT_QUASI_STEADY),
    ManeuverPhase.RELEASE_UNWIND: (ManeuverPhase.STRAIGHT_QUASI_STEADY, ManeuverPhase.TURN_IN, ManeuverPhase.DIRECT_HANDOFF),
    ManeuverPhase.DIRECT_HANDOFF: (ManeuverPhase.HOLD, ManeuverPhase.RELEASE_UNWIND, ManeuverPhase.STRAIGHT_QUASI_STEADY),
  }
  return next_span.phase in allowed[left.phase]


def _censor_spans(
  samples: tuple[BehaviorSample, ...],
  spans: tuple[_Span, ...],
) -> tuple[_Span, ...]:
  """Censor from intervention onset through the whole active episode.

  Straight curvature is not a lifecycle boundary.  Neither an invalid input
  nor a recorded gap can prove that the driver released control.  Only a real
  lateral-inactive sample clears the censor; evidence may then resume on the
  next inactive-to-active transition.
  """
  censor_by_sample: list[int | None] = []
  censor_ns: int | None = None
  for sample in samples:
    if not sample.lateral_active:
      censor_ns = None
    elif sample.driver_intervention_onset and censor_ns is None:
      censor_ns = sample.mono_time_ns
    censor_by_sample.append(censor_ns)

  result: list[_Span] = []
  for span in spans:
    active_censor_ns = next(
      (
        value
        for value in censor_by_sample[span.start:span.end]
        if value is not None
      ),
      None,
    )
    result.append(_Span(
      span.start,
      span.end,
      span.phase,
      span.run,
      active_censor_ns,
    ))
  return tuple(result)


def _qualifies(span: _Span, samples: tuple[BehaviorSample, ...], config: SegmentationConfig) -> bool:
  count = span.end - span.start
  duration = samples[span.end - 1].route_time_s - samples[span.start].route_time_s
  return count >= config.minimum_phase_samples and duration >= config.minimum_phase_duration_s


def _maneuver_class(span: _Span, samples: tuple[BehaviorSample, ...], config: SegmentationConfig) -> ManeuverClass:
  if span.phase is ManeuverPhase.STRAIGHT_QUASI_STEADY:
    return ManeuverClass.STRAIGHT
  if span.phase is ManeuverPhase.DIRECT_HANDOFF:
    return ManeuverClass.DIRECT_HANDOFF
  peak = max(abs(sample.anchored_curvature_1pm) for sample in samples[span.start:span.end])
  return ManeuverClass.TURN if peak >= config.turn_class_curvature_threshold_1pm else ManeuverClass.CURVE


def _metric_crossing_observed(
  span: _Span,
  samples: tuple[BehaviorSample, ...],
  config: SegmentationConfig,
) -> bool | None:
  """Report whether the committed timing crossing exists inside the phase."""
  magnitudes = tuple(abs(sample.anchored_curvature_1pm) for sample in samples[span.start:span.end])
  peak = max(magnitudes)
  if span.phase is ManeuverPhase.TURN_IN:
    threshold = peak * config.turn_in_crossing_fraction
    return any(left < threshold <= right for left, right in zip(magnitudes, magnitudes[1:], strict=False))
  if span.phase is ManeuverPhase.RELEASE_UNWIND:
    threshold = peak * config.release_onset_fraction
    return any(left >= threshold > right for left, right in zip(magnitudes, magnitudes[1:], strict=False))
  return None


def _episode_bounds(spans: tuple[_Span, ...], anchor_index: int) -> tuple[int, int]:
  start = anchor_index
  while start > 0 and spans[start - 1].run is spans[anchor_index].run and spans[start - 1].phase is not ManeuverPhase.STRAIGHT_QUASI_STEADY:
    start -= 1
  end = anchor_index + 1
  while end < len(spans) and spans[end].run is spans[anchor_index].run and spans[end].phase is not ManeuverPhase.STRAIGHT_QUASI_STEADY:
    end += 1
  return start, end


def _event_coverage(
  samples: tuple[BehaviorSample, ...],
  spans: tuple[_Span, ...],
  events: tuple[EventLocator, ...],
  config: SegmentationConfig,
) -> tuple[EventCoverage, ...]:
  coverages: list[EventCoverage] = []
  route_start = samples[0].mono_time_ns
  route_end = samples[-1].mono_time_ns
  extension_ns = round(config.maximum_phase_extension_s * 1e9)
  for event_index, event in enumerate(events):
    before_ns = round(event.analysis_window_before_s * 1e9)
    after_ns = round(event.analysis_window_after_s * 1e9)
    nominal_start = max(route_start, event.occurred_mono_time_ns - before_ns)
    nominal_end = min(route_end, event.occurred_mono_time_ns + after_ns)
    candidates = [
      (index, span)
      for index, span in enumerate(spans)
      if span.phase is not ManeuverPhase.STRAIGHT_QUASI_STEADY
      and samples[span.end - 1].mono_time_ns >= nominal_start
      and samples[span.start].mono_time_ns <= nominal_end
    ]
    if not candidates:
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
    anchor_index, _ = min(
      candidates,
      key=lambda item: (
        0 if samples[item[1].start].mono_time_ns <= event.occurred_mono_time_ns <= samples[item[1].end - 1].mono_time_ns else 1,
        abs(samples[item[1].start].mono_time_ns - event.occurred_mono_time_ns),
        item[0],
      ),
    )
    episode_start, episode_end = _episode_bounds(spans, anchor_index)
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


def segment_behavior_route(
  route_id: str,
  source: BehaviorSourceIdentity,
  samples: Iterable[BehaviorSample],
  event_locators: Iterable[EventLocator],
  config: SegmentationConfig,
) -> SegmentationResult:
  """Segment one route into canonical, non-overlapping physical phases."""
  sample_values = tuple(samples)
  event_values = tuple(event_locators)
  _validate_inputs(route_id, sample_values, event_values)
  runs = _valid_runs(sample_values, config)
  raw_spans = _raw_spans(sample_values, runs, config)
  censored_spans = _censor_spans(sample_values, raw_spans)
  coverage = _event_coverage(sample_values, censored_spans, event_values, config)
  windows: list[SegmentedBehaviorWindow] = []
  assigned: set[int] = set()
  for span_index, span in enumerate(censored_spans):
    if not _qualifies(span, sample_values, config):
      continue
    # The onset-containing phase retains post-contact samples as diagnostic
    # context; BehaviorWindow censors them.  Later phases in the same maneuver
    # are omitted through the remainder of the same lateral-active episode.
    # Only a real inactive-to-active boundary can begin independent evidence.
    if span.driver_censor_ns is not None and not any(
      sample.driver_intervention_onset for sample in sample_values[span.start:span.end]
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
    attached_events = tuple(
      coverage_item.locator
      for coverage_item in coverage
      if coverage_item.physical_start_mono_time_ns is not None
      and coverage_item.physical_end_mono_time_ns is not None
      and sample_values[span.end - 1].mono_time_ns >= coverage_item.physical_start_mono_time_ns
      and sample_values[span.start].mono_time_ns <= coverage_item.physical_end_mono_time_ns
    )
    window_id = f"{route_id}:{len(windows):06d}:{span.phase.value}:{sample_values[span.start].mono_time_ns}-{sample_values[span.end - 1].mono_time_ns}"
    window = BehaviorWindow(
      route_id=route_id,
      window_id=window_id,
      source=source,
      maneuver_class=_maneuver_class(span, sample_values, config),
      phase=span.phase,
      samples=sample_values[span.start:span.end],
      event_locators=attached_events,
    )
    windows.append(SegmentedBehaviorWindow(
      window=window,
      start_sample_index=span.start,
      end_sample_index_exclusive=span.end,
      observability=WindowObservability(
        onset_observed=onset_observed,
        completion_observed=completion_observed,
        metric_crossing_observed=_metric_crossing_observed(span, sample_values, config),
        reasons=tuple(dict.fromkeys(reasons)),
        driver_censor_mono_time_ns=span.driver_censor_ns,
      ),
    ))
    assigned.update(range(span.start, span.end))
  return SegmentationResult(
    route_id=route_id,
    source_identity_sha256=source.sha256,
    config_sha256=config.sha256,
    windows=tuple(windows),
    event_coverage=coverage,
    unassigned_sample_indices=tuple(index for index in range(len(sample_values)) if index not in assigned),
  )

"""Deterministic Smooth/Swift/Strong scoring for offroad behavior evidence.

The three contracts remain independent.  This module never combines them into
one weighted quality score.  Every threshold, noise discriminator, speed node,
and route-contribution cap is caller-supplied artifact data; there are no
hidden tuning constants.
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
  BehaviorWindow,
  ManeuverClass,
  ManeuverPhase,
  canonical_json,
)


class BehaviorContract(StrEnum):
  SMOOTH = "smooth"
  SWIFT = "swift"
  STRONG = "strong"


class BehaviorMetricName(StrEnum):
  RAW_TORQUE_RATE_RMS = "raw_torque_rate_rms"
  APPLIED_TORQUE_RATE_RMS = "applied_torque_rate_rms"
  RAW_WORST_BURST_RMS = "raw_worst_1s_torque_rate_rms"
  APPLIED_WORST_BURST_RMS = "applied_worst_1s_torque_rate_rms"
  RAW_CHATTER_REVERSALS_PER_S = "raw_chatter_reversals_per_s"
  APPLIED_CHATTER_REVERSALS_PER_S = "applied_chatter_reversals_per_s"
  RELEASE_OVERSHOOT_1PM = "release_overshoot_1pm"
  SIGNED_TURN_IN_LAG_S = "signed_delivered_turn_in_lag_s"
  SIGNED_RELEASE_LAG_S = "signed_delivered_release_lag_s"
  CORRECTION_LATENCY_S = "correction_latency_s"
  RACK_RATE_ERROR_RMS_DEG_S = "rack_rate_error_rms_deg_s"
  INTEGRATED_CURVATURE_ERROR = "integrated_abs_curvature_error_1pm_s"
  PEAK_CURVATURE_ERROR = "peak_abs_curvature_error_1pm"
  HOLD_BIAS_1PM = "signed_hold_bias_1pm"
  DELIVERED_FRACTION = "delivered_curvature_fraction"
  COMPLETION = "maneuver_completion_fraction"
  GROWING_ERROR_UNUSED_HEADROOM = "growing_error_unused_headroom_fraction"


class MetricDisposition(StrEnum):
  """Why a metric has, or does not have, a physical value."""

  DEFINED = "defined"
  NOT_APPLICABLE = "not_applicable"
  COVERAGE_EXCLUDED = "coverage_excluded"
  PHYSICAL_UNSCOREABLE = "physical_unscoreable"


class MetricReducer(StrEnum):
  POOLED_RMS = "pooled_rms"
  MAXIMUM = "maximum"
  MEDIAN = "median"
  WEIGHTED_MEAN = "weighted_mean"


_CONTRACT_BY_METRIC = {
  BehaviorMetricName.RAW_TORQUE_RATE_RMS: BehaviorContract.SMOOTH,
  BehaviorMetricName.APPLIED_TORQUE_RATE_RMS: BehaviorContract.SMOOTH,
  BehaviorMetricName.RAW_WORST_BURST_RMS: BehaviorContract.SMOOTH,
  BehaviorMetricName.APPLIED_WORST_BURST_RMS: BehaviorContract.SMOOTH,
  BehaviorMetricName.RAW_CHATTER_REVERSALS_PER_S: BehaviorContract.SMOOTH,
  BehaviorMetricName.APPLIED_CHATTER_REVERSALS_PER_S: BehaviorContract.SMOOTH,
  BehaviorMetricName.RELEASE_OVERSHOOT_1PM: BehaviorContract.SMOOTH,
  BehaviorMetricName.SIGNED_TURN_IN_LAG_S: BehaviorContract.SWIFT,
  BehaviorMetricName.SIGNED_RELEASE_LAG_S: BehaviorContract.SWIFT,
  BehaviorMetricName.CORRECTION_LATENCY_S: BehaviorContract.SWIFT,
  BehaviorMetricName.RACK_RATE_ERROR_RMS_DEG_S: BehaviorContract.SWIFT,
  BehaviorMetricName.INTEGRATED_CURVATURE_ERROR: BehaviorContract.STRONG,
  BehaviorMetricName.PEAK_CURVATURE_ERROR: BehaviorContract.STRONG,
  BehaviorMetricName.HOLD_BIAS_1PM: BehaviorContract.STRONG,
  BehaviorMetricName.DELIVERED_FRACTION: BehaviorContract.STRONG,
  BehaviorMetricName.COMPLETION: BehaviorContract.STRONG,
  BehaviorMetricName.GROWING_ERROR_UNUSED_HEADROOM: BehaviorContract.STRONG,
}

_REDUCER_BY_METRIC = {
  BehaviorMetricName.RAW_TORQUE_RATE_RMS: MetricReducer.POOLED_RMS,
  BehaviorMetricName.APPLIED_TORQUE_RATE_RMS: MetricReducer.POOLED_RMS,
  BehaviorMetricName.RAW_WORST_BURST_RMS: MetricReducer.MAXIMUM,
  BehaviorMetricName.APPLIED_WORST_BURST_RMS: MetricReducer.MAXIMUM,
  BehaviorMetricName.RAW_CHATTER_REVERSALS_PER_S: MetricReducer.WEIGHTED_MEAN,
  BehaviorMetricName.APPLIED_CHATTER_REVERSALS_PER_S: MetricReducer.WEIGHTED_MEAN,
  BehaviorMetricName.RELEASE_OVERSHOOT_1PM: MetricReducer.MAXIMUM,
  BehaviorMetricName.SIGNED_TURN_IN_LAG_S: MetricReducer.MEDIAN,
  BehaviorMetricName.SIGNED_RELEASE_LAG_S: MetricReducer.MEDIAN,
  BehaviorMetricName.CORRECTION_LATENCY_S: MetricReducer.MAXIMUM,
  BehaviorMetricName.RACK_RATE_ERROR_RMS_DEG_S: MetricReducer.POOLED_RMS,
  BehaviorMetricName.INTEGRATED_CURVATURE_ERROR: MetricReducer.WEIGHTED_MEAN,
  BehaviorMetricName.PEAK_CURVATURE_ERROR: MetricReducer.MAXIMUM,
  BehaviorMetricName.HOLD_BIAS_1PM: MetricReducer.WEIGHTED_MEAN,
  BehaviorMetricName.DELIVERED_FRACTION: MetricReducer.WEIGHTED_MEAN,
  BehaviorMetricName.COMPLETION: MetricReducer.WEIGHTED_MEAN,
  BehaviorMetricName.GROWING_ERROR_UNUSED_HEADROOM: MetricReducer.WEIGHTED_MEAN,
}


def behavior_metric_contract(name: str | BehaviorMetricName) -> BehaviorContract:
  """Return the versioned Smooth/Swift/Strong ownership for a metric."""
  return _CONTRACT_BY_METRIC[BehaviorMetricName(name)]


def behavior_metric_reducer(name: str | BehaviorMetricName) -> MetricReducer:
  """Return the versioned reducer; callers cannot substitute an average."""
  return _REDUCER_BY_METRIC[BehaviorMetricName(name)]


@dataclass(frozen=True, slots=True)
class BehaviorMetricConfig:
  """Versionable metric/segmentation inputs supplied by the harness."""

  burst_window_s: float
  chatter_torque_rate_threshold_per_s: float
  turn_in_crossing_fraction: float
  release_crossing_fraction: float
  correction_curvature_threshold_1pm: float
  unused_headroom_threshold: float
  growing_error_epsilon_1pm: float
  completion_delivered_fraction: float
  minimum_samples: int
  speed_nodes_mps: tuple[float, ...]
  maximum_route_windows_per_stratum: int

  def __post_init__(self) -> None:
    positive = (
      self.burst_window_s,
      self.chatter_torque_rate_threshold_per_s,
      self.correction_curvature_threshold_1pm,
      self.unused_headroom_threshold,
      self.growing_error_epsilon_1pm,
      self.completion_delivered_fraction,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in positive):
      raise ValueError("metric thresholds must be finite and positive")
    if not 0.0 < self.turn_in_crossing_fraction < 1.0:
      raise ValueError("turn-in crossing fraction must be in (0, 1)")
    if not 0.0 < self.release_crossing_fraction < 1.0:
      raise ValueError("release crossing fraction must be in (0, 1)")
    if self.minimum_samples < 2:
      raise ValueError("minimum_samples must be at least two")
    if self.maximum_route_windows_per_stratum <= 0:
      raise ValueError("route contribution cap must be positive")
    if not self.speed_nodes_mps:
      raise ValueError("at least one speed node is required")
    if any(
      not math.isfinite(node) or node < 0.0
      for node in self.speed_nodes_mps
    ) or any(
      right <= left
      for left, right in zip(
        self.speed_nodes_mps,
        self.speed_nodes_mps[1:],
        strict=False,
      )
    ):
      raise ValueError("speed nodes must be finite, non-negative, and increasing")

  def to_dict(self) -> dict[str, Any]:
    return {
      "burstWindowS": self.burst_window_s,
      "chatterTorqueRateThresholdPerS": self.chatter_torque_rate_threshold_per_s,
      "completionDeliveredFraction": self.completion_delivered_fraction,
      "correctionCurvatureThreshold1pm": self.correction_curvature_threshold_1pm,
      "growingErrorEpsilon1pm": self.growing_error_epsilon_1pm,
      "maximumRouteWindowsPerStratum": self.maximum_route_windows_per_stratum,
      "minimumSamples": self.minimum_samples,
      "releaseCrossingFraction": self.release_crossing_fraction,
      "speedNodesMps": list(self.speed_nodes_mps),
      "turnInCrossingFraction": self.turn_in_crossing_fraction,
      "unusedHeadroomThreshold": self.unused_headroom_threshold,
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())

  @property
  def sha256(self) -> str:
    return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class MetricValue:
  name: BehaviorMetricName
  contract: BehaviorContract
  value: float | None
  denominator: int
  exclusions: tuple[str, ...]
  disposition: MetricDisposition

  def __post_init__(self) -> None:
    if self.contract is not _CONTRACT_BY_METRIC[self.name]:
      raise ValueError("metric assigned to the wrong behavior contract")
    if self.denominator < 0:
      raise ValueError("metric denominator must be non-negative")
    if self.value is not None and not math.isfinite(self.value):
      raise ValueError("defined metric value must be finite")
    if self.disposition is MetricDisposition.DEFINED:
      if self.value is None or self.exclusions:
        raise ValueError("defined metrics require a value and no exclusions")
    elif self.value is not None or not self.exclusions:
      raise ValueError("non-defined metrics require reasons and no value")

  @property
  def defined(self) -> bool:
    return self.disposition is MetricDisposition.DEFINED

  def to_dict(self) -> dict[str, Any]:
    return {
      "contract": self.contract.value,
      "defined": self.defined,
      "denominator": self.denominator,
      "disposition": self.disposition.value,
      "exclusions": list(self.exclusions),
      "name": self.name.value,
      "value": self.value,
    }


@dataclass(frozen=True, slots=True)
class WindowMetricSet:
  route_id: str
  window_id: str
  source_identity_sha256: str
  maneuver_class: ManeuverClass
  phase: ManeuverPhase
  mean_speed_mps: float | None
  speed_node_support: tuple[tuple[float, float], ...]
  clean_sample_count: int
  intervention_mono_time_ns: int | None
  metrics: tuple[MetricValue, ...]

  def metric(self, name: BehaviorMetricName) -> MetricValue:
    return next(metric for metric in self.metrics if metric.name is name)


@dataclass(frozen=True, slots=True)
class StratumKey:
  speed_node_mps: float
  maneuver_class: ManeuverClass


@dataclass(frozen=True, slots=True)
class AggregateMetric:
  name: BehaviorMetricName
  contract: BehaviorContract
  value: float | None
  window_count: int
  route_count: int
  weighted_support: float
  exclusions: tuple[str, ...]
  disposition: MetricDisposition
  retained_window_ids: tuple[str, ...]
  coverage_excluded_window_ids: tuple[str, ...]
  physical_failure_window_ids: tuple[str, ...]
  not_applicable_window_ids: tuple[str, ...]
  route_ids: tuple[str, ...]
  route_values: tuple[tuple[str, float], ...]

  def __post_init__(self) -> None:
    value_ids = tuple(route_id for route_id, _ in self.route_values)
    if value_ids != tuple(sorted(set(value_ids))):
      raise ValueError("aggregate route values must be unique and sorted")
    if any(not math.isfinite(value) for _, value in self.route_values):
      raise ValueError("aggregate route values must be finite")
    if self.defined:
      if value_ids != self.route_ids:
        raise ValueError("defined aggregate route values must cover route IDs")
    elif self.route_values:
      raise ValueError("undefined aggregate cannot expose route values")

  @property
  def defined(self) -> bool:
    return self.disposition is MetricDisposition.DEFINED

  @property
  def coverage_identity_sha256(self) -> str:
    return hashlib.sha256(canonical_json({
      "coverageExcludedWindowIds": list(self.coverage_excluded_window_ids),
      "retainedWindowIds": list(self.retained_window_ids),
      "routeIds": list(self.route_ids),
    }).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class StratumScore:
  key: StratumKey
  metrics: tuple[AggregateMetric, ...]


@dataclass(frozen=True, slots=True)
class BehaviorScorecard:
  metric_config_sha256: str
  windows: tuple[WindowMetricSet, ...]
  strata: tuple[StratumScore, ...]
  balanced_metrics: tuple[AggregateMetric, ...]

  def to_dict(self) -> dict[str, Any]:
    return {
      "balancedMetrics": [_aggregate_to_dict(value) for value in self.balanced_metrics],
      "metricConfigSha256": self.metric_config_sha256,
      "strata": [
        {
          "maneuverClass": stratum.key.maneuver_class.value,
          "metrics": [_aggregate_to_dict(value) for value in stratum.metrics],
          "speedNodeMps": stratum.key.speed_node_mps,
        }
        for stratum in self.strata
      ],
      "windows": [
        {
          "cleanSampleCount": window.clean_sample_count,
          "interventionMonoTimeNs": window.intervention_mono_time_ns,
          "maneuverClass": window.maneuver_class.value,
          "meanSpeedMps": window.mean_speed_mps,
          "speedNodeSupport": [
            {"speedNodeMps": node, "weight": weight}
            for node, weight in window.speed_node_support
          ],
          "metrics": [metric.to_dict() for metric in window.metrics],
          "phase": window.phase.value,
          "routeId": window.route_id,
          "sourceIdentitySha256": window.source_identity_sha256,
          "windowId": window.window_id,
        }
        for window in self.windows
      ],
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())


def _aggregate_to_dict(value: AggregateMetric) -> dict[str, Any]:
  return {
    "contract": value.contract.value,
    "defined": value.defined,
    "disposition": value.disposition.value,
    "coverageExcludedWindowIds": list(value.coverage_excluded_window_ids),
    "exclusions": list(value.exclusions),
    "name": value.name.value,
    "routeCount": value.route_count,
    "routeIds": list(value.route_ids),
    "routeValues": [
      {"routeId": route_id, "value": route_value}
      for route_id, route_value in value.route_values
    ],
    "notApplicableWindowIds": list(value.not_applicable_window_ids),
    "physicalFailureWindowIds": list(value.physical_failure_window_ids),
    "retainedWindowIds": list(value.retained_window_ids),
    "value": value.value,
    "weightedSupport": value.weighted_support,
    "windowCount": value.window_count,
  }


def _metric(
  name: BehaviorMetricName,
  value: float | None,
  denominator: int,
  *exclusions: str,
  disposition: MetricDisposition | None = None,
) -> MetricValue:
  if disposition is None:
    disposition = (
      MetricDisposition.DEFINED
      if value is not None
      else _undefined_disposition(exclusions)
    )
  return MetricValue(
    name=name,
    contract=_CONTRACT_BY_METRIC[name],
    value=value,
    denominator=denominator,
    exclusions=tuple(exclusions),
    disposition=disposition,
  )


def _undefined_disposition(reasons: tuple[str, ...]) -> MetricDisposition:
  """Classify legacy reason strings at their physical boundary.

  This central registry keeps every metric path explicit while avoiding a
  second, subtly different classification in the policy layer.
  """
  if not reasons:
    raise ValueError("undefined metric lacks a disposition reason")
  reason = reasons[0]
  if reason.startswith("not_") or reason in {
    "no_reference_correction",
    "no_unconstrained_headroom_transitions",
  }:
    return MetricDisposition.NOT_APPLICABLE
  if reason.startswith("delivered_"):
    return MetricDisposition.PHYSICAL_UNSCOREABLE
  return MetricDisposition.COVERAGE_EXCLUDED


def _rates(samples: tuple[BehaviorSample, ...], field: str) -> tuple[tuple[float, float], ...]:
  result: list[tuple[float, float]] = []
  for left, right in zip(samples, samples[1:], strict=False):
    dt_s = (right.mono_time_ns - left.mono_time_ns) * 1e-9
    if dt_s <= 0.0:
      continue
    result.append((right.route_time_s, (getattr(right, field) - getattr(left, field)) / dt_s))
  return tuple(result)


def _rms(values: Iterable[float]) -> float | None:
  sequence = tuple(values)
  if not sequence:
    return None
  return math.sqrt(math.fsum(value * value for value in sequence) / len(sequence))


def _rate_rms(
  name: BehaviorMetricName,
  samples: tuple[BehaviorSample, ...],
  field: str,
) -> MetricValue:
  rates = _rates(samples, field)
  value = _rms(rate for _, rate in rates)
  return (
    _metric(name, value, len(rates))
    if value is not None
    else _metric(name, None, 0, "insufficient_adjacent_samples")
  )


def _worst_burst(
  name: BehaviorMetricName,
  samples: tuple[BehaviorSample, ...],
  field: str,
  burst_window_s: float,
) -> MetricValue:
  rates = _rates(samples, field)
  if not rates or samples[-1].route_time_s - samples[0].route_time_s < burst_window_s:
    return _metric(name, None, 0, "window_shorter_than_burst_interval")
  worst: float | None = None
  valid_windows = 0
  left = 0
  squared_sum = 0.0
  earliest_sample_time_s = samples[0].route_time_s
  for end, (end_time_s, value) in enumerate(rates):
    squared_sum += value * value
    start_time_s = end_time_s - burst_window_s
    while left <= end and rates[left][0] <= start_time_s:
      squared_sum -= rates[left][1] * rates[left][1]
      left += 1
    count = end - left + 1
    if count <= 0 or end_time_s - earliest_sample_time_s < burst_window_s:
      continue
    rms = math.sqrt(max(0.0, squared_sum) / count)
    worst = rms if worst is None else max(worst, rms)
    valid_windows += 1
  return (
    _metric(name, worst, valid_windows)
    if worst is not None
    else _metric(name, None, 0, "no_complete_burst_interval")
  )


def _chatter(
  name: BehaviorMetricName,
  samples: tuple[BehaviorSample, ...],
  field: str,
  threshold: float,
) -> MetricValue:
  rates = _rates(samples, field)
  significant_signs = [
    1 if rate > 0.0 else -1
    for _, rate in rates
    if abs(rate) >= threshold
  ]
  duration_s = samples[-1].route_time_s - samples[0].route_time_s if len(samples) >= 2 else 0.0
  if duration_s <= 0.0:
    return _metric(name, None, 0, "insufficient_duration")
  reversals = sum(
    right != left
    for left, right in zip(significant_signs, significant_signs[1:], strict=False)
  )
  return _metric(
    name,
    reversals / duration_s,
    len(significant_signs),
  )


def _direction_and_peak(samples: tuple[BehaviorSample, ...]) -> tuple[float, float, int] | None:
  if not samples:
    return None
  peak_index = max(range(len(samples)), key=lambda index: abs(samples[index].anchored_curvature_1pm))
  peak = samples[peak_index].anchored_curvature_1pm
  if peak == 0.0:
    return None
  return (1.0 if peak > 0.0 else -1.0), abs(peak), peak_index


def _release_overshoot(samples: tuple[BehaviorSample, ...], phase: ManeuverPhase) -> MetricValue:
  if phase is not ManeuverPhase.RELEASE_UNWIND:
    return _metric(BehaviorMetricName.RELEASE_OVERSHOOT_1PM, None, 0, "not_release_phase")
  peak = _direction_and_peak(samples)
  if peak is None:
    return _metric(BehaviorMetricName.RELEASE_OVERSHOOT_1PM, None, 0, "zero_reference_direction")
  direction, _, _ = peak
  overshoot = max(
    0.0,
    max(direction * (sample.measured_curvature_1pm - sample.anchored_curvature_1pm) for sample in samples),
  )
  return _metric(BehaviorMetricName.RELEASE_OVERSHOOT_1PM, overshoot, len(samples))


def _first_crossing_time(
  samples: tuple[BehaviorSample, ...],
  field: str,
  direction: float,
  threshold: float,
) -> float | None:
  for sample in samples:
    if direction * getattr(sample, field) >= threshold:
      return sample.route_time_s
  return None


def _turn_in_lag(
  samples: tuple[BehaviorSample, ...],
  phase: ManeuverPhase,
  fraction: float,
) -> MetricValue:
  if phase not in (ManeuverPhase.TURN_IN, ManeuverPhase.DIRECT_HANDOFF):
    return _metric(BehaviorMetricName.SIGNED_TURN_IN_LAG_S, None, 0, "not_turn_in_phase")
  peak = _direction_and_peak(samples)
  if peak is None:
    return _metric(BehaviorMetricName.SIGNED_TURN_IN_LAG_S, None, 0, "zero_reference_direction")
  direction, magnitude, _ = peak
  threshold = magnitude * fraction
  desired_time = _first_crossing_time(samples, "anchored_curvature_1pm", direction, threshold)
  delivered_time = _first_crossing_time(samples, "measured_curvature_1pm", direction, threshold)
  if desired_time is None:
    return _metric(BehaviorMetricName.SIGNED_TURN_IN_LAG_S, None, len(samples), "reference_crossing_unobservable")
  if delivered_time is None:
    return _metric(BehaviorMetricName.SIGNED_TURN_IN_LAG_S, None, len(samples), "delivered_crossing_unobservable")
  return _metric(
    BehaviorMetricName.SIGNED_TURN_IN_LAG_S,
    delivered_time - desired_time,
    len(samples),
  )


def _release_crossing_time(
  samples: tuple[BehaviorSample, ...],
  field: str,
  direction: float,
  threshold: float,
  start_index: int,
) -> float | None:
  for sample in samples[start_index:]:
    if direction * getattr(sample, field) <= threshold:
      return sample.route_time_s
  return None


def _release_lag(
  samples: tuple[BehaviorSample, ...],
  phase: ManeuverPhase,
  fraction: float,
) -> MetricValue:
  if phase is not ManeuverPhase.RELEASE_UNWIND:
    return _metric(BehaviorMetricName.SIGNED_RELEASE_LAG_S, None, 0, "not_release_phase")
  peak = _direction_and_peak(samples)
  if peak is None:
    return _metric(BehaviorMetricName.SIGNED_RELEASE_LAG_S, None, 0, "zero_reference_direction")
  direction, magnitude, peak_index = peak
  threshold = magnitude * fraction
  desired_time = _release_crossing_time(
    samples,
    "anchored_curvature_1pm",
    direction,
    threshold,
    peak_index,
  )
  delivered_peak_index = max(
    range(len(samples)),
    key=lambda index: direction * samples[index].measured_curvature_1pm,
  )
  if direction * samples[delivered_peak_index].measured_curvature_1pm < threshold:
    return _metric(
      BehaviorMetricName.SIGNED_RELEASE_LAG_S,
      None,
      len(samples),
      "delivered_peak_below_release_threshold",
    )
  delivered_time = _release_crossing_time(
    samples,
    "measured_curvature_1pm",
    direction,
    threshold,
    delivered_peak_index,
  )
  if desired_time is None:
    return _metric(BehaviorMetricName.SIGNED_RELEASE_LAG_S, None, len(samples), "reference_release_incomplete")
  if delivered_time is None:
    return _metric(BehaviorMetricName.SIGNED_RELEASE_LAG_S, None, len(samples), "delivered_release_incomplete")
  return _metric(
    BehaviorMetricName.SIGNED_RELEASE_LAG_S,
    delivered_time - desired_time,
    len(samples),
  )


def _correction_latency(
  samples: tuple[BehaviorSample, ...],
  threshold: float,
) -> MetricValue:
  if not samples:
    return _metric(BehaviorMetricName.CORRECTION_LATENCY_S, None, 0, "no_clean_samples")
  desired_start = samples[0].anchored_curvature_1pm
  measured_start = samples[0].measured_curvature_1pm
  desired_time: float | None = None
  direction = 0.0
  for sample in samples:
    delta = sample.anchored_curvature_1pm - desired_start
    if abs(delta) >= threshold:
      desired_time = sample.route_time_s
      direction = 1.0 if delta > 0.0 else -1.0
      break
  if desired_time is None:
    return _metric(BehaviorMetricName.CORRECTION_LATENCY_S, None, len(samples), "no_reference_correction")
  for sample in samples:
    if sample.route_time_s < desired_time:
      continue
    if direction * (sample.measured_curvature_1pm - measured_start) >= threshold:
      return _metric(
        BehaviorMetricName.CORRECTION_LATENCY_S,
        sample.route_time_s - desired_time,
        len(samples),
      )
  return _metric(BehaviorMetricName.CORRECTION_LATENCY_S, None, len(samples), "delivered_correction_unobservable")


def _rate_error(samples: tuple[BehaviorSample, ...]) -> MetricValue:
  value = _rms(
    sample.measured_rack_rate_deg_s - sample.desired_rack_rate_deg_s
    for sample in samples
  )
  return (
    _metric(BehaviorMetricName.RACK_RATE_ERROR_RMS_DEG_S, value, len(samples))
    if value is not None
    else _metric(BehaviorMetricName.RACK_RATE_ERROR_RMS_DEG_S, None, 0, "no_clean_samples")
  )


def _integrated_error(samples: tuple[BehaviorSample, ...]) -> MetricValue:
  if len(samples) < 2:
    return _metric(BehaviorMetricName.INTEGRATED_CURVATURE_ERROR, None, 0, "insufficient_adjacent_samples")
  integral = 0.0
  transitions = 0
  for left, right in zip(samples, samples[1:], strict=False):
    dt_s = (right.mono_time_ns - left.mono_time_ns) * 1e-9
    if dt_s <= 0.0:
      continue
    left_error = abs(left.measured_curvature_1pm - left.anchored_curvature_1pm)
    right_error = abs(right.measured_curvature_1pm - right.anchored_curvature_1pm)
    integral += 0.5 * (left_error + right_error) * dt_s
    transitions += 1
  return (
    _metric(BehaviorMetricName.INTEGRATED_CURVATURE_ERROR, integral, transitions)
    if transitions
    else _metric(BehaviorMetricName.INTEGRATED_CURVATURE_ERROR, None, 0, "no_valid_transitions")
  )


def _peak_error(samples: tuple[BehaviorSample, ...]) -> MetricValue:
  if not samples:
    return _metric(BehaviorMetricName.PEAK_CURVATURE_ERROR, None, 0, "no_clean_samples")
  return _metric(
    BehaviorMetricName.PEAK_CURVATURE_ERROR,
    max(abs(sample.measured_curvature_1pm - sample.anchored_curvature_1pm) for sample in samples),
    len(samples),
  )


def _hold_bias(samples: tuple[BehaviorSample, ...], phase: ManeuverPhase) -> MetricValue:
  if phase is not ManeuverPhase.HOLD:
    return _metric(BehaviorMetricName.HOLD_BIAS_1PM, None, 0, "not_hold_phase")
  directions = [
    1.0 if sample.anchored_curvature_1pm > 0.0 else -1.0
    for sample in samples
    if sample.anchored_curvature_1pm != 0.0
  ]
  if len(directions) != len(samples) or not directions:
    return _metric(BehaviorMetricName.HOLD_BIAS_1PM, None, len(samples), "zero_or_signless_hold_reference")
  bias = math.fsum(
    direction * (sample.measured_curvature_1pm - sample.anchored_curvature_1pm)
    for direction, sample in zip(directions, samples, strict=True)
  ) / len(samples)
  return _metric(BehaviorMetricName.HOLD_BIAS_1PM, bias, len(samples))


def _delivered_and_completion(
  samples: tuple[BehaviorSample, ...],
  completion_threshold: float,
) -> tuple[MetricValue, MetricValue]:
  peak = _direction_and_peak(samples)
  if peak is None:
    undefined = "zero_reference_direction"
    return (
      _metric(BehaviorMetricName.DELIVERED_FRACTION, None, 0, undefined),
      _metric(BehaviorMetricName.COMPLETION, None, 0, undefined),
    )
  direction, magnitude, _ = peak
  delivered = max(direction * sample.measured_curvature_1pm for sample in samples)
  fraction = delivered / magnitude
  return (
    _metric(BehaviorMetricName.DELIVERED_FRACTION, fraction, len(samples)),
    _metric(
      BehaviorMetricName.COMPLETION,
      1.0 if fraction >= completion_threshold else 0.0,
      len(samples),
    ),
  )


def _unused_headroom(
  samples: tuple[BehaviorSample, ...],
  headroom_threshold: float,
  error_epsilon: float,
) -> MetricValue:
  eligible = 0
  growing = 0
  for left, right in zip(samples, samples[1:], strict=False):
    if right.actuator_constrained or right.torque_headroom < headroom_threshold:
      continue
    eligible += 1
    left_error = abs(left.anchored_curvature_1pm - left.measured_curvature_1pm)
    right_error = abs(right.anchored_curvature_1pm - right.measured_curvature_1pm)
    if right_error > left_error + error_epsilon:
      growing += 1
  if not eligible:
    return _metric(
      BehaviorMetricName.GROWING_ERROR_UNUSED_HEADROOM,
      None,
      0,
      "no_unconstrained_headroom_transitions",
    )
  return _metric(
    BehaviorMetricName.GROWING_ERROR_UNUSED_HEADROOM,
    growing / eligible,
    eligible,
  )


def score_window(window: BehaviorWindow, config: BehaviorMetricConfig) -> WindowMetricSet:
  samples = window.clean_pre_intervention_samples
  mean_speed = (
    math.fsum(sample.speed_mps for sample in samples) / len(samples)
    if samples
    else None
  )
  support_by_node = dict.fromkeys(config.speed_nodes_mps, 0.0)
  for sample in samples:
    for node, weight in speed_node_weights(sample.speed_mps, config.speed_nodes_mps):
      support_by_node[node] += weight
  speed_node_support = tuple(
    (node, weight / len(samples))
    for node, weight in support_by_node.items()
    if samples and weight > 0.0
  )
  if len(samples) < config.minimum_samples:
    metrics = tuple(
      _metric(name, None, len(samples), "insufficient_clean_pre_intervention_samples")
      for name in BehaviorMetricName
    )
  else:
    delivered, completion = _delivered_and_completion(
      samples,
      config.completion_delivered_fraction,
    )
    metrics = (
      _rate_rms(BehaviorMetricName.RAW_TORQUE_RATE_RMS, samples, "raw_requested_torque"),
      _rate_rms(BehaviorMetricName.APPLIED_TORQUE_RATE_RMS, samples, "envelope_applied_torque"),
      _worst_burst(BehaviorMetricName.RAW_WORST_BURST_RMS, samples, "raw_requested_torque", config.burst_window_s),
      _worst_burst(BehaviorMetricName.APPLIED_WORST_BURST_RMS, samples, "envelope_applied_torque", config.burst_window_s),
      _chatter(
        BehaviorMetricName.RAW_CHATTER_REVERSALS_PER_S,
        samples,
        "raw_requested_torque",
        config.chatter_torque_rate_threshold_per_s,
      ),
      _chatter(
        BehaviorMetricName.APPLIED_CHATTER_REVERSALS_PER_S,
        samples,
        "envelope_applied_torque",
        config.chatter_torque_rate_threshold_per_s,
      ),
      _release_overshoot(samples, window.phase),
      _turn_in_lag(samples, window.phase, config.turn_in_crossing_fraction),
      _release_lag(samples, window.phase, config.release_crossing_fraction),
      _correction_latency(samples, config.correction_curvature_threshold_1pm),
      _rate_error(samples),
      _integrated_error(samples),
      _peak_error(samples),
      _hold_bias(samples, window.phase),
      delivered,
      completion,
      _unused_headroom(
        samples,
        config.unused_headroom_threshold,
        config.growing_error_epsilon_1pm,
      ),
    )
  return WindowMetricSet(
    route_id=window.route_id,
    window_id=window.window_id,
    source_identity_sha256=window.source.sha256,
    maneuver_class=window.maneuver_class,
    phase=window.phase,
    mean_speed_mps=mean_speed,
    speed_node_support=speed_node_support,
    clean_sample_count=len(samples),
    intervention_mono_time_ns=window.intervention_mono_time_ns,
    metrics=metrics,
  )


def speed_node_weights(
  speed_mps: float,
  nodes: tuple[float, ...],
) -> tuple[tuple[float, float], ...]:
  """Continuous adjacent-node weights with flat endpoint extrapolation."""
  if speed_mps <= nodes[0]:
    return ((nodes[0], 1.0),)
  if speed_mps >= nodes[-1]:
    return ((nodes[-1], 1.0),)
  for left, right in zip(nodes, nodes[1:], strict=False):
    if speed_mps == left:
      return ((left, 1.0),)
    if speed_mps < right:
      right_weight = (speed_mps - left) / (right - left)
      if right_weight == 0.0:
        return ((left, 1.0),)
      if right_weight == 1.0:
        return ((right, 1.0),)
      return ((left, 1.0 - right_weight), (right, right_weight))
  raise AssertionError("increasing speed nodes must bracket finite speed")


def _aggregate_stratum(
  weighted_windows: tuple[tuple[WindowMetricSet, float], ...],
  cap: int,
) -> tuple[AggregateMetric, ...]:
  """Aggregate without erasing inapplicability, coverage, or physical failure."""
  aggregates: list[AggregateMetric] = []
  for name in BehaviorMetricName:
    route_values: list[float] = []
    used_windows = 0
    weighted_support = 0.0
    retained_ids: list[str] = []
    coverage_ids: list[str] = []
    physical_ids: list[str] = []
    not_applicable_ids: list[str] = []
    reasons: list[str] = []
    used_route_ids: list[str] = []
    for route_id in sorted({window.route_id for window, _ in weighted_windows}):
      # Selection is value-independent: a metric cannot hide its own worst
      # windows by changing their value, and route naming has no priority.
      route_windows = sorted(
        (
          (window, weight)
          for window, weight in weighted_windows
          if window.route_id == route_id
        ),
        key=lambda item: (
          hashlib.sha256(
            f"{item[0].route_id}\0{item[0].window_id}".encode(),
          ).digest(),
          item[0].window_id,
        ),
      )
      selected = tuple(route_windows[:cap])
      defined: list[tuple[float, float, int]] = []
      route_physical = False
      for window, weight in selected:
        metric = window.metric(name)
        identity = f"{window.route_id}/{window.window_id}"
        if metric.disposition is MetricDisposition.DEFINED:
          assert metric.value is not None
          if weight > 0.0:
            defined.append((metric.value, weight, metric.denominator))
            retained_ids.append(identity)
        elif metric.disposition is MetricDisposition.PHYSICAL_UNSCOREABLE:
          physical_ids.append(identity)
          reasons.extend(f"{identity}:{reason}" for reason in metric.exclusions)
          route_physical = True
        elif metric.disposition is MetricDisposition.COVERAGE_EXCLUDED:
          coverage_ids.append(identity)
          reasons.extend(f"{identity}:{reason}" for reason in metric.exclusions)
        else:
          not_applicable_ids.append(identity)
      if route_physical:
        continue
      if defined:
        route_values.append(_reduce_values(name, tuple(defined)))
        used_route_ids.append(route_id)
        used_windows += len(defined)
        weighted_support += math.fsum(weight for _, weight, _ in defined)
    disposition = _aggregate_disposition(
      route_values,
      physical_ids,
      coverage_ids,
    )
    value = (
      _reduce_route_values(name, tuple(route_values))
      if disposition is MetricDisposition.DEFINED
      else None
    )
    if disposition is not MetricDisposition.DEFINED and not reasons:
      reasons.append(
        "metric_not_applicable"
        if disposition is MetricDisposition.NOT_APPLICABLE
        else "no_eligible_metric_coverage"
      )
    aggregates.append(AggregateMetric(
      name=name,
      contract=_CONTRACT_BY_METRIC[name],
      value=value,
      window_count=used_windows,
      route_count=len(used_route_ids),
      weighted_support=weighted_support,
      exclusions=tuple(sorted(set(reasons))),
      disposition=disposition,
      retained_window_ids=tuple(sorted(set(retained_ids))),
      coverage_excluded_window_ids=tuple(sorted(set(coverage_ids))),
      physical_failure_window_ids=tuple(sorted(set(physical_ids))),
      not_applicable_window_ids=tuple(sorted(set(not_applicable_ids))),
      route_ids=tuple(used_route_ids),
      route_values=(
        tuple(zip(used_route_ids, route_values, strict=True))
        if disposition is MetricDisposition.DEFINED
        else ()
      ),
    ))
  return tuple(aggregates)


def _weighted_median(values: tuple[tuple[float, float], ...]) -> float:
  ordered = tuple(sorted(values, key=lambda item: item[0]))
  total = math.fsum(weight for _, weight in ordered)
  midpoint = total * 0.5
  cumulative = 0.0
  for value, weight in ordered:
    cumulative += weight
    if cumulative >= midpoint:
      return value
  raise AssertionError("positive weighted values must have a median")


def _reduce_values(
  name: BehaviorMetricName,
  values: tuple[tuple[float, float, int], ...],
) -> float:
  reducer = _REDUCER_BY_METRIC[name]
  if reducer is MetricReducer.MAXIMUM:
    return max(value for value, _, _ in values)
  if reducer is MetricReducer.MEDIAN:
    return _weighted_median(tuple((value, weight) for value, weight, _ in values))
  support = tuple(
    (value, weight * max(denominator, 1))
    for value, weight, denominator in values
  )
  denominator = math.fsum(weight for _, weight in support)
  if reducer is MetricReducer.POOLED_RMS:
    return math.sqrt(
      math.fsum(value * value * weight for value, weight in support) / denominator,
    )
  return math.fsum(value * weight for value, weight in support) / denominator


def _reduce_route_values(
  name: BehaviorMetricName,
  values: tuple[float, ...],
) -> float:
  reducer = _REDUCER_BY_METRIC[name]
  if reducer is MetricReducer.MAXIMUM:
    return max(values)
  if reducer is MetricReducer.MEDIAN:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return (
      ordered[middle]
      if len(ordered) % 2
      else 0.5 * (ordered[middle - 1] + ordered[middle])
    )
  if reducer is MetricReducer.POOLED_RMS:
    return math.sqrt(math.fsum(value * value for value in values) / len(values))
  return math.fsum(values) / len(values)


def _aggregate_disposition(
  defined_values: list[float],
  physical_ids: list[str],
  coverage_ids: list[str],
) -> MetricDisposition:
  if physical_ids:
    return MetricDisposition.PHYSICAL_UNSCOREABLE
  if defined_values:
    return MetricDisposition.DEFINED
  if coverage_ids:
    return MetricDisposition.COVERAGE_EXCLUDED
  return MetricDisposition.NOT_APPLICABLE


def score_behavior(
  windows: Iterable[BehaviorWindow],
  config: BehaviorMetricConfig,
) -> BehaviorScorecard:
  """Score windows with equal route and equal speed/class-stratum weight."""
  scored = tuple(
    sorted(
      (score_window(window, config) for window in windows),
      key=lambda value: (value.route_id, value.window_id),
    )
  )
  grouped: dict[StratumKey, list[tuple[WindowMetricSet, float]]] = {}
  for window in scored:
    for speed_node_mps, weight in window.speed_node_support:
      key = StratumKey(
        speed_node_mps=speed_node_mps,
        maneuver_class=window.maneuver_class,
      )
      grouped.setdefault(key, []).append((window, weight))
  strata = tuple(
    StratumScore(
      key=key,
      metrics=_aggregate_stratum(
        tuple(grouped[key]),
        config.maximum_route_windows_per_stratum,
      ),
    )
    for key in sorted(grouped, key=lambda value: (value.speed_node_mps, value.maneuver_class.value))
  )
  balanced: list[AggregateMetric] = []
  for name in BehaviorMetricName:
    values = [
      next(metric for metric in stratum.metrics if metric.name is name)
      for stratum in strata
    ]
    defined = [metric for metric in values if metric.defined]
    physical_ids = tuple(sorted({
      identity
      for metric in values
      for identity in metric.physical_failure_window_ids
    }))
    coverage_ids = tuple(sorted({
      identity
      for metric in values
      for identity in metric.coverage_excluded_window_ids
    }))
    retained_ids = tuple(sorted({
      identity
      for metric in values
      for identity in metric.retained_window_ids
    }))
    not_applicable_ids = tuple(sorted({
      identity
      for metric in values
      for identity in metric.not_applicable_window_ids
    }))
    if physical_ids:
      disposition = MetricDisposition.PHYSICAL_UNSCOREABLE
    elif defined:
      disposition = MetricDisposition.DEFINED
    elif coverage_ids:
      disposition = MetricDisposition.COVERAGE_EXCLUDED
    else:
      disposition = MetricDisposition.NOT_APPLICABLE
    route_ids = tuple(sorted({
      route_id
      for metric in defined
      for route_id in metric.route_ids
    }))
    route_values: list[tuple[str, float]] = []
    if disposition is MetricDisposition.DEFINED:
      for route_id in route_ids:
        stratum_values = tuple(
          route_value
          for metric in defined
          for candidate_route_id, route_value in metric.route_values
          if candidate_route_id == route_id
        )
        if stratum_values:
          route_values.append((
            route_id,
            _reduce_route_values(name, stratum_values),
          ))
    balanced.append(AggregateMetric(
      name=name,
      contract=_CONTRACT_BY_METRIC[name],
      value=(
        _reduce_route_values(
          name,
          tuple(metric.value for metric in defined if metric.value is not None),
        )
        if disposition is MetricDisposition.DEFINED
        else None
      ),
      window_count=sum(metric.window_count for metric in defined),
      route_count=len({route_id for metric in defined for route_id in metric.route_ids}),
      weighted_support=math.fsum(metric.weighted_support for metric in defined),
      exclusions=tuple(
        f"{stratum.key.speed_node_mps}:{stratum.key.maneuver_class.value}:{reason}"
        for stratum, metric in zip(strata, values, strict=True)
        for reason in metric.exclusions
      ),
      disposition=disposition,
      retained_window_ids=retained_ids,
      coverage_excluded_window_ids=coverage_ids,
      physical_failure_window_ids=physical_ids,
      not_applicable_window_ids=not_applicable_ids,
      route_ids=route_ids,
      route_values=tuple(route_values),
    ))
  return BehaviorScorecard(
    metric_config_sha256=config.sha256,
    windows=scored,
    strata=strata,
    balanced_metrics=tuple(balanced),
  )

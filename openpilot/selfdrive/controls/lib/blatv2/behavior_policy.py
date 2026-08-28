"""Bounded, deterministic offroad search for BLaTv2 response policy.

Only two physically meaningful controller dials are eligible here:
speed-independent closed-loop natural frequency and damping ratio.  This
module cannot alter the model reference or its timing, vehicle calibration,
friction, observer, torque envelope, safety limits, or any live state.

Training chooses at most one frozen winner.  Held-out validation receives only
that winner and may accept or reject it; it has no fallback-selection API.
Smooth, Swift, and Strong remain independent contracts throughout.
Every eligible stratum must match or beat exact stock and the accepted
incumbent; balanced metrics remain reporting and ranking diagnostics.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from enum import StrEnum
import hashlib
import json
import math
import sys
from typing import TYPE_CHECKING, Any

from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import canonical_json
from openpilot.selfdrive.controls.lib.blatv2.behavior_metrics import (
  BehaviorContract,
  BehaviorMetricName,
  BehaviorScorecard,
  MetricDisposition,
  behavior_metric_contract,
)

if TYPE_CHECKING:
  from openpilot.selfdrive.controls.lib.blatv2.policy import ControllerPolicy


class MetricPreference(StrEnum):
  LOWER_IS_BETTER = "lower_is_better"
  HIGHER_IS_BETTER = "higher_is_better"


PAIRED_ROUTE_UNCERTAINTY_METHOD = "observed_whole_route_envelope_v1"


@dataclass(frozen=True, slots=True)
class BehaviorPolicy:
  """The complete behavior-learning parameter surface, initially two dials."""

  natural_frequency_per_s: float
  damping_ratio: float

  def __post_init__(self) -> None:
    if (
      not math.isfinite(self.natural_frequency_per_s)
      or self.natural_frequency_per_s <= 0.0
      or not math.isfinite(self.damping_ratio)
      or self.damping_ratio <= 0.0
    ):
      raise ValueError("behavior policy values must be finite and positive")

  def to_dict(self) -> dict[str, float]:
    return {
      "dampingRatio": self.damping_ratio,
      "naturalFrequencyPerS": self.natural_frequency_per_s,
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())

  @property
  def sha256(self) -> str:
    return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

  @classmethod
  def from_controller_policy(cls, policy: ControllerPolicy) -> BehaviorPolicy:
    """Extract only tracking semantics from the versioned live artifact."""
    return cls(
      natural_frequency_per_s=policy.natural_frequency_per_s,
      damping_ratio=policy.damping_ratio,
    )

  def into_controller_policy(self, template: ControllerPolicy) -> ControllerPolicy:
    """Replace tracking dials while preserving observer and artifact metadata.

    ``dataclasses.replace`` deliberately leaves every non-behavior field on
    the template untouched.  The behavior learner therefore neither copies
    nor owns disturbance-observer semantics.
    """
    return replace(
      template,
      natural_frequency_per_s=self.natural_frequency_per_s,
      damping_ratio=self.damping_ratio,
    )


@dataclass(frozen=True, slots=True)
class PolicyGridSpec:
  """Explicit log-space offsets and physical bounds around an incumbent."""

  incumbent: BehaviorPolicy
  natural_frequency_log_offsets: tuple[float, ...]
  damping_ratio_log_offsets: tuple[float, ...]
  minimum_natural_frequency_per_s: float
  maximum_natural_frequency_per_s: float
  minimum_damping_ratio: float
  maximum_damping_ratio: float

  def __post_init__(self) -> None:
    for name, offsets in (
      ("natural_frequency_log_offsets", self.natural_frequency_log_offsets),
      ("damping_ratio_log_offsets", self.damping_ratio_log_offsets),
    ):
      if not offsets or 0.0 not in offsets:
        raise ValueError(f"{name} must be non-empty and include zero")
      if any(not math.isfinite(value) for value in offsets) or any(
        right <= left
        for left, right in zip(offsets, offsets[1:], strict=False)
      ):
        raise ValueError(f"{name} must be finite and strictly increasing")
    bounds = (
      self.minimum_natural_frequency_per_s,
      self.maximum_natural_frequency_per_s,
      self.minimum_damping_ratio,
      self.maximum_damping_ratio,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in bounds):
      raise ValueError("policy bounds must be finite and positive")
    if self.minimum_natural_frequency_per_s >= self.maximum_natural_frequency_per_s:
      raise ValueError("natural-frequency bounds are inverted")
    if self.minimum_damping_ratio >= self.maximum_damping_ratio:
      raise ValueError("damping-ratio bounds are inverted")
    if not (
      self.minimum_natural_frequency_per_s
      <= self.incumbent.natural_frequency_per_s
      <= self.maximum_natural_frequency_per_s
      and self.minimum_damping_ratio
      <= self.incumbent.damping_ratio
      <= self.maximum_damping_ratio
    ):
      raise ValueError("incumbent must lie inside policy bounds")


@dataclass(frozen=True, slots=True)
class PolicyCandidate:
  canonical_index: int
  policy: BehaviorPolicy
  squared_log_displacement: float


_GRID_DECIMAL_CONTEXT = Context(prec=50, rounding=ROUND_HALF_EVEN)


def _deterministic_exp_scale(base: float, offset: float) -> float:
  """Deterministic decimal exp before one correctly rounded binary64 cast."""
  with localcontext(_GRID_DECIMAL_CONTEXT):
    value = Decimal.from_float(base) * Decimal.from_float(offset).exp()
  return float(value)


def build_candidate_grid(spec: PolicyGridSpec) -> tuple[PolicyCandidate, ...]:
  candidates: list[PolicyCandidate] = []
  seen: set[BehaviorPolicy] = set()
  canonical_index = 0
  for frequency_offset in spec.natural_frequency_log_offsets:
    frequency = _deterministic_exp_scale(
      spec.incumbent.natural_frequency_per_s,
      frequency_offset,
    )
    for damping_offset in spec.damping_ratio_log_offsets:
      damping = _deterministic_exp_scale(
        spec.incumbent.damping_ratio,
        damping_offset,
      )
      if not (
        spec.minimum_natural_frequency_per_s <= frequency <= spec.maximum_natural_frequency_per_s
        and spec.minimum_damping_ratio <= damping <= spec.maximum_damping_ratio
      ):
        canonical_index += 1
        continue
      policy = BehaviorPolicy(
        natural_frequency_per_s=frequency,
        damping_ratio=damping,
      )
      if policy in seen:
        raise ValueError("candidate grid produced a duplicate policy")
      seen.add(policy)
      candidates.append(PolicyCandidate(
        canonical_index=canonical_index,
        policy=policy,
        squared_log_displacement=(
          frequency_offset * frequency_offset
          + damping_offset * damping_offset
        ),
      ))
      canonical_index += 1
  if not candidates or spec.incumbent not in seen:
    raise ValueError("bounded candidate grid excluded the incumbent")
  return tuple(candidates)


@dataclass(frozen=True, slots=True)
class PolicyStratumMetric:
  stratum: str
  value: float | None
  disposition: MetricDisposition
  exclusions: tuple[str, ...]
  route_count: int
  window_count: int
  weighted_support: float
  coverage_identity_sha256: str
  physical_failure_window_ids: tuple[str, ...]
  coverage_excluded_window_ids: tuple[str, ...]
  not_applicable_window_ids: tuple[str, ...]
  route_values: tuple[tuple[str, float], ...]

  def __post_init__(self) -> None:
    if not self.stratum.strip():
      raise ValueError("policy stratum metric needs a stratum")
    if type(self.disposition) is not MetricDisposition:
      raise ValueError("policy stratum metric disposition is not registered")
    if self.value is not None and not math.isfinite(self.value):
      raise ValueError("policy stratum metric value must be finite")
    if self.route_count < 0 or self.window_count < 0:
      raise ValueError("policy stratum coverage counts must be non-negative")
    if not math.isfinite(self.weighted_support) or self.weighted_support < 0.0:
      raise ValueError("policy stratum support must be finite and non-negative")
    if len(self.coverage_identity_sha256) != 64 or any(
      character not in "0123456789abcdef"
      for character in self.coverage_identity_sha256
    ):
      raise ValueError("policy stratum coverage identity must be lowercase SHA-256")
    for label, window_ids in (
      ("physical failures", self.physical_failure_window_ids),
      ("coverage exclusions", self.coverage_excluded_window_ids),
      ("not-applicable windows", self.not_applicable_window_ids),
    ):
      if tuple(sorted(set(window_ids))) != window_ids:
        raise ValueError(f"policy stratum {label} must be unique and sorted")
    route_ids = tuple(route_id for route_id, _ in self.route_values)
    if route_ids != tuple(sorted(set(route_ids))):
      raise ValueError("policy stratum route values must be unique and sorted")
    if any(not math.isfinite(value) for _, value in self.route_values):
      raise ValueError("policy stratum route values must be finite")
    evidence_disposition = (
      MetricDisposition.PHYSICAL_UNSCOREABLE
      if self.physical_failure_window_ids
      else MetricDisposition.DEFINED
      if self.value is not None
      else MetricDisposition.COVERAGE_EXCLUDED
      if self.coverage_excluded_window_ids
      else MetricDisposition.NOT_APPLICABLE
      if self.not_applicable_window_ids
      else None
    )
    if self.disposition is not evidence_disposition:
      raise ValueError("policy stratum disposition disagrees with window evidence")
    if self.defined:
      if len(route_ids) != self.route_count:
        raise ValueError("defined policy stratum metric has inconsistent evidence")
    elif self.route_values or not self.exclusions:
      raise ValueError("undefined policy stratum metric has inconsistent evidence")

  @property
  def defined(self) -> bool:
    return self.disposition is MetricDisposition.DEFINED

  @property
  def route_values_by_id(self) -> dict[str, float]:
    return dict(self.route_values)

  def to_dict(self) -> dict[str, Any]:
    return {
      "coverageIdentitySha256": self.coverage_identity_sha256,
      "coverageExcludedWindowIds": list(self.coverage_excluded_window_ids),
      "defined": self.defined,
      "disposition": self.disposition.value,
      "exclusions": list(self.exclusions),
      "notApplicableWindowIds": list(self.not_applicable_window_ids),
      "physicalFailureWindowIds": list(self.physical_failure_window_ids),
      "routeCount": self.route_count,
      "routeValues": [
        {"routeId": route_id, "value": value}
        for route_id, value in self.route_values
      ],
      "stratum": self.stratum,
      "value": self.value,
      "weightedSupport": self.weighted_support,
      "windowCount": self.window_count,
    }

  @classmethod
  def from_dict(cls, payload: object) -> PolicyStratumMetric:
    keys = frozenset((
      "coverageIdentitySha256",
      "coverageExcludedWindowIds",
      "defined",
      "disposition",
      "exclusions",
      "notApplicableWindowIds",
      "physicalFailureWindowIds",
      "routeCount",
      "routeValues",
      "stratum",
      "value",
      "weightedSupport",
      "windowCount",
    ))
    if type(payload) is not dict or frozenset(payload) != keys:
      raise ValueError("policy stratum metric keys do not match schema")
    if (
      type(payload["stratum"]) is not str
      or type(payload["defined"]) is not bool
      or type(payload["disposition"]) is not str
    ):
      raise ValueError("policy stratum identity and defined flag have invalid types")
    try:
      disposition = MetricDisposition(payload["disposition"])
    except ValueError as exc:
      raise ValueError("policy stratum metric disposition is not registered") from exc
    for key in (
      "coverageExcludedWindowIds",
      "exclusions",
      "notApplicableWindowIds",
      "physicalFailureWindowIds",
    ):
      if type(payload[key]) is not list or any(type(value) is not str for value in payload[key]):
        raise ValueError(f"policy stratum {key} must be a text array")
    if type(payload["routeCount"]) is not int or type(payload["windowCount"]) is not int:
      raise ValueError("policy stratum coverage counts must be integer")
    if type(payload["weightedSupport"]) not in (int, float):
      raise ValueError("policy stratum weighted support must be numeric")
    if payload["value"] is not None and type(payload["value"]) not in (int, float):
      raise ValueError("policy stratum value must be numeric or null")
    raw_route_values = payload["routeValues"]
    if type(raw_route_values) is not list:
      raise ValueError("policy stratum route values must be an array")
    route_values: list[tuple[str, float]] = []
    for raw in raw_route_values:
      if type(raw) is not dict or frozenset(raw) != {"routeId", "value"}:
        raise ValueError("policy stratum route value keys do not match schema")
      if type(raw["routeId"]) is not str or type(raw["value"]) not in (int, float):
        raise ValueError("policy stratum route value has invalid types")
      route_values.append((raw["routeId"], float(raw["value"])))
    metric = cls(
      stratum=payload["stratum"],
      value=None if payload["value"] is None else float(payload["value"]),
      disposition=disposition,
      exclusions=tuple(payload["exclusions"]),
      route_count=payload["routeCount"],
      window_count=payload["windowCount"],
      weighted_support=float(payload["weightedSupport"]),
      coverage_identity_sha256=payload["coverageIdentitySha256"],
      physical_failure_window_ids=tuple(payload["physicalFailureWindowIds"]),
      coverage_excluded_window_ids=tuple(payload["coverageExcludedWindowIds"]),
      not_applicable_window_ids=tuple(payload["notApplicableWindowIds"]),
      route_values=tuple(route_values),
    )
    if payload["defined"] != metric.defined:
      raise ValueError("policy stratum defined flag disagrees with value")
    return metric


@dataclass(frozen=True, slots=True)
class PolicyMetric:
  name: str
  value: float | None
  denominator: int
  exclusions: tuple[str, ...]
  route_count: int
  window_count: int
  weighted_support: float
  coverage_identity_sha256: str
  strata: tuple[str, ...]
  stratum_metrics: tuple[PolicyStratumMetric, ...]
  physical_failure_window_ids: tuple[str, ...]
  route_values: tuple[tuple[str, float], ...]

  def __post_init__(self) -> None:
    try:
      BehaviorMetricName(self.name)
    except ValueError as exc:
      raise ValueError("policy metric name is not registered") from exc
    if self.value is not None and not math.isfinite(self.value):
      raise ValueError("policy metric value must be finite")
    if self.denominator < 0:
      raise ValueError("policy metric denominator must be non-negative")
    if self.route_count < 0 or self.window_count < 0:
      raise ValueError("policy metric coverage counts must be non-negative")
    if not math.isfinite(self.weighted_support) or self.weighted_support < 0.0:
      raise ValueError("policy metric weighted support must be finite and non-negative")
    if len(self.coverage_identity_sha256) != 64 or any(
      character not in "0123456789abcdef"
      for character in self.coverage_identity_sha256
    ):
      raise ValueError("policy metric coverage identity must be lowercase SHA-256")
    if tuple(sorted(set(self.strata))) != self.strata:
      raise ValueError("policy metric strata must be unique and sorted")
    stratum_names = tuple(metric.stratum for metric in self.stratum_metrics)
    if stratum_names != tuple(sorted(set(stratum_names))) or stratum_names != self.strata:
      raise ValueError("policy metric stratum evidence must exactly cover strata")
    if tuple(sorted(set(self.physical_failure_window_ids))) != self.physical_failure_window_ids:
      raise ValueError("physical failure window IDs must be unique and sorted")
    route_value_ids = tuple(route_id for route_id, _ in self.route_values)
    if route_value_ids != tuple(sorted(set(route_value_ids))):
      raise ValueError("policy metric route values must be unique and sorted")
    if any(not math.isfinite(value) for _, value in self.route_values):
      raise ValueError("policy metric route values must be finite")
    if self.defined:
      if len(route_value_ids) != self.route_count:
        raise ValueError("defined policy metric route values must match route count")
    elif self.route_values:
      raise ValueError("undefined policy metric cannot expose route values")
    if self.value is None and not self.exclusions:
      raise ValueError("undefined policy metrics need an exclusion reason")

  @property
  def defined(self) -> bool:
    return self.value is not None

  @property
  def route_ids(self) -> tuple[str, ...]:
    return tuple(route_id for route_id, _ in self.route_values)

  @property
  def route_values_by_id(self) -> dict[str, float]:
    return dict(self.route_values)

  def to_dict(self) -> dict[str, Any]:
    return {
      "defined": self.defined,
      "denominator": self.denominator,
      "exclusions": list(self.exclusions),
      "name": self.name,
      "routeCount": self.route_count,
      "windowCount": self.window_count,
      "weightedSupport": self.weighted_support,
      "coverageIdentitySha256": self.coverage_identity_sha256,
      "strata": list(self.strata),
      "stratumMetrics": [metric.to_dict() for metric in self.stratum_metrics],
      "physicalFailureWindowIds": list(self.physical_failure_window_ids),
      "routeValues": [
        {"routeId": route_id, "value": value}
        for route_id, value in self.route_values
      ],
      "value": self.value,
    }

  @classmethod
  def from_dict(cls, payload: object) -> PolicyMetric:
    keys = frozenset((
      "coverageIdentitySha256",
      "defined",
      "denominator",
      "exclusions",
      "name",
      "physicalFailureWindowIds",
      "routeCount",
      "routeValues",
      "strata",
      "stratumMetrics",
      "value",
      "weightedSupport",
      "windowCount",
    ))
    if type(payload) is not dict or frozenset(payload) != keys:
      raise ValueError("policy metric keys do not match schema")
    for key in ("exclusions", "physicalFailureWindowIds", "strata"):
      if type(payload[key]) is not list or any(
        type(value) is not str for value in payload[key]
      ):
        raise ValueError(f"policy metric {key} must be a text array")
    route_values_payload = payload["routeValues"]
    if type(route_values_payload) is not list:
      raise ValueError("policy metric routeValues must be an array")
    route_values: list[tuple[str, float]] = []
    for value in route_values_payload:
      if type(value) is not dict or frozenset(value) != {"routeId", "value"}:
        raise ValueError("policy metric route value keys do not match schema")
      if type(value["routeId"]) is not str or type(value["value"]) not in (int, float):
        raise ValueError("policy metric route value has invalid types")
      route_values.append((value["routeId"], float(value["value"])))
    if type(payload["stratumMetrics"]) is not list:
      raise ValueError("policy metric stratumMetrics must be an array")
    for key in ("denominator", "routeCount", "windowCount"):
      if type(payload[key]) is not int:
        raise ValueError(f"policy metric {key} must be integer")
    if type(payload["weightedSupport"]) not in (int, float):
      raise ValueError("policy metric weightedSupport must be numeric")
    if payload["value"] is not None and type(payload["value"]) not in (int, float):
      raise ValueError("policy metric value must be numeric or null")
    if type(payload["defined"]) is not bool:
      raise ValueError("policy metric defined must be boolean")
    metric = cls(
      name=payload["name"],
      value=None if payload["value"] is None else float(payload["value"]),
      denominator=payload["denominator"],
      exclusions=tuple(payload["exclusions"]),
      route_count=payload["routeCount"],
      window_count=payload["windowCount"],
      weighted_support=float(payload["weightedSupport"]),
      coverage_identity_sha256=payload["coverageIdentitySha256"],
      strata=tuple(payload["strata"]),
      stratum_metrics=tuple(
        PolicyStratumMetric.from_dict(value)
        for value in payload["stratumMetrics"]
      ),
      physical_failure_window_ids=tuple(payload["physicalFailureWindowIds"]),
      route_values=tuple(route_values),
    )
    if payload["defined"] != metric.defined:
      raise ValueError("policy metric defined flag disagrees with value")
    return metric

  @classmethod
  def from_scorecard(
    cls,
    scorecard: BehaviorScorecard,
    name: BehaviorMetricName,
  ) -> PolicyMetric:
    aggregate = next(metric for metric in scorecard.balanced_metrics if metric.name is name)
    stratum_metrics = tuple(sorted(
      (
        PolicyStratumMetric(
          stratum=f"{stratum.key.speed_node_mps:g}:{stratum.key.maneuver_class.value}",
          value=value.value,
          disposition=value.disposition,
          exclusions=value.exclusions,
          route_count=value.route_count,
          window_count=value.window_count,
          weighted_support=value.weighted_support,
          coverage_identity_sha256=value.coverage_identity_sha256,
          physical_failure_window_ids=value.physical_failure_window_ids,
          coverage_excluded_window_ids=value.coverage_excluded_window_ids,
          not_applicable_window_ids=value.not_applicable_window_ids,
          route_values=value.route_values,
        )
        for stratum in scorecard.strata
        for value in (next(metric for metric in stratum.metrics if metric.name is name),)
      ),
      key=lambda metric: metric.stratum,
    ))
    strata = tuple(metric.stratum for metric in stratum_metrics)
    return cls(
      name=name.value,
      value=aggregate.value,
      denominator=aggregate.window_count,
      exclusions=aggregate.exclusions,
      route_count=aggregate.route_count,
      window_count=aggregate.window_count,
      weighted_support=aggregate.weighted_support,
      coverage_identity_sha256=aggregate.coverage_identity_sha256,
      strata=strata,
      stratum_metrics=stratum_metrics,
      physical_failure_window_ids=aggregate.physical_failure_window_ids,
      route_values=aggregate.route_values,
    )


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
  """Replay result for one exact artifact over one route partition."""

  artifact_identity: str
  policy: BehaviorPolicy | None
  route_ids: tuple[str, ...]
  metrics: tuple[PolicyMetric, ...]

  def __post_init__(self) -> None:
    if not self.artifact_identity.strip():
      raise ValueError("artifact_identity must not be empty")
    if not self.route_ids or tuple(sorted(set(self.route_ids))) != self.route_ids:
      raise ValueError("route_ids must be non-empty, unique, and sorted")
    names = tuple(metric.name for metric in self.metrics)
    if not names or len(set(names)) != len(names):
      raise ValueError("policy evaluation metrics must be non-empty and unique")

  def metric(self, name: str) -> PolicyMetric:
    try:
      return next(metric for metric in self.metrics if metric.name == name)
    except StopIteration as exc:
      raise ValueError(f"evaluation lacks metric {name}") from exc

  def to_dict(self) -> dict[str, Any]:
    return {
      "artifactIdentity": self.artifact_identity,
      "metrics": [metric.to_dict() for metric in sorted(self.metrics, key=lambda metric: metric.name)],
      "policy": None if self.policy is None else self.policy.to_dict(),
      "routeIds": list(self.route_ids),
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())

  @property
  def sha256(self) -> str:
    return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

  @classmethod
  def from_dict(cls, payload: object) -> PolicyEvaluation:
    keys = frozenset(("artifactIdentity", "metrics", "policy", "routeIds"))
    if type(payload) is not dict or frozenset(payload) != keys:
      raise ValueError("policy evaluation keys do not match schema")
    route_ids = payload["routeIds"]
    metrics = payload["metrics"]
    if type(payload["artifactIdentity"]) is not str:
      raise ValueError("policy evaluation artifact identity must be text")
    if type(route_ids) is not list or any(type(value) is not str for value in route_ids):
      raise ValueError("policy evaluation routeIds must be a text array")
    if type(metrics) is not list:
      raise ValueError("policy evaluation metrics must be an array")
    policy_payload = payload["policy"]
    policy = None
    if policy_payload is not None:
      if type(policy_payload) is not dict or frozenset(policy_payload) != {
        "dampingRatio",
        "naturalFrequencyPerS",
      }:
        raise ValueError("policy evaluation behavior policy keys do not match schema")
      if any(type(value) not in (int, float) for value in policy_payload.values()):
        raise ValueError("policy evaluation behavior policy values must be numeric")
      policy = BehaviorPolicy(
        natural_frequency_per_s=float(policy_payload["naturalFrequencyPerS"]),
        damping_ratio=float(policy_payload["dampingRatio"]),
      )
    return cls(
      artifact_identity=payload["artifactIdentity"],
      policy=policy,
      route_ids=tuple(route_ids),
      metrics=tuple(PolicyMetric.from_dict(value) for value in metrics),
    )


@dataclass(frozen=True, slots=True)
class _EvaluationCoreIdentity:
  controller_name: str
  core_artifact_sha256: str
  source_openpilot_commit: str
  opendbc_commit: str
  panda_commit: str

  @property
  def platform_identity(self) -> tuple[str, str, str]:
    return self.source_openpilot_commit, self.opendbc_commit, self.panda_commit


@dataclass(frozen=True, slots=True)
class _EvaluationArtifactIdentity:
  role: str
  core: _EvaluationCoreIdentity
  behavior_policy_sha256: str | None
  composed_controller_artifact_sha256: str


def _lower_hex(value: object, length: int, name: str) -> str:
  if (
    type(value) is not str
    or len(value) != length
    or any(character not in "0123456789abcdef" for character in value)
  ):
    raise ValueError(f"{name} must be {length}-character lowercase hexadecimal")
  return value


def _source_commit(value: object, name: str) -> str:
  if (
    type(value) is not str
    or len(value) not in (40, 64)
    or any(character not in "0123456789abcdef" for character in value)
  ):
    raise ValueError(f"{name} must be a full lowercase source commit")
  return value


def _parse_evaluation_artifact_identity(encoded: object) -> _EvaluationArtifactIdentity:
  """Parse the coordinator wire identity without importing it cyclically."""
  if type(encoded) is not str or not encoded:
    raise ValueError("artifact_identity must be canonical identity JSON")
  try:
    payload = json.loads(encoded)
  except (TypeError, ValueError, json.JSONDecodeError) as exc:
    raise ValueError("artifact_identity must be canonical identity JSON") from exc
  if type(payload) is not dict or set(payload) != {
    "behaviorPolicySha256",
    "composedControllerArtifactSha256",
    "core",
    "role",
  }:
    raise ValueError("artifact identity keys do not match schema")
  if canonical_json(payload) != encoded:
    raise ValueError("artifact identity JSON is not canonical")
  role = payload["role"]
  if role not in {"exact_stock", "currently_accepted", "candidate"}:
    raise ValueError("artifact identity role is invalid")
  core_payload = payload["core"]
  if type(core_payload) is not dict or set(core_payload) != {
    "controllerName",
    "coreArtifactSha256",
    "opendbcCommit",
    "pandaCommit",
    "sourceOpenpilotCommit",
  }:
    raise ValueError("artifact core identity keys do not match schema")
  controller_name = core_payload["controllerName"]
  if type(controller_name) is not str or not controller_name.strip():
    raise ValueError("artifact controller name must be non-empty text")
  policy_sha256 = payload["behaviorPolicySha256"]
  if policy_sha256 is not None:
    policy_sha256 = _lower_hex(policy_sha256, 64, "artifact policy identity")
  core = _EvaluationCoreIdentity(
    controller_name=controller_name,
    core_artifact_sha256=_lower_hex(
      core_payload["coreArtifactSha256"], 64, "artifact core identity",
    ),
    source_openpilot_commit=_source_commit(
      core_payload["sourceOpenpilotCommit"], "artifact source commit",
    ),
    opendbc_commit=_source_commit(
      core_payload["opendbcCommit"], "artifact opendbc commit",
    ),
    panda_commit=_source_commit(
      core_payload["pandaCommit"], "artifact panda commit",
    ),
  )
  composed = _lower_hex(
    payload["composedControllerArtifactSha256"],
    64,
    "composed controller artifact identity",
  )
  expected_composed = hashlib.sha256(canonical_json({
    "behaviorPolicySha256": policy_sha256,
    "core": core_payload,
  }).encode("utf-8")).hexdigest()
  if composed != expected_composed:
    raise ValueError("composed controller artifact identity is invalid")
  return _EvaluationArtifactIdentity(role, core, policy_sha256, composed)


@dataclass(frozen=True, slots=True)
class MetricGateRule:
  """Explicit physical bound, comparison uncertainty, and normalization."""

  metric_name: str
  contract: BehaviorContract
  preference: MetricPreference
  noise_floor: float
  margin_normalization: float
  minimum_allowed: float | None
  maximum_allowed: float | None
  minimum_route_count: int
  minimum_window_count: int
  minimum_weighted_support: float
  required_strata: tuple[str, ...]

  def __post_init__(self) -> None:
    try:
      expected_contract = behavior_metric_contract(self.metric_name)
    except ValueError as exc:
      raise ValueError("gate metric is not registered") from exc
    if self.contract is not expected_contract:
      raise ValueError("gate metric assigned to the wrong behavior contract")
    if not math.isfinite(self.noise_floor) or self.noise_floor < 0.0:
      raise ValueError("metric noise floor must be finite and non-negative")
    if not math.isfinite(self.margin_normalization) or self.margin_normalization <= 0.0:
      raise ValueError("metric margin normalization must be finite and positive")
    for value in (self.minimum_allowed, self.maximum_allowed):
      if value is not None and not math.isfinite(value):
        raise ValueError("metric bounds must be finite when present")
    if (
      self.minimum_allowed is not None
      and self.maximum_allowed is not None
      and self.minimum_allowed > self.maximum_allowed
    ):
      raise ValueError("metric bounds are inverted")
    if self.minimum_route_count <= 0 or self.minimum_window_count <= 0:
      raise ValueError("metric coverage counts must be positive")
    if (
      not math.isfinite(self.minimum_weighted_support)
      or self.minimum_weighted_support <= 0.0
    ):
      raise ValueError("minimum weighted support must be finite and positive")
    if not self.required_strata or tuple(sorted(set(self.required_strata))) != self.required_strata:
      raise ValueError("required strata must be non-empty, unique, and sorted")

  def to_dict(self) -> dict[str, Any]:
    return {
      "contract": self.contract.value,
      "marginNormalization": self.margin_normalization,
      "maximumAllowed": self.maximum_allowed,
      "metricName": self.metric_name,
      "minimumAllowed": self.minimum_allowed,
      "minimumRouteCount": self.minimum_route_count,
      "minimumWindowCount": self.minimum_window_count,
      "minimumWeightedSupport": self.minimum_weighted_support,
      "noiseFloor": self.noise_floor,
      "preference": self.preference.value,
      "requiredStrata": list(self.required_strata),
    }


@dataclass(frozen=True, slots=True)
class PairedRouteUncertainty:
  """Observed envelope of preference-normalized whole-route deltas."""

  route_ids: tuple[str, ...]
  mean: float
  uncertainty: float
  lower: float
  upper: float
  tolerance: float

  def to_dict(self) -> dict[str, Any]:
    return {
      "lower": self.lower,
      "mean": self.mean,
      "routeCount": len(self.route_ids),
      "routeIds": list(self.route_ids),
      "tolerance": self.tolerance,
      "uncertainty": self.uncertainty,
      "upper": self.upper,
    }


@dataclass(frozen=True, slots=True)
class MetricGateVerdict:
  metric_name: str
  contract: BehaviorContract
  passed: bool
  margin: float | None
  reasons: tuple[str, ...]
  paired_against_stock: PairedRouteUncertainty | None = None
  paired_against_accepted: PairedRouteUncertainty | None = None


@dataclass(frozen=True, slots=True)
class ContractGateVerdict:
  contract: BehaviorContract
  passed: bool
  margin: float | None
  metrics: tuple[MetricGateVerdict, ...]


@dataclass(frozen=True, slots=True)
class CandidateGateVerdict:
  candidate: PolicyCandidate
  passed: bool
  target_metric_name: str
  target_improvement: float | None
  target_noise_floor: float
  target_materially_improved: bool
  worst_contract_margin: float | None
  contracts: tuple[ContractGateVerdict, ...]
  route_ids: tuple[str, ...]
  candidate_evaluation_sha256: str
  exact_stock_evaluation_sha256: str
  accepted_evaluation_sha256: str
  gate_spec_sha256: str

  def to_dict(self) -> dict[str, Any]:
    return {
      "candidate": {
        "canonicalIndex": self.candidate.canonical_index,
        "squaredLogDisplacement": self.candidate.squared_log_displacement,
        "policy": self.candidate.policy.to_dict(),
      },
      "candidateEvaluationSha256": self.candidate_evaluation_sha256,
      "contracts": [
        {
          "contract": contract.contract.value,
          "margin": contract.margin,
          "metrics": [
            {
              "contract": metric.contract.value,
              "margin": metric.margin,
              "metricName": metric.metric_name,
              "pairedAgainstAccepted": (
                None
                if metric.paired_against_accepted is None
                else metric.paired_against_accepted.to_dict()
              ),
              "pairedAgainstStock": (
                None
                if metric.paired_against_stock is None
                else metric.paired_against_stock.to_dict()
              ),
              "passed": metric.passed,
              "reasons": list(metric.reasons),
            }
            for metric in contract.metrics
          ],
          "passed": contract.passed,
        }
        for contract in self.contracts
      ],
      "passed": self.passed,
      "acceptedEvaluationSha256": self.accepted_evaluation_sha256,
      "exactStockEvaluationSha256": self.exact_stock_evaluation_sha256,
      "gateSpecSha256": self.gate_spec_sha256,
      "routeIds": list(self.route_ids),
      "targetImprovement": self.target_improvement,
      "targetMateriallyImproved": self.target_materially_improved,
      "targetMetricName": self.target_metric_name,
      "targetNoiseFloor": self.target_noise_floor,
      "worstContractMargin": self.worst_contract_margin,
    }


def _paired_route_uncertainty(
  candidate: PolicyMetric | PolicyStratumMetric,
  reference: PolicyMetric | PolicyStratumMetric,
  preference: MetricPreference,
  method: str,
  minimum_paired_route_count: int,
) -> tuple[PairedRouteUncertainty | None, str | None]:
  if method != PAIRED_ROUTE_UNCERTAINTY_METHOD:
    return None, "paired_uncertainty_method_unsupported"
  candidate_values = candidate.route_values_by_id
  reference_values = reference.route_values_by_id
  if tuple(candidate_values) != tuple(reference_values):
    return None, "paired_route_values_mismatch"
  route_ids = tuple(candidate_values)
  if len(route_ids) < minimum_paired_route_count:
    return None, "paired_route_count_below_minimum"
  deltas: list[float] = []
  tolerances: list[float] = []
  for route_id in route_ids:
    candidate_value = candidate_values[route_id]
    reference_value = reference_values[route_id]
    delta = (
      reference_value - candidate_value
      if preference is MetricPreference.LOWER_IS_BETTER
      else candidate_value - reference_value
    )
    tolerance = (
      64.0
      * sys.float_info.epsilon
      * max(abs(candidate_value), abs(reference_value), 1.0)
    )
    if abs(delta) <= tolerance:
      delta = 0.0
    deltas.append(delta)
    tolerances.append(tolerance)
  mean = math.fsum(deltas) / len(deltas)
  uncertainty = max(abs(delta - mean) for delta in deltas)
  return PairedRouteUncertainty(
    route_ids=route_ids,
    mean=mean,
    uncertainty=uncertainty,
    lower=mean - uncertainty,
    upper=mean + uncertainty,
    tolerance=max(tolerances),
  ), None


def _metric_margin(
  candidate: PolicyMetric,
  stock: PolicyMetric,
  accepted: PolicyMetric,
  rule: MetricGateRule,
  paired_uncertainty_method: str,
  minimum_paired_route_count: int,
) -> MetricGateVerdict:
  reasons: list[str] = []
  margins: list[float] = []
  candidate_strata = {metric.stratum: metric for metric in candidate.stratum_metrics}
  stock_strata = {metric.stratum: metric for metric in stock.stratum_metrics}
  accepted_strata = {metric.stratum: metric for metric in accepted.stratum_metrics}

  all_strata = tuple(sorted(
    set(candidate_strata) | set(stock_strata) | set(accepted_strata) | set(rule.required_strata),
  ))
  for stratum in all_strata:
    values = (
      ("candidate", candidate_strata.get(stratum)),
      ("stock", stock_strata.get(stratum)),
      ("accepted", accepted_strata.get(stratum)),
    )
    required = stratum in rule.required_strata
    if not required and all(
      metric is not None and metric.disposition is MetricDisposition.NOT_APPLICABLE
      for _, metric in values
    ):
      continue
    stratum_failed = False
    for label, metric in values:
      if metric is None:
        reason = "missing_required_stratum" if required else "missing_stratum"
        reasons.append(f"{stratum}:{label}_{reason}")
        stratum_failed = True
        continue
      if metric.route_count < rule.minimum_route_count:
        reasons.append(f"{stratum}:{label}_route_coverage_below_minimum")
        stratum_failed = True
      if metric.window_count < rule.minimum_window_count:
        reasons.append(f"{stratum}:{label}_window_coverage_below_minimum")
        stratum_failed = True
      if metric.weighted_support < rule.minimum_weighted_support:
        reasons.append(f"{stratum}:{label}_weighted_support_below_minimum")
        stratum_failed = True
      if metric.physical_failure_window_ids:
        reasons.extend(
          f"{stratum}:{label}_physical_unscoreable:{window_id}"
          for window_id in metric.physical_failure_window_ids
        )
        stratum_failed = True
      if not metric.defined:
        reasons.extend(
          f"{stratum}:{label}_metric_undefined:{reason}"
          for reason in metric.exclusions
        )
        stratum_failed = True
    if stratum_failed:
      continue
    candidate_value = values[0][1]
    stock_value = values[1][1]
    accepted_value = values[2][1]
    assert candidate_value is not None and candidate_value.value is not None
    assert stock_value is not None and accepted_value is not None
    if len({
      candidate_value.coverage_identity_sha256,
      stock_value.coverage_identity_sha256,
      accepted_value.coverage_identity_sha256,
    }) != 1:
      reasons.append(f"{stratum}:candidate_reference_coverage_mismatch")
      continue
    if rule.minimum_allowed is not None:
      margins.append(
        (candidate_value.value - rule.minimum_allowed) / rule.margin_normalization,
      )
      if candidate_value.value < rule.minimum_allowed:
        reasons.append(f"{stratum}:below_absolute_minimum")
    if rule.maximum_allowed is not None:
      margins.append(
        (rule.maximum_allowed - candidate_value.value) / rule.margin_normalization,
      )
      if candidate_value.value > rule.maximum_allowed:
        reasons.append(f"{stratum}:above_absolute_maximum")
    for label, reference in (("stock", stock_value), ("accepted", accepted_value)):
      paired, error = _paired_route_uncertainty(
        candidate_value,
        reference,
        rule.preference,
        paired_uncertainty_method,
        minimum_paired_route_count,
      )
      if error is not None:
        reasons.append(f"{stratum}:{label}_{error}")
        continue
      assert paired is not None
      comparison = paired.lower + rule.noise_floor
      margins.append(comparison / rule.margin_normalization)
      if paired.lower < -rule.noise_floor:
        reasons.append(f"{stratum}:regressed_vs_{label}")

  stock_pair = None
  accepted_pair = None
  if candidate.defined and stock.defined and accepted.defined:
    stock_pair, _ = _paired_route_uncertainty(
      candidate,
      stock,
      rule.preference,
      paired_uncertainty_method,
      minimum_paired_route_count,
    )
    accepted_pair, _ = _paired_route_uncertainty(
      candidate,
      accepted,
      rule.preference,
      paired_uncertainty_method,
      minimum_paired_route_count,
    )
  margin = min(margins) if margins else None
  return MetricGateVerdict(
    metric_name=rule.metric_name,
    contract=rule.contract,
    passed=not reasons,
    margin=margin,
    reasons=tuple(reasons),
    paired_against_stock=stock_pair,
    paired_against_accepted=accepted_pair,
  )


def _validate_comparison_identities(
  candidate: PolicyCandidate,
  candidate_evaluation: PolicyEvaluation,
  exact_stock_evaluation: PolicyEvaluation,
  accepted_evaluation: PolicyEvaluation,
) -> None:
  """Fail closed before any metric can vote under a mislabeled opponent."""
  candidate_identity = _parse_evaluation_artifact_identity(
    candidate_evaluation.artifact_identity,
  )
  stock_identity = _parse_evaluation_artifact_identity(
    exact_stock_evaluation.artifact_identity,
  )
  accepted_identity = _parse_evaluation_artifact_identity(
    accepted_evaluation.artifact_identity,
  )
  if candidate_evaluation.policy != candidate.policy:
    raise ValueError("candidate evaluation policy identity mismatch")
  if (
    candidate_identity.role != "candidate"
    or candidate_identity.behavior_policy_sha256 != candidate.policy.sha256
  ):
    raise ValueError("candidate evaluation artifact role or policy mismatch")
  if exact_stock_evaluation.policy is not None:
    raise ValueError("exact-stock evaluation must have a null policy")
  if (
    stock_identity.role != "exact_stock"
    or stock_identity.behavior_policy_sha256 is not None
  ):
    raise ValueError("exact-stock evaluation artifact identity mismatch")

  if accepted_evaluation.policy is None:
    if accepted_identity.behavior_policy_sha256 is not None:
      raise ValueError("bootstrap accepted evaluation has a policy identity")
    if accepted_identity.role not in {"exact_stock", "currently_accepted"}:
      raise ValueError("bootstrap accepted evaluation artifact role mismatch")
    if accepted_identity.core != stock_identity.core:
      raise ValueError("bootstrap incumbent does not alias exact-stock core")
    if accepted_evaluation.metrics != exact_stock_evaluation.metrics:
      raise ValueError("bootstrap incumbent does not alias exact-stock evaluation")
  elif (
    accepted_identity.role != "currently_accepted"
    or accepted_identity.behavior_policy_sha256 != accepted_evaluation.policy.sha256
  ):
    raise ValueError("accepted evaluation artifact role or policy mismatch")

  platform_identities = {
    candidate_identity.core.platform_identity,
    stock_identity.core.platform_identity,
    accepted_identity.core.platform_identity,
  }
  if len(platform_identities) != 1:
    raise ValueError("controller evaluations use different platform source identities")


def evaluate_candidate(
  candidate: PolicyCandidate,
  candidate_evaluation: PolicyEvaluation,
  exact_stock_evaluation: PolicyEvaluation,
  accepted_evaluation: PolicyEvaluation,
  rules: tuple[MetricGateRule, ...],
  target_metric_name: str,
  paired_uncertainty_method: str,
  minimum_paired_route_count: int,
) -> CandidateGateVerdict:
  """Evaluate three non-tradeable contracts against both reference artifacts."""
  if paired_uncertainty_method != PAIRED_ROUTE_UNCERTAINTY_METHOD:
    raise ValueError("paired route uncertainty method is unsupported")
  if minimum_paired_route_count < 2:
    raise ValueError("paired route uncertainty requires at least two routes")
  _validate_comparison_identities(
    candidate,
    candidate_evaluation,
    exact_stock_evaluation,
    accepted_evaluation,
  )
  if not (
    candidate_evaluation.route_ids
    == exact_stock_evaluation.route_ids
    == accepted_evaluation.route_ids
  ):
    raise ValueError("candidate and references must use the same route partition")
  if len({rule.metric_name for rule in rules}) != len(rules):
    raise ValueError("gate rule metric names must be unique")
  if {rule.contract for rule in rules} != set(BehaviorContract):
    raise ValueError("every Smooth/Swift/Strong contract needs at least one rule")
  ordered_rules = tuple(sorted(rules, key=lambda rule: rule.metric_name))
  target_rules = tuple(rule for rule in ordered_rules if rule.metric_name == target_metric_name)
  if len(target_rules) != 1:
    raise ValueError("declared target must identify exactly one gate rule")

  metric_verdicts = tuple(
    _metric_margin(
      candidate_evaluation.metric(rule.metric_name),
      exact_stock_evaluation.metric(rule.metric_name),
      accepted_evaluation.metric(rule.metric_name),
      rule,
      paired_uncertainty_method,
      minimum_paired_route_count,
    )
    for rule in ordered_rules
  )
  contract_verdicts: list[ContractGateVerdict] = []
  for contract in BehaviorContract:
    metrics = tuple(verdict for verdict in metric_verdicts if verdict.contract is contract)
    finite_margins = tuple(metric.margin for metric in metrics if metric.margin is not None)
    contract_verdicts.append(ContractGateVerdict(
      contract=contract,
      passed=all(metric.passed for metric in metrics),
      margin=min(finite_margins) if len(finite_margins) == len(metrics) else None,
      metrics=metrics,
    ))

  target_rule = target_rules[0]
  target_verdict = next(
    verdict
    for verdict in metric_verdicts
    if verdict.metric_name == target_metric_name
  )
  target_pairs = tuple(
    pair
    for pair in (
      target_verdict.paired_against_stock,
      target_verdict.paired_against_accepted,
    )
    if pair is not None
  )
  target_improvement = (
    min(pair.mean for pair in target_pairs)
    if len(target_pairs) == 2
    else None
  )
  # The balanced target remains a ranking diagnostic. Acceptance is decided
  # only by the independently validated speed-by-maneuver contract strata.
  target_material = len(target_pairs) == 2 and all(
    pair.lower > target_rule.noise_floor
    for pair in target_pairs
  )
  contract_margins = tuple(
    contract.margin
    for contract in contract_verdicts
    if contract.margin is not None
  )
  worst_margin = (
    min(contract_margins)
    if len(contract_margins) == len(contract_verdicts)
    else None
  )
  if target_improvement is not None and worst_margin is not None:
    worst_margin = min(
      worst_margin,
      *(
        (pair.lower - target_rule.noise_floor)
        / target_rule.margin_normalization
        for pair in target_pairs
      ),
    )
  passed = all(contract.passed for contract in contract_verdicts)
  gate_spec_sha256 = hashlib.sha256(canonical_json({
    "rules": [rule.to_dict() for rule in ordered_rules],
    "minimumPairedRouteCount": minimum_paired_route_count,
    "pairedUncertaintyMethod": paired_uncertainty_method,
    "targetMetricName": target_metric_name,
  }).encode("utf-8")).hexdigest()
  return CandidateGateVerdict(
    candidate=candidate,
    passed=passed,
    target_metric_name=target_metric_name,
    target_improvement=target_improvement,
    target_noise_floor=target_rule.noise_floor,
    target_materially_improved=target_material,
    worst_contract_margin=worst_margin,
    contracts=tuple(contract_verdicts),
    route_ids=candidate_evaluation.route_ids,
    candidate_evaluation_sha256=candidate_evaluation.sha256,
    exact_stock_evaluation_sha256=exact_stock_evaluation.sha256,
    accepted_evaluation_sha256=accepted_evaluation.sha256,
    gate_spec_sha256=gate_spec_sha256,
  )


@dataclass(frozen=True, slots=True)
class TrainingSelection:
  training_route_ids: tuple[str, ...]
  winner: PolicyCandidate
  winner_verdict: CandidateGateVerdict
  all_verdicts: tuple[CandidateGateVerdict, ...]
  candidate_grid_sha256: str

  def to_dict(self) -> dict[str, Any]:
    return {
      "allVerdicts": [verdict.to_dict() for verdict in self.all_verdicts],
      "candidateGridSha256": self.candidate_grid_sha256,
      "trainingRouteIds": list(self.training_route_ids),
      "winnerCanonicalIndex": self.winner.canonical_index,
      "winnerPolicy": self.winner.policy.to_dict(),
      "winnerVerdict": self.winner_verdict.to_dict(),
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())

  @property
  def sha256(self) -> str:
    return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def _required_margin(verdict: CandidateGateVerdict) -> float:
  if verdict.worst_contract_margin is None:
    raise AssertionError("a passing candidate must have a defined gate margin")
  return verdict.worst_contract_margin


def select_training_winner(
  grid: tuple[PolicyCandidate, ...],
  candidate_evaluations: Iterable[PolicyEvaluation],
  exact_stock_evaluation: PolicyEvaluation,
  accepted_evaluation: PolicyEvaluation,
  rules: tuple[MetricGateRule, ...],
  target_metric_name: str,
  paired_uncertainty_method: str,
  minimum_paired_route_count: int,
) -> TrainingSelection | None:
  """Freeze one training winner using canonical, fully deterministic ties."""
  evaluations = tuple(candidate_evaluations)
  evaluation_by_policy = {
    evaluation.policy: evaluation
    for evaluation in evaluations
  }
  if len(evaluation_by_policy) != len(evaluations):
    raise ValueError("candidate evaluations contain duplicate policy identities")
  if None in evaluation_by_policy:
    raise ValueError("candidate evaluations must identify their policy")
  if set(evaluation_by_policy) != {candidate.policy for candidate in grid}:
    raise ValueError("candidate evaluations must cover the exact bounded grid")
  verdicts = tuple(
    evaluate_candidate(
      candidate,
      evaluation_by_policy[candidate.policy],
      exact_stock_evaluation,
      accepted_evaluation,
      rules,
      target_metric_name,
      paired_uncertainty_method,
      minimum_paired_route_count,
    )
    for candidate in grid
  )
  passing = tuple(verdict for verdict in verdicts if verdict.passed)
  if not passing:
    return None
  winner_verdict = min(
    passing,
    key=lambda verdict: (
      -_required_margin(verdict),
      verdict.candidate.squared_log_displacement,
      verdict.candidate.canonical_index,
    ),
  )
  return TrainingSelection(
    training_route_ids=exact_stock_evaluation.route_ids,
    winner=winner_verdict.candidate,
    winner_verdict=winner_verdict,
    all_verdicts=verdicts,
    candidate_grid_sha256=hashlib.sha256(canonical_json([
      {
        "canonicalIndex": candidate.canonical_index,
        "squaredLogDisplacement": candidate.squared_log_displacement,
        "policy": candidate.policy.to_dict(),
      }
      for candidate in grid
    ]).encode("utf-8")).hexdigest(),
  )


@dataclass(frozen=True, slots=True)
class HeldOutValidation:
  selection_sha256: str
  validation_route_ids: tuple[str, ...]
  accepted: bool
  frozen_winner_verdict: CandidateGateVerdict

  def to_dict(self) -> dict[str, Any]:
    return {
      "accepted": self.accepted,
      "frozenWinnerVerdict": self.frozen_winner_verdict.to_dict(),
      "selectionSha256": self.selection_sha256,
      "validationRouteIds": list(self.validation_route_ids),
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())


def validate_frozen_winner(
  selection: TrainingSelection,
  frozen_winner_evaluation: PolicyEvaluation,
  exact_stock_evaluation: PolicyEvaluation,
  accepted_evaluation: PolicyEvaluation,
  rules: tuple[MetricGateRule, ...],
  target_metric_name: str,
  paired_uncertainty_method: str,
  minimum_paired_route_count: int,
) -> HeldOutValidation:
  """Accept or reject only the training winner on disjoint held-out routes."""
  validation_routes = frozen_winner_evaluation.route_ids
  if set(selection.training_route_ids) & set(validation_routes):
    raise ValueError("training and held-out routes must be disjoint")
  if frozen_winner_evaluation.policy != selection.winner.policy:
    raise ValueError("held-out evaluation is not the frozen training winner")
  verdict = evaluate_candidate(
    selection.winner,
    frozen_winner_evaluation,
    exact_stock_evaluation,
    accepted_evaluation,
    rules,
    target_metric_name,
    paired_uncertainty_method,
    minimum_paired_route_count,
  )
  return HeldOutValidation(
    selection_sha256=selection.sha256,
    validation_route_ids=validation_routes,
    accepted=verdict.passed,
    frozen_winner_verdict=verdict,
  )

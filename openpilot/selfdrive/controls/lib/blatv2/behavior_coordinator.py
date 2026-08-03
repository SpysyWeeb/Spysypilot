"""Pure offroad coordinator for BLaTv2 behavioral learning.

This module owns reproducible route partitioning and orchestration only.  It
does not decode routes, run a controller, persist Params, publish an approved
artifact, or activate anything.  Behavioral evidence is intentionally stricter
than physical-calibration evidence: every route in one behavioral population
must have the same exact recorded controller/source identity.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import math
import re
from typing import Any

from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import (
  BehaviorSourceIdentity,
  canonical_json,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_metrics import (
  BehaviorContract,
  BehaviorMetricConfig,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_policy import (
  PAIRED_ROUTE_UNCERTAINTY_METHOD,
  BehaviorPolicy,
  CandidateGateVerdict,
  HeldOutValidation,
  MetricGateRule,
  MetricPreference,
  PolicyEvaluation,
  PolicyGridSpec,
  TrainingSelection,
  build_candidate_grid,
  select_training_winner,
  validate_frozen_winner,
)


BEHAVIOR_GATE_SPEC_SCHEMA_VERSION = 3
BEHAVIOR_FINALIZATION_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


def _sha256_json(payload: object) -> str:
  return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _strict_object(value: object, keys: frozenset[str], name: str) -> dict[str, Any]:
  if type(value) is not dict or frozenset(value) != keys:
    raise ValueError(f"{name} keys do not match the schema")
  return value


def _strict_number(value: object, name: str) -> float:
  if type(value) not in (int, float):
    raise ValueError(f"{name} must be numeric")
  result = float(value)
  if not math.isfinite(result):
    raise ValueError(f"{name} must be finite")
  return result


def _strict_optional_number(value: object, name: str) -> float | None:
  return None if value is None else _strict_number(value, name)


def _strict_optional_int(value: object, name: str) -> int | None:
  if value is None:
    return None
  if type(value) is not int:
    raise ValueError(f"{name} must be an integer or null")
  return value


@dataclass(frozen=True, slots=True)
class CandidateGridBounds:
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
        raise ValueError(f"{name} must include zero")
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
      raise ValueError("candidate-grid bounds must be finite and positive")
    if self.minimum_natural_frequency_per_s >= self.maximum_natural_frequency_per_s:
      raise ValueError("natural-frequency grid bounds are inverted")
    if self.minimum_damping_ratio >= self.maximum_damping_ratio:
      raise ValueError("damping grid bounds are inverted")

  def policy_grid(self, incumbent: BehaviorPolicy) -> PolicyGridSpec:
    return PolicyGridSpec(
      incumbent=incumbent,
      natural_frequency_log_offsets=self.natural_frequency_log_offsets,
      damping_ratio_log_offsets=self.damping_ratio_log_offsets,
      minimum_natural_frequency_per_s=self.minimum_natural_frequency_per_s,
      maximum_natural_frequency_per_s=self.maximum_natural_frequency_per_s,
      minimum_damping_ratio=self.minimum_damping_ratio,
      maximum_damping_ratio=self.maximum_damping_ratio,
    )

  def to_dict(self) -> dict[str, Any]:
    return {
      "dampingRatioLogOffsets": list(self.damping_ratio_log_offsets),
      "maximumDampingRatio": self.maximum_damping_ratio,
      "maximumNaturalFrequencyPerS": self.maximum_natural_frequency_per_s,
      "minimumDampingRatio": self.minimum_damping_ratio,
      "minimumNaturalFrequencyPerS": self.minimum_natural_frequency_per_s,
      "naturalFrequencyLogOffsets": list(self.natural_frequency_log_offsets),
    }


@dataclass(frozen=True, slots=True)
class RoutePartitionSpec:
  """Exactly one validation sizing method plus a hash-bound seed identity."""

  validation_fraction: float | None
  validation_route_count: int | None
  seed_identity_sha256: str

  def __post_init__(self) -> None:
    if (self.validation_fraction is None) == (self.validation_route_count is None):
      raise ValueError("exactly one validation fraction or count is required")
    if self.validation_fraction is not None and (
      not math.isfinite(self.validation_fraction)
      or not 0.0 < self.validation_fraction < 1.0
    ):
      raise ValueError("validation fraction must be finite and in (0, 1)")
    if self.validation_route_count is not None and self.validation_route_count <= 0:
      raise ValueError("validation route count must be positive")
    if _SHA256_RE.fullmatch(self.seed_identity_sha256) is None:
      raise ValueError("partition seed identity must be lowercase SHA-256")

  def to_dict(self) -> dict[str, Any]:
    return {
      "seedIdentitySha256": self.seed_identity_sha256,
      "validationFraction": self.validation_fraction,
      "validationRouteCount": self.validation_route_count,
    }


@dataclass(frozen=True, slots=True)
class BehaviorGateSpec:
  """Complete versioned metric, gate, search, and partition authority."""

  schema_version: int
  provenance: str
  metric_config: BehaviorMetricConfig
  metric_rules: tuple[MetricGateRule, ...]
  target_metric_name: str
  paired_uncertainty_method: str
  minimum_paired_route_count: int
  candidate_grid: CandidateGridBounds
  route_partition: RoutePartitionSpec

  def __post_init__(self) -> None:
    if self.schema_version != BEHAVIOR_GATE_SPEC_SCHEMA_VERSION:
      raise ValueError("behavior gate-spec schema is incompatible")
    if not self.provenance.strip():
      raise ValueError("behavior gate-spec provenance must not be empty")
    names = tuple(rule.metric_name for rule in self.metric_rules)
    if not names or len(set(names)) != len(names):
      raise ValueError("behavior gate rules must be non-empty and unique")
    if {rule.contract for rule in self.metric_rules} != set(BehaviorContract):
      raise ValueError("Smooth, Swift, and Strong must each have a gate")
    if names.count(self.target_metric_name) != 1:
      raise ValueError("target metric must name exactly one gate rule")
    if self.paired_uncertainty_method != PAIRED_ROUTE_UNCERTAINTY_METHOD:
      raise ValueError("behavior gate paired uncertainty method is unsupported")
    if self.minimum_paired_route_count < 2:
      raise ValueError("behavior gate needs at least two paired routes")

  def to_dict(self) -> dict[str, Any]:
    return {
      "candidateGrid": self.candidate_grid.to_dict(),
      "metricConfig": self.metric_config.to_dict(),
      "metricRules": [
        rule.to_dict()
        for rule in sorted(self.metric_rules, key=lambda rule: rule.metric_name)
      ],
      "minimumPairedRouteCount": self.minimum_paired_route_count,
      "pairedUncertaintyMethod": self.paired_uncertainty_method,
      "provenance": self.provenance,
      "routePartition": self.route_partition.to_dict(),
      "schemaVersion": self.schema_version,
      "targetMetricName": self.target_metric_name,
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())

  @property
  def sha256(self) -> str:
    return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

  @classmethod
  def from_json(cls, encoded: str) -> BehaviorGateSpec:
    try:
      payload = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
      raise ValueError("behavior gate spec is invalid JSON") from exc
    root = _strict_object(payload, frozenset((
      "candidateGrid",
      "metricConfig",
      "metricRules",
      "minimumPairedRouteCount",
      "pairedUncertaintyMethod",
      "provenance",
      "routePartition",
      "schemaVersion",
      "targetMetricName",
    )), "behavior gate spec")
    if type(root["schemaVersion"]) is not int:
      raise ValueError("behavior gate-spec schema version must be integer")
    if (
      type(root["provenance"]) is not str
      or type(root["targetMetricName"]) is not str
      or type(root["pairedUncertaintyMethod"]) is not str
    ):
      raise ValueError("gate-spec provenance and target metric must be text")
    if type(root["minimumPairedRouteCount"]) is not int:
      raise ValueError("minimum paired route count must be integer")

    metric_payload = _strict_object(root["metricConfig"], frozenset((
      "burstWindowS",
      "chatterTorqueRateThresholdPerS",
      "completionDeliveredFraction",
      "correctionCurvatureThreshold1pm",
      "growingErrorEpsilon1pm",
      "maximumRouteWindowsPerStratum",
      "minimumSamples",
      "releaseCrossingFraction",
      "speedNodesMps",
      "turnInCrossingFraction",
      "unusedHeadroomThreshold",
    )), "behavior metric config")
    if type(metric_payload["minimumSamples"]) is not int:
      raise ValueError("minimumSamples must be integer")
    if type(metric_payload["maximumRouteWindowsPerStratum"]) is not int:
      raise ValueError("maximumRouteWindowsPerStratum must be integer")
    speed_nodes = metric_payload["speedNodesMps"]
    if type(speed_nodes) is not list:
      raise ValueError("speedNodesMps must be an array")
    metric_config = BehaviorMetricConfig(
      burst_window_s=_strict_number(metric_payload["burstWindowS"], "burstWindowS"),
      chatter_torque_rate_threshold_per_s=_strict_number(
        metric_payload["chatterTorqueRateThresholdPerS"],
        "chatterTorqueRateThresholdPerS",
      ),
      turn_in_crossing_fraction=_strict_number(
        metric_payload["turnInCrossingFraction"],
        "turnInCrossingFraction",
      ),
      release_crossing_fraction=_strict_number(
        metric_payload["releaseCrossingFraction"],
        "releaseCrossingFraction",
      ),
      correction_curvature_threshold_1pm=_strict_number(
        metric_payload["correctionCurvatureThreshold1pm"],
        "correctionCurvatureThreshold1pm",
      ),
      unused_headroom_threshold=_strict_number(
        metric_payload["unusedHeadroomThreshold"],
        "unusedHeadroomThreshold",
      ),
      growing_error_epsilon_1pm=_strict_number(
        metric_payload["growingErrorEpsilon1pm"],
        "growingErrorEpsilon1pm",
      ),
      completion_delivered_fraction=_strict_number(
        metric_payload["completionDeliveredFraction"],
        "completionDeliveredFraction",
      ),
      minimum_samples=metric_payload["minimumSamples"],
      speed_nodes_mps=tuple(
        _strict_number(value, "speedNodesMps item")
        for value in speed_nodes
      ),
      maximum_route_windows_per_stratum=metric_payload[
        "maximumRouteWindowsPerStratum"
      ],
    )

    rules_payload = root["metricRules"]
    if type(rules_payload) is not list:
      raise ValueError("metricRules must be an array")
    metric_rules: list[MetricGateRule] = []
    for index, raw_rule in enumerate(rules_payload):
      rule = _strict_object(raw_rule, frozenset((
        "contract",
        "marginNormalization",
        "maximumAllowed",
        "metricName",
        "minimumAllowed",
        "minimumRouteCount",
        "minimumWeightedSupport",
        "minimumWindowCount",
        "noiseFloor",
        "preference",
        "requiredStrata",
      )), f"metric rule {index}")
      if type(rule["metricName"]) is not str:
        raise ValueError("metric rule name must be text")
      if type(rule["minimumRouteCount"]) is not int or type(rule["minimumWindowCount"]) is not int:
        raise ValueError("metric coverage counts must be integers")
      required_strata = rule["requiredStrata"]
      if type(required_strata) is not list or any(type(value) is not str for value in required_strata):
        raise ValueError("requiredStrata must be a text array")
      try:
        contract = BehaviorContract(rule["contract"])
        preference = MetricPreference(rule["preference"])
      except (TypeError, ValueError) as exc:
        raise ValueError("metric rule enum value is invalid") from exc
      metric_rules.append(MetricGateRule(
        metric_name=rule["metricName"],
        contract=contract,
        preference=preference,
        noise_floor=_strict_number(rule["noiseFloor"], "noiseFloor"),
        margin_normalization=_strict_number(
          rule["marginNormalization"],
          "marginNormalization",
        ),
        minimum_allowed=_strict_optional_number(rule["minimumAllowed"], "minimumAllowed"),
        maximum_allowed=_strict_optional_number(rule["maximumAllowed"], "maximumAllowed"),
        minimum_route_count=rule["minimumRouteCount"],
        minimum_window_count=rule["minimumWindowCount"],
        minimum_weighted_support=_strict_number(
          rule["minimumWeightedSupport"],
          "minimumWeightedSupport",
        ),
        required_strata=tuple(required_strata),
      ))

    grid_payload = _strict_object(root["candidateGrid"], frozenset((
      "dampingRatioLogOffsets",
      "maximumDampingRatio",
      "maximumNaturalFrequencyPerS",
      "minimumDampingRatio",
      "minimumNaturalFrequencyPerS",
      "naturalFrequencyLogOffsets",
    )), "candidate grid")
    natural_offsets = grid_payload["naturalFrequencyLogOffsets"]
    damping_offsets = grid_payload["dampingRatioLogOffsets"]
    if type(natural_offsets) is not list or type(damping_offsets) is not list:
      raise ValueError("candidate-grid offsets must be arrays")
    candidate_grid = CandidateGridBounds(
      natural_frequency_log_offsets=tuple(
        _strict_number(value, "naturalFrequencyLogOffsets item")
        for value in natural_offsets
      ),
      damping_ratio_log_offsets=tuple(
        _strict_number(value, "dampingRatioLogOffsets item")
        for value in damping_offsets
      ),
      minimum_natural_frequency_per_s=_strict_number(
        grid_payload["minimumNaturalFrequencyPerS"],
        "minimumNaturalFrequencyPerS",
      ),
      maximum_natural_frequency_per_s=_strict_number(
        grid_payload["maximumNaturalFrequencyPerS"],
        "maximumNaturalFrequencyPerS",
      ),
      minimum_damping_ratio=_strict_number(
        grid_payload["minimumDampingRatio"],
        "minimumDampingRatio",
      ),
      maximum_damping_ratio=_strict_number(
        grid_payload["maximumDampingRatio"],
        "maximumDampingRatio",
      ),
    )

    partition_payload = _strict_object(root["routePartition"], frozenset((
      "seedIdentitySha256",
      "validationFraction",
      "validationRouteCount",
    )), "route partition")
    if type(partition_payload["seedIdentitySha256"]) is not str:
      raise ValueError("partition seed identity must be text")
    route_partition = RoutePartitionSpec(
      validation_fraction=_strict_optional_number(
        partition_payload["validationFraction"],
        "validationFraction",
      ),
      validation_route_count=_strict_optional_int(
        partition_payload["validationRouteCount"],
        "validationRouteCount",
      ),
      seed_identity_sha256=partition_payload["seedIdentitySha256"],
    )
    return cls(
      schema_version=root["schemaVersion"],
      provenance=root["provenance"],
      metric_config=metric_config,
      metric_rules=tuple(metric_rules),
      target_metric_name=root["targetMetricName"],
      paired_uncertainty_method=root["pairedUncertaintyMethod"],
      minimum_paired_route_count=root["minimumPairedRouteCount"],
      candidate_grid=candidate_grid,
      route_partition=route_partition,
    )


@dataclass(frozen=True, slots=True)
class BehaviorRouteEvidenceIdentity:
  """Whole-route evidence identity; no frame-level partition API exists."""

  route_id: str
  route_evidence_sha256: str
  recorded_source: BehaviorSourceIdentity

  def __post_init__(self) -> None:
    if not self.route_id.strip():
      raise ValueError("route_id must not be empty")
    if _SHA256_RE.fullmatch(self.route_evidence_sha256) is None:
      raise ValueError("route evidence identity must be lowercase SHA-256")

  def to_dict(self) -> dict[str, Any]:
    return {
      "recordedSource": self.recorded_source.to_dict(),
      "routeEvidenceSha256": self.route_evidence_sha256,
      "routeId": self.route_id,
    }


@dataclass(frozen=True, slots=True)
class RoutePartition:
  seed_identity_sha256: str
  recorded_source_identity_sha256: str
  training_routes: tuple[BehaviorRouteEvidenceIdentity, ...]
  validation_routes: tuple[BehaviorRouteEvidenceIdentity, ...]

  @property
  def training_route_ids(self) -> tuple[str, ...]:
    return tuple(route.route_id for route in self.training_routes)

  @property
  def validation_route_ids(self) -> tuple[str, ...]:
    return tuple(route.route_id for route in self.validation_routes)

  def to_dict(self) -> dict[str, Any]:
    return {
      "recordedSourceIdentitySha256": self.recorded_source_identity_sha256,
      "seedIdentitySha256": self.seed_identity_sha256,
      "trainingRoutes": [route.to_dict() for route in self.training_routes],
      "validationRoutes": [route.to_dict() for route in self.validation_routes],
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())

  @property
  def sha256(self) -> str:
    return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def partition_whole_routes(
  routes: Iterable[BehaviorRouteEvidenceIdentity],
  spec: RoutePartitionSpec,
) -> RoutePartition:
  """Hash-rank canonical route IDs; a route can never straddle partitions."""
  canonical_routes = tuple(sorted(routes, key=lambda route: route.route_id))
  route_ids = tuple(route.route_id for route in canonical_routes)
  if len(canonical_routes) < 2:
    raise ValueError("at least two whole routes are required")
  if len(set(route_ids)) != len(route_ids):
    raise ValueError("route IDs must be unique")
  source_hashes = {route.recorded_source.sha256 for route in canonical_routes}
  if len(source_hashes) != 1:
    raise ValueError("behavior route population mixes exact source identities")
  if spec.validation_route_count is not None:
    validation_count = spec.validation_route_count
  else:
    assert spec.validation_fraction is not None
    validation_count = math.ceil(len(canonical_routes) * spec.validation_fraction)
  if not 0 < validation_count < len(canonical_routes):
    raise ValueError("route partition must retain training and validation routes")
  ranked = tuple(sorted(
    canonical_routes,
    key=lambda route: (
      hashlib.sha256(
        f"{spec.seed_identity_sha256}:{route.route_id}".encode(),
      ).hexdigest(),
      route.route_id,
    ),
  ))
  validation_ids = {route.route_id for route in ranked[:validation_count]}
  training = tuple(route for route in canonical_routes if route.route_id not in validation_ids)
  validation = tuple(route for route in canonical_routes if route.route_id in validation_ids)
  return RoutePartition(
    seed_identity_sha256=spec.seed_identity_sha256,
    recorded_source_identity_sha256=next(iter(source_hashes)),
    training_routes=training,
    validation_routes=validation,
  )


class ReplayRole(StrEnum):
  EXACT_STOCK = "exact_stock"
  CURRENTLY_ACCEPTED = "currently_accepted"
  CANDIDATE = "candidate"


@dataclass(frozen=True, slots=True)
class ReplayCoreIdentity:
  """Declared numerical-core contract and source commits for replay.

  This value is hash identity, not execution authority.  Production replay
  validates it against a reviewed implementation contract and fixed adapter.
  """

  controller_name: str
  core_artifact_sha256: str
  source_openpilot_commit: str
  opendbc_commit: str
  panda_commit: str

  @classmethod
  def compose(
    cls,
    *,
    controller_name: str,
    implementation_contract: str,
    replay_input_schema_version: int,
    source_openpilot_commit: str,
    opendbc_commit: str,
    panda_commit: str,
  ) -> ReplayCoreIdentity:
    """Bind reviewed implementation semantics to exact source commits."""
    if not controller_name.strip() or not implementation_contract.strip():
      raise ValueError("replay core implementation identity is empty")
    if replay_input_schema_version <= 0:
      raise ValueError("replay input schema version must be positive")
    core_sha256 = _sha256_json({
      "behaviorReplayInputSchemaVersion": replay_input_schema_version,
      "controllerName": controller_name,
      "implementationContract": implementation_contract,
      "opendbcCommit": opendbc_commit,
      "pandaCommit": panda_commit,
      "sourceOpenpilotCommit": source_openpilot_commit,
    })
    return cls(
      controller_name=controller_name,
      core_artifact_sha256=core_sha256,
      source_openpilot_commit=source_openpilot_commit,
      opendbc_commit=opendbc_commit,
      panda_commit=panda_commit,
    )

  def __post_init__(self) -> None:
    if not self.controller_name.strip():
      raise ValueError("controller_name must not be empty")
    if _SHA256_RE.fullmatch(self.core_artifact_sha256) is None:
      raise ValueError("core artifact identity must be lowercase SHA-256")
    for name, value in (
      ("source_openpilot_commit", self.source_openpilot_commit),
      ("opendbc_commit", self.opendbc_commit),
      ("panda_commit", self.panda_commit),
    ):
      if _COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a full lowercase commit")

  def to_dict(self) -> dict[str, Any]:
    return {
      "controllerName": self.controller_name,
      "coreArtifactSha256": self.core_artifact_sha256,
      "opendbcCommit": self.opendbc_commit,
      "pandaCommit": self.panda_commit,
      "sourceOpenpilotCommit": self.source_openpilot_commit,
    }


@dataclass(frozen=True, slots=True)
class ReplayArtifactIdentity:
  role: ReplayRole
  core: ReplayCoreIdentity
  behavior_policy_sha256: str | None
  composed_controller_artifact_sha256: str

  @classmethod
  def compose(
    cls,
    role: ReplayRole,
    core: ReplayCoreIdentity,
    policy: BehaviorPolicy | None,
  ) -> ReplayArtifactIdentity:
    policy_sha = None if policy is None else policy.sha256
    composed = _sha256_json({
      "behaviorPolicySha256": policy_sha,
      "core": core.to_dict(),
    })
    return cls(
      role=role,
      core=core,
      behavior_policy_sha256=policy_sha,
      composed_controller_artifact_sha256=composed,
    )

  def to_dict(self) -> dict[str, Any]:
    return {
      "behaviorPolicySha256": self.behavior_policy_sha256,
      "composedControllerArtifactSha256": self.composed_controller_artifact_sha256,
      "core": self.core.to_dict(),
      "role": self.role.value,
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())


ReplayEvaluationCallback = Callable[
  [ReplayArtifactIdentity, BehaviorPolicy | None, tuple[str, ...]],
  PolicyEvaluation,
]


def _behavior_selection_hash(
  *,
  final_policy_sha256: str,
  gate_spec_sha256: str,
  recorded_source_identity_sha256: str,
  route_partition_sha256: str,
  training_selection_sha256: str,
  validation_sha256: str,
) -> str:
  return _sha256_json({
    "finalBehaviorPolicySha256": final_policy_sha256,
    "gateSpecSha256": gate_spec_sha256,
    "recordedSourceIdentitySha256": recorded_source_identity_sha256,
    "routePartitionSha256": route_partition_sha256,
    "trainingSelectionSha256": training_selection_sha256,
    "validationSha256": validation_sha256,
  })


class FinalizationReason(StrEnum):
  PASSED = "passed"
  INSUFFICIENT_ROUTES = "insufficient_routes"
  MIXED_ROUTE_SOURCE_IDENTITIES = "mixed_route_source_identities"
  ROUTE_PARTITION_INVALID = "route_partition_invalid"
  BOOTSTRAP_IDENTITY_INVALID = "bootstrap_identity_invalid"
  CANDIDATE_GRID_INVALID = "candidate_grid_invalid"
  EVALUATION_CALLBACK_FAILED = "evaluation_callback_failed"
  EVALUATION_IDENTITY_MISMATCH = "evaluation_identity_mismatch"
  EVALUATION_PARTITION_MISMATCH = "evaluation_partition_mismatch"
  EVALUATION_POLICY_MISMATCH = "evaluation_policy_mismatch"
  EVALUATION_METRICS_INVALID = "evaluation_metrics_invalid"
  UNDEFINED_TRAINING_METRIC = "undefined_training_metric"
  NO_TRAINING_WINNER = "no_training_winner"
  UNDEFINED_VALIDATION_METRIC = "undefined_validation_metric"
  SMOOTH_CROSS_FIT_REGRESSION = "smooth_cross_fit_regression"
  SWIFT_CROSS_FIT_REGRESSION = "swift_cross_fit_regression"
  STRONG_CROSS_FIT_REGRESSION = "strong_cross_fit_regression"
  TARGET_VALIDATION_NOT_MATERIAL = "target_validation_not_material"


@dataclass(frozen=True, slots=True)
class BehaviorLearningFinalization:
  schema_version: int
  gate_spec_sha256: str
  route_partition_sha256: str | None
  recorded_source_identity_sha256: str | None
  training_selection_sha256: str | None
  validation_sha256: str | None
  smooth_passed: bool
  swift_passed: bool
  strong_passed: bool
  target_materially_improved: bool
  final_behavior_policy: BehaviorPolicy | None
  final_behavior_policy_sha256: str | None
  behavior_selection_sha256: str | None
  reasons: tuple[FinalizationReason, ...]

  def __post_init__(self) -> None:
    if self.schema_version != BEHAVIOR_FINALIZATION_SCHEMA_VERSION:
      raise ValueError("behavior finalization schema is incompatible")
    passed = (
      self.smooth_passed
      and self.swift_passed
      and self.strong_passed
      and self.target_materially_improved
    )
    emitted = (
      self.final_behavior_policy is not None
      and self.final_behavior_policy_sha256 is not None
      and self.behavior_selection_sha256 is not None
    )
    if passed != emitted:
      raise ValueError("final behavior policy exists if and only if every gate passes")
    if self.final_behavior_policy is not None and (
      self.final_behavior_policy.sha256 != self.final_behavior_policy_sha256
    ):
      raise ValueError("final behavior policy hash mismatch")
    for name, value in (
      ("gate_spec_sha256", self.gate_spec_sha256),
      ("route_partition_sha256", self.route_partition_sha256),
      ("recorded_source_identity_sha256", self.recorded_source_identity_sha256),
      ("training_selection_sha256", self.training_selection_sha256),
      ("validation_sha256", self.validation_sha256),
      ("final_behavior_policy_sha256", self.final_behavior_policy_sha256),
      ("behavior_selection_sha256", self.behavior_selection_sha256),
    ):
      if value is not None and _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256 when present")
    if passed:
      required = (
        self.route_partition_sha256,
        self.recorded_source_identity_sha256,
        self.training_selection_sha256,
        self.validation_sha256,
      )
      if any(value is None for value in required):
        raise ValueError("successful finalization lacks selection provenance")
      assert self.final_behavior_policy_sha256 is not None
      expected_selection = _behavior_selection_hash(
        final_policy_sha256=self.final_behavior_policy_sha256,
        gate_spec_sha256=self.gate_spec_sha256,
        recorded_source_identity_sha256=self.recorded_source_identity_sha256,
        route_partition_sha256=self.route_partition_sha256,
        training_selection_sha256=self.training_selection_sha256,
        validation_sha256=self.validation_sha256,
      )
      if self.behavior_selection_sha256 != expected_selection:
        raise ValueError("behavior selection hash mismatch")
    if passed and self.reasons != (FinalizationReason.PASSED,):
      raise ValueError("successful finalization must have only the passed reason")
    if not passed and (not self.reasons or FinalizationReason.PASSED in self.reasons):
      raise ValueError("failed finalization needs explicit failure reasons")

  @property
  def passed(self) -> bool:
    return self.final_behavior_policy is not None

  def to_dict(self) -> dict[str, Any]:
    return {
      "behaviorSelectionSha256": self.behavior_selection_sha256,
      "finalBehaviorPolicy": (
        None if self.final_behavior_policy is None else self.final_behavior_policy.to_dict()
      ),
      "finalBehaviorPolicySha256": self.final_behavior_policy_sha256,
      "gateSpecSha256": self.gate_spec_sha256,
      "recordedSourceIdentitySha256": self.recorded_source_identity_sha256,
      "reasons": [reason.value for reason in self.reasons],
      "routePartitionSha256": self.route_partition_sha256,
      "schemaVersion": self.schema_version,
      "smoothPassed": self.smooth_passed,
      "strongPassed": self.strong_passed,
      "swiftPassed": self.swift_passed,
      "targetMateriallyImproved": self.target_materially_improved,
      "trainingSelectionSha256": self.training_selection_sha256,
      "validationSha256": self.validation_sha256,
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())

  @property
  def sha256(self) -> str:
    return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

  @classmethod
  def from_json(cls, encoded: str) -> BehaviorLearningFinalization:
    try:
      payload = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
      raise ValueError("behavior finalization is invalid JSON") from exc
    root = _strict_object(payload, frozenset((
      "behaviorSelectionSha256",
      "finalBehaviorPolicy",
      "finalBehaviorPolicySha256",
      "gateSpecSha256",
      "recordedSourceIdentitySha256",
      "reasons",
      "routePartitionSha256",
      "schemaVersion",
      "smoothPassed",
      "strongPassed",
      "swiftPassed",
      "targetMateriallyImproved",
      "trainingSelectionSha256",
      "validationSha256",
    )), "behavior finalization")
    if type(root["schemaVersion"]) is not int:
      raise ValueError("behavior finalization schema version must be integer")
    for key in (
      "smoothPassed",
      "strongPassed",
      "swiftPassed",
      "targetMateriallyImproved",
    ):
      if type(root[key]) is not bool:
        raise ValueError(f"{key} must be boolean")
    reasons_payload = root["reasons"]
    if type(reasons_payload) is not list or any(
      type(value) is not str for value in reasons_payload
    ):
      raise ValueError("finalization reasons must be a text array")
    try:
      reasons = tuple(FinalizationReason(value) for value in reasons_payload)
    except ValueError as exc:
      raise ValueError("finalization contains an unknown reason") from exc
    policy_payload = root["finalBehaviorPolicy"]
    final_policy = None
    if policy_payload is not None:
      policy = _strict_object(policy_payload, frozenset((
        "dampingRatio",
        "naturalFrequencyPerS",
      )), "final behavior policy")
      final_policy = BehaviorPolicy(
        natural_frequency_per_s=_strict_number(
          policy["naturalFrequencyPerS"],
          "naturalFrequencyPerS",
        ),
        damping_ratio=_strict_number(policy["dampingRatio"], "dampingRatio"),
      )

    def optional_sha(key: str) -> str | None:
      value = root[key]
      if value is not None and type(value) is not str:
        raise ValueError(f"{key} must be text or null")
      return value

    if type(root["gateSpecSha256"]) is not str:
      raise ValueError("gateSpecSha256 must be text")
    return cls(
      schema_version=root["schemaVersion"],
      gate_spec_sha256=root["gateSpecSha256"],
      route_partition_sha256=optional_sha("routePartitionSha256"),
      recorded_source_identity_sha256=optional_sha(
        "recordedSourceIdentitySha256",
      ),
      training_selection_sha256=optional_sha("trainingSelectionSha256"),
      validation_sha256=optional_sha("validationSha256"),
      smooth_passed=root["smoothPassed"],
      swift_passed=root["swiftPassed"],
      strong_passed=root["strongPassed"],
      target_materially_improved=root["targetMateriallyImproved"],
      final_behavior_policy=final_policy,
      final_behavior_policy_sha256=optional_sha("finalBehaviorPolicySha256"),
      behavior_selection_sha256=optional_sha("behaviorSelectionSha256"),
      reasons=reasons,
    )


class _EvaluationInputError(ValueError):
  def __init__(self, reason: FinalizationReason):
    super().__init__(reason.value)
    self.reason = reason


def _evaluate_checked(
  callback: ReplayEvaluationCallback,
  identity: ReplayArtifactIdentity,
  policy: BehaviorPolicy | None,
  route_ids: tuple[str, ...],
) -> PolicyEvaluation:
  try:
    evaluation = callback(identity, policy, route_ids)
  except Exception as exc:
    raise _EvaluationInputError(
      FinalizationReason.EVALUATION_CALLBACK_FAILED,
    ) from exc
  if not isinstance(evaluation, PolicyEvaluation):
    raise _EvaluationInputError(FinalizationReason.EVALUATION_CALLBACK_FAILED)
  if evaluation.artifact_identity != identity.to_json():
    raise _EvaluationInputError(FinalizationReason.EVALUATION_IDENTITY_MISMATCH)
  if evaluation.route_ids != route_ids:
    raise _EvaluationInputError(FinalizationReason.EVALUATION_PARTITION_MISMATCH)
  if evaluation.policy != policy:
    raise _EvaluationInputError(FinalizationReason.EVALUATION_POLICY_MISMATCH)
  return evaluation


def _undefined_required_metric_exists(
  evaluations: Iterable[PolicyEvaluation],
  rules: tuple[MetricGateRule, ...],
) -> bool:
  """Return whether a mandatory gate metric is undefined.

  Replay may carry additional informational metrics whose physical phase is
  absent on a route partition (for example, hold bias on a drive with no hold
  phase).  Those remain explicitly undefined in the report but cannot veto an
  otherwise complete gate that never named them.
  """
  required_names = {rule.metric_name for rule in rules}
  return any(
    metric.name in required_names and not metric.defined
    for evaluation in evaluations
    for metric in evaluation.metrics
  )


def _failed_finalization(
  gate_spec: BehaviorGateSpec,
  reason: FinalizationReason,
  partition: RoutePartition | None = None,
  selection: TrainingSelection | None = None,
  validation: HeldOutValidation | None = None,
  smooth_passed: bool = False,
  swift_passed: bool = False,
  strong_passed: bool = False,
  target_materially_improved: bool = False,
) -> BehaviorLearningFinalization:
  validation_sha = (
    None
    if validation is None
    else hashlib.sha256(validation.to_json().encode("utf-8")).hexdigest()
  )
  return BehaviorLearningFinalization(
    schema_version=BEHAVIOR_FINALIZATION_SCHEMA_VERSION,
    gate_spec_sha256=gate_spec.sha256,
    route_partition_sha256=None if partition is None else partition.sha256,
    recorded_source_identity_sha256=(
      None if partition is None else partition.recorded_source_identity_sha256
    ),
    training_selection_sha256=None if selection is None else selection.sha256,
    validation_sha256=validation_sha,
    smooth_passed=smooth_passed,
    swift_passed=swift_passed,
    strong_passed=strong_passed,
    target_materially_improved=target_materially_improved,
    final_behavior_policy=None,
    final_behavior_policy_sha256=None,
    behavior_selection_sha256=None,
    reasons=(reason,),
  )


def _contract_passes(verdict: CandidateGateVerdict) -> dict[BehaviorContract, bool]:
  return {contract.contract: contract.passed for contract in verdict.contracts}


def finalize_behavior_learning(
  gate_spec: BehaviorGateSpec,
  routes: Iterable[BehaviorRouteEvidenceIdentity],
  accepted_policy: BehaviorPolicy | None,
  search_center_policy: BehaviorPolicy,
  exact_stock_core: ReplayCoreIdentity,
  accepted_core: ReplayCoreIdentity | None,
  candidate_core: ReplayCoreIdentity,
  replay_evaluate: ReplayEvaluationCallback,
) -> BehaviorLearningFinalization:
  """Select on whole training routes, freeze once, then validate once.

  This is a pure selection primitive, not replay authority: its callback is
  intentionally caller-owned.  Production reaches it only after
  ``run_behavior_learning_transaction`` has admitted the reviewed fixed replay
  adapters, or through the compact aggregate path whose route evaluations
  validate those adapters directly.  Calling this selector with synthetic
  evaluations is useful in tests but cannot establish an exact-stock row.

  Before the first modular artifact is approved, ``accepted_policy`` and
  ``accepted_core`` are both ``None``.  Exact stock then supplies both
  reference roles while ``search_center_policy`` remains an explicit search
  seed only.  Once a modular artifact exists, search is always centered on
  that accepted policy.
  """
  try:
    partition = partition_whole_routes(routes, gate_spec.route_partition)
  except ValueError as exc:
    if "source identities" in str(exc):
      reason = FinalizationReason.MIXED_ROUTE_SOURCE_IDENTITIES
    elif "at least two" in str(exc):
      reason = FinalizationReason.INSUFFICIENT_ROUTES
    else:
      reason = FinalizationReason.ROUTE_PARTITION_INVALID
    return _failed_finalization(gate_spec, reason)

  if not isinstance(search_center_policy, BehaviorPolicy):
    return _failed_finalization(
      gate_spec,
      FinalizationReason.CANDIDATE_GRID_INVALID,
      partition=partition,
    )
  if accepted_policy is None:
    if accepted_core is not None:
      return _failed_finalization(
        gate_spec,
        FinalizationReason.BOOTSTRAP_IDENTITY_INVALID,
        partition=partition,
      )
    effective_accepted_core = exact_stock_core
  else:
    if accepted_core is None or search_center_policy != accepted_policy:
      return _failed_finalization(
        gate_spec,
        FinalizationReason.BOOTSTRAP_IDENTITY_INVALID,
        partition=partition,
      )
    effective_accepted_core = accepted_core

  try:
    grid = build_candidate_grid(
      gate_spec.candidate_grid.policy_grid(search_center_policy),
    )
  except ValueError:
    return _failed_finalization(
      gate_spec,
      FinalizationReason.CANDIDATE_GRID_INVALID,
      partition=partition,
    )
  training_ids = partition.training_route_ids
  stock_identity = ReplayArtifactIdentity.compose(ReplayRole.EXACT_STOCK, exact_stock_core, None)
  accepted_identity = ReplayArtifactIdentity.compose(
    ReplayRole.CURRENTLY_ACCEPTED,
    effective_accepted_core,
    accepted_policy,
  )
  try:
    stock_training = _evaluate_checked(replay_evaluate, stock_identity, None, training_ids)
    accepted_training = _evaluate_checked(
      replay_evaluate,
      accepted_identity,
      accepted_policy,
      training_ids,
    )
    candidate_training = tuple(
      _evaluate_checked(
        replay_evaluate,
        ReplayArtifactIdentity.compose(ReplayRole.CANDIDATE, candidate_core, candidate.policy),
        candidate.policy,
        training_ids,
      )
      for candidate in grid
    )
  except _EvaluationInputError as exc:
    return _failed_finalization(gate_spec, exc.reason, partition=partition)

  all_training = (stock_training, accepted_training, *candidate_training)
  try:
    selection = select_training_winner(
      grid,
      candidate_training,
      stock_training,
      accepted_training,
      gate_spec.metric_rules,
      gate_spec.target_metric_name,
      gate_spec.paired_uncertainty_method,
      gate_spec.minimum_paired_route_count,
    )
  except ValueError:
    return _failed_finalization(
      gate_spec,
      FinalizationReason.EVALUATION_METRICS_INVALID,
      partition=partition,
    )
  if selection is None:
    reason = (
      FinalizationReason.UNDEFINED_TRAINING_METRIC
      if _undefined_required_metric_exists(
        all_training,
        gate_spec.metric_rules,
      )
      else FinalizationReason.NO_TRAINING_WINNER
    )
    return _failed_finalization(gate_spec, reason, partition=partition)

  validation_ids = partition.validation_route_ids
  winner_identity = ReplayArtifactIdentity.compose(
    ReplayRole.CANDIDATE,
    candidate_core,
    selection.winner.policy,
  )
  try:
    stock_validation = _evaluate_checked(replay_evaluate, stock_identity, None, validation_ids)
    accepted_validation = _evaluate_checked(
      replay_evaluate,
      accepted_identity,
      accepted_policy,
      validation_ids,
    )
    winner_validation = _evaluate_checked(
      replay_evaluate,
      winner_identity,
      selection.winner.policy,
      validation_ids,
    )
  except _EvaluationInputError as exc:
    return _failed_finalization(
      gate_spec,
      exc.reason,
      partition=partition,
      selection=selection,
    )
  if _undefined_required_metric_exists(
    (stock_validation, accepted_validation, winner_validation),
    gate_spec.metric_rules,
  ):
    return _failed_finalization(
      gate_spec,
      FinalizationReason.UNDEFINED_VALIDATION_METRIC,
      partition=partition,
      selection=selection,
    )

  try:
    validation = validate_frozen_winner(
      selection,
      winner_validation,
      stock_validation,
      accepted_validation,
      gate_spec.metric_rules,
      gate_spec.target_metric_name,
      gate_spec.paired_uncertainty_method,
      gate_spec.minimum_paired_route_count,
    )
  except ValueError:
    return _failed_finalization(
      gate_spec,
      FinalizationReason.EVALUATION_METRICS_INVALID,
      partition=partition,
      selection=selection,
    )
  contract_passes = _contract_passes(validation.frozen_winner_verdict)
  validation_sha = hashlib.sha256(validation.to_json().encode("utf-8")).hexdigest()
  smooth_passed = contract_passes[BehaviorContract.SMOOTH]
  swift_passed = contract_passes[BehaviorContract.SWIFT]
  strong_passed = contract_passes[BehaviorContract.STRONG]
  target_passed = validation.frozen_winner_verdict.target_materially_improved
  if not validation.accepted:
    if not smooth_passed:
      reason = FinalizationReason.SMOOTH_CROSS_FIT_REGRESSION
    elif not swift_passed:
      reason = FinalizationReason.SWIFT_CROSS_FIT_REGRESSION
    elif not strong_passed:
      reason = FinalizationReason.STRONG_CROSS_FIT_REGRESSION
    else:
      reason = FinalizationReason.TARGET_VALIDATION_NOT_MATERIAL
    return _failed_finalization(
      gate_spec,
      reason,
      partition=partition,
      selection=selection,
      validation=validation,
      smooth_passed=smooth_passed,
      swift_passed=swift_passed,
      strong_passed=strong_passed,
      target_materially_improved=target_passed,
    )

  final_policy = selection.winner.policy
  behavior_selection_sha256 = _behavior_selection_hash(
    final_policy_sha256=final_policy.sha256,
    gate_spec_sha256=gate_spec.sha256,
    recorded_source_identity_sha256=partition.recorded_source_identity_sha256,
    route_partition_sha256=partition.sha256,
    training_selection_sha256=selection.sha256,
    validation_sha256=validation_sha,
  )
  return BehaviorLearningFinalization(
    schema_version=BEHAVIOR_FINALIZATION_SCHEMA_VERSION,
    gate_spec_sha256=gate_spec.sha256,
    route_partition_sha256=partition.sha256,
    recorded_source_identity_sha256=partition.recorded_source_identity_sha256,
    training_selection_sha256=selection.sha256,
    validation_sha256=validation_sha,
    smooth_passed=smooth_passed,
    swift_passed=swift_passed,
    strong_passed=strong_passed,
    target_materially_improved=target_passed,
    final_behavior_policy=final_policy,
    final_behavior_policy_sha256=final_policy.sha256,
    behavior_selection_sha256=behavior_selection_sha256,
    reasons=(FinalizationReason.PASSED,),
  )

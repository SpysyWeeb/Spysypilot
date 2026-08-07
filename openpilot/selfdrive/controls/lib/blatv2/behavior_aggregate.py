"""Compact cross-route aggregation for the offline BLaTv2 trainer.

One-route replay owns authenticated evidence, exact controller execution, and
frozen physical windows.  This module owns the next boundary: it binds an
ordered scenario population and immutable whole-route partition, verifies
that compact route results describe one homogeneous experiment, delegates all
metric reduction to :func:`aggregate_behavior_metrics`, and delegates all
candidate decisions to the shared behavior-policy selector.

Recorded controller requests and recorded controller identities remain route
provenance only.  They are deliberately allowed to differ across scenarios;
counterfactual stock, incumbent, and candidate artifacts are the homogeneous
objects being compared.

This module is a strict deterministic reducer, not a provenance authority.
Its compact dataclasses can prove internal consistency, but Python callers can
construct self-consistent rows.  A publishable trainer must replay reviewed
route evidence and aggregate it inside one reviewed execution boundary; it
must never accept a caller-supplied aggregate as proof that a controller ran.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import math

from openpilot.selfdrive.controls.lib.blatv2.behavior_coordinator import (
  BehaviorGateSpec,
  ReplayArtifactIdentity,
  ReplayCoreIdentity,
  ReplayRole,
  RoutePartitionSpec,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import (
  BehaviorScenarioSetIdentity,
  canonical_json,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_metrics import (
  BehaviorMetricConfig,
  BehaviorMetricName,
  BehaviorScorecard,
  WindowMetricSet,
  aggregate_behavior_metrics,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_policy import (
  BehaviorPolicy,
  HeldOutValidation,
  PolicyEvaluation,
  PolicyMetric,
  TrainingSelection,
  build_candidate_grid,
  select_training_winner,
  validate_frozen_winner,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_replay import (
  BehaviorReplayError,
  validate_reviewed_replay_core_identity,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_route_evaluator import (
  BEHAVIOR_ROUTE_PREPARATION_SCHEMA_VERSION,
  BehaviorRouteEvaluation,
)


BEHAVIOR_AGGREGATE_SPEC_SCHEMA_VERSION = 1
BEHAVIOR_AGGREGATE_ARTIFACT_SCHEMA_VERSION = 2
BEHAVIOR_AGGREGATE_EVALUATION_SCHEMA_VERSION = 1
BEHAVIOR_TRAINING_COMPARISON_SCHEMA_VERSION = 1
BEHAVIOR_AGGREGATE_SELECTION_SCHEMA_VERSION = 1
BEHAVIOR_METRIC_AGGREGATION_CONTRACT = "aggregate_behavior_metrics_v1"


class BehaviorAggregateError(ValueError):
  """Compact route results cannot form one authenticated experiment."""


def _sha256_json(value: object) -> str:
  return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _validate_sha256(value: str, name: str) -> None:
  if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
    raise BehaviorAggregateError(f"{name} must be lowercase SHA-256")


def _partition_spec_sha256(value: RoutePartitionSpec) -> str:
  return _sha256_json(value.to_dict())


class BehaviorRouteSplit(StrEnum):
  TRAINING = "training"
  VALIDATION = "validation"


@dataclass(frozen=True, slots=True)
class BehaviorRouteAssignment:
  route_id: str
  split: BehaviorRouteSplit

  def __post_init__(self) -> None:
    if not self.route_id.strip():
      raise BehaviorAggregateError("route assignment needs a route ID")
    if not isinstance(self.split, BehaviorRouteSplit):
      raise TypeError("route assignment split has the wrong type")

  def to_dict(self) -> dict[str, str]:
    return {"routeId": self.route_id, "split": self.split.value}


@dataclass(frozen=True, slots=True)
class FrozenBehaviorRoutePartition:
  """Scenario-ordered, whole-route train/validation assignment."""

  partition_spec_sha256: str
  assignments: tuple[BehaviorRouteAssignment, ...]

  def __post_init__(self) -> None:
    _validate_sha256(self.partition_spec_sha256, "partition specification identity")
    if type(self.assignments) is not tuple or not self.assignments:
      raise BehaviorAggregateError("frozen route partition must not be empty")
    if any(not isinstance(value, BehaviorRouteAssignment) for value in self.assignments):
      raise TypeError("frozen route assignments have the wrong type")
    route_ids = self.route_ids
    if len(set(route_ids)) != len(route_ids):
      raise BehaviorAggregateError("a route may appear only once in a frozen partition")
    if not self.training_route_ids:
      raise BehaviorAggregateError("frozen route partition has no training routes")
    if not self.validation_route_ids:
      raise BehaviorAggregateError("publishable comparison requires validation routes")

  @property
  def route_ids(self) -> tuple[str, ...]:
    return tuple(assignment.route_id for assignment in self.assignments)

  @property
  def training_route_ids(self) -> tuple[str, ...]:
    return tuple(
      assignment.route_id
      for assignment in self.assignments
      if assignment.split is BehaviorRouteSplit.TRAINING
    )

  @property
  def validation_route_ids(self) -> tuple[str, ...]:
    return tuple(
      assignment.route_id
      for assignment in self.assignments
      if assignment.split is BehaviorRouteSplit.VALIDATION
    )

  def route_ids_for(self, split: BehaviorRouteSplit) -> tuple[str, ...]:
    if not isinstance(split, BehaviorRouteSplit):
      raise TypeError("route split has the wrong type")
    return tuple(
      assignment.route_id
      for assignment in self.assignments
      if assignment.split is split
    )

  def to_dict(self) -> dict[str, object]:
    return {
      "assignments": [assignment.to_dict() for assignment in self.assignments],
      "partitionSpecSha256": self.partition_spec_sha256,
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())

  @property
  def sha256(self) -> str:
    return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def freeze_behavior_route_partition(
  scenarios: BehaviorScenarioSetIdentity,
  spec: RoutePartitionSpec,
) -> FrozenBehaviorRoutePartition:
  """Apply the committed hash ranking without collapsing recorded sources."""
  if not isinstance(scenarios, BehaviorScenarioSetIdentity):
    raise TypeError("route partition requires an ordered scenario set")
  if not isinstance(spec, RoutePartitionSpec):
    raise TypeError("route partition requires a partition specification")
  route_ids = tuple(source.route_id for source in scenarios.sources)
  if len(route_ids) < 2:
    raise BehaviorAggregateError("at least two whole routes are required")
  if spec.validation_route_count is not None:
    validation_count = spec.validation_route_count
  else:
    assert spec.validation_fraction is not None
    validation_count = math.ceil(len(route_ids) * spec.validation_fraction)
  if not 0 < validation_count < len(route_ids):
    raise BehaviorAggregateError(
      "route partition must retain training and validation routes",
    )
  ranked = tuple(sorted(
    route_ids,
    key=lambda route_id: (
      hashlib.sha256(f"{spec.seed_identity_sha256}:{route_id}".encode()).hexdigest(),
      route_id,
    ),
  ))
  validation_ids = frozenset(ranked[:validation_count])
  return FrozenBehaviorRoutePartition(
    partition_spec_sha256=_partition_spec_sha256(spec),
    assignments=tuple(
      BehaviorRouteAssignment(
        route_id,
        (
          BehaviorRouteSplit.VALIDATION
          if route_id in validation_ids
          else BehaviorRouteSplit.TRAINING
        ),
      )
      for route_id in route_ids
    ),
  )


@dataclass(frozen=True, slots=True)
class BehaviorAggregateSpec:
  """Complete typed authority for one compact trainer experiment."""

  schema_version: int
  scenarios: BehaviorScenarioSetIdentity
  partition: FrozenBehaviorRoutePartition
  gate_spec: BehaviorGateSpec
  exact_stock_core: ReplayCoreIdentity
  modular_core: ReplayCoreIdentity
  physical_profile_sha256: str
  provisional_dynamics_sha256: str
  preparation_schema_version: int
  segmentation_config_sha256: str
  aggregation_contract: str = BEHAVIOR_METRIC_AGGREGATION_CONTRACT

  def __post_init__(self) -> None:
    if self.schema_version != BEHAVIOR_AGGREGATE_SPEC_SCHEMA_VERSION:
      raise BehaviorAggregateError("behavior aggregate specification is incompatible")
    if not isinstance(self.scenarios, BehaviorScenarioSetIdentity):
      raise TypeError("behavior aggregate specification requires scenarios")
    if not isinstance(self.partition, FrozenBehaviorRoutePartition):
      raise TypeError("behavior aggregate specification requires a frozen partition")
    if not isinstance(self.gate_spec, BehaviorGateSpec):
      raise TypeError("behavior aggregate specification requires gate authority")
    if not isinstance(self.exact_stock_core, ReplayCoreIdentity):
      raise TypeError("behavior aggregate specification requires exact-stock core identity")
    if not isinstance(self.modular_core, ReplayCoreIdentity):
      raise TypeError("behavior aggregate specification requires modular core identity")
    if self.preparation_schema_version != BEHAVIOR_ROUTE_PREPARATION_SCHEMA_VERSION:
      raise BehaviorAggregateError("aggregate route preparation contract is incompatible")
    try:
      validate_reviewed_replay_core_identity(self.exact_stock_core, exact_stock=True)
      validate_reviewed_replay_core_identity(self.modular_core, exact_stock=False)
    except (BehaviorReplayError, TypeError, ValueError) as exc:
      raise BehaviorAggregateError(
        "replay core differs from reviewed implementation contract",
      ) from exc
    stock_platform = (
      self.exact_stock_core.source_openpilot_commit,
      self.exact_stock_core.opendbc_commit,
      self.exact_stock_core.panda_commit,
    )
    modular_platform = (
      self.modular_core.source_openpilot_commit,
      self.modular_core.opendbc_commit,
      self.modular_core.panda_commit,
    )
    if stock_platform != modular_platform:
      raise BehaviorAggregateError("reviewed replay cores use different platform commits")
    for name, value in (
      ("physical profile", self.physical_profile_sha256),
      ("provisional dynamics", self.provisional_dynamics_sha256),
      ("segmentation configuration", self.segmentation_config_sha256),
    ):
      _validate_sha256(value, name)
    if self.aggregation_contract != BEHAVIOR_METRIC_AGGREGATION_CONTRACT:
      raise BehaviorAggregateError("behavior metric aggregation contract is unsupported")
    if self.partition.route_ids != tuple(source.route_id for source in self.scenarios.sources):
      raise BehaviorAggregateError("partition order differs from ordered scenario set")
    if len({source.vehicle_identity for source in self.scenarios.sources}) != 1:
      raise BehaviorAggregateError("behavior population mixes vehicle identities")
    expected = freeze_behavior_route_partition(
      self.scenarios,
      self.gate_spec.route_partition,
    )
    if self.partition != expected:
      raise BehaviorAggregateError("frozen route partition differs from gate authority")

  @classmethod
  def freeze(
    cls,
    scenarios: BehaviorScenarioSetIdentity,
    gate_spec: BehaviorGateSpec,
    exact_stock_core: ReplayCoreIdentity,
    modular_core: ReplayCoreIdentity,
    physical_profile_sha256: str,
    provisional_dynamics_sha256: str,
    segmentation_config_sha256: str,
  ) -> BehaviorAggregateSpec:
    return cls(
      schema_version=BEHAVIOR_AGGREGATE_SPEC_SCHEMA_VERSION,
      scenarios=scenarios,
      partition=freeze_behavior_route_partition(
        scenarios,
        gate_spec.route_partition,
      ),
      gate_spec=gate_spec,
      exact_stock_core=exact_stock_core,
      modular_core=modular_core,
      physical_profile_sha256=physical_profile_sha256,
      provisional_dynamics_sha256=provisional_dynamics_sha256,
      preparation_schema_version=BEHAVIOR_ROUTE_PREPARATION_SCHEMA_VERSION,
      segmentation_config_sha256=segmentation_config_sha256,
    )

  def to_dict(self) -> dict[str, object]:
    return {
      "aggregationContract": self.aggregation_contract,
      "exactStockCore": self.exact_stock_core.to_dict(),
      "gateSpec": self.gate_spec.to_dict(),
      "modularCore": self.modular_core.to_dict(),
      "partition": self.partition.to_dict(),
      "physicalProfileSha256": self.physical_profile_sha256,
      "preparationSchemaVersion": self.preparation_schema_version,
      "provisionalDynamicsSha256": self.provisional_dynamics_sha256,
      "scenarios": self.scenarios.to_dict(),
      "schemaVersion": self.schema_version,
      "segmentationConfigSha256": self.segmentation_config_sha256,
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())

  @property
  def sha256(self) -> str:
    return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class BehaviorAggregateRouteIdentity:
  route_id: str
  scenario_sha256: str
  preparation_sha256: str
  route_evaluation_sha256: str
  window_contract_sha256: str

  def __post_init__(self) -> None:
    if not self.route_id.strip():
      raise BehaviorAggregateError("aggregate route identity needs a route ID")
    for name, value in (
      ("scenario", self.scenario_sha256),
      ("preparation", self.preparation_sha256),
      ("route evaluation", self.route_evaluation_sha256),
      ("window contract", self.window_contract_sha256),
    ):
      _validate_sha256(value, name)

  def to_dict(self) -> dict[str, str]:
    return {
      "preparationSha256": self.preparation_sha256,
      "routeEvaluationSha256": self.route_evaluation_sha256,
      "routeId": self.route_id,
      "scenarioSha256": self.scenario_sha256,
      "windowContractSha256": self.window_contract_sha256,
    }


@dataclass(frozen=True, slots=True)
class BehaviorAggregateArtifactIdentity:
  """Exact controller and experiment inputs behind one partition aggregate."""

  schema_version: int
  aggregate_spec_sha256: str
  split: BehaviorRouteSplit
  replay_artifact: ReplayArtifactIdentity
  physical_profile_sha256: str
  provisional_dynamics_sha256: str
  plant_member_id: str
  metric_config_sha256: str
  segmentation_config_sha256: str
  preparation_schema_version: int
  aggregation_contract: str
  ordered_routes: tuple[BehaviorAggregateRouteIdentity, ...]

  def __post_init__(self) -> None:
    if self.schema_version != BEHAVIOR_AGGREGATE_ARTIFACT_SCHEMA_VERSION:
      raise BehaviorAggregateError("aggregate artifact identity is incompatible")
    if not isinstance(self.split, BehaviorRouteSplit):
      raise TypeError("aggregate artifact split has the wrong type")
    if not isinstance(self.replay_artifact, ReplayArtifactIdentity):
      raise TypeError("aggregate artifact requires replay identity")
    if self.preparation_schema_version != BEHAVIOR_ROUTE_PREPARATION_SCHEMA_VERSION:
      raise BehaviorAggregateError("aggregate preparation contract is incompatible")
    if self.aggregation_contract != BEHAVIOR_METRIC_AGGREGATION_CONTRACT:
      raise BehaviorAggregateError("aggregate metric contract is incompatible")
    for name, value in (
      ("aggregate specification", self.aggregate_spec_sha256),
      ("physical profile", self.physical_profile_sha256),
      ("provisional dynamics", self.provisional_dynamics_sha256),
      ("plant member", self.plant_member_id),
      ("metric configuration", self.metric_config_sha256),
      ("segmentation configuration", self.segmentation_config_sha256),
    ):
      _validate_sha256(value, name)
    if type(self.ordered_routes) is not tuple or not self.ordered_routes:
      raise BehaviorAggregateError("aggregate artifact needs route identities")
    if any(not isinstance(value, BehaviorAggregateRouteIdentity) for value in self.ordered_routes):
      raise TypeError("aggregate route identities have the wrong type")
    route_ids = tuple(route.route_id for route in self.ordered_routes)
    if len(set(route_ids)) != len(route_ids):
      raise BehaviorAggregateError("aggregate artifact contains duplicate routes")

  @property
  def route_ids(self) -> tuple[str, ...]:
    return tuple(route.route_id for route in self.ordered_routes)

  def to_dict(self) -> dict[str, object]:
    return {
      "aggregateSpecSha256": self.aggregate_spec_sha256,
      "aggregationContract": self.aggregation_contract,
      "metricConfigSha256": self.metric_config_sha256,
      "orderedRoutes": [route.to_dict() for route in self.ordered_routes],
      "physicalProfileSha256": self.physical_profile_sha256,
      "plantMemberId": self.plant_member_id,
      "preparationSchemaVersion": self.preparation_schema_version,
      "provisionalDynamicsSha256": self.provisional_dynamics_sha256,
      "replayArtifact": self.replay_artifact.to_dict(),
      "schemaVersion": self.schema_version,
      "segmentationConfigSha256": self.segmentation_config_sha256,
      "split": self.split.value,
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())

  @property
  def sha256(self) -> str:
    return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class BehaviorAggregateEvaluation:
  """Compact windows, scorecard, and selector input for one opponent/split."""

  schema_version: int
  identity: BehaviorAggregateArtifactIdentity
  policy: BehaviorPolicy | None
  metric_config: BehaviorMetricConfig
  route_results: tuple[BehaviorRouteEvaluation, ...]
  scorecard: BehaviorScorecard
  policy_evaluation: PolicyEvaluation

  def __post_init__(self) -> None:
    if self.schema_version != BEHAVIOR_AGGREGATE_EVALUATION_SCHEMA_VERSION:
      raise BehaviorAggregateError("behavior aggregate evaluation is incompatible")
    if not isinstance(self.identity, BehaviorAggregateArtifactIdentity):
      raise TypeError("behavior aggregate evaluation requires artifact identity")
    if self.policy is not None and not isinstance(self.policy, BehaviorPolicy):
      raise TypeError("behavior aggregate policy has the wrong type")
    if not isinstance(self.metric_config, BehaviorMetricConfig):
      raise TypeError("behavior aggregate evaluation requires metric authority")
    if type(self.route_results) is not tuple or not self.route_results:
      raise BehaviorAggregateError("behavior aggregate evaluation needs route evidence")
    if any(not isinstance(value, BehaviorRouteEvaluation) for value in self.route_results):
      raise TypeError("behavior aggregate route evidence has the wrong type")
    expected_replay = ReplayArtifactIdentity.compose(
      self.identity.replay_artifact.role,
      self.identity.replay_artifact.core,
      self.policy,
    )
    if self.identity.replay_artifact != expected_replay:
      raise BehaviorAggregateError("aggregate replay role, core, and policy disagree")
    if self.metric_config.sha256 != self.identity.metric_config_sha256:
      raise BehaviorAggregateError("aggregate metric authority differs from identity")
    expected_route_identities: list[BehaviorAggregateRouteIdentity] = []
    expected_windows: list[WindowMetricSet] = []
    expected_metric_names = tuple(BehaviorMetricName)
    for result in self.route_results:
      if result.artifact_identity != self.identity.replay_artifact:
        raise BehaviorAggregateError("aggregate route evidence uses another replay artifact")
      if (
        result.physical_profile_sha256 != self.identity.physical_profile_sha256
        or result.provisional_dynamics_sha256 != self.identity.provisional_dynamics_sha256
        or result.plant_member_id != self.identity.plant_member_id
        or result.metric_config_sha256 != self.identity.metric_config_sha256
        or result.segmentation_config_sha256 != self.identity.segmentation_config_sha256
        or result.preparation_schema_version != self.identity.preparation_schema_version
      ):
        raise BehaviorAggregateError("aggregate route evidence uses another experiment")
      for window in result.windows:
        if window.source_identity_sha256 != result.scenario.recorded_source.sha256:
          raise BehaviorAggregateError("aggregate window source differs from route evidence")
        if tuple(metric.name for metric in window.metrics) != expected_metric_names:
          raise BehaviorAggregateError("aggregate route metric registry is non-canonical")
      expected_route_identities.append(BehaviorAggregateRouteIdentity(
        route_id=result.scenario.route_id,
        scenario_sha256=result.scenario.sha256,
        preparation_sha256=result.preparation_sha256,
        route_evaluation_sha256=result.sha256,
        window_contract_sha256=_window_contract_sha256(result.windows),
      ))
      expected_windows.extend(result.windows)
    if tuple(expected_route_identities) != self.identity.ordered_routes:
      raise BehaviorAggregateError("aggregate route evidence differs from identity")
    expected_scorecard = aggregate_behavior_metrics(expected_windows, self.metric_config)
    if self.scorecard != expected_scorecard:
      raise BehaviorAggregateError("aggregate scorecard differs from route evidence")
    if self.scorecard.metric_config_sha256 != self.identity.metric_config_sha256:
      raise BehaviorAggregateError("aggregate scorecard metric configuration differs")
    if self.policy_evaluation.artifact_identity != self.identity.replay_artifact.to_json():
      raise BehaviorAggregateError("selector artifact identity differs from aggregate")
    if self.policy_evaluation.policy != self.policy:
      raise BehaviorAggregateError("selector policy differs from aggregate")
    if self.policy_evaluation.route_ids != tuple(sorted(self.identity.route_ids)):
      raise BehaviorAggregateError("selector routes differ from aggregate routes")
    expected_metrics = tuple(
      PolicyMetric.from_scorecard(self.scorecard, name)
      for name in BehaviorMetricName
    )
    if self.policy_evaluation.metrics != expected_metrics:
      raise BehaviorAggregateError("selector metrics differ from aggregate scorecard")
    allowed_routes = frozenset(self.identity.route_ids)
    if any(window.route_id not in allowed_routes for window in self.scorecard.windows):
      raise BehaviorAggregateError("aggregate scorecard contains a foreign route")
    for route in self.identity.ordered_routes:
      route_windows = tuple(
        window
        for window in self.scorecard.windows
        if window.route_id == route.route_id
      )
      if _window_contract_sha256(route_windows) != route.window_contract_sha256:
        raise BehaviorAggregateError("aggregate scorecard window contract differs")

  def to_dict(self) -> dict[str, object]:
    return {
      "identity": self.identity.to_dict(),
      "metricConfigSha256": self.metric_config.sha256,
      "policy": None if self.policy is None else self.policy.to_dict(),
      "policyEvaluation": self.policy_evaluation.to_dict(),
      "schemaVersion": self.schema_version,
      "scorecard": self.scorecard.to_dict(),
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())

  @property
  def sha256(self) -> str:
    return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def _window_contract_sha256(windows: tuple[WindowMetricSet, ...]) -> str:
  """Bind controller-independent prepared-window geometry and provenance."""
  return _sha256_json([
    {
      "interventionMonoTimeNs": window.intervention_mono_time_ns,
      "maneuverClass": window.maneuver_class.value,
      "phase": window.phase.value,
      "routeId": window.route_id,
      "sourceIdentitySha256": window.source_identity_sha256,
      "windowId": window.window_id,
    }
    for window in windows
  ])


def _expected_scenarios(
  spec: BehaviorAggregateSpec,
  split: BehaviorRouteSplit,
) -> tuple[object, ...]:
  route_ids = frozenset(spec.partition.route_ids_for(split))
  return tuple(source for source in spec.scenarios.sources if source.route_id in route_ids)


def aggregate_behavior_route_results(
  spec: BehaviorAggregateSpec,
  split: BehaviorRouteSplit,
  route_results: Iterable[BehaviorRouteEvaluation],
  policy: BehaviorPolicy | None,
) -> BehaviorAggregateEvaluation:
  """Validate and reduce compact rows; this does not authenticate their origin."""
  if not isinstance(spec, BehaviorAggregateSpec):
    raise TypeError("behavior aggregation requires an aggregate specification")
  if not isinstance(split, BehaviorRouteSplit):
    raise TypeError("behavior aggregation requires a route split")
  if policy is not None and not isinstance(policy, BehaviorPolicy):
    raise TypeError("behavior aggregation policy has the wrong type")
  results = tuple(route_results)
  expected_scenarios = _expected_scenarios(spec, split)
  expected_route_ids = tuple(source.route_id for source in expected_scenarios)
  if tuple(result.scenario.route_id for result in results) != expected_route_ids:
    raise BehaviorAggregateError(
      "route results must exactly follow the frozen scenario order and membership",
    )
  if not results:
    raise BehaviorAggregateError("behavior aggregate partition has no route results")
  first = results[0]
  if first.physical_profile_sha256 != spec.physical_profile_sha256:
    raise BehaviorAggregateError("route result physical profile differs from authority")
  if first.provisional_dynamics_sha256 != spec.provisional_dynamics_sha256:
    raise BehaviorAggregateError("route result dynamics differ from authority")
  expected_artifact = ReplayArtifactIdentity.compose(
    first.artifact_identity.role,
    first.artifact_identity.core,
    policy,
  )
  if first.artifact_identity != expected_artifact:
    raise BehaviorAggregateError("aggregate opponent role, core, and policy disagree")
  expected_core = (
    spec.exact_stock_core
    if expected_artifact.role is ReplayRole.EXACT_STOCK
    else spec.modular_core
  )
  if expected_artifact.core != expected_core:
    raise BehaviorAggregateError("aggregate opponent uses an unreviewed replay core")
  expected_metric_names = tuple(BehaviorMetricName)
  route_identities: list[BehaviorAggregateRouteIdentity] = []
  all_windows: list[WindowMetricSet] = []
  for result, scenario in zip(results, expected_scenarios, strict=True):
    if result.scenario != scenario:
      raise BehaviorAggregateError("route result scenario differs from aggregate authority")
    if result.artifact_identity != expected_artifact:
      raise BehaviorAggregateError("route results mix replay artifacts")
    if result.physical_profile_sha256 != spec.physical_profile_sha256:
      raise BehaviorAggregateError("route results mix physical profiles")
    if result.provisional_dynamics_sha256 != spec.provisional_dynamics_sha256:
      raise BehaviorAggregateError("route results mix provisional dynamics")
    if result.plant_member_id != first.plant_member_id:
      raise BehaviorAggregateError("route results mix counterfactual plant members")
    if result.metric_config_sha256 != spec.gate_spec.metric_config.sha256:
      raise BehaviorAggregateError("route result metric configuration differs from gate")
    if result.segmentation_config_sha256 != spec.segmentation_config_sha256:
      raise BehaviorAggregateError("route result segmentation configuration differs")
    if result.preparation_schema_version != BEHAVIOR_ROUTE_PREPARATION_SCHEMA_VERSION:
      raise BehaviorAggregateError("route result preparation contract differs")
    for window in result.windows:
      if (
        window.route_id != result.scenario.route_id
        or window.source_identity_sha256 != result.scenario.recorded_source.sha256
      ):
        raise BehaviorAggregateError("window provenance differs from route scenario")
      names = tuple(metric.name for metric in window.metrics)
      if names != expected_metric_names:
        raise BehaviorAggregateError("window metric registry is incomplete or non-canonical")
    route_identities.append(BehaviorAggregateRouteIdentity(
      route_id=result.scenario.route_id,
      scenario_sha256=result.scenario.sha256,
      preparation_sha256=result.preparation_sha256,
      route_evaluation_sha256=result.sha256,
      window_contract_sha256=_window_contract_sha256(result.windows),
    ))
    all_windows.extend(result.windows)
  scorecard = aggregate_behavior_metrics(
    all_windows,
    spec.gate_spec.metric_config,
  )
  identity = BehaviorAggregateArtifactIdentity(
    schema_version=BEHAVIOR_AGGREGATE_ARTIFACT_SCHEMA_VERSION,
    aggregate_spec_sha256=spec.sha256,
    split=split,
    replay_artifact=expected_artifact,
    physical_profile_sha256=spec.physical_profile_sha256,
    provisional_dynamics_sha256=spec.provisional_dynamics_sha256,
    plant_member_id=first.plant_member_id,
    metric_config_sha256=spec.gate_spec.metric_config.sha256,
    segmentation_config_sha256=spec.segmentation_config_sha256,
    preparation_schema_version=BEHAVIOR_ROUTE_PREPARATION_SCHEMA_VERSION,
    aggregation_contract=spec.aggregation_contract,
    ordered_routes=tuple(route_identities),
  )
  evaluation = PolicyEvaluation(
    artifact_identity=expected_artifact.to_json(),
    policy=policy,
    route_ids=tuple(sorted(expected_route_ids)),
    metrics=tuple(
      PolicyMetric.from_scorecard(scorecard, name)
      for name in BehaviorMetricName
    ),
  )
  return BehaviorAggregateEvaluation(
    schema_version=BEHAVIOR_AGGREGATE_EVALUATION_SCHEMA_VERSION,
    identity=identity,
    policy=policy,
    metric_config=spec.gate_spec.metric_config,
    route_results=results,
    scorecard=scorecard,
    policy_evaluation=evaluation,
  )


def _route_contract(
  evaluation: BehaviorAggregateEvaluation,
) -> tuple[tuple[str, str, str, str], ...]:
  return tuple(
    (
      route.route_id,
      route.scenario_sha256,
      route.preparation_sha256,
      route.window_contract_sha256,
    )
    for route in evaluation.identity.ordered_routes
  )


def _platform_identity(evaluation: BehaviorAggregateEvaluation) -> tuple[str, str, str]:
  core = evaluation.identity.replay_artifact.core
  return core.source_openpilot_commit, core.opendbc_commit, core.panda_commit


def _validate_opponent_population(
  spec: BehaviorAggregateSpec,
  split: BehaviorRouteSplit,
  stock: BehaviorAggregateEvaluation,
  accepted: BehaviorAggregateEvaluation | None,
  candidates: tuple[BehaviorAggregateEvaluation, ...],
) -> None:
  population = (stock, *((accepted,) if accepted is not None else ()), *candidates)
  expected_routes = spec.partition.route_ids_for(split)
  for evaluation in population:
    if evaluation.identity.aggregate_spec_sha256 != spec.sha256:
      raise BehaviorAggregateError("aggregate evaluation belongs to another experiment")
    if evaluation.identity.split is not split:
      raise BehaviorAggregateError("aggregate evaluation belongs to another route split")
    if evaluation.identity.route_ids != expected_routes:
      raise BehaviorAggregateError("aggregate evaluation route order differs")
  if stock.identity.replay_artifact.role is not ReplayRole.EXACT_STOCK or stock.policy is not None:
    raise BehaviorAggregateError("comparison requires a policy-null exact-stock baseline")
  if stock.identity.replay_artifact.core != spec.exact_stock_core:
    raise BehaviorAggregateError("comparison exact-stock core differs from authority")
  if accepted is not None and (
    accepted.identity.replay_artifact.role is not ReplayRole.CURRENTLY_ACCEPTED
    or accepted.policy is None
  ):
    raise BehaviorAggregateError("modular incumbent identity is invalid")
  if accepted is not None and accepted.identity.replay_artifact.core != spec.modular_core:
    raise BehaviorAggregateError("comparison incumbent core differs from authority")
  if not candidates:
    raise BehaviorAggregateError("comparison requires at least one candidate")
  if any(
    candidate.identity.replay_artifact.role is not ReplayRole.CANDIDATE
    or candidate.policy is None
    or candidate.identity.replay_artifact.core != spec.modular_core
    for candidate in candidates
  ):
    raise BehaviorAggregateError("candidate identity is invalid")
  candidate_policies = tuple(candidate.policy for candidate in candidates)
  if len(set(candidate_policies)) != len(candidate_policies):
    raise BehaviorAggregateError("comparison contains duplicate candidate policies")
  candidate_cores = {
    candidate.identity.replay_artifact.core
    for candidate in candidates
  }
  if len(candidate_cores) != 1:
    raise BehaviorAggregateError("candidate population mixes numerical cores")
  if len({_platform_identity(evaluation) for evaluation in population}) != 1:
    raise BehaviorAggregateError("comparison mixes platform source commits")
  if accepted is not None:
    for candidate in candidates:
      if (
        candidate.policy == accepted.policy
        and candidate.identity.replay_artifact.core
        == accepted.identity.replay_artifact.core
        and candidate.scorecard != accepted.scorecard
      ):
        raise BehaviorAggregateError(
          "identical incumbent and candidate artifacts produced different metrics",
        )
  envelope = {
    (
      evaluation.identity.physical_profile_sha256,
      evaluation.identity.provisional_dynamics_sha256,
      evaluation.identity.plant_member_id,
      evaluation.identity.metric_config_sha256,
      evaluation.identity.segmentation_config_sha256,
      evaluation.identity.preparation_schema_version,
      evaluation.identity.aggregation_contract,
    )
    for evaluation in population
  }
  expected_envelope = {
    (
      spec.physical_profile_sha256,
      spec.provisional_dynamics_sha256,
      stock.identity.plant_member_id,
      spec.gate_spec.metric_config.sha256,
      spec.segmentation_config_sha256,
      spec.preparation_schema_version,
      spec.aggregation_contract,
    ),
  }
  if envelope != expected_envelope:
    raise BehaviorAggregateError("comparison mixes physical or metric contracts")
  route_contract = _route_contract(stock)
  if any(_route_contract(evaluation) != route_contract for evaluation in population[1:]):
    raise BehaviorAggregateError("comparison opponents use different prepared windows")


@dataclass(frozen=True, slots=True)
class BehaviorTrainingComparison:
  """Training-only opponents; validation candidates are not admitted here."""

  schema_version: int
  spec: BehaviorAggregateSpec
  exact_stock: BehaviorAggregateEvaluation
  accepted: BehaviorAggregateEvaluation | None
  candidates: tuple[BehaviorAggregateEvaluation, ...]

  def __post_init__(self) -> None:
    if self.schema_version != BEHAVIOR_TRAINING_COMPARISON_SCHEMA_VERSION:
      raise BehaviorAggregateError("behavior training comparison is incompatible")
    if not isinstance(self.spec, BehaviorAggregateSpec):
      raise TypeError("behavior training comparison requires aggregate authority")
    if type(self.candidates) is not tuple:
      raise TypeError("behavior training candidates must be a tuple")
    _validate_opponent_population(
      self.spec,
      BehaviorRouteSplit.TRAINING,
      self.exact_stock,
      self.accepted,
      self.candidates,
    )

  @property
  def accepted_policy_evaluation(self) -> PolicyEvaluation:
    if self.accepted is not None:
      return self.accepted.policy_evaluation
    return _bootstrap_accepted_alias(self.exact_stock)

  def to_dict(self) -> dict[str, object]:
    return {
      "acceptedAliasExactStock": self.accepted is None,
      "acceptedSelectorArtifactIdentity": (
        self.accepted_policy_evaluation.artifact_identity
      ),
      "acceptedEvaluationSha256": (
        self.exact_stock.sha256 if self.accepted is None else self.accepted.sha256
      ),
      "candidateEvaluationSha256s": [candidate.sha256 for candidate in self.candidates],
      "exactStockEvaluationSha256": self.exact_stock.sha256,
      "schemaVersion": self.schema_version,
      "specSha256": self.spec.sha256,
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())

  @property
  def sha256(self) -> str:
    return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


class BehaviorAggregateSelectionDisposition(StrEnum):
  STOCK_RETAINED = "stock_retained"
  INCUMBENT_RETAINED = "incumbent_retained"
  CANDIDATE_PROMOTED = "candidate_promoted"


@dataclass(frozen=True, slots=True)
class BehaviorAggregateSelection:
  schema_version: int
  comparison: BehaviorTrainingComparison
  search_center_policy: BehaviorPolicy
  disposition: BehaviorAggregateSelectionDisposition
  selected_artifact: ReplayArtifactIdentity
  selected_policy: BehaviorPolicy | None
  training_selection: TrainingSelection | None
  held_out_validation: HeldOutValidation | None
  stock_validation: BehaviorAggregateEvaluation | None
  accepted_validation: BehaviorAggregateEvaluation | None
  winner_validation: BehaviorAggregateEvaluation | None

  def __post_init__(self) -> None:
    if self.schema_version != BEHAVIOR_AGGREGATE_SELECTION_SCHEMA_VERSION:
      raise BehaviorAggregateError("behavior aggregate selection is incompatible")
    if not isinstance(self.comparison, BehaviorTrainingComparison):
      raise TypeError("behavior aggregate selection requires its training comparison")
    if not isinstance(self.search_center_policy, BehaviorPolicy):
      raise TypeError("behavior aggregate selection requires its search-center policy")
    if not isinstance(self.disposition, BehaviorAggregateSelectionDisposition):
      raise TypeError("behavior aggregate selection disposition has the wrong type")
    if not isinstance(self.selected_artifact, ReplayArtifactIdentity):
      raise TypeError("behavior aggregate selection requires a replay artifact")
    expected_artifact = ReplayArtifactIdentity.compose(
      self.selected_artifact.role,
      self.selected_artifact.core,
      self.selected_policy,
    )
    if self.selected_artifact != expected_artifact:
      raise BehaviorAggregateError("selected artifact and policy disagree")
    expected_training = _select_training_candidate(
      self.comparison,
      self.search_center_policy,
    )
    if self.training_selection != expected_training:
      raise BehaviorAggregateError("training selection differs from comparison authority")
    if expected_training is None:
      if any(value is not None for value in (
        self.held_out_validation,
        self.stock_validation,
        self.accepted_validation,
        self.winner_validation,
      )):
        raise BehaviorAggregateError("no-winner selection cannot contain validation evidence")
      retained = (
        self.comparison.exact_stock
        if self.comparison.accepted is None
        else self.comparison.accepted
      )
      expected_disposition = (
        BehaviorAggregateSelectionDisposition.STOCK_RETAINED
        if self.comparison.accepted is None
        else BehaviorAggregateSelectionDisposition.INCUMBENT_RETAINED
      )
      if (
        self.disposition is not expected_disposition
        or self.selected_artifact != retained.identity.replay_artifact
        or self.selected_policy != retained.policy
      ):
        raise BehaviorAggregateError("no-winner selection does not retain the incumbent")
      return
    if (
      self.held_out_validation is None
      or self.stock_validation is None
      or self.winner_validation is None
    ):
      raise BehaviorAggregateError("frozen winner selection lacks validation evidence")
    if self.comparison.accepted is None:
      if self.accepted_validation is not None:
        raise BehaviorAggregateError("bootstrap selection cannot carry modular validation")
    elif self.accepted_validation is None:
      raise BehaviorAggregateError("modular incumbent lacks validation evidence")
    _validate_opponent_population(
      self.comparison.spec,
      BehaviorRouteSplit.VALIDATION,
      self.stock_validation,
      self.accepted_validation,
      (self.winner_validation,),
    )
    if (
      self.stock_validation.identity.replay_artifact
      != self.comparison.exact_stock.identity.replay_artifact
    ):
      raise BehaviorAggregateError("stock validation uses a different replay artifact")
    if self.comparison.accepted is None:
      effective_accepted_validation = _bootstrap_accepted_alias(self.stock_validation)
    else:
      assert self.accepted_validation is not None
      assert self.comparison.accepted is not None
      if (
        self.accepted_validation.identity.replay_artifact
        != self.comparison.accepted.identity.replay_artifact
      ):
        raise BehaviorAggregateError("accepted validation uses a different replay artifact")
      effective_accepted_validation = self.accepted_validation.policy_evaluation
    winner_training = next(
      candidate
      for candidate in self.comparison.candidates
      if candidate.policy == expected_training.winner.policy
    )
    if (
      self.winner_validation.identity.replay_artifact
      != winner_training.identity.replay_artifact
    ):
      raise BehaviorAggregateError("winner validation uses a different replay artifact")
    gate = self.comparison.spec.gate_spec
    expected_held_out = validate_frozen_winner(
      expected_training,
      self.winner_validation.policy_evaluation,
      self.stock_validation.policy_evaluation,
      effective_accepted_validation,
      gate.metric_rules,
      gate.target_metric_name,
      gate.paired_uncertainty_method,
      gate.minimum_paired_route_count,
    )
    if self.held_out_validation != expected_held_out:
      raise BehaviorAggregateError("held-out verdict differs from validation evidence")
    promoted = expected_held_out.accepted
    retained = (
      self.comparison.exact_stock
      if self.comparison.accepted is None
      else self.comparison.accepted
    )
    expected_disposition = (
      BehaviorAggregateSelectionDisposition.CANDIDATE_PROMOTED
      if promoted
      else (
        BehaviorAggregateSelectionDisposition.STOCK_RETAINED
        if self.comparison.accepted is None
        else BehaviorAggregateSelectionDisposition.INCUMBENT_RETAINED
      )
    )
    expected_selected = self.winner_validation if promoted else retained
    if (
      self.disposition is not expected_disposition
      or self.selected_artifact != expected_selected.identity.replay_artifact
      or self.selected_policy != expected_selected.policy
    ):
      raise BehaviorAggregateError("selection result differs from held-out authority")

  @property
  def comparison_sha256(self) -> str:
    return self.comparison.sha256

  @property
  def stock_validation_sha256(self) -> str | None:
    return None if self.stock_validation is None else self.stock_validation.sha256

  @property
  def accepted_validation_sha256(self) -> str | None:
    if self.stock_validation is None:
      return None
    return (
      self.stock_validation.sha256
      if self.accepted_validation is None
      else self.accepted_validation.sha256
    )

  @property
  def winner_validation_sha256(self) -> str | None:
    return None if self.winner_validation is None else self.winner_validation.sha256

  @property
  def promoted(self) -> bool:
    return self.disposition is BehaviorAggregateSelectionDisposition.CANDIDATE_PROMOTED

  def to_dict(self) -> dict[str, object]:
    return {
      "acceptedValidationSha256": self.accepted_validation_sha256,
      "comparisonSha256": self.comparison_sha256,
      "disposition": self.disposition.value,
      "heldOutValidation": (
        None if self.held_out_validation is None else self.held_out_validation.to_dict()
      ),
      "schemaVersion": self.schema_version,
      "searchCenterPolicy": self.search_center_policy.to_dict(),
      "selectedArtifact": self.selected_artifact.to_dict(),
      "selectedPolicy": (
        None if self.selected_policy is None else self.selected_policy.to_dict()
      ),
      "stockValidationSha256": self.stock_validation_sha256,
      "trainingSelection": (
        None if self.training_selection is None else self.training_selection.to_dict()
      ),
      "winnerValidationSha256": self.winner_validation_sha256,
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())

  @property
  def sha256(self) -> str:
    return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


FrozenValidationEvaluator = Callable[
  [ReplayArtifactIdentity, BehaviorPolicy],
  BehaviorAggregateEvaluation,
]


def _bootstrap_accepted_alias(
  stock: BehaviorAggregateEvaluation,
) -> PolicyEvaluation:
  """Give the selector its incumbent role without replaying a fake opponent."""
  if stock.identity.replay_artifact.role is not ReplayRole.EXACT_STOCK:
    raise BehaviorAggregateError("bootstrap alias requires exact-stock evaluation")
  identity = ReplayArtifactIdentity.compose(
    ReplayRole.CURRENTLY_ACCEPTED,
    stock.identity.replay_artifact.core,
    None,
  )
  return PolicyEvaluation(
    artifact_identity=identity.to_json(),
    policy=None,
    route_ids=stock.policy_evaluation.route_ids,
    metrics=stock.policy_evaluation.metrics,
  )


def _retained_selection(
  comparison: BehaviorTrainingComparison,
  search_center_policy: BehaviorPolicy,
  training_selection: TrainingSelection | None = None,
  held_out_validation: HeldOutValidation | None = None,
  stock_validation: BehaviorAggregateEvaluation | None = None,
  accepted_validation: BehaviorAggregateEvaluation | None = None,
  winner_validation: BehaviorAggregateEvaluation | None = None,
) -> BehaviorAggregateSelection:
  retained = comparison.exact_stock if comparison.accepted is None else comparison.accepted
  disposition = (
    BehaviorAggregateSelectionDisposition.STOCK_RETAINED
    if comparison.accepted is None
    else BehaviorAggregateSelectionDisposition.INCUMBENT_RETAINED
  )
  return BehaviorAggregateSelection(
    schema_version=BEHAVIOR_AGGREGATE_SELECTION_SCHEMA_VERSION,
    comparison=comparison,
    search_center_policy=search_center_policy,
    disposition=disposition,
    selected_artifact=retained.identity.replay_artifact,
    selected_policy=retained.policy,
    training_selection=training_selection,
    held_out_validation=held_out_validation,
    stock_validation=stock_validation,
    accepted_validation=accepted_validation,
    winner_validation=winner_validation,
  )


def _select_training_candidate(
  comparison: BehaviorTrainingComparison,
  search_center_policy: BehaviorPolicy,
) -> TrainingSelection | None:
  if not isinstance(comparison, BehaviorTrainingComparison):
    raise TypeError("behavior selection requires a training comparison")
  if not isinstance(search_center_policy, BehaviorPolicy):
    raise TypeError("behavior selection requires a search-center policy")
  if comparison.accepted is not None and comparison.accepted.policy != search_center_policy:
    raise BehaviorAggregateError("modular search must be centered on the incumbent")
  grid = build_candidate_grid(
    comparison.spec.gate_spec.candidate_grid.policy_grid(search_center_policy),
  )
  if tuple(candidate.policy for candidate in comparison.candidates) != tuple(
    candidate.policy for candidate in grid
  ):
    raise BehaviorAggregateError("training candidates differ from the gate-owned grid")
  gate = comparison.spec.gate_spec
  return select_training_winner(
    grid,
    tuple(candidate.policy_evaluation for candidate in comparison.candidates),
    comparison.exact_stock.policy_evaluation,
    comparison.accepted_policy_evaluation,
    gate.metric_rules,
    gate.target_metric_name,
    gate.paired_uncertainty_method,
    gate.minimum_paired_route_count,
  )


def select_behavior_candidate(
  comparison: BehaviorTrainingComparison,
  search_center_policy: BehaviorPolicy,
  stock_validation: BehaviorAggregateEvaluation,
  accepted_validation: BehaviorAggregateEvaluation | None,
  evaluate_frozen_validation: FrozenValidationEvaluator,
) -> BehaviorAggregateSelection:
  """Freeze on training, evaluate exactly one winner, then retain or promote."""
  selection = _select_training_candidate(comparison, search_center_policy)
  if selection is None:
    return _retained_selection(comparison, search_center_policy)
  gate = comparison.spec.gate_spec
  winner_training = next(
    candidate
    for candidate in comparison.candidates
    if candidate.policy == selection.winner.policy
  )
  winner_validation = evaluate_frozen_validation(
    winner_training.identity.replay_artifact,
    selection.winner.policy,
  )
  if not isinstance(winner_validation, BehaviorAggregateEvaluation):
    raise BehaviorAggregateError("frozen validation callback returned the wrong type")
  _validate_opponent_population(
    comparison.spec,
    BehaviorRouteSplit.VALIDATION,
    stock_validation,
    accepted_validation,
    (winner_validation,),
  )
  if stock_validation.identity.replay_artifact != comparison.exact_stock.identity.replay_artifact:
    raise BehaviorAggregateError("stock validation uses a different replay artifact")
  if comparison.accepted is None:
    if accepted_validation is not None:
      raise BehaviorAggregateError("bootstrap validation must reuse exact stock")
    effective_accepted_validation = _bootstrap_accepted_alias(stock_validation)
  else:
    if accepted_validation is None:
      raise BehaviorAggregateError("modular incumbent lacks validation evidence")
    if (
      accepted_validation.identity.replay_artifact
      != comparison.accepted.identity.replay_artifact
    ):
      raise BehaviorAggregateError("accepted validation uses a different replay artifact")
    effective_accepted_validation = accepted_validation.policy_evaluation
  if winner_validation.identity.replay_artifact != winner_training.identity.replay_artifact:
    raise BehaviorAggregateError("winner validation uses a different replay artifact")
  held_out = validate_frozen_winner(
    selection,
    winner_validation.policy_evaluation,
    stock_validation.policy_evaluation,
    effective_accepted_validation,
    gate.metric_rules,
    gate.target_metric_name,
    gate.paired_uncertainty_method,
    gate.minimum_paired_route_count,
  )
  if not held_out.accepted:
    return _retained_selection(
      comparison,
      search_center_policy,
      training_selection=selection,
      held_out_validation=held_out,
      stock_validation=stock_validation,
      accepted_validation=accepted_validation,
      winner_validation=winner_validation,
    )
  return BehaviorAggregateSelection(
    schema_version=BEHAVIOR_AGGREGATE_SELECTION_SCHEMA_VERSION,
    comparison=comparison,
    search_center_policy=search_center_policy,
    disposition=BehaviorAggregateSelectionDisposition.CANDIDATE_PROMOTED,
    selected_artifact=winner_validation.identity.replay_artifact,
    selected_policy=selection.winner.policy,
    training_selection=selection,
    held_out_validation=held_out,
    stock_validation=stock_validation,
    accepted_validation=accepted_validation,
    winner_validation=winner_validation,
  )

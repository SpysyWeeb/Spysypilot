from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

from openpilot.selfdrive.controls.lib.blatv2.behavior_aggregate import (
  BEHAVIOR_TRAINING_COMPARISON_SCHEMA_VERSION,
  BehaviorAggregateError,
  BehaviorAggregateSelectionDisposition,
  BehaviorAggregateSpec,
  BehaviorRouteSplit,
  BehaviorTrainingComparison,
  aggregate_behavior_route_results,
  select_behavior_candidate,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_coordinator import (
  BEHAVIOR_GATE_SPEC_SCHEMA_VERSION,
  BehaviorGateSpec,
  CandidateGridBounds,
  ReplayArtifactIdentity,
  ReplayCoreIdentity,
  ReplayRole,
  RoutePartitionSpec,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import (
  BEHAVIOR_SCENARIO_PROVENANCE_SCHEMA_VERSION,
  BehaviorScenarioProvenance,
  BehaviorScenarioSetIdentity,
  BehaviorSourceIdentity,
  ManeuverClass,
  ManeuverPhase,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_metrics import (
  BehaviorContract,
  BehaviorMetricConfig,
  BehaviorMetricName,
  MetricDisposition,
  MetricValue,
  WindowMetricSet,
  aggregate_behavior_metrics,
  behavior_metric_contract,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_policy import (
  PAIRED_ROUTE_UNCERTAINTY_METHOD,
  BehaviorPolicy,
  MetricGateRule,
  MetricPreference,
  PolicyMetric,
  build_candidate_grid,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_replay import (
  reviewed_replay_core_identity,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_route_evaluator import (
  BEHAVIOR_ROUTE_EVALUATION_SCHEMA_VERSION,
  BEHAVIOR_ROUTE_PREPARATION_SCHEMA_VERSION,
  BehaviorRouteEvaluation,
)


def _hash(label: str) -> str:
  return hashlib.sha256(label.encode()).hexdigest()


class AggregateFixture:
  profile_sha = _hash("profile")
  dynamics_sha = _hash("dynamics")
  segmentation_sha = _hash("segmentation")
  center = BehaviorPolicy(10.0, 1.0)

  def __init__(self) -> None:
    self.metric_config = BehaviorMetricConfig(
      burst_window_s=1.0,
      chatter_torque_rate_threshold_per_s=0.1,
      turn_in_crossing_fraction=0.5,
      release_crossing_fraction=0.9,
      correction_curvature_threshold_1pm=0.001,
      unused_headroom_threshold=0.1,
      growing_error_epsilon_1pm=0.0001,
      completion_delivered_fraction=0.9,
      minimum_samples=2,
      speed_nodes_mps=(5.0,),
      maximum_route_windows_per_stratum=4,
    )
    self.gate = BehaviorGateSpec(
      schema_version=BEHAVIOR_GATE_SPEC_SCHEMA_VERSION,
      provenance="aggregate-test",
      metric_config=self.metric_config,
      metric_rules=(
        self._rule(
          BehaviorMetricName.RAW_TORQUE_RATE_RMS,
          BehaviorContract.SMOOTH,
          MetricPreference.LOWER_IS_BETTER,
          maximum=20.0,
        ),
        self._rule(
          BehaviorMetricName.SIGNED_TURN_IN_LAG_S,
          BehaviorContract.SWIFT,
          MetricPreference.LOWER_IS_BETTER,
          maximum=1.0,
        ),
        self._rule(
          BehaviorMetricName.DELIVERED_FRACTION,
          BehaviorContract.STRONG,
          MetricPreference.HIGHER_IS_BETTER,
          minimum=0.5,
          maximum=2.0,
        ),
      ),
      target_metric_name=BehaviorMetricName.DELIVERED_FRACTION.value,
      paired_uncertainty_method=PAIRED_ROUTE_UNCERTAINTY_METHOD,
      minimum_paired_route_count=2,
      candidate_grid=CandidateGridBounds(
        natural_frequency_log_offsets=(0.0, 0.1),
        damping_ratio_log_offsets=(0.0,),
        minimum_natural_frequency_per_s=5.0,
        maximum_natural_frequency_per_s=20.0,
        minimum_damping_ratio=0.5,
        maximum_damping_ratio=2.0,
      ),
      route_partition=RoutePartitionSpec(
        validation_fraction=None,
        validation_route_count=2,
        seed_identity_sha256=_hash("split-seed"),
      ),
    )
    self.scenarios = BehaviorScenarioSetIdentity(tuple(
      self._scenario(index)
      for index in range(4)
    ))
    commits = {
      "source_openpilot_commit": "1" * 40,
      "opendbc_commit": "2" * 40,
      "panda_commit": "3" * 40,
    }
    self.stock_core = reviewed_replay_core_identity(exact_stock=True, **commits)
    self.modular_core = reviewed_replay_core_identity(exact_stock=False, **commits)
    self.spec = BehaviorAggregateSpec.freeze(
      self.scenarios,
      self.gate,
      self.stock_core,
      self.modular_core,
      self.profile_sha,
      self.dynamics_sha,
      self.segmentation_sha,
    )
    self.grid = build_candidate_grid(self.gate.candidate_grid.policy_grid(self.center))

  @staticmethod
  def _rule(
    name: BehaviorMetricName,
    contract: BehaviorContract,
    preference: MetricPreference,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
  ) -> MetricGateRule:
    return MetricGateRule(
      metric_name=name.value,
      contract=contract,
      preference=preference,
      noise_floor=0.01,
      margin_normalization=1.0,
      minimum_allowed=minimum,
      maximum_allowed=maximum,
      minimum_route_count=2,
      minimum_window_count=2,
      minimum_weighted_support=2.0,
      required_strata=("5:turn",),
    )

  @staticmethod
  def _scenario(index: int) -> BehaviorScenarioProvenance:
    recorded = BehaviorSourceIdentity(
      controller_name=f"recorded-{'stock' if index % 2 == 0 else 'blat'}",
      controller_artifact_sha256=_hash(f"recorded-artifact-{index}"),
      source_openpilot_commit=f"{index + 4:x}" * 40,
      opendbc_commit=f"{index + 8:x}" * 40,
      panda_commit=f"{index + 12:x}" * 40,
      evidence_schema_version=1,
    )
    return BehaviorScenarioProvenance(
      schema_version=BEHAVIOR_SCENARIO_PROVENANCE_SCHEMA_VERSION,
      route_id=f"route-{index}",
      route_evidence_sha256=_hash(f"route-evidence-{index}"),
      recorded_source=recorded,
      recorded_behavior_eligible=True,
      recorded_behavior_ineligible_reason="eligible",
      vehicle_identity="test-car",
      runtime_identity=_hash(f"runtime-{index}"),
      preparation_cache_key=_hash(f"preparation-cache-{index}"),
    )

  def _metrics(
    self,
    roughness: float,
    lag: float,
    delivered: float,
    *,
    undefined: bool = False,
  ) -> tuple[MetricValue, ...]:
    overrides = {
      BehaviorMetricName.RAW_TORQUE_RATE_RMS: roughness,
      BehaviorMetricName.SIGNED_TURN_IN_LAG_S: lag,
      BehaviorMetricName.DELIVERED_FRACTION: delivered,
    }
    values = []
    for name in BehaviorMetricName:
      if undefined and name in overrides:
        values.append(MetricValue(
          name=name,
          contract=behavior_metric_contract(name),
          value=None,
          denominator=0,
          exclusions=("fixture_coverage",),
          disposition=MetricDisposition.COVERAGE_EXCLUDED,
        ))
      else:
        values.append(MetricValue(
          name=name,
          contract=behavior_metric_contract(name),
          value=overrides.get(name, 0.25),
          denominator=10,
          exclusions=(),
          disposition=MetricDisposition.DEFINED,
        ))
    return tuple(values)

  def route_result(
    self,
    scenario: BehaviorScenarioProvenance,
    role: ReplayRole,
    policy: BehaviorPolicy | None,
    values: tuple[float, float, float],
    *,
    undefined: bool = False,
    plant_member_id: str | None = None,
  ) -> BehaviorRouteEvaluation:
    core = self.stock_core if role is ReplayRole.EXACT_STOCK else self.modular_core
    window = WindowMetricSet(
      route_id=scenario.route_id,
      window_id=f"{scenario.route_id}-turn",
      source_identity_sha256=scenario.recorded_source.sha256,
      maneuver_class=ManeuverClass.TURN,
      phase=ManeuverPhase.TURN_IN,
      mean_speed_mps=5.0,
      speed_node_support=((5.0, 1.0),),
      clean_sample_count=10,
      intervention_mono_time_ns=None,
      metrics=self._metrics(*values, undefined=undefined),
    )
    return BehaviorRouteEvaluation(
      schema_version=BEHAVIOR_ROUTE_EVALUATION_SCHEMA_VERSION,
      preparation_sha256=_hash(f"prepared-{scenario.route_id}"),
      preparation_schema_version=BEHAVIOR_ROUTE_PREPARATION_SCHEMA_VERSION,
      scenario=scenario,
      single_route_scenario_set_sha256=BehaviorScenarioSetIdentity((scenario,)).sha256,
      artifact_identity=ReplayArtifactIdentity.compose(role, core, policy),
      physical_profile_sha256=self.profile_sha,
      provisional_dynamics_sha256=self.dynamics_sha,
      plant_member_id=(
        _hash("plant-member") if plant_member_id is None else plant_member_id
      ),
      segmentation_config_sha256=self.segmentation_sha,
      metric_config_sha256=self.metric_config.sha256,
      windows=(window,),
    )

  def aggregate(
    self,
    split: BehaviorRouteSplit,
    role: ReplayRole,
    policy: BehaviorPolicy | None,
    values: tuple[float, float, float],
    *,
    undefined: bool = False,
    plant_member_id: str | None = None,
  ):
    scenarios = {
      scenario.route_id: scenario
      for scenario in self.scenarios.sources
    }
    results = tuple(
      self.route_result(
        scenarios[route_id],
        role,
        policy,
        values,
        undefined=undefined,
        plant_member_id=plant_member_id,
      )
      for route_id in self.spec.partition.route_ids_for(split)
    )
    return aggregate_behavior_route_results(self.spec, split, results, policy)

  def comparison(
    self,
    *,
    candidate_values: tuple[float, float, float] = (7.0, 0.2, 1.0),
    undefined: bool = False,
  ) -> BehaviorTrainingComparison:
    stock = self.aggregate(
      BehaviorRouteSplit.TRAINING,
      ReplayRole.EXACT_STOCK,
      None,
      (10.0, 0.5, 0.8),
    )
    accepted = self.aggregate(
      BehaviorRouteSplit.TRAINING,
      ReplayRole.CURRENTLY_ACCEPTED,
      self.center,
      (9.0, 0.4, 0.85),
    )
    candidates = (
      self.aggregate(
        BehaviorRouteSplit.TRAINING,
        ReplayRole.CANDIDATE,
        self.grid[0].policy,
        (9.0, 0.4, 0.85),
      ),
      self.aggregate(
        BehaviorRouteSplit.TRAINING,
        ReplayRole.CANDIDATE,
        self.grid[1].policy,
        candidate_values,
        undefined=undefined,
      ),
    )
    return BehaviorTrainingComparison(
      schema_version=BEHAVIOR_TRAINING_COMPARISON_SCHEMA_VERSION,
      spec=self.spec,
      exact_stock=stock,
      accepted=accepted,
      candidates=candidates,
    )

  def validation_references(self):
    return (
      self.aggregate(
        BehaviorRouteSplit.VALIDATION,
        ReplayRole.EXACT_STOCK,
        None,
        (10.0, 0.5, 0.8),
      ),
      self.aggregate(
        BehaviorRouteSplit.VALIDATION,
        ReplayRole.CURRENTLY_ACCEPTED,
        self.center,
        (9.0, 0.4, 0.85),
      ),
    )


class TestBehaviorAggregate(unittest.TestCase):
  def setUp(self) -> None:
    self.fixture = AggregateFixture()

  def test_mixed_recorded_sources_aggregate_deterministically(self):
    evaluation = self.fixture.aggregate(
      BehaviorRouteSplit.TRAINING,
      ReplayRole.EXACT_STOCK,
      None,
      (10.0, 0.5, 0.8),
    )
    repeated = self.fixture.aggregate(
      BehaviorRouteSplit.TRAINING,
      ReplayRole.EXACT_STOCK,
      None,
      (10.0, 0.5, 0.8),
    )
    self.assertEqual(evaluation.sha256, repeated.sha256)
    self.assertEqual(
      evaluation.policy_evaluation.metrics,
      tuple(
        PolicyMetric.from_scorecard(evaluation.scorecard, name)
        for name in BehaviorMetricName
      ),
    )
    recorded_names = {
      scenario.recorded_source.controller_name
      for scenario in self.fixture.scenarios.sources
    }
    self.assertEqual(recorded_names, {"recorded-stock", "recorded-blat"})

  def test_opponents_from_different_plant_members_cannot_compare(self):
    comparison = self.fixture.comparison()
    foreign = self.fixture.aggregate(
      BehaviorRouteSplit.TRAINING,
      ReplayRole.CANDIDATE,
      self.fixture.grid[1].policy,
      (7.0, 0.2, 1.0),
      plant_member_id=_hash("foreign-plant-member"),
    )
    with self.assertRaisesRegex(BehaviorAggregateError, "physical or metric contracts"):
      replace(
        comparison,
        candidates=(comparison.candidates[0], foreign),
      )

  def test_route_order_duplicate_and_mixed_vehicle_fail_closed(self):
    route_ids = self.fixture.spec.partition.route_ids_for(BehaviorRouteSplit.TRAINING)
    scenarios = {scenario.route_id: scenario for scenario in self.fixture.scenarios.sources}
    results = tuple(
      self.fixture.route_result(
        scenarios[route_id], ReplayRole.EXACT_STOCK, None, (10.0, 0.5, 0.8),
      )
      for route_id in route_ids
    )
    for malformed in ((results[1], results[0]), (results[0], results[0])):
      with self.assertRaises(BehaviorAggregateError):
        aggregate_behavior_route_results(
          self.fixture.spec, BehaviorRouteSplit.TRAINING, malformed, None,
        )
    mixed = replace(self.fixture.scenarios.sources[-1], vehicle_identity="another-car")
    with self.assertRaises(BehaviorAggregateError):
      BehaviorAggregateSpec.freeze(
        BehaviorScenarioSetIdentity((*self.fixture.scenarios.sources[:-1], mixed)),
        self.fixture.gate,
        self.fixture.stock_core,
        self.fixture.modular_core,
        self.fixture.profile_sha,
        self.fixture.dynamics_sha,
        self.fixture.segmentation_sha,
      )

  def test_spec_rejects_reviewed_name_with_forged_core_hash(self):
    forged = ReplayCoreIdentity(
      controller_name=self.fixture.stock_core.controller_name,
      core_artifact_sha256=_hash("forged-reviewed-name"),
      source_openpilot_commit=self.fixture.stock_core.source_openpilot_commit,
      opendbc_commit=self.fixture.stock_core.opendbc_commit,
      panda_commit=self.fixture.stock_core.panda_commit,
    )
    with self.assertRaises(BehaviorAggregateError):
      replace(self.fixture.spec, exact_stock_core=forged)

  def test_spec_binds_partition_profile_dynamics_and_segmentation(self):
    moved = replace(
      self.fixture.spec.partition.assignments[0],
      split=(
        BehaviorRouteSplit.VALIDATION
        if self.fixture.spec.partition.assignments[0].split is BehaviorRouteSplit.TRAINING
        else BehaviorRouteSplit.TRAINING
      ),
    )
    with self.assertRaises(BehaviorAggregateError):
      replace(
        self.fixture.spec,
        partition=replace(
          self.fixture.spec.partition,
          assignments=(moved, *self.fixture.spec.partition.assignments[1:]),
        ),
      )
    self.assertNotEqual(
      self.fixture.spec.sha256,
      replace(self.fixture.spec, physical_profile_sha256=_hash("profile-2")).sha256,
    )
    self.assertNotEqual(
      self.fixture.spec.sha256,
      replace(self.fixture.spec, provisional_dynamics_sha256=_hash("dynamics-2")).sha256,
    )
    self.assertNotEqual(
      self.fixture.spec.sha256,
      replace(self.fixture.spec, segmentation_config_sha256=_hash("segmentation-2")).sha256,
    )

  def test_aggregation_rejects_role_policy_and_contract_spoofs(self):
    route_id = self.fixture.spec.partition.training_route_ids[0]
    scenario = next(value for value in self.fixture.scenarios.sources if value.route_id == route_id)
    stock = self.fixture.route_result(
      scenario, ReplayRole.EXACT_STOCK, None, (10.0, 0.5, 0.8),
    )
    with self.assertRaises(BehaviorAggregateError):
      aggregate_behavior_route_results(
        self.fixture.spec,
        BehaviorRouteSplit.TRAINING,
        (replace(stock, physical_profile_sha256=_hash("foreign-profile")), stock),
        None,
      )
    candidate = self.fixture.route_result(
      scenario, ReplayRole.CANDIDATE, self.fixture.grid[0].policy, (9.0, 0.4, 0.85),
    )
    with self.assertRaises(BehaviorAggregateError):
      aggregate_behavior_route_results(
        self.fixture.spec,
        BehaviorRouteSplit.TRAINING,
        (candidate, candidate),
        self.fixture.grid[1].policy,
      )

  def test_scorecard_window_geometry_cannot_be_substituted(self):
    evaluation = self.fixture.aggregate(
      BehaviorRouteSplit.TRAINING,
      ReplayRole.EXACT_STOCK,
      None,
      (10.0, 0.5, 0.8),
    )
    altered_window = replace(
      evaluation.scorecard.windows[0],
      intervention_mono_time_ns=123,
    )
    altered_scorecard = replace(
      evaluation.scorecard,
      windows=(altered_window, *evaluation.scorecard.windows[1:]),
    )
    with self.assertRaises(BehaviorAggregateError):
      replace(
        evaluation,
        scorecard=altered_scorecard,
        policy_evaluation=replace(
          evaluation.policy_evaluation,
          metrics=tuple(
            PolicyMetric.from_scorecard(altered_scorecard, name)
            for name in BehaviorMetricName
          ),
        ),
      )

  def test_candidate_specific_clean_metadata_does_not_poison_population(self):
    comparison = self.fixture.comparison()
    scenarios = {
      scenario.route_id: scenario
      for scenario in self.fixture.scenarios.sources
    }
    policy = self.fixture.grid[1].policy
    altered_results = []
    for route_id in self.fixture.spec.partition.training_route_ids:
      result = self.fixture.route_result(
        scenarios[route_id], ReplayRole.CANDIDATE, policy, (7.0, 0.2, 1.0),
      )
      altered_results.append(replace(
        result,
        windows=(replace(
          result.windows[0],
          clean_sample_count=5,
          mean_speed_mps=4.75,
        ),),
      ))
    candidate = aggregate_behavior_route_results(
      self.fixture.spec,
      BehaviorRouteSplit.TRAINING,
      tuple(altered_results),
      policy,
    )
    comparison = replace(
      comparison,
      candidates=(comparison.candidates[0], candidate),
    )
    stock_validation, accepted_validation = self.fixture.validation_references()
    result = select_behavior_candidate(
      comparison,
      self.fixture.center,
      stock_validation,
      accepted_validation,
      lambda _artifact, candidate_policy: self.fixture.aggregate(
        BehaviorRouteSplit.VALIDATION,
        ReplayRole.CANDIDATE,
        candidate_policy,
        (7.0, 0.2, 1.0),
      ),
    )
    self.assertTrue(result.promoted)
    self.assertEqual(result.selected_policy, policy)

  def test_aggregate_metrics_cannot_be_forged_away_from_route_evidence(self):
    evaluation = self.fixture.aggregate(
      BehaviorRouteSplit.TRAINING,
      ReplayRole.EXACT_STOCK,
      None,
      (10.0, 0.5, 0.8),
    )
    original_window = evaluation.scorecard.windows[0]
    original_metric = original_window.metrics[0]
    altered_window = replace(
      original_window,
      metrics=(replace(original_metric, value=1.0), *original_window.metrics[1:]),
    )
    forged_scorecard = aggregate_behavior_metrics(
      (altered_window, *evaluation.scorecard.windows[1:]),
      self.fixture.metric_config,
    )
    forged_policy_evaluation = replace(
      evaluation.policy_evaluation,
      metrics=tuple(
        PolicyMetric.from_scorecard(forged_scorecard, name)
        for name in BehaviorMetricName
      ),
    )
    with self.assertRaisesRegex(BehaviorAggregateError, "route evidence"):
      replace(
        evaluation,
        scorecard=forged_scorecard,
        policy_evaluation=forged_policy_evaluation,
      )

  def test_training_win_validation_regression_retains_incumbent_once(self):
    comparison = self.fixture.comparison()
    stock_validation, accepted_validation = self.fixture.validation_references()
    calls = []

    def evaluate(artifact, policy):
      calls.append((artifact, policy))
      return self.fixture.aggregate(
        BehaviorRouteSplit.VALIDATION,
        ReplayRole.CANDIDATE,
        policy,
        (12.0, 0.7, 0.6),
      )

    result = select_behavior_candidate(
      comparison,
      self.fixture.center,
      stock_validation,
      accepted_validation,
      evaluate,
    )
    self.assertEqual(len(calls), 1)
    self.assertEqual(result.disposition, BehaviorAggregateSelectionDisposition.INCUMBENT_RETAINED)
    self.assertFalse(result.held_out_validation.accepted)

  def test_training_and_validation_pass_promotes_and_constructor_rejects_forgery(self):
    comparison = self.fixture.comparison()
    stock_validation, accepted_validation = self.fixture.validation_references()

    def evaluate(_artifact, policy):
      return self.fixture.aggregate(
        BehaviorRouteSplit.VALIDATION,
        ReplayRole.CANDIDATE,
        policy,
        (7.0, 0.2, 1.0),
      )

    result = select_behavior_candidate(
      comparison,
      self.fixture.center,
      stock_validation,
      accepted_validation,
      evaluate,
    )
    self.assertTrue(result.promoted)
    self.assertEqual(result.selected_policy, self.fixture.grid[1].policy)
    with self.assertRaises(BehaviorAggregateError):
      replace(result, disposition=BehaviorAggregateSelectionDisposition.INCUMBENT_RETAINED)
    with self.assertRaises(BehaviorAggregateError):
      replace(result, selected_policy=self.fixture.center)
    with self.assertRaises(BehaviorAggregateError):
      replace(result, held_out_validation=replace(result.held_out_validation, accepted=False))

  def test_exact_tie_and_undefined_coverage_do_not_touch_validation(self):
    stock = self.fixture.aggregate(
      BehaviorRouteSplit.TRAINING,
      ReplayRole.EXACT_STOCK,
      None,
      (10.0, 0.5, 0.8),
    )
    candidates = tuple(
      self.fixture.aggregate(
        BehaviorRouteSplit.TRAINING,
        ReplayRole.CANDIDATE,
        candidate.policy,
        (10.0, 0.5, 0.8),
      )
      for candidate in self.fixture.grid
    )
    comparison = BehaviorTrainingComparison(
      BEHAVIOR_TRAINING_COMPARISON_SCHEMA_VERSION,
      self.fixture.spec,
      stock,
      None,
      candidates,
    )
    stock_validation = self.fixture.aggregate(
      BehaviorRouteSplit.VALIDATION,
      ReplayRole.EXACT_STOCK,
      None,
      (10.0, 0.5, 0.8),
    )
    calls = []
    result = select_behavior_candidate(
      comparison,
      self.fixture.center,
      stock_validation,
      None,
      lambda *_: calls.append(True),
    )
    self.assertEqual(calls, [])
    self.assertEqual(result.disposition, BehaviorAggregateSelectionDisposition.STOCK_RETAINED)

    undefined = self.fixture.comparison(undefined=True)
    stock_contract = tuple(
      route.window_contract_sha256
      for route in undefined.exact_stock.identity.ordered_routes
    )
    self.assertTrue(all(
      tuple(
        route.window_contract_sha256
        for route in candidate.identity.ordered_routes
      ) == stock_contract
      for candidate in undefined.candidates
    ))
    result = select_behavior_candidate(
      undefined,
      self.fixture.center,
      *self.fixture.validation_references(),
      lambda *_: calls.append(True),
    )
    self.assertEqual(calls, [])
    self.assertEqual(result.disposition, BehaviorAggregateSelectionDisposition.INCUMBENT_RETAINED)


if __name__ == "__main__":
  unittest.main()

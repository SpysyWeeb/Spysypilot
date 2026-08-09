from __future__ import annotations

import json
import unittest

from openpilot.selfdrive.controls.lib.blatv2.behavior_coordinator import (
  BEHAVIOR_GATE_SPEC_SCHEMA_VERSION,
  BehaviorGateSpec,
  BehaviorLearningFinalization,
  BehaviorRouteEvidenceIdentity,
  CandidateGridBounds,
  FinalizationReason,
  ReplayArtifactIdentity,
  ReplayCoreIdentity,
  ReplayRole,
  RoutePartitionSpec,
  finalize_behavior_learning,
  partition_whole_routes,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import BehaviorSourceIdentity
from openpilot.selfdrive.controls.lib.blatv2.behavior_metrics import (
  BehaviorContract,
  BehaviorMetricConfig,
  BehaviorMetricName,
  MetricDisposition,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_policy import (
  PAIRED_ROUTE_UNCERTAINTY_METHOD,
  BehaviorPolicy,
  MetricGateRule,
  MetricPreference,
  PolicyEvaluation,
  PolicyMetric,
  PolicyStratumMetric,
)
from openpilot.selfdrive.controls.lib.blatv2.policy import ControllerPolicy


def metric_config() -> BehaviorMetricConfig:
  return BehaviorMetricConfig(
    burst_window_s=1.0,
    chatter_torque_rate_threshold_per_s=0.05,
    turn_in_crossing_fraction=0.5,
    release_crossing_fraction=0.9,
    correction_curvature_threshold_1pm=0.002,
    unused_headroom_threshold=0.1,
    growing_error_epsilon_1pm=0.0001,
    completion_delivered_fraction=0.95,
    minimum_samples=2,
    speed_nodes_mps=(0.0, 5.0, 10.0, 20.0, 30.0),
    maximum_route_windows_per_stratum=3,
  )


def gate_rules() -> tuple[MetricGateRule, ...]:
  return (
    MetricGateRule(
      metric_name=BehaviorMetricName.RAW_TORQUE_RATE_RMS.value,
      contract=BehaviorContract.SMOOTH,
      preference=MetricPreference.LOWER_IS_BETTER,
      noise_floor=0.1,
      margin_normalization=1.0,
      minimum_allowed=0.0,
      maximum_allowed=20.0,
      minimum_route_count=1,
      minimum_window_count=1,
      minimum_weighted_support=1.0,
      required_strata=("5:turn",),
    ),
    MetricGateRule(
      metric_name=BehaviorMetricName.SIGNED_TURN_IN_LAG_S.value,
      contract=BehaviorContract.SWIFT,
      preference=MetricPreference.LOWER_IS_BETTER,
      noise_floor=0.01,
      margin_normalization=0.1,
      minimum_allowed=0.0,
      maximum_allowed=1.0,
      minimum_route_count=1,
      minimum_window_count=1,
      minimum_weighted_support=1.0,
      required_strata=("5:turn",),
    ),
    MetricGateRule(
      metric_name=BehaviorMetricName.DELIVERED_FRACTION.value,
      contract=BehaviorContract.STRONG,
      preference=MetricPreference.HIGHER_IS_BETTER,
      noise_floor=0.01,
      margin_normalization=0.1,
      minimum_allowed=0.8,
      maximum_allowed=2.0,
      minimum_route_count=1,
      minimum_window_count=1,
      minimum_weighted_support=1.0,
      required_strata=("5:turn",),
    ),
  )


def gate_spec(*, validation_count: int | None = 2, validation_fraction: float | None = None) -> BehaviorGateSpec:
  return BehaviorGateSpec(
    schema_version=BEHAVIOR_GATE_SPEC_SCHEMA_VERSION,
    provenance="test-owned explicit behavior gate",
    metric_config=metric_config(),
    metric_rules=gate_rules(),
    target_metric_name=BehaviorMetricName.RAW_TORQUE_RATE_RMS.value,
    paired_uncertainty_method=PAIRED_ROUTE_UNCERTAINTY_METHOD,
    minimum_paired_route_count=2,
    candidate_grid=CandidateGridBounds(
      natural_frequency_log_offsets=(-0.1, 0.0, 0.1),
      damping_ratio_log_offsets=(0.0,),
      minimum_natural_frequency_per_s=5.0,
      maximum_natural_frequency_per_s=20.0,
      minimum_damping_ratio=0.5,
      maximum_damping_ratio=2.0,
    ),
    route_partition=RoutePartitionSpec(
      validation_fraction=validation_fraction,
      validation_route_count=validation_count,
      seed_identity_sha256="e" * 64,
    ),
  )


def source(*, controller_hash: str = "a") -> BehaviorSourceIdentity:
  return BehaviorSourceIdentity(
    controller_name="recorded-controller",
    controller_artifact_sha256=controller_hash * 64,
    source_openpilot_commit="b" * 40,
    opendbc_commit="c" * 40,
    panda_commit="d" * 40,
    evidence_schema_version=1,
  )


def routes() -> tuple[BehaviorRouteEvidenceIdentity, ...]:
  recorded_source = source()
  return tuple(
    BehaviorRouteEvidenceIdentity(
      route_id=f"route-{index}",
      route_evidence_sha256=f"{index + 1:x}" * 64,
      recorded_source=recorded_source,
    )
    for index in range(5)
  )


def core(name: str, token: str) -> ReplayCoreIdentity:
  return ReplayCoreIdentity(
    controller_name=name,
    core_artifact_sha256=token * 64,
    source_openpilot_commit="1" * 40,
    opendbc_commit="2" * 40,
    panda_commit="3" * 40,
  )


def policy_metric(
  name: BehaviorMetricName,
  value: float | None,
  route_ids: tuple[str, ...],
  exclusions: tuple[str, ...] = (),
) -> PolicyMetric:
  disposition = (
    MetricDisposition.DEFINED
    if value is not None
    else MetricDisposition.NOT_APPLICABLE
    if exclusions == ("no_hold_phase",)
    else MetricDisposition.COVERAGE_EXCLUDED
  )
  stratum_metric = PolicyStratumMetric(
    stratum="5:turn",
    value=value,
    disposition=disposition,
    exclusions=exclusions,
    route_count=len(route_ids),
    window_count=20,
    weighted_support=20.0,
    coverage_identity_sha256="f" * 64,
    physical_failure_window_ids=(),
    coverage_excluded_window_ids=("fixture/coverage-excluded",)
    if disposition is MetricDisposition.COVERAGE_EXCLUDED
    else (),
    not_applicable_window_ids=("fixture/not-applicable",)
    if disposition is MetricDisposition.NOT_APPLICABLE
    else (),
    route_values=(
      tuple((route_id, value) for route_id in route_ids)
      if value is not None
      else ()
    ),
  )
  return PolicyMetric(
    name=name.value,
    value=value,
    denominator=20 if value is not None else 0,
    exclusions=exclusions,
    route_count=len(route_ids),
    window_count=20,
    weighted_support=20.0,
    coverage_identity_sha256="f" * 64,
    strata=("5:turn",),
    stratum_metrics=(stratum_metric,),
    physical_failure_window_ids=(),
    route_values=(
      tuple((route_id, value) for route_id in route_ids)
      if value is not None
      else ()
    ),
  )


class ReplayFixture:
  def __init__(self, mode: str = "success") -> None:
    self.mode = mode
    self.calls: list[tuple[ReplayArtifactIdentity, BehaviorPolicy | None, tuple[str, ...]]] = []

  def __call__(
    self,
    identity: ReplayArtifactIdentity,
    policy: BehaviorPolicy | None,
    route_ids: tuple[str, ...],
  ) -> PolicyEvaluation:
    self.calls.append((identity, policy, route_ids))
    artifact_identity = identity.to_json()
    if self.mode == "identity_mismatch" and len(self.calls) == 1:
      artifact_identity = "not-the-requested-core"

    roughness: float | None = 10.0
    turn_lag = 0.5
    delivered = 0.9
    exclusion: tuple[str, ...] = ()
    if identity.role is ReplayRole.CANDIDATE:
      assert policy is not None
      if self.mode == "no_winner":
        roughness, turn_lag, delivered = 11.0, 0.6, 0.85
      elif self.mode == "undefined_training" and len(route_ids) == 3:
        roughness = None
        exclusion = ("unobservable",)
      elif self.mode == "validation_regression" and len(route_ids) == 2:
        roughness, turn_lag, delivered = 12.0, 0.4, 0.95
      elif policy.natural_frequency_per_s != 10.0:
        roughness, turn_lag, delivered = 8.0, 0.4, 0.95

    metrics = [
      policy_metric(BehaviorMetricName.RAW_TORQUE_RATE_RMS, roughness, route_ids, exclusion),
      policy_metric(BehaviorMetricName.SIGNED_TURN_IN_LAG_S, turn_lag, route_ids),
      policy_metric(BehaviorMetricName.DELIVERED_FRACTION, delivered, route_ids),
    ]
    if self.mode == "undefined_informational":
      metrics.append(policy_metric(
        BehaviorMetricName.HOLD_BIAS_1PM,
        None,
        route_ids,
        ("no_hold_phase",),
      ))
    return PolicyEvaluation(
      artifact_identity=artifact_identity,
      policy=policy,
      route_ids=route_ids,
      metrics=tuple(metrics),
    )


def finalize(callback: ReplayFixture, *, route_set=None):
  return finalize_behavior_learning(
    gate_spec=gate_spec(),
    routes=routes() if route_set is None else route_set,
    accepted_policy=BehaviorPolicy(10.0, 1.0),
    search_center_policy=BehaviorPolicy(10.0, 1.0),
    exact_stock_core=core("stock", "4"),
    accepted_core=core("accepted", "5"),
    candidate_core=core("candidate", "6"),
    replay_evaluate=callback,
  )


class TestBehaviorCoordinator(unittest.TestCase):
  def test_partition_is_stable_whole_route_and_supports_explicit_fraction(self):
    count_partition = partition_whole_routes(routes(), gate_spec().route_partition)
    reordered = partition_whole_routes(reversed(routes()), gate_spec().route_partition)
    fraction_partition = partition_whole_routes(
      routes(),
      gate_spec(validation_count=None, validation_fraction=0.4).route_partition,
    )

    self.assertEqual(count_partition.to_json(), reordered.to_json())
    self.assertEqual(count_partition.validation_route_ids, fraction_partition.validation_route_ids)
    self.assertEqual(len(count_partition.validation_routes), 2)
    self.assertFalse(set(count_partition.training_route_ids) & set(count_partition.validation_route_ids))
    self.assertEqual(
      set(count_partition.training_route_ids) | set(count_partition.validation_route_ids),
      {route.route_id for route in routes()},
    )
    self.assertFalse(hasattr(count_partition, "training_frames"))

  def test_gate_spec_is_strict_versioned_and_all_three_contracts_are_mandatory(self):
    spec = gate_spec()
    decoded = BehaviorGateSpec.from_json(spec.to_json())
    self.assertEqual(decoded.to_json(), spec.to_json())
    self.assertEqual(decoded.sha256, spec.sha256)

    old = json.loads(spec.to_json())
    old["schemaVersion"] = 0
    with self.assertRaisesRegex(ValueError, "incompatible"):
      BehaviorGateSpec.from_json(json.dumps(old))
    unknown = json.loads(spec.to_json())
    unknown["hiddenTune"] = 1.0
    with self.assertRaisesRegex(ValueError, "keys"):
      BehaviorGateSpec.from_json(json.dumps(unknown))
    with self.assertRaisesRegex(ValueError, "Smooth, Swift, and Strong"):
      BehaviorGateSpec(
        schema_version=BEHAVIOR_GATE_SPEC_SCHEMA_VERSION,
        provenance="incomplete",
        metric_config=metric_config(),
        metric_rules=gate_rules()[:2],
        target_metric_name=BehaviorMetricName.RAW_TORQUE_RATE_RMS.value,
        paired_uncertainty_method=PAIRED_ROUTE_UNCERTAINTY_METHOD,
        minimum_paired_route_count=2,
        candidate_grid=spec.candidate_grid,
        route_partition=spec.route_partition,
      )

  def test_success_freezes_exactly_one_winner_and_replays_only_it_heldout(self):
    callback = ReplayFixture()
    result = finalize(callback)

    self.assertTrue(result.passed)
    self.assertTrue(result.smooth_passed)
    self.assertTrue(result.swift_passed)
    self.assertTrue(result.strong_passed)
    self.assertTrue(result.target_materially_improved)
    self.assertIsNotNone(result.behavior_selection_sha256)
    training_candidate_calls = [
      call
      for call in callback.calls
      if call[0].role is ReplayRole.CANDIDATE and len(call[2]) == 3
    ]
    validation_candidate_calls = [
      call
      for call in callback.calls
      if call[0].role is ReplayRole.CANDIDATE and len(call[2]) == 2
    ]
    self.assertEqual(len(training_candidate_calls), 3)
    self.assertEqual(len(validation_candidate_calls), 1)
    self.assertEqual(validation_candidate_calls[0][1], result.final_behavior_policy)
    self.assertIn("coreArtifactSha256", validation_candidate_calls[0][0].to_json())

  def test_exact_stock_bootstrap_uses_distinct_search_seed_without_faking_acceptance(self):
    callback = ReplayFixture()
    stock_core = core("stock", "4")
    search_seed = BehaviorPolicy(10.0, 1.0)
    result = finalize_behavior_learning(
      gate_spec=gate_spec(),
      routes=routes(),
      accepted_policy=None,
      search_center_policy=search_seed,
      exact_stock_core=stock_core,
      accepted_core=None,
      candidate_core=core("candidate", "6"),
      replay_evaluate=callback,
    )

    self.assertTrue(result.passed)
    current_calls = [
      call for call in callback.calls
      if call[0].role is ReplayRole.CURRENTLY_ACCEPTED
    ]
    self.assertTrue(current_calls)
    self.assertTrue(all(call[0].core == stock_core for call in current_calls))
    self.assertTrue(all(call[1] is None for call in current_calls))
    candidate_calls = [
      call for call in callback.calls
      if call[0].role is ReplayRole.CANDIDATE
    ]
    self.assertIn(search_seed, {call[1] for call in candidate_calls})

  def test_bootstrap_and_nonbootstrap_identity_ambiguity_fail_closed(self):
    bootstrap = finalize_behavior_learning(
      gate_spec=gate_spec(),
      routes=routes(),
      accepted_policy=None,
      search_center_policy=BehaviorPolicy(10.0, 1.0),
      exact_stock_core=core("stock", "4"),
      accepted_core=core("not-stock", "5"),
      candidate_core=core("candidate", "6"),
      replay_evaluate=ReplayFixture(),
    )
    shifted = finalize_behavior_learning(
      gate_spec=gate_spec(),
      routes=routes(),
      accepted_policy=BehaviorPolicy(10.0, 1.0),
      search_center_policy=BehaviorPolicy(11.0, 1.0),
      exact_stock_core=core("stock", "4"),
      accepted_core=core("accepted", "5"),
      candidate_core=core("candidate", "6"),
      replay_evaluate=ReplayFixture(),
    )

    self.assertEqual(
      bootstrap.reasons,
      (FinalizationReason.BOOTSTRAP_IDENTITY_INVALID,),
    )
    self.assertEqual(
      shifted.reasons,
      (FinalizationReason.BOOTSTRAP_IDENTITY_INVALID,),
    )

  def test_no_training_winner_and_undefined_metrics_emit_no_policy(self):
    no_winner = finalize(ReplayFixture("no_winner"))
    undefined = finalize(ReplayFixture("undefined_training"))

    self.assertFalse(no_winner.passed)
    self.assertEqual(no_winner.reasons, (FinalizationReason.NO_TRAINING_WINNER,))
    self.assertIsNone(no_winner.final_behavior_policy)
    self.assertIsNone(no_winner.behavior_selection_sha256)
    self.assertFalse(undefined.passed)
    self.assertEqual(undefined.reasons, (FinalizationReason.UNDEFINED_TRAINING_METRIC,))
    self.assertIsNone(undefined.final_behavior_policy)

  def test_heldout_regression_fails_closed_and_exposes_independent_contracts(self):
    result = finalize(ReplayFixture("validation_regression"))

    self.assertFalse(result.passed)
    self.assertEqual(result.reasons, (FinalizationReason.SMOOTH_CROSS_FIT_REGRESSION,))
    self.assertFalse(result.smooth_passed)
    self.assertTrue(result.swift_passed)
    self.assertTrue(result.strong_passed)
    self.assertFalse(result.target_materially_improved)
    self.assertIsNotNone(result.training_selection_sha256)
    self.assertIsNotNone(result.validation_sha256)
    self.assertIsNone(result.final_behavior_policy)

  def test_undefined_informational_metric_does_not_invent_a_gate(self):
    result = finalize(ReplayFixture("undefined_informational"))

    self.assertTrue(result.passed)
    self.assertEqual(result.reasons, (FinalizationReason.PASSED,))

  def test_route_source_and_replay_identity_mismatches_fail_closed(self):
    mixed = list(routes())
    mixed[-1] = BehaviorRouteEvidenceIdentity(
      route_id=mixed[-1].route_id,
      route_evidence_sha256=mixed[-1].route_evidence_sha256,
      recorded_source=source(controller_hash="f"),
    )
    mixed_result = finalize(ReplayFixture(), route_set=tuple(mixed))
    replay_result = finalize(ReplayFixture("identity_mismatch"))

    self.assertEqual(
      mixed_result.reasons,
      (FinalizationReason.MIXED_ROUTE_SOURCE_IDENTITIES,),
    )
    self.assertIsNone(mixed_result.final_behavior_policy)
    self.assertEqual(
      replay_result.reasons,
      (FinalizationReason.EVALUATION_IDENTITY_MISMATCH,),
    )
    self.assertIsNone(replay_result.final_behavior_policy)

  def test_finalization_is_byte_deterministic_and_maps_without_observer_mutation(self):
    first = finalize(ReplayFixture())
    second = finalize(ReplayFixture())
    template = ControllerPolicy(
      revision=7,
      provenance="accepted template",
      provisional=False,
      natural_frequency_per_s=10.0,
      damping_ratio=1.0,
      observer_time_constant_s=0.5,
      observer_max_abs_disturbance_torque=0.2,
    )
    self.assertIsNotNone(first.final_behavior_policy)
    mapped = first.final_behavior_policy.into_controller_policy(template)

    self.assertEqual(first.to_json(), second.to_json())
    self.assertEqual(first.sha256, second.sha256)
    decoded = BehaviorLearningFinalization.from_json(first.to_json())
    self.assertEqual(decoded.to_json(), first.to_json())
    unknown = json.loads(first.to_json())
    unknown["hiddenApproval"] = True
    with self.assertRaisesRegex(ValueError, "keys"):
      BehaviorLearningFinalization.from_json(json.dumps(unknown))
    old = json.loads(first.to_json())
    old["schemaVersion"] = 0
    with self.assertRaisesRegex(ValueError, "incompatible"):
      BehaviorLearningFinalization.from_json(json.dumps(old))
    self.assertEqual(mapped.observer_time_constant_s, template.observer_time_constant_s)
    self.assertEqual(
      mapped.observer_max_abs_disturbance_torque,
      template.observer_max_abs_disturbance_torque,
    )
    self.assertEqual(mapped.revision, template.revision)


if __name__ == "__main__":
  unittest.main()

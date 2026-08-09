from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
import hashlib
import inspect
import json
import math
import unittest

from openpilot.selfdrive.controls.lib.blatv2 import behavior_evidence, behavior_metrics, behavior_policy
from openpilot.selfdrive.controls.lib.blatv2.behavior_metrics import (
  BehaviorContract,
  BehaviorMetricName,
  MetricDisposition,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_coordinator import (
  ReplayArtifactIdentity,
  ReplayCoreIdentity,
  ReplayRole,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_policy import (
  PAIRED_ROUTE_UNCERTAINTY_METHOD,
  BehaviorPolicy,
  MetricGateRule,
  MetricPreference,
  PolicyEvaluation,
  PolicyGridSpec,
  PolicyMetric,
  PolicyStratumMetric,
  build_candidate_grid,
  evaluate_candidate,
  select_training_winner,
  validate_frozen_winner,
)
from openpilot.selfdrive.controls.lib.blatv2.policy import ControllerPolicy


UNCERTAINTY_ARGS = (PAIRED_ROUTE_UNCERTAINTY_METHOD, 2)


def rules() -> tuple[MetricGateRule, ...]:
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


def evaluation(
  identity: str,
  policy: BehaviorPolicy | None,
  route_ids: tuple[str, ...],
  roughness: float,
  signed_turn_lag: float,
  delivered_fraction: float,
) -> PolicyEvaluation:
  role = (
    ReplayRole.EXACT_STOCK
    if identity.startswith("stock")
    else ReplayRole.CURRENTLY_ACCEPTED
    if identity.startswith("accepted")
    else ReplayRole.CANDIDATE
  )
  core_token = hashlib.sha256(identity.encode()).hexdigest()
  artifact = ReplayArtifactIdentity.compose(
    role,
    ReplayCoreIdentity(
      controller_name=identity,
      core_artifact_sha256=core_token,
      source_openpilot_commit="1" * 40,
      opendbc_commit="2" * 40,
      panda_commit="3" * 40,
    ),
    policy,
  )
  return PolicyEvaluation(
    artifact_identity=artifact.to_json(),
    policy=policy,
    route_ids=route_ids,
    metrics=(
      policy_metric(BehaviorMetricName.RAW_TORQUE_RATE_RMS, roughness, route_ids=route_ids),
      policy_metric(BehaviorMetricName.SIGNED_TURN_IN_LAG_S, signed_turn_lag, route_ids=route_ids),
      policy_metric(BehaviorMetricName.DELIVERED_FRACTION, delivered_fraction, route_ids=route_ids),
    ),
  )


def policy_metric(
  name: BehaviorMetricName,
  value: float | None,
  *,
  route_ids: tuple[str, ...] = ("route-a", "route-b"),
  coverage: str = "f" * 64,
  physical_failures: tuple[str, ...] = (),
) -> PolicyMetric:
  stratum_metric = PolicyStratumMetric(
    stratum="5:turn",
    value=value,
    disposition=(
      MetricDisposition.DEFINED
      if value is not None
      else MetricDisposition.PHYSICAL_UNSCOREABLE
      if physical_failures
      else MetricDisposition.COVERAGE_EXCLUDED
    ),
    exclusions=() if value is not None else ("unscoreable",),
    route_count=len(route_ids),
    window_count=10,
    weighted_support=10.0,
    coverage_identity_sha256=coverage,
    physical_failure_window_ids=physical_failures,
    coverage_excluded_window_ids=(
      ("fixture/coverage-excluded",)
      if value is None and not physical_failures
      else ()
    ),
    not_applicable_window_ids=(),
    route_values=(
      tuple((route_id, value) for route_id in route_ids)
      if value is not None
      else ()
    ),
  )
  return PolicyMetric(
    name.value,
    value,
    10 if value is not None else 0,
    () if value is not None else ("unscoreable",),
    route_count=len(route_ids),
    window_count=10,
    weighted_support=10.0,
    coverage_identity_sha256=coverage,
    strata=("5:turn",),
    stratum_metrics=(stratum_metric,),
    physical_failure_window_ids=physical_failures,
    route_values=(
      tuple((route_id, value) for route_id in route_ids)
      if value is not None
      else ()
    ),
  )


def with_metric_strata(
  source: PolicyEvaluation,
  name: BehaviorMetricName,
  aggregate_value: float,
  values: tuple[tuple[str, float], ...],
) -> PolicyEvaluation:
  metrics = []
  for metric in source.metrics:
    if metric.name != name.value:
      metrics.append(metric)
      continue
    base = metric.stratum_metrics[0]
    strata = tuple(
      replace(
        base,
        stratum=stratum,
        value=value,
        exclusions=(),
        route_values=tuple(
          (route_id, value) for route_id in source.route_ids
        ),
      )
      for stratum, value in values
    )
    metrics.append(replace(
      metric,
      value=aggregate_value,
      route_values=tuple(
        (route_id, aggregate_value) for route_id in source.route_ids
      ),
      strata=tuple(stratum.stratum for stratum in strata),
      stratum_metrics=strata,
    ))
  return replace(source, metrics=tuple(metrics))


def with_extra_metric_stratum(
  source: PolicyEvaluation,
  name: BehaviorMetricName,
  extra: PolicyStratumMetric,
) -> PolicyEvaluation:
  return replace(
    source,
    metrics=tuple(
      replace(
        metric,
        strata=tuple(sorted((*metric.strata, extra.stratum))),
        stratum_metrics=tuple(sorted(
          (*metric.stratum_metrics, extra),
          key=lambda value: value.stratum,
        )),
      )
      if metric.name == name.value
      else metric
      for metric in source.metrics
    ),
  )


def grid() -> tuple:
  return build_candidate_grid(PolicyGridSpec(
    incumbent=BehaviorPolicy(10.0, 1.0),
    natural_frequency_log_offsets=(-0.1, 0.0, 0.1),
    damping_ratio_log_offsets=(0.0,),
    minimum_natural_frequency_per_s=5.0,
    maximum_natural_frequency_per_s=20.0,
    minimum_damping_ratio=0.5,
    maximum_damping_ratio=2.0,
  ))


class TestBehaviorPolicy(unittest.TestCase):
  def test_candidate_grid_has_cross_architecture_golden_identity(self):
    payload = [
      (
        candidate.canonical_index,
        candidate.policy.natural_frequency_per_s.hex(),
        candidate.policy.damping_ratio.hex(),
        candidate.squared_log_displacement.hex(),
      )
      for candidate in grid()
    ]
    encoded = json.dumps(payload, separators=(",", ":"))
    self.assertEqual(
      hashlib.sha256(encoded.encode()).hexdigest(),
      "78aa987e48c09d59ba85e481bb3d61d3e7ed254e43e7cea5417103b67fbbf9ec",
    )

  def test_gate_registry_rejects_unknown_or_misclassified_metrics(self):
    base = rules()[0]
    with self.assertRaisesRegex(ValueError, "not registered"):
      replace(base, metric_name="invented_smoothness")
    with self.assertRaisesRegex(ValueError, "wrong behavior contract"):
      replace(base, contract=BehaviorContract.STRONG)

  def test_policy_has_exactly_two_dials_and_preserves_controller_observer(self):
    controller = ControllerPolicy(
      revision=9,
      provenance="accepted template",
      provisional=False,
      natural_frequency_per_s=10.0,
      damping_ratio=1.0,
      observer_time_constant_s=0.5,
      observer_max_abs_disturbance_torque=0.2,
    )
    behavior = BehaviorPolicy.from_controller_policy(controller)
    changed = BehaviorPolicy(8.0, 0.9).into_controller_policy(controller)

    self.assertEqual(set(BehaviorPolicy.__dataclass_fields__), {"natural_frequency_per_s", "damping_ratio"})
    self.assertEqual(behavior, BehaviorPolicy(10.0, 1.0))
    self.assertEqual(changed.natural_frequency_per_s, 8.0)
    self.assertEqual(changed.damping_ratio, 0.9)
    self.assertEqual(changed.observer_time_constant_s, controller.observer_time_constant_s)
    self.assertEqual(
      changed.observer_max_abs_disturbance_torque,
      controller.observer_max_abs_disturbance_torque,
    )
    self.assertEqual(changed.revision, controller.revision)

  def test_early_turn_in_rejects_candidate_even_if_other_contracts_improve(self):
    candidates = grid()
    candidate = candidates[0]
    stock = evaluation("stock", None, ("train-a", "train-b"), 10.0, 0.5, 0.9)
    accepted = evaluation("accepted", candidates[1].policy, ("train-a", "train-b"), 10.0, 0.5, 0.9)
    proposed = evaluation("candidate", candidate.policy, ("train-a", "train-b"), 1.0, -0.01, 1.1)

    verdict = evaluate_candidate(
      candidate,
      proposed,
      stock,
      accepted,
      rules(),
      BehaviorMetricName.RAW_TORQUE_RATE_RMS.value,
      *UNCERTAINTY_ARGS,
    )

    self.assertFalse(verdict.passed)
    swift = next(contract for contract in verdict.contracts if contract.contract is BehaviorContract.SWIFT)
    self.assertFalse(swift.passed)
    self.assertIn("5:turn:below_absolute_minimum", swift.metrics[0].reasons)

  def test_physical_failure_and_coverage_mismatch_fail_before_averaging(self):
    candidates = grid()
    candidate = candidates[0]
    stock = evaluation("stock", None, ("train-a", "train-b"), 10.0, 0.5, 0.9)
    accepted = evaluation("accepted", candidates[1].policy, ("train-a", "train-b"), 10.0, 0.5, 0.9)
    proposed = evaluation("candidate", candidate.policy, ("train-a", "train-b"), 1.0, 0.4, 1.0)
    turn_name = BehaviorMetricName.SIGNED_TURN_IN_LAG_S.value
    failed_metrics = tuple(
      replace(
        metric,
        value=None,
        denominator=0,
        exclusions=("delivered_crossing_unobservable",),
        physical_failure_window_ids=("route-a/hard-turn",),
        route_values=(),
        stratum_metrics=(replace(
          metric.stratum_metrics[0],
          value=None,
          disposition=MetricDisposition.PHYSICAL_UNSCOREABLE,
          exclusions=("delivered_crossing_unobservable",),
          physical_failure_window_ids=("route-a/hard-turn",),
          route_values=(),
        ),),
      )
      if metric.name == turn_name else metric
      for metric in proposed.metrics
    )
    failed = replace(proposed, metrics=failed_metrics)
    verdict = evaluate_candidate(
      candidate,
      failed,
      stock,
      accepted,
      rules(),
      BehaviorMetricName.RAW_TORQUE_RATE_RMS.value,
      *UNCERTAINTY_ARGS,
    )
    swift = next(value for value in verdict.contracts if value.contract is BehaviorContract.SWIFT)
    self.assertFalse(swift.passed)
    self.assertTrue(any("physical_unscoreable" in reason for reason in swift.metrics[0].reasons))

    mismatched_metrics = tuple(
      replace(
        metric,
        coverage_identity_sha256="e" * 64,
        stratum_metrics=(replace(
          metric.stratum_metrics[0],
          coverage_identity_sha256="e" * 64,
        ),),
      )
      if metric.name == BehaviorMetricName.RAW_TORQUE_RATE_RMS.value else metric
      for metric in proposed.metrics
    )
    mismatched = replace(proposed, metrics=mismatched_metrics)
    mismatch_verdict = evaluate_candidate(
      candidate,
      mismatched,
      stock,
      accepted,
      rules(),
      BehaviorMetricName.RAW_TORQUE_RATE_RMS.value,
      *UNCERTAINTY_ARGS,
    )
    smooth = next(value for value in mismatch_verdict.contracts if value.contract is BehaviorContract.SMOOTH)
    self.assertIn("5:turn:candidate_reference_coverage_mismatch", smooth.metrics[0].reasons)

  def test_sparse_easy_speed_evidence_fails_coverage_gate(self):
    candidates = grid()
    candidate = candidates[0]
    stock = evaluation("stock", None, ("train-a", "train-b"), 10.0, 0.5, 0.9)
    accepted = evaluation("accepted", candidates[1].policy, ("train-a", "train-b"), 10.0, 0.5, 0.9)
    proposed = evaluation("candidate", candidate.policy, ("train-a", "train-b"), 1.0, 0.4, 1.0)
    sparse = replace(
      rules()[0],
      minimum_route_count=3,
      required_strata=("0:turn", "5:turn"),
    )
    verdict = evaluate_candidate(
      candidate,
      proposed,
      stock,
      accepted,
      (sparse, *rules()[1:]),
      BehaviorMetricName.RAW_TORQUE_RATE_RMS.value,
      *UNCERTAINTY_ARGS,
    )
    smooth = next(value for value in verdict.contracts if value.contract is BehaviorContract.SMOOTH)
    self.assertFalse(smooth.passed)
    self.assertTrue(any("route_coverage" in reason for reason in smooth.metrics[0].reasons))
    self.assertTrue(any("missing_required_stratum" in reason for reason in smooth.metrics[0].reasons))

  def test_bad_populated_stratum_vetoes_good_balanced_aggregate(self):
    candidates = grid()
    candidate = candidates[0]
    route_ids = ("train-a", "train-b")
    stock = with_metric_strata(
      evaluation("stock", None, route_ids, 15.0, 0.5, 0.9),
      BehaviorMetricName.RAW_TORQUE_RATE_RMS,
      15.0,
      (("10:turn", 15.0), ("5:turn", 15.0)),
    )
    accepted = with_metric_strata(
      evaluation("accepted", candidates[1].policy, route_ids, 15.0, 0.5, 0.9),
      BehaviorMetricName.RAW_TORQUE_RATE_RMS,
      15.0,
      (("10:turn", 15.0), ("5:turn", 15.0)),
    )
    proposed = with_metric_strata(
      evaluation("candidate", candidate.policy, route_ids, 13.0, 0.4, 1.0),
      BehaviorMetricName.RAW_TORQUE_RATE_RMS,
      13.0,
      (("10:turn", 25.0), ("5:turn", 1.0)),
    )
    verdict = evaluate_candidate(
      candidate,
      proposed,
      stock,
      accepted,
      rules(),
      BehaviorMetricName.RAW_TORQUE_RATE_RMS.value,
      *UNCERTAINTY_ARGS,
    )
    smooth = next(
      contract for contract in verdict.contracts
      if contract.contract is BehaviorContract.SMOOTH
    )
    self.assertFalse(verdict.passed)
    self.assertTrue(verdict.target_materially_improved)
    self.assertIn("10:turn:above_absolute_maximum", smooth.metrics[0].reasons)
    self.assertIn("10:turn:regressed_vs_stock", smooth.metrics[0].reasons)

  def test_unsupported_required_stratum_fails_current_support_rule(self):
    candidates = grid()
    candidate = candidates[0]
    route_ids = ("train-a", "train-b")
    stock = evaluation("stock", None, route_ids, 10.0, 0.5, 0.9)
    accepted = evaluation("accepted", candidates[1].policy, route_ids, 10.0, 0.5, 0.9)
    proposed = evaluation("candidate", candidate.policy, route_ids, 1.0, 0.4, 1.0)
    proposed = replace(
      proposed,
      metrics=tuple(
        replace(
          metric,
          stratum_metrics=(replace(
            metric.stratum_metrics[0],
            weighted_support=0.5,
          ),),
        )
        if metric.name == BehaviorMetricName.RAW_TORQUE_RATE_RMS.value
        else metric
        for metric in proposed.metrics
      ),
    )
    verdict = evaluate_candidate(
      candidate,
      proposed,
      stock,
      accepted,
      rules(),
      BehaviorMetricName.RAW_TORQUE_RATE_RMS.value,
      *UNCERTAINTY_ARGS,
    )
    smooth = next(
      contract for contract in verdict.contracts
      if contract.contract is BehaviorContract.SMOOTH
    )
    self.assertFalse(smooth.passed)
    self.assertIn(
      "5:turn:candidate_weighted_support_below_minimum",
      smooth.metrics[0].reasons,
    )

  def test_nonrequired_request_off_physical_stratum_is_not_skipped(self):
    candidates = grid()
    route_ids = ("train-a", "train-b")
    stock = evaluation("stock", None, route_ids, 10.0, 0.5, 0.9)
    accepted = evaluation("accepted", candidates[1].policy, route_ids, 10.0, 0.5, 0.9)
    proposed = evaluation("candidate", candidates[0].policy, route_ids, 8.0, 0.4, 1.0)
    name = BehaviorMetricName.RAW_TORQUE_RATE_RMS

    def unavailable(source: PolicyEvaluation, disposition: MetricDisposition) -> PolicyStratumMetric:
      base = source.metric(name.value).stratum_metrics[0]
      physical = disposition is MetricDisposition.PHYSICAL_UNSCOREABLE
      return replace(
        base,
        stratum="10:turn",
        value=None,
        disposition=disposition,
        exclusions=(
          "request_suppressed_maximum_authority"
          if physical
          else "not_applicable_in_route_stratum",
        ),
        route_count=0,
        window_count=2 if physical else 0,
        weighted_support=0.0,
        physical_failure_window_ids=("train-a/request-off",) if physical else (),
        coverage_excluded_window_ids=(),
        not_applicable_window_ids=("train-a/not-applicable",) if not physical else (),
        route_values=(),
      )

    proposed = with_extra_metric_stratum(
      proposed,
      name,
      unavailable(proposed, MetricDisposition.PHYSICAL_UNSCOREABLE),
    )
    stock = with_extra_metric_stratum(
      stock,
      name,
      unavailable(stock, MetricDisposition.NOT_APPLICABLE),
    )
    accepted = with_extra_metric_stratum(
      accepted,
      name,
      unavailable(accepted, MetricDisposition.NOT_APPLICABLE),
    )

    verdict = evaluate_candidate(
      candidates[0], proposed, stock, accepted, rules(), name.value, *UNCERTAINTY_ARGS,
    )
    smooth = next(value for value in verdict.contracts if value.contract is BehaviorContract.SMOOTH)
    self.assertFalse(verdict.passed)
    self.assertIn(
      "10:turn:candidate_physical_unscoreable:train-a/request-off",
      smooth.metrics[0].reasons,
    )

  def test_nonrequired_low_reference_coverage_is_not_skipped(self):
    candidates = grid()
    route_ids = ("train-a", "train-b")
    name = BehaviorMetricName.RAW_TORQUE_RATE_RMS
    stock = with_metric_strata(
      evaluation("stock", None, route_ids, 10.0, 0.5, 0.9),
      name,
      10.0,
      (("10:turn", 10.0), ("5:turn", 10.0)),
    )
    accepted = with_metric_strata(
      evaluation("accepted", candidates[1].policy, route_ids, 10.0, 0.5, 0.9),
      name,
      10.0,
      (("10:turn", 10.0), ("5:turn", 10.0)),
    )
    proposed = with_metric_strata(
      evaluation("candidate", candidates[0].policy, route_ids, 8.0, 0.4, 1.0),
      name,
      8.0,
      (("10:turn", 8.0), ("5:turn", 8.0)),
    )
    low_reference = replace(
      stock.metric(name.value).stratum_metrics[0],
      value=None,
      disposition=MetricDisposition.COVERAGE_EXCLUDED,
      exclusions=("low_reference_coverage",),
      route_count=0,
      window_count=0,
      weighted_support=0.0,
      coverage_excluded_window_ids=("train-a/low-reference",),
      not_applicable_window_ids=(),
      route_values=(),
    )
    stock = replace(
      stock,
      metrics=tuple(
        replace(
          metric,
          stratum_metrics=(low_reference, metric.stratum_metrics[1]),
        )
        if metric.name == name.value
        else metric
        for metric in stock.metrics
      ),
    )

    verdict = evaluate_candidate(
      candidates[0], proposed, stock, accepted, rules(), name.value, *UNCERTAINTY_ARGS,
    )
    smooth = next(value for value in verdict.contracts if value.contract is BehaviorContract.SMOOTH)
    self.assertFalse(verdict.passed)
    self.assertIn("10:turn:stock_route_coverage_below_minimum", smooth.metrics[0].reasons)
    self.assertIn(
      "10:turn:stock_metric_undefined:low_reference_coverage",
      smooth.metrics[0].reasons,
    )

  def test_stratum_serialization_is_deterministic_and_tamper_closed(self):
    candidate = grid()[0]
    source = evaluation(
      "candidate",
      candidate.policy,
      ("train-a", "train-b"),
      8.0,
      0.4,
      0.95,
    )
    encoded = source.to_json()
    decoded = PolicyEvaluation.from_dict(json.loads(encoded))
    self.assertEqual(decoded.to_json(), encoded)
    self.assertEqual(decoded.sha256, source.sha256)
    base = source.metrics[0].stratum_metrics[0]
    undefined = tuple(
      replace(
        base,
        value=None,
        disposition=disposition,
        exclusions=(reason,),
        route_count=0,
        window_count=2 if disposition is not MetricDisposition.NOT_APPLICABLE else 0,
        weighted_support=0.0,
        physical_failure_window_ids=("train-a/request-off",)
        if disposition is MetricDisposition.PHYSICAL_UNSCOREABLE
        else (),
        coverage_excluded_window_ids=("train-a/coverage-excluded",)
        if disposition is MetricDisposition.COVERAGE_EXCLUDED
        else (),
        not_applicable_window_ids=("train-a/not-applicable",)
        if disposition is MetricDisposition.NOT_APPLICABLE
        else (),
        route_values=(),
      )
      for disposition, reason in (
        (MetricDisposition.NOT_APPLICABLE, "not_applicable_in_route_stratum"),
        (MetricDisposition.COVERAGE_EXCLUDED, "low_reference_coverage"),
        (MetricDisposition.PHYSICAL_UNSCOREABLE, "request_suppressed_maximum_authority"),
      )
    )
    for metric in (base, *undefined):
      self.assertEqual(PolicyStratumMetric.from_dict(metric.to_dict()), metric)
    self.assertEqual(
      {metric.to_dict()["disposition"] for metric in (base, *undefined)},
      {disposition.value for disposition in MetricDisposition},
    )

    tampered = json.loads(encoded)
    tampered["metrics"][0]["stratumMetrics"][0]["routeCount"] += 1
    with self.assertRaisesRegex(ValueError, "inconsistent evidence"):
      PolicyEvaluation.from_dict(tampered)

    wrong_disposition = json.loads(encoded)
    wrong_disposition["metrics"][0]["stratumMetrics"][0]["disposition"] = (
      MetricDisposition.COVERAGE_EXCLUDED.value
    )
    with self.assertRaisesRegex(ValueError, "disposition disagrees with window evidence"):
      PolicyEvaluation.from_dict(wrong_disposition)

    for source_disposition, target_disposition in (
      (MetricDisposition.NOT_APPLICABLE, MetricDisposition.COVERAGE_EXCLUDED),
      (MetricDisposition.COVERAGE_EXCLUDED, MetricDisposition.NOT_APPLICABLE),
    ):
      forged = next(
        metric.to_dict()
        for metric in undefined
        if metric.disposition is source_disposition
      )
      forged["disposition"] = target_disposition.value
      with self.assertRaisesRegex(ValueError, "disposition disagrees with window evidence"):
        PolicyStratumMetric.from_dict(forged)

    missing = json.loads(encoded)
    del missing["metrics"][0]["stratumMetrics"][0]["disposition"]
    with self.assertRaisesRegex(ValueError, "keys do not match schema"):
      PolicyEvaluation.from_dict(missing)

  def test_training_search_is_deterministic_and_uses_canonical_tie_break(self):
    candidates = grid()
    stock = evaluation("stock", None, ("train-a", "train-b"), 10.0, 0.5, 0.9)
    accepted = evaluation("accepted", candidates[1].policy, ("train-a", "train-b"), 10.0, 0.5, 0.9)
    candidate_evaluations = (
      evaluation("lower", candidates[0].policy, ("train-a", "train-b"), 8.0, 0.4, 0.95),
      evaluation("incumbent", candidates[1].policy, ("train-a", "train-b"), 10.0, 0.5, 0.9),
      evaluation("upper", candidates[2].policy, ("train-a", "train-b"), 8.0, 0.4, 0.95),
    )

    first = select_training_winner(
      candidates,
      reversed(candidate_evaluations),
      stock,
      accepted,
      rules(),
      BehaviorMetricName.RAW_TORQUE_RATE_RMS.value,
      *UNCERTAINTY_ARGS,
    )
    second = select_training_winner(
      candidates,
      candidate_evaluations,
      stock,
      accepted,
      rules(),
      BehaviorMetricName.RAW_TORQUE_RATE_RMS.value,
      *UNCERTAINTY_ARGS,
    )

    self.assertIsNotNone(first)
    self.assertEqual(first, second)
    self.assertEqual(first.winner.canonical_index, candidates[0].canonical_index)
    self.assertEqual(first.to_json(), second.to_json())
    self.assertEqual(first.sha256, second.sha256)

  def test_paired_route_envelope_rejects_dust_and_inconsistent_mean_gain(self):
    candidates = grid()
    candidate = candidates[0]
    route_ids = ("train-a", "train-b")
    stock = evaluation("stock", None, route_ids, 10.0, 0.5, 0.9)
    accepted = evaluation("accepted", candidates[1].policy, route_ids, 10.0, 0.5, 0.9)
    proposed = evaluation("candidate", candidate.policy, route_ids, 9.0, 0.4, 0.95)

    # Mean roughness improves by one, but only one route improves.  The
    # observed envelope lower bound is exactly zero, so the target is not a
    # material improvement even though its point aggregate is smaller.
    roughness_name = BehaviorMetricName.RAW_TORQUE_RATE_RMS.value
    inconsistent = replace(
      proposed,
      metrics=tuple(
        replace(
          metric,
          route_values=(("train-a", 8.0), ("train-b", 10.0)),
        )
        if metric.name == roughness_name
        else metric
        for metric in proposed.metrics
      ),
    )
    verdict = evaluate_candidate(
      candidate,
      inconsistent,
      stock,
      accepted,
      rules(),
      roughness_name,
      *UNCERTAINTY_ARGS,
    )
    smooth = next(
      contract
      for contract in verdict.contracts
      if contract.contract is BehaviorContract.SMOOTH
    )
    diagnostic = smooth.metrics[0].paired_against_accepted
    assert diagnostic is not None
    self.assertEqual(diagnostic.mean, 1.0)
    self.assertEqual(diagnostic.uncertainty, 1.0)
    self.assertEqual(diagnostic.lower, 0.0)
    self.assertFalse(verdict.target_materially_improved)
    self.assertTrue(verdict.passed)

    dust = math.nextafter(10.0, math.inf)
    dust_candidate = replace(
      proposed,
      metrics=tuple(
        replace(
          metric,
          value=dust,
          route_values=(("train-a", dust), ("train-b", dust)),
        )
        if metric.name == roughness_name
        else metric
        for metric in proposed.metrics
      ),
    )
    dust_verdict = evaluate_candidate(
      candidate,
      dust_candidate,
      stock,
      accepted,
      rules(),
      roughness_name,
      *UNCERTAINTY_ARGS,
    )
    self.assertEqual(dust_verdict.target_improvement, 0.0)
    self.assertFalse(dust_verdict.target_materially_improved)

  def test_balanced_target_loss_is_diagnostic_when_strata_pass(self):
    candidates = grid()
    candidate = candidates[0]
    route_ids = ("train-a", "train-b")
    stock = evaluation("stock", None, route_ids, 9.0, 0.5, 0.9)
    accepted = evaluation(
      "accepted",
      candidates[1].policy,
      route_ids,
      10.0,
      0.5,
      0.9,
    )
    proposed = evaluation(
      "candidate",
      candidate.policy,
      route_ids,
      9.0,
      0.4,
      0.95,
    )

    verdict = evaluate_candidate(
      candidate,
      proposed,
      stock,
      accepted,
      rules(),
      BehaviorMetricName.RAW_TORQUE_RATE_RMS.value,
      *UNCERTAINTY_ARGS,
    )

    self.assertEqual(verdict.target_improvement, 0.0)
    self.assertFalse(verdict.target_materially_improved)
    self.assertTrue(verdict.passed)

  def test_held_out_validator_rejects_only_frozen_winner_and_routes_do_not_leak(self):
    candidates = grid()
    train_stock = evaluation("stock-train", None, ("train-a", "train-b"), 10.0, 0.5, 0.9)
    train_accepted = evaluation("accepted-train", candidates[1].policy, ("train-a", "train-b"), 10.0, 0.5, 0.9)
    train_evaluations = (
      evaluation("lower", candidates[0].policy, ("train-a", "train-b"), 8.0, 0.4, 0.95),
      evaluation("incumbent", candidates[1].policy, ("train-a", "train-b"), 10.0, 0.5, 0.9),
      evaluation("upper", candidates[2].policy, ("train-a", "train-b"), 9.0, 0.45, 0.94),
    )
    selection = select_training_winner(
      candidates,
      train_evaluations,
      train_stock,
      train_accepted,
      rules(),
      BehaviorMetricName.RAW_TORQUE_RATE_RMS.value,
      *UNCERTAINTY_ARGS,
    )
    self.assertIsNotNone(selection)

    overlap = evaluation("winner-overlap", selection.winner.policy, ("train-a", "validation-a"), 8.0, 0.4, 0.95)
    with self.assertRaisesRegex(ValueError, "disjoint"):
      validate_frozen_winner(
        selection,
        overlap,
        train_stock,
        train_accepted,
        rules(),
        BehaviorMetricName.RAW_TORQUE_RATE_RMS.value,
        *UNCERTAINTY_ARGS,
      )

    validation_stock = evaluation("stock-validation", None, ("validation-a", "validation-b"), 10.0, 0.5, 0.9)
    validation_accepted = evaluation(
      "accepted-validation",
      candidates[1].policy,
      ("validation-a", "validation-b"),
      10.0,
      0.5,
      0.9,
    )
    failed_winner = evaluation(
      "winner-validation",
      selection.winner.policy,
      ("validation-a", "validation-b"),
      12.0,
      0.4,
      0.95,
    )
    validation = validate_frozen_winner(
      selection,
      failed_winner,
      validation_stock,
      validation_accepted,
      rules(),
      BehaviorMetricName.RAW_TORQUE_RATE_RMS.value,
      *UNCERTAINTY_ARGS,
    )
    self.assertFalse(validation.accepted)
    self.assertEqual(validation.frozen_winner_verdict.candidate, selection.winner)

    wrong_policy = evaluation(
      "wrong-validation",
      candidates[2].policy,
      ("validation-a", "validation-b"),
      8.0,
      0.4,
      0.95,
    )
    with self.assertRaisesRegex(ValueError, "frozen"):
      validate_frozen_winner(
        selection,
        wrong_policy,
        validation_stock,
        validation_accepted,
        rules(),
        BehaviorMetricName.RAW_TORQUE_RATE_RMS.value,
        *UNCERTAINTY_ARGS,
      )

  def test_selector_rejects_mislabeled_stock_and_platform_mismatch(self):
    candidates = grid()
    candidate = candidates[0]
    route_ids = ("train-a", "train-b")
    stock = evaluation("stock", None, route_ids, 10.0, 0.5, 0.9)
    accepted = evaluation("accepted", candidates[1].policy, route_ids, 10.0, 0.5, 0.9)
    proposed = evaluation("candidate", candidate.policy, route_ids, 8.0, 0.4, 0.95)

    stock_core = ReplayCoreIdentity(
      controller_name="mislabeled-stock",
      core_artifact_sha256="a" * 64,
      source_openpilot_commit="1" * 40,
      opendbc_commit="2" * 40,
      panda_commit="3" * 40,
    )
    mislabeled_stock = replace(
      stock,
      artifact_identity=ReplayArtifactIdentity.compose(
        ReplayRole.EXACT_STOCK,
        stock_core,
        candidate.policy,
      ).to_json(),
      policy=candidate.policy,
    )
    with self.assertRaisesRegex(ValueError, "exact-stock.*null policy"):
      evaluate_candidate(
        candidate, proposed, mislabeled_stock, accepted, rules(),
        BehaviorMetricName.RAW_TORQUE_RATE_RMS.value, *UNCERTAINTY_ARGS,
      )

    candidate_identity = ReplayArtifactIdentity.compose(
      ReplayRole.CANDIDATE,
      ReplayCoreIdentity(
        controller_name="candidate-wrong-platform",
        core_artifact_sha256="b" * 64,
        source_openpilot_commit="1" * 40,
        opendbc_commit="9" * 40,
        panda_commit="3" * 40,
      ),
      candidate.policy,
    )
    wrong_platform = replace(proposed, artifact_identity=candidate_identity.to_json())
    with self.assertRaisesRegex(ValueError, "different platform source"):
      select_training_winner(
        (candidate,), (wrong_platform,), stock, accepted, rules(),
        BehaviorMetricName.RAW_TORQUE_RATE_RMS.value, *UNCERTAINTY_ARGS,
      )

  def test_bootstrap_accepted_evaluation_must_alias_stock_metrics(self):
    candidate = grid()[0]
    route_ids = ("train-a", "train-b")
    stock = evaluation("stock", None, route_ids, 10.0, 0.5, 0.9)
    stock_payload = json.loads(stock.artifact_identity)
    stock_core = ReplayCoreIdentity(
      controller_name=stock_payload["core"]["controllerName"],
      core_artifact_sha256=stock_payload["core"]["coreArtifactSha256"],
      source_openpilot_commit=stock_payload["core"]["sourceOpenpilotCommit"],
      opendbc_commit=stock_payload["core"]["opendbcCommit"],
      panda_commit=stock_payload["core"]["pandaCommit"],
    )
    accepted_identity = ReplayArtifactIdentity.compose(
      ReplayRole.CURRENTLY_ACCEPTED,
      stock_core,
      None,
    )
    bad_bootstrap = PolicyEvaluation(
      artifact_identity=accepted_identity.to_json(),
      policy=None,
      route_ids=route_ids,
      metrics=tuple(
        replace(metric, value=9.0, route_values=(("train-a", 9.0), ("train-b", 9.0)))
        if metric.name == BehaviorMetricName.RAW_TORQUE_RATE_RMS.value
        else metric
        for metric in stock.metrics
      ),
    )
    proposed = evaluation("candidate", candidate.policy, route_ids, 8.0, 0.4, 0.95)
    with self.assertRaisesRegex(ValueError, "does not alias exact-stock evaluation"):
      evaluate_candidate(
        candidate, proposed, stock, bad_bootstrap, rules(),
        BehaviorMetricName.RAW_TORQUE_RATE_RMS.value, *UNCERTAINTY_ARGS,
      )

  def test_artifacts_are_immutable_and_modules_have_no_actuation_or_storage_import(self):
    policy = BehaviorPolicy(10.0, 1.0)
    with self.assertRaises(FrozenInstanceError):
      policy.damping_ratio = 0.8

    forbidden_import_tokens = (
      "cereal",
      "messaging",
      "params",
      "pathlib",
      "live_controller",
      "carcontrol",
      "carcontroller",
    )
    for module in (behavior_evidence, behavior_metrics, behavior_policy):
      tree = ast.parse(inspect.getsource(module))
      imports = tuple(
        alias.name.lower()
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
      )
      with self.subTest(module=module.__name__):
        self.assertFalse(
          any(token in imported for token in forbidden_import_tokens for imported in imports),
          imports,
        )


if __name__ == "__main__":
  unittest.main()

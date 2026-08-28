from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import unittest

from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import (
  BehaviorSample,
  BehaviorSourceIdentity,
  BehaviorWindow,
  ManeuverClass,
  ManeuverPhase,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_metrics import (
  aggregate_behavior_metrics,
  BehaviorContract,
  BehaviorMetricConfig,
  BehaviorMetricName,
  MetricDisposition,
  score_behavior,
  score_window,
  retain_route_metric_windows,
  speed_node_weights,
)
from openpilot.selfdrive.controls.lib.blatv2 import behavior_policy
from openpilot.selfdrive.controls.lib.blatv2.behavior_policy import (
  PAIRED_ROUTE_UNCERTAINTY_METHOD,
  MetricGateRule,
  MetricPreference,
  PolicyMetric,
)


SOURCE = BehaviorSourceIdentity(
  controller_name="test",
  controller_artifact_sha256="a" * 64,
  source_openpilot_commit="b" * 40,
  opendbc_commit="c" * 40,
  panda_commit="d" * 40,
  evidence_schema_version=1,
)


def config() -> BehaviorMetricConfig:
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
    maximum_route_windows_per_stratum=2,
  )


def make_sample(
  index: int,
  *,
  speed: float = 5.0,
  desired: float = 0.01,
  measured: float = 0.01,
  raw_torque: float = 0.1,
  applied_torque: float = 0.1,
  intervention: bool = False,
) -> BehaviorSample:
  return BehaviorSample(
    mono_time_ns=1_000_000_000 + index * 100_000_000,
    route_time_s=1.0 + index * 0.1,
    speed_mps=speed,
    scalar_curvature_1pm=desired,
    desired_curvature_1pm=desired,
    anchored_curvature_1pm=desired,
    desired_rack_angle_deg=desired * 1000.0,
    desired_rack_rate_deg_s=desired * 100.0,
    desired_rack_accel_deg_s2=0.0,
    measured_curvature_1pm=measured,
    measured_rack_angle_deg=measured * 1000.0,
    measured_rack_rate_deg_s=measured * 100.0,
    measured_rack_accel_deg_s2=0.0,
    raw_requested_torque=raw_torque,
    planned_requested_torque=raw_torque,
    reachable_envelope_torque=applied_torque,
    envelope_applied_torque=applied_torque,
    torque_headroom=0.5,
    actuator_constrained=False,
    steering_request_active=True,
    maximum_authority_required=False,
    lateral_active=True,
    inputs_valid=True,
    steering_pressed=False,
    controller_fault=False,
    driver_intervention_onset=intervention,
  )


def make_window(
  route_id: str,
  window_id: str,
  phase: ManeuverPhase,
  samples: tuple[BehaviorSample, ...],
  maneuver_class: ManeuverClass = ManeuverClass.TURN,
) -> BehaviorWindow:
  return BehaviorWindow(
    route_id=route_id,
    window_id=window_id,
    source=SOURCE,
    maneuver_class=maneuver_class,
    phase=phase,
    samples=samples,
  )


class TestBehaviorMetrics(unittest.TestCase):
  def test_bounded_window_metric_reduction_matches_eager_scorecard(self):
    windows = (
      make_window(
        "route-b",
        "hold",
        ManeuverPhase.HOLD,
        (make_sample(0), make_sample(1)),
      ),
      make_window(
        "route-a",
        "turn",
        ManeuverPhase.TURN_IN,
        (
          make_sample(0, desired=0.0, measured=0.0),
          make_sample(1, desired=0.01, measured=0.008),
          make_sample(2, desired=0.02, measured=0.02),
        ),
      ),
    )
    eager = score_behavior(windows, config())
    bounded = aggregate_behavior_metrics(
      (score_window(window, config()) for window in reversed(windows)),
      config(),
    )

    self.assertEqual(bounded, eager)
    self.assertEqual(bounded.to_json(), eager.to_json())

  def test_streaming_retention_matches_value_independent_route_prefixes(self):
    metric_config = replace(config(), maximum_route_windows_per_stratum=3)
    scored = tuple(
      score_window(
        make_window(
          "route-a",
          f"window-{index:04d}",
          ManeuverPhase.HOLD,
          (
            make_sample(index * 2, speed=2.5 + index % 4 * 5.0),
            make_sample(index * 2 + 1, speed=2.5 + index % 4 * 5.0),
          ),
          ManeuverClass.STRAIGHT if index % 2 else ManeuverClass.CURVE,
        ),
        metric_config,
      )
      for index in range(1_000)
    )
    retained = retain_route_metric_windows(iter(scored), metric_config)
    expected_ids: set[tuple[str, str]] = set()
    strata = {
      (node, window.maneuver_class)
      for window in scored
      for node, weight in window.speed_node_support
      if weight > 0.0
    }
    for node, maneuver_class in strata:
      eligible = (
        window
        for window in scored
        if window.maneuver_class is maneuver_class
        and any(item_node == node and weight > 0.0 for item_node, weight in window.speed_node_support)
      )
      selected = sorted(
        eligible,
        key=lambda window: (
          hashlib.sha256(f"{window.route_id}\0{window.window_id}".encode()).digest(),
          window.window_id,
        ),
      )[:metric_config.maximum_route_windows_per_stratum]
      expected_ids.update((window.route_id, window.window_id) for window in selected)

    self.assertEqual(
      {
        (window.route_id, window.window_id)
        for window in retained
        if window.summary_metric_name is None
      },
      expected_ids,
    )
    summary_names = {
      BehaviorMetricName.MIN_SIGNED_TURN_IN_LAG_S,
      BehaviorMetricName.MAX_SIGNED_TURN_IN_LAG_S,
      BehaviorMetricName.MIN_SIGNED_RELEASE_LAG_S,
      BehaviorMetricName.MAX_SIGNED_RELEASE_LAG_S,
      BehaviorMetricName.MIN_DELIVERED_FRACTION,
      BehaviorMetricName.MAX_DELIVERED_FRACTION,
      BehaviorMetricName.DELIVERED_ERROR_CROSSINGS_PER_S,
      BehaviorMetricName.HOLD_ERROR_CROSSINGS_PER_S,
      BehaviorMetricName.MAXIMUM_AUTHORITY_SHORTFALL,
    }
    self.assertEqual(
      {
        (window.speed_node_support[0][0], window.maneuver_class, window.summary_metric_name)
        for window in retained
        if window.summary_metric_name is not None
      },
      {
        (node, maneuver_class, name)
        for node, maneuver_class in strata
        for name in summary_names
      },
    )
    self.assertLessEqual(
      len(retained),
      len(strata) * (
        metric_config.maximum_route_windows_per_stratum + len(summary_names)
      ),
    )
    self.assertEqual(
      aggregate_behavior_metrics(iter(scored), metric_config),
      aggregate_behavior_metrics(iter(retained), metric_config),
    )

  def test_signed_early_turn_in_remains_negative(self):
    window = make_window(
      "route-a",
      "turn-in",
      ManeuverPhase.TURN_IN,
      (
        make_sample(0, desired=0.0, measured=0.0),
        make_sample(1, desired=0.002, measured=0.006),
        make_sample(2, desired=0.006, measured=0.008),
        make_sample(3, desired=0.010, measured=0.010),
      ),
    )

    metric = score_window(window, config()).metric(BehaviorMetricName.SIGNED_TURN_IN_LAG_S)
    self.assertTrue(metric.defined)
    self.assertAlmostEqual(metric.value, -0.1)

  def test_intervention_frame_and_later_response_cannot_score_delivery(self):
    window = make_window(
      "route-a",
      "takeover",
      ManeuverPhase.HOLD,
      (
        make_sample(0, desired=0.01, measured=0.009),
        make_sample(1, desired=0.01, measured=0.009),
        make_sample(2, desired=0.01, measured=1.0, intervention=True),
        make_sample(3, desired=0.01, measured=1.0),
      ),
    )

    result = score_window(window, config())
    self.assertEqual(result.clean_sample_count, 2)
    self.assertAlmostEqual(result.metric(BehaviorMetricName.PEAK_CURVATURE_ERROR).value, 0.001)

  def test_faulting_candidate_cannot_remove_its_physical_window(self):
    physical = (
      make_sample(0, speed=2.5),
      make_sample(1, speed=5.0),
      make_sample(2, speed=7.5),
    )
    healthy = score_window(
      make_window("route-a", "physical-window", ManeuverPhase.HOLD, physical),
      config(),
    )
    faulted = score_window(
      make_window(
        "route-a",
        "physical-window",
        ManeuverPhase.HOLD,
        tuple(replace(sample, controller_fault=True) for sample in physical),
      ),
      config(),
    )

    self.assertEqual(faulted.mean_speed_mps, healthy.mean_speed_mps)
    self.assertEqual(faulted.speed_node_support, healthy.speed_node_support)
    self.assertEqual(faulted.clean_sample_count, 0)
    self.assertTrue(all(
      metric.disposition is MetricDisposition.COVERAGE_EXCLUDED
      for metric in faulted.metrics
    ))
    self.assertEqual(
      tuple(window.window_id for window in retain_route_metric_windows((healthy,), config())),
      tuple(window.window_id for window in retain_route_metric_windows((faulted,), config())),
    )

  def test_phase_specific_release_metrics_are_explicit(self):
    samples = (
      make_sample(0, desired=0.010, measured=0.010),
      make_sample(1, desired=0.010, measured=0.012),
      make_sample(2, desired=0.008, measured=0.011),
      make_sample(3, desired=0.005, measured=0.006),
    )
    release = score_window(
      make_window("route-a", "release", ManeuverPhase.RELEASE_UNWIND, samples),
      config(),
    )
    turn = score_window(
      make_window("route-a", "turn", ManeuverPhase.TURN_IN, samples),
      config(),
    )

    self.assertAlmostEqual(release.metric(BehaviorMetricName.RELEASE_OVERSHOOT_1PM).value, 0.003)
    self.assertAlmostEqual(release.metric(BehaviorMetricName.SIGNED_RELEASE_LAG_S).value, 0.1)
    self.assertFalse(turn.metric(BehaviorMetricName.RELEASE_OVERSHOOT_1PM).defined)
    self.assertIn("not_release_phase", turn.metric(BehaviorMetricName.RELEASE_OVERSHOOT_1PM).exclusions)

  def test_raw_and_applied_chatter_and_burst_remain_separate(self):
    samples = tuple(
      make_sample(
        index,
        raw_torque=0.1 if index % 2 else -0.1,
        applied_torque=0.02,
      )
      for index in range(12)
    )
    result = score_window(
      make_window("route-a", "chatter", ManeuverPhase.HOLD, samples),
      config(),
    )

    self.assertGreater(
      result.metric(BehaviorMetricName.RAW_CHATTER_REVERSALS_PER_S).value,
      0.0,
    )
    self.assertEqual(
      result.metric(BehaviorMetricName.APPLIED_CHATTER_REVERSALS_PER_S).value,
      0.0,
    )
    self.assertTrue(result.metric(BehaviorMetricName.RAW_WORST_BURST_RMS).defined)
    self.assertTrue(result.metric(BehaviorMetricName.APPLIED_WORST_BURST_RMS).defined)

  def test_delivered_error_crossings_use_hysteresis_not_torque_chatter(self):
    quiet_errors = (-0.001, 0.001, -0.001, 0.001, -0.001, 0.001)
    quiet = score_window(
      make_window(
        "route-a",
        "torque-chatter",
        ManeuverPhase.HOLD,
        tuple(
          make_sample(
            index,
            measured=0.01 + error,
            raw_torque=0.1 if index % 2 else -0.1,
          )
          for index, error in enumerate(quiet_errors)
        ),
      ),
      config(),
    )
    self.assertGreater(
      quiet.metric(BehaviorMetricName.RAW_CHATTER_REVERSALS_PER_S).value,
      0.0,
    )
    self.assertEqual(
      quiet.metric(BehaviorMetricName.DELIVERED_ERROR_CROSSINGS_PER_S).value,
      0.0,
    )

    ringing_errors = (0.003, 0.001, -0.001, -0.003, -0.001, 0.003)
    ringing = score_window(
      make_window(
        "route-a",
        "path-ringing",
        ManeuverPhase.HOLD,
        tuple(
          make_sample(index, measured=0.01 + error, raw_torque=0.1)
          for index, error in enumerate(ringing_errors)
        ),
      ),
      config(),
    )
    self.assertEqual(
      ringing.metric(BehaviorMetricName.RAW_CHATTER_REVERSALS_PER_S).value,
      0.0,
    )
    self.assertAlmostEqual(
      ringing.metric(BehaviorMetricName.DELIVERED_ERROR_CROSSINGS_PER_S).value,
      4.0,
    )
    self.assertAlmostEqual(
      ringing.metric(BehaviorMetricName.HOLD_ERROR_CROSSINGS_PER_S).value,
      4.0,
    )

  def test_all_window_tails_survive_cap_and_veto_cancellation(self):
    metric_config = replace(config(), maximum_route_windows_per_stratum=2)
    window_ids = tuple(f"window-{index:02d}" for index in range(10))
    retained_ids = frozenset(sorted(
      window_ids,
      key=lambda window_id: hashlib.sha256(
        f"route-a\0{window_id}".encode(),
      ).digest(),
    )[:metric_config.maximum_route_windows_per_stratum])
    exceptional_ids = tuple(
      window_id for window_id in window_ids if window_id not in retained_ids
    )[:4]
    early_id, late_id, under_id, over_id = exceptional_ids

    patterns = {
      early_id: ((0.0, 0.004, 0.01), (0.0, 0.006, 0.01)),
      late_id: ((0.0, 0.006, 0.01, 0.01), (0.0, 0.0, 0.004, 0.01)),
      under_id: ((0.0, 0.006, 0.01), (0.0, 0.0048, 0.008)),
      over_id: ((0.0, 0.006, 0.01), (0.0, 0.0072, 0.012)),
    }
    scored = []
    for window_id in window_ids:
      desired, measured = patterns.get(
        window_id,
        ((0.0, 0.006, 0.01), (0.0, 0.006, 0.01)),
      )
      scored.append(score_window(
        make_window(
          "route-a",
          window_id,
          ManeuverPhase.TURN_IN,
          tuple(
            make_sample(index, desired=target, measured=delivered)
            for index, (target, delivered) in enumerate(
              zip(desired, measured, strict=True),
            )
          ),
        ),
        metric_config,
      ))
    scorecard = aggregate_behavior_metrics(scored, metric_config)
    self.assertTrue(set(exceptional_ids).isdisjoint({
      window.window_id
      for window in scorecard.windows
      if window.summary_metric_name is None
    }))

    def policy_metric(name: BehaviorMetricName) -> PolicyMetric:
      return PolicyMetric.from_scorecard(scorecard, name)

    minimum_lag = policy_metric(BehaviorMetricName.MIN_SIGNED_TURN_IN_LAG_S)
    maximum_lag = policy_metric(BehaviorMetricName.MAX_SIGNED_TURN_IN_LAG_S)
    minimum_delivery = policy_metric(BehaviorMetricName.MIN_DELIVERED_FRACTION)
    maximum_delivery = policy_metric(BehaviorMetricName.MAX_DELIVERED_FRACTION)
    average_delivery = policy_metric(BehaviorMetricName.DELIVERED_FRACTION)
    self.assertAlmostEqual(minimum_lag.value, -0.1)
    self.assertAlmostEqual(maximum_lag.value, 0.2)
    self.assertAlmostEqual(minimum_delivery.value, 0.8)
    self.assertAlmostEqual(maximum_delivery.value, 1.2)
    self.assertEqual(average_delivery.value, 1.0)
    self.assertEqual(minimum_lag.window_count, len(window_ids))
    self.assertEqual(maximum_lag.window_count, len(window_ids))

    def verdict(
      candidate: PolicyMetric,
      *,
      safe_value: float,
      contract: BehaviorContract,
      preference: MetricPreference,
      minimum: float,
      maximum: float,
    ):
      reference = replace(
        candidate,
        value=safe_value,
        route_values=(("route-a", safe_value),),
        stratum_metrics=(replace(
          candidate.stratum_metrics[0],
          value=safe_value,
          route_values=(("route-a", safe_value),),
        ),),
      )
      return behavior_policy._metric_margin(
        candidate,
        reference,
        reference,
        MetricGateRule(
          metric_name=candidate.name,
          contract=contract,
          preference=preference,
          noise_floor=0.0,
          margin_normalization=1.0,
          minimum_allowed=minimum,
          maximum_allowed=maximum,
          minimum_route_count=1,
          minimum_window_count=1,
          minimum_weighted_support=1.0,
          required_strata=("5:turn",),
        ),
        PAIRED_ROUTE_UNCERTAINTY_METHOD,
        1,
      )

    self.assertFalse(verdict(
      minimum_lag,
      safe_value=0.05,
      contract=BehaviorContract.SWIFT,
      preference=MetricPreference.HIGHER_IS_BETTER,
      minimum=0.0,
      maximum=0.12,
    ).passed)
    self.assertFalse(verdict(
      maximum_lag,
      safe_value=0.05,
      contract=BehaviorContract.SWIFT,
      preference=MetricPreference.LOWER_IS_BETTER,
      minimum=0.0,
      maximum=0.12,
    ).passed)
    self.assertFalse(verdict(
      minimum_delivery,
      safe_value=1.0,
      contract=BehaviorContract.STRONG,
      preference=MetricPreference.HIGHER_IS_BETTER,
      minimum=0.95,
      maximum=1.1,
    ).passed)
    self.assertFalse(verdict(
      maximum_delivery,
      safe_value=1.0,
      contract=BehaviorContract.STRONG,
      preference=MetricPreference.LOWER_IS_BETTER,
      minimum=0.0,
      maximum=1.1,
    ).passed)

  def test_maximum_authority_shortfall_censors_request_suppression(self):
    def authority_metric(
      *,
      planned: float,
      reachable: float,
      applied: float,
      request_active: bool,
    ):
      samples = tuple(
        replace(
          make_sample(index, raw_torque=1.5),
          planned_requested_torque=planned,
          reachable_envelope_torque=reachable,
          envelope_applied_torque=applied,
          steering_request_active=request_active,
          maximum_authority_required=True,
        )
        for index in range(2)
      )
      return score_window(
        make_window("route-a", "maximum-authority", ManeuverPhase.TURN_IN, samples),
        config(),
      ).metric(BehaviorMetricName.MAXIMUM_AUTHORITY_SHORTFALL)

    reachable = authority_metric(
      planned=0.4,
      reachable=0.4,
      applied=0.4,
      request_active=True,
    )
    short = authority_metric(
      planned=0.3,
      reachable=0.4,
      applied=0.3,
      request_active=True,
    )
    suppressed = authority_metric(
      planned=0.4,
      reachable=0.4,
      applied=0.4,
      request_active=False,
    )
    self.assertEqual(reachable.value, 0.0)
    self.assertAlmostEqual(short.value, 0.1)
    self.assertEqual(suppressed.disposition, MetricDisposition.PHYSICAL_UNSCOREABLE)
    self.assertIn("request_suppressed_maximum_authority", suppressed.exclusions)

  def test_all_window_unscoreable_summaries_count_affected_windows(self):
    suppressed = tuple(
      score_window(
        make_window(
          "route-a",
          f"request-off-{window_index}",
          ManeuverPhase.TURN_IN,
          tuple(
            replace(
              make_sample(sample_index),
              steering_request_active=False,
              maximum_authority_required=True,
            )
            for sample_index in range(2)
          ),
        ),
        config(),
      )
      for window_index in range(2)
    )
    excluded = tuple(
      score_window(
        make_window(
          "route-a",
          f"coverage-off-{window_index}",
          ManeuverPhase.TURN_IN,
          tuple(
            replace(make_sample(sample_index), controller_fault=True)
            for sample_index in range(2)
          ),
        ),
        config(),
      )
      for window_index in range(2)
    )

    for scored, name, disposition in (
      (
        suppressed,
        BehaviorMetricName.MAXIMUM_AUTHORITY_SHORTFALL,
        MetricDisposition.PHYSICAL_UNSCOREABLE,
      ),
      (
        excluded,
        BehaviorMetricName.MIN_DELIVERED_FRACTION,
        MetricDisposition.COVERAGE_EXCLUDED,
      ),
    ):
      retained = retain_route_metric_windows(scored, config())
      summary = next(window for window in retained if window.summary_metric_name is name)
      self.assertEqual(summary.metric(name).disposition, disposition)
      self.assertEqual(summary.metric(name).denominator, len(scored))
      self.assertEqual(summary.clean_sample_count, len(scored))

      policy = PolicyMetric.from_scorecard(
        aggregate_behavior_metrics(scored, config()),
        name,
      )
      self.assertEqual(policy.stratum_metrics[0].disposition, disposition)
      self.assertEqual(policy.stratum_metrics[0].window_count, len(scored))

  def test_mixed_all_window_failures_precede_defined_values(self):
    metric_config = config()
    delivered_windows = tuple(
      score_window(
        make_window(
          "route-b" if window_index == 2 else "route-a",
          f"delivered-{window_index}",
          ManeuverPhase.TURN_IN,
          (make_sample(0), make_sample(1)),
        ),
        metric_config,
      )
      for window_index in range(3)
    )
    coverage_window = score_window(
      make_window(
        "route-a",
        "delivered-coverage",
        ManeuverPhase.TURN_IN,
        tuple(
          replace(make_sample(index), controller_fault=True)
          for index in range(2)
        ),
      ),
      metric_config,
    )
    delivered_name = BehaviorMetricName.MIN_DELIVERED_FRACTION
    retained = retain_route_metric_windows(
      (*delivered_windows, coverage_window),
      metric_config,
    )
    route_summary = next(
      window
      for window in retained
      if window.route_id == "route-a" and window.summary_metric_name is delivered_name
    )
    self.assertEqual(
      route_summary.metric(delivered_name).disposition,
      MetricDisposition.COVERAGE_EXCLUDED,
    )
    self.assertEqual(route_summary.metric(delivered_name).denominator, 3)
    self.assertEqual(route_summary.clean_sample_count, 3)

    delivered = PolicyMetric.from_scorecard(
      aggregate_behavior_metrics((*delivered_windows, coverage_window), metric_config),
      delivered_name,
    )
    self.assertIsNone(delivered.value)
    self.assertEqual(
      delivered.stratum_metrics[0].disposition,
      MetricDisposition.COVERAGE_EXCLUDED,
    )
    self.assertEqual(delivered.stratum_metrics[0].window_count, 4)
    reference = PolicyMetric.from_scorecard(
      aggregate_behavior_metrics(delivered_windows, metric_config),
      delivered_name,
    )
    verdict = behavior_policy._metric_margin(
      delivered,
      reference,
      reference,
      MetricGateRule(
        metric_name=delivered_name.value,
        contract=BehaviorContract.STRONG,
        preference=MetricPreference.HIGHER_IS_BETTER,
        noise_floor=0.0,
        margin_normalization=1.0,
        minimum_allowed=0.95,
        maximum_allowed=1.1,
        minimum_route_count=1,
        minimum_window_count=1,
        minimum_weighted_support=1.0,
        required_strata=("5:turn",),
      ),
      PAIRED_ROUTE_UNCERTAINTY_METHOD,
      1,
    )
    self.assertFalse(verdict.passed)

    authority_windows = tuple(
      score_window(
        make_window(
          "route-b" if window_index == 2 else "route-a",
          f"authority-{window_index}",
          ManeuverPhase.TURN_IN,
          tuple(
            replace(make_sample(index), maximum_authority_required=True)
            for index in range(2)
          ),
        ),
        metric_config,
      )
      for window_index in range(3)
    )
    suppressed = score_window(
      make_window(
        "route-a",
        "authority-suppressed",
        ManeuverPhase.TURN_IN,
        tuple(
          replace(
            make_sample(index),
            steering_request_active=False,
            maximum_authority_required=True,
          )
          for index in range(2)
        ),
      ),
      metric_config,
    )
    authority_name = BehaviorMetricName.MAXIMUM_AUTHORITY_SHORTFALL
    authority = PolicyMetric.from_scorecard(
      aggregate_behavior_metrics((*authority_windows, suppressed), metric_config),
      authority_name,
    )
    self.assertEqual(
      authority.stratum_metrics[0].disposition,
      MetricDisposition.PHYSICAL_UNSCOREABLE,
    )
    self.assertEqual(authority.stratum_metrics[0].window_count, 4)

  def test_speed_transition_uses_adjacent_linear_weights_without_boundary(self):
    self.assertEqual(speed_node_weights(0.0, (0.0, 5.0, 10.0)), ((0.0, 1.0),))
    self.assertEqual(speed_node_weights(2.5, (0.0, 5.0, 10.0)), ((0.0, 0.5), (5.0, 0.5)))
    self.assertEqual(speed_node_weights(12.0, (0.0, 5.0, 10.0)), ((10.0, 1.0),))

    window = make_window(
      "route-a",
      "transition",
      ManeuverPhase.HOLD,
      (
        make_sample(0, speed=7.5, desired=0.02, measured=0.01),
        make_sample(1, speed=7.5, desired=0.02, measured=0.01),
      ),
    )
    scorecard = score_behavior((window,), config())
    support = {
      stratum.key.speed_node_mps: next(
        metric
        for metric in stratum.metrics
        if metric.name is BehaviorMetricName.PEAK_CURVATURE_ERROR
      ).weighted_support
      for stratum in scorecard.strata
    }
    self.assertEqual(support, {5.0: 0.5, 10.0: 0.5})
    encoded = scorecard.to_json()
    self.assertEqual(json.dumps(json.loads(encoded), sort_keys=True, separators=(",", ":")), encoded)
    self.assertEqual(scorecard.metric_config_sha256, config().sha256)

  def test_balanced_metric_does_not_let_highway_volume_drown_low_speed(self):
    low = make_window(
      "route-low",
      "low",
      ManeuverPhase.HOLD,
      (
        make_sample(0, speed=5.0, desired=0.02, measured=0.01),
        make_sample(1, speed=5.0, desired=0.02, measured=0.01),
      ),
    )
    highway = tuple(
      make_window(
        "route-high",
        f"high-{index:02d}",
        ManeuverPhase.HOLD,
        (
          replace(make_sample(0, speed=20.0), mono_time_ns=1_000_000_000 + index * 1_000_000_000, route_time_s=1.0 + index),
          replace(make_sample(1, speed=20.0), mono_time_ns=1_100_000_000 + index * 1_000_000_000, route_time_s=1.1 + index),
        ),
      )
      for index in range(20)
    )
    scorecard = score_behavior((low, *highway), config())
    peak = next(
      metric
      for metric in scorecard.balanced_metrics
      if metric.name is BehaviorMetricName.PEAK_CURVATURE_ERROR
    )

    # Peak error is a worst-case contract, not an average that lets the
    # highway stratum dilute the low-speed miss.
    self.assertAlmostEqual(peak.value, 0.01)
    self.assertEqual(len(scorecard.strata), 2)

  def test_candidate_physical_failure_is_not_dropped_by_easy_windows(self):
    hard = tuple(
      make_window(
        "route-a",
        f"hard-{index}",
        ManeuverPhase.TURN_IN,
        (
          make_sample(0, desired=0.0, measured=0.0),
          make_sample(1, desired=0.01, measured=0.0),
          make_sample(2, desired=0.02, measured=0.0),
        ),
      )
      for index in range(9)
    )
    easy = make_window(
      "route-a",
      "easy",
      ManeuverPhase.TURN_IN,
      (
        make_sample(0, desired=0.0, measured=0.0),
        make_sample(1, desired=0.01, measured=0.01),
        make_sample(2, desired=0.02, measured=0.02),
      ),
    )
    expanded = replace(config(), maximum_route_windows_per_stratum=20)
    scorecard = score_behavior((*hard, easy), expanded)
    metric = next(
      value
      for value in scorecard.balanced_metrics
      if value.name is BehaviorMetricName.SIGNED_TURN_IN_LAG_S
    )
    self.assertEqual(metric.disposition, MetricDisposition.PHYSICAL_UNSCOREABLE)
    self.assertFalse(metric.defined)
    self.assertEqual(len(metric.physical_failure_window_ids), 9)

  def test_phase_inapplicability_is_not_a_physical_failure(self):
    window = make_window(
      "route-a",
      "turn-only",
      ManeuverPhase.TURN_IN,
      (make_sample(0), make_sample(1)),
    )
    metric = score_window(window, config()).metric(
      BehaviorMetricName.SIGNED_RELEASE_LAG_S,
    )
    self.assertEqual(metric.disposition, MetricDisposition.NOT_APPLICABLE)
    self.assertFalse(metric.defined)


if __name__ == "__main__":
  unittest.main()

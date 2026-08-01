from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from openpilot.selfdrive.controls.lib.blatv2.behavior_configuration import (
  BEHAVIOR_GATE_SPEC_PATH,
  BEHAVIOR_SEGMENTATION_CONFIG_PATH,
  load_behavior_gate_spec,
  load_behavior_segmentation_config,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import canonical_json
from openpilot.selfdrive.controls.lib.blatv2.behavior_metrics import (
  BehaviorContract,
  BehaviorMetricName,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_policy import (
  PAIRED_ROUTE_UNCERTAINTY_METHOD,
  BehaviorPolicy,
  build_candidate_grid,
)


class TestBehaviorConfiguration(unittest.TestCase):
  def test_committed_behavior_authorities_are_canonical_and_strict(self) -> None:
    gate = load_behavior_gate_spec()
    segmentation = load_behavior_segmentation_config()

    self.assertEqual(
      BEHAVIOR_GATE_SPEC_PATH.read_text(),
      gate.to_json() + "\n",
    )
    self.assertEqual(
      BEHAVIOR_SEGMENTATION_CONFIG_PATH.read_text(),
      canonical_json(segmentation.to_dict()) + "\n",
    )
    grid = build_candidate_grid(gate.candidate_grid.policy_grid(
      BehaviorPolicy(natural_frequency_per_s=10.0, damping_ratio=1.0),
    ))
    self.assertEqual(len(grid), 16)

  def test_gate_is_two_dial_bounded_and_non_tradeable(self) -> None:
    gate = load_behavior_gate_spec()
    self.assertEqual(len(gate.candidate_grid.natural_frequency_log_offsets), 4)
    self.assertEqual(len(gate.candidate_grid.damping_ratio_log_offsets), 4)
    self.assertEqual(
      {rule.contract for rule in gate.metric_rules},
      set(BehaviorContract),
    )
    self.assertEqual(
      gate.target_metric_name,
      BehaviorMetricName.INTEGRATED_CURVATURE_ERROR.value,
    )
    self.assertEqual(
      gate.paired_uncertainty_method,
      PAIRED_ROUTE_UNCERTAINTY_METHOD,
    )
    self.assertEqual(gate.minimum_paired_route_count, 2)
    required_strata = {
      rule.metric_name: set(rule.required_strata)
      for rule in gate.metric_rules
    }
    for metric_name in (
      BehaviorMetricName.APPLIED_TORQUE_RATE_RMS.value,
      BehaviorMetricName.APPLIED_WORST_BURST_RMS.value,
      BehaviorMetricName.CORRECTION_LATENCY_S.value,
    ):
      self.assertIn("30:straight", required_strata[metric_name])
    for metric_name in (
      BehaviorMetricName.DELIVERED_FRACTION.value,
      BehaviorMetricName.INTEGRATED_CURVATURE_ERROR.value,
      BehaviorMetricName.RELEASE_OVERSHOOT_1PM.value,
      BehaviorMetricName.SIGNED_RELEASE_LAG_S.value,
      BehaviorMetricName.SIGNED_TURN_IN_LAG_S.value,
    ):
      self.assertIn("30:curve", required_strata[metric_name])
    turn = next(
      rule
      for rule in gate.metric_rules
      if rule.metric_name == BehaviorMetricName.SIGNED_TURN_IN_LAG_S.value
    )
    release = next(
      rule
      for rule in gate.metric_rules
      if rule.metric_name == BehaviorMetricName.SIGNED_RELEASE_LAG_S.value
    )
    self.assertEqual(
      (turn.minimum_allowed, turn.maximum_allowed),
      (0.0, 0.12),
    )
    self.assertEqual(
      (release.minimum_allowed, release.maximum_allowed),
      (0.0, 0.12),
    )

  def test_noncanonical_configuration_fails_closed(self) -> None:
    cases = (
      (BEHAVIOR_GATE_SPEC_PATH, lambda value: value + " ", load_behavior_gate_spec),
      (
        BEHAVIOR_SEGMENTATION_CONFIG_PATH,
        lambda value: value.replace("\n", " \n"),
        load_behavior_segmentation_config,
      ),
    )
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for source, mutation, loader in cases:
        with self.subTest(source=source.name):
          altered = root / source.name
          altered.write_text(mutation(source.read_text()))
          with self.assertRaisesRegex(ValueError, "not canonical"):
            loader(altered)

  def test_unknown_segmentation_key_fails_closed(self) -> None:
    payload = json.loads(BEHAVIOR_SEGMENTATION_CONFIG_PATH.read_text())
    payload["surprise"] = 1
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "segmentation.json"
      path.write_text(canonical_json(payload) + "\n")
      with self.assertRaisesRegex(ValueError, "keys"):
        load_behavior_segmentation_config(path)

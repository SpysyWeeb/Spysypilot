from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from openpilot.selfdrive.controls.lib.blatv2.policy import (
  CONTROLLER_POLICY_SCHEMA_VERSION,
  ControllerPolicy,
)


POLICY_PATH = (
  Path(__file__).parents[1]
  / "lib"
  / "blatv2"
  / "provisional_controller_policy.json"
)


class TestBlatV2Policy(unittest.TestCase):
  def test_committed_policy_is_explicitly_provisional_and_speed_independent(self):
    policy = ControllerPolicy.from_json_file(POLICY_PATH)

    self.assertEqual(CONTROLLER_POLICY_SCHEMA_VERSION, 1)
    self.assertTrue(policy.provisional)
    self.assertEqual(policy.revision, 0)
    self.assertIsNone(policy.observer_policy)
    self.assertEqual(policy.tracking_policy.natural_frequency_per_s, 10.0)
    self.assertEqual(policy.tracking_policy.damping_ratio, 1.0)
    self.assertIn("unelected", policy.provenance)

  def test_policy_encoding_and_hash_are_deterministic(self):
    first = ControllerPolicy.from_json_file(POLICY_PATH)
    second = ControllerPolicy.from_json_file(POLICY_PATH)

    self.assertEqual(first, second)
    self.assertEqual(first.to_json(), second.to_json())
    self.assertEqual(first.sha256, second.sha256)
    self.assertEqual(json.loads(first.to_json()), first.to_dict())

  def test_complete_observer_policy_is_constructed_without_defaults(self):
    policy = ControllerPolicy(
      revision=1,
      provenance="held-out residual selection",
      provisional=False,
      natural_frequency_per_s=8.0,
      damping_ratio=0.9,
      observer_time_constant_s=0.5,
      observer_max_abs_disturbance_torque=0.2,
    )

    self.assertEqual(policy.observer_policy.time_constant_s, 0.5)
    self.assertEqual(policy.observer_policy.max_abs_disturbance_torque, 0.2)

  def test_invalid_or_partial_policy_fails_closed(self):
    overrides_cases = (
      {"revision": -1},
      {"provenance": ""},
      {"natural_frequency_per_s": 0.0},
      {"damping_ratio": 0.0},
      {"observer_time_constant_s": 0.5},
      {"observer_max_abs_disturbance_torque": 0.2},
    )
    for overrides in overrides_cases:
      with self.subTest(overrides=overrides):
        values = {
          "revision": 0,
          "provenance": "explicit",
          "provisional": True,
          "natural_frequency_per_s": 10.0,
          "damping_ratio": 1.0,
          "observer_time_constant_s": None,
          "observer_max_abs_disturbance_torque": None,
        }
        values.update(overrides)
        with self.assertRaises(ValueError):
          ControllerPolicy(**values)

  def test_json_schema_rejects_unknown_fields(self):
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    payload["hidden_speed_gain"] = 2.0
    with tempfile.TemporaryDirectory() as temporary_directory:
      path = Path(temporary_directory) / "policy.json"
      path.write_text(json.dumps(payload), encoding="utf-8")

      with self.assertRaisesRegex(ValueError, "keys"):
        ControllerPolicy.from_json_file(path)

  def test_policy_source_has_no_speed_or_maneuver_schedule(self):
    fields = set(ControllerPolicy.__dataclass_fields__)
    forbidden = (
      "speed",
      "turn",
      "unwind",
      "handoff",
      "boost",
      "preview",
    )
    self.assertFalse(
      any(token in field.lower() for field in fields for token in forbidden),
    )

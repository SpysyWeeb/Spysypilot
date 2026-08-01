from __future__ import annotations

from dataclasses import replace
import json
import unittest

from openpilot.selfdrive.controls.lib.blatv2.behavior_learning_status import (
  BEHAVIOR_LEARNING_STATUS_PARAM,
  BEHAVIOR_LEARNING_STATUS_SCHEMA_VERSION,
  BehaviorLearningDiagnostic,
  BehaviorLearningState,
  BehaviorLearningStatus,
  BehaviorLearningStatusPublisher,
  BehaviorQualificationDisposition,
)


def training_status() -> BehaviorLearningStatus:
  return BehaviorLearningStatus(
    schema_version=BEHAVIOR_LEARNING_STATUS_SCHEMA_VERSION,
    informational_only=True,
    operation_id="1" * 32,
    sequence=7,
    state=BehaviorLearningState.TRAINING,
    diagnostic=BehaviorLearningDiagnostic.REPLAYING_TRAINING_GRID,
    terminal=False,
    started_mono_ns=1_000,
    updated_mono_ns=2_000,
    vehicle_identity="test-car",
    runtime_vehicle_identity_sha256="2" * 64,
    physical_generation_sha256="3" * 64,
    physical_profile_sha256="4" * 64,
    recorded_source_identity_sha256="5" * 64,
    eligible_route_count=5,
    required_route_count=4,
    training_route_count=3,
    validation_route_count=2,
    current_route_identity="route-2",
    current_route_index=1,
    total_route_count=5,
    current_candidate_index=4,
    total_candidate_count=16,
    completed_replay_jobs=17,
    total_replay_jobs=54,
    gate_spec_sha256="6" * 64,
    segmentation_config_sha256="7" * 64,
    transaction_sha256=None,
    behavior_finalization_sha256=None,
    behavior_selection_sha256=None,
    selected_behavior_policy_sha256=None,
    smooth_passed=None,
    swift_passed=None,
    strong_passed=None,
    target_materially_improved=None,
    qualification_disposition=None,
    reasons=(),
  )


def qualified_status() -> BehaviorLearningStatus:
  return replace(
    training_status(),
    sequence=8,
    state=BehaviorLearningState.COMPLETE,
    diagnostic=BehaviorLearningDiagnostic.CANDIDATE_QUALIFIED,
    terminal=True,
    current_route_identity=None,
    current_route_index=None,
    current_candidate_index=None,
    completed_replay_jobs=54,
    transaction_sha256="8" * 64,
    behavior_finalization_sha256="9" * 64,
    behavior_selection_sha256="a" * 64,
    selected_behavior_policy_sha256="b" * 64,
    smooth_passed=True,
    swift_passed=True,
    strong_passed=True,
    target_materially_improved=True,
    qualification_disposition=(
      BehaviorQualificationDisposition.QUALIFIED_CANDIDATE_AVAILABLE
    ),
    reasons=("passed",),
  )


class FakeParams:
  def __init__(self) -> None:
    self.puts: list[tuple[str, dict[str, object], bool]] = []
    self.failure: Exception | None = None

  def put(self, key: str, value: dict[str, object], *, block: bool) -> None:
    if self.failure is not None:
      raise self.failure
    self.puts.append((key, dict(value), block))


def publisher_context(**overrides: object) -> dict[str, object]:
  context: dict[str, object] = {
    "vehicle_identity": "test-car",
    "runtime_vehicle_identity_sha256": "2" * 64,
    "physical_generation_sha256": "3" * 64,
    "physical_profile_sha256": "4" * 64,
    "recorded_source_identity_sha256": "5" * 64,
    "eligible_route_count": 5,
    "required_route_count": 4,
    "training_route_count": 3,
    "validation_route_count": 2,
    "total_route_count": 5,
    "total_candidate_count": 16,
    "total_replay_jobs": 54,
    "gate_spec_sha256": "6" * 64,
    "segmentation_config_sha256": "7" * 64,
  }
  context.update(overrides)
  return context


class SequenceFactory:
  def __init__(self, *values: int | str) -> None:
    self._values = iter(values)

  def __call__(self):
    return next(self._values)


class TestBehaviorLearningStatus(unittest.TestCase):
  def test_separate_param_and_strict_canonical_roundtrip(self) -> None:
    status = qualified_status()
    restored = BehaviorLearningStatus.from_json(status.to_json())

    self.assertEqual(BEHAVIOR_LEARNING_STATUS_PARAM, "BLaTv2BehaviorLearningStatus")
    self.assertEqual(restored, status)
    self.assertEqual(restored.to_json(), status.to_json())
    self.assertTrue(restored.informational_only)

  def test_nonterminal_progress_has_no_gate_or_selection_authority(self) -> None:
    status = BehaviorLearningStatus.from_json(training_status().to_json())

    self.assertFalse(status.terminal)
    self.assertIsNone(status.qualification_disposition)
    self.assertIsNone(status.smooth_passed)
    self.assertIsNone(status.transaction_sha256)

  def test_failed_operation_retains_stock_without_inventing_gate_results(self) -> None:
    status = replace(
      training_status(),
      state=BehaviorLearningState.FAILED,
      diagnostic=BehaviorLearningDiagnostic.REPLAY_NONDETERMINISTIC,
      terminal=True,
      current_route_identity=None,
      current_route_index=None,
      current_candidate_index=None,
      qualification_disposition=BehaviorQualificationDisposition.STOCK_RETAINED,
      reasons=("replay_nondeterministic",),
    )

    self.assertEqual(
      BehaviorLearningStatus.from_json(status.to_json()),
      status,
    )
    self.assertIsNone(status.selected_behavior_policy_sha256)

  def test_streaming_required_is_a_terminal_stock_retained_diagnostic(self) -> None:
    status = replace(
      training_status(),
      state=BehaviorLearningState.FAILED,
      diagnostic=BehaviorLearningDiagnostic.BEHAVIOR_STREAMING_REQUIRED,
      terminal=True,
      current_route_identity=None,
      current_route_index=None,
      current_candidate_index=None,
      completed_replay_jobs=0,
      total_replay_jobs=0,
      qualification_disposition=BehaviorQualificationDisposition.STOCK_RETAINED,
      reasons=("behavior_streaming_required",),
    )

    self.assertEqual(BehaviorLearningStatus.from_json(status.to_json()), status)
    self.assertIsNone(status.smooth_passed)

  def test_malformed_unknown_and_authority_smuggling_fail_closed(self) -> None:
    status = training_status()
    payload = json.loads(status.to_json())
    payload["surprise"] = True
    with self.assertRaisesRegex(ValueError, "keys"):
      BehaviorLearningStatus.from_json(json.dumps(payload))

    with self.assertRaisesRegex(ValueError, "informational"):
      replace(status, informational_only=False)
    with self.assertRaisesRegex(ValueError, "non-terminal.*gate"):
      replace(status, smooth_passed=True)
    with self.assertRaisesRegex(ValueError, "selection provenance"):
      replace(qualified_status(), selected_behavior_policy_sha256=None)
    with self.assertRaisesRegex(ValueError, "diagnostic"):
      replace(
        status,
        diagnostic=BehaviorLearningDiagnostic.PUBLISHING_BEHAVIOR_GENERATION,
      )


class TestBehaviorLearningStatusPublisher(unittest.TestCase):
  def publisher(
    self,
    params: FakeParams | None = None,
    *,
    times: tuple[int, ...] = (100, 110, 120, 130),
    identities: tuple[str, ...] = ("c" * 32, "d" * 32),
  ) -> tuple[FakeParams, BehaviorLearningStatusPublisher]:
    resolved_params = FakeParams() if params is None else params
    return resolved_params, BehaviorLearningStatusPublisher(
      resolved_params,
      monotonic_ns=SequenceFactory(*times),
      operation_id_factory=SequenceFactory(*identities),
    )

  def test_waiting_operation_has_canonical_blocking_projection(self) -> None:
    params, publisher = self.publisher()
    status = publisher.publish(
      BehaviorLearningState.WAITING_FOR_PHYSICAL_PROFILE,
      BehaviorLearningDiagnostic.PHYSICAL_PROFILE_UNQUALIFIED,
      new_operation=True,
      **publisher_context(
        physical_generation_sha256=None,
        physical_profile_sha256=None,
        recorded_source_identity_sha256=None,
        eligible_route_count=0,
        training_route_count=0,
        validation_route_count=0,
        total_route_count=0,
        total_candidate_count=0,
        total_replay_jobs=0,
      ),
    )

    self.assertEqual(status.operation_id, "c" * 32)
    self.assertEqual(status.sequence, 0)
    self.assertEqual(status.started_mono_ns, 100)
    self.assertIs(publisher.last_status, status)
    self.assertEqual(len(params.puts), 1)
    key, payload, block = params.puts[0]
    self.assertEqual(key, BEHAVIOR_LEARNING_STATUS_PARAM)
    self.assertTrue(block)
    self.assertEqual(payload, json.loads(status.to_json()))
    self.assertEqual(BehaviorLearningStatus.from_json(json.dumps(payload)), status)

  def test_complete_candidate_publishes_full_provenance(self) -> None:
    params, publisher = self.publisher()
    training = publisher.publish(
      BehaviorLearningState.TRAINING,
      BehaviorLearningDiagnostic.REPLAYING_TRAINING_GRID,
      new_operation=True,
      **publisher_context(
        current_route_identity="route-2",
        current_route_index=1,
        current_candidate_index=4,
        completed_replay_jobs=17,
      ),
    )
    complete = publisher.publish(
      BehaviorLearningState.COMPLETE,
      BehaviorLearningDiagnostic.CANDIDATE_QUALIFIED,
      completed_replay_jobs=54,
      transaction_sha256="8" * 64,
      behavior_finalization_sha256="9" * 64,
      behavior_selection_sha256="a" * 64,
      selected_behavior_policy_sha256="b" * 64,
      smooth_passed=True,
      swift_passed=True,
      strong_passed=True,
      target_materially_improved=True,
      qualification_disposition=(
        BehaviorQualificationDisposition.QUALIFIED_CANDIDATE_AVAILABLE
      ),
      reasons=("passed",),
    )

    self.assertEqual((training.sequence, complete.sequence), (0, 1))
    self.assertEqual(complete.started_mono_ns, training.started_mono_ns)
    self.assertTrue(complete.terminal)
    self.assertEqual(
      complete.selected_behavior_policy_sha256,
      "b" * 64,
    )
    self.assertEqual(len(params.puts), 2)

  def test_complete_stock_retained_and_failed_are_explicit(self) -> None:
    _, stock_publisher = self.publisher()
    retained = stock_publisher.publish(
      BehaviorLearningState.COMPLETE,
      BehaviorLearningDiagnostic.STOCK_RETAINED,
      new_operation=True,
      **publisher_context(
        completed_replay_jobs=54,
        transaction_sha256="8" * 64,
        behavior_finalization_sha256="9" * 64,
        smooth_passed=True,
        swift_passed=False,
        strong_passed=True,
        target_materially_improved=False,
        qualification_disposition=BehaviorQualificationDisposition.STOCK_RETAINED,
        reasons=("swift_gate_failed",),
      ),
    )
    self.assertIsNone(retained.selected_behavior_policy_sha256)
    self.assertEqual(
      retained.qualification_disposition,
      BehaviorQualificationDisposition.STOCK_RETAINED,
    )

    _, failed_publisher = self.publisher()
    failed = failed_publisher.publish(
      BehaviorLearningState.FAILED,
      BehaviorLearningDiagnostic.REPLAY_NONDETERMINISTIC,
      new_operation=True,
      **publisher_context(
        qualification_disposition=BehaviorQualificationDisposition.STOCK_RETAINED,
        reasons=("replay_nondeterministic",),
      ),
    )
    self.assertTrue(failed.terminal)
    self.assertIsNone(failed.transaction_sha256)

  def test_operation_lifecycle_and_established_context_are_immutable(self) -> None:
    _, publisher = self.publisher(times=tuple(range(100, 500, 10)))
    first = publisher.publish(
      BehaviorLearningState.PREPARING,
      BehaviorLearningDiagnostic.VALIDATING_ROUTE_EVIDENCE,
      new_operation=True,
      **publisher_context(completed_replay_jobs=0),
    )
    second = publisher.publish(
      BehaviorLearningState.TRAINING,
      BehaviorLearningDiagnostic.REPLAYING_TRAINING_GRID,
      completed_replay_jobs=1,
      current_candidate_index=0,
    )
    self.assertEqual((first.sequence, second.sequence), (0, 1))
    self.assertEqual(second.started_mono_ns, 100)
    self.assertEqual(second.vehicle_identity, "test-car")

    stable_mutations = (
      {"vehicle_identity": "another-car"},
      {"runtime_vehicle_identity_sha256": "f" * 64},
      {"physical_generation_sha256": "f" * 64},
      {"physical_profile_sha256": "f" * 64},
      {"recorded_source_identity_sha256": "f" * 64},
      {"gate_spec_sha256": "f" * 64},
      {"segmentation_config_sha256": "f" * 64},
      {"eligible_route_count": 6},
      {"required_route_count": 5},
      {"training_route_count": 2},
      {"validation_route_count": 1},
      {"total_route_count": 6},
      {"total_candidate_count": 15},
      {"total_replay_jobs": 55},
    )
    for mutation in stable_mutations:
      with self.assertRaisesRegex(ValueError, "cannot change"):
        publisher.publish(
          BehaviorLearningState.TRAINING,
          BehaviorLearningDiagnostic.REPLAYING_TRAINING_GRID,
          **mutation,
        )

    with self.assertRaisesRegex(ValueError, "move backwards"):
      publisher.publish(
        BehaviorLearningState.TRAINING,
        BehaviorLearningDiagnostic.REPLAYING_TRAINING_GRID,
        completed_replay_jobs=0,
      )
    with self.assertRaisesRegex(ValueError, "exceed"):
      publisher.publish(
        BehaviorLearningState.TRAINING,
        BehaviorLearningDiagnostic.REPLAYING_TRAINING_GRID,
        completed_replay_jobs=55,
      )

  def test_terminal_operation_cannot_resume_without_new_identity(self) -> None:
    _, publisher = self.publisher()
    terminal = publisher.publish(
      BehaviorLearningState.FAILED,
      BehaviorLearningDiagnostic.BEHAVIOR_TRANSACTION_FAILED,
      new_operation=True,
      **publisher_context(
        qualification_disposition=BehaviorQualificationDisposition.STOCK_RETAINED,
        reasons=("transaction_failed",),
      ),
    )
    with self.assertRaisesRegex(ValueError, "requires a new operation"):
      publisher.publish(
        BehaviorLearningState.PREPARING,
        BehaviorLearningDiagnostic.VALIDATING_ROUTE_EVIDENCE,
      )
    restarted = publisher.publish(
      BehaviorLearningState.PREPARING,
      BehaviorLearningDiagnostic.VALIDATING_ROUTE_EVIDENCE,
      new_operation=True,
      **publisher_context(),
    )
    self.assertNotEqual(restarted.operation_id, terminal.operation_id)
    self.assertEqual(restarted.sequence, 0)

  def test_malformed_id_unknown_context_and_clock_regression_fail_closed(self) -> None:
    _, malformed = self.publisher(identities=("ABC",))
    with self.assertRaisesRegex(ValueError, "invalid identity"):
      malformed.publish(
        BehaviorLearningState.PREPARING,
        BehaviorLearningDiagnostic.VALIDATING_ROUTE_EVIDENCE,
        new_operation=True,
        **publisher_context(),
      )
    self.assertIsNone(malformed.last_status)

    _, publisher = self.publisher(times=(100, 99))
    publisher.publish(
      BehaviorLearningState.PREPARING,
      BehaviorLearningDiagnostic.VALIDATING_ROUTE_EVIDENCE,
      new_operation=True,
      **publisher_context(),
    )
    with self.assertRaisesRegex(ValueError, "time cannot move backwards"):
      publisher.publish(
        BehaviorLearningState.TRAINING,
        BehaviorLearningDiagnostic.REPLAYING_TRAINING_GRID,
      )
    with self.assertRaisesRegex(ValueError, "unknown behavior-learning context"):
      publisher.publish(
        BehaviorLearningState.TRAINING,
        BehaviorLearningDiagnostic.REPLAYING_TRAINING_GRID,
        activation_approved=True,
      )

  def test_params_failure_propagates_without_advancing_publisher(self) -> None:
    params = FakeParams()
    params.failure = RuntimeError("params unavailable")
    _, publisher = self.publisher(
      params,
      times=(100, 110, 120, 130),
      identities=("c" * 32, "d" * 32),
    )
    with self.assertRaisesRegex(RuntimeError, "params unavailable"):
      publisher.publish(
        BehaviorLearningState.PREPARING,
        BehaviorLearningDiagnostic.VALIDATING_ROUTE_EVIDENCE,
        new_operation=True,
        **publisher_context(),
      )
    self.assertIsNone(publisher.last_status)

    params.failure = None
    status = publisher.publish(
      BehaviorLearningState.PREPARING,
      BehaviorLearningDiagnostic.VALIDATING_ROUTE_EVIDENCE,
      new_operation=True,
      **publisher_context(),
    )
    self.assertEqual(status.operation_id, "d" * 32)
    self.assertEqual(status.sequence, 0)

    params.failure = RuntimeError("params unavailable again")
    with self.assertRaisesRegex(RuntimeError, "params unavailable again"):
      publisher.publish(
        BehaviorLearningState.TRAINING,
        BehaviorLearningDiagnostic.REPLAYING_TRAINING_GRID,
        completed_replay_jobs=1,
      )
    self.assertIs(publisher.last_status, status)

    params.failure = None
    recovered = publisher.publish(
      BehaviorLearningState.TRAINING,
      BehaviorLearningDiagnostic.REPLAYING_TRAINING_GRID,
      completed_replay_jobs=1,
    )
    self.assertEqual(recovered.sequence, 1)
    self.assertEqual(recovered.updated_mono_ns, 130)


if __name__ == "__main__":
  unittest.main()

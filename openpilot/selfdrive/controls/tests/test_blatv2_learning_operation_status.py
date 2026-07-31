from __future__ import annotations

import hashlib
import json

import pytest  # noqa: TID251

from openpilot.selfdrive.controls.lib.blatv2.learning_operation_status import (
  LEARNING_OPERATION_STATUS_PARAM,
  LearningOperationState,
  LearningOperationStatusPublisher,
  build_learning_operation_status_bytes,
  decode_learning_operation_status,
  route_identity_sha256,
)


SHA_A = hashlib.sha256(b"a").hexdigest()
SHA_B = hashlib.sha256(b"b").hexdigest()
SHA_C = hashlib.sha256(b"c").hexdigest()


class FakeParams:
  def __init__(self) -> None:
    self.values: dict[str, dict[str, object]] = {}
    self.puts: list[tuple[str, dict[str, object], bool]] = []

  def put(
    self,
    key: str,
    value: dict[str, object],
    *,
    block: bool,
  ) -> None:
    self.values[key] = dict(value)
    self.puts.append((key, dict(value), block))


def base_payload(**overrides):
  payload = {
    "accepted_sample_count": 0,
    "current_route_identity": None,
    "current_route_index": None,
    "diagnostic": "collecting_current_drive",
    "evidence_sha256": None,
    "informational_only": True,
    "last_route_identity": None,
    "ledger_sha256": None,
    "operation_id": "12" * 16,
    "rejected_sample_count": 0,
    "retry_count": 0,
    "runtime_identity_sha256": SHA_A,
    "schema_version": 1,
    "sequence": 0,
    "started_mono_ns": 100,
    "state": "collecting",
    "terminal": False,
    "total_route_count": None,
    "updated_mono_ns": 100,
    "vehicle_identity": "CAR",
  }
  payload.update(overrides)
  return payload


def test_operation_status_is_canonical_and_informational() -> None:
  encoded = build_learning_operation_status_bytes(**base_payload())
  assert encoded == json.dumps(
    base_payload(),
    sort_keys=True,
    separators=(",", ":"),
  ).encode()
  assert decode_learning_operation_status(encoded) == base_payload()


@pytest.mark.parametrize(
  ("state", "diagnostic", "terminal"),
  [
    ("preparing", "waiting_for_car_params", False),
    ("ready_no_evidence", "ready_for_first_drive", True),
    ("collecting", "collecting_current_drive", False),
    ("finalizing", "finalizing_drive", False),
    ("retry_pending", "persist_retry_pending", False),
    ("backfilling", "scanning_routes", False),
    ("idle", "evidence_ready", True),
    (
      "drive_skipped_identity_mismatch",
      "car_params_identity_mismatch",
      True,
    ),
    ("failed", "unexpected_error", True),
  ],
)
def test_all_operation_states_have_pinned_terminal_semantics(
  state: str,
  diagnostic: str,
  terminal: bool,
) -> None:
  overrides = {
    "state": state,
    "diagnostic": diagnostic,
    "terminal": terminal,
  }
  if state == "ready_no_evidence":
    overrides |= {
      "accepted_sample_count": 0,
      "rejected_sample_count": 0,
    }
  if state == "idle":
    overrides["evidence_sha256"] = SHA_B
  decoded = decode_learning_operation_status(
    build_learning_operation_status_bytes(**base_payload(**overrides)),
  )
  assert decoded["state"] == state
  assert decoded["terminal"] is terminal


def test_route_progress_is_one_based_and_hash_only() -> None:
  route_identity = route_identity_sha256("000000ca--bd6b1b11ef")
  payload = base_payload(
    state="backfilling",
    diagnostic="replaying_route",
    current_route_identity=route_identity,
    current_route_index=2,
    total_route_count=3,
  )
  assert decode_learning_operation_status(
    build_learning_operation_status_bytes(**payload),
  )["current_route_index"] == 2
  assert len(route_identity) == 64

  with pytest.raises(ValueError, match="one-based"):
    build_learning_operation_status_bytes(**(
      payload | {"current_route_index": 0}
    ))
  with pytest.raises(ValueError, match="no resolved current route"):
    build_learning_operation_status_bytes(**base_payload(
      state="backfilling",
      diagnostic="scanning_routes",
      current_route_identity=route_identity,
    ))


def test_state_specific_identity_and_diagnostic_invariants_fail_closed() -> None:
  with pytest.raises(ValueError, match="incompatible"):
    build_learning_operation_status_bytes(**base_payload(
      diagnostic="ready_for_first_drive",
    ))
  with pytest.raises(ValueError, match="runtime identity"):
    build_learning_operation_status_bytes(**base_payload(
      runtime_identity_sha256=None,
    ))
  with pytest.raises(ValueError, match="committed evidence"):
    build_learning_operation_status_bytes(**base_payload(
      state="idle",
      diagnostic="evidence_ready",
      terminal=True,
    ))
  with pytest.raises(ValueError, match="without evidence"):
    build_learning_operation_status_bytes(**base_payload(
      ledger_sha256=SHA_C,
    ))


def test_publisher_sequences_one_operation_and_starts_another() -> None:
  params = FakeParams()
  clock = iter((100, 110, 200))
  ids = iter(("01" * 16, "02" * 16))
  publisher = LearningOperationStatusPublisher(
    params,
    monotonic_ns=lambda: next(clock),
    operation_id_factory=lambda: next(ids),
  )
  first = decode_learning_operation_status(publisher.publish(
    state=LearningOperationState.PREPARING,
    diagnostic="restoring_runtime",
    new_operation=True,
  ))
  second = decode_learning_operation_status(publisher.publish(
    state=LearningOperationState.COLLECTING,
    diagnostic="collecting_current_drive",
    vehicle_identity="CAR",
    runtime_identity_sha256=SHA_A,
  ))
  third = decode_learning_operation_status(publisher.publish(
    state=LearningOperationState.READY_NO_EVIDENCE,
    diagnostic="ready_for_first_drive",
    new_operation=True,
    vehicle_identity="CAR",
    runtime_identity_sha256=SHA_A,
  ))

  assert (first["operation_id"], first["sequence"]) == ("01" * 16, 0)
  assert (second["operation_id"], second["sequence"]) == ("01" * 16, 1)
  assert second["started_mono_ns"] == 100
  assert (third["operation_id"], third["sequence"]) == ("02" * 16, 0)
  assert third["started_mono_ns"] == 200
  assert all(key == LEARNING_OPERATION_STATUS_PARAM for key, _, _ in params.puts)
  assert all(block for _, _, block in params.puts)

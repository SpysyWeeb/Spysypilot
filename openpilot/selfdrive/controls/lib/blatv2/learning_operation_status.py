"""Strict informational status for live and historical learning work.

This Params projection is deliberately not an artifact, ledger, approval, or
controller-selection input.  It exists only so an offroad UI can distinguish
an empty learner from active finalization, a retry, historical replay, or a
drive that was intentionally skipped.  Deleting or editing it cannot affect
learning evidence or actuation.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
import hashlib
import json
import re
import secrets
import time
from typing import Any


LEARNING_OPERATION_STATUS_PARAM = "BLaTv2LearningOperationStatus"
LEARNING_OPERATION_STATUS_SCHEMA_VERSION = 1
_HEX_32_RE = re.compile(r"[0-9a-f]{32}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TOP_LEVEL_KEYS = {
  "accepted_sample_count",
  "current_route_identity",
  "current_route_index",
  "diagnostic",
  "evidence_sha256",
  "informational_only",
  "last_route_identity",
  "ledger_sha256",
  "operation_id",
  "rejected_sample_count",
  "retry_count",
  "runtime_identity_sha256",
  "schema_version",
  "sequence",
  "started_mono_ns",
  "state",
  "terminal",
  "total_route_count",
  "updated_mono_ns",
  "vehicle_identity",
}


class LearningOperationState(StrEnum):
  PREPARING = "preparing"
  READY_NO_EVIDENCE = "ready_no_evidence"
  COLLECTING = "collecting"
  FINALIZING = "finalizing"
  RETRY_PENDING = "retry_pending"
  BACKFILLING = "backfilling"
  IDLE = "idle"
  DRIVE_SKIPPED_IDENTITY_MISMATCH = (
    "drive_skipped_identity_mismatch"
  )
  FAILED = "failed"


DIAGNOSTICS_BY_STATE: dict[LearningOperationState, frozenset[str]] = {
  LearningOperationState.PREPARING: frozenset({
    "discovering_remote_worker",
    "waiting_for_car_params",
    "restoring_runtime",
  }),
  LearningOperationState.READY_NO_EVIDENCE: frozenset({
    "ready_for_first_drive",
  }),
  LearningOperationState.COLLECTING: frozenset({
    "collecting_current_drive",
  }),
  LearningOperationState.FINALIZING: frozenset({
    "finalizing_drive",
    "verifying_backfill",
    "publishing_backfill",
  }),
  LearningOperationState.RETRY_PENDING: frozenset({
    "persist_retry_pending",
  }),
  LearningOperationState.BACKFILLING: frozenset({
    "scanning_routes",
    "replaying_route",
  }),
  LearningOperationState.IDLE: frozenset({
    "evidence_ready",
    "backfill_complete",
    "backfill_complete_late_older_skipped",
    "backfill_complete_with_rejections",
  }),
  LearningOperationState.DRIVE_SKIPPED_IDENTITY_MISMATCH: frozenset({
    "car_params_identity_mismatch",
  }),
  LearningOperationState.FAILED: frozenset({
    "architecture_verification_failed",
    "architecture_verification_interrupted",
    "runtime_restore_failed",
    "backfill_reader_unavailable",
    "backfill_route_incompatible",
    "backfill_corrupt_log",
    "backfill_nondeterministic",
    "backfill_publish_failed",
    "backfill_untracked_evidence",
    "backfill_no_complete_routes",
    "unexpected_error",
  }),
}
_TERMINAL_STATES = frozenset({
  LearningOperationState.READY_NO_EVIDENCE,
  LearningOperationState.IDLE,
  LearningOperationState.DRIVE_SKIPPED_IDENTITY_MISMATCH,
  LearningOperationState.FAILED,
})


def route_identity_sha256(route_name: str) -> str:
  """Return a privacy-preserving identity for one canonical local route."""
  if type(route_name) is not str:
    raise TypeError("route name must be a string")
  canonical = route_name.strip()
  if (
    not canonical
    or len(canonical) > 128
    or any(character not in "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ._-" for character in canonical)
  ):
    raise ValueError("route name is outside the canonical local format")
  return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _nonnegative_int(value: object, name: str) -> int:
  if type(value) is not int or value < 0:
    raise ValueError(f"{name} must be a nonnegative integer")
  return value


def _optional_sha256(value: object, name: str) -> str | None:
  if value is None:
    return None
  if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
    raise ValueError(f"{name} must be null or a lowercase SHA-256")
  return value


def _optional_identity(value: object, name: str) -> str | None:
  if value is None:
    return None
  if type(value) is not str:
    raise TypeError(f"{name} must be null or a string")
  identity = value.strip()
  if not identity or len(identity) > 256:
    raise ValueError(f"{name} must be a bounded nonempty string")
  return identity


def validate_learning_operation_status_payload(
  payload: object,
) -> dict[str, object]:
  """Validate and normalize the exact schema consumed by the C++ UI."""
  if type(payload) is not dict or set(payload) != _TOP_LEVEL_KEYS:
    raise ValueError("learning operation status keys do not match")
  if (
    payload["schema_version"] != LEARNING_OPERATION_STATUS_SCHEMA_VERSION
    or type(payload["schema_version"]) is not int
    or payload["informational_only"] is not True
  ):
    raise ValueError("learning operation status schema is incompatible")

  try:
    state = LearningOperationState(payload["state"])
  except (TypeError, ValueError) as exc:
    raise ValueError("learning operation state is unknown") from exc
  diagnostic = payload["diagnostic"]
  if (
    type(diagnostic) is not str
    or diagnostic not in DIAGNOSTICS_BY_STATE[state]
  ):
    raise ValueError("diagnostic is incompatible with operation state")
  terminal = payload["terminal"]
  if type(terminal) is not bool or terminal != (state in _TERMINAL_STATES):
    raise ValueError("terminal flag is incompatible with operation state")

  operation_id = payload["operation_id"]
  if (
    type(operation_id) is not str
    or _HEX_32_RE.fullmatch(operation_id) is None
  ):
    raise ValueError("operation_id must be 32 lowercase hex characters")
  sequence = _nonnegative_int(payload["sequence"], "sequence")
  started_mono_ns = _nonnegative_int(
    payload["started_mono_ns"],
    "started_mono_ns",
  )
  updated_mono_ns = _nonnegative_int(
    payload["updated_mono_ns"],
    "updated_mono_ns",
  )
  if updated_mono_ns < started_mono_ns:
    raise ValueError("operation update predates operation start")

  vehicle_identity = _optional_identity(
    payload["vehicle_identity"],
    "vehicle_identity",
  )
  runtime_identity = _optional_sha256(
    payload["runtime_identity_sha256"],
    "runtime_identity_sha256",
  )
  current_route = _optional_sha256(
    payload["current_route_identity"],
    "current_route_identity",
  )
  last_route = _optional_sha256(
    payload["last_route_identity"],
    "last_route_identity",
  )
  evidence_identity = _optional_sha256(
    payload["evidence_sha256"],
    "evidence_sha256",
  )
  ledger_identity = _optional_sha256(
    payload["ledger_sha256"],
    "ledger_sha256",
  )

  route_index = payload["current_route_index"]
  route_count = payload["total_route_count"]
  if (route_index is None) != (route_count is None):
    raise ValueError("route index and count must both be null or both present")
  if route_index is not None:
    index = _nonnegative_int(route_index, "current_route_index")
    count = _nonnegative_int(route_count, "total_route_count")
    if index == 0 or count == 0 or index > count or current_route is None:
      raise ValueError("route progress must be one-based and identify a route")
  if diagnostic == "replaying_route" and route_index is None:
    raise ValueError("route replay must expose route progress")
  if diagnostic == "scanning_routes" and (
    current_route is not None or route_index is not None
  ):
    raise ValueError("route scan has no resolved current route")
  if state is LearningOperationState.COLLECTING and (
    route_index is not None
  ):
    raise ValueError("live collection cannot expose backfill route progress")
  if state is not LearningOperationState.COLLECTING and (
    state is not LearningOperationState.BACKFILLING
    or diagnostic != "replaying_route"
  ) and (
    current_route is not None
    or route_index is not None
    or route_count is not None
  ):
    raise ValueError("operation state cannot expose a current route")
  if state in {
    LearningOperationState.PREPARING,
    LearningOperationState.READY_NO_EVIDENCE,
    LearningOperationState.COLLECTING,
    LearningOperationState.BACKFILLING,
  } and last_route is not None:
    raise ValueError("active route state cannot expose a completed route")

  accepted_count = _nonnegative_int(
    payload["accepted_sample_count"],
    "accepted_sample_count",
  )
  rejected_count = _nonnegative_int(
    payload["rejected_sample_count"],
    "rejected_sample_count",
  )
  retry_count = _nonnegative_int(payload["retry_count"], "retry_count")

  runtime_required = state in {
    LearningOperationState.READY_NO_EVIDENCE,
    LearningOperationState.COLLECTING,
    LearningOperationState.FINALIZING,
    LearningOperationState.RETRY_PENDING,
    LearningOperationState.BACKFILLING,
    LearningOperationState.IDLE,
  }
  if runtime_required and (vehicle_identity is None or runtime_identity is None):
    raise ValueError("resolved operation state requires runtime identity")
  if (
    state is LearningOperationState.DRIVE_SKIPPED_IDENTITY_MISMATCH
    and vehicle_identity is None
  ):
    raise ValueError("identity-mismatch skip requires vehicle identity")
  if state is LearningOperationState.READY_NO_EVIDENCE and (
    evidence_identity is not None
    or ledger_identity is not None
    or accepted_count != 0
    or rejected_count != 0
  ):
    raise ValueError("ready-no-evidence state cannot claim evidence")
  if state is LearningOperationState.IDLE and (
    evidence_identity is None
    or current_route is not None
    or route_index is not None
  ):
    raise ValueError("idle state requires committed evidence and no route")
  if ledger_identity is not None and evidence_identity is None:
    raise ValueError("ledger identity cannot exist without evidence")

  return {
    "accepted_sample_count": accepted_count,
    "current_route_identity": current_route,
    "current_route_index": route_index,
    "diagnostic": diagnostic,
    "evidence_sha256": evidence_identity,
    "informational_only": True,
    "last_route_identity": last_route,
    "ledger_sha256": ledger_identity,
    "operation_id": operation_id,
    "rejected_sample_count": rejected_count,
    "retry_count": retry_count,
    "runtime_identity_sha256": runtime_identity,
    "schema_version": LEARNING_OPERATION_STATUS_SCHEMA_VERSION,
    "sequence": sequence,
    "started_mono_ns": started_mono_ns,
    "state": state.value,
    "terminal": terminal,
    "total_route_count": route_count,
    "updated_mono_ns": updated_mono_ns,
    "vehicle_identity": vehicle_identity,
  }


def build_learning_operation_status_bytes(**fields: object) -> bytes:
  payload = validate_learning_operation_status_payload(fields)
  return json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
  ).encode("utf-8")


def decode_learning_operation_status(
  encoded: str | bytes,
) -> dict[str, object]:
  try:
    payload = json.loads(encoded)
  except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise ValueError("learning operation status is not valid JSON") from exc
  return validate_learning_operation_status_payload(payload)


class LearningOperationStatusPublisher:
  """Publish one sequenced operation without making status authoritative."""

  def __init__(
    self,
    params: Any,
    *,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    operation_id_factory: Callable[[], str] = (
      lambda: secrets.token_hex(16)
    ),
  ) -> None:
    self._params = params
    self._monotonic_ns = monotonic_ns
    self._operation_id_factory = operation_id_factory
    self._operation_id: str | None = None
    self._started_mono_ns = 0
    self._sequence = -1
    self._last_payload: dict[str, object] | None = None
    self._route_total: int | None = None
    self._route_index = 0

  @property
  def last_payload(self) -> dict[str, object] | None:
    return None if self._last_payload is None else dict(self._last_payload)

  def publish(
    self,
    *,
    state: LearningOperationState | str,
    diagnostic: str,
    new_operation: bool = False,
    **context: object,
  ) -> bytes:
    now = int(self._monotonic_ns())
    if new_operation or self._operation_id is None:
      operation_id = self._operation_id_factory()
      if (
        type(operation_id) is not str
        or _HEX_32_RE.fullmatch(operation_id) is None
      ):
        raise ValueError("operation id factory returned an invalid identity")
      self._operation_id = operation_id
      self._started_mono_ns = now
      self._sequence = 0
      self._route_total = None
      self._route_index = 0
    else:
      self._sequence += 1

    resolved_state = LearningOperationState(state)
    fields = {
      "accepted_sample_count": 0,
      "current_route_identity": None,
      "current_route_index": None,
      "diagnostic": diagnostic,
      "evidence_sha256": None,
      "informational_only": True,
      "last_route_identity": None,
      "ledger_sha256": None,
      "operation_id": self._operation_id,
      "rejected_sample_count": 0,
      "retry_count": 0,
      "runtime_identity_sha256": None,
      "schema_version": LEARNING_OPERATION_STATUS_SCHEMA_VERSION,
      "sequence": self._sequence,
      "started_mono_ns": self._started_mono_ns,
      "state": resolved_state.value,
      "terminal": resolved_state in _TERMINAL_STATES,
      "total_route_count": None,
      "updated_mono_ns": now,
      "vehicle_identity": None,
    }
    unknown = set(context) - set(fields)
    if unknown:
      raise ValueError(f"unknown operation context: {sorted(unknown)}")
    fields.update(context)
    encoded = build_learning_operation_status_bytes(**fields)
    payload = decode_learning_operation_status(encoded)
    previous = self._last_payload
    if previous is not None and not new_operation:
      if previous["terminal"] is True and payload["terminal"] is False:
        raise ValueError(
          "terminal operation cannot resume without a new operation",
        )
      for name in (
        "accepted_sample_count",
        "rejected_sample_count",
        "retry_count",
      ):
        if payload[name] < previous[name]:
          raise ValueError("operation counters cannot move backwards")
    route_total = payload["total_route_count"]
    route_index = payload["current_route_index"]
    if route_total is not None:
      if self._route_total is None:
        self._route_total = route_total
      elif route_total != self._route_total:
        raise ValueError("operation route total cannot change")
      if route_index < self._route_index:
        raise ValueError("operation route index cannot move backwards")
      self._route_index = route_index
    self._params.put(
      LEARNING_OPERATION_STATUS_PARAM,
      payload,
      block=True,
    )
    self._last_payload = payload
    return encoded

"""Display-only progress for PC-assisted BLaTv2 route preparation.

This Params projection describes transport and certification work that is not
part of the local historical-replay progress contract.  Per-segment detail
remains in ``BLaTv2BackfillProgress``; display consumers may combine the two
optional projections when both belong to the current work.  This projection
is deliberately separate from learning evidence, ledgers, profiles, approval,
and controller selection.  Deleting, editing, or omitting it cannot affect
learning or steering; consumers must treat it as optional diagnostic
information.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
import json
import re
import secrets
import time
from typing import Any


OFFDEVICE_PROGRESS_PARAM = "BLaTv2OffdeviceProgress"
OFFDEVICE_PROGRESS_SCHEMA_VERSION = 2

_HEX_32_RE = re.compile(r"[0-9a-f]{32}")
_TOP_LEVEL_KEYS = {
  "architecture_domain_count",
  "architecture_domain_index",
  "architecture_route_identity_sha256",
  "architecture_segment_count",
  "architecture_segment_index",
  "completed_artifact_count",
  "completed_bytes",
  "certified_domain_count",
  "certified_route_count",
  "fallback_reason_code",
  "informational_only",
  "phase",
  "remote_authority_count",
  "remote_authority_index",
  "remote_only_rejection_excluded_count",
  "remote_route_count",
  "remote_route_index",
  "schema_version",
  "sequence",
  "session_id",
  "total_artifact_count",
  "total_bytes",
  "total_certification_domain_count",
  "total_certification_route_count",
  "updated_mono_ns",
}


class OffdeviceProgressPhase(StrEnum):
  REMOTE_PROCESSING = "remote_processing"
  DOWNLOADING = "downloading"
  ARM_CERTIFYING = "arm_certifying"
  # Wire spelling is retained for display compatibility.  The operation is a
  # bounded cross-architecture verification, not a full ARM route replay.
  ARCHITECTURE_VERIFYING = "arm_certifying"
  REMOTE_READY = "remote_ready"
  LOCAL_FALLBACK = "local_fallback"


class OffdeviceFallbackReason(StrEnum):
  WORKER_UNAVAILABLE = "worker_unavailable"
  WORKER_BUSY = "worker_busy"
  NETWORK_INTERRUPTED = "network_interrupted"
  REMOTE_JOB_FAILED = "remote_job_failed"
  REMOTE_JOB_CANCELED = "remote_job_canceled"
  REMOTE_ROUTE_LIMIT = "remote_route_limit"
  REMOTE_ARTIFACT_UNAVAILABLE = "remote_artifact_unavailable"
  REMOTE_CERTIFICATION_UNAVAILABLE = "remote_certification_unavailable"
  REMOTE_PREPARATION_UNAVAILABLE = "remote_preparation_unavailable"


_PHASE_TRANSITIONS = {
  OffdeviceProgressPhase.REMOTE_PROCESSING: frozenset({
    OffdeviceProgressPhase.REMOTE_PROCESSING,
    OffdeviceProgressPhase.DOWNLOADING,
    OffdeviceProgressPhase.ARM_CERTIFYING,
    OffdeviceProgressPhase.LOCAL_FALLBACK,
  }),
  OffdeviceProgressPhase.DOWNLOADING: frozenset({
    OffdeviceProgressPhase.DOWNLOADING,
    OffdeviceProgressPhase.ARM_CERTIFYING,
    OffdeviceProgressPhase.LOCAL_FALLBACK,
  }),
  OffdeviceProgressPhase.ARM_CERTIFYING: frozenset({
    OffdeviceProgressPhase.ARM_CERTIFYING,
    OffdeviceProgressPhase.REMOTE_READY,
    OffdeviceProgressPhase.LOCAL_FALLBACK,
  }),
  OffdeviceProgressPhase.REMOTE_READY: frozenset({
    OffdeviceProgressPhase.REMOTE_READY,
  }),
  OffdeviceProgressPhase.LOCAL_FALLBACK: frozenset({
    OffdeviceProgressPhase.LOCAL_FALLBACK,
  }),
}

_REMOTE_FIELDS = (
  "remote_authority_count",
  "remote_authority_index",
  "remote_route_count",
  "remote_route_index",
)
_DOWNLOAD_FIELDS = (
  "completed_artifact_count",
  "completed_bytes",
  "total_artifact_count",
  "total_bytes",
)
_CERTIFICATION_FIELDS = (
  "certified_domain_count",
  "certified_route_count",
  "remote_only_rejection_excluded_count",
  "total_certification_domain_count",
  "total_certification_route_count",
)
_ARCHITECTURE_FIELDS = (
  "architecture_domain_count",
  "architecture_domain_index",
  "architecture_route_identity_sha256",
  "architecture_segment_count",
  "architecture_segment_index",
)


def _nonnegative_int(value: object, name: str) -> int:
  if type(value) is not int or value < 0:
    raise ValueError(f"{name} must be a nonnegative integer")
  return value


def _optional_nonnegative_int(value: object, name: str) -> int | None:
  if value is None:
    return None
  return _nonnegative_int(value, name)


def _require_null(payload: dict[str, object], names: tuple[str, ...], phase: str) -> None:
  if any(payload[name] is not None for name in names):
    raise ValueError(f"{phase} cannot publish unrelated progress fields")


def _certification_values(
  payload: dict[str, object],
  *,
  required: bool,
) -> tuple[int, int, int, int, int] | None:
  values = tuple(
    _optional_nonnegative_int(payload[name], name)
    for name in _CERTIFICATION_FIELDS
  )
  present = tuple(value is not None for value in values)
  if required and not all(present):
    raise ValueError("certification progress requires every certification counter")
  if not required and not any(present):
    return None
  if not all(present):
    raise ValueError("certification progress counters must be all present or all null")

  certified_domains, certified_routes, excluded_routes, total_domains, total_routes = values
  assert certified_domains is not None
  assert certified_routes is not None
  assert excluded_routes is not None
  assert total_domains is not None
  assert total_routes is not None
  if total_routes == 0:
    raise ValueError("certification route total must be positive")
  if (
    certified_domains > total_domains
    or certified_routes > total_routes
    or excluded_routes > total_routes
    or certified_routes + excluded_routes > total_routes
  ):
    raise ValueError("certification progress is outside its bounds")
  return (
    certified_domains,
    certified_routes,
    excluded_routes,
    total_domains,
    total_routes,
  )


def _architecture_values(
  payload: dict[str, object],
  *,
  required: bool,
) -> tuple[int, int, str | None, int | None, int | None] | None:
  raw = tuple(payload[name] for name in _ARCHITECTURE_FIELDS)
  if not required and all(value is None for value in raw):
    return None
  domain_count = _nonnegative_int(raw[0], "architecture_domain_count")
  domain_index = _nonnegative_int(raw[1], "architecture_domain_index")
  route_identity = raw[2]
  segment_count = _optional_nonnegative_int(
    raw[3],
    "architecture_segment_count",
  )
  segment_index = _optional_nonnegative_int(
    raw[4],
    "architecture_segment_index",
  )
  if domain_index > domain_count:
    raise ValueError("architecture domain coordinate is outside its bounds")
  if route_identity is not None and (
    type(route_identity) is not str
    or re.fullmatch(r"[0-9a-f]{64}", route_identity) is None
  ):
    raise ValueError("architecture route identity is invalid")
  if (segment_count is None) != (segment_index is None):
    raise ValueError("architecture segment coordinate must be all present or null")
  if segment_count is not None and (
    segment_count == 0
    or segment_index is None
    or not 1 <= segment_index <= segment_count <= 3
  ):
    raise ValueError("architecture segment coordinate is outside its bounds")
  if route_identity is None and segment_count is not None:
    raise ValueError("architecture segment progress requires a route identity")
  if route_identity is not None and domain_index == 0:
    raise ValueError("architecture route requires an active domain")
  return (
    domain_count,
    domain_index,
    route_identity,
    segment_count,
    segment_index,
  )


def validate_offdevice_progress_payload(payload: object) -> dict[str, object]:
  """Validate and normalize the exact informational schema."""
  if type(payload) is not dict or set(payload) != _TOP_LEVEL_KEYS:
    raise ValueError("off-device progress keys do not match")
  if (
    type(payload["schema_version"]) is not int
    or payload["schema_version"] != OFFDEVICE_PROGRESS_SCHEMA_VERSION
    or payload["informational_only"] is not True
  ):
    raise ValueError("off-device progress schema is incompatible")

  session_id = payload["session_id"]
  if type(session_id) is not str or _HEX_32_RE.fullmatch(session_id) is None:
    raise ValueError("session_id must be 32 lowercase hex characters")
  sequence = _nonnegative_int(payload["sequence"], "sequence")
  updated_mono_ns = _nonnegative_int(payload["updated_mono_ns"], "updated_mono_ns")
  try:
    phase = OffdeviceProgressPhase(payload["phase"])
  except (TypeError, ValueError) as exc:
    raise ValueError("off-device progress phase is unknown") from exc

  fallback_reason: str | None = None
  raw_fallback_reason = payload["fallback_reason_code"]
  if phase is OffdeviceProgressPhase.LOCAL_FALLBACK:
    try:
      fallback_reason = OffdeviceFallbackReason(raw_fallback_reason).value
    except (TypeError, ValueError) as exc:
      raise ValueError("local fallback requires a stable reason code") from exc
  elif raw_fallback_reason is not None:
    raise ValueError("fallback reason is valid only during local fallback")

  remote_values = tuple(
    _optional_nonnegative_int(payload[name], name)
    for name in _REMOTE_FIELDS
  )
  download_values = tuple(
    _optional_nonnegative_int(payload[name], name)
    for name in _DOWNLOAD_FIELDS
  )

  if phase is OffdeviceProgressPhase.REMOTE_PROCESSING:
    if any(value is None for value in remote_values):
      raise ValueError("remote processing requires every route coordinate")
    authority_count, authority_index, route_count, route_index = remote_values
    assert authority_count is not None
    assert authority_index is not None
    assert route_count is not None
    assert route_index is not None
    if (
      authority_count != 2
      or route_count == 0
      or authority_index > authority_count
      or route_index > route_count
      or ((authority_index == 0) != (route_index == 0))
    ):
      raise ValueError("remote processing coordinate is outside its bounds")
    _require_null(payload, _DOWNLOAD_FIELDS, phase.value)
    _require_null(payload, _CERTIFICATION_FIELDS, phase.value)
    _require_null(payload, _ARCHITECTURE_FIELDS, phase.value)
  elif phase is OffdeviceProgressPhase.DOWNLOADING:
    if any(value is None for value in download_values):
      raise ValueError("artifact download requires every transfer counter")
    completed_artifacts, completed_bytes, total_artifacts, total_bytes = download_values
    assert completed_artifacts is not None
    assert completed_bytes is not None
    assert total_artifacts is not None
    assert total_bytes is not None
    if (
      total_artifacts == 0
      or total_bytes == 0
      or completed_artifacts > total_artifacts
      or completed_bytes > total_bytes
    ):
      raise ValueError("artifact download progress is outside its bounds")
    _require_null(payload, _REMOTE_FIELDS, phase.value)
    _require_null(payload, _CERTIFICATION_FIELDS, phase.value)
    _require_null(payload, _ARCHITECTURE_FIELDS, phase.value)
  elif phase is OffdeviceProgressPhase.ARM_CERTIFYING:
    _require_null(payload, _REMOTE_FIELDS, phase.value)
    _require_null(payload, _DOWNLOAD_FIELDS, phase.value)
    _certification_values(payload, required=True)
    architecture = _architecture_values(payload, required=True)
    assert architecture is not None
    if architecture[0] != payload["total_certification_domain_count"]:
      raise ValueError("architecture and certification domain totals disagree")
  elif phase is OffdeviceProgressPhase.REMOTE_READY:
    _require_null(payload, _REMOTE_FIELDS, phase.value)
    _require_null(payload, _DOWNLOAD_FIELDS, phase.value)
    certification = _certification_values(payload, required=True)
    assert certification is not None
    certified_domains, certified_routes, excluded_routes, total_domains, total_routes = certification
    if certified_domains != total_domains or certified_routes + excluded_routes != total_routes:
      raise ValueError("remote-ready progress requires complete certification coverage")
    _require_null(payload, _ARCHITECTURE_FIELDS, phase.value)
  else:
    _require_null(payload, _REMOTE_FIELDS, phase.value)
    _require_null(payload, _DOWNLOAD_FIELDS, phase.value)
    _certification_values(payload, required=False)
    _require_null(payload, _ARCHITECTURE_FIELDS, phase.value)

  normalized = dict(payload)
  normalized.update({
    "fallback_reason_code": fallback_reason,
    "informational_only": True,
    "phase": phase.value,
    "schema_version": OFFDEVICE_PROGRESS_SCHEMA_VERSION,
    "sequence": sequence,
    "session_id": session_id,
    "updated_mono_ns": updated_mono_ns,
  })
  return normalized


def build_offdevice_progress_bytes(**fields: object) -> bytes:
  payload = validate_offdevice_progress_payload(fields)
  return json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
  ).encode("utf-8")


def decode_offdevice_progress(encoded: str | bytes) -> dict[str, object]:
  try:
    payload = json.loads(encoded)
  except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise ValueError("off-device progress is not valid JSON") from exc
  return validate_offdevice_progress_payload(payload)


class OffdeviceProgressPublisher:
  """Publish one forward-only, display-only remote-preparation session."""

  def __init__(
    self,
    params: Any,
    *,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    session_id_factory: Callable[[], str] = lambda: secrets.token_hex(16),
  ) -> None:
    self._params = params
    self._monotonic_ns = monotonic_ns
    self._session_id_factory = session_id_factory
    self._session_id: str | None = None
    self._sequence = -1
    self._last_payload: dict[str, object] | None = None

  @property
  def last_payload(self) -> dict[str, object] | None:
    return None if self._last_payload is None else dict(self._last_payload)

  def clear(self) -> None:
    self._params.remove(OFFDEVICE_PROGRESS_PARAM)
    self._session_id = None
    self._sequence = -1
    self._last_payload = None

  @staticmethod
  def _validate_update(
    previous: dict[str, object],
    current: dict[str, object],
  ) -> None:
    previous_phase = OffdeviceProgressPhase(previous["phase"])
    current_phase = OffdeviceProgressPhase(current["phase"])
    if current_phase not in _PHASE_TRANSITIONS[previous_phase]:
      raise ValueError("off-device progress phase moved backward")
    if current["updated_mono_ns"] <= previous["updated_mono_ns"]:
      raise ValueError("off-device progress timestamp did not advance")

    if previous_phase is current_phase is OffdeviceProgressPhase.REMOTE_PROCESSING:
      for name in ("remote_authority_count", "remote_route_count"):
        if current[name] != previous[name]:
          raise ValueError("remote processing inventory changed")
      previous_coordinate = (
        previous["remote_authority_index"],
        previous["remote_route_index"],
      )
      current_coordinate = (
        current["remote_authority_index"],
        current["remote_route_index"],
      )
      if current_coordinate < previous_coordinate:
        raise ValueError("remote processing coordinate moved backward")

    if previous_phase is current_phase is OffdeviceProgressPhase.DOWNLOADING:
      for name in ("total_artifact_count", "total_bytes"):
        if current[name] != previous[name]:
          raise ValueError("artifact download inventory changed")
      for name in ("completed_artifact_count", "completed_bytes"):
        if current[name] < previous[name]:
          raise ValueError("artifact download progress moved backward")

    previous_certification = _certification_values(previous, required=False)
    current_certification = _certification_values(current, required=False)
    if previous_certification is not None:
      if current_certification is None:
        raise ValueError("certification progress disappeared")
      previous_domains, previous_routes, previous_excluded, previous_total_domains, previous_total_routes = previous_certification
      current_domains, current_routes, current_excluded, current_total_domains, current_total_routes = current_certification
      if (
        current_total_domains != previous_total_domains
        or current_total_routes != previous_total_routes
      ):
        raise ValueError("certification inventory changed")
      if (
        current_domains < previous_domains
        or current_routes < previous_routes
        or current_excluded < previous_excluded
      ):
        raise ValueError("certification progress moved backward")

    previous_architecture = _architecture_values(previous, required=False)
    current_architecture = _architecture_values(current, required=False)
    if previous_architecture is not None and current_architecture is not None:
      if current_architecture[0] != previous_architecture[0]:
        raise ValueError("architecture verification inventory changed")
      previous_domain = previous_architecture[1]
      current_domain = current_architecture[1]
      if current_domain < previous_domain:
        raise ValueError("architecture verification domain moved backward")
      if (
        current_domain == previous_domain
        and current_architecture[2] == previous_architecture[2]
        and current_architecture[4] is not None
        and previous_architecture[4] is not None
        and current_architecture[4] < previous_architecture[4]
      ):
        raise ValueError("architecture verification segment moved backward")

    if previous_phase is OffdeviceProgressPhase.LOCAL_FALLBACK:
      if current["fallback_reason_code"] != previous["fallback_reason_code"]:
        raise ValueError("local fallback reason changed")

  def publish(
    self,
    *,
    phase: OffdeviceProgressPhase | str,
    new_session: bool = False,
    remote_authority_count: int | None = None,
    remote_authority_index: int | None = None,
    remote_route_count: int | None = None,
    remote_route_index: int | None = None,
    completed_artifact_count: int | None = None,
    completed_bytes: int | None = None,
    total_artifact_count: int | None = None,
    total_bytes: int | None = None,
    certified_domain_count: int | None = None,
    certified_route_count: int | None = None,
    remote_only_rejection_excluded_count: int | None = None,
    total_certification_domain_count: int | None = None,
    total_certification_route_count: int | None = None,
    architecture_domain_count: int | None = None,
    architecture_domain_index: int | None = None,
    architecture_route_identity_sha256: str | None = None,
    architecture_segment_count: int | None = None,
    architecture_segment_index: int | None = None,
    fallback_reason_code: OffdeviceFallbackReason | str | None = None,
  ) -> bytes:
    resolved_phase = OffdeviceProgressPhase(phase)
    if new_session:
      if resolved_phase not in {
        OffdeviceProgressPhase.REMOTE_PROCESSING,
        OffdeviceProgressPhase.LOCAL_FALLBACK,
      }:
        raise ValueError(
          "a new off-device session must begin with remote processing or local fallback",
        )
      session_id = self._session_id_factory()
      if type(session_id) is not str or _HEX_32_RE.fullmatch(session_id) is None:
        raise ValueError("session id factory returned an invalid identity")
      sequence = 0
      previous = None
    else:
      if self._session_id is None:
        raise ValueError("off-device progress requires an explicit new session")
      session_id = self._session_id
      sequence = self._sequence + 1
      previous = self._last_payload

    now = int(self._monotonic_ns())
    encoded = build_offdevice_progress_bytes(
      architecture_domain_count=architecture_domain_count,
      architecture_domain_index=architecture_domain_index,
      architecture_route_identity_sha256=(
        architecture_route_identity_sha256
      ),
      architecture_segment_count=architecture_segment_count,
      architecture_segment_index=architecture_segment_index,
      completed_artifact_count=completed_artifact_count,
      completed_bytes=completed_bytes,
      certified_domain_count=certified_domain_count,
      certified_route_count=certified_route_count,
      fallback_reason_code=(
        None
        if fallback_reason_code is None
        else OffdeviceFallbackReason(fallback_reason_code).value
      ),
      informational_only=True,
      phase=resolved_phase.value,
      remote_authority_count=remote_authority_count,
      remote_authority_index=remote_authority_index,
      remote_only_rejection_excluded_count=(
        remote_only_rejection_excluded_count
      ),
      remote_route_count=remote_route_count,
      remote_route_index=remote_route_index,
      schema_version=OFFDEVICE_PROGRESS_SCHEMA_VERSION,
      sequence=sequence,
      session_id=session_id,
      total_artifact_count=total_artifact_count,
      total_bytes=total_bytes,
      total_certification_domain_count=total_certification_domain_count,
      total_certification_route_count=total_certification_route_count,
      updated_mono_ns=now,
    )
    payload = decode_offdevice_progress(encoded)
    if previous is not None:
      self._validate_update(previous, payload)

    self._params.put(OFFDEVICE_PROGRESS_PARAM, payload, block=True)
    self._session_id = session_id
    self._sequence = sequence
    self._last_payload = payload
    return encoded

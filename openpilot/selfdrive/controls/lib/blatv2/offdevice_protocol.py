"""Authenticated, canonical wire protocol for off-device BLaTv2 replay.

This module is deliberately standard-library-only and independent of Params,
the learner, messaging, and every actuating path.  It defines protocol version
2 exactly; changing a key, type, limit, or authentication rule requires a new
protocol version.

HMAC authenticates the canonical JSON bytes.  It does not make the LAN traffic
confidential.  The bridge transports recorded route data, never credentials or
an approved controller artifact.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import base64
import hashlib
import hmac
import json
import re
import time
from typing import Final


PROTOCOL_VERSION: Final = 2
DISCOVERY_PORT: Final = 47831
HTTP_PORT: Final = 47830
MAX_CLOCK_SKEW_MS: Final = 30_000
MAX_DISCOVERY_BYTES: Final = 4 * 1024
MAX_CONTROL_BYTES: Final = 5_592_406
MAX_CHUNK_BYTES: Final = 4 * 1024 * 1024
MAX_UPLOAD_REQUEST_BYTES: Final = 6 * 1024 * 1024
MAX_ARTIFACT_BYTES: Final = 512 * 1024 * 1024
MAX_ARTIFACT_HEADER_BYTES: Final = 64 * 1024
MAX_ROUTE_COUNT: Final = 128
MAX_SEGMENT_COUNT: Final = 128
MAX_PROGRESS_COUNT: Final = 10_000_000_000

REQUEST_KEYS: Final = frozenset({
  "client_id",
  "hmac_sha256",
  "nonce",
  "operation",
  "payload",
  "protocol_version",
  "sent_unix_ms",
})
RESPONSE_KEYS: Final = frozenset({
  "client_id",
  "hmac_sha256",
  "payload",
  "protocol_version",
  "request_nonce",
  "sent_unix_ms",
  "service_id",
  "status",
})

OPERATIONS: Final = frozenset({
  "discover",
  "health",
  "route_inventory",
  "job_create",
  "job_status",
  "job_cancel",
  "route_upload",
  "route_commit",
  "artifact_download",
})
JOB_STATES: Final = frozenset({
  "queued",
  "running",
  "completed",
  "cancel_requested",
  "canceled",
  "failed",
})
PROGRESS_PHASES: Final = frozenset({
  "queued",
  "preparing",
  "verifying",
  "complete",
  "canceling",
  "failed",
})
ARTIFACT_DISPOSITIONS: Final = frozenset({"prepared", "rejected"})
ERROR_CODES: Final = frozenset({
  "invalid_payload",
  "source_mismatch",
  "contract_mismatch",
  "route_unavailable",
  "job_not_found",
  "job_conflict",
  "job_failed",
  "upload_invalid",
  "artifact_not_found",
  "artifact_bound_exceeded",
  "busy",
  "internal_error",
})

_HEX_16_RE = re.compile(r"[0-9a-f]{16}")
_HEX_32_RE = re.compile(r"[0-9a-f]{32}")
_HEX_40_RE = re.compile(r"[0-9a-f]{40}")
_HEX_64_RE = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z._-]{0,63}")
_ROUTE_RE = re.compile(r"[0-9a-f]{8}--[0-9a-f]{10}")
_ARCHIVE_RE = re.compile(r"[0-9a-f]{16}_[0-9a-f]{8}--[0-9a-f]{10}")


def unix_time_ms() -> int:
  return time.time_ns() // 1_000_000


class BridgeError(Exception):
  """Base class for stable bridge error categories."""


class BridgeUnavailableError(BridgeError):
  """No authenticated worker was reachable; local replay may be used."""


class BridgeIncompatibleError(BridgeError):
  """An authenticated peer speaks an incompatible contract; fail closed."""


class BridgeCorruptError(BridgeError):
  """Authenticated data is malformed, replayed, or corrupt; fail closed."""


class BridgeAuthenticationError(BridgeCorruptError):
  """The message did not authenticate with the configured secret."""


class BridgeAbortedError(BridgeError):
  """Offroad ownership ended or the caller otherwise canceled the work."""


class BridgeRemoteError(BridgeError):
  """The authenticated worker returned a bounded protocol error."""

  def __init__(self, code: str, message: str) -> None:
    super().__init__(f"{code}: {message}")
    self.code = code
    self.message = message


@dataclass(frozen=True)
class ProtocolLimits:
  discovery_bytes: int = MAX_DISCOVERY_BYTES
  control_bytes: int = MAX_CONTROL_BYTES
  upload_request_bytes: int = MAX_UPLOAD_REQUEST_BYTES
  chunk_bytes: int = MAX_CHUNK_BYTES
  artifact_bytes: int = MAX_ARTIFACT_BYTES
  artifact_header_bytes: int = MAX_ARTIFACT_HEADER_BYTES
  clock_skew_ms: int = MAX_CLOCK_SKEW_MS

  def __post_init__(self) -> None:
    for name, value in self.__dict__.items():
      if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if self.chunk_bytes > MAX_CHUNK_BYTES:
      raise ValueError("chunk_bytes exceeds protocol v1")


def _duplicate_rejecting_object(
  pairs: list[tuple[str, object]],
) -> dict[str, object]:
  result: dict[str, object] = {}
  for key, value in pairs:
    if key in result:
      raise BridgeCorruptError("JSON contains a duplicate key")
    result[key] = value
  return result


def _reject_nonfinite(token: str) -> None:
  raise BridgeCorruptError(f"non-finite JSON number is forbidden: {token}")


def _validate_json_tree(value: object, *, depth: int = 0) -> None:
  if depth > 24:
    raise BridgeCorruptError("JSON nesting exceeds protocol bound")
  if value is None or type(value) in (str, int, bool):
    return
  if type(value) is float:
    raise BridgeCorruptError("floating-point JSON values are forbidden")
  if type(value) is list:
    if len(value) > MAX_PROGRESS_COUNT:
      raise BridgeCorruptError("JSON list exceeds protocol bound")
    for item in value:
      _validate_json_tree(item, depth=depth + 1)
    return
  if type(value) is dict:
    for key, item in value.items():
      if type(key) is not str:
        raise BridgeCorruptError("JSON object key is not a string")
      _validate_json_tree(item, depth=depth + 1)
    return
  raise BridgeCorruptError("JSON contains an unsupported value type")


def canonical_json_bytes(value: object) -> bytes:
  """Encode one finite JSON tree using protocol-v1 canonical bytes."""
  _validate_json_tree(value)
  try:
    return json.dumps(
      value,
      sort_keys=True,
      separators=(",", ":"),
      ensure_ascii=True,
      allow_nan=False,
    ).encode("utf-8")
  except (TypeError, ValueError, RecursionError) as exc:
    raise BridgeCorruptError("value cannot be canonically encoded") from exc


def decode_canonical_json(data: bytes, *, maximum_bytes: int) -> dict[str, object]:
  """Decode strict canonical JSON, rejecting duplicates and alternate bytes."""
  if type(data) is not bytes:
    raise TypeError("canonical JSON input must be bytes")
  if not data or len(data) > maximum_bytes:
    raise BridgeCorruptError("canonical JSON size is outside the bound")
  try:
    decoded = json.loads(
      data.decode("utf-8"),
      object_pairs_hook=_duplicate_rejecting_object,
      parse_constant=_reject_nonfinite,
    )
  except BridgeCorruptError:
    raise
  except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
    raise BridgeCorruptError("invalid canonical JSON") from exc
  if type(decoded) is not dict:
    raise BridgeCorruptError("protocol envelope must be an object")
  _validate_json_tree(decoded)
  if canonical_json_bytes(decoded) != data:
    raise BridgeCorruptError("JSON bytes are not canonical")
  return decoded


def _bounded_string(value: object, name: str, maximum: int) -> str:
  if type(value) is not str or not value or len(value) > maximum:
    raise BridgeCorruptError(f"{name} must be a bounded nonempty string")
  return value


def _matching_string(value: object, name: str, pattern: re.Pattern[str]) -> str:
  text = _bounded_string(value, name, 128)
  if pattern.fullmatch(text) is None:
    raise BridgeCorruptError(f"{name} has an invalid format")
  return text


def _integer(
  value: object,
  name: str,
  *,
  minimum: int = 0,
  maximum: int = MAX_PROGRESS_COUNT,
) -> int:
  if type(value) is not int or not minimum <= value <= maximum:
    raise BridgeCorruptError(f"{name} is outside its integer bound")
  return value


def _exact_dict(
  value: object,
  name: str,
  keys: set[str] | frozenset[str],
) -> dict[str, object]:
  if type(value) is not dict or set(value) != set(keys):
    raise BridgeCorruptError(f"{name} keys do not match protocol v1")
  return value


def _validate_identifier(value: object, name: str) -> str:
  return _matching_string(value, name, _IDENTIFIER_RE)


def _validate_route(value: object) -> str:
  return _matching_string(value, "route_name", _ROUTE_RE)


def _validate_archive_name(value: object) -> str:
  return _matching_string(value, "archive_name", _ARCHIVE_RE)


def _validate_b64(
  value: object,
  name: str,
  maximum_decoded: int,
  *,
  allow_empty: bool = False,
) -> bytes:
  if value == "" and type(value) is str and allow_empty:
    return b""
  text = _bounded_string(value, name, ((maximum_decoded + 2) // 3) * 4 + 4)
  try:
    decoded = base64.b64decode(text, validate=True)
  except (ValueError, base64.binascii.Error) as exc:
    raise BridgeCorruptError(f"{name} is not canonical base64") from exc
  if len(decoded) > maximum_decoded or base64.b64encode(decoded).decode() != text:
    raise BridgeCorruptError(f"{name} exceeds its bound or is noncanonical")
  return decoded


def _validate_contract(value: object) -> None:
  contract = _exact_dict(value, "contract", {
    "car_params_b64",
    "car_params_sha256",
    "descriptor_registry_sha256",
    "dongle_id",
    "historical_descriptor_registry_sha256",
    "opendbc_commit",
    "panda_commit",
    "runtime_identity_sha256",
    "source_commit",
    "vehicle_fingerprint",
  })
  for key in ("source_commit", "opendbc_commit", "panda_commit"):
    _matching_string(contract[key], key, _HEX_40_RE)
  for key in (
    "car_params_sha256",
    "descriptor_registry_sha256",
    "historical_descriptor_registry_sha256",
    "runtime_identity_sha256",
  ):
    _matching_string(contract[key], key, _HEX_64_RE)
  _matching_string(contract["dongle_id"], "dongle_id", _HEX_16_RE)
  fingerprint = _bounded_string(
    contract["vehicle_fingerprint"], "vehicle_fingerprint", 128,
  )
  if any(ord(character) < 0x20 or ord(character) > 0x7e for character in fingerprint):
    raise BridgeCorruptError("vehicle_fingerprint must be printable ASCII")
  car_params = _validate_b64(
    contract["car_params_b64"],
    "car_params_b64",
    1024 * 1024,
  )
  if hashlib.sha256(car_params).hexdigest() != contract["car_params_sha256"]:
    raise BridgeCorruptError("car_params_b64 does not match car_params_sha256")


def validate_request_payload(operation: str, payload: object) -> None:
  """Validate the exact operation request schema."""
  if operation in {"discover", "health"}:
    _exact_dict(payload, f"{operation} payload", set())
  elif operation == "route_inventory":
    body = _exact_dict(payload, "route_inventory payload", {"cursor"})
    if body["cursor"] is not None:
      _validate_archive_name(body["cursor"])
  elif operation == "job_create":
    body = _exact_dict(payload, "job_create payload", {
      "client_job_id", "contract", "routes",
    })
    _matching_string(body["client_job_id"], "client_job_id", _HEX_32_RE)
    routes = body["routes"]
    if type(routes) is not list or not 1 <= len(routes) <= MAX_ROUTE_COUNT:
      raise BridgeCorruptError("job routes are outside the count bound")
    seen: set[str] = set()
    for route_value in routes:
      route = _validate_route(route_value)
      if route in seen:
        raise BridgeCorruptError("job routes contain a duplicate")
      seen.add(route)
    _validate_contract(body["contract"])
  elif operation in {"job_status", "job_cancel"}:
    body = _exact_dict(payload, f"{operation} payload", {"job_id"})
    _matching_string(body["job_id"], "job_id", _HEX_32_RE)
  elif operation == "route_upload":
    body = _exact_dict(payload, "route_upload payload", {
      "chunk_b64",
      "dongle_id",
      "final",
      "offset",
      "route_name",
      "segment_index",
      "segment_sha256",
      "segment_size_bytes",
    })
    _matching_string(body["dongle_id"], "dongle_id", _HEX_16_RE)
    _validate_route(body["route_name"])
    _integer(body["segment_index"], "segment_index", maximum=MAX_SEGMENT_COUNT - 1)
    size = _integer(
      body["segment_size_bytes"],
      "segment_size_bytes",
      minimum=1,
      maximum=MAX_ARTIFACT_BYTES,
    )
    _matching_string(body["segment_sha256"], "segment_sha256", _HEX_64_RE)
    offset = _integer(body["offset"], "offset", maximum=size)
    if type(body["final"]) is not bool:
      raise BridgeCorruptError("final must be a boolean")
    chunk = _validate_b64(
      body["chunk_b64"],
      "chunk_b64",
      MAX_CHUNK_BYTES,
      allow_empty=True,
    )
    if not chunk:
      if offset != 0 or body["final"] is not False:
        raise BridgeCorruptError("empty upload chunk is valid only as a probe")
      return
    if offset + len(chunk) > size:
      raise BridgeCorruptError("upload chunk exceeds the declared segment")
    if body["final"] != (offset + len(chunk) == size):
      raise BridgeCorruptError("final is inconsistent with upload extent")
  elif operation == "route_commit":
    body = _exact_dict(payload, "route_commit payload", {
      "dongle_id", "route_name", "segments",
    })
    _matching_string(body["dongle_id"], "dongle_id", _HEX_16_RE)
    _validate_route(body["route_name"])
    segments = body["segments"]
    if type(segments) is not list or not 1 <= len(segments) <= MAX_SEGMENT_COUNT:
      raise BridgeCorruptError("route commit segment count is outside protocol bounds")
    for expected_index, segment in enumerate(segments):
      _validate_segment_descriptor(segment)
      if segment["index"] != expected_index:
        raise BridgeCorruptError(
          "route commit segments must be contiguous from zero",
        )
  elif operation == "artifact_download":
    body = _exact_dict(payload, "artifact_download payload", {
      "artifact_id", "job_id", "length", "offset",
    })
    _matching_string(body["job_id"], "job_id", _HEX_32_RE)
    _matching_string(body["artifact_id"], "artifact_id", _HEX_64_RE)
    _integer(body["offset"], "offset", maximum=MAX_ARTIFACT_BYTES)
    _integer(body["length"], "length", minimum=1, maximum=MAX_CHUNK_BYTES)
  else:
    raise BridgeIncompatibleError("operation is not part of protocol v1")


def _validate_error_payload(value: object) -> None:
  body = _exact_dict(value, "error payload", {"code", "message"})
  code = _validate_identifier(body["code"], "error code")
  if code not in ERROR_CODES:
    raise BridgeIncompatibleError("error code is not part of protocol v1")
  message = _bounded_string(body["message"], "error message", 256)
  if any(ord(character) < 0x20 and character not in "\t" for character in message):
    raise BridgeCorruptError("error message contains a control character")


def _validate_segment_descriptor(value: object) -> None:
  segment = _exact_dict(value, "segment descriptor", {
    "index", "sha256", "size_bytes",
  })
  _integer(segment["index"], "segment index", maximum=MAX_SEGMENT_COUNT - 1)
  _integer(
    segment["size_bytes"], "segment size", minimum=1,
    maximum=MAX_ARTIFACT_BYTES,
  )
  _matching_string(segment["sha256"], "segment sha256", _HEX_64_RE)


def _validate_route_descriptor(value: object) -> None:
  route = _exact_dict(value, "route descriptor", {
    "archive_name", "complete", "dongle_id", "route_name", "segments",
  })
  archive_name = _validate_archive_name(route["archive_name"])
  dongle_id = _matching_string(route["dongle_id"], "dongle_id", _HEX_16_RE)
  route_name = _validate_route(route["route_name"])
  if archive_name != f"{dongle_id}_{route_name}":
    raise BridgeCorruptError("archive_name does not match route identity")
  if route["complete"] is not True:
    raise BridgeCorruptError("inventory may contain only complete routes")
  segments = route["segments"]
  if type(segments) is not list or not 1 <= len(segments) <= MAX_SEGMENT_COUNT:
    raise BridgeCorruptError("route segment count is outside protocol bounds")
  for expected_index, segment in enumerate(segments):
    _validate_segment_descriptor(segment)
    if segment["index"] != expected_index:
      raise BridgeCorruptError("route segments must be contiguous from zero")


def _validate_progress(value: object) -> None:
  progress = _exact_dict(value, "job progress", {
    "authority_count",
    "authority_index",
    "phase",
    "prepared_frame_count",
    "rejected_route_count",
    "route_count",
    "route_index",
    "route_name",
    "segment_count",
    "segment_index",
  })
  if type(progress["phase"]) is not str or progress["phase"] not in PROGRESS_PHASES:
    raise BridgeIncompatibleError("job progress phase is incompatible")
  authority_index = _integer(progress["authority_index"], "authority_index", maximum=2)
  if progress["authority_count"] != 2 or type(progress["authority_count"]) is not int:
    raise BridgeCorruptError("authority_count must be exactly two")
  route_count = _integer(progress["route_count"], "route_count", maximum=MAX_ROUTE_COUNT)
  route_index = _integer(progress["route_index"], "route_index", maximum=route_count)
  route_name = progress["route_name"]
  if route_index == 0:
    if route_name is not None:
      raise BridgeCorruptError("route_name must be null before route start")
  else:
    _validate_route(route_name)
  segment_count = _integer(
    progress["segment_count"], "segment_count", maximum=MAX_SEGMENT_COUNT,
  )
  segment_index = _integer(
    progress["segment_index"], "segment_index", maximum=segment_count,
  )
  if route_index == 0 and segment_index != 0:
    raise BridgeCorruptError("segment cannot advance before route start")
  _integer(
    progress["prepared_frame_count"], "prepared_frame_count",
    maximum=MAX_PROGRESS_COUNT,
  )
  _integer(
    progress["rejected_route_count"], "rejected_route_count",
    maximum=route_count,
  )
  if authority_index == 0 and progress["phase"] not in {"queued", "canceling", "failed"}:
    raise BridgeCorruptError("active progress requires an authority index")


def _validate_outcome(value: object) -> None:
  if type(value) is not dict:
    raise BridgeCorruptError("job outcome must be an object")
  disposition_value = value.get("disposition")
  if type(disposition_value) is not str or disposition_value not in ARTIFACT_DISPOSITIONS:
    raise BridgeCorruptError("job outcome disposition is invalid")
  disposition = disposition_value
  if disposition == "prepared":
    outcome = _exact_dict(value, "prepared outcome", {
      "authority_index", "certification_vector", "descriptor", "disposition",
      "route_name",
    })
    descriptor = _exact_dict(outcome["descriptor"], "artifact descriptor", {
      "artifact_id",
      "frame_count",
      "provenance",
      "route_name",
      "sha256",
      "size_bytes",
    })
    _matching_string(descriptor["artifact_id"], "artifact_id", _HEX_64_RE)
    _matching_string(descriptor["sha256"], "artifact sha256", _HEX_64_RE)
    _validate_route(descriptor["route_name"])
    if descriptor["route_name"] != outcome["route_name"]:
      raise BridgeCorruptError("artifact route does not match outcome route")
    _integer(
      descriptor["size_bytes"], "artifact size", minimum=1,
      maximum=MAX_ARTIFACT_BYTES,
    )
    _integer(descriptor["frame_count"], "frame_count", maximum=MAX_PROGRESS_COUNT)
    if type(descriptor["provenance"]) is not dict:
      raise BridgeCorruptError("artifact provenance must be an object")
    vector = _exact_dict(
      outcome["certification_vector"],
      "certification vector descriptor",
      {
        "artifact_id",
        "route_name",
        "selected_controls_witnesses",
        "selected_segment_count",
        "selection_identity_sha256",
        "sha256",
        "size_bytes",
      },
    )
    _matching_string(vector["artifact_id"], "vector artifact_id", _HEX_64_RE)
    _matching_string(vector["sha256"], "vector sha256", _HEX_64_RE)
    _matching_string(
      vector["selection_identity_sha256"],
      "vector selection identity",
      _HEX_64_RE,
    )
    _validate_route(vector["route_name"])
    if vector["route_name"] != outcome["route_name"]:
      raise BridgeCorruptError("vector route does not match outcome route")
    _integer(
      vector["size_bytes"],
      "vector artifact size",
      minimum=1,
      maximum=MAX_ARTIFACT_HEADER_BYTES,
    )
    _integer(
      vector["selected_segment_count"],
      "vector selected segment count",
      minimum=1,
      maximum=3,
    )
    _integer(
      vector["selected_controls_witnesses"],
      "vector selected controls witnesses",
      maximum=30_000,
    )
  else:
    outcome = _exact_dict(value, "rejected outcome", {
      "authority_index", "disposition", "message", "reason", "route_name",
    })
    _validate_identifier(outcome["reason"], "rejection reason")
    _bounded_string(outcome["message"], "rejection message", 256)
  if outcome["authority_index"] not in {1, 2} or type(outcome["authority_index"]) is not int:
    raise BridgeCorruptError("outcome authority index must be one or two")
  _validate_route(outcome["route_name"])


def validate_response_payload(operation: str, status: str, payload: object) -> None:
  """Validate an authenticated response using the endpoint operation."""
  if status == "error":
    _validate_error_payload(payload)
    return
  if status != "ok":
    raise BridgeIncompatibleError("response status is incompatible")
  if operation == "discover":
    body = _exact_dict(payload, "discover response", {
      "http_host", "http_port", "protocol_version", "source_commit",
    })
    _bounded_string(body["http_host"], "http_host", 64)
    if body["http_port"] != HTTP_PORT or type(body["http_port"]) is not int:
      raise BridgeIncompatibleError("worker HTTP port is incompatible")
    if body["protocol_version"] != PROTOCOL_VERSION or type(body["protocol_version"]) is not int:
      raise BridgeIncompatibleError("discovery protocol version is incompatible")
    _matching_string(body["source_commit"], "source_commit", _HEX_40_RE)
  elif operation == "health":
    body = _exact_dict(payload, "health response", {
      "historical_descriptor_registry_sha256",
      "opendbc_commit",
      "panda_commit",
      "preparation_implementation_sha256",
      "source_commit",
      "state",
      "worker_extractor_sha256",
      "worker_implementation_commit",
      "worker_implementation_sha256",
      "worker_instance_id",
      "worker_numerical_environment_sha256",
      "worker_preparation_implementation_sha256",
      "worker_count",
    })
    if body["state"] != "ready" or type(body["state"]) is not str:
      raise BridgeIncompatibleError("worker is not ready")
    for key in ("source_commit", "opendbc_commit", "panda_commit"):
      _matching_string(body[key], key, _HEX_40_RE)
    _matching_string(
      body["historical_descriptor_registry_sha256"],
      "historical_descriptor_registry_sha256",
      _HEX_64_RE,
    )
    _matching_string(
      body["worker_extractor_sha256"],
      "worker_extractor_sha256",
      _HEX_64_RE,
    )
    _matching_string(
      body["worker_implementation_commit"],
      "worker_implementation_commit",
      _HEX_40_RE,
    )
    _matching_string(
      body["worker_implementation_sha256"],
      "worker_implementation_sha256",
      _HEX_64_RE,
    )
    for key in (
      "preparation_implementation_sha256",
      "worker_numerical_environment_sha256",
      "worker_preparation_implementation_sha256",
    ):
      _matching_string(body[key], key, _HEX_64_RE)
    _matching_string(body["worker_instance_id"], "worker_instance_id", _HEX_64_RE)
    if body["worker_count"] != 4 or type(body["worker_count"]) is not int:
      raise BridgeIncompatibleError("worker count is incompatible")
  elif operation == "route_inventory":
    body = _exact_dict(
      payload,
      "inventory response",
      {"next_cursor", "routes"},
    )
    routes = body["routes"]
    if type(routes) is not list or len(routes) > MAX_ROUTE_COUNT:
      raise BridgeCorruptError("inventory route count exceeds protocol bound")
    archive_names: list[str] = []
    for route in routes:
      _validate_route_descriptor(route)
      archive_names.append(str(route["archive_name"]))
    if archive_names != sorted(set(archive_names)):
      raise BridgeCorruptError(
        "inventory page is not strictly ordered by archive_name",
      )
    next_cursor = body["next_cursor"]
    if next_cursor is not None:
      cursor = _validate_archive_name(next_cursor)
      if not archive_names or cursor != archive_names[-1]:
        raise BridgeCorruptError(
          "inventory next_cursor must identify the page tail",
        )
  elif operation == "route_upload":
    body = _exact_dict(payload, "upload response", {"complete", "next_offset"})
    if type(body["complete"]) is not bool:
      raise BridgeCorruptError("upload complete must be boolean")
    _integer(body["next_offset"], "next_offset", maximum=MAX_ARTIFACT_BYTES)
  elif operation == "route_commit":
    body = _exact_dict(
      payload,
      "route commit response",
      {"complete", "segment_count"},
    )
    if body["complete"] is not True:
      raise BridgeCorruptError("route commit must report complete=true")
    _integer(
      body["segment_count"],
      "route commit segment_count",
      minimum=1,
      maximum=MAX_SEGMENT_COUNT,
    )
  elif operation == "job_create":
    body = _exact_dict(payload, "job create response", {
      "job_id",
      "preparation_implementation_sha256",
      "route_count",
      "state",
      "worker_implementation_commit",
      "worker_implementation_sha256",
      "worker_instance_id",
      "worker_numerical_environment_sha256",
      "worker_preparation_implementation_sha256",
    })
    _matching_string(body["job_id"], "job_id", _HEX_32_RE)
    _matching_string(
      body["worker_implementation_commit"],
      "worker_implementation_commit",
      _HEX_40_RE,
    )
    _matching_string(
      body["worker_implementation_sha256"],
      "worker_implementation_sha256",
      _HEX_64_RE,
    )
    _matching_string(body["worker_instance_id"], "worker_instance_id", _HEX_64_RE)
    for key in (
      "preparation_implementation_sha256",
      "worker_numerical_environment_sha256",
      "worker_preparation_implementation_sha256",
    ):
      _matching_string(body[key], key, _HEX_64_RE)
    # Idempotent create may return the durable result of an earlier request
    # whose acknowledgement was lost, including a terminal failure/cancel.
    if type(body["state"]) is not str or body["state"] not in JOB_STATES:
      raise BridgeIncompatibleError("job create state is incompatible")
    _integer(body["route_count"], "route_count", minimum=1, maximum=MAX_ROUTE_COUNT)
  elif operation == "job_cancel":
    body = _exact_dict(payload, "job cancel response", {"job_id", "state"})
    _matching_string(body["job_id"], "job_id", _HEX_32_RE)
    if type(body["state"]) is not str or body["state"] not in {"cancel_requested", "canceled", "completed", "failed"}:
      raise BridgeIncompatibleError("job cancel state is incompatible")
  elif operation == "job_status":
    body = _exact_dict(payload, "job status response", {
      "created_unix_ms",
      "error",
      "job_id",
      "outcomes",
      "preparation_implementation_sha256",
      "progress",
      "state",
      "updated_unix_ms",
      "worker_extractor_sha256",
      "worker_implementation_commit",
      "worker_implementation_sha256",
      "worker_instance_id",
      "worker_numerical_environment_sha256",
      "worker_preparation_implementation_sha256",
    })
    _matching_string(body["job_id"], "job_id", _HEX_32_RE)
    if type(body["state"]) is not str or body["state"] not in JOB_STATES:
      raise BridgeIncompatibleError("job status state is incompatible")
    created = _integer(body["created_unix_ms"], "created_unix_ms", maximum=9_999_999_999_999)
    updated = _integer(body["updated_unix_ms"], "updated_unix_ms", maximum=9_999_999_999_999)
    if updated < created:
      raise BridgeCorruptError("job update predates creation")
    if body["error"] is not None:
      _validate_error_payload(body["error"])
    if body["state"] == "failed" and body["error"] is None:
      raise BridgeCorruptError("failed job lacks an error")
    _matching_string(
      body["worker_extractor_sha256"],
      "worker_extractor_sha256",
      _HEX_64_RE,
    )
    _matching_string(
      body["worker_implementation_commit"],
      "worker_implementation_commit",
      _HEX_40_RE,
    )
    _matching_string(
      body["worker_implementation_sha256"],
      "worker_implementation_sha256",
      _HEX_64_RE,
    )
    _matching_string(body["worker_instance_id"], "worker_instance_id", _HEX_64_RE)
    for key in (
      "preparation_implementation_sha256",
      "worker_numerical_environment_sha256",
      "worker_preparation_implementation_sha256",
    ):
      _matching_string(body[key], key, _HEX_64_RE)
    _validate_progress(body["progress"])
    outcomes = body["outcomes"]
    if type(outcomes) is not list or len(outcomes) > 2 * MAX_ROUTE_COUNT:
      raise BridgeCorruptError("job outcome count exceeds protocol bound")
    seen_outcomes: set[tuple[int, str]] = set()
    for outcome in outcomes:
      _validate_outcome(outcome)
      identity = (outcome["authority_index"], outcome["route_name"])
      if identity in seen_outcomes:
        raise BridgeCorruptError("job contains a duplicate outcome")
      seen_outcomes.add(identity)
  elif operation == "artifact_download":
    body = _exact_dict(payload, "artifact response", {
      "artifact_id", "body_sha256", "eof", "length", "offset",
    })
    _matching_string(body["artifact_id"], "artifact_id", _HEX_64_RE)
    _matching_string(body["body_sha256"], "body_sha256", _HEX_64_RE)
    _integer(body["offset"], "offset", maximum=MAX_ARTIFACT_BYTES)
    _integer(body["length"], "length", maximum=MAX_CHUNK_BYTES)
    if type(body["eof"]) is not bool:
      raise BridgeCorruptError("artifact eof must be boolean")
  else:
    raise BridgeIncompatibleError("response operation is not protocol v1")


def _signature(secret: bytes, unsigned: Mapping[str, object]) -> str:
  if type(secret) is not bytes or len(secret) != 32:
    raise BridgeIncompatibleError("bridge secret must contain exactly 32 bytes")
  return hmac.new(secret, canonical_json_bytes(dict(unsigned)), hashlib.sha256).hexdigest()


def build_request(
  *,
  secret: bytes,
  client_id: str,
  operation: str,
  payload: dict[str, object],
  nonce: str,
  sent_unix_ms: int | None = None,
) -> bytes:
  if type(operation) is not str or operation not in OPERATIONS:
    raise BridgeIncompatibleError("operation is not part of protocol v1")
  _validate_identifier(client_id, "client_id")
  _matching_string(nonce, "nonce", _HEX_32_RE)
  validate_request_payload(operation, payload)
  sent = unix_time_ms() if sent_unix_ms is None else sent_unix_ms
  _integer(sent, "sent_unix_ms", maximum=9_999_999_999_999)
  unsigned: dict[str, object] = {
    "client_id": client_id,
    "nonce": nonce,
    "operation": operation,
    "payload": payload,
    "protocol_version": PROTOCOL_VERSION,
    "sent_unix_ms": sent,
  }
  return canonical_json_bytes(unsigned | {"hmac_sha256": _signature(secret, unsigned)})


def _validate_timestamp(sent: object, now_unix_ms: int, skew_ms: int) -> int:
  timestamp = _integer(sent, "sent_unix_ms", maximum=9_999_999_999_999)
  if abs(timestamp - now_unix_ms) > skew_ms:
    raise BridgeCorruptError("message timestamp is stale or from the future")
  return timestamp


def validate_request(
  data: bytes,
  *,
  secret: bytes,
  expected_operation: str | None = None,
  now_unix_ms: int | None = None,
  maximum_bytes: int = MAX_CONTROL_BYTES,
) -> dict[str, object]:
  envelope = decode_canonical_json(data, maximum_bytes=maximum_bytes)
  _exact_dict(envelope, "request envelope", REQUEST_KEYS)
  if type(envelope["protocol_version"]) is not int or envelope["protocol_version"] != PROTOCOL_VERSION:
    raise BridgeIncompatibleError("request protocol version is incompatible")
  operation = envelope["operation"]
  if type(operation) is not str or operation not in OPERATIONS:
    raise BridgeIncompatibleError("request operation is incompatible")
  if expected_operation is not None and operation != expected_operation:
    raise BridgeIncompatibleError("request operation does not match endpoint")
  _validate_identifier(envelope["client_id"], "client_id")
  _matching_string(envelope["nonce"], "nonce", _HEX_32_RE)
  _matching_string(envelope["hmac_sha256"], "hmac_sha256", _HEX_64_RE)
  supplied = envelope["hmac_sha256"]
  unsigned = {key: value for key, value in envelope.items() if key != "hmac_sha256"}
  if not hmac.compare_digest(supplied, _signature(secret, unsigned)):
    raise BridgeAuthenticationError("request HMAC is invalid")
  now = unix_time_ms() if now_unix_ms is None else now_unix_ms
  _validate_timestamp(envelope["sent_unix_ms"], now, MAX_CLOCK_SKEW_MS)
  validate_request_payload(operation, envelope["payload"])
  return envelope


def build_response(
  *,
  secret: bytes,
  service_id: str,
  client_id: str,
  request_nonce: str,
  status: str,
  payload: dict[str, object],
  sent_unix_ms: int | None = None,
) -> bytes:
  _validate_identifier(service_id, "service_id")
  _validate_identifier(client_id, "client_id")
  _matching_string(request_nonce, "request_nonce", _HEX_32_RE)
  if type(status) is not str or status not in {"ok", "error"}:
    raise BridgeIncompatibleError("response status is incompatible")
  if type(payload) is not dict:
    raise BridgeCorruptError("response payload must be an object")
  sent = unix_time_ms() if sent_unix_ms is None else sent_unix_ms
  _integer(sent, "sent_unix_ms", maximum=9_999_999_999_999)
  unsigned: dict[str, object] = {
    "client_id": client_id,
    "payload": payload,
    "protocol_version": PROTOCOL_VERSION,
    "request_nonce": request_nonce,
    "sent_unix_ms": sent,
    "service_id": service_id,
    "status": status,
  }
  return canonical_json_bytes(unsigned | {"hmac_sha256": _signature(secret, unsigned)})


class ResponseReplayGuard:
  """Bounded one-use response-nonce memory for one client process."""

  def __init__(self, maximum_entries: int = 4096) -> None:
    if type(maximum_entries) is not int or maximum_entries <= 0:
      raise ValueError("maximum_entries must be positive")
    self.maximum_entries = maximum_entries
    self._order: list[str] = []
    self._seen: set[str] = set()

  def accept(self, request_nonce: str) -> None:
    if request_nonce in self._seen:
      raise BridgeCorruptError("response nonce was replayed")
    self._seen.add(request_nonce)
    self._order.append(request_nonce)
    if len(self._order) > self.maximum_entries:
      self._seen.remove(self._order.pop(0))


def validate_response(
  data: bytes,
  *,
  secret: bytes,
  expected_client_id: str,
  expected_request_nonce: str,
  expected_service_id: str | None = None,
  now_unix_ms: int | None = None,
  maximum_bytes: int = MAX_CONTROL_BYTES,
  replay_guard: ResponseReplayGuard | None = None,
) -> dict[str, object]:
  envelope = decode_canonical_json(data, maximum_bytes=maximum_bytes)
  _exact_dict(envelope, "response envelope", RESPONSE_KEYS)
  if type(envelope["protocol_version"]) is not int or envelope["protocol_version"] != PROTOCOL_VERSION:
    raise BridgeIncompatibleError("response protocol version is incompatible")
  _validate_identifier(envelope["service_id"], "service_id")
  _validate_identifier(envelope["client_id"], "client_id")
  _matching_string(envelope["request_nonce"], "request_nonce", _HEX_32_RE)
  _matching_string(envelope["hmac_sha256"], "hmac_sha256", _HEX_64_RE)
  supplied = envelope["hmac_sha256"]
  unsigned = {key: value for key, value in envelope.items() if key != "hmac_sha256"}
  if not hmac.compare_digest(supplied, _signature(secret, unsigned)):
    raise BridgeAuthenticationError("response HMAC is invalid")
  if envelope["client_id"] != expected_client_id:
    raise BridgeCorruptError("response client_id does not match request")
  if envelope["request_nonce"] != expected_request_nonce:
    raise BridgeCorruptError("response nonce does not match request")
  if expected_service_id is not None and envelope["service_id"] != expected_service_id:
    raise BridgeCorruptError("response service_id changed")
  status = envelope["status"]
  if type(status) is not str or status not in {"ok", "error"}:
    raise BridgeIncompatibleError("response status is incompatible")
  if type(envelope["payload"]) is not dict:
    raise BridgeCorruptError("response payload must be an object")
  now = unix_time_ms() if now_unix_ms is None else now_unix_ms
  _validate_timestamp(envelope["sent_unix_ms"], now, MAX_CLOCK_SKEW_MS)
  if replay_guard is not None:
    replay_guard.accept(expected_request_nonce)
  return envelope


def response_authenticates(
  data: bytes,
  *,
  secret: bytes,
  maximum_bytes: int,
) -> bool:
  """Return whether canonical bytes carry a valid response HMAC.

  Discovery uses this lightweight check to ignore unauthenticated LAN noise
  without allowing a spoofed incompatible packet to suppress local fallback.
  Once this returns true, the caller must run ``validate_response`` and treat
  every schema/nonce/time failure as authenticated corruption.
  """
  try:
    envelope = decode_canonical_json(data, maximum_bytes=maximum_bytes)
    supplied = envelope.get("hmac_sha256")
    if type(supplied) is not str or _HEX_64_RE.fullmatch(supplied) is None:
      return False
    unsigned = {
      key: value for key, value in envelope.items() if key != "hmac_sha256"
    }
    return hmac.compare_digest(supplied, _signature(secret, unsigned))
  except BridgeError:
    return False


def decode_response_header(value: str, *, maximum_bytes: int) -> bytes:
  """Decode an unpadded base64url signed-response header."""
  if type(value) is not str or not value or len(value) > maximum_bytes * 2:
    raise BridgeCorruptError("artifact response header is outside its bound")
  if re.fullmatch(r"[0-9A-Za-z_-]+", value) is None:
    raise BridgeCorruptError("artifact response header is not base64url")
  padded = value + "=" * (-len(value) % 4)
  try:
    decoded = base64.b64decode(padded, altchars=b"-_", validate=True)
  except (ValueError, base64.binascii.Error) as exc:
    raise BridgeCorruptError("artifact response header is invalid") from exc
  if len(decoded) > maximum_bytes:
    raise BridgeCorruptError("artifact response header exceeds its bound")
  canonical = base64.urlsafe_b64encode(decoded).decode().rstrip("=")
  if canonical != value:
    raise BridgeCorruptError("artifact response header is noncanonical")
  return decoded


def raise_remote_error(payload: object) -> None:
  body = _exact_dict(payload, "error payload", {"code", "message"})
  code = _validate_identifier(body["code"], "error code")
  message = _bounded_string(body["message"], "error message", 512)
  raise BridgeRemoteError(code, message)


def check_abort(abort_requested: Callable[[], bool]) -> None:
  try:
    aborted = abort_requested()
  except Exception as exc:
    raise BridgeAbortedError("abort state could not be verified") from exc
  if type(aborted) is not bool or aborted:
    raise BridgeAbortedError("offroad bridge operation was aborted")

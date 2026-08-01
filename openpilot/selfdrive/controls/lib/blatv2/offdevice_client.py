"""Offroad-only client foundation for the BLaTv2 PC replay worker.

Nothing in this module selects routes, mutates a learning ledger, publishes a
generation, reads/writes Params values, or interacts with cereal.  Callers own
those safety boundaries.  This client only discovers one authenticated worker,
transfers content-addressed bytes, and validates signed protocol-v1 results.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import base64
import hashlib
import http.client
import ipaddress
import os
from pathlib import Path
import secrets
import socket
import stat
import time
from typing import Any, BinaryIO, Final, Protocol

from openpilot.selfdrive.controls.lib.blatv2.offdevice_protocol import (
  DISCOVERY_PORT,
  HTTP_PORT,
  MAX_ARTIFACT_HEADER_BYTES,
  MAX_DISCOVERY_BYTES,
  BridgeCorruptError,
  BridgeError,
  BridgeIncompatibleError,
  BridgeUnavailableError,
  ProtocolLimits,
  ResponseReplayGuard,
  build_request,
  check_abort,
  decode_response_header,
  raise_remote_error,
  response_authenticates,
  validate_response,
  validate_response_payload,
  unix_time_ms,
)


BRIDGE_CONFIG_DIRECTORY: Final = "blatv2-offdevice-bridge"
BRIDGE_SECRET_FILE: Final = "shared_secret.bin"
BRIDGE_WORKER_HOST_FILE: Final = "worker_host.txt"
DISCOVERY_BROADCAST: Final = "255.255.255.255"
DISCOVERY_TIMEOUT_S: Final = 2.0
HTTP_TIMEOUT_S: Final = 10.0
HTTP_POLL_TIMEOUT_S: Final = 0.5
MAX_DISCOVERY_TIMEOUT_S: Final = 5.0
MAX_HTTP_TIMEOUT_S: Final = 30.0
ARTIFACT_RESPONSE_HEADER: Final = "X-BLATV2-Response"
# Upload requests have their own 6 MiB protocol-v1 body bound because binary
# chunks expand under base64. Keep a fixed 4096-byte margin below the 4 MiB
# decoded-chunk bound for canonical envelope metadata and future-safe framing.
MAX_SAFE_UPLOAD_CHUNK_BYTES: Final = 4 * 1024 * 1024 - 4096
_DEFAULT_LIMITS: Final = ProtocolLimits()
_RFC1918_NETWORKS: Final = tuple(
  ipaddress.ip_network(value)
  for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)

_HTTP_ENDPOINTS: Final = {
  "health": "/v1/health",
  "route_inventory": "/v1/routes/inventory",
  "job_create": "/v1/jobs/create",
  "job_status": "/v1/jobs/status",
  "job_cancel": "/v1/jobs/cancel",
  "route_upload": "/v1/routes/upload",
  "route_commit": "/v1/routes/commit",
  "artifact_download": "/v1/artifacts/download",
}


class _SocketLike(Protocol):
  def setsockopt(self, level: int, option: int, value: int) -> None: ...
  def settimeout(self, value: float) -> None: ...
  def sendto(self, data: bytes, address: tuple[str, int]) -> int: ...
  def recvfrom(self, size: int) -> tuple[bytes, tuple[Any, ...]]: ...
  def close(self) -> None: ...


class _HTTPConnectionLike(Protocol):
  def request(
    self,
    method: str,
    url: str,
    body: bytes | None = None,
    headers: Mapping[str, str] | None = None,
  ) -> None: ...
  def getresponse(self) -> Any: ...
  def close(self) -> None: ...


@dataclass(frozen=True)
class DiscoveredWorker:
  host: str
  port: int
  service_id: str
  source_commit: str


def default_bridge_config_directory(params: Any) -> Path:
  """Return the dedicated non-Params secret directory without creating it."""
  active_params_directory = Path(params.get_param_path())
  return active_params_directory.parent / BRIDGE_CONFIG_DIRECTORY


def _validated_bridge_config_directory(config_directory: str | Path) -> Path:
  directory = Path(config_directory)
  try:
    directory_stat = directory.lstat()
  except FileNotFoundError as exc:
    raise BridgeUnavailableError("off-device bridge is not configured") from exc
  except OSError as exc:
    raise BridgeCorruptError("bridge config directory cannot be inspected") from exc
  if not stat.S_ISDIR(directory_stat.st_mode) or directory.is_symlink():
    raise BridgeCorruptError("bridge config directory is not a real directory")
  if stat.S_IMODE(directory_stat.st_mode) != 0o700:
    raise BridgeCorruptError("bridge config directory mode must be 0700")
  if directory_stat.st_uid != os.geteuid():
    raise BridgeCorruptError("bridge config directory owner is incorrect")
  return directory


def load_bridge_secret(config_directory: str | Path) -> bytes:
  """Load the raw 32-byte secret from a 0700 directory and 0600 file.

  Directories require execute permission to be traversable; therefore the
  dedicated directory is exactly 0700 while the credential itself is exactly
  0600.  Missing configuration means the optional PC worker is unavailable.
  Symlinks, wrong ownership, permissive modes, and malformed credentials fail
  closed as corrupt configuration.
  """
  directory = _validated_bridge_config_directory(config_directory)

  secret_path = directory / BRIDGE_SECRET_FILE
  try:
    secret_stat = secret_path.lstat()
  except FileNotFoundError as exc:
    raise BridgeUnavailableError("off-device bridge secret is absent") from exc
  except OSError as exc:
    raise BridgeCorruptError("bridge secret cannot be inspected") from exc
  if not stat.S_ISREG(secret_stat.st_mode) or secret_path.is_symlink():
    raise BridgeCorruptError("bridge secret is not a regular file")
  if stat.S_IMODE(secret_stat.st_mode) != 0o600:
    raise BridgeCorruptError("bridge secret mode must be 0600")
  if secret_stat.st_uid != os.geteuid():
    raise BridgeCorruptError("bridge secret owner is incorrect")
  if secret_stat.st_size != 32:
    raise BridgeCorruptError("bridge secret must contain exactly 32 bytes")
  flags = os.O_RDONLY
  if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
  try:
    descriptor = os.open(secret_path, flags)
    try:
      secret = os.read(descriptor, 33)
      after = os.fstat(descriptor)
    finally:
      os.close(descriptor)
  except OSError as exc:
    raise BridgeCorruptError("bridge secret could not be read safely") from exc
  if len(secret) != 32 or after.st_ino != secret_stat.st_ino:
    raise BridgeCorruptError("bridge secret changed while being read")
  return secret


def load_bridge_secret_for_params(params: Any) -> bytes:
  """Load only from the dedicated sibling of this Params installation."""
  return load_bridge_secret(default_bridge_config_directory(params))


def load_bridge_worker_host(config_directory: str | Path) -> str | None:
  """Load an optional protected static worker IPv4 address.

  Some mixed Ethernet/Wi-Fi networks suppress layer-2 broadcasts even though
  direct LAN traffic is available.  A paired installation may therefore pin
  one worker address beside the secret.  Absence keeps broadcast discovery;
  a present but unsafe file is configuration corruption, never a reason to
  silently ignore the pin.
  """
  directory = _validated_bridge_config_directory(config_directory)
  host_path = directory / BRIDGE_WORKER_HOST_FILE
  try:
    host_stat = host_path.lstat()
  except FileNotFoundError:
    return None
  except OSError as exc:
    raise BridgeCorruptError("bridge worker host cannot be inspected") from exc
  if (
    not stat.S_ISREG(host_stat.st_mode)
    or host_path.is_symlink()
    or stat.S_IMODE(host_stat.st_mode) != 0o600
    or host_stat.st_uid != os.geteuid()
    or not (1 <= host_stat.st_size <= 65)
  ):
    raise BridgeCorruptError("bridge worker host file is not protected")
  flags = os.O_RDONLY
  if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
  try:
    descriptor = os.open(host_path, flags)
    try:
      encoded = os.read(descriptor, 66)
      after = os.fstat(descriptor)
    finally:
      os.close(descriptor)
  except OSError as exc:
    raise BridgeCorruptError("bridge worker host could not be read safely") from exc
  if (
    after.st_dev != host_stat.st_dev
    or after.st_ino != host_stat.st_ino
    or not stat.S_ISREG(after.st_mode)
    or stat.S_IMODE(after.st_mode) != 0o600
    or after.st_uid != os.geteuid()
    or after.st_size != host_stat.st_size
    or len(encoded) != after.st_size
  ):
    raise BridgeCorruptError("bridge worker host changed while being read")
  try:
    text = encoded.decode("ascii")
  except UnicodeDecodeError as exc:
    raise BridgeCorruptError("bridge worker host is not ASCII") from exc
  value = text[:-1] if text.endswith("\n") else text
  if not value or text not in {value, value + "\n"}:
    raise BridgeCorruptError("bridge worker host encoding is not canonical")
  try:
    address = ipaddress.ip_address(value)
  except ValueError as exc:
    raise BridgeCorruptError("bridge worker host is not an IP literal") from exc
  if (
    not isinstance(address, ipaddress.IPv4Address)
    or not any(address in network for network in _RFC1918_NETWORKS)
  ):
    raise BridgeCorruptError("bridge worker host is not an RFC1918 IPv4 address")
  return str(address)


def _default_udp_socket() -> _SocketLike:
  return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def _private_ip_literal(value: object, name: str) -> str:
  if type(value) is not str or not value or len(value) > 64:
    raise BridgeCorruptError(f"{name} is not a bounded IP literal")
  try:
    address = ipaddress.ip_address(value)
  except ValueError as exc:
    raise BridgeCorruptError(f"{name} is not an IP literal") from exc
  if address.is_unspecified or address.is_multicast:
    raise BridgeCorruptError(f"{name} is not a usable private IP literal")
  if not (address.is_private or address.is_loopback):
    raise BridgeCorruptError(f"{name} is outside the private LAN")
  return str(address)


def discover_worker(
  *,
  secret: bytes,
  client_id: str,
  expected_source_commit: str,
  abort_requested: Callable[[], bool],
  configured_host: str | None = None,
  timeout_s: float = DISCOVERY_TIMEOUT_S,
  socket_factory: Callable[[], _SocketLike] = _default_udp_socket,
  monotonic: Callable[[], float] = time.monotonic,
  wall_time_ms: Callable[[], int] = unix_time_ms,
  nonce_factory: Callable[[], str] = lambda: secrets.token_hex(16),
) -> DiscoveredWorker:
  """Discover one authenticated worker or raise ``BridgeUnavailableError``.

  Unauthenticated LAN packets are ignored.  Once a packet authenticates, any
  schema, time, nonce, source, or build mismatch fails closed rather than
  allowing a spoofable downgrade to local replay.  A protected configured
  host replaces broadcast transport on networks that suppress broadcasts;
  the signed discovery exchange and all identity checks remain unchanged.
  """
  if type(timeout_s) not in (int, float) or not 0.0 < timeout_s <= MAX_DISCOVERY_TIMEOUT_S:
    raise ValueError("discovery timeout is outside its bound")
  check_abort(abort_requested)
  target_host = DISCOVERY_BROADCAST
  if configured_host is not None:
    target_host = _private_ip_literal(configured_host, "configured worker host")
    target_address = ipaddress.ip_address(target_host)
    if (
      not isinstance(target_address, ipaddress.IPv4Address)
      or not any(target_address in network for network in _RFC1918_NETWORKS)
    ):
      raise BridgeCorruptError("configured worker host must be RFC1918 IPv4")
  nonce = nonce_factory()
  request = build_request(
    secret=secret,
    client_id=client_id,
    operation="discover",
    payload={},
    nonce=nonce,
    sent_unix_ms=wall_time_ms(),
  )
  if len(request) > MAX_DISCOVERY_BYTES:
    raise BridgeIncompatibleError("discovery request exceeds protocol bound")
  deadline = monotonic() + float(timeout_s)
  try:
    udp_socket = socket_factory()
  except OSError as exc:
    raise BridgeUnavailableError("UDP discovery socket is unavailable") from exc
  try:
    if configured_host is None:
      udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    udp_socket.settimeout(min(0.1, float(timeout_s)))
    sent = udp_socket.sendto(request, (target_host, DISCOVERY_PORT))
    if sent != len(request):
      raise BridgeUnavailableError("UDP discovery request was truncated")
    while monotonic() < deadline:
      check_abort(abort_requested)
      remaining = max(0.001, deadline - monotonic())
      udp_socket.settimeout(min(0.1, remaining))
      try:
        response, source = udp_socket.recvfrom(MAX_DISCOVERY_BYTES + 1)
      except TimeoutError:
        continue
      except OSError as exc:
        raise BridgeUnavailableError("UDP discovery receive failed") from exc
      if len(response) > MAX_DISCOVERY_BYTES:
        continue
      if not response_authenticates(
        response,
        secret=secret,
        maximum_bytes=MAX_DISCOVERY_BYTES,
      ):
        continue
      if (
        type(source) is not tuple
        or len(source) < 2
        or type(source[1]) is not int
        or source[1] != DISCOVERY_PORT
      ):
        raise BridgeCorruptError("authenticated discovery source is malformed")
      source_host = _private_ip_literal(source[0], "UDP source")
      if configured_host is not None and source_host != target_host:
        raise BridgeCorruptError(
          "authenticated discovery source is not the configured worker",
        )
      envelope = validate_response(
        response,
        secret=secret,
        expected_client_id=client_id,
        expected_request_nonce=nonce,
        now_unix_ms=wall_time_ms(),
        maximum_bytes=MAX_DISCOVERY_BYTES,
      )
      validate_response_payload("discover", envelope["status"], envelope["payload"])
      if envelope["status"] == "error":
        raise_remote_error(envelope["payload"])
      payload = envelope["payload"]
      # The advertised host is schema-validated but never used as a redirect;
      # recvfrom is the sole network-address authority.
      _private_ip_literal(payload["http_host"], "advertised HTTP host")
      if payload["source_commit"] != expected_source_commit:
        raise BridgeIncompatibleError("worker source commit does not match device")
      return DiscoveredWorker(
        host=source_host,
        port=HTTP_PORT,
        service_id=envelope["service_id"],
        source_commit=payload["source_commit"],
      )
  except BridgeError:
    raise
  except OSError as exc:
    raise BridgeUnavailableError("UDP discovery is unavailable") from exc
  finally:
    udp_socket.close()
  raise BridgeUnavailableError("no authenticated off-device worker responded")


def _default_http_connection(host: str, port: int, timeout: float) -> _HTTPConnectionLike:
  return http.client.HTTPConnection(host, port, timeout=timeout)


class OffdeviceBridgeClient:
  """Strict authenticated client for one discovered protocol-v1 worker."""

  def __init__(
    self,
    *,
    worker: DiscoveredWorker,
    secret: bytes,
    client_id: str,
    abort_requested: Callable[[], bool],
    timeout_s: float = HTTP_TIMEOUT_S,
    limits: ProtocolLimits = _DEFAULT_LIMITS,
    connection_factory: Callable[[str, int, float], _HTTPConnectionLike] = (
      _default_http_connection
    ),
    wall_time_ms: Callable[[], int] = unix_time_ms,
    nonce_factory: Callable[[], str] = lambda: secrets.token_hex(16),
  ) -> None:
    if type(timeout_s) not in (int, float) or not 0.0 < timeout_s <= MAX_HTTP_TIMEOUT_S:
      raise ValueError("HTTP timeout is outside its bound")
    _private_ip_literal(worker.host, "worker host")
    if worker.port != HTTP_PORT or type(worker.port) is not int:
      raise BridgeIncompatibleError("worker port is incompatible")
    if type(secret) is not bytes or len(secret) != 32:
      raise BridgeIncompatibleError("bridge secret must contain exactly 32 bytes")
    self.worker = worker
    self.secret = secret
    self.client_id = client_id
    self.abort_requested = abort_requested
    self.timeout_s = float(timeout_s)
    self.limits = limits
    self.connection_factory = connection_factory
    self.wall_time_ms = wall_time_ms
    self.nonce_factory = nonce_factory
    self.replay_guard = ResponseReplayGuard()
    self._worker_implementation_identity: tuple[str, str, str] | None = None

  def _check_worker_implementation_identity(
    self,
    payload: Mapping[str, object],
    *,
    establish: bool,
  ) -> None:
    identity = (
      str(payload["worker_instance_id"]),
      str(payload["worker_implementation_commit"]),
      str(payload["worker_implementation_sha256"]),
    )
    expected = self._worker_implementation_identity
    if expected is None:
      if establish:
        self._worker_implementation_identity = identity
        return
      raise BridgeIncompatibleError(
        "worker health identity has not been established",
      )
    if identity != expected:
      raise BridgeUnavailableError(
        "worker instance or implementation identity changed",
      )

  def _require_worker_health_identity(self) -> None:
    if self._worker_implementation_identity is None:
      raise BridgeIncompatibleError(
        "worker health identity must be established before job operations",
      )

  def _read_exact_response(self, response: Any, *, maximum_bytes: int) -> bytes:
    transfer_encoding = response.getheader("Transfer-Encoding")
    if transfer_encoding is not None:
      raise BridgeCorruptError("chunked HTTP responses are forbidden")
    raw_length = response.getheader("Content-Length")
    if type(raw_length) is not str or not raw_length.isascii() or not raw_length.isdigit():
      raise BridgeCorruptError("HTTP response lacks a numeric Content-Length")
    length = int(raw_length)
    if length > maximum_bytes:
      raise BridgeCorruptError("HTTP response exceeds its size bound")
    chunks: list[bytes] = []
    remaining = length
    while remaining:
      check_abort(self.abort_requested)
      try:
        chunk = response.read(min(64 * 1024, remaining))
      except (OSError, TimeoutError, http.client.HTTPException) as exc:
        raise BridgeUnavailableError("HTTP response was interrupted") from exc
      if not chunk:
        raise BridgeCorruptError("HTTP response body was truncated")
      chunks.append(chunk)
      remaining -= len(chunk)
    check_abort(self.abort_requested)
    return b"".join(chunks)

  def _connect_and_post(
    self,
    operation: str,
    encoded_request: bytes,
  ) -> tuple[Any, _HTTPConnectionLike]:
    check_abort(self.abort_requested)
    # Small control requests poll quickly so an offroad->onroad ownership
    # handoff is observed promptly. Multi-megabyte upload/download chunks get
    # the separately bounded transfer timeout; 500 ms is too fragile for
    # ordinary Wi-Fi while abort checks still run between every chunk.
    connection_timeout = (
      self.timeout_s
      if operation in {"route_upload", "artifact_download"}
      else min(self.timeout_s, HTTP_POLL_TIMEOUT_S)
    )
    try:
      connection = self.connection_factory(
        self.worker.host,
        self.worker.port,
        connection_timeout,
      )
      connection.request(
        "POST",
        _HTTP_ENDPOINTS[operation],
        body=encoded_request,
        headers={
          "Accept": "application/json, application/octet-stream",
          "Content-Length": str(len(encoded_request)),
          "Content-Type": "application/json",
        },
      )
      response = connection.getresponse()
      return response, connection
    except BridgeError:
      raise
    except (OSError, TimeoutError, http.client.HTTPException) as exc:
      try:
        connection.close()
      except (NameError, OSError):
        pass
      raise BridgeUnavailableError("off-device HTTP worker is unavailable") from exc

  @staticmethod
  def _classify_unsigned_http_status(status: object) -> None:
    if type(status) is not int:
      raise BridgeCorruptError("HTTP status is malformed")
    if status >= 500:
      raise BridgeUnavailableError("off-device worker has a transient HTTP failure")
    if status == 401:
      raise BridgeCorruptError("off-device worker rejected authentication")
    if status in {400, 404, 405}:
      raise BridgeIncompatibleError("off-device worker rejected protocol v1")
    raise BridgeCorruptError("off-device worker returned an unexpected HTTP status")

  def _request(
    self,
    operation: str,
    payload: dict[str, object],
  ) -> dict[str, object]:
    nonce = self.nonce_factory()
    encoded = build_request(
      secret=self.secret,
      client_id=self.client_id,
      operation=operation,
      payload=payload,
      nonce=nonce,
      sent_unix_ms=self.wall_time_ms(),
    )
    maximum_request = (
      self.limits.upload_request_bytes
      if operation == "route_upload"
      else self.limits.control_bytes
    )
    if len(encoded) > maximum_request:
      raise BridgeIncompatibleError("HTTP request exceeds protocol bound")
    response, connection = self._connect_and_post(operation, encoded)
    try:
      if response.status != 200:
        self._classify_unsigned_http_status(response.status)
      content_type = response.getheader("Content-Type")
      if content_type != "application/json":
        raise BridgeCorruptError("HTTP JSON response has the wrong Content-Type")
      response_bytes = self._read_exact_response(
        response,
        maximum_bytes=self.limits.control_bytes,
      )
    finally:
      connection.close()
    envelope = validate_response(
      response_bytes,
      secret=self.secret,
      expected_client_id=self.client_id,
      expected_request_nonce=nonce,
      expected_service_id=self.worker.service_id,
      now_unix_ms=self.wall_time_ms(),
      maximum_bytes=self.limits.control_bytes,
      replay_guard=self.replay_guard,
    )
    validate_response_payload(operation, envelope["status"], envelope["payload"])
    if envelope["status"] == "error":
      raise_remote_error(envelope["payload"])
    return envelope["payload"]

  def health(self) -> dict[str, object]:
    result = self._request("health", {})
    if result["source_commit"] != self.worker.source_commit:
      raise BridgeIncompatibleError("health source commit changed after discovery")
    self._check_worker_implementation_identity(result, establish=True)
    return result

  def route_inventory(self) -> dict[str, object]:
    """Read the complete append-only archive through bounded pages."""
    routes: list[object] = []
    cursor: str | None = None
    while True:
      page = self._request("route_inventory", {"cursor": cursor})
      page_routes = page["routes"]
      assert type(page_routes) is list
      if page_routes:
        first = str(page_routes[0]["archive_name"])
        if cursor is not None and first <= cursor:
          raise BridgeCorruptError(
            "inventory page did not advance beyond its exclusive cursor",
          )
        if routes and first <= str(routes[-1]["archive_name"]):
          raise BridgeCorruptError(
            "inventory pages are not globally strictly ordered",
          )
        routes.extend(page_routes)
      next_cursor = page["next_cursor"]
      if next_cursor is None:
        return {"routes": routes}
      assert type(next_cursor) is str
      if cursor is not None and next_cursor <= cursor:
        raise BridgeCorruptError("inventory cursor did not advance")
      cursor = next_cursor

  def create_job(
    self,
    *,
    client_job_id: str,
    routes: list[str],
    contract: dict[str, object],
  ) -> dict[str, object]:
    self._require_worker_health_identity()
    result = self._request("job_create", {
      "client_job_id": client_job_id,
      "contract": contract,
      "routes": routes,
    })
    if result["route_count"] != len(routes):
      raise BridgeCorruptError("job acknowledgement route count is incorrect")
    self._check_worker_implementation_identity(result, establish=False)
    return result

  def job_status(self, job_id: str) -> dict[str, object]:
    self._require_worker_health_identity()
    result = self._request("job_status", {"job_id": job_id})
    if result["job_id"] != job_id:
      raise BridgeCorruptError("job status identity does not match request")
    self._check_worker_implementation_identity(result, establish=False)
    return result

  def cancel_job(self, job_id: str) -> dict[str, object]:
    result = self._request("job_cancel", {"job_id": job_id})
    if result["job_id"] != job_id:
      raise BridgeCorruptError("job cancel identity does not match request")
    return result

  def upload_segment(
    self,
    *,
    dongle_id: str,
    route_name: str,
    segment_index: int,
    segment_path: str | Path,
    segment_size_bytes: int,
    segment_sha256: str,
    resume_offset: int = 0,
  ) -> dict[str, object]:
    """Upload one immutable segment, resuming at a server-acknowledged offset."""
    if type(resume_offset) is not int or not 0 <= resume_offset < segment_size_bytes:
      raise BridgeCorruptError("resume offset is outside the segment")
    path = Path(segment_path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
      flags |= os.O_NOFOLLOW
    try:
      descriptor = os.open(path, flags)
    except OSError as exc:
      raise BridgeCorruptError("segment cannot be opened safely") from exc
    try:
      initial = os.fstat(descriptor)
      if not stat.S_ISREG(initial.st_mode) or initial.st_size != segment_size_bytes:
        raise BridgeCorruptError("segment file identity or size is invalid")
      digest = hashlib.sha256()
      while True:
        check_abort(self.abort_requested)
        block = os.read(descriptor, min(1024 * 1024, self.limits.chunk_bytes))
        if not block:
          break
        digest.update(block)
      if digest.hexdigest() != segment_sha256:
        raise BridgeCorruptError("segment bytes do not match the planned SHA-256")
      offset = resume_offset
      final_ack: dict[str, object] | None = None
      if offset == 0:
        # Query the worker's durable partial extent before sending bytes.  The
        # signed zero-byte probe is non-mutating and makes retries after a
        # device/worker/network interruption genuinely resumable.
        probe = self._request("route_upload", {
          "chunk_b64": "",
          "dongle_id": dongle_id,
          "final": False,
          "offset": 0,
          "route_name": route_name,
          "segment_index": segment_index,
          "segment_sha256": segment_sha256,
          "segment_size_bytes": segment_size_bytes,
        })
        probed_offset = probe["next_offset"]
        if not 0 <= probed_offset <= segment_size_bytes:
          raise BridgeCorruptError("upload probe offset is outside the segment")
        if probe["complete"] is not (probed_offset == segment_size_bytes):
          raise BridgeCorruptError("upload probe completion is inconsistent")
        if probe["complete"]:
          return probe
        offset = probed_offset
      os.lseek(descriptor, offset, os.SEEK_SET)
      upload_chunk_bytes = min(
        self.limits.chunk_bytes,
        MAX_SAFE_UPLOAD_CHUNK_BYTES,
      )
      while offset < segment_size_bytes:
        check_abort(self.abort_requested)
        chunk = os.read(
          descriptor,
          min(upload_chunk_bytes, segment_size_bytes - offset),
        )
        if not chunk:
          raise BridgeCorruptError("segment was truncated during upload")
        expected_next = offset + len(chunk)
        final = expected_next == segment_size_bytes
        ack = self._request("route_upload", {
          "chunk_b64": base64.b64encode(chunk).decode("ascii"),
          "dongle_id": dongle_id,
          "final": final,
          "offset": offset,
          "route_name": route_name,
          "segment_index": segment_index,
          "segment_sha256": segment_sha256,
          "segment_size_bytes": segment_size_bytes,
        })
        if ack["next_offset"] != expected_next or ack["complete"] is not final:
          raise BridgeCorruptError("upload acknowledgement does not match sent extent")
        offset = expected_next
        final_ack = ack
      after = os.fstat(descriptor)
      if (
        after.st_ino != initial.st_ino
        or after.st_size != initial.st_size
        or after.st_mtime_ns != initial.st_mtime_ns
      ):
        raise BridgeCorruptError("segment changed during upload")
      if final_ack is None:
        raise BridgeCorruptError("zero-length route segments are forbidden")
      return final_ack
    finally:
      os.close(descriptor)

  def commit_route(
    self,
    *,
    dongle_id: str,
    route_name: str,
    segments: list[dict[str, object]],
  ) -> dict[str, object]:
    """Atomically expose one route only after every staged segment is bound.

    Segment upload acknowledgements prove only that individual segment bytes
    are durable. This explicit route-wide commit prevents an interrupted
    contiguous prefix from ever appearing as a complete archive route.
    """
    result = self._request("route_commit", {
      "dongle_id": dongle_id,
      "route_name": route_name,
      "segments": segments,
    })
    if result["segment_count"] != len(segments):
      raise BridgeCorruptError(
        "route commit acknowledgement has the wrong segment count",
      )
    return result

  def _request_artifact_chunk(
    self,
    *,
    job_id: str,
    artifact_id: str,
    offset: int,
    length: int,
  ) -> tuple[bytes, dict[str, object]]:
    operation = "artifact_download"
    nonce = self.nonce_factory()
    encoded = build_request(
      secret=self.secret,
      client_id=self.client_id,
      operation=operation,
      payload={
        "artifact_id": artifact_id,
        "job_id": job_id,
        "length": length,
        "offset": offset,
      },
      nonce=nonce,
      sent_unix_ms=self.wall_time_ms(),
    )
    response, connection = self._connect_and_post(operation, encoded)
    try:
      if response.status != 200:
        self._classify_unsigned_http_status(response.status)
      content_type = response.getheader("Content-Type")
      if content_type == "application/json":
        response_bytes = self._read_exact_response(
          response,
          maximum_bytes=self.limits.control_bytes,
        )
        envelope = validate_response(
          response_bytes,
          secret=self.secret,
          expected_client_id=self.client_id,
          expected_request_nonce=nonce,
          expected_service_id=self.worker.service_id,
          now_unix_ms=self.wall_time_ms(),
          maximum_bytes=self.limits.control_bytes,
          replay_guard=self.replay_guard,
        )
        validate_response_payload(operation, envelope["status"], envelope["payload"])
        if envelope["status"] == "error":
          raise_remote_error(envelope["payload"])
        raise BridgeCorruptError("artifact endpoint returned JSON success")
      if content_type != "application/octet-stream":
        raise BridgeCorruptError("artifact response has the wrong Content-Type")
      signed_header = response.getheader(ARTIFACT_RESPONSE_HEADER)
      if type(signed_header) is not str:
        raise BridgeCorruptError("artifact response lacks signed metadata")
      header_bytes = decode_response_header(
        signed_header,
        maximum_bytes=min(
          self.limits.artifact_header_bytes,
          MAX_ARTIFACT_HEADER_BYTES,
        ),
      )
      body = self._read_exact_response(
        response,
        maximum_bytes=self.limits.chunk_bytes,
      )
    finally:
      connection.close()
    envelope = validate_response(
      header_bytes,
      secret=self.secret,
      expected_client_id=self.client_id,
      expected_request_nonce=nonce,
      expected_service_id=self.worker.service_id,
      now_unix_ms=self.wall_time_ms(),
      maximum_bytes=self.limits.artifact_header_bytes,
      replay_guard=self.replay_guard,
    )
    validate_response_payload(operation, envelope["status"], envelope["payload"])
    if envelope["status"] == "error":
      raise_remote_error(envelope["payload"])
    metadata = envelope["payload"]
    if (
      metadata["artifact_id"] != artifact_id
      or metadata["offset"] != offset
      or metadata["length"] != len(body)
      or len(body) > length
      or hashlib.sha256(body).hexdigest() != metadata["body_sha256"]
    ):
      raise BridgeCorruptError("artifact bytes do not match signed metadata")
    return body, metadata

  def download_artifact(
    self,
    *,
    job_id: str,
    artifact_id: str,
    expected_size_bytes: int,
    expected_sha256: str,
    sink: BinaryIO | None = None,
  ) -> bytes | None:
    """Download, bound, and hash one artifact without accepting a remote path."""
    if (
      type(expected_size_bytes) is not int
      or not 1 <= expected_size_bytes <= self.limits.artifact_bytes
    ):
      raise BridgeCorruptError("artifact size is outside the configured bound")
    collected = bytearray() if sink is None else None
    digest = hashlib.sha256()
    offset = 0
    while offset < expected_size_bytes:
      check_abort(self.abort_requested)
      requested = min(self.limits.chunk_bytes, expected_size_bytes - offset)
      body, metadata = self._request_artifact_chunk(
        job_id=job_id,
        artifact_id=artifact_id,
        offset=offset,
        length=requested,
      )
      if not body:
        raise BridgeCorruptError("artifact response made no forward progress")
      expected_eof = offset + len(body) == expected_size_bytes
      if metadata["eof"] is not expected_eof:
        raise BridgeCorruptError("artifact EOF does not match expected size")
      digest.update(body)
      if sink is None:
        assert collected is not None
        collected.extend(body)
      else:
        written = sink.write(body)
        if written is not None and written != len(body):
          raise BridgeCorruptError("artifact sink accepted a partial write")
      offset += len(body)
    if digest.hexdigest() != expected_sha256:
      raise BridgeCorruptError("complete artifact SHA-256 is incorrect")
    return bytes(collected) if collected is not None else None

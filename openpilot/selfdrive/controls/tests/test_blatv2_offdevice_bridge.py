from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest  # noqa: TID251

from openpilot.selfdrive.controls.lib.blatv2.offdevice_client import (
  ARTIFACT_RESPONSE_HEADER,
  DISCOVERY_BROADCAST,
  BRIDGE_SECRET_FILE,
  BRIDGE_WORKER_HOST_FILE,
  DiscoveredWorker,
  OffdeviceBridgeClient,
  default_bridge_config_directory,
  discover_worker,
  load_bridge_secret,
  load_bridge_secret_for_params,
  load_bridge_worker_host,
)
from openpilot.selfdrive.controls.lib.blatv2.offdevice_protocol import (
  MAX_CONTROL_BYTES,
  BridgeAbortedError,
  BridgeCorruptError,
  BridgeIncompatibleError,
  BridgeUnavailableError,
  ProtocolLimits,
  ResponseReplayGuard,
  build_request,
  build_response,
  canonical_json_bytes,
  decode_canonical_json,
  validate_request,
  validate_response,
  validate_response_payload,
)


SECRET = bytes(range(32))
CLIENT_ID = "device-client"
SERVICE_ID = "pc-worker"
NONCE = "12" * 16
NOW_MS = 1_800_000_000_000
SOURCE_COMMIT = "a" * 40
OPENDBC_COMMIT = "b" * 40
PANDA_COMMIT = "c" * 40
DESCRIPTOR_SHA = "d" * 64
EXTRACTOR_SHA = "e" * 64
WORKER_INSTANCE_ID = "9" * 64
WORKER_IMPLEMENTATION_COMMIT = "8" * 40
WORKER_IMPLEMENTATION_SHA = "7" * 64
JOB_ID = "34" * 16
ARTIFACT_ID = hashlib.sha256(b"artifact").hexdigest()
ROUTE = "000000d6--b52cbf6188"
DONGLE_ID = "f" * 16
DEFAULT_LIMITS = ProtocolLimits()


def health_payload() -> dict[str, object]:
  return {
    "historical_descriptor_registry_sha256": DESCRIPTOR_SHA,
    "opendbc_commit": OPENDBC_COMMIT,
    "panda_commit": PANDA_COMMIT,
    "source_commit": SOURCE_COMMIT,
    "state": "ready",
    "worker_count": 4,
    "worker_extractor_sha256": EXTRACTOR_SHA,
    "worker_implementation_commit": WORKER_IMPLEMENTATION_COMMIT,
    "worker_implementation_sha256": WORKER_IMPLEMENTATION_SHA,
    "worker_instance_id": WORKER_INSTANCE_ID,
  }


def signed_response(
  *,
  nonce: str = NONCE,
  payload: dict[str, object] | None = None,
  status: str = "ok",
  sent_unix_ms: int = NOW_MS,
) -> bytes:
  return build_response(
    secret=SECRET,
    service_id=SERVICE_ID,
    client_id=CLIENT_ID,
    request_nonce=nonce,
    status=status,
    payload=health_payload() if payload is None else payload,
    sent_unix_ms=sent_unix_ms,
  )


class FakeResponse:
  def __init__(
    self,
    body: bytes,
    *,
    status: int = 200,
    content_type: str = "application/json",
    content_length: int | None = None,
    headers: dict[str, str] | None = None,
    fail_after_reads: int | None = None,
  ) -> None:
    self.body = body
    self.status = status
    self.position = 0
    self.read_count = 0
    self.fail_after_reads = fail_after_reads
    self.headers = {
      "Content-Length": str(len(body) if content_length is None else content_length),
      "Content-Type": content_type,
    }
    if headers is not None:
      self.headers.update(headers)

  def getheader(self, name: str) -> str | None:
    return self.headers.get(name)

  def read(self, size: int) -> bytes:
    self.read_count += 1
    if self.fail_after_reads is not None and self.read_count > self.fail_after_reads:
      raise TimeoutError("interrupted")
    chunk = self.body[self.position:self.position + size]
    self.position += len(chunk)
    return chunk


class FakeConnection:
  def __init__(self, responder, calls: list[dict[str, object]]) -> None:
    self.responder = responder
    self.calls = calls
    self.request_record: dict[str, object] | None = None
    self.closed = False

  def request(self, method, url, body=None, headers=None) -> None:
    self.request_record = {
      "body": body,
      "headers": headers,
      "method": method,
      "url": url,
    }
    self.calls.append(self.request_record)

  def getresponse(self):
    assert self.request_record is not None
    return self.responder(self.request_record)

  def close(self) -> None:
    self.closed = True


class FakeConnectionFactory:
  def __init__(self, responder) -> None:
    self.responder = responder
    self.calls: list[dict[str, object]] = []
    self.timeouts: list[float] = []

  def __call__(self, _host: str, _port: int, timeout: float) -> FakeConnection:
    self.timeouts.append(timeout)
    return FakeConnection(self.responder, self.calls)


def response_for_request(
  request_record: dict[str, object],
  payload: dict[str, object],
  *,
  status: str = "ok",
) -> FakeResponse:
  request = validate_request(
    request_record["body"],
    secret=SECRET,
    now_unix_ms=NOW_MS,
    maximum_bytes=MAX_CONTROL_BYTES,
  )
  return FakeResponse(build_response(
    secret=SECRET,
    service_id=SERVICE_ID,
    client_id=CLIENT_ID,
    request_nonce=request["nonce"],
    status=status,
    payload=payload,
    sent_unix_ms=NOW_MS,
  ))


def health_then_payload_responder(payload: dict[str, object]):
  def responder(request):
    decoded = validate_request(
      request["body"],
      secret=SECRET,
      now_unix_ms=NOW_MS,
      maximum_bytes=MAX_CONTROL_BYTES,
    )
    selected = health_payload() if decoded["operation"] == "health" else payload
    return response_for_request(request, selected)

  return responder


def client(
  responder,
  *,
  abort_requested=lambda: False,
  limits: ProtocolLimits = DEFAULT_LIMITS,
) -> tuple[OffdeviceBridgeClient, FakeConnectionFactory]:
  factory = FakeConnectionFactory(responder)
  nonces = (f"{value:032x}" for value in range(1, 100))
  instance = OffdeviceBridgeClient(
    worker=DiscoveredWorker(
      host="127.0.0.1",
      port=47830,
      service_id=SERVICE_ID,
      source_commit=SOURCE_COMMIT,
    ),
    secret=SECRET,
    client_id=CLIENT_ID,
    abort_requested=abort_requested,
    limits=limits,
    connection_factory=factory,
    wall_time_ms=lambda: NOW_MS,
    nonce_factory=lambda: next(nonces),
  )
  return instance, factory


def test_request_canonical_hmac_and_exact_schema() -> None:
  encoded = build_request(
    secret=SECRET,
    client_id=CLIENT_ID,
    operation="health",
    payload={},
    nonce=NONCE,
    sent_unix_ms=NOW_MS,
  )
  assert hashlib.sha256(encoded).hexdigest() == (
    "8f92400c0c3d13c53a393d88e3233b082f76c0ff264a4e87faae0506074db68e"
  )
  assert b'"hmac_sha256":"dc9e8808554c791b9ae8bb49f9646d5f672cfc640b561defa3788bb200934534"' in encoded
  assert encoded == canonical_json_bytes(json.loads(encoded))
  decoded = validate_request(
    encoded,
    secret=SECRET,
    expected_operation="health",
    now_unix_ms=NOW_MS,
  )
  assert decoded["nonce"] == NONCE

  expanded = json.loads(encoded)
  expanded["unexpected"] = 1
  with pytest.raises(BridgeCorruptError, match="keys"):
    validate_request(
      canonical_json_bytes(expanded), secret=SECRET, now_unix_ms=NOW_MS,
    )


def test_noncanonical_duplicate_nonfinite_and_oversize_json_fail() -> None:
  with pytest.raises(BridgeCorruptError, match="canonical"):
    decode_canonical_json(b'{"b":1, "a":2}', maximum_bytes=100)
  with pytest.raises(BridgeCorruptError, match="duplicate"):
    decode_canonical_json(b'{"a":1,"a":2}', maximum_bytes=100)
  with pytest.raises(BridgeCorruptError, match="non-finite"):
    decode_canonical_json(b'{"a":NaN}', maximum_bytes=100)
  with pytest.raises(BridgeCorruptError, match="size"):
    decode_canonical_json(b"{}", maximum_bytes=1)


def test_tamper_stale_nonce_and_replay_fail_closed() -> None:
  response = signed_response()
  tampered = response.replace(b'"ready"', b'"readdy"')
  with pytest.raises(BridgeCorruptError):
    validate_response(
      tampered,
      secret=SECRET,
      expected_client_id=CLIENT_ID,
      expected_request_nonce=NONCE,
      now_unix_ms=NOW_MS,
    )
  with pytest.raises(BridgeCorruptError, match="nonce"):
    validate_response(
      response,
      secret=SECRET,
      expected_client_id=CLIENT_ID,
      expected_request_nonce="ff" * 16,
      now_unix_ms=NOW_MS,
    )
  stale = signed_response(sent_unix_ms=NOW_MS - 30_001)
  with pytest.raises(BridgeCorruptError, match="stale"):
    validate_response(
      stale,
      secret=SECRET,
      expected_client_id=CLIENT_ID,
      expected_request_nonce=NONCE,
      now_unix_ms=NOW_MS,
    )
  guard = ResponseReplayGuard()
  validate_response(
    response,
    secret=SECRET,
    expected_client_id=CLIENT_ID,
    expected_request_nonce=NONCE,
    now_unix_ms=NOW_MS,
    replay_guard=guard,
  )
  with pytest.raises(BridgeCorruptError, match="replayed"):
    validate_response(
      response,
      secret=SECRET,
      expected_client_id=CLIENT_ID,
      expected_request_nonce=NONCE,
      now_unix_ms=NOW_MS,
      replay_guard=guard,
    )


@pytest.mark.parametrize("bad_mode", [0o644, 0o640, 0o400])
def test_secret_requires_exact_file_mode(tmp_path: Path, bad_mode: int) -> None:
  config = tmp_path / "bridge"
  config.mkdir(mode=0o700)
  secret = config / BRIDGE_SECRET_FILE
  secret.write_bytes(SECRET)
  secret.chmod(bad_mode)
  with pytest.raises(BridgeCorruptError, match="0600"):
    load_bridge_secret(config)


def test_secret_success_missing_bad_directory_and_symlink(tmp_path: Path) -> None:
  with pytest.raises(BridgeUnavailableError):
    load_bridge_secret(tmp_path / "missing")
  config = tmp_path / "bridge"
  config.mkdir(mode=0o700)
  secret = config / BRIDGE_SECRET_FILE
  secret.write_bytes(SECRET)
  secret.chmod(0o600)
  assert load_bridge_secret(config) == SECRET
  config.chmod(0o755)
  with pytest.raises(BridgeCorruptError, match="0700"):
    load_bridge_secret(config)
  config.chmod(0o700)
  secret.unlink()
  target = tmp_path / "target"
  target.write_bytes(SECRET)
  target.chmod(0o600)
  secret.symlink_to(target)
  with pytest.raises(BridgeCorruptError, match="regular"):
    load_bridge_secret(config)


def test_secret_default_is_dedicated_sibling_not_a_params_value(
  tmp_path: Path,
) -> None:
  params_directory = tmp_path / "params" / "d"
  params_directory.mkdir(parents=True)

  class FakeParams:
    def get_param_path(self) -> str:
      return str(params_directory)

  config = default_bridge_config_directory(FakeParams())
  assert config == tmp_path / "params" / "blatv2-offdevice-bridge"
  config.mkdir(mode=0o700)
  secret = config / BRIDGE_SECRET_FILE
  secret.write_bytes(SECRET)
  secret.chmod(0o600)
  assert load_bridge_secret_for_params(FakeParams()) == SECRET


def test_worker_host_is_optional_protected_private_ipv4(tmp_path: Path) -> None:
  config = tmp_path / "bridge"
  config.mkdir(mode=0o700)
  assert load_bridge_worker_host(config) is None

  host = config / BRIDGE_WORKER_HOST_FILE
  host.write_text("192.168.1.241\n", encoding="ascii")
  host.chmod(0o600)
  assert load_bridge_worker_host(config) == "192.168.1.241"

  host.chmod(0o644)
  with pytest.raises(BridgeCorruptError, match="protected"):
    load_bridge_worker_host(config)

  host.chmod(0o600)
  config.chmod(0o755)
  with pytest.raises(BridgeCorruptError, match="0700"):
    load_bridge_worker_host(config)


@pytest.mark.parametrize(
  "encoded",
  [
    b"8.8.8.8\n",
    b"127.0.0.1\n",
    b"::1\n",
    b"192.168.1.241 \n",
    b"192.168.1.241\r\n",
    b"192.168.1.241\n\n",
    b"worker.local\n",
    b"",
    b"1" * 66,
    b"\xff\n",
  ],
)
def test_worker_host_rejects_nonprivate_or_noncanonical_values(
  tmp_path: Path,
  encoded: bytes,
) -> None:
  config = tmp_path / "bridge"
  config.mkdir(mode=0o700)
  host = config / BRIDGE_WORKER_HOST_FILE
  host.write_bytes(encoded)
  host.chmod(0o600)
  with pytest.raises(BridgeCorruptError):
    load_bridge_worker_host(config)


def test_worker_host_rejects_symlink(tmp_path: Path) -> None:
  config = tmp_path / "bridge"
  config.mkdir(mode=0o700)
  target = tmp_path / "host"
  target.write_text("192.168.1.241\n", encoding="ascii")
  target.chmod(0o600)
  (config / BRIDGE_WORKER_HOST_FILE).symlink_to(target)
  with pytest.raises(BridgeCorruptError, match="protected"):
    load_bridge_worker_host(config)


class FakeClock:
  def __init__(self) -> None:
    self.value = 0.0

  def __call__(self) -> float:
    self.value += 0.01
    return self.value


class FakeUDPSocket:
  def __init__(self, responses: list[tuple[bytes, tuple[str, int]]]) -> None:
    self.responses = list(responses)
    self.sent: list[tuple[bytes, tuple[str, int]]] = []
    self.closed = False

  def setsockopt(self, *_args) -> None:
    pass

  def settimeout(self, _value: float) -> None:
    pass

  def sendto(self, data: bytes, address: tuple[str, int]) -> int:
    self.sent.append((data, address))
    return len(data)

  def recvfrom(self, _size: int):
    if not self.responses:
      raise TimeoutError
    return self.responses.pop(0)

  def close(self) -> None:
    self.closed = True


def discovery_response(
  request: bytes,
  *,
  secret: bytes = SECRET,
  nonce: str | None = None,
  source_commit: str = SOURCE_COMMIT,
) -> bytes:
  request_payload = validate_request(
    request,
    secret=SECRET,
    expected_operation="discover",
    now_unix_ms=NOW_MS,
    maximum_bytes=4096,
  )
  return build_response(
    secret=secret,
    service_id=SERVICE_ID,
    client_id=CLIENT_ID,
    request_nonce=request_payload["nonce"] if nonce is None else nonce,
    status="ok",
    payload={
      "http_host": "10.0.0.99",
      "http_port": 47830,
      "protocol_version": 1,
      "source_commit": source_commit,
    },
    sent_unix_ms=NOW_MS,
  )


def test_discovery_ignores_spoof_and_uses_udp_source() -> None:
  fake_socket = FakeUDPSocket([])

  def socket_factory():
    original_send = fake_socket.sendto

    def send(data, address):
      sent = original_send(data, address)
      fake_socket.responses.extend([
        (discovery_response(data, secret=b"x" * 32), ("10.0.0.66", 47831)),
        (discovery_response(data), ("192.168.1.10", 47831)),
      ])
      return sent

    fake_socket.sendto = send
    return fake_socket

  worker = discover_worker(
    secret=SECRET,
    client_id=CLIENT_ID,
    expected_source_commit=SOURCE_COMMIT,
    abort_requested=lambda: False,
    timeout_s=0.2,
    socket_factory=socket_factory,
    monotonic=FakeClock(),
    wall_time_ms=lambda: NOW_MS,
    nonce_factory=lambda: NONCE,
  )
  assert worker.host == "192.168.1.10"
  assert worker.host != "10.0.0.99"
  assert fake_socket.sent[0][1] == (DISCOVERY_BROADCAST, 47831)
  assert fake_socket.closed


def test_configured_discovery_uses_pinned_unicast_and_source() -> None:
  fake_socket = FakeUDPSocket([])

  def socket_factory():
    original_send = fake_socket.sendto

    def send(data, address):
      sent = original_send(data, address)
      fake_socket.responses.append((
        discovery_response(data),
        ("192.168.1.241", 47831),
      ))
      return sent

    fake_socket.sendto = send
    return fake_socket

  worker = discover_worker(
    secret=SECRET,
    client_id=CLIENT_ID,
    expected_source_commit=SOURCE_COMMIT,
    abort_requested=lambda: False,
    configured_host="192.168.1.241",
    timeout_s=0.2,
    socket_factory=socket_factory,
    monotonic=FakeClock(),
    wall_time_ms=lambda: NOW_MS,
    nonce_factory=lambda: NONCE,
  )
  assert worker.host == "192.168.1.241"
  assert fake_socket.sent[0][1] == ("192.168.1.241", 47831)

  wrong_source = FakeUDPSocket([])

  def wrong_source_factory():
    original_send = wrong_source.sendto

    def send(data, address):
      sent = original_send(data, address)
      wrong_source.responses.append((
        discovery_response(data),
        ("192.168.1.242", 47831),
      ))
      return sent

    wrong_source.sendto = send
    return wrong_source

  with pytest.raises(BridgeCorruptError, match="configured worker"):
    discover_worker(
      secret=SECRET,
      client_id=CLIENT_ID,
      expected_source_commit=SOURCE_COMMIT,
      abort_requested=lambda: False,
      configured_host="192.168.1.241",
      timeout_s=0.2,
      socket_factory=wrong_source_factory,
      monotonic=FakeClock(),
      wall_time_ms=lambda: NOW_MS,
      nonce_factory=lambda: NONCE,
    )


def test_configured_discovery_does_not_require_broadcast_socket_option() -> None:
  class NoBroadcastSocket(FakeUDPSocket):
    def setsockopt(self, *_args) -> None:
      raise OSError("broadcast is unavailable")

  configured_socket = NoBroadcastSocket([])

  def configured_factory():
    original_send = configured_socket.sendto

    def send(data, address):
      sent = original_send(data, address)
      configured_socket.responses.append((
        discovery_response(data),
        ("192.168.1.241", 47831),
      ))
      return sent

    configured_socket.sendto = send
    return configured_socket

  assert discover_worker(
    secret=SECRET,
    client_id=CLIENT_ID,
    expected_source_commit=SOURCE_COMMIT,
    abort_requested=lambda: False,
    configured_host="192.168.1.241",
    timeout_s=0.2,
    socket_factory=configured_factory,
    monotonic=FakeClock(),
    wall_time_ms=lambda: NOW_MS,
    nonce_factory=lambda: NONCE,
  ).host == "192.168.1.241"

  with pytest.raises(BridgeUnavailableError):
    discover_worker(
      secret=SECRET,
      client_id=CLIENT_ID,
      expected_source_commit=SOURCE_COMMIT,
      abort_requested=lambda: False,
      timeout_s=0.2,
      socket_factory=lambda: NoBroadcastSocket([]),
      monotonic=FakeClock(),
      wall_time_ms=lambda: NOW_MS,
      nonce_factory=lambda: NONCE,
    )


def test_configured_discovery_wrong_source_port_fails_closed() -> None:
  fake_socket = FakeUDPSocket([])

  def socket_factory():
    original_send = fake_socket.sendto

    def send(data, address):
      sent = original_send(data, address)
      fake_socket.responses.append((
        discovery_response(data),
        ("192.168.1.241", 47832),
      ))
      return sent

    fake_socket.sendto = send
    return fake_socket

  with pytest.raises(BridgeCorruptError, match="source"):
    discover_worker(
      secret=SECRET,
      client_id=CLIENT_ID,
      expected_source_commit=SOURCE_COMMIT,
      abort_requested=lambda: False,
      configured_host="192.168.1.241",
      timeout_s=0.2,
      socket_factory=socket_factory,
      monotonic=FakeClock(),
      wall_time_ms=lambda: NOW_MS,
      nonce_factory=lambda: NONCE,
    )


def test_discovery_authenticated_wrong_nonce_fails_closed() -> None:
  fake_socket = FakeUDPSocket([])

  def socket_factory():
    original_send = fake_socket.sendto

    def send(data, address):
      sent = original_send(data, address)
      fake_socket.responses.append((
        discovery_response(data, nonce="ff" * 16),
        ("192.168.1.10", 47831),
      ))
      return sent

    fake_socket.sendto = send
    return fake_socket

  with pytest.raises(BridgeCorruptError, match="nonce"):
    discover_worker(
      secret=SECRET,
      client_id=CLIENT_ID,
      expected_source_commit=SOURCE_COMMIT,
      abort_requested=lambda: False,
      timeout_s=0.2,
      socket_factory=socket_factory,
      monotonic=FakeClock(),
      wall_time_ms=lambda: NOW_MS,
      nonce_factory=lambda: NONCE,
    )


def test_discovery_no_network_falls_back_and_abort_is_distinct() -> None:
  with pytest.raises(BridgeUnavailableError):
    discover_worker(
      secret=SECRET,
      client_id=CLIENT_ID,
      expected_source_commit=SOURCE_COMMIT,
      abort_requested=lambda: False,
      socket_factory=lambda: (_ for _ in ()).throw(OSError("offline")),
      wall_time_ms=lambda: NOW_MS,
      nonce_factory=lambda: NONCE,
    )
  with pytest.raises(BridgeAbortedError):
    discover_worker(
      secret=SECRET,
      client_id=CLIENT_ID,
      expected_source_commit=SOURCE_COMMIT,
      abort_requested=lambda: True,
      wall_time_ms=lambda: NOW_MS,
      nonce_factory=lambda: NONCE,
    )


def test_health_uses_fixed_path_and_exposes_worker_identities() -> None:
  def responder(request):
    return response_for_request(request, health_payload())

  instance, factory = client(responder)
  result = instance.health()
  assert result["worker_extractor_sha256"] == EXTRACTOR_SHA
  assert result["worker_implementation_commit"] == WORKER_IMPLEMENTATION_COMMIT
  assert result["worker_implementation_sha256"] == WORKER_IMPLEMENTATION_SHA
  assert factory.calls[0]["method"] == "POST"
  assert factory.calls[0]["url"] == "/v1/health"
  assert factory.calls[0]["headers"]["Content-Type"] == "application/json"
  assert factory.timeouts == [0.5]


def test_binary_transfers_use_bounded_wifi_timeout(tmp_path: Path) -> None:
  data = b"one segment"
  segment = tmp_path / "rlog.zst"
  segment.write_bytes(data)
  segment_sha = hashlib.sha256(data).hexdigest()

  def upload_responder(request):
    decoded = validate_request(
      request["body"], secret=SECRET, now_unix_ms=NOW_MS,
      maximum_bytes=MAX_CONTROL_BYTES,
    )
    payload = decoded["payload"]
    chunk = base64.b64decode(payload["chunk_b64"])
    return response_for_request(request, {
      "complete": payload["final"],
      "next_offset": payload["offset"] + len(chunk),
    })

  instance, factory = client(upload_responder)
  instance.upload_segment(
    dongle_id=DONGLE_ID,
    route_name=ROUTE,
    segment_index=0,
    segment_path=segment,
    segment_size_bytes=len(data),
    segment_sha256=segment_sha,
  )
  assert factory.timeouts == [10.0, 10.0]

  artifact = b"prepared route spool"
  instance, factory = client(
    lambda request: artifact_response(request, artifact),
  )
  instance.download_artifact(
    job_id=JOB_ID,
    artifact_id=ARTIFACT_ID,
    expected_size_bytes=len(artifact),
    expected_sha256=hashlib.sha256(artifact).hexdigest(),
  )
  assert factory.timeouts == [10.0]


@pytest.mark.parametrize(
  ("status", "error"),
  [
    (500, BridgeUnavailableError),
    (404, BridgeIncompatibleError),
    (405, BridgeIncompatibleError),
    (401, BridgeCorruptError),
  ],
)
def test_unsigned_http_status_categories(status, error) -> None:
  instance, _ = client(lambda _request: FakeResponse(b"", status=status))
  with pytest.raises(error):
    instance.health()


def test_http_tamper_truncation_oversize_and_wrong_type_fail_closed() -> None:
  good = signed_response(nonce="00" * 15 + "01")

  cases = [
    FakeResponse(good[:-1] + b"0"),
    FakeResponse(good[:-4], content_length=len(good)),
    FakeResponse(good, content_length=MAX_CONTROL_BYTES + 1),
    FakeResponse(good, content_type="text/plain"),
  ]
  for response in cases:
    instance, _ = client(lambda _request, response=response: response)
    with pytest.raises(BridgeCorruptError):
      instance.health()


def test_http_interruption_and_offroad_abort() -> None:
  response = FakeResponse(
    signed_response(nonce="00" * 15 + "01"),
    fail_after_reads=0,
  )
  instance, _ = client(lambda _request: response)
  with pytest.raises(BridgeUnavailableError):
    instance.health()

  calls = 0

  def abort_requested() -> bool:
    nonlocal calls
    calls += 1
    return calls >= 3

  response = FakeResponse(signed_response(nonce="00" * 15 + "01"))
  instance, _ = client(lambda _request: response, abort_requested=abort_requested)
  with pytest.raises(BridgeAbortedError):
    instance.health()


def test_inventory_rejects_archive_path_and_noncontiguous_segments() -> None:
  bad_inventory = {
    "next_cursor": None,
    "routes": [{
      "archive_name": "../../escape",
      "complete": True,
      "dongle_id": DONGLE_ID,
      "route_name": ROUTE,
      "segments": [{
        "index": 1,
        "sha256": "a" * 64,
        "size_bytes": 100,
      }],
    }],
  }
  with pytest.raises(BridgeCorruptError):
    validate_response_payload("route_inventory", "ok", bad_inventory)


def inventory_descriptor(route_number: int) -> dict[str, object]:
  route_name = f"{route_number:08x}--{route_number:010x}"
  return {
    "archive_name": f"{DONGLE_ID}_{route_name}",
    "complete": True,
    "dongle_id": DONGLE_ID,
    "route_name": route_name,
    "segments": [{
      "index": 0,
      "sha256": f"{route_number:064x}",
      "size_bytes": 100,
    }],
  }


def test_inventory_pagination_reads_past_job_route_bound() -> None:
  archive = [inventory_descriptor(index) for index in range(1, 131)]
  cursors: list[str | None] = []

  def responder(request):
    decoded = validate_request(
      request["body"],
      secret=SECRET,
      expected_operation="route_inventory",
      now_unix_ms=NOW_MS,
      maximum_bytes=MAX_CONTROL_BYTES,
    )
    cursor = decoded["payload"]["cursor"]
    cursors.append(cursor)
    start = 0 if cursor is None else 128
    page = archive[start:start + 128]
    next_cursor = page[-1]["archive_name"] if start == 0 else None
    return response_for_request(
      request,
      {"next_cursor": next_cursor, "routes": page},
    )

  instance, factory = client(responder)
  result = instance.route_inventory()

  assert result == {"routes": archive}
  assert cursors == [None, archive[127]["archive_name"]]
  assert len(factory.calls) == 2


def test_inventory_pagination_rejects_replayed_page() -> None:
  page = [inventory_descriptor(1)]

  def responder(request):
    return response_for_request(
      request,
      {"next_cursor": page[-1]["archive_name"], "routes": page},
    )

  instance, _ = client(responder)
  with pytest.raises(BridgeCorruptError, match="advance"):
    instance.route_inventory()


def test_upload_is_resumable_bounded_and_hash_checked(tmp_path: Path) -> None:
  data = b"abcdefghij"
  segment = tmp_path / "rlog.zst"
  segment.write_bytes(data)
  sha = hashlib.sha256(data).hexdigest()
  offsets: list[int] = []

  def responder(request):
    decoded = validate_request(
      request["body"], secret=SECRET, now_unix_ms=NOW_MS,
      maximum_bytes=MAX_CONTROL_BYTES,
    )
    body = decoded["payload"]
    chunk = base64.b64decode(body["chunk_b64"])
    offsets.append(body["offset"])
    return response_for_request(request, {
      "complete": body["final"],
      "next_offset": body["offset"] + len(chunk),
    })

  limits = ProtocolLimits(chunk_bytes=3)
  instance, _ = client(responder, limits=limits)
  result = instance.upload_segment(
    dongle_id=DONGLE_ID,
    route_name=ROUTE,
    segment_index=0,
    segment_path=segment,
    segment_size_bytes=len(data),
    segment_sha256=sha,
    resume_offset=4,
  )
  assert offsets == [4, 7]
  assert result == {"complete": True, "next_offset": 10}

  with pytest.raises(BridgeCorruptError, match="SHA"):
    instance.upload_segment(
      dongle_id=DONGLE_ID,
      route_name=ROUTE,
      segment_index=0,
      segment_path=segment,
      segment_size_bytes=len(data),
      segment_sha256="0" * 64,
    )


def test_upload_probe_resumes_or_accepts_completed_segment(tmp_path: Path) -> None:
  data = b"abcdefghij"
  segment = tmp_path / "rlog.zst"
  segment.write_bytes(data)
  sha = hashlib.sha256(data).hexdigest()
  observed: list[tuple[int, int]] = []

  def responder(request):
    decoded = validate_request(
      request["body"], secret=SECRET, now_unix_ms=NOW_MS,
      maximum_bytes=MAX_CONTROL_BYTES,
    )
    body = decoded["payload"]
    chunk = base64.b64decode(body["chunk_b64"])
    observed.append((body["offset"], len(chunk)))
    next_offset = 4 if not chunk else body["offset"] + len(chunk)
    return response_for_request(request, {
      "complete": next_offset == len(data),
      "next_offset": next_offset,
    })

  instance, _ = client(responder, limits=ProtocolLimits(chunk_bytes=3))
  assert instance.upload_segment(
    dongle_id=DONGLE_ID,
    route_name=ROUTE,
    segment_index=0,
    segment_path=segment,
    segment_size_bytes=len(data),
    segment_sha256=sha,
  ) == {"complete": True, "next_offset": len(data)}
  assert observed == [(0, 0), (4, 3), (7, 3)]

  observed.clear()

  def completed(request):
    decoded = validate_request(
      request["body"], secret=SECRET, now_unix_ms=NOW_MS,
      maximum_bytes=MAX_CONTROL_BYTES,
    )
    chunk = base64.b64decode(decoded["payload"]["chunk_b64"])
    observed.append((decoded["payload"]["offset"], len(chunk)))
    return response_for_request(request, {
      "complete": True,
      "next_offset": len(data),
    })

  instance, _ = client(completed)
  assert instance.upload_segment(
    dongle_id=DONGLE_ID,
    route_name=ROUTE,
    segment_index=0,
    segment_path=segment,
    segment_size_bytes=len(data),
    segment_sha256=sha,
  )["complete"] is True
  assert observed == [(0, 0)]


def test_route_commit_binds_complete_manifest_and_acknowledgement() -> None:
  manifest = [{
    "index": 0,
    "sha256": "1" * 64,
    "size_bytes": 123,
  }, {
    "index": 1,
    "sha256": "2" * 64,
    "size_bytes": 456,
  }]
  observed: list[dict[str, object]] = []

  def responder(request):
    decoded = validate_request(
      request["body"],
      secret=SECRET,
      expected_operation="route_commit",
      now_unix_ms=NOW_MS,
      maximum_bytes=MAX_CONTROL_BYTES,
    )
    observed.append(decoded["payload"])
    return response_for_request(request, {
      "complete": True,
      "segment_count": 2,
    })

  instance, factory = client(responder)
  assert instance.commit_route(
    dongle_id=DONGLE_ID,
    route_name=ROUTE,
    segments=manifest,
  ) == {"complete": True, "segment_count": 2}
  assert observed == [{
    "dongle_id": DONGLE_ID,
    "route_name": ROUTE,
    "segments": manifest,
  }]
  assert factory.calls[0]["url"] == "/v1/routes/commit"

  instance, _ = client(lambda request: response_for_request(request, {
    "complete": True,
    "segment_count": 1,
  }))
  with pytest.raises(BridgeCorruptError, match="segment count"):
    instance.commit_route(
      dongle_id=DONGLE_ID,
      route_name=ROUTE,
      segments=manifest,
    )


def test_upload_rejects_bad_ack_symlink_and_interruption(tmp_path: Path) -> None:
  segment = tmp_path / "rlog.zst"
  segment.write_bytes(b"segment")
  sha = hashlib.sha256(segment.read_bytes()).hexdigest()
  symlink = tmp_path / "link"
  symlink.symlink_to(segment)

  instance, _ = client(lambda request: response_for_request(request, {
    "complete": False,
    "next_offset": 1,
  }))
  with pytest.raises(BridgeCorruptError, match="acknowledgement"):
    instance.upload_segment(
      dongle_id=DONGLE_ID,
      route_name=ROUTE,
      segment_index=0,
      segment_path=segment,
      segment_size_bytes=7,
      segment_sha256=sha,
    )
  with pytest.raises(BridgeCorruptError, match="opened"):
    instance.upload_segment(
      dongle_id=DONGLE_ID,
      route_name=ROUTE,
      segment_index=0,
      segment_path=symlink,
      segment_size_bytes=7,
      segment_sha256=sha,
    )

  acknowledged = False

  def responder(request):
    nonlocal acknowledged
    decoded = validate_request(
      request["body"], secret=SECRET, now_unix_ms=NOW_MS,
      maximum_bytes=MAX_CONTROL_BYTES,
    )
    body = decoded["payload"]
    acknowledged = True
    return response_for_request(request, {
      "complete": body["final"],
      "next_offset": body["offset"] + len(base64.b64decode(body["chunk_b64"])),
    })

  instance, _ = client(
    responder,
    abort_requested=lambda: acknowledged,
    limits=ProtocolLimits(chunk_bytes=3),
  )
  with pytest.raises(BridgeAbortedError):
    instance.upload_segment(
      dongle_id=DONGLE_ID,
      route_name=ROUTE,
      segment_index=0,
      segment_path=segment,
      segment_size_bytes=7,
      segment_sha256=sha,
    )


def artifact_response(request, artifact: bytes) -> FakeResponse:
  decoded = validate_request(
    request["body"], secret=SECRET, now_unix_ms=NOW_MS,
    maximum_bytes=MAX_CONTROL_BYTES,
  )
  payload = decoded["payload"]
  offset = payload["offset"]
  body = artifact[offset:offset + payload["length"]]
  metadata = build_response(
    secret=SECRET,
    service_id=SERVICE_ID,
    client_id=CLIENT_ID,
    request_nonce=decoded["nonce"],
    status="ok",
    payload={
      "artifact_id": payload["artifact_id"],
      "body_sha256": hashlib.sha256(body).hexdigest(),
      "eof": offset + len(body) == len(artifact),
      "length": len(body),
      "offset": offset,
    },
    sent_unix_ms=NOW_MS,
  )
  header = base64.urlsafe_b64encode(metadata).decode().rstrip("=")
  return FakeResponse(
    body,
    content_type="application/octet-stream",
    headers={ARTIFACT_RESPONSE_HEADER: header},
  )


def test_artifact_download_chunks_and_verifies_complete_sha() -> None:
  artifact = b"this is one canonical replay artifact"
  artifact_id = hashlib.sha256(b"id").hexdigest()
  instance, factory = client(
    lambda request: artifact_response(request, artifact),
    limits=ProtocolLimits(chunk_bytes=7),
  )
  result = instance.download_artifact(
    job_id=JOB_ID,
    artifact_id=artifact_id,
    expected_size_bytes=len(artifact),
    expected_sha256=hashlib.sha256(artifact).hexdigest(),
  )
  assert result == artifact
  assert all(call["url"] == "/v1/artifacts/download" for call in factory.calls)


def test_artifact_tamper_truncation_and_oversize_fail_closed() -> None:
  artifact = b"artifact bytes"
  artifact_id = hashlib.sha256(b"id").hexdigest()

  def tampered(request):
    response = artifact_response(request, artifact)
    response.body = response.body + b"x"
    response.headers["Content-Length"] = str(len(response.body))
    return response

  instance, _ = client(tampered)
  with pytest.raises(BridgeCorruptError, match="metadata"):
    instance.download_artifact(
      job_id=JOB_ID,
      artifact_id=artifact_id,
      expected_size_bytes=len(artifact),
      expected_sha256=hashlib.sha256(artifact).hexdigest(),
    )

  instance, _ = client(lambda request: artifact_response(request, artifact))
  with pytest.raises(BridgeCorruptError, match="size"):
    instance.download_artifact(
      job_id=JOB_ID,
      artifact_id=artifact_id,
      expected_size_bytes=513 * 1024 * 1024,
      expected_sha256=hashlib.sha256(artifact).hexdigest(),
    )


def test_job_status_validates_worker_extractor_and_progress() -> None:
  payload = {
    "created_unix_ms": NOW_MS - 100,
    "error": None,
    "job_id": JOB_ID,
    "outcomes": [],
    "progress": {
      "authority_count": 2,
      "authority_index": 1,
      "phase": "preparing",
      "prepared_frame_count": 100,
      "rejected_route_count": 0,
      "route_count": 1,
      "route_index": 1,
      "route_name": ROUTE,
      "segment_count": 2,
      "segment_index": 1,
    },
    "state": "running",
    "updated_unix_ms": NOW_MS,
    "worker_extractor_sha256": EXTRACTOR_SHA,
    "worker_implementation_commit": WORKER_IMPLEMENTATION_COMMIT,
    "worker_implementation_sha256": WORKER_IMPLEMENTATION_SHA,
    "worker_instance_id": WORKER_INSTANCE_ID,
  }
  instance, _ = client(health_then_payload_responder(payload))
  instance.health()
  result = instance.job_status(JOB_ID)
  assert result["worker_extractor_sha256"] == EXTRACTOR_SHA

  invalid = dict(payload)
  invalid.pop("worker_implementation_sha256")
  instance, _ = client(health_then_payload_responder(invalid))
  instance.health()
  with pytest.raises(BridgeCorruptError, match="keys"):
    instance.job_status(JOB_ID)


@pytest.mark.parametrize("state", ["failed", "canceled", "cancel_requested"])
def test_idempotent_job_create_accepts_prior_terminal_state(state: str) -> None:
  instance, _ = client(health_then_payload_responder({
    "job_id": JOB_ID,
    "route_count": 1,
    "state": state,
    "worker_implementation_commit": WORKER_IMPLEMENTATION_COMMIT,
    "worker_implementation_sha256": WORKER_IMPLEMENTATION_SHA,
    "worker_instance_id": WORKER_INSTANCE_ID,
  }))
  instance.health()
  result = instance.create_job(
    client_job_id="12" * 16,
    routes=[ROUTE],
    contract={
      "car_params_b64": base64.b64encode(b"cp").decode("ascii"),
      "car_params_sha256": hashlib.sha256(b"cp").hexdigest(),
      "descriptor_registry_sha256": DESCRIPTOR_SHA,
      "dongle_id": DONGLE_ID,
      "historical_descriptor_registry_sha256": "f" * 64,
      "opendbc_commit": OPENDBC_COMMIT,
      "panda_commit": PANDA_COMMIT,
      "runtime_identity_sha256": "1" * 64,
      "source_commit": SOURCE_COMMIT,
      "vehicle_fingerprint": "HYUNDAI PALISADE 2020",
    },
  )
  assert result["state"] == state


def test_job_operations_require_health_identity_before_network_request() -> None:
  def unexpected_request(_request):
    raise AssertionError("job request must not precede worker health")

  instance, factory = client(unexpected_request)
  with pytest.raises(BridgeIncompatibleError, match="health identity"):
    instance.create_job(client_job_id="12" * 16, routes=[ROUTE], contract={})
  with pytest.raises(BridgeIncompatibleError, match="health identity"):
    instance.job_status(JOB_ID)
  assert factory.calls == []


def test_client_detects_mid_instance_worker_implementation_change() -> None:
  calls = 0

  def responder(request):
    nonlocal calls
    calls += 1
    payload = health_payload()
    if calls == 2:
      payload["worker_implementation_sha256"] = "6" * 64
    return response_for_request(request, payload)

  instance, _ = client(responder)
  instance.health()
  with pytest.raises(BridgeUnavailableError, match="implementation"):
    instance.health()


def test_client_binds_job_create_to_health_worker_implementation() -> None:
  def responder(request):
    decoded = validate_request(
      request["body"],
      secret=SECRET,
      now_unix_ms=NOW_MS,
      maximum_bytes=MAX_CONTROL_BYTES,
    )
    if decoded["operation"] == "health":
      payload = health_payload()
    else:
      payload = {
        "job_id": JOB_ID,
        "route_count": 1,
        "state": "queued",
        "worker_implementation_commit": WORKER_IMPLEMENTATION_COMMIT,
        "worker_implementation_sha256": "6" * 64,
        "worker_instance_id": WORKER_INSTANCE_ID,
      }
    return response_for_request(request, payload)

  instance, _ = client(responder)
  instance.health()
  with pytest.raises(BridgeUnavailableError, match="implementation"):
    instance.create_job(
      client_job_id="12" * 16,
      routes=[ROUTE],
      contract={
        "car_params_b64": base64.b64encode(b"cp").decode("ascii"),
        "car_params_sha256": hashlib.sha256(b"cp").hexdigest(),
        "descriptor_registry_sha256": DESCRIPTOR_SHA,
        "dongle_id": DONGLE_ID,
        "historical_descriptor_registry_sha256": "f" * 64,
        "opendbc_commit": OPENDBC_COMMIT,
        "panda_commit": PANDA_COMMIT,
        "runtime_identity_sha256": "1" * 64,
        "source_commit": SOURCE_COMMIT,
        "vehicle_fingerprint": "HYUNDAI PALISADE 2020",
      },
    )


def test_authenticated_error_is_bounded_and_unsigned_remote_cannot_redirect() -> None:
  error_payload = {"code": "busy", "message": "worker is busy"}
  instance, _ = client(
    lambda request: response_for_request(request, error_payload, status="error"),
  )
  with pytest.raises(Exception, match="busy") as raised:
    instance.health()
  assert raised.type.__name__ == "BridgeRemoteError"

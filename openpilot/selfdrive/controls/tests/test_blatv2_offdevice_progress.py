from __future__ import annotations

import json

import pytest  # noqa: TID251

from openpilot.selfdrive.controls.lib.blatv2.offdevice_progress import (
  OFFDEVICE_PROGRESS_PARAM,
  OFFDEVICE_PROGRESS_SCHEMA_VERSION,
  OffdeviceFallbackReason,
  OffdeviceProgressPhase,
  OffdeviceProgressPublisher,
  build_offdevice_progress_bytes,
  decode_offdevice_progress,
  validate_offdevice_progress_payload,
)


SESSION_ID = "12" * 16


class FakeParams:
  def __init__(self) -> None:
    self.values: dict[str, object] = {
      "BLaTv2LearningEvidence": b"unchanged",
    }
    self.puts: list[tuple[str, dict[str, object], bool]] = []
    self.removes: list[str] = []

  def put(
    self,
    key: str,
    value: dict[str, object],
    *,
    block: bool,
  ) -> None:
    self.values[key] = dict(value)
    self.puts.append((key, dict(value), block))

  def remove(self, key: str) -> None:
    self.values.pop(key, None)
    self.removes.append(key)


def empty_payload(**changes: object) -> dict[str, object]:
  payload: dict[str, object] = {
    "architecture_domain_count": None,
    "architecture_domain_index": None,
    "architecture_route_identity_sha256": None,
    "architecture_segment_count": None,
    "architecture_segment_index": None,
    "completed_artifact_count": None,
    "completed_bytes": None,
    "certified_domain_count": None,
    "certified_route_count": None,
    "fallback_reason_code": None,
    "informational_only": True,
    "phase": "remote_processing",
    "remote_authority_count": 2,
    "remote_authority_index": 0,
    "remote_only_rejection_excluded_count": None,
    "remote_route_count": 38,
    "remote_route_index": 0,
    "schema_version": OFFDEVICE_PROGRESS_SCHEMA_VERSION,
    "sequence": 0,
    "session_id": SESSION_ID,
    "total_artifact_count": None,
    "total_bytes": None,
    "total_certification_domain_count": None,
    "total_certification_route_count": None,
    "updated_mono_ns": 100,
  }
  payload.update(changes)
  return payload


def downloading_payload(**changes: object) -> dict[str, object]:
  payload = empty_payload(
    completed_artifact_count=3,
    completed_bytes=300,
    phase="downloading",
    remote_authority_count=None,
    remote_authority_index=None,
    remote_route_count=None,
    remote_route_index=None,
    total_artifact_count=36,
    total_bytes=1_770_000_000,
  )
  payload.update(changes)
  return payload


def certifying_payload(**changes: object) -> dict[str, object]:
  payload = empty_payload(
    architecture_domain_count=5,
    architecture_domain_index=2,
    certified_domain_count=2,
    certified_route_count=17,
    phase="arm_certifying",
    remote_authority_count=None,
    remote_authority_index=None,
    remote_only_rejection_excluded_count=1,
    remote_route_count=None,
    remote_route_index=None,
    total_certification_domain_count=5,
    total_certification_route_count=38,
  )
  payload.update(changes)
  return payload


def ready_payload(**changes: object) -> dict[str, object]:
  payload = certifying_payload(
    architecture_domain_count=None,
    architecture_domain_index=None,
    certified_domain_count=5,
    certified_route_count=37,
    phase="remote_ready",
  )
  payload.update(changes)
  return payload


def fallback_payload(**changes: object) -> dict[str, object]:
  payload = empty_payload(
    fallback_reason_code="worker_unavailable",
    phase="local_fallback",
    remote_authority_count=None,
    remote_authority_index=None,
    remote_route_count=None,
    remote_route_index=None,
  )
  payload.update(changes)
  return payload


def publish_remote(
  publisher: OffdeviceProgressPublisher,
  *,
  new_session: bool = False,
  authority: int = 1,
  route: int = 1,
) -> dict[str, object]:
  return decode_offdevice_progress(publisher.publish(
    phase=OffdeviceProgressPhase.REMOTE_PROCESSING,
    new_session=new_session,
    remote_authority_count=2,
    remote_authority_index=authority,
    remote_route_count=38,
    remote_route_index=route,
  ))


def publish_download(
  publisher: OffdeviceProgressPublisher,
  *,
  artifacts: int,
  byte_count: int,
) -> dict[str, object]:
  return decode_offdevice_progress(publisher.publish(
    phase=OffdeviceProgressPhase.DOWNLOADING,
    completed_artifact_count=artifacts,
    completed_bytes=byte_count,
    total_artifact_count=36,
    total_bytes=1_770_000_000,
  ))


def publish_certification(
  publisher: OffdeviceProgressPublisher,
  *,
  domains: int,
  routes: int,
  excluded: int,
) -> dict[str, object]:
  return decode_offdevice_progress(publisher.publish(
    phase=OffdeviceProgressPhase.ARM_CERTIFYING,
    architecture_domain_count=5,
    architecture_domain_index=domains,
    certified_domain_count=domains,
    certified_route_count=routes,
    remote_only_rejection_excluded_count=excluded,
    total_certification_domain_count=5,
    total_certification_route_count=38,
  ))


def test_schema_is_exact_canonical_and_display_only() -> None:
  payload = empty_payload()
  encoded = build_offdevice_progress_bytes(**payload)

  assert encoded == json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
  ).encode()
  assert decode_offdevice_progress(encoded) == payload
  assert decode_offdevice_progress(encoded.decode()) == payload
  assert validate_offdevice_progress_payload(payload) == payload
  with pytest.raises(ValueError, match="keys do not match"):
    validate_offdevice_progress_payload({**payload, "extra": 1})
  with pytest.raises(ValueError, match="keys do not match"):
    validate_offdevice_progress_payload({key: value for key, value in payload.items() if key != "phase"})


@pytest.mark.parametrize(
  "payload",
  (
    empty_payload(),
    empty_payload(remote_authority_index=2, remote_route_index=38),
    downloading_payload(),
    certifying_payload(),
    ready_payload(),
    fallback_payload(),
    fallback_payload(
      certified_domain_count=2,
      certified_route_count=17,
      remote_only_rejection_excluded_count=1,
      total_certification_domain_count=5,
      total_certification_route_count=38,
    ),
  ),
)
def test_every_legal_phase_shape_round_trips(payload: dict[str, object]) -> None:
  assert decode_offdevice_progress(
    build_offdevice_progress_bytes(**payload),
  ) == payload


@pytest.mark.parametrize(
  ("changes", "message"),
  (
    ({"schema_version": 3}, "schema"),
    ({"schema_version": True}, "schema"),
    ({"informational_only": False}, "schema"),
    ({"session_id": "A" * 32}, "session_id"),
    ({"sequence": True}, "nonnegative"),
    ({"sequence": -1}, "nonnegative"),
    ({"updated_mono_ns": -1}, "nonnegative"),
    ({"phase": "unknown"}, "phase"),
    ({"remote_authority_count": 3}, "coordinate"),
    ({"remote_authority_index": 3}, "coordinate"),
    ({"remote_route_count": 0}, "coordinate"),
    ({"remote_route_index": 39}, "coordinate"),
    ({"remote_authority_index": 0, "remote_route_index": 1}, "coordinate"),
  ),
)
def test_exact_types_and_remote_bounds_fail_closed(
  changes: dict[str, object],
  message: str,
) -> None:
  with pytest.raises(ValueError, match=message):
    build_offdevice_progress_bytes(**empty_payload(**changes))


@pytest.mark.parametrize(
  ("payload", "message"),
  (
    (
      empty_payload(completed_artifact_count=0),
      "unrelated",
    ),
    (
      downloading_payload(remote_route_count=38),
      "unrelated",
    ),
    (
      downloading_payload(remote_only_rejection_excluded_count=1),
      "unrelated",
    ),
    (
      certifying_payload(completed_bytes=0),
      "unrelated",
    ),
    (
      ready_payload(fallback_reason_code="worker_busy"),
      "fallback reason",
    ),
    (
      fallback_payload(completed_artifact_count=1),
      "unrelated",
    ),
  ),
)
def test_cross_phase_fields_are_rejected(
  payload: dict[str, object],
  message: str,
) -> None:
  with pytest.raises(ValueError, match=message):
    build_offdevice_progress_bytes(**payload)


@pytest.mark.parametrize(
  ("payload", "message"),
  (
    (downloading_payload(total_artifact_count=0), "download"),
    (downloading_payload(total_bytes=0), "download"),
    (downloading_payload(completed_artifact_count=37), "download"),
    (downloading_payload(completed_bytes=1_770_000_001), "download"),
    (certifying_payload(certified_domain_count=6), "certification"),
    (certifying_payload(certified_route_count=38), "certification"),
    (certifying_payload(remote_only_rejection_excluded_count=22), "certification"),
    (certifying_payload(certified_route_count=None), "every certification"),
    (ready_payload(certified_domain_count=4), "complete certification"),
    (ready_payload(certified_route_count=36), "complete certification"),
    (fallback_payload(fallback_reason_code=None), "stable reason"),
    (fallback_payload(fallback_reason_code="free-form text"), "stable reason"),
    (
      fallback_payload(remote_only_rejection_excluded_count=1),
      "all present or all null",
    ),
  ),
)
def test_phase_counter_bounds_and_fallback_reason_fail_closed(
  payload: dict[str, object],
  message: str,
) -> None:
  with pytest.raises(ValueError, match=message):
    build_offdevice_progress_bytes(**payload)


def test_remote_only_exclusions_are_explicit_in_certification_ready_and_fallback() -> None:
  certification = certifying_payload(remote_only_rejection_excluded_count=20)
  assert decode_offdevice_progress(
    build_offdevice_progress_bytes(**certification),
  )["remote_only_rejection_excluded_count"] == 20


def test_architecture_verification_reports_active_route_and_vector_segment() -> None:
  payload = certifying_payload(
    architecture_domain_index=3,
    architecture_route_identity_sha256="ab" * 32,
    architecture_segment_count=3,
    architecture_segment_index=2,
    certified_domain_count=2,
  )
  decoded = decode_offdevice_progress(
    build_offdevice_progress_bytes(**payload),
  )
  assert decoded["phase"] == "arm_certifying"
  assert decoded["architecture_domain_index"] == 3
  assert decoded["architecture_route_identity_sha256"] == "ab" * 32
  assert decoded["architecture_segment_index"] == 2
  assert decoded["architecture_segment_count"] == 3

  with pytest.raises(ValueError, match="outside its bounds"):
    build_offdevice_progress_bytes(**{
      **payload,
      "architecture_segment_index": 4,
    })
  with pytest.raises(ValueError, match="requires a route"):
    build_offdevice_progress_bytes(**{
      **payload,
      "architecture_route_identity_sha256": None,
    })

  ready = ready_payload(certified_route_count=18, remote_only_rejection_excluded_count=20)
  assert decode_offdevice_progress(
    build_offdevice_progress_bytes(**ready),
  )["remote_only_rejection_excluded_count"] == 20

  fallback = fallback_payload(
    certified_domain_count=2,
    certified_route_count=17,
    remote_only_rejection_excluded_count=20,
    total_certification_domain_count=5,
    total_certification_route_count=38,
  )
  assert decode_offdevice_progress(
    build_offdevice_progress_bytes(**fallback),
  )["remote_only_rejection_excluded_count"] == 20


def test_publisher_requires_explicit_session_and_follows_complete_forward_path() -> None:
  params = FakeParams()
  publisher = OffdeviceProgressPublisher(
    params,
    monotonic_ns=iter((100, 110, 120, 130, 140, 150, 160)).__next__,
    session_id_factory=lambda: SESSION_ID,
  )
  with pytest.raises(ValueError, match="explicit new session"):
    publish_remote(publisher)

  first = publish_remote(publisher, new_session=True, authority=0, route=0)
  second = publish_remote(publisher, authority=1, route=1)
  third = publish_remote(publisher, authority=2, route=1)
  fourth = publish_download(publisher, artifacts=0, byte_count=0)
  fifth = publish_download(publisher, artifacts=36, byte_count=1_770_000_000)
  sixth = publish_certification(publisher, domains=2, routes=17, excluded=1)
  seventh = decode_offdevice_progress(publisher.publish(
    phase=OffdeviceProgressPhase.REMOTE_READY,
    certified_domain_count=5,
    certified_route_count=37,
    remote_only_rejection_excluded_count=1,
    total_certification_domain_count=5,
    total_certification_route_count=38,
  ))

  assert [item["sequence"] for item in (first, second, third, fourth, fifth, sixth, seventh)] == list(range(7))
  assert seventh["phase"] == "remote_ready"
  assert publisher.last_payload == seventh
  assert all(key == OFFDEVICE_PROGRESS_PARAM and block for key, _, block in params.puts)


def test_publisher_allows_direct_fallback_when_discovery_never_started() -> None:
  params = FakeParams()
  publisher = OffdeviceProgressPublisher(
    params,
    monotonic_ns=lambda: 100,
    session_id_factory=lambda: SESSION_ID,
  )
  payload = decode_offdevice_progress(publisher.publish(
    phase=OffdeviceProgressPhase.LOCAL_FALLBACK,
    new_session=True,
    fallback_reason_code=OffdeviceFallbackReason.WORKER_UNAVAILABLE,
  ))

  assert payload["sequence"] == 0
  assert payload["phase"] == "local_fallback"
  assert payload["remote_route_count"] is None
  assert payload["fallback_reason_code"] == "worker_unavailable"


@pytest.mark.parametrize(
  "start_phase",
  (
    OffdeviceProgressPhase.DOWNLOADING,
    OffdeviceProgressPhase.ARM_CERTIFYING,
    OffdeviceProgressPhase.REMOTE_READY,
  ),
)
def test_new_session_cannot_invent_midstream_progress(start_phase: OffdeviceProgressPhase) -> None:
  publisher = OffdeviceProgressPublisher(FakeParams())
  with pytest.raises(ValueError, match="must begin"):
    publisher.publish(phase=start_phase, new_session=True)


def test_local_fallback_is_terminal_and_reason_is_stable() -> None:
  publisher = OffdeviceProgressPublisher(
    FakeParams(),
    monotonic_ns=iter((100, 110, 120, 130, 140)).__next__,
    session_id_factory=lambda: SESSION_ID,
  )
  publish_remote(publisher, new_session=True)
  first = decode_offdevice_progress(publisher.publish(
    phase=OffdeviceProgressPhase.LOCAL_FALLBACK,
    fallback_reason_code=OffdeviceFallbackReason.NETWORK_INTERRUPTED,
  ))
  same = decode_offdevice_progress(publisher.publish(
    phase=OffdeviceProgressPhase.LOCAL_FALLBACK,
    fallback_reason_code=OffdeviceFallbackReason.NETWORK_INTERRUPTED,
  ))
  assert same["sequence"] == first["sequence"] + 1

  with pytest.raises(ValueError, match="reason changed"):
    publisher.publish(
      phase=OffdeviceProgressPhase.LOCAL_FALLBACK,
      fallback_reason_code=OffdeviceFallbackReason.WORKER_BUSY,
    )
  with pytest.raises(ValueError, match="phase moved backward"):
    publish_remote(publisher)


def test_publisher_rejects_regressed_coordinates_counters_inventory_and_clock() -> None:
  publisher = OffdeviceProgressPublisher(
    FakeParams(),
    monotonic_ns=iter((100, 110, 120, 130, 140, 150, 160, 150)).__next__,
    session_id_factory=lambda: SESSION_ID,
  )
  publish_remote(publisher, new_session=True, authority=1, route=2)
  with pytest.raises(ValueError, match="coordinate moved backward"):
    publish_remote(publisher, authority=1, route=1)
  publish_download(publisher, artifacts=2, byte_count=200)
  with pytest.raises(ValueError, match="inventory changed"):
    publisher.publish(
      phase=OffdeviceProgressPhase.DOWNLOADING,
      completed_artifact_count=2,
      completed_bytes=200,
      total_artifact_count=35,
      total_bytes=1_770_000_000,
    )
  with pytest.raises(ValueError, match="moved backward"):
    publish_download(publisher, artifacts=1, byte_count=100)
  publish_certification(publisher, domains=2, routes=17, excluded=1)
  with pytest.raises(ValueError, match="moved backward"):
    publish_certification(publisher, domains=1, routes=17, excluded=1)
  with pytest.raises(ValueError, match="timestamp did not advance"):
    publish_certification(publisher, domains=2, routes=18, excluded=1)


def test_new_session_resets_identity_sequence_and_monotonic_counters() -> None:
  params = FakeParams()
  identities = iter((SESSION_ID, "34" * 16))
  publisher = OffdeviceProgressPublisher(
    params,
    monotonic_ns=iter((100, 110, 200)).__next__,
    session_id_factory=identities.__next__,
  )
  first = publish_remote(publisher, new_session=True, authority=2, route=38)
  decode_offdevice_progress(publisher.publish(
    phase=OffdeviceProgressPhase.LOCAL_FALLBACK,
    fallback_reason_code=OffdeviceFallbackReason.REMOTE_JOB_FAILED,
  ))
  second = publish_remote(publisher, new_session=True, authority=0, route=0)

  assert first["session_id"] == SESSION_ID
  assert second["session_id"] == "34" * 16
  assert second["sequence"] == 0
  assert second["remote_route_index"] == 0


def test_failed_publish_is_not_recorded_and_clear_touches_only_display_param() -> None:
  params = FakeParams()
  publisher = OffdeviceProgressPublisher(
    params,
    monotonic_ns=iter((100, 90)).__next__,
    session_id_factory=lambda: SESSION_ID,
  )
  first = publish_remote(publisher, new_session=True)
  with pytest.raises(ValueError, match="timestamp"):
    publish_remote(publisher, route=2)

  assert publisher.last_payload == first
  assert len(params.puts) == 1
  assert params.values["BLaTv2LearningEvidence"] == b"unchanged"
  publisher.clear()
  assert params.removes == [OFFDEVICE_PROGRESS_PARAM]
  assert params.values["BLaTv2LearningEvidence"] == b"unchanged"
  assert publisher.last_payload is None


@pytest.mark.parametrize(
  "encoded",
  (
    b"not-json",
    b"[]",
    json.dumps(empty_payload(schema_version=0)).encode(),
    json.dumps(empty_payload(updated_mono_ns=-1)).encode(),
  ),
)
def test_decoder_rejects_malformed_or_stale_shaped_payloads(encoded: bytes) -> None:
  with pytest.raises(ValueError):
    decode_offdevice_progress(encoded)

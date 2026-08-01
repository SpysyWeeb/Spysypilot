from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest  # noqa: TID251

from openpilot.selfdrive.controls.lib.blatv2.learning_backfill import (
  BackfillError,
  BuildDescriptor,
  FullRlogDiscovery,
  PreparedRoute,
  RouteCandidate,
  RouteRejected,
  RouteSegment,
  build_current_historical_descriptor,
  git_blob_sha1,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill_spool import (
  open_prepared_route_spool,
  write_prepared_route_spool,
)
from openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill import (
  BridgeFallbackUnavailableError,
  RemotePreparationSession,
  RemoteRoutePlan,
  _PreparedOutcome,
  _RemoteProgressProjector,
  _certify_preparation_domains as _certify_preparation_domains_impl,
  _exclude_unverified_remote_rejections,
  _upload_missing_routes,
  _prior_generation_extractor_sha256,
  _validated_outcomes,
  build_remote_route_plan,
  prepare_remote_session,
)
from openpilot.selfdrive.controls.lib.blatv2.offdevice_protocol import (
  BridgeCorruptError,
  BridgeRemoteError,
  BridgeUnavailableError,
  canonical_json_bytes,
)
from openpilot.selfdrive.controls.lib.blatv2.offdevice_progress import (
  OffdeviceFallbackReason,
  OffdeviceProgressPhase,
  OffdeviceProgressPublisher,
)
from openpilot.selfdrive.controls.tests.blatv2_artifact_test_helpers import (
  route_evidence_for_frames,
)


DONGLE = "f" * 16
SECRET = bytes(range(32))
WORKER_EXTRACTOR = "e" * 64
WORKER_INSTANCE = "9" * 64
WORKER_IMPLEMENTATION_COMMIT = "7" * 40
WORKER_IMPLEMENTATION_SHA = "8" * 64


def _certify_preparation_domains(**arguments):
  arguments.setdefault(
    "worker_implementation_commit",
    WORKER_IMPLEMENTATION_COMMIT,
  )
  arguments.setdefault(
    "worker_implementation_sha256",
    WORKER_IMPLEMENTATION_SHA,
  )
  return _certify_preparation_domains_impl(**arguments)


def contract() -> dict[str, object]:
  return {
    "descriptor_registry_sha256": "d" * 64,
    "historical_descriptor_registry_sha256": "c" * 64,
    "opendbc_commit": "b" * 40,
    "panda_commit": "c" * 40,
    "runtime_identity_sha256": "a" * 64,
    "source_commit": "a" * 40,
  }


def prepared_provenance() -> dict[str, object]:
  return {
    "canonical_join_schema_version": 2,
    "car_params_sha256": "1" * 64,
    "dongle_id_sha256": "2" * 64,
    "extractor_schema_version": 1,
    "log_schema_blob": "3" * 40,
    "opendbc_commit": "4" * 40,
    "panda_commit": "5" * 40,
    "physical_compatibility_sha256": "6" * 64,
    "route_version": "test",
    "selected_event_stream_sha256": "7" * 64,
    "superproject_commit": "8" * 40,
  }


def route(
  root: Path,
  name: str,
  segment_hashes: tuple[str, ...],
) -> RouteCandidate:
  return RouteCandidate(
    route_name=name,
    route_counter=int(name[:8], 16),
    segments=tuple(
      RouteSegment(
        index=index,
        path=root / f"{name}--{index}" / "rlog.zst",
        sha256=sha,
        size_bytes=100 + index,
      )
      for index, sha in enumerate(segment_hashes)
    ),
  )


def inventory_route(candidate: RouteCandidate, dongle: str = DONGLE) -> dict[str, object]:
  return {
    "archive_name": f"{dongle}_{candidate.route_name}",
    "complete": True,
    "dongle_id": dongle,
    "route_name": candidate.route_name,
    "segments": [
      {
        "index": segment.index,
        "sha256": segment.sha256,
        "size_bytes": segment.size_bytes,
      }
      for segment in candidate.segments
    ],
  }


class FakeEngine:
  def __init__(self, root: Path) -> None:
    self.expected_dongle_id = DONGLE
    self.pending_route_identity = None
    self.pending_route_quiescence_observed = False
    self._runtime = SimpleNamespace(
      artifact_paths=SimpleNamespace(root=root),
      runtime_bundle=SimpleNamespace(calibration_identity_sha256="a" * 64),
    )

  def runtime_factory(self):
    return self._runtime


def patch_empty_ledger(monkeypatch) -> None:
  monkeypatch.setattr(
    "openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill.load_ledger",
    lambda *_args, **_kwargs: {
      "entries": [],
      "runtime_identity_sha256": "a" * 64,
      "schema_version": 1,
      "watermark_route_counter": None,
    },
  )
  monkeypatch.setattr(
    "openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill.verify_known_route_hashes",
    lambda *_args, **_kwargs: None,
  )
  monkeypatch.setattr(
    "openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill.ledger_routes",
    lambda _ledger: {},
  )


def test_route_plan_device_selects_archive_and_uploads_only_missing(
  tmp_path: Path,
  monkeypatch,
) -> None:
  patch_empty_ledger(monkeypatch)
  archived = route(tmp_path, "00000001--1111111111", ("1" * 64,))
  local = route(tmp_path, "00000002--2222222222", ("2" * 64, "3" * 64))
  other_car = route(tmp_path, "00000003--3333333333", ("4" * 64,))
  plan = build_remote_route_plan(
    local_discovery=FullRlogDiscovery((local,), True),
    inventory_payload={
      "routes": [
        inventory_route(archived),
        inventory_route(other_car, "e" * 16),
      ],
    },
    expected_dongle_id=DONGLE,
    placeholder_root=tmp_path,
    engine=FakeEngine(tmp_path),
  )
  assert [item.route_name for item in plan.discovery.candidates] == [
    archived.route_name,
    local.route_name,
  ]
  assert plan.discovery.pending_logger_close is True
  assert plan.upload_candidates == (local,)
  assert plan.replay_candidates == plan.discovery.candidates


def test_route_upload_commits_only_after_complete_ordered_manifest(
  tmp_path: Path,
) -> None:
  candidate = route(
    tmp_path,
    "00000002--2222222222",
    ("2" * 64, "3" * 64),
  )
  calls: list[tuple[str, object]] = []

  class Client:
    def upload_segment(self, **arguments) -> None:
      calls.append(("upload", arguments["segment_index"]))

    def commit_route(self, **arguments) -> None:
      calls.append(("commit", arguments))

  _upload_missing_routes(
    client=Client(),
    routes=(candidate,),
    dongle_id=DONGLE,
  )

  assert [item[0] for item in calls] == ["upload", "upload", "commit"]
  assert calls[-1][1] == {
    "dongle_id": DONGLE,
    "route_name": candidate.route_name,
    "segments": [
      {"index": 0, "sha256": "2" * 64, "size_bytes": 100},
      {"index": 1, "sha256": "3" * 64, "size_bytes": 101},
    ],
  }


def test_route_plan_rejects_same_name_different_bytes(
  tmp_path: Path,
  monkeypatch,
) -> None:
  patch_empty_ledger(monkeypatch)
  local = route(tmp_path, "00000002--2222222222", ("2" * 64,))
  remote = route(tmp_path, local.route_name, ("3" * 64,))
  with pytest.raises(BridgeCorruptError, match="disagree"):
    build_remote_route_plan(
      local_discovery=FullRlogDiscovery((local,), False),
      inventory_payload={"routes": [inventory_route(remote)]},
      expected_dongle_id=DONGLE,
      placeholder_root=tmp_path,
      engine=FakeEngine(tmp_path),
    )


def prepared_outcome(
  authority: int,
  route_name: str,
  *,
  sha: str = "a" * 64,
) -> dict[str, object]:
  return {
    "authority_index": authority,
    "descriptor": {
      "artifact_id": sha,
      "frame_count": 0,
      "provenance": {"source": "test"},
      "route_name": route_name,
      "sha256": sha,
      "size_bytes": 100,
    },
    "disposition": "prepared",
    "route_name": route_name,
  }


def rejected_outcome(
  authority: int,
  route_name: str,
  *,
  reason: str = "event_decode_failed",
  message: str = "bounded route event could not be decoded",
) -> dict[str, object]:
  return {
    "authority_index": authority,
    "disposition": "rejected",
    "message": message,
    "reason": reason,
    "route_name": route_name,
  }


def test_completed_job_requires_two_identical_authorities(tmp_path: Path) -> None:
  candidate = route(tmp_path, "00000002--2222222222", ("2" * 64,))
  status = {
    "outcomes": [
      prepared_outcome(1, candidate.route_name),
      prepared_outcome(2, candidate.route_name),
    ],
  }
  assert len(_validated_outcomes(status=status, routes=(candidate,))) == 2
  status["outcomes"][1] = prepared_outcome(
    2,
    candidate.route_name,
    sha="b" * 64,
  )
  with pytest.raises(BridgeCorruptError, match="artifacts differ"):
    _validated_outcomes(status=status, routes=(candidate,))


def test_remote_only_rejection_is_removed_from_effective_transaction(
  tmp_path: Path,
) -> None:
  accepted = route(tmp_path, "00000002--2222222222", ("2" * 64,))
  excluded = route(tmp_path, "00000003--3333333333", ("3" * 64,))
  late = route(tmp_path, "00000001--1111111111", ("1" * 64,))
  plan = RemoteRoutePlan(
    discovery=FullRlogDiscovery((late, accepted, excluded), True),
    replay_candidates=(accepted, excluded),
    late_candidates=(late,),
    upload_candidates=(),
    locally_available_route_names=frozenset({accepted.route_name}),
  )
  raw = {
    "outcomes": [
      prepared_outcome(1, accepted.route_name),
      prepared_outcome(2, accepted.route_name),
      rejected_outcome(1, excluded.route_name),
      rejected_outcome(2, excluded.route_name),
    ],
  }
  validated = _validated_outcomes(
    status=raw,
    routes=plan.replay_candidates,
  )
  effective, outcomes = _exclude_unverified_remote_rejections(
    plan=plan,
    outcomes=validated,
  )

  assert effective.discovery == FullRlogDiscovery((late, accepted), True)
  assert effective.replay_candidates == (accepted,)
  assert effective.late_candidates == (late,)
  assert effective.upload_candidates == ()
  assert effective.locally_available_route_names == frozenset({accepted.route_name})
  assert set(outcomes) == {
    (1, accepted.route_name),
    (2, accepted.route_name),
  }
  assert len(effective.unverified_exclusions) == 1
  record = effective.unverified_exclusions[0]
  assert record.route_identity_sha256 == excluded.display_identity
  assert record.rejection_reason == "event_decode_failed"
  assert record.rejection_message == "bounded route event could not be decoded"


def test_remote_rejection_partition_never_excludes_local_or_unvalidated(
  tmp_path: Path,
) -> None:
  local = route(tmp_path, "00000002--2222222222", ("2" * 64,))
  plan = certification_plan(local)
  validated = _validated_outcomes(
    status={
      "outcomes": [
        rejected_outcome(1, local.route_name),
        rejected_outcome(2, local.route_name),
      ],
    },
    routes=plan.replay_candidates,
  )
  effective, outcomes = _exclude_unverified_remote_rejections(
    plan=plan,
    outcomes=validated,
  )
  assert effective == plan
  assert outcomes == validated

  nonlocal_plan = certification_plan(local, locally_available=False)
  mismatched = dict(validated)
  mismatched[(2, local.route_name)] = rejected_outcome(
    2,
    local.route_name,
    reason="different_reason",
  )
  with pytest.raises(BridgeCorruptError, match="disagree"):
    _exclude_unverified_remote_rejections(
      plan=nonlocal_plan,
      outcomes=mismatched,
    )
  with pytest.raises(BridgeCorruptError, match="complete"):
    _exclude_unverified_remote_rejections(
      plan=nonlocal_plan,
      outcomes={(1, local.route_name): validated[(1, local.route_name)]},
    )


def test_mixed_remote_session_continues_without_unverified_rejection(
  tmp_path: Path,
  monkeypatch,
) -> None:
  accepted = route(tmp_path, "00000002--2222222222", ("2" * 64,))
  excluded = route(tmp_path, "00000003--3333333333", ("3" * 64,))
  original_plan = RemoteRoutePlan(
    discovery=FullRlogDiscovery((accepted, excluded), False),
    replay_candidates=(accepted, excluded),
    late_candidates=(),
    upload_candidates=(),
    locally_available_route_names=frozenset({accepted.route_name}),
  )
  monkeypatch.setattr(
    "openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill.discover_full_rlog_state",
    lambda *_args, **_kwargs: original_plan.discovery,
  )
  monkeypatch.setattr(
    "openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill.build_remote_route_plan",
    lambda **_kwargs: original_plan,
  )

  class Projector:
    complete_count = 0

    def __init__(self, **kwargs) -> None:
      self.offdevice_progress = kwargs["offdevice_progress"]
      self.route_count = len(kwargs["routes"])

    def start(self) -> None:
      self.offdevice_progress.publish(
        phase=OffdeviceProgressPhase.REMOTE_PROCESSING,
        new_session=True,
        remote_authority_count=2,
        remote_authority_index=0,
        remote_route_count=self.route_count,
        remote_route_index=0,
      )

    def update(self, _progress: object) -> None:
      pass

    def complete(self) -> None:
      self.complete_count += 1

  monkeypatch.setattr(
    "openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill._RemoteProgressProjector",
    Projector,
  )
  scratch = tmp_path / ".blatv2-remote-prepare-filtered"
  downloaded_keys: set[tuple[int, str]] = set()

  def download(*, routes, outcomes, progress, **_kwargs):
    assert routes == (accepted,)
    downloaded_keys.update(outcomes)
    progress(0, 2, 0, 200)
    progress(2, 2, 200, 200)
    scratch.mkdir(mode=0o700)
    provenance = {"source": "mixed-session-test"}
    evidence = route_evidence_for_frames(accepted.route_name, (), provenance)
    resolved = {}
    for authority in (1, 2):
      descriptor = write_prepared_route_spool(
        scratch,
        accepted.route_name,
        (),
        controls_witness_count=0,
        unresolved_witness_count=0,
        gap_count=0,
        provenance=provenance,
        max_frames=1,
        abort_requested=lambda: False,
        filename=f"a{authority}-{accepted.route_name}.spool",
        route_evidence=evidence,
      )
      resolved[(authority, accepted.route_name)] = _PreparedOutcome(
        descriptor=descriptor,
        rejection_reason=None,
        rejection_message=None,
      )
    return scratch, resolved

  def certify(*, plan, outcomes, progress, **_kwargs):
    assert plan.replay_candidates == (accepted,)
    assert plan.discovery.candidates == (accepted,)
    progress(0, 0, 1, 1)
    progress(1, 1, 1, 1)
    return {
      key: _PreparedOutcome(
        descriptor=value.descriptor,
        rejection_reason=value.rejection_reason,
        rejection_message=value.rejection_message,
        certification_identity_sha256="a" * 64,
      )
      for key, value in outcomes.items()
    }

  monkeypatch.setattr(
    "openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill._download_outcomes",
    download,
  )
  monkeypatch.setattr(
    "openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill._certify_preparation_domains",
    certify,
  )

  class Client:
    secret = SECRET

    def health(self) -> dict[str, object]:
      return {
        **contract(),
        "state": "ready",
        "worker_count": 4,
        "worker_extractor_sha256": WORKER_EXTRACTOR,
        "worker_implementation_commit": WORKER_IMPLEMENTATION_COMMIT,
        "worker_implementation_sha256": WORKER_IMPLEMENTATION_SHA,
        "worker_instance_id": WORKER_INSTANCE,
      }

    def route_inventory(self) -> dict[str, object]:
      return {"routes": []}

    def create_job(self, **_kwargs) -> dict[str, object]:
      return {
        "job_id": "1" * 32,
        "route_count": 2,
        "state": "running",
        "worker_implementation_commit": WORKER_IMPLEMENTATION_COMMIT,
        "worker_implementation_sha256": WORKER_IMPLEMENTATION_SHA,
        "worker_instance_id": WORKER_INSTANCE,
      }

    def job_status(self, _job_id: str) -> dict[str, object]:
      return {
        "error": None,
        "outcomes": [
          prepared_outcome(1, accepted.route_name),
          prepared_outcome(2, accepted.route_name),
          rejected_outcome(1, excluded.route_name),
          rejected_outcome(2, excluded.route_name),
        ],
        "progress": {},
        "state": "completed",
        "worker_extractor_sha256": WORKER_EXTRACTOR,
        "worker_implementation_commit": WORKER_IMPLEMENTATION_COMMIT,
        "worker_implementation_sha256": WORKER_IMPLEMENTATION_SHA,
        "worker_instance_id": WORKER_INSTANCE,
      }

  engine = FakeEngine(tmp_path)
  engine.log_root = tmp_path / "logs"
  class ProgressParams:
    def __init__(self) -> None:
      self.values: dict[str, object] = {}

    def put(self, key: str, value: object, *, block: bool) -> None:
      assert block is True
      self.values[key] = value

    def remove(self, key: str) -> None:
      self.values.pop(key, None)

  progress_params = ProgressParams()
  offdevice_progress = OffdeviceProgressPublisher(progress_params)
  session = prepare_remote_session(
    engine=engine,
    client=Client(),
    contract=contract(),
    scratch_parent=tmp_path,
    abort_requested=lambda: False,
    offdevice_progress=offdevice_progress,
  )
  try:
    assert session.plan.discovery.candidates == (accepted,)
    assert session.plan.replay_candidates == (accepted,)
    assert downloaded_keys == {
      (1, accepted.route_name),
      (2, accepted.route_name),
    }
    assert session.outcomes.keys() == downloaded_keys
    assert session.unverified_exclusions == session.plan.unverified_exclusions
    assert len(session.unverified_exclusions) == 1
    assert offdevice_progress.last_payload is not None
    assert offdevice_progress.last_payload["phase"] == "remote_ready"
    assert offdevice_progress.last_payload["certified_route_count"] == 1
    assert (
      offdevice_progress.last_payload[
        "remote_only_rejection_excluded_count"
      ]
      == 1
    )
    assert offdevice_progress.last_payload["total_certification_route_count"] == 2
    prepared = session._source(1, accepted, lambda: False)
    assert tuple(prepared.iter_frames()) == ()
    prepared.cleanup()
    with pytest.raises(BackfillError, match="absent"):
      session._source(1, excluded, lambda: False)
  finally:
    session.close()


def test_all_excluded_late_failure_never_claims_remote_ready(
  tmp_path: Path,
  monkeypatch,
) -> None:
  excluded = route(tmp_path, "00000003--3333333333", ("3" * 64,))
  late = route(tmp_path, "00000001--1111111111", ("1" * 64,))
  plan = RemoteRoutePlan(
    discovery=FullRlogDiscovery((late, excluded), False),
    replay_candidates=(excluded,),
    late_candidates=(late,),
    upload_candidates=(),
    locally_available_route_names=frozenset(),
  )
  monkeypatch.setattr(
    "openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill.discover_full_rlog_state",
    lambda *_args, **_kwargs: plan.discovery,
  )
  monkeypatch.setattr(
    "openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill.build_remote_route_plan",
    lambda **_kwargs: plan,
  )

  class Projector:
    complete_count = 0

    def __init__(self, **kwargs) -> None:
      self.progress = kwargs["offdevice_progress"]

    def start(self) -> None:
      self.progress.publish(
        phase=OffdeviceProgressPhase.REMOTE_PROCESSING,
        new_session=True,
        remote_authority_count=2,
        remote_authority_index=0,
        remote_route_count=1,
        remote_route_index=0,
      )

    def update(self, _progress: object) -> None:
      pass

    def complete(self) -> None:
      self.complete_count += 1

  monkeypatch.setattr(
    "openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill._RemoteProgressProjector",
    Projector,
  )

  def prior_failure(_engine: object) -> str:
    raise BridgeUnavailableError("prior extractor unavailable")

  monkeypatch.setattr(
    "openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill._prior_generation_extractor_sha256",
    prior_failure,
  )

  class Client:
    secret = SECRET

    def health(self) -> dict[str, object]:
      return {
        **contract(),
        "state": "ready",
        "worker_count": 4,
        "worker_extractor_sha256": WORKER_EXTRACTOR,
        "worker_implementation_commit": WORKER_IMPLEMENTATION_COMMIT,
        "worker_implementation_sha256": WORKER_IMPLEMENTATION_SHA,
        "worker_instance_id": WORKER_INSTANCE,
      }

    def route_inventory(self) -> dict[str, object]:
      return {"routes": []}

    def create_job(self, **_kwargs) -> dict[str, object]:
      return {
        "job_id": "1" * 32,
        "route_count": 1,
        "state": "running",
        "worker_implementation_commit": WORKER_IMPLEMENTATION_COMMIT,
        "worker_implementation_sha256": WORKER_IMPLEMENTATION_SHA,
        "worker_instance_id": WORKER_INSTANCE,
      }

    def job_status(self, _job_id: str) -> dict[str, object]:
      return {
        "error": None,
        "outcomes": [
          rejected_outcome(1, excluded.route_name),
          rejected_outcome(2, excluded.route_name),
        ],
        "progress": {},
        "state": "completed",
        "worker_extractor_sha256": WORKER_EXTRACTOR,
        "worker_implementation_commit": WORKER_IMPLEMENTATION_COMMIT,
        "worker_implementation_sha256": WORKER_IMPLEMENTATION_SHA,
        "worker_instance_id": WORKER_INSTANCE,
      }

  class ProgressParams:
    def __init__(self) -> None:
      self.values: dict[str, object] = {}

    def put(self, key: str, value: object, *, block: bool) -> None:
      assert block is True
      self.values[key] = value

    def remove(self, key: str) -> None:
      self.values.pop(key, None)

  progress = OffdeviceProgressPublisher(ProgressParams())
  engine = FakeEngine(tmp_path)
  engine.log_root = tmp_path / "logs"
  with pytest.raises(BridgeUnavailableError, match="prior extractor"):
    prepare_remote_session(
      engine=engine,
      client=Client(),
      contract=contract(),
      scratch_parent=tmp_path,
      abort_requested=lambda: False,
      offdevice_progress=progress,
    )
  assert progress.last_payload is not None
  assert progress.last_payload["phase"] == "arm_certifying"
  assert progress.last_payload["remote_only_rejection_excluded_count"] == 1
  assert Projector.complete_count == 0


def test_session_opens_exact_spool_and_cleans_it(
  tmp_path: Path,
) -> None:
  scratch = tmp_path / ".blatv2-remote-prepare-test"
  scratch.mkdir(mode=0o700)
  candidate = route(tmp_path, "00000002--2222222222", ("2" * 64,))
  provenance = {"source": "test"}
  evidence = route_evidence_for_frames(candidate.route_name, (), provenance)
  descriptor = write_prepared_route_spool(
    scratch,
    candidate.route_name,
    (),
    controls_witness_count=0,
    unresolved_witness_count=0,
    gap_count=0,
    provenance=provenance,
    max_frames=1,
    abort_requested=lambda: False,
    filename="a1-route.spool",
    route_evidence=evidence,
  )
  # Prove the same descriptor is independently valid before handing it to the
  # source callback; the callback/replay owns deletion after this point.
  opened = open_prepared_route_spool(
    scratch,
    descriptor,
    expected_route_name=candidate.route_name,
    max_frames=1,
  )
  assert tuple(opened.iter_frames()) == ()
  session = RemotePreparationSession(
    base_engine=FakeEngine(tmp_path),
    plan=SimpleNamespace(discovery=FullRlogDiscovery((candidate,), False)),
    scratch_directory=scratch,
    outcomes={
      (1, candidate.route_name): _PreparedOutcome(
        descriptor=descriptor,
        rejection_reason=None,
        rejection_message=None,
        certification_identity_sha256="a" * 64,
      ),
    },
    worker_extractor_sha256="a" * 64,
  )
  remote = session._source(1, candidate, lambda: False)
  assert tuple(remote.iter_frames()) == ()
  remote.cleanup()
  assert not (scratch / descriptor.filename).exists()
  session.close()
  assert not scratch.exists()


def test_route_plan_over_protocol_bound_falls_back_before_job(
  tmp_path: Path,
  monkeypatch,
) -> None:
  patch_empty_ledger(monkeypatch)
  candidates = tuple(
    route(
      tmp_path,
      f"{index:08x}--{index:010x}",
      (f"{index:064x}",),
    )
    for index in range(1, 130)
  )
  with pytest.raises(
    BridgeFallbackUnavailableError,
    match="count",
  ) as raised:
    build_remote_route_plan(
      local_discovery=FullRlogDiscovery(candidates, False),
      inventory_payload={"routes": []},
      expected_dongle_id=DONGLE,
      placeholder_root=tmp_path,
      engine=FakeEngine(tmp_path),
    )
  assert raised.value.fallback_reason is OffdeviceFallbackReason.REMOTE_ROUTE_LIMIT


def test_route_plan_identifies_late_only_generation(
  tmp_path: Path,
  monkeypatch,
) -> None:
  candidate = route(tmp_path, "00000002--2222222222", ("2" * 64,))
  monkeypatch.setattr(
    "openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill.load_ledger",
    lambda *_args, **_kwargs: {
      "entries": [],
      "runtime_identity_sha256": "a" * 64,
      "schema_version": 1,
      "watermark_route_counter": 3,
    },
  )
  monkeypatch.setattr(
    "openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill.verify_known_route_hashes",
    lambda *_args, **_kwargs: None,
  )
  monkeypatch.setattr(
    "openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill.ledger_routes",
    lambda _ledger: {},
  )
  plan = build_remote_route_plan(
    local_discovery=FullRlogDiscovery((candidate,), False),
    inventory_payload={"routes": []},
    expected_dongle_id=DONGLE,
    placeholder_root=tmp_path,
    engine=FakeEngine(tmp_path),
  )
  assert plan.replay_candidates == ()
  assert plan.late_candidates == (candidate,)
  assert plan.upload_candidates == (candidate,)


def test_locally_processed_route_is_later_archived_without_replay(
  tmp_path: Path,
  monkeypatch,
) -> None:
  candidate = route(tmp_path, "00000002--2222222222", ("2" * 64,))
  monkeypatch.setattr(
    "openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill.load_ledger",
    lambda *_args, **_kwargs: {
      "entries": [],
      "runtime_identity_sha256": "a" * 64,
      "schema_version": 1,
      "watermark_route_counter": 2,
    },
  )
  monkeypatch.setattr(
    "openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill.verify_known_route_hashes",
    lambda *_args, **_kwargs: None,
  )
  monkeypatch.setattr(
    "openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill.ledger_routes",
    lambda _ledger: {candidate.route_name: object()},
  )

  plan = build_remote_route_plan(
    local_discovery=FullRlogDiscovery((candidate,), False),
    inventory_payload={"routes": []},
    expected_dongle_id=DONGLE,
    placeholder_root=tmp_path,
    engine=FakeEngine(tmp_path),
  )

  assert plan.replay_candidates == ()
  assert plan.late_candidates == ()
  assert plan.upload_candidates == (candidate,)


def test_archive_sync_runs_even_when_no_remote_replay_job(
  tmp_path: Path,
  monkeypatch,
) -> None:
  candidate = route(tmp_path, "00000002--2222222222", ("2" * 64,))
  plan = RemoteRoutePlan(
    discovery=FullRlogDiscovery((candidate,), False),
    replay_candidates=(),
    late_candidates=(),
    upload_candidates=(candidate,),
    locally_available_route_names=frozenset({candidate.route_name}),
  )
  uploads: list[tuple[RouteCandidate, ...]] = []
  monkeypatch.setattr(
    "openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill.discover_full_rlog_state",
    lambda *_args, **_kwargs: plan.discovery,
  )
  monkeypatch.setattr(
    "openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill.build_remote_route_plan",
    lambda **_kwargs: plan,
  )
  monkeypatch.setattr(
    "openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill._upload_missing_routes",
    lambda *, routes, **_kwargs: uploads.append(routes),
  )

  class Client:
    def health(self) -> dict[str, object]:
      return {
        **contract(),
        "state": "ready",
        "worker_count": 4,
        "worker_extractor_sha256": WORKER_EXTRACTOR,
        "worker_implementation_commit": WORKER_IMPLEMENTATION_COMMIT,
        "worker_implementation_sha256": WORKER_IMPLEMENTATION_SHA,
        "worker_instance_id": WORKER_INSTANCE,
      }

    def route_inventory(self) -> dict[str, object]:
      return {"routes": []}

  engine = FakeEngine(tmp_path)
  engine.log_root = tmp_path / "logs"
  session = prepare_remote_session(
    engine=engine,
    client=Client(),
    contract=contract(),
    scratch_parent=tmp_path,
    abort_requested=lambda: False,
  )

  assert uploads == [(candidate,)]
  assert session.scratch_directory is None
  assert session.worker_extractor_sha256 is None


@pytest.mark.parametrize(
  "status_mode",
  ["failed", "not_found", "canceled", "implementation_changed"],
)
def test_job_status_resource_loss_or_identity_change_falls_back(
  tmp_path: Path,
  monkeypatch,
  status_mode: str,
) -> None:
  candidate = route(tmp_path, "00000002--2222222222", ("2" * 64,))
  plan = RemoteRoutePlan(
    discovery=FullRlogDiscovery((candidate,), False),
    replay_candidates=(candidate,),
    late_candidates=(),
    upload_candidates=(),
    locally_available_route_names=frozenset({candidate.route_name}),
  )
  monkeypatch.setattr(
    "openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill.discover_full_rlog_state",
    lambda *_args, **_kwargs: plan.discovery,
  )
  monkeypatch.setattr(
    "openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill.build_remote_route_plan",
    lambda **_kwargs: plan,
  )

  class Projector:
    def __init__(self, **_kwargs) -> None:
      pass

    def start(self) -> None:
      pass

    def update(self, _progress: object) -> None:
      pass

  monkeypatch.setattr(
    "openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill._RemoteProgressProjector",
    Projector,
  )

  class Client:
    canceled = False

    def health(self) -> dict[str, object]:
      return {
        **contract(),
        "state": "ready",
        "worker_count": 4,
        "worker_extractor_sha256": WORKER_EXTRACTOR,
        "worker_implementation_commit": WORKER_IMPLEMENTATION_COMMIT,
        "worker_implementation_sha256": WORKER_IMPLEMENTATION_SHA,
        "worker_instance_id": WORKER_INSTANCE,
      }

    def route_inventory(self) -> dict[str, object]:
      return {"routes": []}

    def create_job(self, **_kwargs) -> dict[str, object]:
      return {
        "job_id": "1" * 32,
        "route_count": 1,
        "state": "running",
        "worker_implementation_commit": WORKER_IMPLEMENTATION_COMMIT,
        "worker_implementation_sha256": WORKER_IMPLEMENTATION_SHA,
        "worker_instance_id": WORKER_INSTANCE,
      }

    def job_status(self, _job_id: str) -> dict[str, object]:
      if status_mode == "not_found":
        raise BridgeRemoteError("job_not_found", "worker restarted")
      if status_mode == "canceled":
        return {
          "error": None,
          "progress": {},
          "state": "canceled",
          "worker_implementation_commit": WORKER_IMPLEMENTATION_COMMIT,
          "worker_implementation_sha256": WORKER_IMPLEMENTATION_SHA,
          "worker_instance_id": WORKER_INSTANCE,
        }
      return {
        "error": {"code": "job_failed", "message": "worker resource lost"},
        "progress": {},
        "state": "failed",
        "worker_implementation_commit": WORKER_IMPLEMENTATION_COMMIT,
        "worker_implementation_sha256": (
          "6" * 64
          if status_mode == "implementation_changed"
          else WORKER_IMPLEMENTATION_SHA
        ),
        "worker_instance_id": WORKER_INSTANCE,
      }

    def cancel_job(self, _job_id: str) -> dict[str, object]:
      self.canceled = True
      return {"job_id": _job_id, "state": "canceled"}

  engine = FakeEngine(tmp_path)
  engine.log_root = tmp_path / "logs"
  client = Client()
  with pytest.raises(BridgeUnavailableError) as raised:
    prepare_remote_session(
      engine=engine,
      client=client,
      contract=contract(),
      scratch_parent=tmp_path,
      abort_requested=lambda: False,
    )
  expected_reason = {
    "failed": OffdeviceFallbackReason.REMOTE_JOB_FAILED,
    "not_found": OffdeviceFallbackReason.NETWORK_INTERRUPTED,
    "canceled": OffdeviceFallbackReason.REMOTE_JOB_CANCELED,
  }.get(status_mode)
  if expected_reason is None:
    assert not isinstance(raised.value, BridgeFallbackUnavailableError)
  else:
    assert isinstance(raised.value, BridgeFallbackUnavailableError)
    assert raised.value.fallback_reason is expected_reason
  assert client.canceled is True


def test_route_plan_preserves_pending_route_quiescence_guard(
  tmp_path: Path,
  monkeypatch,
) -> None:
  patch_empty_ledger(monkeypatch)
  candidate = route(tmp_path, "00000002--2222222222", ("2" * 64,))
  engine = FakeEngine(tmp_path)
  engine.pending_route_identity = candidate.display_identity
  plan = build_remote_route_plan(
    local_discovery=FullRlogDiscovery((candidate,), False),
    inventory_payload={"routes": []},
    expected_dongle_id=DONGLE,
    placeholder_root=tmp_path,
    engine=engine,
  )
  assert plan.discovery.candidates == (candidate,)
  assert plan.replay_candidates == ()
  assert plan.late_candidates == ()
  assert plan.upload_candidates == ()


def test_remote_progress_restamps_pass_major_without_evidence_clock(
  tmp_path: Path,
) -> None:
  candidate = route(
    tmp_path,
    "00000002--2222222222",
    ("2" * 64, "3" * 64),
  )

  class OperationStatus:
    def __init__(self) -> None:
      self.last_payload = None
      self.new_operation_count = 0

    def publish(self, **kwargs) -> None:
      if kwargs.get("new_operation"):
        self.new_operation_count += 1
      self.last_payload = {
        "operation_id": self.new_operation_count,
        "sequence": 1,
      }

  class Progress:
    def __init__(self) -> None:
      self.publications = []
      self.clear_count = 0

    def clear(self) -> None:
      self.clear_count += 1

    def publish(self, **kwargs) -> None:
      self.publications.append(kwargs)

  operation = OperationStatus()
  progress = Progress()
  offdevice = Progress()
  runtime = SimpleNamespace(
    runtime_bundle=SimpleNamespace(
      calibration_identity_sha256="a" * 64,
      vehicle_identity="CAR",
    ),
  )
  engine = SimpleNamespace(
    runtime_factory=lambda: runtime,
    operation_status=operation,
    backfill_progress=progress,
  )
  projector = _RemoteProgressProjector(
    engine=engine,
    routes=(candidate,),
    offdevice_progress=offdevice,
  )
  projector.start()

  def update(authority: int, segment: int) -> None:
    projector.update({
      "authority_index": authority,
      "route_index": 1,
      "segment_index": segment,
      "route_name": candidate.route_name,
      "route_count": 1,
      "segment_count": 2,
    })

  update(1, 1)
  update(1, 2)
  update(2, 1)
  update(2, 2)
  projector.complete()
  assert operation.new_operation_count == 2
  assert progress.clear_count == 2
  reading = [
    item
    for item in progress.publications
    if item["phase"].value == "reading_segment"
  ]
  assert [item["pass_index"] for item in reading] == [1, 1, 2, 2]
  assert [
    item["completed_replay_segment_count"] for item in reading
  ] == [0, 1, 2, 3]
  assert progress.publications[-1]["completed_replay_segment_count"] == 4
  assert offdevice.publications[0] == {
    "phase": OffdeviceProgressPhase.REMOTE_PROCESSING,
    "new_session": True,
    "remote_authority_count": 2,
    "remote_authority_index": 0,
    "remote_route_count": 1,
    "remote_route_index": 0,
  }
  assert [
    (item["remote_authority_index"], item["remote_route_index"])
    for item in offdevice.publications[1:]
  ] == [(1, 1), (1, 1), (2, 1), (2, 1)]

  with pytest.raises(BridgeCorruptError, match="backward"):
    update(1, 2)


def test_late_only_reads_prior_authenticated_extractor(tmp_path: Path) -> None:
  provenance_path = tmp_path / "provenance.json"
  payload = {
    "canonical_join_schema_version": 2,
    "descriptor_registry_sha256": "b" * 64,
    "extractor_schema_version": 1,
    "extractor_sha256": WORKER_EXTRACTOR,
    "ledger_sha256": "c" * 64,
    "previous_generation_sha256": None,
    "runtime_identity_sha256": "a" * 64,
    "schema_version": 1,
    "source": "complete_full_rlog_only",
  }
  provenance_path.write_bytes(canonical_json_bytes(payload))

  class Paths:
    backfill_provenance = provenance_path

    def resolved(self):
      return self

  runtime = SimpleNamespace(
    artifact_paths=Paths(),
    runtime_bundle=SimpleNamespace(calibration_identity_sha256="a" * 64),
  )
  engine = SimpleNamespace(runtime_factory=lambda: runtime)
  assert _prior_generation_extractor_sha256(engine) == WORKER_EXTRACTOR
  provenance_path.write_text(json.dumps(payload, indent=2))
  with pytest.raises(BridgeUnavailableError, match="provenance"):
    _prior_generation_extractor_sha256(engine)


class CertificationEngine:
  def __init__(
    self,
    root: Path,
    prepared: PreparedRoute | RouteRejected,
  ) -> None:
    self.extractor_path = root / "blatv2_rlog_extract"
    self.extractor_path.write_bytes(b"one deterministic ARM extractor")
    self._runtime = SimpleNamespace(
      artifact_paths=SimpleNamespace(root=root),
    )
    self.prepared = prepared
    self.prepare_count = 0

  def runtime_factory(self):
    return self._runtime

  def _prepare(self, *_args, **_kwargs):
    self.prepare_count += 1
    if isinstance(self.prepared, RouteRejected):
      raise self.prepared
    return self.prepared


def certification_plan(
  candidate: RouteCandidate,
  *,
  locally_available: bool = True,
) -> RemoteRoutePlan:
  return RemoteRoutePlan(
    discovery=FullRlogDiscovery((candidate,), False),
    replay_candidates=(candidate,),
    late_candidates=(),
    upload_candidates=(),
    locally_available_route_names=(
      frozenset({candidate.route_name})
      if locally_available
      else frozenset()
    ),
  )


def accepted_spool_outcomes(
  scratch: Path,
  candidate: RouteCandidate,
  provenance: dict[str, object],
  *,
  runtime_identity: str | None = None,
) -> dict[tuple[int, str], _PreparedOutcome]:
  result: dict[tuple[int, str], _PreparedOutcome] = {}
  for authority in (1, 2):
    evidence = route_evidence_for_frames(
      candidate.route_name,
      (),
      provenance,
      runtime_identity=runtime_identity,
    )
    descriptor = write_prepared_route_spool(
      scratch,
      candidate.route_name,
      (),
      controls_witness_count=0,
      unresolved_witness_count=0,
      gap_count=0,
      provenance=provenance,
      max_frames=1,
      abort_requested=lambda: False,
      filename=f"a{authority}-{candidate.route_name}.spool",
      route_evidence=evidence,
    )
    result[(authority, candidate.route_name)] = _PreparedOutcome(
      descriptor=descriptor,
      rejection_reason=None,
      rejection_message=None,
    )
  return result


def empty_prepared_route(
  candidate: RouteCandidate,
  provenance: dict[str, object],
  *,
  runtime_identity: str | None = None,
) -> PreparedRoute:
  return PreparedRoute(
    (), 0, 0, 0, provenance,
    route_evidence=route_evidence_for_frames(
      candidate.route_name,
      (),
      provenance,
      runtime_identity=runtime_identity,
    ),
  )


def test_accepted_domain_requires_one_byte_exact_arm_certification(
  tmp_path: Path,
) -> None:
  scratch = tmp_path / ".blatv2-remote-prepare-cert"
  scratch.mkdir(mode=0o700)
  candidate = route(tmp_path, "00000002--2222222222", ("2" * 64,))
  provenance = prepared_provenance()
  outcomes = accepted_spool_outcomes(scratch, candidate, provenance)
  engine = CertificationEngine(
    tmp_path,
    empty_prepared_route(candidate, provenance),
  )
  certified = _certify_preparation_domains(
    engine=engine,
    plan=certification_plan(candidate),
    scratch_directory=scratch,
    outcomes=outcomes,
    contract=contract(),
    worker_extractor_sha256=WORKER_EXTRACTOR,
    worker_instance_id=WORKER_INSTANCE,
    secret=SECRET,
    abort_requested=lambda: False,
  )
  assert engine.prepare_count == 1
  assert all(
    outcome.certification_identity_sha256 is not None
    for outcome in certified.values()
  )
  certificates = list(
    (tmp_path / ".blatv2-offdevice-certifications").glob("*.json"),
  )
  assert len(certificates) == 1

  # The same compatibility domain reuses its authenticated smoke vector.
  _certify_preparation_domains(
    engine=engine,
    plan=certification_plan(candidate),
    scratch_directory=scratch,
    outcomes=outcomes,
    contract=contract(),
    worker_extractor_sha256=WORKER_EXTRACTOR,
    worker_instance_id=WORKER_INSTANCE,
    secret=SECRET,
    abort_requested=lambda: False,
  )
  assert engine.prepare_count == 1

  # A committed worker-package change under the same process identity is also
  # a new cross-architecture compatibility boundary.
  _certify_preparation_domains(
    engine=engine,
    plan=certification_plan(candidate),
    scratch_directory=scratch,
    outcomes=outcomes,
    contract=contract(),
    worker_extractor_sha256=WORKER_EXTRACTOR,
    worker_implementation_sha256="6" * 64,
    worker_instance_id=WORKER_INSTANCE,
    secret=SECRET,
    abort_requested=lambda: False,
  )
  assert engine.prepare_count == 2

  # A service restart changes only the authenticated transport session. The
  # same source, implementation, extractor, and semantic domain reuse the
  # durable ARM numerical certificate.
  _certify_preparation_domains(
    engine=engine,
    plan=certification_plan(candidate),
    scratch_directory=scratch,
    outcomes=outcomes,
    contract=contract(),
    worker_extractor_sha256=WORKER_EXTRACTOR,
    worker_instance_id="8" * 64,
    secret=SECRET,
    abort_requested=lambda: False,
  )
  assert engine.prepare_count == 2


def test_full_car_params_hash_does_not_fragment_physical_domain(
  tmp_path: Path,
) -> None:
  scratch = tmp_path / ".blatv2-remote-prepare-cert"
  scratch.mkdir(mode=0o700)
  archived = route(tmp_path, "00000002--2222222222", ("2" * 64,))
  local = route(tmp_path, "00000003--3333333333", ("3" * 64,))
  archived_provenance = prepared_provenance()
  local_provenance = dict(archived_provenance)
  local_provenance["car_params_sha256"] = "f" * 64
  local_provenance["selected_event_stream_sha256"] = "e" * 64
  outcomes = {
    **accepted_spool_outcomes(
      scratch,
      archived,
      archived_provenance,
    ),
    **accepted_spool_outcomes(scratch, local, local_provenance),
  }
  plan = RemoteRoutePlan(
    discovery=FullRlogDiscovery((archived, local), False),
    replay_candidates=(archived, local),
    late_candidates=(),
    upload_candidates=(),
    locally_available_route_names=frozenset({local.route_name}),
  )
  engine = CertificationEngine(
    tmp_path,
    empty_prepared_route(local, local_provenance),
  )

  certified = _certify_preparation_domains(
    engine=engine,
    plan=plan,
    scratch_directory=scratch,
    outcomes=outcomes,
    contract=contract(),
    worker_extractor_sha256=WORKER_EXTRACTOR,
    worker_instance_id=WORKER_INSTANCE,
    secret=SECRET,
    abort_requested=lambda: False,
  )

  assert engine.prepare_count == 1
  assert len(list(
    (tmp_path / ".blatv2-offdevice-certifications").glob("*.json"),
  )) == 1
  identities = {
    outcome.certification_identity_sha256
    for outcome in certified.values()
  }
  assert None not in identities
  assert len(identities) == 1


def test_runtime_vehicle_semantics_remain_a_certification_boundary(
  tmp_path: Path,
) -> None:
  scratch = tmp_path / ".blatv2-remote-prepare-cert"
  scratch.mkdir(mode=0o700)
  archived = route(tmp_path, "00000002--2222222222", ("2" * 64,))
  local = route(tmp_path, "00000003--3333333333", ("3" * 64,))
  archived_provenance = prepared_provenance()
  local_provenance = dict(archived_provenance)
  local_provenance["car_params_sha256"] = "f" * 64
  local_provenance["selected_event_stream_sha256"] = "e" * 64
  archived_runtime = "a" * 64
  local_runtime = "b" * 64
  outcomes = {
    **accepted_spool_outcomes(
      scratch,
      archived,
      archived_provenance,
      runtime_identity=archived_runtime,
    ),
    **accepted_spool_outcomes(
      scratch,
      local,
      local_provenance,
      runtime_identity=local_runtime,
    ),
  }
  plan = RemoteRoutePlan(
    discovery=FullRlogDiscovery((archived, local), False),
    replay_candidates=(archived, local),
    late_candidates=(),
    upload_candidates=(),
    locally_available_route_names=frozenset({local.route_name}),
  )

  with pytest.raises(BridgeUnavailableError, match="no local"):
    _certify_preparation_domains(
      engine=CertificationEngine(
        tmp_path,
        empty_prepared_route(
          local,
          local_provenance,
          runtime_identity=local_runtime,
        ),
      ),
      plan=plan,
      scratch_directory=scratch,
      outcomes=outcomes,
      contract=contract(),
      worker_extractor_sha256=WORKER_EXTRACTOR,
      worker_instance_id=WORKER_INSTANCE,
      secret=SECRET,
      abort_requested=lambda: False,
    )


def test_corrupt_or_nonlocal_accepted_domain_never_reaches_learner(
  tmp_path: Path,
) -> None:
  scratch = tmp_path / ".blatv2-remote-prepare-cert"
  scratch.mkdir(mode=0o700)
  candidate = route(tmp_path, "00000002--2222222222", ("2" * 64,))
  provenance = prepared_provenance()
  outcomes = accepted_spool_outcomes(scratch, candidate, provenance)
  engine = CertificationEngine(
    tmp_path,
    empty_prepared_route(candidate, provenance),
  )
  _certify_preparation_domains(
    engine=engine,
    plan=certification_plan(candidate),
    scratch_directory=scratch,
    outcomes=outcomes,
    contract=contract(),
    worker_extractor_sha256=WORKER_EXTRACTOR,
    worker_instance_id=WORKER_INSTANCE,
    secret=SECRET,
    abort_requested=lambda: False,
  )
  certificate = next(
    (tmp_path / ".blatv2-offdevice-certifications").glob("*.json"),
  )
  certificate.write_bytes(b"{}")
  with pytest.raises(BridgeUnavailableError, match="certification"):
    _certify_preparation_domains(
      engine=engine,
      plan=certification_plan(candidate),
      scratch_directory=scratch,
      outcomes=outcomes,
      contract=contract(),
      worker_extractor_sha256=WORKER_EXTRACTOR,
      worker_instance_id=WORKER_INSTANCE,
      secret=SECRET,
      abort_requested=lambda: False,
    )


def test_accepted_domain_arm_byte_mismatch_falls_back(tmp_path: Path) -> None:
  scratch = tmp_path / ".blatv2-remote-prepare-cert"
  scratch.mkdir(mode=0o700)
  candidate = route(tmp_path, "00000002--2222222222", ("2" * 64,))
  remote_provenance = prepared_provenance()
  local_provenance = dict(remote_provenance)
  local_provenance["selected_event_stream_sha256"] = "f" * 64
  outcomes = accepted_spool_outcomes(
    scratch,
    candidate,
    remote_provenance,
  )
  with pytest.raises(BridgeUnavailableError, match="byte-exact"):
    _certify_preparation_domains(
      engine=CertificationEngine(
        tmp_path,
        empty_prepared_route(candidate, local_provenance),
      ),
      plan=certification_plan(candidate),
      scratch_directory=scratch,
      outcomes=outcomes,
      contract=contract(),
      worker_extractor_sha256=WORKER_EXTRACTOR,
      worker_instance_id=WORKER_INSTANCE,
      secret=SECRET,
      abort_requested=lambda: False,
    )



def test_nonlocal_accepted_domain_falls_back(tmp_path: Path) -> None:
  fresh_root = tmp_path / "fresh"
  fresh_root.mkdir()
  fresh_scratch = fresh_root / ".blatv2-remote-prepare-cert"
  fresh_scratch.mkdir(mode=0o700)
  candidate = route(
    fresh_root,
    "00000002--2222222222",
    ("2" * 64,),
  )
  provenance = prepared_provenance()
  fresh_outcomes = accepted_spool_outcomes(
    fresh_scratch,
    candidate,
    provenance,
  )
  with pytest.raises(BridgeUnavailableError, match="no local"):
    _certify_preparation_domains(
      engine=CertificationEngine(
        fresh_root,
        empty_prepared_route(candidate, provenance),
      ),
      plan=certification_plan(candidate, locally_available=False),
      scratch_directory=fresh_scratch,
      outcomes=fresh_outcomes,
      contract=contract(),
      worker_extractor_sha256=WORKER_EXTRACTOR,
      worker_instance_id=WORKER_INSTANCE,
      secret=SECRET,
      abort_requested=lambda: False,
    )


def test_rejected_route_requires_identical_local_arm_rejection(
  tmp_path: Path,
) -> None:
  scratch = tmp_path / ".blatv2-remote-prepare-reject"
  scratch.mkdir(mode=0o700)
  candidate = route(tmp_path, "00000002--2222222222", ("2" * 64,))
  outcomes = {
    (authority, candidate.route_name): _PreparedOutcome(
      descriptor=None,
      rejection_reason="event_decode_failed",
      rejection_message="bounded route event could not be decoded",
    )
    for authority in (1, 2)
  }
  engine = CertificationEngine(
    tmp_path,
    RouteRejected(
      "event_decode_failed",
      "bounded route event could not be decoded",
    ),
  )
  progress: list[tuple[int, int, int, int]] = []
  certified = _certify_preparation_domains(
    engine=engine,
    plan=certification_plan(candidate),
    scratch_directory=scratch,
    outcomes=outcomes,
    contract=contract(),
    worker_extractor_sha256=WORKER_EXTRACTOR,
    worker_instance_id=WORKER_INSTANCE,
    secret=SECRET,
    abort_requested=lambda: False,
    progress=lambda *values: progress.append(values),
  )
  assert engine.prepare_count == 1
  assert progress == [(0, 0, 0, 1), (0, 1, 0, 1)]
  assert all(
    outcome.certification_identity_sha256 is not None
    for outcome in certified.values()
  )

  other_root = tmp_path / "other"
  other_root.mkdir()
  with pytest.raises(BridgeUnavailableError, match="no local"):
    _certify_preparation_domains(
      engine=CertificationEngine(
        other_root,
        RouteRejected("event_decode_failed", "same"),
      ),
      plan=certification_plan(candidate, locally_available=False),
      scratch_directory=scratch,
      outcomes=outcomes,
      contract=contract(),
      worker_extractor_sha256=WORKER_EXTRACTOR,
      worker_instance_id=WORKER_INSTANCE,
      secret=SECRET,
      abort_requested=lambda: False,
    )


@pytest.mark.parametrize(
  "arm_result_kind",
  [
    "accepted",
    "different_rejection",
  ],
)
def test_rejected_route_arm_disagreement_falls_back(
  tmp_path: Path,
  arm_result_kind: str,
) -> None:
  scratch = tmp_path / ".blatv2-remote-prepare-reject"
  scratch.mkdir(mode=0o700)
  candidate = route(tmp_path, "00000002--2222222222", ("2" * 64,))
  arm_result: PreparedRoute | RouteRejected = (
    empty_prepared_route(candidate, prepared_provenance())
    if arm_result_kind == "accepted"
    else RouteRejected("different_reason", "different message")
  )
  outcomes = {
    (authority, candidate.route_name): _PreparedOutcome(
      descriptor=None,
      rejection_reason="event_decode_failed",
      rejection_message="bounded route event could not be decoded",
    )
    for authority in (1, 2)
  }
  with pytest.raises(BridgeUnavailableError, match="ARM|PC"):
    _certify_preparation_domains(
      engine=CertificationEngine(tmp_path, arm_result),
      plan=certification_plan(candidate),
      scratch_directory=scratch,
      outcomes=outcomes,
      contract=contract(),
      worker_extractor_sha256=WORKER_EXTRACTOR,
      worker_instance_id=WORKER_INSTANCE,
      secret=SECRET,
      abort_requested=lambda: False,
    )


def test_shared_current_descriptor_matches_prior_construction(tmp_path: Path) -> None:
  schema = tmp_path / "log.capnp"
  schema.write_text("struct Test { value @0 :UInt8; }\n")
  limits = SimpleNamespace(
    steer_max=409,
    delta_up=4,
    delta_down=7,
    steer_step=1,
    driver_allowance=50,
    driver_multiplier=2,
    driver_factor=1,
    production_envelope_verified=True,
  )
  bundle = SimpleNamespace(
    torque_limits=limits,
    calibration_seed_profile=SimpleNamespace(nodes=(
      SimpleNamespace(parameters=SimpleNamespace(
        rack_rate_resolution_deg_s=4.0,
      )),
      SimpleNamespace(parameters=SimpleNamespace(
        rack_rate_resolution_deg_s=4.0,
      )),
    )),
  )
  car_params = SimpleNamespace(carFingerprint="HYUNDAI PALISADE 2020")
  actual = build_current_historical_descriptor(
    source_commit="a" * 40,
    opendbc_commit="b" * 40,
    panda_commit="c" * 40,
    log_schema_path=schema,
    current_car_params=car_params,
    current_runtime_bundle=bundle,
  )
  expected = BuildDescriptor(
    superproject_commit="a" * 40,
    opendbc_commit="b" * 40,
    panda_commit="c" * 40,
    log_schema_blob=git_blob_sha1(schema),
    supported_vehicle_identity="HYUNDAI PALISADE 2020",
    steer_max=409,
    steer_delta_up=4,
    steer_delta_down=7,
    steer_step=1,
    driver_allowance=50,
    driver_multiplier=2,
    driver_factor=1,
    production_envelope_verified=True,
    rack_rate_resolution_deg_s=4.0,
  )
  assert actual == expected

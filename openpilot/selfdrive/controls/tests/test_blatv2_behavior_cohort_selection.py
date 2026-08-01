from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

from openpilot.selfdrive.controls.lib.blatv2 import learning_backfill
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill import (
  ReplayResult,
  RouteCandidate,
  RouteSegment,
  extend_ledger,
  select_homogeneous_behavior_cohort,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_runtime import (
  MeasuredLearningFrame,
)
from openpilot.selfdrive.controls.lib.blatv2.route_evidence import (
  RouteEvidenceArtifact,
  RouteEvidenceStore,
)
from openpilot.selfdrive.controls.tests.blatv2_artifact_test_helpers import (
  route_evidence_for_frames,
)


RUNTIME = hashlib.sha256(b"cohort-runtime").hexdigest()


def _frame(mono_ns: int) -> MeasuredLearningFrame:
  return MeasuredLearningFrame(
    sample_mono_ns=mono_ns,
    response_mono_ns=mono_ns - 1,
    applied_report_mono_ns=mono_ns - 2,
    applied_effective_mono_ns=mono_ns - 3,
    speed_mps=5.0,
    steering_angle_deg=0.0,
    steering_rate_deg_s=0.0,
    steering_torque=0.0,
    applied_torque=0.0,
    steering_pressed=False,
    standstill=False,
    steer_fault_temporary=False,
    steer_fault_permanent=False,
    can_valid=True,
    can_timeout=False,
    lateral_active=False,
    live_parameters_valid=True,
    angle_offset_valid=True,
    steer_ratio_valid=True,
    stiffness_factor_valid=True,
    angle_offset_deg=0.0,
    steer_ratio=15.0,
    stiffness_factor=1.0,
    roll_rad=0.0,
    inputs_valid=True,
  )


def _artifact(
  counter: int,
  *,
  source_tag: str = "current",
  eligible: bool = True,
) -> RouteEvidenceArtifact:
  route_name = f"{counter:08x}--{counter:010x}"
  provenance = {"fixture": "cohort-selection"}
  base = route_evidence_for_frames(
    route_name,
    (_frame(counter * 1_000_000_000),),
    provenance,
  )
  source = replace(
    base.source_identity,
    controller_source_kind=("stock_canonical" if eligible else "ineligible"),
    controller_artifact_sha256=hashlib.sha256(source_tag.encode()).hexdigest(),
    behavior_eligible=eligible,
    behavior_ineligible_reason=("eligible" if eligible else "exact_model_link_missing"),
    runtime_identity=RUNTIME,
    model_link_failure_count=(0 if eligible else 1),
  )
  return RouteEvidenceArtifact(
    source,
    bytes(base.car_params_bytes),
    bytes(base.physical_bytes),
    base.model_publications,
    base.control_witnesses,
    base.live_torque_parameters,
    base.live_delays,
    base.lateral_maneuver_plans,
    base.event_locators,
  )


def _route(artifact: RouteEvidenceArtifact) -> RouteCandidate:
  source = artifact.source_identity
  return RouteCandidate(
    route_name=source.route_id,
    route_counter=int(source.route_id[:8], 16),
    segments=tuple(
      RouteSegment(index, Path("/not-read"), sha, size)
      for index, (sha, size) in enumerate(zip(
        source.route_segment_sha256,
        source.route_segment_size_bytes,
        strict=True,
      ))
    ),
  )


def _provenance(route_name: str) -> dict[str, object]:
  return {
    "canonical_join_schema_version": learning_backfill.CANONICAL_JOIN_SCHEMA_VERSION,
    "car_params_sha256": hashlib.sha256(b"cp").hexdigest(),
    "dongle_id_sha256": hashlib.sha256(b"dongle").hexdigest(),
    "extractor_schema_version": learning_backfill.NATIVE_EXTRACTOR_SCHEMA_VERSION,
    "log_schema_blob": "4" * 40,
    "opendbc_commit": "2" * 40,
    "panda_commit": "3" * 40,
    "physical_compatibility_sha256": hashlib.sha256(b"physical").hexdigest(),
    "route_version": "test-version",
    "selected_event_stream_sha256": hashlib.sha256(route_name.encode()).hexdigest(),
    "superproject_commit": "1" * 40,
  }


def _ingested(artifact: RouteEvidenceArtifact) -> ReplayResult:
  source = artifact.source_identity
  return ReplayResult(
    route=_route(artifact),
    disposition="ingested",
    diagnostic="ingested",
    provenance=_provenance(source.route_id),
    accepted_sample_count=0,
    rejected_sample_count=source.controls_witness_count,
    controls_witness_count=source.controls_witness_count,
    unresolved_witness_count=source.unresolved_witness_count,
    route_evidence_sha256=artifact.sha256,
    route_evidence_model_publication_count=len(artifact.model_publications),
    route_evidence_control_witness_count=len(artifact.control_witnesses),
    route_evidence_event_locator_count=len(artifact.event_locators),
  )


def _non_ingested(counter: int, disposition: str) -> ReplayResult:
  artifact = _artifact(counter)
  diagnostic = disposition
  return ReplayResult(
    route=_route(artifact),
    disposition=disposition,
    diagnostic=diagnostic,
    provenance=None,
    accepted_sample_count=0,
    rejected_sample_count=0,
    controls_witness_count=0,
    unresolved_witness_count=0,
  )


def _empty_ledger() -> dict[str, object]:
  return {
    "entries": [],
    "runtime_identity_sha256": RUNTIME,
    "schema_version": learning_backfill.BACKFILL_LEDGER_SCHEMA_VERSION,
    "watermark_route_counter": None,
  }


def _publish(store: RouteEvidenceStore, *artifacts: RouteEvidenceArtifact) -> None:
  for artifact in artifacts:
    store.publish(artifact, RouteEvidenceArtifact.from_bytes(artifact.canonical_bytes))


def test_newest_ineligible_route_blocks_instead_of_scanning_past_it(tmp_path: Path) -> None:
  older = _artifact(0x10)
  newest = _artifact(0x20, eligible=False)
  store = RouteEvidenceStore(tmp_path / "store")
  _publish(store, older, newest)
  ledger = extend_ledger(
    _empty_ledger(), late_routes=(),
    replay_results=(_ingested(older), _ingested(newest)),
  )

  selection = select_homogeneous_behavior_cohort(ledger=ledger, store=store)

  assert selection.status == "blocked"
  assert selection.blocking_route_name == newest.source_identity.route_id
  assert selection.reason == "route_evidence_ineligible:exact_model_link_missing"
  assert selection.artifacts == ()


def test_rejected_route_inside_current_source_population_blocks(tmp_path: Path) -> None:
  older = _artifact(0x10)
  newest = _artifact(0x30)
  store = RouteEvidenceStore(tmp_path / "store")
  _publish(store, older, newest)
  ledger = extend_ledger(
    _empty_ledger(), late_routes=(),
    replay_results=(
      _ingested(older),
      _non_ingested(0x20, "rejected"),
      _ingested(newest),
    ),
  )

  selection = select_homogeneous_behavior_cohort(ledger=ledger, store=store)

  assert selection.status == "blocked"
  assert selection.reason == "interleaved_route_rejected"
  assert selection.blocking_route_name == "00000020--0000000020"


def test_different_eligible_source_is_a_boundary(tmp_path: Path) -> None:
  old_rejected = _non_ingested(0x05, "rejected")
  old_source = _artifact(0x10, source_tag="old")
  current_one = _artifact(0x20)
  current_two = _artifact(0x30)
  store = RouteEvidenceStore(tmp_path / "store")
  _publish(store, old_source, current_one, current_two)
  ledger = extend_ledger(
    _empty_ledger(), late_routes=(),
    replay_results=(
      old_rejected,
      _ingested(old_source),
      _ingested(current_one),
      _ingested(current_two),
    ),
  )

  selection = select_homogeneous_behavior_cohort(ledger=ledger, store=store)

  assert selection.ready
  assert tuple(
    artifact.source_identity.route_id for artifact in selection.artifacts
  ) == (
    current_one.source_identity.route_id,
    current_two.source_identity.route_id,
  )


def test_late_older_skips_are_ignored(tmp_path: Path) -> None:
  current = _artifact(0x30)
  store = RouteEvidenceStore(tmp_path / "store")
  _publish(store, current)
  initial = extend_ledger(
    _empty_ledger(), late_routes=(), replay_results=(_ingested(current),),
  )
  late = _route(_artifact(0x10))
  ledger = extend_ledger(initial, late_routes=(late,), replay_results=())

  selection = select_homogeneous_behavior_cohort(ledger=ledger, store=store)

  assert selection.ready
  assert selection.artifacts == (current,)

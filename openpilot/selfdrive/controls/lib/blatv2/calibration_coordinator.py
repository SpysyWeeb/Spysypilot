"""Lifecycle and persistence boundary for observable BLaTv2 calibration.

This module is deliberately parallel to the retired physical-profile learning
coordinator.  Onroad code may only accumulate measured response in memory.
Finalization and all filesystem writes are offroad-only, and the manifest is
written last so it acts as the commit record for the independently atomic
evidence and optional candidate files.

The coordinator can create an *unapproved* candidate only after every speed
node and every populated interpolation stratum qualifies through schema-17 route-grouped
leave-one-route-out evidence drawn exclusively from the caller's global TRAIN
partition. An evaluated all-seed outcome is
successful and emits an immutable selected-profile proof for downstream
behavior replay, but it intentionally emits no redundant *new calibration*
candidate. It has no API for approval, activation, controller selection, or
mutation of the profile used by a live controller.

Schema 17 permits one owner-authorized terminal 30 m/s authority route only
when both the node and terminal interpolation retain the exact seed.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, StrEnum
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from openpilot.selfdrive.controls.lib.blatv2.calibration_learner import (
  CALIBRATION_EVIDENCE_SCHEMA_VERSION,
  CalibrationLearningResult,
  CalibrationProfileLearner,
  CalibrationSampleAccounting,
  CalibrationSampleDisposition,
  CalibrationRouteCommitment,
  calibration_evidence_sha256,
  minimum_calibration_support_s,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_profile import (
  VehicleCalibrationProfile,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_source import (
  CalibrationIngestionCoordinate,
)
from openpilot.selfdrive.controls.lib.blatv2.learner import LearningSample


CALIBRATION_LEARNING_COORDINATOR_ARTIFACT_SCHEMA_VERSION = 17
# Short alias retained for callers that treat this as the only calibration
# coordinator. Both names identify the same wire artifact, never two schemas.
CALIBRATION_COORDINATOR_ARTIFACT_SCHEMA_VERSION = CALIBRATION_LEARNING_COORDINATOR_ARTIFACT_SCHEMA_VERSION


class CalibrationLearningLifecycleState(StrEnum):
  OFFROAD = "offroad"
  ONROAD = "onroad"


@dataclass(frozen=True, slots=True)
class CalibrationNodeSupportDiagnostic:
  """Independent support populations for one speed-local calibration node."""

  node_index: int
  speed_mps: float
  minimum_base_support_s: float
  clean_support_s: float
  supported_sample_count: int
  base_support_s: float
  base_sample_count: int
  full_fit_support_s: float
  full_fit_count: int
  completed_route_count: int
  base_completed_route_count: int
  moving_support_s: float
  moving_sample_count: int
  moving_full_fit_support_s: float
  moving_full_fit_count: int
  moving_completed_route_count: int
  breakaway_support_s: float
  breakaway_sample_count: int
  breakaway_full_fit_support_s: float
  breakaway_full_fit_count: int
  breakaway_episode_completed_route_count: int
  authority_support_s: float
  authority_sample_count: int
  authority_magnitude_sample_count: int
  authority_slew_build_sample_count: int
  authority_slew_release_sample_count: int
  authority_unresolved_sample_count: int
  authority_fit_support_s: float
  authority_fit_sample_count: int
  authority_full_fit_support_s: float
  authority_full_fit_count: int
  authority_completed_route_count: int
  lateral_accel_span_mps2: float
  applied_torque_span: float
  lateral_accel_directions: int
  applied_torque_directions: int
  rack_reversals: int


@dataclass(frozen=True, slots=True)
class CalibrationLearningFinalization:
  """Canonical evidence, selected proof, and optional learned calibration."""

  manifest_bytes: bytes
  manifest_sha256: str
  evidence_bytes: bytes
  evidence_sha256: str
  selected_profile_json: bytes | None
  selected_profile_sha256: str | None
  candidate_profile_json: bytes | None
  candidate_profile_sha256: str | None
  learning_result: CalibrationLearningResult
  sample_accounting: CalibrationSampleAccounting

  @property
  def all_nodes_qualified(self) -> bool:
    return self.learning_result.all_nodes_qualified


def _canonical_json_bytes(payload: object) -> bytes:
  return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _profile_bytes(profile: VehicleCalibrationProfile) -> bytes:
  return profile.to_json().encode("utf-8")


def _profile_sha256(profile: VehicleCalibrationProfile) -> str:
  return hashlib.sha256(_profile_bytes(profile)).hexdigest()


def _finite_float_hex(value: float, field_name: str) -> str:
  numeric = float(value)
  if not math.isfinite(numeric):
    raise ValueError(f"{field_name} must be finite")
  return numeric.hex()


def _manifest_value(value: object, field_name: str) -> object:
  """Encode a report value without losing a single deterministic field.

  Qualification reports are an audit surface.  Walking their dataclass fields
  keeps the manifest faithful when a measured-evidence diagnostic is added,
  while the explicit legacy-name guard prevents retired rack dynamics from
  silently re-entering the observable-calibration contract.
  """
  if value is None or isinstance(value, (str, bool, int)):
    return value
  if isinstance(value, float):
    return _finite_float_hex(value, field_name)
  if isinstance(value, Enum):
    return value.value
  if isinstance(value, (tuple, list)):
    return [_manifest_value(item, f"{field_name}[]") for item in value]
  if is_dataclass(value) and not isinstance(value, type):
    encoded: dict[str, object] = {}
    for report_field in fields(value):
      name = report_field.name
      lowered = name.lower()
      if any(term in lowered for term in ("rack_gain", "rack_damping", "rack_acceleration", "plant_dynamics")):
        raise ValueError(f"retired rack-dynamics field is forbidden in calibration manifest: {name}")
      encoded[name] = _manifest_value(getattr(value, name), f"{field_name}.{name}")
    return encoded
  raise TypeError(f"{field_name} has unsupported manifest type {type(value).__name__}")


def _qualification_manifest(report: object) -> dict[str, object]:
  encoded = _manifest_value(report, "node_report")
  if not isinstance(encoded, dict):
    raise TypeError("calibration qualification report must be a dataclass")
  reasons = encoded.get("reasons")
  if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
    raise ValueError("calibration qualification reasons are malformed")
  encoded["moving_reasons"] = [reason for reason in reasons if "moving" in reason]
  encoded["breakaway_reasons"] = [reason for reason in reasons if "breakaway" in reason]
  # These fields are the semantic distinction from the physical-profile fit.
  required = {
    "base_support_s",
    "base_sample_count",
    "moving_support_s",
    "moving_sample_count",
    "moving_reasons",
    "moving_full_fit_seed_rms",
    "moving_full_fit_candidate_rms",
    "breakaway_support_s",
    "breakaway_sample_count",
    "breakaway_reasons",
    "breakaway_full_fit_seed_rms",
    "breakaway_full_fit_candidate_rms",
  }
  missing = required - encoded.keys()
  if missing:
    raise ValueError(f"calibration qualification report is incomplete: missing={sorted(missing)}")
  return encoded


def _atomic_write_bytes(path: str | os.PathLike[str], encoded: bytes) -> None:
  """Atomically replace one artifact and durably commit its directory entry."""
  if type(encoded) is not bytes:
    raise TypeError("atomic calibration artifact must be bytes")
  target = Path(path)
  parent = target.parent
  temporary_fd, temporary_name = tempfile.mkstemp(
    dir=parent,
    prefix=f".{target.name}.",
    suffix=".tmp",
  )
  try:
    with os.fdopen(temporary_fd, "wb") as temporary:
      temporary_fd = -1
      temporary.write(encoded)
      temporary.flush()
      os.fsync(temporary.fileno())
    os.replace(temporary_name, target)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(parent, directory_flags)
    try:
      os.fsync(directory_fd)
    finally:
      os.close(directory_fd)
  except BaseException:
    if temporary_fd >= 0:
      os.close(temporary_fd)
    try:
      os.unlink(temporary_name)
    except FileNotFoundError:
      pass
    raise


class CalibrationLearningCoordinator:
  """Measured-only in-memory learner with an offroad persistence boundary."""

  def __init__(
    self,
    seed_profile: VehicleCalibrationProfile,
    evidence_bytes: bytes | None = None,
    *,
    candidate_provenance: str = "measured observable casual-driving evidence",
    expected_route_commitments: tuple[tuple[str, str], ...] | None = None,
  ) -> None:
    if not isinstance(seed_profile, VehicleCalibrationProfile):
      raise TypeError("calibration coordinator requires a VehicleCalibrationProfile")
    provenance = str(candidate_provenance).strip()
    if not provenance:
      raise ValueError("candidate provenance must not be empty")

    if evidence_bytes is None:
      self._learner = CalibrationProfileLearner(seed_profile)
    elif expected_route_commitments is None:
      self._learner = CalibrationProfileLearner.from_evidence(
        seed_profile,
        evidence_bytes,
      )
    else:
      self._learner = CalibrationProfileLearner.from_evidence(
        seed_profile,
        evidence_bytes,
        expected_route_commitments=expected_route_commitments,
      )
    self._seed_profile = seed_profile
    self._candidate_provenance = provenance
    self._state = CalibrationLearningLifecycleState.OFFROAD
    self._clean_sample_count = 0
    self._evidence_generation = 0
    self._finalized_generation = -1
    self._cached_finalization: CalibrationLearningFinalization | None = None

  @property
  def state(self) -> CalibrationLearningLifecycleState:
    return self._state

  @property
  def vehicle_identity(self) -> str:
    return self._seed_profile.vehicle_identity

  @property
  def seed_profile_sha256(self) -> str:
    return _profile_sha256(self._seed_profile)

  @property
  def ingested_sample_count(self) -> int:
    return self.sample_accounting.ingested_sample_count

  @property
  def clean_sample_count(self) -> int:
    return self._clean_sample_count

  @property
  def accepted_sample_count(self) -> int:
    return self.sample_accounting.accepted_sample_count

  @property
  def rejected_sample_count(self) -> int:
    return self.sample_accounting.rejected_sample_count

  @property
  def sample_accounting(self) -> CalibrationSampleAccounting:
    accounting = self._learner.sample_accounting
    if not isinstance(accounting, CalibrationSampleAccounting):
      raise TypeError("calibration learner emitted invalid sample accounting")
    return accounting

  @property
  def support_diagnostics(self) -> tuple[CalibrationNodeSupportDiagnostic, ...]:
    diagnostics: list[CalibrationNodeSupportDiagnostic] = []
    for node_index, speed_mps in enumerate(self._learner.speed_nodes_mps):
      evidence = self._learner.evidence_for_node(node_index)
      diagnostics.append(
        CalibrationNodeSupportDiagnostic(
          node_index=node_index,
          speed_mps=speed_mps,
          minimum_base_support_s=minimum_calibration_support_s(speed_mps),
          clean_support_s=evidence.clean_support_s,
          supported_sample_count=evidence.supported_sample_count,
          base_support_s=evidence.base_support_s,
          base_sample_count=evidence.base_sample_count,
          full_fit_support_s=evidence.full_fit_support_s,
          full_fit_count=evidence.full_fit_count,
          completed_route_count=evidence.completed_route_count,
          base_completed_route_count=evidence.base_completed_route_count,
          moving_support_s=evidence.moving_support_s,
          moving_sample_count=evidence.moving_sample_count,
          moving_full_fit_support_s=evidence.moving_full_fit_support_s,
          moving_full_fit_count=evidence.moving_full_fit_count,
          moving_completed_route_count=evidence.moving_completed_route_count,
          breakaway_support_s=evidence.breakaway_support_s,
          breakaway_sample_count=evidence.breakaway_sample_count,
          breakaway_full_fit_support_s=evidence.breakaway_full_fit_support_s,
          breakaway_full_fit_count=evidence.breakaway_full_fit_count,
          breakaway_episode_completed_route_count=evidence.breakaway_episode_completed_route_count,
          authority_support_s=evidence.authority_support_s,
          authority_sample_count=evidence.authority_sample_count,
          authority_magnitude_sample_count=evidence.authority_magnitude_sample_count,
          authority_slew_build_sample_count=evidence.authority_slew_build_sample_count,
          authority_slew_release_sample_count=evidence.authority_slew_release_sample_count,
          authority_unresolved_sample_count=evidence.authority_unresolved_sample_count,
          authority_fit_support_s=evidence.authority_fit_support_s,
          authority_fit_sample_count=evidence.authority_fit_sample_count,
          authority_full_fit_support_s=evidence.authority_full_fit_support_s,
          authority_full_fit_count=evidence.authority_full_fit_count,
          authority_completed_route_count=evidence.authority_completed_route_count,
          lateral_accel_span_mps2=evidence.lateral_accel_span_mps2,
          applied_torque_span=evidence.applied_torque_span,
          lateral_accel_directions=evidence.lateral_accel_directions,
          applied_torque_directions=evidence.applied_torque_directions,
          rack_reversals=evidence.rack_reversals,
        )
      )
    return tuple(diagnostics)

  def transition_onroad(
    self,
    route_identity_sha256: str,
    route_content_sha256: str | None = None,
    *,
    route_counter: int,
  ) -> None:
    if self._state is not CalibrationLearningLifecycleState.OFFROAD:
      raise RuntimeError("calibration coordinator is already onroad")
    self._learner.begin_route(
      route_identity_sha256,
      route_content_sha256,
      route_counter=route_counter,
    )
    self._state = CalibrationLearningLifecycleState.ONROAD

  def transition_offroad(self) -> None:
    if self._state is not CalibrationLearningLifecycleState.ONROAD:
      raise RuntimeError("calibration coordinator is already offroad")
    self._learner.end_route()
    self._state = CalibrationLearningLifecycleState.OFFROAD
    # Route-level paired uncertainty changes even when every frame in a route
    # is rejected.  Never retain a finalization across a completed route.
    self._evidence_generation += 1
    self._cached_finalization = None
    self._finalized_generation = -1

  def ingest(
    self,
    sample: LearningSample,
    *,
    source_coordinate: CalibrationIngestionCoordinate | None = None,
    upstream_rejection: CalibrationSampleDisposition | None = None,
  ) -> bool:
    """Accumulate and durably classify one measured frame in memory."""
    if self._state is not CalibrationLearningLifecycleState.ONROAD:
      raise RuntimeError("calibration samples may be ingested only while onroad")
    if not isinstance(sample, LearningSample):
      raise TypeError("calibration coordinator requires a measured-only LearningSample")

    disposition = self._learner.add_sample_with_disposition(
      sample,
      upstream_rejection=upstream_rejection,
      source_coordinate=source_coordinate,
    )
    if not isinstance(disposition, CalibrationSampleDisposition):
      raise TypeError("calibration learner emitted an invalid disposition")
    accepted = disposition is CalibrationSampleDisposition.ACCEPTED
    if accepted:
      # Inverse calibration legitimately retains signed reversal rows even
      # though legacy ``LearningSample.clean`` excludes them for its dynamic
      # acceleration fit. Authority observations remain a separate population.
      if not sample.authority_evidence:
        self._clean_sample_count += 1
    # Rejections are durable evidence too. Every disposition invalidates a
    # cached finalization so persisted accounting can never lag ingestion.
    self._evidence_generation += 1
    self._cached_finalization = None
    self._finalized_generation = -1
    return accepted

  def finalize(self) -> CalibrationLearningFinalization:
    """Export canonical evidence and an optional all-node candidate offroad."""
    if self._state is not CalibrationLearningLifecycleState.OFFROAD:
      raise RuntimeError("calibration may be finalized only while offroad")
    if self._cached_finalization is not None and self._finalized_generation == self._evidence_generation:
      return self._cached_finalization

    evidence_bytes = self._learner.export_authoritative_evidence()
    evidence_identity = calibration_evidence_sha256(evidence_bytes)
    result = self._learner.qualify(self._candidate_provenance)
    candidate = result.candidate_profile
    selected = result.selected_profile
    if (selected is not None) != result.all_nodes_qualified:
      raise ValueError("qualified calibration selection completeness is inconsistent")
    if candidate is not None and not result.all_nodes_qualified:
      raise ValueError("calibration learner candidate completeness is inconsistent")
    if (
      candidate is None
      and result.contains_learned_change
      and result.all_nodes_qualified
    ):
      raise ValueError("qualified learned calibration disappeared before publication")
    if selected is not None and (
      not selected.qualified
      or selected.vehicle_identity != self.vehicle_identity
      or selected.schema_version != self._seed_profile.schema_version
    ):
      raise ValueError("calibration learner emitted an incompatible selected profile")
    if candidate is not None and (
      not candidate.qualified or candidate.vehicle_identity != self.vehicle_identity or candidate.schema_version != self._seed_profile.schema_version
    ):
      raise ValueError("calibration learner emitted an incompatible candidate profile")
    if candidate is not None and candidate != selected:
      raise ValueError("learned calibration candidate differs from selected profile")
    selected_json = None if selected is None else _profile_bytes(selected)
    selected_identity = None if selected_json is None else hashlib.sha256(selected_json).hexdigest()
    candidate_json = None if candidate is None else _profile_bytes(candidate)
    candidate_identity = None if candidate_json is None else hashlib.sha256(candidate_json).hexdigest()
    candidate_manifest: dict[str, object] | None = None
    if candidate is not None:
      candidate_manifest = {
        "profile_sha256": candidate_identity,
        "provenance": candidate.provenance,
        "revision": candidate.revision,
        "schema_version": candidate.schema_version,
      }

    manifest = {
      "all_nodes_qualified": result.all_nodes_qualified,
      "artifact_schema_version": CALIBRATION_COORDINATOR_ARTIFACT_SCHEMA_VERSION,
      "candidate_profile": candidate_manifest,
      "evidence_schema_version": CALIBRATION_EVIDENCE_SCHEMA_VERSION,
      "evidence_sha256": evidence_identity,
      "node_reports": [_qualification_manifest(report) for report in result.node_reports],
      "interpolation_reports": [
        _manifest_value(report, "interpolation_report")
        for report in result.interpolation_reports
      ],
      "route_commitments": [
        {
          "assignment_chain_sha256": commitment.assignment_chain_sha256,
          "assignment_record_count": commitment.assignment_record_count,
          "route_commitment_sha256": commitment.route_commitment_sha256,
          "route_content_sha256": commitment.route_content_sha256,
          "route_counter": commitment.route_counter,
          "route_identity_sha256": commitment.route_identity_sha256,
          "route_index": commitment.route_index,
        }
        for commitment in self._learner.route_commitments
      ],
      "seed_profile_revision": self._seed_profile.revision,
      "seed_profile_schema_version": self._seed_profile.schema_version,
      "seed_profile_sha256": self.seed_profile_sha256,
      "selected_profile": (
        None
        if selected is None
        else {
          "profile_sha256": selected_identity,
          "provenance": selected.provenance,
          "revision": selected.revision,
          "schema_version": selected.schema_version,
        }
      ),
      "vehicle_identity": self.vehicle_identity,
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    finalization = CalibrationLearningFinalization(
      manifest_bytes=manifest_bytes,
      manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
      evidence_bytes=evidence_bytes,
      evidence_sha256=evidence_identity,
      selected_profile_json=selected_json,
      selected_profile_sha256=selected_identity,
      candidate_profile_json=candidate_json,
      candidate_profile_sha256=candidate_identity,
      learning_result=result,
      sample_accounting=self.sample_accounting,
    )
    self._cached_finalization = finalization
    self._finalized_generation = self._evidence_generation
    return finalization

  def persist_finalized(
    self,
    *,
    evidence_path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    selected_profile_path: str | os.PathLike[str] | None = None,
    candidate_profile_path: str | os.PathLike[str] | None = None,
  ) -> CalibrationLearningFinalization:
    """Persist evidence/selection/candidate, then commit the manifest last."""
    if self._state is not CalibrationLearningLifecycleState.OFFROAD:
      raise RuntimeError("calibration artifacts may be written only while offroad")
    finalization = self.finalize()
    if (
      candidate_profile_path is not None
      and finalization.candidate_profile_json is None
      and not finalization.all_nodes_qualified
    ):
      raise RuntimeError("partial calibration evidence cannot emit a candidate profile")

    _atomic_write_bytes(evidence_path, finalization.evidence_bytes)
    if selected_profile_path is not None:
      if finalization.selected_profile_json is None:
        raise RuntimeError(
          "selected-profile path was supplied without a qualified selection"
        )
      _atomic_write_bytes(
        selected_profile_path,
        finalization.selected_profile_json,
      )
    if candidate_profile_path is not None and finalization.candidate_profile_json is not None:
      candidate_json = finalization.candidate_profile_json
      _atomic_write_bytes(candidate_profile_path, candidate_json)
    _atomic_write_bytes(manifest_path, finalization.manifest_bytes)
    return finalization

  @property
  def route_commitments(self) -> tuple[CalibrationRouteCommitment, ...]:
    return self._learner.route_commitments

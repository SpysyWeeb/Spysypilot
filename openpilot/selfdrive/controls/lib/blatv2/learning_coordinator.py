"""Lifecycle coordinator for BLaTv2's measured-only slow learner.

This module deliberately owns no fitting arithmetic and no live-controller
state.  It wraps :class:`ProfileLearner` with an explicit
offroad/onroad/offroad lifecycle, canonical artifact identities, and optional
atomic writes to caller-supplied paths.  Learning samples contain recorded
physical response only; desired/model curvature, controller requests, and
candidate outputs have no input path here.

The coordinator can produce an offroad candidate, but it cannot approve,
activate, or mutate a live profile.  A candidate profile is emitted only when
``ProfileLearner`` qualifies every speed node.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from openpilot.selfdrive.controls.lib.blatv2.learner import (
  LEARNING_EVIDENCE_SCHEMA_VERSION,
  LearningResult,
  LearningSample,
  NodeQualificationReport,
  ProfileLearner,
  learner_evidence_sha256,
  minimum_clean_support_s,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  VehicleProfile,
)


LEARNING_COORDINATOR_ARTIFACT_SCHEMA_VERSION = 1


class LearningLifecycleState(StrEnum):
  OFFROAD = "offroad"
  ONROAD = "onroad"


@dataclass(frozen=True, slots=True)
class NodeSupportDiagnostic:
  """Raw node-local evidence coverage, without a steering-feel judgment."""

  node_index: int
  speed_mps: float
  minimum_clean_support_s: float
  clean_support_s: float
  supported_sample_count: int
  lateral_accel_min_mps2: float
  lateral_accel_max_mps2: float
  lateral_accel_energy_mps4_s: float
  rack_travel_deg: float
  applied_torque_min: float
  applied_torque_max: float
  rack_reversals: int


@dataclass(frozen=True, slots=True)
class LearningFinalization:
  """Canonical offroad evidence, qualification manifest, and full candidate."""

  manifest_bytes: bytes
  manifest_sha256: str
  evidence_bytes: bytes
  evidence_sha256: str
  candidate_profile_json: bytes | None
  candidate_profile_sha256: str | None
  learning_result: LearningResult

  @property
  def all_nodes_qualified(self) -> bool:
    return self.candidate_profile_json is not None


def _canonical_json_bytes(payload: object) -> bytes:
  return json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
  ).encode("utf-8")


def _profile_sha256(profile: VehicleProfile) -> str:
  return hashlib.sha256(profile.to_json().encode("utf-8")).hexdigest()


def _finite_float_hex(value: float, field_name: str) -> str:
  numeric = float(value)
  if not math.isfinite(numeric):
    raise ValueError(f"{field_name} must be finite")
  return numeric.hex()


def _optional_float_hex(
  value: float | None,
  field_name: str,
) -> str | None:
  if value is None:
    return None
  return _finite_float_hex(value, field_name)


def _qualification_manifest(
  report: NodeQualificationReport,
) -> dict[str, object]:
  context = f"node_reports[{report.node_index}]"
  return {
    "applied_torque_span": _finite_float_hex(
      report.applied_torque_span,
      f"{context}.applied_torque_span",
    ),
    "candidate_validation_rms": _optional_float_hex(
      report.candidate_validation_rms,
      f"{context}.candidate_validation_rms",
    ),
    "clean_support_s": _finite_float_hex(
      report.clean_support_s,
      f"{context}.clean_support_s",
    ),
    "confidence": _finite_float_hex(
      report.confidence,
      f"{context}.confidence",
    ),
    "lateral_accel_rms_mps2": _finite_float_hex(
      report.lateral_accel_rms_mps2,
      f"{context}.lateral_accel_rms_mps2",
    ),
    "lateral_accel_span_mps2": _finite_float_hex(
      report.lateral_accel_span_mps2,
      f"{context}.lateral_accel_span_mps2",
    ),
    "minimum_support_s": _finite_float_hex(
      report.minimum_support_s,
      f"{context}.minimum_support_s",
    ),
    "node_index": report.node_index,
    "qualified": report.qualified,
    "rack_reversals": report.rack_reversals,
    "rack_travel_deg": _finite_float_hex(
      report.rack_travel_deg,
      f"{context}.rack_travel_deg",
    ),
    "reasons": [reason.value for reason in report.reasons],
    "seed_validation_rms": _optional_float_hex(
      report.seed_validation_rms,
      f"{context}.seed_validation_rms",
    ),
    "speed_mps": _finite_float_hex(
      report.speed_mps,
      f"{context}.speed_mps",
    ),
    "supported_sample_count": report.supported_sample_count,
    "training_count": report.training_count,
    "validation_count": report.validation_count,
    "validation_support_s": _finite_float_hex(
      report.validation_support_s,
      f"{context}.validation_support_s",
    ),
  }


def _atomic_write_bytes(
  path: str | os.PathLike[str],
  encoded: bytes,
) -> None:
  """Atomically replace one caller-selected artifact and fsync its directory."""
  if type(encoded) is not bytes:
    raise TypeError("atomic learning artifact must be bytes")
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


class LearningCoordinator:
  """Explicit slow-learning lifecycle around the physical ``ProfileLearner``."""

  def __init__(
    self,
    seed_profile: VehicleProfile,
    evidence_bytes: bytes | None = None,
    *,
    candidate_provenance: str = "measured casual-driving evidence",
  ) -> None:
    provenance = str(candidate_provenance).strip()
    if not provenance:
      raise ValueError("candidate provenance must not be empty")

    self._learner = (
      ProfileLearner(seed_profile)
      if evidence_bytes is None
      else ProfileLearner.from_evidence(seed_profile, evidence_bytes)
    )
    self._seed_profile = seed_profile
    self._candidate_provenance = provenance
    self._state = LearningLifecycleState.OFFROAD
    self._ingested_sample_count = 0
    self._clean_sample_count = 0
    self._accepted_sample_count = 0
    self._evidence_generation = 0
    self._finalized_generation = -1
    self._cached_finalization: LearningFinalization | None = None

  @property
  def state(self) -> LearningLifecycleState:
    return self._state

  @property
  def vehicle_identity(self) -> str:
    return self._seed_profile.vehicle_identity

  @property
  def seed_profile_sha256(self) -> str:
    return _profile_sha256(self._seed_profile)

  @property
  def ingested_sample_count(self) -> int:
    """Return samples presented in this in-memory coordinator lifetime."""
    return self._ingested_sample_count

  @property
  def clean_sample_count(self) -> int:
    """Return clean samples seen in this in-memory coordinator lifetime."""
    return self._clean_sample_count

  @property
  def accepted_sample_count(self) -> int:
    """Return samples accepted by ``ProfileLearner`` in this lifetime."""
    return self._accepted_sample_count

  @property
  def rejected_sample_count(self) -> int:
    return self._ingested_sample_count - self._accepted_sample_count

  @property
  def support_diagnostics(self) -> tuple[NodeSupportDiagnostic, ...]:
    diagnostics = []
    for node_index, speed_mps in enumerate(self._learner.speed_nodes_mps):
      evidence = self._learner.evidence_for_node(node_index)
      diagnostics.append(NodeSupportDiagnostic(
        node_index=node_index,
        speed_mps=speed_mps,
        minimum_clean_support_s=minimum_clean_support_s(speed_mps),
        clean_support_s=evidence.clean_support_s,
        supported_sample_count=evidence.supported_sample_count,
        lateral_accel_min_mps2=evidence.lateral_accel_min_mps2,
        lateral_accel_max_mps2=evidence.lateral_accel_max_mps2,
        lateral_accel_energy_mps4_s=(
          evidence.lateral_accel_energy_mps4_s
        ),
        rack_travel_deg=evidence.rack_travel_deg,
        applied_torque_min=evidence.applied_torque_min,
        applied_torque_max=evidence.applied_torque_max,
        rack_reversals=evidence.rack_reversals,
      ))
    return tuple(diagnostics)

  def transition_onroad(self) -> None:
    if self._state is not LearningLifecycleState.OFFROAD:
      raise RuntimeError("learning coordinator is already onroad")
    # Separate drives/routes have unrelated derivative and rack-direction
    # histories. Cumulative support and train/validation ordinals remain.
    self._learner.reset_route_transients()
    self._state = LearningLifecycleState.ONROAD

  def transition_offroad(self) -> None:
    if self._state is not LearningLifecycleState.ONROAD:
      raise RuntimeError("learning coordinator is already offroad")
    self._learner.reset_route_transients()
    self._state = LearningLifecycleState.OFFROAD

  def ingest(self, sample: LearningSample) -> bool:
    """Ingest measured response in memory; this method never writes storage."""
    if self._state is not LearningLifecycleState.ONROAD:
      raise RuntimeError("learning samples may be ingested only while onroad")
    if not isinstance(sample, LearningSample):
      raise TypeError("learning coordinator requires a LearningSample")

    self._ingested_sample_count += 1
    if sample.clean:
      self._clean_sample_count += 1
    accepted = self._learner.add_sample(sample)
    if accepted:
      self._accepted_sample_count += 1
      self._evidence_generation += 1
      self._cached_finalization = None
      self._finalized_generation = -1
    return accepted

  def finalize(self) -> LearningFinalization:
    """Export canonical evidence and, only if complete, a candidate profile."""
    if self._state is not LearningLifecycleState.OFFROAD:
      raise RuntimeError("learning may be finalized only while offroad")
    if (
      self._cached_finalization is not None
      and self._finalized_generation == self._evidence_generation
    ):
      return self._cached_finalization

    evidence_bytes = self._learner.export_evidence()
    evidence_identity = learner_evidence_sha256(evidence_bytes)
    result = self._learner.qualify(self._candidate_provenance)
    candidate = result.candidate_profile
    candidate_json = (
      None if candidate is None else candidate.to_json().encode("utf-8")
    )
    candidate_identity = (
      None
      if candidate_json is None
      else hashlib.sha256(candidate_json).hexdigest()
    )
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
      "artifact_schema_version": (
        LEARNING_COORDINATOR_ARTIFACT_SCHEMA_VERSION
      ),
      "candidate_profile": candidate_manifest,
      "evidence_schema_version": LEARNING_EVIDENCE_SCHEMA_VERSION,
      "evidence_sha256": evidence_identity,
      "node_reports": [
        _qualification_manifest(report)
        for report in result.node_reports
      ],
      "seed_profile_revision": self._seed_profile.revision,
      "seed_profile_schema_version": self._seed_profile.schema_version,
      "seed_profile_sha256": self.seed_profile_sha256,
      "vehicle_identity": self.vehicle_identity,
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    finalization = LearningFinalization(
      manifest_bytes=manifest_bytes,
      manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
      evidence_bytes=evidence_bytes,
      evidence_sha256=evidence_identity,
      candidate_profile_json=candidate_json,
      candidate_profile_sha256=candidate_identity,
      learning_result=result,
    )
    self._cached_finalization = finalization
    self._finalized_generation = self._evidence_generation
    return finalization

  def persist_finalized(
    self,
    *,
    evidence_path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    candidate_profile_path: str | os.PathLike[str] | None = None,
  ) -> LearningFinalization:
    """Atomically write a finalized offroad artifact to caller-owned paths."""
    if self._state is not LearningLifecycleState.OFFROAD:
      raise RuntimeError("learning artifacts may be written only while offroad")
    finalization = self.finalize()
    if (
      candidate_profile_path is not None
      and finalization.candidate_profile_json is None
    ):
      raise RuntimeError(
        "partial learning evidence cannot emit a candidate profile",
      )

    _atomic_write_bytes(evidence_path, finalization.evidence_bytes)
    if candidate_profile_path is not None:
      candidate_json = finalization.candidate_profile_json
      if candidate_json is None:
        raise AssertionError("candidate profile disappeared after preflight")
      _atomic_write_bytes(candidate_profile_path, candidate_json)
    # The manifest is written last and acts as the identity/commit record for
    # the independently atomic evidence and optional full-profile artifacts.
    _atomic_write_bytes(manifest_path, finalization.manifest_bytes)
    return finalization

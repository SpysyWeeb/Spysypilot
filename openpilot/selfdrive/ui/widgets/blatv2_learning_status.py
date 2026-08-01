"""Strict, display-only decoding for the BLaTv2 learning dashboard.

The UI must never infer controller activation from learner progress. Learning
and lifecycle snapshots are independent, rebuildable Params caches produced by
the owning daemons. This module contains no controller imports and performs no
file, route, evidence, fit, approval, or actuation work.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import re


LEARNING_STATUS_SCHEMA_VERSION = 3
LIFECYCLE_STATUS_SCHEMA_VERSION = 1
LEARNING_OPERATION_STATUS_SCHEMA_VERSION = 1
BACKFILL_PROGRESS_SCHEMA_VERSION = 1
BEHAVIOR_LEARNING_STATUS_SCHEMA_VERSION = 1

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_OPERATION_ID_RE = re.compile(r"[0-9a-f]{32}")
_TOP_LEVEL_KEYS = frozenset((
  "schema_version",
  "informational_only",
  "vehicle_identity",
  "runtime_identity_sha256",
  "seed_profile_sha256",
  "evidence_sha256",
  "manifest_sha256",
  "all_intervals_qualified",
  "all_nodes_evaluated",
  "all_nodes_qualified",
  "candidate_profile_available",
  "candidate_profile_sha256",
  "candidate_profile_revision",
  "interpolation_reports",
  "last_drive_complete",
  "nodes",
))
_NODE_KEYS = frozenset((
  "applied_torque_directions",
  "applied_torque_span",
  "authority_candidate_validation_rms",
  "authority_fit_sample_count",
  "authority_fit_support_s",
  "authority_sample_count",
  "authority_seed_validation_rms",
  "authority_support_s",
  "authority_training_count",
  "authority_validation_count",
  "base_sample_count",
  "base_support_s",
  "breakaway_candidate_validation_rms",
  "breakaway_sample_count",
  "breakaway_seed_validation_rms",
  "breakaway_support_s",
  "breakaway_training_count",
  "breakaway_validation_count",
  "candidate_parameters",
  "candidate_validation_rms",
  "clean_support_s",
  "confidence",
  "evaluation_status",
  "fit_diagnostics",
  "last_drive_accepted_sample_count",
  "last_drive_authority_fit_sample_count",
  "last_drive_authority_fit_support_s",
  "last_drive_authority_sample_count",
  "last_drive_authority_support_s",
  "last_drive_base_sample_count",
  "last_drive_base_support_s",
  "last_drive_breakaway_sample_count",
  "last_drive_breakaway_support_s",
  "last_drive_clean_support_s",
  "last_drive_moving_sample_count",
  "last_drive_moving_support_s",
  "lateral_accel_directions",
  "lateral_accel_rms_mps2",
  "lateral_accel_span_mps2",
  "minimum_support_s",
  "minimum_validation_support_s",
  "moving_candidate_validation_rms",
  "moving_sample_count",
  "moving_seed_validation_rms",
  "moving_support_s",
  "moving_training_count",
  "moving_validation_count",
  "node_index",
  "qualified",
  "rack_reversals",
  "rack_travel_deg",
  "reasons",
  "seed_validation_rms",
  "speed_mps",
  "supported_sample_count",
  "training_count",
  "training_outcome",
  "training_paired_loss",
  "validation_count",
  "validation_paired_loss",
  "validation_support_s",
))
_FIT_DIAGNOSTIC_KEYS = frozenset((
  "breakaway_parameter_count",
  "breakaway_rank",
  "condition_estimate",
  "model",
  "moving_parameter_count",
  "moving_rank",
  "status",
))
_PAIRED_LOSS_KEYS = frozenset((
  "lower_bound_mse",
  "mean_candidate_minus_seed_mse",
  "numerical_tolerance_mse",
  "route_count",
  "uncertainty_mse",
  "upper_bound_mse",
))
_INTERPOLATION_KEYS = frozenset((
  "interval_index",
  "lower_speed_mps",
  "qualified",
  "reasons",
  "training_paired_loss",
  "upper_speed_mps",
  "validation_paired_loss",
))
_PARAMETER_KEYS = frozenset((
  "kinetic_friction_torque",
  "lateral_accel_offset_correction_mps2",
  "static_breakaway_torque",
  "torque_per_lateral_accel",
))
_SUPPORT_SUM_REL_TOL = 1e-12
_SUPPORT_SUM_ABS_TOL = 1e-12
_LIFECYCLE_KEYS = frozenset((
  "schema_version",
  "informational_only",
  "vehicle_identity",
  "runtime_identity_sha256",
  "source_openpilot_commit",
  "opendbc_commit",
  "activation_state_sha256",
  "diagnostic",
  "controller_state",
  "effective_controller",
  "production_envelope_verified",
  "active_profile",
  "staged_profile",
  "rejected_profile_count",
))
_OPERATION_KEYS = frozenset((
  "schema_version",
  "informational_only",
  "state",
  "diagnostic",
  "operation_id",
  "sequence",
  "started_mono_ns",
  "updated_mono_ns",
  "terminal",
  "vehicle_identity",
  "runtime_identity_sha256",
  "current_route_identity",
  "current_route_index",
  "total_route_count",
  "last_route_identity",
  "accepted_sample_count",
  "rejected_sample_count",
  "retry_count",
  "evidence_sha256",
  "ledger_sha256",
))
_BACKFILL_PROGRESS_KEYS = frozenset((
  "schema_version",
  "informational_only",
  "operation_id",
  "operation_sequence",
  "sequence",
  "updated_mono_ns",
  "phase",
  "pass_index",
  "pass_count",
  "current_route_identity",
  "current_route_index",
  "total_route_count",
  "current_segment_index",
  "current_route_segment_count",
  "completed_replay_segment_count",
  "total_replay_segment_count",
  "completed_work_units",
  "total_work_units",
  "approximate_remaining_seconds",
))
_PROFILE_IDENTITY_KEYS = frozenset((
  "artifact_sha256",
  "profile_sha256",
  "profile_revision",
))
_BEHAVIOR_STATUS_KEYS = frozenset((
  "behaviorFinalizationSha256",
  "behaviorSelectionSha256",
  "completedReplayJobs",
  "currentCandidateIndex",
  "currentRouteIdentity",
  "currentRouteIndex",
  "diagnostic",
  "eligibleRouteCount",
  "gateSpecSha256",
  "informationalOnly",
  "operationId",
  "physicalGenerationSha256",
  "physicalProfileSha256",
  "qualificationDisposition",
  "reasons",
  "recordedSourceIdentitySha256",
  "requiredRouteCount",
  "runtimeVehicleIdentitySha256",
  "schemaVersion",
  "segmentationConfigSha256",
  "selectedBehaviorPolicySha256",
  "sequence",
  "smoothPassed",
  "startedMonoNs",
  "state",
  "strongPassed",
  "swiftPassed",
  "targetMateriallyImproved",
  "terminal",
  "totalCandidateCount",
  "totalReplayJobs",
  "totalRouteCount",
  "trainingRouteCount",
  "transactionSha256",
  "updatedMonoNs",
  "validationRouteCount",
  "vehicleIdentity",
))
_BEHAVIOR_STATE_DIAGNOSTICS = {
  "waiting_for_physical_profile": frozenset((
    "physical_profile_unqualified",
  )),
  "waiting_for_routes": frozenset((
    "insufficient_homogeneous_routes",
  )),
  "preparing": frozenset(("validating_route_evidence",)),
  "training": frozenset(("replaying_training_grid",)),
  "selecting": frozenset(("selecting_training_winner",)),
  "validating": frozenset(("replaying_frozen_winner",)),
  "publishing": frozenset(("publishing_behavior_generation",)),
  "complete": frozenset(("candidate_qualified", "stock_retained")),
  "failed": frozenset((
    "route_evidence_invalid",
    "replay_nondeterministic",
    "behavior_transaction_failed",
    "behavior_publish_failed",
  )),
}
_BEHAVIOR_TERMINAL_STATES = frozenset(("complete", "failed"))
_BEHAVIOR_DISPOSITIONS = frozenset((
  "stock_retained",
  "qualified_candidate_available",
))
_REASONS = frozenset((
  "qualified",
  "learned",
  "seed_retained",
  "insufficient_support",
  "insufficient_validation",
  "insufficient_excitation",
  "insufficient_moving_evidence",
  "insufficient_breakaway_evidence",
  "rank_deficient_fit",
  "ill_conditioned_fit",
  "singular_fit",
  "invalid_parameters",
  "validation_inconclusive",
  "validation_regression",
  "moving_validation_regression",
  "breakaway_validation_regression",
  "authority_validation_regression",
  "interpolation_training_inconclusive",
  "interpolation_training_regression",
  "interpolation_validation_inconclusive",
  "interpolation_validation_regression",
))
_REASON_LABELS = {
  "qualified": "Qualified",
  "learned": "Learned",
  "seed_retained": "Seed retained (calibration already good)",
  "insufficient_support": "Collecting clean driving",
  "insufficient_validation": "Needs held-out validation",
  "insufficient_excitation": "Needs more steering variety",
  "insufficient_moving_evidence": "Needs moving-rack evidence",
  "insufficient_breakaway_evidence": "Needs breakaway evidence",
  "rank_deficient_fit": "Rank deficient",
  "ill_conditioned_fit": "Ill conditioned",
  "singular_fit": "Fit not identifiable",
  "invalid_parameters": "Rejected: invalid fit",
  "validation_inconclusive": "Validation inconclusive",
  "validation_regression": "Rejected: validation regressed",
  "moving_validation_regression": "Rejected: moving fit regressed",
  "breakaway_validation_regression": "Rejected: breakaway fit regressed",
  "authority_validation_regression": (
    "Rejected: authority validation regressed"
  ),
  "interpolation_training_inconclusive": "Interpolation training inconclusive",
  "interpolation_training_regression": "Interpolation training regressed",
  "interpolation_validation_inconclusive": "Interpolation validation inconclusive",
  "interpolation_validation_regression": "Interpolation validation regressed",
}
_REASON_PRIORITY = (
  "insufficient_support",
  "insufficient_validation",
  "insufficient_excitation",
  "insufficient_moving_evidence",
  "insufficient_breakaway_evidence",
  "rank_deficient_fit",
  "ill_conditioned_fit",
  "invalid_parameters",
  "validation_inconclusive",
  "authority_validation_regression",
  "breakaway_validation_regression",
  "moving_validation_regression",
  "validation_regression",
  "singular_fit",
  "learned",
  "seed_retained",
  "qualified",
)
_EVALUATION_STATUSES = frozenset((
  "evidence_insufficient",
  "ill_conditioned",
  "invalid_parameters",
  "learned",
  "numerical_failure",
  "rank_deficient",
  "seed_retained",
  "validation_inconclusive",
  "validation_regressed",
))
_FIT_MODELS = frozenset((
  "static_only",
  "friction_map",
  "offset_and_friction",
  "full_map",
))
_FIT_STATUSES = frozenset((
  "identifiable",
  "rank_deficient",
  "ill_conditioned",
  "no_solution",
))
_NODE_OUTCOME_LABELS = {
  "learned": "Learned",
  "seed_retained": "Seed retained (calibration already good)",
  "evidence_insufficient": "Needs more/varied evidence",
  "rank_deficient": "Rank deficient",
  "ill_conditioned": "Ill conditioned",
  "validation_inconclusive": "Validation inconclusive",
  "validation_regressed": "Validation regressed",
  "invalid_parameters": "Calibration invalid",
  "numerical_failure": "Numerical fit failure",
}
_CONTROLLER_STATES = frozenset((
  "stock",
  "staged",
  "provisional",
  "approved",
  "rollback_pending",
  "unavailable",
))
_EFFECTIVE_CONTROLLERS = frozenset(("stock", "modular"))
_ARTIFACT_DIAGNOSTICS = frozenset((
  "ok",
  "absent",
  "param_read_error",
  "malformed",
  "profile_hash_mismatch",
  "policy_hash_mismatch",
  "unqualified_profile",
  "provisional_policy",
  "vehicle_mismatch",
  "runtime_vehicle_mismatch",
  "source_commit_mismatch",
  "opendbc_commit_mismatch",
  "unverified_actuation_envelope",
  "gate_failed",
  "state_invalid",
  "state_stale_build",
))
_OPERATION_STATE_DIAGNOSTICS = {
  "preparing": frozenset(("waiting_for_car_params", "restoring_runtime")),
  "ready_no_evidence": frozenset(("ready_for_first_drive",)),
  "collecting": frozenset(("collecting_current_drive",)),
  "finalizing": frozenset((
    "finalizing_drive",
    "verifying_backfill",
    "publishing_backfill",
  )),
  "retry_pending": frozenset(("persist_retry_pending",)),
  "backfilling": frozenset(("scanning_routes", "replaying_route")),
  "idle": frozenset((
    "evidence_ready",
    "backfill_complete",
    "backfill_complete_late_older_skipped",
    "backfill_complete_with_rejections",
  )),
  "drive_skipped_identity_mismatch": frozenset(("car_params_identity_mismatch",)),
  "failed": frozenset((
    "runtime_restore_failed",
    "backfill_reader_unavailable",
    "backfill_route_incompatible",
    "backfill_corrupt_log",
    "backfill_nondeterministic",
    "backfill_publish_failed",
    "backfill_untracked_evidence",
    "backfill_no_complete_routes",
    "unexpected_error",
  )),
}
_TERMINAL_OPERATION_STATES = frozenset((
  "ready_no_evidence",
  "idle",
  "drive_skipped_identity_mismatch",
  "failed",
))
_BACKFILL_PROGRESS_PHASES = frozenset((
  "reading_segment",
  "applying_route",
  "comparing",
  "publishing",
))
_OPERATION_DIAGNOSTIC_LABELS = {
  "waiting_for_car_params": "Waiting for vehicle configuration",
  "restoring_runtime": "Restoring the prepared learning runtime",
  "ready_for_first_drive": "Ready to collect the first drive",
  "collecting_current_drive": "Collecting clean evidence from this drive",
  "finalizing_drive": "Validating and saving the completed drive",
  "verifying_backfill": "Verifying replayed route evidence",
  "publishing_backfill": "Publishing the rebuilt evidence snapshot",
  "persist_retry_pending": "Saving failed; a safe retry is pending",
  "scanning_routes": "Scanning compatible routes on device",
  "replaying_route": "Replaying a compatible historical route",
  "evidence_ready": "Validated learning evidence is ready",
  "backfill_complete": "Historical route processing is complete",
  "backfill_complete_late_older_skipped": ("Backfill complete; late older routes were safely skipped"),
  "backfill_complete_with_rejections": ("Backfill complete; incompatible routes were rejected"),
  "car_params_identity_mismatch": ("Vehicle configuration changed; prepared for the next drive"),
  "runtime_restore_failed": "Prepared learner runtime could not be restored",
  "backfill_reader_unavailable": "Historical route reader is unavailable",
  "backfill_route_incompatible": "A historical route is incompatible",
  "backfill_corrupt_log": "A historical route log is corrupt",
  "backfill_nondeterministic": "Historical replay did not reproduce exactly",
  "backfill_publish_failed": "Rebuilt evidence could not be published",
  "backfill_untracked_evidence": ("Backfill unavailable: stored evidence has no route ledger"),
  "backfill_no_complete_routes": "No complete compatible routes were found",
  "unexpected_error": "The learner reported an unexpected error",
}


class LearningStatusError(ValueError):
  """A stable fail-closed reason for an unavailable learning display."""

  def __init__(self, code: str, message: str):
    super().__init__(message)
    self.code = code


@dataclass(frozen=True, slots=True)
class CandidateParameters:
  torque_per_lateral_accel: float
  lateral_accel_offset_correction_mps2: float
  kinetic_friction_torque: float
  static_breakaway_torque: float


@dataclass(frozen=True, slots=True)
class FitDiagnostic:
  model: str
  status: str
  moving_rank: int
  moving_parameter_count: int
  condition_estimate: float | None
  breakaway_rank: int
  breakaway_parameter_count: int


@dataclass(frozen=True, slots=True)
class PairedLoss:
  route_count: int
  mean_candidate_minus_seed_mse: float | None
  numerical_tolerance_mse: float | None
  uncertainty_mse: float | None
  lower_bound_mse: float | None
  upper_bound_mse: float | None


@dataclass(frozen=True, slots=True)
class InterpolationStatus:
  interval_index: int
  lower_speed_mps: float
  upper_speed_mps: float
  qualified: bool
  reasons: tuple[str, ...]
  training_paired_loss: PairedLoss
  validation_paired_loss: PairedLoss

  @property
  def outcome(self) -> str:
    if self.qualified:
      return "qualified"
    if any("regression" in reason for reason in self.reasons):
      return "regressed"
    return "inconclusive"


@dataclass(frozen=True, slots=True)
class LearningNodeStatus:
  node_index: int
  speed_mps: float
  minimum_support_s: float
  clean_support_s: float
  last_drive_clean_support_s: float | None
  supported_sample_count: int
  last_drive_accepted_sample_count: int | None
  base_support_s: float
  base_sample_count: int
  last_drive_base_support_s: float | None
  last_drive_base_sample_count: int | None
  moving_support_s: float
  moving_sample_count: int
  moving_training_count: int
  moving_validation_count: int
  last_drive_moving_support_s: float | None
  last_drive_moving_sample_count: int | None
  breakaway_support_s: float
  breakaway_sample_count: int
  breakaway_training_count: int
  breakaway_validation_count: int
  last_drive_breakaway_support_s: float | None
  last_drive_breakaway_sample_count: int | None
  authority_support_s: float
  authority_sample_count: int
  authority_fit_support_s: float
  authority_fit_sample_count: int
  authority_training_count: int
  authority_validation_count: int
  last_drive_authority_support_s: float | None
  last_drive_authority_sample_count: int | None
  last_drive_authority_fit_support_s: float | None
  last_drive_authority_fit_sample_count: int | None
  training_count: int
  validation_count: int
  validation_support_s: float
  minimum_validation_support_s: float
  lateral_accel_span_mps2: float
  lateral_accel_rms_mps2: float
  rack_travel_deg: float
  applied_torque_span: float
  rack_reversals: int
  lateral_accel_directions: int
  applied_torque_directions: int
  seed_validation_rms: float | None
  candidate_validation_rms: float | None
  moving_seed_validation_rms: float | None
  moving_candidate_validation_rms: float | None
  breakaway_seed_validation_rms: float | None
  breakaway_candidate_validation_rms: float | None
  authority_seed_validation_rms: float | None
  authority_candidate_validation_rms: float | None
  confidence: float
  qualified: bool
  evaluation_status: str
  fit_diagnostics: tuple[FitDiagnostic, ...]
  training_outcome: str | None
  training_paired_loss: PairedLoss | None
  validation_paired_loss: PairedLoss | None
  reasons: tuple[str, ...]
  candidate_parameters: CandidateParameters | None

  @property
  def support_fraction(self) -> float:
    if self.minimum_support_s <= 0.0:
      return 0.0
    return max(0.0, min(1.0, self.clean_support_s / self.minimum_support_s))

  @property
  def validation_fraction(self) -> float:
    if self.minimum_validation_support_s <= 0.0:
      return 0.0
    return max(
      0.0,
      min(1.0, self.validation_support_s / self.minimum_validation_support_s),
    )

  @property
  def primary_reason(self) -> str:
    # Sparse evidence can make the provisional normal equations singular or
    # physically invalid long before the node has enough coverage to judge a
    # fit. Collection blockers therefore take display priority; a red fit
    # failure is meaningful only after time, validation, and excitation pass.
    for reason in _REASON_PRIORITY:
      if reason in self.reasons:
        return reason
    return "insufficient_support"

  @property
  def outcome_label(self) -> str:
    return _NODE_OUTCOME_LABELS[self.evaluation_status]

  @property
  def collection_complete(self) -> bool:
    return not any(
      reason in self.reasons
      for reason in (
        "insufficient_support",
        "insufficient_validation",
        "insufficient_excitation",
        "insufficient_moving_evidence",
        "insufficient_breakaway_evidence",
      )
    )

  @property
  def moving_ready(self) -> bool:
    return "insufficient_moving_evidence" not in self.reasons

  @property
  def breakaway_ready(self) -> bool:
    return "insufficient_breakaway_evidence" not in self.reasons


@dataclass(frozen=True, slots=True)
class LearningStatus:
  vehicle_identity: str
  runtime_identity_sha256: str
  seed_profile_sha256: str
  evidence_sha256: str
  manifest_sha256: str
  all_nodes_evaluated: bool
  all_intervals_qualified: bool
  all_nodes_qualified: bool
  candidate_profile_available: bool
  candidate_profile_sha256: str | None
  candidate_profile_revision: int | None
  last_drive_complete: bool
  nodes: tuple[LearningNodeStatus, ...]
  interpolation_reports: tuple[InterpolationStatus, ...]

  @property
  def qualified_node_count(self) -> int:
    return sum(node.qualified for node in self.nodes)

  @property
  def learned_node_count(self) -> int:
    return sum(node.evaluation_status == "learned" for node in self.nodes)

  @property
  def retained_node_count(self) -> int:
    return sum(node.evaluation_status == "seed_retained" for node in self.nodes)


@dataclass(frozen=True, slots=True)
class LearningSummaryLine:
  text: str
  tone: str


def node_outcome_tone(node: LearningNodeStatus) -> str:
  """Map the learner's explicit node verdict to the dashboard color grammar."""
  if node.evaluation_status in ("learned", "seed_retained"):
    return "green"
  if node.evaluation_status == "evidence_insufficient":
    return "gray" if node.clean_support_s <= 0.0 else "blue"
  if node.evaluation_status == "validation_inconclusive":
    return "amber"
  return "red"


def learning_summary_lines(status: LearningStatus) -> tuple[LearningSummaryLine, ...]:
  """Return three independent rows: node results, artifact, interpolation."""
  unresolved = len(status.nodes) - status.qualified_node_count
  node_parts = (
    f"{status.learned_node_count} LEARNED",
    f"{status.retained_node_count} SEED RETAINED",
    f"{unresolved} NEED REVIEW",
  )
  node_tone = "green" if unresolved == 0 else "blue"

  if status.candidate_profile_available:
    artifact = LearningSummaryLine(
      "NEW CANDIDATE ARTIFACT AVAILABLE",
      "green",
    )
  elif status.all_nodes_qualified:
    artifact = LearningSummaryLine(
      "SEED PROFILE RETAINED | NO NEW ARTIFACT NEEDED",
      "green",
    )
  elif status.all_nodes_evaluated:
    artifact = LearningSummaryLine(
      "NO CANDIDATE ARTIFACT | INTERPOLATION NOT QUALIFIED",
      "amber",
    )
  else:
    artifact = LearningSummaryLine(
      "CANDIDATE ARTIFACT PENDING NODE EVALUATION",
      "gray",
    )

  if not status.interpolation_reports:
    interpolation = LearningSummaryLine("INTERPOLATION PENDING", "gray")
  else:
    qualified = sum(report.qualified for report in status.interpolation_reports)
    regressed = sum(report.outcome == "regressed" for report in status.interpolation_reports)
    inconclusive = sum(report.outcome == "inconclusive" for report in status.interpolation_reports)
    if status.all_intervals_qualified:
      text = f"INTERPOLATION {qualified}/{len(status.interpolation_reports)} QUALIFIED"
      tone = "green"
    else:
      details = []
      if inconclusive:
        details.append(f"{inconclusive} INCONCLUSIVE")
      if regressed:
        details.append(f"{regressed} REGRESSED")
      text = "INTERPOLATION | " + " | ".join(details)
      tone = "red" if regressed else "amber"
    interpolation = LearningSummaryLine(text, tone)

  return (
    LearningSummaryLine(" | ".join(node_parts), node_tone),
    artifact,
    interpolation,
  )


@dataclass(frozen=True, slots=True)
class LearningOperationStatus:
  """Display-only state of learner collection, finalization, and backfill."""

  state: str
  diagnostic: str
  operation_id: str
  sequence: int
  started_mono_ns: int
  updated_mono_ns: int
  terminal: bool
  vehicle_identity: str | None
  runtime_identity_sha256: str | None
  current_route_identity: str | None
  current_route_index: int | None
  total_route_count: int | None
  last_route_identity: str | None
  accepted_sample_count: int
  rejected_sample_count: int
  retry_count: int
  evidence_sha256: str | None
  ledger_sha256: str | None

  @property
  def active(self) -> bool:
    return not self.terminal


@dataclass(frozen=True, slots=True)
class BackfillProgressStatus:
  """Optional display-only detail bound to one operation-status snapshot."""

  operation_id: str
  operation_sequence: int
  sequence: int
  updated_mono_ns: int
  phase: str
  pass_index: int
  pass_count: int
  current_route_identity: str | None
  current_route_index: int | None
  total_route_count: int
  current_segment_index: int | None
  current_route_segment_count: int | None
  completed_replay_segment_count: int
  total_replay_segment_count: int
  completed_work_units: int
  total_work_units: int
  approximate_remaining_seconds: int | None

  @property
  def progress_fraction(self) -> float:
    return self.completed_work_units / self.total_work_units


@dataclass(frozen=True, slots=True)
class BehaviorLearningStatus:
  """Display-only projection of one immutable behavior-learning operation."""

  operation_id: str
  sequence: int
  state: str
  diagnostic: str
  terminal: bool
  started_mono_ns: int
  updated_mono_ns: int
  vehicle_identity: str
  runtime_vehicle_identity_sha256: str
  physical_generation_sha256: str | None
  physical_profile_sha256: str | None
  recorded_source_identity_sha256: str | None
  eligible_route_count: int
  required_route_count: int
  training_route_count: int
  validation_route_count: int
  current_route_identity: str | None
  current_route_index: int | None
  total_route_count: int
  current_candidate_index: int | None
  total_candidate_count: int
  completed_replay_jobs: int
  total_replay_jobs: int
  gate_spec_sha256: str
  segmentation_config_sha256: str
  transaction_sha256: str | None
  behavior_finalization_sha256: str | None
  behavior_selection_sha256: str | None
  selected_behavior_policy_sha256: str | None
  smooth_passed: bool | None
  swift_passed: bool | None
  strong_passed: bool | None
  target_materially_improved: bool | None
  qualification_disposition: str | None
  reasons: tuple[str, ...]

  @property
  def active(self) -> bool:
    return not self.terminal

  @property
  def route_readiness_fraction(self) -> float:
    if self.required_route_count <= 0:
      return 0.0
    return min(1.0, self.eligible_route_count / self.required_route_count)

  @property
  def replay_progress_fraction(self) -> float:
    if self.total_replay_jobs <= 0:
      return 0.0
    return self.completed_replay_jobs / self.total_replay_jobs


@dataclass(frozen=True, slots=True)
class BehaviorPresentation:
  """Pure copy/color model; never an activation decision."""

  title: str
  detail: str
  tone: str
  progress_fraction: float | None


@dataclass(frozen=True, slots=True)
class OperationPresentation:
  """Pure presentation model shared by both dashboard pages."""

  title: str
  detail: str
  tone: str
  show_banner: bool
  phase_detail: str | None = None
  meta: str | None = None
  compact_meta: str | None = None
  progress_fraction: float | None = None


@dataclass(frozen=True, slots=True)
class ProfileIdentity:
  artifact_sha256: str
  profile_sha256: str
  profile_revision: int


@dataclass(frozen=True, slots=True)
class LifecycleStatus:
  vehicle_identity: str
  runtime_identity_sha256: str
  source_openpilot_commit: str
  opendbc_commit: str
  activation_state_sha256: str | None
  diagnostic: str
  controller_state: str
  effective_controller: str
  production_envelope_verified: bool
  active_profile: ProfileIdentity | None
  staged_profile: ProfileIdentity | None
  rejected_profile_count: int

  @property
  def badge(self) -> str:
    return {
      "stock": "STOCK ACTIVE",
      "staged": "STOCK ACTIVE",
      "provisional": "BLATV2 PROVISIONAL",
      "approved": "BLATV2 APPROVED",
      "rollback_pending": "ROLLBACK PENDING | STOCK ACTIVE",
      "unavailable": "CONTROLLER STATUS UNAVAILABLE",
    }[self.controller_state]

  @property
  def lifecycle_position(self) -> int:
    return {
      "stock": 0,
      "staged": 2,
      "provisional": 3,
      "approved": 4,
      "rollback_pending": 0,
      "unavailable": -1,
    }[self.controller_state]


@dataclass(frozen=True, slots=True)
class GridCell:
  x: float
  y: float
  width: float
  height: float


def _exact_object(
  value: object,
  keys: frozenset[str],
  field: str,
) -> dict[str, object]:
  if type(value) is not dict or set(value) != keys:
    raise LearningStatusError("malformed", f"{field} keys do not match schema")
  return value


def select_value_provider(
  default_provider: Callable[[], bool],
  supplied_provider: Callable[[], bool] | None,
) -> Callable[[], bool]:
  """Select a callback without truth-testing the callback object itself."""
  return default_provider if supplied_provider is None else supplied_provider


def _bool(value: object, field: str) -> bool:
  if type(value) is not bool:
    raise LearningStatusError("malformed", f"{field} must be boolean")
  return value


def _integer(value: object, field: str, *, nullable: bool = False) -> int | None:
  if nullable and value is None:
    return None
  if type(value) is not int or value < 0:
    raise LearningStatusError(
      "malformed",
      f"{field} must be a non-negative integer",
    )
  return value


def _number(value: object, field: str, *, nullable: bool = False) -> float | None:
  if nullable and value is None:
    return None
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise LearningStatusError("malformed", f"{field} must be numeric")
  result = float(value)
  if not math.isfinite(result) or result < 0.0:
    raise LearningStatusError(
      "malformed",
      f"{field} must be finite and non-negative",
    )
  return result


def _signed_number(value: object, field: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise LearningStatusError("malformed", f"{field} must be numeric")
  result = float(value)
  if not math.isfinite(result):
    raise LearningStatusError("malformed", f"{field} must be finite")
  return 0.0 if result == 0.0 else result


def _nullable_signed_number(value: object, field: str) -> float | None:
  return None if value is None else _signed_number(value, field)


def _text(value: object, field: str) -> str:
  if type(value) is not str or not value.strip():
    raise LearningStatusError("malformed", f"{field} must be non-empty text")
  return value.strip()


def _nullable_text(value: object, field: str) -> str | None:
  if value is None:
    return None
  result = _text(value, field)
  if len(result) > 256:
    raise LearningStatusError("malformed", f"{field} exceeds 256 characters")
  return result


def _sha256(value: object, field: str, *, nullable: bool = False) -> str | None:
  if nullable and value is None:
    return None
  result = _text(value, field)
  if _SHA256_RE.fullmatch(result) is None:
    raise LearningStatusError("malformed", f"{field} must be lowercase SHA-256")
  return result


def _operation_id(value: object, field: str) -> str:
  result = _text(value, field)
  if _OPERATION_ID_RE.fullmatch(result) is None:
    raise LearningStatusError(
      "malformed",
      f"{field} must be 32 lowercase hexadecimal characters",
    )
  return result


def _commit(value: object, field: str) -> str:
  result = _text(value, field)
  if _GIT_COMMIT_RE.fullmatch(result) is None:
    raise LearningStatusError("malformed", f"{field} must be a full Git commit")
  return result


def _candidate_parameters(
  value: object,
  field: str,
) -> CandidateParameters | None:
  if value is None:
    return None
  payload = _exact_object(value, _PARAMETER_KEYS, field)
  parameters = CandidateParameters(
    torque_per_lateral_accel=_number(
      payload["torque_per_lateral_accel"],
      f"{field}.torque_per_lateral_accel",
    ),
    lateral_accel_offset_correction_mps2=_signed_number(
      payload["lateral_accel_offset_correction_mps2"],
      f"{field}.lateral_accel_offset_correction_mps2",
    ),
    kinetic_friction_torque=_number(
      payload["kinetic_friction_torque"],
      f"{field}.kinetic_friction_torque",
    ),
    static_breakaway_torque=_number(
      payload["static_breakaway_torque"],
      f"{field}.static_breakaway_torque",
    ),
  )
  if parameters.torque_per_lateral_accel <= 0.0:
    raise LearningStatusError(
      "malformed",
      f"{field}.torque_per_lateral_accel must be positive",
    )
  if parameters.kinetic_friction_torque > parameters.static_breakaway_torque:
    raise LearningStatusError(
      "malformed",
      f"{field} kinetic friction exceeds static breakaway",
    )
  return parameters


def _paired_loss(
  value: object,
  field: str,
  *,
  optional: bool,
) -> PairedLoss | None:
  if value is None:
    if optional:
      return None
    raise LearningStatusError("malformed", f"{field} must be present")
  payload = _exact_object(value, _PAIRED_LOSS_KEYS, field)
  result = PairedLoss(
    route_count=_integer(payload["route_count"], f"{field}.route_count"),
    mean_candidate_minus_seed_mse=_nullable_signed_number(
      payload["mean_candidate_minus_seed_mse"],
      f"{field}.mean_candidate_minus_seed_mse",
    ),
    numerical_tolerance_mse=_number(
      payload["numerical_tolerance_mse"],
      f"{field}.numerical_tolerance_mse",
      nullable=True,
    ),
    uncertainty_mse=_number(
      payload["uncertainty_mse"],
      f"{field}.uncertainty_mse",
      nullable=True,
    ),
    lower_bound_mse=_nullable_signed_number(
      payload["lower_bound_mse"],
      f"{field}.lower_bound_mse",
    ),
    upper_bound_mse=_nullable_signed_number(
      payload["upper_bound_mse"],
      f"{field}.upper_bound_mse",
    ),
  )
  values = (
    result.mean_candidate_minus_seed_mse,
    result.numerical_tolerance_mse,
    result.uncertainty_mse,
    result.lower_bound_mse,
    result.upper_bound_mse,
  )
  if result.route_count == 0:
    if any(item is not None for item in values):
      raise LearningStatusError("malformed", f"{field} empty loss has values")
  elif result.route_count == 1:
    if (
      result.mean_candidate_minus_seed_mse is None
      or result.numerical_tolerance_mse is None
      or any(item is not None for item in values[2:])
    ):
      raise LearningStatusError("malformed", f"{field} one-route loss is inconsistent")
  elif any(item is None for item in values):
    raise LearningStatusError("malformed", f"{field} multi-route loss is incomplete")
  else:
    assert result.mean_candidate_minus_seed_mse is not None
    assert result.uncertainty_mse is not None
    assert result.lower_bound_mse is not None
    assert result.upper_bound_mse is not None
    if (
      not math.isclose(
        result.lower_bound_mse,
        result.mean_candidate_minus_seed_mse - result.uncertainty_mse,
        rel_tol=1e-12,
        abs_tol=1e-12,
      )
      or not math.isclose(
        result.upper_bound_mse,
        result.mean_candidate_minus_seed_mse + result.uncertainty_mse,
        rel_tol=1e-12,
        abs_tol=1e-12,
      )
    ):
      raise LearningStatusError("malformed", f"{field} uncertainty bounds disagree")
  return result


def _fit_diagnostics(value: object, field: str) -> tuple[FitDiagnostic, ...]:
  if type(value) is not list or not value:
    raise LearningStatusError("malformed", f"{field} must be a non-empty list")
  result: list[FitDiagnostic] = []
  for position, item in enumerate(value):
    context = f"{field}[{position}]"
    payload = _exact_object(item, _FIT_DIAGNOSTIC_KEYS, context)
    model = _text(payload["model"], f"{context}.model")
    status = _text(payload["status"], f"{context}.status")
    if model not in _FIT_MODELS or any(existing.model == model for existing in result):
      raise LearningStatusError("malformed", f"{context}.model is invalid or duplicated")
    if status not in _FIT_STATUSES:
      raise LearningStatusError("malformed", f"{context}.status is invalid")
    diagnostic = FitDiagnostic(
      model=model,
      status=status,
      moving_rank=_integer(payload["moving_rank"], f"{context}.moving_rank"),
      moving_parameter_count=_integer(
        payload["moving_parameter_count"],
        f"{context}.moving_parameter_count",
      ),
      condition_estimate=_number(
        payload["condition_estimate"],
        f"{context}.condition_estimate",
        nullable=True,
      ),
      breakaway_rank=_integer(
        payload["breakaway_rank"],
        f"{context}.breakaway_rank",
      ),
      breakaway_parameter_count=_integer(
        payload["breakaway_parameter_count"],
        f"{context}.breakaway_parameter_count",
      ),
    )
    if (
      diagnostic.moving_rank > diagnostic.moving_parameter_count
      or diagnostic.breakaway_rank > diagnostic.breakaway_parameter_count
    ):
      raise LearningStatusError("malformed", f"{context} rank exceeds parameter count")
    full_rank = (
      diagnostic.moving_rank == diagnostic.moving_parameter_count
      and diagnostic.breakaway_rank == diagnostic.breakaway_parameter_count
    )
    if status == "identifiable" and not full_rank:
      raise LearningStatusError("malformed", f"{context} identifiable fit lacks full rank")
    if status == "rank_deficient" and full_rank:
      raise LearningStatusError("malformed", f"{context} rank-deficient fit is full rank")
    result.append(diagnostic)
  if {diagnostic.model for diagnostic in result} != _FIT_MODELS:
    raise LearningStatusError("malformed", f"{field} model family is incomplete")
  return tuple(result)


def _interpolation_status(
  value: object,
  position: int,
  nodes: tuple[LearningNodeStatus, ...],
) -> InterpolationStatus:
  field = f"interpolation_reports[{position}]"
  payload = _exact_object(value, _INTERPOLATION_KEYS, field)
  index = _integer(payload["interval_index"], f"{field}.interval_index")
  if index != position or position + 1 >= len(nodes):
    raise LearningStatusError("malformed", f"{field}.interval_index is invalid")
  lower = _number(payload["lower_speed_mps"], f"{field}.lower_speed_mps")
  upper = _number(payload["upper_speed_mps"], f"{field}.upper_speed_mps")
  if lower != nodes[position].speed_mps or upper != nodes[position + 1].speed_mps:
    raise LearningStatusError("malformed", f"{field} speed bounds disagree")
  raw_reasons = payload["reasons"]
  if type(raw_reasons) is not list or not raw_reasons:
    raise LearningStatusError("malformed", f"{field}.reasons must be non-empty")
  reasons = tuple(_text(reason, f"{field}.reasons") for reason in raw_reasons)
  if len(set(reasons)) != len(reasons) or any(reason not in _REASONS for reason in reasons):
    raise LearningStatusError("malformed", f"{field}.reasons are invalid")
  qualified = _bool(payload["qualified"], f"{field}.qualified")
  if qualified != (reasons == ("qualified",)):
    raise LearningStatusError("malformed", f"{field}.qualification reasons disagree")
  training = _paired_loss(
    payload["training_paired_loss"],
    f"{field}.training_paired_loss",
    optional=False,
  )
  validation = _paired_loss(
    payload["validation_paired_loss"],
    f"{field}.validation_paired_loss",
    optional=False,
  )
  assert training is not None and validation is not None
  return InterpolationStatus(
    interval_index=index,
    lower_speed_mps=lower,
    upper_speed_mps=upper,
    qualified=qualified,
    reasons=reasons,
    training_paired_loss=training,
    validation_paired_loss=validation,
  )


def _node(value: object, position: int) -> LearningNodeStatus:
  field = f"nodes[{position}]"
  payload = _exact_object(value, _NODE_KEYS, field)
  index = _integer(payload["node_index"], f"{field}.node_index")
  if index != position:
    raise LearningStatusError("malformed", "node indices must be contiguous")

  raw_reasons = payload["reasons"]
  if type(raw_reasons) is not list or not raw_reasons:
    raise LearningStatusError("malformed", f"{field}.reasons must be non-empty")
  reasons: list[str] = []
  for reason_position, raw_reason in enumerate(raw_reasons):
    reason = _text(raw_reason, f"{field}.reasons[{reason_position}]")
    if reason not in _REASONS or reason in reasons:
      raise LearningStatusError(
        "malformed",
        f"{field}.reasons contains an unknown or duplicate reason",
      )
    reasons.append(reason)

  qualified = _bool(payload["qualified"], f"{field}.qualified")
  if qualified != (tuple(reasons) in (("learned",), ("seed_retained",))):
    raise LearningStatusError(
      "malformed",
      f"{field} qualified flag and reasons disagree",
    )

  evaluation_status = _text(
    payload["evaluation_status"],
    f"{field}.evaluation_status",
  )
  if evaluation_status not in _EVALUATION_STATUSES:
    raise LearningStatusError(
      "malformed",
      f"{field}.evaluation_status is invalid",
    )
  fit_diagnostics = _fit_diagnostics(
    payload["fit_diagnostics"],
    f"{field}.fit_diagnostics",
  )
  fit_statuses = {diagnostic.status for diagnostic in fit_diagnostics}
  if evaluation_status == "rank_deficient" and "rank_deficient" not in fit_statuses:
    raise LearningStatusError(
      "malformed",
      f"{field}.rank-deficient status lacks fit evidence",
    )
  if evaluation_status == "ill_conditioned" and "ill_conditioned" not in fit_statuses:
    raise LearningStatusError(
      "malformed",
      f"{field}.ill-conditioned status lacks fit evidence",
    )

  training_outcome = payload["training_outcome"]
  if training_outcome not in (None, "learned", "seed_retained"):
    raise LearningStatusError(
      "malformed",
      f"{field}.training_outcome is invalid",
    )
  training_paired_loss = _paired_loss(
    payload["training_paired_loss"],
    f"{field}.training_paired_loss",
    optional=True,
  )
  validation_paired_loss = _paired_loss(
    payload["validation_paired_loss"],
    f"{field}.validation_paired_loss",
    optional=True,
  )
  if qualified and (
    evaluation_status != reasons[0] or training_outcome != reasons[0]
  ):
    raise LearningStatusError(
      "malformed",
      f"{field}.qualified outcome projection disagrees",
    )
  if not qualified and evaluation_status in ("learned", "seed_retained"):
    raise LearningStatusError(
      "malformed",
      f"{field}.failed node has a qualified status",
    )

  confidence = _number(payload["confidence"], f"{field}.confidence")
  if confidence > 1.0:
    raise LearningStatusError("malformed", f"{field}.confidence exceeds one")

  node = LearningNodeStatus(
    node_index=index,
    speed_mps=_number(payload["speed_mps"], f"{field}.speed_mps"),
    minimum_support_s=_number(
      payload["minimum_support_s"],
      f"{field}.minimum_support_s",
    ),
    clean_support_s=_number(
      payload["clean_support_s"],
      f"{field}.clean_support_s",
    ),
    last_drive_clean_support_s=_number(
      payload["last_drive_clean_support_s"],
      f"{field}.last_drive_clean_support_s",
      nullable=True,
    ),
    supported_sample_count=_integer(
      payload["supported_sample_count"],
      f"{field}.supported_sample_count",
    ),
    last_drive_accepted_sample_count=_integer(
      payload["last_drive_accepted_sample_count"],
      f"{field}.last_drive_accepted_sample_count",
      nullable=True,
    ),
    base_support_s=_number(
      payload["base_support_s"],
      f"{field}.base_support_s",
    ),
    base_sample_count=_integer(
      payload["base_sample_count"],
      f"{field}.base_sample_count",
    ),
    last_drive_base_support_s=_number(
      payload["last_drive_base_support_s"],
      f"{field}.last_drive_base_support_s",
      nullable=True,
    ),
    last_drive_base_sample_count=_integer(
      payload["last_drive_base_sample_count"],
      f"{field}.last_drive_base_sample_count",
      nullable=True,
    ),
    moving_support_s=_number(
      payload["moving_support_s"],
      f"{field}.moving_support_s",
    ),
    moving_sample_count=_integer(
      payload["moving_sample_count"],
      f"{field}.moving_sample_count",
    ),
    moving_training_count=_integer(
      payload["moving_training_count"],
      f"{field}.moving_training_count",
    ),
    moving_validation_count=_integer(
      payload["moving_validation_count"],
      f"{field}.moving_validation_count",
    ),
    last_drive_moving_support_s=_number(
      payload["last_drive_moving_support_s"],
      f"{field}.last_drive_moving_support_s",
      nullable=True,
    ),
    last_drive_moving_sample_count=_integer(
      payload["last_drive_moving_sample_count"],
      f"{field}.last_drive_moving_sample_count",
      nullable=True,
    ),
    breakaway_support_s=_number(
      payload["breakaway_support_s"],
      f"{field}.breakaway_support_s",
    ),
    breakaway_sample_count=_integer(
      payload["breakaway_sample_count"],
      f"{field}.breakaway_sample_count",
    ),
    breakaway_training_count=_integer(
      payload["breakaway_training_count"],
      f"{field}.breakaway_training_count",
    ),
    breakaway_validation_count=_integer(
      payload["breakaway_validation_count"],
      f"{field}.breakaway_validation_count",
    ),
    last_drive_breakaway_support_s=_number(
      payload["last_drive_breakaway_support_s"],
      f"{field}.last_drive_breakaway_support_s",
      nullable=True,
    ),
    last_drive_breakaway_sample_count=_integer(
      payload["last_drive_breakaway_sample_count"],
      f"{field}.last_drive_breakaway_sample_count",
      nullable=True,
    ),
    authority_support_s=_number(
      payload["authority_support_s"],
      f"{field}.authority_support_s",
    ),
    authority_sample_count=_integer(
      payload["authority_sample_count"],
      f"{field}.authority_sample_count",
    ),
    authority_fit_support_s=_number(
      payload["authority_fit_support_s"],
      f"{field}.authority_fit_support_s",
    ),
    authority_fit_sample_count=_integer(
      payload["authority_fit_sample_count"],
      f"{field}.authority_fit_sample_count",
    ),
    authority_training_count=_integer(
      payload["authority_training_count"],
      f"{field}.authority_training_count",
    ),
    authority_validation_count=_integer(
      payload["authority_validation_count"],
      f"{field}.authority_validation_count",
    ),
    last_drive_authority_support_s=_number(
      payload["last_drive_authority_support_s"],
      f"{field}.last_drive_authority_support_s",
      nullable=True,
    ),
    last_drive_authority_sample_count=_integer(
      payload["last_drive_authority_sample_count"],
      f"{field}.last_drive_authority_sample_count",
      nullable=True,
    ),
    last_drive_authority_fit_support_s=_number(
      payload["last_drive_authority_fit_support_s"],
      f"{field}.last_drive_authority_fit_support_s",
      nullable=True,
    ),
    last_drive_authority_fit_sample_count=_integer(
      payload["last_drive_authority_fit_sample_count"],
      f"{field}.last_drive_authority_fit_sample_count",
      nullable=True,
    ),
    training_count=_integer(
      payload["training_count"],
      f"{field}.training_count",
    ),
    validation_count=_integer(
      payload["validation_count"],
      f"{field}.validation_count",
    ),
    validation_support_s=_number(
      payload["validation_support_s"],
      f"{field}.validation_support_s",
    ),
    minimum_validation_support_s=_number(
      payload["minimum_validation_support_s"],
      f"{field}.minimum_validation_support_s",
    ),
    lateral_accel_span_mps2=_number(
      payload["lateral_accel_span_mps2"],
      f"{field}.lateral_accel_span_mps2",
    ),
    lateral_accel_rms_mps2=_number(
      payload["lateral_accel_rms_mps2"],
      f"{field}.lateral_accel_rms_mps2",
    ),
    rack_travel_deg=_number(
      payload["rack_travel_deg"],
      f"{field}.rack_travel_deg",
    ),
    applied_torque_span=_number(
      payload["applied_torque_span"],
      f"{field}.applied_torque_span",
    ),
    rack_reversals=_integer(
      payload["rack_reversals"],
      f"{field}.rack_reversals",
    ),
    lateral_accel_directions=_integer(
      payload["lateral_accel_directions"],
      f"{field}.lateral_accel_directions",
    ),
    applied_torque_directions=_integer(
      payload["applied_torque_directions"],
      f"{field}.applied_torque_directions",
    ),
    seed_validation_rms=_number(
      payload["seed_validation_rms"],
      f"{field}.seed_validation_rms",
      nullable=True,
    ),
    candidate_validation_rms=_number(
      payload["candidate_validation_rms"],
      f"{field}.candidate_validation_rms",
      nullable=True,
    ),
    moving_seed_validation_rms=_number(
      payload["moving_seed_validation_rms"],
      f"{field}.moving_seed_validation_rms",
      nullable=True,
    ),
    moving_candidate_validation_rms=_number(
      payload["moving_candidate_validation_rms"],
      f"{field}.moving_candidate_validation_rms",
      nullable=True,
    ),
    breakaway_seed_validation_rms=_number(
      payload["breakaway_seed_validation_rms"],
      f"{field}.breakaway_seed_validation_rms",
      nullable=True,
    ),
    breakaway_candidate_validation_rms=_number(
      payload["breakaway_candidate_validation_rms"],
      f"{field}.breakaway_candidate_validation_rms",
      nullable=True,
    ),
    authority_seed_validation_rms=_number(
      payload["authority_seed_validation_rms"],
      f"{field}.authority_seed_validation_rms",
      nullable=True,
    ),
    authority_candidate_validation_rms=_number(
      payload["authority_candidate_validation_rms"],
      f"{field}.authority_candidate_validation_rms",
      nullable=True,
    ),
    confidence=confidence,
    qualified=qualified,
    evaluation_status=evaluation_status,
    fit_diagnostics=fit_diagnostics,
    training_outcome=training_outcome,
    training_paired_loss=training_paired_loss,
    validation_paired_loss=validation_paired_loss,
    reasons=tuple(reasons),
    candidate_parameters=_candidate_parameters(
      payload["candidate_parameters"],
      f"{field}.candidate_parameters",
    ),
  )
  # Mirror the authoritative evidence and display-projection tolerance.
  # Long binary64 accumulations can differ from the sum of their mutually
  # exclusive populations by several ULPs without losing any evidence.
  if not math.isclose(
    node.clean_support_s,
    node.base_support_s + node.moving_support_s + node.breakaway_support_s,
    rel_tol=_SUPPORT_SUM_REL_TOL,
    abs_tol=_SUPPORT_SUM_ABS_TOL,
  ):
    raise LearningStatusError(
      "malformed",
      f"{field} clean support populations disagree",
    )
  if node.supported_sample_count != (
    node.base_sample_count
    + node.moving_sample_count
    + node.breakaway_sample_count
  ):
    raise LearningStatusError(
      "malformed",
      f"{field} clean sample populations disagree",
    )
  if (
    node.authority_fit_support_s > node.authority_support_s + 1e-12
    or node.authority_fit_sample_count > node.authority_sample_count
  ):
    raise LearningStatusError(
      "malformed",
      f"{field} authority fit exceeds authority evidence",
    )
  if node.qualified and node.candidate_parameters is None:
    raise LearningStatusError(
      "malformed",
      f"{field} qualified node lacks candidate parameters",
    )
  return node


def parse_learning_status(
  raw: object,
  *,
  expected_vehicle_identity: str | None,
) -> LearningStatus:
  """Decode the exact current display schema or fail closed.

  ``expected_vehicle_identity`` comes from the UI's already-decoded current
  CarParams. A missing expected identity is unavailable rather than permission
  to display a potentially stale snapshot from another vehicle.
  """
  if raw is None:
    raise LearningStatusError("absent", "Learning data is not available yet")
  if expected_vehicle_identity is None or not expected_vehicle_identity.strip():
    raise LearningStatusError(
      "vehicle_unavailable",
      "Current vehicle identity is unavailable",
    )
  if type(raw) is not dict:
    raise LearningStatusError(
      "malformed",
      "learning status must be a Params JSON object",
    )
  if (
    type(raw.get("schema_version")) is not int
    or raw["schema_version"] != LEARNING_STATUS_SCHEMA_VERSION
  ):
    raise LearningStatusError(
      "schema_mismatch",
      "Learning snapshot version is not supported",
    )
  data = _exact_object(raw, _TOP_LEVEL_KEYS, "learning status")
  if data["informational_only"] is not True:
    raise LearningStatusError(
      "malformed",
      "Learning snapshot is not marked display-only",
    )

  vehicle_identity = _text(data["vehicle_identity"], "vehicle_identity")
  if vehicle_identity != expected_vehicle_identity.strip():
    raise LearningStatusError(
      "wrong_vehicle",
      "Learning data belongs to a different vehicle",
    )

  raw_nodes = data["nodes"]
  if type(raw_nodes) is not list or not raw_nodes:
    raise LearningStatusError("malformed", "nodes must be a non-empty list")
  nodes = tuple(_node(value, index) for index, value in enumerate(raw_nodes))
  if any(
    right.speed_mps <= left.speed_mps
    for left, right in zip(nodes, nodes[1:], strict=False)
  ):
    raise LearningStatusError(
      "malformed",
      "node speeds must be strictly increasing",
    )

  last_drive_complete = _bool(
    data["last_drive_complete"],
    "last_drive_complete",
  )
  for node in nodes:
    delta_values = (
      node.last_drive_clean_support_s,
      node.last_drive_accepted_sample_count,
      node.last_drive_base_support_s,
      node.last_drive_base_sample_count,
      node.last_drive_moving_support_s,
      node.last_drive_moving_sample_count,
      node.last_drive_breakaway_support_s,
      node.last_drive_breakaway_sample_count,
      node.last_drive_authority_support_s,
      node.last_drive_authority_sample_count,
      node.last_drive_authority_fit_support_s,
      node.last_drive_authority_fit_sample_count,
    )
    node_complete = all(value is not None for value in delta_values)
    node_empty = all(value is None for value in delta_values)
    if (last_drive_complete and not node_complete) or (
      not last_drive_complete and not node_empty
    ):
      raise LearningStatusError(
        "malformed",
        "last-drive completeness and node deltas disagree",
      )
  all_nodes_qualified = _bool(
    data["all_nodes_qualified"],
    "all_nodes_qualified",
  )
  all_nodes_evaluated = _bool(
    data["all_nodes_evaluated"],
    "all_nodes_evaluated",
  )
  expected_nodes_evaluated = all(node.qualified for node in nodes)
  if all_nodes_evaluated != expected_nodes_evaluated:
    raise LearningStatusError(
      "malformed",
      "all_nodes_evaluated disagrees with node reports",
    )

  raw_interpolation_reports = data["interpolation_reports"]
  if type(raw_interpolation_reports) is not list:
    raise LearningStatusError(
      "malformed",
      "interpolation_reports must be a list",
    )
  if not all_nodes_evaluated and raw_interpolation_reports:
    raise LearningStatusError(
      "malformed",
      "unevaluated node grid carries interpolation reports",
    )
  interpolation_reports = tuple(
    _interpolation_status(value, index, nodes)
    for index, value in enumerate(raw_interpolation_reports)
  )
  expected_intervals_qualified = (
    len(interpolation_reports) == len(nodes) - 1
    and all(report.qualified for report in interpolation_reports)
  )
  all_intervals_qualified = _bool(
    data["all_intervals_qualified"],
    "all_intervals_qualified",
  )
  if all_intervals_qualified != expected_intervals_qualified:
    raise LearningStatusError(
      "malformed",
      "all_intervals_qualified disagrees with interval reports",
    )
  expected_all_qualified = all_nodes_evaluated and all_intervals_qualified
  if all_nodes_qualified != expected_all_qualified:
    raise LearningStatusError(
      "malformed",
      "all_nodes_qualified disagrees with node and interval reports",
    )

  candidate_sha = _sha256(
    data["candidate_profile_sha256"],
    "candidate_profile_sha256",
    nullable=True,
  )
  candidate_revision = _integer(
    data["candidate_profile_revision"],
    "candidate_profile_revision",
    nullable=True,
  )
  candidate_complete = candidate_sha is not None and candidate_revision is not None
  candidate_empty = candidate_sha is None and candidate_revision is None
  if not candidate_complete and not candidate_empty:
    raise LearningStatusError(
      "malformed",
      "candidate hash and revision completeness disagree",
    )
  candidate_profile_available = _bool(
    data["candidate_profile_available"],
    "candidate_profile_available",
  )
  if candidate_profile_available != candidate_complete:
    raise LearningStatusError(
      "malformed",
      "candidate availability and identity disagree",
    )
  if candidate_complete and not all_nodes_qualified:
    raise LearningStatusError(
      "malformed",
      "candidate exists before full qualification",
    )
  if (
    all_nodes_qualified
    and not candidate_complete
    and any(node.training_outcome == "learned" for node in nodes)
  ):
    raise LearningStatusError(
      "malformed",
      "qualified learned node lacks candidate artifact",
    )

  return LearningStatus(
    vehicle_identity=vehicle_identity,
    runtime_identity_sha256=_sha256(
      data["runtime_identity_sha256"],
      "runtime_identity_sha256",
    ),
    seed_profile_sha256=_sha256(
      data["seed_profile_sha256"],
      "seed_profile_sha256",
    ),
    evidence_sha256=_sha256(
      data["evidence_sha256"],
      "evidence_sha256",
    ),
    manifest_sha256=_sha256(
      data["manifest_sha256"],
      "manifest_sha256",
    ),
    all_nodes_evaluated=all_nodes_evaluated,
    all_intervals_qualified=all_intervals_qualified,
    all_nodes_qualified=all_nodes_qualified,
    candidate_profile_available=candidate_profile_available,
    candidate_profile_sha256=candidate_sha,
    candidate_profile_revision=candidate_revision,
    last_drive_complete=last_drive_complete,
    nodes=nodes,
    interpolation_reports=interpolation_reports,
  )


def parse_learning_operation_status(
  raw: object,
  *,
  expected_vehicle_identity: str | None,
  expected_runtime_identity_sha256: str | None,
  now_mono_ns: int,
) -> LearningOperationStatus:
  """Decode the learner's operational projection without inferring history.

  The status is rebuildable and display-only. Its monotonic timestamps share
  the current manager boot because the Params key clears on manager start.
  There is deliberately no UI-defined age timeout: the backend owns
  transitions, retry, and failure. A timestamp from a future monotonic epoch
  is rejected as stale rather than displayed indefinitely.
  """
  if raw is None:
    raise LearningStatusError(
      "operation_absent",
      "Learner operation status has not been published",
    )
  if type(now_mono_ns) is not int or now_mono_ns < 0:
    raise ValueError("now_mono_ns must be a non-negative integer")
  if type(raw) is not dict:
    raise LearningStatusError(
      "malformed",
      "learner operation status must be a Params JSON object",
    )
  if (
    type(raw.get("schema_version")) is not int
    or raw["schema_version"] != LEARNING_OPERATION_STATUS_SCHEMA_VERSION
  ):
    raise LearningStatusError(
      "schema_mismatch",
      "Learner operation status version is not supported",
    )
  data = _exact_object(raw, _OPERATION_KEYS, "learner operation status")
  if data["informational_only"] is not True:
    raise LearningStatusError(
      "malformed",
      "Learner operation status is not marked display-only",
    )

  state = _text(data["state"], "state")
  if state not in _OPERATION_STATE_DIAGNOSTICS:
    raise LearningStatusError("malformed", "operation state is not recognized")
  diagnostic = _text(data["diagnostic"], "diagnostic")
  if diagnostic not in _OPERATION_STATE_DIAGNOSTICS[state]:
    raise LearningStatusError(
      "malformed",
      "operation diagnostic does not match its state",
    )

  terminal = _bool(data["terminal"], "terminal")
  if terminal != (state in _TERMINAL_OPERATION_STATES):
    raise LearningStatusError(
      "malformed",
      "operation terminal flag does not match its state",
    )

  started_mono_ns = _integer(data["started_mono_ns"], "started_mono_ns")
  updated_mono_ns = _integer(data["updated_mono_ns"], "updated_mono_ns")
  if updated_mono_ns < started_mono_ns:
    raise LearningStatusError(
      "stale",
      "Learner operation timestamp precedes its start",
    )
  if updated_mono_ns > now_mono_ns:
    raise LearningStatusError(
      "stale",
      "Learner operation status belongs to another monotonic epoch",
    )

  vehicle_identity = _nullable_text(
    data["vehicle_identity"],
    "vehicle_identity",
  )
  identity_optional_states = frozenset(("preparing", "failed"))
  if vehicle_identity is None and state not in identity_optional_states:
    raise LearningStatusError(
      "malformed",
      "operation state requires a vehicle identity",
    )
  if vehicle_identity is not None:
    if expected_vehicle_identity is None or not expected_vehicle_identity.strip():
      raise LearningStatusError(
        "vehicle_unavailable",
        "Current vehicle identity is unavailable",
      )
    if vehicle_identity != expected_vehicle_identity.strip():
      raise LearningStatusError(
        "wrong_vehicle",
        "Learner operation belongs to a different vehicle",
      )

  runtime_identity = _sha256(
    data["runtime_identity_sha256"],
    "runtime_identity_sha256",
    nullable=True,
  )
  runtime_optional_states = frozenset((
    "preparing",
    "failed",
    "drive_skipped_identity_mismatch",
  ))
  if runtime_identity is None and state not in runtime_optional_states:
    raise LearningStatusError(
      "malformed",
      "operation state requires a runtime identity",
    )
  if (
    runtime_identity is not None
    and expected_runtime_identity_sha256 is not None
    and runtime_identity != expected_runtime_identity_sha256
  ):
    raise LearningStatusError(
      "runtime_mismatch",
      "Learning snapshot and operation describe different runtimes",
    )

  current_route_identity = _sha256(
    data["current_route_identity"],
    "current_route_identity",
    nullable=True,
  )
  current_route_index = _integer(
    data["current_route_index"],
    "current_route_index",
    nullable=True,
  )
  total_route_count = _integer(
    data["total_route_count"],
    "total_route_count",
    nullable=True,
  )
  index_pair_complete = (
    current_route_index is not None and total_route_count is not None
  )
  index_pair_empty = current_route_index is None and total_route_count is None
  if not (index_pair_complete or index_pair_empty):
    raise LearningStatusError(
      "malformed",
      "route index and total must both be present or absent",
    )
  if index_pair_complete and (
    current_route_index < 1
    or total_route_count < 1
    or current_route_index > total_route_count
  ):
    raise LearningStatusError(
      "malformed",
      "route progress is outside its one-based bounds",
    )
  if index_pair_complete and current_route_identity is None:
    raise LearningStatusError(
      "malformed",
      "indexed route progress lacks a route identity",
    )
  if state != "backfilling" and not index_pair_empty:
    raise LearningStatusError(
      "malformed",
      "only backfill operations may publish route progress",
    )
  if (
    current_route_identity is not None
    and state not in ("collecting", "backfilling")
  ):
    raise LearningStatusError(
      "malformed",
      "operation state cannot claim a current route",
    )
  if (
    state == "backfilling"
    and diagnostic == "replaying_route"
    and (not index_pair_complete or current_route_identity is None)
  ):
    raise LearningStatusError(
      "malformed",
      "route replay requires complete progress",
    )
  if (
    state == "backfilling"
    and diagnostic == "scanning_routes"
    and (not index_pair_empty or current_route_identity is not None)
  ):
    raise LearningStatusError(
      "malformed",
      "route scanning cannot claim replay progress",
    )

  evidence_sha256 = _sha256(
    data["evidence_sha256"],
    "evidence_sha256",
    nullable=True,
  )
  ledger_sha256 = _sha256(
    data["ledger_sha256"],
    "ledger_sha256",
    nullable=True,
  )
  if state == "idle" and evidence_sha256 is None:
    raise LearningStatusError(
      "malformed",
      "idle operation status lacks persisted evidence identity",
    )
  if state == "ready_no_evidence" and (
    evidence_sha256 is not None
    or ledger_sha256 is not None
    or data["accepted_sample_count"] != 0
    or data["rejected_sample_count"] != 0
  ):
    raise LearningStatusError(
      "malformed",
      "ready-no-evidence operation unexpectedly claims evidence",
    )
  if ledger_sha256 is not None and evidence_sha256 is None:
    raise LearningStatusError(
      "malformed",
      "backfill ledger identity lacks persisted evidence identity",
    )

  return LearningOperationStatus(
    state=state,
    diagnostic=diagnostic,
    operation_id=_operation_id(data["operation_id"], "operation_id"),
    sequence=_integer(data["sequence"], "sequence"),
    started_mono_ns=started_mono_ns,
    updated_mono_ns=updated_mono_ns,
    terminal=terminal,
    vehicle_identity=vehicle_identity,
    runtime_identity_sha256=runtime_identity,
    current_route_identity=current_route_identity,
    current_route_index=current_route_index,
    total_route_count=total_route_count,
    last_route_identity=_sha256(
      data["last_route_identity"],
      "last_route_identity",
      nullable=True,
    ),
    accepted_sample_count=_integer(
      data["accepted_sample_count"],
      "accepted_sample_count",
    ),
    rejected_sample_count=_integer(
      data["rejected_sample_count"],
      "rejected_sample_count",
    ),
    retry_count=_integer(data["retry_count"], "retry_count"),
    evidence_sha256=evidence_sha256,
    ledger_sha256=ledger_sha256,
  )


def parse_backfill_progress_status(
  raw: object,
  *,
  operation_status: LearningOperationStatus | None,
  now_mono_ns: int,
) -> BackfillProgressStatus:
  """Decode optional replay detail only when it binds to the current operation.

  This projection may improve display detail, but it is never evidence that
  learning, publication, or controller activation occurred. Any torn,
  malformed, or stale companion read is discarded by the caller in favor of
  the coarse operation-status presentation.
  """
  if raw is None:
    raise LearningStatusError(
      "progress_absent",
      "Detailed backfill progress has not been published",
    )
  if type(now_mono_ns) is not int or now_mono_ns < 0:
    raise ValueError("now_mono_ns must be a non-negative integer")
  if type(raw) is not dict:
    raise LearningStatusError(
      "malformed",
      "backfill progress must be a Params JSON object",
    )
  if (
    type(raw.get("schema_version")) is not int
    or raw["schema_version"] != BACKFILL_PROGRESS_SCHEMA_VERSION
  ):
    raise LearningStatusError(
      "schema_mismatch",
      "Backfill progress version is not supported",
    )
  data = _exact_object(raw, _BACKFILL_PROGRESS_KEYS, "backfill progress")
  if data["informational_only"] is not True:
    raise LearningStatusError(
      "malformed",
      "Backfill progress is not marked display-only",
    )

  operation_id = _operation_id(data["operation_id"], "operation_id")
  operation_sequence = _integer(
    data["operation_sequence"],
    "operation_sequence",
  )
  if (
    operation_status is None
    or operation_id != operation_status.operation_id
    or operation_sequence != operation_status.sequence
  ):
    raise LearningStatusError(
      "progress_mismatch",
      "Backfill progress does not match the current learner operation",
    )

  updated_mono_ns = _integer(data["updated_mono_ns"], "updated_mono_ns")
  if (
    updated_mono_ns < operation_status.started_mono_ns
    or updated_mono_ns < operation_status.updated_mono_ns
    or updated_mono_ns > now_mono_ns
  ):
    raise LearningStatusError(
      "stale",
      "Backfill progress timestamp is stale",
    )

  phase = _text(data["phase"], "phase")
  if phase not in _BACKFILL_PROGRESS_PHASES:
    raise LearningStatusError("malformed", "backfill phase is not recognized")

  pass_index = _integer(data["pass_index"], "pass_index")
  pass_count = _integer(data["pass_count"], "pass_count")
  if pass_count != 2 or pass_index < 1 or pass_index > pass_count:
    raise LearningStatusError(
      "malformed",
      "backfill pass progress is outside its bounds",
    )

  total_route_count = _integer(
    data["total_route_count"],
    "total_route_count",
  )
  if total_route_count < 1:
    raise LearningStatusError(
      "malformed",
      "backfill route total must be positive",
    )
  current_route_identity = _sha256(
    data["current_route_identity"],
    "current_route_identity",
    nullable=True,
  )
  current_route_index = _integer(
    data["current_route_index"],
    "current_route_index",
    nullable=True,
  )
  current_segment_index = _integer(
    data["current_segment_index"],
    "current_segment_index",
    nullable=True,
  )
  current_route_segment_count = _integer(
    data["current_route_segment_count"],
    "current_route_segment_count",
    nullable=True,
  )
  route_phase = phase in ("reading_segment", "applying_route")
  route_group_complete = (
    current_route_identity is not None
    and current_route_index is not None
  )
  segment_group_complete = (
    current_segment_index is not None
    and current_route_segment_count is not None
  )
  if route_phase:
    if not route_group_complete or not segment_group_complete:
      raise LearningStatusError(
        "malformed",
        "active replay progress requires route and segment coordinates",
      )
    if current_route_index < 1 or current_route_index > total_route_count:
      raise LearningStatusError(
        "malformed",
        "backfill route progress is outside its bounds",
      )
    if (
      current_segment_index < 1
      or current_route_segment_count < 1
      or current_segment_index > current_route_segment_count
    ):
      raise LearningStatusError(
        "malformed",
        "backfill segment progress is outside its bounds",
      )
    if (
      phase == "applying_route"
      and current_segment_index != current_route_segment_count
    ):
      raise LearningStatusError(
        "malformed",
        "route application must retain the final segment coordinate",
      )
  elif route_group_complete or segment_group_complete or any((
    current_route_identity is not None,
    current_route_index is not None,
    current_segment_index is not None,
    current_route_segment_count is not None,
  )):
    raise LearningStatusError(
      "malformed",
      "comparison and publication cannot claim a current route or segment",
    )

  completed_segments = _integer(
    data["completed_replay_segment_count"],
    "completed_replay_segment_count",
  )
  total_segments = _integer(
    data["total_replay_segment_count"],
    "total_replay_segment_count",
  )
  completed_work = _integer(
    data["completed_work_units"],
    "completed_work_units",
  )
  total_work = _integer(data["total_work_units"], "total_work_units")
  if (
    total_segments < 1
    or completed_segments > total_segments
    or total_work < 1
    or completed_work > total_work
  ):
    raise LearningStatusError(
      "malformed",
      "backfill cumulative progress is outside its bounds",
    )

  approximate_remaining_seconds = _integer(
    data["approximate_remaining_seconds"],
    "approximate_remaining_seconds",
    nullable=True,
  )
  if route_phase:
    if completed_work >= total_work:
      raise LearningStatusError(
        "malformed",
        "active replay progress cannot claim all work is complete",
      )
    expected_state = (
      ("backfilling", "replaying_route")
      if pass_index == 1
      else ("finalizing", "verifying_backfill")
    )
    if (operation_status.state, operation_status.diagnostic) != expected_state:
      raise LearningStatusError(
        "progress_mismatch",
        "Backfill phase does not match the learner operation state",
      )
    if pass_index == 1 and (
      current_route_identity != operation_status.current_route_identity
      or current_route_index != operation_status.current_route_index
      or total_route_count != operation_status.total_route_count
    ):
      raise LearningStatusError(
        "progress_mismatch",
        "Pass-one detail does not match coarse route progress",
      )
  else:
    expected_diagnostic = (
      "verifying_backfill" if phase == "comparing" else "publishing_backfill"
    )
    if (
      pass_index != 2
      or operation_status.state != "finalizing"
      or operation_status.diagnostic != expected_diagnostic
    ):
      raise LearningStatusError(
        "progress_mismatch",
        "Final backfill phase does not match the learner operation state",
      )
    if (
      completed_segments != total_segments
      or completed_work != total_work
      or approximate_remaining_seconds is not None
    ):
      raise LearningStatusError(
        "malformed",
        "final backfill phase must report complete replay work without an ETA",
      )

  return BackfillProgressStatus(
    operation_id=operation_id,
    operation_sequence=operation_sequence,
    sequence=_integer(data["sequence"], "sequence"),
    updated_mono_ns=updated_mono_ns,
    phase=phase,
    pass_index=pass_index,
    pass_count=pass_count,
    current_route_identity=current_route_identity,
    current_route_index=current_route_index,
    total_route_count=total_route_count,
    current_segment_index=current_segment_index,
    current_route_segment_count=current_route_segment_count,
    completed_replay_segment_count=completed_segments,
    total_replay_segment_count=total_segments,
    completed_work_units=completed_work,
    total_work_units=total_work,
    approximate_remaining_seconds=approximate_remaining_seconds,
  )


def validate_backfill_progress_update(
  previous: BackfillProgressStatus | None,
  current: BackfillProgressStatus,
) -> None:
  """Reject a regressed companion projection within one learner operation."""
  if previous is None or previous.operation_id != current.operation_id:
    return
  if current.operation_sequence < previous.operation_sequence:
    raise LearningStatusError(
      "stale",
      "Backfill progress operation sequence moved backward",
    )
  if current.sequence < previous.sequence:
    raise LearningStatusError(
      "stale",
      "Backfill progress sequence moved backward",
    )
  if current.sequence == previous.sequence and current != previous:
    raise LearningStatusError(
      "stale",
      "Backfill progress changed without advancing its sequence",
    )
  if (
    current.sequence > previous.sequence
    and current.updated_mono_ns < previous.updated_mono_ns
  ):
    raise LearningStatusError(
      "stale",
      "Backfill progress advanced with a stale timestamp",
    )
  if (
    current.pass_count != previous.pass_count
    or current.total_route_count != previous.total_route_count
    or current.total_replay_segment_count
    != previous.total_replay_segment_count
    or current.total_work_units != previous.total_work_units
  ):
    raise LearningStatusError(
      "stale",
      "Backfill progress totals changed during replay",
    )
  if (
    current.completed_replay_segment_count
    < previous.completed_replay_segment_count
    or current.completed_work_units < previous.completed_work_units
  ):
    raise LearningStatusError(
      "stale",
      "Backfill cumulative progress moved backward",
    )
  if current.pass_index < previous.pass_index:
    raise LearningStatusError(
      "stale",
      "Backfill pass progress moved backward",
    )
  if (
    current.pass_index == previous.pass_index
    and previous.current_route_index is not None
    and current.current_route_index is not None
  ):
    if current.current_route_index < previous.current_route_index:
      raise LearningStatusError(
        "stale",
        "Backfill route progress moved backward",
      )
    if current.current_route_index == previous.current_route_index:
      if (
        current.current_route_identity != previous.current_route_identity
        or current.current_route_segment_count
        != previous.current_route_segment_count
      ):
        raise LearningStatusError(
          "stale",
          "Backfill current-route identity or total changed",
        )
      if (
        previous.current_segment_index is not None
        and current.current_segment_index is not None
        and current.current_segment_index < previous.current_segment_index
      ):
        raise LearningStatusError(
          "stale",
          "Backfill segment progress moved backward",
        )
      if previous.phase == "applying_route" and current.phase == "reading_segment":
        raise LearningStatusError(
          "stale",
          "Backfill returned to reading an applied route",
        )
  if previous.phase == "comparing" and current.phase not in (
    "comparing",
    "publishing",
  ):
    raise LearningStatusError(
      "stale",
      "Backfill resumed replay after comparison",
    )
  if previous.phase == "publishing" and current.phase != "publishing":
    raise LearningStatusError(
      "stale",
      "Backfill resumed work after publication began",
    )


def validate_operation_update(
  previous: LearningOperationStatus | None,
  current: LearningOperationStatus,
) -> None:
  """Reject a regressed or internally contradictory sequenced operation."""
  if previous is None or previous.operation_id != current.operation_id:
    return
  if current.sequence < previous.sequence:
    raise LearningStatusError(
      "stale",
      "Learner operation sequence moved backward",
    )
  if current.sequence == previous.sequence and current != previous:
    raise LearningStatusError(
      "stale",
      "Learner operation changed without advancing its sequence",
    )
  if (
    current.sequence > previous.sequence
    and current.updated_mono_ns <= previous.updated_mono_ns
  ):
    raise LearningStatusError(
      "stale",
      "Learner operation advanced with a stale timestamp",
    )
  if (
    current.accepted_sample_count < previous.accepted_sample_count
    or current.rejected_sample_count < previous.rejected_sample_count
    or current.retry_count < previous.retry_count
  ):
    raise LearningStatusError(
      "stale",
      "Learner operation cumulative counts moved backward",
    )
  if (
    previous.total_route_count is not None
    and current.total_route_count is not None
    and current.total_route_count != previous.total_route_count
  ):
    raise LearningStatusError(
      "stale",
      "Learner operation route total changed during replay",
    )
  if (
    previous.current_route_index is not None
    and current.current_route_index is not None
    and current.current_route_index < previous.current_route_index
  ):
    raise LearningStatusError(
      "stale",
      "Learner operation route progress moved backward",
    )
  if previous.terminal and current.active:
    raise LearningStatusError(
      "stale",
      "Terminal learner operation resumed without a new identity",
    )


def parse_behavior_learning_status(
  raw: object,
  *,
  expected_vehicle_identity: str | None,
  expected_runtime_vehicle_identity_sha256: str | None,
  now_mono_ns: int,
) -> BehaviorLearningStatus:
  """Decode the exact behavior display schema without importing its owner.

  The status can explain qualification progress, but is never authority for
  controller staging or activation. Its monotonic timestamps are accepted
  only in the current manager epoch, and identity is checked whenever the
  physical dashboard has supplied the matching runtime identity.
  """
  if raw is None:
    raise LearningStatusError(
      "behavior_absent",
      "Behavior learning status has not been published",
    )
  if type(now_mono_ns) is not int or now_mono_ns < 0:
    raise ValueError("now_mono_ns must be a non-negative integer")
  if expected_vehicle_identity is None or not expected_vehicle_identity.strip():
    raise LearningStatusError(
      "vehicle_unavailable",
      "Current vehicle identity is unavailable",
    )
  if type(raw) is not dict:
    raise LearningStatusError(
      "malformed",
      "behavior learning status must be a Params JSON object",
    )
  if (
    type(raw.get("schemaVersion")) is not int
    or raw["schemaVersion"] != BEHAVIOR_LEARNING_STATUS_SCHEMA_VERSION
  ):
    raise LearningStatusError(
      "schema_mismatch",
      "Behavior learning status version is not supported",
    )
  data = _exact_object(raw, _BEHAVIOR_STATUS_KEYS, "behavior learning status")
  if data["informationalOnly"] is not True:
    raise LearningStatusError(
      "malformed",
      "Behavior learning status is not marked display-only",
    )

  vehicle_identity = _text(data["vehicleIdentity"], "vehicleIdentity")
  if vehicle_identity != expected_vehicle_identity.strip():
    raise LearningStatusError(
      "wrong_vehicle",
      "Behavior learning data belongs to a different vehicle",
    )
  runtime_identity = _sha256(
    data["runtimeVehicleIdentitySha256"],
    "runtimeVehicleIdentitySha256",
  )
  if (
    expected_runtime_vehicle_identity_sha256 is not None
    and runtime_identity != expected_runtime_vehicle_identity_sha256
  ):
    raise LearningStatusError(
      "runtime_mismatch",
      "Physical and behavior learning describe different runtimes",
    )

  state = _text(data["state"], "state")
  if state not in _BEHAVIOR_STATE_DIAGNOSTICS:
    raise LearningStatusError("malformed", "behavior state is not recognized")
  diagnostic = _text(data["diagnostic"], "diagnostic")
  if diagnostic not in _BEHAVIOR_STATE_DIAGNOSTICS[state]:
    raise LearningStatusError(
      "malformed",
      "behavior diagnostic does not match its state",
    )
  terminal = _bool(data["terminal"], "terminal")
  if terminal != (state in _BEHAVIOR_TERMINAL_STATES):
    raise LearningStatusError(
      "malformed",
      "behavior terminal flag does not match its state",
    )

  started_mono_ns = _integer(data["startedMonoNs"], "startedMonoNs")
  updated_mono_ns = _integer(data["updatedMonoNs"], "updatedMonoNs")
  if updated_mono_ns < started_mono_ns or updated_mono_ns > now_mono_ns:
    raise LearningStatusError(
      "stale",
      "Behavior learning status belongs to an invalid monotonic epoch",
    )

  counts = {
    name: _integer(data[key], key)
    for name, key in (
      ("eligible_route_count", "eligibleRouteCount"),
      ("required_route_count", "requiredRouteCount"),
      ("training_route_count", "trainingRouteCount"),
      ("validation_route_count", "validationRouteCount"),
      ("total_route_count", "totalRouteCount"),
      ("total_candidate_count", "totalCandidateCount"),
      ("completed_replay_jobs", "completedReplayJobs"),
      ("total_replay_jobs", "totalReplayJobs"),
    )
  }
  if (
    counts["training_route_count"] + counts["validation_route_count"]
    > counts["eligible_route_count"]
  ):
    raise LearningStatusError(
      "malformed",
      "behavior route partitions exceed homogeneous eligible evidence",
    )
  if counts["completed_replay_jobs"] > counts["total_replay_jobs"]:
    raise LearningStatusError(
      "malformed",
      "completed behavior replay jobs exceed the total",
    )

  current_route_identity = _nullable_text(
    data["currentRouteIdentity"],
    "currentRouteIdentity",
  )
  current_route_index = _integer(
    data["currentRouteIndex"],
    "currentRouteIndex",
    nullable=True,
  )
  if (current_route_identity is None) != (current_route_index is None):
    raise LearningStatusError(
      "malformed",
      "behavior route identity and index completeness disagree",
    )
  if current_route_index is not None and not (
    0 <= current_route_index < counts["total_route_count"]
  ):
    raise LearningStatusError(
      "malformed",
      "behavior route index is outside its zero-based bounds",
    )

  current_candidate_index = _integer(
    data["currentCandidateIndex"],
    "currentCandidateIndex",
    nullable=True,
  )
  if current_candidate_index is not None and not (
    0 <= current_candidate_index < counts["total_candidate_count"]
  ):
    raise LearningStatusError(
      "malformed",
      "behavior candidate index is outside its zero-based bounds",
    )

  gates = tuple(
    None if data[key] is None else _bool(data[key], key)
    for key in (
      "smoothPassed",
      "swiftPassed",
      "strongPassed",
      "targetMateriallyImproved",
    )
  )
  disposition = _nullable_text(
    data["qualificationDisposition"],
    "qualificationDisposition",
  )
  if disposition is not None and disposition not in _BEHAVIOR_DISPOSITIONS:
    raise LearningStatusError(
      "malformed",
      "behavior qualification disposition is not recognized",
    )
  raw_reasons = data["reasons"]
  if (
    type(raw_reasons) is not list
    or any(type(reason) is not str or not reason.strip() for reason in raw_reasons)
    or raw_reasons != sorted(set(raw_reasons))
  ):
    raise LearningStatusError(
      "malformed",
      "behavior reasons must be non-empty, unique, sorted text",
    )
  reasons = tuple(raw_reasons)

  transaction_sha256 = _sha256(
    data["transactionSha256"], "transactionSha256", nullable=True,
  )
  finalization_sha256 = _sha256(
    data["behaviorFinalizationSha256"],
    "behaviorFinalizationSha256",
    nullable=True,
  )
  selection_sha256 = _sha256(
    data["behaviorSelectionSha256"],
    "behaviorSelectionSha256",
    nullable=True,
  )
  policy_sha256 = _sha256(
    data["selectedBehaviorPolicySha256"],
    "selectedBehaviorPolicySha256",
    nullable=True,
  )
  if not terminal:
    if (
      any(value is not None for value in gates)
      or disposition is not None
      or reasons
      or any(value is not None for value in (
        transaction_sha256,
        finalization_sha256,
        selection_sha256,
        policy_sha256,
      ))
    ):
      raise LearningStatusError(
        "malformed",
        "active behavior status exposes terminal qualification data",
      )
  else:
    if disposition is None or not reasons:
      raise LearningStatusError(
        "malformed",
        "terminal behavior status lacks a disposition or reasons",
      )
    if state == "complete" and (
      any(type(value) is not bool for value in gates)
      or transaction_sha256 is None
      or finalization_sha256 is None
    ):
      raise LearningStatusError(
        "malformed",
        "completed behavior status lacks gate or transaction provenance",
      )
    qualified = disposition == "qualified_candidate_available"
    if qualified and (
      gates != (True, True, True, True)
      or state != "complete"
      or selection_sha256 is None
      or policy_sha256 is None
    ):
      raise LearningStatusError(
        "malformed",
        "qualified behavior candidate lacks passing gates or provenance",
      )
    if not qualified and (selection_sha256 is not None or policy_sha256 is not None):
      raise LearningStatusError(
        "malformed",
        "stock-retained behavior status exposes a selected policy",
      )

  return BehaviorLearningStatus(
    operation_id=_operation_id(data["operationId"], "operationId"),
    sequence=_integer(data["sequence"], "sequence"),
    state=state,
    diagnostic=diagnostic,
    terminal=terminal,
    started_mono_ns=started_mono_ns,
    updated_mono_ns=updated_mono_ns,
    vehicle_identity=vehicle_identity,
    runtime_vehicle_identity_sha256=runtime_identity,
    physical_generation_sha256=_sha256(
      data["physicalGenerationSha256"],
      "physicalGenerationSha256",
      nullable=True,
    ),
    physical_profile_sha256=_sha256(
      data["physicalProfileSha256"],
      "physicalProfileSha256",
      nullable=True,
    ),
    recorded_source_identity_sha256=_sha256(
      data["recordedSourceIdentitySha256"],
      "recordedSourceIdentitySha256",
      nullable=True,
    ),
    current_route_identity=current_route_identity,
    current_route_index=current_route_index,
    current_candidate_index=current_candidate_index,
    gate_spec_sha256=_sha256(data["gateSpecSha256"], "gateSpecSha256"),
    segmentation_config_sha256=_sha256(
      data["segmentationConfigSha256"],
      "segmentationConfigSha256",
    ),
    transaction_sha256=transaction_sha256,
    behavior_finalization_sha256=finalization_sha256,
    behavior_selection_sha256=selection_sha256,
    selected_behavior_policy_sha256=policy_sha256,
    smooth_passed=gates[0],
    swift_passed=gates[1],
    strong_passed=gates[2],
    target_materially_improved=gates[3],
    qualification_disposition=disposition,
    reasons=reasons,
    **counts,
  )


def validate_behavior_status_update(
  previous: BehaviorLearningStatus | None,
  current: BehaviorLearningStatus,
) -> None:
  """Reject torn or regressed snapshots within one immutable operation."""
  if previous is None or previous.operation_id != current.operation_id:
    return
  if current.sequence < previous.sequence:
    raise LearningStatusError("stale", "Behavior sequence moved backward")
  if current.sequence == previous.sequence and current != previous:
    raise LearningStatusError(
      "stale",
      "Behavior status changed without advancing its sequence",
    )
  if current.sequence > previous.sequence and (
    current.updated_mono_ns <= previous.updated_mono_ns
  ):
    raise LearningStatusError(
      "stale",
      "Behavior status advanced with a stale timestamp",
    )
  immutable_fields = (
    "started_mono_ns",
    "vehicle_identity",
    "runtime_vehicle_identity_sha256",
    "gate_spec_sha256",
    "segmentation_config_sha256",
  )
  if any(getattr(previous, name) != getattr(current, name) for name in immutable_fields):
    raise LearningStatusError(
      "stale",
      "Behavior operation identity or immutable inputs changed",
    )
  for name in (
    "physical_generation_sha256",
    "physical_profile_sha256",
    "recorded_source_identity_sha256",
  ):
    prior_value = getattr(previous, name)
    if prior_value is not None and getattr(current, name) != prior_value:
      raise LearningStatusError(
        "stale",
        "Established behavior provenance changed",
      )
  for name in (
    "eligible_route_count",
    "required_route_count",
    "training_route_count",
    "validation_route_count",
    "total_route_count",
    "total_candidate_count",
    "total_replay_jobs",
  ):
    prior_value = getattr(previous, name)
    current_value = getattr(current, name)
    if prior_value != 0 and current_value != prior_value:
      raise LearningStatusError(
        "stale",
        "Established behavior totals changed",
      )
  if current.completed_replay_jobs < previous.completed_replay_jobs:
    raise LearningStatusError("stale", "Behavior progress moved backward")
  state_order = {
    "waiting_for_physical_profile": 0,
    "waiting_for_routes": 1,
    "preparing": 2,
    "training": 3,
    "selecting": 4,
    "validating": 5,
    "publishing": 6,
    "complete": 7,
  }
  if (
    previous.state != "failed"
    and current.state != "failed"
    and state_order[current.state] < state_order[previous.state]
  ):
    raise LearningStatusError("stale", "Behavior state moved backward")
  if previous.terminal and current.sequence > previous.sequence:
    raise LearningStatusError(
      "stale",
      "Terminal behavior operation changed without a new identity",
    )


def behavior_presentation(
  status: BehaviorLearningStatus | None,
  *,
  error_code: str | None,
  error_message: str | None,
) -> BehaviorPresentation:
  """Describe behavior qualification without implying activation."""
  if status is None:
    return BehaviorPresentation(
      title="BEHAVIOR STATUS UNAVAILABLE",
      detail=error_message or "Awaiting behavior learner status",
      tone=(
        "gray"
        if error_code in (None, "behavior_absent", "vehicle_unavailable")
        else "red"
      ),
      progress_fraction=None,
    )
  if status.state == "waiting_for_physical_profile":
    return BehaviorPresentation(
      "BEHAVIOR LEARNING WAITING",
      "Physical calibration must qualify before behavior replay",
      "gray",
      None,
    )
  if status.state == "waiting_for_routes":
    return BehaviorPresentation(
      "WAITING FOR HOMOGENEOUS ROUTES",
      f"{status.eligible_route_count}/{status.required_route_count} compatible routes ready",
      "blue",
      status.route_readiness_fraction,
    )
  if status.state == "preparing":
    return BehaviorPresentation(
      "VALIDATING BEHAVIOR EVIDENCE",
      f"{status.eligible_route_count} homogeneous routes | training and validation remain separate",
      "blue",
      None,
    )
  if status.state == "training":
    return BehaviorPresentation(
      "TRAINING BEHAVIOR POLICY",
      f"{status.completed_replay_jobs}/{status.total_replay_jobs} replay jobs | candidate grid {status.total_candidate_count}",
      "blue",
      status.replay_progress_fraction,
    )
  if status.state == "selecting":
    return BehaviorPresentation(
      "SELECTING TRAINING WINNER",
      f"{status.completed_replay_jobs}/{status.total_replay_jobs} replay jobs | held-out routes remain unopened",
      "blue",
      status.replay_progress_fraction,
    )
  if status.state == "validating":
    return BehaviorPresentation(
      "VALIDATING FROZEN WINNER",
      f"{status.completed_replay_jobs}/{status.total_replay_jobs} replay jobs | {status.validation_route_count} held-out routes",
      "blue",
      status.replay_progress_fraction,
    )
  if status.state == "publishing":
    return BehaviorPresentation(
      "SAVING BEHAVIOR RESULT",
      "Publishing immutable informational qualification",
      "blue",
      status.replay_progress_fraction,
    )
  if status.state == "failed":
    return BehaviorPresentation(
      "BEHAVIOR LEARNING FAILED",
      status.diagnostic.replace("_", " ").upper(),
      "red",
      None,
    )
  if status.qualification_disposition == "qualified_candidate_available":
    return BehaviorPresentation(
      "BEHAVIOR CANDIDATE QUALIFIED",
      "Smooth PASS | Swift PASS | Strong PASS | informational only",
      "green",
      1.0,
    )
  return BehaviorPresentation(
    "STOCK BEHAVIOR RETAINED",
    "Behavior candidate did not clear every independent gate",
    "amber",
    1.0,
  )


def _profile_identity(value: object, field: str) -> ProfileIdentity | None:
  if value is None:
    return None
  payload = _exact_object(value, _PROFILE_IDENTITY_KEYS, field)
  return ProfileIdentity(
    artifact_sha256=_sha256(
      payload["artifact_sha256"],
      f"{field}.artifact_sha256",
    ),
    profile_sha256=_sha256(
      payload["profile_sha256"],
      f"{field}.profile_sha256",
    ),
    profile_revision=_integer(
      payload["profile_revision"],
      f"{field}.profile_revision",
    ),
  )


def parse_lifecycle_status(
  raw: object,
  *,
  expected_vehicle_identity: str | None,
  expected_runtime_identity_sha256: str | None,
) -> LifecycleStatus:
  """Decode the profiled-owned lifecycle projection, never raw authority."""
  if raw is None:
    raise LearningStatusError(
      "activation_absent",
      "Controller status is not available yet",
    )
  if expected_vehicle_identity is None or not expected_vehicle_identity.strip():
    raise LearningStatusError(
      "vehicle_unavailable",
      "Current vehicle identity is unavailable",
    )
  if type(raw) is not dict:
    raise LearningStatusError(
      "malformed",
      "lifecycle status must be a Params JSON object",
    )
  if (
    type(raw.get("schema_version")) is not int
    or raw["schema_version"] != LIFECYCLE_STATUS_SCHEMA_VERSION
  ):
    raise LearningStatusError(
      "schema_mismatch",
      "Controller status version is not supported",
    )
  data = _exact_object(raw, _LIFECYCLE_KEYS, "lifecycle status")
  if data["informational_only"] is not True:
    raise LearningStatusError(
      "malformed",
      "Controller status is not marked display-only",
    )

  vehicle_identity = _text(data["vehicle_identity"], "vehicle_identity")
  if vehicle_identity != expected_vehicle_identity.strip():
    raise LearningStatusError(
      "wrong_vehicle",
      "Controller status belongs to a different vehicle",
    )
  runtime_identity = _sha256(
    data["runtime_identity_sha256"],
    "runtime_identity_sha256",
  )
  if (
    expected_runtime_identity_sha256 is not None
    and runtime_identity != expected_runtime_identity_sha256
  ):
    raise LearningStatusError(
      "runtime_mismatch",
      "Learning and controller snapshots describe different runtimes",
    )

  diagnostic = _text(data["diagnostic"], "diagnostic")
  if diagnostic not in _ARTIFACT_DIAGNOSTICS:
    raise LearningStatusError("malformed", "diagnostic is not recognized")
  controller_state = _text(data["controller_state"], "controller_state")
  if controller_state not in _CONTROLLER_STATES:
    raise LearningStatusError("malformed", "controller_state is not recognized")
  effective_controller = _text(
    data["effective_controller"],
    "effective_controller",
  )
  if effective_controller not in _EFFECTIVE_CONTROLLERS:
    raise LearningStatusError(
      "malformed",
      "effective_controller is not recognized",
    )

  active_profile = _profile_identity(data["active_profile"], "active_profile")
  staged_profile = _profile_identity(data["staged_profile"], "staged_profile")
  production_verified = _bool(
    data["production_envelope_verified"],
    "production_envelope_verified",
  )
  active_states = frozenset(("provisional", "approved", "rollback_pending"))
  if (controller_state in active_states) != (active_profile is not None):
    raise LearningStatusError(
      "malformed",
      "controller state and active profile disagree",
    )
  if controller_state == "stock" and staged_profile is not None:
    raise LearningStatusError(
      "malformed",
      "stock lifecycle unexpectedly contains a staged profile",
    )
  if controller_state == "staged" and staged_profile is None:
    raise LearningStatusError(
      "malformed",
      "staged lifecycle lacks a staged profile",
    )
  expected_effective = (
    "modular"
    if controller_state in ("provisional", "approved")
    else "stock"
  )
  if effective_controller != expected_effective:
    raise LearningStatusError(
      "malformed",
      "controller state and effective controller disagree",
    )
  if controller_state in ("provisional", "approved") and (
    diagnostic != "ok" or not production_verified
  ):
    raise LearningStatusError(
      "malformed",
      "active modular lifecycle is not validated",
    )
  if controller_state in ("staged", "rollback_pending") and (
    diagnostic != "ok" or not production_verified
  ):
    raise LearningStatusError(
      "malformed",
      "profile-bearing stock lifecycle is not validated",
    )
  if controller_state == "unavailable" and diagnostic in ("ok", "absent"):
    raise LearningStatusError(
      "malformed",
      "unavailable lifecycle lacks a failure diagnostic",
    )
  activation_state_sha256 = _sha256(
    data["activation_state_sha256"],
    "activation_state_sha256",
    nullable=True,
  )
  if diagnostic in ("ok", "absent") and activation_state_sha256 is None:
    raise LearningStatusError(
      "malformed",
      "validated lifecycle lacks its activation-state identity",
    )

  return LifecycleStatus(
    vehicle_identity=vehicle_identity,
    runtime_identity_sha256=runtime_identity,
    source_openpilot_commit=_commit(
      data["source_openpilot_commit"],
      "source_openpilot_commit",
    ),
    opendbc_commit=_commit(data["opendbc_commit"], "opendbc_commit"),
    activation_state_sha256=activation_state_sha256,
    diagnostic=diagnostic,
    controller_state=controller_state,
    effective_controller=effective_controller,
    production_envelope_verified=production_verified,
    active_profile=active_profile,
    staged_profile=staged_profile,
    rejected_profile_count=_integer(
      data["rejected_profile_count"],
      "rejected_profile_count",
    ),
  )


def operation_presentation(
  status: LearningOperationStatus | None,
  *,
  error_code: str | None,
  error_message: str | None,
  has_learning_snapshot: bool,
  backfill_progress: BackfillProgressStatus | None = None,
) -> OperationPresentation:
  """Return truthful copy without treating absence as an empty history."""
  if status is None:
    unavailable = error_message or "Awaiting the learner's current status"
    if error_code == "operation_absent":
      unavailable = "Awaiting learner status; drive history is unknown"
    return OperationPresentation(
      title="LEARNER STATUS UNAVAILABLE",
      detail=unavailable,
      tone=("gray" if error_code in (None, "operation_absent", "vehicle_unavailable") else "red"),
      show_banner=has_learning_snapshot,
    )

  if backfill_progress is not None and (
    backfill_progress.operation_id != status.operation_id
    or backfill_progress.operation_sequence != status.sequence
  ):
    backfill_progress = None

  prior = "Prior snapshot shown | " if has_learning_snapshot and status.active else ""
  counts = " | ".join((
    f"{status.accepted_sample_count:,} incorporated",
    f"{status.rejected_sample_count:,} excluded",
  ))
  diagnostic = _OPERATION_DIAGNOSTIC_LABELS[status.diagnostic]

  if backfill_progress is not None:
    progress = backfill_progress
    if progress.phase in ("reading_segment", "applying_route"):
      title = (
        "PROCESSING PRIOR ROUTES"
        if progress.pass_index == 1
        else "VERIFYING PRIOR ROUTES"
      )
      detail = " | ".join((
        f"Pass {progress.pass_index}/{progress.pass_count}",
        f"Route {progress.current_route_index}/{progress.total_route_count}",
        f"Segment {progress.current_segment_index}/{progress.current_route_segment_count}",
      ))
      phase_detail = (
        "Reading and validating this route segment"
        if progress.phase == "reading_segment"
        else "Applying validated route evidence"
      )
      if has_learning_snapshot:
        phase_detail += " | Prior snapshot shown"
      if progress.approximate_remaining_seconds is None:
        compact_meta = "Estimating time"
      else:
        percentage = int(progress.progress_fraction * 100.0)
        minutes = (
          progress.approximate_remaining_seconds + 59
        ) // 60
        compact_meta = f"{percentage}% | About {minutes} min left"
      return OperationPresentation(
        title=title,
        detail=detail,
        tone="blue",
        show_banner=has_learning_snapshot,
        phase_detail=phase_detail,
        meta=f"{compact_meta} | {counts}",
        compact_meta=compact_meta,
        progress_fraction=progress.progress_fraction,
      )
    if progress.phase == "comparing":
      return OperationPresentation(
        title="COMPARING REPLAY PASSES",
        detail="Checking that both independent replay passes match exactly",
        tone="blue",
        show_banner=has_learning_snapshot,
      )
    return OperationPresentation(
      title="SAVING VERIFIED LEARNING DATA",
      detail="Replay passes matched; publishing the verified snapshot",
      tone="blue",
      show_banner=has_learning_snapshot,
    )

  if status.state == "preparing":
    return OperationPresentation(
      title="PREPARING LEARNER",
      detail=f"{prior}{diagnostic}",
      tone="blue",
      show_banner=has_learning_snapshot,
    )
  if status.state == "ready_no_evidence":
    return OperationPresentation(
      title="READY FOR FIRST DRIVE",
      detail="Complete one drive to begin collecting clean support",
      tone="blue",
      show_banner=has_learning_snapshot,
    )
  if status.state == "collecting":
    return OperationPresentation(
      title="COLLECTING THIS DRIVE",
      detail=f"{prior}{counts}",
      tone="blue",
      show_banner=has_learning_snapshot,
    )
  if status.state == "finalizing":
    return OperationPresentation(
      title="FINALIZING LEARNING DATA",
      detail=f"{prior}{diagnostic}",
      tone="blue",
      show_banner=has_learning_snapshot,
    )
  if status.state == "retry_pending":
    retry = f"Retry {status.retry_count} pending"
    return OperationPresentation(
      title="SAVE RETRY PENDING",
      detail=f"{prior}{retry} | {counts}",
      tone="amber",
      show_banner=has_learning_snapshot,
    )
  if status.state == "backfilling":
    if status.diagnostic == "scanning_routes":
      progress = "Scanning compatible routes"
    else:
      progress = f"Route {status.current_route_index}/{status.total_route_count}"
    return OperationPresentation(
      title="PROCESSING PRIOR ROUTES",
      detail=f"{prior}{progress} | {counts}",
      tone="blue",
      show_banner=has_learning_snapshot,
    )
  if status.state == "idle":
    return OperationPresentation(
      title="LEARNER READY",
      detail=diagnostic,
      tone="green",
      show_banner=has_learning_snapshot,
    )
  if status.state == "drive_skipped_identity_mismatch":
    return OperationPresentation(
      title="DRIVE SKIPPED | NEXT DRIVE READY",
      detail=diagnostic,
      tone="amber",
      show_banner=has_learning_snapshot,
    )
  return OperationPresentation(
    title="LEARNER FAILED",
    detail=diagnostic,
    tone="red",
    show_banner=has_learning_snapshot,
  )


def learning_panel_presentation(
  status: LearningOperationStatus | None,
  *,
  operation_error_code: str | None,
  operation_error_message: str | None,
  learning_error_code: str | None,
  learning_error_message: str | None,
  has_learning_snapshot: bool,
  backfill_progress: BackfillProgressStatus | None = None,
) -> OperationPresentation:
  """Plan the banner or empty state without discarding a valid snapshot."""
  operation = operation_presentation(
    status,
    error_code=operation_error_code,
    error_message=operation_error_message,
    has_learning_snapshot=has_learning_snapshot,
    backfill_progress=backfill_progress,
  )
  if has_learning_snapshot:
    return operation

  # An explicit ready_no_evidence status is the only authority for first-drive
  # copy. Missing caches reveal nothing about drive history.
  if status is not None and status.state != "idle":
    return operation
  if learning_error_code not in (None, "absent", "vehicle_unavailable"):
    return OperationPresentation(
      title="LEARNING DATA UNAVAILABLE",
      detail=(learning_error_message or "Validated current-build data could not be read"),
      tone="red",
      show_banner=False,
    )
  if status is not None and status.state == "idle":
    return OperationPresentation(
      title="LEARNING SNAPSHOT UNAVAILABLE",
      detail=(learning_error_message or "Learner is ready, but its validated snapshot is unavailable"),
      tone="red",
      show_banner=False,
    )
  return operation


def reason_label(reason: str) -> str:
  return _REASON_LABELS.get(reason, "Status unavailable")


def format_duration(seconds: float) -> str:
  total = max(0, int(round(seconds)))
  hours, remainder = divmod(total, 3600)
  minutes, secs = divmod(remainder, 60)
  if hours:
    return f"{hours:d}h {minutes:02d}m"
  return f"{minutes:d}:{secs:02d}"


def format_speed(speed_mps: float, *, metric: bool) -> str:
  converted = speed_mps * (3.6 if metric else 2.2369362920544)
  return f"{converted:.0f} {'km/h' if metric else 'mph'}"


def grid_cells(
  width: float,
  height: float,
  count: int,
  *,
  columns: int = 2,
  gap: float = 14.0,
) -> tuple[GridCell, ...]:
  if count <= 0 or columns <= 0 or width <= 0.0 or height <= 0.0:
    return ()
  rows = (count + columns - 1) // columns
  cell_width = (width - gap * (columns - 1)) / columns
  cell_height = (height - gap * (rows - 1)) / rows
  if cell_width <= 0.0 or cell_height <= 0.0:
    return ()
  return tuple(
    GridCell(
      x=(index % columns) * (cell_width + gap),
      y=(index // columns) * (cell_height + gap),
      width=cell_width,
      height=cell_height,
    )
    for index in range(count)
  )


def cycle_page_index(current: int, direction: int, page_count: int) -> int:
  if page_count <= 0:
    raise ValueError("page_count must be positive")
  if direction not in (-1, 1):
    raise ValueError("direction must be -1 or 1")
  if not 0 <= current < page_count:
    raise ValueError("current page is outside the carousel")
  return (current + direction) % page_count

"""Pure transaction that turns frozen route evidence into a behavior policy.

This is the only orchestration layer between the shared route-evidence
artifact and :func:`finalize_behavior_learning`.  It deliberately owns no
decoder, controller implementation, persistence, Params write, or activation
path.  The caller injects a narrow decoder and three replay cores; this module
then enforces the invariants that are easy to lose when those pieces are
invoked independently:

* a fully-qualified, immutable physical calibration is frozen first;
* all routes have one exact recorded controller/source identity;
* the model's scalar-anchored path is the sole target (lane lines do not exist
  in this API);
* reference and rack mapping are evaluated at every 100-Hz witness with the
  live timing and the frozen physical profile;
* segmentation runs once from that immutable target and logger events are
  locator context only;
* every controller receives the same whole-route timeline and phase windows;
* driver contact censors the response after contact, but never votes on
  quality; and
* no policy is returned for activation unless training and held-out
  Smooth/Swift/Strong gates all pass.

The numerical replay callback is intentionally opaque here.  Production can
bind it to the shared harness artifact while tests can provide a small pure
in-memory core.  Its output is still checked frame-for-frame before scoring.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
import ctypes
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import math
import multiprocessing
import os
import re
import signal
import threading
import time
from typing import Any
from typing import Protocol

from openpilot.selfdrive.controls.lib.blatv2.behavior_coordinator import (
  BehaviorGateSpec,
  BehaviorLearningFinalization,
  BehaviorRouteEvidenceIdentity,
  ReplayArtifactIdentity,
  ReplayCoreIdentity,
  ReplayRole,
  finalize_behavior_learning,
  partition_whole_routes,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_configuration import (
  load_behavior_gate_spec,
  load_behavior_segmentation_config,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import (
  BehaviorControlResponse,
  BehaviorReferenceAtControl,
  BehaviorSample,
  BehaviorSourceIdentity,
  BehaviorWindow,
  EventLocator,
  SparseModelBehaviorIntent,
  assemble_behavior_sample,
  canonical_json,
  derive_behavior_reference,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_metrics import (
  BehaviorMetricName,
  BehaviorScorecard,
  score_behavior,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_policy import (
  BehaviorPolicy,
  PolicyEvaluation,
  PolicyMetric,
  build_candidate_grid,
  select_training_winner,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_segmentation import (
  SegmentationConfig,
  SegmentationResult,
  segment_behavior_route,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_profile import (
  VehicleCalibrationProfile,
)
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import RackMappingSnapshot


BEHAVIOR_TRANSACTION_SCHEMA_VERSION = 2
MAX_BEHAVIOR_REPLAY_WORKERS = 4
BEHAVIOR_WORKER_STARTUP_TIMEOUT_S = 2.0
BEHAVIOR_WORKER_POLL_INTERVAL_S = 0.05
BEHAVIOR_WORKER_TERM_TIMEOUT_S = 0.5
BEHAVIOR_WORKER_KILL_TIMEOUT_S = 0.5
BEHAVIOR_WORKER_RESULT_EXIT_TIMEOUT_S = 2.0
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PR_SET_PDEATHSIG = 1


class BehaviorTransactionError(RuntimeError):
  """The proposed transaction is not safe or reproducible enough to score."""


class BehaviorTransactionAborted(BehaviorTransactionError):
  """Offroad ownership ended before the replay transaction completed."""


class BehaviorReplayProgressPhase(StrEnum):
  """Display-only phase of the immutable replay workload."""

  TRAINING = "training"
  VALIDATION = "validation"


@dataclass(frozen=True, slots=True)
class BehaviorReplayProgress:
  """Parent-process progress over the complete, pre-counted replay workload.

  The total includes every training role/grid candidate and exactly the three
  held-out roles (stock, incumbent, and the eventually frozen winner).  Roles
  count independently even when stock and incumbent use the same core bytes.
  A transaction with no training winner stops before the held-out portion, so
  its final ``completed_jobs`` intentionally remains below ``total_jobs``.
  """

  phase: BehaviorReplayProgressPhase
  completed_jobs: int
  total_jobs: int
  phase_completed_jobs: int
  phase_total_jobs: int

  def __post_init__(self) -> None:
    if not 0 <= self.completed_jobs <= self.total_jobs:
      raise ValueError("replay progress completed count is outside its total")
    if not 0 <= self.phase_completed_jobs <= self.phase_total_jobs:
      raise ValueError("replay phase completed count is outside its total")


type BehaviorReplayProgressCallback = Callable[[BehaviorReplayProgress], None]


@dataclass(frozen=True, slots=True)
class CanonicalBehaviorControlInput:
  """Controller-independent input at one canonical control witness.

  ``core_input`` is the decoder-owned, canonical byte payload consumed by the
  injected numerical replay artifact.  Keeping it opaque prevents this
  orchestration module from growing a second rlog or controller decoder.
  """

  mono_time_ns: int
  route_time_s: float
  speed_mps: float
  model_publication_index: int | None
  live_rack_mapping: RackMappingSnapshot | None
  nominal_rack_mapping: RackMappingSnapshot
  core_input: bytes
  inputs_valid: bool
  lateral_active: bool
  steering_pressed: bool
  platform_fault: bool
  driver_intervention_onset: bool

  def __post_init__(self) -> None:
    if self.mono_time_ns < 0:
      raise ValueError("control timestamp must be non-negative")
    if self.model_publication_index is not None and (
      type(self.model_publication_index) is not int
      or self.model_publication_index < 0
    ):
      raise ValueError("linked model index must be non-negative or None")
    if not math.isfinite(self.route_time_s) or self.route_time_s < 0.0:
      raise ValueError("route time must be finite and non-negative")
    if not math.isfinite(self.speed_mps) or self.speed_mps < 0.0:
      raise ValueError("speed must be finite and non-negative")
    if not isinstance(self.nominal_rack_mapping, RackMappingSnapshot):
      raise TypeError("nominal rack mapping must be a RackMappingSnapshot")
    if self.live_rack_mapping is not None and not isinstance(
      self.live_rack_mapping,
      RackMappingSnapshot,
    ):
      raise TypeError("live rack mapping must be a RackMappingSnapshot or None")
    if type(self.core_input) is not bytes:
      raise TypeError("core_input must be immutable canonical bytes")


@dataclass(frozen=True, slots=True)
class DecodedBehaviorRoute:
  """Narrow decoder output over one already-shared route-evidence artifact."""

  route_id: str
  route_evidence_sha256: str
  vehicle_identity: str
  recorded_source: BehaviorSourceIdentity
  model_publications: tuple[SparseModelBehaviorIntent, ...]
  control_inputs: tuple[CanonicalBehaviorControlInput, ...]
  event_locators: tuple[EventLocator, ...]

  def __post_init__(self) -> None:
    if not self.route_id.strip() or not self.vehicle_identity.strip():
      raise ValueError("route and vehicle identities must not be empty")
    if _SHA256_RE.fullmatch(self.route_evidence_sha256) is None:
      raise ValueError("route evidence identity must be lowercase SHA-256")
    if not self.model_publications or not self.control_inputs:
      raise ValueError("behavior replay requires model and control evidence")
    if any(
      right.publication_mono_time_ns <= left.publication_mono_time_ns
      for left, right in zip(
        self.model_publications,
        self.model_publications[1:],
        strict=False,
      )
    ):
      raise ValueError("model publications must be strictly time ordered")
    if any(
      right.mono_time_ns <= left.mono_time_ns
      or right.route_time_s <= left.route_time_s
      for left, right in zip(self.control_inputs, self.control_inputs[1:], strict=False)
    ):
      raise ValueError("control inputs must be strictly time ordered")
    for control in self.control_inputs:
      if control.model_publication_index is None:
        continue
      if control.model_publication_index >= len(self.model_publications):
        raise ValueError("control input references a missing model publication")
      model = self.model_publications[control.model_publication_index]
      if model.publication_mono_time_ns > control.mono_time_ns:
        raise ValueError("control input references a future model publication")
    event_keys = tuple(
      (event.occurred_mono_time_ns, event.event_type, event.severity)
      for event in self.event_locators
    )
    if event_keys != tuple(sorted(event_keys)):
      raise ValueError("event locators must be in canonical timestamp order")

  @property
  def identity(self) -> BehaviorRouteEvidenceIdentity:
    return BehaviorRouteEvidenceIdentity(
      route_id=self.route_id,
      route_evidence_sha256=self.route_evidence_sha256,
      recorded_source=self.recorded_source,
    )


class RouteEvidenceDecoder(Protocol):
  """Decode one shared artifact without duplicating its binary format here."""

  def __call__(
    self,
    artifact: object,
    physical_profile: VehicleCalibrationProfile,
  ) -> DecodedBehaviorRoute: ...


@dataclass(frozen=True, slots=True)
class ControllerFrameOutput:
  """Counterfactual response produced by one exact numerical core."""

  mono_time_ns: int
  measured_curvature_1pm: float
  measured_rack_angle_deg: float
  measured_rack_rate_deg_s: float
  measured_rack_accel_deg_s2: float
  raw_requested_torque: float
  envelope_applied_torque: float
  torque_headroom: float
  actuator_constrained: bool
  controller_fault: bool
  response_eligible: bool

  def __post_init__(self) -> None:
    if self.mono_time_ns < 0:
      raise ValueError("controller output timestamp must be non-negative")
    values = (
      self.measured_curvature_1pm,
      self.measured_rack_angle_deg,
      self.measured_rack_rate_deg_s,
      self.measured_rack_accel_deg_s2,
      self.raw_requested_torque,
      self.envelope_applied_torque,
      self.torque_headroom,
    )
    if not all(math.isfinite(value) for value in values):
      raise ValueError("controller output values must be finite")
    if self.torque_headroom < 0.0:
      raise ValueError("controller output headroom must be non-negative")


@dataclass(frozen=True, slots=True)
class ControllerReplayRequest:
  """One whole route presented to one fresh replay-core invocation."""

  artifact_identity: ReplayArtifactIdentity
  policy: BehaviorPolicy | None
  route: DecodedBehaviorRoute
  references: tuple[BehaviorReferenceAtControl, ...]
  physical_profile: VehicleCalibrationProfile


type ControllerReplayCallback = Callable[
  [ControllerReplayRequest],
  Iterable[ControllerFrameOutput],
]


@dataclass(frozen=True, slots=True)
class BehaviorReplayCore:
  """Exact core identity paired with a whole-route, reset-on-call replay."""

  identity: ReplayCoreIdentity
  replay_route: ControllerReplayCallback

  def __post_init__(self) -> None:
    if not callable(self.replay_route):
      raise TypeError("replay_route must be callable")


class QualificationDisposition(StrEnum):
  """Qualification result only; this enum has no activation semantics."""

  STOCK_RETAINED = "stock_retained"
  QUALIFIED_CANDIDATE_AVAILABLE = "qualified_candidate_available"


@dataclass(frozen=True, slots=True)
class BehaviorLearningTransactionResult:
  """Immutable report only; it has no persistence or activation methods."""

  schema_version: int
  physical_profile_sha256: str
  route_evidence_sha256s: tuple[tuple[str, str], ...]
  segmentation_config_sha256: str
  segmentation_sha256s: tuple[tuple[str, str], ...]
  evaluations: tuple[PolicyEvaluation, ...]
  finalization: BehaviorLearningFinalization
  qualification_disposition: QualificationDisposition

  def __post_init__(self) -> None:
    if self.schema_version != BEHAVIOR_TRANSACTION_SCHEMA_VERSION:
      raise ValueError("behavior transaction schema is incompatible")
    for value in (
      self.physical_profile_sha256,
      self.segmentation_config_sha256,
      *(digest for _, digest in self.route_evidence_sha256s),
      *(digest for _, digest in self.segmentation_sha256s),
    ):
      if _SHA256_RE.fullmatch(value) is None:
        raise ValueError("transaction identities must be lowercase SHA-256")
    route_ids = tuple(route_id for route_id, _ in self.route_evidence_sha256s)
    segmentation_routes = tuple(route_id for route_id, _ in self.segmentation_sha256s)
    if not route_ids or route_ids != tuple(sorted(set(route_ids))):
      raise ValueError("transaction routes must be non-empty, unique, and sorted")
    if segmentation_routes != route_ids:
      raise ValueError("every route must have exactly one segmentation identity")
    expected = (
      QualificationDisposition.QUALIFIED_CANDIDATE_AVAILABLE
      if self.finalization.passed
      else QualificationDisposition.STOCK_RETAINED
    )
    if self.qualification_disposition is not expected:
      raise ValueError("qualification disposition disagrees with finalization")
    evaluation_keys = tuple(
      (evaluation.artifact_identity, evaluation.route_ids)
      for evaluation in self.evaluations
    )
    if evaluation_keys != tuple(sorted(set(evaluation_keys))):
      raise ValueError("transaction evaluations must be unique and canonical")

  @property
  def stock_retained(self) -> bool:
    """Whether qualification retained stock; actual activation lives elsewhere."""
    return self.qualification_disposition is QualificationDisposition.STOCK_RETAINED

  @property
  def selected_policy(self) -> BehaviorPolicy | None:
    return self.finalization.final_behavior_policy

  def to_dict(self) -> dict[str, object]:
    return {
      "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
      "finalization": self.finalization.to_dict(),
      "physicalProfileSha256": self.physical_profile_sha256,
      "qualificationDisposition": self.qualification_disposition.value,
      "routeEvidence": [
        {"routeId": route_id, "sha256": digest}
        for route_id, digest in self.route_evidence_sha256s
      ],
      "schemaVersion": self.schema_version,
      "segmentationConfigSha256": self.segmentation_config_sha256,
      "segmentations": [
        {"routeId": route_id, "sha256": digest}
        for route_id, digest in self.segmentation_sha256s
      ],
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())

  @property
  def sha256(self) -> str:
    return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

  @classmethod
  def from_json(cls, encoded: str) -> BehaviorLearningTransactionResult:
    """Strictly restore an immutable published transaction document."""
    try:
      payload = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
      raise ValueError("behavior transaction is invalid JSON") from exc
    keys = frozenset((
      "evaluations",
      "finalization",
      "physicalProfileSha256",
      "qualificationDisposition",
      "routeEvidence",
      "schemaVersion",
      "segmentationConfigSha256",
      "segmentations",
    ))
    if type(payload) is not dict or frozenset(payload) != keys:
      raise ValueError("behavior transaction keys do not match schema")
    if type(payload["schemaVersion"]) is not int:
      raise ValueError("behavior transaction schema version must be integer")
    if type(payload["qualificationDisposition"]) is not str:
      raise ValueError("behavior qualification disposition must be text")
    for key in ("physicalProfileSha256", "segmentationConfigSha256"):
      if type(payload[key]) is not str:
        raise ValueError(f"{key} must be text")
    if type(payload["evaluations"]) is not list:
      raise ValueError("behavior transaction evaluations must be an array")

    def identity_pairs(name: str, id_key: str) -> tuple[tuple[str, str], ...]:
      values = payload[name]
      if type(values) is not list:
        raise ValueError(f"behavior transaction {name} must be an array")
      result: list[tuple[str, str]] = []
      expected = frozenset((id_key, "sha256"))
      for value in values:
        if type(value) is not dict or frozenset(value) != expected:
          raise ValueError(f"behavior transaction {name} item keys do not match schema")
        if type(value[id_key]) is not str or type(value["sha256"]) is not str:
          raise ValueError(f"behavior transaction {name} item values must be text")
        result.append((value[id_key], value["sha256"]))
      return tuple(result)

    if type(payload["finalization"]) is not dict:
      raise ValueError("behavior transaction finalization must be an object")
    result = cls(
      schema_version=payload["schemaVersion"],
      physical_profile_sha256=payload["physicalProfileSha256"],
      route_evidence_sha256s=identity_pairs("routeEvidence", "routeId"),
      segmentation_config_sha256=payload["segmentationConfigSha256"],
      segmentation_sha256s=identity_pairs("segmentations", "routeId"),
      evaluations=tuple(
        PolicyEvaluation.from_dict(value)
        for value in payload["evaluations"]
      ),
      finalization=BehaviorLearningFinalization.from_json(
        canonical_json(payload["finalization"]),
      ),
      qualification_disposition=QualificationDisposition(
        payload["qualificationDisposition"],
      ),
    )
    if result.to_json() != encoded:
      raise ValueError("behavior transaction JSON is not canonical")
    return result


@dataclass(frozen=True, slots=True)
class _PreparedRoute:
  route: DecodedBehaviorRoute
  references: tuple[BehaviorReferenceAtControl, ...]
  segmentation: SegmentationResult


@dataclass(frozen=True, slots=True)
class _ReplayJob:
  identity: ReplayArtifactIdentity
  policy: BehaviorPolicy | None
  core: BehaviorReplayCore
  route: _PreparedRoute

  @property
  def key(self) -> tuple[str, str, str]:
    return (
      self.identity.to_json(),
      "" if self.policy is None else self.policy.sha256,
      self.route.route.route_id,
    )


def _neutral_response(
  control: CanonicalBehaviorControlInput,
  physical_profile: VehicleCalibrationProfile,
) -> BehaviorControlResponse:
  parameters = physical_profile.parameters_at(control.speed_mps).parameters
  return BehaviorControlResponse(
    mono_time_ns=control.mono_time_ns,
    route_time_s=control.route_time_s,
    speed_mps=control.speed_mps,
    transport_delay_s=parameters.transport_delay_s,
    live_rack_mapping=control.live_rack_mapping,
    nominal_rack_mapping=control.nominal_rack_mapping,
    measured_curvature_1pm=0.0,
    measured_rack_angle_deg=0.0,
    measured_rack_rate_deg_s=0.0,
    measured_rack_accel_deg_s2=0.0,
    raw_requested_torque=0.0,
    envelope_applied_torque=0.0,
    torque_headroom=1.0,
    actuator_constrained=False,
    lateral_active=control.lateral_active,
    inputs_valid=control.inputs_valid and parameters.qualified,
    steering_pressed=control.steering_pressed,
    controller_fault=control.platform_fault,
    driver_intervention_onset=control.driver_intervention_onset,
  )


def _prepare_route(
  route: DecodedBehaviorRoute,
  physical_profile: VehicleCalibrationProfile,
  segmentation_config: SegmentationConfig,
  abort_requested: Callable[[], bool],
) -> _PreparedRoute:
  references: list[BehaviorReferenceAtControl] = []
  target_samples: list[BehaviorSample] = []
  for control_index, control in enumerate(route.control_inputs):
    if control_index % 256 == 0:
      _check_replay_abort(abort_requested)
    neutral = _neutral_response(control, physical_profile)
    if control.model_publication_index is None:
      # Startup witnesses preceding the first model publication are retained
      # as lifecycle context.  This explicitly invalid sentinel cannot enter
      # segmentation or metrics, but lets every replay core see the exact
      # inactive-to-active history without fabricating a plan link.
      reference = BehaviorReferenceAtControl(
        model_publication_mono_time_ns=0,
        plan_time_now_s=0.0,
        physical_effect_plan_s=neutral.transport_delay_s,
        scalar_curvature_1pm=0.0,
        anchored_curvature_1pm=0.0,
        anchored_curvature_rate_1pm_s=0.0,
        anchored_curvature_accel_1pm_s2=0.0,
        desired_rack_angle_deg=0.0,
        desired_rack_rate_deg_s=0.0,
        desired_rack_accel_deg_s2=0.0,
        valid=False,
      )
    else:
      reference = derive_behavior_reference(
        route.model_publications[control.model_publication_index],
        neutral,
      )
    references.append(reference)
    # Measured fields equal the immutable target only for segmentation.  They
    # are discarded before scoring and therefore cannot make the target vote
    # for a controller.  Actual response is attached after the one shared
    # segmentation has been frozen.
    target_samples.append(assemble_behavior_sample(
      reference,
      BehaviorControlResponse(
        mono_time_ns=neutral.mono_time_ns,
        route_time_s=neutral.route_time_s,
        speed_mps=neutral.speed_mps,
        transport_delay_s=neutral.transport_delay_s,
        live_rack_mapping=neutral.live_rack_mapping,
        nominal_rack_mapping=neutral.nominal_rack_mapping,
        measured_curvature_1pm=reference.anchored_curvature_1pm,
        measured_rack_angle_deg=reference.desired_rack_angle_deg,
        measured_rack_rate_deg_s=reference.desired_rack_rate_deg_s,
        measured_rack_accel_deg_s2=reference.desired_rack_accel_deg_s2,
        raw_requested_torque=0.0,
        envelope_applied_torque=0.0,
        torque_headroom=1.0,
        actuator_constrained=False,
        lateral_active=neutral.lateral_active,
        inputs_valid=neutral.inputs_valid,
        steering_pressed=neutral.steering_pressed,
        controller_fault=neutral.controller_fault,
        driver_intervention_onset=neutral.driver_intervention_onset,
      ),
    ))
  _check_replay_abort(abort_requested)
  segmentation = segment_behavior_route(
    route.route_id,
    route.recorded_source,
    target_samples,
    route.event_locators,
    segmentation_config,
  )
  _check_replay_abort(abort_requested)
  return _PreparedRoute(route, tuple(references), segmentation)


def _response_source(
  identity: ReplayArtifactIdentity,
  evidence_schema_version: int,
) -> BehaviorSourceIdentity:
  return BehaviorSourceIdentity(
    controller_name=f"{identity.role.value}:{identity.core.controller_name}",
    controller_artifact_sha256=identity.composed_controller_artifact_sha256,
    source_openpilot_commit=identity.core.source_openpilot_commit,
    opendbc_commit=identity.core.opendbc_commit,
    panda_commit=identity.core.panda_commit,
    evidence_schema_version=evidence_schema_version,
  )


def _check_replay_abort(abort_requested: Callable[[], bool]) -> None:
  requested = abort_requested()
  if type(requested) is not bool:
    raise BehaviorTransactionError("behavior replay abort guard must return bool")
  if requested:
    raise BehaviorTransactionAborted("behavior replay ownership ended")


def _run_replay_job(
  job: _ReplayJob,
  physical_profile: VehicleCalibrationProfile,
  abort_requested: Callable[[], bool],
) -> tuple[
  tuple[str, str, str],
  tuple[BehaviorWindow, ...],
]:
  _check_replay_abort(abort_requested)
  request = ControllerReplayRequest(
    artifact_identity=job.identity,
    policy=job.policy,
    route=job.route.route,
    references=job.route.references,
    physical_profile=physical_profile,
  )
  outputs_list: list[ControllerFrameOutput] = []
  for output_index, output in enumerate(job.core.replay_route(request)):
    # Poll between bounded frame chunks as well as whole jobs. This lets the
    # owned child cooperate before parent-enforced group teardown and keeps a
    # direct test/harness invocation cancellable as well.
    if output_index % 256 == 0:
      _check_replay_abort(abort_requested)
    outputs_list.append(output)
  _check_replay_abort(abort_requested)
  outputs = tuple(outputs_list)
  controls = job.route.route.control_inputs
  if len(outputs) != len(controls):
    raise BehaviorTransactionError("controller replay output count mismatch")
  response_samples: list[BehaviorSample] = []
  for control, reference, output in zip(
    controls,
    job.route.references,
    outputs,
    strict=True,
  ):
    if not isinstance(output, ControllerFrameOutput):
      raise BehaviorTransactionError("controller replay emitted an incompatible frame")
    if output.mono_time_ns != control.mono_time_ns:
      raise BehaviorTransactionError("controller replay output timeline mismatch")
    neutral = _neutral_response(control, physical_profile)
    response_samples.append(assemble_behavior_sample(
      reference,
      BehaviorControlResponse(
        mono_time_ns=control.mono_time_ns,
        route_time_s=control.route_time_s,
        speed_mps=control.speed_mps,
        transport_delay_s=neutral.transport_delay_s,
        live_rack_mapping=control.live_rack_mapping,
        nominal_rack_mapping=control.nominal_rack_mapping,
        measured_curvature_1pm=output.measured_curvature_1pm,
        measured_rack_angle_deg=output.measured_rack_angle_deg,
        measured_rack_rate_deg_s=output.measured_rack_rate_deg_s,
        measured_rack_accel_deg_s2=output.measured_rack_accel_deg_s2,
        raw_requested_torque=output.raw_requested_torque,
        envelope_applied_torque=output.envelope_applied_torque,
        torque_headroom=output.torque_headroom,
        actuator_constrained=output.actuator_constrained,
        lateral_active=control.lateral_active,
        inputs_valid=neutral.inputs_valid and output.response_eligible,
        steering_pressed=control.steering_pressed,
        controller_fault=control.platform_fault or output.controller_fault,
        driver_intervention_onset=control.driver_intervention_onset,
      ),
    ))
  source = _response_source(
    job.identity,
    job.route.route.recorded_source.evidence_schema_version,
  )
  windows = tuple(
    BehaviorWindow(
      route_id=segmented.window.route_id,
      window_id=segmented.window.window_id,
      source=source,
      maneuver_class=segmented.window.maneuver_class,
      phase=segmented.window.phase,
      samples=tuple(response_samples[
        segmented.start_sample_index:segmented.end_sample_index_exclusive
      ]),
      event_locators=segmented.window.event_locators,
    )
    for segmented in job.route.segmentation.windows
  )
  return job.key, windows


_FORK_REPLAY_JOBS: tuple[_ReplayJob, ...] | None = None
_FORK_PHYSICAL_PROFILE: VehicleCalibrationProfile | None = None
_FORK_RUNNER_LOCK = threading.Lock()


def _run_inherited_replay_job(
  index: int,
  abort_requested: Callable[[], bool],
) -> tuple[
  tuple[str, str, str],
  tuple[BehaviorWindow, ...],
]:
  """Run one canonical job inherited by a forked worker.

  Only an integer crosses the task queue.  Replay callbacks and route bytes
  remain read-only, copy-on-write inherited objects, so local closures are
  not silently pickled into a different execution contract.
  """
  jobs = _FORK_REPLAY_JOBS
  profile = _FORK_PHYSICAL_PROFILE
  if jobs is None or profile is None or not 0 <= index < len(jobs):
    raise BehaviorTransactionError("fork replay worker was not initialized")
  return _run_replay_job(jobs[index], profile, abort_requested)


def _establish_owned_process_group(expected_parent_pid: int) -> None:
  """Own one process group and arm parent-death teardown for its whole tree."""

  worker_pid = os.getpid()
  os.setsid()
  if os.getsid(0) != worker_pid or os.getpgrp() != worker_pid:
    raise BehaviorTransactionError("behavior replay worker failed process-group isolation")

  def terminate_owned_group(_signal_number: int, _frame: object) -> None:
    # Reset first so the worker receives the group SIGKILL instead of
    # recursively re-entering this handler. The verified group contains only
    # this replay worker and descendants it may create in the future.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
      os.killpg(worker_pid, signal.SIGKILL)
    finally:
      os._exit(128 + signal.SIGTERM)

  signal.signal(signal.SIGTERM, terminate_owned_group)
  libc = ctypes.CDLL(None, use_errno=True)
  if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
    error_number = ctypes.get_errno()
    raise OSError(error_number, os.strerror(error_number))
  if os.getppid() != expected_parent_pid:
    terminate_owned_group(signal.SIGTERM, None)


def _behavior_replay_worker_entry(
  connection: Any,
  job_index: int,
  cancel_requested: Any,
  start_requested: Any,
  expected_parent_pid: int,
  inherited_close_fds: tuple[int, ...],
) -> None:
  """Run one replay job in a parent-owned Linux process group."""

  def child_abort_requested() -> bool:
    return bool(cancel_requested.is_set()) or os.getppid() != expected_parent_pid

  try:
    for descriptor in inherited_close_fds:
      os.close(descriptor)
    _establish_owned_process_group(expected_parent_pid)
    connection.send(("ready", os.getpid(), os.getsid(0), os.getpgrp()))
    while not start_requested.wait(timeout=BEHAVIOR_WORKER_POLL_INTERVAL_S):
      _check_replay_abort(child_abort_requested)
    _check_replay_abort(child_abort_requested)
    result = _run_inherited_replay_job(job_index, child_abort_requested)
    connection.send(("ok", job_index, result))
  except BaseException as exc:
    try:
      connection.send(("error", job_index, type(exc).__name__, str(exc)))
    except (BrokenPipeError, EOFError, OSError):
      pass
  finally:
    connection.close()


class _BehaviorReplayWorker:
  """One exact replay job with verified process-group ownership."""

  def __init__(
    self,
    *,
    context: Any,
    job_index: int,
    abort_requested: Callable[[], bool],
    inherited_close_fds: tuple[int, ...],
  ) -> None:
    receive = None
    send = None
    self.job_index = job_index
    self._abort_requested = abort_requested
    self._closed = False
    self._group_ready = False
    self._started = False
    try:
      receive, send = context.Pipe(duplex=False)
      cancel_requested = context.Event()
      start_requested = context.Event()
      process = context.Process(
        target=_behavior_replay_worker_entry,
        args=(
          send,
          job_index,
          cancel_requested,
          start_requested,
          os.getpid(),
          (*inherited_close_fds, receive.fileno()),
        ),
        name=f"blatv2-behavior-{job_index}",
      )
      self._receive = receive
      self._cancel_requested = cancel_requested
      self._start_requested = start_requested
      self._process = process
      process.start()
      self._started = True
      send.close()
      deadline = time.monotonic() + BEHAVIOR_WORKER_STARTUP_TIMEOUT_S
      while not receive.poll(BEHAVIOR_WORKER_POLL_INTERVAL_S):
        _check_replay_abort(abort_requested)
        if not process.is_alive():
          raise BehaviorTransactionError("behavior replay worker exited during startup")
        if time.monotonic() >= deadline:
          raise BehaviorTransactionError("behavior replay worker startup timed out")
      ready = receive.recv()
      if (
        type(ready) is not tuple
        or len(ready) != 4
        or ready[0] != "ready"
        or ready[1] != process.pid
        or ready[2] != process.pid
        or ready[3] != process.pid
      ):
        raise BehaviorTransactionError(
          "behavior replay worker did not establish owned process-group isolation",
        )
      self._group_ready = True
      _check_replay_abort(abort_requested)
      start_requested.set()
    except BaseException:
      if send is not None:
        try:
          send.close()
        except OSError:
          pass
      if hasattr(self, "_process"):
        self.cancel()
      elif receive is not None:
        receive.close()
      raise

  @property
  def receive_fd(self) -> int:
    return self._receive.fileno()

  def _signal_owned_group(self, signal_number: int) -> None:
    if not self._started:
      return
    if not self._group_ready:
      if self._process.is_alive():
        if signal_number == signal.SIGTERM:
          self._process.terminate()
        else:
          self._process.kill()
      return
    try:
      os.killpg(self._process.pid, signal_number)
    except ProcessLookupError:
      pass

  def cancel(self) -> None:
    """Immediately terminate this exact owned group, then reap its leader."""
    if self._closed:
      return
    self._cancel_requested.set()
    if self._started:
      self._signal_owned_group(signal.SIGTERM)
      self._process.join(timeout=BEHAVIOR_WORKER_TERM_TIMEOUT_S)
      if self._process.is_alive():
        self._signal_owned_group(signal.SIGKILL)
        self._process.join(timeout=BEHAVIOR_WORKER_KILL_TIMEOUT_S)
      if self._process.is_alive():
        raise BehaviorTransactionError("behavior replay worker could not be reaped")
    self._receive.close()
    self._closed = True

  def poll_result(self) -> object | None:
    if self._closed:
      raise BehaviorTransactionError("behavior replay worker is already closed")
    if not self._receive.poll(0.0):
      if not self._process.is_alive():
        raise BehaviorTransactionError("behavior replay worker exited without a result")
      return None
    try:
      payload = self._receive.recv()
    except (EOFError, OSError) as exc:
      raise BehaviorTransactionError("behavior replay worker result channel failed") from exc
    self._process.join(timeout=BEHAVIOR_WORKER_RESULT_EXIT_TIMEOUT_S)
    if self._process.is_alive():
      raise BehaviorTransactionError("behavior replay worker did not exit after result")
    if self._process.exitcode != 0:
      raise BehaviorTransactionError("behavior replay worker exited abnormally")
    self._receive.close()
    self._closed = True
    return payload


def _decode_worker_payload(
  payload: object,
  expected_index: int,
) -> tuple[tuple[str, str, str], tuple[BehaviorWindow, ...]]:
  if (
    type(payload) is tuple
    and len(payload) == 3
    and payload[0] == "ok"
    and payload[1] == expected_index
    and type(payload[2]) is tuple
  ):
    return payload[2]
  if (
    type(payload) is tuple
    and len(payload) == 4
    and payload[0] == "error"
    and payload[1] == expected_index
    and type(payload[2]) is str
    and type(payload[3]) is str
  ):
    raise BehaviorTransactionError(
      f"behavior replay worker failed: {payload[2]}: {payload[3]}",
    )
  raise BehaviorTransactionError("behavior replay worker returned malformed output")


def _run_replay_jobs(
  jobs: tuple[_ReplayJob, ...],
  physical_profile: VehicleCalibrationProfile,
  worker_count: int,
  abort_requested: Callable[[], bool],
  result_collected: Callable[[int], None] | None = None,
) -> tuple[tuple[tuple[str, str, str], tuple[BehaviorWindow, ...]], ...]:
  """Run a canonical batch inline or with bounded Linux ``fork`` workers.

  ``fork`` is intentional: the production callback registry and immutable
  route artifacts are inherited without serialization.  A spawn-only host
  cannot honestly execute that same artifact and therefore fails closed.
  The offroad caller must invoke this only after its other worker threads have
  joined. Every worker count, including one, uses the same owned child so an
  eager whole-route replay remains promptly cancellable. Results are buffered
  and reduced in canonical index order regardless of completion order.
  """
  _check_replay_abort(abort_requested)
  if not jobs:
    return ()
  if worker_count > MAX_BEHAVIOR_REPLAY_WORKERS:
    raise ValueError(
      f"worker_count exceeds bounded maximum {MAX_BEHAVIOR_REPLAY_WORKERS}",
    )
  if "fork" not in multiprocessing.get_all_start_methods():
    raise BehaviorTransactionError(
      "parallel behavior replay requires fork; no spawn fallback is permitted",
    )
  if threading.active_count() != 1:
    raise BehaviorTransactionError(
      "behavior replay cannot fork from a multithreaded process",
    )

  global _FORK_REPLAY_JOBS, _FORK_PHYSICAL_PROFILE
  with _FORK_RUNNER_LOCK:
    if _FORK_REPLAY_JOBS is not None or _FORK_PHYSICAL_PROFILE is not None:
      raise BehaviorTransactionError("parallel behavior replay is already active")
    _FORK_REPLAY_JOBS = jobs
    _FORK_PHYSICAL_PROFILE = physical_profile
    try:
      context = multiprocessing.get_context("fork")
      results: list[tuple[tuple[str, str, str], tuple[BehaviorWindow, ...]]] = []
      for wave_start in range(0, len(jobs), worker_count):
        _check_replay_abort(abort_requested)
        indexes = tuple(range(wave_start, min(wave_start + worker_count, len(jobs))))
        workers: list[_BehaviorReplayWorker] = []
        wave_results: dict[int, tuple[tuple[str, str, str], tuple[BehaviorWindow, ...]]] = {}
        try:
          for index in indexes:
            workers.append(_BehaviorReplayWorker(
              context=context,
              job_index=index,
              abort_requested=abort_requested,
              inherited_close_fds=tuple(worker.receive_fd for worker in workers),
            ))
          while len(wave_results) != len(workers):
            _check_replay_abort(abort_requested)
            made_progress = False
            for worker in workers:
              if worker.job_index in wave_results:
                continue
              payload = worker.poll_result()
              if payload is None:
                continue
              result = _decode_worker_payload(payload, worker.job_index)
              if result[0] != jobs[worker.job_index].key:
                raise BehaviorTransactionError("behavior replay worker job identity mismatch")
              wave_results[worker.job_index] = result
              made_progress = True
            if not made_progress and len(wave_results) != len(workers):
              time.sleep(BEHAVIOR_WORKER_POLL_INTERVAL_S)
          for index in indexes:
            results.append(wave_results[index])
            if result_collected is not None:
              result_collected(len(results))
        finally:
          cleanup_error: BaseException | None = None
          for worker in workers:
            try:
              worker.cancel()
            except BaseException as exc:
              if cleanup_error is None:
                cleanup_error = exc
          if cleanup_error is not None:
            raise BehaviorTransactionError(
              "behavior replay worker cleanup failed",
            ) from cleanup_error
      return tuple(results)
    finally:
      _FORK_REPLAY_JOBS = None
      _FORK_PHYSICAL_PROFILE = None


def _evaluation_from_windows(
  identity: ReplayArtifactIdentity,
  policy: BehaviorPolicy | None,
  route_ids: tuple[str, ...],
  windows_by_job: dict[tuple[str, str, str], tuple[BehaviorWindow, ...]],
  metric_config,
  abort_requested: Callable[[], bool],
) -> PolicyEvaluation:
  _check_replay_abort(abort_requested)
  policy_key = "" if policy is None else policy.sha256
  windows = tuple(
    window
    for route_id in route_ids
    for window in windows_by_job[(identity.to_json(), policy_key, route_id)]
  )
  scorecard: BehaviorScorecard = score_behavior(windows, metric_config)
  _check_replay_abort(abort_requested)
  metrics = tuple(
    PolicyMetric.from_scorecard(scorecard, name)
    for name in BehaviorMetricName
  )
  return PolicyEvaluation(
    artifact_identity=identity.to_json(),
    policy=policy,
    route_ids=route_ids,
    metrics=metrics,
  )


def _decode_and_validate_routes(
  artifacts: Iterable[object],
  decoder: RouteEvidenceDecoder,
  physical_profile: VehicleCalibrationProfile,
  abort_requested: Callable[[], bool],
) -> tuple[DecodedBehaviorRoute, ...]:
  decoded_values: list[DecodedBehaviorRoute] = []
  for artifact in artifacts:
    _check_replay_abort(abort_requested)
    decoded_values.append(decoder(artifact, physical_profile))
    _check_replay_abort(abort_requested)
  decoded = tuple(decoded_values)
  if not decoded or any(not isinstance(route, DecodedBehaviorRoute) for route in decoded):
    raise BehaviorTransactionError("route decoder produced no compatible evidence")
  canonical = tuple(sorted(decoded, key=lambda route: route.route_id))
  route_ids = tuple(route.route_id for route in canonical)
  evidence_hashes = tuple(route.route_evidence_sha256 for route in canonical)
  if len(set(route_ids)) != len(route_ids):
    raise BehaviorTransactionError("behavior transaction contains duplicate route IDs")
  if len(set(evidence_hashes)) != len(evidence_hashes):
    raise BehaviorTransactionError("behavior transaction contains duplicate route evidence")
  if any(route.vehicle_identity != physical_profile.vehicle_identity for route in canonical):
    raise BehaviorTransactionError("route evidence belongs to a different vehicle")
  if len({route.recorded_source.sha256 for route in canonical}) != 1:
    raise BehaviorTransactionError("behavior transaction mixes recorded source identities")
  return canonical


def run_behavior_learning_transaction(
  *,
  route_evidence_artifacts: Iterable[object],
  decode_route_evidence: RouteEvidenceDecoder,
  physical_profile: VehicleCalibrationProfile,
  accepted_policy: BehaviorPolicy | None,
  search_center_policy: BehaviorPolicy,
  exact_stock: BehaviorReplayCore,
  currently_accepted: BehaviorReplayCore | None,
  candidate: BehaviorReplayCore,
  segmentation_config: SegmentationConfig | None = None,
  gate_spec: BehaviorGateSpec | None = None,
  worker_count: int = 1,
  progress_callback: BehaviorReplayProgressCallback | None = None,
  abort_requested: Callable[[], bool] = lambda: False,
) -> BehaviorLearningTransactionResult:
  """Replay, score, and finalize one non-persisting behavioral transaction.

  Training evaluates the complete bounded candidate grid in one canonical
  worker-pool batch.  Held-out routes are not even replayed until the training
  winner is frozen; validation then evaluates stock, the current artifact,
  and only that winner in one second batch.  Candidate jobs may execute in
  bounded fork workers, but collection and reduction remain in canonical
  order.  The replay callback must reset/rebootstrap at every recorded
  inactive-to-active episode boundary and censor response after driver contact
  until the next clean inactive-to-active boundary.

  ``progress_callback`` is a display-only observer called from the parent at
  deterministic, canonical result-collection boundaries.  Ordinary callback
  exceptions are isolated and cannot alter transaction bytes or qualification.
  No progress callback is inherited through the fork worker registry.

  ``abort_requested`` is the caller's ownership guard. Parallel replay polls
  it in the parent only and tears down every exact child-owned process group
  before propagating cancellation. Serial replay polls it between jobs and
  bounded output chunks. It has no influence on canonical result bytes.

  ``accepted_policy=None`` is the exact-stock bootstrap.  In that state the
  current baseline aliases ``exact_stock`` and ``search_center_policy`` is an
  explicit, committed search seed—not a fictitious accepted policy.
  """
  if not isinstance(physical_profile, VehicleCalibrationProfile):
    raise TypeError("physical_profile must be a VehicleCalibrationProfile")
  if not physical_profile.qualified:
    raise BehaviorTransactionError("physical calibration must be fully qualified first")
  if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count <= 0:
    raise ValueError("worker_count must be a positive integer")
  if worker_count > MAX_BEHAVIOR_REPLAY_WORKERS:
    raise ValueError(
      f"worker_count exceeds bounded maximum {MAX_BEHAVIOR_REPLAY_WORKERS}",
    )
  if not callable(decode_route_evidence):
    raise TypeError("decode_route_evidence must be callable")
  if progress_callback is not None and not callable(progress_callback):
    raise TypeError("progress_callback must be callable or None")
  if not callable(abort_requested):
    raise TypeError("abort_requested must be callable")
  _check_replay_abort(abort_requested)
  if (segmentation_config is None) != (gate_spec is None):
    raise BehaviorTransactionError(
      "gate and segmentation authorities must both be committed or both injected",
    )
  if segmentation_config is None:
    segmentation_config = load_behavior_segmentation_config()
    gate_spec = load_behavior_gate_spec()
  assert gate_spec is not None
  if not isinstance(search_center_policy, BehaviorPolicy):
    raise TypeError("search_center_policy must be a BehaviorPolicy")
  if accepted_policy is None:
    if currently_accepted is not None:
      raise BehaviorTransactionError(
        "exact-stock bootstrap must omit a separate currently-accepted core",
      )
    effective_accepted = exact_stock
  else:
    if not isinstance(accepted_policy, BehaviorPolicy):
      raise TypeError("accepted_policy must be a BehaviorPolicy or None")
    if currently_accepted is None:
      raise BehaviorTransactionError(
        "an accepted behavior policy requires its exact replay core",
      )
    if search_center_policy != accepted_policy:
      raise BehaviorTransactionError(
        "non-bootstrap search must be centered on the accepted policy",
      )
    effective_accepted = currently_accepted

  routes = _decode_and_validate_routes(
    route_evidence_artifacts,
    decode_route_evidence,
    physical_profile,
    abort_requested,
  )
  route_identities = tuple(route.identity for route in routes)
  # Partition here only to schedule immutable jobs.  The coordinator repeats
  # and hash-binds the same whole-route partition during finalization.
  partition = partition_whole_routes(route_identities, gate_spec.route_partition)
  grid = build_candidate_grid(
    gate_spec.candidate_grid.policy_grid(search_center_policy),
  )
  prepared_values: list[_PreparedRoute] = []
  for route in routes:
    _check_replay_abort(abort_requested)
    prepared_values.append(_prepare_route(
      route,
      physical_profile,
      segmentation_config,
      abort_requested,
    ))
  prepared = tuple(prepared_values)
  prepared_by_id = {
    route.route.route_id: route
    for route in prepared
  }

  stock_identity = ReplayArtifactIdentity.compose(
    ReplayRole.EXACT_STOCK,
    exact_stock.identity,
    None,
  )
  accepted_identity = ReplayArtifactIdentity.compose(
    ReplayRole.CURRENTLY_ACCEPTED,
    effective_accepted.identity,
    accepted_policy,
  )
  variants = (
    (stock_identity, None, exact_stock),
    (accepted_identity, accepted_policy, effective_accepted),
    *tuple(
      (
        ReplayArtifactIdentity.compose(
          ReplayRole.CANDIDATE,
          candidate.identity,
          candidate_value.policy,
        ),
        candidate_value.policy,
        candidate,
      )
      for candidate_value in grid
    ),
  )
  variant_lookup = {
    identity.to_json(): (policy, core)
    for identity, policy, core in variants
  }
  if len(variant_lookup) != len(variants):
    raise BehaviorTransactionError("behavior replay variants are not unique")
  evaluation_lookup: dict[
    tuple[str, tuple[str, ...]],
    PolicyEvaluation,
  ] = {}

  training_job_count = len(variants) * len(partition.training_route_ids)
  validation_job_count = 3 * len(partition.validation_route_ids)
  total_job_count = training_job_count + validation_job_count
  completed_job_count = 0

  def notify_progress(
    phase: BehaviorReplayProgressPhase,
    phase_completed_jobs: int,
    phase_total_jobs: int,
  ) -> None:
    if progress_callback is None:
      return
    progress = BehaviorReplayProgress(
      phase=phase,
      completed_jobs=completed_job_count + phase_completed_jobs,
      total_jobs=total_job_count,
      phase_completed_jobs=phase_completed_jobs,
      phase_total_jobs=phase_total_jobs,
    )
    try:
      progress_callback(progress)
    except Exception:
      # Display plumbing has no authority over evidence, selection, or output.
      # A broken UI observer therefore cannot fail or perturb learning.
      pass

  def replay_batch(
    batch_variants: tuple[
      tuple[ReplayArtifactIdentity, BehaviorPolicy | None, BehaviorReplayCore],
      ...
    ],
    route_ids: tuple[str, ...],
    phase: BehaviorReplayProgressPhase,
  ) -> None:
    nonlocal completed_job_count
    try:
      selected_routes = tuple(prepared_by_id[route_id] for route_id in route_ids)
    except KeyError as exc:
      raise BehaviorTransactionError("behavior replay requested unavailable route evidence") from exc
    jobs = tuple(sorted(
      (
        _ReplayJob(identity, policy, core, route)
        for identity, policy, core in batch_variants
        for route in selected_routes
      ),
      key=lambda job: job.key,
    ))
    expected_job_count = len(batch_variants) * len(route_ids)
    if len(jobs) != expected_job_count:
      raise BehaviorTransactionError("behavior replay batch job count mismatch")
    notify_progress(phase, 0, expected_job_count)
    replayed = _run_replay_jobs(
      jobs,
      physical_profile,
      worker_count,
      abort_requested,
      lambda completed: notify_progress(phase, completed, expected_job_count),
    )
    _check_replay_abort(abort_requested)
    windows_by_job = dict(replayed)
    if len(windows_by_job) != len(jobs):
      raise BehaviorTransactionError("behavior replay jobs did not have unique identities")
    for identity, policy, _core in batch_variants:
      evaluation = _evaluation_from_windows(
        identity,
        policy,
        route_ids,
        windows_by_job,
        gate_spec.metric_config,
        abort_requested,
      )
      evaluation_lookup[(identity.to_json(), route_ids)] = evaluation
    completed_job_count += expected_job_count

  # The expensive work is deliberately scheduled independently of the
  # coordinator's scalar callback order: all training work crosses one pool,
  # then the exact existing selector freezes one winner before validation can
  # read a single held-out control witness.
  replay_batch(
    variants,
    partition.training_route_ids,
    BehaviorReplayProgressPhase.TRAINING,
  )
  stock_training = evaluation_lookup[(
    stock_identity.to_json(),
    partition.training_route_ids,
  )]
  accepted_training = evaluation_lookup[(
    accepted_identity.to_json(),
    partition.training_route_ids,
  )]
  candidate_training = tuple(
    evaluation_lookup[(identity.to_json(), partition.training_route_ids)]
    for identity, _policy, _core in variants
    if identity.role is ReplayRole.CANDIDATE
  )
  try:
    _check_replay_abort(abort_requested)
    frozen_training_selection = select_training_winner(
      grid,
      candidate_training,
      stock_training,
      accepted_training,
      gate_spec.metric_rules,
      gate_spec.target_metric_name,
      gate_spec.paired_uncertainty_method,
      gate_spec.minimum_paired_route_count,
    )
  except ValueError:
    frozen_training_selection = None
  _check_replay_abort(abort_requested)

  if frozen_training_selection is not None:
    winner_policy = frozen_training_selection.winner.policy
    winner_identity = ReplayArtifactIdentity.compose(
      ReplayRole.CANDIDATE,
      candidate.identity,
      winner_policy,
    )
    try:
      winner_policy_from_lookup, winner_core = variant_lookup[winner_identity.to_json()]
    except KeyError as exc:
      raise BehaviorTransactionError("frozen winner is absent from the training grid") from exc
    if winner_policy_from_lookup != winner_policy:
      raise BehaviorTransactionError("frozen winner policy identity mismatch")
    replay_batch(
      (
        (stock_identity, None, exact_stock),
        (accepted_identity, accepted_policy, effective_accepted),
        (winner_identity, winner_policy, winner_core),
      ),
      partition.validation_route_ids,
      BehaviorReplayProgressPhase.VALIDATION,
    )

  def replay_evaluate(
    identity: ReplayArtifactIdentity,
    policy: BehaviorPolicy | None,
    route_ids: tuple[str, ...],
  ) -> PolicyEvaluation:
    key = (identity.to_json(), route_ids)
    try:
      evaluation = evaluation_lookup[key]
    except KeyError as exc:
      raise BehaviorTransactionError(
        "coordinator requested a replay outside the frozen two-batch plan",
      ) from exc
    if evaluation.policy != policy:
      raise BehaviorTransactionError("cached evaluation policy identity mismatch")
    return evaluation

  _check_replay_abort(abort_requested)
  finalization = finalize_behavior_learning(
    gate_spec=gate_spec,
    routes=route_identities,
    accepted_policy=accepted_policy,
    search_center_policy=search_center_policy,
    exact_stock_core=exact_stock.identity,
    accepted_core=(
      None if accepted_policy is None else effective_accepted.identity
    ),
    candidate_core=candidate.identity,
    replay_evaluate=replay_evaluate,
  )
  _check_replay_abort(abort_requested)
  evaluations = tuple(sorted(
    evaluation_lookup.values(),
    key=lambda evaluation: (evaluation.artifact_identity, evaluation.route_ids),
  ))
  profile_sha = hashlib.sha256(physical_profile.to_json().encode("utf-8")).hexdigest()
  return BehaviorLearningTransactionResult(
    schema_version=BEHAVIOR_TRANSACTION_SCHEMA_VERSION,
    physical_profile_sha256=profile_sha,
    route_evidence_sha256s=tuple(
      (route.route_id, route.route_evidence_sha256)
      for route in routes
    ),
    segmentation_config_sha256=segmentation_config.sha256,
    segmentation_sha256s=tuple(
      (route.route.route_id, route.segmentation.sha256)
      for route in prepared
    ),
    evaluations=evaluations,
    finalization=finalization,
    qualification_disposition=(
      QualificationDisposition.QUALIFIED_CANDIDATE_AVAILABLE
      if finalization.passed
      else QualificationDisposition.STOCK_RETAINED
    ),
  )

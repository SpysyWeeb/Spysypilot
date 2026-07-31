"""Persistent measured-only runtime for the modular BLaTv2 learner.

This module bridges the pure learner lifecycle to detected vehicle facts.  It
contains no messaging and accepts no model, desired-curvature, requested
torque, or candidate-controller signal.  The only torque input is the command
actually emitted by ``CarController`` in ``carOutput.actuatorsOutput``.

Evidence remains in memory onroad.  The onroad-to-offroad transition is the
only operation that creates directories or writes artifacts.  Restore verifies
the exact seed-bound evidence, manifest, and optional hash-addressed candidate;
an incomplete or corrupt artifact set raises instead of silently starting over.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path

from opendbc.car.structs import car
from opendbc.car.vehicle_model import VehicleModel
from openpilot.selfdrive.controls.lib.blatv2.actuator import (
  RuntimeTorqueLimits,
  apply_torque_envelope,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_coordinator import (
  LearningCoordinator,
  LearningFinalization,
  LearningLifecycleState,
)
from openpilot.selfdrive.controls.lib.blatv2.measurement import (
  MAX_CONTINUOUS_MEASUREMENT_GAP_S,
  LearningMeasurementBuilder,
)
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import (
  RackMappingSnapshot,
)
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  ProvisionalRackDynamics,
  RuntimeVehicleBundle,
  build_runtime_vehicle_bundle,
)


CASUAL_DRIVING_CANDIDATE_PROVENANCE = "measured casual-driving evidence"
_CONTROLLER_LIMIT_FIELDS = (
  "STEER_MAX",
  "STEER_DELTA_UP",
  "STEER_DELTA_DOWN",
  "STEER_STEP",
  "STEER_DRIVER_ALLOWANCE",
  "STEER_DRIVER_MULTIPLIER",
  "STEER_DRIVER_FACTOR",
)


class LearningRestoreError(RuntimeError):
  """Persistent learner state is incomplete, corrupt, or seed-incompatible."""


@dataclass(frozen=True, slots=True)
class LearningArtifactPaths:
  """Caller-selected artifact directory with hash-addressed candidates."""

  root: Path
  # A runtime restore snapshots CURRENT exactly once. The hidden resolved root
  # prevents one restore from mixing two immutable generations if CURRENT
  # flips between evidence/manifest/candidate reads.
  _resolved_root: Path | None = field(
    default=None,
    repr=False,
    compare=False,
  )

  @property
  def backfill_pointer(self) -> Path:
    return self.root / "backfill_current.json"

  @property
  def backfill_generations(self) -> Path:
    return self.root / "backfill_generations"

  def _active_root(self) -> Path:
    if self._resolved_root is not None:
      return self._resolved_root
    if not self.backfill_pointer.is_file():
      return self.root
    encoded = self.backfill_pointer.read_bytes()
    payload = json.loads(encoded)
    expected_keys = {"generation_sha256", "schema_version"}
    if (
      type(payload) is not dict
      or set(payload) != expected_keys
      or type(payload["schema_version"]) is not int
      or payload["schema_version"] != 1
      or type(payload["generation_sha256"]) is not str
      or len(payload["generation_sha256"]) != 64
      or any(
        character not in "0123456789abcdef"
        for character in payload["generation_sha256"]
      )
      or encoded != json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
      ).encode("utf-8")
    ):
      raise ValueError("backfill pointer is not canonical")
    return self.backfill_generations / payload["generation_sha256"]

  def resolved(self) -> LearningArtifactPaths:
    if self._resolved_root is not None:
      return self
    return LearningArtifactPaths(
      root=self.root,
      _resolved_root=self._active_root(),
    )

  @property
  def evidence(self) -> Path:
    return self._active_root() / "evidence.json"

  @property
  def manifest(self) -> Path:
    return self._active_root() / "manifest.json"

  @property
  def candidates(self) -> Path:
    return self._active_root() / "candidates"

  @property
  def backfill_ledger(self) -> Path:
    return self._active_root() / "ledger.json"

  @property
  def backfill_provenance(self) -> Path:
    return self._active_root() / "provenance.json"

  @property
  def backfill_commit(self) -> Path:
    return self._active_root() / "commit.json"

  def candidate(self, profile_sha256: str) -> Path:
    identity = str(profile_sha256)
    if (
      len(identity) != 64
      or any(character not in "0123456789abcdef" for character in identity)
    ):
      raise ValueError("candidate profile identity must be lowercase SHA-256")
    return self.candidates / f"{identity}.json"


@dataclass(frozen=True, slots=True)
class MeasuredLearningFrame:
  """One time-aligned physical-response frame; no command intent is accepted."""

  sample_mono_ns: int
  speed_mps: float
  steering_angle_deg: float
  steering_rate_deg_s: float
  steering_torque: float
  steering_pressed: bool
  standstill: bool
  steer_fault_temporary: bool
  steer_fault_permanent: bool
  can_valid: bool
  can_timeout: bool
  applied_torque: float
  lateral_active: bool
  live_parameters_valid: bool
  angle_offset_valid: bool
  steer_ratio_valid: bool
  stiffness_factor_valid: bool
  angle_offset_deg: float
  steer_ratio: float
  stiffness_factor: float
  roll_rad: float
  inputs_valid: bool


def controller_params_from_detected_interface(
  car_interface: object,
  car_params: car.CarParams,
) -> object:
  """Resolve authoritative opendbc limits through the detected interface."""
  controller_class = getattr(car_interface, "CarController", None)
  if controller_class is None:
    raise RuntimeError("detected CarInterface has no CarController class")
  module = importlib.import_module(controller_class.__module__)
  params_class = getattr(module, "CarControllerParams", None)
  if params_class is None:
    raise RuntimeError(
      "detected CarController does not expose CarControllerParams",
    )

  signature = inspect.signature(params_class)
  positional = tuple(
    parameter
    for parameter in signature.parameters.values()
    if parameter.kind in (
      inspect.Parameter.POSITIONAL_ONLY,
      inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
  )
  if len(positional) == 0:
    controller_params = params_class()
  elif len(positional) == 1:
    controller_params = params_class(car_params)
  else:
    raise RuntimeError(
      "detected CarControllerParams has an unsupported constructor",
    )
  if not all(
    hasattr(controller_params, field)
    for field in _CONTROLLER_LIMIT_FIELDS
  ):
    raise RuntimeError(
      "detected CarControllerParams lacks the torque-envelope contract",
    )
  return controller_params


def build_detected_runtime_bundle(
  *,
  car_params: car.CarParams,
  provisional_rack_dynamics: ProvisionalRackDynamics,
  interface_registry: Mapping[str, type] | None = None,
) -> tuple[RuntimeVehicleBundle, object, object]:
  """Construct a bundle generically from detected CarParams and opendbc."""
  registry = interface_registry
  if registry is None:
    from opendbc.car.car_helpers import interfaces

    registry = interfaces
  fingerprint = str(car_params.carFingerprint)
  try:
    interface_class = registry[fingerprint]
  except KeyError as exc:
    raise RuntimeError(
      "detected car fingerprint has no registered CarInterface",
    ) from exc
  car_interface = interface_class(car_params)
  controller_params = controller_params_from_detected_interface(
    car_interface,
    car_params,
  )
  bundle = build_runtime_vehicle_bundle(
    car_params=car_params,
    car_interface_or_callback=car_interface,
    controller_params=controller_params,
    vehicle_identity=fingerprint,
    provisional_rack_dynamics=provisional_rack_dynamics,
  )
  return bundle, car_interface, controller_params


class _MeasuredEnvelopeConstraint:
  """Conservatively reject applied torque sitting on an opendbc boundary."""

  __slots__ = (
    "_limits",
    "_previous_applied",
    "_previous_time_s",
    "_torque_count_tolerance",
  )

  def __init__(self, limits: RuntimeTorqueLimits) -> None:
    self._limits = limits
    self._previous_applied = 0.0
    self._previous_time_s: float | None = None
    # CarController output is normalized after an integer-count limiter. Half
    # a count admits Float32/log round-trip but cannot hide a whole slew step.
    self._torque_count_tolerance = 0.5 / limits.steer_max

  def reset(self) -> None:
    self._previous_applied = 0.0
    self._previous_time_s = None

  def update(
    self,
    *,
    sample_time_s: float,
    applied_torque: float,
    driver_torque: float,
    inputs_valid: bool,
  ) -> bool:
    values = (sample_time_s, applied_torque, driver_torque)
    if (
      not inputs_valid
      or not all(math.isfinite(value) for value in values)
      or abs(applied_torque) > 1.0
    ):
      self.reset()
      return True

    constrained = True
    if self._previous_time_s is not None:
      dt_s = sample_time_s - self._previous_time_s
      if 0.0 < dt_s <= MAX_CONTINUOUS_MEASUREMENT_GAP_S:
        positive = apply_torque_envelope(
          self._limits,
          1.0,
          self._previous_applied,
          driver_torque,
        ).applied_torque
        negative = apply_torque_envelope(
          self._limits,
          -1.0,
          self._previous_applied,
          driver_torque,
        ).applied_torque
        lower = min(positive, negative)
        upper = max(positive, negative)
        tolerance = self._torque_count_tolerance
        inside = lower - tolerance <= applied_torque <= upper + tolerance
        on_boundary = (
          abs(applied_torque - lower) <= tolerance
          or abs(applied_torque - upper) <= tolerance
        )
        constrained = not inside or on_boundary

    self._previous_time_s = sample_time_s
    self._previous_applied = applied_torque
    return constrained


class PersistentLearningRuntime:
  """One detected vehicle's in-memory drive learner and offroad persistence."""

  def __init__(
    self,
    *,
    car_params: car.CarParams,
    runtime_bundle: RuntimeVehicleBundle,
    artifact_paths: LearningArtifactPaths,
    coordinator: LearningCoordinator,
  ) -> None:
    if (
      coordinator.vehicle_identity
      != runtime_bundle.seed_profile.vehicle_identity
    ):
      raise ValueError("coordinator and runtime bundle vehicle mismatch")
    self.car_params = car_params
    self.runtime_bundle = runtime_bundle
    self.artifact_paths = artifact_paths
    self.coordinator = coordinator
    self.vehicle_model = VehicleModel(car_params)
    self.measurement_builder = LearningMeasurementBuilder()
    self.envelope_constraint = _MeasuredEnvelopeConstraint(
      runtime_bundle.torque_limits,
    )
    self.last_sample_accepted = False
    self.last_actuator_constrained = True
    self.last_live_mapping_valid = False

  @classmethod
  def restore(
    cls,
    *,
    car_params: car.CarParams,
    runtime_bundle: RuntimeVehicleBundle,
    artifact_paths: LearningArtifactPaths,
  ) -> PersistentLearningRuntime:
    # Resolve the generation once for this transaction. The returned runtime
    # continues to reference that immutable generation until rebuilt.
    artifact_paths = artifact_paths.resolved()
    try:
      if artifact_paths.backfill_pointer.is_file():
        cls._validate_backfill_generation(
          artifact_paths=artifact_paths,
          runtime_bundle=runtime_bundle,
        )
    except LearningRestoreError:
      raise
    except (OSError, TypeError, ValueError) as exc:
      raise LearningRestoreError(
        "stored backfill generation failed canonical restore",
      ) from exc

    evidence_exists = artifact_paths.evidence.is_file()
    manifest_exists = artifact_paths.manifest.is_file()
    if evidence_exists != manifest_exists:
      raise LearningRestoreError(
        "learner evidence and manifest must either both exist or both be absent",
      )

    if not evidence_exists:
      coordinator = LearningCoordinator(
        runtime_bundle.seed_profile,
        candidate_provenance=CASUAL_DRIVING_CANDIDATE_PROVENANCE,
      )
      return cls(
        car_params=car_params,
        runtime_bundle=runtime_bundle,
        artifact_paths=artifact_paths,
        coordinator=coordinator,
      )

    try:
      evidence_bytes = artifact_paths.evidence.read_bytes()
      manifest_bytes = artifact_paths.manifest.read_bytes()
      coordinator = LearningCoordinator(
        runtime_bundle.seed_profile,
        evidence_bytes,
        candidate_provenance=CASUAL_DRIVING_CANDIDATE_PROVENANCE,
      )
      finalization = coordinator.finalize()
      if finalization.manifest_bytes != manifest_bytes:
        raise LearningRestoreError(
          "stored learner manifest does not identify the exact evidence",
        )
      candidate_json = finalization.candidate_profile_json
      candidate_identity = finalization.candidate_profile_sha256
      if candidate_json is not None:
        if candidate_identity is None:
          raise AssertionError("candidate JSON lacks its canonical identity")
        candidate_path = artifact_paths.candidate(candidate_identity)
        if (
          not candidate_path.is_file()
          or candidate_path.read_bytes() != candidate_json
        ):
          raise LearningRestoreError(
            "stored learner candidate is missing or does not match manifest",
          )
    except LearningRestoreError:
      raise
    except (OSError, TypeError, ValueError) as exc:
      raise LearningRestoreError(
        "stored learner artifacts failed canonical restore",
      ) from exc

    return cls(
      car_params=car_params,
      runtime_bundle=runtime_bundle,
      artifact_paths=artifact_paths,
      coordinator=coordinator,
    )

  @staticmethod
  def _validate_backfill_generation(
    *,
    artifact_paths: LearningArtifactPaths,
    runtime_bundle: RuntimeVehicleBundle,
  ) -> None:
    commit_path = artifact_paths.backfill_commit
    if not commit_path.is_file():
      raise LearningRestoreError("backfill generation lacks commit record")
    commit_bytes = commit_path.read_bytes()
    commit = json.loads(commit_bytes)
    expected_keys = {
      "candidate_profile_sha256",
      "evidence_sha256",
      "ledger_sha256",
      "manifest_sha256",
      "provenance_sha256",
      "runtime_identity_sha256",
      "schema_version",
    }
    if (
      type(commit) is not dict
      or set(commit) != expected_keys
      or type(commit["schema_version"]) is not int
      or commit["schema_version"] != 1
      or commit_bytes != json.dumps(
        commit,
        sort_keys=True,
        separators=(",", ":"),
      ).encode("utf-8")
    ):
      raise LearningRestoreError("backfill commit record is not canonical")
    generation_identity = hashlib.sha256(commit_bytes).hexdigest()
    if artifact_paths.backfill_commit.parent.name != generation_identity:
      raise LearningRestoreError(
        "backfill generation directory does not match commit identity",
      )

    def verify_file(path: Path, expected: object, name: str) -> None:
      if (
        type(expected) is not str
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
        or not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != expected
      ):
        raise LearningRestoreError(
          f"backfill {name} does not match commit record",
        )

    if commit["runtime_identity_sha256"] != runtime_bundle.identity_sha256:
      raise LearningRestoreError(
        "backfill generation belongs to another runtime",
      )
    verify_file(
      artifact_paths.evidence,
      commit["evidence_sha256"],
      "evidence",
    )
    verify_file(
      artifact_paths.manifest,
      commit["manifest_sha256"],
      "manifest",
    )
    verify_file(
      artifact_paths.backfill_ledger,
      commit["ledger_sha256"],
      "ledger",
    )
    verify_file(
      artifact_paths.backfill_provenance,
      commit["provenance_sha256"],
      "provenance",
    )
    candidate_identity = commit["candidate_profile_sha256"]
    if candidate_identity is not None:
      verify_file(
        artifact_paths.candidate(candidate_identity),
        candidate_identity,
        "candidate",
      )

  def transition_onroad(self) -> None:
    self.measurement_builder.reset()
    self.envelope_constraint.reset()
    self.coordinator.transition_onroad()

  def transition_offroad_and_persist(self) -> LearningFinalization:
    self.coordinator.transition_offroad()
    self.measurement_builder.reset()
    self.envelope_constraint.reset()
    return self.persist_offroad()

  def transition_offroad_without_persist(self) -> None:
    """End a live preview drive without claiming durable route ownership."""
    self.coordinator.transition_offroad()
    self.measurement_builder.reset()
    self.envelope_constraint.reset()

  def persist_offroad(self) -> LearningFinalization:
    """Persist the current finalized state; safe to retry after I/O failure."""
    if self.coordinator.state is not LearningLifecycleState.OFFROAD:
      raise RuntimeError("learner persistence is permitted only offroad")
    if self.artifact_paths.backfill_pointer.is_file():
      raise RuntimeError(
        "ledger-owned evidence may only be persisted by backfill",
      )
    finalization = self.coordinator.finalize()

    # No directory creation is reachable while the coordinator is onroad.
    self.artifact_paths.root.mkdir(parents=True, exist_ok=True)
    candidate_path: Path | None = None
    if finalization.candidate_profile_sha256 is not None:
      self.artifact_paths.candidates.mkdir(parents=True, exist_ok=True)
      candidate_path = self.artifact_paths.candidate(
        finalization.candidate_profile_sha256,
      )
    return self.coordinator.persist_finalized(
      evidence_path=self.artifact_paths.evidence,
      manifest_path=self.artifact_paths.manifest,
      candidate_profile_path=candidate_path,
    )

  def _live_mapping(
    self,
    frame: MeasuredLearningFrame,
  ) -> RackMappingSnapshot | None:
    valid = (
      frame.inputs_valid
      and frame.live_parameters_valid
      and frame.angle_offset_valid
      and frame.steer_ratio_valid
      and frame.stiffness_factor_valid
    )
    values = (
      frame.angle_offset_deg,
      frame.steer_ratio,
      frame.stiffness_factor,
      frame.roll_rad,
    )
    if (
      not valid
      or not all(math.isfinite(value) for value in values)
      or frame.steer_ratio <= 0.0
      or frame.stiffness_factor <= 0.0
    ):
      self.last_live_mapping_valid = False
      return None
    try:
      self.vehicle_model.update_params(
        frame.stiffness_factor,
        frame.steer_ratio,
      )
      mapping = RackMappingSnapshot.from_vehicle_model(
        self.vehicle_model,
        roll_rad=frame.roll_rad,
        angle_offset_deg=frame.angle_offset_deg,
        valid=True,
      )
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
      self.last_live_mapping_valid = False
      return None
    self.last_live_mapping_valid = True
    return mapping

  def ingest(self, frame: MeasuredLearningFrame) -> bool:
    if self.coordinator.state is not LearningLifecycleState.ONROAD:
      raise RuntimeError("measured frames may be ingested only while onroad")
    if not isinstance(frame, MeasuredLearningFrame):
      raise TypeError("persistent learner requires MeasuredLearningFrame")

    sample_time_s = int(frame.sample_mono_ns) * 1e-9
    numeric_valid = frame.inputs_valid and all(math.isfinite(value) for value in (
      sample_time_s,
      frame.speed_mps,
      frame.steering_angle_deg,
      frame.steering_rate_deg_s,
      frame.steering_torque,
      frame.applied_torque,
    ))
    standstill = bool(frame.standstill)
    below_steer_speed = abs(frame.speed_mps) <= max(
      float(self.car_params.minSteerSpeed),
      0.3,
    )
    lateral_active = (
      numeric_valid
      and frame.lateral_active
      and frame.can_valid
      and not frame.can_timeout
      and not frame.steer_fault_temporary
      and not frame.steer_fault_permanent
      and (
        not (standstill or below_steer_speed)
        or bool(self.car_params.steerAtStandstill)
      )
    )
    actuator_constrained = self.envelope_constraint.update(
      sample_time_s=sample_time_s,
      applied_torque=frame.applied_torque,
      driver_torque=frame.steering_torque,
      inputs_valid=numeric_valid and lateral_active,
    )
    self.last_actuator_constrained = actuator_constrained
    live_mapping = self._live_mapping(frame)
    torque_tuning = self.car_params.lateralTuning.torque
    measurement_inputs_valid = (
      numeric_valid
      and self.last_live_mapping_valid
      and not frame.steering_pressed
      and not actuator_constrained
    )
    sample = self.measurement_builder.update(
      sample_time_s=sample_time_s,
      speed_mps=frame.speed_mps,
      measured_rack_angle_deg=frame.steering_angle_deg,
      measured_rack_rate_deg_s=frame.steering_rate_deg_s,
      applied_torque=frame.applied_torque,
      lateral_accel_offset=float(torque_tuning.latAccelOffset),
      live_mapping=live_mapping,
      nominal_mapping=self.runtime_bundle.nominal_rack_mapping,
      engaged=lateral_active,
      # Driver interaction and an envelope boundary invalidate derivative
      # history as well as the current frame. The first clean frame afterward
      # is therefore warm-up, not a fit of the release transient.
      inputs_valid=measurement_inputs_valid,
      steering_pressed=frame.steering_pressed,
      actuator_constrained=actuator_constrained,
      standstill=standstill,
    )
    accepted = self.coordinator.ingest(sample)
    self.last_sample_accepted = accepted
    return accepted


def artifact_paths_for_bundle(
  storage_root: str | os.PathLike[str],
  runtime_bundle: RuntimeVehicleBundle,
) -> LearningArtifactPaths:
  """Keep different detected vehicle/seed bundles in separate directories."""
  return LearningArtifactPaths(
    Path(storage_root) / runtime_bundle.identity_sha256,
  )


def build_persistent_learning_runtime(
  *,
  car_params: car.CarParams,
  storage_root: str | os.PathLike[str],
  provisional_rack_dynamics: ProvisionalRackDynamics,
  interface_registry: Mapping[str, type] | None = None,
) -> PersistentLearningRuntime:
  bundle, _, _ = build_detected_runtime_bundle(
    car_params=car_params,
    provisional_rack_dynamics=provisional_rack_dynamics,
    interface_registry=interface_registry,
  )
  return PersistentLearningRuntime.restore(
    car_params=car_params,
    runtime_bundle=bundle,
    artifact_paths=artifact_paths_for_bundle(storage_root, bundle),
  )

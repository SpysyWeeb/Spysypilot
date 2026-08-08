"""Bounded one-route counterfactual evaluation for the offline PC trainer.

The reference-only preparation pass authenticates one shared route artifact
and freezes its physical phase spans exactly once.  Exact stock and every
modular candidate then replay the same canonical scenario against those same
spans.  Recorded controller requests remain provenance only and never enter
either controller's state. Active lateral-maneuver injections are rejected:
stock legitimately honors that override, while the modular controller follows
modelV2, so such a route is physical-identification evidence rather than a
like-for-like behavioral scenario.

This module has no Params, publication, activation, or actuation path.  Route
samples live in an owned temporary fixed-record scratch file; the returned
objects contain only immutable scenario identity, phase descriptors, and
per-window metric results.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, TypeVar

from opendbc.car.vehicle_model import VehicleModel

from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import (
  BehaviorControlResponse,
  BehaviorReferenceAtControl,
  BehaviorSample,
  BehaviorScenarioProvenance,
  BehaviorScenarioSetIdentity,
  BehaviorSourceIdentity,
  EventLocator,
  SparseModelBehaviorIntent,
  assemble_behavior_sample,
  canonical_json,
  derive_behavior_reference,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_coordinator import (
  ReplayArtifactIdentity,
  ReplayCoreIdentity,
  ReplayRole,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_filebacked import (
  BehaviorSampleScratch,
  BehaviorSampleSpan,
  FileBackedBehaviorWindow,
  score_file_backed_window,
  segment_file_backed_behavior_route,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_metrics import (
  BehaviorMetricConfig,
  WindowMetricSet,
  retain_route_metric_windows,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_policy import BehaviorPolicy
from openpilot.selfdrive.controls.lib.blatv2.behavior_replay import (
  BehaviorReplayError,
  BehaviorReplayStepper,
  ReplayFrameInput,
  behavior_scenario_provenance_from_route_source,
  build_canonical_behavior_frame,
  decode_behavior_car_params,
  validate_reviewed_replay_core_identity,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_segmentation import (
  BehaviorPhaseSpan,
  EventCoverage,
  SegmentationConfig,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_transaction import (
  CanonicalBehaviorControlInput,
  ControllerFrameOutput,
  neutral_behavior_response,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_profile import (
  VehicleCalibrationProfile,
)
from openpilot.selfdrive.controls.lib.blatv2.counterfactual_plant import (
  CounterfactualPlantMember,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_runtime import (
  build_detected_runtime_bundle,
)
from openpilot.selfdrive.controls.lib.blatv2.route_evidence import (
  ControlsWitness,
  RouteEvidenceError,
  RouteEvidenceStreamReader,
)
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  ProvisionalRackDynamics,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  compose_controller_profile,
)


BEHAVIOR_ROUTE_PREPARATION_SCHEMA_VERSION = 3
BEHAVIOR_ROUTE_EVALUATION_SCHEMA_VERSION = 5
_RACK_ACCELERATION_MAXIMUM_GAP_NS = 15_000_000
_MAXIMUM_ROUTE_POLICIES = 64


class BehaviorRouteEvaluationError(RuntimeError):
  """One route cannot satisfy the authenticated scenario contract."""


def _sha256_text(value: str) -> str:
  return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _span_dict(span: BehaviorPhaseSpan) -> dict[str, object]:
  return {
    "endSampleIndexExclusive": span.end_sample_index_exclusive,
    "eventLocators": [event.to_dict() for event in span.event_locators],
    "maneuverClass": span.maneuver_class.value,
    "observability": span.observability.to_dict(),
    "phase": span.phase.value,
    "startSampleIndex": span.start_sample_index,
    "windowId": span.window_id,
  }


def _metric_set_dict(value: WindowMetricSet) -> dict[str, object]:
  return {
    "cleanSampleCount": value.clean_sample_count,
    "interventionMonoTimeNs": value.intervention_mono_time_ns,
    "maneuverClass": value.maneuver_class.value,
    "meanSpeedMps": value.mean_speed_mps,
    "metrics": [metric.to_dict() for metric in value.metrics],
    "phase": value.phase.value,
    "routeId": value.route_id,
    "sourceIdentitySha256": value.source_identity_sha256,
    "speedNodeSupport": [list(item) for item in value.speed_node_support],
    "summaryMetricName": (
      None if value.summary_metric_name is None else value.summary_metric_name.value
    ),
    "windowId": value.window_id,
  }


def _unassigned_ranges(
  sample_count: int,
  spans: tuple[BehaviorPhaseSpan, ...],
) -> tuple[tuple[int, int], ...]:
  output: list[tuple[int, int]] = []
  cursor = 0
  for span in spans:
    if span.start_sample_index < cursor or span.end_sample_index_exclusive > sample_count:
      raise BehaviorRouteEvaluationError("segmentation spans are not canonical")
    if cursor < span.start_sample_index:
      output.append((cursor, span.start_sample_index))
    cursor = span.end_sample_index_exclusive
  if cursor < sample_count:
    output.append((cursor, sample_count))
  return tuple(output)


@dataclass(frozen=True, slots=True)
class BehaviorRoutePreparation:
  """Reference-only phase authority shared by every route opponent."""

  schema_version: int
  scenario: BehaviorScenarioProvenance
  physical_profile_sha256: str
  provisional_dynamics_sha256: str
  segmentation_config_sha256: str
  reference_samples_sha256: str
  file_backed_segmentation_sha256: str
  sample_count: int
  spans: tuple[BehaviorPhaseSpan, ...]
  event_coverage: tuple[EventCoverage, ...]
  unassigned_sample_ranges: tuple[tuple[int, int], ...]

  def __post_init__(self) -> None:
    if self.schema_version != BEHAVIOR_ROUTE_PREPARATION_SCHEMA_VERSION:
      raise ValueError("behavior route preparation schema is incompatible")
    if not isinstance(self.scenario, BehaviorScenarioProvenance):
      raise TypeError("behavior route preparation requires scenario provenance")
    if self.sample_count <= 0:
      raise ValueError("behavior route preparation requires route samples")
    for name, value in (
      ("physical profile", self.physical_profile_sha256),
      ("provisional dynamics", self.provisional_dynamics_sha256),
      ("segmentation config", self.segmentation_config_sha256),
      ("reference samples", self.reference_samples_sha256),
      ("file-backed segmentation", self.file_backed_segmentation_sha256),
    ):
      if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} identity must be SHA-256")
    if type(self.spans) is not tuple or type(self.event_coverage) is not tuple:
      raise TypeError("behavior route preparation populations must be tuples")
    if self.unassigned_sample_ranges != _unassigned_ranges(self.sample_count, self.spans):
      raise ValueError("unassigned sample ranges disagree with phase spans")

  def to_dict(self) -> dict[str, object]:
    return {
      "eventCoverage": [value.to_dict() for value in self.event_coverage],
      "fileBackedSegmentationSha256": self.file_backed_segmentation_sha256,
      "physicalProfileSha256": self.physical_profile_sha256,
      "provisionalDynamicsSha256": self.provisional_dynamics_sha256,
      "referenceSamplesSha256": self.reference_samples_sha256,
      "sampleCount": self.sample_count,
      "scenario": self.scenario.to_dict(),
      "schemaVersion": self.schema_version,
      "segmentationConfigSha256": self.segmentation_config_sha256,
      "spans": [_span_dict(span) for span in self.spans],
      "unassignedSampleRanges": [list(value) for value in self.unassigned_sample_ranges],
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())

  @property
  def sha256(self) -> str:
    return _sha256_text(self.to_json())


@dataclass(frozen=True, slots=True)
class BehaviorRouteEvaluation:
  """Compact per-window result for one exact route/opponent pair."""

  schema_version: int
  preparation_sha256: str
  preparation_schema_version: int
  scenario: BehaviorScenarioProvenance
  single_route_scenario_set_sha256: str
  artifact_identity: ReplayArtifactIdentity
  physical_profile_sha256: str
  provisional_dynamics_sha256: str
  plant_member_id: str
  segmentation_config_sha256: str
  metric_config_sha256: str
  windows: tuple[WindowMetricSet, ...]

  def __post_init__(self) -> None:
    if self.schema_version != BEHAVIOR_ROUTE_EVALUATION_SCHEMA_VERSION:
      raise ValueError("behavior route evaluation schema is incompatible")
    if self.preparation_schema_version != BEHAVIOR_ROUTE_PREPARATION_SCHEMA_VERSION:
      raise ValueError("behavior route preparation contract is incompatible")
    if not isinstance(self.artifact_identity, ReplayArtifactIdentity):
      raise TypeError("behavior route requires an exact replay artifact identity")
    if self.artifact_identity.role not in {
      ReplayRole.EXACT_STOCK,
      ReplayRole.CURRENTLY_ACCEPTED,
      ReplayRole.CANDIDATE,
    }:
      raise ValueError("behavior route opponent role is invalid")
    if (
      self.artifact_identity.behavior_policy_sha256 is None
    ) != (self.artifact_identity.role is ReplayRole.EXACT_STOCK):
      raise ValueError("behavior route opponent and policy identity disagree")
    expected_scenario_set_sha256 = BehaviorScenarioSetIdentity((self.scenario,)).sha256
    if self.single_route_scenario_set_sha256 != expected_scenario_set_sha256:
      raise ValueError("single-route scenario set identity disagrees with scenario")
    for value in (
      self.preparation_sha256,
      self.single_route_scenario_set_sha256,
      self.physical_profile_sha256,
      self.provisional_dynamics_sha256,
      self.plant_member_id,
      self.segmentation_config_sha256,
      self.metric_config_sha256,
      self.artifact_identity.composed_controller_artifact_sha256,
    ):
      if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("behavior route evaluation identity must be SHA-256")
    if type(self.windows) is not tuple:
      raise TypeError("behavior route metrics must be a tuple")
    keys = tuple((value.route_id, value.window_id) for value in self.windows)
    if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
      raise ValueError("behavior route metrics are not canonical")
    if any(value.route_id != self.scenario.route_id for value in self.windows):
      raise ValueError("behavior route metrics belong to a different scenario")

  def to_dict(self) -> dict[str, object]:
    return {
      "metricConfigSha256": self.metric_config_sha256,
      "artifactIdentity": self.artifact_identity.to_dict(),
      "physicalProfileSha256": self.physical_profile_sha256,
      "preparationSha256": self.preparation_sha256,
      "preparationSchemaVersion": self.preparation_schema_version,
      "provisionalDynamicsSha256": self.provisional_dynamics_sha256,
      "plantMemberId": self.plant_member_id,
      "scenario": self.scenario.to_dict(),
      "schemaVersion": self.schema_version,
      "segmentationConfigSha256": self.segmentation_config_sha256,
      "singleRouteScenarioSetSha256": self.single_route_scenario_set_sha256,
      "windows": [_metric_set_dict(value) for value in self.windows],
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())

  @property
  def sha256(self) -> str:
    return _sha256_text(self.to_json())


Publication = TypeVar("Publication")


class _SparseCursor[Publication]:
  """One monotonic canonical-index cursor over a sparse evidence plane."""

  __slots__ = ("_current", "_current_index", "_iterator", "_name", "_requested_index")

  def __init__(self, values: Iterator[Publication], name: str) -> None:
    self._iterator = values
    self._name = name
    self._current: Publication | None = None
    self._current_index = -1
    self._requested_index = -1

  def select(self, index: int, available: bool) -> Publication | None:
    if available != (index >= 0):
      raise BehaviorRouteEvaluationError(
        f"{self._name} availability and index disagree",
      )
    if index < self._requested_index:
      raise BehaviorRouteEvaluationError(f"{self._name} links are not monotonic")
    self._requested_index = index
    if index < 0:
      return None
    while self._current_index < index:
      try:
        self._current = next(self._iterator)
      except StopIteration as error:
        raise BehaviorRouteEvaluationError(
          f"{self._name} link references a missing publication",
        ) from error
      self._current_index += 1
    return self._current

  def finish(self) -> None:
    for _ in self._iterator:
      pass

  def close(self) -> None:
    close = getattr(self._iterator, "close", None)
    if close is not None:
      close()


@contextmanager
def _plane_iterator_scope(*values: object) -> Iterator[None]:
  """Close held plane generators before their owning reader can close."""
  failed = False
  try:
    yield
  except BaseException:
    failed = True
    raise
  finally:
    cleanup_error: BaseException | None = None
    for value in reversed(values):
      close = getattr(value, "close", None)
      if close is None:
        continue
      try:
        close()
      except BaseException as error:
        if cleanup_error is None:
          cleanup_error = error
    if cleanup_error is not None and not failed:
      raise cleanup_error


@dataclass(slots=True)
class _StreamRuntime:
  reader: RouteEvidenceStreamReader
  source: Any
  scenario: BehaviorScenarioProvenance
  recorded_source: BehaviorSourceIdentity
  car_params_bytes: bytes
  car_params: Any
  nominal_rack_mapping: Any
  mapping_model: VehicleModel
  physical_profile_sha256: str


@dataclass(frozen=True, slots=True)
class _StreamFrame:
  control: CanonicalBehaviorControlInput
  frame_input: ReplayFrameInput
  model_intent: SparseModelBehaviorIntent | None
  reference: BehaviorReferenceAtControl


@contextmanager
def _reader_scope(
  value: RouteEvidenceStreamReader | str | Path,
) -> Iterator[RouteEvidenceStreamReader]:
  if isinstance(value, RouteEvidenceStreamReader):
    yield value
    return
  with RouteEvidenceStreamReader(value) as reader:
    yield reader


def _runtime(
  reader: RouteEvidenceStreamReader,
  physical_profile: VehicleCalibrationProfile,
  provisional_dynamics: ProvisionalRackDynamics,
  interface_registry: Mapping[str, type] | None,
) -> _StreamRuntime:
  if not isinstance(physical_profile, VehicleCalibrationProfile):
    raise TypeError("behavior route evaluation requires a physical profile")
  if not isinstance(provisional_dynamics, ProvisionalRackDynamics):
    raise TypeError("behavior route evaluation requires provisional dynamics")
  source = reader.summary.source_identity
  scenario = behavior_scenario_provenance_from_route_source(
    source,
    reader.summary.sha256,
  )
  car_params_bytes = reader.read_car_params_bytes()
  try:
    car_params = decode_behavior_car_params(car_params_bytes)
    bundle, _, _ = build_detected_runtime_bundle(
      car_params=car_params,
      provisional_rack_dynamics=provisional_dynamics,
      interface_registry=interface_registry,
    )
    controller_profile = compose_controller_profile(
      physical_profile,
      bundle.seed_profile,
    )
  except Exception as error:
    raise BehaviorRouteEvaluationError(
      "route evidence cannot bind a verified physical runtime",
    ) from error
  if (
    source.vehicle_identity != bundle.vehicle_identity
    or physical_profile.vehicle_identity != bundle.vehicle_identity
    or controller_profile.vehicle_identity != bundle.vehicle_identity
  ):
    raise BehaviorRouteEvaluationError("route/profile/runtime vehicle identities differ")
  return _StreamRuntime(
    reader=reader,
    source=source,
    scenario=scenario,
    recorded_source=scenario.recorded_source,
    car_params_bytes=car_params_bytes,
    car_params=car_params,
    nominal_rack_mapping=bundle.nominal_rack_mapping,
    mapping_model=VehicleModel(car_params),
    physical_profile_sha256=_sha256_text(physical_profile.to_json()),
  )


def _invalid_reference(transport_delay_s: float) -> BehaviorReferenceAtControl:
  return BehaviorReferenceAtControl(
    model_publication_mono_time_ns=0,
    plan_time_now_s=0.0,
    physical_effect_plan_s=transport_delay_s,
    scalar_curvature_1pm=0.0,
    anchored_curvature_1pm=0.0,
    anchored_curvature_rate_1pm_s=0.0,
    anchored_curvature_accel_1pm_s2=0.0,
    desired_rack_angle_deg=0.0,
    desired_rack_rate_deg_s=0.0,
    desired_rack_accel_deg_s2=0.0,
    valid=False,
  )


def _stream_frames(
  runtime: _StreamRuntime,
  physical_profile: VehicleCalibrationProfile,
) -> Iterator[_StreamFrame]:
  reader = runtime.reader
  physical_values = reader.iter_physical_frames()
  control_values = reader.iter_control_witnesses()
  model_values = _SparseCursor(reader.iter_model_publications(), "model")
  torque_values = _SparseCursor(reader.iter_live_torque_parameters(), "live torque")
  delay_values = _SparseCursor(reader.iter_live_delays(), "live delay")
  maneuver_values = _SparseCursor(
    reader.iter_lateral_maneuver_plans(),
    "lateral maneuver",
  )
  previous_response_mono_ns: int | None = None
  previous_rate_deg_s = 0.0
  active_count = 0
  sentinel = object()
  index = 0
  with _plane_iterator_scope(
    physical_values,
    control_values,
    model_values,
    torque_values,
    delay_values,
    maneuver_values,
  ):
    while True:
      physical = next(physical_values, sentinel)
      witness = next(control_values, sentinel)
      if physical is sentinel or witness is sentinel:
        if physical is not sentinel or witness is not sentinel:
          raise BehaviorRouteEvaluationError(
            "physical/control evidence populations disagree",
          )
        break
      assert isinstance(witness, ControlsWitness)
      gap_ns = (
        0
        if previous_response_mono_ns is None
        else physical.response_mono_ns - previous_response_mono_ns
      )
      acceleration_valid = (
        previous_response_mono_ns is not None
        and 0 < gap_ns <= _RACK_ACCELERATION_MAXIMUM_GAP_NS
      )
      acceleration = (
        (physical.steering_rate_deg_s - previous_rate_deg_s) / (gap_ns * 1e-9)
        if acceleration_valid
        else 0.0
      )
      previous_response_mono_ns = physical.response_mono_ns
      previous_rate_deg_s = physical.steering_rate_deg_s
      model = model_values.select(
        witness.model_publication_index,
        witness.model_link_valid,
      )
      live_torque = torque_values.select(
        witness.live_torque_parameters_index,
        witness.live_torque_parameters_available,
      )
      live_delay = delay_values.select(
        witness.live_delay_index,
        witness.live_delay_available,
      )
      maneuver = maneuver_values.select(
        witness.lateral_maneuver_plan_index,
        witness.maneuver_plan_available,
      )
      try:
        control, frame_input, model_intent = build_canonical_behavior_frame(
          index=index,
          source=runtime.source,
          car_params=runtime.car_params,
          car_params_bytes=runtime.car_params_bytes,
          nominal_rack_mapping=runtime.nominal_rack_mapping,
          mapping_model=runtime.mapping_model,
          witness=witness,
          physical=physical,
          rack_acceleration_deg_s2=acceleration,
          rack_acceleration_valid=acceleration_valid,
          model=model,
          live_torque=live_torque,
          live_delay=live_delay,
          maneuver=maneuver,
          scenario_only=True,
        )
      except BehaviorReplayError as error:
        raise BehaviorRouteEvaluationError(str(error)) from error
      except (TypeError, ValueError) as error:
        raise BehaviorRouteEvaluationError(
          "canonical behavior frame reconstruction failed",
        ) from error
      if witness.lateral_active:
        active_count += 1
        if model_intent is None:
          raise BehaviorRouteEvaluationError(
            "active lateral scenario lacks exact model intent",
          )
      neutral = neutral_behavior_response(control, physical_profile)
      reference = (
        _invalid_reference(neutral.transport_delay_s)
        if model_intent is None
        else derive_behavior_reference(model_intent, neutral)
      )
      yield _StreamFrame(control, frame_input, model_intent, reference)
      index += 1
    model_values.finish()
    torque_values.finish()
    delay_values.finish()
    maneuver_values.finish()
  if index == 0:
    raise BehaviorRouteEvaluationError("route evidence contains no control samples")
  if active_count == 0:
    raise BehaviorRouteEvaluationError("route evidence has no active lateral scenario")


def _event_locators(
  reader: RouteEvidenceStreamReader,
  maximum: int,
) -> tuple[EventLocator, ...]:
  values: list[EventLocator] = []
  for value in reader.iter_event_locators():
    if not value.message_valid:
      continue
    if len(values) >= maximum:
      raise BehaviorRouteEvaluationError("route exceeds maximum event-locator work")
    values.append(EventLocator(
        event_type=value.event_type,
        occurred_mono_time_ns=value.occurred_mono_time_ns,
        analysis_window_before_s=value.analysis_window_before_s,
        analysis_window_after_s=value.analysis_window_after_s,
        severity=value.severity,
    ))
  return tuple(sorted(
    values,
    key=lambda value: (
      value.occurred_mono_time_ns,
      value.event_type,
      value.severity,
    ),
  ))


def _target_sample(
  frame: _StreamFrame,
  physical_profile: VehicleCalibrationProfile,
) -> BehaviorSample:
  neutral = neutral_behavior_response(frame.control, physical_profile)
  reference = frame.reference
  return assemble_behavior_sample(
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
      planned_requested_torque=0.0,
      reachable_envelope_torque=0.0,
      envelope_applied_torque=0.0,
      torque_headroom=1.0,
      actuator_constrained=False,
      steering_request_active=False,
      maximum_authority_required=False,
      lateral_active=neutral.lateral_active,
      inputs_valid=neutral.inputs_valid,
      steering_pressed=neutral.steering_pressed,
      controller_fault=neutral.controller_fault,
      driver_intervention_onset=neutral.driver_intervention_onset,
    ),
  )


def _update_reference_digest(digest: Any, sample: BehaviorSample) -> None:
  encoded = canonical_json(sample.to_dict()).encode("utf-8")
  digest.update(len(encoded).to_bytes(8, "little"))
  digest.update(encoded)


def prepare_behavior_route_scenario(
  evidence: RouteEvidenceStreamReader | str | Path,
  physical_profile: VehicleCalibrationProfile,
  provisional_dynamics: ProvisionalRackDynamics,
  segmentation_config: SegmentationConfig,
  *,
  interface_registry: Mapping[str, type] | None = None,
) -> BehaviorRoutePreparation:
  """Authenticate and freeze one route's reference-only physical phases."""
  if not isinstance(segmentation_config, SegmentationConfig):
    raise TypeError("behavior route preparation requires segmentation config")
  try:
    with _reader_scope(evidence) as reader:
      runtime = _runtime(
        reader,
        physical_profile,
        provisional_dynamics,
        interface_registry,
      )
      events = _event_locators(reader, segmentation_config.maximum_event_locators)
      reference_digest = hashlib.sha256()
      with BehaviorSampleScratch() as scratch:
        sample_count = 0
        for frame in _stream_frames(runtime, physical_profile):
          sample = _target_sample(frame, physical_profile)
          _update_reference_digest(reference_digest, sample)
          scratch.append(sample)
          sample_count += 1
        samples = scratch.finish()
        segmentation = segment_file_backed_behavior_route(
          runtime.source.route_id,
          runtime.recorded_source,
          samples,
          events,
          segmentation_config,
        )
        spans = tuple(window.descriptor for window in segmentation.windows)
        ranges = _unassigned_ranges(sample_count, spans)
        if sum(end - start for start, end in ranges) != len(
          segmentation.unassigned_sample_indices
        ):
          raise BehaviorRouteEvaluationError(
            "compact and canonical unassigned segmentation differ",
          )
        return BehaviorRoutePreparation(
          schema_version=BEHAVIOR_ROUTE_PREPARATION_SCHEMA_VERSION,
          scenario=runtime.scenario,
          physical_profile_sha256=runtime.physical_profile_sha256,
          provisional_dynamics_sha256=provisional_dynamics.identity_sha256,
          segmentation_config_sha256=segmentation.config_sha256,
          reference_samples_sha256=reference_digest.hexdigest(),
          file_backed_segmentation_sha256=segmentation.sha256,
          sample_count=sample_count,
          spans=spans,
          event_coverage=segmentation.event_coverage,
          unassigned_sample_ranges=ranges,
        )
  except (RouteEvidenceError, BehaviorReplayError, ValueError) as error:
    if isinstance(error, BehaviorRouteEvaluationError):
      raise
    raise BehaviorRouteEvaluationError("behavior route preparation failed") from error


def _response_sample(
  frame: _StreamFrame,
  output: ControllerFrameOutput,
  physical_profile: VehicleCalibrationProfile,
) -> BehaviorSample:
  if output.mono_time_ns != frame.control.mono_time_ns:
    raise BehaviorRouteEvaluationError("controller replay output timeline mismatch")
  neutral = neutral_behavior_response(frame.control, physical_profile)
  return assemble_behavior_sample(
    frame.reference,
    BehaviorControlResponse(
      mono_time_ns=frame.control.mono_time_ns,
      route_time_s=frame.control.route_time_s,
      speed_mps=frame.control.speed_mps,
      transport_delay_s=neutral.transport_delay_s,
      live_rack_mapping=frame.control.live_rack_mapping,
      nominal_rack_mapping=frame.control.nominal_rack_mapping,
      measured_curvature_1pm=output.measured_curvature_1pm,
      measured_rack_angle_deg=output.measured_rack_angle_deg,
      measured_rack_rate_deg_s=output.measured_rack_rate_deg_s,
      measured_rack_accel_deg_s2=output.measured_rack_accel_deg_s2,
      raw_requested_torque=output.raw_requested_torque,
      planned_requested_torque=output.planned_requested_torque,
      reachable_envelope_torque=output.reachable_envelope_torque,
      envelope_applied_torque=output.envelope_applied_torque,
      torque_headroom=output.torque_headroom,
      actuator_constrained=output.actuator_constrained,
      steering_request_active=output.steering_request_active,
      maximum_authority_required=output.maximum_authority_required,
      lateral_active=frame.control.lateral_active,
      inputs_valid=neutral.inputs_valid and output.response_eligible,
      steering_pressed=frame.control.steering_pressed,
      controller_fault=frame.control.platform_fault or output.controller_fault,
      driver_intervention_onset=frame.control.driver_intervention_onset,
    ),
  )


def _evaluate_prepared_behavior_route_with_registry_for_test(
  evidence: RouteEvidenceStreamReader | str | Path,
  preparation: BehaviorRoutePreparation,
  physical_profile: VehicleCalibrationProfile,
  provisional_dynamics: ProvisionalRackDynamics,
  policy: BehaviorPolicy | None,
  metric_config: BehaviorMetricConfig,
  *,
  opponent_role: ReplayRole,
  core_identity: ReplayCoreIdentity,
  plant_member: CounterfactualPlantMember | None = None,
  interface_registry: Mapping[str, type] | None = None,
) -> BehaviorRouteEvaluation:
  """Replay one opponent with an explicitly injected test interface registry."""
  _, evaluations = _evaluate_behavior_route_policies_with_registry_for_test(
    evidence,
    preparation,
    physical_profile,
    provisional_dynamics,
    (policy,),
    metric_config,
    opponent_roles=(opponent_role,),
    core_identities=(core_identity,),
    plant_member=plant_member,
    interface_registry=interface_registry,
  )
  return evaluations[0]


def _validate_policy_population(
  policies: tuple[BehaviorPolicy | None, ...],
  roles: tuple[ReplayRole, ...],
  cores: tuple[ReplayCoreIdentity, ...],
) -> None:
  if not 0 < len(policies) <= _MAXIMUM_ROUTE_POLICIES:
    raise ValueError("behavior route policy population is outside its bound")
  if len(roles) != len(policies) or len(cores) != len(policies):
    raise ValueError("behavior route opponent populations disagree")
  for policy, role, core in zip(policies, roles, cores, strict=True):
    if policy is not None and not isinstance(policy, BehaviorPolicy):
      raise TypeError("behavior route policy has the wrong type")
    if not isinstance(role, ReplayRole):
      raise TypeError("behavior route requires an explicit opponent role")
    if not isinstance(core, ReplayCoreIdentity):
      raise TypeError("behavior route requires replay core identities")
    if (policy is None) != (role is ReplayRole.EXACT_STOCK):
      raise ValueError("only exact stock may replay without a behavior policy")
    if role is ReplayRole.EXACT_STOCK:
      validate_reviewed_replay_core_identity(core, exact_stock=True)


def _preparation_from_reference(
  runtime: _StreamRuntime,
  physical_profile: VehicleCalibrationProfile,
  provisional_dynamics: ProvisionalRackDynamics,
  segmentation_config: SegmentationConfig,
  reference_digest: Any,
  reference_scratch: BehaviorSampleScratch,
  sample_count: int,
  events: tuple[EventLocator, ...],
) -> BehaviorRoutePreparation:
  samples = reference_scratch.finish()
  segmentation = segment_file_backed_behavior_route(
    runtime.source.route_id,
    runtime.recorded_source,
    samples,
    events,
    segmentation_config,
  )
  spans = tuple(window.descriptor for window in segmentation.windows)
  ranges = _unassigned_ranges(sample_count, spans)
  if sum(end - start for start, end in ranges) != len(segmentation.unassigned_sample_indices):
    raise BehaviorRouteEvaluationError("compact and canonical unassigned segmentation differ")
  return BehaviorRoutePreparation(
    schema_version=BEHAVIOR_ROUTE_PREPARATION_SCHEMA_VERSION,
    scenario=runtime.scenario,
    physical_profile_sha256=runtime.physical_profile_sha256,
    provisional_dynamics_sha256=provisional_dynamics.identity_sha256,
    segmentation_config_sha256=segmentation.config_sha256,
    reference_samples_sha256=reference_digest.hexdigest(),
    file_backed_segmentation_sha256=segmentation.sha256,
    sample_count=sample_count,
    spans=spans,
    event_coverage=segmentation.event_coverage,
    unassigned_sample_ranges=ranges,
  )


def _evaluate_behavior_route_policies_with_registry_for_test(
  evidence: RouteEvidenceStreamReader | str | Path,
  preparation: BehaviorRoutePreparation | None,
  physical_profile: VehicleCalibrationProfile,
  provisional_dynamics: ProvisionalRackDynamics,
  policies: tuple[BehaviorPolicy | None, ...],
  metric_config: BehaviorMetricConfig,
  *,
  opponent_roles: tuple[ReplayRole, ...],
  core_identities: tuple[ReplayCoreIdentity, ...],
  segmentation_config: SegmentationConfig | None = None,
  plant_member: CounterfactualPlantMember | None = None,
  interface_registry: Mapping[str, type] | None = None,
) -> tuple[BehaviorRoutePreparation, tuple[BehaviorRouteEvaluation, ...]]:
  """Stream one route once while evaluating a bounded opponent population."""
  _validate_policy_population(policies, opponent_roles, core_identities)
  if preparation is None and not isinstance(segmentation_config, SegmentationConfig):
    raise TypeError("new route preparation requires segmentation config")
  if preparation is not None and not isinstance(preparation, BehaviorRoutePreparation):
    raise TypeError("behavior route evaluation requires a preparation")
  if not isinstance(metric_config, BehaviorMetricConfig):
    raise TypeError("behavior route evaluation requires metric config")
  try:
    with _reader_scope(evidence) as reader:
      runtime = _runtime(reader, physical_profile, provisional_dynamics, interface_registry)
      if preparation is not None and (
        preparation.scenario != runtime.scenario
        or preparation.physical_profile_sha256 != runtime.physical_profile_sha256
        or preparation.provisional_dynamics_sha256 != provisional_dynamics.identity_sha256
      ):
        raise BehaviorRouteEvaluationError("route evidence/profile differs from frozen preparation")
      if preparation is None and segmentation_config is not None:
        events = _event_locators(reader, segmentation_config.maximum_event_locators)
      else:
        for _ in reader.iter_event_locators():
          pass
        events = ()
      reference_digest = hashlib.sha256()
      steppers: tuple[BehaviorReplayStepper, ...] | None = None
      with ExitStack() as stack:
        reference_scratch = (
          stack.enter_context(BehaviorSampleScratch()) if preparation is None else None
        )
        response_scratches = tuple(
          stack.enter_context(BehaviorSampleScratch()) for _ in policies
        )
        sample_count = 0
        for frame in _stream_frames(runtime, physical_profile):
          target = _target_sample(frame, physical_profile)
          _update_reference_digest(reference_digest, target)
          if reference_scratch is not None:
            reference_scratch.append(target)
          if steppers is None:
            steppers = tuple(
              BehaviorReplayStepper(
                vehicle_identity=runtime.source.vehicle_identity,
                physical_profile=physical_profile,
                policy=policy,
                first_frame_input=frame.frame_input,
                nominal_rack_mapping=runtime.nominal_rack_mapping,
                provisional_dynamics=provisional_dynamics,
                plant_member=plant_member,
                interface_registry=interface_registry,
              )
              for policy in policies
            )
          for stepper, scratch in zip(steppers, response_scratches, strict=True):
            output = stepper.step(
              control=frame.control,
              frame_input=frame.frame_input,
              model_intent=frame.model_intent,
              reference=frame.reference,
            )
            if output.controller_fault:
              raise BehaviorRouteEvaluationError(
                "controller fault invalidates the complete route opponent",
              )
            scratch.append(_response_sample(frame, output, physical_profile))
          sample_count += 1
        if steppers is None:
          raise BehaviorRouteEvaluationError("route evidence contains no replay frames")
        if preparation is None:
          assert reference_scratch is not None and segmentation_config is not None
          preparation = _preparation_from_reference(
            runtime,
            physical_profile,
            provisional_dynamics,
            segmentation_config,
            reference_digest,
            reference_scratch,
            sample_count,
            events,
          )
        elif (
          sample_count != preparation.sample_count
          or reference_digest.hexdigest() != preparation.reference_samples_sha256
        ):
          raise BehaviorRouteEvaluationError("route reference differs from frozen preparation")
        evaluations: list[BehaviorRouteEvaluation] = []
        for policy, role, core, scratch in zip(
          policies, opponent_roles, core_identities, response_scratches, strict=True,
        ):
          samples = scratch.finish()
          windows = tuple(
            FileBackedBehaviorWindow(
              route_id=runtime.source.route_id,
              source=runtime.recorded_source,
              descriptor=span,
              samples=BehaviorSampleSpan(
                samples,
                span.start_sample_index,
                span.end_sample_index_exclusive,
              ),
            )
            for span in preparation.spans
          )
          metrics = retain_route_metric_windows(
            (score_file_backed_window(window, metric_config) for window in windows),
            metric_config,
          )
          evaluations.append(BehaviorRouteEvaluation(
            schema_version=BEHAVIOR_ROUTE_EVALUATION_SCHEMA_VERSION,
            preparation_sha256=preparation.sha256,
            preparation_schema_version=preparation.schema_version,
            scenario=runtime.scenario,
            single_route_scenario_set_sha256=BehaviorScenarioSetIdentity(
              (runtime.scenario,),
            ).sha256,
            artifact_identity=ReplayArtifactIdentity.compose(role, core, policy),
            physical_profile_sha256=runtime.physical_profile_sha256,
            provisional_dynamics_sha256=provisional_dynamics.identity_sha256,
            plant_member_id=steppers[0].plant_member.member_id,
            segmentation_config_sha256=preparation.segmentation_config_sha256,
            metric_config_sha256=metric_config.sha256,
            windows=metrics,
          ))
        return preparation, tuple(evaluations)
  except (RouteEvidenceError, BehaviorReplayError, ValueError) as error:
    if isinstance(error, BehaviorRouteEvaluationError):
      raise
    raise BehaviorRouteEvaluationError("behavior route evaluation failed") from error


def evaluate_behavior_route_policies(
  evidence: RouteEvidenceStreamReader | str | Path,
  preparation: BehaviorRoutePreparation | None,
  physical_profile: VehicleCalibrationProfile,
  provisional_dynamics: ProvisionalRackDynamics,
  policies: tuple[BehaviorPolicy | None, ...],
  metric_config: BehaviorMetricConfig,
  *,
  opponent_roles: tuple[ReplayRole, ...],
  core_identities: tuple[ReplayCoreIdentity, ...],
  plant_member: CounterfactualPlantMember,
  segmentation_config: SegmentationConfig | None = None,
) -> tuple[BehaviorRoutePreparation, tuple[BehaviorRouteEvaluation, ...]]:
  """Production one-scan route-major replay for a bounded policy population."""
  return _evaluate_behavior_route_policies_with_registry_for_test(
    evidence,
    preparation,
    physical_profile,
    provisional_dynamics,
    policies,
    metric_config,
    opponent_roles=opponent_roles,
    core_identities=core_identities,
    plant_member=plant_member,
    segmentation_config=segmentation_config,
    interface_registry=None,
  )


def evaluate_prepared_behavior_route(
  evidence: RouteEvidenceStreamReader | str | Path,
  preparation: BehaviorRoutePreparation,
  physical_profile: VehicleCalibrationProfile,
  provisional_dynamics: ProvisionalRackDynamics,
  policy: BehaviorPolicy | None,
  metric_config: BehaviorMetricConfig,
  *,
  opponent_role: ReplayRole,
  core_identity: ReplayCoreIdentity,
  plant_member: CounterfactualPlantMember,
) -> BehaviorRouteEvaluation:
  """Replay one exact opponent through the detected production interface.

  Interface-registry injection is deliberately absent from this authority
  boundary.  Exact-stock behavior includes the detected port's torque
  conversion and controller limits, so substituting a caller-owned interface
  would make the core digest describe code that did not produce the row.
  """
  return _evaluate_prepared_behavior_route_with_registry_for_test(
    evidence,
    preparation,
    physical_profile,
    provisional_dynamics,
    policy,
    metric_config,
    opponent_role=opponent_role,
    core_identity=core_identity,
    plant_member=plant_member,
    interface_registry=None,
  )


def evaluate_behavior_route(
  evidence: RouteEvidenceStreamReader | str | Path,
  physical_profile: VehicleCalibrationProfile,
  provisional_dynamics: ProvisionalRackDynamics,
  policy: BehaviorPolicy | None,
  segmentation_config: SegmentationConfig,
  metric_config: BehaviorMetricConfig,
  *,
  opponent_role: ReplayRole,
  core_identity: ReplayCoreIdentity,
  plant_member: CounterfactualPlantMember,
) -> BehaviorRouteEvaluation:
  """Prepare and evaluate one route while keeping both passes bounded."""
  with _reader_scope(evidence) as reader:
    preparation = prepare_behavior_route_scenario(
      reader,
      physical_profile,
      provisional_dynamics,
      segmentation_config,
    )
    return evaluate_prepared_behavior_route(
      reader,
      preparation,
      physical_profile,
      provisional_dynamics,
      policy,
      metric_config,
      opponent_role=opponent_role,
      core_identity=core_identity,
      plant_member=plant_member,
    )

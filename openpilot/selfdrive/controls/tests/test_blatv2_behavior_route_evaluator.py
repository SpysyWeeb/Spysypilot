from __future__ import annotations

from dataclasses import replace
import gc
from pathlib import Path
import tempfile
import tracemalloc
from unittest.mock import patch
import warnings

from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import (
  BehaviorControlResponse,
  BehaviorWindow,
  assemble_behavior_sample,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_metrics import (
  BehaviorMetricConfig,
  score_window,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_coordinator import ReplayRole
from openpilot.selfdrive.controls.lib.blatv2.behavior_replay import (
  BehaviorReplayError,
  make_behavior_scenario_route_evidence_decoder,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_route_evaluator import (
  BehaviorRouteEvaluationError,
  _evaluate_prepared_behavior_route_with_registry_for_test as evaluate_prepared_behavior_route,
  prepare_behavior_route_scenario,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_segmentation import (
  SegmentationConfig,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_transaction import (
  _prepare_route,
  neutral_behavior_response,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill_spool import (
  _encode_frame,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_runtime import (
  MeasuredLearningFrame,
)
from openpilot.selfdrive.controls.lib.blatv2.route_evidence import (
  ControlsWitness,
  DrivingEventLocator,
  LateralManeuverPlanPublication,
  ModelPublication,
  RouteEvidenceArtifact,
  RouteEvidenceError,
  RouteEvidenceSourceIdentity,
  RouteEvidenceStreamReader,
)
from openpilot.selfdrive.controls.tests.test_blatv2_behavior_replay import (
  BASE_NS,
  FINGERPRINT,
  INTERFACES,
  behavior_policy,
  core_identity,
  modular_core,
  physical_profile,
  provisional,
  reviewed_core_identity,
  request,
  stock_core,
  synthetic_cp,
)


def _curve(index: int, count: int) -> float:
  scale = count / 180.0
  straight_end = max(2, round(10 * scale))
  turn_end = max(straight_end + 3, round(50 * scale))
  hold_end = max(turn_end + 3, round(100 * scale))
  release_end = max(hold_end + 3, round(140 * scale))
  if index < straight_end:
    return 0.0
  if index < turn_end:
    return 0.03 * (index - straight_end + 1) / (turn_end - straight_end)
  if index < hold_end:
    return 0.03
  if index < release_end:
    return 0.03 * (release_end - index - 1) / (release_end - hold_end)
  return 0.0


def _artifact(
  count: int = 180,
  *,
  raw_request_offset: float = 0.0,
) -> RouteEvidenceArtifact:
  cp_bytes = synthetic_cp().to_bytes()
  physical_frames = tuple(
    MeasuredLearningFrame(
      sample_mono_ns=BASE_NS + index * 10_000_000 - 2_000_000,
      response_mono_ns=BASE_NS + index * 10_000_000 - 1_000_000,
      applied_report_mono_ns=BASE_NS + index * 10_000_000 - 3_000_000,
      applied_effective_mono_ns=BASE_NS + index * 10_000_000 - 13_000_000,
      speed_mps=8.0,
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
      lateral_active=index > 0,
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
    for index in range(count)
  )
  times = tuple(sample * 0.05 for sample in range(20))
  models = tuple(
    ModelPublication(
      segment_index=0,
      ordinal=index,
      mono_time_ns=BASE_NS + index * 10_000_000 - 5_000_000,
      frame_id=100 + index,
      timestamp_eof_ns=BASE_NS + index * 10_000_000 - 6_000_000,
      scalar_curvature=_curve(index, count),
      desired_curvature_time_s=0.25,
      plan_times=times,
      orientation_rate_z=tuple(_curve(index, count) * 8.0 for _ in times),
      velocity_x=tuple(8.0 for _ in times),
      message_valid=True,
      native_grid_valid=True,
    )
    for index in range(count)
  )
  witnesses = tuple(
    ControlsWitness(
      segment_index=0,
      ordinal=index,
      mono_time_ns=BASE_NS + index * 10_000_000,
      physical_record_index=index,
      model_publication_index=index,
      live_torque_parameters_index=-1,
      live_delay_index=-1,
      lateral_maneuver_plan_index=-1,
      poll_mono_time_ns=BASE_NS + index * 10_000_000 - 1_000_000,
      state_sample_mono_ns=BASE_NS + index * 10_000_000 - 2_000_000,
      live_parameters_mono_ns=BASE_NS + index * 10_000_000 - 2_500_000,
      car_output_report_mono_ns=BASE_NS + index * 10_000_000 - 3_000_000,
      car_output_effective_mono_ns=BASE_NS + index * 10_000_000 - 13_000_000,
      car_control_mono_ns=BASE_NS + index * 10_000_000,
      raw_request_torque=raw_request_offset + (index % 17 - 8) / 10.0,
      measured_curvature=0.0,
      desired_curvature=_curve(index, count),
      envelope_headroom=1.0,
      torque_output_can_count=0,
      message_valid=True,
      model_message_alive=True,
      model_link_valid=True,
      inputs_valid=True,
      lateral_active=index > 0,
      driver_intervening=False,
      steer_fault=False,
      intervention_onset=False,
      intervention_onset_uncertain=False,
      race_unresolved=False,
      gap_from_previous=False,
      car_control_paired=True,
      torque_output_can_valid=True,
      maneuver_plan_available=False,
      live_torque_parameters_available=False,
      live_delay_available=False,
      live_torque_parameters_checks_passed=False,
      live_torque_parameters_health_exact=True,
    )
    for index in range(count)
  )
  source = RouteEvidenceSourceIdentity(
    route_id="synthetic-stream-evaluation",
    route_time_origin_mono_ns=BASE_NS,
    route_segment_sha256=("e" * 64,),
    route_segment_size_bytes=(1234,),
    source_superproject_commit="1" * 40,
    source_opendbc_commit="2" * 40,
    source_panda_commit="3" * 40,
    controller_source_kind="ineligible",
    controller_artifact_sha256="4" * 64,
    behavior_eligible=False,
    behavior_ineligible_reason="unverified_controller_source",
    vehicle_identity=FINGERPRINT,
    runtime_identity="5" * 64,
    schema_versions={"extractor": 3, "route_evidence": 2},
    preparation_provenance={"canonical": True},
    physical_plane_encoding_id="blatv2-measured-learning-frame-v1",
    physical_record_count=count,
    preparation_cache_key="6" * 64,
    controls_witness_count=count,
    unresolved_witness_count=0,
    gap_count=0,
    model_link_failure_count=0,
  )
  return RouteEvidenceArtifact(
    source,
    cp_bytes,
    b"".join(_encode_frame(frame) for frame in physical_frames),
    models,
    witnesses,
    live_torque_parameters=(),
    live_delays=(),
    lateral_maneuver_plans=(),
    event_locators=(DrivingEventLocator(
      0,
      0,
      BASE_NS + count * 10_000_000,
      BASE_NS + round(count * 0.55) * 10_000_000,
      1.0,
      1.0,
      "event-1",
      "lat.turnStopTurn",
      "warning",
      True,
    ),),
  )


def _write(tmp_path: Path, artifact: RouteEvidenceArtifact, name: str = "route.evidence") -> Path:
  path = tmp_path / name
  path.write_bytes(artifact.canonical_bytes)
  return path


def _segmentation_config() -> SegmentationConfig:
  return SegmentationConfig.provisional_offline_gate()


def _metric_config() -> BehaviorMetricConfig:
  return BehaviorMetricConfig(
    burst_window_s=1.0,
    chatter_torque_rate_threshold_per_s=0.5,
    turn_in_crossing_fraction=0.5,
    release_crossing_fraction=0.9,
    correction_curvature_threshold_1pm=0.001,
    unused_headroom_threshold=0.2,
    growing_error_epsilon_1pm=1e-6,
    completion_delivered_fraction=0.9,
    minimum_samples=2,
    speed_nodes_mps=(0.0, 8.0, 16.0, 30.0),
    maximum_route_windows_per_stratum=20,
  )


def _eager_metrics(
  path: Path,
  *,
  modular: bool,
) -> tuple[str, tuple]:
  artifact = RouteEvidenceArtifact.from_file(path)
  decoder = make_behavior_scenario_route_evidence_decoder(
    provisional_dynamics=provisional(),
    interface_registry=INTERFACES,
  )
  route = decoder(artifact, physical_profile())
  prepared = _prepare_route(
    route,
    physical_profile(),
    _segmentation_config(),
    lambda: False,
  )
  replay_request = request(route, modular=modular)
  assert replay_request.references == prepared.references
  outputs = tuple(
    (modular_core() if modular else stock_core()).replay_route(replay_request)
  )
  samples = []
  for control, reference, output in zip(
    route.control_inputs,
    prepared.references,
    outputs,
    strict=True,
  ):
    neutral = neutral_behavior_response(control, physical_profile())
    samples.append(assemble_behavior_sample(
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
  metrics = tuple(
    score_window(
      BehaviorWindow(
        route_id=route.route_id,
        window_id=value.window.window_id,
        source=route.recorded_source,
        maneuver_class=value.window.maneuver_class,
        phase=value.window.phase,
        samples=tuple(samples[
          value.start_sample_index:value.end_sample_index_exclusive
        ]),
        event_locators=value.window.event_locators,
      ),
      _metric_config(),
    )
    for value in prepared.segmentation.windows
  )
  return prepared.segmentation.sha256, metrics


def test_streamed_stock_and_modular_match_eager_windows_exactly(tmp_path: Path) -> None:
  path = _write(tmp_path, _artifact())
  preparation = prepare_behavior_route_scenario(
    path,
    physical_profile(),
    provisional(),
    _segmentation_config(),
    interface_registry=INTERFACES,
  )
  assert preparation.spans

  for modular in (False, True):
    expected_segmentation_sha, expected_metrics = _eager_metrics(path, modular=modular)
    result = evaluate_prepared_behavior_route(
      path,
      preparation,
      physical_profile(),
      provisional(),
      behavior_policy() if modular else None,
      _metric_config(),
      opponent_role=(ReplayRole.CANDIDATE if modular else ReplayRole.EXACT_STOCK),
      core_identity=(
        core_identity("modular candidate", "c")
        if modular
        else reviewed_core_identity(modular=False)
      ),
      interface_registry=INTERFACES,
    )
    assert preparation.file_backed_segmentation_sha256 == expected_segmentation_sha
    assert result.windows == expected_metrics


def test_result_identity_binds_exact_stock_core_and_dynamics(tmp_path: Path) -> None:
  path = _write(tmp_path, _artifact())
  dynamics = provisional()
  preparation = prepare_behavior_route_scenario(
    path,
    physical_profile(),
    dynamics,
    _segmentation_config(),
    interface_registry=INTERFACES,
  )
  first = evaluate_prepared_behavior_route(
    path,
    preparation,
    physical_profile(),
    dynamics,
    None,
    _metric_config(),
    opponent_role=ReplayRole.EXACT_STOCK,
    core_identity=reviewed_core_identity(modular=False),
    interface_registry=INTERFACES,
  )
  try:
    evaluate_prepared_behavior_route(
      path,
      preparation,
      physical_profile(),
      dynamics,
      None,
      _metric_config(),
      opponent_role=ReplayRole.EXACT_STOCK,
      core_identity=core_identity("exact stock", "e"),
      interface_registry=INTERFACES,
    )
  except BehaviorReplayError as error:
    assert "reviewed implementation" in str(error)
  else:
    raise AssertionError("caller identity relabeled an arbitrary row as exact stock")
  assert first.artifact_identity.role.value == "exact_stock"
  assert first.provisional_dynamics_sha256 == dynamics.identity_sha256
  assert preparation.provisional_dynamics_sha256 == dynamics.identity_sha256

  incumbent = evaluate_prepared_behavior_route(
    path,
    preparation,
    physical_profile(),
    dynamics,
    behavior_policy(),
    _metric_config(),
    opponent_role=ReplayRole.CURRENTLY_ACCEPTED,
    core_identity=core_identity("accepted modular", "a"),
    interface_registry=INTERFACES,
  )
  candidate = evaluate_prepared_behavior_route(
    path,
    preparation,
    physical_profile(),
    dynamics,
    behavior_policy(),
    _metric_config(),
    opponent_role=ReplayRole.CANDIDATE,
    core_identity=core_identity("candidate modular", "c"),
    interface_registry=INTERFACES,
  )
  assert incumbent.artifact_identity.role is ReplayRole.CURRENTLY_ACCEPTED
  assert candidate.artifact_identity.role is ReplayRole.CANDIDATE
  assert incumbent.windows == candidate.windows
  assert incumbent.sha256 != candidate.sha256

  for role, policy in (
    (ReplayRole.EXACT_STOCK, behavior_policy()),
    (ReplayRole.CURRENTLY_ACCEPTED, None),
    (ReplayRole.CANDIDATE, None),
  ):
    try:
      evaluate_prepared_behavior_route(
        path,
        preparation,
        physical_profile(),
        dynamics,
        policy,
        _metric_config(),
        opponent_role=role,
        core_identity=core_identity("invalid role pairing", "f"),
        interface_registry=INTERFACES,
      )
    except ValueError as error:
      assert "only exact stock" in str(error)
    else:
      raise AssertionError("opponent role accepted an invalid policy pairing")

  changed_dynamics = replace(dynamics, rack_gain_deg_s2_per_torque=1501.0)
  try:
    evaluate_prepared_behavior_route(
      path,
      preparation,
      physical_profile(),
      changed_dynamics,
      None,
      _metric_config(),
      opponent_role=ReplayRole.EXACT_STOCK,
      core_identity=reviewed_core_identity(modular=False),
      interface_registry=INTERFACES,
    )
  except BehaviorRouteEvaluationError as error:
    assert "frozen preparation" in str(error)
  else:
    raise AssertionError("changed provisional dynamics reused frozen preparation")


def test_recorded_request_mutation_cannot_change_either_opponent(tmp_path: Path) -> None:
  first_path = _write(tmp_path, _artifact(), "first.evidence")
  second_path = _write(
    tmp_path,
    _artifact(raw_request_offset=12345.0),
    "second.evidence",
  )
  preparations = tuple(
    prepare_behavior_route_scenario(
      path,
      physical_profile(),
      provisional(),
      _segmentation_config(),
      interface_registry=INTERFACES,
    )
    for path in (first_path, second_path)
  )
  assert preparations[0].reference_samples_sha256 == preparations[1].reference_samples_sha256
  assert preparations[0].file_backed_segmentation_sha256 == preparations[1].file_backed_segmentation_sha256
  for policy in (None, behavior_policy()):
    results = tuple(
      evaluate_prepared_behavior_route(
        path,
        preparation,
        physical_profile(),
        provisional(),
        policy,
        _metric_config(),
        opponent_role=(
          ReplayRole.EXACT_STOCK if policy is None else ReplayRole.CANDIDATE
        ),
        core_identity=(
          reviewed_core_identity(modular=False)
          if policy is None
          else core_identity("modular candidate", "c")
        ),
        interface_registry=INTERFACES,
      )
      for path, preparation in zip(
        (first_path, second_path),
        preparations,
        strict=True,
      )
    )
    assert results[0].windows == results[1].windows


def test_streaming_rejects_future_sparse_link_and_cleans_scratch(
  tmp_path: Path,
  monkeypatch,
) -> None:
  artifact = _artifact()
  models = tuple(
    replace(
      model,
      mono_time_ns=model.mono_time_ns + 20_000_000,
      timestamp_eof_ns=model.timestamp_eof_ns + 20_000_000,
    )
    for model in artifact.model_publications
  )
  future = RouteEvidenceArtifact(
    artifact.source_identity,
    bytes(artifact.car_params_bytes),
    bytes(artifact.physical_bytes),
    models,
    artifact.control_witnesses,
    artifact.live_torque_parameters,
    artifact.live_delays,
    artifact.lateral_maneuver_plans,
    artifact.event_locators,
  )
  path = _write(tmp_path, future)
  scratch_root = tmp_path / "scratch"
  scratch_root.mkdir()
  monkeypatch.setattr(tempfile, "tempdir", str(scratch_root))
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    try:
      prepare_behavior_route_scenario(
        path,
        physical_profile(),
        provisional(),
        _segmentation_config(),
        interface_registry=INTERFACES,
      )
    except BehaviorRouteEvaluationError as error:
      assert "reconstruction" in str(error)
    else:
      raise AssertionError("future sparse publication was accepted")
  assert tuple(scratch_root.iterdir()) == ()


def test_streaming_rejects_missing_sparse_publication(tmp_path: Path) -> None:
  path = _write(tmp_path, _artifact())
  original = RouteEvidenceStreamReader.iter_model_publications

  def missing_last(self):
    values = iter(original(self))
    previous = next(values)
    for value in values:
      yield previous
      previous = value

  with patch.object(RouteEvidenceStreamReader, "iter_model_publications", missing_last):
    try:
      prepare_behavior_route_scenario(
        path,
        physical_profile(),
        provisional(),
        _segmentation_config(),
        interface_registry=INTERFACES,
      )
    except BehaviorRouteEvaluationError as error:
      assert "missing publication" in str(error)
    else:
      raise AssertionError("missing sparse publication was accepted")


def test_active_invalid_witness_fails_closed_in_eager_and_streaming(
  tmp_path: Path,
) -> None:
  for invalid_field in ("message_valid", "inputs_valid"):
    artifact = _artifact()
    witnesses = list(artifact.control_witnesses)
    witnesses[20] = replace(witnesses[20], **{invalid_field: False})
    invalid = RouteEvidenceArtifact(
      artifact.source_identity,
      bytes(artifact.car_params_bytes),
      bytes(artifact.physical_bytes),
      artifact.model_publications,
      tuple(witnesses),
      artifact.live_torque_parameters,
      artifact.live_delays,
      artifact.lateral_maneuver_plans,
      artifact.event_locators,
    )
    decoder = make_behavior_scenario_route_evidence_decoder(
      provisional_dynamics=provisional(),
      interface_registry=INTERFACES,
    )
    try:
      decoder(invalid, physical_profile())
    except BehaviorReplayError as error:
      assert "invalid evidence" in str(error)
    else:
      raise AssertionError(f"active {invalid_field}=False was accepted eagerly")

    path = _write(tmp_path, invalid, f"{invalid_field}.evidence")
    try:
      prepare_behavior_route_scenario(
        path,
        physical_profile(),
        provisional(),
        _segmentation_config(),
        interface_registry=INTERFACES,
      )
    except BehaviorRouteEvaluationError as error:
      assert "invalid evidence" in str(error)
    else:
      raise AssertionError(f"active {invalid_field}=False was streamed")


def test_active_lateral_maneuver_override_is_not_behavior_scenario(
  tmp_path: Path,
) -> None:
  artifact = _artifact()
  witnesses = list(artifact.control_witnesses)
  witnesses[20] = replace(
    witnesses[20],
    lateral_maneuver_plan_index=0,
    maneuver_plan_available=True,
  )
  maneuver = LateralManeuverPlanPublication(
    segment_index=0,
    ordinal=0,
    mono_time_ns=witnesses[20].mono_time_ns - 1,
    desired_curvature=0.02,
    message_valid=True,
  )
  overridden = RouteEvidenceArtifact(
    artifact.source_identity,
    bytes(artifact.car_params_bytes),
    bytes(artifact.physical_bytes),
    artifact.model_publications,
    tuple(witnesses),
    artifact.live_torque_parameters,
    artifact.live_delays,
    (maneuver,),
    artifact.event_locators,
  )
  decoder = make_behavior_scenario_route_evidence_decoder(
    provisional_dynamics=provisional(),
    interface_registry=INTERFACES,
  )
  try:
    decoder(overridden, physical_profile())
  except BehaviorReplayError as error:
    assert "lateral maneuver plan override" in str(error)
  else:
    raise AssertionError("maneuver-injection route was admitted eagerly")

  path = _write(tmp_path, overridden, "maneuver-override.evidence")
  try:
    prepare_behavior_route_scenario(
      path,
      physical_profile(),
      provisional(),
      _segmentation_config(),
      interface_registry=INTERFACES,
    )
  except BehaviorRouteEvaluationError as error:
    assert "lateral maneuver plan override" in str(error)
  else:
    raise AssertionError("maneuver-injection route was streamed")


def test_successful_preparation_fully_consumes_every_evidence_plane(
  tmp_path: Path,
) -> None:
  path = _write(tmp_path, _artifact())

  class CountingReader(RouteEvidenceStreamReader):
    __slots__ = ("counts",)

    def __init__(self, route_path: Path) -> None:
      super().__init__(route_path)
      self.counts: dict[str, int] = {}

    def _count(self, name: str, values):
      for value in values:
        self.counts[name] = self.counts.get(name, 0) + 1
        yield value

    def iter_physical_frames(self):
      return self._count("physical", super().iter_physical_frames())

    def iter_control_witnesses(self):
      return self._count("controls", super().iter_control_witnesses())

    def iter_model_publications(self):
      return self._count("models", super().iter_model_publications())

    def iter_live_torque_parameters(self):
      return self._count("torque", super().iter_live_torque_parameters())

    def iter_live_delays(self):
      return self._count("delay", super().iter_live_delays())

    def iter_lateral_maneuver_plans(self):
      return self._count("maneuver", super().iter_lateral_maneuver_plans())

    def iter_event_locators(self):
      return self._count("events", super().iter_event_locators())

  reader = CountingReader(path)
  try:
    prepare_behavior_route_scenario(
      reader,
      physical_profile(),
      provisional(),
      _segmentation_config(),
      interface_registry=INTERFACES,
    )
    assert reader.counts == {
      "controls": 180,
      "events": 1,
      "models": 180,
      "physical": 180,
    }
  finally:
    reader.close()


def test_streaming_rejects_unresolved_inactive_and_wrong_vehicle(tmp_path: Path) -> None:
  artifact = _artifact()
  witnesses = list(artifact.control_witnesses)
  witnesses[20] = replace(witnesses[20], race_unresolved=True)
  unresolved_source = replace(artifact.source_identity, unresolved_witness_count=1)
  unresolved = RouteEvidenceArtifact(
    unresolved_source,
    bytes(artifact.car_params_bytes),
    bytes(artifact.physical_bytes),
    artifact.model_publications,
    tuple(witnesses),
    artifact.live_torque_parameters,
    artifact.live_delays,
    artifact.lateral_maneuver_plans,
    artifact.event_locators,
  )
  inactive = RouteEvidenceArtifact(
    artifact.source_identity,
    bytes(artifact.car_params_bytes),
    bytes(artifact.physical_bytes),
    artifact.model_publications,
    tuple(replace(value, lateral_active=False) for value in artifact.control_witnesses),
    artifact.live_torque_parameters,
    artifact.live_delays,
    artifact.lateral_maneuver_plans,
    artifact.event_locators,
  )
  for name, value, phrase in (
    ("unresolved", unresolved, "unresolved"),
    ("inactive", inactive, "no active lateral"),
  ):
    path = _write(tmp_path, value, f"{name}.evidence")
    try:
      prepare_behavior_route_scenario(
        path,
        physical_profile(),
        provisional(),
        _segmentation_config(),
        interface_registry=INTERFACES,
      )
    except BehaviorRouteEvaluationError as error:
      assert phrase in str(error)
    else:
      raise AssertionError(f"{name} scenario was accepted")

  path = _write(tmp_path, artifact, "wrong-profile.evidence")
  wrong = replace(physical_profile(), vehicle_identity="wrong-vehicle")
  try:
    prepare_behavior_route_scenario(
      path,
      wrong,
      provisional(),
      _segmentation_config(),
      interface_registry=INTERFACES,
    )
  except BehaviorRouteEvaluationError as error:
    assert "verified physical runtime" in str(error)
  else:
    raise AssertionError("wrong vehicle profile was accepted")


def test_corrupt_and_truncated_artifacts_fail_before_replay(tmp_path: Path) -> None:
  artifact = _artifact()
  for name, payload in (
    ("corrupt", artifact.canonical_bytes[:-1] + bytes([artifact.canonical_bytes[-1] ^ 1])),
    ("truncated", artifact.canonical_bytes[:-37]),
  ):
    path = tmp_path / f"{name}.evidence"
    path.write_bytes(payload)
    try:
      prepare_behavior_route_scenario(
        path,
        physical_profile(),
        provisional(),
        _segmentation_config(),
        interface_registry=INTERFACES,
      )
    except (BehaviorRouteEvaluationError, RouteEvidenceError):
      pass
    else:
      raise AssertionError(f"{name} route evidence was accepted")


def test_streaming_peak_memory_does_not_follow_route_sample_count(tmp_path: Path) -> None:
  peaks = []
  for count in (180, 1800):
    path = _write(tmp_path, _artifact(count), f"route-{count}.evidence")
    gc.collect()
    tracemalloc.start()
    preparation = prepare_behavior_route_scenario(
      path,
      physical_profile(),
      provisional(),
      _segmentation_config(),
      interface_registry=INTERFACES,
    )
    evaluate_prepared_behavior_route(
      path,
      preparation,
      physical_profile(),
      provisional(),
      None,
      _metric_config(),
      opponent_role=ReplayRole.EXACT_STOCK,
      core_identity=reviewed_core_identity(modular=False),
      interface_registry=INTERFACES,
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peaks.append(peak)
  assert peaks[1] <= peaks[0] + 2_500_000

from __future__ import annotations

from dataclasses import replace
import json
import math
import multiprocessing
import os
import signal
import threading
import time
import unittest
from unittest.mock import patch

import openpilot.selfdrive.controls.lib.blatv2.behavior_transaction as behavior_transaction
from openpilot.selfdrive.controls.lib.blatv2.behavior_coordinator import (
  BEHAVIOR_GATE_SPEC_SCHEMA_VERSION,
  BehaviorGateSpec,
  CandidateGridBounds,
  ReplayArtifactIdentity,
  ReplayCoreIdentity,
  ReplayRole,
  RoutePartitionSpec,
  partition_whole_routes,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import (
  BehaviorSourceIdentity,
  EventLocator,
  SparseModelBehaviorIntent,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_metrics import (
  BehaviorContract,
  BehaviorMetricConfig,
  BehaviorMetricName,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_policy import (
  PAIRED_ROUTE_UNCERTAINTY_METHOD,
  BehaviorPolicy,
  MetricGateRule,
  MetricPreference,
  build_candidate_grid,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_replay import (
  make_exact_stock_behavior_replay_core,
  make_modular_behavior_replay_core,
  reviewed_replay_core_identity,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_segmentation import SegmentationConfig
from openpilot.selfdrive.controls.lib.blatv2.behavior_transaction import (
  BehaviorLearningTransactionResult,
  BehaviorReplayCore,
  BehaviorReplayProgress,
  BehaviorReplayProgressPhase,
  BehaviorTransactionAborted,
  BehaviorTransactionError,
  CanonicalBehaviorControlInput,
  ControllerFrameOutput,
  ControllerReplayRequest,
  DecodedBehaviorRoute,
  QualificationDisposition,
  _run_behavior_learning_transaction_unchecked_for_test as run_behavior_learning_transaction,
  run_behavior_learning_transaction as run_reviewed_behavior_learning_transaction,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_profile import (
  CalibrationParameters,
  CalibrationProfileNode,
  VehicleCalibrationProfile,
)
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import RackMappingSnapshot
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import ProvisionalRackDynamics


def physical_profile(*, qualified: bool = True) -> VehicleCalibrationProfile:
  parameters = CalibrationParameters(
    torque_per_lateral_accel=0.4,
    lateral_accel_offset_correction_mps2=0.0,
    kinetic_friction_torque=0.03,
    static_breakaway_torque=0.09,
    transport_delay_s=0.0,
    rack_rate_resolution_deg_s=4.0,
    confidence=1.0 if qualified else 0.0,
    qualified=qualified,
  )
  nodes = tuple(
    CalibrationProfileNode(
      speed_mps=speed,
      parameters=parameters,
      base_support_s=600.0,
      base_sample_count=6_000,
      moving_support_s=300.0,
      moving_sample_count=3_000,
      breakaway_support_s=30.0,
      breakaway_sample_count=300,
      cross_fit_route_count=1_000,
      full_fit_candidate_rms=0.01,
      breakaway_full_fit_candidate_rms=0.01,
    )
    for speed in (0.0, 30.0)
  )
  return VehicleCalibrationProfile(
    vehicle_identity="test-car",
    revision=7,
    provenance="qualified transaction fixture",
    nodes=nodes,
  )


def rack_mapping() -> RackMappingSnapshot:
  return RackMappingSnapshot(
    mass_kg=2_200.0,
    wheelbase_m=2.9,
    center_to_front_m=1.2,
    center_to_rear_m=1.7,
    tire_stiffness_front=160_000.0,
    tire_stiffness_rear=200_000.0,
    steer_ratio_rear=0.0,
    steer_ratio=15.0,
    roll_rad=0.0,
    angle_offset_deg=0.0,
    valid=True,
  )


def source(token: str = "a") -> BehaviorSourceIdentity:
  return BehaviorSourceIdentity(
    controller_name="recorded-stock",
    controller_artifact_sha256=token * 64,
    source_openpilot_commit="b" * 40,
    opendbc_commit="c" * 40,
    panda_commit="d" * 40,
    evidence_schema_version=1,
  )


def replay_core_identity(name: str, token: str) -> ReplayCoreIdentity:
  return ReplayCoreIdentity(
    controller_name=name,
    core_artifact_sha256=token * 64,
    source_openpilot_commit="1" * 40,
    opendbc_commit="2" * 40,
    panda_commit="3" * 40,
  )


BASE_SHAPE = (
  0.0, 0.0, 0.0,
  0.001, 0.004, 0.008, 0.012,
  0.012, 0.012, 0.012,
  0.009, 0.005, 0.002,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
)


def decoded_route(
  index: int,
  *,
  recorded_source: BehaviorSourceIdentity | None = None,
  hard: bool = False,
  intervention: bool = False,
  incomplete_release: bool = False,
) -> DecodedBehaviorRoute:
  values = list(BASE_SHAPE)
  if incomplete_release:
    values.extend((0.001, 0.004, 0.008, 0.012, 0.0118, 0.0116))
  models = []
  controls = []
  base_ns = (index + 1) * 10_000_000_000
  mapping = rack_mapping()
  for sample_index, curvature in enumerate(values):
    mono_time_ns = base_ns + sample_index * 100_000_000
    models.append(SparseModelBehaviorIntent(
      plan_origin_mono_time_ns=mono_time_ns,
      publication_mono_time_ns=mono_time_ns,
      model_frame_id=sample_index,
      plan_valid=True,
      scalar_curvature_1pm=curvature,
      scalar_action_plan_s=0.0,
      native_times_s=(0.0, 1.0),
      orientation_rates_z=(curvature * 5.0, curvature * 5.0),
      velocities_x=(5.0, 5.0),
    ))
    is_release = sample_index in (10, 11, 12)
    if incomplete_release and sample_index >= len(BASE_SHAPE) + 3:
      is_release = True
    tag = b"release" if is_release else (b"hard" if hard else b"")
    onset = intervention and sample_index == 8
    controls.append(CanonicalBehaviorControlInput(
      mono_time_ns=mono_time_ns,
      route_time_s=sample_index * 0.1,
      speed_mps=5.0,
      model_publication_index=sample_index,
      live_rack_mapping=mapping,
      nominal_rack_mapping=mapping,
      core_input=tag,
      inputs_valid=True,
      lateral_active=True,
      steering_pressed=onset,
      platform_fault=False,
      driver_intervention_onset=onset,
    ))
  event = EventLocator(
    event_type="lat.turnStopTurn",
    occurred_mono_time_ns=base_ns + 700_000_000,
    analysis_window_before_s=0.5,
    analysis_window_after_s=0.5,
    severity="warning",
  )
  return DecodedBehaviorRoute(
    route_id=f"route-{index}",
    route_evidence_sha256=f"{index + 1:x}" * 64,
    vehicle_identity="test-car",
    recorded_source=source() if recorded_source is None else recorded_source,
    model_publications=tuple(models),
    control_inputs=tuple(controls),
    event_locators=(event,),
  )


def segmentation_config() -> SegmentationConfig:
  return SegmentationConfig(
    schema_version=1,
    reference_zero_threshold_1pm=0.0005,
    quasi_steady_rate_threshold_1pm_s=0.002,
    monotonic_progress_epsilon_1pm_s=0.00001,
    turn_class_curvature_threshold_1pm=0.005,
    direct_handoff_min_peak_curvature_1pm=0.002,
    direct_handoff_max_neutral_duration_s=0.3,
    minimum_phase_duration_s=0.09,
    minimum_phase_samples=2,
    maximum_phase_extension_s=2.0,
    maximum_sample_gap_s=0.15,
    turn_in_crossing_fraction=0.5,
    release_onset_fraction=0.9,
    maximum_raw_phase_spans=65_536,
    maximum_phase_windows=4_096,
    maximum_event_locators=4_096,
    maximum_event_phase_attachments=65_536,
  )


def metric_config() -> BehaviorMetricConfig:
  return BehaviorMetricConfig(
    burst_window_s=0.2,
    chatter_torque_rate_threshold_per_s=0.05,
    turn_in_crossing_fraction=0.5,
    release_crossing_fraction=0.9,
    correction_curvature_threshold_1pm=0.002,
    unused_headroom_threshold=0.05,
    growing_error_epsilon_1pm=0.00001,
    completion_delivered_fraction=0.95,
    minimum_samples=2,
    speed_nodes_mps=(5.0,),
    maximum_route_windows_per_stratum=20,
  )


def rule(
  metric: BehaviorMetricName,
  contract: BehaviorContract,
  preference: MetricPreference,
  *,
  minimum: float,
  maximum: float,
) -> MetricGateRule:
  return MetricGateRule(
    metric_name=metric.value,
    contract=contract,
    preference=preference,
    noise_floor=1e-7,
    margin_normalization=1.0,
    minimum_allowed=minimum,
    maximum_allowed=maximum,
    minimum_route_count=1,
    minimum_window_count=1,
    minimum_weighted_support=0.5,
    required_strata=("5:turn",),
  )


def gate_spec() -> BehaviorGateSpec:
  rules = (
    rule(
      BehaviorMetricName.RAW_CHATTER_REVERSALS_PER_S,
      BehaviorContract.SMOOTH,
      MetricPreference.LOWER_IS_BETTER,
      minimum=0.0,
      maximum=10.0,
    ),
    rule(
      BehaviorMetricName.SIGNED_TURN_IN_LAG_S,
      BehaviorContract.SWIFT,
      MetricPreference.LOWER_IS_BETTER,
      minimum=-0.01,
      maximum=1.0,
    ),
    rule(
      BehaviorMetricName.INTEGRATED_CURVATURE_ERROR,
      BehaviorContract.STRONG,
      MetricPreference.LOWER_IS_BETTER,
      minimum=0.0,
      maximum=1.0,
    ),
  )
  return BehaviorGateSpec(
    schema_version=BEHAVIOR_GATE_SPEC_SCHEMA_VERSION,
    provenance="transaction test gate",
    metric_config=metric_config(),
    metric_rules=rules,
    target_metric_name=BehaviorMetricName.INTEGRATED_CURVATURE_ERROR.value,
    paired_uncertainty_method=PAIRED_ROUTE_UNCERTAINTY_METHOD,
    minimum_paired_route_count=2,
    candidate_grid=CandidateGridBounds(
      natural_frequency_log_offsets=(0.0, math.log(1.2)),
      damping_ratio_log_offsets=(0.0,),
      minimum_natural_frequency_per_s=5.0,
      maximum_natural_frequency_per_s=20.0,
      minimum_damping_ratio=0.5,
      maximum_damping_ratio=2.0,
    ),
    route_partition=RoutePartitionSpec(
      validation_fraction=None,
      validation_route_count=2,
      seed_identity_sha256="e" * 64,
    ),
  )


def core_callback(*, noisy_after_contact: bool = False):
  def replay(request: ControllerReplayRequest):
    candidate_improved = (
      request.artifact_identity.role.value == "candidate"
      and request.policy is not None
      and request.policy.natural_frequency_per_s > 10.0
    )
    contacted = False
    outputs = []
    for control, reference in zip(
      request.route.control_inputs,
      request.references,
      strict=True,
    ):
      if control.driver_intervention_onset:
        contacted = True
      factor = 1.0 if candidate_improved else 0.6
      # Release is kept exact so an incomplete reference release remains a
      # symmetric coverage exclusion rather than a controller-side failure.
      if control.core_input == b"release":
        factor = 1.0
      if control.core_input == b"hard" and candidate_improved:
        factor = 0.2
      if noisy_after_contact and contacted:
        factor = -20.0
      raw = reference.anchored_curvature_1pm * 10.0 * factor
      outputs.append(ControllerFrameOutput(
        mono_time_ns=control.mono_time_ns,
        measured_curvature_1pm=reference.anchored_curvature_1pm * factor,
        measured_rack_angle_deg=reference.desired_rack_angle_deg * factor,
        measured_rack_rate_deg_s=reference.desired_rack_rate_deg_s * factor,
        measured_rack_accel_deg_s2=reference.desired_rack_accel_deg_s2 * factor,
        raw_requested_torque=raw,
        planned_requested_torque=raw,
        reachable_envelope_torque=raw,
        envelope_applied_torque=raw,
        torque_headroom=max(0.0, 1.0 - abs(raw)),
        actuator_constrained=False,
        steering_request_active=True,
        maximum_authority_required=False,
        controller_fault=False,
        response_eligible=True,
      ))
    return tuple(outputs)
  return replay


def cores(*, noisy_after_contact: bool = False):
  callback = core_callback(noisy_after_contact=noisy_after_contact)
  return (
    BehaviorReplayCore(replay_core_identity("stock", "4"), callback),
    BehaviorReplayCore(replay_core_identity("accepted", "5"), callback),
    BehaviorReplayCore(replay_core_identity("candidate", "6"), callback),
  )


def run(
  routes,
  *,
  worker_count: int = 1,
  noisy_after_contact: bool = False,
  progress_callback=None,
):
  stock, accepted, candidate = cores(noisy_after_contact=noisy_after_contact)
  return run_behavior_learning_transaction(
    route_evidence_artifacts=routes,
    decode_route_evidence=lambda artifact, _: artifact,
    physical_profile=physical_profile(),
    accepted_policy=BehaviorPolicy(10.0, 1.0),
    search_center_policy=BehaviorPolicy(10.0, 1.0),
    exact_stock=stock,
    currently_accepted=accepted,
    candidate=candidate,
    segmentation_config=segmentation_config(),
    gate_spec=gate_spec(),
    worker_count=worker_count,
    progress_callback=progress_callback,
  )


def private_replay_jobs(replay, count: int = 4):
  profile = physical_profile()
  policy = BehaviorPolicy(10.0, 1.0)
  core = BehaviorReplayCore(replay_core_identity("blocking", "7"), replay)
  identity = ReplayArtifactIdentity.compose(ReplayRole.CANDIDATE, core.identity, policy)
  jobs = tuple(
    behavior_transaction._ReplayJob(
      identity,
      policy,
      core,
      behavior_transaction._prepare_route(
        decoded_route(index),
        profile,
        segmentation_config(),
        lambda: False,
      ),
    )
    for index in range(count)
  )
  return jobs, profile


def process_is_running(pid: int) -> bool:
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  try:
    with open(f"/proc/{pid}/stat", encoding="utf-8") as stream:
      state = stream.read().split()[2]
  except (FileNotFoundError, IndexError, OSError):
    return False
  return state != "Z"


def wait_for_processes_gone(pids: tuple[int, ...], timeout_s: float = 3.0) -> bool:
  deadline = time.monotonic() + timeout_s
  while time.monotonic() < deadline:
    if not any(process_is_running(pid) for pid in pids):
      return True
    time.sleep(0.02)
  return not any(process_is_running(pid) for pid in pids)


class TestBehaviorTransaction(unittest.TestCase):
  def test_public_transaction_rejects_opponents_with_different_dynamics(self):
    common = {
      "source_openpilot_commit": "1" * 40,
      "opendbc_commit": "2" * 40,
      "panda_commit": "3" * 40,
    }
    stock_dynamics = ProvisionalRackDynamics(4000.0, 10.0, 4.0, "stock")
    candidate_dynamics = replace(stock_dynamics, rack_gain_deg_s2_per_torque=3999.0)
    stock = make_exact_stock_behavior_replay_core(
      reviewed_replay_core_identity(exact_stock=True, **common),
      provisional_dynamics=stock_dynamics,
    )
    candidate = make_modular_behavior_replay_core(
      reviewed_replay_core_identity(exact_stock=False, **common),
      provisional_dynamics=candidate_dynamics,
    )

    with self.assertRaisesRegex(
      BehaviorTransactionError,
      "identical provisional rack dynamics",
    ):
      run_reviewed_behavior_learning_transaction(
        route_evidence_artifacts=(),
        decode_route_evidence=lambda artifact, _: artifact,
        physical_profile=physical_profile(),
        accepted_policy=None,
        search_center_policy=BehaviorPolicy(10.0, 1.0),
        exact_stock=stock,
        currently_accepted=None,
        candidate=candidate,
        segmentation_config=segmentation_config(),
        gate_spec=gate_spec(),
      )

  def test_public_transaction_rejects_callback_spoofing_reviewed_stock_hash(self):
    callback = core_callback()
    stock_identity = reviewed_replay_core_identity(
      exact_stock=True,
      source_openpilot_commit="1" * 40,
      opendbc_commit="2" * 40,
      panda_commit="3" * 40,
    )
    stock = BehaviorReplayCore(stock_identity, callback)
    _ignored_stock, accepted, candidate = cores()

    with self.assertRaisesRegex(
      BehaviorTransactionError,
      "requires reviewed replay-core adapters",
    ):
      run_reviewed_behavior_learning_transaction(
        route_evidence_artifacts=tuple(decoded_route(index) for index in range(4)),
        decode_route_evidence=lambda artifact, _: artifact,
        physical_profile=physical_profile(),
        accepted_policy=BehaviorPolicy(10.0, 1.0),
        search_center_policy=BehaviorPolicy(10.0, 1.0),
        exact_stock=stock,
        currently_accepted=accepted,
        candidate=candidate,
        segmentation_config=segmentation_config(),
        gate_spec=gate_spec(),
      )

  def test_every_worker_count_promptly_reaps_blocked_workers_on_abort(self):
    context = multiprocessing.get_context("fork")
    for worker_count in (1, 4):
      with self.subTest(worker_count=worker_count):
        pids = context.Array("i", (0,) * worker_count, lock=True)
        all_started = context.Event()

        def blocked_replay(
          request: ControllerReplayRequest,
          pids=pids,
          all_started=all_started,
        ):
          index = int(request.route.route_id.removeprefix("route-"))
          with pids.get_lock():
            pids[index] = os.getpid()
            if all(value > 0 for value in pids):
              all_started.set()
          while True:
            time.sleep(0.1)

        jobs, profile = private_replay_jobs(blocked_replay, worker_count)
        started_at = time.monotonic()
        with self.assertRaises(BehaviorTransactionAborted):
          behavior_transaction._run_replay_jobs(
            jobs,
            profile,
            worker_count,
            all_started.is_set,
          )
        self.assertLess(time.monotonic() - started_at, 3.0)
        worker_pids = tuple(int(value) for value in pids)
        self.assertTrue(all(pid > 0 for pid in worker_pids))
        self.assertTrue(wait_for_processes_gone(worker_pids))
        self.assertIsNone(behavior_transaction._FORK_REPLAY_JOBS)
        self.assertIsNone(behavior_transaction._FORK_PHYSICAL_PROFILE)

  def test_parallel_worker_error_reaps_blocked_peers(self):
    context = multiprocessing.get_context("fork")
    pids = context.Array("i", (0, 0, 0, 0), lock=True)
    all_started = context.Event()

    def one_error_replay(request: ControllerReplayRequest):
      index = int(request.route.route_id.removeprefix("route-"))
      with pids.get_lock():
        pids[index] = os.getpid()
        if all(value > 0 for value in pids):
          all_started.set()
      while not all_started.wait(timeout=0.05):
        pass
      if index == 0:
        raise RuntimeError("deliberate replay failure")
      while True:
        time.sleep(0.1)

    jobs, profile = private_replay_jobs(one_error_replay)
    with self.assertRaisesRegex(BehaviorTransactionError, "deliberate replay failure"):
      behavior_transaction._run_replay_jobs(
        jobs,
        profile,
        4,
        lambda: False,
      )
    worker_pids = tuple(int(value) for value in pids)
    self.assertTrue(all(pid > 0 for pid in worker_pids))
    self.assertTrue(wait_for_processes_gone(worker_pids))
    self.assertIsNone(behavior_transaction._FORK_REPLAY_JOBS)
    self.assertIsNone(behavior_transaction._FORK_PHYSICAL_PROFILE)

  def test_worker_parent_death_kills_owned_descendant_group(self):
    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)

    def outer_owner():
      receive.close()

      def replay_with_descendant(_request: ControllerReplayRequest):
        descendant_pid = os.fork()
        if descendant_pid == 0:
          while True:
            time.sleep(0.1)
        send.send((os.getpid(), descendant_pid))
        while True:
          time.sleep(0.1)

      jobs, profile = private_replay_jobs(replay_with_descendant, 1)
      behavior_transaction._run_replay_jobs(
        jobs,
        profile,
        1,
        lambda: False,
      )

    owner = context.Process(target=outer_owner, name="behavior-owner-death-test")
    owner.start()
    send.close()
    self.assertTrue(receive.poll(3.0))
    worker_pid, descendant_pid = receive.recv()
    os.kill(owner.pid, signal.SIGKILL)
    owner.join(timeout=2.0)
    self.assertFalse(owner.is_alive())
    self.assertTrue(wait_for_processes_gone((worker_pid, descendant_pid)))
    receive.close()

  def test_worker_start_failure_closes_resources_without_joining_unstarted_process(self):
    context = multiprocessing.get_context("fork")
    children_before = {process.pid for process in multiprocessing.active_children()}
    with patch.object(context.Process, "start", side_effect=OSError("start failed")):
      with self.assertRaisesRegex(OSError, "start failed"):
        behavior_transaction._BehaviorReplayWorker(
          context=context,
          job_index=0,
          abort_requested=lambda: False,
          inherited_close_fds=(),
        )
    self.assertEqual(
      {process.pid for process in multiprocessing.active_children()},
      children_before,
    )

  def test_parallel_fork_fails_closed_when_parent_has_another_thread(self):
    release = threading.Event()
    active = threading.Event()

    def hold_thread() -> None:
      active.set()
      release.wait(timeout=3.0)

    thread = threading.Thread(target=hold_thread)
    thread.start()
    self.assertTrue(active.wait(timeout=1.0))
    jobs, profile = private_replay_jobs(core_callback(), 1)
    try:
      with self.assertRaisesRegex(BehaviorTransactionError, "multithreaded"):
        behavior_transaction._run_replay_jobs(
          jobs,
          profile,
          1,
          lambda: False,
        )
    finally:
      release.set()
      thread.join(timeout=1.0)
    self.assertFalse(thread.is_alive())

  def test_transaction_document_strictly_round_trips_and_rejects_mutation(self):
    result = run(tuple(decoded_route(index) for index in range(4)))
    encoded = result.to_json()

    restored = BehaviorLearningTransactionResult.from_json(encoded)
    self.assertEqual(restored.to_json(), encoded)
    self.assertEqual(restored.sha256, result.sha256)

    with self.assertRaisesRegex(ValueError, "not canonical"):
      BehaviorLearningTransactionResult.from_json(encoded + "\n")

    unknown = json.loads(encoded)
    unknown["activationApproved"] = True
    with self.assertRaisesRegex(ValueError, "keys"):
      BehaviorLearningTransactionResult.from_json(
        json.dumps(unknown, separators=(",", ":"), sort_keys=True),
      )

    wrong_type = json.loads(encoded)
    wrong_type["schemaVersion"] = True
    with self.assertRaisesRegex(ValueError, "schema version"):
      BehaviorLearningTransactionResult.from_json(
        json.dumps(wrong_type, separators=(",", ":"), sort_keys=True),
      )

  def test_exact_stock_bootstrap_has_no_fictitious_accepted_policy(self):
    routes = tuple(decoded_route(index) for index in range(4))
    stock, accepted, candidate = cores()
    result = run_behavior_learning_transaction(
      route_evidence_artifacts=routes,
      decode_route_evidence=lambda artifact, _: artifact,
      physical_profile=physical_profile(),
      accepted_policy=None,
      search_center_policy=BehaviorPolicy(10.0, 1.0),
      exact_stock=stock,
      currently_accepted=None,
      candidate=candidate,
      segmentation_config=segmentation_config(),
      gate_spec=gate_spec(),
    )

    accepted_evaluations = tuple(
      evaluation
      for evaluation in result.evaluations
      if '"role":"currently_accepted"' in evaluation.artifact_identity
    )
    self.assertTrue(accepted_evaluations)
    self.assertTrue(all(evaluation.policy is None for evaluation in accepted_evaluations))

    with self.assertRaisesRegex(BehaviorTransactionError, "must omit"):
      run_behavior_learning_transaction(
        route_evidence_artifacts=routes,
        decode_route_evidence=lambda artifact, _: artifact,
        physical_profile=physical_profile(),
        accepted_policy=None,
        search_center_policy=BehaviorPolicy(10.0, 1.0),
        exact_stock=stock,
        currently_accepted=accepted,
        candidate=candidate,
        segmentation_config=segmentation_config(),
        gate_spec=gate_spec(),
      )

  def test_successful_whole_route_transaction_returns_one_qualified_policy(self):
    result = run(tuple(decoded_route(index) for index in range(4)), worker_count=2)

    self.assertTrue(result.finalization.passed)
    self.assertFalse(result.stock_retained)
    self.assertEqual(
      result.qualification_disposition,
      QualificationDisposition.QUALIFIED_CANDIDATE_AVAILABLE,
    )
    self.assertIsNotNone(result.selected_policy)
    assert result.selected_policy is not None
    self.assertGreater(result.selected_policy.natural_frequency_per_s, 10.0)
    self.assertTrue(result.finalization.smooth_passed)
    self.assertTrue(result.finalization.swift_passed)
    self.assertTrue(result.finalization.strong_passed)

  def test_balanced_target_loss_cannot_veto_passing_strata(self):
    routes = tuple(decoded_route(index, hard=index == 2) for index in range(4))
    result = run(routes)

    self.assertTrue(result.finalization.passed)
    self.assertFalse(result.finalization.target_materially_improved)
    self.assertTrue(result.finalization.smooth_passed)
    self.assertTrue(result.finalization.swift_passed)
    self.assertTrue(result.finalization.strong_passed)
    self.assertFalse(result.stock_retained)
    self.assertIsNotNone(result.selected_policy)

  def test_hard_turn_that_never_crosses_is_fatal_despite_easy_windows(self):
    routes = tuple(decoded_route(index, hard=index == 1) for index in range(4))
    result = run(routes)
    candidate_evaluations = tuple(
      evaluation
      for evaluation in result.evaluations
      if '"role":"candidate"' in evaluation.artifact_identity
      and evaluation.policy is not None
      and evaluation.policy.natural_frequency_per_s > 10.0
    )

    self.assertTrue(candidate_evaluations)
    turn_metrics = tuple(
      evaluation.metric(BehaviorMetricName.SIGNED_TURN_IN_LAG_S.value)
      for evaluation in candidate_evaluations
    )
    self.assertTrue(any(metric.physical_failure_window_ids for metric in turn_metrics))
    self.assertFalse(result.finalization.passed)
    self.assertTrue(result.stock_retained)

  def test_reference_unobservable_release_is_excluded_symmetrically(self):
    routes = tuple(
      decoded_route(index, incomplete_release=index == 0)
      for index in range(4)
    )
    result = run(routes)
    release_by_partition: dict[tuple[str, ...], list] = {}
    for evaluation in result.evaluations:
      release_by_partition.setdefault(evaluation.route_ids, []).append(
        evaluation.metric(BehaviorMetricName.SIGNED_RELEASE_LAG_S.value),
      )

    found = False
    for metrics in release_by_partition.values():
      covered = [
        metric for metric in metrics
        if any("reference_release_incomplete" in reason for reason in metric.exclusions)
      ]
      if covered:
        found = True
        self.assertEqual(len({metric.coverage_identity_sha256 for metric in covered}), 1)
        self.assertFalse(any(metric.physical_failure_window_ids for metric in covered))
    self.assertTrue(found)

  def test_route_candidate_order_and_worker_count_cannot_change_result(self):
    routes = tuple(decoded_route(index) for index in range(4))
    serial = run(routes, worker_count=1)
    actual_grid = build_candidate_grid(
      gate_spec().candidate_grid.policy_grid(BehaviorPolicy(10.0, 1.0)),
    )
    with patch(
      "openpilot.selfdrive.controls.lib.blatv2.behavior_transaction.build_candidate_grid",
      return_value=tuple(reversed(actual_grid)),
    ):
      parallel = run(tuple(reversed(routes)), worker_count=4)

    self.assertEqual(serial.to_json(), parallel.to_json())
    self.assertEqual(serial.sha256, parallel.sha256)

  def test_training_and_validation_use_exactly_two_canonical_batches(self):
    routes = tuple(decoded_route(index) for index in range(4))
    partition = partition_whole_routes(
      tuple(route.identity for route in routes),
      gate_spec().route_partition,
    )
    with patch.object(
      behavior_transaction,
      "_run_replay_jobs",
      wraps=behavior_transaction._run_replay_jobs,
    ) as replay_batches:
      result = run(routes, worker_count=2)

    self.assertEqual(replay_batches.call_count, 2)
    training_jobs = replay_batches.call_args_list[0].args[0]
    validation_jobs = replay_batches.call_args_list[1].args[0]
    grid = build_candidate_grid(
      gate_spec().candidate_grid.policy_grid(BehaviorPolicy(10.0, 1.0)),
    )
    self.assertEqual(
      len(training_jobs),
      (2 + len(grid)) * len(partition.training_route_ids),
    )
    self.assertEqual(
      {job.route.route.route_id for job in training_jobs},
      set(partition.training_route_ids),
    )
    self.assertEqual(
      {
        job.policy
        for job in training_jobs
        if job.identity.role.value == "candidate"
      },
      {candidate.policy for candidate in grid},
    )

    # Held-out evidence sees three semantically distinct roles only.  No losing
    # grid policy can be present even when roles happen to share core bytes.
    self.assertEqual(
      len(validation_jobs),
      3 * len(partition.validation_route_ids),
    )
    self.assertEqual(
      {job.route.route.route_id for job in validation_jobs},
      set(partition.validation_route_ids),
    )
    self.assertEqual(
      {job.identity.role.value for job in validation_jobs},
      {"exact_stock", "currently_accepted", "candidate"},
    )
    self.assertEqual(
      {
        job.policy
        for job in validation_jobs
        if job.identity.role.value == "candidate"
      },
      {result.selected_policy},
    )

  def test_progress_is_parent_only_monotonic_and_cannot_change_result(self):
    routes = tuple(decoded_route(index) for index in range(4))
    baseline = run(routes, worker_count=1)
    parent_pid = os.getpid()
    progress: list[tuple[int, BehaviorReplayProgress]] = []

    def collect(value: BehaviorReplayProgress) -> None:
      progress.append((os.getpid(), value))

    observed = run(routes, worker_count=4, progress_callback=collect)
    self.assertEqual(observed.to_json(), baseline.to_json())
    self.assertTrue(progress)
    self.assertEqual({pid for pid, _ in progress}, {parent_pid})
    completed = [value.completed_jobs for _, value in progress]
    self.assertEqual(completed, sorted(completed))
    self.assertEqual(completed[0], 0)
    self.assertEqual(completed[-1], progress[-1][1].total_jobs)
    self.assertEqual(
      {value.phase for _, value in progress},
      {BehaviorReplayProgressPhase.TRAINING, BehaviorReplayProgressPhase.VALIDATION},
    )

    callback_calls = 0

    def broken_callback(_value: BehaviorReplayProgress) -> None:
      nonlocal callback_calls
      callback_calls += 1
      raise RuntimeError("display failed")

    with_broken_display = run(
      routes,
      worker_count=2,
      progress_callback=broken_callback,
    )
    self.assertGreater(callback_calls, 0)
    self.assertEqual(with_broken_display.to_json(), baseline.to_json())
    self.assertEqual(with_broken_display.sha256, baseline.sha256)

  def test_bootstrap_progress_counts_roles_not_unique_core_bytes(self):
    routes = tuple(decoded_route(index) for index in range(4))
    stock, _accepted, candidate = cores()
    progress: list[BehaviorReplayProgress] = []
    result = run_behavior_learning_transaction(
      route_evidence_artifacts=routes,
      decode_route_evidence=lambda artifact, _: artifact,
      physical_profile=physical_profile(),
      accepted_policy=None,
      search_center_policy=BehaviorPolicy(10.0, 1.0),
      exact_stock=stock,
      currently_accepted=None,
      candidate=candidate,
      segmentation_config=segmentation_config(),
      gate_spec=gate_spec(),
      progress_callback=progress.append,
    )
    self.assertTrue(result.finalization.passed)
    grid_count = len(build_candidate_grid(
      gate_spec().candidate_grid.policy_grid(BehaviorPolicy(10.0, 1.0)),
    ))
    partition = partition_whole_routes(
      tuple(route.identity for route in routes),
      gate_spec().route_partition,
    )
    expected = (
      (2 + grid_count) * len(partition.training_route_ids)
      + 3 * len(partition.validation_route_ids)
    )
    self.assertEqual(progress[0].total_jobs, expected)
    self.assertEqual(progress[-1].completed_jobs, expected)

  def test_scheduler_refactor_preserves_prebatch_transaction_bytes(self):
    result = run(tuple(decoded_route(index) for index in range(4)))
    self.assertEqual(
      result.sha256,
      # Transaction schema 3 binds raw/planned/reachable/applied commands,
      # request state, exact stratum dispositions, and bounded all-window summaries.
        "1c8b6191c63c9cf408691c8232d5d7cd42b83f02001fd068ddffc18cfe5519f3",
    )

  def test_validation_replays_only_the_frozen_training_winner(self):
    result = run(tuple(decoded_route(index) for index in range(4)))
    validation_ids = partition_whole_routes(
      tuple(decoded_route(index).identity for index in range(4)),
      gate_spec().route_partition,
    ).validation_route_ids
    validation_candidates = tuple(
      evaluation
      for evaluation in result.evaluations
      if evaluation.route_ids == validation_ids
      and '"role":"candidate"' in evaluation.artifact_identity
    )

    self.assertEqual(len(validation_candidates), 1)
    self.assertEqual(validation_candidates[0].policy, result.selected_policy)

  def test_duplicate_and_mixed_route_identity_are_rejected(self):
    route = decoded_route(0)
    with self.assertRaisesRegex(BehaviorTransactionError, "duplicate route"):
      run((route, route, decoded_route(1), decoded_route(2)))

    mixed = replace(decoded_route(3), recorded_source=source("f"))
    with self.assertRaisesRegex(BehaviorTransactionError, "mixes recorded source"):
      run((decoded_route(0), decoded_route(1), decoded_route(2), mixed))

  def test_driver_intervention_censors_after_contact_without_casting_a_vote(self):
    routes = tuple(
      decoded_route(index, intervention=index == 0)
      for index in range(4)
    )
    ordinary = run(routes)
    poisoned_after_contact = run(routes, noisy_after_contact=True)

    self.assertEqual(ordinary.to_json(), poisoned_after_contact.to_json())
    self.assertEqual(ordinary.sha256, poisoned_after_contact.sha256)

  def test_aa_is_exact_and_unqualified_physical_profile_cannot_start(self):
    routes = tuple(decoded_route(index) for index in range(4))
    first = run(routes, worker_count=2)
    second = run(routes, worker_count=2)
    self.assertEqual(first.to_json(), second.to_json())
    self.assertEqual(first.sha256, second.sha256)

    stock, accepted, candidate = cores()
    with self.assertRaisesRegex(BehaviorTransactionError, "fully qualified"):
      run_behavior_learning_transaction(
        route_evidence_artifacts=routes,
        decode_route_evidence=lambda artifact, _: artifact,
        physical_profile=physical_profile(qualified=False),
        accepted_policy=BehaviorPolicy(10.0, 1.0),
        search_center_policy=BehaviorPolicy(10.0, 1.0),
        exact_stock=stock,
        currently_accepted=accepted,
        candidate=candidate,
        segmentation_config=segmentation_config(),
        gate_spec=gate_spec(),
      )


if __name__ == "__main__":
  unittest.main()

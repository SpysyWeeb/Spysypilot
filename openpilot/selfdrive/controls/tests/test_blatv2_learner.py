from __future__ import annotations

import math
import unittest

from openpilot.selfdrive.controls.lib.blatv2.learner import (
  ActuatorBoundary,
  HIGH_SPEED_MIN_CLEAN_SUPPORT_S,
  LOW_SPEED_MIN_CLEAN_SUPPORT_S,
  MID_SPEED_MIN_CLEAN_SUPPORT_S,
  MIN_AUTHORITY_VALIDATION_SAMPLES,
  TRAIN_VALIDATION_BLOCK_SAMPLES,
  LearningSample,
  ProfileLearner,
  QualificationReason,
  _attest_authority_sample,
  learning_sample_field_names,
  minimum_clean_support_s,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  DEFAULT_SPEED_NODES_MPS,
  make_seed_profile,
)


DT = 0.1
TRUE_TORQUE_PER_LATACCEL = 0.32
TRUE_RACK_GAIN = 1800.0
TRUE_RACK_DAMPING = 7.0
SEED_TORQUE_PER_LATACCEL = 0.50
SEED_RACK_GAIN = 1000.0
SEED_RACK_DAMPING = 3.0
KINETIC_FRICTION = 0.03
STATIC_FRICTION = 0.09
RATE_RESOLUTION = 1.0


def seed_profile(vehicle_identity: str = "test-platform"):
  return make_seed_profile(
    vehicle_identity=vehicle_identity,
    torque_per_lateral_accel=SEED_TORQUE_PER_LATACCEL,
    rack_gain_deg_s2_per_torque=SEED_RACK_GAIN,
    rack_damping_per_s=SEED_RACK_DAMPING,
    transport_delay_s=0.12,
    static_friction_torque=STATIC_FRICTION,
    kinetic_friction_torque=KINETIC_FRICTION,
    rack_rate_resolution_deg_s=RATE_RESOLUTION,
  )


def excitation_values(index: int) -> tuple[float, float, float]:
  lateral_levels = (-1.2, -0.8, -0.35, 0.25, 0.65, 1.0, 1.3)
  rate_levels = (-18.0, -11.0, -5.0, 4.0, 9.0, 15.0, 20.0, -7.0)
  acceleration_levels = (
    -140.0, -95.0, -55.0, -15.0, 20.0, 50.0,
    85.0, 120.0, 65.0, -35.0, 105.0,
  )
  return (
    lateral_levels[index % len(lateral_levels)],
    rate_levels[(index * 3) % len(rate_levels)],
    acceleration_levels[(index * 5) % len(acceleration_levels)],
  )


def measured_sample(
  speed_mps: float,
  index: int,
  *,
  torque_per_lataccel: float = TRUE_TORQUE_PER_LATACCEL,
  rack_gain: float = TRUE_RACK_GAIN,
  rack_damping: float = TRUE_RACK_DAMPING,
  kinetic_friction: float = KINETIC_FRICTION,
  dt_s: float = DT,
  engaged: bool = True,
  valid: bool = True,
  steering_pressed: bool = False,
  actuator_constrained: bool = False,
  standstill: bool = False,
) -> LearningSample:
  lateral_accel, rack_rate, rack_acceleration = excitation_values(index)
  friction = kinetic_friction * (1.0 if rack_rate > 0.0 else -1.0)
  applied_torque = (
    friction
    - torque_per_lataccel * lateral_accel
    + rack_acceleration / rack_gain
    + rack_damping * rack_rate / rack_gain
  )
  return LearningSample(
    speed_mps=speed_mps,
    dt_s=dt_s,
    applied_torque=applied_torque,
    measured_lateral_accel_mps2=lateral_accel,
    rack_rate_deg_s=rack_rate,
    rack_acceleration_deg_s2=rack_acceleration,
    engaged=engaged,
    valid=valid,
    steering_pressed=steering_pressed,
    actuator_constrained=actuator_constrained,
    standstill=standstill,
  )


def settled_authority_sample(
  *,
  dwell_s: float,
  dt_s: float = 0.01,
  rack_rate_deg_s: float = 8.0,
  torque_per_lataccel: float = TRUE_TORQUE_PER_LATACCEL,
  rack_gain: float = TRUE_RACK_GAIN,
  rack_damping: float = TRUE_RACK_DAMPING,
  kinetic_friction: float = KINETIC_FRICTION,
) -> LearningSample:
  lateral_accel = -1.2
  rack_acceleration = (
    (
      1.0
      - kinetic_friction
      + torque_per_lataccel * lateral_accel
      - rack_damping * rack_rate_deg_s / rack_gain
    )
    * rack_gain
  )
  return _attest_authority_sample(
    LearningSample(
      speed_mps=3.0,
      dt_s=dt_s,
      applied_torque=1.0,
      measured_lateral_accel_mps2=lateral_accel,
      rack_rate_deg_s=rack_rate_deg_s,
      rack_acceleration_deg_s2=rack_acceleration,
      engaged=True,
      valid=True,
      steering_pressed=False,
      actuator_constrained=True,
      standstill=False,
    ),
    boundary=ActuatorBoundary.MAGNITUDE,
    magnitude_boundary_dwell_s=dwell_s,
  )


def add_qualified_node_data(
  learner: ProfileLearner,
  node_index: int,
  *,
  kinetic_friction: float = KINETIC_FRICTION,
) -> None:
  speed = learner.speed_nodes_mps[node_index]
  sample_count = (
    math.ceil(minimum_clean_support_s(speed) / DT)
    + 2 * TRAIN_VALIDATION_BLOCK_SAMPLES
  )
  for index in range(sample_count):
    assert learner.add_sample(measured_sample(
      speed,
      index,
      kinetic_friction=kinetic_friction,
    ))


class TestBLaTv2Learner(unittest.TestCase):
  def test_input_contract_contains_measured_physics_only(self) -> None:
    names = learning_sample_field_names()
    self.assertEqual(names, (
      "speed_mps",
      "dt_s",
      "applied_torque",
      "measured_lateral_accel_mps2",
      "rack_rate_deg_s",
      "rack_acceleration_deg_s2",
      "engaged",
      "valid",
      "steering_pressed",
      "actuator_constrained",
      "standstill",
    ))
    self.assertFalse(any("desired" in name for name in names))
    self.assertFalse(any("candidate" in name for name in names))
    self.assertFalse(any("request" in name for name in names))

  def test_unclean_samples_do_not_mutate_evidence(self) -> None:
    learner = ProfileLearner(seed_profile())
    before = tuple(
      learner.evidence_for_node(index).to_bytes()
      for index in range(len(DEFAULT_SPEED_NODES_MPS))
    )
    unclean = (
      measured_sample(10.0, 0, engaged=False),
      measured_sample(10.0, 0, valid=False),
      measured_sample(10.0, 0, steering_pressed=True),
      measured_sample(10.0, 0, actuator_constrained=True),
      measured_sample(10.0, 0, standstill=True),
      measured_sample(10.0, 0, dt_s=0.2),
      LearningSample(
        speed_mps=10.0,
        dt_s=DT,
        applied_torque=math.nan,
        measured_lateral_accel_mps2=0.0,
        rack_rate_deg_s=0.0,
        rack_acceleration_deg_s2=0.0,
        engaged=True,
        valid=True,
        steering_pressed=False,
        actuator_constrained=False,
        standstill=False,
      ),
    )
    for sample in unclean:
      self.assertFalse(learner.add_sample(sample))
    after = tuple(
      learner.evidence_for_node(index).to_bytes()
      for index in range(len(DEFAULT_SPEED_NODES_MPS))
    )
    self.assertEqual(after, before)

  def test_valid_slew_boundary_is_separate_authority_evidence(self) -> None:
    first = ProfileLearner(seed_profile())
    second = ProfileLearner(seed_profile())
    before = first.evidence_for_node(0).to_bytes()

    for index in range(16):
      sample = _attest_authority_sample(
        measured_sample(
          3.0,
          index,
          actuator_constrained=True,
        ),
        boundary=ActuatorBoundary.SLEW_BUILD,
        magnitude_boundary_dwell_s=0.0,
      )
      self.assertFalse(sample.clean)
      self.assertTrue(sample.authority_evidence)
      self.assertTrue(first.add_sample(sample))
      self.assertTrue(second.add_sample(sample))

    snapshot = first.evidence_for_node(0)
    self.assertNotEqual(snapshot.to_bytes(), before)
    self.assertEqual(snapshot.supported_sample_count, 0)
    self.assertEqual(snapshot.authority_sample_count, 16)
    self.assertEqual(snapshot.authority_slew_build_sample_count, 16)
    self.assertEqual(snapshot.authority_fit_sample_count, 0)
    encoded = first.export_evidence()
    self.assertEqual(encoded, second.export_evidence())
    restored = ProfileLearner.from_evidence(seed_profile(), encoded)
    self.assertEqual(
      restored.evidence_for_node(0).to_bytes(),
      snapshot.to_bytes(),
    )

  def test_settled_magnitude_motion_enters_authority_fit(self) -> None:
    learner = ProfileLearner(seed_profile())
    before_delay = settled_authority_sample(dwell_s=0.129)
    at_delay = settled_authority_sample(dwell_s=0.13)

    self.assertTrue(before_delay.authority_evidence)
    self.assertTrue(learner.add_sample(before_delay))
    self.assertEqual(
      learner.evidence_for_node(0).authority_fit_sample_count,
      0,
    )
    self.assertTrue(learner.add_sample(at_delay))
    snapshot = learner.evidence_for_node(0)
    self.assertEqual(snapshot.authority_magnitude_sample_count, 2)
    self.assertEqual(snapshot.authority_fit_sample_count, 1)
    self.assertEqual(snapshot.authority_unresolved_sample_count, 0)
    self.assertEqual(snapshot.supported_sample_count, 0)

    stuck = settled_authority_sample(
      dwell_s=0.4,
      rack_rate_deg_s=0.0,
    )
    self.assertTrue(learner.add_sample(stuck))
    stuck_snapshot = learner.evidence_for_node(0)
    self.assertEqual(stuck_snapshot.authority_fit_sample_count, 1)
    self.assertEqual(stuck_snapshot.authority_unresolved_sample_count, 1)

  def test_sparse_authority_fit_cannot_move_unvalidated_candidate(self) -> None:
    ordinary = ProfileLearner(seed_profile())
    with_sparse_authority = ProfileLearner(seed_profile())
    for node_index in range(len(DEFAULT_SPEED_NODES_MPS)):
      add_qualified_node_data(ordinary, node_index)
      add_qualified_node_data(with_sparse_authority, node_index)
    for _ in range(64):
      self.assertTrue(with_sparse_authority.add_sample(
        settled_authority_sample(dwell_s=0.2),
      ))

    baseline = ordinary.qualify("ordinary baseline")
    candidate = with_sparse_authority.qualify("authority deferred")
    self.assertIsNotNone(baseline.candidate_profile)
    self.assertIsNotNone(candidate.candidate_profile)
    baseline_parameters = tuple(
      node.parameters for node in baseline.candidate_profile.nodes
    )
    candidate_parameters = tuple(
      node.parameters for node in candidate.candidate_profile.nodes
    )
    self.assertEqual(candidate_parameters, baseline_parameters)
    self.assertFalse(candidate.node_reports[0].authority_fit_active)
    self.assertEqual(
      candidate.node_reports[0].authority_training_count,
      64,
    )
    self.assertEqual(
      candidate.node_reports[0].authority_validation_count,
      0,
    )

  def test_authority_fit_activates_only_with_held_out_rows(self) -> None:
    learner = ProfileLearner(seed_profile())
    for node_index in range(len(DEFAULT_SPEED_NODES_MPS)):
      add_qualified_node_data(learner, node_index)
    for _ in range(
      TRAIN_VALIDATION_BLOCK_SAMPLES + MIN_AUTHORITY_VALIDATION_SAMPLES
    ):
      self.assertTrue(learner.add_sample(
        settled_authority_sample(dwell_s=0.2),
      ))

    result = learner.qualify("authority validation ready")
    report = result.node_reports[0]
    self.assertTrue(report.authority_fit_active)
    self.assertEqual(
      report.authority_training_count,
      TRAIN_VALIDATION_BLOCK_SAMPLES,
    )
    self.assertEqual(
      report.authority_validation_count,
      MIN_AUTHORITY_VALIDATION_SAMPLES,
    )
    self.assertIsNotNone(report.authority_seed_validation_rms)
    self.assertIsNotNone(report.authority_candidate_validation_rms)

  def test_authority_validation_regression_blocks_profile(self) -> None:
    learner = ProfileLearner(seed_profile())
    for node_index in range(len(DEFAULT_SPEED_NODES_MPS)):
      add_qualified_node_data(learner, node_index)
    for _ in range(TRAIN_VALIDATION_BLOCK_SAMPLES):
      self.assertTrue(learner.add_sample(
        settled_authority_sample(dwell_s=0.2),
      ))
    for _ in range(MIN_AUTHORITY_VALIDATION_SAMPLES):
      self.assertTrue(learner.add_sample(settled_authority_sample(
        dwell_s=0.2,
        torque_per_lataccel=SEED_TORQUE_PER_LATACCEL,
        rack_gain=SEED_RACK_GAIN,
        rack_damping=SEED_RACK_DAMPING,
      )))

    result = learner.qualify("authority held-out regression")
    self.assertIsNone(result.candidate_profile)
    self.assertIn(
      QualificationReason.AUTHORITY_VALIDATION_REGRESSION,
      result.node_reports[0].reasons,
    )

  def test_sample_supports_only_adjacent_nodes(self) -> None:
    learner = ProfileLearner(seed_profile())
    before = tuple(
      learner.evidence_for_node(index).to_bytes()
      for index in range(len(DEFAULT_SPEED_NODES_MPS))
    )
    self.assertTrue(learner.add_sample(measured_sample(7.5, 0)))
    snapshots = tuple(
      learner.evidence_for_node(index)
      for index in range(len(DEFAULT_SPEED_NODES_MPS))
    )
    self.assertAlmostEqual(snapshots[1].clean_support_s, DT * 0.5)
    self.assertAlmostEqual(snapshots[2].clean_support_s, DT * 0.5)
    for index in (0, 3, 4, 5):
      self.assertEqual(snapshots[index].to_bytes(), before[index])

  def test_highway_samples_leave_low_speed_nodes_byte_identical(self) -> None:
    learner = ProfileLearner(seed_profile())
    low_before = tuple(
      learner.evidence_for_node(index).to_bytes() for index in (0, 1, 2)
    )
    for index in range(500):
      self.assertTrue(learner.add_sample(measured_sample(25.0, index)))
    low_after = tuple(
      learner.evidence_for_node(index).to_bytes() for index in (0, 1, 2)
    )
    self.assertEqual(low_after, low_before)
    self.assertGreater(learner.evidence_for_node(4).clean_support_s, 0.0)
    self.assertGreater(learner.evidence_for_node(5).clean_support_s, 0.0)

  def test_support_floors_follow_low_mid_high_contract(self) -> None:
    self.assertEqual(
      minimum_clean_support_s(0.0), LOW_SPEED_MIN_CLEAN_SUPPORT_S,
    )
    self.assertEqual(
      minimum_clean_support_s(5.0), LOW_SPEED_MIN_CLEAN_SUPPORT_S,
    )
    self.assertEqual(
      minimum_clean_support_s(10.0), MID_SPEED_MIN_CLEAN_SUPPORT_S,
    )
    self.assertEqual(
      minimum_clean_support_s(15.0), MID_SPEED_MIN_CLEAN_SUPPORT_S,
    )
    self.assertEqual(
      minimum_clean_support_s(20.0), HIGH_SPEED_MIN_CLEAN_SUPPORT_S,
    )
    self.assertEqual(
      minimum_clean_support_s(30.0), HIGH_SPEED_MIN_CLEAN_SUPPORT_S,
    )

  def test_straight_time_does_not_qualify_without_excitation(self) -> None:
    learner = ProfileLearner(seed_profile())
    count = math.ceil(MID_SPEED_MIN_CLEAN_SUPPORT_S / DT) + 256
    straight = LearningSample(
      speed_mps=10.0,
      dt_s=DT,
      applied_torque=0.0,
      measured_lateral_accel_mps2=0.0,
      rack_rate_deg_s=0.0,
      rack_acceleration_deg_s2=0.0,
      engaged=True,
      valid=True,
      steering_pressed=False,
      actuator_constrained=False,
      standstill=False,
    )
    for _ in range(count):
      self.assertTrue(learner.add_sample(straight))
    result = learner.qualify("straight-only")
    report = result.node_reports[2]
    self.assertGreaterEqual(report.clean_support_s, report.minimum_support_s)
    self.assertIn(
      QualificationReason.INSUFFICIENT_EXCITATION, report.reasons,
    )
    self.assertIn(QualificationReason.SINGULAR_FIT, report.reasons)
    self.assertIsNone(result.candidate_profile)

  def test_train_validation_partition_is_deterministic(self) -> None:
    first = ProfileLearner(seed_profile())
    second = ProfileLearner(seed_profile())
    count = 5 * TRAIN_VALIDATION_BLOCK_SAMPLES
    for index in range(count):
      sample = measured_sample(10.0, index)
      first.add_sample(sample)
      second.add_sample(sample)
    self.assertEqual(
      first.evidence_for_node(2).to_bytes(),
      second.evidence_for_node(2).to_bytes(),
    )
    report = first.qualify("partition").node_reports[2]
    self.assertEqual(report.training_count, 3 * TRAIN_VALIDATION_BLOCK_SAMPLES)
    self.assertEqual(report.validation_count, 2 * TRAIN_VALIDATION_BLOCK_SAMPLES)

  def test_singular_fit_is_refused(self) -> None:
    learner = ProfileLearner(seed_profile())
    count = math.ceil(MID_SPEED_MIN_CLEAN_SUPPORT_S / DT) + 256
    levels = (-1.0, -0.5, 0.5, 1.0)
    for index in range(count):
      value = levels[index % len(levels)]
      lateral_accel = value
      rack_rate = 10.0 * value
      rack_acceleration = 100.0 * value
      friction = math.copysign(KINETIC_FRICTION, rack_rate)
      torque = (
        friction
        - TRUE_TORQUE_PER_LATACCEL * lateral_accel
        + rack_acceleration / TRUE_RACK_GAIN
        + TRUE_RACK_DAMPING * rack_rate / TRUE_RACK_GAIN
      )
      learner.add_sample(LearningSample(
        speed_mps=10.0,
        dt_s=DT,
        applied_torque=torque,
        measured_lateral_accel_mps2=lateral_accel,
        rack_rate_deg_s=rack_rate,
        rack_acceleration_deg_s2=rack_acceleration,
        engaged=True,
        valid=True,
        steering_pressed=False,
        actuator_constrained=False,
        standstill=False,
      ))
    report = learner.qualify("singular").node_reports[2]
    self.assertIn(QualificationReason.SINGULAR_FIT, report.reasons)
    self.assertIsNone(report.candidate_parameters)

  def test_held_out_regression_is_refused(self) -> None:
    learner = ProfileLearner(seed_profile())
    count = math.ceil(MID_SPEED_MIN_CLEAN_SUPPORT_S / DT) + 512
    for index in range(count):
      validation_block = (
        index // TRAIN_VALIDATION_BLOCK_SAMPLES
      ) % 2 == 1
      learner.add_sample(measured_sample(
        10.0,
        index,
        torque_per_lataccel=(
          SEED_TORQUE_PER_LATACCEL
          if validation_block
          else TRUE_TORQUE_PER_LATACCEL
        ),
        rack_gain=(
          SEED_RACK_GAIN if validation_block else TRUE_RACK_GAIN
        ),
        rack_damping=(
          SEED_RACK_DAMPING if validation_block else TRUE_RACK_DAMPING
        ),
        kinetic_friction=(
          KINETIC_FRICTION if validation_block else 0.06
        ),
      ))
    report = learner.qualify("regressive-holdout").node_reports[2]
    self.assertIn(
      QualificationReason.VALIDATION_REGRESSION, report.reasons,
    )
    self.assertIsNotNone(report.candidate_validation_rms)
    self.assertIsNotNone(report.seed_validation_rms)
    self.assertGreater(
      report.candidate_validation_rms, report.seed_validation_rms,
    )

  def test_one_qualified_node_cannot_promote_profile(self) -> None:
    learner = ProfileLearner(seed_profile())
    add_qualified_node_data(learner, 0)
    result = learner.qualify("one-node-only")
    self.assertTrue(result.node_reports[0].qualified)
    self.assertTrue(
      all(not report.qualified for report in result.node_reports[1:]),
    )
    self.assertIsNone(result.candidate_profile)

  def test_complete_profile_recovers_all_four_coefficients_deterministically(self) -> None:
    first = ProfileLearner(seed_profile("vehicle-A"))
    second = ProfileLearner(seed_profile("vehicle-A"))
    for node_index in range(len(DEFAULT_SPEED_NODES_MPS)):
      add_qualified_node_data(first, node_index)
      add_qualified_node_data(second, node_index)

    first_result = first.qualify("synthetic-known-plant")
    second_result = second.qualify("synthetic-known-plant")
    self.assertTrue(first_result.all_nodes_qualified)
    self.assertIsNotNone(first_result.candidate_profile)
    self.assertIsNotNone(second_result.candidate_profile)
    first_profile = first_result.candidate_profile
    second_profile = second_result.candidate_profile
    self.assertEqual(first_profile.to_json(), second_profile.to_json())
    self.assertEqual(first_profile.vehicle_identity, "vehicle-A")
    expected_revision = 1 + sum(
      first.evidence_for_node(index).supported_sample_count
      for index in range(len(DEFAULT_SPEED_NODES_MPS))
    )
    self.assertEqual(first_profile.revision, expected_revision)
    self.assertIn("fit_seed_revision=0", first_profile.provenance)
    self.assertIn(
      f"evidence_revision={expected_revision}",
      first_profile.provenance,
    )
    self.assertTrue(first_profile.qualified)

    for node in first_profile.nodes:
      params = node.parameters
      self.assertAlmostEqual(
        params.torque_per_lateral_accel,
        TRUE_TORQUE_PER_LATACCEL,
        places=10,
      )
      self.assertAlmostEqual(
        params.rack_gain_deg_s2_per_torque,
        TRUE_RACK_GAIN,
        places=6,
      )
      self.assertAlmostEqual(
        params.rack_damping_per_s,
        TRUE_RACK_DAMPING,
        places=9,
      )
      self.assertEqual(params.transport_delay_s, 0.12)
      self.assertEqual(params.static_friction_torque, STATIC_FRICTION)
      self.assertAlmostEqual(
        params.kinetic_friction_torque,
        KINETIC_FRICTION,
        places=12,
      )
      self.assertEqual(
        params.rack_rate_resolution_deg_s, RATE_RESOLUTION,
      )

  def test_zero_and_near_static_kinetic_friction_are_recovered(self) -> None:
    for kinetic_friction in (0.0, STATIC_FRICTION - 1e-4):
      with self.subTest(kinetic_friction=kinetic_friction):
        learner = ProfileLearner(seed_profile())
        add_qualified_node_data(
          learner,
          0,
          kinetic_friction=kinetic_friction,
        )
        report = learner.qualify("friction-boundary").node_reports[0]
        self.assertTrue(report.qualified, report.reasons)
        self.assertIsNotNone(report.candidate_parameters)
        self.assertAlmostEqual(
          report.candidate_parameters.kinetic_friction_torque,
          kinetic_friction,
          places=11,
        )
        self.assertEqual(
          report.candidate_parameters.static_friction_torque,
          STATIC_FRICTION,
        )

  def test_negative_and_above_static_friction_are_refused(self) -> None:
    for kinetic_friction in (-0.01, STATIC_FRICTION + 0.01):
      with self.subTest(kinetic_friction=kinetic_friction):
        learner = ProfileLearner(seed_profile())
        add_qualified_node_data(
          learner,
          0,
          kinetic_friction=kinetic_friction,
        )
        result = learner.qualify("invalid-friction")
        report = result.node_reports[0]
        self.assertIn(
          QualificationReason.INVALID_PARAMETERS,
          report.reasons,
        )
        self.assertIsNone(report.candidate_parameters)
        self.assertIsNone(result.candidate_profile)

  def test_one_direction_evidence_cannot_qualify_friction(self) -> None:
    learner = ProfileLearner(seed_profile())
    speed = learner.speed_nodes_mps[0]
    sample_count = (
      math.ceil(minimum_clean_support_s(speed) / DT)
      + 2 * TRAIN_VALIDATION_BLOCK_SAMPLES
    )
    for index in range(sample_count):
      lateral_accel, rack_rate, rack_acceleration = excitation_values(index)
      rack_rate = abs(rack_rate)
      applied_torque = (
        KINETIC_FRICTION
        - TRUE_TORQUE_PER_LATACCEL * lateral_accel
        + rack_acceleration / TRUE_RACK_GAIN
        + TRUE_RACK_DAMPING * rack_rate / TRUE_RACK_GAIN
      )
      self.assertTrue(learner.add_sample(LearningSample(
        speed_mps=speed,
        dt_s=DT,
        applied_torque=applied_torque,
        measured_lateral_accel_mps2=lateral_accel,
        rack_rate_deg_s=rack_rate,
        rack_acceleration_deg_s2=rack_acceleration,
        engaged=True,
        valid=True,
        steering_pressed=False,
        actuator_constrained=False,
        standstill=False,
      )))
    result = learner.qualify("one-direction")
    report = result.node_reports[0]
    self.assertIn(
      QualificationReason.INSUFFICIENT_EXCITATION,
      report.reasons,
    )
    self.assertEqual(report.rack_reversals, 0)
    self.assertIsNone(result.candidate_profile)

  def test_vehicle_identity_and_legacy_payload_are_not_interchangeable(self) -> None:
    learner = ProfileLearner(seed_profile("vehicle-A"))
    self.assertEqual(learner.seed_profile.vehicle_identity, "vehicle-A")
    with self.assertRaises(TypeError):
      ProfileLearner({
        "schema_version": 0,
        "param_key": "BLaTv2AdaptiveProfile",
      })


if __name__ == "__main__":
  unittest.main()

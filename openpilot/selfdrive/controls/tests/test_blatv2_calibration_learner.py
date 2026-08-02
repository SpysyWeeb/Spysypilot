from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
import unittest
from unittest.mock import patch

from openpilot.selfdrive.controls.lib.blatv2.breakaway_episode import (
  BreakawayCategory,
  BreakawayDecision,
  BreakawayEpisodeDetector,
)

from openpilot.selfdrive.controls.lib.blatv2.calibration_learner import (
  CALIBRATION_EVIDENCE_SCHEMA_VERSION,
  CalibrationFitStatus,
  CalibrationModelId,
  CalibrationProfileLearner,
  CalibrationQualificationReason,
  CalibrationSampleDisposition,
  _JointRegression,
  _Regression,
  _fit_bounded_subset,
  _seed_coefficients,
  _solve,
  calibration_evidence_sha256,
  calibration_learning_sample_field_names,
  minimum_calibration_support_s,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_profile import (
  CALIBRATION_PROFILE_SCHEMA_VERSION,
  CalibrationParameters,
  DEFAULT_SPEED_NODES_MPS,
  VehicleCalibrationProfile,
  make_calibration_seed_profile,
)
from openpilot.selfdrive.controls.lib.blatv2.learner import (
  ActuatorBoundary,
  LearningSample,
  _attest_authority_sample,
)


DT = 0.1
TRUE_GAIN = 0.32
TRUE_OFFSET_MPS2 = 0.04
TRUE_KINETIC = 0.03
TRUE_STATIC = 0.09
RATE_QUANTUM_DEG_S = 4.0


def route_sha(counter: int) -> str:
  return hashlib.sha256(f"route-{counter}".encode()).hexdigest()


def seed_profile(vehicle_identity: str = "calibration-learner-test") -> VehicleCalibrationProfile:
  return make_calibration_seed_profile(
    vehicle_identity=vehicle_identity,
    torque_callback_slope=0.50,
    stock_friction_torque=0.06,
    transport_delay_s=0.12,
    rack_rate_resolution_deg_s=RATE_QUANTUM_DEG_S,
  )


def inverse_torque(lateral_accel: float, *, moving_sign: int = 0, breakaway_sign: int = 0) -> float:
  return TRUE_GAIN * (-lateral_accel + TRUE_OFFSET_MPS2) + TRUE_KINETIC * moving_sign + TRUE_STATIC * breakaway_sign


def sample(
  speed_mps: float,
  lateral_accel: float,
  rack_rate: float,
  *,
  moving_sign: int = 0,
  breakaway_sign: int = 0,
  rack_angle: float = 0.0,
  torque_delta: float = 0.0,
  rack_acceleration: float = 0.0,
  reversal: bool = False,
) -> LearningSample:
  return LearningSample(
    speed_mps=speed_mps,
    dt_s=DT,
    applied_torque=(
      inverse_torque(
        lateral_accel,
        moving_sign=moving_sign,
        breakaway_sign=breakaway_sign,
      )
      + torque_delta
    ),
    measured_lateral_accel_mps2=lateral_accel,
    rack_rate_deg_s=rack_rate,
    rack_acceleration_deg_s2=rack_acceleration,
    engaged=True,
    valid=True,
    steering_pressed=False,
    actuator_constrained=False,
    standstill=False,
    rack_direction_reversal=reversal,
    measured_rack_angle_deg=rack_angle,
  )


def add_identifiable_stream(
  learner: CalibrationProfileLearner,
  speed_mps: float,
  sample_count: int,
  *,
  torque_delta: float = 0.0,
) -> None:
  """Generate stationary, breakaway, and moving rows in both directions."""
  index = 0
  while index < sample_count:
    direction = -1 if (index // 6) % 2 else 1
    lateral_magnitudes = (0.30, 0.55, 0.85, 1.10, 0.70, 0.42)
    # Three 100 ms observations span 200 ms of physical dwell, clearing the
    # 120 ms transport delay without counting the first timestamp as elapsed.
    for base_offset in range(3):
      if index >= sample_count:
        return
      lateral_accel = direction * lateral_magnitudes[(index + base_offset) % len(lateral_magnitudes)]
      if not learner.add_sample(sample(
        speed_mps,
        lateral_accel,
        0.0,
        breakaway_sign=(direction if base_offset == 2 else 0),
        torque_delta=torque_delta,
      )):
        raise AssertionError("known-clean base sample was rejected")
      index += 1
    if index >= sample_count:
      return
    lateral_accel = direction * lateral_magnitudes[index % len(lateral_magnitudes)]
    if not learner.add_sample(
      sample(
        speed_mps,
        lateral_accel,
        direction * RATE_QUANTUM_DEG_S,
        breakaway_sign=direction,
        torque_delta=torque_delta,
      )
    ):
      raise AssertionError("known-clean breakaway sample was rejected")
    index += 1
    for _ in range(2):
      if index >= sample_count:
        return
      lateral_accel = direction * lateral_magnitudes[index % len(lateral_magnitudes)]
      if not learner.add_sample(
        sample(
          speed_mps,
          lateral_accel,
          direction * (RATE_QUANTUM_DEG_S + 4.0),
          moving_sign=direction,
          torque_delta=torque_delta,
        )
      ):
        raise AssertionError("known-clean moving sample was rejected")
      index += 1


def add_identifiable_route(
  learner: CalibrationProfileLearner,
  speed_mps: float,
  sample_count: int,
  route_counter: int,
  *,
  torque_delta: float = 0.0,
) -> None:
  learner.begin_route(route_sha(route_counter), route_counter=route_counter)
  try:
    add_identifiable_stream(
      learner,
      speed_mps,
      sample_count,
      torque_delta=torque_delta,
    )
  finally:
    learner.end_route()


def add_identifiable_route_for_parameters(
  learner: CalibrationProfileLearner,
  speed_mps: float,
  sample_count: int,
  route_counter: int,
  parameters: CalibrationParameters,
) -> None:
  learner.begin_route(route_sha(route_counter), route_counter=route_counter)
  try:
    index = 0
    while index < sample_count:
      direction = -1 if (index // 6) % 2 else 1
      lateral_magnitudes = (0.30, 0.55, 0.85, 1.10, 0.70, 0.42)

      def measured(
        lateral_accel: float,
        rack_rate: float,
        *,
        moving_sign: int = 0,
        breakaway_sign: int = 0,
      ) -> LearningSample:
        base = sample(
          speed_mps,
          lateral_accel,
          rack_rate,
          moving_sign=moving_sign,
          breakaway_sign=breakaway_sign,
        )
        torque = (
          parameters.torque_per_lateral_accel
          * (
            -lateral_accel
            + parameters.lateral_accel_offset_correction_mps2
          )
          + parameters.kinetic_friction_torque * moving_sign
          + parameters.static_breakaway_torque * breakaway_sign
        )
        return replace(base, applied_torque=torque)

      for base_offset in range(3):
        if index >= sample_count:
          return
        lateral_accel = direction * lateral_magnitudes[
          (index + base_offset) % len(lateral_magnitudes)
        ]
        assert learner.add_sample(measured(
          lateral_accel,
          0.0,
          breakaway_sign=(direction if base_offset == 2 else 0),
        ))
        index += 1
      if index >= sample_count:
        return
      lateral_accel = direction * lateral_magnitudes[index % len(lateral_magnitudes)]
      assert learner.add_sample(measured(
        lateral_accel,
        direction * RATE_QUANTUM_DEG_S,
        breakaway_sign=direction,
      ))
      index += 1
      for _ in range(2):
        if index >= sample_count:
          return
        lateral_accel = direction * lateral_magnitudes[index % len(lateral_magnitudes)]
        assert learner.add_sample(measured(
          lateral_accel,
          direction * (RATE_QUANTUM_DEG_S + 4.0),
          moving_sign=direction,
        ))
        index += 1
  finally:
    learner.end_route()


def add_complete_evidence(
  learner: CalibrationProfileLearner,
  parameters_by_speed: dict[float, CalibrationParameters] | None = None,
) -> None:
  for speed_index, speed in enumerate(DEFAULT_SPEED_NODES_MPS):
    support_samples = math.ceil(minimum_calibration_support_s(speed) / DT) + 512
    route_samples = math.ceil(support_samples / 4)
    for route_counter in (0, 2, 1, 3):
      unique_route_counter = speed_index * 16 + route_counter
      if parameters_by_speed is None or speed not in parameters_by_speed:
        add_identifiable_route(
          learner,
          speed,
          route_samples,
          unique_route_counter,
        )
      else:
        add_identifiable_route_for_parameters(
          learner,
          speed,
          route_samples,
          unique_route_counter,
          parameters_by_speed[speed],
        )


def canonical(payload: object) -> bytes:
  return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def add_angle_assisted_episode(
  learner: CalibrationProfileLearner,
  speed_mps: float,
  direction: int,
) -> None:
  """Add one dwell -> angle onset -> rate confirmation episode."""
  for index in range(3):
    assert learner.add_sample(sample(
      speed_mps,
      direction * (0.30 + 0.05 * index),
      0.0,
      rack_angle=0.0,
      breakaway_sign=(direction if index == 2 else 0),
    ))
  assert learner.add_sample(sample(
    speed_mps,
    direction * 0.50,
    0.0,
    rack_angle=direction * 0.25,
    breakaway_sign=direction,
  ))
  assert learner.add_sample(sample(
    speed_mps,
    direction * 0.55,
    direction * RATE_QUANTUM_DEG_S,
    rack_angle=direction * 0.65,
    breakaway_sign=direction,
  ))


class TestBLaTv2CalibrationLearner(unittest.TestCase):
  def test_interval_statistics_match_exact_runtime_gain_offset_interpolation(self) -> None:
    lower = CalibrationParameters(
      torque_per_lateral_accel=0.22,
      lateral_accel_offset_correction_mps2=-0.18,
      kinetic_friction_torque=0.02,
      static_breakaway_torque=0.08,
      transport_delay_s=0.12,
      rack_rate_resolution_deg_s=4.0,
      confidence=1.0,
      qualified=True,
    )
    upper = CalibrationParameters(
      torque_per_lateral_accel=0.71,
      lateral_accel_offset_correction_mps2=0.24,
      kinetic_friction_torque=0.05,
      static_breakaway_torque=0.11,
      transport_delay_s=0.12,
      rack_rate_resolution_deg_s=4.0,
      confidence=1.0,
      qualified=True,
    )
    evidence = _JointRegression()
    rows = (
      ((-0.8, 1.0, 1.0, 0.0), 0.10, -0.10, 0.25),
      ((0.4, 1.0, -1.0, 0.0), 0.35, 0.20, 0.50),
      ((-1.2, 1.0, 0.0, 1.0), 0.80, -0.35, 0.75),
    )
    direct_error = 0.0
    total_weight = 0.0
    for predictors, weight, target, upper_weight in rows:
      evidence.add(predictors, target, weight, upper_weight)
      lower_weight = 1.0 - upper_weight
      gain = lower_weight * lower.torque_per_lateral_accel + upper_weight * upper.torque_per_lateral_accel
      offset = (
        lower_weight * lower.lateral_accel_offset_correction_mps2
        + upper_weight * upper.lateral_accel_offset_correction_mps2
      )
      kinetic = lower_weight * lower.kinetic_friction_torque + upper_weight * upper.kinetic_friction_torque
      static = lower_weight * lower.static_breakaway_torque + upper_weight * upper.static_breakaway_torque
      prediction = (
        gain * (predictors[0] + offset * predictors[1])
        + kinetic * predictors[2]
        + static * predictors[3]
      )
      direct_error += weight * (target - prediction) ** 2
      total_weight += weight

    encoded_mse = evidence.mse(lower, upper)
    self.assertIsNotNone(encoded_mse)
    assert encoded_mse is not None
    self.assertAlmostEqual(encoded_mse, direct_error / total_weight, places=14)

  def test_constrained_fit_resolves_unobservable_breakaway_at_physical_boundary(self) -> None:
    evidence = _Regression()
    gain = 0.32
    intercept = gain * 0.04
    for lateral_accel in (-1.0, -0.4, 0.4, 1.0):
      base = gain * -lateral_accel + intercept
      evidence.add((-lateral_accel, 1.0, 0.0, 0.0), base, 1.0)
      for direction in (-1, 1):
        evidence.add(
          (-lateral_accel, 1.0, float(direction), 0.0),
          base + 0.09 * direction,
          1.0,
        )
        # Deliberately impossible unconstrained evidence: its first-motion
        # residual is below moving friction. The physical fit must re-solve
        # the static==kinetic face rather than clamp a raw coefficient.
        evidence.add(
          (-lateral_accel, 1.0, 0.0, float(direction)),
          base + 0.03 * direction,
          1.0,
        )

    coefficients = _solve(evidence)
    self.assertIsNotNone(coefficients)
    if coefficients is None:
      self.fail("physically constrained observable fit was singular")
    fitted_gain, fitted_intercept, kinetic, static = coefficients
    self.assertAlmostEqual(fitted_gain, gain, places=12)
    self.assertAlmostEqual(fitted_intercept, intercept, places=12)
    self.assertAlmostEqual(kinetic, 0.06, places=12)
    self.assertEqual(static, kinetic)

  def test_input_contract_is_measured_only_and_acceleration_invariant(self) -> None:
    fields = calibration_learning_sample_field_names()
    self.assertIn("measured_lateral_accel_mps2", fields)
    self.assertIn("applied_torque", fields)
    self.assertNotIn("desired_curvature", fields)
    self.assertNotIn("model_curvature", fields)
    self.assertNotIn("candidate_torque", fields)

    first = CalibrationProfileLearner(seed_profile())
    second = CalibrationProfileLearner(seed_profile())
    first.begin_route(route_sha(0), route_counter=0)
    second.begin_route(route_sha(0), route_counter=0)
    for index in range(300):
      direction = -1 if index % 2 else 1
      lateral_accel = direction * (0.3 + 0.01 * (index % 40))
      rack_rate = direction * RATE_QUANTUM_DEG_S
      common = sample(10.0, lateral_accel, rack_rate, moving_sign=direction)
      self.assertTrue(first.add_sample(replace(common, rack_acceleration_deg_s2=-500.0 + index)))
      self.assertTrue(second.add_sample(replace(common, rack_acceleration_deg_s2=900.0 - 2.0 * index)))
    first.end_route()
    second.end_route()
    self.assertEqual(first.export_evidence(), second.export_evidence())

  def test_first_rate_quantum_is_motion_and_reversal_does_not_fake_breakaway(self) -> None:
    learner = CalibrationProfileLearner(seed_profile())
    learner.begin_route(route_sha(0), route_counter=0)
    self.assertTrue(learner.add_sample(sample(10.0, 0.4, RATE_QUANTUM_DEG_S, moving_sign=1)))
    snapshot = learner.evidence_for_node(2)
    self.assertEqual(snapshot.base_sample_count, 0)
    self.assertEqual(snapshot.moving_sample_count, 1)
    self.assertEqual(snapshot.breakaway_sample_count, 0)
    self.assertEqual(learner.evidence_for_node(1).supported_sample_count, 0)
    self.assertEqual(learner.evidence_for_node(3).supported_sample_count, 0)

    # Build a dwell that would classify the next ordinary motion as breakaway.
    self.assertTrue(learner.add_sample(sample(10.0, 0.35, 0.0)))
    self.assertTrue(learner.add_sample(sample(10.0, -0.35, 0.0)))
    reversal = sample(
      10.0,
      -0.5,
      -RATE_QUANTUM_DEG_S,
      moving_sign=-1,
      reversal=True,
      rack_acceleration=10_000.0,
    )
    self.assertFalse(reversal.clean)
    self.assertTrue(learner.add_sample(reversal))
    snapshot = learner.evidence_for_node(2)
    self.assertEqual(snapshot.base_sample_count, 2)
    self.assertEqual(snapshot.moving_sample_count, 2)
    self.assertEqual(snapshot.breakaway_sample_count, 0)
    self.assertEqual(snapshot.rack_reversals, 1)
    learner.end_route()

  def test_independent_support_and_authority_counts(self) -> None:
    learner = CalibrationProfileLearner(seed_profile())
    learner.begin_route(route_sha(0), route_counter=0)
    self.assertTrue(learner.add_sample(sample(10.0, 0.4, 0.0)))
    self.assertTrue(learner.add_sample(sample(10.0, -0.4, 0.0)))
    self.assertTrue(learner.add_sample(sample(10.0, 0.2, 0.0)))
    self.assertTrue(learner.add_sample(sample(10.0, 0.6, RATE_QUANTUM_DEG_S, breakaway_sign=1)))
    self.assertTrue(learner.add_sample(sample(10.0, 0.7, RATE_QUANTUM_DEG_S * 2.0, moving_sign=1)))

    raw_authority = LearningSample(
      speed_mps=10.0,
      dt_s=DT,
      applied_torque=1.0,
      measured_lateral_accel_mps2=-1.5,
      rack_rate_deg_s=RATE_QUANTUM_DEG_S,
      rack_acceleration_deg_s2=math.nan,
      engaged=True,
      valid=True,
      steering_pressed=False,
      actuator_constrained=True,
      standstill=False,
    )
    # Non-finite measured input is never evidence, including at authority.
    invalid_authority = _attest_authority_sample(
      raw_authority,
      boundary=ActuatorBoundary.MAGNITUDE,
      magnitude_boundary_dwell_s=DT,
    )
    self.assertFalse(learner.add_sample(invalid_authority))
    raw_authority = replace(raw_authority, rack_acceleration_deg_s2=0.0)
    stationary_authority = _attest_authority_sample(
      replace(raw_authority, rack_rate_deg_s=0.0),
      boundary=ActuatorBoundary.MAGNITUDE,
      magnitude_boundary_dwell_s=DT,
    )
    self.assertTrue(learner.add_sample(stationary_authority))
    stationary_snapshot = learner.evidence_for_node(2)
    self.assertEqual(stationary_snapshot.authority_sample_count, 1)
    self.assertEqual(stationary_snapshot.authority_fit_sample_count, 0)
    self.assertEqual(stationary_snapshot.authority_unresolved_sample_count, 1)

    authority = _attest_authority_sample(
      raw_authority,
      boundary=ActuatorBoundary.MAGNITUDE,
      magnitude_boundary_dwell_s=DT,
    )
    self.assertTrue(learner.add_sample(authority))

    snapshot = learner.evidence_for_node(2)
    self.assertEqual(snapshot.base_sample_count, 3)
    self.assertEqual(snapshot.breakaway_sample_count, 1)
    self.assertEqual(snapshot.moving_sample_count, 1)
    self.assertEqual(snapshot.authority_sample_count, 2)
    self.assertEqual(snapshot.authority_fit_sample_count, 1)
    self.assertEqual(snapshot.authority_magnitude_sample_count, 2)
    self.assertEqual(snapshot.authority_unresolved_sample_count, 1)
    self.assertAlmostEqual(snapshot.authority_support_s, 2.0 * DT)
    self.assertAlmostEqual(snapshot.authority_fit_support_s, DT)
    self.assertEqual(
      snapshot.training_count + snapshot.validation_count,
      snapshot.supported_sample_count,
    )
    self.assertAlmostEqual(
      snapshot.training_support_s + snapshot.validation_support_s,
      snapshot.clean_support_s,
    )
    self.assertEqual(snapshot.lateral_accel_directions, 2)
    self.assertEqual(snapshot.applied_torque_directions, 2)
    self.assertGreater(snapshot.lateral_accel_span_mps2, 0.0)
    self.assertGreater(snapshot.applied_torque_span, 0.0)
    learner.end_route()

  def test_route_partition_is_maneuver_atomic_and_active_route_cannot_publish(self) -> None:
    self.assertEqual(int(route_sha(1)[-1], 16) & 1, 0)
    for route_counter, validation in ((0, False), (1, True)):
      learner = CalibrationProfileLearner(seed_profile())
      learner.begin_route(route_sha(route_counter), route_counter=route_counter)
      add_identifiable_stream(learner, 10.0, 24)

      with self.assertRaisesRegex(RuntimeError, "active route evidence"):
        learner.qualify("must not publish a partial route")
      with self.assertRaisesRegex(RuntimeError, "active route evidence"):
        learner.export_evidence()

      snapshot = learner.evidence_for_node(2)
      selected_counts = (
        snapshot.validation_count,
        snapshot.moving_validation_count,
        snapshot.breakaway_validation_count,
        snapshot.breakaway_episode_validation_count,
      ) if validation else (
        snapshot.training_count,
        snapshot.moving_training_count,
        snapshot.breakaway_training_count,
        snapshot.breakaway_episode_training_count,
      )
      rejected_counts = (
        snapshot.training_count,
        snapshot.moving_training_count,
        snapshot.breakaway_training_count,
        snapshot.breakaway_episode_training_count,
      ) if validation else (
        snapshot.validation_count,
        snapshot.moving_validation_count,
        snapshot.breakaway_validation_count,
        snapshot.breakaway_episode_validation_count,
      )
      self.assertTrue(all(count > 0 for count in selected_counts))
      self.assertEqual(rejected_counts, (0, 0, 0, 0))
      learner.end_route()

  def test_route_identity_prevents_duplicate_uncertainty_and_partition_leakage(self) -> None:
    learner = CalibrationProfileLearner(seed_profile())
    learner.begin_route(route_sha(2), "a" * 64, route_counter=2)
    learner.end_route()
    with self.assertRaisesRegex(ValueError, "already ingested"):
      learner.begin_route(route_sha(2), "b" * 64, route_counter=2)
    with self.assertRaisesRegex(ValueError, "already ingested"):
      learner.begin_route(route_sha(4), "a" * 64, route_counter=4)
    with self.assertRaisesRegex(ValueError, "already ingested"):
      learner.begin_route(route_sha(4), "b" * 64, route_counter=2)

    for invalid_counter in (-1, 0x100000000, True):
      with self.subTest(route_counter=invalid_counter):
        with self.assertRaisesRegex(ValueError, "unsigned 32-bit"):
          CalibrationProfileLearner(seed_profile()).begin_route(
            route_sha(6), route_counter=invalid_counter,
          )

    restored = CalibrationProfileLearner.from_evidence(
      seed_profile(),
      learner.export_evidence(),
    )
    with self.assertRaisesRegex(ValueError, "already ingested"):
      restored.begin_route(route_sha(2), "a" * 64, route_counter=2)

  def test_validation_targets_cannot_select_model_or_change_parameter_bytes(self) -> None:
    seed = seed_profile()
    training = CalibrationProfileLearner(seed)
    for route_counter in (0, 2):
      add_identifiable_route(training, 10.0, 240, route_counter)
    training_evidence = training.export_evidence()

    nominal = CalibrationProfileLearner.from_evidence(seed, training_evidence)
    poisoned = CalibrationProfileLearner.from_evidence(seed, training_evidence)
    for route_counter in (1, 3):
      add_identifiable_route(nominal, 10.0, 240, route_counter)
      add_identifiable_route(
        poisoned,
        10.0,
        240,
        route_counter,
        torque_delta=0.12,
      )

    nominal_report = nominal.qualify("nominal validation").node_reports[2]
    poisoned_report = poisoned.qualify("poisoned validation").node_reports[2]
    self.assertIsNotNone(nominal_report.selected_model)
    self.assertEqual(poisoned_report.selected_model, nominal_report.selected_model)
    self.assertIsNotNone(nominal_report.candidate_parameters)
    self.assertIsNotNone(poisoned_report.candidate_parameters)
    self.assertEqual(
      canonical(asdict(poisoned_report.candidate_parameters)),
      canonical(asdict(nominal_report.candidate_parameters)),
    )
    self.assertNotEqual(
      poisoned_report.candidate_validation_rms,
      nominal_report.candidate_validation_rms,
    )

  def test_dense_ordinary_fit_cannot_outvote_breakaway_and_nested_safe_model_wins(self) -> None:
    learner = CalibrationProfileLearner(seed_profile())
    node = learner._nodes[2]
    seed = _seed_coefficients(learner.seed_profile.nodes[2].parameters)
    xs = (-1.0, -0.5, 0.5, 1.0)

    # Dense moving data asks for a radically different gain. Sparse base and
    # breakaway data ask only for an intercept correction. Aggregate least
    # squares therefore prefers the richer model, but that model makes the
    # rare breakaway population materially worse than the seed.
    for route_counter in (0, 2):
      learner.begin_route(route_sha(route_counter), route_counter=route_counter)
      for _ in range(25):
        for x in xs:
          direction = -1 if x < 0.0 else 1
          predictors = (x, 1.0, float(direction), 0.0)
          target = 0.30 * x + 0.02 + 0.06 * direction
          learner._add_regression(2, "moving_training", predictors, target, 1.0)
          learner._add_regression(2, "training", predictors, target, 1.0)
      for x in xs:
        direction = -1 if x < 0.0 else 1
        base_predictors = (x, 1.0, 0.0, 0.0)
        base_target = 0.50 * x + 0.02
        learner._add_regression(2, "training", base_predictors, base_target, 1.0)
        breakaway_predictors = (x, 1.0, 0.0, float(direction))
        breakaway_target = base_target + 0.06 * direction
        learner._add_regression(2, "breakaway_training", breakaway_predictors, breakaway_target, 1.0)
        learner._add_regression(2, "breakaway_episode_training", breakaway_predictors, breakaway_target, 1.0)
        learner._add_regression(2, "training", breakaway_predictors, breakaway_target, 1.0)
      learner.end_route()

    richer = _fit_bounded_subset(
      node.moving_training,
      seed,
      (0, 1),
    )
    self.assertIsNotNone(richer)
    if richer is None:
      self.fail("dense moving population unexpectedly produced no fit")
    self.assertLess(node.training.rms(richer), node.training.rms(seed))
    self.assertGreater(
      node.breakaway_training.rms(richer),
      node.breakaway_training.rms(seed),
    )

    report = learner._node_report(2)
    self.assertEqual(
      report.selected_model,
      CalibrationModelId.OFFSET_AND_FRICTION,
    )
    self.assertIsNotNone(report.candidate_parameters)
    if report.candidate_parameters is None:
      self.fail("safe nested model unexpectedly produced no parameters")
    self.assertAlmostEqual(
      report.candidate_parameters.torque_per_lateral_accel,
      0.50,
      places=12,
    )
    self.assertAlmostEqual(
      report.candidate_parameters.lateral_accel_offset_correction_mps2,
      0.04,
      places=12,
    )

  def test_high_speed_route_cannot_mutate_lower_speed_nodes(self) -> None:
    learner = CalibrationProfileLearner(seed_profile())
    lower_before = tuple(
      learner.evidence_for_node(index)
      for index in range(len(DEFAULT_SPEED_NODES_MPS) - 1)
    )
    add_identifiable_route(learner, 30.0, 240, 0)
    lower_after = tuple(
      learner.evidence_for_node(index)
      for index in range(len(DEFAULT_SPEED_NODES_MPS) - 1)
    )
    self.assertEqual(lower_after, lower_before)
    self.assertGreater(
      learner.evidence_for_node(len(DEFAULT_SPEED_NODES_MPS) - 1).supported_sample_count,
      0,
    )

  def test_strict_v9_evidence_is_deterministic_and_restorable(self) -> None:
    seed = seed_profile()
    learner = CalibrationProfileLearner(seed)
    for route_counter, direction in ((0, 1), (2, -1), (1, -1), (3, 1)):
      learner.begin_route(route_sha(route_counter), route_counter=route_counter)
      add_angle_assisted_episode(learner, 10.0, direction)
      add_identifiable_stream(learner, 10.0, 345)
      learner.end_route()
    encoded = learner.export_evidence()
    self.assertEqual(calibration_evidence_sha256(encoded), hashlib.sha256(encoded).hexdigest())
    envelope = json.loads(encoded)
    self.assertEqual(envelope["payload"]["evidence_schema_version"], CALIBRATION_EVIDENCE_SCHEMA_VERSION)
    self.assertEqual(envelope["payload"]["profile_schema_version"], CALIBRATION_PROFILE_SCHEMA_VERSION)
    self.assertEqual(
      [route["route_counter"] for route in envelope["payload"]["routes"]],
      [0, 2, 1, 3],
    )

    restored = CalibrationProfileLearner.from_evidence(seed, encoded)
    self.assertEqual(restored.export_evidence(), encoded)
    self.assertEqual(restored.evidence_for_node(2), learner.evidence_for_node(2))
    original_report = learner.qualify("v7 diagnostics").node_reports[2]
    restored_report = restored.qualify("v7 diagnostics").node_reports[2]
    self.assertGreater(original_report.breakaway_episode_training_count, 0)
    self.assertGreater(original_report.breakaway_episode_validation_count, 0)
    self.assertGreater(original_report.breakaway_episode_dwell_s, 0.0)
    self.assertGreater(original_report.breakaway_angle_assisted_count, 0)
    self.assertIsNotNone(original_report.selected_model)
    self.assertEqual(restored_report.selected_model, original_report.selected_model)
    self.assertEqual(
      restored_report.breakaway_episode_training_count,
      original_report.breakaway_episode_training_count,
    )
    self.assertEqual(
      restored_report.breakaway_episode_validation_count,
      original_report.breakaway_episode_validation_count,
    )
    self.assertEqual(
      restored_report.breakaway_angle_assisted_count,
      original_report.breakaway_angle_assisted_count,
    )
    self.assertEqual(
      restored_report.breakaway_mean_bracket_width,
      original_report.breakaway_mean_bracket_width,
    )

    with self.assertRaisesRegex(ValueError, "canonical"):
      CalibrationProfileLearner.from_evidence(seed, encoded + b"\n")
    tampered = json.loads(encoded)
    tampered["payload"]["evidence_schema_version"] = CALIBRATION_EVIDENCE_SCHEMA_VERSION + 1
    tampered["payload_sha256"] = hashlib.sha256(canonical(tampered["payload"])).hexdigest()
    with self.assertRaisesRegex(ValueError, "evidence schema"):
      CalibrationProfileLearner.from_evidence(seed, canonical(tampered))
    retired = json.loads(encoded)
    retired["payload"]["evidence_schema_version"] = 6
    retired["payload_sha256"] = hashlib.sha256(canonical(retired["payload"])).hexdigest()
    with self.assertRaisesRegex(ValueError, "evidence schema"):
      CalibrationProfileLearner.from_evidence(seed, canonical(retired))
    wrong_partition = json.loads(encoded)
    wrong_partition["payload"]["routes"][0]["validation"] = True
    wrong_partition["payload_sha256"] = hashlib.sha256(canonical(wrong_partition["payload"])).hexdigest()
    with self.assertRaisesRegex(ValueError, "partition does not match counter"):
      CalibrationProfileLearner.from_evidence(seed, canonical(wrong_partition))
    inconsistent = json.loads(encoded)
    inconsistent["payload"]["nodes"][2]["supported_sample_count"] += 1
    inconsistent["payload_sha256"] = hashlib.sha256(canonical(inconsistent["payload"])).hexdigest()
    with self.assertRaisesRegex(ValueError, "stratum counts"):
      CalibrationProfileLearner.from_evidence(seed, canonical(inconsistent))
    with self.assertRaisesRegex(ValueError, "different seed"):
      CalibrationProfileLearner.from_evidence(seed_profile("other-vehicle"), encoded)

  def test_sample_dispositions_are_durable_strict_and_non_mutating(self) -> None:
    seed = seed_profile()
    learner = CalibrationProfileLearner(seed)
    learner.begin_route(route_sha(0), route_counter=0)
    clean = sample(10.0, 0.4, RATE_QUANTUM_DEG_S, moving_sign=1)
    before = learner.evidence_for_node(2)
    disposition = learner.add_sample_with_disposition(
      clean,
      upstream_rejection=CalibrationSampleDisposition.LIVE_RACK_MAPPING_INVALID,
    )
    self.assertIs(disposition, CalibrationSampleDisposition.LIVE_RACK_MAPPING_INVALID)
    self.assertEqual(learner.evidence_for_node(2), before)
    self.assertTrue(learner.add_sample(clean))
    self.assertFalse(learner.add_sample(replace(clean, valid=False)))
    with patch.object(
      BreakawayEpisodeDetector,
      "update",
      return_value=BreakawayDecision(BreakawayCategory.DISCARDED),
    ):
      self.assertFalse(learner.add_sample(clean))
    learner.end_route()

    accounting = learner.sample_accounting
    self.assertEqual(accounting.ingested_sample_count, 4)
    self.assertEqual(accounting.accepted_sample_count, 1)
    self.assertEqual(accounting.rejected_sample_count, 3)
    self.assertEqual(
      accounting.count(CalibrationSampleDisposition.LIVE_RACK_MAPPING_INVALID),
      1,
    )
    self.assertEqual(
      accounting.count(CalibrationSampleDisposition.LEARNER_INELIGIBLE),
      1,
    )
    self.assertEqual(
      accounting.count(CalibrationSampleDisposition.BREAKAWAY_EPISODE_DISCARDED),
      1,
    )

    encoded = learner.export_evidence()
    restored = CalibrationProfileLearner.from_evidence(seed, encoded)
    self.assertEqual(restored.sample_accounting, accounting)
    self.assertEqual(restored.export_evidence(), encoded)
    restored.begin_route(route_sha(1), route_counter=1)
    restored.add_sample_with_disposition(
      clean,
      upstream_rejection=CalibrationSampleDisposition.VEHICLE_INPUT_INVALID,
    )
    restored.end_route()
    continued = restored.export_evidence()
    continued_accounting = CalibrationProfileLearner.from_evidence(
      seed,
      continued,
    ).sample_accounting
    self.assertEqual(continued_accounting.ingested_sample_count, 5)
    self.assertEqual(
      continued_accounting.count(CalibrationSampleDisposition.VEHICLE_INPUT_INVALID),
      1,
    )

    tampered = json.loads(encoded)
    tampered["payload"]["sample_accounting"]["ingested_sample_count"] += 1
    tampered["payload_sha256"] = hashlib.sha256(
      canonical(tampered["payload"]),
    ).hexdigest()
    with self.assertRaisesRegex(ValueError, "totals disagree"):
      CalibrationProfileLearner.from_evidence(seed, canonical(tampered))

    unknown = json.loads(encoded)
    unknown["payload"]["sample_accounting"]["rejection_reasons"]["other"] = 0
    unknown["payload_sha256"] = hashlib.sha256(
      canonical(unknown["payload"]),
    ).hexdigest()
    with self.assertRaisesRegex(ValueError, "accounting is invalid"):
      CalibrationProfileLearner.from_evidence(seed, canonical(unknown))

  def test_partial_never_emits_candidate_and_full_fit_qualifies_every_node(self) -> None:
    partial = CalibrationProfileLearner(seed_profile())
    add_identifiable_route(partial, 0.0, 200, 0)
    add_identifiable_route(partial, 0.0, 200, 1)
    partial_result = partial.qualify("partial test")
    self.assertIsNone(partial_result.candidate_profile)
    self.assertFalse(partial_result.all_nodes_qualified)
    self.assertIn(
      CalibrationQualificationReason.INSUFFICIENT_SUPPORT,
      partial_result.node_reports[-1].reasons,
    )

    seed = seed_profile()
    learner = CalibrationProfileLearner(seed)
    add_complete_evidence(learner)
    result = learner.qualify("synthetic observable calibration")
    if result.candidate_profile is None:
      self.fail(f"identifiable evidence failed qualification: {[report.reasons for report in result.node_reports]}")
    candidate = result.candidate_profile
    self.assertTrue(result.all_nodes_qualified)
    self.assertTrue(candidate.qualified)
    self.assertGreater(candidate.revision, seed.revision)
    for report, node in zip(result.node_reports, candidate.nodes, strict=True):
      self.assertTrue(report.qualified)
      self.assertEqual(report.reasons, (CalibrationQualificationReason.LEARNED,))
      self.assertGreater(report.base_sample_count, 0)
      self.assertGreater(report.moving_sample_count, 0)
      self.assertGreater(report.breakaway_sample_count, 0)
      self.assertIsNotNone(report.moving_candidate_validation_rms)
      self.assertIsNotNone(report.breakaway_candidate_validation_rms)
      self.assertAlmostEqual(node.parameters.torque_per_lateral_accel, TRUE_GAIN, places=9)
      self.assertAlmostEqual(node.parameters.lateral_accel_offset_correction_mps2, TRUE_OFFSET_MPS2, places=9)
      self.assertAlmostEqual(node.parameters.kinetic_friction_torque, TRUE_KINETIC, places=9)
      self.assertAlmostEqual(node.parameters.static_breakaway_torque, TRUE_STATIC, places=9)

  def test_no_improvement_retains_seed_without_emitting_artifact(self) -> None:
    seed = seed_profile()
    learner = CalibrationProfileLearner(seed)
    add_complete_evidence(
      learner,
      {speed: seed.nodes[index].parameters for index, speed in enumerate(DEFAULT_SPEED_NODES_MPS)},
    )

    result = learner.qualify("exact seed evidence")
    self.assertTrue(result.all_nodes_qualified)
    self.assertFalse(result.contains_learned_change)
    self.assertIsNone(result.candidate_profile)
    self.assertIsNotNone(result.selected_profile)
    assert result.selected_profile is not None
    self.assertTrue(result.selected_profile.qualified)
    self.assertTrue(all(report.seed_retained for report in result.node_reports))
    self.assertTrue(all(report.candidate_parameters is not None for report in result.node_reports))
    for report, seed_node in zip(result.node_reports, seed.nodes, strict=True):
      assert report.candidate_parameters is not None
      self.assertEqual(
        _seed_coefficients(report.candidate_parameters),
        _seed_coefficients(seed_node.parameters),
      )

  def test_mixed_profile_learns_supported_node_and_retains_exact_seed_elsewhere(self) -> None:
    seed = seed_profile()
    learner = CalibrationProfileLearner(seed)
    seed_laws = {
      speed: seed.nodes[index].parameters
      for index, speed in enumerate(DEFAULT_SPEED_NODES_MPS)
      if speed != 10.0
    }
    add_complete_evidence(learner, seed_laws)

    result = learner.qualify("mixed learned and retained profile")
    self.assertTrue(result.all_nodes_qualified)
    self.assertIsNotNone(result.candidate_profile)
    self.assertEqual(result.selected_profile, result.candidate_profile)
    assert result.candidate_profile is not None
    self.assertTrue(result.node_reports[2].learned)
    for index, report in enumerate(result.node_reports):
      if index == 2:
        continue
      self.assertTrue(report.seed_retained)
      self.assertEqual(
        _seed_coefficients(result.candidate_profile.nodes[index].parameters),
        _seed_coefficients(seed.nodes[index].parameters),
      )

  def test_rank_deficiency_is_distinct_from_seed_retention(self) -> None:
    learner = CalibrationProfileLearner(seed_profile())
    # Repeated identical predictors have support but no independent rank.
    for route_counter in (0, 2):
      learner.begin_route(route_sha(route_counter), route_counter=route_counter)
      for _ in range(8):
        predictors = (1.0, 1.0, 1.0, 0.0)
        learner._add_regression(2, "training", predictors, 0.2, 1.0)
        learner._add_regression(2, "moving_training", predictors, 0.2, 1.0)
      learner.end_route()
    report = learner._node_report(2)
    self.assertIn(
      CalibrationQualificationReason.RANK_DEFICIENT_FIT,
      report.reasons,
    )
    self.assertNotIn(
      CalibrationQualificationReason.SEED_RETAINED,
      report.reasons,
    )
    self.assertTrue(all(
      diagnostic.status is CalibrationFitStatus.RANK_DEFICIENT
      for diagnostic in report.fit_diagnostics
    ))

  def test_validation_regression_rejects_frozen_training_choice(self) -> None:
    seed = seed_profile()
    learner = CalibrationProfileLearner(seed)
    for route_counter in (0, 2):
      add_identifiable_route(learner, 10.0, 600, route_counter)
    for route_counter in (1, 3):
      add_identifiable_route_for_parameters(
        learner,
        10.0,
        600,
        route_counter,
        seed.nodes[2].parameters,
      )
    report = learner._node_report(2)
    self.assertIsNotNone(report.selected_model)
    self.assertIn(
      CalibrationQualificationReason.VALIDATION_REGRESSION,
      report.reasons,
    )
    self.assertNotIn(
      CalibrationQualificationReason.SEED_RETAINED,
      report.reasons,
    )

  def test_runtime_interpolation_is_validated_as_a_complete_profile(self) -> None:
    learner = CalibrationProfileLearner(seed_profile())
    add_complete_evidence(learner)
    nominal = learner.qualify("nominal interpolation")
    self.assertIsNotNone(nominal.candidate_profile)
    assert nominal.candidate_profile is not None
    interval_index = 0
    seed_lower = _seed_coefficients(learner.seed_profile.nodes[0].parameters)
    seed_upper = _seed_coefficients(learner.seed_profile.nodes[1].parameters)
    predictors = (1.0, 0.0, 0.0, 0.0)
    seed_target = 0.5 * (seed_lower[0] + seed_upper[0])
    for route in learner._routes:
      if not route.validation:
        continue
      for _ in range(100):
        route.intervals[interval_index].add(
          predictors,
          seed_target,
          1.0,
          0.5,
        )

    rejected = learner.qualify("interpolation regression")
    self.assertIsNone(rejected.candidate_profile)
    self.assertIn(
      CalibrationQualificationReason.INTERPOLATION_VALIDATION_REGRESSION,
      rejected.interpolation_reports[interval_index].reasons,
    )


if __name__ == "__main__":
  unittest.main()

from __future__ import annotations

from dataclasses import replace
import math
import struct
import unittest

from openpilot.selfdrive.controls.lib.blatv2.observer import (
  DisturbanceObserver,
  ObserverMeasurement,
  ObserverPolicy,
  ObserverStatus,
  instantaneous_disturbance_torque,
  observer_measurement_field_names,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  PhysicalParameters,
)


DT = 0.01


def parameters(
  *,
  qualified: bool = True,
  static_friction: float = 0.09,
  kinetic_friction: float = 0.03,
) -> PhysicalParameters:
  return PhysicalParameters(
    torque_per_lateral_accel=0.4,
    rack_gain_deg_s2_per_torque=2000.0,
    rack_damping_per_s=8.0,
    transport_delay_s=0.12,
    static_friction_torque=static_friction,
    kinetic_friction_torque=kinetic_friction,
    rack_rate_resolution_deg_s=4.0,
    confidence=1.0 if qualified else 0.0,
    qualified=qualified,
  )


def measurement_for_disturbance(
  disturbance_torque: float,
  *,
  rack_rate_deg_s: float = 12.0,
  rack_acceleration_deg_s2: float = 90.0,
  aligning_torque: float = -0.12,
  friction_torque: float = 0.03,
  params: PhysicalParameters | None = None,
  **flags,
) -> ObserverMeasurement:
  plant = parameters() if params is None else params
  applied_torque = (
    aligning_torque
    + friction_torque
    + disturbance_torque
    + (
      rack_acceleration_deg_s2
      + plant.rack_damping_per_s * rack_rate_deg_s
    ) / plant.rack_gain_deg_s2_per_torque
  )
  values = {
    "applied_torque": applied_torque,
    "rack_rate_deg_s": rack_rate_deg_s,
    "rack_acceleration_deg_s2": rack_acceleration_deg_s2,
    "aligning_torque": aligning_torque,
    "friction_torque": friction_torque,
    "lateral_active": True,
    "lateral_valid": True,
    "engagement_boundary": False,
    "model_valid": True,
    "vehicle_state_valid": True,
    "live_parameters_valid": True,
    "steering_pressed": False,
    "actuator_constrained": False,
    "output_constrained": False,
    "standstill": False,
  }
  values.update(flags)
  return ObserverMeasurement(**values)


def seeded_observer(
  *,
  policy: ObserverPolicy | None = None,
  disturbance: float = 0.2,
) -> DisturbanceObserver:
  selected_policy = (
    ObserverPolicy(0.05, 0.8) if policy is None else policy
  )
  observer = DisturbanceObserver(selected_policy, DT)
  for _ in range(20):
    observer.update(
      measurement_for_disturbance(disturbance), parameters(),
    )
  return observer


class TestBLaTv2Observer(unittest.TestCase):
  def test_equation_and_sign_recovery(self) -> None:
    plant = parameters()
    for disturbance in (-0.23, 0.17):
      with self.subTest(disturbance=disturbance):
        measurement = measurement_for_disturbance(
          disturbance, params=plant,
        )
        self.assertAlmostEqual(
          instantaneous_disturbance_torque(measurement, plant),
          disturbance,
          places=15,
        )
        observer = DisturbanceObserver(
          ObserverPolicy(0.1, 0.5), DT,
        )
        result = observer.update(measurement, plant)
        self.assertEqual(result.status, ObserverStatus.ACTIVE)
        self.assertAlmostEqual(
          result.instantaneous_disturbance_torque,
          disturbance,
          places=15,
        )

  def test_exact_first_order_convergence(self) -> None:
    time_constant = 0.2
    disturbance = 0.3
    steps = 80
    observer = DisturbanceObserver(
      ObserverPolicy(time_constant, 0.8), DT,
    )
    result = None
    for _ in range(steps):
      result = observer.update(
        measurement_for_disturbance(disturbance), parameters(),
      )
    self.assertIsNotNone(result)
    expected = disturbance * (
      1.0 - math.exp(-steps * DT / time_constant)
    )
    self.assertAlmostEqual(
      result.estimated_disturbance_torque, expected, places=14,
    )
    self.assertFalse(result.saturated)

  def test_bound_is_independent_of_friction(self) -> None:
    plant = parameters(static_friction=0.40, kinetic_friction=0.20)
    bound = 0.27
    observer = DisturbanceObserver(
      ObserverPolicy(0.01, bound), DT,
    )
    result = None
    for _ in range(30):
      result = observer.update(
        measurement_for_disturbance(
          1.0,
          friction_torque=0.20,
          params=plant,
        ),
        plant,
      )
    self.assertIsNotNone(result)
    self.assertEqual(result.estimated_disturbance_torque, bound)
    self.assertTrue(result.saturated)
    self.assertNotEqual(
      result.estimated_disturbance_torque,
      plant.static_friction_torque,
    )

  def test_every_reset_transition_zeroes_state(self) -> None:
    reset_cases = (
      (
        {"engagement_boundary": True},
        ObserverStatus.RESET_ENGAGEMENT_BOUNDARY,
      ),
      (
        {"lateral_active": False},
        ObserverStatus.RESET_LATERAL_INACTIVE,
      ),
      (
        {"lateral_valid": False},
        ObserverStatus.RESET_LATERAL_INVALID,
      ),
      ({"standstill": True}, ObserverStatus.RESET_STANDSTILL),
      (
        {"model_valid": False},
        ObserverStatus.RESET_MODEL_INVALID,
      ),
      (
        {"vehicle_state_valid": False},
        ObserverStatus.RESET_VEHICLE_INVALID,
      ),
      (
        {"live_parameters_valid": False},
        ObserverStatus.RESET_LIVE_PARAMETERS_INVALID,
      ),
    )
    for flags, expected_status in reset_cases:
      with self.subTest(status=expected_status):
        observer = seeded_observer()
        self.assertNotEqual(observer.estimate_torque, 0.0)
        result = observer.update(
          measurement_for_disturbance(-0.7, **flags), parameters(),
        )
        self.assertEqual(result.status, expected_status)
        self.assertEqual(result.estimated_disturbance_torque, 0.0)
        self.assertEqual(observer.estimate_torque, 0.0)
        self.assertFalse(result.saturated)

  def test_nonfinite_measurement_resets(self) -> None:
    observer = seeded_observer()
    invalid = replace(
      measurement_for_disturbance(0.2),
      rack_acceleration_deg_s2=math.nan,
    )
    result = observer.update(invalid, parameters())
    self.assertEqual(
      result.status, ObserverStatus.RESET_MEASUREMENT_INVALID,
    )
    self.assertEqual(result.estimated_disturbance_torque, 0.0)
    self.assertEqual(observer.estimate_torque, 0.0)

  def test_driver_and_constraint_freeze_without_contamination(self) -> None:
    freeze_cases = (
      (
        {"steering_pressed": True},
        ObserverStatus.FROZEN_STEERING_PRESSED,
      ),
      (
        {"actuator_constrained": True},
        ObserverStatus.FROZEN_OUTPUT_CONSTRAINT,
      ),
      (
        {"output_constrained": True},
        ObserverStatus.FROZEN_OUTPUT_CONSTRAINT,
      ),
    )
    for flags, expected_status in freeze_cases:
      with self.subTest(status=expected_status, flags=flags):
        observer = seeded_observer(disturbance=0.2)
        before = observer.estimate_torque
        result = observer.update(
          measurement_for_disturbance(-0.75, **flags), parameters(),
        )
        self.assertEqual(result.status, expected_status)
        self.assertEqual(result.estimated_disturbance_torque, before)
        self.assertEqual(observer.estimate_torque, before)
        self.assertEqual(result.instantaneous_disturbance_torque, 0.0)
        self.assertFalse(result.saturated)

  def test_reset_precedes_freeze(self) -> None:
    observer = seeded_observer()
    result = observer.update(
      measurement_for_disturbance(
        -0.7,
        model_valid=False,
        steering_pressed=True,
        output_constrained=True,
      ),
      parameters(),
    )
    self.assertEqual(result.status, ObserverStatus.RESET_MODEL_INVALID)
    self.assertEqual(result.estimated_disturbance_torque, 0.0)

  def test_left_right_symmetry(self) -> None:
    positive = DisturbanceObserver(
      ObserverPolicy(0.08, 0.7), DT,
    )
    negative = DisturbanceObserver(
      ObserverPolicy(0.08, 0.7), DT,
    )
    plant = parameters()
    for index in range(50):
      disturbance = 0.25 + 0.03 * math.sin(index * 0.2)
      positive_measurement = measurement_for_disturbance(
        disturbance,
        rack_rate_deg_s=12.0,
        rack_acceleration_deg_s2=80.0,
        aligning_torque=-0.1,
        friction_torque=0.03,
        params=plant,
      )
      negative_measurement = ObserverMeasurement(
        applied_torque=-positive_measurement.applied_torque,
        rack_rate_deg_s=-positive_measurement.rack_rate_deg_s,
        rack_acceleration_deg_s2=(
          -positive_measurement.rack_acceleration_deg_s2
        ),
        aligning_torque=-positive_measurement.aligning_torque,
        friction_torque=-positive_measurement.friction_torque,
        lateral_active=True,
        lateral_valid=True,
        engagement_boundary=False,
        model_valid=True,
        vehicle_state_valid=True,
        live_parameters_valid=True,
        steering_pressed=False,
        actuator_constrained=False,
        output_constrained=False,
        standstill=False,
      )
      positive_result = positive.update(positive_measurement, plant)
      negative_result = negative.update(negative_measurement, plant)
      self.assertAlmostEqual(
        negative_result.instantaneous_disturbance_torque,
        -positive_result.instantaneous_disturbance_torque,
        places=15,
      )
      self.assertAlmostEqual(
        negative_result.estimated_disturbance_torque,
        -positive_result.estimated_disturbance_torque,
        places=15,
      )

  def test_repeated_trace_is_byte_deterministic(self) -> None:
    def replay() -> bytes:
      observer = DisturbanceObserver(
        ObserverPolicy(0.07, 0.6), DT,
      )
      encoded = bytearray()
      for index in range(200):
        disturbance = 0.18 * math.sin(index * 0.071)
        result = observer.update(
          measurement_for_disturbance(disturbance), parameters(),
        )
        encoded.extend(struct.pack(
          "<idd?",
          int(result.status),
          result.instantaneous_disturbance_torque,
          result.estimated_disturbance_torque,
          result.saturated,
        ))
      return bytes(encoded)

    expected = replay()
    for _ in range(5):
      self.assertEqual(replay(), expected)

  def test_input_api_cannot_receive_candidate_or_desired_values(self) -> None:
    names = observer_measurement_field_names()
    for forbidden in ("candidate", "desired", "command", "request"):
      self.assertFalse(
        any(forbidden in name.lower() for name in names),
        msg=f"forbidden observer input: {forbidden}",
      )
    self.assertIn("applied_torque", names)
    self.assertFalse(hasattr(DisturbanceObserver, "set_estimate"))

  def test_disabled_state_requires_explicit_policy_and_qualified_profile(self) -> None:
    with self.assertRaises(TypeError):
      DisturbanceObserver()
    with self.assertRaises(ValueError):
      ObserverPolicy(0.0, 0.5)
    with self.assertRaises(ValueError):
      ObserverPolicy(0.1, math.inf)
    with self.assertRaises(ValueError):
      DisturbanceObserver(ObserverPolicy(0.1, 0.5), 0.0)

    no_policy = DisturbanceObserver(None, DT)
    result = no_policy.update(
      measurement_for_disturbance(0.3), parameters(),
    )
    self.assertEqual(result.status, ObserverStatus.DISABLED_NO_POLICY)
    self.assertEqual(result.estimated_disturbance_torque, 0.0)

    unqualified = DisturbanceObserver(
      ObserverPolicy(0.1, 0.5), DT,
    )
    result = unqualified.update(
      measurement_for_disturbance(0.3),
      parameters(qualified=False),
    )
    self.assertEqual(
      result.status, ObserverStatus.DISABLED_UNQUALIFIED_PROFILE,
    )
    self.assertEqual(result.estimated_disturbance_torque, 0.0)


if __name__ == "__main__":
  unittest.main()

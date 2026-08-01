import math
import unittest

import numpy as np

from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR
from opendbc.car.vehicle_model import VehicleModel
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import (
  RackMappingSnapshot,
  curvature_from_measured_angle,
  map_reference,
  map_reference_into,
)


def vehicle_model() -> VehicleModel:
  CP = CarInterface.get_non_essential_params(CAR.HYUNDAI_PALISADE)
  return VehicleModel(CP)


def snapshot(
  vehicle_model: VehicleModel,
  *,
  roll: float = 0.0,
  offset: float = 0.0,
  valid: bool = True,
) -> RackMappingSnapshot:
  return RackMappingSnapshot.from_vehicle_model(
    vehicle_model,
    roll_rad=roll,
    angle_offset_deg=offset,
    valid=valid,
  )


def _polynomial_sample(
  time_s: float,
) -> tuple[float, float, float, float, float, float]:
  curvature = 0.004 + 0.006 * time_s - 0.0025 * time_s * time_s
  curvature_rate = 0.006 - 0.005 * time_s
  curvature_acceleration = -0.005
  speed = 16.0 + 1.4 * time_s - 0.3 * time_s * time_s
  speed_rate = 1.4 - 0.6 * time_s
  speed_acceleration = -0.6
  return (
    curvature,
    curvature_rate,
    curvature_acceleration,
    speed,
    speed_rate,
    speed_acceleration,
  )


class TestBlatV2RackMapper(unittest.TestCase):
  def setUp(self):
    self.vehicle_model = vehicle_model()

  def test_scalar_mapping_matches_vehicle_model(self) -> None:
    for speed in (0.0, 0.1, 1.0, 8.0, 20.0, 35.0):
      for curvature in (-0.03, -0.005, 0.0, 0.005, 0.03):
        for roll in (-0.04, 0.0, 0.03):
          with self.subTest(speed=speed, curvature=curvature, roll=roll):
            offset = 1.25
            alignment = snapshot(self.vehicle_model, roll=roll, offset=offset)
            result = map_reference(
              curvature, 0.0, 0.0, speed, 0.0, 0.0,
              alignment, alignment,
            )
            expected = (
              math.degrees(self.vehicle_model.get_steer_from_curvature(
                -curvature, speed, roll,
              ))
              + offset
            )
            self.assertAlmostEqual(result.angle_deg, expected, delta=1e-12)

  def test_curvature_angle_round_trip(self) -> None:
    alignment = snapshot(self.vehicle_model, roll=0.025, offset=-0.7)
    for speed in (0.0, 2.0, 12.0, 30.0):
      for curvature in (-0.04, -0.003, 0.0, 0.006, 0.035):
        with self.subTest(speed=speed, curvature=curvature):
          angle = map_reference(
            curvature, 0.0, 0.0, speed, 0.0, 0.0,
            alignment, alignment,
          )
          recovered = curvature_from_measured_angle(
            angle.angle_deg, speed, alignment, alignment,
          )
          self.assertAlmostEqual(recovered.curvature, curvature, delta=1e-12)

  def test_analytic_derivatives_match_continuous_polynomial(self) -> None:
    alignment = snapshot(self.vehicle_model, roll=0.018, offset=0.4)
    time_s = 0.35
    step = 1e-4

    center = map_reference(
      *_polynomial_sample(time_s), alignment, alignment,
    )
    before_values = _polynomial_sample(time_s - step)
    after_values = _polynomial_sample(time_s + step)
    before = map_reference(
      before_values[0], 0.0, 0.0, before_values[3], 0.0, 0.0,
      alignment, alignment,
    )
    after = map_reference(
      after_values[0], 0.0, 0.0, after_values[3], 0.0, 0.0,
      alignment, alignment,
    )

    finite_rate = (after.angle_deg - before.angle_deg) / (2.0 * step)
    finite_acceleration = (
      after.angle_deg - 2.0 * center.angle_deg + before.angle_deg
    ) / (step * step)
    self.assertAlmostEqual(center.rate_deg_s, finite_rate, delta=1e-6)
    self.assertAlmostEqual(
      center.acceleration_deg_s2, finite_acceleration, delta=2e-4,
    )

  def test_speed_derivatives_contribute(self) -> None:
    alignment = snapshot(self.vehicle_model, roll=0.03)
    moving = map_reference(
      0.012, 0.0, 0.0, 18.0, 2.0, -0.5,
      alignment, alignment,
    )
    stationary = map_reference(
      0.012, 0.0, 0.0, 18.0, 0.0, 0.0,
      alignment, alignment,
    )
    self.assertNotEqual(moving.rate_deg_s, 0.0)
    self.assertNotEqual(moving.acceleration_deg_s2, 0.0)
    self.assertEqual(stationary.rate_deg_s, 0.0)
    self.assertEqual(stationary.acceleration_deg_s2, 0.0)

  def test_constant_reference_has_zero_derivatives(self) -> None:
    alignment = snapshot(self.vehicle_model, roll=0.02)
    result = map_reference(
      0.015, 0.0, 0.0, 10.0, 0.0, 0.0,
      alignment, alignment,
    )
    self.assertEqual(result.rate_deg_s, 0.0)
    self.assertEqual(result.acceleration_deg_s2, 0.0)

  def test_left_right_sign_symmetry(self) -> None:
    alignment = snapshot(self.vehicle_model)
    left = map_reference(
      0.012, 0.004, -0.003, 14.0, 0.5, -0.2,
      alignment, alignment,
    )
    right = map_reference(
      -0.012, -0.004, 0.003, 14.0, 0.5, -0.2,
      alignment, alignment,
    )
    self.assertAlmostEqual(left.angle_deg, -right.angle_deg, delta=1e-12)
    self.assertAlmostEqual(left.rate_deg_s, -right.rate_deg_s, delta=1e-12)
    self.assertAlmostEqual(
      left.acceleration_deg_s2, -right.acceleration_deg_s2, delta=1e-12,
    )

  def test_invalid_live_snapshot_uses_nominal_without_retained_state(self) -> None:
    nominal = snapshot(self.vehicle_model)
    prior_live = snapshot(self.vehicle_model, roll=0.08, offset=7.0)
    invalid_live = snapshot(
      self.vehicle_model, roll=-0.08, offset=-9.0, valid=False,
    )
    prior = map_reference(
      0.01, 0.0, 0.0, 12.0, 0.0, 0.0,
      prior_live, nominal,
    )
    degraded = map_reference(
      0.01, 0.0, 0.0, 12.0, 0.0, 0.0,
      invalid_live, nominal,
    )
    expected = map_reference(
      0.01, 0.0, 0.0, 12.0, 0.0, 0.0,
      nominal, nominal,
    )
    self.assertNotEqual(prior.angle_deg, expected.angle_deg)
    self.assertEqual(degraded.angle_deg, expected.angle_deg)
    self.assertFalse(degraded.valid)
    self.assertTrue(degraded.degraded)

  def test_caller_buffer_api(self) -> None:
    alignment = snapshot(self.vehicle_model)
    angles = [math.nan] * 3
    rates = [math.nan] * 3
    accelerations = [math.nan] * 3
    status = map_reference_into(
      [0.0, 0.01, 0.02],
      [0.0, 0.01, 0.02],
      [0.0, -0.01, -0.02],
      [5.0, 10.0, 15.0],
      [0.0, 0.5, 1.0],
      [0.0, -0.2, -0.4],
      3,
      alignment,
      alignment,
      angles,
      rates,
      accelerations,
    )
    self.assertEqual(status.count, 3)
    self.assertTrue(status.valid)
    self.assertFalse(status.degraded)
    self.assertTrue(np.all(np.isfinite(angles)))
    self.assertTrue(np.all(np.isfinite(rates)))
    self.assertTrue(np.all(np.isfinite(accelerations)))

  def test_nonfinite_and_negative_speed_reject_explicitly(self) -> None:
    alignment = snapshot(self.vehicle_model)
    values_cases = (
      (math.nan, 0.0, 0.0, 10.0, 0.0, 0.0),
      (0.01, math.inf, 0.0, 10.0, 0.0, 0.0),
      (0.01, 0.0, 0.0, -1.0, 0.0, 0.0),
    )
    for values in values_cases:
      with self.subTest(values=values), self.assertRaises(ValueError):
        map_reference(*values, alignment, alignment)

from types import SimpleNamespace
import unittest

from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR, CarControllerParams
from opendbc.car.lateral import apply_driver_steer_torque_limits
from openpilot.selfdrive.controls.lib.blatv2.actuator import (
  RuntimeTorqueLimits,
  apply_torque_envelope,
  apply_torque_envelope_counts,
)


def limits() -> RuntimeTorqueLimits:
  return RuntimeTorqueLimits(
    steer_max=409,
    delta_up=4,
    delta_down=7,
    steer_step=1,
    driver_allowance=50,
    driver_multiplier=2,
    driver_factor=1,
  )


class TestBlatV2Actuator(unittest.TestCase):
  def test_limits_are_loaded_from_detected_vehicle_params(self):
    source = SimpleNamespace(
      STEER_MAX=270,
      STEER_DELTA_UP=2,
      STEER_DELTA_DOWN=3,
      STEER_STEP=2,
      STEER_DRIVER_ALLOWANCE=250,
      STEER_DRIVER_MULTIPLIER=2,
      STEER_DRIVER_FACTOR=1,
    )
    self.assertEqual(
      RuntimeTorqueLimits.from_controller_params(source),
      RuntimeTorqueLimits(270, 2, 3, 2, 250, 2, 1),
    )

  def test_palisade_limits_are_consumed_from_real_opendbc_params(self):
    cp = CarInterface.get_non_essential_params(CAR.HYUNDAI_PALISADE)
    opendbc_limits = CarControllerParams(cp)

    self.assertEqual(
      RuntimeTorqueLimits.from_controller_params(opendbc_limits),
      RuntimeTorqueLimits(
        steer_max=opendbc_limits.STEER_MAX,
        delta_up=opendbc_limits.STEER_DELTA_UP,
        delta_down=opendbc_limits.STEER_DELTA_DOWN,
        steer_step=opendbc_limits.STEER_STEP,
        driver_allowance=opendbc_limits.STEER_DRIVER_ALLOWANCE,
        driver_multiplier=opendbc_limits.STEER_DRIVER_MULTIPLIER,
        driver_factor=opendbc_limits.STEER_DRIVER_FACTOR,
        production_envelope_verified=True,
      ),
    )

  def test_missing_explicit_production_contract_is_shadow_only(self):
    source = SimpleNamespace(
      STEER_MAX=384,
      STEER_DELTA_UP=3,
      STEER_DELTA_DOWN=7,
      STEER_STEP=1,
      STEER_DRIVER_ALLOWANCE=50,
      STEER_DRIVER_MULTIPLIER=2,
      STEER_DRIVER_FACTOR=1,
    )
    runtime_limits = RuntimeTorqueLimits.from_controller_params(source)
    self.assertFalse(runtime_limits.production_envelope_verified)

  def test_build_release_sign_crossing_and_saturation(self):
    cases = (
      (0, 100, 4),
      (100, 0, 93),
      (-100, 0, -93),
      (3, -100, -4),
      (-3, 100, 4),
      (409, 500, 409),
      (-409, -500, -409),
    )
    for previous, requested, expected in cases:
      with self.subTest(previous=previous, requested=requested, expected=expected):
        self.assertEqual(
          apply_torque_envelope_counts(limits(), requested, previous, 0.0),
          expected,
        )

  def test_envelope_is_exactly_opendbc_production_arithmetic(self):
    runtime_limits = limits()
    for previous in range(-409, 410, 17):
      for requested in range(-500, 501, 31):
        for driver in (-200.0, -49.0, 0.0, 49.0, 200.0):
          with self.subTest(previous=previous, requested=requested, driver=driver):
            expected = apply_driver_steer_torque_limits(requested, previous, driver, runtime_limits)
            self.assertEqual(
              apply_torque_envelope_counts(runtime_limits, requested, previous, driver),
              expected,
            )

  def test_normalized_result_reports_integer_quantization_and_constraint(self):
    result = apply_torque_envelope(limits(), 1.0, 0.0, 0.0)
    self.assertEqual(result.requested_counts, 409)
    self.assertEqual(result.applied_counts, 4)
    self.assertEqual(result.applied_torque, 4 / 409)
    self.assertTrue(result.constrained)

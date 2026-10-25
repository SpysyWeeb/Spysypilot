import math
from collections import defaultdict
from types import SimpleNamespace

import pytest

import opendbc.car.interfaces as car_interfaces
from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.carcontroller import CarController
from opendbc.car.hyundai.values import CAR as HYUNDAI, DBC, CarControllerParams
from opendbc.car.lateral import apply_driver_steer_torque_limits
from opendbc.car.structs import car
from opendbc.car.tesla.values import CAR as TESLA
from opendbc.car.toyota.values import CAR as TOYOTA
from openpilot.spysypilot.aol.helpers import (DRIVER_ASSIST_CEILING, MAX_DRIVER_ASSIST_TORQUE, DriverAssistLimits,
                                              driver_assist_envelope, driver_assist_limits_from_cp)


def car_params(platform):
  return interfaces[platform].get_non_essential_params(platform)


PALISADE_CP = car_params(HYUNDAI.HYUNDAI_PALISADE)
PALISADE = driver_assist_limits_from_cp(PALISADE_CP)


def test_floor_and_ceiling_keep_their_meaning():
  assert MAX_DRIVER_ASSIST_TORQUE == 0.5 and DRIVER_ASSIST_CEILING == 1.0


@pytest.mark.parametrize('command', [1.0, 0.8, 1e-3, -1e-3, -0.8, -1.0])
def test_envelope_bounds(command):
  for driver_torque in range(-600, 601, 15):
    assert MAX_DRIVER_ASSIST_TORQUE <= driver_assist_envelope(float(driver_torque), command, PALISADE) <= DRIVER_ASSIST_CEILING


@pytest.mark.parametrize('driver_torque', [-600.0, -300.0, 0.0, 236.0, 600.0])
def test_zero_command_has_no_direction_to_cap(driver_torque):
  assert driver_assist_envelope(driver_torque, 0.0, PALISADE) == DRIVER_ASSIST_CEILING


@pytest.mark.parametrize('command,driver_torque', [(0.8, 236.0), (-0.8, -236.0), (1.0, 151.0), (-1.0, -151.0),
                                                   (0.8, 0.0), (0.8, -50.0), (-0.8, 50.0)])
def test_agreeing_driver_keeps_full_authority(command, driver_torque):
  # exact, not approximate: d >= -ALLOWANCE leaves the platform's own allowance at STEER_MAX
  assert driver_assist_envelope(driver_torque, command, PALISADE) == DRIVER_ASSIST_CEILING


@pytest.mark.parametrize('command,driver_torque', [(0.8, -300.0), (-0.8, 300.0), (0.8, -153.0), (1.0, -1000.0)])
def test_opposing_driver_floors_at_the_old_cap(command, driver_torque):
  assert driver_assist_envelope(driver_torque, command, PALISADE) == MAX_DRIVER_ASSIST_TORQUE


def test_ramp_between_the_breakpoint_and_the_floor():
  # Palisade: 1.0 down to d = -50, then 2/409 per count, reaching the floor at d = -152.25
  assert driver_assist_envelope(-51.0, 0.8, PALISADE) == pytest.approx((409 - 2) / 409)
  assert driver_assist_envelope(-100.0, 0.8, PALISADE) == pytest.approx((409 - 100) / 409)
  assert driver_assist_envelope(-152.25, 0.8, PALISADE) == pytest.approx(MAX_DRIVER_ASSIST_TORQUE)
  assert driver_assist_envelope(100.0, -0.8, PALISADE) == driver_assist_envelope(-100.0, 0.8, PALISADE)


@pytest.mark.parametrize('driver_torque', [float('nan'), float('inf'), -float('inf')])
def test_unknown_driver_torque_gets_the_floor(driver_torque):
  assert driver_assist_envelope(driver_torque, 0.8, PALISADE) == MAX_DRIVER_ASSIST_TORQUE
  assert driver_assist_envelope(driver_torque, -0.8, PALISADE) == MAX_DRIVER_ASSIST_TORQUE


def test_envelope_matches_platform_formula():
  # Ported from BLaTv3 (test_driver_assist_envelope_matches_platform_formula, FM4.9). Iterating the real
  # limiter washes out its rate term; what remains is its driver-allowance term, which the envelope mirrors
  # down to the floor breakpoint. Below it the envelope holds the floor while the platform keeps shrinking,
  # and the platform stays the final authority downstream.
  limits = CarControllerParams(PALISADE_CP)
  floor_d = ((MAX_DRIVER_ASSIST_TORQUE - 1.0) * limits.STEER_MAX / limits.STEER_DRIVER_MULTIPLIER
             - limits.STEER_DRIVER_ALLOWANCE) / limits.STEER_DRIVER_FACTOR
  assert floor_d == pytest.approx(-152.25)
  for direction in (1.0, -1.0):
    request = int(math.copysign(limits.STEER_MAX, direction))
    for driver_torque in range(-400, 401, 8):
      envelope = driver_assist_envelope(float(driver_torque), direction, PALISADE)
      applied = 0
      for _ in range(400):
        applied = apply_driver_steer_torque_limits(request, applied, float(driver_torque), limits)
      if driver_torque * direction < floor_d:
        assert envelope == MAX_DRIVER_ASSIST_TORQUE
        assert abs(applied) < MAX_DRIVER_ASSIST_TORQUE * limits.STEER_MAX
      else:
        assert abs(abs(applied) - envelope * limits.STEER_MAX) <= 1.0


@pytest.mark.parametrize('direction', [1.0, -1.0])
def test_platform_controller_reads_agreement_with_the_same_sign(direction):
  # The Hyundai controller limits actuators.torque * STEER_MAX against CS.out.steeringTorque, so a driver torque
  # of the request's own sign is the one the platform lets through: the convention the envelope assumes.
  request = 0.8 * direction
  for driver_torque, expected_counts in ((250.0 * direction, round(0.8 * 409)), (-250.0 * direction, 409 + (50 - 250) * 2)):
    controller = CarController(DBC[HYUNDAI.HYUNDAI_PALISADE], PALISADE_CP)
    cs = SimpleNamespace(out=car.CarState.new_message(vEgo=10.0, steeringTorque=driver_torque, steeringPressed=True),
                         is_metric=True, lkas11=defaultdict(int))
    cc = car.CarControl.new_message(latActive=True)
    cap = driver_assist_envelope(driver_torque, request, PALISADE)
    cc.actuators.torque = max(-cap, min(cap, request))
    for frame in range(200):
      controller.update(cc.as_reader(), cs, frame * 10_000_000)
    assert controller.apply_torque_last == direction * expected_counts


def test_resolver_reads_the_palisade_default_branch():
  assert PALISADE_CP.brand == 'hyundai' and PALISADE_CP.steerControlType == car.CarParams.SteerControlType.torque
  assert PALISADE == DriverAssistLimits(STEER_MAX=409.0, STEER_DRIVER_ALLOWANCE=50.0, STEER_DRIVER_MULTIPLIER=2.0, STEER_DRIVER_FACTOR=1.0)
  assert driver_assist_limits_from_cp(PALISADE_CP.as_reader()) == PALISADE  # controlsd holds a reader


def test_resolver_follows_the_platform_flag_branch():
  assert driver_assist_limits_from_cp(car_params(HYUNDAI.HYUNDAI_IONIQ_5)) == \
    DriverAssistLimits(STEER_MAX=270.0, STEER_DRIVER_ALLOWANCE=250.0, STEER_DRIVER_MULTIPLIER=2.0, STEER_DRIVER_FACTOR=1.0)


@pytest.mark.parametrize('CP', [
  car.CarParams.new_message(brand='mock', carFingerprint='MOCK'),  # no CarControllerParams at all
  car.CarParams.new_message(brand='nonexistent'),
  car.CarParams.new_message(),
  car_params(TOYOTA.TOYOTA_RAV4),  # torque, but a measured-torque limiter with no driver allowance
  car_params(TESLA.TESLA_MODEL_3),  # CarControllerParams that takes no CP
], ids=['mock', 'unknown-brand', 'empty', 'toyota', 'tesla'])
def test_resolver_without_driver_allowance_limits(CP):
  assert driver_assist_limits_from_cp(CP) is None


class _FullLimits:
  def __init__(self, CP):
    self.STEER_MAX = 300
    self.STEER_DRIVER_ALLOWANCE = 65
    self.STEER_DRIVER_MULTIPLIER = 4
    self.STEER_DRIVER_FACTOR = 100


class _ZeroSteerMax(_FullLimits):
  def __init__(self, CP):
    super().__init__(CP)
    self.STEER_MAX = 0


class _MissingFactor(_FullLimits):
  def __init__(self, CP):
    super().__init__(CP)
    del self.STEER_DRIVER_FACTOR


class _RaisingConstructor:
  def __init__(self, CP):
    raise KeyError(CP.carFingerprint)


@pytest.mark.parametrize('params_class,expected', [
  (_FullLimits, DriverAssistLimits(STEER_MAX=300.0, STEER_DRIVER_ALLOWANCE=65.0, STEER_DRIVER_MULTIPLIER=4.0, STEER_DRIVER_FACTOR=100.0)),
  (_ZeroSteerMax, None), (_MissingFactor, None), (_RaisingConstructor, None),
])
def test_resolver_is_brand_generic(monkeypatch, params_class, expected):
  monkeypatch.setattr(car_interfaces, 'get_interface_attr', lambda attr: {'somebrand': params_class} if attr == 'CarControllerParams' else {})
  assert driver_assist_limits_from_cp(car.CarParams.new_message(brand='somebrand', carFingerprint='SOME_CAR')) == expected

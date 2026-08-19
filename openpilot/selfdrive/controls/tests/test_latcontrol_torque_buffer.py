import math

from openpilot.common.test import OpenpilotTestCase
from openpilot.common.parameterized import parameterized

from openpilot.cereal import log
from opendbc.car.structs import car
from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.values import CAR as HYUNDAI
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque, LAT_ACCEL_REQUEST_BUFFER_SECONDS


def get_controller(car_name):
  CarInterface = interfaces[car_name]
  CP = CarInterface.get_non_essential_params(car_name)
  CI = CarInterface(CP)
  VM = VehicleModel(CP)
  return LatControlTorque(CP.as_reader(), CI, DT_CTRL), VM


def set_curvature(CS, VM, curvature):
  CS.steeringAngleDeg = math.degrees(VM.get_steer_from_curvature(-curvature, CS.vEgo, 0.0))


def run_speed_change(car_name, old_speed, new_speed, curvature):
  controller, VM = get_controller(car_name)
  CS = car.CarState.new_message()
  CS.vEgo = old_speed
  CS.steeringPressed = False
  params = log.VehicleParameters.new_message()
  set_curvature(CS, VM, curvature)

  for _ in range(int(LAT_ACCEL_REQUEST_BUFFER_SECONDS / DT_CTRL)):
    controller.update(True, CS, VM, params, False, curvature, False, 0.2)

  CS.vEgo = new_speed
  set_curvature(CS, VM, curvature)
  output, _, lac_log = controller.update(True, CS, VM, params, False, curvature, False, 0.2)
  return controller, output, lac_log


class TestLatControlTorqueBuffer(OpenpilotTestCase):

  @parameterized.expand([(TOYOTA.TOYOTA_COROLLA_TSS2,), (HYUNDAI.HYUNDAI_PALISADE,)])
  def test_request_buffer_consistency(self, car_name):
    buffer_steps = int(LAT_ACCEL_REQUEST_BUFFER_SECONDS / DT_CTRL)
    controller, VM = get_controller(car_name)

    CS = car.CarState.new_message()
    CS.vEgo = 30
    CS.steeringPressed = False
    params = log.VehicleParameters.new_message()

    for _ in range(buffer_steps):
      controller.update(True, CS, VM, params, False, 0.001, False, 0.2)
    assert all(val != 0 for val in controller.lat_accel_request_buffer)
    assert all(val != 0 for val in controller.curvature_request_buffer)

    for _ in range(buffer_steps):
      controller.update(False, CS, VM, params, False, 0.0, False, 0.2)
    assert all(val == 0 for val in controller.lat_accel_request_buffer)
    assert all(val == 0 for val in controller.curvature_request_buffer)

  def test_palisade_aligns_delayed_curvature_at_current_speed(self):
    for old_speed, new_speed in ((3.0, 3.0), (3.0, 4.0), (4.0, 3.0)):
      results = []
      for curvature in (0.02, -0.02):
        controller, output, lac_log = run_speed_change(HYUNDAI.HYUNDAI_PALISADE, old_speed, new_speed, curvature)
        self.assertEqual(lac_log.version, 4)
        self.assertAlmostEqual(lac_log.desiredLateralAccel, curvature * new_speed ** 2, delta=1e-6)
        self.assertAlmostEqual(lac_log.actualLateralAccel, curvature * new_speed ** 2, delta=1e-6)
        self.assertAlmostEqual(lac_log.error, 0.0, delta=1e-6)
        self.assertAlmostEqual(lac_log.p, controller.pid.p, delta=1e-6)
        results.append((output, lac_log.p))

      self.assertAlmostEqual(results[0][0], -results[1][0], delta=1e-6)
      self.assertAlmostEqual(results[0][1], -results[1][1], delta=1e-6)

  def test_other_cars_keep_legacy_delayed_lateral_acceleration(self):
    curvature = 0.02
    _, _, lac_log = run_speed_change(TOYOTA.TOYOTA_COROLLA_TSS2, 3.0, 4.0, curvature)
    self.assertEqual(lac_log.version, 1)
    self.assertAlmostEqual(lac_log.desiredLateralAccel, curvature * 3.0 ** 2, delta=1e-6)
    self.assertAlmostEqual(lac_log.error, curvature * (3.0 ** 2 - 4.0 ** 2), delta=1e-6)

  def test_palisade_uses_stock_proportional_gain(self):
    controller, VM = get_controller(HYUNDAI.HYUNDAI_PALISADE)
    CS = car.CarState.new_message()
    CS.vEgo = 3.0
    CS.steeringPressed = False
    params = log.VehicleParameters.new_message()
    desired = 0.4
    actual = 0.38
    set_curvature(CS, VM, actual / CS.vEgo ** 2)

    lac_log = None
    for _ in range(int(LAT_ACCEL_REQUEST_BUFFER_SECONDS / DT_CTRL)):
      _, _, lac_log = controller.update(True, CS, VM, params, False, desired / CS.vEgo ** 2, False, 0.2)

    assert lac_log is not None
    self.assertAlmostEqual(lac_log.error, desired - actual, delta=1e-6)
    self.assertAlmostEqual(lac_log.p, 30.0 * (desired - actual), delta=1e-5)
    self.assertAlmostEqual(lac_log.p, controller.pid.p, delta=1e-6)

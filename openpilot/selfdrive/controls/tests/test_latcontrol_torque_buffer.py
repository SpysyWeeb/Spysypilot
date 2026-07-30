import math

from openpilot.common.test import OpenpilotTestCase
from openpilot.common.parameterized import parameterized

from openpilot.cereal import log
from opendbc.car.structs import car
from opendbc.car.car_helpers import interfaces
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque, LAT_ACCEL_REQUEST_BUFFER_SECONDS, VERSION


def get_controller(car_name):
  CarInterface = interfaces[car_name]
  CP = CarInterface.get_non_essential_params(car_name)
  CI = CarInterface(CP)
  VM = VehicleModel(CP)
  controller = LatControlTorque(CP.as_reader(), CI, DT_CTRL)
  return controller, VM

class TestLatControlTorqueBuffer(OpenpilotTestCase):
  @staticmethod
  def set_curvature(car_state, vehicle_model, curvature):
    car_state.steeringAngleDeg = math.degrees(vehicle_model.get_steer_from_curvature(-curvature, car_state.vEgo, 0.0))

  @parameterized.expand([(TOYOTA.TOYOTA_COROLLA_TSS2,)])
  def test_request_buffer_consistency(self, car_name):
    buffer_steps = int(LAT_ACCEL_REQUEST_BUFFER_SECONDS / DT_CTRL)
    controller, VM = get_controller(car_name)

    CS = car.CarState.new_message()
    CS.vEgo = 30
    CS.steeringPressed = False
    params = log.LiveParametersData.new_message()

    for _ in range(buffer_steps):
      controller.update(True, CS, VM, params, False, 0.001, False, 0.2)
    assert all(val != 0 for val in controller.lat_accel_request_buffer)
    assert all(val != 0 for val in controller.curvature_request_buffer)

    for _ in range(buffer_steps):
      controller.update(False, CS, VM, params, False, 0.0, False, 0.2)
    assert all(val == 0 for val in controller.lat_accel_request_buffer)
    assert all(val == 0 for val in controller.curvature_request_buffer)

  @parameterized.expand([(10.0, 20.0), (20.0, 10.0)])
  def test_feedback_uses_current_speed_for_delayed_curvature(self, buffer_speed, feedback_speed):
    controller, vehicle_model = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
    car_state = car.CarState.new_message()
    car_state.steeringPressed = False
    params = log.LiveParametersData.new_message()
    curvature = 0.001
    lateral_delay = 0.2
    buffer_steps = int(LAT_ACCEL_REQUEST_BUFFER_SECONDS / DT_CTRL)

    car_state.vEgo = buffer_speed
    self.set_curvature(car_state, vehicle_model, curvature)
    for _ in range(buffer_steps):
      controller.update(True, car_state, vehicle_model, params, False, curvature, False, lateral_delay)

    car_state.vEgo = feedback_speed
    self.set_curvature(car_state, vehicle_model, curvature)
    _, _, torque_log = controller.update(True, car_state, vehicle_model, params, False, curvature, False, lateral_delay)

    aligned_reference = curvature * feedback_speed**2
    legacy_reference = curvature * buffer_speed**2
    assert torque_log.version == VERSION
    self.assertAlmostEqual(torque_log.error, 0.0, delta=1e-6)
    self.assertAlmostEqual(torque_log.desiredLateralAccel, aligned_reference, delta=1e-6)
    self.assertAlmostEqual(torque_log.actualLateralAccel, aligned_reference, delta=1e-6)
    self.assertAlmostEqual(torque_log.delayedDesiredCurvature, curvature, delta=1e-6)
    self.assertAlmostEqual(torque_log.legacyDesiredLateralAccel, legacy_reference, delta=1e-6)
    self.assertAlmostEqual(torque_log.speedAlignmentCorrection, aligned_reference - legacy_reference, delta=1e-6)

  def test_constant_speed_reference_is_unchanged(self):
    controller, vehicle_model = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
    car_state = car.CarState.new_message()
    car_state.vEgo = 15.0
    car_state.steeringPressed = False
    params = log.LiveParametersData.new_message()
    curvature = -0.002
    self.set_curvature(car_state, vehicle_model, curvature)
    buffer_steps = int(LAT_ACCEL_REQUEST_BUFFER_SECONDS / DT_CTRL)

    torque_log = None
    for _ in range(buffer_steps):
      _, _, torque_log = controller.update(True, car_state, vehicle_model, params, False, curvature, False, 0.2)

    assert torque_log is not None
    expected_lateral_accel = curvature * car_state.vEgo**2
    self.assertAlmostEqual(torque_log.error, 0.0, delta=1e-6)
    self.assertAlmostEqual(torque_log.desiredLateralAccel, expected_lateral_accel, delta=1e-6)
    self.assertAlmostEqual(torque_log.legacyDesiredLateralAccel, expected_lateral_accel, delta=1e-6)
    self.assertAlmostEqual(torque_log.speedAlignmentCorrection, 0.0, delta=1e-7)

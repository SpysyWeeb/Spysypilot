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
  controller = LatControlTorque(CP.as_reader(), CI, DT_CTRL)
  return controller, VM

class TestLatControlTorqueBuffer(OpenpilotTestCase):

  @parameterized.expand([(TOYOTA.TOYOTA_COROLLA_TSS2,)])
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

    for _ in range(buffer_steps):
      controller.update(False, CS, VM, params, False, 0.0, False, 0.2)
    assert all(val == 0 for val in controller.lat_accel_request_buffer)

  def test_palisade_without_rack_trajectory_uses_stock_pid(self):
    controller, vehicle_model = get_controller(HYUNDAI.HYUNDAI_PALISADE)
    state = car.CarState.new_message()
    state.vEgo = 3.0
    params = log.VehicleParameters.new_message()
    controller_log = None
    for _ in range(int(LAT_ACCEL_REQUEST_BUFFER_SECONDS / DT_CTRL)):
      _, _, controller_log = controller.update(True, state, vehicle_model, params, False, .02, False, .2)

    assert controller_log is not None
    self.assertIsNone(controller.rack_trajectory)
    self.assertEqual(controller_log.version, 1)
    self.assertAlmostEqual(controller_log.p, controller.pid.p, delta=1e-6)

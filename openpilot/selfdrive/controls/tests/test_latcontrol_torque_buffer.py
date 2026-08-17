import pytest

from openpilot.common.test import OpenpilotTestCase
from openpilot.common.parameterized import parameterized

from openpilot.cereal import log
from opendbc.car.structs import car
from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.values import CAR as HYUNDAI
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.constants import CV
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

  def test_palisade_low_speed_p_preserves_stock_state_and_high_speed_output(self):
    def settle(controller, VM, speed, curvature=0.02):
      CS = car.CarState.new_message()
      CS.vEgo = speed
      CS.steeringPressed = False
      params = log.VehicleParameters.new_message()
      for _ in range(int(LAT_ACCEL_REQUEST_BUFFER_SECONDS / DT_CTRL)):
        output, _, lac_log = controller.update(True, CS, VM, params, False, curvature, False, 0.2)
      return CS, params, output, lac_log

    candidate, candidate_vm = get_controller(HYUNDAI.HYUNDAI_PALISADE)
    reference, reference_vm = get_controller(HYUNDAI.HYUNDAI_PALISADE)
    reference.palisade_low_speed_kp = False
    CS, params, candidate_output, candidate_log = settle(candidate, candidate_vm, 3.0)
    _, _, reference_output, _ = settle(reference, reference_vm, 3.0)

    assert candidate_log.version == 2
    assert candidate_log.p == pytest.approx(10 * candidate_log.error)
    assert candidate.pid.p == pytest.approx(reference.pid.p)
    assert abs(candidate_output) < abs(reference_output)
    assert candidate.pid.i == pytest.approx(reference.pid.i)
    assert candidate.pid.control == pytest.approx(reference.pid.control)
    assert list(candidate.lat_accel_request_buffer) == pytest.approx(list(reference.lat_accel_request_buffer))
    assert candidate.jerk_filter.x == pytest.approx(reference.jerk_filter.x)

    mirrored, mirrored_vm = get_controller(HYUNDAI.HYUNDAI_PALISADE)
    _, _, mirrored_output, mirrored_log = settle(mirrored, mirrored_vm, 3.0, -0.02)
    assert mirrored_output == pytest.approx(-candidate_output)
    assert mirrored_log.p == pytest.approx(-candidate_log.p)

    at_knot, at_knot_vm = get_controller(HYUNDAI.HYUNDAI_PALISADE)
    at_knot_reference, at_knot_reference_vm = get_controller(HYUNDAI.HYUNDAI_PALISADE)
    at_knot_reference.palisade_low_speed_kp = False
    _, _, _, at_knot_log = settle(at_knot, at_knot_vm, 5.0)
    _, _, _, _ = settle(at_knot_reference, at_knot_reference_vm, 5.0)
    assert at_knot_log.p == pytest.approx(10 * at_knot_log.error)
    assert at_knot.pid.i == pytest.approx(at_knot_reference.pid.i)
    assert at_knot.pid.control == pytest.approx(at_knot_reference.pid.control)

    other, other_vm = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
    _, _, _, other_log = settle(other, other_vm, 3.0)
    assert other_log.version == 1
    assert other_log.p == pytest.approx(other.pid.p)

    CS.vEgo = 15 * CV.MPH_TO_MS
    candidate_output, _, candidate_log = candidate.update(True, CS, candidate_vm, params, False, 0.02, False, 0.2)
    reference_output, _, reference_log = reference.update(True, CS, reference_vm, params, False, 0.02, False, 0.2)
    assert candidate_output == reference_output
    assert candidate_log.p == reference_log.p
    assert candidate.pid.i == pytest.approx(reference.pid.i)
    assert candidate.pid.control == pytest.approx(reference.pid.control)

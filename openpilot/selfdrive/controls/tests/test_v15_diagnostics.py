import ast
from pathlib import Path

import pytest

from openpilot.cereal import log
from opendbc.car.car_helpers import interfaces
from opendbc.car.structs import car
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque, VERSION


def test_v15_controller_version_is_logged():
  car_interface = interfaces[TOYOTA.TOYOTA_COROLLA_TSS2]
  car_params = car_interface.get_non_essential_params(TOYOTA.TOYOTA_COROLLA_TSS2)
  controller = LatControlTorque(car_params.as_reader(), car_interface(car_params), DT_CTRL)
  vehicle_model = VehicleModel(car_params)
  car_state = car.CarState.new_message()
  live_parameters = log.LiveParametersData.new_message()

  _, _, torque_log = controller.update(
    False,
    car_state,
    vehicle_model,
    live_parameters,
    False,
    0.0,
    False,
    0.2,
  )

  assert VERSION == 15
  assert torque_log.version == 15


def test_v15_reference_diagnostics_round_trip():
  torque_log = log.ControlsState.LateralTorqueState.new_message()
  torque_log.scalarAnchorDeviation = 0.125
  torque_log.referencePersistenceGateHold = True

  with log.ControlsState.LateralTorqueState.from_bytes(torque_log.to_bytes()) as decoded:
    assert decoded.scalarAnchorDeviation == pytest.approx(0.125)
    assert decoded.referencePersistenceGateHold is True


def test_controlsd_maps_v15_reference_diagnostics_at_cereal_boundary():
  controlsd_path = Path(__file__).parents[1] / "controlsd.py"
  tree = ast.parse(controlsd_path.read_text())
  assignments = {
    ast.unparse(node.targets[0]): ast.unparse(node.value)
    for node in ast.walk(tree)
    if isinstance(node, ast.Assign) and len(node.targets) == 1
  }

  assert assignments["lac_log.scalarAnchorDeviation"] == "float(reference_log.scalar_anchor_deviation)"
  assert assignments["lac_log.referencePersistenceGateHold"] == "bool(reference_log.persistence_gate_hold)"

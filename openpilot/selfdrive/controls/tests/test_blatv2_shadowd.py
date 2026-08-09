from __future__ import annotations

import ast
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from openpilot.cereal import log
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR
from opendbc.car.structs import car
from openpilot.selfdrive.controls import blatv2_shadowd
from openpilot.selfdrive.controls.blatv2_shadowd import (
  MODULAR_SHADOW_SCHEMA_VERSION,
  PUBLISHED_SERVICES,
  _CanonicalModelSelector,
  ModularShadowRunner,
  assert_no_actuation_publishers,
  controller_params_from_interface,
  populate_shadow_message,
)
from openpilot.selfdrive.controls.lib.blatv2.policy import ControllerPolicy
from openpilot.selfdrive.controls.lib.blatv2.horizon import HorizonPolicy
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  ProvisionalRackDynamics,
  build_runtime_vehicle_bundle,
)
from openpilot.selfdrive.modeld.constants import ModelConstants


def build_runner() -> tuple[ModularShadowRunner, car.CarParams]:
  cp = CarInterface.get_non_essential_params(CAR.HYUNDAI_PALISADE)
  interface = CarInterface(cp)
  controller_params = controller_params_from_interface(interface, cp)
  dynamics = ProvisionalRackDynamics.from_json_file(
    blatv2_shadowd.PROVISIONAL_RACK_DYNAMICS_PATH,
  )
  policy = ControllerPolicy.from_json_file(
    blatv2_shadowd.PROVISIONAL_POLICY_PATH,
  )
  horizon_policy = HorizonPolicy.from_json_file(
    blatv2_shadowd.PROVISIONAL_HORIZON_POLICY_PATH,
  )
  bundle = build_runtime_vehicle_bundle(
    car_params=cp,
    car_interface_or_callback=interface,
    controller_params=controller_params,
    vehicle_identity=str(cp.carFingerprint),
    provisional_rack_dynamics=dynamics,
  )
  return (
    ModularShadowRunner(
      car_params=cp,
      car_interface=interface,
      controller_params=controller_params,
      runtime_bundle=bundle,
      policy=policy,
      horizon_policy=horizon_policy,
    ),
    cp,
  )


def build_model(
  *,
  frame_id: int,
  origin_ns: int,
  desired_curvature_time_s: float = 0.347,
) -> object:
  model = log.ModelDataV2.new_message()
  native_times = list(ModelConstants.T_IDXS)
  planned_speed = 10.0
  scalar_curvature = 0.012
  model.frameId = frame_id
  model.timestampEof = origin_ns
  model.orientationRate.t = native_times
  model.orientationRate.z = [scalar_curvature * planned_speed for _ in native_times]
  model.velocity.t = native_times
  model.velocity.x = [planned_speed for _ in native_times]
  model.action.desiredCurvature = scalar_curvature
  model.action.desiredCurvatureTime = desired_curvature_time_s
  return model


def build_vehicle_messages(cp: car.CarParams) -> tuple[object, ...]:
  car_state = car.CarState.new_message()
  car_state.vEgo = 10.0
  car_state.steeringAngleDeg = 0.0
  car_state.steeringRateDeg = 0.0
  car_state.steeringTorque = 0.0
  car_state.steeringPressed = False
  car_state.standstill = False

  car_control = car.CarControl.new_message()
  car_control.latActive = True
  car_control.actuators.torque = 0.0

  car_output = car.CarOutput.new_message()
  car_output.actuatorsOutput.torque = 0.0
  car_output.actuatorsOutput.torqueOutputCan = 0.0
  car_output.actuatorsOutput.steeringRequestActive = True
  car_output.actuatorsOutput.steeringRequestActiveValid = True
  car_output.actuatorsOutput.steeringRequestFaultAvoidanceCounter = 0

  selfdrive_state = log.SelfdriveState.new_message()
  selfdrive_state.active = True

  live_parameters = log.LiveParametersData.new_message()
  live_parameters.valid = True
  live_parameters.angleOffsetValid = True
  live_parameters.steerRatioValid = True
  live_parameters.stiffnessFactorValid = True
  live_parameters.angleOffsetDeg = 0.0
  live_parameters.stiffnessFactor = cp.tireStiffnessFactor
  live_parameters.steerRatio = cp.steerRatio
  live_parameters.roll = 0.0
  return (
    car_state,
    car_control,
    car_output,
    selfdrive_state,
    live_parameters,
  )


def run_valid_core_frame(
  runner: ModularShadowRunner,
  cp: car.CarParams,
) -> tuple[int, object]:
  origin_ns = 10_000_000_000
  messages = build_vehicle_messages(cp)
  result = None
  control_ns = 0
  for frame in range(20):
    control_ns = origin_ns + 200_000_000 + frame * 10_000_000
    publication_ns = control_ns - 10_000_000
    result = runner.update(
      state_sample_mono_ns=control_ns - 5_000_000,
      control_witness_mono_ns=control_ns,
      model_publication_mono_ns=publication_ns,
      model_message=build_model(
        frame_id=100 + frame,
        origin_ns=origin_ns,
      ),
      car_state=messages[0],
      car_control=messages[1],
      car_output=messages[2],
      selfdrive_state=messages[3],
      live_parameters=messages[4],
      model_message_valid=True,
      model_message_alive=True,
      vehicle_inputs_valid=True,
      live_parameters_inputs_valid=True,
    )
  assert result is not None and result.valid
  assert runner.valid
  return control_ns, result


class Snapshot:
  def __init__(self, name: str):
    self.name = name

  def as_builder(self) -> Snapshot:
    return self


def test_canonical_model_selection_current_cached_and_unresolved() -> None:
  selector = _CanonicalModelSelector()
  first = Snapshot("first")
  second = Snapshot("second")

  assert selector.select(
    controls_model_mono_ns=100,
    current_message=first,
    current_mono_ns=100,
    current_valid=True,
    current_alive=True,
    current_available=True,
  )
  assert selector.selected_message is first

  # SubMaster has exposed the next publication, but controlsd's witness still
  # identifies the immediately preceding cached model.
  assert selector.select(
    controls_model_mono_ns=100,
    current_message=second,
    current_mono_ns=200,
    current_valid=True,
    current_alive=True,
    current_available=True,
  )
  assert selector.selected_message is first
  assert selector.selected_mono_ns == 100

  assert not selector.select(
    controls_model_mono_ns=150,
    current_message=second,
    current_mono_ns=200,
    current_valid=True,
    current_alive=True,
    current_available=True,
  )
  assert selector.selected_message is None
  assert selector.selected_mono_ns == 0


def test_controller_params_follow_detected_interface_without_platform_path() -> None:
  cp = CarInterface.get_non_essential_params(CAR.HYUNDAI_PALISADE)
  interface = CarInterface(cp)
  params = controller_params_from_interface(interface, cp)
  assert type(params).__module__ == interface.CarController.__module__.replace(
    "carcontroller",
    "values",
  )
  assert (
    params.STEER_MAX,
    params.STEER_DELTA_UP,
    params.STEER_DELTA_DOWN,
  ) == (409, 4, 7)


def test_real_core_frame_populates_and_serializes_every_modular_field() -> None:
  runner, cp = build_runner()
  control_ns, result = run_valid_core_frame(runner, cp)

  # Exercise the actual numpy-scalar boundary that pycapnp rejects without
  # explicit float/int/bool coercion.
  runner.feasible_torque = np.float64(runner.feasible_torque)
  runner.unmet_torque = np.float64(runner.unmet_torque)
  result.raw_torque = np.float64(result.raw_torque)
  result.observer_status = np.uint8(result.observer_status)

  event = log.Event.new_message()
  shadow = event.init("blatV2Shadow")
  populate_shadow_message(
    event,
    shadow,
    runner,
    log_mono_time_ns=control_ns,
    compute_time_seconds=np.float64(0.0004),
  )
  encoded = event.to_bytes()
  with log.Event.from_bytes(encoded) as decoded:
    assert decoded.which() == "blatV2Shadow"
    output = decoded.blatV2Shadow
    assert decoded.valid
    assert output.modularSchemaVersion == MODULAR_SHADOW_SCHEMA_VERSION
    assert output.modularValid
    assert output.modularIntentUsable
    assert output.modularReferenceValid
    assert output.modularActuationEnvelopeVerified
    assert not output.modularProfileQualified
    assert not output.modularScalarOnly
    assert not output.modularNominalMappingUsed
    assert output.modularModelFrameId == result.model_frame_id
    assert output.modularRuntimeVehicleIdentityHash == (runner.runtime_vehicle_identity_hash)
    assert output.modularPolicyHash == runner.policy_hash
    assert output.modularHorizonPolicyHash == runner.horizon_policy_hash
    assert output.modularProfileHash == runner.profile_hash
    assert output.modularDesiredCurvature == result.desired_curvature
    assert output.modularRawTorque == result.raw_torque
    assert output.modularFeasibleTorque == runner.feasible_torque
    assert output.modularUnmetTorque == runner.unmet_torque
    assert output.modularComputeTimeSeconds == 0.0004
    assert output.modularStateSampleMonoTime == control_ns - 5_000_000
    assert output.modularControlWitnessMonoTime == control_ns
    assert output.modularStateAgeSeconds == result.state_age_s
    assert (
      output.modularTotalPredictionHorizonSeconds
      == result.total_prediction_horizon_s
    )
    assert math.isclose(
      output.modularDesiredCurvatureTimeSeconds,
      float(np.float32(0.347)),
      rel_tol=0.0,
      abs_tol=1e-12,
    )
    assert output.mpcCommandTorque == 0.0

    for name, field in log.BlatV2Shadow.schema.fields.items():
      ordinal = int(field.proto.ordinal.explicit)
      if ordinal < 103 or ordinal == 128:
        continue
      field_type = field.proto.slot.type.which()
      if field_type == "float64":
        assert math.isfinite(getattr(output, name)), name


def test_feasibility_is_one_step_from_previous_command_only() -> None:
  runner, cp = build_runner()
  _, result = run_valid_core_frame(runner, cp)
  expected = blatv2_shadowd.apply_torque_envelope(
    runner.runtime_bundle.torque_limits,
    result.planned_torque,
    runner.measured_previous_applied_torque,
    runner.measured_driver_torque,
  )
  assert runner.feasible_torque == expected.applied_torque
  assert runner.unmet_torque == (result.planned_torque - expected.applied_torque)
  assert runner.core_result.raw_torque == result.raw_torque


def test_repeated_shadow_trace_is_byte_exact_when_timing_is_excluded() -> None:
  first, first_cp = build_runner()
  second, second_cp = build_runner()
  first_time, _ = run_valid_core_frame(first, first_cp)
  second_time, _ = run_valid_core_frame(second, second_cp)
  assert first_time == second_time

  messages = []
  for runner in (first, second):
    event = log.Event.new_message()
    populate_shadow_message(
      event,
      event.init("blatV2Shadow"),
      runner,
      log_mono_time_ns=first_time,
      # Environment timing is excluded from deterministic parity. Holding it
      # equal here proves every remaining publisher field is byte-exact.
      compute_time_seconds=0.0,
    )
    messages.append(event.to_bytes())
  assert messages[0] == messages[1]


def test_shadow_signs_unsigned_rack_rate_from_angle_motion() -> None:
  runner, cp = build_runner()
  messages = build_vehicle_messages(cp)
  messages[0].steeringRateDeg = 8.0
  origin_ns = 10_000_000_000
  result = None
  for frame, angle in enumerate((0.0, -0.1, -0.2)):
    messages[0].steeringAngleDeg = angle
    control_ns = origin_ns + 200_000_000 + frame * 10_000_000
    result = runner.update(
      state_sample_mono_ns=control_ns - 5_000_000,
      control_witness_mono_ns=control_ns,
      model_publication_mono_ns=control_ns - 10_000_000,
      model_message=build_model(frame_id=100 + frame, origin_ns=origin_ns),
      car_state=messages[0],
      car_control=messages[1],
      car_output=messages[2],
      selfdrive_state=messages[3],
      live_parameters=messages[4],
      model_message_valid=True,
      model_message_alive=True,
      vehicle_inputs_valid=True,
      live_parameters_inputs_valid=True,
    )

  assert result is not None and result.valid
  assert result.measured_rate_deg_s == -8.0
  assert runner.measured_acceleration_deg_s2 == 0.0

  messages[4].steerRatio = -1.0
  runner.update(
    state_sample_mono_ns=origin_ns + 225_000_000,
    control_witness_mono_ns=origin_ns + 230_000_000,
    model_publication_mono_ns=origin_ns + 220_000_000,
    model_message=build_model(frame_id=103, origin_ns=origin_ns),
    car_state=messages[0],
    car_control=messages[1],
    car_output=messages[2],
    selfdrive_state=messages[3],
    live_parameters=messages[4],
    model_message_valid=True,
    model_message_alive=True,
    vehicle_inputs_valid=True,
    live_parameters_inputs_valid=True,
  )
  assert not runner.lateral_valid


def test_invalid_model_clears_plan_and_publishes_invalid_without_crashing() -> None:
  runner, cp = build_runner()
  control_ns, _ = run_valid_core_frame(runner, cp)
  assert any(value != 0.0 for value in runner.plan_times_s[1:])
  messages = build_vehicle_messages(cp)

  result = runner.update(
    state_sample_mono_ns=control_ns + 5_000_000,
    control_witness_mono_ns=control_ns + 10_000_000,
    model_publication_mono_ns=0,
    model_message=object(),
    car_state=messages[0],
    car_control=messages[1],
    car_output=messages[2],
    selfdrive_state=messages[3],
    live_parameters=messages[4],
    model_message_valid=False,
    model_message_alive=False,
    vehicle_inputs_valid=True,
    live_parameters_inputs_valid=True,
  )
  assert not result.valid
  assert not runner.valid
  assert not runner.model_input_valid
  assert all(value == 0.0 for value in runner.plan_times_s)
  assert all(value == 0.0 for value in runner.orientation_rates_z)
  assert all(value == 0.0 for value in runner.velocities_x)

  event = log.Event.new_message()
  populate_shadow_message(
    event,
    event.init("blatV2Shadow"),
    runner,
    log_mono_time_ns=control_ns + 10_000_000,
    compute_time_seconds=0.0,
  )
  with log.Event.from_bytes(event.to_bytes()) as decoded:
    assert not decoded.valid
    assert not decoded.blatV2Shadow.modularValid
    assert not decoded.blatV2Shadow.modularIntentUsable
    assert decoded.blatV2Shadow.modularDesiredCurvature == 0.0


def test_no_actuation_structure_and_startup_assertion() -> None:
  assert PUBLISHED_SERVICES == ("blatV2Shadow",)
  assert_no_actuation_publishers()
  with unittest.TestCase().assertRaises(AssertionError):
    assert_no_actuation_publishers(("blatV2Shadow", "carControl"))

  def reject_startup(_services=PUBLISHED_SERVICES):
    raise RuntimeError("startup assertion reached")

  with (
    patch.object(
      blatv2_shadowd,
      "assert_no_actuation_publishers",
      reject_startup,
    ),
    unittest.TestCase().assertRaisesRegex(
      RuntimeError,
      "startup assertion reached",
    ),
  ):
    blatv2_shadowd.BlatV2Shadow()


def test_shadow_runner_is_not_manager_registered() -> None:
  config_path = Path(__file__).parents[3] / "system" / "manager" / "process_config.py"
  tree = ast.parse(config_path.read_text(encoding="utf-8"))
  calls = [
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Name)
    and node.func.id == "PythonProcess"
    and node.args
    and isinstance(node.args[0], ast.Constant)
    and node.args[0].value == "blatv2_shadowd"
  ]
  assert calls == []


def test_main_uses_roaming_low_priority_affinity() -> None:
  calls = []

  class FinishedShadow:
    def run(self) -> None:
      calls.append(("run",))

  with (
    patch.object(
      blatv2_shadowd,
      "config_realtime_process",
      lambda cores, priority: calls.append((cores, priority)),
    ),
    patch.object(blatv2_shadowd, "BlatV2Shadow", FinishedShadow),
  ):
    blatv2_shadowd.main()
  assert calls == [
    ([0, 1, 2, 3, 4], blatv2_shadowd.Priority.CTRL_LOW),
    ("run",),
  ]


def test_step_uses_car_state_sample_and_direct_precompute_witness() -> None:
  origin_ns = 10_000_000_000
  publication_ns = 10_190_000_000
  state_sample_ns = 10_195_000_000
  witness_ns = 10_200_000_000
  runner_calls = []
  publisher_calls = []

  controls_state = log.ControlsState.new_message()
  controls_state.lateralPlanMonoTime = publication_ns

  class ModelSnapshot:
    def __init__(self, model):
      self.model = model

    def __getattr__(self, name: str):
      return getattr(self.model, name)

    def as_builder(self):
      return self

  services = {
    "modelV2": ModelSnapshot(
      build_model(frame_id=7, origin_ns=origin_ns),
    ),
    "controlsState": controls_state,
    "carState": object(),
    "carControl": object(),
    "carOutput": object(),
    "selfdriveState": object(),
    "liveParameters": object(),
  }

  class FakeSubMaster:
    updated = {"controlsState": True}
    seen = dict.fromkeys(services, True)
    valid = dict.fromkeys(services, True)
    alive = dict.fromkeys(services, True)
    logMonoTime = {
      **dict.fromkeys(services, state_sample_ns),
      "modelV2": publication_ns,
      "carState": state_sample_ns,
    }

    def update(self) -> None:
      return None

    def __getitem__(self, service: str):
      return services[service]

    def all_checks(self, selected_services: list[str]) -> bool:
      return all(name in services for name in selected_services)

  class FakeRunner:
    def update(self, **kwargs):
      runner_calls.append(kwargs)

  class FakePubMaster:
    def send(self, service: str, message: object) -> None:
      publisher_calls.append(("send", service, message))

  process = blatv2_shadowd.BlatV2Shadow.__new__(
    blatv2_shadowd.BlatV2Shadow,
  )
  process.sm = FakeSubMaster()
  process.pm = FakePubMaster()
  process.runner = FakeRunner()
  process.message = object()
  process.shadow = object()
  process.model_selector = _CanonicalModelSelector()

  def capture_publish(
    message,
    shadow,
    runner,
    *,
    log_mono_time_ns,
    compute_time_seconds,
  ):
    publisher_calls.append(
      (
        "populate",
        message,
        shadow,
        runner,
        log_mono_time_ns,
        compute_time_seconds,
      ),
    )

  with (
    patch.object(
      blatv2_shadowd,
      "_control_witness_mono_ns",
      return_value=witness_ns,
    ),
    patch.object(
      blatv2_shadowd,
      "populate_shadow_message",
      capture_publish,
    ),
  ):
    process.step()

  assert len(runner_calls) == 1
  assert runner_calls[0]["state_sample_mono_ns"] == state_sample_ns
  assert runner_calls[0]["control_witness_mono_ns"] == witness_ns
  assert publisher_calls[0][0] == "populate"
  assert publisher_calls[0][4] == witness_ns
  assert publisher_calls[1][:2] == ("send", "blatV2Shadow")

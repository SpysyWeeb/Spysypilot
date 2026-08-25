from __future__ import annotations

import importlib
import math
from unittest.mock import patch

from openpilot.cereal import log, messaging
from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.values import CAR as HYUNDAI, CarControllerParams
from opendbc.car.lateral import apply_driver_steer_torque_limits
from opendbc.car.structs import car
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.realtime import DT_CTRL
import openpilot.selfdrive.controls.lib.rack_trajectory as rack_trajectory_module
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque, palisade_rack_trajectory_compatible
from openpilot.selfdrive.controls.lib.rack_trajectory import (
  JerkLimitedRackPlanner,
  MAX_DRIVER_ASSIST_TORQUE,
  MEASURED_RATE_FILTER_RC_S,
  MotionLimits,
  RackRateEstimator,
  RackPlan,
  RackReferenceGovernor,
  RackTarget,
  RESPONSE_TIME_S,
  REFERENCE_REVERSAL_RC_S,
  HORIZON_OFFSETS_S,
  HORIZON_ACCELERATION_BLEND,
  HORIZON_POSITION_TOLERANCE_DEG,
  PREVIEW_S,
  horizon_desired_acceleration,
  horizon_candidate_preserves_immediate_path,
  model_path_target,
  model_path_targets,
  PalisadeRackTrajectoryController,
  STATUS_ACTIVE,
  STATUS_INVALID_ACTION_TIME,
  STATUS_INVALID_PATH,
  STATUS_STALE_MODEL,
  STATUS_INVALID_VEHICLE_STATE,
)


class LinearVehicleModel:
  @staticmethod
  def get_steer_from_curvature(curvature: float, speed: float, roll: float) -> float:
    del speed, roll
    return curvature * 10.0


def test_rack_trajectory_components_are_independent_modules() -> None:
  contracts = importlib.import_module("openpilot.selfdrive.controls.lib.rack_trajectory_contracts")
  planner = importlib.import_module("openpilot.selfdrive.controls.lib.rack_trajectory_planner")
  reference = importlib.import_module("openpilot.selfdrive.controls.lib.rack_trajectory_reference")
  state = importlib.import_module("openpilot.selfdrive.controls.lib.rack_trajectory_state")

  assert MotionLimits is contracts.MotionLimits
  assert RackTarget is contracts.RackTarget
  assert JerkLimitedRackPlanner is planner.JerkLimitedRackPlanner
  assert RackReferenceGovernor is reference.RackReferenceGovernor
  assert RackRateEstimator is state.RackRateEstimator
  assert model_path_target is reference.model_path_target


def test_palisade_actual_command_authority_is_409_4_7() -> None:
  car_params = interfaces[HYUNDAI.HYUNDAI_PALISADE].get_non_essential_params(HYUNDAI.HYUNDAI_PALISADE)
  limits = CarControllerParams(car_params)
  assert (limits.STEER_MAX, limits.STEER_DELTA_UP, limits.STEER_DELTA_DOWN) == (409, 4, 7)


def test_model_path_compiles_complete_quarter_second_horizon() -> None:
  targets = model_path_targets(
    native_times_s=[0.0, 1.0, 2.0, 3.0],
    orientation_rates_z=[0.0, .05, .1, .15],
    velocities_x=[5.0] * 4,
    scalar_curvature=.03,
    scalar_action_plan_s=1.0,
    plan_time_now_s=.1,
    measured_v_ego=5.0,
    query_times_s=[.1 + offset for offset in HORIZON_OFFSETS_S],
    vehicle_model=LinearVehicleModel(),
    roll_rad=0.0,
    angle_offset_deg=0.0,
  )
  assert HORIZON_OFFSETS_S == tuple(index * .25 for index in range(9))
  assert PREVIEW_S == .25
  assert len(targets) == len(HORIZON_OFFSETS_S)
  assert targets[0].curvature < targets[-1].curvature


def test_horizon_acceleration_uses_future_shape_without_targeting_endpoint() -> None:
  planner = JerkLimitedRackPlanner(0.0)
  buildup = tuple(
    (offset, RackTarget(0.0 if offset <= .5 else 10.0 * (offset - .5), 0.0))
    for offset in HORIZON_OFFSETS_S if offset > 0.0
  )
  unwind = tuple(
    (offset, RackTarget(5.0 if offset <= .5 else 5.0 - 4.0 * (offset - .5), 0.0))
    for offset in HORIZON_OFFSETS_S if offset > 0.0
  )
  assert horizon_desired_acceleration(planner, buildup) > 0.0
  assert horizon_desired_acceleration(JerkLimitedRackPlanner(5.0), unwind) < 0.0

  endpoint_only = ((2.0, RackTarget(15.0, 0.0)),)
  full_shape = tuple((offset, RackTarget(0.0 if offset < 2.0 else 15.0, 0.0))
                     for offset in HORIZON_OFFSETS_S if offset > 0.0)
  assert abs(horizon_desired_acceleration(planner, full_shape)) < abs(horizon_desired_acceleration(planner, endpoint_only))


def test_horizon_admission_rejects_each_immediate_path_violation() -> None:
  target = RackTarget(1.0, 0.0)
  baseline = RackPlan(.1, 0.0, 0.0, False, False, False)
  allowed = RackPlan(.2, .1, 0.0, False, False, False)
  wrong_side = RackPlan(-.02, 0.0, 0.0, False, False, False)
  position_worse = RackPlan(2.0, 0.0, 0.0, False, False, False)
  rate_worse = RackPlan(.1, 1.0, 0.0, False, False, False)
  assert horizon_candidate_preserves_immediate_path(0.0, target, baseline, allowed)
  assert not horizon_candidate_preserves_immediate_path(0.0, target, baseline, wrong_side)
  assert not horizon_candidate_preserves_immediate_path(0.0, target, baseline, position_worse)
  assert not horizon_candidate_preserves_immediate_path(0.0, target, baseline, rate_worse)


def test_live_horizon_prepares_for_future_shape_without_moving_the_immediate_target() -> None:
  car_interface = interfaces[HYUNDAI.HYUNDAI_PALISADE]
  car_params = car_interface.get_non_essential_params(HYUNDAI.HYUNDAI_PALISADE)
  interface = car_interface(car_params)
  vehicle_model = VehicleModel(car_params)
  params = log.VehicleParameters.new_message()
  state = car.CarState.new_message()
  state.vEgo = 5.0

  def update(orientation_rates: list[float]):
    controller = PalisadeRackTrajectoryController()
    model = messaging.new_message("modelV2").modelV2
    model.timestampEof = 1_000_000_000
    model.action.desiredCurvature = 0.0
    model.action.desiredCurvatureTime = .5
    model.orientationRate.t = [0.0, .5, 1.0, 1.5, 2.0, 2.5]
    model.orientationRate.z = orientation_rates
    model.velocity.x = [state.vEgo] * 6
    controller.set_model(model, 1_050_000_000)
    return controller.update(
      True, state, vehicle_model, params, car_params.lateralTuning.torque,
      interface.torque_from_lateral_accel(), .2, 0.0,
    ), model

  flat, _ = update([0.0] * 6)
  future_turn, future_model = update([0.0, 0.0, .0001, .0002, .0001, 0.0])
  assert flat is not None and future_turn is not None
  assert abs(flat.target_angle_deg - future_turn.target_angle_deg) < 1e-9
  assert abs(future_turn.target_angle_deg) < 1e-9
  assert future_turn.planned_acceleration_deg_s2 != flat.planned_acceleration_deg_s2
  assert abs(future_turn.planned_angle_deg) <= HORIZON_POSITION_TOLERANCE_DEG
  future_targets = model_path_targets(
    native_times_s=future_model.orientationRate.t,
    orientation_rates_z=future_model.orientationRate.z,
    velocities_x=future_model.velocity.x,
    scalar_curvature=0.0,
    scalar_action_plan_s=.5,
    plan_time_now_s=.05,
    measured_v_ego=state.vEgo,
    query_times_s=[.05 + offset for offset in HORIZON_OFFSETS_S],
    vehicle_model=vehicle_model,
    roll_rad=0.0,
    angle_offset_deg=0.0,
  )
  fitted = horizon_desired_acceleration(
    JerkLimitedRackPlanner(0.0),
    tuple((offset, RackTarget(target.angle_deg, target.rate_deg_s))
          for offset, target in zip(HORIZON_OFFSETS_S, future_targets, strict=True) if offset > 0.0),
  )
  assert HORIZON_ACCELERATION_BLEND == .1
  assert abs(future_turn.planned_acceleration_deg_s2 - HORIZON_ACCELERATION_BLEND * fitted) < 1e-9

  original_admission = rack_trajectory_module.horizon_candidate_preserves_immediate_path
  try:
    rack_trajectory_module.horizon_candidate_preserves_immediate_path = lambda *_: False
    fallback, _ = update([0.0, 0.0, .0001, .0002, .0001, 0.0])
  finally:
    rack_trajectory_module.horizon_candidate_preserves_immediate_path = original_admission
  assert fallback is not None
  assert fallback.planned_angle_deg == flat.planned_angle_deg
  assert fallback.planned_rate_deg_s == flat.planned_rate_deg_s
  assert fallback.planned_acceleration_deg_s2 == flat.planned_acceleration_deg_s2


def test_full_horizon_faults_reset_and_recover_cold() -> None:
  car_interface = interfaces[HYUNDAI.HYUNDAI_PALISADE]
  car_params = car_interface.get_non_essential_params(HYUNDAI.HYUNDAI_PALISADE)
  interface = car_interface(car_params)
  vehicle_model = VehicleModel(car_params)
  params = log.VehicleParameters.new_message()
  state = car.CarState.new_message()
  state.vEgo = 5.0

  def model(times, rates, speeds):
    message = messaging.new_message("modelV2").modelV2
    message.timestampEof = 1_000_000_000
    message.action.desiredCurvature = 0.0
    message.action.desiredCurvatureTime = .5
    message.orientationRate.t = times
    message.orientationRate.z = rates
    message.velocity.x = speeds
    return message

  valid = model([0.0, .5, 1.0, 1.5, 2.0, 2.5], [0.0] * 6, [5.0] * 6)

  def update(controller, path, mono_ns=1_050_000_000):
    controller.set_model(path, mono_ns)
    return controller.update(
      True, state, vehicle_model, params, car_params.lateralTuning.torque,
      interface.torque_from_lateral_accel(), .2, 0.0,
    )

  invalid_paths = (
    model([0.0, .5, 1.0], [0.0] * 3, [5.0] * 3),
    model([0.0, .5, 1.0, 1.5, 2.0, 2.5], [0.0] * 5, [5.0] * 6),
    model([0.0, .5, .4, 1.5, 2.0, 2.5], [0.0] * 6, [5.0] * 6),
    model([0.0, .5, 1.0, 1.5, 2.0, 2.5], [0.0, 0.0, math.nan, 0.0, 0.0, 0.0], [5.0] * 6),
    model([0.0, .5, 1.0, 1.5, 2.0, 2.5], [0.0] * 6, [5.0, 5.0, 5.0, 0.0, 5.0, 5.0]),
  )
  cases = tuple((path, 1_050_000_000, STATUS_INVALID_PATH) for path in invalid_paths) + (
    (valid, 1_200_000_001, STATUS_STALE_MODEL),
  )
  for invalid, mono_ns, expected_status in cases:
    controller = PalisadeRackTrajectoryController()
    assert update(controller, valid) is not None
    assert controller.planner is not None and controller.reference_governor.accepted is not None
    assert update(controller, invalid, mono_ns) is None
    assert controller.status == expected_status
    assert controller.planner is None
    assert controller.reference_governor.accepted is None
    recovered = update(controller, valid)
    assert recovered is not None
    assert controller.status == STATUS_ACTIVE
    assert all(math.isfinite(value) for value in (
      recovered.torque, recovered.planned_angle_deg, recovered.planned_rate_deg_s,
    ))


def test_rack_trajectory_is_palisade_not_telluride_scoped() -> None:
  car_interface = interfaces[HYUNDAI.HYUNDAI_PALISADE]

  def params_with_platform_code(code: bytes):
    car_params = car_interface.get_non_essential_params(HYUNDAI.HYUNDAI_PALISADE)
    firmware = car.CarParams.CarFw.new_message()
    firmware.fwVersion = b"\xf1\x00" + code + b" MFC  AT USA LHD 1.00 1.00 99211-S8100 220222"
    car_params.carFw = [firmware]
    return car_params

  assert palisade_rack_trajectory_compatible(params_with_platform_code(b"LX2"))
  assert not palisade_rack_trajectory_compatible(params_with_platform_code(b"ON"))
  assert not palisade_rack_trajectory_compatible(car_interface.get_non_essential_params(HYUNDAI.HYUNDAI_PALISADE))


def govern_reference(governor: RackReferenceGovernor, target: RackTarget, planner: JerkLimitedRackPlanner,
                     model_frame: int, *, bypass: bool = False) -> RackTarget:
  return governor.update(target, planner, 0.0, 1_000_000_000 + model_frame * 50_000_000, .01, bypass)


def test_model_path_is_scalar_anchored_and_signed() -> None:
  times = [0.0, 1.0, 2.0, 3.0]
  speeds = [5.0] * 4
  target = model_path_target(
    native_times_s=times,
    orientation_rates_z=[0.0, .05, .1, .15],
    velocities_x=speeds,
    scalar_curvature=.03,
    scalar_action_plan_s=1.0,
    plan_time_now_s=.1,
    measured_v_ego=5.0,
    query_time_s=2.0,
    vehicle_model=LinearVehicleModel(),
    roll_rad=0.0,
    angle_offset_deg=0.0,
  )
  expected_curvature = .03 + (.1 / 5.0) - (.05 / 5.0)
  assert abs(target.curvature - expected_curvature) < 1e-12
  assert abs(target.angle_deg - math.degrees(-expected_curvature * 10.0)) < 1e-12
  assert target.rate_deg_s < 0.0
  try:
    model_path_target(
      native_times_s=times,
      orientation_rates_z=[0.0, .05, .1, .15],
      velocities_x=speeds,
      scalar_curvature=.03,
      scalar_action_plan_s=1.0,
      plan_time_now_s=.1,
      measured_v_ego=5.0,
      query_time_s=4.0,
      vehicle_model=LinearVehicleModel(),
      roll_rad=0.0,
      angle_offset_deg=0.0,
    )
  except ValueError:
    pass
  else:
    raise AssertionError("out-of-horizon model path accepted")

  stopped = model_path_target(
    native_times_s=times,
    orientation_rates_z=[0.0, .05, .1, .15],
    velocities_x=[5.0, 5.0, 0.0, 0.0],
    scalar_curvature=.03,
    scalar_action_plan_s=.5,
    plan_time_now_s=.1,
    measured_v_ego=5.0,
    query_time_s=1.0,
    vehicle_model=LinearVehicleModel(),
    roll_rad=0.0,
    angle_offset_deg=0.0,
  )
  assert math.isfinite(stopped.angle_deg)


def test_jerk_limited_planner_is_bounded_and_symmetric() -> None:
  limits = MotionLimits(120.0, 500.0, 2500.0)
  right = JerkLimitedRackPlanner(0.0)
  left = JerkLimitedRackPlanner(0.0)
  previous_acceleration = 0.0
  positive = None
  for _ in range(300):
    positive = right.update(RackTarget(90.0, 0.0), limits, .01)
    negative = left.update(RackTarget(-90.0, 0.0), limits, .01)
    assert abs(positive.position_deg + negative.position_deg) < 1e-9
    assert abs(positive.rate_deg_s) <= limits.max_rate_deg_s + 1e-9
    assert abs(positive.acceleration_deg_s2) <= limits.max_acceleration_deg_s2 + 1e-9
    assert abs((positive.acceleration_deg_s2 - previous_acceleration) / .01) <= limits.max_jerk_deg_s3 + 1e-7
    previous_acceleration = positive.acceleration_deg_s2
  assert positive is not None
  assert abs(positive.position_deg - 90.0) < .25
  assert abs(positive.rate_deg_s) < 1.0


def test_envelope_recovery_does_not_drop_torque_for_virtual_or_measured_limit() -> None:
  car_interface = interfaces[HYUNDAI.HYUNDAI_PALISADE]
  car_params = car_interface.get_non_essential_params(HYUNDAI.HYUNDAI_PALISADE)
  interface = car_interface(car_params)
  vehicle_model = VehicleModel(car_params)
  params = log.VehicleParameters.new_message()
  state = car.CarState.new_message()
  state.vEgo = 13.6
  boundary_curvature = -3.0 / state.vEgo ** 2
  boundary_angle = math.degrees(vehicle_model.get_steer_from_curvature(-boundary_curvature, state.vEgo, 0.0))
  target_curvature = -2.99 / state.vEgo ** 2
  model = messaging.new_message("modelV2").modelV2
  model.timestampEof = 1_000_000_000
  model.action.desiredCurvature = target_curvature
  model.action.desiredCurvatureTime = .5
  model.orientationRate.t = [0.0, .5, 1.0, 1.5, 2.0, 2.5]
  model.orientationRate.z = [0.0] * 6
  model.velocity.x = [state.vEgo] * 6

  def update(controller: PalisadeRackTrajectoryController):
    controller.set_model(model, 1_050_000_000)
    return controller.update(
      True, state, vehicle_model, params, car_params.lateralTuning.torque,
      interface.torque_from_lateral_accel(), .2, target_curvature,
    )

  virtual = PalisadeRackTrajectoryController()
  virtual.planner = JerkLimitedRackPlanner(boundary_angle + .2, 20.0)
  raw_planner = virtual.planner
  state.steeringAngleDeg = boundary_angle - 15.0
  output = update(virtual)
  assert output is not None
  executed_curvature = -vehicle_model.calc_curvature(math.radians(output.planned_angle_deg), state.vEgo, 0.0)
  assert abs(executed_curvature) * state.vEgo ** 2 <= 3.0 + 1e-6
  assert output.torque > 0.0
  torques = [output.torque]
  for _ in range(300):
    output = update(virtual)
    assert output is not None and virtual.planner is raw_planner
    executed_curvature = -vehicle_model.calc_curvature(math.radians(output.planned_angle_deg), state.vEgo, 0.0)
    assert abs(executed_curvature) * state.vEgo ** 2 <= 3.0 + 1e-6
    assert output.torque > 0.0
    torques.append(output.torque)
    if raw_planner.position_deg <= boundary_angle:
      break
  else:
    raise AssertionError("raw planner did not recover inside the execution envelope")
  assert max(abs(torques[index] - torques[index - 1]) for index in range(1, len(torques))) < .1

  measured = PalisadeRackTrajectoryController()
  state.steeringAngleDeg = boundary_angle + 10.0
  output = update(measured)
  assert output is not None
  assert output.torque != 0.0
  assert output.feedforward_torque > 0.0
  assert output.position_feedback_torque * (output.planned_angle_deg - state.steeringAngleDeg) > 0.0


def test_signed_rack_rate_handles_signed_and_unsigned_samples() -> None:
  controller = PalisadeRackTrajectoryController()
  assert controller._measured_rate(-1.0, -5.0) == (-5.0, True)
  positive_rate, valid = controller._measured_rate(-.9, 5.0)
  alpha = DT_CTRL / (MEASURED_RATE_FILTER_RC_S + DT_CTRL)
  assert valid and abs(positive_rate - (-5.0 + alpha * 10.0)) < 1e-12

  controller.reset()
  assert controller._measured_rate(1.0, 5.0) == (0.0, False)
  assert controller._measured_rate(1.0, 0.0) == (0.0, True)
  step_rate = 0.0
  for index in range(5):
    step_rate, valid = controller._measured_rate(1.1 + index * .1, 8.0)
  assert valid and abs(step_rate - 8.0 * (1.0 - (1.0 - alpha) ** 5)) < 1e-12
  reversal_rate, valid = controller._measured_rate(1.0, 8.0)
  assert valid and reversal_rate > 0.0
  for index in range(100):
    reversal_rate, valid = controller._measured_rate(.9 - index * .1, 8.0)
  assert valid and abs(reversal_rate + 8.0) < .1

  controller.reset()
  assert controller._measured_rate(1.0, 5.0) == (0.0, False)
  filtered = [controller._measured_rate(1.1 if index % 2 else 1.0, 8.0)[0] for index in range(1, 21)]
  assert sum(abs(filtered[index] - filtered[index - 1]) for index in range(1, len(filtered))) < 8.0 * 19 * .4


def test_unwind_feedforward_releases_hold_torque_continuously() -> None:
  car_interface = interfaces[HYUNDAI.HYUNDAI_PALISADE]
  car_params = car_interface.get_non_essential_params(HYUNDAI.HYUNDAI_PALISADE)
  vehicle_model = VehicleModel(car_params)
  params = log.VehicleParameters.new_message()
  state = car.CarState.new_message()
  state.vEgo = 13.6
  torque_params = car_params.lateralTuning.torque
  torque_params.friction = 0.0
  torque_params.latAccelOffset = 0.0

  def run(planned_angle: float, measured_angle: float, target_angle: float | None = None, feedback_gain: float | None = 0.0):
    controller = PalisadeRackTrajectoryController()
    controller.planner = JerkLimitedRackPlanner(planned_angle)
    state.steeringAngleDeg = measured_angle
    target_angle = planned_angle if target_angle is None else target_angle
    target_curvature = -vehicle_model.calc_curvature(math.radians(target_angle), state.vEgo, 0.0)
    model = messaging.new_message("modelV2").modelV2
    model.timestampEof = 1_000_000_000
    model.action.desiredCurvature = target_curvature
    model.action.desiredCurvatureTime = .5
    model.orientationRate.t = [0.0, .5, 1.0, 1.5, 2.0, 2.5]
    model.orientationRate.z = [0.0] * 6
    model.velocity.x = [state.vEgo] * 6
    controller.set_model(model, 1_050_000_000)
    arguments = (
      True, state, vehicle_model, params, torque_params, lambda lateral_accel, _: lateral_accel, .2, target_curvature,
    )
    if feedback_gain is None:
      output = controller.update(*arguments)
    else:
      with patch.object(controller, "_feedback_gain", return_value=feedback_gain):
        output = controller.update(*arguments)
    assert output is not None
    return output

  hold = run(2.0, 2.0)
  unwinds = [run(2.0, measured_angle) for measured_angle in (2.5, 3.0, 4.0)]
  unwind = unwinds[-1]
  turn_in = run(2.0, 1.0)
  transient_turn_in_hold = run(1.0, 1.0, 4.0)
  transient_turn_in_lag = run(1.0, 2.0, 4.0)
  left_unwind = run(-2.0, -4.0)
  cross_center = run(-2.0, 4.0)
  at_center = run(-2.0, 0.0)
  already_centerward = run(2.0, 4.0, feedback_gain=None)

  assert abs(hold.feedforward_torque) > .1
  assert math.isclose(turn_in.torque, hold.torque, rel_tol=0.0, abs_tol=1e-9)
  assert math.isclose(transient_turn_in_lag.torque, transient_turn_in_hold.torque, rel_tol=0.0, abs_tol=1e-9)
  for target_angle in (1.0, 1.5, 2.0, 2.5):
    directional_hold = run(1.0, 1.0, target_angle)
    directional_lag = run(1.0, 2.0, target_angle)
    intended_angle = directional_lag.target_angle_deg + RESPONSE_TIME_S * directional_lag.target_rate_deg_s
    turn_in_angle = abs(intended_angle) if intended_angle > 0.0 else 0.0
    expected_scale = min(1.0, max(abs(directional_lag.planned_angle_deg), turn_in_angle) / 2.0)
    assert math.isclose(
      directional_lag.feedforward_torque, directional_hold.feedforward_torque, rel_tol=0.0, abs_tol=1e-9,
    )
    assert math.isclose(
      directional_lag.torque, expected_scale * directional_hold.torque, rel_tol=0.0, abs_tol=1e-9,
    )
  for measured_angle, output in zip((2.5, 3.0, 4.0), unwinds, strict=True):
    assert math.isclose(output.feedforward_torque, hold.feedforward_torque, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(output.torque, 2.0 / measured_angle * hold.torque, rel_tol=0.0, abs_tol=1e-9)
  assert math.isclose(left_unwind.feedforward_torque, -unwind.feedforward_torque, rel_tol=0.0, abs_tol=1e-9)
  assert math.isclose(left_unwind.torque, -unwind.torque, rel_tol=0.0, abs_tol=1e-9)
  assert math.isclose(cross_center.torque, -hold.torque, rel_tol=0.0, abs_tol=1e-9)
  assert math.isclose(at_center.torque, -hold.torque, rel_tol=0.0, abs_tol=1e-9)
  assert already_centerward.torque * already_centerward.planned_angle_deg < 0.0
  assert math.isclose(
    already_centerward.torque,
    already_centerward.feedforward_torque + already_centerward.feedback_torque,
    rel_tol=0.0,
    abs_tol=1e-9,
  )

  with patch.object(rack_trajectory_module, "get_friction", return_value=-.04):
    hold_with_friction = run(2.0, 2.0)
    unwind_with_friction = run(2.0, 4.0)
  assert math.isclose(
    hold_with_friction.feedforward_torque - hold.feedforward_torque,
    unwind_with_friction.feedforward_torque - unwind.feedforward_torque,
    rel_tol=0.0,
    abs_tol=1e-9,
  )

  with patch.object(rack_trajectory_module, "get_friction", return_value=-.4):
    outward_cross_center = run(-2.0, 4.0)
    outward_at_center = run(-2.0, 0.0)
  with patch.object(rack_trajectory_module, "get_friction", return_value=.4):
    mirrored_outward_at_center = run(2.0, 0.0)
  assert outward_cross_center.feedforward_torque > 0.0
  assert outward_cross_center.torque == 0.0
  assert outward_at_center.torque == 0.0
  assert mirrored_outward_at_center.torque == 0.0


def test_feedforward_boundary_and_driver_handoff_cap() -> None:
  car_interface = interfaces[HYUNDAI.HYUNDAI_PALISADE]
  car_params = car_interface.get_non_essential_params(HYUNDAI.HYUNDAI_PALISADE)
  interface = car_interface(car_params)
  controller = PalisadeRackTrajectoryController()
  vehicle_model = VehicleModel(car_params)
  state = car.CarState.new_message()
  state.vEgo = 5.0
  params = log.VehicleParameters.new_message()
  model = messaging.new_message("modelV2").modelV2
  model.timestampEof = 1_000_000_000
  model.action.desiredCurvature = -.02
  model.action.desiredCurvatureTime = .5
  model.orientationRate.t = [0.0, .5, 1.0, 1.5, 2.0, 2.5]
  model.orientationRate.z = [0.0] * 6
  model.velocity.x = [5.0] * 6
  torque_from_lateral_accel = interface.torque_from_lateral_accel()

  def update(torque_scale: float = 1.0):
    controller.set_model(model, 1_050_000_000)
    return controller.update(
      True, state, vehicle_model, params, car_params.lateralTuning.torque,
      lambda lateral_accel, torque_params: torque_scale * torque_from_lateral_accel(lateral_accel, torque_params), .2, -.02,
    )

  output = update()
  assert output is not None
  target_angle = output.target_angle_deg
  feedforward = []
  for index in range(100):
    state.steeringAngleDeg = target_angle + (-.2 if index % 2 else .2)
    output = update()
    assert output is not None
    feedforward.append(abs(output.feedforward_torque))
  boundary_steps = [abs(feedforward[index + 1] - feedforward[index]) for index in range(0, len(feedforward), 2)]
  assert max(boundary_steps) < .05

  controller.reset()
  state.steeringAngleDeg = target_angle + 20.0
  output = update()
  assert output is not None
  assert abs(output.feedforward_torque) > .05
  planner_before_override = controller.planner
  assert planner_before_override is not None
  plan_offset_before_override = planner_before_override.position_deg - state.steeringAngleDeg

  state.steeringPressed = True
  state.steeringTorque = -300.0
  state.steeringAngleDeg += 2.0
  handoff_output = update(4.0)
  assert handoff_output is not None
  assert MAX_DRIVER_ASSIST_TORQUE == .5
  normal_torque = max(-1.0, min(1.0, handoff_output.feedforward_torque + handoff_output.feedback_torque))
  assert normal_torque > MAX_DRIVER_ASSIST_TORQUE
  assert handoff_output.torque == MAX_DRIVER_ASSIST_TORQUE
  assert handoff_output.feedback_torque != 0.0
  assert controller.status == STATUS_ACTIVE
  assert not hasattr(controller, "driver_override_resume")

  negative_handoff_output = update(-4.0)
  assert negative_handoff_output is not None
  normal_torque = max(-1.0, min(1.0, negative_handoff_output.feedforward_torque + negative_handoff_output.feedback_torque))
  assert normal_torque < -MAX_DRIVER_ASSIST_TORQUE
  assert negative_handoff_output.torque == -MAX_DRIVER_ASSIST_TORQUE
  assert controller.planner is planner_before_override
  plan_offset_during_override = planner_before_override.position_deg - state.steeringAngleDeg
  assert abs(plan_offset_during_override - plan_offset_before_override) < .25

  state.steeringPressed = False
  state.steeringTorque = 0.0
  release_output = update()
  assert release_output is not None
  assert release_output.feedforward_torque != 0.0
  assert controller.status == STATUS_ACTIVE

  controller.reset()
  state.steeringAngleDeg = target_angle
  state.steeringPressed = True
  state.steeringTorque = -300.0
  subcap_output = update()
  assert subcap_output is not None
  normal_torque = max(-1.0, min(1.0, subcap_output.feedforward_torque + subcap_output.feedback_torque))
  assert abs(normal_torque) < MAX_DRIVER_ASSIST_TORQUE
  assert math.isclose(subcap_output.torque, normal_torque, rel_tol=0.0, abs_tol=1e-9)

  limits = CarControllerParams(car_params)
  capped_request = -round(MAX_DRIVER_ASSIST_TORQUE * limits.STEER_MAX)
  for driver_torque, expected in ((150.0, capped_request), (200.0, -109), (255.0, 0)):
    applied = capped_request
    for _ in range(100):
      applied = apply_driver_steer_torque_limits(capped_request, applied, driver_torque, limits)
    assert applied == expected


def test_reference_governor_holds_short_small_reversal() -> None:
  planner = JerkLimitedRackPlanner(5.0)
  governor = RackReferenceGovernor()

  assert govern_reference(governor, RackTarget(5.0, 0.0), planner, 0) == RackTarget(5.0, 0.0)
  assert govern_reference(governor, RackTarget(5.5, 0.0), planner, 1) == RackTarget(5.5, 0.0)
  accepted = govern_reference(governor, RackTarget(4.9, 0.0), planner, 2)
  for _ in range(20):
    accepted = govern_reference(governor, RackTarget(4.9, 0.0), planner, 2)
    assert 4.9 < accepted.position_deg < 5.5
    assert governor.limited


def test_reference_governor_filters_small_neutral_reversal() -> None:
  planner = JerkLimitedRackPlanner(0.0)
  governor = RackReferenceGovernor()

  govern_reference(governor, RackTarget(0.0, 0.0), planner, 0)
  govern_reference(governor, RackTarget(.4, 0.0), planner, 1)
  accepted = govern_reference(governor, RackTarget(-.2, 0.0), planner, 2)

  assert -.2 < accepted.position_deg < .4
  assert governor.limited


def test_reference_governor_uses_constant_response_and_converges() -> None:
  planner = JerkLimitedRackPlanner(5.0)
  governor = RackReferenceGovernor()
  target = RackTarget(4.9, 0.0)

  assert REFERENCE_REVERSAL_RC_S == .12
  govern_reference(governor, RackTarget(5.0, 0.0), planner, 0)
  govern_reference(governor, RackTarget(5.5, 0.0), planner, 1)
  alpha = .01 / (REFERENCE_REVERSAL_RC_S + .01)
  accepted = governor.accepted
  assert accepted is not None
  for model_frame in range(2, 80):
    for _ in range(5):
      previous = accepted
      accepted = govern_reference(governor, target, planner, model_frame)
      assert abs(accepted.position_deg - (previous.position_deg + alpha * (target.position_deg - previous.position_deg))) < 1e-12
  assert abs(accepted.position_deg - target.position_deg) < 1e-9
  assert not governor.limited


def test_reference_governor_preserves_gradual_neutral_crossing_continuity() -> None:
  planner = JerkLimitedRackPlanner(0.0)
  governor = RackReferenceGovernor()

  govern_reference(governor, RackTarget(0.0, 0.0), planner, 0)
  accepted = govern_reference(governor, RackTarget(.4, 2.0), planner, 1)
  maximum_step = 0.0
  frame_positions = []
  for model_frame, position in enumerate((.3, .2, .1, 0.0, -.1, -.2, -.3, -.4), 2):
    for _ in range(5):
      previous = accepted
      accepted = govern_reference(governor, RackTarget(position, -2.0), planner, model_frame)
      assert accepted.position_deg <= previous.position_deg
      maximum_step = max(maximum_step, previous.position_deg - accepted.position_deg)
    frame_positions.append(accepted.position_deg)

  assert frame_positions[0] > 0.0 > frame_positions[-1]
  assert maximum_step < .1


def test_reference_governor_passes_large_persistent_and_necessary_reversals() -> None:
  large = RackReferenceGovernor()
  stationary = JerkLimitedRackPlanner(5.0)
  govern_reference(large, RackTarget(5.0, 0.0), stationary, 0)
  govern_reference(large, RackTarget(6.0, 0.0), stationary, 1)
  assert govern_reference(large, RackTarget(3.0, 0.0), stationary, 2) == RackTarget(3.0, 0.0)
  assert large.accepted == RackTarget(3.0, 0.0)

  persistent = RackReferenceGovernor()
  govern_reference(persistent, RackTarget(5.0, 0.0), stationary, 0)
  govern_reference(persistent, RackTarget(5.5, 0.0), stationary, 1)
  first = govern_reference(persistent, RackTarget(4.9, 0.0), stationary, 2)
  accepted = first
  for frame in (3, 4, 5):
    accepted = govern_reference(persistent, RackTarget(4.9, 0.0), stationary, frame)
  assert abs(accepted.position_deg - 4.9) < abs(first.position_deg - 4.9)

  necessary = RackReferenceGovernor()
  lagging = JerkLimitedRackPlanner(5.0)
  govern_reference(necessary, RackTarget(5.0, 0.0), lagging, 0)
  govern_reference(necessary, RackTarget(5.5, 0.0), lagging, 1)
  lagging.position_deg = 3.0
  assert govern_reference(necessary, RackTarget(4.9, 0.0), lagging, 2) == RackTarget(4.9, 0.0)
  assert not necessary.limited

  crossing = RackReferenceGovernor()
  govern_reference(crossing, RackTarget(0.0, 0.0), stationary, 0)
  govern_reference(crossing, RackTarget(1.0, 0.0), stationary, 1)
  assert govern_reference(crossing, RackTarget(-.1, 0.0), stationary, 2) == RackTarget(-.1, 0.0)
  assert crossing.accepted == RackTarget(-.1, 0.0)

  fast = RackReferenceGovernor()
  govern_reference(fast, RackTarget(5.0, 0.0), stationary, 0)
  govern_reference(fast, RackTarget(5.5, 0.0), stationary, 1)
  assert govern_reference(fast, RackTarget(5.0, -6.0), stationary, 2) == RackTarget(5.0, -6.0)
  assert not fast.limited

  recovery = RackReferenceGovernor()
  govern_reference(recovery, RackTarget(5.0, 0.0), stationary, 0)
  govern_reference(recovery, RackTarget(5.5, 0.0), stationary, 1)
  govern_reference(recovery, RackTarget(4.9, 0.0), stationary, 2)
  assert recovery.active
  assert govern_reference(recovery, RackTarget(4.8, 0.0), stationary, 3, bypass=True) == RackTarget(4.8, 0.0)
  assert not recovery.active and not recovery.limited and recovery.direction == 0


def test_model_wobble_is_governed_in_rack_space_at_every_speed() -> None:
  car_interface = interfaces[HYUNDAI.HYUNDAI_PALISADE]
  car_params = car_interface.get_non_essential_params(HYUNDAI.HYUNDAI_PALISADE)
  interface = car_interface(car_params)
  vehicle_model = VehicleModel(car_params)
  params = log.VehicleParameters.new_message()

  for speed in (5.0, 15.0, 30.0):
    controller = PalisadeRackTrajectoryController()
    controller.planner = JerkLimitedRackPlanner(5.1)
    state = car.CarState.new_message()
    state.vEgo = speed
    state.steeringAngleDeg = 5.1
    model = messaging.new_message("modelV2").modelV2
    model.timestampEof = 1_000_000_000
    model.action.desiredCurvatureTime = .5
    model.orientationRate.t = [0.0, .5, 1.0, 1.5, 2.0, 2.5]
    model.orientationRate.z = [0.0] * 6
    model.velocity.x = [speed] * 6

    limited = []
    accelerations = []
    for index in range(15):
      model_frame = index // 5
      target_angle = (5.0, 5.5, 5.0)[model_frame]
      desired_curvature = -vehicle_model.calc_curvature(math.radians(target_angle), speed, 0.0)
      model.timestampEof = 1_000_000_000 + model_frame * 50_000_000
      model.action.desiredCurvature = desired_curvature
      controller.set_model(model, model.timestampEof + 50_000_000)
      output = controller.update(
        True, state, vehicle_model, params, car_params.lateralTuning.torque,
        interface.torque_from_lateral_accel(), .2, desired_curvature,
      )
      assert output is not None
      limited.append(controller.reference_governor.limited)
      accelerations.append(output.planned_acceleration_deg_s2)

    assert any(limited[10:])
    assert accelerations[9] * accelerations[10] >= 0.0


def test_reference_governor_threshold_edges_and_raw_feedforward_isolation() -> None:
  stationary = JerkLimitedRackPlanner(5.0)

  below_distance = RackReferenceGovernor()
  govern_reference(below_distance, RackTarget(5.0, 0.0), stationary, 0)
  govern_reference(below_distance, RackTarget(5.5, 0.0), stationary, 1)
  governed = govern_reference(below_distance, RackTarget(4.5001, 0.0), stationary, 2)
  assert governed != RackTarget(4.5001, 0.0)
  assert below_distance.limited

  at_distance = RackReferenceGovernor()
  govern_reference(at_distance, RackTarget(5.0, 0.0), stationary, 0)
  govern_reference(at_distance, RackTarget(5.5, 0.0), stationary, 1)
  assert govern_reference(at_distance, RackTarget(4.5, 0.0), stationary, 2) == RackTarget(4.5, 0.0)
  assert not at_distance.limited

  below_rate = RackReferenceGovernor()
  govern_reference(below_rate, RackTarget(5.0, 0.0), stationary, 0)
  govern_reference(below_rate, RackTarget(5.5, 0.0), stationary, 1)
  assert govern_reference(below_rate, RackTarget(5.0, -4.9999), stationary, 2) != RackTarget(5.0, -4.9999)
  assert below_rate.limited

  at_rate = RackReferenceGovernor()
  govern_reference(at_rate, RackTarget(5.0, 0.0), stationary, 0)
  govern_reference(at_rate, RackTarget(5.5, 0.0), stationary, 1)
  assert govern_reference(at_rate, RackTarget(5.0, -5.0), stationary, 2) == RackTarget(5.0, -5.0)
  assert not at_rate.limited

  car_interface = interfaces[HYUNDAI.HYUNDAI_PALISADE]
  car_params = car_interface.get_non_essential_params(HYUNDAI.HYUNDAI_PALISADE)
  interface = car_interface(car_params)
  controller = PalisadeRackTrajectoryController()
  controller.planner = JerkLimitedRackPlanner(5.1)
  vehicle_model = VehicleModel(car_params)
  state = car.CarState.new_message()
  state.vEgo = 30.0
  state.steeringAngleDeg = 5.1
  params = log.VehicleParameters.new_message()
  torque_params = car_params.lateralTuning.torque
  torque_params.friction = 0.0
  torque_params.latAccelOffset = 0.0
  model = messaging.new_message("modelV2").modelV2
  model.timestampEof = 1_000_000_000
  model.action.desiredCurvatureTime = .5
  model.orientationRate.t = [0.0, .5, 1.0, 1.5, 2.0, 2.5]
  model.orientationRate.z = [0.0] * 6
  model.velocity.x = [state.vEgo] * 6

  output = None
  for model_frame, target_angle in enumerate((5.0, 5.5, 5.0)):
    desired_curvature = -vehicle_model.calc_curvature(math.radians(target_angle), state.vEgo, 0.0)
    model.timestampEof = 1_000_000_000 + model_frame * 50_000_000
    model.action.desiredCurvature = desired_curvature
    controller.set_model(model, model.timestampEof + 50_000_000)
    output = controller.update(
      True, state, vehicle_model, params, torque_params,
      interface.torque_from_lateral_accel(), .2, desired_curvature,
    )
    assert output is not None

  assert output is not None
  assert controller.reference_governor.limited
  expected_planned = -float(interface.torque_from_lateral_accel()(output.desired_lateral_accel, torque_params))
  raw_lateral_accel = output.target_curvature * state.vEgo ** 2
  raw_bypass = -float(interface.torque_from_lateral_accel()(raw_lateral_accel, torque_params))
  assert abs(output.feedforward_torque - expected_planned) < 1e-9
  assert abs(output.feedforward_torque - raw_bypass) > 1e-4


def test_profile_transition_headroom_does_not_walk_outward() -> None:
  controller = PalisadeRackTrajectoryController()
  controller.planner = JerkLimitedRackPlanner(0.0, 300.0)
  controller.planner.acceleration_deg_s2 = 500.0
  profile = MotionLimits(40.0, 100.0, 5000.0)
  limits, transition = controller._motion_limits(profile)
  assert transition
  rate_ceiling = limits.max_rate_deg_s
  acceleration_ceiling = limits.max_acceleration_deg_s2
  for index in range(600):
    recovery_acceleration = controller._recovery_acceleration(profile, transition)
    plan = controller.planner.update(RackTarget(10_000.0, 10_000.0), limits, .01, recovery_acceleration)
    limits, transition = controller._motion_limits(profile)
    assert limits.max_rate_deg_s <= rate_ceiling + 1e-9
    assert limits.max_acceleration_deg_s2 <= acceleration_ceiling + 1e-9
    assert abs(plan.rate_deg_s) <= rate_ceiling + 1e-9
    assert abs(plan.acceleration_deg_s2) <= acceleration_ceiling + 1e-9
    if index == 199:
      assert abs(plan.rate_deg_s) < 300.0
  assert not transition
  assert abs(plan.rate_deg_s) <= profile.max_rate_deg_s + 1e-6


def test_live_candidate_is_process_selected_and_fails_closed() -> None:
  car_interface = interfaces[HYUNDAI.HYUNDAI_PALISADE]
  car_params = car_interface.get_non_essential_params(HYUNDAI.HYUNDAI_PALISADE)
  controller = LatControlTorque(car_params.as_reader(), car_interface(car_params), DT_CTRL, use_rack_trajectory=True)
  vehicle_model = VehicleModel(car_params)
  state = car.CarState.new_message()
  state.vEgo = 3.0
  params = log.VehicleParameters.new_message()

  torque, _, controller_log = controller.update(True, state, vehicle_model, params, False, -.02, False, .2)
  assert torque == 0.0
  assert controller_log.version == 6

  model = messaging.new_message("modelV2").modelV2
  model.timestampEof = 1_000_000_000
  model.action.desiredCurvature = -.15
  model.orientationRate.t = [0.0, .5, 1.0, 1.5, 2.0, 2.5]
  model.orientationRate.z = [0.0, -.03, -.06, -.09, -.12, -.15]
  model.velocity.x = [3.0] * 6

  controller.set_rack_trajectory_model(model, 1_050_000_000)
  torque, _, _ = controller.update(True, state, vehicle_model, params, False, -.02, False, .2)
  assert torque == 0.0
  assert controller.rack_trajectory is not None
  assert controller.rack_trajectory.status == STATUS_INVALID_ACTION_TIME

  model.action.desiredCurvatureTime = .5
  controller.set_rack_trajectory_model(model, 1_050_000_000)
  torque, _, controller_log = controller.update(True, state, vehicle_model, params, False, -.02, False, .2)
  assert torque > 0.0
  assert controller_log.version == 6
  assert controller_log.i == 0.0
  assert controller.rack_trajectory_output is not None
  expected_target = model_path_target(
    native_times_s=model.orientationRate.t,
    orientation_rates_z=model.orientationRate.z,
    velocities_x=model.velocity.x,
    scalar_curvature=-.02,
    scalar_action_plan_s=.5,
    plan_time_now_s=.05,
    measured_v_ego=3.0,
    query_time_s=.05 + PREVIEW_S,
    vehicle_model=vehicle_model,
    roll_rad=0.0,
    angle_offset_deg=0.0,
  )
  assert abs(controller.rack_trajectory_output.target_angle_deg - expected_target.angle_deg) < 1e-5
  assert math.isfinite(controller.rack_trajectory_output.planned_angle_deg)
  assert math.isfinite(controller.rack_trajectory_output.planned_rate_deg_s)
  assert abs(controller.rack_trajectory_output.feedback_torque) <= .35
  position_error = controller.rack_trajectory_output.planned_angle_deg - state.steeringAngleDeg
  rate_error = controller.rack_trajectory_output.planned_rate_deg_s - controller.rack_trajectory_output.measured_rate_deg_s
  assert controller.rack_trajectory_output.position_feedback_torque * position_error >= 0.0
  assert controller.rack_trajectory_output.rate_feedback_torque * rate_error >= 0.0
  assert controller.rack_trajectory_output.infeasible == (
    controller.rack_trajectory_output.motion_limited
    or controller.rack_trajectory_output.feedback_limited
    or controller.rack_trajectory_output.torque_limited
    or controller.rack_trajectory_output.profile_transition
    or controller.rack_trajectory_output.path_limited
  )
  max_turn_in_feedback = abs(controller.rack_trajectory_output.feedback_torque)
  for _ in range(200):
    controller.set_rack_trajectory_model(model, 1_050_000_000)
    _, _, _ = controller.update(True, state, vehicle_model, params, False, -.02, False, .2)
    assert controller.rack_trajectory_output is not None
    max_turn_in_feedback = max(max_turn_in_feedback, abs(controller.rack_trajectory_output.feedback_torque))
  assert .5 < max_turn_in_feedback <= .7

  transition = LatControlTorque(car_params.as_reader(), car_interface(car_params), DT_CTRL, use_rack_trajectory=True)
  assert transition.rack_trajectory is not None
  transition.rack_trajectory.planner = JerkLimitedRackPlanner(0.0, 100.0)
  model.action.desiredCurvature = 0.0
  model.orientationRate.z = [0.0] * 6
  model.velocity.x = [20.0] * 6
  state.vEgo = 20.0
  transition.set_rack_trajectory_model(model, 1_050_000_000)
  _, _, _ = transition.update(True, state, vehicle_model, params, False, 0.0, False, .2)
  assert transition.rack_trajectory_output is not None
  assert transition.rack_trajectory_output.profile_transition
  assert transition.rack_trajectory_output.planned_rate_deg_s < 100.0
  max_transition_excursion = abs(transition.rack_trajectory_output.planned_angle_deg)
  for _ in range(1000):
    transition.set_rack_trajectory_model(model, 1_050_000_000)
    _, _, _ = transition.update(True, state, vehicle_model, params, False, 0.0, False, .2)
    if transition.rack_trajectory_output is not None:
      max_transition_excursion = max(max_transition_excursion, abs(transition.rack_trajectory_output.planned_angle_deg))
  assert transition.rack_trajectory_output is not None
  assert not transition.rack_trajectory_output.profile_transition
  assert abs(transition.rack_trajectory_output.planned_rate_deg_s) <= transition.rack_trajectory._limits(20.0).max_rate_deg_s + 1e-9
  assert max_transition_excursion < 60.0

  unwind = LatControlTorque(car_params.as_reader(), car_interface(car_params), DT_CTRL, use_rack_trajectory=True)
  assert unwind.rack_trajectory is not None
  unwind.rack_trajectory.planner = JerkLimitedRackPlanner(200.0)
  state.vEgo = 3.0
  state.steeringAngleDeg = 200.0
  model.velocity.x = [3.0] * 6
  unwind.set_rack_trajectory_model(model, 1_050_000_000)
  _, _, _ = unwind.update(True, state, vehicle_model, params, False, 0.0, False, .2)
  assert unwind.rack_trajectory_output is not None
  assert abs(unwind.rack_trajectory_output.target_angle_deg) < 1e-6
  assert abs(unwind.rack_trajectory_output.feedforward_torque) < .05

  crossing = LatControlTorque(car_params.as_reader(), car_interface(car_params), DT_CTRL, use_rack_trajectory=True)
  assert crossing.rack_trajectory is not None
  crossing.rack_trajectory.planner = JerkLimitedRackPlanner(20.0)
  state.steeringAngleDeg = 20.0
  crossing.set_rack_trajectory_model(model, 1_050_000_000)
  _, _, _ = crossing.update(True, state, vehicle_model, params, False, -.02, False, .2)
  crossing.set_rack_trajectory_model(model, 1_050_000_000)
  torque, _, _ = crossing.update(True, state, vehicle_model, params, False, .02, False, .2)
  assert crossing.rack_trajectory_output is not None
  assert crossing.rack_trajectory_output.planned_angle_deg * crossing.rack_trajectory_output.target_angle_deg < 0.0
  assert torque * crossing.rack_trajectory_output.target_angle_deg >= 0.0

  bounded = LatControlTorque(car_params.as_reader(), car_interface(car_params), DT_CTRL, use_rack_trajectory=True)
  bounded.set_rack_trajectory_model(model, 1_050_000_000)
  _, _, _ = bounded.update(True, state, vehicle_model, params, False, -1.0, False, .2)
  assert bounded.rack_trajectory_output is not None
  assert bounded.rack_trajectory_output.path_limited
  assert abs(bounded.rack_trajectory_output.target_curvature) <= .2
  assert abs(bounded.rack_trajectory_output.target_curvature) * state.vEgo ** 2 <= 3.0 + 1e-6
  bounded_angle_curvature = -vehicle_model.calc_curvature(
    math.radians(bounded.rack_trajectory_output.target_angle_deg - params.angleOffsetDeg), state.vEgo, params.roll,
  )
  assert abs(bounded_angle_curvature) * state.vEgo ** 2 <= 3.0 + 1e-6

  released = LatControlTorque(car_params.as_reader(), car_interface(car_params), DT_CTRL, use_rack_trajectory=True)
  state.vEgo = 27.0
  model.velocity.x = [27.0] * 6
  released.set_rack_trajectory_model(model, 1_050_000_000)
  state.steeringAngleDeg = math.degrees(vehicle_model.get_steer_from_curvature(-3.7 / state.vEgo ** 2, state.vEgo, 0.0))
  torque, _, _ = released.update(True, state, vehicle_model, params, False, -1.0, False, .2)
  assert released.rack_trajectory is not None
  assert released.rack_trajectory_output is not None
  assert released.rack_trajectory.status == STATUS_ACTIVE
  assert torque != 0.0
  assert released.rack_trajectory_output.feedforward_torque != 0.0
  assert released.rack_trajectory_output.position_feedback_torque * (
    released.rack_trajectory_output.planned_angle_deg - state.steeringAngleDeg
  ) > 0.0

  state.steeringAngleDeg = math.degrees(vehicle_model.get_steer_from_curvature(-2.9 / state.vEgo ** 2, state.vEgo, 0.0))
  released.set_rack_trajectory_model(model, 1_050_000_000)
  released.update(True, state, vehicle_model, params, False, -1.0, False, .2)
  assert released.rack_trajectory_output is not None
  assert released.rack_trajectory.status == STATUS_ACTIVE

  mirrored = LatControlTorque(car_params.as_reader(), car_interface(car_params), DT_CTRL, use_rack_trajectory=True)
  mirrored.set_rack_trajectory_model(model, 1_050_000_000)
  state.steeringAngleDeg = math.degrees(vehicle_model.get_steer_from_curvature(3.7 / state.vEgo ** 2, state.vEgo, 0.0))
  torque, _, _ = mirrored.update(True, state, vehicle_model, params, False, 1.0, False, .2)
  assert mirrored.rack_trajectory_output is not None
  assert mirrored.rack_trajectory is not None
  assert mirrored.rack_trajectory.status == STATUS_ACTIVE
  assert torque != 0.0
  assert mirrored.rack_trajectory_output.feedforward_torque != 0.0
  assert mirrored.rack_trajectory_output.position_feedback_torque * (
    mirrored.rack_trajectory_output.planned_angle_deg - state.steeringAngleDeg
  ) > 0.0

  assist = LatControlTorque(car_params.as_reader(), car_interface(car_params), DT_CTRL, use_rack_trajectory=True)
  state.vEgo = 3.0
  state.steeringAngleDeg = 0.0
  state.steeringRateDeg = 0.0
  model.action.desiredCurvature = -.15
  model.orientationRate.z = [0.0, -.03, -.06, -.09, -.12, -.15]
  model.velocity.x = [3.0] * 6
  assist.set_rack_trajectory_model(model, 1_050_000_000)
  state.steeringPressed = True
  state.steeringTorque = 300.0
  torque, _, _ = assist.update(True, state, vehicle_model, params, False, -.02, False, .2)
  assert 0.0 < torque <= MAX_DRIVER_ASSIST_TORQUE

  assist.set_rack_trajectory_model(model, 1_050_000_000)
  state.steeringTorque = -300.0
  torque, _, _ = assist.update(True, state, vehicle_model, params, False, -.02, False, .2)
  assert assist.rack_trajectory is not None and assist.rack_trajectory_output is not None
  normal_torque = max(-1.0, min(1.0, assist.rack_trajectory_output.feedforward_torque + assist.rack_trajectory_output.feedback_torque))
  assert torque == max(-MAX_DRIVER_ASSIST_TORQUE, min(MAX_DRIVER_ASSIST_TORQUE, normal_torque))
  assert assist.rack_trajectory.status == STATUS_ACTIVE

  invalid_driver = LatControlTorque(car_params.as_reader(), car_interface(car_params), DT_CTRL, use_rack_trajectory=True)
  invalid_driver.set_rack_trajectory_model(model, 1_050_000_000)
  state.steeringTorque = math.nan
  torque, _, _ = invalid_driver.update(True, state, vehicle_model, params, False, -.02, False, .2)
  assert torque == 0.0
  assert invalid_driver.rack_trajectory is not None
  assert invalid_driver.rack_trajectory.status == STATUS_INVALID_VEHICLE_STATE

  state.steeringPressed = False
  state.steeringTorque = 0.0
  invalid_model = messaging.new_message("modelV2").modelV2
  invalid_model.timestampEof = 1_000_000_000
  invalid_model.action.desiredCurvature = -.02
  invalid_model.action.desiredCurvatureTime = .5
  controller.set_rack_trajectory_model(invalid_model, 1_050_000_000)
  torque, _, controller_log = controller.update(True, state, vehicle_model, params, False, -.02, False, .2)
  assert torque == 0.0
  assert not controller_log.active
  assert controller.rack_trajectory is not None
  assert controller.rack_trajectory.status == STATUS_INVALID_PATH

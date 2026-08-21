from __future__ import annotations

import importlib
import math

from openpilot.cereal import log, messaging
from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.values import CAR as HYUNDAI
from opendbc.car.structs import car
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque, palisade_rack_trajectory_compatible
from openpilot.selfdrive.controls.lib.rack_trajectory import (
  JerkLimitedRackPlanner,
  MEASURED_RATE_FILTER_RC_S,
  MotionLimits,
  RackRateEstimator,
  RackReferenceGovernor,
  RackTarget,
  model_path_target,
  PalisadeRackTrajectoryController,
  STATUS_ACTIVE,
  STATUS_INVALID_ACTION_TIME,
  STATUS_INVALID_PATH,
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


def test_unwind_feedforward_ignores_boundary_chatter_but_suppresses_large_motion() -> None:
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
  model.orientationRate.t = [0.0, .5, 1.0, 1.5, 2.0]
  model.orientationRate.z = [0.0] * 5
  model.velocity.x = [5.0] * 5

  def update():
    controller.set_model(model, 1_050_000_000)
    return controller.update(
      True, state, vehicle_model, params, car_params.lateralTuning.torque,
      interface.torque_from_lateral_accel(), .2, -.02,
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
  state.steeringTorque = 300.0
  state.steeringAngleDeg += 2.0
  assert update() is None
  assert controller.planner is planner_before_override
  plan_offset_during_override = planner_before_override.position_deg - state.steeringAngleDeg
  assert abs(plan_offset_during_override - plan_offset_before_override) < .25
  for _ in range(199):
    state.steeringAngleDeg += .02
    assert update() is None
    assert controller.planner is planner_before_override
  state.steeringPressed = False
  state.steeringTorque = 0.0

  release_output = update()
  assert release_output is not None
  assert release_output.feedforward_torque == 0.0
  assert abs(release_output.torque) <= .35
  assert release_output.torque * (release_output.planned_angle_deg - state.steeringAngleDeg) >= 0.0

  state.steeringPressed = True
  state.steeringTorque = 300.0
  assert update() is None
  state.steeringPressed = False
  state.steeringTorque = 0.0

  invalid_model = messaging.new_message("modelV2").modelV2
  invalid_model.timestampEof = 1_000_000_000
  invalid_model.action.desiredCurvature = -.02
  invalid_model.action.desiredCurvatureTime = .5
  controller.set_model(invalid_model, 1_050_000_000)
  assert controller.update(
    True, state, vehicle_model, params, car_params.lateralTuning.torque,
    interface.torque_from_lateral_accel(), .2, -.02,
  ) is None
  assert controller.driver_override_resume

  output = update()
  assert output is not None
  assert abs(output.feedforward_torque) < .05


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
    assert governor.reversal_s == 0.0


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
  assert abs(persistent.reversal_s - .15) < 1e-9
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
    model.orientationRate.t = [0.0, .5, 1.0, 1.5, 2.0]
    model.orientationRate.z = [0.0] * 5
    model.velocity.x = [speed] * 5

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
  model.orientationRate.t = [0.0, .5, 1.0, 1.5, 2.0]
  model.orientationRate.z = [0.0] * 5
  model.velocity.x = [state.vEgo] * 5

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
  model.orientationRate.t = [0.0, .5, 1.0, 1.5, 2.0]
  model.orientationRate.z = [0.0, -.03, -.06, -.09, -.12]
  model.velocity.x = [3.0] * 5

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
    query_time_s=.55,
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
  model.orientationRate.z = [0.0] * 5
  model.velocity.x = [20.0] * 5
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
  model.velocity.x = [3.0] * 5
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
  model.velocity.x = [27.0] * 5
  released.set_rack_trajectory_model(model, 1_050_000_000)
  state.steeringAngleDeg = math.degrees(vehicle_model.get_steer_from_curvature(-3.7 / state.vEgo ** 2, state.vEgo, 0.0))
  torque, _, _ = released.update(True, state, vehicle_model, params, False, -1.0, False, .2)
  assert released.rack_trajectory is not None
  assert released.rack_trajectory_output is not None
  assert released.rack_trajectory.status == STATUS_ACTIVE
  assert torque * state.steeringAngleDeg <= 0.0

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
  assert torque * state.steeringAngleDeg <= 0.0

  assist = LatControlTorque(car_params.as_reader(), car_interface(car_params), DT_CTRL, use_rack_trajectory=True)
  state.vEgo = 3.0
  state.steeringAngleDeg = 0.0
  state.steeringRateDeg = 0.0
  model.action.desiredCurvature = -.15
  model.orientationRate.z = [0.0, -.03, -.06, -.09, -.12]
  model.velocity.x = [3.0] * 5
  assist.set_rack_trajectory_model(model, 1_050_000_000)
  state.steeringPressed = True
  state.steeringTorque = 300.0
  torque, _, _ = assist.update(True, state, vehicle_model, params, False, -.02, False, .2)
  assert 0.0 < torque <= .35

  assist.set_rack_trajectory_model(model, 1_050_000_000)
  state.steeringTorque = -300.0
  torque, _, _ = assist.update(True, state, vehicle_model, params, False, -.02, False, .2)
  assert torque == 0.0
  assert assist.rack_trajectory is not None
  assert assist.rack_trajectory.status == 2

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

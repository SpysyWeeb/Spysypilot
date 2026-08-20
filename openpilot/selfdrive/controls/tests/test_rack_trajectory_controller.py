from __future__ import annotations

import math

from openpilot.cereal import log, messaging
from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.values import CAR as HYUNDAI
from opendbc.car.structs import car
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.controls.lib.rack_trajectory import (
  JerkLimitedRackPlanner,
  MotionLimits,
  RackTarget,
  model_path_target,
  PalisadeRackTrajectoryController,
  STATUS_INVALID_PATH,
  STATUS_MEASURED_OUT_OF_BOUNDS,
)


class LinearVehicleModel:
  @staticmethod
  def get_steer_from_curvature(curvature: float, speed: float, roll: float) -> float:
    del speed, roll
    return curvature * 10.0


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
  assert controller._measured_rate(-.9, 5.0) == (5.0, True)
  controller.reset()
  assert controller._measured_rate(1.0, 5.0) == (0.0, False)
  assert controller._measured_rate(1.1, 5.0) == (5.0, True)
  assert controller._measured_rate(1.0, 5.0) == (-5.0, True)


def test_profile_transition_headroom_does_not_walk_outward() -> None:
  controller = PalisadeRackTrajectoryController()
  controller.planner = JerkLimitedRackPlanner(0.0, 300.0)
  controller.planner.acceleration_deg_s2 = 500.0
  profile = MotionLimits(40.0, 100.0, 5000.0)
  limits, transition = controller._motion_limits(profile)
  assert transition
  rate_ceiling = limits.max_rate_deg_s
  acceleration_ceiling = limits.max_acceleration_deg_s2
  for _ in range(200):
    plan = controller.planner.update(RackTarget(10_000.0, 10_000.0), limits, .01)
    limits, transition = controller._motion_limits(profile)
    assert limits.max_rate_deg_s <= rate_ceiling + 1e-9
    assert limits.max_acceleration_deg_s2 <= acceleration_ceiling + 1e-9
    assert abs(plan.rate_deg_s) <= rate_ceiling + 1e-9
    assert abs(plan.acceleration_deg_s2) <= acceleration_ceiling + 1e-9


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
  if hasattr(model.action, "desiredCurvatureTime"):
    model.action.desiredCurvatureTime = .5
  model.orientationRate.t = [0.0, .5, 1.0, 1.5, 2.0]
  model.orientationRate.z = [0.0, -.03, -.06, -.09, -.12]
  model.velocity.x = [3.0] * 5
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
    scalar_action_plan_s=.5 if hasattr(model.action, "desiredCurvatureTime") else .2 + 1.5 * DT_MDL,
    plan_time_now_s=.05,
    measured_v_ego=3.0,
    query_time_s=.45,
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
  released.set_rack_trajectory_model(model, 1_050_000_000)
  state.steeringAngleDeg = math.degrees(vehicle_model.get_steer_from_curvature(-5.0 / state.vEgo ** 2, state.vEgo, 0.0))
  torque, _, _ = released.update(True, state, vehicle_model, params, False, 0.0, False, .2)
  assert torque == 0.0
  assert released.rack_trajectory is not None
  assert released.rack_trajectory.status == STATUS_MEASURED_OUT_OF_BOUNDS

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

  state.steeringPressed = False
  invalid_model = messaging.new_message("modelV2").modelV2
  invalid_model.timestampEof = 1_000_000_000
  invalid_model.action.desiredCurvature = -.02
  controller.set_rack_trajectory_model(invalid_model, 1_050_000_000)
  torque, _, controller_log = controller.update(True, state, vehicle_model, params, False, -.02, False, .2)
  assert torque == 0.0
  assert not controller_log.active
  assert controller.rack_trajectory is not None
  assert controller.rack_trajectory.status == STATUS_INVALID_PATH

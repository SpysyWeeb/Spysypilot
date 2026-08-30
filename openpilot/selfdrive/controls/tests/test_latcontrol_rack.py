import math

import numpy as np

from openpilot.common.test import OpenpilotTestCase

from openpilot.cereal import log, messaging
from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.values import CAR as HYUNDAI, CarControllerParams
from opendbc.car.lateral import apply_driver_steer_torque_limits
from opendbc.car.structs import car
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import MAX_LATERAL_ACCEL_NO_ROLL, MIN_SPEED
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.controls.lib.latcontrol_rack import FALLBACK_HOLD_S, LatControlRack
from openpilot.selfdrive.controls.lib.rack_trajectory import (
  _smoothstep,
  HORIZON_ACCELERATION_BLEND,
  HORIZON_OFFSETS_S,
  HORIZON_POSITION_TOLERANCE_DEG,
  horizon_candidate_preserves_immediate_path,
  horizon_desired_acceleration,
  JerkLimitedRackPlanner,
  MAX_DRIVER_ASSIST_TORQUE,
  MEASURED_RATE_FILTER_RC_S,
  MotionLimits,
  PathTarget,
  RackPlan,
  RackRateEstimator,
  RackTarget,
  RackTrajectoryController,
  model_path_targets,
  PREVIEW_ADMIT_DEVIATION_M,
  PREVIEW_LENGTHEN_UPDATES,
  PreviewScheduler,
  REFERENCE_FILTER_TRAIL_MAX_DEG,
  ReferenceFilter,
  reference_trail_limit_deg,
  RESPONSE_TIME_PREVIEW_S,
  DIRECTION_GUARD_RC_S,
  TURN_IN_BLEND_DEG,
  RESPONSE_TIME_S,
  STATUS_ACTIVE,
  STATUS_INACTIVE,
  STATUS_INVALID_PREVIEW,
  STATUS_INVALID_PATH,
  STATUS_INVALID_VEHICLE_STATE,
  STATUS_NO_MODEL,
  STATUS_STALE_MODEL,
  INACTIVE_HOLD_FRAMES,
  STALE_MODEL_S,
)


class LinearVehicleModel:
  @staticmethod
  def get_steer_from_curvature(curvature, speed, roll):
    del speed, roll
    return curvature * 10.0


def get_rack_controller(car_name=HYUNDAI.HYUNDAI_PALISADE):
  CarInterface = interfaces[car_name]
  CP = CarInterface.get_non_essential_params(car_name)
  CI = CarInterface(CP)
  VM = VehicleModel(CP)
  controller = LatControlRack(CP.as_reader(), CI, DT_CTRL)
  stock = LatControlTorque(CP.as_reader(), CI, DT_CTRL)
  return controller, stock, VM


def set_curvature_preview(model, action_t=0.5):
  """Fill the action's curvature preview from the message's own yaw-rate and speed plan, the way
  modeld fills it from its plan head: sampled from the action time on, one horizon step apart."""
  times = list(model.orientationRate.t)
  rates = list(model.orientationRate.z)
  speeds = list(model.velocity.x)
  preview_times = [action_t + offset for offset in HORIZON_OFFSETS_S]
  model.action.desiredCurvatureTime = action_t
  model.action.desiredCurvaturePreviewTimes = preview_times
  model.action.desiredCurvaturePreview = [
    float(np.interp(t, times, rates)) / max(float(np.interp(t, times, speeds)), MIN_SPEED) for t in preview_times
  ]


def horizon_model(times, rates, speeds, path_y=None):
  message = messaging.new_message("modelV2").modelV2
  message.timestampEof = 1_000_000_000
  message.action.desiredCurvature = 0.0
  message.orientationRate.t = times
  message.orientationRate.z = rates
  message.velocity.t = times
  message.velocity.x = speeds
  # the plan's path: driven at the plan's speed, straight unless the test says otherwise
  message.position.t = times
  message.position.x = [float(x) for x in np.concatenate(([0.0], np.cumsum(np.diff(times) * 0.5 * (np.array(speeds[1:]) + np.array(speeds[:-1])))))]
  message.position.y = list(path_y) if path_y is not None else [0.0] * len(times)
  message.position.yStd = [0.0] * len(times)
  message.confidence = "green"  # the schema's default is red, which admits no preview
  set_curvature_preview(message)
  return message


def build_feedforward_boundary_controller():
  car_interface = interfaces[HYUNDAI.HYUNDAI_PALISADE]
  car_params = car_interface.get_non_essential_params(HYUNDAI.HYUNDAI_PALISADE)
  interface = car_interface(car_params)
  controller = RackTrajectoryController()
  vehicle_model = VehicleModel(car_params)
  state = car.CarState.new_message()
  state.vEgo = 5.0
  params = log.VehicleParameters.new_message()
  model = horizon_model([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], [0.0] * 6, [5.0] * 6)
  model.action.desiredCurvature = -0.02
  torque_from_lateral_accel = interface.torque_from_lateral_accel()

  def update(torque_scale=1.0):
    controller.set_model(model, 1_050_000_000)
    return controller.update(
      True, state, vehicle_model, params, car_params.lateralTuning.torque,
      lambda lateral_accel, torque_params: torque_scale * torque_from_lateral_accel(lateral_accel, torque_params), 0.2, -0.02,
    )

  return controller, state, update


class TestLatControlRack(OpenpilotTestCase):

  def setUp(self):
    CarInterface = interfaces[HYUNDAI.HYUNDAI_PALISADE]
    self.CP = CarInterface.get_non_essential_params(HYUNDAI.HYUNDAI_PALISADE)
    self.CI = CarInterface(self.CP)
    self.VM = VehicleModel(self.CP)
    self.CS = car.CarState.new_message()
    self.CS.vEgo = 15.0
    self.params = log.VehicleParameters.new_message()

  # ---- model_path_targets / JerkLimitedRackPlanner / ReferenceFilter / PreviewScheduler / RackRateEstimator ----

  def test_model_path_is_scalar_anchored_and_signed(self):
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    speeds = [5.0] * 5
    # modeld's preview from the action time (1 s) on, 0.01 of curvature per second; the pin puts the
    # scalar (0.03) at the action time. A query one second past now reads one second past the action time
    preview_times = [1.0, 2.0, 3.0]
    preview = [0.01, 0.02, 0.03]
    target = model_path_targets(
      native_times_s=times, velocities_x=speeds, preview_times_s=preview_times, preview_curvatures=preview,
      scalar_curvature=0.03, plan_time_now_s=0.1, measured_v_ego=5.0,
      query_times_s=(1.1,), vehicle_model=LinearVehicleModel(), roll_rad=0.0, angle_offset_deg=0.0,
    )[0]
    expected_curvature = 0.03 + 0.02 - 0.01
    assert abs(target.curvature - expected_curvature) < 1e-12
    assert abs(target.angle_deg - math.degrees(-expected_curvature * 10.0)) < 1e-12
    assert abs(target.rate_deg_s - math.degrees(-0.01 * 10.0)) < 1e-9

    # at the plan's age the target is the scalar as published, moving at the preview's slope; past the
    # preview's end the last sample holds, at rest
    now, late = model_path_targets(
      native_times_s=times, velocities_x=speeds, preview_times_s=preview_times, preview_curvatures=preview,
      scalar_curvature=0.03, plan_time_now_s=0.1, measured_v_ego=5.0,
      query_times_s=(0.1, 3.5), vehicle_model=LinearVehicleModel(), roll_rad=0.0, angle_offset_deg=0.0,
    )
    assert abs(now.curvature - 0.03) < 1e-12
    assert abs(now.rate_deg_s - target.rate_deg_s) < 1e-9
    assert abs(late.curvature - 0.05) < 1e-12
    assert abs(late.rate_deg_s) < 1e-9

    for bad in (
      {"query_times_s": (5.0,)},  # the speed plan does not cover the query
      {"preview_times_s": [0.0, 1.0, 2.0]},  # a preview starting at the plan origin has no action time
      {"preview_curvatures": [0.01, math.nan, 0.03]},
      {"preview_curvatures": [0.01, 0.02]},
      {"velocities_x": [5.0, math.nan, 5.0, 5.0]},
    ):
      arguments = {
        "native_times_s": times, "velocities_x": speeds, "preview_times_s": preview_times, "preview_curvatures": preview,
        "scalar_curvature": 0.03, "plan_time_now_s": 0.1, "measured_v_ego": 5.0,
        "query_times_s": (2.0,), "vehicle_model": LinearVehicleModel(), "roll_rad": 0.0, "angle_offset_deg": 0.0,
      }
      arguments.update(bad)
      with self.assertRaises(ValueError):
        model_path_targets(**arguments)

    stopped = model_path_targets(
      native_times_s=times, velocities_x=[5.0, 5.0, 0.0, 0.0, 0.0], preview_times_s=preview_times, preview_curvatures=preview,
      scalar_curvature=0.03, plan_time_now_s=0.1, measured_v_ego=5.0,
      query_times_s=(2.0,), vehicle_model=LinearVehicleModel(), roll_rad=0.0, angle_offset_deg=0.0,
    )[0]
    # a plan that stops inside the horizon still covers it: the curvature is the preview's, the speed floors
    assert abs(stopped.curvature - (0.03 + 0.029 - 0.01)) < 1e-12
    assert stopped.speed_mps == MIN_SPEED
    assert math.isfinite(stopped.angle_deg)

  def test_jerk_limited_planner_is_bounded_and_symmetric(self):
    limits = MotionLimits(120.0, 500.0, 2500.0)
    right = JerkLimitedRackPlanner(0.0)
    left = JerkLimitedRackPlanner(0.0)
    previous_acceleration = 0.0
    positive = None
    for _ in range(300):
      positive = right.update(RackTarget(90.0, 0.0), limits, 0.01)
      negative = left.update(RackTarget(-90.0, 0.0), limits, 0.01)
      assert abs(positive.position_deg + negative.position_deg) < 1e-9
      assert abs(positive.rate_deg_s) <= limits.max_rate_deg_s + 1e-9
      assert abs(positive.acceleration_deg_s2) <= limits.max_acceleration_deg_s2 + 1e-9
      assert abs((positive.acceleration_deg_s2 - previous_acceleration) / 0.01) <= limits.max_jerk_deg_s3 + 1e-7
      previous_acceleration = positive.acceleration_deg_s2
    assert positive is not None
    assert abs(positive.position_deg - 90.0) < 0.25
    assert abs(positive.rate_deg_s) < 1.0

  def test_reference_filter_passes_a_large_step_within_its_bound_and_settles(self):
    filter_ = ReferenceFilter()
    assert filter_.update(RackTarget(0.0, 0.0), 3.0, 0.01) == RackTarget(0.0, 0.0)
    served = filter_.update(RackTarget(30.0, 300.0), 3.0, 0.01)
    # the step passes at once, short of the target by the bound, at the target's own rate
    assert math.isclose(served.position_deg, 27.0) and served.rate_deg_s == 300.0 and filter_.limited
    for _ in range(30):
      served = filter_.update(RackTarget(30.0, 0.0), 3.0, 0.01)
      assert 27.0 - 1e-9 <= served.position_deg <= 30.0
    assert abs(30.0 - served.position_deg) < 0.2 and not filter_.limited

  def test_reference_filter_smooths_small_jitter(self):
    filter_ = ReferenceFilter()
    filter_.update(RackTarget(0.0, 0.0), 3.0, 0.01)
    served = []
    for frame in range(400):
      # the model's target flipping by 2 degrees every model frame, as it does at low speed
      raw = 2.0 if (frame // 5) % 2 == 0 else -2.0
      served.append(filter_.update(RackTarget(raw, 0.0), 3.0, 0.01).position_deg)
    assert max(abs(value) for value in served[100:]) < 1.0
    assert not filter_.limited

  def test_reference_filter_bypass_and_reset(self):
    filter_ = ReferenceFilter()
    filter_.update(RackTarget(0.0, 0.0), 3.0, 0.01)
    assert filter_.update(RackTarget(20.0, 0.0), 3.0, 0.01, bypass=True) == RackTarget(20.0, 0.0) and not filter_.limited
    filter_.reset()
    assert filter_.target is None
    assert filter_.update(RackTarget(-5.0, 1.0), 3.0, 0.01) == RackTarget(-5.0, 1.0)

  def test_reference_trail_limit_is_the_wheel_cap_at_low_speed_and_a_lateral_accel_at_speed(self):
    assert reference_trail_limit_deg(self.VM, 5.0) == REFERENCE_FILTER_TRAIL_MAX_DEG
    assert reference_trail_limit_deg(self.VM, 0.0) == REFERENCE_FILTER_TRAIL_MAX_DEG
    fast = reference_trail_limit_deg(self.VM, 30.0)
    assert 0.5 < fast < 2.0
    assert reference_trail_limit_deg(self.VM, 20.0) > fast

  def _scheduler_targets(self, curvatures, speed):
    return tuple(PathTarget(k, speed, math.degrees(self.VM.get_steer_from_curvature(-k, speed, 0.0)), 0.0) for k in curvatures)

  def test_preview_scheduler_lengthens_on_a_consistent_straight_and_shortens_fast(self):
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    straight = horizon_model(times, [0.0] * 7, [20.0] * 7)
    targets = self._scheduler_targets([0.0] * len(HORIZON_OFFSETS_S), 20.0)
    scheduler = PreviewScheduler()
    for frame in range(1, 40):
      index = scheduler.update(straight, frame, 0.5, targets, False)
      assert index == min(8, frame // PREVIEW_LENGTHEN_UPDATES)
    assert scheduler.preview_s == 2.0
    # the same model frame again decides nothing new
    assert scheduler.update(straight, 39, 0.5, targets, False) == 8
    # the path now jogs 1.5 m from 1.5 s on: everything past 1 s disagrees, and the preview shortens in two frames
    jog = horizon_model(times, [0.0] * 7, [20.0] * 7, path_y=[0.0, 0.0, 0.0, 1.5, 1.5, 1.5, 1.5])
    assert scheduler.update(jog, 40, 0.5, targets, False) == 8
    assert scheduler.update(jog, 41, 0.5, targets, False) == 3
    assert scheduler.preview_s == 0.75

  def test_preview_scheduler_grows_through_periodic_texture(self):
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    straight = horizon_model(times, [0.0] * 7, [20.0] * 7)
    bump = horizon_model(times, [0.0] * 7, [20.0] * 7, path_y=[0.0] * 2 + [0.3] * 5)
    targets = self._scheduler_targets([0.0] * len(HORIZON_OFFSETS_S), 20.0)
    scheduler = PreviewScheduler()
    # every third model frame disagrees (a joint, a rumble strip): the preview still grows and never shortens
    for frame in range(1, 40):
      index = scheduler.update(bump if frame % 3 == 0 else straight, frame, 0.5, targets, False)
      assert index >= min(8, (frame - frame // 3) // PREVIEW_LENGTHEN_UPDATES) - 1
    assert scheduler.index == 8

  def test_hands_and_lane_changes_pin_the_preview_through_the_controller(self):
    for what in ("hands", "lane change"):
      controller = RackTrajectoryController()
      CS = car.CarState.new_message()
      CS.vEgo = 15.0
      params = log.VehicleParameters.new_message()
      model = horizon_model([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], [0.0] * 7, [15.0] * 7)
      output = None
      for frame in range(100):
        model.timestampEof = 1_000_000_000 + (frame // 5) * 50_000_000
        controller.set_model(model, model.timestampEof + 30_000_000)
        output = controller.update(True, CS, self.VM, params, self.CP.lateralTuning.torque, self.CI.torque_from_lateral_accel(), .2, 0.0)
      assert output is not None and output.preview_time_s > 1.0
      if what == "hands":
        CS.steeringPressed = True
      else:
        model.meta.laneChangeState = "laneChangeStarting"
      model.timestampEof += 50_000_000
      controller.set_model(model, model.timestampEof + 30_000_000)
      output = controller.update(True, CS, self.VM, params, self.CP.lateralTuning.torque, self.CI.torque_from_lateral_accel(), .2, 0.0)
      assert output is not None and output.preview_time_s == 0.0

  def test_preview_never_replaces_the_near_target(self):
    controller = RackTrajectoryController()
    CS = car.CarState.new_message()
    CS.vEgo = 15.0
    params = log.VehicleParameters.new_message()
    # a consistent straight whose far targets bend gently within the heading gate: the near target stays in charge
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    model = horizon_model(times, [0.0, 0.0, 0.0005, 0.001, 0.0015, 0.002, 0.002], [15.0] * 7)
    for frame in range(150):
      model.timestampEof = 1_000_000_000 + (frame // 5) * 50_000_000
      controller.set_model(model, model.timestampEof + 30_000_000)
      output = controller.update(True, CS, self.VM, params, self.CP.lateralTuning.torque, self.CI.torque_from_lateral_accel(), .2, 0.0)
      assert output is not None
    assert output.preview_time_s >= 1.0
    assert abs(output.target_angle_deg) < 1e-6 and abs(output.target_curvature) < 1e-9

  def test_preview_slows_the_filter_within_its_bound(self):
    filter_ = ReferenceFilter()
    filter_.update(RackTarget(0.0, 0.0), 3.0, 0.01)
    quick = [filter_.update(RackTarget(1.0, 0.0), 3.0, 0.01).position_deg for _ in range(10)]
    filter_.reset()
    filter_.update(RackTarget(0.0, 0.0), 3.0, 0.01)
    calm = [filter_.update(RackTarget(1.0, 0.0), 3.0, 0.01, rc_s=0.3).position_deg for _ in range(10)]
    assert quick[-1] > calm[-1] > 0.0
    # a step past the bound is served at the bound regardless of the time constant
    assert math.isclose(filter_.update(RackTarget(20.0, 100.0), 3.0, 0.01, rc_s=0.3).position_deg, 17.0)

  def test_feedback_keeps_authority_at_standstill(self):
    controller = RackTrajectoryController()
    CS = car.CarState.new_message()
    CS.vEgo = 0.0
    CS.steeringAngleDeg = 10.0
    params = log.VehicleParameters.new_message()
    model = horizon_model([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], [0.0] * 6, [0.0] * 6)
    output = None
    for frame in range(300):
      model.timestampEof = 1_000_000_000 + (frame // 5) * 50_000_000
      controller.set_model(model, model.timestampEof + 30_000_000)
      output = controller.update(True, CS, self.VM, params, self.CP.lateralTuning.torque, self.CI.torque_from_lateral_accel(), .2, 0.0)
      assert output is not None
    # a 10 degree error while stopped produces corrective torque toward zero, not silence
    assert output.position_feedback_torque < -0.01

  def test_turn_in_feedback_cap_is_continuous(self):
    caps = []
    assert TURN_IN_BLEND_DEG > 0.0
    for offset in [x * 0.1 for x in range(-60, 61)]:
      # sweep the target past the measured angle and record the cap the controller applies
      controller = RackTrajectoryController()
      controller.planner = JerkLimitedRackPlanner(5.0)
      CS = car.CarState.new_message()
      CS.vEgo = 15.0
      CS.steeringAngleDeg = 5.0
      params = log.VehicleParameters.new_message()
      target_angle = 5.0 + offset
      curvature = -self.VM.calc_curvature(math.radians(target_angle), CS.vEgo, 0.0)
      model = horizon_model([0.0, .5, 1.0, 1.5, 2.0, 2.5], [curvature * CS.vEgo] * 6, [CS.vEgo] * 6)
      model.action.desiredCurvature = curvature
      controller.set_model(model, 1_050_000_000)
      output = controller.update(True, CS, self.VM, params, self.CP.lateralTuning.torque, self.CI.torque_from_lateral_accel(), .2, curvature)
      assert output is not None
      caps.append(output.feedback_torque)
    steps = [abs(b - a) for a, b in zip(caps, caps[1:], strict=False)]
    assert max(steps) < 0.06  # no 0.35-unit jump anywhere in the sweep (FM3.5)

  def test_direction_guard_ramps_down_not_snaps(self):
    # the plan still on the old side of a reversal while the target is across zero: feedback pulls
    # toward the plan, opposing the target, and the guard drains the torque at its time constant
    controller = RackTrajectoryController()
    controller.planner = JerkLimitedRackPlanner(-8.0)
    CS = car.CarState.new_message()
    CS.vEgo = 15.0
    CS.steeringAngleDeg = 0.0
    params = log.VehicleParameters.new_message()
    target_angle = 5.0
    curvature = -self.VM.calc_curvature(math.radians(target_angle), CS.vEgo, 0.0)
    model = horizon_model([0.0, .5, 1.0, 1.5, 2.0, 2.5], [curvature * CS.vEgo] * 6, [CS.vEgo] * 6)
    model.action.desiredCurvature = curvature
    controller.set_model(model, 1_050_000_000)
    scales = []
    for _ in range(4):
      output = controller.update(True, CS, self.VM, params, self.CP.lateralTuning.torque, self.CI.torque_from_lateral_accel(), .2, curvature)
      assert output is not None
      scales.append(controller.direction_guard_scale)
    steps = [a - b for a, b in zip(scales, scales[1:], strict=False)]
    assert all(0.0 < scales[i + 1] < scales[i] for i in range(len(scales) - 1))
    assert all(abs(step - controller.dt / DIRECTION_GUARD_RC_S) < 1e-9 for step in steps)
    assert output.direction_guarded and not output.saturated

  def test_driver_assist_cap_reports_itself_not_saturation(self):
    controller, state, update = build_feedforward_boundary_controller()
    state.steeringPressed = True
    state.steeringTorque = -300.0
    output = None
    for _ in range(20):
      # a demand between the driver-assist cap and the platform ceiling: only the cap engages
      output = update(torque_scale=2.0)
      assert output is not None
    assert abs(output.torque) == MAX_DRIVER_ASSIST_TORQUE
    assert output.driver_assist_limited and not output.saturated

  def test_saturated_means_the_platform_ceiling_only(self):
    controller, _, VM = get_rack_controller()
    CS = car.CarState.new_message()
    CS.vEgo = 3.0
    CS.steeringAngleDeg = 0.0
    params = log.VehicleParameters.new_message()
    curvature = 0.09
    model = horizon_model([0.0, .5, 1.0, 1.5, 2.0, 2.5], [curvature * 3.0] * 6, [3.0] * 6)
    model.action.desiredCurvature = curvature
    for frame in range(200):
      model.timestampEof = 1_000_000_000 + (frame // 5) * 50_000_000
      controller.rack.set_model(model, model.timestampEof + 30_000_000)
      controller.update(True, CS, VM, params, False, curvature, False, .2, model=model, mono_time_ns=model.timestampEof + 30_000_000)
    # a huge demand at low speed rides the +-1.0 clip: the output reports it as saturation
    assert controller.output is not None and abs(controller.output.torque) > 0.99
    assert controller.output.saturated and not controller.output.driver_assist_limited

  def test_full_preview_keeps_steering_without_a_farther_target(self):
    controller = RackTrajectoryController()
    CS = car.CarState.new_message()
    CS.vEgo = 15.0  # 40 m of preview covers the full 2 s only below 20 m/s
    params = log.VehicleParameters.new_message()
    model = horizon_model([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], [0.0] * 7, [15.0] * 7)
    for frame in range(150):
      model.timestampEof = 1_000_000_000 + (frame // 5) * 50_000_000
      controller.set_model(model, model.timestampEof + 30_000_000)
      output = controller.update(True, CS, self.VM, params, self.CP.lateralTuning.torque, self.CI.torque_from_lateral_accel(), .2, 0.0)
      assert output is not None and controller.status == STATUS_ACTIVE
    assert output.preview_time_s == 2.0

  def test_preview_scheduler_holds_inside_the_hysteresis_band(self):
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    straight = horizon_model(times, [0.0] * 7, [20.0] * 7)
    targets = self._scheduler_targets([0.0] * len(HORIZON_OFFSETS_S), 20.0)
    scheduler = PreviewScheduler()
    for frame in range(1, 20):
      scheduler.update(straight, frame, 0.5, targets, False)
    assert scheduler.index == 8
    # a deviation above the admission tolerance but below the keep tolerance keeps the preview where it is
    wobble = horizon_model(times, [0.0] * 7, [20.0] * 7, path_y=[0.0] * 3 + [PREVIEW_ADMIT_DEVIATION_M + 0.02] * 4)
    for frame in range(20, 30):
      assert scheduler.update(wobble, frame, 0.5, targets, False) == 8
    fresh = PreviewScheduler()
    for frame in range(1, 20):
      fresh.update(wobble, frame, 0.5, targets, False)
    assert fresh.index == 3

  def test_preview_scheduler_gates_heading_flicker_distance_hands_and_confidence(self):
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    model = horizon_model(times, [0.0] * 7, [20.0] * 7)
    # the far target turns: admitted only as far as the heading stays within a degree of the near target
    turning = self._scheduler_targets([0.0, 0.0, 0.0002, 0.0002, 0.001, 0.001, 0.001, 0.001, 0.001], 20.0)
    scheduler = PreviewScheduler()
    assert scheduler._admissible(model, 0.5, turning, tuple(t.angle_deg for t in turning)) == 3
    # a far target that swings between model frames more than the near one is not admitted
    steady = self._scheduler_targets([0.0] * 9, 20.0)
    flicker = list(steady)
    flicker[6] = PathTarget(0.0, 20.0, 1.5, 0.0)
    scheduler = PreviewScheduler()
    scheduler.update(model, 1, 0.5, steady, False)
    assert scheduler._admissible(model, 0.5, tuple(flicker), tuple(t.angle_deg for t in flicker)) == 5
    # 40 m of preview at 35 m/s is 1.14 s
    fast = horizon_model(times, [0.0] * 7, [35.0] * 7)
    assert PreviewScheduler()._admissible(fast, 0.5, steady, tuple(t.angle_deg for t in steady)) == 4
    # the driver's hands pin the preview at the action time at once
    scheduler = PreviewScheduler()
    for frame in range(1, 20):
      scheduler.update(model, frame, 0.5, steady, False)
    assert scheduler.update(model, 20, 0.5, steady, True) == 0 and scheduler.preview_s == 0.0
    # a red-confidence model admits nothing, and neither does a path with a hole in it
    model.confidence = "red"
    assert PreviewScheduler()._admissible(model, 0.5, steady, tuple(t.angle_deg for t in steady)) == 0
    holed = horizon_model(times, [0.0] * 7, [20.0] * 7)
    holed.position.y = [0.0, 0.0, math.nan, 0.0, 0.0, 0.0, 0.0]
    assert PreviewScheduler()._admissible(holed, 0.5, steady, tuple(t.angle_deg for t in steady)) == 0
    holed = horizon_model(times, [0.0] * 7, [20.0] * 7)
    holed.position.yStd = [0.0, 0.0, 0.0, math.nan, 0.0, 0.0, 0.0]
    assert PreviewScheduler()._admissible(holed, 0.5, steady, tuple(t.angle_deg for t in steady)) == 0

  def test_motion_limits_carry_the_scheduled_response_time_through_a_transition(self):
    controller = RackTrajectoryController()
    profile = controller._limits(20.0, RESPONSE_TIME_S + RESPONSE_TIME_PREVIEW_S)
    assert profile.response_time_s == 0.5
    controller.planner = JerkLimitedRackPlanner(0.0, 2.0 * profile.max_rate_deg_s)
    limits, transition = controller._motion_limits(profile)
    assert transition and limits.response_time_s == 0.5

  def test_low_speed_turn_in_and_unwind_pass_the_filter_within_its_bound(self):
    controller = RackTrajectoryController()
    CS = car.CarState.new_message()
    CS.vEgo = 4.0
    params = log.VehicleParameters.new_message()
    model = horizon_model([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], [0.0] * 6, [4.0] * 6)
    # a 250 degree turn-in over 3 s and its unwind, the plan's curvature still building along the horizon: the served
    # target follows within the bound every frame and the far preview is never admitted
    worst = 0.0
    for frame in range(600):
      curvature = 0.08 * min(1.0, frame / 300.0) if frame < 300 else 0.08 * max(0.0, 1.0 - (frame - 300) / 300.0)
      slope = 0.08 / 3.0 if frame < 300 else -0.08 / 3.0
      model.orientationRate.z = [max(0.0, curvature + slope * t) * 4.0 for t in model.orientationRate.t]
      set_curvature_preview(model)
      model.action.desiredCurvature = curvature
      model.timestampEof = 1_000_000_000 + (frame // 5) * 50_000_000
      controller.set_model(model, model.timestampEof + 30_000_000)
      output = controller.update(True, CS, self.VM, params, self.CP.lateralTuning.torque, self.CI.torque_from_lateral_accel(), .2, curvature)
      assert output is not None
      # while the plan is still curving no far preview is admitted; once it has straightened it may be
      assert curvature < 0.02 or output.preview_time_s == 0.0
      raw_angle = math.degrees(self.VM.get_steer_from_curvature(-curvature, 4.0, 0.0))
      worst = max(worst, abs(output.target_angle_deg - raw_angle))
    assert worst <= REFERENCE_FILTER_TRAIL_MAX_DEG + 1e-6
    assert abs(worst - REFERENCE_FILTER_TRAIL_MAX_DEG) < 0.5

  def test_signed_rack_rate_handles_signed_and_unsigned_samples(self):
    estimator = RackRateEstimator(DT_CTRL)
    assert estimator.update(-1.0, -5.0) == (-5.0, True)
    positive_rate, valid = estimator.update(-0.9, 5.0)
    alpha = DT_CTRL / (MEASURED_RATE_FILTER_RC_S + DT_CTRL)
    assert valid and abs(positive_rate - (-5.0 + alpha * 10.0)) < 1e-12

    estimator.reset()
    assert estimator.update(1.0, 5.0) == (0.0, False)
    assert estimator.update(1.0, 0.0) == (0.0, True)
    step_rate = 0.0
    for index in range(5):
      step_rate, valid = estimator.update(1.1 + index * 0.1, 8.0)
    assert valid and abs(step_rate - 8.0 * (1.0 - (1.0 - alpha) ** 5)) < 1e-12
    reversal_rate, valid = estimator.update(1.0, 8.0)
    assert valid and reversal_rate > 0.0
    for index in range(100):
      reversal_rate, valid = estimator.update(0.9 - index * 0.1, 8.0)
    assert valid and abs(reversal_rate + 8.0) < 0.1

    estimator.reset()
    assert estimator.update(1.0, 5.0) == (0.0, False)
    filtered = [estimator.update(1.1 if index % 2 else 1.0, 8.0)[0] for index in range(1, 21)]
    assert sum(abs(filtered[index] - filtered[index - 1]) for index in range(1, len(filtered))) < 8.0 * 19 * 0.4

  # ---- RackTrajectoryController: full pipeline behaviors ----

  def _update_horizon(self, controller, path, mono_ns=1_050_000_000):
    controller.set_model(path, mono_ns)
    return controller.update(
      True, self.CS, self.VM, self.params, self.CP.lateralTuning.torque,
      self.CI.torque_from_lateral_accel(), 0.2, 0.0,
    )

  def test_full_horizon_faults_reset_and_recover_cold(self):
    self.CS.vEgo = 5.0
    valid = horizon_model([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], [0.0] * 6, [5.0] * 6)
    short_preview = horizon_model([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], [0.0] * 6, [5.0] * 6)
    short_preview.action.desiredCurvaturePreview = list(short_preview.action.desiredCurvaturePreview)[:-1]
    invalid_paths = (
      horizon_model([0.0, 0.5, 1.0], [0.0] * 3, [5.0] * 3),
      horizon_model([0.0, 0.5, 0.4, 1.5, 2.0, 2.5], [0.0] * 6, [5.0] * 6),
      horizon_model([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], [0.0, 0.0, math.nan, 0.0, 0.0, 0.0], [5.0] * 6),
    )
    cases = tuple((path, 1_050_000_000, STATUS_INVALID_PATH) for path in invalid_paths) + (
      (short_preview, 1_050_000_000, STATUS_INVALID_PREVIEW),
      (valid, 1_600_000_001, STATUS_STALE_MODEL),
    )
    for invalid, mono_ns, expected_status in cases:
      controller = RackTrajectoryController()
      assert self._update_horizon(controller, valid) is not None
      assert controller.planner is not None and controller.reference_filter.target is not None
      assert self._update_horizon(controller, invalid, mono_ns) is None
      assert controller.status == expected_status
      assert controller.planner is None
      assert controller.reference_filter.target is None
      recovered = self._update_horizon(controller, valid)
      assert recovered is not None
      assert controller.status == STATUS_ACTIVE
      assert all(math.isfinite(value) for value in (
        recovered.torque, recovered.planned_angle_deg, recovered.planned_rate_deg_s,
      ))

  def test_stopping_plan_within_horizon_is_valid(self):
    self.CS.vEgo = 5.0
    # a plan that stops inside the horizon is a complete path, not a fault
    stopping = horizon_model([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], [0.0, 0.02, 0.02, 0.0, 0.0, 0.0], [5.0, 4.0, 2.0, 0.0, 0.0, 0.0])
    controller = RackTrajectoryController()
    assert self._update_horizon(controller, stopping) is not None
    assert controller.status == STATUS_ACTIVE

  def test_feedforward_boundary_is_continuous_across_the_deadzone(self):
    _, state, update = build_feedforward_boundary_controller()
    output = update()
    assert output is not None
    target_angle = output.target_angle_deg
    feedforward = []
    for index in range(100):
      state.steeringAngleDeg = target_angle + (-0.2 if index % 2 else 0.2)
      output = update()
      assert output is not None
      feedforward.append(abs(output.feedforward_torque))
    boundary_steps = [abs(feedforward[index + 1] - feedforward[index]) for index in range(0, len(feedforward), 2)]
    assert max(boundary_steps) < 0.05

  def test_driver_handoff_caps_torque_and_preserves_plan_offset(self):
    _, _, probe = build_feedforward_boundary_controller()
    target_angle = probe().target_angle_deg

    # a fresh planner seeded far from the target: the driver has already overpowered it
    controller, state, update = build_feedforward_boundary_controller()
    state.steeringAngleDeg = target_angle + 20.0
    output = update()
    assert output is not None and abs(output.feedforward_torque) > 0.05
    offset_before = output.planned_angle_deg - state.steeringAngleDeg

    state.steeringPressed = True
    state.steeringTorque = -300.0
    state.steeringAngleDeg += 2.0
    handoff = update(4.0)
    assert handoff is not None
    normal_torque = max(-1.0, min(1.0, handoff.feedforward_torque + handoff.feedback_torque))
    assert normal_torque > MAX_DRIVER_ASSIST_TORQUE and handoff.torque == MAX_DRIVER_ASSIST_TORQUE
    assert handoff.feedback_torque != 0.0 and controller.status == STATUS_ACTIVE

    negative_handoff = update(-4.0)
    assert negative_handoff is not None
    normal_torque = max(-1.0, min(1.0, negative_handoff.feedforward_torque + negative_handoff.feedback_torque))
    assert normal_torque < -MAX_DRIVER_ASSIST_TORQUE and negative_handoff.torque == -MAX_DRIVER_ASSIST_TORQUE
    offset_during = negative_handoff.planned_angle_deg - state.steeringAngleDeg
    assert abs(offset_during - offset_before) < 0.25

    state.steeringPressed = False
    state.steeringTorque = 0.0
    release = update()
    assert release is not None and release.feedforward_torque != 0.0 and controller.status == STATUS_ACTIVE

  def test_driver_handoff_below_cap_passes_through_unclamped(self):
    _, _, probe = build_feedforward_boundary_controller()
    target_angle = probe().target_angle_deg

    # a fresh planner seeded exactly at the target: the driver's push is within the normal envelope
    _, state, update = build_feedforward_boundary_controller()
    state.steeringAngleDeg = target_angle
    state.steeringPressed = True
    state.steeringTorque = -300.0
    subcap = update()
    assert subcap is not None
    normal_torque = max(-1.0, min(1.0, subcap.feedforward_torque + subcap.feedback_torque))
    assert abs(normal_torque) < MAX_DRIVER_ASSIST_TORQUE
    assert math.isclose(subcap.torque, normal_torque, rel_tol=0.0, abs_tol=1e-9)

  # ---- LatControlRack: wraps the controller with a stock fallback ----

  def test_rack_matches_stock_torque_controller_without_a_model(self):
    controller, stock, VM = get_rack_controller()
    CS = car.CarState.new_message()
    CS.vEgo = 15.0
    params = log.VehicleParameters.new_message()
    for _ in range(50):
      torque, angle, rack_log = controller.update(True, CS, VM, params, False, 0.002, False, 0.2)
      stock_torque, _, _ = stock.update(True, CS, VM, params, False, 0.002, False, 0.2)
      assert math.isclose(torque, stock_torque, rel_tol=0.0, abs_tol=1e-12)
      assert angle == 0.0
      assert rack_log.fallback
      assert rack_log.status == STATUS_NO_MODEL
      assert rack_log.schema.node.id == log.ControlsState.LateralRackState.schema.node.id

  def test_rack_fallback_is_the_stock_controller_after_rack_frames(self):
    controller, stock, VM = get_rack_controller()
    CS = car.CarState.new_message()
    CS.vEgo = 15.0
    params = log.VehicleParameters.new_message()
    model = horizon_model([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], [0.0] * 6, [15.0] * 6)

    # a curve entry steered by the rack controller: the stock reference mirrors its idle, reset PID
    for frame in range(150):
      curvature = min(0.004, frame * 0.00004)
      CS.steeringAngleDeg = math.degrees(VM.get_steer_from_curvature(-curvature, CS.vEgo, 0.0))
      model.action.desiredCurvature = curvature
      model.orientationRate.z = [curvature * CS.vEgo] * 6
      set_curvature_preview(model)
      model.timestampEof = 1_000_000_000 + frame * 10_000_000
      mono_ns = model.timestampEof + 50_000_000
      _, _, rack_log = controller.update(True, CS, VM, params, False, curvature, False, 0.2, model=model, mono_time_ns=mono_ns)
      stock.update(True, CS, VM, params, False, curvature, False, 0.2)
      stock.pid.reset()
      assert not rack_log.fallback
      assert not rack_log.saturated
      # one saturation timer, shared with the idle stock controller
      assert controller.torque.sat_time == controller.sat_time

    # the model goes stale: every fallback frame is exactly what the stock controller would have done
    mono_ns = model.timestampEof + 600_000_000
    for frame in range(50):
      torque, _, rack_log = controller.update(True, CS, VM, params, False, 0.004, False, 0.2, model=model, mono_time_ns=mono_ns)
      stock_torque, _, stock_log = stock.update(True, CS, VM, params, False, 0.004, False, 0.2)
      assert controller.rack.status == STATUS_STALE_MODEL
      assert rack_log.fallback
      assert math.isclose(torque, stock_torque, rel_tol=0.0, abs_tol=1e-12)
      assert math.isclose(rack_log.f, stock_log.f, rel_tol=0.0, abs_tol=1e-12)
      assert controller.torque.sat_time == controller.sat_time
      if frame == 0:
        # the rack controller never saturated, so the handover must not report saturation from the idle stock controller
        assert not rack_log.saturated

  def test_torque_parameters_reach_the_stock_shadow(self):
    controller, _, _ = get_rack_controller()
    controller.update_torque_parameters(2.5, 0.125, 0.25)
    assert controller.torque.torque_params.latAccelFactor == 2.5
    assert controller.torque.torque_params.latAccelOffset == 0.125
    assert controller.torque.torque_params.friction == 0.25

  def _curve_model(self):
    model = horizon_model([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], [0.03] * 6, [15.0] * 6)
    model.action.desiredCurvature = 0.002
    return model

  def test_dropped_model_frames_keep_the_last_plan(self):
    controller, _, VM = get_rack_controller()
    CS = car.CarState.new_message()
    CS.vEgo = 15.0
    params = log.VehicleParameters.new_message()
    model = self._curve_model()
    controller.update(True, CS, VM, params, False, 0.002, False, 0.2, model=model, mono_time_ns=1_050_000_000)
    planner = controller.rack.planner

    # dropped model frames inside the stale window: the last plan is kept and advanced
    for frame in range(1, 4):
      _, _, rack_log = controller.update(True, CS, VM, params, False, 0.002, False, 0.2, model=None, mono_time_ns=1_050_000_000 + frame * 10_000_000)
      assert rack_log.status == STATUS_ACTIVE
      assert not rack_log.fallback
      assert controller.rack.planner is planner

    # beyond the window the stock controller takes over and the rack starts over
    stale_ns = 1_000_000_000 + int(STALE_MODEL_S * 1e9) + 10_000_000
    _, _, rack_log = controller.update(True, CS, VM, params, False, 0.002, False, 0.2, model=None, mono_time_ns=stale_ns)
    assert rack_log.fallback
    assert rack_log.status == STATUS_STALE_MODEL
    assert controller.rack.planner is None

  def test_fallback_holds_before_the_rack_resumes(self):
    controller, _, VM = get_rack_controller()
    CS = car.CarState.new_message()
    CS.vEgo = 15.0
    params = log.VehicleParameters.new_message()
    model = self._curve_model()
    controller.update(True, CS, VM, params, False, 0.002, False, 0.2, model=model, mono_time_ns=1_050_000_000)
    stale_ns = 1_000_000_000 + int(STALE_MODEL_S * 1e9) + 10_000_000
    _, _, rack_log = controller.update(True, CS, VM, params, False, 0.002, False, 0.2, model=model, mono_time_ns=stale_ns)
    assert rack_log.fallback

    # a fresh model the very next frame does not hand control back until the hold has elapsed
    hold_frames = int(FALLBACK_HOLD_S / DT_CTRL)
    for frame in range(hold_frames):
      model.timestampEof = stale_ns + frame * 10_000_000
      _, _, rack_log = controller.update(True, CS, VM, params, False, 0.002, False, 0.2, model=model, mono_time_ns=model.timestampEof)
      assert rack_log.fallback
    model.timestampEof = stale_ns + hold_frames * 10_000_000
    _, _, rack_log = controller.update(True, CS, VM, params, False, 0.002, False, 0.2, model=model, mono_time_ns=model.timestampEof)
    assert not rack_log.fallback
    assert rack_log.status == STATUS_ACTIVE

  def test_short_inactive_blip_keeps_the_planned_rack(self):
    controller, _, VM = get_rack_controller()
    CS = car.CarState.new_message()
    CS.vEgo = 15.0
    params = log.VehicleParameters.new_message()
    model = self._curve_model()
    controller.update(True, CS, VM, params, False, 0.002, False, 0.2, model=model, mono_time_ns=1_050_000_000)
    planner = controller.rack.planner

    # controlsd resets the controller before every inactive frame, e.g. across the standstill gate
    for _ in range(INACTIVE_HOLD_FRAMES):
      controller.reset()
      torque, _, rack_log = controller.update(False, CS, VM, params, False, 0.002, False, 0.2, model=model, mono_time_ns=1_050_000_000)
      assert torque == 0.0
      assert not rack_log.active
    assert controller.rack.planner is planner

    controller.update(True, CS, VM, params, False, 0.002, False, 0.2, model=model, mono_time_ns=1_060_000_000)
    assert controller.rack.planner is planner
    assert controller.rack.status == STATUS_ACTIVE

    for _ in range(INACTIVE_HOLD_FRAMES + 1):
      controller.reset()
    assert controller.rack.planner is None

  def test_content_fault_hands_back_on_the_next_good_frame(self):
    controller, _, VM = get_rack_controller()
    CS = car.CarState.new_message()
    CS.vEgo = 15.0
    params = log.VehicleParameters.new_message()
    model = self._curve_model()
    controller.update(True, CS, VM, params, False, 0.002, False, 0.2, model=model, mono_time_ns=1_050_000_000)

    # one garbage model frame is steered by stock; a good frame right after resumes the rack without a hold
    model.orientationRate.z = [0.03, 0.03, math.nan, 0.03, 0.03, 0.03]
    set_curvature_preview(model)
    _, _, rack_log = controller.update(True, CS, VM, params, False, 0.002, False, 0.2, model=model, mono_time_ns=1_060_000_000)
    assert rack_log.fallback
    assert rack_log.status == STATUS_INVALID_PATH
    model.orientationRate.z = [0.03] * 6
    set_curvature_preview(model)
    _, _, rack_log = controller.update(True, CS, VM, params, False, 0.002, False, 0.2, model=model, mono_time_ns=1_070_000_000)
    assert not rack_log.fallback
    assert rack_log.status == STATUS_ACTIVE

  def test_held_plan_follows_the_wheel_through_a_blip(self):
    controller, _, VM = get_rack_controller()
    CS = car.CarState.new_message()
    CS.vEgo = 15.0
    params = log.VehicleParameters.new_message()
    model = self._curve_model()
    controller.update(True, CS, VM, params, False, 0.002, False, 0.2, model=model, mono_time_ns=1_050_000_000)
    held_position = controller.rack.planner.position_deg

    # the wheel moves 3 degrees while the plan is held: the plan is carried along on resumption
    for _ in range(3):
      controller.reset()
      controller.update(False, CS, VM, params, False, 0.002, False, 0.2, model=model, mono_time_ns=1_050_000_000)
    CS.steeringAngleDeg = 3.0
    controller.update(True, CS, VM, params, False, 0.002, False, 0.2, model=model, mono_time_ns=1_090_000_000)
    assert controller.rack.planner is not None
    assert abs(controller.rack.planner.position_deg - held_position - 3.0) < 0.1

    # the same, with the driver's hand on the wheel as the plan resumes: the motion is carried once, not twice
    controller, _, VM = get_rack_controller()
    CS.steeringAngleDeg = 0.0
    controller.update(True, CS, VM, params, False, 0.002, False, 0.2, model=model, mono_time_ns=1_050_000_000)
    held_position = controller.rack.planner.position_deg
    for _ in range(3):
      controller.reset()
      controller.update(False, CS, VM, params, False, 0.002, False, 0.2, model=model, mono_time_ns=1_050_000_000)
    CS.steeringAngleDeg = 3.0
    CS.steeringPressed = True
    controller.update(True, CS, VM, params, False, 0.002, False, 0.2, model=model, mono_time_ns=1_090_000_000)
    assert abs(controller.rack.planner.position_deg - held_position - 3.0) < 0.1

  def test_blip_during_a_fallback_hold_keeps_the_hold(self):
    controller, _, VM = get_rack_controller()
    CS = car.CarState.new_message()
    CS.vEgo = 15.0
    params = log.VehicleParameters.new_message()
    model = self._curve_model()
    controller.update(True, CS, VM, params, False, 0.002, False, 0.2, model=model, mono_time_ns=1_050_000_000)
    stale_ns = 1_000_000_000 + int(STALE_MODEL_S * 1e9) + 10_000_000
    _, _, rack_log = controller.update(True, CS, VM, params, False, 0.002, False, 0.2, model=model, mono_time_ns=stale_ns)
    assert rack_log.fallback

    # an inactive blip inside the hold pauses it rather than ending it
    controller.reset()
    controller.update(False, CS, VM, params, False, 0.002, False, 0.2, model=model, mono_time_ns=stale_ns)
    model.timestampEof = stale_ns
    _, _, rack_log = controller.update(True, CS, VM, params, False, 0.002, False, 0.2, model=model, mono_time_ns=stale_ns)
    assert rack_log.fallback
    # a real disengage ends it
    for _ in range(INACTIVE_HOLD_FRAMES + 1):
      controller.reset()
    _, _, rack_log = controller.update(True, CS, VM, params, False, 0.002, False, 0.2, model=model, mono_time_ns=stale_ns)
    assert not rack_log.fallback

  def test_reset_clears_rack_state_and_output(self):
    controller, _, VM = get_rack_controller()
    CS = car.CarState.new_message()
    CS.vEgo = 15.0
    params = log.VehicleParameters.new_message()
    model = self._curve_model()
    controller.update(True, CS, VM, params, False, 0.002, False, 0.2, model=model, mono_time_ns=1_050_000_000)
    assert controller.rack.status == STATUS_ACTIVE
    assert controller.output is not None

    # one inactive frame holds the planned rack; a real disengage clears it
    controller.reset()
    assert controller.rack.status == STATUS_INACTIVE
    assert controller.rack.planner is not None
    assert controller.output is None
    for _ in range(INACTIVE_HOLD_FRAMES):
      controller.reset()
    assert controller.rack.planner is None
    assert controller.rack.reference_filter.target is None
    assert controller.rack.rack_rate_estimator.previous_angle_deg is None
    assert controller.rack.jerk_filter.x == 0.0
    assert controller.output is None

  # ---- behaviors carried over from BLaTv2's test_rack_trajectory_controller.py ----

  def test_envelope_recovery_does_not_drop_torque_for_virtual_or_measured_limit(self):
    self.CS.vEgo = 13.6
    boundary_curvature = -MAX_LATERAL_ACCEL_NO_ROLL / self.CS.vEgo ** 2
    boundary_angle = math.degrees(self.VM.get_steer_from_curvature(-boundary_curvature, self.CS.vEgo, 0.0))
    target_curvature = -(MAX_LATERAL_ACCEL_NO_ROLL - .01) / self.CS.vEgo ** 2
    model = horizon_model([0.0, .5, 1.0, 1.5, 2.0, 2.5], [0.0] * 6, [self.CS.vEgo] * 6)
    model.action.desiredCurvature = target_curvature

    def update(controller):
      controller.set_model(model, 1_050_000_000)
      return controller.update(True, self.CS, self.VM, self.params, self.CP.lateralTuning.torque,
                                self.CI.torque_from_lateral_accel(), .2, target_curvature)

    def run(seed_angle, seed_rate):
      controller = RackTrajectoryController()
      controller.planner = JerkLimitedRackPlanner(seed_angle, seed_rate)
      torques = []
      for _ in range(50):
        output = update(controller)
        assert output is not None
        torques.append(output.torque)
      assert all(torque != 0.0 for torque in torques)
      assert max(abs(torques[index] - torques[index - 1]) for index in range(1, len(torques))) < .1

    self.CS.steeringAngleDeg = boundary_angle - 15.0
    run(boundary_angle + .2, 20.0)  # the planner itself overshoots the virtual (planned) limit
    self.CS.steeringAngleDeg = boundary_angle + 10.0
    run(boundary_angle + 10.0, 0.0)  # the planner is seeded from a measured-angle overshoot instead

  def test_unwind_feedforward_releases_hold_torque_continuously(self):
    self.CS.vEgo = 13.6
    torque_params = self.CP.lateralTuning.torque
    torque_params.friction = 0.0
    torque_params.latAccelOffset = 0.0

    def run(planned_angle, measured_angle):
      controller = RackTrajectoryController()
      controller.planner = JerkLimitedRackPlanner(planned_angle)
      self.CS.steeringAngleDeg = measured_angle
      target_curvature = -self.VM.calc_curvature(math.radians(planned_angle), self.CS.vEgo, 0.0)
      model = horizon_model([0.0, .5, 1.0, 1.5, 2.0, 2.5], [0.0] * 6, [self.CS.vEgo] * 6)
      model.action.desiredCurvature = target_curvature
      controller.set_model(model, 1_050_000_000)
      output = controller.update(True, self.CS, self.VM, self.params, torque_params,
                                  lambda lateral_accel, _: lateral_accel, .2, target_curvature)
      assert output is not None
      return output

    # feedforward tracks only the target/plan: it stays continuous across the zero crossing and past
    # it, independent of the measured steering angle (position/rate feedback is a separate, additive term)
    hold = run(2.0, 2.0)
    at_center = run(-2.0, 0.0)
    cross_center = run(-2.0, 4.0)
    assert abs(hold.feedforward_torque) > .05
    assert math.isclose(at_center.feedforward_torque, -hold.feedforward_torque, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(cross_center.feedforward_torque, -hold.feedforward_torque, rel_tol=0.0, abs_tol=1e-9)

    torque_params.friction = -.5
    outward_cross_center = run(-2.0, 4.0)
    outward_at_center = run(-2.0, 0.0)
    # strong friction the wrong way flips feedforward's sign; with the plan and the commanded motion both
    # past center there is nothing left to hold, so torque still pushing the wheel outward is indefensible:
    # the direction guard ramps it out (at center exactly, the zero-crossing guard still zeroes outright)
    assert outward_cross_center.feedforward_torque > 0.0
    assert outward_cross_center.direction_guarded
    assert 0.0 <= outward_cross_center.torque < outward_cross_center.feedforward_torque + outward_cross_center.feedback_torque
    assert outward_at_center.torque == 0.0

    # held in the same state, the guard converges to zero torque within its ramp time
    controller = RackTrajectoryController()
    controller.planner = JerkLimitedRackPlanner(-2.0)
    self.CS.steeringAngleDeg = 4.0
    target_curvature = -self.VM.calc_curvature(math.radians(-2.0), self.CS.vEgo, 0.0)
    model = horizon_model([0.0, .5, 1.0, 1.5, 2.0, 2.5], [0.0] * 6, [self.CS.vEgo] * 6)
    model.action.desiredCurvature = target_curvature
    controller.set_model(model, 1_050_000_000)
    previous = math.inf
    output = None
    for _ in range(14):
      output = controller.update(True, self.CS, self.VM, self.params, torque_params,
                                 lambda lateral_accel, _: lateral_accel, .2, target_curvature)
      assert output is not None
      assert output.torque <= previous + 1e-12
      previous = output.torque
    assert output.torque == 0.0
    assert output.direction_guarded

  def test_reference_filter_isolates_raw_feedforward_during_a_small_reversal(self):
    controller = RackTrajectoryController()
    controller.planner = JerkLimitedRackPlanner(5.1)
    self.CS.vEgo = 30.0
    self.CS.steeringAngleDeg = 5.1
    torque_params = self.CP.lateralTuning.torque
    torque_params.friction = 0.0
    torque_params.latAccelOffset = 0.0
    torque_from_lateral_accel = self.CI.torque_from_lateral_accel()
    model = horizon_model([0.0, .5, 1.0, 1.5, 2.0, 2.5], [0.0] * 6, [self.CS.vEgo] * 6)

    output = None
    for model_frame, target_angle in enumerate((5.0, 5.5, 5.0)):
      desired_curvature = -self.VM.calc_curvature(math.radians(target_angle), self.CS.vEgo, 0.0)
      model.timestampEof = 1_000_000_000 + model_frame * 50_000_000
      model.action.desiredCurvature = desired_curvature
      controller.set_model(model, model.timestampEof + 50_000_000)
      output = controller.update(True, self.CS, self.VM, self.params, torque_params,
                                  torque_from_lateral_accel, .2, desired_curvature)
      assert output is not None

    # the served target is still on its way back from the wobble: the feedforward follows the plan, not the raw target
    assert controller.reference_filter.target is not None and controller.reference_filter.target.position_deg > 5.02
    expected_planned = -float(torque_from_lateral_accel(output.desired_lateral_accel, torque_params))
    raw_bypass = -float(torque_from_lateral_accel(output.target_curvature * self.CS.vEgo ** 2, torque_params))
    assert abs(output.feedforward_torque - expected_planned) < 0.01
    assert abs(output.feedforward_torque - raw_bypass) > 1e-4

  def test_model_wobble_is_smoothed_in_rack_space_at_every_speed(self):
    torque_from_lateral_accel = self.CI.torque_from_lateral_accel()
    for speed in (5.0, 15.0, 30.0):
      controller = RackTrajectoryController()
      controller.planner = JerkLimitedRackPlanner(5.1)
      CS = car.CarState.new_message()
      CS.vEgo = speed
      CS.steeringAngleDeg = 5.1
      model = horizon_model([0.0, .5, 1.0, 1.5, 2.0, 2.5], [0.0] * 6, [speed] * 6)

      served = []
      accelerations = []
      for index in range(15):
        model_frame = index // 5
        target_angle = (5.0, 5.5, 5.0)[model_frame]
        desired_curvature = -self.VM.calc_curvature(math.radians(target_angle), speed, 0.0)
        model.timestampEof = 1_000_000_000 + model_frame * 50_000_000
        model.action.desiredCurvature = desired_curvature
        controller.set_model(model, model.timestampEof + 50_000_000)
        output = controller.update(True, CS, self.VM, self.params, self.CP.lateralTuning.torque,
                                    torque_from_lateral_accel, .2, desired_curvature)
        assert output is not None
        served.append(output.target_angle_deg)
        accelerations.append(output.planned_acceleration_deg_s2)

      # after the wobble reverses, the served target is still between the two raw values
      assert all(5.0 < angle < 5.5 for angle in served[10:])
      assert accelerations[9] * accelerations[10] >= 0.0

  def test_profile_transition_headroom_does_not_walk_outward(self):
    self.CS.vEgo = 30.0
    controller = RackTrajectoryController()
    controller.planner = JerkLimitedRackPlanner(0.0, 300.0)
    controller.planner.acceleration_deg_s2 = 500.0
    model = horizon_model([0.0, .5, 1.0, 1.5, 2.0, 2.5], [0.0] * 6, [self.CS.vEgo] * 6)
    torque_from_lateral_accel = self.CI.torque_from_lateral_accel()

    rate_ceiling = acceleration_ceiling = None
    output = None
    for frame in range(600):
      model.timestampEof = 1_000_000_000 + frame * 10_000_000
      controller.set_model(model, model.timestampEof + 50_000_000)
      output = controller.update(True, self.CS, self.VM, self.params, self.CP.lateralTuning.torque,
                                  torque_from_lateral_accel, .2, 0.0)
      assert output is not None
      if rate_ceiling is None:
        rate_ceiling, acceleration_ceiling = output.rate_limit_deg_s, output.acceleration_limit_deg_s2
      # the headroom given to an overshooting planner never grows once established: it only relaxes
      assert output.rate_limit_deg_s <= rate_ceiling + 1e-9
      assert output.acceleration_limit_deg_s2 <= acceleration_ceiling + 1e-9

    assert not output.profile_transition
    assert abs(output.planned_rate_deg_s) <= output.rate_limit_deg_s + 1e-6

  def test_invalid_vehicle_state_falls_back_to_stock_torque_and_status(self):
    controller, stock, VM = get_rack_controller()
    CS = car.CarState.new_message()
    CS.vEgo = 15.0
    CS.steeringTorque = math.nan
    params = log.VehicleParameters.new_message()
    model = horizon_model([0.0, .5, 1.0, 1.5, 2.0, 2.5], [.03] * 6, [15.0] * 6)
    model.action.desiredCurvature = 0.002
    torque, _, rack_log = controller.update(True, CS, VM, params, False, 0.002, False, .2, model=model, mono_time_ns=1_050_000_000)
    stock_torque, _, _ = stock.update(True, CS, VM, params, False, 0.002, False, .2)
    assert controller.rack.status == STATUS_INVALID_VEHICLE_STATE
    assert rack_log.fallback
    assert math.isclose(torque, stock_torque, rel_tol=0.0, abs_tol=1e-12)

  def test_missing_preview_falls_back_to_stock_torque_and_status(self):
    controller, stock, VM = get_rack_controller()
    CS = car.CarState.new_message()
    CS.vEgo = 15.0
    params = log.VehicleParameters.new_message()
    model = horizon_model([0.0, .5, 1.0, 1.5, 2.0, 2.5], [.03] * 6, [15.0] * 6)
    model.action.desiredCurvature = 0.002
    # a modeld that predates the preview leaves the lists empty: there is no path to build
    model.action.desiredCurvaturePreview = []
    model.action.desiredCurvaturePreviewTimes = []
    torque, _, rack_log = controller.update(True, CS, VM, params, False, 0.002, False, .2, model=model, mono_time_ns=1_050_000_000)
    stock_torque, _, _ = stock.update(True, CS, VM, params, False, 0.002, False, .2)
    assert controller.rack.status == STATUS_INVALID_PREVIEW
    assert rack_log.fallback
    assert math.isclose(torque, stock_torque, rel_tol=0.0, abs_tol=1e-12)

  def test_horizon_acceleration_uses_future_shape_without_targeting_endpoint(self):
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

  def test_horizon_admission_rejects_each_immediate_path_violation(self):
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

  def test_live_horizon_prepares_for_future_shape_without_moving_the_immediate_target(self):
    self.CS.vEgo = 5.0

    def update(orientation_rates):
      controller = RackTrajectoryController()
      model = horizon_model([0.0, .5, 1.0, 1.5, 2.0, 2.5], orientation_rates, [self.CS.vEgo] * 6)
      controller.set_model(model, 1_050_000_000)
      return controller.update(True, self.CS, self.VM, self.params, self.CP.lateralTuning.torque,
                                self.CI.torque_from_lateral_accel(), .2, 0.0), model

    flat, _ = update([0.0] * 6)
    future_turn, future_model = update([0.0, 0.0, .0001, .0002, .0001, 0.0])
    assert flat is not None and future_turn is not None
    # the immediate target is the scalar as published, with the preview's slope at the action time as its rate
    assert abs(flat.target_angle_deg - future_turn.target_angle_deg) < 1e-9
    assert abs(future_turn.target_angle_deg) < 1e-9
    assert -0.2 < future_turn.target_rate_deg_s < 0.0
    assert future_turn.planned_acceleration_deg_s2 != flat.planned_acceleration_deg_s2
    assert abs(future_turn.planned_angle_deg) <= HORIZON_POSITION_TOLERANCE_DEG

    future_targets = model_path_targets(
      native_times_s=future_model.velocity.t, velocities_x=future_model.velocity.x,
      preview_times_s=future_model.action.desiredCurvaturePreviewTimes,
      preview_curvatures=future_model.action.desiredCurvaturePreview, scalar_curvature=0.0, plan_time_now_s=.05,
      measured_v_ego=self.CS.vEgo, query_times_s=[.05 + offset for offset in HORIZON_OFFSETS_S],
      vehicle_model=self.VM, roll_rad=0.0, angle_offset_deg=0.0,
    )
    fitted = horizon_desired_acceleration(
      JerkLimitedRackPlanner(0.0),
      tuple((offset, RackTarget(target.angle_deg, target.rate_deg_s))
            for offset, target in zip(HORIZON_OFFSETS_S, future_targets, strict=True) if offset > 0.0),
    )
    assert HORIZON_ACCELERATION_BLEND == .1
    # a fresh planner at rest at zero: the reactive term is the critically damped tracker's response
    natural_frequency = 2.0 / RESPONSE_TIME_S
    reactive = natural_frequency ** 2 * future_turn.target_angle_deg + 2.0 * natural_frequency * future_turn.target_rate_deg_s
    assert abs(future_turn.planned_acceleration_deg_s2 - (reactive + HORIZON_ACCELERATION_BLEND * (fitted - reactive))) < 1e-9

  def test_driver_handoff_converges_through_platform_torque_limits(self):
    limits = CarControllerParams(self.CP)
    capped_request = -round(MAX_DRIVER_ASSIST_TORQUE * limits.STEER_MAX)
    # this fork's Palisade authority (STEER_MAX/DELTA_UP/DOWN) differs from BLaTv2's 409/4/7 tune, so the
    # converged values differ too; the mechanism under test -- driver override progressively releasing
    # the capped rack request back toward neutral -- is what's being ported, not the specific numbers
    for driver_torque, expected in ((150.0, -184), (200.0, -84), (255.0, 0)):
      applied = capped_request
      for _ in range(100):
        applied = apply_driver_steer_torque_limits(capped_request, applied, driver_torque, limits)
      assert applied == expected

  def test_rack_active_saturation_sets_and_clears(self):
    controller, _, VM = get_rack_controller()
    CS = car.CarState.new_message()
    CS.vEgo = 15.0
    params = log.VehicleParameters.new_message()
    model = horizon_model([0.0, .5, 1.0, 1.5, 2.0, 2.5], [0.0] * 6, [15.0] * 6)

    def step(frame, curvature):
      model.action.desiredCurvature = curvature
      model.timestampEof = 1_000_000_000 + frame * 10_000_000
      mono_ns = model.timestampEof + 50_000_000
      return controller.update(True, CS, VM, params, False, curvature, False, .2, model=model, mono_time_ns=mono_ns)

    rack_log = None
    for frame in range(200):
      _, _, rack_log = step(frame, .05)
      assert not rack_log.fallback
    assert rack_log.saturated

    for frame in range(200, 500):
      _, _, rack_log = step(frame, 0.0)
    assert not rack_log.saturated

  # ---- phase 3 step 2: slew-aware early release + direction fraction ----

  def _flip_model(self, angle_deg, flip=True):
    curvature_now = -self.VM.calc_curvature(math.radians(angle_deg), self.CS.vEgo, 0.0)
    yaw = curvature_now * self.CS.vEgo
    rates = [yaw, yaw, -yaw, -yaw, -yaw, -yaw] if flip else [yaw] * 6
    model = horizon_model([0.0, .5, 1.0, 1.5, 2.0, 2.5], rates, [self.CS.vEgo] * 6)
    model.action.desiredCurvature = curvature_now
    return model, curvature_now



  def test_direction_fraction_fades_at_center(self):
    # a near-center dither with the plan lagging to one side must not pin the fraction at +/-1,
    # strip the rate damping, or arm the direction guard on alternating frames
    torque_params = self.CP.lateralTuning.torque
    torque_params.friction = 0.0
    torque_params.latAccelOffset = 0.0
    controller = RackTrajectoryController()
    controller.planner = JerkLimitedRackPlanner(1.8)
    model = horizon_model([0.0, .5, 1.0, 1.5, 2.0, 2.5], [0.0] * 6, [15.0] * 6)
    model.action.desiredCurvature = 0.0
    controller.set_model(model, 1_050_000_000)
    for frame in range(8):
      self.CS.steeringAngleDeg = 0.1 if frame % 2 == 0 else -0.1
      self.CS.steeringRateDeg = -20.0
      output = controller.update(True, self.CS, self.VM, self.params, torque_params,
                                 lambda lateral_accel, _: lateral_accel, .2, 0.0)
      assert output is not None
      assert abs(output.direction_fraction) < 0.05
      assert not output.direction_guarded


  def test_direction_fraction_relaxes_rate_feedback_on_unwind(self):
    torque_params = self.CP.lateralTuning.torque
    torque_params.friction = 0.0
    torque_params.latAccelOffset = 0.0
    gain = RackTrajectoryController._feedback_gain(15.0)
    lateral_accel_per_degree = -self.VM.calc_curvature(math.radians(1.0), 15.0, 0.0) * 15.0 ** 2

    def run(planner_angle, measured_angle, target_angle):
      controller = RackTrajectoryController()
      controller.planner = JerkLimitedRackPlanner(planner_angle)
      self.CS.steeringAngleDeg = measured_angle
      self.CS.steeringRateDeg = -80.0
      curvature = -self.VM.calc_curvature(math.radians(target_angle), 15.0, 0.0)
      yaw = curvature * 15.0
      model = horizon_model([0.0, .5, 1.0, 1.5, 2.0, 2.5], [yaw] * 6, [15.0] * 6)
      model.action.desiredCurvature = curvature
      controller.set_model(model, 1_050_000_000)
      output = None
      for _frame in range(2):
        output = controller.update(True, self.CS, self.VM, self.params, torque_params,
                                   lambda lateral_accel, _: lateral_accel, .2, curvature)
        assert output is not None
      return output

    def expected_rate_feedback(output):
      scale = _smoothstep(-output.direction_fraction, -1.0, 0.0)
      return -(gain * scale * lateral_accel_per_degree * .1
               * (output.planned_rate_deg_s - output.measured_rate_deg_s)), scale

    # unwinding: the wheel far past what the plan still needs, the rate feedback relaxed by the fraction
    unwind = run(3.0, 12.0, 0.0)
    assert 0.2 < unwind.direction_fraction < 1.0
    expected, scale = expected_rate_feedback(unwind)
    assert 0.0 < scale < 1.0
    assert math.isclose(unwind.rate_feedback_torque, expected, rel_tol=1e-9, abs_tol=1e-12)
    assert abs(unwind.rate_feedback_torque) > 1e-6

    # the fraction is computed on offset-corrected angles: shifting the whole scene by the
    # angle offset leaves it unchanged
    self.params.angleOffsetDeg = 4.0
    shifted = run(3.0 + 4.0, 12.0 + 4.0, 0.0)
    self.params.angleOffsetDeg = 0.0
    assert math.isclose(shifted.direction_fraction, unwind.direction_fraction, abs_tol=1e-6)

    # holding the needed angle: fraction zero, bit-identical to the unscaled law
    hold = run(12.0, 12.0, 12.0)
    assert abs(hold.direction_fraction) < 1e-6
    expected, scale = expected_rate_feedback(hold)
    assert math.isclose(scale, 1.0, rel_tol=1e-9)
    assert math.isclose(hold.rate_feedback_torque, expected, rel_tol=1e-9, abs_tol=1e-12)

  def test_aligning_mechanisms_agree_in_sign(self):
    torque_params = self.CP.lateralTuning.torque
    torque_params.friction = 0.0
    torque_params.latAccelOffset = 0.0

    def run(sign):
      controller = RackTrajectoryController()
      controller.planner = JerkLimitedRackPlanner(sign * 3.0)
      # a real unwind: the angle falls toward center while the CAN rate reports the return; the raw
      # rate is unsigned-magnitude on the positive side, so the moving angle is what signs it
      self.CS.steeringRateDeg = sign * -80.0
      model = horizon_model([0.0, .5, 1.0, 1.5, 2.0, 2.5], [0.0] * 6, [15.0] * 6)
      model.action.desiredCurvature = 0.0
      controller.set_model(model, 1_050_000_000)
      output = None
      for angle in (12.4, 12.0):
        self.CS.steeringAngleDeg = sign * angle
        output = controller.update(True, self.CS, self.VM, self.params, torque_params,
                                   lambda lateral_accel, _: lateral_accel, .2, 0.0)
        assert output is not None
      return output

    for sign in (1.0, -1.0):
      output = run(sign)
      scale = _smoothstep(-output.direction_fraction, -1.0, 0.0)
      assert 0.0 < scale < 1.0
      # mechanism 1: what the relaxed rate feedback gave up relative to the unscaled law
      mechanism_1 = output.rate_feedback_torque - output.rate_feedback_torque / scale
      # mechanism 2 (cross-check only): the feedforward re-evaluated at MEASURED instead of planned
      # lateral accel, with the file's own negation. During an unwind the wheel sits beyond the plan,
      # so this delta is EXTRA torque toward holding -- the surplus the road's self-aligning force is
      # already supplying on its own. Mechanism 1 gives up resistance instead of adding that surplus,
      # so its delta must OPPOSE mechanism 2's sign while tracking its rough magnitude. (The design
      # text asked for sign agreement; on its own reference frame the literal FF delta and the
      # relaxation are numerically opposite -- review-verified -- because one measures the surplus
      # and the other the resistance given up.)
      mechanism_2 = -output.actual_lateral_accel + output.desired_lateral_accel
      assert abs(mechanism_1) > 1e-6 and abs(mechanism_2) > 1e-6
      assert mechanism_1 * mechanism_2 < 0.0
      assert 1.0 / 30.0 < abs(mechanism_1) / abs(mechanism_2) < 30.0

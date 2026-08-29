import math

from openpilot.common.test import OpenpilotTestCase

from openpilot.cereal import log, messaging
from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.values import CAR as HYUNDAI
from opendbc.car.structs import car
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.controls.lib.latcontrol_rack import LatControlRack
from openpilot.selfdrive.controls.lib.rack_trajectory import (
  JerkLimitedRackPlanner,
  MAX_DRIVER_ASSIST_TORQUE,
  MEASURED_RATE_FILTER_RC_S,
  MotionLimits,
  RackRateEstimator,
  RackReferenceGovernor,
  RackTarget,
  RackTrajectoryController,
  model_path_targets,
  STATUS_ACTIVE,
  STATUS_INACTIVE,
  STATUS_INVALID_PATH,
  STATUS_NO_MODEL,
  STATUS_STALE_MODEL,
)


class LinearVehicleModel:
  @staticmethod
  def get_steer_from_curvature(curvature, speed, roll):
    del speed, roll
    return curvature * 10.0


def govern_reference(governor, target, planner, model_frame, *, bypass=False):
  return governor.update(target, planner, 1_000_000_000 + model_frame * 50_000_000, 0.01, bypass)


def get_rack_controller(car_name=HYUNDAI.HYUNDAI_PALISADE):
  CarInterface = interfaces[car_name]
  CP = CarInterface.get_non_essential_params(car_name)
  CI = CarInterface(CP)
  VM = VehicleModel(CP)
  controller = LatControlRack(CP.as_reader(), CI, DT_CTRL)
  stock = LatControlTorque(CP.as_reader(), CI, DT_CTRL)
  return controller, stock, VM


def horizon_model(times, rates, speeds):
  message = messaging.new_message("modelV2").modelV2
  message.timestampEof = 1_000_000_000
  message.action.desiredCurvature = 0.0
  message.action.desiredCurvatureTime = 0.5
  message.orientationRate.t = times
  message.orientationRate.z = rates
  message.velocity.x = speeds
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
  model = messaging.new_message("modelV2").modelV2
  model.timestampEof = 1_000_000_000
  model.action.desiredCurvature = -0.02
  model.action.desiredCurvatureTime = 0.5
  model.orientationRate.t = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
  model.orientationRate.z = [0.0] * 6
  model.velocity.x = [5.0] * 6
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

  # ---- model_path_targets / JerkLimitedRackPlanner / RackReferenceGovernor / RackRateEstimator ----

  def test_model_path_is_scalar_anchored_and_signed(self):
    times = [0.0, 1.0, 2.0, 3.0]
    speeds = [5.0] * 4
    target = model_path_targets(
      native_times_s=times, orientation_rates_z=[0.0, 0.05, 0.1, 0.15], velocities_x=speeds,
      scalar_curvature=0.03, scalar_action_plan_s=1.0, plan_time_now_s=0.1, measured_v_ego=5.0,
      query_times_s=(2.0,), vehicle_model=LinearVehicleModel(), roll_rad=0.0, angle_offset_deg=0.0,
    )[0]
    expected_curvature = 0.03 + (0.1 / 5.0) - (0.05 / 5.0)
    assert abs(target.curvature - expected_curvature) < 1e-12
    assert abs(target.angle_deg - math.degrees(-expected_curvature * 10.0)) < 1e-12
    assert target.rate_deg_s < 0.0

    raised = False
    try:
      model_path_targets(
        native_times_s=times, orientation_rates_z=[0.0, 0.05, 0.1, 0.15], velocities_x=speeds,
        scalar_curvature=0.03, scalar_action_plan_s=1.0, plan_time_now_s=0.1, measured_v_ego=5.0,
        query_times_s=(4.0,), vehicle_model=LinearVehicleModel(), roll_rad=0.0, angle_offset_deg=0.0,
      )
    except ValueError:
      raised = True
    assert raised

    stopped = model_path_targets(
      native_times_s=times, orientation_rates_z=[0.0, 0.05, 0.1, 0.15], velocities_x=[5.0, 5.0, 0.0, 0.0],
      scalar_curvature=0.03, scalar_action_plan_s=1.0, plan_time_now_s=0.1, measured_v_ego=5.0,
      query_times_s=(2.0,), vehicle_model=LinearVehicleModel(), roll_rad=0.0, angle_offset_deg=0.0,
    )[0]
    # the stopped samples are kept and hold the last well-conditioned curvature (the 0.05 / 5.0 knot at 1 s)
    assert abs(stopped.curvature - 0.03) < 1e-12
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

  def test_reference_governor_holds_and_filters_small_reversals(self):
    planner = JerkLimitedRackPlanner(5.0)
    governor = RackReferenceGovernor()
    assert govern_reference(governor, RackTarget(5.0, 0.0), planner, 0) == RackTarget(5.0, 0.0)
    assert govern_reference(governor, RackTarget(5.5, 0.0), planner, 1) == RackTarget(5.5, 0.0)
    accepted = govern_reference(governor, RackTarget(4.9, 0.0), planner, 2)
    for _ in range(20):
      accepted = govern_reference(governor, RackTarget(4.9, 0.0), planner, 2)
      assert 4.9 < accepted.position_deg < 5.5
      assert governor.limited

    neutral_planner = JerkLimitedRackPlanner(0.0)
    neutral_governor = RackReferenceGovernor()
    govern_reference(neutral_governor, RackTarget(0.0, 0.0), neutral_planner, 0)
    govern_reference(neutral_governor, RackTarget(0.4, 0.0), neutral_planner, 1)
    neutral_accepted = govern_reference(neutral_governor, RackTarget(-0.2, 0.0), neutral_planner, 2)
    assert -0.2 < neutral_accepted.position_deg < 0.4
    assert neutral_governor.limited

  def test_reference_governor_passes_large_persistent_and_necessary_reversals(self):
    stationary = JerkLimitedRackPlanner(5.0)

    large = RackReferenceGovernor()
    govern_reference(large, RackTarget(5.0, 0.0), stationary, 0)
    govern_reference(large, RackTarget(6.0, 0.0), stationary, 1)
    assert govern_reference(large, RackTarget(3.0, 0.0), stationary, 2) == RackTarget(3.0, 0.0)

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
    invalid_paths = (
      horizon_model([0.0, 0.5, 1.0], [0.0] * 3, [5.0] * 3),
      horizon_model([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], [0.0] * 5, [5.0] * 6),
      horizon_model([0.0, 0.5, 0.4, 1.5, 2.0, 2.5], [0.0] * 6, [5.0] * 6),
      horizon_model([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], [0.0, 0.0, math.nan, 0.0, 0.0, 0.0], [5.0] * 6),
    )
    cases = tuple((path, 1_050_000_000, STATUS_INVALID_PATH) for path in invalid_paths) + (
      (valid, 1_200_000_001, STATUS_STALE_MODEL),
    )
    for invalid, mono_ns, expected_status in cases:
      controller = RackTrajectoryController()
      assert self._update_horizon(controller, valid) is not None
      assert controller.planner is not None and controller.reference_governor.accepted is not None
      assert self._update_horizon(controller, invalid, mono_ns) is None
      assert controller.status == expected_status
      assert controller.planner is None
      assert controller.reference_governor.accepted is None
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
    model = messaging.new_message("modelV2").modelV2
    model.action.desiredCurvatureTime = 0.5
    model.orientationRate.t = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    model.velocity.x = [15.0] * 6

    # a curve entry steered by the rack controller: the stock reference mirrors its idle, reset PID
    for frame in range(150):
      curvature = min(0.004, frame * 0.00004)
      CS.steeringAngleDeg = math.degrees(VM.get_steer_from_curvature(-curvature, CS.vEgo, 0.0))
      model.action.desiredCurvature = curvature
      model.orientationRate.z = [curvature * CS.vEgo] * 6
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
    mono_ns = model.timestampEof + 500_000_000
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

  def test_reset_clears_rack_state_and_output(self):
    controller, _, VM = get_rack_controller()
    CS = car.CarState.new_message()
    CS.vEgo = 15.0
    params = log.VehicleParameters.new_message()
    model = messaging.new_message("modelV2").modelV2
    model.timestampEof = 1_000_000_000
    model.action.desiredCurvature = 0.002
    model.action.desiredCurvatureTime = 0.5
    model.orientationRate.t = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    model.orientationRate.z = [0.03] * 6
    model.velocity.x = [15.0] * 6
    controller.update(True, CS, VM, params, False, 0.002, False, 0.2, model=model, mono_time_ns=1_050_000_000)
    assert controller.rack.status == STATUS_ACTIVE
    assert controller.output is not None

    controller.reset()
    assert controller.rack.status == STATUS_INACTIVE
    assert controller.rack.planner is None
    assert controller.output is None

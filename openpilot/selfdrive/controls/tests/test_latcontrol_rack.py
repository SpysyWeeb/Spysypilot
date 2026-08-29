import math

from openpilot.common.test import OpenpilotTestCase

from openpilot.cereal import log, messaging
from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.values import CAR as HYUNDAI, CarControllerParams
from opendbc.car.lateral import apply_driver_steer_torque_limits
from opendbc.car.structs import car
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import MAX_LATERAL_ACCEL_NO_ROLL
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.controls.lib.latcontrol_rack import FALLBACK_HOLD_S, LatControlRack
from openpilot.selfdrive.controls.lib.rack_trajectory import (
  HORIZON_ACCELERATION_BLEND,
  HORIZON_OFFSETS_S,
  HORIZON_POSITION_TOLERANCE_DEG,
  horizon_candidate_preserves_immediate_path,
  horizon_desired_acceleration,
  JerkLimitedRackPlanner,
  MAX_DRIVER_ASSIST_TORQUE,
  MEASURED_RATE_FILTER_RC_S,
  MotionLimits,
  RackPlan,
  RackRateEstimator,
  RackReferenceGovernor,
  RackTarget,
  RackTrajectoryController,
  model_path_targets,
  REFERENCE_REVERSAL_RC_S,
  STATUS_ACTIVE,
  STATUS_INACTIVE,
  STATUS_INVALID_ACTION_TIME,
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
      (valid, 1_600_000_001, STATUS_STALE_MODEL),
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
    model = messaging.new_message("modelV2").modelV2
    model.timestampEof = 1_000_000_000
    model.action.desiredCurvature = 0.002
    model.action.desiredCurvatureTime = 0.5
    model.orientationRate.t = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    model.orientationRate.z = [0.03] * 6
    model.velocity.x = [15.0] * 6
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
    _, _, rack_log = controller.update(True, CS, VM, params, False, 0.002, False, 0.2, model=model, mono_time_ns=1_060_000_000)
    assert rack_log.fallback
    assert rack_log.status == STATUS_INVALID_PATH
    model.orientationRate.z = [0.03] * 6
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

    # one inactive frame holds the planned rack; a real disengage clears it
    controller.reset()
    assert controller.rack.status == STATUS_INACTIVE
    assert controller.rack.planner is not None
    assert controller.output is None
    for _ in range(INACTIVE_HOLD_FRAMES):
      controller.reset()
    assert controller.rack.planner is None
    assert controller.rack.reference_governor.accepted is None
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
    # strong friction the wrong way flips feedforward's sign; the unwind/cross-center sign logic then
    # zeroes raw torque outright rather than drive the wheel further off the driver-relevant target
    assert outward_cross_center.feedforward_torque > 0.0
    assert outward_cross_center.torque == 0.0
    assert outward_at_center.torque == 0.0

  def test_reference_governor_uses_constant_response_and_converges(self):
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

  def test_reference_governor_preserves_gradual_neutral_crossing_continuity(self):
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

  def test_reference_governor_threshold_edges_for_distance_and_rate(self):
    stationary = JerkLimitedRackPlanner(5.0)

    below_distance = RackReferenceGovernor()
    govern_reference(below_distance, RackTarget(5.0, 0.0), stationary, 0)
    govern_reference(below_distance, RackTarget(5.5, 0.0), stationary, 1)
    governed = govern_reference(below_distance, RackTarget(4.5001, 0.0), stationary, 2)
    assert governed != RackTarget(4.5001, 0.0) and below_distance.limited

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

  def test_reference_governor_isolates_raw_feedforward_during_a_limited_reversal(self):
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

    assert controller.reference_governor.limited
    expected_planned = -float(torque_from_lateral_accel(output.desired_lateral_accel, torque_params))
    raw_bypass = -float(torque_from_lateral_accel(output.target_curvature * self.CS.vEgo ** 2, torque_params))
    assert abs(output.feedforward_torque - expected_planned) < 1e-9
    assert abs(output.feedforward_torque - raw_bypass) > 1e-4

  def test_reference_governor_passes_crossing_and_fast_reversals_and_bypass_recovery(self):
    stationary = JerkLimitedRackPlanner(5.0)

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

  def test_model_wobble_is_governed_in_rack_space_at_every_speed(self):
    torque_from_lateral_accel = self.CI.torque_from_lateral_accel()
    for speed in (5.0, 15.0, 30.0):
      controller = RackTrajectoryController()
      controller.planner = JerkLimitedRackPlanner(5.1)
      CS = car.CarState.new_message()
      CS.vEgo = speed
      CS.steeringAngleDeg = 5.1
      model = horizon_model([0.0, .5, 1.0, 1.5, 2.0, 2.5], [0.0] * 6, [speed] * 6)

      limited = []
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
        limited.append(controller.reference_governor.limited)
        accelerations.append(output.planned_acceleration_deg_s2)

      assert any(limited[10:])
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

  def test_invalid_action_time_falls_back_to_stock_torque_and_status(self):
    controller, stock, VM = get_rack_controller()
    CS = car.CarState.new_message()
    CS.vEgo = 15.0
    params = log.VehicleParameters.new_message()
    model = horizon_model([0.0, .5, 1.0, 1.5, 2.0, 2.5], [.03] * 6, [15.0] * 6)
    model.action.desiredCurvature = 0.002
    model.action.desiredCurvatureTime = 0.0  # left at its zero default: the action horizon is undefined
    torque, _, rack_log = controller.update(True, CS, VM, params, False, 0.002, False, .2, model=model, mono_time_ns=1_050_000_000)
    stock_torque, _, _ = stock.update(True, CS, VM, params, False, 0.002, False, .2)
    assert controller.rack.status == STATUS_INVALID_ACTION_TIME
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
    assert abs(flat.target_angle_deg - future_turn.target_angle_deg) < 1e-9
    assert abs(future_turn.target_angle_deg) < 1e-9
    assert future_turn.planned_acceleration_deg_s2 != flat.planned_acceleration_deg_s2
    assert abs(future_turn.planned_angle_deg) <= HORIZON_POSITION_TOLERANCE_DEG

    future_targets = model_path_targets(
      native_times_s=future_model.orientationRate.t, orientation_rates_z=future_model.orientationRate.z,
      velocities_x=future_model.velocity.x, scalar_curvature=0.0, scalar_action_plan_s=.5, plan_time_now_s=.05,
      measured_v_ego=self.CS.vEgo, query_times_s=[.05 + offset for offset in HORIZON_OFFSETS_S],
      vehicle_model=self.VM, roll_rad=0.0, angle_offset_deg=0.0,
    )
    fitted = horizon_desired_acceleration(
      JerkLimitedRackPlanner(0.0),
      tuple((offset, RackTarget(target.angle_deg, target.rate_deg_s))
            for offset, target in zip(HORIZON_OFFSETS_S, future_targets, strict=True) if offset > 0.0),
    )
    assert HORIZON_ACCELERATION_BLEND == .1
    assert abs(future_turn.planned_acceleration_deg_s2 - HORIZON_ACCELERATION_BLEND * fitted) < 1e-9

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

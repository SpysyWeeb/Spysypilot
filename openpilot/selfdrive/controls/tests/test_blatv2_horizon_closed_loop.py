from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from pathlib import Path

from openpilot.selfdrive.controls.lib.blatv2.actuator import (
  RuntimeTorqueLimits,
  apply_torque_envelope_counts,
)
from openpilot.selfdrive.controls.lib.blatv2.horizon import (
  CONTROL_DT_SECONDS,
  HORIZON_SAMPLE_COUNT,
  HorizonController,
  HorizonPolicy,
  HorizonStatus,
)
from openpilot.selfdrive.controls.lib.blatv2.plant import (
  RackState,
  TrackingPolicy,
  step_plant,
)
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import (
  RackMappingSnapshot,
  curvature_from_measured_angle,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  PhysicalParameters,
  ProfileNode,
  VehicleProfile,
)


POLICY_PATH = Path(__file__).parents[1] / "lib" / "blatv2" / "provisional_horizon_policy.json"
STEER_MAX = 409
TRACKING_TOLERANCE_DEG = 0.04

AnglePlan = Callable[[float, float], float]
MotionPlan = Callable[[float, float], tuple[float, float, float]]
ScalarPlan = Callable[[float], float]
PressedPlan = Callable[[float], bool]


@dataclass(frozen=True, slots=True)
class ClosedLoopFrame:
  time_s: float
  speed_mps: float
  desired_angle_deg: float
  horizon_angle_deg: float
  reactive_angle_deg: float
  horizon_rate_deg_s: float
  reactive_rate_deg_s: float
  horizon_counts: int
  reactive_counts: int
  horizon_previous_counts: int
  reactive_previous_counts: int
  horizon_raw_torque: float
  reactive_raw_torque: float
  preparation_active: bool
  preparation_scheduled: bool


def limits() -> RuntimeTorqueLimits:
  return RuntimeTorqueLimits(
    steer_max=STEER_MAX,
    delta_up=4,
    delta_down=7,
    steer_step=1,
    driver_allowance=50,
    driver_multiplier=2,
    driver_factor=1,
    production_envelope_verified=True,
  )


def mapping() -> RackMappingSnapshot:
  return RackMappingSnapshot(
    mass_kg=2100.0,
    wheelbase_m=2.9,
    center_to_front_m=1.2,
    center_to_rear_m=1.7,
    tire_stiffness_front=100000.0,
    tire_stiffness_rear=110000.0,
    steer_ratio_rear=0.0,
    steer_ratio=15.0,
    roll_rad=0.0,
    angle_offset_deg=0.0,
    valid=True,
  )


def profile() -> VehicleProfile:
  transient_nodes = (
    (0.0, 4000.0, 10.0),
    (5.0, 4000.0, 10.0),
    (10.0, 3200.0, 14.0),
    (15.0, 3200.0, 14.0),
    (20.0, 3200.0, 14.0),
    (30.0, 3200.0, 14.0),
  )
  return VehicleProfile(
    vehicle_identity="HYUNDAI_PALISADE",
    revision=0,
    provenance="closed-loop provisional horizon regression",
    nodes=tuple(
      ProfileNode(
        speed_mps=speed_mps,
        parameters=PhysicalParameters(
          torque_per_lateral_accel=0.39298251926074423,
          rack_gain_deg_s2_per_torque=gain,
          rack_damping_per_s=damping,
          transport_delay_s=0.1,
          static_friction_torque=0.1301424652338028,
          kinetic_friction_torque=0.1301424652338028,
          rack_rate_resolution_deg_s=4.0,
          confidence=0.0,
          qualified=False,
        ),
        clean_support_s=0.0,
        sample_count=0,
        cross_fit_route_count=0,
        full_fit_candidate_rms=0.0,
      )
      for speed_mps, gain, damping in transient_nodes
    ),
  )


def controller() -> HorizonController:
  return HorizonController(
    fixed_dt_s=CONTROL_DT_SECONDS,
    limits=limits(),
    profile=profile(),
    tracking_policy=TrackingPolicy(10.0, 1.0),
    horizon_policy=HorizonPolicy.from_json_file(POLICY_PATH),
    nominal_mapping=mapping(),
  )


def cubic_angle(
  time_s: float,
  *,
  initial: float,
  final: float,
  start_s: float,
  duration_s: float,
) -> float:
  return cubic_motion(
    time_s,
    initial=initial,
    final=final,
    start_s=start_s,
    duration_s=duration_s,
  )[0]


def cubic_motion(
  time_s: float,
  *,
  initial: float,
  final: float,
  start_s: float,
  duration_s: float,
) -> tuple[float, float, float]:
  fraction = (time_s - start_s) / duration_s
  if fraction < 0.0:
    return initial, 0.0, 0.0
  if fraction >= 1.0:
    return final, 0.0, 0.0
  difference = final - initial
  smoothstep = 3.0 * fraction * fraction - 2.0 * fraction * fraction * fraction
  rate = difference * (6.0 * fraction - 6.0 * fraction * fraction) / duration_s
  acceleration = difference * (6.0 - 12.0 * fraction) / (duration_s * duration_s)
  return initial + difference * smoothstep, rate, acceleration


def motion_from_angles(
  angles_deg: Sequence[float],
) -> tuple[list[float], list[float]]:
  rates = [0.0] * HORIZON_SAMPLE_COUNT
  accelerations = [0.0] * HORIZON_SAMPLE_COUNT
  for index in range(1, HORIZON_SAMPLE_COUNT - 1):
    rates[index] = (angles_deg[index + 1] - angles_deg[index - 1]) / (2.0 * CONTROL_DT_SECONDS)
  rates[0] = (angles_deg[1] - angles_deg[0]) / CONTROL_DT_SECONDS
  rates[-1] = (angles_deg[-1] - angles_deg[-2]) / CONTROL_DT_SECONDS
  for index in range(1, HORIZON_SAMPLE_COUNT - 1):
    accelerations[index] = (rates[index + 1] - rates[index - 1]) / (2.0 * CONTROL_DT_SECONDS)
  accelerations[0] = (rates[1] - rates[0]) / CONTROL_DT_SECONDS
  accelerations[-1] = (rates[-1] - rates[-2]) / CONTROL_DT_SECONDS
  return rates, accelerations


def reference(
  plan: AnglePlan,
  command_time_s: float,
  speed_mps: float,
  *,
  zero_curvature: bool,
  motion_plan: MotionPlan | None = None,
  speed_plan: ScalarPlan | None = None,
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
  if motion_plan is None:
    angles = [
      plan(
        command_time_s + index * CONTROL_DT_SECONDS,
        command_time_s,
      )
      for index in range(HORIZON_SAMPLE_COUNT)
    ]
    rates, accelerations = motion_from_angles(angles)
  else:
    motion = [
      motion_plan(
        command_time_s + index * CONTROL_DT_SECONDS,
        command_time_s,
      )
      for index in range(HORIZON_SAMPLE_COUNT)
    ]
    angles = [sample[0] for sample in motion]
    rates = [sample[1] for sample in motion]
    accelerations = [sample[2] for sample in motion]
  speeds = [speed_mps if speed_plan is None else speed_plan(command_time_s + index * CONTROL_DT_SECONDS) for index in range(HORIZON_SAMPLE_COUNT)]
  nominal_mapping = mapping()
  curvatures = (
    [0.0] * HORIZON_SAMPLE_COUNT
    if zero_curvature
    else [curvature_from_measured_angle(angle_deg, speed, nominal_mapping, nominal_mapping).curvature for angle_deg, speed in zip(angles, speeds, strict=True)]
  )
  return curvatures, angles, rates, accelerations, speeds


def simulate(
  plan: AnglePlan,
  *,
  duration_s: float,
  speed_mps: float,
  initial_state: RackState | None = None,
  previous_counts: int = 0,
  driver_plan: ScalarPlan = lambda _: 0.0,
  pressed_plan: PressedPlan = lambda _: False,
  zero_curvature: bool = False,
  motion_plan: MotionPlan | None = None,
  speed_plan: ScalarPlan | None = None,
) -> list[ClosedLoopFrame]:
  horizon = controller()
  reactive = controller()
  horizon_state = initial_state or RackState(0.0, 0.0, 0.0)
  reactive_state = horizon_state
  horizon_previous = previous_counts
  reactive_previous = previous_counts
  nominal_mapping = mapping()
  frames: list[ClosedLoopFrame] = []

  for frame_index in range(round(duration_s / CONTROL_DT_SECONDS)):
    time_s = frame_index * CONTROL_DT_SECONDS
    curvatures, angles, rates, accelerations, speeds = reference(
      plan,
      time_s,
      speed_mps,
      zero_curvature=zero_curvature,
      motion_plan=motion_plan,
      speed_plan=speed_plan,
    )
    driver_torque = driver_plan(time_s)
    steering_pressed = pressed_plan(time_s)
    common = {
      "desired_curvatures": curvatures,
      "desired_angles_deg": angles,
      "desired_rates_deg_s": rates,
      "desired_accelerations_deg_s2": accelerations,
      "planned_speeds_mps": speeds,
      "driver_torque": driver_torque,
      "steering_pressed": steering_pressed,
      "lateral_active": True,
      "steering_request_fault_avoidance_counter": 0,
      "steering_request_state_valid": True,
      "live_mapping": nominal_mapping,
      "lateral_accel_offset_mps2": 0.0,
      "disturbance_torque": 0.0,
    }
    horizon_result = horizon.update(
      initial_state=horizon_state,
      previous_applied_counts=horizon_previous,
      current_steering_angle_deg=horizon_state.angle_deg,
      **common,
    )
    reactive_result = reactive.update(
      initial_state=reactive_state,
      previous_applied_counts=reactive_previous,
      current_steering_angle_deg=reactive_state.angle_deg,
      **common,
    )
    assert horizon_result.valid
    assert reactive_result.valid

    horizon_counts = horizon_result.planned_counts
    reactive_counts = reactive_result.reactive_counts
    runtime_limits = limits()
    assert (
      apply_torque_envelope_counts(
        runtime_limits,
        horizon_counts,
        horizon_previous,
        driver_torque,
      )
      == horizon_counts
    )
    assert (
      apply_torque_envelope_counts(
        runtime_limits,
        reactive_counts,
        reactive_previous,
        driver_torque,
      )
      == reactive_counts
    )

    current_speed_mps = speeds[0]
    parameters = profile().parameters_at(current_speed_mps).parameters
    horizon_state = step_plant(
      horizon_state,
      horizon_counts / STEER_MAX,
      current_speed_mps,
      nominal_mapping,
      nominal_mapping,
      0.0,
      parameters,
      0.0,
      CONTROL_DT_SECONDS,
    ).state
    reactive_state = step_plant(
      reactive_state,
      reactive_counts / STEER_MAX,
      current_speed_mps,
      nominal_mapping,
      nominal_mapping,
      0.0,
      parameters,
      0.0,
      CONTROL_DT_SECONDS,
    ).state
    frames.append(
      ClosedLoopFrame(
        time_s=time_s,
        speed_mps=current_speed_mps,
        desired_angle_deg=angles[0],
        horizon_angle_deg=horizon_state.angle_deg,
        reactive_angle_deg=reactive_state.angle_deg,
        horizon_rate_deg_s=horizon_state.rate_deg_s,
        reactive_rate_deg_s=reactive_state.rate_deg_s,
        horizon_counts=horizon_counts,
        reactive_counts=reactive_counts,
        horizon_previous_counts=horizon_previous,
        reactive_previous_counts=reactive_previous,
        horizon_raw_torque=horizon_result.raw_torque,
        reactive_raw_torque=reactive_result.raw_torque,
        preparation_active=horizon_result.preparation_active,
        preparation_scheduled=horizon_result.preparation_scheduled,
      )
    )
    horizon_previous = horizon_counts
    reactive_previous = reactive_counts

  return frames


def rmse(frames: Sequence[ClosedLoopFrame], angle_field: str) -> float:
  return math.sqrt(sum((float(getattr(frame, angle_field)) - frame.desired_angle_deg) ** 2 for frame in frames) / len(frames))


def integrated_absolute_error(
  frames: Sequence[ClosedLoopFrame],
  angle_field: str,
) -> float:
  return sum(abs(float(getattr(frame, angle_field)) - frame.desired_angle_deg) * CONTROL_DT_SECONDS for frame in frames)


def peak_error(frames: Sequence[ClosedLoopFrame], angle_field: str) -> float:
  return max(abs(float(getattr(frame, angle_field)) - frame.desired_angle_deg) for frame in frames)


def first_below(
  frames: Sequence[ClosedLoopFrame],
  angle_field: str,
  threshold_deg: float,
) -> float:
  return next(frame.time_s for frame in frames if float(getattr(frame, angle_field)) <= threshold_deg)


def test_constrained_cubic_entry_is_not_worse_than_reactive() -> None:
  def plan(sample_time_s: float, _: float) -> float:
    return cubic_angle(
      sample_time_s,
      initial=0.0,
      final=8.0,
      start_s=0.5,
      duration_s=0.3,
    )

  frames = simulate(
    plan,
    duration_s=1.5,
    speed_mps=0.0,
    zero_curvature=True,
  )
  pre_onset = [frame for frame in frames if frame.time_s < 0.5]
  assert max(abs(frame.horizon_angle_deg) for frame in pre_onset) == 0.0
  assert max(abs(frame.reactive_angle_deg) for frame in pre_onset) == 0.0

  comparison = frames[50:150]
  horizon_rmse = rmse(comparison, "horizon_angle_deg")
  reactive_rmse = rmse(comparison, "reactive_angle_deg")
  horizon_peak = peak_error(comparison, "horizon_angle_deg")
  reactive_peak = peak_error(comparison, "reactive_angle_deg")
  message = " ".join(
    (
      f"preview regressed constrained entry: rmse={horizon_rmse}/{reactive_rmse},",
      f"peak={horizon_peak}/{reactive_peak},",
      f"k49={frames[49].horizon_counts}/{frames[49].reactive_counts}",
    )
  )
  assert horizon_rmse <= reactive_rmse and horizon_peak <= reactive_peak, message


def test_known_release_executes_preparation_and_is_not_worse() -> None:
  def motion(sample_time_s: float, _: float) -> tuple[float, float, float]:
    if sample_time_s < 0.2:
      return 0.0, 0.0, 0.0
    if sample_time_s < 0.5:
      return cubic_motion(
        sample_time_s,
        initial=0.0,
        final=8.0,
        start_s=0.2,
        duration_s=0.3,
      )
    if sample_time_s < 1.5:
      return 8.0, 0.0, 0.0
    return cubic_motion(
      sample_time_s,
      initial=8.0,
      final=0.0,
      start_s=1.5,
      duration_s=0.25,
    )

  def plan(sample_time_s: float, command_time_s: float) -> float:
    return motion(sample_time_s, command_time_s)[0]

  frames = simulate(
    plan,
    duration_s=3.0,
    speed_mps=10.0,
    motion_plan=motion,
  )
  held = [frame for frame in frames if 0.8 <= frame.time_s < 1.5]
  assert max(abs(frame.horizon_angle_deg - 8.0) for frame in held) <= TRACKING_TOLERANCE_DEG

  release = [frame for frame in frames if frame.time_s >= 1.5]
  horizon_rmse = rmse(release, "horizon_angle_deg")
  reactive_rmse = rmse(release, "reactive_angle_deg")
  horizon_iae = integrated_absolute_error(release, "horizon_angle_deg")
  reactive_iae = integrated_absolute_error(release, "reactive_angle_deg")
  desired_50 = first_below(release, "desired_angle_deg", 4.0)
  desired_90 = first_below(release, "desired_angle_deg", 0.8)
  horizon_lag_50 = first_below(release, "horizon_angle_deg", 4.0) - desired_50
  reactive_lag_50 = first_below(release, "reactive_angle_deg", 4.0) - desired_50
  horizon_lag_90 = first_below(release, "horizon_angle_deg", 0.8) - desired_90
  reactive_lag_90 = first_below(release, "reactive_angle_deg", 0.8) - desired_90
  preparation_executed = any(current.horizon_counts < previous.horizon_counts for previous, current in zip(held, held[1:], strict=False))
  message = " ".join(
    (
      "preview perpetually deferred release:",
      f"counts={held[0].horizon_counts}->{held[-1].horizon_counts},",
      f"rmse={horizon_rmse}/{reactive_rmse},",
      f"iae={horizon_iae}/{reactive_iae},",
      f"lag50={horizon_lag_50}/{reactive_lag_50},",
      f"lag90={horizon_lag_90}/{reactive_lag_90}",
    )
  )
  assert (
    preparation_executed
    and horizon_rmse <= reactive_rmse
    and horizon_iae <= reactive_iae
    and horizon_lag_50 <= reactive_lag_50
    and horizon_lag_90 <= reactive_lag_90
  ), message


def test_static_friction_band_cannot_deadlock_tracking_error() -> None:
  def plan(_: float, __: float) -> float:
    return 2.0

  frames = simulate(
    plan,
    duration_s=1.0,
    speed_mps=10.0,
    initial_state=RackState(2.145, 0.0, -31 / STEER_MAX),
    previous_counts=-31,
  )
  initial_error = 0.145
  horizon_terminal_error = abs(frames[-1].horizon_angle_deg - 2.0)
  reactive_terminal_error = abs(frames[-1].reactive_angle_deg - 2.0)
  progressed = min(abs(frame.horizon_angle_deg - 2.0) for frame in frames) < initial_error - 0.01
  message = " ".join(
    (
      "preview remained inside static friction:",
      f"counts={frames[0].horizon_counts}->{frames[-1].horizon_counts},",
      f"error={horizon_terminal_error}/{reactive_terminal_error}",
    )
  )
  assert progressed and horizon_terminal_error <= reactive_terminal_error + 0.01, message


def test_infeasible_sharp_entry_uses_safe_best_effort() -> None:
  def plan(sample_time_s: float, _: float) -> float:
    return cubic_angle(
      sample_time_s,
      initial=0.0,
      final=12.0,
      start_s=0.8,
      duration_s=0.2,
    )

  frames = simulate(plan, duration_s=2.0, speed_mps=10.0)
  preparation = [frame for frame in frames if 0.6 <= frame.time_s < 0.8]
  assert max(abs(frame.horizon_angle_deg) for frame in preparation) <= TRACKING_TOLERANCE_DEG
  horizon_preload = max(frame.horizon_counts for frame in preparation)
  reactive_preload = max(frame.reactive_counts for frame in preparation)
  authored_motion = [frame for frame in frames if frame.time_s >= 0.8]
  horizon_iae = integrated_absolute_error(
    authored_motion,
    "horizon_angle_deg",
  )
  reactive_iae = integrated_absolute_error(
    authored_motion,
    "reactive_angle_deg",
  )
  moving = [frame for frame in frames if 0.8 <= frame.time_s < 1.0]
  settled = [frame for frame in frames if frame.time_s >= 1.0]
  maximum_signed_lag = max(frame.reactive_angle_deg - frame.horizon_angle_deg for frame in moving)
  maximum_settled_error_excess = max(
    abs(frame.horizon_angle_deg - frame.desired_angle_deg) - abs(frame.reactive_angle_deg - frame.desired_angle_deg) for frame in settled
  )
  message = " ".join(
    (
      "infeasible future did not produce safe best effort:",
      f"preload={horizon_preload}/{reactive_preload},",
      f"moving_lag={maximum_signed_lag},",
      f"settled_error_excess={maximum_settled_error_excess},",
      f"iae={horizon_iae}/{reactive_iae}",
    )
  )
  assert (
    horizon_preload > reactive_preload
    and maximum_signed_lag <= TRACKING_TOLERANCE_DEG
    and maximum_settled_error_excess <= TRACKING_TOLERANCE_DEG
    and horizon_iae <= reactive_iae
  ), message


def test_saturated_entry_uses_exact_maximum_reachable_authority() -> None:
  def plan(sample_time_s: float, _: float) -> float:
    return cubic_angle(
      sample_time_s,
      initial=0.0,
      final=40.0,
      start_s=0.5,
      duration_s=0.2,
    )

  frames = simulate(plan, duration_s=2.0, speed_mps=20.0)
  horizon_saturated = 0
  reactive_saturated = 0
  runtime_limits = limits()
  for frame in frames:
    if abs(frame.horizon_raw_torque) >= 1.0:
      target = int(math.copysign(STEER_MAX, frame.horizon_raw_torque))
      assert frame.horizon_counts == apply_torque_envelope_counts(
        runtime_limits,
        target,
        frame.horizon_previous_counts,
        0.0,
      )
      horizon_saturated += 1
    if abs(frame.reactive_raw_torque) >= 1.0:
      target = int(math.copysign(STEER_MAX, frame.reactive_raw_torque))
      assert frame.reactive_counts == apply_torque_envelope_counts(
        runtime_limits,
        target,
        frame.reactive_previous_counts,
        0.0,
      )
      reactive_saturated += 1
  assert horizon_saturated > 20
  assert reactive_saturated > 20


def test_malformed_live_mapping_fails_closed() -> None:
  def plan(_: float, __: float) -> float:
    return 0.0

  curvatures, angles, rates, accelerations, speeds = reference(
    plan,
    0.0,
    10.0,
    zero_curvature=False,
  )
  result = controller().update(
    desired_curvatures=curvatures,
    desired_angles_deg=angles,
    desired_rates_deg_s=rates,
    desired_accelerations_deg_s2=accelerations,
    planned_speeds_mps=speeds,
    initial_state=RackState(0.0, 0.0, 0.0),
    previous_applied_counts=0,
    driver_torque=0.0,
    steering_pressed=False,
    lateral_active=True,
    current_steering_angle_deg=0.0,
    steering_request_fault_avoidance_counter=0,
    steering_request_state_valid=True,
    live_mapping=object(),
    lateral_accel_offset_mps2=0.0,
    disturbance_torque=0.0,
  )
  assert not result.valid
  assert result.status == HorizonStatus.INVALID_INPUT
  assert result.planned_counts == 0


def test_reversal_does_not_bank_cumulative_path_lead() -> None:
  def motion(sample_time_s: float, _: float) -> tuple[float, float, float]:
    if sample_time_s < 0.2:
      return 0.0, 0.0, 0.0
    if sample_time_s < 0.5:
      return cubic_motion(
        sample_time_s,
        initial=0.0,
        final=8.0,
        start_s=0.2,
        duration_s=0.3,
      )
    if sample_time_s < 1.5:
      return 8.0, 0.0, 0.0
    return cubic_motion(
      sample_time_s,
      initial=8.0,
      final=-8.0,
      start_s=1.5,
      duration_s=0.4,
    )

  def plan(sample_time_s: float, command_time_s: float) -> float:
    return motion(sample_time_s, command_time_s)[0]

  frames = simulate(
    plan,
    duration_s=3.0,
    speed_mps=20.0,
    motion_plan=motion,
  )
  hold = [frame for frame in frames if 1.2 <= frame.time_s < 1.5]
  reversal = [frame for frame in frames if frame.time_s >= 1.5]
  parameters = profile().parameters_at(20.0).parameters
  no_lead_tolerance = parameters.rack_rate_resolution_deg_s * HorizonPolicy.from_json_file(POLICY_PATH).no_lead_position_tolerance_s
  maximum_early_unwind = max(8.0 - frame.horizon_angle_deg for frame in hold)
  horizon_iae = integrated_absolute_error(reversal, "horizon_angle_deg")
  reactive_iae = integrated_absolute_error(reversal, "reactive_angle_deg")
  message = " ".join(
    (
      f"future reversal banked path lead: lead={maximum_early_unwind}/{no_lead_tolerance},",
      f"iae={horizon_iae}/{reactive_iae}",
    )
  )
  assert maximum_early_unwind <= no_lead_tolerance and horizon_iae <= reactive_iae, message


def test_20hz_plan_cancellation_has_no_early_path_motion() -> None:
  def plan(sample_time_s: float, command_time_s: float) -> float:
    model_time_s = (
      math.floor(
        (command_time_s + 1e-9) / 0.05,
      )
      * 0.05
    )
    if model_time_s < 0.3:
      return cubic_angle(
        sample_time_s,
        initial=0.0,
        final=8.0,
        start_s=0.8,
        duration_s=0.25,
      )
    if model_time_s < 0.55:
      return 0.0
    return cubic_angle(
      sample_time_s,
      initial=0.0,
      final=-6.0,
      start_s=1.0,
      duration_s=0.3,
    )

  frames = simulate(plan, duration_s=2.0, speed_mps=10.0)
  before_authored_motion = [frame for frame in frames if frame.time_s < 1.0]
  assert max(abs(frame.horizon_angle_deg) for frame in before_authored_motion) <= TRACKING_TOLERANCE_DEG
  assert max(abs(frame.reactive_angle_deg) for frame in before_authored_motion) <= TRACKING_TOLERANCE_DEG


def test_speed_changes_preserve_entry_release_and_reversal_tracking() -> None:
  def motion(sample_time_s: float, _: float) -> tuple[float, float, float]:
    phases = (
      (0.3, 0.4, 0.0, 6.0),
      (1.3, 0.3, 6.0, 0.0),
      (1.9, 0.4, 0.0, -6.0),
      (2.7, 0.4, -6.0, 5.0),
    )
    previous = 0.0
    for start_s, duration_s, initial, final in phases:
      if sample_time_s < start_s:
        return previous, 0.0, 0.0
      if sample_time_s < start_s + duration_s:
        return cubic_motion(
          sample_time_s,
          initial=initial,
          final=final,
          start_s=start_s,
          duration_s=duration_s,
        )
      previous = final
    return previous, 0.0, 0.0

  def plan(sample_time_s: float, command_time_s: float) -> float:
    return motion(sample_time_s, command_time_s)[0]

  def speed(sample_time_s: float) -> float:
    phases = (
      (0.2, 0.6, 5.0, 18.0),
      (1.2, 0.5, 18.0, 8.0),
      (1.8, 0.7, 8.0, 24.0),
      (2.6, 0.6, 24.0, 12.0),
    )
    previous = 5.0
    for start_s, duration_s, initial, final in phases:
      if sample_time_s < start_s:
        return previous
      if sample_time_s < start_s + duration_s:
        return cubic_angle(
          sample_time_s,
          initial=initial,
          final=final,
          start_s=start_s,
          duration_s=duration_s,
        )
      previous = final
    return previous

  frames = simulate(
    plan,
    duration_s=3.5,
    speed_mps=5.0,
    motion_plan=motion,
    speed_plan=speed,
  )
  before_entry = [frame for frame in frames if frame.time_s < 0.3]
  assert max(abs(frame.horizon_angle_deg) for frame in before_entry) <= TRACKING_TOLERANCE_DEG
  assert min(frame.speed_mps for frame in frames) == 5.0
  assert max(frame.speed_mps for frame in frames) == 24.0

  episodes = (
    (0.3, 1.3, "entry and hold"),
    (1.3, 1.9, "release"),
    (1.9, 2.7, "negative entry and hold"),
    (2.7, 3.5, "reversal"),
  )
  for start_s, end_s, name in episodes:
    episode = [frame for frame in frames if start_s <= frame.time_s < end_s]
    horizon_iae = integrated_absolute_error(episode, "horizon_angle_deg")
    reactive_iae = integrated_absolute_error(episode, "reactive_angle_deg")
    assert horizon_iae <= reactive_iae + TRACKING_TOLERANCE_DEG * (end_s - start_s), f"dynamic-speed {name} regressed: {horizon_iae}/{reactive_iae}"

  assert rmse(frames, "horizon_angle_deg") <= rmse(frames, "reactive_angle_deg")

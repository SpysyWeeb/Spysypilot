from dataclasses import dataclass

import numpy as np

from openpilot.common.constants import CV
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants


HORIZON_SECONDS = 1.5
TRAJECTORY_DT = DT_MDL
TRAJECTORY_T = np.arange(0.0, HORIZON_SECONDS + TRAJECTORY_DT / 2.0, TRAJECTORY_DT)
TRAJECTORY_SIZE = len(TRAJECTORY_T)

# The model path starts at camera capture time. modeld currently samples the
# scalar action one frame plus half an action interval into the future.
MODEL_PATH_TIME_OFFSET = 1.5 * DT_MDL

MIN_MODEL_SPEED = 1.0
MAX_SANITY_CURVATURE = 0.2
MAX_SANITY_LATERAL_ACCEL = 6.0
LOW_SPEED_REFERENCE_SPEED = 12.0 * CV.MPH_TO_MS
FULL_SPEED_REFERENCE_SPEED = 15.0 * CV.MPH_TO_MS
FULL_SPEED_COST_SPEED = 15.0
FULL_LOW_SPEED_CURVATURE = 0.015
ZERO_LOW_SPEED_CURVATURE = 0.04

# Keep the proven low-speed curvature tuning unchanged through 12 mph. Above
# that, smoothly hand off to physical lateral jerk and jerk-rate costs through
# the lateral_acceleration = v^2 * curvature relationship.
LOW_SPEED_CURVATURE_RATE_WEIGHT = 0.1
LOW_SPEED_CURVATURE_ACCEL_WEIGHT = 0.001
LATERAL_JERK_WEIGHT = 1e-5
LATERAL_JERK_RATE_WEIGHT = 1e-13
LOW_SPEED_PREVIOUS_CURVATURE_WEIGHT = 0.1
PREVIOUS_LATERAL_ACCEL_WEIGHT = 3e-5
INITIAL_LATERAL_ACCEL_WEIGHT = 1e-4
INITIAL_CURVATURE_WEIGHT = 0.01
HIGH_SPEED_YAW_WEIGHT_DECAY = 0.8

# The preview can lead the legacy scalar reference, but only by this much in
# lateral-acceleration space on any one replan. This keeps path timing separate
# from steering authority: the preview may prepare for a release, but it cannot
# erase the turn that the model requested.
MAX_PREVIEW_LATERAL_ACCEL_DELTA = 0.20
ACTUATOR_PREVIEW_GAIN = 0.5
ACTUATOR_PREVIEW_NOMINAL_SLEW = 0.20
ACTUATOR_PREVIEW_MAX_EXTRA = 0.35
PATH_PHASE_LOOKAHEAD = 0.20
UNWIND_JERK_START = 0.10
UNWIND_JERK_FULL = 0.60
AUTHORITY_RESTORE_START = 0.03
AUTHORITY_RESTORE_FULL = 0.10


def _difference_matrix(order: int) -> np.ndarray:
  matrix = np.eye(TRAJECTORY_SIZE)
  for _ in range(order):
    matrix = np.diff(matrix, axis=0)
  return matrix


CURVATURE_RATE_MATRIX = _difference_matrix(1)
CURVATURE_ACCEL_MATRIX = _difference_matrix(2)
IDENTITY_MATRIX = np.eye(TRAJECTORY_SIZE)
MODEL_T_IDXS = np.asarray(ModelConstants.T_IDXS)


@dataclass(frozen=True)
class ActuatorPreviewConfig:
  max_torque: float
  delta_up: float
  delta_down: float
  steer_step: int = 1


@dataclass
class LateralReferenceDiagnostics:
  version: int = 2
  base_curvature: float = 0.0
  output_curvature: float = 0.0
  sample_time: float = 0.0
  extra_time: float = 0.0
  target_torque: float = 0.0
  applied_torque: float = 0.0
  unwind_scale: float = 0.0
  authority_restored: float = 0.0
  preview_correction: float = 0.0


class LateralReferencePlanner:
  """Build a continuous curvature reference from the model's yaw horizon."""

  def __init__(self, dt: float = DT_CTRL):
    self.dt = dt
    self.solution: np.ndarray | None = None
    self.predicted_speeds: np.ndarray | None = None
    self.solution_age = 0.0
    self.actuator_config: ActuatorPreviewConfig | None = None
    self.diagnostics = LateralReferenceDiagnostics()

  def configure_actuator(self, config: ActuatorPreviewConfig | None) -> None:
    self.actuator_config = config

  def reset(self) -> None:
    self.solution = None
    self.predicted_speeds = None
    self.solution_age = 0.0
    self.diagnostics = LateralReferenceDiagnostics()

  @staticmethod
  def _smoothstep(value: float, start: float, end: float) -> float:
    fraction = np.clip((value - start) / (end - start), 0.0, 1.0)
    return float(fraction * fraction * (3.0 - 2.0 * fraction))

  @staticmethod
  def _read_model_trajectory(model_msg, v_ego: float) -> tuple[np.ndarray, np.ndarray] | None:
    yaws = np.asarray(model_msg.orientation.z, dtype=float)
    speeds = np.asarray(model_msg.velocity.x, dtype=float)
    if len(yaws) != len(MODEL_T_IDXS) or len(speeds) != len(MODEL_T_IDXS):
      return None
    if not np.all(np.isfinite(yaws)) or not np.all(np.isfinite(speeds)):
      return None

    yaws = np.interp(TRAJECTORY_T, MODEL_T_IDXS, yaws)
    yaws -= yaws[0]

    # Preserve the model's predicted speed changes while anchoring its current
    # speed to carState. This keeps yaw-to-curvature conversion time-aligned.
    speeds = np.interp(TRAJECTORY_T, MODEL_T_IDXS, speeds)
    speeds += v_ego - speeds[0]
    speeds = np.maximum(speeds, MIN_MODEL_SPEED)
    return yaws, speeds

  def update(self, model_msg, current_curvature: float, v_ego: float) -> bool:
    if not np.isfinite(current_curvature):
      self.reset()
      return False

    trajectory = self._read_model_trajectory(model_msg, v_ego)
    if trajectory is None:
      self.reset()
      return False
    yaws, speeds = trajectory

    # yaw[i] = sum(speed[j] * curvature[j] * dt), j < i
    dynamics = np.zeros((TRAJECTORY_SIZE, TRAJECTORY_SIZE))
    for i in range(1, TRAJECTORY_SIZE):
      dynamics[i, :i] = speeds[:i] * TRAJECTORY_DT

    high_speed_cost_scale = self._smoothstep(v_ego, LOW_SPEED_REFERENCE_SPEED, FULL_SPEED_COST_SPEED)
    low_speed_cost_scale = 1.0 - high_speed_cost_scale
    yaw_weight_decay = 1.0 + high_speed_cost_scale * (HIGH_SPEED_YAW_WEIGHT_DECAY - 1.0)
    yaw_weights = np.exp(-TRAJECTORY_T / yaw_weight_decay)
    weighted_dynamics = yaw_weights[:, None] * dynamics
    weighted_yaws = yaw_weights * yaws
    lhs = weighted_dynamics.T @ weighted_dynamics
    rhs = weighted_dynamics.T @ weighted_yaws

    lhs += low_speed_cost_scale * LOW_SPEED_CURVATURE_RATE_WEIGHT * (CURVATURE_RATE_MATRIX.T @ CURVATURE_RATE_MATRIX)
    lhs += low_speed_cost_scale * LOW_SPEED_CURVATURE_ACCEL_WEIGHT * (CURVATURE_ACCEL_MATRIX.T @ CURVATURE_ACCEL_MATRIX)

    lateral_accel_matrix = np.diag(speeds**2)
    lateral_jerk_matrix = (CURVATURE_RATE_MATRIX @ lateral_accel_matrix) / TRAJECTORY_DT
    lateral_jerk_rate_matrix = np.diff(lateral_jerk_matrix, axis=0) / TRAJECTORY_DT
    lhs += high_speed_cost_scale * LATERAL_JERK_WEIGHT * (lateral_jerk_matrix.T @ lateral_jerk_matrix)
    lhs += high_speed_cost_scale * LATERAL_JERK_RATE_WEIGHT * (lateral_jerk_rate_matrix.T @ lateral_jerk_rate_matrix)

    if self.solution is not None:
      shifted_solution = np.interp(TRAJECTORY_T + self.solution_age, TRAJECTORY_T, self.solution, left=self.solution[0], right=self.solution[-1])
      lhs += low_speed_cost_scale * LOW_SPEED_PREVIOUS_CURVATURE_WEIGHT * IDENTITY_MATRIX
      rhs += low_speed_cost_scale * LOW_SPEED_PREVIOUS_CURVATURE_WEIGHT * shifted_solution
      lateral_accel_cost = lateral_accel_matrix.T @ lateral_accel_matrix
      lhs += high_speed_cost_scale * PREVIOUS_LATERAL_ACCEL_WEIGHT * lateral_accel_cost
      rhs += high_speed_cost_scale * PREVIOUS_LATERAL_ACCEL_WEIGHT * lateral_accel_cost @ shifted_solution

    initial_lateral_accel_scale = speeds[0] ** 2
    initial_weight = INITIAL_CURVATURE_WEIGHT * low_speed_cost_scale + high_speed_cost_scale * INITIAL_LATERAL_ACCEL_WEIGHT * initial_lateral_accel_scale**2
    lhs[0, 0] += initial_weight
    rhs[0] += initial_weight * current_curvature

    try:
      solution = np.linalg.solve(lhs + 1e-9 * IDENTITY_MATRIX, rhs)
    except np.linalg.LinAlgError:
      self.reset()
      return False

    solution_lateral_accel = speeds**2 * solution
    if (
      not np.all(np.isfinite(solution)) or np.max(np.abs(solution)) > MAX_SANITY_CURVATURE or np.max(np.abs(solution_lateral_accel)) > MAX_SANITY_LATERAL_ACCEL
    ):
      self.reset()
      return False

    self.solution = solution
    self.predicted_speeds = speeds
    self.solution_age = 0.0
    return True

  def _reference_scale(self, raw_curvature: float, v_ego: float) -> float:
    high_speed_reference_scale = self._smoothstep(v_ego, LOW_SPEED_REFERENCE_SPEED, FULL_SPEED_REFERENCE_SPEED)
    low_speed_curvature_scale = np.clip(
      (ZERO_LOW_SPEED_CURVATURE - abs(raw_curvature)) / (ZERO_LOW_SPEED_CURVATURE - FULL_LOW_SPEED_CURVATURE), 0.0, 1.0
    )
    return float(low_speed_curvature_scale + high_speed_reference_scale * (1.0 - low_speed_curvature_scale))

  def _predicted_speed(self, sample_time: float, fallback: float) -> float:
    if self.predicted_speeds is None:
      return max(fallback, MIN_MODEL_SPEED)
    return float(np.interp(sample_time, TRAJECTORY_T, self.predicted_speeds))

  def _target_torque(self, curvature: float, sample_time: float, v_ego: float, lat_accel_factor: float, friction: float) -> float:
    speed = self._predicted_speed(sample_time, v_ego)
    lateral_accel = curvature * speed**2
    friction_torque = friction * np.sign(lateral_accel) if abs(lateral_accel) > 0.02 else 0.0
    return float(np.clip(-(lateral_accel / max(lat_accel_factor, 1e-3) + friction_torque), -1.0, 1.0))

  def _transition_time(self, current: float, target: float) -> float:
    assert self.actuator_config is not None
    frames_per_second = 1.0 / (self.dt * max(self.actuator_config.steer_step, 1))
    rate_up = self.actuator_config.delta_up * frames_per_second / self.actuator_config.max_torque
    rate_down = self.actuator_config.delta_down * frames_per_second / self.actuator_config.max_torque
    if current * target >= 0.0:
      rate = rate_up if abs(target) > abs(current) else rate_down
      return abs(target - current) / max(rate, 1e-3)
    return abs(current) / max(rate_down, 1e-3) + abs(target) / max(rate_up, 1e-3)

  def _unwind_scale(self, sample_time: float) -> float:
    assert self.solution is not None
    phase_time = min(sample_time + PATH_PHASE_LOOKAHEAD, TRAJECTORY_T[-1])
    if phase_time <= sample_time:
      return 0.0
    curvature = float(np.interp(sample_time, TRAJECTORY_T, self.solution))
    future_curvature = float(np.interp(phase_time, TRAJECTORY_T, self.solution))
    # Classify the geometry, not the driver's throttle/brake input. Holding the
    # same curvature while speed changes is not an unwind. Predicted speed is
    # still used below for actuator demand and timing.
    speed = self._predicted_speed(sample_time, 0.0)
    unwind_jerk = (abs(curvature) - abs(future_curvature)) * speed**2 / (phase_time - sample_time)
    return self._smoothstep(unwind_jerk, UNWIND_JERK_START, UNWIND_JERK_FULL)

  def get_curvature(self, raw_curvature: float, v_ego: float, lateral_delay: float, applied_torque: float = 0.0,
                    lat_accel_factor: float = 1.0, friction: float = 0.0) -> float:
    actuator_preview = self.actuator_config is not None
    if not actuator_preview and v_ego <= LOW_SPEED_REFERENCE_SPEED and abs(raw_curvature) >= ZERO_LOW_SPEED_CURVATURE:
      self.reset()
      return raw_curvature
    if self.solution is None:
      return raw_curvature

    base_time = lateral_delay + MODEL_PATH_TIME_OFFSET + self.solution_age
    base_curvature = float(np.interp(base_time, TRAJECTORY_T, self.solution))
    reference_scale = self._reference_scale(raw_curvature, v_ego)
    legacy_reference = reference_scale * base_curvature + (1.0 - reference_scale) * raw_curvature

    sample_time = base_time
    target_torque = self._target_torque(base_curvature, base_time, v_ego, lat_accel_factor, friction)
    if actuator_preview:
      # Only lead a predicted release. During turn-in, looking farther ahead can
      # select the later unwind and weaken the torque build that is needed now.
      base_unwind_scale = self._unwind_scale(base_time)
      for _ in range(3):
        planned_curvature = float(np.interp(sample_time, TRAJECTORY_T, self.solution))
        target_torque = self._target_torque(planned_curvature, sample_time, v_ego, lat_accel_factor, friction)
        slew_time = self._transition_time(applied_torque, target_torque)
        extra_time = np.clip(ACTUATOR_PREVIEW_GAIN * max(slew_time - ACTUATOR_PREVIEW_NOMINAL_SLEW, 0.0),
                             0.0, ACTUATOR_PREVIEW_MAX_EXTRA)
        sample_time = base_time + base_unwind_scale * extra_time

    planned_curvature = float(np.interp(sample_time, TRAJECTORY_T, self.solution))
    unwind_scale = self._unwind_scale(sample_time) if actuator_preview else 0.0
    authority_restored = 0.0
    authority_reference = planned_curvature
    if actuator_preview and raw_curvature * planned_curvature > 0.0 and abs(planned_curvature) < abs(raw_curvature):
      raw_lateral_accel = abs(raw_curvature) * v_ego**2
      authority_scale = (1.0 - unwind_scale) * self._smoothstep(
        raw_lateral_accel, AUTHORITY_RESTORE_START, AUTHORITY_RESTORE_FULL
      )
      authority_reference = planned_curvature + authority_scale * (raw_curvature - planned_curvature)
      authority_restored = (authority_reference - planned_curvature) * v_ego**2
      planned_curvature = authority_reference

    if actuator_preview:
      max_curvature_delta = MAX_PREVIEW_LATERAL_ACCEL_DELTA / max(v_ego**2, 1.0)
      planned_curvature = float(np.clip(planned_curvature, legacy_reference - max_curvature_delta,
                                        legacy_reference + max_curvature_delta))
      # Meaningful turn-in/hold authority is deliberately exempt from the
      # preview cap. Tiny straight-road actions retain the smoothed reference.
      if abs(authority_reference) > abs(planned_curvature):
        planned_curvature = authority_reference

    self.solution_age += self.dt

    output_curvature = planned_curvature if actuator_preview else legacy_reference
    target_torque = self._target_torque(output_curvature, sample_time, v_ego, lat_accel_factor, friction)
    self.diagnostics = LateralReferenceDiagnostics(
      base_curvature=float(base_curvature),
      output_curvature=float(output_curvature),
      sample_time=float(sample_time),
      extra_time=float(sample_time - base_time),
      target_torque=float(target_torque),
      applied_torque=float(applied_torque),
      unwind_scale=float(unwind_scale),
      authority_restored=float(authority_restored),
      preview_correction=float((output_curvature - legacy_reference) * v_ego**2),
    )
    return float(output_curvature)

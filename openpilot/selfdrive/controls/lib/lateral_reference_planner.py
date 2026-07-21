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


def _difference_matrix(order: int) -> np.ndarray:
  matrix = np.eye(TRAJECTORY_SIZE)
  for _ in range(order):
    matrix = np.diff(matrix, axis=0)
  return matrix


CURVATURE_RATE_MATRIX = _difference_matrix(1)
CURVATURE_ACCEL_MATRIX = _difference_matrix(2)
IDENTITY_MATRIX = np.eye(TRAJECTORY_SIZE)
MODEL_T_IDXS = np.asarray(ModelConstants.T_IDXS)


class LateralReferencePlanner:
  """Build a continuous curvature reference from the model's yaw horizon."""

  def __init__(self, dt: float = DT_CTRL):
    self.dt = dt
    self.solution: np.ndarray | None = None
    self.solution_age = 0.0

  def reset(self) -> None:
    self.solution = None
    self.solution_age = 0.0

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
    self.solution_age = 0.0
    return True

  def get_curvature(self, raw_curvature: float, v_ego: float, lateral_delay: float) -> float:
    if self.solution is None:
      return raw_curvature

    sample_time = lateral_delay + MODEL_PATH_TIME_OFFSET + self.solution_age
    planned_curvature = float(np.interp(sample_time, TRAJECTORY_T, self.solution))
    self.solution_age += self.dt

    high_speed_reference_scale = self._smoothstep(v_ego, LOW_SPEED_REFERENCE_SPEED, FULL_SPEED_REFERENCE_SPEED)
    # Retain the low-speed sharp-turn escape path, then remove it smoothly so
    # the trajectory reference is fully active from 15 mph upward.
    low_speed_curvature_scale = np.clip(
      (ZERO_LOW_SPEED_CURVATURE - abs(raw_curvature)) / (ZERO_LOW_SPEED_CURVATURE - FULL_LOW_SPEED_CURVATURE), 0.0, 1.0
    )
    reference_scale = low_speed_curvature_scale + high_speed_reference_scale * (1.0 - low_speed_curvature_scale)
    return float(reference_scale * planned_curvature + (1.0 - reference_scale) * raw_curvature)

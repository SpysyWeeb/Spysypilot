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

FULL_PLANNER_SPEED = 12.0 * CV.MPH_TO_MS
PLANNER_RESET_SPEED = 15.0 * CV.MPH_TO_MS
FULL_PLANNER_CURVATURE = 0.015
PLANNER_RESET_CURVATURE = 0.04
MIN_MODEL_SPEED = 1.0
MAX_SANITY_CURVATURE = 0.2

# These weights were selected on a full-rlog low-speed route, then checked on
# the complete route. They regularize the planned trajectory, not the torque
# controller or measured steering rate.
CURVATURE_RATE_WEIGHT = 0.1
CURVATURE_ACCEL_WEIGHT = 0.001
PREVIOUS_TRAJECTORY_WEIGHT = 0.1
INITIAL_CURVATURE_WEIGHT = 0.01
YAW_WEIGHT_DECAY_SECONDS = 1.0


def _difference_matrix(order: int) -> np.ndarray:
  matrix = np.eye(TRAJECTORY_SIZE)
  for _ in range(order):
    matrix = np.diff(matrix, axis=0)
  return matrix


CURVATURE_RATE_MATRIX = _difference_matrix(1)
CURVATURE_ACCEL_MATRIX = _difference_matrix(2)
IDENTITY_MATRIX = np.eye(TRAJECTORY_SIZE)
MODEL_T_IDXS = np.asarray(ModelConstants.T_IDXS)
YAW_WEIGHTS = np.exp(-TRAJECTORY_T / YAW_WEIGHT_DECAY_SECONDS)


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
    if v_ego >= PLANNER_RESET_SPEED or not np.isfinite(current_curvature):
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

    weighted_dynamics = YAW_WEIGHTS[:, None] * dynamics
    weighted_yaws = YAW_WEIGHTS * yaws
    lhs = weighted_dynamics.T @ weighted_dynamics
    rhs = weighted_dynamics.T @ weighted_yaws

    lhs += CURVATURE_RATE_WEIGHT * (CURVATURE_RATE_MATRIX.T @ CURVATURE_RATE_MATRIX)
    lhs += CURVATURE_ACCEL_WEIGHT * (CURVATURE_ACCEL_MATRIX.T @ CURVATURE_ACCEL_MATRIX)

    if self.solution is not None:
      shifted_solution = np.interp(TRAJECTORY_T + self.solution_age, TRAJECTORY_T, self.solution, left=self.solution[0], right=self.solution[-1])
      lhs += PREVIOUS_TRAJECTORY_WEIGHT * IDENTITY_MATRIX
      rhs += PREVIOUS_TRAJECTORY_WEIGHT * shifted_solution

    lhs[0, 0] += INITIAL_CURVATURE_WEIGHT
    rhs[0] += INITIAL_CURVATURE_WEIGHT * current_curvature

    try:
      solution = np.linalg.solve(lhs + 1e-9 * IDENTITY_MATRIX, rhs)
    except np.linalg.LinAlgError:
      self.reset()
      return False

    if not np.all(np.isfinite(solution)) or np.max(np.abs(solution)) > MAX_SANITY_CURVATURE:
      self.reset()
      return False

    self.solution = solution
    self.solution_age = 0.0
    return True

  def get_curvature(self, raw_curvature: float, v_ego: float, lateral_delay: float) -> float:
    if abs(raw_curvature) >= PLANNER_RESET_CURVATURE:
      self.reset()
      return raw_curvature
    if self.solution is None or v_ego >= PLANNER_RESET_SPEED:
      return raw_curvature

    sample_time = lateral_delay + MODEL_PATH_TIME_OFFSET + self.solution_age
    planned_curvature = float(np.interp(sample_time, TRAJECTORY_T, self.solution))
    self.solution_age += self.dt

    speed_blend = np.clip((PLANNER_RESET_SPEED - v_ego) / (PLANNER_RESET_SPEED - FULL_PLANNER_SPEED), 0.0, 1.0)
    curvature_blend = np.clip((PLANNER_RESET_CURVATURE - abs(raw_curvature)) / (PLANNER_RESET_CURVATURE - FULL_PLANNER_CURVATURE), 0.0, 1.0)
    blend = speed_blend * curvature_blend
    return float(blend * planned_curvature + (1.0 - blend) * raw_curvature)

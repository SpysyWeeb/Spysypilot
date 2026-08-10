from collections import deque

import numpy as np

from openpilot.common.constants import CV
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX
from openpilot.selfdrive.modeld.constants import ModelConstants


# Model-path curvature observed at the three owner-driven calibration points.
# These differ slightly from vehicle curvature measured by deviceMotion, since the
# online limiter must be calibrated in the same model space it consumes.
CURVATURE_BP = np.array([0.00501, 0.04666, 0.08188])
CURVE_SPEED_V = np.array([50.0, 22.0, 13.0]) * CV.MPH_TO_MS

APPROACH_DECEL = 0.5  # m/s^2
MIN_CURVATURE = 1e-4
MIN_MODEL_SPEED = 1.0  # m/s; avoids unstable curvature near predicted stops
MAX_CURVE_SPEED = V_CRUISE_MAX * CV.KPH_TO_MS


def _median_filter_three(values):
  """Three-sample spatial median, including full windows at both edges."""
  filtered = np.empty_like(values)
  filtered[1:-1] = np.median(np.vstack((values[:-2], values[1:-1], values[2:])), axis=0)
  filtered[0] = np.median(values[:3])
  filtered[-1] = np.median(values[-3:])
  return filtered


def curve_speed_for_curvature(curvature):
  """Map absolute curvature to the field-calibrated maximum curve speed."""
  curvature = np.abs(np.asarray(curvature, dtype=float))
  speed = np.interp(curvature, CURVATURE_BP, CURVE_SPEED_V)

  # Continue beyond the field points at the lateral acceleration represented
  # by the nearest point. This remains monotonic while avoiding arbitrary
  # speed floors or ceilings outside the observed range.
  below = CURVE_SPEED_V[0] * np.sqrt(CURVATURE_BP[0] / np.maximum(curvature, MIN_CURVATURE))
  above = CURVE_SPEED_V[-1] * np.sqrt(CURVATURE_BP[-1] / np.maximum(curvature, MIN_CURVATURE))
  speed = np.where(curvature < CURVATURE_BP[0], below, speed)
  speed = np.where(curvature > CURVATURE_BP[-1], above, speed)
  speed = np.where(curvature < MIN_CURVATURE, MAX_CURVE_SPEED, speed)
  return np.minimum(speed, MAX_CURVE_SPEED)


class ModelCurveSpeedLimiter:
  """Caps cruise speed using filtered curvature over the model path."""

  def __init__(self, approach_decel=APPROACH_DECEL):
    self.approach_decel = approach_decel
    self._target_history = deque([MAX_CURVE_SPEED] * 3, maxlen=3)
    self.active = False
    self.v_target = MAX_CURVE_SPEED
    self.curvature = 0.0
    self.distance = 0.0

  def update(self, model, v_cruise):
    self.active = False
    self.v_target = v_cruise
    self.curvature = 0.0
    self.distance = 0.0

    position_x = np.asarray(model.position.x, dtype=float)
    position_y = np.asarray(model.position.y, dtype=float)
    velocity_x = np.asarray(model.velocity.x, dtype=float)
    yaw_rate = np.asarray(model.orientationRate.z, dtype=float)

    expected_shape = (ModelConstants.IDX_N,)
    if any(a.shape != expected_shape for a in (position_x, position_y, velocity_x, yaw_rate)):
      self._target_history = deque([MAX_CURVE_SPEED] * 3, maxlen=3)
      return v_cruise
    if not all(np.all(np.isfinite(a)) for a in (position_x, position_y, velocity_x, yaw_rate)):
      self._target_history = deque([MAX_CURVE_SPEED] * 3, maxlen=3)
      return v_cruise

    path_distance = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(position_x), np.diff(position_y)))))
    curvature = np.abs(yaw_rate) / np.maximum(np.abs(velocity_x), MIN_MODEL_SPEED)
    filtered_curvature = _median_filter_three(curvature)
    curve_speed = curve_speed_for_curvature(filtered_curvature)

    # Kinematics rearranged from v_curve^2 = v_now^2 + 2*a*distance.
    # Each horizon point provides a maximum speed allowed now; the strictest
    # point makes braking begin only when needed to reach its curve speed.
    allowed_now = np.sqrt(np.maximum(curve_speed ** 2 + 2.0 * self.approach_decel * path_distance, 0.0))
    target_idx = int(np.argmin(allowed_now))
    self._target_history.append(float(allowed_now[target_idx]))
    filtered_target = float(np.median(self._target_history))
    target = min(v_cruise, filtered_target)

    self.active = target < v_cruise
    self.v_target = target
    self.curvature = float(filtered_curvature[target_idx])
    self.distance = float(path_distance[target_idx])
    return target

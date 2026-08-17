from collections import deque

import numpy as np

from opendbc.car.hyundai.values import CAR
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY, CV
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX
from openpilot.selfdrive.modeld.constants import ModelConstants


# Model-path curvature observed at the three owner-driven calibration points.
# These differ slightly from vehicle curvature measured by deviceMotion, since the
# online limiter must be calibrated in the same model space it consumes.
CURVATURE_BP = np.array([0.00501, 0.04666, 0.08188])
CURVE_SPEED_V = np.array([50.0, 22.0, 13.0]) * CV.MPH_TO_MS

APPROACH_DECEL = 0.5  # m/s^2
CURVE_TARGET_RELEASE_RATE = 0.2  # m/s^2; lower targets apply immediately, opening curves release gently
TORQUE_BUDGET = 0.93
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


def _torque_values(params):
  try:
    values = (float(params.latAccelFactorFiltered), float(params.latAccelOffsetFiltered),
              float(params.frictionCoefficientFiltered))
  except (AttributeError, TypeError, ValueError, OverflowError):
    try:
      values = (float(params.latAccelFactor), float(params.latAccelOffset), float(params.friction))
    except (AttributeError, TypeError, ValueError, OverflowError):
      return None
  return values if np.all(np.isfinite(values)) and values[0] > 0.0 and 0.0 <= values[2] < TORQUE_BUDGET else None


class ModelCurveSpeedLimiter:
  """Caps cruise speed using filtered curvature over the model path."""

  def __init__(self, CP=None, approach_decel=APPROACH_DECEL):
    self.approach_decel = approach_decel
    self.torque_params = None
    lateral_tuning = getattr(CP, "lateralTuning", None)
    if (getattr(CP, "carFingerprint", None) == CAR.HYUNDAI_PALISADE and lateral_tuning is not None
        and lateral_tuning.which() == "torque"):
      self.torque_params = _torque_values(lateral_tuning.torque)
    self._target_history = deque([MAX_CURVE_SPEED] * 3, maxlen=3)
    self._torque_veto_history = deque([False] * 3, maxlen=3)
    self.active = False
    self.torque_veto = False
    self.predicted_torque = 0.0
    self.v_target = MAX_CURVE_SPEED
    self.curvature = 0.0
    self.distance = 0.0

  def _set_target(self, v_cruise, filtered_target):
    target = min(v_cruise, filtered_target)
    if target < v_cruise:
      target = min(target, self.v_target + CURVE_TARGET_RELEASE_RATE * DT_MDL)
    self.active = target < v_cruise
    self.v_target = target
    return target

  def _invalid(self, v_cruise):
    self._target_history.append(MAX_CURVE_SPEED)
    self._torque_veto_history.append(False)
    self.torque_veto = sum(self._torque_veto_history) >= 2
    return self._set_target(v_cruise, float(np.median(self._target_history)))

  def update(self, model, v_cruise, v_ego=0.0, lateral_active=False, roll=0.0, torque_params=None):
    self.active = False
    self.curvature = 0.0
    self.distance = 0.0
    self.predicted_torque = 0.0

    try:
      position_x = np.asarray(model.position.x, dtype=float)
      position_y = np.asarray(model.position.y, dtype=float)
      velocity_x = np.asarray(model.velocity.x, dtype=float)
      yaw_rate = np.asarray(model.orientationRate.z, dtype=float)
    except (AttributeError, TypeError, ValueError, OverflowError):
      return self._invalid(v_cruise)

    expected_shape = (ModelConstants.IDX_N,)
    if any(a.shape != expected_shape for a in (position_x, position_y, velocity_x, yaw_rate)):
      return self._invalid(v_cruise)
    if not all(np.all(np.isfinite(a)) for a in (position_x, position_y, velocity_x, yaw_rate)):
      return self._invalid(v_cruise)

    path_distance = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(position_x), np.diff(position_y)))))
    curvature = np.abs(yaw_rate) / np.maximum(np.abs(velocity_x), MIN_MODEL_SPEED)
    filtered_curvature = _median_filter_three(curvature)
    curve_speed = curve_speed_for_curvature(filtered_curvature)

    active_torque_params = _torque_values(torque_params) if self.torque_params is not None and torque_params is not None else None
    active_torque_params = active_torque_params or self.torque_params
    if lateral_active and active_torque_params is not None and np.isfinite(v_ego) and np.isfinite(roll):
      factor, offset, friction = active_torque_params
      signed_curvature = _median_filter_three(yaw_rate / np.maximum(np.abs(velocity_x), MIN_MODEL_SPEED))
      bias = roll * ACCELERATION_DUE_TO_GRAVITY + offset
      torque_margin = (TORQUE_BUDGET - friction) * factor
      torque_speed_squared = np.divide(bias + np.sign(signed_curvature) * torque_margin, signed_curvature,
                                       out=np.full_like(signed_curvature, np.inf),
                                       where=np.abs(signed_curvature) >= MIN_CURVATURE)
      curve_speed = np.minimum(curve_speed, np.sqrt(np.maximum(torque_speed_squared, 0.0)))
      self.predicted_torque = float(np.max(np.abs(signed_curvature * max(v_ego, 0.0) ** 2 - bias) / factor + friction))
    self._torque_veto_history.append(self.predicted_torque >= TORQUE_BUDGET)
    self.torque_veto = sum(self._torque_veto_history) >= 2

    # Kinematics rearranged from v_curve^2 = v_now^2 + 2*a*distance.
    # Each horizon point provides a maximum speed allowed now; the strictest
    # point makes braking begin only when needed to reach its curve speed.
    allowed_now = np.sqrt(np.maximum(curve_speed ** 2 + 2.0 * self.approach_decel * path_distance, 0.0))
    target_idx = int(np.argmin(allowed_now))
    self._target_history.append(float(allowed_now[target_idx]))
    filtered_target = float(np.median(self._target_history))

    self.curvature = float(filtered_curvature[target_idx])
    self.distance = float(path_distance[target_idx])
    return self._set_target(v_cruise, filtered_target)

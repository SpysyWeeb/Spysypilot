from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY


def clip_scalar(value: float, lower: float, upper: float) -> float:
  return min(max(float(value), lower), upper)


def neutral_torque(roll: float, lat_accel_offset: float, lat_accel_factor: float) -> float:
  return clip_scalar(
    (roll * ACCELERATION_DUE_TO_GRAVITY + lat_accel_offset) / max(lat_accel_factor, 1e-3),
    -1.0,
    1.0,
  )


def torque_transition_time(current: float, target: float, torque_build_rate: float, torque_decay_rate: float) -> float:
  """Return time for an asymmetric torque slew limiter to move between two torques."""
  build_rate = max(torque_build_rate, 1e-3)
  decay_rate = max(torque_decay_rate, 1e-3)
  if current * target >= 0.0:
    rate = build_rate if abs(target) > abs(current) else decay_rate
    return abs(target - current) / rate
  return abs(current) / decay_rate + abs(target) / build_rate

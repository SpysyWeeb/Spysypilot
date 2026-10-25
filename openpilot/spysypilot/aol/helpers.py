import math
from dataclasses import dataclass, fields

MAX_DRIVER_ASSIST_TORQUE = 0.5  # the envelope's FLOOR: an opposing driver never gets less than the old fixed cap
DRIVER_ASSIST_CEILING = 1.0     # the envelope's ceiling: the full actuator range, not a new limit


def is_hyundai_always_allow(CP) -> bool:
  """Hyundai with LDA button or CAN-FD (e.g. Palisade) can activate AOL
  without requiring cruise to be available first."""
  try:
    from opendbc.car.hyundai.values import HyundaiFlags
    return bool(CP.flags & (HyundaiFlags.HAS_LDA_BUTTON | HyundaiFlags.CANFD))
  except Exception:
    return CP.brand == 'hyundai'


@dataclass(frozen=True, slots=True)
class DriverAssistLimits:
  """The four platform CarControllerParams fields the driver-allowance term of
  opendbc.car.lateral.apply_driver_steer_torque_limits reads, as plain floats.
  Field names match opendbc's own so the formula reads exactly like it."""
  STEER_MAX: float
  STEER_DRIVER_ALLOWANCE: float
  STEER_DRIVER_MULTIPLIER: float
  STEER_DRIVER_FACTOR: float


def driver_assist_envelope(driver_torque_counts: float, commanded_torque: float, limits: DriverAssistLimits) -> float:
  """Driver-press cap on a normalized torque request, in [MAX_DRIVER_ASSIST_TORQUE, DRIVER_ASSIST_CEILING].

  Mirrors the driver-allowance term of opendbc.car.lateral.apply_driver_steer_torque_limits for the
  request's direction (BLaTv3 _driver_assist_envelope, failure catalog FM4.9): a driver pushing WITH
  the request keeps full authority, exactly as the platform limiter already allows; one opposing it
  shrinks the cap as the platform does, but never below the floor. The platform limiter itself still
  runs downstream. A driver torque that is not a finite number gets the floor. Pure, stateless."""
  if commanded_torque == 0.0:
    return DRIVER_ASSIST_CEILING
  if not math.isfinite(driver_torque_counts):
    return MAX_DRIVER_ASSIST_TORQUE
  direction = math.copysign(1.0, commanded_torque)
  d = driver_torque_counts * direction  # d > 0: the driver pushes with the request
  allowed = limits.STEER_MAX + (limits.STEER_DRIVER_ALLOWANCE + d * limits.STEER_DRIVER_FACTOR) * limits.STEER_DRIVER_MULTIPLIER
  allowed = max(min(limits.STEER_MAX, allowed), 0.0)  # the same clamp as the platform function
  return max(allowed / limits.STEER_MAX, MAX_DRIVER_ASSIST_TORQUE)


def driver_assist_limits_from_cp(CP) -> DriverAssistLimits | None:
  """The brand's own driver-allowance limits for this CarParams, or None when its CarControllerParams
  is missing, cannot be built from CP, or lacks any of the four fields (no driver-allowance limiter)."""
  try:
    from opendbc.car.interfaces import get_interface_attr
    params_class = get_interface_attr('CarControllerParams').get(CP.brand)
    if params_class is None:
      return None
    params = params_class(CP)  # constructor signatures differ by brand; a brand that can't take CP has no limits here
  except Exception:
    return None

  values = {}
  for field in fields(DriverAssistLimits):
    value = getattr(params, field.name, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
      return None
    values[field.name] = float(value)
  if values['STEER_MAX'] <= 0.0:
    return None
  return DriverAssistLimits(**values)

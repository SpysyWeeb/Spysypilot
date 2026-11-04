"""Calm steering: a low-pass on the torque controller's friction term, used only on a highway straight.

LatControlTorque adds a friction feedforward, get_friction(error + JERK_GAIN * desired_lateral_jerk): a ramp over
+/-0.2 m/s^2 of error onto +/-friction * latAccelFactor (0.331 m/s^2, 0.130 normalized torque on the Palisade). That
is a second proportional path about 2x the P gain at 30 m/s, fed by the measured steering angle, whose 0.1 deg
quantum is 0.0165 m/s^2 at 30 m/s (6.05 deg per m/s^2), and by the jerk look-ahead, which turns each 20 Hz model
step into a pulse. At highway speed the same angle noise is a v^2-larger torque swing, which the driver feels as
small corrections that jump too quickly.

Here the friction term's OUTPUT passes through a 0.1 s first-order filter, blended in by a gate that is 1.0 only when
all of these hold, each through a smoothstep so nothing steps:
  - speed: opens from 20 to 28 m/s (72 to 101 km/h);
  - near straight: closes as max(|desired|, |measured|) steering angle goes from 0 to 3 deg (0.50 m/s^2 at 30 m/s);
  - holding, not moving: closes as the faster of the desired angle's rate (0.1 s window) and the measured angle's
    rate (0.2 s window) goes from 5 to 8 deg/s (0.83 to 1.32 m/s^3 at 30 m/s); closed until both windows are full;
  - hands off and lateral active: exactly 0.0 while steeringPressed or inactive.
The gate closes at once and reopens at most dt / 0.25 s per frame, so it never steps the output either way. The
trajectory, roll and latAccelOffset feedforward terms, P, I and everything else are untouched. With the gate at 0.0
the friction value passes through unchanged, so the controller is bit-identical to stock there (town speeds, curves,
turn-ins, hands on). The filter and windows update every frame, active or not, so the filter is always warm.

Visible effects: controlsState.lateralControlState.torqueState.f (pid_log.f) is the whole feedforward, whose friction
summand is the blended one while the gate is open, and process-replay controlsd reference outputs change on
highway-straight frames. No param, toggle or schema change; the numbers come from the BLaTv3 highway feedforward
taper (field-checked on route 54).
"""
import math
from collections import deque

import numpy as np

from openpilot.common.filter_simple import FirstOrderFilter

CALM_RC_S = 0.1                 # friction filter time constant: a 0.0165 m/s^2 (0.1 deg) step at 30 m/s takes 0.1 s to 63 %
CALM_SPEED_MPS = 28.0           # fully open from 28 m/s (101 km/h), the owner's cruising speed, where 0.1 deg is 0.0153 m/s^2
CALM_SPEED_BLEND_MPS = 8.0      # opens from 20 m/s (72 km/h), where 0.1 deg is 0.0098 m/s^2
CALM_ANGLE_DEG = 3.0            # closed at 3 deg of steering angle: 0.50 m/s^2 at 30 m/s
CALM_RATE_DEG_S = 5.0           # starts closing at 5 deg/s of angle rate: 0.83 m/s^3 at 30 m/s
CALM_RATE_BLEND_DEG_S = 3.0     # closed at 8 deg/s: 1.32 m/s^3 at 30 m/s
CALM_DESIRED_WINDOW_S = 0.1     # desired angle rate over 0.1 s (11 samples): a 0.09 m/s^2 model step reads 5.4 deg/s
CALM_MEASURED_WINDOW_S = 0.2    # measured angle rate over 0.2 s (21 samples): the 0.1 deg quantum reads 0.5 deg/s (0.083 m/s^3)
CALM_OPEN_TIME_S = 0.25         # gate rise slew, 0 to 1 in 0.25 s; it closes at once


def smoothstep(x: float, lo: float, hi: float) -> float:
  t = float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))
  return t * t * (3.0 - 2.0 * t)


class CalmSteering:
  def __init__(self, dt: float):
    self.dt = dt
    self.filter = FirstOrderFilter(0.0, CALM_RC_S, dt, initialized=False)
    self.desired_angles: deque[float] = deque(maxlen=round(CALM_DESIRED_WINDOW_S / dt) + 1)
    self.measured_angles: deque[float] = deque(maxlen=round(CALM_MEASURED_WINDOW_S / dt) + 1)
    self.last_gate = 0.0

  @staticmethod
  def gate_from(v_ego: float, angle_deg: float, rate_deg_s: float, closed: bool) -> float:
    if closed or not all(math.isfinite(x) for x in (v_ego, angle_deg, rate_deg_s)):
      return 0.0
    return (smoothstep(v_ego, CALM_SPEED_MPS - CALM_SPEED_BLEND_MPS, CALM_SPEED_MPS)
            * (1.0 - smoothstep(abs(angle_deg), 0.0, CALM_ANGLE_DEG))
            * (1.0 - smoothstep(rate_deg_s, CALM_RATE_DEG_S, CALM_RATE_DEG_S + CALM_RATE_BLEND_DEG_S)))

  def windows_full(self) -> bool:
    return len(self.desired_angles) == self.desired_angles.maxlen and len(self.measured_angles) == self.measured_angles.maxlen

  def rate(self) -> float:
    # the faster of the two angle rates, from each window's ends rather than per-frame derivatives; NaN if either is
    rates = [abs(w[-1] - w[0]) / ((len(w) - 1) * self.dt) for w in (self.desired_angles, self.measured_angles)]
    return max(rates) if all(math.isfinite(r) for r in rates) else math.nan

  def gate(self, v_ego: float, desired_angle_deg: float, measured_angle_deg: float, closed: bool) -> float:
    # the angles enter only through abs() and rate magnitudes; the caller appends them to the windows first
    if not self.windows_full() or not (math.isfinite(desired_angle_deg) and math.isfinite(measured_angle_deg)):
      return 0.0
    angle = max(abs(desired_angle_deg), abs(measured_angle_deg))
    target = self.gate_from(v_ego, angle, self.rate(), closed)
    return min(target, self.last_gate + self.dt / CALM_OPEN_TIME_S)

  def apply(self, friction: float, gate: float) -> float:
    if gate == 0.0 or not self.filter.initialized:
      return friction
    blended = friction + gate * (self.filter.x - friction)
    return blended if math.isfinite(blended) else friction

  def update(self, friction: float, CS, VM, params, desired_curvature: float, active: bool) -> float:
    # desired and measured angles share CS.steeringAngleDeg's sign: for a consistent loop they are equal
    desired_angle_deg = math.degrees(VM.get_steer_from_curvature(-desired_curvature, CS.vEgo, params.roll))
    measured_angle_deg = CS.steeringAngleDeg - params.angleOffsetDeg
    self.desired_angles.append(desired_angle_deg)
    self.measured_angles.append(measured_angle_deg)
    if math.isfinite(friction):
      self.filter.update(friction)
    self.last_gate = self.gate(CS.vEgo, desired_angle_deg, measured_angle_deg, CS.steeringPressed or not active)
    return self.apply(friction, self.last_gate)

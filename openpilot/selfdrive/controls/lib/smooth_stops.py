"""
Original concept and implementation by SpysyWeeb (github.com/SpysyWeeb)

Smooth Stops. The smooth landing is produced entirely in longcontrol, at
control rate, by feathering the brake all the way down to a *true* standstill and
only then handing off to the standstill hold clamp. See SmoothStopController for
the rationale.
"""
from opendbc.car.interfaces import ACCEL_MIN
from openpilot.common.realtime import DT_CTRL

# --- handoff to the hold clamp ---
STANDSTILL_SPEED = 0.05        # m/s, arm the stopping/hold clamp once the car is essentially stopped
STANDSTILL_HOLD_SPEED = 0.15   # m/s, ceiling for trusting CS.standstill -- never arm the hold above this,
                               # so the car's own standstill signal can't clamp down while still rolling

# --- the settle feather: a controlled deceleration to a true stop ---
SETTLE_DECEL = 0.80       # m/s^2, decel while feathering down from the approach (>= TAPER_SPEED)
TAPER_SPEED = 1.0         # m/s (~2.2 mph), below this ease the brake off toward the stop (limo roll-to-stop)
STOP_KISS_DECEL = 0.25    # m/s^2, gentle residual decel right at the stop -> small step -> minimal end jerk
STOP_GAP_MARGIN = 3.0     # m, settle brakes to stop at least this far behind the lead (anti-creep-in)
MIN_GAP_BUDGET = 0.5      # m, lower bound on the gap budget; bounds required decel as the gap -> 0
PROGRESS_EPS = 0.02       # m/s, speed drop (vs the running minimum) that counts as "still slowing"
ANTI_CREEP_RATE = 0.50    # m/s^2 added per second the car is NOT slowing (kills creep / held creep)
SETTLE_JERK = 2.5         # m/s^3, smoothness of the brake command itself
EMERGENCY_DECEL = 3.0     # m/s^2, at/below -this the jerk limit is dropped (true-collision bypass)


class SmoothStopController:
  """
  Owns the final approach to a stop, inside longcontrol (control rate, true v_ego).

  The harsh "headbang" stop is the stock handoff: the moment the plan's speed drops
  below vEgoStopping, the state machine jumps to the `stopping` state and ramps the
  command to stopAccel (-2.0 m/s^2) -- while the car is usually still rolling. This
  controller instead keeps the car in a closed-loop *settle* feather that decelerates
  it smoothly to a true standstill, and only lets the hold clamp engage once stopped:

    settle()    -- command a gentle baseline decel; firm up toward the decel required to
                   stop behind the lead (lead-aware "how swiftly"); ramp firmer still if
                   the car stops making progress (anti-creep). Never brake softer than the
                   MPC's plan (collision safety) and jerk-limit the command so the brake
                   never steps -- except at/below EMERGENCY_DECEL, applied immediately.
    want_hold() -- arm the stopping/hold clamp only once v_ego is at/below STANDSTILL_SPEED
                   (or the car reports standstill), so the clamp lands on a stopped car.

  Knobs (SETTLE_DECEL gentleness, EMERGENCY_DECEL bailout) are conservative defaults to
  be tuned on-device.
  """
  def __init__(self):
    self.frame = 0
    self._v_min = float("inf")
    self._stall_s = 0.0

  def reset(self) -> None:
    self._v_min = float("inf")
    self._stall_s = 0.0

  def update(self) -> None:
    self.frame += 1

  def want_hold(self, should_stop: bool, v_ego: float, standstill: bool) -> bool:
    # Arm the stopping/hold clamp only once the car is actually stopped, so it never
    # clamps down while still rolling (the headbang). CS.standstill is only a backstop for
    # a noisy v_ego near zero -- it is NOT trusted to arm the hold while the car is still
    # moving (some cars, incl. the Palisade, assert standstill a hair early), hence the
    # STANDSTILL_HOLD_SPEED ceiling.
    return bool(should_stop and (v_ego <= STANDSTILL_SPEED or (standstill and v_ego <= STANDSTILL_HOLD_SPEED)))

  def settle(self, a_target: float, v_ego: float, lead_distance: float, has_lead: bool, last_output: float) -> float:
    # Ease the brake off as v -> 0 so deceleration nearly vanishes at the moment of stopping
    # -- the limo "roll to a stop" (release pressure before the wheels stop). A small kiss
    # decel remains so the car still reaches 0 against creep torque; the firm baseline
    # returns above TAPER_SPEED.
    landing = STOP_KISS_DECEL + (SETTLE_DECEL - STOP_KISS_DECEL) * min(v_ego / TAPER_SPEED, 1.0)
    a_settle = -landing

    # lead-aware: brake hard enough to stop short of the lead (how swiftly to execute)
    if has_lead and lead_distance > 0.0:
      gap = max(lead_distance - STOP_GAP_MARGIN, MIN_GAP_BUDGET)
      a_settle = min(a_settle, -(v_ego * v_ego) / (2.0 * gap))

    # anti-creep: while the car is NOT actually slowing, firm up -- relative to the command,
    # so the ease-off above is preserved whenever the car *is* still slowing (even slowly).
    if v_ego < self._v_min - PROGRESS_EPS:
      self._v_min = v_ego
      self._stall_s = 0.0
    else:
      self._stall_s += DT_CTRL
    a_settle -= ANTI_CREEP_RATE * self._stall_s

    a_settle = max(a_settle, ACCEL_MIN)

    # never brake softer than the MPC's plan -- it owns collision avoidance
    target = min(a_settle, a_target)

    # at/below the emergency decel this is a true-collision stop: apply it now, no feather
    if target <= -EMERGENCY_DECEL:
      return target

    # otherwise feather the command so the brake itself never steps (persistent smoothing)
    step = SETTLE_JERK * DT_CTRL
    return min(max(target, last_output - step), last_output + step)

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
SETTLE_JERK = 2.5         # m/s^3, smoothness of the settle pressure itself (plan braking is never feathered)
HOLD_RELEASE_FRAMES = 10  # frames of should_stop=False required to release the hold (kills one-frame
                          # plan flickers that blip the brake at standstill; costs 0.1s of launch latency)


class SmoothStopController:
  """
  Owns the final approach to a stop, inside longcontrol (control rate, true v_ego).

  The harsh "headbang" stop is the stock handoff: the moment the plan's speed drops
  below vEgoStopping, the state machine jumps to the `stopping` state and ramps the
  command to stopAccel (-2.0 m/s^2) -- while the car is usually still rolling. This
  controller instead keeps the car in a closed-loop *settle* feather that decelerates
  it smoothly to a true standstill, and only lets the hold clamp engage once stopped:

    settle()       -- command a gentle baseline decel; firm up toward the decel required to
                      stop behind the lead (lead-aware "how swiftly"); ratchet firmer if the
                      car stops making progress (anti-creep). Only the settle pressure is
                      feathered -- braking demanded by the MPC's plan (which owns collision
                      avoidance) is applied immediately, so full braking force is never delayed.
    want_hold()    -- arm the stopping/hold clamp only once v_ego is at/below STANDSTILL_SPEED
                      (or the car reports standstill), so the clamp lands on a stopped car.
    hold_release() -- once holding, require should_stop to stay false for HOLD_RELEASE_FRAMES
                      before releasing, so a one-frame plan flicker can't blip the brake.

  Knobs (SETTLE_DECEL gentleness, ANTI_CREEP_RATE) are conservative defaults to
  be tuned on-device.
  """
  def __init__(self):
    self._v_min = float("inf")
    self._stall_s = 0.0
    self._creep_decel = 0.0
    self._no_stop_frames = 0

  def reset(self) -> None:
    self._v_min = float("inf")
    self._stall_s = 0.0
    self._creep_decel = 0.0

  def want_hold(self, should_stop: bool, v_ego: float, standstill: bool) -> bool:
    # Arm the stopping/hold clamp only once the car is actually stopped, so it never
    # clamps down while still rolling (the headbang). CS.standstill is only a backstop for
    # a noisy v_ego near zero -- it is NOT trusted to arm the hold while the car is still
    # moving (some cars, incl. the Palisade, assert standstill a hair early), hence the
    # STANDSTILL_HOLD_SPEED ceiling.
    hold = bool(should_stop and (v_ego <= STANDSTILL_SPEED or (standstill and v_ego <= STANDSTILL_HOLD_SPEED)))
    if hold:
      self._no_stop_frames = 0  # arm the hold with a fresh release debounce
    return hold

  def hold_release(self, should_stop: bool) -> bool:
    # Debounce the hold exit: the plan's should_stop can flicker false for a frame while
    # stopped (seen with e2e in experimental mode), which would blip the state machine to
    # starting and release brake pressure at standstill. Only release once should_stop has
    # been false for HOLD_RELEASE_FRAMES straight. The counter is deliberately NOT cleared
    # by reset() -- reset() runs every frame while holding, which would defeat the debounce.
    if should_stop:
      self._no_stop_frames = 0
    else:
      self._no_stop_frames += 1
    return self._no_stop_frames >= HOLD_RELEASE_FRAMES

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

    # anti-creep: while the car is NOT actually slowing, ratchet the pressure firmer. The
    # firmness is latched (never released until reset()) -- resetting it on each sliver of
    # progress limit-cycles against creep torque: brake firms, car slows a hair, firmness
    # snaps back, creep pushes back, repeat (a shuddery ~10s crawl to standstill in sim).
    if v_ego < self._v_min - PROGRESS_EPS:
      self._v_min = v_ego
      self._stall_s = 0.0
    else:
      self._stall_s += DT_CTRL
    self._creep_decel = max(self._creep_decel, ANTI_CREEP_RATE * self._stall_s)
    a_settle -= self._creep_decel

    a_settle = max(a_settle, ACCEL_MIN)

    # feather only the settle pressure so the brake never steps; anchoring on last_output
    # also feathers the release after hard plan braking ends
    step = SETTLE_JERK * DT_CTRL
    a_settle = min(max(a_settle, last_output - step), last_output + step)

    # never brake softer than the MPC's plan -- it owns collision avoidance, so its demand
    # is applied immediately, unfeathered (full braking force is never delayed)
    return min(a_settle, a_target)

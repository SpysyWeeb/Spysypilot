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
STOP_KISS_DECEL = 0.12    # m/s^2, residual decel right at the stop; the anti-creep ratchet guarantees
                          # the stop still completes, so this can be truly gentle
MIN_ENTRY_SPEED = 0.1     # m/s, floor on the latched entry speed (guards the taper division)
STOP_GAP_MARGIN = 2.5     # m, settle brakes to stop at least this far behind the lead (anti-creep-in).
                          # 3.0 -> 2.5 after route 3c t=1343: a legitimately tight 2.9m roll-up put the
                          # budget near zero and the lead floor grabbed -1.2 at walking speed; contact
                          # safety re-verified by replay at 2.5 (floor keeps full strength inside ~2.7m)
MIN_GAP_BUDGET = 0.15     # m, floor on the gap budget -- only guards the division, must NOT meaningfully
                          # soften the lead-floor term close-in (ACCEL_MIN + jerk limiting downstream
                          # already bound the result; a looser floor here previously let this term go
                          # weak exactly where it matters most: close range at low speed)
PROGRESS_EPS = 0.02       # m/s, speed drop (vs the running minimum) that counts as "still slowing"
ANTI_CREEP_RATE = 0.50    # m/s^2 added per second the car is NOT slowing (kills creep / held creep)
LEAD_MOVING_ENTER = 0.30  # m/s, moving queue rather than vehicle creep
LEAD_MOVING_EXIT = 0.18   # m/s, hysteresis prevents threshold chatter
LEAD_DROPOUT_GRACE = 0.50 # s, retain moving-queue state through brief radar loss
CREEP_DECAY_RATE = 1.0    # m/s^3, release obsolete anti-creep pressure while the queue moves
URGENT_GAP = 2.0          # m, inside this gap budget the anti-creep ratchet escalates faster than the
                          # open-road rate -- a stall near a lead is a proximity problem, not just comfort
URGENT_RATE_MULT = 4.0    # ratchet-rate multiplier at gap -> 0, ramped in linearly from 1x at URGENT_GAP
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

    settle()       -- fade the braking level you *entered* with linearly down to the kiss as
                      v -> 0 (entry-anchored: no step at engage by construction, whatever the
                      plan was doing); firm up toward the decel required to stop behind the
                      lead (lead-aware "how swiftly"); ratchet firmer if the car stops making
                      progress (anti-creep), escalating faster the tighter the remaining gap
                      to a lead is -- a stall that's eating into a close gap is urgent, not
                      just uncomfortable. Only the settle pressure is feathered -- braking
                      demanded by the MPC's plan (which owns collision avoidance) is applied
                      immediately, so full braking force is never delayed.
    want_hold()    -- request the stopping/hold clamp only once v_ego is at/below
                      STANDSTILL_SPEED (or the car reports standstill), so the clamp lands
                      on a stopped car. Pure query, no state.
    arm_hold()     -- called by longcontrol on every transition into the stopping state
                      (including the stock off/starting edges that bypass want_hold), so
                      the hold always starts with a fresh release debounce.
    hold_release() -- once holding, require should_stop to stay false for HOLD_RELEASE_FRAMES
                      before releasing, so a one-frame plan flicker can't blip the brake.

  Knobs (STOP_KISS_DECEL end-gentleness, ANTI_CREEP_RATE) are conservative defaults to
  be tuned on-device.
  """
  def __init__(self):
    self._v_min = float("inf")
    self._stall_s = 0.0
    self._creep_decel = 0.0
    self._entry_v = 0.0
    self._entry_decel = 0.0
    self._no_stop_frames = 0
    self._lead_moving = False
    self._lead_dropout_s = 0.0

  def reset(self) -> None:
    self._v_min = float("inf")
    self._stall_s = 0.0
    self._creep_decel = 0.0
    self._entry_v = 0.0
    self._entry_decel = 0.0
    self._lead_moving = False
    self._lead_dropout_s = 0.0

  def want_hold(self, should_stop: bool, v_ego: float, standstill: bool) -> bool:
    # Request the stopping/hold clamp only once the car is actually stopped, so it never
    # clamps down while still rolling (the headbang). CS.standstill is only a backstop for
    # a noisy v_ego near zero -- it is NOT trusted to arm the hold while the car is still
    # moving (some cars, incl. the Palisade, assert standstill a hair early), hence the
    # STANDSTILL_HOLD_SPEED ceiling.
    return bool(should_stop and (v_ego <= STANDSTILL_SPEED or (standstill and v_ego <= STANDSTILL_HOLD_SPEED)))

  def arm_hold(self) -> None:
    # Fresh release debounce for a new hold. Longcontrol calls this on every transition
    # into the stopping state -- the stock edges (off/starting -> stopping) never go
    # through want_hold, and a counter left >= HOLD_RELEASE_FRAMES by the previous stop
    # would let a one-frame should_stop flicker release the new hold instantly.
    self._no_stop_frames = 0

  def hold_release(self, should_stop: bool, has_lead: bool = False, lead_speed: float = 0.0) -> bool:
    # Debounce the hold exit: the plan's should_stop can flicker false for a frame while
    # stopped (seen with e2e in experimental mode), which would blip the state machine to
    # starting and release brake pressure at standstill. Only release once should_stop has
    # been false for HOLD_RELEASE_FRAMES straight; a measured departing lead releases at
    # once. A stopped lead in radar view gets no longer wait: the plan owns departure
    # confirmation, and the car itself needs ~1.3 s to exit standstill after the release
    # (Palisade, field-measured 2026-08-29), so every extra frame here lands on launch latency.
    # The counter is deliberately NOT cleared by reset() -- reset() runs every frame
    # while holding, which would defeat the debounce.
    if not should_stop and has_lead and lead_speed >= LEAD_MOVING_ENTER:
      self._no_stop_frames = 0
      return True
    self._no_stop_frames = 0 if should_stop else self._no_stop_frames + 1
    return self._no_stop_frames >= HOLD_RELEASE_FRAMES

  def _update_lead_motion(self, has_lead: bool, lead_speed: float) -> bool:
    if has_lead:
      self._lead_dropout_s = 0.0
      if lead_speed >= LEAD_MOVING_ENTER:
        self._lead_moving = True
      elif lead_speed <= LEAD_MOVING_EXIT:
        self._lead_moving = False
    elif self._lead_moving:
      self._lead_dropout_s += DT_CTRL
      if self._lead_dropout_s > LEAD_DROPOUT_GRACE:
        self._lead_moving = False
    return self._lead_moving

  def settle(self, a_target: float, v_ego: float, lead_distance: float, has_lead: bool, last_output: float,
             lead_speed: float = 0.0) -> float:
    # Entry-anchored taper: latch the braking level the car entered settle with and fade it
    # linearly down to the kiss as v -> 0 -- the limo "roll to a stop" (release pressure
    # before the wheels stop). Anchoring on entry means no step at engage: a fixed baseline
    # grabbed the brake whenever the plan's own taper was gentler than it (field-observed).
    if self._entry_v <= 0.0:
      self._entry_v = max(v_ego, MIN_ENTRY_SPEED)
      self._entry_decel = max(-last_output, STOP_KISS_DECEL)
    landing = STOP_KISS_DECEL + (self._entry_decel - STOP_KISS_DECEL) * min(v_ego / self._entry_v, 1.0)
    a_settle = -landing

    # lead-aware: brake hard enough to stop short of the lead (how swiftly to execute). Also
    # sets how urgently anti-creep below should escalate -- a stall that eats into a tight gap
    # is a proximity problem, not just a comfort one, and shouldn't get the open-road grace period.
    creep_rate = ANTI_CREEP_RATE
    if has_lead and lead_distance > 0.0:
      gap = max(lead_distance - STOP_GAP_MARGIN, MIN_GAP_BUDGET)
      closing_speed = max(v_ego - lead_speed, 0.0)
      a_settle = min(a_settle, -(closing_speed * closing_speed) / (2.0 * gap))
      urgency = max(0.0, min(1.0, 1.0 - gap / URGENT_GAP))
      creep_rate = ANTI_CREEP_RATE * (1.0 + urgency * (URGENT_RATE_MULT - 1.0))

    # anti-creep: while the car is NOT actually slowing, ratchet the pressure firmer. The
    # firmness is latched until a moving queue makes it obsolete or settle resets.
    if self._update_lead_motion(has_lead, lead_speed):
      self._v_min = v_ego
      self._stall_s = max(self._stall_s - DT_CTRL, 0.0)
      self._creep_decel = max(self._creep_decel - CREEP_DECAY_RATE * DT_CTRL, 0.0)
    else:
      if v_ego < self._v_min - PROGRESS_EPS:
        self._v_min = v_ego
        self._stall_s = 0.0
      else:
        self._stall_s += DT_CTRL
      self._creep_decel = max(self._creep_decel, creep_rate * self._stall_s)
    a_settle -= self._creep_decel

    a_settle = max(a_settle, ACCEL_MIN)

    # feather only the settle pressure so the brake never steps; anchoring on last_output
    # also feathers the release after hard plan braking ends
    step = SETTLE_JERK * DT_CTRL
    a_settle = min(max(a_settle, last_output - step), last_output + step)

    # never brake softer than the MPC's plan -- it owns collision avoidance, so its demand
    # is applied immediately, unfeathered (full braking force is never delayed)
    return min(a_settle, a_target)

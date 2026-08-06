"""BLoTv2 limousine-stop profile and standstill handoff.

An ordinary planned stop is shaped from stop-intent entry through standstill:
braking ramps in, holds the planner's requested deceleration, and releases
progressively near zero speed. Collision/driver safety overrides remain able
to pass through stronger braking.
"""
from opendbc.car.interfaces import ACCEL_MIN
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.longitudinal_lead import (
  LeadObservation,
  closing_decel_requirement,
)

# Handoff to the stock standstill clamp.
STANDSTILL_SPEED = 0.05
STANDSTILL_HOLD_SPEED = 0.15
HOLD_RELEASE_FRAMES = 10

# Limousine profile.
STOP_KISS_DECEL = 0.12
LIMO_BRAKE_JERK = 1.5
LIMO_RELEASE_START_SPEED = 4.0
LIMO_RELEASE_END_SPEED = 0.15

# Keep the old name available to downstream tests/tools while making the
# profile's lower jerk explicit.
SETTLE_JERK = LIMO_BRAKE_JERK

# Relative-frame lead margin and anti-creep.
STOP_GAP_MARGIN = 2.5
MIN_GAP_BUDGET = 0.15
PROGRESS_EPS = 0.02
ANTI_CREEP_RATE = 0.50
URGENT_GAP = 2.0
URGENT_RATE_MULT = 4.0

# A moving queue is not vehicle creep. Hysteresis rejects threshold chatter,
# and a short radar dropout does not permanently ratchet in more brake.
LEAD_MOVING_ENTER = 0.30
LEAD_MOVING_EXIT = 0.18
LEAD_DROPOUT_GRACE = 0.50
CREEP_DECAY_RATE = 1.0


class SmoothStopController:
  """Shape ordinary stop braking and hand off cleanly to the stock hold state."""

  def __init__(self):
    self._no_stop_frames = 0
    self.reset()

  def reset(self) -> None:
    self._v_min = float("inf")
    self._stall_s = 0.0
    self._creep_decel = 0.0
    self._lead_moving = False
    self._lead_dropout_s = 0.0

  def want_hold(self, should_stop: bool, v_ego: float, standstill: bool) -> bool:
    return bool(should_stop and (
      v_ego <= STANDSTILL_SPEED or (standstill and v_ego <= STANDSTILL_HOLD_SPEED)
    ))

  def arm_hold(self) -> None:
    self._no_stop_frames = 0

  def hold_release(self, should_stop: bool) -> bool:
    if should_stop:
      self._no_stop_frames = 0
    else:
      self._no_stop_frames += 1
    return self._no_stop_frames >= HOLD_RELEASE_FRAMES

  def _update_lead_motion(self, lead: LeadObservation) -> bool:
    if lead.present:
      self._lead_dropout_s = 0.0
      if lead.speed >= LEAD_MOVING_ENTER:
        self._lead_moving = True
      elif lead.speed <= LEAD_MOVING_EXIT:
        self._lead_moving = False
    elif self._lead_moving:
      self._lead_dropout_s += DT_CTRL
      if self._lead_dropout_s > LEAD_DROPOUT_GRACE:
        self._lead_moving = False

    return self._lead_moving

  def settle(self, a_target: float, v_ego: float, last_output: float,
             lead: LeadObservation | None = None, emergency: bool = False) -> float:
    lead = lead if lead is not None else LeadObservation()

    # Ordinary braking follows the desired deceleration, then releases it
    # progressively over the final ~9 mph. The jerk limit applies both to
    # brake application and release, producing a smooth pressure hill rather
    # than a late step into the stock stopping clamp.
    requested_decel = max(-float(a_target), STOP_KISS_DECEL)
    if v_ego <= LIMO_RELEASE_END_SPEED:
      release_fraction = 0.0
    else:
      release_fraction = min(
        max((v_ego - LIMO_RELEASE_END_SPEED) /
            (LIMO_RELEASE_START_SPEED - LIMO_RELEASE_END_SPEED), 0.0),
        1.0,
      )
    a_profile = -(
      STOP_KISS_DECEL + (requested_decel - STOP_KISS_DECEL) * release_fraction
    )

    # Lead collision geometry is a safety floor, not a comfort request. It is
    # allowed to override the limo profile when relative closing requires it.
    creep_rate = ANTI_CREEP_RATE
    if lead.present:
      a_safety = -closing_decel_requirement(
        v_ego, lead, STOP_GAP_MARGIN, MIN_GAP_BUDGET,
      )
      a_profile = min(a_profile, a_safety)
      gap = max(lead.distance - STOP_GAP_MARGIN, MIN_GAP_BUDGET)
      urgency = min(max(1.0 - gap / URGENT_GAP, 0.0), 1.0)
      creep_rate *= 1.0 + urgency * (URGENT_RATE_MULT - 1.0)

    # A trusted moving lead means ego is queue-following, not stuck against
    # creep torque. Decay prior ratchet pressure smoothly and re-anchor progress
    # at the current queue speed. Brief radar loss retains this state.
    if self._update_lead_motion(lead):
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

    a_profile = max(a_profile - self._creep_decel, ACCEL_MIN)

    if emergency:
      # FCW, force-decel, or an equivalent collision/emergency signal owns the
      # command. The ordinary profile must never delay a safety intervention.
      return min(float(a_target), a_profile)

    step = LIMO_BRAKE_JERK * DT_CTRL
    return min(max(a_profile, last_output - step), last_output + step)

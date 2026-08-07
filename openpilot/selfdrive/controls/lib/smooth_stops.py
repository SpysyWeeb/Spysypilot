"""BLoTv2 final-approach and standstill-handoff controller.

Smooth Stops owns only the last low-speed landing. The planner remains the
collision-avoidance authority: any stronger planner braking passes through
immediately.
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

# Entry-anchored landing taper.
STOP_KISS_DECEL = 0.12
MIN_ENTRY_SPEED = 0.1
SETTLE_JERK = 2.5

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
  """Feather a rolling stop to zero speed before entering the stock hold state."""

  def __init__(self):
    self._no_stop_frames = 0
    self.reset()

  def reset(self) -> None:
    self._v_min = float("inf")
    self._stall_s = 0.0
    self._creep_decel = 0.0
    self._entry_v = 0.0
    self._entry_decel = 0.0
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
             lead: LeadObservation | None = None) -> float:
    lead = lead if lead is not None else LeadObservation()

    # Latch the pressure present at settle entry and release it continuously as
    # speed falls. This gives a no-step entry and the low residual "kiss."
    if self._entry_v <= 0.0:
      self._entry_v = max(v_ego, MIN_ENTRY_SPEED)
      self._entry_decel = max(-last_output, STOP_KISS_DECEL)
    landing = STOP_KISS_DECEL + (self._entry_decel - STOP_KISS_DECEL) * min(
      max(v_ego, 0.0) / self._entry_v, 1.0,
    )
    a_settle = -landing

    # Add only the braking required by relative closing motion. The old
    # absolute-ego-speed floor over-braked equal-speed creeping queues.
    creep_rate = ANTI_CREEP_RATE
    if lead.present:
      a_settle = min(a_settle, -closing_decel_requirement(
        v_ego, lead, STOP_GAP_MARGIN, MIN_GAP_BUDGET,
      ))
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

    a_settle = max(a_settle - self._creep_decel, ACCEL_MIN)

    # Jerk-limit only comfort pressure. Stronger planner braking is an immediate
    # pass-through so Smooth Stops cannot weaken collision avoidance.
    step = SETTLE_JERK * DT_CTRL
    a_settle = min(max(a_settle, last_output - step), last_output + step)
    return min(a_settle, a_target)

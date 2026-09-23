from collections import deque

import numpy as np

from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.radard import RADAR_TO_CAMERA

LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection

LANE_CHANGE_T_FOLLOW = 0.6   # s, time gap kept to the lead being left behind while the car moves over
RELAX_TIME_MAX = 2.5         # s, the model hands the lead over well within this (field p90 2.25 s)
MIN_HEADROOM = 0.5           # m/s, the set speed must be this far above the current speed
LEAD_BRAKING = -1.0          # m/s^2, a lead braking harder than this keeps its full gap
LEAD_SLOWING = 0.5           # m/s, a lead this much below its speed since the change started is braking
LEAD_SPEED_FRAMES = 3        # a radar speed glitch lasts one frame
LEAD_CONTINUITY = 2.0        # m, the same car across a track id change, or frame to frame for a vision lead
LEAD_SPEED_CONTINUITY = 1.5  # m/s
LANE_WIDTH_DEFAULT = 3.5     # m, target lane width when the model's lines are not confident
LANE_WIDTH_MIN = 2.7
LANE_WIDTH_MAX = 4.5
LINE_OFFSET_MIN = 1.0        # m, distance from the car to the line it will cross
LINE_OFFSET_MAX = 2.5
TRACK_MIN_DISTANCE = 2.0     # m, closer returns are beside the car, which the blind spot monitor covers
TRACK_MOVING_SPEED = 2.0     # m/s absolute, below this a return is clutter, a stopped car or oncoming traffic
ARRIVAL_TIME = 3.0           # s, a target lane car inside the follow gap by then blocks
BLOCKED_HOLD = 0.5           # s, radar returns flicker


class LaneLines:
  # the model's two lane lines in the radar's left positive frame, sampled at a distance ahead so a bend keeps
  # the target lane band on the lane
  def __init__(self, model):
    lane_lines, probs = model.laneLines, model.laneLineProbs
    valid = len(lane_lines) >= 3 and len(probs) >= 3 and len(lane_lines[1].x) > 0 and len(lane_lines[2].x) > 0
    self.left_valid = valid and probs[1] > 0.5
    self.right_valid = valid and probs[2] > 0.5
    self.x = (np.array(lane_lines[1].x), np.array(lane_lines[2].x)) if valid else None
    self.y = (-np.array(lane_lines[1].y), -np.array(lane_lines[2].y)) if valid else None

  def line(self, i, x):
    return float(np.interp(x, self.x[i], self.y[i]))

  def sweep(self, i, x):
    # a line's lateral drift out to x, from the target side's line or else the other one
    if self.left_valid and (i == 0 or not self.right_valid):
      return self.line(0, x) - self.line(0, 0.0)
    if self.right_valid:
      return self.line(1, x) - self.line(1, 0.0)
    return 0.0

  def band(self, direction, x):
    # the lane the car is moving into: from the line it crosses to one lane width beyond, shifted by the
    # lines' sweep out to x; the car's offset in its lane is bounded, the sweep of a bend is not
    left = self.line(0, 0.0) if self.left_valid else None
    right = self.line(1, 0.0) if self.right_valid else None
    width = float(np.clip(left - right, LANE_WIDTH_MIN, LANE_WIDTH_MAX)) if left is not None and right is not None else LANE_WIDTH_DEFAULT
    if direction == LaneChangeDirection.left:
      near = float(np.clip(left, LINE_OFFSET_MIN, LINE_OFFSET_MAX)) if left is not None else width / 2
      near += self.sweep(0, x)
      return near, near + width
    near = float(np.clip(-right, LINE_OFFSET_MIN, LINE_OFFSET_MAX)) if right is not None else width / 2
    near -= self.sweep(1, x)
    return -(near + width), -near


def target_lane_blocked(direction, lines, radar_tracks, v_ego, t_follow, stop_distance, comfort_brake):
  # a car the MPC would already be following once the lane change lands: the gap it keeps behind a car at that
  # speed (the MPC's comfort distance less the car's own stopping distance), judged where the car will be
  follow_gap = t_follow * v_ego + stop_distance
  for pt in radar_tracks.points:
    if pt.dRel < TRACK_MIN_DISTANCE:
      continue
    lo, hi = lines.band(direction, pt.dRel + RADAR_TO_CAMERA)
    if not lo <= pt.yRel <= hi:
      continue
    v_track = v_ego + pt.vRel
    if v_track < TRACK_MOVING_SPEED:
      # stationary returns are mostly clutter, so only one already inside the follow gap counts as a stopped car;
      # oncoming traffic is judged where it will be
      arrival = pt.dRel + pt.vRel * ARRIVAL_TIME if v_track < -TRACK_MOVING_SPEED else pt.dRel
      if arrival < follow_gap:
        return True
      continue
    gap = max(v_ego**2 - v_track**2, 0.0) / (2 * comfort_brake) + follow_gap
    if pt.dRel + pt.vRel * ARRIVAL_TIME < gap:
      return True
  return False


class LaneChangeGap:
  """Owns the follow gap while a lane change starts.

  The model keeps the lead in the lane being left for about 1.5 s after the change begins, so the
  planner brakes behind a car the driver is pulling out to pass. While the change is starting, that
  car is still the lead, the target lane is clear and the set speed is above the current speed, the
  MPC's time gap to it is relaxed to LANE_CHANGE_T_FOLLOW so the cruise acceleration takes over, and
  the MPC targets the relaxed distance. The relaxation ends for the rest of the change when the model
  hands the lead over, the lead brakes (its filtered deceleration, a drop in its raw speed or the
  model's own estimate), the driver backs out (blinker off, counter-steer, a gas or brake override)
  or after RELAX_TIME_MAX.
  """

  def __init__(self, CP, dt=DT_MDL):
    self.dt = dt
    # the check needs a view of the target lane: front radar tracks ahead, the blind spot monitor beside
    self.enabled = not CP.radarUnavailable and CP.enableBsm
    self.starting_prev = False
    self.backed_out = False
    self.lead_id = -1
    self.lead_distance = 0.0
    self.lead_v_rel = 0.0
    self.lead_v_max = 0.0
    self.lead_speeds = deque(maxlen=LEAD_SPEED_FRAMES)
    self.reset()

  def reset(self):
    # a planner reset mid change (a gas or brake override) ends the relaxation and the acceleration for that change
    self.armed = False
    self.backed_out = self.starting_prev
    self.relax_timer = 0.0
    self.blocked_timer = 0.0
    self.t_follow_pad = 0.0
    self.accelerate = False

  def remember(self, lead):
    self.lead_id = lead.radarTrackId if lead.radar else -1
    self.lead_distance = lead.dRel
    self.lead_v_rel = lead.vRel
    # the fastest the lead has read for LEAD_SPEED_FRAMES frames running, so one high glitch cannot raise it
    self.lead_v_max = max(self.lead_v_max, min(self.lead_speeds))

  def same_lead(self, lead):
    if not lead.present:
      return False
    if lead.radar and lead.radarTrackId == self.lead_id:
      return True
    # a track id can change on the same car, and a vision lead has none
    return abs(lead.dRel - (self.lead_distance + self.lead_v_rel * self.dt)) < LEAD_CONTINUITY and \
           abs(lead.vRel - self.lead_v_rel) < LEAD_SPEED_CONTINUITY

  def lead_braking(self, lead, model):
    # the radar's filtered deceleration lags a hard brake by about half a second; the raw speed and the
    # model's estimate of the lead's acceleration both read it earlier
    model_lead = model.leadsV3[0] if len(model.leadsV3) > 0 else None
    model_braking = model_lead is not None and model_lead.prob > 0.5 and len(model_lead.a) > 0 and model_lead.a[0] < LEAD_BRAKING
    slowing = len(self.lead_speeds) == LEAD_SPEED_FRAMES and max(self.lead_speeds) < self.lead_v_max - LEAD_SLOWING
    return lead.aLeadK < LEAD_BRAKING or slowing or model_braking

  @staticmethod
  def driver_backs_out(CS, direction):
    if direction == LaneChangeDirection.left:
      return not CS.leftBlinker or (CS.steeringPressed and CS.steeringTorque < 0)
    return not CS.rightBlinker or (CS.steeringPressed and CS.steeringTorque > 0)

  def target_lane_clear(self, direction, CS, model, radar_tracks, radar_ok, v_ego, t_follow, stop_distance, comfort_brake):
    blind_spot = CS.leftBlindspot if direction == LaneChangeDirection.left else CS.rightBlindspot
    if blind_spot or not radar_ok or \
       target_lane_blocked(direction, LaneLines(model), radar_tracks, v_ego, t_follow, stop_distance, comfort_brake):
      self.blocked_timer = BLOCKED_HOLD
    else:
      self.blocked_timer = max(self.blocked_timer - self.dt, 0.0)
    return self.blocked_timer <= 0.0

  def update(self, model, CS, radar_state, radar_tracks, radar_ok, v_ego, v_cruise, t_follow, stop_distance, comfort_brake):
    direction = model.meta.laneChangeDirection
    starting = model.meta.laneChangeState == LaneChangeState.laneChangeStarting and direction != LaneChangeDirection.none
    lead = radar_state.leadOne

    if starting and not self.starting_prev:
      self.armed = self.enabled and lead.present
      self.backed_out = False
      self.relax_timer = 0.0
      self.lead_speeds.clear()
      self.lead_speeds.append(lead.vLead)
      self.lead_v_max = lead.vLead
      self.remember(lead)
    elif starting and self.armed:
      self.relax_timer += self.dt
      self.lead_speeds.append(lead.vLead)
    if starting and self.armed:
      # the model handing the lead over or the time limit end the relaxation only; the driver backing out or the
      # lead braking end the acceleration for this change too
      if self.relax_timer > RELAX_TIME_MAX or not self.same_lead(lead):
        self.armed = False
      elif self.driver_backs_out(CS, direction) or self.lead_braking(lead, model):
        self.armed = False
        self.backed_out = True
      else:
        self.remember(lead)
    if not starting:
      self.armed = False
      self.backed_out = False
      self.blocked_timer = 0.0
    self.starting_prev = starting

    self.accelerate = False
    self.t_follow_pad = 0.0
    if starting:
      clear = self.target_lane_clear(direction, CS, model, radar_tracks, radar_ok, v_ego, t_follow, stop_distance, comfort_brake)
      self.accelerate = self.enabled and clear and not self.backed_out and v_cruise - v_ego > MIN_HEADROOM
      if self.accelerate and self.armed:
        self.t_follow_pad = min(LANE_CHANGE_T_FOLLOW - t_follow, 0.0)
    return self.t_follow_pad

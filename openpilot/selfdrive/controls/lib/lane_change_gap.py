from collections import deque

import numpy as np

from opendbc.car.chrysler.values import ChryslerFlags
from opendbc.car.ford.values import FordFlags
from opendbc.car.gm.values import GMFlags
from opendbc.car.honda.values import HondaFlags
from opendbc.car.hyundai.values import HyundaiFlags
from opendbc.car.subaru.values import SubaruFlags
from opendbc.car.toyota.values import ToyotaFlags
from opendbc.car.volkswagen.values import VolkswagenFlags
from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LEAD_DANGER_FACTOR
from openpilot.selfdrive.controls.radard import RADAR_TO_CAMERA

LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection

LANE_CHANGE_T_FOLLOW = 0.6   # s, time gap kept to the lead being left behind while the car moves over
RELAX_TIME_MAX = 2.5         # s, the model hands the lead over well within this (field p90 2.25 s)
MIN_HEADROOM = 0.5           # m/s, the set speed must be this far above the current speed
LEAD_BRAKING = -1.0          # m/s^2, a lead braking harder than this keeps its full gap
LEAD_SLOWING = 0.5           # m/s, a lead this much below its speed since the change started is braking
LEAD_SPEED_FRAMES = 3        # a radar speed glitch lasts one frame
CLOSE_FRAMES = 3             # a lead at the floor's edge, or another car in its slot for a frame, reads too close a frame or two
LEAD_CONTINUITY = 2.0        # m, the same car across a track id change, or frame to frame for a vision lead
LEAD_SPEED_CONTINUITY = 1.5  # m/s
LEAD_OFF_PATH = 1.0          # m, a lead this far off the planned path, toward the lane being left, is being passed
LEAD_ON_PATH = 0.5           # m, and the same lead back this close is in front again; a radar lateral steps ~0.6 m
LEAD_BESIDE = 2.0            # m, lateral toward the lane being left where the two cars' bodies stop overlapping
LEAD_UNPLACED_STD = 1.0      # m, the model's lateral std on a lead it cannot put in a lane
LEAD_UNPLACED_STD_HOLD = 0.8
RESTORE_DECEL = 1.0          # m/s^2, the steady braking a lead handed back may need to stop closing before the follow distance
RESTORE_LAG = 1.0            # s, the car keeps its acceleration this long before a lead handed back slows it
LANE_WIDTH_DEFAULT = 3.5     # m, target lane width when the model's lines are not confident
LANE_WIDTH_MIN = 2.7
LANE_WIDTH_MAX = 4.5
LINE_OFFSET_MIN = 1.0        # m, distance from the car to the line it will cross
LINE_OFFSET_MAX = 2.5
LINE_STD_MAX = 0.3           # m, a line placed this well marks the lane even while the model doubts it is the car's
LINE_HOLD = 1.0              # s, the model places no line for a moment while it relabels them
LANDED_MARGIN = 0.5          # m, the car's center past the line it crossed
LANDED_FRAMES = 3            # the model's relabel can land the car for a frame
ROADSIDE_MARGIN = 0.9        # m, a stopped car in the target lane shows at least half its width inside the outer line
TRACK_MIN_DISTANCE = 2.0     # m, closer returns are beside the car, which the blind spot monitor covers
TRACK_MOVING_SPEED = 2.0     # m/s absolute, below this a return is clutter, a stopped car or oncoming traffic
ARRIVAL_TIME = 3.0           # s, a target lane car inside the follow gap by then blocks
BLOCKED_HOLD = 0.5           # s, radar returns flicker

# opendbc marks a blind spot monitor with each brand's own flag (commaai/opendbc#3788)
BSM_FLAGS = {
  'chrysler': ChryslerFlags.HAS_BSM,
  'ford': FordFlags.HAS_BSM,
  'gm': GMFlags.HAS_BSM,
  'honda': HondaFlags.HAS_BSM,
  'hyundai': HyundaiFlags.HAS_BSM,
  'subaru': SubaruFlags.HAS_BSM,
  'toyota': ToyotaFlags.HAS_BSM,
  'volkswagen': VolkswagenFlags.HAS_BSM,
}


def side_left(direction):
  # +1 when the lane being left is on the radar's positive (left) side, a change to the right
  return 1.0 if direction == LaneChangeDirection.right else -1.0


def path_offset(model, d_rel, y_rel, direction):
  # how far a radar frame point sits from the planned path at its own distance, toward the lane being left; a bend
  # and the car's yaw move the path and a car in the lane together
  if len(model.position.x) == 0:
    return 0.0
  return side_left(direction) * (y_rel + float(np.interp(d_rel + RADAR_TO_CAMERA, model.position.x, model.position.y)))


def line_sweep(line, x):
  # a held line's lateral drift out to x, which carries the bend and the car's yaw
  line_x, line_y = line
  return float(np.interp(x, line_x, line_y)) - float(line_y[0])


class LaneLines:
  # the model's two lane lines in the radar's left positive frame, sampled at a distance ahead so a bend keeps
  # the target lane band on the lane. While the car straddles the line it crosses, the model doubts which lane is
  # the car's and the lines' probabilities collapse, so the owner passes that line in, followed by position, with
  # the last confident lane width
  def __init__(self, model, crossed=None, width=None):
    lane_lines, probs = model.laneLines, model.laneLineProbs
    valid = len(lane_lines) >= 3 and len(probs) >= 3 and len(lane_lines[1].x) > 0 and len(lane_lines[2].x) > 0
    self.left_valid = valid and probs[1] > 0.5
    self.right_valid = valid and probs[2] > 0.5
    self.x = (np.array(lane_lines[1].x), np.array(lane_lines[2].x)) if valid else None
    self.y = (-np.array(lane_lines[1].y), -np.array(lane_lines[2].y)) if valid else None
    self.crossed = crossed
    self.width = width

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
    if self.crossed is not None:
      width = self.width if self.width is not None else LANE_WIDTH_DEFAULT
      # the near edge stays off the car's own path, the far edge one lane width past the line itself
      line = -side_left(direction) * float(self.crossed[1][0])
      near = float(np.clip(line, LINE_OFFSET_MIN, LINE_OFFSET_MAX))
      far = min(line, LINE_OFFSET_MAX) + width
      sweep = line_sweep(self.crossed, x)
      if direction == LaneChangeDirection.left:
        return near + sweep, far + sweep
      return -far + sweep, -near + sweep
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
    if v_track < -TRACK_MOVING_SPEED:
      # oncoming traffic is judged where it will be, on the lane's outer edge too, which is where it drives on a
      # two-lane road
      if pt.dRel + pt.vRel * ARRIVAL_TIME < follow_gap:
        return True
    elif v_track < TRACK_MOVING_SPEED:
      # stationary returns are mostly clutter, so only one already inside the follow gap counts as a stopped car, and
      # not on the lane's outer edge: that is the roadside, where a stopped car in the lane cannot sit
      outer = hi if direction == LaneChangeDirection.left else lo
      if pt.dRel < follow_gap and abs(outer - pt.yRel) >= ROADSIDE_MARGIN:
        return True
    else:
      gap = max(v_ego**2 - v_track**2, 0.0) / (2 * comfort_brake) + follow_gap
      if pt.dRel + pt.vRel * ARRIVAL_TIME < gap:
        return True
  return False


def model_lead(model, i):
  return model.leadsV3[i] if len(model.leadsV3) > i else None


def lateral_agrees(lead, lead_model):
  # radard pairs the model's lead with its likeliest radar track on range and speed, with no lateral bound, so a car the
  # radar does not see, a motorcycle say, can come with the track of a car in the next lane. Where the model places its
  # lead, the track stands for it only in the same place
  if lead_model is None or len(lead_model.y) == 0 or len(lead_model.yStd) == 0 or lead_model.yStd[0] >= LEAD_UNPLACED_STD_HOLD:
    return True
  return abs(lead.yRel + lead_model.y[0]) < LEAD_CONTINUITY


def too_close(d_rel, v_rel, v_lead, a_ego, t_follow, stop_distance):
  # whether a car handed back to the MPC now, after the car has kept its acceleration for RESTORE_LAG, needs more than
  # RESTORE_DECEL of steady braking to stop closing before the follow distance behind it. That bounds the average, not
  # the MPC: while closing it aims further back and is jerk limited, so its peak runs about 1.1 to 1.5 times that at
  # 5 m/s of closing and up to 1.7 times at 7 m/s. A car already inside the follow distance asks for little while it
  # holds or opens the gap, so there it counts only while closing or inside LEAD_DANGER_FACTOR of that distance
  gap = t_follow * max(v_lead, 0.0) + stop_distance
  closing = -v_rel + a_ego * RESTORE_LAG
  room = d_rel - gap - max(closing - v_rel, 0.0) / 2 * RESTORE_LAG
  if room <= 0.0:
    return closing > 0.0 or d_rel < LEAD_DANGER_FACTOR * gap
  return max(closing, 0.0)**2 / (2 * room) > RESTORE_DECEL


class LaneChangeGap:
  """Owns the leads the MPC follows while a lane change starts.

  The model keeps the lead in the lane being left for about 1.5 s after the change begins, so the
  planner brakes behind a car the driver is pulling out to pass. While the change is starting, that
  car is still the lead, the target lane is clear and the set speed is above the current speed, the
  MPC's time gap to it is relaxed to LANE_CHANGE_T_FOLLOW so the cruise acceleration takes over, and
  the MPC targets the relaxed distance. The relaxation ends for the rest of the change when the model
  hands the lead over, the lead brakes (its filtered deceleration, a drop in its raw speed or the
  model's own estimate), the driver backs out (blinker off before the car is across, counter-steer,
  a gas or brake override), the floor below hands a lead back, or after RELAX_TIME_MAX.

  A slower lead cannot be passed at any finite gap, and after the handoff the model's lead slides onto
  the lane being left with no radar return behind it. So while the target lane is clear and the driver
  has not backed out, the MPC stops following a radar lead the planned path has left toward that lane,
  unless the model places the lead radard matched it to elsewhere, and, once the car being passed has
  gone off the path and until the car is across, a lead only the model sees while it is off the path
  too, the model cannot place it and radar sees nothing on the path. A stopped lead is always
  followed, and so is one the floor, too_close(), finds too close: one that would need more than
  RESTORE_DECEL of steady braking to stop closing before the follow distance, or is inside that
  distance while still closing or within LEAD_DANGER_FACTOR of it. Once a lead, or the car being
  passed, has been too close for CLOSE_FRAMES running, every lead is followed at the full gap until
  the car being passed is beside the car. A radar lead beside the car is past the floor and the latch
  while the plan keeps as clear of it. Far out the path can stray LEAD_OFF_PATH from a car still in
  the lane, but a far lead rarely binds and the same floor bounds it. leadTwo gets leadOne's verdict
  when radard gives both slots the same radar track, is judged on its own otherwise, and is never
  released while braking.

  The release has no timer: it ends with the change, and a stall is caught by the floor. What the MPC
  plans on, followed(), is also what its crash check sees, so FCW does not warn for a released lead;
  radarState itself, and hasLead with it, stays as radard published it. A planner reset (a gas or
  brake override) ends the release for the rest of the change. In experimental mode the model's own
  acceleration stays among the planner's candidates, and a release does not lift it.
  """

  def __init__(self, CP, dt=DT_MDL):
    self.dt = dt
    # the check needs a view of the target lane: front radar tracks ahead, the blind spot monitor beside
    self.enabled = not CP.radarUnavailable and bool(CP.flags & BSM_FLAGS.get(CP.brand, 0))
    self.starting_prev = False
    self.backed_out = False
    self.lead_id = -1
    self.lead_distance = 0.0
    self.lead_v_rel = 0.0
    self.lead_v_max = 0.0
    self.lead_speeds = deque(maxlen=LEAD_SPEED_FRAMES)
    self.passing_id = -1
    self.passing_distance = 0.0
    self.passing_y_rel = 0.0
    self.passing_v_rel = 0.0
    self.leaving = False
    self.moved_over = False
    self.beside_frames = 0
    self.closing = False
    self.close_frames = 0
    self.landed_frames = 0
    self.across = False
    self.lines_seen = [None, None]
    self.lines_age = [np.inf, np.inf]
    self.lane_width = None
    self.crossed = None
    self.crossed_age = 0.0
    self.reset()

  def reset(self):
    # a planner reset mid change (a gas or brake override) ends the relaxation and the acceleration for that change
    self.armed = False
    self.backed_out = self.starting_prev
    self.relax_timer = 0.0
    self.blocked_timer = 0.0
    self.t_follow_pad = 0.0
    self.accelerate = False
    self.released = [False, False]
    self.released_leads = []
    self.released_vision = [False, False]

  def remember(self, lead):
    self.lead_id = lead.radarTrackId if lead.radar else -1
    self.lead_distance = lead.dRel
    self.lead_v_rel = lead.vRel
    # the fastest the lead has read for LEAD_SPEED_FRAMES frames running, so one high glitch cannot raise it
    if len(self.lead_speeds) == LEAD_SPEED_FRAMES:
      self.lead_v_max = max(self.lead_v_max, min(self.lead_speeds))

  def same_lead(self, lead):
    if not lead.present:
      return False
    if lead.radar and lead.radarTrackId == self.lead_id:
      return True
    # a track id can change on the same car, and a vision lead has none
    return abs(lead.dRel - (self.lead_distance + self.lead_v_rel * self.dt)) < LEAD_CONTINUITY and \
           abs(lead.vRel - self.lead_v_rel) < LEAD_SPEED_CONTINUITY

  def follow_passing(self, radar_tracks):
    # the car the change started behind, by its radar track or one continuing it, whether or not radard still
    # calls it the lead; None while the radar does not see it
    if self.passing_id < 0:
      return None
    expected = self.passing_distance + self.passing_v_rel * self.dt
    track = next((pt for pt in radar_tracks.points if pt.trackId == self.passing_id), None)
    if track is None:
      # a target lane car can match its range and speed, so a new track continues it only in the same place
      near = [pt for pt in radar_tracks.points if abs(pt.dRel - expected) < LEAD_CONTINUITY and
              abs(pt.yRel - self.passing_y_rel) < LEAD_CONTINUITY and abs(pt.vRel - self.passing_v_rel) < LEAD_SPEED_CONTINUITY]
      track = min(near, key=lambda pt: abs(pt.dRel - expected)) if near else None
    if track is None:
      self.passing_distance = expected
      return None
    self.passing_id, self.passing_distance, self.passing_y_rel, self.passing_v_rel = track.trackId, track.dRel, track.yRel, track.vRel
    return track

  @staticmethod
  def model_braking(model, i):
    lead_model = model_lead(model, i)
    return lead_model is not None and lead_model.prob > 0.5 and len(lead_model.a) > 0 and lead_model.a[0] < LEAD_BRAKING

  def lead_braking(self, lead, model):
    # the radar's filtered deceleration lags a hard brake by about half a second; the raw speed and the
    # model's estimate of the lead's acceleration both read it earlier. The speed history belongs to the
    # lead being left, so it only counts while that lead is still the one in front
    slowing = self.armed and len(self.lead_speeds) == LEAD_SPEED_FRAMES and max(self.lead_speeds) < self.lead_v_max - LEAD_SLOWING
    return (lead.present and (lead.aLeadK < LEAD_BRAKING or slowing)) or self.model_braking(model, 0)

  def lead_two_braking(self, radar_state, model):
    # a second lead that brakes is never released, and one released is followed again, as the first one is
    lead = radar_state.leadTwo
    return (lead.present and lead.aLeadK < LEAD_BRAKING) or self.model_braking(model, 1)

  @staticmethod
  def driver_backs_out(CS, direction, across):
    # once the car is across, switching the blinker off finishes the change rather than abandoning it
    if direction == LaneChangeDirection.left:
      return (not CS.leftBlinker and not across) or (CS.steeringPressed and CS.steeringTorque < 0)
    return (not CS.rightBlinker and not across) or (CS.steeringPressed and CS.steeringTorque > 0)

  def follow_lines(self, model, direction, starting):
    # the car's own lines whenever the model places them outside a change, aged only there, and through the change the
    # line being crossed, followed by position: the model keeps placing it while it doubts it is the car's, and
    # relabels it once the car is across
    lines, probs, stds = model.laneLines, model.laneLineProbs, model.laneLineStds
    valid = len(lines) == 4 and len(probs) == 4 and all(len(line.x) > 0 for line in lines)
    placed = [valid and (probs[i] > 0.5 or (len(stds) == 4 and stds[i] < LINE_STD_MAX)) for i in range(4)]
    if valid and probs[1] > 0.5 and probs[2] > 0.5:
      self.lane_width = float(np.clip(lines[2].y[0] - lines[1].y[0], LANE_WIDTH_MIN, LANE_WIDTH_MAX))
    if not (starting and self.starting_prev):
      for side, i in enumerate((1, 2)):
        self.lines_age[side] += self.dt
        if placed[i]:
          self.lines_seen[side], self.lines_age[side] = (np.array(lines[i].x), -np.array(lines[i].y)), 0.0
    if not starting:
      self.crossed = None
    elif not self.starting_prev:
      side = 0 if direction == LaneChangeDirection.left else 1
      seen, age = self.lines_seen[side], self.lines_age[side]
      self.crossed = seen if seen is not None and age <= LINE_HOLD and abs(seen[1][0]) < LINE_OFFSET_MAX else None
      self.crossed_age = 0.0
    elif self.crossed is not None:
      # the placed line nearest where the crossed line was; the next one over is at least a lane away
      gaps = [abs(-lines[i].y[0] - self.crossed[1][0]) if placed[i] else np.inf for i in range(4)]
      i = int(np.argmin(gaps))
      if gaps[i] < LANE_WIDTH_MIN / 2:
        self.crossed, self.crossed_age = (np.array(lines[i].x), -np.array(lines[i].y)), 0.0
      else:
        self.crossed_age += self.dt
        if self.crossed_age > LINE_HOLD:
          self.crossed = None

  def landed(self, direction):
    return self.crossed is not None and side_left(direction) * self.crossed[1][0] >= LANDED_MARGIN

  def target_lane_clear(self, direction, CS, model, radar_tracks, radar_ok, v_ego, t_follow, stop_distance, comfort_brake):
    blind_spot = CS.leftBlindspot if direction == LaneChangeDirection.left else CS.rightBlindspot
    # once across the line, the lane moved into is the car's own, where its leads are followed
    if blind_spot or not radar_ok or (not self.landed(direction) and target_lane_blocked(
       direction, LaneLines(model, self.crossed, self.lane_width), radar_tracks, v_ego, t_follow, stop_distance, comfort_brake)):
      self.blocked_timer = BLOCKED_HOLD
    else:
      self.blocked_timer = max(self.blocked_timer - self.dt, 0.0)
    return self.blocked_timer <= 0.0

  def beside(self, d_rel, y_rel, direction):
    # the radar lateral toward the lane being left, measured from the crossed line's sweep out to that distance, so
    # neither a bend nor the car's yaw mid change puts a car still in its lane beside the car
    sweep = line_sweep(self.crossed, d_rel + RADAR_TO_CAMERA) if self.crossed is not None else 0.0
    return side_left(direction) * (y_rel - sweep) >= LEAD_BESIDE

  def was_released(self, lead):
    # the same car as one released last frame: its radar track, or its distance and speed across an id change
    for radar, track_id, d_rel, v_rel in self.released_leads:
      if lead.radar and radar and lead.radarTrackId == track_id:
        return True
      if abs(lead.dRel - (d_rel + v_rel * self.dt)) < LEAD_CONTINUITY and abs(lead.vRel - v_rel) < LEAD_SPEED_CONTINUITY:
        return True
    return False

  @staticmethod
  def same_track(a, b):
    return a.present and b.present and a.radar and b.radar and a.radarTrackId == b.radarTrackId

  @staticmethod
  def corridor_clear(model, direction, radar_tracks, d_rel, offset_min):
    # radar sees nothing on the planned path from beside the car to the far end of radard's own match window
    d_max = d_rel + max(0.25 * d_rel, 5.0)
    return all(abs(path_offset(model, pt.dRel, pt.yRel, direction)) >= offset_min
               for pt in radar_tracks.points if TRACK_MIN_DISTANCE < pt.dRel < d_max)

  def releases(self, i, lead, model, direction, radar_tracks):
    if not lead.present or lead.vLead < TRACK_MOVING_SPEED:
      return False
    offset = path_offset(model, lead.dRel, lead.yRel, direction)
    lead_model = model_lead(model, i)
    if lead.radar:
      return lateral_agrees(lead, lead_model) and offset > (LEAD_ON_PATH if self.was_released(lead) else LEAD_OFF_PATH)
    # a lead only the model sees: released once the car being passed has gone off the path, while the model cannot
    # place it, radar sees nothing on the path out past it and it is off the path itself, as a radar lead is. The
    # model's range on such a lead can run 5 to 15 m long, so one on the path may be the car ahead. It is the model's
    # slot rather than a tracked car, so the slot's release holds while its lead stays unplaced and off the path. Once
    # the car is across, the model's lead is the one in the lane moved into
    if lead_model is None or len(lead_model.yStd) == 0 or not self.leaving or self.across:
      return False
    held = self.released_vision[i]
    return offset > (LEAD_ON_PATH if held else LEAD_OFF_PATH) and \
           lead_model.yStd[0] >= (LEAD_UNPLACED_STD_HOLD if held else LEAD_UNPLACED_STD) and \
           self.corridor_clear(model, direction, radar_tracks, lead.dRel, LEAD_ON_PATH if held else LEAD_OFF_PATH)

  def release_leads(self, radar_state, model, direction, radar_tracks, passing, v_ego, a_ego, t_follow, stop_distance):
    leads = (radar_state.leadOne, radar_state.leadTwo)
    one = self.releases(0, leads[0], model, direction, radar_tracks)
    # one radar track in both slots is one car, whatever the model's second lead shows; a pair matched on range and speed
    # alone can be two cars, so each lead is judged on its own
    two = one if self.same_track(*leads) else self.releases(1, leads[1], model, direction, radar_tracks)
    two = two and not self.lead_two_braking(radar_state, model)
    # a car beside is past the floor while the plan keeps clear of it too; one the path swings back toward is judged again
    beside = [lead.radar and self.beside(lead.dRel, lead.yRel, direction) and path_offset(model, lead.dRel, lead.yRel, direction) >= LEAD_BESIDE
              for lead in leads]
    close = [candidate and not b and too_close(lead.dRel, lead.vRel, lead.vLead, a_ego, t_follow, stop_distance)
             for candidate, b, lead in zip((one, two), beside, leads, strict=True)]
    # the car being passed still overlaps the car after the model has moved its lead elsewhere
    passing_close = passing is not None and not self.moved_over and \
                    not any(lead.present and lead.radar and lead.radarTrackId == passing.trackId for lead in leads) and \
                    too_close(passing.dRel, passing.vRel, v_ego + passing.vRel, a_ego, t_follow, stop_distance)
    self.close_frames = self.close_frames + 1 if any(close) or passing_close else 0
    if self.close_frames >= CLOSE_FRAMES:
      # followed at the full gap until the car being passed is beside the car
      self.closing = True
      self.armed = False
    released = []
    for k, (candidate, lead) in enumerate(zip((one, two), leads, strict=True)):
      # the debounce keeps the car released last frame, not whichever car now fills its slot
      held = self.close_frames < CLOSE_FRAMES and self.was_released(lead)
      released.append(bool(candidate and (beside[k] or not self.closing) and (not close[k] or held)))
    return released

  def update(self, model, CS, radar_state, radar_tracks, radar_ok, v_ego, v_cruise, t_follow, stop_distance, comfort_brake):
    direction = model.meta.laneChangeDirection
    starting = model.meta.laneChangeState == LaneChangeState.laneChangeStarting and direction != LaneChangeDirection.none
    lead = radar_state.leadOne
    self.follow_lines(model, direction, starting)

    if starting and not self.starting_prev:
      self.armed = self.enabled and lead.present
      self.backed_out = False
      self.relax_timer = 0.0
      self.lead_speeds.clear()
      self.lead_speeds.append(lead.vLead)
      # unknown until a full window has been read, so a glitch on this frame cannot set it
      self.lead_v_max = -np.inf
      self.remember(lead)
      # the car being passed is the lead's radar track, where the model puts its lead
      self.passing_id = lead.radarTrackId if lead.present and lead.radar and lateral_agrees(lead, model_lead(model, 0)) else -1
      self.passing_distance, self.passing_y_rel, self.passing_v_rel = lead.dRel, lead.yRel, lead.vRel
      self.leaving = False
      self.moved_over = False
      self.beside_frames = 0
      self.closing = False
      self.close_frames = 0
      self.landed_frames = 0
      self.across = False
    elif starting and self.armed:
      self.relax_timer += self.dt
      self.lead_speeds.append(lead.vLead)
    passing = self.follow_passing(radar_tracks) if starting else None
    if starting:
      # counted on the line as placed this frame: one held through a blackout is no new sign of landing
      if self.crossed_age == 0.0:
        self.landed_frames = self.landed_frames + 1 if self.landed(direction) else 0
      self.across = self.across or self.landed_frames >= LANDED_FRAMES
      # the model handing the lead over or the time limit end the relaxation only; the driver backing out or
      # the lead braking, whichever lead is in front, end the acceleration for this change too
      if self.armed and (self.relax_timer > RELAX_TIME_MAX or not self.same_lead(lead)):
        self.armed = False
      if self.driver_backs_out(CS, direction, self.across) or self.lead_braking(lead, model) or \
         (self.released[1] and self.lead_two_braking(radar_state, model)):
        self.armed = False
        self.backed_out = True
      elif self.armed:
        self.remember(lead)
      if passing is not None:
        # the path leaving the car being passed shows the car is moving over; the path coming back undoes it
        offset = path_offset(model, passing.dRel, passing.yRel, direction)
        if offset > LEAD_OFF_PATH:
          self.leaving = True
        elif offset < LEAD_ON_PATH:
          self.leaving = False
        # beside for as many frames running as the latch takes to set: one wide radar frame is no car moved over
        self.beside_frames = self.beside_frames + 1 if self.beside(passing.dRel, passing.yRel, direction) else 0
        if not self.moved_over and self.beside_frames >= CLOSE_FRAMES:
          self.moved_over = True
          self.closing = False
    if not starting:
      self.armed = False
      self.backed_out = False
      self.blocked_timer = 0.0
    self.starting_prev = starting

    self.accelerate = False
    self.t_follow_pad = 0.0
    released = [False, False]
    if starting:
      clear = self.target_lane_clear(direction, CS, model, radar_tracks, radar_ok, v_ego, t_follow, stop_distance, comfort_brake)
      # the target lane gate and the back-out cues but not the headroom: a car reaching its set speed mid pass is not
      # handed back the car it is passing
      if self.enabled and clear and not self.backed_out:
        released = self.release_leads(radar_state, model, direction, radar_tracks, passing, v_ego, CS.aEgo, t_follow, stop_distance)
      else:
        # the floor latches on frames it judges running
        self.close_frames = 0
      self.accelerate = self.enabled and clear and not self.backed_out and v_cruise - v_ego > MIN_HEADROOM
      if self.accelerate and self.armed:
        self.t_follow_pad = min(LANE_CHANGE_T_FOLLOW - t_follow, 0.0)
    self.released = released
    self.released_leads = [(ld.radar, ld.radarTrackId, ld.dRel, ld.vRel)
                           for ld, r in zip((radar_state.leadOne, radar_state.leadTwo), released, strict=True) if r]
    self.released_vision = [r and not ld.radar for ld, r in zip((radar_state.leadOne, radar_state.leadTwo), released, strict=True)]
    return self.t_follow_pad

  def followed(self, radar_state):
    # the leads the MPC plans for: radard's, less the ones released for this lane change; radarState itself stays
    # as radard published it for everything else
    if not any(self.released):
      return radar_state
    followed = log.RadarState.new_message(leadOne=radar_state.leadOne, leadTwo=radar_state.leadTwo)
    followed.leadOne.present = radar_state.leadOne.present and not self.released[0]
    followed.leadTwo.present = radar_state.leadTwo.present and not self.released[1]
    return followed

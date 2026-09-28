import numpy as np

from openpilot.cereal import log
from opendbc.car.structs import car
from opendbc.car.hyundai.values import HyundaiFlags
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.lane_change_gap import (LaneChangeGap, LaneLines, target_lane_blocked, ARRIVAL_TIME, BLOCKED_HOLD,
                                                              LANDED_MARGIN, LANE_CHANGE_T_FOLLOW, LANE_WIDTH_DEFAULT, LEAD_BRAKING,
                                                              LEAD_CONTINUITY, LEAD_SLOWING, LEAD_SPEED_CONTINUITY, LEAD_SPEED_FRAMES, LINE_HOLD,
                                                              MIN_HEADROOM, RADAR_TO_CAMERA, RELAX_TIME_MAX, ROADSIDE_MARGIN)
from openpilot.selfdrive.modeld.constants import ModelConstants

LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection

V_EGO = 25.0
T_FOLLOW = 1.45
STOP_DISTANCE = 7.0
COMFORT_BRAKE = 2.5
PAD = LANE_CHANGE_T_FOLLOW - T_FOLLOW
FOLLOW_GAP = STOP_DISTANCE + T_FOLLOW * V_EGO  # behind a car at the same speed
X_IDXS = np.array(ModelConstants.X_IDXS)
CP = car.CarParams.new_message(radarUnavailable=False, brand='hyundai', flags=HyundaiFlags.HAS_BSM.value)


def get_model(state=LaneChangeState.laneChangeStarting, direction=LaneChangeDirection.left, left_y=-1.6, right_y=1.76, probs=(1.0, 1.0),
              curvature=0.0, lead_accel=0.0, lead_prob=1.0, line_stds=None, lines=None):
  model = log.ModelDataV2.new_message()
  model.init('leadsV3', 1)
  model.leadsV3[0].prob = lead_prob
  model.leadsV3[0].a = [lead_accel] * len(ModelConstants.LEAD_T_IDXS)
  model.meta.laneChangeState = state
  model.meta.laneChangeDirection = direction
  model.init('laneLines', 4)
  lines = lines if lines is not None else (left_y - 3.5, left_y, right_y, right_y + 3.5)
  for line, y in zip(model.laneLines, lines, strict=True):
    line.x = X_IDXS.tolist()
    line.y = (y + curvature * X_IDXS**2 / 2).tolist()
  model.laneLineProbs = [1.0, *probs, 1.0] if len(probs) == 2 else list(probs)
  if line_stds is not None:
    model.laneLineStds = list(line_stds)
  return model


def get_radar_state(present=True, radar=True, track_id=7, d_rel=35.0, v_rel=-1.0, a_lead=0.0):
  radar_state = log.RadarState.new_message()
  lead = radar_state.leadOne
  lead.present = present
  lead.radar = radar
  lead.radarTrackId = track_id
  lead.dRel = d_rel
  lead.vRel = v_rel
  lead.vLead = V_EGO + v_rel
  lead.aLeadK = a_lead
  return radar_state


def get_tracks(*points):
  tracks = car.RadarData.new_message()
  tracks.init('points', len(points))
  for pt, (d_rel, y_rel, v_rel) in zip(tracks.points, points, strict=True):
    pt.dRel = d_rel
    pt.yRel = y_rel
    pt.vRel = v_rel
  return tracks


def get_car_state(left_blinker=True, **kwargs):
  return car.CarState.new_message(vEgo=V_EGO, leftBlinker=left_blinker, **kwargs)


def blocked(tracks, direction=LaneChangeDirection.left, model=None, v_ego=V_EGO):
  lines = LaneLines(model if model is not None else get_model())
  return target_lane_blocked(direction, lines, tracks, v_ego, T_FOLLOW, STOP_DISTANCE, COMFORT_BRAKE)


def step(gap, model=None, CS=None, radar_state=None, tracks=None, radar_ok=True, v_cruise=V_EGO + 5.0, t_follow=T_FOLLOW, n=1):
  model = model if model is not None else get_model()
  CS = CS if CS is not None else get_car_state()
  radar_state = radar_state if radar_state is not None else get_radar_state()
  tracks = tracks if tracks is not None else get_tracks()
  pad = 0.0
  for _ in range(n):
    pad = gap.update(model, CS, radar_state, tracks, radar_ok, V_EGO, v_cruise, t_follow, STOP_DISTANCE, COMFORT_BRAKE)
  return pad


def relaxed(gap):
  # a change starting behind a slower lead with a clear lane and headroom
  return np.isclose(step(gap), PAD)


class TestTargetLane:
  def test_band_follows_the_lane_lines(self):
    lines = LaneLines(get_model(left_y=-1.6, right_y=1.76))
    lo, hi = lines.band(LaneChangeDirection.left, 0.0)
    assert np.isclose(lo, 1.6) and np.isclose(hi, 1.6 + 3.36)
    lo, hi = lines.band(LaneChangeDirection.right, 0.0)
    assert np.isclose(lo, -(1.76 + 3.36)) and np.isclose(hi, -1.76)

  def test_band_falls_back_without_confident_lines(self):
    lo, hi = LaneLines(get_model(probs=(0.2, 0.2))).band(LaneChangeDirection.left, 0.0)
    assert np.isclose(lo, LANE_WIDTH_DEFAULT / 2) and np.isclose(hi, 1.5 * LANE_WIDTH_DEFAULT)
    assert LaneLines(log.ModelDataV2.new_message()).band(LaneChangeDirection.right, 40.0) == (-1.5 * LANE_WIDTH_DEFAULT, -LANE_WIDTH_DEFAULT / 2)

  def test_band_bends_with_the_other_line_when_the_near_one_is_lost(self):
    curvature = -1 / 500
    d_rel = 45.0
    offset = -curvature * (d_rel + RADAR_TO_CAMERA)**2 / 2
    lo, hi = LaneLines(get_model(curvature=curvature, probs=(0.2, 1.0))).band(LaneChangeDirection.left, d_rel + RADAR_TO_CAMERA)
    assert np.isclose(lo, LANE_WIDTH_DEFAULT / 2 + offset, atol=0.02) and np.isclose(hi, 1.5 * LANE_WIDTH_DEFAULT + offset, atol=0.02)

  def test_band_bends_with_the_road(self):
    # a left bend of 500 m puts the target lane 2 m further left at 45 m; the model's y is right positive
    curvature = -1 / 500
    model = get_model(curvature=curvature)
    d_rel = 45.0
    offset = -curvature * (d_rel + RADAR_TO_CAMERA)**2 / 2
    car_ahead = (d_rel, 3.4 + offset, -3.0)
    assert blocked(get_tracks(car_ahead), model=model)
    assert not blocked(get_tracks(car_ahead))
    assert not blocked(get_tracks((d_rel, 3.4, -3.0)), model=model)

  def test_car_inside_the_follow_gap_blocks(self):
    assert blocked(get_tracks((FOLLOW_GAP - 1.0, 3.4, 0.0)))
    assert not blocked(get_tracks((FOLLOW_GAP + 1.0, 3.4, 0.0)))

  def test_slower_car_needs_the_mpc_gap_at_its_speed(self):
    v_rel = -1.0
    gap = (V_EGO**2 - (V_EGO + v_rel)**2) / (2 * COMFORT_BRAKE) + FOLLOW_GAP
    # judged where the car will be after the change, so the closing over ARRIVAL_TIME counts too
    boundary = gap - ARRIVAL_TIME * v_rel
    assert blocked(get_tracks((boundary - 1.0, 3.4, v_rel)))
    assert not blocked(get_tracks((boundary + 1.0, 3.4, v_rel)))
    # a car pulling away is judged where it will be, without a stopping distance surplus
    assert not blocked(get_tracks((FOLLOW_GAP + 1.0, 3.4, 3.0)))
    assert not blocked(get_tracks((FOLLOW_GAP - 1.0, 3.4, 1.0)))
    assert blocked(get_tracks((FOLLOW_GAP - 1.0 - ARRIVAL_TIME * 0.2, 3.4, 0.2)))

  def test_stopped_car_and_oncoming_traffic(self):
    stopped_close = (FOLLOW_GAP - 1.0, 3.4, -V_EGO)
    clutter = (FOLLOW_GAP + 1.0, 3.4, -V_EGO)
    oncoming_arriving = (FOLLOW_GAP + 2.0 * V_EGO * ARRIVAL_TIME - 1.0, 3.4, -2.0 * V_EGO)
    oncoming_far = (FOLLOW_GAP + 2.0 * V_EGO * ARRIVAL_TIME + 1.0, 3.4, -2.0 * V_EGO)
    assert blocked(get_tracks(stopped_close))
    assert not blocked(get_tracks(clutter))
    assert blocked(get_tracks(oncoming_arriving))
    assert not blocked(get_tracks(oncoming_far))

  def test_other_lanes_and_beside_returns_are_ignored(self):
    own_lane = (20.0, 0.0, -1.0)
    other_side = (20.0, -3.4, -1.0)
    beside = (1.0, 3.4, -1.0)
    assert not blocked(get_tracks(own_lane, other_side, beside))
    assert blocked(get_tracks(other_side), direction=LaneChangeDirection.right)


class TestLaneChangeGap:
  def test_relaxes_the_gap_at_once_toward_the_lead_being_left(self):
    gap = LaneChangeGap(CP, DT_MDL)
    assert relaxed(gap)
    assert gap.accelerate

  def test_effective_gap_follows_the_personality(self):
    gap = LaneChangeGap(CP, DT_MDL)
    assert np.isclose(step(gap, t_follow=1.75), LANE_CHANGE_T_FOLLOW - 1.75)
    assert np.isclose(step(gap, t_follow=1.0), LANE_CHANGE_T_FOLLOW - 1.0)
    assert step(gap, t_follow=0.5) == 0.0

  def test_needs_headroom(self):
    gap = LaneChangeGap(CP, DT_MDL)
    assert step(gap, v_cruise=V_EGO + MIN_HEADROOM) == 0.0
    assert not gap.accelerate

  def test_needs_the_sensors(self):
    for cp in (car.CarParams.new_message(radarUnavailable=True, brand='hyundai', flags=HyundaiFlags.HAS_BSM.value),
               car.CarParams.new_message(radarUnavailable=False, brand='hyundai'),
               car.CarParams.new_message(radarUnavailable=False, brand='mazda', flags=HyundaiFlags.HAS_BSM.value)):
      gap = LaneChangeGap(cp, DT_MDL)
      assert step(gap) == 0.0
      assert not gap.accelerate

  def test_only_while_the_change_is_starting(self):
    gap = LaneChangeGap(CP, DT_MDL)
    assert step(gap, model=get_model(state=LaneChangeState.preLaneChange)) == 0.0
    assert step(gap, model=get_model(state=LaneChangeState.off)) == 0.0
    assert step(gap, model=get_model(direction=LaneChangeDirection.none)) == 0.0

  def test_arms_only_as_the_change_starts(self):
    # a lead that appears later in the change is the target lane's, not the one being left
    gap = LaneChangeGap(CP, DT_MDL)
    assert step(gap, radar_state=get_radar_state(present=False)) == 0.0
    assert gap.accelerate
    assert step(gap, n=10) == 0.0
    # a new change arms again
    step(gap, model=get_model(state=LaneChangeState.off))
    assert relaxed(gap)

  def test_reset_ends_the_relaxation_and_the_acceleration_for_that_change(self):
    gap = LaneChangeGap(CP, DT_MDL)
    assert relaxed(gap)
    gap.reset()
    assert step(gap, n=10) == 0.0
    assert not gap.accelerate
    assert step(gap, radar_state=get_radar_state(track_id=8)) == 0.0
    step(gap, model=get_model(state=LaneChangeState.off))
    assert relaxed(gap)
    # a reset outside a change does not spoil the next one
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=get_model(state=LaneChangeState.off))
    gap.reset()
    assert relaxed(gap)

  def test_lead_handoff_snaps_the_gap_back_for_the_rest_of_the_change(self):
    gap = LaneChangeGap(CP, DT_MDL)
    assert relaxed(gap)
    assert step(gap, radar_state=get_radar_state(track_id=8, d_rel=60.0)) == 0.0
    assert not gap.armed
    # neither the new lead nor the old one coming back re-arms it
    assert step(gap, radar_state=get_radar_state(track_id=7), n=50) == 0.0
    assert gap.accelerate

  def test_lead_lost_releases(self):
    gap = LaneChangeGap(CP, DT_MDL)
    assert relaxed(gap)
    assert step(gap, radar_state=get_radar_state(present=False)) == 0.0

  def test_same_car_survives_a_track_id_change(self):
    gap = LaneChangeGap(CP, DT_MDL)
    assert relaxed(gap)
    assert np.isclose(step(gap, radar_state=get_radar_state(track_id=8, d_rel=35.0 - 1.0 * DT_MDL)), PAD)
    assert np.isclose(step(gap, radar_state=get_radar_state(radar=False, track_id=-1, d_rel=35.0 - 2.0 * DT_MDL)), PAD)

  def test_vision_lead_is_followed_by_distance_and_speed(self):
    gap = LaneChangeGap(CP, DT_MDL)
    vision = {"radar": False, "track_id": -1}
    assert np.isclose(step(gap, radar_state=get_radar_state(**vision, d_rel=35.0)), PAD)
    assert np.isclose(step(gap, radar_state=get_radar_state(**vision, d_rel=35.0 + LEAD_CONTINUITY / 2)), PAD)
    assert step(gap, radar_state=get_radar_state(**vision, d_rel=35.0 + LEAD_CONTINUITY / 2, v_rel=-1.0 - 2 * LEAD_SPEED_CONTINUITY)) == 0.0
    gap = LaneChangeGap(CP, DT_MDL)
    assert relaxed(gap)
    assert step(gap, radar_state=get_radar_state(**vision, d_rel=35.0 + 2 * LEAD_CONTINUITY)) == 0.0

  def test_braking_lead_keeps_its_gap(self):
    gap = LaneChangeGap(CP, DT_MDL)
    assert relaxed(gap)
    assert step(gap, radar_state=get_radar_state(a_lead=LEAD_BRAKING - 0.5)) == 0.0
    assert not gap.accelerate
    assert step(gap, n=10) == 0.0
    assert not gap.accelerate
    # a change that starts behind a braking lead never relaxes
    gap = LaneChangeGap(CP, DT_MDL)
    assert step(gap, radar_state=get_radar_state(a_lead=LEAD_BRAKING - 0.5)) == 0.0

  def test_lead_speed_drop_and_model_estimate_read_a_brake_early(self):
    gap = LaneChangeGap(CP, DT_MDL)
    assert relaxed(gap)
    assert np.isclose(step(gap, radar_state=get_radar_state(v_rel=-1.0 - LEAD_SLOWING / 2), n=LEAD_SPEED_FRAMES), PAD)
    slower = get_radar_state(v_rel=-1.0 - 2 * LEAD_SLOWING)
    assert np.isclose(step(gap, radar_state=slower, n=LEAD_SPEED_FRAMES - 1), PAD)
    assert step(gap, radar_state=slower) == 0.0
    assert not gap.accelerate
    # one frame's glitch, low or high, is not a brake
    gap = LaneChangeGap(CP, DT_MDL)
    assert relaxed(gap)
    assert np.isclose(step(gap, radar_state=get_radar_state(v_rel=-1.0 - 5.0)), PAD)
    assert np.isclose(step(gap, radar_state=get_radar_state(v_rel=-1.0 + 5.0)), PAD)
    assert np.isclose(step(gap, n=20), PAD)
    # a lead that vanishes is a handoff, not a brake, and the acceleration stays
    gap = LaneChangeGap(CP, DT_MDL)
    assert relaxed(gap)
    assert step(gap, radar_state=get_radar_state(present=False)) == 0.0
    assert gap.accelerate
    gap = LaneChangeGap(CP, DT_MDL)
    assert relaxed(gap)
    assert np.isclose(step(gap, model=get_model(lead_accel=LEAD_BRAKING - 1.0, lead_prob=0.3)), PAD)
    assert step(gap, model=get_model(lead_accel=LEAD_BRAKING - 1.0)) == 0.0
    assert not gap.accelerate
    # a lead that sped up first is judged from its fastest
    gap = LaneChangeGap(CP, DT_MDL)
    assert relaxed(gap)
    assert np.isclose(step(gap, radar_state=get_radar_state(v_rel=-1.0 + LEAD_SLOWING), n=LEAD_SPEED_FRAMES), PAD)
    assert step(gap, radar_state=get_radar_state(v_rel=-1.0 - LEAD_SLOWING / 2), n=LEAD_SPEED_FRAMES) == 0.0

  def test_driver_backing_out_ends_it(self):
    for CS in (get_car_state(left_blinker=False), get_car_state(steeringPressed=True, steeringTorque=-50.0)):
      gap = LaneChangeGap(CP, DT_MDL)
      assert relaxed(gap)
      assert step(gap, CS=CS) == 0.0
      assert not gap.accelerate
      assert step(gap, n=10) == 0.0
      assert not gap.accelerate
      gap = LaneChangeGap(CP, DT_MDL)
      assert step(gap, CS=CS) == 0.0
    gap = LaneChangeGap(CP, DT_MDL)
    assert relaxed(gap)
    assert np.isclose(step(gap, CS=get_car_state(steeringPressed=True, steeringTorque=50.0)), PAD)

  def test_backing_out_ends_the_acceleration_at_any_point_of_the_change(self):
    off = get_car_state(left_blinker=False)
    # after the handoff
    gap = LaneChangeGap(CP, DT_MDL)
    assert relaxed(gap)
    step(gap, radar_state=get_radar_state(track_id=8, d_rel=60.0))
    assert gap.accelerate
    step(gap, CS=off)
    assert not gap.accelerate
    assert step(gap, n=10) == 0.0 and not gap.accelerate
    # after the time limit
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, n=int(round(RELAX_TIME_MAX / DT_MDL)) + 2)
    assert gap.accelerate
    step(gap, CS=off)
    assert not gap.accelerate
    # with no lead at the start
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, radar_state=get_radar_state(present=False))
    assert gap.accelerate
    step(gap, radar_state=get_radar_state(present=False), CS=off)
    assert not gap.accelerate
    # the new lead braking after the handoff
    gap = LaneChangeGap(CP, DT_MDL)
    assert relaxed(gap)
    step(gap, radar_state=get_radar_state(track_id=8, d_rel=60.0))
    assert gap.accelerate
    step(gap, radar_state=get_radar_state(track_id=8, d_rel=60.0, a_lead=LEAD_BRAKING - 0.5))
    assert not gap.accelerate

  def test_speed_glitch_on_the_arming_frame_is_harmless(self):
    gap = LaneChangeGap(CP, DT_MDL)
    assert np.isclose(step(gap, radar_state=get_radar_state(v_rel=-1.0 + 3.0)), PAD)
    assert np.isclose(step(gap, n=20), PAD)

  def test_relaxation_times_out_but_the_acceleration_stays(self):
    gap = LaneChangeGap(CP, DT_MDL)
    steps = int(round(RELAX_TIME_MAX / DT_MDL))
    assert np.isclose(step(gap, n=steps), PAD)
    assert step(gap, n=2) == 0.0
    assert gap.accelerate

  def test_blind_spot_blocks_and_holds(self):
    gap = LaneChangeGap(CP, DT_MDL)
    assert step(gap, CS=get_car_state(leftBlindspot=True)) == 0.0
    steps = int(round(BLOCKED_HOLD / DT_MDL))
    assert step(gap, n=steps) == 0.0
    assert np.isclose(step(gap), PAD)
    gap = LaneChangeGap(CP, DT_MDL)
    assert np.isclose(step(gap, CS=get_car_state(rightBlindspot=True)), PAD)

  def test_target_lane_car_and_radar_health_block(self):
    gap = LaneChangeGap(CP, DT_MDL)
    assert step(gap, tracks=get_tracks((30.0, 3.4, -1.0))) == 0.0
    gap = LaneChangeGap(CP, DT_MDL)
    assert step(gap, radar_ok=False) == 0.0

  def test_window_end_clears_everything(self):
    gap = LaneChangeGap(CP, DT_MDL)
    assert relaxed(gap)
    assert step(gap, model=get_model(state=LaneChangeState.off)) == 0.0
    assert not gap.accelerate and not gap.armed


def change_right(**kwargs):
  return get_model(direction=LaneChangeDirection.right, **kwargs)


def right_blinker(**kwargs):
  return get_car_state(left_blinker=False, rightBlinker=True, **kwargs)


class TestCrossedLine:
  # a change to the right: the car's lines at -1.8 and +1.2 m (model frame, right positive), 3.0 m apart
  PLACED = (0.1, 0.1, 0.1, 0.1)
  DOUBTED = (0.0, 0.05, 0.05, 0.0)

  def run(self, gap, right_line, probs, stds, tracks=(), state=LaneChangeState.laneChangeStarting, CS=None, n=1, lines=None):
    lines = lines if lines is not None else (right_line - 6.0, right_line - 3.0, right_line, right_line + 3.5)
    model = change_right(state=state, lines=lines, probs=probs, line_stds=stds)
    step(gap, model=model, CS=CS if CS is not None else right_blinker(), radar_state=get_radar_state(present=False),
         tracks=get_tracks(*tracks), n=n)
    return gap.blocked_timer > 0.0

  def test_band_follows_the_crossed_line_the_model_doubts(self):
    # route bb: a pickup two lanes over at 30 m, 5.2 m to the right, slowing to turn off
    pickup = [(30.0, -5.2, -4.9)]
    target_lane_car = [(30.0, -2.5, -4.9)]
    gap = LaneChangeGap(CP, DT_MDL)
    self.run(gap, 1.2, (0.8, 0.9, 0.8, 0.8), self.PLACED, state=LaneChangeState.preLaneChange)
    assert not self.run(gap, 1.0, self.DOUBTED, self.PLACED, tracks=pickup, n=10)
    assert self.run(gap, 1.0, self.DOUBTED, self.PLACED, tracks=target_lane_car)
    # the lines as such fall back to the default lane, which reaches the pickup
    lo, hi = LaneLines(change_right(probs=(0.0, 0.05, 0.05, 0.0))).band(LaneChangeDirection.right, 30.0 + RADAR_TO_CAMERA)
    assert lo <= -5.2 <= hi

  def test_crossed_line_is_followed_across_the_relabel(self):
    beyond = [(30.0, -4.5, -4.9)]
    gap = LaneChangeGap(CP, DT_MDL)
    self.run(gap, 1.2, (0.8, 0.9, 0.8, 0.8), self.PLACED, state=LaneChangeState.preLaneChange)
    for right_line in np.arange(1.2, 0.0, -0.1):
      self.run(gap, right_line, self.DOUBTED, self.PLACED)
    # the model relabels: the crossed line is now the car's left line, 0.6 m to its left
    relabelled = (-3.6, -LANDED_MARGIN - 0.1, 2.9, 6.4)
    assert not self.run(gap, None, (0.5, 0.7, 0.7, 0.5), self.PLACED, lines=relabelled, tracks=beyond)
    assert gap.landed(LaneChangeDirection.right)
    # the blind spot still counts after landing
    assert self.run(gap, None, (0.5, 0.7, 0.7, 0.5), self.PLACED, lines=relabelled, CS=right_blinker(rightBlindspot=True))

  def test_crossed_line_hold_expires(self):
    pickup = [(30.0, -5.2, -4.9)]
    lost = (1.0, 1.0, 1.0, 1.0)
    gap = LaneChangeGap(CP, DT_MDL)
    self.run(gap, 1.2, (0.8, 0.9, 0.8, 0.8), self.PLACED, state=LaneChangeState.preLaneChange)
    assert not self.run(gap, 1.0, self.DOUBTED, self.PLACED, tracks=pickup)
    assert not self.run(gap, 1.0, self.DOUBTED, lost, tracks=pickup, n=int(round(LINE_HOLD / DT_MDL)) - 1)
    assert self.run(gap, 1.0, self.DOUBTED, lost, tracks=pickup, n=2)
    assert gap.crossed is None

  def test_crossed_line_is_seeded_before_the_change(self):
    pickup = [(30.0, -5.2, -4.9)]
    lost = (1.0, 1.0, 1.0, 1.0)
    gap = LaneChangeGap(CP, DT_MDL)
    self.run(gap, 1.2, (0.8, 0.9, 0.8, 0.8), self.PLACED, state=LaneChangeState.preLaneChange)
    self.run(gap, 1.2, self.DOUBTED, lost, state=LaneChangeState.preLaneChange, n=int(round(LINE_HOLD / DT_MDL)) + 2)
    assert self.run(gap, 1.0, self.DOUBTED, lost, tracks=pickup)
    assert gap.crossed is None

  def test_band_width_is_the_last_confident_lane(self):
    gap = LaneChangeGap(CP, DT_MDL)
    self.run(gap, 1.2, (0.8, 0.9, 0.8, 0.8), self.PLACED, state=LaneChangeState.preLaneChange)
    self.run(gap, 1.0, self.DOUBTED, self.PLACED)
    lo, hi = LaneLines(change_right(probs=self.DOUBTED), gap.crossed, gap.lane_width).band(LaneChangeDirection.right, 0.0)
    assert np.isclose(hi - lo, 3.0) and np.isclose(hi, -1.0)

  def test_far_edge_stays_one_lane_past_the_crossed_line(self):
    # with the car 0.2 m from the line, the near edge is kept off its path but the far edge does not move out
    # into the next lane over
    gap = LaneChangeGap(CP, DT_MDL)
    self.run(gap, 1.2, (0.8, 0.9, 0.8, 0.8), self.PLACED, state=LaneChangeState.preLaneChange)
    for right_line in np.arange(1.2, 0.2, -0.1):
      self.run(gap, right_line, self.DOUBTED, self.PLACED)
    far = 0.2 + 3.0
    assert not self.run(gap, 0.2, self.DOUBTED, self.PLACED, tracks=[(30.0, -(far + 0.5), -4.9)])
    assert self.run(gap, 0.2, self.DOUBTED, self.PLACED, tracks=[(30.0, -(far - 0.5), -4.9)])


class TestRoadside:
  def test_stationary_return_on_the_outer_line_is_the_roadside(self):
    _, outer = LaneLines(get_model()).band(LaneChangeDirection.left, FOLLOW_GAP - 1.0 + RADAR_TO_CAMERA)
    assert not blocked(get_tracks((FOLLOW_GAP - 1.0, outer - ROADSIDE_MARGIN / 2, -V_EGO)))
    assert blocked(get_tracks((FOLLOW_GAP - 1.0, outer - 2 * ROADSIDE_MARGIN, -V_EGO)))
    # a car moving with traffic there is a car
    assert blocked(get_tracks((FOLLOW_GAP - 1.0, outer - ROADSIDE_MARGIN / 2, 0.0)))

  def test_oncoming_car_on_the_outer_line_is_a_car(self):
    # on a two-lane road the target lane's outer edge is where oncoming traffic drives
    arriving = FOLLOW_GAP + 2.0 * V_EGO * ARRIVAL_TIME - 1.0
    _, outer = LaneLines(get_model()).band(LaneChangeDirection.left, arriving + RADAR_TO_CAMERA)
    assert blocked(get_tracks((arriving, outer - ROADSIDE_MARGIN / 2, -2.0 * V_EGO)))
    assert not blocked(get_tracks((arriving + 2.0, outer - ROADSIDE_MARGIN / 2, -2.0 * V_EGO)))

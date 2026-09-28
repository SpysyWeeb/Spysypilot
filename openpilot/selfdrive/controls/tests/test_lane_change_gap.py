import numpy as np

from openpilot.cereal import log
from opendbc.car.structs import car
from opendbc.car.hyundai.values import HyundaiFlags
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.lane_change_gap import (LaneChangeGap, LaneLines, target_lane_blocked, ARRIVAL_TIME, BLOCKED_HOLD, CLOSE_FRAMES,
                                                              LANDED_FRAMES, LANDED_MARGIN, LANE_CHANGE_T_FOLLOW, LANE_WIDTH_DEFAULT, LEAD_BESIDE,
                                                              LEAD_BRAKING, LEAD_CONTINUITY, LEAD_DANGER_FACTOR, LEAD_OFF_PATH, LEAD_ON_PATH,
                                                              LEAD_SLOWING, LEAD_SPEED_CONTINUITY, LEAD_SPEED_FRAMES, LEAD_UNPLACED_STD,
                                                              LEAD_UNPLACED_STD_HOLD, LINE_HOLD, LINE_OFFSET_MIN, MIN_HEADROOM,
                                                              RADAR_TO_CAMERA, RELAX_TIME_MAX, ROADSIDE_MARGIN, TRACK_MIN_DISTANCE, TRACK_MOVING_SPEED,
                                                              too_close)
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
              curvature=0.0, lead_accel=0.0, lead_prob=1.0, path=None, lead_std=None, lead_two_std=None, lead_two_accel=0.0,
              line_stds=None, lines=None, lead_y=None, lead_two_y=None):
  model = log.ModelDataV2.new_message()
  model.init('leadsV3', 1 if lead_two_std is None else 2)
  model.leadsV3[0].prob = lead_prob
  model.leadsV3[0].a = [lead_accel] * len(ModelConstants.LEAD_T_IDXS)
  if lead_std is not None:
    model.leadsV3[0].yStd = [lead_std] * len(ModelConstants.LEAD_T_IDXS)
  if lead_y is not None:
    # where the model places its lead, right positive
    model.leadsV3[0].y = [lead_y] * len(ModelConstants.LEAD_T_IDXS)
  if lead_two_std is not None:
    model.leadsV3[1].prob = lead_prob
    model.leadsV3[1].a = [lead_two_accel] * len(ModelConstants.LEAD_T_IDXS)
    model.leadsV3[1].yStd = [lead_two_std] * len(ModelConstants.LEAD_T_IDXS)
    if lead_two_y is not None:
      model.leadsV3[1].y = [lead_two_y] * len(ModelConstants.LEAD_T_IDXS)
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
  if path is not None:
    # the planned path, right positive, reaching path meters over at 30 m and following the road's bend
    model.position.x = X_IDXS.tolist()
    model.position.y = (path * X_IDXS / 30.0 + curvature * X_IDXS**2 / 2).tolist()
  return model


def path_for(offset, d_rel, direction=LaneChangeDirection.left):
  # the path that puts a lead straight ahead at d_rel offset meters off it, toward the lane being left
  sign = -1.0 if direction == LaneChangeDirection.left else 1.0
  return sign * offset * 30.0 / (d_rel + RADAR_TO_CAMERA)


def set_lead(lead, present=True, radar=True, track_id=7, d_rel=35.0, v_rel=-1.0, a_lead=0.0, y_rel=0.0, v_ego=V_EGO):
  lead.present = present
  lead.radar = radar
  lead.radarTrackId = track_id
  lead.dRel = d_rel
  lead.yRel = y_rel
  lead.vRel = v_rel
  lead.vLead = v_ego + v_rel
  lead.aLeadK = a_lead


def get_radar_state(present=True, radar=True, track_id=7, d_rel=35.0, v_rel=-1.0, a_lead=0.0, y_rel=0.0, lead_two=None, v_ego=V_EGO):
  # lead_two: None for no second lead, 'same' for the model's usual mirror of the first, or set_lead() arguments
  radar_state = log.RadarState.new_message()
  set_lead(radar_state.leadOne, present, radar, track_id, d_rel, v_rel, a_lead, y_rel, v_ego)
  if lead_two == 'same':
    set_lead(radar_state.leadTwo, present, radar, track_id, d_rel, v_rel, a_lead, y_rel, v_ego)
  elif lead_two is not None:
    set_lead(radar_state.leadTwo, **lead_two)
  return radar_state


def get_tracks(*points):
  # (dRel, yRel, vRel) or (dRel, yRel, vRel, trackId)
  tracks = car.RadarData.new_message()
  tracks.init('points', len(points))
  for pt, point in zip(tracks.points, points, strict=True):
    pt.dRel, pt.yRel, pt.vRel = point[:3]
    pt.trackId = point[3] if len(point) > 3 else 0
  return tracks


def get_car_state(left_blinker=True, **kwargs):
  return car.CarState.new_message(vEgo=V_EGO, leftBlinker=left_blinker, **kwargs)


def blocked(tracks, direction=LaneChangeDirection.left, model=None, v_ego=V_EGO):
  lines = LaneLines(model if model is not None else get_model())
  return target_lane_blocked(direction, lines, tracks, v_ego, T_FOLLOW, STOP_DISTANCE, COMFORT_BRAKE)


def step(gap, model=None, CS=None, radar_state=None, tracks=None, radar_ok=True, v_cruise=V_EGO + 5.0, t_follow=T_FOLLOW, n=1, v_ego=V_EGO):
  model = model if model is not None else get_model()
  CS = CS if CS is not None else get_car_state()
  radar_state = radar_state if radar_state is not None else get_radar_state()
  tracks = tracks if tracks is not None else get_tracks()
  pad = 0.0
  for _ in range(n):
    pad = gap.update(model, CS, radar_state, tracks, radar_ok, v_ego, v_cruise, t_follow, STOP_DISTANCE, COMFORT_BRAKE)
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


FAR = 60.0  # m, a lead the MPC could take back gently at V_EGO and T_FOLLOW


def released(gap):
  return list(gap.released)


def change_right(**kwargs):
  return get_model(direction=LaneChangeDirection.right, **kwargs)


def right_blinker(**kwargs):
  return get_car_state(left_blinker=False, rightBlinker=True, **kwargs)


def on_path(d_rel, path):
  # the radar lateral of a lead sitting on a path that reaches path meters over at 30 m
  return -path * (d_rel + RADAR_TO_CAMERA) / 30.0


class TestLeadRelease:
  def test_off_path_radar_lead_is_released_in_both_slots(self):
    gap = LaneChangeGap(CP, DT_MDL)
    radar_state = get_radar_state(d_rel=FAR, lead_two='same')
    step(gap, model=get_model(path=path_for(LEAD_OFF_PATH - 0.1, FAR)), radar_state=radar_state)
    assert released(gap) == [False, False]
    step(gap, model=get_model(path=path_for(LEAD_OFF_PATH + 0.1, FAR)), radar_state=radar_state)
    assert released(gap) == [True, True]
    followed = gap.followed(radar_state)
    assert not followed.leadOne.present and not followed.leadTwo.present
    # radarState itself stays as radard published it
    assert radar_state.leadOne.present and radar_state.leadTwo.present

  def test_only_toward_the_lane_being_left(self):
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=get_model(path=path_for(-1.5, FAR)), radar_state=get_radar_state(d_rel=FAR), n=5)
    assert released(gap) == [False, False]
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=change_right(path=path_for(1.5, FAR, LaneChangeDirection.right)), CS=right_blinker(), radar_state=get_radar_state(d_rel=FAR))
    assert released(gap)[0]

  def test_hysteresis_holds_the_same_car_only(self):
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=get_model(path=path_for(LEAD_OFF_PATH + 0.2, FAR)), radar_state=get_radar_state(d_rel=FAR))
    assert released(gap)[0]
    step(gap, model=get_model(path=path_for(LEAD_ON_PATH + 0.1, FAR)), radar_state=get_radar_state(d_rel=FAR))
    assert released(gap)[0]
    step(gap, model=get_model(path=path_for(LEAD_ON_PATH - 0.1, FAR)), radar_state=get_radar_state(d_rel=FAR))
    assert not released(gap)[0]
    step(gap, model=get_model(path=path_for(LEAD_ON_PATH + 0.1, FAR)), radar_state=get_radar_state(d_rel=FAR))
    assert not released(gap)[0]
    # the same car under a new track id keeps it
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=get_model(path=path_for(LEAD_OFF_PATH + 0.2, FAR)), radar_state=get_radar_state(d_rel=FAR))
    step(gap, model=get_model(path=path_for(LEAD_ON_PATH + 0.1, FAR)), radar_state=get_radar_state(track_id=8, d_rel=FAR))
    assert released(gap)[0]
    # a different car in the slot has to earn its own release
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=get_model(path=path_for(LEAD_OFF_PATH + 0.2, FAR)), radar_state=get_radar_state(d_rel=FAR))
    assert released(gap)[0]
    step(gap, model=get_model(path=path_for(LEAD_ON_PATH + 0.1, FAR + 10.0)), radar_state=get_radar_state(track_id=8, d_rel=FAR + 10.0))
    assert not released(gap)[0]

  def test_lead_moving_over_the_same_way_is_followed_again(self):
    # the path leaves the lead before the lead itself moves toward the target lane; back on the path, it is in front again
    path = path_for(1.5, FAR)
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=get_model(path=path), radar_state=get_radar_state(d_rel=FAR))
    assert released(gap)[0]
    step(gap, model=get_model(path=path), radar_state=get_radar_state(d_rel=FAR, y_rel=on_path(FAR, path) - (LEAD_ON_PATH + 0.1)))
    assert released(gap)[0]
    radar_state = get_radar_state(d_rel=FAR, y_rel=on_path(FAR, path) - (LEAD_ON_PATH - 0.1))
    step(gap, model=get_model(path=path), radar_state=radar_state)
    assert released(gap) == [False, False] and gap.followed(radar_state) is radar_state
    step(gap, model=get_model(path=path), radar_state=get_radar_state(d_rel=FAR, y_rel=on_path(FAR, path)), n=5)
    assert released(gap) == [False, False]

  def test_no_planned_path_releases_nothing(self):
    radar_state = get_radar_state(d_rel=FAR, y_rel=-1.5)
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=get_model(path=0.0), radar_state=radar_state)
    assert released(gap)[0]
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, radar_state=radar_state, n=5)
    assert released(gap) == [False, False]

  def test_path_turning_back_is_read_out_to_its_farthest_point(self):
    # a left turn at a junction taken as a change: the path runs round a 20 m radius until it heads back toward the car,
    # and a car turning ahead sits on it 70 degrees round
    radius, turned = 20.0, np.radians(70.0)
    arc = np.linspace(0.0, 0.75 * np.pi, len(X_IDXS))
    model = get_model()
    model.position.x = (radius * np.sin(arc)).tolist()
    model.position.y = (-radius * (1 - np.cos(arc))).tolist()
    radar_state = get_radar_state(d_rel=float(radius * np.sin(turned)) - RADAR_TO_CAMERA, y_rel=float(radius * (1 - np.cos(turned))), v_rel=-0.5, v_ego=6.0)
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=model, radar_state=radar_state, v_ego=6.0, n=5)
    assert released(gap) == [False, False]

  def test_stopped_or_crawling_lead_is_never_released(self):
    # in town a crawling car 100 m out asks the MPC for little braking, so only its own speed keeps it followed
    v_ego, d_rel = 12.0, 100.0
    for v_lead, followed in ((0.0, True), (TRACK_MOVING_SPEED - 0.1, True), (TRACK_MOVING_SPEED + 0.1, False)):
      assert not too_close(d_rel, v_lead - v_ego, v_lead, 0.0, T_FOLLOW, STOP_DISTANCE)
      gap = LaneChangeGap(CP, DT_MDL)
      step(gap, model=get_model(path=path_for(3.0, d_rel)), radar_state=get_radar_state(d_rel=d_rel, v_rel=v_lead - v_ego, v_ego=v_ego), v_ego=v_ego, n=5)
      assert released(gap)[0] != followed

  def test_lead_the_mpc_could_not_take_back_is_followed_until_the_car_is_beside(self):
    model = get_model(path=path_for(2.0, FAR))
    # inside the follow distance
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=model, radar_state=get_radar_state(d_rel=FOLLOW_GAP - 5.0), n=5)
    assert released(gap) == [False, False]
    # released far out, then closing: held for CLOSE_FRAMES - 1 frames, then followed at the full gap
    gap = LaneChangeGap(CP, DT_MDL)
    assert np.isclose(step(gap, model=model, radar_state=get_radar_state(d_rel=FAR)), PAD)
    assert released(gap)[0]
    closing = get_radar_state(d_rel=45.0, v_rel=-4.0)
    step(gap, model=model, radar_state=closing, n=CLOSE_FRAMES - 1)
    assert released(gap)[0]
    assert step(gap, model=model, radar_state=closing) == 0.0
    assert not released(gap)[0] and gap.closing
    assert step(gap, model=model, radar_state=get_radar_state(d_rel=FAR), n=5) == 0.0
    assert not released(gap)[0]
    # the car being passed beside the car for CLOSE_FRAMES running lifts it, and a lead beside is released however close
    beside = get_radar_state(d_rel=20.0, v_rel=-4.0, y_rel=-LEAD_BESIDE)
    step(gap, model=get_model(path=0.0), radar_state=beside, tracks=get_tracks((20.0, -LEAD_BESIDE, -4.0, 7)), n=CLOSE_FRAMES - 1)
    assert gap.closing and released(gap)[0]
    step(gap, model=get_model(path=0.0), radar_state=beside, tracks=get_tracks((20.0, -LEAD_BESIDE, -4.0, 7)))
    assert not gap.closing and released(gap)[0]

  def test_floor_ends_the_relaxation_only_after_a_release(self):
    # with nothing released the relaxation runs as it would without the release; once something has been released, the
    # leads the floor hands back are followed at the full gap
    model = get_model(path=path_for(2.0, FAR))
    closing = get_radar_state(d_rel=45.0, v_rel=-4.0)
    gap = LaneChangeGap(CP, DT_MDL)
    assert np.isclose(step(gap, model=model, radar_state=closing, n=CLOSE_FRAMES + 5), PAD)
    assert gap.closing and released(gap) == [False, False]
    gap = LaneChangeGap(CP, DT_MDL)
    assert np.isclose(step(gap, model=model, radar_state=get_radar_state(d_rel=FAR)), PAD)
    assert released(gap)[0]
    assert step(gap, model=model, radar_state=closing, n=CLOSE_FRAMES) == 0.0
    assert gap.closing and not gap.armed
    # the next change starts with nothing released
    step(gap, model=get_model(state=LaneChangeState.off))
    assert np.isclose(step(gap, model=model, radar_state=closing, n=CLOSE_FRAMES + 5), PAD)
    assert gap.closing

  def test_car_being_passed_in_no_slot_is_watched_by_the_floor(self):
    # the model has moved its lead to a car far out that the path has left, while the car being passed, track 7, is in
    # neither slot and still overlaps the car
    path = path_for(2.0, FAR)
    model = get_model(path=path)
    far_lead = (80.0, on_path(80.0, path) - 2.0, -1.0, 8)
    leads = get_radar_state(track_id=8, d_rel=far_lead[0], y_rel=far_lead[1])

    def passed(d_rel, y_rel=0.0):
      return get_tracks((d_rel, y_rel, -4.0, 7), far_lead)

    def handed_over(d_rel):
      gap = LaneChangeGap(CP, DT_MDL)
      step(gap, model=get_model(path=0.0), radar_state=get_radar_state(d_rel=d_rel, v_rel=-4.0), tracks=passed(d_rel))
      step(gap, model=model, radar_state=leads, tracks=passed(d_rel), n=CLOSE_FRAMES - 1)
      return gap

    # judged at its own speed: 52 m behind a car closing at 4 m/s the MPC could still take it back gently
    gap = handed_over(52.0)
    step(gap, model=model, radar_state=leads, tracks=passed(52.0), n=5)
    assert released(gap)[0] and not gap.closing
    # at 45 m it could not: the released lead is handed back after CLOSE_FRAMES running and followed at the full gap
    gap = handed_over(45.0)
    assert released(gap)[0]
    step(gap, model=model, radar_state=leads, tracks=passed(45.0))
    assert released(gap) == [False, False] and gap.closing
    # a car beside the car is released through the latch
    step(gap, model=model, radar_state=get_radar_state(track_id=9, d_rel=20.0, y_rel=-(LEAD_BESIDE + 0.5)), tracks=passed(45.0))
    assert released(gap)[0] and gap.closing
    # one wide radar frame is not the car being passed beside
    step(gap, model=model, radar_state=leads, tracks=passed(45.0, y_rel=-(LEAD_BESIDE + 0.1)))
    step(gap, model=model, radar_state=leads, tracks=passed(45.0), n=5)
    assert released(gap) == [False, False] and gap.closing
    # beside for CLOSE_FRAMES running, it lifts the latch and the floor stops watching it
    step(gap, model=model, radar_state=leads, tracks=passed(45.0, y_rel=-(LEAD_BESIDE + 0.5)), n=CLOSE_FRAMES)
    assert released(gap)[0] and not gap.closing
    # in a slot it is judged there: followed on the path, it does not hand the second lead back
    on_the_path = on_path(45.0, path)
    both = get_radar_state(d_rel=45.0, v_rel=-4.0, y_rel=on_the_path, lead_two={'track_id': 8, 'd_rel': far_lead[0], 'y_rel': far_lead[1]})
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=model, radar_state=both, tracks=get_tracks((45.0, on_the_path, -4.0, 7), far_lead), n=5)
    assert released(gap) == [False, True] and not gap.closing

  def test_floor_holds_only_the_car_it_released(self):
    # a different car too close in a slot released last frame is new to the floor, not a glitch of the released one
    model = get_model(path=path_for(2.0, FAR))
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=model, radar_state=get_radar_state(d_rel=FAR))
    assert released(gap)[0]
    d_rel = FOLLOW_GAP - 8.0
    step(gap, model=model, radar_state=get_radar_state(track_id=8, d_rel=d_rel, v_rel=-3.0, y_rel=on_path(d_rel, path_for(2.0, FAR)) - 2.0))
    assert released(gap) == [False, False]

  def test_floor_counts_frames_running(self):
    # close frames on either side of a blocked target lane are not frames running: the count starts again on the frame the
    # block ends, the first one the floor judges
    model = get_model(path=path_for(2.0, FAR))
    closing = get_radar_state(d_rel=45.0, v_rel=-4.0)
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=model, radar_state=get_radar_state(d_rel=FAR))
    step(gap, model=model, radar_state=closing, n=CLOSE_FRAMES - 1)
    step(gap, model=model, radar_state=closing, tracks=get_tracks((30.0, 3.4, -1.0)))
    while gap.blocked_timer > 0.0:
      step(gap, model=model, radar_state=closing)
    step(gap, model=model, radar_state=closing, n=CLOSE_FRAMES - 2)
    assert not gap.closing
    step(gap, model=model, radar_state=closing)
    assert gap.closing

  def test_lead_inside_the_follow_distance_is_too_close_only_while_closing(self):
    # a car holding or opening the gap asks the MPC for little braking, unless it is inside the MPC's danger distance
    d_rel = FOLLOW_GAP - 5.0
    assert d_rel > LEAD_DANGER_FACTOR * FOLLOW_GAP
    model = get_model(path=path_for(1.5, d_rel))
    for v_rel, followed in ((0.0, False), (0.5, False), (-0.3, True)):
      gap = LaneChangeGap(CP, DT_MDL)
      step(gap, model=model, radar_state=get_radar_state(d_rel=d_rel, v_rel=v_rel), n=5)
      assert released(gap)[0] != followed
    # the car accelerating toward it will be closing once handed back
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=model, radar_state=get_radar_state(d_rel=d_rel, v_rel=0.0), CS=get_car_state(aEgo=0.5), n=5)
    assert not released(gap)[0]
    d_rel = LEAD_DANGER_FACTOR * FOLLOW_GAP - 1.0
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=get_model(path=path_for(1.5, d_rel)), radar_state=get_radar_state(d_rel=d_rel, v_rel=0.5), n=5)
    assert not released(gap)[0]

  def test_accelerating_car_is_judged_where_it_will_be(self):
    model = get_model(path=path_for(2.0, 50.0))
    radar_state = get_radar_state(d_rel=50.0, v_rel=-2.0)
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=model, radar_state=radar_state, CS=get_car_state(aEgo=0.0))
    assert released(gap)[0]
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=model, radar_state=radar_state, CS=get_car_state(aEgo=2.0))
    assert not released(gap)[0]

  def test_beside_is_measured_not_planned(self):
    # the plan swinging 3 m over does not put a car that is still dead ahead beside the car
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=get_model(path=path_for(3.0, 15.0)), radar_state=get_radar_state(d_rel=15.0, v_rel=-3.0), n=5)
    assert released(gap) == [False, False]
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=get_model(path=0.0), radar_state=get_radar_state(d_rel=15.0, v_rel=-3.0, y_rel=-LEAD_BESIDE))
    assert released(gap)[0]

  def test_car_beside_is_judged_again_once_the_path_swings_back_toward_it(self):
    # the model gives up the change without a cue from the driver once the car is beside a slower one, and the path comes
    # back toward it while the car is still over
    d_rel, v_rel, y_rel = 15.0, -3.0, -(LEAD_BESIDE + 0.5)
    radar_state = get_radar_state(d_rel=d_rel, v_rel=v_rel, y_rel=y_rel, lead_two='same')
    tracks = get_tracks((d_rel, y_rel, v_rel, 7))
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=get_model(path=path_for(1.5, d_rel)), radar_state=radar_state, tracks=tracks, n=5)
    assert released(gap) == [True, True] and gap.moved_over
    back = get_model(path=path_for(LEAD_BESIDE - 0.1 + y_rel, d_rel))
    step(gap, model=back, radar_state=radar_state, tracks=tracks, n=CLOSE_FRAMES - 1)
    assert released(gap) == [True, True]
    step(gap, model=back, radar_state=radar_state, tracks=tracks)
    assert released(gap) == [False, False] and gap.closing

  def test_car_in_its_lane_on_a_bend_is_not_beside(self):
    # a left change on a right hand bend: the car ahead in the lane reads more than LEAD_BESIDE to the right of the
    # car's heading, but sits on the lane like the line being crossed
    d_rel = 20.0
    curvature = 2 * (LEAD_BESIDE + 0.2) / (d_rel + RADAR_TO_CAMERA)**2
    y_rel = -curvature * (d_rel + RADAR_TO_CAMERA)**2 / 2
    radar_state = get_radar_state(d_rel=d_rel, y_rel=y_rel)
    tracks = get_tracks((d_rel, y_rel, -1.0, 7))
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=get_model(curvature=curvature, path=path_for(1.5, d_rel)), radar_state=radar_state, tracks=tracks, n=5)
    assert released(gap) == [False, False] and not gap.moved_over
    # the same lateral on a straight road is a car beside
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=get_model(path=path_for(1.5, d_rel)), radar_state=radar_state, tracks=tracks, n=CLOSE_FRAMES)
    assert released(gap)[0] and gap.moved_over

  def test_track_of_a_car_in_the_next_lane_is_not_the_leads(self):
    # radard matches the model's lead to a radar track on range and speed alone: with no return from the lead itself, a
    # motorcycle say, it comes with the track of a car in the lane being left, 3.3 m to the side of it
    d_rel, y_rel = 30.0, LEAD_BESIDE + 1.3
    radar_state = get_radar_state(d_rel=d_rel, v_rel=-5.0, y_rel=y_rel, lead_two='same')
    tracks = get_tracks((d_rel, y_rel, -5.0, 7))
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=change_right(path=0.0, lead_std=0.2, lead_y=0.0), CS=right_blinker(), radar_state=radar_state, tracks=tracks, n=5)
    assert released(gap) == [False, False]
    assert gap.passing_id == -1 and not gap.moved_over
    # where the model puts its lead there too, or cannot place it, the track is the lead: beside, and released however close
    for lead_std, lead_y in ((0.2, -y_rel), (LEAD_UNPLACED_STD_HOLD, 0.0)):
      gap = LaneChangeGap(CP, DT_MDL)
      step(gap, model=change_right(path=0.0, lead_std=lead_std, lead_y=lead_y), CS=right_blinker(), radar_state=radar_state, tracks=tracks)
      assert released(gap) == [True, True] and gap.passing_id == 7
    # released beside where the model put its lead, then with the model's lead back on the path, it is no car beside
    # and the floor judges it again
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=change_right(path=0.0, lead_std=0.2, lead_y=-y_rel), CS=right_blinker(), radar_state=radar_state, tracks=tracks)
    back = change_right(path=0.0, lead_std=0.2, lead_y=0.0)
    step(gap, model=back, CS=right_blinker(), radar_state=radar_state, tracks=tracks, n=CLOSE_FRAMES - 1)
    assert released(gap) == [True, True]
    step(gap, model=back, CS=right_blinker(), radar_state=radar_state, tracks=tracks)
    assert released(gap) == [False, False] and gap.closing
    # far out, where the floor would let it go, the track is still no new release for a lead the model places elsewhere
    far = get_radar_state(d_rel=FAR, y_rel=y_rel, lead_two='same')
    far_tracks = get_tracks((FAR, y_rel, -1.0, 7))
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=change_right(path=0.0, lead_std=0.2, lead_y=0.0), CS=right_blinker(), radar_state=far, tracks=far_tracks, n=5)
    assert released(gap) == [False, False]
    # a car released where the model put its lead holds its release on the path when the model's lead moves off it, as the
    # floor watches it
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=change_right(path=0.0, lead_std=0.2, lead_y=-y_rel), CS=right_blinker(), radar_state=far, tracks=far_tracks)
    step(gap, model=change_right(path=0.0, lead_std=0.2, lead_y=0.0), CS=right_blinker(), radar_state=far, tracks=far_tracks, n=5)
    assert released(gap) == [True, True]

  def test_car_being_passed_is_followed_across_a_track_id_change(self):
    path = path_for(1.5, FAR)
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=get_model(path=0.0), radar_state=get_radar_state(d_rel=FAR), tracks=get_tracks((FAR, 0.0, -1.0, 7)))
    step(gap, model=get_model(path=path), radar_state=get_radar_state(present=False), tracks=get_tracks((FAR - DT_MDL, 0.0, -1.0, 7)))
    assert gap.leaving
    step(gap, model=get_model(path=path), radar_state=get_radar_state(present=False), tracks=get_tracks((FAR - 2 * DT_MDL, 0.3, -1.0, 8)))
    assert gap.passing_id == 8
    # the path coming back to the car under its new track
    step(gap, model=get_model(path=0.0), radar_state=get_radar_state(present=False), tracks=get_tracks((FAR - 3 * DT_MDL, 0.3, -1.0, 8)))
    assert not gap.leaving
    # unseen for a second, it is looked for where it will have got to
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=get_model(path=0.0), radar_state=get_radar_state(d_rel=FAR, v_rel=-5.0), tracks=get_tracks((FAR, 0.0, -5.0, 7)))
    unseen = int(round(1.0 / DT_MDL))
    step(gap, model=get_model(path=path), radar_state=get_radar_state(present=False), n=unseen)
    d_rel = FAR - 5.0 * (unseen + 1) * DT_MDL
    step(gap, model=get_model(path=path), radar_state=get_radar_state(present=False), tracks=get_tracks((d_rel, 0.3, -5.0, 8)))
    assert gap.passing_id == 8

  def test_target_lane_car_does_not_continue_the_car_being_passed(self):
    # a car in the target lane at the range and speed the car being passed would have
    path = path_for(1.5, FAR)
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=get_model(path=0.0), radar_state=get_radar_state(d_rel=FAR), tracks=get_tracks((FAR, 0.0, -1.0, 7)))
    step(gap, model=get_model(path=path), radar_state=get_radar_state(present=False), tracks=get_tracks((FAR - DT_MDL, 0.0, -1.0, 7)))
    assert gap.leaving
    step(gap, model=get_model(path=path), radar_state=get_radar_state(present=False), tracks=get_tracks((FAR - 2 * DT_MDL, 3.4, -1.0, 9)))
    assert gap.passing_id == 7 and gap.leaving

  def vision_after_handoff(self, gap, passed_offset, lead_std=1.5, lead_offset=LEAD_OFF_PATH + 0.5, d_rel=70.0, tracks=(), left_line=-1.6, n=1,
                           lead_two_offset=None, probs=(1.0, 1.0), line_stds=None):
    # the change starts behind radar track 7 at FAR; the model has since handed its lead to one only it sees, lead_offset
    # meters off the path toward the lane being left, and optionally a second one a meter further at the same speed
    path = path_for(passed_offset, FAR)
    lead_two = None
    if lead_two_offset is not None:
      lead_two = {'radar': False, 'track_id': -1, 'd_rel': d_rel + 1.0, 'y_rel': on_path(d_rel + 1.0, path) - lead_two_offset}
    radar_state = get_radar_state(radar=False, track_id=-1, d_rel=d_rel, y_rel=on_path(d_rel, path) - lead_offset, lead_two=lead_two)
    model = get_model(path=path, lead_std=lead_std, lead_two_std=None if lead_two is None else lead_std, left_y=left_line, right_y=left_line + 3.36,
                      probs=probs, line_stds=line_stds)
    step(gap, model=model, radar_state=radar_state, tracks=get_tracks((FAR, 0.0, -1.0, 7), *tracks), n=n)
    return released(gap)[0]

  def start_behind_track_7(self):
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=get_model(path=0.0), radar_state=get_radar_state(d_rel=FAR), tracks=get_tracks((FAR, 0.0, -1.0, 7)))
    return gap

  def test_vision_lead_needs_the_car_being_passed_off_the_path(self):
    gap = self.start_behind_track_7()
    assert not self.vision_after_handoff(gap, LEAD_OFF_PATH - 0.1)
    assert self.vision_after_handoff(gap, LEAD_OFF_PATH + 0.1)
    assert gap.leaving
    assert self.vision_after_handoff(gap, (LEAD_ON_PATH + LEAD_OFF_PATH) / 2)
    assert gap.leaving
    # the path coming back to the car being passed
    assert not self.vision_after_handoff(gap, LEAD_ON_PATH - 0.1)
    assert not gap.leaving
    # with the car being passed gone from the radar, nothing shows the car moving over, however clear the path
    gap = self.start_behind_track_7()
    path = path_for(1.5, FAR)
    vision = get_radar_state(radar=False, track_id=-1, d_rel=70.0, y_rel=on_path(70.0, path) - (LEAD_OFF_PATH + 0.5))
    step(gap, model=get_model(path=path, lead_std=1.5), radar_state=vision, n=5)
    assert released(gap) == [False, False] and not gap.leaving

  def test_vision_lead_the_model_can_place_is_followed(self):
    gap = self.start_behind_track_7()
    assert not self.vision_after_handoff(gap, 1.5, lead_std=LEAD_UNPLACED_STD - 0.1)
    assert self.vision_after_handoff(gap, 1.5, lead_std=LEAD_UNPLACED_STD + 0.1)
    assert self.vision_after_handoff(gap, 1.5, lead_std=LEAD_UNPLACED_STD_HOLD + 0.05)
    assert not self.vision_after_handoff(gap, 1.5, lead_std=LEAD_UNPLACED_STD_HOLD - 0.05)
    assert not self.vision_after_handoff(gap, 1.5, lead_std=LEAD_UNPLACED_STD_HOLD + 0.05)

  def test_slot_released_on_a_radar_lead_holds_nothing_for_the_next(self):
    # the model's lead taking over a slot released on a radar track is a new lead there, held to the release thresholds
    for lead_offset, lead_std in (((LEAD_ON_PATH + LEAD_OFF_PATH) / 2, 1.5), (1.5, (LEAD_UNPLACED_STD_HOLD + LEAD_UNPLACED_STD) / 2)):
      gap = LaneChangeGap(CP, DT_MDL)
      step(gap, model=get_model(path=path_for(1.5, FAR)), radar_state=get_radar_state(d_rel=FAR), tracks=get_tracks((FAR, 0.0, -1.0, 7)))
      assert released(gap)[0]
      assert not self.vision_after_handoff(gap, 1.5, lead_offset=lead_offset, lead_std=lead_std)
      # released on a lead only the model sees, the slot holds it there
      gap = self.start_behind_track_7()
      assert self.vision_after_handoff(gap, 1.5)
      assert self.vision_after_handoff(gap, 1.5, lead_offset=lead_offset, lead_std=lead_std)

  def test_release_stays_with_the_car_across_the_models_reading_of_it(self):
    # radard can swap a released car's radar track in its slot for the model's reading of the same car, which the model
    # puts on the path
    path = path_for(1.5, FAR)
    model = get_model(path=path, lead_std=1.5, lead_two_std=1.5)

    def track(k):
      return get_tracks((FAR - k * DT_MDL, 0.0, -1.0, 7))

    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=model, radar_state=get_radar_state(d_rel=FAR, lead_two='same'), tracks=track(0))
    assert released(gap) == [True, True]
    for k in range(1, 6):
      # the model reads the car short, within LEAD_CONTINUITY of where radar has it
      d_rel = FAR - k * DT_MDL - (LEAD_CONTINUITY - 0.05)
      camera = {'radar': False, 'track_id': -1, 'd_rel': d_rel, 'y_rel': on_path(d_rel, path)}
      radar_state = get_radar_state(**camera, lead_two=camera)
      step(gap, model=model, radar_state=radar_state, tracks=track(k))
      assert released(gap) == [True, True]
      assert not gap.followed(radar_state).leadOne.present and not gap.followed(radar_state).leadTwo.present
    # back on its track it is still the car released
    step(gap, model=model, radar_state=get_radar_state(d_rel=FAR - 6 * DT_MDL, lead_two='same'), tracks=track(6))
    assert released(gap) == [True, True]
    # the path coming back to where radar has it ends the release through the model's reading too
    camera = {'radar': False, 'track_id': -1, 'd_rel': FAR - 7 * DT_MDL, 'y_rel': 0.0}
    step(gap, model=get_model(path=0.0, lead_std=1.5, lead_two_std=1.5), radar_state=get_radar_state(**camera, lead_two=camera), tracks=track(7))
    assert released(gap) == [False, False]

  def test_models_reading_of_another_car_is_judged_on_its_own(self):
    path = path_for(1.5, FAR)
    model = get_model(path=path, lead_std=1.5, lead_two_std=1.5)
    tracks = get_tracks((FAR, 0.0, -1.0, 7))
    # a reading on the path at another distance or speed than the released car's is a car ahead
    for d_rel, v_rel in ((FAR - DT_MDL - LEAD_CONTINUITY - 0.1, -1.0), (FAR - DT_MDL, -1.0 + LEAD_SPEED_CONTINUITY + 0.1)):
      gap = LaneChangeGap(CP, DT_MDL)
      step(gap, model=model, radar_state=get_radar_state(d_rel=FAR), tracks=tracks)
      assert released(gap)[0]
      camera = get_radar_state(radar=False, track_id=-1, d_rel=d_rel, v_rel=v_rel, y_rel=on_path(d_rel, path))
      step(gap, model=model, radar_state=camera, tracks=tracks)
      assert released(gap) == [False, False] and gap.followed(camera).leadOne.present
    # and so is one in the other slot: radard swapped nothing there
    gap = LaneChangeGap(CP, DT_MDL)
    lead_two = {'radar': False, 'track_id': -1, 'd_rel': FAR + 0.5, 'y_rel': on_path(FAR + 0.5, path)}
    step(gap, model=model, radar_state=get_radar_state(d_rel=FAR, lead_two=lead_two), tracks=tracks, n=3)
    assert released(gap) == [True, False]

  def test_floor_judges_the_car_across_the_models_reading_of_it(self):
    # on radar's reading of the car while radar still tracks it, and on the model's reading once radar has lost it; here
    # the model reads the car further out and closing slower, where the floor would let it go
    path = path_for(1.5, FAR)
    model = get_model(path=path, lead_std=1.5)
    v_rel = -4.0
    for radar_has_it, still_released in ((True, CLOSE_FRAMES - 1), (False, 5)):
      gap = LaneChangeGap(CP, DT_MDL)
      step(gap, model=model, radar_state=get_radar_state(d_rel=FAR, v_rel=v_rel), tracks=get_tracks((FAR, 0.0, v_rel, 7)))
      assert released(gap)[0]
      for k in range(1, 6):
        d_car = FAR + k * v_rel * DT_MDL
        d_rel = d_car + LEAD_CONTINUITY - 0.5
        camera = get_radar_state(radar=False, track_id=-1, d_rel=d_rel, v_rel=v_rel + LEAD_SPEED_CONTINUITY - 0.1, y_rel=on_path(d_rel, path))
        tracks = get_tracks((d_car, 0.0, v_rel, 7)) if radar_has_it else get_tracks()
        step(gap, model=model, radar_state=camera, tracks=tracks, CS=get_car_state(aEgo=2.0))
        assert released(gap)[0] == (k <= still_released)
      assert gap.closing == (still_released < 5)
    # once radar has lost it, a reading too close is handed back like any released car. The car released here is not the
    # one the change started behind, which the floor also watches outside the slots
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=model, radar_state=get_radar_state(present=False))
    step(gap, model=model, radar_state=get_radar_state(track_id=9, d_rel=FAR, v_rel=v_rel), tracks=get_tracks((FAR, 0.0, v_rel, 9)))
    assert released(gap)[0] and gap.passing_id == -1
    for k in range(1, CLOSE_FRAMES + 1):
      d_rel = FAR + k * v_rel * DT_MDL - (LEAD_CONTINUITY - 0.5)
      camera = get_radar_state(radar=False, track_id=-1, d_rel=d_rel, v_rel=v_rel - (LEAD_SPEED_CONTINUITY - 0.1), y_rel=on_path(d_rel, path))
      step(gap, model=model, radar_state=camera, CS=get_car_state(aEgo=1.5))
      assert released(gap)[0] == (k < CLOSE_FRAMES)
    assert gap.closing

  def test_car_is_judged_on_radars_reading_across_the_models_reading_of_it(self):
    # while radar still tracks the car, its reading this frame is the car, whatever radard shows in its slot: one moving
    # over the same way is followed again once radar has it back on the path
    path = path_for(1.5, FAR)
    model = get_model(path=path, lead_std=1.5)
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=model, radar_state=get_radar_state(d_rel=FAR), tracks=get_tracks((FAR, 0.0, -1.0, 7)))
    for k, off_path in ((1, LEAD_ON_PATH + 0.1), (2, LEAD_ON_PATH - 0.1)):
      d_rel = FAR - k * DT_MDL
      camera = get_radar_state(radar=False, track_id=-1, d_rel=d_rel, y_rel=on_path(d_rel, path))
      step(gap, model=model, radar_state=camera, tracks=get_tracks((d_rel, on_path(d_rel, path) - off_path, -1.0, 7)))
      assert released(gap)[0] == (off_path > LEAD_ON_PATH)
    # and one slowing is handed back by the floor on radar's speed
    d_rel = 50.0
    path = path_for(1.5, d_rel)
    model = get_model(path=path, lead_std=1.5)
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=model, radar_state=get_radar_state(d_rel=d_rel), tracks=get_tracks((d_rel, 0.0, -1.0, 7)), CS=get_car_state(aEgo=1.0))
    assert released(gap)[0]
    for k in range(1, CLOSE_FRAMES + 1):
      d_k = d_rel - k * DT_MDL
      camera = get_radar_state(radar=False, track_id=-1, d_rel=d_k, v_rel=-2.4, y_rel=on_path(d_k, path))
      step(gap, model=model, radar_state=camera, tracks=get_tracks((d_k, 0.0, -3.5, 7)), CS=get_car_state(aEgo=1.0))
      assert released(gap)[0] == (k < CLOSE_FRAMES)

  def test_models_reading_drifting_off_the_car_is_judged_on_its_own(self):
    # once radar has lost the car, the model's reading of it can walk onto a car ahead a little each frame. The car is where
    # its last radar reading carried forward puts it, and a reading LEAD_CONTINUITY off that is judged on its own
    path = path_for(1.5, FAR)
    model = get_model(path=path, lead_std=1.5)
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=model, radar_state=get_radar_state(d_rel=FAR), tracks=get_tracks((FAR, 0.0, -1.0, 7)))
    assert released(gap)[0]
    drift = 0.8
    for k in range(1, 5):
      d_rel = FAR - k * DT_MDL - k * drift
      step(gap, model=model, radar_state=get_radar_state(radar=False, track_id=-1, d_rel=d_rel, y_rel=on_path(d_rel, path)))
      assert released(gap)[0] == (k * drift < LEAD_CONTINUITY)

  def test_car_beside_is_judged_by_the_floor_across_the_models_reading_of_it(self):
    # the lateral of the model's reading is no measure of a car beside: a released car radard swaps for it, here still
    # closing well inside the follow distance, is judged by the floor, whether or not radar still tracks it
    d_rel, v_rel, y_rel = 15.0, -3.0, -(LEAD_BESIDE + 0.5)
    path = path_for(1.5, d_rel)
    model = get_model(path=path, lead_std=1.5)
    for radar_has_it in (True, False):
      gap = LaneChangeGap(CP, DT_MDL)
      step(gap, model=model, radar_state=get_radar_state(d_rel=d_rel, v_rel=v_rel, y_rel=y_rel), tracks=get_tracks((d_rel, y_rel, v_rel, 7)))
      assert released(gap)[0]
      for k in range(1, CLOSE_FRAMES + 1):
        d_k = d_rel + k * v_rel * DT_MDL
        camera = get_radar_state(radar=False, track_id=-1, d_rel=d_k, v_rel=v_rel, y_rel=on_path(d_k, path))
        step(gap, model=model, radar_state=camera, tracks=get_tracks((d_k, y_rel, v_rel, 7)) if radar_has_it else get_tracks())
        assert released(gap)[0] == (k < CLOSE_FRAMES)
      assert gap.closing

  def test_vision_lead_needs_an_empty_radar_corridor(self):
    path = path_for(1.5, FAR)
    for v_rel in (-1.0, -V_EGO):  # moving or stationary
      gap = self.start_behind_track_7()
      assert not self.vision_after_handoff(gap, 1.5, tracks=[(40.0, on_path(40.0, path), v_rel, 3)])
      # past the lead, out to the far end of radard's match window for a lead at 70 m
      assert not self.vision_after_handoff(gap, 1.5, tracks=[(80.0, on_path(80.0, path), v_rel, 3)])
      assert self.vision_after_handoff(gap, 1.5, tracks=[(95.0, on_path(95.0, path), v_rel, 3)])
    # a return beside the car is the blind spot monitor's
    gap = self.start_behind_track_7()
    assert self.vision_after_handoff(gap, 1.5, tracks=[(TRACK_MIN_DISTANCE - 0.5, on_path(TRACK_MIN_DISTANCE - 0.5, path), -1.0, 3)])
    # a return 0.7 m off the path keeps a held release but blocks a new one
    gap = self.start_behind_track_7()
    near_path = [(40.0, on_path(40.0, path) - 0.7, -1.0, 3)]
    assert self.vision_after_handoff(gap, 1.5)
    assert self.vision_after_handoff(gap, 1.5, tracks=near_path)
    gap = self.start_behind_track_7()
    assert not self.vision_after_handoff(gap, 1.5, tracks=near_path)

  def test_vision_lead_on_the_path_is_followed(self):
    # the model's range on a lead only it sees can run 5 to 15 m long, so one on the path may be the car ahead: it is
    # released only off the path, and held there like a radar lead
    gap = self.start_behind_track_7()
    assert not self.vision_after_handoff(gap, 1.5, lead_offset=0.0, n=5)
    assert not self.vision_after_handoff(gap, 1.5, lead_offset=LEAD_OFF_PATH - 0.1)
    assert self.vision_after_handoff(gap, 1.5, lead_offset=LEAD_OFF_PATH + 0.1)
    assert self.vision_after_handoff(gap, 1.5, lead_offset=LEAD_ON_PATH + 0.1)
    assert not self.vision_after_handoff(gap, 1.5, lead_offset=LEAD_ON_PATH - 0.1)
    assert not self.vision_after_handoff(gap, 1.5, lead_offset=LEAD_ON_PATH + 0.1)

  def test_vision_lead_off_to_the_side_is_still_judged_by_the_floor(self):
    # the lateral of a lead the model cannot place is no measure of a car beside
    gap = self.start_behind_track_7()
    assert not self.vision_after_handoff(gap, 1.5, lead_offset=LEAD_BESIDE + 1.0, d_rel=20.0, n=CLOSE_FRAMES)

  def test_vision_lead_on_the_target_side_is_followed(self):
    gap = self.start_behind_track_7()
    assert not self.vision_after_handoff(gap, 1.5, lead_offset=-(LEAD_ON_PATH + 0.1), n=5)

  def test_target_lane_car_on_a_bend_is_followed(self):
    # camera check case 27: a left change on a road bending left, where the path runs a lane over through a car only
    # the model sees, 7.8 m left of the car's heading and pulling away
    d_rel = 38.3
    x = d_rel + RADAR_TO_CAMERA
    curvature = -2 * (7.8 - 3.5) / x**2
    path = -3.5 * 30.0 / x
    passed = -curvature * (FAR + RADAR_TO_CAMERA)**2 / 2
    tracks = get_tracks((FAR, passed, -1.0, 7))
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=get_model(curvature=curvature, path=0.0), radar_state=get_radar_state(d_rel=FAR, y_rel=passed), tracks=tracks)
    model = get_model(curvature=curvature, path=path, lead_std=1.82)
    step(gap, model=model, radar_state=get_radar_state(radar=False, track_id=-1, d_rel=d_rel, v_rel=0.5, y_rel=7.8), tracks=tracks, n=5)
    assert gap.leaving and released(gap) == [False, False]
    # the same lead off the path toward the lane being left is a car in that lane
    step(gap, model=model, radar_state=get_radar_state(radar=False, track_id=-1, d_rel=d_rel, v_rel=0.5, y_rel=7.8 - 1.5), tracks=tracks)
    assert released(gap)[0]

  def test_lead_two_on_the_target_side_is_followed(self):
    # a second lead at the first one's range and speed can be a car in the lane moved into, or the model placing the
    # same one there
    gap = self.start_behind_track_7()
    self.vision_after_handoff(gap, 1.5, lead_two_offset=1.5)
    assert released(gap) == [True, True]
    self.vision_after_handoff(gap, 1.5, lead_two_offset=-(LEAD_ON_PATH + 0.1))
    assert released(gap) == [True, False]
    # a radar lead the path has left, and a model-only one at its range and speed in the target lane
    path = path_for(1.5, FAR)
    lead_two = {'radar': False, 'track_id': -1, 'd_rel': FAR + 0.5, 'y_rel': on_path(FAR + 0.5, path) + 2.0}
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=get_model(path=path, lead_std=1.5, lead_two_std=1.5), radar_state=get_radar_state(d_rel=FAR, lead_two=lead_two))
    assert released(gap) == [True, False]

  def test_model_only_lead_two_is_judged_on_its_own(self):
    # a second lead only the model sees, at the first one's range and speed, can be another car: it is released only off
    # the path itself, like any lead only the model sees
    gap = self.start_behind_track_7()
    for lead_two_offset in (0.0, LEAD_OFF_PATH - 0.1):
      self.vision_after_handoff(gap, 1.5, lead_two_offset=lead_two_offset)
      assert released(gap) == [True, False]
    self.vision_after_handoff(gap, 1.5, lead_two_offset=LEAD_OFF_PATH + 0.1)
    assert released(gap) == [True, True]
    # a radar lead the path has left, and one only the model sees at its range and speed, on the path and placed
    path = path_for(1.5, FAR)
    lead_two = {'radar': False, 'track_id': -1, 'd_rel': FAR + 0.5, 'y_rel': on_path(FAR + 0.5, path)}
    radar_state = get_radar_state(d_rel=FAR, lead_two=lead_two)
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=get_model(path=path, lead_two_std=0.3), radar_state=radar_state)
    assert released(gap) == [True, False] and gap.followed(radar_state).leadTwo.present

  def test_one_radar_track_in_both_slots_is_one_car(self):
    # radard can match the model's second lead, placed elsewhere, to the track of the car in front: followed through
    # leadTwo, that car would still hold the MPC back
    path = path_for(1.5, FAR)
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=get_model(path=path, lead_two_std=0.3, lead_two_y=-(LEAD_CONTINUITY + 0.5)), radar_state=get_radar_state(d_rel=FAR, lead_two='same'))
    assert released(gap) == [True, True]
    # two radar tracks at the same range and speed are two cars
    gap = LaneChangeGap(CP, DT_MDL)
    lead_two = {'track_id': 9, 'd_rel': FAR + 0.5, 'y_rel': on_path(FAR + 0.5, path)}
    step(gap, model=get_model(path=path), radar_state=get_radar_state(d_rel=FAR, lead_two=lead_two))
    assert released(gap) == [True, False]

  def test_vision_lead_is_followed_once_the_car_is_across(self):
    # across the line, a lead on the path is in the lane moved into; the model's relabel can land the car for a frame
    gap = self.start_behind_track_7()
    for left_line in np.linspace(-1.6, LANDED_MARGIN - 0.1, 11):
      assert self.vision_after_handoff(gap, 1.5, left_line=left_line)
    landed, not_landed = LANDED_MARGIN + 0.1, LANDED_MARGIN - 0.1
    assert self.vision_after_handoff(gap, 1.5, left_line=landed)
    assert self.vision_after_handoff(gap, 1.5, left_line=not_landed)
    assert self.vision_after_handoff(gap, 1.5, left_line=landed, n=LANDED_FRAMES - 1)
    assert not self.vision_after_handoff(gap, 1.5, left_line=landed)
    assert not self.vision_after_handoff(gap, 1.5, left_line=not_landed, n=5)
    # a radar lead the path has left is still released
    step(gap, model=get_model(path=path_for(1.5, FAR), left_y=landed, right_y=landed + 3.36), radar_state=get_radar_state(track_id=8, d_rel=FAR),
         tracks=get_tracks((FAR, 0.0, -1.0, 7), (FAR, 0.0, -1.0, 8)))
    assert released(gap)[0]
    # one landed frame, then the model places no line for a moment: the line held meanwhile is no new sign of landing
    gap = self.start_behind_track_7()
    for left_line in np.linspace(-1.6, LANDED_MARGIN - 0.1, 11):
      self.vision_after_handoff(gap, 1.5, left_line=left_line)
    assert self.vision_after_handoff(gap, 1.5, left_line=landed)
    assert self.vision_after_handoff(gap, 1.5, left_line=landed, probs=(0.0,) * 4, line_stds=(1.0,) * 4, n=5)
    assert self.vision_after_handoff(gap, 1.5, left_line=landed, n=LANDED_FRAMES - 2)
    assert not self.vision_after_handoff(gap, 1.5, left_line=landed)

  def test_lead_two_is_judged_on_its_own(self):
    gap = LaneChangeGap(CP, DT_MDL)
    path = path_for(1.5, FAR)
    on_path_two = {'track_id': 9, 'd_rel': FAR + 20.0, 'y_rel': on_path(FAR + 20.0, path)}
    step(gap, model=get_model(path=path), radar_state=get_radar_state(d_rel=FAR, lead_two=on_path_two))
    assert released(gap) == [True, False]
    followed = gap.followed(get_radar_state(d_rel=FAR, lead_two=on_path_two))
    assert not followed.leadOne.present and followed.leadTwo.present

  def test_braking_restores_every_slot_for_the_rest_of_the_change(self):
    path = path_for(1.5, FAR)
    off_path_two = {'track_id': 9, 'd_rel': FAR + 20.0, 'y_rel': on_path(FAR + 20.0, path) - 1.5}
    cases = (
      ({'a_lead': LEAD_BRAKING - 0.5}, {}),
      ({'lead_two': dict(off_path_two, a_lead=LEAD_BRAKING - 0.5)}, {}),
      ({}, {'lead_two_std': 0.3, 'lead_two_accel': LEAD_BRAKING - 0.5}),
    )
    for radar_kwargs, model_kwargs in cases:
      gap = LaneChangeGap(CP, DT_MDL)
      step(gap, model=get_model(path=path, lead_two_std=0.3), radar_state=get_radar_state(d_rel=FAR, lead_two=off_path_two))
      assert released(gap) == [True, True]
      model = get_model(path=path, **{'lead_two_std': 0.3, **model_kwargs})
      step(gap, model=model, radar_state=get_radar_state(d_rel=FAR, **{'lead_two': off_path_two, **radar_kwargs}))
      assert released(gap) == [False, False] and gap.backed_out
      step(gap, model=get_model(path=path, lead_two_std=0.3), radar_state=get_radar_state(d_rel=FAR, lead_two=off_path_two), n=10)
      assert released(gap) == [False, False]

  def test_second_lead_braking_as_it_comes_is_never_released(self):
    path = path_for(1.5, FAR)
    off_path_two = {'track_id': 9, 'd_rel': FAR + 20.0, 'y_rel': on_path(FAR + 20.0, path) - 1.5}
    for radar_kwargs, model_kwargs in (({'a_lead': LEAD_BRAKING - 0.5}, {}), ({}, {'lead_two_accel': LEAD_BRAKING - 0.5})):
      gap = LaneChangeGap(CP, DT_MDL)
      step(gap, model=get_model(path=path, lead_two_std=0.3), radar_state=get_radar_state(d_rel=FAR))
      assert released(gap) == [True, False]
      radar_state = get_radar_state(d_rel=FAR, lead_two={**off_path_two, **radar_kwargs})
      model = get_model(path=path, **{'lead_two_std': 0.3, **model_kwargs})
      for _ in range(3):
        step(gap, model=model, radar_state=radar_state)
        assert released(gap) == [True, False] and gap.followed(radar_state).leadTwo.present

  def test_driver_backing_out_restores_for_the_rest_of_the_change(self):
    model = get_model(path=path_for(1.5, FAR))
    radar_state = get_radar_state(d_rel=FAR, lead_two='same')
    for back_out in (get_car_state(left_blinker=False), get_car_state(steeringPressed=True, steeringTorque=-50.0), 'reset'):
      gap = LaneChangeGap(CP, DT_MDL)
      step(gap, model=model, radar_state=radar_state)
      assert released(gap) == [True, True]
      if back_out == 'reset':
        gap.reset()
        step(gap, model=model, radar_state=radar_state)
      else:
        step(gap, model=model, radar_state=radar_state, CS=back_out)
      assert released(gap) == [False, False]
      step(gap, model=model, radar_state=radar_state, n=10)
      assert released(gap) == [False, False]
    # the planner resets before it updates, so a reset on the frame the change starts comes before the change
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=get_model(state=LaneChangeState.off))
    gap.reset()
    step(gap, model=model, radar_state=radar_state, n=10)
    assert released(gap) == [True, True]

  @staticmethod
  def lines_at(direction, line):
    # the car's lines with the one on the change side at line, right positive
    if direction == LaneChangeDirection.left:
      return get_model(path=0.0, left_y=line, right_y=line + 3.36)
    return change_right(path=0.0, left_y=line - 3.36, right_y=line)

  def cross_the_line(self, gap, direction, CS, radar_state, tracks):
    # the car's line on the change side walks in from 1.6 m to just short of landing; returns the model that lands it
    side = -1.0 if direction == LaneChangeDirection.left else 1.0
    for line in np.linspace(1.6, -(LANDED_MARGIN - 0.1), 11):
      step(gap, model=self.lines_at(direction, side * line), CS=CS, radar_state=radar_state, tracks=tracks)
    assert not gap.landed(direction)
    return self.lines_at(direction, -side * (LANDED_MARGIN + 0.1))

  def test_blinker_off_once_the_car_is_across_finishes_the_change(self):
    blinker_off = get_car_state(left_blinker=False)
    for direction, blinker_on, counter_steer in ((LaneChangeDirection.left, get_car_state(), -50.0), (LaneChangeDirection.right, right_blinker(), 50.0)):
      # the car being passed beside the car, released however close
      y_rel = -(LEAD_BESIDE + 0.5) if direction == LaneChangeDirection.left else LEAD_BESIDE + 0.5
      beside = get_radar_state(d_rel=20.0, y_rel=y_rel, lead_two='same')
      tracks = get_tracks((20.0, y_rel, -1.0, 7))
      gap = LaneChangeGap(CP, DT_MDL)
      landed = self.cross_the_line(gap, direction, blinker_on, beside, tracks)
      step(gap, model=landed, CS=blinker_on, radar_state=beside, tracks=tracks, n=LANDED_FRAMES)
      assert gap.landed(direction) and released(gap) == [True, True]
      step(gap, model=landed, CS=blinker_off, radar_state=beside, tracks=tracks, n=10)
      assert released(gap) == [True, True] and not gap.backed_out and gap.accelerate
      # counter-steer still backs out
      step(gap, model=landed, CS=get_car_state(left_blinker=False, steeringPressed=True, steeringTorque=counter_steer), radar_state=beside, tracks=tracks)
      assert released(gap) == [False, False] and gap.backed_out
      # landed for fewer frames than the model's relabel can last, the same blinker off backs out
      gap = LaneChangeGap(CP, DT_MDL)
      landed = self.cross_the_line(gap, direction, blinker_on, beside, tracks)
      step(gap, model=landed, CS=blinker_on, radar_state=beside, tracks=tracks, n=LANDED_FRAMES - 2)
      assert released(gap) == [True, True]
      step(gap, model=landed, CS=blinker_off, radar_state=beside, tracks=tracks)
      assert gap.landed(direction) and released(gap) == [False, False] and gap.backed_out

  def test_blocked_target_lane_restores_while_blocked(self):
    model = get_model(path=path_for(1.5, FAR))
    radar_state = get_radar_state(d_rel=FAR)
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=model, radar_state=radar_state)
    assert released(gap)[0]
    step(gap, model=model, radar_state=radar_state, tracks=get_tracks((30.0, 3.4, -1.0)))
    assert not released(gap)[0]
    step(gap, model=model, radar_state=radar_state, n=int(round(BLOCKED_HOLD / DT_MDL)))
    assert not released(gap)[0]
    step(gap, model=model, radar_state=radar_state)
    assert released(gap)[0]

  def test_release_does_not_need_headroom(self):
    # a car reaching its set speed mid pass is not handed back the car it is passing
    gap = LaneChangeGap(CP, DT_MDL)
    assert step(gap, model=get_model(path=path_for(1.5, FAR)), radar_state=get_radar_state(d_rel=FAR), v_cruise=V_EGO) == 0.0
    assert not gap.accelerate and released(gap)[0]

  def test_release_only_while_the_change_is_starting(self):
    radar_state = get_radar_state(d_rel=FAR)
    for model in (get_model(state=LaneChangeState.preLaneChange, path=path_for(1.5, FAR)), get_model(state=LaneChangeState.off, path=path_for(1.5, FAR)),
                  get_model(state=LaneChangeState.laneChangeFinishing, path=path_for(1.5, FAR)),
                  get_model(direction=LaneChangeDirection.none, path=path_for(1.5, FAR))):
      gap = LaneChangeGap(CP, DT_MDL)
      step(gap, model=get_model(path=path_for(1.5, FAR)), radar_state=radar_state)
      assert released(gap)[0]
      step(gap, model=model, radar_state=radar_state)
      assert released(gap) == [False, False]
      assert gap.followed(radar_state) is radar_state

  def test_latch_and_evidence_reset_with_the_next_change(self):
    model = get_model(path=path_for(2.0, FAR))
    gap = LaneChangeGap(CP, DT_MDL)
    step(gap, model=model, radar_state=get_radar_state(d_rel=45.0, v_rel=-4.0), tracks=get_tracks((45.0, 0.0, -4.0, 7)), n=CLOSE_FRAMES + 1)
    assert gap.closing
    step(gap, model=get_model(state=LaneChangeState.off))
    step(gap, model=model, radar_state=get_radar_state(d_rel=FAR))
    assert not gap.closing and released(gap)[0]

  def test_needs_the_sensors_to_release(self):
    for cp in (car.CarParams.new_message(radarUnavailable=True, brand='hyundai', flags=HyundaiFlags.HAS_BSM.value),
               car.CarParams.new_message(radarUnavailable=False, brand='hyundai')):
      gap = LaneChangeGap(cp, DT_MDL)
      step(gap, model=get_model(path=path_for(1.5, FAR)), radar_state=get_radar_state(d_rel=FAR), n=5)
      assert released(gap) == [False, False]


class TestCrossedLine:
  # a change to the right: the car's lines at -1.8 and +1.2 m (model frame, right positive), 3.0 m apart
  PLACED = (0.1, 0.1, 0.1, 0.1)
  DOUBTED = (0.0, 0.05, 0.05, 0.0)

  def run(self, gap, right_line, probs, stds, tracks=(), state=LaneChangeState.laneChangeStarting, CS=None, n=1, lines=None, curvature=0.0):
    lines = lines if lines is not None else (right_line - 6.0, right_line - 3.0, right_line, right_line + 3.5)
    model = change_right(state=state, lines=lines, probs=probs, line_stds=stds, curvature=curvature)
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
    # the model relabels: the crossed line is now the car's left line, 0.6 m to its left, and the lane moved into is the
    # car's own, where a car ahead is a lead to follow rather than a blocked target lane
    relabelled = (-3.6, -LANDED_MARGIN - 0.1, 2.9, 6.4)
    ahead = [(30.0, -1.5, -4.9)]
    assert not self.run(gap, None, (0.5, 0.7, 0.7, 0.5), self.PLACED, lines=relabelled, tracks=beyond + ahead)
    assert gap.landed(LaneChangeDirection.right)
    # the blind spot still counts after landing
    assert self.run(gap, None, (0.5, 0.7, 0.7, 0.5), self.PLACED, lines=relabelled, CS=right_blinker(rightBlindspot=True))

  def test_band_bends_with_the_crossed_line(self):
    # a right hand bend puts the lane moved into 2 m further right 30 m out than beside the car
    curvature = 2 * 2.0 / (30.0 + RADAR_TO_CAMERA)**2
    for y_rel, blocks in ((-4.5, True), (-2.0, False)):
      gap = LaneChangeGap(CP, DT_MDL)
      self.run(gap, 1.2, (0.8, 0.9, 0.8, 0.8), self.PLACED, state=LaneChangeState.preLaneChange, curvature=curvature)
      assert self.run(gap, 1.0, self.DOUBTED, self.PLACED, tracks=[(30.0, y_rel, -4.9)], curvature=curvature) == blocks

  def test_crossed_line_is_held_rather_than_moved_a_lane(self):
    # with the crossed line not placed for a moment, the nearest placed line is the next one over, 3 m away
    gap = LaneChangeGap(CP, DT_MDL)
    self.run(gap, 1.2, (0.8, 0.9, 0.8, 0.8), self.PLACED, state=LaneChangeState.preLaneChange)
    self.run(gap, 1.0, self.DOUBTED, self.PLACED)
    assert self.run(gap, 1.0, self.DOUBTED, (0.1, 0.1, 1.0, 0.1), tracks=[(30.0, -2.5, -4.9)])
    assert np.isclose(gap.crossed[1][0], -1.0) and gap.crossed_age > 0.0

  def test_crossed_line_hold_expires(self):
    pickup = [(30.0, -5.2, -4.9)]
    lost = (1.0, 1.0, 1.0, 1.0)
    gap = LaneChangeGap(CP, DT_MDL)
    self.run(gap, 1.2, (0.8, 0.9, 0.8, 0.8), self.PLACED, state=LaneChangeState.preLaneChange)
    assert not self.run(gap, 1.0, self.DOUBTED, self.PLACED, tracks=pickup)
    assert not self.run(gap, 1.0, self.DOUBTED, lost, tracks=pickup, n=int(round(LINE_HOLD / DT_MDL)) - 1)
    assert self.run(gap, 1.0, self.DOUBTED, lost, tracks=pickup, n=2)
    assert gap.crossed is None

  def test_line_well_away_does_not_seed(self):
    # a line past LINE_OFFSET_MAX is the next one over, not the one the car crosses
    gap = LaneChangeGap(CP, DT_MDL)
    self.run(gap, 2.6, (0.8, 0.9, 0.8, 0.8), self.PLACED, state=LaneChangeState.preLaneChange)
    assert self.run(gap, 2.6, self.DOUBTED, self.PLACED, tracks=[(30.0, -2.0, -4.9)])
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

  def test_near_edge_stays_off_the_cars_path(self):
    # with the car 0.2 m from the line, a return just past it is on the car's own path, and the band starts LINE_OFFSET_MIN
    # out from the car
    gap = LaneChangeGap(CP, DT_MDL)
    self.run(gap, 1.2, (0.8, 0.9, 0.8, 0.8), self.PLACED, state=LaneChangeState.preLaneChange)
    for right_line in np.arange(1.2, 0.2, -0.1):
      self.run(gap, right_line, self.DOUBTED, self.PLACED)
    assert not self.run(gap, 0.2, self.DOUBTED, self.PLACED, tracks=[(30.0, -(LINE_OFFSET_MIN - 0.1), -4.9)])
    assert self.run(gap, 0.2, self.DOUBTED, self.PLACED, tracks=[(30.0, -(LINE_OFFSET_MIN + 0.1), -4.9)])

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
    # a return just short of oncoming speed is one of the stationary ones, for the margin as for its arrival
    assert not blocked(get_tracks((FOLLOW_GAP - 1.0, outer - ROADSIDE_MARGIN / 2, -V_EGO - TRACK_MOVING_SPEED)))
    assert not blocked(get_tracks((FOLLOW_GAP + 1.0, 3.4, -V_EGO - TRACK_MOVING_SPEED)))

  def test_oncoming_car_on_the_outer_line_is_a_car(self):
    # on a two-lane road the target lane's outer edge is where oncoming traffic drives
    arriving = FOLLOW_GAP + 2.0 * V_EGO * ARRIVAL_TIME - 1.0
    _, outer = LaneLines(get_model()).band(LaneChangeDirection.left, arriving + RADAR_TO_CAMERA)
    assert blocked(get_tracks((arriving, outer - ROADSIDE_MARGIN / 2, -2.0 * V_EGO)))
    assert not blocked(get_tracks((arriving + 2.0, outer - ROADSIDE_MARGIN / 2, -2.0 * V_EGO)))

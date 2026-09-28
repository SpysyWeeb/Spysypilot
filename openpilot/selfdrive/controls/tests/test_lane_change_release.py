from collections import deque

import numpy as np
import pytest

from openpilot.cereal import log
import openpilot.cereal.messaging as messaging
from opendbc.car.structs import car
from opendbc.car.hyundai.values import HyundaiFlags
from openpilot.common.realtime import DT_MDL
from openpilot.common.simple_kalman import KF1D
from openpilot.selfdrive.controls.lib import longitudinal_planner
from openpilot.selfdrive.controls.lib.lane_change_gap import LEAD_BRAKING, LEAD_SPEED_FRAMES
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import get_stopped_equivalence_factor, get_T_FOLLOW
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.radard import KalmanParams, RADAR_TO_CAMERA, _LEAD_ACCEL_TAU
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import _PlantSubMaster

LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection
AGGRESSIVE = log.LongitudinalPersonality.aggressive
X_IDXS = np.array(ModelConstants.X_IDXS)
LANES = (-5.25, -1.75, 1.75, 5.25, 8.75)  # m, lane lines right positive: the car starts in the lane at 0, the target lane is at +3.5
TARGET = 3.5


def get_cp(bsm=True):
  return car.CarParams.new_message(brand='hyundai', flags=HyundaiFlags.HAS_BSM.value if bsm else 0, radarUnavailable=False,
                                   openpilotLongitudinalControl=True, longitudinalActuatorDelay=0.5, steerRatio=17.9, wheelbase=2.9)


def lead_data(lead, d_rel, v_lead, v_ego, a_lead=0.0, y_rel=0.0, radar=True, track_id=1):
  lead.present = True
  lead.radar = radar
  lead.radarTrackId = track_id if radar else -1
  lead.dRel, lead.yRel, lead.vRel, lead.vLead, lead.vLeadK = d_rel, y_rel, v_lead - v_ego, v_lead, v_lead
  lead.aLeadK, lead.aLeadTau, lead.modelProb = a_lead, _LEAD_ACCEL_TAU, 1.0


def frame(v_ego, a_ego, d_rel, v_lead, y_ego=0.0, path=0.0, state=LaneChangeState.laneChangeStarting, blinker=True, a_lead=0.0,
          model_a_lead=0.0, v_cruise=22.35, long_active=True, steering_torque=0.0, extra_tracks=(), lead_present=True):
  # one planner frame of a change to the right: the car y_ego meters over, the car it started behind at d_rel straight
  # ahead of where it was, the planned path reaching path meters further right at 30 m
  v_ego, a_ego, d_rel, v_lead, y_ego, path = (float(v) for v in (v_ego, a_ego, d_rel, v_lead, y_ego, path))
  starting = state == LaneChangeState.laneChangeStarting
  model = log.ModelDataV2.new_message()
  model.meta.laneChangeState = state
  model.meta.laneChangeDirection = LaneChangeDirection.right if starting else LaneChangeDirection.none
  model.init('laneLines', 4)
  first = 0 if y_ego < 1.75 else 1
  for line, y in zip(model.laneLines, LANES[first:first + 4], strict=True):
    line.x = X_IDXS.tolist()
    line.y = [y - y_ego] * len(X_IDXS)
  crossing = starting and abs(y_ego - 1.75) < 1.2
  model.laneLineProbs = [0.8, 0.1 if crossing else 0.9, 0.1 if crossing else 0.9, 0.8]
  model.laneLineStds = [0.2, 0.15, 0.15, 0.2]
  model.position.x = X_IDXS.tolist()
  model.position.y = [float(y) for y in path * X_IDXS / 30.0]
  model.init('leadsV3', 2)
  for model_lead in model.leadsV3:
    model_lead.prob = 1.0
    model_lead.x = [d_rel + RADAR_TO_CAMERA] * 6
    model_lead.y = [-y_ego] * 6
    model_lead.yStd = [0.3] * 6
    model_lead.a = [model_a_lead] * 6
  radar_state = log.RadarState.new_message()
  if lead_present:
    for lead in (radar_state.leadOne, radar_state.leadTwo):
      lead_data(lead, d_rel, v_lead, v_ego, a_lead, y_rel=y_ego)
  tracks = car.RadarData.new_message()
  points = [(1, d_rel, y_ego, v_lead - v_ego), *extra_tracks]
  tracks.init('points', len(points))
  for pt, (track_id, d, y, v) in zip(tracks.points, points, strict=True):
    pt.trackId, pt.dRel, pt.yRel, pt.vRel = track_id, d, y, v
  car_state = car.CarState.new_message(vEgo=v_ego, aEgo=a_ego, vCruise=v_cruise * 3.6, rightBlinker=blinker, standstill=False,
                                       steeringPressed=steering_torque != 0.0, steeringTorque=steering_torque)
  controls_state = messaging.new_message('controlsState').controlsState
  controls_state.longControlState = 'pid' if long_active else 'off'
  selfdrive_state = messaging.new_message('selfdriveState').selfdriveState
  selfdrive_state.personality = AGGRESSIVE
  selfdrive_state.enabled = True
  return _PlantSubMaster({'modelV2': model, 'radarState': radar_state, 'radarTracks': tracks, 'carState': car_state,
                          'controlsState': controls_state, 'selfdriveState': selfdrive_state,
                          'carControl': messaging.new_message('carControl').carControl,
                          'vehicleParameters': messaging.new_message('vehicleParameters').vehicleParameters})


def released_frames(planner, v_ego=12.0, d_rel=40.0, v_lead=11.0, n=5):
  # a change well under way past a slower car: path and car both 1.5 m over, the lead released
  for _ in range(n):
    planner.update(frame(v_ego, 0.0, d_rel, v_lead, y_ego=1.5, path=2.0))
  assert any(planner.lane_change_gap.released)


def mpc_follows(planner, d_rel, v_lead):
  # the MPC's obstacle this solve is the lead at d_rel, not the fast far lead a released slot becomes
  return np.isclose(planner.mpc.params[0, 2], d_rel + get_stopped_equivalence_factor(v_lead), atol=0.5)


class TestPlanningView:
  def test_released_lead_is_left_out_of_the_plan_only(self):
    planner = LongitudinalPlanner(get_cp(), init_v=12.0)
    for _ in range(20):
      planner.update(frame(12.0, 0.0, 40.0, 11.0, state=LaneChangeState.off, blinker=False))
    assert mpc_follows(planner, 40.0, 11.0)
    a_following = planner.output_a_target
    sm = frame(12.0, 0.0, 40.0, 11.0, y_ego=1.5, path=2.0)
    for _ in range(5):
      planner.update(sm)
    assert any(planner.lane_change_gap.released) and not mpc_follows(planner, 40.0, 11.0)
    assert planner.output_a_target > a_following and not planner.fcw
    # radard's message is untouched for everything else, hasLead among them
    assert sm['radarState'].leadOne.present and sm['radarState'].leadTwo.present

  @pytest.mark.parametrize('cue', ['blinker off', 'counter-steer', 'lead braking', 'model braking', 'blocked', 'change over', 'path back',
                                   'too close'])
  def test_every_restore_reaches_the_mpc_in_the_same_frame(self, cue):
    planner = LongitudinalPlanner(get_cp(), init_v=12.0)
    released_frames(planner)
    d_rel, v_lead = 40.0, 11.0
    kwargs = {
      'blinker off': {'blinker': False},
      'counter-steer': {'steering_torque': 100.0},
      'lead braking': {'a_lead': LEAD_BRAKING - 1.0},
      'model braking': {'model_a_lead': LEAD_BRAKING - 1.0},
      'blocked': {'extra_tracks': [(5, 15.0, -2.0, -1.0)]},
      'change over': {'state': LaneChangeState.off},
      'path back': {'path': -1.5},
    }.get(cue, {})
    if cue == 'too close':
      d_rel, v_lead = 24.0, 9.0
      for _ in range(LEAD_SPEED_FRAMES - 1):
        planner.update(frame(12.0, 0.0, d_rel, v_lead, y_ego=1.5, path=2.0))
      assert any(planner.lane_change_gap.released)
    planner.update(frame(12.0, 0.0, d_rel, v_lead, **{'y_ego': 1.5, 'path': 2.0, **kwargs}))
    assert not any(planner.lane_change_gap.released)
    assert mpc_follows(planner, d_rel, v_lead)

  def test_override_resets_the_release(self):
    planner = LongitudinalPlanner(get_cp(), init_v=12.0)
    released_frames(planner)
    planner.update(frame(12.0, 0.0, 40.0, 11.0, y_ego=1.5, path=2.0, long_active=False))
    planner.update(frame(12.0, 0.0, 40.0, 11.0, y_ego=1.5, path=2.0))
    assert not any(planner.lane_change_gap.released) and mpc_follows(planner, 40.0, 11.0)

  def test_fcw_sees_a_car_it_could_hit(self):
    # a stopped car is never released, so the crash check sees it as it would without the change
    planner = LongitudinalPlanner(get_cp(), init_v=20.0)
    released_frames(planner, v_ego=20.0, d_rel=60.0, v_lead=18.0)
    for _ in range(3):
      planner.update(frame(20.0, 0.0, 12.0, 0.0, y_ego=1.5, path=2.0))
    assert not any(planner.lane_change_gap.released)
    assert planner.fcw

  def test_plan_is_stock_outside_a_change(self):
    planner, stock = LongitudinalPlanner(get_cp(), init_v=12.0), LongitudinalPlanner(get_cp(bsm=False), init_v=12.0)
    for state in (LaneChangeState.off, LaneChangeState.preLaneChange):
      for _ in range(30):
        sm = frame(12.0, 0.0, 40.0, 11.0, state=state, path=2.0)
        planner.update(sm)
        stock.update(sm)
        assert planner.output_a_target == stock.output_a_target

  def test_mpc_gets_radards_message_while_nothing_is_released(self):
    planner = LongitudinalPlanner(get_cp(), init_v=12.0)
    sm = frame(12.0, 0.0, 40.0, 11.0, path=0.3)
    planner.update(sm)
    assert not any(planner.lane_change_gap.released)
    assert planner.lane_change_gap.followed(sm['radarState']) is sm['radarState']


class LaneChangePlant:
  # a change to the right past a slower car: lateral cosine move over move_time from 0.3 s (optionally stalling at y_stall),
  # 0.2 s actuator delay and a 0.3 s lag, radard's lead filter on the lead speed, the change ending at 5.05 s
  def __init__(self, bsm, v_ego, v_lead, d_rel, v_cruise=22.35, move_time=4.0, y_stall=None, abort_at=None, lead_brake=0.0, t_brake=np.inf):
    self.planner = LongitudinalPlanner(get_cp(bsm), init_v=v_ego)
    self.v, self.a, self.x = v_ego, 0.0, 0.0
    self.v_lead, self.x_lead = v_lead, d_rel
    self.v_cruise, self.move_time, self.y_stall, self.abort_at = v_cruise, move_time, y_stall, abort_at
    self.lead_brake, self.t_brake = lead_brake, t_brake
    self.commands = deque([0.0] * 5, maxlen=5)
    self.kf = KF1D([[v_lead], [0.0]], KalmanParams(DT_MDL).A, KalmanParams(DT_MDL).C, KalmanParams(DT_MDL).K)

  def run(self, duration=10.0):
    log_ = []
    for k in range(-40, int(duration / DT_MDL)):
      t = k * DT_MDL
      s = np.clip((t - 0.3) / self.move_time, 0.0, 1.0)
      y = TARGET * (1 - np.cos(np.pi * s)) / 2
      y = min(y, self.y_stall) if self.y_stall is not None else y
      aborted = self.abort_at is not None and t >= self.abort_at
      if aborted:
        y = max(y - (t - self.abort_at), 0.0)
      changing = 0.0 <= t < 5.05 and not (self.abort_at is not None and t >= self.abort_at + 0.3)
      path = (TARGET - y) * min((t + 0.5) / 1.4, 1.0) if changing else 0.0
      self.kf.update(self.v_lead)
      d = self.x_lead - self.x
      sm = frame(self.v, self.a, d, self.v_lead, y_ego=y, path=path, blinker=t >= -0.5 and not aborted, a_lead=float(self.kf.x[1][0]),
                 state=LaneChangeState.laneChangeStarting if changing else LaneChangeState.off, v_cruise=self.v_cruise,
                 lead_present=y < 2.5)
      self.planner.update(sm)
      a_target = float(self.planner.output_a_target)
      self.commands.append(a_target)
      if t < 0.0:
        self.a = a_target
        continue
      self.a += (self.commands[0] - self.a) * DT_MDL / 0.3
      self.v = max(self.v + self.a * DT_MDL, 0.0)
      self.x += self.v * DT_MDL
      self.v_lead = max(self.v_lead + (self.lead_brake if t >= self.t_brake else 0.0) * DT_MDL, 0.0)
      self.x_lead += self.v_lead * DT_MDL
      log_.append((t, self.x_lead - self.x, y, a_target, any(self.planner.lane_change_gap.released), self.planner.fcw))
    return np.array(log_, dtype=float)


def one_second_gap(monkeypatch):
  # the gaps the car drives on: 1.0 s aggressive and a 7 m stop distance. The MPC's own stop distance is compiled into its
  # solver and stays 6 m here; the lane change module is handed 7 m
  def get_t_follow(personality=log.LongitudinalPersonality.standard):
    return 1.0 if personality == AGGRESSIVE else get_T_FOLLOW(personality)
  monkeypatch.setattr(long_mpc, 'get_T_FOLLOW', get_t_follow)
  monkeypatch.setattr(longitudinal_planner, 'get_T_FOLLOW', get_t_follow)
  monkeypatch.setattr(longitudinal_planner, 'STOP_DISTANCE', 7.0)


def summary(log_):
  t, gap, y, a_target, released, fcw = log_.T
  overlapping = y < 2.0
  return {'min_gap': gap[overlapping].min(), 'min_a': a_target.min(), 'onsets': int(np.sum(np.diff(np.r_[0, released]) > 0)),
          'first_release': t[np.argmax(released > 0)] if released.any() else None, 'fcw': bool(fcw.any()),
          'a_target': a_target, 'released': released > 0, 't': t}


class TestLaneChangeManeuvers:
  # bounds measured on this plant with the branch's aggressive gap (1.25 s) and 6 m stop distance; the release starts
  # before the car moves over, which is the worst case for a lateral stall
  def test_passing_a_slower_car(self):
    # route bb's speeds, 28 m behind: outside the branch's aggressive follow distance
    today = summary(LaneChangePlant(False, 12.8, 11.1, 28.0).run())
    design = summary(LaneChangePlant(True, 12.8, 11.1, 28.0).run())
    assert design['first_release'] <= 1.0
    assert design['a_target'][int(round((design['first_release'] + 0.3) / DT_MDL))] > 0.5
    assert design['min_gap'] >= today['min_gap'] - 3.0
    assert design['onsets'] <= 2 and not design['fcw']

  def test_lateral_stall_hands_the_car_back(self):
    today = summary(LaneChangePlant(False, 12.8, 11.1, 28.0, y_stall=1.2).run())
    design = summary(LaneChangePlant(True, 12.8, 11.1, 28.0, y_stall=1.2).run())
    assert design['onsets'] == 1 and not design['fcw']
    assert design['min_gap'] >= today['min_gap'] - 5.5
    assert design['min_a'] >= today['min_a'] - 0.6

  def test_much_slower_car_on_the_highway_is_released_beside(self):
    today = summary(LaneChangePlant(False, 29.0, 22.0, 45.0, v_cruise=31.3).run())
    design = summary(LaneChangePlant(True, 29.0, 22.0, 45.0, v_cruise=31.3).run())
    assert design['first_release'] >= 2.4
    before = design['t'] < design['first_release']
    assert np.allclose(design['a_target'][before], today['a_target'][before], atol=0.05)
    assert design['min_gap'] >= today['min_gap'] - 0.5

  def test_driver_aborts_after_the_release(self):
    # 40 m behind, the car being passed is still released when the driver switches the blinker off and steers back
    today = summary(LaneChangePlant(False, 12.8, 11.1, 40.0, abort_at=1.5).run())
    design = summary(LaneChangePlant(True, 12.8, 11.1, 40.0, abort_at=1.5).run())
    aborted = design['t'] >= 1.5 - 1e-6
    assert design['released'][~aborted][-1] and not design['released'][aborted].any()
    assert design['onsets'] == 1 and not design['fcw']
    assert design['min_gap'] >= today['min_gap'] - 5.6
    assert design['min_a'] >= today['min_a'] - 0.75

  def test_set_speed_close_does_not_flicker(self):
    today = summary(LaneChangePlant(False, 12.8, 11.1, 28.0, v_cruise=12.8 + 0.8).run())
    design = summary(LaneChangePlant(True, 12.8, 11.1, 28.0, v_cruise=12.8 + 0.8).run())
    assert design['onsets'] <= 2
    assert design['min_a'] >= today['min_a'] - 0.3

  def test_passed_car_brakes_hard_after_the_release(self):
    today = summary(LaneChangePlant(False, 12.8, 11.1, 28.0, lead_brake=-6.0, t_brake=1.85).run())
    design = summary(LaneChangePlant(True, 12.8, 11.1, 28.0, lead_brake=-6.0, t_brake=1.85).run())
    assert design['min_gap'] >= today['min_gap'] - 3.0

  # route bb's speeds and 21.4 m at the 1.0 s gap: inside the branch's own follow distance, where the release waits for
  # the car being passed to be beside, but not inside this one's, so the release starts before the car moves over
  def test_passing_a_slower_car_at_a_one_second_gap(self, monkeypatch):
    one_second_gap(monkeypatch)
    today = summary(LaneChangePlant(False, 12.8, 11.1, 21.4).run())
    design = summary(LaneChangePlant(True, 12.8, 11.1, 21.4).run())
    assert design['first_release'] <= 1.0
    assert design['a_target'][int(round((design['first_release'] + 0.3) / DT_MDL))] > 0.5
    assert design['min_gap'] >= today['min_gap'] - 2.5
    assert design['onsets'] <= 2 and not design['fcw']

  def test_lateral_stall_hands_the_car_back_at_a_one_second_gap(self, monkeypatch):
    one_second_gap(monkeypatch)
    today = summary(LaneChangePlant(False, 12.8, 11.1, 21.4, y_stall=1.2).run())
    design = summary(LaneChangePlant(True, 12.8, 11.1, 21.4, y_stall=1.2).run())
    assert design['onsets'] == 1 and not design['fcw']
    assert design['min_gap'] >= today['min_gap'] - 4.6
    assert design['min_a'] >= today['min_a'] - 0.4

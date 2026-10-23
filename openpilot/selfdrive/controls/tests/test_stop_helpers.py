from openpilot.cereal import log
import math


import openpilot.cereal.messaging as messaging
from openpilot.selfdrive.controls.lib.stop_helpers import (STOP_DIRECT_CONFIDENCE, STOP_EARLY_CONFIDENCE, STOP_EARLY_HINT_CONFIDENCE,
                                                           STOP_EARLY_HINT_ENTRY_CONFIDENCE, STOP_TRAJECTORY_CONFIDENCE,
                                                           lead_owns_stop, leads_clear_of_stop_path, observe_model_stop, stop_release_open)
from openpilot.selfdrive.modeld.constants import ModelConstants

N = ModelConstants.IDX_N


def model(path_end=90.0, terminal_speed=10.0, should_stop=False, desired_accel=0.0, curvature=0.0, heading=0.0, leads=()):
  md = messaging.new_message('modelV2').modelV2
  md.position.x = [path_end * i / (N - 1) for i in range(N)]
  md.position.y = [0.0] * N
  md.velocity.x = [10.0 + (terminal_speed - 10.0) * i / (N - 1) for i in range(N)]
  md.orientation.z = [heading * i / (N - 1) for i in range(N)]
  md.action.shouldStop = should_stop
  md.action.desiredAcceleration = desired_accel
  md.action.desiredCurvature = curvature
  md.init('leadsV3', len(leads))
  for lead, (prob, x, y) in zip(md.leadsV3, leads, strict=True):
    lead.prob = prob
    lead.x = [x] * len(ModelConstants.LEAD_T_IDXS)
    lead.y = [y] * len(ModelConstants.LEAD_T_IDXS)
  return md


def car_state(v_ego=10.0, blinker=False, steering_angle=0.0):
  cs = messaging.new_message('carState').carState
  cs.vEgo = v_ego
  cs.leftBlinker = blinker
  cs.steeringAngleDeg = steering_angle
  return cs


def radar_state(d_one=None, d_two=None, v_one=0.0, v_two=0.0):
  rs = messaging.new_message('radarState').radarState
  for lead, d, v in ((rs.leadOne, d_one, v_one), (rs.leadTwo, d_two, v_two)):
    if d is not None:
      lead.present = True
      lead.dRel = d
      lead.vLead = v
  return rs


def observe(md, cs=None, rs=None):
  return observe_model_stop(md, cs or car_state(), rs or radar_state())


class TestStopTiers:
  def test_ordinary_driving_is_no_evidence(self):
    obs = observe(model())
    assert obs.confidence == 0.0 and obs.complete and not obs.strict_stop and not obs.early_stop

  def test_should_stop_is_direct_evidence(self):
    assert observe(model(should_stop=True)).confidence == STOP_DIRECT_CONFIDENCE

  def test_strict_trajectory_stop_inside_the_horizon(self):
    assert observe(model(path_end=30.0, terminal_speed=0.2, desired_accel=-0.3)).confidence == STOP_TRAJECTORY_CONFIDENCE
    assert observe(model(path_end=60.0, terminal_speed=0.2, desired_accel=-0.3)).confidence == 0.0

  def test_high_speed_early_tier_needs_a_straight_braking_approach(self):
    v = 19.0  # 42 mph, route d7: 138 m path, 5.85 m/s terminal, -0.5 m/s^2
    cs = car_state(v)
    assert observe(model(path_end=138.0, terminal_speed=5.85, desired_accel=-0.5), cs).confidence == STOP_EARLY_CONFIDENCE
    assert observe(model(path_end=138.0, terminal_speed=5.85, desired_accel=-0.5, heading=math.radians(30)), cs).early_stop is False
    assert observe(model(path_end=138.0, terminal_speed=5.85, desired_accel=-0.5, curvature=0.01), cs).early_stop is False
    assert observe(model(path_end=138.0, terminal_speed=5.85, desired_accel=-0.5), car_state(v, blinker=True)).early_stop is False

  def test_urban_hint_qualifies_only_at_urban_speed(self):
    hint = model(path_end=150.0, terminal_speed=9.0, desired_accel=-0.3)
    assert observe(hint, car_state(20.0)).confidence == STOP_EARLY_HINT_ENTRY_CONFIDENCE
    assert observe(hint, car_state(24.0)).confidence == STOP_EARLY_HINT_CONFIDENCE

  def test_highway_slowdown_is_not_a_stop(self):
    obs = observe(model(path_end=180.0, terminal_speed=8.0, desired_accel=-0.6), car_state(24.6))
    assert not obs.early_stop and obs.confidence < STOP_EARLY_CONFIDENCE

  def test_incomplete_model_is_no_evidence_but_keeps_the_raw_lead(self):
    md = messaging.new_message('modelV2').modelV2
    obs = observe(md, rs=radar_state(d_two=150.0))
    assert not obs.complete and obs.confidence == 0.0 and obs.lead_present


class TestVetoes:
  def test_relevant_lead_scales_with_speed_and_path(self):
    assert observe(model(), rs=radar_state(d_one=30.0)).relevant_lead
    assert not observe(model(), rs=radar_state(d_one=120.0)).relevant_lead
    assert observe(model(path_end=115.0), rs=radar_state(d_one=120.0)).relevant_lead

  def test_committed_turn_needs_blinker_and_a_real_turn(self):
    assert observe(model(), car_state(5.0, blinker=True, steering_angle=45.0)).committed_turn
    assert observe(model(curvature=0.05), car_state(5.0, blinker=True)).committed_turn
    assert not observe(model(), car_state(5.0, blinker=True, steering_angle=10.0)).committed_turn
    assert not observe(model(), car_state(12.0, blinker=True, steering_angle=45.0)).committed_turn


def stop_ahead(leads=()):
  # a 50 m stop; the model publishes LEAD_MHP_SELECTION hypotheses, the ones not given here unlikely and far away
  leads = tuple(leads) + ((0.0, 200.0, 0.0),) * (ModelConstants.LEAD_MHP_SELECTION - len(leads))
  return model(path_end=50.0, terminal_speed=0.2, desired_accel=-0.3, leads=leads)


class TestLeadOwnership:
  # a 50 m stop: a lead owns it when it is a queue (a relevant lead at a crawl), when the MPC, stopping STOP_DISTANCE behind
  # the lead's rest point, would stop the car at or before it, or when the model's likely lead forecast ends by the stop
  # point plus LEAD_PATH_MARGIN
  def test_a_lead_stopping_short_of_the_stop_owns_it(self):
    assert observe(stop_ahead(), rs=radar_state(d_one=30.0)).lead_owns_stop
    assert observe(stop_ahead(), rs=radar_state(d_one=30.0, v_one=8.0)).lead_owns_stop  # rest point 30 + 12.8 - 7 m
    assert observe(stop_ahead(), rs=radar_state(d_one=52.0, v_one=5.0)).lead_owns_stop  # rest point exactly at the stop point

  def test_a_lead_going_through_the_stop_does_not_own_it(self):
    # the yellow light: a lead 31 m ahead at 14 m/s rests past a 50 m stop; it stays the relevant lead Force Stops holds for
    obs = observe(stop_ahead(), rs=radar_state(d_one=31.0, v_one=14.0))
    assert obs.relevant_lead and not obs.lead_owns_stop
    assert not observe(stop_ahead(), rs=radar_state(d_one=52.5, v_one=5.0)).lead_owns_stop

  def test_a_queue_owns_the_stop_the_model_makes_behind_it(self):
    # the model stops a few meters farther back than the MPC's STOP_DISTANCE: a stopped or creeping car whose rest point
    # lies past the stop is the queue the model stops for, not a car going through
    for v_lead in (-0.5, 0.0, 0.9):
      obs = observe(stop_ahead(), rs=radar_state(d_one=58.0, v_one=v_lead))
      assert obs.relevant_lead and obs.lead_owns_stop
    assert not observe(stop_ahead(), rs=radar_state(d_one=58.0, v_one=2.0)).lead_owns_stop  # rest point 51.8 m

  def test_a_queue_beyond_the_relevance_window_does_not_own_the_stop(self):
    # a lead going through, and a stopped car beyond the window: past the junction, not in it
    obs = observe(stop_ahead(), rs=radar_state(d_one=31.0, v_one=14.0, d_two=100.0, v_two=0.0))
    assert obs.relevant_lead and not obs.lead_owns_stop
    assert observe(stop_ahead(), rs=radar_state(d_one=31.0, v_one=14.0, d_two=60.0, v_two=0.0)).lead_owns_stop

  def test_either_lead_slot_can_own_the_stop(self):
    assert observe(stop_ahead(), rs=radar_state(d_one=31.0, d_two=45.0, v_one=14.0, v_two=5.0)).lead_owns_stop

  def test_the_models_likely_lead_forecast_also_owns_the_stop(self):
    through = radar_state(d_one=31.0, v_one=14.0)
    assert observe(stop_ahead(((0.6, 58.0, 0.0),)), rs=through).lead_owns_stop
    assert not observe(stop_ahead(((0.6, 62.0, 0.0),)), rs=through).lead_owns_stop
    # an unlikely hypothesis is no forecast; the radar alone decides
    assert not observe(stop_ahead(((0.4, 40.0, 0.0),)), rs=through).lead_owns_stop

  def test_the_radar_geometry_owns_the_stop_whatever_the_forecast(self):
    assert observe(stop_ahead(((0.9, 200.0, 0.0),)), rs=radar_state(d_one=30.0, v_one=5.0)).lead_owns_stop

  def test_only_a_relevant_lead_can_own_the_stop(self):
    obs = observe(stop_ahead(((0.9, 40.0, 0.0),)), rs=radar_state(d_one=120.0))
    assert not obs.relevant_lead and not obs.lead_owns_stop

  def test_ownership_fails_closed(self):
    nan = float('nan')
    through = radar_state(d_one=31.0, v_one=14.0)
    assert lead_owns_stop(stop_ahead(), through, 10.0, 0.0)
    assert lead_owns_stop(stop_ahead(), through, 10.0, nan)
    assert lead_owns_stop(stop_ahead(), radar_state(d_one=nan), 10.0, 50.0)
    assert lead_owns_stop(stop_ahead(), radar_state(d_one=31.0, v_one=nan), 10.0, 50.0)
    assert lead_owns_stop(stop_ahead(), radar_state(d_one=31.0, v_one=14.0, d_two=nan), 10.0, 50.0)
    md = stop_ahead(((0.6, 80.0, 0.0),))
    md.leadsV3[0].x = [80.0, 80.0, nan, 80.0, 80.0, 80.0]
    assert lead_owns_stop(md, through, 10.0, 50.0)
    md = stop_ahead(((0.6, 80.0, 0.0),))
    md.leadsV3[0].x = [80.0] * 3
    assert lead_owns_stop(md, through, 10.0, 50.0)
    assert lead_owns_stop(stop_ahead(((nan, 80.0, 0.0),)), through, 10.0, 50.0)
    assert not lead_owns_stop(stop_ahead(((0.6, 80.0, 0.0),)), through, 10.0, 50.0)

  def test_a_short_hypothesis_list_fails_closed(self):
    # with no likely hypothesis the radar alone decides; a list the model did not fill is no such reading
    through = radar_state(d_one=31.0, v_one=14.0)
    for leads in ((), ((0.9, 200.0, 0.0),), ((0.9, 200.0, 0.0),) * 2):
      md = model(path_end=50.0, terminal_speed=0.2, desired_accel=-0.3, leads=leads)
      assert lead_owns_stop(md, through, 10.0, 50.0)
      assert observe(md, rs=through).lead_owns_stop
    assert not lead_owns_stop(stop_ahead(), through, 10.0, 50.0)


class TestRelease:
  def test_release_open_is_a_long_moving_non_stop_plan(self):
    assert stop_release_open(model(path_end=60.0, terminal_speed=8.0))
    assert not stop_release_open(model(path_end=60.0, terminal_speed=8.0, should_stop=True))
    assert not stop_release_open(model(path_end=15.0, terminal_speed=8.0))
    assert not stop_release_open(model(path_end=60.0, terminal_speed=2.0))
    assert stop_release_open(model(path_end=60.0, terminal_speed=8.0, desired_accel=-1.0))

  def test_every_positive_probability_hypothesis_counts(self):
    far = (0.9, 200.0, 0.0)
    assert leads_clear_of_stop_path(model(path_end=40.0, leads=(far, (0.0, 20.0, 0.0), (0.0, 20.0, 0.0))), 40.0)
    assert not leads_clear_of_stop_path(model(path_end=40.0, leads=(far, (0.2, 37.0, 0.3), (0.0, 20.0, 0.0))), 40.0)
    assert leads_clear_of_stop_path(model(path_end=40.0, leads=(far, (0.6, 37.0, 3.0), (0.0, 20.0, 0.0))), 40.0)
    assert leads_clear_of_stop_path(model(path_end=40.0, leads=(far, (0.6, 55.0, 0.0), (0.0, 20.0, 0.0))), 40.0)

  def test_malformed_paths_and_hypotheses_fail_closed(self):
    md = model(path_end=40.0, leads=((0.0, 20.0, 0.0),) * 3)
    md.position.x = [float(N - i) for i in range(N)]
    assert not leads_clear_of_stop_path(md, 40.0)
    # a flat path is what the model publishes at standstill; a lead straight ahead still blocks, one beside does not
    flat = model(path_end=40.0, leads=((0.5, 10.0, 0.5), (0.0, 20.0, 0.0), (0.0, 20.0, 0.0)))
    flat.position.x = [0.0] * N
    assert not leads_clear_of_stop_path(flat, 0.0)
    flat.leadsV3[0].y = [3.0] * len(ModelConstants.LEAD_T_IDXS)
    assert leads_clear_of_stop_path(flat, 0.0)
    assert not leads_clear_of_stop_path(model(path_end=40.0, leads=((0.0, 20.0, 0.0),) * 2), 40.0)
    md = model(path_end=40.0, leads=((0.5, 20.0, 5.0),) * 3)
    md.leadsV3[0].x = [20.0] * 3
    assert not leads_clear_of_stop_path(md, 40.0)
    assert observe(md).corridor_clear is False
    assert observe(model(path_end=40.0, leads=((0.0, 20.0, 0.0),) * 3)).corridor_clear

  def test_observation_carries_release_and_corridor_facts(self):
    obs = observe(model(path_end=60.0, terminal_speed=8.0, leads=((0.0, 20.0, 0.0),) * 3))
    assert obs.release_open and obs.corridor_clear and obs.terminal_moving and not obs.braking
    assert math.isclose(obs.path_end, 60.0, rel_tol=1e-6, abs_tol=1e-9)


class TestLaneChanging:
  def test_reads_the_models_lane_change_state(self):
    # the enum lives at log.LaneChangeState; a wrong path here silently disabled the lane-change re-qualification for three days
    from openpilot.selfdrive.controls.lib.stop_helpers import lane_changing
    md = messaging.new_message('modelV2').modelV2
    assert not lane_changing(md)
    md.meta.laneChangeState = log.LaneChangeState.laneChangeStarting
    assert lane_changing(md)


def speeds(md, start, low, end):
  # the plan's speed profile: down from start to low at its middle, then on to end
  mid = (N - 1) // 2
  md.velocity.x = [start + (low - start) * i / mid if i <= mid else low + (end - low) * (i - mid) / (N - 1 - mid) for i in range(N)]
  return md


class TestSlowing:
  def test_a_dip_and_relaunch_is_slowing_though_its_end_reads_open(self):
    obs = observe(speeds(model(path_end=40.0), 8.0, 1.0, 6.0), car_state(8.0))
    assert obs.slowing and obs.release_open

  def test_the_low_point_must_shed_most_of_the_speed(self):
    assert observe(model(path_end=85.0, terminal_speed=3.5), car_state(12.5)).slowing
    assert not observe(model(path_end=85.0, terminal_speed=5.0), car_state(12.5)).slowing
    # near the line the low point must also sit SLOWING_MARGIN below the car
    assert observe(speeds(model(path_end=5.0), 2.0, 0.6, 0.6), car_state(2.0)).slowing
    assert not observe(speeds(model(path_end=2.0), 1.2, 0.3, 0.3), car_state(1.2)).slowing

  def test_standstill_and_an_accelerating_plan_are_not_slowing(self):
    assert not observe(speeds(model(path_end=0.5), 0.0, -0.05, 0.0), car_state(0.0)).slowing
    assert not observe(speeds(model(path_end=60.0), 6.0, 6.0, 12.0), car_state(6.0)).slowing
    assert not observe(messaging.new_message('modelV2').modelV2).slowing

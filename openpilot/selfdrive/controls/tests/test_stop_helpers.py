import math

import pytest

import openpilot.cereal.messaging as messaging
from openpilot.selfdrive.controls.lib.stop_helpers import (STOP_DIRECT_CONFIDENCE, STOP_EARLY_CONFIDENCE, STOP_EARLY_HINT_CONFIDENCE,
                                                           STOP_EARLY_HINT_ENTRY_CONFIDENCE, STOP_TRAJECTORY_CONFIDENCE,
                                                           leads_clear_of_stop_path, observe_model_stop, stop_release_open)
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


def radar_state(d_one=None, d_two=None):
  rs = messaging.new_message('radarState').radarState
  for lead, d in ((rs.leadOne, d_one), (rs.leadTwo, d_two)):
    if d is not None:
      lead.present = True
      lead.dRel = d
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
    assert pytest.approx(60.0) == obs.path_end

import math

import numpy as np

from openpilot.cereal import log
import openpilot.cereal.messaging as messaging
from openpilot.selfdrive.controls.lib.longitudinal_lead import (LEAD_T_IDXS, LeadObservation, anchor_model_lead, closing_decel_requirement,
                                                                lead_present, relevant_lead, time_to_collision, total_decel_requirement)


def radar_lead(present=True, dRel=12.0, vLead=4.0, vLeadK=4.0, aLeadK=0.0, modelProb=0.9):
  lead = log.RadarState.LeadData.new_message()
  lead.present = present
  lead.dRel = dRel
  lead.vLead = vLead
  lead.vLeadK = vLeadK
  lead.aLeadK = aLeadK
  lead.modelProb = modelProb
  return lead


def model_lead(v, x=None, prob=0.9, x_std=1.0, v_std=0.5, t=None):
  v = np.asarray(v, dtype=float)
  if x is None:
    x = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2.0 * np.diff(LEAD_T_IDXS))])
  lead = log.ModelDataV2.LeadDataV3.new_message()
  lead.prob = prob
  lead.x = [float(p) for p in x]
  lead.v = [float(s) for s in v]
  lead.xStd = [float(x_std)] * len(v)
  lead.vStd = [float(v_std)] * len(v)
  lead.t = [float(s) for s in (LEAD_T_IDXS if t is None else t)]
  return lead


class TestLeadObservation:
  def test_needs_a_live_present_lead(self):
    assert not LeadObservation.from_radar(radar_lead(), False).present
    assert not LeadObservation.from_radar(radar_lead(present=False), True).present

  def test_rejects_nonfinite_or_impossible_values(self):
    assert not LeadObservation.from_radar(radar_lead(dRel=math.nan), True).present
    assert not LeadObservation.from_radar(radar_lead(dRel=0.0), True).present
    assert not LeadObservation.from_radar(radar_lead(vLeadK=math.inf), True).present

  def test_sanitizes_filtered_values(self):
    lead = LeadObservation.from_radar(radar_lead(vLeadK=-0.2, aLeadK=-20.0, modelProb=1.4), True)
    assert lead.present
    assert lead.speed == 0.0
    assert lead.acceleration == -10.0
    assert lead.model_prob == 1.0


class TestLeadPhysics:
  def test_equal_speed_needs_nothing(self):
    assert closing_decel_requirement(0.6, LeadObservation(True, distance=2.9, speed=0.6), 2.5, 0.15) == 0.0

  def test_stopped_lead_is_v_squared_over_two_d(self):
    assert math.isclose(closing_decel_requirement(2.0, LeadObservation(True, distance=6.5, speed=0.0), 2.5, 0.15), 0.5, rel_tol=1e-6, abs_tol=1e-9)

  def test_total_requirement_takes_the_larger_of_closing_and_stopping(self):
    lead = LeadObservation(True, distance=14.0, speed=8.0, acceleration=-1.0)
    # closing: 2^2 / (2 * 10) = 0.2; stopping behind the lead's 32 m braking path: 10^2 / (2 * 42) = 1.19
    assert abs((total_decel_requirement(10.0, lead, 4.0, 1.0)) - (1.190)) <= 1e-3
    assert math.isclose(total_decel_requirement(10.0, LeadObservation(True, distance=14.0, speed=8.0), 4.0, 1.0), 0.2, rel_tol=1e-6, abs_tol=1e-9)

  def test_near_stopped_lead_does_not_sustain_its_instantaneous_deceleration(self):
    assert total_decel_requirement(3.58, LeadObservation(True, distance=11.7, speed=0.22, acceleration=-0.81), 4.0, 1.0) < 1.0

  def test_ttc_ignores_a_lead_that_is_not_closing(self):
    lead = LeadObservation(True, distance=10.0, speed=10.0)
    assert math.isinf(time_to_collision(9.0, lead))
    assert math.isclose(time_to_collision(12.0, lead), 5.0, rel_tol=1e-6, abs_tol=1e-9)


class TestLeadPresence:
  def radar_state(self, d_one=None, d_two=None):
    rs = messaging.new_message('radarState').radarState
    for lead, d in ((rs.leadOne, d_one), (rs.leadTwo, d_two)):
      if d is not None:
        lead.present = True
        lead.dRel = d
    return rs

  def test_lead_present_is_raw(self):
    assert not lead_present(self.radar_state())
    assert lead_present(self.radar_state(d_two=150.0))

  def test_relevance_scales_with_speed_and_the_stop_path(self):
    assert relevant_lead(self.radar_state(d_one=30.0), 12.0)
    assert not relevant_lead(self.radar_state(d_one=120.0), 12.0)
    assert relevant_lead(self.radar_state(d_one=120.0), 12.0, path_end=115.0)
    assert relevant_lead(self.radar_state(d_two=60.0), 20.0)


class TestModelLeadAnchor:
  def test_anchors_the_model_shape_to_radar(self):
    anchor = anchor_model_lead(model_lead([10.0, 8.0, 6.0, 4.0, 2.0, 0.0]), radar_lead(dRel=30.0, vLead=12.0, vLeadK=11.5))
    assert anchor is not None
    assert math.isclose(anchor.x[0], 30.0, rel_tol=1e-6, abs_tol=1e-9)
    assert math.isclose(anchor.v[0], 12.0, rel_tol=1e-6, abs_tol=1e-9)
    assert math.isclose(anchor.v[-1], 2.0, rel_tol=1e-6, abs_tol=1e-9)
    assert math.isclose(anchor.accel, -1.0, rel_tol=1e-6, abs_tol=1e-9)
    assert math.isclose(anchor.speed, 11.5 - 2.0, rel_tol=1e-6, abs_tol=1e-9)

  def test_predicted_speed_never_goes_negative(self):
    anchor = anchor_model_lead(model_lead([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]), radar_lead(dRel=8.0, vLead=1.9, vLeadK=0.1))
    assert anchor is not None
    assert anchor.speed == 0.0

  def test_rejects_untrusted_or_malformed_forecasts(self):
    for bad in ({'prob': 0.5}, {'prob': 1.1}, {'x_std': 60.0}, {'v_std': 20.0}, {'t': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}):
      assert anchor_model_lead(model_lead([10.0] * 6, **bad), radar_lead(dRel=30.0, vLead=10.0)) is None, bad

  def test_rejects_a_shape_that_contradicts_its_speeds(self):
    assert anchor_model_lead(model_lead([10.0] * 6, x=[0.0] * 6), radar_lead(dRel=30.0, vLead=10.0)) is None

  def test_tolerates_stationary_sensor_noise_but_not_a_reversing_lead(self):
    # a stopped lead reads a few cm/s below zero on radar and in the model; the departure forecast must survive that
    noisy = model_lead([-0.04, 2.6, 4.0, 5.0, 6.0, 7.0])
    anchor = anchor_model_lead(noisy, radar_lead(dRel=6.3, vLead=-0.03, vLeadK=-0.01))
    assert anchor is not None
    assert anchor.v[0] == 0.0 and math.isclose(anchor.x[0], 6.3, rel_tol=1e-6, abs_tol=1e-9)
    assert math.isclose(anchor.speed, 2.64, rel_tol=1e-6, abs_tol=1e-9)
    assert anchor_model_lead(model_lead([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0]), radar_lead(dRel=6.3, vLead=0.0)) is None
    assert anchor_model_lead(noisy, radar_lead(dRel=6.3, vLead=-0.5)) is None

  def test_needs_a_confirmed_radar_lead(self):
    forecast = model_lead([10.0] * 6)
    assert anchor_model_lead(forecast, radar_lead(present=False)) is None
    assert anchor_model_lead(forecast, radar_lead(dRel=30.0, vLead=10.0, modelProb=0.4)) is None
    assert anchor_model_lead(forecast, radar_lead(dRel=30.0, vLead=-1.0)) is None
    assert anchor_model_lead(forecast, radar_lead(dRel=30.0, vLead=math.nan)) is None

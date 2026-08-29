from dataclasses import dataclass
import math

import numpy as np

from openpilot.selfdrive.modeld.constants import ModelConstants

LEAD_T_IDXS = np.array(ModelConstants.LEAD_T_IDXS)
MODEL_LEAD_PROB_MIN = 0.5
MODEL_LEAD_X_STD_MAX = 50.0
MODEL_LEAD_V_STD_MAX = 10.0
MODEL_LEAD_V_DELTA_MAX = 5.0
MODEL_LEAD_STATIONARY_NOISE = 0.2  # m/s; a stopped lead reads a few cm/s below zero on both sensors, a reversing one reads much less
# a lead this close in time or distance keeps stop handling with the lead logic
LEAD_RELEVANCE_MIN_DISTANCE = 35.0
LEAD_RELEVANCE_TIME = 3.5
LEAD_PATH_MARGIN = 10.0


@dataclass(frozen=True, slots=True)
class LeadObservation:
  # finite, filtered values from one valid radarState sample; the MPC obstacle uses the raw vLead as stock does
  present: bool = False
  distance: float = math.inf
  speed: float = 0.0
  acceleration: float = 0.0
  model_prob: float = 0.0

  @classmethod
  def from_radar(cls, lead, service_valid):
    if not service_valid or not lead.present:
      return cls()
    values = (lead.dRel, lead.vLeadK, lead.aLeadK, lead.modelProb)
    if not all(math.isfinite(v) for v in values) or lead.dRel <= 0.0:
      return cls()
    return cls(True, lead.dRel, max(lead.vLeadK, 0.0), float(np.clip(lead.aLeadK, -10.0, 5.0)), float(np.clip(lead.modelProb, 0.0, 1.0)))


@dataclass(frozen=True, slots=True)
class ModelLeadAnchor:
  # the model's future for a radar-confirmed lead: shape relative to its first sample, plus the first-horizon change
  x: np.ndarray
  v: np.ndarray
  accel: float
  speed: float


def lead_present(radar_state):
  return radar_state.leadOne.present or radar_state.leadTwo.present


def relevant_lead(radar_state, v_ego, path_end=None):
  limit = max(LEAD_RELEVANCE_MIN_DISTANCE, v_ego * LEAD_RELEVANCE_TIME)
  if path_end is not None:
    limit = max(limit, path_end + LEAD_PATH_MARGIN)
  return any(lead.present and 0.0 <= lead.dRel <= limit for lead in (radar_state.leadOne, radar_state.leadTwo))


def anchor_model_lead(model_lead, radar_lead):
  if not radar_lead.present:
    return None
  x = np.asarray(model_lead.x, dtype=np.float64)
  x_std = np.asarray(model_lead.xStd, dtype=np.float64)
  v = np.asarray(model_lead.v, dtype=np.float64)
  v_std = np.asarray(model_lead.vStd, dtype=np.float64)
  t = np.asarray(model_lead.t, dtype=np.float64)
  scalars = (model_lead.prob, radar_lead.modelProb, radar_lead.dRel, radar_lead.vLead, radar_lead.vLeadK)
  valid = (all(math.isfinite(s) for s in scalars)
           and MODEL_LEAD_PROB_MIN < model_lead.prob <= 1.0
           and MODEL_LEAD_PROB_MIN < radar_lead.modelProb <= 1.0
           and radar_lead.dRel > 0.0 and radar_lead.vLead >= -MODEL_LEAD_STATIONARY_NOISE
           and all(a.shape == LEAD_T_IDXS.shape and np.all(np.isfinite(a)) for a in (x, x_std, v, v_std, t))
           and np.array_equal(t, LEAD_T_IDXS)
           and np.all(x_std >= 0.0) and np.max(x_std) < MODEL_LEAD_X_STD_MAX
           and np.all(v_std >= 0.0) and np.max(v_std) < MODEL_LEAD_V_STD_MAX
           and np.all(np.diff(x) >= -MODEL_LEAD_STATIONARY_NOISE * np.diff(LEAD_T_IDXS)) and np.all(v >= -MODEL_LEAD_STATIONARY_NOISE)
           and np.max(np.abs(np.gradient(x, LEAD_T_IDXS, edge_order=2) - v)) <= MODEL_LEAD_V_DELTA_MAX)
  if not valid:
    return None
  # the model contributes the future shape; radar anchors where the lead is and how fast it goes now.
  # Sensor noise on a stopped lead is tolerated above and squashed here: the MPC only knows forward motion
  x_lead = np.maximum.accumulate(radar_lead.dRel + x - x[0])
  v_lead = np.maximum(max(radar_lead.vLead, 0.0) + v - v[0], 0.0)
  if np.max(np.abs(np.gradient(x_lead, LEAD_T_IDXS, edge_order=2) - v_lead)) > MODEL_LEAD_V_DELTA_MAX:
    return None
  horizon = LEAD_T_IDXS[1] - LEAD_T_IDXS[0]
  return ModelLeadAnchor(x_lead, v_lead, float((v[1] - v[0]) / horizon), max(max(radar_lead.vLeadK, 0.0) + v[1] - v[0], 0.0))


def closing_speed(v_ego, lead):
  return max(v_ego - lead.speed, 0.0) if lead.present else 0.0


def closing_decel_requirement(v_ego, lead, stop_distance, min_gap_budget):
  # relative-frame physics: equal-speed queue motion needs nothing, a stopped lead gives v^2 / 2d
  if not lead.present:
    return 0.0
  closing = closing_speed(v_ego, lead)
  return closing * closing / (2.0 * max(lead.distance - stop_distance, min_gap_budget))


def total_decel_requirement(v_ego, lead, stop_distance, min_gap_budget):
  # the larger of shedding the closing speed and stopping behind the lead's own finite braking path
  if not lead.present:
    return 0.0
  closing_requirement = closing_decel_requirement(v_ego, lead, stop_distance, min_gap_budget)
  lead_braking = max(-lead.acceleration, 0.0)
  if lead_braking == 0.0:
    return closing_requirement
  gap_budget = max(lead.distance - stop_distance, min_gap_budget)
  lead_stop_distance = lead.speed * lead.speed / (2.0 * lead_braking)
  return max(closing_requirement, v_ego * v_ego / (2.0 * (gap_budget + lead_stop_distance)))


def time_to_collision(v_ego, lead, min_closing_speed=0.3):
  if not lead.present:
    return math.inf
  closing = closing_speed(v_ego, lead)
  return lead.distance / closing if closing > min_closing_speed else math.inf

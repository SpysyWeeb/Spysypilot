from dataclasses import dataclass
import math

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_lead import LEAD_PATH_MARGIN, lead_present, relevant_lead, relevant_leads
from openpilot.selfdrive.modeld.constants import ModelConstants

# The deployed model publishes no traffic-light or stop-sign class. These tiers describe a generic,
# temporally persistent model stop intent from action.shouldStop and the 10 s path, nothing more.
STOP_PREDICTION_HORIZON_S = 5.0
STOP_PATH_MIN_DISTANCE = 4.0
STOP_TERMINAL_SPEED_MAX = 1.0

# at urban speed the strict tier recognizes a stop too late for a comfortable approach; the early tier
# reads the shape of the intent instead, on straight, unsignaled, lead-free approaches only
STOP_EARLY_MIN_SPEED = 13.0
STOP_EARLY_COMFORT_DECEL = 1.3
STOP_EARLY_RESPONSE_BUFFER_S = 0.5
STOP_EARLY_TERMINAL_SPEED_RATIO = 0.35
STOP_EARLY_TERMINAL_SPEED_MAX = 6.5
STOP_EARLY_DESIRED_ACCEL_MAX = -0.5
STOP_EARLY_MAX_LATERAL_ACCEL = 1.0
STOP_EARLY_MAX_HEADING_CHANGE = math.radians(20.0)
STOP_EARLY_HINT_HORIZON_S = 8.0
STOP_EARLY_HINT_TERMINAL_SPEED_RATIO = 0.55
STOP_EARLY_HINT_DESIRED_ACCEL_MAX = -0.25
STOP_EARLY_HINT_ENTRY_MAX_SPEED = 22.0

STOP_DIRECT_CONFIDENCE = 1.0
STOP_TRAJECTORY_CONFIDENCE = 0.85
STOP_EARLY_CONFIDENCE = 0.80
STOP_EARLY_HINT_CONFIDENCE = 0.45
STOP_EARLY_HINT_ENTRY_CONFIDENCE = 0.70
STOP_SAMPLE_MIN_CONFIDENCE = 0.70

STOP_RELEASE_PATH_MIN_DISTANCE = 20.0
PATH_OPEN_LENGTH = 30.0    # m, at standstill a raw path this long reads as "the way ahead is open" (stubs sit at 2-5 m; greens 30-60 m)
STOP_RELEASE_TERMINAL_SPEED_MIN = 3.0
SLOWING_MARGIN = 1.0       # m/s, the plan's low point must sit at least this far below the car's speed to read as slowing
STOP_COMMIT_MAX_DISTANCE = 100.0
LEAD_STOP_PATH_HALF_WIDTH = 1.5
LEAD_CRAWL_SPEED = 1.0     # m/s, a relevant lead slower than this is stopped or creeping: a queue, which owns the stop
MODEL_INVALID_RELEASE_S = 0.5

# below the early tiers' speed, with nobody ahead, a plan that rolls on slowly to a short end (route 0x1f8 t=536.5-540.2: the lead
# changed lanes toward a red light with the car at 13 mph, the plan then ended 37-42 m out at 2.7-4.0 m/s with the model asking 0 to
# +0.17, no tier read it and Chill's cruise accelerated). Not a tier: CEM alone reads it (D50). SLOW_PLAN_MAX_DISTANCE = 0.0 switches
# the reading off
SLOW_PLAN_MAX_DISTANCE = 60.0      # m, the plan ends within this
SLOW_PLAN_MAX_END_SPEED = 4.5      # m/s, at or below this, and no faster than the car is going now
SLOW_PLAN_MAX_DESIRED_ACCEL = 0.3  # m/s^2, the model asking no more than this
SLOW_PLAN_MIN_SPEED = 1.0          # m/s, the car rolling above this: at a crawl the strict tier and the stop bit decide, as before
SLOW_PLAN_HOLD_MAX_END_SPEED = 6.5  # m/s, the end speed limit for holding a mode that is on (slow_plan_hold); entry keeps 4.5. A single
                                    # limit was a cliff: 0x1f8 t=539.3-540.4 the plan's end rose to 4.5-6.4 m/s with the model asking
                                    # -0.15 to -0.42, and the mode ended itself a tick before the driver's brake. 4.5 removes the margin

COMMITTED_TURN_MAX_SPEED = 8.0
COMMITTED_TURN_MIN_STEERING_DEG = 35.0
COMMITTED_TURN_MIN_CURVATURE = 0.04


@dataclass(frozen=True, slots=True)
class StopObservation:
  confidence: float = 0.0
  path_end: float | None = None
  should_stop: bool = False
  strict_stop: bool = False
  early_stop: bool = False
  complete: bool = False
  braking: bool = False
  terminal_moving: bool = False
  relevant_lead: bool = False
  lead_present: bool = False
  committed_turn: bool = False
  release_open: bool = False
  corridor_clear: bool = False
  lane_change: bool = False     # the model is changing lanes: its endpoint is about to belong to another lane
  slowing: bool = False         # the plan's low point sheds most of the car's speed, wherever the plan ends
  lead_owns_stop: bool = False  # a relevant lead is a queue, or a lead cannot be read (lead_owns_stop below)
  slowing_approach: bool = False  # below STOP_EARLY_MIN_SPEED: slowing, straight, toward a point ahead, the model braking (D47; CEM entry only)
  slow_plan: bool = False         # below STOP_EARLY_MIN_SPEED: straight, a short plan rolling on no faster than the car, the model not
                                  # asking to speed up (D50; CEM alone reads it)
  slow_plan_hold: bool = False    # slow_plan with the hold's end speed limit, SLOW_PLAN_HOLD_MAX_END_SPEED (D50; CEM's hold alone)


def _finite(values):
  return all(math.isfinite(v) for v in values)


def model_complete_and_finite(model):
  return (len(model.position.x) == ModelConstants.IDX_N and len(model.velocity.x) == ModelConstants.IDX_N
          and _finite(model.position.x) and _finite(model.velocity.x))


def path_open(path_end):
  # the one "road opened" predicate on the raw path length; stop_release_open below is its moving-path form
  return path_end is not None and path_end > PATH_OPEN_LENGTH


def stop_release_open(model):
  # a strong launch sample: the model no longer wants to stop and plans a long, moving path
  return (model_complete_and_finite(model) and not model.action.shouldStop and math.isfinite(model.action.desiredAcceleration)
          and model.position.x[ModelConstants.IDX_N - 1] >= STOP_RELEASE_PATH_MIN_DISTANCE
          and model.velocity.x[ModelConstants.IDX_N - 1] >= STOP_RELEASE_TERMINAL_SPEED_MIN)


def leads_clear_of_stop_path(model, path_end):
  # fail closed unless every model lead hypothesis with any probability sits outside the stop corridor
  path_x, path_y, leads = list(model.position.x), list(model.position.y), model.leadsV3
  n_lead = len(ModelConstants.LEAD_T_IDXS)
  if (len(path_x) != ModelConstants.IDX_N or len(path_y) != ModelConstants.IDX_N or len(leads) != ModelConstants.LEAD_MHP_SELECTION
      or not _finite(path_x) or not _finite(path_y) or any(b < a for a, b in zip(path_x, path_x[1:], strict=False))):
    return False
  for lead in leads:
    lead_x, lead_y = list(lead.x), list(lead.y)
    if (not math.isfinite(lead.prob) or not 0.0 <= lead.prob <= 1.0 or len(lead_x) != n_lead or len(lead_y) != n_lead
        or not _finite(lead_x) or not _finite(lead_y)):
      return False
    if lead.prob == 0.0:
      continue
    for x, y in zip(lead_x, lead_y, strict=True):
      if x <= 0.0:
        return False
      if x > path_end + LEAD_PATH_MARGIN:
        continue
      if x <= path_x[0]:
        center = path_y[0]
      elif x >= path_x[-1]:
        center = path_y[-1]
      else:
        i = next(i for i, px in enumerate(path_x) if px >= x)
        span = path_x[i] - path_x[i - 1]
        center = path_y[i - 1] + (x - path_x[i - 1]) / span * (path_y[i] - path_y[i - 1]) if span > 0.0 else path_y[i]
      if abs(y - center) <= LEAD_STOP_PATH_HALF_WIDTH:
        return False
  return True


def lead_owns_stop(radar_state, v_ego, path_end):
  # a relevant radar lead owns the stop only as a queue, stopped or creeping below LEAD_CRAWL_SPEED; a moving lead does not,
  # wherever it would come to rest (D46). Fails closed on a missing stop point (the relevance window) or a lead whose place or
  # speed cannot be read
  if path_end is None or not math.isfinite(path_end) or path_end <= 0.0:
    return True
  if not all(math.isfinite(lead.dRel) and math.isfinite(lead.vLead) for lead in (radar_state.leadOne, radar_state.leadTwo) if lead.present):
    return True
  return any(lead.vLead < LEAD_CRAWL_SPEED for lead in relevant_leads(radar_state, v_ego, path_end))


def observe_model_stop(model, car_state, radar_state):
  if not model_complete_and_finite(model) or not math.isfinite(car_state.vEgo):
    return StopObservation(lead_present=lead_present(radar_state))
  v_ego = max(car_state.vEgo, 0.0)
  action = model.action
  desired_accel = action.desiredAcceleration if math.isfinite(action.desiredAcceleration) else 0.0
  last = ModelConstants.IDX_N - 1
  path_end = model.position.x[last]
  terminal_speed = model.velocity.x[last]
  terminal_heading = model.orientation.z[last] if len(model.orientation.z) == ModelConstants.IDX_N else math.nan
  # judged on the plan's low point, not its end: a plan that dips toward the line and relaunches is still slowing for it
  low_speed = min(list(model.velocity.x)[1:])
  slowing = v_ego - low_speed >= SLOWING_MARGIN and low_speed <= v_ego * STOP_EARLY_TERMINAL_SPEED_RATIO

  blinker = car_state.leftBlinker or car_state.rightBlinker
  lateral_accel = abs(action.desiredCurvature) * v_ego ** 2
  straight = (not blinker and math.isfinite(lateral_accel) and lateral_accel <= STOP_EARLY_MAX_LATERAL_ACCEL
              and math.isfinite(terminal_heading) and abs(terminal_heading) <= STOP_EARLY_MAX_HEADING_CHANGE)

  in_horizon = 0.0 < path_end <= max(STOP_PATH_MIN_DISTANCE, v_ego * STOP_PREDICTION_HORIZON_S)
  strict_stop = in_horizon and terminal_speed <= STOP_TERMINAL_SPEED_MAX
  early_path = 0.0 < path_end <= v_ego ** 2 / (2.0 * STOP_EARLY_COMFORT_DECEL) + v_ego * STOP_EARLY_RESPONSE_BUFFER_S
  early_stop = (v_ego >= STOP_EARLY_MIN_SPEED and straight and early_path and desired_accel <= STOP_EARLY_DESIRED_ACCEL_MAX
                and terminal_speed <= min(STOP_EARLY_TERMINAL_SPEED_MAX, v_ego * STOP_EARLY_TERMINAL_SPEED_RATIO))
  early_hint = (v_ego >= STOP_EARLY_MIN_SPEED and straight and 0.0 < path_end <= v_ego * STOP_EARLY_HINT_HORIZON_S
                and terminal_speed <= v_ego * STOP_EARLY_HINT_TERMINAL_SPEED_RATIO and desired_accel <= STOP_EARLY_HINT_DESIRED_ACCEL_MAX)
  # below the early tiers' speed a plan that slows toward a point ahead may still end above the strict tier's 1.0 m/s, creeping on
  # past its low point (route 0x1ef t=209.3-213.9: no tier read a red light). Not a tier: CEM alone reads it, as entry evidence (D47)
  slowing_approach = (v_ego < STOP_EARLY_MIN_SPEED and slowing and straight and path_end > 0.0
                      and desired_accel <= STOP_EARLY_HINT_DESIRED_ACCEL_MAX)
  # the plan's reading alone, as slowing_approach: the lead condition is CEM's. A request that cannot be read is no reading
  slow_rolling = (SLOW_PLAN_MIN_SPEED < v_ego < STOP_EARLY_MIN_SPEED and straight and 0.0 < path_end <= SLOW_PLAN_MAX_DISTANCE
                  and terminal_speed <= v_ego and math.isfinite(action.desiredAcceleration)
                  and action.desiredAcceleration <= SLOW_PLAN_MAX_DESIRED_ACCEL)
  slow_plan = slow_rolling and terminal_speed <= SLOW_PLAN_MAX_END_SPEED
  slow_plan_hold = slow_rolling and terminal_speed <= max(SLOW_PLAN_MAX_END_SPEED, SLOW_PLAN_HOLD_MAX_END_SPEED)

  if action.shouldStop:
    confidence = STOP_DIRECT_CONFIDENCE
  elif strict_stop:
    confidence = STOP_TRAJECTORY_CONFIDENCE
  elif early_stop:
    confidence = STOP_EARLY_CONFIDENCE
  elif early_hint:
    confidence = STOP_EARLY_HINT_ENTRY_CONFIDENCE if v_ego <= STOP_EARLY_HINT_ENTRY_MAX_SPEED else STOP_EARLY_HINT_CONFIDENCE
  else:
    confidence = 0.0

  steering_angle = abs(car_state.steeringAngleDeg) if math.isfinite(car_state.steeringAngleDeg) else 0.0
  committed_turn = blinker and v_ego <= COMMITTED_TURN_MAX_SPEED and (steering_angle >= COMMITTED_TURN_MIN_STEERING_DEG
                                                                        or abs(action.desiredCurvature) >= COMMITTED_TURN_MIN_CURVATURE)
  relevant = relevant_lead(radar_state, v_ego, path_end)
  return StopObservation(confidence, path_end, bool(action.shouldStop), strict_stop, early_stop, True,
                         desired_accel <= STOP_EARLY_DESIRED_ACCEL_MAX, terminal_speed >= STOP_TERMINAL_SPEED_MAX,
                         relevant, lead_present(radar_state), committed_turn, stop_release_open(model),
                         path_end > 0.0 and leads_clear_of_stop_path(model, path_end), lane_changing(model), slowing,
                         relevant and lead_owns_stop(radar_state, v_ego, path_end), slowing_approach, slow_plan, slow_plan_hold)


def lane_changing(model):
  # log.LaneChangeState is the top-level enum (as controlsd and desire_helper read it); the earlier guarded lookup under
  # LateralPlan never resolved and returned False on every frame, so the lane-change re-qualification (D21) was inert until 2026-09-02
  return model.meta.laneChangeState != log.LaneChangeState.off

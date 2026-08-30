from dataclasses import dataclass
import math

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_lead import lead_present, relevant_lead
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
STOP_RELEASE_TERMINAL_SPEED_MIN = 3.0
STOP_COMMIT_MAX_DISTANCE = 100.0
LEAD_STOP_PATH_HALF_WIDTH = 1.5
LEAD_PATH_MARGIN = 10.0
MODEL_INVALID_RELEASE_S = 0.5

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


def _finite(values):
  return all(math.isfinite(v) for v in values)


def model_complete_and_finite(model):
  return (len(model.position.x) == ModelConstants.IDX_N and len(model.velocity.x) == ModelConstants.IDX_N
          and _finite(model.position.x) and _finite(model.velocity.x))


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
  return StopObservation(confidence, path_end, bool(action.shouldStop), strict_stop, early_stop, True,
                         desired_accel <= STOP_EARLY_DESIRED_ACCEL_MAX, terminal_speed >= STOP_TERMINAL_SPEED_MAX,
                         relevant_lead(radar_state, v_ego, path_end),
                         lead_present(radar_state), committed_turn, stop_release_open(model),
                         path_end > 0.0 and leads_clear_of_stop_path(model, path_end), lane_changing(model))


def lane_changing(model):
  try:
    return model.meta.laneChangeState != log.LateralPlan.LaneChangeState.off
  except (AttributeError, TypeError, ValueError):
    return False

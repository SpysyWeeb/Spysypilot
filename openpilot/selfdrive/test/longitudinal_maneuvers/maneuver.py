import numpy as np
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant


class Maneuver:
  def __init__(self, title, duration, **kwargs):
    # Was tempted to make a builder class
    self.distance_lead = kwargs.get("initial_distance_lead", 200.0)
    self.speed = kwargs.get("initial_speed", 0.0)
    self.lead_relevancy = kwargs.get("lead_relevancy", 0)

    self.breakpoints = kwargs.get("breakpoints", [0.0, duration])
    self.speed_lead_values = kwargs.get("speed_lead_values", [0.0 for i in range(len(self.breakpoints))])
    self.prob_lead_values = kwargs.get("prob_lead_values", [1.0 for i in range(len(self.breakpoints))])
    self.prob_throttle_values = kwargs.get("prob_throttle_values", [1.0 for i in range(len(self.breakpoints))])
    self.cruise_values = kwargs.get("cruise_values", [50.0 for i in range(len(self.breakpoints))])
    self.pitch_values = kwargs.get("pitch_values", [0.0 for i in range(len(self.breakpoints))])

    self.radar_valid_breakpoints = kwargs.get("radar_valid_breakpoints", self.breakpoints)
    self.radar_valid_values = kwargs.get("radar_valid_values", [1.0 for i in range(len(self.radar_valid_breakpoints))])
    self.model_valid_breakpoints = kwargs.get("model_valid_breakpoints", self.breakpoints)
    self.model_valid_values = kwargs.get("model_valid_values", [1.0 for i in range(len(self.model_valid_breakpoints))])

    self.only_lead2 = kwargs.get("only_lead2", False)
    self.only_radar = kwargs.get("only_radar", False)
    self.ensure_start = kwargs.get("ensure_start", False)
    self.ensure_slowdown = kwargs.get("ensure_slowdown", False)
    self.enabled = kwargs.get("enabled", True)
    self.e2e = kwargs.get("e2e", False)
    self.personality = kwargs.get("personality", 0)
    self.force_decel = kwargs.get("force_decel", False)
    self.stop_line = kwargs.get("stop_line", None)
    self.stop_line_horizon_s = kwargs.get("stop_line_horizon_s", 5.0)
    self.curve = kwargs.get("curve", None)
    self.torque_factor = kwargs.get("torque_factor", 2.7)
    self.torque_friction = kwargs.get("torque_friction", 0.11)
    self.curve_model_scale = kwargs.get("curve_model_scale", 1.0)
    self.e2e_landing_push = kwargs.get("e2e_landing_push", 0.0)
    self.actuator_lag = kwargs.get("actuator_lag", None)

    self.duration = duration
    self.title = title

  def evaluate(self):
    plant = Plant(
      lead_relevancy=self.lead_relevancy,
      speed=self.speed,
      distance_lead=self.distance_lead,
      enabled=self.enabled,
      only_lead2=self.only_lead2,
      only_radar=self.only_radar,
      e2e=self.e2e,
      personality=self.personality,
      force_decel=self.force_decel,
      stop_line=self.stop_line,
      stop_line_horizon_s=self.stop_line_horizon_s,
      curve=self.curve,
      torque_factor=self.torque_factor,
      torque_friction=self.torque_friction,
      curve_model_scale=self.curve_model_scale,
      e2e_landing_push=self.e2e_landing_push,
      actuator_lag=self.actuator_lag,
    )

    valid = True
    logs = []
    not_starting_t = 0.0
    while plant.current_time < self.duration:
      speed_lead = np.interp(plant.current_time, self.breakpoints, self.speed_lead_values)
      prob_lead = np.interp(plant.current_time, self.breakpoints, self.prob_lead_values)
      cruise = np.interp(plant.current_time, self.breakpoints, self.cruise_values)
      pitch = np.interp(plant.current_time, self.breakpoints, self.pitch_values)
      prob_throttle = np.interp(plant.current_time, self.breakpoints, self.prob_throttle_values)
      radar_valid = bool(np.interp(plant.current_time, self.radar_valid_breakpoints, self.radar_valid_values) > 0.5)
      model_valid = bool(np.interp(plant.current_time, self.model_valid_breakpoints, self.model_valid_values) > 0.5)
      log = plant.step(speed_lead, prob_lead, cruise, pitch, prob_throttle, radar_valid, model_valid)

      d_rel = log['distance_lead'] - log['distance'] if self.lead_relevancy else 200.
      v_rel = speed_lead - log['speed'] if self.lead_relevancy else 0.
      log['d_rel'] = d_rel
      log['v_rel'] = v_rel
      logs.append(np.array([plant.current_time,
                            log['distance'],
                            log['distance_lead'],
                            log['speed'],
                            speed_lead,
                            log['acceleration'],
                            log['d_rel']]))

      if d_rel < .4 and (self.only_radar or prob_lead > 0.5):
        print("Crashed!!!!")
        valid = False

      # This assertion protects the launch phase. Once ego is moving, a faster
      # lead plus a brief non-positive command can be normal gap settling and is
      # not evidence that the planner failed to start.
      if self.ensure_start and log['speed'] < 0.5 and log['v_rel'] > 0 and log['acceleration'] < 1e-3:
        if not_starting_t == 0.0:
          not_starting_t = plant.current_time
        elif plant.current_time - not_starting_t > 0.5:
          print('LongitudinalPlanner not starting!')
          valid = False
      else:
        not_starting_t = 0.0

    if self.ensure_slowdown and log['speed'] > 5.5:
      print('LongitudinalPlanner not slowing down!')
      valid = False

    if self.force_decel and log['speed'] > 1e-1 and log['acceleration'] > -0.04:
      print('Not stopping with force decel')
      valid = False


    print("maneuver end", valid)
    return valid, np.array(logs)

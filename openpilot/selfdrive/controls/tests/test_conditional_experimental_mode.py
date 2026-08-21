import copy
import math
from types import SimpleNamespace

from openpilot.selfdrive.controls.lib.conditional_experimental_mode import (
  ConditionalExperimentalMode,
  DRIVER_OVERRIDE_SUPPRESS_S,
  LEAD_RELEASE_HYSTERESIS_S,
  STOP_EARLY_COMFORT_DECEL,
  STOP_EARLY_CONFIDENCE,
  STOP_EARLY_HINT_CONFIDENCE,
  STOP_EARLY_HINT_ENTRY_MAX_SPEED,
  FORCE_STOP_COMMIT_DISTANCE_M,
  FORCE_STOP_QUALIFY_S,
  FORCE_STOP_WORLD_TOLERANCE_M,
  STOP_EARLY_MAX_HEADING_CHANGE,
  STOP_EARLY_MAX_LATERAL_ACCEL,
  STOP_EARLY_MIN_SPEED,
  STOP_EARLY_RESPONSE_BUFFER_S,
  STOP_PREDICTION_HORIZON_S,
  STOP_INTENT_HOLD_S,
  STOP_SAMPLE_MIN_CONFIDENCE,
  model_stop_release_open,
  observe_model_stop_intent,
)
from openpilot.selfdrive.modeld.constants import ModelConstants


DT = 0.05


def model(*, should_stop=False, path_end=90.0, terminal_speed=10.0, desired_accel=0.0,
          desired_curvature=0.0, terminal_heading=0.0):
  position_x = [] if path_end is None else [path_end * i / (ModelConstants.IDX_N - 1) for i in range(ModelConstants.IDX_N)]
  velocity_x = [] if terminal_speed is None else [10.0 + (terminal_speed - 10.0) * i / (ModelConstants.IDX_N - 1)
                                                   for i in range(ModelConstants.IDX_N)]
  orientation_z = [] if terminal_heading is None else [terminal_heading * i / (ModelConstants.IDX_N - 1)
                                                         for i in range(ModelConstants.IDX_N)]
  return SimpleNamespace(
    action=SimpleNamespace(
      shouldStop=should_stop,
      desiredAcceleration=desired_accel,
      desiredCurvature=desired_curvature,
    ),
    position=SimpleNamespace(x=position_x),
    velocity=SimpleNamespace(x=velocity_x),
    orientation=SimpleNamespace(z=orientation_z),
  )


def model_lead(*, prob=0.0, x=200.0, y=0.0):
  count = len(ModelConstants.LEAD_T_IDXS)
  return SimpleNamespace(prob=prob, x=[x] * count, y=[y] * count)


def with_model_leads(md, *leads):
  md.position.y = [0.0] * ModelConstants.IDX_N
  md.leadsV3 = list(leads)
  return md


def car_state(*, v_ego=10.0, standstill=False, gas=False, brake=False,
              left_blinker=False, right_blinker=False, steering_angle=0.0):
  return SimpleNamespace(
    vEgo=v_ego,
    standstill=standstill,
    gasPressed=gas,
    brakePressed=brake,
    leftBlinker=left_blinker,
    rightBlinker=right_blinker,
    steeringAngleDeg=steering_angle,
  )


def radar_state(*, present=False, distance=1000.0, secondary_present=False, secondary_distance=1000.0):
  return SimpleNamespace(
    leadOne=SimpleNamespace(present=present, dRel=distance),
    leadTwo=SimpleNamespace(present=secondary_present, dRel=secondary_distance),
  )


def new_cem():
  return ConditionalExperimentalMode(control_dt=DT, model_dt=DT)


def run_frames(cem, count, md, cs=None, radar=None, *, enabled=True, valid=True):
  cs = cs or car_state()
  radar = radar or radar_state()
  outputs = []
  for _ in range(count):
    outputs.append(cem.update(md, cs, radar, controls_enabled=enabled, model_updated=True, model_valid=valid))
  return outputs


def confirmed_stop_model(v_ego=10.0):
  return model(path_end=v_ego * 3.0, terminal_speed=0.2, desired_accel=-0.3)


def clear_model():
  return model(path_end=90.0, terminal_speed=10.0, desired_accel=0.0)


def route_d7_early_stop_model():
  # First strong high-speed evidence from route 000000d7--cc6308b4d0.
  return model(path_end=138.0, terminal_speed=5.85, desired_accel=-0.50,
               desired_curvature=-0.00007, terminal_heading=-0.001)


def activate(cem, cs=None):
  cs = cs or car_state()
  assert run_frames(cem, 30, confirmed_stop_model(cs.vEgo), cs)[-1]


def test_normal_driving_starts_and_stays_chill():
  cem = new_cem()
  assert not cem.experimental_mode
  assert not any(run_frames(cem, 60, clear_model()))


def test_trajectory_stop_enters_experimental_inside_prediction_horizon():
  cem = new_cem()
  v_ego = 10.0

  outside_horizon = model(path_end=v_ego * STOP_PREDICTION_HORIZON_S + 1.0, terminal_speed=0.0)
  assert not any(run_frames(cem, 30, outside_horizon, car_state(v_ego=v_ego)))

  inside_horizon = model(path_end=v_ego * (STOP_PREDICTION_HORIZON_S - 0.1), terminal_speed=0.0)
  assert run_frames(cem, 30, inside_horizon, car_state(v_ego=v_ego))[-1]
  assert cem.last_observation.reason == "trajectory"


def test_route_d7_high_speed_stop_enters_before_strict_terminal_threshold():
  cem = new_cem()
  v_ego = 18.95
  stop = route_d7_early_stop_model()
  observation = observe_model_stop_intent(stop, car_state(v_ego=v_ego), radar_state())

  assert stop.position.x[-1] > v_ego * STOP_PREDICTION_HORIZON_S
  assert stop.velocity.x[-1] > 1.0
  assert observation.reason == "earlyTrajectory"
  assert observation.confidence == STOP_EARLY_CONFIDENCE
  assert run_frames(cem, 30, stop, car_state(v_ego=v_ego))[-1]


def test_sustained_urban_early_hint_requests_experimental():
  cem = new_cem()
  v_ego = 20.0
  hint = model(path_end=159.3, terminal_speed=9.93, desired_accel=-0.32,
               desired_curvature=-0.00005, terminal_heading=-0.001)
  observation = observe_model_stop_intent(hint, car_state(v_ego=v_ego), radar_state())

  assert observation.reason == "earlyHint"
  assert observation.confidence >= STOP_SAMPLE_MIN_CONFIDENCE
  assert run_frames(cem, 30, hint, car_state(v_ego=v_ego))[-1]


def test_urban_early_hint_entry_speed_boundary_is_inclusive():
  hint = model(path_end=159.3, terminal_speed=9.93, desired_accel=-0.32,
               desired_curvature=-0.00005, terminal_heading=-0.001)

  at_limit = observe_model_stop_intent(hint, car_state(v_ego=STOP_EARLY_HINT_ENTRY_MAX_SPEED), radar_state())
  above_limit = observe_model_stop_intent(hint, car_state(v_ego=STOP_EARLY_HINT_ENTRY_MAX_SPEED + 0.0001), radar_state())

  assert at_limit.confidence >= STOP_SAMPLE_MIN_CONFIDENCE
  assert above_limit.confidence == STOP_EARLY_HINT_CONFIDENCE


def test_urban_early_hint_extends_beyond_comfort_envelope_only_inside_eight_seconds():
  v_ego = 18.0
  distance_limit = v_ego ** 2 / (2.0 * STOP_EARLY_COMFORT_DECEL) + v_ego * STOP_EARLY_RESPONSE_BUFFER_S
  hint = model(path_end=distance_limit + 0.1, terminal_speed=4.0, desired_accel=-0.6)
  outside_hint = model(path_end=v_ego * 8.0 + 0.1, terminal_speed=4.0, desired_accel=-0.6)
  observation = observe_model_stop_intent(hint, car_state(v_ego=v_ego), radar_state())

  assert observation.reason == "earlyHint"
  assert run_frames(new_cem(), 40, hint, car_state(v_ego=v_ego))[-1]
  assert not any(run_frames(new_cem(), 40, outside_hint, car_state(v_ego=v_ego)))


def test_highway_slowdown_does_not_qualify_as_an_early_stop():
  # Route d7 segment 22 predicts slowing for a 25 mph exit but not stopping.
  # Its terminal velocity must remain weak filter-only evidence.
  v_ego = 24.54
  slowdown = model(path_end=170.13, terminal_speed=8.41, desired_accel=-1.06,
                   desired_curvature=0.00015, terminal_heading=0.004)
  observation = observe_model_stop_intent(slowdown, car_state(v_ego=v_ego), radar_state())

  assert observation.reason == "earlyHint"
  assert not any(run_frames(new_cem(), 100, slowdown, car_state(v_ego=v_ego)))


def test_early_high_speed_tier_requires_straight_valid_unsignaled_geometry():
  v_ego = 18.95
  high_curvature = (STOP_EARLY_MAX_LATERAL_ACCEL + 0.1) / v_ego ** 2
  guarded_models_and_states = (
    (model(path_end=138.0, terminal_speed=5.85, desired_accel=-0.5,
           desired_curvature=high_curvature), car_state(v_ego=v_ego)),
    (model(path_end=138.0, terminal_speed=5.85, desired_accel=-0.5,
           terminal_heading=STOP_EARLY_MAX_HEADING_CHANGE + 0.01), car_state(v_ego=v_ego)),
    (model(path_end=138.0, terminal_speed=5.85, desired_accel=-0.5,
           desired_curvature=float("nan")), car_state(v_ego=v_ego)),
    (model(path_end=138.0, terminal_speed=5.85, desired_accel=-0.5,
           terminal_heading=None), car_state(v_ego=v_ego)),
    (route_d7_early_stop_model(), car_state(v_ego=v_ego, left_blinker=True)),
  )

  for guarded_model, guarded_state in guarded_models_and_states:
    observation = observe_model_stop_intent(guarded_model, guarded_state, radar_state())
    assert observation.reason not in ("earlyTrajectory", "earlyHint")
    assert not any(run_frames(new_cem(), 40, guarded_model, guarded_state))


def test_early_high_speed_tier_does_not_change_low_speed_detection():
  v_ego = STOP_EARLY_MIN_SPEED - 0.1
  early_shape_only = model(path_end=v_ego * 6.0, terminal_speed=3.0, desired_accel=-0.6)
  assert observe_model_stop_intent(early_shape_only, car_state(v_ego=v_ego), radar_state()).reason == "none"

  strict_stop = model(path_end=v_ego * 4.0, terminal_speed=0.5, desired_accel=-0.6)
  assert run_frames(new_cem(), 30, strict_stop, car_state(v_ego=v_ego))[-1]


def test_relevant_lead_blocks_early_high_speed_entry():
  cem = new_cem()
  v_ego = 18.95
  close_lead = radar_state(present=True, distance=80.0)

  assert not any(run_frames(cem, 50, route_d7_early_stop_model(), car_state(v_ego=v_ego), close_lead))
  assert cem.last_observation.reason == "earlyTrajectory"
  assert cem.last_observation.relevant_lead


def test_recent_lead_hysteresis_blocks_entry_during_a_short_radar_dropout():
  cem = new_cem()
  v_ego = 18.95
  cs = car_state(v_ego=v_ego)
  stop = route_d7_early_stop_model()
  run_frames(cem, 20, stop, cs, radar_state(present=True, distance=80.0))

  dropout_frames = int(LEAD_RELEASE_HYSTERESIS_S / DT) - 1
  assert not any(run_frames(cem, dropout_frames, stop, cs, radar_state()))
  assert run_frames(cem, 40, stop, cs, radar_state())[-1]


def test_early_trajectory_alone_cannot_release_recent_lead_hysteresis():
  cem = new_cem()
  v_ego = 18.95
  cs = car_state(v_ego=v_ego)
  stop = with_model_leads(
    route_d7_early_stop_model(),
    model_lead(prob=0.0902, x=92.94, y=2.11),
    model_lead(prob=0.0244, x=92.82, y=2.02),
    model_lead(prob=0.0307, x=92.68, y=1.99),
  )
  run_frames(cem, 1, stop, cs, radar_state(present=True, distance=80.0))

  outputs = run_frames(cem, 20, stop, cs, radar_state())

  assert not any(outputs)
  assert cem._lead_veto_remaining > 0.0


def test_strict_clear_frame_starts_revocable_lead_release_through_strict_flicker():
  cem = new_cem()
  v_ego = 18.95
  cs = car_state(v_ego=v_ego)
  strict = with_model_leads(
    model(path_end=83.6443, terminal_speed=0.6702, desired_accel=-0.7),
    model_lead(prob=0.1, x=97.11, y=0.0),
    model_lead(prob=0.1, x=96.62, y=0.0),
    model_lead(prob=0.1, x=96.27, y=0.0),
  )
  early = with_model_leads(
    route_d7_early_stop_model(),
    model_lead(prob=0.0902, x=92.94, y=2.11),
    model_lead(prob=0.0244, x=92.82, y=2.02),
    model_lead(prob=0.0307, x=92.68, y=1.99),
  )
  run_frames(cem, 1, strict, cs, radar_state(present=True, distance=80.0))

  assert not cem.update(strict, cs, radar_state(), controls_enabled=True, model_updated=True, model_valid=True)
  outputs = run_frames(cem, 20, early, cs, radar_state())

  assert outputs[-1]
  assert cem._lead_veto_remaining > 0.0



def test_recent_lead_hysteresis_releases_early_for_distant_model_lead_beyond_stop():
  cem = new_cem()
  v_ego = 15.0
  cs = car_state(v_ego=v_ego)
  strict = with_model_leads(
    model(path_end=69.6756, terminal_speed=0.7223, desired_accel=-0.847),
    model_lead(prob=0.4991, x=92.07, y=0.13),
    model_lead(prob=0.4781, x=91.91, y=0.15),
    model_lead(prob=0.4080, x=91.77, y=0.16),
  )
  stop = with_model_leads(
    model(path_end=71.828, terminal_speed=1.137, desired_accel=-0.847),
    model_lead(prob=0.4991, x=94.96, y=0.13),
    model_lead(prob=0.4781, x=94.85, y=0.15),
    model_lead(prob=0.4080, x=94.75, y=0.16),
  )
  run_frames(cem, 1, strict, cs, radar_state(present=True, distance=60.0))

  assert not cem.update(strict, cs, radar_state(), controls_enabled=True, model_updated=True, model_valid=True)
  outputs = run_frames(cem, 20, stop, cs, radar_state())

  assert outputs[-1]
  assert cem._lead_veto_remaining > 0.0


def test_in_path_replacement_model_lead_preserves_recent_lead_hysteresis():
  cem = new_cem()
  v_ego = 15.0
  cs = car_state(v_ego=v_ego)
  stop = with_model_leads(
    model(path_end=75.0, terminal_speed=1.0, desired_accel=-0.8),
    model_lead(prob=0.2215, x=36.93, y=0.98),
    model_lead(prob=0.2030, x=36.91, y=0.87),
    model_lead(prob=0.3812, x=36.82, y=0.87),
  )
  run_frames(cem, 1, stop, cs, radar_state(present=True, distance=45.0))

  outputs = run_frames(cem, 20, stop, cs, radar_state())

  assert not any(outputs)
  assert cem._lead_veto_remaining > 0.0


def test_model_lead_corridor_clamps_to_authored_path_endpoint():
  from openpilot.selfdrive.controls.lib.conditional_experimental_mode import model_lead_stop_path_clear

  stop = with_model_leads(
    model(path_end=75.0, terminal_speed=1.0, desired_accel=-0.8),
    model_lead(prob=0.2, x=85.0, y=15.0),
    model_lead(), model_lead(),
  )
  stop.position.y = [15.0 * i / (ModelConstants.IDX_N - 1) for i in range(ModelConstants.IDX_N)]

  assert not model_lead_stop_path_clear(stop, 75.0)


def test_model_lead_path_boundaries_are_inclusive_and_every_positive_probability_counts():
  from openpilot.selfdrive.controls.lib.conditional_experimental_mode import model_lead_stop_path_clear

  template = with_model_leads(
    model(path_end=75.0, terminal_speed=0.5, desired_accel=-0.8),
    model_lead(), model_lead(), model_lead(),
  )

  for x, y, probability, expected_clear in (
    (85.0, 0.0, 1e-9, False),
    (85.0 + 1e-6, 0.0, 1e-9, True),
    (36.9, 1.5, 1e-9, False),
    (36.9, 1.5 + 1e-6, 1e-9, True),
    (36.9, 0.0, 0.0, True),
    (36.9, 0.0, 1e-9, False),
  ):
    stop = copy.deepcopy(template)
    stop.leadsV3[0] = model_lead(prob=probability, x=x, y=y)
    assert model_lead_stop_path_clear(stop, 75.0) is expected_clear


def test_model_lead_future_cut_in_blocks_early_release():
  from openpilot.selfdrive.controls.lib.conditional_experimental_mode import model_lead_stop_path_clear

  stop = with_model_leads(
    model(path_end=75.0, terminal_speed=0.5, desired_accel=-0.8),
    model_lead(prob=0.2, x=36.9, y=1.5 + 1e-6),
    model_lead(), model_lead(),
  )
  stop.leadsV3[0].y[1] = 0.0

  assert not model_lead_stop_path_clear(stop, 75.0)


def test_model_lead_duplicate_path_points_fail_closed():
  from openpilot.selfdrive.controls.lib.conditional_experimental_mode import model_lead_stop_path_clear

  stop = with_model_leads(
    model(path_end=75.0, terminal_speed=0.5, desired_accel=-0.8),
    model_lead(), model_lead(), model_lead(),
  )
  stop.position.x[11] = stop.position.x[10]
  stop.position.y[10] = 10.0
  stop.leadsV3[0] = model_lead(prob=0.2, x=stop.position.x[10], y=0.0)

  assert not model_lead_stop_path_clear(stop, 75.0)


def test_active_lead_release_revokes_on_health_loss_or_replacement():
  cs = car_state(v_ego=15.0)
  strict = with_model_leads(
    model(path_end=75.0, terminal_speed=0.5, desired_accel=-0.8),
    model_lead(prob=0.2, x=95.0, y=0.0),
    model_lead(prob=0.2, x=95.0, y=0.0),
    model_lead(prob=0.2, x=95.0, y=0.0),
  )
  replacement = with_model_leads(
    model(path_end=75.0, terminal_speed=0.5, desired_accel=-0.8),
    model_lead(prob=1e-9, x=36.9, y=0.0),
    model_lead(), model_lead(),
  )

  for fault in ("model", "radar", "replacement"):
    cem = new_cem()
    run_frames(cem, 1, strict, cs, radar_state(present=True, distance=45.0))
    assert not cem.update(strict, cs, radar_state(), controls_enabled=True, model_updated=True, model_valid=True)
    assert cem._lead_release_active

    if fault == "model":
      cem.update(strict, cs, radar_state(), controls_enabled=True, model_updated=False, model_valid=False)
    elif fault == "radar":
      cem.update(strict, cs, radar_state(), controls_enabled=True, model_updated=False, model_valid=True, radar_valid=False)
    else:
      cem.update(replacement, cs, radar_state(), controls_enabled=True, model_updated=True, model_valid=True)

    assert not cem._lead_release_active
    assert not cem.stop_qualified
    assert cem._entry_elapsed == 0.0


def test_malformed_model_lead_path_falls_back_to_recent_lead_hysteresis():
  v_ego = 15.0
  cs = car_state(v_ego=v_ego)
  template = with_model_leads(
    model(path_end=75.0, terminal_speed=1.0, desired_accel=-0.8),
    model_lead(prob=0.2, x=95.0, y=0.0),
    model_lead(prob=0.2, x=95.0, y=0.0),
    model_lead(prob=0.2, x=95.0, y=0.0),
  )

  def assert_blocked(stop):
    cem = new_cem()
    run_frames(cem, 1, stop, cs, radar_state(present=True, distance=45.0))
    assert not any(run_frames(cem, 20, stop, cs, radar_state()))
    assert cem._lead_veto_remaining > 0.0

  for bad_value in (float("nan"), float("inf"), float("-inf")):
    for sample in range(ModelConstants.IDX_N):
      stop = copy.deepcopy(template)
      stop.position.y[sample] = bad_value
      assert_blocked(stop)
    for lead_index in range(3):
      for axis in ("x", "y"):
        for sample in range(len(ModelConstants.LEAD_T_IDXS)):
          stop = copy.deepcopy(template)
          getattr(stop.leadsV3[lead_index], axis)[sample] = bad_value
          assert_blocked(stop)

  for probability in (float("nan"), float("inf"), float("-inf"), -0.1, 1.1):
    stop = copy.deepcopy(template)
    stop.leadsV3[0].prob = probability
    assert_blocked(stop)

  for mutation in (
    lambda stop: setattr(stop.position, "y", stop.position.y[:-1]),
    lambda stop: setattr(stop, "leadsV3", stop.leadsV3[:-1]),
    lambda stop: setattr(stop.leadsV3[0], "x", stop.leadsV3[0].x[:-1]),
    lambda stop: setattr(stop.leadsV3[0], "y", stop.leadsV3[0].y[:-1]),
    lambda stop: stop.position.x.__setitem__(10, stop.position.x[9] - 1.0),
    lambda stop: stop.leadsV3[0].x.__setitem__(0, 0.0),
  ):
    stop = copy.deepcopy(template)
    mutation(stop)
    assert_blocked(stop)


def test_direct_should_stop_cannot_bypass_recent_lead_hysteresis():
  cem = new_cem()
  cs = car_state(v_ego=15.0)
  direct = with_model_leads(
    model(should_stop=True, path_end=75.0, terminal_speed=15.0, desired_accel=-0.1),
    model_lead(), model_lead(), model_lead(),
  )
  run_frames(cem, 1, direct, cs, radar_state(present=True, distance=45.0))

  assert not any(run_frames(cem, 20, direct, cs, radar_state()))
  assert cem._lead_veto_remaining > 0.0


def test_raw_lead_reappearance_cancels_early_release_qualification_same_tick():
  for raw_lead in (
    radar_state(present=True, distance=45.0),
    radar_state(secondary_present=True, secondary_distance=45.0),
  ):
    cem = new_cem()
    cs = car_state(v_ego=15.0)
    stop = with_model_leads(
      model(path_end=75.0, terminal_speed=0.5, desired_accel=-0.8),
      model_lead(prob=0.2, x=95.0, y=0.0),
      model_lead(prob=0.2, x=95.0, y=0.0),
      model_lead(prob=0.2, x=95.0, y=0.0),
    )
    run_frames(cem, 1, stop, cs, radar_state(present=True, distance=45.0))
    assert not any(run_frames(cem, 3, stop, cs, radar_state()))
    assert cem.intent_filter.x > 0.0
    assert cem._lead_release_active

    assert not cem.update(stop, cs, raw_lead, controls_enabled=True, model_updated=True, model_valid=True)
    assert not cem._lead_release_active
    assert not cem.stop_qualified
    assert cem._entry_elapsed == 0.0


def test_raw_lead_on_control_tick_clears_partial_admission_state():
  for raw_lead in (
    radar_state(present=True, distance=45.0),
    radar_state(secondary_present=True, secondary_distance=45.0),
  ):
    cem = new_cem()
    cs = car_state(v_ego=15.0)
    stop = confirmed_stop_model(cs.vEgo)
    for _ in range(20):
      assert not cem.update(stop, cs, radar_state(), controls_enabled=True, model_updated=True, model_valid=True)
      if cem._entry_elapsed > 0.0:
        break
    assert cem._entry_elapsed > 0.0
    assert cem.intent_filter.x > 0.0

    assert not cem.update(stop, cs, raw_lead, controls_enabled=True, model_updated=False, model_valid=True)
    assert cem._entry_elapsed == 0.0
    assert cem.intent_filter.x == 0.0
    assert cem._intent_hold_remaining == 0.0


def test_raw_lead_control_ticks_do_not_starve_active_mode_release():
  control_dt = 0.01
  cem = ConditionalExperimentalMode(control_dt=control_dt, model_dt=DT)
  activate(cem)

  outputs = []
  for frame in range(int((STOP_INTENT_HOLD_S + 2.0) / control_dt)):
    outputs.append(cem.update(
      clear_model(), car_state(), radar_state(present=True, distance=1000.0),
      controls_enabled=True, model_updated=frame % int(DT / control_dt) == 0, model_valid=True,
    ))

  assert not outputs[-1]


def test_should_stop_is_direct_evidence_when_trajectory_is_missing():
  cem = new_cem()
  direct_stop = model(should_stop=True, path_end=None, terminal_speed=None, desired_accel=-0.1)
  assert run_frames(cem, 30, direct_stop)[-1]
  assert cem.last_observation.reason == "shouldStop"


def test_braking_path_is_a_missing_velocity_fallback():
  cem = new_cem()
  fallback = model(path_end=30.0, terminal_speed=None, desired_accel=-0.6)
  assert run_frames(cem, 30, fallback)[-1]
  assert cem.last_observation.reason == "path+braking"


def test_brief_stop_prediction_does_not_pass_filter_and_debounce():
  cem = new_cem()
  assert not any(run_frames(cem, 3, confirmed_stop_model()))
  assert not any(run_frames(cem, 30, clear_model()))


def test_force_stop_qualification_requires_sustained_full_trajectory_inside_commit_range():
  cem = new_cem()
  early_cs = car_state(v_ego=18.0)
  early = model(path_end=95.0, terminal_speed=1.0, desired_accel=-0.8)
  assert observe_model_stop_intent(early, early_cs, radar_state()).confidence == STOP_EARLY_CONFIDENCE
  for _ in range(int(FORCE_STOP_QUALIFY_S / DT)):
    cem.update(early, early_cs, radar_state(), controls_enabled=True, model_updated=True, model_valid=True)
    assert not cem.stop_qualified

  cs = car_state(v_ego=25.0)
  stop = model(path_end=FORCE_STOP_COMMIT_DISTANCE_M - 5.0, terminal_speed=1.0, desired_accel=-0.8)
  for _ in range(int(FORCE_STOP_QUALIFY_S / DT)):
    cem.update(stop, cs, radar_state(), controls_enabled=True, model_updated=True, model_valid=True)
    assert not cem.stop_qualified
    stop.position.x[-1] -= cs.vEgo * DT
  cem.update(stop, cs, radar_state(), controls_enabled=True, model_updated=True, model_valid=True)
  assert cem.stop_qualified

  tracked_distance = cem.stop_distance
  stop.position.x[-1] = tracked_distance - cs.vEgo * DT + FORCE_STOP_WORLD_TOLERANCE_M - 0.1
  cem.update(stop, cs, radar_state(), controls_enabled=True, model_updated=True, model_valid=True)
  assert cem.stop_qualified
  assert cem.stop_distance == stop.position.x[-1]

  stop.position.x[-1] = FORCE_STOP_COMMIT_DISTANCE_M + 20.0
  cem.update(stop, cs, radar_state(), controls_enabled=True, model_updated=True, model_valid=True)
  assert not cem.stop_qualified


def test_force_stop_qualification_rejects_an_endpoint_moving_in_world_coordinates():
  cem = new_cem()
  cs = car_state(v_ego=25.0)
  moving_world_endpoint = model(path_end=95.0, terminal_speed=1.0, desired_accel=-0.8)

  for _ in range(2 * int(FORCE_STOP_QUALIFY_S / DT)):
    cem.update(moving_world_endpoint, cs, radar_state(), controls_enabled=True, model_updated=True, model_valid=True)
    assert not cem.stop_qualified


def test_force_stop_qualification_rejects_direct_fallback_and_nonfinite_ego_evidence():
  missing_velocity = model(should_stop=True, path_end=95.0, terminal_speed=25.0)
  missing_velocity.velocity.x = []
  nonfinite_velocity = model(should_stop=True, path_end=95.0, terminal_speed=0.5)
  nonfinite_velocity.velocity.x[10] = float("nan")
  probes = (
    (model(should_stop=True, path_end=95.0, terminal_speed=25.0), car_state(v_ego=25.0)),
    (missing_velocity, car_state(v_ego=25.0)),
    (nonfinite_velocity, car_state(v_ego=25.0)),
    (model(path_end=95.0, terminal_speed=0.5, desired_accel=float("nan")), car_state(v_ego=25.0)),
    (model(path_end=95.0, terminal_speed=0.5), car_state(v_ego=float("nan"))),
  )
  for stop, cs in probes:
    cem = new_cem()
    for _ in range(2 * int(FORCE_STOP_QUALIFY_S / DT) + 1):
      cem.update(stop, cs, radar_state(), controls_enabled=True, model_updated=True, model_valid=True)
      assert not cem.stop_qualified
      if math.isfinite(cs.vEgo):
        stop.position.x[-1] -= cs.vEgo * DT


def test_nonfinite_trajectory_resets_every_cem_lifecycle_state():
  for lifecycle in ("cold", "partial", "qualified", "active"):
    for axis in ("position", "velocity"):
      for bad_value in (float("nan"), float("inf"), float("-inf")):
        for sample in range(ModelConstants.IDX_N):
          cem = new_cem()
          cs = car_state()
          if lifecycle == "partial":
            cem.update(confirmed_stop_model(), cs, radar_state(), controls_enabled=True,
                       model_updated=True, model_valid=True)
            assert cem.intent_filter.x > 0.0
          elif lifecycle == "qualified":
            cem._post_stop_remaining = 10.0
            stop = confirmed_stop_model()
            for _ in range(int(FORCE_STOP_QUALIFY_S / DT) + 1):
              cem.update(stop, cs, radar_state(), controls_enabled=True, model_updated=True, model_valid=True)
              stop.position.x[-1] -= cs.vEgo * DT
            assert cem.stop_qualified and not cem.experimental_mode
            cem._post_stop_remaining = 0.0
          elif lifecycle == "active":
            activate(cem, cs)

          malformed = confirmed_stop_model()
          getattr(malformed, axis).x[sample] = bad_value
          cem.update(malformed, cs, radar_state(), controls_enabled=True, model_updated=True, model_valid=True)
          assert cem.intent_filter.x == 0.0
          assert cem._intent_hold_remaining == 0.0
          assert cem._entry_elapsed == 0.0
          assert not cem.stop_qualified

          if lifecycle == "active":
            assert not run_frames(cem, int(5.0 / DT), malformed, cs)[-1]

          finite = confirmed_stop_model()
          fresh = new_cem()
          assert cem.update(finite, cs, radar_state(), controls_enabled=True,
                            model_updated=True, model_valid=True) == fresh.update(
                              finite, cs, radar_state(), controls_enabled=True,
                              model_updated=True, model_valid=True)
          assert cem.intent_filter.x == fresh.intent_filter.x
          assert cem._intent_hold_remaining == fresh._intent_hold_remaining
          assert cem._entry_elapsed == fresh._entry_elapsed
          assert not cem.stop_qualified


def test_secondary_lead_blocks_force_stop_qualification_through_release_hysteresis():
  cem = new_cem()
  cs = car_state(v_ego=25.0)
  stop = model(path_end=95.0, terminal_speed=0.5)
  lead = radar_state(secondary_present=True, secondary_distance=50.0)

  for _ in range(2 * int(FORCE_STOP_QUALIFY_S / DT) + 1):
    stop.position.x[-1] -= cs.vEgo * DT
    cem.update(stop, cs, lead, controls_enabled=True, model_updated=True, model_valid=True)
    assert not cem.stop_qualified

  for _ in range(int(LEAD_RELEASE_HYSTERESIS_S / DT)):
    stop.position.x[-1] -= cs.vEgo * DT
    cem.update(stop, cs, radar_state(), controls_enabled=True, model_updated=True, model_valid=True)
    assert not cem.stop_qualified


def test_far_secondary_lead_cannot_precharge_force_stop_qualification():
  cem = new_cem()
  cs = car_state(v_ego=25.0)
  stop = model(path_end=95.0, terminal_speed=0.5)
  far_lead = radar_state(secondary_present=True, secondary_distance=1000.0)

  for _ in range(2 * int(FORCE_STOP_QUALIFY_S / DT) + 1):
    stop.position.x[-1] -= cs.vEgo * DT
    cem.update(stop, cs, far_lead, controls_enabled=True, model_updated=True, model_valid=True)
    assert not cem.stop_qualified

  for _ in range(int(FORCE_STOP_QUALIFY_S / DT)):
    stop.position.x[-1] -= cs.vEgo * DT
    cem.update(stop, cs, radar_state(), controls_enabled=True, model_updated=True, model_valid=True)
    assert not cem.stop_qualified


def test_raw_lead_on_control_tick_resets_qualified_force_stop():
  cem = new_cem()
  cs = car_state(v_ego=25.0)
  stop = model(path_end=95.0, terminal_speed=0.5)

  for _ in range(int(FORCE_STOP_QUALIFY_S / DT)):
    cem.update(stop, cs, radar_state(), controls_enabled=True, model_updated=True, model_valid=True)
    assert not cem.stop_qualified
    stop.position.x[-1] -= cs.vEgo * DT
  cem.update(stop, cs, radar_state(), controls_enabled=True, model_updated=True, model_valid=True)
  assert cem.stop_qualified

  far_lead = radar_state(secondary_present=True, secondary_distance=1000.0)
  cem.update(stop, cs, far_lead, controls_enabled=True, model_updated=False, model_valid=True)
  assert not cem.stop_qualified

  for _ in range(int(FORCE_STOP_QUALIFY_S / DT)):
    stop.position.x[-1] -= cs.vEgo * DT
    cem.update(stop, cs, radar_state(), controls_enabled=True, model_updated=True, model_valid=True)
    assert not cem.stop_qualified
  stop.position.x[-1] -= cs.vEgo * DT
  cem.update(stop, cs, radar_state(), controls_enabled=True, model_updated=True, model_valid=True)
  assert cem.stop_qualified


def test_repeated_control_ticks_do_not_count_one_model_frame_more_than_once():
  cem = new_cem()
  stop = confirmed_stop_model()
  cs = car_state()
  radar = radar_state()

  for _ in range(100):
    assert not cem.update(stop, cs, radar, controls_enabled=True, model_updated=False, model_valid=True)

  assert cem.intent_filter.x == 0.0


def test_invalid_model_or_radar_tick_resets_partial_force_stop_qualification():
  cs = car_state(v_ego=25.0)

  for invalid_kwargs in (
    {'model_valid': False, 'radar_valid': True},
    {'model_valid': True, 'radar_valid': False},
  ):
    cem = new_cem()
    stop = model(path_end=95.0, terminal_speed=0.5)
    for _ in range(int(FORCE_STOP_QUALIFY_S / DT)):
      cem.update(stop, cs, radar_state(), controls_enabled=True, model_updated=True,
                 model_valid=True, radar_valid=True)
      assert not cem.stop_qualified
      stop.position.x[-1] -= cs.vEgo * DT

    cem.update(stop, cs, radar_state(), controls_enabled=True, model_updated=False, **invalid_kwargs)
    assert not cem.stop_qualified

    cem.update(stop, cs, radar_state(), controls_enabled=True, model_updated=True,
               model_valid=True, radar_valid=True)
    assert not cem.stop_qualified


def test_release_hysteresis_ignores_brief_clear_prediction():
  cem = new_cem()
  activate(cem)

  assert all(run_frames(cem, 4, clear_model()))
  assert run_frames(cem, 5, confirmed_stop_model())[-1]


def test_confirmed_stop_holds_experimental_through_prediction_flicker():
  control_dt = 0.01
  cem = ConditionalExperimentalMode(control_dt=control_dt, model_dt=DT)
  stop = confirmed_stop_model()
  clear = clear_model()
  cs = car_state()
  radar = radar_state()
  active = False

  for frame in range(200):
    active = cem.update(stop, cs, radar, controls_enabled=True, model_updated=frame % 5 == 0, model_valid=True)
  assert active

  for frame in range(200):
    active = cem.update(clear, cs, radar, controls_enabled=True, model_updated=frame % 5 == 0, model_valid=True)
  assert active

  assert cem.update(stop, cs, radar, controls_enabled=True, model_updated=True, model_valid=True)
  hold_frames = int((STOP_INTENT_HOLD_S - 0.5) / control_dt)
  assert all(cem.update(clear, cs, radar, controls_enabled=True, model_updated=frame % 5 == 0, model_valid=True)
             for frame in range(hold_frames))
  for frame in range(int(1.0 / control_dt)):
    active = cem.update(clear, cs, radar, controls_enabled=True, model_updated=frame % 5 == 0, model_valid=True)
  assert not active


def test_filter_only_hint_cannot_sustain_an_active_mode_latch():
  cem = new_cem()
  v_ego = 24.54
  activate(cem, car_state(v_ego=v_ego))
  hint = model(path_end=170.13, terminal_speed=8.41, desired_accel=-1.06,
               desired_curvature=0.00015, terminal_heading=0.004)

  outputs = run_frames(cem, 100, hint, car_state(v_ego=v_ego))
  assert outputs[0]
  assert not outputs[-1]


def test_experimental_holds_at_standstill_while_stop_remains_valid():
  cem = new_cem()
  activate(cem)
  stopped = car_state(v_ego=0.0, standstill=True)
  stop = model(should_stop=True, path_end=2.0, terminal_speed=0.0)

  assert all(run_frames(cem, 80, stop, stopped))


def test_green_release_returns_to_chill_after_standstill_latch_and_hysteresis():
  cem = new_cem()
  activate(cem)
  stopped = car_state(v_ego=0.0, standstill=True)
  stop = model(should_stop=True, path_end=2.0, terminal_speed=0.0)
  run_frames(cem, 25, stop, stopped)

  outputs = run_frames(cem, 100, clear_model(), stopped)
  assert outputs[0]
  assert not outputs[-1]


def test_sustained_open_model_releases_stop_hold_before_mode_hysteresis():
  cem = new_cem()
  activate(cem)
  stopped = car_state(v_ego=0.0, standstill=True)
  run_frames(cem, 25, model(should_stop=True, path_end=4.0, terminal_speed=0.0), stopped)

  green = model(path_end=34.0, terminal_speed=12.0, desired_accel=0.3)
  assert all(run_frames(cem, 4, green, stopped))
  assert cem.stop_latched
  assert run_frames(cem, 5, green, stopped)[-1]
  assert cem.experimental_mode
  assert not cem.stop_latched

  run_frames(cem, 8, model(should_stop=True, path_end=4.0, terminal_speed=0.0), stopped)
  assert cem.stop_latched


def test_stop_hold_fails_closed_on_weak_or_malformed_open_predictions():
  malformed = model(path_end=34.0, terminal_speed=12.0)
  malformed.position.x[5] = math.nan
  nonfinite_action = model(path_end=34.0, terminal_speed=12.0)
  nonfinite_action.action.desiredAcceleration = math.nan
  candidates = (
    model(path_end=34.0, terminal_speed=2.0),
    model(should_stop=True, path_end=34.0, terminal_speed=12.0),
    model(path_end=15.0, terminal_speed=12.0),
    model(path_end=34.0, terminal_speed=12.0, desired_accel=-1.0),
  )
  for candidate in candidates:
    cem = new_cem()
    activate(cem)
    assert all(run_frames(cem, 20, candidate))
    assert cem.stop_latched
  assert not model_stop_release_open(malformed)
  assert not model_stop_release_open(nonfinite_action)
  assert model_stop_release_open(model(path_end=34.0, terminal_speed=12.0, desired_accel=-1.0), require_nonbraking=False)


def test_confirmed_open_hold_recloses_on_health_lead_or_turn_veto():
  green = model(path_end=34.0, terminal_speed=12.0, desired_accel=0.3)
  for fault in ("model", "radar", "lead", "turn"):
    cem = new_cem()
    activate(cem)
    run_frames(cem, 9, green)
    assert not cem.stop_latched

    kwargs = {"controls_enabled": True, "model_updated": True, "model_valid": fault != "model",
              "radar_valid": fault != "radar"}
    cs = car_state(v_ego=5.0, left_blinker=fault == "turn", steering_angle=40.0 if fault == "turn" else 0.0)
    radar = radar_state(present=fault == "lead", distance=20.0)
    assert cem.update(green, cs, radar, **kwargs)
    assert cem.stop_latched


def test_generic_stop_latch_does_not_release_on_a_brief_standstill_clear():
  cem = new_cem()
  activate(cem)
  stopped = car_state(v_ego=0.0, standstill=True)
  stop = model(should_stop=True, path_end=2.0, terminal_speed=0.0)
  run_frames(cem, 4, stop, stopped)

  # There is no semantic stop-sign classifier in BLoTv2. Every confirmed stop
  # gets this minimum standstill latch, which covers a stop-sign model flicker.
  assert all(run_frames(cem, 12, clear_model(), stopped))


def test_resume_releases_latched_mode_and_starts_post_stop_suppression():
  cem = new_cem()
  activate(cem)
  run_frames(cem, 2, confirmed_stop_model(0.0), car_state(v_ego=0.0, standstill=True))

  resumed = car_state(v_ego=1.0, standstill=False)
  assert not cem.update(confirmed_stop_model(1.0), resumed, radar_state(),
                        controls_enabled=True, model_updated=True, model_valid=True)
  assert not run_frames(cem, 10, confirmed_stop_model(1.0), resumed)[-1]


def test_relevant_lead_blocks_entry_but_does_not_flicker_an_existing_stop():
  stop = confirmed_stop_model()
  close_lead = radar_state(present=True, distance=20.0)

  blocked = new_cem()
  assert not any(run_frames(blocked, 40, stop, radar=close_lead))
  assert blocked.last_observation.relevant_lead

  latched = new_cem()
  activate(latched)
  assert all(run_frames(latched, 20, stop, radar=close_lead))


def test_far_lead_does_not_hide_a_model_stop():
  cem = new_cem()
  assert run_frames(cem, 30, confirmed_stop_model(), radar=radar_state(present=True, distance=100.0))[-1]


def test_curve_speed_plan_and_turn_signal_do_not_false_trigger():
  curve = model(path_end=30.0, terminal_speed=8.0, desired_accel=0.0, desired_curvature=0.05)
  assert not any(run_frames(new_cem(), 40, curve))

  committed_turn = car_state(v_ego=5.0, left_blinker=True, steering_angle=40.0)
  cem = new_cem()
  assert not any(run_frames(cem, 40, confirmed_stop_model(5.0), committed_turn))
  assert cem.last_observation.committed_turn


def test_straight_blinker_approach_can_still_detect_a_real_stop():
  indicated = car_state(v_ego=8.0, left_blinker=True, steering_angle=0.0)
  assert run_frames(new_cem(), 30, confirmed_stop_model(8.0), indicated)[-1]


def test_accelerator_override_returns_to_chill_and_suppresses_reentry():
  cem = new_cem()
  activate(cem)

  gas = car_state(gas=True)
  assert not cem.update(confirmed_stop_model(), gas, radar_state(),
                        controls_enabled=True, model_updated=True, model_valid=True)
  assert cem.driver_override_active

  suppress_frames = int(DRIVER_OVERRIDE_SUPPRESS_S / DT) - 1
  assert not any(run_frames(cem, suppress_frames, confirmed_stop_model()))


def test_brake_override_returns_to_chill():
  cem = new_cem()
  activate(cem)
  brake = car_state(brake=True)

  assert not cem.update(confirmed_stop_model(), brake, radar_state(),
                        controls_enabled=True, model_updated=True, model_valid=True)
  assert cem.driver_override_active


def test_missing_model_signals_never_trigger_and_invalid_model_releases_active_mode():
  missing = SimpleNamespace()
  cem = new_cem()
  assert not any(run_frames(cem, 40, missing))
  assert observe_model_stop_intent(missing, car_state(), radar_state()).confidence == 0.0

  activate(cem)
  for _ in range(20):
    output = cem.update(missing, car_state(), radar_state(),
                        controls_enabled=True, model_updated=False, model_valid=False)
  assert not output


def test_disable_and_restart_reset_all_conditional_state_to_chill():
  cem = new_cem()
  activate(cem)
  assert not cem.update(confirmed_stop_model(), car_state(), radar_state(),
                        controls_enabled=False, model_updated=True, model_valid=True)
  assert not cem.stop_latched
  assert cem.intent_filter.x == 0.0

  restarted = new_cem()
  assert not restarted.experimental_mode
  assert not restarted.driver_override_active

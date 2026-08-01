from types import SimpleNamespace

from openpilot.selfdrive.controls.lib.conditional_experimental_mode import (
  ConditionalExperimentalMode,
  DRIVER_OVERRIDE_SUPPRESS_S,
  LEAD_RELEASE_HYSTERESIS_S,
  STOP_EARLY_COMFORT_DECEL,
  STOP_EARLY_CONFIDENCE,
  STOP_EARLY_HINT_CONFIDENCE,
  STOP_EARLY_MAX_HEADING_CHANGE,
  STOP_EARLY_MAX_LATERAL_ACCEL,
  STOP_EARLY_MIN_SPEED,
  STOP_EARLY_RESPONSE_BUFFER_S,
  STOP_ENTRY_DEBOUNCE_S,
  STOP_PREDICTION_HORIZON_S,
  STOP_SAMPLE_MIN_CONFIDENCE,
  observe_model_stop_intent,
)


DT = 0.05


def model(*, should_stop=False, path_end=90.0, terminal_speed=10.0, desired_accel=0.0,
          desired_curvature=0.0, terminal_heading=0.0):
  position_x = [] if path_end is None else [0.0, path_end]
  velocity_x = [] if terminal_speed is None else [10.0, terminal_speed]
  orientation_z = [] if terminal_heading is None else [0.0, terminal_heading]
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


def radar_state(*, present=False, distance=1000.0):
  return SimpleNamespace(leadOne=SimpleNamespace(present=present, dRel=distance))


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


def frames_until_active(cem, md, cs, limit=100):
  for frame in range(1, limit + 1):
    if cem.update(md, cs, radar_state(), controls_enabled=True, model_updated=True, model_valid=True):
      return frame
  return None


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


def test_high_speed_hint_precharges_filter_but_cannot_request_experimental():
  cem = new_cem()
  v_ego = 20.0
  hint = model(path_end=159.3, terminal_speed=9.93, desired_accel=-0.32,
               desired_curvature=-0.00005, terminal_heading=-0.001)
  observation = observe_model_stop_intent(hint, car_state(v_ego=v_ego), radar_state())

  assert observation.reason == "earlyHint"
  assert observation.confidence == STOP_EARLY_HINT_CONFIDENCE
  assert observation.confidence < STOP_SAMPLE_MIN_CONFIDENCE
  assert not any(run_frames(cem, 200, hint, car_state(v_ego=v_ego)))
  assert cem._entry_elapsed == 0.0


def test_high_speed_hint_reduces_filter_latency_without_bypassing_debounce():
  v_ego = 20.0
  cs = car_state(v_ego=v_ego)
  hint = model(path_end=159.3, terminal_speed=9.93, desired_accel=-0.32,
               desired_curvature=-0.00005, terminal_heading=-0.001)
  strong = model(path_end=145.0, terminal_speed=6.0, desired_accel=-0.55,
                 desired_curvature=-0.00005, terminal_heading=-0.001)

  cold_frames = frames_until_active(new_cem(), strong, cs)
  primed = new_cem()
  assert not any(run_frames(primed, 30, hint, cs))
  primed_frames = frames_until_active(primed, strong, cs)

  assert cold_frames is not None and primed_frames is not None
  assert primed_frames < cold_frames
  assert primed_frames * DT >= STOP_ENTRY_DEBOUNCE_S


def test_early_high_speed_tier_stays_inside_comfort_stopping_envelope():
  v_ego = 18.0
  distance_limit = v_ego ** 2 / (2.0 * STOP_EARLY_COMFORT_DECEL) + v_ego * STOP_EARLY_RESPONSE_BUFFER_S
  outside = model(path_end=distance_limit + 0.1, terminal_speed=4.0, desired_accel=-0.6)
  observation = observe_model_stop_intent(outside, car_state(v_ego=v_ego), radar_state())

  assert observation.reason != "earlyTrajectory"
  assert not any(run_frames(new_cem(), 40, outside, car_state(v_ego=v_ego)))


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


def test_repeated_control_ticks_do_not_count_one_model_frame_more_than_once():
  cem = new_cem()
  stop = confirmed_stop_model()
  cs = car_state()
  radar = radar_state()

  for _ in range(100):
    assert not cem.update(stop, cs, radar, controls_enabled=True, model_updated=False, model_valid=True)

  assert cem.intent_filter.x == 0.0


def test_release_hysteresis_ignores_brief_clear_prediction():
  cem = new_cem()
  activate(cem)

  assert all(run_frames(cem, 4, clear_model()))
  assert run_frames(cem, 5, confirmed_stop_model())[-1]


def test_filter_only_hint_cannot_sustain_an_active_mode_latch():
  cem = new_cem()
  activate(cem, car_state(v_ego=20.0))
  hint = model(path_end=159.3, terminal_speed=9.93, desired_accel=-0.32,
               desired_curvature=-0.00005, terminal_heading=-0.001)

  outputs = run_frames(cem, 80, hint, car_state(v_ego=20.0))
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

  outputs = run_frames(cem, 60, clear_model(), stopped)
  assert outputs[0]
  assert not outputs[-1]


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

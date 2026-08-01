from types import SimpleNamespace

from openpilot.selfdrive.controls.lib.conditional_experimental_mode import (
  ConditionalExperimentalMode,
  DRIVER_OVERRIDE_SUPPRESS_S,
  STOP_PREDICTION_HORIZON_S,
  observe_model_stop_intent,
)


DT = 0.05


def model(*, should_stop=False, path_end=90.0, terminal_speed=10.0, desired_accel=0.0, desired_curvature=0.0):
  position_x = [] if path_end is None else [0.0, path_end]
  velocity_x = [] if terminal_speed is None else [10.0, terminal_speed]
  return SimpleNamespace(
    action=SimpleNamespace(
      shouldStop=should_stop,
      desiredAcceleration=desired_accel,
      desiredCurvature=desired_curvature,
    ),
    position=SimpleNamespace(x=position_x),
    velocity=SimpleNamespace(x=velocity_x),
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

from openpilot.selfdrive.controls.lib.smooth_stops import HOLD_RELEASE_FRAMES, STOP_KISS_DECEL, SmoothStopController


def test_equal_speed_lead_does_not_add_braking():
  controller = SmoothStopController()

  output = controller.settle(
    a_target=0.0,
    v_ego=1.0,
    lead_distance=3.0,
    has_lead=True,
    last_output=-STOP_KISS_DECEL,
    lead_speed=1.0,
  )

  assert abs(output + STOP_KISS_DECEL) < 1e-9


def test_moving_queue_releases_anti_creep_through_brief_radar_dropout():
  controller = SmoothStopController()
  output = -STOP_KISS_DECEL

  for _ in range(100):
    output = controller.settle(0.0, 0.5, 10.0, False, output)
  stalled_output = output

  output = controller.settle(0.0, 0.5, 10.0, True, output, lead_speed=0.31)
  for _ in range(20):
    output = controller.settle(0.0, 0.5, 10.0, False, output)

  assert output > stalled_output + 0.15


def test_stopped_lead_does_not_lengthen_the_hold_release():
  # the plan already corroborates a lead's departure before dropping its stop bit; the car adds ~1.3 s of its own
  controller = SmoothStopController()
  controller.arm_hold()
  for _ in range(HOLD_RELEASE_FRAMES - 1):
    assert not controller.hold_release(False, has_lead=True, lead_speed=0.0)
  assert controller.hold_release(False, has_lead=True, lead_speed=0.0)
  controller.arm_hold()
  assert controller.hold_release(False, has_lead=True, lead_speed=0.5)  # measured departure: immediate

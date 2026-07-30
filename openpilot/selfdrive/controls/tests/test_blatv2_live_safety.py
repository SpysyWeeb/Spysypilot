import math

from openpilot.selfdrive.controls.lib.blatv2.actuator import RuntimeTorqueLimits
from openpilot.selfdrive.controls.lib.blatv2.live_safety import (
  LiveSafetyState,
  InvalidOutputGuard,
  RECOVERY_OK_FRAMES,
)


LIMITS = RuntimeTorqueLimits(409, 4, 7, 1, 50, 2, 1)


def update(
  guard: InvalidOutputGuard,
  *,
  core_ok: bool,
  raw: float,
  applied: float,
  active: bool = True,
):
  return guard.update(
    active=active,
    core_ok=core_ok,
    raw_torque=raw,
    applied_torque=applied,
    driver_torque=0.0,
    limits=LIMITS,
  )


def test_valid_output_uses_exact_runtime_envelope():
  guard = InvalidOutputGuard(0.01)
  result = update(guard, core_ok=True, raw=1.0, applied=0.0)
  assert result.torque == 4 / 409
  assert result.state == LiveSafetyState.OK
  assert result.controls_valid and result.car_control_valid


def test_invalid_holds_decays_and_latches_without_bypassing_down_rate():
  guard = InvalidOutputGuard(0.01)
  first = update(guard, core_ok=False, raw=math.nan, applied=1.0)
  first_torque = first.torque
  assert first_torque == 1.0
  assert first.state == LiveSafetyState.HOLDING_FIRST_INVALID

  counts = [round(first_torque * LIMITS.steer_max)]
  current = first
  for _ in range(guard.invalid_latch_frames - 1):
    current = update(
      guard, core_ok=False, raw=math.nan, applied=current.torque,
    )
    counts.append(round(current.torque * LIMITS.steer_max))

  # The 250 ms validity latch is independent of command decay. Starting from
  # full torque, the exact seven-count down rate cannot have reached zero yet.
  assert current.state == LiveSafetyState.COMM_ISSUE_LATCHED
  assert counts[-1] == (
    LIMITS.steer_max
    - (guard.invalid_latch_frames - 1) * LIMITS.delta_down
  )
  assert counts[-1] > 0
  assert not current.controls_valid
  assert not current.car_control_valid
  assert all(
    0 <= counts[index] - counts[index + 1] <= LIMITS.delta_down
    for index in range(len(counts) - 1)
  )

  while counts[-1] > 0:
    current = update(
      guard, core_ok=False, raw=math.nan, applied=current.torque,
    )
    counts.append(round(current.torque * LIMITS.steer_max))
    assert current.state == LiveSafetyState.COMM_ISSUE_LATCHED
    assert not current.controls_valid
  assert all(
    0 <= counts[index] - counts[index + 1] <= LIMITS.delta_down
    for index in range(len(counts) - 1)
  )


def test_ten_consecutive_ok_frames_clear_same_latch_and_resume_through_slew():
  guard = InvalidOutputGuard(0.01)
  current = update(guard, core_ok=False, raw=math.nan, applied=0.2)
  for _ in range(guard.invalid_latch_frames - 1):
    current = update(
      guard, core_ok=False, raw=math.nan, applied=current.torque,
    )
  assert not current.controls_valid

  for frame in range(RECOVERY_OK_FRAMES - 1):
    current = update(guard, core_ok=True, raw=1.0, applied=0.0)
    assert current.recovery_ok_frames == frame + 1
    assert not current.controls_valid
    assert current.torque == 0.0
  recovered = update(guard, core_ok=True, raw=1.0, applied=0.0)
  assert recovered.state == LiveSafetyState.OK
  assert recovered.controls_valid and recovered.car_control_valid
  assert recovered.torque == 4 / 409


def test_invalid_during_recovery_restarts_consecutive_ok_count():
  guard = InvalidOutputGuard(0.01)
  first = update(guard, core_ok=False, raw=0.0, applied=0.1)
  update(guard, core_ok=True, raw=0.2, applied=first.torque)
  reset = update(guard, core_ok=False, raw=0.0, applied=first.torque)
  assert reset.recovery_ok_frames == 0


def test_inactive_resets_lifecycle_and_never_commands_torque():
  guard = InvalidOutputGuard(0.01)
  update(guard, core_ok=False, raw=math.nan, applied=0.5)
  inactive = update(
    guard, core_ok=False, raw=math.nan, applied=0.5, active=False,
  )
  assert inactive.torque == 0.0
  assert inactive.state == LiveSafetyState.INACTIVE
  assert inactive.controls_valid and inactive.car_control_valid

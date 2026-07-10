"""
Original concept and implementation by SpysyWeeb (github.com/SpysyWeeb).

Smooth Release. During one approach to a stop, the plan often pumps the brake:
it brakes, lets almost fully off mid-approach (a lead creeps forward, the e2e
plan wavers), then brakes again seconds later (route 00000035 stop 1: -1.4,
release to -0.15, re-brake, twice). Human drivers hold one application and bleed
pressure off gradually -- the owner's manual stops release at ~+0.2 m/s^2 per
second, in a single taper, every time.

This governor is that taper. While braking, the commanded accel may only *rise*
at a limited rate: the human taper while the car is rolling and the plan still
reads as a slowdown, or a brisk (but still continuous) rate the moment the plan
wants actual acceleration or the car is nearly stopped -- so a launch is never
meaningfully delayed, and the command never steps. Once the rising command
catches the raw demand the governor lets go on its own; because the limit grows
every frame, it always catches up and can never hold the car back indefinitely.

Safety posture: braking demands always pass through unlimited in the braking
direction -- only the *release* of brake is slowed, which errs toward more
braking, never less.
"""
from openpilot.common.realtime import DT_CTRL

RELEASE_JERK = 0.18     # m/s^3, floor on the release rate: the owner's fitted final-taper
RELEASE_HORIZON = 4.0   # s, any braking level bleeds to ~zero within this window (the measured
                        # length of the owner's release phase), so deep braking releases
                        # proportionally faster. FIELD LESSON (route 00000036): a flat 0.20 cap
                        # held slam-level braking against a relaxing plan for seconds (plans
                        # legitimately relax at ~0.55 m/s^3 from deep decel) and once parked the
                        # car 19m behind a lead; depth-scaled, the governor passes honest plan
                        # tapers and only resists the fast let-offs that become pumps
CANCEL_JERK = 2.5     # m/s^3, release rate once the plan wants to go or the car is nearly stopped --
                      # brisk (full release from -2.0 in 0.8s) but continuous, so cancelling the
                      # taper never steps the command the way an instant disarm would
GOVERN_FLOOR = -0.05  # m/s^2, braking level below which a release starts being governed; above it
                      # (cruising, accelerating) the governor never arms and output passes untouched
CANCEL_ACCEL = 0.05   # m/s^2, plan demand above this means "go" -- switch to the brisk rate
MIN_GOVERN_SPEED = 4.5  # m/s, below this use the brisk rate, never the slow taper. Every brake
                        # pump ever measured happened at road speed (10-13 m/s); at queue-creep
                        # speeds the plan's fast relaxations are always legitimate, and route 37
                        # showed the taper holding 0.4-0.7 extra there nearly parked the car 13m
                        # behind a creeping lead. Settle owns the landing regardless


class SmoothRelease:
  """Asymmetric jerk limit on the longcontrol output: braking engages instantly, releases as one taper."""

  def __init__(self):
    self.engaged = False  # clamping right now; read by longcontrol to freeze the PID integrator
                          # (holding more brake than the plan wants makes the error positive, and an
                          # integrator winding up against the clamp would overshoot on release)

  def reset(self) -> None:
    self.engaged = False

  def govern(self, output_accel: float, a_target: float, last_output: float, v_ego: float) -> float:
    # Arm only from real braking. Once armed, keep rate-limiting until the limit catches the raw
    # demand (engaged clears itself below), so letting go is always continuous, never a step.
    if not self.engaged and last_output >= GOVERN_FLOOR:
      return output_accel

    taper = v_ego >= MIN_GOVERN_SPEED and a_target <= CANCEL_ACCEL
    # fast-then-gentle, like a foot easing off: rate is proportional to remaining pressure
    rate = max(RELEASE_JERK, -last_output / RELEASE_HORIZON) if taper else CANCEL_JERK
    limit = last_output + rate * DT_CTRL
    self.engaged = output_accel > limit
    # min(): deeper braking passes through unlimited, only the release is rate-limited
    return min(output_accel, limit)

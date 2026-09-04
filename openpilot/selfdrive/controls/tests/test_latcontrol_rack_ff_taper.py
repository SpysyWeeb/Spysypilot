import math
from unittest import mock

from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.values import CAR as HYUNDAI
from opendbc.car.structs import car
from opendbc.car.vehicle_model import VehicleModel

from openpilot.cereal import log
from openpilot.common.test import OpenpilotTestCase
import openpilot.selfdrive.controls.lib.rack_trajectory as rack_trajectory
from openpilot.selfdrive.controls.lib.rack_trajectory import (
  FF_TAPER_ANGLE_DEG,
  FF_TAPER_RATE_BLEND_DEG_S,
  FF_TAPER_RATE_DEG_S,
  FF_TAPER_SPEED_BLEND_MPS,
  FF_TAPER_SPEED_MPS,
  JerkLimitedRackPlanner,
  RackTrajectoryController,
  _ff_taper_gate,
)
from openpilot.selfdrive.controls.tests.test_latcontrol_rack import horizon_model

# F3 highway feedforward taper. The gate itself (_ff_taper_gate) is tested directly, the way
# _direction_guard's own conflict/scale math is tested directly elsewhere in this package -- the
# route-wide old-build-vs-new frame check (bit-identity, and the mechanism's effect on the named
# 128 km/h segments) is the replay harness's job, not this unit test
# (spysypilot-route-audit/phase3/stability_2026-09-04_4d/impl_F3/).


class TestFFTaperGate(OpenpilotTestCase):

  def test_zero_below_highway_speed_regardless_of_angle_or_rate(self):
    # the two owner windows sit at 13-15 m/s (35-53 km/h): nowhere near FF_TAPER_SPEED_MPS
    for angle, rate in ((0.0, 0.0), (28.0, 0.0), (0.0, 40.0), (15.0, 15.0)):
      assert _ff_taper_gate(13.0, angle, angle, rate) == 0.0
    # and exactly at the speed gate's own lower edge, still exactly 0.0 (not epsilon over)
    assert _ff_taper_gate(FF_TAPER_SPEED_MPS - FF_TAPER_SPEED_BLEND_MPS, 0.0, 0.0, 0.0) == 0.0

  def test_zero_at_highway_speed_once_curvature_or_rate_leaves_the_deadband(self):
    # a genuine highway curve (target_angle past the deadband) must not be touched, whatever the wheel is doing
    assert _ff_taper_gate(35.0, FF_TAPER_ANGLE_DEG, 0.0, 0.0) == 0.0
    assert _ff_taper_gate(35.0, 10.0, 0.0, 0.0) == 0.0
    # ditto a genuine turn-in/unwind in progress (plan rate past the deadband), even at zero curvature
    assert _ff_taper_gate(35.0, 0.0, 0.0, FF_TAPER_RATE_DEG_S + FF_TAPER_RATE_BLEND_DEG_S) == 0.0
    assert _ff_taper_gate(35.0, 0.0, 0.0, 20.0) == 0.0
    # either side alone reopening (angle small again but rate still high, and vice versa) stays shut
    assert _ff_taper_gate(35.0, 0.2, 0.2, 20.0) == 0.0
    assert _ff_taper_gate(35.0, 10.0, 10.0, 0.5) == 0.0

  def test_fully_open_deep_in_a_highway_straight_hold(self):
    assert _ff_taper_gate(35.0, 0.0, 0.0, 0.0) == 1.0
    # matches the report's own worst window (seg 44, torque_decomp/rough_windows.json: 128 km/h,
    # near range 0.77 deg, target range 0.40 deg) -- well inside the deadband, meaningfully open
    gate = _ff_taper_gate(35.6, 0.8, 0.8, 0.0)
    assert 0.5 < gate < 1.0

  def test_continuous_through_every_boundary(self):
    # R7: each factor is a smoothstep, so the product is continuous everywhere, never a step
    speeds = [FF_TAPER_SPEED_MPS - FF_TAPER_SPEED_BLEND_MPS + step for step in range(0, 9)]
    gates = [_ff_taper_gate(speed, 0.0, 0.0, 0.0) for speed in speeds]
    assert gates == sorted(gates)  # monotonic in speed
    assert all(0.0 <= gate <= 1.0 for gate in gates)
    for previous, current in zip(gates, gates[1:]):
      assert current - previous < 0.4  # no boundary jump; a 1 m/s step moves the gate smoothly

    angles = [step * FF_TAPER_ANGLE_DEG / 8.0 for step in range(9)]
    angle_gates = [_ff_taper_gate(35.0, angle, angle, 0.0) for angle in angles]
    assert angle_gates == sorted(angle_gates, reverse=True)  # monotonically closes as curvature grows
    for previous, current in zip(angle_gates, angle_gates[1:]):
      assert previous - current < 0.4

  def test_symmetric_in_sign(self):
    # a left turn's dither must taper exactly as much as a right turn's (angle and rate enter as abs())
    for angle, rate in ((1.0, -2.0), (-1.0, 2.0), (-1.0, -2.0), (1.0, 2.0)):
      assert math.isclose(_ff_taper_gate(35.0, angle, angle, rate), _ff_taper_gate(35.0, 1.0, 1.0, 2.0))


class TestFFTaperController(OpenpilotTestCase):

  def setUp(self):
    CarInterface = interfaces[HYUNDAI.HYUNDAI_PALISADE]
    self.CP = CarInterface.get_non_essential_params(HYUNDAI.HYUNDAI_PALISADE)
    self.CI = CarInterface(self.CP)
    self.VM = VehicleModel(self.CP)
    self.params = log.VehicleParameters.new_message()

  def _run_highway_dither(self, speed, disable_taper=False, frames=80, half_period_frames=3, dither_deg=0.6):
    """A near-zero-curvature target flipping sign every `half_period_frames` model frames -- the
    same kind of small, fast near-target dither the report's rough_windows.json finds dominating
    segments 37-44 (near/target range well under a degree at ~128 km/h), reproduced here at a
    fixed, controlled speed and amplitude so before/after can be compared frame for frame."""
    torque_from_lateral_accel = self.CI.torque_from_lateral_accel()
    controller = RackTrajectoryController()
    controller.planner = JerkLimitedRackPlanner(0.0)
    CS = car.CarState.new_message()
    CS.vEgo = speed
    CS.steeringAngleDeg = 0.0
    model = horizon_model([0.0, .5, 1.0, 1.5, 2.0, 2.5], [0.0] * 6, [speed] * 6)
    patches = [mock.patch.object(rack_trajectory, "FF_TAPER_SPEED_MPS", 1e9)] if disable_taper else []
    for patch in patches:
      patch.start()
    try:
      outputs = []
      for index in range(frames):
        target_angle = dither_deg if (index // half_period_frames) % 2 == 0 else -dither_deg
        desired_curvature = -self.VM.calc_curvature(math.radians(target_angle), speed, 0.0)
        model.timestampEof = 1_000_000_000 + (index // 5) * 50_000_000
        model.action.desiredCurvature = desired_curvature
        controller.set_model(model, model.timestampEof + 50_000_000)
        output = controller.update(True, CS, self.VM, self.params, self.CP.lateralTuning.torque,
                                    torque_from_lateral_accel, .2, desired_curvature)
        assert output is not None
        outputs.append(output)
    finally:
      for patch in patches:
        patch.stop()
    return outputs

  @staticmethod
  def _mean_abs_step(values, warmup=10):
    settled = values[warmup:]
    return sum(abs(b - a) for a, b in zip(settled, settled[1:])) / (len(settled) - 1)

  def test_taper_meaningfully_smooths_highway_straight_feedforward(self):
    speed = 35.0  # m/s, 126 km/h -- inside segments 37-44's own ~120-129 km/h band
    without = self._run_highway_dither(speed, disable_taper=True)
    with_taper = self._run_highway_dither(speed, disable_taper=False)
    feedforward_without = [output.feedforward_torque for output in without]
    feedforward_with = [output.feedforward_torque for output in with_taper]
    chatter_without = self._mean_abs_step(feedforward_without)
    chatter_with = self._mean_abs_step(feedforward_with)
    assert chatter_without > 0.001  # sanity: the untapered fixture really does chatter
    # meaningfully smoothed (measured: ~4.2x on this fixture), not just nudged
    assert chatter_with < 0.5 * chatter_without
    # steady-state authority is preserved (measured: <1% mean shift on this fixture); the
    # authoritative check is the real route: impl_F3/analyze_4d.py's "steady-state authority"
    # section on segments 37-44 (route 4d, the report's own worst highway stretch)
    mean_without = sum(feedforward_without[10:]) / len(feedforward_without[10:])
    mean_with = sum(feedforward_with[10:]) / len(feedforward_with[10:])
    assert abs(mean_with - mean_without) < 0.1 * abs(mean_without)

  def test_taper_is_bit_identical_below_highway_speed(self):
    # the same dither fixture at 13 m/s (47 km/h, inside the two owner windows' own 35-53 km/h
    # band): the speed gate alone is closed the entire run, so the taper and its disabled reference
    # must be bit-for-bit identical every frame
    speed = 13.0
    without = self._run_highway_dither(speed, disable_taper=True)
    with_taper = self._run_highway_dither(speed, disable_taper=False)
    for baseline_output, taper_output in zip(without, with_taper, strict=True):
      assert taper_output.feedforward_torque == baseline_output.feedforward_torque
      assert taper_output.torque == baseline_output.torque

  def test_taper_transient_at_a_highway_speed_turn_in_onset_is_bounded_and_short(self):
    # highway speed (above FF_TAPER_SPEED_MPS) with a genuine, fast turn-in starting from zero: the
    # angle gate is keyed to target_angle (the reference-filtered target, same variable
    # turn_in_fraction uses) and the rate gate to plan.rate_deg_s (the virtual rack's own rate, same
    # variable the hold top-up's plan_rate_gate uses) -- and *both*, like the top-up's own gate,
    # necessarily start open at rest and take a fraction of a response time to close as a new
    # demand ramps up. Verified here, not assumed: during that ramp the taper is NOT bit-identical
    # to baseline -- it blends in a small, bounded amount for a handful of frames before the gate
    # shuts for the remainder of the turn-in. This is the honest characterization: bounded and
    # brief, not absent.
    speed = 35.0
    torque_from_lateral_accel = self.CI.torque_from_lateral_accel()

    def run(disable_taper):
      controller = RackTrajectoryController()
      controller.planner = JerkLimitedRackPlanner(0.0)
      CS = car.CarState.new_message()
      CS.vEgo = speed
      CS.steeringAngleDeg = 0.0
      model = horizon_model([0.0, .5, 1.0, 1.5, 2.0, 2.5], [0.0] * 6, [speed] * 6)
      patches = [mock.patch.object(rack_trajectory, "FF_TAPER_SPEED_MPS", 1e9)] if disable_taper else []
      for patch in patches:
        patch.start()
      try:
        outputs = []
        for index in range(60):
          # a real, fast highway curve entry: 0 -> 12 deg over 30 frames (0.3s, 40 deg/s), well past
          # the deadband once underway
          target_angle = min(12.0, index * 0.4)
          desired_curvature = -self.VM.calc_curvature(math.radians(target_angle), speed, 0.0)
          model.timestampEof = 1_000_000_000 + (index // 5) * 50_000_000
          model.action.desiredCurvature = desired_curvature
          controller.set_model(model, model.timestampEof + 50_000_000)
          output = controller.update(True, CS, self.VM, self.params, self.CP.lateralTuning.torque,
                                      torque_from_lateral_accel, .2, desired_curvature)
          assert output is not None
          outputs.append(output)
      finally:
        for patch in patches:
          patch.stop()
      return outputs

    without = run(True)
    with_taper = run(False)
    deltas = [taper.torque - base.torque for base, taper in zip(without, with_taper)]
    # bounded: on this fixture the largest deviation is ~0.0036 (~1.5 CAN counts of 409, opendbc's
    # STEER_MAX for this platform -- materials.md), well under R7_MAX_TORQUE_STEP (0.05, ~20 counts).
    # A real highway turn-in can cost more than this synthetic fixture shows -- see
    # impl_F3/highway_turnin_scan.py's route-4d scan (median 0.014, worst 0.051 across 17 events at
    # RC=0.1s) and FIX_NOTE.md's honest risk writeup; this unit test only pins the mechanism's shape.
    assert max(abs(delta) for delta in deltas) < 0.006
    # and it does not linger: exactly bit-identical again well before the ramp finishes (by the time
    # the served -- reference-filtered -- target has cleared the deadband, index 11 on this fixture,
    # ~0.11 s into a 0.3 s turn-in)
    for baseline_output, taper_output in list(zip(without, with_taper))[11:]:
      assert taper_output.torque == baseline_output.torque
      assert taper_output.feedforward_torque == baseline_output.feedforward_torque

  def test_reset_zeroes_the_taper_filter(self):
    controller = RackTrajectoryController()
    controller.ff_taper_filter.x = 0.42
    controller.reset()
    assert controller.ff_taper_filter.x == 0.0

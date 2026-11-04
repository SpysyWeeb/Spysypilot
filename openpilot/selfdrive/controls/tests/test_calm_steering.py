import math
import os
import tempfile
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from openpilot.cereal import log
from opendbc.car.structs import car
from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.values import CAR
from opendbc.car.lateral import apply_driver_steer_torque_limits
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.parameterized import parameterized
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_CTRL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.calm_steering import CalmSteering, smoothstep
from openpilot.selfdrive.controls.lib.latcontrol_torque import INTERP_SPEEDS, KP_INTERP
from openpilot.tools.lib.logreader import LogReader, save_log
from tools.scripts.calm_steering_replay import chatter, frames_from_log, make_controller, replay, summarize

CAR_NAME = CAR.HYUNDAI_PALISADE
CP = interfaces[CAR_NAME].get_non_essential_params(CAR_NAME)
CI = interfaces[CAR_NAME](CP)
VM = VehicleModel(CP)
LAF = CP.lateralTuning.torque.latAccelFactor
LAT_DELAY = 0.3
LEAD = 200  # 2 s of straight road first, so the windows are full and the gate is open where it can be
ENVELOPES = ((384, 3, 7), (409, 4, 7))  # stock opendbc Hyundai and the daily branch's fork


def angle_for(lat_accel, v):
  # the steering angle (CS.steeringAngleDeg sign, left positive) whose lateral acceleration the controller measures as
  # lat_accel; LatControlTorque negates VM.calc_curvature, so this negates VM.get_steer_from_curvature
  return -math.degrees(VM.get_steer_from_curvature(lat_accel / v**2, v, 0.0))


def curvature_for(angle_deg, v, roll=0.0):
  # the desired curvature whose desired angle (calm steering's convention) is angle_deg
  return -VM.calc_curvature(math.radians(angle_deg), v, roll)


def quantize(angle_deg):
  return round(angle_deg / 0.1) * 0.1


def open_fraction(v):
  # the speed factor from literals, so a silent re-tune of the 22-30 m/s ramp fails here
  t = min(max((v - 20.0) / (28.0 - 20.0), 0.0), 1.0)
  return t * t * (3 - 2 * t)


def delayed(x, frames):
  return np.concatenate([np.full(frames, x[0]), x[:-frames]])


@dataclass
class Run:
  out: np.ndarray
  gate: np.ndarray
  f: np.ndarray       # pid_log.f: the feedforward, which is the (blended) friction when desired, roll and offset are 0
  error: np.ndarray


def drive(ctl, v, angle, curvature, pressed=False, active=True, lat_delay=LAT_DELAY, roll=0.0, angle_offset=0.0) -> Run:
  n = len(angle)
  v, pressed, active = (np.broadcast_to(np.asarray(x), (n,)) for x in (v, pressed, active))
  params = log.VehicleParameters.new_message(roll=roll, angleOffsetDeg=angle_offset)
  run = Run(*(np.zeros(n) for _ in range(4)))
  for i in range(n):
    CS = car.CarState.new_message(vEgo=float(v[i]), steeringAngleDeg=float(angle[i]), steeringPressed=bool(pressed[i]))
    run.out[i], _, lac_log = ctl.update(bool(active[i]), CS, VM, params, False, float(curvature[i]), False, lat_delay)
    run.gate[i], run.f[i], run.error[i] = ctl.calm.last_gate, lac_log.f, lac_log.error
  return run


def both(*args, **kwargs) -> tuple[Run, Run]:
  return drive(make_controller(CP, CI, False), *args, **kwargs), drive(make_controller(CP, CI, True), *args, **kwargs)


def staircase(v, seconds=40.0):
  # the car follows a slow 0.3 deg wander exactly; only the 0.1 deg sensor quantum differs from the target
  t = np.arange(round(seconds / DT_CTRL)) * DT_CTRL
  angle = np.concatenate([np.zeros(LEAD), 0.3 * np.sin(2 * np.pi * t / 8.0)])
  return [quantize(a) for a in angle], [curvature_for(a, v) for a in angle]


def move(rate_deg_s, peak_deg, hold_s=2.0, back=False, v=30.0):
  # straight, then the desired angle ramps to peak_deg at rate_deg_s (and back), the measured angle 0.3 s behind
  ramp = np.arange(0.0, peak_deg, rate_deg_s * DT_CTRL)
  desired = np.concatenate([np.zeros(LEAD), ramp, np.full(round(hold_s / DT_CTRL), peak_deg)])
  if back:
    desired = np.concatenate([desired, ramp[::-1], np.zeros(round(hold_s / DT_CTRL))])
  measured = [quantize(a) for a in delayed(desired, 30)]
  return measured, [curvature_for(a, v) for a in desired], LEAD + len(ramp)


def model_dither(amplitude, seconds=20.0, v=30.0):
  n = LEAD + round(seconds / DT_CTRL)
  lat_accel = [0.0 if i < LEAD else (amplitude if (i // 5) % 2 else -amplitude) for i in range(n)]  # a new value every 20 Hz frame
  return np.zeros(n), np.array(lat_accel) / v**2


def onset(out, start, fraction, reference):
  return start + int(np.argmax(np.abs(out[start:]) >= fraction * abs(reference)))


class SteeringPlant:
  """The wheel does not move until the applied torque is more than `stiction` beyond the self-aligning torque of the
  angle it holds; then it moves toward the angle that torque (less the stiction) holds, with a first-order lag. The
  command goes through apply_driver_steer_torque_limits and a transport delay first; the angle sensor reads 0.1 deg
  steps. stiction 0.0 is a quantization-only plant."""
  def __init__(self, v, lag, delay, envelope, stiction=0.13):
    steer_max, delta_up, delta_down = envelope
    self.limits = SimpleNamespace(STEER_MAX=steer_max, STEER_DELTA_UP=delta_up, STEER_DELTA_DOWN=delta_down,
                                  STEER_DRIVER_ALLOWANCE=50, STEER_DRIVER_MULTIPLIER=2, STEER_DRIVER_FACTOR=1)
    self.v, self.lag, self.stiction = v, lag, stiction
    self.counts = 0
    self.pipe = deque([0.0] * round(delay / DT_CTRL))
    self.lat_accel = 0.0  # controller convention: what LatControlTorque measures

  def angle(self):
    return quantize(angle_for(self.lat_accel, self.v))

  def step(self, command):
    self.counts = apply_driver_steer_torque_limits(round(command * self.limits.STEER_MAX), self.counts, 0.0, self.limits)
    self.pipe.append(-self.counts / self.limits.STEER_MAX)  # the controller returns -output_torque
    torque = self.pipe.popleft()
    beyond = torque - self.lat_accel / LAF
    if abs(beyond) > self.stiction:
      target = LAF * (torque - math.copysign(self.stiction, beyond))
      self.lat_accel += (target - self.lat_accel) * DT_CTRL / (self.lag + DT_CTRL)


def closed_loop(gate_open, target, seconds, lag=0.15, delay=0.1, lat_delay=LAT_DELAY, envelope=ENVELOPES[0], stiction=0.0, v=30.0):
  ctl = make_controller(CP, CI, gate_open)
  plant = SteeringPlant(v, lag, delay, envelope, stiction)
  params = log.VehicleParameters.new_message()
  n = round(seconds / DT_CTRL)
  out, lat_accel, gate = np.zeros(n), np.zeros(n), np.zeros(n)
  for i in range(n):
    CS = car.CarState.new_message(vEgo=v, steeringAngleDeg=plant.angle())
    out[i] = ctl.update(True, CS, VM, params, False, target(i * DT_CTRL) / v**2, False, lat_delay)[0]
    plant.step(out[i])
    lat_accel[i], gate[i] = plant.lat_accel, ctl.calm.last_gate
  return out, lat_accel, gate


def settling_frame(lat_accel, start, final):
  outside = np.flatnonzero(np.abs(lat_accel[start:] - final) > 0.1 * abs(final))
  return int(outside[-1]) + 1 if len(outside) else 0


def golden_sine(v):
  t = np.arange(2000) * DT_CTRL
  angle = [quantize(-1.5 * math.sin(2 * math.pi * 0.3 * ti)) for ti in t]
  return angle, 0.3 * np.sin(2 * np.pi * 0.3 * (t - 0.3)) / v**2


def golden_boundary():
  # 20.0 m/s, the desired angle ramping at 5.0 deg/s to 3.0 deg and back: the speed factor is exactly 0 throughout and the
  # angle factor wherever 3.0 deg is reached
  t = np.arange(2000) * DT_CTRL
  phase = t % 2.0
  desired = np.clip(np.minimum(5.0 * phase, 5.0 * (2.0 - phase)), 0.0, 3.0)
  curvature = [-VM.calc_curvature(math.radians(a), 20.0, 0.0) for a in desired]
  return [quantize(a) for a in np.concatenate([np.zeros(30), desired[:-30]])], curvature


class TestCalmSteering(OpenpilotTestCase):
  # (a) recorded from the unmodified LatControlTorque (upstream 38a2cb9) before the hook was added: (sum, out[500], out[1999])
  @parameterized.expand([
    ('town', golden_sine(15.0), 15.0, False, (-14.893529756723115, -0.403047405825556, 0.39999973948644113)),
    ('highway_hands_on', golden_sine(30.0), 30.0, True, (-2.225311632847072, -0.2689719138137526, 0.2748371188038936)),
    ('gate_boundary', golden_boundary(), 20.0, False, (164.82866129079488, 0.12670704866893678, -0.08461346363425336)),
  ])
  def test_golden(self, _, sequence, v, pressed, golden):
    run = drive(make_controller(CP, CI, True), v, *sequence, pressed=pressed)
    np.testing.assert_allclose((run.out.sum(), run.out[500], run.out[1999]), golden, rtol=0, atol=1e-12)
    assert np.all(run.gate == 0.0)
    assert CalmSteering.gate_from(20.0, 3.0, 5.0, False) == 0.0
    assert CalmSteering.gate_from(30.0, 3.0, 0.0, False) == 0.0
    assert CalmSteering.gate_from(20.0, 0.0, 0.0, False) == 0.0
    assert CalmSteering.gate_from(30.0, 0.0, 5.0, False) == 1.0

  # (b1) the sensor staircase: a 0.1 deg step is 0.0159 of torque through P and friction at 30 m/s
  @parameterized.expand([(20.0,), (26.0,), (30.0,), (35.0,)])
  def test_staircase(self, v):
    stock, calm = both(v, *staircase(v))
    s, c = chatter(stock.out[LEAD:]), chatter(calm.out[LEAD:])
    # precondition scaled by the torque one 0.1 deg step makes through P and friction at this speed (0.014 at 30 m/s)
    per_step = lambda u: (np.interp(u, INTERP_SPEEDS, KP_INTERP) + 1.656) * 0.1 / abs(angle_for(1.0, u))  # noqa: E731
    assert s[0] >= 0.014 * per_step(v) / per_step(30.0)
    open_ = open_fraction(v)
    if open_ == 0.0:
      assert np.array_equal(calm.out, stock.out)
      return
    assert np.mean(calm.gate[LEAD:]) >= 0.95 * open_
    assert c[0] <= (1 - 0.4 * open_) * s[0]

  # (b4)
  def test_staircase_no_bias(self):
    stock, calm = both(30.0, *staircase(30.0))
    assert abs(np.mean(calm.out) - np.mean(stock.out)) < 0.001

  # (b2) the sensor flip-flopping between two quanta
  def flip_flop(self):
    dwell, angle, k = [3, 5, 4, 2, 6, 3], [0.0] * LEAD, 0
    while len(angle) < LEAD + 2000:
      angle += [0.0 if k % 2 == 0 else -0.1] * dwell[k % len(dwell)]
      k += 1
    angle = angle[:LEAD + 2000]
    stock, calm = both(30.0, angle, np.zeros(len(angle)))
    return chatter(stock.out[LEAD:]), chatter(calm.out[LEAD:])

  def test_flip_flop(self):
    s, c = self.flip_flop()
    assert s[2] >= 600
    assert c[1] <= 0.6 * s[1]

  def test_flip_flop_reversals(self):
    s, c = self.flip_flop()
    # the P term alone moves 0.005 per sensor step and crosses the 0.008 hysteresis: calm removes the friction share of
    # the reversals, not the P share (measured 513 vs 1560/min, 33 %)
    assert c[2] <= 0.4 * s[2]

  # (b3) the model's own 20 Hz dither: alternate frames +/-0.02 m/s^2 stay under the rate knee, so the gate stays open
  def test_model_dither(self):
    stock, calm = both(30.0, *model_dither(0.02))
    assert chatter(calm.out[LEAD:])[1] <= 0.5 * chatter(stock.out[LEAD:])[1]
    assert np.mean(calm.gate[LEAD:]) >= 0.95

  # (b5) a bigger dither: only never worse is asserted; the gate averages 0.97 here, not the ~0.5-0.7 first estimated
  # (strictly alternating 20 Hz values look the same at both ends of the 0.1 s window)
  def test_big_model_dither(self):
    stock, calm = both(30.0, *model_dither(0.05))
    assert chatter(calm.out[LEAD:])[1] <= chatter(stock.out[LEAD:])[1]

  # (c) a highway turn-in closes the gate and keeps stock's timing
  def test_turn_in(self):
    measured, curvature, end = move(8.0, 4.0)
    stock, calm = both(30.0, measured, curvature)
    start = LEAD
    assert np.all(calm.gate[start - 10:start] > 0.95)
    assert calm.gate[start + 15] < 0.5
    reference = stock.out[end + 100]
    assert onset(calm.out, start, 0.25, reference) <= onset(stock.out, start, 0.25, reference) + 1

  # (c2) calm never steps harder than stock, and its difference from stock never steps by more than 0.02
  @parameterized.expand([('turn_in', False), ('turn_in_and_back', True)])
  def test_moves_step_no_harder(self, _, back):
    measured, curvature, _ = move(8.0, 4.0, hold_s=0.5 if back else 2.0, back=back)
    stock, calm = both(30.0, measured, curvature)
    assert np.abs(np.diff(calm.out)).max() <= np.abs(np.diff(stock.out)).max()
    assert np.abs(np.diff(calm.out - stock.out)).max() <= 0.02

  def test_model_step_no_harder(self):
    lat_accel = np.concatenate([np.zeros(LEAD), np.full(300, 0.09)])
    measured = [quantize(angle_for(a, 30.0)) for a in delayed(lat_accel, 30)]
    stock, calm = both(30.0, measured, lat_accel / 30.0**2)
    assert np.abs(np.diff(calm.out)).max() <= np.abs(np.diff(stock.out)).max()

  # (d) gentle highway moves: the filter costs at most 6 frames to 25 % and 11 frames (one RC) to 63 %
  @parameterized.expand([(1.0,), (1.5,), (2.5,)])
  def test_gentle_move_onset(self, rate):
    measured, curvature, end = move(rate, rate * 1.0)
    stock, calm = both(30.0, measured, curvature)
    reference = stock.out[end + 100]
    assert onset(calm.out, LEAD, 0.25, reference) - onset(stock.out, LEAD, 0.25, reference) <= 6
    assert onset(calm.out, LEAD, 0.63, reference) - onset(stock.out, LEAD, 0.63, reference) <= 11

  # (e) hands on the wheel: stock exactly, and back in by ramp
  def test_hands(self):
    # a sensor-only ripple (the target is straight), phased so a 0.1 deg step lands a few frames before the press and the release
    t = np.arange(3000) * DT_CTRL
    angle = [quantize(0.3 * math.sin(2 * math.pi * (ti + 0.316) / 10.0)) for ti in t]
    assert angle[990] != angle[999] and angle[1990] != angle[1999]
    pressed = (t >= 10.0 - 1e-9) & (t < 20.0 - 1e-9)
    assert pressed[1000] and not pressed[999] and not pressed[2000]
    stock, calm = both(30.0, angle, np.zeros(3000), pressed=pressed)
    diff = calm.out - stock.out
    assert np.abs(diff[LEAD:1000]).max() >= 0.005
    assert calm.gate[999] > 0.9 and calm.gate[1000] == 0.0
    assert np.array_equal(calm.out[1000:2000], stock.out[1000:2000])
    assert abs(diff[1000] - diff[999]) <= 0.011
    ramp = slice(2000, 2000 + round(0.25 / DT_CTRL) + 1)
    assert np.all(np.diff(calm.gate[ramp.start - 1:ramp.stop]) <= DT_CTRL / 0.25 + 1e-12)
    assert calm.gate[ramp.stop] > 0.9
    # the filter still lags the step from before the release, so the ramp has something to blend in
    assert np.abs(calm.f[ramp] - stock.f[ramp]).max() / LAF > 0.0005
    # the blended friction (the feedforward here) never moves by more than one count at 409 in a frame during the ramp
    assert np.abs(np.diff(calm.f[ramp.start - 1:ramp.stop])).max() / LAF <= 1 / 409

  # (f) the filter is warm: it follows the friction while the gate is closed or lateral is inactive
  def test_warm_filter(self):
    n_town, n_ramp, n_hwy = 6000, 500, 200
    v = np.concatenate([np.full(n_town, 15.0), np.linspace(15.0, 30.0, n_ramp), np.full(n_hwy, 30.0)])
    # a steady error, so the friction has a DC part, plus a slow wander, so it is still moving when the gate opens
    angle = 0.5 + 0.3 * np.sin(2 * np.pi * np.arange(len(v)) * DT_CTRL / 4.0)
    runs = []
    for gate_open in (False, True):
      ctl = make_controller(CP, CI, gate_open)
      ctl.update_torque_parameters(LAF, 0.2, CP.lateralTuning.torque.friction)
      runs.append(drive(ctl, v, angle, np.zeros(len(v))))
    stock, calm = runs
    first = int(np.argmax(calm.gate > 0.0))
    assert n_town < first < n_town + n_ramp
    assert abs(calm.out[first] - stock.out[first]) < 0.005
    assert np.abs(calm.out[first:first + 50] - stock.out[first:first + 50]).max() < 0.005
    # the blend is against a 0.1 s filter that has run on every frame since the first, the closed town ones too
    # (pid_log.f, a float32, is the friction less the 0.2 offset here); a cold filter, snapping when the gate opens, misses this
    friction = stock.f + 0.2
    reference = FirstOrderFilter(0.0, 0.1, DT_CTRL, initialized=False)
    warm = np.array([reference.update(x) for x in friction])
    assert np.abs(warm - friction)[first:first + 50].min() > 0.001  # the filter lags the moving friction, so the check has teeth
    np.testing.assert_allclose(calm.f[first:] + 0.2, (friction + calm.gate * (warm - friction))[first:], rtol=0, atol=1e-7)

  @parameterized.expand([('inactive',), ('pressed',)])
  def test_warm_after_inactive(self, closed_by):
    # engaged at +0.4 deg, then 30 s with the gate shut (inactive, or hands on) while the error flips to -0.4 deg, then back
    n = 1000 + 3000 + 100
    angle = np.where(np.arange(n) < 1000, 0.4, -0.4)
    shut = (np.arange(n) >= 1000) & (np.arange(n) < 4000)
    kwargs = {'active': ~shut} if closed_by == 'inactive' else {'pressed': shut}
    resumed = drive(make_controller(CP, CI, True), 30.0, angle, np.zeros(n), **kwargs)
    always = drive(make_controller(CP, CI, True), 30.0, angle, np.zeros(n))
    stock = drive(make_controller(CP, CI, False), 30.0, angle, np.zeros(n))
    # the first sample snaps the filter: a steady friction passes through from the first frame the gate opens (pid_log.f is a float32)
    assert always.gate[20] > 0.0 and always.gate[999] > 0.9
    np.testing.assert_allclose(always.f[:1000], stock.f[:1000], rtol=0, atol=1e-7)
    assert resumed.gate[999] > 0.9 and resumed.gate[3999] == 0.0 and resumed.gate[4000] > 0.0
    # the filter followed the flip while shut: no +0.4 deg friction left in it to blend back in
    assert abs(stock.f[999] - stock.f[4000]) > 0.1
    assert abs(resumed.f[4000] - always.f[4000]) < 1e-9
    np.testing.assert_allclose(resumed.f[4000:], stock.f[4000:], rtol=0, atol=1e-7)

  @parameterized.expand([('inactive',), ('pressed',)])
  def test_windows_while_inactive(self, closed_by):
    # the angle windows take every frame, inactive and hands-on ones too: lateral off (or the driver's hands on) for 1 s
    # while the wheel goes from +0.9 to -0.9 deg, then back on (released). The first open frame reads the angle as held,
    # as a controller active and hands-off throughout does, and the gate starts its rise; windows still holding the
    # samples from before would read 9 deg/s and keep it shut for 0.2 s more
    frame = np.arange(1130)
    angle = np.where(frame < 1050, 0.9, -0.9)
    curvature = [curvature_for(a, 30.0) for a in angle]
    shut = (frame >= 1000) & (frame < 1100)
    kwargs = {'active': ~shut[:1101]} if closed_by == 'inactive' else {'pressed': shut[:1101]}
    resumed, always = make_controller(CP, CI, True), make_controller(CP, CI, True)
    before = drive(resumed, 30.0, angle[:1101], curvature[:1101], **kwargs)
    drive(always, 30.0, angle[:1101], curvature[:1101])
    for window in ('desired_angles', 'measured_angles'):
      assert list(getattr(resumed.calm, window)) == list(getattr(always.calm, window))
    assert resumed.calm.rate() == always.calm.rate() == 0.0
    after = drive(resumed, 30.0, angle[1101:], curvature[1101:])
    gate = np.concatenate([before.gate, after.gate])
    assert gate[999] > 0.75 and np.all(gate[1000:1100] == 0.0)
    np.testing.assert_allclose(np.diff(gate[1099:1119]), 0.01 / 0.25, rtol=0, atol=1e-12)

  # (g) closed loop: a quantization limit cycle on a highway hold, and never worse elsewhere
  @parameterized.expand(ENVELOPES)
  def test_limit_cycle(self, *envelope):
    stock, calm = (closed_loop(g, lambda t: 0.1, 25.0, lag=0.15, delay=0.1, envelope=envelope)[0] for g in (False, True))
    stock_p2p, calm_p2p = np.ptp(stock[-1000:]), np.ptp(calm[-1000:])
    assert stock_p2p >= 0.02
    assert calm_p2p <= 0.25 * stock_p2p

  @parameterized.expand(ENVELOPES)
  def test_limit_cycle_with_stiction(self, *envelope):
    stock, calm = (closed_loop(g, lambda t: 0.1, 25.0, envelope=envelope, stiction=0.13)[0] for g in (False, True))
    # with 0.13 of stiction the wheel sticks on the hold (stock p2p 0.0159, one quantum event): there is no cycle to
    # suppress, so the bound is not worse
    assert np.ptp(calm[-1000:]) <= np.ptp(stock[-1000:]) + 0.005

  GRID = [(lag, delay, lat_delay, envelope, stiction) for stiction in (0.0, 0.13) for envelope in ENVELOPES
          for lag, delay in ((0.2, 0.1), (0.3, 0.2), (0.2, 0.2), (0.4, 0.3)) for lat_delay in (0.10, 0.25)]
  # two tiers, measured 2026-11-04. Where stock holds (p2p <= 0.1) calm is within 0.005 of it: the stuck-wheel creep
  # cases go 0.0159 -> 0.0203..0.0207, one 0.1 deg event inside the 10 s window. Where stock itself is unstable (lag
  # 0.2 s, delay 0.2 s, lat_delay 0.10: 0.289 -> 0.307 and 0.353 -> 0.384) calm is within 10 %: the gate opens and
  # shuts with the swing; held open it would be 19-26 % smaller
  UNSTABLE_P2P = 0.1

  @parameterized.expand(GRID)
  def test_never_worse(self, lag, delay, lat_delay, envelope, stiction):
    stock, calm = (closed_loop(g, lambda t: 0.1, 25.0, lag, delay, lat_delay, envelope, stiction)[0] for g in (False, True))
    stock_p2p, calm_p2p = np.ptp(stock[-1000:]), np.ptp(calm[-1000:])
    if stock_p2p <= self.UNSTABLE_P2P:
      assert calm_p2p <= stock_p2p + 0.005
    else:
      assert calm_p2p <= 1.1 * stock_p2p

  def test_never_worse_unstable_tier(self):
    # the 1.1 bound is not dead code: at least one grid case has stock itself unstable
    case = (0.2, 0.2, 0.10, ENVELOPES[0], 0.0)
    assert case in self.GRID
    stock = closed_loop(False, lambda t: 0.1, 25.0, *case)[0]
    assert np.ptp(stock[-1000:]) > self.UNSTABLE_P2P

  @parameterized.expand([(envelope, stiction) for envelope in ENVELOPES for stiction in (0.0, 0.13)])
  def test_step_settling(self, envelope, stiction):
    target = lambda t: 0.0 if t < 2.0 else 0.5  # noqa: E731
    (stock, stock_accel, _), (calm, calm_accel, gate) = (closed_loop(g, target, 12.0, envelope=envelope, stiction=stiction) for g in (False, True))
    assert gate[190:200].min() > 0.95
    assert abs(settling_frame(calm_accel, 200, 0.5) - settling_frame(stock_accel, 200, 0.5)) <= 5
    # the 0.5 m/s^2 hold is 3.0 deg at 30 m/s: the gate is shut and the output is stock's
    assert np.all(gate[200:] == 0.0)
    assert np.array_equal(calm[-500:], stock[-500:])

  # (h) the gate
  def test_gate_table(self):
    gate_from = CalmSteering.gate_from
    for v, expected in ((22.0, 0.15625), (24.0, 0.5), (26.0, 0.84375)):
      assert gate_from(v, 0.0, 0.0, False) == expected
    assert gate_from(20.0, 0.0, 0.0, False) == 0.0
    assert gate_from(28.0, 0.0, 0.0, False) == 1.0
    for angle, expected in ((0.75, 0.84375), (1.5, 0.5), (2.25, 0.15625)):
      assert gate_from(30.0, angle, 0.0, False) == expected
      assert gate_from(30.0, -angle, 0.0, False) == expected
    for rate in (2.75, 3.5, 4.25):
      assert gate_from(30.0, 0.0, rate, False) == 1.0
    for rate, expected in ((5.75, 0.84375), (6.5, 0.5), (7.25, 0.15625)):
      assert gate_from(30.0, 0.0, rate, False) == expected
    assert gate_from(30.0, 0.0, 8.0, False) == 0.0
    assert gate_from(30.0, 0.0, 0.0, True) == 0.0
    for bad in (math.nan, math.inf, -math.inf):
      assert gate_from(bad, 0.0, 0.0, False) == 0.0
      assert gate_from(30.0, bad, 0.0, False) == 0.0
      assert gate_from(30.0, 0.0, bad, False) == 0.0
    assert smoothstep(22.0, 22.0, 30.0) == 0.0 and smoothstep(30.0, 22.0, 30.0) == 1.0

  def held_gate(self, desired_deg, measured_deg):
    calm = CalmSteering(DT_CTRL)
    CS = car.CarState.new_message(vEgo=30.0, steeringAngleDeg=measured_deg)
    params = log.VehicleParameters.new_message()
    for _ in range(100):
      calm.update(0.0, CS, VM, params, curvature_for(desired_deg, 30.0), True)
    return calm.last_gate

  def test_gate_angles(self):
    for angle, expected in ((0.75, 0.84375), (1.5, 0.5), (2.25, 0.15625)):
      np.testing.assert_allclose(self.held_gate(angle, 0.0), expected, atol=1e-9)
      np.testing.assert_allclose(self.held_gate(0.0, -angle), expected, atol=1e-9)

  def window(self, desired, measured):
    calm = CalmSteering(DT_CTRL)
    params = log.VehicleParameters.new_message()
    gates = []
    for d, m in zip(desired, measured, strict=True):
      calm.update(0.0, car.CarState.new_message(vEgo=30.0, steeringAngleDeg=float(m)), VM, params, curvature_for(d, 30.0), True)
      gates.append(calm.last_gate)
    return np.array(gates)

  def test_measured_window(self):
    # a one-frame 1.8 deg sensor step (9 deg/s over the 0.2 s window) shuts the gate for exactly the window's 20 frames
    measured = np.concatenate([np.full(200, -0.9), np.full(100, 0.9)])
    gates = self.window(np.zeros(300), measured)
    assert gates[199] > 0.75
    assert np.all(gates[200:220] == 0.0)
    assert gates[220] > 0.0 or gates[221] > 0.0
    # the same step over a 0.45 deg span only reads 2.25 deg/s and leaves the rate factor at 1
    small = self.window(np.zeros(300), np.concatenate([np.zeros(200), np.full(100, 0.45)]))
    assert np.all(small[200:] > 0.9)

  def test_desired_window(self):
    # a one-frame model step of 0.15 m/s^2 (0.91 deg, 9.1 deg/s over the 0.1 s window) shuts the gate within 2 frames, for 10
    lat_accel = np.concatenate([np.full(200, -0.075), np.full(100, 0.075)])
    gates = self.window([angle_for(a, 30.0) for a in lat_accel], np.zeros(300))
    assert gates[199] > 0.9
    assert gates[201] == 0.0
    assert np.all(gates[200:210] == 0.0) and gates[210] > 0.0
    # a 0.09 m/s^2 step reads 5.4 deg/s: it trims the gate, it does not shut it
    trimmed = self.window([angle_for(a, 30.0) for a in np.concatenate([np.zeros(200), np.full(100, 0.09)])], np.zeros(300))
    assert 0.5 < trimmed[200:210].min() < trimmed[199] - 0.05

  def ramp(self, desired_rate, measured_rate, frames=100):
    # desired and measured angles ramping at the given rates (deg/s), both reaching 0 deg on the last frame. CS is a
    # plain namespace so the measured angle stays float64 (car.CarState rounds it to float32)
    calm = CalmSteering(DT_CTRL)
    params = log.VehicleParameters.new_message()
    for i in range(frames + 1):
      t = (i - frames) * DT_CTRL
      CS = SimpleNamespace(vEgo=30.0, steeringAngleDeg=measured_rate * t, steeringPressed=False)
      calm.update(0.0, CS, VM, params, curvature_for(desired_rate * t, 30.0), True)
    return calm

  def test_rate_scale(self):
    # rate() is the end-to-end difference over the window's span, (samples - 1) * dt: 0.1 s for the 11 desired samples,
    # 0.2 s for the 21 measured ones. Dividing by samples * dt would read 9 % and 5 % low and move the knee to ~5.5 deg/s
    assert len(self.ramp(0.0, 0.0).desired_angles) == 11 and len(self.ramp(0.0, 0.0).measured_angles) == 21
    np.testing.assert_allclose(self.ramp(6.0, 4.0).rate(), 6.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(self.ramp(4.0, 6.0).rate(), 6.0, rtol=0, atol=1e-9)
    # through update(), at 0 deg so only the rate factor is left: just under the 5 deg/s knee the gate is fully open,
    # just over it the gate is 1 - smoothstep(rate, 5.0, 8.0), from either window
    for desired_rate, measured_rate in ((4.9, 0.0), (0.0, 4.9)):
      assert self.ramp(desired_rate, measured_rate).last_gate == 1.0
    over = 1.0 - (lambda u: u * u * (3 - 2 * u))((5.1 - 5.0) / (8.0 - 5.0))
    for desired_rate, measured_rate in ((5.1, 0.0), (0.0, 5.1)):
      np.testing.assert_allclose(self.ramp(desired_rate, measured_rate).last_gate, over, rtol=0, atol=1e-9)

  def test_windows_fill_first(self):
    gates = self.window(np.zeros(60), np.zeros(60))
    assert np.all(gates[:20] == 0.0) and gates[20] > 0.0
    np.testing.assert_allclose(np.diff(gates[20:45]), DT_CTRL / 0.25)
    assert gates[44] == 1.0

  def test_roll(self):
    calm = CalmSteering(DT_CTRL)
    params = log.VehicleParameters.new_message(roll=0.03)
    for lat_accel in (-0.5, 0.0, 0.3):
      desired_curvature = lat_accel / 30.0**2
      # the wheel angle that makes the controller measure the same curvature at this roll
      angle = math.degrees(VM.get_steer_from_curvature(-desired_curvature, 30.0, 0.03))
      assert abs(-VM.calc_curvature(math.radians(angle), 30.0, 0.03) - desired_curvature) < 1e-12
      calm.update(0.0, car.CarState.new_message(vEgo=30.0, steeringAngleDeg=angle), VM, params, desired_curvature, True)
      assert abs(calm.desired_angles[-1] - calm.measured_angles[-1]) < 0.05
      no_roll = math.degrees(VM.get_steer_from_curvature(-desired_curvature, 30.0, 0.0))
      assert abs(no_roll - angle) > 0.05  # the roll matters at this size, so the check above has teeth

  def test_roll_through_controller(self):
    # the controller hands calm steering its desired curvature and roll: a consistent loop at roll 0.03 gates on the same
    # wheel angles as one at roll 0 (a sign or roll slip in the hook puts the desired angle about 1.6 deg off here)
    measured, _ = staircase(30.0, seconds=10.0)
    flat = drive(make_controller(CP, CI, True), 30.0, measured, [curvature_for(a, 30.0) for a in measured])
    rolled = drive(make_controller(CP, CI, True), 30.0, measured, [curvature_for(a, 30.0, 0.03) for a in measured], roll=0.03)
    assert np.mean(flat.gate[LEAD:]) > 0.95
    np.testing.assert_allclose(rolled.gate, flat.gate, rtol=0, atol=1e-6)

  def test_angle_offset(self):
    # the gate reads the angle less params.angleOffsetDeg, as the controller does: a consistent loop with a 1.5 deg offset
    # behaves as it does with none (to the float32 rounding of carState.steeringAngleDeg)
    measured, curvature = staircase(30.0, seconds=10.0)
    plain = drive(make_controller(CP, CI, True), 30.0, measured, curvature)
    offset = drive(make_controller(CP, CI, True), 30.0, [a + 1.5 for a in measured], curvature, angle_offset=1.5)
    assert np.mean(plain.gate[LEAD:]) > 0.95  # open, so a 1.5 deg error in the angle would shut most of it
    np.testing.assert_allclose(offset.gate, plain.gate, rtol=0, atol=1e-6)
    np.testing.assert_allclose(offset.out, plain.out, rtol=0, atol=1e-6)

  def test_sign(self):
    n = LEAD + 300
    stock, _ = both(30.0, np.full(n, quantize(angle_for(0.3, 30.0))), np.full(n, 0.3 / 30.0**2))
    assert np.abs(stock.error[-100:]).max() < 0.02
    measured, curvature = staircase(30.0, seconds=10.0)
    _, calm = both(30.0, measured, curvature)
    _, mirrored = both(30.0, [-a for a in measured], [-c for c in curvature])
    np.testing.assert_allclose(mirrored.out, -calm.out, rtol=0, atol=1e-12)
    assert np.array_equal(mirrored.gate, calm.gate)

  # (i) the filter
  def test_filter_step(self):
    calm = CalmSteering(DT_CTRL)
    calm.gate = lambda *a, **k: 1.0
    CS, params = car.CarState.new_message(vEgo=30.0), log.VehicleParameters.new_message()
    calm.update(0.0, CS, VM, params, 0.0, True)
    blended = [calm.update(0.2, CS, VM, params, 0.0, True) for _ in range(10)]
    alpha = 0.01 / (0.1 + 0.01)
    assert abs(blended[-1] - 0.2 * (1 - (1 - alpha)**10)) < 1e-4

  def test_non_finite(self):
    # inf and nan friction (on the first frames and with the gate open) and one nan steering angle
    calm, stock = CalmSteering(DT_CTRL), CalmSteering(DT_CTRL)
    stock.gate = lambda *a, **k: 0.0
    params = log.VehicleParameters.new_message()
    frictions = [math.inf, math.nan] + [0.05 * math.sin(i / 7) for i in range(1000)]
    frictions[600], frictions[700] = math.inf, math.nan
    angles = [quantize(0.3 * math.sin(i / 50)) for i in range(len(frictions))]
    angles[400] = math.nan
    outs = []
    for friction, angle in zip(frictions, angles, strict=True):
      CS = car.CarState.new_message(vEgo=30.0, steeringAngleDeg=angle)
      outs.append((calm.update(friction, CS, VM, params, 0.0, True), stock.update(friction, CS, VM, params, 0.0, True), calm.last_gate))
    calm_out, stock_out, gate = np.array(outs).T
    finite = np.isfinite(frictions)
    assert np.all(np.isfinite(calm_out[finite]))
    assert np.array_equal(calm_out[gate == 0.0], stock_out[gate == 0.0], equal_nan=True)
    assert np.array_equal(calm_out[~finite], stock_out[~finite], equal_nan=True)  # a bad friction value passes through as in stock
    # the nan angle shuts the gate while it is the newest and again when it is the oldest sample of the 0.2 s window
    assert gate[399] > 0.9 and gate[400] == 0.0 and gate[419] > 0.0 and gate[420] == 0.0
    # the filter took only the finite values, so calm still blends against it after every bad frame (one that took an inf or
    # a nan would stay non-finite, and calm would silently fall back to stock for the rest of the drive)
    reference = FirstOrderFilter(0.0, 0.1, DT_CTRL, initialized=False)
    warm = np.array([reference.update(f) if math.isfinite(f) else reference.x for f in frictions])[finite]
    friction, opened = np.array(frictions)[finite], gate[finite]
    np.testing.assert_allclose(calm_out[finite], friction + opened * (warm - friction), rtol=0, atol=1e-12)
    assert np.abs(calm_out[701:] - stock_out[701:])[gate[701:] > 0.9].max() > 0.001

  def test_non_finite_windows(self):
    # gate() and rate() directly: a non-finite angle passed in, or at either end of either window, shuts the gate
    params = log.VehicleParameters.new_message()
    for window in ('desired_angles', 'measured_angles'):
      for end in (0, -1):
        calm = CalmSteering(DT_CTRL)
        for _ in range(50):
          calm.update(0.0, car.CarState.new_message(vEgo=30.0), VM, params, 0.0, True)
        assert calm.rate() == 0.0 and calm.gate(30.0, 0.0, 0.0, False) == 1.0
        getattr(calm, window)[end] = math.nan
        assert math.isnan(calm.rate()) and calm.gate(30.0, 0.0, 0.0, False) == 0.0
    calm = CalmSteering(DT_CTRL)
    for _ in range(50):
      calm.update(0.0, car.CarState.new_message(vEgo=30.0), VM, params, 0.0, True)
    for bad in (math.nan, math.inf, -math.inf):
      assert calm.gate(30.0, bad, 0.0, False) == 0.0 and calm.gate(30.0, 0.0, bad, False) == 0.0
    # with the gate at 0.0 apply() hands back the friction value itself, not a blend of it: -0.0 stays -0.0
    assert math.copysign(1.0, calm.apply(-0.0, 0.0)) == -1.0

  # (j) a lane-centering-sized ramp on the closed loop
  @parameterized.expand(ENVELOPES)
  def test_lane_centering_ramp(self, *envelope):
    target = lambda t: float(np.clip((t - 2.0) / 0.25, 0.0, 1.0)) * 0.2  # noqa: E731
    (stock, stock_accel, _), (calm, calm_accel, _) = (closed_loop(g, target, 15.0, envelope=envelope) for g in (False, True))
    assert settling_frame(calm_accel, 200, 0.2) <= settling_frame(stock_accel, 200, 0.2) + 5
    assert np.ptp(calm[-500:]) <= np.ptp(stock[-500:]) + 0.004

  # (k) speeding up through the gate's speed ramp and braking back through it on a straight
  def test_speed_sweep(self):
    v = np.concatenate([np.full(LEAD, 20.0), np.arange(20.0, 32.0, 3.0 * DT_CTRL), np.full(100, 32.0), np.arange(32.0, 20.0, -8.0 * DT_CTRL)])
    t = np.arange(len(v)) * DT_CTRL
    angle = 0.3 * np.sin(2 * np.pi * t / 8.0)
    stock, calm = both(v, angle, np.zeros(len(v)))
    assert np.abs(np.diff(calm.out)).max() < 3 / 384
    assert calm.gate.max() > 0.9

  # (l) the replay tool on a synthetic rlog driven by the stock controller
  def test_replay_tool(self):
    msgs, generated = synthetic_log()
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, 'rlog.zst')
      save_log(path, msgs)
      logged_CP, frames = frames_from_log(LogReader(path))
    assert len(frames) == len(generated)
    stock, _ = replay(frames, False, logged_CP)
    np.testing.assert_allclose(stock, generated, rtol=0, atol=1e-9)
    summary = summarize(logged_CP, frames)
    highway = summary['buckets'][(100, 150)]
    assert highway['calm'][1] < 0.7 * highway['stock'][1]
    assert highway['gate_duty'] > 0.9
    assert summary['parity'][0] < 1e-6
    assert (0, 20) not in summary['buckets'] or summary['buckets'][(0, 20)]['minutes'] == 0
    np.testing.assert_allclose(highway['friction'], 0.12, rtol=1e-6)


def synthetic_log(seconds=60.0):
  """An rlog of the stock controller on the Palisade: 10 s at 15 m/s, up to 30 m/s, then a sensor staircase on a straight,
  lateral inactive for 2 s and the wheel touched for 1 s. Returns the messages and the torque the controller made."""
  ctl = make_controller(CP, CI, False)
  vm = VehicleModel(CP)
  msgs, generated = [], []
  mono = 0

  def add(which, valid=True, **fields):
    nonlocal mono
    mono += 1000
    msg = log.Event.new_message(logMonoTime=mono, valid=valid)
    msg.init(which)
    for k, v in fields.items():
      setattr(getattr(msg, which), k, v)
    msgs.append(msg)
    return getattr(msg, which)

  add('carParams')
  msgs[-1].carParams = CP
  steer_limited, last_counts, carOutput = False, 0, None
  limits = SimpleNamespace(STEER_MAX=384, STEER_DELTA_UP=3, STEER_DELTA_DOWN=7, STEER_DRIVER_ALLOWANCE=50, STEER_DRIVER_MULTIPLIER=2, STEER_DRIVER_FACTOR=1)
  torque, vp, delay = None, None, None
  for i in range(round(seconds / DT_CTRL)):
    t = i * DT_CTRL
    v = 15.0 if t < 10 else min(15.0 + 3.0 * (t - 10), 30.0)
    true_angle = 0.3 * math.sin(2 * math.pi * t / 8.0)
    cs = add('carState', vEgo=v, steeringAngleDeg=quantize(true_angle), steeringPressed=40.0 <= t < 41.0)
    if i % 5 == 0:
      vp = add('vehicleParameters', steerRatio=CP.steerRatio, stiffnessFactor=1.0, angleOffsetDeg=0.0, roll=0.0)
    if i % 25 == 0:
      delay = add('lateralDelay', lateralDelay=0.3)
      torque = add('lateralTorqueParameters', useParams=True, latAccelFactorFiltered=2.6, latAccelOffsetFiltered=0.01, frictionCoefficientFiltered=0.12)
    active = not (30.0 <= t < 32.0)
    desired_curvature = float(np.float32(curvature_for(true_angle, v)))
    vm.update_params(max(vp.stiffnessFactor, 0.1), max(vp.steerRatio, 0.1))
    ctl.update_torque_parameters(torque.latAccelFactorFiltered, torque.latAccelOffsetFiltered, torque.frictionCoefficientFiltered)
    out, _, lac_log = ctl.update(active, cs, vm, vp, steer_limited, desired_curvature, False, delay.lateralDelay)
    generated.append(out)
    controls = add('controlsState', desiredCurvature=desired_curvature, curvature=0.0)
    controls.lateralControlState.torqueState = lac_log
    cc = add('carControl', latActive=active)
    cc.actuators.torque = float(out)
    # controlsd compares this frame's command with the car's last output, for the next frame
    if active and carOutput is not None:
      steer_limited = abs(cc.actuators.torque - carOutput.actuatorsOutput.torque) > 1e-2
    last_counts = apply_driver_steer_torque_limits(round(out * 384), last_counts, 0.0, limits) if active else 0
    carOutput = add('carOutput')
    carOutput.actuatorsOutput.torque = last_counts / 384
  return [m.as_reader() for m in msgs], np.array(generated)

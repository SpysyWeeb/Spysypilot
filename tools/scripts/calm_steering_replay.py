#!/usr/bin/env python3
"""Replay a route's logged inputs through the torque controller twice, stock (calm steering gate forced to 0) and
with calm steering, and compare the steering torque each produces on highway-speed straights.

  tools/scripts/calm_steering_replay.py <route or rlog path> [--min-run-s 5]

The replay is OPEN LOOP on the logged measured steering angle: the car's response to the recorded torque is what
was measured, so this shows how much less of that input's noise reaches the torque. It cannot show a closed-loop limit
cycle being suppressed (the wheel would have moved differently); the closed-loop tests in
openpilot/selfdrive/controls/tests/test_calm_steering.py are the check for that.

PARITY compares the stock replay with the logged torqueState.output. It is ~0 for a route driven on this controller.
Routes from other lateral controllers (BLaT branches, angle or PID cars) will not match, and the comparison means
little there; a WARNING is printed when the median difference exceeds 0.01.
"""
import argparse
import math
from dataclasses import dataclass

import numpy as np

from openpilot.cereal import log
from opendbc.car.car_helpers import interfaces
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque

# modeld.py's LAT_SMOOTH_SECONDS, which controlsd adds to lateralDelay; copied rather than imported, so this needs
# neither modeld nor controlsd
LAT_SMOOTH_SECONDS = 0.0
SPEED_BUCKETS_KPH = ((0, 20), (20, 60), (60, 100), (100, 150))
STRAIGHT_DEG = 3.0
PARITY_WARN = 0.01


def chatter(out, hyst=0.008, fps=100):
  """(max_step, mean_step, reversals_per_min) of a torque sequence. A reversal is a pivot of a hysteresis zig-zag:
  the output turning back by more than hyst from its running extreme. hyst 0.008 is half the 0.0159 normalized torque
  a 0.1 deg steering angle step makes through P and friction at 30 m/s on the Palisade."""
  out = np.asarray(out, dtype=float)
  if len(out) < 2:
    return 0.0, 0.0, 0.0
  steps = np.abs(np.diff(out))
  pivots, direction = 0, 0
  lo = hi = extreme = out[0]
  for x in out[1:]:
    if direction == 0:
      lo, hi = min(lo, x), max(hi, x)
      if x - lo > hyst:
        direction, extreme = 1, x
      elif hi - x > hyst:
        direction, extreme = -1, x
    elif direction * (x - extreme) > 0:
      extreme = x
    elif direction * (extreme - x) > hyst:
      pivots += 1
      direction, extreme = -direction, x
  minutes = len(out) / fps / 60
  return float(steps.max()), float(steps.mean()), pivots / minutes


def make_controller(CP, CI, gate_open: bool) -> LatControlTorque:
  """The torque controller; with gate_open False its calm steering gate is forced to 0.0, which is stock."""
  ctl = LatControlTorque(CP.as_reader() if hasattr(CP, 'as_reader') else CP, CI, DT_CTRL)
  if not gate_open:
    ctl.calm.gate = lambda *a, **k: 0.0
  return ctl


@dataclass
class Frame:
  CS: object                      # carState
  params: object                  # vehicleParameters
  desired_curvature: float        # controlsState.desiredCurvature: after clip_curvature, what the controller got
  lat_delay: float
  torque_params: tuple | None     # (latAccelFactor, latAccelOffset, friction) when lateralTorqueParameters is in use
  active: bool                    # carControl.latActive
  steer_limited_by_safety: bool
  logged_output: float            # controlsState.lateralControlState.torqueState.output


def frames_from_log(msgs) -> tuple[object, list[Frame]]:
  """carParams and one Frame per controlsState, with the inputs controlsd gave the controller on that frame."""
  CP = None
  latest: dict[str, object] = {'vehicleParameters': log.VehicleParameters.new_message(), 'lateralDelay': None,
                               'lateralTorqueParameters': None, 'carState': None, 'carOutput': None}
  torque_valid = False
  frames: list[Frame] = []
  pending = None
  steer_limited = False
  for msg in sorted(msgs, key=lambda m: m.logMonoTime):
    which = msg.which()
    if which == 'carParams':
      CP = msg.carParams
    elif which in latest:
      latest[which] = getattr(msg, which)
      torque_valid = msg.valid if which == 'lateralTorqueParameters' else torque_valid
    elif which == 'controlsState' and latest['carState'] is not None:
      cs = msg.controlsState
      tp = latest['lateralTorqueParameters']
      use_torque = tp is not None and torque_valid and tp.useParams
      delay = latest['lateralDelay']
      pending = Frame(latest['carState'], latest['vehicleParameters'], cs.desiredCurvature,
                      (delay.lateralDelay if delay is not None else 0.0) + LAT_SMOOTH_SECONDS,
                      (tp.latAccelFactorFiltered, tp.latAccelOffsetFiltered, tp.frictionCoefficientFiltered) if use_torque else None,
                      False, steer_limited,
                      cs.lateralControlState.torqueState.output if cs.lateralControlState.which() == 'torqueState' else math.nan)
    elif which == 'carControl' and pending is not None:
      cc = msg.carControl
      pending.active = cc.latActive
      frames.append(pending)
      pending = None
      # controlsd sets this after publishing, for the next frame (its own check uses selfdriveState.active)
      co = latest['carOutput']
      if cc.latActive and co is not None:
        steer_limited = abs(cc.actuators.torque - co.actuatorsOutput.torque) > 1e-2
  return CP, frames


def replay(frames: list[Frame], gate_open: bool, CP) -> tuple[np.ndarray, np.ndarray]:
  """Torque output and calm gate per frame; update() runs on every frame, active or not, like controlsd."""
  CI = interfaces[CP.carFingerprint](CP)
  VM = VehicleModel(CP)
  ctl = make_controller(CP, CI, gate_open)
  outputs, gates = np.zeros(len(frames)), np.zeros(len(frames))
  for i, f in enumerate(frames):
    VM.update_params(max(f.params.stiffnessFactor, 0.1), max(f.params.steerRatio, 0.1))
    if f.torque_params is not None:
      ctl.update_torque_parameters(*f.torque_params)
    if not f.active:
      ctl.reset()
    outputs[i] = ctl.update(f.active, f.CS, VM, f.params, f.steer_limited_by_safety, f.desired_curvature, False, f.lat_delay)[0]
    gates[i] = ctl.calm.last_gate
  return outputs, gates


def straight_runs(frames: list[Frame], min_run_s: float) -> list[slice]:
  """Contiguous stretches of at least min_run_s: active, hands off, |measured angle| under 3 deg."""
  ok = [f.active and not f.CS.steeringPressed and abs(f.CS.steeringAngleDeg - f.params.angleOffsetDeg) < STRAIGHT_DEG for f in frames]
  runs, start = [], None
  for i, good in enumerate([*ok, False]):
    if good and start is None:
      start = i
    elif not good and start is not None:
      if (i - start) * DT_CTRL >= min_run_s:
        runs.append(slice(start, i))
      start = None
  return runs


def summarize(CP, frames: list[Frame], min_run_s: float = 5.0) -> dict:
  stock, _ = replay(frames, False, CP)
  calm, gates = replay(frames, True, CP)
  tp = CP.lateralTuning.torque
  summary: dict = {'buckets': {}}
  runs = straight_runs(frames, min_run_s)
  for lo, hi in SPEED_BUCKETS_KPH:
    sel = [r for r in runs if lo <= 3.6 * np.mean([f.CS.vEgo for f in frames[r]]) < hi]
    if not sel:
      continue
    weights = np.array([r.stop - r.start for r in sel], dtype=float)
    stock_c = np.array([chatter(stock[r]) for r in sel])
    calm_c = np.array([chatter(calm[r]) for r in sel])
    used = [f.torque_params for r in sel for f in frames[r] if f.torque_params is not None]
    summary['buckets'][(lo, hi)] = {
      'minutes': weights.sum() * DT_CTRL / 60,
      'stock': (stock_c[:, 0].max(), *np.average(stock_c[:, 1:], axis=0, weights=weights)),
      'calm': (calm_c[:, 0].max(), *np.average(calm_c[:, 1:], axis=0, weights=weights)),
      'gate_duty': float(np.mean(np.concatenate([gates[r] for r in sel]))),
      'friction': float(np.median([u[2] for u in used])) if used else float(tp.friction),
      'lat_accel_factor': float(np.median([u[0] for u in used])) if used else float(tp.latAccelFactor),
    }
  active = np.array([f.active for f in frames], dtype=bool)
  logged = np.array([f.logged_output for f in frames])
  diff = np.abs(stock - logged)[active & np.isfinite(logged)]
  summary['parity'] = (float(np.median(diff)), float(np.percentile(diff, 95))) if len(diff) else (math.nan, math.nan)
  summary['stock'], summary['calm'], summary['gates'] = stock, calm, gates
  return summary


def print_summary(summary: dict) -> None:
  print('straight runs (active, hands off, |angle| < 3 deg); chatter = max step, mean step, reversals/min')
  for (lo, hi), b in summary['buckets'].items():
    print(f"{lo:3d}-{hi:3d} km/h  {b['minutes']:6.1f} min  gate duty {b['gate_duty']:.2f}  " +
          f"friction {b['friction']:.4f}  latAccelFactor {b['lat_accel_factor']:.4f}")
    for name in ('stock', 'calm'):
      max_step, mean_step, reversals = b[name]
      print(f'    {name:5s}  max step {max_step:.4f}  mean step {mean_step:.5f}  reversals {reversals:6.1f}/min')
  median, p95 = summary['parity']
  print(f'PARITY stock replay vs logged torqueState.output: median {median:.2e}  p95 {p95:.2e}')
  if not median <= PARITY_WARN:
    print('WARNING: the stock replay does not match the logged output; this route was likely driven by another controller')


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('route_or_path', help='route name or segment range, or a local rlog path (local files need no network)')
  parser.add_argument('--min-run-s', type=float, default=5.0, help='shortest straight run to count, seconds')
  args = parser.parse_args()

  from openpilot.tools.lib.logreader import LogReader
  CP, frames = frames_from_log(LogReader(args.route_or_path))
  if CP is None or CP.lateralTuning.which() != 'torque':
    raise SystemExit('needs carParams of a torque-controlled car in the log')
  print_summary(summarize(CP, frames, args.min_run_s))


if __name__ == '__main__':
  main()

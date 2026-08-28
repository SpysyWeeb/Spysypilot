#!/usr/bin/env python3
"""Evaluate BLaTv2 rack dynamics on immutable route-evidence artifacts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.selfdrive.controls.lib.blatv2.contextual_dynamics_study import (
  DIRECTIONS,
  PHASES,
  SPEED_NODES_MPS,
  DynamicsCandidate,
  WindowBatch,
  maneuver_context,
  nearest_speed_node,
  squared_angle_error,
)
from openpilot.selfdrive.controls.lib.blatv2.route_evidence import (
  RouteEvidenceStreamReader,
)


GAIN_GRID = (800.0, 1600.0, 2400.0, 3200.0, 4000.0, 5200.0, 6800.0)
DAMPING_GRID = (4.0, 6.0, 8.0, 10.0, 14.0, 20.0)
SEED = DynamicsCandidate(4000.0, 10.0)
FIELD_CANDIDATE_SCHEDULE = {
  0.0: "4000/10",
  5.0: "4000/10",
  10.0: "3200/14",
  15.0: "3200/14",
  20.0: "3200/14",
  30.0: "3200/14",
}
TORQUE_PER_LATERAL_ACCEL = 0.39335
STATIC_FRICTION_TORQUE = 0.13
KINETIC_FRICTION_TORQUE = 0.13
RACK_RATE_RESOLUTION_DEG_S = 4.0
WINDOW_FRAMES = 25
WINDOW_STRIDE = 10
WINDOWS_PER_STRATUM = 16
MAX_SAMPLE_GAP_S = 0.016
MINIMUM_SPEED_MPS = 2.0
MINIMUM_TORQUE_EXCURSION = 0.08
MINIMUM_ANGLE_TRAVEL_DEG = 1.0


def _load_partition(path: Path) -> tuple[str, tuple[dict[str, object], ...]]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if type(payload) is not dict or payload.get("schemaVersion") != 2:
    raise ValueError("partition receipt must use schema 2")
  partition = payload.get("partition")
  if type(partition) is not dict or partition.get("schemaVersion") != 1:
    raise ValueError("partition payload must use schema 1")
  assignments = partition.get("assignments")
  if type(assignments) is not list or not assignments:
    raise ValueError("partition has no route assignments")
  output: list[dict[str, object]] = []
  for assignment in assignments:
    if (
      type(assignment) is not dict
      or assignment.get("split") not in ("training", "validation", "test")
      or type(assignment.get("routeId")) is not str
      or type(assignment.get("artifactSha256")) is not str
    ):
      raise ValueError("partition route assignment is malformed")
    output.append(dict(assignment))
  return str(payload["partitionSha256"]), tuple(output)


def _eligible_arrays(path: Path) -> dict[str, np.ndarray]:
  with RouteEvidenceStreamReader(path) as reader:
    frames = list(reader.iter_physical_frames())
    controls = list(reader.iter_control_witnesses())
  count = len(frames)
  measured_curvature = np.zeros(count, dtype=np.float64)
  desired_curvature = np.zeros(count, dtype=np.float64)
  control_valid = np.zeros(count, dtype=np.bool_)
  for control in controls:
    index = control.physical_record_index
    if not 0 <= index < count:
      continue
    measured_curvature[index] = control.measured_curvature
    desired_curvature[index] = control.desired_curvature
    control_valid[index] = (
      control.inputs_valid
      and control.lateral_active
      and not control.driver_intervening
      and not control.steer_fault
      and not control.gap_from_previous
      and not control.race_unresolved
      and control.car_control_paired
      and control.torque_output_can_valid
    )

  response_time_s = np.asarray(
    [frame.response_mono_ns * 1e-9 for frame in frames], dtype=np.float64,
  )
  valid = control_valid & np.asarray([
    frame.inputs_valid
    and frame.lateral_active
    and not frame.steering_pressed
    and not frame.steer_fault_temporary
    and not frame.steer_fault_permanent
    and frame.can_valid
    and not frame.can_timeout
    and frame.live_parameters_valid
    and frame.angle_offset_valid
    and frame.steer_ratio_valid
    and frame.stiffness_factor_valid
    for frame in frames
  ], dtype=np.bool_)
  return {
    "time": response_time_s,
    "angle": np.asarray([frame.steering_angle_deg for frame in frames]),
    "rate": np.asarray([frame.steering_rate_deg_s for frame in frames]),
    "torque": np.asarray([frame.applied_torque for frame in frames]),
    "speed": np.asarray([frame.speed_mps for frame in frames]),
    "roll": np.asarray([frame.roll_rad for frame in frames]),
    "measured_curvature": measured_curvature,
    "desired_curvature": desired_curvature,
    "valid": valid,
  }


def _route_windows(path: Path) -> WindowBatch | None:
  values = _eligible_arrays(path)
  count = len(values["time"])
  if count <= WINDOW_FRAMES:
    return None
  sample_dt = np.diff(values["time"])
  lateral_accel = values["desired_curvature"] * np.square(values["speed"])
  measured_lateral_accel = (
    values["measured_curvature"] * np.square(values["speed"])
    - values["roll"] * ACCELERATION_DUE_TO_GRAVITY
  )
  aligning = -measured_lateral_accel * TORQUE_PER_LATERAL_ACCEL
  # Reversing/noisy negative speeds are excluded below. Clamping only for the
  # total nearest-node projection keeps that helper's physical domain strict.
  speed_nodes = nearest_speed_node(np.maximum(values["speed"], 0.0))

  ranked: dict[tuple[float, str, str], list[tuple[tuple[float, float, int], int]]] = defaultdict(list)
  for start in range(0, count - WINDOW_FRAMES, WINDOW_STRIDE):
    stop = start + WINDOW_FRAMES
    if (
      values["speed"][start:stop + 1].min() < MINIMUM_SPEED_MPS
      or not values["valid"][start:stop + 1].all()
      or (sample_dt[start:stop] <= 0.0).any()
      or sample_dt[start:stop].max() > MAX_SAMPLE_GAP_S
    ):
      continue
    context = maneuver_context(lateral_accel[start], lateral_accel[stop])
    if context is None:
      continue
    phase, direction = context
    torque_excursion = float(np.ptp(values["torque"][start:stop + 1]))
    angle_travel = float(np.ptp(values["angle"][start:stop + 1]))
    if (
      torque_excursion < MINIMUM_TORQUE_EXCURSION
      and angle_travel < MINIMUM_ANGLE_TRAVEL_DEG
    ):
      continue
    key = (float(speed_nodes[start]), phase, direction)
    ranked[key].append(((torque_excursion, angle_travel, -start), start))

  selected: list[int] = []
  for key in sorted(ranked):
    candidates = sorted(ranked[key], reverse=True)[:WINDOWS_PER_STRATUM]
    selected.extend(start for _, start in candidates)
  selected.sort()
  if not selected:
    return None

  starts = np.asarray(selected, dtype=np.int64)
  offsets = np.arange(WINDOW_FRAMES, dtype=np.int64)
  sample_indices = starts[:, None] + offsets[None, :]
  contexts = tuple(
    maneuver_context(lateral_accel[start], lateral_accel[start + WINDOW_FRAMES])
    for start in selected
  )
  if any(context is None for context in contexts):
    raise AssertionError("selected route context disappeared")
  return WindowBatch(
    initial_angle_deg=values["angle"][starts],
    initial_rate_deg_s=values["rate"][starts],
    applied_torque=values["torque"][sample_indices],
    aligning_torque=aligning[sample_indices],
    dt_s=sample_dt[sample_indices],
    measured_final_angle_deg=values["angle"][starts + WINDOW_FRAMES],
    speed_node_mps=speed_nodes[starts],
    phase=tuple(context[0] for context in contexts if context is not None),
    direction=tuple(context[1] for context in contexts if context is not None),
  )


def _evaluate(windows: WindowBatch, candidate: DynamicsCandidate) -> dict[str, object]:
  error = squared_angle_error(
    windows,
    candidate,
    static_friction_torque=STATIC_FRICTION_TORQUE,
    kinetic_friction_torque=KINETIC_FRICTION_TORQUE,
    rack_rate_resolution_deg_s=RACK_RATE_RESOLUTION_DEG_S,
  )
  strata: dict[str, object] = {}
  for speed_node in SPEED_NODES_MPS:
    for phase in PHASES:
      for direction in DIRECTIONS:
        mask = np.asarray([
          observed_speed == speed_node
          and observed_phase == phase
          and observed_direction == direction
          for observed_speed, observed_phase, observed_direction in zip(
            windows.speed_node_mps,
            windows.phase,
            windows.direction,
            strict=True,
          )
        ])
        if mask.any():
          strata[f"{speed_node:g}.{phase}.{direction}"] = {
            "count": int(mask.sum()),
            "rmseDeg": math.sqrt(float(error[mask].mean())),
          }
  return {
    "count": windows.count,
    "rmseDeg": math.sqrt(float(error.mean())),
    "strata": strata,
  }


def _candidate_grid() -> tuple[DynamicsCandidate, ...]:
  return tuple(
    DynamicsCandidate(gain, damping)
    for gain in GAIN_GRID
    for damping in DAMPING_GRID
  )


def _split_summary(
  routes: list[dict[str, object]],
  selected_key: str,
  seed_key: str,
) -> dict[str, object]:
  selected = [route["evaluations"][selected_key]["rmseDeg"] for route in routes]
  seed = [route["evaluations"][seed_key]["rmseDeg"] for route in routes]
  stratum_keys = sorted({
    stratum
    for route in routes
    for stratum in route["evaluations"][selected_key]["strata"]
  })
  strata: dict[str, object] = {}
  for stratum in stratum_keys:
    paired = [
      (
        route["evaluations"][selected_key]["strata"][stratum],
        route["evaluations"][seed_key]["strata"][stratum],
      )
      for route in routes
      if stratum in route["evaluations"][selected_key]["strata"]
    ]
    selected_values = [value[0]["rmseDeg"] for value in paired]
    seed_values = [value[1]["rmseDeg"] for value in paired]
    strata[stratum] = {
      "meanSeedRmseDeg": float(np.mean(seed_values)),
      "meanSelectedRmseDeg": float(np.mean(selected_values)),
      "pairedImprovedRouteCount": sum(
        a < b for a, b in zip(selected_values, seed_values, strict=True)
      ),
      "pairedRegressedRouteCount": sum(
        a > b for a, b in zip(selected_values, seed_values, strict=True)
      ),
      "routeCount": len(paired),
      "windowCount": sum(value[0]["count"] for value in paired),
    }
  return {
    "meanSelectedRmseDeg": None if not selected else float(np.mean(selected)),
    "meanSeedRmseDeg": None if not seed else float(np.mean(seed)),
    "pairedImprovedRouteCount": sum(
      a < b for a, b in zip(selected, seed, strict=True)
    ),
    "pairedRegressedRouteCount": sum(
      a > b for a, b in zip(selected, seed, strict=True)
    ),
    "routeCount": len(routes),
    "strata": strata,
    "windowCount": sum(int(route["windowCount"]) for route in routes),
  }


def _speed_metric(evaluation: dict[str, object], speed_node: float) -> tuple[float, int] | None:
  prefix = f"{speed_node:g}."
  strata = [
    value
    for key, value in evaluation["strata"].items()
    if key.startswith(prefix)
  ]
  count = sum(value["count"] for value in strata)
  if count == 0:
    return None
  mse = sum(value["rmseDeg"] ** 2 * value["count"] for value in strata) / count
  return math.sqrt(mse), count


def _select_speed_schedule(
  training: list[dict[str, object]],
  candidate_keys: tuple[str, ...],
) -> dict[float, str]:
  schedule: dict[float, str] = {}
  for speed_node in SPEED_NODES_MPS:
    scores: dict[str, float] = {}
    for candidate_key in candidate_keys:
      route_values = [
        metric[0]
        for route in training
        if (
          metric := _speed_metric(
            route["evaluations"][candidate_key], speed_node,
          )
        ) is not None
      ]
      if route_values:
        scores[candidate_key] = float(np.mean(route_values))
    if not scores:
      raise ValueError(f"training routes have no {speed_node:g} m/s windows")
    schedule[speed_node] = min(scores, key=lambda key: (scores[key], key))
  return schedule


def _scheduled_evaluation(
  route: dict[str, object],
  schedule: dict[float, str],
) -> dict[str, object]:
  selected_strata: dict[str, object] = {}
  for speed_node, candidate_key in schedule.items():
    prefix = f"{speed_node:g}."
    selected_strata.update({
      key: value
      for key, value in route["evaluations"][candidate_key]["strata"].items()
      if key.startswith(prefix)
    })
  count = sum(value["count"] for value in selected_strata.values())
  if count == 0:
    return {"count": 0, "rmseDeg": None, "strata": {}}
  mse = sum(
    value["rmseDeg"] ** 2 * value["count"]
    for value in selected_strata.values()
  ) / count
  return {
    "count": count,
    "rmseDeg": math.sqrt(mse),
    "strata": selected_strata,
  }


def _schedule_split_summary(
  routes: list[dict[str, object]],
  schedule: dict[float, str],
  seed_key: str,
) -> dict[str, object]:
  scheduled_routes: list[dict[str, object]] = []
  for route in routes:
    scheduled = _scheduled_evaluation(route, schedule)
    if scheduled["count"] == 0:
      continue
    scheduled_routes.append({
      "evaluations": {
        "schedule": scheduled,
        "seed": route["evaluations"][seed_key],
      },
      "windowCount": scheduled["count"],
    })
  return _split_summary(scheduled_routes, "schedule", "seed")


def run(partition_path: Path, evidence_root: Path) -> dict[str, object]:
  partition_sha256, assignments = _load_partition(partition_path)
  candidates = _candidate_grid()
  route_results: list[dict[str, object]] = []
  for assignment in assignments:
    artifact_sha256 = str(assignment["artifactSha256"])
    object_path = evidence_root / "objects" / f"{artifact_sha256}.route-evidence"
    windows = _route_windows(object_path)
    if windows is None:
      route_results.append({
        "artifactSha256": artifact_sha256,
        "routeId": assignment["routeId"],
        "split": assignment["split"],
        "windowCount": 0,
      })
      continue
    evaluations = {
      f"{candidate.rack_gain_deg_s2_per_torque:g}/{candidate.rack_damping_per_s:g}": _evaluate(windows, candidate)
      for candidate in candidates
    }
    route_results.append({
      "artifactSha256": artifact_sha256,
      "evaluations": evaluations,
      "routeId": assignment["routeId"],
      "split": assignment["split"],
      "windowCount": windows.count,
    })

  training = [route for route in route_results if route["split"] == "training" and route["windowCount"]]
  if len(training) < 2:
    raise ValueError("context study requires at least two training routes")
  candidate_keys = tuple(training[0]["evaluations"])
  training_scores = {
    key: float(np.mean([route["evaluations"][key]["rmseDeg"] for route in training]))
    for key in candidate_keys
  }
  selected_key = min(candidate_keys, key=lambda key: (training_scores[key], key))
  seed_key = f"{SEED.rack_gain_deg_s2_per_torque:g}/{SEED.rack_damping_per_s:g}"
  selected_schedule = _select_speed_schedule(training, candidate_keys)
  if any(key not in candidate_keys for key in FIELD_CANDIDATE_SCHEDULE.values()):
    raise AssertionError("field candidate is outside the diagnostic grid")

  splits: dict[str, object] = {}
  for split in ("training", "validation", "test"):
    routes = [route for route in route_results if route["split"] == split and route["windowCount"]]
    splits[split] = _split_summary(routes, selected_key, seed_key)
  schedule_splits = {
    split: _schedule_split_summary(
      [
        route for route in route_results
        if route["split"] == split and route["windowCount"]
      ],
      selected_schedule,
      seed_key,
    )
    for split in ("training", "validation", "test")
  }
  field_candidate_splits = {
    split: _schedule_split_summary(
      [
        route for route in route_results
        if route["split"] == split and route["windowCount"]
      ],
      FIELD_CANDIDATE_SCHEDULE,
      seed_key,
    )
    for split in ("training", "validation", "test")
  }
  return {
    "candidateGrid": {
      "rackDampingPerS": list(DAMPING_GRID),
      "rackGainDegS2PerTorque": list(GAIN_GRID),
    },
    "diagnosticOnly": True,
    "fieldCandidateSpeedSchedule": {
      f"{speed_node:g}": candidate_key
      for speed_node, candidate_key in FIELD_CANDIDATE_SCHEDULE.items()
    },
    "fieldCandidateSplits": field_candidate_splits,
    "fieldCandidateUsesInspectedHeldOutStrata": True,
    "partitionSha256": partition_sha256,
    "routes": route_results,
    "schemaVersion": 1,
    "seed": {
      "rackDampingPerS": SEED.rack_damping_per_s,
      "rackGainDegS2PerTorque": SEED.rack_gain_deg_s2_per_torque,
    },
    "selected": {
      "key": selected_key,
      "rackDampingPerS": float(selected_key.split("/")[1]),
      "rackGainDegS2PerTorque": float(selected_key.split("/")[0]),
    },
    "selectedSpeedSchedule": {
      f"{speed_node:g}": {
        "key": candidate_key,
        "rackDampingPerS": float(candidate_key.split("/")[1]),
        "rackGainDegS2PerTorque": float(candidate_key.split("/")[0]),
      }
      for speed_node, candidate_key in selected_schedule.items()
    },
    "speedScheduleSplits": schedule_splits,
    "splits": splits,
  }


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("partition_receipt", type=Path)
  parser.add_argument("evidence_root", type=Path)
  arguments = parser.parse_args()
  result = run(arguments.partition_receipt.resolve(), arguments.evidence_root.resolve())
  print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

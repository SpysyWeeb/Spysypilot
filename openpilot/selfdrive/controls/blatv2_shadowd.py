#!/usr/bin/env python3
"""Telemetry-only BLaTv2 plant/reference shadow.

This process owns no actuator publisher. Its sole output is ``blatV2Shadow``.
"""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any

import openpilot.cereal.messaging as messaging
from opendbc.car.hyundai.values import CarControllerParams
from opendbc.car.structs import car
from openpilot.selfdrive.controls.lib.blatv2.controller import ControllerParams
from openpilot.selfdrive.controls.lib.blatv2.plant import PlantParams
from openpilot.selfdrive.controls.lib.blatv2.shadow import ShadowCore, ShadowResult

SHADOW_VERSION = 4
PUBLISHED_SERVICES = ("blatV2Shadow",)
SUBSCRIBED_SERVICES = (
  "modelV2",
  "carState",
  "carControl",
  "carOutput",
  "controlsState",
  "liveParameters",
  "liveDelay",
)
SEED_PATH = Path(__file__).resolve().parent / "lib" / "blatv2" / "plant_seed_params.json"
CONTROLLER_SEED_PATH = Path(__file__).resolve().parent / "lib" / "blatv2" / "controller_seed_params.json"


def populate_shadow_message(
  message: Any,
  shadow: Any,
  result: ShadowResult,
  *,
  log_mono_time_ns: int,
  message_valid: bool,
  compute_seconds: float,
  shared_compute_seconds: float,
  mpc_compute_seconds: float,
  fallback_compute_seconds: float,
) -> None:
  """Normalize the numerical core at the Cap'n Proto type boundary.

  Solver fields may be numpy scalar types. Cap'n Proto accepts only native
  Python scalar values, so every published field is converted here rather
  than relying on the type produced by any particular solver path.
  """
  message.logMonoTime = int(log_mono_time_ns)
  message.valid = bool(message_valid)
  shadow.shadowVersion = int(SHADOW_VERSION)
  shadow.valid = bool(result.valid)
  shadow.referenceCurvature = float(result.reference_curvature)
  shadow.torqueDemand = float(result.torque_demand)
  shadow.feasibleTorque = float(result.feasible_torque)
  shadow.plantResidual = float(result.plant_residual)
  shadow.scalarPlanDisagreement = float(result.scalar_plan_disagreement)
  shadow.horizon = float(result.horizon)
  shadow.computeTimeSeconds = float(compute_seconds)
  shadow.sharedComputeTimeSeconds = float(shared_compute_seconds)
  shadow.vEgo = float(result.v_ego)
  shadow.aligningTorque = float(result.aligning_torque)
  shadow.alignInputsValid = bool(result.align_inputs_valid)
  shadow.disturbanceEstimate = float(result.disturbance_estimate)
  shadow.observerStatus = int(result.observer_status)
  shadow.observerUnconstrainedUpdate = float(
    result.observer_unconstrained_update
  )
  shadow.mpcCommandTorque = float(result.mpc_command_torque)
  shadow.mpcStatus = int(result.mpc_status)
  shadow.mpcCandidateCount = int(result.mpc_candidate_count)
  shadow.mpcOptimalityResidual = float(result.mpc_optimality_residual)
  shadow.mpcComputeTimeSeconds = float(mpc_compute_seconds)
  shadow.fallbackCommandTorque = float(result.fallback_command_torque)
  shadow.fallbackStatus = int(result.fallback_status)
  shadow.fallbackCandidateCount = int(result.fallback_candidate_count)
  shadow.fallbackOptimalityResidual = float(
    result.fallback_optimality_residual
  )
  shadow.fallbackComputeTimeSeconds = float(fallback_compute_seconds)


class BlatV2Shadow:
  def __init__(self) -> None:
    from openpilot.common.params import Params

    # This is a structural actuation boundary, not merely a convention.
    assert "carControl" not in PUBLISHED_SERVICES
    assert "sendcan" not in PUBLISHED_SERVICES

    params = Params()
    cp_bytes = params.get("CarParams", block=True)
    self.CP = messaging.log_from_bytes(cp_bytes, car.CarParams)
    self.seed_params = PlantParams.from_seed_file(SEED_PATH, CarControllerParams(self.CP))
    self.controller_params = ControllerParams.from_seed_file(CONTROLLER_SEED_PATH)
    self.torque_params = self.CP.lateralTuning.torque
    self.core = ShadowCore(self.seed_params, self.torque_params, self.CP, self.controller_params)

    self.subscribed_services = list(SUBSCRIBED_SERVICES)
    self.sm = messaging.SubMaster(self.subscribed_services, poll="controlsState")
    self.pm = messaging.PubMaster(list(PUBLISHED_SERVICES))
    self.message = messaging.new_message("blatV2Shadow")
    self.shadow = self.message.blatV2Shadow

  def _begin_frame(self) -> ShadowResult:
    return self.core.begin_frame(
      self.sm["modelV2"],
      self.sm["carState"],
      self.sm["carControl"],
      self.sm["carOutput"],
      self.sm["liveParameters"],
      self.sm.valid["liveParameters"],
      float(self.sm["liveDelay"].lateralDelay),
      self.sm.valid["liveDelay"],
      self.sm.valid["modelV2"],
    )

  def step(self) -> None:
    self.sm.update()
    if not self.sm.updated["controlsState"]:
      return

    started_ns = time.perf_counter_ns()
    shared_compute_seconds = 0.0
    mpc_compute_seconds = 0.0
    fallback_compute_seconds = 0.0
    phase = "shared"
    phase_started_ns = time.perf_counter_ns()
    try:
      result = self._begin_frame()
      phase_finished_ns = time.perf_counter_ns()
      shared_compute_seconds = (
        phase_finished_ns - phase_started_ns
      ) * 1e-9

      phase = "mpc"
      phase_started_ns = phase_finished_ns
      self.core.compute_mpc()
      phase_finished_ns = time.perf_counter_ns()
      mpc_compute_seconds = (
        phase_finished_ns - phase_started_ns
      ) * 1e-9

      phase = "fallback"
      phase_started_ns = phase_finished_ns
      self.core.compute_fallback()
      phase_finished_ns = time.perf_counter_ns()
      fallback_compute_seconds = (
        phase_finished_ns - phase_started_ns
      ) * 1e-9
      phase = "finalize"
      result = self.core.end_frame()
    except (RuntimeError, ValueError, OverflowError):
      failed_phase_seconds = (
        time.perf_counter_ns() - phase_started_ns
      ) * 1e-9
      if phase == "shared":
        shared_compute_seconds = failed_phase_seconds
      elif phase == "mpc":
        mpc_compute_seconds = failed_phase_seconds
      elif phase == "fallback":
        fallback_compute_seconds = failed_phase_seconds
      result = self.core.invalid_result()
    compute_seconds = (time.perf_counter_ns() - started_ns) * 1e-9
    assert (
      math.isfinite(compute_seconds)
      and math.isfinite(shared_compute_seconds)
      and math.isfinite(mpc_compute_seconds)
      and math.isfinite(fallback_compute_seconds)
      and compute_seconds >= 0.0
      and shared_compute_seconds >= 0.0
      and mpc_compute_seconds >= 0.0
      and fallback_compute_seconds >= 0.0
    )

    populate_shadow_message(
      self.message,
      self.shadow,
      result,
      log_mono_time_ns=int(time.monotonic() * 1e9),
      message_valid=bool(
        result.valid and self.sm.all_checks(self.subscribed_services)
      ),
      compute_seconds=compute_seconds,
      shared_compute_seconds=shared_compute_seconds,
      mpc_compute_seconds=mpc_compute_seconds,
      fallback_compute_seconds=fallback_compute_seconds,
    )
    self.pm.send("blatV2Shadow", self.message)

  def run(self) -> None:
    while True:
      self.step()


def main() -> None:
  BlatV2Shadow().run()


if __name__ == "__main__":
  main()

#!/usr/bin/env python3
"""Telemetry-only BLaTv2 plant/reference shadow.

This process owns no actuator publisher. Its sole output is ``blatV2Shadow``.
"""

from __future__ import annotations

import math
from pathlib import Path
import time

import openpilot.cereal.messaging as messaging
from opendbc.car.hyundai.values import CarControllerParams
from opendbc.car.structs import car
from openpilot.selfdrive.controls.lib.blatv2.controller import ControllerParams
from openpilot.selfdrive.controls.lib.blatv2.plant import PlantParams
from openpilot.selfdrive.controls.lib.blatv2.shadow import ShadowCore, ShadowResult

SHADOW_VERSION = 3
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
    mpc_compute_seconds = 0.0
    fallback_compute_seconds = 0.0
    try:
      common_started_ns = time.perf_counter_ns()
      result = self._begin_frame()
      common_compute_ns = time.perf_counter_ns() - common_started_ns

      candidate_started_ns = time.perf_counter_ns()
      self.core.compute_mpc()
      mpc_compute_seconds = (common_compute_ns + time.perf_counter_ns() - candidate_started_ns) * 1e-9

      candidate_started_ns = time.perf_counter_ns()
      self.core.compute_fallback()
      fallback_compute_seconds = (common_compute_ns + time.perf_counter_ns() - candidate_started_ns) * 1e-9
      result = self.core.end_frame()
    except (RuntimeError, ValueError, OverflowError):
      result = self.core.invalid_result()
      failed_compute_seconds = (time.perf_counter_ns() - started_ns) * 1e-9
      mpc_compute_seconds = failed_compute_seconds
      fallback_compute_seconds = failed_compute_seconds
    compute_seconds = (time.perf_counter_ns() - started_ns) * 1e-9
    assert (
      math.isfinite(compute_seconds)
      and math.isfinite(mpc_compute_seconds)
      and math.isfinite(fallback_compute_seconds)
      and compute_seconds >= 0.0
      and mpc_compute_seconds >= 0.0
      and fallback_compute_seconds >= 0.0
    )

    message = self.message
    message.logMonoTime = int(time.monotonic() * 1e9)
    message.valid = bool(result.valid and self.sm.all_checks(self.subscribed_services))
    shadow = self.shadow
    shadow.shadowVersion = SHADOW_VERSION
    shadow.valid = result.valid
    shadow.referenceCurvature = result.reference_curvature
    shadow.torqueDemand = result.torque_demand
    shadow.feasibleTorque = result.feasible_torque
    shadow.plantResidual = result.plant_residual
    shadow.scalarPlanDisagreement = result.scalar_plan_disagreement
    shadow.horizon = result.horizon
    shadow.computeTimeSeconds = float(compute_seconds)
    shadow.vEgo = result.v_ego
    shadow.aligningTorque = result.aligning_torque
    shadow.alignInputsValid = result.align_inputs_valid
    shadow.disturbanceEstimate = result.disturbance_estimate
    shadow.observerStatus = result.observer_status
    shadow.observerUnconstrainedUpdate = result.observer_unconstrained_update
    shadow.mpcCommandTorque = result.mpc_command_torque
    shadow.mpcStatus = result.mpc_status
    shadow.mpcCandidateCount = result.mpc_candidate_count
    shadow.mpcOptimalityResidual = result.mpc_optimality_residual
    shadow.mpcComputeTimeSeconds = float(mpc_compute_seconds)
    shadow.fallbackCommandTorque = result.fallback_command_torque
    shadow.fallbackStatus = result.fallback_status
    shadow.fallbackCandidateCount = result.fallback_candidate_count
    shadow.fallbackOptimalityResidual = result.fallback_optimality_residual
    shadow.fallbackComputeTimeSeconds = float(fallback_compute_seconds)
    self.pm.send("blatV2Shadow", message)

  def run(self) -> None:
    while True:
      self.step()


def main() -> None:
  BlatV2Shadow().run()


if __name__ == "__main__":
  main()

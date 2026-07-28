#!/usr/bin/env python3
"""Telemetry-only BLaTv2 plant/reference and frozen-v14 shadow.

This process owns no actuator publisher. Its sole output is ``blatV2Shadow``.
"""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any

import openpilot.cereal.messaging as messaging
from opendbc.car.car_helpers import interfaces
from opendbc.car.structs import car
from openpilot.common.realtime import config_realtime_process, Priority
from openpilot.selfdrive.controls.lib.blatv2.controller import ControllerParams
from openpilot.selfdrive.controls.lib.blatv2.plant import PlantParams
from openpilot.selfdrive.controls.lib.blatv2.shadow import ShadowCore, ShadowResult
from openpilot.selfdrive.controls.lib.blatv2.v14_shadow import (
  FrozenV14ShadowController,
  V14ShadowResult,
)

SHADOW_VERSION = 8
PUBLISHED_SERVICES = ("blatV2Shadow",)
SUBSCRIBED_SERVICES = (
  "modelV2",
  "carState",
  "carControl",
  "carOutput",
  "controlsState",
  "liveParameters",
  "liveTorqueParameters",
  "liveDelay",
  "lateralManeuverPlan",
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
  live_lqi_command_torque: float,
  live_lqi_status: int,
  live_lqi_compute_seconds: float,
  live_lqi_output_valid: bool,
  live_lqi_invalid_frames: int,
  live_lqi_recovery_ok_frames: int,
  live_lqi_controller_version: int,
  v14_result: V14ShadowResult,
  v14_compute_seconds: float,
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
  shadow.liveLqiCommandTorque = float(live_lqi_command_torque)
  shadow.liveLqiStatus = int(live_lqi_status)
  shadow.liveLqiComputeTimeSeconds = float(live_lqi_compute_seconds)
  shadow.liveLqiOutputValid = bool(live_lqi_output_valid)
  shadow.liveLqiInvalidFrames = int(live_lqi_invalid_frames)
  shadow.liveLqiRecoveryOkFrames = int(live_lqi_recovery_ok_frames)
  shadow.liveLqiControllerVersion = int(live_lqi_controller_version)
  shadow.v14CommandTorque = float(v14_result.command_torque)
  shadow.v14DesiredCurvature = float(v14_result.desired_curvature)
  shadow.v14ControllerVersion = int(v14_result.controller_version)
  shadow.v14Valid = bool(v14_result.valid)
  shadow.v14ComputeTimeSeconds = float(v14_compute_seconds)


class BlatV2Shadow:
  def __init__(self) -> None:
    from openpilot.common.params import Params

    # This is a structural actuation boundary, not merely a convention.
    assert "carControl" not in PUBLISHED_SERVICES
    assert "sendcan" not in PUBLISHED_SERVICES

    params = Params()
    cp_bytes = params.get("CarParams", block=True)
    self.CP = messaging.log_from_bytes(cp_bytes, car.CarParams)
    self.CI = interfaces[self.CP.carFingerprint](self.CP)
    self.seed_params = PlantParams.from_seed_file(
      SEED_PATH, self.CI.CC.params,
    )
    self.controller_params = ControllerParams.from_seed_file(CONTROLLER_SEED_PATH)
    self.torque_params = self.CP.lateralTuning.torque
    self.core = ShadowCore(self.seed_params, self.torque_params, self.CP, self.controller_params)
    self.v14 = FrozenV14ShadowController(self.CP, self.CI)

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
    v14_compute_seconds = 0.0
    phase = "shared"
    phase_started_ns = time.perf_counter_ns()
    try:
      result = self._begin_frame()
      phase_finished_ns = time.perf_counter_ns()
      shared_compute_seconds = (
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
      result = self.core.invalid_result()

    controls_state = self.sm["controlsState"]
    torque_state_valid = (
      controls_state.lateralControlState.which() == "torqueState"
    )
    torque_state = controls_state.lateralControlState.torqueState
    live_lateral_active = bool(
      torque_state.active if torque_state_valid else False
    )

    v14_started_ns = time.perf_counter_ns()
    try:
      v14_result = self.v14.step(
        self.sm["modelV2"],
        self.sm.valid["modelV2"],
        self.sm.updated["modelV2"],
        self.sm["carState"],
        self.sm["carOutput"],
        live_lateral_active,
        self.sm["liveParameters"],
        self.sm["liveTorqueParameters"],
        self.sm.all_checks(["liveTorqueParameters"]),
        float(self.sm["liveDelay"].lateralDelay),
        self.sm["lateralManeuverPlan"],
        self.sm.valid["lateralManeuverPlan"],
      )
    except (RuntimeError, ValueError, OverflowError):
      self.v14.reset()
      v14_result = self.v14.result
    v14_compute_seconds = (time.perf_counter_ns() - v14_started_ns) * 1e-9

    live_lqi_command = (
      float(torque_state.blatV2CommandTorque)
      if torque_state_valid else 0.0
    )
    live_lqi_status = (
      int(torque_state.blatV2Status)
      if torque_state_valid else 1
    )
    live_lqi_compute_seconds = (
      float(torque_state.blatV2ComputeTimeSeconds)
      if torque_state_valid else 0.0
    )
    live_lqi_output_valid = (
      bool(torque_state.blatV2OutputValid)
      if torque_state_valid else False
    )
    live_lqi_invalid_frames = (
      int(torque_state.blatV2InvalidFrames)
      if torque_state_valid else 0
    )
    live_lqi_recovery_ok_frames = (
      int(torque_state.blatV2RecoveryOkFrames)
      if torque_state_valid else 0
    )
    live_lqi_controller_version = (
      int(torque_state.version) if torque_state_valid else 0
    )
    compute_seconds = (time.perf_counter_ns() - started_ns) * 1e-9
    assert (
      math.isfinite(compute_seconds)
      and math.isfinite(shared_compute_seconds)
      and math.isfinite(live_lqi_compute_seconds)
      and math.isfinite(v14_compute_seconds)
      and compute_seconds >= 0.0
      and shared_compute_seconds >= 0.0
      and live_lqi_compute_seconds >= 0.0
      and v14_compute_seconds >= 0.0
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
      live_lqi_command_torque=live_lqi_command,
      live_lqi_status=live_lqi_status,
      live_lqi_compute_seconds=live_lqi_compute_seconds,
      live_lqi_output_valid=live_lqi_output_valid,
      live_lqi_invalid_frames=live_lqi_invalid_frames,
      live_lqi_recovery_ok_frames=live_lqi_recovery_ok_frames,
      live_lqi_controller_version=live_lqi_controller_version,
      v14_result=v14_result,
      v14_compute_seconds=v14_compute_seconds,
    )
    self.pm.send("blatV2Shadow", self.message)

  def run(self) -> None:
    while True:
      self.step()


def main() -> None:
  # CTRL_LOW preserves controlsd's preemption guarantee without forcing this
  # passive process to compete with CTRL_HIGH on core 4 every cycle. Roam only
  # on cores 0-4: core 5 carries equal-priority planning work, while 6/7 are
  # reserved for camera/model workloads. The standard setup also disables
  # cyclic GC before the hot loop.
  config_realtime_process([0, 1, 2, 3, 4], Priority.CTRL_LOW)
  BlatV2Shadow().run()


if __name__ == "__main__":
  main()

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
from openpilot.selfdrive.controls.lib.blatv2.plant import PlantParams
from openpilot.selfdrive.controls.lib.blatv2.reference import horizon
from openpilot.selfdrive.controls.lib.blatv2.shadow import ShadowCore, ShadowResult

SHADOW_VERSION = 1
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
    self.torque_params = self.CP.lateralTuning.torque
    self.core = ShadowCore(self.seed_params, self.torque_params)

    self.sm = messaging.SubMaster(list(SUBSCRIBED_SERVICES), poll="controlsState")
    self.pm = messaging.PubMaster(list(PUBLISHED_SERVICES))

  def _compute(self) -> ShadowResult:
    return self.core.compute(
      self.sm["modelV2"],
      self.sm["carState"],
      self.sm["carOutput"],
      self.sm["liveParameters"],
      float(self.sm["liveDelay"].lateralDelay),
      self.sm.valid["liveDelay"],
    )

  def step(self) -> None:
    self.sm.update()
    if not self.sm.updated["controlsState"]:
      return

    started_ns = time.perf_counter_ns()
    result = ShadowResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, horizon(self.seed_params))
    try:
      result = self._compute()
    except (ValueError, OverflowError):
      self.core.reset()
    compute_seconds = (time.perf_counter_ns() - started_ns) * 1e-9
    assert math.isfinite(compute_seconds) and compute_seconds >= 0.0

    message = messaging.new_message("blatV2Shadow")
    message.valid = bool(result.valid and self.sm.all_checks(list(SUBSCRIBED_SERVICES)))
    shadow = message.blatV2Shadow
    shadow.shadowVersion = SHADOW_VERSION
    shadow.valid = result.valid
    shadow.referenceCurvature = result.reference_curvature
    shadow.torqueDemand = result.torque_demand
    shadow.feasibleTorque = result.feasible_torque
    shadow.plantResidual = result.plant_residual
    shadow.scalarPlanDisagreement = result.scalar_plan_disagreement
    shadow.horizon = result.horizon
    shadow.computeTimeSeconds = float(compute_seconds)
    self.pm.send("blatV2Shadow", message)

  def run(self) -> None:
    while True:
      self.step()


def main() -> None:
  BlatV2Shadow().run()


if __name__ == "__main__":
  main()

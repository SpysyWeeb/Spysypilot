from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from openpilot.selfdrive.controls.lib.blatv2.controller import CandidateStatus
from openpilot.selfdrive.controls.lib.blatv2.live import (
  INVALID_DISENGAGE_FRAMES,
  RECOVERY_OK_FRAMES,
  LiveLQIController,
)


def ns(**values):
  return SimpleNamespace(**values)


def model():
  return ns(
    action=ns(desiredCurvature=0.01),
    orientationRate=ns(
      t=(0.0, 0.2, 0.4, 0.8),
      z=(0.1, 0.1, 0.1, 0.1),
    ),
    velocity=ns(
      t=(0.0, 0.2, 0.4, 0.8),
      x=(10.0, 10.0, 10.0, 10.0),
    ),
  )


def controller() -> LiveLQIController:
  car_params = ns(
    mass=2000.0,
    wheelbase=3.0,
    centerToFront=1.2,
    tireStiffnessFront=100000.0,
    tireStiffnessRear=110000.0,
    steerRatio=15.0,
    steerRatioRear=0.0,
  )
  limits = ns(
    STEER_MAX=409,
    STEER_DELTA_UP=4,
    STEER_DELTA_DOWN=7,
    STEER_STEP=1,
  )
  torque = ns(latAccelFactor=2.5, latAccelOffset=0.0, friction=0.1)
  return LiveLQIController(car_params, limits, torque)


def step(
  live: LiveLQIController,
  *,
  model_valid: bool,
  lateral_active: bool = True,
):
  car_state = ns(
    vEgo=10.0,
    steeringAngleDeg=2.0,
    steeringRateDeg=0.5,
    steeringPressed=False,
    standstill=False,
  )
  observer_control = ns(
    latActive=lateral_active,
    actuators=ns(torque=live.command_torque),
  )
  car_output = ns(actuatorsOutput=ns(torque=live.command_torque))
  parameters = ns(
    roll=0.01,
    angleOffsetDeg=0.2,
    stiffnessFactor=0.9,
    steerRatio=15.5,
  )
  return live.step(
    model(),
    car_state,
    observer_control,
    car_output,
    parameters,
    True,
    0.12,
    True,
    model_valid,
    lateral_active,
  )


def test_invalid_holds_once_decays_then_latches_comm_issue():
  live = controller()
  step(live, model_valid=True, lateral_active=False)
  step(live, model_valid=True, lateral_active=False)
  live.command_torque = 0.5

  first = step(live, model_valid=False)
  assert first.command_torque == 0.5
  assert first.output_valid
  second = step(live, model_valid=False)
  assert 0.0 < second.command_torque < first.command_torque

  result = second
  for _ in range(INVALID_DISENGAGE_FRAMES - 2):
    result = step(live, model_valid=False)
  assert result.command_torque == 0.0
  assert not result.output_valid
  assert result.invalid_frames == INVALID_DISENGAGE_FRAMES


def test_comm_issue_and_controller_recover_on_same_tenth_ok_frame():
  live = controller()
  step(live, model_valid=True, lateral_active=False)
  step(live, model_valid=True, lateral_active=False)
  for _ in range(INVALID_DISENGAGE_FRAMES):
    failed = step(live, model_valid=False)
  assert not failed.output_valid

  for expected in range(1, RECOVERY_OK_FRAMES):
    recovering = step(live, model_valid=True, lateral_active=False)
    assert recovering.recovery_ok_frames == expected
    assert not recovering.output_valid
    assert recovering.command_torque == 0.0

  recovered = step(live, model_valid=True, lateral_active=False)
  assert recovered.recovery_ok_frames == RECOVERY_OK_FRAMES
  assert recovered.output_valid
  assert recovered.status == int(CandidateStatus.OK)


def test_frozen_v14_sources_match_authority_blobs():
  root = Path(__file__).resolve().parents[0].parent
  expected = {
    "v14/latcontrol_torque.py": (
      "f042bde83638f0f536eadae593f3e1b8516cd3e1ae92bf313d46239df9b032e2"
    ),
    "v14/lateral_reference_planner.py": (
      "99c9ef94cf9609a63d85934fbd8c8bea19a9909639c968eebe0813aa44bb56b5"
    ),
  }
  for relative, digest in expected.items():
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest
  helper = root.parent / "lateral_torque_utils.py"
  assert hashlib.sha256(helper.read_bytes()).hexdigest() == (
    "c87b2416b80f278c5fa5e2d4a492a8ee3cece12527233cb5aa0f9e49a27c8b21"
  )

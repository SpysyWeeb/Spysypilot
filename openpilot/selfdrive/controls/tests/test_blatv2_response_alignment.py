from __future__ import annotations

from openpilot.selfdrive.controls.lib.blatv2.learner import ActuatorBoundary
from openpilot.selfdrive.controls.lib.blatv2.response_alignment import (
  CausalTorqueResponseAligner,
)


def aligner(maximum_delay: float = 0.12):
  return CausalTorqueResponseAligner(
    maximum_transport_delay_s=maximum_delay,
    maximum_gap_s=0.015,
  )


def record(
  instance: CausalTorqueResponseAligner,
  *,
  report: float,
  effective: float,
  torque: float,
  constrained: bool = False,
  boundary: ActuatorBoundary = ActuatorBoundary.NONE,
  dwell: float = 0.0,
  valid: bool = True,
):
  return instance.record(
    report_time_s=report,
    effective_time_s=effective,
    applied_torque=torque,
    actuator_constrained=constrained,
    boundary=boundary,
    magnitude_boundary_dwell_s=dwell,
    valid=valid,
  )


def test_zoh_alignment_uses_newest_command_not_after_causal_target():
  instance = aligner()
  for index in range(12):
    assert record(
      instance,
      report=1.01 + 0.01 * index,
      effective=1.00 + 0.01 * index,
      torque=index / 100.0,
    )

  # 1.125 - 0.105 = 1.020. The command at 1.02 is eligible; 1.03 is
  # forbidden even though it is closer in absolute time.
  selected = instance.aligned(
    response_time_s=1.125,
    transport_delay_s=0.105,
  )
  assert selected is not None
  assert selected.effective_time_s == 1.02
  assert selected.applied_torque == 0.02


def test_zero_delay_preserves_card_previous_cycle_convention():
  instance = aligner(maximum_delay=0.0)
  assert record(
    instance, report=1.01, effective=1.00, torque=0.25,
  )
  selected = instance.aligned(
    response_time_s=1.005,
    transport_delay_s=0.0,
  )
  assert selected is not None
  assert selected.effective_time_s == 1.00
  assert selected.applied_torque == 0.25


def test_boundary_and_dwell_metadata_move_with_exact_torque():
  instance = aligner()
  assert record(
    instance,
    report=1.01,
    effective=1.00,
    torque=1.0,
    constrained=True,
    boundary=ActuatorBoundary.MAGNITUDE,
    dwell=0.04,
  )
  for index in range(1, 11):
    assert record(
      instance,
      report=1.01 + 0.01 * index,
      effective=1.00 + 0.01 * index,
      torque=0.5,
    )
  selected = instance.aligned(
    response_time_s=1.105,
    transport_delay_s=0.105,
  )
  assert selected is not None
  assert selected.actuator_constrained
  assert selected.boundary is ActuatorBoundary.MAGNITUDE
  assert selected.magnitude_boundary_dwell_s == 0.04


def test_duplicate_report_is_idempotent_but_changed_payload_fails_closed():
  instance = aligner()
  kwargs = {
    "report": 1.01,
    "effective": 1.00,
    "torque": 0.25,
  }
  assert record(instance, **kwargs)
  assert record(instance, **kwargs)
  assert not record(instance, **(kwargs | {"torque": 0.5}))
  assert instance.aligned(
    response_time_s=1.005,
    transport_delay_s=0.0,
  ) is None


def test_gap_invalidity_and_timestamp_regression_flush_history():
  for second_report in (1.03, 1.00):
    instance = aligner()
    assert record(
      instance, report=1.01, effective=1.00, torque=0.25,
    )
    assert not record(
      instance,
      report=second_report,
      effective=second_report - 0.01,
      torque=0.3,
    )
    assert instance.aligned(
      response_time_s=1.01,
      transport_delay_s=0.0,
    ) is None


def test_invalid_epoch_and_excessive_response_age_never_bridge():
  instance = aligner()
  assert record(
    instance, report=1.01, effective=1.00, torque=0.25,
  )
  assert instance.aligned(
    response_time_s=1.03,
    transport_delay_s=0.0,
  ) is None
  assert not record(
    instance,
    report=1.02,
    effective=1.01,
    torque=0.3,
    valid=False,
  )
  assert instance.aligned(
    response_time_s=1.02,
    transport_delay_s=0.0,
  ) is None

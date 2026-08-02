from __future__ import annotations

import math
import struct

import pytest  # noqa: TID251

from openpilot.selfdrive.controls.lib.blatv2 import learning_backfill as backfill


class _FakeVehicleModel:
  def update_params(self, stiffness_factor: float, steer_ratio: float) -> None:
    self.stiffness_factor = stiffness_factor
    self.steer_ratio = steer_ratio

  def calc_curvature(
    self,
    steering_angle_radians: float,
    v_ego: float,
    roll: float,
  ) -> float:
    return steering_angle_radians + 0.01 * v_ego + roll


def _record(
  mono_ns: int,
  source_order: int,
  payload: object,
  *,
  valid: bool = True,
) -> backfill._TimedRouteRecord:
  return backfill._TimedRouteRecord(
    mono_ns=mono_ns,
    segment_index=0,
    ordinal=source_order,
    source_order=source_order,
    valid=valid,
    payload=payload,
  )


def _state(angle_deg: float) -> backfill._RecordedCarState:
  return backfill._RecordedCarState(
    v_ego=5.0,
    steering_angle_deg=angle_deg,
    steering_rate_deg_s=0.0,
    steering_torque=0.0,
    steering_pressed=False,
    standstill=False,
    steer_fault_temporary=False,
    steer_fault_permanent=False,
    can_valid=True,
    can_timeout=False,
  )


def _parameters() -> backfill._RecordedLiveParameters:
  return backfill._RecordedLiveParameters(
    valid=True,
    angle_offset_valid=True,
    steer_ratio_valid=True,
    stiffness_factor_valid=True,
    angle_offset_deg=0.0,
    steer_ratio=15.0,
    stiffness_factor=1.0,
    roll_rad=0.0,
  )


def _logged_curvature(angle_deg: float) -> float:
  calculated = -(math.radians(angle_deg) + 0.05)
  return struct.unpack("<f", struct.pack("<f", calculated))[0]


def _controls(
  mono_ns: int,
  source_order: int,
  curvature: float,
) -> backfill._TimedRouteRecord:
  return _record(
    mono_ns,
    source_order,
      backfill._RecordedControlsState(
        lateral_plan_mono_ns=0,
        measured_curvature=curvature,
        desired_curvature=curvature,
      modular_architecture="",
      modular_selection=-1,
      modular_artifact_sha256="",
      modular_source_openpilot_commit="",
      modular_opendbc_commit="",
      modular_selection_bound=False,
    ),
  )


def test_canonical_join_uses_float32_oracle_and_poll_owned_output() -> None:
  controls = (
    _controls(95, 0, _logged_curvature(1.0)),
    _controls(110, 1, _logged_curvature(2.0)),
  )
  joins, dropped, segment_context_dropped = backfill._build_canonical_control_joins(
    route_name="00000001--0000000001",
    controls=controls,
    polls=(_record(100, 2, None),),
    car_states=(
      _record(90, 3, _state(1.0)),
      _record(105, 4, _state(2.0)),
    ),
    live_parameters=(_record(80, 5, _parameters()),),
    car_outputs=(
      _record(94, 6, backfill._RecordedCarOutput(0.1, 41, True)),
      _record(106, 7, backfill._RecordedCarOutput(0.2, 82, True)),
    ),
    car_controls=(
      _record(90, 8, backfill._RecordedCarControl(True, 0.1)),
      _record(111, 9, backfill._RecordedCarControl(True, 0.2)),
    ),
    vehicle_model=_FakeVehicleModel(),
  )

  assert dropped == (95,)
  assert segment_context_dropped == ()
  assert len(joins) == 1
  joined = joins[0]
  assert joined.car_state.mono_ns == 105
  # carOutput is owned by the preceding selfdriveState poll, not a newer race.
  assert joined.car_output.mono_ns == 94
  # carControl is the first unmatched command after the controls witness.
  assert joined.car_control is not None
  assert joined.car_control.mono_ns == 111
  assert joined.curvature_unresolved is False


def test_canonical_join_no_match_retains_poll_baseline_and_marks_unresolved() -> None:
  joins, dropped, segment_context_dropped = backfill._build_canonical_control_joins(
    route_name="00000001--0000000001",
    controls=(_controls(110, 0, 123.0),),
    polls=(_record(100, 1, None),),
    car_states=(
      _record(90, 2, _state(1.0)),
      _record(105, 3, _state(2.0)),
    ),
    live_parameters=(_record(80, 4, _parameters()),),
    car_outputs=(_record(
      94,
      5,
      backfill._RecordedCarOutput(0.1, 41, True),
    ),),
    car_controls=(_record(
      111,
      6,
      backfill._RecordedCarControl(True, 0.2),
    ),),
    vehicle_model=_FakeVehicleModel(),
  )

  assert dropped == ()
  assert segment_context_dropped == ()
  assert joins[0].car_state.mono_ns == 90
  assert joins[0].curvature_unresolved is True
  assert backfill._measured_frame_from_join(joins[0]).inputs_valid is False


def test_same_cycle_car_control_is_not_reused_or_taken_from_next_cycle() -> None:
  controls = (
    _controls(110, 0, _logged_curvature(1.0)),
    _controls(120, 1, _logged_curvature(1.0)),
  )
  joins, _, _ = backfill._build_canonical_control_joins(
    route_name="00000001--0000000001",
    controls=controls,
    polls=(_record(100, 2, None),),
    car_states=(_record(90, 3, _state(1.0)),),
    live_parameters=(_record(80, 4, _parameters()),),
    car_outputs=(_record(
      94,
      5,
      backfill._RecordedCarOutput(0.1, 41, True),
    ),),
    car_controls=(
      # This belongs to the next witness and must not be stolen by 110.
      _record(120, 6, backfill._RecordedCarControl(True, 0.3)),
      _record(121, 7, backfill._RecordedCarControl(True, 0.4)),
    ),
    vehicle_model=_FakeVehicleModel(),
  )

  assert joins[0].car_control is None
  assert joins[1].car_control is not None
  assert joins[1].car_control.mono_ns == 120


def test_exact_model_link_ignores_a_newer_publication_in_the_race() -> None:
  controls = _controls(130, 0, _logged_curvature(1.0))
  controls = backfill._TimedRouteRecord(
    mono_ns=controls.mono_ns,
    segment_index=controls.segment_index,
    ordinal=controls.ordinal,
    source_order=controls.source_order,
    valid=controls.valid,
    payload=backfill._RecordedControlsState(
      lateral_plan_mono_ns=100,
      measured_curvature=controls.payload.measured_curvature,
      desired_curvature=controls.payload.desired_curvature,
      modular_architecture="blatv2.modular.inverse-rack",
      modular_selection=0,
      modular_artifact_sha256="",
      modular_source_openpilot_commit="",
      modular_opendbc_commit="",
      modular_selection_bound=False,
    ),
  )
  joins, _, _ = backfill._build_canonical_control_joins(
    route_name="00000001--0000000001",
    controls=(controls,),
    polls=(_record(120, 1, None),),
    car_states=(_record(90, 2, _state(1.0)),),
    live_parameters=(_record(80, 3, _parameters()),),
    car_outputs=(_record(
      110,
      4,
      backfill._RecordedCarOutput(0.1, 41, True),
    ),),
    car_controls=(_record(
      131,
      5,
      backfill._RecordedCarControl(True, 0.2),
    ),),
    vehicle_model=_FakeVehicleModel(),
  )
  models = (
    _record(100, 6, object()),
    _record(125, 7, object()),
  )

  assert backfill._exact_model_indices(joins, models) == (0,)


def _join_with_activity(*, lateral_active: bool) -> backfill._CanonicalControlJoin:
  joins, dropped, segment_context_dropped = backfill._build_canonical_control_joins(
    route_name="00000001--0000000001",
    controls=(_controls(110, 0, _logged_curvature(1.0)),),
    polls=(_record(100, 1, None),),
    car_states=(_record(90, 2, _state(1.0)),),
    live_parameters=(_record(80, 3, _parameters()),),
    car_outputs=(_record(
      94,
      4,
      backfill._RecordedCarOutput(0.1, 41, True),
    ),),
    car_controls=(_record(
      111,
      5,
      backfill._RecordedCarControl(lateral_active, 0.2),
    ),),
    vehicle_model=_FakeVehicleModel(),
  )
  assert dropped == ()
  assert segment_context_dropped == ()
  return joins[0]


def test_certification_drops_only_initial_missing_segment_measurements() -> None:
  arguments = {
    "route_name": "00000001--0000000001",
    "controls": (
      _controls(110, 0, _logged_curvature(1.0)),
      _controls(130, 1, _logged_curvature(2.0)),
    ),
    "polls": (
      _record(100, 2, None),
      _record(120, 3, None),
    ),
    # The first segment-local measurements arrive after the first witness.
    "car_states": (_record(115, 4, _state(2.0)),),
    "live_parameters": (_record(115, 5, _parameters()),),
    "car_outputs": (_record(
      116,
      6,
      backfill._RecordedCarOutput(0.1, 41, True),
    ),),
    "car_controls": (
      _record(111, 7, backfill._RecordedCarControl(True, 0.1)),
      _record(131, 8, backfill._RecordedCarControl(True, 0.2)),
    ),
    "vehicle_model": _FakeVehicleModel(),
  }

  with pytest.raises(
    backfill.RouteRejected,
    match="selfdriveState poll precedes the first carState",
  ):
    backfill._build_canonical_control_joins(**arguments)

  joins, pre_poll_dropped, segment_context_dropped = (
    backfill._build_canonical_control_joins(
      **arguments,
      certification_segment_mode=True,
    )
  )
  assert pre_poll_dropped == ()
  assert segment_context_dropped == (110,)
  assert len(joins) == 1
  assert joins[0].witness.mono_ns == 130
  assert joins[0].poll_mono_ns == 120


def test_certification_does_not_suppress_later_canonical_error() -> None:
  with pytest.raises(
    backfill.RouteRejected,
    match="carState compact payload has an invalid type",
  ):
    backfill._build_canonical_control_joins(
      route_name="00000001--0000000001",
      controls=(
        _controls(110, 0, _logged_curvature(1.0)),
        _controls(130, 1, _logged_curvature(2.0)),
        _controls(150, 2, _logged_curvature(3.0)),
      ),
      polls=(
        _record(100, 3, None),
        _record(120, 4, None),
        _record(140, 5, None),
      ),
      car_states=(
        _record(115, 6, _state(2.0)),
        _record(135, 7, object()),
      ),
      live_parameters=(_record(115, 8, _parameters()),),
      car_outputs=(
        _record(116, 9, backfill._RecordedCarOutput(0.1, 41, True)),
        _record(136, 10, backfill._RecordedCarOutput(0.2, 82, True)),
      ),
      car_controls=(
        _record(111, 11, backfill._RecordedCarControl(True, 0.1)),
        _record(131, 12, backfill._RecordedCarControl(True, 0.2)),
        _record(151, 13, backfill._RecordedCarControl(True, 0.3)),
      ),
      vehicle_model=_FakeVehicleModel(),
      certification_segment_mode=True,
    )


def test_inactive_missing_model_is_retained_startup_context() -> None:
  inactive = _join_with_activity(lateral_active=False)

  assert not backfill._active_witness_missing_exact_model_link(
    (inactive,),
    (None,),
  )


def test_active_missing_model_makes_behavior_preparation_ineligible() -> None:
  active = _join_with_activity(lateral_active=True)

  assert backfill._active_witness_missing_exact_model_link(
    (active,),
    (None,),
  )

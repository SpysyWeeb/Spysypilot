from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import replace
import math
import struct
from types import SimpleNamespace

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


def _output(
  torque: float,
  count: int,
  *,
  request_active: bool = True,
) -> backfill._RecordedCarOutput:
  return backfill._RecordedCarOutput(
    applied_torque=torque,
    torque_output_can_count=count,
    torque_output_can_valid=True,
    steering_request_active=request_active,
    steering_request_active_valid=True,
    steering_request_fault_avoidance_counter=0,
    steering_request_fault_avoidance_counter_valid=True,
  )


def _legacy_output(torque: float, count: int) -> backfill._RecordedCarOutput:
  return backfill._RecordedCarOutput(
    applied_torque=torque,
    torque_output_can_count=count,
    torque_output_can_valid=True,
    steering_request_active=False,
    steering_request_active_valid=False,
    steering_request_fault_avoidance_counter=0,
    steering_request_fault_avoidance_counter_valid=False,
  )


def _historical_descriptor(
  *,
  vehicle: str = "HYUNDAI_PALISADE",
) -> backfill.BuildDescriptor:
  return backfill.BuildDescriptor(
    superproject_commit="624d4c7677947cedf516d2bfad88591795975557",
    opendbc_commit="ab40b765445d1d18750b58ca6524b16ebe219b6b",
    panda_commit="7f245a890f7bc00712ca4ebf903190a084c7f86b",
    log_schema_blob="d40096ff46dc7d1b0dec3698e3e9c77a63b3fb72",
    supported_vehicle_identity=vehicle,
    steer_max=409,
    steer_delta_up=4,
    steer_delta_down=7,
    steer_step=1,
    driver_allowance=50,
    driver_multiplier=2,
    driver_factor=1,
    production_envelope_verified=True,
    rack_rate_resolution_deg_s=4.0,
  )


def _production_lkas11(
  torque_count: int,
  request_active: bool,
) -> backfill._RecordedOutgoingCanFrame:
  from opendbc.can import CANPacker
  from opendbc.car import Bus
  from opendbc.car.hyundai import hyundaican
  from opendbc.car.hyundai.values import CAR, DBC, HyundaiFlags

  cp = SimpleNamespace(
    carFingerprint=CAR.HYUNDAI_PALISADE,
    flags=HyundaiFlags.CHECKSUM_CRC8,
  )
  address, dat, bus = hyundaican.create_lkas11(
    CANPacker(DBC[CAR.HYUNDAI_PALISADE][Bus.pt]),
    3,
    cp,
    torque_count,
    request_active,
    not request_active,
    defaultdict(int),
    False,
    0,
    True,
    False,
    False,
    0,
    0,
  )
  return backfill._RecordedOutgoingCanFrame(address, bytes(dat), bus)


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


def test_car_output_request_state_is_explicit_and_missing_fields_fail_closed() -> None:
  legacy = backfill._copy_car_output(SimpleNamespace(
    actuatorsOutput=SimpleNamespace(torque=0.5, torqueOutputCan=193.0),
  ))
  assert legacy.torque_output_can_count == 193
  assert not legacy.steering_request_active
  assert not legacy.steering_request_active_valid
  assert legacy.steering_request_fault_avoidance_counter == 0
  assert not legacy.steering_request_fault_avoidance_counter_valid

  for request_active, counter in ((False, 90), (True, 0)):
    current = backfill._copy_car_output(SimpleNamespace(
      actuatorsOutput=SimpleNamespace(
        torque=0.5,
        torqueOutputCan=193.0,
        steeringRequestActive=request_active,
        steeringRequestActiveValid=True,
        steeringRequestFaultAvoidanceCounter=counter,
      ),
    ))
    assert current.torque_output_can_count == 193
    assert current.steering_request_active is request_active
    assert current.steering_request_active_valid
    assert current.steering_request_fault_avoidance_counter == counter
    assert current.steering_request_fault_avoidance_counter_valid


def test_device_type_and_timing_telemetry_are_retained_or_explicitly_unavailable() -> None:
  event = SimpleNamespace(
    which=lambda: "initData",
    logMonoTime=123,
    valid=True,
    initData=SimpleNamespace(
      gitCommit="1" * 40,
      dirty=False,
      dongleId="dongle",
      version="1.0",
      deviceType="tici",
    ),
  )
  decoded = backfill._decode_extracted_event(
    backfill.ExtractedEvent(0, 123, 0, b"event"),
    lambda _: nullcontext(event),
  )
  assert decoded.payload[-1] == "tici"

  fields = {
    "active": True,
    "modularArchitecture": "blatv2.modular.preview-rack",
    "modularSelection": 1,
    "modularArtifactHash": "",
    "modularSourceOpenpilotCommit": "1" * 40,
    "modularOpendbcCommit": "2" * 40,
    "modularSelectionBound": True,
    "modularProfileHash": "3" * 64,
    "modularPolicyHash": "4" * 64,
    "modularRuntimeIdentityHash": "5" * 64,
    "modularHorizonPolicyHash": "6" * 64,
    "modularControllerVersion": 2,
    "modularComputeTimeSeconds": 0.004,
    "modularControlWitnessMonoTime": 1_000_000_000,
    "modularIntentStatus": 0,
    "modularSafetyState": 1,
    "modularInvalidFrames": 0,
    "modularRecoveryOkFrames": 0,
    "modularControlsValid": True,
    "modularCarControlValid": True,
    "modularVehicleStateValid": True,
    "modularLiveParametersValid": True,
    "modularHorizonValid": True,
    "modularControlCadenceValid": True,
    "modularAdapterException": False,
    "modularProductionEnvelopeVerified": True,
    "modularFinalExpectedCounts": 41,
    "modularFinalCountResidual": 0,
    "modularFinalCountMatchValid": True,
    "modularFinalLimiterAltered": False,
  }

  def copy(torque_state: SimpleNamespace) -> backfill._RecordedControlsState:
    return backfill._copy_controls_state(SimpleNamespace(
      lateralPlanMonoTime=10,
      curvature=0.01,
      desiredCurvature=0.02,
      lateralControlState=SimpleNamespace(
        which=lambda: "torqueState",
        torqueState=torque_state,
      ),
    ))

  current = copy(SimpleNamespace(**fields))
  assert current.modular_telemetry_available
  assert current.modular_active
  assert current.modular_compute_time_seconds == 0.004
  assert current.modular_control_witness_mono_ns == 1_000_000_000
  assert current.modular_profile_sha256 == "3" * 64
  assert current.modular_policy_sha256 == "4" * 64
  assert current.modular_runtime_identity_sha256 == "5" * 64
  assert current.modular_horizon_policy_sha256 == "6" * 64
  assert current.modular_controls_valid
  assert current.modular_car_control_valid
  assert current.modular_vehicle_state_valid
  assert current.modular_live_parameters_valid
  assert current.modular_horizon_valid
  assert current.modular_control_cadence_valid
  assert not current.modular_adapter_exception
  assert current.modular_production_envelope_verified
  assert current.modular_final_expected_counts == 41
  assert current.modular_final_count_residual == 0
  assert current.modular_final_count_match_valid
  assert not current.modular_final_limiter_altered

  legacy_fields = {
    name: value
    for name, value in fields.items()
    if not name.startswith("modularProfile")
  }
  legacy = copy(SimpleNamespace(**legacy_fields))
  assert legacy.modular_selection == 1
  assert legacy.modular_selection_bound
  assert not legacy.modular_telemetry_available
  assert legacy.modular_compute_time_seconds == 0.0
  assert legacy.modular_control_witness_mono_ns == 0
  assert not legacy.modular_controls_valid
  assert not legacy.modular_final_count_match_valid


def test_current_car_output_schema_request_fields_round_trip() -> None:
  from opendbc.car import structs

  actuators = structs.CarControl.Actuators(
    torque=193 / 409.0,
    torqueOutputCan=193,
    steeringRequestActive=False,
    steeringRequestActiveValid=True,
    steeringRequestFaultAvoidanceCounter=90,
  )
  with structs.CarControl.Actuators.from_bytes(actuators.to_bytes()) as reader:
    copied = backfill._copy_car_output(SimpleNamespace(
      actuatorsOutput=reader,
    ))
  assert copied.torque_output_can_count == 193
  assert not copied.steering_request_active
  assert copied.steering_request_active_valid
  assert copied.steering_request_fault_avoidance_counter == 90
  assert copied.steering_request_fault_avoidance_counter_valid


@pytest.mark.parametrize("request_active", (True, False))
def test_historical_lkas11_decode_matches_production_packer(
  request_active: bool,
) -> None:
  decoded = backfill._decode_hyundai_lkas11(_production_lkas11(
    193,
    request_active,
  ))
  assert decoded is not None
  assert decoded.torque_count == 193
  assert decoded.steering_request_active is request_active


@pytest.mark.parametrize(
  "frame",
  (
    backfill._RecordedOutgoingCanFrame(0x341, b"\0" * 8, 0),
    backfill._RecordedOutgoingCanFrame(0x340, b"\0" * 8, 1),
    backfill._RecordedOutgoingCanFrame(0x340, b"\0" * 7, 0),
  ),
  ids=("address", "bus", "length"),
)
def test_historical_lkas11_decode_rejects_unknown_or_malformed_frame(
  frame: backfill._RecordedOutgoingCanFrame,
) -> None:
  assert backfill._decode_hyundai_lkas11(frame) is None


def test_prior_cycle_request_cut_preserves_command_and_censors_physical_fit() -> None:
  outputs = [
    _record(100_000_000, 0, _legacy_output(0.0, 0)),
    _record(120_000_000, 2, _legacy_output(193 / 409.0, 193)),
    _record(150_000_000, 4, _legacy_output(193 / 409.0, 193)),
  ]
  sends = [
    _record(
      105_000_000,
      1,
      backfill._RecordedSendcan((_production_lkas11(193, False),)),
    ),
    # card.py emits this after CarOutput_1. It cannot describe CarOutput_1.
    _record(
      121_000_000,
      3,
      backfill._RecordedSendcan((_production_lkas11(12, True),)),
    ),
  ]
  backfill._bind_prior_cycle_historical_steering_requests(
    descriptor=_historical_descriptor(),
    car_outputs=outputs,
    sendcan_records=sends,
  )

  cut = outputs[1].payload
  assert isinstance(cut, backfill._RecordedCarOutput)
  assert cut.torque_output_can_count == 193
  assert cut.applied_torque == pytest.approx(193 / 409.0)
  assert cut.steering_request_active_valid
  assert not cut.steering_request_active
  assert not cut.steering_request_fault_avoidance_counter_valid
  stale = outputs[2].payload
  assert isinstance(stale, backfill._RecordedCarOutput)
  assert not stale.steering_request_active_valid

  active_join = _join_with_activity(lateral_active=True)
  assert backfill._measured_frame_from_join(active_join).inputs_valid
  cut_join = replace(
    active_join,
    car_output=replace(active_join.car_output, payload=cut),
  )
  cut_frame = backfill._measured_frame_from_join(cut_join)
  assert cut_frame.applied_torque == pytest.approx(193 / 409.0)
  assert not cut_frame.inputs_valid


def test_historical_request_association_is_provenance_and_time_bounded() -> None:
  def bind(
    descriptor: backfill.BuildDescriptor,
    send_time_ns: int,
    *,
    send_count: int = 102,
  ) -> backfill._RecordedCarOutput:
    outputs = [
      _record(100_000_000, 0, _legacy_output(0.0, 0)),
      _record(120_000_000, 2, _legacy_output(0.25, 102)),
    ]
    backfill._bind_prior_cycle_historical_steering_requests(
      descriptor=descriptor,
      car_outputs=outputs,
      sendcan_records=[_record(
        send_time_ns,
        1,
        backfill._RecordedSendcan((_production_lkas11(send_count, True),)),
      )],
    )
    result = outputs[1].payload
    assert isinstance(result, backfill._RecordedCarOutput)
    return result

  assert bind(_historical_descriptor(), 105_000_000).steering_request_active_valid
  f4 = replace(
    _historical_descriptor(),
    superproject_commit="1021699bac528ba4ce39db23990c4d2e7867d4ba",
    opendbc_commit="68fda8e06e648fd23e2cdac6a5d04ef3df67f29b",
  )
  assert bind(f4, 105_000_000).steering_request_active_valid
  assert not bind(
    _historical_descriptor(vehicle="HYUNDAI_SONATA"),
    105_000_000,
  ).steering_request_active_valid
  assert not bind(
    _historical_descriptor(),
    101_000_000,
  ).steering_request_active_valid
  assert not bind(
    _historical_descriptor(),
    105_000_000,
    send_count=103,
  ).steering_request_active_valid


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
      _record(94, 6, _output(0.1, 41)),
      _record(106, 7, _output(0.2, 82)),
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
      _output(0.1, 41),
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
      _output(0.1, 41),
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
      _output(0.1, 41),
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
      _output(0.1, 41),
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
      _output(0.1, 41),
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
        _record(116, 9, _output(0.1, 41)),
        _record(136, 10, _output(0.2, 82)),
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

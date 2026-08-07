from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest

from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR, CarControllerParams
from opendbc.car.structs import CarParams
from opendbc.car.vehicle_model import VehicleModel
from openpilot.selfdrive.controls.lib.blatv2.actuator import (
  RuntimeTorqueLimits,
)
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import (
  RackMappingSnapshot,
)
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  PROVISIONAL_RACK_DYNAMICS_SCHEMA_VERSION,
  ProvisionalRackDynamics,
  RackDynamicsNode,
  RuntimeVehicleCompatibility,
  RuntimeVehicleCompatibilityError,
  build_runtime_vehicle_bundle,
)


PROVISIONAL_SEED_PATH = (
  Path(__file__).parents[1]
  / "lib"
  / "blatv2"
  / "provisional_rack_dynamics.json"
)


def palisade_cp():
  return CarInterface.get_non_essential_params(CAR.HYUNDAI_PALISADE)


def rack_dynamics() -> ProvisionalRackDynamics:
  return ProvisionalRackDynamics(
    rack_gain_deg_s2_per_torque=4000.0,
    rack_damping_per_s=10.0,
    rack_rate_resolution_deg_s=4.0,
    provenance="provisional casual-driving fit pending",
  )


def build_palisade(
  palisade_cp,
  rack_dynamics,
  *,
  car_interface_or_callback=None,
  controller_params=None,
  vehicle_identity: str = "palisade-test-identity",
):
  interface = (
    CarInterface(palisade_cp)
    if car_interface_or_callback is None
    else car_interface_or_callback
  )
  params = (
    CarControllerParams(palisade_cp)
    if controller_params is None
    else controller_params
  )
  return build_runtime_vehicle_bundle(
    car_params=palisade_cp,
    car_interface_or_callback=interface,
    controller_params=params,
    vehicle_identity=vehicle_identity,
    provisional_rack_dynamics=rack_dynamics,
  )


def _assert_raises(
  exception: type[BaseException],
  regex: str | None = None,
):
  case = unittest.TestCase()
  return (
    case.assertRaises(exception)
    if regex is None
    else case.assertRaisesRegex(exception, regex)
  )


def _test_real_palisade_uses_detected_opendbc_facts(
  palisade_cp,
  rack_dynamics,
) -> None:
  interface = CarInterface(palisade_cp)
  source_limits = CarControllerParams(palisade_cp)
  bundle = build_palisade(
    palisade_cp,
    rack_dynamics,
    car_interface_or_callback=interface,
    controller_params=source_limits,
  )
  assert bundle.compatibility == RuntimeVehicleCompatibility.SUPPORTED
  assert bundle.torque_limits == RuntimeTorqueLimits(
    steer_max=source_limits.STEER_MAX,
    delta_up=source_limits.STEER_DELTA_UP,
    delta_down=source_limits.STEER_DELTA_DOWN,
    steer_step=source_limits.STEER_STEP,
    driver_allowance=source_limits.STEER_DRIVER_ALLOWANCE,
    driver_multiplier=source_limits.STEER_DRIVER_MULTIPLIER,
    driver_factor=source_limits.STEER_DRIVER_FACTOR,
    production_envelope_verified=True,
  )

  callback = interface.torque_from_lateral_accel()
  expected_slope = float(
    callback(1.0, palisade_cp.lateralTuning.torque),
  )
  assert bundle.torque_callback_slope == expected_slope
  assert (
    bundle.torque_callback_max_abs_residual
    <= bundle.torque_callback_representation_tolerance
  )
  assert bundle.car_fingerprint == str(palisade_cp.carFingerprint)
  assert bundle.provisional_rack_provenance == rack_dynamics.provenance

  expected_mapping = RackMappingSnapshot.from_vehicle_model(
    VehicleModel(palisade_cp),
    roll_rad=0.0,
    angle_offset_deg=0.0,
    valid=True,
  )
  assert bundle.nominal_rack_mapping == expected_mapping
  assert bundle.nominal_rack_mapping.valid
  assert bundle.nominal_rack_mapping.roll_rad == 0.0
  assert bundle.nominal_rack_mapping.angle_offset_deg == 0.0

  friction = float(palisade_cp.lateralTuning.torque.friction)
  delay = float(palisade_cp.steerActuatorDelay)
  assert not bundle.calibration_seed_profile.qualified
  assert all(
    not node.parameters.qualified
    for node in bundle.calibration_seed_profile.nodes
  )
  for node in bundle.calibration_seed_profile.nodes:
    parameters = node.parameters
    assert parameters.torque_per_lateral_accel == expected_slope
    assert parameters.lateral_accel_offset_correction_mps2 == 0.0
    # This is already normalized torque in current opendbc. A callback-slope
    # multiplication here would silently shrink or enlarge breakaway support.
    assert parameters.static_breakaway_torque == friction
    assert parameters.kinetic_friction_torque == friction
    assert parameters.transport_delay_s == delay
    assert (
      parameters.rack_rate_resolution_deg_s
      == source_limits.BLATV2_RACK_RATE_RESOLUTION_DEG_S
    )
    assert not hasattr(parameters, "rack_gain_deg_s2_per_torque")
    assert not hasattr(parameters, "rack_damping_per_s")
  assert not bundle.seed_profile.qualified
  assert all(not node.parameters.qualified for node in bundle.seed_profile.nodes)
  assert all(node.sample_count == 0 for node in bundle.seed_profile.nodes)
  for node in bundle.seed_profile.nodes:
    parameters = node.parameters
    assert parameters.torque_per_lateral_accel == expected_slope
    assert parameters.transport_delay_s == delay
    assert parameters.static_friction_torque == friction
    assert parameters.kinetic_friction_torque == friction
    assert (
      parameters.rack_gain_deg_s2_per_torque
      == rack_dynamics.rack_gain_deg_s2_per_torque
    )
    assert parameters.rack_damping_per_s == rack_dynamics.rack_damping_per_s
    assert (
      parameters.rack_rate_resolution_deg_s
      == source_limits.BLATV2_RACK_RATE_RESOLUTION_DEG_S
    )
  assert str(palisade_cp.carFingerprint) in bundle.seed_profile.provenance
  assert rack_dynamics.provenance in bundle.seed_profile.provenance


def _test_alternate_limits_and_linear_callback_propagate_generically(
  palisade_cp,
  rack_dynamics,
) -> None:
  alternate_limits = SimpleNamespace(
    STEER_MAX=137,
    STEER_DELTA_UP=6,
    STEER_DELTA_DOWN=9,
    STEER_STEP=2,
    STEER_DRIVER_ALLOWANCE=17,
    STEER_DRIVER_MULTIPLIER=3,
    STEER_DRIVER_FACTOR=2,
  )

  def alternate_callback(lateral_accel, _torque_tuning):
    return 0.25 * lateral_accel

  bundle = build_palisade(
    palisade_cp,
    rack_dynamics,
    car_interface_or_callback=alternate_callback,
    controller_params=alternate_limits,
    vehicle_identity="alternate-vehicle",
  )
  assert bundle.torque_limits == RuntimeTorqueLimits(
    137, 6, 9, 2, 17, 3, 2,
  )
  assert bundle.torque_callback_slope == 0.25
  assert all(
    node.parameters.torque_per_lateral_accel == 0.25
    for node in bundle.seed_profile.nodes
  )


def _test_verified_envelope_requires_vehicle_owned_rate_resolution(
  palisade_cp,
  rack_dynamics,
) -> None:
  source = CarControllerParams(palisade_cp)
  missing_resolution = SimpleNamespace(
    STEER_MAX=source.STEER_MAX,
    STEER_DELTA_UP=source.STEER_DELTA_UP,
    STEER_DELTA_DOWN=source.STEER_DELTA_DOWN,
    STEER_STEP=source.STEER_STEP,
    STEER_DRIVER_ALLOWANCE=source.STEER_DRIVER_ALLOWANCE,
    STEER_DRIVER_MULTIPLIER=source.STEER_DRIVER_MULTIPLIER,
    STEER_DRIVER_FACTOR=source.STEER_DRIVER_FACTOR,
    BLATV2_RUNTIME_ENVELOPE_COMPATIBLE=True,
  )
  with _assert_raises(RuntimeVehicleCompatibilityError) as error:
    build_palisade(
      palisade_cp,
      rack_dynamics,
      controller_params=missing_resolution,
    )
  assert (
    error.exception.status
    == RuntimeVehicleCompatibility.MISSING_RACK_RATE_RESOLUTION
  )


def _test_verified_envelope_requires_100_hz_command_cadence(
  palisade_cp,
  rack_dynamics,
) -> None:
  source = CarControllerParams(palisade_cp)
  wrong_cadence = SimpleNamespace(
    STEER_MAX=source.STEER_MAX,
    STEER_DELTA_UP=source.STEER_DELTA_UP,
    STEER_DELTA_DOWN=source.STEER_DELTA_DOWN,
    STEER_STEP=2,
    STEER_DRIVER_ALLOWANCE=source.STEER_DRIVER_ALLOWANCE,
    STEER_DRIVER_MULTIPLIER=source.STEER_DRIVER_MULTIPLIER,
    STEER_DRIVER_FACTOR=source.STEER_DRIVER_FACTOR,
    BLATV2_RUNTIME_ENVELOPE_COMPATIBLE=True,
    BLATV2_RACK_RATE_RESOLUTION_DEG_S=4.0,
  )
  with _assert_raises(RuntimeVehicleCompatibilityError) as error:
    build_palisade(
      palisade_cp,
      rack_dynamics,
      controller_params=wrong_cadence,
    )
  assert (
    error.exception.status
    == RuntimeVehicleCompatibility.INVALID_CONTROLLER_LIMITS
  )


def _test_nonlinear_asymmetric_or_offset_callbacks_fail_closed(
  palisade_cp,
  rack_dynamics,
  callback,
) -> None:
  with _assert_raises(
    RuntimeVehicleCompatibilityError,
    regex="nonlinear, asymmetric, or offset",
  ) as error:
    build_palisade(
      palisade_cp,
      rack_dynamics,
      car_interface_or_callback=callback,
    )
  assert (
    error.exception.status
    == RuntimeVehicleCompatibility.INCOMPATIBLE_TORQUE_CALLBACK
  )


def _test_non_torque_steer_control_fails_closed(
  palisade_cp,
  rack_dynamics,
  steer_control_type,
) -> None:
  incompatible = palisade_cp.copy()
  incompatible.steerControlType = steer_control_type
  with _assert_raises(RuntimeVehicleCompatibilityError) as error:
    build_palisade(incompatible, rack_dynamics)
  assert (
    error.exception.status
    == RuntimeVehicleCompatibility.UNSUPPORTED_STEER_CONTROL
  )


def _test_non_torque_lateral_tuning_fails_closed(
  palisade_cp,
  rack_dynamics,
) -> None:
  incompatible = palisade_cp.copy()
  incompatible.lateralTuning.init("pid")
  with _assert_raises(RuntimeVehicleCompatibilityError) as error:
    build_palisade(incompatible, rack_dynamics)
  assert (
    error.exception.status
    == RuntimeVehicleCompatibility.UNSUPPORTED_LATERAL_TUNING
  )


def _test_missing_fractional_or_invalid_limits_fail_closed(
  palisade_cp,
  rack_dynamics,
  controller_params,
) -> None:
  with _assert_raises(RuntimeVehicleCompatibilityError) as error:
    build_palisade(
      palisade_cp,
      rack_dynamics,
      controller_params=controller_params,
    )
  assert (
    error.exception.status
    == RuntimeVehicleCompatibility.INVALID_CONTROLLER_LIMITS
  )


def _test_invalid_vehicle_geometry_or_calibration_fails_closed(
  palisade_cp,
  rack_dynamics,
  field,
  value,
) -> None:
  incompatible = palisade_cp.copy()
  setattr(incompatible, field, value)
  with _assert_raises(RuntimeVehicleCompatibilityError) as error:
    build_palisade(incompatible, rack_dynamics)
  assert error.exception.status in (
    RuntimeVehicleCompatibility.INVALID_VEHICLE_CALIBRATION,
    RuntimeVehicleCompatibility.INCOMPATIBLE_TORQUE_CALLBACK,
  )


def _test_identity_fingerprint_callback_and_provenance_are_required(
  palisade_cp,
  rack_dynamics,
) -> None:
  with _assert_raises(RuntimeVehicleCompatibilityError) as error:
    build_palisade(palisade_cp, rack_dynamics, vehicle_identity=" ")
  assert error.exception.status == RuntimeVehicleCompatibility.INVALID_IDENTITY

  no_fingerprint = palisade_cp.copy()
  no_fingerprint.carFingerprint = ""
  with _assert_raises(RuntimeVehicleCompatibilityError) as error:
    build_palisade(
      no_fingerprint,
      rack_dynamics,
      car_interface_or_callback=(
        lambda lateral_accel, _: 0.25 * lateral_accel
      ),
    )
  assert (
    error.exception.status
    == RuntimeVehicleCompatibility.INVALID_VEHICLE_CALIBRATION
  )

  with _assert_raises(RuntimeVehicleCompatibilityError) as error:
    build_palisade(
      palisade_cp,
      rack_dynamics,
      car_interface_or_callback=object(),
    )
  assert (
    error.exception.status
    == RuntimeVehicleCompatibility.MISSING_TORQUE_CALLBACK
  )

  with _assert_raises(RuntimeVehicleCompatibilityError) as error:
    ProvisionalRackDynamics(4000.0, 10.0, 4.0, " ")
  assert (
    error.exception.status
    == RuntimeVehicleCompatibility.INVALID_PROVISIONAL_DYNAMICS
  )


def _test_provisional_dynamics_have_no_unknown_defaults(values) -> None:
  with _assert_raises(RuntimeVehicleCompatibilityError) as error:
    ProvisionalRackDynamics(*values, "explicit")
  assert (
    error.exception.status
    == RuntimeVehicleCompatibility.INVALID_PROVISIONAL_DYNAMICS
  )


def _test_bundle_json_and_identity_are_deterministic(
  palisade_cp,
  rack_dynamics,
) -> None:
  first = build_palisade(palisade_cp, rack_dynamics)
  second = build_palisade(palisade_cp, rack_dynamics)
  assert first == second
  assert first.to_dict()["schema_version"] == 1
  assert "calibration_seed_profile" not in first.to_dict()
  assert first.to_json() == second.to_json()
  assert first.identity_sha256 == second.identity_sha256
  assert (
    first.calibration_identity_sha256
    == second.calibration_identity_sha256
  )
  assert first.identity_sha256 == hashlib.sha256(
    first.to_json().encode("utf-8"),
  ).hexdigest()

  changed_legacy_dynamics = build_palisade(
    palisade_cp,
    ProvisionalRackDynamics(
      rack_gain_deg_s2_per_torque=(
        rack_dynamics.rack_gain_deg_s2_per_torque * 2.0
      ),
      rack_damping_per_s=rack_dynamics.rack_damping_per_s + 1.0,
      rack_rate_resolution_deg_s=(
        rack_dynamics.rack_rate_resolution_deg_s
      ),
      provenance="different retired rack seed",
    ),
  )
  assert changed_legacy_dynamics.identity_sha256 != first.identity_sha256
  assert (
    changed_legacy_dynamics.calibration_identity_sha256
    == first.calibration_identity_sha256
  )
  assert (
    first.calibration_identity_dict()["calibration_seed_profile"]
    == first.calibration_seed_profile.to_dict()
  )


def _test_calibration_identity_tracks_stock_lateral_accel_offset(
  palisade_cp,
  rack_dynamics,
) -> None:
  baseline = build_palisade(palisade_cp, rack_dynamics)
  changed_cp = palisade_cp.copy()
  changed_cp.lateralTuning.torque.latAccelOffset = (
    float(palisade_cp.lateralTuning.torque.latAccelOffset) + 0.125
  )
  changed = build_palisade(changed_cp, rack_dynamics)

  assert baseline.stock_lateral_accel_offset_mps2 == float(
    palisade_cp.lateralTuning.torque.latAccelOffset,
  )
  assert changed.stock_lateral_accel_offset_mps2 == float(
    changed_cp.lateralTuning.torque.latAccelOffset,
  )
  assert "stock_lateral_accel_offset_mps2" not in baseline.to_dict()
  assert baseline.to_dict() == changed.to_dict()
  assert baseline.identity_sha256 == changed.identity_sha256
  assert (
    baseline.calibration_identity_sha256
    != changed.calibration_identity_sha256
  )


def _test_unverified_calibration_identity_excludes_provisional_rack_dynamics(
  palisade_cp,
) -> None:
  controller_params = SimpleNamespace(
    STEER_MAX=137,
    STEER_DELTA_UP=6,
    STEER_DELTA_DOWN=9,
    STEER_STEP=2,
    STEER_DRIVER_ALLOWANCE=17,
    STEER_DRIVER_MULTIPLIER=3,
    STEER_DRIVER_FACTOR=2,
  )

  def callback(lateral_accel, _torque_tuning):
    return 0.25 * lateral_accel

  first_dynamics = ProvisionalRackDynamics(
    rack_gain_deg_s2_per_torque=4000.0,
    rack_damping_per_s=10.0,
    rack_rate_resolution_deg_s=4.0,
    provenance="first provisional rack dynamics",
  )
  changed_dynamics = ProvisionalRackDynamics(
    rack_gain_deg_s2_per_torque=8000.0,
    rack_damping_per_s=25.0,
    rack_rate_resolution_deg_s=4.0,
    provenance="second provisional rack dynamics",
  )
  first = build_palisade(
    palisade_cp,
    first_dynamics,
    car_interface_or_callback=callback,
    controller_params=controller_params,
    vehicle_identity="unverified-calibration-platform",
  )
  changed = build_palisade(
    palisade_cp,
    changed_dynamics,
    car_interface_or_callback=callback,
    controller_params=controller_params,
    vehicle_identity="unverified-calibration-platform",
  )

  assert first.identity_sha256 != changed.identity_sha256
  assert first.calibration_identity_sha256 == changed.calibration_identity_sha256
  assert first.calibration_seed_profile.provenance.endswith(
    "rack_rate_resolution_source=provisional",
  )
  assert first_dynamics.provenance not in first.calibration_seed_profile.provenance
  assert changed_dynamics.provenance not in changed.calibration_seed_profile.provenance

  changed_resolution = build_palisade(
    palisade_cp,
    ProvisionalRackDynamics(
      rack_gain_deg_s2_per_torque=first_dynamics.rack_gain_deg_s2_per_torque,
      rack_damping_per_s=first_dynamics.rack_damping_per_s,
      rack_rate_resolution_deg_s=5.0,
      provenance="resolution-only provisional change",
    ),
    car_interface_or_callback=callback,
    controller_params=controller_params,
    vehicle_identity="unverified-calibration-platform",
  )
  assert (
    first.calibration_identity_sha256
    != changed_resolution.calibration_identity_sha256
  )


def _test_committed_provisional_seed_is_explicit_and_unqualified(
  palisade_cp,
) -> None:
  dynamics = ProvisionalRackDynamics.from_json_file(
    PROVISIONAL_SEED_PATH,
  )
  bundle = build_palisade(palisade_cp, dynamics)

  assert PROVISIONAL_RACK_DYNAMICS_SCHEMA_VERSION == 2
  assert dynamics.provenance
  assert not bundle.seed_profile.qualified
  assert all(
    not node.parameters.qualified
    for node in bundle.seed_profile.nodes
  )
  assert "not an approved artifact" in dynamics.provenance
  assert "owner field testing" in dynamics.provenance

  expected = (
    (0.0, 4000.0, 10.0),
    (5.0, 4000.0, 10.0),
    (10.0, 3200.0, 14.0),
    (15.0, 3200.0, 14.0),
    (20.0, 3200.0, 14.0),
    (30.0, 3200.0, 14.0),
  )
  assert tuple(
    (
      node.speed_mps,
      node.rack_gain_deg_s2_per_torque,
      node.rack_damping_per_s,
    )
    for node in dynamics.nodes
  ) == expected
  assert dynamics.parameters_at_speed(7.5) == (3600.0, 12.0)
  assert tuple(
    (
      node.speed_mps,
      node.parameters.rack_gain_deg_s2_per_torque,
      node.parameters.rack_damping_per_s,
    )
    for node in bundle.seed_profile.nodes
  ) == expected


def _test_provisional_speed_schedule_fails_closed() -> None:
  valid_nodes = tuple(
    RackDynamicsNode(speed, 4000.0, 10.0)
    for speed in (0.0, 5.0, 10.0, 15.0, 20.0, 30.0)
  )
  with _assert_raises(RuntimeVehicleCompatibilityError):
    ProvisionalRackDynamics(
      4000.0,
      10.0,
      4.0,
      "explicit",
      valid_nodes[:-1],
    )
  replaced = replace(
    ProvisionalRackDynamics(
      4000.0,
      10.0,
      4.0,
      "explicit",
      valid_nodes,
    ),
    rack_gain_deg_s2_per_torque=3200.0,
  )
  assert all(
    node.rack_gain_deg_s2_per_torque == 3200.0
    for node in replaced.nodes
  )


def _test_provisional_seed_schema_fails_closed(
  tmp_path,
  payload,
) -> None:
  path = tmp_path / "seed.json"
  path.write_text(payload, encoding="utf-8")
  with _assert_raises(RuntimeVehicleCompatibilityError) as error:
    ProvisionalRackDynamics.from_json_file(path)
  assert (
    error.exception.status
    == RuntimeVehicleCompatibility.INVALID_PROVISIONAL_DYNAMICS
  )


def _test_production_adapter_has_no_platform_or_actuator_literals() -> None:
  source_path = (
    Path(__file__).parents[1]
    / "lib"
    / "blatv2"
    / "runtime_vehicle.py"
  )
  source = source_path.read_text(encoding="utf-8")
  lowered = source.lower()
  assert "hyundai" not in lowered
  assert "livedelay" not in lowered
  assert "dt_mdl" not in lowered
  assert "lat_smooth" not in lowered
  assert "409" not in source
  assert "384" not in source
  assert re.search(
    r"steer_(?:max|delta_up|delta_down|step)\\s*=\\s*\\d",
    lowered,
  ) is None


class TestBlatV2RuntimeVehicle(unittest.TestCase):
  def setUp(self):
    self.palisade_cp = palisade_cp()
    self.rack_dynamics = rack_dynamics()
    self.temporary_directory = tempfile.TemporaryDirectory()
    self.tmp_path = Path(self.temporary_directory.name)

  def tearDown(self):
    self.temporary_directory.cleanup()

  def test_real_palisade_uses_detected_opendbc_facts(self):
    _test_real_palisade_uses_detected_opendbc_facts(
      self.palisade_cp,
      self.rack_dynamics,
    )

  def test_calibration_identity_tracks_stock_lateral_accel_offset(self):
    _test_calibration_identity_tracks_stock_lateral_accel_offset(
      self.palisade_cp,
      self.rack_dynamics,
    )

  def test_unverified_calibration_identity_excludes_provisional_rack_dynamics(
    self,
  ):
    _test_unverified_calibration_identity_excludes_provisional_rack_dynamics(
      self.palisade_cp,
    )

  def test_alternate_limits_and_linear_callback_propagate_generically(self):
    _test_alternate_limits_and_linear_callback_propagate_generically(
      self.palisade_cp,
      self.rack_dynamics,
    )

  def test_verified_envelope_requires_vehicle_owned_rate_resolution(self):
    _test_verified_envelope_requires_vehicle_owned_rate_resolution(
      self.palisade_cp,
      self.rack_dynamics,
    )

  def test_verified_envelope_requires_100_hz_command_cadence(self):
    _test_verified_envelope_requires_100_hz_command_cadence(
      self.palisade_cp,
      self.rack_dynamics,
    )

  def test_nonlinear_asymmetric_or_offset_callbacks_fail_closed(self):
    callbacks = (
      lambda lateral_accel, _: (
        0.25 * lateral_accel + 0.01 * lateral_accel * lateral_accel
      ),
      lambda lateral_accel, _: (
        (0.20 if lateral_accel >= 0.0 else 0.30) * lateral_accel
      ),
      lambda lateral_accel, _: 0.25 * lateral_accel + 0.001,
    )
    for callback in callbacks:
      with self.subTest(callback=callback):
        _test_nonlinear_asymmetric_or_offset_callbacks_fail_closed(
          self.palisade_cp,
          self.rack_dynamics,
          callback,
        )

  def test_non_torque_steer_control_fails_closed(self):
    for steer_control_type in (
      CarParams.SteerControlType.angle,
      CarParams.SteerControlType.curvature,
    ):
      with self.subTest(steer_control_type=steer_control_type):
        _test_non_torque_steer_control_fails_closed(
          self.palisade_cp,
          self.rack_dynamics,
          steer_control_type,
        )

  def test_non_torque_lateral_tuning_fails_closed(self):
    _test_non_torque_lateral_tuning_fails_closed(
      self.palisade_cp,
      self.rack_dynamics,
    )

  def test_missing_fractional_or_invalid_limits_fail_closed(self):
    controller_params_cases = (
      SimpleNamespace(),
      SimpleNamespace(
        STEER_MAX=100.5,
        STEER_DELTA_UP=2,
        STEER_DELTA_DOWN=3,
        STEER_STEP=1,
        STEER_DRIVER_ALLOWANCE=50,
        STEER_DRIVER_MULTIPLIER=2,
        STEER_DRIVER_FACTOR=1,
      ),
      SimpleNamespace(
        STEER_MAX=100,
        STEER_DELTA_UP=0,
        STEER_DELTA_DOWN=3,
        STEER_STEP=1,
        STEER_DRIVER_ALLOWANCE=50,
        STEER_DRIVER_MULTIPLIER=2,
        STEER_DRIVER_FACTOR=1,
      ),
    )
    for controller_params in controller_params_cases:
      with self.subTest(controller_params=controller_params):
        _test_missing_fractional_or_invalid_limits_fail_closed(
          self.palisade_cp,
          self.rack_dynamics,
          controller_params,
        )

  def test_invalid_vehicle_geometry_or_calibration_fails_closed(self):
    cases = (
      ("mass", 0.0),
      ("rotationalInertia", 0.0),
      ("wheelbase", 0.0),
      ("centerToFront", math.inf),
      ("steerRatio", 0.0),
      ("tireStiffnessFront", 0.0),
      ("tireStiffnessRear", math.nan),
      ("steerActuatorDelay", -0.1),
      ("maxLateralAccel", 0.0),
    )
    for field, value in cases:
      with self.subTest(field=field, value=value):
        _test_invalid_vehicle_geometry_or_calibration_fails_closed(
          self.palisade_cp,
          self.rack_dynamics,
          field,
          value,
        )

  def test_identity_fingerprint_callback_and_provenance_are_required(self):
    _test_identity_fingerprint_callback_and_provenance_are_required(
      self.palisade_cp,
      self.rack_dynamics,
    )

  def test_provisional_dynamics_have_no_unknown_defaults(self):
    for values in (
      (math.nan, 10.0, 4.0),
      (0.0, 10.0, 4.0),
      (4000.0, -1.0, 4.0),
      (4000.0, 10.0, math.inf),
    ):
      with self.subTest(values=values):
        _test_provisional_dynamics_have_no_unknown_defaults(values)

  def test_bundle_json_and_identity_are_deterministic(self):
    _test_bundle_json_and_identity_are_deterministic(
      self.palisade_cp,
      self.rack_dynamics,
    )

  def test_committed_provisional_seed_is_explicit_and_unqualified(self):
    _test_committed_provisional_seed_is_explicit_and_unqualified(
      self.palisade_cp,
    )

  def test_provisional_seed_schema_fails_closed(self):
    payloads = (
      "{}",
      "".join((
        '{"schema_version":2,"provisional":true,',
        '"rack_gain_deg_s2_per_torque":4000.0,',
        '"rack_damping_per_s":10.0,',
        '"rack_rate_resolution_deg_s":4.0,',
        '"provenance":"explicit"}',
      )),
      "".join((
        '{"schema_version":1,"provisional":false,',
        '"rack_gain_deg_s2_per_torque":4000.0,',
        '"rack_damping_per_s":10.0,',
        '"rack_rate_resolution_deg_s":4.0,',
        '"provenance":"explicit"}',
      )),
    )
    for payload in payloads:
      with self.subTest(payload=payload):
        _test_provisional_seed_schema_fails_closed(
          self.tmp_path,
          payload,
        )

  def test_provisional_speed_schedule_fails_closed(self):
    _test_provisional_speed_schedule_fails_closed()

  def test_production_adapter_has_no_platform_or_actuator_literals(self):
    _test_production_adapter_has_no_platform_or_actuator_literals()

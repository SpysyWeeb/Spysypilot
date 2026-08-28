"""Canonical comma-device timing/comms receipt from route evidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

from openpilot.selfdrive.controls.lib.blatv2.bootstrap import (
  ControllerSelection,
)
from openpilot.selfdrive.controls.lib.blatv2.intent import IntentStatusCode
from openpilot.selfdrive.controls.lib.blatv2.live_adapter import (
  MAX_RECORDED_FRAME_GAP_NS,
)
from openpilot.selfdrive.controls.lib.blatv2.live_safety import LiveSafetyState
from openpilot.selfdrive.controls.lib.blatv2.route_evidence import (
  RouteEvidenceStreamReader,
)


DEVICE_ACCEPTANCE_SCHEMA_VERSION = 1
MAX_DEVICE_ACCEPTANCE_ROUTES = 128
MAX_DEVICE_COMPUTE_SECONDS = 0.010
NOMINAL_CONTROL_PERIOD_NS = 10_000_000
PERCENTILE_METHOD = "nearest_rank"
MODULAR_ARCHITECTURE = "blatv2.modular.preview-rack"
COMMA_DEVICE_TYPES = frozenset(("tici", "tizi", "mici"))
FAILURE_REASONS = (
  "actuator_correspondence_invalid",
  "adapter_exception",
  "car_control_invalid",
  "compute_budget_exceeded",
  "compute_invalid",
  "control_cadence_invalid",
  "control_witness_invalid",
  "controller_architecture_mismatch",
  "controller_policy_identity_mismatch",
  "controls_state_invalid",
  "device_identity_mismatch",
  "horizon_invalid",
  "horizon_policy_identity_mismatch",
  "inferred_cadence_gap",
  "invalid_frames_nonzero",
  "intent_not_ok",
  "live_artifact_identity_mismatch",
  "live_parameters_invalid",
  "modular_binding_invalid",
  "no_eligible_samples",
  "non_comma_device",
  "production_envelope_unverified",
  "profile_identity_mismatch",
  "pre_poll_witness_dropped",
  "recorded_cadence_gap",
  "recovery_frames_nonzero",
  "route_gap_summary_nonzero",
  "runtime_identity_mismatch",
  "safety_not_ok",
  "source_identity_mismatch",
  "telemetry_unavailable",
  "unresolved_witness_summary_nonzero",
  "vehicle_identity_mismatch",
  "vehicle_state_invalid",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_RECEIPT_KEYS = frozenset(
  (
    "schemaVersion",
    "routeEvidenceSha256s",
    "deviceType",
    "vehicleIdentity",
    "controllerArchitecture",
    "sourceOpenpilotCommit",
    "opendbcCommit",
    "pandaCommit",
    "liveArtifactSha256",
    "runtimeIdentitySha256",
    "profileSha256",
    "controllerPolicySha256",
    "horizonPolicySha256",
    "sampleCount",
    "percentileMethod",
    "computeP50Seconds",
    "computeP90Seconds",
    "computeP99Seconds",
    "computeMaxSeconds",
    "dropCount",
    "failureCounts",
  )
)


class DeviceAcceptanceError(ValueError):
  pass


def _uint(value: object, name: str, maximum: int = (1 << 63) - 1) -> int:
  if type(value) is not int or value < 0 or value > maximum:
    raise DeviceAcceptanceError(f"{name} is out of range")
  return value


def _text(value: object, name: str, maximum: int, *, empty: bool = False) -> str:
  try:
    size = len(value.encode("utf-8")) if type(value) is str else 0
  except UnicodeEncodeError as error:
    raise DeviceAcceptanceError(f"{name} is invalid") from error
  if type(value) is not str or (not empty and not value) or "\x00" in value or size > maximum:
    raise DeviceAcceptanceError(f"{name} is invalid")
  return value


def _optional_hash(value: object, name: str) -> str:
  if value == "":
    return ""
  if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
    raise DeviceAcceptanceError(f"{name} must be empty or a SHA-256")
  return value


def _finite(value: object, name: str) -> float:
  if type(value) not in (int, float) or not math.isfinite(float(value)):
    raise DeviceAcceptanceError(f"{name} must be finite")
  result = float(value)
  if result < 0.0:
    raise DeviceAcceptanceError(f"{name} must be nonnegative")
  return result


def _canonical_json(value: object) -> str:
  return json.dumps(
    value,
    allow_nan=False,
    separators=(",", ":"),
    sort_keys=True,
  )


@dataclass(frozen=True, slots=True)
class DeviceAcceptanceReceipt:
  route_evidence_sha256s: tuple[str, ...]
  device_type: str
  vehicle_identity: str
  controller_architecture: str
  source_openpilot_commit: str
  opendbc_commit: str
  panda_commit: str
  live_artifact_sha256: str
  runtime_identity_sha256: str
  profile_sha256: str
  controller_policy_sha256: str
  horizon_policy_sha256: str
  sample_count: int
  percentile_method: str
  compute_p50_seconds: float
  compute_p90_seconds: float
  compute_p99_seconds: float
  compute_max_seconds: float
  drop_count: int
  failure_counts: tuple[tuple[str, int], ...]
  schema_version: int = DEVICE_ACCEPTANCE_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if type(self.schema_version) is not int or self.schema_version != DEVICE_ACCEPTANCE_SCHEMA_VERSION:
      raise DeviceAcceptanceError("device receipt schema is incompatible")
    if (
      type(self.route_evidence_sha256s) is not tuple
      or not self.route_evidence_sha256s
      or len(self.route_evidence_sha256s) > MAX_DEVICE_ACCEPTANCE_ROUTES
      or any(type(value) is not str or _SHA256_RE.fullmatch(value) is None for value in self.route_evidence_sha256s)
      or self.route_evidence_sha256s != tuple(sorted(set(self.route_evidence_sha256s)))
    ):
      raise DeviceAcceptanceError("route evidence identities are invalid")
    _text(self.device_type, "device type", 64)
    _text(self.vehicle_identity, "vehicle identity", 4096)
    _text(
      self.controller_architecture,
      "controller architecture",
      128,
      empty=True,
    )
    for name, value in (
      ("source openpilot commit", self.source_openpilot_commit),
      ("opendbc commit", self.opendbc_commit),
      ("panda commit", self.panda_commit),
    ):
      if type(value) is not str or _COMMIT_RE.fullmatch(value) is None:
        raise DeviceAcceptanceError(f"{name} must be a full commit")
    for name, value in (
      ("live artifact hash", self.live_artifact_sha256),
      ("runtime identity", self.runtime_identity_sha256),
      ("profile hash", self.profile_sha256),
      ("controller policy hash", self.controller_policy_sha256),
      ("horizon policy hash", self.horizon_policy_sha256),
    ):
      _optional_hash(value, name)
    _uint(self.sample_count, "sample count")
    if self.percentile_method != PERCENTILE_METHOD:
      raise DeviceAcceptanceError("percentile method is unsupported")
    percentiles = tuple(
      _finite(value, name)
      for name, value in (
        ("compute p50", self.compute_p50_seconds),
        ("compute p90", self.compute_p90_seconds),
        ("compute p99", self.compute_p99_seconds),
        ("compute max", self.compute_max_seconds),
      )
    )
    if percentiles != tuple(sorted(percentiles)):
      raise DeviceAcceptanceError("compute percentiles are not ordered")
    _uint(self.drop_count, "drop count")
    if (
      type(self.failure_counts) is not tuple
      or any(type(item) is not tuple or len(item) != 2 for item in self.failure_counts)
      or tuple(item[0] for item in self.failure_counts) != FAILURE_REASONS
    ):
      raise DeviceAcceptanceError("failure counts are not canonical")
    for name, count in self.failure_counts:
      _uint(count, f"{name} failures")

  @property
  def passed(self) -> bool:
    return (
      self.sample_count > 0
      and self.device_type in COMMA_DEVICE_TYPES
      and self.controller_architecture == MODULAR_ARCHITECTURE
      and all(
        (
          self.runtime_identity_sha256,
          self.profile_sha256,
          self.controller_policy_sha256,
          self.horizon_policy_sha256,
        )
      )
      and self.compute_p50_seconds > 0.0
      and self.compute_max_seconds < MAX_DEVICE_COMPUTE_SECONDS
      and self.drop_count == 0
      and all(count == 0 for _, count in self.failure_counts)
    )

  def failure_count(self, reason: str) -> int:
    return dict(self.failure_counts)[reason]

  def to_param(self) -> dict[str, object]:
    return {
      "schemaVersion": self.schema_version,
      "routeEvidenceSha256s": list(self.route_evidence_sha256s),
      "deviceType": self.device_type,
      "vehicleIdentity": self.vehicle_identity,
      "controllerArchitecture": self.controller_architecture,
      "sourceOpenpilotCommit": self.source_openpilot_commit,
      "opendbcCommit": self.opendbc_commit,
      "pandaCommit": self.panda_commit,
      "liveArtifactSha256": self.live_artifact_sha256,
      "runtimeIdentitySha256": self.runtime_identity_sha256,
      "profileSha256": self.profile_sha256,
      "controllerPolicySha256": self.controller_policy_sha256,
      "horizonPolicySha256": self.horizon_policy_sha256,
      "sampleCount": self.sample_count,
      "percentileMethod": self.percentile_method,
      "computeP50Seconds": self.compute_p50_seconds,
      "computeP90Seconds": self.compute_p90_seconds,
      "computeP99Seconds": self.compute_p99_seconds,
      "computeMaxSeconds": self.compute_max_seconds,
      "dropCount": self.drop_count,
      "failureCounts": dict(self.failure_counts),
    }

  def to_json(self) -> str:
    return _canonical_json(self.to_param())

  @property
  def sha256(self) -> str:
    return hashlib.sha256(self.to_json().encode()).hexdigest()

  @classmethod
  def from_json(cls, encoded: object) -> DeviceAcceptanceReceipt:
    try:
      encoded_size = len(encoded.encode("utf-8")) if type(encoded) is str else 0
    except UnicodeEncodeError as error:
      raise DeviceAcceptanceError("device receipt JSON is invalid") from error
    if type(encoded) is not str or encoded_size > 64 * 1024:
      raise DeviceAcceptanceError("device receipt JSON is invalid")
    try:
      payload = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
      raise DeviceAcceptanceError("device receipt JSON is invalid") from error
    if type(payload) is not dict or frozenset(payload) != _RECEIPT_KEYS:
      raise DeviceAcceptanceError("device receipt keys are not canonical")
    routes = payload["routeEvidenceSha256s"]
    failures = payload["failureCounts"]
    if type(routes) is not list or type(failures) is not dict:
      raise DeviceAcceptanceError("device receipt collections are invalid")
    receipt = cls(
      route_evidence_sha256s=tuple(routes),
      device_type=payload["deviceType"],
      vehicle_identity=payload["vehicleIdentity"],
      controller_architecture=payload["controllerArchitecture"],
      source_openpilot_commit=payload["sourceOpenpilotCommit"],
      opendbc_commit=payload["opendbcCommit"],
      panda_commit=payload["pandaCommit"],
      live_artifact_sha256=payload["liveArtifactSha256"],
      runtime_identity_sha256=payload["runtimeIdentitySha256"],
      profile_sha256=payload["profileSha256"],
      controller_policy_sha256=payload["controllerPolicySha256"],
      horizon_policy_sha256=payload["horizonPolicySha256"],
      sample_count=payload["sampleCount"],
      percentile_method=payload["percentileMethod"],
      compute_p50_seconds=payload["computeP50Seconds"],
      compute_p90_seconds=payload["computeP90Seconds"],
      compute_p99_seconds=payload["computeP99Seconds"],
      compute_max_seconds=payload["computeMaxSeconds"],
      drop_count=payload["dropCount"],
      failure_counts=tuple((reason, failures.get(reason)) for reason in FAILURE_REASONS),
      schema_version=payload["schemaVersion"],
    )
    if frozenset(failures) != frozenset(FAILURE_REASONS) or receipt.to_json() != encoded:
      raise DeviceAcceptanceError("device receipt JSON is not canonical")
    return receipt


def _nearest_rank(values: list[float], fraction: float) -> float:
  if not values:
    return 0.0
  return values[math.ceil(fraction * len(values)) - 1]


def build_device_acceptance_receipt(
  paths: Sequence[str | Path],
) -> DeviceAcceptanceReceipt:
  if isinstance(paths, (str, Path)):
    raise DeviceAcceptanceError("route evidence paths must be a sequence")
  selected_paths = tuple(Path(path) for path in paths)
  if not 1 <= len(selected_paths) <= MAX_DEVICE_ACCEPTANCE_ROUTES:
    raise DeviceAcceptanceError("route evidence path count is out of range")

  failures = dict.fromkeys(FAILURE_REASONS, 0)
  identities = {
    name: set()
    for name in (
      "device_type",
      "vehicle_identity",
      "controller_architecture",
      "source_openpilot_commit",
      "opendbc_commit",
      "panda_commit",
      "live_artifact_sha256",
      "runtime_identity_sha256",
      "profile_sha256",
      "controller_policy_sha256",
      "horizon_policy_sha256",
    )
  }
  route_hashes: set[str] = set()
  compute_samples: list[float] = []
  sample_count = 0
  drop_count = 0

  for path in selected_paths:
    with RouteEvidenceStreamReader(path) as reader:
      route_sha256 = reader.summary.sha256
      if route_sha256 in route_hashes:
        continue
      if len(route_hashes) >= MAX_DEVICE_ACCEPTANCE_ROUTES:
        raise DeviceAcceptanceError("unique route evidence count is out of range")
      route_hashes.add(route_sha256)
      source = reader.summary.source_identity
      route_identity = {
        "device_type": source.device_type,
        "vehicle_identity": source.vehicle_identity,
        "controller_architecture": source.controller_architecture,
        "source_openpilot_commit": source.source_superproject_commit,
        "opendbc_commit": source.source_opendbc_commit,
        "panda_commit": source.source_panda_commit,
        "live_artifact_sha256": source.live_artifact_sha256,
        "runtime_identity_sha256": source.recorded_runtime_identity_sha256,
        "profile_sha256": source.recorded_profile_sha256,
        "controller_policy_sha256": source.recorded_controller_policy_sha256,
        "horizon_policy_sha256": source.recorded_horizon_policy_sha256,
      }
      for name, value in route_identity.items():
        identities[name].add(value)
      if source.device_type not in COMMA_DEVICE_TYPES:
        failures["non_comma_device"] += 1
      if source.controller_architecture != MODULAR_ARCHITECTURE:
        failures["controller_architecture_mismatch"] += 1
      if source.recorded_source_openpilot_commit != source.source_superproject_commit or source.recorded_opendbc_commit != source.source_opendbc_commit:
        failures["source_identity_mismatch"] += 1
      if source.recorded_runtime_identity_sha256 != source.runtime_identity:
        failures["runtime_identity_mismatch"] += 1

      pre_poll_drop_count = len(source.pre_poll_dropped_timestamps_ns)
      if source.unresolved_witness_count:
        failures["unresolved_witness_summary_nonzero"] += source.unresolved_witness_count
      if source.gap_count:
        failures["route_gap_summary_nonzero"] += source.gap_count
      if pre_poll_drop_count:
        failures["pre_poll_witness_dropped"] += pre_poll_drop_count

      previous_witness_ns: int | None = None
      bootstrap_allowed = False
      route_drop_count = 0
      active_interval_modular: bool | None = None
      for witness in reader.iter_control_witnesses():
        if not witness.lateral_active:
          clean_inactive_boundary = (
            not witness.modular_active
            and witness.message_valid
            and witness.modular_selection == int(ControllerSelection.STOCK)
            and not witness.modular_selection_bound
          )
          if clean_inactive_boundary:
            bootstrap_allowed = True
            active_interval_modular = None
            previous_witness_ns = None
          else:
            failures["modular_binding_invalid"] += 1
            bootstrap_allowed = False
          continue

        claims_modular = witness.modular_selection == int(ControllerSelection.MODULAR) or witness.modular_active or witness.modular_selection_bound
        interval_binding_invalid = active_interval_modular is not None and active_interval_modular != claims_modular
        if active_interval_modular is None:
          active_interval_modular = claims_modular
        if not claims_modular:
          if interval_binding_invalid:
            failures["modular_binding_invalid"] += 1
          bootstrap_allowed = False
          continue
        exact_binding = witness.modular_selection == int(ControllerSelection.MODULAR) and witness.modular_active and witness.modular_selection_bound
        if not exact_binding or interval_binding_invalid:
          failures["modular_binding_invalid"] += 1

        sample_count += 1
        allow_bootstrap = bootstrap_allowed
        bootstrap_allowed = False
        compute = witness.modular_compute_time_seconds
        compute_samples.append(compute)
        if not witness.modular_telemetry_available:
          failures["telemetry_unavailable"] += 1
        if not math.isfinite(compute) or compute <= 0.0:
          failures["compute_invalid"] += 1
        elif compute >= MAX_DEVICE_COMPUTE_SECONDS:
          failures["compute_budget_exceeded"] += 1
        if witness.modular_control_witness_mono_ns <= 0:
          failures["control_witness_invalid"] += 1
        if witness.modular_intent_status != int(IntentStatusCode.OK):
          failures["intent_not_ok"] += 1
        if witness.modular_safety_state != int(LiveSafetyState.OK):
          failures["safety_not_ok"] += 1
        if witness.modular_invalid_frames != 0:
          failures["invalid_frames_nonzero"] += 1
        if witness.modular_recovery_ok_frames != 0:
          failures["recovery_frames_nonzero"] += 1
        if not witness.message_valid or not witness.modular_controls_valid:
          failures["controls_state_invalid"] += 1
        for valid, reason in (
          (witness.modular_car_control_valid, "car_control_invalid"),
          (witness.modular_vehicle_state_valid, "vehicle_state_invalid"),
          (witness.modular_live_parameters_valid, "live_parameters_invalid"),
          (witness.modular_horizon_valid, "horizon_invalid"),
        ):
          if not valid:
            failures[reason] += 1
        if not witness.modular_control_cadence_valid:
          failures["control_cadence_invalid"] += 1
        if witness.modular_adapter_exception:
          failures["adapter_exception"] += 1
        if not witness.modular_production_envelope_verified:
          failures["production_envelope_unverified"] += 1
        # Valid request suppression is safe at runtime, but it is not clean
        # actuator-correspondence evidence for promotion.
        exact_correspondence = (
          witness.torque_output_can_valid
          and witness.steering_request_active
          and witness.steering_request_active_valid
          and witness.steering_request_fault_avoidance_counter_valid
        )
        if witness.modular_final_count_match_valid:
          exact_correspondence = exact_correspondence and (
            witness.modular_final_expected_counts == witness.torque_output_can_count
            and witness.modular_final_count_residual == 0
            and not witness.modular_final_limiter_altered
          )
        else:
          exact_correspondence = exact_correspondence and (
            allow_bootstrap
            and witness.modular_final_expected_counts == 0
            and witness.modular_final_count_residual == 0
            and not witness.modular_final_limiter_altered
          )
        if not exact_correspondence:
          failures["actuator_correspondence_invalid"] += 1

        frame_drops = 0
        if witness.gap_from_previous or not witness.modular_control_cadence_valid:
          failures["recorded_cadence_gap"] += 1
          frame_drops = 1
        if previous_witness_ns is not None:
          interval_ns = witness.modular_control_witness_mono_ns - previous_witness_ns
          if interval_ns <= 0 or interval_ns > MAX_RECORDED_FRAME_GAP_NS:
            failures["inferred_cadence_gap"] += 1
            inferred = 1 if interval_ns <= 0 else max(1, round(interval_ns / NOMINAL_CONTROL_PERIOD_NS) - 1)
            frame_drops = max(frame_drops, inferred)
        route_drop_count += frame_drops
        previous_witness_ns = witness.modular_control_witness_mono_ns if witness.modular_control_witness_mono_ns > 0 else None
      drop_count += pre_poll_drop_count + max(
        source.gap_count,
        route_drop_count,
      )

  def exact_identity(
    name: str,
    reason: str,
    *,
    optional: bool = False,
    retain_one: bool = False,
  ) -> str:
    values = identities[name]
    if len(values) == 1:
      value = next(iter(values))
      if value or optional:
        return value
    failures[reason] += 1
    return min(values) if retain_one and values else ""

  device_type = exact_identity(
    "device_type",
    "device_identity_mismatch",
    retain_one=True,
  )
  vehicle_identity = exact_identity(
    "vehicle_identity",
    "vehicle_identity_mismatch",
    retain_one=True,
  )
  controller_architecture = exact_identity(
    "controller_architecture",
    "controller_architecture_mismatch",
  )
  source_openpilot_commit = exact_identity(
    "source_openpilot_commit",
    "source_identity_mismatch",
    retain_one=True,
  )
  opendbc_commit = exact_identity(
    "opendbc_commit",
    "source_identity_mismatch",
    retain_one=True,
  )
  panda_commit = exact_identity(
    "panda_commit",
    "source_identity_mismatch",
    retain_one=True,
  )
  live_artifact_sha256 = exact_identity(
    "live_artifact_sha256",
    "live_artifact_identity_mismatch",
    optional=True,
  )
  runtime_identity_sha256 = exact_identity(
    "runtime_identity_sha256",
    "runtime_identity_mismatch",
  )
  profile_sha256 = exact_identity("profile_sha256", "profile_identity_mismatch")
  controller_policy_sha256 = exact_identity(
    "controller_policy_sha256",
    "controller_policy_identity_mismatch",
  )
  horizon_policy_sha256 = exact_identity(
    "horizon_policy_sha256",
    "horizon_policy_identity_mismatch",
  )
  if sample_count == 0:
    failures["no_eligible_samples"] += 1

  compute_samples.sort()
  return DeviceAcceptanceReceipt(
    route_evidence_sha256s=tuple(sorted(route_hashes)),
    device_type=device_type,
    vehicle_identity=vehicle_identity,
    controller_architecture=controller_architecture,
    source_openpilot_commit=source_openpilot_commit,
    opendbc_commit=opendbc_commit,
    panda_commit=panda_commit,
    live_artifact_sha256=live_artifact_sha256,
    runtime_identity_sha256=runtime_identity_sha256,
    profile_sha256=profile_sha256,
    controller_policy_sha256=controller_policy_sha256,
    horizon_policy_sha256=horizon_policy_sha256,
    sample_count=sample_count,
    percentile_method=PERCENTILE_METHOD,
    compute_p50_seconds=_nearest_rank(compute_samples, 0.50),
    compute_p90_seconds=_nearest_rank(compute_samples, 0.90),
    compute_p99_seconds=_nearest_rank(compute_samples, 0.99),
    compute_max_seconds=(compute_samples[-1] if compute_samples else 0.0),
    drop_count=drop_count,
    failure_counts=tuple((reason, failures[reason]) for reason in FAILURE_REASONS),
  )

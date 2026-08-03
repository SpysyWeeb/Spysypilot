"""Bounded cross-architecture proof for BLaTv2 route preparation.

The workstation still prepares and byte-compares the *complete* route twice.
This module projects a source-selected set of whole rlog segments from that
same canonical full-route preparation context.  The projection covers the
production extractor, canonical race reconstruction, measured-frame/vehicle-
model path, route-evidence encoders, and physical spool encoder.  It
deliberately does not run a learner or fit a profile.

Selection is a pure function of the immutable route manifest.  Preparing a
selected segment in isolation is intentionally forbidden: SubMaster health,
the canonical input race, sparse-publication indices, and intervention state
all carry across rlog segment boundaries.  Reconstructing them from a reset
segment would certify a different input timeline.  The selected bytes and
witness populations remain bounded, while the preparation context uses the
same existing full-route bound as the artifact it proves.  The physical proof
plane follows the canonical measured-frame input contract without requiring
optional controller context.  The behavior plane proves the recorded
controller source and its full context.  A third, controller-independent
scenario plane proves only the inputs consumed by counterfactual replay.  Each
plane owns its counts, timestamps, exclusions, and encoded digest; complete
source planes remain covered by their production section hashes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from typing import Any, Final

from openpilot.selfdrive.controls.lib.blatv2.learning_backfill import (
  BuildDescriptor,
  BuildDescriptorRegistry,
  PreparedRoute,
  RouteCandidate,
  RouteSegment,
  prepare_route,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill_spool import (
  _encode_frame,
)
from openpilot.selfdrive.controls.lib.blatv2.preparation_frame import (
  MeasuredLearningFrame,
)
from openpilot.selfdrive.controls.lib.blatv2.route_evidence import (
  ControlsWitness,
  LateralManeuverPlanPublication,
  LiveDelayPublication,
  LiveTorqueParametersPublication,
  ModelPublication,
  RouteEvidenceArtifact,
  _encode_controls,
  _encode_delay,
  _encode_events,
  _encode_maneuvers,
  _encode_models,
  _encode_torque,
)
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  RuntimeVehicleBundle,
)


CERTIFICATION_VECTOR_SCHEMA_VERSION: Final = 5
CERTIFICATION_VECTOR_MAGIC: Final = b"BLATCV01"
CERTIFICATION_VECTOR_MAX_SEGMENTS: Final = 3
CERTIFICATION_VECTOR_MAX_COMPRESSED_BYTES: Final = 96 * 1024 * 1024
CERTIFICATION_VECTOR_MAX_CONTROLS_WITNESSES: Final = 30_000
CERTIFICATION_VECTOR_MAX_BYTES: Final = 64 * 1024
_VECTOR_DOMAIN: Final = b"blatv2-cross-architecture-vector-v5\0"
# Sampling itself did not change in schema 5.  Keep the reviewed v4 sample
# stable so a proof-semantics upgrade does not silently select different road
# data; the signed selection *identity* below is independently v5-domain-bound.
_SELECTION_DOMAIN: Final = b"blatv2-certification-segment-selection-v4\0"
_SCENARIO_PROOF_DOMAIN: Final = b"blatv2-certification-scenario-proof-v5\0"
_HEADER: Final = struct.Struct("<8sHHI")
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")


class CertificationVectorError(RuntimeError):
  """The bounded certification vector cannot be built or authenticated."""


@dataclass(frozen=True, slots=True)
class CertificationVector:
  manifest: dict[str, object]
  canonical_bytes: bytes
  sha256: str

  @classmethod
  def from_manifest(cls, manifest: dict[str, object]) -> CertificationVector:
    _validate_manifest(manifest)
    encoded = _canonical_json(manifest)
    canonical = _HEADER.pack(
      CERTIFICATION_VECTOR_MAGIC,
      CERTIFICATION_VECTOR_SCHEMA_VERSION,
      0,
      len(encoded),
    ) + encoded
    if len(canonical) > CERTIFICATION_VECTOR_MAX_BYTES:
      raise CertificationVectorError("certification vector exceeds its bound")
    return cls(
      manifest=dict(manifest),
      canonical_bytes=canonical,
      sha256=hashlib.sha256(canonical).hexdigest(),
    )

  @classmethod
  def from_bytes(cls, value: bytes | bytearray | memoryview) -> CertificationVector:
    encoded = bytes(value)
    if len(encoded) < _HEADER.size or len(encoded) > CERTIFICATION_VECTOR_MAX_BYTES:
      raise CertificationVectorError("certification vector size is invalid")
    magic, version, reserved, length = _HEADER.unpack_from(encoded)
    if (
      magic != CERTIFICATION_VECTOR_MAGIC
      or version != CERTIFICATION_VECTOR_SCHEMA_VERSION
      or reserved != 0
      or length != len(encoded) - _HEADER.size
    ):
      raise CertificationVectorError("certification vector header is invalid")
    payload = encoded[_HEADER.size:]
    try:
      manifest: object = json.loads(payload)
    except (
      UnicodeDecodeError,
      json.JSONDecodeError,
      RecursionError,
      ValueError,
    ) as exc:
      raise CertificationVectorError("certification vector JSON is invalid") from exc
    if type(manifest) is not dict:
      raise CertificationVectorError("certification vector is not canonical")
    _validate_json_depth(manifest)
    if payload != _canonical_json(manifest):
      raise CertificationVectorError("certification vector is not canonical")
    rebuilt = cls.from_manifest(manifest)
    if rebuilt.canonical_bytes != encoded:
      raise CertificationVectorError("certification vector round trip changed")
    return rebuilt


def _canonical_json(value: object) -> bytes:
  try:
    return json.dumps(
      value,
      allow_nan=False,
      separators=(",", ":"),
      sort_keys=True,
    ).encode("utf-8")
  except (TypeError, ValueError) as exc:
    raise CertificationVectorError("certification vector is not JSON-safe") from exc


def _source_manifest(route: RouteCandidate) -> list[dict[str, object]]:
  if type(route) is not RouteCandidate or not route.segments:
    raise CertificationVectorError("certification route is invalid")
  result: list[dict[str, object]] = []
  previous = -1
  for segment in route.segments:
    if (
      type(segment) is not RouteSegment
      or segment.index <= previous
      or segment.size_bytes <= 0
      or _SHA256_RE.fullmatch(segment.sha256) is None
    ):
      raise CertificationVectorError("certification route manifest is invalid")
    previous = segment.index
    result.append({
      "index": segment.index,
      "sha256": segment.sha256,
      "size_bytes": segment.size_bytes,
    })
  return result


def certification_vector_selection(
  route: RouteCandidate,
) -> tuple[RouteSegment, ...]:
  """Select at most three whole segments from immutable source identity.

  The normal shape is beginning, one hash-selected interior segment, and end.
  If that exceeds the byte cap, selection falls back to the smallest whole
  segments under the pinned ``(size, sha256, index)`` ranking while retaining
  the first segment whenever it fits.  Raw byte prefixes are never used.
  """
  manifest = _source_manifest(route)
  segments = route.segments
  desired: list[RouteSegment] = [segments[0]]
  if len(segments) > 2:
    digest = hashlib.sha256(
      _SELECTION_DOMAIN + _canonical_json({
        "route_name": route.route_name,
        "segments": manifest,
      }),
    ).digest()
    interior = segments[1:-1]
    desired.append(interior[int.from_bytes(digest[:8], "big") % len(interior)])
  if len(segments) > 1:
    desired.append(segments[-1])
  desired = list({segment.index: segment for segment in desired}.values())
  desired.sort(key=lambda segment: segment.index)
  if (
    len(desired) <= CERTIFICATION_VECTOR_MAX_SEGMENTS
    and sum(segment.size_bytes for segment in desired)
    <= CERTIFICATION_VECTOR_MAX_COMPRESSED_BYTES
  ):
    return tuple(desired)

  selected: list[RouteSegment] = []
  remaining = CERTIFICATION_VECTOR_MAX_COMPRESSED_BYTES
  first = segments[0]
  if first.size_bytes > remaining:
    # InitData is repeated in every segment, but many recording builds emit
    # CarParams only in segment zero.  A vector that omits segment zero cannot
    # authenticate the route-wide CarParams seed used by later segments.
    raise CertificationVectorError(
      "route provenance segment exceeds the certification-vector byte bound",
    )
  selected.append(first)
  remaining -= first.size_bytes
  for segment in sorted(
    (segment for segment in segments if segment.index != first.index),
    key=lambda item: (item.size_bytes, item.sha256, item.index),
  ):
    if len(selected) >= CERTIFICATION_VECTOR_MAX_SEGMENTS:
      break
    if segment.size_bytes <= remaining:
      selected.append(segment)
      remaining -= segment.size_bytes
  if not selected:
    raise CertificationVectorError(
      "no whole route segment fits the certification-vector byte bound",
    )
  selected.sort(key=lambda segment: segment.index)
  return tuple(selected)


def certification_vector_selection_identity(
  route: RouteCandidate,
  selected: tuple[RouteSegment, ...] | None = None,
) -> str:
  chosen = certification_vector_selection(route) if selected is None else selected
  return hashlib.sha256(_canonical_json({
    "domain": "blatv2-certification-vector-selection-v5",
    "route_name": route.route_name,
    "source_segments": _source_manifest(route),
    "selected_segments": [
      {
        "index": segment.index,
        "sha256": segment.sha256,
        "size_bytes": segment.size_bytes,
      }
      for segment in chosen
    ],
  })).hexdigest()


def _physical_exclusion_reason(
  witness: ControlsWitness,
  frame: MeasuredLearningFrame,
) -> str | None:
  """Apply the canonical measured-frame input boundary, not behavior policy.

  ``MeasuredLearningFrame.inputs_valid`` is produced by
  ``_measured_frame_from_join`` from the exact source-message validity,
  reconstructed clocks, and carControl pairing used by the physical learner.
  Segment-local cadence is carried by the witness and must be applied here as
  well: it is deliberately not folded into the frame's route-global wire
  representation.
  """
  if witness.race_unresolved:
    return "race_unresolved"
  if witness.gap_from_previous:
    return "segment_local_gap"
  if not witness.car_control_paired:
    return "car_control_context_missing"
  if type(frame) is not MeasuredLearningFrame:
    raise CertificationVectorError("physical frame type is malformed")
  if witness.inputs_valid != frame.inputs_valid:
    raise CertificationVectorError("physical witness/frame validity disagrees")
  if not frame.inputs_valid:
    return "physical_inputs_invalid"
  return None


def _behavior_exclusion_reason(
  witness: ControlsWitness,
  frame: MeasuredLearningFrame,
) -> str | None:
  physical_reason = _physical_exclusion_reason(witness, frame)
  if physical_reason is not None:
    if physical_reason == "physical_inputs_invalid":
      return "physical_plane_inputs_invalid"
    return f"physical_plane_{physical_reason}"
  if not witness.model_link_valid:
    return "model_context_missing"
  if not witness.live_torque_parameters_available:
    return "live_torque_context_missing"
  if not witness.live_delay_available:
    return "live_delay_context_missing"
  if not witness.maneuver_plan_available:
    return "maneuver_context_missing"
  return None


def _linked_scenario_publication(
  values: tuple[Any, ...],
  index: int,
  available: bool,
  witness_mono_ns: int,
) -> Any | None:
  if available != (index >= 0):
    return None
  if index < 0:
    return None
  if index >= len(values):
    raise CertificationVectorError("scenario publication index is out of range")
  publication = values[index]
  if publication.mono_time_ns > witness_mono_ns:
    return None
  return publication


def _scenario_model_payload_valid(model: ModelPublication) -> bool:
  return (
    model.frame_id >= 0
    and 0 <= model.timestamp_eof_ns <= model.mono_time_ns
    and math.isfinite(model.desired_curvature_time_s)
    and model.desired_curvature_time_s >= 0.0
    and model.native_grid_valid
    and bool(model.plan_times)
    and math.isfinite(model.plan_times[0])
    and model.plan_times[0] >= 0.0
  )


def _scenario_exclusion_reason(
  witness: ControlsWitness,
  frame: MeasuredLearningFrame,
  artifact: RouteEvidenceArtifact,
) -> str | None:
  physical_reason = _physical_exclusion_reason(witness, frame)
  if physical_reason is not None:
    if physical_reason == "physical_inputs_invalid":
      return "physical_plane_inputs_invalid"
    return f"physical_plane_{physical_reason}"
  if not witness.message_valid:
    return "scenario_message_invalid"

  model = _linked_scenario_publication(
    artifact.model_publications,
    witness.model_publication_index,
    witness.model_link_valid,
    witness.mono_time_ns,
  )
  if not isinstance(model, ModelPublication):
    return "model_context_missing"
  if not model.message_valid:
    return "model_message_invalid"
  if not witness.model_message_alive:
    return "model_not_alive"
  if not _scenario_model_payload_valid(model):
    return "model_native_grid_invalid"
  if not witness.torque_output_can_valid:
    return "applied_torque_context_missing"

  live_torque = _linked_scenario_publication(
    artifact.live_torque_parameters,
    witness.live_torque_parameters_index,
    witness.live_torque_parameters_available,
    witness.mono_time_ns,
  )
  if not isinstance(live_torque, LiveTorqueParametersPublication):
    return "live_torque_context_missing"
  if not witness.live_torque_parameters_health_exact:
    return "live_torque_health_inexact"
  if (
    not live_torque.message_valid
    or (
      witness.live_torque_parameters_checks_passed
      and live_torque.use_params
      and (
        not math.isfinite(live_torque.lat_accel_factor)
        or not math.isfinite(live_torque.lat_accel_offset)
        or not math.isfinite(live_torque.friction)
        or live_torque.lat_accel_factor <= 0.0
        or live_torque.friction < 0.0
      )
    )
  ):
    return "live_torque_payload_invalid"

  live_delay = _linked_scenario_publication(
    artifact.live_delays,
    witness.live_delay_index,
    witness.live_delay_available,
    witness.mono_time_ns,
  )
  if not isinstance(live_delay, LiveDelayPublication):
    return "live_delay_context_missing"
  if (
    not live_delay.message_valid
    or not math.isfinite(live_delay.lateral_delay_s)
    or live_delay.lateral_delay_s < 0.0
  ):
    return "live_delay_payload_invalid"

  maneuver = _linked_scenario_publication(
    artifact.lateral_maneuver_plans,
    witness.lateral_maneuver_plan_index,
    witness.maneuver_plan_available,
    witness.mono_time_ns,
  )
  maneuver_link_present = (
    witness.maneuver_plan_available
    or witness.lateral_maneuver_plan_index >= 0
  )
  if maneuver_link_present and not isinstance(
    maneuver,
    LateralManeuverPlanPublication,
  ):
    return "maneuver_context_invalid"
  if witness.lateral_active and maneuver is not None and maneuver.message_valid:
    return "active_maneuver_override"
  return None


def _segment_vector(
  *,
  segment: RouteSegment,
  prepared: PreparedRoute,
  include_pre_poll_exclusions: bool = True,
  include_segment_context_exclusions: bool = True,
) -> dict[str, object]:
  """Project one segment from a canonical full-route preparation.

  The encoded controls are the unmodified, route-global ``ControlsWitness``
  records.  In particular, publication indices, physical-record indices,
  SubMaster health, and transition state are never rebased at the segment
  boundary.  This makes the proof byte-comparable with the immutable complete
  route artifact rather than with a second, segment-reset reconstruction.

  The two inclusion flags exist for the single-segment unit fixture and for
  assigning route-boundary exclusions exactly once.  Production full-route
  construction enables pre-poll exclusions only on the structural first
  segment and never creates segment-local context exclusions.
  """
  if (
    type(include_pre_poll_exclusions) is not bool
    or type(include_segment_context_exclusions) is not bool
  ):
    raise CertificationVectorError("segment exclusion ownership is invalid")
  artifact = prepared.route_evidence
  if type(artifact) is not RouteEvidenceArtifact:
    raise CertificationVectorError("segment preparation lacks route evidence")
  all_controls = artifact.control_witnesses
  all_frames = prepared.frames
  if len(all_controls) != len(all_frames):
    raise CertificationVectorError("route control/frame populations disagree")
  pairs = tuple(
    (witness, frame)
    for witness, frame in zip(all_controls, all_frames, strict=True)
    if witness.segment_index == segment.index
  )
  if not pairs:
    raise CertificationVectorError(
      "selected segment has no canonical controls witnesses",
    )
  controls = tuple(pair[0] for pair in pairs)
  frames = tuple(pair[1] for pair in pairs)

  physical_excluded = Counter[str]()
  behavior_excluded = Counter[str]()
  scenario_excluded = Counter[str]()
  retained_physical = 0
  retained_behavior = 0
  retained_scenario = 0
  retained_active_scenario = 0
  physical_digest = hashlib.sha256()
  physical_controls_digest = hashlib.sha256()
  behavior_controls_digest = hashlib.sha256()
  scenario_controls_digest = hashlib.sha256()
  physical_first_mono_ns: int | None = None
  physical_last_mono_ns: int | None = None
  behavior_first_mono_ns: int | None = None
  behavior_last_mono_ns: int | None = None
  scenario_first_mono_ns: int | None = None
  scenario_last_mono_ns: int | None = None
  for witness, frame in zip(controls, frames, strict=True):
    physical_reason = _physical_exclusion_reason(witness, frame)
    if physical_reason is None:
      retained_physical += 1
      physical_digest.update(_encode_frame(frame))
      physical_controls_digest.update(_encode_controls((witness,)))
      physical_first_mono_ns = (
        witness.mono_time_ns
        if physical_first_mono_ns is None
        else physical_first_mono_ns
      )
      physical_last_mono_ns = witness.mono_time_ns
    else:
      physical_excluded[physical_reason] += 1

    behavior_reason = _behavior_exclusion_reason(witness, frame)
    if behavior_reason is None:
      retained_behavior += 1
      behavior_controls_digest.update(_encode_controls((witness,)))
      behavior_first_mono_ns = (
        witness.mono_time_ns
        if behavior_first_mono_ns is None
        else behavior_first_mono_ns
      )
      behavior_last_mono_ns = witness.mono_time_ns
    else:
      behavior_excluded[behavior_reason] += 1

    scenario_reason = _scenario_exclusion_reason(witness, frame, artifact)
    if scenario_reason is None:
      retained_scenario += 1
      retained_active_scenario += int(witness.lateral_active)
      scenario_controls_digest.update(_encode_controls((witness,)))
      scenario_first_mono_ns = (
        witness.mono_time_ns
        if scenario_first_mono_ns is None
        else scenario_first_mono_ns
      )
      scenario_last_mono_ns = witness.mono_time_ns
    else:
      scenario_excluded[scenario_reason] += 1
  if retained_physical == 0:
    raise CertificationVectorError(
      "certification segment has no valid physical-learning witnesses",
    )

  source = artifact.source_identity
  if source.behavior_eligible and retained_behavior == 0:
    raise CertificationVectorError(
      "behavior-eligible segment has no certification witnesses",
    )
  pre_poll_count = (
    prepared.pre_poll_dropped_count
    if include_pre_poll_exclusions
    else 0
  )
  source_excluded = Counter[str]()
  if pre_poll_count:
    source_excluded["pre_poll_controls"] += pre_poll_count
  segment_context_count = (
    prepared.segment_local_measurement_context_dropped_count
    if include_segment_context_exclusions
    else 0
  )
  if segment_context_count:
    source_excluded["segment_local_measurement_context"] += segment_context_count

  models = tuple(
    value
    for value in artifact.model_publications
    if value.segment_index == segment.index
  )
  torque = tuple(
    value
    for value in artifact.live_torque_parameters
    if value.segment_index == segment.index
  )
  delays = tuple(
    value
    for value in artifact.live_delays
    if value.segment_index == segment.index
  )
  maneuvers = tuple(
    value
    for value in artifact.lateral_maneuver_plans
    if value.segment_index == segment.index
  )
  events = tuple(
    value
    for value in artifact.event_locators
    if value.segment_index == segment.index
  )
  coverage = {
    "controls_total": len(controls),
    "driving_events": len(events),
    "lateral_maneuver_plans": len(maneuvers),
    "live_delays": len(delays),
    "live_torque_parameters": len(torque),
    "models": len(models),
    "physical_frames_total": len(frames),
  }
  return {
    "behavior_plane": {
      "controls_retained": retained_behavior,
      "encoded_controls_sha256": behavior_controls_digest.hexdigest(),
      "exclusions": dict(sorted(behavior_excluded.items())),
      "first_retained_mono_ns": behavior_first_mono_ns,
      "last_retained_mono_ns": behavior_last_mono_ns,
      "proof_eligible": source.behavior_eligible and retained_behavior > 0,
      "source_eligible": source.behavior_eligible,
      "source_eligibility_reason": source.behavior_ineligible_reason,
    },
    "car_params_sha256": artifact.manifest["car_params_sha256"],
    "coverage": coverage,
    "encoded_source_plane_sha256": {
      "driving_events": hashlib.sha256(_encode_events(events)).hexdigest(),
      "lateral_maneuver_plans": hashlib.sha256(
        _encode_maneuvers(maneuvers),
      ).hexdigest(),
      "live_delays": hashlib.sha256(_encode_delay(delays)).hexdigest(),
      "live_torque_parameters": hashlib.sha256(
        _encode_torque(torque),
      ).hexdigest(),
      "models": hashlib.sha256(_encode_models(models)).hexdigest(),
      "route_evidence_complete": artifact.sha256,
    },
    "extractor_stream_sha256": prepared.provenance[
      "selected_event_stream_sha256"
    ],
    "physical_plane": {
      "encoded_controls_sha256": physical_controls_digest.hexdigest(),
      "encoded_frames_sha256": physical_digest.hexdigest(),
      "exclusions": dict(sorted(physical_excluded.items())),
      "first_retained_mono_ns": physical_first_mono_ns,
      "frames_retained": retained_physical,
      "last_retained_mono_ns": physical_last_mono_ns,
    },
    "scenario_plane": {
      "active_controls_retained": retained_active_scenario,
      "controls_retained": retained_scenario,
      "encoded_controls_sha256": scenario_controls_digest.hexdigest(),
      "exclusions": dict(sorted(scenario_excluded.items())),
      "first_retained_mono_ns": scenario_first_mono_ns,
      "last_retained_mono_ns": scenario_last_mono_ns,
      "proof_eligible": retained_scenario > 0,
    },
    "preparation_domain": {
      "canonical_join_schema_version": prepared.provenance[
        "canonical_join_schema_version"
      ],
      "extractor_schema_version": prepared.provenance[
        "extractor_schema_version"
      ],
      "log_schema_blob": prepared.provenance["log_schema_blob"],
      "opendbc_commit": prepared.provenance["opendbc_commit"],
      "panda_commit": prepared.provenance["panda_commit"],
      "physical_compatibility_sha256": prepared.provenance[
        "physical_compatibility_sha256"
      ],
      "runtime_vehicle_bundle_sha256": source.runtime_identity,
      "source_superproject_commit": prepared.provenance[
        "superproject_commit"
      ],
    },
    "segment": {
      "index": segment.index,
      "sha256": segment.sha256,
      "size_bytes": segment.size_bytes,
    },
    "segment_init_data_mono_ns": source.route_time_origin_mono_ns,
    "source_boundary_exclusions": dict(sorted(source_excluded.items())),
  }


def _scenario_proof_summary(
  segment_vectors: list[dict[str, object]],
) -> dict[str, object]:
  retained = 0
  active = 0
  identity_segments: list[dict[str, object]] = []
  for result in segment_vectors:
    scenario = result["scenario_plane"]
    if type(scenario) is not dict:
      raise CertificationVectorError("scenario proof plane is malformed")
    retained += int(scenario["controls_retained"])
    active += int(scenario["active_controls_retained"])
    identity_segments.append({
      "car_params_sha256": result["car_params_sha256"],
      "coverage": result["coverage"],
      "encoded_source_plane_sha256": result[
        "encoded_source_plane_sha256"
      ],
      "extractor_stream_sha256": result["extractor_stream_sha256"],
      "physical_plane": result["physical_plane"],
      "preparation_domain": result["preparation_domain"],
      "scenario_plane": scenario,
      "segment": result["segment"],
      "segment_init_data_mono_ns": result["segment_init_data_mono_ns"],
      "source_boundary_exclusions": result["source_boundary_exclusions"],
    })
  identity = hashlib.sha256(
    _SCENARIO_PROOF_DOMAIN + _canonical_json(identity_segments),
  ).hexdigest()
  # The bounded vector proves the decoder/input contract. Full-route replay,
  # not this deterministic sample, owns active-driving metric coverage.
  return {
    "active_controls_retained": active,
    "controls_retained": retained,
    "proof_eligible": retained > 0,
    "selected_inputs_sha256": identity,
  }


def build_certification_vector_from_prepared_route(
  route: RouteCandidate,
  prepared: PreparedRoute,
) -> CertificationVector:
  """Project one bounded vector from an immutable full-route preparation.

  This function is the canonical construction path for PC preparation
  authorities.  Callers should invoke it before releasing the ``PreparedRoute``
  they already produced for the complete route.  It never re-decodes a
  selected segment with reset state.
  """
  if type(route) is not RouteCandidate or type(prepared) is not PreparedRoute:
    raise CertificationVectorError("certification route preparation is invalid")
  selected = certification_vector_selection(route)
  total_compressed = sum(segment.size_bytes for segment in selected)
  if total_compressed > CERTIFICATION_VECTOR_MAX_COMPRESSED_BYTES:
    raise CertificationVectorError("certification segment bytes exceed the bound")
  source_manifest = _source_manifest(route)
  artifact = prepared.route_evidence
  if type(artifact) is not RouteEvidenceArtifact:
    raise CertificationVectorError("route preparation lacks canonical evidence")
  source = artifact.source_identity
  if (
    source.route_id != route.route_name
    or source.route_segment_sha256
    != tuple(segment.sha256 for segment in route.segments)
    or source.route_segment_size_bytes
    != tuple(segment.size_bytes for segment in route.segments)
  ):
    raise CertificationVectorError(
      "route preparation differs from certification source identity",
    )
  if prepared.segment_local_measurement_context_dropped_count:
    raise CertificationVectorError(
      "full-route certification contains segment-reset exclusions",
    )
  if (
    len(prepared.frames) != len(artifact.control_witnesses)
    or prepared.controls_witness_count != source.controls_witness_count
    or prepared.unresolved_witness_count != source.unresolved_witness_count
    or prepared.gap_count != source.gap_count
    or prepared.pre_poll_dropped_count
    != len(source.pre_poll_dropped_timestamps_ns)
    or prepared.behavior_eligible != source.behavior_eligible
    or prepared.behavior_ineligible_reason
    != source.behavior_ineligible_reason
    or prepared.provenance != source.preparation_provenance
  ):
    raise CertificationVectorError(
      "full-route certification populations disagree",
    )
  physical_hash = hashlib.sha256()
  for frame in prepared.frames:
    physical_hash.update(_encode_frame(frame))
  if physical_hash.hexdigest() != artifact.manifest["physical_plane_sha256"]:
    raise CertificationVectorError(
      "full-route certification physical plane differs from its artifact",
    )

  segment_vectors: list[dict[str, object]] = []
  decoded_controls = 0
  first_index = route.segments[0].index
  for segment in selected:
    result = _segment_vector(
      segment=segment,
      prepared=prepared,
      include_pre_poll_exclusions=(segment.index == first_index),
      include_segment_context_exclusions=False,
    )
    coverage = result["coverage"]
    source_exclusions = result["source_boundary_exclusions"]
    if type(coverage) is not dict or type(source_exclusions) is not dict:
      raise CertificationVectorError("certification segment accounting is invalid")
    decoded_controls += int(coverage["controls_total"]) + sum(
      int(value) for value in source_exclusions.values()
    )
    if decoded_controls > CERTIFICATION_VECTOR_MAX_CONTROLS_WITNESSES:
      raise CertificationVectorError(
        "certification controls witness population exceeds its bound",
      )
    segment_vectors.append(result)

  domains = [item["preparation_domain"] for item in segment_vectors]
  if not domains or any(domain != domains[0] for domain in domains[1:]):
    raise CertificationVectorError(
      "selected segments disagree on build or physical provenance",
    )
  manifest: dict[str, object] = {
    "bounds": {
      "maximum_compressed_bytes": CERTIFICATION_VECTOR_MAX_COMPRESSED_BYTES,
      "maximum_controls_witnesses": CERTIFICATION_VECTOR_MAX_CONTROLS_WITNESSES,
      "maximum_segments": CERTIFICATION_VECTOR_MAX_SEGMENTS,
      "selected_compressed_bytes": total_compressed,
      "selected_controls_witnesses": decoded_controls,
    },
    "domain": "blatv2-cross-architecture-vector-v5",
    "route_name": route.route_name,
    "route_provenance_seed": {
      "car_params_sha256": hashlib.sha256(
        artifact.car_params_bytes,
      ).hexdigest(),
      "source_segment_index": first_index,
      "source_segment_sha256": route.segments[0].sha256,
    },
    "scenario_proof": _scenario_proof_summary(segment_vectors),
    "schema_version": CERTIFICATION_VECTOR_SCHEMA_VERSION,
    "segment_results": segment_vectors,
    "selection_identity_sha256": certification_vector_selection_identity(
      route,
      selected,
    ),
    "source_manifest": source_manifest,
  }
  vector = CertificationVector.from_manifest(manifest)
  # Domain-separate the externally exposed identity even if another artifact
  # format were ever to produce the same bytes.
  identity = hashlib.sha256(_VECTOR_DOMAIN + vector.canonical_bytes).hexdigest()
  manifest["vector_identity_sha256"] = identity
  return CertificationVector.from_manifest(manifest)


def build_certification_vector(
  route: RouteCandidate,
  *,
  extractor_path: str | Path,
  event_reader: Callable[[bytes], AbstractContextManager[Any]],
  car_params_decoder: Callable[[bytes], Any],
  descriptor_registry: BuildDescriptorRegistry,
  route_bundle_factory: Callable[[Any, BuildDescriptor], RuntimeVehicleBundle],
  current_car_params: Any,
  current_bundle: RuntimeVehicleBundle,
  expected_dongle_id: str,
  expected_extractor_sha256: str | None = None,
  abort_requested: Callable[[], bool] = lambda: False,
  segment_started: Callable[[RouteSegment, int, int], None] | None = None,
  segment_completed: Callable[[RouteSegment, int, int], None] | None = None,
) -> CertificationVector:
  """Prepare one complete route, then project its bounded proof vector.

  This compatibility entry point is PC-only.  The decoded route population is
  bounded by the normal production preparation contract; only the selected
  proof population is admitted to the 64 KiB vector.  Production workers that
  already hold a complete ``PreparedRoute`` should call
  ``build_certification_vector_from_prepared_route`` instead.
  """
  try:
    prepared = prepare_route(
      route,
      extractor_path=extractor_path,
      event_reader=event_reader,
      car_params_decoder=car_params_decoder,
      descriptor_registry=descriptor_registry,
      route_bundle_factory=route_bundle_factory,
      current_car_params=current_car_params,
      current_bundle=current_bundle,
      expected_dongle_id=expected_dongle_id,
      expected_extractor_sha256=expected_extractor_sha256,
      abort_requested=abort_requested,
      segment_started=segment_started,
      segment_completed=segment_completed,
    )
  except ValueError as exc:
    raise CertificationVectorError(
      "certification route preparation contract is invalid",
    ) from exc
  return build_certification_vector_from_prepared_route(route, prepared)


def _validate_manifest(manifest: dict[str, object]) -> None:
  _validate_json_depth(manifest)
  expected = {
    "bounds",
    "domain",
    "route_name",
    "route_provenance_seed",
    "scenario_proof",
    "schema_version",
    "segment_results",
    "selection_identity_sha256",
    "source_manifest",
  }
  if "vector_identity_sha256" in manifest:
    expected.add("vector_identity_sha256")
  if set(manifest) != expected:
    raise CertificationVectorError("certification vector manifest keys differ")
  if (
    manifest["schema_version"] != CERTIFICATION_VECTOR_SCHEMA_VERSION
    or manifest["domain"] != "blatv2-cross-architecture-vector-v5"
    or type(manifest["route_name"]) is not str
    or not manifest["route_name"]
  ):
    raise CertificationVectorError("certification vector identity is invalid")
  if (
    type(manifest["selection_identity_sha256"]) is not str
    or _SHA256_RE.fullmatch(manifest["selection_identity_sha256"]) is None
  ):
    raise CertificationVectorError("certification selection identity is invalid")
  if "vector_identity_sha256" in manifest and (
    type(manifest["vector_identity_sha256"]) is not str
    or _SHA256_RE.fullmatch(manifest["vector_identity_sha256"]) is None
  ):
    raise CertificationVectorError("certification vector identity is invalid")
  if "vector_identity_sha256" in manifest:
    unsigned = dict(manifest)
    claimed = str(unsigned.pop("vector_identity_sha256"))
    unsigned_vector = CertificationVector.from_manifest(unsigned)
    observed = hashlib.sha256(
      _VECTOR_DOMAIN + unsigned_vector.canonical_bytes,
    ).hexdigest()
    if claimed != observed:
      raise CertificationVectorError("certification vector identity mismatch")
  bounds = manifest["bounds"]
  if type(bounds) is not dict or set(bounds) != {
    "maximum_compressed_bytes",
    "maximum_controls_witnesses",
    "maximum_segments",
    "selected_compressed_bytes",
    "selected_controls_witnesses",
  }:
    raise CertificationVectorError("certification vector bounds are invalid")
  expected_bounds = {
    "maximum_compressed_bytes": CERTIFICATION_VECTOR_MAX_COMPRESSED_BYTES,
    "maximum_controls_witnesses": CERTIFICATION_VECTOR_MAX_CONTROLS_WITNESSES,
    "maximum_segments": CERTIFICATION_VECTOR_MAX_SEGMENTS,
  }
  if any(bounds[key] != value for key, value in expected_bounds.items()):
    raise CertificationVectorError("certification vector bound version changed")
  for key in ("selected_compressed_bytes", "selected_controls_witnesses"):
    if type(bounds[key]) is not int or bounds[key] < 0:
      raise CertificationVectorError("certification vector observed bound is invalid")
  if (
    bounds["selected_compressed_bytes"] > CERTIFICATION_VECTOR_MAX_COMPRESSED_BYTES
    or bounds["selected_controls_witnesses"] > CERTIFICATION_VECTOR_MAX_CONTROLS_WITNESSES
  ):
    raise CertificationVectorError("certification vector exceeds its bounds")
  source = manifest["source_manifest"]
  results = manifest["segment_results"]
  if (
    type(source) is not list
    or not source
    or type(results) is not list
    or not (1 <= len(results) <= CERTIFICATION_VECTOR_MAX_SEGMENTS)
  ):
    raise CertificationVectorError("certification vector populations are invalid")
  _validate_vector_semantics(manifest)
  encoded = _canonical_json(manifest)
  if len(encoded) + _HEADER.size > CERTIFICATION_VECTOR_MAX_BYTES:
    raise CertificationVectorError("certification vector exceeds its bound")


def _validate_json_depth(value: object) -> None:
  pending = [(value, 0)]
  nodes = 0
  while pending:
    current, depth = pending.pop()
    nodes += 1
    if nodes > 20_000 or depth > 16:
      raise CertificationVectorError("certification vector nesting exceeds its bound")
    if type(current) is dict:
      for key, child in current.items():
        if type(key) is not str:
          raise CertificationVectorError("certification vector key is invalid")
        pending.append((child, depth + 1))
    elif type(current) is list:
      pending.extend((child, depth + 1) for child in current)
    elif type(current) is float:
      if not math.isfinite(current):
        raise CertificationVectorError("certification vector float is invalid")
    elif type(current) not in (str, int, bool, type(None)):
      raise CertificationVectorError("certification vector value is invalid")


def _bounded_nonnegative(value: object, name: str, maximum: int = 1 << 60) -> int:
  if type(value) is not int or not 0 <= value <= maximum:
    raise CertificationVectorError(f"{name} is invalid")
  return value


def _hash_value(value: object, name: str) -> str:
  if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
    raise CertificationVectorError(f"{name} is invalid")
  return value


def _exclusion_total(
  value: object,
  name: str,
  allowed: frozenset[str],
) -> int:
  if type(value) is not dict or not set(value).issubset(allowed):
    raise CertificationVectorError(f"{name} exclusions are malformed")
  total = 0
  for reason, count in value.items():
    if type(reason) is not str or type(count) is not int or count <= 0:
      raise CertificationVectorError(f"{name} exclusions are malformed")
    total += count
  return total


def _validate_retained_timestamps(
  first: object,
  last: object,
  count: int,
  name: str,
) -> None:
  if count == 0:
    if first is not None or last is not None:
      raise CertificationVectorError(f"empty {name} timestamps are not canonical")
    return
  if (
    type(first) is not int
    or type(last) is not int
    or first < 0
    or last < first
  ):
    raise CertificationVectorError(f"{name} timestamp coverage is invalid")


def _validate_vector_semantics(manifest: dict[str, object]) -> None:
  source = manifest["source_manifest"]
  results = manifest["segment_results"]
  bounds = manifest["bounds"]
  assert type(source) is list and type(results) is list and type(bounds) is dict
  source_by_index: dict[int, dict[str, object]] = {}
  previous = -1
  for item in source:
    if type(item) is not dict or set(item) != {"index", "sha256", "size_bytes"}:
      raise CertificationVectorError("source segment identity is malformed")
    index = _bounded_nonnegative(item["index"], "source segment index", (1 << 32) - 1)
    size = _bounded_nonnegative(item["size_bytes"], "source segment size", 1 << 40)
    if index <= previous or size == 0:
      raise CertificationVectorError("source segments are not ordered")
    _hash_value(item["sha256"], "source segment hash")
    source_by_index[index] = item
    previous = index
  seed = manifest.get("route_provenance_seed")
  if type(seed) is not dict or set(seed) != {
    "car_params_sha256", "source_segment_index", "source_segment_sha256",
  }:
    raise CertificationVectorError("route provenance seed is malformed")
  seed_index = _bounded_nonnegative(seed["source_segment_index"], "seed segment index", (1 << 32) - 1)
  if seed_index != int(source[0]["index"]):
    raise CertificationVectorError("route provenance seed is not segment zero")
  _hash_value(seed["car_params_sha256"], "seed CarParams hash")
  if seed["source_segment_sha256"] != source[0]["sha256"]:
    raise CertificationVectorError("route provenance seed source changed")

  selected_manifest: list[dict[str, object]] = []
  controls_total = 0
  compressed_total = 0
  previous = -1
  route_context: dict[str, object] | None = None
  expected_result_keys = {
    "behavior_plane", "car_params_sha256", "coverage",
    "encoded_source_plane_sha256", "extractor_stream_sha256",
    "physical_plane", "preparation_domain", "scenario_plane", "segment",
    "segment_init_data_mono_ns", "source_boundary_exclusions",
  }
  for result in results:
    if type(result) is not dict or set(result) != expected_result_keys:
      raise CertificationVectorError("segment result shape is invalid")
    segment = result["segment"]
    if type(segment) is not dict or set(segment) != {"index", "sha256", "size_bytes"}:
      raise CertificationVectorError("selected segment identity is malformed")
    index = _bounded_nonnegative(segment["index"], "selected segment index", (1 << 32) - 1)
    if index <= previous or source_by_index.get(index) != segment:
      raise CertificationVectorError("selected segment is not in the source manifest")
    previous = index
    compressed_total += int(segment["size_bytes"])
    selected_manifest.append(dict(segment))
    coverage = result["coverage"]
    coverage_keys = {
      "controls_total", "driving_events",
      "lateral_maneuver_plans", "live_delays", "live_torque_parameters",
      "models", "physical_frames_total",
    }
    if type(coverage) is not dict or set(coverage) != coverage_keys:
      raise CertificationVectorError("segment coverage is malformed")
    for key in coverage_keys:
      _bounded_nonnegative(coverage[key], f"coverage {key}", 1_000_000)
    if (
      coverage["controls_total"] == 0
      or coverage["physical_frames_total"] != coverage["controls_total"]
    ):
      raise CertificationVectorError("segment coverage is not meaningful")

    source_exclusion_total = _exclusion_total(
      result["source_boundary_exclusions"],
      "source boundary",
      frozenset({"pre_poll_controls"}),
    )
    if index != int(source[0]["index"]) and source_exclusion_total:
      raise CertificationVectorError(
        "route-boundary exclusions escaped the provenance segment",
      )
    controls_total += int(coverage["controls_total"]) + int(
      source_exclusion_total,
    )

    physical = result["physical_plane"]
    expected_physical_keys = {
      "encoded_controls_sha256", "encoded_frames_sha256", "exclusions",
      "first_retained_mono_ns", "frames_retained",
      "last_retained_mono_ns",
    }
    if type(physical) is not dict or set(physical) != expected_physical_keys:
      raise CertificationVectorError("physical proof plane is malformed")
    physical_count = _bounded_nonnegative(
      physical["frames_retained"],
      "physical retained frames",
      1_000_000,
    )
    physical_exclusion_total = _exclusion_total(
      physical["exclusions"],
      "physical plane",
      frozenset({
        "car_control_context_missing",
        "physical_inputs_invalid",
        "race_unresolved",
        "segment_local_gap",
      }),
    )
    if (
      physical_count == 0
      or physical_count + physical_exclusion_total
      != coverage["controls_total"]
    ):
      raise CertificationVectorError("physical proof coverage is not meaningful")
    _validate_retained_timestamps(
      physical["first_retained_mono_ns"],
      physical["last_retained_mono_ns"],
      physical_count,
      "physical plane",
    )
    physical_controls_hash = _hash_value(
      physical["encoded_controls_sha256"],
      "physical controls plane",
    )
    physical_frames_hash = _hash_value(
      physical["encoded_frames_sha256"],
      "physical frames plane",
    )
    empty_hash = hashlib.sha256().hexdigest()
    if physical_controls_hash == empty_hash or physical_frames_hash == empty_hash:
      raise CertificationVectorError("nonempty physical plane has an empty digest")

    behavior = result["behavior_plane"]
    expected_behavior_keys = {
      "controls_retained", "encoded_controls_sha256", "exclusions",
      "first_retained_mono_ns", "last_retained_mono_ns", "proof_eligible",
      "source_eligible", "source_eligibility_reason",
    }
    if type(behavior) is not dict or set(behavior) != expected_behavior_keys:
      raise CertificationVectorError("behavior proof plane is malformed")
    behavior_count = _bounded_nonnegative(
      behavior["controls_retained"],
      "behavior retained controls",
      1_000_000,
    )
    behavior_exclusion_total = _exclusion_total(
      behavior["exclusions"],
      "behavior plane",
      frozenset({
        "live_delay_context_missing",
        "live_torque_context_missing",
        "maneuver_context_missing",
        "model_context_missing",
        "physical_plane_car_control_context_missing",
        "physical_plane_inputs_invalid",
        "physical_plane_race_unresolved",
        "physical_plane_segment_local_gap",
      }),
    )
    if (
      behavior_count + behavior_exclusion_total != coverage["controls_total"]
      or behavior_count > physical_count
    ):
      raise CertificationVectorError("behavior proof coverage is not meaningful")
    _validate_retained_timestamps(
      behavior["first_retained_mono_ns"],
      behavior["last_retained_mono_ns"],
      behavior_count,
      "behavior plane",
    )
    behavior_hash = _hash_value(
      behavior["encoded_controls_sha256"],
      "behavior controls plane",
    )
    if behavior_count == 0:
      if behavior_hash != empty_hash:
        raise CertificationVectorError("empty behavior plane digest is not canonical")
    elif behavior_hash == empty_hash:
      raise CertificationVectorError("nonempty behavior plane has an empty digest")
    source_eligible = behavior["source_eligible"]
    proof_eligible = behavior["proof_eligible"]
    source_reason = behavior["source_eligibility_reason"]
    if (
      type(source_eligible) is not bool
      or type(proof_eligible) is not bool
      or type(source_reason) is not str
      or not source_reason
      or source_eligible != (source_reason == "eligible")
      or proof_eligible != (source_eligible and behavior_count > 0)
      or (source_eligible and behavior_count == 0)
    ):
      raise CertificationVectorError("behavior source eligibility is invalid")
    if behavior_count:
      physical_first = physical["first_retained_mono_ns"]
      physical_last = physical["last_retained_mono_ns"]
      behavior_first = behavior["first_retained_mono_ns"]
      behavior_last = behavior["last_retained_mono_ns"]
      if not all(
        type(value) is int
        for value in (physical_first, physical_last, behavior_first, behavior_last)
      ):
        raise CertificationVectorError("retained timestamp coverage is invalid")
      if (
        behavior_first < physical_first
        or behavior_last > physical_last
      ):
        raise CertificationVectorError("behavior timestamps escape physical coverage")

    scenario = result["scenario_plane"]
    expected_scenario_keys = {
      "active_controls_retained", "controls_retained",
      "encoded_controls_sha256", "exclusions",
      "first_retained_mono_ns", "last_retained_mono_ns", "proof_eligible",
    }
    if type(scenario) is not dict or set(scenario) != expected_scenario_keys:
      raise CertificationVectorError("scenario proof plane is malformed")
    scenario_count = _bounded_nonnegative(
      scenario["controls_retained"],
      "scenario retained controls",
      1_000_000,
    )
    scenario_active_count = _bounded_nonnegative(
      scenario["active_controls_retained"],
      "scenario retained active controls",
      1_000_000,
    )
    scenario_exclusion_total = _exclusion_total(
      scenario["exclusions"],
      "scenario plane",
      frozenset({
        "active_maneuver_override",
        "applied_torque_context_missing",
        "live_delay_context_missing",
        "live_delay_payload_invalid",
        "live_torque_context_missing",
        "live_torque_health_inexact",
        "live_torque_payload_invalid",
        "maneuver_context_invalid",
        "model_context_missing",
        "model_message_invalid",
        "model_native_grid_invalid",
        "model_not_alive",
        "physical_plane_car_control_context_missing",
        "physical_plane_inputs_invalid",
        "physical_plane_race_unresolved",
        "physical_plane_segment_local_gap",
        "scenario_message_invalid",
      }),
    )
    if (
      scenario_count + scenario_exclusion_total != coverage["controls_total"]
      or scenario_count > physical_count
      or scenario_active_count > scenario_count
      or type(scenario["proof_eligible"]) is not bool
      or scenario["proof_eligible"] != (scenario_count > 0)
    ):
      raise CertificationVectorError("scenario proof coverage is not meaningful")
    _validate_retained_timestamps(
      scenario["first_retained_mono_ns"],
      scenario["last_retained_mono_ns"],
      scenario_count,
      "scenario plane",
    )
    scenario_hash = _hash_value(
      scenario["encoded_controls_sha256"],
      "scenario controls plane",
    )
    if scenario_count == 0:
      if scenario_hash != empty_hash:
        raise CertificationVectorError("empty scenario plane digest is not canonical")
    elif scenario_hash == empty_hash:
      raise CertificationVectorError("nonempty scenario plane has an empty digest")
    if scenario_count:
      physical_first = physical["first_retained_mono_ns"]
      physical_last = physical["last_retained_mono_ns"]
      scenario_first = scenario["first_retained_mono_ns"]
      scenario_last = scenario["last_retained_mono_ns"]
      if not all(
        type(value) is int
        for value in (physical_first, physical_last, scenario_first, scenario_last)
      ):
        raise CertificationVectorError("retained timestamp coverage is invalid")
      if scenario_first < physical_first or scenario_last > physical_last:
        raise CertificationVectorError("scenario timestamps escape physical coverage")

    hashes = result["encoded_source_plane_sha256"]
    expected_hashes = {
      "driving_events", "lateral_maneuver_plans", "live_delays",
      "live_torque_parameters", "models", "route_evidence_complete",
    }
    if type(hashes) is not dict or set(hashes) != expected_hashes:
      raise CertificationVectorError("encoded plane identities are malformed")
    for key in expected_hashes:
      _hash_value(hashes[key], f"encoded plane {key}")
    _hash_value(result["car_params_sha256"], "segment CarParams hash")
    if result["car_params_sha256"] != seed["car_params_sha256"]:
      raise CertificationVectorError("segment CarParams identity changed")
    _hash_value(result["extractor_stream_sha256"], "extractor stream hash")
    _bounded_nonnegative(result["segment_init_data_mono_ns"], "InitData time")
    domain = result["preparation_domain"]
    expected_domain = {
      "canonical_join_schema_version", "extractor_schema_version",
      "log_schema_blob", "opendbc_commit", "panda_commit",
      "physical_compatibility_sha256", "runtime_vehicle_bundle_sha256",
      "source_superproject_commit",
    }
    if type(domain) is not dict or set(domain) != expected_domain:
      raise CertificationVectorError("preparation domain is malformed")
    for key in ("canonical_join_schema_version", "extractor_schema_version"):
      if type(domain[key]) is not int or domain[key] <= 0:
        raise CertificationVectorError("preparation schema is invalid")
    for key in ("physical_compatibility_sha256", "runtime_vehicle_bundle_sha256"):
      _hash_value(domain[key], f"preparation domain {key}")
    for key in ("log_schema_blob", "opendbc_commit", "panda_commit", "source_superproject_commit"):
      if type(domain[key]) is not str or re.fullmatch(r"[0-9a-f]{40}", domain[key]) is None:
        raise CertificationVectorError(f"preparation domain {key} is invalid")
    candidate_context = {
      "behavior_source_eligible": behavior["source_eligible"],
      "behavior_source_reason": behavior["source_eligibility_reason"],
      "extractor_stream_sha256": result["extractor_stream_sha256"],
      "preparation_domain": domain,
      "route_evidence_complete": hashes["route_evidence_complete"],
      "segment_init_data_mono_ns": result["segment_init_data_mono_ns"],
    }
    if route_context is None:
      route_context = candidate_context
    elif candidate_context != route_context:
      raise CertificationVectorError(
        "selected proofs do not share one full-route preparation context",
      )
  scenario_proof = manifest["scenario_proof"]
  expected_scenario_proof_keys = {
    "active_controls_retained", "controls_retained", "proof_eligible",
    "selected_inputs_sha256",
  }
  if (
    type(scenario_proof) is not dict
    or set(scenario_proof) != expected_scenario_proof_keys
  ):
    raise CertificationVectorError("route scenario proof is malformed")
  _bounded_nonnegative(
    scenario_proof["active_controls_retained"],
    "route scenario retained active controls",
    CERTIFICATION_VECTOR_MAX_CONTROLS_WITNESSES,
  )
  _bounded_nonnegative(
    scenario_proof["controls_retained"],
    "route scenario retained controls",
    CERTIFICATION_VECTOR_MAX_CONTROLS_WITNESSES,
  )
  _hash_value(
    scenario_proof["selected_inputs_sha256"],
    "route scenario selected inputs",
  )
  if type(scenario_proof["proof_eligible"]) is not bool:
    raise CertificationVectorError("route scenario proof eligibility is invalid")
  rebuilt_scenario_proof = _scenario_proof_summary(results)
  if scenario_proof != rebuilt_scenario_proof:
    raise CertificationVectorError("route scenario proof identity mismatch")
  if selected_manifest[0]["index"] != source[0]["index"]:
    raise CertificationVectorError("certification vector omits provenance segment")
  if compressed_total != bounds["selected_compressed_bytes"]:
    raise CertificationVectorError("selected byte accounting changed")
  if controls_total != bounds["selected_controls_witnesses"] or controls_total == 0:
    raise CertificationVectorError("selected controls accounting changed")
  expected_selection = hashlib.sha256(_canonical_json({
    "domain": "blatv2-certification-vector-selection-v5",
    "route_name": manifest["route_name"],
    "source_segments": source,
    "selected_segments": selected_manifest,
  })).hexdigest()
  if expected_selection != manifest["selection_identity_sha256"]:
    raise CertificationVectorError("certification selection identity mismatch")

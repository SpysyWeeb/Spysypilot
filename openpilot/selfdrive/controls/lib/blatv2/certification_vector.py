"""Bounded cross-architecture proof for BLaTv2 route preparation.

The workstation still prepares and byte-compares the *complete* route twice.
This module proves only the architecture-sensitive part on comma hardware: a
source-selected set of whole rlog segments is decoded with the production
extractor, canonical race reconstruction, measured-frame/vehicle-model path,
route-evidence encoders, and physical spool encoder.  It deliberately does
not run a learner, fit a profile, or materialize a full ``PreparedRoute``.

Selection is a pure function of the immutable route manifest.  Every selected
segment is prepared independently, so state can never leak across a segment
boundary.  Witnesses lacking segment-local context are removed from the
comparison plane and counted in the canonical result.  The complete encoded
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
  MAXIMUM_ROUTE_FRAMES,
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
from openpilot.selfdrive.controls.lib.blatv2.route_evidence import (
  ControlsWitness,
  RouteEvidenceArtifact,
  _encode_controls,
)
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  RuntimeVehicleBundle,
)


CERTIFICATION_VECTOR_SCHEMA_VERSION: Final = 1
CERTIFICATION_VECTOR_MAGIC: Final = b"BLATCV01"
CERTIFICATION_VECTOR_MAX_SEGMENTS: Final = 3
CERTIFICATION_VECTOR_MAX_COMPRESSED_BYTES: Final = 96 * 1024 * 1024
CERTIFICATION_VECTOR_MAX_CONTROLS_WITNESSES: Final = 30_000
CERTIFICATION_VECTOR_MAX_BYTES: Final = 64 * 1024
_VECTOR_DOMAIN: Final = b"blatv2-cross-architecture-vector-v1\0"
_SELECTION_DOMAIN: Final = b"blatv2-certification-segment-selection-v1\0"
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
    "domain": "blatv2-certification-vector-selection-v1",
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


def _boundary_exclusion_reason(witness: ControlsWitness) -> str | None:
  if witness.race_unresolved:
    return "race_unresolved"
  if witness.gap_from_previous:
    return "segment_local_gap"
  if not witness.car_control_paired:
    return "car_control_context_missing"
  if not witness.model_link_valid:
    return "model_context_missing"
  if not witness.live_torque_parameters_available:
    return "live_torque_context_missing"
  if not witness.live_delay_available:
    return "live_delay_context_missing"
  if not witness.maneuver_plan_available:
    return "maneuver_context_missing"
  return None


def _segment_vector(
  *,
  segment: RouteSegment,
  prepared: PreparedRoute,
) -> dict[str, object]:
  artifact = prepared.route_evidence
  if type(artifact) is not RouteEvidenceArtifact:
    raise CertificationVectorError("segment preparation lacks route evidence")
  controls = artifact.control_witnesses
  frames = prepared.frames
  if len(controls) != len(frames):
    raise CertificationVectorError("segment control/frame populations disagree")

  excluded = Counter[str]()
  retained_controls: list[ControlsWitness] = []
  physical_digest = hashlib.sha256()
  controls_digest = hashlib.sha256()
  first_mono_ns: int | None = None
  last_mono_ns: int | None = None
  for witness, frame in zip(controls, frames, strict=True):
    reason = _boundary_exclusion_reason(witness)
    if reason is not None:
      excluded[reason] += 1
      continue
    retained_controls.append(witness)
    physical_digest.update(_encode_frame(frame))
    controls_digest.update(_encode_controls((witness,)))
    first_mono_ns = (
      witness.mono_time_ns if first_mono_ns is None else first_mono_ns
    )
    last_mono_ns = witness.mono_time_ns
  pre_poll_count = prepared.pre_poll_dropped_count
  if pre_poll_count:
    excluded["pre_poll_controls"] += pre_poll_count

  source = artifact.source_identity
  section_hashes = artifact.manifest.get("section_sha256")
  if type(section_hashes) is not dict:
    raise CertificationVectorError("segment evidence section hashes are absent")
  coverage = {
    "controls_total": len(controls),
    "controls_retained": len(retained_controls),
    "driving_events": len(artifact.event_locators),
    "lateral_maneuver_plans": len(artifact.lateral_maneuver_plans),
    "live_delays": len(artifact.live_delays),
    "live_torque_parameters": len(artifact.live_torque_parameters),
    "models": len(artifact.model_publications),
    "physical_frames_total": len(frames),
    "physical_frames_retained": len(retained_controls),
  }
  return {
    "boundary_exclusions": dict(sorted(excluded.items())),
    "car_params_sha256": artifact.manifest["car_params_sha256"],
    "coverage": coverage,
    "encoded_plane_sha256": {
      "controls_retained": controls_digest.hexdigest(),
      "driving_events": section_hashes["events"],
      "lateral_maneuver_plans": section_hashes["maneuvers"],
      "live_delays": section_hashes["live_delay"],
      "live_torque_parameters": section_hashes["live_torque"],
      "models": section_hashes["models"],
      "physical_frames_retained": physical_digest.hexdigest(),
      "route_evidence_complete": artifact.sha256,
    },
    "extractor_stream_sha256": prepared.provenance[
      "selected_event_stream_sha256"
    ],
    "first_retained_mono_ns": first_mono_ns,
    "last_retained_mono_ns": last_mono_ns,
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
  }


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
  """Build one bounded vector using the production route-preparation core."""
  selected = certification_vector_selection(route)
  total_compressed = sum(segment.size_bytes for segment in selected)
  if total_compressed > CERTIFICATION_VECTOR_MAX_COMPRESSED_BYTES:
    raise CertificationVectorError("certification segment bytes exceed the bound")
  source_manifest = _source_manifest(route)
  segment_vectors: list[dict[str, object]] = []
  decoded_controls = 0
  route_car_params_seed: bytes | None = None
  first_index = route.segments[0].index
  last_index = route.segments[-1].index
  for position, segment in enumerate(selected, start=1):
    remaining = CERTIFICATION_VECTOR_MAX_CONTROLS_WITNESSES - decoded_controls
    if remaining <= 0:
      raise CertificationVectorError(
        "certification controls witness population exceeds its bound",
      )
    if segment_started is not None:
      segment_started(segment, position, len(selected))
    try:
      prepared = prepare_route(
        RouteCandidate(
          route_name=route.route_name,
          route_counter=route.route_counter,
          segments=(segment,),
        ),
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
        structural_first_segment_index=first_index,
        structural_last_segment_index=last_index,
        maximum_controls_witnesses=remaining,
        route_car_params_seed=(
          None if segment.index == first_index else route_car_params_seed
        ),
      )
    except ValueError as exc:
      raise CertificationVectorError(
        "certification segment preparation contract is invalid",
      ) from exc
    decoded_controls += prepared.controls_witness_count
    if decoded_controls > CERTIFICATION_VECTOR_MAX_CONTROLS_WITNESSES:
      raise CertificationVectorError(
        "certification controls witness population exceeds its bound",
      )
    if segment.index == first_index:
      route_car_params_seed = bytes(prepared.route_evidence.car_params_bytes)
      if not route_car_params_seed:
        raise CertificationVectorError(
          "certification provenance segment lacks CarParams",
        )
    elif route_car_params_seed is None:
      raise CertificationVectorError(
        "certification segment precedes its authenticated CarParams seed",
      )
    segment_vectors.append(_segment_vector(segment=segment, prepared=prepared))
    del prepared
    if segment_completed is not None:
      segment_completed(segment, position, len(selected))

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
    "domain": "blatv2-cross-architecture-vector-v1",
    "route_name": route.route_name,
    "route_provenance_seed": {
      "car_params_sha256": hashlib.sha256(
        b"" if route_car_params_seed is None else route_car_params_seed,
      ).hexdigest(),
      "source_segment_index": first_index,
      "source_segment_sha256": route.segments[0].sha256,
    },
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


def _validate_manifest(manifest: dict[str, object]) -> None:
  _validate_json_depth(manifest)
  expected = {
    "bounds",
    "domain",
    "route_name",
    "route_provenance_seed",
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
    or manifest["domain"] != "blatv2-cross-architecture-vector-v1"
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
  expected_result_keys = {
    "boundary_exclusions", "car_params_sha256", "coverage",
    "encoded_plane_sha256", "extractor_stream_sha256",
    "first_retained_mono_ns", "last_retained_mono_ns",
    "preparation_domain", "segment", "segment_init_data_mono_ns",
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
      "controls_total", "controls_retained", "driving_events",
      "lateral_maneuver_plans", "live_delays", "live_torque_parameters",
      "models", "physical_frames_total", "physical_frames_retained",
    }
    if type(coverage) is not dict or set(coverage) != coverage_keys:
      raise CertificationVectorError("segment coverage is malformed")
    for key in coverage_keys:
      _bounded_nonnegative(coverage[key], f"coverage {key}", 1_000_000)
    if (
      coverage["controls_total"] == 0
      or coverage["controls_retained"] == 0
      or coverage["controls_retained"] > coverage["controls_total"]
      or coverage["physical_frames_total"] != coverage["controls_total"]
      or coverage["physical_frames_retained"] != coverage["controls_retained"]
    ):
      raise CertificationVectorError("segment coverage is not meaningful")
    exclusions = result["boundary_exclusions"]
    if type(exclusions) is not dict or any(
      type(key) is not str or not key or type(value) is not int or value < 0
      for key, value in exclusions.items()
    ):
      raise CertificationVectorError("boundary exclusions are malformed")
    controls_total += int(coverage["controls_total"]) + int(
      exclusions.get("pre_poll_controls", 0),
    )
    hashes = result["encoded_plane_sha256"]
    expected_hashes = {
      "controls_retained", "driving_events", "lateral_maneuver_plans",
      "live_delays", "live_torque_parameters", "models",
      "physical_frames_retained", "route_evidence_complete",
    }
    if type(hashes) is not dict or set(hashes) != expected_hashes:
      raise CertificationVectorError("encoded plane identities are malformed")
    for key in expected_hashes:
      _hash_value(hashes[key], f"encoded plane {key}")
    _hash_value(result["car_params_sha256"], "segment CarParams hash")
    if result["car_params_sha256"] != seed["car_params_sha256"]:
      raise CertificationVectorError("segment CarParams identity changed")
    _hash_value(result["extractor_stream_sha256"], "extractor stream hash")
    first = result["first_retained_mono_ns"]
    last = result["last_retained_mono_ns"]
    if type(first) is not int or type(last) is not int or first < 0 or last < first:
      raise CertificationVectorError("retained timestamp coverage is invalid")
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
  if selected_manifest[0]["index"] != source[0]["index"]:
    raise CertificationVectorError("certification vector omits provenance segment")
  if compressed_total != bounds["selected_compressed_bytes"]:
    raise CertificationVectorError("selected byte accounting changed")
  if controls_total != bounds["selected_controls_witnesses"] or controls_total == 0:
    raise CertificationVectorError("selected controls accounting changed")
  expected_selection = hashlib.sha256(_canonical_json({
    "domain": "blatv2-certification-vector-selection-v1",
    "route_name": manifest["route_name"],
    "source_segments": source,
    "selected_segments": selected_manifest,
  })).hexdigest()
  if expected_selection != manifest["selection_identity_sha256"]:
    raise CertificationVectorError("certification selection identity mismatch")

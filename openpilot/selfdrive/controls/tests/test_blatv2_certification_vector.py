from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct

import pytest  # noqa: TID251

from openpilot.selfdrive.controls.lib.blatv2.certification_vector import (
  CERTIFICATION_VECTOR_MAX_COMPRESSED_BYTES,
  CERTIFICATION_VECTOR_MAX_CONTROLS_WITNESSES,
  CERTIFICATION_VECTOR_MAX_SEGMENTS,
  CERTIFICATION_VECTOR_SCHEMA_VERSION,
  CertificationVector,
  CertificationVectorError,
  _VECTOR_DOMAIN,
  _segment_vector,
  certification_vector_selection_identity,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill import (
  PreparedRoute,
  RouteCandidate,
  RouteSegment,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill_spool import (
  _encode_frame,
)
from openpilot.selfdrive.controls.lib.blatv2.route_evidence import (
  RouteEvidenceArtifact,
  _encode_controls,
)
from openpilot.selfdrive.controls.tests.test_blatv2_route_evidence import (
  CP,
  DELAY,
  EVENTS,
  FRAMES,
  MANEUVER,
  PHYSICAL,
  TORQUE,
  artifact,
  model,
  source,
  witness,
)


ROUTE = "000000b7--a6b3b1f175"
SEGMENT = RouteSegment(0, Path("rlog.zst"), "a" * 64, 1234)
PROVENANCE = {
  "canonical_join_schema_version": 3,
  "extractor_schema_version": 4,
  "log_schema_blob": "1" * 40,
  "opendbc_commit": "2" * 40,
  "panda_commit": "3" * 40,
  "physical_compatibility_sha256": "4" * 64,
  "selected_event_stream_sha256": "5" * 64,
  "superproject_commit": "6" * 40,
}


def _prepared(
  evidence: RouteEvidenceArtifact,
  frames=FRAMES,
  *,
  pre_poll: int = 0,
  segment_context: int = 0,
) -> PreparedRoute:
  return PreparedRoute(
    frames=frames,
    controls_witness_count=len(frames) + pre_poll + segment_context,
    unresolved_witness_count=pre_poll + segment_context,
    gap_count=0,
    provenance=dict(PROVENANCE),
    route_evidence=evidence,
    pre_poll_dropped_count=pre_poll,
    behavior_eligible=evidence.source_identity.behavior_eligible,
    behavior_ineligible_reason=(
      evidence.source_identity.behavior_ineligible_reason
    ),
    segment_local_measurement_context_dropped_count=segment_context,
  )


def _without_maneuver_plan(
  *,
  source_eligible: bool = False,
) -> RouteEvidenceArtifact:
  controls = tuple(
    replace(
      witness(index),
      lateral_maneuver_plan_index=-1,
      maneuver_plan_available=False,
    )
    for index in range(len(FRAMES))
  )
  identity = replace(
    source(),
    controller_source_kind=("stock_canonical" if source_eligible else "ineligible"),
    behavior_eligible=source_eligible,
    behavior_ineligible_reason=("eligible" if source_eligible else "lateral_maneuver_plan_missing"),
  )
  return RouteEvidenceArtifact(
    identity,
    CP,
    PHYSICAL,
    (model(0), model(1)),
    controls,
    TORQUE,
    DELAY,
    (),
    EVENTS,
  )


def _signed_vector(result: dict[str, object]) -> CertificationVector:
  candidate = RouteCandidate(ROUTE, 1, (SEGMENT,))
  source_manifest = [SEGMENT.to_ledger_dict()]
  source_exclusions = result["source_boundary_exclusions"]
  coverage = result["coverage"]
  assert isinstance(source_exclusions, dict)
  assert isinstance(coverage, dict)
  unsigned = {
    "bounds": {
      "maximum_compressed_bytes": CERTIFICATION_VECTOR_MAX_COMPRESSED_BYTES,
      "maximum_controls_witnesses": CERTIFICATION_VECTOR_MAX_CONTROLS_WITNESSES,
      "maximum_segments": CERTIFICATION_VECTOR_MAX_SEGMENTS,
      "selected_compressed_bytes": SEGMENT.size_bytes,
      "selected_controls_witnesses": (
        int(coverage["controls_total"])
        + sum(int(value) for value in source_exclusions.values())
      ),
    },
    "domain": "blatv2-cross-architecture-vector-v3",
    "route_name": ROUTE,
    "route_provenance_seed": {
      "car_params_sha256": hashlib.sha256(CP).hexdigest(),
      "source_segment_index": SEGMENT.index,
      "source_segment_sha256": SEGMENT.sha256,
    },
    "schema_version": CERTIFICATION_VECTOR_SCHEMA_VERSION,
    "segment_results": [result],
    "selection_identity_sha256": certification_vector_selection_identity(
      candidate,
      (SEGMENT,),
    ),
    "source_manifest": source_manifest,
  }
  unsigned_vector = CertificationVector.from_manifest(unsigned)
  manifest = dict(unsigned)
  manifest["vector_identity_sha256"] = hashlib.sha256(
    _VECTOR_DOMAIN + unsigned_vector.canonical_bytes,
  ).hexdigest()
  return CertificationVector.from_manifest(manifest)


def test_no_maneuver_plan_preserves_physical_proof() -> None:
  evidence = _without_maneuver_plan()
  result = _segment_vector(
    segment=SEGMENT,
    prepared=_prepared(evidence),
  )

  physical = result["physical_plane"]
  behavior = result["behavior_plane"]
  assert physical == {
    "encoded_controls_sha256": hashlib.sha256(
      b"".join(
        _encode_controls((control,))
        for control in evidence.control_witnesses
      ),
    ).hexdigest(),
    "encoded_frames_sha256": hashlib.sha256(
      b"".join(_encode_frame(frame) for frame in FRAMES),
    ).hexdigest(),
    "exclusions": {},
    "first_retained_mono_ns": 1_000,
    "frames_retained": 2,
    "last_retained_mono_ns": 1_010,
  }
  assert behavior == {
    "controls_retained": 0,
    "encoded_controls_sha256": hashlib.sha256().hexdigest(),
    "exclusions": {"maneuver_context_missing": 2},
    "first_retained_mono_ns": None,
    "last_retained_mono_ns": None,
    "proof_eligible": False,
    "source_eligible": False,
    "source_eligibility_reason": "lateral_maneuver_plan_missing",
  }
  restored = CertificationVector.from_bytes(
    _signed_vector(result).canonical_bytes,
  )
  assert restored.manifest["segment_results"][0] == result


def test_behavior_eligible_segment_proves_both_planes() -> None:
  result = _segment_vector(segment=SEGMENT, prepared=_prepared(artifact()))
  physical = result["physical_plane"]
  behavior = result["behavior_plane"]
  assert physical["frames_retained"] == 2
  assert behavior["controls_retained"] == 2
  assert behavior["proof_eligible"] is True
  assert behavior["source_eligible"] is True
  assert behavior["source_eligibility_reason"] == "eligible"
  assert behavior["first_retained_mono_ns"] == physical["first_retained_mono_ns"]
  assert behavior["last_retained_mono_ns"] == physical["last_retained_mono_ns"]


def test_source_ineligibility_never_presents_context_as_proof() -> None:
  evidence = artifact()
  ineligible = RouteEvidenceArtifact(
    replace(
      evidence.source_identity,
      controller_source_kind="ineligible",
      behavior_eligible=False,
      behavior_ineligible_reason="controller_source_unapproved",
    ),
    CP,
    PHYSICAL,
    evidence.model_publications,
    evidence.control_witnesses,
    TORQUE,
    DELAY,
    MANEUVER,
    EVENTS,
  )
  result = _segment_vector(segment=SEGMENT, prepared=_prepared(ineligible))
  behavior = result["behavior_plane"]
  assert behavior["controls_retained"] == 2
  assert behavior["source_eligible"] is False
  assert behavior["proof_eligible"] is False


def test_empty_physical_plane_and_empty_eligible_behavior_fail_closed() -> None:
  evidence = _without_maneuver_plan()
  invalid_frames = tuple(replace(frame, inputs_valid=False) for frame in FRAMES)
  invalid_controls = tuple(
    replace(control, inputs_valid=False)
    for control in evidence.control_witnesses
  )
  physically_empty = RouteEvidenceArtifact(
    evidence.source_identity,
    CP,
    b"".join(_encode_frame(frame) for frame in invalid_frames),
    evidence.model_publications,
    invalid_controls,
    TORQUE,
    DELAY,
    (),
    EVENTS,
  )
  with pytest.raises(CertificationVectorError, match="no valid physical"):
    _segment_vector(
      segment=SEGMENT,
      prepared=_prepared(physically_empty, invalid_frames),
    )

  with pytest.raises(CertificationVectorError, match="behavior-eligible"):
    _segment_vector(
      segment=SEGMENT,
      prepared=_prepared(_without_maneuver_plan(source_eligible=True)),
    )


def test_dual_plane_accounting_and_determinism() -> None:
  first = _segment_vector(
    segment=SEGMENT,
    prepared=_prepared(_without_maneuver_plan(), pre_poll=1, segment_context=2),
  )
  second = _segment_vector(
    segment=SEGMENT,
    prepared=_prepared(_without_maneuver_plan(), pre_poll=1, segment_context=2),
  )
  assert first == second
  assert first["source_boundary_exclusions"] == {
    "pre_poll_controls": 1,
    "segment_local_measurement_context": 2,
  }
  first_vector = _signed_vector(first)
  second_vector = _signed_vector(second)
  assert first_vector.canonical_bytes == second_vector.canonical_bytes
  assert first_vector.sha256 == second_vector.sha256


@pytest.mark.parametrize(
  ("path", "value", "message"),
  (
    (("physical_plane", "frames_retained"), 1, "physical proof coverage"),
    (("physical_plane", "first_retained_mono_ns"), None, "timestamp"),
    (("behavior_plane", "source_eligible"), True, "source eligibility"),
    (("behavior_plane", "encoded_controls_sha256"), "f" * 64, "empty behavior"),
  ),
)
def test_manifest_semantic_tampering_fails_closed(
  path: tuple[str, str],
  value: object,
  message: str,
) -> None:
  vector = _signed_vector(_segment_vector(
    segment=SEGMENT,
    prepared=_prepared(_without_maneuver_plan()),
  ))
  manifest = deepcopy(vector.manifest)
  manifest.pop("vector_identity_sha256")
  segment = manifest["segment_results"][0]
  segment[path[0]][path[1]] = value
  with pytest.raises(CertificationVectorError, match=message):
    CertificationVector.from_manifest(manifest)


def test_signed_hash_tampering_and_v2_header_fail_closed() -> None:
  vector = _signed_vector(_segment_vector(
    segment=SEGMENT,
    prepared=_prepared(_without_maneuver_plan()),
  ))
  manifest = deepcopy(vector.manifest)
  manifest["segment_results"][0]["physical_plane"][
    "encoded_frames_sha256"
  ] = "f" * 64
  payload = json.dumps(
    manifest,
    allow_nan=False,
    separators=(",", ":"),
    sort_keys=True,
  ).encode()
  tampered = bytearray(vector.canonical_bytes[:16] + payload)
  assert len(tampered) == len(vector.canonical_bytes)
  with pytest.raises(CertificationVectorError, match="identity mismatch"):
    CertificationVector.from_bytes(tampered)

  old_header = bytearray(vector.canonical_bytes)
  struct.pack_into("<H", old_header, 8, 2)
  with pytest.raises(CertificationVectorError, match="header"):
    CertificationVector.from_bytes(old_header)


def test_selected_controls_bound_remains_strict() -> None:
  vector = _signed_vector(_segment_vector(
    segment=SEGMENT,
    prepared=_prepared(_without_maneuver_plan()),
  ))
  manifest = deepcopy(vector.manifest)
  manifest.pop("vector_identity_sha256")
  manifest["bounds"]["selected_controls_witnesses"] = (
    CERTIFICATION_VECTOR_MAX_CONTROLS_WITNESSES + 1
  )
  with pytest.raises(CertificationVectorError, match="exceeds its bounds"):
    CertificationVector.from_manifest(manifest)

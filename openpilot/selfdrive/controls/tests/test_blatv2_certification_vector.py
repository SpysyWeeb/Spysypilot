from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest  # noqa: TID251

from openpilot.selfdrive.controls.lib.blatv2.certification_vector import (
  CERTIFICATION_VECTOR_MAX_COMPRESSED_BYTES,
  CERTIFICATION_VECTOR_MAX_CONTROLS_WITNESSES,
  CERTIFICATION_VECTOR_MAX_SEGMENTS,
  CERTIFICATION_VECTOR_SCHEMA_VERSION,
  CertificationVector,
  CertificationVectorError,
  _VECTOR_DOMAIN,
  _scenario_exclusion_reason,
  _scenario_model_payload_valid,
  _scenario_proof_summary,
  _segment_vector,
  build_certification_vector_from_prepared_route,
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
  frame,
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


def _with_planes(
  evidence: RouteEvidenceArtifact,
  *,
  controls=None,
  models=None,
  torque=None,
  delay=None,
  maneuvers=None,
) -> RouteEvidenceArtifact:
  return RouteEvidenceArtifact(
    evidence.source_identity,
    CP,
    PHYSICAL,
    evidence.model_publications if models is None else tuple(models),
    evidence.control_witnesses if controls is None else tuple(controls),
    evidence.live_torque_parameters if torque is None else tuple(torque),
    evidence.live_delays if delay is None else tuple(delay),
    evidence.lateral_maneuver_plans if maneuvers is None else tuple(maneuvers),
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
    "domain": "blatv2-cross-architecture-vector-v5",
    "route_name": ROUTE,
    "route_provenance_seed": {
      "car_params_sha256": hashlib.sha256(CP).hexdigest(),
      "source_segment_index": SEGMENT.index,
      "source_segment_sha256": SEGMENT.sha256,
    },
    "scenario_proof": _scenario_proof_summary([result]),
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


def test_no_maneuver_plan_proves_controller_independent_scenario() -> None:
  evidence = _without_maneuver_plan()
  result = _segment_vector(
    segment=SEGMENT,
    prepared=_prepared(evidence),
  )

  physical = result["physical_plane"]
  behavior = result["behavior_plane"]
  scenario = result["scenario_plane"]
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
  assert scenario == {
    "active_controls_retained": 2,
    "controls_retained": 2,
    "encoded_controls_sha256": hashlib.sha256(
      b"".join(
        _encode_controls((control,))
        for control in evidence.control_witnesses
      ),
    ).hexdigest(),
    "exclusions": {},
    "first_retained_mono_ns": 1_000,
    "last_retained_mono_ns": 1_010,
    "proof_eligible": True,
  }
  restored = CertificationVector.from_bytes(
    _signed_vector(result).canonical_bytes,
  )
  assert restored.manifest["segment_results"][0] == result
  assert restored.manifest["scenario_proof"]["proof_eligible"] is True


def test_behavior_plane_remains_recorded_controller_proof() -> None:
  result = _segment_vector(segment=SEGMENT, prepared=_prepared(artifact()))
  physical = result["physical_plane"]
  behavior = result["behavior_plane"]
  scenario = result["scenario_plane"]
  assert physical["frames_retained"] == 2
  assert behavior["controls_retained"] == 2
  assert behavior["proof_eligible"] is True
  assert behavior["source_eligible"] is True
  assert behavior["source_eligibility_reason"] == "eligible"
  assert behavior["first_retained_mono_ns"] == physical["first_retained_mono_ns"]
  assert behavior["last_retained_mono_ns"] == physical["last_retained_mono_ns"]
  assert scenario["controls_retained"] == 0
  assert scenario["active_controls_retained"] == 0
  assert scenario["exclusions"] == {"active_maneuver_override": 2}
  assert scenario["proof_eligible"] is False


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

  without_maneuver = _without_maneuver_plan()
  scenario_evidence = RouteEvidenceArtifact(
    replace(
      without_maneuver.source_identity,
      behavior_ineligible_reason="controller_source_unapproved",
    ),
    CP,
    PHYSICAL,
    without_maneuver.model_publications,
    without_maneuver.control_witnesses,
    TORQUE,
    DELAY,
    (),
    EVENTS,
  )
  scenario = _segment_vector(
    segment=SEGMENT,
    prepared=_prepared(scenario_evidence),
  )["scenario_plane"]
  assert scenario["controls_retained"] == 2
  assert scenario["proof_eligible"] is True


def test_route_scenario_proof_does_not_fit_eligibility_to_active_sample() -> None:
  evidence = _without_maneuver_plan()
  inactive_controls = tuple(
    replace(control, lateral_active=False)
    for control in evidence.control_witnesses
  )
  inactive = _segment_vector(
    segment=SEGMENT,
    prepared=_prepared(_with_planes(evidence, controls=inactive_controls)),
  )
  active = _segment_vector(segment=SEGMENT, prepared=_prepared(evidence))

  inactive_only = _scenario_proof_summary([inactive])
  assert inactive_only["controls_retained"] == 2
  assert inactive_only["active_controls_retained"] == 0
  assert inactive_only["proof_eligible"] is True

  mixed = _scenario_proof_summary([inactive, active])
  assert mixed["controls_retained"] == 4
  assert mixed["active_controls_retained"] == 2
  assert mixed["proof_eligible"] is True
  assert mixed["selected_inputs_sha256"] != inactive_only[
    "selected_inputs_sha256"
  ]


def test_inactive_or_invalid_maneuver_override_is_not_scenario_authority() -> None:
  evidence = artifact()
  inactive_controls = tuple(
    replace(control, lateral_active=False)
    for control in evidence.control_witnesses
  )
  inactive = _segment_vector(
    segment=SEGMENT,
    prepared=_prepared(_with_planes(evidence, controls=inactive_controls)),
  )["scenario_plane"]
  assert inactive["controls_retained"] == 2
  assert inactive["active_controls_retained"] == 0
  assert inactive["proof_eligible"] is True

  invalid_maneuver = _segment_vector(
    segment=SEGMENT,
    prepared=_prepared(_with_planes(
      evidence,
      maneuvers=(replace(MANEUVER[0], message_valid=False),),
    )),
  )["scenario_plane"]
  assert invalid_maneuver["controls_retained"] == 2
  assert invalid_maneuver["active_controls_retained"] == 2
  assert invalid_maneuver["proof_eligible"] is True


def test_exact_unhealthy_ignored_torque_payload_remains_authoritative() -> None:
  evidence = _without_maneuver_plan()
  controls = tuple(
    replace(
      control,
      live_torque_parameters_checks_passed=False,
      live_torque_parameters_health_exact=True,
    )
    for control in evidence.control_witnesses
  )
  ignored_payload = replace(
    TORQUE[0],
    lat_accel_factor=-1.0,
    friction=-1.0,
  )
  scenario = _segment_vector(
    segment=SEGMENT,
    prepared=_prepared(_with_planes(
      evidence,
      controls=controls,
      torque=(ignored_payload,),
    )),
  )["scenario_plane"]
  assert scenario["controls_retained"] == 2
  assert scenario["active_controls_retained"] == 2
  assert scenario["exclusions"] == {}
  assert scenario["proof_eligible"] is True


@pytest.mark.parametrize("value", (float("nan"), float("inf")))
def test_scenario_model_payload_rejects_nonfinite_action_and_plan_times(
  value: float,
) -> None:
  action_time = deepcopy(model(0))
  object.__setattr__(action_time, "desired_curvature_time_s", value)
  plan_time = deepcopy(model(0))
  object.__setattr__(
    plan_time,
    "plan_times",
    (value, *plan_time.plan_times[1:]),
  )

  assert not _scenario_model_payload_valid(action_time)
  assert not _scenario_model_payload_valid(plan_time)


@pytest.mark.parametrize(
  ("publication", "value", "reason"),
  (
    ("lat_accel_factor", float("nan"), "live_torque_payload_invalid"),
    ("lat_accel_offset", float("inf"), "live_torque_payload_invalid"),
    ("friction", float("nan"), "live_torque_payload_invalid"),
    ("lateral_delay_s", float("nan"), "live_delay_payload_invalid"),
    ("lateral_delay_s", float("inf"), "live_delay_payload_invalid"),
  ),
)
def test_scenario_consumed_calibration_rejects_nonfinite_values(
  publication: str,
  value: float,
  reason: str,
) -> None:
  evidence = _without_maneuver_plan()
  torque = evidence.live_torque_parameters
  delay = evidence.live_delays
  if publication == "lateral_delay_s":
    delay = (replace(delay[0], lateral_delay_s=value),)
  else:
    torque = (replace(torque[0], **{publication: value}),)
  linked = SimpleNamespace(
    model_publications=evidence.model_publications,
    live_torque_parameters=torque,
    live_delays=delay,
    lateral_maneuver_plans=evidence.lateral_maneuver_plans,
  )

  assert _scenario_exclusion_reason(
    evidence.control_witnesses[0],
    FRAMES[0],
    linked,
  ) == reason


@pytest.mark.parametrize(
  ("case", "reason"),
  (
    ("message_invalid", "scenario_message_invalid"),
    ("model_missing", "model_context_missing"),
    ("model_future", "model_context_missing"),
    ("model_message_invalid", "model_message_invalid"),
    ("model_not_alive", "model_not_alive"),
    ("model_native_grid_invalid", "model_native_grid_invalid"),
    ("applied_count_missing", "applied_torque_context_missing"),
    ("live_torque_missing", "live_torque_context_missing"),
    ("live_torque_health_inexact", "live_torque_health_inexact"),
    ("live_torque_payload_invalid", "live_torque_payload_invalid"),
    ("live_delay_missing", "live_delay_context_missing"),
    ("live_delay_payload_invalid", "live_delay_payload_invalid"),
    ("maneuver_link_invalid", "maneuver_context_invalid"),
    ("active_maneuver_override", "active_maneuver_override"),
  ),
)
def test_scenario_plane_excludes_non_authoritative_inputs(
  case: str,
  reason: str,
) -> None:
  evidence = _without_maneuver_plan()
  controls = list(evidence.control_witnesses)
  models = list(evidence.model_publications)
  torque = list(evidence.live_torque_parameters)
  delay = list(evidence.live_delays)
  maneuvers = list(evidence.lateral_maneuver_plans)
  if case == "message_invalid":
    controls = [replace(value, message_valid=False) for value in controls]
  elif case == "model_missing":
    controls = [
      replace(value, model_publication_index=-1, model_link_valid=False)
      for value in controls
    ]
  elif case == "model_future":
    models = [
      replace(value, mono_time_ns=2_000 + index, timestamp_eof_ns=1_900 + index)
      for index, value in enumerate(models)
    ]
  elif case == "model_message_invalid":
    models = [replace(value, message_valid=False) for value in models]
  elif case == "model_not_alive":
    controls = [replace(value, model_message_alive=False) for value in controls]
  elif case == "model_native_grid_invalid":
    models = [
      replace(
        value,
        plan_times=(),
        orientation_rate_z=(),
        velocity_x=(),
        native_grid_valid=False,
      )
      for value in models
    ]
  elif case == "applied_count_missing":
    controls = [
      replace(value, torque_output_can_count=0, torque_output_can_valid=False)
      for value in controls
    ]
  elif case == "live_torque_missing":
    controls = [
      replace(
        value,
        live_torque_parameters_index=-1,
        live_torque_parameters_available=False,
      )
      for value in controls
    ]
  elif case == "live_torque_health_inexact":
    controls = [
      replace(value, live_torque_parameters_health_exact=False)
      for value in controls
    ]
  elif case == "live_torque_payload_invalid":
    torque = [replace(value, lat_accel_factor=-1.0) for value in torque]
  elif case == "live_delay_missing":
    controls = [
      replace(value, live_delay_index=-1, live_delay_available=False)
      for value in controls
    ]
  elif case == "live_delay_payload_invalid":
    delay = [replace(value, lateral_delay_s=-1.0) for value in delay]
  elif case == "maneuver_link_invalid":
    controls = [
      replace(
        value,
        lateral_maneuver_plan_index=0,
        maneuver_plan_available=False,
      )
      for value in controls
    ]
    maneuvers = [MANEUVER[0]]
  elif case == "active_maneuver_override":
    controls = [
      replace(
        value,
        lateral_maneuver_plan_index=0,
        maneuver_plan_available=True,
      )
      for value in controls
    ]
    maneuvers = [MANEUVER[0]]
  else:
    raise AssertionError(f"unhandled scenario case: {case}")
  mutated = _with_planes(
    evidence,
    controls=controls,
    models=models,
    torque=torque,
    delay=delay,
    maneuvers=maneuvers,
  )
  scenario = _segment_vector(
    segment=SEGMENT,
    prepared=_prepared(mutated),
  )["scenario_plane"]
  assert scenario["controls_retained"] == 0
  assert scenario["active_controls_retained"] == 0
  assert scenario["exclusions"] == {reason: 2}
  assert scenario["proof_eligible"] is False


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


def test_scenario_plane_applies_physical_exclusion_first() -> None:
  evidence = _without_maneuver_plan()
  frames = (replace(FRAMES[0], inputs_valid=False), FRAMES[1])
  controls = (
    replace(evidence.control_witnesses[0], inputs_valid=False),
    evidence.control_witnesses[1],
  )
  physical = b"".join(_encode_frame(frame) for frame in frames)
  partially_invalid = RouteEvidenceArtifact(
    evidence.source_identity,
    CP,
    physical,
    evidence.model_publications,
    controls,
    TORQUE,
    DELAY,
    (),
    EVENTS,
  )
  scenario = _segment_vector(
    segment=SEGMENT,
    prepared=_prepared(partially_invalid, frames),
  )["scenario_plane"]
  assert scenario["controls_retained"] == 1
  assert scenario["active_controls_retained"] == 1
  assert scenario["exclusions"] == {"physical_plane_inputs_invalid": 1}


def test_dual_plane_accounting_and_determinism() -> None:
  first = _segment_vector(
    segment=SEGMENT, prepared=_prepared(_without_maneuver_plan(), pre_poll=1),
  )
  second = _segment_vector(
    segment=SEGMENT, prepared=_prepared(_without_maneuver_plan(), pre_poll=1),
  )
  assert first == second
  assert first["source_boundary_exclusions"] == {"pre_poll_controls": 1}
  first_vector = _signed_vector(first)
  second_vector = _signed_vector(second)
  assert first_vector.canonical_bytes == second_vector.canonical_bytes
  assert first_vector.sha256 == second_vector.sha256

  reset_segment = _segment_vector(
    segment=SEGMENT,
    prepared=_prepared(_without_maneuver_plan(), segment_context=2),
  )
  with pytest.raises(CertificationVectorError, match="source boundary"):
    _signed_vector(reset_segment)


def test_full_route_projection_preserves_cross_segment_control_state() -> None:
  second_segment = RouteSegment(1, Path("rlog-1.zst"), "b" * 64, 2345)
  route = RouteCandidate(ROUTE, 1, (SEGMENT, second_segment))
  frames = (*FRAMES, frame(2_000), frame(2_010))
  models = (
    model(0),
    model(1),
    replace(model(0), segment_index=1, ordinal=0, mono_time_ns=1_950,
            timestamp_eof_ns=1_940, frame_id=200),
    replace(model(1), segment_index=1, ordinal=1, mono_time_ns=1_951,
            timestamp_eof_ns=1_941, frame_id=201),
  )
  torque = (
    *TORQUE,
    replace(TORQUE[0], segment_index=1, ordinal=0, mono_time_ns=1_940),
  )
  delay = (
    *DELAY,
    replace(DELAY[0], segment_index=1, ordinal=0, mono_time_ns=1_941),
  )
  maneuvers = (
    *MANEUVER,
    replace(MANEUVER[0], segment_index=1, ordinal=0, mono_time_ns=1_942),
  )
  controls = (
    witness(0),
    witness(1),
    replace(
      witness(0),
      segment_index=1,
      ordinal=0,
      mono_time_ns=2_000,
      physical_record_index=2,
      model_publication_index=2,
      live_torque_parameters_index=1,
      live_delay_index=1,
      lateral_maneuver_plan_index=1,
      poll_mono_time_ns=1_990,
      state_sample_mono_ns=1_980,
      live_parameters_mono_ns=1_970,
      car_output_report_mono_ns=1_965,
      car_output_effective_mono_ns=1_955,
      car_control_mono_ns=2_001,
      live_torque_parameters_checks_passed=False,
      live_torque_parameters_health_exact=False,
    ),
    replace(
      witness(1),
      segment_index=1,
      ordinal=1,
      mono_time_ns=2_010,
      physical_record_index=3,
      model_publication_index=3,
      live_torque_parameters_index=1,
      live_delay_index=1,
      lateral_maneuver_plan_index=1,
      poll_mono_time_ns=2_000,
      state_sample_mono_ns=1_990,
      live_parameters_mono_ns=1_970,
      car_output_report_mono_ns=1_975,
      car_output_effective_mono_ns=1_965,
      car_control_mono_ns=2_011,
      live_torque_parameters_checks_passed=False,
      live_torque_parameters_health_exact=False,
    ),
  )
  identity = replace(
    source(),
    route_segment_sha256=(SEGMENT.sha256, second_segment.sha256),
    route_segment_size_bytes=(SEGMENT.size_bytes, second_segment.size_bytes),
    preparation_provenance=dict(PROVENANCE),
    physical_record_count=4,
    controls_witness_count=4,
  )
  evidence = RouteEvidenceArtifact(
    identity,
    CP,
    b"".join(_encode_frame(value) for value in frames),
    models,
    controls,
    torque,
    delay,
    maneuvers,
    EVENTS,
  )
  prepared = _prepared(evidence, frames)

  vector = build_certification_vector_from_prepared_route(route, prepared)
  second = vector.manifest["segment_results"][1]
  expected_controls = hashlib.sha256(
    _encode_controls(controls[2:]),
  ).hexdigest()
  assert second["physical_plane"] == {
    "encoded_controls_sha256": expected_controls,
    "encoded_frames_sha256": hashlib.sha256(
      b"".join(_encode_frame(value) for value in frames[2:]),
    ).hexdigest(),
    "exclusions": {},
    "first_retained_mono_ns": 2_000,
    "frames_retained": 2,
    "last_retained_mono_ns": 2_010,
  }
  assert second["source_boundary_exclusions"] == {}
  reset_controls = tuple(
    replace(value, physical_record_index=index)
    for index, value in enumerate(controls[2:])
  )
  assert hashlib.sha256(_encode_controls(reset_controls)).hexdigest() != expected_controls

  split_context = deepcopy(vector.manifest)
  split_context.pop("vector_identity_sha256")
  split_context["segment_results"][1]["encoded_source_plane_sha256"][
    "route_evidence_complete"
  ] = "f" * 64
  with pytest.raises(CertificationVectorError, match="full-route preparation context"):
    CertificationVector.from_manifest(split_context)


@pytest.mark.parametrize(
  ("path", "value", "message"),
  (
    (("physical_plane", "frames_retained"), 1, "physical proof coverage"),
    (("physical_plane", "first_retained_mono_ns"), None, "timestamp"),
    (("behavior_plane", "source_eligible"), True, "source eligibility"),
    (("behavior_plane", "encoded_controls_sha256"), "f" * 64, "empty behavior"),
    (("scenario_plane", "controls_retained"), 1, "scenario proof coverage"),
    (("scenario_plane", "active_controls_retained"), 3, "scenario proof coverage"),
    (("scenario_plane", "proof_eligible"), False, "scenario proof coverage"),
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


def test_route_scenario_recount_and_source_identity_tampering_fail_closed() -> None:
  vector = _signed_vector(_segment_vector(
    segment=SEGMENT,
    prepared=_prepared(_without_maneuver_plan()),
  ))
  manifest = deepcopy(vector.manifest)
  manifest.pop("vector_identity_sha256")
  manifest["scenario_proof"]["active_controls_retained"] = 1
  with pytest.raises(CertificationVectorError, match="scenario proof identity"):
    CertificationVector.from_manifest(manifest)

  manifest = deepcopy(vector.manifest)
  manifest.pop("vector_identity_sha256")
  manifest["segment_results"][0]["encoded_source_plane_sha256"][
    "models"
  ] = "f" * 64
  with pytest.raises(CertificationVectorError, match="scenario proof identity"):
    CertificationVector.from_manifest(manifest)


def test_signed_hash_tampering_and_v4_header_fail_closed() -> None:
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
  struct.pack_into("<H", old_header, 8, 4)
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

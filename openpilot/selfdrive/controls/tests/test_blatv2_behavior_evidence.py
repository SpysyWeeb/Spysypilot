from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
import unittest

from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import (
  BehaviorControlResponse,
  BehaviorSample,
  BehaviorSourceIdentity,
  BehaviorWindow,
  EventLocator,
  ManeuverClass,
  ManeuverPhase,
  SparseModelBehaviorIntent,
  assemble_behavior_sample,
  derive_behavior_reference,
)
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import RackMappingSnapshot


def source_identity() -> BehaviorSourceIdentity:
  return BehaviorSourceIdentity(
    controller_name="exact-test-controller",
    controller_artifact_sha256="a" * 64,
    source_openpilot_commit="b" * 40,
    opendbc_commit="c" * 40,
    panda_commit="d" * 40,
    evidence_schema_version=1,
  )


def rack_mapping() -> RackMappingSnapshot:
  return RackMappingSnapshot(
    mass_kg=2200.0,
    wheelbase_m=2.9,
    center_to_front_m=1.2,
    center_to_rear_m=1.7,
    tire_stiffness_front=160_000.0,
    tire_stiffness_rear=200_000.0,
    steer_ratio_rear=0.0,
    steer_ratio=15.0,
    roll_rad=0.0,
    angle_offset_deg=0.0,
    valid=True,
  )


def sample(index: int, *, intervention: bool = False, steering_pressed: bool = False) -> BehaviorSample:
  return BehaviorSample(
    mono_time_ns=1_000_000_000 + index * 10_000_000,
    route_time_s=1.0 + index * 0.01,
    speed_mps=5.0,
    scalar_curvature_1pm=0.01,
    desired_curvature_1pm=0.01,
    anchored_curvature_1pm=0.01,
    desired_rack_angle_deg=30.0,
    desired_rack_rate_deg_s=2.0,
    desired_rack_accel_deg_s2=1.0,
    measured_curvature_1pm=0.009,
    measured_rack_angle_deg=28.0,
    measured_rack_rate_deg_s=1.8,
    measured_rack_accel_deg_s2=0.9,
    raw_requested_torque=0.2,
    planned_requested_torque=0.2,
    reachable_envelope_torque=0.2,
    envelope_applied_torque=0.19,
    torque_headroom=0.8,
    actuator_constrained=False,
    steering_request_active=True,
    maximum_authority_required=False,
    lateral_active=True,
    inputs_valid=True,
    steering_pressed=steering_pressed,
    controller_fault=False,
    driver_intervention_onset=intervention,
  )


class TestBehaviorEvidence(unittest.TestCase):
  def test_phase_and_class_vocabulary_is_explicit(self):
    self.assertEqual(
      {phase.value for phase in ManeuverPhase},
      {
        "straight_quasi_steady",
        "turn_in",
        "hold",
        "release_unwind",
        "direct_handoff",
      },
    )
    self.assertIn(ManeuverClass.LANE_CHANGE, tuple(ManeuverClass))

  def test_intervention_bookmarks_and_censors_without_becoming_a_vote(self):
    window = BehaviorWindow(
      route_id="route-a",
      window_id="turn-1",
      source=source_identity(),
      maneuver_class=ManeuverClass.TURN,
      phase=ManeuverPhase.TURN_IN,
      samples=(sample(0), sample(1), sample(2, intervention=True), sample(3)),
    )

    self.assertEqual(window.clean_pre_intervention_samples, (sample(0), sample(1)))
    self.assertEqual(window.intervention_mono_time_ns, sample(2).mono_time_ns)
    self.assertFalse(window.intervention_is_quality_vote)

  def test_logger_event_is_a_locator_and_does_not_choose_the_phase(self):
    locator = EventLocator(
      event_type="lat.turnStopTurn",
      occurred_mono_time_ns=1_010_000_000,
      analysis_window_before_s=6.0,
      analysis_window_after_s=2.0,
      severity="warning",
    )
    window = BehaviorWindow(
      route_id="route-a",
      window_id="release-1",
      source=source_identity(),
      maneuver_class=ManeuverClass.CURVE,
      phase=ManeuverPhase.RELEASE_UNWIND,
      samples=(sample(0), sample(1)),
      event_locators=(locator,),
    )

    self.assertEqual(window.phase, ManeuverPhase.RELEASE_UNWIND)
    self.assertEqual(window.event_locators[0].event_type, "lat.turnStopTurn")

  def test_sparse_model_is_mapped_at_each_control_frame_and_rejects_future_data(self):
    model = SparseModelBehaviorIntent(
      plan_origin_mono_time_ns=900_000_000,
      publication_mono_time_ns=990_000_000,
      model_frame_id=42,
      plan_valid=True,
      scalar_curvature_1pm=0.012,
      scalar_action_plan_s=0.10,
      native_times_s=(0.0, 0.1, 0.2, 0.3),
      orientation_rates_z=(0.05, 0.10, 0.15, 0.20),
      velocities_x=(10.0, 10.0, 10.0, 10.0),
    )
    response = BehaviorControlResponse(
      mono_time_ns=1_000_000_000,
      route_time_s=1.0,
      speed_mps=5.0,
      transport_delay_s=0.05,
      live_rack_mapping=rack_mapping(),
      nominal_rack_mapping=rack_mapping(),
      measured_curvature_1pm=0.009,
      measured_rack_angle_deg=18.0,
      measured_rack_rate_deg_s=1.5,
      measured_rack_accel_deg_s2=0.4,
      raw_requested_torque=0.2,
      planned_requested_torque=0.2,
      reachable_envelope_torque=0.2,
      envelope_applied_torque=0.19,
      torque_headroom=0.8,
      actuator_constrained=False,
      steering_request_active=True,
      maximum_authority_required=False,
      lateral_active=True,
      inputs_valid=True,
      steering_pressed=False,
      controller_fault=False,
      driver_intervention_onset=False,
    )

    reference = derive_behavior_reference(model, response)
    result = assemble_behavior_sample(reference, response)
    self.assertEqual(result.desired_curvature_1pm, result.anchored_curvature_1pm)
    self.assertEqual(result.scalar_curvature_1pm, 0.012)

    slow = derive_behavior_reference(model, replace(response, speed_mps=3.0))
    fast = derive_behavior_reference(model, replace(response, speed_mps=25.0))
    self.assertNotEqual(slow.desired_rack_angle_deg, fast.desired_rack_angle_deg)
    self.assertNotEqual(slow.desired_rack_rate_deg_s, fast.desired_rack_rate_deg_s)

    future = SparseModelBehaviorIntent(
      plan_origin_mono_time_ns=1_000_000_000,
      publication_mono_time_ns=1_000_000_001,
      model_frame_id=43,
      plan_valid=True,
      scalar_curvature_1pm=0.0,
      scalar_action_plan_s=0.0,
      native_times_s=(0.0, 0.1),
      orientation_rates_z=(0.0, 0.0),
      velocities_x=(5.0, 5.0),
    )
    with self.assertRaisesRegex(ValueError, "publication"):
      derive_behavior_reference(future, response)

    malformed = SparseModelBehaviorIntent(
      plan_origin_mono_time_ns=900_000_000,
      publication_mono_time_ns=990_000_000,
      model_frame_id=44,
      plan_valid=False,
      scalar_curvature_1pm=0.012,
      scalar_action_plan_s=0.10,
      native_times_s=(0.0, float("nan")),
      orientation_rates_z=(0.0,),
      velocities_x=(),
    )
    degraded = derive_behavior_reference(malformed, response)
    self.assertFalse(degraded.valid)
    self.assertEqual(degraded.anchored_curvature_1pm, 0.012)
    self.assertEqual(degraded.anchored_curvature_rate_1pm_s, 0.0)

  def test_contract_is_immutable_canonical_and_contains_no_lane_target(self):
    negative_zero = sample(0)
    window = BehaviorWindow(
      route_id="route-a",
      window_id="hold-1",
      source=source_identity(),
      maneuver_class=ManeuverClass.CURVE,
      phase=ManeuverPhase.HOLD,
      samples=(negative_zero, sample(1)),
    )
    first = window.to_json()
    second = window.to_json()

    self.assertEqual(first, second)
    self.assertEqual(window.sha256, window.sha256)
    self.assertEqual(json.dumps(json.loads(first), sort_keys=True, separators=(",", ":")), first)
    self.assertFalse(any("lane" in field.lower() for field in BehaviorSample.__dataclass_fields__))
    self.assertIn("pandaCommit", window.source.to_dict())
    with self.assertRaises(FrozenInstanceError):
      negative_zero.speed_mps = 6.0


if __name__ == "__main__":
  unittest.main()

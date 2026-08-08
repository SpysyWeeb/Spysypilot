from __future__ import annotations

import unittest

from openpilot.cereal import log
from openpilot.cereal.services import SERVICE_LIST


# These manifests are the permanent wire contract. LateralTorqueState @0..12
# originate in stock, @13..104 in origin/combo, and @105..126 in the v234
# historical schema. BlatV2Shadow @0..63 originates in origin/combo and
# @64..85 in v234, except that combo's original non-Void @15..24/@26 fields
# remain authoritative.
LATERAL_TORQUE_MANIFEST_TEXT = """
0 active Bool
1 error Float32
2 p Float32
3 i Float32
4 d Float32
5 f Float32
6 output Float32
7 saturated Bool
8 errorRate Float32
9 actualLateralAccel Float32
10 desiredLateralAccel Float32
11 desiredLateralJerk Float32
12 version Int32
13 measurementRate Float32
14 rateBrake Float32
15 rateBrakeScale Float32
16 delayedDesiredCurvature Float32
17 legacyDesiredLateralAccel Float32
18 speedAlignmentCorrection Float32
19 actuationSpeed Float32
20 currentSpeedDesiredLateralAccel Float32
21 speedProjectionCorrection Float32
22 longitudinalLateralAccelRate Float32
23 rateBrakeSpeedScale Float32
24 referenceVersion Int32
25 referenceBaseCurvature Float32
26 referenceOutputCurvature Float32
27 referencePreviewTime Float32
28 referencePreviewExtraTime Float32
29 referenceTargetTorque Float32
30 referenceAppliedTorque Float32
31 referenceUnwindScale Float32
32 referenceAuthorityRestored Float32
33 referencePreviewCorrection Float32
34 referenceRate Float32
35 trackingMeasurementRate Float32
36 rateTrackingError Float32
37 rateTrackingCorrection Float32
38 rateTrackingSpeedScale Float32
39 referenceCurvatureRate Float32
40 measurementCurvatureRate Float32
41 cascadePositionError Float32
42 cascadeCatchupRate Float32
43 cascadeDesiredRate Float32
44 cascadeRateError Float32
45 actuatorAppliedLateralAccel Float32
46 actuatorStateCorrection Float32
47 cascadePScale Float32
48 unwindBrakeActivation Float32
49 unwindTorqueZeroTime Float32
50 unwindProjectedPositionError Float32
51 unwindTorqueCorrection Float32
52 cascadeBasePScale Float32
53 dampingTurnInBlocked Bool
54 referenceGeometricTargetTorque Float32
55 referenceNeutralTorque Float32
56 referenceReachableTargetTorque Float32
57 unwindEffectivePhase Float32
58 unwindPhaseDirection Float32
59 unwindDeliveryGap Float32
60 unwindPhaseOverspeed Float32
61 unwindNeutralTorque Float32
62 unwindTorqueNeutralTime Float32
63 unwindSameEpisode Bool
64 unwindOppositeTime Float32
65 unwindEpisodeArmed Bool
66 finiteDifferenceReferenceCurvatureRate Float32
67 trajectoryReferenceCurvatureRate Float32
68 trajectoryReferenceRateValid Bool
69 trajectoryReferenceInnovation Float32
70 filteredTrajectoryReferenceInnovation Float32
71 referenceSustainedUnwindScale Float32
72 referenceEpisodeTargetTorque Float32
73 referenceEpisodeLateralAccel Float32
74 blatV2Status UInt8
75 blatV2ComputeTimeSeconds Float64
76 blatV2OutputValid Bool
77 blatV2InvalidFrames UInt16
78 blatV2RecoveryOkFrames UInt8
79 blatV2CommandTorque Float64
80 blatV2RawCommandTorque Float64
81 blatV2FeedforwardTorque Float64
82 blatV2FeedbackTorque Float64
83 blatV2DesiredAngleDeg Float64
84 blatV2DesiredRateDegS Float64
85 blatV2DesiredAccelerationDegS2 Float64
86 blatV2PredictedAngleDeg Float64
87 blatV2PredictedRateDegS Float64
88 blatV2RequiredAccelerationDegS2 Float64
89 blatV2ActionSpeedMps Float64
90 blatV2AligningTorque Float64
91 blatV2FrictionTorque Float64
92 blatV2DynamicTorque Float64
93 blatV2ActionTimeSeconds Float64
94 blatV2SlewConstrained Bool
95 blatV2BreakawayActive Bool
96 blatV2BreakawayPersistenceFrames UInt16
97 blatV2HorizonAssistActive Bool
98 blatV2HorizonTorqueDemand Float64
99 blatV2HorizonDemandTimeSeconds Float64
100 blatV2NoLeadLimited Bool
101 blatV2PredictionDelaySeconds Float64
102 blatV2SignedRackRateDegS Float64
103 blatV2HeldStaticLoad Float64
104 blatV2RackStationary Bool
105 blatV2AdaptiveModelVersion UInt16
106 blatV2AdaptiveGain Float64
107 blatV2AdaptiveDamping Float64
108 blatV2AdaptiveAlignGain Float64
109 blatV2AdaptiveMovingFriction Float64
110 blatV2AdaptiveRoadLoad Float64
111 blatV2AdaptiveConfidence Float64
112 blatV2AdaptiveSampleCount UInt64
113 blatV2AdaptiveLearningActive Bool
114 blatV2AdaptiveRateDegS Float64
115 blatV2AdaptiveAccelerationDegS2 Float64
116 blatV2AdaptiveRateResolutionDegS Float64
117 blatV2AdaptiveResponseLagSeconds Float64
118 blatV2AdaptiveOutcomeConfidence Float64
119 blatV2AdaptiveOutcomePhase UInt8
120 blatV2AdaptiveOutcomeSignedLagSeconds Float64
121 blatV2AdaptiveOutcomeTrackingErrorFraction Float64
122 blatV2AdaptiveOutcomeReleaseOvershootMps2 Float64
123 blatV2AdaptiveOutcomeRoughnessPerS Float64
124 blatV2AdaptiveOutcomeBurstPerS Float64
125 blatV2AdaptiveOutcomeCount UInt64
126 blatV2AdaptiveOutcomeLearningActive Bool
127 modularArchitecture Text
128 modularControllerVersion UInt16
129 modularSelection UInt8
130 modularBindingReason UInt8
131 modularCandidateStatus UInt8
132 modularCoreStatus UInt8
133 modularArtifactHash Text
134 modularProfileHash Text
135 modularPolicyHash Text
136 modularRuntimeIdentityHash Text
137 modularSourceOpenpilotCommit Text
138 modularOpendbcCommit Text
139 modularControlWitnessMonoTime UInt64
140 modularStateSampleMonoTime UInt64
141 modularModelPublicationMonoTime UInt64
142 modularModelTimestampEof UInt64
143 modularDesiredCurvatureTimeSeconds Float64
144 modularRawScalarCurvature Float64
145 modularReferenceCurvature Float64
146 modularRawTorque Float64
147 modularCommandTorque Float64
148 modularFeasibleTorque Float64
149 modularAligningTorque Float64
150 modularFrictionTorque Float64
151 modularMotionFeedforwardTorque Float64
152 modularPositionFeedbackTorque Float64
153 modularRateFeedbackTorque Float64
154 modularDisturbanceTorque Float64
155 modularDesiredAngleDeg Float64
156 modularDesiredRateDegS Float64
157 modularDesiredAccelerationDegS2 Float64
158 modularMeasuredAngleDeg Float64
159 modularMeasuredRateDegS Float64
160 modularMeasuredAccelerationDegS2 Float64
161 modularPredictedAngleDeg Float64
162 modularPredictedRateDegS Float64
163 modularPreviousAppliedCounts Int32
164 modularPreviousAppliedTorque Float64
165 modularDriverTorque Float64
166 modularConstraintActive Bool
167 modularConstraintReason UInt8
168 modularFeasibilityStatus UInt8
169 modularSafetyState UInt8
170 modularControlsValid Bool
171 modularCarControlValid Bool
172 modularInvalidFrames UInt16
173 modularRecoveryOkFrames UInt8
174 modularPreviousOutputConstrained Bool
175 modularPreviousActuatorConstrained Bool
176 modularVehicleStateValid Bool
177 modularLiveParametersValid Bool
178 modularIntentStatus UInt8
179 modularComputeTimeSeconds Float64
180 modularStateAgeSeconds Float64
181 modularTotalPredictionHorizonSeconds Float64
182 modularTransportDelaySeconds Float64
183 modularCommandEnvelopeApplied Bool
184 modularManeuverForcedStock Bool
185 modularProductionEnvelopeVerified Bool
186 modularSelectionBound Bool
187 modularHorizonPolicyHash Text
188 modularPlannedTorque Float64
189 modularPlannedCounts Int32
190 modularReactiveTorque Float64
191 modularReactiveCounts Int32
192 modularRawRequestedCounts Int32
193 modularRawToPlannedResidualCounts Int32
194 modularRawToPlannedUnmetTorque Float64
195 modularPreparationActive Bool
196 modularPreparationScheduled Bool
197 modularHorizonStatus UInt8
198 modularHorizonValid Bool
199 modularDriverSuppressed Bool
200 modularFutureBandReachable Bool
201 modularFirstUnreachableIndex Int16
202 modularFirstUnreachableTimeSeconds Float64
203 modularMaximumBandResidualCounts UInt16
204 modularMaximumPathLeadDeg Float64
205 modularMaximumPathRateLeadDegS Float64
206 modularPathLeadConstrainedSamples UInt16
207 modularMaximumAuthorityRequired Bool
208 modularMaximumAuthorityActive Bool
209 modularMaximumUrgency Float64
210 modularPreviousCommandCounts Int32
211 modularRecordedAppliedTorque Float64
212 modularSteeringRequestActive Bool
213 modularSteeringRequestValid Bool
214 modularSteeringRequestFaultAvoidanceCounter UInt8
215 modularControlCadenceValid Bool
216 modularTransportReprimed Bool
217 modularAdapterException Bool
218 modularRawToPlannedConstrained Bool
219 modularFinalExpectedCounts Int32
220 modularFinalCountResidual Int32
221 modularFinalCountMatchValid Bool
222 modularFinalLimiterAltered Bool
"""

BLAT_V2_SHADOW_MANIFEST_TEXT = """
0 shadowVersion UInt16
1 valid Bool
2 referenceCurvature Float64
3 torqueDemand Float64
4 feasibleTorque Float64
5 plantResidual Float64
6 scalarPlanDisagreement Float64
7 horizon Float64
8 computeTimeSeconds Float64
9 vEgo Float64
10 aligningTorque Float64
11 alignInputsValid Bool
12 disturbanceEstimate Float64
13 observerStatus UInt8
14 observerUnconstrainedUpdate Float64
15 mpcCommandTorque Float64
16 mpcStatus UInt8
17 mpcCandidateCount UInt16
18 mpcOptimalityResidual Float64
19 mpcComputeTimeSeconds Float64
20 fallbackCommandTorque Float64
21 fallbackStatus UInt8
22 fallbackCandidateCount UInt16
23 fallbackOptimalityResidual Float64
24 fallbackComputeTimeSeconds Float64
25 sharedComputeTimeSeconds Float64
26 mpcAvailableScheduleCount UInt16
27 liveLqiCommandTorque Float64
28 liveLqiStatus UInt8
29 liveLqiComputeTimeSeconds Float64
30 liveLqiOutputValid Bool
31 liveLqiInvalidFrames UInt16
32 liveLqiRecoveryOkFrames UInt8
33 v14CommandTorque Float64
34 v14DesiredCurvature Float64
35 v14ControllerVersion Int32
36 v14Valid Bool
37 v14ComputeTimeSeconds Float64
38 liveLqiControllerVersion Int32
39 liveActionRawCommandTorque Float64
40 liveActionFeedforwardTorque Float64
41 liveActionFeedbackTorque Float64
42 liveActionDesiredAngleDeg Float64
43 liveActionDesiredRateDegS Float64
44 liveActionDesiredAccelerationDegS2 Float64
45 liveActionPredictedAngleDeg Float64
46 liveActionPredictedRateDegS Float64
47 liveActionRequiredAccelerationDegS2 Float64
48 liveActionSpeedMps Float64
49 liveActionAligningTorque Float64
50 liveActionFrictionTorque Float64
51 liveActionDynamicTorque Float64
52 liveActionTimeSeconds Float64
53 liveActionSlewConstrained Bool
54 liveActionBreakawayActive Bool
55 liveActionBreakawayPersistenceFrames UInt16
56 liveActionHorizonAssistActive Bool
57 liveActionHorizonTorqueDemand Float64
58 liveActionHorizonDemandTimeSeconds Float64
59 liveActionNoLeadLimited Bool
60 liveActionPredictionDelaySeconds Float64
61 signedRackRateDegS Float64
62 liveActionHeldStaticLoad Float64
63 rackStationary Bool
64 liveAdaptiveModelVersion UInt16
65 liveAdaptiveGain Float64
66 liveAdaptiveDamping Float64
67 liveAdaptiveAlignGain Float64
68 liveAdaptiveMovingFriction Float64
69 liveAdaptiveRoadLoad Float64
70 liveAdaptiveConfidence Float64
71 liveAdaptiveSampleCount UInt64
72 liveAdaptiveLearningActive Bool
73 liveAdaptiveRateDegS Float64
74 liveAdaptiveAccelerationDegS2 Float64
75 liveAdaptiveRateResolutionDegS Float64
76 liveAdaptiveResponseLagSeconds Float64
77 liveAdaptiveOutcomeConfidence Float64
78 liveAdaptiveOutcomePhase UInt8
79 liveAdaptiveOutcomeSignedLagSeconds Float64
80 liveAdaptiveOutcomeTrackingErrorFraction Float64
81 liveAdaptiveOutcomeReleaseOvershootMps2 Float64
82 liveAdaptiveOutcomeRoughnessPerS Float64
83 liveAdaptiveOutcomeBurstPerS Float64
84 liveAdaptiveOutcomeCount UInt64
85 liveAdaptiveOutcomeLearningActive Bool
"""

MODULAR_SHADOW_MANIFEST_TEXT = """
86 modularSchemaVersion UInt16
87 modularRuntimeVehicleIdentityHash Text
88 modularPolicyHash Text
89 modularProfileHash Text
90 modularModelFrameId UInt32
91 modularIntentStatus UInt8
92 modularCoreStatus UInt8
93 modularValid Bool
94 modularIntentUsable Bool
95 modularProfileQualified Bool
96 modularReferenceValid Bool
97 modularScalarOnly Bool
98 modularNominalMappingUsed Bool
99 modularLiveParametersValid Bool
100 modularRecordedActuatorConstrained Bool
101 modularFeasibilityConstrained Bool
102 modularObserverSaturated Bool
103 modularRawTorque Float64
104 modularFeasibleTorque Float64
105 modularUnmetTorque Float64
106 modularAligningTorque Float64
107 modularFrictionTorque Float64
108 modularMotionFeedforwardTorque Float64
109 modularPositionFeedbackTorque Float64
110 modularRateFeedbackTorque Float64
111 modularDisturbanceTorque Float64
112 modularDesiredCurvature Float64
113 modularDesiredCurvatureRate Float64
114 modularDesiredCurvatureAcceleration Float64
115 modularDesiredAngleDeg Float64
116 modularDesiredRateDegS Float64
117 modularDesiredAccelerationDegS2 Float64
118 modularMeasuredAngleDeg Float64
119 modularMeasuredRateDegS Float64
120 modularMeasuredAccelerationDegS2 Float64
121 modularPredictedAngleDeg Float64
122 modularPredictedRateDegS Float64
123 modularPositionErrorDeg Float64
124 modularRateErrorDegS Float64
125 modularRequiredAccelerationDegS2 Float64
126 modularObserverEstimateTorque Float64
127 modularObserverInstantaneousTorque Float64
128 modularObserverStatus UInt8
129 modularProfileLowerNodeSpeedMps Float64
130 modularProfileUpperNodeSpeedMps Float64
131 modularProfileUpperWeight Float64
132 modularTorquePerLateralAccel Float64
133 modularRackGainDegS2PerTorque Float64
134 modularRackDampingPerS Float64
135 modularTransportDelaySeconds Float64
136 modularStaticFrictionTorque Float64
137 modularKineticFrictionTorque Float64
138 modularRackRateResolutionDegS Float64
139 modularProfileConfidence Float64
140 modularPlanAgeSeconds Float64
141 modularDesiredCurvatureTimeSeconds Float64
142 modularPlanTimeNowSeconds Float64
143 modularPhysicalEffectPlanSeconds Float64
144 modularCurrentSpeedMps Float64
145 modularEffectSpeedMps Float64
146 modularMeasuredPreviousAppliedTorque Float64
147 modularMeasuredDriverTorque Float64
148 modularComputeTimeSeconds Float64
149 modularModelInputValid Bool
150 modularVehicleStateValid Bool
151 modularLateralActive Bool
152 modularLateralValid Bool
153 modularActuationEnvelopeVerified Bool
154 modularStateSampleMonoTime UInt64
155 modularControlWitnessMonoTime UInt64
156 modularStateAgeSeconds Float64
157 modularTotalPredictionHorizonSeconds Float64
158 modularHorizonPolicyHash Text
"""


def _parse_manifest(text: str) -> tuple[tuple[int, str, str], ...]:
  return tuple((int(ordinal), name, field_type) for ordinal, name, field_type in (line.split() for line in text.splitlines() if line.strip()))


LATERAL_TORQUE_MANIFEST = _parse_manifest(LATERAL_TORQUE_MANIFEST_TEXT)
BLAT_V2_SHADOW_MANIFEST = _parse_manifest(
  BLAT_V2_SHADOW_MANIFEST_TEXT,
)
MODULAR_SHADOW_MANIFEST = _parse_manifest(MODULAR_SHADOW_MANIFEST_TEXT)

_PRIMITIVE_TYPE_NAMES = {
  "bool": "Bool",
  "float32": "Float32",
  "float64": "Float64",
  "int8": "Int8",
  "int16": "Int16",
  "int32": "Int32",
  "int64": "Int64",
  "text": "Text",
  "uint8": "UInt8",
  "uint16": "UInt16",
  "uint32": "UInt32",
  "uint64": "UInt64",
  "void": "Void",
}


def _field_ordinal(field) -> int:
  return int(field.proto.ordinal.explicit)


def _field_type_name(field) -> str:
  field_type = field.proto.slot.type.which()
  return _PRIMITIVE_TYPE_NAMES[field_type]


def _struct_manifest(struct_module) -> tuple[tuple[int, str, str], ...]:
  fields = (
    (
      _field_ordinal(field),
      name,
      _field_type_name(field),
    )
    for name, field in struct_module.schema.fields.items()
  )
  return tuple(sorted(fields))


class TestBLaTv2SchemaCompatibility(unittest.TestCase):
  def test_lateral_torque_manifest_is_exact(self) -> None:
    actual = _struct_manifest(log.ControlsState.LateralTorqueState)
    self.assertEqual(actual, LATERAL_TORQUE_MANIFEST)
    self.assertEqual(len(actual), 223)
    self.assertEqual(actual[-1][0], 222)
    self.assertEqual(
      tuple(ordinal for ordinal, _, _ in actual),
      tuple(range(223)),
    )

  def test_blat_v2_shadow_manifest_is_exact(self) -> None:
    actual = _struct_manifest(log.BlatV2Shadow)
    historical = tuple(field for field in actual if field[0] <= 85)
    modular = tuple(field for field in actual if field[0] >= 86)
    self.assertEqual(historical, BLAT_V2_SHADOW_MANIFEST)
    self.assertEqual(modular, MODULAR_SHADOW_MANIFEST)
    self.assertEqual(len(actual), 159)
    self.assertEqual(actual[-1][0], 158)
    self.assertEqual(
      tuple(ordinal for ordinal, _, _ in actual),
      tuple(range(159)),
    )

  def test_original_candidate_slots_remain_non_void(self) -> None:
    by_ordinal = {ordinal: (name, field_type) for ordinal, name, field_type in _struct_manifest(log.BlatV2Shadow)}
    expected = {
      15: ("mpcCommandTorque", "Float64"),
      16: ("mpcStatus", "UInt8"),
      17: ("mpcCandidateCount", "UInt16"),
      18: ("mpcOptimalityResidual", "Float64"),
      19: ("mpcComputeTimeSeconds", "Float64"),
      20: ("fallbackCommandTorque", "Float64"),
      21: ("fallbackStatus", "UInt8"),
      22: ("fallbackCandidateCount", "UInt16"),
      23: ("fallbackOptimalityResidual", "Float64"),
      24: ("fallbackComputeTimeSeconds", "Float64"),
      26: ("mpcAvailableScheduleCount", "UInt16"),
    }
    for ordinal, expected_field in expected.items():
      with self.subTest(ordinal=ordinal):
        self.assertEqual(by_ordinal[ordinal], expected_field)
        self.assertNotEqual(by_ordinal[ordinal][1], "Void")

  def test_action_timestamp_keeps_ordinal_three(self) -> None:
    field = log.ModelDataV2.Action.schema.fields["desiredCurvatureTime"]
    self.assertEqual(_field_ordinal(field), 3)
    self.assertEqual(_field_type_name(field), "Float32")

  def test_event_union_prerequisites_and_shadow_ordinals(self) -> None:
    expected = {
      152: ("lateralEvent", log.LateralEvent),
      153: ("drivingEvent", log.DrivingEvent),
      154: ("drivingEventRecorded", log.DrivingEventRecorded),
      155: ("blatV2Shadow", log.BlatV2Shadow),
    }
    for ordinal, (name, struct_module) in expected.items():
      with self.subTest(ordinal=ordinal, name=name):
        field = log.Event.schema.fields[name]
        self.assertEqual(_field_ordinal(field), ordinal)
        self.assertEqual(field.proto.slot.type.which(), "struct")
        self.assertEqual(
          int(field.proto.slot.type.struct.typeId),
          int(struct_module.schema.get_proto().id),
        )

  def test_prerequisite_struct_ids_and_representative_payloads(self) -> None:
    self.assertEqual(
      int(log.LateralEvent.schema.get_proto().id),
      0xD4281D502164ABD1,
    )
    self.assertEqual(
      int(log.DrivingEvent.schema.get_proto().id),
      0xD9F3C9B84F67A2E1,
    )
    self.assertEqual(
      int(log.DrivingEventRecorded.schema.get_proto().id),
      0xCAC5F5A6B137D821,
    )
    representative_fields = (
      (log.LateralEvent, "reason", 19, "text"),
      (
        log.DrivingEvent.Payload,
        "rollingLeadResponse",
        28,
        "struct",
      ),
      (
        log.DrivingEvent.LateralPayload,
        "takeoverConfirmationDurationS",
        171,
        "float32",
      ),
      (
        log.DrivingEvent.LateralPayload.StallReleaseEvidence,
        "phase",
        21,
        "text",
      ),
      (log.DrivingEventRecorded, "markerAccepted", 14, "bool"),
    )
    for struct_module, name, ordinal, field_type in representative_fields:
      with self.subTest(struct=struct_module, name=name):
        field = struct_module.schema.fields[name]
        self.assertEqual(_field_ordinal(field), ordinal)
        self.assertEqual(field.proto.slot.type.which(), field_type)

  def test_shadow_service_matches_historical_contract(self) -> None:
    service = SERVICE_LIST["blatV2Shadow"]
    self.assertTrue(service.should_log)
    self.assertEqual(service.frequency, 100.0)
    self.assertEqual(service.decimation, 10)

  def test_representative_historical_fields_round_trip(self) -> None:
    torque = log.ControlsState.LateralTorqueState.new_message()
    torque.active = True
    torque.referenceTargetTorque = 0.375
    torque.blatV2AdaptiveOutcomeLearningActive = True
    with log.ControlsState.LateralTorqueState.from_bytes(
      torque.to_bytes(),
    ) as decoded_torque:
      self.assertTrue(decoded_torque.active)
      self.assertAlmostEqual(
        decoded_torque.referenceTargetTorque,
        0.375,
      )
      self.assertTrue(
        decoded_torque.blatV2AdaptiveOutcomeLearningActive,
      )

    shadow = log.BlatV2Shadow.new_message()
    shadow.shadowVersion = 7
    shadow.mpcCommandTorque = -0.25
    shadow.liveAdaptiveOutcomeLearningActive = True
    with log.BlatV2Shadow.from_bytes(
      shadow.to_bytes(),
    ) as decoded_shadow:
      self.assertEqual(decoded_shadow.shadowVersion, 7)
      self.assertEqual(decoded_shadow.mpcCommandTorque, -0.25)
      self.assertTrue(
        decoded_shadow.liveAdaptiveOutcomeLearningActive,
      )

  def test_event_shadow_union_round_trip(self) -> None:
    event = log.Event.new_message()
    shadow = event.init("blatV2Shadow")
    shadow.shadowVersion = 19
    shadow.referenceCurvature = 0.0125
    shadow.liveAdaptiveOutcomeCount = 42
    with log.Event.from_bytes(event.to_bytes()) as decoded:
      self.assertEqual(decoded.which(), "blatV2Shadow")
      self.assertEqual(decoded.blatV2Shadow.shadowVersion, 19)
      self.assertEqual(
        decoded.blatV2Shadow.referenceCurvature,
        0.0125,
      )
      self.assertEqual(
        decoded.blatV2Shadow.liveAdaptiveOutcomeCount,
        42,
      )


if __name__ == "__main__":
  unittest.main()

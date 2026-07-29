from pathlib import Path

from openpilot.selfdrive.controls.blatv2_shadowd import PUBLISHED_SERVICES, SHADOW_VERSION, SUBSCRIBED_SERVICES
from openpilot.selfdrive.controls.lib.blatv2.controller import LIVE_CONTROLLER_VERSION


def test_shadow_is_structurally_telemetry_only():
  assert PUBLISHED_SERVICES == ("blatV2Shadow",)
  assert "carControl" not in PUBLISHED_SERVICES
  assert "sendcan" not in PUBLISHED_SERVICES


def test_shadow_subscriptions_are_pinned():
  assert SUBSCRIBED_SERVICES == (
    "modelV2",
    "carState",
    "carControl",
    "carOutput",
    "controlsState",
    "liveParameters",
    "liveTorqueParameters",
    "liveDelay",
    "lateralManeuverPlan",
  )


def test_shadow_process_is_registered_onroad_without_toggle():
  config = Path("openpilot/system/manager/process_config.py").read_text()
  registration = 'PythonProcess("blatv2_shadowd", "openpilot.selfdrive.controls.blatv2_shadowd", only_onroad, restart_if_crash=True)'
  assert registration in config


def test_shadow_runs_below_controlsd_without_competing_on_one_core():
  source = Path(
    "openpilot/selfdrive/controls/blatv2_shadowd.py",
  ).read_text()
  assert "config_realtime_process([0, 1, 2, 3, 4], Priority.CTRL_LOW)" in source
  assert "config_realtime_process(4, Priority.CTRL_LOW)" not in source
  assert "Priority.CTRL_HIGH" not in source


def test_shadow_v13_schema_has_signed_rate_and_exact_v14_fields():
  assert SHADOW_VERSION == 16
  assert LIVE_CONTROLLER_VERSION == 220
  schema = Path("openpilot/cereal/log.capnp").read_text()
  shadow_source = Path(
    "openpilot/selfdrive/controls/blatv2_shadowd.py",
  ).read_text()
  assert "vEgo @9 :Float64;" in schema
  assert "aligningTorque @10 :Float64;" in schema
  assert "alignInputsValid @11 :Bool;" in schema
  assert "disturbanceEstimate @12 :Float64;" in schema
  assert "observerStatus @13 :UInt8;" in schema
  shipped_candidate_fields = (
    "mpcCommandTorque @15 :Float64;",
    "mpcStatus @16 :UInt8;",
    "mpcCandidateCount @17 :UInt16;",
    "mpcOptimalityResidual @18 :Float64;",
    "mpcComputeTimeSeconds @19 :Float64;",
    "fallbackCommandTorque @20 :Float64;",
    "fallbackStatus @21 :UInt8;",
    "fallbackCandidateCount @22 :UInt16;",
    "fallbackOptimalityResidual @23 :Float64;",
    "fallbackComputeTimeSeconds @24 :Float64;",
    "mpcAvailableScheduleCount @26 :UInt16;",
  )
  for field in shipped_candidate_fields:
    assert field in schema
  assert "sharedComputeTimeSeconds @25 :Float64;" in schema
  assert "liveLqiCommandTorque @27 :Float64;" in schema
  assert "liveLqiStatus @28 :UInt8;" in schema
  assert "liveLqiComputeTimeSeconds @29 :Float64;" in schema
  assert "v14CommandTorque @33 :Float64;" in schema
  assert "v14ControllerVersion @35 :Int32;" in schema
  assert "liveLqiControllerVersion @38 :Int32;" in schema
  # The old wire slots decode historical routes but shadow v13 neither solves
  # nor publishes either retired tournament candidate.
  assert "self.core.compute_mpc()" not in shadow_source
  assert "self.core.compute_fallback()" not in shadow_source
  assert "shadow.mpcCommandTorque" not in shadow_source
  assert "shadow.fallbackCommandTorque" not in shadow_source
  assert "liveActionRawCommandTorque @39 :Float64;" in schema
  assert "liveActionDesiredAngleDeg @42 :Float64;" in schema
  assert "liveActionRequiredAccelerationDegS2 @47 :Float64;" in schema
  assert "liveActionSlewConstrained @53 :Bool;" in schema
  assert "liveActionBreakawayActive @54 :Bool;" in schema
  assert "liveActionBreakawayPersistenceFrames @55 :UInt16;" in schema
  assert "liveActionHorizonAssistActive @56 :Bool;" in schema
  assert "liveActionHorizonTorqueDemand @57 :Float64;" in schema
  assert "liveActionHorizonDemandTimeSeconds @58 :Float64;" in schema
  assert "liveActionNoLeadLimited @59 :Bool;" in schema
  assert "liveActionPredictionDelaySeconds @60 :Float64;" in schema
  assert "blatV2RawCommandTorque @80 :Float64;" in schema
  assert "blatV2SlewConstrained @94 :Bool;" in schema
  assert "blatV2BreakawayActive @95 :Bool;" in schema
  assert "blatV2BreakawayPersistenceFrames @96 :UInt16;" in schema
  assert "blatV2HorizonAssistActive @97 :Bool;" in schema
  assert "blatV2HorizonTorqueDemand @98 :Float64;" in schema
  assert "blatV2HorizonDemandTimeSeconds @99 :Float64;" in schema
  assert "blatV2NoLeadLimited @100 :Bool;" in schema
  assert "blatV2PredictionDelaySeconds @101 :Float64;" in schema
  assert "blatV2SignedRackRateDegS @102 :Float64;" in schema
  assert "signedRackRateDegS @61 :Float64;" in schema

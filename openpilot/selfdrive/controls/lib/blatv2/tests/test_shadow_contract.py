from pathlib import Path

from openpilot.selfdrive.controls.blatv2_shadowd import PUBLISHED_SERVICES, SHADOW_VERSION, SUBSCRIBED_SERVICES


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
    "liveDelay",
  )


def test_shadow_process_is_registered_onroad_without_toggle():
  config = Path("openpilot/system/manager/process_config.py").read_text()
  registration = 'PythonProcess("blatv2_shadowd", "openpilot.selfdrive.controls.blatv2_shadowd", only_onroad, restart_if_crash=True)'
  assert registration in config


def test_shadow_runs_below_controlsd_on_the_same_realtime_core():
  source = Path(
    "openpilot/selfdrive/controls/blatv2_shadowd.py",
  ).read_text()
  assert "config_realtime_process(4, Priority.CTRL_LOW)" in source
  assert "config_realtime_process(4, Priority.CTRL_HIGH)" not in source


def test_shadow_v5_schema_has_bounded_mpc_and_split_timing_fields():
  assert SHADOW_VERSION == 5
  schema = Path("openpilot/cereal/log.capnp").read_text()
  assert "vEgo @9 :Float64;" in schema
  assert "aligningTorque @10 :Float64;" in schema
  assert "alignInputsValid @11 :Bool;" in schema
  assert "disturbanceEstimate @12 :Float64;" in schema
  assert "observerStatus @13 :UInt8;" in schema
  assert "mpcCommandTorque @15 :Float64;" in schema
  assert "mpcStatus @16 :UInt8;" in schema
  assert "mpcCandidateCount @17 :UInt16;" in schema
  assert "mpcOptimalityResidual @18 :Float64;" in schema
  assert "mpcComputeTimeSeconds @19 :Float64;" in schema
  assert "fallbackCommandTorque @20 :Float64;" in schema
  assert "fallbackStatus @21 :UInt8;" in schema
  assert "fallbackCandidateCount @22 :UInt16;" in schema
  assert "fallbackOptimalityResidual @23 :Float64;" in schema
  assert "fallbackComputeTimeSeconds @24 :Float64;" in schema
  assert "sharedComputeTimeSeconds @25 :Float64;" in schema
  assert "mpcAvailableScheduleCount @26 :UInt16;" in schema

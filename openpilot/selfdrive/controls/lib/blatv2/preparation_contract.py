"""Shared file/decode contract for offroad and off-device preparation."""

from __future__ import annotations

from pathlib import Path

from opendbc.car.structs import car
from openpilot.common.basedir import BASEDIR


MAXIMUM_CAR_PARAMS_TRAVERSAL_WORDS = (64 * 1024 * 1024) // 8
BLATV2_LIBRARY_ROOT = Path(__file__).resolve().parent
HISTORICAL_BUILD_DESCRIPTORS = BLATV2_LIBRARY_ROOT / "historical_build_descriptors.json"
PROVISIONAL_RACK_DYNAMICS_PATH = BLATV2_LIBRARY_ROOT / "provisional_rack_dynamics.json"
NATIVE_EXTRACTOR_PATH = (
  Path(BASEDIR)
  / "openpilot"
  / "selfdrive"
  / "controls"
  / "blatv2_rlog_extract"
)


def decode_car_params(encoded: bytes) -> car.CarParams:
  """Decode owned CarParams with the same bounded contract on ARM and PC."""
  with car.CarParams.from_bytes(
    encoded,
    traversal_limit_in_words=MAXIMUM_CAR_PARAMS_TRAVERSAL_WORDS,
    nesting_limit=64,
  ) as reader:
    return reader.as_builder()

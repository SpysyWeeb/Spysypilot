"""Strict file-backed authority for BLaTv2 behavioral learning.

The learner has no hidden gate, segmentation, or search constants.  Both
committed JSON artifacts are parsed strictly, re-encoded canonically, and
hash-bound into every transaction.  Editing either file therefore creates a
new learning authority instead of silently reinterpreting old results.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from openpilot.selfdrive.controls.lib.blatv2.behavior_coordinator import (
  BehaviorGateSpec,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import canonical_json
from openpilot.selfdrive.controls.lib.blatv2.behavior_segmentation import (
  SegmentationConfig,
)


_ROOT = Path(__file__).resolve().parent
BEHAVIOR_GATE_SPEC_PATH = _ROOT / "behavior_gate_spec.json"
BEHAVIOR_SEGMENTATION_CONFIG_PATH = _ROOT / "behavior_segmentation_config.json"

_SEGMENTATION_KEYS = frozenset((
  "directHandoffMaxNeutralDurationS",
  "directHandoffMinPeakCurvature1pm",
  "maximumPhaseExtensionS",
  "maximumSampleGapS",
  "minimumPhaseDurationS",
  "minimumPhaseSamples",
  "monotonicProgressEpsilon1pmS",
  "quasiSteadyRateThreshold1pmS",
  "referenceZeroThreshold1pm",
  "releaseOnsetFraction",
  "schemaVersion",
  "turnClassCurvatureThreshold1pm",
  "turnInCrossingFraction",
))


def _read_canonical(path: str | Path) -> tuple[str, dict[str, object]]:
  encoded = Path(path).read_text(encoding="utf-8")
  try:
    payload = json.loads(encoded)
  except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise ValueError("behavior configuration is unreadable") from exc
  if type(payload) is not dict:
    raise ValueError("behavior configuration root must be an object")
  canonical = canonical_json(payload)
  if encoded != canonical + "\n":
    raise ValueError("behavior configuration file is not canonical JSON")
  return canonical, payload


def load_behavior_gate_spec(
  path: str | Path = BEHAVIOR_GATE_SPEC_PATH,
) -> BehaviorGateSpec:
  canonical, _ = _read_canonical(path)
  return BehaviorGateSpec.from_json(canonical)


def _number(payload: dict[str, object], key: str) -> float:
  value = payload[key]
  if type(value) not in (int, float):
    raise ValueError(f"{key} must be numeric")
  result = float(value)
  if not math.isfinite(result):
    raise ValueError(f"{key} must be finite")
  return result


def load_behavior_segmentation_config(
  path: str | Path = BEHAVIOR_SEGMENTATION_CONFIG_PATH,
) -> SegmentationConfig:
  _, payload = _read_canonical(path)
  if frozenset(payload) != _SEGMENTATION_KEYS:
    raise ValueError("behavior segmentation keys do not match the schema")
  if type(payload["schemaVersion"]) is not int:
    raise ValueError("segmentation schema version must be integer")
  if type(payload["minimumPhaseSamples"]) is not int:
    raise ValueError("minimum phase samples must be integer")
  return SegmentationConfig(
    schema_version=payload["schemaVersion"],
    reference_zero_threshold_1pm=_number(
      payload,
      "referenceZeroThreshold1pm",
    ),
    quasi_steady_rate_threshold_1pm_s=_number(
      payload,
      "quasiSteadyRateThreshold1pmS",
    ),
    monotonic_progress_epsilon_1pm_s=_number(
      payload,
      "monotonicProgressEpsilon1pmS",
    ),
    turn_class_curvature_threshold_1pm=_number(
      payload,
      "turnClassCurvatureThreshold1pm",
    ),
    direct_handoff_min_peak_curvature_1pm=_number(
      payload,
      "directHandoffMinPeakCurvature1pm",
    ),
    direct_handoff_max_neutral_duration_s=_number(
      payload,
      "directHandoffMaxNeutralDurationS",
    ),
    minimum_phase_duration_s=_number(payload, "minimumPhaseDurationS"),
    minimum_phase_samples=payload["minimumPhaseSamples"],
    maximum_phase_extension_s=_number(payload, "maximumPhaseExtensionS"),
    maximum_sample_gap_s=_number(payload, "maximumSampleGapS"),
    turn_in_crossing_fraction=_number(payload, "turnInCrossingFraction"),
    release_onset_fraction=_number(payload, "releaseOnsetFraction"),
  )

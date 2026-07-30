"""Versioned controller-response policy for modular BLaTv2.

Physical vehicle variation belongs to ``VehicleProfile``. This artifact owns
the small, speed-independent response objective shared by every vehicle:
closed-loop natural frequency and damping, plus an optional physical
disturbance-observer policy. It contains no maneuver gates or speed schedule.

A provisional policy may be used in shadow/replay but is not field eligible.
Its deterministic hash is part of every candidate identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from openpilot.selfdrive.controls.lib.blatv2.observer import ObserverPolicy
from openpilot.selfdrive.controls.lib.blatv2.plant import TrackingPolicy


CONTROLLER_POLICY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ControllerPolicy:
  revision: int
  provenance: str
  provisional: bool
  natural_frequency_per_s: float
  damping_ratio: float
  observer_time_constant_s: float | None
  observer_max_abs_disturbance_torque: float | None
  schema_version: int = CONTROLLER_POLICY_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if self.schema_version != CONTROLLER_POLICY_SCHEMA_VERSION:
      raise ValueError("controller policy schema is incompatible")
    if self.revision < 0:
      raise ValueError("controller policy revision must be non-negative")
    if not self.provenance.strip():
      raise ValueError("controller policy provenance must not be empty")
    if (
      not math.isfinite(self.natural_frequency_per_s)
      or self.natural_frequency_per_s <= 0.0
      or not math.isfinite(self.damping_ratio)
      or self.damping_ratio <= 0.0
    ):
      raise ValueError("tracking policy values must be finite and positive")
    observer_values = (
      self.observer_time_constant_s,
      self.observer_max_abs_disturbance_torque,
    )
    if (observer_values[0] is None) != (observer_values[1] is None):
      raise ValueError("observer policy is either complete or disabled")
    if observer_values[0] is not None and (
      not math.isfinite(observer_values[0])
      or observer_values[0] <= 0.0
      or not math.isfinite(observer_values[1])
      or observer_values[1] <= 0.0
    ):
      raise ValueError("observer policy values must be finite and positive")

  @property
  def tracking_policy(self) -> TrackingPolicy:
    return TrackingPolicy(
      natural_frequency_per_s=self.natural_frequency_per_s,
      damping_ratio=self.damping_ratio,
    )

  @property
  def observer_policy(self) -> ObserverPolicy | None:
    if self.observer_time_constant_s is None:
      return None
    return ObserverPolicy(
      time_constant_s=self.observer_time_constant_s,
      max_abs_disturbance_torque=(
        self.observer_max_abs_disturbance_torque
      ),
    )

  def to_dict(self) -> dict[str, Any]:
    observer = (
      None
      if self.observer_time_constant_s is None
      else {
        "max_abs_disturbance_torque": (
          self.observer_max_abs_disturbance_torque
        ),
        "time_constant_s": self.observer_time_constant_s,
      }
    )
    return {
      "damping_ratio": self.damping_ratio,
      "natural_frequency_per_s": self.natural_frequency_per_s,
      "observer": observer,
      "provenance": self.provenance,
      "provisional": self.provisional,
      "revision": self.revision,
      "schema_version": self.schema_version,
    }

  def to_json(self) -> str:
    return json.dumps(
      self.to_dict(),
      sort_keys=True,
      separators=(",", ":"),
    )

  @property
  def sha256(self) -> str:
    return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

  @classmethod
  def from_json_file(cls, path: str | Path) -> ControllerPolicy:
    try:
      payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
      raise ValueError("controller policy file is unreadable") from exc
    expected_keys = {
      "damping_ratio",
      "natural_frequency_per_s",
      "observer",
      "provenance",
      "provisional",
      "revision",
      "schema_version",
    }
    if type(payload) is not dict or set(payload) != expected_keys:
      raise ValueError("controller policy keys do not match the schema")
    if type(payload["schema_version"]) is not int:
      raise ValueError("controller policy schema version must be an integer")
    if type(payload["revision"]) is not int:
      raise ValueError("controller policy revision must be an integer")
    if type(payload["provisional"]) is not bool:
      raise ValueError("controller provisional flag must be boolean")
    if type(payload["provenance"]) is not str:
      raise ValueError("controller policy provenance must be text")
    observer = payload["observer"]
    observer_time_constant_s = None
    observer_max_abs_disturbance_torque = None
    if observer is not None:
      if type(observer) is not dict or set(observer) != {
        "max_abs_disturbance_torque",
        "time_constant_s",
      }:
        raise ValueError("controller observer policy keys are invalid")
      observer_time_constant_s = float(observer["time_constant_s"])
      observer_max_abs_disturbance_torque = float(
        observer["max_abs_disturbance_torque"],
      )
    return cls(
      revision=payload["revision"],
      provenance=payload["provenance"],
      provisional=payload["provisional"],
      natural_frequency_per_s=float(
        payload["natural_frequency_per_s"],
      ),
      damping_ratio=float(payload["damping_ratio"]),
      observer_time_constant_s=observer_time_constant_s,
      observer_max_abs_disturbance_torque=(
        observer_max_abs_disturbance_torque
      ),
      schema_version=payload["schema_version"],
    )

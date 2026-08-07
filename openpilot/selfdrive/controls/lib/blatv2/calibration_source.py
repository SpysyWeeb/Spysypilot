"""Immutable source coordinates for authenticated calibration replay."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CalibrationIngestionCoordinate:
  """One controlsState witness in an authenticated route artifact."""

  route_content_sha256: str
  segment_index: int
  log_mono_time_ns: int
  recorded_ordinal: int

  def __post_init__(self) -> None:
    if (
      len(self.route_content_sha256) != 64
      or any(character not in "0123456789abcdef" for character in self.route_content_sha256)
    ):
      raise ValueError("calibration source route content must be lowercase SHA-256")
    if type(self.segment_index) is not int or self.segment_index < 0:
      raise ValueError("calibration source segment index must be nonnegative")
    if type(self.log_mono_time_ns) is not int or self.log_mono_time_ns <= 0:
      raise ValueError("calibration source logMonoTime must be positive")
    if type(self.recorded_ordinal) is not int or self.recorded_ordinal < 0:
      raise ValueError("calibration source ordinal must be nonnegative")

  def payload(self) -> dict[str, object]:
    return {
      "log_mono_time_ns": self.log_mono_time_ns,
      "recorded_ordinal": self.recorded_ordinal,
      "route_content_sha256": self.route_content_sha256,
      "segment_index": self.segment_index,
    }

  @property
  def ordering_key(self) -> tuple[int, int, int]:
    return (self.log_mono_time_ns, self.segment_index, self.recorded_ordinal)

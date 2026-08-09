"""Causal alignment of recorded applied torque to measured rack response.

The physical inverse model is evaluated at rack-response time ``t``. Only its
input is shifted: the aligned input is the newest exact, driver-free
``carOutput`` command whose effective time is no later than
``t - transport_delay``. Torque is zero-order-held and never interpolated.
Response rate, acceleration, lateral acceleration, speed, mapping, and node
weights all remain measured at ``t``.

``card.py`` publishes ``last_actuators_output`` before applying the next
command. A carOutput payload therefore became effective at the preceding
carOutput publication time. Callers provide both clocks explicitly so this
one-cycle convention cannot be hidden inside a tuned delay.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

from openpilot.selfdrive.controls.lib.blatv2.learner import ActuatorBoundary


@dataclass(frozen=True, slots=True)
class RecordedAppliedTorque:
  report_mono_ns: int
  effective_mono_ns: int
  applied_torque: float
  actuator_constrained: bool
  boundary: ActuatorBoundary
  magnitude_boundary_dwell_s: float


class CausalTorqueResponseAligner:
  """Bounded exact-command history for one uninterrupted validity epoch."""

  __slots__ = (
    "_history",
    "_last_report_mono_ns",
    "_maximum_delay_ns",
    "_maximum_delay_s",
    "_maximum_gap_ns",
  )

  def __init__(
    self,
    *,
    maximum_transport_delay_s: float,
    maximum_gap_ns: int,
  ) -> None:
    maximum_delay = float(maximum_transport_delay_s)
    try:
      maximum_gap = int(maximum_gap_ns)
    except (TypeError, ValueError, OverflowError) as error:
      raise ValueError("response aligner gap must be an exact positive integer") from error
    if (
      not math.isfinite(maximum_delay)
      or maximum_delay < 0.0
      or isinstance(maximum_gap_ns, bool)
      or maximum_gap != maximum_gap_ns
      or maximum_gap <= 0
    ):
      raise ValueError("response aligner bounds must be finite and physical")
    self._maximum_delay_s = maximum_delay
    self._maximum_delay_ns = round(maximum_delay * 1e9)
    self._maximum_gap_ns = maximum_gap
    self._history: deque[RecordedAppliedTorque] = deque()
    self._last_report_mono_ns: int | None = None

  def reset(self) -> None:
    self._history.clear()
    self._last_report_mono_ns = None

  def record(
    self,
    *,
    report_mono_ns: int,
    effective_mono_ns: int,
    applied_torque: float,
    actuator_constrained: bool,
    boundary: ActuatorBoundary,
    magnitude_boundary_dwell_s: float,
    valid: bool,
  ) -> bool:
    try:
      report = int(report_mono_ns)
      effective = int(effective_mono_ns)
      applied = float(applied_torque)
      dwell = float(magnitude_boundary_dwell_s)
    except (TypeError, ValueError, OverflowError):
      self.reset()
      return False
    if (
      not valid
      or isinstance(report_mono_ns, bool)
      or isinstance(effective_mono_ns, bool)
      or report != report_mono_ns
      or effective != effective_mono_ns
      or not all(math.isfinite(value) for value in (applied, dwell))
      or report <= 0
      or effective <= 0
      or effective >= report
      or abs(applied) > 1.0
      or dwell < 0.0
      or not isinstance(boundary, ActuatorBoundary)
    ):
      self.reset()
      return False

    if self._last_report_mono_ns is not None:
      report_gap = report - self._last_report_mono_ns
      if report_gap == 0:
        previous = self._history[-1] if self._history else None
        duplicate = previous is not None and previous == RecordedAppliedTorque(
          report_mono_ns=report,
          effective_mono_ns=effective,
          applied_torque=applied,
          actuator_constrained=bool(actuator_constrained),
          boundary=boundary,
          magnitude_boundary_dwell_s=dwell,
        )
        if not duplicate:
          self.reset()
        return duplicate
      if report_gap < 0 or report_gap > self._maximum_gap_ns:
        self.reset()
        return False
      if self._history and effective <= self._history[-1].effective_mono_ns:
        self.reset()
        return False

    observation = RecordedAppliedTorque(
      report_mono_ns=report,
      effective_mono_ns=effective,
      applied_torque=applied,
      actuator_constrained=bool(actuator_constrained),
      boundary=boundary,
      magnitude_boundary_dwell_s=dwell,
    )
    self._history.append(observation)
    self._last_report_mono_ns = report

    retention_floor = (
      report - self._maximum_delay_ns - 2 * self._maximum_gap_ns
    )
    while (
      len(self._history) > 1
      and self._history[1].effective_mono_ns < retention_floor
    ):
      self._history.popleft()
    return True

  def aligned(
    self,
    *,
    response_mono_ns: int,
    transport_delay_s: float,
  ) -> RecordedAppliedTorque | None:
    try:
      response = int(response_mono_ns)
      delay = float(transport_delay_s)
    except (TypeError, ValueError, OverflowError):
      return None
    if (
      isinstance(response_mono_ns, bool)
      or response != response_mono_ns
      or response <= 0
      or not math.isfinite(delay)
      or not 0.0 <= delay <= self._maximum_delay_s
      or self._last_report_mono_ns is None
      or abs(response - self._last_report_mono_ns) > self._maximum_gap_ns
    ):
      return None

    target = response - round(delay * 1e9)
    selected: RecordedAppliedTorque | None = None
    for observation in self._history:
      if observation.effective_mono_ns <= target:
        selected = observation
      else:
        break
    return selected

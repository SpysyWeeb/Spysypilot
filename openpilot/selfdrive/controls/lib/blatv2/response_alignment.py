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
  report_time_s: float
  effective_time_s: float
  applied_torque: float
  actuator_constrained: bool
  boundary: ActuatorBoundary
  magnitude_boundary_dwell_s: float


class CausalTorqueResponseAligner:
  """Bounded exact-command history for one uninterrupted validity epoch."""

  __slots__ = (
    "_history",
    "_last_report_time_s",
    "_maximum_delay_s",
    "_maximum_gap_s",
  )

  def __init__(
    self,
    *,
    maximum_transport_delay_s: float,
    maximum_gap_s: float,
  ) -> None:
    maximum_delay = float(maximum_transport_delay_s)
    maximum_gap = float(maximum_gap_s)
    if (
      not math.isfinite(maximum_delay)
      or maximum_delay < 0.0
      or not math.isfinite(maximum_gap)
      or maximum_gap <= 0.0
    ):
      raise ValueError("response aligner bounds must be finite and physical")
    self._maximum_delay_s = maximum_delay
    self._maximum_gap_s = maximum_gap
    self._history: deque[RecordedAppliedTorque] = deque()
    self._last_report_time_s: float | None = None

  def reset(self) -> None:
    self._history.clear()
    self._last_report_time_s = None

  def record(
    self,
    *,
    report_time_s: float,
    effective_time_s: float,
    applied_torque: float,
    actuator_constrained: bool,
    boundary: ActuatorBoundary,
    magnitude_boundary_dwell_s: float,
    valid: bool,
  ) -> bool:
    report = float(report_time_s)
    effective = float(effective_time_s)
    applied = float(applied_torque)
    dwell = float(magnitude_boundary_dwell_s)
    if (
      not valid
      or not all(math.isfinite(value) for value in (
        report,
        effective,
        applied,
        dwell,
      ))
      or report <= 0.0
      or effective <= 0.0
      or effective >= report
      or abs(applied) > 1.0
      or dwell < 0.0
      or not isinstance(boundary, ActuatorBoundary)
    ):
      self.reset()
      return False

    if self._last_report_time_s is not None:
      report_gap = report - self._last_report_time_s
      if report_gap == 0.0:
        previous = self._history[-1] if self._history else None
        duplicate = previous is not None and previous == RecordedAppliedTorque(
          report_time_s=report,
          effective_time_s=effective,
          applied_torque=applied,
          actuator_constrained=bool(actuator_constrained),
          boundary=boundary,
          magnitude_boundary_dwell_s=dwell,
        )
        if not duplicate:
          self.reset()
        return duplicate
      if report_gap < 0.0 or report_gap > self._maximum_gap_s:
        self.reset()
        return False
      if self._history and effective <= self._history[-1].effective_time_s:
        self.reset()
        return False

    observation = RecordedAppliedTorque(
      report_time_s=report,
      effective_time_s=effective,
      applied_torque=applied,
      actuator_constrained=bool(actuator_constrained),
      boundary=boundary,
      magnitude_boundary_dwell_s=dwell,
    )
    self._history.append(observation)
    self._last_report_time_s = report

    retention_floor = (
      report - self._maximum_delay_s - 2.0 * self._maximum_gap_s
    )
    while (
      len(self._history) > 1
      and self._history[1].effective_time_s < retention_floor
    ):
      self._history.popleft()
    return True

  def aligned(
    self,
    *,
    response_time_s: float,
    transport_delay_s: float,
  ) -> RecordedAppliedTorque | None:
    response = float(response_time_s)
    delay = float(transport_delay_s)
    if (
      not math.isfinite(response)
      or response <= 0.0
      or not math.isfinite(delay)
      or not 0.0 <= delay <= self._maximum_delay_s
      or self._last_report_time_s is None
      or abs(response - self._last_report_time_s) > self._maximum_gap_s
    ):
      return None

    target = response - delay
    selected: RecordedAppliedTorque | None = None
    for observation in self._history:
      if observation.effective_time_s <= target + 1e-12:
        selected = observation
      else:
        break
    return selected

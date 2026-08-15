"""BLoTv2 longitudinal necessity policy.

The supervisor never commands acceleration. It changes only two runtime inputs
the stock MPC already owns: acceleration-change/jerk cost and following time.
That keeps trajectory generation inside one solver while making its response
proportional to measured need.
"""
from dataclasses import dataclass
import math
from typing import Any

from opendbc.car.interfaces import ACCEL_MAX
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_lead import (
  LeadObservation,
  closing_speed,
  time_to_collision,
  total_decel_requirement,
)
from openpilot.selfdrive.modeld.constants import ModelConstants

# Shared necessity frame.
D_MIN = 4.0
MIN_GAP_BUDGET = 1.0

# BLoTv2 requests BLoT v1's proven launch authority, but the deployed opendbc
# envelope remains the source of truth. This keeps the stock-based feature
# branch at 2.0 m/s² while allowing combo's existing 4.0 m/s² safety lineage.
BLOTV2_ACCEL_REQUEST_MAX = 4.0
BLOTV2_ACCEL_MAX = min(ACCEL_MAX, BLOTV2_ACCEL_REQUEST_MAX)

# Recovery and model-forecast response.
MIN_BOOST_BRAKE = 0.8
EXCESS_ON = 0.4
EXCESS_OFF = 0.15
EXCESS_DEBOUNCE = 0.4
MODEL_HARD_THRESHOLD = -0.5
MODEL_ARM_DEBOUNCE = 0.3
MODEL_LEAD_PROB_MIN = 0.5

# Launch response.
LAUNCH_VREL_ON = 0.5
LAUNCH_VREL_OFF = 0.2
LAUNCH_ALEAD_ON = 0.5
LAUNCH_SHORTFALL_ON = 0.6
LAUNCH_SHORTFALL_OFF = 0.2
LAUNCH_DEBOUNCE = 0.4
RATCHET_LEAD_BRAKE = 0.2

# MPC policy bounds and continuity.
JERK_SCALE_MIN = 0.3
JERK_SCALE_RATE = 1.5
ONSET_LEAD_DECEL = 0.4
ONSET_PAD_MAX = 0.45
STOPPED_LEAD_PAD_MAX = 0.75
ONSET_FULL_DECEL = 1.5
ONSET_MAX_A_REQ = 1.5
EMERGENCY_SHORTFALL_MIN = 0.15
ONSET_RATE_UP = 0.8
ONSET_RATE_DOWN = 0.5
MIN_TTC = 3.5
MIN_SPEED = 1.0

# Standstill lead-departure pre-release.
LEAD_DEPARTURE_SPEED = 0.3
LEAD_MOVING_SPEED = 0.25
LEAD_DEPARTURE_CONFIRM = 0.2
LEAD_DEPARTURE_CANCEL = 0.2


@dataclass(frozen=True, slots=True)
class LongitudinalPolicy:
  jerk_scale: float
  t_follow: float
  required_decel: float
  emergency: bool
  recovery_active: bool
  model_active: bool
  launch_active: bool


class DebouncedTrigger:
  def __init__(self, debounce: float, dt: float):
    self.debounce = debounce
    self.dt = dt
    self._seconds = 0.0

  def reset(self) -> None:
    self._seconds = 0.0

  def step(self, arm: bool, disarm: bool) -> bool:
    if arm:
      self._seconds += self.dt
    elif disarm:
      self.reset()
    return self._seconds + 1e-9 >= self.debounce


def model_predicted_acceleration(model_lead: Any) -> float | None:
  """Return the model lead's first-horizon acceleration when trustworthy."""
  try:
    if model_lead is None or float(model_lead.prob) < MODEL_LEAD_PROB_MIN or len(model_lead.v) < 2:
      return None
    v0 = float(model_lead.v[0])
    v1 = float(model_lead.v[1])
    horizon = float(ModelConstants.LEAD_T_IDXS[1] - ModelConstants.LEAD_T_IDXS[0])
  except (AttributeError, IndexError, TypeError, ValueError):
    return None

  if horizon <= 0.0 or not all(math.isfinite(value) for value in (v0, v1, horizon)):
    return None
  return (v1 - v0) / horizon


def model_predicted_speed(model_lead: Any, lead: LeadObservation) -> float | None:
  """Anchor the model's first future speed delta to filtered radar speed."""
  if not lead.present:
    return None
  try:
    if model_lead is None or float(model_lead.prob) < MODEL_LEAD_PROB_MIN or len(model_lead.v) < 2:
      return None
    v0 = float(model_lead.v[0])
    v1 = float(model_lead.v[1])
  except (AttributeError, IndexError, TypeError, ValueError):
    return None
  if not all(math.isfinite(value) for value in (v0, v1)):
    return None
  return max(lead.speed + v1 - v0, 0.0)


class LeadDeparturePreRelease:
  """Release only the MPC stop intent when a lead departure is corroborated."""

  def __init__(self, dt: float = DT_MDL):
    self.dt = dt
    self.reset()

  def reset(self) -> None:
    self._prediction_s = 0.0
    self._cancel_s = 0.0
    self._released = False

  def update(self, active: bool, standstill: bool, lead: LeadObservation,
             predicted_speed: float | None) -> bool:
    if not (active and standstill and lead.present):
      self.reset()
      return False

    if lead.speed > LEAD_DEPARTURE_SPEED:
      self._released = True
      self._prediction_s = 0.0
      self._cancel_s = 0.0
      return True

    departure_valid = lead.speed > LEAD_MOVING_SPEED or (
      predicted_speed is not None
      and math.isfinite(predicted_speed)
      and predicted_speed > LEAD_DEPARTURE_SPEED
    )
    if departure_valid:
      self._prediction_s += self.dt
      self._cancel_s = 0.0
      if self._prediction_s + 1e-9 >= LEAD_DEPARTURE_CONFIRM:
        self._released = True
    else:
      self._prediction_s = 0.0
      if self._released:
        self._cancel_s += self.dt
        if self._cancel_s + 1e-9 >= LEAD_DEPARTURE_CANCEL:
          self.reset()

    return self._released


class BLoTv2Supervisor:
  def __init__(self, dt: float = DT_MDL):
    self.dt = dt
    self.jerk_scale = 1.0
    self.t_follow_pad = 0.0
    self._recovery = DebouncedTrigger(EXCESS_DEBOUNCE, dt)
    self._model = DebouncedTrigger(MODEL_ARM_DEBOUNCE, dt)
    self._launch = DebouncedTrigger(LAUNCH_DEBOUNCE, dt)
    self._triggers = (self._recovery, self._model, self._launch)

  def reset(self) -> None:
    self.jerk_scale = 1.0
    self.t_follow_pad = 0.0
    for trigger in self._triggers:
      trigger.reset()

  def _slew(self, current: float, target: float, rate: float) -> float:
    step = rate * self.dt
    return min(max(target, current - step), current + step)

  def update(self, lead: LeadObservation, v_ego: float, a_mpc: float,
             t_follow_base: float, predicted_lead_accel: float | None = None) -> LongitudinalPolicy:
    scale_target = 1.0
    pad_target = 0.0
    required_decel = 0.0
    emergency = False
    recovery_active = False
    model_active = False
    launch_active = False

    if lead.present and v_ego > MIN_SPEED:
      required_decel = total_decel_requirement(
        v_ego, lead, D_MIN, MIN_GAP_BUDGET,
      )
      closing = closing_speed(v_ego, lead)
      braking_shortfall = required_decel + a_mpc if math.isfinite(a_mpc) else math.inf
      emergency = (
        time_to_collision(v_ego, lead) < MIN_TTC
        and required_decel >= ONSET_MAX_A_REQ
        and braking_shortfall > EMERGENCY_SHORTFALL_MIN
      )

      if not emergency:
        excess = -a_mpc - required_decel
        recovery_active = self._recovery.step(
          arm=excess > EXCESS_ON and -a_mpc > MIN_BOOST_BRAKE,
          disarm=excess < EXCESS_OFF or -a_mpc <= MIN_BOOST_BRAKE,
        )

        predicted_hard = (
          predicted_lead_accel is not None
          and math.isfinite(predicted_lead_accel)
          and predicted_lead_accel < MODEL_HARD_THRESHOLD
        )
        model_active = self._model.step(
          arm=predicted_hard,
          disarm=not predicted_hard,
        )

        receding = lead.speed - v_ego
        shortfall = lead.acceleration - a_mpc
        launch_active = self._launch.step(
          arm=(
            receding > LAUNCH_VREL_ON
            and lead.acceleration > LAUNCH_ALEAD_ON
            and required_decel < 0.1
            and shortfall > LAUNCH_SHORTFALL_ON
          ),
          disarm=(
            shortfall < LAUNCH_SHORTFALL_OFF
            or lead.acceleration < LAUNCH_ALEAD_ON
            or receding < LAUNCH_VREL_OFF
          ),
        )

        if recovery_active or model_active or launch_active:
          scale_target = JERK_SCALE_MIN

        # Do not stiffen a previously responsive solution in the middle of a
        # lead-braking/ego-closing reversal.
        if (
          scale_target > self.jerk_scale
          and lead.acceleration < -RATCHET_LEAD_BRAKE
          and closing > 0.0
        ):
          scale_target = self.jerk_scale

        onset_lead_accel = lead.acceleration
        if model_active and predicted_lead_accel is not None:
          onset_lead_accel = min(onset_lead_accel, predicted_lead_accel)

        recovering = v_ego <= lead.speed + 0.2 or lead.acceleration > 0.2
        if (
          onset_lead_accel < -ONSET_LEAD_DECEL
          and required_decel < ONSET_MAX_A_REQ
          and not recovering
        ):
          pad_target = ONSET_PAD_MAX * min(
            -onset_lead_accel / ONSET_FULL_DECEL,
            1.0,
          )
        if (
          lead.speed < 2.0
          and 0.3 < required_decel < ONSET_MAX_A_REQ
          and not recovering
        ):
          pad_target = max(
            pad_target,
            STOPPED_LEAD_PAD_MAX * min(required_decel / 1.2, 1.0),
          )
      else:
        for trigger in self._triggers:
          trigger.reset()
    else:
      for trigger in self._triggers:
        trigger.reset()

    self.jerk_scale = self._slew(
      self.jerk_scale,
      scale_target,
      JERK_SCALE_RATE,
    )
    self.t_follow_pad = self._slew(
      self.t_follow_pad,
      pad_target,
      ONSET_RATE_UP if pad_target > self.t_follow_pad else ONSET_RATE_DOWN,
    )

    return LongitudinalPolicy(
      jerk_scale=self.jerk_scale,
      t_follow=t_follow_base + self.t_follow_pad,
      required_decel=required_decel,
      emergency=emergency,
      recovery_active=recovery_active,
      model_active=model_active,
      launch_active=launch_active,
    )

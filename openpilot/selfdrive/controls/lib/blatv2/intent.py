"""Canonical, stateless adaptation of model intent for live control and replay.

The adapter keeps three clocks distinct:

* the control witness is when the controller consumed the model message;
* the model publication timestamp determines message freshness;
* ``timestampEof`` is the origin of every query on the native model plan.

The scalar action time is consumed exactly as published by modeld. Physical
transport delay shifts only the time when a command can affect the rack. No
controller-side timing reconstruction is performed.

All populated plan data is copied into caller-owned, fixed-capacity buffers.
The function retains no state, and every call clears the full output capacity
before returning, so malformed future data cannot expose an earlier plan.
"""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass
from enum import IntEnum
import math

from openpilot.selfdrive.controls.lib.blatv2.contracts import (
  CanonicalFrame,
  FrameTiming,
  FrameValidity,
)


INTENT_CAPACITY = 33
MODEL_PUBLICATION_HZ = 20
MODEL_PUBLICATION_PERIOD_NS = 1_000_000_000 // MODEL_PUBLICATION_HZ
# One missed publication is permitted. A message becomes stale only after it
# has aged through two complete periods of the model's published cadence.
MAX_MODEL_PUBLICATION_AGE_NS = 2 * MODEL_PUBLICATION_PERIOD_NS


class IntentStatusCode(IntEnum):
  """Primary result classification, stable for live and replay diagnostics."""

  OK = 0
  SCALAR_ONLY_MALFORMED_FUTURE = 1
  SCALAR_ONLY_UNSUPPORTED_QUERY = 2
  INVALID_TIMING = 3
  MESSAGE_NOT_ALIVE = 4
  MESSAGE_INVALID = 5
  MESSAGE_STALE = 6
  INVALID_SCALAR = 7
  INVALID_VEHICLE_STATE = 8
  INVALID_MODEL_FRAME_ID = 9


@dataclass(frozen=True, slots=True)
class IntentBuildStatus:
  """Immutable status and derived timing facts for one adaptation."""

  code: IntentStatusCode
  count: int
  model_frame_id: int
  scalar_valid: bool
  future_valid: bool
  scalar_only: bool
  publication_age_s: float
  state_age_s: float
  plan_time_now_s: float
  scalar_deadline_mono_s: float
  physical_effect_mono_s: float
  physical_effect_plan_s: float
  total_prediction_horizon_s: float
  action_query_supported: bool
  current_query_supported: bool
  effect_query_supported: bool

  @property
  def usable(self) -> bool:
    return self.code in (
      IntentStatusCode.OK,
      IntentStatusCode.SCALAR_ONLY_MALFORMED_FUTURE,
      IntentStatusCode.SCALAR_ONLY_UNSUPPORTED_QUERY,
    )

  @property
  def stale(self) -> bool:
    return self.code == IntentStatusCode.MESSAGE_STALE


@dataclass(frozen=True, slots=True)
class IntentAdaptation:
  """Canonical frame plus status for caller-owned intent buffers."""

  frame: CanonicalFrame | None
  status: IntentBuildStatus


def _clear_outputs(
  output_plan_times_s: MutableSequence[float],
  output_orientation_rates_z: MutableSequence[float],
  output_velocities_x: MutableSequence[float],
  output_plan_curvatures: MutableSequence[float],
) -> None:
  outputs = (
    output_plan_times_s,
    output_orientation_rates_z,
    output_velocities_x,
    output_plan_curvatures,
  )
  if any(len(output) != INTENT_CAPACITY for output in outputs):
    raise ValueError(
      f"intent output buffers must each have exactly {INTENT_CAPACITY} samples",
    )
  for index in range(INTENT_CAPACITY):
    output_plan_times_s[index] = 0.0
    output_orientation_rates_z[index] = 0.0
    output_velocities_x[index] = 0.0
    output_plan_curvatures[index] = 0.0


def _as_uint(value: object, maximum: int) -> int | None:
  if isinstance(value, bool):
    return None
  try:
    converted = int(value)
  except (TypeError, ValueError, OverflowError):
    return None
  if isinstance(value, float) and not value.is_integer():
    return None
  if converted < 0 or converted > maximum or converted != value:
    return None
  return converted


def _status(
  code: IntentStatusCode,
  *,
  count: int,
  model_frame_id: int,
  scalar_valid: bool,
  future_valid: bool,
  scalar_only: bool,
  publication_age_s: float,
  state_age_s: float,
  plan_time_now_s: float,
  scalar_deadline_mono_s: float,
  physical_effect_mono_s: float,
  physical_effect_plan_s: float,
  total_prediction_horizon_s: float,
  action_query_supported: bool = False,
  current_query_supported: bool = False,
  effect_query_supported: bool = False,
) -> IntentBuildStatus:
  return IntentBuildStatus(
    code=code,
    count=count,
    model_frame_id=model_frame_id,
    scalar_valid=scalar_valid,
    future_valid=future_valid,
    scalar_only=scalar_only,
    publication_age_s=publication_age_s,
    state_age_s=state_age_s,
    plan_time_now_s=plan_time_now_s,
    scalar_deadline_mono_s=scalar_deadline_mono_s,
    physical_effect_mono_s=physical_effect_mono_s,
    physical_effect_plan_s=physical_effect_plan_s,
    total_prediction_horizon_s=total_prediction_horizon_s,
    action_query_supported=action_query_supported,
    current_query_supported=current_query_supported,
    effect_query_supported=effect_query_supported,
  )


def _invalid_timing_result(
  model_frame_id: int,
  code: IntentStatusCode = IntentStatusCode.INVALID_TIMING,
) -> IntentAdaptation:
  invalid = math.nan
  return IntentAdaptation(
    frame=None,
    status=_status(
      code,
      count=0,
      model_frame_id=model_frame_id,
      scalar_valid=False,
      future_valid=False,
      scalar_only=False,
      publication_age_s=invalid,
      state_age_s=invalid,
      plan_time_now_s=invalid,
      scalar_deadline_mono_s=invalid,
      physical_effect_mono_s=invalid,
      physical_effect_plan_s=invalid,
      total_prediction_horizon_s=invalid,
    ),
  )


def _write_scalar_only(
  scalar_curvature: float,
  scalar_action_plan_s: float,
  output_plan_times_s: MutableSequence[float],
  output_plan_curvatures: MutableSequence[float],
) -> None:
  output_plan_times_s[0] = scalar_action_plan_s
  output_plan_curvatures[0] = scalar_curvature


def _future_arrays_valid(
  native_plan_times_s: Sequence[float],
  native_orientation_rates_z: Sequence[float],
  native_velocities_x: Sequence[float],
) -> bool:
  count = len(native_plan_times_s)
  if (
    count < 2
    or len(native_orientation_rates_z) != count
    or len(native_velocities_x) != count
  ):
    return False
  previous_time = -math.inf
  for index in range(count):
    try:
      plan_time = float(native_plan_times_s[index])
      orientation_rate = float(native_orientation_rates_z[index])
      planned_speed = float(native_velocities_x[index])
    except (TypeError, ValueError, OverflowError):
      return False
    if (
      not math.isfinite(plan_time)
      or plan_time < 0.0
      or plan_time <= previous_time
      or not math.isfinite(orientation_rate)
      or not math.isfinite(planned_speed)
      or planned_speed <= 0.0
    ):
      return False
    previous_time = plan_time
  return True


def adapt_model_intent_into(
  *,
  state_sample_mono_ns: int,
  control_witness_mono_ns: int,
  model_publication_mono_ns: int,
  plan_origin_mono_ns: int,
  model_frame_id: int,
  message_valid: bool,
  message_alive: bool,
  scalar_desired_curvature: float,
  published_desired_curvature_time_s: float,
  native_plan_times_s: Sequence[float],
  native_orientation_rates_z: Sequence[float],
  native_velocities_x: Sequence[float],
  current_v_ego_m_s: float,
  physical_transport_delay_s: float,
  output_plan_times_s: MutableSequence[float],
  output_orientation_rates_z: MutableSequence[float],
  output_velocities_x: MutableSequence[float],
  output_plan_curvatures: MutableSequence[float],
) -> IntentAdaptation:
  """Adapt a model message into canonical caller-owned intent buffers.

  Runtime data errors return a legible status and never retain an earlier
  future plan. Insufficient output capacity or a native plan exceeding the
  schema's fixed capacity is an API/configuration error and raises
  ``ValueError``.
  """
  _clear_outputs(
    output_plan_times_s,
    output_orientation_rates_z,
    output_velocities_x,
    output_plan_curvatures,
  )

  native_count = len(native_plan_times_s)
  if native_count > INTENT_CAPACITY:
    raise ValueError(
      f"native intent exceeds fixed capacity {INTENT_CAPACITY}",
    )

  frame_id = _as_uint(model_frame_id, (1 << 32) - 1)
  if frame_id is None:
    return _invalid_timing_result(
      0,
      IntentStatusCode.INVALID_MODEL_FRAME_ID,
    )

  state_sample_ns = _as_uint(state_sample_mono_ns, (1 << 64) - 1)
  control_ns = _as_uint(control_witness_mono_ns, (1 << 64) - 1)
  publication_ns = _as_uint(model_publication_mono_ns, (1 << 64) - 1)
  origin_ns = _as_uint(plan_origin_mono_ns, (1 << 64) - 1)
  action_time = float(published_desired_curvature_time_s)
  transport_delay = float(physical_transport_delay_s)
  if (
    state_sample_ns is None
    or control_ns is None
    or publication_ns is None
    or origin_ns is None
    or state_sample_ns > control_ns
    or publication_ns > control_ns
    or origin_ns > publication_ns
    or not math.isfinite(action_time)
    or action_time < 0.0
    or not math.isfinite(transport_delay)
    or transport_delay < 0.0
  ):
    return _invalid_timing_result(frame_id)

  publication_age_ns = control_ns - publication_ns
  publication_age_s = publication_age_ns * 1e-9
  state_age_s = (control_ns - state_sample_ns) * 1e-9
  plan_time_now_s = (control_ns - origin_ns) * 1e-9
  origin_s = origin_ns * 1e-9
  control_s = control_ns * 1e-9
  scalar_deadline_mono_s = origin_s + action_time
  physical_effect_mono_s = control_s + transport_delay
  physical_effect_plan_s = plan_time_now_s + transport_delay
  total_prediction_horizon_s = state_age_s + transport_delay

  scalar = float(scalar_desired_curvature)
  v_ego = float(current_v_ego_m_s)
  scalar_valid = math.isfinite(scalar)
  vehicle_state_valid = math.isfinite(v_ego) and v_ego >= 0.0
  alive = bool(message_alive)
  message_is_valid = bool(message_valid)
  fresh = publication_age_ns <= MAX_MODEL_PUBLICATION_AGE_NS
  model_valid = alive and message_is_valid and fresh and scalar_valid

  timing = FrameTiming(
    state_sample_mono_ns=state_sample_ns,
    control_witness_mono_ns=control_ns,
    plan_origin_mono_ns=origin_ns,
    plan_publication_mono_ns=publication_ns,
    scalar_action_plan_s=action_time,
    transport_delay_s=transport_delay,
  )

  base_status_kwargs = {
    "model_frame_id": frame_id,
    "scalar_valid": scalar_valid,
    "publication_age_s": publication_age_s,
    "state_age_s": state_age_s,
    "plan_time_now_s": plan_time_now_s,
    "scalar_deadline_mono_s": scalar_deadline_mono_s,
    "physical_effect_mono_s": physical_effect_mono_s,
    "physical_effect_plan_s": physical_effect_plan_s,
    "total_prediction_horizon_s": total_prediction_horizon_s,
  }

  if not vehicle_state_valid:
    frame = CanonicalFrame(
      timing=timing,
      validity=FrameValidity(
        model_valid=model_valid,
        plan_valid=False,
        vehicle_state_valid=False,
        calibration_valid=True,
      ),
    )
    return IntentAdaptation(
      frame=frame,
      status=_status(
        IntentStatusCode.INVALID_VEHICLE_STATE,
        count=0,
        future_valid=False,
        scalar_only=False,
        **base_status_kwargs,
      ),
    )

  invalid_model_code = None
  if not alive:
    invalid_model_code = IntentStatusCode.MESSAGE_NOT_ALIVE
  elif not message_is_valid:
    invalid_model_code = IntentStatusCode.MESSAGE_INVALID
  elif not fresh:
    invalid_model_code = IntentStatusCode.MESSAGE_STALE
  elif not scalar_valid:
    invalid_model_code = IntentStatusCode.INVALID_SCALAR

  if invalid_model_code is not None:
    frame = CanonicalFrame(
      timing=timing,
      validity=FrameValidity(
        model_valid=False,
        plan_valid=False,
        vehicle_state_valid=True,
        calibration_valid=True,
      ),
    )
    return IntentAdaptation(
      frame=frame,
      status=_status(
        invalid_model_code,
        count=0,
        future_valid=False,
        scalar_only=False,
        **base_status_kwargs,
      ),
    )

  future_arrays_valid = _future_arrays_valid(
    native_plan_times_s,
    native_orientation_rates_z,
    native_velocities_x,
  )
  action_supported = False
  current_supported = False
  effect_supported = False
  if future_arrays_valid:
    first_time = float(native_plan_times_s[0])
    last_time = float(native_plan_times_s[native_count - 1])
    action_supported = first_time <= action_time <= last_time
    current_supported = first_time <= plan_time_now_s <= last_time
    effect_supported = first_time <= physical_effect_plan_s <= last_time

  future_valid = (
    future_arrays_valid
    and action_supported
    and current_supported
    and effect_supported
  )
  frame = CanonicalFrame(
    timing=timing,
    validity=FrameValidity(
      model_valid=True,
      plan_valid=future_valid,
      vehicle_state_valid=True,
      calibration_valid=True,
    ),
  )
  if not future_valid:
    _write_scalar_only(
      scalar,
      action_time,
      output_plan_times_s,
      output_plan_curvatures,
    )
    code = (
      IntentStatusCode.SCALAR_ONLY_UNSUPPORTED_QUERY
      if future_arrays_valid
      else IntentStatusCode.SCALAR_ONLY_MALFORMED_FUTURE
    )
    return IntentAdaptation(
      frame=frame,
      status=_status(
        code,
        count=1,
        future_valid=False,
        scalar_only=True,
        action_query_supported=action_supported,
        current_query_supported=current_supported,
        effect_query_supported=effect_supported,
        **base_status_kwargs,
      ),
    )

  for index in range(native_count):
    plan_time = float(native_plan_times_s[index])
    orientation_rate = float(native_orientation_rates_z[index])
    planned_speed = float(native_velocities_x[index])
    output_plan_times_s[index] = plan_time
    output_orientation_rates_z[index] = orientation_rate
    output_velocities_x[index] = planned_speed
    output_plan_curvatures[index] = orientation_rate / planned_speed

  return IntentAdaptation(
    frame=frame,
    status=_status(
      IntentStatusCode.OK,
      count=native_count,
      future_valid=True,
      scalar_only=False,
      action_query_supported=True,
      current_query_supported=True,
      effect_query_supported=True,
      **base_status_kwargs,
    ),
  )

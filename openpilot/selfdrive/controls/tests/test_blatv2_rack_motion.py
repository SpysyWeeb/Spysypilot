from __future__ import annotations

import math

from openpilot.selfdrive.controls.lib.blatv2.rack_motion import (
  RackMotionSource,
  SignedRackMotionNormalizer,
)


GAP = 0.015
RESOLUTION = 4.0


def observe(
  normalizer: SignedRackMotionNormalizer,
  *,
  time_s: float,
  angle_deg: float,
  rate_deg_s: float,
  valid: bool = True,
):
  return normalizer.update(
    sample_time_s=time_s,
    steering_angle_deg=angle_deg,
    raw_rate_deg_s=rate_deg_s,
    rate_resolution_deg_s=RESOLUTION,
    lifecycle_valid=valid,
    maximum_gap_s=GAP,
  )


def test_unsigned_magnitude_uses_angle_direction_without_changing_magnitude():
  normalizer = SignedRackMotionNormalizer()
  first = observe(
    normalizer, time_s=1.00, angle_deg=0.0, rate_deg_s=8.0,
  )
  second = observe(
    normalizer, time_s=1.01, angle_deg=-0.1, rate_deg_s=8.0,
  )
  third = observe(
    normalizer, time_s=1.02, angle_deg=-0.1, rate_deg_s=8.0,
  )

  assert not first.sign_valid
  assert second.signed_rate_deg_s == -8.0
  assert not second.derivative_continuous
  assert third.signed_rate_deg_s == -8.0
  assert third.derivative_continuous
  assert third.source is RackMotionSource.CONTINUOUS_HOLD
  assert abs(third.signed_rate_deg_s) == 8.0


def test_zero_breaks_direction_carry_and_requires_new_physical_motion():
  normalizer = SignedRackMotionNormalizer()
  observe(normalizer, time_s=1.00, angle_deg=0.0, rate_deg_s=0.0)
  positive = observe(
    normalizer, time_s=1.01, angle_deg=0.1, rate_deg_s=4.0,
  )
  zero = observe(
    normalizer, time_s=1.02, angle_deg=0.1, rate_deg_s=0.0,
  )
  ambiguous = observe(
    normalizer, time_s=1.03, angle_deg=0.1, rate_deg_s=4.0,
  )
  negative = observe(
    normalizer, time_s=1.04, angle_deg=0.0, rate_deg_s=4.0,
  )

  assert positive.signed_rate_deg_s == 4.0
  assert zero.signed_rate_deg_s == 0.0
  assert not ambiguous.sign_valid
  assert negative.signed_rate_deg_s == -4.0
  assert not negative.derivative_continuous


def test_unsigned_nonzero_reversal_follows_measured_angle_motion():
  normalizer = SignedRackMotionNormalizer()
  observe(normalizer, time_s=1.00, angle_deg=0.0, rate_deg_s=8.0)
  positive = observe(
    normalizer, time_s=1.01, angle_deg=0.1, rate_deg_s=8.0,
  )
  reverse = observe(
    normalizer, time_s=1.02, angle_deg=0.0, rate_deg_s=8.0,
  )
  continued = observe(
    normalizer, time_s=1.03, angle_deg=-0.1, rate_deg_s=8.0,
  )

  assert positive.signed_rate_deg_s == 8.0
  assert reverse.signed_rate_deg_s == -8.0
  assert reverse.direction_reversal
  assert not reverse.derivative_continuous
  assert continued.signed_rate_deg_s == -8.0
  assert continued.derivative_continuous


def test_signed_source_preserves_both_raw_signs():
  normalizer = SignedRackMotionNormalizer()
  observe(normalizer, time_s=1.00, angle_deg=0.0, rate_deg_s=0.0)
  negative = observe(
    normalizer, time_s=1.01, angle_deg=-0.1, rate_deg_s=-8.0,
  )
  positive = observe(
    normalizer, time_s=1.02, angle_deg=0.0, rate_deg_s=8.0,
  )

  assert negative.signed_rate_deg_s == -8.0
  assert positive.signed_rate_deg_s == 8.0
  assert positive.direction_reversal
  assert not positive.derivative_continuous


def test_gap_invalid_and_route_reset_cannot_leak_direction():
  normalizer = SignedRackMotionNormalizer()
  observe(normalizer, time_s=1.00, angle_deg=0.0, rate_deg_s=0.0)
  observe(normalizer, time_s=1.01, angle_deg=0.1, rate_deg_s=8.0)
  gap = observe(
    normalizer, time_s=1.01 + GAP + 1e-6,
    angle_deg=0.2, rate_deg_s=8.0,
  )
  ambiguous = observe(
    normalizer, time_s=1.01 + GAP + 0.010001,
    angle_deg=0.2, rate_deg_s=8.0,
  )
  normalizer.reset()
  route_seed = observe(
    normalizer, time_s=2.00, angle_deg=0.2, rate_deg_s=8.0,
  )

  assert not gap.derivative_continuous
  assert not ambiguous.sign_valid
  assert not route_seed.sign_valid


def test_nonfinite_and_invalid_lifecycle_fail_closed():
  for kwargs in (
    {"time_s": math.nan, "angle_deg": 0.0, "rate_deg_s": 0.0},
    {"time_s": 1.0, "angle_deg": math.nan, "rate_deg_s": 0.0},
    {"time_s": 1.0, "angle_deg": 0.0, "rate_deg_s": math.nan},
    {
      "time_s": 1.0,
      "angle_deg": 0.0,
      "rate_deg_s": 0.0,
      "valid": False,
    },
  ):
    result = observe(SignedRackMotionNormalizer(), **kwargs)
    assert not result.sign_valid
    assert not result.derivative_continuous
    assert result.signed_rate_deg_s == 0.0

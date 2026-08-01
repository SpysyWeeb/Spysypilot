from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

from openpilot.selfdrive.controls.lib.blatv2.breakaway_episode import (
  BreakawayCategory,
  BreakawayEpisodeDetector,
)
from openpilot.selfdrive.controls.lib.blatv2.learner import LearningSample


DT = 0.01
RATE_RESOLUTION_DEG_S = 4.0
TRANSPORT_DELAY_S = 0.025


def sample(
  *,
  angle_deg: float,
  rate_deg_s: float = 0.0,
  speed_mps: float = 3.0,
  dt_s: float = DT,
  valid: bool = True,
  engaged: bool = True,
  steering_pressed: bool = False,
  standstill: bool = False,
) -> LearningSample:
  return LearningSample(
    speed_mps=speed_mps,
    dt_s=dt_s,
    applied_torque=0.2,
    measured_lateral_accel_mps2=0.3,
    rack_rate_deg_s=rate_deg_s,
    rack_acceleration_deg_s2=0.0,
    engaged=engaged,
    valid=valid,
    steering_pressed=steering_pressed,
    actuator_constrained=False,
    standstill=standstill,
    measured_rack_angle_deg=angle_deg,
  )


def update(
  detector: BreakawayEpisodeDetector,
  value: LearningSample,
):
  return detector.update(
    value,
    rack_rate_resolution_deg_s=RATE_RESOLUTION_DEG_S,
    transport_delay_s=TRANSPORT_DELAY_S,
  )


def prime_dwell(
  detector: BreakawayEpisodeDetector,
  angle_deg: float = 0.0,
) -> None:
  # The seed point establishes continuity; the following three measured
  # intervals provide 30 ms of stationary dwell.
  for _ in range(4):
    decision = update(detector, sample(angle_deg=angle_deg))
    if decision.category is not BreakawayCategory.BASE:
      raise AssertionError("stationary dwell was not classified as base")
    if decision.episode is not None:
      raise AssertionError("stationary dwell emitted an episode")


class TestBLaTv2BreakawayEpisode(unittest.TestCase):
  def test_angle_motion_precedes_rate_confirmation_in_both_directions(
    self,
  ) -> None:
    for direction in (-1, 1):
      with self.subTest(direction=direction):
        detector = BreakawayEpisodeDetector()
        prime_dwell(detector)

        first_motion = sample(
          angle_deg=direction * 0.1,
          speed_mps=2.5,
        )
        pending = update(detector, first_motion)
        self.assertIs(
          pending.category,
          BreakawayCategory.PENDING,
        )
        self.assertEqual(pending.direction, direction)
        self.assertIsNone(pending.episode)

        still_pending = update(
          detector,
          sample(angle_deg=direction * 0.2),
        )
        self.assertIs(
          still_pending.category,
          BreakawayCategory.PENDING,
        )

        confirmation = sample(
          angle_deg=direction * 0.3,
          rate_deg_s=direction * RATE_RESOLUTION_DEG_S,
        )
        decision = update(detector, confirmation)
        self.assertIs(
          decision.category,
          BreakawayCategory.BREAKAWAY,
        )
        self.assertEqual(decision.direction, direction)
        episode = decision.episode
        self.assertIsNotNone(episode)
        if episode is None:
          self.fail("confirmed breakaway lacks an episode")
        self.assertEqual(episode.direction, direction)
        self.assertEqual(
          episode.onset_speed_mps,
          first_motion.speed_mps,
        )
        self.assertAlmostEqual(episode.dwell_s, 3.0 * DT)
        self.assertEqual(
          episode.last_stuck.measured_rack_angle_deg,
          0.0,
        )
        self.assertEqual(
          episode.first_motion.measured_rack_angle_deg,
          first_motion.measured_rack_angle_deg,
        )
        self.assertEqual(
          episode.rate_confirmation.rack_rate_deg_s,
          confirmation.rack_rate_deg_s,
        )
        self.assertTrue(episode.angle_assisted)
        with self.assertRaises(FrozenInstanceError):
          episode.direction = -direction  # type: ignore[misc]

  def test_direct_resolved_rate_onset_after_dwell(self) -> None:
    for direction in (-1, 1):
      with self.subTest(direction=direction):
        detector = BreakawayEpisodeDetector()
        prime_dwell(detector)

        motion = sample(
          angle_deg=direction * 0.1,
          rate_deg_s=direction * RATE_RESOLUTION_DEG_S,
          speed_mps=4.0,
        )
        decision = update(detector, motion)

        self.assertIs(
          decision.category,
          BreakawayCategory.BREAKAWAY,
        )
        episode = decision.episode
        self.assertIsNotNone(episode)
        if episode is None:
          self.fail("direct breakaway lacks an episode")
        self.assertEqual(episode.direction, direction)
        self.assertEqual(episode.onset_speed_mps, 4.0)
        self.assertFalse(episode.angle_assisted)
        self.assertEqual(
          episode.first_motion,
          episode.rate_confirmation,
        )

  def test_insufficient_dwell_is_ordinary_motion(self) -> None:
    detector = BreakawayEpisodeDetector()
    self.assertIs(
      update(detector, sample(angle_deg=0.0)).category,
      BreakawayCategory.BASE,
    )
    self.assertIs(
      update(detector, sample(angle_deg=0.0)).category,
      BreakawayCategory.BASE,
    )

    decision = update(
      detector,
      sample(
        angle_deg=0.1,
        rate_deg_s=RATE_RESOLUTION_DEG_S,
      ),
    )
    self.assertIs(decision.category, BreakawayCategory.MOVING)
    self.assertEqual(decision.direction, 1)
    self.assertIsNone(decision.episode)

  def test_angle_resolved_motion_without_dwell_keeps_its_direction(
    self,
  ) -> None:
    for direction in (-1, 1):
      with self.subTest(direction=direction):
        detector = BreakawayEpisodeDetector()
        self.assertIs(
          update(detector, sample(angle_deg=0.0)).category,
          BreakawayCategory.BASE,
        )
        decision = update(
          detector,
          sample(angle_deg=direction * 0.1),
        )
        self.assertIs(decision.category, BreakawayCategory.MOVING)
        self.assertEqual(decision.direction, direction)

  def test_half_rate_quantum_is_the_angle_motion_boundary(self) -> None:
    detector = BreakawayEpisodeDetector()
    prime_dwell(detector)

    below = update(detector, sample(angle_deg=0.019))
    self.assertIs(below.category, BreakawayCategory.BASE)

    at_boundary = update(detector, sample(angle_deg=0.039))
    self.assertIs(at_boundary.category, BreakawayCategory.PENDING)
    self.assertIsNone(at_boundary.episode)

  def test_opposite_rate_confirmation_discards_pending(self) -> None:
    detector = BreakawayEpisodeDetector()
    prime_dwell(detector)
    self.assertIs(
      update(detector, sample(angle_deg=0.1)).category,
      BreakawayCategory.PENDING,
    )

    mismatch = update(
      detector,
      sample(
        angle_deg=0.1,
        rate_deg_s=-RATE_RESOLUTION_DEG_S,
      ),
    )
    self.assertIs(mismatch.category, BreakawayCategory.MOVING)
    self.assertEqual(mismatch.direction, -1)
    self.assertIsNone(mismatch.episode)
    self.assertIs(
      update(
        detector,
        sample(
          angle_deg=0.0,
          rate_deg_s=-RATE_RESOLUTION_DEG_S,
        ),
      ).category,
      BreakawayCategory.MOVING,
    )

  def test_opposite_angle_motion_discards_pending(self) -> None:
    detector = BreakawayEpisodeDetector()
    prime_dwell(detector)
    self.assertIs(
      update(detector, sample(angle_deg=0.1)).category,
      BreakawayCategory.PENDING,
    )

    mismatch = update(detector, sample(angle_deg=0.0))
    self.assertIs(mismatch.category, BreakawayCategory.MOVING)
    self.assertEqual(mismatch.direction, -1)
    self.assertIsNone(mismatch.episode)

  def test_disagreeing_angle_and_rate_is_discarded(self) -> None:
    detector = BreakawayEpisodeDetector()
    self.assertIs(
      update(detector, sample(angle_deg=0.0)).category,
      BreakawayCategory.BASE,
    )
    mismatch = update(
      detector,
      sample(
        angle_deg=0.1,
        rate_deg_s=-RATE_RESOLUTION_DEG_S,
      ),
    )
    self.assertIs(mismatch.category, BreakawayCategory.DISCARDED)
    self.assertEqual(mismatch.direction, 0)

  def test_rate_confirmation_after_transport_delay_expires(self) -> None:
    detector = BreakawayEpisodeDetector()
    prime_dwell(detector)
    self.assertIs(
      update(detector, sample(angle_deg=0.1)).category,
      BreakawayCategory.PENDING,
    )
    self.assertIs(
      update(detector, sample(angle_deg=0.1)).category,
      BreakawayCategory.PENDING,
    )
    self.assertIs(
      update(detector, sample(angle_deg=0.1)).category,
      BreakawayCategory.PENDING,
    )

    expired = update(
      detector,
      sample(
        angle_deg=0.2,
        rate_deg_s=RATE_RESOLUTION_DEG_S,
      ),
    )
    self.assertIs(expired.category, BreakawayCategory.MOVING)
    self.assertEqual(expired.direction, 1)
    self.assertIsNone(expired.episode)

  def test_timeout_frame_with_disagreeing_sensors_is_discarded(self) -> None:
    detector = BreakawayEpisodeDetector()
    prime_dwell(detector)
    self.assertIs(
      update(detector, sample(angle_deg=0.1)).category,
      BreakawayCategory.PENDING,
    )
    self.assertIs(
      update(detector, sample(angle_deg=0.1)).category,
      BreakawayCategory.PENDING,
    )
    self.assertIs(
      update(detector, sample(angle_deg=0.1)).category,
      BreakawayCategory.PENDING,
    )
    mismatch = update(
      detector,
      sample(
        angle_deg=0.2,
        rate_deg_s=-RATE_RESOLUTION_DEG_S,
      ),
    )
    self.assertIs(mismatch.category, BreakawayCategory.DISCARDED)
    self.assertEqual(mismatch.direction, 0)

  def test_invalid_lifecycle_and_gap_reset_dwell(self) -> None:
    reset_samples = (
      sample(angle_deg=0.0, valid=False),
      sample(angle_deg=0.0, engaged=False),
      sample(angle_deg=0.0, steering_pressed=True),
      sample(angle_deg=0.0, standstill=True),
      sample(angle_deg=0.0, dt_s=0.11),
    )
    for reset_sample in reset_samples:
      with self.subTest(reset_sample=reset_sample):
        detector = BreakawayEpisodeDetector()
        prime_dwell(detector)

        reset_decision = update(detector, reset_sample)
        self.assertIs(
          reset_decision.category,
          BreakawayCategory.BASE,
        )
        self.assertIsNone(reset_decision.episode)
        after_reset = update(
          detector,
          sample(
            angle_deg=0.1,
            rate_deg_s=RATE_RESOLUTION_DEG_S,
          ),
        )
        self.assertIs(
          after_reset.category,
          BreakawayCategory.MOVING,
        )
        self.assertIsNone(after_reset.episode)

  def test_gap_discards_a_pending_angle_onset(self) -> None:
    detector = BreakawayEpisodeDetector()
    prime_dwell(detector)
    self.assertIs(
      update(detector, sample(angle_deg=0.1)).category,
      BreakawayCategory.PENDING,
    )

    reset_decision = update(
      detector,
      sample(angle_deg=0.2, dt_s=0.11),
    )
    self.assertIs(reset_decision.category, BreakawayCategory.BASE)
    self.assertIsNone(reset_decision.episode)

    confirmation = update(
      detector,
      sample(
        angle_deg=0.3,
        rate_deg_s=RATE_RESOLUTION_DEG_S,
      ),
    )
    self.assertIs(confirmation.category, BreakawayCategory.MOVING)
    self.assertIsNone(confirmation.episode)

  def test_equal_streams_produce_equal_decisions_and_episodes(
    self,
  ) -> None:
    stream = (
      sample(angle_deg=0.0),
      sample(angle_deg=0.0),
      sample(angle_deg=0.0),
      sample(angle_deg=0.0),
      sample(angle_deg=-0.1),
      sample(angle_deg=-0.2),
      sample(
        angle_deg=-0.3,
        rate_deg_s=-RATE_RESOLUTION_DEG_S,
      ),
      sample(
        angle_deg=-0.4,
        rate_deg_s=-2.0 * RATE_RESOLUTION_DEG_S,
      ),
    )
    first = BreakawayEpisodeDetector()
    second = BreakawayEpisodeDetector()

    first_decisions = tuple(update(first, value) for value in stream)
    second_decisions = tuple(update(second, value) for value in stream)

    self.assertEqual(first_decisions, second_decisions)
    self.assertIs(
      first_decisions[-2].category,
      BreakawayCategory.BREAKAWAY,
    )
    self.assertEqual(
      first_decisions[-2].episode,
      second_decisions[-2].episode,
    )

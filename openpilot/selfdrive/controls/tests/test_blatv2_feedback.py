from pathlib import Path
import unittest

from openpilot.selfdrive.controls.lib.blatv2.feedback import (
  FEEDBACK_REQUEST_PARAM,
  FEEDBACK_RESPONSE_PARAM,
  FeedbackChoice,
  FeedbackPromptState,
  FeedbackRequest,
  FeedbackResponse,
  FeedbackValidationError,
  pending_feedback_request,
  write_feedback_response,
)


PROFILE_HASH = "a" * 64
ARTIFACT_HASH = "b" * 64


class MemoryParams:
  def __init__(self):
    self.values = {}
    self.puts = []

  def get(self, key, block=False):
    del block
    return self.values.get(key)

  def put(self, key, value, block=False):
    self.values[key] = value
    self.puts.append((key, value, block))


def request(revision=7, artifact_hash=ARTIFACT_HASH):
  return FeedbackRequest(artifact_hash, PROFILE_HASH, revision)


class TestBlatV2Feedback(unittest.TestCase):
  def test_request_and_response_round_trip_exact_canonical_contract(self):
    req = request()
    self.assertEqual(FeedbackRequest.from_param(req.to_param()), req)
    for choice in FeedbackChoice:
      with self.subTest(choice=choice):
        response = FeedbackResponse.for_request(req, choice)
        self.assertEqual(FeedbackResponse.from_param(response.to_param()), response)
        self.assertTrue(response.matches(req))

  def test_request_validation_rejects_noncanonical_payloads(self):
    payloads = (
      None,
      [],
      {},
      {
        "schemaVersion": 2,
        "artifactSha256": ARTIFACT_HASH,
        "profileSha256": PROFILE_HASH.upper(),
        "profileRevision": 7,
      },
      {
        "schemaVersion": 2,
        "artifactSha256": ARTIFACT_HASH,
        "profileSha256": PROFILE_HASH,
        "profileRevision": True,
      },
      {
        "schemaVersion": 2,
        "artifactSha256": ARTIFACT_HASH,
        "profileSha256": PROFILE_HASH,
        "profileRevision": 7,
        "extra": False,
      },
    )
    for payload in payloads:
      with self.subTest(payload=payload), self.assertRaises(FeedbackValidationError):
        FeedbackRequest.from_param(payload)

  def test_response_validation_rejects_unknown_choice_and_extra_keys(self):
    payload = FeedbackResponse.for_request(request(), FeedbackChoice.BETTER).to_param()
    payload["choice"] = "GOOD"
    with self.assertRaisesRegex(FeedbackValidationError, "choice"):
      FeedbackResponse.from_param(payload)
    payload["choice"] = FeedbackChoice.BETTER.value
    payload["driverIntervened"] = True
    with self.assertRaisesRegex(FeedbackValidationError, "canonical"):
      FeedbackResponse.from_param(payload)

  def test_valid_request_stays_pending_across_manager_and_road_transitions(self):
    params = MemoryParams()
    req = request()
    params.values[FEEDBACK_REQUEST_PARAM] = req.to_param()
    state = FeedbackPromptState()

    self.assertEqual(state.update(params, offroad=True), req)
    self.assertIsNone(state.update(params, offroad=True))
    self.assertIsNone(state.update(params, offroad=False))
    self.assertEqual(params.values[FEEDBACK_REQUEST_PARAM], req.to_param())
    self.assertNotIn(FEEDBACK_RESPONSE_PARAM, params.values)
    self.assertEqual(state.update(params, offroad=True), req)

  def test_matching_response_suppresses_repeat_but_other_profile_does_not(self):
    params = MemoryParams()
    req = request()
    params.values[FEEDBACK_REQUEST_PARAM] = req.to_param()
    params.values[FEEDBACK_RESPONSE_PARAM] = FeedbackResponse.for_request(
      request(revision=6),
      FeedbackChoice.BETTER,
    ).to_param()
    self.assertEqual(pending_feedback_request(params), req)

    self.assertTrue(write_feedback_response(params, req, FeedbackChoice.ABOUT_SAME))
    self.assertEqual(params.puts[-1][0], FEEDBACK_RESPONSE_PARAM)
    self.assertIs(params.puts[-1][2], True)
    self.assertIsNone(pending_feedback_request(params))

  def test_same_profile_wrapped_by_another_artifact_never_matches(self):
    params = MemoryParams()
    req = request()
    params.values[FEEDBACK_REQUEST_PARAM] = req.to_param()
    params.values[FEEDBACK_RESPONSE_PARAM] = FeedbackResponse.for_request(
      request(artifact_hash="c" * 64),
      FeedbackChoice.BETTER,
    ).to_param()
    self.assertEqual(pending_feedback_request(params), req)

  def test_replaced_request_and_external_matching_response_update_prompt_state(self):
    params = MemoryParams()
    original = request()
    params.values[FEEDBACK_REQUEST_PARAM] = original.to_param()
    state = FeedbackPromptState()
    self.assertEqual(state.update(params, offroad=True), original)

    replacement = request(revision=8)
    params.values[FEEDBACK_REQUEST_PARAM] = replacement.to_param()
    self.assertEqual(state.update(params, offroad=True), replacement)
    self.assertEqual(state.presented_request, replacement)

    params.values[FEEDBACK_RESPONSE_PARAM] = FeedbackResponse.for_request(
      replacement,
      FeedbackChoice.NOT_SURE,
    ).to_param()
    self.assertIsNone(state.update(params, offroad=True))
    self.assertIsNone(state.presented_request)

  def test_submit_binds_exact_request_and_never_writes_onroad_or_stale(self):
    params = MemoryParams()
    original = request()
    params.values[FEEDBACK_REQUEST_PARAM] = original.to_param()
    state = FeedbackPromptState()
    self.assertEqual(state.update(params, offroad=True), original)

    self.assertFalse(state.submit(params, FeedbackChoice.WORSE, offroad=False))
    self.assertEqual(params.puts, [])

    replacement = request(revision=8)
    params.values[FEEDBACK_REQUEST_PARAM] = replacement.to_param()
    self.assertFalse(state.submit(params, FeedbackChoice.WORSE, offroad=True))
    self.assertEqual(params.puts, [])

  def test_malformed_params_fail_closed_without_crashing_generic_ui(self):
    params = MemoryParams()
    params.values[FEEDBACK_REQUEST_PARAM] = {"profileSha256": PROFILE_HASH}
    params.values[FEEDBACK_RESPONSE_PARAM] = "not an object"
    self.assertIsNone(pending_feedback_request(params))
    self.assertIsNone(FeedbackPromptState().update(params, offroad=True))

    params.values[FEEDBACK_REQUEST_PARAM] = request().to_param()
    self.assertEqual(pending_feedback_request(params), request())

  def test_params_keys_are_persistent_json_and_have_no_clear_flags(self):
    root = Path(__file__).resolve().parents[3]
    keys = (root / "common" / "params_keys.h").read_text()
    self.assertIn('{"BLaTv2FeedbackRequest", {PERSISTENT, JSON}}', keys)
    self.assertIn('{"BLaTv2FeedbackResponse", {PERSISTENT, JSON}}', keys)

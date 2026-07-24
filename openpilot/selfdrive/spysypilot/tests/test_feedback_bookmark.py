from openpilot.selfdrive.ui.feedback.feedbackd import publish_bookmark


class FakePubMaster:
  def __init__(self):
    self.service = None
    self.message = None

  def send(self, service, message):
    self.service = service
    self.message = message.as_reader()


def test_audio_feedback_bookmark_remains_backward_compatible():
  pm = FakePubMaster()
  publish_bookmark(pm)
  bookmark = pm.message.userBookmark

  assert str(bookmark.source) == "generic"
  assert bookmark.eventType == ""
  assert bookmark.alertText1 == ""

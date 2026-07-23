from openpilot.selfdrive.ui.feedback.feedbackd import manual_long_event, publish_bookmark


class FakePubMaster:
  def __init__(self):
    self.service = None
    self.message = None

  def send(self, service, message):
    self.service = service
    self.message = message.as_reader()


def test_manual_bookmark_has_structured_ui_and_manifest_fields():
  pm = FakePubMaster()
  publish_bookmark(pm, manual_long_event(), source="manual")
  bookmark = pm.message.userBookmark

  assert pm.service == "userBookmark"
  assert str(bookmark.source) == "manual"
  assert bookmark.eventType == "manual_long_event"
  assert bookmark.alertText1 == "Long Event Logged"
  assert bookmark.alertText2 == "Manual bookmark"
  assert bookmark.confidence == 1.0


def test_generic_bookmark_remains_backward_compatible():
  pm = FakePubMaster()
  publish_bookmark(pm)
  bookmark = pm.message.userBookmark

  assert str(bookmark.source) == "generic"
  assert bookmark.eventType == ""
  assert bookmark.alertText1 == ""

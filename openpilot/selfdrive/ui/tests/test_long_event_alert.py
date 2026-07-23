from openpilot.selfdrive.spysypilot.long_event_detector import bookmark_alert_text


class Obj:
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)


def test_long_event_alert_uses_structured_text():
  bookmark = Obj(alertText1="Long Event Logged", alertText2="Late launch +1.6 s")
  assert bookmark_alert_text(bookmark) == ("Long Event Logged", "Late launch +1.6 s")


def test_generic_bookmark_keeps_stock_text():
  bookmark = Obj(alertText1="", alertText2="")
  assert bookmark_alert_text(bookmark) == ("Bookmark Saved", "")

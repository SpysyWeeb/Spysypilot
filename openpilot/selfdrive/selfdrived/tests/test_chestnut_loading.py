from openpilot.selfdrive.selfdrived.events import ET, Events, EventName


def test_big_model_loading_is_visible_and_blocks_entry():
  events = Events()
  events.add(EventName.bigModelLoading)
  assert events.contains(ET.NO_ENTRY)
  assert events.contains(ET.PERMANENT)
  alerts = events.create_alerts([ET.PERMANENT, ET.NO_ENTRY])
  assert {a.event_type for a in alerts} == {ET.PERMANENT, ET.NO_ENTRY}
  permanent = next(a for a in alerts if a.event_type == ET.PERMANENT)
  assert permanent.alert_text_1 == 'Big Model Loading'
  assert permanent.alert_text_2 == 'Waiting for the Chestnut, up to two minutes'

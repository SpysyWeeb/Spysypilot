#!/usr/bin/env python3
import openpilot.cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from opendbc.car.structs import car
from openpilot.selfdrive.spysypilot.long_event_detector import LaunchSample, LeadLaunchDetector, LongEvent
from openpilot.system.micd import SAMPLE_RATE, SAMPLE_BUFFER

FEEDBACK_MAX_DURATION = 10.0
ButtonType = car.CarState.ButtonEvent.Type


def publish_bookmark(pm: messaging.PubMaster, event: LongEvent | None = None, source: str = "generic") -> None:
  msg = messaging.new_message('userBookmark', valid=True)
  msg.userBookmark.source = source
  if event is not None:
    msg.userBookmark.eventType = event.event_type
    msg.userBookmark.alertText1 = event.title
    msg.userBookmark.alertText2 = event.detail
    msg.userBookmark.severity = event.severity
    msg.userBookmark.confidence = event.confidence
    msg.userBookmark.leadToEgoS = event.lead_to_ego_s
    msg.userBookmark.commandToEgoS = event.command_to_ego_s
    msg.userBookmark.planToLeadS = event.plan_to_lead_s
    msg.userBookmark.commandToLeadS = event.command_to_lead_s
    msg.userBookmark.forecastToLeadS = event.forecast_to_lead_s
  pm.send('userBookmark', msg)


def manual_long_event() -> LongEvent:
  return LongEvent(
    event_type="manual_long_event",
    title="Long Event Logged",
    detail="Manual bookmark",
    severity=0,
    confidence=1.0,
  )


def detector_sample(sm: messaging.SubMaster) -> LaunchSample:
  lead = sm['radarState'].leadOne
  leads_v3 = sm['modelV2'].leadsV3
  forecast_valid = sm.valid['modelV2'] and len(leads_v3) > 0 and leads_v3[0].prob > 0.5 and len(leads_v3[0].v) > 1
  predicted_lead_v_2s = 0.0
  if forecast_valid:
    predicted_lead_v_2s = float(lead.vLead) + float(leads_v3[0].v[1]) - float(leads_v3[0].v[0])

  return LaunchSample(
    t=sm.logMonoTime['carState'] / 1e9,
    active=bool(sm.valid['carState'] and sm.valid['carControl'] and sm['carControl'].longActive),
    standstill=bool(sm['carState'].standstill),
    v_ego=float(sm['carState'].vEgo),
    lead_present=bool(lead.present),
    radar_valid=bool(sm.valid['radarState']),
    d_rel=float(lead.dRel),
    v_lead=float(lead.vLead),
    v_lead_k=float(lead.vLeadK),
    radar_track_id=int(lead.radarTrackId),
    plan_valid=bool(sm.valid['longitudinalPlan']),
    plan_should_stop=bool(sm['longitudinalPlan'].shouldStop),
    output_valid=bool(sm.valid['carOutput']),
    output_accel=float(sm['carOutput'].actuatorsOutput.accel),
    forecast_valid=forecast_valid,
    predicted_lead_v_2s=predicted_lead_v_2s,
  )


def main():
  params = Params()
  pm = messaging.PubMaster(['userBookmark', 'audioFeedback'])
  sm = messaging.SubMaster(['rawAudioData', 'bookmarkButton', 'carState', 'radarState',
                            'longitudinalPlan', 'carControl', 'carOutput', 'modelV2'])
  launch_detector = LeadLaunchDetector()
  should_record_audio = False
  block_num = 0
  waiting_for_release = False
  early_stop_triggered = False

  while True:
    sm.update()
    should_send_bookmark = False

    # TODO: https://github.com/commaai/openpilot/issues/36015
    if False and sm.updated['carState'] and sm['carState'].canValid:
      for be in sm['carState'].buttonEvents:
        if be.type == ButtonType.lkas:
          if be.pressed:
            if not should_record_audio:
              if params.get_bool("RecordAudioFeedback"):  # Start recording on first press if toggle set
                should_record_audio = True
                block_num = 0
                waiting_for_release = False
                early_stop_triggered = False
                cloudlog.info("LKAS button pressed - starting 10-second audio feedback")
              else:
                should_send_bookmark = True  # immediately send bookmark if toggle false
                cloudlog.info("LKAS button pressed - bookmarking")
            elif should_record_audio and not waiting_for_release:  # Wait for release of second press to stop recording early
              waiting_for_release = True
          elif waiting_for_release:  # Second press released
            waiting_for_release = False
            early_stop_triggered = True
            cloudlog.info("LKAS button released - ending recording early")

    if should_record_audio and sm.updated['rawAudioData']:
      raw_audio = sm['rawAudioData']
      msg = messaging.new_message('audioFeedback', valid=True)
      msg.audioFeedback.audio.data = raw_audio.data
      msg.audioFeedback.audio.sampleRate = raw_audio.sampleRate
      msg.audioFeedback.blockNum = block_num
      block_num += 1
      if (block_num * SAMPLE_BUFFER / SAMPLE_RATE) >= FEEDBACK_MAX_DURATION or early_stop_triggered:  # Check for timeout or early stop
        should_send_bookmark = True  # send bookmark at end of audio segment
        should_record_audio = False
        early_stop_triggered = False
        cloudlog.info("10-second recording completed or second button press - stopping audio feedback")
      pm.send('audioFeedback', msg)

    if sm.updated['bookmarkButton']:
      cloudlog.info("Bookmark button pressed!")
      publish_bookmark(pm, manual_long_event(), source="manual")

    if sm.updated['carState']:
      event = launch_detector.update(detector_sample(sm))
      if event is not None:
        cloudlog.event("long_event_logged", event_type=event.event_type, severity=event.severity,
                       confidence=event.confidence, detail=event.detail)
        publish_bookmark(pm, event, source="automatic")

    if should_send_bookmark:
      publish_bookmark(pm)


if __name__ == '__main__':
  main()

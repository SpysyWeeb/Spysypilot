from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.values import CAR
from opendbc.car.structs import car
from opendbc.car.vehicle_model import VehicleModel

import openpilot.cereal.messaging as messaging
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.controlsd import Controls
from openpilot.selfdrive.controls.controlsd_ext import ControlsExt
from openpilot.selfdrive.controls.lib.lane_centering import LaneCentering
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.controls.lib.longcontrol import LongControl

LongCtrlState = car.CarControl.Actuators.LongControlState


class StubSubMaster:
  # Controls.__init__ needs a running msgq and a CarParams in the param store, so the tests below build the instance
  # directly and hand state_control() the same messages a live SubMaster would. LongControl is the real one: D31 is
  # about the order of a read against it, so mocking it would test nothing.
  def __init__(self, messages):
    self.messages = messages
    self.valid = dict.fromkeys(messages, True)

  def __getitem__(self, service):
    return self.messages[service]

  def all_checks(self, services=None):
    return True


def controls():
  CP = interfaces[CAR.HYUNDAI_PALISADE].get_non_essential_params(CAR.HYUNDAI_PALISADE)
  CP.openpilotLongitudinalControl = True
  CI = interfaces[CAR.HYUNDAI_PALISADE](CP)
  CP = CP.as_reader()

  messages = {}
  for service in ('lateralDelay', 'vehicleParameters', 'lateralTorqueParameters', 'modelV2', 'selfdriveState',
                  'longitudinalPlan', 'lateralManeuverPlan', 'carState', 'spysydriveStateSP'):
    messages[service] = getattr(messaging.new_message(service), service)
  messages['onroadEvents'] = []
  messages['vehicleParameters'].steerRatio = 16.0
  messages['vehicleParameters'].stiffnessFactor = 1.0
  # engaged longitudinally, hands-off lateral: latActive stays False so the lateral controller is a no-op here
  messages['selfdriveState'].enabled = True
  messages['carState'].vEgo = 10.0

  controlsd = Controls.__new__(Controls)
  controlsd.CP = CP
  controlsd.CI = CI
  controlsd.VM = VehicleModel(CP)
  controlsd.LoC = LongControl(CP)
  controlsd.LaC = LatControlTorque(CP, CI, DT_CTRL)
  # combo: the lateral tuning kind, lane centering and the Sometimes-On-Lateral extension state_control() consults
  # (AOL unavailable here)
  controlsd.lateral_tuning_type = CP.lateralTuning.which()
  controlsd.is_torque_lateral = controlsd.lateral_tuning_type == 'torque'
  controlsd.lane_centering = LaneCentering()
  controlsd.controls_ext = ControlsExt()
  controlsd.sm = StubSubMaster(messages)
  controlsd.steer_limited_by_safety = False
  controlsd.curvature = 0.0
  controlsd.desired_curvature = 0.0
  return controlsd, messages


def test_the_published_state_is_the_one_the_engage_frame_computed():
  # D31: on the first engaged frame LongControl leaves off for pid; reading the field before the update published off
  # alongside a pid acceleration, so the car interface saw last frame's state with this frame's accel.
  controlsd, messages = controls()
  messages['longitudinalPlan'].shouldStop = False

  CC, _ = controlsd.state_control()

  assert controlsd.LoC.long_control_state == LongCtrlState.pid
  assert CC.actuators.longControlState.raw == controlsd.LoC.long_control_state


def test_the_published_state_enters_stopping_on_the_frame_that_asks_to_stop():
  # the stop request the Hyundai interface sends, and the standstill-exit jerk limit, are derived from this field: on the
  # shouldStop frame the accel is already the stopping ramp, so the state published with it has to be stopping too
  controlsd, messages = controls()
  messages['longitudinalPlan'].shouldStop = False
  controlsd.state_control()

  messages['longitudinalPlan'].shouldStop = True
  CC, _ = controlsd.state_control()

  assert controlsd.LoC.long_control_state == LongCtrlState.stopping
  assert CC.actuators.longControlState.raw == controlsd.LoC.long_control_state


def test_the_published_state_leaves_stopping_on_the_frame_that_releases_the_stop():
  # the mirror case, the launch edge: stopping -> pid has to reach the interface with the frame that stops commanding
  # the stop, or the standstill-exit jerk limit is held one frame past the release (route 0x7e green-launch work)
  controlsd, messages = controls()
  messages['longitudinalPlan'].shouldStop = True
  controlsd.state_control()
  assert controlsd.LoC.long_control_state == LongCtrlState.stopping

  messages['longitudinalPlan'].shouldStop = False
  CC, _ = controlsd.state_control()

  assert controlsd.LoC.long_control_state == LongCtrlState.pid
  assert CC.actuators.longControlState.raw == controlsd.LoC.long_control_state

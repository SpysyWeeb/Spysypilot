import ctypes
from unittest.mock import Mock

import pytest

from openpilot.system.hardware.chestnut import flash


@pytest.mark.parametrize('states, expected', [([0x78], True), ([0, 0x78], True), ([0, 0], False), ([OSError()], False)])
def test_link_checks_before_powering(monkeypatch, states, expected):
  monkeypatch.setattr(flash, 'find_chestnut', lambda: ('/unused', None, None))
  monkeypatch.setattr(flash, 'open_device', lambda _: 123)
  close = Mock()
  monkeypatch.setattr(flash.os, 'close', close)
  requests, replies = [], iter(states)

  def ioctl(fd, request, ctrl):
    assert (fd, request) == (123, flash.USBDEVFS_CONTROL)
    requests.append(ctrl.request)
    if ctrl.request == 0xE4:
      assert (ctrl.request_type, ctrl.value, ctrl.length) == (0xC0, 0xB450, 1)
      value = next(replies)
      if isinstance(value, Exception):
        raise value
      ctypes.cast(ctrl.data, ctypes.POINTER(ctypes.c_ubyte))[0] = value
    else:
      assert (ctrl.request_type, ctrl.request, ctrl.value) == (0x40, 0xF3, 1)

  monkeypatch.setattr(flash.fcntl, 'ioctl', ioctl)
  assert flash.link_up() is expected
  assert requests == ([0xE4] if len(states) == 1 else [0xE4, 0xF3, 0xE4])
  close.assert_called_once_with(123)


def test_absent_chestnut_never_opens_usb(monkeypatch):
  monkeypatch.setattr(flash, 'find_chestnut', lambda: (None, None, None))
  opened = Mock(side_effect=AssertionError('must not open USB'))
  monkeypatch.setattr(flash, 'open_device', opened)
  assert not flash.link_up()
  opened.assert_not_called()

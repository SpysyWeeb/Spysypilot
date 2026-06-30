import collections
import subprocess
import threading
import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import Widget

_BG     = rl.Color(15, 15, 15, 255)
_TEXT   = rl.Color(140, 255, 140, 255)
_DIM    = rl.Color(255, 255, 255, 80)
_BORDER = rl.Color(255, 255, 255, 25)

MAX_LINES = 35
FONT_SIZE  = 24
LINE_H     = 30
PAD        = 16


class TerminalWidget(Widget):
    def __init__(self):
        super().__init__()
        self._lines: collections.deque[str] = collections.deque(maxlen=MAX_LINES)
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._background_tap_callback = None

    def set_background_tap_callback(self, cb) -> None:
        self._background_tap_callback = cb

    def show_event(self) -> None:
        super().show_event()
        self._start()

    def hide_event(self) -> None:
        super().hide_event()
        self._stop()

    def _start(self) -> None:
        self._stop()
        with self._lock:
            self._lines.clear()
        try:
            self._proc = subprocess.Popen(
                ['journalctl', '-f', '-n', '50', '--no-pager', '-o', 'short-monotonic', '--no-hostname'],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
        except FileNotFoundError:
            with self._lock:
                self._lines.append('[journalctl not found]')
            return
        threading.Thread(target=self._reader, daemon=True).start()

    def _stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None

    def _reader(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            if self._proc is None:
                break
            with self._lock:
                self._lines.append(line.rstrip('\n'))

    def _handle_mouse_release(self, mouse_pos) -> None:
        if self._background_tap_callback:
            self._background_tap_callback()

    def _render(self, rect: rl.Rectangle) -> None:
        rl.draw_rectangle_rounded(rect, 0.025, 10, _BG)
        rl.draw_rectangle_rounded_lines_ex(rect, 0.025, 10, 1, _BORDER)

        with self._lock:
            lines = list(self._lines)

        fn = gui_app.font(FontWeight.NORMAL)

        if not lines:
            rl.draw_text_ex(fn, 'Waiting for output...', rl.Vector2(int(rect.x + PAD), int(rect.y + PAD)),
                            FONT_SIZE, 0, _DIM)
            return

        max_visible = max(1, int((rect.height - PAD * 2) / LINE_H))
        visible = lines[-max_visible:]
        max_chars = max(1, int((rect.width - PAD * 2) / (FONT_SIZE * 0.56)))

        rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
        y = int(rect.y + PAD)
        x = int(rect.x + PAD)
        for line in visible:
            rl.draw_text_ex(fn, line[:max_chars], rl.Vector2(x, y), FONT_SIZE, 0, _TEXT)
            y += LINE_H
        rl.end_scissor_mode()

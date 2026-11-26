"""Optional offline voice prompts (espeak-ng + paplay). Silent no-op if tools are missing."""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import tempfile
import threading

AUDIO_SINK = os.environ.get("ECG_AUDIO_SINK", "bluez_output.AC_55_68_36_A0_A8.1")
RATE = "145"


class VoiceWorker:
    def __init__(self, enabled: bool = True):
        self.available = bool(enabled and shutil.which("espeak-ng") and shutil.which("paplay"))
        self._q: "queue.Queue[str | None]" = queue.Queue()
        self._t = None
        if self.available:
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()

    def say(self, text: str):
        if self.available and text:
            while self._q.qsize() > 1:       # drop stale prompts
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
            self._q.put(text)

    def _run(self):
        while True:
            msg = self._q.get()
            if msg is None:
                return
            wav = None
            try:
                fd, wav = tempfile.mkstemp(prefix="ecg_voice_", suffix=".wav")
                os.close(fd)
                ok = False
                for voice in ("mb-us1", "en-us+f3", "en-us"):
                    r = subprocess.run(["espeak-ng", "-v", voice, "-s", RATE, "-w", wav, msg],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
                    if r.returncode == 0 and os.path.getsize(wav) > 44:
                        ok = True
                        break
                if not ok:
                    continue
                cmd = ["paplay"] + ([f"--device={AUDIO_SINK}"] if AUDIO_SINK else []) + [wav]
                r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=35)
                if r.returncode != 0 and AUDIO_SINK:          # fall back to default sink
                    subprocess.run(["paplay", wav], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=35)
            except Exception:
                pass
            finally:
                if wav and os.path.exists(wav):
                    os.remove(wav)

    def close(self):
        if self._t:
            self._q.put(None)

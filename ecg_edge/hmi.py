"""Tk HMI. One START press = one 5 s capture + one classification, then idle.

Single process: the HMI owns the ADC and the models directly (no web server,
no HTTP polling). Acquisition runs in a worker thread that is created per
cycle and ends when the cycle ends; results come back through a queue, so Tk
is only ever touched from the main thread.
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from . import config as C
from . import decision as D
from .acquisition import Cancelled, Progress
from .pipeline import CycleResult, run_cycle

BG, FG, MUTED, BORDER = "#ffffff", "#17212b", "#5f6b76", "#d6dde4"
SEV = {"NORMAL": ("#eaf6ee", "#198754"), "WARNING": ("#fff6df", "#b77900"),
       "CRITICAL": ("#fff0ef", "#c62828"), "INVALID": ("#eceff1", "#455a64")}
IDLE, ACQUIRING, PROCESSING = "IDLE", "ACQUIRING", "PROCESSING"


def _log(msg: str):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:        # legacy Windows console / redirected output
        print(msg.encode("ascii", "replace").decode(), flush=True)



class ECGApp:
    def __init__(self, root: tk.Tk, adc, cascade, voice=None):
        self.root, self.adc, self.cascade, self.voice = root, adc, cascade, voice
        self.state = IDLE
        self.cycles = 0
        self.closed = False
        self.progress: Optional[Progress] = None
        self.cancel_ev: Optional[threading.Event] = None
        self.worker: Optional[threading.Thread] = None
        self.results: "queue.Queue[CycleResult | None]" = queue.Queue()
        self.last_speech = "No result yet."

        root.title("ECG Edge AI — 5 s Arrhythmia Check")
        root.geometry("940x600")
        root.minsize(780, 520)
        root.configure(bg=BG)
        self._build()
        root.protocol("WM_DELETE_WINDOW", self.exit_app)
        self._set_state(IDLE, "Fix the electrodes, then press START.")
        if self.voice:
            self.voice.say("Please fix the electrodes, then press Start.")

    # ----------------------------------------------------------------- UI
    def _build(self):
        f = tk.Frame(self.root, bg=BG, padx=12, pady=6)
        f.pack(fill="both", expand=True)
        top = tk.Frame(f, bg=BG); top.pack(fill="x")
        tk.Label(top, text="ECG ARRHYTHMIA EDGE AI", bg=BG, fg=FG,
                 font=("DejaVu Sans", 13, "bold")).pack(side="left")
        if getattr(self.adc, "simulated", False):
            tk.Label(top, text=f"  SIMULATED SIGNAL ({getattr(self.adc, 'kind', '?')})  ", bg="#c62828",
                     fg="white", font=("DejaVu Sans", 9, "bold")).pack(side="left", padx=10)

        self.banner_var = tk.StringVar(value="READY")
        self.banner = tk.Label(f, textvariable=self.banner_var, bg="#eceff1", fg="#455a64",
                               font=("DejaVu Sans", 22, "bold"), pady=12)
        self.banner.pack(fill="x", pady=(6, 2))
        self.detail_var = tk.StringVar(value="")
        tk.Label(f, textvariable=self.detail_var, bg=BG, fg=MUTED, font=("DejaVu Sans", 9),
                 anchor="w", wraplength=900, justify="left").pack(fill="x")

        tk.Label(f, text="Research prototype — not a medical device. Do not use for treatment decisions.",
                 bg=BG, fg=MUTED, font=("DejaVu Sans", 7)).pack(side="bottom", pady=(2, 0))
        row = tk.Frame(f, bg=BG); row.pack(side="bottom", fill="x", pady=(4, 0))
        def btn(text, cmd, fg):
            b = tk.Button(row, text=text, command=cmd, fg=fg, font=("DejaVu Sans", 10, "bold"), padx=12, pady=6)
            b.pack(side="left", expand=True, fill="x", padx=3)
            return b
        self.start_btn = btn("START", self.start, "#198754")
        self.cancel_btn = btn("CANCEL", self.cancel, "#c62828")
        self.audio_btn = btn("REPEAT AUDIO", lambda: self.voice and self.voice.say(self.last_speech), FG)
        self.exit_btn = btn("EXIT", self.exit_app, "#c62828")
        self.warn_var = tk.StringVar()
        tk.Label(f, textvariable=self.warn_var, bg=BG, fg="#b77900", font=("DejaVu Sans", 8),
                 anchor="w").pack(side="bottom", fill="x")
        self.status_var = tk.StringVar()
        tk.Label(f, textvariable=self.status_var, bg=BG, fg=FG, font=("DejaVu Sans", 9),
                 anchor="w").pack(side="bottom", fill="x")

        self.fig = Figure(figsize=(9.3, 2.7), dpi=80, facecolor=BG)
        self.ax = self.fig.add_subplot(111)
        self._reset_plot()
        self.canvas = FigureCanvasTkAgg(self.fig, master=f)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=(4, 3))

    def _reset_plot(self, title="Filtered ECG (5 s, 0.5–40 Hz + 50 Hz notch) — waiting for capture"):
        ax = self.ax
        ax.clear()
        ax.set_facecolor("#ffffff")
        ax.set_title(title, color=FG, fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=8); ax.set_ylabel("mV", fontsize=8)
        ax.set_xlim(0, C.CAPTURE_SEC); ax.set_ylim(-1.5, 1.5)
        ax.grid(True, color="#cfd8df", linewidth=0.5)
        ax.tick_params(labelsize=7)
        self.line, = ax.plot([], [], color="#b71c1c", linewidth=0.9)
        self.fig.subplots_adjust(left=0.06, right=0.975, bottom=0.2, top=0.88)

    def _set_state(self, state, status):
        self.state = state
        self.status_var.set(status)
        busy = state != IDLE
        self.start_btn.configure(state="disabled" if busy else "normal")
        self.cancel_btn.configure(state="normal" if state == ACQUIRING else "disabled")

    # ------------------------------------------------------------ cycle control
    def start(self):
        if self.state != IDLE or self.closed:
            return                                   # Start is ignored unless idle
        self.progress, self.cancel_ev = Progress(), threading.Event()
        self.banner_var.set("ACQUIRING…")
        self.banner.configure(bg="#eceff1", fg="#455a64")
        self.detail_var.set(""); self.warn_var.set("")
        self._reset_plot("Acquiring 5 s…")
        self.canvas.draw_idle()
        self._set_state(ACQUIRING, "Acquiring ECG: 0.0 / 5.0 s — keep still.")
        self.worker = threading.Thread(target=self._work, args=(self.progress, self.cancel_ev),
                                       daemon=True, name="ECGCycle")
        self.worker.start()
        self.root.after(100, self._poll)

    def _work(self, progress, cancel):
        try:
            res = run_cycle(self.adc, self.cascade, progress=progress, cancel=cancel)
        except Cancelled:
            res = None
        except Exception as exc:                     # never leave the HMI stuck in a busy state
            res = CycleResult(D.error(f"{type(exc).__name__}: {exc}"))
        self.results.put(res)

    def _poll(self):
        if self.closed:
            return
        try:
            res = self.results.get_nowait()
        except queue.Empty:
            if self.progress and self.state == ACQUIRING:
                n = self.progress.n
                secs = n / C.ADC_SPS
                if n >= self.progress.total:
                    self._set_state(PROCESSING, "Processing + classifying…")
                else:
                    self.status_var.set(f"Acquiring ECG: {secs:.1f} / {C.CAPTURE_SEC:.1f} s — keep still.")
            self.root.after(100, self._poll)
            return
        self._show(res)

    def cancel(self):
        if self.state == ACQUIRING and self.cancel_ev:
            self.cancel_ev.set()

    # ------------------------------------------------------------------ display
    def _show(self, res: Optional[CycleResult]):
        self.cycles += 1
        if res is None:                              # cancelled
            self.banner_var.set("CANCELLED")
            self._reset_plot()
            self.canvas.draw_idle()
            self._set_state(IDLE, "Cancelled. Press START to run a new 5 s capture.")
            return
        d = res.decision
        bg, fg = SEV[d.severity]
        self.banner_var.set(d.headline)
        self.banner.configure(bg=bg, fg=fg)
        extra = ("  |  " + "; ".join(d.notes)) if d.notes else ""
        self.detail_var.set((d.details or "") + extra)
        self.warn_var.set("  ".join(res.warnings))
        self._reset_plot("No valid capture — waveform not shown")
        if res.filtered is not None:
            t = np.arange(len(res.filtered)) / res.fs
            lim = max(1.0, 1.2 * float(np.max(np.abs(res.filtered))))
            self.line.set_data(t, res.filtered)
            self.ax.set_ylim(-lim, lim)
            self.ax.set_title("Filtered ECG — exactly the 5 s window the models classified "
                              "(0.5–40 Hz + 50 Hz notch)", color=FG, fontsize=8)
        self.canvas.draw_idle()
        cap = res.capture
        timing = f" | capture {cap.elapsed_s:.3f} s" if cap else ""
        self.last_speech = d.speech
        if self.voice:
            self.voice.say(d.speech)
        _log(f"[cycle {self.cycles}] {d.headline} | {d.details}{timing}")
        self._set_state(IDLE, f"Done{timing} | latency {res.latency_ms:.0f} ms. "
                              "Idle — press START for a new 5 s capture.")

    def exit_app(self):
        if self.closed:
            return
        self.closed = True
        if self.cancel_ev:
            self.cancel_ev.set()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=2.0)
        try:
            if self.voice:
                self.voice.close()
            self.adc.close()
        finally:
            self.root.destroy()

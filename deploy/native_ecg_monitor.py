#!/usr/bin/env python3
"""ECG Edge-AI Native HMI — Multi-Tier Arrhythmia Decision Monitor.

The app connects to the edge backend (http://127.0.0.1:5000/data) and visualizes:
- Tier 1: Shockable Screener (Model 1: Shockable vs Non-Shockable)
- Tier 2: 4-Class General Arrhythmia (Model 2: NSR, Tachy, Bradycardia, Ventricular)
- Tier 3: Ventricular Specialist (Model 3: VT vs VF Subtype)
- Adaptive AFib Branch: Patient-Adaptive Atrial Fibrillation Alert
- Master Unified Decision & Triage Urgency Level

Waveform handling:
- Repeating 8-second captures with 4-second animated reveal.
- Fixed -2 mV to +2 mV scale.
- Impulse spike suppression & 0.5-40 Hz bandpass filtering.

This is an experimental interface, not a validated medical device. Do not use
model labels alone to make treatment or defibrillation decisions.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import tkinter as tk
from datetime import datetime
from urllib.request import Request, urlopen

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, find_peaks, medfilt
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

API_URL = os.environ.get("ECG_API_URL", "http://127.0.0.1:5000/data")
CAPTURE_SECONDS = 8.0
ANIMATION_SECONDS = 4.0
POLL_SECONDS = 0.15
POLL_MS = 300
AUDIO_SINK = os.environ.get("ECG_AUDIO_SINK", "bluez_output.AC_55_68_36_A0_A8.1")
VOICE_RATE = "145"

# Clean Medical-Grade Theme
BG, FG, MUTED, BORDER = "#ffffff", "#17212b", "#5f6b76", "#d6dde4"
GREEN, RED, AMBER = "#198754", "#c62828", "#b77900"
LIGHT_GREEN, LIGHT_RED, LIGHT_AMBER = "#eaf6ee", "#fff0ef", "#fff6df"


def get_state(payload):
    state = payload.get("state", {}) if isinstance(payload, dict) else {}
    return state if isinstance(state, dict) else {}


def classify_telemetry(payload):
    """Extracts all 4 decision layers from the backend JSON payload."""
    state = get_state(payload)
    quality = str(state.get("signal_quality", "UNKNOWN")).upper()
    details = str(state.get("signal_details", "")).upper()

    # Tier 0: Signal Quality Guard
    if any(x in quality for x in ("POOR", "BAD", "INVALID", "UNUSABLE", "LEAD_OFF")) or any(
        x in details for x in ("POOR SIGNAL", "LEAD OFF", "INVALID SIGNAL")
    ):
        return {
            "status": "POOR_SIGNAL",
            "shockable": "POOR SIGNAL / LEAD OFF",
            "decision": "CHECK ELECTRODE CONTACT",
            "triage": "SIGNAL_INVALID",
            "tier2_rhythm": "--",
            "tier3_subtype": "--",
            "afib_summary": "AFib: N/A",
            "voice": "Poor signal quality. Please check ECG electrode contact.",
        }

    # Tier 1: Model 1 Shockable Screener
    m1 = state.get("model1", {}) if isinstance(state.get("model1"), dict) else {}
    shockable_bool = m1.get("is_shockable")
    label_m1 = str(m1.get("label", "")).upper()
    if isinstance(shockable_bool, bool):
        shockable_str = "SHOCKABLE" if shockable_bool else "NOT SHOCKABLE"
    elif "SHOCKABLE" in label_m1 and "NON" not in label_m1 and "NOT" not in label_m1:
        shockable_str = "SHOCKABLE"
    else:
        shockable_str = "NOT SHOCKABLE"

    # Tier 2: Model 2 Multi-Arrhythmia
    m2 = state.get("model2", {}) if isinstance(state.get("model2"), dict) else {}
    raw_m2 = str(m2.get("rhythm") or m2.get("raw_m2") or "--").upper().strip()
    if "BRADY" in raw_m2:
        tier2_clean = "BRADYCARDIA"
    elif raw_m2 in ("NSR", "NORMAL"):
        tier2_clean = "NORMAL SINUS"
    elif "TACHY" in raw_m2:
        tier2_clean = "TACHYCARDIA"
    elif "VENTRICULAR" in raw_m2:
        tier2_clean = "VENTRICULAR"
    else:
        tier2_clean = raw_m2.replace("_", " ")

    # Tier 3: Model 3 Ventricular Subtype Specialist (VT vs VF)
    m3 = state.get("model3", {}) if isinstance(state.get("model3"), dict) else {}
    m3_active = bool(m3.get("active", False))
    m3_subtype = str(m3.get("subtype", "--")).upper().strip()
    if m3_active and m3_subtype in ("VT", "VF"):
        tier3_str = f"SUBTYPE: {m3_subtype}"
    else:
        tier3_str = "--"

    # AFib Adaptive Branch
    afib = state.get("afib", {}) if isinstance(state.get("afib"), dict) else {}
    afib_alert = bool(afib.get("alert", False))
    afib_prob = float(afib.get("probability", 0.0))
    if afib_alert:
        afib_summary = f"AFib: ACTIVE ({afib_prob:.0f}%)"
    else:
        afib_summary = f"AFib: Neg ({afib_prob:.0f}%)"

    # Master Unified Decision from Backend Arbitration
    master_raw = str(state.get("final_decision", "")).upper().replace("-", "_").replace(" ", "_")
    triage = str(state.get("triage_level", "ROUTINE_NORMAL")).upper()

    if "VENTRICULAR_FIBRILLATION" in master_raw or (m3_active and m3_subtype == "VF"):
        master_clean = "VENTRICULAR FIBRILLATION"
        shockable_str = "SHOCKABLE"
        voice_msg = "Critical emergency: Ventricular Fibrillation detected! Immediate defibrillation shock advised."
    elif "VENTRICULAR_TACHYCARDIA" in master_raw or (m3_active and m3_subtype == "VT"):
        master_clean = "VENTRICULAR TACHYCARDIA"
        voice_msg = "Critical warning: Ventricular Tachycardia detected. Evaluate patient vitals immediately."
    elif "ATRIAL_FIBRILLATION" in master_raw or afib_alert:
        master_clean = "ATRIAL FIBRILLATION"
        voice_msg = "Atrial Fibrillation detected by adaptive rhythm analysis."
    elif "BRADY" in master_raw or tier2_clean == "BRADYCARDIA":
        master_clean = "BRADYCARDIA / PAUSE"
        voice_msg = "Bradycardia detected. Heart rate is abnormally slow."
    elif "TACHY" in master_raw or tier2_clean == "TACHYCARDIA":
        master_clean = "TACHYCARDIA"
        voice_msg = "Tachycardia detected. Heart rate is elevated."
    elif "NORMAL" in master_raw or tier2_clean == "NORMAL SINUS":
        master_clean = "NORMAL SINUS RHYTHM"
        voice_msg = "Normal Sinus Rhythm. Rhythm is stable."
    elif "UNCERTAIN" in master_raw or "CONFLICT" in master_raw:
        master_clean = "UNCERTAIN / CONFLICTING"
        voice_msg = "Uncertain or conflicting rhythm detection. Clinical verification advised."
    else:
        master_clean = master_raw.replace("_", " ") if master_raw else tier2_clean
        voice_msg = f"Classification: {shockable_str}. Rhythm: {master_clean}."

    return {
        "status": "OK",
        "shockable": shockable_str,
        "decision": master_clean,
        "triage": triage,
        "tier2_rhythm": tier2_clean,
        "tier3_subtype": tier3_str,
        "afib_summary": afib_summary,
        "afib_alert": afib_alert,
        "voice": voice_msg,
    }


def waveform_from_payload(payload):
    if not isinstance(payload, dict):
        return np.array([], dtype=float)
    for key in ("filtered", "ecg_filtered", "ecg", "waveform", "raw"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            try:
                arr = np.asarray(value, dtype=float).ravel()
                return arr[np.isfinite(arr)]
            except (TypeError, ValueError):
                continue
    return np.array([], dtype=float)


def robust_spike_suppression(x, fs):
    """Replace isolated impulsive touch/noise spikes using a local Hampel-like test."""
    if len(x) < 7:
        return x, 0
    if SCIPY_OK:
        window = max(7, int(round(0.025 * fs)))
        if window % 2 == 0:
            window += 1
        window = min(window, len(x) if len(x) % 2 else len(x) - 1)
        if window < 3:
            return x, 0
        local_med = medfilt(x, kernel_size=window)
        residual = x - local_med
        residual_med = np.median(residual)
        residual_mad = np.median(np.abs(residual - residual_med))
        robust_sigma = 1.4826 * residual_mad
        dx = np.diff(x, prepend=x[0])
        dx_med = np.median(dx)
        dx_mad = 1.4826 * np.median(np.abs(dx - dx_med))
        threshold = max(8.0 * robust_sigma, 0.20)
        derivative_threshold = max(10.0 * dx_mad, 0.25)
        outlier = (np.abs(residual) > threshold) & (np.abs(dx - dx_med) > derivative_threshold)
        if outlier.sum() <= max(1, int(0.03 * len(x))):
            x[outlier] = local_med[outlier]
            return x, int(outlier.sum())
    else:
        med = np.median(x)
        mad = 1.4826 * np.median(np.abs(x - med))
        if mad > 0:
            threshold = 8.0 * mad
            outlier = np.abs(x - med) > threshold
            if outlier.sum() <= max(1, int(0.02 * len(x))):
                x[outlier] = med
                return x, int(outlier.sum())
    return x, 0


def clean_ecg(signal, fs):
    """Baseline-normalize, suppress isolated spikes, then apply suitable ECG filters."""
    x = np.asarray(signal, dtype=float).copy()
    x = x[np.isfinite(x)]
    if x.size < 8:
        return x, "Too few samples for full filtering", None
    x -= np.median(x)
    notes = []
    x, suppressed = robust_spike_suppression(x, fs)
    notes.append(f"spikes: {suppressed}")

    if SCIPY_OK and fs > 2:
        try:
            nyquist = fs / 2.0
            low = 0.5
            high = min(40.0, nyquist * 0.85)
            if high > low and x.size > max(30, int(fs * 0.5)):
                sos = butter(3, [low, high], btype="bandpass", fs=fs, output="sos")
                x = sosfiltfilt(sos, x)
                notes.append(f"{low:g}-{high:g} Hz band-pass")
            if 50.0 < nyquist and x.size > 30:
                b, a = iirnotch(50.0, 30.0, fs)
                x = filtfilt(b, a, x)
                notes.append("50 Hz notch")
        except Exception as exc:
            notes.append("filter fallback: " + str(exc)[:40])
    else:
        notes.append("SciPy unavailable; baseline normalization only")

    x -= np.median(x)
    x, suppressed_after = robust_spike_suppression(x, fs)
    if suppressed_after:
        notes.append(f"post-filter spikes: {suppressed_after}")
    return x, "; ".join(notes), estimate_qrs_count(x, fs)


def estimate_qrs_count(x, fs):
    """Rough Pan–Tompkins-style QRS estimate."""
    if not SCIPY_OK or fs <= 0 or len(x) < int(2 * fs):
        return None
    try:
        dx = np.diff(x, prepend=x[0])
        energy = dx * dx
        win = max(1, int(0.12 * fs))
        integrated = np.convolve(energy, np.ones(win) / win, mode="same")
        threshold = np.median(integrated) + 2.5 * 1.4826 * np.median(
            np.abs(integrated - np.median(integrated))
        )
        peaks, _ = find_peaks(integrated, height=threshold, distance=max(1, int(0.25 * fs)))
        return int(len(peaks))
    except Exception:
        return None


class VoiceWorker:
    """Offline speech synthesis worker."""
    def __init__(self, status_callback):
        self.status_callback = status_callback
        self.messages = queue.Queue()
        self.closed = False
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def say(self, message):
        if message and not self.closed:
            try:
                while self.messages.qsize() > 2:
                    self.messages.get_nowait()
                    self.messages.task_done()
            except queue.Empty:
                pass
            self.messages.put(str(message))

    def _status(self, text):
        try:
            self.status_callback(text)
        except Exception:
            pass

    def _run(self):
        while not self.closed:
            try:
                message = self.messages.get(timeout=0.2)
            except queue.Empty:
                continue
            if message is None:
                self.messages.task_done()
                break
            wav = None
            try:
                if not shutil.which("espeak-ng") or not shutil.which("paplay"):
                    self._status("Audio tools missing")
                    continue
                fd, wav = tempfile.mkstemp(prefix="ecg_voice_", suffix=".wav")
                os.close(fd)
                generated = False
                for voice in ("mb-us1", "en-us+f3", "en-us"):
                    result = subprocess.run(
                        ["espeak-ng", "-v", voice, "-s", VOICE_RATE, "-w", wav, message],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        timeout=20, check=False
                    )
                    if result.returncode == 0 and os.path.getsize(wav) > 44:
                        generated = True
                        break
                if not generated:
                    self._status("Speech generation failed")
                    continue
                command = ["paplay"]
                if AUDIO_SINK:
                    command.append("--device=" + AUDIO_SINK)
                command.append(wav)
                played = subprocess.run(command, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, timeout=35, check=False)
                if played.returncode != 0 and AUDIO_SINK:
                    subprocess.run(["paplay", wav], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=35, check=False)
                self._status("Voice prompt played")
            except Exception as exc:
                self._status("Audio error: " + str(exc)[:50])
            finally:
                if wav:
                    try:
                        os.remove(wav)
                    except OSError:
                        pass
                self.messages.task_done()

    def close(self):
        self.closed = True
        self.messages.put(None)


class ECGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG Edge AI — Unified Arrhythmia Monitor")
        self.root.geometry("940x520")
        self.root.minsize(780, 430)
        self.root.configure(bg=BG)
        self.closed = False
        self.running = False
        self.latest_payload = {}
        self.last_voice = "No classification is available yet."
        self.last_capture = np.array([], dtype=float)
        self.last_fs = 0.0
        self.animation_id = None
        self.capture_number = 0

        self.voice_status = tk.StringVar(value="Voice initializing…")
        self.status = tk.StringVar(value="Fix the electrodes, then press START.")
        self.shockable_val = tk.StringVar(value="--")
        self.master_decision_val = tk.StringVar(value="-- SYSTEM STANDBY --")
        self.sub_diagnostics_val = tk.StringVar(value="Tier 2: -- | Tier 3: -- | AFib: --")
        self.filter_info = tk.StringVar(value="Stored waveform: --")

        self.voice = VoiceWorker(self.set_voice_status)
        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.exit_app)
        self.root.after(300, self.check_audio)

    def set_voice_status(self, text):
        if self.closed:
            return
        try:
            self.root.after(0, lambda: self.voice_status.set(text))
        except Exception:
            pass

    def build_ui(self):
        outer = tk.Frame(self.root, bg=BG, padx=12, pady=6)
        outer.pack(fill="both", expand=True)

        # Header Row
        header = tk.Frame(outer, bg=BG)
        header.pack(fill="x")
        tk.Label(header, text="ECG ARRHHYTHMIA EDGE AI", bg=BG, fg=FG,
                 font=("DejaVu Sans", 13, "bold")).pack(side="left")
        tk.Label(header, textvariable=self.voice_status, bg=BG, fg=MUTED,
                 font=("DejaVu Sans", 8)).pack(side="right")

        # Master Triage & Decision Row
        resultrow = tk.Frame(outer, bg=BG)
        resultrow.pack(fill="x", pady=(5, 2))

        self.shock_label = tk.Label(
            resultrow, textvariable=self.shockable_val, bg=LIGHT_AMBER, fg=AMBER,
            font=("DejaVu Sans", 12, "bold"), padx=12, pady=5, width=20, anchor="center"
        )
        self.shock_label.pack(side="left")

        self.decision_label = tk.Label(
            resultrow, textvariable=self.master_decision_val, bg="#f0f4f8", fg=FG,
            font=("DejaVu Sans", 12, "bold"), padx=14, pady=5, anchor="w"
        )
        self.decision_label.pack(side="left", fill="x", expand=True, padx=(8, 0))

        # Secondary Tier Diagnostics (Model 2, Model 3 VT/VF, AFib Branch)
        subrow = tk.Frame(outer, bg=BG)
        subrow.pack(fill="x", pady=(2, 4))
        self.sub_label = tk.Label(
            subrow, textvariable=self.sub_diagnostics_val, bg=BG, fg=MUTED,
            font=("DejaVu Sans", 8, "bold"), anchor="w"
        )
        self.sub_label.pack(fill="x")

        # Matplotlib Plot
        self.fig = Figure(figsize=(9.3, 2.7), dpi=80, facecolor=BG)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#ffffff")
        self.ax.set_title("Stored 8-second ECG capture (Unit-mV Calibrated)", color=FG, fontsize=8)
        self.ax.set_xlabel("Time (seconds)", color=FG, fontsize=8)
        self.ax.set_ylabel("Amplitude (mV)", color=FG, fontsize=8)
        self.ax.tick_params(colors=FG, labelsize=7)
        self.ax.grid(True, color="#cfd8df", linewidth=0.5, alpha=0.8)
        self.ax.set_xlim(0, CAPTURE_SECONDS)
        self.ax.set_ylim(-2.0, 2.0)
        self.ax.set_xticks(np.arange(0, CAPTURE_SECONDS + 0.1, 1.0))
        for spine in self.ax.spines.values():
            spine.set_color(BORDER)
        self.line, = self.ax.plot([], [], color="#b71c1c", linewidth=0.85)
        self.fig.tight_layout(pad=1.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=outer)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=(2, 3))

        tk.Label(outer, textvariable=self.status, bg=BG, fg=FG,
                 font=("DejaVu Sans", 8), anchor="w").pack(fill="x")
        tk.Label(outer, textvariable=self.filter_info, bg=BG, fg=MUTED,
                 font=("DejaVu Sans", 7), anchor="w").pack(fill="x", pady=(1, 3))

        # Controls
        controls = tk.Frame(outer, bg=BG)
        controls.pack(fill="x")
        self.start_button = tk.Button(
            controls, text="START", command=self.start, bg="#f2f7f3", fg=GREEN,
            font=("DejaVu Sans", 10, "bold"), padx=12, pady=5
        )
        self.start_button.pack(side="left", expand=True, fill="x", padx=3)
        self.stop_button = tk.Button(
            controls, text="STOP", command=self.stop, bg="#fff1f0", fg=RED,
            font=("DejaVu Sans", 10, "bold"), padx=12, pady=5
        )
        self.stop_button.pack(side="left", expand=True, fill="x", padx=3)
        self.repeat_button = tk.Button(
            controls, text="REPEAT AUDIO", command=self.repeat_audio, bg="#f3f6f9", fg=FG,
            font=("DejaVu Sans", 10, "bold"), padx=12, pady=5
        )
        self.repeat_button.pack(side="left", expand=True, fill="x", padx=3)
        self.exit_button = tk.Button(
            controls, text="EXIT", command=self.exit_app, bg="#fff1f0", fg=RED,
            font=("DejaVu Sans", 10, "bold"), padx=12, pady=5
        )
        self.exit_button.pack(side="left", expand=True, fill="x", padx=3)

    def check_audio(self):
        if self.closed:
            return
        has_tools = shutil.which("espeak-ng") and shutil.which("paplay")
        self.voice_status.set("Audio ready" if has_tools else "Audio tools missing")
        if has_tools:
            self.voice.say("Please fix the ECG electrodes correctly, then press Start.")
        self.root.after(3000, self._audio_status_only)

    def _audio_status_only(self):
        if not self.closed:
            self.voice_status.set("Voice prompts enabled" if shutil.which("espeak-ng") and shutil.which("paplay") else "Audio tools missing")

    def fetch_payload(self):
        req = Request(API_URL, headers={"Accept": "application/json", "Cache-Control": "no-cache"})
        with urlopen(req, timeout=2.0) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
        if not isinstance(data, dict):
            raise ValueError("Backend returned non-object JSON")
        return data

    def start(self):
        if self.running or self.closed:
            return

        try:
            req = Request(
                "http://127.0.0.1:5000/control/start",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=5) as response:
                response.read()
        except Exception as exc:
            self.status.set(f"Backend start failed: {exc}")
            return

        self.running = True
        self.start_button.configure(state="disabled")
        self.status.set("Capturing ECG — collecting 8-second window across full 3-model cascade...")
        threading.Thread(
            target=self.capture_loop,
            daemon=True,
            name="ECGCaptureLoop",
        ).start()

    def capture_one_window(self):
        chunks = []
        fs = 0.0
        newest_payload = {}
        last_total = None
        start_time = time.monotonic()
        first_wave = np.array([], dtype=float)

        while self.running and not self.closed and time.monotonic() - start_time < CAPTURE_SECONDS:
            try:
                payload = self.fetch_payload()
                newest_payload = payload
                state = get_state(payload)
                fs = float(state.get("hardware_sps", payload.get("fs", fs or 360)) or fs or 360)
                wave = waveform_from_payload(payload)
                total_value = state.get("total_samples")
                total = int(total_value) if total_value is not None else None

                if wave.size and not first_wave.size:
                    first_wave = wave.copy()

                if wave.size and total is not None:
                    if last_total is not None and total > last_total:
                        delta = total - last_total
                        chunks.append(wave[-min(delta, wave.size):].copy())
                    elif last_total is None:
                        take = min(wave.size, max(1, int(fs * POLL_SECONDS)))
                        chunks.append(wave[-take:].copy())
                    last_total = total
            except Exception:
                pass
            time.sleep(POLL_SECONDS)

        if not self.running or self.closed:
            return None

        target = max(1, int(round((fs or 360.0) * CAPTURE_SECONDS)))
        if chunks:
            signal = np.concatenate(chunks)
            signal = signal[-target:]
        else:
            signal = waveform_from_payload(newest_payload)
            signal = signal[-target:]

        if signal.size < max(8, int(0.5 * target)) and first_wave.size:
            signal = first_wave[-target:]
        return newest_payload, signal, fs or 360.0

    def capture_loop(self):
        while self.running and not self.closed:
            cycle_start = time.monotonic()
            try:
                captured = self.capture_one_window()
                if captured is None or not self.running or self.closed:
                    break
                payload, signal, fs = captured
                telemetry = classify_telemetry(payload)
                cleaned, filter_note, qrs_count = clean_ecg(signal, fs)
                self.capture_number += 1
                self.root.after(
                    0, self.capture_done, self.capture_number, payload, cleaned,
                    fs, telemetry, filter_note, qrs_count
                )
            except Exception as exc:
                if self.running and not self.closed:
                    self.root.after(0, self.capture_error, str(exc))
                    time.sleep(0.5)
            if time.monotonic() - cycle_start < 0.01:
                time.sleep(0.01)

    def capture_done(self, number, payload, signal, fs, telemetry, filter_note, qrs_count):
        if self.closed:
            return
        self.latest_payload = payload
        self.last_capture = np.asarray(signal, dtype=float)
        self.last_fs = fs
        self.last_voice = telemetry["voice"]

        # 1. Update Labels
        self.shockable_val.set(telemetry["shockable"])
        self.master_decision_val.set(telemetry["decision"])
        self.sub_diagnostics_val.set(
            f"Tier 2: {telemetry['tier2_rhythm']}  |  Tier 3: {telemetry['tier3_subtype']}  |  {telemetry['afib_summary']}"
        )

        # 2. Color-code Shockable Badge
        if telemetry["shockable"] == "SHOCKABLE":
            self.shock_label.configure(bg=LIGHT_RED, fg=RED)
        elif telemetry["shockable"] == "NOT SHOCKABLE":
            self.shock_label.configure(bg=LIGHT_GREEN, fg=GREEN)
        else:
            self.shock_label.configure(bg=LIGHT_AMBER, fg=AMBER)

        # 3. Color-code Master Decision Badge
        triage = telemetry["triage"]
        if "CRITICAL" in triage or "FIBRILLATION" in telemetry["decision"]:
            self.decision_label.configure(bg=LIGHT_RED, fg=RED)
        elif "URGENT" in triage or telemetry.get("afib_alert"):
            self.decision_label.configure(bg=LIGHT_AMBER, fg=AMBER)
        elif "NORMAL" in triage or "NORMAL" in telemetry["decision"]:
            self.decision_label.configure(bg=LIGHT_GREEN, fg=GREEN)
        else:
            self.decision_label.configure(bg="#f0f4f8", fg=FG)

        q = get_state(payload).get("signal_quality", "--")
        self.status.set(f"Capture {number}: 8-second window | Signal quality: {q} | Samples: {len(self.last_capture)}")
        qrs_text = f" | Pan–Tompkins-style QRS estimate: {qrs_count}" if qrs_count is not None else ""
        self.filter_info.set(f"Processing: {filter_note}{qrs_text} | Fixed scale: -2 to +2 mV")

        # Telemetry terminal log
        print(f"\n=== CAPTURE {number}: FILTERED ECG TELEMETRY ===", flush=True)
        print(f"SHOCKABLE: {telemetry['shockable']} | DIAGNOSIS: {telemetry['decision']}", flush=True)
        print(f"DETAILS: {self.sub_diagnostics_val.get()}", flush=True)
        print("=== END CAPTURE ===", flush=True)

        self.animate_capture(number, self.last_capture.copy(), fs)

    def animate_capture(self, number, signal, fs):
        if self.closed:
            return
        if self.animation_id is not None:
            try:
                self.root.after_cancel(self.animation_id)
            except Exception:
                pass
            self.animation_id = None

        self.line.set_data([], [])
        self.ax.set_xlim(0, CAPTURE_SECONDS)
        self.ax.set_ylim(-2.0, 2.0)
        self.canvas.draw_idle()

        if signal.size == 0:
            self.status.set(f"Capture {number}: no waveform samples received.")
            return

        total_points = len(signal)
        frame_count = max(1, int(ANIMATION_SECONDS * 30))
        frame_index = 0
        start_clock = time.monotonic()
        t = np.arange(total_points) / fs if fs > 0 else np.linspace(0, CAPTURE_SECONDS, total_points)

        def frame():
            nonlocal frame_index
            if self.closed:
                return
            elapsed = time.monotonic() - start_clock
            progress = min(1.0, elapsed / ANIMATION_SECONDS)
            end = max(1, min(total_points, int(progress * total_points)))
            self.line.set_data(t[:end], signal[:end])
            self.canvas.draw_idle()
            frame_index += 1
            if progress < 1.0 and self.running:
                self.animation_id = self.root.after(max(1, int(1000 / 30)), frame)
            else:
                self.animation_id = None

        self.animation_id = self.root.after(1, frame)

    def capture_error(self, error):
        if self.closed:
            return
        self.status.set("Capture error; retrying while START remains active: " + error)
        print("ECG capture error:", error, flush=True)

    def stop(self):
        self.running = False
        self.start_button.configure(state="normal")
        try:
            req = Request(
                "http://127.0.0.1:5000/control/stop",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=3) as response:
                response.read()
        except Exception as exc:
            print(f"Backend stop request failed: {exc}", flush=True)

        self.status.set("Acquisition stopped. Press START to resume.")
        if self.animation_id is not None:
            try:
                self.root.after_cancel(self.animation_id)
            except Exception:
                pass
            self.animation_id = None

    def repeat_audio(self):
        self.voice.say(self.last_voice)

    def exit_app(self):
        if self.closed:
            return
        self.closed = True
        self.running = False
        try:
            self.voice.close()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    ECGApp(root)
    root.mainloop()

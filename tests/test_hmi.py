"""HMI behaviour (needs tkinter + a display; use xvfb-run on a headless box)."""
import os
import time

import pytest

tk = pytest.importorskip("tkinter")
pytestmark = pytest.mark.skipif(not (os.environ.get("DISPLAY") or os.name == "nt"), reason="no display")

from ecg_edge.acquisition import MockADS1292R


def _pump(root, until, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout and not until():
        root.update(); time.sleep(0.01)
    assert until(), "timed out"


def test_start_cycle_idle_start_again(cascade):
    from ecg_edge.hmi import ECGApp, IDLE
    root = tk.Tk()
    adc = MockADS1292R("VF", realtime=False)
    app = ECGApp(root, adc, cascade, voice=None)
    root.update()
    assert app.state == IDLE and adc.start_calls == 0
    app.start_btn.invoke()
    _pump(root, lambda: app.cycles == 1 and app.state == IDLE)
    assert app.banner_var.get() == "VF/VT SHOCKABLE"
    assert len(app.line.get_xdata()) == 1800
    assert str(app.start_btn["state"]) == "normal"
    n = adc.samples_read
    for _ in range(30):
        root.update(); time.sleep(0.01)
    assert adc.samples_read == n and app.cycles == 1
    app.start_btn.invoke()
    _pump(root, lambda: app.cycles == 2 and app.state == IDLE)
    assert adc.start_calls == 2
    app.exit_app()


def test_start_ignored_while_busy_and_cancel(cascade):
    from ecg_edge.hmi import ECGApp, ACQUIRING, IDLE
    root = tk.Tk()
    adc = MockADS1292R("NSR", realtime=True)
    app = ECGApp(root, adc, cascade, voice=None)
    app.start(); root.update()
    assert app.state == ACQUIRING and str(app.start_btn["state"]) == "disabled"
    app.start()
    time.sleep(0.5); root.update()
    app.cancel_btn.invoke()
    _pump(root, lambda: app.state == IDLE)
    assert app.banner_var.get() == "CANCELLED" and adc.start_calls == 1 and not adc.running
    app.exit_app()


def test_poor_signal_banner(cascade):
    from ecg_edge.hmi import ECGApp
    root = tk.Tk()
    app = ECGApp(root, MockADS1292R("LEADOFF", realtime=False), cascade, voice=None)
    app.start()
    _pump(root, lambda: app.cycles == 1)
    assert "POOR SIGNAL" in app.banner_var.get()
    app.exit_app()

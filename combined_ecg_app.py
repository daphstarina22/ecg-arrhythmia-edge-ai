#!/usr/bin/env python3
"""
Root entrypoint for Unified ECG Arrhythmia Edge AI Monitor.
Runs deploy/combined_ecg_app.py.
"""
import sys
from pathlib import Path

# Forward execution to deploy/combined_ecg_app.py
deploy_dir = Path(__file__).resolve().parent / "deploy"
sys.path.insert(0, str(deploy_dir))

from deploy.combined_ecg_app import app, acquisition_and_inference_worker
import argparse
import threading
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified ECG Arrhythmia Edge AI Monitor")
    parser.add_argument("--mock", action="store_true", help="Run in offline simulation mode without physical SPI")
    parser.add_argument("--port", type=int, default=5000, help="Web dashboard port (default: 5000)")
    parser.add_argument("--interval", type=float, default=10.0, help="Inference interval in seconds (default: 10.0)")
    args = parser.parse_args()

    worker_t = threading.Thread(
        target=acquisition_and_inference_worker,
        kwargs={"mock": args.mock, "interval_sec": args.interval},
        daemon=True
    )
    worker_t.start()

    time.sleep(1.0)
    print(f"\n===========================================================================")
    print(f"  ECG ARRHYTHMIA EDGE AI DASHBOARD READY")
    print(f"  Open in browser: http://0.0.0.0:{args.port}")
    print(f"  (Or from your laptop: http://<raspberry-pi-ip>:{args.port})")
    print(f"===========================================================================\n")
    try:
        app.run(host="0.0.0.0", port=args.port, debug=False)
    except OSError as err:
        if "Address already in use" in str(err) or getattr(err, "errno", None) in (98, 10048):
            print(f"\n[PORT ERROR] Port {args.port} is already in use by a background process!")
            print(f"  To stop the existing process holding port {args.port}, run on the Pi:")
            print(f"      sudo fuser -k {args.port}/tcp")
            print(f"  Or start the dashboard on a different port, e.g.:")
            print(f"      python3 combined_ecg_app.py --port {args.port + 1}\n")
        else:
            raise

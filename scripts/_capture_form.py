"""Capture screenshot of running form page using headless Chrome via subprocess.

Run from project root with the .venv python:
    .venv/bin/python scripts/_capture_form.py
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Ignore SIGINT in this process so a stray Ctrl-C from the controlling
# terminal doesn't abort the screenshot wait loop.
signal.signal(signal.SIGINT, signal.SIG_IGN)

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
DEFAULT_URL = "http://localhost:8088/"
DEFAULT_SHOT = Path(__file__).resolve().parents[1] / "screenshots" / "task6_form.png"


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    shot = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else DEFAULT_SHOT
    SHOT = shot
    SHOT.parent.mkdir(parents=True, exist_ok=True)
    if SHOT.exists():
        SHOT.unlink()

    with tempfile.TemporaryDirectory(prefix="chrome_") as tmpdir:
        log_path = Path("/tmp/chrome.log")
        log_fh = log_path.open("w")
        cmd = [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-sandbox",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            "--disable-component-update",
            "--disable-background-networking",
            f"--user-data-dir={tmpdir}",
            "--window-size=1280,1400",
            "--virtual-time-budget=12000",
            f"--screenshot={SHOT}",
            url,
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.time() + 45
        while time.time() < deadline:
            if SHOT.exists() and SHOT.stat().st_size > 0:
                break
            if proc.poll() is not None:
                break
            time.sleep(1)
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)
        log_fh.close()
    if SHOT.exists() and SHOT.stat().st_size > 0:
        size = SHOT.stat().st_size
        print(f"OK screenshot saved: {SHOT}  ({size} bytes)")
        return 0
    print("FAIL: screenshot was not produced")
    print("--- chrome.log tail ---")
    if log_path.exists():
        sys.stdout.write(log_path.read_text()[-2000:])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Screenshots of the running dashboard, one PNG per page, taken with headless Google Chrome over the DevTools protocol (dashboard S5 evidence).

Usage (the dashboard must be running, see ``dashboard/app.py``)::

    .venv/bin/python scripts/dashboard_screenshots.py --url http://localhost:8765 --out tracker/features/2026-10-22-dashboard/artifacts/screens

Uses only ``websockets`` (a Streamlit dependency) and the Chrome binary; it clicks the sidebar page radio, waits for the page to settle and saves a full-page PNG.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import time
import urllib.request
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PAGES = ("Overview", "Rollouts", "Episodes", "Clusters", "Bad vs harmful")


async def run(url: str, out: Path, port: int, width: int, wait: float, pages) -> list[Path]:
    import websockets

    ws_url = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://localhost:{port}/json/new?{url}", method="PUT"), timeout=10).read())["webSocketDebuggerUrl"]
    saved = []
    async with websockets.connect(ws_url, max_size=64 * 1024 * 1024) as ws:
        n = 0

        async def call(method, **params):
            nonlocal n
            n += 1
            await ws.send(json.dumps({"id": n, "method": method, "params": params}))
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 60))
                if msg.get("id") == n:
                    return msg.get("result", {})

        await call("Emulation.setDeviceMetricsOverride", width=width, height=1000, deviceScaleFactor=1, mobile=False)
        await call("Page.enable")
        for page in pages:
            await call("Runtime.evaluate", expression=f"""(() => {{ const l=[...document.querySelectorAll('label')].find(x=>x.innerText.trim()==={json.dumps(page)}); if(l) l.click(); return !!l; }})()""")
            await asyncio.sleep(wait)
            h = (await call("Runtime.evaluate", expression="Math.max(document.body.scrollHeight, document.querySelector('section.main, [data-testid=stMain]')?.scrollHeight||0)", returnByValue=True))["result"]["value"]
            await call("Emulation.setDeviceMetricsOverride", width=width, height=min(int(h) + 80, 9000), deviceScaleFactor=1, mobile=False)
            await asyncio.sleep(1.5)
            png = (await call("Page.captureScreenshot", format="png", captureBeyondViewport=True))["data"]
            p = out / f"{page.lower().replace(' ', '_')}.png"
            p.write_bytes(base64.b64decode(png))
            saved.append(p)
            print(f"{page}: {p} ({p.stat().st_size} bytes, page height {h})", flush=True)
    return saved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://localhost:8765")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--port", type=int, default=9333)
    ap.add_argument("--width", type=int, default=1500)
    ap.add_argument("--wait", type=float, default=9.0)
    ap.add_argument("--pages", nargs="+", default=list(PAGES))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", f"--remote-debugging-port={a.port}", "--remote-allow-origins=*", "--no-first-run", "--user-data-dir=" + str(a.out / ".chrome-profile"), "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(40):
            try:
                urllib.request.urlopen(f"http://localhost:{a.port}/json/version", timeout=1)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        asyncio.run(run(a.url, a.out, a.port, a.width, a.wait, a.pages))
    finally:
        proc.terminate()
        import shutil

        shutil.rmtree(a.out / ".chrome-profile", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

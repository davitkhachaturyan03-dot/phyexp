#!/usr/bin/env python
"""Look at the running dashboard with headless Google Chrome over the DevTools protocol: full-page screenshots and the rendering measurements of AC5.10 (dashboard S5).

Usage (the dashboard must be running, see ``dashboard/app.py``)::

    .venv/bin/python scripts/dashboard_screenshots.py --url http://localhost:8765 --out tracker/features/2026-10-22-dashboard/artifacts/screens [--check]

Uses only ``websockets`` (a Streamlit dependency) and the Chrome binary. ``--check`` also measures, at a 1400 px window: every Altair chart's plot area (SVG ``path.background``,
must be >= 200 px tall; per chart, the largest background rectangle), that the sidebar holds no auto-generated page links, that no ``st.metric`` value is truncated, and that clicking a row of the Rollouts table selects that
rollout (a real mouse click on the grid's row marker). The measurements are returned by ``measure()`` and printed as JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PAGES = ("Overview", "Rollouts", "Episodes", "Clusters", "Bad vs harmful")


class Cdp:
    """A minimal DevTools client: one page, ``call``, ``js`` and ``shot``."""

    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def call(self, method, **params):
        self.n += 1
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(await asyncio.wait_for(self.ws.recv(), 60))
            if msg.get("id") == self.n:
                return msg.get("result", {})

    async def js(self, expr):
        return (await self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)).get("result", {}).get("value")

    async def shot(self, path: Path):
        png = (await self.call("Page.captureScreenshot", format="png", captureBeyondViewport=True))["data"]
        path.write_bytes(base64.b64decode(png))

    async def resize(self, width, height):
        await self.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=False)

    async def click(self, x, y):
        for t in ("mouseMoved", "mousePressed", "mouseReleased"):
            await self.call("Input.dispatchMouseEvent", type=t, x=x, y=y, button="left", clickCount=1)
            await asyncio.sleep(0.05)

    async def goto_page(self, page: str, wait: float):
        await self.js(f"""(() => {{ const l=[...document.querySelectorAll('label')].find(x=>x.innerText.trim()==={json.dumps(page)}); if(l) l.click(); return !!l; }})()""")
        await asyncio.sleep(wait)

    async def height(self):
        return int(await self.js("Math.max(document.body.scrollHeight, document.querySelector('[data-testid=stMain]')?.scrollHeight||0)") or 1000)


class Chrome:
    """Headless Chrome with a DevTools port; ``session(url)`` yields a ``Cdp`` on a fresh tab."""

    def __init__(self, port: int, profile: Path):
        self.port, self.profile = port, profile
        self.proc = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", f"--remote-debugging-port={port}", "--remote-allow-origins=*", "--no-first-run", f"--user-data-dir={profile}", "about:blank"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://localhost:{port}/json/version", timeout=1)
                return
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        raise RuntimeError("Chrome did not start")

    def ws_url(self, url: str) -> str:
        return json.loads(urllib.request.urlopen(urllib.request.Request(f"http://localhost:{self.port}/json/new?{url}", method="PUT"), timeout=10).read())["webSocketDebuggerUrl"]

    def close(self):
        self.proc.terminate()
        shutil.rmtree(self.profile, ignore_errors=True)


async def measure(url: str, port: int, profile: Path, width: int = 1400, wait: float = 10.0, rollout_ids: list[str] | None = None, row: int = 4) -> dict:
    """The AC5.10 measurements (module docstring) on every page; ``rollout_ids`` are the Rollouts table's ids in display order (for the click check)."""
    import websockets

    ch = Chrome(port, profile)
    out: dict = {"width": width, "pages": {}}
    try:
        async with websockets.connect(ch.ws_url(url), max_size=64 * 1024 * 1024) as ws:
            c = Cdp(ws)
            await c.resize(width, 1100)
            await c.call("Page.enable")
            await asyncio.sleep(wait)
            out["sidebar_nav_links"] = await c.js("document.querySelectorAll('[data-testid=stSidebarNav] a').length")
            for page in PAGES:
                await c.goto_page(page, wait)
                m = await c.js("""(() => ({
                    plots: [...document.querySelectorAll('.vega-embed svg')].map(svg => { let best = [0, 0]; svg.querySelectorAll('path.background').forEach(p => { const b = p.getBBox(); if (b.width * b.height > best[0] * best[1]) best = [Math.round(b.width), Math.round(b.height)]; }); return best; }),
                    metrics: [...document.querySelectorAll('[data-testid=stMetricValue]')].map(e => ({text: e.innerText, truncated: e.scrollWidth > e.clientWidth + 1})),
                    images: [...document.querySelectorAll('[data-testid=stImage] img')].map(i => [i.naturalWidth, i.naturalHeight, Math.round(i.getBoundingClientRect().height)]),
                    exceptions: document.querySelectorAll('[data-testid=stException]').length}))()""")
                out["pages"][page] = m
            if rollout_ids:
                await c.goto_page("Rollouts", wait)
                rect = await c.js("""(() => { const e = document.querySelector('[data-testid=stDataFrame]'); const r = e.getBoundingClientRect(); return {x: r.left, y: r.top}; })()""")
                await c.js("window.scrollTo(0, 0)")
                before = await c.js("""(() => { const l=[...document.querySelectorAll('[data-testid=stWidgetLabel]')].find(x=>x.innerText.trim()==='Selected rollout'); return l.closest('[data-testid=stSelectbox]').querySelector('input[role=combobox]').value; })()""")
                await c.click(rect["x"] + 14, rect["y"] + 35 + row * 35 + 17)
                await asyncio.sleep(5)
                after = await c.js("""(() => { const l=[...document.querySelectorAll('[data-testid=stWidgetLabel]')].find(x=>x.innerText.trim()==='Selected rollout'); return l.closest('[data-testid=stSelectbox]').querySelector('input[role=combobox]').value; })()""")
                norm = lambda x: (x or "").replace("_", " ")  # the page shows names with spaces
                out["row_click"] = {"row": row, "expected": rollout_ids[row], "before": before, "after": after, "selected_after_click": norm(rollout_ids[row]) in norm(after)}
    finally:
        ch.close()
    return out


async def screenshots(url: str, out: Path, port: int, width: int, wait: float, pages) -> list[Path]:
    import websockets

    ch = Chrome(port, out / ".chrome-profile")
    saved = []
    try:
        async with websockets.connect(ch.ws_url(url), max_size=64 * 1024 * 1024) as ws:
            c = Cdp(ws)
            await c.resize(width, 1000)
            await c.call("Page.enable")
            await asyncio.sleep(wait)
            for page in pages:
                await c.goto_page(page, wait)
                h = await c.height()
                await c.resize(width, min(h + 80, 9000))
                await asyncio.sleep(1.5)
                p = out / f"{page.lower().replace(' ', '_')}.png"
                await c.shot(p)
                saved.append(p)
                print(f"{page}: {p} ({p.stat().st_size} bytes, page height {h})", flush=True)
    finally:
        ch.close()
    return saved


def main() -> int:
    """Command line: ``--url``, ``--out``, optional ``--check`` (module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://localhost:8765")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--port", type=int, default=9333)
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--wait", type=float, default=10.0)
    ap.add_argument("--pages", nargs="+", default=list(PAGES))
    ap.add_argument("--check", action="store_true", help="also print the AC5.10 measurements")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    asyncio.run(screenshots(a.url, a.out, a.port, a.width, a.wait, a.pages))
    if a.check:
        print(json.dumps(asyncio.run(measure(a.url, a.port + 1, a.out / ".chrome-profile2", a.width, a.wait)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

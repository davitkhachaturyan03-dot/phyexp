"""Evaluation-layer S2 (a): list PhAIL's public dataset and fetch its per-episode metadata sidecars.

Grounding: PoC §3.5 B and "Practical caveats" (p. 8): "Confirm which parts of the dataset, including
the fine-tuning set, are downloadable and under what license." PoC §3.1 (p. 6): PhAIL publishes "~990
evaluation episodes including 396 same-fixture human-teleoperation reference rollouts".

Source: the public bucket ``positronic-public`` on Nebius S3, anonymous HTTPS, endpoint
``https://storage.eu-north1.nebius.cloud`` (quoted: positronic ``cfg/ds/__init__.py`` at 498ef5e,
``PUBLIC = Profile(... endpoint='https://storage.eu-north1.nebius.cloud', public=True ...)``), prefix
``phail/v1.0/dataset/``. Plain S3 ListObjectsV2 + GET over the standard library; no pos3/boto3 needed.
The phail-paper README's recipe fetches the same thing with ``exclude=['*.parquet', '*.mp4']``.

    .venv/bin/python scripts/phail_fetch.py --list      # listing only -> data/phail/listing.json + summary
    .venv/bin/python scripts/phail_fetch.py --fetch     # sidecars (not *.parquet / *.mp4) -> data/phail/dataset/
    .venv/bin/python scripts/phail_fetch.py --probe s3://inference/phail_act_groot/   # access check (HTTP status)

Owner approval (brief.md, 2026-10-02): the metadata sidecars (~180 MB) are approved; the 13.95 GB
fine-tuning set is NOT (it is only listed and counted here, never downloaded).
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "https://storage.eu-north1.nebius.cloud"
BUCKET = "positronic-public"
PREFIX = "phail/v1.0/dataset/"
DATA = REPO_ROOT / "data" / "phail"
ART = REPO_ROOT / "tracker" / "features" / "2026-10-01-evaluation-layer" / "artifacts"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
# The python.org framework build has no system CA bundle configured; verify against certifi's (installed).
try:
    import certifi

    _CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # pragma: no cover
    _CTX = ssl.create_default_context()


def _open(url: str, timeout: int):
    return urllib.request.urlopen(url, timeout=timeout, context=_CTX)
EXCLUDE = (".parquet", ".mp4")


def list_objects(bucket: str, prefix: str) -> list[dict]:
    """Every object under ``prefix`` (ListObjectsV2, paginated): ``[{"key", "size", "etag", "last_modified"}]``."""
    out, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        with _open(f"{ENDPOINT}/{bucket}?{urllib.parse.urlencode(q)}", 60) as r:
            root = ET.fromstring(r.read())
        for c in root.findall("s3:Contents", NS):
            out.append({"key": c.find("s3:Key", NS).text, "size": int(c.find("s3:Size", NS).text),
                        "etag": c.find("s3:ETag", NS).text.strip('"'), "last_modified": c.find("s3:LastModified", NS).text})
        if root.findtext("s3:IsTruncated", namespaces=NS) != "true":
            return out
        token = root.findtext("s3:NextContinuationToken", namespaces=NS)


def summarize(objs: list[dict]) -> dict:
    """Per top-level directory under the prefix: episodes (dirs with static.json), files, bytes, by extension."""
    top = defaultdict(lambda: {"files": 0, "bytes": 0, "episodes": 0, "by_ext": Counter(), "bytes_by_ext": Counter()})
    for o in objs:
        rel = o["key"][len(PREFIX):]
        d = rel.split("/", 1)[0] if "/" in rel else "(root)"
        ext = Path(rel).suffix or "(none)"
        t = top[d]
        t["files"] += 1
        t["bytes"] += o["size"]
        t["by_ext"][ext] += 1
        t["bytes_by_ext"][ext] += o["size"]
        if rel.endswith("/static.json"):
            t["episodes"] += 1
    return {d: {**v, "by_ext": dict(v["by_ext"]), "bytes_by_ext": dict(v["bytes_by_ext"])} for d, v in sorted(top.items())}


def _get(o: dict) -> int:
    dest = DATA / "dataset" / o["key"][len(PREFIX):]
    if dest.is_file() and dest.stat().st_size == o["size"]:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{ENDPOINT}/{BUCKET}/{urllib.parse.quote(o['key'])}"
    for attempt in range(4):  # the endpoint occasionally drops a connection (RemoteDisconnected, measured)
        try:
            with _open(url, 120) as r:
                data = r.read()
            break
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    if len(data) != o["size"]:
        raise IOError(f"{o['key']}: got {len(data)} bytes, listed {o['size']}")
    dest.write_bytes(data)
    return len(data)


def probe(s3_url: str) -> dict:
    """HTTP status of an anonymous ListObjectsV2 on an s3:// prefix (bucket = first path part)."""
    bucket, _, prefix = s3_url.split("://", 1)[1].partition("/")
    url = f"{ENDPOINT}/{bucket}?{urllib.parse.urlencode({'list-type': '2', 'prefix': prefix, 'max-keys': '5'})}"
    try:
        with _open(url, 60) as r:
            return {"url": s3_url, "status": r.status}
    except urllib.error.HTTPError as e:
        return {"url": s3_url, "status": e.code, "reason": e.reason}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    p.add_argument("--fetch", action="store_true")
    p.add_argument("--probe", nargs="*", default=None)
    a = p.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if a.list or a.fetch:
        objs = list_objects(BUCKET, PREFIX)
        (DATA / "listing.json").write_text(json.dumps(objs))
        summ = summarize(objs)
        total = {"files": len(objs), "bytes": sum(o["size"] for o in objs)}
        side = [o for o in objs if not o["key"].endswith(EXCLUDE) and not o["key"].endswith("/")]  # skip dir markers
        total["sidecar_files"] = len(side)
        total["sidecar_bytes"] = sum(o["size"] for o in side)
        out = {"endpoint": ENDPOINT, "bucket": BUCKET, "prefix": PREFIX, "total": total, "by_dir": summ}
        (ART / "s2_phail_listing_summary.json").write_text(json.dumps(out, indent=2))
        for d, v in summ.items():
            print(f"{d:>14}: {v['episodes']:4d} episodes (static.json), {v['files']:6d} files, {v['bytes'] / 1e9:7.2f} GB; "
                  f"by ext {v['by_ext']}")
        print(f"total {total['files']} files {total['bytes'] / 1e9:.2f} GB; sidecars (no .parquet/.mp4) "
              f"{total['sidecar_files']} files {total['sidecar_bytes'] / 1e6:.1f} MB")
        if a.fetch:
            with ThreadPoolExecutor(16) as ex:
                got = list(ex.map(_get, side))
            print(f"fetched {sum(1 for g in got if g)} files, {sum(got) / 1e6:.1f} MB new into {DATA.relative_to(REPO_ROOT)}/dataset "
                  f"({len(side)} sidecar files present)")
    if a.probe is not None:
        urls = a.probe or [f"s3://{BUCKET}/{PREFIX}", "s3://inference/phail_act_groot/", "s3://inference/phail_final/",
                           "s3://inference/spoons_ablation/"]
        res = [probe(u) for u in urls]
        for r in res:
            print(f"probe {r['url']}: HTTP {r['status']} {r.get('reason', '')}")
        (ART / "s2_phail_access_probe.json").write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

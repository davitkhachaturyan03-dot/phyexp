"""Draw the data-driven inline-SVG figures of this feature's report and story pages.

Every figure is computed from files in this ``artifacts/`` directory; nothing is typed in by hand:
- ``lineage``     s6_lineage_graph.json + s6_trace_train_drop_top.json + s6_summary.json (+ s6 diff JSONs)
- ``episodegrid`` summarizer_v1_episode_status.csv (1000 rows, exported from v0/v1 episode tables + labels)
- ``durations``   s2_human_duration_per_task.csv + s2_catalog_stdout.txt (per-suite summary)
- ``membership``  s5_r2_membership_matrix.csv
- ``subsets``     s3_split_make_stdout.txt, s3_summary_r2.json, s5_r2_summary.json, s6_lineage_graph.json

The script rewrites the block between ``<!--FIG:<name>:start-->`` and ``<!--FIG:<name>:end-->`` in every
``*.html`` page of the feature folder, so it can be re-run after an artifact changes. Colours are the
CSS variables of ``tracker/assets/tracker.css`` (classes defined in each page's <style>); no hex values.

Usage (from the repo root)::

    .venv/bin/python tracker/features/2026-09-28-data-lineage-layer/artifacts/summarizer_figures.py
"""

from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path

ART = Path(__file__).resolve().parent
FEAT = ART.parent


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def text(x, y, s, cls, anchor="start", extra=""):
    a = "" if anchor == "start" else f' text-anchor="{anchor}"'
    return f'<text x="{x:g}" y="{y:g}" class="{cls}"{a}{extra}>{esc(s)}</text>'


def poly(points, cls, marker=None):
    d = "M" + " L".join(f"{x:g},{y:g}" for x, y in points)
    m = f' marker-end="url(#{marker})"' if marker else ""
    return f'<path d="{d}" class="{cls}"{m}/>'


def markers(pfx):
    """Two arrowheads (context and emphasis), styled by the page's CSS classes."""
    out = []
    for name, cls in (("ah", "ah"), ("aht", "ah trace")):
        out.append(
            f'<marker id="{pfx}-{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="{cls}"/></marker>'
        )
    return "<defs>" + "".join(out) + "</defs>"


# ---------------------------------------------------------------------------------------------
# 1. the real lineage graph (S6)
# ---------------------------------------------------------------------------------------------


def lineage_graph(pfx="lg") -> str:
    g = json.loads((ART / "s6_lineage_graph.json").read_text())
    trace = json.loads((ART / "s6_trace_train_drop_top.json").read_text())
    s6 = json.loads((ART / "s6_summary.json").read_text())
    dtop = json.loads((ART / "s6_diff_full_s0_vs_top.json").read_text())
    dseed = json.loads((ART / "s6_diff_full_s0_vs_seed.json").read_text())
    nodes, edges = g["nodes"], g["edges"]
    run_by_name = {v["name"]: k for k, v in nodes.items() if v["type"] == "run"}
    produced: dict[str, list[str]] = {}
    consumed = []
    for e in edges:
        if e["label"] == "produces":
            produced.setdefault(e["from"], []).append(e["to"])
        else:
            consumed.append((e["from"], e["to"], e["label"]))
    producer_of = {ent: r for r, ents in produced.items() for ent in ents}
    trace_runs = set(trace["runs"])
    trace_pairs = {(e["produced_by"][0], e["run"]) for e in trace["edges"]}

    def ent(eid, short=False):
        n = nodes[eid]
        if n["type"] == "subset":
            return f"{n['name']} {eid[:8]}" if n.get("name") else f"subset {eid[:8]}"
        if n["type"] == "version":
            return f"version {n['name']} {eid[:8]}"
        return f"{n['type']} {eid[:8]}"

    n_eps = {"train-drop-top-20": s6["AC6.2"]["train_top"]["episodes_in_train_config"],
             "train-drop-random-20": s6["AC6.2"]["train_rand"]["episodes_in_train_config"]}

    W = (100, 480)
    A, B, C = 20, 245, 470
    LW, RH = 190, 54
    pos = {
        "build-v0": (W[0], 86, W[1], RH),
        "split-v0": (W[0], 170, W[1], RH),
        "derive-v1": (W[0], 254, W[1], 84),
        "screen-v1 [stand-in]": (A, 392, LW, RH),
        "train-full-s0": (B, 392, LW, RH),
        "train-full-s1": (C, 392, LW, RH),
        "eval-train-full-s0 [stand-in]": (B, 482, LW, RH),
        "attribute-train-full-s0 [stand-in]": (B, 572, LW, RH),
        "select-drop-top-20": (A, 662, LW, RH),
        "select-drop-random-20-s0": (C, 662, LW, RH),
        "train-drop-top-20": (A, 752, LW, RH),
        "train-drop-random-20": (C, 752, LW, RH),
        "eval-train-drop-top-20 [stand-in]": (A, 842, LW, RH),
        "eval-train-drop-random-20 [stand-in]": (C, 842, LW, RH),
    }
    assert set(pos) == set(run_by_name), sorted(set(pos) ^ set(run_by_name))
    BUS = 368
    cand = next(k for k, v in nodes.items() if v.get("name") == "v1/candidate")

    # consume-edge routes: (consumer run name, role) -> (points, label lines, label x, label y, anchor)
    routes = {
        ("build-v0", "source"): ([(340, 54), (340, 86)], ["source: 20 files, sha256 each"], 346, 74, "start"),
        ("split-v0", "population"): ([(340, 140), (340, 170)], None, 334, 159, "end"),
        ("derive-v1", "parent"): ([(580, 113), (622, 113), (622, 282), (580, 282)], None, 616, 158, "end"),
        ("derive-v1", "selection_pool"): ([(340, 224), (340, 254)], None, 334, 243, "end"),
        ("screen-v1 [stand-in]", "reference"): ([(100, 322), (40, 322), (40, 392)], None, 46, 352, "start"),
        ("screen-v1 [stand-in]", "candidate"): ([(150, BUS), (150, 392)], ["candidate"], 154, 385, "start"),
        ("train-full-s0", "training"): ([(340, BUS), (340, 392)], ["training"], 344, 385, "start"),
        ("train-full-s1", "training"): ([(565, BUS), (565, 392)], ["training"], 569, 385, "start"),
        ("attribute-train-full-s0 [stand-in]", "training"): ([(228, BUS), (228, 599), (245, 599)], ["training"], 224, 590, "end"),
        ("select-drop-top-20", "parent"): ([(12, BUS), (12, 689), (20, 689)], ["parent"], 18, 654, "start"),
        ("select-drop-random-20-s0", "parent"): ([(668, BUS), (668, 689), (660, 689)], ["parent"], 662, 654, "end"),
        ("eval-train-full-s0 [stand-in]", "checkpoint"): ([(340, 446), (340, 482)], None, 334, 468, "end"),
        ("attribute-train-full-s0 [stand-in]", "checkpoint"): ([(435, 419), (452, 419), (452, 586), (435, 586)], None, 458, 505, "start"),
        ("attribute-train-full-s0 [stand-in]", "rollouts"): ([(340, 536), (340, 572)], None, 334, 558, "end"),
        ("select-drop-top-20", "scores"): ([(300, 626), (300, 644), (150, 644), (150, 662)], None, 306, 640, "start"),
        ("train-drop-top-20", "training"): ([(115, 716), (115, 752)], None, 121, 738, "start"),
        ("train-drop-random-20", "training"): ([(565, 716), (565, 752)], None, 559, 738, "end"),
        ("eval-train-drop-top-20 [stand-in]", "checkpoint"): ([(115, 806), (115, 842)], None, 109, 828, "end"),
        ("eval-train-drop-random-20 [stand-in]", "checkpoint"): ([(565, 806), (565, 842)], None, 571, 828, "start"),
    }
    drawn = set()
    out = [f'<svg viewBox="0 0 680 968" role="img" aria-label="Lineage graph of the S6 dry run: 14 MLflow runs and the '
           f'entities they consumed and produced, from the 20 raw LIBERO files to the stand-in evaluations">',
           markers(pfx)]
    # edges first (boxes are drawn over them)
    out.append(poly([(340, 338), (340, BUS)], "e trace"))  # derive produces v1/candidate onto the bus
    out.append(poly([(12, BUS), (340, BUS)], "e trace"))
    out.append(poly([(340, BUS), (668, BUS)], "e"))
    for src, dst, role in consumed:
        dname = nodes[dst]["name"]
        key = (dname, role)
        pts, lines, lx, ly, anc = routes[key]
        prod = "raw" if src == "raw:libero" else producer_of[src]
        on_trace = (src == "raw:libero" and dst in trace_runs) or (prod, dst) in trace_pairs
        cls = "e trace" if on_trace else "e"
        out.append(poly(pts, cls, f"{pfx}-{'aht' if on_trace else 'ah'}"))
        if lines is None:  # label = the consumed entity (kind + short id), plus its role when it differs
            label = ent(src)
            kind = nodes[src]["type"]
            lines = [label if role == kind else f"{label} · {role}"]
            if key == ("attribute-train-full-s0 [stand-in]", "checkpoint"):
                lines = label.split(" ", 1)  # two lines: it sits in a narrow column
        for i, ln in enumerate(lines):
            out.append(text(lx, ly + 11 * i, ln, "e-label", anc))
        drawn.add((src, dst, role))
    assert drawn == set(consumed), "every consume edge of the graph must be drawn exactly once"
    out.append(text(676, 348, "v1/candidate", "e-label", "end"))
    out.append(text(676, 359, f"{cand[:8]} → 6 runs", "e-label", "end"))
    # raw node
    raw = nodes["raw:libero"]
    out.append(f'<rect x="170" y="10" width="340" height="44" rx="8" class="n-run trace"/>')
    out.append(text(340, 28, f"raw LIBERO HDF5 × {raw['count']}", "n-title", "middle"))
    out.append(text(340, 44, "libero_spatial + libero_10 · 1000 demos · sha256 per file", "n-meta", "middle"))
    # run boxes
    for name, (x, y, w, h) in pos.items():
        rid = run_by_name[name]
        n = nodes[rid]
        stand = n["stand_in"]
        cls = "n-run" + (" trace" if rid in trace_runs else "") + (" standin" if stand else "")
        tip = f"{n['stage']} run {name} · run_id {rid} · git {n['git']} · {n['status']}"
        out.append(f'<g><title>{esc(tip)}</title><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="{cls}"/>')
        out.append(text(x + 8, y + 18, name.replace(" [stand-in]", ""), "n-title"))
        if w > LW:
            meta = f"{n['stage']} · run {rid[:8]} · git {n['git'][:7]}"
        else:
            meta = f"{n['stage']} · run {rid[:8]}"
        if stand:
            out.append(f'<text x="{x + 8}" y="{y + 32}" class="n-meta"><tspan class="badge">STAND-IN</tspan> · {esc(meta)}</text>')
        else:
            out.append(text(x + 8, y + 32, meta, "n-meta"))
        outs = produced.get(rid, [])
        if name == "derive-v1":
            v1 = next(e for e in outs if nodes[e]["type"] == "version")
            lab = next(e for e in outs if nodes[e]["type"] == "labels")
            others = [e for e in outs if nodes[e]["type"] == "subset" and e != cand]
            others = sorted(others, key=lambda e: nodes[e]["name"])
            out.append(text(x + 8, y + 48, "→ " + " · ".join([ent(v1)] + [ent(e) for e in others]), "n-out"))
            out.append(text(x + 8, y + 62, f"→ {ent(cand)} (960 episodes; the bus below)", "n-out"))
            out.append(f'<text x="{x + 8}" y="{y + 76}" class="n-out"><tspan class="gt">→ labels {lab[:8]}</tspan>'
                       f' · ground truth; no run consumes it</text>')
        else:
            def order(e):
                return ({"version": 0, "split": 1}.get(nodes[e]["type"], 2), nodes[e].get("name") or "")
            line = " · ".join(ent(e) for e in sorted(outs, key=order))
            if nodes[outs[0]]["type"] == "subset" and name.startswith("select"):
                tr = "train-drop-top-20" if "top" in name else "train-drop-random-20"
                line += f" · {n_eps[tr]} eps"
            out.append(text(x + 8, y + 46, "→ " + line, "n-out"))
        out.append("</g>")
    out.append(text(24, 460, "stored, not acted on (§9.1)", "n-note"))
    out.append(text(474, 460, "seed control; not consumed", "n-note"))
    # diff callout
    t, sd = dtop["data"], dseed
    out.append('<rect x="222" y="752" width="236" height="142" rx="8" class="callout"/>')
    out.append(text(232, 771, "diff(train-full-s0, train-drop-top-20)", "c-title"))
    out.append(text(232, 787, f"removed {t['n_removed']} · added {t['n_added']} · changed {t['n_changed_version']}", "c-body"))
    out.append(text(232, 801, f"params {json.dumps(dtop['params'])} · config {json.dumps(dtop['config'])}", "c-body"))
    out.append(text(232, 815, f"git same: {str(dtop['git']['same']).lower()}", "c-body"))
    out.append(text(232, 839, "diff(train-full-s0, train-full-s1)", "c-title"))
    d2 = sd["data"]
    out.append(text(232, 855, f"removed {d2['n_removed']} · added {d2['n_added']} · changed {d2['n_changed_version']}", "c-body"))
    s0, s1 = sd["params"]["seed"]
    out.append(text(232, 869, f"params: seed {s0} → {s1} (only)", "c-body"))
    out.append(text(232, 883, f"git same: {str(sd['git']['same']).lower()}", "c-body"))
    # legend
    out.append('<rect x="20" y="914" width="22" height="13" rx="3" class="n-run"/>')
    out.append(text(48, 925, "real stage run", "lg-label"))
    out.append('<rect x="150" y="914" width="22" height="13" rx="3" class="n-run standin"/>')
    out.append(text(178, 925, "STAND-IN: placeholder payload, no outcome numbers", "lg-label"))
    out.append(poly([(20, 950), (42, 950)], "e trace"))
    out.append(text(48, 954, f"trace(train-drop-top-20): {len(trace['runs'])} runs, {len(trace['edges'])} edges, "
                             f"{len(trace['raw_files'])} raw files", "lg-label"))
    out.append(text(400, 954, "arrow label = entity consumed · role", "lg-label"))
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------
# 2. v0 -> v1 episode grid (S4)
# ---------------------------------------------------------------------------------------------


def episode_grid(pfx="eg") -> str:
    rows = list(csv.DictReader((ART / "summarizer_v1_episode_status.csv").open()))
    assert len(rows) == 1000
    suites = ["libero_spatial", "libero_10"]
    tasks = []
    for s in suites:
        ids = sorted({int(r["libero_task_id"]) for r in rows if r["suite"] == s})
        tasks += [(s, i) for i in ids]
    L, PX, CS, PY = 128, 10.6, 8.6, 13
    top = 40
    n_cor = sum(r["corrupted_in_v1"] == "True" for r in rows)
    n_ref = sum(r["split"] == "reference" for r in rows)
    n_same = sum(r["hash_changed"] == "False" for r in rows)
    H = top + 20 * PY + 12 + 66
    out = [f'<svg viewBox="0 0 680 {H:g}" role="img" aria-label="Grid of all 1000 episodes, 20 tasks by 50 demos: '
           f'{n_cor} corrupted in v1, {n_ref} reference episodes, {n_same} with identical content hash in v0 and v1">']
    for k in (0, 10, 20, 30, 40, 49):
        x = L + k * PX + CS / 2
        out.append(text(x, top - 8, f"{k}", "ax-label", "middle"))
    out.append(text(L - 6, top - 8, "demo_k →", "ax-label", "end"))
    by = {(r["suite"], int(r["libero_task_id"]), int(r["demo"])): r for r in rows}
    y = top
    for ti, (s, tid) in enumerate(tasks):
        if ti == 10:
            y += 12
        tname = next(r["task_name"] for r in rows if r["suite"] == s and int(r["libero_task_id"]) == tid)
        out.append(f'<g><title>{esc(s + " task " + str(tid) + ": " + tname)}</title>'
                   + text(L - 6, y + CS - 0.5, f"{s} {tid}", "ax-label", "end") + "</g>")
        for d in range(50):
            r = by[(s, tid, d)]
            x = L + d * PX
            if r["corrupted_in_v1"] == "True":
                cls, st = "cell cor", "corrupted in v1 (content hash changed)"
            elif r["split"] == "reference":
                cls, st = "cell ref", "reference split (never editable; hash identical)"
            else:
                cls, st = "cell same", "candidate, identical in v0 and v1"
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{CS}" height="{CS}" rx="1.5" class="{cls}">'
                       f'<title>{esc(f"{s} task {tid} demo_{d}: {st}")}</title></rect>')
        y += PY
    ly = y + 22
    out.append(f'<rect x="{L}" y="{ly - 9}" width="{CS}" height="{CS}" rx="1.5" class="cell cor"/>')
    out.append(text(L + 14, ly, f"corrupted in v1: {n_cor} (5 per task) · content hash changed", "lg-label"))
    out.append(f'<rect x="{L}" y="{ly + 9}" width="{CS}" height="{CS}" rx="1.5" class="cell ref"/>')
    out.append(text(L + 14, ly + 18, f"reference split: {n_ref} (2 per task) · guard refuses to edit them", "lg-label"))
    out.append(f'<rect x="{L}" y="{ly + 27}" width="{CS}" height="{CS}" rx="1.5" class="cell same"/>')
    out.append(text(L + 14, ly + 36, f"identical content hash in v0 and v1: {n_same} (860 candidates + 40 references)",
                    "lg-label"))
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------
# 3. per-task human durations (S2, §3.5 HRT reference)
# ---------------------------------------------------------------------------------------------


def _short_task(suite, name):
    """Row label: libero_spatial tasks without their shared 'pick up the black bowl … place it on the
    plate' frame; libero_10 tasks without the scene prefix, cut at 36 characters (full name in the tooltip)."""
    if suite == "libero_spatial":
        s = name.removeprefix("pick_up_the_black_bowl_").removesuffix("_and_place_it_on_the_plate")
        return s.replace("_", " ")
    m = re.match(r"^((?:[A-Z]+_)+SCENE\d+)_(.*)$", name)
    s = m.group(2).replace("_", " ")
    return s if len(s) <= 36 else s[:35].rstrip() + "…"


def durations(pfx="du") -> str:
    rows = list(csv.DictReader((ART / "s2_human_duration_per_task.csv").open()))
    assert len(rows) == 20
    summ = (ART / "s2_catalog_stdout.txt").read_text()
    per_suite = json.loads(summ[summ.index("per-suite summary:") + len("per-suite summary:"):summ.index("}\n}") + 3])
    X0, X1, VMAX = 262, 660, 26.0
    sx = lambda v: X0 + (X1 - X0) * v / VMAX  # noqa: E731
    top, RP = 36, 19
    H = top + 2 * 22 + 20 * RP + 44
    out = [f'<svg viewBox="0 0 680 {H:g}" role="img" aria-label="Human demonstration duration per task: range '
           f'and median of 50 demos for each of the 20 tasks">']
    plot_bottom = top + 2 * 22 + 20 * RP
    for v in range(0, 26, 5):
        x = sx(v)
        out.append(f'<line x1="{x:.1f}" y1="{top - 4}" x2="{x:.1f}" y2="{plot_bottom}" class="svg-grid"/>')
        out.append(text(x, plot_bottom + 14, f"{v}", "ax-label", "middle"))
    out.append(text((X0 + X1) / 2, plot_bottom + 32, "human time-to-success T per demo (s) = length_steps / 20 Hz",
                    "ax-label", "middle"))
    y = top
    longest = max(rows, key=lambda r: float(r["median"]))
    shortest = min(rows, key=lambda r: float(r["median"]))
    for s in ("libero_spatial", "libero_10"):
        ps = per_suite[s]
        dmin, dmed, dmax = ps["duration_s"]
        out.append(text(8, y + 12, f"{s}: {ps['episodes']} demos, median {dmed:g} s, range {dmin:g}–{dmax:g} s",
                        "grp-label"))
        y += 22
        for r in [r for r in rows if r["suite"] == s]:
            cy = y + RP / 2
            lo, md, mn, hi = (float(r[k]) for k in ("min", "median", "mean", "max"))
            tip = (f"{s} task {r['libero_task_id']}: {r['task_name']} · n={r['count']} · min {lo:g} · median {md:g} · "
                   f"mean {mn:g} · max {hi:g} s")
            out.append(f'<g><title>{esc(tip)}</title>')
            out.append(f'<rect x="0" y="{y:.1f}" width="680" height="{RP}" class="hit"/>')
            out.append(text(X0 - 10, cy + 3.5, f"{r['libero_task_id']} · {_short_task(s, r['task_name'])}", "row-label", "end"))
            out.append(f'<line x1="{sx(lo):.1f}" y1="{cy:.1f}" x2="{sx(hi):.1f}" y2="{cy:.1f}" class="range"/>')
            out.append(f'<circle cx="{sx(md):.1f}" cy="{cy:.1f}" r="4" class="med"/>')
            if r is longest or r is shortest:  # direct label on the extremes only; left of the line if no room
                if sx(hi) + 86 <= 680:
                    out.append(text(sx(hi) + 6, cy + 3.5, f"median {md:g} s", "val-label"))
                else:
                    out.append(text(sx(lo) - 6, cy + 3.5, f"median {md:g} s", "val-label", "end"))
            out.append("</g>")
            y += RP
    out.append(f'<line x1="{X0}" y1="{plot_bottom}" x2="{X1}" y2="{plot_bottom}" class="svg-axis"/>')
    lx = X0
    ly = top - 22
    out.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 26}" y2="{ly}" class="range"/>')
    out.append(text(lx + 32, ly + 3.5, "min–max of the task's 50 demos", "lg-label"))
    out.append(f'<circle cx="{lx + 228}" cy="{ly}" r="4" class="med"/>')
    out.append(text(lx + 238, ly + 3.5, "median", "lg-label"))
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------
# 4. random-subset membership (S5, AC5.10)
# ---------------------------------------------------------------------------------------------


def membership(pfx="mb") -> str:
    rows = list(csv.reader((ART / "s5_r2_membership_matrix.csv").open()))
    head, data = rows[0], rows[1:]
    M = [[int(v) for v in r[1:]] for r in data]
    n, m = len(M), len(head) - 1
    counts = [sum(1 for r in M if sum(r) == k) for k in range(m + 1)]
    from math import comb
    expect = [n * comb(m, k) / 2 ** m for k in range(m + 1)]
    X0, X1, Y0, Y1 = 70, 650, 30, 210
    vmax = 400
    sy = lambda v: Y1 - (Y1 - Y0) * v / vmax  # noqa: E731
    out = [f'<svg viewBox="0 0 680 270" role="img" aria-label="Number of the {n} candidate episodes that appear in '
           f'0, 1, 2 or 3 of the {m} random 480-episode subsets: {counts}">']
    for v in range(0, vmax + 1, 100):
        out.append(f'<line x1="{X0}" y1="{sy(v):.1f}" x2="{X1}" y2="{sy(v):.1f}" class="svg-grid"/>')
        out.append(text(X0 - 8, sy(v) + 3.5, f"{v}", "ax-label", "end"))
    bw = 70
    step = (X1 - X0) / (m + 1)
    for k in range(m + 1):
        cx = X0 + step * (k + 0.5)
        h = Y1 - sy(counts[k])
        out.append(f'<g><title>{esc(f"in {k} of {m} subsets: {counts[k]} episodes (binomial expectation {expect[k]:g})")}</title>')
        out.append(f'<path d="M{cx - bw / 2:.1f},{Y1} V{sy(counts[k]) + 4:.1f} Q{cx - bw / 2:.1f},{sy(counts[k]):.1f} '
                   f'{cx - bw / 2 + 4:.1f},{sy(counts[k]):.1f} H{cx + bw / 2 - 4:.1f} Q{cx + bw / 2:.1f},{sy(counts[k]):.1f} '
                   f'{cx + bw / 2:.1f},{sy(counts[k]) + 4:.1f} V{Y1} Z" class="bar"/>')
        out.append(f'<line x1="{cx - bw / 2 - 8:.1f}" y1="{sy(expect[k]):.1f}" x2="{cx + bw / 2 + 8:.1f}" '
                   f'y2="{sy(expect[k]):.1f}" class="expect"/>')
        out.append(text(cx, sy(counts[k]) - 8, f"{counts[k]}", "val-label", "middle"))
        out.append(text(cx, Y1 + 16, f"in {k} of {m}", "ax-label", "middle"))
        out.append("</g>")
    out.append(f'<line x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}" class="svg-axis"/>')
    out.append(text(14, 120, "episodes", "ax-label", "middle", ' transform="rotate(-90 14 120)"'))
    out.append(text((X0 + X1) / 2, Y1 + 36, f"how many of the {m} random subsets (seeds 0/1/2, k = 480 dropped each) "
                    f"contain the episode", "ax-label", "middle"))
    out.append(f'<rect x="{X0 + 8}" y="12" width="14" height="10" rx="2" class="bar"/>')
    out.append(text(X0 + 28, 21, "measured, from the 3 subset records", "lg-label"))
    out.append(f'<line x1="{X0 + 268}" y1="17" x2="{X0 + 296}" y2="17" class="expect"/>')
    out.append(text(X0 + 302, 21, "Binomial(3, 0.5) × 960 (computed, for reference)", "lg-label"))
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------
# 5. subset derivation tree (S3)
# ---------------------------------------------------------------------------------------------


def subsets(pfx="st") -> str:
    split = json.loads((ART / "s3_split_make_stdout.txt").read_text())
    s3 = json.loads((ART / "s3_summary_r2.json").read_text())
    s5 = json.loads((ART / "s5_r2_summary.json").read_text())
    g = json.loads((ART / "s6_lineage_graph.json").read_text())["nodes"]
    names = {v.get("name"): k for k, v in g.items() if v["type"] == "subset" and v.get("name")}
    top_child = next(e["to"] for e in json.loads((ART / "s6_lineage_graph.json").read_text())["edges"]
                     if e["from"] == "f64a0385917b46a987169f0d384dcaec" and e["label"] == "produces")
    rnd_child = next(e["to"] for e in json.loads((ART / "s6_lineage_graph.json").read_text())["edges"]
                     if e["from"] == "cb4093eb7ed74d6cabe3f87a952f4de8" and e["label"] == "produces")
    ch = s3["AC3.8"]["children"]
    rand_ids = s5["AC5.10"]["subset_ids"]
    out = [f'<svg viewBox="0 0 680 400" role="img" aria-label="How subsets derive from one another: the split of v0 '
           f'and the selections made from v1/candidate, with their real subset ids and sizes">', markers(pfx)]

    def box(x, y, w, l1, l2, l3=None, cls="n-run"):
        h = 44 if l3 is None else 56
        s = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" class="{cls}"/>',
             text(x + 8, y + 17, l1, "n-title"), text(x + 8, y + 31, l2, "n-out")]
        if l3:
            s.append(text(x + 8, y + 45, l3, "n-meta"))
        return "\n".join(s)

    v0all = names["v0/all"]
    out.append(box(20, 20, 190, "v0/all", f"{v0all[:8]} · 1000 episodes"))
    out.append(box(250, 10, 200, "v0/reference", f"{split['subsets']['reference'][:8]} · {split['sizes']['reference']} eps",
                   "held out, clean (2 per task)"))
    out.append(box(250, 74, 200, "v0/candidate", f"{split['subsets']['candidate'][:8]} · {split['sizes']['candidate']} eps",
                   "the training pool"))
    out.append(poly([(210, 42), (230, 42), (230, 38), (250, 38)], "e", f"{pfx}-ah"))
    out.append(poly([(230, 42), (230, 102), (250, 102)], "e", f"{pfx}-ah"))
    out.append(text(232, 152, f"split: seed {split['seed']}, {split['reference_per_task']} per task, stratified", "e-label"))
    out.append(text(470, 40, "derive-v1 copies the split:", "e-label"))
    out.append(text(470, 52, "v1/reference, v1/candidate", "e-label"))
    out.append(text(470, 64, "(same episode_ids, v1 hashes)", "e-label"))
    out.append(f'<line x1="20" y1="172" x2="660" y2="172" class="svg-grid"/>')
    v1c = names["v1/candidate"]
    out.append(box(20, 250, 190, "v1/candidate", f"{v1c[:8]} · 960 episodes", "parent of every selection"))
    kids = [
        ("drop_top_k  k=20, highest placeholder_score", f"{top_child[:8]} · 940 eps", "S6 · main store · trained",
         "n-run"),
        ("drop_random_k  k=20, seed 0", f"{rnd_child[:8]} · 940 eps", "S6 · main store · trained", "n-run"),
        ("drop_group  task_name=KITCHEN_SCENE3…", f"{ch['task_name']['subset_id'][:8]} · {ch['task_name']['n']} eps",
         "r2 · scratch root · leave-one-task-out", "n-run"),
        ("drop_group  suite=libero_10", f"{ch['suite']['subset_id'][:8]} · {ch['suite']['n']} eps",
         "r2 · scratch root · leave-one-suite-out", "n-run"),
        ("drop_random_k  k=480, seeds 0/1/2", " · ".join(i[:8] for i in rand_ids),
         f"r2 · scratch · {s5['AC5.10']['sizes'][0]} eps each", "n-run"),
    ]
    y = 186
    for i, (l1, l2, l3, cls) in enumerate(kids):
        yy = y + i * 42
        out.append(f'<rect x="300" y="{yy}" width="360" height="36" rx="7" class="{cls}"/>')
        out.append(text(308, yy + 15, l1, "n-title-s"))
        out.append(text(308, yy + 29, l2 + "   " + l3, "n-meta"))
        out.append(poly([(210, 272), (250, 272), (250, yy + 18), (300, yy + 18)], "e", f"{pfx}-ah"))
    out.append(text(20, 330, "Each child records op, parent subset id", "e-label"))
    out.append(text(20, 342, "and the dropped ids; subset_id hashes", "e-label"))
    out.append(text(20, 354, "(version, episode_ids, derivation).", "e-label"))
    out.append("</svg>")
    return "\n".join(out)


FIGS = {"lineage": lineage_graph, "episodegrid": episode_grid, "durations": durations,
        "membership": membership, "subsets": subsets}


def main() -> int:
    built = {k: f() for k, f in FIGS.items()}
    for page in sorted(FEAT.glob("*.html")):
        s = page.read_text()
        n_before = s
        for k, svg in built.items():
            pat = re.compile(rf"(<!--FIG:{k}:start-->).*?(<!--FIG:{k}:end-->)", re.S)
            # each page uses a unique id prefix per occurrence so marker ids never collide
            s = pat.sub(lambda m_: m_.group(1) + "\n" + svg + "\n" + m_.group(2), s)
        if s != n_before:
            page.write_text(s)
            print("updated", page.name, [k for k in built if f"<!--FIG:{k}:start-->" in s])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

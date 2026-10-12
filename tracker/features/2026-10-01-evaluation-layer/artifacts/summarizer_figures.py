"""Draw the data-driven inline-SVG figures and generated tables of this feature's report and story pages.

Every figure is computed from files in this ``artifacts/`` directory; no number is typed in by hand:
- ``summarizer_data.json``         (written by ``summarizer_extract.py`` from the real data with the feature's own code)
- ``phail_validation.json``        S2: the four yardsticks with CIs
- ``s7_phail_verdicts.json``       S7: the six PhAIL verdicts (seed 0)
- ``s7_ac7_1_seed_sweep.json``     S7: the five-seed sweep (r3 criterion)
- ``s7_ac7_2_null_calibration_stdout.txt``, ``s6_hrt_linearization_curve.csv``, ``s6_ac6_10_returns_vs_verdict.csv``,
  ``s6_metrics_registry.json``, ``s4_human_operation_stats.csv``, ``s4_operations_table.csv``,
  ``s8_ac8_2_normalizer_smolvla_0a33bc75.json`` / ``s8_ac8_2_3_stdout.txt``, ``s9_lineage_graph.mmd``, ``s0_*``

The script rewrites the block between ``<!--FIG:<name>:start-->`` and ``<!--FIG:<name>:end-->`` in every ``*.html``
page of the feature folder, so it can be re-run after an artifact changes. Colours are the CSS variables of
``tracker/assets/tracker.css`` (classes defined in each page's <style>); no hex values. Native ``<title>`` elements give
hover tooltips without scripts, so the pages work offline as ``file://``.

Usage (repo root)::

    .venv/bin/python tracker/features/2026-10-01-evaluation-layer/artifacts/summarizer_figures.py
"""

from __future__ import annotations

import csv
import html
import json
import math
import re
from pathlib import Path

ART = Path(__file__).resolve().parent
FEAT = ART.parent


def load(name):
    return json.loads((ART / name).read_text())


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def T(x, y, s, cls, anchor="start", extra="") -> str:
    a = "" if anchor == "start" else f' text-anchor="{anchor}"'
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}"{a}{extra}>{esc(s)}</text>'


def L(x1, y1, x2, y2, cls, extra="") -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}"{extra}/>'


def P(pts, cls, extra="") -> str:
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return f'<path d="{d}" class="{cls}"{extra}/>'


def R(x, y, w, h, cls, rx=0, title=None) -> str:
    a = f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" rx="{rx}" class="{cls}"'
    return a + (f"><title>{esc(title)}</title></rect>" if title else "/>")


def C(x, y, r, cls, title=None) -> str:
    t = f"<title>{esc(title)}</title>" if title else ""
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" class="{cls}">{t}</circle>'


def svg(w, h, label, body) -> str:
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}">'
            + "".join(body) + "</svg>")


def arrow_defs(pfx):
    out = []
    for name, cls in (("ah", "ah"), ("aht", "ah hl")):
        out.append(f'<marker id="{pfx}-{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                   f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="{cls}"/></marker>')
    return "<defs>" + "".join(out) + "</defs>"


class Scale:
    def __init__(self, d0, d1, r0, r1, log=False):
        self.d0, self.d1, self.r0, self.r1, self.log = d0, d1, r0, r1, log

    def __call__(self, v):
        if self.log:
            a, b, v = math.log10(self.d0), math.log10(self.d1), math.log10(max(v, 1e-300))
        else:
            a, b = self.d0, self.d1
        return self.r0 + (v - a) / (b - a) * (self.r1 - self.r0)


def step_path(steps, sx, sy, x_max):
    """KM step function (t, F) -> pixel path, right-continuous; points that share a pixel column collapse."""
    pts = []
    prev = None
    for t, f in steps:
        if t > x_max:
            break
        x, y = sx(t), sy(f)
        if prev is not None:
            pts.append((x, prev[1]))  # horizontal run to the jump
        pts.append((x, y))
        prev = (x, y)
    if prev is not None:
        pts.append((sx(x_max), prev[1]))
    # collapse vertical jitter within one 0.6 px column
    out = []
    for x, y in pts:
        if out and abs(out[-1][0] - x) < 0.6 and len(out) >= 2 and abs(out[-2][0] - x) < 0.6:
            out[-1] = (x, y)
        else:
            out.append((x, y))
    return out


def axes(body, sx, sy, xt, yt, x0, x1, y0, y1, xlab, ylab, xfmt=str, yfmt=str, grid=True, ylab_dx=34):
    for v in yt:
        y = sy(v)
        if grid:
            body.append(L(x0, y, x1, y, "gr"))
        body.append(T(x0 - 5, y + 3.5, yfmt(v), "tk", "end"))
    for v in xt:
        x = sx(v)
        body.append(L(x, y1, x, y1 + 4, "ax"))
        body.append(T(x, y1 + 15, xfmt(v), "tk", "middle"))
    body.append(L(x0, y1, x1, y1, "ax"))
    body.append(L(x0, y0, x0, y1, "ax"))
    if xlab:
        body.append(T((x0 + x1) / 2, y1 + 30, xlab, "lbm", "middle"))
    if ylab:
        body.append(f'<text x="{x0 - ylab_dx:.1f}" y="{(y0 + y1) / 2:.1f}" class="lbm" text-anchor="middle" '
                    f'transform="rotate(-90 {x0 - ylab_dx:.1f} {(y0 + y1) / 2:.1f})">{esc(ylab)}</text>')


def fmt_g(v):
    return f"{v:g}"


# ---------------------------------------------------------------------------------------------
# PhAIL: four yardsticks, three winners (slope chart of ranks)
# ---------------------------------------------------------------------------------------------

def phail_slope() -> str:
    tab = load("phail_validation.json")["AC2.4"]["table"]
    models = ["OpenPI", "GR00T", "ACT", "SmolVLA"]
    better_low = {"global RMST(240 s) (s)"}
    short = {"F(30 s)": ("F(30 s)", "higher = better"), "HRT macro (%)": ("HRT macro", "higher = better"),
             "global RMST(240 s) (s)": ("global RMST(240 s)", "lower = better"),
             "AUC vs human (macro)": ("AUC vs human", "higher = better")}
    fmt = {"F(30 s)": "{:.3f}", "HRT macro (%)": "{:.2f}%", "global RMST(240 s) (s)": "{:.1f} s", "AUC vs human (macro)": "{:.3f}"}
    style = {"GR00T": "sl-a", "ACT": "sl-b", "OpenPI": "sl-c", "SmolVLA": "sl-d"}
    W, H = 680, 300
    x0, x1 = 150, 560
    xs = [x0 + i * (x1 - x0) / 3 for i in range(4)]
    ys = {1: 78, 2: 128, 3: 178, 4: 228}
    body = []
    ranks = {}
    for j, row in enumerate(tab):
        k = row["yardstick"]
        order = sorted(models, key=lambda m: row[m][0], reverse=k not in better_low)
        for r, m in enumerate(order, 1):
            ranks[(m, j)] = (r, row[m][0], row[m][1], row[m][2])
        body.append(T(xs[j], 26, short[k][0], "tl", "middle"))
        body.append(T(xs[j], 41, short[k][1], "lbs", "middle"))
        body.append(L(xs[j], 58, xs[j], 248, "gr"))
    for r, y in ys.items():
        body.append(T(12, y + 4, f"rank {r}", "tk", "start"))
    for m in models:
        pts = [(xs[j], ys[ranks[(m, j)][0]]) for j in range(4)]
        body.append(P(pts, style[m]))
    for m in models:
        for j, row in enumerate(tab):
            r, v, lo, hi = ranks[(m, j)]
            k = row["yardstick"]
            x, y = xs[j], ys[r]
            tip = f"{m} · {short[k][0]} = {fmt[k].format(v)} (95% CI {fmt[k].format(lo)} – {fmt[k].format(hi)}), rank {r}"
            body.append(C(x, y, 6.5 if r == 1 else 4.5, "dot-win" if r == 1 else "dot", tip))
            body.append(T(x, y - 10, fmt[k].format(v), "val", "middle"))
        body.append(T(xs[0] - 14, ys[ranks[(m, 0)][0]] + 4, m, "lb", "end"))
        body.append(T(xs[3] + 14, ys[ranks[(m, 3)][0]] + 4, m, "lb", "start"))
    body.append(C(150, 276, 6.5, "dot-win"))
    body.append(T(162, 280, "top-1 under that yardstick (3 different models win the 4 yardsticks)", "lbm"))
    return svg(W, H, "Rank of four PhAIL models under four yardsticks, computed from PhAIL's released data", body)


# ---------------------------------------------------------------------------------------------
# PhAIL: GR00T vs ACT pooled KM CDFs cross
# ---------------------------------------------------------------------------------------------

def phail_km_cross() -> str:
    d = load("summarizer_data.json")
    km = d["phail_km"]
    cross = d["phail_cross"]
    W, H = 680, 300
    body = []
    panels = [(70, 320, 30.0, [0, 5, 10, 15, 20, 25, 30], "A · first 30 s"),
              (400, 600, 240.0, [0, 60, 120, 180, 240], "B · full horizon, τ = 240 s")]
    y0, y1 = 40, 240
    sy = Scale(0, 1, y1, y0)
    for (x0, x1, xm, xt, title) in panels:
        sx = Scale(0, xm, x0, x1)
        axes(body, sx, sy, xt, [0, 0.25, 0.5, 0.75, 1.0], x0, x1, y0, y1, "time to place one item, T (s)",
             "F(t) = P(T ≤ t)" if x0 < 100 else "", fmt_g, lambda v: f"{v:g}")
        body.append(T(x0, 26, title, "tl"))
        for m, cls in (("Teleop", "s-c"), ("ACT", "s-b"), ("GR00T", "s-a")):
            body.append(P(step_path(km[m]["steps"], sx, sy, xm), cls))
        if xm >= 30:
            xc = sx(cross["sign_changes_s"][0])
            if x0 < 100:
                body.append(L(xc, y0, xc, y1, "ref"))
                body.append(T(xc + 4, sy(0.62), f"curves cross at {cross['sign_changes_s'][0]:.1f} s", "lbs"))
                body.append(T(sx(1.5), sy(0.88), "Teleop (human)", "lbm"))
                xa = sx(cross["max_act_lead_at_s"])
                body.append(T(xa - 4, sy(0.20), "ACT ahead", "lbs", "end"))
    # direct labels on panel B ends
    sxB = Scale(0, 240, 400, 600)
    for m in ("GR00T", "ACT"):
        f = km[m]["F_probe"]["240.0"]
        body.append(T(sxB(240) + 4, sy(f) + 4, f"{m} {f:.2f}", "lb"))
    body.append(T(sxB(150), sy(0.97) + 4, "Teleop 1.00", "lbm", "middle"))
    # legend
    lx = 80
    for i, (m, cls, lab) in enumerate((("GR00T", "s-a", "GR00T"), ("ACT", "s-b", "ACT"), ("Teleop", "s-c", "human teleop reference"))):
        x = lx + i * 160
        body.append(L(x, 288, x + 26, 288, cls))
        body.append(T(x + 32, 292, f"{lab} (n {km[m]['n_episodes']} episodes)", "lbm"))
    return svg(W, H, "Pooled Kaplan-Meier CDFs of GR00T, ACT and the human teleop reference on PhAIL's released data", body)


# ---------------------------------------------------------------------------------------------
# PhAIL: per-metric differences with 95% CI (forest), GR00T vs ACT and OpenPI vs ACT, seed 0
# ---------------------------------------------------------------------------------------------

def verdict_forest() -> str:
    vs = {tuple(v["arms"]): v for v in load("s7_phail_verdicts.json")}
    pairs = [("GR00T", "ACT"), ("OpenPI", "ACT")]
    names = {"phail_f30": ("F(30 s)", "{:+.3f}"), "phail_hrt240": ("HRT(240 s)", "{:+.3f}"),
             "phail_rmst240": ("global RMST(240 s)", "{:+.1f} s"), "phail_auc": ("AUC vs human", "{:+.3f}")}
    W = 680
    row_h, top = 30, 20
    body = []
    y = top
    for a, b in pairs:
        v = vs[(a, b)]
        g = v["gate"]
        out = v["outcome"]
        y += 18
        body.append(T(16, y, f"{a} vs {b}", "tl"))
        body.append(T(150, y, f"gate: macro-KS D̄ {g['D']:.3f}, p {g['p']:.4f} → {'fires' if g['fires'] else 'does not fire'};"
                      f" outcome ({out})", "lbm"))
        y += 14
        cx0, cx1 = 330, 560
        mid = (cx0 + cx1) / 2
        body.append(T(cx0, y + 2, f"← favours {b}", "lbs"))
        body.append(T(cx1, y + 2, f"favours {a} →", "lbs", "end"))
        y_rows0 = y + 8
        for k in ("phail_f30", "phail_hrt240", "phail_rmst240", "phail_auc"):
            m = v["metrics"][k]
            sign = 1 if m["higher_is_better"] else -1
            lo, hi = sorted((sign * m["ci95"][0], sign * m["ci95"][1]))
            pt = sign * m["diff"]
            span = max(abs(lo), abs(hi)) * 1.15
            sx = Scale(-span, span, cx0, cx1)
            yy = y + 8 + row_h / 2
            sig = m["significant"]
            body.append(T(16, yy + 4, names[k][0], "lb"))
            body.append(T(150, yy + 4, f"Δ {a}−{b} {names[k][1].format(m['diff'])}", "lbs"))
            body.append(L(cx0, yy, cx1, yy, "gr"))
            body.append(L(sx(lo), yy, sx(hi), yy, "ci-hl" if sig else "ci"))
            body.append(C(sx(pt), yy, 5 if sig else 4, "dot-win" if sig else "dot",
                          f"{names[k][0]}: Δ {m['diff']:+.4f}, 95% CI [{m['ci95'][0]:+.4f}, {m['ci95'][1]:+.4f}], p {m['p']:.4f}, Holm {m['p_holm']:.4f}"))
            body.append(T(572, yy + 4, f"Holm p {m['p_holm']:.3f}{' ✱' if sig else ''}", "lbs" if not sig else "lb"))
            y += row_h
        body.append(L(mid, y_rows0, mid, y + 8, "zero"))
        y += 22
    body.append(T(16, y + 4, "✱ significant after Holm (4 metrics, α 0.05). Each row has its own scale; the line marks no difference.", "lbs"))
    return svg(W, y + 16, "Per-metric differences with episode-clustered bootstrap 95% CIs for two PhAIL pairs", body)


# ---------------------------------------------------------------------------------------------
# PhAIL: the five-seed sweep (gate p and Holm p), log scale, α line
# ---------------------------------------------------------------------------------------------

def seed_sweep() -> str:
    sw = load("s7_ac7_1_seed_sweep.json")
    res = sw["results"]
    seeds = sw["seeds"]
    rows = [
        ("OpenPI vs GR00T · gate p", [res[f"OpenPI-GR00T-seed{s}"]["gate_p"] for s in seeds], "(iii) 5/5"),
        ("OpenPI vs ACT · gate p", [res[f"OpenPI-ACT-seed{s}"]["gate_p"] for s in seeds], "(iii) 5/5"),
        ("GR00T vs ACT · gate p", [res[f"GR00T-ACT-seed{s}"]["gate_p"] for s in seeds], "fires 5/5"),
        ("GR00T vs ACT · RMST Holm p", [res[f"GR00T-ACT-seed{s}"]["p_holm"]["phail_rmst240"] for s in seeds], "→ GR00T 5/5"),
        ("GR00T vs ACT · AUC Holm p", [res[f"GR00T-ACT-seed{s}"]["p_holm"]["phail_auc"] for s in seeds], "→ ACT 4/5"),
        ("each vs SmolVLA · gate p", [res[f"{m}-SmolVLA-seed{s}"]["gate_p"] for m in ("OpenPI", "GR00T", "ACT") for s in seeds], "(i) 15/15"),
    ]
    W = 680
    x0, x1 = 200, 560
    sx = Scale(0.0003, 0.3, x0, x1, log=True)
    top, rh = 30, 34
    H = top + rh * len(rows) + 50
    body = []
    y_end = top + rh * len(rows)
    for v in (0.001, 0.01, 0.05, 0.1, 0.3):
        body.append(L(sx(v), top - 6, sx(v), y_end, "gr"))
        body.append(T(sx(v), y_end + 15, f"{v:g}", "tk", "middle"))
    body.append(L(x0, y_end, x1, y_end, "ax"))
    body.append(L(sx(0.05), top - 10, sx(0.05), y_end, "alpha"))
    body.append(T(sx(0.05) + 4, top - 12, "α = 0.05", "lbs"))
    body.append(T((x0 + x1) / 2, y_end + 32, "p (log scale) · each dot one bootstrap seed (0–4), n_boot 1000, sorted export", "lbm", "middle"))
    for i, (lab, ps, out) in enumerate(rows):
        y = top + rh * i + rh / 2
        body.append(T(x0 - 10, y + 4, lab, "lb", "end"))
        body.append(T(x1 + 10, y + 4, out, "lb"))
        for j, p in enumerate(ps):
            jitter = ((j % 5) - 2) * 2.4
            hit = p < 0.05
            seed = seeds[j % 5]
            body.append(C(sx(max(p, 0.0003)), y + jitter, 4, "dot-win" if hit else "dot",
                          f"{lab}, seed {seed}: p {p:.4f}"))
    return svg(W, H, "Gate p-values and Holm-adjusted metric p-values across five bootstrap seeds for the PhAIL pairs", body)


# ---------------------------------------------------------------------------------------------
# LIBERO human reference: T_success (goal check) vs duration_s, both suites
# ---------------------------------------------------------------------------------------------

def ecdf_pts(xs, sx, sy):
    xs = sorted(xs)
    n = len(xs)
    steps = [[0.0, 0.0]] + [[x, (i + 1) / n] for i, x in enumerate(xs)]
    return steps


def human_ref() -> str:
    d = load("summarizer_data.json")["human"]
    tab = {(r["suite"], r["libero_task_id"]): r for r in csv.DictReader(open(ART / "s3_human_reference_table.csv"))}
    W, H = 680, 290
    body = []
    panels = [("libero_spatial", 70, 320, 12.0, [0, 2, 4, 6, 8, 10, 12]), ("libero_10", 400, 650, 30.0, [0, 5, 10, 15, 20, 25, 30])]
    y0, y1 = 40, 230
    sy = Scale(0, 1, y1, y0)
    for s, x0, x1, xm, xt in panels:
        sx = Scale(0, xm, x0, x1)
        axes(body, sx, sy, xt, [0, 0.5, 1.0], x0, x1, y0, y1, "seconds", "fraction of 500 demos" if x0 < 100 else "", fmt_g, fmt_g)
        r = tab[(s, "all")]
        body.append(T(x0, 26, f"{s} · HRT inflation {float(r['HRT_bias_factor_duration_over_T']):.3f}×", "tl"))
        body.append(P(step_path(ecdf_pts(d[s]["duration_s"], sx, sy), sx, sy, xm), "s-b"))
        body.append(P(step_path(ecdf_pts(d[s]["T_success_s"], sx, sy), sx, sy, xm), "s-a"))
        for key, cls, dy in (("median_T", "f-a", -8), ("median_D", "f-b", 14)):
            v = d[s][key]
            body.append(C(sx(v), sy(0.5), 3.5, cls, f"{s} median {'T_success' if key == 'median_T' else 'duration_s'} {v:.2f} s"))
        body.append(T(sx(d[s]["median_T"]) - 6, sy(0.5) - 6, f"{d[s]['median_T']:.2f} s", "val", "end"))
        body.append(T(sx(d[s]["median_D"]) + 6, sy(0.5) + 12, f"{d[s]['median_D']:.2f} s", "val"))
    body.append(L(80, 278, 106, 278, "s-a"))
    body.append(T(112, 282, "T_success_s (LIBERO goal check, replayed)", "lbm"))
    body.append(L(400, 278, 426, 278, "s-b"))
    body.append(T(432, 282, "duration_s (old anchor: demo length)", "lbm"))
    return svg(W, H, "Human reference time-to-success from the goal-check replay versus demo duration, both LIBERO suites", body)


# ---------------------------------------------------------------------------------------------
# per-sub-task T for libero_10: median per operation with clustered 95% CI
# ---------------------------------------------------------------------------------------------

def subtask_T() -> str:
    st = list(csv.DictReader(open(ART / "s4_human_operation_stats.csv")))
    ops = {int(r["libero_task_id"]): r for r in csv.DictReader(open(ART / "s4_operations_table.csv")) if r["suite"] == "libero_10"}
    W = 680
    x0, x1 = 300, 640
    sx = Scale(3, 12, x0, x1)
    top, rh = 34, 26
    H = top + rh * len(st) + 56
    body = []
    yb = top + rh * len(st)
    for v in range(3, 13):
        body.append(L(sx(v), top - 4, sx(v), yb, "gr"))
        body.append(T(sx(v), yb + 15, str(v), "tk", "middle"))
    body.append(L(x0, yb, x1, yb, "ax"))
    body.append(T((x0 + x1) / 2, yb + 30, "median operation time (s) · 95% episode-clustered CI · 50 demos/task", "lbm", "middle"))
    body.append(T(16, 20, "task · goal operations (BDDL)", "tl"))

    def ci(s):
        a, b = json.loads(s)
        return float(a), float(b)

    for i, r in enumerate(sorted(st, key=lambda r: int(r["libero_task_id"]))):
        t = int(r["libero_task_id"])
        y = top + rh * i + rh / 2
        o = ops[t]
        oplist = [re.sub(r"[()]", "", s).split()[1].rsplit("_", 1)[0] for s in json.loads(o["operations"].replace("'", '"'))]
        maint = json.loads(o["maintained"].replace("'", '"'))
        lab = f"{t} · " + " + ".join(oplist) + ("  (stove on: maintained)" if maint else "")
        body.append(T(16, y + 4, lab[:52], "lb"))
        for k, cls, shape in ((1, "f-a", "c"), (2, "f-b", "d")):
            med = r[f"median_T_op{k}"]
            if not med:
                continue
            m = float(med)
            lo, hi = ci(r[f"median_T_op{k}_ci"])
            dy = -4 if k == 1 else 4
            body.append(L(sx(lo), y + dy, sx(hi), y + dy, "ci" if k == 2 else "ci-hl"))
            tip = f"task {t} operation {k}: median {m:.2f} s [{lo:.2f}, {hi:.2f}]"
            if shape == "c":
                body.append(C(sx(m), y + dy, 4.5, cls, tip))
            else:
                x, yy = sx(m), y + dy
                body.append(f'<path d="M{x:.1f},{yy - 5:.1f} L{x + 5:.1f},{yy:.1f} L{x:.1f},{yy + 5:.1f} L{x - 5:.1f},{yy:.1f} Z" class="{cls}"><title>{esc(tip)}</title></path>')
    body.append(C(316, H - 10, 4.5, "f-a"))
    body.append(T(326, H - 6, "operation 1", "lbm"))
    x = 410
    body.append(f'<path d="M{x},{H - 15} L{x + 5},{H - 10} L{x},{H - 5} L{x - 5},{H - 10} Z" class="f-b"/>')
    body.append(T(x + 10, H - 6, "operation 2", "lbm"))
    return svg(W, H, "Per-sub-task human time to success for the ten libero_10 tasks", body)


# ---------------------------------------------------------------------------------------------
# SmolVLA harness rollouts: KM per task vs the human reference, censored at 60 s
# ---------------------------------------------------------------------------------------------

def smolvla_km() -> str:
    d = load("summarizer_data.json")
    W, H = 680, 300
    body = []
    y0, y1 = 44, 230
    sy = Scale(0, 1, y1, y0)
    panels = [("libero_spatial/0", 70, 320, "libero_spatial:0 · bowl to plate"),
              ("libero_10/8", 400, 650, "libero_10:8 · two moka pots on the stove")]
    for cell, x0, x1, title in panels:
        sm = d["smolvla"][cell]
        sx = Scale(0, 60, x0, x1)
        axes(body, sx, sy, [0, 10, 20, 30, 40, 50, 60], [0, 0.25, 0.5, 0.75, 1.0], x0, x1, y0, y1,
             "simulated seconds (20 Hz steps / 20)", "F(t), whole-task (family W)" if x0 < 100 else "", fmt_g, fmt_g)
        body.append(T(x0, 20, title, "tl"))
        n = len(sm["rows"])
        n_ok = sum(r["event"] for r in sm["rows"])
        body.append(T(x0, 36, f"{n_ok}/{n} succeeded · F(30 s) {sm['F30_W']:.2f} · RMST(60) {sm['rmst60_W']:.1f} s", "lbs"))
        body.append(P(step_path(ecdf_pts(d["human_task"][cell], sx, sy), sx, sy, 60), "s-c"))
        f60 = sm["km_W"][-1][1]
        body.append(R(sx(0), sy(1.0), sx(60) - sx(0), sy(f60) - sy(1.0), "shade"))
        body.append(T(sx(45), (sy(1.0) + sy(f60)) / 2 + 4, f"censored at τ = 60 s: {1 - f60:.0%}", "lbs", "middle"))
        body.append(P(step_path(sm["km_W"], sx, sy, 60), "s-a"))
        for r in sm["rows"]:
            if r["event"]:
                body.append(C(sx(r["T_s"]), sy(0.0) + 8, 2.5, "f-a", f"{r['rollout_id']}: success at {r['T_s']:.2f} s"))
        n_c = n - n_ok
        body.append(L(sx(60), sy(f60) - 7, sx(60), sy(f60) + 7, "cens"))
        body.append(T(sx(60) - 4, sy(f60) + 18, f"{n_c} censored", "lbs", "end"))
        body.append(L(sx(30), y0, sx(30), y1, "ref"))
    body.append(L(80, 288, 106, 288, "s-a"))
    body.append(T(112, 292, "SmolVLA rollouts (KM, 10 init states)", "lbm"))
    body.append(L(400, 288, 426, 288, "s-c"))
    body.append(T(432, 292, "50 human demos, same task", "lbm"))
    return svg(W, H, "Kaplan-Meier time-to-success of 20 real SmolVLA rollouts on two LIBERO tasks, against the human reference", body)


# ---------------------------------------------------------------------------------------------
# per-rollout returns: episode x metric table with sequential shading (HTML)
# ---------------------------------------------------------------------------------------------

def returns_table() -> str:
    d = load("summarizer_data.json")["returns"]
    cols = [("ret_f30", "ret_f30", 1, "{:.0f}"), ("ret_rmst60", "ret_rmst60 (s)", 1, "{:.2f}"),
            ("ret_auc_expected", "ret_auc_expected", 1, "{:.2f}"), ("ret_auc_sampled", "ret_auc_sampled", 1, "{:.1f}"),
            ("ret_hrt60_lin", "ret_hrt60_lin", 1, "{:.4f}"), ("ret_censored60", "ret_censored60", -1, "{:.0f}")]
    rng = {c: (min(r[c] for r in d), max(r[c] for r in d)) for c, *_ in cols}
    out = ['<div class="scroll"><table class="heat">', "<thead><tr><th>rollout (cell / init / seed)</th><th class=\"num\">T (s)</th><th>failure type</th>"]
    for c, lab, sgn, _ in cols:
        out.append(f'<th class="num">{esc(lab)}<br><span class="dir">{"↑" if sgn > 0 else "↓"} better</span></th>')
    out.append("</tr></thead><tbody>")
    for r in d:
        rid = r["rollout_id"].replace("/seed", " / s").replace("/init", " / i")
        out.append(f"<tr><td><code>{esc(rid)}</code></td><td class=\"num\">{'—' if r['T_s'] is None else f'{r['T_s']:.2f}'}</td>"
                   f"<td>{esc(r['failure_type'])}</td>")
        for c, lab, sgn, f in cols:
            lo, hi = rng[c]
            g = 0.0 if hi == lo else (r[c] - lo) / (hi - lo)
            if sgn < 0:
                g = 1 - g
            pct = round(8 + 52 * g)
            out.append(f'<td class="num" style="background: color-mix(in srgb, var(--accent) {pct}%, transparent)" '
                       f'title="{esc(c)} = {r[c]:.6g}">{f.format(r[c])}</td>')
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


# ---------------------------------------------------------------------------------------------
# returns == verdict functional (AC6.10 table, HTML)
# ---------------------------------------------------------------------------------------------

def identity_table() -> str:
    rows = list(csv.DictReader(open(ART / "s6_ac6_10_returns_vs_verdict.csv")))
    out = ['<div class="scroll"><table>', "<thead><tr><th>eval run</th><th>family</th><th>metric</th><th>relation</th>"
           "<th class=\"num\">returns side</th><th class=\"num\">verdict side</th><th class=\"num\">|diff|</th></tr></thead><tbody>"]
    for r in rows:
        run = "spatial:0 (e138fec9)" if "spatial0" in r["run"] else "libero_10:8 (e92604a8)"
        out.append(f"<tr><td>{run}</td><td>{esc(r['family'])}</td><td><code>{esc(r['metric'])}</code></td><td><code>{esc(r['relation'])}</code></td>"
                   f"<td class=\"num\">{float(r['returns_side']):.6g}</td><td class=\"num\">{float(r['verdict_side']):.6g}</td>"
                   f"<td class=\"num\">{float(r['abs_diff']):.1e}</td></tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


# ---------------------------------------------------------------------------------------------
# the metric registry (HTML)
# ---------------------------------------------------------------------------------------------

def registry_table() -> str:
    ents = load("s6_metrics_registry.json")
    pill = {"built": "good", "built (linearized)": "good", "built (descriptive)": "good", "built (adapted)": "good",
            "PhAIL validation": "accent", "significance gate (S7)": "good", "visualization": "", "built (sanity line)": "good",
            "scoring only": "", "projection only": "warn", "adapted, scoring only": "warn", "adapted": "warn", "replaced": "warn",
            "not transferable": "bad", "not built": "bad"}
    out = ['<div class="scroll"><table class="reg">', "<thead><tr><th>PoC metric</th><th>family</th><th>transfer status</th>"
           "<th>return column(s) / verdict</th><th>reason (from the registry)</th></tr></thead><tbody>"]
    for e in ents:
        st = e.get("transfer_status") or ""
        cls = pill.get(st, "bad" if st.startswith("not") else "")
        rc = ", ".join(f"<code>{esc(c)}</code>" for c in e.get("return_columns") or [])
        ver = e.get("verdict")
        vtxt = f"verdict: <code>{esc(ver['compute'])}</code>" if isinstance(ver, dict) else ""
        out.append(f"<tr><td>{esc(e.get('metric'))}</td><td><code>{esc(e.get('family'))}</code></td>"
                   f"<td><span class=\"pill {cls}\">{esc(st)}</span></td><td>{rc}{'<br>' if rc and vtxt else ''}{vtxt}</td>"
                   f"<td>{esc(e.get('reason') or '')}</td></tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


# ---------------------------------------------------------------------------------------------
# PhAIL verdict table (HTML): seed-0 numbers + the five-seed ranges
# ---------------------------------------------------------------------------------------------

def verdict_table() -> str:
    vs = load("s7_phail_verdicts.json")
    sw = load("s7_ac7_1_seed_sweep.json")
    res, seeds = sw["results"], sw["seeds"]
    expected = {"OpenPI-GR00T": "(iii)", "OpenPI-ACT": "(iii)", "GR00T-ACT": "(ii) in ≥ 4/5 (r3)", "OpenPI-SmolVLA": "(i)",
                "GR00T-SmolVLA": "(i)", "ACT-SmolVLA": "(i)"}
    out = ['<div class="scroll"><table>', "<thead><tr><th>pair</th><th class=\"num\">macro-KS D̄</th><th class=\"num\">gate p, seed 0</th>"
           "<th class=\"num\">gate p, seeds 0–4</th><th>Holm-significant metrics (seed 0)</th><th>outcome, seed 0</th>"
           "<th>outcomes over 5 seeds</th><th>expected (AC7.1)</th></tr></thead><tbody>"]
    for v in vs:
        a, b = v["arms"]
        key = f"{a}-{b}"
        ps = [res[f"{key}-seed{s}"]["gate_p"] for s in seeds]
        outs = [res[f"{key}-seed{s}"]["outcome"] for s in seeds]
        sig = [f"{m.replace('phail_', '')} → {mm['favours']}" for m, mm in v["metrics"].items() if mm["significant"]]
        cnt = {o: outs.count(o) for o in dict.fromkeys(outs)}
        ostr = ", ".join(f"({o}) {n}/5" for o, n in cnt.items())
        fires = "fires" if v["gate"]["fires"] else "does not fire"
        out.append(f"<tr><td><b>{a} vs {b}</b></td><td class=\"num\">{v['gate']['D']:.4f}</td><td class=\"num\">{v['gate']['p']:.4f} ({fires})</td>"
                   f"<td class=\"num\">{min(ps):.4f}–{max(ps):.4f}</td><td>{esc('; '.join(sig) or 'none')}</td>"
                   f"<td><b>({v['outcome']})</b>{(' ' + v['direction']) if v.get('direction') else ''}</td><td>{esc(ostr)}</td>"
                   f"<td>{esc(expected[key])}</td></tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


# ---------------------------------------------------------------------------------------------
# libero_10:8 operation timeline with the capped operation returns
# ---------------------------------------------------------------------------------------------

def op_timeline() -> str:
    ops = load("summarizer_data.json")["op_returns"]
    by = {}
    for o in ops:
        by.setdefault(o["rollout_id"], []).append(o)
    W = 680
    x0, x1 = 130, 530
    sx = Scale(0, 1200, x0, x1)
    top, rh = 34, 22
    H = top + rh * len(by) + 58
    body = []
    yb = top + rh * len(by)
    for v in range(0, 1201, 200):
        body.append(L(sx(v), top - 4, sx(v), yb, "gr"))
        body.append(T(sx(v), yb + 15, f"{v}", "tk", "middle"))
    body.append(L(x0, yb, x1, yb, "ax"))
    body.append(T((x0 + x1) / 2, yb + 30, "policy step (20 Hz; 1200 steps = τ_episode = 60 s)", "lbm", "middle"))
    body.append(T(16, 20, "rollout", "tl"))
    body.append(T(x1 + 10, 20, "return(s), s", "tl"))
    for i, (rid, os_) in enumerate(sorted(by.items())):
        y = top + rh * i + 4
        body.append(T(16, y + 11, rid.replace("libero_10/8/", ""), "lb"))
        rets = []
        for o in sorted(os_, key=lambda o: o["op_k"]):
            w = sx(o["t_end"]) - sx(o["t_start"])
            cls = "bar-done" if o["event"] else "bar-open"
            tip = (f"{rid} op {o['op_k']}: steps {o['t_start']}–{o['t_end']} ({o['T_op_s']:.2f} s), "
                   f"{'completed' if o['event'] else 'unfinished attempt (censored)'}, return {o['ret']:.2f}")
            body.append(f'<rect x="{sx(o["t_start"]) + 1:.1f}" y="{y:.1f}" width="{max(w - 2, 1):.1f}" height="14" rx="3" class="{cls}"><title>{esc(tip)}</title></rect>')
            if w > 34:
                body.append(T(sx(o["t_start"]) + w / 2, y + 11, f"op {o['op_k']}", "barlab", "middle"))
            rets.append(f"{o['ret']:.2f}")
        body.append(T(x1 + 10, y + 11, " , ".join(rets), "lbs"))
    body.append(R(130, H - 18, 22, 12, "bar-done", 3))
    body.append(T(158, H - 8, "completed operation: return −min(T_op, 60)", "lbm"))
    body.append(R(420, H - 18, 22, 12, "bar-open", 3))
    body.append(T(448, H - 8, "unfinished attempt: return −60", "lbm"))
    return svg(W, H, "Operation segments of the ten SmolVLA rollouts on libero_10 task 8 with their capped operation returns", body)


# ---------------------------------------------------------------------------------------------
# HRT linearization error (AC6.2, synthetic)
# ---------------------------------------------------------------------------------------------

def hrt_lin() -> str:
    rows = list(csv.DictReader(open(ART / "s6_hrt_linearization_curve.csv")))
    W, H = 680, 260
    x0, x1, y0, y1 = 110, 620, 30, 200
    sx = Scale(0, 0.11, x0, x1)
    sy = Scale(0, 0.11, y1, y0)
    body = []
    axes(body, sx, sy, [0, 0.02, 0.04, 0.06, 0.08, 0.10], [0, 0.025, 0.05, 0.075, 0.10], x0, x1, y0, y1,
         "per-cell relative change in model time |δ| (SYNTHETIC perturbations)", "relative error of the linear prediction",
         lambda v: f"{v:.0%}", lambda v: f"{v:.1%}", ylab_dx=58)
    body.append(L(x0, sy(0.10), x1, sy(0.10), "alpha"))
    body.append(T(x1, sy(0.10) - 6, "AC6.2 criterion: 10%", "lbs", "end"))
    for r in rows:
        dlt, err = abs(float(r["delta"])), float(r["rel_err"])
        uni = r["mode"] == "uniform"
        jitter = -5 if float(r["delta"]) < 0 else 5
        body.append(C(sx(dlt) + jitter, sy(err), 4.5, "f-a" if uni else "dot",
                      f"δ {float(r['delta']):+.2f} ({r['mode']}): |ΔHRT|/HRT {abs(float(r['rel_change_hrt'])):.2%}, rel. error {err:.2%}"))
    body.append(C(90, 245, 4.5, "f-a"))
    body.append(T(100, 249, "all cells slowed/sped up by δ", "lbm"))
    body.append(C(320, 245, 4.5, "dot"))
    body.append(T(330, 249, "one cell of three changed by δ (worst in-scope case: 9.51%)", "lbm"))
    return svg(W, H, "HRT linearization error against perturbation size on synthetic data", body)


# ---------------------------------------------------------------------------------------------
# null calibration (AC7.2)
# ---------------------------------------------------------------------------------------------

def null_cal() -> str:
    lines = (ART / "s7_ac7_2_null_calibration_stdout.txt").read_text().splitlines()
    rows = []
    for ln in lines:
        m = re.search(r"split\) (\w+): .* mean p ([\d.]+), rejection rate at α=0.05 ([\d.]+) ± ([\d.]+)", ln)
        if m:
            rows.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    W, H = 680, 230
    x0, x1 = 120, 540
    sx = Scale(0, 0.14, x0, x1)
    top, rh = 30, 30
    yb = top + rh * len(rows)
    body = [R(sx(0.02), top - 8, sx(0.10) - sx(0.02), yb - top + 8, "shade")]
    body.append(T(sx(0.06), top - 12, "plan band [0.02, 0.10] at 200 trials", "lbs", "middle"))
    for v in (0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14):
        body.append(T(sx(v), yb + 15, f"{v:.2f}", "tk", "middle"))
        body.append(L(sx(v), yb, sx(v), yb + 4, "ax"))
    body.append(L(x0, yb, x1, yb, "ax"))
    body.append(L(sx(0.05), top - 8, sx(0.05), yb, "alpha"))
    body.append(T((x0 + x1) / 2, yb + 30, "rejection rate of the macro-KS gate on same-model random halves (true null), α = 0.05", "lbm", "middle"))
    for i, (m, mp, r, se) in enumerate(rows):
        y = top + rh * i + rh / 2
        body.append(T(x0 - 10, y + 4, m, "lb", "end"))
        body.append(L(sx(max(r - se, 0)), y, sx(r + se), y, "ci-hl"))
        body.append(C(sx(r), y, 4.5, "f-a", f"{m}: rejection rate {r:.3f} ± {se:.3f} (1.96 SE), mean p {mp:.3f}"))
        body.append(T(x1 + 8, y + 4, f"{r:.3f} (mean p {mp:.2f})", "lbs"))
    return svg(W, H, "Null calibration of the macro-KS gate on PhAIL's data: rejection rates with 1.96 SE", body)


# ---------------------------------------------------------------------------------------------
# per-subset normalizer (AC8.2): saved tensors vs subset stats vs full stats, log scale
# ---------------------------------------------------------------------------------------------

def normalizer() -> str:
    lines = (ART / "s8_ac8_2_3_stdout.txt").read_text().splitlines()
    rows = []
    for ln in lines:
        m = re.search(r"step_5_normalizer_processor\.safetensors: (\S+) \(float32\) max \|saved - subset stats\| = (\S+); max \|saved - full stats.json\| = (\S+)", ln)
        if m:
            rows.append((m.group(1), float(m.group(2)), float(m.group(3))))
    W, H = 680, 210
    x0, x1 = 170, 620
    sx = Scale(1e-10, 1e-1, x0, x1, log=True)
    top, rh = 30, 30
    yb = top + rh * len(rows)
    body = []
    for e in range(-10, 0):
        v = 10 ** e
        body.append(L(sx(v), top - 6, sx(v), yb, "gr"))
        body.append(T(sx(v), yb + 15, f"1e{e}", "tk", "middle"))
    body.append(L(x0, yb, x1, yb, "ax"))
    body.append(L(sx(1e-6), top - 10, sx(1e-6), yb, "alpha"))
    body.append(T(sx(1e-6) + 4, top - 12, "AC8.2 bound 1e-6", "lbs"))
    body.append(T((x0 + x1) / 2, yb + 30, "max |saved checkpoint tensor − reference| (log scale)", "lbm", "middle"))
    for i, (k, a, b) in enumerate(rows):
        y = top + rh * i + rh / 2
        body.append(T(x0 - 10, y + 4, k, "lb", "end"))
        body.append(L(sx(a), y, sx(b), y, "gr"))
        body.append(C(sx(a), y, 4.5, "f-a", f"{k}: vs the subset's own stats {a:.2e}"))
        body.append(C(sx(b), y, 4.5, "f-b", f"{k}: vs the full dataset's stats.json {b:.2e}"))
    body.append(C(180, H - 10, 4.5, "f-a"))
    body.append(T(190, H - 6, "vs the 40-episode subset's own stats", "lbm"))
    body.append(C(420, H - 10, 4.5, "f-b"))
    body.append(T(430, H - 6, "vs the full v0 stats.json", "lbm"))
    return svg(W, H, "Saved SmolVLA normalizer tensors match the subset statistics, not the full-dataset statistics", body)


# ---------------------------------------------------------------------------------------------
# S0: the leak channels before and after
# ---------------------------------------------------------------------------------------------

def leak_s0() -> str:
    before = json.loads((ART / "s0_check_blind_v1_before.txt").read_text().split("\ncheck_blind")[0])
    vb = load("s0_verify_blind.json")["AC0.3"]
    W, H = 680, 262
    body = []
    x0, x1 = 30, 650
    sx = Scale(0, 1000, x0, x1)
    body.append(T(x0, 22, "v1 as first derived (delete_episodes + merge_datasets), by episode_index", "tl"))
    body.append(R(sx(0), 34, sx(900) - sx(0), 26, "seg-plain", 3, title="900 unedited episodes"))
    body.append(R(sx(900), 34, sx(1000) - sx(900), 26, "seg-hl", 3, title="episode_index 900–999 = the 100 labelled (corrupted) episodes"))
    for v in (0, 300, 600, 900, 999):
        body.append(T(sx(v), 74, str(v), "tk", "middle"))
    body.append(T(sx(450), 51, "900 unedited episodes, in v0 order", "lbs", "middle"))
    body.append(T(sx(1000), 92, "100 edited = the labelled set", "lb", "end"))
    n_diff = before["tables"]["version_episodes"]["differing_rows"]["episode_index"]
    body.append(T(x0, 112, f"also: parent_content_hash differs in exactly those 100 rows; episode_index differs from v0 in {n_diff} rows", "lbs"))
    body.append(T(x0, 126, "and the edited episodes sit in their own data files (73–80 of 81)", "lbs"))
    y = 156
    body.append(T(x0, y, f"derive_version now: one writer pass in parent order (scratch test, {vb['n_episodes']} episodes, 3 edits)", "tl"))
    sx2 = Scale(0, vb["n_episodes"], x0, x1)
    for i in range(vb["n_episodes"]):
        hl = i in vb["edited_child_index"]
        body.append(R(sx2(i) + 1, y + 12, sx2(i + 1) - sx2(i) - 2, 26, "seg-hl" if hl else "seg-plain", 3,
                      title=f'episode_index {i}{" (edited; parent index " + str(i) + ")" if hl else ""}'))
        body.append(T((sx2(i) + sx2(i + 1)) / 2, y + 52, str(i), "tk", "middle"))
    body.append(T(x0, y + 78, f"episode_index equal to parent's: {vb['episode_index_equal']}/{vb['n_episodes']}; edited at "
                  f"{vb['edited_child_index']} = parent {vb['edited_parent_index']}; check_blind(child) = {vb['check_blind']}", "lbs"))
    body.append(T(x0, y + 94, "v1 is not re-derived (owner ruling A): check_blind(v1) = ['episode_index']; fenced", "lbs"))
    return svg(W, H, "The answer-key leak: where the edited episodes sat in v1, and how derivations keep parent order now", body)


# ---------------------------------------------------------------------------------------------
# S9: the real lineage graph of the one-command chain (from s9_lineage_graph.mmd)
# ---------------------------------------------------------------------------------------------

def lineage_s9() -> str:
    txt = (ART / "s9_lineage_graph.mmd").read_text()
    labels = {m.group(1): m.group(2) for m in re.finditer(r'^\s+(n[0-9a-f]+)[\[\(]+"([^"]+)"', txt, re.M)}
    edges = re.findall(r"(n[0-9a-f]+) -->\|(\w+)\| (n[0-9a-f]+)", txt)

    def lab(nid):
        return labels[nid]  # KeyError if the node is not in the real graph

    W, H = 680, 686
    body = [arrow_defs("lg")]
    pos = {}

    def node(nid, x, y, w, kind, hl=False, sub=None):
        pos[nid] = (x, y, w)
        cls = "box-ent" if kind == "ent" else ("box-hl" if hl else "box")
        h = 30 if sub is None else 40
        body.append(R(x - w / 2, y - 15, w, h, cls, 6 if kind != "ent" else 14))
        body.append(T(x, y + 4, lab(nid), "nlab" if kind != "ent" else "nent", "middle"))
        if sub:
            body.append(T(x, y + 17, sub, "nsub", "middle"))

    def edge(a, b, hl=True, label=None):
        xa, ya, _ = pos[a]
        xb, yb, _ = pos[b]
        p0 = (xa, ya + 16 if yb > ya else ya - 16)
        p1 = (xb, yb - 17 if yb > ya else yb + 17)
        body.append(f'<path d="M{p0[0]:.1f},{p0[1]:.1f} L{p1[0]:.1f},{p1[1]:.1f}" class="{"edge-hl" if hl else "edge"}" '
                    f'marker-end="url(#lg-{"aht" if hl else "ah"})"/>')
        if label:
            body.append(T((p0[0] + p1[0]) / 2 + 5, (p0[1] + p1[1]) / 2 + 3, label, "elab"))

    ids = {v: k for k, v in labels.items()}

    def find(s, nth=0):
        hits = [k for k, v in labels.items() if v == s]
        return hits[nth]

    raw = find("raw LIBERO HDF5 x20")
    build = find("build: build-v0")
    v0 = find("version v0 3c4dca68")
    v0all = find("subset v0/all 58b3c415")
    split = find("split: split-v0")
    ref = find("subset v0/reference b6a72d5e")
    cand = find("subset v0/candidate 8ee096b7")
    sel = find("select: select-drop-random-5-s0 [eval-smoke]")
    child = find("subset  e470d671")
    tr_a = [k for k in labels if labels[k] in ("train: train-ref [eval-smoke]", "train: train-act-2step-persubset-v0ref")]
    tr_b = [k for k in labels if labels[k] in ("train: train-ref-drop-random-5 [eval-smoke]", "train: train-act-2step-persubset-v0ref-drop5")]
    ck_a = find("checkpoint  cdee42fb")
    ck_b = find("checkpoint  fac999b0")
    ev_a = find("eval: eval-train-ref [eval-smoke]")
    ev_b = find("eval: eval-train-ref-drop-random-5 [eval-smoke]")
    ro_a = find("rollouts  4413ecad")
    ro_b = find("rollouts  3249ff35")
    ver = find("verify: verify-ref-vs-drop-random-5 [eval-smoke]")
    vd = find("verdict  730cc2c1")
    re_a = find("returns: returns-train-ref [eval-smoke]")
    re_b = find("returns: returns-train-ref-drop-random-5 [eval-smoke]")
    rp_a = find("returns  18fb45c6")
    rp_b = find("returns  d0209dea")
    assert {(raw, "source", build), (ck_a, "checkpoint", ev_a), (ro_a, "rollouts_a", ver), (ro_b, "rollouts_b", ver),
            (ver, "produces", vd)} <= set(edges), "layout edges must exist in the real graph"

    node(raw, 340, 24, 190, "ent")
    node(build, 340, 74, 160, "run", True)
    node(v0, 250, 124, 160, "ent")
    node(v0all, 440, 124, 170, "ent")
    node(split, 440, 174, 140, "run", True)
    node(ref, 340, 224, 200, "ent")
    node(cand, 560, 224, 190, "ent")
    node(sel, 520, 280, 290, "run", True, "×2 runs (two chains), same child")
    node(child, 520, 336, 140, "ent")
    body.append(R(20, 365, 300, 50, "box-hl", 6))
    body.append(T(170, 384, "train: train-ref [eval-smoke] ×2", "nlab", "middle"))
    body.append(T(170, 400, "+ train-act-2step-persubset-v0ref (S8)", "nlab", "middle"))
    pos["TA"] = (170, 390, 300)
    body.append(R(360, 365, 300, 50, "box-hl", 6))
    body.append(T(510, 384, "train: train-ref-drop-random-5 [eval-smoke] ×2", "nlab", "middle"))
    body.append(T(510, 400, "+ train-act-2step-persubset-v0ref-drop5 (S8)", "nlab", "middle"))
    pos["TB"] = (510, 390, 300)
    node(ck_a, 170, 446, 160, "ent", sub=None)
    node(ck_b, 510, 446, 160, "ent")
    node(ev_a, 170, 496, 220, "run", True)
    node(ev_b, 510, 496, 280, "run", True)
    node(ro_a, 170, 546, 150, "ent")
    node(ro_b, 510, 546, 150, "ent")
    node(ver, 340, 600, 300, "run", True)
    node(vd, 600, 600, 130, "ent")
    edge(raw, build, label="source")
    edge(build, v0)
    edge(build, v0all)
    edge(v0all, split, label="population")
    edge(split, ref)
    edge(split, cand, hl=False)
    edge(ref, sel, label="parent")
    edge(sel, child)
    # subset -> train groups
    body.append(f'<path d="M{pos[ref][0] - 60:.1f},239 L170,364" class="edge-hl" marker-end="url(#lg-aht)"/>')
    body.append(T(200, 300, "training (40 eps)", "elab"))
    body.append(f'<path d="M520,352 L510,364" class="edge-hl" marker-end="url(#lg-aht)"/>')
    body.append(T(530, 360, "training (35)", "elab"))
    body.append(f'<path d="M170,416 L170,430" class="edge-hl" marker-end="url(#lg-aht)"/>')
    body.append(f'<path d="M510,416 L510,430" class="edge-hl" marker-end="url(#lg-aht)"/>')
    edge(ck_a, ev_a, label="checkpoint")
    edge(ck_b, ev_b, label="checkpoint")
    edge(ev_a, ro_a)
    edge(ev_b, ro_b)
    edge(ro_a, ver, label="rollouts_a")
    edge(ro_b, ver, label="rollouts_b")
    body.append(f'<path d="M490,600 L534,600" class="edge-hl" marker-end="url(#lg-aht)"/>')
    # returns runs beside the verdict
    body.append(T(16, 648, f"beside the verdict (same rollouts in): {lab(re_a)} → {lab(rp_a)} + operation_returns,", "nsub"))
    body.append(T(16, 662, f"returns_meta, metrics_registry; {lab(re_b)} → {lab(rp_b)} + 3 more", "nsub"))
    body.append(T(16, 678, "bold boxes: MLflow runs; grey pills: content-addressed entities (sha256 prefix)", "nsub"))
    return svg(W, H, "The real lineage graph of the one-command evaluation chain, from the raw LIBERO files to the verdict", body)


# ---------------------------------------------------------------------------------------------
# dataflow: where the feature sits (static diagram, real module/function names)
# ---------------------------------------------------------------------------------------------

def dataflow() -> str:
    W = 680
    rows = [
        ("inputs", "data/lerobot/libero_spatial_10/ + data/lineage/", "v0: 1000 LIBERO demos; versions, subsets, MLflow", "§10 Foundation", "p. 20 (data+lineage)"),
        ("inputs", "data/phail/cohort_mixed.parquet", "990 real-robot episodes · phail_export_cohort.py", "§3.1, §3.5 B PhAIL data", "pp. 6, 8"),
        ("mod", "vlapoc/metrics_registry.py", "ENTRIES (28) · families W, O, PhAIL", "Declared targets", "§3.4 p. 7 · §4.4 p. 10"),
        ("mod", "third_party/phail_stats.py → evaluation.py", "kaplan_meier · F_at · rmst · hrt · macro_ks", "§3.2 primitives", "p. 6 · §10 p. 20"),
        ("check", "scripts/phail_validate.py", "PhAIL's code vs ours on the same cohort", "§3.3 headline evidence", "p. 7 · §4.1 p. 9"),
        ("mod", "libero_goals.py + human_reference.py", "replay_goal_trace · operations_from_trace", "§3.5 A human reference,", "per sub-task T, pp. 7–8"),
        ("mod", "subset_stats.py + lerobot_train_entry.py", "aggregate_stats (LeRobot) · install_subset_stats", "(implicit) train", "owner decision 3"),
        ("mod", "vlapoc/rollouts.py + runs.eval_run", "ProbedLiberoEnv · run_rollouts → rollouts.parquet", "Step 2 Evaluation", "§9.1 p. 18"),
        ("mod", "vlapoc/returns.py", "rollout_returns (W) · operation_returns (O)", "Step 3 input: one return", "per metric, p. 19 · §4.3 p. 10"),
        ("mod", "vlapoc/verdict.py", "compare_policies: gate → per-metric CIs + Holm", "Step 4 verification gate", "p. 19 · §4.4 p. 10 · §2.6 p. 5"),
        ("guard", "vlapoc/lineage.py (S0)", "compare_blind · check_blind · assert_blind", "Blind methods under test", "§2.1 item 4, p. 3"),
    ]
    rh, top = 50, 34
    body = [arrow_defs("df")]
    body.append(T(16, 20, "code (this feature), in data-flow order", "tl"))
    body.append(T(486, 20, "PoC part it implements", "tl"))
    for i, (kind, f, fn, poc1, poc2) in enumerate(rows):
        y = top + rh * i
        cls = {"inputs": "box-ent", "mod": "box", "check": "box-hl", "guard": "box"}[kind]
        body.append(R(16, y, 446, 40, cls, 8))
        body.append(T(26, y + 16, f, "nfile"))
        body.append(T(26, y + 31, fn, "nsub"))
        body.append(R(486, y, 178, 40, "box", 8))
        body.append(T(496, y + 17, poc1, "nlab"))
        body.append(T(496, y + 31, poc2, "nsub"))
        body.append(L(462, y + 20, 486, y + 20, "conn"))
        if i < len(rows) - 2:
            body.append(f'<path d="M240,{y + 40} L240,{y + rh}" class="edge" marker-end="url(#df-ah)"/>')
    yb = top + rh * len(rows)
    body.append(T(16, yb + 6, "scripts/lineage_dry_run.py --eval-smoke: select → train ×2 → eval ×2 → returns ×2 → verify → trace()", "nsub"))
    return svg(W, yb + 16, "Dataflow of the evaluation layer: modules and functions mapped to the PoC architecture steps", body)


FIGS = {
    "dataflow": dataflow, "phail_slope": phail_slope, "phail_km_cross": phail_km_cross, "verdict_forest": verdict_forest,
    "seed_sweep": seed_sweep, "human_ref": human_ref, "subtask_T": subtask_T, "smolvla_km": smolvla_km,
    "returns_table": returns_table, "identity_table": identity_table, "registry_table": registry_table,
    "verdict_table": verdict_table, "op_timeline": op_timeline, "hrt_lin": hrt_lin, "null_cal": null_cal,
    "normalizer": normalizer, "leak_s0": leak_s0, "lineage_s9": lineage_s9,
}


def main() -> int:
    built = {}
    pages = sorted(FEAT.glob("*.html"))
    needed = {k for p in pages for k in FIGS if f"<!--FIG:{k}:start-->" in p.read_text()}
    for k in sorted(needed):
        built[k] = FIGS[k]()
    for page in pages:
        s = page.read_text()
        new = s
        for k, v in built.items():
            pat = re.compile(rf"(<!--FIG:{k}:start-->).*?(<!--FIG:{k}:end-->)", re.S)
            new = pat.sub(lambda m: m.group(1) + v + m.group(2), new)
        if new != s:
            page.write_text(new)
        print("updated" if new != s else "unchanged", page.name, [k for k in built if f"<!--FIG:{k}:start-->" in s])
    empty = [p.name for p in pages if re.search(r"<!--FIG:\w+:start--><!--FIG:\w+:end-->", p.read_text())]
    print("empty figure blocks:", empty)
    return 1 if empty else 0


if __name__ == "__main__":
    raise SystemExit(main())

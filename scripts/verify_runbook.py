#!/usr/bin/env python
"""Local checks of the server runbook skill ``.claude/skills/server-run/SKILL.md`` (dashboard S8).

Grounding: owner request 2026-10-22 ("an exact step-by-step runbook for a new claude session inside the actual gpu server"); vla-build ("commands exist"); the brief's rule
that server-specific steps are marked unverified. Run from the repo root::

    .venv/bin/python scripts/verify_runbook.py

AC8.1 every fenced shell command that names a ``scripts/*.py`` or ``python -m vlapoc.*`` entry exists and its subcommand's ``--help`` exits 0 here (n/n);
AC8.2 every dry-run count written in the skill (``<!-- count: key jobs trainings -->``) equals a fresh local ``server_pipeline.py --dry-run``, and the first-run command is the
v3 one in the skill and in ``server_pipeline.py``'s docstring; AC8.3 every step is tagged, the verified and unverified steps are counted and listed, and no unverified step states an
expected output without a label; AC8.4 at least 7 STOP AND ASK markers covering the required cases; AC8.5 ``CLAUDE.md`` is untouched by this batch (its ``git diff --stat`` equals the
recorded one); AC8.6 the frontmatter parses and the skill is in ``.claude/skills/``. Writes ``tracker/features/2026-10-22-dashboard/artifacts/verify_runbook.json``.
"""

from __future__ import annotations

import ast
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKILL = REPO / ".claude" / "skills" / "server-run" / "SKILL.md"
ART = REPO / "tracker" / "features" / "2026-10-22-dashboard" / "artifacts"
PY = str(REPO / ".venv" / "bin" / "python")
COUNT_ARGS = {  # key -> server_pipeline.py arguments (after --tag v3cand --dry-run)
    "round_a_v3": ["--subset", "v3/candidate", "--curation", "per_task", "--metrics", "f30", "hrt60_lin"],
    "round_a_v0_variant": ["--subset", "v0/candidate", "--curation", "per_task", "--metrics", "f30", "hrt60_lin"],
    "round_b_clusters_2": ["--subset", "v3/candidate", "--round-b", "--clusters", "clusters-v3cand", "--pick", "a", "b"],
    "round_b_clusters_2_matched": ["--subset", "v3/candidate", "--round-b", "--clusters", "clusters-v3cand", "--pick", "a", "b", "--matched-random"],
    "round_b_planted": ["--subset", "v3/candidate", "--round-b-planted"],
    "round_b_planted_no_oracle": ["--subset", "v3/candidate", "--round-b-planted", "--no-oracle"],
}
REQUIRED_STOPS = {"preflight failure": "preflight", "revision mismatch": "revision", "failed job": "failed job", "disk below threshold": "disk", "any needed code change": "code change",
                  "the Round B cluster pick": "cluster pick", "SYNTHETIC or SMOKE banner on real results": "banner"}


def fenced_blocks(text: str) -> list[tuple[str, str]]:
    """``[(step title, code)]`` for every fenced ``bash`` block, with the heading of the step it sits in."""
    out, step = [], ""
    for chunk in re.split(r"(?m)^(## .*)$", text):
        if chunk.startswith("## "):
            step = chunk[3:].strip()
        else:
            out += [(step, m.group(1)) for m in re.finditer(r"```bash\n(.*?)```", chunk, re.S)]
    return out


def commands(code: str) -> list[str]:
    """Logical commands of a block (backslash continuations joined, comments and blanks dropped)."""
    joined = re.sub(r"\\\n\s*", " ", code)
    return [ln.strip() for ln in joined.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def run(cmd: list[str], timeout: int = 180) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)[-300:]


def main() -> int:
    text = SKILL.read_text()
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}", flush=True)

    # --- AC8.1 ---
    seen, rows, external = set(), [], []
    for step, code in fenced_blocks(text):
        for snippet in re.findall(r'python -c "(.*?)"(?=\s*\n|\s*$)', code, re.S):  # inline python blocks can span lines
            try:
                ast.parse(snippet)
                rows.append((f"inline python in {step[:30]}", True, "inline python: syntax ok"))
            except SyntaxError as e:
                rows.append((f"inline python in {step[:30]}", False, f"inline python: {e}"))
        code = re.sub(r'python -c ".*?"(?=\s*\n|\s*$)', "", code, flags=re.S)
        for cmd in commands(code):
            try:
                toks = shlex.split(cmd)
            except ValueError:
                toks = cmd.split()
            toks = [t for t in toks if not re.match(r"^[A-Za-z_]+=", t)]
            script = next((t for t in toks if re.match(r"(\.venv/bin/python\S*|python)$", t)), None)
            if script is None:
                if toks and toks[0].endswith("streamlit"):
                    rows.append(("streamlit run dashboard/app.py", (REPO / "dashboard" / "app.py").is_file() and run([PY, "-m", "streamlit", "--version"])[0] == 0, "dashboard/app.py exists and streamlit imports"))
                else:
                    external.append(toks[0] if toks else cmd)
                continue
            i = toks.index(script)
            rest = toks[i + 1:]
            if rest[:1] == ["-m"] and rest[1].startswith("vlapoc."):
                base = [PY, "-m", rest[1]]
                after = rest[2:]
                kind = f"module {rest[1]}"
            elif rest[:2] == ["-m", "pip"] or rest[:2] == ["-m", "streamlit"]:
                external.append("python -m " + rest[1])
                continue
            elif rest and rest[0].startswith("scripts/") and rest[0].endswith(".py"):
                if not (REPO / rest[0]).is_file():
                    rows.append((cmd[:70], False, f"{rest[0]} does not exist"))
                    continue
                base = [PY, rest[0]]
                after = rest[1:]
                kind = rest[0]
            else:
                external.append(" ".join(rest[:2]))
                continue
            sub = None
            if kind in ("scripts/stage.py", "module vlapoc.lineage"):  # tools with subcommands: the first word that is not a global option or its value
                j = 0
                while j < len(after):
                    if after[j] in ("--chain", "--lineage-root", "--feats-root"):
                        j += 2
                    elif after[j].startswith("-"):
                        j += 1
                    else:
                        sub = after[j] if re.match(r"^[a-z][a-z0-9-]*$", after[j]) else None
                        break
            key = (kind, sub)
            if key in seen:
                continue
            seen.add(key)
            args = base + ([sub] if sub else []) + ["--help"]
            rc, tail = run(args)
            if rc != 0 and sub:
                rc2, _ = run(base + ["--help"])
                rows.append((f"{kind} {sub}", False, f"`{sub} --help` rc={rc}; top-level --help rc={rc2}"))
            else:
                rows.append((f"{kind}{' ' + sub if sub else ''}", rc == 0, f"--help rc={rc}"))
    ok = [r for r in rows if r[1]]
    for r in rows:
        print(f"      {'ok ' if r[1] else 'BAD'} {r[0]}  {r[2]}")
    check("8.1 every fenced shell command naming a scripts/*.py, python -m vlapoc.* or the dashboard exists and its subcommand's --help exits 0 here", len(ok) == len(rows) and len(rows) > 0,
          f"{len(ok)}/{len(rows)} entries ok; external commands not checked (they are server tools): {sorted(set(external))}")
    # --- AC8.2 ---
    marks = re.findall(r"<!-- count: (\w+) (\d+) (\d+) -->", text)
    bad, fresh = [], {}
    for key, jobs, trains in marks:
        o = subprocess.run([PY, "scripts/server_pipeline.py", "--tag", "v3cand", "--dry-run", *COUNT_ARGS[key]], cwd=REPO, capture_output=True, text=True).stdout
        n = len(re.findall(r"^\[dry \d+\]", o, re.M))
        t = len(re.findall(r"^\[dry \d+\] train-", o, re.M))
        fresh[key] = (n, t)
        if (n, t) != (int(jobs), int(trains)):
            bad.append((key, (jobs, trains), (n, t)))
    doc = (REPO / "scripts" / "server_pipeline.py").read_text().split('"""')[1]
    first = "--tag v3cand --subset v3/candidate --curation per_task --metrics f30 hrt60_lin"
    check("8.2 every dry-run count in the skill equals a fresh local dry run: Round A on v3/candidate (116 jobs, 24 trainings; v0/candidate only as a labelled variant), Round B per cluster (2 picks, and with --matched-random), Round B planted (default and --no-oracle); "
          "the first server run is the v3 command in the skill and in server_pipeline.py's docstring",
          not bad and {k for k, *_ in marks} == set(COUNT_ARGS) and first in text and first in doc and fresh.get("round_a_v3") == (116, 24) and fresh.get("round_b_planted") == (129, 33),
          f"fresh counts {fresh}; mismatches {bad}")
    # --- AC8.3 ---
    steps = re.split(r"(?m)^## (?=Step )", text)[1:]
    tagged, verified, unverified, unlabeled = [], [], [], []
    for s in steps:
        head = s.splitlines()[0]
        v = "[verified locally]" in head
        u = "[unverified until a server session]" in head
        tagged.append(v != u)
        (verified if v else unverified).append(head.split(" [")[0])
        for ln in re.findall(r"(?m)^(Expected[^\n]*(?:\n(?![#`\n<*-]).*)*)", s):
            lab = re.search(r"\((?:[^)]*?)(measured|quoted|inferred)", ln) is not None or "inferred" in ln or "measured" in ln or "quoted" in ln
            if u and not lab:
                unlabeled.append(head)
            if v and "measured" not in ln and "quoted" not in ln:
                unlabeled.append(head + " (verified step without a measured expectation)")
    check("8.3 every step is tagged [verified locally] or [unverified until a server session]; the verifier counts both and lists the unverified; no expected output in an unverified step lacks a measured/quoted/inferred label (and verified steps cite measured outputs)",
          steps and all(tagged) and not unlabeled, f"{len(verified)} verified: {verified}; {len(unverified)} unverified: {unverified}; unlabelled expectations: {unlabeled}")
    # --- AC8.4 ---
    stops = re.findall(r"\*\*STOP AND ASK \(([^)]*)\)", text)
    missing = [k for k, w in REQUIRED_STOPS.items() if not any(w in s for s in stops)]
    check("8.4 at least 7 STOP AND ASK markers, covering preflight failure, revision mismatch, any FAILED job, disk below threshold, any needed code change, the Round B cluster pick and a SYNTHETIC/SMOKE banner on real results",
          len(stops) >= 7 and not missing, f"{len(stops)} markers: {stops}; missing {missing}")
    # --- AC8.5 ---
    now = subprocess.run(["git", "diff", "--stat", "CLAUDE.md"], cwd=REPO, capture_output=True, text=True).stdout
    before = (ART / "b1_claude_md_diffstat_before.txt").read_text()
    check("8.5 CLAUDE.md is untouched by the dashboard batches: `git diff --stat CLAUDE.md` equals the recorded value from before B1", now == before, f"{now.strip()!r}")
    # --- AC8.6 ---
    import yaml

    fm = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    meta = yaml.safe_load(fm.group(1)) if fm else {}
    check("8.6 the frontmatter parses (name, description) and the skill is in .claude/skills/", meta.get("name") == "server-run" and len(meta.get("description", "")) > 80 and SKILL.is_file(),
          f"name {meta.get('name')!r}, description {len(meta.get('description', ''))} characters")
    ok_all = all(r[1] for r in results)
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "verify_runbook.json").write_text(json.dumps({"pass": ok_all, "checks": [{"name": n, "pass": p, "detail": d} for n, p, d in results], "verified_steps": verified, "unverified_steps": unverified,
                                                         "fresh_dry_run_counts": {k: list(v) for k, v in fresh.items()}}, indent=1))
    print(f"runbook: pass={ok_all} ({sum(r[1] for r in results)}/{len(results)})")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

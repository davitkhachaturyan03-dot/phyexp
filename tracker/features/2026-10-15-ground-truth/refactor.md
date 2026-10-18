# Refactor: ground truth (injection experiment)

Stage: refactor, after analyzer round 3 PASS. Scope: the feature's code only. Behaviour was not changed anywhere (see the behaviour check). Commits: `b8b8ef5`, `092da10`, `6f343df`.

## What was changed

| File | Change | Why it helps a reader |
|---|---|---|
| `vlapoc/methods/heuristics.py` | Module docstring: added "where this sits" (derive -> screen -> score) and a runnable CLI example with `--subset v3/candidate` and `--out ...heuristics_v3/` (analysis-r3 non-blocking item 3: it said `v2/candidate`, which is fenced and would be refused); notes `--no-track` and `--only`. New helper `score_column(table)` replaces two copies of the same list comprehension in `main`. `main` docstring said "H1-H6"; it now says the default is H1-H8. | The CLI line can be pasted and works. One name for "the score column of a table". |
| `vlapoc/corruptions.py` | Module docstring: where the module sits and which three things to read first. New constant `NO_CLIP_STATS` replaces three identical 4-line loops (`instruction_swap`, `dropped_frames`, `idle_padding`) that wrote zero clip counts; same keys in the same order, so the label `params` JSON is unchanged. Removed the unused `before` argument of `clip_actions` (no caller passed it). | The three "this type never touches the actions" types now visibly share one definition. No dead parameter. |
| `vlapoc/groundtruth.py` | Module docstring: title now covers select, derive and score (it still said "S3 answer key"), the pipeline position, a reading order, `derive` example with `v3`, and examples for `score` and `retire`. `main` docstring points at them. | The three subcommands each have a command a newcomer can run from the repo root, and it says which ones write to the real MLflow store. |
| `vlapoc/method_data.py` | Module docstring: where it sits (between stored versions and any method) and a two-step reading order. | Names the boundary the rest of the feature is built around. |
| `vlapoc/derive.py` | Module docstring: replaced a stale sentence ("the corruption suite is a separate, owner-approved later feature") with a pointer to `vlapoc.corruptions` and `vlapoc.groundtruth`. It misstated the code. `derive_version` docstring: split one 400-character run-on paragraph into two. | The docstring now matches what the file does. |
| `vlapoc/lineage.py` | `assert_blind` docstring now states the two refusal messages (supersession wording for v2, "not blind" wording for v1 and leaks). | The two fence outcomes are documented where they are raised. |
| `scripts/verify_injection_s4.py` | Docstring and usage line updated to the v3 command (`--version v3 --prefix s4_v3 --prior v2 --run-id ...`), AC4.1-AC4.8. | The usage line is the command the log actually ran. |
| `scripts/verify_recovery_s5.py`, `scripts/verify_consumers_s6.py` | Docstrings say v3 (they said v2). In S6, a local variable named `v2` (it held the version being tested, `--version v3` by default) is renamed `ver`. | A variable named `v2` while testing v3 was misleading. |

`vlapoc/runs.py`, `scripts/verify_corruptions_s1.py`, `verify_derive_gt_s2.py`, `verify_method_boundary_s3.py` and `s5_recovery_table.py` were read and needed no change (see below).

## Behaviour check

All run before the edits (baseline, from commit `e70017e`) and after (`6f343df`), outputs under `data/scratch/refactor/{before,after}/`. Script: `data/scratch/refactor/run_all.sh`. Real store: `data/lineage/mlflow.db` sha256 `306af4123fda7e63...` at start and end of both runs (measured, `db_sha_start.txt` / `db_sha_end.txt`). No MLflow run was written to the real store; S2, S3 and S6 use scratch lineage copies under `data/scratch/`. v3 and v2 were not re-derived.

| Command | Before == After? | Note |
|---|---|---|
| `verify_corruptions_s1.py --version v0 --per-type 100 --seed 0 --dump-hashes ...` | Yes | "all invariants hold: True". `s1_hashes.json` (every corrupted episode's content hash) byte-identical. `s1_invariants.json` differs only in a wall-clock field (`seconds` 32.6 vs 32.5). |
| same, `--check-hashes` (second process, AC1.5) | Yes | text output identical modulo timings. |
| `verify_derive_gt_s2.py` | Yes | `s2_verify.json` byte-identical. Text differs only in `frames/s` and seconds. All S2 checks pass. |
| `verify_method_boundary_s3.py` | Yes | `s3_verify.json` byte-identical to before (and to the committed one). Text differs only in `frames/s`. All S3 checks pass. |
| `verify_injection_s4.py --version v3 --parent v0 --prefix s4_v3 --prior v2 --run-id d3dd9fd4...` | Yes | `s4_v3_verify.json`, `s4_v3_allocation.csv` and `s4_v3_filmstrip.png` byte-identical. Reads stored bytes only. |
| `verify_recovery_s5.py --version v3 --prefix s5_v3` | Yes | `s5_v3_verify.json` byte-identical. |
| `s5_recovery_table.py` on the stored v3 r2 recovery JSONs (to scratch) | Yes | table, bands and v2-vs-v3 `time_shift` CSV byte-identical. |
| `verify_consumers_s6.py --version v3 --prefix s6_v3` | Yes | `s6_v3_action_stats_vs_v0.csv` byte-identical. `s6_v3_verify.json` differs only in the SmolVLA smoke run's `run_id` and `seconds`. mlflow.db unchanged. |
| `heuristics.main(--subset v3/candidate --no-track)` (added check, in-process via `data/scratch/refactor/check_heuristics.py`) | Yes | All 9 score parquets and `directions.json` identical before and after, and all 9 equal the stored tables under `data/eval/groundtruth/heuristics_v3*` (`DataFrame.equals`). Not run with tracking, so no `screen` runs were written. |

Differences that are not results, and were ignored: process ids and library-load warnings (`objc[...]`), progress bars, `frames/s`, elapsed seconds, MLflow run ids and the timestamp in the scratch run directory name. Verifiers that overwrite committed artifacts (`s6_v3_verify.json`, both runs) were diffed (only run id and seconds) and restored with `git checkout`; `git status` of `tracker/` is clean.

## Left as-is (and why)

- **`corruptions.py` constructors (`time_shift`, `detour`, `idle_padding`, ...).** Each is one closure per type with its own `params` block. Merging them would hide the per-type `source` and `parameters` provenance that the labels record. `detour` and `idle_padding` build index orders in loops; they are clear enough with their docstrings and a vectorised rewrite would risk the seeded draw order that S1's hashes pin.
- **`_check_score_keys` alias in `lineage.py`** (public `check_score_keys` plus the old private name). Kept: other modules still use the old name; removing it touches files outside this feature.
- **`groundtruth.py` local imports** (`mlflow`, `runs`, `lineage._root`, `write_fence` inside functions) and the `_vdir` helper defined after its first use. Deliberate (MLflow is slow to import, and `runs` imports a lot) or harmless.
- **`recovery_metrics` and `exp_pos_top`.** Hand-written expected-count-at-k with fractional tie handling. sklearn has no equivalent for that tie rule; it uses sklearn for AUROC, AP and ROC. Left alone, and the tie rule is stated as inferred in the docstring.
- **Verifier scripts.** They are long but linear and each check is labelled with its AC number; re-sectioning them would change printed text that the execution log quotes.
- **`runs.py`.** The feature's change is one added stage name; nothing to improve.
- **`CLAUDE.md`, `.claude/`, `calls/`.** The owner's uncommitted changes; not touched and not committed (they show as modified in `git status`).

## Reading guide

Read in this order. The pipeline is: choose episodes, corrupt them while writing a new dataset version, let a method screen it without seeing the answer, then score its ranking against the answer key.

1. `tracker/features/2026-10-15-ground-truth/brief.md`, "Owner decisions" and "Owner rulings": clipping, scope (no viewpoint), blindness enforced now, and ruling A (`time_shift` trims instead of padding).
2. `vlapoc/corruptions.py`: the seven corruption types as pure functions. Read `make_edit`, then `action_noise` (simplest), `time_shift` (changes length, ruling A), `clip_actions` (owner decision 1). Each type's `params` records its source and whether each parameter is quoted, measured or inferred.
3. `vlapoc/derive.py`, `EpisodeEdit` and `derive_version`: applies edits while writing the new dataset in the parent's order, writes `labels.parquet` (the answer key) and `parents.parquet`, and refuses to register a version that is not blind. The instruction swap (`new_task`) is handled here.
4. `vlapoc/lineage.py`: `compare_blind` / `check_blind` / `assert_blind` (what counts as leaking an edit; the instruction is content, provenance is bookkeeping) and `write_fence` (v1 fenced for a positional leak, v2 fenced as superseded).
5. `vlapoc/groundtruth.py`: first `select_injection` (13 episodes per type, 4 or 5 corrupted per task) and `build_edits` / `derive_injection` (the v3 command), then `recovery_metrics`, `score_recovery` and `leak_check`, and last `retire_version`.
6. `vlapoc/method_data.py`: `open_for_method`, the one door a method uses to read a version. It verifies bytes, refuses fenced or leaking versions, refuses ground-truth-derived subsets and refuses opening two versions of one lineage in a process (a diff would name the edits).
7. `vlapoc/methods/heuristics.py`: eight label-free baseline rankers (H1-H8) that go through that door. `scan_episodes` and `compute` are the core. Their per-type scores are the "table to beat" in the report.
8. Verifiers, in story order; each docstring has its run command:
   - `scripts/verify_corruptions_s1.py`: invariants of the seven transforms on real v0 episodes, and the cross-process hash check.
   - `scripts/verify_derive_gt_s2.py`: a 20-episode scratch derivation, `PYTHONHASHSEED` 0 vs 1.
   - `scripts/verify_method_boundary_s3.py`: refusal table for the loader, plus a source scan that `method_data.py` and `methods/` never import `groundtruth`.
   - `scripts/verify_injection_s4.py`: v3 recomputed from stored bytes (allocation, trim, blindness, lineage, v2 fenced).
   - `scripts/verify_recovery_s5.py` and `scripts/s5_recovery_table.py`: hand-computed metric fixtures, the v1 positional-leak check, and the heuristic-by-type table with detectability bands.
   - `scripts/verify_consumers_s6.py`: per-subset stats and a 2-step SmolVLA smoke on a scratch copy.
9. `tracker/features/2026-10-15-ground-truth/artifacts/server-runbook.md`: what still needs the server (GT-R1 to GT-R6).

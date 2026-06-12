---
type: plan
status: done
created: 2026-06-12
source_plan: "[[01-codecheck-roadmap]]"
---

# Iteration: CodeHaluEval (stdin/stdout) as a third dataset

> **Implemented.** Schema corrections found during execution (the plan's HF assumptions were
> partly wrong): CodeHaluEval ships as **8 per-category JSON-list files** that `load_dataset`
> cannot merge (null-vs-string cast error), so the loader downloads + parses them directly;
> `input`/`output` are **raw strings** (or occasionally a list of lines), **not** JSON-encoded;
> `solutions` is a JSON-encoded list string and is often empty/partial; `task_id` is an int
> (stored as `CodeHalu/<id>`). Verified end-to-end: a shipped solution reproduces the shipped
> expected stdout (25/25 on real tasks), and a live `--dataset codehalu` run produces all four
> scores. The canonical is never run, so its fragmentary `solutions` do not affect labeling.

## Context

MBPP+ and HumanEval+ are both **function-call** datasets, and on a capable model HumanEval+
saturated (few incorrect mains, near-identical samples) — a weak testbed for the
confident-consistent blind spot. **CodeHaluEval** (`Yuchen111/CodeHaluEval`) was deferred
earlier because it is **whole-program stdin→stdout** (Codeforces-style), which the existing
`fn(*args)` harness cannot run. It is worth pulling now: it is *built* to induce
hallucinations (6 `halu_type` categories), so it should give a real positive class and
genuinely stress the blind spot.

Decisions (made): **reuse the existing pipeline** and inject stdio-specific parts; use the
**dataset-provided expected outputs** as ground truth (no reference run); scope to the
**382 stdin/stdout tasks** (skip the 33 `fn_name`/LeetCode tasks).

## Dataset facts (verified)

- 415 tasks after grouping by `task_id`; **382 stdin/stdout** (`fn_name` is null), 33 fn_name.
- One row per `(task_id, test_case_id)`: `question` (NL prompt), `solutions` (JSON list of
  full reference programs), `input`/`output` (**JSON-encoded** stdin/stdout strings),
  `halu_type`, `fn_name`, `starter_code`. Mean **14.5 test cases/task** (max 250).
- Programs read `input()` / print to stdout. Expected output **ships with the data**.

## Key architectural insight (why reuse works)

The scorers and labelers compare **normalized outcomes**, not function values:
`exec_inconsistency`, `is_correct`, `count_outcomes`, `has_error` all operate on lists of
`("value", x) | ("status", ...)` and use `==`. A stdout string is a perfectly good `value`.
So the only stdio-specific pieces are: **execution** (run a program with stdin, capture
stdout), **generation** (ask for a whole program), **expected source** (provided, not run),
and the **Prompt template** wording. The four scoring metrics stay intact.

## Per-variant adjustment (the core of this iteration)

- **Exec — real new mechanism.** New whole-program harness: feed each `input` to stdin, run
  the program, capture stdout, compare to the next program's stdout. Metric meaning
  unchanged (behavioral divergence), but the substrate is stdin/stdout strings, so add a
  shared **stdout normalizer** (strip trailing whitespace / unify line endings) used by both
  the harness and the loader's expected. Caveat: numeric-output problems compared as exact
  strings may over-count `fail` on formatting; note it, optional numeric tolerance later.
- **Prompt — stdio-specific template.** Reword the (divergence-seeking) judge for programs:
  "two programs that read stdin and write stdout — would they print the same output for every
  stdin?" Add `CODEHALU_JUDGE_TEMPLATE` and select it by `--dataset`, extending the
  dataset→template mechanism already built for HumanEval. Same Yes/No/N-A mapping.
- **AST — reuse, documented caveat.** `ast.parse` handles full modules; node-type fingerprint
  works as-is. Competitive solutions share heavy I/O boilerplate (`input()`, parsing), which
  inflates structural similarity → expect AST even weaker here. No code change; document.
- **CodeBERT — reuse, documented caveat.** Embeds whole-program text; same saturation plus
  the shared boilerplate → expect it weakest, as on the other datasets. No code change.

So Exec and Prompt get genuine adjustments; AST and CodeBERT reuse on whole-program text with
recorded caveats — which is itself part of the cross-dataset story (the local-similarity
methods degrade further where boilerplate dominates).

## Changes

**`codecheck/models.py`** — `CodeProblem` gains optional `expected: list | None = None`
(pre-supplied normalized ground-truth outcomes; `None` = run the canonical, current
behavior). `to_dict`/`from_dict` updated. `entry_point` may be `""` for stdio.

**`codecheck/pipeline.py` (`score_problem`)** — one line:
`expected = problem.expected if problem.expected is not None else expected_outputs(problem, harness, timeout)`.
Everything else (exec/prompt/ast/codebert branches, labeling, count) unchanged.

**`codecheck/execution/stdio_sandbox.py` (new)** — `run_stdio_batch_in_subprocess(code,
_entry_point, stdin_list, timeout)` mirroring the batch-harness signature. Per stdin:
`subprocess.run([sys.executable, prog], input=stdin, capture_output=True, text=True,
timeout=...)`, returning `("ok", normalized_stdout) | ("err", repr) | ("timeout", None)`
aligned to `stdin_list`. Reuse the daemon/kill discipline from `sandbox.py`. Shared
`normalize_stdout(s)` helper (also used by the loader).

**`codecheck/generation/generator.py`** — add a stdio prompt builder (whole-program: "read
stdin, write stdout, output only the program"); make `CodeGenerator` accept an injectable
`prompt_builder` (default = current function-call `build_prompt`). `extract_code` unchanged.

**`codecheck/dataset.py`** — `load_codehalu_eval(limit, randomize, seed, index,
max_cases=..., cache_path=data/codehalu.json)`: stream the HF dataset, group by `task_id`,
keep `fn_name is None`, `json.loads` `input`/`output`/`solutions`, build each `CodeProblem`
with `prompt=question`, `entry_point=""`, `canonical_solution=solutions[0]` (reference only),
`inputs=[stdin...]`, `expected=[("value", normalize_stdout(out)) ...]`, `atol=0`. Cap test
cases per task at `max_cases` (default ~25) to bound runtime; reuse `_finalize` for
selection/cache.

**`run_codecheck.py` (`_cmd_run`)** — `--dataset` gains `codehalu`. When selected: use
`load_codehalu_eval`, the stdio harness, the stdio generator prompt, and
`CODEHALU_JUDGE_TEMPLATE`. (Expected comes from the problem, so labeling is automatic.)

**`codecheck/score/prompt.py`** — add `CODEHALU_JUDGE_TEMPLATE` (program-oriented,
divergence-seeking).

**Docs** — `docs/plans/01-codecheck-roadmap.md`: move CodeHaluEval from *deferred* to this
iteration; `codecheck/README.md`: add a CodeHaluEval (stdin/stdout) dataset subsection +
note the Exec/Prompt stdio adjustments and AST/CodeBERT caveats.

## Critical files

- `codecheck/execution/stdio_sandbox.py` (new), `codecheck/dataset.py`,
  `codecheck/models.py`, `codecheck/pipeline.py`, `codecheck/generation/generator.py`,
  `codecheck/score/prompt.py`, `run_codecheck.py`.
- Tests: `tests/test_codecheck_stdio_sandbox.py` (new), `tests/test_codecheck_dataset.py`,
  `tests/test_codecheck_cli.py`, `tests/test_codecheck_pipeline.py` (expected-injection path).

## Reuse

- `exec_inconsistency`, `is_correct`/`count_outcomes`/`has_error`, `normalize_output`
  (string passes through as a value) — unchanged.
- `_finalize` selection/cache in `dataset.py`; `--dataset` flag + judge-template routing.
- `PromptJudge`/`map_staggered`, `ASTScorer`, `CodeBERTScorer`, `run_dataset`,
  incremental-save/resume — unchanged.

## Verification (small subset — user runs the full sweep)

1. **stdio harness unit (no API):** a trivial program `print(int(input())+1)` over a few
   stdins returns the right `("ok", ...)`; a hanging program → `("timeout", None)`; a raising
   program → `("err", ...)`. Run via a real `.py` driver (spawn/subprocess needs a real file).
2. **Loader unit (network-free):** monkeypatch the HF `load_dataset` with a tiny fake
   (2 tasks, a few cases, mixed fn_name) → asserts grouping, `fn_name` filtering,
   `json.loads`, the assembled `expected`, and the `max_cases` cap.
3. **Expected-injection unit:** `score_problem` with a problem carrying `expected` does **not**
   call the canonical-run path and labels from the provided outputs.
4. **Offline correctness check (the key one):** load 3 real CodeHaluEval tasks, run each
   `solutions[0]` through the stdio harness on the task's stdins, assert stdout matches the
   provided `output` for all — proving harness + normalization + expected wiring end to end.
5. **Small live run:** `run --dataset codehalu --limit 5 --n 5 --method all --timeout 5` →
   completes, results carry CodeHaluEval task_ids, four scores, and a real incorrect rate;
   `evaluate` prints the readout. Confirms generation + stdio execution + scoring.
6. **Suite:** `pytest tests/ -q -k codecheck` green; MBPP+/HumanEval+ paths unaffected.
7. Commit (no push), per project convention.

## Open notes

- Exact stdout match may over-count `fail` on float/formatting-sensitive problems; numeric
  tolerance is a deferred refinement, not in this iteration.
- `halu_type` is carried for possible later analysis (which hallucination categories each
  variant catches) but is not used in the core metric.

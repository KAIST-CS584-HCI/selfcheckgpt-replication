---
type: plan
status: ready
created: 2026-06-12
source_plan: "[[01-codecheck-roadmap]]"
---

# Iteration 5 — second dataset (HumanEval+), cross-dataset four-method comparison

## Context

Iter 5 was "run all four variants on CodeHaluEval, then Collu-Bench." Inspection of the
real datasets overturned that:
- **CodeHaluEval** (`Yuchen111/CodeHaluEval`) is ~98% Codeforces **stdin/stdout** programs —
  the existing `fn(*args)` harness can't run it; it needs a whole-program exec path.
  **Deferred** to a later iteration (noted in the roadmap, not built now).
- **Collu-Bench** (`lt-asset/collu-bench`) is a token-logprob hallucination-*detection*
  benchmark (pre-generated outputs + token annotations), **not** a generate-and-sample task
  source. **Dropped** from the roadmap.

Decision: iter 5 = add **HumanEval+** as the second dataset, **loader-only**, reusing the
entire pipeline. It is the standard function-call EvalPlus benchmark (same interface as
MBPP+), giving a clean cross-dataset four-method point with zero harness change. Not
"hallucination-targeted" — that goal moves to the deferred CodeHaluEval work. Role split:
implement + verify correctness on a small subset; the user runs the full sweep.

## Key facts (verified against evalplus)

- `get_human_eval_plus()` → 164 problems, schema **identical** to MBPP+ (`task_id`, `prompt`,
  `entry_point`, `canonical_solution`, `base_input`, `plus_input`, `atol`). `_to_problem` in
  `codecheck/dataset.py:14` maps it verbatim **except one asymmetry**:
  - MBPP+ `canonical_solution` is a standalone `def ...` (runnable as-is).
  - HumanEval+ `canonical_solution` is the **body only**; `prompt` is the signature+docstring.
    Runnable canonical = **`prompt + canonical_solution`**. The harness does `exec(code);
    fn = ns[entry_point]`, so without prepending the prompt the entry_point is undefined and
    every problem would mislabel. This is the single correctness-critical detail.
- Generation is unaffected: `CodeGenerator` prompts with `problem.prompt` and `extract_code`
  pulls a full `def` from the reply, for both datasets.
- `task_id` namespaces differ (`HumanEval/0` vs `Mbpp/2`), so resume-by-task_id and mixed
  files stay unambiguous. `evaluate` is dataset-agnostic (reads a results file) — no change.

## Changes

**`codecheck/dataset.py`** — factor the shared tail (select/index/cache-write) into a private
`_finalize(problems, limit, randomize, seed, index, cache_path)`; keep `load_mbpp_plus`
behavior identical (canonical as-is). Add:
- `load_human_eval_plus(limit, randomize, seed, index, cache_path=data/human_eval_plus.json)`
  using `from evalplus.data import get_human_eval_plus`, building each `CodeProblem` with
  `canonical_solution = item["prompt"] + item["canonical_solution"]` (the only difference
  from `_to_problem`), then `_finalize(...)`.

**`run_codecheck.py`** — add `--dataset {mbpp,humaneval}` (default `mbpp`, back-compat) to the
`run` subparser; in `_cmd_run` dispatch to the matching loader (replacing the direct
`load_mbpp_plus` call at line ~82); include the dataset name in the `Running methods=...`
print. `evaluate`/`codebert` untouched.

**Tests (network-free, mirror existing style):**
- `tests/test_codecheck_dataset.py`: monkeypatch `ds.get_human_eval_plus` with a tiny fake
  whose `canonical_solution` is body-only; assert the loaded `CodeProblem.canonical_solution`
  == `prompt + body` (the assembly), and that selection/index/cache reuse works.
- `tests/test_codecheck_cli.py`: `--dataset humaneval` parses; default is `mbpp`; dispatch
  picks `load_human_eval_plus` (monkeypatch both loaders, assert the right one is called).

**Docs / memory:**
- `docs/plans/01-codecheck-roadmap.md`: rewrite iter 5 — HumanEval+ (loader-only) as the
  shipped second dataset; record CodeHaluEval as **deferred** (stdin/stdout path, with the
  finding) and Collu-Bench as **dropped** (logit-detection benchmark, not a task source).
  Update iter 6 to "four methods × {MBPP+, HumanEval+}".
- Serena memory `codecheck/core` + auto-memory `codecheck-design`: note the second dataset
  and the two dataset-fit findings.

## Critical files

- `codecheck/dataset.py` — `_finalize` extraction + `load_human_eval_plus` (canonical assembly).
- `run_codecheck.py` — `--dataset` flag + loader dispatch in `_cmd_run`.
- `tests/test_codecheck_dataset.py`, `tests/test_codecheck_cli.py`.
- `docs/plans/01-codecheck-roadmap.md`, memories.

## Reuse

- `_to_problem` / `_select` / cache logic in `codecheck/dataset.py`.
- Entire pipeline unchanged: `CodeGenerator`, `run_batch_in_subprocess`, `score_problem`,
  `run_dataset`, all four scorers, `evaluate`, resume/incremental-save.

## Verification (small subset — user runs the full sweep)

1. **Unit:** `pytest tests/test_codecheck_dataset.py tests/test_codecheck_cli.py -q` green,
   incl. the new canonical-assembly test.
2. **Offline correctness check (no API), the key one:** load 3 real HumanEval+ problems and
   run each `canonical_solution` through `run_batch_in_subprocess` on its own inputs; assert
   `is_correct` is True for all three. This proves the `prompt + body` assembly is runnable —
   if the prompt weren't prepended, the canonical would error and this fails loudly.
3. **Small live run (needs OPENROUTER_API_KEY, cheap):**
   `python run_codecheck.py run --dataset humaneval --limit 2 --n 2 --method exec --timeout 5
   --output output/he-smoke.json` → completes, results carry `HumanEval/*` task_ids and an
   `exec` score; `evaluate --results output/he-smoke.json` prints a readout. Confirms
   generation + execution + scoring end-to-end on the new dataset.
4. **Full suite:** `pytest tests/ -q -k codecheck` → all green (no regression on MBPP+).
5. Commit (no push), per project convention. The full HumanEval+ run is the user's.

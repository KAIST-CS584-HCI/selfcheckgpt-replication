---
type: plan
status: ready
created: 2026-06-12
source_plan: "[[08-codecheck-iteration5-humaneval]]"
---

# Offline `prompt` subcommand — re-score the Prompt variant on saved results

## Context

A HumanEval+ run is in progress using the **old** judge template; we don't want to stop it.
We need to apply the new `HUMANEVAL_JUDGE_TEMPLATE` (the divergence-seeking judge) to
**already-generated** results without re-running generation. The `codebert` subcommand
already does exactly this shape for embeddings (reuse stored `main_code` + `sample_codes`,
rewrite `scores[...]` in place). Mirror it for `prompt`.

Difference from `codebert`: the Prompt judge **calls the API** (LLM), so this command needs
`OPENROUTER_API_KEY`, model/base-url setup, and must pick the judge template per dataset.

## Behavior

`python run_codecheck.py prompt --results <file> [--dataset {mbpp,humaneval}]`

- Load the results (`load_results`), reuse each result's stored `main_code` + `sample_codes`.
- For each result, compute the judge score and overwrite `scores["prompt"]`.
- Save in place atomically (`save_results`). Idempotent (re-run overwrites).
- Print `Re-scored prompt on N results in <file>` and `Judge parse failures: K`.

**Template selection (per result, robust to the wrong file):**
- `--dataset` is **optional**. If given, force that template for all results.
- If omitted, auto-detect per result from the `task_id` prefix: `HumanEval/...` →
  `HUMANEVAL_JUDGE_TEMPLATE`, otherwise the default `JUDGE_TEMPLATE`. This prevents applying
  the HumanEval prompt to an MBPP+ file by accident and handles a mixed file.
- Implementation: partition results by chosen template, build one `PromptJudge` per
  template group, score each group, sum `parse_failures`.

## Changes (all in `run_codecheck.py`, mirroring `_cmd_codebert`)

**`_cmd_prompt(args)`** — new, structured like `_cmd_codebert`:
- Read `OPENROUTER_API_KEY` (exit cleanly if missing, like `_cmd_run` lines ~57-59), `model`,
  `base_url`; build the `OpenAI` client (same args as `_cmd_run`).
- `load_results(args.results)` with the same `FileNotFoundError` / `ValueError` guards as
  `_cmd_codebert`.
- Choose template per result (override or task_id auto-detect, above); group; for each group
  build `PromptJudge(client, model=model, template=<tmpl>)` and set
  `r.scores["prompt"] = judge.score(r.main_code, r.sample_codes)`.
- `save_results(results, args.results)`; print summary + total judge parse failures.

**Subparser** — add a `prompt` subcommand beside `codebert`:
```python
pr_p = sub.add_parser("prompt", help="re-score the prompt (LLM-judge) variant on an existing results file")
pr_p.add_argument("--results", type=str, default=str(DEFAULT_OUTPUT), help="results JSON path to augment in place")
pr_p.add_argument("--dataset", choices=["mbpp", "humaneval"], default=None,
                  help="force a judge template; default auto-detects per result from the task_id prefix")
pr_p.set_defaults(func=_cmd_prompt)
```

No changes to `codecheck/score/prompt.py` (the templates + `PromptJudge(template=...)` already
exist) or the pipeline. `parse_judgment`, `load_results`/`save_results` reused as-is.

## Critical files

- `run_codecheck.py` — `_cmd_prompt` + `prompt` subparser.
- `tests/test_codecheck_cli.py` — parse + augment tests.
- `codecheck/README.md` — one line documenting the offline `prompt` subcommand alongside
  `codebert`.

## Reuse

- `_cmd_codebert` shape (load → per-result rescore → save → summary), `run_codecheck.py`.
- API-key/model/base-url/client setup from `_cmd_run`.
- `PromptJudge`, `JUDGE_TEMPLATE`, `HUMANEVAL_JUDGE_TEMPLATE` (`codecheck/score/prompt.py`).
- `load_results`/`save_results` (`codecheck/pipeline.py`).

## Verification

1. **Parse:** `prompt --results x.json` parses; `--dataset humaneval/mbpp` accepted; bad value
   rejected; default `dataset` is `None`.
2. **Augment unit (no real API):** monkeypatch `PromptJudge` with a fake scorer, write a small
   results file (one `HumanEval/*` + one `Mbpp/*` task_id), run `_cmd_prompt`, assert both get
   `scores["prompt"]` overwritten, other scores preserved, and the file is rewritten.
3. **Template routing unit:** assert a `HumanEval/*` result is scored with
   `HUMANEVAL_JUDGE_TEMPLATE` and an `Mbpp/*` result with `JUDGE_TEMPLATE` when `--dataset`
   is omitted; `--dataset` forces one template (capture via a fake judge that records its
   template).
4. **Missing-key / corrupt-file guards** behave like `_cmd_codebert`.
5. **Small live confirm (cheap):** copy a finished HumanEval results file, run
   `prompt --results <copy>`, confirm `scores["prompt"]` changes vs the stored (old-template)
   values and `evaluate` reflects them.
6. `pytest tests/ -q -k codecheck` green. Commit (no push).

## Caveat

`prompt --results` **rewrites the file in place** — point it at a finished or copied results
file, **not** the one the live run is still writing.

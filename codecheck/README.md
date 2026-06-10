# codecheck — SelfCheckGPT for Code

Extends SelfCheckGPT's sample-consistency idea to code generation. Hallucination = an
incorrect implementation. Correctness is auto-verified by execution, so no human
annotation is needed.

Per problem: generate **1 main** implementation at `T=0` and **N samples** at `T=1`.
Two numbers come out: `exec_score` (the detection signal) and `is_correct` (ground truth).

## Methods

Three variants are planned (`docs/source/04-codecheck-methods.md`). **Only SelfCheck-Exec
is implemented** in this iteration.

### SelfCheck-Exec (implemented) — behavioral consistency

Replaces the original BERTScore variant with behavioral I/O equivalence.

1. **Generate** — `generation.py`: prompt the LLM with the problem; produce 1 main impl
   (`T=0`) + N samples (`T=1`). Strip markdown fences.
2. **Execute** — `execution.py`: run every impl on each input in a fresh spawned
   subprocess (`run_in_subprocess`), killed on timeout (SIGKILL). Outcome is one of
   `("ok", value)` / `("err", repr)` / `("timeout", None)`.
3. **Normalize** — `execution.py`: `normalize_output` maps each outcome to a hashable
   tuple (`("value", canonical)` or `("status", status)`), floats bucketed by `atol`,
   sets/dicts canonicalized. Equality of these tuples is the consistency primitive.
4. **Score** — `exec_score.py`: for each sample, `agreement = matches_with_main / n_inputs`;
   `exec_score = 1 - mean(agreement)`. `0.0` = all samples behave like main (consistent),
   `1.0` = none do (likely hallucinated).
5. **Label** — `labeling.py`: run the problem's `canonical_solution` through the same
   harness; `is_correct = (main outputs == canonical outputs)`, position by position.
6. **Evaluate** — `evaluate.py`: AUC-PR with positive class = incorrect, score =
   `exec_score`. Returns `nan` if only one class is present (all correct / all wrong).

### SelfCheck-AST (planned) — structural consistency

Parse main + samples to ASTs; score tree similarity. Replaces the n-gram variant.

### SelfCheck-Prompt (planned) — LLM judge

Ask a judge LLM whether each sample's behavior is consistent with main; aggregate
Yes/No/N-A. Carried over from the original Prompt variant.

## Dataset (current): MBPP+

Loaded via `evalplus` (`dataset.py` → `load_mbpp_plus`). MBPP+ augments MBPP with extra
test inputs (EvalPlus). Each problem maps to a `CodeProblem`:

| Field               | Source (evalplus key)         | Meaning                                  |
|---------------------|-------------------------------|------------------------------------------|
| `task_id`           | `task_id`                     | e.g. `"Mbpp/2"`                          |
| `prompt`            | `prompt`                      | function signature + docstring, fed to LLM |
| `entry_point`       | `entry_point`                 | function name to call                    |
| `canonical_solution`| `canonical_solution`          | reference impl (ground-truth source)     |
| `inputs`            | `base_input` + `plus_input`   | list of argument-lists                   |
| `atol`              | `atol`                        | float comparison tolerance               |

`inputs` is a `list[list]`: each element is one call's positional args, invoked as
`entry_point(*args)`.

Example `CodeProblem`:

```python
CodeProblem(
    task_id="Mbpp/2",
    prompt="def f(x):\n    'add one'\n",
    entry_point="f",
    canonical_solution="def f(x):\n    return x + 1\n",
    inputs=[[1], [2], [3]],   # f(1), f(2), f(3)
    atol=1e-6,
)
```

Result rows (`CodeResult`, saved by `pipeline.save_results`):

```json
{"task_id": "Mbpp/2", "exec_score": 0.0, "is_correct": true,
 "main_code": "...", "sample_codes": ["...", "..."]}
```

## Run

```bash
python run_codecheck.py run --limit 10 --n 5 --timeout 5   # generate + score + save
python run_codecheck.py evaluate --results output/codecheck-exec.json
```

`run` needs `OPENROUTER_API_KEY` (and optional `OPENROUTER_MODEL`, default `qwen/qwen3.5-9b`)
in `.env`.

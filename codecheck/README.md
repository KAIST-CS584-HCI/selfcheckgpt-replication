# SelfCheckGPT for Code

SelfCheckGPT idea on code: model that knows a solution writes it consistently across tries;
hallucinated solution drifts. Hallucination = **incorrect code**; correctness checked by
running it (no human labels).

Per problem: one main answer (temp 0) + N samples (temp 1). Yields a **consistency signal**
(sample agreement with main) and a **ground-truth label** (does main actually work).

## How the methods work

All variants share generation + ground-truth steps; differ only in how they measure
main-vs-sample consistency. Score scale: **higher = more likely incorrect**.

### SelfCheck-Exec — behavioral

Consistency via *behavior*: same outputs on same inputs?

1. **Generate** main + N sample implementations.
2. **Run** every impl on shared inputs in isolated, time-limited sandbox. Each run → value,
   error, or timeout.
3. **Compare**: two impls "agree" on an input when same result (float tolerance).
4. **Score**: how often samples disagree with main. More agreement → lower score.
5. **Correctness**: run reference solution on same inputs, check main matches. Ground truth.
6. **Evaluate**: across problems, how well score predicts incorrect.

### SelfCheck-Prompt — LLM-as-judge

Consistency by *asking a model*, not running. Per sample, judge LLM sees main + sample, asks
if sample behavior consistent with main.

1. **Generate** — same as Exec.
2. **Judge each sample**: consistent with main? Yes / No / N/A + one-sentence reason. N
   judgments run concurrently.
3. **Map**: Yes→`0.0`, No→`1.0`, N/A→`0.5`. Unparseable → `0.5`, counted as parse failure.
4. **Score**: mean of per-sample values.
5. **Correctness/evaluate** — same as Exec; judge never sees ground truth.

No execution needed → catches code that runs but is subtly wrong. Complementary to Exec
(blind to samples that consistently agree on the *same* wrong behavior). Same generations →
directly comparable.

### SelfCheck-AST — structural

Consistency by *parsing*, not running/asking. Each impl → structural fingerprint, samples
compared to main.

1. **Generate** — same as above.
2. **Fingerprint**: parse to AST, count node types (`Name`, `BinOp`, `Constant`, …), ignore
   identifier text + literal values → rename/literal-invariant.
3. **Compare**: dissimilarity = `1 − multiset Jaccard`. `0.0` identical, up to `1.0` no shared
   node types.
4. **Score**: mean per-sample dissimilarity. Unparseable impl → `1.0`, counted as parse failure.
5. **Correctness/evaluate** — same as Exec.

No API cost (local parse). Hypothesized to share Exec's confident-consistent blind spot:
consistently-wrong samples are also structurally similar to wrong main → AST ≈0, misses it.

## Methods

`run --method {exec,prompt,ast,code_bert,all}`. One run generates each problem's impls once,
scores with every selected method → methods always compared on identical data.

- `exec` (default) — behavioral I/O divergence.
- `prompt` — LLM-judge, Yes→0.0 / No→1.0 / N-A→0.5, mean over N. On HumanEval+ a sharper,
  oracle-free judge prompt is used (hunts for an edge-case input where impls diverge), because
  a capable model writes near-identical samples on those canonical problems.
- `ast` — structural divergence, rename/literal-invariant, mean over N. No API cost. Metric via
  `--ast-metric`.
- `code_bert` — embedding divergence: mean `1 − cosine` of CodeBERT (`microsoft/codebert-base`,
  mean-pooled) embeddings, main vs each sample. Learned semantic/lexical similarity (not
  rename-invariant). No API cost; loads ~500MB model on first use. Usually run **offline**.
- `all` — exec + prompt + ast + code_bert; `evaluate` prints per-method comparison.

**Offline `code_bert`** (reuse saved codes, no regen):

```bash
python run_codecheck.py codebert --results results/run.json   # fills missing code_bert in place
python run_codecheck.py evaluate --results results/run.json    # now shows a [code_bert] block
```

Default fills only results lacking `code_bert` (resumes partial pass cheaply); `--recompute`
rescores all.

**Offline `prompt`** (re-score judge on saved codes, no regen — e.g. apply updated judge
template). Unlike `codebert` it **calls the API** (needs `OPENROUTER_API_KEY`). Template chosen
per result from `task_id` prefix (HumanEval/* → sharper divergence prompt), or forced via
`--dataset`:

```bash
python run_codecheck.py prompt --results results/run.json                  # auto per task_id
python run_codecheck.py prompt --results results/run.json --dataset humaneval
```

**Rewrites in place** — point at a finished/copied file, not one a live run is still writing.

**`--ast-metric {jaccard,ted}`** (only with `ast`):

- `jaccard` (default) — `1 − multiset Jaccard` of AST node-type counts. Count-based, ignores
  nesting/ordering. Out-discriminated `ted` on MBPP+ (`docs/reports/09-codecheck-ast-ted-result.md`).
- `ted` — tree edit distance (Zhang-Shasha, via `zss`), normalized `[0,1]`. Structure-aware,
  but on MBPP+ penalizes correct-but-restructured samples and compresses the signal. Available,
  not default.

Judge reuses `OPENROUTER_MODEL`. After a run the CLI prints judge + AST parse-failure counts.

```bash
python run_codecheck.py run --method all --limit 30 --n 5 --seed 1 --timeout 5 \
  --output output/iter3-all.json
python run_codecheck.py evaluate --results output/iter3-all.json
```

### Run/report protocol (iteration-1 findings)

- Sample problems **randomly across the full set** with a capable model — never the
  low-numbered slice (unrepresentatively easy, single class).
- `evaluate` prints per method: two trapezoidal **AUC-PR** (detect-incorrect = paper `nonfact`,
  scores as-is; detect-correct = paper `factual`, negated), **Pearson/Spearman** of score vs
  incorrectness, **prevalence baseline** (PR no-skill floor), **per-class score histogram**.
  Same `precision_recall_curve` + `auc` as the WikiBio replication
  (`replication/evaluation/metrics.py`). Read AUC-PR against baseline; histogram exposes tie
  pile-ups at 0 that make the scalar fragile.

## Datasets

`run --dataset {mbpp,humaneval,codehalu}` (default `mbpp`). **MBPP+** (378) and **HumanEval+**
(164) are EvalPlus **function-call** datasets — each a function with a test-input suite that
doubles as the shared run inputs — share the whole pipeline + all four scorers. **CodeHaluEval**
is **whole-program stdin→stdout** (Codeforces-style); adjusts Exec harness + Prompt template.

### MBPP+ (378)

378 Python problems, each with reference solution + test inputs (extended MBPP). Test inputs =
the shared run inputs.

| Part | What |
|------|------|
| ID | problem identifier |
| Prompt | function signature + docstring shown to model |
| Function name | function to call |
| Reference solution | known-correct impl, grading only |
| Inputs | many argument sets |
| Tolerance | float-equality tolerance |

Example (`Mbpp/2`): "find shared elements from two lists."

- **Function:** `similar_elements`
- **Reference:**
  ```python
  def similar_elements(test_tup1, test_tup2):
      return tuple(set(test_tup1) & set(test_tup2))
  ```
- **Inputs:** `similar_elements((3,4,5,6),(5,7,4,10))` → `(4,5)`; … ~100 more (empty, large,
  duplicates).
- **Tolerance:** `0` (exact).

### HumanEval+ (164)

EvalPlus extension of HumanEval: 164 function problems + expanded test inputs. Same shape as
MBPP+, one difference: **prompt** = imports + signature + docstring (text shown to model);
**reference** = function *body only* → runnable reference is `prompt + body` (loader assembles).

| Part | What |
|------|------|
| ID | e.g. `HumanEval/0` |
| Prompt | imports + signature + docstring (doctests) |
| Function name | function to call |
| Reference solution | **body only**; runnable as `prompt + body` |
| Inputs | many argument sets |
| Tolerance | float-equality tolerance |

Example (`HumanEval/0`): "any two numbers closer than a threshold."

- **Function:** `has_close_elements`
- **Prompt:**
  ```python
  from typing import List


  def has_close_elements(numbers: List[float], threshold: float) -> bool:
      """ Check if in given list of numbers, are any two numbers closer to each other than
      given threshold.
      >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
      False
      >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
      True
      """
  ```
- **Reference** (body only):
  ```python
      sorted_numbers = sorted(numbers)
      for i in range(len(sorted_numbers) - 1):
          if sorted_numbers[i + 1] - sorted_numbers[i] < threshold:
              return True
      return False
  ```
- **Inputs:** `has_close_elements([1.0,2.0,3.9,4.0,5.0,2.2], 0.3)` → `True`; … ~1000 more (7
  base + 999 plus).
- **Tolerance:** `0` (boolean).

### CodeHaluEval (stdin/stdout)

`Yuchen111/CodeHaluEval`: hallucination-focused Codeforces-style problems reading **stdin** /
writing **stdout** (no callable). Run its stdin/stdout tasks (skip `fn_name` call tasks). Built
to induce hallucinations → real incorrect class, unlike saturated HumanEval+.

Two pieces differ; four metrics unchanged:

- **Exec — whole-program harness.** Each impl runs in fresh `python` subprocess, task stdin
  piped in, stdout captured + normalized (unify line endings, strip trailing ws), compared
  across impls. Same behavioral metric over stdout strings.
- **Prompt — program-oriented judge.** stdin/stdout variant of divergence-seeking judge ("same
  output for every stdin?"), auto-selected by `--dataset codehalu`. Same Yes/No/N-A mapping.
- **AST / CodeBERT — reused, weaker.** Competitive solutions share heavy I/O boilerplate
  (`input()`, parsing) → inflated structural/embedding similarity → both degrade further.

Whole-program competitive problems are much harder than one function, so **generation reasons
by default** on `--dataset codehalu` (= `--think` for main + samples); other datasets stay
reasoning-off unless `--think`.

Ground truth ships with data: each test case carries expected stdout → correctness labeled
direct, reference **never run** (stored `solutions` often partial/absent → reference-only).
Numeric/formatting output compared as exact strings → can over-count `fail` on float-format
differences (deferred refinement).

Per problem we record: consistency score, `is_correct`, `is_error` (main raised/timed out on
any input vs ran-but-wrong), `count` (per-input breakdown vs canonical: `{total, pass, fail,
error}`, summing to `total`), original `prompt`, and generated code.

## Running it

```bash
python run_codecheck.py run --limit 10 --n 5 --timeout 5    # generate, score, save
python run_codecheck.py evaluate                            # report detection AUC-PR
```

**`run` parameters**

- `--dataset` — `mbpp` (default), `humaneval`, or `codehalu`.
- `--limit` — problems to use (default: entire dataset).
- `--index` — single problem at this 0-based position; not combinable with `--limit`/`--random`/`--seed`.
- `--n` — samples per problem (temp 1).
- `--method` — `exec` (default), `prompt`, `ast`, `code_bert`, `all`.
- `--ast-metric` — `jaccard` (default) or `ted` (only with `ast`).
- `--timeout` — max seconds per code execution before kill.
- `--api-timeout` — per-API-call wall-clock budget (SDK request + retry cap). Default 300s when
  reasoning on (`--think` or `--dataset codehalu`), else 60s. Reasoning calls take minutes; the
  60s cap would abort every one and fail the run.
- `--output` — results path (JSON array, rewritten per problem so progress is never lost).
  Existing file → **auto-resume** (skip recorded `task_id`s, add the rest).
- `--seed` — reproducible sample (with `--random`).
- `--random` — random `--limit` sample instead of first `--limit` in order.
- `-v` / `--verbose` — log per-call API detail (latency, `finish_reason`, tokens) at DEBUG.
  Without it, still warns on truncated (`finish_reason != stop`) / empty responses.

Default = entire dataset in order; `--limit N` takes first N; `--random` (+ `--seed`) for a
reproducible random sample. Saved results record which problems were used → reproducible.

**`evaluate` parameters**

- `--results` — results file to score.

Generation calls a hosted model → needs an API key in local `.env`. Evaluation runs offline.

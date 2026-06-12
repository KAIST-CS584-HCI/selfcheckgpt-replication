# SelfCheckGPT for Code

Brings SelfCheckGPT's core idea to code generation: a model that truly "knows" a
solution tends to write it consistently across many tries; a hallucinated solution drifts.
Here a hallucination just means **incorrect code** and correctness is checked by actually
running the code, so no human labeling is needed.

For each problem we ask the model for one main answer (at temperature 0) and several
extra samples (at temperature 1). From these we get two things: a **consistency signal**
(how much the samples agree with the main answer) and a **ground-truth correctness label**
(whether the main answer actually works).

## How the methods work

All three variants share the same generation and ground-truth steps; they differ only in
how they measure consistency between the main answer and the samples. Each produces a score
on the same scale: **higher = more likely hallucinated (incorrect)**.

### SelfCheck-Exec — behavioral consistency

Measures consistency through *behavior*: do the implementations produce the same outputs
when given the same inputs?

1. **Generate.** Ask the model for one main implementation and several sampled
   implementations of the same problem.
2. **Run them.** Execute every implementation on a shared set of inputs, in an isolated,
   time-limited sandbox so bad code can't hang or harm the run. Each run ends as a value,
   an error, or a timeout.
3. **Compare outputs.** Two implementations "agree" on an input when they return the same
   result (with a small tolerance for floating-point numbers).
4. **Score consistency.** Measure how often the samples disagree with the main answer.
   Lots of agreement means a low score (looks reliable); lots of disagreement means a high
   score (looks hallucinated).
5. **Check correctness.** Separately, run the problem's known-good reference solution on
   the same inputs and see whether the main answer matches it. This is the ground truth.
6. **Evaluate.** Across many problems, check how well the consistency score predicts the
   incorrect answers.

### SelfCheck-Prompt — LLM-as-judge

Measures consistency by *asking a model* instead of running the code. For each sampled
implementation, a judge LLM is shown the main implementation and that sample and asked
whether the sample's behavior is consistent with the main one.

1. **Generate.** Same as Exec — one main implementation plus several samples.
2. **Judge each sample.** For every sample, ask the judge: is its behavior consistent with
   the main implementation? It answers Yes, No, or N/A with a one-sentence justification.
   The N judgments run concurrently.
3. **Map answers.** Yes (consistent) → `0.0`, No (inconsistent) → `1.0`, N/A → `0.5`. An
   unparseable answer counts as `0.5` and is tallied as a parse failure.
4. **Score consistency.** Average the per-sample values: low when the judge keeps saying the
   samples match the main answer, high when it flags disagreement.
5. **Check correctness / evaluate.** Identical to Exec — the ground-truth label still comes
   from running the reference solution; the judge never sees it.

The judge needs no execution, so it can flag cases where the code runs but is subtly wrong —
complementary to Exec, which is blind to samples that consistently agree on the *same* wrong
behavior. The two run on the **same generated implementations**, so their scores are directly
comparable.

### SelfCheck-AST — structural consistency

Measures consistency by *parsing the code* instead of running it or asking a model. Each
implementation is reduced to a structural fingerprint and the samples are compared against
the main one.

1. **Generate.** Same as Exec/Prompt — one main implementation plus several samples.
2. **Fingerprint.** Parse each implementation to an AST and count node types
   (`Name`, `BinOp`, `Constant`, …), ignoring identifier text and literal values — so
   renaming variables or changing constants does not change the fingerprint.
3. **Compare structure.** Dissimilarity = `1 − multiset Jaccard` of the two fingerprints:
   `0.0` for identical structure, up to `1.0` for no shared node types.
4. **Score consistency.** Average the per-sample dissimilarities. An implementation that does
   not parse counts as `1.0` (maximally divergent) and is tallied as a parse failure.
5. **Check correctness / evaluate.** Identical to Exec — the ground-truth label still comes
   from running the reference solution.

AST adds **no API cost** (pure local parsing). It is hypothesized to share Exec's
confident-consistent blind spot: when a model is consistently wrong, the samples are also
structurally similar to the wrong main, so AST scores ≈0 and misses it.

## Methods

`run --method {exec,prompt,ast,all}` selects the consistency scorer(s). A single run
generates each problem's implementations once and scores them with every selected
method, so the methods are always compared on identical data.

- `exec` (default) — behavioral I/O divergence across samples (described above).
- `prompt` — LLM-as-judge: per sample, is its behavior consistent with the main
  implementation? Yes→0.0 / No→1.0 / N-A→0.5, averaged over the N samples. On HumanEval+ a
  sharper, still oracle-free judge prompt is used (it makes the judge hunt for an edge-case
  input on which the two implementations diverge, instead of affirming surface similarity),
  because a capable model writes near-identical samples on those canonical problems.
- `ast` — structural divergence between the main and each sample, rename- and
  literal-invariant, averaged over the N samples. No API cost. The metric is chosen with
  `--ast-metric` (below).
- `code_bert` — embedding divergence: mean `1 − cosine` of CodeBERT
  (`microsoft/codebert-base`, mean-pooled) embeddings of the main vs each sample. A
  learned semantic/lexical similarity (not rename-invariant, unlike `ast`). No API cost,
  but loads a ~500MB model on first use. Usually run **offline** (see below).
- `all` — runs exec + prompt + ast + code_bert; `evaluate` prints a per-method comparison.

**Offline `code_bert` (reuse saved results):** results already store `main_code` +
`sample_codes`, so you can add `code_bert` to an existing file without regenerating:

```bash
python run_codecheck.py codebert --results results/run.json   # adds code_bert in place
python run_codecheck.py evaluate --results results/run.json    # now shows a [code_bert] block
```

**Offline `prompt` (re-score the judge on saved results):** same idea for the Prompt
variant — recompute `scores["prompt"]` from the stored code without regenerating (useful to
apply an updated judge template to an existing file). Unlike `codebert` it **calls the API**,
so it needs `OPENROUTER_API_KEY`. The judge template is chosen per result from the `task_id`
prefix (HumanEval/* uses the sharper divergence-seeking prompt), or forced with `--dataset`:

```bash
python run_codecheck.py prompt --results results/run.json                  # auto per task_id
python run_codecheck.py prompt --results results/run.json --dataset humaneval
```

It **rewrites the file in place** — point it at a finished or copied results file, not one a
live run is still writing.

**`--ast-metric {jaccard,ted}`** (only used when `--method` includes `ast`):

- `jaccard` (default) — `1 − multiset Jaccard` of AST node-type counts. Count-based: it
  ignores nesting/ordering. The default because it out-discriminated `ted` on MBPP+
  (see `docs/reports/09-codecheck-ast-ted-result.md`).
- `ted` — tree edit distance (Zhang-Shasha, via `zss`) between the AST shapes.
  Structure-aware (two programs with the same node-type counts but different nesting score
  apart), normalized to `[0,1]`. More sensitive to shape, but on MBPP+ that penalizes
  correct-but-restructured samples and compresses the signal — kept available, not default.

The judge reuses `OPENROUTER_MODEL`. After a run the CLI prints the judge parse-failure
count (if prompt ran) and the AST parse-failure count (if ast ran).

Three-way run:

```bash
python run_codecheck.py run --method all --limit 30 --n 5 --seed 1 --timeout 5 \
  --output output/iter3-all.json
python run_codecheck.py evaluate --results output/iter3-all.json
```

### Standing run/report protocol (from iteration-1 findings)

- Sample problems **randomly across the full set** with a capable model — never the
  low-numbered slice (it is unrepresentatively easy and yields a single class).
- `evaluate` prints, per method, two trapezoidal **AUC-PR** numbers — detect-incorrect
  (the paper's `nonfact`, scores as-is) and detect-correct (the paper's `factual`,
  negated scores) — the **Pearson/Spearman** correlation of score vs incorrectness, the
  **prevalence baseline** (the PR no-skill floor), and a **per-class score histogram**.
  Same `precision_recall_curve` + `auc` as the WikiBio replication
  (`replication/evaluation/metrics.py`). Read AUC-PR against the baseline, not in
  isolation; the histogram exposes tie pile-ups at 0 that make the scalar fragile.

## Datasets: MBPP+, HumanEval+, and CodeHaluEval

Selected with `run --dataset {mbpp,humaneval,codehalu}` (default `mbpp`). **MBPP+** (378
problems) and **HumanEval+** (164 problems) are EvalPlus **function-call** datasets — each
problem is a function with a rich suite of test inputs that double as the shared inputs we
run every implementation on — so they share the entire pipeline and all four scorers.
**CodeHaluEval** is a **whole-program stdin→stdout** dataset (Codeforces-style) and adjusts
the Exec harness and the Prompt template; see its subsection below.

### MBPP+ (378 problems)

MBPP+ is a set of 378 Python programming problems, each shipped with a reference solution
and a rich suite of test inputs (an extended version of the MBPP benchmark). The test inputs
double as the shared inputs we run every implementation on.

Each problem gives us:

| Part                | What it is                                                   |
|---------------------|--------------------------------------------------------------|
| ID                  | a problem identifier                                         |
| Prompt              | the function signature and docstring shown to the model      |
| Function name       | the function to call                                         |
| Reference solution  | a known-correct implementation, used only for grading        |
| Inputs              | many argument sets the function is called with               |
| Tolerance           | how close floating-point results must be to count as equal   |

Example problem (real row `Mbpp/2`):

> **Task:** Write a function to find the shared elements from two lists.

- **Function name:** `similar_elements`
- **Reference solution:**
  ```python
  def similar_elements(test_tup1, test_tup2):
      return tuple(set(test_tup1) & set(test_tup2))
  ```
- **Inputs the function is tested on** (two tuples per call):
  - `similar_elements((3, 4, 5, 6), (5, 7, 4, 10))` → expects `(4, 5)`
  - `similar_elements((1, 2, 3, 4), (5, 4, 3, 7))` → expects `(3, 4)`
  - `similar_elements((11, 12, 14, 13), (17, 15, 14, 13))` → expects `(13, 14)`
  - ...plus ~100 more edge-case inputs (empty tuples, large tuples, duplicates)
- **Tolerance:** `0` (exact match; this problem has no floating-point results)

### HumanEval+ (164 problems)

HumanEval+ is the EvalPlus extension of OpenAI's HumanEval: 164 Python problems, each a
function with an expanded suite of test inputs. Same function-call shape as MBPP+, with one
structural difference: the **prompt** is the import block + signature + docstring (the actual
text shown to the model), and the **reference solution** is the function *body only* — so the
runnable reference is `prompt + body` (the loader assembles this for grading).

Each problem gives us:

| Part                | What it is                                                       |
|---------------------|------------------------------------------------------------------|
| ID                  | a problem identifier (e.g. `HumanEval/0`)                        |
| Prompt              | imports + function signature + docstring (with doctest examples) |
| Function name       | the function to call                                             |
| Reference solution  | the function **body only**; runnable as `prompt + body`         |
| Inputs              | many argument sets the function is called with                   |
| Tolerance           | how close floating-point results must be to count as equal       |

Example problem (real row `HumanEval/0`):

> **Task:** Check whether any two numbers in a list are closer than a given threshold.

- **Function name:** `has_close_elements`
- **Prompt** (shown to the model):
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
- **Reference solution** (body only — the dataset stores just this):
  ```python
      sorted_numbers = sorted(numbers)
      for i in range(len(sorted_numbers) - 1):
          if sorted_numbers[i + 1] - sorted_numbers[i] < threshold:
              return True
      return False
  ```
  The runnable canonical used for grading is `prompt + body` — the signature from the prompt
  plus this body.
- **Inputs the function is tested on** (a list and a threshold per call):
  - `has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3)` → expects `True`
  - `has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05)` → expects `False`
  - `has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95)` → expects `True`
  - ...plus ~1000 more edge-case inputs (7 base + 999 plus inputs)
- **Tolerance:** `0` (exact match; the result is a boolean)

### CodeHaluEval (stdin/stdout)

CodeHaluEval (`Yuchen111/CodeHaluEval`) is a hallucination-focused benchmark of
Codeforces-style problems that read **stdin** and write **stdout** (no callable). We run its
stdin/stdout tasks (the `fn_name`-style call tasks are skipped). It is built to induce
hallucinations, so unlike the saturated HumanEval+ it gives a real incorrect class to stress
the confident-consistent blind spot.

Two pieces differ from the function-call datasets; the four scoring metrics are unchanged:

- **Exec — whole-program harness.** Each implementation runs in a fresh `python` subprocess
  with the task's stdin piped in; its stdout is captured, normalized (unify line endings,
  strip trailing whitespace), and compared across implementations. Same behavioral-divergence
  metric, over stdout strings instead of return values.
- **Prompt — program-oriented judge.** A stdin/stdout variant of the divergence-seeking judge
  ("would these two programs print the same output for every stdin?"), selected automatically
  by `--dataset codehalu`. Same Yes/No/N-A mapping.
- **AST / CodeBERT — reused, weaker here.** Competitive solutions share heavy I/O boilerplate
  (`input()`, parsing), which inflates structural and embedding similarity, so both
  local-similarity methods are expected to degrade further than on the function-call datasets.

Because these are whole-program competitive problems (far harder than a single function),
**generation reasons by default** on `--dataset codehalu` (equivalent to `--think` for the
main + samples), to produce usable programs; the other datasets stay reasoning-off unless
`--think` is passed.

Ground truth ships with the data: each test case carries its expected stdout, so correctness
is labeled directly against it and the reference solution is **never run**. (CodeHaluEval's
stored `solutions` are often partial fragments or absent, so they are reference-only.)
Numeric/formatting-sensitive output is compared as exact strings, which can over-count `fail`
on float-formatting differences — a deferred refinement, noted in the iteration plan.

For each problem we record the consistency score, the correctness label (`is_correct`),
an error flag (`is_error` — the main raised or timed out on any input, vs ran but gave a
wrong answer), `count` (a per-input breakdown vs the canonical: `{total, pass, fail,
error}` — `pass` matched, `fail` ran but wrong, `error` raised/timed out, summing to
`total`), the original problem `prompt`, and the actual code that was generated, so
results can be inspected later.

## Running it

```bash
# generate, score, and save results for a handful of problems
python run_codecheck.py run --limit 10 --n 5 --timeout 5

# report how well the consistency score detected incorrect answers
python run_codecheck.py evaluate
```

**`run` parameters**

- `--dataset` — `mbpp` (default, MBPP+), `humaneval` (HumanEval+), or `codehalu`
  (CodeHaluEval stdin/stdout)
- `--limit` — how many problems to use (default: the entire dataset)
- `--index` — run only the single problem at this 0-based dataset position; cannot be
  combined with `--limit`/`--random`/`--seed`
- `--n` — sampled implementations per problem (extra tries at temperature 1)
- `--method` — consistency scorer: `exec` (default), `prompt`, `ast`, `code_bert`, or `all`
- `--ast-metric` — AST metric when `ast` runs: `jaccard` (default) or `ted`
- `--timeout` — max seconds per code execution before it's killed
- `--output` — where to save the results (JSON array, rewritten after each problem
  finishes so progress is never lost). If the file already exists, the run
  **auto-resumes**: problems whose `task_id` is already recorded are skipped and only the
  remainder is added.
- `--seed` — random seed for a reproducible sample (only with `--random`)
- `--random` — take a random sample of `--limit` problems instead of the first `--limit` in order
- `-v` / `--verbose` — log per-call API detail (latency, `finish_reason`, completion tokens)
  at DEBUG. Without it, a run still warns on truncated (`finish_reason != stop`) or empty
  responses.

By default the run uses the **entire dataset** in order; pass `--limit N` to take the first
`N` in dataset order, and `--random` for a random sample of that size (add `--seed` to
reproduce it). The saved results record which problems were used, so any run is reproducible
from its output.

**`evaluate` parameters**

- `--results` — results file to score

Generation calls a hosted model, so it needs an API key in a local `.env` file. Evaluation
runs offline on the saved results.

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

Both variants share the same generation and ground-truth steps; they differ only in how
they measure consistency between the main answer and the samples. Both produce a score on
the same scale: **higher = more likely hallucinated (incorrect)**.

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

## Methods

`run --method {exec,prompt,both}` selects the consistency scorer(s). A single run
generates each problem's implementations once and scores them with every selected
method, so Exec and Prompt are always compared on identical data.

- `exec` (default) — behavioral I/O divergence across samples (described above).
- `prompt` — LLM-as-judge: per sample, is its behavior consistent with the main
  implementation? Yes→0.0 / No→1.0 / N-A→0.5, averaged over the N samples.
- `both` — runs both; `evaluate` then prints a per-method comparison.

The judge reuses `OPENROUTER_MODEL`. After a judged run the CLI prints the judge
parse-failure count.

### Standing run/report protocol (from iteration-1 findings)

- Sample problems **randomly across the full set** with a capable model — never the
  low-numbered slice (it is unrepresentatively easy and yields a single class).
- `evaluate` prints, per method, the AUC-PR, the **prevalence baseline** (the PR
  no-skill floor), and a **per-class score histogram**. Read AUC-PR against the
  baseline, not in isolation; the histogram exposes tie pile-ups at 0 that make
  the scalar fragile.

## Dataset in use: MBPP+ (378 problems)

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

For each problem we record the consistency score, the correctness label, and the actual
code that was generated, so results can be inspected later.

## Running it

```bash
# generate, score, and save results for a handful of problems
python run_codecheck.py run --limit 10 --n 5 --timeout 5

# report how well the consistency score detected incorrect answers
python run_codecheck.py evaluate
```

**`run` parameters**

- `--limit` — how many problems to use
- `--n` — sampled implementations per problem (extra tries at temperature 1)
- `--method` — consistency scorer: `exec` (default), `prompt`, or `both`
- `--timeout` — max seconds per code execution before it's killed
- `--output` — where to save the results file
- `--seed` — random seed for a reproducible problem sample
- `--no-random` — use the first `--limit` problems in order instead of a random sample

By default the problems are sampled at random (use `--seed` to reproduce a sample, or
`--no-random` for the first `--limit` in dataset order). The saved results record which
problems were used, so any run is reproducible from its output.

**`evaluate` parameters**

- `--results` — results file to score

Generation calls a hosted model, so it needs an API key in a local `.env` file. Evaluation
runs offline on the saved results.

# SelfCheckGPT for Code

SelfCheckGPT idea on code: model that knows a solution writes it consistently across tries;
hallucinated solution drifts. Hallucination = **incorrect code**; correctness checked by
running it (no human labels).

Per problem: one main answer (temp 0) + N samples (temp 1). Yields a **consistency signal**
(sample agreement with main) and a **ground-truth label** (does main actually work).

## Methods

All variants share generation + ground-truth steps; differ only in how they measure
main-vs-sample consistency. Score scale: **higher = more likely incorrect**.
`run --method {exec,prompt,ast,code_bert,all}`; one run generates each problem's impls once,
scores with every selected method, so methods always compared on identical data. `all` makes
`evaluate` print a per-method comparison.

### SelfCheck-Exec (behavioral, default)

Run every impl on shared inputs in isolated, time-limited sandbox; each run gives a value,
error, or timeout. Two impls "agree" on an input when same result (float tolerance). Score =
how often samples disagree with main. Correctness: run reference solution on same inputs, check
main matches. Blind to samples that consistently agree on the *same* wrong behavior.

### SelfCheck-Prompt (LLM-as-judge)

No execution: judge LLM sees main + each sample, asks if sample behavior consistent with main.
Yes→`0.0`, No→`1.0`, N/A→`0.5`; unparseable maps to `0.5`, counted as parse failure. Score =
mean over N; judgments run concurrently; judge never sees ground truth. Catches code that runs
but is subtly wrong, so complementary to Exec. On HumanEval+ a sharper, oracle-free judge
prompt is used (hunts for an edge-case input where impls diverge), because a capable model
writes near-identical samples on those canonical problems. Judge reuses `OPENROUTER_MODEL`.

### SelfCheck-AST (structural)

Parse each impl to AST, count node types (`Name`, `BinOp`, …), ignore identifier text +
literal values for a rename/literal-invariant fingerprint. Dissimilarity = `1 − multiset
Jaccard`; score = mean per-sample dissimilarity; unparseable impl maps to `1.0`, counted as
parse failure. No API cost. Shares Exec's blind spot: consistently-wrong samples are also
structurally similar to wrong main, so AST ≈0 misses it.

`--ast-metric {jaccard,ted}` (only with `ast`): `jaccard` (default) count-based, ignores
nesting/ordering, out-discriminated `ted` on MBPP+
(`docs/reports/09-codecheck-ast-ted-result.md`). `ted` = tree edit distance (Zhang-Shasha via
`zss`), normalized `[0,1]`; structure-aware but penalizes correct-but-restructured samples and
compresses the signal. Available, not default.

### SelfCheck-CodeBERT (embedding)

Mean `1 − cosine` of CodeBERT (`microsoft/codebert-base`, mean-pooled) embeddings, main vs
each sample. Learned semantic/lexical similarity (not rename-invariant). No API cost; loads
~500MB model on first use. Usually run **offline**.

### Offline re-scoring (saved codes, no regen)

```bash
python run_codecheck.py codebert --results results/run.json   # fills missing code_bert in place
python run_codecheck.py prompt --results results/run.json     # re-runs judge (calls API)
python run_codecheck.py evaluate --results results/run.json
```

`codebert` default fills only results lacking `code_bert` (resumes partial pass cheaply);
`--recompute` rescores all. `prompt` needs `OPENROUTER_API_KEY`; judge template chosen per
result from `task_id` prefix (HumanEval/* uses sharper divergence prompt) or forced via
`--dataset`. Both **rewrite in place**: point at a finished/copied file, not one a live run
is still writing.

## Run/report protocol (iteration-1 findings)

- Sample problems **randomly across the full set** with a capable model, never the
  low-numbered slice (unrepresentatively easy, single class).
- `evaluate` prints per method: two trapezoidal **AUC-PR** (detect-incorrect = paper `nonfact`,
  scores as-is; detect-correct = paper `factual`, negated), **Pearson/Spearman** of score vs
  incorrectness, **prevalence baseline** (PR no-skill floor), **per-class score histogram**.
  Same `precision_recall_curve` + `auc` as the WikiBio replication
  (`replication/evaluation/metrics.py`). Read AUC-PR against baseline; histogram exposes tie
  pile-ups at 0 that make the scalar fragile.
- After a run the CLI prints judge + AST parse-failure counts.

## Datasets

`run --dataset {mbpp,humaneval,codehalu}` (default `mbpp`).

**MBPP+** (378) and **HumanEval+** (164): EvalPlus **function-call** datasets, each problem a
function whose test-input suite doubles as the shared run inputs; share the whole pipeline +
all four scorers. Per problem: ID, prompt (signature + docstring), function name, reference
solution (grading only), many input argument sets, float-equality tolerance. HumanEval+
difference: prompt = imports + signature + docstring; reference = function *body only*, so the
runnable reference is `prompt + body` (loader assembles).

**CodeHaluEval** (`Yuchen111/CodeHaluEval`): hallucination-focused Codeforces-style
**whole-program stdin→stdout** problems (no callable; `fn_name` call tasks skipped). Built to
induce hallucinations, giving a real incorrect class unlike saturated HumanEval+. Two pieces
differ, four metrics unchanged:

- **Exec (whole-program harness).** Each impl runs in fresh `python` subprocess, task stdin
  piped in, stdout captured + normalized (unify line endings, strip trailing ws), compared
  across impls.
- **Prompt (program-oriented judge).** stdin/stdout variant of divergence-seeking judge ("same
  output for every stdin?"), auto-selected by `--dataset codehalu`. Same Yes/No/N-A mapping.
- **AST / CodeBERT (reused, weaker).** Competitive solutions share heavy I/O boilerplate,
  inflating structural/embedding similarity, so both degrade further.

Whole-program competitive problems much harder than one function, so **generation reasons by
default** on `--dataset codehalu` (= `--think` for main + samples); other datasets stay
reasoning-off unless `--think`.

Ground truth ships with data: each test case carries expected stdout, so correctness labeled
direct and reference **never run** (stored `solutions` often partial/absent). Output compared
as exact strings, so can over-count `fail` on float-format differences (deferred refinement).

Per problem recorded: consistency score, `is_correct`, `is_error` (main raised/timed out on
any input vs ran-but-wrong), `count` (per-input breakdown vs canonical: `{total, pass, fail,
error}`), original `prompt`, generated code.

## Running it

```bash
python run_codecheck.py run --method all --limit 30 --n 5 --seed 1 --timeout 5 \
  --output output/iter3-all.json
python run_codecheck.py evaluate --results output/iter3-all.json
```

**`run` parameters**

- `--dataset`: `mbpp` (default), `humaneval`, `codehalu`.
- `--limit`: problems to use (default: entire dataset, in order).
- `--index`: single problem at this 0-based position; not combinable with `--limit`/`--random`/`--seed`.
- `--n`: samples per problem (temp 1).
- `--method`: `exec` (default), `prompt`, `ast`, `code_bert`, `all`.
- `--ast-metric`: `jaccard` (default) or `ted` (only with `ast`).
- `--timeout`: max seconds per code execution before kill.
- `--api-timeout`: per-API-call wall-clock budget (SDK request + retry cap). Default 300s when
  reasoning on (`--think` or `--dataset codehalu`), else 60s; reasoning calls take minutes.
- `--output`: results path (JSON array, rewritten per problem so progress never lost). Existing
  file triggers **auto-resume** (skip recorded `task_id`s, add the rest).
- `--random` / `--seed`: reproducible random `--limit` sample instead of first `--limit` in
  order. Saved results record which problems were used, so reproducible.
- `-v` / `--verbose`: per-call API detail (latency, `finish_reason`, tokens) at DEBUG. Without
  it, still warns on truncated (`finish_reason != stop`) / empty responses.

**`evaluate` parameters**: `--results`, results file to score.

Generation calls a hosted model, so needs an API key in local `.env`. Evaluation runs offline.

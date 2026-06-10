# SelfCheckGPT for Code: Consistency Methods (Improvement 2 — Coding Domain)

Status: design, pre-implementation. Feeds the plan phase.

## Goal

Adapt SelfCheckGPT's sample-consistency principle to **code generation**, where
hallucination = an incorrect implementation. Correctness is auto-verifiable
(unit tests / execution), so no human annotation is needed — the ground-truth
label for each generated implementation comes from running it, not from a
labeler.

## Generation setup

Same sampling scheme as the original paper, applied to code:

- **Main implementation:** 1 response at temperature `T = 0`.
- **Sample implementations:** `N = 20` responses at temperature `T = 1`.

The unit being scored is the **implementation itself** (one function/program),
analogous to a "sentence" in the original WikiBio setup.

### Unit granularity — RESOLVED (2026-06-10): per-problem / per-main-implementation

The scored unit is **the single `T = 0` main implementation per problem**. Its
ground-truth label is that implementation's execution correctness. The `N`
sample implementations are **consistency evidence only — never scored as units
themselves**.

This is faithful to SelfCheckGPT: in WikiBio only the main passage `R` is
evaluated (its sentences are the units); the sample passages `S1..SN` are
evidence used to score `R`, never labeled or scored on their own. The code
analogue maps `R` → main implementation (one function ≈ one "sentence") and
`S1..SN` → sample implementations (evidence).

Rejected alternative — *per-sample* (treat each of the `N+1` implementations as
its own labeled unit). It would deviate from the paper: it requires a
self-referential evidence set the paper never defines, and it breaks the Prompt
template below (which compares a "construct from another implementation"
*against* "the implementation above" = the main). It would also change the Exec
and AST metrics' meaning. The one genuine benefit (more labeled units per
problem → less fragile AUC-PR under heavy score ties) is instead addressed by
**scaling the number of problems** (roadmap iteration 4), not by redefining what
a unit is. All three variants share this single definition.

## Methods

Three variants. Two replace original NL-specific variants with code-aware
equivalents; one carries over directly.

### 1. SelfCheck-Exec — *replaces BERTScore*

Behavioral consistency via execution.

- Construct a shared set of inputs `(x1, x2, ...)`.
- Run **every** implementation (main + all N samples) on the same inputs.
- Compare output values across implementations.
- Rationale: a correct implementation is functionally deterministic — same
  inputs produce same outputs; hallucinated implementations diverge in output.

Replaces BERTScore (semantic-similarity of text) because behavioral equivalence
is a stronger, code-native consistency signal than surface text similarity.

Open design points (for plan phase):
- input set construction (provided test inputs vs. generated/fuzzed inputs)
- output comparison rule (exact match, tolerance, exception handling)
- handling non-terminating / crashing samples (timeout, sandbox)

### 2. SelfCheck-AST — *replaces n-gram*

Structural consistency via abstract syntax tree.

- Parse main and sample implementations into ASTs.
- Compute AST tree similarity between main and each sample.
- Rationale: hallucinated implementations vary a lot in semantic structure;
  consistent (factual) ones share structure.

Replaces the n-gram score (token-surface model) with a structure-aware metric.

Open design points:
- AST similarity metric (tree edit distance, subtree overlap, etc.)
- normalization for identifier renaming / equivalent constructs
- parse-failure handling for malformed samples

### 3. SelfCheck-Prompt — *carried over*

LLM-as-judge, same idea as the original Prompt variant.

Prompt template (per unit):

```
Does the following construct from another implementation
have behavior consistent with the implementation above?
Construct: {unit from R}
Answer Yes / No / N/A with a one-sentence justification.
```

Score aggregates Yes/No/N-A judgments across the N samples (mapping per the
original Prompt variant, e.g. Yes→0.0 / No→1.0 / N-A→0.5).

## Variant mapping summary

| Original (NL) | Code variant   | Consistency signal      |
|---------------|----------------|-------------------------|
| BERTScore     | SelfCheck-Exec | behavioral (I/O)        |
| n-gram        | SelfCheck-AST  | structural (AST)        |
| Prompt        | SelfCheck-Prompt | LLM judge             |
| NLI, QA       | (dropped)      | —                       |

Note: CodeBERT and Code-NLI (earlier proposal candidates) are **not** used.

## Datasets

Coding-task datasets under consideration:

- **CodeHaluEval**
- **Collu-Bench**
- **MBPP+**

## Evaluation

Same as replication: sentence/unit-level **AUC-PR** plus passage-level
correlation, with the per-implementation correctness label derived from
execution/tests as ground truth.

# SelfCheckGPT for Code

Brings SelfCheckGPT's core idea to code generation: a model that truly "knows" a
solution tends to write it consistently across many tries; a hallucinated solution drifts.
Here a hallucination just means **incorrect code** and correctness is checked by actually
running the code, so no human labeling is needed.

For each problem we ask the model for one main answer (at temperature 0) and several
extra samples (at temperature 1). From these we get two things: a **consistency signal**
(how much the samples agree with the main answer) and a **ground-truth correctness label**
(whether the main answer actually works).

## How the method works

**SelfCheck-Exec** measures consistency through *behavior*: do the implementations produce
the same outputs when given the same inputs?

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

Two other variants are planned but not built yet: one comparing the *structure* of the
code, and one asking a separate model to judge whether the samples behave like the main
answer.

## Dataset in use: MBPP+

MBPP+ is a set of Python programming problems, each shipped with a reference solution and
a rich suite of test inputs (an extended version of the MBPP benchmark). The test inputs
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

Raw example (one row, abridged):

```python
{
  "task_id": "Mbpp/2",
  "prompt": '"""\nWrite a function to find the shared elements from the given two lists.\n'
            'assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))\n"""\n',
  "entry_point": "similar_elements",
  "canonical_solution": "def similar_elements(test_tup1, test_tup2):\n"
                        "  return tuple(set(test_tup1) & set(test_tup2))\n",
  "base_input": [
    [(3, 4, 5, 6), (5, 7, 4, 10)],          # similar_elements((3,4,5,6),(5,7,4,10))
    [(1, 2, 3, 4), (5, 4, 3, 7)],
    [(11, 12, 14, 13), (17, 15, 14, 13)],
  ],
  "plus_input": [[(), ()], [(1, 2, 3), ()], ...],   # 108 extra stress-test inputs
  "atol": 0,
}
```

Each entry in `base_input` / `plus_input` is one call's argument list; here every call
passes two tuples. We combine both lists into the full input set the implementations run on.

For each problem we record the consistency score, the correctness label, and the actual
code that was generated, so results can be inspected later.

## Running it

```bash
# generate, score, and save results for a handful of problems
python run_codecheck.py run --limit 10 --n 5 --timeout 5

# report how well the consistency score detected incorrect answers
python run_codecheck.py evaluate
```

Generation calls a hosted model, so it needs an API key in a local `.env` file. Evaluation
runs offline on the saved results.

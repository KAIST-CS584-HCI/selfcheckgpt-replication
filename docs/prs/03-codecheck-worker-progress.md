# Feat: Show Per-Problem Worker Progress In The Run Bar

## Context

Each problem takes ~17–65s while its concurrent workers run invisibly, so the progress bar looks frozen mid-problem. Add a live indicator of work happening within a problem.

## Changes

- Progress bar now shows a live per-problem postfix: `generate 18/21 → exec 14/20 → prompt 9/20`, so you can see which phase is running and how far along it is.
- Phases: **generate** (model writing the main + N samples), **exec** (running them), **prompt** (LLM judge). Labels match the method names in the result line.
- Cleared before each per-problem result line; the bar and result lines stay intact (no second bar, no flicker).

## How to run

```bash
python run_codecheck.py run --dataset humaneval --limit 2 --n 3 --method all --timeout 5
# watch the bar postfix flip generate -> exec -> prompt per problem
```

## Notes

- Progress is best-effort: a display hiccup never fails a problem.
- ast/code_bert scoring is instant, so only `generate` shows for those methods.

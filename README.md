# SelfCheckGPT Replication

[![Paper](https://img.shields.io/badge/arXiv-2303.08896-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2303.08896)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![SelfCheckGPT](https://img.shields.io/badge/SelfCheckGPT-replication-2ea44f?style=flat-square)](https://github.com/potsawee/selfcheckgpt)

A clean replication of **SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models** on the WikiBio benchmark.

SelfCheckGPT detects hallucinations by asking: *if the model really knows this fact, it should say the same thing every time it is asked.* We score a passage by comparing it against several other passages sampled from the same model — the more they disagree, the more likely the original is hallucinated.

![SelfCheckGPT prompt workflow](demo/selfcheck_qa_prompt.png)

## Contents

- [What's in this repo](#whats-in-this-repo)
- [Setup](#setup)
- [1. Score passages](#1-score-passages)
- [2. Aggregate](#2-aggregate)
- [3. Evaluate](#3-evaluate)
- [Dataset reference](#dataset-reference)
- [Troubleshooting](#troubleshooting)
- [Resources](#resources)

## What's in this repo

Three scoring methods, one CLI, and helpers to merge and evaluate the results:

| Method | What it does | Paired dataset |
| --- | --- | --- |
| `bert` | Compares passages with BERTScore similarity | `data/dataset-generated.json` |
| `nli` | Uses an NLI model to check entailment | `data/dataset-generated.json` |
| `prompt` | Asks an LLM (via OpenRouter) to judge consistency | `data/dataset-original.json` |

> The `selfcheckgpt/` folder is the original authors' code, kept as-is. Everything under `replication/`, `score.py`, `aggregate.py`, and `evaluate.py` is this replication.

## Setup

Requires Python 3.10+. A GPU helps for NLI; an OpenRouter key is needed for prompt scoring.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For prompt scoring, copy `.env.example` to `.env` and fill in your key:

```env
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=qwen/qwen3.5-9b
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

## 1. Score passages

Run one of the three methods over a slice of the dataset:

```bash
python -m score bert   --start 0 --end 10 --dataset data/dataset-generated.json
python -m score nli    --start 0 --end 10 --dataset data/dataset-generated.json
python -m score prompt --start 0 --end 10 --dataset data/dataset-original.json
```

Each run writes one file like `output/bert-0-to-10.json`.

**Range:** `--start` is inclusive, `--end` is exclusive (Python slicing). `--end` is optional and defaults to the dataset length. So `--start 0 --end 10` covers indices 0–9, and the next run should pick up at `--start 10`.

**Running in parallel:** scoring is slow, so it is normal to open several terminals and split the work into non-overlapping ranges. Each slice writes its own file. If a run crashes, whatever it had scored so far is saved before exiting — nothing is lost.

**Extra options:**

```bash
python -m score bert --start 0 --end 20 --dataset … --output output/my-run.json
python -m score prompt --start 0 --end 5 --dataset … --think    # higher reasoning effort
```

## 2. Aggregate

After running several slices, merge them into a single file:

```bash
python aggregate.py --dataset data/dataset-original.json
```

This produces `output/aggregated.json`, where each passage holds whichever of `bert`, `nli`, and `prompt` scores you have run so far. It also prints how many indices are still missing per method, so you know what to score next:

```text
Aggregated 119 passages -> output/aggregated.json
bert:   0/238 done, missing 0-237
nli:    119/238 done, missing 0-118
prompt: 0/238 done, missing 0-237
```

Re-running is safe: it picks up the existing `aggregated.json` and merges new slices into it.

## 3. Evaluate

Compute AUC-PR (sentence-level) and Pearson/Spearman correlations (passage-level) against the human annotations:

```bash
python -m evaluate --output output/aggregated.json --variant all
```

To evaluate a single slice instead:

```bash
python -m evaluate --output output/bert-0-to-10.json --variant bert
```

The script prints the metrics alongside the paper's baseline numbers for comparison.

## Dataset reference

Normalized WikiBio hallucination data lives in `data/`:

| File | Rows | What it is |
| --- | ---: | --- |
| `dataset-original.json` | 238 | Original SelfCheckGPT WikiBio dataset (GPT-3 main + GPT-3 samples). |
| `dataset-generated.json` | 119 | Subset with GPT-3.5 sampled passages for comparison. |
| `result-original-samples.json` | 238 | Pre-scored snapshot for the full original dataset (all three methods). |
| `result-generated-samples.json` | 119 | Pre-scored snapshot using the GPT-3.5 samples. |

Each dataset row:

| Field | Type | Description |
| --- | --- | --- |
| `wiki_bio_test_idx` | `int` | Identifier from the WikiBio test split. |
| `wiki_bio_text` | `str` | Reference Wikipedia biography. |
| `main_passage` | `str` | The passage being checked for hallucination. |
| `main_sentences` | `list[str]` | Sentence-split form of `main_passage`; scores align to this list. |
| `annotation` | `list[str]` | Human labels per sentence: `accurate`, `minor_inaccurate`, `major_inaccurate`. |
| `sample_passages` | `list[str]` | Other passages sampled from the same model, used as evidence. |

Scored result files keep the dataset fields and add:

- `dataset_idx` — row index
- `scores` — keyed by method (`bert`, `nli`, `prompt`)
- `responses` — raw LLM responses (prompt method only)

## Troubleshooting

> [!WARNING]
> `Missing OPENROUTER_API_KEY` — copy `.env.example` to `.env`, or export the key in your shell.

> [!NOTE]
> If spaCy can't find `en_core_web_sm`, re-run `pip install -r requirements.txt`.

> [!NOTE]
> `score.py` skips ranges whose output file already exists. Pick a new range or pass `--output` to keep a parallel run.

## Resources

- [SelfCheckGPT paper](https://arxiv.org/abs/2303.08896)
- [Original SelfCheckGPT project](https://github.com/potsawee/selfcheckgpt)
- [WikiBio GPT-3 hallucination dataset](https://huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination)

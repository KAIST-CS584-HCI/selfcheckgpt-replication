# SelfCheckGPT Replication

[![Paper](https://img.shields.io/badge/arXiv-2303.08896-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2303.08896)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![SelfCheckGPT](https://img.shields.io/badge/SelfCheckGPT-replication-2ea44f?style=flat-square)](https://github.com/potsawee/selfcheckgpt)

Replication workspace for **SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models**.

This repository keeps the original `selfcheckgpt` module from the paper authors and adds a cleaner replication workflow around it: normalized datasets, typed result entities, a unified scoring CLI, prompt-based OpenRouter configuration, and evaluation utilities.

![SelfCheckGPT prompt workflow](demo/selfcheck_qa_prompt.png)

## Contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
- [Run Scoring](#run-scoring)
- [Aggregate Scores](#aggregate-scores)
- [Evaluate Results](#evaluate-results)
- [Dataset](#dataset)
- [Troubleshooting](#troubleshooting)
- [Resources](#resources)

## Overview

SelfCheckGPT detects hallucinations by checking whether a generated passage is consistent with multiple sampled passages from the same model. This replication focuses on the WikiBio hallucination benchmark and currently supports:

| Method | Recommended dataset | Output default |
| --- | --- | --- |
| BERTScore | `data/dataset-generated.json` | `output/bert-{start}-to-{end}.json` |
| NLI | `data/dataset-generated.json` | `output/nli-{start}-to-{end}.json` |
| Prompt | `data/dataset-original.json` | `output/prompt-{start}-to-{end}.json` |

`--dataset` is required for every scoring run — there is no implicit default. The "recommended" column reflects which dataset each method is normally paired with in this replication.

> [!NOTE]
> The bundled `selfcheckgpt/` package code comes from the SelfCheckGPT paper authors. The replication-specific code in this repository lives mainly under `replication/` and `score.py`.

## Features

- Unified CLI for `bert`, `nli`, and `prompt` scoring, safe to run in parallel across non-overlapping ranges.
- Partial scoring results are flushed on interrupt/crash, so long runs are recoverable.
- `aggregate.py` merges per-slice outputs into a single JSON and reports which dataset indices are still missing per variant.
- Evaluation script for SelfCheckGPT-style AUC-PR and correlation metrics.

## Getting Started

### Prerequisites

- Python 3.10 or newer
- `pip`
- A GPU is optional, but recommended for NLI scoring
- OpenRouter API key for prompt scoring

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure Prompt Scoring

Copy the example environment file and add your OpenRouter key:

```bash
cp .env.example .env
```

```env
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=qwen/qwen3.5-9b
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

## Run Scoring

Use the root CLI module. `--dataset` is required; `--start` is required; `--end` is optional and defaults to `len(dataset)`.

```bash
python -m score bert   --start 0 --end 10 --dataset data/dataset-generated.json
python -m score nli    --start 0 --end 10 --dataset data/dataset-generated.json
python -m score prompt --start 0 --end 10 --dataset data/dataset-original.json
```

By default, results are written to the root `output/` directory:

```text
output/bert-0-to-10.json
output/nli-0-to-10.json
output/prompt-0-to-10.json
```

### Range semantics

`--start` is **inclusive** and `--end` is **exclusive** — the run covers `range(start, end)`, the same half-open `[start, end)` convention used by Python slicing. For example `--start 0 --end 10` scores 10 passages with `dataset_idx` 0..9; the next run should use `--start 10`.

Omitting `--end` scores from `--start` to the end of the dataset (i.e., `end = len(dataset)`), which is reflected in the default output filename:

```bash
# Scores indices 50..len(dataset)-1 → output/bert-50-to-<len>.json
python -m score bert --start 50 --dataset data/dataset-generated.json
```

Useful options:

```bash
python -m score bert \
  --start 0 \
  --end 20 \
  --dataset data/dataset-generated.json \
  --output output/custom-bert-range.json
```

Prompt scoring also supports a higher reasoning setting:

```bash
python -m score prompt --start 0 --end 5 --dataset data/dataset-original.json --think
```

### Parallel runs and partial saves

Scoring a long range is slow, so it is common to launch several terminals with non-overlapping ranges (e.g. `--start 0 --end 50`, `--start 50 --end 100`, …) in parallel. Each run writes its own `output/{method}-{start}-to-{end}.json` slice file. If a run is interrupted or crashes mid-loop, whatever has been scored so far is still flushed to that slice file before the exception propagates, so progress is not lost.

## Aggregate Scores

Once you have several slice files in `output/`, merge them into a single JSON that holds every variant's scores per passage:

```bash
python aggregate.py --dataset data/dataset-original.json
```

`aggregate.py`:

- Scans `output/` for `{bert,nli,prompt}-{start}-to-{end}.json` files (non-matching files are ignored).
- Merges every passage by `dataset_idx`; one entry can carry `scores.bert`, `scores.nli`, and `scores.prompt` together.
- **Reuses any existing `output/aggregated.json`** as the starting point, so re-running after dropping more slice files into `output/` is incremental — old entries are not recomputed or discarded.
- On conflict (same `(dataset_idx, variant)` in multiple slice files), the slice with the newest mtime wins; a `WARN:` line is printed if the scores differ.
- Prints a coverage report against `--dataset`, listing which indices are still missing per variant. Missing indices are compacted into ranges, so the output is easy to paste back into a follow-up `score.py` run:

```text
Aggregated 119 passages -> /…/output/aggregated.json
bert: 0/238 done, missing 0-237
nli: 119/238 done, missing 0-118
prompt: 0/238 done, missing 0-237
```

Options:

| Option | Default | Description |
| --- | --- | --- |
| `--dataset` | *(required)* | Original dataset JSON; defines the total passage count used in the coverage report. |
| `--inputs-dir` | `output/` | Directory to scan for slice files. |
| `--output` | `output/aggregated.json` | Destination for the merged JSON; reused as the starting point if it already exists. |

## Evaluate Results

Evaluate the aggregated file across all variants (recommended after running `aggregate.py`):

```bash
python -m evaluate --output output/aggregated.json --variant all
```

Or point the evaluator at a single slice file when you only want one variant:

```bash
python -m evaluate --output output/bert-0-to-10.json   --variant bert
python -m evaluate --output output/nli-0-to-10.json    --variant nli
python -m evaluate --output output/prompt-0-to-10.json --variant prompt
```

The evaluator reports sentence-level AUC-PR metrics and passage-level Pearson/Spearman correlations, alongside the paper baselines for comparison.

## Dataset

The `data/` directory contains normalized WikiBio hallucination datasets and scored result snapshots used by this replication.

| File | Rows | Description | Generation / scoring source |
| --- | ---: | --- | --- |
| `dataset-original.json` | 238 | Normalized original SelfCheckGPT WikiBio hallucination dataset. | GPT-3 generated the main passages and original sampled passages. |
| `dataset-generated.json` | 119 | Subset with regenerated sampled passages for comparison experiments. | GPT-3 generated the main passages; GPT-3.5 generated the sampled passages. |
| `result-original-samples.json` | 119 | Scored result snapshot using original sampled passages. | Prompt scores use Qwen 3.5 9B; BERTScore and NLI use the provided SelfCheckGPT implementations. |
| `result-generated-samples.json` | 119 | Scored result snapshot using GPT-3.5 sampled passages. | Prompt scores use Qwen 3.5 9B; BERTScore and NLI use the provided SelfCheckGPT implementations. |

Dataset rows use a shared normalized schema:

| Field | Type | Description |
| --- | --- | --- |
| `wiki_bio_test_idx` | `int` | Identifier from the WikiBio test split. |
| `wiki_bio_text` | `str` | Reference Wikipedia biography passage. |
| `main_passage` | `str` | Main generated passage to evaluate for hallucination. |
| `main_sentences` | `list[str]` | Sentence-split version of `main_passage`; scores align to this list. |
| `annotation` | `list[str]` | Human sentence-level labels: `accurate`, `minor_inaccurate`, or `major_inaccurate`. |
| `sample_passages` | `list[str]` | Sampled passages used as consistency evidence by SelfCheckGPT. |

Example dataset row:

```json
{
  "wiki_bio_test_idx": 0,
  "wiki_bio_text": "...",
  "main_passage": "...",
  "main_sentences": ["..."],
  "annotation": ["accurate"],
  "sample_passages": ["..."]
}
```

Result files preserve the dataset fields and add:

| Field | Type | Description |
| --- | --- | --- |
| `dataset_idx` | `int` | Row index from the scored dataset. |
| `scores` | `object` | Method-specific sentence scores, keyed by `prompt`, `bert`, or `nli`. |
| `responses` | `object` | Raw model responses for prompt scoring; empty for score-only methods. |

```json
{
  "scores": {
    "bert": [0.12, 0.34]
  },
  "responses": {}
}
```

Prompt outputs additionally include raw prompt responses under `responses.prompt`.

## Troubleshooting

> [!WARNING]
> If prompt scoring fails with `Missing OPENROUTER_API_KEY`, create `.env` from `.env.example` or export `OPENROUTER_API_KEY` in your shell.

> [!NOTE]
> If spaCy cannot find `en_core_web_sm`, reinstall dependencies with `pip install -r requirements.txt`. The model is declared as a pip dependency.

> [!NOTE]
> Existing output files are skipped by default. Choose a new range or pass a different `--output` path when you want to keep another scoring run.

## Resources

- [SelfCheckGPT paper](https://arxiv.org/abs/2303.08896)
- [Original SelfCheckGPT project](https://github.com/potsawee/selfcheckgpt)
- [WikiBio GPT-3 hallucination dataset](https://huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination)

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
- [Evaluate Results](#evaluate-results)
- [Dataset](#dataset)
- [Troubleshooting](#troubleshooting)
- [Resources](#resources)

## Overview

SelfCheckGPT detects hallucinations by checking whether a generated passage is consistent with multiple sampled passages from the same model. This replication focuses on the WikiBio hallucination benchmark and currently supports:

| Method | Dataset default | Output default |
| --- | --- | --- |
| BERTScore | `data/dataset-generated.json` | `output/bert.json` |
| NLI | `data/dataset-generated.json` | `output/nli.json` |
| Prompt | `data/dataset-original.json` | `output/prompt.json` |

> [!NOTE]
> The bundled `selfcheckgpt/` package code comes from the SelfCheckGPT paper authors. The replication-specific code in this repository lives mainly under `replication/` and `score.py`.

## Features

- Unified CLI for `bert`, `nli`, and `prompt` scoring.
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

Use the root CLI module:

```bash
python -m score bert --start 0 --end 10
python -m score nli --start 0 --end 10
python -m score prompt --start 0 --end 10
```

By default, results are written to the root `output/` directory:

```text
output/bert.json
output/nli.json
output/prompt.json
```

Useful options:

```bash
python -m score bert \
  --start 0 \
  --end 20 \
  --dataset data/dataset-generated.json \
  --output output/bert.json \
  --overwrite
```

Prompt scoring also supports a higher reasoning setting:

```bash
python -m score prompt --start 0 --end 5 --think
```

## Evaluate Results

Evaluate one score file:

```bash
python -m replication.evaluate --results output/bert.json --variant bert
python -m replication.evaluate --results output/nli.json --variant nli
python -m replication.evaluate --results output/prompt.json --variant prompt
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
> Existing output files are skipped by default. Pass `--overwrite` when you intentionally want to replace an output JSON file.

## Resources

- [SelfCheckGPT paper](https://arxiv.org/abs/2303.08896)
- [Original SelfCheckGPT project](https://github.com/potsawee/selfcheckgpt)
- [WikiBio GPT-3 hallucination dataset](https://huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination)

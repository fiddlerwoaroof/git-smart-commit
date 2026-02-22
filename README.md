# git-smart-commit

Analyzes your working tree changes, groups them into logical commits using an
LLM, and optionally executes them.

Works with any OpenAI-compatible API: Ollama (default), OpenRouter, Together,
Anthropic, self-hosted SGLang/vLLM, etc.

## Requirements

- Python 3.11+ with [`uv`](https://github.com/astral-sh/uv)
- [Ollama](https://ollama.ai) running locally (or any OpenAI-compatible API)
- The model: `ollama pull qwen3-coder:30b-a3b-q8_0`

## Installation

```bash
chmod +x git-smart-commit
cp git-smart-commit ~/.local/bin/
```

## Usage

```bash
# Interactive: review proposed commits before executing
git-smart-commit

# Non-interactive: commit immediately without prompting
git-smart-commit --yes

# Dry run: show proposed commits without executing
git-smart-commit --dry-run

# JSON output (for scripts or skill integrations)
git-smart-commit --json

# Different repo
git-smart-commit --repo ~/projects/myapp

# Different model
git-smart-commit --model qwen3-coder:8b
```

### Using a remote API

```bash
# OpenRouter
git-smart-commit \
  --api-base https://openrouter.ai/api/v1 \
  --api-key $OPENROUTER_API_KEY \
  --model deepseek/deepseek-r1

# Self-hosted (SGLang / vLLM on Vast.ai)
git-smart-commit \
  --api-base http://my-vastai-instance:30000/v1 \
  --api-key $MY_KEY \
  --model Qwen/Qwen3-30B-A3B

# API key from environment variable
export LLM_API_KEY=sk-...
git-smart-commit --api-base https://openrouter.ai/api/v1 --model ...
```

## Exit codes

| Code | Meaning            |
|------|--------------------|
| 0    | Success            |
| 1    | No changes found   |
| 2    | Model/API error    |
| 3    | Git error          |
| 4    | User cancelled     |

## How it works

1. Runs `git diff` (staged + unstaged) and `git ls-files --others` to collect all changes
2. Builds per-file diffs; large files (>6k chars) are summarized hunk-by-hunk
3. Sends the diff summary to the model with a structured classification prompt
4. Model groups files into logical commits with conventional commit messages and
   flags code issues (wrong API usage, suspicious patterns, etc.)
5. Overlapping commits (files shared across groups) are merged by a second model call
6. Proposes the result; optionally stages and commits each group in order

---

## Finetuning pipeline

The repo also contains scripts for finetuning a smaller model to match a
frontier teacher's output on this task, at much lower inference cost.

### Pipeline overview

```
harvest-training-data  →  label-training-data  →  prepare-sft-data  →  train.py
       ↓                                                                     ↓
 repo-dataset.jsonl       labeled-training-data.jsonl    train.jsonl / val.jsonl
                                                                             ↓
                                                                        eval.py
                                                                             ↓
                                                               merge-and-quantize.sh
```

### `harvest-training-data`

Walks local git repos, picks random commit ranges, and captures diffs as
training inputs. All read-only — never modifies your working tree.

```bash
harvest-training-data \
  --repo-dirs ~/git-repos \
  --output repo-dataset.jsonl \
  --count 10000 \
  --max-per-repo 20 \
  --all-branches
```

Output: `repo-dataset.jsonl` — one JSON object per line with `diff`,
`changed_files`, `original_commits`, `languages`, `existing_gitignore`, etc.

### `label-training-data`

Calls a teacher model on each harvested example to produce high-quality commit
grouping labels. Idempotent: skips examples already labeled in the output file.

```bash
label-training-data \
  --model step-3.5-flash \
  --api-url http://my-vastai:30000/v1 \
  --api-key $MY_KEY \
  --input repo-dataset.jsonl \
  --output labeled-training-data.jsonl \
  --concurrency 16 \
  --noise-prob 0.3        # inject .DS_Store / *.pyc into 30% of examples
```

For reasoning models (e.g. Step 3.5 Flash, o3-mini):

```bash
label-training-data ... --reasoning-effort medium --max-tokens 8192
```

### `prepare-sft-data`

Converts labeled examples to ChatML format and generates a stratified
train/val split. Also generates augmented noisy variants (junk file injection)
without re-calling the teacher.

```bash
prepare-sft-data \
  --input labeled-training-data.jsonl \
  --train train.jsonl \
  --val val.jsonl \
  --n-augment 1 \
  --val-fraction 0.1
```

### `train.py`

QLoRA finetuning via [Unsloth](https://github.com/unslothai/unsloth). Run on
a Vast.ai A100/H100.

```bash
pip install unsloth trl transformers accelerate bitsandbytes

python train.py \
  --model unsloth/Qwen3-30B-A3B-unsloth-bnb-4bit \
  --dataset train.jsonl \
  --val-dataset val.jsonl \
  --output lora-adapter \
  --lora-rank 32 \
  --epochs 3 \
  --lr 1.5e-4
```

Saves a LoRA adapter + `training_config.json` to `lora-adapter/`.

### `eval.py`

Evaluates any model on the validation set via OpenAI-compatible API.

```bash
# Baseline (untuned model)
eval.py --model qwen3-coder:30b-a3b-q8_0 --val val.jsonl

# Finetuned model
eval.py --model git-smart-commit:latest --val val.jsonl --output results.jsonl
```

Metrics: **file grouping F1**, **commit type accuracy**, **junk routing rate**
(fraction of injected `.DS_Store`/`*.pyc` files correctly sent to `.gitignore`).

### `merge-and-quantize.sh`

Merges the LoRA adapter, converts to GGUF, quantizes, and registers the result
with Ollama.

```bash
./merge-and-quantize.sh \
  --adapter lora-adapter \
  --base unsloth/Qwen3-30B-A3B-unsloth-bnb-4bit \
  --output merged-model \
  --quant Q8_0 \
  --ollama-name git-smart-commit
```

After this, `git-smart-commit --model git-smart-commit` uses the finetuned model.

# git-smart-commit Finetuning Plan

## Current Status

| Milestone | Status |
|-----------|--------|
| `git-smart-commit` tool with OpenAI-compatible API support | ✅ Done |
| `harvest-training-data` script | ✅ Done |
| Raw dataset: 10K examples in `repo-dataset.jsonl` (~190MB) | ✅ Done |
| Step 3.5 Flash running on Vast.ai 2× H200 via SGLang | ✅ Running |
| Model evaluation (MiniMax M2.5, Step 3.5 Flash) | 🔄 In progress |
| Labeling script | ❌ Not started |
| Finetuning | ❌ Not started |

---

## Phase 1: Label the Dataset

**Goal:** Run a frontier-class teacher model on each harvested example to produce
high-quality commit groupings and messages. Output is a labeled JSONL suitable for
SFT training.

### 1a. Write `label-training-data` script

The script reads `repo-dataset.jsonl`, constructs the same prompt that
`git-smart-commit` uses (system prompt + file list + diff summary), calls the
teacher model via OpenAI-compatible API, and saves structured output.

**Input per example** (from harvested JSONL):
- `diff` — the full unified diff
- `changed_files` — list of filenames
- `existing_gitignore` — patterns already in the repo
- `languages` — detected languages

**Output per example** (appended to labeled JSONL):
- Everything from input, plus:
- `label.commits[]` — each with `type`, `scope`, `subject`, `body`, `files`, `issues`
- `label.gitignore` — suggested patterns
- `label.model` — which model produced the label
- `label.usage` — token counts for cost tracking

**Key design decisions:**
- Reuse the exact `SYSTEM_PROMPT` from `git-smart-commit` so the student learns
  the same conventions
- Add noise injection at this stage: randomly inject junk files (`.DS_Store`,
  `*.pyc`, swap files) into the file list and diff to teach the model to route
  them to `.gitignore` instead of commits
- Use plain chat completion (not tool calling) — request JSON output directly,
  since tool-call formats vary across providers and we want provider-agnostic
  labeling
- Parallelize with asyncio + semaphore (e.g. 8-16 concurrent requests)
- Save progress incrementally (append mode) so crashes don't lose work
- Skip examples that already have labels (idempotent reruns)

### 1b. Choose teacher model

Current candidates, ordered by preference:

| Model | Via | Est. cost (10K) | Quality | Notes |
|-------|-----|------------------|---------|-------|
| Step 3.5 Flash | Self-hosted (Vast.ai) | ~$8-16 (GPU rental) | Frontier | Already running, 197B MoE/11B active |
| DeepSeek V3.2 | OpenRouter API | ~$16 | Frontier | No distillation restrictions |
| Qwen 3.5 Plus | Alibaba direct API | ~$9 | Frontier | Cheapest API option |
| MiniMax M2.5 | OpenRouter API | ~$22 | Near-frontier | Tested, works with tool calling |
| GLM-5 | OpenRouter API | ~$29 | Frontier | Open-source |

**Recommendation:** Use Step 3.5 Flash on the current Vast.ai instance since it's
already running. If quality is insufficient on spot-checks, fall back to DeepSeek
V3.2 or Qwen 3.5 Plus via API.

### 1c. Quality gate

Before labeling all 10K:
1. Label 50 examples
2. Manually review ~20 for:
   - Accurate file grouping (files that belong together are grouped)
   - Correct commit types (`feat` vs `fix` vs `refactor`)
   - Specific, non-hallucinated subject lines
   - Body describes what changed, not phantom deltas
   - New file additions described as "what the file does" not "what changed"
   - Junk files routed to `.gitignore`
3. Compare against original commit messages from `original_commits` field
4. If quality is poor, try a different teacher model before scaling up

---

## Phase 2: Prepare Training Data

**Goal:** Convert labeled JSONL into the ChatML SFT format the student model expects.

### 2a. Training format

Each example becomes a single-turn ChatML conversation:

```json
{
  "messages": [
    {"role": "system", "content": "<SYSTEM_PROMPT>"},
    {"role": "user", "content": "<constructed prompt with file list + diff summary>"},
    {"role": "assistant", "content": "<JSON tool call from label>"}
  ]
}
```

The assistant turn should match the tool-call format the student model will be
expected to produce at inference time. For Ollama models this is typically a
JSON blob matching the `ProposeCommitsArgs` schema.

### 2b. Data augmentation / noise

For each example, generate 1-2 augmented variants:
- Inject random junk files into the file list (`.DS_Store`, `__pycache__/`,
  `*.swp`, `.env`, `node_modules/`)
- Add corresponding diff hunks for the junk files
- Expected label: junk files appear in `gitignore`, not in any commit

This teaches the model to distinguish real changes from noise, which is the main
failure mode with un-finetuned models.

### 2c. Train/validation split

- 90% train (~9K examples + augmented variants)
- 10% validation (~1K examples)
- Stratify by language and repo to avoid leakage

---

## Phase 3: Finetune

**Goal:** LoRA finetune a small code-specialized model to match the teacher's
output quality on this specific task.

### 3a. Choose student model

| Model | Params (active) | Why | LoRA VRAM |
|-------|-----------------|-----|-----------|
| Qwen3-Coder 30B-A3B | 30B (3B active) | Already using it in production, MoE = fast inference | ~12GB (QLoRA 4-bit) |
| Qwen3-Coder 8B | 8B | Smaller, faster training, good baseline | ~10GB |
| Qwen 2.5 Coder 7B | 7B | Proven code model, well-supported by training frameworks | ~8GB |

**Recommendation:** Start with Qwen3-Coder 30B-A3B (QLoRA) since it's the
production inference model. If training is too slow or unstable, fall back to
the 8B variant.

### 3b. Training setup

**Framework:** Unsloth (fastest LoRA training, good Qwen3 support) or
LLaMA-Factory (more configuration options, supports MoE QLoRA).

**Hardware:** Vast.ai single A100 80GB or H100 80GB.

**Hyperparameters (starting point):**
- LoRA rank: 32-64
- LoRA alpha: 64-128
- Target modules: all linear layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- Learning rate: 1e-4 to 2e-4
- Batch size: 4 (gradient accumulation to effective batch 16-32)
- Epochs: 2-3
- Warmup: 10% of steps
- Max sequence length: 8192 (covers most diffs; truncate outliers)
- Optimizer: AdamW 8-bit (paged)

**Estimated cost:**
- A100 80GB on Vast.ai: ~$0.80/hr × 2-4 hrs = **$1.60-3.20**
- H100 80GB on Vast.ai: ~$1.75/hr × 1-2 hrs = **$1.75-3.50**

### 3c. Training script

```bash
# On Vast.ai A100
pip install unsloth

python train.py \
  --model Qwen/Qwen3-Coder-30B-A3B \
  --dataset labeled-training-data.jsonl \
  --output ./lora-adapter \
  --lora-rank 32 \
  --epochs 3 \
  --batch-size 4 \
  --lr 1.5e-4 \
  --max-seq-length 8192
```

Output: LoRA adapter weights (~50-200MB), ready to merge or load alongside the
base model.

---

## Phase 4: Evaluate

**Goal:** Verify the finetuned model matches or exceeds the base model + long
prompt on commit message quality.

### 4a. Automated metrics

Run the finetuned model on the 1K validation set and measure:
- **File grouping accuracy** — do the proposed commits group the same files as
  the teacher labels?
- **Type accuracy** — `feat`/`fix`/`refactor`/etc. matches
- **Hallucination rate** — does the body mention things not in the diff?
- **Junk file routing** — are `.DS_Store` etc. in `.gitignore`, not commits?

### 4b. Side-by-side comparison

For 20-30 validation examples, compare:
1. Base model (Qwen3-Coder 30B + full system prompt) — current behavior
2. Finetuned model (Qwen3-Coder 30B + LoRA, minimal prompt)
3. Teacher model labels (ground truth)

Score each on: accuracy, specificity, brevity, type correctness.

### 4c. Production dogfooding

Replace the Ollama model in `git-smart-commit` with the finetuned adapter and
use it for real commits on your own repos for a week. Track pain points.

---

## Phase 5: Ship

### 5a. Merge & quantize

```bash
# Merge LoRA into base model
python merge_lora.py \
  --base Qwen/Qwen3-Coder-30B-A3B \
  --adapter ./lora-adapter \
  --output ./merged-model

# Quantize to GGUF for Ollama
llama-quantize ./merged-model ./git-smart-commit-30b-q8.gguf Q8_0
```

### 5b. Create Ollama model

```
# Modelfile
FROM ./git-smart-commit-30b-q8.gguf
PARAMETER temperature 0.2
PARAMETER num_ctx 128000
SYSTEM """<minimal system prompt — most instructions now baked into weights>"""
```

```bash
ollama create git-smart-commit -f Modelfile
```

### 5c. Update `git-smart-commit` defaults

Change `DEFAULT_MODEL` to `git-smart-commit:latest` and simplify the system
prompt (the finetuned model shouldn't need the extensive anti-hallucination
rules since those behaviors are now in the weights).

### 5d. Optional: publish

- Push GGUF to Hugging Face
- Push LoRA adapter to Hugging Face
- Consider publishing the labeled dataset (after filtering personal repos:
  intellij-settings, dotfiles, nixos-servers, gitolite-admin)

---

## Cost Summary

| Step | Platform | Estimated Cost |
|------|----------|---------------|
| Labeling (10K examples) | Vast.ai self-hosted or API | $8-20 |
| Finetuning (LoRA, 2-4 hrs) | Vast.ai A100/H100 | $2-4 |
| Evaluation (inference on 1K) | Local M4 Max | $0 |
| **Total** | | **$10-24** |

---

## Scripts To Write

1. **`label-training-data`** — reads harvested JSONL, calls teacher model API,
   saves labeled output with commit groupings and messages
2. **`prepare-sft-data`** — converts labeled JSONL to ChatML training format,
   injects noise augmentation, does train/val split
3. **`train.py`** — Unsloth/LLaMA-Factory wrapper for LoRA training with the
   right hyperparameters
4. **`eval.py`** — runs finetuned model on validation set, computes metrics,
   generates side-by-side comparison report
5. **`merge-and-quantize.sh`** — merges LoRA, quantizes to GGUF, creates
   Ollama Modelfile

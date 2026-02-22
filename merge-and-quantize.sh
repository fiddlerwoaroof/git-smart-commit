#!/usr/bin/env bash
# merge-and-quantize.sh: Merge LoRA adapter into base model, quantize to GGUF,
# and create an Ollama model ready for use with git-smart-commit.
#
# Prerequisites (install on the training machine):
#   pip install transformers peft accelerate
#   # Build llama.cpp (for quantize):
#   git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make
#
# Usage:
#   ./merge-and-quantize.sh [--adapter DIR] [--base MODEL] [--output DIR]
#                           [--quant Q8_0|Q4_K_M] [--ollama-name NAME]
#                           [--llamacpp PATH] [--skip-merge] [--skip-quantize]
#
# Steps:
#   1. Merge LoRA adapter into base model weights (merge_lora.py)
#   2. Convert merged model to GGUF format (llama.cpp convert_hf_to_gguf.py)
#   3. Quantize GGUF to target format (llama-quantize)
#   4. Write Modelfile and register with Ollama
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
ADAPTER_DIR="lora-adapter"
BASE_MODEL="unsloth/Qwen3-30B-A3B-unsloth-bnb-4bit"
OUTPUT_DIR="merged-model"
QUANT="Q8_0"
OLLAMA_NAME="git-smart-commit"
LLAMACPP_DIR="${HOME}/llama.cpp"
SKIP_MERGE=false
SKIP_QUANTIZE=false

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --adapter)     ADAPTER_DIR="$2";   shift 2 ;;
        --base)        BASE_MODEL="$2";    shift 2 ;;
        --output)      OUTPUT_DIR="$2";    shift 2 ;;
        --quant)       QUANT="$2";         shift 2 ;;
        --ollama-name) OLLAMA_NAME="$2";   shift 2 ;;
        --llamacpp)    LLAMACPP_DIR="$2";  shift 2 ;;
        --skip-merge)     SKIP_MERGE=true;    shift ;;
        --skip-quantize)  SKIP_QUANTIZE=true; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

GGUF_F16="${OUTPUT_DIR}/model-f16.gguf"
GGUF_QUANT="${OUTPUT_DIR}/model-${QUANT}.gguf"
MODELFILE="${OUTPUT_DIR}/Modelfile"

echo "=== git-smart-commit merge & quantize ==="
echo "  Adapter:     ${ADAPTER_DIR}"
echo "  Base model:  ${BASE_MODEL}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  Quant:       ${QUANT}"
echo "  Ollama name: ${OLLAMA_NAME}"
echo ""

mkdir -p "${OUTPUT_DIR}"

# ── Step 1: Merge LoRA ────────────────────────────────────────────────────────
if [[ "${SKIP_MERGE}" == "false" ]]; then
    echo "[1/4] Merging LoRA adapter into base model…"
    python3 - <<PYEOF
import sys
from pathlib import Path

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch
except ImportError as e:
    print(f"Error: missing dependency: {e}")
    print("Install with: pip install transformers peft accelerate")
    sys.exit(1)

adapter_dir = "${ADAPTER_DIR}"
base_model  = "${BASE_MODEL}"
output_dir  = "${OUTPUT_DIR}"

print(f"  Loading base model: {base_model}", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(adapter_dir or base_model, trust_remote_code=True)

print(f"  Loading LoRA adapter: {adapter_dir}", flush=True)
model = PeftModel.from_pretrained(model, adapter_dir)

print("  Merging weights…", flush=True)
model = model.merge_and_unload()

print(f"  Saving merged model to: {output_dir}", flush=True)
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)
print("  Merge complete.", flush=True)
PYEOF
    echo "  Merge complete."
else
    echo "[1/4] Skipping merge (--skip-merge)."
fi

# ── Step 2: Convert to GGUF (fp16) ───────────────────────────────────────────
if [[ "${SKIP_QUANTIZE}" == "false" ]]; then
    CONVERT_SCRIPT="${LLAMACPP_DIR}/convert_hf_to_gguf.py"
    if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
        echo "Error: llama.cpp convert script not found at ${CONVERT_SCRIPT}" >&2
        echo "Clone and build llama.cpp:" >&2
        echo "  git clone https://github.com/ggerganov/llama.cpp ${LLAMACPP_DIR}" >&2
        echo "  cd ${LLAMACPP_DIR} && make" >&2
        exit 1
    fi

    echo "[2/4] Converting merged model to GGUF (fp16)…"
    python3 "${CONVERT_SCRIPT}" \
        "${OUTPUT_DIR}" \
        --outtype f16 \
        --outfile "${GGUF_F16}"
    echo "  GGUF (fp16) written to: ${GGUF_F16}"

    # ── Step 3: Quantize ─────────────────────────────────────────────────────
    QUANTIZE_BIN="${LLAMACPP_DIR}/llama-quantize"
    if [[ ! -f "${QUANTIZE_BIN}" ]]; then
        QUANTIZE_BIN="${LLAMACPP_DIR}/quantize"  # older llama.cpp name
    fi
    if [[ ! -f "${QUANTIZE_BIN}" ]]; then
        echo "Error: llama-quantize binary not found. Did you run 'make' in ${LLAMACPP_DIR}?" >&2
        exit 1
    fi

    echo "[3/4] Quantizing to ${QUANT}…"
    "${QUANTIZE_BIN}" "${GGUF_F16}" "${GGUF_QUANT}" "${QUANT}"
    echo "  Quantized GGUF: ${GGUF_QUANT}"

    # Clean up fp16 intermediate to save disk space
    rm -f "${GGUF_F16}"
    echo "  Removed intermediate fp16 GGUF."
else
    echo "[2/4] Skipping GGUF conversion (--skip-quantize)."
    echo "[3/4] Skipping quantization (--skip-quantize)."
    GGUF_QUANT="$(ls "${OUTPUT_DIR}"/*.gguf 2>/dev/null | head -1)"
    if [[ -z "${GGUF_QUANT}" ]]; then
        echo "Error: no GGUF found in ${OUTPUT_DIR}" >&2
        exit 1
    fi
    echo "  Using existing GGUF: ${GGUF_QUANT}"
fi

# ── Step 4: Create Ollama model ───────────────────────────────────────────────
echo "[4/4] Writing Modelfile and registering with Ollama…"

# Minimal system prompt — most instruction-following is now baked into weights.
# The finetuned model knows the output format and commit conventions.
cat > "${MODELFILE}" <<'MODELEOF'
FROM ./model-QUANT_PLACEHOLDER.gguf

PARAMETER temperature 0.2
PARAMETER num_ctx 131072

SYSTEM """You are a senior software engineer. Group the provided git changes into logical commits.

Output a JSON object:
{"commits": [{"subject": "type(scope): desc", "files": [...], "body": "...", "issues": [...]}], "gitignore": [...]}"""
MODELEOF

# Substitute the actual quantization type into the Modelfile
sed -i "s/QUANT_PLACEHOLDER/${QUANT}/g" "${MODELFILE}"

# Register with Ollama from the output directory
(cd "${OUTPUT_DIR}" && ollama create "${OLLAMA_NAME}" -f Modelfile)

echo ""
echo "=== Done ==="
echo "  Model registered as: ${OLLAMA_NAME}"
echo "  Test with:"
echo "    git-smart-commit --model ${OLLAMA_NAME} --dry-run"
echo ""
echo "  To update git-smart-commit default, set DEFAULT_MODEL = '${OLLAMA_NAME}'"
echo "  in git-smart-commit.py."

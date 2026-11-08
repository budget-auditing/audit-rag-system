#!/usr/bin/env bash
#
# Run the main Audit-OCIQA pipeline (retrieval + evidence chain + generation
# + evaluation) for every supported backbone and method. Defaults to the
# synthetic demo data; set DATA_DIR to point at another dataset with the
# same layout.
#
# Usage:
#   scripts/run_main_experiments.sh [--backbones qwen,deepseek,llama]
#                                   [--methods closed-book,bm25-rag,...]
#                                   [--split test|test-mini|all]
#                                   [--index-dir DIR]
#
# Extra arguments are forwarded to experiments.run_main.

set -euo pipefail

BACKBONES="qwen,deepseek,llama"
METHODS="closed-book,bm25-rag,naive-dense-rag,hyde-rag,cot-rag,trace"
SPLIT="test"
INDEX_DIR="runs/index"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backbones) BACKBONES="$2"; shift 2 ;;
        --methods)   METHODS="$2";   shift 2 ;;
        --split)     SPLIT="$2";     shift 2 ;;
        --index-dir) INDEX_DIR="$2"; shift 2 ;;
        *)           EXTRA_ARGS+=("$1"); shift ;;
    esac
done

DATA_DIR="${DATA_DIR:-datasets-public/audit-ociqa-demo}"
QRELS="$DATA_DIR/gold.qrels.jsonl"
GOLD="$DATA_DIR/qa.gold.jsonl"
SPLITS_DIR="$DATA_DIR/splits"

missing=()
for path in "$INDEX_DIR" "$QRELS" "$GOLD" "$SPLITS_DIR/${SPLIT}.jsonl"; do
    if [[ ! -e "$path" ]]; then
        missing+=("$path")
    fi
done
if (( ${#missing[@]} > 0 )); then
    echo "[run_main_experiments] ERROR: missing required inputs:" >&2
    for p in "${missing[@]}"; do echo "  - $p" >&2; done
    echo "" >&2
    echo "Generate the synthetic demo with: python -m tools.make_demo_dataset" >&2
    echo "and run scripts/build_index.sh first to generate the FAISS index." >&2
    exit 1
fi

python -m experiments.run_main \
    --backbones "$BACKBONES" \
    --methods   "$METHODS" \
    --split     "$SPLIT" \
    --index-dir "$INDEX_DIR" \
    --splits-dir "$SPLITS_DIR" \
    --qrels     "$QRELS" \
    --gold      "$GOLD" \
    --stage     all \
    "${EXTRA_ARGS[@]}"

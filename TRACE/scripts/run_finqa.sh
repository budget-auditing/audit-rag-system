#!/usr/bin/env bash
#
# Reproduce the FinQA generalisation experiment.
#
# Usage:
#   scripts/run_finqa.sh [--methods bm25-rag,naive-dense-rag,hyde-rag,cot-rag,trace]
#                        [--model Qwen/Qwen3-Next-80B-A3B-Instruct]
#                        [--split test|dev]
#                        [--data-dir datasets/finqa]
#                        [--index-dir runs/finqa/index]
#                        [--outdir runs/finqa]
#
# Extra arguments are forwarded to experiments.run_finqa.

set -euo pipefail

METHODS="bm25-rag,naive-dense-rag,hyde-rag,cot-rag,trace"
MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct"
SPLIT="test"
DATA_DIR="datasets/finqa"
INDEX_DIR="runs/finqa/index"
FINQA_DIR="FinQA/dataset"
OUTDIR="runs/finqa"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --methods)   METHODS="$2";   shift 2 ;;
        --model)     MODEL="$2";     shift 2 ;;
        --split)     SPLIT="$2";     shift 2 ;;
        --data-dir)  DATA_DIR="$2";  shift 2 ;;
        --index-dir) INDEX_DIR="$2"; shift 2 ;;
        --outdir)    OUTDIR="$2";    shift 2 ;;
        *)           EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ ! -d "$FINQA_DIR" ]]; then
    cat >&2 <<EOF
[run_finqa] ERROR: FinQA source files not found at $FINQA_DIR

Clone https://github.com/czyssrs/FinQA and make the train/dev/test JSON files
available under $FINQA_DIR. Then re-run this script; the conversion step will
be executed automatically when datasets/finqa/corpus.paragraphs.jsonl is
missing.
EOF
    exit 1
fi

python -m experiments.run_finqa \
    --methods   "$METHODS" \
    --model     "$MODEL" \
    --split     "$SPLIT" \
    --data-dir  "$DATA_DIR" \
    --index-dir "$INDEX_DIR" \
    --finqa-dir "$FINQA_DIR" \
    --outdir    "$OUTDIR" \
    --stage     all \
    "${EXTRA_ARGS[@]}"

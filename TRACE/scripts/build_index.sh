#!/usr/bin/env bash
#
# Build the FAISS index used by the TRACE retrieval stage.
#
# Usage:
#   scripts/build_index.sh [--dataset audit-ociqa|finqa] [--output-dir DIR]
#
# Requires an OpenAI-compatible embedding endpoint available at
# $EMBED_SERVER_URL (default: http://localhost:8081/v1/embeddings).
# A local BGE-M3 server can be launched via ``python -m tools.embed_server``
# after ``pip install -e '.[embed-server]'``.

set -euo pipefail

DATASET="audit-ociqa"
OUTPUT_DIR=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)       DATASET="$2";    shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2"; shift 2 ;;
        *)               EXTRA_ARGS+=("$1"); shift ;;
    esac
done

case "$DATASET" in
    audit-ociqa)
        CORPUS_PATH="datasets-public/audit-ociqa-demo/corpus.paragraphs.jsonl"
        DEFAULT_OUT="runs/index"
        ;;
    finqa)
        CORPUS_PATH="datasets/finqa/corpus.paragraphs.jsonl"
        DEFAULT_OUT="runs/finqa/index"
        ;;
    *)
        echo "[build_index] ERROR: unknown dataset '$DATASET'." >&2
        exit 1
        ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUT}"

if [[ ! -f "$CORPUS_PATH" ]]; then
    cat >&2 <<EOF
[build_index] ERROR: corpus file not found at $CORPUS_PATH

For the synthetic Audit-OCIQA demo run: python -m tools.make_demo_dataset
For FinQA follow docs/datasets.md, then re-run this script.
EOF
    exit 1
fi

echo "[build_index] dataset=$DATASET corpus=$CORPUS_PATH output=$OUTPUT_DIR"
python -m tools.build_index \
    --corpus-file "$CORPUS_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --add-meta-tokens \
    --clean-noise \
    --inject-dept \
    "${EXTRA_ARGS[@]}"

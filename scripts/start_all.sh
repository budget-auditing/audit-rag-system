#!/usr/bin/env bash
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)

mkdir -p "$ROOT/logs"

# Each service wants a different URL shape, so these are injected per-service
# instead of exported globally:
#   backend  — service roots, read from backend/.env (do not override here)
#   C³-RAG   — OpenAI client base_url, so "/v1" root; service.py ignores C3RAG/.env
#   TRACE    — raw POST targets, so fully-qualified endpoints
# Single source of truth for the chat model is backend/.env.
CHAT_MODEL_ID=$(sed -n "s/^CHAT_MODEL=//p" "$ROOT/backend/.env" | head -1)

C3RAG_ENV=(
    EMBED_URL="http://localhost:8081/v1"
    CHAT_URL="http://localhost:8000/v1"
    CHAT_MODEL="$CHAT_MODEL_ID"
    OPENAI_API_KEY="sk-placeholder"
)

TRACE_ENV=(
    EMBED_URL="http://localhost:8081/v1/embeddings"
    CHAT_URL="http://localhost:8000/v1/chat/completions"
    CHAT_MODEL="$CHAT_MODEL_ID"
    OPENAI_API_KEY="sk-placeholder"
)

# Embeddings — skip if already healthy
if curl -sf -o /dev/null --max-time 2 http://localhost:8081/health; then
    echo "Embed server already running on :8081, skipping."
    EMBED_PID=""
else
    echo "Starting embed server on :8081 ..."
    nohup "$ROOT/TRACE/.venv/bin/python" "$ROOT/TRACE/tools/embed_server.py" > "$ROOT/logs/embed.log" 2>&1 &
    EMBED_PID=$!
fi

# C³-RAG
echo "Starting C³-RAG service on :8101 ..."
cd "$ROOT/C3RAG"
source .venv/bin/activate
nohup env "${C3RAG_ENV[@]}" python service.py > "$ROOT/logs/c3rag.log" 2>&1 &
C3RAG_PID=$!

# TRACE
echo "Starting TRACE service on :8102 ..."
cd "$ROOT/TRACE"
source .venv/bin/activate
nohup env "${TRACE_ENV[@]}" PYTHONPATH=src python service.py > "$ROOT/logs/trace.log" 2>&1 &
TRACE_PID=$!

# Backend
echo "Starting backend on :8001 ..."
cd "$ROOT/backend"
source .venv/bin/activate
nohup python -m app.main > "$ROOT/logs/backend.log" 2>&1 &
BACKEND_PID=$!

sleep 5
echo ""
echo "Services started. PIDs:"
if [ -n "$EMBED_PID" ]; then echo "  embed  : $EMBED_PID"; fi
echo "  C³-RAG : $C3RAG_PID"
echo "  TRACE  : $TRACE_PID"
echo "  backend: $BACKEND_PID"
echo "Logs: $ROOT/logs/*.log"

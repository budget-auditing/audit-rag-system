#!/usr/bin/env bash
set -e

echo "=== Smoke Test ==="

echo -n "Embed :8081/health -> "
curl -s http://localhost:8081/health | head -c 200 || echo "FAIL"
echo ""

echo -n "C³-RAG :8101/health -> "
curl -s http://localhost:8101/health | head -c 200 || echo "FAIL"
echo ""

echo -n "TRACE :8102/health -> "
curl -s http://localhost:8102/health | head -c 200 || echo "FAIL"
echo ""

echo -n "Backend :8001/api/health -> "
curl -s http://localhost:8001/api/health | head -c 200 || echo "FAIL"
echo ""

echo -n "Embed embed test -> "
curl -s -X POST http://localhost:8081/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input":"test"}' | python3 -c "import sys,json; d=json.load(sys.stdin); print('dim', len(d['data'][0]['embedding']))" || echo "FAIL"

echo "=== Done ==="

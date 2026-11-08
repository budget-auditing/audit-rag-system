"""BGE-M3 embedding server: dense vectors (/v1/embeddings) and ColBERT token vectors (/v1/colbert).

Configuration via environment variables: EMBED_MODEL, EMBED_DEVICE (e.g. "cuda:0"; unset = auto),
EMBED_FP16 ("0" to disable), HOST, PORT.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing import freeze_support
from typing import Any, Dict, List

from FlagEmbedding import BGEM3FlagModel


MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
DEVICE = os.getenv("EMBED_DEVICE") or None
USE_FP16 = os.getenv("EMBED_FP16", "1") != "0"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8081"))

MODEL: BGEM3FlagModel | None = None


def _parse_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    raw = handler.rfile.read(length) if length else b""
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _normalize_input(data: Dict[str, Any]) -> List[str]:
    inp = data.get("input")
    if isinstance(inp, str):
        return [inp]
    if isinstance(inp, list):
        return [str(x) for x in inp]
    raise ValueError("`input` must be a string or a list of strings")


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: Dict[str, Any]):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "model": MODEL_NAME})
        else:
            self._send_json(404, {"error": "not_found"})

    def do_POST(self):
        if self.path in ("/v1/embeddings", "/embeddings"):
            self._handle_dense()
        elif self.path in ("/v1/colbert", "/colbert"):
            self._handle_colbert()
        else:
            self._send_json(404, {"error": "not_found"})

    def _handle_dense(self):
        try:
            data = _parse_body(self)
            texts = _normalize_input(data)
            vectors = MODEL.encode(texts)["dense_vecs"]
            response = {
                "object": "list",
                "model": MODEL_NAME,
                "data": [
                    {"object": "embedding", "index": i, "embedding": vec.tolist()}
                    for i, vec in enumerate(vectors)
                ],
            }
            self._send_json(200, response)
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})

    def _handle_colbert(self):
        try:
            data = _parse_body(self)
            texts = _normalize_input(data)
            result = MODEL.encode(texts, return_dense=False, return_colbert_vecs=True)
            colbert_vecs = result["colbert_vecs"]
            response = {
                "object": "list",
                "model": MODEL_NAME,
                "data": [
                    {
                        "object": "colbert",
                        "index": i,
                        "num_tokens": len(vecs),
                        "colbert_vecs": vecs.tolist(),
                    }
                    for i, vecs in enumerate(colbert_vecs)
                ],
            }
            self._send_json(200, response)
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})

    def log_message(self, fmt: str, *args):
        return


def main():
    global MODEL
    MODEL = BGEM3FlagModel(MODEL_NAME, devices=DEVICE, use_fp16=USE_FP16)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"BGE-M3 server started: http://{HOST}:{PORT}  model={MODEL_NAME}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    freeze_support()
    main()


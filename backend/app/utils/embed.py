"""Shared embedding utilities (extracted from old HFA graph module)."""
import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def _get_openai_client():
    try:
        from openai import OpenAI  # type: ignore[import-untyped]
        return OpenAI
    except Exception:
        return None


def _openai_embed(texts: List[str], base_url: str, api_key: str, model: str) -> np.ndarray:
    OpenAI = _get_openai_client()
    if OpenAI is None:
        raise RuntimeError("openai package not installed")
    client = OpenAI(base_url=base_url, api_key=api_key)
    resp = client.embeddings.create(model=model, input=texts, encoding_format="float")
    vectors = [item.embedding for item in resp.data]
    if not vectors:
        raise RuntimeError("empty embeddings from openai client")
    return np.array(vectors, dtype=np.float32)


def _legacy_embed(texts: List[str], url: str, model: str) -> np.ndarray:
    """Fallback direct POST to an OpenAI-compatible /v1/embeddings endpoint."""
    import httpx  # type: ignore[import-untyped]
    r = httpx.post(
        f"{url.rstrip('/')}/v1/embeddings",
        json={"input": texts, "model": model or "BAAI/bge-m3", "encoding_format": "float"},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    vectors = [item["embedding"] for item in data["data"]]
    if not vectors:
        raise RuntimeError("empty embeddings from OpenAI-compatible endpoint")
    return np.array(vectors, dtype=np.float32)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Robust embed with OpenAI → Legacy HTTP → zeros fallback."""
    if not texts:
        return np.array([], dtype=np.float32)

    from app.config import settings

    # 1. Try OpenAI client if we have a key or base_url
    if settings.embed_url:
        try:
            arr = _openai_embed(
                texts,
                base_url=settings.embed_url,
                api_key=settings.openai_api_key,
                model=settings.embed_model,
            )
            logger.info("Embedding via OpenAI client succeeded for %d texts", len(texts))
            return arr
        except Exception as exc:
            logger.warning("OpenAI embedding failed: %s", exc)

    # 2. Try legacy direct POST
    try:
        arr = _legacy_embed(
            texts,
            url=settings.embed_url or "http://localhost:8081",
            model=settings.embed_model,
        )
        logger.info("Legacy embedding succeeded for %d texts", len(texts))
        return arr
    except Exception as exc:
        logger.warning("Legacy embedding failed: %s", exc)

    # 3. Zeros fallback
    dim = _guess_dim()
    logger.warning("Embedding failed — returning zero vectors (dim=%d)", dim)
    return np.zeros((len(texts), dim), dtype=np.float32)


def _guess_dim() -> int:
    from app.config import settings
    if settings.embed_dim > 0:
        return settings.embed_dim
    return 1536


def embed_text(text: str) -> np.ndarray:
    embs = embed_texts([text])
    return embs[0] if len(embs) else np.zeros(_guess_dim(), dtype=np.float32)

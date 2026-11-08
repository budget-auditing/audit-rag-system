"""Pydantic-settings configuration."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root, derived from this file so defaults are portable
# (no absolute machine-specific paths baked in).
_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = f"sqlite:///{_REPO_ROOT / 'data' / 'audit.db'}"
    data_dir: str = str(_REPO_ROOT / "data")
    embed_url: str = "http://localhost:8081"
    chat_url: str = "http://localhost:8000/v1"
    c3rag_service_url: str = "http://localhost:8101"
    trace_service_url: str = "http://localhost:8102"
    openai_api_key: str = "sk-placeholder"
    embed_model: str = "BAAI/bge-m3"
    chat_model: str = "gpt-4o-mini"
    embed_dim: int = 0  # 0 → auto-detect on first successful embedding
    enable_llm_extraction: bool = True  # Use LLM for MSG-RAG triple extraction
    msg_extract_workers: int = 60  # Parallel workers for MSG-RAG fact extraction
    extract_max_tokens: int = 2048  # Cap LLM extraction output to bound latency
    extract_reasoning_effort: str = "none"  # "none" disables model thinking (10x faster); "" to omit


settings = Settings()

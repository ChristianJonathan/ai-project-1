# app/settings.py
from __future__ import annotations
import os

def _env(key: str, default: str | None = None) -> str:
    """Tiny helper – returns env var or default (raises if missing)."""
    val = os.getenv(key, default)
    if val is None:
        raise RuntimeError(f"Missing required env var {key!r}")
    return val

class Settings:
    # ── core paths ────────────────────────────────────────────────────
    DB_PATH:     str = _env("DB_PATH", "/data/chat.db")
    LOG_PATH:    str = _env("LOG_PATH", "/data/chat.log")

    # ── LLM / Ollama ─────────────────────────────────────────────────
    LLM_URL:     str = _env("LLM_URL",
                            "http://host.docker.internal:11434/api/chat")
    MODEL_DEFAULT: str = _env("MODEL_DEFAULT", "qwen3:1.7b")

    # ── Auth / rate-limit ────────────────────────────────────────────
    AUTH_USER:   str = _env("AUTH_USER", "admin")
    AUTH_PASS:   str = _env("AUTH_PASS", "admin")
    RATE_LIMIT:  str = _env("RATE_LIMIT", "10/60")          # req per seconds
    STREAM_LIMIT:str = _env("STREAM_LIMIT", "20/60")

    # ── Prompt tuning ────────────────────────────────────────────────
    SYSTEM_PROMPT: str = (
        "You are a helpful assistant. "
        "Think step-by-step **in ≤ 100 tokens**, enclosed in <think>…</think>. "
        "Then answer the user clearly."
    )

settings = Settings()

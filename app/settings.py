# app/settings.py
from __future__ import annotations
import os

def _env(key: str, default: str | None = None) -> str:
    """Tiny helper – returns env var or default (raises if missing)."""
    val = os.getenv(key, default)
    if val is None:
        raise RuntimeError(f"Missing required env var {key!r}")
    return val

def _csv(key: str, default: str) -> list[str]:
    """Read an env-var that contains a comma-separated list."""
    return [s.strip() for s in _env(key, default).split(",") if s.strip()]

class Settings:
    # ── core paths ────────────────────────────────────────────────────
    DB_PATH:     str = _env("DB_PATH", "/data/chat.db")
    LOG_PATH:    str = _env("LOG_PATH", "/data/chat.log")

    # ── LLM / Ollama ─────────────────────────────────────────────────
    LLM_URL:     str = _env("LLM_URL",
                            "http://host.docker.internal:11434/api/chat")
    
    # ── Local models ────────────────────────────────────────────────
    MODEL_CHOICES: list[str] = _csv(        # <── NEW
        "MODEL_CHOICES",
        "qwen3:1.7b,gemma3:4b"              # default list
    )
    MODEL_DEFAULT: str = MODEL_CHOICES[0]   # first one = default    

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

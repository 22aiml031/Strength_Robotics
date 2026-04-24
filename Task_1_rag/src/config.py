from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

if load_dotenv:
    load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    supabase_url: str
    supabase_key: str
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "all-minilm"
    ollama_chat_model: str = "gemma3:4b"
    embedding_dim: int = 384
    match_count: int = 8
    chunk_size: int = 1000
    chunk_overlap: int = 200


def get_settings(require_supabase: bool = True) -> Settings:
    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
        or os.getenv("SUPABASE_KEY")
        or ""
    )

    if require_supabase and (not supabase_url or not supabase_key):
        raise RuntimeError(
            "Missing Supabase configuration. Set SUPABASE_URL and "
            "SUPABASE_SERVICE_ROLE_KEY in .env or Streamlit secrets."
        )

    return Settings(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        ollama_base_url=os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        ).rstrip("/"),
        ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "all-minilm"),
        ollama_chat_model=os.getenv("OLLAMA_CHAT_MODEL", "gemma3:4b"),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "384")),
        match_count=int(os.getenv("MATCH_COUNT", "8")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
    )
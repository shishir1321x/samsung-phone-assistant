"""Central configuration, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
PHONES_JSON = DATA_DIR / "phones.json"

load_dotenv(BASE_DIR / ".env")

# The 15 models scraped by default: every base + Ultra flagship from S21 to S26,
# plus the current foldables and the mid-range A-series, so rankings and
# comparisons have a meaningful spread of prices, sizes and generations.
DEFAULT_MODELS: list[str] = [
    "Galaxy S21",
    "Galaxy S21 Ultra",
    "Galaxy S22",
    "Galaxy S22 Ultra",
    "Galaxy S23",
    "Galaxy S23 Ultra",
    "Galaxy S24",
    "Galaxy S24 Ultra",
    "Galaxy S25",
    "Galaxy S25 Ultra",
    "Galaxy S26",
    "Galaxy S26 Ultra",
    "Galaxy Z Fold7",
    "Galaxy Z Flip7",
    "Galaxy A57",
]


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    database_url: str
    # LLM: auto | huggingface | ollama | openai | openai_compatible | transformers | none
    llm_provider: str
    llm_model: str | None
    llm_base_url: str | None
    llm_api_key: str | None
    llm_temperature: float
    llm_max_tokens: int
    llm_timeout: float
    hf_token: str | None
    ollama_base_url: str
    openai_api_key: str | None
    # Embeddings: auto | fastembed | lexical
    embedding_backend: str
    embedding_model: str
    # Scraper
    scrape_delay: float
    scrape_max_pages: int
    user_agent: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    default_db = f"sqlite:///{(DATA_DIR / 'samsung_phones.db').as_posix()}"
    return Settings(
        database_url=_env("DATABASE_URL", default_db),
        llm_provider=(_env("LLM_PROVIDER", "auto") or "auto").lower(),
        llm_model=_env("LLM_MODEL"),
        llm_base_url=_env("LLM_BASE_URL"),
        llm_api_key=_env("LLM_API_KEY"),
        llm_temperature=float(_env("LLM_TEMPERATURE", "0.2")),
        llm_max_tokens=int(_env("LLM_MAX_TOKENS", "900")),
        llm_timeout=float(_env("LLM_TIMEOUT", "120")),
        hf_token=_env("HF_TOKEN") or _env("HUGGINGFACEHUB_API_TOKEN"),
        ollama_base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434"),
        openai_api_key=_env("OPENAI_API_KEY"),
        embedding_backend=(_env("EMBEDDING_BACKEND", "auto") or "auto").lower(),
        embedding_model=_env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        scrape_delay=float(_env("SCRAPE_DELAY", "5")),
        scrape_max_pages=int(_env("SCRAPE_MAX_PAGES", "15")),
        user_agent=_env(
            "SCRAPER_USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/130.0 Safari/537.36",
        ),
    )


def reset_settings() -> None:
    """Forget cached settings (used by tests after changing env vars)."""
    get_settings.cache_clear()

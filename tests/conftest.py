"""Test setup: isolated SQLite DB loaded from data/phones.json, offline embeddings, no LLM."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def test_database(tmp_path_factory):
    db_file = tmp_path_factory.mktemp("db") / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_file.as_posix()}"
    os.environ["EMBEDDING_BACKEND"] = "lexical"
    os.environ["LLM_PROVIDER"] = "none"

    from app.config import reset_settings
    from app.db.session import reset_engine
    from app.llm.provider import reset_llm
    from app.rag.embeddings import reset_embedder
    from app.rag.vector_store import invalidate_index

    reset_settings()
    reset_engine()
    reset_embedder()
    reset_llm()
    invalidate_index()

    from app.scraper.pipeline import load_json, store_phones

    store_phones(load_json())
    yield
    reset_engine()


@pytest.fixture(scope="session")
def assistant():
    from app.assistant import PhoneAssistant

    return PhoneAssistant(llm=None)


class FakeLLM:
    """Scripted stand-in for an LLM: returns the given outputs in order (last one repeats)."""

    name = "fake-llm"

    def __init__(self, *outputs: str):
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def generate(self, system: str, prompt: str, max_tokens: int | None = None) -> str:
        self.prompts.append(prompt)
        return self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]


@pytest.fixture
def fake_llm():
    return FakeLLM

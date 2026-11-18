"""Embedding index persisted in the ``rag_chunks`` table, searched in memory.

The corpus is small (15 phones x ~15 chunks), so cosine similarity over a
NumPy matrix is exact and fast; embeddings live in the database so the index
survives restarts and is shared by every API worker.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.db.models import RagChunk
from app.db.repository import all_phones
from app.rag.documents import build_documents
from app.rag.embeddings import Embedder, get_embedder

log = logging.getLogger(__name__)


@dataclass
class Hit:
    chunk_id: int
    phone_id: int
    phone_name: str
    category: str
    text: str
    score: float

    def to_dict(self) -> dict:
        return {"phone": self.phone_name, "category": self.category, "score": round(self.score, 4), "text": self.text}


def build_index(session: Session, embedder: Embedder | None = None) -> int:
    """(Re)build all chunks and embeddings from the phones table."""
    embedder = embedder or get_embedder()
    session.execute(delete(RagChunk))
    docs = [doc for phone in all_phones(session) for doc in build_documents(phone)]
    if not docs:
        return 0
    vectors = embedder.embed_documents([d.text for d in docs])
    for doc, vector in zip(docs, vectors):
        session.add(RagChunk(phone_id=doc.phone_id, category=doc.category, content=doc.text,
                             embedding=np.asarray(vector, dtype=np.float32).tobytes(),
                             embedding_model=embedder.name))
    session.flush()
    log.info("Indexed %d chunks with %s", len(docs), embedder.name)
    return len(docs)


def index_status(session: Session) -> tuple[int, str | None]:
    count = session.scalar(select(func.count(RagChunk.id))) or 0
    model = session.scalar(select(RagChunk.embedding_model).limit(1))
    return count, model


class VectorIndex:
    def __init__(self, session: Session, embedder: Embedder | None = None):
        self.embedder = embedder or get_embedder()
        rows = session.execute(
            select(RagChunk.id, RagChunk.phone_id, RagChunk.category, RagChunk.content,
                   RagChunk.embedding, RagChunk.embedding_model)
        ).all()
        from app.db.models import Phone  # local import keeps module import light

        names = dict(session.execute(select(Phone.id, Phone.name)).all())
        rows = [r for r in rows if r.embedding_model == self.embedder.name]
        self.meta = [(r.id, r.phone_id, names.get(r.phone_id, "?"), r.category, r.content) for r in rows]
        self.matrix = (
            np.stack([np.frombuffer(r.embedding, dtype=np.float32) for r in rows])
            if rows else np.zeros((0, 1), dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.meta)

    def search(
        self,
        query: str,
        k: int = 6,
        phone_names: list[str] | None = None,
        boost_categories: list[str] | None = None,
        boost: float = 0.12,
    ) -> list[Hit]:
        if not self.meta:
            return []
        q = self.embedder.embed_query(query)
        scores = self.matrix @ q
        hits = []
        wanted = set(phone_names or [])
        boosted = {c.lower() for c in (boost_categories or [])}
        for i, (chunk_id, phone_id, name, category, text) in enumerate(self.meta):
            if wanted and name not in wanted:
                continue
            score = float(scores[i]) + (boost if category.lower() in boosted else 0.0)
            hits.append(Hit(chunk_id, phone_id, name, category, text, score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]


_index: VectorIndex | None = None
_lock = threading.Lock()


def get_index(session: Session) -> VectorIndex:
    """Process-wide index; built on first use if the table is empty or stale."""
    global _index
    with _lock:
        if _index is None:
            embedder = get_embedder()
            count, model = index_status(session)
            if count == 0 or model != embedder.name:
                build_index(session, embedder)
                session.commit()
            _index = VectorIndex(session, embedder)
        return _index


def invalidate_index() -> None:
    global _index
    with _lock:
        _index = None

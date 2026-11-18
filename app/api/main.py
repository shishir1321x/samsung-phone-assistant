"""FastAPI application: chat, reviews, comparisons, rankings and raw phone data.

Run with:  uvicorn app.api.main:app --reload
Docs at:   http://127.0.0.1:8000/docs   |   Chat UI at http://127.0.0.1:8000/
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import func, select

from app.analytics import METRICS, rankings_for
from app.api.schemas import (
    BatchRequest,
    BatchResponse,
    ChatRequest,
    ChatResponse,
    CompareRequest,
    HealthResponse,
    PhoneDetail,
    PhoneSummary,
    RankingResponse,
    ReviewRequest,
    ReviewResponse,
)
from app.assistant import PhoneAssistant, PhoneNotFound
from app.config import PHONES_JSON, get_settings
from app.db.models import Phone
from app.db.repository import all_phones, phone_by_name, phone_detail, phone_summary
from app.db.session import init_db, session_scope
from app.llm.provider import get_llm
from app.rag.vector_store import get_index, index_status

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


def bootstrap() -> None:
    """Create tables, load the bundled dataset if the DB is empty, and warm the RAG index."""
    init_db()
    with session_scope() as db:
        count = db.scalar(select(func.count(Phone.id))) or 0
    if count == 0 and PHONES_JSON.exists():
        from app.scraper.pipeline import load_json, store_phones

        log.info("Empty database: loading %s", PHONES_JSON)
        store_phones(load_json(PHONES_JSON))
    with session_scope() as db:
        get_index(db)


@asynccontextmanager
async def lifespan(app: FastAPI):
    bootstrap()
    app.state.assistant = PhoneAssistant()
    yield


app = FastAPI(
    title="Samsung Phone Query & Review API",
    description=(
        "Ask questions about Samsung phones (RAG over GSMArena data), get multi-agent generated reviews, "
        "compare models and rank them by any spec or lab benchmark."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def assistant() -> PhoneAssistant:
    return app.state.assistant


@app.get("/", include_in_schema=False)
def chat_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    with session_scope() as db:
        phones = db.scalar(select(func.count(Phone.id))) or 0
        chunks, embedder = index_status(db)
    llm = get_llm()
    url = get_settings().database_url
    return HealthResponse(status="ok", database=url.split(":", 1)[0].split("+")[0], phones=phones, rag_chunks=chunks,
                          embedder=embedder, llm=llm.name if llm else "none (grounded templates)")


# ---------------------------------------------------------------------- chat
@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(request: ChatRequest) -> dict:
    """Ask anything about the phones. Pass the returned `session_id` back for follow-up questions."""
    return assistant().ask(request.message, request.session_id)


@app.post("/chat/batch", response_model=BatchResponse, tags=["chat"])
def chat_batch(request: BatchRequest) -> dict:
    """Answer several questions in one call (concurrently, or in order within one conversation)."""
    started = time.perf_counter()
    if request.sequential:
        session_id = request.session_id
        results = []
        for question in request.questions:
            result = assistant().ask(question, session_id)
            session_id = result["session_id"]
            results.append(result)
    else:
        with ThreadPoolExecutor(max_workers=min(8, len(request.questions))) as pool:
            results = list(pool.map(lambda q: assistant().ask(q), request.questions))
    return {"results": results, "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}


@app.delete("/chat/{session_id}", tags=["chat"])
def reset_chat(session_id: str) -> dict:
    return {"session_id": session_id, "cleared": assistant().reset_session(session_id)}


# ---------------------------------------------------------------------- reviews & comparisons
@app.post("/reviews", response_model=ReviewResponse, tags=["reviews"])
def create_review(request: ReviewRequest) -> dict:
    """Run the multi-agent pipeline (spec retrieval -> analyst -> writer -> fact-checker) for one phone."""
    try:
        return assistant().review(request.phone, regenerate=request.regenerate)
    except PhoneNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/reviews/{phone}", response_model=ReviewResponse, tags=["reviews"])
def get_review(phone: str) -> dict:
    """Stored review for a phone (generated on first request)."""
    try:
        return assistant().review(phone, regenerate=False)
    except PhoneNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/compare", response_model=ChatResponse, tags=["chat"])
def compare(request: CompareRequest) -> dict:
    try:
        return assistant().compare(request.phones, request.aspects)
    except PhoneNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------- data
@app.get("/phones", response_model=list[PhoneSummary], tags=["data"])
def list_phones() -> list[dict]:
    with session_scope() as db:
        return [phone_summary(p) for p in all_phones(db)]


@app.get("/phones/{phone}", response_model=PhoneDetail, tags=["data"])
def get_phone(phone: str) -> dict:
    """Full scraped spec sheet. `phone` can be a name or alias, e.g. `s23-ultra` or `Galaxy S22`."""
    with session_scope() as db:
        try:
            name = assistant().resolve_phone(db, phone.replace("-", " ").replace("_", " "))
        except PhoneNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return phone_detail(phone_by_name(db, name))


@app.get("/rankings/{metric}", response_model=RankingResponse, tags=["data"])
def rankings(
    metric: str,
    max_price: float | None = Query(None, description="Only phones at or below this USD price"),
    ascending: bool | None = Query(None, description="Override the metric's natural order"),
    limit: int | None = Query(None, ge=1, le=50),
) -> dict:
    if metric not in METRICS:
        raise HTTPException(status_code=404, detail=f"Unknown metric. Choose one of: {', '.join(METRICS)}")
    with session_scope() as db:
        phones = all_phones(db)
        result = rankings_for(phones, metric, ascending=ascending, max_price=max_price, limit=limit)
        return {"rankings": [r.to_dict() for r in result]}


@app.get("/metrics", tags=["data"])
def metrics() -> list[dict]:
    return [{"key": m.key, "label": m.label, "unit": m.unit, "higher_is_better": m.higher_is_better, "note": m.note}
            for m in METRICS.values()]

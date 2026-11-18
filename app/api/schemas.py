"""Request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, examples=["What are the camera specs of the Galaxy S23?"])
    session_id: str | None = Field(None, max_length=100, description="Reuse to keep conversation context.")


class AgentStep(BaseModel):
    agent: str
    role: str
    action: str
    ms: float


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    intent: str
    phones: list[str]
    aspects: list[str]
    generator: str | None = Field(None, description="LLM that wrote the answer, or 'template' for grounded text.")
    verification: dict[str, Any]
    sources: list[dict[str, Any]]
    rankings: list[dict[str, Any]]
    agents: list[AgentStep]
    elapsed_ms: float


class BatchRequest(BaseModel):
    questions: list[str] = Field(..., min_length=1, max_length=20)
    session_id: str | None = None
    sequential: bool = Field(False, description="Answer in order within one conversation (follow-ups work). "
                                                 "Otherwise questions are answered concurrently and independently.")


class BatchResponse(BaseModel):
    results: list[ChatResponse]
    elapsed_ms: float


class ReviewRequest(BaseModel):
    phone: str = Field(..., min_length=1, max_length=100, examples=["Galaxy S24 Ultra"])
    regenerate: bool = Field(False, description="Ignore the stored review and run the agents again.")


class ReviewResponse(BaseModel):
    phone: str
    review: str
    scores: dict[str, float]
    overall: float | None = None
    pros: list[str]
    cons: list[str]
    predecessor: str | None = None
    vs_predecessor: list[str] = []
    generator: str | None
    verification: dict[str, Any]
    cached: bool
    created_at: str | None
    agents: list[AgentStep]
    elapsed_ms: float


class CompareRequest(BaseModel):
    phones: list[str] = Field(..., min_length=2, max_length=4, examples=[["Galaxy S23", "Galaxy S22"]])
    aspects: list[str] | None = Field(None, examples=[["performance", "battery"]])


class PhoneSummary(BaseModel):
    id: int
    name: str
    url: str
    image_url: str | None
    announced: str | None
    display_size_in: float | None
    chipset: str | None
    ram_gb: float | None
    main_camera_mp: float | None
    battery_mah: int | None
    price_usd: float | None


class PhoneDetail(PhoneSummary):
    key_specs: dict[str, Any]
    prices: list[dict[str, Any]]
    benchmarks: list[dict[str, Any]]
    specifications: dict[str, list[dict[str, str]]]
    scraped_at: str | None


class RankingResponse(BaseModel):
    rankings: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    database: str
    phones: int
    rag_chunks: int
    embedder: str | None
    llm: str

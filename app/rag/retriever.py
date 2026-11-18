"""Hybrid retrieval: structured SQL facts + semantic vector search + keyword evidence.

* Questions about named phones pull the exact spec rows for the asked aspects.
* Superlatives ("best battery", "cheapest") become rankings over typed columns
  and lab benchmarks - vector search alone cannot rank.
* Open questions ("which phones have a stylus?") use semantic search across
  all chunks plus a keyword scan of every phone's spec sheet as evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.analytics import METRICS, BenchmarkDelta, Ranking, common_benchmarks, position, rankings_for
from app.db.models import Phone
from app.db.repository import all_phones, benchmark_dicts
from app.rag.nlu import ASPECT_CATEGORIES, QueryPlan
from app.rag.vector_store import Hit, get_index
from app.scraper.parsers import hours_to_text

_STOP = {"the", "a", "an", "of", "and", "or", "is", "are", "to", "in", "on", "for", "with", "what", "which", "who",
         "does", "do", "did", "how", "has", "have", "it", "its", "this", "that", "samsung", "galaxy", "phone",
         "phones", "model", "models", "support", "supports", "feature", "features", "any", "there", "come", "comes",
         "me", "tell", "show", "can", "you", "i", "my", "about", "all", "get", "got", "one", "ones", "be", "use"}


@dataclass
class RetrievedContext:
    plan: QueryPlan
    phones: list[Phone] = field(default_factory=list)
    all_phones: list[Phone] = field(default_factory=list)
    rankings: list[Ranking] = field(default_factory=list)
    deltas: list[BenchmarkDelta] = field(default_factory=list)
    positions: dict[str, tuple[int, int]] = field(default_factory=dict)
    hits: list[Hit] = field(default_factory=list)
    evidence: dict[str, list[str]] = field(default_factory=dict)

    def sources(self) -> list[dict]:
        seen: dict[str, dict] = {}
        for phone in self.phones:
            seen[phone.name] = {"phone": phone.name, "url": phone.url}
        for ranking in self.rankings:
            for row in ranking.rows[:5]:
                phone = self.by_name(row.name)
                if phone:
                    seen.setdefault(phone.name, {"phone": phone.name, "url": phone.url})
        for hit in self.hits[:5]:
            phone = self.by_name(hit.phone_name)
            if phone:
                seen.setdefault(phone.name, {"phone": phone.name, "url": phone.url})
        for name in self.evidence:
            phone = self.by_name(name)
            if phone:
                seen.setdefault(phone.name, {"phone": phone.name, "url": phone.url})
        return list(seen.values())

    def by_name(self, name: str) -> Phone | None:
        return next((p for p in self.all_phones if p.name == name), None)


def keywords(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9.+-]*", text.lower())
    return [t for t in tokens if t not in _STOP and len(t) > 1]


def keyword_evidence(phones: list[Phone], query: str, limit_rows: int = 2) -> dict[str, list[str]]:
    """Spec rows that contain the query's keywords, per phone (all keywords first, then most)."""
    words = keywords(query)
    if not words:
        return {}
    found: dict[str, list[tuple[int, str]]] = {}
    for phone in phones:
        for spec in phone.specs:
            text = f"{spec.category} / {spec.name}: {spec.value}".replace("\n", "; ")
            low = text.lower()
            matched = sum(1 for w in words if re.search(rf"(?<![a-z0-9]){re.escape(w)}", low))
            if matched:
                found.setdefault(phone.name, []).append((matched, text))
    if not found:
        return {}
    best = max(score for rows in found.values() for score, _ in rows)
    needed = len(words) if best == len(words) else best
    evidence = {}
    for name, rows in found.items():
        good = [t for s, t in sorted(rows, key=lambda r: -r[0]) if s >= needed]
        if good:
            evidence[name] = good[:limit_rows]
    return evidence


def retrieve(session: Session, plan: QueryPlan, k: int = 6) -> RetrievedContext:
    phones = all_phones(session)
    ctx = RetrievedContext(plan=plan, all_phones=phones)
    ctx.phones = [p for name in plan.phones for p in phones if p.name == name]

    if plan.intent in ("greeting", "list", "unknown_phone"):
        return ctx

    categories = sorted({c for a in plan.aspects for c in ASPECT_CATEGORIES.get(a, [])})
    index = get_index(session)

    if plan.intent == "rank" and plan.metric:
        ctx.rankings = rankings_for(phones, plan.metric, ascending=plan.ascending, max_price=plan.max_price)
    if plan.metric and plan.intent in ("spec", "compare"):
        for phone in ctx.phones:
            pos = position(phones, phone, plan.metric)
            if pos:
                ctx.positions[phone.name] = pos
    if plan.intent == "compare" and len(ctx.phones) >= 2:
        ctx.deltas = common_benchmarks(ctx.phones)

    if ctx.phones:
        ctx.hits = index.search(plan.query, k=k * len(ctx.phones), phone_names=[p.name for p in ctx.phones],
                                boost_categories=categories)
    else:
        ctx.hits = index.search(plan.query, k=k, boost_categories=categories)
        if plan.intent == "general":
            ctx.evidence = keyword_evidence(phones, plan.query)
    return ctx


# ====================================================================== context for the LLM
def phone_fact_sheet(phone: Phone, aspects: list[str] | None = None) -> str:
    """Compact, LLM-friendly spec sheet (all rows, or only the asked aspects)."""
    wanted = {c for a in (aspects or []) for c in ASPECT_CATEGORIES.get(a, [])}
    lines = [f"### {phone.name}", f"Source: {phone.url}"]
    for spec in phone.specs:
        if wanted and spec.category not in wanted and spec.category != "Launch":
            continue
        if spec.category == "Network" and spec.name.endswith("bands"):
            continue
        if spec.category == "Our Tests" and spec.name == "Camera":
            continue
        lines.append(f"- {spec.category} / {spec.name}: {spec.value}".replace("\n", "; "))
    if not wanted or wanted & {"Our Tests", "Platform"}:
        for b in benchmark_dicts(phone):
            lines.append(f"- Benchmark {b['test']} ({b['version']}): {b['display']}")
    return "\n".join(lines)


def ranking_text(ranking: Ranking, top: int | None = None) -> str:
    rows = ranking.rows[: top or len(ranking.rows)]
    lines = [f"Ranking by {ranking.label} ({'lowest' if ranking.ascending else 'highest'} first"
             + (f", only phones priced <= ${ranking.max_price:,.0f}" if ranking.max_price else "") + "):"]
    for row in rows:
        price = f" (price ${row.price_usd:,.2f})" if row.price_usd else ""
        detail = f" [{row.detail}]" if row.detail else ""
        lines.append(f"{row.rank}. {row.name}: {row.display}{detail}{price}")
    if ranking.missing:
        lines.append(f"No data for: {', '.join(ranking.missing)}")
    if ranking.metric.note:
        lines.append(f"Note: {ranking.metric.note}")
    return "\n".join(lines)


def delta_text(delta: BenchmarkDelta) -> str:
    def fmt(score: float) -> str:
        if delta.unit == "points":
            return f"{score:,.0f}"
        if delta.unit == "hours" and delta.test == "Battery active use":
            return hours_to_text(score) or f"{score:g}h"
        return f"{score:g} {delta.unit}"

    parts = ", ".join(f"{name.replace('Samsung ', '')}: {fmt(score)}" for name, score in delta.values.items())
    return (f"{delta.test} ({delta.version}) - {parts}; "
            f"{delta.leader.replace('Samsung ', '')} leads by {delta.lead_pct}%")


def context_for_llm(ctx: RetrievedContext) -> str:
    plan = ctx.plan
    blocks = []
    if ctx.phones:
        aspects = plan.aspects if plan.intent != "review" else []
        blocks.append("\n\n".join(phone_fact_sheet(p, aspects) for p in ctx.phones))
    for name, (rank, total) in ctx.positions.items():
        blocks.append(f"{name} ranks #{rank} of {total} phones for {METRICS[plan.metric].label}.")
    if ctx.deltas:
        blocks.append("Benchmarks run on the same version for all compared phones:\n"
                      + "\n".join(f"- {delta_text(d)}" for d in ctx.deltas))
    for ranking in ctx.rankings:
        blocks.append(ranking_text(ranking))
    if ctx.evidence:
        blocks.append("Spec rows matching the question's keywords:\n" + "\n".join(
            f"- {name}: {' || '.join(rows)}" for name, rows in ctx.evidence.items()))
    if ctx.hits and not ctx.phones:
        blocks.append("Semantically related spec excerpts:\n" + "\n".join(f"- {h.text}" for h in ctx.hits))
    return "\n\n".join(blocks)

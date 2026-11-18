"""Multi-agent system built with LangGraph (LangChain's agent orchestration library).

                      +--------------+
     query ---------> | Coordinator  |  resolves phones, intent, aspects
                      +------+-------+
                             v
                      +--------------+
                      | Spec         |  SQL facts, rankings, benchmark deltas,
                      | Retrieval    |  vector search (RAG), keyword evidence
                      +------+-------+
              review/compare |            other questions
                      +------v-------+    |
                      | Benchmark &  |    |
                      | Market       |    |
                      | Analyst      |    |
                      +--+--------+--+    |
               review    |        | compare
                 +-------v--+   +-v-------v--+
                 | Review   |   | Answer     |
                 | Writer   |   | Agent      |
                 +-------+--+   +--+---------+
                         v         v
                      +--------------+   unsupported numbers -> back to the
                      | Fact-Checker | ---------------------> writer (1 revision),
                      +------+-------+   then grounded fallback
                             v
                            END

Data agents are deterministic tool users (exact, auditable); the LLM is used
where language matters (writing reviews and answers) and is kept honest by
the fact-checker.
"""

from __future__ import annotations

import logging
import operator
import time
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.agents.analyst import Analysis, analyze
from app.agents.fact_check import verify
from app.agents.writer import (
    ANSWER_SYSTEM,
    REVIEW_SYSTEM,
    answer_prompt,
    complete_review,
    review_prompt,
    takeaway_prompt,
    template_review,
)
from app.rag.answers import compose_answer
from app.rag.nlu import QueryPlan, plan_query
from app.rag.retriever import RetrievedContext, context_for_llm, phone_fact_sheet, retrieve

log = logging.getLogger(__name__)

MAX_REVISIONS = 1
REVIEW_MAX_TOKENS = 1600
DIRECT_INTENTS = {"greeting", "list", "unknown_phone"}
DATA_HEAVY_INTENTS = {"compare", "rank"}


class AgentState(TypedDict, total=False):
    query: str
    mode: str  # "chat" | "review"
    target_phone: str | None
    history: list[dict]
    context_phones: list[str]
    context_aspects: list[str]
    plan: QueryPlan
    retrieved: RetrievedContext
    context: str
    analyses: dict[str, Analysis]
    draft: str
    output: str
    generator: str
    verification: dict
    revisions: int
    feedback: str | None
    next_step: str  # set by the fact-checker: "revise" | "done"
    trace: Annotated[list[dict], operator.add]


def _cfg(config: RunnableConfig, key: str) -> Any:
    return (config.get("configurable") or {}).get(key)


class Agent:
    name = "Agent"
    role = ""

    def run(self, state: AgentState, config: RunnableConfig) -> dict:
        started = time.perf_counter()
        update, action = self.act(state, config)
        entry = {"agent": self.name, "role": self.role, "action": action,
                 "ms": round((time.perf_counter() - started) * 1000, 1)}
        update["trace"] = [entry]
        return update

    def act(self, state: AgentState, config: RunnableConfig) -> tuple[dict, str]:  # pragma: no cover
        raise NotImplementedError


class CoordinatorAgent(Agent):
    name = "Coordinator"
    role = "Understands the request: resolves phone names, intent and aspects, and routes the work."

    def act(self, state, config):
        matcher = _cfg(config, "matcher")
        if state.get("mode") == "review":
            plan = QueryPlan(query=state["query"], intent="review", phones=[state["target_phone"]])
        else:
            plan = plan_query(state["query"], matcher, state.get("context_phones"), state.get("context_aspects"))
        action = (f"intent={plan.intent}; phones={[p.replace('Samsung ', '') for p in plan.phones] or '-'}; "
                  f"aspects={plan.aspects or '-'}" + (f"; metric={plan.metric}" if plan.metric else "")
                  + (f"; max_price=${plan.max_price:,.0f}" if plan.max_price else "")
                  + ("; used conversation context" if plan.used_context else ""))
        return {"plan": plan, "revisions": 0, "feedback": None}, action


class SpecRetrievalAgent(Agent):
    name = "Spec Retrieval Agent"
    role = "Fetches phone specifications, lab benchmarks and prices from the database (SQL + vector search)."

    def act(self, state, config):
        session = _cfg(config, "session")
        ctx = retrieve(session, state["plan"])
        context = context_for_llm(ctx)
        action = (f"loaded {len(ctx.phones)} phone spec sheet(s), {len(ctx.rankings)} ranking(s), "
                  f"{len(ctx.deltas)} shared benchmark(s), {len(ctx.hits)} vector hit(s), "
                  f"{len(ctx.evidence)} keyword match(es)")
        return {"retrieved": ctx, "context": context}, action


class BenchmarkAnalystAgent(Agent):
    name = "Benchmark & Market Analyst"
    role = "Scores phones against the whole dataset (lab tests, specs, price) and derives pros/cons."

    def act(self, state, config):
        ctx = state["retrieved"]
        analyses = {phone.name: analyze(ctx.all_phones, phone) for phone in ctx.phones}
        extra = "\n\n".join(a.as_text() for a in analyses.values())
        summary = "; ".join(f"{name.replace('Samsung ', '')} overall {a.overall}/10" for name, a in analyses.items())
        return {"analyses": analyses, "context": state["context"] + "\n\n" + extra}, summary


class ReviewWriterAgent(Agent):
    name = "Review Writer"
    role = "Writes a detailed product review combining the specs with the analyst's findings."

    def act(self, state, config):
        llm = _cfg(config, "llm")
        ctx = state["retrieved"]
        phone = ctx.phones[0]
        analysis = state["analyses"][phone.name]
        draft = template_review(phone, analysis)
        if llm is None:
            return {"draft": draft, "output": draft, "generator": "template"}, "wrote grounded template review (no LLM configured)"
        try:
            text = llm.generate(REVIEW_SYSTEM, review_prompt(phone, phone_fact_sheet(phone), analysis, state.get("feedback")),
                                max_tokens=REVIEW_MAX_TOKENS)
            text = complete_review(text, draft)
            action = f"wrote review with {llm.name}" + (" (revision)" if state.get("feedback") else "")
            return {"draft": draft, "output": text, "generator": llm.name}, action
        except Exception as exc:  # network errors, rate limits, ...
            log.warning("LLM review failed: %s", exc)
            return {"draft": draft, "output": draft, "generator": "template"}, f"LLM failed ({type(exc).__name__}); used template"


class AnswerAgent(Agent):
    name = "Answer Agent"
    role = "Answers the user's question from the retrieved facts (RAG generation)."

    def act(self, state, config):
        llm = _cfg(config, "llm")
        ctx = state["retrieved"]
        draft = compose_answer(ctx)
        analyses = state.get("analyses") or {}
        if ctx.plan.intent == "compare" and analyses:
            categories = list(dict.fromkeys(k for a in analyses.values() for k in a.scores))
            header = "| Analyst score (1-10) | " + " | ".join(n.replace("Samsung ", "") for n in analyses) + " |"
            rows = [header, "|---" * (len(analyses) + 1) + "|"]
            for cat in categories:
                rows.append(f"| {cat} | " + " | ".join(str(a.scores.get(cat, "n/a")) for a in analyses.values()) + " |")
            rows.append("| **Overall** | " + " | ".join(f"**{a.overall}**" for a in analyses.values()) + " |")
            draft += "\n\n**Analyst scores** (relative to all phones in the database):\n" + "\n".join(rows)
        if llm is None or ctx.plan.intent in DIRECT_INTENTS:
            return {"draft": draft, "output": draft, "generator": "template"}, "composed grounded answer"
        try:
            args = (state["query"], state["context"], draft, state.get("history") or [], state.get("feedback"))
            if ctx.plan.intent in DATA_HEAVY_INTENTS:
                # the LLM writes the verdict; the exact tables/rankings always follow it
                text = llm.generate(ANSWER_SYSTEM, takeaway_prompt(*args), max_tokens=300).strip() + "\n\n" + draft
            else:
                text = llm.generate(ANSWER_SYSTEM, answer_prompt(*args))
            action = f"generated answer with {llm.name}" + (" (revision)" if state.get("feedback") else "")
            return {"draft": draft, "output": text, "generator": llm.name}, action
        except Exception as exc:
            log.warning("LLM answer failed: %s", exc)
            return {"draft": draft, "output": draft, "generator": "template"}, f"LLM failed ({type(exc).__name__}); used grounded answer"


class FactCheckerAgent(Agent):
    name = "Fact-Checker"
    role = "Verifies every figure in the draft against the database; sends it back or falls back if needed."

    def act(self, state, config):
        sources = "\n".join([state.get("query", ""), state.get("context", ""), state.get("draft", "")])
        result = verify(state["output"], sources)
        update: dict = {"verification": result.to_dict(), "next_step": "done"}
        if result.passed:
            return update, f"verified {result.checked} figure(s): all grounded"
        if state.get("generator") != "template" and state.get("revisions", 0) < MAX_REVISIONS:
            update.update(revisions=state.get("revisions", 0) + 1, feedback=result.feedback(), next_step="revise")
            return update, f"{len(result.unsupported)} unsupported figure(s) {result.unsupported[:5]}; requested a revision"
        update.update(output=state["draft"], generator="template (fact-check fallback)",
                      verification={**result.to_dict(), "fallback": True})
        return update, f"{len(result.unsupported)} unsupported figure(s) remained; replaced with the grounded draft"


def _after_retrieval(state: AgentState) -> str:
    return "analyst" if state["plan"].intent in ("review", "compare") and state["retrieved"].phones else "responder"


def _after_analysis(state: AgentState) -> str:
    return "review_writer" if state["plan"].intent == "review" else "responder"


def _after_check(state: AgentState) -> str:
    if state.get("next_step") == "revise":
        return "review_writer" if state["plan"].intent == "review" else "responder"
    return "end"


def build_graph():
    coordinator = CoordinatorAgent()
    retriever = SpecRetrievalAgent()
    analyst = BenchmarkAnalystAgent()
    writer = ReviewWriterAgent()
    responder = AnswerAgent()
    checker = FactCheckerAgent()

    graph = StateGraph(AgentState)
    graph.add_node("coordinator", coordinator.run)
    graph.add_node("spec_retriever", retriever.run)
    graph.add_node("analyst", analyst.run)
    graph.add_node("review_writer", writer.run)
    graph.add_node("responder", responder.run)
    graph.add_node("fact_checker", checker.run)

    graph.add_edge(START, "coordinator")
    graph.add_edge("coordinator", "spec_retriever")
    graph.add_conditional_edges("spec_retriever", _after_retrieval, {"analyst": "analyst", "responder": "responder"})
    graph.add_conditional_edges("analyst", _after_analysis, {"review_writer": "review_writer", "responder": "responder"})
    graph.add_edge("review_writer", "fact_checker")
    graph.add_edge("responder", "fact_checker")
    graph.add_conditional_edges("fact_checker", _after_check,
                                {"review_writer": "review_writer", "responder": "responder", "end": END})
    return graph.compile()


AGENTS = [CoordinatorAgent, SpecRetrievalAgent, BenchmarkAnalystAgent, ReviewWriterAgent, AnswerAgent, FactCheckerAgent]

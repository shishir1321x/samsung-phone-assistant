"""Application service: conversational chatbot + review generation on top of the agent graph."""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

from app.agents.graph import build_graph
from app.db.repository import all_phones, latest_review, phone_by_name, phone_names, save_review
from app.db.session import session_scope
from app.llm.provider import get_llm
from app.rag.nlu import PhoneMatcher


class PhoneNotFound(LookupError):
    pass


@dataclass
class Conversation:
    history: list[dict] = field(default_factory=list)
    last_phones: list[str] = field(default_factory=list)
    last_aspects: list[str] = field(default_factory=list)
    updated: float = field(default_factory=time.time)


class PhoneAssistant:
    """Thread-safe facade used by the API and the CLI."""

    def __init__(self, llm="auto", max_sessions: int = 1000, history_turns: int = 6):
        self._llm = llm
        self.graph = build_graph()
        self.max_sessions = max_sessions
        self.history_turns = history_turns
        self._sessions: OrderedDict[str, Conversation] = OrderedDict()
        self._lock = threading.Lock()
        self._matcher: PhoneMatcher | None = None
        self._matcher_names: tuple[str, ...] = ()

    # ------------------------------------------------------------------ helpers
    @property
    def llm(self):
        return get_llm() if self._llm == "auto" else self._llm

    def matcher(self, db) -> PhoneMatcher:
        names = tuple(phone_names(db))
        with self._lock:
            if self._matcher is None or names != self._matcher_names:
                self._matcher = PhoneMatcher(list(names))
                self._matcher_names = names
            return self._matcher

    def _conversation(self, session_id: str) -> Conversation:
        with self._lock:
            convo = self._sessions.pop(session_id, None) or Conversation()
            self._sessions[session_id] = convo
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
            return convo

    def reset_session(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def resolve_phone(self, db, text: str) -> str:
        mentions = self.matcher(db).find(text)
        if not mentions:
            raise PhoneNotFound(f"No phone in the database matches {text!r}")
        return mentions[0].name

    def _run(self, db, state: dict) -> dict:
        config = {"configurable": {"session": db, "matcher": self.matcher(db), "llm": self.llm}}
        return self.graph.invoke({**state, "trace": []}, config=config)

    @staticmethod
    def _payload(result: dict, started: float) -> dict:
        plan = result["plan"]
        ctx = result.get("retrieved")
        return {
            "answer": result["output"],
            "intent": plan.intent,
            "phones": plan.phones,
            "aspects": plan.aspects,
            "generator": result.get("generator"),
            "verification": result.get("verification", {}),
            "sources": ctx.sources() if ctx else [],
            "rankings": [r.to_dict() for r in ctx.rankings] if ctx else [],
            "agents": result.get("trace", []),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    # ------------------------------------------------------------------ public API
    def ask(self, message: str, session_id: str | None = None) -> dict:
        started = time.perf_counter()
        session_id = session_id or uuid.uuid4().hex
        convo = self._conversation(session_id)
        with session_scope() as db:
            result = self._run(db, {
                "query": message,
                "mode": "chat",
                "history": list(convo.history),
                "context_phones": list(convo.last_phones),
                "context_aspects": list(convo.last_aspects),
            })
            plan = result["plan"]
            if plan.intent == "review" and plan.phones and result.get("analyses"):
                phone = phone_by_name(db, plan.phones[0])
                analysis = result["analyses"][phone.name]
                save_review(db, phone, {"content": result["output"], "scores": analysis.scores, "pros": analysis.pros,
                                        "cons": analysis.cons, "generator": result.get("generator"),
                                        "verification": result.get("verification", {})})
        with self._lock:
            convo.history.extend([{"role": "user", "content": message},
                                  {"role": "assistant", "content": result["output"]}])
            convo.history[:] = convo.history[-2 * self.history_turns:]
            if plan.phones:
                convo.last_phones = list(plan.phones)
            if plan.aspects:
                convo.last_aspects = list(plan.aspects)
            convo.updated = time.time()
        payload = self._payload(result, started)
        payload["session_id"] = session_id
        return payload

    def review(self, phone_text: str, regenerate: bool = False) -> dict:
        started = time.perf_counter()
        with session_scope() as db:
            name = self.resolve_phone(db, phone_text)
            phone = phone_by_name(db, name)
            cached = None if regenerate else latest_review(db, phone)
            if cached is not None:
                return {
                    "phone": name, "review": cached.content, "scores": cached.scores, "pros": cached.pros,
                    "cons": cached.cons, "generator": cached.generator, "verification": cached.verification,
                    "cached": True, "created_at": cached.created_at.isoformat() if cached.created_at else None,
                    "agents": [], "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                }
            result = self._run(db, {"query": f"Write a detailed review of the {name}", "mode": "review",
                                    "target_phone": name})
            analysis = result["analyses"][name]
            row = save_review(db, phone, {"content": result["output"], "scores": analysis.scores,
                                          "pros": analysis.pros, "cons": analysis.cons,
                                          "generator": result.get("generator"),
                                          "verification": result.get("verification", {})})
            return {
                "phone": name, "review": result["output"], "scores": analysis.scores, "overall": analysis.overall,
                "pros": analysis.pros, "cons": analysis.cons, "predecessor": analysis.predecessor,
                "vs_predecessor": analysis.vs_predecessor, "generator": result.get("generator"),
                "verification": result.get("verification", {}), "cached": False,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "agents": result.get("trace", []), "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            }

    def compare(self, phones: list[str], aspects: list[str] | None = None) -> dict:
        if len(phones) < 2:
            raise ValueError("Give at least two phones to compare")
        with session_scope() as db:
            names = [self.resolve_phone(db, p) for p in phones]
        if len(set(names)) < 2:
            raise ValueError("The phones given resolve to the same model")
        question = "Compare the " + " vs ".join(n.replace("Samsung ", "") for n in names)
        if aspects:
            question += " in terms of " + ", ".join(aspects)
        return self.ask(question)

    def phones(self) -> list:
        with session_scope() as db:
            return all_phones(db)

"""Command-line entry point.

    python -m app.cli setup                 # create tables, load data/phones.json, build the RAG index
    python -m app.cli scrape                # scrape GSMArena again (default 15 models) and store
    python -m app.cli scrape --models "Galaxy S24 FE" "Galaxy A56"
    python -m app.cli ask "Which Samsung phone has the best battery life?"
    python -m app.cli chat                  # interactive conversation
    python -m app.cli review "Galaxy S25 Ultra"
    python -m app.cli serve                 # start the FastAPI server
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.config import DEFAULT_MODELS, PHONES_JSON


def _utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:  # pragma: no cover - not a real console
            pass


def cmd_setup(args) -> None:
    from app.db.session import init_db, session_scope
    from app.rag.vector_store import build_index
    from app.scraper.pipeline import load_json, store_phones

    init_db()
    count = store_phones(load_json(args.json))
    print(f"Loaded {count} phones from {args.json}")
    with session_scope() as db:
        chunks = build_index(db)
    print(f"Built RAG index with {chunks} chunks")


def cmd_scrape(args) -> None:
    from app.db.session import session_scope
    from app.rag.vector_store import build_index
    from app.scraper.pipeline import export_json, scrape_phones, store_phones

    phones, missing = scrape_phones(args.models, max_pages=args.max_pages, delay=args.delay, refresh=args.refresh)
    if missing:
        print(f"Not found on GSMArena's Samsung listing: {', '.join(missing)}")
    store_phones(phones)
    path = export_json(phones, args.json)
    with session_scope() as db:
        chunks = build_index(db)
    print(f"Scraped and stored {len(phones)} phones; exported to {path}; RAG index has {chunks} chunks")


def cmd_index(_args) -> None:
    from app.db.session import session_scope
    from app.rag.vector_store import build_index

    with session_scope() as db:
        print(f"Built RAG index with {build_index(db)} chunks")


def _print_result(result: dict, verbose: bool) -> None:
    print(result["answer"])
    if verbose:
        print(f"\n[intent={result['intent']} generator={result['generator']} "
              f"verification={result['verification']} {result['elapsed_ms']} ms]")
        for step in result["agents"]:
            print(f"  - {step['agent']}: {step['action']} ({step['ms']} ms)")


def cmd_ask(args) -> None:
    from app.api.main import bootstrap
    from app.assistant import PhoneAssistant

    bootstrap()
    _print_result(PhoneAssistant().ask(" ".join(args.question)), args.verbose)


def cmd_chat(args) -> None:
    from app.api.main import bootstrap
    from app.assistant import PhoneAssistant

    bootstrap()
    assistant = PhoneAssistant()
    session_id = None
    print("Samsung phone assistant - type a question, or 'quit' to exit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        result = assistant.ask(question, session_id)
        session_id = result["session_id"]
        print("\nAssistant:", end=" ")
        _print_result(result, args.verbose)


def cmd_review(args) -> None:
    from app.api.main import bootstrap
    from app.assistant import PhoneAssistant

    bootstrap()
    result = PhoneAssistant().review(" ".join(args.phone), regenerate=True)
    print(result["review"])
    print(f"\n[generator={result['generator']} verification={result['verification']} {result['elapsed_ms']} ms]")
    for step in result["agents"]:
        print(f"  - {step['agent']}: {step['action']} ({step['ms']} ms)")


def cmd_serve(args) -> None:
    import uvicorn

    uvicorn.run("app.api.main:app", host=args.host, port=args.port, reload=args.reload)


def main(argv: list[str] | None = None) -> None:
    _utf8_console()
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="Samsung Phone Query & Review System")
    parser.add_argument("--log-level", default="WARNING")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup", help="create tables, load the bundled JSON dataset, build the RAG index")
    p.add_argument("--json", default=PHONES_JSON)
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("scrape", help="scrape GSMArena, store in the DB, export JSON, rebuild the index")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="model names as shown on GSMArena")
    p.add_argument("--max-pages", type=int, default=None, help="Samsung listing pages to scan")
    p.add_argument("--delay", type=float, default=None, help="seconds between requests (default 5)")
    p.add_argument("--refresh", action="store_true", help="ignore the HTML cache for spec pages")
    p.add_argument("--json", default=PHONES_JSON)
    p.set_defaults(func=cmd_scrape)

    p = sub.add_parser("index", help="rebuild the RAG embeddings")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("ask", help="ask one question")
    p.add_argument("question", nargs="+")
    p.add_argument("-v", "--verbose", action="store_true", help="show the agent trace")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="interactive chat")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("review", help="generate a review with the multi-agent system")
    p.add_argument("phone", nargs="+")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("serve", help="run the API server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()

"""Scrape -> parse -> store pipeline, plus JSON export/import of the scraped data."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.config import CACHE_DIR, DEFAULT_MODELS, PHONES_JSON, get_settings
from app.db.repository import upsert_phone
from app.db.session import init_db, session_scope
from app.scraper.gsmarena import GSMArenaClient, ScrapedPhone, discover_models, parse_phone_page

log = logging.getLogger(__name__)


def scrape_phones(
    models: list[str] | None = None,
    max_pages: int | None = None,
    delay: float | None = None,
    refresh: bool = False,
    client: GSMArenaClient | None = None,
) -> tuple[list[ScrapedPhone], list[str]]:
    """Scrape the given models from GSMArena. Returns (phones, models not found)."""
    settings = get_settings()
    models = models or DEFAULT_MODELS
    client = client or GSMArenaClient(
        cache_dir=CACHE_DIR,
        delay=settings.scrape_delay if delay is None else delay,
        user_agent=settings.user_agent,
    )
    urls, missing = discover_models(client, models, max_pages=max_pages or settings.scrape_max_pages)
    if missing:
        log.warning("Not found on the listing pages: %s", ", ".join(missing))

    phones: list[ScrapedPhone] = []
    for model in models:
        url = urls.get(model)
        if not url:
            continue
        html = client.get(url, use_cache=not refresh)
        phone = parse_phone_page(html, url)
        log.info("Parsed %s (%d spec rows)", phone.name, len(phone.specs))
        phones.append(phone)
    return phones, missing


def store_phones(phones: list[ScrapedPhone]) -> int:
    init_db()
    with session_scope() as session:
        for phone in phones:
            upsert_phone(session, phone)
    return len(phones)


def export_json(phones: list[ScrapedPhone], path: Path = PHONES_JSON) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "https://www.gsmarena.com",
        "count": len(phones),
        "phones": [p.to_dict() for p in phones],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_json(path: Path = PHONES_JSON) -> list[ScrapedPhone]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ScrapedPhone.from_dict(item) for item in payload["phones"]]

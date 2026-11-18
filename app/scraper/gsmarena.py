"""GSMArena scraper: polite HTTP client, listing/spec-page parsers and field derivation.

Discovery walks the public Samsung brand listing pages
(``samsung-phones-9.php`` -> ``samsung-phones-f-9-0-p2.php`` ...) and matches
model names there, so it never touches the site's search endpoint, which is
protected by a Cloudflare Turnstile challenge. If any page comes back as a
challenge the scraper stops instead of trying to get around it.
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from app.scraper import parsers as p

log = logging.getLogger(__name__)

BASE_URL = "https://www.gsmarena.com/"
SAMSUNG_LISTING = "samsung-phones-9.php"

# data-spec attributes of rows that continue the previous row without a title.
_OTHER_ROW_NAMES = {
    "bodyother": "Other",
    "memoryother": "Other",
    "featuresother": "Other",
    "optionalother": "Other",
    "displayother": "Other",
    "batdescription2": "Other",
}


class ScraperBlocked(RuntimeError):
    """Raised when GSMArena serves a bot challenge or keeps rate-limiting us."""


@dataclass
class SpecRow:
    category: str
    name: str
    value: str
    key: str | None = None


@dataclass
class ScrapedPhone:
    name: str
    url: str
    gsmarena_id: int | None
    image_url: str | None
    popularity_hits: int | None
    fans: int | None
    scraped_at: str
    specs: list[SpecRow] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ScrapedPhone":
        data = dict(data)
        data["specs"] = [SpecRow(**row) for row in data.get("specs", [])]
        return cls(**data)


@dataclass
class ListingEntry:
    name: str  # e.g. "Galaxy S23 Ultra" (as shown on the brand listing)
    url: str


# ====================================================================== client
class GSMArenaClient:
    """requests wrapper with a disk cache, polite delays and 429 back-off."""

    def __init__(
        self,
        cache_dir: Path,
        delay: float = 5.0,
        user_agent: str | None = None,
        max_retries: int = 3,
        session: requests.Session | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.max_retries = max_retries
        self.session = session or requests.Session()
        if user_agent:
            self.session.headers["User-Agent"] = user_agent
        self.session.headers.setdefault("Accept-Language", "en-US,en;q=0.9")
        self._last_request = 0.0
        self.network_requests = 0

    def _cache_path(self, url: str) -> Path:
        name = re.sub(r"[^A-Za-z0-9_.+-]", "_", url.replace(BASE_URL, ""))
        return self.cache_dir / name

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request
        target = self.delay + random.uniform(0, self.delay * 0.5)
        if self._last_request and elapsed < target:
            time.sleep(target - elapsed)

    def get(self, path_or_url: str, use_cache: bool = True) -> str:
        url = urljoin(BASE_URL, path_or_url)
        cache_file = self._cache_path(url)
        if use_cache and cache_file.exists():
            return cache_file.read_text(encoding="utf-8")

        for attempt in range(1, self.max_retries + 1):
            self._wait()
            log.info("GET %s (attempt %d)", url, attempt)
            response = self.session.get(url, timeout=30)
            self._last_request = time.monotonic()
            self.network_requests += 1

            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After", "")
                wait = float(retry_after) if retry_after.isdigit() else 30.0 * attempt
                log.warning("HTTP %s for %s; backing off %.0fs", response.status_code, url, wait)
                time.sleep(wait)
                continue
            response.raise_for_status()
            html = response.text
            if is_challenge_page(html):
                raise ScraperBlocked(
                    f"GSMArena returned a bot-verification page for {url}. "
                    "Stopping; open the site in a browser or retry later."
                )
            cache_file.write_text(html, encoding="utf-8")
            return html
        raise ScraperBlocked(f"Gave up on {url} after {self.max_retries} attempts (rate limited).")


def is_challenge_page(html: str) -> bool:
    head = html[:4000].lower()
    return "turnstile" in head or "cf-challenge" in head or "just a moment..." in head


# ====================================================================== parsing
def parse_listing(html: str) -> tuple[list[ListingEntry], str | None]:
    """Parse a brand listing page -> (entries, next page URL or None)."""
    soup = BeautifulSoup(html, "lxml")
    entries = []
    for anchor in soup.select("div.makers li a[href]"):
        name = anchor.get_text(" ", strip=True)
        if name:
            entries.append(ListingEntry(name=name, url=urljoin(BASE_URL, anchor["href"])))
    next_link = soup.select_one("div.nav-pages a.prevnextbutton[title='Next page']")
    next_url = urljoin(BASE_URL, next_link["href"]) if next_link and next_link.get("href") not in (None, "#") else None
    return entries, next_url


def _cell_text(cell) -> str:
    for br in cell.find_all("br"):
        br.replace_with("\n")
    return p.clean_text(cell.get_text(""))


def parse_phone_page(html: str, url: str) -> ScrapedPhone:
    soup = BeautifulSoup(html, "lxml")
    title = soup.select_one("h1.specs-phone-name-title")
    if title is None:
        raise ValueError(f"{url} does not look like a GSMArena spec page")
    name = p.clean_text(title.get_text(" ", strip=True))

    image = soup.select_one(".specs-photo-main img")
    hits = soup.select_one(".help-popularity .popularity-hits, .help-popularity span")
    fans = soup.select_one(".help-fans strong")
    id_match = re.search(r"-(\d+)\.php", url)

    specs: list[SpecRow] = []
    for table in soup.select("#specs-list table"):
        header = table.select_one("th")
        category = header.get_text(" ", strip=True) if header else "Other"
        if category == "Tests":
            category = "Our Tests"
        previous_name = ""
        for row in table.select("tr"):
            value_cell = row.select_one("td.nfo")
            if value_cell is None:
                continue
            title_cell = row.select_one("td.ttl")
            key = value_cell.get("data-spec")
            row_name = p.clean_text(title_cell.get_text(" ", strip=True)) if title_cell else ""
            if not row_name:
                row_name = _OTHER_ROW_NAMES.get(key or "", previous_name or "Other")
            value = _cell_text(value_cell)
            if not value:
                continue
            specs.append(SpecRow(category=category, name=row_name, value=value, key=key))
            previous_name = row_name

    return ScrapedPhone(
        name=name,
        url=url,
        gsmarena_id=int(id_match.group(1)) if id_match else None,
        image_url=image.get("src") if image else None,
        popularity_hits=p.parse_int(hits.get_text(" ", strip=True)) if hits else None,
        fans=p.parse_int(fans.get_text(" ", strip=True)) if fans else None,
        scraped_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        specs=specs,
    )


# ====================================================================== derivation
class SpecLookup:
    """Find spec values by GSMArena data-spec key or by (category, row name)."""

    def __init__(self, specs: list[SpecRow]):
        self.specs = specs

    def by_key(self, key: str) -> str | None:
        values = [s.value for s in self.specs if s.key == key]
        return "\n".join(values) if values else None

    def by_name(self, category: str, name: str) -> str | None:
        values = [
            s.value
            for s in self.specs
            if s.category.lower() == category.lower() and s.name.lower() == name.lower()
        ]
        return "\n".join(values) if values else None

    def category(self, category: str) -> list[SpecRow]:
        return [s for s in self.specs if s.category.lower() == category.lower()]

    def first_in_category(self, category: str) -> SpecRow | None:
        rows = self.category(category)
        return rows[0] if rows else None


def derive_fields(specs: list[SpecRow]) -> dict:
    """Typed, query-friendly columns for the ``phones`` table."""
    s = SpecLookup(specs)
    display_type = s.by_key("displaytype")
    width, height, ppi = p.parse_resolution(s.by_key("displayresolution"))
    ram_gb, storage_gb = p.parse_memory(s.by_key("internalmemory"))

    main_cam_row = s.first_in_category("Main Camera")
    main_cam = p.parse_camera_modules(s.by_key("cam1modules"))
    selfie = p.parse_camera_modules(s.by_key("cam2modules"))
    charging_text = s.by_name("Battery", "Charging")
    wired, wireless = p.parse_charging(charging_text)
    battery_text = s.by_key("batdescription1")
    dimensions = s.by_key("dimensions")
    status = s.by_key("status")
    body_text = "\n".join(filter(None, [s.by_key("bodyother"), s.by_key("build")]))
    prices = dict(p.parse_prices(s.by_key("price")))

    return {
        "announced": s.by_key("year"),
        "announced_date": p.parse_date(s.by_key("year")),
        "status": status,
        "release_date": p.parse_release_date(status),
        "dimensions": dimensions,
        "thickness_mm": p.parse_thickness(dimensions),
        "weight_g": p.parse_weight(s.by_key("weight")),
        "build": s.by_key("build"),
        "ip_rating": p.parse_ip_rating(body_text),
        "display_type": display_type,
        "display_size_in": p.parse_display_size(s.by_key("displaysize")),
        "display_resolution": s.by_key("displayresolution"),
        "resolution_width": width,
        "resolution_height": height,
        "ppi": ppi,
        "refresh_rate_hz": p.parse_refresh_rate(display_type),
        "peak_brightness_nits": p.parse_peak_brightness(display_type),
        "display_protection": s.by_key("displayprotection"),
        "os": s.by_key("os"),
        "chipset": s.by_key("chipset"),
        "cpu": s.by_key("cpu"),
        "gpu": s.by_key("gpu"),
        "card_slot": s.by_key("memoryslot"),
        "memory_options": s.by_key("internalmemory"),
        "ram_gb": ram_gb,
        "storage_gb_max": storage_gb,
        "main_camera_setup": main_cam_row.name if main_cam_row else None,
        "main_camera_mp": main_cam["max_mp"],
        "main_camera_count": main_cam["count"],
        "max_optical_zoom": main_cam["max_optical_zoom"],
        "main_camera_video": p.parse_top_video(s.by_key("cam1video")),
        "selfie_camera_mp": selfie["max_mp"],
        "selfie_camera_video": p.parse_top_video(s.by_key("cam2video")),
        "battery_mah": p.parse_battery_capacity(battery_text),
        "battery_type": p.parse_battery_type(battery_text),
        "charging": charging_text,
        "charging_wired_w": wired,
        "charging_wireless_w": wireless,
        "colors": s.by_key("colors"),
        "price_text": s.by_key("price"),
        "price_usd": prices.get("USD"),
    }


def derive_prices(specs: list[SpecRow]) -> list[dict]:
    raw = SpecLookup(specs).by_key("price")
    return [{"currency": c, "amount": a, "raw": raw} for c, a in p.parse_prices(raw)]


def derive_benchmarks(specs: list[SpecRow]) -> list[dict]:
    s = SpecLookup(specs)
    results = list(p.parse_performance_tests(s.by_key("tbench")))

    display_test = s.by_name("Our Tests", "Display")
    nits = p.parse_measured_brightness(display_test)
    if nits is not None:
        results.append({"test": "Display brightness", "version": "measured", "score": nits, "unit": "nits", "raw": display_test})

    speaker = s.by_name("Our Tests", "Loudspeaker")
    lufs = p.parse_loudness(speaker)
    if lufs is not None:
        results.append({"test": "Loudspeaker", "version": "LUFS", "score": lufs, "unit": "LUFS", "raw": speaker})

    battery_tests = "\n".join(filter(None, [s.by_key("batlife2"), s.by_key("batlife")]))
    active = p.parse_active_use_hours(battery_tests)
    if active is not None:
        results.append({"test": "Battery active use", "version": "2023+ test", "score": active, "unit": "hours", "raw": s.by_key("batlife2") or battery_tests})
    endurance = p.parse_endurance_hours(battery_tests)
    if endurance is not None:
        results.append({"test": "Battery endurance", "version": "legacy test", "score": endurance, "unit": "hours", "raw": s.by_key("batlife") or battery_tests})
    return results


# ====================================================================== discovery
def canonical_model(name: str) -> str:
    """'Samsung Galaxy S21 5G' / 'Galaxy S21' -> 'galaxy s21'."""
    text = name.lower().replace("samsung", " ")
    text = re.sub(r"\s+5g\b", "", text)
    return " ".join(text.split())


def discover_models(
    client: GSMArenaClient,
    targets: list[str],
    max_pages: int = 15,
) -> tuple[dict[str, str], list[str]]:
    """Map each target model name to its spec-page URL by walking the brand listing.

    Targets that already look like spec-page URLs/paths are used as-is.
    Returns (found mapping, list of targets that were not found).
    """
    found: dict[str, str] = {}
    pending: dict[str, str] = {}
    for target in targets:
        if target.endswith(".php"):
            found[target] = urljoin(BASE_URL, target)
        else:
            pending[canonical_model(target)] = target

    page_url: str | None = SAMSUNG_LISTING
    pages = 0
    while pending and page_url and pages < max_pages:
        html = client.get(page_url)
        pages += 1
        entries, page_url = parse_listing(html)
        for entry in entries:
            key = canonical_model(entry.name)
            if key in pending:
                found[pending.pop(key)] = entry.url
        log.info("Listing page %d scanned; %d targets still pending", pages, len(pending))
    return found, list(pending.values())

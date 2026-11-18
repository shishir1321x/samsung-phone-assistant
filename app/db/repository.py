"""Data-access helpers: persisting scraped phones and reading them back."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, selectinload

from app.db.models import Benchmark, Phone, Price, Review, Specification
from app.scraper.gsmarena import ScrapedPhone, derive_benchmarks, derive_fields, derive_prices
from app.scraper.parsers import hours_to_text


# ====================================================================== writes
def upsert_phone(session: Session, scraped: ScrapedPhone) -> Phone:
    """Insert or fully replace a phone and all its child rows."""
    existing = session.scalar(select(Phone).where(Phone.name == scraped.name))
    if existing is not None:
        session.delete(existing)
        session.flush()

    phone = Phone(
        name=scraped.name,
        gsmarena_id=scraped.gsmarena_id,
        url=scraped.url,
        image_url=scraped.image_url,
        popularity_hits=scraped.popularity_hits,
        fans=scraped.fans,
        scraped_at=datetime.fromisoformat(scraped.scraped_at) if scraped.scraped_at else None,
        **derive_fields(scraped.specs),
    )
    phone.specs = [
        Specification(category=row.category, name=row.name, spec_key=row.key, value=row.value, position=i)
        for i, row in enumerate(scraped.specs)
    ]
    seen_currencies: set[str] = set()
    for price in derive_prices(scraped.specs):
        if price["currency"] not in seen_currencies:
            seen_currencies.add(price["currency"])
            phone.prices.append(Price(**price))
    phone.benchmarks = [Benchmark(**b) for b in derive_benchmarks(scraped.specs)]
    session.add(phone)
    session.flush()
    return phone


def save_review(session: Session, phone: Phone, review: dict) -> Review:
    session.execute(delete(Review).where(Review.phone_id == phone.id))
    row = Review(
        phone_id=phone.id,
        content=review["content"],
        scores=review.get("scores", {}),
        pros=review.get("pros", []),
        cons=review.get("cons", []),
        generator=review.get("generator", "template"),
        verified=bool(review.get("verification", {}).get("passed", False)),
        verification=review.get("verification", {}),
    )
    session.add(row)
    session.flush()
    return row


# ====================================================================== reads
def all_phones(session: Session) -> list[Phone]:
    stmt = (
        select(Phone)
        .options(selectinload(Phone.specs), selectinload(Phone.prices), selectinload(Phone.benchmarks))
        .order_by(Phone.announced_date.desc().nulls_last(), Phone.name)
    )
    return list(session.scalars(stmt))


def phone_by_name(session: Session, name: str) -> Phone | None:
    stmt = (
        select(Phone)
        .where(Phone.name == name)
        .options(selectinload(Phone.specs), selectinload(Phone.prices), selectinload(Phone.benchmarks))
    )
    return session.scalar(stmt)


def phone_names(session: Session) -> list[str]:
    return list(session.scalars(select(Phone.name).order_by(Phone.name)))


def latest_review(session: Session, phone: Phone) -> Review | None:
    stmt = select(Review).where(Review.phone_id == phone.id).order_by(Review.created_at.desc())
    return session.scalars(stmt).first()


def benchmark(phone: Phone, test: str, version: str | None = None) -> Benchmark | None:
    for b in phone.benchmarks:
        if b.test.lower() == test.lower() and (version is None or b.version == version):
            return b
    return None


def dominant_version(phones: list[Phone], test: str) -> str | None:
    """Benchmark version covering most phones (AnTuTu v9 vs v10 are not comparable)."""
    counts: dict[str, int] = {}
    for phone in phones:
        for b in phone.benchmarks:
            if b.test.lower() == test.lower() and b.version:
                counts[b.version] = counts.get(b.version, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda v: (counts[v], _version_key(v)))


def _version_key(version: str) -> tuple:
    parts = []
    for token in version.lstrip("v").replace("-", ".").split("."):
        parts.append(int(token) if token.isdigit() else 0)
    return tuple(parts)


# ====================================================================== serialisation
KEY_FIELDS = [
    ("announced", "Announced"),
    ("status", "Status"),
    ("display_size_in", "Display size (in)"),
    ("display_type", "Display type"),
    ("display_resolution", "Resolution"),
    ("refresh_rate_hz", "Refresh rate (Hz)"),
    ("peak_brightness_nits", "Peak brightness (nits)"),
    ("chipset", "Chipset"),
    ("cpu", "CPU"),
    ("gpu", "GPU"),
    ("os", "OS"),
    ("memory_options", "Memory"),
    ("ram_gb", "Max RAM (GB)"),
    ("storage_gb_max", "Max storage (GB)"),
    ("main_camera_setup", "Main camera setup"),
    ("main_camera_mp", "Main camera (MP)"),
    ("max_optical_zoom", "Max optical zoom (x)"),
    ("main_camera_video", "Best video"),
    ("selfie_camera_mp", "Selfie camera (MP)"),
    ("battery_mah", "Battery (mAh)"),
    ("charging_wired_w", "Wired charging (W)"),
    ("charging_wireless_w", "Wireless charging (W)"),
    ("weight_g", "Weight (g)"),
    ("thickness_mm", "Thickness (mm)"),
    ("ip_rating", "IP rating"),
    ("price_usd", "Price (USD)"),
]


def specs_by_category(phone: Phone) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for spec in phone.specs:
        grouped.setdefault(spec.category, []).append({"name": spec.name, "value": spec.value})
    return grouped


def benchmark_dicts(phone: Phone) -> list[dict]:
    out = []
    for b in phone.benchmarks:
        display = hours_to_text(b.score) if b.unit == "hours" and b.test == "Battery active use" else None
        out.append(
            {
                "test": b.test,
                "version": b.version,
                "score": b.score,
                "unit": b.unit,
                "display": display or format_number(b.score, b.unit),
            }
        )
    return out


def phone_summary(phone: Phone) -> dict:
    return {
        "id": phone.id,
        "name": phone.name,
        "url": phone.url,
        "image_url": phone.image_url,
        "announced": phone.announced,
        "display_size_in": phone.display_size_in,
        "chipset": phone.chipset,
        "ram_gb": phone.ram_gb,
        "main_camera_mp": phone.main_camera_mp,
        "battery_mah": phone.battery_mah,
        "price_usd": phone.price_usd,
    }


def phone_detail(phone: Phone) -> dict:
    data = phone_summary(phone)
    data["key_specs"] = {label: getattr(phone, attr) for attr, label in KEY_FIELDS}
    data["prices"] = [{"currency": p.currency, "amount": p.amount} for p in phone.prices]
    data["benchmarks"] = benchmark_dicts(phone)
    data["specifications"] = specs_by_category(phone)
    data["scraped_at"] = phone.scraped_at.isoformat() if phone.scraped_at else None
    return data


def format_number(value: float | int | None, unit: str | None = None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = f"{value:,}" if isinstance(value, int) else f"{value:,.2f}".rstrip("0").rstrip(".")
    if unit in (None, "", "points"):
        return text
    if unit == "hours":
        return f"{text}h"
    return f"{text} {unit}"

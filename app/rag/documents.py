"""Turn DB rows into retrievable text chunks (one per phone x spec category)."""

from __future__ import annotations

from dataclasses import dataclass

from app.db.models import Phone
from app.db.repository import benchmark_dicts, format_number, specs_by_category

# Long, low-value rows that only add noise to retrieval
_SKIP_ROWS = {("Network", "2G bands"), ("Network", "3G bands"), ("Network", "4G bands"), ("Network", "5G bands"),
              ("Misc", "Models"), ("Misc", "SAR"), ("Misc", "SAR EU"), ("Our Tests", "Camera")}


@dataclass
class Document:
    phone_id: int
    phone_name: str
    category: str
    text: str


def overview_text(phone: Phone) -> str:
    parts = [f"{phone.name} overview."]
    if phone.announced:
        parts.append(f"Announced {phone.announced}.")
    if phone.display_size_in:
        parts.append(f"{phone.display_size_in:g}-inch {(phone.display_type or '').split(',')[0]} display"
                     + (f" at up to {phone.refresh_rate_hz}Hz" if phone.refresh_rate_hz else "") + ".")
    if phone.chipset:
        parts.append(f"Chipset: {phone.chipset}.")
    if phone.memory_options:
        parts.append(f"Memory: {phone.memory_options}.")
    if phone.main_camera_mp:
        parts.append(f"{phone.main_camera_setup or ''} main camera up to {phone.main_camera_mp:g} MP"
                     + (f" with {phone.max_optical_zoom:g}x optical zoom" if phone.max_optical_zoom else "")
                     + f"; {phone.selfie_camera_mp or 'n/a'} MP selfie camera.")
    if phone.battery_mah:
        parts.append(f"{phone.battery_mah} mAh battery"
                     + (f", {phone.charging_wired_w:g}W wired" if phone.charging_wired_w else "")
                     + (f", {phone.charging_wireless_w:g}W wireless charging" if phone.charging_wireless_w else "") + ".")
    if phone.weight_g:
        parts.append(f"Weighs {phone.weight_g:g} g" + (f", {phone.thickness_mm:g} mm thick" if phone.thickness_mm else "") + ".")
    if phone.ip_rating:
        parts.append(f"{phone.ip_rating} rated.")
    if phone.price_text:
        parts.append(f"Price: {phone.price_text}.")
    return " ".join(parts)


def build_documents(phone: Phone) -> list[Document]:
    docs = [Document(phone.id, phone.name, "Overview", overview_text(phone))]
    for category, rows in specs_by_category(phone).items():
        lines = [f"{r['name']}: {r['value']}".replace("\n", "; ") for r in rows if (category, r["name"]) not in _SKIP_ROWS]
        if not lines:
            continue
        docs.append(Document(phone.id, phone.name, category, f"{phone.name} - {category}. " + " | ".join(lines)))
    bench = benchmark_dicts(phone)
    if bench:
        lines = [f"{b['test']} ({b['version']}): {b['display']}" for b in bench]
        docs.append(Document(phone.id, phone.name, "Benchmarks",
                             f"{phone.name} - lab test results and benchmarks. " + " | ".join(lines)))
    if phone.prices:
        lines = [f"{p.currency} {format_number(p.amount)}" for p in phone.prices]
        docs.append(Document(phone.id, phone.name, "Price",
                             f"{phone.name} - price and cost. " + ", ".join(lines)))
    return docs

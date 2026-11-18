"""Pure functions that turn GSMArena spec strings into typed values.

Every function accepts ``None``/empty input and returns ``None`` (or an empty
list) rather than raising, because spec sheets differ between phone
generations and missing rows are normal.
"""

from __future__ import annotations

import re
from datetime import date

_MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ],
        start=1,
    )
}

CURRENCY_SYMBOLS = {
    "C$": "CAD",
    "A$": "AUD",
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "₹": "INR",
    "¥": "JPY",
}


def clean_text(text: str | None) -> str:
    """Normalise the odd whitespace characters GSMArena uses (thin/nbsp)."""
    if not text:
        return ""
    text = text.replace(" ", " ").replace("\xa0", " ").replace(" ", " ")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.replace(",", "").strip())
    except ValueError:
        return None


def _first(pattern: str, text: str | None, flags: int = re.I) -> float | None:
    if not text:
        return None
    match = re.search(pattern, text, flags)
    return to_float(match.group(1)) if match else None


def _all(pattern: str, text: str | None, flags: int = re.I) -> list[float]:
    if not text:
        return []
    return [v for v in (to_float(m) for m in re.findall(pattern, text, flags)) if v is not None]


# ---------------------------------------------------------------- display
def parse_display_size(text: str | None) -> float | None:
    return _first(r"(\d+(?:\.\d+)?)\s*inches", text)


def parse_resolution(text: str | None) -> tuple[int | None, int | None, int | None]:
    """'1080 x 2340 pixels, 19.5:9 ratio (~425 ppi density)' -> (1080, 2340, 425)."""
    if not text:
        return None, None, None
    width = height = ppi = None
    match = re.search(r"(\d{3,4})\s*x\s*(\d{3,4})\s*pixels", text)
    if match:
        width, height = int(match.group(1)), int(match.group(2))
    ppi_value = _first(r"~?(\d+)\s*ppi", text)
    if ppi_value is not None:
        ppi = int(ppi_value)
    return width, height, ppi


def parse_refresh_rate(text: str | None) -> int | None:
    values = _all(r"(\d{2,3})\s*Hz", text)
    return int(max(values)) if values else None


def parse_peak_brightness(text: str | None) -> int | None:
    values = _all(r"(\d{3,5})\s*nits", text)
    return int(max(values)) if values else None


# ---------------------------------------------------------------- memory
def parse_memory(text: str | None) -> tuple[float | None, int | None]:
    """'128GB 8GB RAM, 1TB 12GB RAM' -> (max RAM GB, max storage GB)."""
    if not text:
        return None, None
    rams: list[float] = []
    storages: list[int] = []
    for size, unit, ram in re.findall(r"(\d+)\s*(GB|TB)\s+(\d+(?:\.\d+)?)\s*GB\s+RAM", text, re.I):
        storages.append(int(size) * (1024 if unit.upper() == "TB" else 1))
        rams.append(float(ram))
    return (max(rams) if rams else None), (max(storages) if storages else None)


# ---------------------------------------------------------------- cameras
def parse_camera_modules(text: str | None) -> dict:
    """Summarise a camera block (one lens per line)."""
    lines = [ln for ln in clean_text(text).splitlines() if re.search(r"\d\s*MP", ln)]
    mps = [_first(r"(\d+(?:\.\d+)?)\s*MP", ln) for ln in lines]
    mps = [m for m in mps if m is not None]
    zooms = _all(r"(\d+(?:\.\d+)?)x\s+optical\s+zoom", text)
    return {
        "max_mp": max(mps) if mps else None,
        "count": len(lines) or None,
        "max_optical_zoom": max(zooms) if zooms else None,
    }


def parse_top_video(text: str | None) -> str | None:
    """'8K@24/30fps, 4K@30/60fps, ...' -> '8K@24/30fps'."""
    text = clean_text(text)
    if not text:
        return None
    return text.split(",")[0].strip() or None


# ---------------------------------------------------------------- battery
def parse_battery_capacity(text: str | None) -> int | None:
    value = _first(r"(\d{3,5})\s*mAh", text)
    return int(value) if value is not None else None


def parse_battery_type(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"(Li-\w+|Si/C Li-\w+)", text)
    return match.group(1) if match else None


def parse_charging(text: str | None) -> tuple[float | None, float | None]:
    """Return (wired W, wireless W); reverse wireless charging is ignored."""
    wired = wireless = None
    for line in clean_text(text).splitlines():
        low = line.lower()
        if wired is None:
            wired = _first(r"(\d+(?:\.\d+)?)\s*W\s+wired", line)
        if wireless is None and "reverse" not in low:
            wireless = _first(r"(\d+(?:\.\d+)?)\s*W\s+wireless", line)
    return wired, wireless


# ---------------------------------------------------------------- body
def parse_weight(text: str | None) -> float | None:
    return _first(r"(\d+(?:\.\d+)?)\s*g\b", text)


def parse_thickness(text: str | None) -> float | None:
    """Thickness in mm. For foldables the folded thickness is used."""
    if not text:
        return None
    dims = r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*mm"
    folded = re.search(r"\bFolded:\s*" + dims, text, re.I)  # \b so "Unfolded:" does not match
    match = folded or re.search(dims, text)
    return float(match.group(3)) if match else None


def parse_ip_rating(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"\b(IP\d{2}[A-Z]?)\b", text)
    return match.group(1) if match else None


# ---------------------------------------------------------------- launch
def parse_date(text: str | None) -> date | None:
    """'2023, February 01' / 'Released 2023, February 17' / '2023, February'."""
    if not text:
        return None
    match = re.search(r"(\d{4}),\s*([A-Za-z]+)(?:\s+(\d{1,2}))?", text)
    if not match:
        year = re.search(r"\b(20\d{2})\b", text)
        return date(int(year.group(1)), 1, 1) if year else None
    month = _MONTHS.get(match.group(2).lower())
    if month is None:
        return date(int(match.group(1)), 1, 1)
    return date(int(match.group(1)), month, int(match.group(3) or 1))


def parse_release_date(status: str | None) -> date | None:
    if not status:
        return None
    match = re.search(r"Released\s+(.*)", status, re.I)
    return parse_date(match.group(1)) if match else None


# ---------------------------------------------------------------- price
def parse_prices(text: str | None) -> list[tuple[str, float]]:
    """'$ 799.99 / € 649.00 / ₹ 64,999' or 'About 800 EUR' -> [(currency, amount)]."""
    text = clean_text(text)
    if not text:
        return []
    out: list[tuple[str, float]] = []
    symbols = "|".join(re.escape(s) for s in CURRENCY_SYMBOLS)
    for part in re.split(r"\s*/\s*", text):
        match = re.match(rf"^({symbols})\s*(\d[\d,]*(?:\.\d+)?)", part.strip())
        if match:
            amount = to_float(match.group(2))
            if amount is not None:
                out.append((CURRENCY_SYMBOLS[match.group(1)], amount))
            continue
        match = re.search(r"About\s+(\d[\d,]*(?:\.\d+)?)\s+([A-Z]{3})", part, re.I)
        if match:
            amount = to_float(match.group(1))
            if amount is not None:
                out.append((match.group(2).upper(), amount))
    return out


# ---------------------------------------------------------------- tests
def parse_performance_tests(text: str | None) -> list[dict]:
    """'AnTuTu: 1231075 (v9)\\nGeekBench: 4950 (v5.1)' -> list of benchmark dicts."""
    results: list[dict] = []
    for line in clean_text(text).splitlines():
        if ":" not in line:
            continue
        name, rest = line.split(":", 1)
        for match in re.finditer(r"(\d[\d,]*(?:\.\d+)?)\s*(fps)?\s*\(([^)]*)\)", rest):
            score = to_float(match.group(1))
            if score is None:
                continue
            results.append(
                {
                    "test": name.strip(),
                    "version": match.group(3).strip(),
                    "score": score,
                    "unit": "fps" if match.group(2) else "points",
                    "raw": line.strip(),
                }
            )
    return results


def parse_active_use_hours(text: str | None) -> float | None:
    """'Active use score 11:27h' -> 11.45."""
    if not text:
        return None
    match = re.search(r"Active use score\s*(\d+):(\d{2})\s*h", text, re.I)
    if not match:
        return None
    return round(int(match.group(1)) + int(match.group(2)) / 60, 2)


def parse_endurance_hours(text: str | None) -> float | None:
    return _first(r"Endurance rating\s*(\d+)\s*h", text)


def parse_measured_brightness(text: str | None) -> float | None:
    return _first(r"(\d+)\s*nits\s+max brightness", text)


def parse_loudness(text: str | None) -> float | None:
    return _first(r"(-?\d+(?:\.\d+)?)\s*LUFS", text)


def parse_int(text: str | None) -> int | None:
    value = _first(r"(\d[\d,]*)", text)
    return int(value) if value is not None else None


def hours_to_text(hours: float | None) -> str | None:
    if hours is None:
        return None
    whole = int(hours)
    minutes = round((hours - whole) * 60)
    if minutes == 60:
        whole, minutes = whole + 1, 0
    return f"{whole}:{minutes:02d}h"

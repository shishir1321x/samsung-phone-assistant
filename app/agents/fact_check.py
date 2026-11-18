"""Numeric grounding check: every figure in a generated text must exist in the source data.

LLMs (especially small open models) are prone to inventing plausible specs.
The fact-checker extracts numbers from the draft and verifies each one against
the numbers present in the retrieved context. It tolerates rounding (1.5%, or
5% for "1.2 million"-style figures) and ignores years and the integers 0-3,
which are mostly counts ("two cameras", "3 phones").
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(\s*(?:million|thousand)\b)?", re.I)
_TIME = re.compile(r"(?<![\w.])(\d{1,2}):(\d{2})\s*h", re.I)
_LIST_MARKER = re.compile(r"^\s*(?:#+\s*)?\d+[.)]\s", re.M)


@dataclass
class Verification:
    passed: bool
    checked: int
    unsupported: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def feedback(self) -> str:
        return ("These figures are not in the data: " + ", ".join(self.unsupported[:12])
                + ". Remove them or replace them with the exact values from the context.")


def extract_numbers(text: str) -> list[tuple[str, float, bool]]:
    """Return (raw text, value, approximate?) for every number in the text."""
    text = _LIST_MARKER.sub(" ", text)
    found: list[tuple[str, float, bool]] = []
    for match in _TIME.finditer(text):
        found.append((match.group(0), round(int(match.group(1)) + int(match.group(2)) / 60, 2), False))
    text = _TIME.sub(" ", text)
    for match in _NUMBER.finditer(text):
        whole = match.group(1).replace(",", "")
        value = float(f"{whole}.{match.group(2)}") if match.group(2) else float(whole)
        scale = (match.group(3) or "").strip().lower()
        if scale == "million":
            value *= 1_000_000
        elif scale == "thousand":
            value *= 1_000
        found.append((match.group(0).strip(), value, bool(scale)))
    return found


def _is_trivial(value: float) -> bool:
    if float(value).is_integer() and (value <= 3 or 1990 <= value <= 2035):
        return True
    return False


def _supported(value: float, approx: bool, sources: set[float]) -> bool:
    tolerance = 0.05 if approx else 0.015
    for source in sources:
        if source == value:
            return True
        if source and abs(value - source) <= tolerance * abs(source):
            return True
    return False


def verify(text: str, source_text: str) -> Verification:
    sources = {v for _, v, _ in extract_numbers(source_text)}
    # "11:27h" in the data may be written as "11 hours and 27 minutes"
    for match in _TIME.finditer(source_text):
        sources.update({float(match.group(1)), float(match.group(2))})
    unsupported: list[str] = []
    checked = 0
    for raw, value, approx in extract_numbers(text):
        if _is_trivial(value):
            continue
        checked += 1
        if not _supported(value, approx, sources) and raw not in unsupported:
            unsupported.append(raw)
    return Verification(passed=not unsupported, checked=checked, unsupported=unsupported)

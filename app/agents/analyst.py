"""Benchmark & market analysis used by the review pipeline.

Everything here is computed from the database, never guessed: category scores
are percentiles against the other phones in the dataset, pros/cons come from
those percentiles and from spec features, and the predecessor comparison uses
benchmarks run on the same test version only.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from app.analytics import (
    CATEGORY_WEIGHTS,
    METRICS,
    VERSIONED,
    category_score,
    cohort_version,
    common_benchmarks,
    position,
    relative_score,
)
from app.db.models import Phone
from app.db.repository import format_number
from app.scraper.parsers import hours_to_text

PRO_TEMPLATES = {
    "measured_brightness": "Very bright screen ({v} measured in GSMArena's lab)",
    "sharpness": "Sharp {v} display",
    "battery_life": "Excellent battery life (Active Use score {v})",
    "battery_endurance": "Strong battery endurance rating ({v})",
    "battery_capacity": "Big {v} battery",
    "performance": "Top-tier performance for its generation (AnTuTu {ver}: {v})",
    "zoom": "Long {v} optical zoom",
    "camera": "High-resolution {v} main camera",
    "charging": "Fast {v} wired charging",
    "weight": "Light for its class ({v})",
    "thickness": "Slim profile ({v})",
    "price": "Affordable at {v} on GSMArena",
    "loudness": "Loud stereo speakers ({v})",
}
CON_TEMPLATES = {
    "battery_life": "Weaker battery life than most of the lineup (Active Use score {v})",
    "battery_endurance": "Below-average battery endurance rating ({v})",
    "battery_capacity": "Smaller {v} battery",
    "charging": "Slow {v} wired charging",
    "performance": "Slower than most phones of its benchmark generation (AnTuTu {ver}: {v})",
    "weight": "Heavy ({v})",
    "thickness": "Thick ({v})",
    "price": "Expensive ({v} on GSMArena)",
    "measured_brightness": "Dimmer screen than most rivals ({v} measured)",
}


@dataclass
class Analysis:
    phone: str
    scores: dict[str, float]
    overall: float | None
    pros: list[str]
    cons: list[str]
    positions: dict[str, str]
    predecessor: str | None = None
    vs_predecessor: list[str] = field(default_factory=list)
    dataset_size: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def as_text(self) -> str:
        lines = [f"Analyst scores for {self.phone} (1-10, relative to the {self.dataset_size} Samsung phones in the database):"]
        lines += [f"- {k}: {v}" for k, v in self.scores.items()]
        if self.overall is not None:
            lines.append(f"- Overall: {self.overall}")
        lines.append("Rank positions within the database:")
        lines += [f"- {k}: {v}" for k, v in self.positions.items()]
        lines.append("Pros: " + "; ".join(self.pros))
        lines.append("Cons: " + "; ".join(self.cons))
        if self.predecessor:
            lines.append(f"Compared with its predecessor ({self.predecessor}):")
            lines += [f"- {line}" for line in self.vs_predecessor]
        return "\n".join(lines)


def _metric_value(phone: Phone, metric_key: str, phones: list[Phone]) -> tuple[str, str | None]:
    metric = METRICS[metric_key]
    version = cohort_version(phones, phone, metric_key) if metric_key in VERSIONED else None
    ctx = {f"version:{VERSIONED[metric_key]}": version} if version else {}
    return metric.fmt(metric.getter(phone, ctx)), version


def _to_ten(pct: float) -> float:
    return round(4 + 6 * pct, 1)


def category_scores(phones: list[Phone], phone: Phone) -> dict[str, float]:
    scores = {}
    for category in CATEGORY_WEIGHTS:
        score = category_score(phones, phone, category)
        if score is not None:
            scores[category] = score
    return scores


def _tier(name: str) -> tuple[str, int | None]:
    short = name.replace("Samsung Galaxy ", "").replace(" 5G", "")
    match = re.match(r"(S|A|Z Fold|Z Flip)\s?(\d+)\s*(.*)", short)
    if not match:
        return short, None
    return f"{match.group(1)}|{match.group(3).strip()}", int(match.group(2))


def find_predecessor(phones: list[Phone], phone: Phone) -> Phone | None:
    tier, gen = _tier(phone.name)
    if gen is None:
        return None
    candidates = [(p, _tier(p.name)) for p in phones if p is not phone]
    older = [(p, g) for p, (t, g) in candidates if t == tier and g is not None and g < gen]
    return max(older, key=lambda pg: pg[1])[0] if older else None


def _change(label: str, old, new, unit: str = "", better: str = "high") -> str | None:
    if old is None or new is None or old == new:
        return None
    direction = "up" if new > old else "down"
    improved = (new > old) == (better == "high")
    return f"{label} {direction} from {format_number(old)}{unit} to {format_number(new)}{unit}" + (
        " (improvement)" if improved else " (regression)")


def predecessor_changes(phone: Phone, prev: Phone) -> list[str]:
    lines = []
    if (phone.chipset or "") != (prev.chipset or ""):
        lines.append(f"Chipset: {(prev.chipset or 'n/a').splitlines()[0]} -> {(phone.chipset or 'n/a').splitlines()[0]}")
    for label, attr, unit, better in [
        ("Battery", "battery_mah", " mAh", "high"),
        ("Wired charging", "charging_wired_w", "W", "high"),
        ("Peak brightness", "peak_brightness_nits", " nits", "high"),
        ("Screen size", "display_size_in", "\"", "high"),
        ("Main camera", "main_camera_mp", " MP", "high"),
        ("Optical zoom", "max_optical_zoom", "x", "high"),
        ("Max RAM", "ram_gb", " GB", "high"),
        ("Weight", "weight_g", " g", "low"),
    ]:
        line = _change(label, getattr(prev, attr), getattr(phone, attr), unit, better)
        if line:
            lines.append(line)
    for delta in common_benchmarks([phone, prev]):
        if delta.test in ("AnTuTu", "GeekBench", "Battery active use", "Battery endurance", "Display brightness"):
            new, old = delta.values.get(phone.name), delta.values.get(prev.name)
            if new is None or old is None or old == 0:
                continue
            pct = round((new - old) / abs(old) * 100, 1)
            fmt = (lambda v: hours_to_text(v)) if delta.test == "Battery active use" else (
                lambda v: format_number(v, delta.unit if delta.unit not in ("points",) else None))
            lines.append(f"{delta.test} ({delta.version}): {fmt(old)} -> {fmt(new)} ({pct:+}%)")
    return lines


def _features(phone: Phone) -> tuple[list[str], list[str]]:
    pros, cons = [], []
    spec_text = "\n".join(f"{s.category}/{s.name}: {s.value}" for s in phone.specs).lower()
    if "stylus" in spec_text or "s pen" in spec_text:
        pros.append("Built-in S Pen stylus support")
    if phone.ip_rating and phone.ip_rating.startswith("IP68"):
        pros.append("IP68 dust and water resistance")
    elif phone.ip_rating:
        cons.append(f"Only {phone.ip_rating} protection (weaker dust resistance than IP68)")
    updates = re.search(r"up to (\d+) major os updates", spec_text)
    if updates and int(updates.group(1)) >= 5:
        pros.append(f"Long software support (up to {updates.group(1)} major OS updates)")
    if "titanium" in (phone.build or "").lower():
        pros.append("Titanium frame")
    if phone.charging_wireless_w is None:
        cons.append("No wireless charging")
    if phone.max_optical_zoom is None:
        cons.append("No optical-zoom telephoto camera")
    if (phone.main_camera_video or "").startswith("8K"):
        pros.append(f"Records up to {phone.main_camera_video} video")
    return pros, cons


def analyze(phones: list[Phone], phone: Phone) -> Analysis:
    scores = category_scores(phones, phone)
    overall_parts = [v for v in scores.values()]
    overall = round(sum(overall_parts) / len(overall_parts), 1) if overall_parts else None

    # Value: overall score per dollar, ranked against the dataset
    if phone.price_usd and overall is not None:
        ratios = []
        for other in phones:
            other_scores = category_scores(phones, other) if other is not phone else scores
            if other.price_usd and other_scores:
                ratios.append(sum(other_scores.values()) / len(other_scores) / other.price_usd)
        mine = overall / phone.price_usd
        if len(ratios) > 1:
            pct = sum(1 for r in ratios if r < mine) / (len(ratios) - 1)
            scores["Value"] = _to_ten(pct)

    pros, cons = [], []
    positions = {}
    for metric_key in ["measured_brightness", "sharpness", "performance", "geekbench", "battery_life",
                       "battery_endurance", "battery_capacity", "charging", "camera", "zoom", "weight",
                       "thickness", "price", "loudness"]:
        pct = relative_score(phones, phone, metric_key)
        if pct is None:
            continue
        value, version = _metric_value(phone, metric_key, phones)
        pos = position(phones, phone, metric_key)
        if pos:
            label = METRICS[metric_key].label + (f" {version}" if version else "")
            positions[label] = f"#{pos[0]} of {pos[1]} ({value})"
        if pct >= 0.75 and metric_key in PRO_TEMPLATES:
            pros.append(PRO_TEMPLATES[metric_key].format(v=value, ver=version or ""))
        elif pct <= 0.25 and metric_key in CON_TEMPLATES:
            cons.append(CON_TEMPLATES[metric_key].format(v=value, ver=version or ""))
    feature_pros, feature_cons = _features(phone)
    # keep a mix: the strongest measured advantages plus distinctive features
    pros = pros[:5] + feature_pros[:3]
    cons = cons[:4] + feature_cons[:3]
    if not cons:
        cons.append("No microSD slot or 3.5mm headphone jack" if (phone.card_slot or "").lower().startswith("no")
                    else "Nothing major stands out against the rest of the lineup")

    prev = find_predecessor(phones, phone)
    return Analysis(
        phone=phone.name,
        scores=scores,
        overall=overall,
        pros=pros,
        cons=cons,
        positions=positions,
        predecessor=prev.name if prev else None,
        vs_predecessor=predecessor_changes(phone, prev) if prev else [],
        dataset_size=len(phones),
    )

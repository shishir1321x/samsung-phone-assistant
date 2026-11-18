"""Metrics, rankings and dataset-relative scores computed from the structured data.

Used by the chatbot ("Which phone has the best battery life?") and by the
benchmark-analyst agent when it scores a phone for a review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from app.db.models import Phone
from app.db.repository import benchmark, dominant_version, format_number
from app.scraper.parsers import hours_to_text


@dataclass
class Metric:
    key: str
    label: str
    unit: str
    higher_is_better: bool
    getter: Callable[[Phone, dict], float | None]
    formatter: Callable[[float], str] | None = None
    note: str | None = None
    detail: Callable[[Phone], str] | None = None  # extra context shown next to a ranked value

    def fmt(self, value: float | None) -> str:
        if value is None:
            return "n/a"
        if self.formatter:
            return self.formatter(value)
        return format_number(value, self.unit)


def _bench(test: str) -> Callable[[Phone, dict], float | None]:
    def getter(phone: Phone, ctx: dict) -> float | None:
        b = benchmark(phone, test, ctx.get(f"version:{test}"))
        return b.score if b else None

    return getter


def _attr(name: str) -> Callable[[Phone, dict], float | None]:
    return lambda phone, _ctx: getattr(phone, name)


def _date_ordinal(phone: Phone, _ctx: dict) -> float | None:
    when = phone.release_date or phone.announced_date
    return float(when.toordinal()) if when else None


def _price_fmt(value: float) -> str:
    return f"${value:,.2f}"


# category -> [(metric, weight)]; scores are weighted percentiles against the dataset
CATEGORY_WEIGHTS: dict[str, list[tuple[str, float]]] = {
    "Display": [("measured_brightness", 0.35), ("sharpness", 0.25), ("brightness", 0.2), ("display_size", 0.2)],
    "Performance": [("performance", 0.6), ("geekbench", 0.25), ("ram", 0.15)],
    "Camera": [("zoom", 0.4), ("camera", 0.35), ("selfie", 0.25)],
    "Battery": [("battery_life", 0.45), ("battery_endurance", 0.45), ("battery_capacity", 0.3), ("charging", 0.25)],
    "Design": [("weight", 0.4), ("thickness", 0.3)],
}


def _category_getter(category: str) -> Callable[[Phone, dict], float | None]:
    def getter(phone: Phone, ctx: dict) -> float | None:
        phones = ctx.get("phones")
        if not phones:
            return None
        return category_score(phones, phone, category)

    return getter


def _overall_getter(phone: Phone, ctx: dict) -> float | None:
    phones = ctx.get("phones")
    if not phones:
        return None
    scores = [s for s in (category_score(phones, phone, c) for c in CATEGORY_WEIGHTS) if s is not None]
    return round(sum(scores) / len(scores), 1) if scores else None


def _camera_detail(phone: Phone) -> str:
    zoom = f", {phone.max_optical_zoom:g}x optical zoom" if phone.max_optical_zoom else ", no telephoto"
    return f"{phone.main_camera_setup or ''} {phone.main_camera_mp or 0:g} MP main{zoom}, {phone.selfie_camera_mp or 0:g} MP selfie".strip()


def _display_detail(phone: Phone) -> str:
    measured = benchmark(phone, "Display brightness")
    return (f'{phone.display_size_in or 0:g}", {phone.ppi or "?"} ppi, {phone.peak_brightness_nits or "?"} nits peak'
            + (f", {measured.score:g} nits measured" if measured else ""))


METRICS: dict[str, Metric] = {
    m.key: m
    for m in [
        Metric("battery_life", "battery life (GSMArena Active Use score)", "hours", True,
               _bench("Battery active use"), lambda v: hours_to_text(v) or "n/a",
               "Measured by GSMArena's 2023+ Active Use battery test; higher is better."),
        Metric("battery_endurance", "battery endurance (GSMArena legacy Endurance rating)", "hours", True,
               _bench("Battery endurance"), note="GSMArena's older endurance test; not comparable with the Active Use score."),
        Metric("battery_capacity", "battery capacity", "mAh", True, _attr("battery_mah")),
        Metric("performance", "performance (AnTuTu)", "points", True, _bench("AnTuTu"),
               note="AnTuTu scores are only compared within the same AnTuTu version."),
        Metric("geekbench", "CPU performance (GeekBench multi-core)", "points", True, _bench("GeekBench")),
        Metric("camera", "main camera resolution", "MP", True, _attr("main_camera_mp"),
               note="Megapixels are a rough proxy for camera quality; lens count and zoom matter too."),
        Metric("zoom", "optical zoom", "x", True, _attr("max_optical_zoom"), lambda v: f"{v:g}x"),
        Metric("selfie", "selfie camera resolution", "MP", True, _attr("selfie_camera_mp")),
        Metric("charging", "wired charging speed", "W", True, _attr("charging_wired_w")),
        Metric("wireless_charging", "wireless charging speed", "W", True, _attr("charging_wireless_w")),
        Metric("display_size", "screen size", "inches", True, _attr("display_size_in"), lambda v: f'{v:g}"'),
        Metric("brightness", "peak display brightness (spec)", "nits", True, _attr("peak_brightness_nits")),
        Metric("measured_brightness", "measured max brightness (GSMArena lab)", "nits", True,
               _bench("Display brightness")),
        Metric("loudness", "loudspeaker loudness (GSMArena lab)", "LUFS", True, _bench("Loudspeaker"),
               lambda v: f"{v:g} LUFS"),
        Metric("sharpness", "pixel density", "ppi", True, _attr("ppi")),
        Metric("refresh_rate", "display refresh rate", "Hz", True, _attr("refresh_rate_hz")),
        Metric("price", "price (USD, GSMArena)", "USD", False, _attr("price_usd"), _price_fmt,
               "Current prices as listed on GSMArena when the data was scraped."),
        Metric("weight", "weight", "g", False, _attr("weight_g")),
        Metric("thickness", "thickness", "mm", False, _attr("thickness_mm")),
        Metric("ram", "maximum RAM", "GB", True, _attr("ram_gb")),
        Metric("storage", "maximum storage", "GB", True, _attr("storage_gb_max")),
        Metric("newest", "release date", "", True, _date_ordinal,
               lambda v: date.fromordinal(int(v)).isoformat()),
        Metric("camera_score", "camera score", "/10", True, _category_getter("Camera"), lambda v: f"{v:g}/10",
               "Composite of main-camera resolution, optical zoom and selfie camera, relative to the other "
               "phones in the database (specs only; GSMArena does not publish image-quality scores).",
               _camera_detail),
        Metric("display_score", "display score", "/10", True, _category_getter("Display"), lambda v: f"{v:g}/10",
               "Composite of lab-measured brightness, peak brightness, pixel density and size, relative to the "
               "other phones in the database.", _display_detail),
        Metric("overall_score", "overall score", "/10", True, _overall_getter, lambda v: f"{v:g}/10",
               "Average of the display, performance, camera, battery and design scores, each relative to the "
               "other phones in the database."),
    ]
}


def category_score(phones: list[Phone], phone: Phone, category: str) -> float | None:
    """1-10 score for a category: weighted percentile within the dataset (4 = worst, 10 = best)."""
    total = weight_sum = 0.0
    for metric_key, weight in CATEGORY_WEIGHTS[category]:
        pct = relative_score(phones, phone, metric_key)
        if pct is None:
            continue
        total += pct * weight
        weight_sum += weight
    if weight_sum == 0:
        return None
    pct = total / weight_sum
    if category == "Design":  # water/dust resistance matters for durability
        ip = phone.ip_rating or ""
        pct = 0.75 * pct + 0.25 * (1.0 if ip.startswith("IP68") else 0.5 if ip.startswith("IP") else 0.0)
    return round(4 + 6 * pct, 1)

# Benchmark-backed metrics: GSMArena changed test versions over the years
# (AnTuTu v8 -> v11, GeekBench v5.1 -> v6, battery test redone in 2023), and
# scores are only comparable within one version.
VERSIONED = {"performance": "AnTuTu", "geekbench": "GeekBench"}
# When a ranking misses phones, also show these complementary rankings.
FALLBACKS = {"battery_life": ["battery_endurance", "battery_capacity"], "performance": [], "geekbench": []}


def metric_context(phones: list[Phone]) -> dict:
    """Pick the benchmark versions that make scores comparable across phones."""
    ctx: dict = {"phones": phones}
    for test in ("AnTuTu", "GeekBench"):
        version = dominant_version(phones, test)
        if version:
            ctx[f"version:{test}"] = version
    return ctx


@dataclass
class RankRow:
    rank: int
    name: str
    value: float
    display: str
    price_usd: float | None
    detail: str | None = None


@dataclass
class Ranking:
    metric: Metric
    rows: list[RankRow]
    missing: list[str] = field(default_factory=list)
    version: str | None = None
    ascending: bool = False
    max_price: float | None = None
    filtered_out: int = 0

    @property
    def label(self) -> str:
        label = self.metric.label
        if self.version and self.metric.key in ("performance", "geekbench"):
            label += f" {self.version}"
        return label

    def to_dict(self) -> dict:
        return {
            "metric": self.metric.key,
            "label": self.label,
            "unit": self.metric.unit,
            "order": "ascending" if self.ascending else "descending",
            "max_price_usd": self.max_price,
            "note": self.metric.note,
            "rows": [r.__dict__ for r in self.rows],
            "missing_data": self.missing,
        }


def rank_phones(
    phones: list[Phone],
    metric_key: str,
    ascending: bool | None = None,
    max_price: float | None = None,
    limit: int | None = None,
    version: str | None = None,
) -> Ranking:
    metric = METRICS[metric_key]
    ctx = metric_context(phones)
    if version and metric_key in VERSIONED:
        ctx[f"version:{VERSIONED[metric_key]}"] = version
    if ascending is None:
        ascending = not metric.higher_is_better
    pool = phones
    filtered_out = 0
    if max_price is not None:
        pool = [p for p in phones if p.price_usd is not None and p.price_usd <= max_price]
        filtered_out = len(phones) - len(pool)

    scored = []
    missing = []
    for phone in pool:
        value = metric.getter(phone, ctx)
        if value is None:
            missing.append(phone.name)
        else:
            scored.append((phone, value))
    scored.sort(key=lambda item: item[1], reverse=not ascending)

    rows = []
    rank = 0
    previous = None
    for i, (phone, value) in enumerate(scored[: limit or len(scored)], start=1):
        if value != previous:  # ties share a rank: 1, 1, 1, 4 ...
            rank, previous = i, value
        detail = metric.detail(phone) if metric.detail else None
        rows.append(RankRow(rank, phone.name, value, metric.fmt(value), phone.price_usd, detail))
    used_version = ctx.get(f"version:{VERSIONED[metric_key]}") if metric_key in VERSIONED else None
    return Ranking(metric, rows, missing, used_version, ascending, max_price, filtered_out)


def benchmark_versions(phones: list[Phone], test: str) -> list[str]:
    """Versions of a benchmark present in the data, newest first."""
    versions = {b.version for p in phones for b in p.benchmarks if b.test.lower() == test.lower() and b.version}
    return sorted(versions, key=_version_sort_key, reverse=True)


def _version_sort_key(version: str) -> tuple:
    return tuple(int(t) if t.isdigit() else 0 for t in version.lstrip("v").split("."))


def rankings_for(phones: list[Phone], metric_key: str, **kwargs) -> list[Ranking]:
    """All rankings needed to answer a 'which is best' question accurately.

    * versioned benchmarks -> one ranking per benchmark version (newest first)
    * metrics with partial coverage -> the primary ranking plus complementary ones
    """
    if metric_key in VERSIONED:
        rankings = []
        for version in benchmark_versions(phones, VERSIONED[metric_key]):
            ranking = rank_phones(phones, metric_key, version=version, **kwargs)
            if ranking.rows:
                rankings.append(ranking)
        return rankings or [rank_phones(phones, metric_key, **kwargs)]
    primary = rank_phones(phones, metric_key, **kwargs)
    rankings = [primary]
    if primary.missing:
        for fallback in FALLBACKS.get(metric_key, []):
            rankings.append(rank_phones(phones, fallback, **kwargs))
    return rankings


@dataclass
class BenchmarkDelta:
    test: str
    version: str | None
    unit: str
    values: dict[str, float]
    leader: str
    lead_pct: float


def common_benchmarks(phones: list[Phone], tests: tuple[str, ...] = ("AnTuTu", "GeekBench", "3DMark", "GFXBench",
                                                                        "Battery active use", "Battery endurance",
                                                                        "Display brightness", "Loudspeaker")) -> list[BenchmarkDelta]:
    """Benchmarks that ALL given phones share (same test and version), with the leader's margin."""
    deltas = []
    for test in tests:
        versions = None
        for phone in phones:
            mine = {b.version for b in phone.benchmarks if b.test == test}
            versions = mine if versions is None else versions & mine
        for version in sorted(versions or [], key=lambda v: _version_sort_key(v or ""), reverse=True):
            values = {}
            unit = "points"
            for phone in phones:
                b = benchmark(phone, test, version)
                if b:
                    values[phone.name] = b.score
                    unit = b.unit
            if len(values) < 2:
                continue
            # louder (less negative LUFS) is better for loudspeakers, higher is better elsewhere
            ordered = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
            leader, best = ordered[0]
            runner = ordered[1][1]
            lead_pct = round((best - runner) / abs(runner) * 100, 1) if runner else 0.0
            deltas.append(BenchmarkDelta(test, version, unit, values, leader, lead_pct))
    return deltas


def percentile(phones: list[Phone], phone: Phone, metric_key: str) -> float | None:
    """0..1 position of the phone among all phones for the metric (1 = best)."""
    metric = METRICS[metric_key]
    ctx = metric_context(phones)
    values = [v for v in (metric.getter(p, ctx) for p in phones) if v is not None]
    mine = metric.getter(phone, ctx)
    if mine is None or len(values) < 2:
        return None
    if metric.higher_is_better:
        worse = sum(1 for v in values if v < mine)
    else:
        worse = sum(1 for v in values if v > mine)
    ties = sum(1 for v in values if v == mine) - 1
    return (worse + ties / 2) / (len(values) - 1)


def cohort_version(phones: list[Phone], phone: Phone, metric_key: str) -> str | None:
    """For a versioned benchmark, the phone's version shared with the most other phones."""
    test = VERSIONED[metric_key]
    mine = [b.version for b in phone.benchmarks if b.test == test and b.version]
    if not mine:
        return None
    coverage = {v: sum(1 for p in phones if any(b.test == test and b.version == v for b in p.benchmarks)) for v in mine}
    return max(mine, key=lambda v: (coverage[v], _version_sort_key(v)))


def position(phones: list[Phone], phone: Phone, metric_key: str) -> tuple[int, int] | None:
    """(rank, out_of) of a phone for a metric, or None if it has no value.

    Versioned benchmarks are ranked within the phone's own benchmark cohort.
    """
    version = cohort_version(phones, phone, metric_key) if metric_key in VERSIONED else None
    ranking = rank_phones(phones, metric_key, version=version)
    for row in ranking.rows:
        if row.name == phone.name:
            return row.rank, len(ranking.rows)
    return None


def relative_score(phones: list[Phone], phone: Phone, metric_key: str) -> float | None:
    """Percentile (0..1, 1 = best) within the comparable cohort."""
    if metric_key in VERSIONED:
        version = cohort_version(phones, phone, metric_key)
        if version is None:
            return None
        test = VERSIONED[metric_key]
        cohort = [p for p in phones if any(b.test == test and b.version == version for b in p.benchmarks)]
        metric = METRICS[metric_key]
        ctx = {f"version:{test}": version}
        values = [metric.getter(p, ctx) for p in cohort]
        mine = metric.getter(phone, ctx)
        values = [v for v in values if v is not None]
        if mine is None or len(values) < 2:
            return None
        worse = sum(1 for v in values if v < mine)
        ties = sum(1 for v in values if v == mine) - 1
        return (worse + ties / 2) / (len(values) - 1)
    return percentile(phones, phone, metric_key)

"""Deterministic, fully grounded answers built straight from retrieved data.

They are (1) the answer when no LLM is configured, (2) the "grounded draft"
the LLM rewrites conversationally, and (3) the fallback when the fact-checker
rejects an LLM answer.
"""

from __future__ import annotations

from app.analytics import METRICS, Ranking
from app.db.models import Phone
from app.db.repository import benchmark_dicts, format_number
from app.rag.nlu import ASPECT_CATEGORIES
from app.rag.retriever import RetrievedContext, delta_text

# (attribute, label, better) -- better: "high", "low" or None (just informative)
FIELD_SPECS: dict[str, tuple[str, str | None]] = {
    "announced": ("Announced", None),
    "status": ("Status", None),
    "display_size_in": ("Screen size (in)", None),
    "display_type": ("Display", None),
    "display_resolution": ("Resolution", None),
    "refresh_rate_hz": ("Refresh rate (Hz)", "high"),
    "peak_brightness_nits": ("Peak brightness (nits)", "high"),
    "chipset": ("Chipset", None),
    "cpu": ("CPU", None),
    "gpu": ("GPU", None),
    "os": ("OS", None),
    "memory_options": ("Memory options", None),
    "ram_gb": ("Max RAM (GB)", "high"),
    "storage_gb_max": ("Max storage (GB)", "high"),
    "card_slot": ("Card slot", None),
    "main_camera_setup": ("Rear cameras", None),
    "main_camera_mp": ("Main camera (MP)", "high"),
    "max_optical_zoom": ("Optical zoom (x)", "high"),
    "main_camera_video": ("Best video", None),
    "selfie_camera_mp": ("Selfie camera (MP)", "high"),
    "selfie_camera_video": ("Selfie video", None),
    "battery_mah": ("Battery (mAh)", "high"),
    "charging_wired_w": ("Wired charging (W)", "high"),
    "charging_wireless_w": ("Wireless charging (W)", "high"),
    "dimensions": ("Dimensions", None),
    "weight_g": ("Weight (g)", "low"),
    "thickness_mm": ("Thickness (mm)", "low"),
    "build": ("Build", None),
    "ip_rating": ("IP rating", None),
    "colors": ("Colors", None),
    "price_usd": ("Price (USD, GSMArena)", "low"),
}

ASPECT_FIELDS: dict[str, list[str]] = {
    "camera": ["main_camera_setup", "main_camera_mp", "max_optical_zoom", "main_camera_video", "selfie_camera_mp"],
    "selfie": ["selfie_camera_mp", "selfie_camera_video"],
    "battery": ["battery_mah", "charging_wired_w", "charging_wireless_w"],
    "charging": ["charging_wired_w", "charging_wireless_w"],
    "performance": ["chipset", "cpu", "gpu", "ram_gb"],
    "display": ["display_size_in", "display_resolution", "refresh_rate_hz", "peak_brightness_nits"],
    "memory": ["memory_options", "ram_gb", "storage_gb_max", "card_slot"],
    "price": ["price_usd"],
    "design": ["dimensions", "weight_g", "thickness_mm", "build", "ip_rating"],
    "software": ["os"],
    "launch": ["announced", "status"],
    "colors": ["colors"],
}
OVERVIEW_FIELDS = ["announced", "display_size_in", "refresh_rate_hz", "chipset", "ram_gb", "main_camera_setup",
                   "main_camera_mp", "max_optical_zoom", "battery_mah", "charging_wired_w", "weight_g", "ip_rating",
                   "price_usd"]
BENCH_TESTS_FOR = {
    "performance": ("AnTuTu", "GeekBench", "3DMark", "GFXBench"),
    "battery": ("Battery active use", "Battery endurance"),
    "display": ("Display brightness",),
    "audio": ("Loudspeaker",),
}
PRICE_NOTE = "_Prices are the current listings on GSMArena at scrape time, not launch MSRPs._"


def short(name: str) -> str:
    return name.replace("Samsung ", "", 1)


def value_text(phone: Phone, attr: str) -> str:
    value = getattr(phone, attr)
    if value is None:
        return "n/a"
    if attr == "price_usd":
        return f"${value:,.2f}"
    if isinstance(value, float):
        return format_number(value)
    if isinstance(value, int):
        return f"{value:,}" if attr not in ("refresh_rate_hz",) else str(value)
    return str(value).replace("\n", "; ")


def _aspect_rows(phone: Phone, aspect: str) -> list[str]:
    cats = ASPECT_CATEGORIES.get(aspect, [])
    rows = []
    for spec in phone.specs:
        if spec.category in cats and spec.category != "Our Tests":
            if spec.category == "Network" and spec.name.endswith("bands"):
                continue
            if aspect == "price" and spec.name != "Price":
                continue
            if aspect == "colors" and spec.name != "Colors":
                continue
            if aspect == "charging" and spec.name != "Charging":
                continue
            rows.append(f"- **{spec.category} - {spec.name}:** {spec.value}".replace("\n", "; "))
    tests = BENCH_TESTS_FOR.get(aspect, ())
    for b in benchmark_dicts(phone):
        if b["test"] in tests:
            rows.append(f"- **Lab test - {b['test']} ({b['version']}):** {b['display']}")
    return rows


# ====================================================================== per-intent answers
def greeting_answer(phone_names: list[str]) -> str:
    return (
        "Hi! I can answer questions about these Samsung phones using specs scraped from GSMArena: "
        + ", ".join(short(n) for n in phone_names)
        + ".\n\nTry: *What are the camera specs of the Galaxy S23?*, *Which Samsung phone has the best "
        "battery life?*, *How does the S23 compare to the S22 in performance?* or *Review the Galaxy S25 Ultra*."
    )


def list_answer(phones: list[Phone]) -> str:
    lines = [f"I have detailed GSMArena data for **{len(phones)} Samsung phones**:"]
    for phone in sorted(phones, key=lambda p: (p.announced_date is None, p.announced_date), reverse=True):
        price = f", ${phone.price_usd:,.0f}" if phone.price_usd else ""
        year = phone.announced_date.year if phone.announced_date else "?"
        lines.append(f"- **{short(phone.name)}** ({year}{price}) - {phone.display_size_in:g}\" display, "
                     f"{phone.battery_mah} mAh, {(phone.chipset or '').splitlines()[0]}")
    return "\n".join(lines)


def unknown_answer(unknown: list[str], phone_names: list[str]) -> str:
    return (
        f"I don't have data for {', '.join(unknown)} in my database, so I won't guess its specs. "
        "I can answer questions about: " + ", ".join(short(n) for n in phone_names) + "."
    )


def spec_answer(ctx: RetrievedContext) -> str:
    plan = ctx.plan
    parts = []
    for phone in ctx.phones:
        if plan.aspects:
            lines = [f"**{phone.name}**"]
            for aspect in plan.aspects:
                rows = _aspect_rows(phone, aspect)
                if rows:
                    lines.append(f"\n*{aspect.capitalize()}*")
                    lines.extend(rows)
            if len(lines) == 1:  # aspects with no matching rows -> overview
                lines.extend(f"- **{FIELD_SPECS[a][0]}:** {value_text(phone, a)}" for a in OVERVIEW_FIELDS)
        else:
            lines = [f"**{phone.name}** - key specs:"]
            lines.extend(f"- **{FIELD_SPECS[a][0]}:** {value_text(phone, a)}" for a in OVERVIEW_FIELDS)
        if phone.name in ctx.positions and plan.metric:
            rank, total = ctx.positions[phone.name]
            lines.append(f"\nIt ranks **#{rank} of {total}** phones in my database for {METRICS[plan.metric].label}.")
        parts.append("\n".join(lines))
    text = "\n\n".join(parts)
    if "price" in plan.aspects or not plan.aspects:
        text += "\n\n" + PRICE_NOTE
    return text


def comparison_table(phones: list[Phone], fields: list[str]) -> str:
    header = "| Spec | " + " | ".join(short(p.name) for p in phones) + " |"
    sep = "|---" * (len(phones) + 1) + "|"
    rows = [header, sep]
    for attr in fields:
        label, better = FIELD_SPECS[attr]
        values = [getattr(p, attr) for p in phones]
        cells = [value_text(p, attr) for p in phones]
        numeric = [v for v in values if isinstance(v, (int, float))]
        if better and len(numeric) == len(values) and len(set(numeric)) > 1:
            target = max(numeric) if better == "high" else min(numeric)
            cells = [f"**{c}**" if v == target else c for c, v in zip(cells, values)]
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _winner_lines(phones: list[Phone], fields: list[str]) -> list[str]:
    lines = []
    for attr in fields:
        label, better = FIELD_SPECS[attr]
        values = [(p, getattr(p, attr)) for p in phones]
        if not better or any(v is None for _, v in values) or len({v for _, v in values}) == 1:
            continue
        best = max(values, key=lambda pv: pv[1]) if better == "high" else min(values, key=lambda pv: pv[1])
        others = [f"{short(p.name)} {value_text(p, attr)}" for p, _ in values if p is not best[0]]
        lines.append(f"- **{label}:** {short(best[0].name)} leads ({value_text(best[0], attr)} vs {', '.join(others)})")
    return lines


def compare_answer(ctx: RetrievedContext) -> str:
    plan = ctx.plan
    phones = ctx.phones
    fields: list[str] = []
    for aspect in plan.aspects:
        fields.extend(f for f in ASPECT_FIELDS.get(aspect, []) if f not in fields)
    if not fields:
        fields = OVERVIEW_FIELDS
    names = " vs ".join(short(p.name) for p in phones)
    aspect_label = ", ".join(plan.aspects) if plan.aspects else "overall"
    parts = [f"**{names} - {aspect_label} comparison**", "", comparison_table(phones, fields)]

    wanted_tests = {t for a in (plan.aspects or ["performance", "battery"]) for t in BENCH_TESTS_FOR.get(a, ())}
    deltas = [d for d in ctx.deltas if d.test in wanted_tests]
    if deltas:
        parts.append("\n**Lab results (same benchmark version for every phone):**")
        parts.extend(f"- {delta_text(d)}" for d in deltas)
    elif wanted_tests & {"AnTuTu", "GeekBench", "Battery active use"}:
        lines = []
        for phone in phones:
            scores = [f"{b['test']} {b['version']}: {b['display']}" for b in benchmark_dicts(phone) if b["test"] in wanted_tests]
            if scores:
                lines.append(f"- {short(phone.name)}: " + "; ".join(scores))
        if lines:
            parts.append("\n**Lab results** (different benchmark versions, so not directly comparable):")
            parts.extend(lines)

    winners = _winner_lines(phones, fields)
    if winners:
        parts.append("\n**Who leads where:**")
        parts.extend(winners)
    if "price_usd" in fields:
        parts.append("\n" + PRICE_NOTE)
    return "\n".join(parts)


def ranking_answer(ctx: RetrievedContext, top: int = 5) -> str:
    if not ctx.rankings or not any(r.rows for r in ctx.rankings):
        if ctx.plan.max_price:
            return f"None of the phones in my database are listed at or under ${ctx.plan.max_price:,.0f} on GSMArena."
        return "I couldn't rank the phones on that - the data isn't available."
    parts = []
    primary: Ranking = ctx.rankings[0]
    if primary.rows:
        winner = primary.rows[0]
        key = primary.metric.key
        special = {
            ("price", True): "is the cheapest",
            ("price", False): "is the most expensive",
            ("newest", False): "is the newest (released",
            ("newest", True): "is the oldest (released",
            ("weight", True): "is the lightest",
            ("weight", False): "is the heaviest",
            ("thickness", True): "is the thinnest",
            ("display_size", True): "has the smallest screen",
        }
        phrase = special.get((key, primary.ascending))
        if phrase and phrase.endswith("(released"):
            lead = f"**{short(winner.name)}** {phrase} {winner.display})"
        elif phrase:
            lead = f"**{short(winner.name)}** {phrase} ({winner.display})"
        else:
            label = primary.label.split(" (")[0]
            lead = f"**{short(winner.name)}** comes out on top for {label} with {winner.display}"
            ties = [short(r.name) for r in primary.rows[1:] if r.rank == winner.rank]
            if ties:
                lead += f" (tied with {', '.join(ties)})"
        parts.append(lead + ".")
    for i, ranking in enumerate(ctx.rankings):
        if not ranking.rows:
            continue
        title = f"**{ranking.label[0].upper() + ranking.label[1:]}**"
        if ranking.max_price:
            title += f" (phones priced <= ${ranking.max_price:,.0f})"
        lines = [f"\n{title}"]
        limit = top if i == 0 else 3
        for row in ranking.rows[:limit]:
            price = f" - ${row.price_usd:,.2f}" if row.price_usd and ranking.metric.key != "price" else ""
            detail = f" ({row.detail})" if row.detail else ""
            lines.append(f"{row.rank}. {short(row.name)}: {row.display}{detail}{price}")
        if len(ranking.rows) > limit:
            lines.append(f"_...{len(ranking.rows) - limit} more ranked._")
        if ranking.missing and i == 0 and len(ctx.rankings) > 1:
            lines.append(f"_Not covered by this test: {', '.join(short(m) for m in ranking.missing)}._")
        if ranking.metric.note and i == 0:
            lines.append(f"_{ranking.metric.note}_")
        parts.append("\n".join(lines))
    if primary.metric.key == "price" or primary.max_price:
        parts.append("\n" + PRICE_NOTE)
    return "\n".join(parts)


def general_answer(ctx: RetrievedContext) -> str:
    if ctx.evidence:
        lines = ["Here's what the spec sheets say:"]
        for name, rows in ctx.evidence.items():
            lines.append(f"- **{short(name)}** - {rows[0]}")
        missing = [short(p.name) for p in ctx.all_phones if p.name not in ctx.evidence]
        if missing and len(ctx.evidence) < len(ctx.all_phones):
            lines.append(f"\nNo matching spec entry for: {', '.join(missing)}.")
        return "\n".join(lines)
    if ctx.hits:
        lines = ["These spec entries look most relevant:"]
        for hit in ctx.hits[:4]:
            snippet = hit.text if len(hit.text) < 300 else hit.text[:297] + "..."
            lines.append(f"- **{short(hit.phone_name)}** ({hit.category}): {snippet}")
        return "\n".join(lines)
    return "I couldn't find anything about that in the phone database."


def compose_answer(ctx: RetrievedContext) -> str:
    intent = ctx.plan.intent
    if intent == "greeting":
        return greeting_answer([p.name for p in ctx.all_phones])
    if intent == "list":
        return list_answer(ctx.all_phones)
    if intent == "unknown_phone":
        return unknown_answer(ctx.plan.unknown_models, [p.name for p in ctx.all_phones])
    if intent == "rank":
        return ranking_answer(ctx)
    if intent == "compare" and len(ctx.phones) >= 2:
        return compare_answer(ctx)
    if ctx.phones:
        answer = spec_answer(ctx)
        if ctx.plan.unknown_models:
            answer += f"\n\n_Note: I have no data for {', '.join(ctx.plan.unknown_models)}._"
        return answer
    return general_answer(ctx)

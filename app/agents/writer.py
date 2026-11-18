"""Prompts for the LLM-backed agents and the deterministic review template."""

from __future__ import annotations

import re

from app.agents.analyst import Analysis
from app.db.models import Phone
from app.db.repository import benchmark_dicts, format_number

ANSWER_SYSTEM = (
    "You are a precise, friendly Samsung smartphone expert. Answer the user's question using ONLY the "
    "facts in the provided CONTEXT, which comes from a database of GSMArena specifications. Rules:\n"
    "- Copy every number exactly as it appears in the context; never invent or estimate specs.\n"
    "- If the context does not contain the answer, say so plainly.\n"
    "- Benchmark scores are only comparable within the same benchmark version.\n"
    "- Prices are current GSMArena listings, not launch prices.\n"
    "- Be concise: short paragraphs or bullet points, Markdown allowed, no preamble."
)

REVIEW_SYSTEM = (
    "You are an expert smartphone reviewer writing for a tech publication. Write an engaging, detailed and "
    "balanced review using ONLY the spec sheet and analyst findings provided. Copy every number exactly; do not "
    "add any figure, feature or opinion about hardware that is not supported by the data. Use Markdown."
)

REVIEW_SECTIONS = ("Verdict, Design & build, Display, Performance, Cameras, Battery & charging, Software, "
                   "Price & value, Pros, Cons, Bottom line")


def answer_prompt(question: str, context: str, draft: str, history: list[dict], feedback: str | None = None) -> str:
    convo = "\n".join(f"{turn['role'].upper()}: {turn['content'][:400]}" for turn in history[-4:])
    prompt = (
        (f"CONVERSATION SO FAR:\n{convo}\n\n" if convo else "")
        + f"QUESTION: {question}\n\n"
        f"CONTEXT (database facts):\n{context}\n\n"
        f"GROUNDED DRAFT ANSWER (correct but mechanical):\n{draft}\n\n"
        "Write the final answer to the QUESTION in a natural, helpful tone. Keep the draft's facts and numbers, "
        "add useful interpretation (what the numbers mean for a buyer), and stay under 250 words unless "
        "a comparison table is needed."
    )
    if feedback:
        prompt += f"\n\nA fact-checker rejected your previous answer: {feedback} Fix this."
    return prompt


def takeaway_prompt(question: str, context: str, draft: str, history: list[dict], feedback: str | None = None) -> str:
    """For comparisons and rankings: the LLM writes the verdict, the exact tables are appended after it."""
    convo = "\n".join(f"{turn['role'].upper()}: {turn['content'][:300]}" for turn in history[-4:])
    prompt = (
        (f"CONVERSATION SO FAR:\n{convo}\n\n" if convo else "")
        + f"QUESTION: {question}\n\n"
        f"CONTEXT (database facts):\n{context}\n\n"
        f"DATA TABLES THAT WILL BE SHOWN BELOW YOUR TEXT:\n{draft}\n\n"
        "Write a 2-4 sentence takeaway that answers the QUESTION directly: name the winner(s) and cite the "
        "one or two most important figures exactly as written above (for benchmarks, include the version). "
        "Do not repeat the tables and do not make claims the data does not support."
    )
    if feedback:
        prompt += f"\n\nA fact-checker rejected your previous text: {feedback} Fix this."
    return prompt


def _section(markdown: str, title: str) -> str | None:
    match = re.search(rf"^#+\s*{re.escape(title)}\b.*?(?=^#+\s|\Z)", markdown, re.I | re.M | re.S)
    return match.group(0).strip() if match else None


def complete_review(text: str, template: str) -> str:
    """Repair an LLM review that hit the token limit or skipped required sections."""
    lines = text.rstrip().splitlines()
    if lines and not re.search(r"[.!?)*|:]\s*$", lines[-1]):
        lines = lines[:-1]  # drop a sentence cut off mid-way
    text = "\n".join(lines).rstrip()
    for title in ("Pros", "Cons", "Bottom line"):
        if not re.search(rf"^#+\s*\**{re.escape(title)}", text, re.I | re.M):
            section = _section(template, title)
            if section:
                text += "\n\n" + section
    return text


def review_prompt(phone: Phone, fact_sheet: str, analysis: Analysis, feedback: str | None = None) -> str:
    prompt = (
        f"Write a detailed product review of the {phone.name} (450-650 words) with these Markdown sections: "
        f"{REVIEW_SECTIONS}. Start with a level-1 heading with the phone name.\n\n"
        f"SPEC SHEET (GSMArena):\n{fact_sheet}\n\n"
        f"ANALYST FINDINGS:\n{analysis.as_text()}\n\n"
        "Use the analyst's scores (out of 10, relative to the other Samsung phones in the database) and rank "
        "positions to judge strengths and weaknesses. Mention the predecessor comparison if present."
    )
    if feedback:
        prompt += f"\n\nA fact-checker rejected your previous draft: {feedback} Rewrite the review without them."
    return prompt


def _bench(phone: Phone, test: str) -> list[str]:
    return [f"{b['display']} ({b['version']})" for b in benchmark_dicts(phone) if b["test"] == test]


def _first_line(text: str | None) -> str:
    return (text or "n/a").splitlines()[0]


def template_review(phone: Phone, analysis: Analysis) -> str:
    """A complete, fully grounded review assembled from data (no LLM)."""
    short = phone.name.replace("Samsung ", "", 1)
    scores = analysis.scores
    ranked = sorted(((k, v) for k, v in scores.items() if k != "Value"), key=lambda kv: kv[1], reverse=True)
    best = ranked[0][0].lower() if ranked else "overall balance"
    worst = ranked[-1][0].lower() if len(ranked) > 1 else None
    pos = analysis.positions

    def where(label_prefix: str) -> str:
        for label, text in pos.items():
            if label.startswith(label_prefix):
                return f" - {text.split(' (')[0]} comparable phones in our database"
        return ""

    lines = [f"# {phone.name} review", ""]
    verdict = (f"**Verdict:** The {short} scores **{analysis.overall}/10** against the {analysis.dataset_size} "
               f"Samsung phones in our database. Its strongest area is {best}"
               + (f", and its weakest is {worst}." if worst else "."))
    lines += [verdict, "", "| Category | Score (1-10) |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in scores.items()]
    if analysis.overall is not None:
        lines.append(f"| **Overall** | **{analysis.overall}** |")

    lines += ["", "## Design & build",
              f"The {short} measures {phone.dimensions or 'n/a'} and weighs {format_number(phone.weight_g)} g"
              f"{where('weight')}. Build: {phone.build or 'n/a'}."
              + (f" It is {phone.ip_rating} rated." if phone.ip_rating else "")]

    brightness = _bench(phone, "Display brightness")
    lines += ["", "## Display",
              f"A {phone.display_size_in:g}-inch {phone.display_type or ''} panel with "
              f"{phone.display_resolution or 'n/a'}."
              + (f" GSMArena measured {brightness[0].split(' (')[0]} max brightness{where('measured max brightness')}."
                 if brightness else "")]

    antutu = _bench(phone, "AnTuTu")
    geek = _bench(phone, "GeekBench")
    perf = (f"It runs on the {_first_line(phone.chipset)} with {phone.gpu or 'n/a'} graphics and up to "
            f"{format_number(phone.ram_gb)} GB of RAM ({phone.memory_options or 'n/a'}).")
    if antutu:
        perf += f" AnTuTu: {', '.join(antutu)}{where('performance')}."
    if geek:
        perf += f" GeekBench: {', '.join(geek)}."
    lines += ["", "## Performance", perf]

    cam_rows = [s.value.replace("\n", "; ") for s in phone.specs if s.category == "Main Camera" and s.name not in ("Features", "Video")]
    lines += ["", "## Cameras",
              f"{phone.main_camera_setup or 'The'} rear camera setup: {'; '.join(cam_rows) or 'n/a'}. "
              f"Video tops out at {phone.main_camera_video or 'n/a'}; the selfie camera is "
              f"{format_number(phone.selfie_camera_mp)} MP."]

    active = _bench(phone, "Battery active use")
    endurance = _bench(phone, "Battery endurance")
    battery = (f"A {phone.battery_mah} mAh battery with {phone.charging or 'n/a'}".replace("\n", ", ") + ".")
    if active:
        battery += f" GSMArena's Active Use score is {active[0].split(' (')[0]}{where('battery life')}."
    if endurance:
        battery += f" Its legacy Endurance rating is {endurance[0].split(' (')[0]}."
    lines += ["", "## Battery & charging", battery]

    lines += ["", "## Software", f"{phone.os or 'n/a'}."]
    lines += ["", "## Price & value",
              f"GSMArena currently lists it at {phone.price_text or 'n/a'}."
              + (f" Value score: {scores['Value']}/10 (overall score per dollar vs. the rest of the lineup)."
                 if "Value" in scores else "")]

    if analysis.predecessor and analysis.vs_predecessor:
        lines += ["", f"## Versus the {analysis.predecessor.replace('Samsung ', '')}"]
        lines += [f"- {line}" for line in analysis.vs_predecessor]

    lines += ["", "## Pros"] + [f"- {p}" for p in analysis.pros]
    lines += ["", "## Cons"] + [f"- {c}" for c in analysis.cons]

    audience = {
        "display": "anyone who values a great screen",
        "performance": "power users and mobile gamers",
        "camera": "photography enthusiasts",
        "battery": "heavy users who want all-day endurance",
        "design": "people who want a light, compact phone",
    }.get(best, "most buyers")
    lines += ["", "## Bottom line",
              f"The {short} is best suited to {audience}. "
              + (f"Its main trade-off is {worst}." if worst else "")]
    return "\n".join(lines)

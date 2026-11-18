"""Query understanding: which phones a question is about, what it asks, and how.

Deterministic on purpose: entity resolution and intent routing must be exact
("S23" is not "S23 Ultra"), fast, and testable without an LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

VARIANT_WORDS = {"ultra", "plus", "fe", "edge", "lite", "pro", "mini", "max", "classic", "e", "neo"}

# aspect -> trigger words. Order matters only for display.
ASPECT_KEYWORDS: dict[str, list[str]] = {
    "camera": ["camera", "cameras", "photo", "photos", "photography", "picture", "pictures", "lens", "lenses",
               "megapixel", "megapixels", "mp", "zoom", "telephoto", "ultrawide", "video", "videos", "recording"],
    "selfie": ["selfie", "selfies", "front camera", "front-facing"],
    "battery": ["battery", "batteries", "mah", "endurance", "battery life", "screen on time", "lasts", "last longer"],
    "charging": ["charging", "charger", "charge", "wireless charging", "fast charging", "watt", "watts"],
    "performance": ["performance", "fast", "faster", "fastest", "speed", "speedy", "processor", "chip", "chipset",
                    "cpu", "gpu", "snapdragon", "exynos", "benchmark", "benchmarks", "antutu", "geekbench",
                    "gaming", "games", "powerful", "smooth"],
    "display": ["display", "screen", "screens", "amoled", "refresh rate", "hz", "brightness", "bright", "brightest",
                "nits", "resolution", "ppi", "inch", "inches"],
    "memory": ["ram", "memory", "storage", "gb", "tb", "microsd", "card slot", "ufs"],
    "price": ["price", "prices", "cost", "costs", "cheap", "cheaper", "cheapest", "expensive", "budget",
              "afford", "affordable", "value", "money", "usd", "$", "dollars", "euro", "eur", "pricing"],
    "design": ["design", "build", "weight", "weigh", "weighs", "heavy", "light", "lighter", "lightest",
               "thin", "thinner", "thinnest", "thickness", "dimensions", "size", "glass", "frame", "titanium",
               "waterproof", "water resistant", "water resistance", "ip68", "ip rating", "durable"],
    "software": ["software", "android", "one ui", "os", "updates", "update", "upgrades"],
    "connectivity": ["5g", "wifi", "wi-fi", "bluetooth", "nfc", "usb", "gps", "esim", "sim", "network", "bands"],
    "audio": ["audio", "speaker", "speakers", "loudspeaker", "headphone", "jack", "sound"],
    "features": ["sensor", "sensors", "fingerprint", "s pen", "stylus", "dex", "feature", "features", "ai"],
    "colors": ["color", "colors", "colour", "colours"],
    "launch": ["release", "released", "launch", "launched", "announced", "announcement", "when", "how old", "year"],
}

# aspect -> GSMArena spec categories that answer it
ASPECT_CATEGORIES: dict[str, list[str]] = {
    "camera": ["Main Camera", "Selfie camera"],
    "selfie": ["Selfie camera"],
    "battery": ["Battery", "Our Tests"],
    "charging": ["Battery"],
    "performance": ["Platform", "Memory", "Our Tests"],
    "display": ["Display", "Our Tests"],
    "memory": ["Memory"],
    "price": ["Misc"],
    "design": ["Body"],
    "software": ["Platform"],
    "connectivity": ["Network", "Comms"],
    "audio": ["Sound", "Our Tests"],
    "features": ["Features"],
    "colors": ["Misc"],
    "launch": ["Launch"],
}

# Superlative phrases -> (metric, ascending)
SUPERLATIVES: list[tuple[str, str, bool]] = [
    (r"\b(best|longest|great(est)?|top|strongest|biggest|largest|most)\b.{0,25}\bbattery\b|\bbattery\b.{0,25}\b(best|longest|last(s)? (the )?longest)\b", "battery_life", False),
    (r"\b(biggest|largest|highest|most)\b.{0,15}\b(battery capacity|mah)\b", "battery_capacity", False),
    (r"\b(fastest|quickest|best)\b.{0,15}\bwireless charg", "wireless_charging", False),
    (r"\b(fastest|quickest|best)\b.{0,15}\bcharg", "charging", False),
    (r"\bcharg\w*\b.{0,15}\b(fastest|quickest)\b", "charging", False),
    (r"\b(fastest|best|most powerful|top|strongest|highest)\b.{0,20}\b(performance|processor|chip|chipset|cpu|gpu|gaming|benchmark|antutu)\b|\b(performance|processor|chip|gaming)\b.{0,20}\b(best|fastest|top)\b|\bmost powerful\b|\bfastest\b", "performance", False),
    (r"\b(best|most|highest|biggest)\b.{0,15}\bzoom\b|\blongest zoom\b", "zoom", False),
    (r"\b(best|highest|most)\b.{0,20}\bselfie\b", "selfie", False),
    (r"\b(highest|most|biggest)\b.{0,15}\b(megapixels?|mp)\b", "camera", False),
    (r"\b(best|top|greatest|strongest)\b.{0,20}\b(camera|cameras|photo|photos|photography|pictures|camera phone)\b"
     r"|\bcamera\b.{0,15}\bbest\b", "camera_score", False),
    (r"\b(brightest)\b|\b(highest|most|best)\b.{0,15}\bbrightness\b", "measured_brightness", False),
    (r"\b(sharpest)\b|\b(highest|best)\b.{0,15}\b(resolution|ppi|pixel density)\b", "sharpness", False),
    (r"\b(biggest|largest|widest)\b.{0,15}\b(screen|display)\b", "display_size", False),
    (r"\b(smallest|most compact)\b.{0,15}\b(screen|display|phone)\b|\bmost compact\b", "display_size", True),
    (r"\b(best|top|greatest|nicest)\b.{0,15}\b(screen|display)\b", "display_score", False),
    (r"\b(cheapest|least expensive|lowest price|most affordable|best value|budget)\b", "price", True),
    (r"\b(most expensive|priciest|highest price|costliest)\b", "price", False),
    (r"\b(lightest|least heavy)\b", "weight", True),
    (r"\b(heaviest)\b", "weight", False),
    (r"\b(thinnest|slimmest)\b", "thickness", True),
    (r"\b(most|highest|biggest)\b.{0,10}\b(ram|memory)\b", "ram", False),
    (r"\b(most|highest|biggest|largest)\b.{0,10}\bstorage\b", "storage", False),
    (r"\b(highest|best|fastest|smoothest)\b.{0,15}\brefresh\b", "refresh_rate", False),
    (r"\b(newest|latest|most recent)\b", "newest", False),
    (r"\b(oldest)\b", "newest", True),
    # generic "best phone" comes last so aspect-specific superlatives win
    (r"\b(best|top|greatest)\b\s+(overall\s+)?(samsung\s+|galaxy\s+)?(phone|smartphone|model|device|one|pick|choice)s?\b"
     r"|\bbest overall\b", "overall_score", False),
]

COMPARE_WORDS = re.compile(
    r"\b(compare|compared|comparison|comparing|vs|versus|differ|difference|differences|better|worse|"
    r"against|or the|upgrade from|upgrade over|between)\b"
)
REVIEW_WORDS = re.compile(
    r"\b(review|reviews|pros and cons|pros & cons|worth (it|buying)|should i (buy|get)|verdict|"
    r"recommend|opinion|evaluate|assessment)\b"
)
LIST_WORDS = re.compile(
    r"\b(list|which|what)\b.{0,20}\b(phones|models|devices)\b.{0,30}\b(do you (have|know|cover|support)|"
    r"available|in (the|your) (database|db)|are there|you have)\b|\b(list|show)( me)? (all )?(the )?(phones|models)\b"
)
PRONOUNS = re.compile(r"\b(it|its|it's|this phone|that phone|this one|that one|them|they|both|these|those|either|each)\b")
GREETING = re.compile(r"^\s*(hi|hello|hey|yo|good (morning|afternoon|evening)|thanks|thank you|help|what can you do)\b[\s!.?]*$")
PRICE_LIMIT = re.compile(
    r"\b(under|below|less than|cheaper than|within|up to|max(imum)?|no more than|at most)\s*"
    r"(\$|usd\s*)?\s*(\d[\d,]*)\s*(\$|usd|dollars|bucks)?"
)
MODEL_MENTION = re.compile(
    r"\b(?:galaxy\s+)?((?:z\s+)?(?:fold|flip)\s*\d{1,2}|s\s*\d{1,2}|a\s*\d{2}|m\s*\d{2}|f\s*\d{2}|note\s*\d{1,2})"
    r"((?:\s+(?:ultra|plus|fe|edge|lite|pro))?)\b"
)


def canon(text: str) -> str:
    """Canonical form used for alias matching: 'Galaxy S23+' -> 'galaxy s 23 plus'."""
    t = text.lower().replace("+", " plus ")
    t = re.sub(r"[^a-z0-9$ ]", " ", t)
    t = re.sub(r"([a-z])(\d)", r"\1 \2", t)
    t = re.sub(r"(\d)([a-z])", r"\1 \2", t)
    return " ".join(t.split())


@dataclass
class PhoneMention:
    name: str
    start: int
    end: int


class PhoneMatcher:
    """Resolves phone mentions ('S23', 'galaxy s22 ultra', 'Fold 7') to DB names."""

    def __init__(self, names: list[str]):
        self.names = list(names)
        alias_owner: dict[str, set[str]] = {}
        for name in self.names:
            for alias in self._aliases(name):
                alias_owner.setdefault(alias, set()).add(name)
        self.aliases = {alias: next(iter(owners)) for alias, owners in alias_owner.items() if len(owners) == 1}
        self._patterns = sorted(
            ((alias, re.compile(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])")) for alias in self.aliases),
            key=lambda item: len(item[0]),
            reverse=True,
        )

    @staticmethod
    def _aliases(name: str) -> set[str]:
        full = canon(name)
        forms = {full}
        if full.startswith("samsung "):
            forms.add(full[len("samsung "):])
        for form in list(forms):
            if form.startswith("galaxy "):
                forms.add(form[len("galaxy "):])
        for form in list(forms):
            if form.endswith(" 5 g"):
                forms.add(form[: -len(" 5 g")])
        for form in list(forms):  # 'z fold 7' -> also 'fold 7'
            if form.startswith("z ") and ("fold" in form or "flip" in form):
                forms.add(form[2:])
        # a bare 'galaxy' or a one-token alias like 's' is useless
        return {f for f in forms if len(f) > 2 and f not in {"samsung", "galaxy"}}

    def find(self, text: str) -> list[PhoneMention]:
        query = canon(text)
        taken: list[tuple[int, int]] = []
        found: list[PhoneMention] = []
        for alias, pattern in self._patterns:
            for match in pattern.finditer(query):
                start, end = match.span()
                if any(start < e and end > s for s, e in taken):
                    continue
                # 'S23 plus' must not silently resolve to the S23
                rest = query[end:].split()
                if rest and rest[0] in VARIANT_WORDS:
                    continue
                taken.append((start, end))
                found.append(PhoneMention(self.aliases[alias], start, end))
        found.sort(key=lambda m: m.start)
        unique: list[PhoneMention] = []
        for mention in found:
            if mention.name not in {m.name for m in unique}:
                unique.append(mention)
        return unique

    def unknown_models(self, text: str, mentions: list[PhoneMention]) -> list[str]:
        """Model-like mentions ('Galaxy S20', 'S23 FE') that are not in the database."""
        query = canon(text)
        unknown = []
        for match in MODEL_MENTION.finditer(query):
            start, end = match.span()
            if any(m.start <= start and end <= m.end for m in mentions):
                continue
            if any(start < m.end and end > m.start for m in mentions) and not match.group(2):
                continue
            label = "Galaxy " + re.sub(r"\b([a-z])\s+(\d)", r"\1\2", (match.group(1) + match.group(2)).strip()).upper()
            label = label.replace(" PLUS", "+").replace("FOLD ", "Fold").replace("FLIP ", "Flip")
            if label not in unknown:
                unknown.append(label)
        return unknown


@dataclass
class QueryPlan:
    query: str
    intent: str  # greeting | list | review | compare | rank | spec | general | unknown_phone
    phones: list[str] = field(default_factory=list)
    aspects: list[str] = field(default_factory=list)
    metric: str | None = None
    ascending: bool = False
    max_price: float | None = None
    unknown_models: list[str] = field(default_factory=list)
    used_context: bool = False

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "intent": self.intent,
            "phones": self.phones,
            "aspects": self.aspects,
            "metric": self.metric,
            "ascending": self.ascending,
            "max_price": self.max_price,
            "unknown_models": self.unknown_models,
            "used_context": self.used_context,
        }


def detect_aspects(text: str) -> list[str]:
    low = " " + text.lower() + " "
    found = []
    for aspect, words in ASPECT_KEYWORDS.items():
        for word in words:
            if word == "$":
                hit = "$" in low
            else:
                hit = re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", low) is not None
            if hit:
                found.append(aspect)
                break
    # 'screen size' is a display question, not a design one
    if "display" in found and "design" in found and re.search(r"\b(screen|display) size\b", low):
        found.remove("design")
    return found


def detect_superlative(text: str) -> tuple[str | None, bool]:
    low = text.lower()
    for pattern, metric, ascending in SUPERLATIVES:
        if re.search(pattern, low):
            return metric, ascending
    return None, False


def detect_price_limit(text: str) -> float | None:
    match = PRICE_LIMIT.search(text.lower())
    if not match:
        return None
    value = float(match.group(4).replace(",", ""))
    return value if value >= 50 else None


FOLLOW_UP = re.compile(r"^\s*(what|how)\s+about\b|^\s*and\b|^\s*same\b")
# "Which phones have a stylus?" is about the whole catalogue, not the last phone discussed
ACROSS_PHONES = re.compile(r"\b(which|what|any|all|every)\s+(of\s+the\s+)?(samsung\s+|galaxy\s+)?(phones?|models?|devices?)\b")


def plan_query(
    text: str,
    matcher: PhoneMatcher,
    context_phones: list[str] | None = None,
    context_aspects: list[str] | None = None,
) -> QueryPlan:
    """Turn a user message into a QueryPlan.

    ``context_phones``/``context_aspects`` come from the previous turn and let
    follow-ups like "what about its battery?" or "and the S22?" resolve.
    """
    context_phones = context_phones or []
    low = text.lower().strip()
    mentions = matcher.find(text)
    phones = [m.name for m in mentions]
    unknown = matcher.unknown_models(text, mentions)
    aspects = detect_aspects(text)
    metric, ascending = detect_superlative(text)
    max_price = detect_price_limit(text)
    plan = QueryPlan(query=text, intent="general", phones=phones, aspects=aspects,
                     metric=metric, ascending=ascending, max_price=max_price, unknown_models=unknown)

    if GREETING.match(low):
        plan.intent = "greeting"
        return plan
    if LIST_WORDS.search(low) and not phones and not metric:
        plan.intent = "list"
        return plan

    refers_back = bool(PRONOUNS.search(low))
    is_follow_up = bool(FOLLOW_UP.search(low))
    asks_across = bool(ACROSS_PHONES.search(low)) and not refers_back
    if (not phones and context_phones and not unknown and not asks_across
            and (refers_back or is_follow_up or (aspects and not metric and not max_price))):
        # "What about its battery?" -> same phone(s) as the previous turn
        plan.phones = phones = list(context_phones)
        plan.used_context = True
    elif len(phones) == 1 and context_phones and COMPARE_WORDS.search(low) and (refers_back or is_follow_up):
        # "How does it compare to the S22?" -> previous phone + S22
        previous = [p for p in context_phones if p not in phones]
        if previous:
            plan.phones = phones = previous[:1] + phones
            plan.used_context = True
    if not aspects and context_aspects and is_follow_up:
        # "What are the camera specs of the S23?" ... "What about the S22?"
        plan.aspects = aspects = list(context_aspects)
        plan.used_context = True

    if unknown and not phones:
        plan.intent = "unknown_phone"
    elif REVIEW_WORDS.search(low) and len(phones) == 1:
        plan.intent = "review"
    elif len(phones) >= 2:
        plan.intent = "compare"
    elif len(phones) == 1:
        # A superlative about one phone ("Is the S23 the lightest?") is still a
        # question about that phone; the metric is kept so the answer can say
        # where it ranks.
        plan.intent = "spec"
    elif metric:
        plan.intent = "rank"
    elif max_price is not None:
        plan.intent = "rank"
        plan.metric = "price"
        plan.ascending = True
    else:
        plan.intent = "general"
    return plan

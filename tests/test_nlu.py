import pytest

from app.rag.nlu import PhoneMatcher, detect_aspects, plan_query

NAMES = [
    "Samsung Galaxy S21 5G", "Samsung Galaxy S21 Ultra 5G", "Samsung Galaxy S22 5G", "Samsung Galaxy S22 Ultra 5G",
    "Samsung Galaxy S23", "Samsung Galaxy S23 Ultra", "Samsung Galaxy S24", "Samsung Galaxy S24 Ultra",
    "Samsung Galaxy S25", "Samsung Galaxy S25 Ultra", "Samsung Galaxy Z Fold7", "Samsung Galaxy Z Flip7",
    "Samsung Galaxy A57",
]


@pytest.fixture(scope="module")
def matcher():
    return PhoneMatcher(NAMES)


@pytest.mark.parametrize("text, expected", [
    ("What are the camera specs of the Samsung Galaxy S23?", ["Samsung Galaxy S23"]),
    ("s23 ultra battery", ["Samsung Galaxy S23 Ultra"]),
    ("Galaxy S23Ultra vs S23", ["Samsung Galaxy S23 Ultra", "Samsung Galaxy S23"]),
    ("How does the Galaxy S23 compare to the S22?", ["Samsung Galaxy S23", "Samsung Galaxy S22 5G"]),
    ("Is the S21 5G still good?", ["Samsung Galaxy S21 5G"]),
    ("fold 7 or flip7", ["Samsung Galaxy Z Fold7", "Samsung Galaxy Z Flip7"]),
    ("galaxy a57 price", ["Samsung Galaxy A57"]),
])
def test_phone_resolution(matcher, text, expected):
    assert [m.name for m in matcher.find(text)] == expected


def test_variant_not_in_db_is_not_silently_mapped(matcher):
    mentions = matcher.find("Tell me about the S23 Plus")
    assert mentions == []
    assert matcher.unknown_models("Tell me about the S23 Plus", mentions) == ["Galaxy S23+"]
    assert matcher.unknown_models("Galaxy S20 camera", []) == ["Galaxy S20"]


def test_aspects():
    assert detect_aspects("What are the camera specs?") == ["camera"]
    assert detect_aspects("What is the screen size of the S22?") == ["display"]
    assert set(detect_aspects("battery life and charging speed")) >= {"battery", "charging"}


@pytest.mark.parametrize("text, intent, metric", [
    ("Which Samsung phone has the best battery life?", "rank", "battery_life"),
    ("What's the cheapest phone?", "rank", "price"),
    ("Which is the fastest Samsung phone?", "rank", "performance"),
    ("Best camera phone under $600", "rank", "camera_score"),
    ("What's the best Samsung phone overall?", "rank", "overall_score"),
    ("How does the Galaxy S23 compare to the S22 in terms of performance?", "compare", None),
    ("Review the Galaxy S24 Ultra", "review", None),
    ("What is the screen size of the Galaxy S22?", "spec", None),
    ("Which phones do you have?", "list", None),
    ("hello", "greeting", None),
    ("Specs of the Galaxy S20?", "unknown_phone", None),
    ("Which phones have a stylus?", "general", None),
])
def test_intents(matcher, text, intent, metric):
    plan = plan_query(text, matcher)
    assert plan.intent == intent
    if metric:
        assert plan.metric == metric


def test_price_filter(matcher):
    plan = plan_query("Which phone has the best camera under $500?", matcher)
    assert plan.max_price == 500 and plan.metric == "camera_score"


def test_follow_up_uses_conversation_context(matcher):
    plan = plan_query("What about its battery?", matcher, context_phones=["Samsung Galaxy S23"])
    assert plan.phones == ["Samsung Galaxy S23"] and plan.used_context and plan.aspects == ["battery"]

    plan = plan_query("What about the S22?", matcher, context_phones=["Samsung Galaxy S23"], context_aspects=["camera"])
    assert plan.phones == ["Samsung Galaxy S22 5G"] and plan.aspects == ["camera"]

    plan = plan_query("How does it compare to the S24?", matcher, context_phones=["Samsung Galaxy S23"])
    assert plan.intent == "compare" and plan.phones == ["Samsung Galaxy S23", "Samsung Galaxy S24"]


def test_catalogue_questions_ignore_context(matcher):
    plan = plan_query("Which phones have a stylus?", matcher, context_phones=["Samsung Galaxy S22 5G"])
    assert plan.phones == [] and plan.intent == "general"
    plan = plan_query("Which has the best battery life?", matcher, context_phones=["Samsung Galaxy S22 5G"])
    assert plan.intent == "rank"

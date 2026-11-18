"""End-to-end chatbot behaviour on the real scraped dataset (grounded mode, no LLM)."""

from app.analytics import rank_phones
from app.db.repository import all_phones, phone_by_name
from app.db.session import session_scope


def test_camera_specs_of_s23(assistant):
    r = assistant.ask("What are the camera specs of the Samsung Galaxy S23?")
    assert r["intent"] == "spec" and r["phones"] == ["Samsung Galaxy S23"]
    assert "50 MP, f/1.8" in r["answer"] and "3x optical zoom" in r["answer"] and "Selfie camera" in r["answer"]
    assert r["verification"]["passed"]


def test_best_battery_life_uses_measured_test(assistant):
    r = assistant.ask("Which Samsung phone has the best battery life?")
    assert r["intent"] == "rank"
    ranking = r["rankings"][0]
    assert ranking["metric"] == "battery_life"
    with session_scope() as db:
        expected = rank_phones(all_phones(db), "battery_life").rows[0].name
    assert ranking["rows"][0]["name"] == expected
    assert expected.replace("Samsung ", "") in r["answer"]
    # phones without the newer test are reported, with the legacy test as a complementary ranking
    assert any(rk["metric"] == "battery_endurance" for rk in r["rankings"])


def test_s23_vs_s22_performance_uses_same_benchmark_version(assistant):
    r = assistant.ask("How does the Galaxy S23 compare to the S22 in terms of performance?")
    assert r["intent"] == "compare"
    assert r["phones"] == ["Samsung Galaxy S23", "Samsung Galaxy S22 5G"]
    assert "AnTuTu (v9)" in r["answer"] and "Snapdragon 8 Gen 2" in r["answer"]
    assert "Galaxy S23 leads by" in r["answer"]


def test_screen_size_and_follow_up(assistant):
    first = assistant.ask("What is the screen size of the Galaxy S22?")
    assert "6.1 inches" in first["answer"]
    second = assistant.ask("What about its battery?", first["session_id"])
    assert second["phones"] == ["Samsung Galaxy S22 5G"]
    assert "3700 mAh" in second["answer"]


def test_price_filtered_ranking_respects_budget(assistant):
    r = assistant.ask("Which phone has the best camera under $500?")
    with session_scope() as db:
        for row in r["rankings"][0]["rows"]:
            assert phone_by_name(db, row["name"]).price_usd <= 500


def test_feature_question_lists_every_matching_phone(assistant):
    r = assistant.ask("Which phones have a stylus?")
    assert r["intent"] == "general"
    for name in ("S21 Ultra", "S22 Ultra", "S23 Ultra", "S24 Ultra", "S25 Ultra"):
        assert name in r["answer"]
    assert "Galaxy S23** -" not in r["answer"]


def test_unknown_phone_is_not_hallucinated(assistant):
    r = assistant.ask("What is the battery of the Galaxy S20?")
    assert r["intent"] == "unknown_phone" and "don't have data for Galaxy S20" in r["answer"]


def test_list_and_greeting(assistant):
    assert "15 Samsung phones" in assistant.ask("Which phones do you have?")["answer"]
    assert "Galaxy S23" in assistant.ask("hi")["answer"]


def test_compare_endpoint_logic(assistant):
    r = assistant.compare(["Galaxy S25 Ultra", "Z Fold7"], ["battery"])
    assert r["intent"] == "compare" and len(r["phones"]) == 2
    assert "Battery (mAh)" in r["answer"]

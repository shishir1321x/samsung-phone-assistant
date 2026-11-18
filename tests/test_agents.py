"""Multi-agent pipeline: routing, analysis, review writing and fact-checking."""

from app.agents.analyst import analyze, find_predecessor
from app.agents.fact_check import extract_numbers, verify
from app.assistant import PhoneAssistant
from app.db.repository import all_phones, phone_by_name
from app.db.session import session_scope


def test_fact_checker_accepts_grounded_and_flags_invented_numbers():
    source = "Battery: Li-Ion 3900 mAh. AnTuTu: 1231075 (v9). Active use score 11:27h. Weight 168 g."
    assert verify("A 3,900 mAh cell, about 1.23 million on AnTuTu, 11:27h active use.", source).passed
    result = verify("It has a 4500 mAh battery and weighs 168 g.", source)
    assert not result.passed and result.unsupported == ["4500"]
    # counts and years are not treated as specs
    assert verify("Released in 2023 with 3 cameras.", source).passed
    assert extract_numbers("11:27h")[0][1] == 11.45
    # LLMs spell durations out
    assert verify("It lasts 11 hours and 27 minutes of active use.", source).passed


def test_llm_output_cleanup():
    from app.llm.provider import clean_output

    assert clean_output("<think>plan the answer</think>\nThe 3900 mAh Li‑Ion cell.") == "The 3900 mAh Li-Ion cell."


def test_analyst_scores_and_predecessor():
    with session_scope() as db:
        phones = all_phones(db)
        s24u = phone_by_name(db, "Samsung Galaxy S24 Ultra")
        analysis = analyze(phones, s24u)
        assert find_predecessor(phones, s24u).name == "Samsung Galaxy S23 Ultra"
    assert set(analysis.scores) >= {"Display", "Performance", "Camera", "Battery", "Design"}
    assert all(4 <= v <= 10 for v in analysis.scores.values())
    assert analysis.pros and analysis.cons
    assert any("AnTuTu (v9)" in line for line in analysis.vs_predecessor)


def test_review_pipeline_without_llm(assistant):
    r = assistant.review("Galaxy S23", regenerate=True)
    agents = [step["agent"] for step in r["agents"]]
    assert agents == ["Coordinator", "Spec Retrieval Agent", "Benchmark & Market Analyst", "Review Writer", "Fact-Checker"]
    assert r["review"].startswith("# Samsung Galaxy S23 review")
    assert "## Pros" in r["review"] and "## Cons" in r["review"]
    assert r["verification"]["passed"] and r["generator"] == "template"
    cached = assistant.review("S23")
    assert cached["cached"] and cached["review"] == r["review"]


def test_llm_review_with_hallucination_is_revised_then_replaced(fake_llm):
    llm = fake_llm("# Review\nThe S23 has a 9999 mAh battery.", "# Review\nStill 8888 mAh.")
    agent = PhoneAssistant(llm=llm)
    r = agent.review("Galaxy S23", regenerate=True)
    actions = [step["action"] for step in r["agents"] if step["agent"] == "Fact-Checker"]
    assert len(llm.prompts) == 2 and "9999" in llm.prompts[1]  # feedback reached the writer
    assert "requested a revision" in actions[0] and "replaced with the grounded draft" in actions[1]
    assert r["generator"] == "template (fact-check fallback)"
    assert "9999" not in r["review"] and "8888" not in r["review"]


def test_llm_answer_that_is_grounded_is_kept(fake_llm):
    llm = fake_llm("The Galaxy S23 packs a 3900 mAh battery with 25W wired charging.")
    r = PhoneAssistant(llm=llm).ask("What battery does the Galaxy S23 have?")
    assert r["generator"] == "fake-llm" and r["verification"]["passed"]
    assert r["answer"].startswith("The Galaxy S23 packs")
    assert "3900 mAh" in llm.prompts[0]  # the retrieved facts were in the prompt (RAG)


def test_llm_comparison_keeps_the_exact_data_tables(fake_llm):
    llm = fake_llm("The Galaxy S23 is clearly faster: 1,231,075 vs 881,428 on AnTuTu (v9).")
    r = PhoneAssistant(llm=llm).ask("How does the Galaxy S23 compare to the S22 in terms of performance?")
    assert r["generator"] == "fake-llm" and r["verification"]["passed"]
    assert r["answer"].startswith("The Galaxy S23 is clearly faster")
    assert "| Chipset |" in r["answer"] and "AnTuTu (v9)" in r["answer"]  # grounded tables follow the takeaway


def test_truncated_llm_review_is_completed_from_the_analyst_data():
    from app.agents.writer import complete_review

    template = "# X review\n\n## Pros\n- Bright\n\n## Cons\n- Heavy\n\n## Bottom line\nGood phone."
    truncated = "# X\n\n## Display\nVery bright.\n\n## Pros\n- Bright screen.\n- **IP68"
    fixed = complete_review(truncated, template)
    assert "**IP68" not in fixed
    assert "## Cons\n- Heavy" in fixed and "## Bottom line\nGood phone." in fixed
    assert fixed.count("## Pros") == 1


def test_llm_failure_falls_back_to_grounded_answer():
    class Broken:
        name = "broken"

        def generate(self, *args, **kwargs):
            raise ConnectionError("provider down")

    r = PhoneAssistant(llm=Broken()).ask("What is the screen size of the Galaxy S22?")
    assert r["generator"] == "template" and "6.1 inches" in r["answer"]

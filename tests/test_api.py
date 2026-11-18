import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app.api.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["phones"] == 15 and body["rag_chunks"] > 100


def test_phones_and_detail(client):
    phones = client.get("/phones").json()
    assert len(phones) == 15
    detail = client.get("/phones/s23-ultra")
    assert detail.status_code == 200
    body = detail.json()
    assert body["name"] == "Samsung Galaxy S23 Ultra"
    assert "Main Camera" in body["specifications"] and body["benchmarks"]
    assert client.get("/phones/galaxy-s99").status_code == 404


def test_chat_with_follow_up(client):
    first = client.post("/chat", json={"message": "What are the camera specs of the Galaxy S23?"}).json()
    assert "50 MP" in first["answer"] and first["agents"][0]["agent"] == "Coordinator"
    follow = client.post("/chat", json={"message": "and its battery?", "session_id": first["session_id"]}).json()
    assert follow["phones"] == ["Samsung Galaxy S23"] and "3900 mAh" in follow["answer"]
    assert client.delete(f"/chat/{first['session_id']}").json()["cleared"] is True


def test_batch_concurrent_and_sequential(client):
    questions = ["Which Samsung phone has the best battery life?", "What is the screen size of the Galaxy S22?",
                 "How does the Galaxy S23 compare to the S22 in terms of performance?"]
    body = client.post("/chat/batch", json={"questions": questions}).json()
    assert [r["intent"] for r in body["results"]] == ["rank", "spec", "compare"]
    seq = client.post("/chat/batch", json={"questions": ["Specs of the Galaxy S24?", "What about its price?"],
                                           "sequential": True}).json()
    assert seq["results"][1]["phones"] == ["Samsung Galaxy S24"]
    assert seq["results"][0]["session_id"] == seq["results"][1]["session_id"]


def test_review_endpoints(client):
    created = client.post("/reviews", json={"phone": "Galaxy S25 Ultra", "regenerate": True})
    assert created.status_code == 200
    body = created.json()
    assert body["phone"] == "Samsung Galaxy S25 Ultra" and body["scores"] and body["agents"]
    assert client.get("/reviews/s25-ultra").json()["cached"] is True
    assert client.post("/reviews", json={"phone": "iPhone 15"}).status_code == 404


def test_compare_and_rankings(client):
    body = client.post("/compare", json={"phones": ["Galaxy S23", "Galaxy S22"], "aspects": ["performance"]}).json()
    assert body["intent"] == "compare"
    assert client.post("/compare", json={"phones": ["Galaxy S23", "Galaxy S99"]}).status_code == 404
    ranks = client.get("/rankings/price", params={"max_price": 300}).json()["rankings"][0]
    assert ranks["rows"] and all(row["price_usd"] <= 300 for row in ranks["rows"])
    assert client.get("/rankings/nonsense").status_code == 404
    assert any(m["key"] == "battery_life" for m in client.get("/metrics").json())


def test_validation_errors(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.post("/compare", json={"phones": ["Galaxy S23"]}).status_code == 422

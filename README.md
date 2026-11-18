# Samsung Phone Query & Review System

An intelligent assistant for Samsung smartphones. It **scrapes GSMArena** into a structured
**PostgreSQL** database, answers questions with a **RAG chatbot** built on **open-source models**,
writes product reviews with a **LangGraph multi-agent system**, and exposes everything through a
**FastAPI** API (with a small chat UI).

```
GSMArena ──scraper──▶ PostgreSQL ──▶ hybrid retriever (SQL facts + vector search) ──▶ agents ──▶ FastAPI
 (15 phones)          phones / specifications / prices /         ▲                    LangGraph     /chat
                      benchmarks / reviews / rag_chunks     bge-small embeddings      + LLM         /reviews ...
```

| Requirement | Where | Highlights |
|---|---|---|
| 1. Scrape Samsung phone data | [app/scraper/](app/scraper/) | 15 phones (S21 → S26, Ultra models, Z Fold7/Flip7, A57), ~58 spec rows each, typed fields, prices in 4 currencies, lab benchmarks |
| Store in a structured DB | [app/db/models.py](app/db/models.py) | 6 normalized tables; PostgreSQL (docker-compose) or SQLite fallback |
| 2. Chatbot with open-source LLM + RAG | [app/rag/](app/rag/), [app/llm/provider.py](app/llm/provider.py) | Hybrid retrieval, version-aware benchmark comparisons, follow-up questions, refuses to invent phones |
| 3. Multi-agent system | [app/agents/graph.py](app/agents/graph.py) | 6 LangGraph agents incl. spec retrieval, analyst, review writer and a fact-checker that loops back to the writer |
| 4. API | [app/api/main.py](app/api/main.py) | `/chat`, `/chat/batch` (concurrent), `/reviews`, `/compare`, `/rankings`, `/phones` + Swagger docs |
| Tests | [tests/](tests/) | 57 tests: parsers, scraper, NLU, chatbot accuracy, agents (with a fake hallucinating LLM), API |

---

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Optional: PostgreSQL (otherwise a local SQLite file is used)
docker compose up -d db
cp .env.example .env            # Windows: copy .env.example .env

python -m app.cli setup         # create tables, load data/phones.json, build the RAG index
python -m app.cli serve         # http://127.0.0.1:8000  (chat UI)   /docs (Swagger)
```

`setup` loads the dataset that was scraped from GSMArena and committed in
[data/phones.json](data/phones.json), so you can run the project without scraping. To scrape live:

```bash
python -m app.cli scrape                                   # the default 15 models
python -m app.cli scrape --models "Galaxy S24 FE" "Galaxy A56"   # any models on GSMArena's Samsung listing
```

Try it from the command line:

```bash
python -m app.cli ask "Which Samsung phone has the best battery life?" -v   # -v shows the agent trace
python -m app.cli review "Galaxy S25 Ultra"
python -m app.cli chat                                       # interactive, remembers context
pytest                                                       # 57 tests, offline, ~4 s
```

### Choosing the LLM (open-source by default)

The LLM is configured in `.env`. With `LLM_PROVIDER=auto` the first available option is used:

**Quickest free option (tested): Groq**, which serves open-source models over an OpenAI-compatible API.
Create a key at console.groq.com and put this in `.env`:

```
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=gsk_...
LLM_MODEL=qwen/qwen3.8-27b
```

Answers then take 1–3 s and a full review ~15 s. All providers:

| Provider | Setup | Default model |
|---|---|---|
| `huggingface` | `HF_TOKEN=hf_...` (Hugging Face Inference Providers, OpenAI-compatible router; needs provider credits) | `Qwen/Qwen3-235B-A22B-Instruct-2507` |
| `ollama` | install Ollama, `ollama pull llama3.1` | first pulled model |
| `transformers` | `pip install -r requirements-local-llm.txt` (runs in-process, CPU ok) | `Qwen/Qwen2.5-1.5B-Instruct` |
| `openai` / `openai_compatible` | `OPENAI_API_KEY`, or `LLM_BASE_URL` + `LLM_API_KEY` + `LLM_MODEL` (Groq's free tier, vLLM, LM Studio...) | `gpt-4o-mini` |
| `none` | nothing | grounded template answers |

Any model can be chosen with `LLM_MODEL`. With no LLM configured the whole system still works:
answers and reviews are composed directly from the database, which is also the safety net when an
LLM answer fails fact-checking or the provider is unreachable.

---

## 1. Scraping & database

**How it scrapes** ([app/scraper/gsmarena.py](app/scraper/gsmarena.py)): `requests` + **BeautifulSoup** (lxml).

* **Discovery** walks GSMArena's public Samsung brand listing (`samsung-phones-9.php`, page 2, 3, ...)
  and matches the target model names (`"Galaxy S21"` matches `"Galaxy S21 5G"`). The site's search endpoint
  sits behind a Cloudflare Turnstile challenge; the scraper never uses it, and if any page comes back
  as a challenge it **stops** with a clear error instead of trying to bypass it.
* **Politeness:** 5–7.5 s randomized delay between requests, back-off on HTTP 429/5xx (honours `Retry-After`),
  only paths allowed by `robots.txt`, and every page is **cached on disk** so re-runs and parser changes
  don't re-download. The full run is ~20 requests.
* **Parsing** keeps every spec row (category / name / value, multi-line values preserved) and derives
  typed columns: display size / refresh rate / brightness / ppi, RAM & storage, camera MP / lens count /
  optical zoom / video, battery mAh, wired & wireless charging W, weight, folded thickness, IP rating,
  dates, prices per currency, and GSMArena lab results (AnTuTu, GeekBench, 3DMark, measured brightness,
  loudspeaker, Active Use battery score and the legacy Endurance rating).

**Schema** ([app/db/models.py](app/db/models.py), SQLAlchemy 2.0):

| Table | Contents |
|---|---|
| `phones` | one row per model with ~50 typed, query-friendly columns (used for rankings/filters) |
| `specifications` | every raw GSMArena spec row (≈870 rows): category, name, `data-spec` key, value |
| `prices` | one row per currency (USD/EUR/GBP/INR) |
| `benchmarks` | lab results with **test version** (AnTuTu v8/v9/v10/v11, GeekBench v5.1/v6, ...) |
| `reviews` | generated reviews with scores, pros/cons, generator and fact-check result |
| `rag_chunks` | ~260 text chunks (one per phone × spec category) + embeddings |

## 2. Chatbot (RAG)

Questions go through a hybrid retriever ([app/rag/retriever.py](app/rag/retriever.py)):

1. **Query understanding** ([app/rag/nlu.py](app/rag/nlu.py)) resolves phone mentions exactly (`S23` ≠ `S23 Ultra`,
   `fold 7` → Z Fold7, `S23 Plus` → *not in the database* rather than silently the S23), detects the intent
   (spec question, comparison, ranking, review, open question) and the aspects asked about (camera, battery...),
   price limits ("under $500") and follow-ups ("what about its battery?", "how does it compare to the S24?").
2. **Structured retrieval (SQL):** exact spec rows for the asked aspects, rankings over typed columns,
   and benchmark comparisons.
3. **Semantic retrieval:** cosine search over open-source `BAAI/bge-small-en-v1.5` embeddings (via `fastembed`, ONNX,
   no torch), boosted toward the categories the question is about, plus a keyword scan of every spec sheet for
   open questions like *"Which phones have a stylus?"*.
4. **Generation:** the LLM receives the retrieved facts plus a grounded draft and writes the answer; the
   fact-checker then verifies every number.

**Accuracy details that matter**

* GSMArena changed its benchmark versions over the years (AnTuTu v8 → v11, GeekBench v5.1 → v6) and redid
  its battery test in 2023. Scores are **only compared within the same version**: "fastest phone" returns one
  ranking per AnTuTu version, comparisons use only benchmarks both phones share, and "best battery life" ranks by
  the Active Use score and lists phones measured only on the legacy test separately.
* Prices are GSMArena's current listings at scrape time (not launch MSRPs), and answers say so.
* "Best camera/display/phone" use composite scores computed relative to the dataset, and the answer explains how.

## 3. Multi-agent system (LangGraph)

```
Coordinator ─▶ Spec Retrieval ─┬─▶ Benchmark & Market Analyst ─┬─▶ Review Writer ─┐
                               │                               └─▶ Answer Agent ◀─┤ (compare)
                               └──────────────────────────────────▶ Answer Agent ──┤
                                                                                  ▼
                                     revise once ◀── Fact-Checker ──▶ END (or grounded fallback)
```

| Agent | Responsibility |
|---|---|
| **Coordinator** | Resolves phones, intent and aspects; uses conversation memory; routes the work |
| **Spec Retrieval Agent** | Fetches spec sheets, benchmarks, prices, rankings and vector hits from the database |
| **Benchmark & Market Analyst** | Scores the phone 1–10 per category against the whole dataset (lab tests, specs, price/value), finds rank positions, derives pros/cons, and compares with the predecessor (e.g. S24 Ultra vs S23 Ultra) using same-version benchmarks only |
| **Review Writer** | Writes the full review (LLM) from the spec sheet and analyst findings; builds a grounded template version too |
| **Answer Agent** | Writes chat answers from the retrieved context |
| **Fact-Checker** | Extracts every figure from the draft and checks it against the database. Unsupported numbers go back to the writer with feedback (one revision); if still wrong, the grounded version is returned |

The data agents are deterministic tool users (exact and auditable); the LLM is used where language matters.
Every response includes the **agent trace** (who did what, how long it took) and the fact-check result.

## 4. API

Start with `python -m app.cli serve` and open **http://127.0.0.1:8000/docs**.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/chat` | Ask a question. Pass back `session_id` for follow-ups |
| `POST` | `/chat/batch` | Up to 20 questions, answered concurrently (or `"sequential": true` as one conversation) |
| `DELETE` | `/chat/{session_id}` | Forget a conversation |
| `POST` | `/reviews` | Run the multi-agent review pipeline (`{"phone": "...", "regenerate": false}`) |
| `GET` | `/reviews/{phone}` | Stored review (generated on first request) |
| `POST` | `/compare` | Compare 2–4 phones, optionally on given aspects |
| `GET` | `/rankings/{metric}` | Rank by any metric (`battery_life`, `performance`, `price`, `camera_score`...), with `max_price` |
| `GET` | `/metrics` | Available ranking metrics |
| `GET` | `/phones`, `/phones/{phone}` | Raw data; `{phone}` accepts aliases like `s23-ultra` |
| `GET` | `/health` | DB, index, embedder and LLM status |

```bash
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
     -d '{"message": "How does the Galaxy S23 compare to the S22 in terms of performance?"}'
```

Response (abridged):

```json
{
  "session_id": "26240f92...",
  "answer": "**Galaxy S23 vs Galaxy S22 5G - performance comparison** ...",
  "intent": "compare",
  "phones": ["Samsung Galaxy S23", "Samsung Galaxy S22 5G"],
  "generator": "template",
  "verification": {"passed": true, "checked": 44, "unsupported": []},
  "sources": [{"phone": "Samsung Galaxy S23", "url": "https://www.gsmarena.com/samsung_galaxy_s23-12082.php"}],
  "agents": [{"agent": "Coordinator", "action": "intent=compare; phones=['Galaxy S23', 'Galaxy S22 5G']; aspects=['performance']", "ms": 0.4}, "..."],
  "elapsed_ms": 108.3
}
```

## Sample answers

From the bundled dataset, in grounded mode (no LLM):

> **Which Samsung phone has the best battery life?**
> **Galaxy S26 Ultra** comes out on top for battery life with 16:23h.
> 1. Galaxy S26 Ultra: 16:23h · 2. Galaxy S26: 15:20h · 3. Galaxy S25 Ultra: 14:49h · 4. Galaxy A57: 13:59h · 5. Galaxy S24 Ultra: 13:49h
> *Not covered by this test: S22, S22 Ultra, S21, S21 Ultra* → legacy Endurance rating: S23 Ultra 126h, S21 Ultra 114h, S22 Ultra 108h ...

> **How does the Galaxy S23 compare to the S22 in terms of performance?**
> Snapdragon 8 Gen 2 vs Exynos 2200 / Snapdragon 8 Gen 1, plus lab results on the same benchmark versions:
> AnTuTu (v9) S23 1,231,075 vs S22 881,428 (S23 +39.7%) · GeekBench (v5.1) 4,950 vs 3,682 (+34.4%) · GFXBench 110 vs 69 fps

> **What are the camera specs of the Samsung Galaxy S23?**
> Triple: 50 MP f/1.8 wide (OIS) · 10 MP f/2.4 telephoto, 3x optical zoom · 12 MP ultrawide; 8K@24/30fps video; 12 MP selfie with 4K@30/60fps video

With an open-source LLM (`LLM_PROVIDER=transformers`, `Qwen/Qwen2.5-1.5B-Instruct` on CPU), the same comparison
starts with a model-written takeaway, followed by the exact tables above; the fact-checker verified all 54 figures:

> The Samsung Galaxy S23 outperforms the Samsung Galaxy S22 5G across multiple benchmarks and metrics, including
> AnTuTu, GeekBench, and GFXBench scores. The Galaxy S23 leads by 39.7% in AnTuTu, 34.4% in GeekBench, and 59.4% in
> GFXBench. In terms of display brightness, the Galaxy S23 has a measured max brightness of 1,198 nits, while the
> Galaxy S22 5G has 982 nits. ...

## Project layout

```
app/
  config.py            settings (.env)
  analytics.py         metrics, version-aware rankings, dataset-relative scores
  assistant.py         service layer: sessions/memory, chat, reviews, compare
  cli.py               setup / scrape / ask / chat / review / serve
  scraper/             GSMArena client, page parsers, field derivation, pipeline
  db/                  SQLAlchemy models, session, repository
  rag/                 query understanding, chunking, embeddings, vector index, retriever, grounded answers
  llm/provider.py      Hugging Face / Ollama / transformers / OpenAI-compatible backends
  agents/              LangGraph graph, analyst, writer prompts, fact-checker
  api/                 FastAPI app, schemas, static chat UI
data/phones.json       scraped dataset (15 phones)
tests/                 pytest suite
```

## Notes & limitations

* Prices and specs reflect GSMArena when the data was scraped (see `scraped_at` in every phone record).
* Camera "quality" is inferred from specs (resolution, zoom, selfie); GSMArena does not publish image-quality scores.
* Small local models are slow on CPU: Qwen2.5-1.5B took 2–4 minutes per answer and ~12 minutes per review on a
  6-core desktop. Hosted open models (Groq, Hugging Face router) answer in seconds and write better prose.
* Groq's free tier is rate-limited; an occasional answer can take ~20–30 s while the client waits and retries.
* The fact-checker verifies **numbers**. Wrong qualitative claims (e.g. attributing a feature to the wrong lens) are
  not caught, which is why comparisons and rankings always show the exact data tables under the LLM's takeaway, and
  truncated or incomplete LLM reviews are completed with the analyst's pros/cons/bottom line.
* Conversation memory is in-process; a multi-worker deployment would move it to Redis or the database.

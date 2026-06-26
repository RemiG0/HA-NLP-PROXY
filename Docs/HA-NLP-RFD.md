# Requirements for Functional Document (RFD): Home Assistant NLP Proxy

---

## 1. Project Objective

Create a standalone, self-hosted service that acts as an OpenAI-compatible API proxy for Home Assistant. It intercepts standard `POST /v1/chat/completions` requests, classifies Polish voice commands locally using `allegro/herbert-base-cased` + `LinearSVC`, and returns a native OpenAI `tool_calls` response to trigger Home Assistant intents — with zero dependence on external cloud LLMs during normal operation.

The service additionally exposes a **browser-based admin UI** for:
- Managing intent training data and triggering model retraining
- Syncing live Home Assistant entities and training the entity classifier
- Configuring HA connection, fallback LLM, and classifier thresholds

When local classifier confidence falls below the configured threshold, the request is **transparently forwarded** to a user-configured fallback LLM API (any OpenAI-compatible endpoint, e.g. OpenAI, Ollama, LM Studio).

---

## 2. Technology Stack

| Layer | Technology |
|---|---|
| Package / env manager | [`uv`](https://docs.astral.sh/uv/) — replaces `pip`, `venv`, `pip-tools`; deterministic `uv.lock` |
| API + Web server | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| Admin UI templating | [Jinja2](https://jinja.palletsprojects.com/) (server-side rendered) + [HTMX](https://htmx.org/) for dynamic interactions without a JS build step |
| Embeddings model | HuggingFace Transformers — `allegro/herbert-base-cased` |
| Classification engine | `scikit-learn` — `LinearSVC` |
| Persistence | SQLite via `SQLModel` — stores config, training corpus, entity catalogue, inference logs |
| HA integration | `httpx` async client — HA REST API (`GET /api/states`, `GET /api/config`) |
| Fallback LLM | `httpx` — forwards full OpenAI-format request to any configurable endpoint |
| Target API format | OpenAI Chat Completions |

---

## 3. System Architecture

### 3.1 Request Processing Pipeline

```
Home Assistant
    │
    │  POST /v1/chat/completions
    ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Proxy                         │
│                                                         │
│  1. Extract prompt from messages[]                      │
│  2. Generate HerBERT embedding                          │
│  3. Run LinearSVC — get intent + entity + scores        │
│                                                         │
│  ┌──────────────────────────────────────────────┐       │
│  │  score ≥ threshold?                          │       │
│  │                                              │       │
│  │  YES → build tool_calls response             │       │
│  │  NO  → forward to Fallback LLM API ──────────┼───────┼──► OpenAI / Ollama / LM Studio
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
    │
    │  OpenAI tool_calls JSON  (or fallback LLM response)
    ▼
Home Assistant executes intent
```

### 3.2 Admin Web UI — Page Map

```
http://<host>:8000/admin/
│
├── /admin/                     Dashboard  (model status, last 20 inference logs, quick stats)
├── /admin/config               Configuration  (HA connection, fallback LLM, thresholds)
├── /admin/training/intents     Intent Training  (browse/add/delete training sentences per intent)
├── /admin/training/entities    Entity Training  (sync entities from HA, manage labels, train)
├── /admin/training/rules       Zasady Odmiany  (manage grammatical inflection rules for locative variants)
├── /admin/training/tester      Tester NLP  (test your voice commands locally against the models)
└── /admin/training/run         Trigger retrain  (HTMX endpoint, streams training progress)
```

### 3.3 Component Interaction

```
┌─────────────┐    REST    ┌──────────────────────────────────────────────────────┐
│ Home        │◄──────────►│  FastAPI App                                         │
│ Assistant   │            │                                                      │
│ (client)    │            │  ┌──────────┐  ┌────────────┐  ┌──────────────────┐ │
└─────────────┘            │  │ NLP      │  │ Admin UI   │  │ HA Sync Service  │ │
                           │  │ Router   │  │ (Jinja2 +  │  │ (httpx → HA API) │ │
                           │  │          │  │  HTMX)     │  │                  │ │
                           │  └────┬─────┘  └─────┬──────┘  └────────┬─────────┘ │
                           │       │               │                  │           │
                           │  ┌────▼───────────────▼──────────────────▼─────────┐ │
                           │  │              SQLite (SQLModel)                   │ │
                           │  │  tables: config · intent_samples · entities      │ │
                           │  │          inference_log                           │ │
                           │  └──────────────────────────────────────────────────┘ │
                           │                                                      │
                           │  ┌──────────────────────────────────────────────────┐ │
                           │  │  models/  (joblib artefacts, reloaded hot)       │ │
                           │  └──────────────────────────────────────────────────┘ │
                           └──────────────────────────────────────────────────────┘
```

---

## 4. Implementation Steps

### Step 1: Environment Setup

```bash
# Install uv (once per machine)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Bootstrap the project
mkdir ha-nlp-proxy && cd ha-nlp-proxy
uv init --python 3.11

# Runtime dependencies
uv add fastapi uvicorn "uvicorn[standard]" \
       transformers torch scikit-learn pandas joblib \
       sqlmodel jinja2 httpx python-multipart

# Dev / test dependencies
uv add --dev pytest pytest-cov httpx pytest-asyncio
```

### Step 2: Database Schema (`db.py`)

All persistent state lives in SQLite. Models are defined with `SQLModel` (Pydantic + SQLAlchemy).

```python
from sqlmodel import SQLModel, Field
from typing import Optional
import datetime

class Config(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    key: str = Field(unique=True, index=True)
    value: str

class IntentSample(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    sentence: str
    intent: str                          # e.g. HassTurnOn
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

class Entity(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    entity_id: str = Field(unique=True)  # e.g. light.salon
    friendly_name: str                   # e.g. Światło w salonie
    domain: str                          # light / switch / cover …
    enabled: bool = True                 # exclude from classifier if False

class InferenceLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    prompt: str
    intent: Optional[str]
    entity_id: Optional[str]
    intent_score: float
    entity_score: float
    routed_to_fallback: bool
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
```

### Step 3: Training Pipeline (`train.py`)

The training script is called both from the CLI and from the `/admin/training/run` endpoint (via `asyncio.create_subprocess_exec`).

```python
from sqlmodel import Session, select
from db import engine, IntentSample, Entity
from transformers import AutoTokenizer, AutoModel
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
import torch, joblib, numpy as np, pathlib

MODEL_NAME = "allegro/herbert-base-cased"
OUT = pathlib.Path("models")
OUT.mkdir(exist_ok=True)

def get_embeddings(sentences, tokenizer, model):
    embeddings = []
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            out = model(**inputs)
        embeddings.append(out.last_hidden_state[:, 0, :].squeeze().numpy())
    return np.vstack(embeddings)

def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert = AutoModel.from_pretrained(MODEL_NAME).eval()

    with Session(engine) as session:
        samples = session.exec(select(IntentSample)).all()
        entities = session.exec(select(Entity).where(Entity.enabled == True)).all()

    # ── Intent classifier ─────────────────────────────────
    sentences   = [s.sentence for s in samples]
    intent_lbls = [s.intent   for s in samples]

    X = get_embeddings(sentences, tokenizer, bert)

    le_intent = LabelEncoder().fit(intent_lbls)
    clf_intent = LinearSVC().fit(X, le_intent.transform(intent_lbls))

    # ── Entity classifier ─────────────────────────────────
    # Training sentences = friendly_name (+ paraphrases if added in future)
    ent_sentences = [e.friendly_name for e in entities]
    ent_labels    = [e.entity_id     for e in entities]

    X_ent = get_embeddings(ent_sentences, tokenizer, bert)

    le_entity  = LabelEncoder().fit(ent_labels)
    clf_entity = LinearSVC().fit(X_ent, le_entity.transform(ent_labels))

    # ── Persist ───────────────────────────────────────────
    joblib.dump(clf_intent,  OUT / "intent_clf.joblib")
    joblib.dump(clf_entity,  OUT / "entity_clf.joblib")
    joblib.dump(le_intent,   OUT / "label_enc_intent.joblib")
    joblib.dump(le_entity,   OUT / "label_enc_entity.joblib")
    print("Training complete.")

if __name__ == "__main__":
    train()
```

### Step 4: NLP Router (`nlp.py`)

```python
import joblib, torch, numpy as np, pathlib
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "allegro/herbert-base-cased"
_tokenizer = _bert = _clf_intent = _clf_entity = _le_intent = _le_entity = None

def load_models():
    global _tokenizer, _bert, _clf_intent, _clf_entity, _le_intent, _le_entity
    _tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    _bert       = AutoModel.from_pretrained(MODEL_NAME).eval()
    p = pathlib.Path("models")
    _clf_intent = joblib.load(p / "intent_clf.joblib")
    _clf_entity = joblib.load(p / "entity_clf.joblib")
    _le_intent  = joblib.load(p / "label_enc_intent.joblib")
    _le_entity  = joblib.load(p / "label_enc_entity.joblib")

def classify(text: str, threshold: float):
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        emb = _bert(**inputs).last_hidden_state[:, 0, :].squeeze().numpy().reshape(1, -1)

    intent_score = float(_clf_intent.decision_function(emb).max())
    entity_score = float(_clf_entity.decision_function(emb).max())

    if intent_score < threshold or entity_score < threshold:
        return None, None, intent_score, entity_score

    intent    = _le_intent.inverse_transform(_clf_intent.predict(emb))[0]
    entity_id = _le_entity.inverse_transform(_clf_entity.predict(emb))[0]
    return intent, entity_id, intent_score, entity_score
```

### Step 5: Main Application (`main.py`)

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import httpx, uuid, time, logging

from db import create_db, get_config, log_inference
from nlp import load_models, classify
from routers import admin

logger = logging.getLogger("ha-nlp-proxy")

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db()
    load_models()
    yield

app = FastAPI(title="HA NLP Proxy", lifespan=lifespan)
app.include_router(admin.router, prefix="/admin")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data    = await request.json()
    prompt  = next((m["content"] for m in reversed(data.get("messages", []))
                    if m.get("role") == "user"), "")
    threshold = float(get_config("confidence_threshold", "0.6"))

    intent, entity_id, i_score, e_score = classify(prompt, threshold)
    routed = intent is None

    log_inference(prompt, intent, entity_id, i_score, e_score, routed)

    if routed:
        return await _fallback(data)

    return {
        "id":      f"chatcmpl-{uuid.uuid4()}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   data.get("model", "ha-nlp"),
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {"name": intent, "arguments": f'{{"name": "{entity_id}"}}'}
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

async def _fallback(original_body: dict):
    url = get_config("fallback_url", "")
    key = get_config("fallback_api_key", "")
    if not url:
        return JSONResponse({"choices": [{"message": {
            "role": "assistant",
            "content": "Nie rozumiem polecenia i nie skonfigurowano fallback LLM."
        }, "finish_reason": "stop"}]})
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, json=original_body,
                              headers={"Authorization": f"Bearer {key}"})
        return JSONResponse(r.json(), status_code=r.status_code)

@app.get("/health")
def health():
    return {"status": "ok"}
```

---

## 5. Admin Web UI

### 5.1 Dashboard (`GET /admin/`)

Displays system status at a glance:

| Widget | Data source |
|---|---|
| Model status (loaded / not trained) | Check `models/*.joblib` exists |
| Confidence threshold | `config` table |
| Fallback LLM configured | `config` table |
| # intent samples | `COUNT(intent_sample)` |
| # entities in catalogue | `COUNT(entity)` |
| Last 20 inferences (prompt / intent / score / routed?) | `inference_log` table |

### 5.2 Configuration Page (`GET /admin/config`, `POST /admin/config`)

All values are stored as key/value rows in the `config` table.

| Field | Key | Default |
|---|---|---|
| HA Base URL | `ha_url` | `http://homeassistant.local:8123` |
| HA Long-Lived Access Token | `ha_token` | _(empty)_ |
| Confidence threshold | `confidence_threshold` | `0.6` |
| Fallback type | `fallback_type` | `none` (`none` / `local` / `cloud`) |
| Fallback LLM endpoint URL | `fallback_url` | _(empty)_ |
| Fallback LLM API key *(optional for local)* | `fallback_api_key` | _(empty)_ |
| Fallback LLM model name | `fallback_model` | `gpt-4o-mini` |

> When `fallback_type = local` the UI hides the API key field and pre-fills common local port suggestions. When `fallback_type = cloud` the API key field is required.

The page includes a **"Test HA Connection"** button (`HTMX GET /admin/config/test-ha`) that calls `GET /api/config` on HA and shows the location name on success.

The page includes a **"Test Fallback LLM"** button that sends a minimal ping request (`POST` with an empty user message) and shows `200 OK` or the error response.

### 5.3 Intent Training Page (`GET /admin/training/intents`)

Allows adding, editing, and deleting training sentences for each HA intent.

**Layout:**
```
[ Select intent ▾ ]  [ + Add intent ]

Intent: HassTurnOn
┌─────────────────────────────────────────────┬────────┐
│ włącz światło w salonie                     │ 🗑 Del │
│ zapal lampę w kuchni                        │ 🗑 Del │
│ uruchom wentylator                          │ 🗑 Del │
└─────────────────────────────────────────────┴────────┘
[ New sentence: _________________________ ] [ Add ]

[ 🔁 Retrain Models ]  ← HTMX, shows live progress log
```

**Backend endpoints:**

| Method | Path | Action |
|---|---|---|
| `GET` | `/admin/training/intents` | Render page |
| `POST` | `/admin/training/intents/add` | Insert `IntentSample` row |
| `DELETE` | `/admin/training/intents/{id}` | Delete row |
| `GET` | `/admin/training/intents/data` | HTMX partial — sentence list for selected intent |

### 5.4 Entity Training Page (`GET /admin/training/entities`)

Pulls live entities from Home Assistant and manages the entity classifier catalogue.

**Layout:**
```
[ 🔄 Sync from Home Assistant ]  ← calls HA REST API GET /api/states

Filter: [ domain ▾ ]  [ 🔍 search ]

┌──────────────────────┬──────────────────────────┬────────┬─────────┐
│ Entity ID            │ Friendly Name            │ Domain │ Enable  │
├──────────────────────┼──────────────────────────┼────────┼─────────┤
│ light.salon          │ Światło w salonie         │ light  │ ✅      │
│ light.kuchnia        │ Lampa kuchenna            │ light  │ ✅      │
│ switch.boiler        │ Bojler                   │ switch │ ✅      │
│ sensor.temp_outside  │ Temperatura zewnętrzna   │ sensor │ ❌      │
└──────────────────────┴──────────────────────────┴────────┴─────────┘

Friendly name is editable inline (used as training sentence for the entity classifier).

[ 🔁 Retrain Models ]
```

**HA Sync logic (`ha_sync.py`):**
```python
async def sync_entities_from_ha():
    url   = get_config("ha_url")
    token = get_config("ha_token")
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{url}/api/states",
                             headers={"Authorization": f"Bearer {token}"})
    states = r.json()
    # Upsert into Entity table; preserve enabled flag; skip sensor/input/automation by default
    EXCLUDED_DOMAINS = {"sensor", "binary_sensor", "automation", "script",
                        "input_boolean", "input_number", "input_text", "person",
                        "zone", "sun", "weather"}
    with Session(engine) as session:
        for s in states:
            domain = s["entity_id"].split(".")[0]
            if domain in EXCLUDED_DOMAINS:
                continue
            existing = session.get(Entity, s["entity_id"])
            if existing:
                existing.friendly_name = s["attributes"].get("friendly_name", s["entity_id"])
            else:
                session.add(Entity(
                    entity_id=s["entity_id"],
                    friendly_name=s["attributes"].get("friendly_name", s["entity_id"]),
                    domain=domain
                ))
        session.commit()
```

**Backend endpoints:**

| Method | Path | Action |
|---|---|---|
| `GET` | `/admin/training/entities` | Render page |
| `POST` | `/admin/training/entities/sync` | Trigger HA sync |
| `PATCH` | `/admin/training/entities/{id}` | Update friendly_name or enabled flag |

### 5.5 Retrain Endpoint (`POST /admin/training/run`)

Triggered by HTMX from both training pages. Runs `train.py` as a subprocess and streams stdout line-by-line using `StreamingResponse` so the browser shows a live progress log. On completion, the NLP module hot-reloads the new joblib artefacts without restarting the server.

```python
from fastapi.responses import StreamingResponse
import asyncio

@router.post("/training/run")
async def run_training():
    async def stream():
        proc = await asyncio.create_subprocess_exec(
            "uv", "run", "python", "train.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        async for line in proc.stdout:
            yield line.decode()
        await proc.wait()
        load_models()          # hot-reload artefacts
        yield "✅ Retraining complete. Models reloaded.\n"
    return StreamingResponse(stream(), media_type="text/plain")
```

---

### 5.6 Zasady Odmiany (`GET /admin/training/rules`)
Provides a UI to manage dynamic grammatical inflection rules (locative variants) for the NLP engine. In languages with heavy noun declension (like Polish), users rarely say commands using the base noun (e.g., "włącz światło w sypialnia"). Instead, they inflect the noun ("włącz światło w sypialni"). 
This page allows users to define custom `suffix_in` -> `suffix_out` rules (e.g., `nia` -> `ni`). During entity training, the engine automatically applies these rules to all matching entity and area names to create grammatically correct training phrases.

## 6. Model Persistence & Hot-Reload

Trained artefacts are saved to `models/` by `train.py`. The NLP module exposes `load_models()` which can be called at any time (from startup lifespan, from the retrain endpoint) to atomically swap the in-memory classifiers without dropping any in-flight requests.

> **Constraint:** Models must be loaded **once** at startup. Per-request model loading would add 3–8 s of latency — unacceptable.

---

## 7. Fallback LLM Routing

When `classify()` returns `None` (both intent and entity), the proxy:

1. Reads `fallback_url` and `fallback_api_key` from `config` table.
2. If configured: forwards the **original, unmodified** request body to `fallback_url` via `httpx.AsyncClient.post()`, adds `Authorization: Bearer <key>` and optionally overrides `model` with `fallback_model`.
3. Returns the fallback's response verbatim to Home Assistant.
4. If **not configured**: returns a plain text assistant message in Polish.

This means HA always gets a valid OpenAI-formatted response — either a local `tool_calls` or a natural-language response from the fallback.

**Configurable fallback targets:**

#### Local (self-hosted, no internet required)

| Server | Default port | `fallback_url` example | API key needed? |
|---|---|---|---|
| [Ollama](https://ollama.ai) | 11434 | `http://localhost:11434/v1/chat/completions` | No |
| [LM Studio](https://lmstudio.ai) | 1234 | `http://localhost:1234/v1/chat/completions` | No |
| [Jan](https://jan.ai) | 1337 | `http://localhost:1337/v1/chat/completions` | No |
| [LocalAI](https://localai.io) | 8080 | `http://localhost:8080/v1/chat/completions` | No |
| [llama.cpp server](https://github.com/ggerganov/llama.cpp/tree/master/examples/server) | 8080 | `http://localhost:8080/v1/chat/completions` | No |
| [vLLM](https://docs.vllm.ai) | 8000 | `http://localhost:8000/v1/chat/completions` | Optional |
| [text-generation-webui](https://github.com/oobabooga/text-generation-webui) (oobabooga) | 5000 | `http://localhost:5000/v1/chat/completions` | No |
| Custom local URL | any | `http://<host>:<port>/v1/chat/completions` | Optional |

> All of the above expose an OpenAI-compatible `/v1/chat/completions` endpoint. The proxy sends the original request body unchanged, so any locally hosted server reachable by the machine running the proxy works without code changes.

#### Cloud (requires internet + API key)

| Provider | `fallback_url` example |
|---|---|
| OpenAI | `https://api.openai.com/v1/chat/completions` |
| Azure OpenAI | `https://<resource>.openai.azure.com/openai/deployments/<model>/chat/completions?api-version=2024-02-01` |
| Anthropic (via OpenAI-compat proxy) | `https://api.anthropic.com/v1/chat/completions` |
| Google Gemini (via OpenAI-compat endpoint) | `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions` |

---

## 8. Smart Area Entity Resolution

When the system classifies a user's prompt as targeting an entire area (e.g., "włącz światła w kuchni", where `kuchnia` resolves to `area.kuchnia_id`), the proxy performs local smart entity resolution rather than delegating area intent processing purely to Home Assistant.

### 8.1 NLP Area Resolution Pipeline
1. **Determine Device Type**: The NLP engine (`nlp.py`) extracts domain keywords (e.g., `light`, `światł`, `blind`, `rolet`) from the user's prompt.
2. **Local Database Query**: The proxy queries the local SQLite `Entity` table for all devices physically assigned to the matched `area_id`.
3. **Keyword Filtering**: It cross-references these area entities against the guessed device type. It includes entities whose official HA domain matches (e.g., `light`) OR whose `friendly_name` or custom `aliases` contain the user's keyword (e.g., a smart plug in the `switch` domain named "Taśma LED (światło)").
4. **Multiple Tool Calls**: If the proxy identifies multiple matching entities within the area, it bypasses Home Assistant's built-in area resolution and instead generates a list of exact `tool_calls`—one for each specific entity (e.g., `{"name": "light.kitchen_main"}`, `{"name": "switch.kitchen_led"}`).

This mechanism provides extreme flexibility, enabling users to group arbitrarily domained devices (switches, relays) logically by alias without changing their underlying HA domain configurations.

## 9. Confidence Threshold & Scoring

`LinearSVC` exposes `decision_function` scores (not calibrated probabilities). The threshold is configurable per the Configuration page.

```python
CONFIDENCE_THRESHOLD = float(get_config("confidence_threshold", "0.6"))

intent_score = clf_intent.decision_function(emb).max()
entity_score = clf_entity.decision_function(emb).max()

if intent_score < threshold or entity_score < threshold:
    # route to fallback
```

> **Future:** Replace with `CalibratedClassifierCV(LinearSVC())` to get proper probability scores (0–1) making the threshold more intuitive.

---

## 10. Error Handling & Logging

All unhandled exceptions are caught and returned as valid OpenAI error JSON (not HTTP 500 HTML) to prevent HA from treating the integration as broken.

```python
@app.exception_handler(Exception)
async def global_exc(request, exc):
    logger.exception("Unhandled: %s", exc)
    return JSONResponse(status_code=500, content={
        "error": {"message": str(exc), "type": "proxy_error", "code": None}
    })
```

All inferences are logged to `InferenceLog` with prompt, predicted intent/entity, scores, and whether the fallback was used. This feeds the dashboard.

---

## 11. Project File Structure

```
ha-nlp-proxy/
├── data/                          # (optional) seed CSV for bulk import
│   └── commands_seed.csv
├── models/                        # generated by train.py — gitignored
│   ├── intent_clf.joblib
│   ├── entity_clf.joblib
│   ├── label_enc_intent.joblib
│   └── label_enc_entity.joblib
├── routers/
│   └── admin.py                   # all /admin/* routes
├── templates/                     # Jinja2 HTML templates
│   ├── base.html
│   ├── dashboard.html
│   ├── config.html
│   ├── training_intents.html
│   └── training_entities.html
├── static/
│   ├── htmx.min.js                # HTMX (vendored, no CDN at runtime)
│   └── style.css
├── db.py                          # SQLModel models + helpers
├── nlp.py                         # HerBERT embedding + classify()
├── ha_sync.py                     # HA REST API sync
├── train.py                       # offline / triggered training pipeline
├── main.py                        # FastAPI application entry point
├── ha_nlp.db                      # SQLite database — gitignored
├── pyproject.toml                 # managed by uv
├── uv.lock                        # deterministic lockfile — commit to VCS
├── Dockerfile
└── README.md
```

---

## 12. Deployment

### 12.1 Local (Developer Testing)

```bash
uv sync
uv run python main.py
# Admin UI → http://localhost:8000/admin/
# Proxy    → http://localhost:8000/v1/chat/completions
```

### 12.2 Docker Container (Recommended for Production)

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Dependency layer (cached unless pyproject.toml / uv.lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Pre-download HuggingFace model during build (no internet needed at runtime)
RUN uv run python -c "\
    from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('allegro/herbert-base-cased'); \
    AutoModel.from_pretrained('allegro/herbert-base-cased')"

COPY . .

EXPOSE 8000

# Persist DB and trained models across container restarts
VOLUME ["/app/ha_nlp.db", "/app/models"]

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t ha-nlp-proxy .
docker run -d \
  -p 8000:8000 \
  -v ha_nlp_db:/app/ha_nlp.db \
  -v ha_nlp_models:/app/models \
  --name ha-nlp-proxy \
  ha-nlp-proxy
```

### 12.3 Home Assistant Configuration

1. Open **Settings → Devices & Services → Add Integration → OpenAI Conversation**.
2. Set **Base URL**: `http://<proxy-host>:8000/v1`
3. Set **API Key**: any arbitrary string (proxy ignores it).
4. Open **Settings → Voice Assistants** and assign the integration.

### 12.4 Home Assistant Add-on (Future)

Package as a native HA add-on (`config.yaml` + `Dockerfile`) and publish to a custom add-on repository. Enables one-click install directly from the HA UI without a separate host machine.

### 12.5 Narzędzia Diagnostyczne CLI (CLI Tools)

W głównym katalogu projektu znajduje się plik `cli.py`, który dostarcza podręczny zestaw komend do administracji i diagnostyki bezpośrednio z poziomu konsoli:

```bash
# Wyświetlenie pomocy
uv run python cli.py --help

# Ręczne wymuszenie pobrania encji i obszarów z Home Assistant
uv run python cli.py sync

# Ręczne uruchomienie treningu modeli klasyfikacyjnych (zapisuje do models/)
uv run python cli.py train

# Błyskawiczny test zdania symulujący zapytanie od asystenta głosowego
uv run python cli.py test "włącz światło w kuchni"
```
Przydatne przy debugowaniu skuteczności dopasowywania bez konieczności "wyklikiwania" operacji w interfejsie graficznym.

---

## 13. Testing Strategy

| Level | Tool | What to test |
|---|---|---|
| Unit | `pytest` | `extract_prompt`, `classify`, `sync_entities_from_ha`, config helpers |
| Integration | `pytest` + `httpx.AsyncClient` | Full `/v1/chat/completions` cycle with mocked models; fallback routing when score < threshold |
| UI | `pytest` + Starlette `TestClient` | Admin page renders (200 OK), form submissions, HTMX partials |
| Accuracy | Held-out CSV split | Intent accuracy ≥ 95 %, entity accuracy ≥ 90 % on validation set |
| End-to-end | Manual HA voice command | Correct HA action fired for 20 representative Polish commands |

```bash
uv run pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## 14. Non-Functional Requirements

| Requirement | Target |
|---|---|
| Inference latency (p95) | < 200 ms per request |
| Fallback latency overhead | < 50 ms (proxy adds negligible overhead vs. direct LLM call) |
| Cold-start time | < 15 s (HerBERT load + DB init) |
| Memory footprint | < 1.5 GB RAM (HerBERT base + classifiers + web server) |
| Offline operation | 100 % for local NLP path; fallback requires internet only when triggered |
| Language | Polish (`pl`) primary; architecture is language-agnostic |
| Admin UI auth | Basic auth (username + password in `config` table) — prevents accidental access on LAN |

---

## 15. Open Questions / Future Roadmap

- [ ] **Slot extraction:** `HassLightSet` needs additional arguments (`brightness`, `color_temp`). Requires a second classification head or slot-filling layer.
- [ ] **Multi-intent commands:** "włącz światło i odtwórz muzykę" — requires sentence segmentation before classification.
- [ ] **Context / anaphora:** "wyłącz je" (referring to previously mentioned entities) requires session state.
- [ ] **Seed CSV import:** Bulk-import `data/commands_seed.csv` into `IntentSample` on first run.
- [ ] **Training paraphrases for entities:** Currently entity classifier trains on `friendly_name` only. Allow adding Polish paraphrase variants per entity via the UI.
- [ ] **Calibrated probabilities:** Replace raw SVM margin with `CalibratedClassifierCV` for more intuitive threshold configuration.
- [ ] **HA Add-on packaging:** One-click install from HA UI.
- [ ] **Periodic entity sync:** Schedule automatic re-sync from HA (e.g. every 24 h) so new devices are picked up without manual action.
- [ ] **Admin UI authentication:** Configurable basic auth or HA OAuth token validation.
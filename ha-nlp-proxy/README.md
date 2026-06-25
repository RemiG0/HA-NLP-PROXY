# HA NLP Proxy

A standalone, self-hosted service that acts as an OpenAI-compatible API proxy for Home Assistant. It intercepts standard `POST /v1/chat/completions` requests, classifies Polish voice commands locally using `allegro/herbert-base-cased` + `LinearSVC`, and returns a native OpenAI `tool_calls` response to trigger Home Assistant intents.

When the local model is uncertain (confidence score below threshold), the proxy automatically routes the request to a **Fallback LLM** (e.g. OpenAI, local Ollama, LM Studio) maintaining seamless operation.

## Key Features
- **Local NLP Engine:** Uses Polish HerBERT language model for fast, local classification of Home Assistant intents and entities.
- **Fallback LLM Routing:** If the local model certainty is too low, the query is transparently proxied to a standard LLM.
- **Automated Entity Sync:** Retrieves exposed entities and areas directly from Home Assistant via WebSockets. It keeps original English names for HA tool calls, while translating them to Polish for training the local AI.
- **Admin UI:** Built with FastAPI + HTMX. No page reloads, fast and responsive. Contains an **NLP Tester** to compare local model results directly with the Fallback LLM.
- **Custom Prompts:** Allows customizing the system prompt sent to the Fallback LLM, with dynamic injection of Home Assistant entities (`{{entities}}`).

## Setup & Running

1. **Install uv**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Install dependencies**:
   ```bash
   uv sync
   ```
3. **Run the server**:
   ```bash
   uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Admin UI
Available at `http://localhost:8000/admin/`.

Tabs and capabilities:
- **Konfiguracja:** Configure Home Assistant connection, Fallback LLM connection (with connection testing), Confidence Threshold, and custom System Prompts.
- **Encje:** Sync devices and areas from Home Assistant. Entities are automatically filtered to include only those exposed to Assist.
- **Trening NLP:** Define intents, add custom training sentences, define inflection rules (reguły odmiany), and execute model training.
- **Logi:** View inference logs of past commands and dynamically retrain the model on failed queries.
- **Tester NLP:** Real-time debugging environment comparing Local NLP parsed `tool_calls` alongside the Fallback LLM response.

## Home Assistant Setup
1. Open **Settings → Devices & Services → Add Integration → OpenAI Conversation**.
2. Set **Base URL**: `http://<proxy-host>:8000/v1`
3. Set **API Key**: any string (the proxy validates locally if configured)
4. Assign this integration to your **Voice Assistant pipeline**.

# HA NLP Proxy

A standalone, self-hosted service that acts as an OpenAI-compatible API proxy for Home Assistant. It intercepts standard `POST /v1/chat/completions` requests, classifies Polish voice commands locally using `allegro/herbert-base-cased` + `LinearSVC`, and returns a native OpenAI `tool_calls` response to trigger Home Assistant intents.

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
From here you can configure the Home Assistant connection, sync entities, add intent examples, and train the local classifiers.

## Home Assistant Setup
1. Open **Settings → Devices & Services → Add Integration → OpenAI Conversation**.
2. Set **Base URL**: `http://<proxy-host>:8000/v1`
3. Set **API Key**: any string
4. Assign to Voice Assistant pipeline.

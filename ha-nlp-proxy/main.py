from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import httpx, uuid, time, logging, os

from db import create_db, get_config, log_inference
from nlp import load_models, classify
from routers import admin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ha-nlp-proxy")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database...")
    create_db()
    logger.info("Loading models...")
    load_models()
    yield

app = FastAPI(title="HA NLP Proxy", lifespan=lifespan)
app.include_router(admin.router, prefix="/admin")

# Ensure static and templates dirs exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data    = await request.json()
    prompt  = next((m["content"] for m in reversed(data.get("messages", []))
                    if m.get("role") == "user"), "")
    threshold = float(get_config("confidence_threshold", "0.6"))

    intent, entity_id, i_score, e_score = classify(prompt, threshold)
    routed = intent is None or entity_id is None

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

@app.exception_handler(Exception)
async def global_exc(request, exc):
    logger.exception("Unhandled: %s", exc)
    return JSONResponse(status_code=500, content={
        "error": {"message": str(exc), "type": "proxy_error", "code": None}
    })

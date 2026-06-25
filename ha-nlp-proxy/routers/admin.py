from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, select, func, delete
import asyncio
from pathlib import Path

from db import engine, Config, IntentSample, Entity, InferenceLog, get_config, set_config
from ha_sync import sync_entities_from_ha
from nlp import load_models

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    with Session(engine) as session:
        intent_count = session.exec(select(func.count(IntentSample.id))).one()
        entity_count = session.exec(select(func.count(Entity.id))).one()
        
        # Get recent logs
        logs = session.exec(select(InferenceLog).order_by(InferenceLog.created_at.desc()).limit(20)).all()
        
        # Check model status
        models_path = Path("models")
        trained = models_path.exists() and (models_path / "intent_clf.joblib").exists()
        
        # Config status
        threshold = get_config("confidence_threshold", "0.6")
        fallback_url = get_config("fallback_url")

    return templates.TemplateResponse(request=request, name="dashboard.html", context={
        "intent_count": intent_count,
        "entity_count": entity_count,
        "logs": logs,
        "trained": trained,
        "threshold": threshold,
        "fallback_configured": bool(fallback_url)
    })

@router.get("/config", response_class=HTMLResponse)
async def get_config_page(request: Request):
    with Session(engine) as session:
        configs = session.exec(select(Config)).all()
        config_dict = {c.key: c.value for c in configs}
        
    return templates.TemplateResponse(request=request, name="config.html", context={
        "config": config_dict
    })

@router.post("/config")
async def save_config(
    request: Request,
    ha_url: str = Form(""),
    ha_token: str = Form(""),
    confidence_threshold: str = Form("0.6"),
    fallback_type: str = Form("none"),
    fallback_url: str = Form(""),
    fallback_api_key: str = Form(""),
    fallback_model: str = Form("gpt-4o-mini"),
    included_domains: str = Form("light, switch, cover, climate, media_player, fan, vacuum"),
    tester_system_prompt: str = Form("")
):
    set_config("ha_url", ha_url)
    set_config("ha_token", ha_token)
    set_config("confidence_threshold", confidence_threshold)
    set_config("fallback_type", fallback_type)
    set_config("fallback_url", fallback_url)
    set_config("fallback_api_key", fallback_api_key)
    set_config("fallback_model", fallback_model)
    set_config("included_domains", included_domains)
    set_config("tester_system_prompt", tester_system_prompt)
    
    return templates.TemplateResponse(request=request, name="config.html", context={
        "config": {
            "ha_url": ha_url,
            "ha_token": ha_token,
            "confidence_threshold": confidence_threshold,
            "fallback_type": fallback_type,
            "fallback_url": fallback_url,
            "fallback_api_key": fallback_api_key,
            "fallback_model": fallback_model,
            "included_domains": included_domains,
            "tester_system_prompt": tester_system_prompt
        },
        "message": "Configuration saved successfully."
    })

@router.post("/config/test_ha")
async def test_ha(ha_url: str = Form(""), ha_token: str = Form("")):
    if not ha_url:
        return HTMLResponse("<span style='color: red;'>URL HA jest wymagane do testu.</span>")
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            headers = {}
            if ha_token:
                headers["Authorization"] = f"Bearer {ha_token}"
            r = await client.get(f"{ha_url.rstrip('/')}/api/config", headers=headers)
            if r.status_code == 200:
                return HTMLResponse(f"<span style='color: green;'>✅ Połączono z sukcesem! HA wersja: {r.json().get('version', 'unknown')}</span>")
            else:
                return HTMLResponse(f"<span style='color: orange;'>⚠️ HTTP {r.status_code}: {r.text}</span>")
    except Exception as e:
        return HTMLResponse(f"<span style='color: red;'>❌ Błąd połączenia: {str(e)}</span>")

@router.post("/config/test_fallback")
async def test_fallback(
    fallback_url: str = Form(""),
    fallback_api_key: str = Form(""),
    fallback_model: str = Form("gpt-4o-mini")
):
    if not fallback_url:
        return HTMLResponse("<span style='color: red;'>URL Fallback LLM jest wymagane do testu.</span>")
        
    if fallback_url.endswith("/v1") or fallback_url.endswith("/v1/"):
        fallback_url = fallback_url.rstrip("/") + "/chat/completions"
        
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            headers = {}
            if fallback_api_key:
                headers["Authorization"] = f"Bearer {fallback_api_key}"
            payload = {
                "model": fallback_model,
                "messages": [{"role": "user", "content": "Wpisz tylko słowo 'test'."}]
            }
            r = await client.post(fallback_url, json=payload, headers=headers)
            
            # Fetch models
            models_list = []
            try:
                models_url = fallback_url.replace("/chat/completions", "/models")
                if not models_url.endswith("/models"):
                    models_url = models_url.rstrip("/") + "/models"
                m_r = await client.get(models_url, headers=headers)
                if m_r.status_code == 200:
                    for m in m_r.json().get("data", []):
                        if "id" in m:
                            models_list.append(m["id"])
            except Exception:
                pass
                
            oob_html = ""
            if models_list:
                options = "".join(f"<option value='{m}'></option>" for m in models_list)
                oob_html = f"""
<div class="form-group" id="fallback-model-group" hx-swap-oob="true">
    <label>Model Name</label>
    <input type="text" name="fallback_model" list="fallback-models-list" value="{fallback_model}">
    <datalist id="fallback-models-list">
        {options}
    </datalist>
    <small style="color: green;">Załadowano {len(models_list)} modeli z serwera!</small>
</div>"""
                
            if r.status_code == 200:
                resp_text = r.json().get("choices", [{}])[0].get("message", {}).get("content", "Brak treści")
                return HTMLResponse(f"<span style='color: green;'>✅ Połączono! Model odpowiedział: '{resp_text}'</span>" + oob_html)
            else:
                return HTMLResponse(f"<span style='color: orange;'>⚠️ HTTP {r.status_code}: {r.text}</span>" + oob_html)
    except Exception as e:
        return HTMLResponse(f"<span style='color: red;'>❌ Błąd połączenia: {str(e)}</span>")

@router.get("/training/intents", response_class=HTMLResponse)
async def intents_page(request: Request):
    with Session(engine) as session:
        samples = session.exec(select(IntentSample).order_by(IntentSample.intent, IntentSample.sentence)).all()
            
    return templates.TemplateResponse(request=request, name="training_intents.html", context={
        "samples": samples
    })

@router.get("/training/rules", response_class=HTMLResponse)
async def rules_page(request: Request):
    from db import InflectionRule
    with Session(engine) as session:
        rules = session.exec(select(InflectionRule)).all()
            
    return templates.TemplateResponse(request=request, name="training_rules.html", context={
        "rules": rules
    })

@router.post("/training/rules/add")
async def add_rule(request: Request, suffix_in: str = Form(...), suffix_out: str = Form(...)):
    from db import InflectionRule
    with Session(engine) as session:
        rule = InflectionRule(suffix_in=suffix_in.strip(), suffix_out=suffix_out.strip())
        session.add(rule)
        session.commit()
    
    rules = session.exec(select(InflectionRule)).all()
    return templates.TemplateResponse(request=request, name="training_rules.html", context={
        "rules": rules,
        "message": "Reguła została dodana."
    })

@router.post("/training/rules/delete/{rule_id}")
async def delete_rule(request: Request, rule_id: int):
    from db import InflectionRule
    with Session(engine) as session:
        rule = session.get(InflectionRule, rule_id)
        if rule:
            session.delete(rule)
            session.commit()
            
    rules = session.exec(select(InflectionRule)).all()
    return templates.TemplateResponse(request=request, name="training_rules.html", context={
        "rules": rules,
        "message": "Reguła została usunięta."
    })

@router.post("/training/intents/add")
async def add_intent_sample(request: Request, intent: str = Form(...), sentence: str = Form(...)):
    with Session(engine) as session:
        sample = IntentSample(intent=intent, sentence=sentence)
        session.add(sample)
        session.commit()
        
        samples = session.exec(select(IntentSample).order_by(IntentSample.intent, IntentSample.sentence)).all()
        
    return templates.TemplateResponse("partials/intent_list.html", {
        "request": request,
        "samples": samples
    })

@router.delete("/training/intents/{id}")
async def delete_intent_sample(request: Request, id: int):
    with Session(engine) as session:
        sample = session.get(IntentSample, id)
        if sample:
            session.delete(sample)
            session.commit()
            
            samples = session.exec(select(IntentSample).order_by(IntentSample.intent, IntentSample.sentence)).all()
            return templates.TemplateResponse(request=request, name="partials/intent_list.html", context={
                "samples": samples
            })
    return ""

@router.get("/training/entities", response_class=HTMLResponse)
async def entities_page(request: Request):
    with Session(engine) as session:
        entities = session.exec(select(Entity)).all()
        areas_list = session.exec(select(Entity).where(Entity.domain == "area")).all()
        areas_map = {a.entity_id.replace("area.", ""): a.friendly_name for a in areas_list}
        
    return templates.TemplateResponse(request=request, name="training_entities.html", context={
        "entities": entities,
        "areas_map": areas_map
    })

@router.post("/training/entities/sync")
async def sync_entities(request: Request):
    try:
        await sync_entities_from_ha()
        message = "Entities synced successfully."
    except Exception as e:
        message = f"Error syncing entities: {str(e)}"
        
    with Session(engine) as session:
        entities = session.exec(select(Entity)).all()
        
    return templates.TemplateResponse(request=request, name="training_entities.html", context={
        "entities": entities,
        "message": message
    })

@router.post("/training/entities/clear")
async def clear_entities(request: Request):
    with Session(engine) as session:
        session.exec(delete(Entity))
        session.commit()
    return templates.TemplateResponse(request=request, name="training_entities.html", context={
        "entities": [],
        "message": "Baza encji została wyczyszczona."
    })

@router.post("/training/run")
async def run_training():
    async def stream():
        yield "Starting training process...\n"
        import os
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["TQDM_DISABLE"] = "1"
        
        proc = await asyncio.create_subprocess_exec(
            "uv", "run", "python", "-u", "train.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env
        )
        async for line in proc.stdout:
            yield line.decode()
        await proc.wait()
        
        yield "\nReloading models in memory...\n"
        try:
            load_models()
            yield "✅ Retraining complete. Models reloaded.\n"
        except Exception as e:
            yield f"❌ Error reloading models: {str(e)}\n"
            
    return StreamingResponse(stream(), media_type="text/plain")

from nlp import classify
import json

@router.get("/training/tester", response_class=HTMLResponse)
async def tester_page(request: Request):
    return templates.TemplateResponse(request=request, name="tester.html")

def _build_ha_mock_payload(prompt: str, model_name: str) -> dict:
    with Session(engine) as session:
        entities = session.exec(select(Entity).where(Entity.enabled == True)).all()
        areas_list = session.exec(select(Entity).where(Entity.domain == "area")).all()
        # Fallback to friendly_name if original_name is None
        areas_map = {a.entity_id.replace("area.", ""): (a.original_name or a.friendly_name) for a in areas_list}
        
        static_context = "Static Context: An overview of the areas and the devices in this smart home:\n"
        for e in entities:
            if e.domain == "area":
                continue
            
            names = e.original_name or e.friendly_name
            original_aliases_str = e.original_aliases or e.aliases
            if original_aliases_str:
                names += ", " + original_aliases_str
            
            static_context += f"- names: {names}\n"
            static_context += f"  domain: {e.domain}\n"
            if e.area_id and e.area_id in areas_map:
                static_context += f"  areas: {areas_map[e.area_id]}\n"
                
    custom_prompt = get_config("tester_system_prompt", "").strip()
    if custom_prompt and "{{entities}}" in custom_prompt:
        system_prompt = custom_prompt.replace("{{entities}}", static_context)
    elif custom_prompt:
        system_prompt = custom_prompt + "\n" + static_context
    else:
        system_prompt = f"""You are a voice assistant for Home Assistant.
Answer questions about the world truthfully.
Answer in plain text. Keep it simple and to the point.
Replay in polish language to the user.

**ERROR HANDLING:** If a tool returns `InvalidSlotInfo` or any error, do not claim success. Inform the user about the failure.
Follow these instructions for tools from "assist":
When controlling Home Assistant always call the intent tools. Use HassTurnOn to lock and HassTurnOff to unlock a lock. When controlling a device, prefer passing just name and domain. When controlling an area, prefer passing just area name and domain.
When a user asks to turn on all devices of a specific type, ask user to specify an area, unless there is only one device of that type.

{static_context}
"""

    return {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "HassTurnOn",
                    "description": "Turns on a device or entity",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "area": {"type": "string"},
                            "domain": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "HassTurnOff",
                    "description": "Turns off a device or entity",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "area": {"type": "string"},
                            "domain": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "HassToggle",
                    "description": "Toggles a device or entity",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "area": {"type": "string"},
                            "domain": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                }
            }
        ]
    }

@router.post("/training/tester/run")
async def tester_run(request: Request, prompt: str = Form(...)):
    import time
    
    t0 = time.time()
    threshold = float(get_config("confidence_threshold", "0.6"))
    intent, entity_id, i_score, e_score = classify(prompt, threshold, return_all=True)
    nlp_time = (time.time() - t0) * 1000
    
    entity_name = ""
    if entity_id:
        with Session(engine) as session:
            db_ent = session.exec(select(Entity).where(Entity.entity_id == entity_id)).first()
            if db_ent:
                entity_name = db_ent.friendly_name

    from nlp import build_ha_arguments
    
    routed = (i_score < threshold or e_score < threshold)
    nlp_tool_calls_text = None
    fallback_response_text = None
    fallback_time = 0.0
    
    if not routed and entity_id:
        import uuid
        raw_args_list = build_ha_arguments(entity_id, prompt)
        nlp_tool_calls = []
        for raw_arg in raw_args_list:
            try:
                arg_obj = json.loads(raw_arg)
            except Exception:
                arg_obj = raw_arg
            nlp_tool_calls.append({
                "type": "function",
                "function": {
                    "name": intent,
                    "arguments": arg_obj
                },
                "id": "call_" + uuid.uuid4().hex[:8]
            })
        if nlp_tool_calls:
            nlp_tool_calls_text = json.dumps(nlp_tool_calls, indent=2, ensure_ascii=False)
        
    # Always try to fetch fallback for comparison if configured
    url = get_config("fallback_url", "")
    if url.endswith("/v1") or url.endswith("/v1/"):
        url = url.rstrip("/") + "/chat/completions"
        
    key = get_config("fallback_api_key", "")
    model = get_config("fallback_model", "gpt-4o-mini")
    if url:
        try:
            import httpx
            t1 = time.time()
            async with httpx.AsyncClient(timeout=10) as client:
                payload = _build_ha_mock_payload(prompt, model)
                headers = {}
                if key:
                    headers["Authorization"] = f"Bearer {key}"
                r = await client.post(url, json=payload, headers=headers)
                fallback_time = (time.time() - t1) * 1000
                if r.status_code == 200:
                    data = r.json()
                    msg = data.get("choices", [{}])[0].get("message", {})
                    if msg.get("content"):
                        fallback_response_text = msg.get("content")
                    elif msg.get("tool_calls"):
                        # Parse inner JSON strings in arguments for pretty printing
                        tool_calls_clean = msg.get("tool_calls")
                        for tc in tool_calls_clean:
                            if "function" in tc and "arguments" in tc["function"] and isinstance(tc["function"]["arguments"], str):
                                try:
                                    tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
                                except Exception:
                                    pass
                        fallback_response_text = "Wywołanie narzędzi (tool_calls):\n" + json.dumps(tool_calls_clean, indent=2, ensure_ascii=False)
                    else:
                        fallback_response_text = json.dumps(data, indent=2, ensure_ascii=False)
                else:
                    fallback_response_text = f"HTTP {r.status_code}: {r.text}"
        except Exception as e:
            fallback_response_text = f"Błąd połączenia z fallback LLM: {str(e)}"
    
    return templates.TemplateResponse(request=request, name="partials/tester_result.html", context={
        "prompt": prompt,
        "intent": intent,
        "entity_id": entity_id,
        "entity_name": entity_name,
        "i_score": i_score,
        "e_score": e_score,
        "threshold": threshold,
        "routed": routed,
        "nlp_tool_calls_text": nlp_tool_calls_text,
        "fallback_response_text": fallback_response_text,
        "nlp_time": nlp_time,
        "fallback_time": fallback_time
    })

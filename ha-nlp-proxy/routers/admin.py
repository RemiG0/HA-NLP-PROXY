from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, select, func
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
    fallback_model: str = Form("gpt-4o-mini")
):
    set_config("ha_url", ha_url)
    set_config("ha_token", ha_token)
    set_config("confidence_threshold", confidence_threshold)
    set_config("fallback_type", fallback_type)
    set_config("fallback_url", fallback_url)
    set_config("fallback_api_key", fallback_api_key)
    set_config("fallback_model", fallback_model)
    
    return templates.TemplateResponse(request=request, name="config.html", context={
        "config": {
            "ha_url": ha_url,
            "ha_token": ha_token,
            "confidence_threshold": confidence_threshold,
            "fallback_type": fallback_type,
            "fallback_url": fallback_url,
            "fallback_api_key": fallback_api_key,
            "fallback_model": fallback_model
        },
        "message": "Configuration saved successfully."
    })

@router.get("/training/intents", response_class=HTMLResponse)
async def intents_page(request: Request, intent: str = ""):
    with Session(engine) as session:
        # Get unique intents
        intents_res = session.exec(select(IntentSample.intent).distinct()).all()
        
        samples = []
        if intent:
            samples = session.exec(select(IntentSample).where(IntentSample.intent == intent)).all()
            
    return templates.TemplateResponse(request=request, name="training_intents.html", context={
        "intents": intents_res,
        "selected_intent": intent,
        "samples": samples
    })

@router.post("/training/intents/add")
async def add_intent_sample(request: Request, intent: str = Form(...), sentence: str = Form(...)):
    with Session(engine) as session:
        sample = IntentSample(intent=intent, sentence=sentence)
        session.add(sample)
        session.commit()
        
        samples = session.exec(select(IntentSample).where(IntentSample.intent == intent)).all()
        
    return templates.TemplateResponse("partials/intent_list.html", {
        "request": request,
        "samples": samples,
        "selected_intent": intent
    })

@router.delete("/training/intents/{id}")
async def delete_intent_sample(request: Request, id: int):
    with Session(engine) as session:
        sample = session.get(IntentSample, id)
        if sample:
            intent = sample.intent
            session.delete(sample)
            session.commit()
            
            samples = session.exec(select(IntentSample).where(IntentSample.intent == intent)).all()
            return templates.TemplateResponse(request=request, name="partials/intent_list.html", context={
                "samples": samples,
                "selected_intent": intent
            })
    return ""

@router.get("/training/intents/data")
async def get_intent_data(request: Request, intent: str):
    with Session(engine) as session:
        samples = session.exec(select(IntentSample).where(IntentSample.intent == intent)).all()
    return templates.TemplateResponse("partials/intent_list.html", {
        "request": request,
        "samples": samples,
        "selected_intent": intent
    })

@router.get("/training/entities", response_class=HTMLResponse)
async def entities_page(request: Request):
    with Session(engine) as session:
        entities = session.exec(select(Entity)).all()
        
    return templates.TemplateResponse(request=request, name="training_entities.html", context={
        "entities": entities
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

@router.post("/training/run")
async def run_training():
    async def stream():
        yield "Starting training process...\n"
        proc = await asyncio.create_subprocess_exec(
            "uv", "run", "python", "train.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
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

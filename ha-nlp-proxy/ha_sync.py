import asyncio
import json
import websockets
from sqlmodel import Session
from db import engine, Entity, get_config
from deep_translator import GoogleTranslator

translator = GoogleTranslator(source='en', target='pl')

def translate_to_pl(text: str) -> str:
    if not text:
        return ""
    try:
        return translator.translate(text)
    except Exception:
        return text

async def sync_entities_from_ha():
    url   = get_config("ha_url")
    token = get_config("ha_token")
    
    if not url or not token:
        raise ValueError("HA URL or Token is not configured.")

    included_domains_raw = get_config("included_domains", "light, switch, cover, climate, media_player, fan, vacuum")
    included_domains = {d.strip() for d in included_domains_raw.split(",") if d.strip()}

    ws_url = url.replace("http://", "ws://").replace("https://", "wss://") + "/api/websocket"
    
    async with websockets.connect(ws_url, max_size=None) as websocket:
        # 1. Wait for auth_required
        msg = await websocket.recv()
        data = json.loads(msg)
        
        if data.get("type") == "auth_required":
            await websocket.send(json.dumps({"type": "auth", "access_token": token}))
            msg = await websocket.recv()
            data = json.loads(msg)
            if data.get("type") != "auth_ok":
                raise ValueError("Home Assistant WebSocket Authentication failed. Check your token.")
                
        # 2. Query Exposed Entities List
        await websocket.send(json.dumps({"id": 1, "type": "homeassistant/expose_entity/list"}))
        msg = await websocket.recv()
        data = json.loads(msg)
        
        if not data.get("success"):
            raise ValueError(f"Failed to fetch exposed entities: {data}")
            
        all_exposed = data["result"]["exposed_entities"]
        exposed_to_conversation = [
            entity_id for entity_id, target_pipelines in all_exposed.items()
            if target_pipelines.get("conversation") is True
        ]
        
        registry = []
        msg_id = 100
        for entity_id in exposed_to_conversation:
            domain = entity_id.split(".")[0]
            if domain not in included_domains:
                continue
                
            await websocket.send(json.dumps({
                "id": msg_id,
                "type": "config/entity_registry/get",
                "entity_id": entity_id
            }))
            res = json.loads(await websocket.recv())
            if res.get("success") and res.get("result"):
                registry.append(res["result"])
            msg_id += 1
            
        # 3. Request area registry
        await websocket.send(json.dumps({"id": msg_id, "type": "config/area_registry/list"}))
        msg = await websocket.recv()
        data = json.loads(msg)
        
        if not data.get("success"):
            raise ValueError(f"Failed to fetch area registry: {data}")
            
        area_registry = data.get("result", [])
        
    # Process entities and areas
    with Session(engine) as session:
        for item in registry:
            entity_id = item.get("entity_id")
            domain = entity_id.split(".")[0]
            
            # Filter by domains
            if domain not in included_domains:
                continue
                
            original_name = item.get("name") or item.get("original_name") or entity_id
            ha_aliases = item.get("aliases") or []
            original_aliases = ",".join([a for a in ha_aliases if a is not None])
            
            # Translate
            translated_name = translate_to_pl(original_name)
            translated_aliases = [translate_to_pl(a) for a in ha_aliases if a is not None]
            aliases_str = ",".join(translated_aliases)
            area_id = item.get("area_id")
            
            from sqlmodel import select
            statement = select(Entity).where(Entity.entity_id == entity_id)
            existing = session.exec(statement).first()
            if existing:
                existing.friendly_name = translated_name
                existing.original_name = original_name
                existing.aliases = aliases_str
                existing.original_aliases = original_aliases
                existing.enabled = True
                existing.area_id = area_id
                session.add(existing)
            else:
                session.add(Entity(
                    entity_id=entity_id,
                    friendly_name=translated_name,
                    original_name=original_name,
                    domain=domain,
                    aliases=aliases_str,
                    original_aliases=original_aliases,
                    enabled=True,
                    area_id=area_id
                ))
                
        # Disable entities that are no longer exposed
        synced_entity_ids = [item.get("entity_id") for item in registry]
        all_db_entities = session.exec(select(Entity)).all()
        for db_ent in all_db_entities:
            if not db_ent.entity_id.startswith("area."):
                if db_ent.domain in included_domains and db_ent.entity_id not in synced_entity_ids:
                    db_ent.enabled = False
                    session.add(db_ent)
                
        # Process areas
        for area in area_registry:
            area_id = area.get("area_id")
            if not area_id:
                continue
            entity_id = f"area.{area_id}"
            
            original_name = area.get("name") or area_id
            ha_aliases = area.get("aliases", [])
            original_aliases = ",".join([a for a in ha_aliases if a is not None])
            
            translated_name = translate_to_pl(original_name)
            translated_aliases = [translate_to_pl(a) for a in ha_aliases if a is not None]
            aliases_str = ",".join(translated_aliases)
            
            statement = select(Entity).where(Entity.entity_id == entity_id)
            existing = session.exec(statement).first()
            if existing:
                existing.friendly_name = translated_name
                existing.original_name = original_name
                existing.aliases = aliases_str
                existing.original_aliases = original_aliases
                existing.enabled = True
                session.add(existing)
            else:
                session.add(Entity(
                    entity_id=entity_id,
                    friendly_name=translated_name,
                    original_name=original_name,
                    domain="area",
                    aliases=aliases_str,
                    original_aliases=original_aliases,
                    enabled=True
                ))
                
        session.commit()

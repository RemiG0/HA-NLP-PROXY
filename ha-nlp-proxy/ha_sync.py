import httpx
from sqlmodel import Session
from db import engine, Entity, get_config

async def sync_entities_from_ha():
    url   = get_config("ha_url")
    token = get_config("ha_token")
    
    if not url or not token:
        raise ValueError("HA URL or Token is not configured.")

    async with httpx.AsyncClient() as client:
        r = await client.get(f"{url}/api/states",
                             headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        
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
                session.add(existing)
            else:
                session.add(Entity(
                    entity_id=s["entity_id"],
                    friendly_name=s["attributes"].get("friendly_name", s["entity_id"]),
                    domain=domain
                ))
        session.commit()

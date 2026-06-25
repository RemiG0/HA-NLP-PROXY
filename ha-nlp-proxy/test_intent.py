import asyncio
import json
import websockets
from db import get_config

async def run():
    url   = get_config("ha_url")
    token = get_config("ha_token")
    ws_url = url.replace("http://", "ws://").replace("https://", "wss://") + "/api/websocket"
    
    async with websockets.connect(ws_url, max_size=None) as ws:
        msg = await ws.recv()
        await ws.send(json.dumps({"type": "auth", "access_token": token}))
        msg = await ws.recv()
        
        # Test 1: array of entity_ids
        payload = {
            "id": 100,
            "type": "conversation/process",
            "agent_id": "conversation.home_assistant",
            "text": "włącz test",
            "language": "pl"
        }
        # Actually, conversation/process only takes text. 
        # But wait, our proxy sends the tool call JSON, which is then handled by what?
        # Oh, in the proxy, we just return an OpenAI-compatible function call!
        # {"name": "HassTurnOn", "arguments": "{\"name\": \"...\"}"}
        # The frontend (custom component) receives this function call and executes the intent.
        
run()

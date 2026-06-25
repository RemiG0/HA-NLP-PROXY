import asyncio
import json
import websockets
from db import get_config

async def run():
    url = get_config("ha_url")
    token = get_config("ha_token")
    ws_url = url.replace("http://", "ws://").replace("https://", "wss://") + "/api/websocket"
    
    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "auth", "access_token": token}))
        await ws.recv()
        
        await ws.send(json.dumps({"id": 1, "type": "get_states"}))
        msg = await ws.recv()
        data = json.loads(msg)
        
        with open("ha_dump.json", "w") as f:
            json.dump(data, f, indent=2)

asyncio.run(run())

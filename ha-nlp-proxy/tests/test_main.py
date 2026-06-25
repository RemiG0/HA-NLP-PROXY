import pytest
from unittest.mock import patch
from db import set_config

@pytest.mark.asyncio
async def test_chat_completions_local_success(async_client):
    # Mock classify to return high confidence success
    with patch('main.classify') as mock_classify:
        mock_classify.return_value = ("HassTurnOn", "light.salon", 0.9, 0.9)
        
        request_body = {
            "model": "ha-nlp",
            "messages": [{"role": "user", "content": "włącz światło w salonie"}]
        }
        
        response = await async_client.post("/v1/chat/completions", json=request_body)
        assert response.status_code == 200
        
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["finish_reason"] == "tool_calls"
        
        tool_call = data["choices"][0]["message"]["tool_calls"][0]
        assert tool_call["function"]["name"] == "HassTurnOn"
        assert tool_call["function"]["arguments"] == '{"name": "light.salon"}'

@pytest.mark.asyncio
async def test_chat_completions_fallback_no_url(async_client):
    with patch('main.classify') as mock_classify:
        mock_classify.return_value = (None, None, 0.1, 0.1) # low confidence -> fallback
        
        request_body = {
            "messages": [{"role": "user", "content": "jakieś dziwne polecenie"}]
        }
        
        # We did not set fallback_url in config, should return error text
        response = await async_client.post("/v1/chat/completions", json=request_body)
        assert response.status_code == 200
        data = response.json()
        assert "Nie rozumiem polecenia" in data["choices"][0]["message"]["content"]

@pytest.mark.asyncio
async def test_chat_completions_fallback_success(async_client):
    set_config("fallback_url", "http://mocked_fallback/v1/chat/completions")
    
    with patch('main.classify') as mock_classify, patch('httpx.AsyncClient.post') as mock_post:
        mock_classify.return_value = (None, None, 0.1, 0.1)
        
        # Mock the external httpx post to return a fake fallback response
        class MockResponse:
            status_code = 200
            def json(self):
                return {"choices": [{"message": {"content": "Odpowiedź z chmury"}}]}
        
        mock_post.return_value = MockResponse()
        
        request_body = {
            "messages": [{"role": "user", "content": "opowiedz mi dowcip"}]
        }
        
        response = await async_client.post("/v1/chat/completions", json=request_body)
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Odpowiedź z chmury"

@pytest.mark.asyncio
async def test_chat_completions_area_success(async_client):
    with patch('main.classify') as mock_classify:
        mock_classify.return_value = ("HassTurnOn", "area.salon", 0.9, 0.9)
        
        request_body = {
            "model": "ha-nlp",
            "messages": [{"role": "user", "content": "włącz wszystko w salonie"}]
        }
        
        response = await async_client.post("/v1/chat/completions", json=request_body)
        assert response.status_code == 200
        
        data = response.json()
        tool_call = data["choices"][0]["message"]["tool_calls"][0]
        assert tool_call["function"]["name"] == "HassTurnOn"
        assert tool_call["function"]["arguments"] == '{"area": "salon"}'

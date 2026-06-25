import pytest

@pytest.mark.asyncio
async def test_dashboard_renders(async_client):
    response = await async_client.get("/admin/")
    assert response.status_code == 200
    assert "Dashboard" in response.text
    assert "Ostatnie Logi Inferencji" in response.text

@pytest.mark.asyncio
async def test_config_renders_and_saves(async_client):
    # Check GET
    response = await async_client.get("/admin/config")
    assert response.status_code == 200
    assert "Konfiguracja" in response.text
    
    # Check POST
    post_data = {
        "ha_url": "http://test:8123",
        "ha_token": "test_token",
        "confidence_threshold": "0.75",
        "fallback_type": "local",
        "fallback_url": "",
        "fallback_api_key": "",
        "fallback_model": ""
    }
    response_post = await async_client.post("/admin/config", data=post_data)
    assert response_post.status_code == 200
    assert "Zapisz konfigurację" in response_post.text
    assert 'value="0.75"' in response_post.text

@pytest.mark.asyncio
async def test_training_intents_renders(async_client):
    response = await async_client.get("/admin/training/intents")
    assert response.status_code == 200
    assert "Trening Intencji" in response.text

@pytest.mark.asyncio
async def test_training_entities_renders(async_client):
    response = await async_client.get("/admin/training/entities")
    assert response.status_code == 200
    assert "Trening Encji (Urządzeń)" in response.text

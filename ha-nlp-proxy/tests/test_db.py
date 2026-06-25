from db import get_config, set_config, log_inference, engine, InferenceLog
from sqlmodel import Session, select

def test_get_config_default():
    assert get_config("nonexistent", "mydefault") == "mydefault"
    assert get_config("fallback_url") == ""

def test_set_config():
    set_config("test_key", "test_value")
    assert get_config("test_key") == "test_value"
    
    # Update existing
    set_config("test_key", "new_value")
    assert get_config("test_key") == "new_value"

def test_log_inference():
    log_inference(
        prompt="włącz światło",
        intent="HassTurnOn",
        entity_id="light.salon",
        intent_score=1.5,
        entity_score=0.9,
        routed=False
    )
    
    with Session(engine) as session:
        log = session.exec(select(InferenceLog)).first()
        assert log is not None
        assert log.prompt == "włącz światło"
        assert log.intent == "HassTurnOn"
        assert log.entity_id == "light.salon"
        assert log.intent_score == 1.5
        assert log.routed_to_fallback is False

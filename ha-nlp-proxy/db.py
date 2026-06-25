from sqlmodel import SQLModel, Field
from typing import Optional
import datetime

class Config(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    key: str = Field(unique=True, index=True)
    value: str

class IntentSample(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    sentence: str
    intent: str                          # e.g. HassTurnOn
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

class Entity(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    entity_id: str = Field(unique=True)  # e.g. light.salon
    friendly_name: str                   # e.g. Światło w salonie
    domain: str                          # light / switch / cover …
    enabled: bool = True                 # exclude from classifier if False

class InferenceLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    prompt: str
    intent: Optional[str]
    entity_id: Optional[str]
    intent_score: float
    entity_score: float
    routed_to_fallback: bool
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

from sqlmodel import create_engine, Session, select
import sys

if "pytest" in sys.modules:
    sqlite_url = "sqlite:///:memory:"
else:
    sqlite_file_name = "ha_nlp.db"
    sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)

def create_db():
    SQLModel.metadata.create_all(engine)

def get_config(key: str, default: str = "") -> str:
    with Session(engine) as session:
        statement = select(Config).where(Config.key == key)
        result = session.exec(statement).first()
        if result:
            return result.value
        return default

def set_config(key: str, value: str):
    with Session(engine) as session:
        statement = select(Config).where(Config.key == key)
        result = session.exec(statement).first()
        if result:
            result.value = value
            session.add(result)
        else:
            new_config = Config(key=key, value=value)
            session.add(new_config)
        session.commit()

def log_inference(prompt: str, intent: Optional[str], entity_id: Optional[str], intent_score: float, entity_score: float, routed: bool):
    with Session(engine) as session:
        log = InferenceLog(
            prompt=prompt,
            intent=intent,
            entity_id=entity_id,
            intent_score=intent_score,
            entity_score=entity_score,
            routed_to_fallback=routed
        )
        session.add(log)
        session.commit()

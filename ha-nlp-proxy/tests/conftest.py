import pytest
from httpx import AsyncClient, ASGITransport
import sys
import os

# Add parent dir to path so tests can import from main module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app
from db import engine, SQLModel

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    # Create tables in the :memory: database
    SQLModel.metadata.create_all(engine)
    yield
    # Drop tables after all tests
    SQLModel.metadata.drop_all(engine)

@pytest.fixture(autouse=True)
def clear_db():
    # Clear tables before each test
    from sqlmodel import Session
    from db import Config, IntentSample, Entity, InferenceLog
    with Session(engine) as session:
        for model in [Config, IntentSample, Entity, InferenceLog]:
            session.query(model).delete()
        session.commit()

@pytest.fixture
async def async_client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

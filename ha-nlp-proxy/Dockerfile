FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Dependency layer (cached unless pyproject.toml / uv.lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Pre-download HuggingFace model during build
RUN uv run python -c "\
from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained('allegro/herbert-base-cased'); \
AutoModel.from_pretrained('allegro/herbert-base-cased')"

COPY . .

EXPOSE 8000

# Persist DB and trained models across container restarts
VOLUME ["/app/ha_nlp.db", "/app/models"]

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

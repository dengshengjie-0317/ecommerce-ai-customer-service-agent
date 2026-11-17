FROM docker.m.daocloud.io/library/python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
    PIP_DEFAULT_TIMEOUT=120

COPY requirements-hybrid.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --retries 3 \
    "setuptools==75.8.0" wheel

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=wheelhouse,target=/wheels,readonly \
    python -m pip install --retries 3 \
    /wheels/torch-2.6.0+cpu-cp312-cp312-linux_x86_64.whl \
    -r requirements-hybrid.txt \
    && python -m pip check

COPY agents/ ./agents/
COPY api/ ./api/
COPY core/ ./core/
COPY memory/ ./memory/
COPY retrieval/ ./retrieval/
COPY storage/ ./storage/
COPY knowledge/ ./knowledge/
COPY scripts/ ./scripts/
COPY static/ ./static/

RUN python -c "import torch; from FlagEmbedding import BGEM3FlagModel; from pymilvus import MilvusClient; from langgraph.checkpoint.redis.aio import AsyncRedisSaver; from api.main import app; assert torch.version.cuda is None; print('hybrid imports passed')"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
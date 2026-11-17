import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_FILE)


@dataclass(frozen=True)
class Settings:
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    base_url: str = os.getenv(
        "OPENAI_BASE_URL",
        "https://api.deepseek.com/v1",
    )
    model_name: str = os.getenv("MODEL_NAME", "deepseek-chat")
    redis_url: str = os.getenv(
        "REDIS_URL",
        "redis://localhost:6379/0",
    )
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))
    sqlite_db_path: Path = Path(
        os.getenv("SQLITE_DB_PATH", "./data/ecommerce.db")
    )
    retrieval_backend: str = os.getenv(
        "RETRIEVAL_BACKEND", "bm25"
    ).strip().lower()

    bge_model_path: Path = Path(
        os.getenv("BGE_MODEL_PATH", "./models/bge-m3")
    )

    milvus_db_path: Path = Path(
        os.getenv("MILVUS_DB_PATH", "./data/knowledge.db")
    )

settings = Settings()
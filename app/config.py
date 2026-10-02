"""Application settings (12-factor, from the environment)."""
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://memory:memorypass@db:5432/memory_db"

    # no default: the service must fail fast at startup if MASTER_API_KEY is missing
    master_api_key: str

    # Embeddings
    embeddings_provider: str = "ollama"  # ollama | openai | none
    embeddings_model: str = "bge-m3"
    embeddings_dim: int = 1024
    embeddings_base_url: str = "http://ollama:11434/v1"
    openai_api_key: str = ""

    # LLM for dialog distillation and reflect synthesis (OpenAI-compatible)
    llm_provider: str = "ollama"  # ollama | openai | none
    llm_model: str = "llama3.1:8b"
    llm_base_url: str = "http://ollama:11434/v1"
    llm_api_key: str = ""

    # Local memory folder (identity.md etc.)
    memory_home: str = "/data/memory"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()

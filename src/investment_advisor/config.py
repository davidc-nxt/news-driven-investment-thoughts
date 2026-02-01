"""Configuration management using Pydantic Settings."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database (get connection string from .env)
    database_url: str = "postgresql://advisor:advisor_secret@localhost:5433/investment_advisor"

    # OpenRouter API (compatible with OpenAI client)
    openrouter_api_key: str = ""

    # Embedding configuration
    embedding_provider: str = "local"  # "local" or "openai"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384  # MiniLM-L6-v2 dimension

    # LLM configuration (OpenRouter)
    llm_model: str = "openai/gpt-4o-mini"
    llm_temperature: float = 0.3

    # RAG configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_results: int = 5

    # Logging
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

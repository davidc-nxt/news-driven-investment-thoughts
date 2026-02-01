"""Database connection management."""

from functools import lru_cache
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from investment_advisor.config import get_settings


@lru_cache
def get_engine():
    """Get cached SQLAlchemy engine."""
    settings = get_settings()
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )


def get_session() -> Session:
    """Create a new database session."""
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()

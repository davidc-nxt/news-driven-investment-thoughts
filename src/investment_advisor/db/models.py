"""SQLAlchemy ORM models for the investment advisor database."""

from datetime import datetime, date
from typing import Optional
from sqlalchemy import String, Text, Integer, BigInteger, Boolean, DECIMAL, Date, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class Ticker(Base):
    """Tracked stock tickers."""

    __tablename__ = "tickers"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(10), unique=True, nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(255))
    sector: Mapped[Optional[str]] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class Article(Base):
    """News articles and releases."""

    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker_symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text)
    summary: Mapped[Optional[str]] = mapped_column(Text)
    published_at: Mapped[Optional[datetime]] = mapped_column()
    source: Mapped[Optional[str]] = mapped_column(String(100))
    url: Mapped[Optional[str]] = mapped_column(Text, unique=True)
    article_type: Mapped[str] = mapped_column(String(50), default="news")
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    __table_args__ = (
        Index("idx_articles_published", "published_at"),
    )


class MarketData(Base):
    """Daily OHLCV market data."""

    __tablename__ = "market_data"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker_symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    open_price: Mapped[Optional[float]] = mapped_column(DECIMAL(12, 4))
    high_price: Mapped[Optional[float]] = mapped_column(DECIMAL(12, 4))
    low_price: Mapped[Optional[float]] = mapped_column(DECIMAL(12, 4))
    close_price: Mapped[Optional[float]] = mapped_column(DECIMAL(12, 4))
    adj_close: Mapped[Optional[float]] = mapped_column(DECIMAL(12, 4))
    volume: Mapped[Optional[int]] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    __table_args__ = (
        Index("idx_market_data_ticker_date", "ticker_symbol", "date"),
    )


class Embedding(Base):
    """Vector embeddings for semantic search using pgvector."""

    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(primary_key=True)
    article_id: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    embedding = mapped_column(Vector(384))  # Native pgvector type!
    chunk_metadata: Mapped[dict] = mapped_column(JSONB, default={})
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class AdviceHistory(Base):
    """Generated investment advice history."""

    __tablename__ = "advice_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker_symbol: Mapped[Optional[str]] = mapped_column(String(10), index=True)
    query: Mapped[Optional[str]] = mapped_column(Text)
    context_summary: Mapped[Optional[str]] = mapped_column(Text)
    advice: Mapped[str] = mapped_column(Text, nullable=False)
    recommendation: Mapped[Optional[str]] = mapped_column(String(20))
    risk_score: Mapped[Optional[int]] = mapped_column(Integer)
    model_used: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

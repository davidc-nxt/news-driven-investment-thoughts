"""Data ingestion package initialization."""

from investment_advisor.ingestion.news import NewsFetcher
from investment_advisor.ingestion.market_data import MarketDataFetcher

__all__ = ["NewsFetcher", "MarketDataFetcher"]

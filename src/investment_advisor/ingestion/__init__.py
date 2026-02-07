"""Data ingestion package initialization."""

from investment_advisor.ingestion.news import NewsFetcher
from investment_advisor.ingestion.market_data import MarketDataFetcher
from investment_advisor.ingestion.fundamentals import FinancialDataService

__all__ = ["NewsFetcher", "MarketDataFetcher", "FinancialDataService"]


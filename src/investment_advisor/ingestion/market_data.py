"""Market data fetcher using yfinance."""

from datetime import datetime, timedelta
from typing import Optional
import yfinance as yf
import pandas as pd
from rich.console import Console
from sqlalchemy import select, and_

from investment_advisor.db.connection import get_session
from investment_advisor.db.models import MarketData, Ticker

console = Console()


class MarketDataFetcher:
    """Fetch and store market data from yfinance."""

    def __init__(self):
        self.session = get_session()

    def fetch_market_data(
        self, symbol: str, period: str = "1mo", interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch market data for a single ticker.

        Args:
            symbol: Stock ticker symbol
            period: Time period (e.g., '1mo', '3mo', '1y')
            interval: Data interval (e.g., '1d', '1h')

        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                console.print(f"[yellow]No market data found for {symbol}[/yellow]")
                return None
            console.print(f"[green]Fetched {len(df)} data points for {symbol}[/green]")
            return df
        except Exception as e:
            console.print(f"[red]Error fetching market data for {symbol}: {e}[/red]")
            return None

    def store_market_data(self, symbol: str, df: pd.DataFrame) -> int:
        """
        Store market data in the database.

        Args:
            symbol: Stock ticker symbol
            df: DataFrame with OHLCV data

        Returns:
            Number of rows stored
        """
        stored = 0
        symbol = symbol.upper()

        for idx, row in df.iterrows():
            # idx is the date/datetime
            data_date = idx.date() if hasattr(idx, "date") else idx

            # Check for existing record
            existing = self.session.execute(
                select(MarketData).where(
                    and_(
                        MarketData.ticker_symbol == symbol,
                        MarketData.date == data_date,
                    )
                )
            ).scalar_one_or_none()

            if existing:
                # Update existing record
                existing.open_price = float(row.get("Open", 0))
                existing.high_price = float(row.get("High", 0))
                existing.low_price = float(row.get("Low", 0))
                existing.close_price = float(row.get("Close", 0))
                existing.volume = int(row.get("Volume", 0))
            else:
                # Insert new record
                market_data = MarketData(
                    ticker_symbol=symbol,
                    date=data_date,
                    open_price=float(row.get("Open", 0)),
                    high_price=float(row.get("High", 0)),
                    low_price=float(row.get("Low", 0)),
                    close_price=float(row.get("Close", 0)),
                    adj_close=float(row.get("Close", 0)),
                    volume=int(row.get("Volume", 0)),
                )
                self.session.add(market_data)
                stored += 1

        self.session.commit()
        console.print(f"[blue]Stored/updated {len(df)} market data points for {symbol}[/blue]")
        return stored

    def fetch_and_store(self, symbol: str, period: str = "1mo") -> int:
        """
        Fetch market data and store it.

        Args:
            symbol: Stock ticker symbol
            period: Time period to fetch

        Returns:
            Number of new rows stored
        """
        df = self.fetch_market_data(symbol, period=period)
        if df is None:
            return 0
        return self.store_market_data(symbol, df)

    def fetch_all_tickers(self, period: str = "1mo") -> dict[str, int]:
        """
        Fetch market data for all active tickers.

        Returns:
            Dict mapping symbol to stored count
        """
        tickers = self.session.execute(
            select(Ticker).where(Ticker.is_active == True)
        ).scalars().all()

        results = {}
        for ticker in tickers:
            results[ticker.symbol] = self.fetch_and_store(ticker.symbol, period)

        return results

    def get_latest_price(self, symbol: str) -> Optional[dict]:
        """
        Get the latest price data for a symbol.

        Returns:
            Dict with latest price info or None
        """
        result = self.session.execute(
            select(MarketData)
            .where(MarketData.ticker_symbol == symbol.upper())
            .order_by(MarketData.date.desc())
            .limit(1)
        ).scalar_one_or_none()

        if result:
            return {
                "date": result.date,
                "open": float(result.open_price),
                "high": float(result.high_price),
                "low": float(result.low_price),
                "close": float(result.close_price),
                "volume": result.volume,
            }
        return None

    def close(self):
        """Close database session."""
        self.session.close()

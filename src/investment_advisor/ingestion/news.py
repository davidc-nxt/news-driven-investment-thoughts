"""News fetcher using yfinance."""

from datetime import datetime
from typing import Optional
import yfinance as yf
from rich.console import Console
from sqlalchemy import select

from investment_advisor.db.connection import get_session
from investment_advisor.db.models import Article, Ticker

console = Console()


class NewsFetcher:
    """Fetch and store news articles from yfinance."""

    def __init__(self):
        self.session = get_session()

    def fetch_news(self, symbol: str) -> list[dict]:
        """
        Fetch news for a single ticker.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')

        Returns:
            List of news article dictionaries
        """
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news or []
            console.print(f"[green]Fetched {len(news)} news items for {symbol}[/green]")
            return news
        except Exception as e:
            console.print(f"[red]Error fetching news for {symbol}: {e}[/red]")
            return []

    def _extract_article_data(self, article_data: dict) -> dict:
        """
        Extract article data from yfinance response, handling new API structure.
        
        The new yfinance API nests article data under 'content' key.
        """
        # Check if data is nested under 'content' (new API)
        if "content" in article_data:
            content = article_data["content"]
            
            # Extract URL from canonicalUrl structure
            url = None
            if "canonicalUrl" in content and content["canonicalUrl"]:
                url = content["canonicalUrl"].get("url")
            elif "clickThroughUrl" in content and content["clickThroughUrl"]:
                url = content["clickThroughUrl"].get("url")
            
            # Extract provider/source
            source = "Unknown"
            if "provider" in content and content["provider"]:
                source = content["provider"].get("displayName", "Unknown")
            
            # Extract published timestamp
            published_at = None
            if "pubDate" in content and content["pubDate"]:
                try:
                    published_at = datetime.fromisoformat(
                        content["pubDate"].replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    pass
            
            return {
                "title": content.get("title", ""),
                "summary": content.get("summary", ""),
                "description": content.get("description", ""),
                "url": url,
                "source": source,
                "published_at": published_at,
            }
        else:
            # Old API format (fallback)
            url = article_data.get("link") or article_data.get("url")
            
            published_at = None
            if "providerPublishTime" in article_data:
                published_at = datetime.fromtimestamp(article_data["providerPublishTime"])
            
            return {
                "title": article_data.get("title", ""),
                "summary": article_data.get("summary", ""),
                "description": "",
                "url": url,
                "source": article_data.get("publisher", "Unknown"),
                "published_at": published_at,
            }

    def store_article(self, symbol: str, article_data: dict) -> Optional[Article]:
        """
        Store a news article in the database, avoiding duplicates.

        Args:
            symbol: Stock ticker symbol
            article_data: News article data from yfinance

        Returns:
            Article object if stored, None if duplicate
        """
        extracted = self._extract_article_data(article_data)
        url = extracted["url"]
        
        if not url:
            console.print(f"[yellow]Skipping article without URL: {extracted['title'][:50]}...[/yellow]")
            return None

        # Check for existing article by URL
        existing = self.session.execute(
            select(Article).where(Article.url == url)
        ).scalar_one_or_none()

        if existing:
            return None

        # Combine summary and description for content
        content = extracted["summary"] or extracted["description"] or ""

        article = Article(
            ticker_symbol=symbol.upper(),
            title=extracted["title"],
            content=content,
            summary=extracted["summary"],
            published_at=extracted["published_at"],
            source=extracted["source"],
            url=url,
            article_type="news",
        )

        self.session.add(article)
        self.session.commit()
        self.session.refresh(article)
        return article

    def fetch_and_store(self, symbol: str) -> tuple[int, int]:
        """
        Fetch news and store new articles.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Tuple of (fetched_count, stored_count)
        """
        news = self.fetch_news(symbol)
        stored = 0

        for article_data in news:
            if self.store_article(symbol, article_data):
                stored += 1

        console.print(f"[blue]Stored {stored}/{len(news)} new articles for {symbol}[/blue]")
        return len(news), stored

    def fetch_all_tickers(self) -> dict[str, tuple[int, int]]:
        """
        Fetch news for all active tickers.

        Returns:
            Dict mapping symbol to (fetched_count, stored_count)
        """
        tickers = self.session.execute(
            select(Ticker).where(Ticker.is_active == True)
        ).scalars().all()

        results = {}
        for ticker in tickers:
            results[ticker.symbol] = self.fetch_and_store(ticker.symbol)

        return results

    def close(self):
        """Close database session."""
        self.session.close()

"""Investment Advisor CLI using Typer."""

from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from sqlalchemy import select, func

from investment_advisor.config import get_settings
from investment_advisor.db.connection import get_session
from investment_advisor.db.models import Ticker, Article, Embedding, MarketData
from investment_advisor.ingestion import NewsFetcher, MarketDataFetcher
from investment_advisor.rag import SemanticRetriever
from investment_advisor.advisor import InvestmentAdvisor

app = typer.Typer(
    name="invest",
    help="📈 Investment Advisor CLI - RAG-powered financial news analysis",
    add_completion=False,
)
console = Console()


@app.command()
def fetch(
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Specific ticker to fetch"),
    all_tickers: bool = typer.Option(False, "--all", "-a", help="Fetch for all tracked tickers"),
    period: str = typer.Option("1mo", "--period", "-p", help="Period for market data (1d, 1wk, 1mo, 3mo)"),
    index: bool = typer.Option(True, "--index/--no-index", help="Index articles after fetching"),
):
    """Fetch latest news and market data from yfinance."""
    console.print(Panel("[bold cyan]📊 Fetching Financial Data[/bold cyan]", border_style="cyan"))

    news_fetcher = NewsFetcher()
    market_fetcher = MarketDataFetcher()

    try:
        if ticker:
            # Fetch for specific ticker
            console.print(f"\n[bold]Fetching data for {ticker.upper()}...[/bold]")
            news_fetcher.fetch_and_store(ticker)
            market_fetcher.fetch_and_store(ticker, period=period)

        elif all_tickers:
            # Fetch for all tracked tickers
            console.print("\n[bold]Fetching data for all tracked tickers...[/bold]")
            news_results = news_fetcher.fetch_all_tickers()
            market_results = market_fetcher.fetch_all_tickers(period=period)

            # Summary table
            table = Table(title="Fetch Summary")
            table.add_column("Ticker", style="cyan")
            table.add_column("News Fetched", style="green")
            table.add_column("News Stored", style="blue")
            table.add_column("Market Points", style="magenta")

            for symbol in news_results:
                fetched, stored = news_results[symbol]
                market = market_results.get(symbol, 0)
                table.add_row(symbol, str(fetched), str(stored), str(market))

            console.print(table)

        else:
            console.print("[yellow]Specify --ticker or --all to fetch data[/yellow]")
            raise typer.Exit(1)

        # Index new articles
        if index:
            console.print("\n[bold]Indexing new articles...[/bold]")
            retriever = SemanticRetriever()
            retriever.index_all_unindexed()
            retriever.close()

        console.print("\n[green]✓ Fetch complete![/green]")

    finally:
        news_fetcher.close()
        market_fetcher.close()


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Filter by ticker"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of results"),
):
    """Semantic search across indexed news articles."""
    console.print(Panel(f"[bold cyan]🔍 Searching: {query}[/bold cyan]", border_style="cyan"))

    retriever = SemanticRetriever()
    try:
        results = retriever.search(query, ticker=ticker, top_k=limit)
        retriever.display_search_results(results)
    finally:
        retriever.close()


@app.command()
def advise(
    ticker: str = typer.Argument(..., help="Ticker symbol to analyze"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Specific focus area"),
):
    """Generate AI-powered investment advice for a ticker."""
    console.print(Panel(f"[bold cyan]🤖 Generating Investment Advice: {ticker.upper()}[/bold cyan]", border_style="cyan"))

    settings = get_settings()
    if not settings.openrouter_api_key:
        console.print("[red]Error: OPENROUTER_API_KEY not configured in .env[/red]")
        raise typer.Exit(1)

    advisor = InvestmentAdvisor()
    try:
        advice = advisor.generate_advice(ticker, query)
        advisor.display_advice(ticker, advice)
    finally:
        advisor.close()


@app.command()
def tickers(
    action: str = typer.Argument("list", help="Action: list, add, remove"),
    symbols: Optional[list[str]] = typer.Argument(None, help="Ticker symbols (for add/remove)"),
):
    """Manage tracked tickers."""
    session = get_session()

    try:
        if action == "list":
            # List all tickers
            result = session.execute(select(Ticker)).scalars().all()

            table = Table(title="Tracked Tickers")
            table.add_column("Symbol", style="cyan")
            table.add_column("Name", style="white")
            table.add_column("Sector", style="green")
            table.add_column("Active", style="yellow")

            for t in result:
                table.add_row(t.symbol, t.name or "N/A", t.sector or "N/A", "✓" if t.is_active else "✗")

            console.print(table)

        elif action == "add" and symbols:
            for symbol in symbols:
                existing = session.execute(
                    select(Ticker).where(Ticker.symbol == symbol.upper())
                ).scalar_one_or_none()

                if existing:
                    console.print(f"[yellow]{symbol.upper()} already exists[/yellow]")
                else:
                    ticker = Ticker(symbol=symbol.upper(), is_active=True)
                    session.add(ticker)
                    console.print(f"[green]Added {symbol.upper()}[/green]")

            session.commit()

        elif action == "remove" and symbols:
            for symbol in symbols:
                existing = session.execute(
                    select(Ticker).where(Ticker.symbol == symbol.upper())
                ).scalar_one_or_none()

                if existing:
                    existing.is_active = False
                    console.print(f"[yellow]Deactivated {symbol.upper()}[/yellow]")
                else:
                    console.print(f"[red]{symbol.upper()} not found[/red]")

            session.commit()

        else:
            console.print("[yellow]Usage: invest tickers [list|add|remove] [SYMBOLS...][/yellow]")

    finally:
        session.close()


@app.command()
def status():
    """Check system status and database statistics."""
    console.print(Panel("[bold cyan]📋 System Status[/bold cyan]", border_style="cyan"))

    session = get_session()
    settings = get_settings()

    try:
        # Database connection test
        try:
            session.execute(select(func.count()).select_from(Ticker))
            console.print("[green]✓ Database connected[/green]")
        except Exception as e:
            console.print(f"[red]✗ Database error: {e}[/red]")
            return

        # Statistics
        ticker_count = session.execute(select(func.count()).select_from(Ticker)).scalar()
        article_count = session.execute(select(func.count()).select_from(Article)).scalar()
        embedding_count = session.execute(select(func.count()).select_from(Embedding)).scalar()
        market_data_count = session.execute(select(func.count()).select_from(MarketData)).scalar()

        table = Table(title="Database Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green")

        table.add_row("Tracked Tickers", str(ticker_count))
        table.add_row("News Articles", str(article_count))
        table.add_row("Embeddings (Chunks)", str(embedding_count))
        table.add_row("Market Data Points", str(market_data_count))

        console.print(table)

        # Configuration
        config_table = Table(title="Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="white")

        config_table.add_row("Embedding Provider", settings.embedding_provider)
        config_table.add_row("Embedding Model", settings.embedding_model)
        config_table.add_row("LLM Model", settings.llm_model)
        config_table.add_row("OpenRouter Key", "✓ Configured" if settings.openrouter_api_key else "✗ Not set")

        console.print(config_table)

    finally:
        session.close()


@app.command()
def index():
    """Index all unindexed articles for semantic search."""
    console.print(Panel("[bold cyan]📚 Indexing Articles[/bold cyan]", border_style="cyan"))

    retriever = SemanticRetriever()
    try:
        count = retriever.index_all_unindexed()
        console.print(f"\n[green]✓ Indexed {count} chunks[/green]")
    finally:
        retriever.close()


if __name__ == "__main__":
    app()

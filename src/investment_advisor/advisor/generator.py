"""Investment advice generator using LangChain and OpenRouter."""

from typing import Optional
from datetime import datetime
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from investment_advisor.config import get_settings
from investment_advisor.db.connection import get_session
from investment_advisor.db.models import AdviceHistory, MarketData
from investment_advisor.rag.retriever import SemanticRetriever
from investment_advisor.advisor.prompts import INVESTMENT_ADVICE_PROMPT
from sqlalchemy import select

console = Console()


class InvestmentAdvisor:
    """Generate investment advice using LangChain and RAG with OpenRouter."""

    def __init__(self):
        self.settings = get_settings()
        self.session = get_session()
        self.retriever = SemanticRetriever()
        self._llm = None

    @property
    def llm(self):
        """Lazy load LLM using OpenRouter API."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.settings.llm_model,
                temperature=self.settings.llm_temperature,
                openai_api_key=self.settings.openrouter_api_key,
                openai_api_base="https://openrouter.ai/api/v1",
            )
        return self._llm

    def _get_market_summary(self, ticker: str) -> str:
        """Get market data summary for a ticker."""
        result = self.session.execute(
            select(MarketData)
            .where(MarketData.ticker_symbol == ticker.upper())
            .order_by(MarketData.date.desc())
            .limit(5)
        ).scalars().all()

        if not result:
            return "No market data available."

        latest = result[0]
        lines = [
            f"**Latest Close**: ${float(latest.close_price):.2f} ({latest.date})",
            f"**Volume**: {latest.volume:,}",
        ]

        if len(result) >= 2:
            prev = result[1]
            change = ((float(latest.close_price) - float(prev.close_price)) / float(prev.close_price)) * 100
            lines.append(f"**Day Change**: {change:+.2f}%")

        if len(result) >= 5:
            oldest = result[-1]
            week_change = ((float(latest.close_price) - float(oldest.close_price)) / float(oldest.close_price)) * 100
            lines.append(f"**5-Day Change**: {week_change:+.2f}%")

        return "\n".join(lines)

    def generate_advice(self, ticker: str, query: Optional[str] = None) -> str:
        """
        Generate investment advice for a ticker.

        Args:
            ticker: Stock ticker symbol
            query: Optional specific query/focus

        Returns:
            Investment advice as formatted string
        """
        ticker = ticker.upper()
        console.print(f"[cyan]Generating investment advice for {ticker}...[/cyan]")

        # Get relevant context from RAG
        search_query = query or f"latest news developments for {ticker}"
        context = self.retriever.get_context_for_ticker(ticker, search_query)

        # Get market data summary
        market_data = self._get_market_summary(ticker)

        # Build the prompt
        prompt = INVESTMENT_ADVICE_PROMPT.format(
            ticker=ticker,
            context=context,
            market_data=market_data,
        )

        # Generate advice using LLM
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            advice = response.content

            # Extract recommendation and risk score for storage
            recommendation = self._extract_recommendation(advice)
            risk_score = self._extract_risk_score(advice)

            # Store in history
            history = AdviceHistory(
                ticker_symbol=ticker,
                query=search_query,
                context_summary=context[:500] if context else None,
                advice=advice,
                recommendation=recommendation,
                risk_score=risk_score,
                model_used=self.settings.llm_model,
            )
            self.session.add(history)
            self.session.commit()

            return advice

        except Exception as e:
            console.print(f"[red]Error generating advice: {e}[/red]")
            return f"Error generating advice: {str(e)}"

    def _extract_recommendation(self, advice: str) -> Optional[str]:
        """Extract recommendation from advice text."""
        advice_upper = advice.upper()
        if "**BUY**" in advice_upper or "RECOMMENDATION: BUY" in advice_upper:
            return "BUY"
        elif "**SELL**" in advice_upper or "RECOMMENDATION: SELL" in advice_upper:
            return "SELL"
        elif "**HOLD**" in advice_upper or "RECOMMENDATION: HOLD" in advice_upper:
            return "HOLD"
        return None

    def _extract_risk_score(self, advice: str) -> Optional[int]:
        """Extract risk score from advice text."""
        match = re.search(r"RISK_SCORE:\s*(\d+)", advice)
        if match:
            score = int(match.group(1))
            return min(max(score, 1), 10)  # Clamp to 1-10
        return None

    def display_advice(self, ticker: str, advice: str):
        """Display advice in a formatted panel."""
        console.print(Panel(
            Markdown(advice),
            title=f"[bold cyan]Investment Analysis: {ticker}[/bold cyan]",
            border_style="cyan",
        ))

    def close(self):
        """Close resources."""
        self.retriever.close()
        self.session.close()

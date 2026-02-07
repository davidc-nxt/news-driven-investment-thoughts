"""Multi-step research agent inspired by virattt/dexter.

Decomposes complex financial queries into research steps,
executes them using available tools, and synthesizes findings.
"""

from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from investment_advisor.config import get_settings
from investment_advisor.rag.retriever import SemanticRetriever
from investment_advisor.ingestion.fundamentals import FinancialDataService
from investment_advisor.analysis.technical import TechnicalAnalyzer

console = Console()


class ResearchAgent:
    """
    Autonomous research agent that performs multi-step financial analysis.

    Inspired by Dexter's task planning and self-reflection pattern.
    Steps:
      1. Company Profile → understand the business
      2. Financial Fundamentals → analyze financial health
      3. Technical Analysis → assess price action
      4. News / RAG Search → get recent context
      5. Analyst Sentiment → consensus view
      6. Synthesize → LLM generates comprehensive report
    """

    def __init__(self):
        self.settings = get_settings()
        self.fundamentals = FinancialDataService()
        self.technical = TechnicalAnalyzer()
        self._retriever = None
        self._llm = None

    @property
    def retriever(self):
        """Lazy load retriever (requires DB)."""
        if self._retriever is None:
            try:
                self._retriever = SemanticRetriever()
            except Exception:
                self._retriever = None
        return self._retriever

    @property
    def llm(self):
        """Lazy load LLM."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.settings.llm_model,
                temperature=0.3,
                openai_api_key=self.settings.openrouter_api_key,
                openai_api_base="https://openrouter.ai/api/v1",
            )
        return self._llm

    def research(self, symbol: str, query: Optional[str] = None) -> str:
        """
        Run a comprehensive multi-step research analysis.

        Args:
            symbol: Stock ticker symbol
            query: Optional specific research question

        Returns:
            Comprehensive research report as markdown string
        """
        symbol = symbol.upper()
        research_context = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            # Step 1: Company Profile
            task = progress.add_task("Step 1/6: Fetching company profile...", total=None)
            profile = self.fundamentals.get_company_profile(symbol)
            research_context["profile"] = profile
            progress.update(task, completed=True, description="[green]✓ Company profile loaded[/green]")

            # Step 2: Key Metrics
            task = progress.add_task("Step 2/6: Fetching key metrics...", total=None)
            metrics = self.fundamentals.get_key_metrics(symbol)
            research_context["metrics"] = metrics
            progress.update(task, completed=True, description="[green]✓ Key metrics loaded[/green]")

            # Step 3: Financial Statements
            task = progress.add_task("Step 3/6: Analyzing financials...", total=None)
            income = self.fundamentals.get_income_statement(symbol, limit=3)
            balance = self.fundamentals.get_balance_sheet(symbol, limit=3)
            cashflow = self.fundamentals.get_cash_flow(symbol, limit=3)
            research_context["income"] = income
            research_context["balance"] = balance
            research_context["cashflow"] = cashflow
            progress.update(task, completed=True, description="[green]✓ Financial statements analyzed[/green]")

            # Step 4: Technical Analysis
            task = progress.add_task("Step 4/6: Running technical analysis...", total=None)
            tech = self.technical.analyze(symbol)
            tech_summary = self.technical.generate_summary(tech)
            research_context["technical"] = tech_summary
            progress.update(task, completed=True, description="[green]✓ Technical analysis complete[/green]")

            # Step 5: News Context (RAG)
            task = progress.add_task("Step 5/6: Searching news context...", total=None)
            search_query = query or f"latest news and developments for {symbol}"
            if self.retriever:
                news_context = self.retriever.get_context_for_ticker(symbol, search_query)
            else:
                news_context = "RAG database not available. Run 'invest fetch' with DB running to ingest news."
            research_context["news"] = news_context
            progress.update(task, completed=True, description="[green]✓ News context retrieved[/green]")

            # Step 6: Analyst Recommendations
            task = progress.add_task("Step 6/6: Fetching analyst consensus...", total=None)
            analyst = self.fundamentals.get_analyst_recommendations(symbol)
            insider = self.fundamentals.get_insider_trades(symbol)
            research_context["analyst"] = analyst
            research_context["insider"] = insider
            progress.update(task, completed=True, description="[green]✓ Analyst data loaded[/green]")

        # Build the research prompt
        prompt = self._build_research_prompt(symbol, research_context, query)

        # Generate the report
        console.print("\n[cyan]Synthesizing research report...[/cyan]")
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            console.print(f"[red]Error generating report: {e}[/red]")
            return f"Error generating research report: {str(e)}"

    def compare(self, symbols: list[str], query: Optional[str] = None) -> str:
        """
        Compare multiple tickers side by side.

        Args:
            symbols: List of ticker symbols to compare
            query: Optional comparison focus

        Returns:
            Comparison report as markdown string
        """
        all_data = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for sym in symbols:
                sym = sym.upper()
                task = progress.add_task(f"Analyzing {sym}...", total=None)

                data = {
                    "profile": self.fundamentals.get_company_profile(sym),
                    "metrics": self.fundamentals.get_key_metrics(sym),
                    "technical": self.technical.generate_summary(self.technical.analyze(sym)),
                }
                all_data[sym] = data
                progress.update(task, completed=True, description=f"[green]✓ {sym} analyzed[/green]")

        # Build comparison prompt
        prompt = self._build_comparison_prompt(symbols, all_data, query)

        console.print("\n[cyan]Generating comparison report...[/cyan]")
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"Error generating comparison: {str(e)}"

    def _build_research_prompt(self, symbol: str, ctx: dict, query: Optional[str] = None) -> str:
        """Build the comprehensive research prompt."""
        profile = ctx.get("profile", {})
        metrics = ctx.get("metrics", {})
        analyst = ctx.get("analyst", {})
        insider = ctx.get("insider", {})

        # Format financial data
        income_text = self._format_statement(ctx.get("income", {}))
        balance_text = self._format_statement(ctx.get("balance", {}))
        cashflow_text = self._format_statement(ctx.get("cashflow", {}))

        # Format insider trades
        insider_text = "No insider trades data available."
        trades = insider.get("trades", [])
        if trades:
            insider_lines = []
            for t in trades[:10]:
                insider_lines.append(
                    f"- {t.get('date', 'N/A')}: {t.get('insider', 'N/A')} - "
                    f"{t.get('transaction', 'N/A')} - {t.get('shares', 0):,.0f} shares"
                )
            insider_text = "\n".join(insider_lines)

        return f"""You are a senior financial analyst conducting deep research on {symbol}.
Produce a comprehensive, data-driven investment research report.

## Company Overview
Name: {profile.get('name', 'N/A')}
Sector: {profile.get('sector', 'N/A')} | Industry: {profile.get('industry', 'N/A')}
Market Cap: ${metrics.get('market_cap', 0):,.0f} if available
Description: {profile.get('description', 'N/A')[:300]}

## Key Valuation Metrics
- Price: ${metrics.get('price', 'N/A')}
- P/E (TTM): {metrics.get('pe_trailing', 'N/A')} | Forward P/E: {metrics.get('pe_forward', 'N/A')}
- PEG Ratio: {metrics.get('peg_ratio', 'N/A')}
- P/B: {metrics.get('price_to_book', 'N/A')} | P/S: {metrics.get('price_to_sales', 'N/A')}
- EV/EBITDA: {metrics.get('ev_to_ebitda', 'N/A')}
- Beta: {metrics.get('beta', 'N/A')}
- 52W Range: ${metrics.get('52w_low', 'N/A')} - ${metrics.get('52w_high', 'N/A')}

## Profitability
- Gross Margin: {self._fmt_pct(metrics.get('gross_margin'))}
- Operating Margin: {self._fmt_pct(metrics.get('operating_margin'))}
- Profit Margin: {self._fmt_pct(metrics.get('profit_margin'))}
- ROE: {self._fmt_pct(metrics.get('return_on_equity'))}
- ROA: {self._fmt_pct(metrics.get('return_on_assets'))}

## Financial Health
- Debt/Equity: {metrics.get('debt_to_equity', 'N/A')}
- Current Ratio: {metrics.get('current_ratio', 'N/A')}
- Revenue Growth: {self._fmt_pct(metrics.get('revenue_growth'))}
- Earnings Growth: {self._fmt_pct(metrics.get('earnings_growth'))}

## Income Statement (Recent Periods)
{income_text}

## Balance Sheet (Recent Periods)
{balance_text}

## Cash Flow (Recent Periods)
{cashflow_text}

## Technical Analysis
{ctx.get('technical', 'No technical data available.')}

## Recent News & Developments
{ctx.get('news', 'No recent news available.')}

## Analyst Consensus
Recommendation: {analyst.get('recommendation_key', 'N/A')}
Number of Analysts: {analyst.get('analyst_count', 'N/A')}
Price Target (Mean): ${analyst.get('target_mean', 'N/A')}
Price Target Range: ${analyst.get('target_low', 'N/A')} - ${analyst.get('target_high', 'N/A')}

## Insider Activity
{insider_text}

{f'## Specific Research Focus: {query}' if query else ''}

## Required Output
Provide your analysis in this structure:

### Executive Summary
2-3 sentences on the current investment thesis.

### Financial Analysis
Assessment of revenue growth, profitability, balance sheet strength, and cash flow generation.

### Valuation Assessment
Is the stock fairly valued, overvalued, or undervalued relative to peers and historical metrics?

### Technical Outlook
What does the price action and momentum suggest?

### Catalyst Tracker
Upcoming events/catalysts that could move the stock (earnings, product launches, macro events).

### Risk Assessment
Key risks (company-specific, sector, macro). Rate overall risk 1-10.
Format: RISK_SCORE: [number]

### Investment Recommendation
**BUY**, **HOLD**, or **SELL** with conviction level (High/Medium/Low) and 6-12 month outlook.

---
*Disclaimer: AI-generated analysis for informational purposes only. Not financial advice.*
"""

    def _build_comparison_prompt(
        self, symbols: list[str], all_data: dict, query: Optional[str] = None
    ) -> str:
        """Build comparison prompt."""
        sections = []
        for sym in symbols:
            data = all_data.get(sym.upper(), {})
            profile = data.get("profile", {})
            metrics = data.get("metrics", {})

            sections.append(f"""
### {sym.upper()} - {profile.get('name', 'N/A')}
Sector: {profile.get('sector', 'N/A')} | Industry: {profile.get('industry', 'N/A')}
Market Cap: ${metrics.get('market_cap', 0):,.0f} if available
Price: ${metrics.get('price', 'N/A')} | P/E: {metrics.get('pe_trailing', 'N/A')} | P/B: {metrics.get('price_to_book', 'N/A')}
Gross Margin: {self._fmt_pct(metrics.get('gross_margin'))} | Operating Margin: {self._fmt_pct(metrics.get('operating_margin'))}
ROE: {self._fmt_pct(metrics.get('return_on_equity'))} | Revenue Growth: {self._fmt_pct(metrics.get('revenue_growth'))}
Debt/Equity: {metrics.get('debt_to_equity', 'N/A')} | Current Ratio: {metrics.get('current_ratio', 'N/A')}
Beta: {metrics.get('beta', 'N/A')} | Dividend Yield: {self._fmt_pct(metrics.get('dividend_yield'))}

Technical Summary:
{data.get('technical', 'N/A')}
""")

        tickers_str = " vs ".join(symbols)
        return f"""You are a senior financial analyst comparing {tickers_str}.

## Company Data
{''.join(sections)}

{f'## Specific Comparison Focus: {query}' if query else ''}

## Required Output
Provide a side-by-side comparison:

### Overview Comparison Table
Create a comparison table of key metrics.

### Valuation Comparison
Which is more attractively valued and why?

### Growth Comparison
Which has stronger growth prospects?

### Risk Comparison
Which carries more risk and why?

### Recommendation
Which stock is the better investment right now and why?
Provide clear reasoning with specific data points.

---
*Disclaimer: AI-generated analysis for informational purposes only. Not financial advice.*
"""

    def _format_statement(self, data: dict) -> str:
        """Format financial statement data for prompt."""
        if not data.get("data"):
            return "Data not available."

        lines = []
        for metric, values in data["data"].items():
            vals = [f"{p}: {self._fmt_num(v)}" for p, v in values.items()]
            lines.append(f"- {metric}: {' | '.join(vals)}")

        return "\n".join(lines) if lines else "Data not available."

    @staticmethod
    def _fmt_pct(val) -> str:
        """Format percentage."""
        if val is None:
            return "N/A"
        try:
            return f"{float(val):.2%}"
        except (ValueError, TypeError):
            return str(val)

    @staticmethod
    def _fmt_num(val) -> str:
        """Format number with suffix."""
        if val is None:
            return "N/A"
        try:
            val = float(val)
        except (ValueError, TypeError):
            return str(val)
        abs_val = abs(val)
        if abs_val >= 1e12:
            return f"${val/1e12:.2f}T"
        elif abs_val >= 1e9:
            return f"${val/1e9:.2f}B"
        elif abs_val >= 1e6:
            return f"${val/1e6:.2f}M"
        elif abs_val >= 1e3:
            return f"${val/1e3:.1f}K"
        return f"${val:.2f}"

    def close(self):
        """Close resources."""
        if self._retriever:
            self._retriever.close()

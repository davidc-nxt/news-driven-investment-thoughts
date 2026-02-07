"""Financial fundamentals fetcher using yfinance (free, no API key required).

Inspired by virattt/dexter's financial tools, implemented using yfinance
instead of paid financialdatasets.ai API.
"""

from typing import Optional
import yfinance as yf
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


class FinancialDataService:
    """Fetch financial fundamentals, company profiles, and analyst data from yfinance."""

    def _get_ticker(self, symbol: str) -> yf.Ticker:
        """Get yfinance Ticker object."""
        return yf.Ticker(symbol.upper())

    # ─── Financial Statements ──────────────────────────────────────────

    def get_income_statement(
        self, symbol: str, period: str = "annual", limit: int = 4
    ) -> dict:
        """
        Fetch income statement data.

        Args:
            symbol: Stock ticker symbol
            period: 'annual' or 'quarterly'
            limit: Number of periods to return

        Returns:
            Dict with income statement data
        """
        ticker = self._get_ticker(symbol)
        try:
            if period == "quarterly":
                df = ticker.quarterly_income_stmt
            else:
                df = ticker.income_stmt

            if df is None or df.empty:
                return {"symbol": symbol, "period": period, "data": None}

            # Limit columns (each column is a period)
            df = df.iloc[:, :limit]

            # Extract key metrics
            metrics = {}
            key_rows = [
                "Total Revenue", "Cost Of Revenue", "Gross Profit",
                "Operating Income", "Net Income", "EBITDA",
                "Basic EPS", "Diluted EPS", "Total Expenses",
                "Operating Expense", "Interest Expense",
                "Tax Provision", "Research And Development",
            ]

            for row in key_rows:
                if row in df.index:
                    metrics[row] = {
                        str(col.date()): self._safe_float(df.loc[row, col])
                        for col in df.columns
                    }

            return {
                "symbol": symbol.upper(),
                "period": period,
                "periods": [str(col.date()) for col in df.columns],
                "data": metrics,
            }
        except Exception as e:
            console.print(f"[red]Error fetching income statement for {symbol}: {e}[/red]")
            return {"symbol": symbol, "period": period, "data": None, "error": str(e)}

    def get_balance_sheet(
        self, symbol: str, period: str = "annual", limit: int = 4
    ) -> dict:
        """Fetch balance sheet data."""
        ticker = self._get_ticker(symbol)
        try:
            if period == "quarterly":
                df = ticker.quarterly_balance_sheet
            else:
                df = ticker.balance_sheet

            if df is None or df.empty:
                return {"symbol": symbol, "period": period, "data": None}

            df = df.iloc[:, :limit]

            metrics = {}
            key_rows = [
                "Total Assets", "Total Liabilities Net Minority Interest",
                "Stockholders Equity", "Total Debt", "Cash And Cash Equivalents",
                "Current Assets", "Current Liabilities", "Net Tangible Assets",
                "Working Capital", "Invested Capital", "Total Capitalization",
                "Common Stock Equity", "Retained Earnings",
                "Long Term Debt", "Short Long Term Debt",
            ]

            for row in key_rows:
                if row in df.index:
                    metrics[row] = {
                        str(col.date()): self._safe_float(df.loc[row, col])
                        for col in df.columns
                    }

            return {
                "symbol": symbol.upper(),
                "period": period,
                "periods": [str(col.date()) for col in df.columns],
                "data": metrics,
            }
        except Exception as e:
            console.print(f"[red]Error fetching balance sheet for {symbol}: {e}[/red]")
            return {"symbol": symbol, "period": period, "data": None, "error": str(e)}

    def get_cash_flow(
        self, symbol: str, period: str = "annual", limit: int = 4
    ) -> dict:
        """Fetch cash flow statement data."""
        ticker = self._get_ticker(symbol)
        try:
            if period == "quarterly":
                df = ticker.quarterly_cashflow
            else:
                df = ticker.cashflow

            if df is None or df.empty:
                return {"symbol": symbol, "period": period, "data": None}

            df = df.iloc[:, :limit]

            metrics = {}
            key_rows = [
                "Operating Cash Flow", "Free Cash Flow",
                "Capital Expenditure", "Investing Cash Flow",
                "Financing Cash Flow", "End Cash Position",
                "Changes In Cash", "Repurchase Of Capital Stock",
                "Issuance Of Debt", "Repayment Of Debt",
                "Cash Dividends Paid",
            ]

            for row in key_rows:
                if row in df.index:
                    metrics[row] = {
                        str(col.date()): self._safe_float(df.loc[row, col])
                        for col in df.columns
                    }

            return {
                "symbol": symbol.upper(),
                "period": period,
                "periods": [str(col.date()) for col in df.columns],
                "data": metrics,
            }
        except Exception as e:
            console.print(f"[red]Error fetching cash flow for {symbol}: {e}[/red]")
            return {"symbol": symbol, "period": period, "data": None, "error": str(e)}

    # ─── Company Profile & Key Metrics ─────────────────────────────────

    def get_company_profile(self, symbol: str) -> dict:
        """
        Fetch company profile and metadata.

        Returns sector, industry, description, market cap, employees, etc.
        """
        ticker = self._get_ticker(symbol)
        try:
            info = ticker.info or {}
            return {
                "symbol": symbol.upper(),
                "name": info.get("longName") or info.get("shortName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "description": info.get("longBusinessSummary", "N/A"),
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "employees": info.get("fullTimeEmployees"),
                "website": info.get("website", "N/A"),
                "exchange": info.get("exchange", "N/A"),
                "currency": info.get("currency", "USD"),
                "country": info.get("country", "N/A"),
            }
        except Exception as e:
            return {"symbol": symbol, "error": str(e)}

    def get_key_metrics(self, symbol: str) -> dict:
        """
        Fetch key financial metrics and ratios.

        Returns P/E, P/B, dividend yield, beta, margins, etc.
        """
        ticker = self._get_ticker(symbol)
        try:
            info = ticker.info or {}
            return {
                "symbol": symbol.upper(),
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "market_cap": info.get("marketCap"),
                "pe_trailing": info.get("trailingPE"),
                "pe_forward": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
                "ev_to_revenue": info.get("enterpriseToRevenue"),
                "beta": info.get("beta"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "50d_avg": info.get("fiftyDayAverage"),
                "200d_avg": info.get("twoHundredDayAverage"),
                "dividend_yield": info.get("dividendYield"),
                "payout_ratio": info.get("payoutRatio"),
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "gross_margin": info.get("grossMargins"),
                "return_on_equity": info.get("returnOnEquity"),
                "return_on_assets": info.get("returnOnAssets"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
            }
        except Exception as e:
            return {"symbol": symbol, "error": str(e)}

    # ─── Insider Trades & Analyst Data ─────────────────────────────────

    def get_insider_trades(self, symbol: str) -> dict:
        """Fetch recent insider trading activity."""
        ticker = self._get_ticker(symbol)
        try:
            transactions = ticker.insider_transactions
            if transactions is None or (hasattr(transactions, 'empty') and transactions.empty):
                return {"symbol": symbol.upper(), "trades": []}

            trades = []
            df = transactions if isinstance(transactions, pd.DataFrame) else pd.DataFrame(transactions)
            for _, row in df.head(20).iterrows():
                trades.append({
                    "insider": str(row.get("Insider", row.get("Insider Trading", "N/A"))),
                    "relationship": str(row.get("Position", row.get("Relationship", "N/A"))),
                    "transaction": str(row.get("Text", row.get("Transaction", "N/A"))),
                    "shares": self._safe_float(row.get("Shares", 0)),
                    "value": self._safe_float(row.get("Value", 0)),
                    "date": str(row.get("Start Date", row.get("date", "N/A"))),
                })

            return {"symbol": symbol.upper(), "trades": trades}
        except Exception as e:
            return {"symbol": symbol, "trades": [], "error": str(e)}

    def get_analyst_recommendations(self, symbol: str) -> dict:
        """Fetch analyst recommendations and price targets."""
        ticker = self._get_ticker(symbol)
        result = {"symbol": symbol.upper()}

        try:
            # Recommendations
            recs = ticker.recommendations
            if recs is not None and not recs.empty:
                # Get the most recent recommendation summary
                recent = recs.tail(10)
                result["recommendations"] = []
                for _, row in recent.iterrows():
                    rec = {}
                    for col in recent.columns:
                        rec[col] = str(row[col]) if not pd.isna(row[col]) else None
                    result["recommendations"].append(rec)
            else:
                result["recommendations"] = []

            # Price targets
            info = ticker.info or {}
            result["target_high"] = info.get("targetHighPrice")
            result["target_low"] = info.get("targetLowPrice")
            result["target_mean"] = info.get("targetMeanPrice")
            result["target_median"] = info.get("targetMedianPrice")
            result["recommendation_key"] = info.get("recommendationKey")
            result["analyst_count"] = info.get("numberOfAnalystOpinions")

        except Exception as e:
            result["error"] = str(e)

        return result

    # ─── Display Helpers ───────────────────────────────────────────────

    def display_income_statement(self, data: dict):
        """Display income statement in a formatted table."""
        if not data.get("data"):
            console.print("[yellow]No income statement data available[/yellow]")
            return

        periods = data["periods"]
        table = Table(title=f"Income Statement - {data['symbol']} ({data['period']})")
        table.add_column("Metric", style="cyan", min_width=25)
        for p in periods:
            table.add_column(p, style="white", justify="right", min_width=14)

        for metric, values in data["data"].items():
            row = [metric]
            for p in periods:
                val = values.get(p)
                row.append(self._format_number(val) if val is not None else "—")
            table.add_row(*row)

        console.print(table)

    def display_balance_sheet(self, data: dict):
        """Display balance sheet in a formatted table."""
        if not data.get("data"):
            console.print("[yellow]No balance sheet data available[/yellow]")
            return

        periods = data["periods"]
        table = Table(title=f"Balance Sheet - {data['symbol']} ({data['period']})")
        table.add_column("Metric", style="cyan", min_width=25)
        for p in periods:
            table.add_column(p, style="white", justify="right", min_width=14)

        for metric, values in data["data"].items():
            row = [metric]
            for p in periods:
                val = values.get(p)
                row.append(self._format_number(val) if val is not None else "—")
            table.add_row(*row)

        console.print(table)

    def display_cash_flow(self, data: dict):
        """Display cash flow in a formatted table."""
        if not data.get("data"):
            console.print("[yellow]No cash flow data available[/yellow]")
            return

        periods = data["periods"]
        table = Table(title=f"Cash Flow Statement - {data['symbol']} ({data['period']})")
        table.add_column("Metric", style="cyan", min_width=25)
        for p in periods:
            table.add_column(p, style="white", justify="right", min_width=14)

        for metric, values in data["data"].items():
            row = [metric]
            for p in periods:
                val = values.get(p)
                row.append(self._format_number(val) if val is not None else "—")
            table.add_row(*row)

        console.print(table)

    def display_company_profile(self, data: dict):
        """Display company profile in a panel."""
        if data.get("error"):
            console.print(f"[red]Error: {data['error']}[/red]")
            return

        table = Table(title=f"Company Profile - {data.get('symbol', 'N/A')}")
        table.add_column("Field", style="cyan", min_width=18)
        table.add_column("Value", style="white")

        table.add_row("Name", str(data.get("name", "N/A")))
        table.add_row("Sector", str(data.get("sector", "N/A")))
        table.add_row("Industry", str(data.get("industry", "N/A")))
        table.add_row("Market Cap", self._format_number(data.get("market_cap")))
        table.add_row("Enterprise Value", self._format_number(data.get("enterprise_value")))
        table.add_row("Employees", f"{data.get('employees', 'N/A'):,}" if data.get("employees") else "N/A")
        table.add_row("Exchange", str(data.get("exchange", "N/A")))
        table.add_row("Country", str(data.get("country", "N/A")))
        table.add_row("Website", str(data.get("website", "N/A")))

        console.print(table)

        desc = data.get("description", "")
        if desc and desc != "N/A":
            console.print(Panel(desc[:500] + ("..." if len(desc) > 500 else ""), title="Description"))

    def display_key_metrics(self, data: dict):
        """Display key metrics in a formatted table."""
        if data.get("error"):
            console.print(f"[red]Error: {data['error']}[/red]")
            return

        # Valuation metrics
        val_table = Table(title=f"Valuation Metrics - {data.get('symbol', 'N/A')}")
        val_table.add_column("Metric", style="cyan", min_width=20)
        val_table.add_column("Value", style="white", justify="right")

        val_rows = [
            ("Price", data.get("price"), "${:.2f}"),
            ("Market Cap", data.get("market_cap"), None),
            ("P/E (Trailing)", data.get("pe_trailing"), "{:.2f}"),
            ("P/E (Forward)", data.get("pe_forward"), "{:.2f}"),
            ("PEG Ratio", data.get("peg_ratio"), "{:.2f}"),
            ("P/B Ratio", data.get("price_to_book"), "{:.2f}"),
            ("P/S Ratio", data.get("price_to_sales"), "{:.2f}"),
            ("EV/EBITDA", data.get("ev_to_ebitda"), "{:.2f}"),
            ("EV/Revenue", data.get("ev_to_revenue"), "{:.2f}"),
            ("Beta", data.get("beta"), "{:.2f}"),
            ("52W High", data.get("52w_high"), "${:.2f}"),
            ("52W Low", data.get("52w_low"), "${:.2f}"),
            ("Dividend Yield", data.get("dividend_yield"), "{:.2%}"),
        ]
        for label, val, fmt in val_rows:
            if val is not None:
                if fmt is None:
                    display = self._format_number(val)
                else:
                    try:
                        display = fmt.format(val)
                    except (ValueError, TypeError):
                        display = str(val)
            else:
                display = "—"
            val_table.add_row(label, display)

        console.print(val_table)

        # Profitability metrics
        prof_table = Table(title="Profitability & Financial Health")
        prof_table.add_column("Metric", style="cyan", min_width=20)
        prof_table.add_column("Value", style="white", justify="right")

        prof_rows = [
            ("Profit Margin", data.get("profit_margin"), "{:.2%}"),
            ("Operating Margin", data.get("operating_margin"), "{:.2%}"),
            ("Gross Margin", data.get("gross_margin"), "{:.2%}"),
            ("ROE", data.get("return_on_equity"), "{:.2%}"),
            ("ROA", data.get("return_on_assets"), "{:.2%}"),
            ("Debt/Equity", data.get("debt_to_equity"), "{:.2f}"),
            ("Current Ratio", data.get("current_ratio"), "{:.2f}"),
            ("Revenue Growth", data.get("revenue_growth"), "{:.2%}"),
            ("Earnings Growth", data.get("earnings_growth"), "{:.2%}"),
        ]
        for label, val, fmt in prof_rows:
            if val is not None:
                try:
                    display = fmt.format(val)
                except (ValueError, TypeError):
                    display = str(val)
            else:
                display = "—"
            prof_table.add_row(label, display)

        console.print(prof_table)

    def display_insider_trades(self, data: dict):
        """Display insider trades in a formatted table."""
        trades = data.get("trades", [])
        if not trades:
            console.print(f"[yellow]No insider trades found for {data.get('symbol', 'N/A')}[/yellow]")
            return

        table = Table(title=f"Insider Trades - {data.get('symbol', 'N/A')}")
        table.add_column("Date", style="cyan", width=12)
        table.add_column("Insider", style="white", max_width=20)
        table.add_column("Type", style="magenta", width=12)
        table.add_column("Shares", style="green", justify="right")
        table.add_column("Value", style="yellow", justify="right")

        for trade in trades:
            table.add_row(
                str(trade.get("date", "N/A"))[:10],
                str(trade.get("insider", "N/A"))[:20],
                str(trade.get("transaction", "N/A")),
                self._format_number(trade.get("shares")),
                self._format_number(trade.get("value")),
            )

        console.print(table)

    def display_analyst_recommendations(self, data: dict):
        """Display analyst recommendations."""
        table = Table(title=f"Analyst Consensus - {data.get('symbol', 'N/A')}")
        table.add_column("Metric", style="cyan", min_width=20)
        table.add_column("Value", style="white", justify="right")

        table.add_row("Recommendation", str(data.get("recommendation_key", "N/A")).upper())
        table.add_row("Analyst Count", str(data.get("analyst_count", "N/A")))
        table.add_row("Target High", f"${data['target_high']:.2f}" if data.get("target_high") else "—")
        table.add_row("Target Low", f"${data['target_low']:.2f}" if data.get("target_low") else "—")
        table.add_row("Target Mean", f"${data['target_mean']:.2f}" if data.get("target_mean") else "—")
        table.add_row("Target Median", f"${data['target_median']:.2f}" if data.get("target_median") else "—")

        console.print(table)

    # ─── Utilities ─────────────────────────────────────────────────────

    @staticmethod
    def _safe_float(val) -> Optional[float]:
        """Safely convert value to float."""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _format_number(val) -> str:
        """Format large numbers with suffixes."""
        if val is None:
            return "—"
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
        else:
            return f"${val:.2f}"

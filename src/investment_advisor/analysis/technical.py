"""Technical analysis module.

Computes technical indicators from market data for informed investment analysis.
Inspired by Dexter's analytical capabilities, using pure pandas/numpy.
"""

from typing import Optional
import pandas as pd
import numpy as np
import yfinance as yf
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


class TechnicalAnalyzer:
    """Compute and display technical analysis indicators."""

    def analyze(self, symbol: str, period: str = "6mo") -> dict:
        """
        Run full technical analysis on a ticker.

        Args:
            symbol: Stock ticker symbol
            period: Historical period (3mo, 6mo, 1y, 2y)

        Returns:
            Dict with all technical indicators
        """
        ticker = yf.Ticker(symbol.upper())
        df = ticker.history(period=period, interval="1d")

        if df is None or df.empty:
            return {"symbol": symbol, "error": "No data available"}

        result = {
            "symbol": symbol.upper(),
            "period": period,
            "data_points": len(df),
            "latest_price": float(df["Close"].iloc[-1]),
            "latest_date": str(df.index[-1].date()),
        }

        # Moving Averages
        result["moving_averages"] = self._moving_averages(df)

        # RSI
        result["rsi"] = self._rsi(df)

        # MACD
        result["macd"] = self._macd(df)

        # Bollinger Bands
        result["bollinger"] = self._bollinger_bands(df)

        # Volume Analysis
        result["volume"] = self._volume_analysis(df)

        # Support & Resistance
        result["levels"] = self._support_resistance(df)

        # Overall Signal
        result["signal"] = self._overall_signal(result)

        return result

    # ─── Indicator Calculations ────────────────────────────────────────

    def _moving_averages(self, df: pd.DataFrame) -> dict:
        """Calculate SMA and EMA."""
        close = df["Close"]
        current = float(close.iloc[-1])

        sma_20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None
        sma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
        ema_12 = float(close.ewm(span=12).mean().iloc[-1])
        ema_26 = float(close.ewm(span=26).mean().iloc[-1])

        # Signals
        signals = []
        if sma_50 and sma_200:
            if sma_50 > sma_200:
                signals.append("Golden Cross (Bullish)")
            else:
                signals.append("Death Cross (Bearish)")

        trend = "Neutral"
        above_count = sum(1 for ma in [sma_20, sma_50, sma_200] if ma and current > ma)
        if above_count >= 2:
            trend = "Bullish"
        elif above_count == 0 and sma_20:
            trend = "Bearish"

        return {
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "ema_12": ema_12,
            "ema_26": ema_26,
            "trend": trend,
            "signals": signals,
        }

    def _rsi(self, df: pd.DataFrame, period: int = 14) -> dict:
        """Calculate Relative Strength Index."""
        close = df["Close"]
        delta = close.diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None

        zone = "Neutral"
        if current_rsi is not None:
            if current_rsi > 70:
                zone = "Overbought"
            elif current_rsi < 30:
                zone = "Oversold"

        return {
            "value": round(current_rsi, 2) if current_rsi else None,
            "period": period,
            "zone": zone,
        }

    def _macd(self, df: pd.DataFrame) -> dict:
        """Calculate MACD (12/26/9)."""
        close = df["Close"]
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line

        current_macd = float(macd_line.iloc[-1])
        current_signal = float(signal_line.iloc[-1])
        current_hist = float(histogram.iloc[-1])

        # Check for crossover
        crossover = "None"
        if len(histogram) >= 2:
            prev_hist = float(histogram.iloc[-2])
            if prev_hist < 0 and current_hist > 0:
                crossover = "Bullish Crossover"
            elif prev_hist > 0 and current_hist < 0:
                crossover = "Bearish Crossover"

        return {
            "macd": round(current_macd, 4),
            "signal": round(current_signal, 4),
            "histogram": round(current_hist, 4),
            "crossover": crossover,
            "trend": "Bullish" if current_macd > current_signal else "Bearish",
        }

    def _bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> dict:
        """Calculate Bollinger Bands."""
        close = df["Close"]
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()

        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)

        current_price = float(close.iloc[-1])
        current_upper = float(upper.iloc[-1]) if not pd.isna(upper.iloc[-1]) else None
        current_lower = float(lower.iloc[-1]) if not pd.isna(lower.iloc[-1]) else None
        current_middle = float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else None

        # Bandwidth and %B
        bandwidth = None
        percent_b = None
        if current_upper and current_lower and current_middle:
            bandwidth = (current_upper - current_lower) / current_middle
            percent_b = (current_price - current_lower) / (current_upper - current_lower)

        position = "Middle"
        if percent_b is not None:
            if percent_b > 0.8:
                position = "Near Upper (potential resistance)"
            elif percent_b < 0.2:
                position = "Near Lower (potential support)"

        return {
            "upper": round(current_upper, 2) if current_upper else None,
            "middle": round(current_middle, 2) if current_middle else None,
            "lower": round(current_lower, 2) if current_lower else None,
            "bandwidth": round(bandwidth, 4) if bandwidth else None,
            "percent_b": round(percent_b, 4) if percent_b else None,
            "position": position,
        }

    def _volume_analysis(self, df: pd.DataFrame) -> dict:
        """Analyze volume patterns."""
        vol = df["Volume"]
        avg_20 = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else float(vol.mean())
        current_vol = float(vol.iloc[-1])
        ratio = current_vol / avg_20 if avg_20 > 0 else 1.0

        trend = "Normal"
        if ratio > 1.5:
            trend = "High Volume (strong conviction)"
        elif ratio < 0.5:
            trend = "Low Volume (weak conviction)"

        return {
            "current": int(current_vol),
            "avg_20d": int(avg_20),
            "ratio": round(ratio, 2),
            "trend": trend,
        }

    def _support_resistance(self, df: pd.DataFrame) -> dict:
        """Identify support and resistance levels."""
        recent_30 = df.tail(30)
        recent_high = float(recent_30["High"].max())
        recent_low = float(recent_30["Low"].min())

        # Pivot points
        last = df.iloc[-1]
        pivot = (float(last["High"]) + float(last["Low"]) + float(last["Close"])) / 3
        r1 = (2 * pivot) - float(last["Low"])
        s1 = (2 * pivot) - float(last["High"])
        r2 = pivot + (float(last["High"]) - float(last["Low"]))
        s2 = pivot - (float(last["High"]) - float(last["Low"]))

        return {
            "resistance_1": round(r1, 2),
            "resistance_2": round(r2, 2),
            "support_1": round(s1, 2),
            "support_2": round(s2, 2),
            "pivot": round(pivot, 2),
            "30d_high": round(recent_high, 2),
            "30d_low": round(recent_low, 2),
        }

    def _overall_signal(self, analysis: dict) -> dict:
        """Generate an overall technical signal from all indicators."""
        bullish = 0
        bearish = 0

        # Moving averages
        ma = analysis.get("moving_averages", {})
        if ma.get("trend") == "Bullish":
            bullish += 2
        elif ma.get("trend") == "Bearish":
            bearish += 2

        # RSI
        rsi = analysis.get("rsi", {})
        if rsi.get("zone") == "Oversold":
            bullish += 1
        elif rsi.get("zone") == "Overbought":
            bearish += 1

        # MACD
        macd = analysis.get("macd", {})
        if macd.get("trend") == "Bullish":
            bullish += 1
        else:
            bearish += 1
        if "Bullish" in macd.get("crossover", ""):
            bullish += 1
        elif "Bearish" in macd.get("crossover", ""):
            bearish += 1

        # Volume
        vol = analysis.get("volume", {})
        if vol.get("ratio", 1) > 1.2:
            if bullish > bearish:
                bullish += 1
            else:
                bearish += 1

        total_signals = bullish + bearish
        if total_signals == 0:
            return {"signal": "NEUTRAL", "strength": 0, "bullish_signals": 0, "bearish_signals": 0}

        score = (bullish - bearish) / total_signals
        if score > 0.3:
            signal = "BULLISH"
        elif score < -0.3:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        return {
            "signal": signal,
            "strength": round(abs(score) * 100, 1),
            "bullish_signals": bullish,
            "bearish_signals": bearish,
        }

    # ─── Display ───────────────────────────────────────────────────────

    def display_analysis(self, analysis: dict):
        """Display full technical analysis with Rich formatting."""
        if analysis.get("error"):
            console.print(f"[red]Error: {analysis['error']}[/red]")
            return

        symbol = analysis["symbol"]
        price = analysis["latest_price"]

        # Header
        signal = analysis.get("signal", {})
        signal_text = signal.get("signal", "NEUTRAL")
        signal_color = {"BULLISH": "green", "BEARISH": "red"}.get(signal_text, "yellow")
        console.print(Panel(
            f"[bold]{symbol}[/bold] @ ${price:.2f}  |  Signal: [{signal_color}]"
            f"{signal_text}[/{signal_color}] ({signal.get('strength', 0)}%)",
            title="[bold cyan]Technical Analysis[/bold cyan]",
            border_style="cyan",
        ))

        # Moving Averages
        ma = analysis.get("moving_averages", {})
        ma_table = Table(title="Moving Averages")
        ma_table.add_column("Indicator", style="cyan")
        ma_table.add_column("Value", style="white", justify="right")
        ma_table.add_column("vs Price", style="white", justify="right")

        for label, key in [("SMA 20", "sma_20"), ("SMA 50", "sma_50"), ("SMA 200", "sma_200"),
                           ("EMA 12", "ema_12"), ("EMA 26", "ema_26")]:
            val = ma.get(key)
            if val:
                diff = ((price - val) / val) * 100
                color = "green" if diff > 0 else "red"
                ma_table.add_row(label, f"${val:.2f}", f"[{color}]{diff:+.2f}%[/{color}]")
        ma_table.add_row("Trend", f"[bold]{ma.get('trend', 'N/A')}[/bold]", "")
        console.print(ma_table)

        # Momentum Indicators
        mom_table = Table(title="Momentum Indicators")
        mom_table.add_column("Indicator", style="cyan")
        mom_table.add_column("Value", style="white", justify="right")
        mom_table.add_column("Signal", style="white")

        rsi = analysis.get("rsi", {})
        rsi_val = rsi.get("value")
        rsi_color = "red" if rsi.get("zone") == "Overbought" else ("green" if rsi.get("zone") == "Oversold" else "yellow")
        mom_table.add_row("RSI (14)", f"[{rsi_color}]{rsi_val}[/{rsi_color}]" if rsi_val else "—", rsi.get("zone", "N/A"))

        macd = analysis.get("macd", {})
        mom_table.add_row("MACD", str(macd.get("macd", "—")), macd.get("trend", "N/A"))
        mom_table.add_row("MACD Signal", str(macd.get("signal", "—")), macd.get("crossover", "None"))
        mom_table.add_row("Histogram", str(macd.get("histogram", "—")), "")
        console.print(mom_table)

        # Bollinger Bands
        bb = analysis.get("bollinger", {})
        bb_table = Table(title="Bollinger Bands (20, 2)")
        bb_table.add_column("Band", style="cyan")
        bb_table.add_column("Value", style="white", justify="right")

        bb_table.add_row("Upper", f"${bb['upper']:.2f}" if bb.get("upper") else "—")
        bb_table.add_row("Middle", f"${bb['middle']:.2f}" if bb.get("middle") else "—")
        bb_table.add_row("Lower", f"${bb['lower']:.2f}" if bb.get("lower") else "—")
        bb_table.add_row("Position", bb.get("position", "N/A"))
        bb_table.add_row("%B", f"{bb['percent_b']:.2%}" if bb.get("percent_b") is not None else "—")
        console.print(bb_table)

        # Volume
        vol = analysis.get("volume", {})
        vol_table = Table(title="Volume Analysis")
        vol_table.add_column("Metric", style="cyan")
        vol_table.add_column("Value", style="white", justify="right")

        vol_table.add_row("Current Volume", f"{vol.get('current', 0):,}")
        vol_table.add_row("20D Avg Volume", f"{vol.get('avg_20d', 0):,}")
        vol_table.add_row("Vol Ratio", f"{vol.get('ratio', 0):.2f}x")
        vol_table.add_row("Assessment", vol.get("trend", "N/A"))
        console.print(vol_table)

        # Support / Resistance
        levels = analysis.get("levels", {})
        lvl_table = Table(title="Support & Resistance")
        lvl_table.add_column("Level", style="cyan")
        lvl_table.add_column("Price", style="white", justify="right")

        lvl_table.add_row("[red]Resistance 2[/red]", f"${levels.get('resistance_2', 0):.2f}")
        lvl_table.add_row("[red]Resistance 1[/red]", f"${levels.get('resistance_1', 0):.2f}")
        lvl_table.add_row("[yellow]Pivot[/yellow]", f"${levels.get('pivot', 0):.2f}")
        lvl_table.add_row("[green]Support 1[/green]", f"${levels.get('support_1', 0):.2f}")
        lvl_table.add_row("[green]Support 2[/green]", f"${levels.get('support_2', 0):.2f}")
        lvl_table.add_row("30D High", f"${levels.get('30d_high', 0):.2f}")
        lvl_table.add_row("30D Low", f"${levels.get('30d_low', 0):.2f}")
        console.print(lvl_table)

    def generate_summary(self, analysis: dict) -> str:
        """Generate a text summary of technical analysis for LLM context."""
        if analysis.get("error"):
            return f"Technical analysis unavailable: {analysis['error']}"

        parts = [f"**Technical Analysis for {analysis['symbol']}** (${analysis['latest_price']:.2f})"]

        # Signal
        sig = analysis.get("signal", {})
        parts.append(f"Overall Signal: {sig.get('signal', 'NEUTRAL')} "
                     f"(Strength: {sig.get('strength', 0)}%, "
                     f"Bullish: {sig.get('bullish_signals', 0)}, Bearish: {sig.get('bearish_signals', 0)})")

        # MAs
        ma = analysis.get("moving_averages", {})
        parts.append(f"Moving Average Trend: {ma.get('trend', 'N/A')}")
        for label, key in [("SMA 20", "sma_20"), ("SMA 50", "sma_50"), ("SMA 200", "sma_200")]:
            val = ma.get(key)
            if val:
                parts.append(f"  {label}: ${val:.2f}")

        # RSI
        rsi = analysis.get("rsi", {})
        parts.append(f"RSI (14): {rsi.get('value', 'N/A')} - {rsi.get('zone', 'N/A')}")

        # MACD
        macd = analysis.get("macd", {})
        parts.append(f"MACD: {macd.get('trend', 'N/A')}, Crossover: {macd.get('crossover', 'None')}")

        # Bollinger
        bb = analysis.get("bollinger", {})
        parts.append(f"Bollinger Position: {bb.get('position', 'N/A')}")

        # Volume
        vol = analysis.get("volume", {})
        parts.append(f"Volume: {vol.get('ratio', 0):.2f}x avg - {vol.get('trend', 'N/A')}")

        # Levels
        levels = analysis.get("levels", {})
        parts.append(f"Support: ${levels.get('support_1', 0):.2f} / ${levels.get('support_2', 0):.2f}")
        parts.append(f"Resistance: ${levels.get('resistance_1', 0):.2f} / ${levels.get('resistance_2', 0):.2f}")

        return "\n".join(parts)

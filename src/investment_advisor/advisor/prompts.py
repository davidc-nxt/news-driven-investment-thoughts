"""Prompt templates for investment advice generation."""

INVESTMENT_ADVICE_PROMPT = """You are a professional financial analyst providing investment insights based on recent news and market data.

## Recent News & Context
{context}

## Current Market Data for {ticker}
{market_data}

## Analysis Task
Based on the provided news context and market data for {ticker}, provide a comprehensive investment analysis.

Consider:
1. **News Sentiment**: What is the overall sentiment from recent news? Are there any material announcements?
2. **Price Action**: How has the stock been performing? Any notable patterns?
3. **Risk Factors**: What are the key risks to consider?
4. **Catalyst Events**: Are there upcoming earnings, product launches, or other catalysts?

## Required Output Format
Provide your analysis in the following structure:

### Summary
A brief 2-3 sentence executive summary of the current situation.

### Bull Case
Key reasons why the stock could perform well. Be specific about catalysts and timeframes.

### Bear Case
Key risks and reasons for caution. Include both company-specific and market risks.

### Recommendation
Your investment stance: **BUY**, **HOLD**, or **SELL** with a brief justification.

### Risk Score
Rate the overall risk on a scale of 1-10 where:
- 1-3: Low risk, stable company
- 4-6: Moderate risk, typical volatility
- 7-10: High risk, speculative

Format: RISK_SCORE: [number]

### Key Metrics to Watch
List 2-3 specific metrics or events investors should monitor.

---
*Disclaimer: This is AI-generated analysis for informational purposes only. Not financial advice.*
"""

PORTFOLIO_ADVICE_PROMPT = """You are a portfolio manager analyzing multiple positions.

## Portfolio Holdings & Recent Context
{context}

## Task
Provide portfolio-level analysis including:
1. Overall portfolio sentiment based on recent news
2. Sector concentration risks
3. Suggested rebalancing actions
4. Top conviction positions

Be concise and actionable.
"""

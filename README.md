# News-Driven Investment Thoughts

> **A RAG-powered financial analysis CLI** that combines real-time news retrieval, vector similarity search, deep financial research, and LLM reasoning to generate investment insights — all using **free data sources**.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue)
![pgvector](https://img.shields.io/badge/pgvector-0.5+-green)
![LangChain](https://img.shields.io/badge/LangChain-1.2+-orange)
![yfinance](https://img.shields.io/badge/yfinance-Free%20Data-brightgreen)

## 🎯 What This Project Showcases

A complete **Retrieval-Augmented Generation (RAG)** pipeline for financial analysis, enhanced with **autonomous multi-step research** capabilities inspired by [virattt/dexter](https://github.com/virattt/dexter).

| Component | Technology | Description |
|-----------|------------|-------------|
| **Data Ingestion** | yfinance | Live news, market data, fundamentals, insider trades |
| **Vector Database** | PostgreSQL + pgvector | 384-dim embeddings with native cosine search |
| **Embeddings** | sentence-transformers | Local, free, fast (all-MiniLM-L6-v2) |
| **Technical Analysis** | pandas + numpy | RSI, MACD, Bollinger Bands, SMA/EMA, volume |
| **Research Agent** | Multi-step pipeline | 6-step autonomous analysis → LLM synthesis |
| **LLM Integration** | OpenRouter | Structured research reports & investment advice |

### Key Differentiators

- **100% Free Data** — Uses yfinance instead of paid APIs (no financialdatasets.ai key needed)
- **Local Embeddings** — sentence-transformers runs on-device (no OpenAI embedding costs)
- **Autonomous Research** — Dexter-inspired multi-step analysis with self-reflection
- **Pure Python Technicals** — RSI, MACD, Bollinger computed with pandas/numpy (no TA-Lib)

---

## 📐 Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        Investment Advisor CLI (10 commands)              │
├──────────────────────────────────────────────────────────────────────────┤
│  Data Layer              │  RAG Pipeline           │  Intelligence       │
│  ─────────               │  ────────────           │  ────────────       │
│  yfinance API            │  Document Chunker       │  Prompt Templates   │
│  ├─ News Articles        │  ↓                      │  ↓                  │
│  ├─ Market Data (OHLCV)  │  Embedding Service      │  Context Builder    │
│  ├─ Fundamentals         │  (sentence-transformers)│  ↓                  │
│  │  ├─ Income Statement  │  ↓                      │  OpenRouter LLM     │
│  │  ├─ Balance Sheet     │  pgvector Store         │  (GPT-4o-mini)      │
│  │  └─ Cash Flow         │  (384-dim vectors)      │  ↓                  │
│  ├─ Company Profile      │  ↓                      │  Investment Advice  │
│  ├─ Analyst Consensus    │  Semantic Retriever     │  Research Reports   │
│  ├─ Insider Trades       │  (<=> cosine search)    │  Comparisons        │
│  └─ Technical Indicators │                         │                     │
│     ├─ RSI (14)          │                         │                     │
│     ├─ MACD (12/26/9)    │                         │                     │
│     ├─ Bollinger Bands   │                         │                     │
│     ├─ SMA (20/50/200)   │                         │                     │
│     └─ Support/Resistance│                         │                     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker (for pgvector database)
- OpenRouter API key ([get one free](https://openrouter.ai))

### 1. Clone and Setup

```bash
git clone https://github.com/davidc-nxt/news-driven-investment-thoughts.git
cd news-driven-investment-thoughts

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
```

### 2. Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit .env and add your OpenRouter API key
# OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### 3. Start pgvector Database

```bash
docker compose up -d
```

### 4. Run the CLI

```bash
# Check system status
invest status

# Fetch news and market data
invest fetch --all

# Semantic search
invest search "AI chip demand"

# Generate investment advice
invest advise NVDA

# Deep multi-step research (Dexter-inspired)
invest research NVDA

# Side-by-side comparison
invest compare NVDA AMD

# Financial fundamentals
invest fundamentals AAPL --report income --period quarterly

# Technical analysis
invest technical AAPL --period 6mo
```

---

## 📊 CLI Commands

| Command | Description | Requires DB | Requires LLM |
|---------|-------------|:-----------:|:------------:|
| `invest status` | System status & database stats | ✓ | |
| `invest tickers list` | Manage tracked tickers | ✓ | |
| `invest fetch -t TICKER` | Fetch news & market data | ✓ | |
| `invest index` | Index articles for semantic search | ✓ | |
| `invest search "query"` | Semantic search (pgvector) | ✓ | |
| `invest advise TICKER` | Quick AI investment advice | ✓ | ✓ |
| `invest research TICKER` | **Deep 6-step analysis** | optional | ✓ |
| `invest compare A B C` | **Side-by-side comparison** | | ✓ |
| `invest fundamentals TICKER` | **Financial statements & metrics** | | |
| `invest technical TICKER` | **Technical indicators** | | |

> **Note:** Commands marked "optional" for DB will gracefully degrade without it. `fundamentals` and `technical` work standalone with just yfinance.

---

## 🔬 Deep Research Agent (Dexter-Inspired)

The `invest research` command performs autonomous 6-step analysis:

```
Step 1/6: Company profile → understand the business
Step 2/6: Key metrics → valuation & profitability
Step 3/6: Financial statements → income, balance, cash flow
Step 4/6: Technical analysis → RSI, MACD, Bollinger, SMA
Step 5/6: News context → RAG search for recent developments
Step 6/6: Analyst consensus → recommendations & insider activity
           ↓
   LLM Synthesis → Comprehensive Research Report
```

**Output includes:**
- Executive Summary
- Financial Analysis (revenue, profitability, balance sheet)
- Valuation Assessment (relative to peers and historical)
- Technical Outlook (momentum, support/resistance)
- Catalyst Tracker (upcoming events)
- Risk Assessment (1-10 score)
- Investment Recommendation (BUY/HOLD/SELL with conviction level)

---

## 📈 Technical Analysis

The `invest technical` command computes indicators using **pure pandas/numpy** (no external TA libraries):

| Indicator | Details |
|-----------|---------|
| **RSI** | 14-period Relative Strength Index with overbought/oversold zones |
| **MACD** | 12/26/9 Moving Average Convergence Divergence with crossover detection |
| **Bollinger Bands** | 20-period, 2σ with %B and bandwidth analysis |
| **SMA** | 20, 50, 200-day Simple Moving Averages with trend signal |
| **EMA** | 12, 26-day Exponential Moving Averages |
| **Volume** | Current vs 20-day average with conviction assessment |
| **Support/Resistance** | Pivot points (R1/R2, S1/S2) + 30-day high/low |
| **Overall Signal** | Composite BULLISH/NEUTRAL/BEARISH with strength % |

**Example Output:**
```
╭──────────────── Technical Analysis ─────────────────╮
│ AAPL @ $278.12  |  Signal: BULLISH (50.0%)          │
╰─────────────────────────────────────────────────────╯
         Moving Averages
┏━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃ Indicator ┃   Value ┃ vs Price ┃
┡━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│ SMA 20    │ $260.15 │   +6.91% │
│ SMA 50    │ $268.70 │   +3.50% │
│ EMA 12    │ $266.89 │   +4.21% │
│ Trend     │ Bullish │          │
└───────────┴─────────┴──────────┘
```

---

## 💰 Financial Fundamentals

The `invest fundamentals` command provides 7 report types:

```bash
invest fundamentals NVDA --report profile    # Company overview
invest fundamentals NVDA --report metrics    # P/E, P/B, ROE, margins
invest fundamentals NVDA --report income     # Income statement
invest fundamentals NVDA --report balance    # Balance sheet
invest fundamentals NVDA --report cashflow   # Cash flow statement
invest fundamentals NVDA --report analyst    # Analyst consensus
invest fundamentals NVDA --report insider    # Insider trading activity
invest fundamentals NVDA --report all        # Everything (default)
```

Add `--period quarterly` for quarterly data on financial statements.

---

## 🔍 RAG Pipeline Details

### Document Chunking & Embedding

```python
# Articles split into semantic chunks
chunk_size=500, chunk_overlap=50

# Embeddings via sentence-transformers (runs locally, free)
model: all-MiniLM-L6-v2
dimensions: 384
```

### Vector Storage (pgvector)

```sql
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    article_id INTEGER,
    chunk_text TEXT,
    embedding vector(384),  -- Native pgvector type
    chunk_metadata JSONB
);
```

### Semantic Search

```sql
-- pgvector cosine distance operator
SELECT chunk_text,
       1 - (embedding <=> query_vector) as similarity
FROM embeddings
ORDER BY embedding <=> query_vector
LIMIT 5;
```

---

## 📁 Project Structure

```
news-driven-investment-thoughts/
├── src/investment_advisor/
│   ├── cli.py                # Typer CLI (10 commands)
│   ├── config.py             # Pydantic settings
│   ├── db/
│   │   ├── connection.py     # SQLAlchemy engine
│   │   └── models.py         # ORM models (pgvector type)
│   ├── ingestion/
│   │   ├── news.py           # yfinance news fetcher
│   │   ├── market_data.py    # OHLCV data fetcher
│   │   └── fundamentals.py   # Financial data service (7 endpoints)
│   ├── rag/
│   │   ├── embeddings.py     # Embedding service
│   │   ├── chunker.py        # Document splitter
│   │   └── retriever.py      # pgvector semantic search
│   ├── analysis/
│   │   └── technical.py      # Technical indicators (RSI, MACD, etc.)
│   └── advisor/
│       ├── prompts.py        # LLM prompt templates
│       ├── generator.py      # Investment advice generator
│       └── researcher.py     # Multi-step research agent
├── tests/
│   └── test_e2e.py           # End-to-end tests
├── docker-compose.yml        # pgvector container
├── init_pgvector.sql         # Database schema
└── pyproject.toml            # Dependencies
```

---

## 🧪 Running Tests

```bash
source venv/bin/activate
python tests/test_e2e.py
```

**Expected Output:**
```
           E2E Test Results Summary
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Test              ┃ Execution ┃ Validation ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ System Status     │ ✓         │ ✓          │
│ List Tickers      │ ✓         │ ✓          │
│ Fetch AAPL Data   │ ✓         │ ✓          │
│ Semantic Search   │ ✓         │ ✓          │
│ Investment Advice │ ✓         │ ✓          │
│ Filtered Search   │ ✓         │ ✓          │
└───────────────────┴───────────┴────────────┘

✓ All 6 tests passed!
```

---

## 🔑 Key Technologies

| Technology | Purpose |
|------------|---------|
| **pgvector** | Native PostgreSQL vector operations with HNSW/IVFFlat indexing |
| **LangChain** | RAG orchestration, text splitting, embedding abstraction |
| **sentence-transformers** | Free, fast, local embeddings (no API key needed) |
| **yfinance** | Free financial data — news, fundamentals, technicals, insider trades |
| **OpenRouter** | Unified LLM API (GPT-4o-mini, Claude, Grok, etc.) |
| **Typer + Rich** | Beautiful CLI with tables, panels, and progress bars |
| **SQLAlchemy** | ORM with pgvector type support |
| **pandas + numpy** | Technical indicator computation (no TA-Lib dependency) |

---

## 🙏 Acknowledgments

- [pgvector](https://github.com/pgvector/pgvector) — Vector similarity for PostgreSQL
- [LangChain](https://github.com/langchain-ai/langchain) — RAG framework
- [sentence-transformers](https://www.sbert.net/) — Embedding models
- [OpenRouter](https://openrouter.ai) — LLM API gateway
- [virattt/dexter](https://github.com/virattt/dexter) — Inspiration for multi-step research agent architecture

---

## 📄 License

MIT License — feel free to use this as a learning resource or starting point for your own projects.

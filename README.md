# News-Driven Investment Thoughts

> **A RAG-powered financial analysis CLI** that combines real-time news retrieval, vector similarity search, and LLM reasoning to generate investment insights.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue)
![pgvector](https://img.shields.io/badge/pgvector-0.5+-green)
![LangChain](https://img.shields.io/badge/LangChain-1.2+-orange)

## 🎯 What This Project Showcases

This project demonstrates a complete **Retrieval-Augmented Generation (RAG)** pipeline for financial analysis:

| Component | Technology | Description |
|-----------|------------|-------------|
| **Data Ingestion** | yfinance | Fetches live news and market data |
| **Vector Database** | PostgreSQL + pgvector | Stores 384-dim embeddings in native vector columns |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | Local, free, fast embedding generation |
| **Vector Search** | pgvector `<=>` operator | Native cosine similarity with O(log n) indexing |
| **Ranking** | Similarity scores 0-1 | Results ordered by semantic relevance |
| **LLM Integration** | OpenRouter (GPT-4o-mini) | Structured investment advice generation |

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Investment Advisor CLI                        │
├─────────────────────────────────────────────────────────────────────┤
│  Data Layer           │  RAG Pipeline          │  Intelligence      │
│  ──────────────       │  ────────────          │  ────────────      │
│  yfinance API         │  Document Chunker      │  Prompt Templates  │
│  ↓                    │  ↓                     │  ↓                 │
│  News Articles        │  Embedding Service     │  Context Builder   │
│  Market Data          │  (sentence-transformers)│  ↓                 │
│  ↓                    │  ↓                     │  OpenRouter LLM    │
│  PostgreSQL           │  pgvector Store        │  (GPT-4o-mini)     │
│  (articles, tickers)  │  (384-dim vectors)     │  ↓                 │
│                       │  ↓                     │  Investment Advice │
│                       │  Semantic Retriever    │  (BUY/HOLD/SELL)   │
│                       │  (<=> cosine search)   │                    │
└─────────────────────────────────────────────────────────────────────┘
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
```

---

## 🔍 How It Works

### 1. Data Ingestion

The CLI fetches financial data from Yahoo Finance:

```python
# Fetches news articles and OHLCV market data
invest fetch -t NVDA
# → 10 articles stored
# → 21 market data points stored
# → 14 vector chunks indexed
```

### 2. Document Chunking & Embedding

Articles are split into semantic chunks and converted to 384-dimensional vectors:

```python
# Uses LangChain's RecursiveCharacterTextSplitter
chunk_size=500, chunk_overlap=50

# Embeddings via sentence-transformers (runs locally)
model: all-MiniLM-L6-v2
dimensions: 384
```

### 3. Vector Storage (pgvector)

Embeddings are stored in PostgreSQL using the native `vector` type:

```sql
-- Schema
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    article_id INTEGER,
    chunk_text TEXT,
    embedding vector(384),  -- Native pgvector type
    chunk_metadata JSONB
);
```

### 4. Semantic Search

Queries are embedded and matched using cosine similarity:

```sql
-- Uses pgvector's <=> cosine distance operator
SELECT chunk_text, 
       1 - (embedding <=> query_vector) as similarity
FROM embeddings
ORDER BY embedding <=> query_vector
LIMIT 5;
```

**Example Output:**
```
┏━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Score    ┃ Ticker   ┃ Source          ┃ Text Preview                      ┃
┡━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 0.612    │ NVDA     │ Motley Fool     │ A Once-in-a-Decade Investment...  │
│ 0.520    │ NVDA     │ Yahoo Finance   │ Could there be risks to AI chip...│
│ 0.447    │ NVDA     │ Simply Wall St. │ Intel is in discussions with...   │
└──────────┴──────────┴─────────────────┴───────────────────────────────────┘
```

### 5. LLM-Powered Advice Generation

Retrieved context is passed to GPT-4o-mini via OpenRouter:

```python
# Prompt includes: news context + market data + structured output format
response = llm.invoke([HumanMessage(content=prompt)])

# Output includes:
# - Summary
# - Bull Case (3 points)
# - Bear Case (3 points)
# - Recommendation (BUY/HOLD/SELL)
# - Risk Score (1-10)
```

---

## 📁 Project Structure

```
news-driven-investment-thoughts/
├── src/investment_advisor/
│   ├── cli.py              # Typer CLI commands
│   ├── config.py           # Pydantic settings
│   ├── db/
│   │   ├── connection.py   # SQLAlchemy engine
│   │   └── models.py       # ORM models (Vector type)
│   ├── ingestion/
│   │   ├── news.py         # yfinance news fetcher
│   │   └── market_data.py  # OHLCV data fetcher
│   ├── rag/
│   │   ├── embeddings.py   # Embedding service
│   │   ├── chunker.py      # Document splitter
│   │   └── retriever.py    # pgvector search
│   └── advisor/
│       ├── prompts.py      # LLM prompt templates
│       └── generator.py    # Advice generator
├── tests/
│   └── test_e2e.py         # End-to-end tests
├── docker-compose.yml      # pgvector container
├── init_pgvector.sql       # Database schema
└── pyproject.toml          # Dependencies
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
| **OpenRouter** | Unified LLM API with access to GPT-4, Claude, etc. |
| **Typer + Rich** | Beautiful CLI with tables, panels, and progress bars |
| **SQLAlchemy** | ORM with pgvector type support |

---

## 📄 License

MIT License - feel free to use this as a learning resource or starting point for your own projects.

---

## 🙏 Acknowledgments

- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity for PostgreSQL
- [LangChain](https://github.com/langchain-ai/langchain) - RAG framework
- [sentence-transformers](https://www.sbert.net/) - Embedding models
- [OpenRouter](https://openrouter.ai) - LLM API gateway

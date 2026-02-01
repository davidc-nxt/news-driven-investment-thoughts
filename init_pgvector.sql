-- Initialize pgvector extension and create schema
CREATE EXTENSION IF NOT EXISTS vector;

-- Tracked tickers
CREATE TABLE IF NOT EXISTS tickers (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(255),
    sector VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Raw news articles
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    ticker_symbol VARCHAR(10) NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    summary TEXT,
    published_at TIMESTAMPTZ,
    source VARCHAR(100),
    url TEXT UNIQUE,
    article_type VARCHAR(50) DEFAULT 'news',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_articles_ticker ON articles(ticker_symbol);
CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at DESC);

-- Market data (daily OHLCV)
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    ticker_symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open_price DECIMAL(12, 4),
    high_price DECIMAL(12, 4),
    low_price DECIMAL(12, 4),
    close_price DECIMAL(12, 4),
    adj_close DECIMAL(12, 4),
    volume BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ticker_symbol, date)
);

CREATE INDEX IF NOT EXISTS idx_market_data_ticker_date ON market_data(ticker_symbol, date DESC);

-- Embeddings for semantic search (native pgvector type - 384 dims for MiniLM-L6-v2)
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER DEFAULT 0,
    embedding vector(384),  -- Native pgvector type!
    chunk_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- IVFFlat index for fast similarity search (created after some data exists)
-- CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings 
-- USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_embeddings_article ON embeddings(article_id);

-- Investment advice history
CREATE TABLE IF NOT EXISTS advice_history (
    id SERIAL PRIMARY KEY,
    ticker_symbol VARCHAR(10),
    query TEXT,
    context_summary TEXT,
    advice TEXT NOT NULL,
    recommendation VARCHAR(20),
    risk_score INTEGER,
    model_used VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_advice_ticker ON advice_history(ticker_symbol);

-- Insert some default tickers
INSERT INTO tickers (symbol, name, sector) VALUES
    ('AAPL', 'Apple Inc.', 'Technology'),
    ('MSFT', 'Microsoft Corporation', 'Technology'),
    ('GOOGL', 'Alphabet Inc.', 'Technology'),
    ('NVDA', 'NVIDIA Corporation', 'Technology'),
    ('TSLA', 'Tesla Inc.', 'Automotive'),
    ('AMZN', 'Amazon.com Inc.', 'Consumer Cyclical'),
    ('META', 'Meta Platforms Inc.', 'Technology'),
    ('JPM', 'JPMorgan Chase & Co.', 'Financial Services'),
    ('V', 'Visa Inc.', 'Financial Services'),
    ('UNH', 'UnitedHealth Group Inc.', 'Healthcare')
ON CONFLICT (symbol) DO NOTHING;

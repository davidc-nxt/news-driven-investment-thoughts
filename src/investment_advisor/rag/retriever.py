"""Semantic retriever using LangChain with native pgvector similarity search."""

from typing import Optional
from sqlalchemy import select, text
from langchain_core.documents import Document
from rich.console import Console
from rich.table import Table

from investment_advisor.db.connection import get_session
from investment_advisor.db.models import Article, Embedding
from investment_advisor.rag.embeddings import EmbeddingService
from investment_advisor.rag.chunker import DocumentChunker
from investment_advisor.config import get_settings

console = Console()


class SemanticRetriever:
    """Semantic search and retrieval using LangChain with native pgvector."""

    def __init__(self):
        self.settings = get_settings()
        self.session = get_session()
        self.embedding_service = EmbeddingService()
        self.chunker = DocumentChunker()

    def index_article(self, article: Article) -> int:
        """
        Index an article by creating embeddings for its chunks.

        Args:
            article: Article to index

        Returns:
            Number of chunks indexed
        """
        # Check if article is already indexed
        existing = self.session.execute(
            select(Embedding).where(Embedding.article_id == article.id)
        ).scalars().first()

        if existing:
            console.print(f"[yellow]Article {article.id} already indexed, skipping[/yellow]")
            return 0

        # Chunk the article
        chunks = self.chunker.prepare_article_for_chunking(
            article_id=article.id,
            ticker=article.ticker_symbol,
            title=article.title,
            content=article.content or "",
            source=article.source,
            published_at=article.published_at,
        )

        if not chunks:
            return 0

        # Generate embeddings
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(texts)

        # Store embeddings with native pgvector type
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            emb = Embedding(
                article_id=article.id,
                chunk_text=chunk.page_content,
                chunk_index=i,
                embedding=embedding,  # pgvector handles this natively!
                chunk_metadata=chunk.metadata,
            )
            self.session.add(emb)

        self.session.commit()
        console.print(f"[green]Indexed {len(chunks)} chunks for article {article.id}[/green]")
        return len(chunks)

    def index_all_unindexed(self) -> int:
        """
        Index all articles that haven't been indexed yet.

        Returns:
            Total number of chunks indexed
        """
        # Find articles without embeddings using a subquery
        indexed_ids = select(Embedding.article_id).distinct()
        query = select(Article).where(Article.id.not_in(indexed_ids))

        articles = self.session.execute(query).scalars().all()

        total_chunks = 0
        for article in articles:
            total_chunks += self.index_article(article)

        console.print(f"[blue]Total: Indexed {total_chunks} chunks from {len(articles)} articles[/blue]")
        return total_chunks

    def _format_vector_for_pg(self, embedding: list[float]) -> str:
        """Format a Python list as a PostgreSQL vector literal."""
        return '[' + ','.join(str(x) for x in embedding) + ']'

    def search(
        self,
        query: str,
        ticker: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Perform semantic search using native pgvector cosine similarity.

        This uses PostgreSQL's vector operations for fast similarity search,
        which is much more efficient than computing similarities in Python.

        Args:
            query: Search query
            ticker: Optional ticker to filter by
            top_k: Number of results (default from settings)

        Returns:
            List of results with text, metadata, and similarity score
        """
        top_k = top_k or self.settings.top_k_results

        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)
        
        # Format as PostgreSQL vector literal
        vector_literal = self._format_vector_for_pg(query_embedding)

        # Build SQL with native pgvector cosine distance operator
        # <=> is pgvector's cosine distance operator (1 - similarity)
        if ticker:
            sql = text(f"""
                SELECT 
                    e.id,
                    e.chunk_text,
                    e.chunk_metadata,
                    e.article_id,
                    1 - (e.embedding <=> '{vector_literal}'::vector) as similarity
                FROM embeddings e
                JOIN articles a ON e.article_id = a.id
                WHERE a.ticker_symbol = :ticker
                ORDER BY e.embedding <=> '{vector_literal}'::vector
                LIMIT :limit
            """)
            result = self.session.execute(
                sql, 
                {"ticker": ticker.upper(), "limit": top_k}
            )
        else:
            sql = text(f"""
                SELECT 
                    e.id,
                    e.chunk_text,
                    e.chunk_metadata,
                    e.article_id,
                    1 - (e.embedding <=> '{vector_literal}'::vector) as similarity
                FROM embeddings e
                ORDER BY e.embedding <=> '{vector_literal}'::vector
                LIMIT :limit
            """)
            result = self.session.execute(
                sql, 
                {"limit": top_k}
            )

        # Format results
        results = []
        for row in result:
            results.append({
                "id": row.id,
                "text": row.chunk_text,
                "metadata": row.chunk_metadata or {},
                "article_id": row.article_id,
                "similarity": float(row.similarity),
            })

        return results

    def display_search_results(self, results: list[dict]):
        """Display search results in a formatted table."""
        if not results:
            console.print("[yellow]No results found[/yellow]")
            return

        table = Table(title="Search Results (pgvector)")
        table.add_column("Score", style="cyan", width=8)
        table.add_column("Ticker", style="magenta", width=8)
        table.add_column("Source", style="green", width=15)
        table.add_column("Text Preview", style="white")

        for r in results:
            metadata = r.get("metadata", {})
            ticker = metadata.get("ticker", "N/A")
            source = metadata.get("source", "N/A")
            text_preview = r["text"][:100] + "..." if len(r["text"]) > 100 else r["text"]
            score = f"{r['similarity']:.3f}"

            table.add_row(score, ticker, source, text_preview)

        console.print(table)

    def get_context_for_ticker(self, ticker: str, query: str = "recent developments") -> str:
        """
        Get relevant context for a ticker to use in advice generation.

        Args:
            ticker: Stock ticker symbol
            query: Query to find relevant context

        Returns:
            Concatenated context string
        """
        results = self.search(query, ticker=ticker, top_k=10)

        if not results:
            return "No recent news or context available."

        context_parts = []
        for i, r in enumerate(results, 1):
            metadata = r.get("metadata", {})
            source = metadata.get("source", "Unknown")
            date = metadata.get("published_at", "N/A")
            context_parts.append(f"[{i}] ({source}, {date}): {r['text']}")

        return "\n\n".join(context_parts)

    def close(self):
        """Close database session."""
        self.session.close()

"""Document chunking using LangChain text splitters."""

from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rich.console import Console

from investment_advisor.config import get_settings

console = Console()


class DocumentChunker:
    """Chunk documents for embedding using LangChain text splitters."""

    def __init__(self):
        self.settings = get_settings()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_text(self, text: str, metadata: Optional[dict] = None) -> list[Document]:
        """
        Split text into chunks with metadata.

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of LangChain Document objects
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        chunks = self.splitter.create_documents(
            texts=[text],
            metadatas=[metadata],
        )

        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks

    def chunk_documents(self, documents: list[dict]) -> list[Document]:
        """
        Chunk multiple documents.

        Args:
            documents: List of dicts with 'text' and optional 'metadata' keys

        Returns:
            List of chunked Document objects
        """
        all_chunks = []

        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)

        console.print(f"[blue]Created {len(all_chunks)} chunks from {len(documents)} documents[/blue]")
        return all_chunks

    def prepare_article_for_chunking(
        self,
        article_id: int,
        ticker: str,
        title: str,
        content: str,
        source: Optional[str] = None,
        published_at: Optional[str] = None,
    ) -> list[Document]:
        """
        Prepare an article for chunking with proper metadata.

        Args:
            article_id: Database article ID
            ticker: Stock ticker symbol
            title: Article title
            content: Article content
            source: News source
            published_at: Publication date

        Returns:
            List of Document chunks
        """
        # Combine title and content for better context
        full_text = f"{title}\n\n{content}" if content else title

        metadata = {
            "article_id": article_id,
            "ticker": ticker,
            "title": title,
            "source": source or "unknown",
            "published_at": str(published_at) if published_at else None,
            "type": "news",
        }

        return self.chunk_text(full_text, metadata)

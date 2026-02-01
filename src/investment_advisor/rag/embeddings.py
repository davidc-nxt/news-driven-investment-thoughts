"""Embedding service using LangChain with sentence-transformers."""

import os
import warnings
from functools import lru_cache
from typing import Optional

# Suppress tokenizer parallelism warning and model loading warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from rich.console import Console

from investment_advisor.config import get_settings

console = Console()


class EmbeddingService:
    """Embedding service supporting local and OpenAI embeddings."""

    def __init__(self):
        self.settings = get_settings()
        self._embeddings = None

    @property
    def embeddings(self):
        """Lazy load embeddings model."""
        if self._embeddings is None:
            if self.settings.embedding_provider == "local":
                console.print(f"[dim]Using local embeddings: {self.settings.embedding_model}[/dim]")
                # Suppress verbose model loading output
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.settings.embedding_model,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                    show_progress=False,
                )
            else:
                console.print("[dim]Using OpenAI embeddings[/dim]")
                self._embeddings = OpenAIEmbeddings(
                    openai_api_key=self.settings.openai_api_key,
                )
        return self._embeddings

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        return self.embeddings.embed_query(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)

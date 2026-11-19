from typing import Protocol


class EmbeddingProvider(Protocol):
    """Interface for document and query embedding providers."""

    def embed_documents(
        self, texts: list[str]
    ) -> list[list[float]]:
        """Convert document texts into vectors."""
        ...

    def embed_query(self, query: str) -> list[float]:
        """Convert a search query into a vector."""
        ...
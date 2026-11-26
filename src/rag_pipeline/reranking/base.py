from typing import Protocol

from rag_pipeline.vectorstore.qdrant import SearchHit


class Reranker(Protocol):
    """Interface for re-ordering retrieved passages."""

    def rerank(
        self, query: str, hits: list[SearchHit]
    ) -> list[SearchHit]:
        """Return the hits sorted from most to least relevant."""
        ...

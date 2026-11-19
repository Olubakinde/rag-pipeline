from pathlib import Path

from rag_pipeline.embeddings.base import EmbeddingProvider
from rag_pipeline.ingestion import IngestionPipeline
from rag_pipeline.vectorstore.qdrant import (
    QdrantVectorStore,
    SearchHit,
)


class RAGPipeline:
    """Index documents and retrieve relevant passages."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: QdrantVectorStore,
        ingestion: IngestionPipeline | None = None,
    ):
        self.embedder = embedder
        self.store = store
        self.ingestion = (
            ingestion
            if ingestion is not None
            else IngestionPipeline()
        )

    def index(self, path: str | Path) -> int:
        """Parse, chunk, embed, and store a document."""

        result = self.ingestion.ingest(path)

        texts = [
            chunk.text
            for chunk in result.chunks
        ]

        vectors = self.embedder.embed_documents(texts)

        return self.store.add(result, vectors)

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[SearchHit]:
        """Retrieve relevant document passages."""

        query_vector = self.embedder.embed_query(query)

        return self.store.search(
            query_vector,
            limit=limit,
        )
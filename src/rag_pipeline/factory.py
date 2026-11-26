from collections.abc import Iterator
from contextlib import contextmanager

from rag_pipeline.chunking.boundaries import WordBoundaryChunker
from rag_pipeline.config import Settings
from rag_pipeline.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)
from rag_pipeline.ingestion import IngestionPipeline
from rag_pipeline.reranking.cross_encoder import CrossEncoderReranker
from rag_pipeline.retrieval import RAGPipeline
from rag_pipeline.vectorstore.qdrant import QdrantVectorStore


@contextmanager
def open_pipeline(settings: Settings) -> Iterator[RAGPipeline]:
    """Build the configured pipeline and close its database afterwards."""

    # Load models before opening the store so a failed download
    # cannot leave the local database locked.
    embedder = SentenceTransformerEmbeddingProvider(
        model_name=settings.embedding_model,
    )

    reranker = (
        CrossEncoderReranker(model_name=settings.reranker_model)
        if settings.reranker_model
        else None
    )

    store = QdrantVectorStore(
        collection_name=settings.collection_name,
        vector_size=embedder.dimension,
        path=settings.qdrant_path,
    )

    try:
        yield RAGPipeline(
            embedder=embedder,
            store=store,
            ingestion=IngestionPipeline(
                chunker=WordBoundaryChunker(
                    chunk_size=settings.chunk_size,
                    overlap=settings.chunk_overlap,
                )
            ),
            reranker=reranker,
            rerank_candidates=settings.rerank_candidates,
        )
    finally:
        store.client.close()

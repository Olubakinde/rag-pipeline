import pytest
from qdrant_client import QdrantClient

from rag_pipeline.chunking.fixed import FixedSizeChunker
from rag_pipeline.ingestion import IngestionPipeline
from rag_pipeline.retrieval import RAGPipeline
from rag_pipeline.vectorstore.qdrant import QdrantVectorStore


class FakeEmbedder:
    """Deterministic embeddings for integration tests."""

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, query):
        if "robot" in query.lower():
            return [1.0, 0.0]
        return [0.0, 1.0]


def make_pipeline(store):
    return RAGPipeline(
        embedder=FakeEmbedder(),
        store=store,
        ingestion=IngestionPipeline(
            chunker=FixedSizeChunker(
                chunk_size=100,
                overlap=0,
            )
        ),
    )


def test_index_and_search(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text(
        "Robots use sensors to navigate.",
        encoding="utf-8",
    )

    store = QdrantVectorStore(vector_size=2)
    pipeline = make_pipeline(store)

    assert pipeline.index(file_path).chunk_count == 1

    hits = pipeline.search("How do robots navigate?")

    assert len(hits) == 1
    assert hits[0].source == "notes.txt"
    assert hits[0].chunk.text == (
        "Robots use sensors to navigate."
    )


def test_reindex_same_file_without_duplicates(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text(
        "Robots use sensors to navigate.",
        encoding="utf-8",
    )

    store = QdrantVectorStore(vector_size=2)
    pipeline = make_pipeline(store)

    pipeline.index(file_path)
    pipeline.index(file_path)

    assert len(pipeline.search("robots", limit=10)) == 1


def test_persistent_storage(tmp_path):
    document_path = tmp_path / "research.txt"
    document_path.write_text(
        "Robots use cameras to navigate.",
        encoding="utf-8",
    )

    database_path = tmp_path / "vector_database"

    store = QdrantVectorStore(
        vector_size=2,
        path=database_path,
    )

    pipeline = make_pipeline(store)

    assert pipeline.index(document_path).chunk_count == 1
    store.client.close()

    # Reopen the database in a new client.
    reopened_store = QdrantVectorStore(
        vector_size=2,
        path=database_path,
    )

    try:
        reopened_pipeline = make_pipeline(reopened_store)
        hits = reopened_pipeline.search("robot navigation")

        assert len(hits) == 1
        assert hits[0].source == "research.txt"
    finally:
        reopened_store.client.close()


def test_reject_mismatched_collection_dimension(tmp_path):
    database_path = tmp_path / "vector_database"

    QdrantVectorStore(
        vector_size=2,
        path=database_path,
    ).client.close()

    with pytest.raises(ValueError, match="2-dimensional"):
        QdrantVectorStore(
            vector_size=3,
            path=database_path,
        )

    # The failed attempt must release the local database lock.
    QdrantVectorStore(
        vector_size=2,
        path=database_path,
    ).client.close()


def test_reject_client_and_path(tmp_path):
    client = QdrantClient(":memory:")

    try:
        with pytest.raises(ValueError):
            QdrantVectorStore(
                vector_size=2,
                client=client,
                path=tmp_path / "database",
            )
    finally:
        client.close()
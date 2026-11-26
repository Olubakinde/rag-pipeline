import pytest

from rag_pipeline.chunking.chunk import Chunk
from rag_pipeline.document import Document
from rag_pipeline.ingestion import IngestionResult
from rag_pipeline.vectorstore.qdrant import QdrantVectorStore


def make_result():
    """Create a document with two chunks for testing."""

    text = "Robots navigate. Coffee tastes good."

    document = Document(
        id="doc-001",
        text=text,
        source="notes.txt",
    )

    chunks = [
        Chunk(
            document_id=document.id,
            text=text[0:16],
            index=0,
            start=0,
            end=16,
        ),
        Chunk(
            document_id=document.id,
            text=text[17:],
            index=1,
            start=17,
            end=len(text),
        ),
    ]

    return IngestionResult(
        document=document,
        chunks=chunks,
    )


def test_store_and_search():
    store = QdrantVectorStore(vector_size=2)
    result = make_result()

    count = store.add(
        result,
        [[1.0, 0.0], [0.0, 1.0]],
    )

    hits = store.search([1.0, 0.0], limit=2)

    assert count == 2
    assert len(hits) == 2
    assert hits[0].chunk.text == "Robots navigate."
    assert hits[0].score == pytest.approx(1.0)


def test_source_metadata():
    store = QdrantVectorStore(vector_size=2)
    result = make_result()

    store.add(result, [[1.0, 0.0], [0.0, 1.0]])

    hit = store.search([1.0, 0.0], limit=1)[0]

    assert hit.source == "notes.txt"
    assert hit.chunk.document_id == "doc-001"
    assert hit.chunk.start == 0
    assert hit.chunk.end == 16


def test_reindex_without_duplicates():
    store = QdrantVectorStore(vector_size=2)
    result = make_result()
    vectors = [[1.0, 0.0], [0.0, 1.0]]

    store.add(result, vectors)
    store.add(result, vectors)

    hits = store.search([1.0, 0.0], limit=10)

    assert len(hits) == 2


def test_reindex_replaces_stale_chunks():
    store = QdrantVectorStore(vector_size=2)
    result = make_result()

    store.add(result, [[1.0, 0.0], [0.0, 1.0]])

    # Re-index the same document as a single chunk.
    rechunked = IngestionResult(
        document=result.document,
        chunks=[
            Chunk(
                document_id=result.document.id,
                text=result.document.text,
                index=0,
                start=0,
                end=len(result.document.text),
            )
        ],
    )

    store.add(rechunked, [[1.0, 0.0]])

    hits = store.search([1.0, 0.0], limit=10)

    assert len(hits) == 1
    assert hits[0].chunk.text == result.document.text


def test_failed_add_keeps_existing_chunks():
    store = QdrantVectorStore(vector_size=2)
    result = make_result()

    store.add(result, [[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError):
        store.add(result, [[1.0, 0.0], [0.0, 1.0, 0.0]])

    assert len(store.search([1.0, 0.0], limit=10)) == 2


def test_mismatched_embedding_count():
    store = QdrantVectorStore(vector_size=2)

    with pytest.raises(ValueError):
        store.add(make_result(), [[1.0, 0.0]])


def test_wrong_vector_dimensions():
    store = QdrantVectorStore(vector_size=2)

    with pytest.raises(ValueError):
        store.add(
            make_result(),
            [[1.0, 0.0], [0.0, 1.0, 0.0]],
        )


def test_invalid_search_dimension():
    store = QdrantVectorStore(vector_size=2)

    with pytest.raises(ValueError):
        store.search([1.0, 0.0, 0.0])


def test_invalid_search_limit():
    store = QdrantVectorStore(vector_size=2)

    with pytest.raises(ValueError):
        store.search([1.0, 0.0], limit=0)
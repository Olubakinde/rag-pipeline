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


def make_single_chunk_result(document_id, source, text):
    return IngestionResult(
        document=Document(id=document_id, text=text, source=source),
        chunks=[
            Chunk(
                document_id=document_id,
                text=text,
                index=0,
                start=0,
                end=len(text),
            )
        ],
    )


def test_list_documents():
    store = QdrantVectorStore(vector_size=2)

    store.add(make_result(), [[1.0, 0.0], [0.0, 1.0]])
    store.add(
        make_single_chunk_result("doc-002", "atlas.txt", "Maps."),
        [[1.0, 0.0]],
    )

    documents = store.list_documents()

    assert [
        (document.source, document.document_id, document.chunk_count)
        for document in documents
    ] == [
        ("atlas.txt", "doc-002", 1),
        ("notes.txt", "doc-001", 2),
    ]


def test_list_documents_spans_multiple_pages():
    store = QdrantVectorStore(vector_size=2)

    # More documents than a single scroll page returns.
    for number in range(300):
        store.add(
            make_single_chunk_result(
                f"doc-{number:03}",
                f"file-{number:03}.txt",
                f"Text {number}",
            ),
            [[1.0, 0.0]],
        )

    assert len(store.list_documents()) == 300


def test_list_documents_empty():
    assert QdrantVectorStore(vector_size=2).list_documents() == []


def test_delete_document():
    store = QdrantVectorStore(vector_size=2)

    store.add(make_result(), [[1.0, 0.0], [0.0, 1.0]])
    store.add(
        make_single_chunk_result("doc-002", "atlas.txt", "Maps."),
        [[1.0, 0.0]],
    )

    assert store.delete_document("doc-001") == 2

    hits = store.search([1.0, 0.0], limit=10)

    assert [hit.chunk.document_id for hit in hits] == ["doc-002"]


def test_delete_unknown_document():
    store = QdrantVectorStore(vector_size=2)

    store.add(make_result(), [[1.0, 0.0], [0.0, 1.0]])

    assert store.delete_document("missing") == 0
    assert len(store.search([1.0, 0.0], limit=10)) == 2


def test_search_min_score():
    store = QdrantVectorStore(vector_size=2)

    store.add(make_result(), [[1.0, 0.0], [0.0, 1.0]])

    hits = store.search([1.0, 0.0], limit=10, min_score=0.5)

    assert [hit.chunk.text for hit in hits] == ["Robots navigate."]


def test_search_filtered_by_source():
    store = QdrantVectorStore(vector_size=2)

    store.add(make_result(), [[1.0, 0.0], [0.0, 1.0]])
    store.add(
        make_single_chunk_result("doc-002", "atlas.txt", "Maps."),
        [[1.0, 0.0]],
    )

    hits = store.search([1.0, 0.0], limit=10, sources=["atlas.txt"])

    assert [hit.source for hit in hits] == ["atlas.txt"]


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
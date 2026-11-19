import pytest

from rag_pipeline.chunking.boundaries import WordBoundaryChunker
from rag_pipeline.document import Document
from rag_pipeline.ingestion import IngestionPipeline


def make_document(text):
    return Document(
        id="doc-001",
        text=text,
        source="example.txt",
    )


def test_word_boundaries():
    document = make_document(
        "alpha beta gamma delta epsilon zeta"
    )

    chunker = WordBoundaryChunker(
        chunk_size=16,
        overlap=6,
    )

    chunks = chunker.split(document)

    assert [chunk.text for chunk in chunks] == [
        "alpha beta gamma",
        "gamma delta",
        "delta epsilon",
        "zeta",
    ]


def test_paragraph_boundaries():
    document = make_document(
        "First paragraph is about robots.\n\n"
        "Second paragraph is about coffee."
    )

    chunks = WordBoundaryChunker(
        chunk_size=45,
        overlap=0,
    ).split(document)

    assert [chunk.text for chunk in chunks] == [
        "First paragraph is about robots.",
        "Second paragraph is about coffee.",
    ]


def test_short_document():
    document = make_document("Robots use sensors.")

    chunks = WordBoundaryChunker(
        chunk_size=100,
        overlap=20,
    ).split(document)

    assert len(chunks) == 1
    assert chunks[0].text == document.text


def test_original_offsets():
    document = make_document(
        "  Robots use sensors.\n\nCoffee tastes good.  "
    )

    chunks = WordBoundaryChunker(
        chunk_size=25,
        overlap=0,
    ).split(document)

    for index, chunk in enumerate(chunks):
        assert document.text[chunk.start:chunk.end] == chunk.text
        assert chunk.index == index
        assert chunk.document_id == document.id


def test_long_word():
    document = make_document(
        "Supercalifragilisticexpialidocious"
    )

    chunks = WordBoundaryChunker(
        chunk_size=8,
        overlap=2,
    ).split(document)

    assert len(chunks) == 1
    assert chunks[0].text == document.text


@pytest.mark.parametrize(
    "chunk_size, overlap",
    [
        (0, 0),
        (-1, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)
def test_invalid_configuration(chunk_size, overlap):
    with pytest.raises(ValueError):
        WordBoundaryChunker(
            chunk_size=chunk_size,
            overlap=overlap,
        )


def test_ingestion_with_word_chunker(tmp_path):
    file_path = tmp_path / "example.txt"

    file_path.write_text(
        "Robots use sensors.\n\nCoffee tastes good.",
        encoding="utf-8",
    )

    pipeline = IngestionPipeline(
        chunker=WordBoundaryChunker(
            chunk_size=27,
            overlap=0,
        )
    )

    result = pipeline.ingest(file_path)

    assert [chunk.text for chunk in result.chunks] == [
        "Robots use sensors.",
        "Coffee tastes good.",
    ]

    for chunk in result.chunks:
        assert (
            result.document.text[chunk.start:chunk.end]
            == chunk.text
        )
import pytest

from rag_pipeline.chunking.fixed import FixedSizeChunker
from rag_pipeline.document import Document


# Verify that fixed-size chunking produces the expected segments.
def test_fixed_size_chunking():
    document = Document(
        id="doc-001",
        text="ABCDEFGHIJ",
        source="sample.txt",
    )

    chunker = FixedSizeChunker(chunk_size=4, overlap=1)
    chunks = chunker.split(document)

    assert len(chunks) == 3

    assert [chunk.text for chunk in chunks] == [
        "ABCD",
        "DEFG",
        "GHIJ",
    ]

    assert [chunk.index for chunk in chunks] == [0, 1, 2]


# Ensure each chunk preserves its original document reference.
def test_document_reference():
    document = Document(
        id="doc-002",
        text="ABCDEFGHIJ",
        source="sample.txt",
    )

    chunks = FixedSizeChunker(chunk_size=4, overlap=1).split(document)

    for chunk in chunks:
        assert chunk.document_id == document.id


# Verify that chunk offsets correspond to the original document.
def test_chunk_offsets():
    document = Document(
        id="doc-003",
        text="ABCDEFGHIJ",
        source="sample.txt",
    )

    chunks = FixedSizeChunker(chunk_size=4, overlap=1).split(document)

    assert [(chunk.start, chunk.end) for chunk in chunks] == [
        (0, 4),
        (3, 7),
        (6, 10),
    ]

    for chunk in chunks:
        assert document.text[chunk.start:chunk.end] == chunk.text


# Ensure short documents produce only one chunk.
def test_short_document():
    document = Document(
        id="doc-004",
        text="Hello",
        source="short.txt",
    )

    chunks = FixedSizeChunker(chunk_size=100, overlap=20).split(document)

    assert len(chunks) == 1
    assert chunks[0].text == "Hello"
    assert chunks[0].start == 0
    assert chunks[0].end == 5


# Verify that zero overlap produces consecutive chunks.
def test_zero_overlap():
    document = Document(
        id="doc-005",
        text="ABCDEFGH",
        source="sample.txt",
    )

    chunks = FixedSizeChunker(chunk_size=4, overlap=0).split(document)

    assert [chunk.text for chunk in chunks] == [
        "ABCD",
        "EFGH",
    ]


# Ensure invalid chunk configurations are rejected.
@pytest.mark.parametrize(
    "chunk_size, overlap",
    [
        (0, 0),
        (-1, 0),
        (4, -1),
        (4, 4),
        (4, 5),
    ],
)
def test_invalid_chunk_configuration(chunk_size, overlap):
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)


# Ensure the algorithm does not create redundant trailing chunks.
def test_no_duplicate_final_chunk():
    document = Document(
        id="doc-006",
        text="ABCDE",
        source="sample.txt",
    )

    chunks = FixedSizeChunker(chunk_size=4, overlap=3).split(document)

    assert [chunk.text for chunk in chunks] == [
        "ABCD",
        "BCDE",
    ]
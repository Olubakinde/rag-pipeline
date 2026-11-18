import pytest

from rag_pipeline.document import Document


# Verify that a document can be initialized with valid attributes.
def test_create_valid_document():
    doc = Document(
        id="doc-001",
        text="Introduction to machine learning.",
        source="lecture.pdf"
    )

    assert doc.id == "doc-001"
    assert doc.text == "Introduction to machine learning."
    assert doc.source == "lecture.pdf"
    assert doc.metadata == {}


# Ensure documents with empty text are rejected.
def test_empty_text():
    with pytest.raises(ValueError):
        Document(id="doc-002", text="", source="empty.pdf")


# Ensure whitespace-only content is rejected.
def test_whitespace_text():
    with pytest.raises(ValueError):
        Document(id="doc-003", text="   ", source="empty.pdf")


# Ensure documents cannot have an empty identifier.
def test_empty_id():
    with pytest.raises(ValueError):
        Document(id="", text="Valid content", source="test.pdf")


# Ensure whitespace-only identifiers are rejected.
def test_whitespace_id():
    with pytest.raises(ValueError):
        Document(id="   ", text="Valid content", source="test.pdf")


# Ensure different documents do not share the same metadata dictionary.
def test_metadata_isolation():
    first = Document(id="001", text="First document", source="a.pdf")
    second = Document(id="002", text="Second document", source="b.pdf")

    first.metadata["author"] = "Alex"

    assert second.metadata == {}
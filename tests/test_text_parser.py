import pytest

from rag_pipeline.parsers.text import TextParser


# Verify that a valid text file is parsed into a Document.
def test_parse_valid_text_file(tmp_path):
    file = tmp_path / "sample.txt"
    file.write_text("Hello from the RAG pipeline.", encoding="utf-8")

    document = TextParser().parse(file)

    assert document.text == "Hello from the RAG pipeline."
    assert document.source == "sample.txt"
    assert document.metadata["file_type"] == "txt"
    assert document.id


# Identical content must map to the same ID so re-indexing is idempotent.
def test_document_id_is_content_based(tmp_path):
    first = tmp_path / "first.txt"
    copy = tmp_path / "copy.txt"
    other = tmp_path / "other.txt"

    first.write_text("Robots navigate.", encoding="utf-8")
    copy.write_text("Robots navigate.", encoding="utf-8")
    other.write_text("Coffee tastes good.", encoding="utf-8")

    parser = TextParser()

    assert parser.parse(first).id == parser.parse(first).id
    assert parser.parse(first).id == parser.parse(copy).id
    assert parser.parse(first).id != parser.parse(other).id


# Ensure the parser rejects unsupported file formats.
def test_unsupported_file_type(tmp_path):
    file = tmp_path / "sample.pdf"
    file.write_text("Example content", encoding="utf-8")

    with pytest.raises(ValueError):
        TextParser().parse(file)


# Ensure missing files raise FileNotFoundError.
def test_missing_file(tmp_path):
    file = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        TextParser().parse(file)


# Ensure empty files are rejected by Document validation.
def test_empty_file(tmp_path):
    file = tmp_path / "empty.txt"
    file.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        TextParser().parse(file)


# Ensure files containing only whitespace are rejected.
def test_whitespace_file(tmp_path):
    file = tmp_path / "whitespace.txt"
    file.write_text("   \n   ", encoding="utf-8")

    with pytest.raises(ValueError):
        TextParser().parse(file)
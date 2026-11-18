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
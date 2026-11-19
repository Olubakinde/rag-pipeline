import pytest
from docx import Document as WordDocument

from rag_pipeline.parsers.docx import DocxParser


def test_parse_docx(tmp_path):
    file_path = tmp_path / "sample.docx"

    # Create a Word document.
    word_document = WordDocument()

    word_document.add_paragraph(
        "Retrieval-Augmented Generation"
    )

    word_document.add_paragraph(
        "Combines retrieval with language models."
    )

    word_document.save(file_path)

    # Parse the document.
    document = DocxParser().parse(file_path)

    assert "Retrieval-Augmented Generation" in document.text
    assert "Combines retrieval" in document.text

    assert document.source == "sample.docx"
    assert document.metadata["file_type"] == "docx"
    assert document.metadata["paragraph_count"] == 2
    assert document.metadata["table_count"] == 0

    assert document.id


def test_docx_table_extraction(tmp_path):
    file_path = tmp_path / "table.docx"

    word_document = WordDocument()
    word_document.add_paragraph("Before table")

    # Create a table containing two rows.
    table = word_document.add_table(rows=2, cols=2)

    table.cell(0, 0).text = "Name"
    table.cell(0, 1).text = "Value"

    table.cell(1, 0).text = "Chunk size"
    table.cell(1, 1).text = "1000"

    word_document.add_paragraph("After table")
    word_document.save(file_path)

    document = DocxParser().parse(file_path)

    assert document.text.split("\n\n") == [
        "Before table",
        "Name | Value",
        "Chunk size | 1000",
        "After table",
    ]

    assert document.metadata["table_count"] == 1


def test_unsupported_docx_extension(tmp_path):
    file_path = tmp_path / "sample.txt"

    file_path.write_text(
        "Example content",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        DocxParser().parse(file_path)


def test_missing_docx(tmp_path):
    file_path = tmp_path / "missing.docx"

    with pytest.raises(FileNotFoundError):
        DocxParser().parse(file_path)


def test_empty_docx(tmp_path):
    file_path = tmp_path / "empty.docx"

    word_document = WordDocument()
    word_document.save(file_path)

    with pytest.raises(ValueError):
        DocxParser().parse(file_path)
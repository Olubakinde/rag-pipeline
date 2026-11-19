from pathlib import Path

import pytest
from docx import Document as WordDocument
from reportlab.pdfgen import canvas

from rag_pipeline.document import Document
from rag_pipeline.parsers.registry import ParserRegistry


def test_text_parser_dispatch(tmp_path):
    file_path = tmp_path / "notes.TXT"
    file_path.write_text("Hello from the registry.", encoding="utf-8")

    document = ParserRegistry().parse(file_path)

    assert document.text == "Hello from the registry."
    assert document.metadata["file_type"] == "txt"


def test_pdf_parser_dispatch(tmp_path):
    file_path = tmp_path / "research.pdf"

    pdf = canvas.Canvas(str(file_path))
    pdf.drawString(72, 720, "Research document")
    pdf.save()

    document = ParserRegistry().parse(file_path)

    assert "Research document" in document.text
    assert document.metadata["file_type"] == "pdf"


def test_docx_parser_dispatch(tmp_path):
    file_path = tmp_path / "report.docx"

    word_document = WordDocument()
    word_document.add_paragraph("Word document content")
    word_document.save(file_path)

    document = ParserRegistry().parse(file_path)

    assert "Word document content" in document.text
    assert document.metadata["file_type"] == "docx"


def test_unsupported_extension(tmp_path):
    file_path = tmp_path / "image.png"
    file_path.write_bytes(b"example")

    with pytest.raises(ValueError, match="Unsupported document format"):
        ParserRegistry().parse(file_path)


def test_missing_document(tmp_path):
    file_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        ParserRegistry().parse(file_path)


def test_custom_parser_registration(tmp_path):
    class MarkdownParser:
        def parse(self, path: str | Path) -> Document:
            file_path = Path(path)

            return Document(
                id="markdown-001",
                text=file_path.read_text(encoding="utf-8"),
                source=file_path.name,
                metadata={"file_type": "md"},
            )

    file_path = tmp_path / "README.md"
    file_path.write_text("# My Project", encoding="utf-8")

    registry = ParserRegistry()
    registry.register(".md", MarkdownParser())

    document = registry.parse(file_path)

    assert document.text == "# My Project"
    assert document.metadata["file_type"] == "md"
import pytest
from docx import Document as WordDocument
from reportlab.pdfgen import canvas

from rag_pipeline.chunking.fixed import FixedSizeChunker
from rag_pipeline.ingestion import IngestionPipeline


def test_ingest_text_file(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("ABCDEFGHIJ", encoding="utf-8")

    pipeline = IngestionPipeline(
        chunker=FixedSizeChunker(chunk_size=4, overlap=1)
    )

    result = pipeline.ingest(file_path)

    assert result.document.text == "ABCDEFGHIJ"
    assert result.document.metadata["file_type"] == "txt"

    assert [chunk.text for chunk in result.chunks] == [
        "ABCD",
        "DEFG",
        "GHIJ",
    ]

    for chunk in result.chunks:
        assert chunk.document_id == result.document.id
        assert (
            result.document.text[chunk.start:chunk.end]
            == chunk.text
        )


def test_ingest_pdf_file(tmp_path):
    file_path = tmp_path / "sample.pdf"

    pdf = canvas.Canvas(str(file_path))
    pdf.drawString(72, 720, "First page content")
    pdf.showPage()
    pdf.drawString(72, 720, "Second page content")
    pdf.save()

    pipeline = IngestionPipeline(
        chunker=FixedSizeChunker(chunk_size=20, overlap=5)
    )

    result = pipeline.ingest(file_path)

    assert result.document.metadata["file_type"] == "pdf"
    assert result.document.metadata["page_count"] == 2
    assert len(result.chunks) >= 2

    # Verify PDF page offsets still refer to the extracted text.
    for span in result.document.metadata["page_spans"]:
        page_text = result.document.text[
            span["start"]:span["end"]
        ]
        assert page_text

    # Verify generated chunks preserve their document references.
    for chunk in result.chunks:
        assert chunk.document_id == result.document.id
        assert (
            result.document.text[chunk.start:chunk.end]
            == chunk.text
        )


def test_ingest_docx_file(tmp_path):
    file_path = tmp_path / "sample.docx"

    word_document = WordDocument()
    word_document.add_paragraph("First paragraph")
    word_document.add_paragraph("Second paragraph")
    word_document.save(file_path)

    pipeline = IngestionPipeline(
        chunker=FixedSizeChunker(chunk_size=15, overlap=5)
    )

    result = pipeline.ingest(file_path)

    assert result.document.metadata["file_type"] == "docx"
    assert result.document.metadata["paragraph_count"] == 2
    assert len(result.chunks) >= 2

    for chunk in result.chunks:
        assert (
            result.document.text[chunk.start:chunk.end]
            == chunk.text
        )


def test_custom_chunking_configuration(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("ABCDEFGHIJ", encoding="utf-8")

    pipeline = IngestionPipeline(
        chunker=FixedSizeChunker(chunk_size=4, overlap=0)
    )

    result = pipeline.ingest(file_path)

    assert [chunk.text for chunk in result.chunks] == [
        "ABCD",
        "EFGH",
        "IJ",
    ]


def test_unsupported_file_type(tmp_path):
    file_path = tmp_path / "image.png"
    file_path.write_bytes(b"example")

    pipeline = IngestionPipeline()

    with pytest.raises(ValueError, match="Unsupported document format"):
        pipeline.ingest(file_path)


def test_missing_file(tmp_path):
    file_path = tmp_path / "missing.txt"

    pipeline = IngestionPipeline()

    with pytest.raises(FileNotFoundError):
        pipeline.ingest(file_path)
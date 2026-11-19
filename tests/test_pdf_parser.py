import pytest
from reportlab.pdfgen import canvas

from rag_pipeline.parsers.pdf import PdfParser


# Verify that the parser extracts text from multiple PDF pages.
def test_parse_pdf(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    # Create a two-page PDF for testing.
    pdf = canvas.Canvas(str(pdf_path))

    pdf.drawString(72, 720, "First page information")
    pdf.showPage()

    pdf.drawString(72, 720, "Second page details")
    pdf.save()

    # Parse the generated PDF.
    document = PdfParser().parse(pdf_path)

    assert "First page information" in document.text
    assert "Second page details" in document.text

    assert document.source == "sample.pdf"
    assert document.metadata["file_type"] == "pdf"
    assert document.metadata["page_count"] == 2

    # Verify that both page numbers are preserved.
    spans = document.metadata["page_spans"]
    assert [span["page"] for span in spans] == [1, 2]

    # Verify that stored offsets point to the correct text.
    assert document.text[
        spans[0]["start"]:spans[0]["end"]
    ] == "First page information"


# Ensure unsupported file extensions are rejected.
def test_unsupported_pdf_extension(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Example", encoding="utf-8")

    with pytest.raises(ValueError):
        PdfParser().parse(file_path)


# Ensure missing PDF files raise FileNotFoundError.
def test_missing_pdf(tmp_path):
    pdf_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        PdfParser().parse(pdf_path)


# Ensure PDFs without extractable text are rejected.
def test_empty_pdf(tmp_path):
    pdf_path = tmp_path / "empty.pdf"

    pdf = canvas.Canvas(str(pdf_path))
    pdf.showPage()
    pdf.save()

    with pytest.raises(ValueError):
        PdfParser().parse(pdf_path)
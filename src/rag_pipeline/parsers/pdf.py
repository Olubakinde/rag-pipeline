from pathlib import Path

from pypdf import PdfReader

from rag_pipeline.document import Document, content_id


class PdfParser:
    """Extract text and page metadata from PDF documents."""

    def parse(self, path: str | Path) -> Document:
        file_path = Path(path)

        # Ensure the supplied file has a PDF extension.
        if file_path.suffix.lower() != ".pdf":
            raise ValueError("Unsupported file type. Expected a PDF.")

        # Load the PDF document.
        reader = PdfReader(file_path)

        pages = []
        page_spans = []
        offset = 0

        # Extract text from each page separately.
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            # Skip pages without extractable text.
            if not text:
                continue

            # Account for the separator between pages.
            if pages:
                offset += 2

            start = offset
            pages.append(text)
            offset += len(text)

            # Record where this page appears in the extracted text.
            page_spans.append(
                {
                    "page": page_number,
                    "start": start,
                    "end": offset,
                }
            )

        # Combine extracted page text into one document.
        content = "\n\n".join(pages)

        return Document(
            id=content_id(content),
            text=content,
            source=file_path.name,
            metadata={
                "file_type": "pdf",
                "page_count": len(reader.pages),
                "page_spans": page_spans,
            },
        )
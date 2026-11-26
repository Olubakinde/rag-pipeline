from pathlib import Path

from docx import Document as WordDocument
from docx.table import Table
from docx.text.paragraph import Paragraph

from rag_pipeline.document import Document, content_id


class DocxParser:
    """Extract text from Microsoft Word documents."""

    def parse(self, path: str | Path) -> Document:
        file_path = Path(path)

        # Validate the file extension.
        if file_path.suffix.lower() != ".docx":
            raise ValueError(
                "Unsupported file type. Expected a .docx file."
            )

        # Ensure the file exists before attempting to parse it.
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load the Word document.
        word_document = WordDocument(str(file_path))

        content_parts = []
        paragraph_count = 0
        table_count = 0

        # Process paragraphs and tables in document order.
        for block in word_document.iter_inner_content():

            if isinstance(block, Paragraph):
                text = block.text.strip()

                if text:
                    content_parts.append(text)
                    paragraph_count += 1

            elif isinstance(block, Table):
                table_count += 1

                # Extract text from each table row.
                for row in block.rows:
                    cells = [
                        cell.text.strip()
                        for cell in row.cells
                    ]

                    if any(cells):
                        content_parts.append(
                            " | ".join(cells)
                        )

        # Combine extracted content.
        content = "\n\n".join(content_parts)

        # Return a normalized document.
        return Document(
            id=content_id(content),
            text=content,
            source=file_path.name,
            metadata={
                "file_type": "docx",
                "paragraph_count": paragraph_count,
                "table_count": table_count,
            },
        )
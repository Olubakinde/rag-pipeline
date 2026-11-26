from pathlib import Path

from rag_pipeline.document import Document, content_id


class TextParser:
    """Parse plain-text files into normalized Document objects."""

    def parse(self, path: str | Path) -> Document:
        # Convert the supplied path into a Path object.
        file_path = Path(path)

        # Validate the file extension.
        if file_path.suffix.lower() != ".txt":
            raise ValueError("Unsupported file type. Expected a .txt file.")

        # Read the file contents using UTF-8 encoding.
        content = file_path.read_text(encoding="utf-8")

        # Return a normalized Document object.
        return Document(
            id=content_id(content),
            text=content,
            source=file_path.name,
            metadata={"file_type": "txt"},
        )
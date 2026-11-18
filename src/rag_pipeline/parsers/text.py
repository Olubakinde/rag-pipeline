from pathlib import Path
from uuid import uuid4

from rag_pipeline.document import Document


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

        # Generate a unique identifier.
        document_id = str(uuid4())

        # Return a normalized Document object.
        return Document(
            id=document_id,
            text=content,
            source=file_path.name,
            metadata={"file_type": "txt"},
        )
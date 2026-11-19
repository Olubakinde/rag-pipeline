from pathlib import Path
from typing import Protocol

from rag_pipeline.document import Document


class DocumentParser(Protocol):
    """Define the interface for document parsers."""

    def parse(self, path: str | Path) -> Document:
        """Parse a file into a normalized Document object."""
        ...
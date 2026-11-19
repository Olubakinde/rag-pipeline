from typing import Protocol

from rag_pipeline.chunking.chunk import Chunk
from rag_pipeline.document import Document


class Chunker(Protocol):
    """Interface for document chunking strategies."""

    def split(self, document: Document) -> list[Chunk]:
        """Split a document into source-traceable chunks."""
        ...
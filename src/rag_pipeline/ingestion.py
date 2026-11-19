from dataclasses import dataclass
from pathlib import Path

from rag_pipeline.chunking.base import Chunker
from rag_pipeline.chunking.chunk import Chunk
from rag_pipeline.chunking.fixed import FixedSizeChunker
from rag_pipeline.document import Document
from rag_pipeline.parsers.registry import ParserRegistry


@dataclass
class IngestionResult:
    """Store the original document and its generated chunks."""

    document: Document
    chunks: list[Chunk]


class IngestionPipeline:
    """Orchestrate document parsing and text chunking."""

    def __init__(
        self,
        registry: ParserRegistry | None = None,
        chunker: Chunker | None = None,
    ):
        # Allow custom components while providing defaults.
        self.registry = (
            registry if registry is not None else ParserRegistry()
        )

        self.chunker = (
            chunker if chunker is not None else FixedSizeChunker()
        )

    def ingest(self, path: str | Path) -> IngestionResult:
        """Parse a supported document and generate text chunks."""

        # Automatically select the appropriate document parser.
        document = self.registry.parse(path)

        # Split the extracted text into overlapping chunks.
        chunks = self.chunker.split(document)

        # Return both the source document and its chunks.
        return IngestionResult(
            document=document,
            chunks=chunks,
        )
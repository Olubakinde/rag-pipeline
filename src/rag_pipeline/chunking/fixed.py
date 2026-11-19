from rag_pipeline.chunking.chunk import Chunk
from rag_pipeline.document import Document


class FixedSizeChunker:
    """Split documents into fixed-size, overlapping text chunks."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):

        # Validate the chunk size.
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than zero")

        # Ensure overlap is valid and prevents infinite loops.
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                "Overlap must be non-negative and smaller than chunk size"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: Document) -> list[Chunk]:
        """Split a document while preserving its original text offsets."""

        chunks = []

        # Calculate the distance between consecutive chunk starts.
        step = self.chunk_size - self.overlap

        # Initialize the starting position and chunk index.
        start = 0
        index = 0

        while start < len(document.text):

            # Ensure the end position stays within the document.
            end = min(start + self.chunk_size, len(document.text))

            # Extract the current segment of text.
            chunk_text = document.text[start:end]

            # Create a chunk with its document reference and offsets.
            chunk = Chunk(
                document_id=document.id,
                text=chunk_text,
                index=index,
                start=start,
                end=end,
            )

            chunks.append(chunk)

            # Stop once the final character has been included.
            if end == len(document.text):
                break

            # Advance to the next position, preserving overlap.
            start += step
            index += 1

        return chunks
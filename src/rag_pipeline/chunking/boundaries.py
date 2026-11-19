import re

from rag_pipeline.chunking.chunk import Chunk
from rag_pipeline.document import Document


class WordBoundaryChunker:
    """Split text at word boundaries, preferring paragraph breaks."""

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
    ):
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                "Overlap must be non-negative and smaller than chunk size"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: Document) -> list[Chunk]:
        """Produce overlapping chunks with original character offsets."""

        text = document.text

        # Find each word's position in the original document.
        words = list(re.finditer(r"\S+", text))

        chunks = []
        first = 0

        while first < len(words):
            start = words[first].start()
            last = first

            # Include complete words while staying within the target size.
            while (
                last < len(words)
                and words[last].end() - start <= self.chunk_size
            ):
                last += 1

            # Keep a single long word intact if it exceeds chunk_size.
            if last == first:
                last += 1

            # Prefer a paragraph break when it produces a useful chunk.
            if last < len(words):
                for boundary in range(last - 1, first, -1):
                    gap = text[
                        words[boundary - 1].end():
                        words[boundary].start()
                    ]

                    length = words[boundary - 1].end() - start

                    if (
                        "\n\n" in gap
                        and length >= self.chunk_size // 2
                    ):
                        last = boundary
                        break

            end = words[last - 1].end()

            chunks.append(
                Chunk(
                    document_id=document.id,
                    text=text[start:end],
                    index=len(chunks),
                    start=start,
                    end=end,
                )
            )

            # Stop after including the final word.
            if last == len(words):
                break

            # Find a word boundary near the desired overlap.
            next_first = last

            if self.overlap > 0:
                target = end - self.overlap

                for position in range(first + 1, last):
                    if words[position].start() >= target:
                        next_first = position
                        break

            first = next_first

        return chunks
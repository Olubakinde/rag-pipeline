from dataclasses import dataclass


@dataclass
class Chunk:
    """Represent a text segment from a source document."""

    document_id: str
    text: str
    index: int
    start: int
    end: int
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any


def content_id(text: str) -> str:
    """Derive a stable document ID from its extracted text."""

    return sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Document:
    id: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Reject documents without a valid identifier.
        if not self.id.strip():
            raise ValueError("Document ID cannot be empty")

        # Reject documents containing empty or whitespace-only text.
        if not self.text.strip():
            raise ValueError("Document text cannot be empty")
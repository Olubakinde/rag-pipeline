from pathlib import Path

from rag_pipeline.document import Document
from rag_pipeline.parsers.base import DocumentParser
from rag_pipeline.parsers.docx import DocxParser
from rag_pipeline.parsers.pdf import PdfParser
from rag_pipeline.parsers.text import TextParser


class ParserRegistry:
    """Register and dispatch document parsers by file extension."""

    def __init__(self):
        self._parsers: dict[str, DocumentParser] = {}

        # Register supported document formats.
        self.register(".txt", TextParser())
        self.register(".pdf", PdfParser())
        self.register(".docx", DocxParser())

    def register(
        self,
        extension: str,
        parser: DocumentParser,
    ) -> None:

        extension = extension.lower()

        if not extension.startswith(".") or len(extension) < 2:
            raise ValueError(
                "Extension must begin with a dot, e.g. '.txt'"
            )

        self._parsers[extension] = parser

    @property
    def supported_extensions(self) -> set[str]:
        """Return the file extensions that have a registered parser."""

        return set(self._parsers)

    def get_parser(self, path: str | Path) -> DocumentParser:
        """Find the appropriate parser for a file."""

        file_path = Path(path)
        extension = file_path.suffix.lower()

        parser = self._parsers.get(extension)

        if parser is None:
            raise ValueError(
                f"Unsupported document format: {extension or '(none)'}"
            )

        return parser

    def parse(self, path: str | Path) -> Document:
        """Automatically parse a supported document."""

        parser = self.get_parser(path)

        return parser.parse(path)
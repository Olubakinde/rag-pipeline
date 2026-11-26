from contextlib import contextmanager

import pytest
from typer.testing import CliRunner

from rag_pipeline import cli
from rag_pipeline.chunking.chunk import Chunk
from rag_pipeline.chunking.fixed import FixedSizeChunker
from rag_pipeline.ingestion import IngestionPipeline
from rag_pipeline.retrieval import RAGPipeline
from rag_pipeline.vectorstore.qdrant import QdrantVectorStore, SearchHit

runner = CliRunner()


class FakeEmbedder:
    """Deterministic embeddings for CLI testing."""

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, query):
        if "robot" in query.lower():
            return [1.0, 0.0]
        return [0.0, 1.0]


@pytest.fixture
def pipeline(monkeypatch):
    store = QdrantVectorStore(vector_size=2)

    test_pipeline = RAGPipeline(
        embedder=FakeEmbedder(),
        store=store,
        ingestion=IngestionPipeline(
            chunker=FixedSizeChunker(chunk_size=100, overlap=0)
        ),
    )

    @contextmanager
    def fake_open_pipeline(settings):
        yield test_pipeline

    # Share one in-memory store across CLI invocations.
    monkeypatch.setattr(cli, "open_pipeline", fake_open_pipeline)

    yield test_pipeline

    store.client.close()


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_index_files(pipeline, tmp_path):
    file_path = write(tmp_path / "robots.txt", "Robots navigate.")

    result = runner.invoke(cli.app, ["index", str(file_path)])

    assert result.exit_code == 0
    assert "indexed" in result.output
    assert "1 chunks" in result.output
    assert len(pipeline.list_documents()) == 1


def test_index_directory_skips_unsupported_files(pipeline, tmp_path):
    (tmp_path / "nested").mkdir()

    write(tmp_path / "robots.txt", "Robots navigate.")
    write(tmp_path / "nested" / "coffee.txt", "Coffee.")
    (tmp_path / "image.png").write_bytes(b"not a document")

    result = runner.invoke(cli.app, ["index", str(tmp_path)])

    assert result.exit_code == 0
    assert sorted(
        document.source for document in pipeline.list_documents()
    ) == ["coffee.txt", "robots.txt"]


def test_index_continues_after_failure(pipeline, tmp_path):
    empty = write(tmp_path / "empty.txt", "   ")
    good = write(tmp_path / "robots.txt", "Robots navigate.")

    missing = tmp_path / "missing.txt"

    result = runner.invoke(
        cli.app,
        ["index", str(empty), str(missing), str(good)],
    )

    assert result.exit_code == 1
    assert result.output.count("FAILED") == 2
    assert len(pipeline.list_documents()) == 1


def test_index_empty_directory(pipeline, tmp_path):
    result = runner.invoke(cli.app, ["index", str(tmp_path)])

    assert result.exit_code == 1
    assert "No supported documents found" in result.output


def test_search(pipeline, tmp_path):
    pipeline.index(write(tmp_path / "robots.txt", "Robots navigate."))
    pipeline.index(write(tmp_path / "coffee.txt", "Coffee tastes good."))

    result = runner.invoke(
        cli.app,
        ["search", "robots", "--limit", "1"],
    )

    assert result.exit_code == 0
    assert "1. robots.txt (score 1.000)" in result.output
    assert "Robots navigate." in result.output
    assert "coffee.txt" not in result.output


def test_search_with_source_filter(pipeline, tmp_path):
    pipeline.index(write(tmp_path / "robots.txt", "Robots navigate."))
    pipeline.index(write(tmp_path / "coffee.txt", "Coffee tastes good."))

    result = runner.invoke(
        cli.app,
        ["search", "robots", "--source", "coffee.txt"],
    )

    assert result.exit_code == 0
    assert "coffee.txt" in result.output
    assert "robots.txt" not in result.output


def test_search_truncates_long_passages(pipeline, tmp_path):
    long_text = "Robots " + "word " * 100

    pipeline.ingestion = IngestionPipeline(
        chunker=FixedSizeChunker(chunk_size=1000, overlap=0)
    )
    pipeline.index(write(tmp_path / "long.txt", long_text))

    result = runner.invoke(cli.app, ["search", "robots"])

    assert "..." in result.output


@pytest.mark.parametrize(
    ("pages", "expected"),
    [
        ([1], "report.pdf, page 1 "),
        ([2, 3], "report.pdf, pages 2, 3 "),
    ],
)
def test_search_shows_page_numbers(pipeline, monkeypatch, pages, expected):
    hit = SearchHit(
        chunk=Chunk(
            document_id="doc-001",
            text="Robots navigate.",
            index=0,
            start=0,
            end=16,
        ),
        score=1.0,
        source="report.pdf",
        pages=pages,
    )

    monkeypatch.setattr(pipeline, "search", lambda *args, **kwargs: [hit])

    result = runner.invoke(cli.app, ["search", "robots"])

    assert expected in result.output


def test_search_without_results(pipeline):
    result = runner.invoke(cli.app, ["search", "robots"])

    assert result.exit_code == 0
    assert "No results." in result.output


def test_blank_search_query(pipeline):
    result = runner.invoke(cli.app, ["search", "   "])

    assert result.exit_code != 0


def test_docs_and_delete(pipeline, tmp_path):
    indexed = pipeline.index(
        write(tmp_path / "robots.txt", "Robots navigate.")
    )

    listing = runner.invoke(cli.app, ["docs"])

    assert indexed.document_id in listing.output
    assert "robots.txt" in listing.output

    deleted = runner.invoke(cli.app, ["delete", indexed.document_id])

    assert deleted.exit_code == 0
    assert "Deleted 1 chunks" in deleted.output
    assert "No documents indexed." in runner.invoke(
        cli.app, ["docs"]
    ).output


def test_delete_unknown_document(pipeline):
    result = runner.invoke(cli.app, ["delete", "missing"])

    assert result.exit_code == 1
    assert "No document with ID missing" in result.output
